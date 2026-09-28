"""B3: GPU headroom gates wait for a gpu-truth sample taken after they asked.

gpu-truth used to be sampled only every 30 s, so a wake right after a sleep on
the same GPU saw the previous occupant and was refused (409 "insufficient wake
headroom"). The service-manager now INCRs ``tre:gpu_truth_refresh:<node>`` and
waits for a payload whose ``refresh_seq`` answers that request.
"""

import json

import pytest

from tre_sm.allocator.slots import Slot
from tre_sm.api.v2 import ServiceManagerV2, WakeConflict
from tre_sm.gpu_truth import RedisGpuTruth
from tre_sm.state.store import StateStore

from sm_test_fakes import (
    FakeRuntime,
    FakeVllm,
    LegacyRedis,
    TickingClock,
    binding_of,
    pod,
    registry,
)

NODE_KEY = "tre:gpu_truth:node-a"
REFRESH_KEY = "tre:gpu_truth_refresh:node-a"
BUSY_MIB = 33_000  # an awake resident on GPU-0
FREE_MIB = 900  # only sleeping residents left


class AgentRedis:
    """Redis as seen by RedisGpuTruth, plus a simulated gpu-truth DaemonSet agent.

    ``physical`` is the real used memory of GPU-0 now; the stored payload only
    changes when the agent samples. ``refresh``: the agent serves refresh
    requests (``latency_s`` after the INCR), else it is an old agent whose
    payload has no ``refresh_seq``. ``resample_at`` (clock time): an old agent's
    next periodic sample.
    """

    def __init__(self, clock, *, physical, refresh=True, latency_s=0.3, publish=True, total_mib=40960):
        self.values = {}
        self.total_mib = total_mib
        self.clock = clock
        self.physical = physical
        self.refresh = refresh
        self.latency_s = latency_s
        self.pending = None
        self.served = 0
        self.seq = 0
        self.incr_calls = 0
        self.samples = []
        self.resample_at = None
        if publish:
            self.sample()

    def sample(self):
        self.seq += 1
        payload = {
            "node": "node-a",
            "timestamp": self.clock.now,
            "gpus": [
                {"uuid": "GPU-0", "used_mib": self.physical},
                {"uuid": "GPU-1", "used_mib": 500, "total_mib": 40960},
            ],
            "seq": self.seq,
        }
        if self.total_mib is not None:
            payload["gpus"][0]["total_mib"] = self.total_mib
        if self.refresh:
            payload["refresh_seq"] = self.served
        self.values[NODE_KEY] = json.dumps(payload).encode()
        self.samples.append((self.clock.now, self.physical))

    # redis-py surface used by RedisGpuTruth
    def get(self, key):
        return self.values.get(key)

    def incr(self, key):
        self.incr_calls += 1
        value = int(self.values.get(key, 0)) + 1
        self.values[key] = value
        if self.refresh:
            self.pending = (value, self.clock.now + self.latency_s)
        return value

    # the agent's loop, driven by the SM's sleeps
    def tick(self, now):
        if self.pending is not None and now >= self.pending[1]:
            self.served = self.pending[0]
            self.pending = None
            self.sample()
        if self.resample_at is not None and now >= self.resample_at:
            self.resample_at = None
            self.sample()


def _service(*, physical, refresh=True, require=True, wait_s=10.0, latency_s=0.3, publish=True):
    clock = TickingClock()
    agent = AgentRedis(clock, physical=physical, refresh=refresh, latency_s=latency_s, publish=publish)
    clock.hooks.append(agent.tick)
    sleeping = pod("pod-a", "m1", (0,), ip="10.0.0.1", state="sleeping")
    vllm = FakeVllm()
    vllm.sleeping["10.0.0.1"] = True
    store = StateStore(LegacyRedis())
    store.save([binding_of(sleeping)], expected_version=0)
    service = ServiceManagerV2(
        registry(wake_truth_wait_s=wait_s, wake_max_used_mib=8192),
        store,
        runtime_ops=FakeRuntime([sleeping]),
        vllm_ops=vllm,
        gpu_truth=RedisGpuTruth(agent),
        require_gpu_truth=require,
        sleep_clock=clock,
    )
    return service, agent, vllm, clock


def _woke(vllm):
    return ("wake_up", "10.0.0.1") in vllm.calls


# ------------------------------------------------------------ provider


def test_request_refresh_increments_the_node_counter_outside_the_payload_prefix():
    clock = TickingClock()
    agent = AgentRedis(clock, physical=FREE_MIB)
    truth = RedisGpuTruth(agent)

    assert truth.request_refresh(node="node-a") == 1
    assert truth.request_refresh(node="node-a") == 2
    assert agent.values[REFRESH_KEY] == 2
    assert not REFRESH_KEY.startswith("tre:gpu_truth:")  # UI/sampler SCAN tre:gpu_truth:*


def test_node_truth_parses_seq_and_refresh_seq():
    class R:
        def __init__(self, payload):
            self.payload = payload

        def get(self, key):
            return json.dumps(self.payload)

    truth = RedisGpuTruth(R({"gpus": [], "seq": 4, "refresh_seq": 2})).node_truth(node="n")
    assert (truth.seq, truth.refresh_seq) == (4, 2)
    old = RedisGpuTruth(R({"gpus": []})).node_truth(node="n")
    assert (old.seq, old.refresh_seq) == (None, None)
    odd = RedisGpuTruth(R({"gpus": [], "refresh_seq": True, "seq": "3"})).node_truth(node="n")
    assert (odd.seq, odd.refresh_seq) == (None, None)


def test_request_refresh_failure_returns_none():
    class Broken:
        def get(self, key):
            return None

        def incr(self, key):
            raise ConnectionError("redis down")

    assert RedisGpuTruth(Broken()).request_refresh(node="n") is None


# ------------------------------------------------------------ wake gate


def test_wake_right_after_a_sleep_waits_for_the_fresh_sample():
    # The stored sample was taken while the previous occupant was awake; it has
    # just slept (physical memory is free now) but no periodic sample since.
    service, agent, vllm, clock = _service(physical=BUSY_MIB)
    agent.physical = FREE_MIB
    start = clock.now

    service.put_binding_power("pod-a", awake=True)

    assert _woke(vllm)
    assert agent.incr_calls == 1
    assert agent.samples[-1] == (pytest.approx(start + 0.4, abs=0.25), FREE_MIB)
    assert clock.now - start < 1.0  # one refresh round trip, not a periodic interval


def test_wake_does_not_trust_a_stale_fine_sample_when_the_agent_serves_refreshes():
    # Stale sample: free. Fresh sample: an awake resident / leak -> refuse.
    service, agent, vllm, clock = _service(physical=FREE_MIB, wait_s=2.0)
    agent.physical = BUSY_MIB

    with pytest.raises(WakeConflict, match="insufficient wake headroom"):
        service.put_binding_power("pod-a", awake=True)
    assert not _woke(vllm)
    assert agent.incr_calls >= 2  # a fresh sample with a problem is asked for again


def test_wake_passes_once_a_later_fresh_sample_shows_the_memory_released():
    service, agent, vllm, clock = _service(physical=BUSY_MIB, wait_s=5.0)
    # The sleep is still releasing memory at the first fresh sample.
    clock.hooks.insert(0, lambda now: setattr(agent, "physical", FREE_MIB) if now > clock_start + 1.0 else None)
    clock_start = clock.now

    service.put_binding_power("pod-a", awake=True)

    assert _woke(vllm)
    assert agent.incr_calls >= 2


def test_wake_with_an_old_agent_falls_back_to_rereading_the_periodic_sample():
    service, agent, vllm, clock = _service(physical=BUSY_MIB, refresh=False)
    agent.physical = FREE_MIB
    agent.resample_at = clock.now + 3.0  # the old agent's next periodic sample

    service.put_binding_power("pod-a", awake=True)

    assert _woke(vllm)
    assert agent.incr_calls == 1  # the request was sent; nobody answered it


def test_wake_with_an_old_agent_refuses_after_the_wait():
    service, agent, vllm, clock = _service(physical=BUSY_MIB, refresh=False, wait_s=3.0)
    start = clock.now

    with pytest.raises(WakeConflict, match="used_mib=33000"):
        service.put_binding_power("pod-a", awake=True)
    assert clock.now - start == pytest.approx(3.0, abs=0.3)
    assert not _woke(vllm)


def test_wake_fails_closed_without_gpu_truth_even_with_refresh_requests():
    service, agent, vllm, clock = _service(physical=FREE_MIB, publish=False, latency_s=1e9, wait_s=2.0)

    with pytest.raises(WakeConflict, match="gpu truth unavailable"):
        service.put_binding_power("pod-a", awake=True)
    assert not _woke(vllm)


def test_wake_passes_without_gpu_truth_only_when_explicitly_permissive():
    service, agent, vllm, clock = _service(physical=FREE_MIB, publish=False, latency_s=1e9, require=False)

    service.put_binding_power("pod-a", awake=True)

    assert _woke(vllm)


def test_an_unanswered_refresh_falls_back_to_a_fine_stale_sample_at_the_deadline(caplog):
    # A refresh-aware agent that stopped answering (e.g. wedged nvidia-smi on the
    # refresh path) must not block wakes forever: the TTL-valid sample decides.
    service, agent, vllm, clock = _service(physical=FREE_MIB, latency_s=1e9, wait_s=2.0)
    start = clock.now

    with caplog.at_level("WARNING", logger="tre_sm.api.v2"):
        service.put_binding_power("pod-a", awake=True)

    assert _woke(vllm)
    assert clock.now - start == pytest.approx(2.0, abs=0.3)
    assert "answered refresh" in caplog.text


# ------------------------------------------------------------ cold-start gate


def test_cold_start_gate_waits_for_a_fresh_sample_after_a_pod_deletion():
    service, agent, vllm, clock = _service(physical=30_000)
    agent.physical = 100  # the pod on GPU-0 was just deleted

    service._ensure_create_headroom(Slot("node-a", (0,)), "m1")

    assert agent.incr_calls == 1
    assert agent.samples[-1][1] == 100


def test_cold_start_gate_with_an_old_agent_keeps_the_immediate_decision():
    service, agent, vllm, clock = _service(physical=30_000, refresh=False)
    start = clock.now

    with pytest.raises(ValueError, match="insufficient startup headroom"):
        service._ensure_create_headroom(Slot("node-a", (0,)), "m1")
    assert clock.now == start  # no wait: the previous behaviour


def test_cold_start_gate_fails_closed_without_gpu_truth():
    service, agent, vllm, clock = _service(physical=100, publish=False, latency_s=1e9, wait_s=1.0)

    with pytest.raises(ValueError, match="gpu truth unavailable for node node-a"):
        service._ensure_create_headroom(Slot("node-a", (0,)), "m1")


# ------------------------------------------------- B9: derived cold-start limit
# vLLM starts an engine only with gpu_memory_utilization x total free, so a
# cold start may proceed while used <= total x (1 - util) - margin. A full layout
# (three bindings per GPU, the others asleep) keeps ~4.2 GiB on a GPU.

FULL_LAYOUT_SLEEPING_MIB = 4200


def _create_service(*, physical, util=0.85, env_limit=None, total_mib=40960, **config):
    import dataclasses

    from tre_common.registry import Registry

    clock = TickingClock()
    agent = AgentRedis(clock, physical=physical, total_mib=total_mib)
    clock.hooks.append(agent.tick)
    base = registry(wake_truth_wait_s=10.0, **config)
    args = ("--max-num-seqs", "256", "--gpu-memory-utilization", str(util))
    models = [dataclasses.replace(spec, vllm_extra_args=args) for spec in base.models()]
    reg = Registry(base.topology(), models, service_manager=base.service_manager())
    service = ServiceManagerV2(
        reg,
        StateStore(LegacyRedis()),
        runtime_ops=FakeRuntime([]),
        vllm_ops=FakeVllm(),
        gpu_truth=RedisGpuTruth(agent),
        create_max_used_mib=env_limit,
        sleep_clock=clock,
    )
    return service


def test_derived_create_limit_admits_sleeping_neighbours_of_a_full_layout():
    service = _create_service(physical=FULL_LAYOUT_SLEEPING_MIB)
    # 40960 x (1 - 0.85) - 512 = 5632 >= 4200 (the old absolute 2500 refused it)
    service._ensure_create_headroom(Slot("node-a", (0,)), "m1")


def test_derived_create_limit_refuses_what_vllm_would_refuse():
    service = _create_service(physical=6000)
    with pytest.raises(ValueError, match=r"used_mib=6000 max_used_mib=5632 .*gpu_memory_utilization 0.85"):
        service._ensure_create_headroom(Slot("node-a", (0,)), "m1")


def test_derived_create_limit_follows_the_created_models_utilization():
    # util 0.95: 40960 x 0.05 - 512 = 1536 < 4200
    service = _create_service(physical=FULL_LAYOUT_SLEEPING_MIB, util=0.95)
    with pytest.raises(ValueError, match="max_used_mib=1536"):
        service._ensure_create_headroom(Slot("node-a", (0,)), "m1")


def test_derived_create_limit_uses_the_registry_margin():
    service = _create_service(physical=FULL_LAYOUT_SLEEPING_MIB, create_margin_mib=2000)
    with pytest.raises(ValueError, match="max_used_mib=4144"):
        service._ensure_create_headroom(Slot("node-a", (0,)), "m1")


def test_registry_absolute_create_limit_replaces_the_derived_one():
    service = _create_service(physical=FULL_LAYOUT_SLEEPING_MIB, create_max_used_mib=4000)
    with pytest.raises(ValueError, match=r"max_used_mib=4000 \(service_manager.create.max_used_mib\)"):
        service._ensure_create_headroom(Slot("node-a", (0,)), "m1")
    _create_service(physical=3900, create_max_used_mib=4000)._ensure_create_headroom(
        Slot("node-a", (0,)), "m1"
    )


def test_env_create_limit_overrides_the_registry():
    service = _create_service(
        physical=FULL_LAYOUT_SLEEPING_MIB, env_limit=2500, create_max_used_mib=8000
    )
    with pytest.raises(ValueError, match=r"max_used_mib=2500 \(TRE_CREATE_MAX_USED_MIB\)"):
        service._ensure_create_headroom(Slot("node-a", (0,)), "m1")


def test_derived_create_limit_fails_closed_without_the_gpu_total():
    service = _create_service(physical=100, total_mib=None)
    with pytest.raises(ValueError, match="reports no total memory: refusing cold start"):
        service._ensure_create_headroom(Slot("node-a", (0,)), "m1")
    # an absolute limit does not need the total
    _create_service(physical=100, total_mib=None, create_max_used_mib=4000)._ensure_create_headroom(
        Slot("node-a", (0,)), "m1"
    )


def test_create_limit_for_an_unknown_model_fails_closed():
    service = _create_service(physical=100)
    with pytest.raises(ValueError, match="cannot derive the startup headroom limit"):
        service._ensure_create_headroom(Slot("node-a", (0,)), "no-such-model")


class _Stop(Exception):
    pass


def test_cold_start_callers_pass_the_created_model(monkeypatch):
    from types import SimpleNamespace

    from tre_sm.allocator.slots import Binding, Migration

    service = _create_service(physical=100)
    seen = []

    def spy(slot, model):
        seen.append((slot, model))
        raise _Stop()

    monkeypatch.setattr(service, "_ensure_create_headroom", spy)
    with pytest.raises(_Stop):
        service._create_and_wake_runtime_binding("m1", Slot("node-a", (1,)))
    assert seen[-1] == (Slot("node-a", (1,)), "m1")

    monkeypatch.setattr(service, "_apply_runtime_power_action", lambda *a, **k: None)
    monkeypatch.setattr(service, "_refresh_observed", lambda *a, **k: None)
    service._runtime_ops = SimpleNamespace(
        delete_model_deployment=lambda binding: None, wait_pod_deleted=lambda serve_id: None
    )
    source = Binding("pod-a", "tp2", Slot("node-a", (0, 1)), awake=True)
    migration = Migration(
        serve_id="pod-a", from_slot=source.slot, to_slot=Slot("node-a", (2, 3))
    )
    with pytest.raises(_Stop):
        service._execute_runtime_defrag_migration(source, migration)
    assert seen[-1] == (Slot("node-a", (2, 3)), "tp2")

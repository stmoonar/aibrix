"""S1 (2026-09-30): the wake gate trusts a TTL-valid gpu-truth sample and never
waits for a new one - unless the sample predates the last LOCAL power change on
the GPU (a donor sleep commit, a wake, a start). A missing / untrusted sample is
neither a pass nor a refusal: the residents of the GPU are probed (/is_sleeping);
all asleep -> pass, any awake or unknown -> refuse."""

import json

import pytest

from tre_sm.api.v2 import ServiceManagerV2, WakeConflict
from tre_sm.gpu_truth import RedisGpuTruth
from tre_sm.ops.sleep_primitive import SleepTarget
from tre_sm.state.store import StateStore

from sm_test_fakes import FakeRuntime, FakeVllm, LegacyRedis, TickingClock, binding_of, pod, registry

NODE_KEY = "tre:gpu_truth:node-a"
BUSY_MIB = 33_000
FREE_MIB = 900


class Agent:
    """Redis as RedisGpuTruth sees it + a gpu-truth agent that answers refresh
    requests only when ``serve()`` is called (``refresh=False``: an old agent
    whose payload has no ``refresh_seq``)."""

    def __init__(self, *, physical, refresh=True, publish=True):
        self.values = {}
        self.physical = physical
        self.refresh = refresh
        self.requested = 0
        self.served = 0
        self.seq = 0
        if publish:
            self.sample()

    def sample(self):
        self.seq += 1
        payload = {
            "node": "node-a",
            "timestamp": 1.0,
            "gpus": [
                {"uuid": "GPU-0", "used_mib": self.physical, "total_mib": 40960},
                {"uuid": "GPU-1", "used_mib": 500, "total_mib": 40960},
            ],
            "seq": self.seq,
        }
        if self.refresh:
            payload["refresh_seq"] = self.served
        self.values[NODE_KEY] = json.dumps(payload).encode()

    def serve(self):
        """The agent reads the refresh counter and publishes a new sample."""
        self.served = self.requested
        self.sample()

    def get(self, key):
        return self.values.get(key)

    def incr(self, key):
        self.requested += 1
        return self.requested


def _service(*, physical, residents=(), refresh=True, publish=True, require=True):
    """pod-a (m1, GPU 0) asleep and to be woken; ``residents`` = extra pods on GPU 0
    as (name, model, is_sleeping) - is_sleeping None = unreachable."""
    agent = Agent(physical=physical, refresh=refresh, publish=publish)
    target = pod("pod-a", "m1", (0,), ip="10.0.0.1", state="sleeping")
    vllm = FakeVllm()
    vllm.sleeping["10.0.0.1"] = True
    snapshots = [target]
    for index, (name, model, sleeping) in enumerate(residents):
        ip = f"10.0.1.{index + 1}"
        gpus = (0, 1) if model == "tp2" else (0,)
        snapshots.append(pod(name, model, gpus, ip=ip, state="sleeping" if sleeping else "awake"))
        if sleeping is None:
            vllm.physical_override[ip] = None
        else:
            vllm.sleeping[ip] = sleeping
    store = StateStore(LegacyRedis())
    # The account only knows the target (the residents are what the account
    # cannot see - e.g. a leak or an unmanaged Pod).
    store.save([binding_of(target)], expected_version=0)
    service = ServiceManagerV2(
        registry(wake_max_used_mib=8192),
        store,
        runtime_ops=FakeRuntime(snapshots),
        vllm_ops=vllm,
        gpu_truth=RedisGpuTruth(agent),
        require_gpu_truth=require,
        sleep_clock=TickingClock(),
    )
    return service, agent, vllm


def _woke(vllm):
    return ("wake_up", "10.0.0.1") in vllm.calls


def _refusal(service):
    with pytest.raises(WakeConflict) as caught:
        service.put_binding_power("pod-a", awake=True)
    return caught.value


# ------------------------------------------------------------- trusted sample


def test_ttl_valid_sample_passes_without_waiting_or_probing():
    service, agent, vllm = _service(physical=FREE_MIB, residents=[("pod-b", "tp2", None)])
    clock = service._sleep_clock

    service.put_binding_power("pod-a", awake=True)

    assert _woke(vllm)
    assert clock.slept == []  # never waited for a sample
    # The unreachable resident was never probed: the trusted sample decided.
    # only the power-change refreshes of the wake itself (prepare, commit)
    assert agent.requested == 2


def test_trusted_busy_sample_refuses_at_once_and_asks_for_a_new_sample():
    service, agent, vllm = _service(physical=BUSY_MIB)

    refusal = _refusal(service)

    assert refusal.reason == "gpu_truth_used"
    assert (refusal.node, refusal.gpus) == ("node-a", (0,))
    assert "used_mib=33000" in str(refusal)
    assert service._sleep_clock.slept == []
    assert agent.requested == 1  # the next attempt sees a fresh sample
    assert not _woke(vllm)


# ------------------------------------------- sample older than a power change


def test_truth_fallback_stale_sample_residents_asleep_pass():
    # The stored sample still shows the donor awake; the donor's sleep commit just
    # happened on this SM (the agent has not answered the refresh yet).
    service, agent, vllm = _service(physical=BUSY_MIB, residents=[("donor", "tp2", True)])
    service._note_power_change("node-a", (0,))

    service.put_binding_power("pod-a", awake=True)

    assert _woke(vllm)
    assert service._sleep_clock.slept == []


def test_truth_fallback_stale_sample_awake_resident_refuses():
    service, agent, vllm = _service(physical=FREE_MIB, residents=[("other", "tp2", False)])
    service._note_power_change("node-a", (0,))

    refusal = _refusal(service)

    assert refusal.reason == "resident_awake"
    assert refusal.scope == "gpu"
    assert "tp2/node-a/0,1" in str(refusal)
    assert not _woke(vllm)


def test_truth_fallback_stale_sample_unreachable_resident_refuses():
    service, agent, vllm = _service(physical=FREE_MIB, residents=[("other", "tp2", None)])
    service._note_power_change("node-a", (0,))

    refusal = _refusal(service)

    assert refusal.reason == "resident_unknown"
    assert refusal.scope == "gpu"
    assert not _woke(vllm)


def test_a_sample_answering_the_refresh_is_trusted_again():
    service, agent, vllm = _service(physical=BUSY_MIB, residents=[("other", "tp2", None)])
    service._note_power_change("node-a", (0,))
    agent.physical = FREE_MIB
    agent.serve()  # the sample now answers the refresh sent after the change

    service.put_binding_power("pod-a", awake=True)

    assert _woke(vllm)  # trusted and free: the unreachable resident did not matter


def test_old_agent_sample_is_trusted_only_once_it_was_published_after_the_change():
    service, agent, vllm = _service(
        physical=BUSY_MIB, refresh=False, residents=[("other", "tp2", None)]
    )
    service._note_power_change("node-a", (0,))

    assert _refusal(service).reason == "resident_unknown"  # same publish: untrusted

    agent.physical = FREE_MIB
    agent.sample()  # a later periodic publish
    service.put_binding_power("pod-a", awake=True)
    assert _woke(vllm)


def test_sleep_commit_marks_the_gpus_of_the_slept_binding():
    service, agent, vllm = _service(physical=FREE_MIB)
    donor = binding_of(pod("donor", "tp2", (0, 1), ip="10.0.2.1"))

    service._record_sleep_outcomes(
        [SleepTarget(donor, "10.0.2.1")],
        [{"binding_id": donor.binding_id, "serve_id": "donor", "status": "slept"}],
        update_store=False,
    )

    assert agent.requested == 1
    assert service._untrusted_gpus("node-a", (0, 1), service._gpu_truth.node_truth(node="node-a")) == [0, 1]
    agent.serve()
    assert service._untrusted_gpus("node-a", (0, 1), service._gpu_truth.node_truth(node="node-a")) == []


# ------------------------------------------------------------ missing sample


def test_truth_fallback_missing_sample_is_sleeping_fallback_passes():
    service, agent, vllm = _service(
        physical=FREE_MIB, publish=False, residents=[("donor", "tp2", True)]
    )

    service.put_binding_power("pod-a", awake=True)

    assert _woke(vllm)


def test_truth_fallback_missing_sample_unverifiable_resident_node_scope():
    service, agent, vllm = _service(
        physical=FREE_MIB, publish=False, residents=[("other", "tp2", None)]
    )

    refusal = _refusal(service)

    assert refusal.reason == "gpu_truth_unavailable"
    assert refusal.scope == "node"
    assert "gpu truth unavailable for node node-a" in str(refusal)
    assert not _woke(vllm)


def test_truth_fallback_missing_sample_awake_resident_refuses():
    service, agent, vllm = _service(
        physical=FREE_MIB, publish=False, residents=[("other", "tp2", False)]
    )

    assert _refusal(service).reason == "resident_awake"


def test_missing_sample_passes_without_probing_only_when_explicitly_permissive():
    service, agent, vllm = _service(
        physical=FREE_MIB, publish=False, require=False, residents=[("other", "tp2", None)]
    )

    service.put_binding_power("pod-a", awake=True)

    assert _woke(vllm)


def test_structured_409_body_keeps_the_plain_detail():
    refusal = WakeConflict(
        "slot busy", reason="lease_starting", node="n", gpus=(2, 3), binding_id="m/n/2,3",
        blocking_binding_id="x/n/2,3",
    )

    body = refusal.body(retry_after_s=30.0)

    assert body["detail"] == "slot busy"
    assert body["error"] == "resident_loading"
    assert (body["reason"], body["node"], body["gpu_ids"], body["scope"]) == ("lease_starting", "n", [2, 3], "gpu")
    assert (body["binding_id"], body["blocking_binding_id"], body["retry_after_s"]) == ("m/n/2,3", "x/n/2,3", 30.0)

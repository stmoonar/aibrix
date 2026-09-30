"""Review fixes of the placement / parallel-wake branch (2026-09-30): wakes whose
commit does not complete are handed to the journal recovery (never stuck in this
process), the waking lease and the journal fence the GPU until resolved, the
recovery is bounded and never completes a replaced pod, startup / restart
placeholders are bounded, partial growth is not a silent success, reconcile and
defrag keep off wakes in flight."""

from __future__ import annotations

import dataclasses
import json
import logging
from contextlib import contextmanager

import pytest

from tre_sm.allocator.slots import Binding, Slot
from tre_sm.api.v2 import DefragUnavailable, RetryLater, ServiceManagerV2, WakeConflict, WakeFailed
from tre_sm.gpu_truth import RedisGpuTruth
from tre_sm.state.gpu_leases import GpuLeaseStore
from tre_sm.state.operations import OperationBusy
from tre_sm.state.reconcile import PodRecord, reconcile_state
from tre_sm.state.store import StateStore
from tre_sm.state.wake_journal import WakeJournal

from sm_test_fakes import FakeRedis, Result, fence, pod, registry
from test_review2_sleep import World, _desired
from test_sleep_lock_phases import StrictCoordinator


def _world(snapshots=None, desired=None, *, sm_config=None):
    snapshots = snapshots or [
        pod("pod-a", "m1", (0,), ip="10.0.0.1", state="sleeping"),
        pod("pod-b", "m1", (1,), ip="10.0.0.2", state="sleeping"),
    ]
    desired = desired or [
        _desired("m1/node-a/0", "m1", (0,), "sleeping"),
        _desired("m1/node-a/1", "m1", (1,), "sleeping"),
    ]
    sm_registry = registry()
    if sm_config:
        config = dataclasses.replace(sm_registry.service_manager(), **sm_config)
        sm_registry = type(sm_registry)(sm_registry.topology(), sm_registry.models(), service_manager=config)
    world = World(snapshots, desired, sm_registry=sm_registry)
    world.leases = GpuLeaseStore(world.redis)
    world.service._gpu_leases = world.leases
    world.journal = WakeJournal(world.redis)
    world.service._wake_journal = world.journal
    return world


def _leases(world):
    return {lease.binding_id: (lease.phase, lease.expires_at_ms) for lease in world.leases.load()}


def _events(caplog, name):
    out = []
    for record in caplog.records:
        try:
            payload = json.loads(record.getMessage())
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("event") == name:
            out.append(payload)
    return out


# ------------------------------------------------------------ P1-1 hand-over


def test_parallel_wake_commit_error_hands_over_to_recovery():
    world = _world()
    original = world.service._writer

    @contextmanager
    def broken_commit(kind, **kwargs):
        if kind.endswith("_commit"):
            raise ConnectionError("redis down")
        with original(kind, **kwargs) as operation:
            yield operation

    world.service._writer = broken_commit
    with pytest.raises(ConnectionError):
        world.service.put_binding_power("pod-a", awake=True)

    assert world.service._wakes_in_flight == set()  # never stuck in this process
    assert set(world.journal.entries()) == {"m1/node-a/0"}
    assert _leases(world)["m1/node-a/0"] == ("waking", 0)
    world.service._writer = original
    assert world.service.recover_wake_journal()["resolved"] == [{"binding_id": "m1/node-a/0", "result": "completed"}]
    assert _leases(world)["m1/node-a/0"][0] == "awake"
    world.service.put_model_target("m1", wake_replicas=2)  # the model is not frozen


def test_parallel_wake_prepare_phase_exit_error_hands_over_to_recovery():
    world = _world()
    original = world.coordinator.operation

    @contextmanager
    def failing_exit(kind, **kwargs):
        with original(kind, **kwargs) as handle:
            yield handle
        if kind == "put_binding_power":
            raise ConnectionError("fence lost on exit")

    world.coordinator.operation = failing_exit
    with pytest.raises(ConnectionError):
        world.service.put_binding_power("pod-a", awake=True)
    assert world.service._wakes_in_flight == set()
    assert set(world.journal.entries()) == {"m1/node-a/0"}


def test_parallel_wake_one_commit_error_does_not_stop_the_other_tickets(monkeypatch):
    world = _world()
    original_end = world.journal.end

    def end(binding_id):
        if binding_id == "m1/node-a/0":
            raise ConnectionError("hdel failed")
        original_end(binding_id)

    monkeypatch.setattr(world.journal, "end", end)
    world.service.put_model_target("m1", wake_replicas=2)

    assert {k: v[0] for k, v in _leases(world).items()} == {"m1/node-a/0": "awake", "m1/node-a/1": "awake"}
    assert world.service._wakes_in_flight == set()
    assert set(world.journal.entries()) == {"m1/node-a/0"}  # the recovery completes it again


def test_parallel_wake_commit_retries_once_when_the_lock_is_busy():
    world = _world()

    class OnceBusy(StrictCoordinator):
        refused = False

        @contextmanager
        def operation(self, kind, **kwargs):
            if kind.endswith("_commit") and not OnceBusy.refused:
                OnceBusy.refused = True
                raise OperationBusy("busy once")
            with super().operation(kind, **kwargs) as handle:
                yield handle

    world.coordinator = OnceBusy(world.redis)
    world.service._operation_coordinator = world.coordinator
    world.service.put_binding_power("pod-a", awake=True)
    assert _leases(world)["m1/node-a/0"][0] == "awake"


# ------------------------------------------------------------ P1-2 fence until resolved


def test_parallel_wake_journaled_binding_occupies_its_gpus_without_a_lease():
    world = _world(
        [pod("pod-a", "m1", (0,), ip="10.0.0.1", state="sleeping"),
         pod("pod-t", "tp2", (0, 1), ip="10.0.0.3", state="sleeping")],
        [_desired("m1/node-a/0", "m1", (0,), "sleeping"), _desired("tp2/node-a/0,1", "tp2", (0, 1), "sleeping")],
    )
    world.journal.begin("m1/node-a/0", {"serve_id": "pod-a", "model": "m1", "node": "node-a", "gpu_ids": [0]})
    tp2 = next(b for b in world.store.load().bindings if b.serve_id == "pod-t")

    conflict = world.service._wake_blocker(tp2, world.store.load().bindings, [])

    assert conflict.reason == "lease_waking" and conflict.blocking_binding_id == "m1/node-a/0"


def test_parallel_wake_recovery_waits_for_the_writer_lock():
    world = _world()
    world.journal.begin("m1/node-a/0", {"serve_id": "pod-a", "model": "m1", "node": "node-a", "gpu_ids": [0],
                                        "pod_uid": "uid-pod-a", "previous_power": "sleeping"})
    waits = []
    original = world.service._writer

    @contextmanager
    def spy(kind, **kwargs):
        waits.append((kind, kwargs.get("wait_s")))
        with original(kind, **kwargs) as operation:
            yield operation

    world.service._writer = spy
    world.service.recover_wake_journal()
    assert waits == [("wake_journal_recovery", world.service._sm_config.commit_wait_s)]


# ------------------------------------------------------------ P2-3 restart


def test_parallel_wake_bootstrap_rebuild_keeps_waking_leases_and_marks():
    redis = FakeRedis()
    leases = GpuLeaseStore(redis)
    awake = Binding("pod-x", "m1", Slot("node-a", (2,)), awake=True)
    waking = Binding("pod-a", "m1", Slot("node-a", (0,)), awake=False)
    clash = Binding("pod-y", "tp2", Slot("node-a", (2, 3)), awake=False)
    with fence(redis):
        leases.rebuild_awake([awake], waking_bindings=[waking, clash])
    phases = {lease.binding_id: (lease.phase, lease.expires_at_ms) for lease in leases.load()}
    assert phases == {"m1/node-a/2": ("awake", 0), "m1/node-a/0": ("waking", 0)}

    journal = WakeJournal(redis)
    journal.begin("m1/node-a/0", {"serve_id": "pod-a", "model": "m1", "node": "node-a", "gpu_ids": [0]})
    redis.values["tre:gpu_truth:node-a"] = json.dumps({"gpus": [], "seq": 3, "refresh_seq": 0})
    redis.incr = lambda key: 7
    service = ServiceManagerV2(registry(), StateStore(FakeRedis()), gpu_truth=RedisGpuTruth(redis), wake_journal=journal)
    assert service._untrusted_gpus("node-a", (0,), service._gpu_truth.node_truth(node="node-a")) == [0]


# ------------------------------------------------------------ P2-4 / P2-5 recovery


def test_parallel_wake_recovery_never_completes_a_replaced_pod():
    world = _world()
    world.journal.begin("m1/node-a/0", {"serve_id": "pod-a", "model": "m1", "node": "node-a", "gpu_ids": [0],
                                        "pod_uid": "uid-old", "previous_power": "sleeping"})
    with fence(world.redis):
        world.leases.acquire(Binding("pod-a", "m1", Slot("node-a", (0,)), awake=False), phase="waking")
    world.vllm.sleeping["10.0.0.1"] = False  # the NEW pod (loading / awake)

    result = world.service.recover_wake_journal()

    assert result["resolved"] == [{"binding_id": "m1/node-a/0", "result": "pod_replaced"}]
    assert _leases(world) == {}
    assert world.state("pod-a") == "sleeping"  # the new pod was not touched
    assert not any(call[0] in ("sleep", "wake_up") for call in world.vllm.calls)


def test_parallel_wake_recovery_gives_up_on_an_unreadable_pod(caplog):
    world = _world(sm_config={"wake_recovery_unknown_attempts": 2})
    world.journal.begin("m1/node-a/0", {"serve_id": "pod-a", "model": "m1", "node": "node-a", "gpu_ids": [0],
                                        "pod_uid": "uid-pod-a", "previous_power": "sleeping"})
    world.vllm.physical_override["10.0.0.1"] = None

    assert world.service.recover_wake_journal()["kept"][0]["result"] == "physical_state_unknown"
    with caplog.at_level(logging.ERROR, logger="tre_sm.api.v2"):
        result = world.service.recover_wake_journal()

    assert result["resolved"] == [{"binding_id": "m1/node-a/0", "result": "gave_up"}]
    assert world.journal.entries() == {}
    assert _events(caplog, "wake_recovery_gave_up")


def test_parallel_wake_recovery_gives_up_at_once_when_the_pod_is_not_ready():
    snapshots = [dataclasses.replace(pod("pod-a", "m1", (0,), ip="10.0.0.1", state="sleeping"), ready=False)]
    world = _world(snapshots, [_desired("m1/node-a/0", "m1", (0,), "sleeping")])
    world.journal.begin("m1/node-a/0", {"serve_id": "pod-a", "model": "m1", "node": "node-a", "gpu_ids": [0],
                                        "pod_uid": "uid-pod-a", "previous_power": "sleeping"})
    world.vllm.physical_override["10.0.0.1"] = None
    assert world.service.recover_wake_journal()["resolved"][0]["result"] == "gave_up"


# ------------------------------------------------------------ uncertain outcomes kept


def test_compensating_sleep_failure_keeps_the_entry_and_recovery_completes_the_wake():
    world = _world()
    world.vllm.fail_sleep_for.add("10.0.0.1")

    def late(pod_ip, *, port=None):
        world.vllm.sleeping[pod_ip] = False
        return Result(False, "timed out")

    world.vllm.wake_up = late
    with pytest.raises(WakeFailed):
        world.service.put_binding_power("pod-a", awake=True)

    assert set(world.journal.entries()) == {"m1/node-a/0"}
    assert world.desired()["m1/node-a/0"][0] == "awake"  # not rolled back: it IS awake
    assert world.service.recover_wake_journal()["resolved"][0]["result"] == "completed"
    assert world.store.load().bindings[0].awake is True


def test_parallel_wake_transport_timeout_keeps_the_fence_for_a_delayed_recheck():
    world = _world(sm_config={"wake_transport_recheck_s": 3600.0})

    def timeout(pod_ip, *, port=None):
        world.vllm.sleeping[pod_ip] = True  # still waking on the server
        raise TimeoutError("read timed out")

    world.vllm.wake_up = timeout
    with pytest.raises(WakeFailed) as caught:
        world.service.put_binding_power("pod-a", awake=True)

    assert caught.value.reason == "vllm_wake_failed"
    assert _leases(world)["m1/node-a/0"][0] == "waking"  # not released on the early "asleep"
    entry = world.journal.get("m1/node-a/0")
    assert entry["recover_after_ms"] > 0
    assert world.service.recover_wake_journal() == {"resolved": [], "kept": []}  # not yet
    world.journal.update("m1/node-a/0", recover_after_ms=0)
    world.vllm.sleeping["10.0.0.1"] = False  # it did wake after all
    assert world.service.recover_wake_journal()["resolved"][0]["result"] == "completed"


def test_parallel_wake_shutdown_executor_runs_the_wakes_one_by_one():
    world = _world()
    world.service._wake_executor.shutdown(wait=True)
    world.service.put_model_target("m1", wake_replicas=2)
    assert {k: v[0] for k, v in _leases(world).items()} == {"m1/node-a/0": "awake", "m1/node-a/1": "awake"}


# ------------------------------------------------------------ P2-7 partial growth


def _one_gpu_leaking(world):
    world.redis.values["tre:gpu_truth:node-a"] = json.dumps({"seq": 1, "refresh_seq": 0, "gpus": [
        {"uuid": "GPU-0", "used_mib": 500, "total_mib": 40960},
        {"uuid": "GPU-1", "used_mib": 30000, "total_mib": 40960},
    ]})
    world.redis.incr = lambda key: 0
    world.service._gpu_truth = RedisGpuTruth(world.redis)


def test_structured_409_partial_growth_of_an_exact_target():
    world = _world()
    _one_gpu_leaking(world)

    with pytest.raises(WakeConflict) as caught:
        world.service.put_model_target("m1", wake_replicas=2)

    assert (caught.value.reason, caught.value.error) == ("partial", "partial")
    assert caught.value.node == "node-a" and caught.value.gpus == (1,)
    assert _leases(world)["m1/node-a/0"][0] == "awake"  # what did wake stays awake


def test_partial_growth_with_at_least_reports_unfilled_and_refusals():
    world = _world()
    _one_gpu_leaking(world)

    result = world.service.put_model_target("m1", wake_replicas=2, at_least=True)

    assert result["unfilled"] == 1
    assert result["refusals"][0]["error"] == "gpu_busy" and result["refusals"][0]["gpu_ids"] == [1]


def test_avoid_gpus_keeps_the_sm_off_relay_gpus():
    world = _world()
    result = world.service.put_model_target("m1", wake_replicas=1, at_least=True, avoid_gpus=["node-a/0"])
    assert [p["serve_id"] for p in result["picked"]] == ["pod-b"]


# ------------------------------------------------------------ reconcile / defrag


def test_reconcile_leaves_a_journaled_binding_alone():
    from test_reconcile import FakeK8sClient, FakeLabelWriter, FakeProber, FakeRedis as ReconcileRedis, topology

    store = StateStore(ReconcileRedis())
    store.save([Binding("serve-a", "m1", Slot("node-a", (0,)), awake=False)], expected_version=0)
    k8s = FakeK8sClient([PodRecord(serve_id="serve-a", model="m1", node="node-a", cuda_visible_devices="0",
                                   state="sleeping", pod_ip="10.0.0.1", routable=False)])
    writer = FakeLabelWriter()

    result = reconcile_state(topology(), store, k8s, prober=FakeProber({"serve-a": False}), label_writer=writer,
                             frozen_serve_ids={"serve-a"})

    assert writer.calls == []  # not opened to traffic mid-wake
    assert result.bindings[0].awake is False


def test_defrag_refuses_while_a_wake_is_journaled():
    world = _world()
    world.journal.begin("m1/node-a/0", {"serve_id": "pod-a", "model": "m1", "node": "node-a", "gpu_ids": [0]})
    with pytest.raises(DefragUnavailable, match="wake_in_progress"):
        world.service.defrag(tp_size=2, force=True)


# ------------------------------------------------------------ P1-3 / P2-8 placeholders


def _placeholder_world(**pod_overrides):
    snapshot = dataclasses.replace(
        pod("tp2-new", "tp2", (0, 1), ip="10.0.0.9", state="hidden"), ready=False, **pod_overrides
    )
    world = _world([snapshot], [_desired("tp2/node-a/0,1", "tp2", (0, 1), "sleeping")],
                   sm_config={"startup_placeholder_max_s": 900.0})
    with fence(world.redis):
        world.leases.acquire(Binding("tp2-new", "tp2", Slot("node-a", (0, 1)), awake=False), phase="starting")
    world.vllm.physical_override["10.0.0.9"] = None  # not listening (crash-looping / loading)
    return world


def test_startup_placeholder_released_past_the_bound_when_not_ready(caplog):
    world = _placeholder_world()
    assert world.service.reap_stale_startup_placeholders(now=0.0) == []
    assert world.service.reap_stale_startup_placeholders(now=899.0) == []
    with caplog.at_level(logging.ERROR, logger="tre_sm.api.v2"):
        assert world.service.reap_stale_startup_placeholders(now=901.0) == ["tp2/node-a/0,1"]
    assert _leases(world) == {}
    assert _events(caplog, "startup_placeholder_released")


def test_startup_placeholder_released_early_when_the_pod_crashloops():
    world = _placeholder_world(restart_count=1)
    assert world.service.reap_stale_startup_placeholders(now=0.0) == []
    world.runtime.snapshots["tp2-new"] = dataclasses.replace(world.runtime.snapshots["tp2-new"], restart_count=2)
    assert world.service.reap_stale_startup_placeholders(now=10.0) == ["tp2/node-a/0,1"]


def test_loading_placeholder_kept_while_the_pod_is_ready_or_awake():
    world = _placeholder_world()
    world.runtime.snapshots["tp2-new"] = dataclasses.replace(world.runtime.snapshots["tp2-new"], ready=True)
    world.service.reap_stale_startup_placeholders(now=0.0)
    assert world.service.reap_stale_startup_placeholders(now=10_000.0) == []
    world.runtime.snapshots["tp2-new"] = dataclasses.replace(world.runtime.snapshots["tp2-new"], ready=False)
    world.vllm.physical_override["10.0.0.9"] = False  # reads awake: the convergence's job
    assert world.service.reap_stale_startup_placeholders(now=10_000.0) == []
    assert "tp2/node-a/0,1" in _leases(world)


# ------------------------------------------------------------ P1-4 container restart


def _restart_world(desired_power, *, actuation="active"):
    world = _world([pod("pod-a", "m1", (0,), ip="10.0.0.1", state="sleeping")],
                   [_desired("m1/node-a/0", "m1", (0,), desired_power)])
    world.service._safety_gate.actuation = actuation
    assert world.service.guard_container_restarts() == {"placed": [], "converged": []}  # baseline
    world.runtime.snapshots["pod-a"] = dataclasses.replace(world.runtime.snapshots["pod-a"], restart_count=1)
    world.vllm.sleeping["10.0.0.1"] = False  # it came back AWAKE
    return world


def test_startup_placeholder_for_a_container_restart_then_sleep_when_desired_asleep(caplog):
    world = _restart_world("sleeping")
    with caplog.at_level(logging.WARNING, logger="tre_sm.api.v2"):
        result = world.service.guard_container_restarts()

    assert result["placed"] == ["m1/node-a/0"] and result["converged"] == ["m1/node-a/0"]
    assert _events(caplog, "container_restart_placeholder")
    assert world.vllm.sleeping["10.0.0.1"] is True
    assert _leases(world) == {}


def test_startup_placeholder_for_a_container_restart_becomes_awake_when_desired_awake():
    world = _restart_world("awake")
    world.service.guard_container_restarts()
    assert _leases(world)["m1/node-a/0"][0] == "awake"
    assert world.state("pod-a") == "awake"


def test_startup_placeholder_for_a_container_restart_observe_only_holds_the_gpus():
    world = _restart_world("sleeping", actuation="observe")
    result = world.service.guard_container_restarts()
    assert result == {"placed": ["m1/node-a/0"], "converged": []}
    assert _leases(world)["m1/node-a/0"][0] == "starting"
    assert world.vllm.sleeping["10.0.0.1"] is False  # nothing slept in observe


# ------------------------------------------------------------ P3-9 trim low-water


def test_operation_journal_trims_to_a_low_water_mark():
    from tre_common import rediskeys
    from tre_sm.state.operations import OperationCoordinator

    from test_operations import ScriptRedis

    class Redis(ScriptRedis):
        def hdel(self, key, *fields):
            for field in fields:
                self.hashes.get(key, {}).pop(field, None)

        def hlen(self, key):
            return len(self.hashes.get(key, {}))

    redis = Redis()
    coordinator = OperationCoordinator(redis, owner="sm", max_records=100)
    coordinator.TRIM_EVERY = 10**9  # trim by hand below
    for index in range(120):
        with coordinator.operation(f"op-{index}"):
            pass
    assert coordinator.trim() == 30
    assert len(redis.hashes[rediskeys.SM_OPERATIONS_KEY]) == 90
    assert coordinator.trim() == 0  # below the high-water mark: no full read


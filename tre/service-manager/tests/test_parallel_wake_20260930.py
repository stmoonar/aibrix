"""S6 (2026-09-30): a wake runs in three phases like a sleep - account + waking
lease under the writer lock, /wake_up + /is_sleeping WITHOUT it (concurrently for
the wakes of one request), commit under the lock again. Wakes on different GPUs
overlap; the waking GPU lease keeps every other writer off the same GPU; a wake
journal lets a restarted service-manager finish or roll back what a dead one
left between the phases."""

import threading
import time

import pytest

from tre_sm.api.v2 import RetryLater, ServiceManagerV2, WakeConflict
from tre_sm.state.gpu_leases import GpuLeaseStore
from tre_sm.state.operations import OperationBusy
from tre_sm.state.wake_journal import WakeJournal

from sm_test_fakes import Result, pod
from test_review2_sleep import World, _desired
from test_sleep_lock_phases import StrictCoordinator

WAKE_S = 0.3


def _world(snapshots, desired, *, refuse=()):
    world = World(snapshots, desired)
    world.coordinator = StrictCoordinator(world.redis, refuse=refuse)
    world.service._operation_coordinator = world.coordinator
    world.leases = GpuLeaseStore(world.redis)
    world.service._gpu_leases = world.leases
    world.journal = WakeJournal(world.redis)
    world.service._wake_journal = world.journal
    return world


def _sleeping_pair():
    """m1 on GPU 0 and GPU 1, both asleep and desired asleep."""
    return _world(
        [
            pod("pod-a", "m1", (0,), ip="10.0.0.1", state="sleeping"),
            pod("pod-b", "m1", (1,), ip="10.0.0.2", state="sleeping"),
        ],
        [_desired("m1/node-a/0", "m1", (0,), "sleeping"), _desired("m1/node-a/1", "m1", (1,), "sleeping")],
    )


def _slow_wakes(world, *, during=None):
    """/wake_up takes WAKE_S; records concurrency and whether the writer lock was
    held meanwhile; ``during(pod_ip)`` runs inside the call."""
    original = world.vllm.wake_up
    stats = {"active": 0, "max": 0, "lock_held": []}
    lock = threading.Lock()

    def wake_up(pod_ip, *, port=None):
        with lock:
            stats["active"] += 1
            stats["max"] = max(stats["max"], stats["active"])
        stats["lock_held"].append(world.coordinator.active is not None)
        try:
            if during is not None:
                during(pod_ip)
            time.sleep(WAKE_S)
            return original(pod_ip, port=port)
        finally:
            with lock:
                stats["active"] -= 1

    world.vllm.wake_up = wake_up
    return stats


def _leases(world):
    return {lease.binding_id: lease.phase for lease in world.leases.load()}


# ------------------------------------------------------------ concurrency


def test_two_wakes_of_one_request_run_concurrently_and_without_the_writer_lock():
    world = _sleeping_pair()
    stats = _slow_wakes(world)

    started = time.monotonic()
    result = world.service.put_model_target("m1", wake_replicas=2)
    elapsed = time.monotonic() - started

    assert sorted(a["serve_id"] for a in result["actions"]) == ["pod-a", "pod-b"]
    assert stats["max"] == 2
    assert elapsed < 1.6 * WAKE_S  # ~ one wake, not two back to back
    assert stats["lock_held"] == [False, False]
    assert world.coordinator.kinds == ["put_model_target", "put_model_target_commit"]
    assert _leases(world) == {"m1/node-a/0": "awake", "m1/node-a/1": "awake"}
    assert {b.serve_id: b.awake for b in world.store.load().bindings} == {"pod-a": True, "pod-b": True}
    assert world.desired()["m1/node-a/0"][0] == world.desired()["m1/node-a/1"][0] == "awake"
    assert world.journal.entries() == {}
    assert {p["serve_id"]: tuple(p["gpu_ids"]) for p in result["picked"]} == {"pod-a": (0,), "pod-b": (1,)}


def test_a_wake_on_another_gpu_proceeds_while_the_first_one_is_running():
    world = _sleeping_pair()
    seen = {}

    def during(pod_ip):
        if pod_ip == "10.0.0.1" and "other" not in seen:
            seen["other"] = world.service.put_binding_power("pod-b", awake=True)

    _slow_wakes(world, during=during)

    world.service.put_binding_power("pod-a", awake=True)

    assert seen["other"]["awake"] is True  # got the writer lock while pod-a was waking
    assert _leases(world) == {"m1/node-a/0": "awake", "m1/node-a/1": "awake"}


def test_the_waking_lease_keeps_other_writers_off_the_same_gpu():
    world = _world(
        [
            pod("pod-a", "m1", (0,), ip="10.0.0.1", state="sleeping"),
            pod("pod-t", "tp2", (0, 1), ip="10.0.0.3", state="sleeping"),
        ],
        [_desired("m1/node-a/0", "m1", (0,), "sleeping"), _desired("tp2/node-a/0,1", "tp2", (0, 1), "sleeping")],
    )
    refusals = {}

    def during(pod_ip):
        if pod_ip != "10.0.0.1" or refusals:
            return
        for name in ("pod-t", "pod-a"):
            try:
                world.service.put_binding_power(name, awake=True)
            except WakeConflict as exc:
                refusals[name] = exc.reason
        try:
            world.service.put_binding_power("pod-a", awake=False)
        except RetryLater:
            refusals["sleep"] = "retry_later"
        try:
            world.service.put_model_target("m1", wake_replicas=0)
        except RetryLater:
            refusals["target"] = "retry_later"

    _slow_wakes(world, during=during)

    world.service.put_binding_power("pod-a", awake=True)

    assert refusals == {
        "pod-t": "lease_waking",
        "pod-a": "wake_in_progress",
        "sleep": "retry_later",
        "target": "retry_later",
    }
    assert _leases(world) == {"m1/node-a/0": "awake"}
    assert world.vllm.sleeping["10.0.0.3"] is True


# ------------------------------------------------------------ failures


def test_a_failed_wake_outside_the_lock_is_rolled_back_in_the_commit_phase():
    world = _sleeping_pair()
    world.vllm.wake_up = lambda pod_ip, *, port=None: Result(False, "cuda oom")

    with pytest.raises(ValueError, match="cuda oom"):
        world.service.put_binding_power("pod-a", awake=True)

    assert world.coordinator.kinds == ["put_binding_power", "put_binding_power_commit"]
    assert _leases(world) == {}  # asleep: waking lease released
    assert world.desired()["m1/node-a/0"][0] == "sleeping"  # intent rolled back
    assert world.journal.entries() == {}
    assert world.store.load().bindings[0].awake is False


def test_one_failed_wake_of_a_request_keeps_the_other_awake():
    world = _sleeping_pair()
    original = world.vllm.wake_up

    def wake_up(pod_ip, *, port=None):
        time.sleep(0.05)
        if pod_ip == "10.0.0.2":
            return Result(False, "cuda oom")
        return original(pod_ip, port=port)

    world.vllm.wake_up = wake_up

    with pytest.raises(ValueError, match="cuda oom"):
        world.service.put_model_target("m1", wake_replicas=2)

    assert world.desired()["m1/node-a/0"][0] == "awake"
    assert world.desired()["m1/node-a/1"][0] == "sleeping"
    assert _leases(world) == {"m1/node-a/0": "awake"}
    assert {b.serve_id: b.awake for b in world.store.load().bindings} == {"pod-a": True, "pod-b": False}


# ------------------------------------------------------------ crash recovery


def _crash_after_prepare(world, serve_id):
    """Phase 1 of a binding wake, then the SM 'dies': a new service-manager on the
    same Redis (journal, leases, desired state) knows nothing about it."""
    with world.coordinator.operation("put_binding_power"):
        snapshot = world.store.load()
        binding = next(b for b in snapshot.bindings if b.serve_id == serve_id)
        world.service._begin_binding_wake(binding, snapshot.bindings)
    fresh = ServiceManagerV2(
        world.service._registry,
        world.store,
        runtime_ops=world.runtime,
        vllm_ops=world.vllm,
        operation_coordinator=world.coordinator,
        fleet_store=world.fleet,
        gpu_leases=world.leases,
        wake_journal=WakeJournal(world.redis),
    )
    return binding, fresh


def test_recovery_completes_a_wake_that_happened_before_the_crash():
    world = _sleeping_pair()
    binding, fresh = _crash_after_prepare(world, "pod-a")
    world.vllm.sleeping["10.0.0.1"] = False  # /wake_up went through

    result = fresh.recover_wake_journal()

    assert result["resolved"] == [{"binding_id": "m1/node-a/0", "result": "completed"}]
    assert _leases(world) == {"m1/node-a/0": "awake"}
    assert world.state("pod-a") == "awake"
    assert world.store.load().bindings[0].awake is True
    assert world.desired()["m1/node-a/0"][0] == "awake"
    assert world.journal.entries() == {}


def test_recovery_rolls_back_a_wake_that_never_happened():
    world = _sleeping_pair()
    binding, fresh = _crash_after_prepare(world, "pod-a")
    assert world.desired()["m1/node-a/0"][0] == "awake"  # the intent of phase 1

    result = fresh.recover_wake_journal()

    assert result["resolved"] == [{"binding_id": "m1/node-a/0", "result": "rolled_back"}]
    assert _leases(world) == {}
    assert world.desired()["m1/node-a/0"][0] == "sleeping"  # restored from the journal
    assert world.journal.entries() == {}


def test_recovery_keeps_an_entry_whose_pod_cannot_be_read():
    world = _sleeping_pair()
    binding, fresh = _crash_after_prepare(world, "pod-a")
    world.vllm.physical_override["10.0.0.1"] = None

    result = fresh.recover_wake_journal()

    assert result["kept"] == [{"binding_id": "m1/node-a/0", "result": "physical_state_unknown"}]
    assert world.journal.get("m1/node-a/0")["recovery_attempts"] == 1
    # the model stays fenced until the entry is resolved
    with pytest.raises(RetryLater):
        fresh.put_model_target("m1", wake_replicas=2)


def test_recovery_never_touches_a_wake_this_process_is_running():
    world = _sleeping_pair()
    seen = {}
    _slow_wakes(world, during=lambda ip: seen.setdefault("recovery", world.service.recover_wake_journal()))

    world.service.put_binding_power("pod-a", awake=True)

    assert seen["recovery"] == {"resolved": [], "kept": []}
    assert _leases(world) == {"m1/node-a/0": "awake"}


def test_a_commit_without_the_writer_lock_is_left_to_the_recovery():
    world = _world(
        [pod("pod-a", "m1", (0,), ip="10.0.0.1", state="sleeping")],
        [_desired("m1/node-a/0", "m1", (0,), "sleeping")],
        refuse=("put_binding_power_commit",),
    )

    with pytest.raises(OperationBusy):
        world.service.put_binding_power("pod-a", awake=True)

    assert world.vllm.sleeping["10.0.0.1"] is False  # it woke
    assert set(world.journal.entries()) == {"m1/node-a/0"}
    assert _leases(world) == {"m1/node-a/0": "waking"}
    world.coordinator.refuse.clear()
    assert world.service.recover_wake_journal()["resolved"] == [
        {"binding_id": "m1/node-a/0", "result": "completed"}
    ]
    assert _leases(world) == {"m1/node-a/0": "awake"}

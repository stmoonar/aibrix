"""Wakes of one request run concurrently (S6, 2026-09-30) - since 2026-10-02
under the writer lock from their checks to their commit (whole-lock): no other
writer runs while /wake_up is in flight, and a wake journal lets a restarted
service-manager finish or roll back what a dead one left mid-wake."""

import threading
import time

import pytest

from tre_sm.api.v2 import ServiceManagerV2, WakeConflict
from tre_sm.state.gpu_leases import GpuLeaseStore
from tre_sm.state.operations import OperationBusy
from tre_sm.state.wake_journal import WakeJournal

from sm_test_fakes import Result, StrictCoordinator, pod
from test_review2_sleep import World, _desired

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


def test_parallel_wake_two_wakes_of_one_request_run_concurrently_under_one_lock_hold():
    world = _sleeping_pair()
    stats = _slow_wakes(world)

    started = time.monotonic()
    result = world.service.put_model_target("m1", wake_replicas=2)
    elapsed = time.monotonic() - started

    assert sorted(a["serve_id"] for a in result["actions"]) == ["pod-a", "pod-b"]
    assert stats["max"] == 2
    assert elapsed < 1.6 * WAKE_S  # ~ one wake, not two back to back
    assert stats["lock_held"] == [True, True]  # whole-lock
    assert world.coordinator.kinds == ["put_model_target"]  # ONE lock hold
    assert _leases(world) == {"m1/node-a/0": "awake", "m1/node-a/1": "awake"}
    assert {b.serve_id: b.awake for b in world.store.load().bindings} == {"pod-a": True, "pod-b": True}
    assert world.desired()["m1/node-a/0"][0] == world.desired()["m1/node-a/1"][0] == "awake"
    assert world.journal.entries() == {}
    assert {p["serve_id"]: tuple(p["gpu_ids"]) for p in result["picked"]} == {"pod-a": (0,), "pod-b": (1,)}


def test_parallel_wake_no_other_writer_runs_while_a_wake_is_in_flight():
    world = _sleeping_pair()
    seen = {}

    def during(pod_ip):
        if pod_ip == "10.0.0.1" and "other" not in seen:
            try:
                world.service.put_binding_power("pod-b", awake=True)
                seen["other"] = "ran"
            except OperationBusy:
                seen["other"] = "busy"  # a real coordinator queues it instead

    _slow_wakes(world, during=during)

    world.service.put_binding_power("pod-a", awake=True)

    assert seen["other"] == "busy"
    assert _leases(world) == {"m1/node-a/0": "awake"}


# ------------------------------------------------------------ failures


def test_parallel_wake_a_failed_wake_is_rolled_back_in_the_same_lock_hold():
    world = _sleeping_pair()
    world.vllm.wake_up = lambda pod_ip, *, port=None: Result(False, "cuda oom")

    with pytest.raises(ValueError, match="cuda oom"):
        world.service.put_binding_power("pod-a", awake=True)

    assert world.coordinator.kinds == ["put_binding_power"]
    assert _leases(world) == {}  # asleep: lease released
    assert world.desired()["m1/node-a/0"][0] == "sleeping"  # intent rolled back
    assert world.journal.entries() == {}
    assert world.store.load().bindings[0].awake is False


def test_parallel_wake_one_failed_wake_of_a_request_keeps_the_other_awake():
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
    """The prepare of a binding wake (journal + lease + desired intent), then the
    SM 'dies' before /wake_up returns: a new service-manager on the same Redis
    (journal, leases, desired state) knows nothing about it."""
    with world.coordinator.operation("put_binding_power"):
        snapshot = world.store.load()
        binding = next(b for b in snapshot.bindings if b.serve_id == serve_id)
        world.service._prepare_wake(
            binding, snapshot.bindings, previous_desired=world.service._desired_power_of(binding.binding_id)
        )
        world.service._update_desired(
            {binding.binding_id: {"power": "awake"}}, updated_by="t", reason="t"
        )
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


def test_parallel_wake_a_crash_mid_wake_leaves_the_gpus_taken_and_the_journal():
    world = _sleeping_pair()
    _crash_after_prepare(world, "pod-a")

    assert _leases(world) == {"m1/node-a/0": "awake"}  # the GPUs stay fenced
    assert set(world.journal.entries()) == {"m1/node-a/0"}


def test_parallel_wake_recovery_completes_a_wake_that_happened_before_the_crash():
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


def test_parallel_wake_recovery_rolls_back_a_wake_that_never_happened():
    world = _sleeping_pair()
    binding, fresh = _crash_after_prepare(world, "pod-a")
    assert world.desired()["m1/node-a/0"][0] == "awake"  # the intent of the prepare

    result = fresh.recover_wake_journal()

    assert result["resolved"] == [{"binding_id": "m1/node-a/0", "result": "rolled_back"}]
    assert _leases(world) == {}
    assert world.desired()["m1/node-a/0"][0] == "sleeping"  # restored from the journal
    assert world.journal.entries() == {}


def test_parallel_wake_recovery_keeps_an_entry_whose_pod_cannot_be_read():
    world = _sleeping_pair()
    binding, fresh = _crash_after_prepare(world, "pod-a")
    world.vllm.physical_override["10.0.0.1"] = None

    result = fresh.recover_wake_journal()

    assert result["kept"] == [{"binding_id": "m1/node-a/0", "result": "physical_state_unknown"}]
    assert world.journal.get("m1/node-a/0")["recovery_attempts"] == 1
    # its GPUs stay fenced until the entry is resolved: only pod-b can wake
    with pytest.raises(WakeConflict) as info:
        fresh.put_model_target("m1", wake_replicas=2)
    assert info.value.reason == "partial"
    assert world.vllm.sleeping["10.0.0.2"] is False and ("wake_up", "10.0.0.1") not in world.vllm.calls


def test_parallel_wake_recovery_cannot_run_while_a_wake_holds_the_lock():
    world = _sleeping_pair()
    seen = {}

    def during(_ip):
        try:
            seen["recovery"] = world.service.recover_wake_journal()
        except OperationBusy:
            seen["recovery"] = "busy"

    _slow_wakes(world, during=during)

    world.service.put_binding_power("pod-a", awake=True)

    assert seen["recovery"] == "busy"
    assert _leases(world) == {"m1/node-a/0": "awake"}
    assert world.journal.entries() == {}

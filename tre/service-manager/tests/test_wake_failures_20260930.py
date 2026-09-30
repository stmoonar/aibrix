"""S4 + S3 (2026-09-30): a failed wake whose engine woke anyway is put back to
sleep (compensating sleep); every wake refusal / failure is a structured 409;
the acceptance-test fault hooks are off unless explicitly enabled; wakes leave
their details in the operation record and structured log events."""

import dataclasses
import json
import logging

import pytest
from fastapi.testclient import TestClient

from tre_common import rediskeys
from tre_sm.api.v2 import WakeConflict, WakeFailed, create_app
from tre_sm.state.gpu_leases import GpuLeaseStore
from tre_sm.state.operations import OperationCoordinator
from tre_sm.state.wake_journal import WakeJournal

from sm_test_fakes import FakeCoordinator, Result, pod, registry
from test_operations import ScriptRedis
from test_review2_sleep import World, _desired


def _world(*, test_hooks=False, awake_neighbour=False):
    """pod-b (m1, GPU 1) asleep and to be woken; pod-a (m1, GPU 0) awake; with
    ``awake_neighbour`` a tp2 binding awake on GPUs 2,3 and a sleeping m1 on 2."""
    snapshots = [
        pod("pod-a", "m1", (0,), ip="10.0.0.1"),
        pod("pod-b", "m1", (1,), ip="10.0.0.2", state="sleeping"),
    ]
    desired = [_desired("m1/node-a/0", "m1", (0,), "awake"), _desired("m1/node-a/1", "m1", (1,), "sleeping")]
    if awake_neighbour:
        snapshots += [
            pod("pod-t", "tp2", (2, 3), ip="10.0.0.3"),
            pod("pod-c", "m1", (2,), ip="10.0.0.4", state="sleeping"),
        ]
        desired += [
            _desired("tp2/node-a/2,3", "tp2", (2, 3), "awake"),
            _desired("m1/node-a/2", "m1", (2,), "sleeping"),
        ]
    sm_registry = registry()
    if test_hooks:
        config = dataclasses.replace(sm_registry.service_manager(), test_hooks=True)
        sm_registry = type(sm_registry)(sm_registry.topology(), sm_registry.models(), service_manager=config)
    world = World(snapshots, desired, sm_registry=sm_registry)
    world.leases = GpuLeaseStore(world.redis)
    world.service._gpu_leases = world.leases
    with world.coordinator.operation("seed"):
        for snapshot in snapshots:
            if snapshot.annotations["tre.aibrix.io/state"] == "awake":
                binding = next(b for b in world.store.load().bindings if b.serve_id == snapshot.name)
                world.leases.acquire(binding, phase="awake")
    world.journal = WakeJournal(world.redis)
    world.service._wake_journal = world.journal
    world.service._fault_redis = world.redis
    return world


def _leases(world):
    return {lease.binding_id: lease.phase for lease in world.leases.load()}


def _events(caplog, name):
    found = []
    for record in caplog.records:
        try:
            payload = json.loads(record.getMessage())
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("event") == name:
            found.append(payload)
    return found


# ------------------------------------------------------------ S4 compensating sleep


def test_compensating_sleep_after_a_wake_that_failed_late(caplog):
    world = _world()

    def late_failure(pod_ip, *, port=None):
        world.vllm.sleeping[pod_ip] = False  # the engine did wake
        return Result(False, "timed out")

    world.vllm.wake_up = late_failure

    with caplog.at_level(logging.INFO, logger="tre_sm.api.v2"):
        with pytest.raises(WakeFailed) as caught:
            world.service.put_binding_power("pod-b", awake=True)

    assert world.vllm.sleeping["10.0.0.2"] is True  # put back to sleep
    assert ("sleep", "10.0.0.2", "wait", True) in world.vllm.calls or any(
        call[0] == "sleep" and call[1] == "10.0.0.2" for call in world.vllm.calls
    )
    assert "m1/node-a/1" not in _leases(world)
    assert world.desired()["m1/node-a/1"][0] == "sleeping"
    assert caught.value.physically_awake is True
    assert caught.value.compensating_sleep == {"done": True, "result": "slept"}
    assert world.journal.stats()["wake_compensating_sleep_total"] == 1
    (failed,) = _events(caplog, "wake_failed")
    assert failed["error_code"] == "wake_failed"
    assert failed["compensating_sleep"] == {"done": True, "result": "slept"}
    assert failed["physically_awake"] is False  # asleep again after the sleep


def test_compensating_sleep_when_the_wake_cannot_be_recorded():
    world = _world()
    original = world.runtime.write_binding_annotations

    def refuse_awake(binding, *, state):
        if state == "awake":
            raise RuntimeError("apiserver refused the patch")
        return original(binding, state=state)

    world.runtime.write_binding_annotations = refuse_awake

    with pytest.raises(WakeFailed, match="recording the wake of pod-b failed") as caught:
        world.service.put_binding_power("pod-b", awake=True)

    assert caught.value.reason == "not_recorded"
    assert world.vllm.sleeping["10.0.0.2"] is True
    assert "m1/node-a/1" not in _leases(world)


def test_compensating_sleep_is_not_attempted_when_the_state_is_unknown_entry_kept(caplog):
    world = _world()

    def unknown(pod_ip, *, port=None):
        world.vllm.physical_override[pod_ip] = None
        return Result(False, "connection reset")

    world.vllm.wake_up = unknown

    with caplog.at_level(logging.WARNING, logger="tre_sm.api.v2"):
        with pytest.raises(WakeFailed):
            world.service.put_binding_power("pod-b", awake=True)

    assert not any(call[0] == "sleep" for call in world.vllm.calls)
    # review P1-2 / P2: the waking lease (no TTL) and the journal entry stay for the
    # recovery; the desired intent is not rolled back before the state is known.
    assert _leases(world)["m1/node-a/1"] == "waking"
    assert _events(caplog, "wake_failed_state_unknown")
    assert set(world.journal.entries()) == {"m1/node-a/1"}
    assert world.desired()["m1/node-a/1"][0] == "awake"
    world.vllm.physical_override.pop("10.0.0.2")
    world.vllm.sleeping["10.0.0.2"] = True
    assert world.service.recover_wake_journal()["resolved"] == [{"binding_id": "m1/node-a/1", "result": "rolled_back"}]
    assert world.desired()["m1/node-a/1"][0] == "sleeping"


# ------------------------------------------------------------ S3 structured 409


def test_structured_409_for_an_occupied_gpu():
    world = _world(awake_neighbour=True)
    client = TestClient(create_app(world.service))

    answer = client.put("/v2/bindings/pod-c/power", json={"awake": True})

    assert answer.status_code == 409
    body = answer.json()
    assert body["error"] == "gpu_busy"
    assert body["reason"] == "slot_occupied"
    assert (body["binding_id"], body["node"], body["gpu_ids"]) == ("m1/node-a/2", "node-a", [2])
    assert body["blocking_binding_id"] == "tp2/node-a/2,3"
    assert body["retry_after_s"] == 30.0
    assert "slot already has awake binding" in body["detail"]
    assert world.desired()["m1/node-a/2"][0] == "sleeping"  # no side effect


def test_structured_409_for_a_failed_wake():
    world = _world()
    world.vllm.wake_up = lambda pod_ip, *, port=None: Result(False, "cuda oom")
    client = TestClient(create_app(world.service))

    answer = client.put("/v2/bindings/pod-b/power", json={"awake": True})

    assert answer.status_code == 409
    body = answer.json()
    assert (body["error"], body["reason"]) == ("wake_failed", "vllm_wake_failed")
    assert body["physically_awake"] is False and body["compensating_sleep"] is None


def test_structured_409_for_a_busy_writer_lock():
    world = _world()
    client = TestClient(create_app(world.service))
    with world.coordinator.operation("someone-else"):
        answer = client.put("/v2/bindings/pod-b/power", json={"awake": True})

    assert answer.status_code == 409
    assert answer.json()["error"] == "writer_busy"


# ------------------------------------------------------------ fault hooks


def test_fault_hooks_are_off_by_default_and_never_read():
    world = _world()
    world.redis.values[rediskeys.sm_fault_key("refuse_wake", "node-a", 1)] = "1"
    reads = []
    original = world.redis.get
    world.redis.get = lambda key: (reads.append(key), original(key))[1]

    world.service.put_binding_power("pod-b", awake=True)

    assert world.vllm.sleeping["10.0.0.2"] is False
    assert not [key for key in reads if key.startswith(rediskeys.SM_FAULT_KEY_PREFIX)]


def test_fault_hook_refuse_wake_gives_a_structured_409_gpu_busy():
    world = _world(test_hooks=True)
    world.redis.values[rediskeys.sm_fault_key("refuse_wake", "node-a", 1)] = "1"
    client = TestClient(create_app(world.service))

    answer = client.put("/v2/bindings/pod-b/power", json={"awake": True})

    assert answer.status_code == 409
    assert (answer.json()["error"], answer.json()["reason"]) == ("gpu_busy", "fault_injected")
    assert not any(call[0] == "wake_up" for call in world.vllm.calls)


def test_fault_hook_fail_wake_exercises_the_compensating_sleep():
    world = _world(test_hooks=True)
    world.redis.values[rediskeys.sm_fault_key("fail_wake", "node-a", 1)] = "1"

    with pytest.raises(WakeFailed) as caught:
        world.service.put_binding_power("pod-b", awake=True)

    assert caught.value.reason == "fault_injected"
    assert caught.value.compensating_sleep == {"done": True, "result": "slept"}
    assert world.vllm.sleeping["10.0.0.2"] is True


# ------------------------------------------------------------ observability


def test_wake_details_land_in_the_operation_records(caplog):
    world = _world()
    handles = []
    coordinator = FakeCoordinator(world.redis)
    original = coordinator.operation

    from contextlib import contextmanager

    @contextmanager
    def recording(kind, **kwargs):
        with original(kind, **kwargs) as handle:
            handles.append((kind, handle))
            yield handle

    coordinator.operation = recording
    world.service._operation_coordinator = coordinator

    with caplog.at_level(logging.INFO, logger="tre_sm.api.v2"):
        world.service.put_binding_power("pod-b", awake=True)

    kinds = [kind for kind, _ in handles]
    assert kinds == ["put_binding_power", "put_binding_power_commit"]
    prepare, commit = (handle.notes for _, handle in handles)
    assert prepare["binding_id"] == "m1/node-a/1" and prepare["gpu_ids"] == [1]
    assert set(prepare["phases_ms"]) == {"reserve"}
    assert set(commit["phases_ms"]) == {"reserve", "wake_up", "commit"}
    assert commit["truth_source"] == "none" and commit["wake_attempts"] == 1
    assert "error_code" not in commit
    (done,) = _events(caplog, "wake_done")
    assert done["binding_id"] == "m1/node-a/1" and set(done["phases_ms"]) == {"reserve", "wake_up", "commit"}
    assert done["parallel_inflight"] == 1
    assert world.service.wake_state()["stats"]["wake_done_total"] == 1


def test_operation_notes_survive_the_finish_and_the_journal_is_trimmed():
    class Redis(ScriptRedis):
        def hdel(self, key, *fields):
            for field in fields:
                self.hashes.get(key, {}).pop(field, None)

        def hlen(self, key):
            return len(self.hashes.get(key, {}))

    redis = Redis()
    coordinator = OperationCoordinator(redis, owner="sm", max_records=100)
    coordinator.TRIM_EVERY = 1
    with coordinator.operation("put_binding_power") as operation:
        operation.note(binding_id="m1/node-a/1", phases_ms={"reserve": 3})
        operation.advance("committing_wake", details={"pods": ["pod-b"]})
    record = coordinator.list_operations(limit=1)[0]
    assert record["details"]["binding_id"] == "m1/node-a/1"
    assert record["details"]["phases_ms"] == {"reserve": 3}
    for index in range(150):
        with coordinator.operation(f"op-{index}"):
            pass
    assert len(redis.hashes[rediskeys.SM_OPERATIONS_KEY]) <= 100

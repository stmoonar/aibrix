import json
import os
import time

import pytest
from fastapi.testclient import TestClient

from tre_common import rediskeys
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.api.v2 import ServiceManagerV2, create_app
from tre_sm.state.operations import (
    OperationBusy,
    OperationCoordinator,
    OperationFenceLost,
    _ACQUIRE_SCRIPT,
    _FINISH_SCRIPT,
    _LEAVE_QUEUE_SCRIPT,
    _RENEW_SCRIPT,
    _UPDATE_SCRIPT,
    current_fence,
    current_operation,
)
from tre_sm.state.store import (
    StateConflict,
    StateFenceError,
    StateStore,
    _SAVE_SCRIPT,
)
from tre_sm.state.fleet_store import FleetStateStore, _SAVE_HASH_SCRIPT


class ScriptRedis:
    """Small deterministic Redis/Lua model for fencing and CAS tests."""

    def __init__(self):
        self.values = {}
        self.hashes = {}
        self.zsets = {}
        self.sets = {}
        self.hgetall_calls = 0
        self.eval_calls = []
        self.now_ms = 1_700_000_000_000

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = str(value)

    def delete(self, key):
        self.values.pop(key, None)
        self.hashes.pop(key, None)
        self.sets.pop(key, None)

    def hgetall(self, key):
        self.hgetall_calls += 1
        return dict(self.hashes.get(key, {}))

    def hmget(self, key, fields):
        bucket = self.hashes.get(key, {})
        return [bucket.get(field) for field in fields]

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def sadd(self, key, *values):
        self.sets.setdefault(key, set()).update(values)

    def srem(self, key, *values):
        self.sets.get(key, set()).difference_update(values)

    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    def hset(self, key, mapping=None, **kwargs):
        bucket = self.hashes.setdefault(key, {})
        if mapping:
            bucket.update(mapping)
        bucket.update(kwargs)

    def eval(self, script, numkeys, *keys_and_args):
        self.eval_calls.append(script)
        keys = list(keys_and_args[:numkeys])
        args = list(keys_and_args[numkeys:])
        if script == _ACQUIRE_SCRIPT:
            lock_key, counter_key, operations_key, queue_key, deadlines_key, seq_key, running_key = keys
            owner, _ttl, operation_id, kind, started_at, request_json, ticket, waiter_ttl = args
            queue = self.zsets.setdefault(queue_key, {})
            deadlines = self.hashes.setdefault(deadlines_key, {})
            for waiter in list(queue):
                if int(deadlines.get(waiter, 0)) <= self.now_ms:
                    queue.pop(waiter)
                    deadlines.pop(waiter, None)
            if ticket:
                if ticket not in queue:
                    seq = int(self.values.get(seq_key, 0)) + 1
                    self.values[seq_key] = str(seq)
                    queue[ticket] = seq
                deadlines[ticket] = str(self.now_ms + int(waiter_ttl))
            if lock_key in self.values:
                return [0, self.values[lock_key]]
            head = min(queue, key=queue.get) if queue else None
            if head is not None and head != ticket:
                return [0, f"queued:{head}"]
            token = int(self.values.get(counter_key, 0)) + 1
            self.values[counter_key] = str(token)
            if ticket:
                queue.pop(ticket, None)
                deadlines.pop(ticket, None)
            lock_value = f"{owner}:{token}"
            self.values[lock_key] = lock_value
            self.hashes.setdefault(operations_key, {})[operation_id] = json.dumps(
                {
                    "operation_id": operation_id,
                    "kind": kind,
                    "owner": owner,
                    "fencing_token": token,
                    "status": "running",
                    "phase": "acquired",
                    "started_at": started_at,
                    "updated_at": started_at,
                    "request": json.loads(request_json),
                }
            )
            self.sets.setdefault(running_key, set()).add(operation_id)
            return [token, lock_value]
        if script == _LEAVE_QUEUE_SCRIPT:
            self.zsets.get(keys[0], {}).pop(args[0], None)
            self.hashes.get(keys[1], {}).pop(args[0], None)
            return 1
        if script == _RENEW_SCRIPT:
            return int(self.values.get(keys[0]) == args[0])
        if script == _UPDATE_SCRIPT:
            lock_key, operations_key, running_key = keys
            lock_value, operation_id, record, status = args
            if self.values.get(lock_key) != lock_value:
                return 0
            self.hashes.setdefault(operations_key, {})[operation_id] = record
            running = self.sets.setdefault(running_key, set())
            if status == "running":
                running.add(operation_id)
            else:
                running.discard(operation_id)
            return 1
        if script == _FINISH_SCRIPT:
            lock_key, operations_key, running_key = keys
            lock_value, operation_id, record = args
            if self.values.get(lock_key) != lock_value:
                return 0
            self.hashes.setdefault(operations_key, {})[operation_id] = record
            self.sets.setdefault(running_key, set()).discard(operation_id)
            self.values.pop(lock_key, None)
            return 1
        if script == _SAVE_SCRIPT:
            state_key, version_key, lock_key = keys
            expected, next_version, require_fence, lock_value, *pairs = args
            current = int(self.values.get(version_key, 0))
            if current != int(expected):
                return [0, current]
            if require_fence == "1" and self.values.get(lock_key) != lock_value:
                return [-1, current]
            mapping = dict(zip(pairs[::2], pairs[1::2]))
            self.hashes[state_key] = mapping
            self.values[version_key] = str(next_version)
            return [1, int(next_version)]
        if script == _SAVE_HASH_SCRIPT:
            state_key, version_key, lock_key = keys
            expected, next_version, lock_value, *pairs = args
            current = int(self.values.get(version_key, 0))
            if current != int(expected):
                return [0, current]
            if self.values.get(lock_key) != lock_value:
                return [-1, current]
            self.hashes[state_key] = dict(zip(pairs[::2], pairs[1::2]))
            self.values[version_key] = str(next_version)
            return [1, int(next_version)]
        raise AssertionError("unexpected Lua script")


def _binding():
    return Binding("pod-a", "m1", Slot("node-a", (0,)), awake=True)


def _registry():
    return Registry(
        ClusterTopology(
            nodes=(NodeSpec("node-a", 2, ((0, 1),), ("GPU-0", "GPU-1")),)
        ),
        [
            ModelSpec(
                name="m1",
                weights_path="/m1",
                tp_size=1,
                min_replicas=0,
                max_replicas=1,
                vllm_image="image",
                slo=SloSpec(1000, 100, 10000),
                trs=TrsParams(
                    0.04, 1.0, 2.625, 1.0, 0.5, 1.0,
                    0.8, 1.0, 1.25, 4.0, 0.05, 3,
                ),
            )
        ],
    )


def test_fenced_operation_atomically_writes_state_and_records_success():
    redis = ScriptRedis()
    coordinator = OperationCoordinator(redis, owner="pod-a", lease_ttl_ms=30_000)
    store = StateStore(redis, require_fence=True)

    with coordinator.operation("reconcile") as operation:
        assert current_fence() == operation.fence
        operation.advance("persisting")
        assert store.save([_binding()], expected_version=0) == 1

    assert current_fence() is None
    assert store.load().bindings == [_binding()]
    [record] = coordinator.list_operations()
    assert record["status"] == "succeeded"
    assert record["fencing_token"] == 1
    assert redis.values.get(rediskeys.SM_WRITER_LOCK_KEY) is None
    assert _SAVE_SCRIPT in redis.eval_calls


def test_a_queued_writer_raises_the_shutdown_error_instead_of_taking_the_lock():
    redis = ScriptRedis()
    holder = OperationCoordinator(redis, owner="pod-a")
    waiter = OperationCoordinator(redis, owner="pod-b", poll_interval_s=0.01)
    boom = RuntimeError("shutting down")
    with holder.operation("reconcile"):
        waiter.begin_shutdown(boom)
        with pytest.raises(RuntimeError) as caught:
            waiter.acquire("late", wait_s=5)
        assert caught.value is boom
    assert redis.values.get(rediskeys.SM_WRITER_LOCK_KEY) is None


def test_second_writer_is_rejected_while_lease_is_held():
    redis = ScriptRedis()
    first = OperationCoordinator(redis, owner="pod-a")
    second = OperationCoordinator(redis, owner="pod-b")
    handle = first.acquire("long_repair")

    with pytest.raises(OperationBusy) as exc_info:
        second.acquire("reconcile")

    assert "pod-a:1" in str(exc_info.value)
    assert first._finish(
        handle.fence,
        {
            "operation_id": handle.operation_id,
            "status": "succeeded",
        },
    )


def test_lost_fence_blocks_state_commit_and_marks_operation_lost():
    redis = ScriptRedis()
    coordinator = OperationCoordinator(redis, owner="pod-a")
    store = StateStore(redis, require_fence=True)

    with pytest.raises(OperationFenceLost):
        with coordinator.operation("wake"):
            redis.values[rediskeys.SM_WRITER_LOCK_KEY] = "pod-b:999"
            with pytest.raises(StateFenceError):
                store.save([_binding()], expected_version=0)

    assert store.load().version == 0
    assert store.load().bindings == []


def test_atomic_cas_rejects_stale_version_without_overwrite():
    redis = ScriptRedis()
    coordinator = OperationCoordinator(redis, owner="pod-a")
    store = StateStore(redis, require_fence=True)

    with pytest.raises(StateConflict):
        with coordinator.operation("write"):
            store.save([_binding()], expected_version=0)
            store.save([], expected_version=0)

    assert store.load().version == 1
    assert store.load().bindings == [_binding()]


def test_production_store_refuses_non_atomic_client():
    class NoEvalRedis:
        pass

    store = StateStore(NoEvalRedis(), require_fence=True)

    with pytest.raises(StateFenceError, match="atomic Redis EVAL"):
        store.save([], expected_version=0)


def test_http_mutation_uses_fence_and_exposes_operation_journal():
    redis = ScriptRedis()
    coordinator = OperationCoordinator(redis, owner="sm-pod")
    store = StateStore(redis, require_fence=True)
    with coordinator.operation("bootstrap"):  # one sleeping binding to wake
        store.save([Binding("pod-a", "m1", Slot("node-a", (0,)), awake=False)], expected_version=0)
    service = ServiceManagerV2(_registry(), store, operation_coordinator=coordinator)
    client = TestClient(create_app(service))

    response = client.put("/v2/models/m1/target", json={"wake_replicas": 1})
    operations = [op for op in client.get("/v2/operations").json()["operations"] if op["kind"] != "bootstrap"]

    assert response.status_code == 200
    assert response.json()["version"] == 2
    assert len(operations) == 1
    assert operations[0]["kind"] == "put_model_target"
    assert operations[0]["status"] == "succeeded"
    operation_id = operations[0]["operation_id"]
    assert client.get(f"/v2/operations/{operation_id}").json() == operations[0]


def test_submitted_operation_runs_under_fence_and_persists_result():
    redis = ScriptRedis()
    coordinator = OperationCoordinator(redis, owner="sm-pod")
    observed = []

    operation_id = coordinator.submit(
        "fleet_repair",
        lambda handle: observed.append(
            (handle.operation_id, current_fence(), current_operation())
        ),
    )

    assert coordinator.wait(operation_id, timeout_s=2.0) is True
    assert observed[0][0] == operation_id
    assert observed[0][1] == observed[0][2].fence
    record = coordinator.get_operation(operation_id)
    assert record["status"] == "succeeded"


def test_stale_running_operation_is_superseded_under_new_fence():
    redis = ScriptRedis()
    coordinator = OperationCoordinator(redis, owner="sm-pod")
    stale = coordinator.acquire(
        "fleet_repair", request={"awake_binding_ids": ["m1/node-a/0"]}
    )
    redis.values.pop(rediskeys.SM_WRITER_LOCK_KEY)

    [record] = coordinator.stale_running_operations(kind="fleet_repair")
    assert record["operation_id"] == stale.operation_id

    with coordinator.operation("fleet_repair") as replacement:
        replacement.supersede(stale.operation_id)

    recovered = coordinator.get_operation(stale.operation_id)
    assert recovered["status"] == "superseded"
    assert recovered["replacement_operation_id"] == replacement.operation_id


def test_desired_intent_commits_before_observed_state_converges():
    redis = ScriptRedis()
    coordinator = OperationCoordinator(redis, owner="sm-pod")
    legacy_store = StateStore(redis, require_fence=True)
    fleet_store = FleetStateStore(redis)
    sleeping = Binding("pod-a", "m1", Slot("node-a", (0,)), awake=False)
    with coordinator.operation("bootstrap"):
        legacy_store.save([sleeping], expected_version=0)
        fleet_store.bootstrap([sleeping])
    service = ServiceManagerV2(
        _registry(),
        legacy_store,
        operation_coordinator=coordinator,
        fleet_store=fleet_store,
    )
    client = TestClient(create_app(service))

    response = client.put("/v2/models/m1/target", json={"wake_replicas": 1})
    fleet = client.get("/v2/fleet/state").json()

    assert response.status_code == 200
    assert fleet["desired"][0]["power"] == "awake"
    assert fleet["desired"][0]["generation"] == 2
    assert fleet["observed"][0]["physical_power"] == "sleeping"
    assert fleet["mismatches"] == [
        {
            "code": "desired_power_mismatch",
            "binding_id": "m1/node-a/0",
            "desired_power": "awake",
            "observed_power": "sleeping",
        }
    ]


# ------------------------------------------------------------ review 2 P2-4
def _redis_backends():
    backends = [pytest.param(lambda: (ScriptRedis(), None), id="script-model")]
    url = os.environ.get("TRE_TEST_REDIS_URL")
    if url:
        backends.append(pytest.param(lambda: _real_redis(url), id="real-redis-lua"))
    return backends


def _real_redis(url):
    import redis as redis_lib

    client = redis_lib.Redis.from_url(url)
    for key in (
        rediskeys.SM_WRITER_LOCK_KEY,
        rediskeys.SM_FENCE_COUNTER_KEY,
        rediskeys.SM_OPERATIONS_KEY,
        rediskeys.SM_RUNNING_OPERATIONS_KEY,
        rediskeys.SM_WRITER_QUEUE_KEY,
        rediskeys.SM_WRITER_QUEUE_DEADLINES_KEY,
        rediskeys.SM_WRITER_QUEUE_SEQ_KEY,
    ):
        client.delete(key)
    return client, client


def _expire_waiters(redis, real, ttl_ms):
    if real is None:
        redis.now_ms += ttl_ms + 1
    else:
        time.sleep(ttl_ms / 1000.0 + 0.1)


@pytest.mark.parametrize("make", _redis_backends())
def test_writer_lock_is_first_come_first_served(make):
    redis, real = make()
    holder = OperationCoordinator(redis, owner="holder", poll_interval_s=0.05, waiter_ttl_ms=300)
    commit = OperationCoordinator(redis, owner="commit", poll_interval_s=0.05, waiter_ttl_ms=300)
    poller = OperationCoordinator(redis, owner="poller", poll_interval_s=0.05, waiter_ttl_ms=300)
    held = holder.acquire("put_model_routable")

    # The drained commit starts waiting (joins the queue) while the lock is held.
    with pytest.raises(OperationBusy):
        commit._acquire_once("put_binding_power_commit", ticket="commit-ticket")
    assert holder._finish(held.fence, {"operation_id": held.operation_id, "status": "succeeded"})

    # A later caller - waiting (its own ticket) or not - does not jump the queue.
    with pytest.raises(OperationBusy) as queued:
        poller._acquire_once("reconcile", ticket="poller-ticket")
    assert "queued:commit-ticket" in str(queued.value)
    with pytest.raises(OperationBusy):
        poller._acquire_once("reconcile")

    granted = commit._acquire_once("put_binding_power_commit", ticket="commit-ticket")
    assert granted.fence.owner == "commit"
    assert commit._finish(granted.fence, {"operation_id": granted.operation_id, "status": "succeeded"})
    # next in line
    after = poller._acquire_once("reconcile", ticket="poller-ticket")
    assert poller._finish(after.fence, {"operation_id": after.operation_id, "status": "succeeded"})


@pytest.mark.parametrize("make", _redis_backends())
def test_a_waiter_that_stops_polling_loses_its_place(make):
    redis, real = make()
    ghost = OperationCoordinator(redis, owner="ghost", poll_interval_s=0.05, waiter_ttl_ms=200)
    other = OperationCoordinator(redis, owner="other", poll_interval_s=0.05, waiter_ttl_ms=200)
    held = other.acquire("x")
    with pytest.raises(OperationBusy):
        ghost._acquire_once("y", ticket="ghost-ticket")  # queued, then the caller died
    assert other._finish(held.fence, {"operation_id": held.operation_id, "status": "succeeded"})
    with pytest.raises(OperationBusy):
        other._acquire_once("z")
    _expire_waiters(redis, real, 200)
    handle = other._acquire_once("z")
    assert other._finish(handle.fence, {"operation_id": handle.operation_id, "status": "succeeded"})


@pytest.mark.parametrize("make", _redis_backends())
def test_giving_up_leaves_the_queue(make):
    redis, _real = make()
    holder = OperationCoordinator(redis, owner="holder", poll_interval_s=0.02, waiter_ttl_ms=5_000)
    waiter = OperationCoordinator(redis, owner="waiter", poll_interval_s=0.02, waiter_ttl_ms=5_000)
    held = holder.acquire("x")
    with pytest.raises(OperationBusy):
        waiter.acquire("y", wait_s=0.1)  # times out and dequeues itself
    assert holder._finish(held.fence, {"operation_id": held.operation_id, "status": "succeeded"})
    handle = holder._acquire_once("z")  # nobody queued any more
    assert holder._finish(handle.fence, {"operation_id": handle.operation_id, "status": "succeeded"})


@pytest.mark.parametrize("make", _redis_backends())
def test_a_starting_binding_phase_is_visible_to_the_startup_gate(make):
    """Review 4 P1: the startup gate recognizes a Pod pre-authorized by the
    writer-lock holder from the holder's journal record (phase + details), read
    by ANOTHER process's coordinator while the lock is held."""
    redis, _real = make()
    holder = OperationCoordinator(redis, owner="holder", poll_interval_s=0.05, waiter_ttl_ms=300)
    gate = OperationCoordinator(redis, owner="gate", poll_interval_s=0.05, waiter_ttl_ms=300)
    with holder.operation("defrag", wait_s=0.0) as operation:
        operation.advance("starting_binding", details={"binding_id": "m1/node-a/1"})
        active = gate.active_operation()
        assert (active["kind"], active["phase"]) == ("defrag", "starting_binding")
        assert active["details"] == {"binding_id": "m1/node-a/1"}
        operation.advance("starting_binding", details={"binding_id": "m1/node-a/1", "pod_uid": "uid-1"})
        assert gate.active_operation()["details"]["pod_uid"] == "uid-1"
        with pytest.raises(OperationBusy):
            gate._acquire_once("startup_admit")
        operation.advance("executing")
        assert gate.active_operation()["phase"] == "executing"
    assert gate.active_operation() is None


@pytest.mark.parametrize("make", _redis_backends())
def test_a_running_operation_older_than_the_newest_1000_records_is_still_stale(make):
    """10-28: stale_running_operations looked at the newest 1000 journal records
    only (the supervisor read the whole journal for it, twice per tick), so a
    record a dead SM left running was never superseded or recovered once 1000
    newer operations had run. Both a record written before the running index
    existed and one written with it are found, newest first."""
    redis, _real = make()
    legacy = {
        "operation_id": "legacy-op", "kind": "put_binding_power", "owner": "old-sm",
        "fencing_token": 1, "status": "running", "phase": "acquired",
        "started_at": "2026-10-01T00:00:00+00:00",
    }
    redis.hset(rediskeys.SM_OPERATIONS_KEY, mapping={"legacy-op": json.dumps(legacy)})
    coordinator = OperationCoordinator(redis, owner="sm-pod")
    dead = coordinator.acquire("fleet_repair")
    redis.delete(rediskeys.SM_WRITER_LOCK_KEY)  # its SM died holding the lock
    for _ in range(1001):
        with coordinator.operation("put_binding_power"):
            pass

    with coordinator.operation("probe") as active:
        stale = coordinator.stale_running_operations()
        assert coordinator.active_operation()["operation_id"] == active.operation_id
    assert [record["operation_id"] for record in stale] == [dead.operation_id, "legacy-op"]
    assert [r["operation_id"] for r in coordinator.stale_running_operations(kind="fleet_repair")] == [
        dead.operation_id
    ]
    if _real is None:  # the journal is read in full once per process, not per call
        calls = redis.hgetall_calls
        coordinator.stale_running_operations()
        coordinator.active_operation()
        assert redis.hgetall_calls == calls

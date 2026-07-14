import json

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


class ScriptRedis:
    """Small deterministic Redis/Lua model for fencing and CAS tests."""

    def __init__(self):
        self.values = {}
        self.hashes = {}
        self.eval_calls = []

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = str(value)

    def delete(self, key):
        self.values.pop(key, None)
        self.hashes.pop(key, None)

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

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
            lock_key, counter_key, operations_key = keys
            token = int(self.values.get(counter_key, 0)) + 1
            self.values[counter_key] = str(token)
            if lock_key in self.values:
                return [0, self.values[lock_key]]
            owner, _ttl, operation_id, kind, started_at, request_json = args
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
            return [token, lock_value]
        if script == _RENEW_SCRIPT:
            return int(self.values.get(keys[0]) == args[0])
        if script == _UPDATE_SCRIPT:
            lock_key, operations_key = keys
            lock_value, operation_id, record = args
            if self.values.get(lock_key) != lock_value:
                return 0
            self.hashes.setdefault(operations_key, {})[operation_id] = record
            return 1
        if script == _FINISH_SCRIPT:
            lock_key, operations_key = keys
            lock_value, operation_id, record = args
            if self.values.get(lock_key) != lock_value:
                return 0
            self.hashes.setdefault(operations_key, {})[operation_id] = record
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
    service = ServiceManagerV2(
        _registry(),
        StateStore(redis, require_fence=True),
        operation_coordinator=coordinator,
    )
    client = TestClient(create_app(service))

    response = client.put("/v2/models/m1/target", json={"wake_replicas": 1})
    operations = client.get("/v2/operations").json()["operations"]

    assert response.status_code == 200
    assert response.json()["version"] == 1
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

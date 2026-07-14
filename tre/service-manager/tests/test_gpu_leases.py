import json

import pytest

from tre_common import rediskeys
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.state.gpu_leases import (
    GpuLeaseConflict,
    GpuLeaseStore,
    _ACQUIRE_GPU_SCRIPT,
    _REBUILD_GPU_SCRIPT,
    _RELEASE_GPU_SCRIPT,
)
from tre_sm.state.operations import WriterFence, _CURRENT_FENCE


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.hashes = {}
        self.now_ms = 1000

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def eval(self, script, numkeys, *keys_and_args):
        keys = keys_and_args[:numkeys]
        args = list(keys_and_args[numkeys:])
        if script == _ACQUIRE_GPU_SCRIPT:
            leases_key, lock_key = keys
            if self.values.get(lock_key) != args[0]:
                return [-1, "writer_fence_lost", ""]
            binding_id, node, owner = args[1:4]
            token = int(args[4])
            gpu_ids = json.loads(args[5])
            ttl_ms = int(args[6])
            count = int(args[7])
            fields = args[8 : 8 + count]
            phase = args[8 + count]
            bucket = self.hashes.setdefault(leases_key, {})
            for field in fields:
                if field not in bucket:
                    continue
                existing = json.loads(bucket[field])
                active = (
                    existing["expires_at_ms"] == 0
                    or existing["expires_at_ms"] > self.now_ms
                )
                if active and existing["binding_id"] != binding_id:
                    return [0, field, existing["binding_id"]]
            expires = 0 if ttl_ms == 0 else self.now_ms + ttl_ms
            payload = json.dumps(
                {
                    "binding_id": binding_id,
                    "node": node,
                    "gpu_ids": gpu_ids,
                    "owner": owner,
                    "fencing_token": token,
                    "phase": phase,
                    "expires_at_ms": expires,
                }
            )
            for field in fields:
                bucket[field] = payload
            return [1, str(expires), ""]
        if script == _RELEASE_GPU_SCRIPT:
            leases_key, lock_key = keys
            if self.values.get(lock_key) != args[0]:
                return -1
            binding_id, token, count = args[1], int(args[2]), int(args[3])
            bucket = self.hashes.setdefault(leases_key, {})
            for field in args[4 : 4 + count]:
                existing = json.loads(bucket[field]) if field in bucket else None
                if (
                    existing
                    and existing["binding_id"] == binding_id
                    and existing["fencing_token"] <= token
                ):
                    bucket.pop(field)
            return 1
        if script == _REBUILD_GPU_SCRIPT:
            leases_key, lock_key = keys
            if self.values.get(lock_key) != args[0]:
                return -1
            self.hashes[leases_key] = dict(zip(args[1::2], args[2::2]))
            return 1
        raise AssertionError("unexpected script")


class Fence:
    def __init__(self, redis, token):
        self.redis = redis
        self.fence = WriterFence(f"op-{token}", "sm", token, f"sm:{token}")

    def __enter__(self):
        self.redis.values[rediskeys.SM_WRITER_LOCK_KEY] = self.fence.lock_value
        self.context_token = _CURRENT_FENCE.set(self.fence)

    def __exit__(self, *_args):
        _CURRENT_FENCE.reset(self.context_token)


def _binding(name, model, gpus):
    return Binding(name, model, Slot("node-a", tuple(gpus)), awake=False)


def test_tp2_lease_acquires_both_gpu_fields_atomically():
    redis = FakeRedis()
    store = GpuLeaseStore(redis)
    binding = _binding("pod-a", "m1", (0, 1))

    with Fence(redis, 1):
        lease = store.acquire(binding, phase="starting")

    assert lease.gpu_ids == (0, 1)
    assert lease.phase == "starting"
    assert sorted(redis.hashes[rediskeys.SM_GPU_LEASES_KEY]) == [
        "node-a/0",
        "node-a/1",
    ]


def test_tp2_conflict_does_not_partially_acquire_free_gpu():
    redis = FakeRedis()
    store = GpuLeaseStore(redis)
    first = _binding("pod-a", "m1", (1,))
    second = _binding("pod-b", "m2", (0, 1))
    with Fence(redis, 1):
        store.acquire(first, phase="awake")

    with Fence(redis, 2), pytest.raises(GpuLeaseConflict) as exc_info:
        store.acquire(second, phase="starting")

    assert exc_info.value.gpu == "node-a/1"
    assert "node-a/0" not in redis.hashes[rediskeys.SM_GPU_LEASES_KEY]


def test_newer_fence_can_release_older_awake_lease():
    redis = FakeRedis()
    store = GpuLeaseStore(redis)
    binding = _binding("pod-a", "m1", (0,))
    with Fence(redis, 1):
        store.acquire(binding, phase="awake")

    with Fence(redis, 2):
        store.release(binding)

    assert store.load() == []


def test_rebuild_awake_rejects_overlapping_physical_truth():
    redis = FakeRedis()
    store = GpuLeaseStore(redis)
    first = replace_awake(_binding("pod-a", "m1", (0,)))
    second = replace_awake(_binding("pod-b", "m2", (0,)))

    with Fence(redis, 1), pytest.raises(GpuLeaseConflict):
        store.rebuild_awake([first, second])

    assert store.load() == []


def test_bootstrap_preserves_admitted_starting_lease():
    redis = FakeRedis()
    store = GpuLeaseStore(redis)
    starting = _binding("new-pod", "m1", (0, 1))

    with Fence(redis, 3):
        store.rebuild_awake([], starting_bindings=[starting])

    [lease] = store.load()
    assert lease.binding_id == "m1/node-a/0,1"
    assert lease.phase == "starting"
    assert lease.expires_at_ms > 0
    assert sorted(redis.hashes[rediskeys.SM_GPU_LEASES_KEY]) == [
        "node-a/0",
        "node-a/1",
    ]


def replace_awake(binding):
    return Binding(
        binding.serve_id, binding.model, binding.slot, awake=True
    )

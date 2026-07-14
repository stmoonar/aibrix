import json

import pytest

from tre_common import rediskeys
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.state.fleet_store import (
    DesiredBinding,
    FleetStateStore,
    ObservedBinding,
    _SAVE_HASH_SCRIPT,
)
from tre_sm.state.operations import WriterFence, _CURRENT_FENCE
from tre_sm.state.store import StateFenceError


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.hashes = {}

    def get(self, key):
        return self.values.get(key)

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def eval(self, script, numkeys, *keys_and_args):
        assert script == _SAVE_HASH_SCRIPT
        state_key, version_key, lock_key = keys_and_args[:numkeys]
        expected, next_version, lock_value, *pairs = keys_and_args[numkeys:]
        current = int(self.values.get(version_key, 0))
        if current != int(expected):
            return [0, current]
        if self.values.get(lock_key) != lock_value:
            return [-1, current]
        self.hashes[state_key] = dict(zip(pairs[::2], pairs[1::2]))
        self.values[version_key] = str(next_version)
        return [1, int(next_version)]


class Fence:
    def __init__(self, redis):
        self.redis = redis
        self.fence = WriterFence("op-1", "sm-pod", 1, "sm-pod:1")

    def __enter__(self):
        self.redis.values[rediskeys.SM_WRITER_LOCK_KEY] = self.fence.lock_value
        self.token = _CURRENT_FENCE.set(self.fence)
        return self.fence

    def __exit__(self, *_args):
        _CURRENT_FENCE.reset(self.token)


def _legacy_bindings():
    return [
        Binding("pod-old", "m1", Slot("node-a", (0,)), awake=True),
        Binding("pod-sleep", "m2", Slot("node-a", (0, 1)), awake=False),
    ]


def test_bootstrap_separates_stable_desired_from_observed_pod_instance():
    redis = FakeRedis()
    store = FleetStateStore(redis)

    with Fence(redis):
        versions = store.bootstrap(_legacy_bindings())

    desired = store.load_desired()
    observed = store.load_observed()
    assert versions == {"desired_version": 1, "observed_version": 1}
    assert [item.binding_id for item in desired.bindings] == [
        "m1/node-a/0",
        "m2/node-a/0,1",
    ]
    assert not hasattr(desired.bindings[0], "pod_name")
    assert observed.bindings[0].pod_name == "pod-old"


def test_observed_pod_replacement_does_not_change_desired_generation():
    redis = FakeRedis()
    store = FleetStateStore(redis)
    with Fence(redis):
        store.bootstrap(_legacy_bindings())
    desired_before = store.load_desired()
    observed = store.load_observed()
    replaced = [
        ObservedBinding(
            **{
                **item.__dict__,
                "pod_name": "pod-new" if item.model == "m1" else item.pod_name,
            }
        )
        for item in observed.bindings
    ]

    with Fence(redis):
        store.save_observed(replaced, expected_version=observed.version)

    assert store.load_desired() == desired_before
    assert store.load_observed().bindings[0].pod_name == "pod-new"


def test_desired_generation_only_increments_for_real_intent_change():
    desired = DesiredBinding.from_binding(_legacy_bindings()[0])

    unchanged = desired.with_intent(
        power="awake", updated_by="controller", reason="same_target"
    )
    sleeping = desired.with_intent(
        power="sleeping", updated_by="controller", reason="scale_down"
    )

    assert unchanged is desired
    assert sleeping.generation == desired.generation + 1
    assert sleeping.power == "sleeping"
    assert sleeping.updated_by == "controller"


def test_fleet_store_rejects_write_after_fence_loss():
    redis = FakeRedis()
    store = FleetStateStore(redis)
    desired = [DesiredBinding.from_binding(_legacy_bindings()[0])]

    with Fence(redis):
        redis.values[rediskeys.SM_WRITER_LOCK_KEY] = "other:9"
        with pytest.raises(StateFenceError):
            store.save_desired(desired, expected_version=0)

    assert store.load_desired().version == 0

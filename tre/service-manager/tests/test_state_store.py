import pytest

from tre_sm.allocator.slots import Binding, Slot
from tre_sm.state.store import StateConflict, StateStore


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.hashes = {}

    def get(self, key):
        value = self.values.get(key)
        return None if value is None else str(value).encode("utf-8")

    def set(self, key, value):
        self.values[key] = str(value)

    def delete(self, key):
        self.hashes.pop(key, None)

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def hset(self, key, mapping):
        bucket = self.hashes.setdefault(key, {})
        for field, value in mapping.items():
            bucket[str(field).encode("utf-8")] = str(value).encode("utf-8")


class StringRedis(FakeRedis):
    def get(self, key):
        return self.values.get(key)

    def hgetall(self, key):
        return {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in self.hashes.get(key, {}).items()
        }


def test_state_store_round_trips_bindings_and_versions():
    store = StateStore(FakeRedis())
    binding = Binding(
        serve_id="serve-a",
        model="dsqwen-7b",
        slot=Slot("node-a", (0,)),
        awake=False,
    )

    assert store.load().version == 0

    next_version = store.save([binding], expected_version=0)

    assert next_version == 1
    reloaded = store.load()
    assert reloaded.version == 1
    assert reloaded.bindings == [binding]


def test_state_store_rejects_stale_expected_version_without_overwriting():
    redis = FakeRedis()
    store = StateStore(redis)
    first = Binding(
        serve_id="serve-a",
        model="dsqwen-7b",
        slot=Slot("node-a", (0,)),
        awake=True,
    )
    stale = Binding(
        serve_id="serve-b",
        model="dsqwen-14b",
        slot=Slot("node-a", (2, 3)),
        awake=True,
    )

    store.save([first], expected_version=0)

    with pytest.raises(StateConflict):
        store.save([stale], expected_version=0)

    assert store.load().bindings == [first]


def test_state_store_accepts_string_redis_responses():
    store = StateStore(StringRedis())
    binding = Binding(
        serve_id="serve-a",
        model="dsqwen-7b",
        slot=Slot("node-a", (0,)),
        awake=True,
    )

    store.save([binding], expected_version=0)

    assert store.load().version == 1
    assert store.load().bindings == [binding]

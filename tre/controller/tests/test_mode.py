from __future__ import annotations

from tre_controller.mode import CONTROLLER_MODE_KEY, ObserveModeGate


class FakeRedis:
    def __init__(self, value=None) -> None:
        self.value = value
        self.reads = 0
        self.fail = False

    def get(self, key):
        assert key == CONTROLLER_MODE_KEY
        self.reads += 1
        if self.fail:
            raise RuntimeError("redis down")
        return self.value


def test_observe_gate_detects_mode() -> None:
    assert ObserveModeGate(FakeRedis(b"observe")).is_observe() is True
    assert ObserveModeGate(FakeRedis("observe")).is_observe() is True
    assert ObserveModeGate(FakeRedis(b"active")).is_observe() is False


def test_observe_gate_absent_or_unknown_mode_is_observe_fail_closed() -> None:
    # 2026-09-28: a missing key no longer means active - the console (or the
    # deploy procedure) must set the mode explicitly before TRE actuates.
    assert ObserveModeGate(FakeRedis(None)).is_observe() is True
    assert ObserveModeGate(FakeRedis(b"bogus")).is_observe() is True


def test_observe_gate_caches_reads_within_ttl() -> None:
    clock = {"t": 100.0}
    redis = FakeRedis(b"observe")
    gate = ObserveModeGate(redis, ttl_s=1.0, clock=lambda: clock["t"])

    assert gate.is_observe() is True
    assert gate.is_observe() is True
    assert redis.reads == 1  # second call served from cache

    clock["t"] = 101.5  # past TTL
    redis.value = b"active"
    assert gate.is_observe() is False
    assert redis.reads == 2


def test_observe_gate_fresh_read_bypasses_the_cache() -> None:
    clock = {"t": 100.0}
    redis = FakeRedis(b"active")
    gate = ObserveModeGate(redis, ttl_s=10.0, clock=lambda: clock["t"])
    assert gate.is_observe() is False
    redis.value = b"observe"
    assert gate.is_observe() is False  # cached
    assert gate.is_observe_fresh() is True
    assert gate.is_observe() is True  # the fresh read refreshed the cache
    assert redis.reads == 2


def test_observe_gate_never_read_fails_closed_to_observe() -> None:
    redis = FakeRedis(b"active")
    redis.fail = True
    assert ObserveModeGate(redis).is_observe() is True
    assert ObserveModeGate(redis).last_known_mode() is None


def test_observe_gate_keeps_the_last_known_mode_on_a_read_error() -> None:
    clock = {"t": 0.0}
    active = FakeRedis(b"active")
    gate = ObserveModeGate(active, ttl_s=1.0, clock=lambda: clock["t"])
    assert gate.is_observe() is False
    active.fail = True
    clock["t"] = 5.0
    assert gate.is_observe() is False  # last known: active
    assert gate.is_observe_fresh() is False

    observe = FakeRedis(b"observe")
    gate = ObserveModeGate(observe, ttl_s=1.0, clock=lambda: clock["t"])
    assert gate.is_observe() is True
    observe.fail = True
    clock["t"] = 10.0
    assert gate.is_observe() is True  # last known: observe
    assert gate.last_known_mode() == "observe"

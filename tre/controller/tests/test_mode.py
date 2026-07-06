from __future__ import annotations

from tre_controller.mode import CONTROLLER_MODE_KEY, ObserveModeGate


class FakeRedis:
    def __init__(self, value=None) -> None:
        self.value = value
        self.reads = 0

    def get(self, key):
        assert key == CONTROLLER_MODE_KEY
        self.reads += 1
        return self.value


def test_observe_gate_detects_mode_and_defaults_active() -> None:
    assert ObserveModeGate(FakeRedis(None)).is_observe() is False
    assert ObserveModeGate(FakeRedis(b"observe")).is_observe() is True
    assert ObserveModeGate(FakeRedis("observe")).is_observe() is True
    assert ObserveModeGate(FakeRedis(b"active")).is_observe() is False


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


def test_observe_gate_fails_safe_to_active_on_redis_error() -> None:
    class Boom:
        def get(self, key):
            raise RuntimeError("redis down")

    assert ObserveModeGate(Boom()).is_observe() is False

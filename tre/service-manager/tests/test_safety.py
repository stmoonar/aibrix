import json

import pytest

from tre_common import rediskeys
from tre_sm.state.actuation import SmActuation
from tre_sm.state.safety import ClusterSafetyGate, MaintenanceLockLost


class FakeRedis:
    """Plain key-value Redis; ``fail`` makes every read raise."""

    def __init__(self, values=None):
        self.kv = dict(values or {})
        self.lists: dict[str, list] = {}
        self.sets = []
        self.fail = False

    def get(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        value = self.kv.get(key)
        return value.encode() if isinstance(value, str) else value

    def set(self, key, value):
        self.kv[key] = value
        self.sets.append((key, value))

    def delete(self, *keys):
        for key in keys:
            self.kv.pop(key, None)

    def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)

    def ltrim(self, key, start, stop):
        self.lists[key] = self.lists.get(key, [])[start:stop + 1]


class FakePressure:
    def __init__(self, values):
        self.values = list(values)

    def node_pressure_reasons(self):
        if len(self.values) > 1:
            return self.values.pop(0)
        return self.values[0]


class FakeOperation:
    def __init__(self, operation_id="op-1"):
        self.operation_id = operation_id
        self.phases = []

    def assert_active(self):
        return None

    def advance(self, phase, *, details=None):
        self.phases.append((phase, details))


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


# ----------------------------------------------------------- maintenance lock
def test_the_maintenance_lock_is_independent_of_the_controller_mode():
    """2026-09-28: the fleet repair no longer borrows (nor writes) the
    controller mode; it holds its own SM maintenance key."""
    redis = FakeRedis({rediskeys.CONTROLLER_MODE_KEY: "active"})
    gate = ClusterSafetyGate(redis, FakePressure([{}]))

    gate.acquire_maintenance("op-1", kind="fleet_repair", owner="sm-0")
    gate.assert_maintenance_held("op-1")  # controller active: no objection

    assert json.loads(redis.kv[rediskeys.SM_MAINTENANCE_KEY])["operation_id"] == "op-1"
    assert redis.kv[rediskeys.CONTROLLER_MODE_KEY] == "active"
    assert all(key != rediskeys.CONTROLLER_MODE_KEY for key, _ in redis.sets)
    assert not hasattr(gate, "enter_recovery_observe")
    assert not hasattr(gate, "assert_controller_observe")

    gate.release_maintenance("op-1")
    assert rediskeys.SM_MAINTENANCE_KEY not in redis.kv


def test_a_cleared_or_taken_over_maintenance_lock_aborts_the_repair():
    redis = FakeRedis()
    gate = ClusterSafetyGate(redis, FakePressure([{}]))
    with pytest.raises(MaintenanceLockLost):
        gate.assert_maintenance_held("op-1")  # never taken
    gate.acquire_maintenance("op-2", kind="fleet_repair")
    with pytest.raises(MaintenanceLockLost):
        gate.wait_until_healthy(FakeOperation("op-1"))
    gate.release_maintenance("op-1")  # not the holder: left alone
    assert rediskeys.SM_MAINTENANCE_KEY in redis.kv


def test_safety_gate_pauses_on_pressure_and_requires_clear_hysteresis():
    clock = Clock()
    operation = FakeOperation()
    redis = FakeRedis()
    gate = ClusterSafetyGate(
        redis,
        FakePressure(
            [
                {"node-a": ["DiskPressure"]},
                {},
                {},
                {},
            ]
        ),
        clear_hysteresis_s=10,
        pressure_timeout_s=60,
        poll_interval_s=5,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    gate.acquire_maintenance(operation.operation_id, kind="fleet_repair")

    gate.wait_until_healthy(operation)

    assert clock.now == 15
    assert operation.phases[0][0] == "waiting_node_pressure"
    assert operation.phases[-1][0] == "cluster_healthy"


def test_the_pressure_wait_aborts_when_the_maintenance_lock_is_deleted():
    clock = Clock()
    operation = FakeOperation()
    redis = FakeRedis()

    def sleep(seconds):
        clock.sleep(seconds)
        redis.delete(rediskeys.SM_MAINTENANCE_KEY)  # operator abort

    gate = ClusterSafetyGate(
        redis, FakePressure([{"node-a": ["DiskPressure"]}]),
        poll_interval_s=5, monotonic=clock.monotonic, sleep=sleep,
    )
    gate.acquire_maintenance(operation.operation_id, kind="fleet_repair")
    with pytest.raises(MaintenanceLockLost):
        gate.wait_until_healthy(operation)


# ------------------------------------------------------------ actuation switch
def _actuation(redis, clock=None):
    clock = clock or Clock()
    return SmActuation(redis, ttl_s=0.0, monotonic=clock.monotonic, wall_ms=lambda: 1000)


def test_sm_actuation_prefers_its_own_key_then_the_controller_mode():
    redis = FakeRedis({rediskeys.SM_ACTUATION_KEY: "active", rediskeys.CONTROLLER_MODE_KEY: "observe"})
    actuation = _actuation(redis)
    assert actuation.resolve() == ("active", "sm")

    del redis.kv[rediskeys.SM_ACTUATION_KEY]
    assert actuation.resolve() == ("observe", "controller")
    redis.kv[rediskeys.CONTROLLER_MODE_KEY] = "active"
    assert actuation.resolve() == ("active", "controller")


def test_sm_actuation_with_both_keys_absent_is_observe():
    assert _actuation(FakeRedis()).resolve() == ("observe", "default")
    garbage = FakeRedis({rediskeys.SM_ACTUATION_KEY: "bogus"})
    assert _actuation(garbage).mode() == "observe"


def test_sm_actuation_keeps_the_last_known_mode_on_a_read_error():
    redis = FakeRedis({rediskeys.SM_ACTUATION_KEY: "active"})
    actuation = _actuation(redis)
    assert actuation.mode() == "active"
    redis.fail = True
    assert actuation.resolve() == ("active", "last_known")


def test_sm_actuation_never_read_is_observe_fail_closed():
    redis = FakeRedis({rediskeys.SM_ACTUATION_KEY: "active"})
    redis.fail = True
    assert _actuation(redis).resolve() == ("observe", "fail_closed")


def test_suppressed_actions_are_recorded_once_per_detail():
    clock = Clock()
    redis = FakeRedis()
    actuation = _actuation(redis, clock)

    assert actuation.record_suppressed("fleet_repair", {"drift": [1]}) is True
    assert actuation.record_suppressed("fleet_repair", {"drift": [1]}) is False  # deduplicated
    assert actuation.record_suppressed("fleet_repair", {"drift": [2]}) is True
    clock.now += 301.0
    assert actuation.record_suppressed("fleet_repair", {"drift": [2]}) is True

    stored = [json.loads(item) for item in redis.lists[rediskeys.SM_SUPPRESSED_ACTIONS_KEY]]
    assert [item["detail"] for item in stored] == [{"drift": [2]}, {"drift": [2]}, {"drift": [1]}]
    assert actuation.recent_suppressed()[0]["action"] == "fleet_repair"


def test_the_safety_gate_exposes_the_actuation_state():
    gate = ClusterSafetyGate(FakeRedis({rediskeys.CONTROLLER_MODE_KEY: "observe"}), FakePressure([{}]))
    assert gate.actuation_mode() == "observe"
    gate.record_suppressed("reap_rejected_deployments", {"deployments": ["d1"]})
    state = gate.actuation_state()
    assert state["mode"] == "observe" and state["source"] == "controller"
    assert state["suppressed"][0]["action"] == "reap_rejected_deployments"

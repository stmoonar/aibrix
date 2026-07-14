import pytest

from tre_common import rediskeys
from tre_sm.state.safety import ClusterSafetyGate, ControllerNotPaused


class FakeRedis:
    def __init__(self, mode="observe"):
        self.mode = mode

    def get(self, key):
        assert key == rediskeys.CONTROLLER_MODE_KEY
        return self.mode.encode()


class FakePressure:
    def __init__(self, values):
        self.values = list(values)

    def node_pressure_reasons(self):
        if len(self.values) > 1:
            return self.values.pop(0)
        return self.values[0]


class FakeOperation:
    def __init__(self):
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


def test_safety_gate_rejects_repair_while_controller_active():
    gate = ClusterSafetyGate(FakeRedis("active"), FakePressure([{}]))

    with pytest.raises(ControllerNotPaused):
        gate.assert_controller_observe()


def test_safety_gate_pauses_on_pressure_and_requires_clear_hysteresis():
    clock = Clock()
    operation = FakeOperation()
    gate = ClusterSafetyGate(
        FakeRedis(),
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

    gate.wait_until_healthy(operation)

    assert clock.now == 15
    assert operation.phases[0][0] == "waiting_node_pressure"
    assert operation.phases[-1][0] == "cluster_healthy"

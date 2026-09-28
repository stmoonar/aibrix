import pytest

from tre_sm.ops import sleep_primitive


class VirtualClock(sleep_primitive.Clock):
    """Monotonic time that only moves when someone sleeps (tests never wait)."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += max(float(seconds), 0.001)


@pytest.fixture(autouse=True)
def virtual_sleep_clock(monkeypatch):
    """The sleep primitive's grace delays / drain polls run on virtual time."""
    clock = VirtualClock()
    monkeypatch.setattr(sleep_primitive, "DEFAULT_CLOCK", clock)
    return clock

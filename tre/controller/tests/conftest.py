import pytest


class _VirtualClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(float(seconds), 0.001)


@pytest.fixture(autouse=True)
def _virtual_sm_sleep_clock(monkeypatch):
    """In-process service-manager sleeps (grace delays, drain polls) take no real time."""
    try:
        from tre_sm.ops import sleep_primitive
    except ImportError:  # controller-only environment
        return
    monkeypatch.setattr(sleep_primitive, "DEFAULT_CLOCK", _VirtualClock())

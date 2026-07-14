from __future__ import annotations

import time
from typing import Callable, Protocol

from tre_common import rediskeys
from tre_sm.state.operations import OperationHandle


class SafetyRedis(Protocol):
    def get(self, key: str): ...


class PressureSource(Protocol):
    def node_pressure_reasons(self) -> dict[str, list[str]]: ...


class ControllerNotPaused(RuntimeError):
    pass


class PressureWaitTimeout(RuntimeError):
    pass


class ClusterSafetyGate:
    def __init__(
        self,
        redis_client: SafetyRedis,
        pressure_source: PressureSource,
        *,
        clear_hysteresis_s: float = 60.0,
        pressure_timeout_s: float = 3600.0,
        poll_interval_s: float = 5.0,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._redis = redis_client
        self._pressure_source = pressure_source
        self._clear_hysteresis_s = clear_hysteresis_s
        self._pressure_timeout_s = pressure_timeout_s
        self._poll_interval_s = poll_interval_s
        self._monotonic = monotonic
        self._sleep = sleep

    def assert_controller_observe(self) -> None:
        raw = self._redis.get(rediskeys.CONTROLLER_MODE_KEY)
        mode = "active" if raw is None else _text(raw)
        if mode != "observe":
            raise ControllerNotPaused(
                f"controller must be observe before fleet repair, got {mode}"
            )

    def wait_until_healthy(self, operation: OperationHandle) -> None:
        """Pause during pressure and require a continuous clear hysteresis window."""
        self.assert_controller_observe()
        started = self._monotonic()
        clear_since: float | None = None
        saw_pressure = False
        last_report: tuple[tuple[str, tuple[str, ...]], ...] | None = None
        while True:
            operation.assert_active()
            self.assert_controller_observe()
            reasons = self._pressure_source.node_pressure_reasons()
            now = self._monotonic()
            if reasons:
                saw_pressure = True
                clear_since = None
                report = tuple(
                    (node, tuple(values)) for node, values in sorted(reasons.items())
                )
                if report != last_report:
                    operation.advance(
                        "waiting_node_pressure",
                        details={"node_pressure": reasons},
                    )
                    last_report = report
            else:
                if not saw_pressure:
                    operation.advance("cluster_healthy")
                    return
                if clear_since is None:
                    clear_since = now
                    operation.advance("pressure_clear_hysteresis")
                if now - clear_since >= self._clear_hysteresis_s:
                    operation.advance("cluster_healthy")
                    return
            if now - started >= self._pressure_timeout_s:
                raise PressureWaitTimeout(
                    f"node pressure did not clear within {self._pressure_timeout_s}s"
                )
            self._sleep(self._poll_interval_s)


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)

from __future__ import annotations

import json
import time
from typing import Callable, Protocol

from tre_common import rediskeys
from tre_sm.state.actuation import SmActuation
from tre_sm.state.operations import OperationHandle


class SafetyRedis(Protocol):
    def get(self, key: str): ...
    def set(self, key: str, value: str): ...
    def delete(self, *keys: str): ...


class PressureSource(Protocol):
    def node_pressure_reasons(self) -> dict[str, list[str]]: ...


class MaintenanceLockLost(RuntimeError):
    """The SM maintenance lock of a running fleet repair was cleared or taken
    over (an operator deleted ``tre:v2:sm:maintenance`` to abort it)."""


class PressureWaitTimeout(RuntimeError):
    pass


class NodePressureActive(RuntimeError):
    pass


class ClusterSafetyGate:
    """Cluster-level guards of the service-manager: node pressure, the SM
    maintenance lock held by a fleet repair, and the SM actuation switch.

    The fleet repair used to require (and the supervisor used to force) the
    controller mode ``observe`` as its lock; it now holds its own maintenance
    lock ``tre:v2:sm:maintenance`` and never writes the controller mode (user
    decision 2026-09-28). While it runs, the repair also holds the SM writer
    lock, so every other SM write (controller, APA, operator) gets a retriable
    409 meanwhile."""

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
        actuation: SmActuation | None = None,
        wall_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    ) -> None:
        self._redis = redis_client
        self._pressure_source = pressure_source
        self._clear_hysteresis_s = clear_hysteresis_s
        self._pressure_timeout_s = pressure_timeout_s
        self._poll_interval_s = poll_interval_s
        self._monotonic = monotonic
        self._sleep = sleep
        self._wall_ms = wall_ms
        self._actuation = actuation if actuation is not None else SmActuation(redis_client)

    # ------------------------------------------------------ maintenance lock
    def acquire_maintenance(self, operation_id: str, *, kind: str, owner: str = "") -> None:
        """Take the SM maintenance lock for ``operation_id`` (the caller already
        holds the SM writer lock, so no other SM operation runs; a lock left by
        a dead SM is taken over)."""
        self._redis.set(
            rediskeys.SM_MAINTENANCE_KEY,
            json.dumps(
                {"operation_id": operation_id, "kind": kind, "owner": owner, "since_ms": self._wall_ms()},
                sort_keys=True,
            ),
        )

    def maintenance(self) -> dict | None:
        raw = self._redis.get(rediskeys.SM_MAINTENANCE_KEY)
        if raw is None:
            return None
        try:
            value = json.loads(_text(raw))
        except ValueError:
            return {"invalid": _text(raw)}
        return value if isinstance(value, dict) else {"invalid": _text(raw)}

    def assert_maintenance_held(self, operation_id: str) -> None:
        holder = (self.maintenance() or {}).get("operation_id")
        if holder != operation_id:
            raise MaintenanceLockLost(
                f"SM maintenance lock of {operation_id} lost (holder: {holder}); fleet repair aborted"
            )

    def release_maintenance(self, operation_id: str) -> None:
        if (self.maintenance() or {}).get("operation_id") == operation_id:
            self._redis.delete(rediskeys.SM_MAINTENANCE_KEY)

    # ------------------------------------------------------- actuation switch
    def actuation_mode(self) -> str:
        return self._actuation.mode()

    def actuation_state(self) -> dict:
        mode, source = self._actuation.resolve()
        return {"mode": mode, "source": source, "suppressed": self._actuation.recent_suppressed()[:20]}

    def record_suppressed(self, action: str, detail: dict) -> bool:
        return self._actuation.record_suppressed(action, detail)

    # ---------------------------------------------------------- node pressure
    def assert_no_pressure(self) -> None:
        reasons = self._pressure_source.node_pressure_reasons()
        if reasons:
            raise NodePressureActive(f"node pressure blocks cold start: {reasons}")

    def wait_until_healthy(self, operation: OperationHandle) -> None:
        """Pause during pressure and require a continuous clear hysteresis window.
        Aborts (MaintenanceLockLost) once the operation no longer holds the SM
        maintenance lock."""
        operation_id = operation.operation_id
        self.assert_maintenance_held(operation_id)
        started = self._monotonic()
        clear_since: float | None = None
        saw_pressure = False
        last_report: tuple[tuple[str, tuple[str, ...]], ...] | None = None
        while True:
            operation.assert_active()
            self.assert_maintenance_held(operation_id)
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

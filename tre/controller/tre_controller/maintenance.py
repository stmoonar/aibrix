"""SM maintenance lock as a SafeScale pause (P2-3, 2026-09-28).

The service-manager holds ``tre:v2:sm:maintenance`` (JSON ``{operation_id, kind,
owner, since_ms}``) for the whole run of a fleet repair; every other SM write gets
a retriable 409 meanwhile and the fleet changes under the controller's feet. A
SafeScale probe judged across such a period decides on a window that does not
describe its hide, so the controller treats the lock as a SafeScale pause:

* while the key is present no probe is started (planner loops) and every open
  (``probing``) probe is rolled back, reason ``sm_maintenance``;
* a probe whose observation window overlaps a maintenance period seen at any
  time (by any loop: the watch is shared and remembers every period it saw,
  ``since_ms`` .. last seen) is rolled back too, even when the key is gone by the
  time the SafeScale loop looks - a repair that began and ended between two
  SafeScale ticks but was seen by a planner tick still counts;
* a Redis read error blocks starting a probe (fail-closed, reason
  ``sm_maintenance_unreadable``) but does not roll back open probes.

A period that began and ended between two reads of every loop is not seen.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from tre_common.rediskeys import SM_MAINTENANCE_KEY

LOG = logging.getLogger("tre_controller.maintenance")

#: Probe-start block reasons (planner event ``safescale_probe_skipped:<model>:<reason>``).
MAINTENANCE_REASON = "sm_maintenance"
UNREADABLE_REASON = "sm_maintenance_unreadable"


@dataclass(frozen=True)
class MaintenancePeriod:
    key: str
    operation_id: str | None
    kind: str | None
    since_ms: int
    last_seen_ms: int


@dataclass(frozen=True)
class MaintenanceStatus:
    present: bool
    #: The read failed (``present`` is then False: unknown).
    error: str | None = None
    period: MaintenancePeriod | None = None


class MaintenanceWatch:
    def __init__(
        self,
        redis_client: Any,
        *,
        clock_ms: Callable[[], int] | None = None,
        max_periods: int = 32,
    ) -> None:
        self._redis = redis_client
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._max_periods = max(1, int(max_periods))
        #: period key -> period, oldest first (bounded).
        self._periods: dict[str, MaintenancePeriod] = {}
        self._present_key: str | None = None

    def poll(self) -> MaintenanceStatus:
        """Read the lock now and remember the period it belongs to."""
        now_ms = int(self._clock_ms())
        try:
            raw = self._redis.get(SM_MAINTENANCE_KEY)
        except Exception as exc:  # noqa: BLE001 - fail-closed for probe starts
            return MaintenanceStatus(present=False, error=f"{type(exc).__name__}: {exc}")
        if raw is None:
            if self._present_key is not None:
                LOG.info(json.dumps({"event": "sm_maintenance_cleared", "period": self._present_key}))
            self._present_key = None
            return MaintenanceStatus(present=False)
        period = self._record(raw, now_ms)
        return MaintenanceStatus(present=True, period=period)

    def probe_block_reason(self) -> str | None:
        """Why no SafeScale probe may start now (None = none)."""
        status = self.poll()
        if status.error is not None:
            return UNREADABLE_REASON
        return MAINTENANCE_REASON if status.present else None

    def overlapping(self, start_ms: int) -> MaintenancePeriod | None:
        """A remembered period that was still going at or after ``start_ms`` (the
        probe's start), i.e. overlaps its observation window up to now."""
        for period in self._periods.values():
            if period.last_seen_ms >= int(start_ms):
                return period
        return None

    def periods(self) -> tuple[MaintenancePeriod, ...]:
        return tuple(self._periods.values())

    def _record(self, raw: Any, now_ms: int) -> MaintenancePeriod:
        text = raw.decode() if isinstance(raw, bytes) else str(raw)
        try:
            body = json.loads(text)
        except (TypeError, ValueError):
            body = None
        body = body if isinstance(body, dict) else {}
        operation_id = body.get("operation_id")
        operation_id = str(operation_id) if operation_id else None
        since_ms = _int_or_none(body.get("since_ms"))
        key = operation_id or (f"since:{since_ms}" if since_ms is not None else f"raw:{text[:64]}")
        previous = self._periods.pop(key, None)
        if since_ms is None:
            since_ms = previous.since_ms if previous is not None else now_ms
        period = MaintenancePeriod(
            key=key,
            operation_id=operation_id,
            kind=str(body["kind"]) if body.get("kind") else None,
            since_ms=min(since_ms, now_ms),
            last_seen_ms=now_ms,
        )
        self._periods[key] = period
        while len(self._periods) > self._max_periods:
            self._periods.pop(next(iter(self._periods)))
        if self._present_key != key:
            LOG.warning(
                json.dumps(
                    {"event": "sm_maintenance_seen", "operation_id": operation_id, "kind": period.kind,
                     "since_ms": period.since_ms, "safescale": "paused"},
                    sort_keys=True,
                )
            )
        self._present_key = key
        return period


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

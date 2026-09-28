"""Service-manager actuation switch (user decision 2026-09-28).

``tre:v2:sm:actuation`` = ``active`` | ``observe`` decides whether the SM
SUPERVISOR may change capacity or recreate / delete workloads on its own
(B7 recreate, drift -> fleet repair, stale repair recovery, reaping rejected
Deployments, sleeping residents to admit a Pod nobody asked for). In observe
those only log and record what they would have done. The SM HTTP write API is
never gated by it: the APA arm (AIBrix's own autoscaler) and operators drive
the SM directly.

The switch is INDEPENDENT of the controller mode ``tre:v2:controller:mode``:
both experiment arms run the SM in active (symmetric self-heal) while the APA
arm keeps the controller in observe. Resolution: the SM key; absent (or not a
mode) = observe -- never derived from the controller mode, so deployments must
set it explicitly. A Redis error keeps the last known mode; never known =
observe (fail-closed). Reads are cached for a short TTL.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from typing import Any, Callable

from tre_common import rediskeys
from tre_common.run_mode import OBSERVE, parse_mode

LOG = logging.getLogger(__name__)

#: A suppressed action with the same detail is recorded again only after this.
SUPPRESSED_REPEAT_S = 300.0


class SmActuation:
    def __init__(
        self,
        redis_client: Any,
        *,
        ttl_s: float = 1.0,
        monotonic: Callable[[], float] = time.monotonic,
        wall_ms: Callable[[], int] = lambda: int(time.time() * 1000),
        recent_max: int = 50,
    ) -> None:
        self._redis = redis_client
        self._ttl_s = max(0.0, float(ttl_s))
        self._monotonic = monotonic
        self._wall_ms = wall_ms
        self._lock = threading.Lock()
        self._cached: tuple[str, str] = (OBSERVE, "unknown")
        self._expires_at = float("-inf")
        #: Last successfully read (mode, source); None = never read.
        self._last_known: tuple[str, str] | None = None
        #: action -> (detail signature, monotonic time recorded).
        self._last_recorded: dict[str, tuple[str, float]] = {}
        self._recent: deque[dict] = deque(maxlen=recent_max)

    def mode(self) -> str:
        return self.resolve()[0]

    def is_observe(self) -> bool:
        return self.mode() == OBSERVE

    def resolve(self) -> tuple[str, str]:
        """(mode, source): source is ``sm`` (the SM key), ``default`` (key absent
        or not a mode: observe), ``last_known`` (Redis error) or ``fail_closed``
        (never read)."""
        now = self._monotonic()
        with self._lock:
            if now < self._expires_at:
                return self._cached
        resolved = self._read()
        with self._lock:
            self._cached = resolved
            self._expires_at = now + self._ttl_s
        return resolved

    def _read(self) -> tuple[str, str]:
        try:
            own = parse_mode(self._redis.get(rediskeys.SM_ACTUATION_KEY))
            resolved = (own, "sm") if own is not None else (OBSERVE, "default")
        except Exception as exc:  # noqa: BLE001 - keep the last known mode, fail closed
            with self._lock:
                last = self._last_known
            if last is None:
                LOG.warning("SM actuation unreadable and never read (%r): observe (fail-closed)", exc)
                return (OBSERVE, "fail_closed")
            return (last[0], "last_known")
        with self._lock:
            previous = self._last_known
            self._last_known = resolved
        if previous is None or previous[0] != resolved[0]:
            LOG.warning(
                json.dumps({"event": "sm_actuation_mode", "mode": resolved[0], "source": resolved[1],
                            "previous": previous[0] if previous else None}, sort_keys=True)
            )
        return resolved

    def record_suppressed(self, action: str, detail: dict) -> bool:
        """Log + record (Redis list, capped; in-memory recent list) a supervisor
        action NOT taken because the actuation is observe. The same action with
        the same detail is recorded once per ``SUPPRESSED_REPEAT_S`` (the
        supervisor re-evaluates every few seconds). Returns True if recorded."""
        signature = json.dumps(detail, sort_keys=True, default=str)
        now = self._monotonic()
        with self._lock:
            last = self._last_recorded.get(action)
            if last is not None and last[0] == signature and now - last[1] < SUPPRESSED_REPEAT_S:
                return False
            self._last_recorded[action] = (signature, now)
            entry = {"ts_ms": self._wall_ms(), "action": action, "detail": detail}
            self._recent.appendleft(entry)
        payload = json.dumps(entry, sort_keys=True, default=str)
        LOG.warning(json.dumps({"event": "sm_supervisor_action_suppressed", **entry}, sort_keys=True, default=str))
        try:
            self._redis.lpush(rediskeys.SM_SUPPRESSED_ACTIONS_KEY, payload)
            self._redis.ltrim(rediskeys.SM_SUPPRESSED_ACTIONS_KEY, 0, rediskeys.SM_SUPPRESSED_ACTIONS_MAX - 1)
        except Exception:  # noqa: BLE001 - the log line above is the record of last resort
            LOG.warning("recording the suppressed action %s in Redis failed", action)
        return True

    def recent_suppressed(self) -> list[dict]:
        with self._lock:
            return list(self._recent)

"""Controller run-mode gate.

The console (or an operator) sets ``tre:v2:controller:mode`` (together with the SM
actuation switch ``tre:v2:sm:actuation``, see ``tre_common.run_mode``). ``observe``
means the controller computes and records only - no scaling side effects
(user decision 2026-09-28, tre/docs/design/20260928-observe-mode-semantics.md):

* decisions keep computing and publishing (the UI stays live);
* the ActionQueue drops every re-plannable action (scale, hide, defrag) and a
  hide is re-checked immediately before its SM call (B8 race);
* no SafeScale probe is started or preempted; every open probe is rolled back:
  its held one-shot actions are dropped and the ONLY action taken is the unhide
  of its probe pods (undoing the controller's own half-action), resolved as a
  rollback ``observe_entered``;
* a transfer / SafeScale commit already running re-checks the mode before each
  capacity-changing step and stops (a donor already slept is recorded, its
  receiver is not woken).

Reads are cached for a short TTL so the 0.1s drain poll never hammers Redis;
``is_observe_fresh`` bypasses the cache (used right before capacity-changing SM
calls). Fail-closed: a Redis error keeps the last successfully read mode, and
before the first successful read (or with the key absent / not a mode) the
controller is in observe.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from tre_common.rediskeys import CONTROLLER_MODE_KEY
from tre_common.run_mode import OBSERVE, effective_mode, parse_mode

LOG = logging.getLogger(__name__)


class ObserveModeGate:
    def __init__(
        self,
        redis_client: Any,
        *,
        ttl_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._redis = redis_client
        self._ttl_s = max(0.0, float(ttl_s))
        self._clock = clock
        self._cached = True
        self._expires_at = float("-inf")
        #: Last successfully read mode ("active" / "observe"); None = never read.
        self._last_known: str | None = None

    def is_observe(self) -> bool:
        now = self._clock()
        if now >= self._expires_at:
            self._cached = self._read()
            self._expires_at = now + self._ttl_s
        return self._cached

    def is_observe_fresh(self) -> bool:
        """Uncached read (refreshes the cache): right before a capacity-changing
        SM call, so a switch to observe is honoured without the TTL lag."""
        self._cached = self._read()
        self._expires_at = self._clock() + self._ttl_s
        return self._cached

    def last_known_mode(self) -> str | None:
        return self._last_known

    def _read(self) -> bool:
        try:
            value = self._redis.get(CONTROLLER_MODE_KEY)
        except Exception as exc:  # noqa: BLE001 - keep the last known mode, fail closed
            if self._last_known is None:
                LOG.warning("controller mode unreadable and never read (%r): observe (fail-closed)", exc)
                return True
            return self._last_known == OBSERVE
        mode = effective_mode(parse_mode(value))
        if mode != self._last_known:
            LOG.info("controller mode %s -> %s", self._last_known, mode)
        self._last_known = mode
        return mode == OBSERVE

"""Controller run-mode gate.

The console (or an operator) can set ``tre:v2:controller:mode`` to ``observe`` to PAUSE
actuation without stopping the controller: decisions keep computing and publishing (so the
UI stays live), but the ActionQueue stops dispatching to the service-manager -- no scale,
hide/unhide, defrag, or safescale probe reaches the cluster. Setting it back to ``active``
(or clearing it) resumes. Reads are cached for a short TTL so the 0.1s drain poll never
hammers Redis, and any Redis error fails safe to ``active`` (never silently freezes control).
"""
from __future__ import annotations

import time
from typing import Any, Callable

CONTROLLER_MODE_KEY = "tre:v2:controller:mode"


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
        self._cached = False
        self._expires_at = float("-inf")

    def is_observe(self) -> bool:
        now = self._clock()
        if now >= self._expires_at:
            self._cached = self._read()
            self._expires_at = now + self._ttl_s
        return self._cached

    def _read(self) -> bool:
        try:
            value = self._redis.get(CONTROLLER_MODE_KEY)
        except Exception:  # noqa: BLE001 - a Redis hiccup must not pause the controller
            return False
        if isinstance(value, bytes):
            value = value.decode("utf-8", "replace")
        return value == "observe"

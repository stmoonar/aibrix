"""O1 same-clock check: gateway doc stamps vs the controller clock (review P2-3).

The O1 breakpoint window compares two clocks: routable-change times (the controller's
clock: ActionQueue SM-call returns, ``ClusterView.fetched_ms``) and the gateway's
observation times (``scraped_ms``, the gateway's clock). It assumes they are the same
clock (NTP-synchronised nodes). Node clocks have drifted before (75 ran ~160 s ahead of
76), so the controller checks at start and then periodically.

Direct measurement (review P2-2): every gateway doc carries ``written_ms`` (the
gateway's wall clock when it wrote the doc). The check watches the instant-doc key of
the pod whose newest doc is the newest (a pod the gateway still writes) until a new doc
appears (polling every ``poll_ms``): the write happened between the last poll that saw
the old doc and the first that sees the new one, so ``offset = written_ms - midpoint``
is the gateway clock minus the controller clock to about ``poll_ms / 2`` plus a redis
round trip, independent of the write phase.

Round 4: no measurement, no verdict. The former fallback (``now`` minus the newest doc
stamp, bounded by ``2 x period + tolerance``) is gone: it only guarded against signal
lag with a fixed span, and it read leftover docs as a stalled clock - with every model
down, the old docs suspended O1. Gateway liveness is the sampler's business
(``last_hist_ms``, stale snapshots).

On a violation O1 is suspended (``SignalState.suspend_breakpoint_window``: whole
windows, the ADR-0013 onset guard) until ``resume_after`` consecutive good checks.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Optional

from tre_common.rediskeys import inst_key, pods_key

LOG = logging.getLogger("tre_controller.gateway_clock")


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _written_ms(raw: Any) -> Optional[int]:
    try:
        doc = json.loads(_text(raw))
    except (TypeError, ValueError):
        return None
    value = doc.get("written_ms") if isinstance(doc, dict) else None
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _newest_written(redis_client: Any, key: str) -> Optional[int]:
    rows = redis_client.zrange(key, -1, -1, withscores=True) or ()
    for member, _score in rows:
        return _written_ms(member)
    return None


def _newest_doc(redis_client: Any, key: str) -> tuple[Optional[int], Optional[float]]:
    """(``written_ms``, stamp) of the newest doc of ``key``; (None, None) without one."""
    rows = redis_client.zrange(key, -1, -1, withscores=True) or ()
    for member, score in rows:
        return _written_ms(member), float(score)
    return None, None


def written_probe_key(redis_client: Any, models: Iterable[str]) -> Optional[str]:
    """The instant-doc key whose newest doc is the newest of all (and carries
    ``written_ms``): a pod the gateway still writes, not one whose leftover docs are
    the first in sorted order. None: no doc with ``written_ms``."""
    best: tuple[float, str] | None = None
    for model in models:
        for pod in sorted(_text(item) for item in (redis_client.smembers(pods_key(model)) or ())):
            key = inst_key(pod)
            written, stamp = _newest_doc(redis_client, key)
            if written is None or stamp is None:
                continue
            if best is None or stamp > best[0]:
                best = (stamp, key)
    return best[1] if best is not None else None


def measure_written_offset_ms(
    redis_client: Any,
    key: str,
    *,
    clock_ms: Callable[[], int],
    sleep_s: Callable[[float], None],
    poll_ms: int = 250,
    max_wait_ms: int = 12_000,
) -> Optional[int]:
    """Gateway clock minus controller clock (ms) from the next doc written on ``key``;
    None when no new doc appears within ``max_wait_ms``.

    The write is bracketed by the last poll that saw the old doc (``before``) and the
    first that sees the new one (``now``). When that bracket is wider than
    ``2 * poll_ms`` (the polling thread stalled: GIL, a slow redis call, a suspended
    process) the midpoint no longer dates the write to ``poll_ms / 2``, so the
    measurement is discarded: the new doc becomes the reference and the next write is
    waited for; None if none comes before the deadline (the caller then falls back to
    no verdict)."""
    last = _newest_written(redis_client, key)
    before = clock_ms()
    deadline = before + int(max_wait_ms)
    while clock_ms() < deadline:
        sleep_s(poll_ms / 1000.0)
        now = clock_ms()
        current = _newest_written(redis_client, key)
        if current is not None and current != last:
            if now - before > 2 * int(poll_ms):
                LOG.info(json.dumps({"event": "gateway_clock_measurement_discarded",
                                     "bracket_ms": int(now - before), "poll_ms": int(poll_ms)},
                                    sort_keys=True))
                last = current
                before = now
                continue
            return int(current - (before + now) / 2)
        before = now
    return None


@dataclass(frozen=True)
class ClockReading:
    ok: bool
    lag_ms: Optional[int]
    reason: Optional[str] = None


class GatewayClockMonitor:
    """Suspends / resumes O1 on ``signal_state`` from direct ``written_ms`` readings."""

    def __init__(
        self,
        redis_client: Any,
        models: Iterable[str],
        signal_state: Any,
        *,
        period_ms: int,
        tolerance_ms: int,
        resume_after: int = 3,
        clock_ms: Callable[[], int] | None = None,
        sleep_s: Callable[[float], None] | None = None,
        poll_ms: int = 250,
    ) -> None:
        self._redis = redis_client
        self._models = tuple(models)
        self._state = signal_state
        self.period_ms = int(period_ms)
        self.tolerance_ms = int(tolerance_ms)
        self.resume_after = max(1, int(resume_after))
        self._clock = clock_ms or (lambda: int(time.time() * 1000))
        self._sleep = sleep_s or time.sleep
        self.poll_ms = int(poll_ms)
        self._good = 0

    def _reading(self) -> ClockReading:
        key = written_probe_key(self._redis, self._models)
        if key is None:
            return ClockReading(True, None, "no_written_ms")  # nothing to measure: no verdict
        offset = measure_written_offset_ms(
            self._redis, key, clock_ms=self._clock, sleep_s=self._sleep, poll_ms=self.poll_ms,
            max_wait_ms=self.period_ms + 2_000,
        )
        if offset is None:
            # No new doc (models down, a gateway pause): no measurement, no verdict.
            return ClockReading(True, None, "no_measurement")
        if offset > self.tolerance_ms:
            return ClockReading(False, offset, "gateway_ahead")
        if offset < -self.tolerance_ms:
            return ClockReading(False, offset, "gateway_behind")
        return ClockReading(True, offset)

    def check(self) -> ClockReading:
        try:
            reading = self._reading()
        except Exception as exc:  # noqa: BLE001 - no verdict on a read error
            LOG.warning("gateway clock check: redis read failed: %r", exc)
            return ClockReading(True, None, "read_error")
        suspended = getattr(self._state, "breakpoint_window_suspended", None)
        if not reading.ok:
            self._good = 0
            if suspended is None:
                self._state.suspend_breakpoint_window(f"gateway_clock:{reading.reason}")
                LOG.error(json.dumps({"event": "breakpoint_window_suspended", "reason": reading.reason,
                                      "lag_ms": reading.lag_ms, "tolerance_ms": self.tolerance_ms},
                                     sort_keys=True))
        elif suspended is not None and reading.reason is None:
            self._good += 1
            if self._good >= self.resume_after:
                self._state.resume_breakpoint_window()
                self._good = 0
                LOG.warning(json.dumps({"event": "breakpoint_window_resumed", "lag_ms": reading.lag_ms},
                                       sort_keys=True))
        return reading


async def gateway_clock_task(
    monitor: GatewayClockMonitor,
    interval_s: float,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Check at start, then every ``interval_s`` (the redis read runs in a thread)."""
    while True:
        await asyncio.to_thread(monitor.check)
        await sleep(interval_s)

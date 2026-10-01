"""O1 same-clock check: gateway doc stamps vs the controller clock (review P2-3).

The O1 breakpoint window compares two clocks: routable-change times (the controller's
clock: ActionQueue SM-call returns, ``ClusterView.fetched_ms``) and the 10 s grid of the
gateway's doc stamps (``roundT``, the gateway's clock rounded to the boundary). It
assumes they are the same clock (NTP-synchronised nodes). Node clocks have drifted before
(75 ran ~160 s ahead of 76), so the controller checks at start and then periodically:

* ``lag = now - newest instant stamp`` over the registry models' pods. With one clock the
  gateway writes boundary ``B`` at ``B + phase`` (its ticker phase, 0..one period) and
  the newest stamp is at most about one period old, so ``0 <= lag < 2 * period``;
* ``lag < -tolerance`` (stamps from the future: the gateway runs ahead) or
  ``lag > 2 * period + tolerance`` (stamps from the past: it runs behind, or stopped
  writing) is a violation.

Direct measurement (review P2-2): gateway images from 2026-10-01 add ``written_ms`` (the
gateway's wall clock when it wrote the doc) to every doc. The check watches one pod's
newest instant doc until a new one appears (polling every ``poll_ms``): the write
happened between the last poll that saw the old doc and the first that sees the new one,
so ``offset = written_ms - midpoint`` is the gateway clock minus the controller clock to
about ``poll_ms / 2`` plus a redis round trip, independent of the write phase.

Fallback (older gateways, no ``written_ms``): the stamp-lag bounds below. The write phase
is unknown, so an offset is only visible once it exceeds the phase window - a gateway
ahead by more than ``phase + tolerance`` or behind by more than
``2 * period - phase + tolerance``.

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


def newest_instant_stamp_ms(redis_client: Any, models: Iterable[str]) -> Optional[int]:
    """The newest gateway instant-doc stamp (ms) over ``models``' pods; None without any."""
    newest: Optional[int] = None
    for model in models:
        for pod in redis_client.smembers(pods_key(model)) or ():
            rows = redis_client.zrange(inst_key(_text(pod)), -1, -1, withscores=True) or ()
            for _member, score in rows:
                stamp = int(float(score))
                newest = stamp if newest is None or stamp > newest else newest
    return newest


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


def written_probe_key(redis_client: Any, models: Iterable[str]) -> Optional[str]:
    """An instant-doc key whose newest doc carries ``written_ms`` (None: old gateway)."""
    for model in models:
        for pod in sorted(_text(item) for item in (redis_client.smembers(pods_key(model)) or ())):
            key = inst_key(pod)
            if _newest_written(redis_client, key) is not None:
                return key
    return None


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
    None when no new doc appears within ``max_wait_ms``."""
    last = _newest_written(redis_client, key)
    before = clock_ms()
    deadline = before + int(max_wait_ms)
    while clock_ms() < deadline:
        sleep_s(poll_ms / 1000.0)
        now = clock_ms()
        current = _newest_written(redis_client, key)
        if current is not None and current != last:
            return int(current - (before + now) / 2)
        before = now
    return None


@dataclass(frozen=True)
class ClockReading:
    ok: bool
    lag_ms: Optional[int]
    reason: Optional[str] = None


def evaluate_clock(now_ms: int, newest_ms: Optional[int], *, period_ms: int, tolerance_ms: int) -> ClockReading:
    if newest_ms is None:
        return ClockReading(True, None, "no_stamps")  # idle / nothing written: no verdict
    lag = int(now_ms) - int(newest_ms)
    if lag < -int(tolerance_ms):
        return ClockReading(False, lag, "gateway_ahead")
    if lag > 2 * int(period_ms) + int(tolerance_ms):
        return ClockReading(False, lag, "gateway_behind_or_stalled")
    return ClockReading(True, lag)


class GatewayClockMonitor:
    """Suspends / resumes O1 on ``signal_state`` from :func:`evaluate_clock` readings."""

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
        if key is not None:
            offset = measure_written_offset_ms(
                self._redis, key, clock_ms=self._clock, sleep_s=self._sleep, poll_ms=self.poll_ms,
                max_wait_ms=self.period_ms + 2_000,
            )
            if offset is not None:
                if offset > self.tolerance_ms:
                    return ClockReading(False, offset, "gateway_ahead")
                if offset < -self.tolerance_ms:
                    return ClockReading(False, offset, "gateway_behind")
                return ClockReading(True, offset)
        newest = newest_instant_stamp_ms(self._redis, self._models)
        return evaluate_clock(self._clock(), newest, period_ms=self.period_ms, tolerance_ms=self.tolerance_ms)

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

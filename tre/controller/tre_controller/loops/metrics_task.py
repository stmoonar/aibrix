"""The controller's only metrics fetcher; decision loops just read ``SnapshotBox``.

Two refresh modes (``TRE_METRICS_REFRESH_MODE``):

* ``phase_aligned`` (default, plan §6.9i / D8): :class:`PhaseAlignedSampler`. The gateway
  stamps its instant/histogram samples with a boundary of the
  ``SCRAPE_INTERVAL_MS`` = 10 s grid. The sampler sleeps until ``next boundary + offset``
  (``sleep(next_boundary + offset - now)``, not ``sleep(interval)``), reads the half-open
  window ``(boundary - window_ms, boundary]`` - exactly ``window_ms / 10 s`` instant ticks
  (3 for 30 s) and a token delta spanning the full window - and publishes an immutable
  snapshot whose ``ts_ms`` / ``window_end_ms`` is that boundary. A window missing any
  expected tick (the gateway has not written the boundary yet, or skipped one) is
  re-read every ``retry_ms`` until the next boundary; if it never completes it is
  *stale*: logged, the previous snapshot keeps being served, and after
  ``stale_hold_windows`` consecutive stale windows the served snapshot is marked
  ``stale`` so the decision loops stop acting on it. The fetch runs in a worker thread,
  so a slow redis read never blocks the decision loops.
* ``free_running`` (fallback): the old ``refresh; sleep(metrics_refresh_interval_s)`` loop
  with a sliding window ending at ``now``.

Decision cadence: snapshots change every 10 s in ``phase_aligned`` mode. The rescue loop
still wakes every ``TRE_RESCUE_INTERVAL_SECONDS`` (5 s) but a re-read of the same snapshot
advances neither the EMA nor the band dwell (both keyed by ``window_end_ms``), so the
effective decision cadence is 10 s, reached at most one rescue interval after publish.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Awaitable, Callable, Protocol

from tre_common.metrics_schema import MetricsSnapshot
from tre_common.rediskeys import SCRAPE_INTERVAL_MS

if False:  # annotations are strings (from __future__); avoids an import cycle
    from tre_controller.profiling import TickProfiler

LOG = logging.getLogger("tre_controller.metrics")

REFRESH_PHASE_ALIGNED = "phase_aligned"
REFRESH_FREE_RUNNING = "free_running"
REFRESH_MODES = (REFRESH_PHASE_ALIGNED, REFRESH_FREE_RUNNING)

DEFAULT_PHASE_OFFSET_MS = 2_000
DEFAULT_PHASE_RETRY_MS = 500
DEFAULT_STALE_HOLD_WINDOWS = 2
#: Every this many cycles an adapted (raised) offset is dropped back to the configured one
#: once, so the sampler re-learns a gateway write phase that moved earlier (restart).
DEFAULT_PHASE_RELEARN_CYCLES = 60


class SnapshotStore(Protocol):
    def read_snapshot(
        self, window_start_ms: int, window_end_ms: int, *, use_cache: bool = True
    ) -> MetricsSnapshot: ...


class MetricsTaskConfig(Protocol):
    metrics_window_ms: int
    metrics_window_mode: str
    monitor_interval_s: float
    metrics_refresh_interval_s: float


class SnapshotBox:
    def __init__(self, snapshot: MetricsSnapshot | None = None) -> None:
        self._snapshot = snapshot

    def get(self) -> MetricsSnapshot | None:
        return self._snapshot

    def set(self, snapshot: MetricsSnapshot) -> None:
        self._snapshot = snapshot


def freeze_snapshot(snapshot: MetricsSnapshot) -> MetricsSnapshot:
    """Read-only view of a snapshot: the dataclasses are frozen already, this also makes
    the ``models`` / ``per_pod`` mappings read-only, so no decision loop can alter the
    window another loop is about to read."""
    models = {
        name: replace(window, per_pod=MappingProxyType(dict(window.per_pod)))
        for name, window in snapshot.models.items()
    }
    return replace(snapshot, models=MappingProxyType(models))


# ----------------------------------------------------------------- phase arithmetic


def due_boundary(now_ms: int, *, period_ms: int, offset_ms: int) -> int:
    """The newest boundary whose read time (boundary + offset) is not in the future."""
    return (int(now_ms) - int(offset_ms)) // int(period_ms) * int(period_ms)


def phase_target(
    now_ms: int, *, period_ms: int, offset_ms: int, last_end_ms: int | None
) -> tuple[int, int]:
    """``(window_end_ms, wake_ms)`` for the next read: the first boundary whose read time
    ``boundary + offset`` is at or after ``now`` and that is newer than ``last_end_ms``.
    The caller sleeps ``wake_ms - now``."""
    period = int(period_ms)
    offset = int(offset_ms)
    end = -((offset - int(now_ms)) // period) * period  # ceil((now - offset) / period) * period
    if last_end_ms is not None and end <= last_end_ms:
        end = int(last_end_ms) + period
    return end, end + offset


@dataclass(frozen=True)
class WindowFreshness:
    fresh: bool
    ticks: tuple[int, ...]
    expected: tuple[int, ...]
    reason: str | None = None


def window_freshness(
    snapshot: MetricsSnapshot, *, window_end_ms: int, window_ms: int, period_ms: int = SCRAPE_INTERVAL_MS
) -> WindowFreshness:
    """Whether the window holds every gateway tick it should: ``(end - window, end]`` on
    the ``period_ms`` grid, i.e. 3 ticks for 30 s, the newest exactly at ``end``.

    Ticks are pooled over every pod of every model (the gateway writes all pods on one
    ticker), so a pod that started mid-window does not make the window stale. A window
    with no tick at all is reported fresh with reason ``no_ticks`` - nothing is serving,
    or the store does not report ticks; the sampler decides what that means.
    """
    end = int(window_end_ms)
    start = end - int(window_ms)
    ticks = tuple(
        sorted(
            {
                int(tick)
                for window in snapshot.models.values()
                for tick in getattr(window, "instant_ticks_ms", ())
                if start < int(tick) <= end
            }
        )
    )
    expected = tuple(range(end - (int(window_ms) // int(period_ms) - 1) * int(period_ms), end + 1, int(period_ms)))
    if not ticks:
        return WindowFreshness(True, ticks, expected, "no_ticks")
    missing = [tick for tick in expected if tick not in ticks]
    if missing:
        return WindowFreshness(False, ticks, expected, "missing_ticks:" + ",".join(str(t) for t in missing))
    return WindowFreshness(True, ticks, expected)


# ------------------------------------------------------------------- the sampler


@dataclass(frozen=True)
class SampleOutcome:
    kind: str  # "published" | "stale" | "early"
    window_end_ms: int
    attempts: int = 0
    missed_boundaries: int = 0
    reason: str | None = None
    offset_ms: int = 0


def _default_clock_ms() -> int:
    return int(time.time() * 1000)


class PhaseAlignedSampler:
    """Phase-aligned metrics sampler (module docstring). One instance per controller."""

    def __init__(
        self,
        store: SnapshotStore,
        snapshot_box: SnapshotBox,
        *,
        window_ms: int,
        period_ms: int = SCRAPE_INTERVAL_MS,
        offset_ms: int = DEFAULT_PHASE_OFFSET_MS,
        adapt: bool = True,
        retry_ms: int = DEFAULT_PHASE_RETRY_MS,
        stale_hold_windows: int = DEFAULT_STALE_HOLD_WINDOWS,
        relearn_cycles: int = DEFAULT_PHASE_RELEARN_CYCLES,
        clock_ms: Callable[[], int] = _default_clock_ms,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        fetch: Callable[[int, int], Awaitable[MetricsSnapshot]] | None = None,
        prof: "TickProfiler | None" = None,
    ) -> None:
        if period_ms <= 0 or window_ms <= 0:
            raise ValueError("period_ms and window_ms must be positive")
        if window_ms % period_ms:
            raise ValueError(
                f"window_ms={window_ms} must be a multiple of the gateway period {period_ms} ms"
            )
        if not 0 <= offset_ms < period_ms:
            raise ValueError(f"offset_ms must be in [0, {period_ms})")
        if retry_ms <= 0:
            raise ValueError("retry_ms must be positive")
        self._store = store
        self._box = snapshot_box
        self.window_ms = int(window_ms)
        self.period_ms = int(period_ms)
        self.base_offset_ms = int(offset_ms)
        self.offset_ms = int(offset_ms)
        self.adapt = bool(adapt)
        self.retry_ms = int(retry_ms)
        self.stale_hold_windows = max(0, int(stale_hold_windows))
        self.relearn_cycles = max(0, int(relearn_cycles))
        self._clock = clock_ms
        self._sleep = sleep
        self._fetch = fetch or self._threaded_fetch
        self._prof = prof
        self.last_end_ms: int | None = None
        self.consecutive_stale = 0
        self.cycles = 0
        self._saw_ticks = False

    async def _threaded_fetch(self, start_ms: int, end_ms: int) -> MetricsSnapshot:
        return await asyncio.to_thread(
            self._store.read_snapshot, start_ms, end_ms, use_cache=False, start_exclusive=True
        )

    async def run_forever(self) -> None:
        while True:
            await self.run_once()

    async def run_once(self) -> SampleOutcome:
        now = self._clock()
        if self.last_end_ms is not None and now + self.period_ms < self.last_end_ms:
            LOG.warning(
                "metrics_clock_regressed: now=%d is more than one period before the last "
                "window_end=%d; restarting the phase schedule", now, self.last_end_ms,
            )
            self.last_end_ms = None
        end, wake = phase_target(
            now, period_ms=self.period_ms, offset_ms=self.offset_ms, last_end_ms=self.last_end_ms
        )
        delay_ms = min(max(0, wake - now), self.period_ms + self.offset_ms)
        if delay_ms > 0:
            await self._sleep(delay_ms / 1000.0)
        now = self._clock()
        due = due_boundary(now, period_ms=self.period_ms, offset_ms=self.offset_ms)
        if due < end:
            # Woke before the read time (short sleep or the wall clock stepped back): do
            # not read a window whose newest tick may not exist yet; reschedule.
            return SampleOutcome("early", end, offset_ms=self.offset_ms)
        missed = (due - end) // self.period_ms
        if missed:
            LOG.warning(
                "metrics_boundary_missed: woke at %d, %d boundar%s after the target %d "
                "(late wake-up or clock step); reading the newest window %d",
                now, missed, "y" if missed == 1 else "ies", end, due,
            )
            end = due
        outcome = await self._sample(end, missed)
        self.last_end_ms = end
        self.cycles += 1
        if (
            self.adapt
            and self.relearn_cycles
            and self.cycles % self.relearn_cycles == 0
            and self.offset_ms != self.base_offset_ms
        ):
            LOG.info("metrics_phase_relearn: offset %d -> %d ms", self.offset_ms, self.base_offset_ms)
            self.offset_ms = self.base_offset_ms
        return outcome

    async def _sample(self, end: int, missed: int) -> SampleOutcome:
        start = end - self.window_ms
        attempts = 0
        reason: str | None = None
        while True:
            attempts += 1
            started = self._clock()
            fetch_t0 = time.perf_counter_ns()
            snapshot: MetricsSnapshot | None
            try:
                snapshot = await self._fetch(start, end)
            except Exception as exc:  # noqa: BLE001 - degrade through stale snapshots.
                snapshot = None
                reason = f"error:{exc}"
            fetch_ns = time.perf_counter_ns() - fetch_t0
            if snapshot is not None:
                verdict = window_freshness(
                    snapshot, window_end_ms=end, window_ms=self.window_ms, period_ms=self.period_ms
                )
                fresh = verdict.fresh
                reason = verdict.reason
                if verdict.reason == "no_ticks" and self._saw_ticks:
                    # Ticks were flowing and now none: the gateway went quiet, not idle.
                    fresh, reason = False, "no_ticks_after_ticks"
                if verdict.ticks:
                    self._saw_ticks = True
            else:
                fresh = False
            self._record_poll(fetch_ns, snapshot, stale=not fresh)
            if fresh:
                break
            if self._clock() + self.retry_ms >= end + self.period_ms:
                break  # the next boundary is due: give this window up
            await self._sleep(self.retry_ms / 1000.0)

        if not fresh:
            return self._mark_stale(end, attempts, missed, reason)

        if attempts > 1 and self.adapt:
            learned = min(max(self.base_offset_ms, started - end), self.period_ms - self.retry_ms)
            if learned > self.offset_ms:
                LOG.info(
                    "metrics_phase_adapted: window %d complete only at +%d ms (attempt %d); "
                    "offset %d -> %d ms", end, started - end, attempts, self.offset_ms, learned,
                )
                self.offset_ms = learned
        self.consecutive_stale = 0
        assert snapshot is not None
        published = freeze_snapshot(replace(snapshot, ts_ms=end, stale=False))
        self._box.set(published)
        return SampleOutcome("published", end, attempts, missed, reason, self.offset_ms)

    def _mark_stale(self, end: int, attempts: int, missed: int, reason: str | None) -> SampleOutcome:
        self.consecutive_stale += 1
        previous = self._box.get()
        if previous is None:
            self._box.set(MetricsSnapshot(ts_ms=end, models=MappingProxyType({}), stale=True))
            action = "no previous snapshot; serving an empty stale one"
        elif self.consecutive_stale > self.stale_hold_windows:
            if not previous.stale:
                self._box.set(replace(previous, stale=True))
            action = f"previous window {previous.ts_ms} now marked stale"
        else:
            action = f"keep serving previous window {previous.ts_ms}"
        LOG.warning(
            "metrics_window_stale: window_end=%d reason=%s attempts=%d consecutive=%d; %s",
            end, reason, attempts, self.consecutive_stale, action,
        )
        return SampleOutcome("stale", end, attempts, missed, reason, self.offset_ms)

    def _record_poll(self, fetch_ns: int, snapshot: MetricsSnapshot | None, *, stale: bool) -> None:
        if self._prof is None:
            return
        self._prof.record(
            {
                "kind": "poll",
                "ts_ms": self._prof.now_ms(),
                "fetch_ns": fetch_ns,
                "n_models": len(snapshot.models) if snapshot is not None else 0,
                "stale": bool(stale),
            }
        )


@dataclass(frozen=True)
class MetricsRefreshResult:
    window_start_ms: int
    window_end_ms: int
    snapshot: MetricsSnapshot
    stale: bool
    error: str | None = None


def refresh_metrics_once(
    store: SnapshotStore,
    snapshot_box: SnapshotBox,
    *,
    now_ms: int,
    window_ms: int,
    window_mode: str = "tumbling",
    prof: "TickProfiler | None" = None,
) -> MetricsRefreshResult:
    # window_mode defaults to "tumbling" here so existing callers/tests keep the old
    # behaviour; the live controller passes cfg.metrics_window_mode (default "sliding").
    sliding = window_mode == "sliding"
    if sliding:
        window_start_ms, window_end_ms = _sliding_window(now_ms, window_ms)
    else:
        window_start_ms, window_end_ms = _last_complete_window(now_ms, window_ms)
    # Tumbling calls read_snapshot with its original signature (no use_cache) so existing
    # SnapshotStore fakes keep working; only sliding opts out of the per-window cache
    # (every sliding window is unique -> the cache never hits and would grow, S1.1).
    _fetch_t0 = time.perf_counter_ns() if prof is not None else 0
    try:
        if sliding:
            snapshot = store.read_snapshot(window_start_ms, window_end_ms, use_cache=False)
        else:
            snapshot = store.read_snapshot(window_start_ms, window_end_ms)
    except Exception as exc:  # noqa: BLE001 - metrics loop degrades through stale snapshots.
        snapshot = _stale_snapshot(snapshot_box.get(), fallback_ts_ms=window_end_ms)
        snapshot_box.set(snapshot)
        if prof is not None:
            prof.record(
                {
                    "kind": "poll",
                    "ts_ms": prof.now_ms(),
                    "fetch_ns": time.perf_counter_ns() - _fetch_t0,
                    "n_models": len(snapshot.models),
                    "stale": True,
                }
            )
        return MetricsRefreshResult(
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            snapshot=snapshot,
            stale=True,
            error=str(exc),
        )

    if prof is not None:
        prof.record(
            {
                "kind": "poll",
                "ts_ms": prof.now_ms(),
                "fetch_ns": time.perf_counter_ns() - _fetch_t0,
                "n_models": len(snapshot.models),
                "stale": bool(snapshot.stale),
            }
        )
    snapshot_box.set(snapshot)
    return MetricsRefreshResult(
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        snapshot=snapshot,
        stale=snapshot.stale,
    )


async def metrics_task(
    store: SnapshotStore,
    snapshot_box: SnapshotBox,
    cfg: MetricsTaskConfig,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    prof: "TickProfiler | None" = None,
) -> None:
    # Single snapshot_box only - no fast/slow split (ADR-0011 / plan S1.2: rescue and
    # fairness share one window). A cfg without metrics_refresh_mode (older callers,
    # tests) keeps the free-running loop; ControllerConfig defaults to phase_aligned.
    mode = getattr(cfg, "metrics_refresh_mode", REFRESH_FREE_RUNNING)
    period_ms = getattr(cfg, "instant_sample_interval_ms", SCRAPE_INTERVAL_MS)
    if mode == REFRESH_PHASE_ALIGNED and cfg.metrics_window_ms % period_ms:
        LOG.error(
            "metrics_sampler: phase_aligned needs metrics_window_ms=%d to be a multiple of the "
            "gateway period %d ms; FALLING BACK to free_running", cfg.metrics_window_ms, period_ms,
        )
        mode = REFRESH_FREE_RUNNING
    if mode == REFRESH_PHASE_ALIGNED:
        sampler = PhaseAlignedSampler(
            store,
            snapshot_box,
            window_ms=cfg.metrics_window_ms,
            period_ms=period_ms,
            offset_ms=getattr(cfg, "metrics_phase_offset_ms", DEFAULT_PHASE_OFFSET_MS),
            adapt=getattr(cfg, "metrics_phase_adapt", True),
            retry_ms=getattr(cfg, "metrics_phase_retry_ms", DEFAULT_PHASE_RETRY_MS),
            stale_hold_windows=getattr(cfg, "metrics_stale_hold_windows", DEFAULT_STALE_HOLD_WINDOWS),
            sleep=sleep,
            prof=prof,
        )
        LOG.info(
            "metrics_sampler: phase_aligned period=%d ms offset=%d ms window=(end-%d, end] "
            "adapt=%s retry=%d ms stale_hold=%d",
            sampler.period_ms, sampler.offset_ms, sampler.window_ms, sampler.adapt,
            sampler.retry_ms, sampler.stale_hold_windows,
        )
        await sampler.run_forever()
        return
    if mode != REFRESH_FREE_RUNNING:
        raise ValueError(f"metrics_refresh_mode must be one of {REFRESH_MODES}, got {mode!r}")
    # Fallback (S1.2): refresh, then sleep metrics_refresh_interval_s. The fetch runs
    # before the sleep, so the real period is interval + fetch time (6.8-8.2 s measured
    # with the v1 schema) and the window end drifts against the gateway's 10 s grid.
    refresh_interval_s = getattr(cfg, "metrics_refresh_interval_s", cfg.monitor_interval_s)
    while True:
        refresh_metrics_once(
            store,
            snapshot_box,
            now_ms=int(time.time() * 1000),
            window_ms=cfg.metrics_window_ms,
            window_mode=getattr(cfg, "metrics_window_mode", "sliding"),
            prof=prof,
        )
        await sleep(refresh_interval_s)


def _last_complete_window(now_ms: int, window_ms: int) -> tuple[int, int]:
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    window_end_ms = max(0, int(now_ms) // window_ms * window_ms)
    window_start_ms = max(0, window_end_ms - window_ms)
    return window_start_ms, window_end_ms


def _sliding_window(now_ms: int, window_ms: int) -> tuple[int, int]:
    # Sliding window ending at now: no epoch alignment, no "last complete block".
    # Removes the 60-120s staleness of tumbling by always ending at the newest data (S1.1).
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    window_end_ms = max(0, int(now_ms))
    window_start_ms = max(0, window_end_ms - window_ms)
    return window_start_ms, window_end_ms


def _stale_snapshot(previous: MetricsSnapshot | None, *, fallback_ts_ms: int) -> MetricsSnapshot:
    if previous is None:
        return MetricsSnapshot(ts_ms=fallback_ts_ms, models={}, stale=True)
    return replace(previous, stale=True)

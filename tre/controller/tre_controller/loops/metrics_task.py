from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Protocol

from tre_common.metrics_schema import MetricsSnapshot


class SnapshotStore(Protocol):
    def read_snapshot(
        self, window_start_ms: int, window_end_ms: int, *, use_cache: bool = True
    ) -> MetricsSnapshot: ...


class MetricsTaskConfig(Protocol):
    metrics_window_ms: int
    metrics_window_mode: str
    monitor_interval_s: float


class SnapshotBox:
    def __init__(self, snapshot: MetricsSnapshot | None = None) -> None:
        self._snapshot = snapshot

    def get(self) -> MetricsSnapshot | None:
        return self._snapshot

    def set(self, snapshot: MetricsSnapshot) -> None:
        self._snapshot = snapshot


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
    try:
        if sliding:
            snapshot = store.read_snapshot(window_start_ms, window_end_ms, use_cache=False)
        else:
            snapshot = store.read_snapshot(window_start_ms, window_end_ms)
    except Exception as exc:  # noqa: BLE001 - metrics loop degrades through stale snapshots.
        snapshot = _stale_snapshot(snapshot_box.get(), fallback_ts_ms=window_end_ms)
        snapshot_box.set(snapshot)
        return MetricsRefreshResult(
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            snapshot=snapshot,
            stale=True,
            error=str(exc),
        )

    snapshot_box.set(snapshot)
    return MetricsRefreshResult(
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        snapshot=snapshot,
        stale=snapshot.stale,
    )


async def metrics_task(store: SnapshotStore, snapshot_box: SnapshotBox, cfg: MetricsTaskConfig) -> None:
    while True:
        refresh_metrics_once(
            store,
            snapshot_box,
            now_ms=int(time.time() * 1000),
            window_ms=cfg.metrics_window_ms,
            window_mode=getattr(cfg, "metrics_window_mode", "sliding"),
        )
        await asyncio.sleep(cfg.monitor_interval_s)


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

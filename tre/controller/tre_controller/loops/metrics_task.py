from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Protocol

from tre_common.metrics_schema import MetricsSnapshot


class SnapshotStore(Protocol):
    def read_snapshot(self, window_start_ms: int, window_end_ms: int) -> MetricsSnapshot: ...


class MetricsTaskConfig(Protocol):
    metrics_window_ms: int
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
) -> MetricsRefreshResult:
    window_start_ms, window_end_ms = _last_complete_window(now_ms, window_ms)
    try:
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
        )
        await asyncio.sleep(cfg.monitor_interval_s)


def _last_complete_window(now_ms: int, window_ms: int) -> tuple[int, int]:
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    window_end_ms = max(0, int(now_ms) // window_ms * window_ms)
    window_start_ms = max(0, window_end_ms - window_ms)
    return window_start_ms, window_end_ms


def _stale_snapshot(previous: MetricsSnapshot | None, *, fallback_ts_ms: int) -> MetricsSnapshot:
    if previous is None:
        return MetricsSnapshot(ts_ms=fallback_ts_ms, models={}, stale=True)
    return replace(previous, stale=True)

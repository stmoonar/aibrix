from __future__ import annotations

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_controller.loops.metrics_task import SnapshotBox, _sliding_window, refresh_metrics_once


class FakeStore:
    def __init__(self, snapshot: MetricsSnapshot | None = None, exc: Exception | None = None) -> None:
        self.snapshot = snapshot
        self.exc = exc
        self.calls: list[tuple[int, int]] = []
        self.use_cache_calls: list[bool] = []

    def read_snapshot(
        self, window_start_ms: int, window_end_ms: int, *, use_cache: bool = True
    ) -> MetricsSnapshot:
        self.calls.append((window_start_ms, window_end_ms))
        self.use_cache_calls.append(use_cache)
        if self.exc is not None:
            raise self.exc
        assert self.snapshot is not None
        return self.snapshot


def _metrics(model: str) -> ModelWindowMetrics:
    return ModelWindowMetrics(
        model=model,
        window_start_ms=60_000,
        window_end_ms=120_000,
        prompt_tokens=10.0,
        generation_tokens=20.0,
        avg_waiting=1.0,
        avg_running=2.0,
        avg_swapping=0.0,
        kv_cache_hit_rate=0.5,
        ttft_p95_ms=100.0,
        tpot_p95_ms=20.0,
        e2e_p95_ms=1000.0,
        routable_pods=1,
        assigned_replicas=1,
        per_pod={},
    )


def _snapshot(ts_ms: int = 120_000, *, stale: bool = False) -> MetricsSnapshot:
    return MetricsSnapshot(ts_ms=ts_ms, models={"m": _metrics("m")}, stale=stale)


def test_snapshot_box_starts_empty_and_replaces_latest_snapshot() -> None:
    box = SnapshotBox()
    first = _snapshot(120_000)
    second = MetricsSnapshot(ts_ms=180_000, models={}, stale=False)

    assert box.get() is None

    box.set(first)
    assert box.get() == first

    box.set(second)
    assert box.get() == second


def test_refresh_metrics_once_reads_last_complete_aligned_window() -> None:
    snapshot = _snapshot(120_000)
    store = FakeStore(snapshot=snapshot)
    box = SnapshotBox()

    result = refresh_metrics_once(store, box, now_ms=125_999, window_ms=60_000)

    assert result.snapshot == snapshot
    assert result.window_start_ms == 60_000
    assert result.window_end_ms == 120_000
    assert result.stale is False
    assert store.calls == [(60_000, 120_000)]
    assert box.get() == snapshot


def test_sliding_window_ends_at_now_without_alignment() -> None:
    # S1.1: sliding window ends at now (no epoch alignment, no last-complete block).
    assert _sliding_window(125_000, 60_000) == (65_000, 125_000)
    # Contrast with tumbling for the same inputs (returns (60_000, 120_000)).


def test_refresh_metrics_once_sliding_reads_now_minus_w_to_now_without_cache() -> None:
    snapshot = _snapshot(125_000)
    store = FakeStore(snapshot=snapshot)
    box = SnapshotBox()

    result = refresh_metrics_once(store, box, now_ms=125_000, window_ms=60_000, window_mode="sliding")

    assert result.window_start_ms == 65_000
    assert result.window_end_ms == 125_000
    assert store.calls == [(65_000, 125_000)]
    assert store.use_cache_calls == [False]  # sliding must not populate the per-window cache
    assert box.get() == snapshot


def test_refresh_metrics_once_tumbling_default_uses_cache() -> None:
    snapshot = _snapshot(120_000)
    store = FakeStore(snapshot=snapshot)
    box = SnapshotBox()

    refresh_metrics_once(store, box, now_ms=125_999, window_ms=60_000)  # default tumbling

    assert store.calls == [(60_000, 120_000)]
    assert store.use_cache_calls == [True]


def test_refresh_metrics_once_marks_previous_snapshot_stale_on_read_failure() -> None:
    previous = _snapshot(120_000)
    box = SnapshotBox(previous)
    store = FakeStore(exc=RuntimeError("redis unavailable"))

    result = refresh_metrics_once(store, box, now_ms=181_000, window_ms=60_000)

    assert result.stale is True
    assert result.error == "redis unavailable"
    assert result.window_start_ms == 120_000
    assert result.window_end_ms == 180_000
    assert result.snapshot.ts_ms == previous.ts_ms
    assert result.snapshot.models == previous.models
    assert result.snapshot.stale is True
    assert box.get() == result.snapshot


def test_refresh_metrics_once_creates_empty_stale_snapshot_without_previous_data() -> None:
    box = SnapshotBox()
    store = FakeStore(exc=RuntimeError("redis unavailable"))

    result = refresh_metrics_once(store, box, now_ms=61_000, window_ms=60_000)

    assert result.stale is True
    assert result.snapshot == MetricsSnapshot(ts_ms=60_000, models={}, stale=True)
    assert box.get() == result.snapshot

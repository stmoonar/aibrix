"""D8 (plan §6.9i): phase-aligned metrics sampler.

The sampler sleeps to ``boundary + offset``, reads the half-open window
``(boundary - W, boundary]`` and publishes an immutable snapshot stamped with the boundary.
A window missing any expected gateway tick is re-read until the next boundary, then
declared stale (previous snapshot kept, marked stale after ``stale_hold_windows``).
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from types import SimpleNamespace

import pytest

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_controller.loops.metrics_task import (
    PhaseAlignedSampler,
    SnapshotBox,
    due_boundary,
    freeze_snapshot,
    metrics_task,
    phase_target,
    window_freshness,
)
from tre_controller.store.metrics_store import MetricsStore

P = 10_000
W = 30_000
BASE = 1_790_000_000_000  # a 10 s boundary


def _window(start: int, end: int, ticks: tuple[int, ...]) -> ModelWindowMetrics:
    return ModelWindowMetrics(
        model="m", window_start_ms=start, window_end_ms=end, prompt_tokens=1.0,
        generation_tokens=2.0, avg_waiting=0.0, avg_running=1.0, avg_swapping=0.0,
        kv_cache_hit_rate=0.0, ttft_p95_ms=None, tpot_p95_ms=None, e2e_p95_ms=None,
        routable_pods=1, assigned_replicas=1, per_pod={}, instant_ticks_ms=ticks,
    )


class Clock:
    """Fake wall clock + asyncio sleep. ``late_ms`` oversleeps every sleep; ``jump_ms``
    steps the wall clock once, during the next sleep."""

    def __init__(self, now: int) -> None:
        self.now = now
        self.sleeps: list[float] = []
        self.late_ms = 0
        self.jump_ms = 0

    def __call__(self) -> int:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += int(round(seconds * 1000)) + self.late_ms
        if self.jump_ms:
            self.now += self.jump_ms
            self.jump_ms = 0


class Gateway:
    """Fake store: tick ``t`` becomes readable at ``t + lag_ms``; ``skip`` ticks never do."""

    def __init__(self, clock: Clock, *, lag_ms: int = 0, skip: tuple[int, ...] = (), ticks: bool = True) -> None:
        self.clock = clock
        self.lag_ms = lag_ms
        self.skip = set(skip)
        self.ticks = ticks
        self.calls: list[tuple[int, int, int]] = []
        self.fail = False

    async def fetch(self, start: int, end: int) -> MetricsSnapshot:
        self.calls.append((start, end, self.clock.now))
        if self.fail:
            raise RuntimeError("redis down")
        first = (start // P + 1) * P
        visible = tuple(
            t for t in range(first, end + 1, P)
            if t + self.lag_ms <= self.clock.now and t not in self.skip
        ) if self.ticks else ()
        return MetricsSnapshot(ts_ms=end, models={"m": _window(start, end, visible)}, stale=False)


def _sampler(clock: Clock, gateway: Gateway, box: SnapshotBox | None = None, **kw) -> PhaseAlignedSampler:
    return PhaseAlignedSampler(
        store=None, snapshot_box=box or SnapshotBox(), window_ms=W, period_ms=P,
        clock_ms=clock, sleep=clock.sleep, fetch=gateway.fetch, **kw,
    )


def _run(sampler: PhaseAlignedSampler, n: int = 1):
    async def go():
        return [await sampler.run_once() for _ in range(n)]
    return asyncio.run(go())


# --- phase arithmetic ----------------------------------------------------------------

def test_phase_target_is_next_boundary_plus_offset() -> None:
    assert phase_target(BASE + 3_456, period_ms=P, offset_ms=2_000, last_end_ms=None) == (BASE + 10_000, BASE + 12_000)
    # already past this boundary's read time -> the next boundary
    assert phase_target(BASE + 2_001, period_ms=P, offset_ms=2_000, last_end_ms=None) == (BASE + 10_000, BASE + 12_000)
    # exactly at the read time -> read now
    assert phase_target(BASE + 2_000, period_ms=P, offset_ms=2_000, last_end_ms=None) == (BASE, BASE + 2_000)
    # never re-read a window already published
    assert phase_target(BASE + 2_000, period_ms=P, offset_ms=2_000, last_end_ms=BASE) == (BASE + 10_000, BASE + 12_000)


def test_due_boundary_is_newest_boundary_whose_read_time_passed() -> None:
    assert due_boundary(BASE + 1_999, period_ms=P, offset_ms=2_000) == BASE - P
    assert due_boundary(BASE + 2_000, period_ms=P, offset_ms=2_000) == BASE
    assert due_boundary(BASE + 11_999, period_ms=P, offset_ms=2_000) == BASE


def test_sleeps_until_boundary_plus_offset_not_a_fixed_interval() -> None:
    clock = Clock(BASE + 7_321)
    sampler = _sampler(clock, Gateway(clock))
    _run(sampler)
    assert clock.sleeps == [pytest.approx((BASE + 12_000 - (BASE + 7_321)) / 1000.0)]


# --- steady state: exact 10 s period, 3 ticks, full 30 s span -------------------------

def test_steady_state_publishes_every_boundary_with_three_ticks() -> None:
    clock = Clock(BASE + 5_000)
    gateway = Gateway(clock)
    box = SnapshotBox()
    sampler = _sampler(clock, gateway, box)
    outcomes = _run(sampler, 6)
    ends = [o.window_end_ms for o in outcomes]
    assert ends == [BASE + P * k for k in range(1, 7)]
    assert all(o.kind == "published" and o.attempts == 1 for o in outcomes)
    # every read is (end - 30 s, end] at end + offset: period exactly 10 s, zero jitter
    assert [(s, e) for s, e, _ in gateway.calls] == [(e - W, e) for e in ends]
    read_times = [t for _, _, t in gateway.calls]
    assert {b - a for a, b in zip(read_times, read_times[1:])} == {P}
    assert [t - e for (_, e, t) in gateway.calls] == [2_000] * 6
    snap = box.get()
    assert snap.ts_ms == ends[-1] and snap.models["m"].window_end_ms == ends[-1]
    assert snap.models["m"].instant_ticks_ms == (ends[-1] - 20_000, ends[-1] - 10_000, ends[-1])


def test_small_late_wakeup_keeps_the_window() -> None:
    clock = Clock(BASE + 5_000)
    clock.late_ms = 300  # scheduler jitter
    outcome = _run(_sampler(clock, Gateway(clock)))[0]
    assert (outcome.kind, outcome.window_end_ms, outcome.missed_boundaries) == ("published", BASE + P, 0)


def test_late_wakeup_past_a_boundary_reads_the_newest_window_and_logs(caplog) -> None:
    clock = Clock(BASE + 5_000)
    clock.late_ms = 15_000  # event loop stalled 15 s
    with caplog.at_level(logging.WARNING, logger="tre_controller.metrics"):
        outcome = _run(_sampler(clock, Gateway(clock)))[0]
    assert outcome.kind == "published"
    assert outcome.missed_boundaries == 1
    assert outcome.window_end_ms == BASE + 2 * P
    assert "metrics_boundary_missed" in caplog.text


def test_forward_clock_jump_during_sleep_skips_to_the_newest_window() -> None:
    clock = Clock(BASE + 5_000)
    sampler = _sampler(clock, Gateway(clock))
    _run(sampler)
    clock.jump_ms = 3_600_000
    outcome = _run(sampler)[0]
    assert outcome.missed_boundaries == 360
    assert outcome.window_end_ms == BASE + 2 * P + 3_600_000


def test_backward_clock_jump_during_sleep_does_not_read_early() -> None:
    clock = Clock(BASE + 5_000)
    gateway = Gateway(clock)
    sampler = _sampler(clock, gateway)
    _run(sampler)  # published BASE + P
    clock.jump_ms = -5_000
    early = _run(sampler)[0]
    assert early.kind == "early" and len(gateway.calls) == 1  # no fetch
    again = _run(sampler)[0]  # sleeps to the same target and reads it
    assert (again.kind, again.window_end_ms) == ("published", BASE + 2 * P)


def test_large_backward_clock_step_restarts_the_schedule(caplog) -> None:
    clock = Clock(BASE + 5_000)
    sampler = _sampler(clock, Gateway(clock))
    _run(sampler, 2)
    clock.now -= 3_600_000
    with caplog.at_level(logging.WARNING, logger="tre_controller.metrics"):
        outcome = _run(sampler)[0]
    assert "metrics_clock_regressed" in caplog.text
    assert outcome.kind == "published"
    assert outcome.window_end_ms < BASE  # did not wait an hour for the old schedule


# --- gateway write phase: retry, adapt, stale ---------------------------------------

def test_waits_for_the_boundary_tick_and_learns_the_gateway_phase(caplog) -> None:
    # 09-22 live: the tre-gateway ticker writes ~5.3 s after the boundary it stamps.
    clock = Clock(BASE + 5_000)
    gateway = Gateway(clock, lag_ms=5_300)
    sampler = _sampler(clock, gateway, retry_ms=500)
    with caplog.at_level(logging.INFO, logger="tre_controller.metrics"):
        first, second, third = _run(sampler, 3)
    assert first.kind == "published" and first.attempts == 8  # 2.0, 2.5, ..., 5.5 s
    assert sampler.offset_ms == 5_500
    assert "metrics_phase_adapted" in caplog.text
    assert (second.attempts, third.attempts) == (1, 1)
    assert [e for e in (first.window_end_ms, second.window_end_ms, third.window_end_ms)] == [
        BASE + P, BASE + 2 * P, BASE + 3 * P,
    ]
    assert gateway.calls[-1][2] - gateway.calls[-1][1] == 5_500


def test_without_adapt_the_offset_stays_and_every_cycle_retries() -> None:
    clock = Clock(BASE + 5_000)
    sampler = _sampler(clock, Gateway(clock, lag_ms=5_300), adapt=False)
    outcomes = _run(sampler, 2)
    assert [o.attempts for o in outcomes] == [8, 8]
    assert sampler.offset_ms == 2_000


def test_relearn_drops_an_adapted_offset_back_to_the_floor() -> None:
    clock = Clock(BASE + 5_000)
    gateway = Gateway(clock, lag_ms=5_300)
    sampler = _sampler(clock, gateway, relearn_cycles=3)
    _run(sampler, 2)
    assert sampler.offset_ms == 5_500
    _run(sampler)  # third cycle -> relearn
    assert sampler.offset_ms == 2_000
    gateway.lag_ms = 1_000  # gateway restarted with an earlier phase
    outcome = _run(sampler)[0]
    assert outcome.attempts == 1 and sampler.offset_ms == 2_000


def test_missing_tick_is_stale_previous_snapshot_is_kept_then_marked(caplog) -> None:
    clock = Clock(BASE + 5_000)
    gateway = Gateway(clock)
    box = SnapshotBox()
    sampler = _sampler(clock, gateway, box, stale_hold_windows=2)
    _run(sampler)
    good = box.get()
    # the gateway skips one tick: the next 3 windows all contain it -> 3 stale windows
    gateway.skip = {BASE + 2 * P}
    with caplog.at_level(logging.WARNING, logger="tre_controller.metrics"):
        stale = _run(sampler, 3)
    assert [o.kind for o in stale] == ["stale"] * 3
    assert all(o.reason.startswith("missing_ticks:") for o in stale)
    assert "metrics_window_stale" in caplog.text
    # retried until the next boundary was due, then gave up
    assert stale[0].attempts > 1
    # hold for 2 windows: the very same (non-stale) snapshot object is served
    assert sampler.consecutive_stale == 3
    assert box.get().stale is True and box.get().ts_ms == good.ts_ms
    recovered = _run(sampler)[0]
    assert recovered.kind == "published" and sampler.consecutive_stale == 0
    assert box.get().stale is False and box.get().ts_ms == BASE + 5 * P


def test_stale_hold_serves_the_identical_previous_snapshot() -> None:
    clock = Clock(BASE + 5_000)
    gateway = Gateway(clock)
    box = SnapshotBox()
    sampler = _sampler(clock, gateway, box, stale_hold_windows=2)
    _run(sampler)
    good = box.get()
    gateway.skip = {BASE + 2 * P}
    _run(sampler)
    assert box.get() is good
    _run(sampler)
    assert box.get() is good


def test_first_window_stale_serves_an_empty_stale_snapshot() -> None:
    clock = Clock(BASE + 5_000)
    gateway = Gateway(clock, skip=(BASE + P,))
    box = SnapshotBox()
    outcome = _run(_sampler(clock, gateway, box))[0]
    assert outcome.kind == "stale"
    assert box.get().stale is True and dict(box.get().models) == {}


def test_fetch_error_is_stale() -> None:
    clock = Clock(BASE + 5_000)
    gateway = Gateway(clock)
    gateway.fail = True
    outcome = _run(_sampler(clock, gateway))[0]
    assert outcome.kind == "stale" and outcome.reason == "error:redis down"


def test_no_ticks_at_all_is_fresh_but_ticks_then_silence_is_stale() -> None:
    clock = Clock(BASE + 5_000)
    gateway = Gateway(clock, ticks=False)
    sampler = _sampler(clock, gateway)
    assert _run(sampler)[0].reason == "no_ticks"  # idle cluster / no pods
    gateway.ticks = True
    assert _run(sampler)[0].kind == "published"
    gateway.ticks = False  # gateway went quiet
    outcome = _run(sampler)[0]
    assert (outcome.kind, outcome.reason) == ("stale", "no_ticks_after_ticks")


def test_window_freshness_pools_ticks_over_pods() -> None:
    end = BASE + 3 * P
    a = _window(end - W, end, (end - 20_000, end - 10_000))
    b = dataclasses.replace(_window(end - W, end, (end,)), model="n")
    snap = MetricsSnapshot(ts_ms=end, models={"m": a, "n": b}, stale=False)
    verdict = window_freshness(snap, window_end_ms=end, window_ms=W, period_ms=P)
    assert verdict.fresh and verdict.ticks == (end - 20_000, end - 10_000, end)
    # a tick exactly at the window start is outside the half-open window
    only_start = MetricsSnapshot(ts_ms=end, models={"m": _window(end - W, end, (end - W,))}, stale=False)
    assert window_freshness(only_start, window_end_ms=end, window_ms=W, period_ms=P).ticks == ()


# --- immutability ----------------------------------------------------------------------

def test_published_snapshot_is_immutable() -> None:
    clock = Clock(BASE + 5_000)
    box = SnapshotBox()
    _run(_sampler(clock, Gateway(clock), box))
    snap = box.get()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.ts_ms = 0  # type: ignore[misc]
    with pytest.raises(TypeError):
        snap.models["x"] = snap.models["m"]  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.models["m"].avg_running = 9.0  # type: ignore[misc]


def test_freeze_snapshot_keeps_values_and_blocks_per_pod_writes() -> None:
    window = _window(0, W, ())
    window = dataclasses.replace(window, per_pod={"p": object()})
    frozen = freeze_snapshot(MetricsSnapshot(ts_ms=W, models={"m": window}, stale=False))
    assert frozen.models["m"].avg_running == 1.0 and "p" in frozen.models["m"].per_pod
    with pytest.raises(TypeError):
        frozen.models["m"].per_pod["q"] = None  # type: ignore[index]


# --- the real store: half-open read ------------------------------------------------------

class _Redis:
    def __init__(self) -> None:
        self.sets: dict = {}
        self.zsets: dict = {}

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def zrangebyscore(self, key, lo, hi):
        return [m for s, m in self.zsets.get(key, []) if float(lo) <= s <= float(hi)]

    def add(self, key, ts, doc):
        body = dict(doc, timestamp=ts)
        self.zsets.setdefault(key, []).append((float(ts), json.dumps(body)))
        self.zsets[key].sort(key=lambda item: item[0])


def _store_with_ticks(end: int) -> MetricsStore:
    redis = _Redis()
    redis.sets["tre:v2:pods:m"] = {"default/p"}
    for k, tick in enumerate(range(end - 40_000, end + 1, P)):
        redis.add("tre:v2:inst:default/p", tick, {"pod_name": "p", "model_metrics": {"m/num_requests_running": 3.0}})
        redis.add("tre:v2:hist:default/p", tick, {"pod_name": "p", "model_histogram_metrics": {
            "m/request_generation_tokens": {"sum": 100.0 * k, "count": k, "buckets": {}},
            "m/request_prompt_tokens": {"sum": 10.0 * k, "count": k, "buckets": {}},
        }})
    registry = SimpleNamespace(models=lambda: [SimpleNamespace(name="m")])
    return MetricsStore(redis, registry, instant_sample_interval_ms=P, schema="v2")


def test_store_half_open_read_gives_three_ticks_and_a_30s_token_span() -> None:
    end = BASE + 10 * P
    store = _store_with_ticks(end)
    half = store.read_model_window("m", end - W, end, use_cache=False, start_exclusive=True)
    assert half.instant_ticks_ms == (end - 20_000, end - 10_000, end)
    assert half.avg_running == pytest.approx(3.0)  # 9 / 3 expected
    assert half.generation_tokens == pytest.approx(300.0)  # hist(end) - hist(end - 30 s)
    assert (half.window_start_ms, half.window_end_ms) == (end - W, end)
    # the closed read on a boundary double-counts the start tick (4 ticks / 3) and
    # takes its token baseline one tick early (40 s span) - why the sampler reads (s, e]
    closed = store.read_model_window("m", end - W, end, use_cache=False)
    assert closed.avg_running == pytest.approx(4.0)
    assert closed.generation_tokens == pytest.approx(400.0)


def test_metrics_task_rejects_unknown_refresh_mode() -> None:
    cfg = SimpleNamespace(metrics_refresh_mode="bogus", metrics_window_ms=W, monitor_interval_s=5.0,
                          metrics_refresh_interval_s=5.0, metrics_window_mode="sliding")
    with pytest.raises(ValueError):
        asyncio.run(metrics_task(None, SnapshotBox(), cfg))


def test_metrics_task_falls_back_to_free_running_when_window_is_off_grid(caplog) -> None:
    class Stop(Exception):
        pass

    calls: list = []

    class Store:
        def read_snapshot(self, start, end, *, use_cache=True):
            calls.append((start, end, use_cache))
            return MetricsSnapshot(ts_ms=end, models={}, stale=False)

    async def sleep(_s):
        raise Stop

    cfg = SimpleNamespace(metrics_refresh_mode="phase_aligned", metrics_window_ms=35_000,
                          monitor_interval_s=5.0, metrics_refresh_interval_s=5.0,
                          metrics_window_mode="sliding", instant_sample_interval_ms=P)
    with caplog.at_level(logging.ERROR, logger="tre_controller.metrics"), pytest.raises(Stop):
        asyncio.run(metrics_task(Store(), SnapshotBox(), cfg, sleep=sleep))
    assert "FALLING BACK to free_running" in caplog.text
    assert len(calls) == 1 and calls[0][1] - calls[0][0] == 35_000 and calls[0][2] is False

"""D8 / plan §6.9g pitfall 2: grid-aligned 10 s windows, online == offline.

The phase-aligned controller reads ``(B - 30 s, B]`` for every 10 s boundary B.
``rewindow_from_raw --window-align grid --step-ms 10000`` must produce the same windows
from the raw capture, and the smoothed TSS must be bitwise equal - with the controller's
rescue/fairness loops re-reading every snapshot several times.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tre_common.metrics_schema import MetricsSnapshot
from tre_common.rediskeys import SCRAPE_INTERVAL_MS
from tre_common.registry import load_registry
from tre_controller.loops.fairness_task import run_fairness_tick
from tre_controller.loops.metrics_task import PhaseAlignedSampler, SnapshotBox
from tre_controller.loops.rescue_task import run_rescue_tick
from tre_controller.signals.trs import SignalState
from tre_controller.store.metrics_store import MetricsStore
from scripts import calibration_campaign as campaign
from scripts import openloop, r3_grid, rewindow_from_raw

TRE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = TRE_ROOT / "deploy" / "registry.yaml"
MODEL = "dsqwen-7b"
POD = "default/pod-a"
P = SCRAPE_INTERVAL_MS
W = 30_000
BASE = 1_790_000_000_000
SPAN_S = 300
GAP_S = (120, 185)  # traffic stops: an idle stretch longer than one window


class _Redis:
    def __init__(self) -> None:
        self.sets: dict = {}
        self.zsets: dict = {}

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def zrangebyscore(self, key, lo, hi):
        return [m for s, m in self.zsets.get(key, []) if float(lo) <= s <= float(hi)]

    def add(self, key, ts, doc):
        self.zsets.setdefault(key, []).append((float(ts), json.dumps(dict(doc, timestamp=ts))))
        self.zsets[key].sort(key=lambda item: item[0])


def _busy(t_s: float) -> bool:
    return not (GAP_S[0] <= t_s < GAP_S[1])


def _requests() -> list[dict]:
    out = []
    for i in range(SPAN_S * 3):
        done = BASE + 700 + i * 333 + (i * 97) % 211
        if not _busy((done - BASE) / 1000.0) or done > BASE + SPAN_S * 1000:
            continue
        out.append({"done_ts_ms": done, "input_tokens": 100 + (i * 13) % 900, "output_tokens": 20 + (i * 7) % 300,
                    "ttft_ms": 80.0, "tpot_ms": 20.0, "e2e_ms": 1000.0})
    return out


def _sidecar() -> list[dict]:
    samples = []
    for s in range(SPAN_S + 1):
        busy = _busy(float(s))
        samples.append({
            "ts_ms": BASE + s * 1000 + (s * 37) % 900,  # 1 Hz with jitter, 0-0.9 s late
            "running": float(3 + (s * 5) % 7) if busy else 0.0,
            "waiting": float((s * 3) % 4) if busy else 0.0,
            "swapping": 0.0,
        })
    return openloop.mark_live_grid(samples)


def _online_store(records: list[dict], sidecar: list[dict]) -> MetricsStore:
    """What the gateway would have written: one inst + one cumulative hist doc per tick."""
    redis = _Redis()
    redis.sets[f"tre:v2:pods:{MODEL}"] = {POD}
    live = [s for s in sidecar if s["on_live_grid"]]
    for s in live:
        tick = s["ts_ms"] // P * P
        done = [r for r in records if r["done_ts_ms"] <= tick]
        redis.add(f"tre:v2:inst:{POD}", tick, {"pod_name": "pod-a", "model_metrics": {
            f"{MODEL}/num_requests_running": s["running"],
            f"{MODEL}/num_requests_waiting": s["waiting"],
            f"{MODEL}/num_requests_swapped": 0.0,
        }})
        redis.add(f"tre:v2:hist:{POD}", tick, {"pod_name": "pod-a", "model_histogram_metrics": {
            f"{MODEL}/request_prompt_tokens": {"sum": float(sum(r["input_tokens"] for r in done)), "count": len(done), "buckets": {}},
            f"{MODEL}/request_generation_tokens": {"sum": float(sum(r["output_tokens"] for r in done)), "count": len(done), "buckets": {}},
        }})
    return MetricsStore(redis, load_registry(str(REGISTRY_PATH)), instant_sample_interval_ms=P, schema="v2",
                        min_latency_samples=0)


class _Queue:
    def inflight_models(self) -> set[str]:
        return set()

    def submit(self, actions) -> object:
        return object()


def _offline_rows(records, sidecar, spec):
    cell = r3_grid.GridCell.from_scenario_id("i512_o128_c8")
    return rewindow_from_raw.rewindow_cell(
        records, sidecar, cell, spec, window_ms=W, step_ms=P, percentile_mode="bucket_upper",
        min_latency_samples=0, instant_sample_interval_ms=P, instant_grid="live",
        window_align="grid",
    )


def test_online_phase_aligned_equals_offline_grid_rewindow_bitwise() -> None:
    records, sidecar = _requests(), _sidecar()
    registry = load_registry(str(REGISTRY_PATH))
    spec = registry.model(MODEL)
    rows = _offline_rows(records, sidecar, spec)
    assert rows, "no offline windows"
    offline_ends = [r["window_end_ms"] for r in rows]
    assert all(end % P == 0 for end in offline_ends)
    assert {b - a for a, b in zip(offline_ends, offline_ends[1:])} == {P}

    # online: the sampler reads the store at every boundary + 2 s ...
    store = _online_store(records, sidecar)
    now = {"t": offline_ends[0] - 1_000}
    box = SnapshotBox()

    async def sleep(seconds: float) -> None:
        now["t"] += int(round(seconds * 1000))

    async def fetch(start: int, end: int) -> MetricsSnapshot:
        return store.read_snapshot(start, end, use_cache=False, start_exclusive=True)

    sampler = PhaseAlignedSampler(store, box, window_ms=W, period_ms=P, clock_ms=lambda: now["t"],
                                  sleep=sleep, fetch=fetch)
    state = SignalState(warmup_ms=-1, dwell_windows=2)
    online: dict[int, float | None] = {}

    async def run() -> None:
        while (sampler.last_end_ms or 0) < offline_ends[-1]:
            outcome = await sampler.run_once()
            assert outcome.kind == "published", outcome
            snap = box.get()
            # ... and the decision loops re-read each snapshot (rescue twice at 5 s, fairness)
            for loop in (run_rescue_tick, run_fairness_tick, run_rescue_tick):
                result = loop(snap, queue=_Queue(), registry=registry, signal_state=state)
            ctx = result.model_contexts[MODEL]
            online[snap.ts_ms] = ctx["trs"] if ctx.get("tss_defined", True) else None
            assert snap.models[MODEL].instant_ticks_ms == (snap.ts_ms - 20_000, snap.ts_ms - 10_000, snap.ts_ms)

    asyncio.run(run())
    assert list(online) == offline_ends
    offline_trs = [row["trs"] for row in rows]
    online_trs = [online[end] for end in offline_ends]
    # bitwise: identical floats, not approx; the idle gap (EMA reset) is inside the span
    assert online_trs == offline_trs
    assert any(v is None for v in offline_trs) and any(v is not None for v in offline_trs)


def test_grid_windows_end_on_the_grid_with_three_live_ticks() -> None:
    wins = rewindow_from_raw.enumerate_windows(BASE + 12_345, BASE + 100_000, W, P, align_ms=P)
    assert wins[0] == (BASE + 10_000, BASE + 40_000)
    assert all(end % P == 0 for _, end in wins)
    with pytest.raises(ValueError):
        rewindow_from_raw.enumerate_windows(BASE, BASE + 100_000, W, 5_000, align_ms=P)
    sidecar = [s for s in _sidecar() if s["on_live_grid"]]
    wm = rewindow_from_raw.aggregate_window(
        [], sidecar, MODEL, BASE + 60_000, BASE + 90_000, percentile_mode="bucket_upper",
        min_latency_samples=0, instant_sample_interval_ms=P, half_open_start=True, instant_tick_ms=P,
    )
    expected = [s for s in sidecar if BASE + 60_000 < s["ts_ms"] // P * P <= BASE + 90_000]
    assert len(expected) == 3
    assert wm.avg_running == sum(s["running"] for s in expected) / 3


def test_unaligned_mode_is_unchanged() -> None:
    records, sidecar = _requests(), _sidecar()
    spec = load_registry(str(REGISTRY_PATH)).model(MODEL)
    cell = r3_grid.GridCell.from_scenario_id("i512_o128_c8")
    kwargs = dict(window_ms=W, step_ms=5_000, percentile_mode="bucket_upper", min_latency_samples=0,
                  instant_sample_interval_ms=P, instant_grid="live")
    legacy = rewindow_from_raw.rewindow_cell(records, sidecar, cell, spec, **kwargs)
    explicit = rewindow_from_raw.rewindow_cell(records, sidecar, cell, spec, window_align="none", **kwargs)
    assert legacy == explicit
    assert legacy[0]["window_start_ms"] == min(
        [r["done_ts_ms"] for r in records] + [s["ts_ms"] for s in sidecar if s["on_live_grid"]]
    )
    aligned = _offline_rows(records, sidecar, spec)
    # 300 s of capture: ~57 free-phase windows at 5 s vs ~28 grid windows at 10 s
    assert len(aligned) < len(legacy)


class _Args:
    out_dir = Path("/out")
    window_ms = 30000
    fit_step_ms = 10000
    instant_sample_ms = 1000
    ttft_slo_ms = 500.0
    tpot_slo_ms = 75.0
    max_model_error_rate = 0.05
    envoy_stats_url = None


def test_fit_plan_defaults_to_grid_aligned_10s_windows() -> None:
    plan = campaign.fit_plan(["dsqwen-7b"], Path("/out"), Path("/raw"), _Args())
    assert plan["window_align"] == "grid" and plan["step_ms"] == 10000
    for entry in plan["rewindow"]:
        cmd = entry["command"]
        assert cmd[cmd.index("--window-align") + 1] == "grid"
    legacy = _Args()
    legacy.fit_window_align = "none"
    legacy.fit_step_ms = 5000
    plan = campaign.fit_plan(["dsqwen-7b"], Path("/out"), Path("/raw"), legacy)
    assert all(e["command"][e["command"].index("--window-align") + 1] == "none" for e in plan["rewindow"])


def test_campaign_cli_rejects_an_off_grid_step() -> None:
    with pytest.raises(SystemExit):
        campaign.main(["--out-dir", "/tmp/x", "--fit-step-ms", "5000"])

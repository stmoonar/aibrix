from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from tre_common.registry import load_registry
from tre_controller.store.metrics_store import MetricsStore
from scripts import r3_grid, rewindow_from_raw

TRE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = TRE_ROOT / "deploy" / "registry.yaml"
MODEL = "dsqwen-7b"
POD = "default/pod-a"


# --- minimal FakeRedis (v2 schema), mirrors controller/tests/test_metrics_store.py -------
class FakeRedis:
    def __init__(self) -> None:
        self.sets: dict = {}
        self.zsets: dict = {}

    def sadd(self, key, *values):
        self.sets.setdefault(key, set()).update(values)

    def zadd(self, key, mapping):
        self.zsets.setdefault(key, [])
        for member, score in mapping.items():
            self.zsets[key].append((float(score), member))
        self.zsets[key].sort(key=lambda item: item[0])

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def zrangebyscore(self, key, minimum, maximum):
        lo, hi = float(minimum), float(maximum)
        return [member for score, member in self.zsets.get(key, []) if lo <= score <= hi]


def _add_doc(redis: FakeRedis, key: str, ts_ms: int, doc: dict) -> None:
    body = dict(doc)
    body["timestamp"] = ts_ms
    redis.zadd(key, {json.dumps(body, sort_keys=True): ts_ms})


def _cumulative_buckets_seconds(latencies_ms: list[float]) -> dict:
    counts = Counter(round(v / 1000.0, 9) for v in latencies_ms)
    running = 0
    out: dict = {}
    for le in sorted(counts):
        running += counts[le]
        out[repr(le)] = running
    return out


def _zero_buckets(buckets: dict) -> dict:
    return {le: 0 for le in buckets}


def _entry(sum_, count_, buckets):
    return {"sum": sum_, "count": count_, "buckets": buckets}


def _hist_doc(prompt_sum, prompt_count, gen_sum, gen_count, ttft, tpot, e2e) -> dict:
    return {
        "pod_name": "pod-a",
        "model_histogram_metrics": {
            f"{MODEL}/request_prompt_tokens": _entry(prompt_sum, prompt_count, {"1": prompt_count}),
            f"{MODEL}/request_generation_tokens": _entry(gen_sum, gen_count, {"1": gen_count}),
            f"{MODEL}/time_to_first_token_seconds": _entry(*ttft),
            f"{MODEL}/time_per_output_token_seconds": _entry(*tpot),
            f"{MODEL}/e2e_request_latency_seconds": _entry(*e2e),
        },
    }


def _inst_doc(waiting, running, swapping) -> dict:
    return {
        "pod_name": "pod-a",
        "model_metrics": {
            f"{MODEL}/num_requests_waiting": waiting,
            f"{MODEL}/num_requests_running": running,
            f"{MODEL}/num_requests_swapped": swapping,
        },
    }


# --- synthetic per-request + instant data (single source of truth for both paths) --------
def _make_requests() -> list[dict]:
    records: list[dict] = []
    base = 2_000
    for i in range(20):
        done = base + i * 2_000  # 2s apart -> all inside a 60s window at [1000, 61000)
        ttft = 100.0 + 10.0 * i
        tpot = 20.0 + 2.0 * i
        e2e = 5_000.0 + 100.0 * i
        records.append({
            "send_ts_ms": done - int(e2e),
            "recv_first_token_ts_ms": done - int(e2e) + int(ttft),
            "done_ts_ms": done,
            "input_tokens": 128 + i,
            "output_tokens": 64 + i,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "e2e_ms": e2e,
            "http_status": 200,
            "cell_id": "i512_o128_c8",
        })
    return records


def _make_instant() -> list[dict]:
    return [
        {"ts_ms": 3_000, "waiting": 2.0, "running": 4.0, "swapping": 1.0},
        {"ts_ms": 8_000, "waiting": 3.0, "running": 5.0, "swapping": 0.0},
        {"ts_ms": 13_000, "waiting": 1.0, "running": 6.0, "swapping": 2.0},
    ]


def _build_online_store(records: list[dict], instant: list[dict], mode: str, window_end: int) -> MetricsStore:
    redis = FakeRedis()
    redis.sadd(f"tre:v2:pods:{MODEL}", POD)
    ttft = [r["ttft_ms"] for r in records]
    tpot = [r["tpot_ms"] for r in records]
    e2e = [r["e2e_ms"] for r in records]
    prompt_total = sum(r["input_tokens"] for r in records)
    gen_total = sum(r["output_tokens"] for r in records)
    n = len(records)
    ttft_end = _cumulative_buckets_seconds(ttft)
    tpot_end = _cumulative_buckets_seconds(tpot)
    e2e_end = _cumulative_buckets_seconds(e2e)
    # baseline (pre-window) doc: cumulative counters at zero
    _add_doc(redis, f"tre:v2:hist:{POD}", 500, _hist_doc(
        0, 0, 0, 0,
        (0.0, 0, _zero_buckets(ttft_end)),
        (0.0, 0, _zero_buckets(tpot_end)),
        (0.0, 0, _zero_buckets(e2e_end)),
    ))
    # end-of-window doc: cumulative counters over all requests
    _add_doc(redis, f"tre:v2:hist:{POD}", window_end, _hist_doc(
        prompt_total, n, gen_total, n,
        (sum(ttft) / 1000.0, n, ttft_end),
        (sum(tpot) / 1000.0, n, tpot_end),
        (sum(e2e) / 1000.0, n, e2e_end),
    ))
    for s in instant:
        _add_doc(redis, f"tre:v2:inst:{POD}", s["ts_ms"], _inst_doc(s["waiting"], s["running"], s["swapping"]))
    registry = load_registry(str(REGISTRY_PATH))
    return MetricsStore(
        redis, registry, instant_sample_interval_ms=5_000,
        percentile_mode=mode, schema="v2", min_latency_samples=0,
    )


@pytest.mark.parametrize("mode", ["bucket_upper", "interpolated"])
def test_aggregate_window_matches_online_metrics_store(mode: str) -> None:
    # doc15 §4 gate: rewindow's 60s aggregation must equal the online MetricsStore path
    # for the same data. Because rewindow reuses histogram_percentile over the exact
    # samples and the online histogram's buckets ARE those samples, the p95 columns match
    # exactly (not just within tolerance); tokens/queue match by the shared formulas.
    records = _make_requests()
    instant = _make_instant()
    ws, we = 1_000, 61_000

    offline = rewindow_from_raw.aggregate_window(
        records, instant, MODEL, ws, we,
        percentile_mode=mode, min_latency_samples=0, instant_sample_interval_ms=5_000,
    )
    store = _build_online_store(records, instant, mode, we)
    online = store.read_model_window(MODEL, ws, we)

    assert offline.prompt_tokens == pytest.approx(online.prompt_tokens)
    assert offline.generation_tokens == pytest.approx(online.generation_tokens)
    assert offline.avg_waiting == pytest.approx(online.avg_waiting)
    assert offline.avg_running == pytest.approx(online.avg_running)
    assert offline.avg_swapping == pytest.approx(online.avg_swapping)
    assert offline.ttft_p95_ms == pytest.approx(online.ttft_p95_ms, abs=1e-6)
    assert offline.tpot_p95_ms == pytest.approx(online.tpot_p95_ms, abs=1e-6)
    assert offline.e2e_p95_ms == pytest.approx(online.e2e_p95_ms, abs=1e-6)


def test_rewindow_cell_matches_online_trs_column() -> None:
    # The whole row (incl. the trs column) reuses r3_grid.compute_window_results, so an
    # offline 60s single-window row equals what the online driver would emit for that
    # window from the same aggregated metrics.
    records = _make_requests()
    instant = _make_instant()
    ws, we = 1_000, 61_000
    registry = load_registry(str(REGISTRY_PATH))
    spec = registry.model(MODEL)
    cell = r3_grid.GridCell.from_scenario_id("i512_o128_c8")

    rows = rewindow_from_raw.rewindow_cell(
        records, instant, cell, spec,
        window_ms=60_000, step_ms=60_000, percentile_mode="bucket_upper",
        min_latency_samples=0, instant_sample_interval_ms=5_000,
        start_ms=ws, end_ms=we,
    )
    assert len(rows) == 1

    store = _build_online_store(records, instant, "bucket_upper", we)
    wm = store.read_model_window(MODEL, ws, we)
    online_rows = [
        r3_grid.window_row(cell, w, res.TRS, res.Q_ctl)
        for w, res in zip([wm], r3_grid.compute_window_results([wm], spec))
    ]
    assert rows[0]["trs"] == pytest.approx(online_rows[0]["trs"])
    assert rows[0]["queue_control"] == pytest.approx(online_rows[0]["queue_control"])
    assert rows[0]["p95_e2e"] == pytest.approx(online_rows[0]["p95_e2e"], abs=1e-6)


def test_same_raw_produces_20s_and_60s(tmp_path: Path) -> None:
    # doc15 §4 gate: one raw capture -> both 20s and 60s CSVs, no re-run.
    records = _make_requests()
    instant = _make_instant()
    registry = load_registry(str(REGISTRY_PATH))
    spec = registry.model(MODEL)
    cell = r3_grid.GridCell.from_scenario_id("i512_o128_c8")

    rows_60 = rewindow_from_raw.rewindow_cell(
        records, instant, cell, spec,
        window_ms=60_000, step_ms=60_000, percentile_mode="bucket_upper",
        min_latency_samples=0, instant_sample_interval_ms=5_000,
        start_ms=1_000, end_ms=61_000,
    )
    rows_20 = rewindow_from_raw.rewindow_cell(
        records, instant, cell, spec,
        window_ms=20_000, step_ms=20_000, percentile_mode="bucket_upper",
        min_latency_samples=0, instant_sample_interval_ms=5_000,
        start_ms=1_000, end_ms=61_000,
    )
    assert len(rows_60) == 1
    assert len(rows_20) == 3  # [1000,21000) [21000,41000) [41000,61000)
    # tokens conserved across the re-window: sum of the three 20s windows == the 60s window
    assert sum(r["prompt_tokens_total"] for r in rows_20) == pytest.approx(rows_60[0]["prompt_tokens_total"])
    assert sum(r["generation_tokens_total"] for r in rows_20) == pytest.approx(rows_60[0]["generation_tokens_total"])


def test_enumerate_windows_sliding_and_tumbling() -> None:
    tumbling = rewindow_from_raw.enumerate_windows(0, 60_000, 20_000, 20_000)
    assert tumbling == [(0, 20_000), (20_000, 40_000), (40_000, 60_000)]
    sliding = rewindow_from_raw.enumerate_windows(0, 40_000, 20_000, 10_000)
    assert sliding == [(0, 20_000), (10_000, 30_000), (20_000, 40_000)]


def test_min_latency_guard_nulls_noisy_p95() -> None:
    # N1 guard parity with MetricsStore: below the sample floor, p95 is None (not 0).
    records = _make_requests()[:5]
    wm = rewindow_from_raw.aggregate_window(
        records, [], MODEL, 1_000, 61_000,
        percentile_mode="bucket_upper", min_latency_samples=10, instant_sample_interval_ms=5_000,
    )
    assert wm.ttft_p95_ms is None
    assert wm.e2e_p95_ms is None
    # tokens are still aggregated (the guard is latency-only)
    assert wm.prompt_tokens == pytest.approx(sum(r["input_tokens"] for r in records))


def test_load_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "i128_o128_c1.jsonl"
    recs = _make_requests()[:3]
    with path.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
        fh.write("\n")  # blank line tolerated
        fh.write("{ not json\n")  # garbage tolerated
    loaded = rewindow_from_raw.load_jsonl(path)
    assert len(loaded) == 3
    assert loaded[0]["cell_id"] == "i512_o128_c8"

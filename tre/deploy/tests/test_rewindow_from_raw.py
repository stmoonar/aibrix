from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from tre_common.rediskeys import SCRAPE_INTERVAL_MS
from tre_common.registry import load_registry
from tre_controller.store.metrics_store import MetricsStore
from scripts import openloop, r3_grid, rewindow_from_raw

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


# --- sidecar cadence: the capture is 1 Hz, the live control path is a 10 s grid --------
def _hz_sidecar(
    *,
    n: int = 60,
    spacing_ms: int = 1_000,
    waiting_at_s: tuple[int, ...] = (),
    running_at_s: tuple[int, ...] = (),
    tagged: bool = True,
) -> list[dict]:
    """``n`` samples ``spacing_ms`` apart from ts 0, non-zero only at the given seconds.

    ``tagged=True`` runs them through ``openloop.mark_live_grid``, i.e. exactly what the
    campaign sidecar carries; ``tagged=False`` reproduces a pre-existing closed-loop
    capture that predates the flag.
    """
    samples = [
        {
            "ts_ms": i * spacing_ms,
            "waiting": 3.0 if i in waiting_at_s else 0.0,
            "running": 2.0 if i in running_at_s else 0.0,
            "swapping": 0.0,
        }
        for i in range(n)
    ]
    return openloop.mark_live_grid(samples) if tagged else samples


def test_live_grid_rejects_capture_cadence_naming_both_numbers() -> None:
    # Hazard 1: --instant-sample-ms is the divisor for the queue average. In live mode the
    # consumed spacing IS the gateway cadence, so the campaign's 1000 would divide by 10x
    # too little. Must fail loudly, naming both numbers and the consequence.
    samples = _hz_sidecar()
    with pytest.raises(rewindow_from_raw.CadenceMismatchError) as exc:
        rewindow_from_raw.resolve_instant_cadence(
            samples, instant_grid="live", instant_sample_ms=1_000
        )
    message = str(exc.value)
    assert "1000" in message
    assert str(SCRAPE_INTERVAL_MS) in message
    assert "10x" in message


def test_live_grid_accepts_scrape_interval() -> None:
    samples = _hz_sidecar()
    assert rewindow_from_raw.resolve_instant_cadence(
        samples, instant_grid="live", instant_sample_ms=SCRAPE_INTERVAL_MS
    ) == SCRAPE_INTERVAL_MS


def test_raw_mode_rejects_spacing_mismatch_on_tagged_sidecar() -> None:
    # The default --instant-sample-ms is the live cadence; pointed at a 1 Hz campaign
    # sidecar in raw mode it would scale every queue average by 10x.
    samples = _hz_sidecar()
    with pytest.raises(rewindow_from_raw.CadenceMismatchError) as exc:
        rewindow_from_raw.resolve_instant_cadence(
            samples, instant_grid="raw", instant_sample_ms=SCRAPE_INTERVAL_MS, source="i512_o128_c8"
        )
    message = str(exc.value)
    assert "1000 ms between samples" in message
    assert f"--instant-sample-ms={SCRAPE_INTERVAL_MS}" in message
    assert "10x" in message
    assert "i512_o128_c8" in message


def test_raw_mode_accepts_matching_cadence() -> None:
    samples = _hz_sidecar()
    assert rewindow_from_raw.resolve_instant_cadence(
        samples, instant_grid="raw", instant_sample_ms=1_000
    ) == 1_000


def test_raw_mode_tolerates_sampler_jitter() -> None:
    # Wall-clock jitter is normal; only an order-of-magnitude mismatch is an error.
    samples = _hz_sidecar(spacing_ms=1_100)
    assert rewindow_from_raw.resolve_instant_cadence(
        samples, instant_grid="raw", instant_sample_ms=1_000
    ) == 1_000


def test_raw_mode_keeps_untagged_sidecar_working() -> None:
    # Pre-existing closed-loop captures carry no on_live_grid field: nothing to check
    # against, so they must keep working (this is the r3_grid sidecar at 10 s).
    samples = _hz_sidecar(n=6, spacing_ms=5_000, tagged=False)
    assert not rewindow_from_raw.sidecar_has_live_grid_tags(samples)
    assert rewindow_from_raw.resolve_instant_cadence(
        samples, instant_grid="raw", instant_sample_ms=SCRAPE_INTERVAL_MS
    ) == SCRAPE_INTERVAL_MS


def test_resolve_instant_cadence_rejects_nonpositive() -> None:
    with pytest.raises(rewindow_from_raw.CadenceMismatchError):
        rewindow_from_raw.resolve_instant_cadence([], instant_grid="raw", instant_sample_ms=0)


def test_observed_sample_spacing_is_median_not_mean() -> None:
    samples = [{"ts_ms": 0}, {"ts_ms": 1_000}, {"ts_ms": 2_000}, {"ts_ms": 60_000}]
    assert rewindow_from_raw.observed_sample_spacing_ms(samples) == pytest.approx(1_000.0)
    assert rewindow_from_raw.observed_sample_spacing_ms([{"ts_ms": 0}]) is None


def test_select_instant_samples_live_keeps_only_grid_samples() -> None:
    samples = _hz_sidecar(n=30)
    live = rewindow_from_raw.select_instant_samples(samples, "live")
    assert [s["ts_ms"] for s in live] == [0, 10_000, 20_000]
    assert len(rewindow_from_raw.select_instant_samples(samples, "raw")) == 30
    with pytest.raises(ValueError):
        rewindow_from_raw.select_instant_samples(samples, "grid")


# --- observability gap -------------------------------------------------------------
def test_observability_gap_all_bursts_missed() -> None:
    # Queue spikes at t=5s and t=15s fall strictly between live-grid samples: the 1 Hz
    # capture sees two 10 s windows cross, the controller's grid sees none.
    samples = _hz_sidecar(waiting_at_s=(5, 15))
    gap = rewindow_from_raw.observability_gap(samples, window_ms=10_000)
    assert (gap.raw_crossings, gap.live_crossings) == (2, 0)
    assert gap.total_windows == 5  # span 0..59s -> five whole 10s windows
    assert gap.gap == pytest.approx(1.0)
    assert gap.as_dict()["observability_gap"] == pytest.approx(1.0)


def test_observability_gap_partial() -> None:
    # t=20s IS a live-grid sample, so one of the three crossings is observed.
    samples = _hz_sidecar(waiting_at_s=(5, 15, 20))
    gap = rewindow_from_raw.observability_gap(samples, window_ms=10_000)
    assert (gap.raw_crossings, gap.live_crossings) == (3, 1)
    assert gap.gap == pytest.approx(1.0 - 1.0 / 3.0)


def test_observability_gap_zero_when_nothing_to_miss() -> None:
    gap = rewindow_from_raw.observability_gap(_hz_sidecar(), window_ms=10_000)
    assert gap.raw_crossings == 0
    assert gap.gap == 0.0


def test_observability_gap_accepts_any_sidecar_key() -> None:
    samples = _hz_sidecar(waiting_at_s=(5,), running_at_s=(25,))
    on_running = rewindow_from_raw.observability_gap(samples, window_ms=10_000, key="running")
    assert (on_running.raw_crossings, on_running.live_crossings) == (1, 0)
    assert on_running.key == "running"
    # threshold above the sample value -> no crossing at all
    high = rewindow_from_raw.observability_gap(samples, window_ms=10_000, threshold=5.0)
    assert (high.raw_crossings, high.gap) == (0, 0.0)


def test_observability_gap_reuses_openloop_windowing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Guard against a private re-implementation drifting from the capture's own definition
    # of the live grid: the metric must go through openloop.windows_observing.
    calls: list[dict] = []
    real = openloop.windows_observing

    def spy(samples, **kwargs):
        calls.append(kwargs)
        return real(samples, **kwargs)

    monkeypatch.setattr(openloop, "windows_observing", spy)
    rewindow_from_raw.observability_gap(_hz_sidecar(waiting_at_s=(5,)), window_ms=10_000)
    assert [c["grid_only"] for c in calls] == [False, True]


def test_combine_observability_gaps_pools_counts() -> None:
    a = rewindow_from_raw.observability_gap(_hz_sidecar(waiting_at_s=(5, 15)), window_ms=10_000)
    b = rewindow_from_raw.observability_gap(_hz_sidecar(waiting_at_s=(20,)), window_ms=10_000)
    pooled = rewindow_from_raw.combine_observability_gaps([a, b])
    assert (pooled.raw_crossings, pooled.live_crossings) == (3, 1)
    assert pooled.gap == pytest.approx(1.0 - 1.0 / 3.0)
    assert rewindow_from_raw.combine_observability_gaps([]) is None


# --- live-grid re-window + meta provenance -----------------------------------------
def test_rewindow_cell_live_grid_uses_only_grid_samples() -> None:
    # theta is a threshold on the signal the controller consumes, so a live-grid fit must
    # average the 10 s subsample (/6 over a 60 s window), not the 1 Hz stream.
    registry = load_registry(str(REGISTRY_PATH))
    spec = registry.model(MODEL)
    cell = r3_grid.GridCell.from_scenario_id("i512_o128_c8")
    samples = _hz_sidecar(waiting_at_s=(5, 20))

    live_rows = rewindow_from_raw.rewindow_cell(
        _make_requests(), samples, cell, spec,
        window_ms=60_000, step_ms=60_000, percentile_mode="bucket_upper",
        min_latency_samples=0, instant_sample_interval_ms=SCRAPE_INTERVAL_MS,
        instant_grid="live", start_ms=0, end_ms=60_000,
    )
    raw_rows = rewindow_from_raw.rewindow_cell(
        _make_requests(), samples, cell, spec,
        window_ms=60_000, step_ms=60_000, percentile_mode="bucket_upper",
        min_latency_samples=0, instant_sample_interval_ms=1_000,
        instant_grid="raw", start_ms=0, end_ms=60_000,
    )
    # live: only the t=20s spike is on the grid -> 3.0 / (60000/10000)
    assert live_rows[0]["avg_waiting"] == pytest.approx(3.0 / 6)
    # raw: both spikes, divided by 60 expected samples
    assert raw_rows[0]["avg_waiting"] == pytest.approx(6.0 / 60)


def test_rewindow_cell_fails_loudly_on_cadence_mismatch() -> None:
    registry = load_registry(str(REGISTRY_PATH))
    spec = registry.model(MODEL)
    cell = r3_grid.GridCell.from_scenario_id("i512_o128_c8")
    with pytest.raises(rewindow_from_raw.CadenceMismatchError):
        rewindow_from_raw.rewindow_cell(
            _make_requests(), _hz_sidecar(), cell, spec,
            window_ms=60_000, step_ms=60_000, percentile_mode="bucket_upper",
            min_latency_samples=0, instant_sample_interval_ms=SCRAPE_INTERVAL_MS,
            instant_grid="raw", start_ms=0, end_ms=60_000,
        )


def test_meta_path_sits_next_to_the_csv(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "rewindow_20s.csv"
    assert rewindow_from_raw.meta_path_for(out) == tmp_path / "sub" / "rewindow_20s.meta.json"


def test_build_meta_records_cadence_and_gap() -> None:
    gap = rewindow_from_raw.observability_gap(_hz_sidecar(waiting_at_s=(5,)), window_ms=10_000)
    meta = rewindow_from_raw.build_meta(
        model=MODEL, raw_dir="/data/raw", cells=["i512_o128_c8"],
        window_ms=20_000, step_ms=20_000, instant_sample_ms=1_000, instant_grid="raw",
        percentile_mode="bucket_upper", min_latency_samples=10,
        routable_pods=1, assigned_replicas=1, rows=3,
        gap_per_cell={"i512_o128_c8": gap}, gap_overall=gap,
        git_sha="deadbee", generated_at="2026-01-01T00:00:00+00:00",
    )
    assert meta["instant_sample_ms"] == 1_000
    assert meta["instant_grid"] == "raw"
    assert meta["live_grid_ms"] == SCRAPE_INTERVAL_MS
    assert meta["git_short_sha"] == "deadbee"
    assert meta["observability_gap"]["overall"]["raw_crossings"] == 1
    assert "i512_o128_c8" in meta["observability_gap"]["per_cell"]
    json.dumps(meta)  # must be serialisable as written


def _write_campaign_raw(raw_dir: Path, cell_id: str = "i512_o128_c8") -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with (raw_dir / f"{cell_id}.jsonl").open("w", encoding="utf-8") as fh:
        for r in _make_requests():
            fh.write(json.dumps(r) + "\n")
    with (raw_dir / f"{cell_id}.instant.jsonl").open("w", encoding="utf-8") as fh:
        for s in _hz_sidecar(waiting_at_s=(5, 15)):
            fh.write(json.dumps(s) + "\n")


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["rewindow_from_raw.py", *argv])
    return rewindow_from_raw.main()


def test_main_writes_meta_json_beside_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    raw_dir = tmp_path / "raw"
    _write_campaign_raw(raw_dir)
    out = tmp_path / "out" / "rewindow_20s.csv"
    rc = _run_main(monkeypatch, [
        "--model", MODEL, "--raw-dir", str(raw_dir), "--output", str(out),
        "--window-ms", "20000", "--instant-sample-ms", "1000", "--instant-grid", "raw",
        "--min-latency-samples", "0", "--registry", str(REGISTRY_PATH),
    ])
    assert rc == 0
    assert out.exists()
    meta = json.loads((tmp_path / "out" / "rewindow_20s.meta.json").read_text(encoding="utf-8"))
    assert meta["instant_sample_ms"] == 1_000
    assert meta["instant_grid"] == "raw"
    assert meta["window_ms"] == 20_000 and meta["step_ms"] == 20_000
    assert meta["model"] == MODEL
    assert meta["raw_dir"] == str(raw_dir)
    assert meta["cells"] == ["i512_o128_c8"]
    assert meta["percentile_mode"] == "bucket_upper"
    assert meta["min_latency_samples"] == 0
    assert meta["routable_pods"] == 1 and meta["assigned_replicas"] == 1
    assert meta["generated_at_utc"].endswith("+00:00")
    assert "git_short_sha" in meta
    assert meta["observability_gap"]["overall"]["observability_gap"] == pytest.approx(1.0)
    assert meta["observability_gap"]["per_cell"]["i512_o128_c8"]["key"] == "waiting"
    # the cadence is on stdout too, so a campaign log records it without opening the json
    printed = capsys.readouterr().out
    assert "--instant-grid raw --instant-sample-ms 1000" in printed


def test_main_refuses_mismatched_cadence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_dir = tmp_path / "raw"
    _write_campaign_raw(raw_dir)
    with pytest.raises(rewindow_from_raw.CadenceMismatchError):
        _run_main(monkeypatch, [
            "--model", MODEL, "--raw-dir", str(raw_dir),
            "--output", str(tmp_path / "out" / "bad.csv"),
            "--window-ms", "20000", "--instant-grid", "live", "--instant-sample-ms", "1000",
            "--registry", str(REGISTRY_PATH),
        ])


def test_main_live_grid_records_gap_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_dir = tmp_path / "raw"
    _write_campaign_raw(raw_dir)
    out = tmp_path / "out" / "rewindow_live.csv"
    rc = _run_main(monkeypatch, [
        "--model", MODEL, "--raw-dir", str(raw_dir), "--output", str(out),
        "--window-ms", "20000", "--instant-grid", "live",
        "--instant-sample-ms", str(SCRAPE_INTERVAL_MS),
        "--gap-threshold", "5.0", "--min-latency-samples", "0",
        "--registry", str(REGISTRY_PATH),
    ])
    assert rc == 0
    meta = json.loads((tmp_path / "out" / "rewindow_live.meta.json").read_text(encoding="utf-8"))
    assert meta["instant_grid"] == "live"
    assert meta["instant_sample_ms"] == SCRAPE_INTERVAL_MS
    gap = meta["observability_gap"]["overall"]
    assert gap["threshold"] == pytest.approx(5.0)
    assert gap["raw_crossings"] == 0  # waiting never exceeds 5 -> nothing to miss
    assert gap["observability_gap"] == 0.0


def test_the_materialised_prompt_file_is_not_mistaken_for_a_cell(tmp_path) -> None:
    """A cell writes its prompts next to its raw capture and they end in .jsonl too.
    Re-windowing one as if it held per-request measurements would invent a cell out of
    the load generator's own input."""
    (tmp_path / "i256_o128_c60.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "i256_o128_c60.instant.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "i256_o128_c60.failures.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "i256_o128_c60.prompts.jsonl").write_text("{}\n", encoding="utf-8")

    kept, _skipped = rewindow_from_raw.discover_cell_files(tmp_path)
    assert [path.name for path in kept] == ["i256_o128_c60.jsonl"]


def test_the_prompt_file_name_the_replayer_writes_is_the_one_excluded() -> None:
    """The two ends of the exclusion are in different packages; a drift would put the
    prompts back into the fit."""
    from tre_replayer.engine import prompt_store

    assert prompt_store.PROMPT_FILE_SUFFIX in rewindow_from_raw.SIDECAR_JSONL_SUFFIXES

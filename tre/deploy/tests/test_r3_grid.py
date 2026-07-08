from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts import r3_grid


def test_enumerate_cells_cartesian_product() -> None:
    cells = r3_grid.enumerate_cells([128, 512], [128], [1, 4])
    ids = [c.scenario_id for c in cells]
    assert ids == ["i128_o128_c1", "i128_o128_c4", "i512_o128_c1", "i512_o128_c4"]
    assert cells[0].scenario_family == "i128_o128"


@dataclass
class _FakeWindow:
    window_start_ms: int
    window_end_ms: int
    prompt_tokens: float
    generation_tokens: float
    ttft_p95_ms: float
    tpot_p95_ms: float
    avg_waiting: float = 0.0
    avg_running: float = 0.0
    avg_swapping: float = 0.0
    e2e_p95_ms: float = 0.0


def test_window_row_maps_calibration_columns() -> None:
    cell = r3_grid.GridCell(512, 128, 8)
    wm = _FakeWindow(1000, 61000, 4096.0, 1024.0, 480.0, 55.0,
                     avg_waiting=1.5, avg_running=3.0, avg_swapping=0.5, e2e_p95_ms=9000.0)
    row = r3_grid.window_row(cell, wm, trs=734.5, queue_control=7.4375)
    assert row["scenario_id"] == "i512_o128_c8"
    assert row["scenario_family"] == "i512_o128"
    assert row["prompt_tokens_total"] == 4096.0
    assert row["generation_tokens_total"] == 1024.0
    assert row["avg_waiting"] == 1.5
    assert row["avg_running"] == 3.0
    assert row["avg_swapping"] == 0.5
    assert row["queue_control"] == 7.4375
    assert row["p95_ttft"] == 480.0
    assert row["p95_tpot"] == 55.0
    assert row["p95_e2e"] == 9000.0
    assert row["trs"] == 734.5
    assert set(row.keys()) == set(r3_grid.CSV_COLUMNS)


def test_write_csv_roundtrip(tmp_path: Path) -> None:
    import csv
    cell = r3_grid.GridCell(128, 128, 1)
    wm = _FakeWindow(0, 60000, 1.0, 2.0, 3.0, 4.0)
    out = tmp_path / "grid.csv"
    r3_grid.write_csv([r3_grid.window_row(cell, wm, 10.0, queue_control=1.0)], out)
    reader = list(csv.DictReader(out.open()))
    assert reader[0]["scenario_id"] == "i128_o128_c1"
    assert reader[0]["trs"] == "10.0"
    assert reader[0]["queue_control"] == "1.0"


def test_compute_window_trs_uses_time_constant_ema() -> None:
    # S1.4: the trs column must use the live shared time-constant EMA (ema_tau_ms +
    # window_end_ms deltas), not the old fixed-alpha EMA, so theta is fit on the live signal.
    import math
    from types import SimpleNamespace

    trs = SimpleNamespace(
        w_p=0.04, w_d=1.0, lambda_wait=2.625, qmin=1.0,
        ema_alpha=0.5, ema_tau_ms=20000, theta_m=100.0,
    )
    spec = SimpleNamespace(trs=trs)

    def win(gen: float, running: float, end: int) -> SimpleNamespace:
        return SimpleNamespace(
            prompt_tokens=0.0, generation_tokens=gen, avg_waiting=0.0, avg_running=running,
            avg_swapping=0.0, routable_pods=1, assigned_replicas=1, kv_cache_hit_rate=0.0,
            window_end_ms=end,
        )

    windows = [win(200.0, 2.0, 30000), win(400.0, 2.0, 60000)]  # raw TRS 100 then 200
    results = r3_grid.compute_window_results(windows, spec)
    vals = [r.TRS for r in results]
    assert vals[0] == 100.0  # first window seeds from raw
    decay = math.exp(-30000 / 20000)  # dt=30000 (tumbling), tau=20000
    expected = decay * 100.0 + (1 - decay) * 200.0
    assert abs(vals[1] - expected) < 1e-6
    assert abs(vals[1] - 200.0) > 1.0  # not raw
    assert abs(vals[1] - 150.0) > 1.0  # not the old fixed-alpha(0.5) value
    # Q_ctl is exposed for the queue_control CSV column (S2/S3): running=2, waiting=0 -> 2.0
    assert results[0].Q_ctl == 2.0


def test_checkpoint_resume(tmp_path: Path) -> None:
    p = tmp_path / "ck.json"
    ck = r3_grid.Checkpoint.load(p)
    cell = r3_grid.GridCell(128, 128, 1)
    assert not ck.is_done(cell)
    ck.mark(cell)
    # reload from disk -> resumes
    ck2 = r3_grid.Checkpoint.load(p)
    assert ck2.is_done(cell)


# --- S4 raw logging (doc15 §4) -----------------------------------------------------------
from types import SimpleNamespace


def _stream_result(status=200, first_token_ms=120.0, done_ms=5000.0, prompt_tokens=130, completion_tokens=64):
    return SimpleNamespace(
        status=status, first_token_ms=first_token_ms, done_ms=done_ms,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
    )


def test_from_scenario_id_roundtrip() -> None:
    cell = r3_grid.GridCell(512, 128, 8)
    assert r3_grid.GridCell.from_scenario_id(cell.scenario_id) == cell
    import pytest
    with pytest.raises(ValueError):
        r3_grid.GridCell.from_scenario_id("i512_o128_c8.instant")


def test_build_raw_record_maps_schema() -> None:
    res = _stream_result(first_token_ms=120.0, done_ms=5000.0, prompt_tokens=130, completion_tokens=64)
    rec = r3_grid.build_raw_record("i512_o128_c8", send_ts_ms=1_000, res=res)
    assert set(rec.keys()) == set(r3_grid.RAW_COLUMNS)
    assert rec["send_ts_ms"] == 1_000
    assert rec["recv_first_token_ts_ms"] == 1_120
    assert rec["done_ts_ms"] == 6_000
    assert rec["input_tokens"] == 130
    assert rec["output_tokens"] == 64
    assert rec["ttft_ms"] == 120.0
    assert rec["e2e_ms"] == 5000.0
    # tpot = (e2e - ttft) / (completion - 1)
    assert rec["tpot_ms"] == (5000.0 - 120.0) / (64 - 1)
    assert rec["http_status"] == 200
    assert rec["cell_id"] == "i512_o128_c8"


def test_build_raw_record_nulls_when_unavailable() -> None:
    # error/no-usage response: durations + tokens genuinely unavailable -> null, not faked.
    res = _stream_result(status=0, first_token_ms=None, done_ms=4200.0, prompt_tokens=None, completion_tokens=None)
    rec = r3_grid.build_raw_record("i128_o128_c1", send_ts_ms=500, res=res)
    assert rec["ttft_ms"] is None
    assert rec["recv_first_token_ts_ms"] is None
    assert rec["done_ts_ms"] == 4_700
    assert rec["input_tokens"] is None
    assert rec["output_tokens"] is None
    assert rec["tpot_ms"] is None  # needs ttft + completion>1


def test_build_raw_record_tpot_null_single_token() -> None:
    res = _stream_result(first_token_ms=100.0, done_ms=100.0, completion_tokens=1)
    rec = r3_grid.build_raw_record("i128_o128_c1", send_ts_ms=0, res=res)
    assert rec["tpot_ms"] is None


def test_drive_cell_writes_raw_jsonl(tmp_path) -> None:
    import json as _json
    import time as _time

    calls = {"n": 0}

    def fake_stream_call(url, headers, body, timeout):
        calls["n"] += 1
        _time.sleep(0.01)
        return _stream_result()

    raw_path = tmp_path / "i512_o128_c8.jsonl"
    cell = r3_grid.GridCell(512, 128, 8)
    start_ms, end_ms = r3_grid.drive_cell(
        "http://gw", "dsqwen-7b", cell, duration_s=0.05,
        raw_path=raw_path, stream_call=fake_stream_call,
    )
    assert end_ms >= start_ms
    lines = [l for l in raw_path.read_text().splitlines() if l.strip()]
    assert lines  # at least one request logged
    rec = _json.loads(lines[0])
    assert set(rec.keys()) == set(r3_grid.RAW_COLUMNS)
    assert rec["cell_id"] == "i512_o128_c8"
    assert rec["input_tokens"] == 130


def test_drive_cell_writes_instant_sidecar(tmp_path) -> None:
    import json as _json

    def fake_stream_call(url, headers, body, timeout):
        return _stream_result()

    def fake_sampler(now):
        return {"waiting": 2.0, "running": 5.0, "swapping": 1.0}

    raw_path = tmp_path / "i128_o128_c1.jsonl"
    instant_path = tmp_path / "i128_o128_c1.instant.jsonl"
    cell = r3_grid.GridCell(128, 128, 1)
    r3_grid.drive_cell(
        "http://gw", "dsqwen-7b", cell, duration_s=0.06,
        raw_path=raw_path, instant_path=instant_path,
        instant_sampler=fake_sampler, instant_interval_s=0.02, stream_call=fake_stream_call,
    )
    inst_lines = [l for l in instant_path.read_text().splitlines() if l.strip()]
    assert inst_lines
    snap = _json.loads(inst_lines[0])
    assert set(snap.keys()) == {"ts_ms", "waiting", "running", "swapping"}
    assert snap["running"] == 5.0


def test_make_live_instant_sampler_reads_latest_not_window_average() -> None:
    # The sidecar sampler must delegate to store.read_latest_instant (freshest bucket),
    # NOT read_model_window (which would zero-out on the freshness gap or halve the value).
    # r3 SMOKE_FINDINGS defect 1.
    class _FakeStore:
        def __init__(self):
            self.latest_calls = []
            self.window_calls = 0

        def read_latest_instant(self, model, now_ms, lookback_ms):
            self.latest_calls.append((model, now_ms, lookback_ms))
            return {"waiting": 2.0, "running": 6.0, "swapping": 0.0}

        def read_model_window(self, *a, **k):  # must NOT be used by the sampler
            self.window_calls += 1
            raise AssertionError("sampler must not use read_model_window")

    store = _FakeStore()
    sampler = r3_grid._make_live_instant_sampler(store, "dsqwen-7b", lookback_ms=20_000)
    snap = sampler(123_000)
    assert snap == {"waiting": 2.0, "running": 6.0, "swapping": 0.0}
    assert store.window_calls == 0
    assert store.latest_calls == [("dsqwen-7b", 123_000, 20_000)]


def test_r3_grid_uses_single_source_scrape_cadence() -> None:
    # r3_grid must derive the instant-sample / expected_samples cadence from the single
    # source of truth (tre_common.rediskeys.SCRAPE_INTERVAL_MS = 10s gateway write
    # cadence), not a local magic number (r3 SMOKE_FINDINGS defect 2).
    from tre_common.rediskeys import SCRAPE_INTERVAL_MS

    assert r3_grid.SCRAPE_INTERVAL_MS is SCRAPE_INTERVAL_MS
    assert SCRAPE_INTERVAL_MS == 10_000


def test_estimate_capture_bytes_scales_with_grid() -> None:
    cells = r3_grid.enumerate_cells([128], [128], [1, 2])
    est = r3_grid.estimate_capture_bytes(cells, cell_seconds=60.0, assumed_rps_per_worker=2.0)
    # (1 + 2) workers * 60s * 2 rps * 200 bytes
    assert est == int((1 + 2) * 60.0 * 2.0 * r3_grid.RAW_BYTES_PER_REQUEST)

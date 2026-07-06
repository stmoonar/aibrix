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


def test_window_row_maps_calibration_columns() -> None:
    cell = r3_grid.GridCell(512, 128, 8)
    wm = _FakeWindow(1000, 61000, 4096.0, 1024.0, 480.0, 55.0)
    row = r3_grid.window_row(cell, wm, trs=734.5)
    assert row["scenario_id"] == "i512_o128_c8"
    assert row["scenario_family"] == "i512_o128"
    assert row["prompt_tokens_total"] == 4096.0
    assert row["generation_tokens_total"] == 1024.0
    assert row["p95_ttft"] == 480.0
    assert row["p95_tpot"] == 55.0
    assert row["trs"] == 734.5
    assert set(row.keys()) == set(r3_grid.CSV_COLUMNS)


def test_write_csv_roundtrip(tmp_path: Path) -> None:
    import csv
    cell = r3_grid.GridCell(128, 128, 1)
    wm = _FakeWindow(0, 60000, 1.0, 2.0, 3.0, 4.0)
    out = tmp_path / "grid.csv"
    r3_grid.write_csv([r3_grid.window_row(cell, wm, 10.0)], out)
    reader = list(csv.DictReader(out.open()))
    assert reader[0]["scenario_id"] == "i128_o128_c1"
    assert reader[0]["trs"] == "10.0"


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
    vals = r3_grid.compute_window_trs(windows, spec)
    assert vals[0] == 100.0  # first window seeds from raw
    decay = math.exp(-30000 / 20000)  # dt=30000 (tumbling), tau=20000
    expected = decay * 100.0 + (1 - decay) * 200.0
    assert abs(vals[1] - expected) < 1e-6
    assert abs(vals[1] - 200.0) > 1.0  # not raw
    assert abs(vals[1] - 150.0) > 1.0  # not the old fixed-alpha(0.5) value


def test_checkpoint_resume(tmp_path: Path) -> None:
    p = tmp_path / "ck.json"
    ck = r3_grid.Checkpoint.load(p)
    cell = r3_grid.GridCell(128, 128, 1)
    assert not ck.is_done(cell)
    ck.mark(cell)
    # reload from disk -> resumes
    ck2 = r3_grid.Checkpoint.load(p)
    assert ck2.is_done(cell)

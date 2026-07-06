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


def test_checkpoint_resume(tmp_path: Path) -> None:
    p = tmp_path / "ck.json"
    ck = r3_grid.Checkpoint.load(p)
    cell = r3_grid.GridCell(128, 128, 1)
    assert not ck.is_done(cell)
    ck.mark(cell)
    # reload from disk -> resumes
    ck2 = r3_grid.Checkpoint.load(p)
    assert ck2.is_done(cell)

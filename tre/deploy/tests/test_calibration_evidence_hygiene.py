"""Guards on which evidence is allowed to reach a theta fit.

Everything here is a rule about *discarding* data. Each one is separately testable and
separately silent when it breaks: a cell that should have been voided and was not simply
contributes windows, and theta moves with no error anywhere. So each rule gets its own
test naming the bias it prevents.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import gen_calibration_schedules as gen
from scripts import r3_grid, rewindow_from_raw


# ------------------------------------------------------- window rows carry the verdict


def test_the_window_csv_has_somewhere_to_record_a_model_error() -> None:
    # Without these columns a window whose slowest requests all errored out reads as a
    # comfortable p95 and is scored healthy, which pulls theta towards health.
    assert "model_errors" in r3_grid.CSV_COLUMNS
    assert "slo_violated" in r3_grid.CSV_COLUMNS


def test_a_window_row_defaults_to_no_errors_seen() -> None:
    class _WM:
        window_start_ms = 0
        window_end_ms = 30000
        prompt_tokens = 10
        generation_tokens = 20
        avg_waiting = 0.0
        avg_running = 1.0
        avg_swapping = 0.0
        ttft_p95_ms = 100.0
        tpot_p95_ms = 10.0
        e2e_p95_ms = 200.0

    row = r3_grid.window_row(
        r3_grid.GridCell(256, 128, 95), _WM(), 1.0, 1.0, client=_WM(), server=None
    )
    assert row["model_errors"] == 0 and row["client_timeouts"] == 0
    # No label until a caller that knows the SLO applies one - never a default False.
    assert row["slo_label"] is None and row["slo_violated"] is None
    assert set(row) == set(r3_grid.CSV_COLUMNS)


def test_a_marked_window_counts_as_an_slo_crossing() -> None:
    rows = [
        {"p95_ttft_client_ms": 50.0, "p95_tpot_client_ms": 5.0, "model_errors": 1},
        {"p95_ttft_client_ms": 50.0, "p95_tpot_client_ms": 5.0, "client_timeouts": 2},
        {"p95_ttft_client_ms": 50.0, "p95_tpot_client_ms": 5.0},
        {"p95_ttft_client_ms": 900.0, "p95_tpot_client_ms": 5.0},
        # unlabeled: neither a crossing nor healthy
        {"p95_ttft_client_ms": None, "p95_tpot_client_ms": None},
    ]
    assert r3_grid.count_slo_windows(rows, ttft_slo_ms=500.0, tpot_slo_ms=75.0) == 3


# ------------------------------------------------------------------ hold cell ids


def test_a_hold_schedule_names_its_own_rho_in_its_cell_id(tmp_path: Path) -> None:
    # The boundary search decides a probe's rho at campaign time, so there is no fixed
    # load code to look up; it travels in the file name instead and must still parse.
    class _Seg:
        input_tokens = 256
        max_output_tokens = 128

    cell_id = r3_grid._cell_id_from_schedule(
        Path("S1_hold1087.json"), "dsqwen-7b", [_Seg()]
    )
    assert cell_id == f"i256_o128_c{gen.hold_load_code(0.87)}" == "i256_o128_c1087"
    assert r3_grid.GridCell.from_scenario_id(cell_id).concurrency == 1087


def test_an_unnameable_schedule_still_refuses_to_guess() -> None:
    class _Seg:
        input_tokens = 256
        max_output_tokens = 128

    with pytest.raises(SystemExit, match="pass --cell-id"):
        r3_grid._cell_id_from_schedule(Path("S1_wobble.json"), "dsqwen-7b", [_Seg()])


def test_the_committed_primitives_keep_their_fixed_load_codes(tmp_path: Path) -> None:
    class _Seg:
        input_tokens = 2048
        max_output_tokens = 96

    assert r3_grid._cell_id_from_schedule(
        Path("S3_ramp.json"), "dsqwen-7b", [_Seg()]
    ) == f"i2048_o96_c{gen.LOAD_CODE['ramp']}"


# ------------------------------------------------------------- held-out data at the fit


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"send_ts_ms": 1}\n', encoding="utf-8")


def test_raw_cells_are_discovered_recursively(tmp_path: Path) -> None:
    # The campaign writes one directory per cell under the raw root. A flat glob finds
    # nothing there and produces an empty CSV without saying so.
    _touch(tmp_path / "dsqwen-7b_S1_steps" / "i256_o128_c95.jsonl")
    _touch(tmp_path / "dsqwen-7b_S3_ramp" / "i2048_o96_c120.jsonl")
    _touch(tmp_path / "dsqwen-7b_S1_steps" / "i256_o128_c95.instant.jsonl")
    _touch(tmp_path / "dsqwen-7b_S1_steps" / "i256_o128_c95.failures.jsonl")

    kept, skipped = rewindow_from_raw.discover_cell_files(tmp_path)
    assert sorted(p.stem for p in kept) == ["i2048_o96_c120", "i256_o128_c95"]
    # the sidecar and the classified-failure logs are not cells
    assert skipped == []


def test_the_held_out_cells_are_excluded_from_the_fitting_set(tmp_path: Path) -> None:
    # The single structural guarantee that the validation shape never trains the fit.
    _touch(tmp_path / "a" / "i256_o128_c95.jsonl")
    _touch(tmp_path / "b" / "i0_o0_c95.jsonl")
    index = {"schedules": [
        {"cell_id": "i0_o0_c95", "held_out": True, "shape": gen.MIXTURE_NAME},
        {"cell_id": "i256_o128_c95", "held_out": False, "shape": "S1"},
    ]}
    excluded = rewindow_from_raw.held_out_cell_ids(index)
    assert excluded == {"i0_o0_c95"}

    kept, skipped = rewindow_from_raw.discover_cell_files(tmp_path, exclude=excluded)
    assert [p.stem for p in kept] == ["i256_o128_c95"]
    assert skipped == ["i0_o0_c95"]


def test_only_cell_id_builds_the_validation_set_from_exactly_those_cells(
    tmp_path: Path,
) -> None:
    _touch(tmp_path / "a" / "i256_o128_c95.jsonl")
    _touch(tmp_path / "b" / "i0_o0_c95.jsonl")
    kept, _skipped = rewindow_from_raw.discover_cell_files(
        tmp_path, exclude={"i0_o0_c95"}, only=["i0_o0_c95"]
    )
    # an explicit inclusion list is a stronger statement than a default exclusion
    assert [p.stem for p in kept] == ["i0_o0_c95"]


def test_the_committed_index_marks_the_held_out_cells_so_the_fit_can_find_them() -> None:
    root = Path(__file__).resolve().parents[2]
    index = json.loads(
        (root / "replayer" / "traces_v2" / "calibration" / "INDEX.json").read_text(
            encoding="utf-8"
        )
    )
    excluded = rewindow_from_raw.held_out_cell_ids(index)
    assert excluded
    held_shapes = {
        entry["shape"] for entry in index["schedules"]
        if entry["cell_id"] in excluded
    }
    assert held_shapes == {gen.MIXTURE_NAME}


# --------------------------------------------- a voided capture stays out of the fit


def test_the_window_csv_has_somewhere_to_record_an_unserved_request() -> None:
    # Separate from model_errors: a connection that died under a request is not the
    # engine failing, and a window must not be able to claim it was.
    assert "proxy_transient_errors" in r3_grid.CSV_COLUMNS
    assert "model_errors" in r3_grid.CSV_COLUMNS


def test_a_quarantined_capture_falls_outside_the_fitting_glob(tmp_path: Path) -> None:
    # The fitting re-window globs every <cell>.jsonl under the raw root and filters only
    # by cell id, so a voided capture left in place is silently re-windowed - and a
    # re-run, writing the same cell id again, would pool both attempts into one fit.
    kept_file = tmp_path / "i256_o128_c95.jsonl"
    kept_file.write_text("{}\n", encoding="utf-8")
    voided = tmp_path / ("i256_o128_c99.jsonl" + r3_grid.VOID_RAW_SUFFIX)
    voided.write_text("{}\n", encoding="utf-8")

    kept, _ = rewindow_from_raw.discover_cell_files(tmp_path)
    assert [p.name for p in kept] == ["i256_o128_c95.jsonl"]

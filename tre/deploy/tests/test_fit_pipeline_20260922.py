"""Labels and pipeline fixes of plan 2026-09-21 §6.3 (B2, B3, B5) on the deploy side."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts import openloop, r3_grid, rewindow_from_raw as rw
from tre_common.registry import load_registry

TRE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = TRE_ROOT / "deploy" / "registry.yaml"
MODEL = "dsqwen-7b"


# ------------------------------------------------------------------------------ B2


def test_a_client_timeout_marks_its_window_violated_in_its_own_column() -> None:
    rows = [
        {"window_start_ms": 0, "window_end_ms": 1000},
        {"window_start_ms": 1000, "window_end_ms": 2000},
    ]
    live = {"http_status": 0, "error": "TimeoutError", "client_timeout": True, "actual_send_ts_ms": 1500}
    marked = openloop.mark_unserved_request_windows(rows, [live])
    assert [r["slo_violated"] for r in marked] == [False, True]
    assert [r["client_timeouts"] for r in marked] == [0, 1]
    assert [r["model_errors"] for r in marked] == [0, 0]


def test_a_persisted_failure_record_is_judged_by_its_recorded_class() -> None:
    # .failures.jsonl rows drop the sender-side client_timeout flag; re-classifying one
    # (http_status 0, no flag) would call it a model error. The recorded verdict wins.
    rows = [{"window_start_ms": 0, "window_end_ms": 1000}]
    persisted = {"send_ts_ms": 500, "http_status": 0, "error": "TimeoutError",
                 "failure_class": "client_timeout", "outcome": "client_timeout"}
    [row] = openloop.mark_unserved_request_windows(rows, [persisted])
    assert row["client_timeouts"] == 1 and row["model_errors"] == 0 and row["slo_violated"]


def test_csv_has_a_client_timeouts_column() -> None:
    assert "client_timeouts" in r3_grid.CSV_COLUMNS


def _write_cell(dirpath: Path, cell_id: str, *, with_failure: bool) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    t0 = 1_000_000
    with (dirpath / f"{cell_id}.jsonl").open("w") as fh:
        for i in range(60):
            send = t0 + i * 1000
            fh.write(json.dumps({
                "send_ts_ms": send, "recv_first_token_ts_ms": send + 50, "done_ts_ms": send + 900,
                "input_tokens": 256, "output_tokens": 128, "ttft_ms": 50.0, "tpot_ms": 7.0,
                "e2e_ms": 900.0, "http_status": 200, "cell_id": cell_id, "target_pod": None,
            }) + "\n")
    with (dirpath / f"{cell_id}.instant.jsonl").open("w") as fh:
        for i in range(0, 61, 10):
            fh.write(json.dumps({"ts_ms": t0 + i * 1000, "waiting": 0.0, "running": 2.0,
                                 "swapping": 0.0, "on_live_grid": True}) + "\n")
    if with_failure:
        with (dirpath / f"{cell_id}.failures.jsonl").open("w") as fh:
            fh.write(json.dumps({"send_ts_ms": t0 + 45_000, "http_status": 0,
                                 "failure_class": "client_timeout"}) + "\n")


def test_rewindow_reads_the_failures_sidecar_and_marks_the_windows(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_cell(raw / f"{MODEL}_S3_steps", "i2048_o96_c95", with_failure=True)
    out = tmp_path / "fit.csv"
    import sys
    from unittest import mock

    argv = ["rewindow_from_raw", "--model", MODEL, "--raw-dir", str(raw), "--output", str(out),
            "--window-ms", "30000", "--step-ms", "5000", "--instant-grid", "live",
            "--instant-sample-ms", "10000", "--min-latency-samples", "0", "--registry", str(REGISTRY)]
    with mock.patch.object(sys, "argv", argv):
        assert rw.main() == 0
    rows = list(csv.DictReader(out.open()))
    violated = [r for r in rows if r["slo_violated"] == "True"]
    assert violated, "the timed-out request's windows must be violated"
    assert all(int(r["client_timeouts"]) == 1 for r in violated)
    starts = {int(r["window_start_ms"]) for r in violated}
    assert all(s <= 1_045_000 < s + 30_000 for s in starts)


# ------------------------------------------------------------------------------ B3


def test_only_shape_selects_cell_directories_from_disk_hold_cells_included(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_cell(raw / f"{MODEL}_S3_ramp", "i2048_o96_c120", with_failure=False)
    _write_cell(raw / f"{MODEL}_S3_S3_hold1060_a1", "i2048_o96_c1060", with_failure=False)
    _write_cell(raw / f"{MODEL}_S4_steps", "i256_o448_c95", with_failure=False)
    _write_cell(raw / f"{MODEL}_M_steps", "i0_o0_c95", with_failure=False)
    kept, skipped = rw.discover_cell_files(raw, only_shapes=["S3", "T8"], model=MODEL, exclude=["i0_o0_c95"])
    assert sorted(p.stem for p in kept) == ["i2048_o96_c1060", "i2048_o96_c120"]
    assert "i256_o448_c95" in skipped
    assert rw.raw_dir_shape(f"{MODEL}_S3_S3_hold1060_a1", MODEL) == "S3"
    assert rw.raw_dir_shape("other_S3_ramp", MODEL) is None


# ------------------------------------------------------------------------------ B5


def _fit_csv(path: Path, *, seed: int) -> Path:
    import random

    rng = random.Random(seed)
    spec = load_registry(str(REGISTRY)).model(MODEL)
    fields = r3_grid.CSV_COLUMNS
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for cell in range(16):
            load = 0.5 + 0.07 * cell
            shape = ("i2048_o96", "i256_o448")[cell % 2]
            start = 0
            for k in range(14):
                running = 20.0 * load
                gen = 30_000.0 * min(load, 1.0) * rng.uniform(0.95, 1.05)
                ratio = load * rng.uniform(0.85, 1.15)
                writer.writerow({
                    "scenario_id": f"{shape}_c{cell}", "scenario_family": shape,
                    "input_tokens": 0, "output_tokens": 0, "concurrency": cell,
                    "window_start_ms": start, "window_end_ms": start + 30_000,
                    "prompt_tokens_total": gen, "generation_tokens_total": gen,
                    "avg_waiting": 0.0, "avg_running": running, "avg_swapping": 0.0,
                    "queue_control": running, "p95_ttft": 400.0 * ratio, "p95_tpot": 30.0,
                    "p95_e2e": 1.0, "trs": "", "model_errors": 0, "proxy_transient_errors": 0,
                    "client_timeouts": 0, "slo_violated": False,
                })
                start += 5_000
    assert spec.trs.w_p > 0
    return path


def test_theta_verdict_and_holdout_run_end_to_end(tmp_path: Path) -> None:
    from scripts import theta_verdict

    fit = _fit_csv(tmp_path / "fit.csv", seed=1)
    fam = _fit_csv(tmp_path / "fam.csv", seed=2)
    val = _fit_csv(tmp_path / "val.csv", seed=3)
    out = tmp_path / "verdict.json"
    assert theta_verdict.main([
        "verdict", "--model", MODEL, "--fitting-csv", str(fit), "--family", f"prefill_heavy={fam}",
        "--w-p", "0.02", "--lambda-wait", "3", "--ttft-slo-mode", "fixed", "--ttft-p95-ms", "500", "--tpot-p95-ms", "75",
        "--n-resamples", "20", "--family-resamples", "10", "--output", str(out),
    ]) == 0
    verdict = json.loads(out.read_text())
    assert verdict["label_def"]["e2e"] == "excluded"
    assert verdict["tss"]["ema_tau_ms"] == 20000.0
    assert verdict["merged"]["delta_crit"]["method"] == "ba_grid"
    assert verdict["family_verdict"]["publish"] in ("merged", "max_family")
    assert set(verdict["stop_rule"]) >= {"satisfied", "reasons"}
    hold = tmp_path / "holdout.json"
    assert theta_verdict.main(["holdout", "--verdict", str(out), "--validation-csv", str(val),
                               "--output", str(hold)]) == 0
    report = json.loads(hold.read_text())
    assert report["windows"] > 0 and "balanced_accuracy" in report["at_published_theta"]

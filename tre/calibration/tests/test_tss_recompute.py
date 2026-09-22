"""Offline TSS recompute == the online signal (plan 2026-09-21 §6.4).

The window CSV's ``trs`` column is produced by the controller's own TRSComputer
(``r3_grid.compute_window_results``). Recomputing it from the raw columns with the same
parameters - in the dataset loader or in the refit grid - must give the same floats, and a
row a filter drops must still advance the EMA (the controller saw that window).
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

from tre_calibration.dataset import TssRecompute, load_windows_from_csv, recompute_tss_rows
from tre_calibration.signals import SignalInputs, tss_series
from tre_common.metrics_schema import ModelWindowMetrics
from tre_common.registry import TrsParams
from tre_controller.signals.trs import TRSComputer, TRSInput

FIELDS = [
    "scenario_id", "scenario_family", "window_start_ms", "window_end_ms",
    "prompt_tokens_total", "generation_tokens_total", "avg_waiting", "avg_running",
    "avg_swapping", "p95_ttft", "p95_tpot", "trs",
]
PARAMS = TrsParams(
    w_p=0.02, w_d=1.0, lambda_wait=3.0, qmin=1.0, ema_alpha=0.2485, theta_m=50.0,
    tau_crit=0.8, tau_low=1.0, tau_high=1.25, qsat=4.0, epsat=0.05, hsat=3, ema_tau_ms=20_000,
)


def _rows(seed: int = 3) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for cell in ("i256_o128_c60", "i2048_o96_c95"):
        computer = TRSComputer(ema_tau_ms=PARAMS.ema_tau_ms)
        start = 1_000_000
        for i in range(40):
            idle = rng.random() < 0.1
            wm = ModelWindowMetrics(
                model="m", window_start_ms=start, window_end_ms=start + 30_000,
                prompt_tokens=float(rng.randint(0, 60_000)), generation_tokens=float(rng.randint(0, 20_000)),
                avg_waiting=0.0 if idle else rng.uniform(0, 30), avg_running=0.0 if idle else rng.uniform(0.2, 80),
                avg_swapping=0.0, kv_cache_hit_rate=0.0,
                ttft_p95_ms=None if rng.random() < 0.15 else rng.uniform(50, 900),
                tpot_p95_ms=rng.uniform(10, 120), e2e_p95_ms=None, routable_pods=1, assigned_replicas=1,
                per_pod={},
            )
            result = computer.compute(TRSInput.from_metrics(wm, PARAMS), window_end_ms=wm.window_end_ms)
            rows.append({
                "scenario_id": cell, "scenario_family": cell.rsplit("_", 1)[0],
                "window_start_ms": wm.window_start_ms, "window_end_ms": wm.window_end_ms,
                "prompt_tokens_total": wm.prompt_tokens, "generation_tokens_total": wm.generation_tokens,
                "avg_waiting": wm.avg_waiting, "avg_running": wm.avg_running, "avg_swapping": 0.0,
                "p95_ttft": "" if wm.ttft_p95_ms is None else wm.ttft_p95_ms, "p95_tpot": wm.tpot_p95_ms,
                "trs": result.TRS if result.defined else "",
            })
            start += 5_000
    return rows


def _write(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_recompute_reproduces_the_online_trs_column_exactly(tmp_path) -> None:
    path = _write(tmp_path / "w.csv", _rows())
    rows = _read(path)
    values = recompute_tss_rows(rows, TssRecompute(w_p=0.02, lambda_wait=3.0, qmin=1.0, ema_tau_ms=20_000))
    for row, value in zip(rows, values):
        if row["trs"] == "":
            assert value is None
        else:
            assert value == float(row["trs"])


def test_loader_recompute_matches_the_column_after_filtering(tmp_path) -> None:
    path = _write(tmp_path / "w.csv", _rows())
    slo = {"ttft_p95": 500.0, "tpot_p95": 75.0}
    from_column = load_windows_from_csv(path, latency_slo_ms=slo, trim_ramp_windows=1)
    recomputed = load_windows_from_csv(
        path, latency_slo_ms=slo, trim_ramp_windows=1,
        tss=TssRecompute(w_p=0.02, lambda_wait=3.0, qmin=1.0, ema_tau_ms=20_000),
    )
    assert [w.signal for w in recomputed] == [w.signal for w in from_column]
    assert len(recomputed) < 80  # the missing-latency rows really were filtered


def test_refit_series_with_preceding_rows_matches_the_column(tmp_path) -> None:
    from scripts.refit_trs_params import load_windows_and_inputs

    path = _write(tmp_path / "w.csv", _rows())
    windows, inputs = load_windows_and_inputs(
        path, latency_slo_ms={"ttft_p95": 500.0, "tpot_p95": 75.0}, trim_ramp_windows=1
    )
    assert any(item.preceding for item in inputs)
    series = tss_series(inputs, w_p=0.02, lambda_wait=3.0, qmin=1.0, ema_tau_ms=20_000)
    assert series == [w.signal for w in windows]


def test_lambda_zero_recompute_differs_only_through_the_queue(tmp_path) -> None:
    path = _write(tmp_path / "w.csv", _rows())
    rows = _read(path)
    lw0 = recompute_tss_rows(rows, TssRecompute(w_p=0.02, lambda_wait=0.0, ema_tau_ms=None))
    for row, value in zip(rows, lw0):
        running, waiting = float(row["avg_running"]), float(row["avg_waiting"])
        if running + waiting == 0.0:
            assert value is None
            continue
        window_s = (float(row["window_end_ms"]) - float(row["window_start_ms"])) / 1000.0
        rate = (0.02 * float(row["prompt_tokens_total"]) + float(row["generation_tokens_total"]) * 1.0)
        assert abs(value - (0.02 * float(row["prompt_tokens_total"]) / window_s
                            + float(row["generation_tokens_total"]) / window_s) / max(running, 1.0)) < 1e-9
        assert rate >= 0.0


def test_signal_inputs_need_a_window() -> None:
    import pytest

    with pytest.raises(ValueError):
        tss_series([SignalInputs(1.0, 1.0, 0.0, 1.0, 0.0)], w_p=0.02, lambda_wait=3.0, qmin=1.0)

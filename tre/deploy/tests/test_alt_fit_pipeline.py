"""The alternative signals go through the TSS fit pipeline, not a parallel one (plan 6.9/6.10).

* theta_verdict is parameterised by the signal: an alt verdict has the same blocks as the
  TSS one (bootstrap, delta_crit AND delta_high by BA grid, stop rule, family rule) plus
  the ranking (AUROC, inert flag) and the opposite-direction fit;
* the family rule picks the conservative side per orientation (min theta for a
  lower_is_healthier signal);
* queue_len searches distinct values; delta margins are fitted on the controller's z;
* fit_alt_thresholds writes registry-shaped entries from verdict_report and verdict
  JSONs that theta_verdict holdout scores.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
import yaml

from scripts import calibration_campaign as campaign
from scripts import r3_grid, theta_verdict
from tre_calibration.dataset import CalibrationWindow
from tre_calibration.fit import (
    ThetaFitConfig,
    fit_delta_margins,
    fit_theta_by_balanced_accuracy,
    signal_z,
)

MODEL = "dsqwen-7b"
TRE_ROOT = Path(__file__).resolve().parents[2]


def _csv(path: Path, seed: int) -> Path:
    """Load scan where the queue grows with load and latency violates above load ~1."""
    rng = random.Random(seed)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=r3_grid.CSV_COLUMNS)
        writer.writeheader()
        for cell in range(16):
            load = 0.5 + 0.07 * cell
            shape = ("i2048_o96", "i256_o448")[cell % 2]
            start = 0
            for _k in range(14):
                running = round(20.0 * load * rng.uniform(0.9, 1.1))
                waiting = max(0.0, round(30.0 * (load - 0.95))) if load > 0.95 else 0.0
                gen = 30_000.0 * min(load, 1.0) * rng.uniform(0.95, 1.05)
                ratio = load * rng.uniform(0.85, 1.15)
                writer.writerow({
                    "scenario_id": f"{shape}_c{cell}", "scenario_family": shape,
                    "input_tokens": 0, "output_tokens": 0, "concurrency": cell,
                    "window_start_ms": start, "window_end_ms": start + 30_000,
                    "prompt_tokens_total": gen, "generation_tokens_total": gen,
                    "avg_waiting": waiting, "avg_running": running, "avg_swapping": 0.0,
                    "queue_control": running, "p95_ttft": 400.0 * ratio, "p95_tpot": 30.0,
                    "p95_e2e": 1.0, "trs": "", "model_errors": 0, "proxy_transient_errors": 0,
                    "client_timeouts": 0, "slo_violated": False,
                })
                start += 5_000
    return path


def _verdict(tmp_path: Path, signal: str, *extra: str) -> dict:
    fit = _csv(tmp_path / "fit.csv", 1)
    fam = _csv(tmp_path / "fam.csv", 2)
    out = tmp_path / f"verdict_{signal}.json"
    assert theta_verdict.main([
        "verdict", "--model", MODEL, "--fitting-csv", str(fit), "--family", f"prefill_heavy={fam}",
        "--signal", signal, "--ttft-p95-ms", "500", "--tpot-p95-ms", "75",
        "--n-resamples", "20", "--family-resamples", "10", "--output", str(out), *extra,
    ]) == 0
    return json.loads(out.read_text())


def test_an_alt_signal_verdict_has_every_block_the_tss_verdict_has(tmp_path: Path) -> None:
    tss = _verdict(tmp_path, "tss", "--w-p", "0.02", "--lambda-wait", "3")
    alt = _verdict(tmp_path, "queue_len", "--label-lambda-wait", "3")
    for key in ("merged", "families", "family_verdict", "stop_rule", "published", "label_def"):
        assert key in alt and key in tss
    for key in ("bootstrap", "delta_crit", "delta_high", "ranking", "opposite_direction", "near_theta"):
        assert key in alt["merged"] and key in tss["merged"]
    assert alt["direction"] == "lower_is_healthier" and tss["direction"] == "higher_is_healthier"
    assert alt["fit_config"]["candidate_grid"] == "unique"
    assert alt["merged"]["delta_high"]["method"] == tss["merged"]["delta_high"]["method"] == "ba_grid"
    assert alt["merged"]["delta_crit"]["bootstrap"]["high"]["n_fitted"] >= 0
    assert alt["signal_spec"]["label_lambda_wait"] == tss["signal_spec"]["label_lambda_wait"] == 3.0
    assert alt["merged"]["ranking"]["auroc"] > 0.6 and alt["merged"]["ranking"]["inert"] is False
    assert alt["published"]["tau_crit"] == pytest.approx(1 - alt["published"]["delta_crit"])
    # hold-out scores the alt verdict on its own signal definition
    val = _csv(tmp_path / "val.csv", 3)
    hold = tmp_path / "hold.json"
    assert theta_verdict.main(["holdout", "--verdict", str(tmp_path / "verdict_queue_len.json"),
                               "--validation-csv", str(val), "--output", str(hold)]) == 0
    report = json.loads(hold.read_text())
    assert report["signal"] == "queue_len" and report["windows"] > 0
    assert "opposite_direction" in report and report["ranking"]["direction"] == "lower_is_healthier"


def test_an_alt_verdict_needs_the_label_queue_weight(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="label-lambda-wait"):
        _verdict(tmp_path, "decode_tps")


def test_the_family_rule_is_conservative_in_both_orientations() -> None:
    fams = {"prefill_heavy": 1300.0, "decode_heavy": 700.0}
    higher = campaign.family_theta_verdict(1000.0, 20.0, fams)
    lower = campaign.family_theta_verdict(1000.0, 20.0, fams, direction="lower_is_healthier")
    assert (higher["publish"], higher["theta"]) == ("max_family", 1300.0)
    assert (lower["publish"], lower["theta"]) == ("min_family", 700.0)
    # lower_is_healthier: z = theta / value, so the smaller theta calls more windows critical
    value = 800.0
    assert signal_z(value, 700.0, "lower_is_healthier") < signal_z(value, 1300.0, "lower_is_healthier")


def test_the_unique_grid_searches_distinct_values_and_skips_a_zero_threshold() -> None:
    windows = []
    for cell in range(10):
        fam = "a" if cell % 2 else "b"
        windows += [CalibrationWindow(f"c{cell}", fam, 0.0, True)] * 6  # mass at zero
        windows += [CalibrationWindow(f"c{cell}", fam, 3.0, True), CalibrationWindow(f"c{cell}", fam, 5.0, True)]
        windows += [CalibrationWindow(f"c{cell}", fam, 9.0, False), CalibrationWindow(f"c{cell}", fam, 12.0, False)]
    quantile = fit_theta_by_balanced_accuracy(windows, direction="lower_is_healthier")
    unique = fit_theta_by_balanced_accuracy(windows, direction="lower_is_healthier", candidate_grid="unique")
    assert unique.publish and unique.theta == 5.0 and unique.balanced_accuracy == 1.0
    # the fixed quantile grid lands on the zero mass for most quantiles
    assert quantile.candidate_count == 14
    assert unique.candidate_count == 2  # 3.0 and 5.0; the zero is not a threshold
    cfg = ThetaFitConfig(direction="lower_is_healthier", candidate_grid="unique")
    assert cfg.as_dict()["candidate_grid"] == "unique"
    assert "candidate_grid" not in ThetaFitConfig().as_dict()


def test_delta_margins_use_the_controller_z_of_the_signal() -> None:
    # A lower_is_healthier signal: critical windows have LARGE values, i.e. small z = theta/v.
    windows = []
    for cell in range(8):
        for v, ok, ratio in ((50.0, True, 0.3), (80.0, True, 0.6), (98.0, True, 0.9),
                             (110.0, False, 1.2), (140.0, False, 1.8), (200.0, False, 2.5)):
            windows.append(CalibrationWindow(f"c{cell}", "f", v, ok, latency_ratio_p95=ratio, queue_raw=v / 10))
    fit = fit_delta_margins(windows, theta=100.0, direction="lower_is_healthier")
    assert not fit.crit.used_fallback and fit.crit.method == "ba_grid"
    # the most severe violations (140, 200) have z 0.71 / 0.5; 110 has z 0.91
    assert 1.0 - fit.crit.delta > 100.0 / 140.0
    assert fit.high.method == "ba_grid" and fit.high.tau > 1.0


def test_fit_alt_thresholds_is_a_driver_over_the_verdict(tmp_path: Path) -> None:
    path = TRE_ROOT / "calibration" / "scripts" / "fit_alt_thresholds.py"
    spec = importlib.util.spec_from_file_location("tre_fit_alt_thresholds_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fit = _csv(tmp_path / "fit.csv", 1)
    fam = _csv(tmp_path / "fam.csv", 2)
    out = tmp_path / "alt.yaml"
    assert module.main([
        "--model-input", f"{MODEL}={fit}", "--family", f"{MODEL}:prefill_heavy={fam}",
        "--ttft-p95-ms", "500", "--tpot-p95-ms", "75", "--signal", "decode_tps",
        "--n-resamples", "10", "--family-resamples", "5", "--verdict-dir", str(tmp_path),
        "--output", str(out),
    ]) == 0
    report = yaml.safe_load(out.read_text())
    entry = report["models"][MODEL]["alt_thresholds"]["decode_tps"]
    assert set(entry) == {"theta", "direction", "delta_crit", "delta_high"}
    assert report["fit_config"]["pipeline"] == "scripts.theta_verdict.verdict_report"
    verdict = json.loads((tmp_path / f"{MODEL}_verdict_decode_tps.json").read_text())
    assert verdict["published"]["theta_m"] == entry["theta"]
    assert "prefill_heavy" in verdict["families"]

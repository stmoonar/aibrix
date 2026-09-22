from __future__ import annotations

import csv
import json

from tre_calibration import cli as calibration_cli
from tre_calibration.dataset import CalibrationWindow
from tre_calibration.fit import (
    DEFAULT_SIGNAL_DIRECTION,
    DEFAULT_THETA_CRITERION,
    THETA_METHOD_BALANCED_ACCURACY,
    THETA_METHOD_RELIABILITY,
    ThetaFitConfig,
)

from scripts import eval_ranking_separation


def _separable_windows() -> list[CalibrationWindow]:
    """12 scenarios, 2 families; violations below TRS 100, healthy above."""
    rows: list[CalibrationWindow] = []
    for idx in range(3):
        rows.append(CalibrationWindow(f"steady-bad-{idx}", "steady", 60.0 + idx, False, 0.2))
        rows.append(CalibrationWindow(f"steady-bad-{idx}", "steady", 62.0 + idx, False, 0.25))
    for idx in range(3):
        rows.append(CalibrationWindow(f"burst-bad-{idx}", "burst", 70.0 + idx, False, 0.3))
        rows.append(CalibrationWindow(f"burst-bad-{idx}", "burst", 72.0 + idx, False, 0.35))
    for idx in range(3):
        rows.append(CalibrationWindow(f"steady-good-{idx}", "steady", 130.0 + idx, True, 0.8))
        rows.append(CalibrationWindow(f"steady-good-{idx}", "steady", 132.0 + idx, True, 0.85))
    for idx in range(3):
        rows.append(CalibrationWindow(f"burst-good-{idx}", "burst", 140.0 + idx, True, 0.9))
        rows.append(CalibrationWindow(f"burst-good-{idx}", "burst", 142.0 + idx, True, 0.95))
    return rows


def test_run_ranking_separation_fits_on_train_and_reports_test() -> None:
    windows = _separable_windows()

    report = eval_ranking_separation.run_ranking_separation(
        windows,
        model_name="dsqwen-7b",
        test_fraction=0.25,
        split_seed="tre-v2-ranking",
        generated_at="2026-07-08T00:00:00+00:00",
    )

    split = report["split"]
    # No scenario appears in both sets, and the union is every scenario.
    assert not (set(split["train_scenarios"]) & set(split["test_scenarios"]))
    assert set(split["train_scenarios"]) | set(split["test_scenarios"]) == {
        window.scenario_id for window in windows
    }
    assert split["leakage_free"] is True
    assert split["total_scenarios"] == 12
    assert len(split["test_scenarios"]) == 3  # 12 * 0.25

    # Train published a theta under the balanced-accuracy criterion...
    assert report["train"]["fit"]["publish"] is True
    theta = report["train"]["fit"]["theta_m"]
    assert theta is not None
    # ...and it separates the held-out test scenarios cleanly.
    assert report["test"]["threshold"] is not None
    assert report["test"]["threshold"]["theta"] == theta
    assert report["test"]["direction"]["auroc"] == 1.0
    assert report["test"]["threshold"]["false_healthy"] == 0
    assert report["test"]["threshold"]["false_violation"] == 0
    assert report["generalizes"] is True


def test_run_ranking_separation_is_deterministic() -> None:
    windows = _separable_windows()
    kwargs = dict(model_name="dsqwen-7b", generated_at="2026-07-08T00:00:00+00:00")
    first = eval_ranking_separation.run_ranking_separation(windows, **kwargs)
    second = eval_ranking_separation.run_ranking_separation(list(reversed(windows)), **kwargs)
    assert first == second


def test_run_ranking_separation_single_scenario_has_no_test_threshold() -> None:
    windows = [
        CalibrationWindow("only", "steady", 100.0 + idx, True, 0.9) for idx in range(4)
    ]
    report = eval_ranking_separation.run_ranking_separation(windows, model_name="dsqwen-7b")
    assert report["split"]["test_scenarios"] == []
    assert report["test"]["threshold"] is None
    assert report["generalizes"] is False


def _write_windows_csv(path, windows: list[CalibrationWindow]) -> None:
    fieldnames = [
        "scenario_id",
        "scenario_family",
        "trs",
        "p95_ttft_client_ms",
        "p95_tpot_client_ms",
        "prompt_tokens_total",
        "generation_tokens_total",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for window in windows:
            # Reconstruct p95 latencies consistent with slo_met (SLO ttft=100, tpot=50).
            writer.writerow(
                {
                    "scenario_id": window.scenario_id,
                    "scenario_family": window.scenario_family,
                    "trs": window.signal,
                    "p95_ttft_client_ms": 80.0 if window.slo_met else 130.0,
                    "p95_tpot_client_ms": 40.0,
                    "prompt_tokens_total": 100.0,
                    "generation_tokens_total": 50.0,
                }
            )


def test_cli_writes_report_from_csv(tmp_path) -> None:
    src = tmp_path / "windows.csv"
    out = tmp_path / "report.json"
    _write_windows_csv(src, _separable_windows())

    rc = eval_ranking_separation.main(
        [
            "--input", str(src),
            "--output", str(out),
            "--model-name", "dsqwen-7b",
            "--ttft-p95-ms", "100",
            "--tpot-p95-ms", "50",
            "--test-fraction", "0.25",
            "--generated-at", "2026-07-08T00:00:00+00:00",
        ]
    )

    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["model_name"] == "dsqwen-7b"
    assert report["split"]["trim_ramp_windows"] == 1
    assert report["split"]["total_windows"] == 12
    assert report["split"]["train_window_count"] == 9
    assert report["split"]["test_window_count"] == 3
    assert report["split"]["leakage_free"] is True
    assert report["train"]["fit"]["publish"] is True
    assert report["test"]["threshold"] is not None
    assert report["generalizes"] is True


def _overlapping_windows() -> list[CalibrationWindow]:
    """Healthy and violating bands that overlap, so the two criteria fit different thetas."""
    rows: list[CalibrationWindow] = []
    for family in ("steady", "burst"):
        for cell in range(6):
            cid = f"{family}-{cell}"
            for signal in (40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0):
                rows.append(CalibrationWindow(cid, family, signal, False, 0.2))
            for signal in (50.0, 60.0, 70.0, 80.0, 90.0, 100.0):
                rows.append(CalibrationWindow(cid, family, signal, True, 0.9))
    return rows


def test_acceptance_fits_under_the_criterion_the_calibration_cli_publishes() -> None:
    """s1.4 acceptance must judge the threshold that actually ships.

    The train-split fit goes through the same entry point and the same default criterion
    as ``tre_calibration.cli``; a report fitted under a different criterion would accept
    or reject a threshold nobody publishes.
    """
    report = eval_ranking_separation.run_ranking_separation(
        _overlapping_windows(),
        model_name="dsqwen-7b",
        generated_at="2026-09-21T00:00:00+00:00",
    )
    assert report["method"]["theta_m_method"] == THETA_METHOD_BALANCED_ACCURACY
    assert report["fit_config"]["theta_criterion"] == DEFAULT_THETA_CRITERION
    assert report["fit_config"]["direction"] == DEFAULT_SIGNAL_DIRECTION
    # Balanced-accuracy fit fields are reported, not reliability-only ones.
    assert "balanced_accuracy" in report["train"]["fit"]
    assert report["train"]["fit"]["publish"] is True


def test_the_criterion_changes_the_theta_the_acceptance_judges() -> None:
    """The criterion is load-bearing here, not a label on an unchanged number."""
    windows = _overlapping_windows()
    kwargs = dict(model_name="dsqwen-7b", generated_at="2026-09-21T00:00:00+00:00")
    balanced = eval_ranking_separation.run_ranking_separation(
        windows, config=ThetaFitConfig(criterion="balanced_accuracy"), **kwargs
    )
    reliability = eval_ranking_separation.run_ranking_separation(
        windows, config=ThetaFitConfig(criterion="reliability"), **kwargs
    )
    assert reliability["method"]["theta_m_method"] == THETA_METHOD_RELIABILITY
    assert balanced["train"]["fit"]["theta_m"] != reliability["train"]["fit"]["theta_m"]
    assert (
        balanced["test"]["threshold"]["theta"] != reliability["test"]["threshold"]["theta"]
    )


def test_cli_defaults_match_the_calibration_cli_defaults() -> None:
    required = [
        "--input", "x.csv",
        "--output", "y.json",
        "--model-name", "m",
        "--ttft-p95-ms", "100",
        "--tpot-p95-ms", "50",
    ]
    eval_args = eval_ranking_separation._parse_args(required)
    fit_args = calibration_cli._parse_args(required)
    for knob in (
        "theta_criterion",
        "direction",
        "min_healthy_recall",
        "healthy_quantiles",
        "reliability_target",
        "min_support",
        "min_confidence",
        "min_scenario_families",
        "max_single_scenario_ratio",
    ):
        assert getattr(eval_args, knob) == getattr(fit_args, knob), knob
    assert eval_args.theta_criterion == DEFAULT_THETA_CRITERION


def test_cli_report_records_the_criterion(tmp_path) -> None:
    src = tmp_path / "windows.csv"
    out = tmp_path / "report.json"
    _write_windows_csv(src, _separable_windows())

    rc = eval_ranking_separation.main(
        [
            "--input", str(src),
            "--output", str(out),
            "--model-name", "dsqwen-7b",
            "--ttft-p95-ms", "100",
            "--tpot-p95-ms", "50",
            "--test-fraction", "0.25",
            "--generated-at", "2026-09-21T00:00:00+00:00",
        ]
    )

    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["method"]["theta_m_method"] == THETA_METHOD_BALANCED_ACCURACY
    assert report["fit_config"]["theta_criterion"] == DEFAULT_THETA_CRITERION
    assert report["fit_config"]["direction"] == DEFAULT_SIGNAL_DIRECTION

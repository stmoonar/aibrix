"""The bootstrap CI driver must bracket the theta the calibration CLI publishes.

The campaign stop rule reads ``publish_rate`` and the CI half-width off this driver's
report. Those numbers only answer "is the calibration done" if the interval was fitted
under the same criterion, orientation and gates as the theta that gets published, so the
tests below pin the defaults to the calibration CLI's and pin the report to state them.
"""
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
)

from scripts import bootstrap_theta_ci


def _overlapping_windows() -> list[CalibrationWindow]:
    """Healthy and violating bands that overlap, so the two criteria disagree."""
    rows: list[CalibrationWindow] = []
    for family in ("steady", "burst"):
        for cell in range(4):
            cid = f"{family}-{cell}"
            for signal in (40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0):
                rows.append(CalibrationWindow(cid, family, signal, False))
            for signal in (50.0, 60.0, 70.0, 80.0, 90.0, 100.0):
                rows.append(CalibrationWindow(cid, family, signal, True))
    return rows


def _write_csv(path, windows: list[CalibrationWindow]) -> None:
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


def _run(tmp_path, name: str = "run", *extra: str) -> dict:
    src = tmp_path / f"windows-{name}.csv"
    out = tmp_path / f"report-{name}.json"
    _write_csv(src, _overlapping_windows())
    rc = bootstrap_theta_ci.main(
        [
            "--input", str(src),
            "--output", str(out),
            "--model-name", "dsqwen-7b",
            "--ttft-p95-ms", "100",
            "--tpot-p95-ms", "50",
            "--trim-ramp-windows", "0",
            "--n-resamples", "120",
            "--seed", "42",
            "--generated-at", "2026-09-21T00:00:00+00:00",
            *extra,
        ]
    )
    assert rc == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_cli_defaults_match_the_calibration_cli_defaults() -> None:
    """Same defaults as the fit CLI, so an unflagged CI run describes the published theta."""
    required = [
        "--input", "x.csv",
        "--output", "y.json",
        "--model-name", "m",
        "--ttft-p95-ms", "100",
        "--tpot-p95-ms", "50",
    ]
    ci_args = bootstrap_theta_ci._parse_args(required)
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
        assert getattr(ci_args, knob) == getattr(fit_args, knob), knob
    assert ci_args.theta_criterion == DEFAULT_THETA_CRITERION
    assert ci_args.direction == DEFAULT_SIGNAL_DIRECTION


def test_report_states_the_criterion_it_bootstrapped_under(tmp_path) -> None:
    report = _run(tmp_path, "defaults")
    assert report["method"]["theta_m_method"] == THETA_METHOD_BALANCED_ACCURACY
    assert report["fit_config"]["theta_criterion"] == DEFAULT_THETA_CRITERION
    assert report["fit_config"]["direction"] == DEFAULT_SIGNAL_DIRECTION
    # The resamples' own record, taken off the result rather than off argv.
    assert report["bootstrap"]["fit_config"]["theta_criterion"] == DEFAULT_THETA_CRITERION
    # Balanced-accuracy fit fields, not reliability-only ones.
    assert "balanced_accuracy" in report["point_fit"]
    assert report["point_fit"]["publish"] is True
    assert report["bootstrap"]["n_published"] > 0


def test_point_estimate_and_interval_move_together_when_the_criterion_changes(
    tmp_path,
) -> None:
    """Interval and point estimate belong to one criterion: change it and both move."""
    balanced = _run(tmp_path, "balanced", "--theta-criterion", "balanced_accuracy")
    reliability = _run(tmp_path, "reliability", "--theta-criterion", "reliability")

    assert reliability["method"]["theta_m_method"] == THETA_METHOD_RELIABILITY
    assert balanced["point_fit"]["theta"] != reliability["point_fit"]["theta"]
    assert (
        balanced["bootstrap"]["theta_p2_5"],
        balanced["bootstrap"]["theta_p97_5"],
    ) != (
        reliability["bootstrap"]["theta_p2_5"],
        reliability["bootstrap"]["theta_p97_5"],
    )
    for report in (balanced, reliability):
        b = report["bootstrap"]
        assert b["theta_p2_5"] <= report["point_fit"]["theta"] <= b["theta_p97_5"]

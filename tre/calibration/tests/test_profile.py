from __future__ import annotations

from tre_calibration.fit import (
    THETA_METHOD_BALANCED_ACCURACY,
    THETA_METHOD_RELIABILITY,
    DELTA_METHOD,
    BalancedAccuracyThetaFit,
    ReliabilityThetaFit,
    fit_delta_margins,
)
from tre_calibration.dataset import CalibrationWindow
from tre_calibration.profile import build_profile_patch, theta_method_of
from tre_calibration.signals import ParameterCandidateScore


def _parameter_score() -> ParameterCandidateScore:
    return ParameterCandidateScore(
        w_p=3.0,
        lambda_wait=1.0,
        qmin=1.0,
        objective=1.0,
        spearman_health=1.0,
        auroc=1.0,
        scored_windows=[],
    )


def test_build_profile_patch_is_deterministic_and_publishable() -> None:
    theta_fit = ReliabilityThetaFit(
        publish=True,
        theta=105.0,
        support=3,
        attainment=1.0,
        confidence=1.0,
        coverage_pass=True,
        family_counts={"burst": 1, "steady": 2},
        reject_reason=None,
        candidate_count=5,
    )

    patch = build_profile_patch(
        "dsqwen-7b",
        theta_fit=theta_fit,
        parameter_score=_parameter_score(),
        generated_at="2026-07-04T00:00:00+00:00",
        fit_config={"theta_criterion": "reliability", "trim_ramp_windows": 0},
    )

    assert patch == {
        "generated_at": "2026-07-04T00:00:00+00:00",
        "model_name": "dsqwen-7b",
        "publish": True,
        "method": {
            "theta_m_method": THETA_METHOD_RELIABILITY,
            "delta_method": None,
        },
        "fit_config": {"theta_criterion": "reliability", "trim_ramp_windows": 0},
        "fit": {
            "attainment": 1.0,
            "candidate_count": 5,
            "confidence": 1.0,
            "coverage_pass": True,
            "direction": "higher_is_healthier",
            "family_counts": {"burst": 1, "steady": 2},
            "reject_reason": None,
            "support": 3,
        },
        "metrics": {
            "auroc": 1.0,
            "objective": 1.0,
            "spearman_health": 1.0,
        },
        "trs": {
            "lambda_wait": 1.0,
            "qmin": 1.0,
            "theta_m": 105.0,
            "w_p": 3.0,
        },
    }


def test_build_profile_patch_records_the_balanced_accuracy_method_and_its_knobs() -> None:
    theta_fit = BalancedAccuracyThetaFit(
        publish=True,
        theta=306.0,
        healthy_quantile=0.20,
        balanced_accuracy=0.90,
        recall_good=0.80,
        specificity_bad=1.0,
        precision_good=1.0,
        healthy_window_count=100,
        violating_window_count=30,
        candidate_count=10,
        min_healthy_recall=0.0,
        family_counts={"burst": 40, "steady": 40},
        coverage_pass=True,
        reject_reason=None,
    )

    patch = build_profile_patch(
        "dsqwen-7b",
        theta_fit=theta_fit,
        parameter_score=_parameter_score(),
        generated_at="2026-09-20T00:00:00+00:00",
        fit_config={"min_healthy_recall": 0.0, "theta_criterion": "balanced_accuracy", "trim_ramp_windows": 0},
        inputs={"csv_path": "/tmp/scan.csv", "csv_sha256": "deadbeef"},
    )

    assert theta_method_of(theta_fit) == THETA_METHOD_BALANCED_ACCURACY
    assert patch["method"] == {
        "theta_m_method": THETA_METHOD_BALANCED_ACCURACY,
        "delta_method": None,
    }
    assert patch["fit_config"]["min_healthy_recall"] == 0.0
    assert patch["fit_config"]["trim_ramp_windows"] == 0
    assert patch["fit"]["healthy_quantile"] == 0.20
    assert patch["fit"]["recall_good"] == 0.80
    # The orientation the threshold was fitted under is part of the rule, not a detail:
    # the same theta means the opposite thing on a lower-is-healthier signal.
    assert patch["fit"]["direction"] == "higher_is_healthier"
    assert patch["trs"]["theta_m"] == 306.0
    assert patch["inputs"] == {"csv_path": "/tmp/scan.csv", "csv_sha256": "deadbeef"}


def test_build_profile_patch_carries_fitted_delta_margins_into_trs() -> None:
    theta_fit = BalancedAccuracyThetaFit(
        publish=True,
        theta=100.0,
        healthy_quantile=0.20,
        balanced_accuracy=0.9,
        recall_good=0.8,
        specificity_bad=1.0,
        precision_good=1.0,
        healthy_window_count=40,
        violating_window_count=30,
        candidate_count=10,
        min_healthy_recall=0.0,
        family_counts={"burst": 20, "steady": 20},
        coverage_pass=True,
        reject_reason=None,
    )
    delta_fit = fit_delta_margins(_delta_windows(), theta=100.0)

    patch = build_profile_patch(
        "dsqwen-7b",
        theta_fit=theta_fit,
        parameter_score=_parameter_score(),
        generated_at="2026-09-20T00:00:00+00:00",
        delta_fit=delta_fit,
    )

    assert patch["method"]["delta_method"] == DELTA_METHOD
    assert patch["trs"]["tau_crit"] == delta_fit.crit.tau
    assert patch["trs"]["tau_high"] == delta_fit.high.tau
    assert patch["trs"]["delta_crit"] == delta_fit.crit.delta
    assert patch["trs"]["delta_high"] == delta_fit.high.delta
    assert patch["delta_fit"]["labels"]["queue_available"] is True


def _delta_windows() -> list[CalibrationWindow]:
    rows: list[CalibrationWindow] = []
    for i in range(20):
        rows.append(
            CalibrationWindow(
                scenario_id=f"bad-{i}",
                scenario_family="burst",
                signal=40.0 + i,
                slo_met=False,
                latency_ratio_p95=2.0 + 0.1 * i,
                queue_raw=40.0 + i,
            )
        )
    for i in range(20):
        rows.append(
            CalibrationWindow(
                scenario_id=f"good-{i}",
                scenario_family="steady",
                signal=180.0 + 2.0 * i,
                slo_met=True,
                latency_ratio_p95=0.2 + 0.005 * i,
                queue_raw=1.0 + 0.05 * i,
            )
        )
    return rows

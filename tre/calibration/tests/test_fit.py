from __future__ import annotations

from tre_calibration.dataset import CalibrationWindow, split_by_scenario
from tre_calibration.evaluate import evaluate_threshold
from tre_calibration.fit import fit_theta_from_health


def _synthetic_windows() -> list[CalibrationWindow]:
    return [
        CalibrationWindow(scenario_id="steady-a", scenario_family="steady", signal=60.0, slo_met=False, health_score=0.2),
        CalibrationWindow(scenario_id="steady-a", scenario_family="steady", signal=80.0, slo_met=False, health_score=0.3),
        CalibrationWindow(scenario_id="burst-b", scenario_family="burst", signal=90.0, slo_met=False, health_score=0.4),
        CalibrationWindow(scenario_id="steady-c", scenario_family="steady", signal=110.0, slo_met=True, health_score=0.7),
        CalibrationWindow(scenario_id="burst-d", scenario_family="burst", signal=120.0, slo_met=True, health_score=0.8),
        CalibrationWindow(scenario_id="burst-d", scenario_family="burst", signal=140.0, slo_met=True, health_score=0.9),
    ]


def test_fit_theta_recovers_known_synthetic_boundary() -> None:
    fit = fit_theta_from_health(_synthetic_windows(), signal_name="trs")

    assert fit.signal_name == "trs"
    assert fit.theta == 100.0
    assert fit.violation_max == 90.0
    assert fit.healthy_min == 110.0
    assert fit.sample_count == 6


def test_split_by_scenario_keeps_whole_scenarios_out_of_train() -> None:
    rows = _synthetic_windows()

    train, test = split_by_scenario(rows, test_scenarios={"burst-b", "burst-d"})

    assert {row.scenario_id for row in train} == {"steady-a", "steady-c"}
    assert {row.scenario_id for row in test} == {"burst-b", "burst-d"}
    assert not {row.scenario_id for row in train} & {row.scenario_id for row in test}


def test_evaluate_threshold_reports_correct_direction_on_synthetic_data() -> None:
    metrics = evaluate_threshold(_synthetic_windows(), theta=100.0)

    assert metrics.auroc == 1.0
    assert metrics.spearman_health == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.false_healthy == 0
    assert metrics.false_violation == 0

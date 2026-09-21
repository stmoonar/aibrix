from __future__ import annotations

from tre_calibration.dataset import CalibrationWindow
from tre_calibration.evaluate import evaluate_threshold


def _synthetic_windows() -> list[CalibrationWindow]:
    return [
        CalibrationWindow(scenario_id="steady-a", scenario_family="steady", signal=60.0, slo_met=False, health_score=0.2),
        CalibrationWindow(scenario_id="steady-a", scenario_family="steady", signal=80.0, slo_met=False, health_score=0.3),
        CalibrationWindow(scenario_id="burst-b", scenario_family="burst", signal=90.0, slo_met=False, health_score=0.4),
        CalibrationWindow(scenario_id="steady-c", scenario_family="steady", signal=110.0, slo_met=True, health_score=0.7),
        CalibrationWindow(scenario_id="burst-d", scenario_family="burst", signal=120.0, slo_met=True, health_score=0.8),
        CalibrationWindow(scenario_id="burst-d", scenario_family="burst", signal=140.0, slo_met=True, health_score=0.9),
    ]


def test_evaluate_threshold_reports_correct_direction_on_synthetic_data() -> None:
    metrics = evaluate_threshold(_synthetic_windows(), theta=100.0)

    assert metrics.auroc == 1.0
    assert metrics.spearman_health == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.false_healthy == 0
    assert metrics.false_violation == 0

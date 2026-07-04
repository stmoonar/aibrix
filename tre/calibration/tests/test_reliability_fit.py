from __future__ import annotations

from tre_calibration.dataset import CalibrationWindow
from tre_calibration.fit import fit_theta_by_reliability


def test_fit_theta_by_reliability_publishes_first_covered_threshold() -> None:
    rows = [
        CalibrationWindow("steady-low", "steady", 60.0, False),
        CalibrationWindow("burst-low", "burst", 80.0, False),
        CalibrationWindow("steady-good", "steady", 105.0, True),
        CalibrationWindow("burst-good", "burst", 120.0, True),
        CalibrationWindow("steady-high", "steady", 140.0, True),
    ]

    fit = fit_theta_by_reliability(
        rows,
        reliability_target=0.9,
        min_support=3,
        min_confidence=0.9,
        min_scenario_families=2,
        max_single_scenario_ratio=0.7,
    )

    assert fit.publish is True
    assert fit.theta == 105.0
    assert fit.support == 3
    assert fit.attainment == 1.0
    assert fit.confidence == 1.0
    assert fit.coverage_pass is True
    assert fit.family_counts == {"burst": 1, "steady": 2}
    assert fit.reject_reason is None


def test_fit_theta_by_reliability_rejects_insufficient_family_coverage() -> None:
    rows = [
        CalibrationWindow("steady-low", "steady", 60.0, False),
        CalibrationWindow("steady-good-a", "steady", 105.0, True),
        CalibrationWindow("steady-good-b", "steady", 120.0, True),
    ]

    fit = fit_theta_by_reliability(
        rows,
        reliability_target=0.9,
        min_support=2,
        min_confidence=0.9,
        min_scenario_families=2,
        max_single_scenario_ratio=0.7,
    )

    assert fit.publish is False
    assert fit.theta == 105.0
    assert fit.coverage_pass is False
    assert fit.reject_reason == "insufficient_coverage"

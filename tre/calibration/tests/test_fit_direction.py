from __future__ import annotations

import pytest

from tre_calibration.dataset import CalibrationWindow
from tre_calibration.fit import fit_theta_by_reliability


_CONFIG = {
    "reliability_target": 0.9,
    "min_support": 3,
    "min_confidence": 0.9,
    "min_scenario_families": 2,
    "max_single_scenario_ratio": 0.7,
}


def test_fit_lower_is_healthier_recovers_queue_boundary() -> None:
    windows = [
        CalibrationWindow("steady-1", "steady", 1.0, True),
        CalibrationWindow("burst-3", "burst", 3.0, True),
        CalibrationWindow("steady-5", "steady", 5.0, True),
        CalibrationWindow("burst-6", "burst", 6.0, False),
        CalibrationWindow("steady-8", "steady", 8.0, False),
    ]

    fit = fit_theta_by_reliability(
        windows,
        direction="lower_is_healthier",
        **_CONFIG,
    )

    assert fit.publish is True
    assert fit.theta == 5.0
    assert fit.support == 3
    assert fit.attainment == 1.0
    assert fit.family_counts == {"burst": 1, "steady": 2}


def test_fit_rejects_unknown_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        fit_theta_by_reliability([], direction="sideways", **_CONFIG)

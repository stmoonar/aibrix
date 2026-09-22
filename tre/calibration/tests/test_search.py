from __future__ import annotations

from tre_calibration.dataset import CalibrationWindow
from tre_calibration.signals import SignalInputs, grid_search_parameters


def test_grid_search_parameters_selects_best_direction_candidate() -> None:
    windows = [
        CalibrationWindow("low-a", "steady", 0.0, False, health_score=0.2),
        CalibrationWindow("low-b", "burst", 0.0, False, health_score=0.3),
        CalibrationWindow("high-a", "steady", 0.0, True, health_score=0.8),
        CalibrationWindow("high-b", "burst", 0.0, True, health_score=0.9),
    ]
    inputs = [
        SignalInputs(10.0, 120.0, 0.0, 1.0, 0.0),
        SignalInputs(20.0, 100.0, 0.0, 1.0, 0.0),
        SignalInputs(100.0, 20.0, 0.0, 1.0, 0.0),
        SignalInputs(120.0, 10.0, 0.0, 1.0, 0.0),
    ]

    result = grid_search_parameters(
        windows,
        inputs,
        w_p_candidates=[0.0, 3.0],
        lambda_wait_candidates=[1.0],
        qmin_candidates=[1.0],
    )

    assert len(result.candidates) == 2
    assert result.best.w_p == 3.0
    assert result.best.auroc == 1.0
    assert result.best.spearman_health == 1.0
    assert result.best.objective == 1.0
    assert [window.signal for window in result.best.scored_windows] == [150.0, 160.0, 320.0, 370.0]

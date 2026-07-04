from __future__ import annotations

from tre_calibration.dataset import CalibrationWindow
from tre_calibration.signals import SignalInputs, compute_trs, score_parameter_candidate


def test_compute_trs_matches_archived_formula() -> None:
    breakdown = compute_trs(
        SignalInputs(
            prompt_tokens_total=100.0,
            generation_tokens_total=50.0,
            avg_waiting=2.0,
            avg_running=8.0,
            avg_swapping=0.0,
            assigned_replicas=2.0,
            routable_pods=1.0,
            kv_cache_hit_rate=0.25,
        ),
        w_p=2.0,
        lambda_wait=0.5,
        qmin=10.0,
    )

    assert breakdown.total_tokens == 200.0
    assert breakdown.queue_raw == 9.0
    assert breakdown.queue_floor == 10.0
    assert breakdown.trs_floor == 40.0
    assert round(breakdown.trs_no_floor, 6) == round((200.0 / 9.0) * 2.0, 6)


def test_score_parameter_candidate_reports_direction_metrics() -> None:
    windows = [
        CalibrationWindow("low-a", "steady", 0.0, False, health_score=0.2),
        CalibrationWindow("low-b", "burst", 0.0, False, health_score=0.3),
        CalibrationWindow("high-a", "steady", 0.0, True, health_score=0.8),
        CalibrationWindow("high-b", "burst", 0.0, True, health_score=0.9),
    ]
    inputs = [
        SignalInputs(0.0, 50.0, 0.0, 1.0, 0.0),
        SignalInputs(0.0, 60.0, 0.0, 1.0, 0.0),
        SignalInputs(0.0, 100.0, 0.0, 1.0, 0.0),
        SignalInputs(0.0, 120.0, 0.0, 1.0, 0.0),
    ]

    score = score_parameter_candidate(windows, inputs, w_p=1.0, lambda_wait=1.0, qmin=1.0)

    assert score.w_p == 1.0
    assert score.lambda_wait == 1.0
    assert score.qmin == 1.0
    assert score.auroc == 1.0
    assert score.spearman_health == 1.0
    assert score.objective == 1.0
    assert [window.signal for window in score.scored_windows] == [50.0, 60.0, 100.0, 120.0]

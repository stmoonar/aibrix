"""Acceptance item B uses the controller's dwell (plan 2026-09-21 6.9f B, 6.11 task 4)."""
from __future__ import annotations

from scripts.theta_verdict import critical_dwell_flags, dwell_acceptance
from tre_calibration.dataset import CalibrationWindow


def _w(cell: str, t: int, signal: float, cls: str | None) -> CalibrationWindow:
    return CalibrationWindow(cell, "f", signal, cls is None, window_start_ms=t * 10_000, violation_class=cls)


def test_single_critical_window_is_not_confirmed_two_in_a_row_are() -> None:
    # theta 100, tau_crit 0.5 -> CRITICAL iff signal < 50
    ws = [_w("a", 0, 200, None), _w("a", 1, 10, "both"), _w("a", 2, 200, None),
          _w("a", 3, 10, "both"), _w("a", 4, 10, "tpot_only"), _w("a", 5, 10, "ttft_only")]
    assert critical_dwell_flags(ws, theta=100.0, tau_crit=0.5, direction="higher_is_healthier") == [
        False, False, False, False, True, True]
    assert critical_dwell_flags(ws, theta=100.0, tau_crit=0.5, direction="higher_is_healthier",
                                dwell_windows=1) == [False, True, False, True, True, True]


def test_dwell_runs_per_cell_in_window_order() -> None:
    # rows out of order and interleaved across cells: a run never spans two cells
    ws = [_w("b", 1, 10, "both"), _w("a", 0, 10, "both"), _w("b", 0, 10, "both"), _w("a", 1, 200, None)]
    assert critical_dwell_flags(ws, theta=100.0, tau_crit=0.5, direction="higher_is_healthier") == [
        True, False, False, False]


def test_dwell_acceptance_reports_criterion_b_classes() -> None:
    ws = [_w("a", 0, 10, "both"), _w("a", 1, 10, "tpot_only"), _w("a", 2, 10, None),
          _w("a", 3, 200, "ttft_only"), _w("a", 4, 200, None)]
    out = dwell_acceptance(ws, theta=100.0, tau_crit=0.5, direction="higher_is_healthier")
    assert out["dwell_windows"] == 2
    assert out["critical_recall_both_tpot"] == 0.5 and out["both_tpot_windows"] == 2
    assert out["critical_false_alarm_on_healthy"] == 0.5
    assert out["violation_classes"]["ttft_only"] == {"windows": 1, "critical_recall": 0.0}

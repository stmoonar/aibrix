"""Regression guard for the theta_m criterion.

The fixture below has a clean healthy/violating boundary at signal 300: every violating
window sits at or below 300 and 80 of the 100 healthy windows sit above it. Twenty
healthy windows form a low tail that overlaps the violating range, which is what a real
load scan looks like and what makes the choice of criterion matter.

Three criteria are exercised on the *same* windows:

* the balanced-accuracy fit with the recall floor off -- lands on the boundary;
* the same fit with ``min_healthy_recall=0.90`` -- the floor vetoes the optimum and
  drags theta down into the overlap region;
* the cumulative-attainment fit -- also lands in the overlap region.

If the default criterion or the default recall floor ever regresses, the first test
fails, because it asserts the boundary theta *and* that the boundary beats both biased
answers on balanced accuracy.
"""
from __future__ import annotations

from tre_calibration.dataset import CalibrationWindow
from tre_calibration.fit import (
    DEFAULT_MIN_HEALTHY_RECALL,
    DEFAULT_THETA_CRITERION,
    _balanced_accuracy_at,
    fit_theta_by_balanced_accuracy,
    fit_theta_by_reliability,
)

BOUNDARY = 300.0


def _windows() -> list[CalibrationWindow]:
    rows: list[CalibrationWindow] = []
    # 30 violating windows, all at or below the boundary.
    for i in range(30):
        signal = 10.0 * (i + 1)
        rows.append(_window(f"bad-{i}", signal, slo_met=False))
    # 20 healthy windows in the overlap region, 80 above the boundary.
    for i in range(20):
        rows.append(_window(f"good-low-{i}", 100.0 + 10.0 * i, slo_met=True))
    for i in range(80):
        rows.append(_window(f"good-high-{i}", 310.0 + 10.0 * i, slo_met=True))
    return rows


def _window(scenario_id: str, signal: float, *, slo_met: bool) -> CalibrationWindow:
    return CalibrationWindow(
        scenario_id=scenario_id,
        scenario_family="steady",
        signal=signal,
        slo_met=slo_met,
    )


def _families_alternating(rows: list[CalibrationWindow]) -> list[CalibrationWindow]:
    """Spread the rows over two scenario families so the coverage gate passes."""
    return [
        CalibrationWindow(
            scenario_id=row.scenario_id,
            scenario_family="steady" if index % 2 == 0 else "burst",
            signal=row.signal,
            slo_met=row.slo_met,
        )
        for index, row in enumerate(rows)
    ]


def test_defaults_are_the_balanced_accuracy_criterion_without_a_recall_floor() -> None:
    assert DEFAULT_THETA_CRITERION == "balanced_accuracy"
    assert DEFAULT_MIN_HEALTHY_RECALL == 0.0


def test_balanced_accuracy_fit_lands_on_the_boundary_the_data_supports() -> None:
    rows = _families_alternating(_windows())

    fit = fit_theta_by_balanced_accuracy(rows)

    assert fit.publish is True
    # The healthy 20th percentile straddles the boundary (290 -> 310).
    assert fit.healthy_quantile == 0.20
    assert BOUNDARY <= fit.theta <= 310.0
    # Every violating window is below theta, 80 of 100 healthy windows above it.
    assert fit.specificity_bad == 1.0
    assert fit.recall_good == 0.80
    assert fit.balanced_accuracy == 0.90


def test_recall_floor_vetoes_the_optimum_and_reintroduces_the_low_theta_bias() -> None:
    rows = _families_alternating(_windows())
    scores = [row.signal for row in rows]
    labels = [1 if row.slo_met else 0 for row in rows]

    unfloored = fit_theta_by_balanced_accuracy(rows, min_healthy_recall=0.0)
    floored = fit_theta_by_balanced_accuracy(rows, min_healthy_recall=0.90)

    assert floored.theta < BOUNDARY
    assert floored.recall_good >= 0.90
    assert floored.balanced_accuracy < unfloored.balanced_accuracy
    # ... and it is genuinely worse on the data, not merely a different tie-break.
    assert (
        _balanced_accuracy_at(scores, labels, floored.theta)["balanced_accuracy"]
        < _balanced_accuracy_at(scores, labels, unfloored.theta)["balanced_accuracy"]
    )


def test_cumulative_attainment_criterion_also_sits_below_the_boundary() -> None:
    rows = _families_alternating(_windows())
    scores = [row.signal for row in rows]
    labels = [1 if row.slo_met else 0 for row in rows]

    reliability = fit_theta_by_reliability(
        rows,
        reliability_target=0.9,
        min_support=3,
        min_confidence=0.9,
        min_scenario_families=2,
        max_single_scenario_ratio=0.7,
    )
    balanced = fit_theta_by_balanced_accuracy(rows)

    assert reliability.theta < BOUNDARY
    # Local attainment just above the containment threshold is poor: plenty of
    # violating windows are predicted healthy there.
    assert (
        _balanced_accuracy_at(scores, labels, reliability.theta)["specificity_bad"] < 1.0
    )
    assert (
        _balanced_accuracy_at(scores, labels, reliability.theta)["balanced_accuracy"]
        < _balanced_accuracy_at(scores, labels, balanced.theta)["balanced_accuracy"]
    )


def test_balanced_accuracy_fit_rejects_when_there_are_no_violating_windows() -> None:
    rows = _families_alternating(
        [_window(f"good-{i}", 100.0 + i, slo_met=True) for i in range(10)]
    )

    fit = fit_theta_by_balanced_accuracy(rows)

    assert fit.publish is False
    assert fit.reject_reason == "no_violating_windows"

from __future__ import annotations

import pytest

from tre_calibration.dataset import CalibrationWindow
from tre_calibration.fit import (
    FALLBACK_DELTA_CRIT,
    FALLBACK_DELTA_HIGH,
    fit_delta_margins,
)

THETA = 100.0


def _window(
    index: int,
    *,
    signal: float,
    slo_met: bool,
    latency_ratio_p95: float,
    queue_raw: float | None,
) -> CalibrationWindow:
    return CalibrationWindow(
        scenario_id=f"cell-{index}",
        scenario_family="steady" if index % 2 == 0 else "burst",
        signal=signal,
        slo_met=slo_met,
        health_score=1.0 / (1.0 + latency_ratio_p95),
        latency_ratio_p95=latency_ratio_p95,
        queue_raw=queue_raw,
    )


def _windows(*, queue_raw: bool = True) -> list[CalibrationWindow]:
    rows: list[CalibrationWindow] = []
    index = 0
    # Severe violations sit well below z = 1 and carry deep queues.
    for i in range(20):
        rows.append(
            _window(
                index,
                signal=40.0 + i,
                slo_met=False,
                latency_ratio_p95=2.0 + 0.1 * i,
                queue_raw=40.0 + i if queue_raw else None,
            )
        )
        index += 1
    # Mild violations just under z = 1.
    for i in range(10):
        rows.append(
            _window(
                index,
                signal=90.0 + i,
                slo_met=False,
                latency_ratio_p95=1.05 + 0.01 * i,
                queue_raw=20.0 + i if queue_raw else None,
            )
        )
        index += 1
    # Healthy but busy: above z = 1, still loaded.
    for i in range(20):
        rows.append(
            _window(
                index,
                signal=110.0 + i,
                slo_met=True,
                latency_ratio_p95=0.85 - 0.005 * i,
                queue_raw=15.0 - 0.2 * i if queue_raw else None,
            )
        )
        index += 1
    # Comfortable surplus: high z, shallow queue, plenty of latency headroom.
    for i in range(20):
        rows.append(
            _window(
                index,
                signal=180.0 + 2.0 * i,
                slo_met=True,
                latency_ratio_p95=0.3 - 0.005 * i,
                queue_raw=2.0 - 0.05 * i if queue_raw else None,
            )
        )
        index += 1
    return rows


def test_delta_margins_bracket_tau_low_and_separate_the_labels() -> None:
    fit = fit_delta_margins(_windows(), theta=THETA)

    assert fit.labels.queue_available is True
    assert fit.labels.critical_positive_count > 0
    assert fit.labels.surplus_positive_count > 0

    assert fit.crit.used_fallback is False
    assert fit.high.used_fallback is False
    assert fit.crit.tau < fit.tau_low < fit.high.tau
    assert fit.crit.delta > 0.0
    assert fit.high.delta > 0.0
    # tau_crit must catch the severe violations, tau_high must be picky about surplus.
    assert fit.crit.recall_pos >= 0.85
    assert fit.high.precision_pos >= 0.80


def test_delta_high_falls_back_to_the_controller_default_without_queue_depth() -> None:
    fit = fit_delta_margins(_windows(queue_raw=False), theta=THETA)

    assert fit.labels.queue_available is False
    assert fit.labels.surplus_positive_count == 0
    assert fit.high.used_fallback is True
    assert fit.high.reject_reason == "insufficient_label_separation"
    assert fit.high.delta == pytest.approx(FALLBACK_DELTA_HIGH)
    # The critical side does not need the queue column and still fits.
    assert fit.crit.used_fallback is False


def test_delta_crit_falls_back_when_no_window_is_labelled_critical() -> None:
    rows = [
        _window(i, signal=150.0 + i, slo_met=True, latency_ratio_p95=0.5, queue_raw=2.0)
        for i in range(10)
    ]

    fit = fit_delta_margins(rows, theta=THETA)

    assert fit.crit.used_fallback is True
    assert fit.crit.delta == pytest.approx(FALLBACK_DELTA_CRIT)
    assert fit.crit.tau == pytest.approx(1.0 - FALLBACK_DELTA_CRIT)


def _floor_fixture() -> list[CalibrationWindow]:
    """Windows where the best-BA threshold misses the recall floor by exactly one window.

    20 critical windows: 16 at z <= 0.80, 2 in the last per-mille under z = 1, 2 above it.
    The best threshold (~0.84) recalls 16 of 20 = 0.80 with perfect specificity; the floor
    of 0.85 needs 17. Every candidate that clears the floor has to reach into the busy-but-
    healthy crowd just under tau_low, and the best of those is clipped to the bound - so
    the floor, applied first, buys 0.05 of recall for 0.20 of balanced accuracy. This is
    dsqwen-14b's situation in miniature (116 of 137 recalled, 117 needed).
    """
    rows: list[CalibrationWindow] = []
    index = 0
    for i in range(16):  # criticals, comfortably below the split
        rows.append(_window(index, signal=THETA * (0.50 + 0.02 * i), slo_met=False,
                            latency_ratio_p95=2.0, queue_raw=40.0))
        index += 1
    for z in (0.995, 0.997):  # criticals hiding in the last per-mille under tau_low
        rows.append(_window(index, signal=THETA * z, slo_met=False,
                            latency_ratio_p95=2.0, queue_raw=40.0))
        index += 1
    for z in (1.10, 1.20):  # criticals above tau_low: their quantiles get clipped
        rows.append(_window(index, signal=THETA * z, slo_met=False,
                            latency_ratio_p95=2.0, queue_raw=40.0))
        index += 1
    for i in range(30):  # healthy but busy: between the split and tau_low
        rows.append(_window(index, signal=THETA * (0.86 + 0.004 * i), slo_met=True,
                            latency_ratio_p95=0.85, queue_raw=14.0))
        index += 1
    for i in range(30):  # healthy with headroom
        rows.append(_window(index, signal=THETA * (1.10 + 0.02 * i), slo_met=True,
                            latency_ratio_p95=0.30, queue_raw=2.0))
        index += 1
    return rows


def test_soft_floor_keeps_the_best_balanced_accuracy_candidate() -> None:
    fit = fit_delta_margins(_floor_fixture(), theta=THETA)

    assert fit.crit.floor_mode == "soft"
    assert fit.crit.meets_target_floor is False  # one window short of 0.85
    assert fit.crit.recall_pos == pytest.approx(0.80)
    assert fit.crit.clamped is False and fit.crit.clamp_reason is None
    assert fit.crit.tau < 0.9  # a real threshold, not the tau_low bound
    assert fit.crit.delta > 0.05


def test_strict_floor_reproduces_the_clamped_choice_and_records_it() -> None:
    # Pins the former quantile rule (kept as a baseline via crit_method="quantile").
    soft = fit_delta_margins(_floor_fixture(), theta=THETA, crit_method="quantile")
    strict = fit_delta_margins(
        _floor_fixture(), theta=THETA, floor_mode="strict", crit_method="quantile"
    )

    assert strict.crit.floor_mode == "strict"
    assert strict.crit.meets_target_floor is True
    # The floor outranks the objective, so strict pays for it in balanced accuracy...
    assert strict.crit.balanced_accuracy < soft.crit.balanced_accuracy
    # ...and lands on the bound, which must now be visible rather than silent.
    assert strict.crit.clamped is True
    assert strict.crit.clamp_reason == "quantile_on_wrong_side_of_tau_low_clipped_to_bound"
    assert strict.crit.tau == pytest.approx(1.0 - 1e-6)
    assert strict.crit.used_fallback is False  # the old silent-clamp signature


def test_clamp_is_recorded_when_every_candidate_sits_above_tau_low() -> None:
    rows: list[CalibrationWindow] = []
    index = 0
    for i in range(10):  # all critical windows above tau_low: no candidate can fit below
        rows.append(_window(index, signal=THETA * (1.05 + 0.03 * i), slo_met=False,
                            latency_ratio_p95=2.0, queue_raw=40.0))
        index += 1
    for i in range(10):
        rows.append(_window(index, signal=THETA * (1.40 + 0.05 * i), slo_met=True,
                            latency_ratio_p95=0.30, queue_raw=2.0))
        index += 1

    fit = fit_delta_margins(rows, theta=THETA, crit_method="quantile")

    assert fit.crit.clamped is True
    assert fit.crit.clamp_reason is not None
    assert fit.crit.delta == pytest.approx(1e-6)

    # The BA grid reports the same situation as a delta on its lower edge.
    grid = fit_delta_margins(rows, theta=THETA)
    assert grid.crit.method == "ba_grid"
    assert grid.crit.delta == 0.0 and grid.crit.clamped is True
    assert grid.crit.clamp_reason == "delta_at_grid_lower_edge"


def _sharp_boundary() -> list[CalibrationWindow]:
    """Every critical window sits just under tau_low, every healthy one above it - the
    shape of 7b's S1/S3/T8 boundary, where the quantile rule clamped delta_crit to 0."""
    rows: list[CalibrationWindow] = []
    for i in range(12):
        # the lowest-z violations are the most severe ones (-> the critical labels)
        rows.append(_window(i, signal=THETA * (0.955 + 0.004 * i), slo_met=False,
                            latency_ratio_p95=3.2 - 0.1 * i, queue_raw=40.0))
    for i in range(12):
        rows.append(_window(20 + i, signal=THETA * (1.02 + 0.03 * i), slo_met=True,
                            latency_ratio_p95=0.4, queue_raw=2.0))
    return rows


def test_ba_grid_finds_a_real_delta_where_the_quantile_rule_clamps() -> None:
    grid = fit_delta_margins(_sharp_boundary(), theta=THETA)
    assert grid.crit.method == "ba_grid"
    assert grid.crit.balanced_accuracy == pytest.approx(1.0)
    assert grid.crit.clamped is False
    assert grid.crit.delta == pytest.approx(0.03)  # tau 0.97 splits 0.967 | 0.971
    # prediction is strict "z < tau_crit", exactly as classify_model decides CRITICAL
    assert grid.crit.tau == pytest.approx(1.0 - grid.crit.delta)


def test_delta_grid_is_bounded_to_half() -> None:
    from tre_calibration.fit import DEFAULT_DELTA_CRIT_GRID

    assert DEFAULT_DELTA_CRIT_GRID[0] == 0.0 and DEFAULT_DELTA_CRIT_GRID[-1] == 0.5
    assert len(DEFAULT_DELTA_CRIT_GRID) == 51


def test_unknown_crit_method_is_rejected() -> None:
    with pytest.raises(ValueError):
        fit_delta_margins(_windows(), theta=THETA, crit_method="guess")


def test_unknown_floor_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        fit_delta_margins(_windows(), theta=THETA, floor_mode="lenient")

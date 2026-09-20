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

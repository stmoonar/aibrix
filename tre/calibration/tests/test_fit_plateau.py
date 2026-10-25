"""Plateau-midpoint selection (fit method M2, user 2026-10-25) and the BA-argmax method it
sits next to, which must stay what produced the 2026-10-05 freeze."""
from __future__ import annotations

import math
import random

import pytest

from tre_calibration.dataset import CalibrationWindow
from tre_calibration.fit import (
    FIT_METHOD_ARGMAX,
    FIT_METHOD_M2,
    SELECTION_PLATEAU_MIDPOINT,
    ThetaFitConfig,
    fit_delta_margins,
    fit_method,
    fit_theta_by_balanced_accuracy,
    threshold_balanced_accuracy,
)


def _windows(seed: int = 20261025, n: int = 600) -> list[CalibrationWindow]:
    """A noisy boundary near signal 100: P(healthy) rises smoothly with the signal, so the
    BA curve has a flat top several candidates wide."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        signal = rng.uniform(20.0, 260.0)
        healthy = rng.random() < 1.0 / (1.0 + math.exp(-(signal - 100.0) / 18.0))
        ratio = rng.uniform(0.2, 0.9) if healthy else rng.uniform(1.0, 3.0) * (140.0 / signal)
        out.append(CalibrationWindow(
            scenario_id=f"cell-{i % 40}", scenario_family=("steady", "burst", "ramp")[i % 3],
            signal=round(signal, 3), slo_met=healthy, health_score=1.0 / (1.0 + ratio),
            latency_ratio_p95=ratio, latency_ratio_avg=0.8 * ratio, queue_raw=round(400.0 / signal, 3)))
    return out


def _ba(ws, theta: float) -> float:
    return threshold_balanced_accuracy(ws, theta=theta)["balanced_accuracy"]


def test_m2_theta_is_the_midpoint_of_the_ba_plateau() -> None:
    ws = _windows()
    argmax = fit_theta_by_balanced_accuracy(ws, candidate_grid="unique")
    m2 = fit_method(FIT_METHOD_M2).theta_config().fit(ws)
    lo, hi = m2.plateau
    assert m2.selection == SELECTION_PLATEAU_MIDPOINT and m2.plateau_eps == 0.005
    assert m2.theta == pytest.approx(0.5 * (lo + hi))
    # both plateau ends are within eps of the best BA, and the argmax lies inside
    assert min(_ba(ws, lo), _ba(ws, hi)) >= argmax.balanced_accuracy - 0.005 - 1e-12
    assert lo <= argmax.theta <= hi and hi > lo
    # the reported metrics are the midpoint's own
    assert m2.balanced_accuracy == pytest.approx(_ba(ws, m2.theta))


def test_m2_bands_are_the_midpoints_of_their_ba_plateaus() -> None:
    ws = _windows()
    theta = fit_method(FIT_METHOD_M2).theta_config().fit(ws).theta
    argmax = fit_delta_margins(ws, theta=theta)
    m2 = fit_delta_margins(ws, theta=theta, **fit_method(FIT_METHOD_M2).band_kwargs())
    for side, a in (("crit", argmax.crit), ("high", argmax.high)):
        s = getattr(m2, side)
        lo, hi = s.plateau
        assert s.selection == SELECTION_PLATEAU_MIDPOINT
        assert s.tau == pytest.approx(round(0.5 * (lo + hi), 6))
        assert lo - 1e-9 <= a.tau <= hi + 1e-9
    assert m2.crit.tau == pytest.approx(1.0 - m2.crit.delta)
    assert m2.high.tau == pytest.approx(1.0 + m2.high.delta)


#: The fixture's theta / tau_crit / tau_high under the BA argmax, computed with the code that
#: made the 2026-10-05 freeze (commit e8faa232): the named argmax method must keep them.
GOLDEN_ARGMAX = {"theta": 126.013, "tau_crit": 0.51, "tau_high": 1.42}


def test_the_argmax_method_is_the_2026_10_05_freeze_fit() -> None:
    ws = _windows()
    legacy = fit_method(FIT_METHOD_ARGMAX)
    assert legacy.theta_config() == ThetaFitConfig()          # the old default, field for field
    assert legacy.band_kwargs() == {}
    assert set(ThetaFitConfig().as_dict()) == {
        "direction", "healthy_quantile_candidates", "max_single_scenario_ratio", "min_confidence",
        "min_healthy_recall", "min_scenario_families", "min_support", "reliability_target", "theta_criterion"}
    fit = legacy.theta_config().fit(ws)
    d = fit_delta_margins(ws, theta=fit.theta)
    assert fit.theta == GOLDEN_ARGMAX["theta"]
    assert (d.crit.tau, d.high.tau) == (GOLDEN_ARGMAX["tau_crit"], GOLDEN_ARGMAX["tau_high"])

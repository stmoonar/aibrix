"""Scenario/cell-level bootstrap confidence interval for the reliability theta_m fit.

This is a permanent QA tool for calibration rounds, not a one-off script. It quantifies how
much the published ``theta_m`` (from :func:`tre_calibration.fit.fit_theta_by_reliability`)
could have moved had the load-scan happened to sample a slightly different set of grid cells.

The resampling unit is the **distinct ``scenario_id``** (one load-scan grid point / "cell"),
NOT the individual calibration window. Windows inside a cell are produced by a 30s window
sliding in 5s steps (ADR-0012), so consecutive windows share ~5/6 of their raw requests and
are heavily autocorrelated; resampling windows directly would treat those near-duplicates as
independent draws and badly understate the true sampling variance. Cell-level resampling keeps
each grid point atomic: a cell drawn twice contributes its whole (correlated) window block
twice, so the CI reflects variability across grid points, which is the thing an operator
actually re-rolls when they re-run a load scan.

The fit itself is reused verbatim -- this module never reimplements the reliability logic; it
only re-feeds resampled window lists into ``fit_theta_by_reliability`` with the exact same
config the caller passes (which callers set to the production fit-config).
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

from tre_calibration.dataset import CalibrationWindow
from tre_calibration.fit import fit_theta_by_reliability


@dataclass(frozen=True)
class BootstrapThetaResult:
    """Distribution of ``theta_m`` across cell-level bootstrap resamples.

    ``theta_values`` collects ``fit.theta`` only from resamples where ``fit.publish`` was True;
    resamples whose fit was rejected (coverage/support/confidence gate) are still counted in
    ``n_resamples`` so ``publish_rate`` (= ``n_published / n_resamples``) is informative. When
    no resample published, the summary statistics are ``None``.
    """

    n_resamples: int
    n_published: int
    theta_values: tuple[float, ...]
    theta_p2_5: float | None
    theta_p50: float | None
    theta_p97_5: float | None
    theta_mean: float | None
    theta_std: float | None
    publish_rate: float


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolation percentile (numpy default 'linear' method) on a sorted list."""
    if not sorted_vals:
        raise ValueError("percentile of empty sequence")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def bootstrap_theta(
    windows: list[CalibrationWindow],
    *,
    n_resamples: int,
    seed: int,
    reliability_target: float,
    min_support: int,
    min_confidence: float,
    min_scenario_families: int,
    max_single_scenario_ratio: float,
) -> BootstrapThetaResult:
    """Cell-level bootstrap of the reliability ``theta_m`` fit.

    Algorithm: group ``windows`` by ``scenario_id`` (cells), let ``cells`` be the sorted set of
    cell ids and ``n_cells = len(cells)``. Seed a single ``random.Random(seed)`` ONCE and reuse
    it across iterations (a fresh ``Random(seed)`` per iteration would make every iteration
    identical). For each of ``n_resamples`` iterations draw ``n_cells`` cell ids with
    replacement (``rng.choices(cells, k=n_cells)``), concatenate -- in order -- ALL of each drawn
    cell's original windows (a cell drawn twice contributes its windows twice), and run
    ``fit_theta_by_reliability`` on that resampled list with the exact config passed here. Record
    ``fit.theta`` iff ``fit.publish``. Deterministic given ``seed``.
    """
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    by_cell: dict[str, list[CalibrationWindow]] = {}
    for window in windows:
        by_cell.setdefault(window.scenario_id, []).append(window)
    cells = sorted(by_cell)
    n_cells = len(cells)

    rng = random.Random(seed)
    published: list[float] = []
    for _ in range(n_resamples):
        drawn = rng.choices(cells, k=n_cells) if n_cells else []
        resampled: list[CalibrationWindow] = []
        for cell_id in drawn:
            resampled.extend(by_cell[cell_id])
        fit = fit_theta_by_reliability(
            resampled,
            reliability_target=reliability_target,
            min_support=min_support,
            min_confidence=min_confidence,
            min_scenario_families=min_scenario_families,
            max_single_scenario_ratio=max_single_scenario_ratio,
        )
        if fit.publish and fit.theta is not None:
            published.append(fit.theta)

    n_published = len(published)
    publish_rate = n_published / n_resamples
    if n_published:
        srt = sorted(published)
        theta_p2_5: float | None = _percentile(srt, 2.5)
        theta_p50 = _percentile(srt, 50.0)
        theta_p97_5 = _percentile(srt, 97.5)
        theta_mean = statistics.fmean(published)
        theta_std = statistics.pstdev(published) if n_published > 1 else 0.0
    else:
        theta_p2_5 = theta_p50 = theta_p97_5 = theta_mean = theta_std = None

    return BootstrapThetaResult(
        n_resamples=n_resamples,
        n_published=n_published,
        theta_values=tuple(published),
        theta_p2_5=theta_p2_5,
        theta_p50=theta_p50,
        theta_p97_5=theta_p97_5,
        theta_mean=theta_mean,
        theta_std=theta_std,
        publish_rate=publish_rate,
    )

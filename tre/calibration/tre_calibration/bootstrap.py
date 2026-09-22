"""Scenario/cell-level bootstrap confidence interval for the ``theta_m`` fit.

This is a permanent QA tool for calibration rounds, not a one-off script. It quantifies how
much the published ``theta_m`` could have moved had the load scan happened to sample a
slightly different set of grid cells.

The resampling unit is the **distinct ``scenario_id``** (one load-scan grid point / "cell"),
NOT the individual calibration window. Windows inside a cell are produced by a 30s window
sliding in 5s steps (ADR-0012), so consecutive windows share ~5/6 of their raw requests and
are heavily autocorrelated; resampling windows directly would treat those near-duplicates as
independent draws and badly understate the true sampling variance. Cell-level resampling keeps
each grid point atomic: a cell drawn twice contributes its whole (correlated) window block
twice, so the CI reflects variability across grid points, which is the thing an operator
actually re-rolls when they re-run a load scan.

The fit itself is reused verbatim: this module never reimplements a fitting criterion, it
re-feeds resampled window lists into :meth:`tre_calibration.fit.ThetaFitConfig.fit`. The
caller hands over one :class:`~tre_calibration.fit.ThetaFitConfig`, and that same object must
also produce the point estimate the interval is reported around. Carrying the whole
configuration rather than individual knobs is what keeps the interval and the threshold it
brackets the same quantity: an interval fitted under a different criterion, orientation or
acceptance gate than the published theta is an interval for something else, and a campaign
stop rule reading ``publish_rate`` or the CI half-width off it would be measuring the wrong
thing.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

from typing import Any, Mapping

from tre_calibration.dataset import CalibrationWindow
from tre_calibration.fit import ThetaFitConfig, fit_delta_margins


@dataclass(frozen=True)
class BootstrapThetaResult:
    """Distribution of ``theta_m`` across cell-level bootstrap resamples.

    ``theta_values`` collects ``fit.theta`` only from resamples where ``fit.publish`` was True;
    resamples whose fit was rejected (coverage/support/confidence gate) are still counted in
    ``n_resamples`` so ``publish_rate`` (= ``n_published / n_resamples``) is informative. When
    no resample published, the summary statistics are ``None``.

    ``config`` is the configuration every resample was fitted under, kept on the result so a
    reader never has to trust that the caller used the same one for the point estimate.
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
    config: ThetaFitConfig = ThetaFitConfig()


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
    config: ThetaFitConfig | None = None,
) -> BootstrapThetaResult:
    """Cell-level bootstrap of the ``theta_m`` fit under ``config``.

    Algorithm: group ``windows`` by ``scenario_id`` (cells), let ``cells`` be the sorted set of
    cell ids and ``n_cells = len(cells)``. Seed a single ``random.Random(seed)`` ONCE and reuse
    it across iterations (a fresh ``Random(seed)`` per iteration would make every iteration
    identical). For each of ``n_resamples`` iterations draw ``n_cells`` cell ids with
    replacement (``rng.choices(cells, k=n_cells)``), concatenate -- in order -- ALL of each drawn
    cell's original windows (a cell drawn twice contributes its windows twice), and run
    ``config.fit`` on that resampled list. Record ``fit.theta`` iff ``fit.publish``.
    Deterministic given ``seed``.

    ``config`` defaults to :class:`~tre_calibration.fit.ThetaFitConfig`'s defaults, which are
    the calibration CLI's defaults; pass the caller's own configuration -- the same object used
    for the point estimate -- whenever the fit was not run at defaults.
    """
    result, _deltas = _bootstrap(windows, n_resamples=n_resamples, seed=seed, config=config)
    return result


@dataclass(frozen=True)
class BootstrapDeltaResult:
    """Distribution of ``delta_crit`` across the same cell-level resamples as theta.

    Each resample refits theta under the theta configuration and, when it publishes,
    fits ``delta_crit`` on that resample at that theta; ``n_fitted`` counts the resamples
    whose delta fit did not fall back to the default margin.
    """

    n_resamples: int
    n_fitted: int
    delta_values: tuple[float, ...]
    delta_p2_5: float | None
    delta_p50: float | None
    delta_p97_5: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_resamples": self.n_resamples,
            "n_fitted": self.n_fitted,
            "delta_p2_5": self.delta_p2_5,
            "delta_p50": self.delta_p50,
            "delta_p97_5": self.delta_p97_5,
        }


def bootstrap_theta_and_delta_crit(
    windows: list[CalibrationWindow],
    *,
    n_resamples: int,
    seed: int,
    config: ThetaFitConfig | None = None,
    delta_kwargs: Mapping[str, Any] | None = None,
) -> tuple[BootstrapThetaResult, BootstrapDeltaResult]:
    """:func:`bootstrap_theta` plus a delta_crit fit per resample (plan §6.3 B6).

    The resample draws are exactly :func:`bootstrap_theta`'s for the same ``seed``, so the
    theta interval it returns is identical to a plain theta bootstrap.
    """
    result, deltas = _bootstrap(
        windows, n_resamples=n_resamples, seed=seed, config=config,
        delta_kwargs=dict(delta_kwargs or {}),
    )
    assert deltas is not None
    if deltas:
        srt = sorted(deltas)
        summary = BootstrapDeltaResult(
            n_resamples=n_resamples,
            n_fitted=len(deltas),
            delta_values=tuple(deltas),
            delta_p2_5=_percentile(srt, 2.5),
            delta_p50=_percentile(srt, 50.0),
            delta_p97_5=_percentile(srt, 97.5),
        )
    else:
        summary = BootstrapDeltaResult(n_resamples, 0, (), None, None, None)
    return result, summary


def _bootstrap(
    windows: list[CalibrationWindow],
    *,
    n_resamples: int,
    seed: int,
    config: ThetaFitConfig | None,
    delta_kwargs: dict[str, Any] | None = None,
) -> tuple[BootstrapThetaResult, list[float] | None]:
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    config = config or ThetaFitConfig()
    deltas: list[float] | None = [] if delta_kwargs is not None else None

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
        fit = config.fit(resampled)
        if fit.publish and fit.theta is not None:
            published.append(fit.theta)
            if deltas is not None and fit.theta > 0.0:
                crit = fit_delta_margins(resampled, theta=fit.theta, **(delta_kwargs or {})).crit
                if not crit.used_fallback:
                    deltas.append(crit.delta)

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
        config=config,
    ), deltas

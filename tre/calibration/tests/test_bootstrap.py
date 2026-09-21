from __future__ import annotations

import inspect

import pytest

from tre_calibration.bootstrap import BootstrapThetaResult, bootstrap_theta
from tre_calibration.dataset import CalibrationWindow
from tre_calibration.fit import (
    DEFAULT_THETA_CRITERION,
    ThetaFitConfig,
    fit_theta,
)

_CONFIG = ThetaFitConfig(criterion="reliability")


def _separable_windows() -> list[CalibrationWindow]:
    """Two families x four cells each, cleanly separable: every violating window sits at a
    single low trs (50) far below the healthy band (>=100), so the reliability fit always
    lands theta at exactly 100.0 whenever the resample covers both families."""
    rows: list[CalibrationWindow] = []
    for family in ("steady", "burst"):
        for k in range(4):
            cid = f"{family}-{k}"
            rows.append(CalibrationWindow(cid, family, 50.0, False))
            rows.append(CalibrationWindow(cid, family, 100.0, True))
            rows.append(CalibrationWindow(cid, family, 130.0, True))
    return rows


def _overlapping_windows() -> list[CalibrationWindow]:
    """Cells whose healthy and violating bands overlap between trs 50 and 70.

    The two criteria answer differently here, which is the point. The containment rule
    walks up from the bottom and stops at the first threshold whose whole upper set
    attains its reliability target, so it must clear the top of the violating band and
    lands at 80. Balanced accuracy weighs the two error kinds against each other and lands
    at 70, inside the overlap. A CI computed under one criterion and reported beside a
    theta fitted under the other is the spread of a threshold nobody published.
    """
    rows: list[CalibrationWindow] = []
    for family in ("steady", "burst"):
        for cell in range(4):
            cid = f"{family}-{cell}"
            for signal in (40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0):
                rows.append(CalibrationWindow(cid, family, signal, False))
            for signal in (50.0, 60.0, 70.0, 80.0, 90.0, 100.0):
                rows.append(CalibrationWindow(cid, family, signal, True))
    return rows


def test_bootstrap_is_deterministic_for_same_seed() -> None:
    windows = _separable_windows()
    a = bootstrap_theta(windows, n_resamples=200, seed=7, config=_CONFIG)
    b = bootstrap_theta(windows, n_resamples=200, seed=7, config=_CONFIG)
    assert a == b
    assert isinstance(a, BootstrapThetaResult)


def test_bootstrap_different_seed_changes_draws_but_ci_stays_sane() -> None:
    windows = _separable_windows()
    a = bootstrap_theta(windows, n_resamples=200, seed=1, config=_CONFIG)
    b = bootstrap_theta(windows, n_resamples=200, seed=2, config=_CONFIG)
    # different seeds -> generally different resample sequences (values may differ)
    assert a.theta_values != b.theta_values or a.n_published != b.n_published


def test_bootstrap_ci_brackets_point_theta_on_clean_data() -> None:
    windows = _separable_windows()
    point = _CONFIG.fit(windows)
    assert point.publish is True

    assert point.theta == 100.0

    result = bootstrap_theta(windows, n_resamples=500, seed=42, config=_CONFIG)
    # 8 cells across 2 families: most resamples publish, but lopsided draws (e.g. 6 steady /
    # 2 burst) push the winning subset's max family ratio past 0.7 and fail the coverage gate,
    # so publish_rate is high-but-not-1 -- exactly the kind of fragility this CI is meant to show.
    assert result.publish_rate > 0.5
    # ordered percentiles, and the point estimate lands inside the 95% CI.
    assert result.theta_p2_5 <= result.theta_p50 <= result.theta_p97_5
    assert result.theta_p2_5 <= point.theta <= result.theta_p97_5
    # boundary is fully stable -> every published resample fits theta at exactly 100.0.
    assert result.theta_p2_5 == result.theta_p50 == result.theta_p97_5 == 100.0


def test_bootstrap_publish_rate_between_zero_and_one_when_coverage_is_fragile() -> None:
    """One cell per family: resamples that draw only a single family fail the >=2-family
    coverage gate, so some publish and some do not -> 0 < publish_rate < 1."""
    windows = [
        CalibrationWindow("cell-a", "family-a", 50.0, False),
        CalibrationWindow("cell-a", "family-a", 100.0, True),
        CalibrationWindow("cell-a", "family-a", 110.0, True),
        CalibrationWindow("cell-b", "family-b", 60.0, False),
        CalibrationWindow("cell-b", "family-b", 105.0, True),
        CalibrationWindow("cell-b", "family-b", 120.0, True),
    ]
    result = bootstrap_theta(windows, n_resamples=400, seed=42, config=_CONFIG)
    assert 0.0 < result.publish_rate < 1.0
    assert result.n_published == len(result.theta_values)
    # published resamples are the mixed draws, which fit theta at the 100.0 boundary.
    assert result.theta_p50 == 100.0


def test_bootstrap_defaults_to_the_criterion_the_fit_cli_publishes() -> None:
    """An unconfigured bootstrap must bracket the theta an unconfigured fit publishes."""
    assert ThetaFitConfig().criterion == DEFAULT_THETA_CRITERION
    assert (
        ThetaFitConfig().criterion
        == inspect.signature(fit_theta).parameters["criterion"].default
    )
    result = bootstrap_theta(_overlapping_windows(), n_resamples=20, seed=3)
    assert result.config == ThetaFitConfig()
    assert result.config.criterion == DEFAULT_THETA_CRITERION


def test_every_resample_is_fitted_under_the_point_estimates_whole_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resample fits and the point estimate must share one configuration object.

    Not just the criterion: a CI whose resamples used a different orientation, healthy
    quantile grid or acceptance gate is an interval for a different quantity than the
    published theta, and the campaign stop rule reads that interval.
    """
    config = ThetaFitConfig(
        criterion="balanced_accuracy",
        min_healthy_recall=0.25,
        healthy_quantile_candidates=(0.1, 0.3),
        max_single_scenario_ratio=0.9,
    )
    seen: list[ThetaFitConfig] = []
    original_fit = ThetaFitConfig.fit

    def _recording_fit(self: ThetaFitConfig, windows):  # type: ignore[no-untyped-def]
        seen.append(self)
        return original_fit(self, windows)

    monkeypatch.setattr(ThetaFitConfig, "fit", _recording_fit)

    point = config.fit(_overlapping_windows())
    result = bootstrap_theta(_overlapping_windows(), n_resamples=25, seed=11, config=config)

    assert point.publish is True
    assert len(seen) == 26  # the point estimate plus one fit per resample
    # Same object, not merely an equal one: there is one configuration in play.
    assert all(used is config for used in seen)
    assert result.config is config


def test_the_two_criteria_give_different_intervals_on_the_same_windows() -> None:
    """The fix has teeth only if the criterion moves the interval, not just its label."""
    windows = _overlapping_windows()
    balanced_config = ThetaFitConfig(criterion="balanced_accuracy")
    reliability_config = ThetaFitConfig(criterion="reliability")

    balanced_point = balanced_config.fit(windows)
    reliability_point = reliability_config.fit(windows)
    assert balanced_point.publish is True
    assert reliability_point.publish is True
    assert balanced_point.theta != reliability_point.theta

    balanced = bootstrap_theta(windows, n_resamples=300, seed=5, config=balanced_config)
    reliability = bootstrap_theta(windows, n_resamples=300, seed=5, config=reliability_config)

    assert balanced.n_published > 0 and reliability.n_published > 0
    # Distinct intervals: reporting one of them beside the other criterion's theta would
    # be reporting the spread of a threshold nobody published.
    assert (balanced.theta_p2_5, balanced.theta_p97_5) != (
        reliability.theta_p2_5,
        reliability.theta_p97_5,
    )
    # And each interval brackets its own criterion's point estimate.
    assert balanced.theta_p2_5 <= balanced_point.theta <= balanced.theta_p97_5
    assert reliability.theta_p2_5 <= reliability_point.theta <= reliability.theta_p97_5

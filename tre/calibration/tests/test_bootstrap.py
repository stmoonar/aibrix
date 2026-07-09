from __future__ import annotations

from tre_calibration.bootstrap import BootstrapThetaResult, bootstrap_theta
from tre_calibration.dataset import CalibrationWindow
from tre_calibration.fit import fit_theta_by_reliability

_CONFIG = dict(
    reliability_target=0.9,
    min_support=3,
    min_confidence=0.9,
    min_scenario_families=2,
    max_single_scenario_ratio=0.7,
)


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


def test_bootstrap_is_deterministic_for_same_seed() -> None:
    windows = _separable_windows()
    a = bootstrap_theta(windows, n_resamples=200, seed=7, **_CONFIG)
    b = bootstrap_theta(windows, n_resamples=200, seed=7, **_CONFIG)
    assert a == b
    assert isinstance(a, BootstrapThetaResult)


def test_bootstrap_different_seed_changes_draws_but_ci_stays_sane() -> None:
    windows = _separable_windows()
    a = bootstrap_theta(windows, n_resamples=200, seed=1, **_CONFIG)
    b = bootstrap_theta(windows, n_resamples=200, seed=2, **_CONFIG)
    # different seeds -> generally different resample sequences (values may differ)
    assert a.theta_values != b.theta_values or a.n_published != b.n_published


def test_bootstrap_ci_brackets_point_theta_on_clean_data() -> None:
    windows = _separable_windows()
    point = fit_theta_by_reliability(windows, **_CONFIG)
    assert point.publish is True

    assert point.theta == 100.0

    result = bootstrap_theta(windows, n_resamples=500, seed=42, **_CONFIG)
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
    result = bootstrap_theta(windows, n_resamples=400, seed=42, **_CONFIG)
    assert 0.0 < result.publish_rate < 1.0
    assert result.n_published == len(result.theta_values)
    # published resamples are the mixed draws, which fit theta at the 100.0 boundary.
    assert result.theta_p50 == 100.0

"""``ThetaFitConfig`` is the one description of how a theta was fitted.

Every tool that fits theta more than once -- the bootstrap CI, the train/test ranking
separation report -- threads one of these through, so the tests here pin the properties
that make that worth doing: its defaults are ``fit_theta``'s defaults (which are the fit
CLI's defaults), it fits through ``fit_theta`` rather than through a criterion of its own,
and it records itself with the key names the calibration artifact uses.
"""
from __future__ import annotations

import inspect

import pytest

from tre_calibration.cli import _parse_args
from tre_calibration.dataset import CalibrationWindow
from tre_calibration.fit import (
    DEFAULT_SIGNAL_DIRECTION,
    DEFAULT_THETA_CRITERION,
    ThetaFitConfig,
    fit_theta,
)

_FIT_ARGV = [
    "--input", "windows.csv",
    "--output", "patch.json",
    "--model-name", "dsqwen-7b",
    "--ttft-p95-ms", "100",
    "--tpot-p95-ms", "50",
]


def _windows() -> list[CalibrationWindow]:
    rows: list[CalibrationWindow] = []
    for family in ("steady", "burst"):
        for cell in range(3):
            cid = f"{family}-{cell}"
            for signal in (40.0, 50.0, 60.0, 70.0):
                rows.append(CalibrationWindow(cid, family, signal, False))
            for signal in (50.0, 70.0, 90.0, 110.0):
                rows.append(CalibrationWindow(cid, family, signal, True))
    return rows


def test_config_defaults_are_fit_theta_defaults() -> None:
    """One default per knob, defined once: a drifting copy is what this catches."""
    defaults = inspect.signature(fit_theta).parameters
    config = ThetaFitConfig()
    for field, parameter in (
        ("criterion", "criterion"),
        ("direction", "direction"),
        ("healthy_quantile_candidates", "healthy_quantile_candidates"),
        ("min_healthy_recall", "min_healthy_recall"),
        ("reliability_target", "reliability_target"),
        ("min_support", "min_support"),
        ("min_confidence", "min_confidence"),
        ("min_scenario_families", "min_scenario_families"),
        ("max_single_scenario_ratio", "max_single_scenario_ratio"),
    ):
        assert getattr(config, field) == defaults[parameter].default, field
    assert config.criterion == DEFAULT_THETA_CRITERION
    assert config.direction == DEFAULT_SIGNAL_DIRECTION


def test_config_defaults_are_the_fit_clis_defaults() -> None:
    """What an unflagged calibration run publishes is what an unconfigured config fits."""
    args = _parse_args(_FIT_ARGV)
    config = ThetaFitConfig()
    assert config.criterion == args.theta_criterion
    assert config.direction == args.direction
    assert config.min_healthy_recall == args.min_healthy_recall
    assert config.reliability_target == args.reliability_target
    assert config.min_support == args.min_support
    assert config.min_confidence == args.min_confidence
    assert config.min_scenario_families == args.min_scenario_families
    assert config.max_single_scenario_ratio == args.max_single_scenario_ratio
    assert args.healthy_quantiles == ""  # empty means "the config's candidate grid"


def test_config_fit_matches_calling_fit_theta_with_the_same_knobs() -> None:
    windows = _windows()
    for criterion in ("balanced_accuracy", "reliability"):
        config = ThetaFitConfig(criterion=criterion, min_scenario_families=2)
        assert config.fit(windows) == fit_theta(
            windows,
            criterion=criterion,
            direction=config.direction,
            healthy_quantile_candidates=config.healthy_quantile_candidates,
            min_healthy_recall=config.min_healthy_recall,
            reliability_target=config.reliability_target,
            min_support=config.min_support,
            min_confidence=config.min_confidence,
            min_scenario_families=config.min_scenario_families,
            max_single_scenario_ratio=config.max_single_scenario_ratio,
        )


def test_as_dict_is_keyed_like_the_artifacts_fit_config() -> None:
    record = ThetaFitConfig().as_dict()
    assert record["theta_criterion"] == DEFAULT_THETA_CRITERION
    assert record["direction"] == DEFAULT_SIGNAL_DIRECTION
    assert isinstance(record["healthy_quantile_candidates"], list)
    # Every key the fit CLI writes into fit_config for the theta fit appears here too, so a
    # report and the artifact it describes can be diffed field by field.
    assert set(record) <= {
        "direction",
        "healthy_quantile_candidates",
        "max_single_scenario_ratio",
        "min_confidence",
        "min_healthy_recall",
        "min_scenario_families",
        "min_support",
        "reliability_target",
        "theta_criterion",
    }


def test_config_rejects_an_unknown_criterion_or_orientation() -> None:
    with pytest.raises(ValueError):
        ThetaFitConfig(criterion="eyeball")
    with pytest.raises(ValueError):
        ThetaFitConfig(direction="sideways")


def test_config_is_hashable_and_comparable() -> None:
    """Frozen and value-equal, so a report can state the config it ran under."""
    assert ThetaFitConfig() == ThetaFitConfig()
    assert ThetaFitConfig(criterion="reliability") != ThetaFitConfig()
    assert len({ThetaFitConfig(), ThetaFitConfig()}) == 1
    # A list of candidates is normalised to a tuple, so equality is not order-of-typing.
    assert ThetaFitConfig(healthy_quantile_candidates=[0.1, 0.2]) == ThetaFitConfig(
        healthy_quantile_candidates=(0.1, 0.2)
    )

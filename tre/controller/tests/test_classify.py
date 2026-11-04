from __future__ import annotations

from typing import Any

import pytest

from golden.legacy_classify import (
    LegacyModelClassification,
    LegacyTauThresholds,
    legacy_classify_all_models,
    legacy_classify_model,
)
from tre_controller.planning.classify import (
    ModelClassification,
    TauThresholds,
    classify_all_models,
    classify_model,
)


def _simple(item: ModelClassification | LegacyModelClassification) -> dict[str, Any]:
    # 2026-11-02: eta_m / eta_crit / eta_low / donor_tier removed from the controller; the
    # legacy reference still carries them, so only the band classification is compared.
    return {
        "model_name": item.model_name,
        "state": item.state.value,
        "role": item.role.value,
        "Z_m": item.Z_m,
        "trs": item.trs,
        "theta_m": item.theta_m,
        "tau_low": item.tau.tau_low,
        "tau_crit": item.tau.tau_crit,
        "tau_high": item.tau.tau_high,
    }


@pytest.mark.parametrize("z_m", [None, 0.79, 0.80, 0.99, 1.0, 1.25, 1.26, 1.5])
def test_classify_model_matches_legacy_boundaries(z_m: float | None) -> None:
    legacy_tau = LegacyTauThresholds.from_control(delta_crit=0.2, delta_high=0.25)
    migrated_tau = TauThresholds.from_control(delta_crit=0.2, delta_high=0.25)

    expected = legacy_classify_model(
        model_name="m",
        trs=1234.0,
        Z_m=z_m,
        eta_m=None,
        theta_m=1000.0,
        tau=legacy_tau,
    )
    actual = classify_model(
        model_name="m",
        trs=1234.0,
        Z_m=z_m,
        theta_m=1000.0,
        tau=migrated_tau,
    )

    assert _simple(actual) == _simple(expected)


def test_classify_all_models_matches_legacy_zero_load_and_model_controls() -> None:
    contexts = {
        "idle": {"Y_m": 0.0, "Q": 0.0, "z_m": 9.0, "trs": 99.0, "theta_m": 11.0},
        "critical": {"Y_m": 100.0, "Q": 2.0, "z_m": 0.69, "trs": 690.0, "theta_m": 1000.0},
        "low": {"Y_m": 100.0, "Q": 2.0, "z_m": 0.95, "trs": 950.0, "theta_m": 1000.0},
        "healthy": {"Y_m": 100.0, "Q": 2.0, "z_m": 1.1, "trs": 1100.0, "theta_m": 1000.0},
        "high": {"Y_m": 100.0, "Q": 2.0, "z_m": 1.51, "trs": 1510.0, "theta_m": 1000.0},
        "unknown": {"Y_m": 100.0, "Q": 2.0, "z_m": None, "trs": 0.0, "theta_m": None},
    }
    controls = {
        "critical": {"delta_crit": 0.3, "delta_high": 0.4},
        "high": {"delta_crit": 0.1, "delta_high": 0.5},
    }

    expected = legacy_classify_all_models(contexts, model_control_configs=controls)
    actual = classify_all_models(contexts, model_control_configs=controls)

    assert [_simple(item) for item in actual] == [_simple(item) for item in expected]


def test_idle_rps_guard_neutralizes_low_rate_nonzero_queue_signal() -> None:
    classification = classify_model(
        model_name="queue-arm",
        trs=0.0,
        Z_m=0.1,
        theta_m=None,
        tau=TauThresholds.from_control(),
        request_rate_rps=0.01,
        idle_rps_eps=0.05,
    )

    assert classification.state.value == "healthy"
    assert classification.role.value == "neutral"


def test_idle_rps_guard_is_noop_for_recorded_busy_zm_contexts() -> None:
    contexts = {
        "critical": {
            "Y_m": 100.0,
            "Q": 2.0,
            "z_m": 0.7,
            "trs": 700.0,
            "theta_m": 1000.0,
            "request_rate_rps": 0.8,
        },
        "healthy": {
            "Y_m": 100.0,
            "Q": 1.0,
            "z_m": 1.1,
            "trs": 1100.0,
            "theta_m": 1000.0,
            "request_rate_rps": 1.2,
        },
        "high": {
            "Y_m": 100.0,
            "Q": 1.0,
            "z_m": 1.5,
            "trs": 1500.0,
            "theta_m": 1000.0,
            "request_rate_rps": 0.4,
        },
    }

    without_guard = classify_all_models(contexts, signal_idle_rps_eps=0.0)
    with_guard = classify_all_models(contexts, signal_idle_rps_eps=0.05)

    assert [_simple(item) for item in with_guard] == [
        _simple(item) for item in without_guard
    ]


def test_zero_load_still_uses_idle_reclamation_tier() -> None:
    contexts = {
        "idle": {
            "Y_m": 0.0,
            "Q": 0.0,
            "z_m": 0.0,
            "trs": 0.0,
            "theta_m": 1000.0,
            "request_rate_rps": 0.0,
        }
    }

    classification = classify_all_models(contexts, signal_idle_rps_eps=0.05)[0]
    assert classification.state.value == "idle"
    assert classification.role.value == "donor"

from __future__ import annotations

from typing import Any

import pytest

from golden.legacy_classify import (
    LegacyModelClassification,
    LegacyTauThresholds,
    legacy_build_comparison_log,
    legacy_classify_all_models,
    legacy_classify_model,
    legacy_filter_donors_by_eta,
    legacy_split_receivers_donors,
)
from tre_controller.planning.classify import (
    ModelClassification,
    TauThresholds,
    build_comparison_log,
    classify_all_models,
    classify_model,
    donor_mock_cost_key,
    filter_donors_by_eta,
    split_receivers_donors,
)


def _simple(item: ModelClassification | LegacyModelClassification) -> dict[str, Any]:
    return {
        "model_name": item.model_name,
        "state": item.state.value,
        "role": item.role.value,
        "Z_m": item.Z_m,
        "eta_m": item.eta_m,
        "trs": item.trs,
        "theta_m": item.theta_m,
        "tau_low": item.tau.tau_low,
        "tau_crit": item.tau.tau_crit,
        "tau_high": item.tau.tau_high,
        "donor_tier": item.donor_tier,
        "eta_crit": item.eta_crit,
        "eta_low": item.eta_low,
    }


@pytest.mark.parametrize(
    "z_m,eta_m",
    [
        (None, None),
        (0.79, 100.0),
        (0.80, 150.0),
        (0.99, 200.0),
        (1.0, 250.0),
        (1.25, 350.0),
        (1.26, 250.0),
        (1.5, 350.0),
    ],
)
def test_classify_model_matches_legacy_boundaries(z_m: float | None, eta_m: float | None) -> None:
    legacy_tau = LegacyTauThresholds.from_control(delta_crit=0.2, delta_high=0.25)
    migrated_tau = TauThresholds.from_control(delta_crit=0.2, delta_high=0.25)

    expected = legacy_classify_model(
        model_name="m",
        trs=1234.0,
        Z_m=z_m,
        eta_m=eta_m,
        theta_m=1000.0,
        tau=legacy_tau,
        eta_crit=200.0,
        eta_low=300.0,
    )
    actual = classify_model(
        model_name="m",
        trs=1234.0,
        Z_m=z_m,
        eta_m=eta_m,
        theta_m=1000.0,
        tau=migrated_tau,
        eta_crit=200.0,
        eta_low=300.0,
    )

    assert _simple(actual) == _simple(expected)


def test_classify_all_models_matches_legacy_zero_load_and_model_controls() -> None:
    contexts = {
        "idle": {"Y_m": 0.0, "Q": 0.0, "z_m": 9.0, "eta_m": 0.0, "trs": 99.0, "theta_m": 11.0},
        "critical": {"Y_m": 100.0, "Q": 2.0, "z_m": 0.69, "eta_m": 250.0, "trs": 690.0, "theta_m": 1000.0},
        "low": {"Y_m": 100.0, "Q": 2.0, "z_m": 0.95, "eta_m": 320.0, "trs": 950.0, "theta_m": 1000.0},
        "healthy": {"Y_m": 100.0, "Q": 2.0, "z_m": 1.1, "eta_m": 330.0, "trs": 1100.0, "theta_m": 1000.0},
        "high": {"Y_m": 100.0, "Q": 2.0, "z_m": 1.51, "eta_m": 350.0, "trs": 1510.0, "theta_m": 1000.0},
        "unknown": {"Y_m": 100.0, "Q": 2.0, "z_m": None, "eta_m": None, "trs": 0.0, "theta_m": None},
    }
    controls = {
        "critical": {"delta_crit": 0.3, "delta_high": 0.4, "receiver_thrashing_eff": 125.0, "donor_waste_eff": 425.0},
        "high": {"delta_crit": 0.1, "delta_high": 0.5, "receiver_thrashing_eff": 100.0, "donor_waste_eff": 360.0},
    }

    expected = legacy_classify_all_models(contexts, model_control_configs=controls)
    actual = classify_all_models(contexts, model_control_configs=controls)

    assert [_simple(item) for item in actual] == [_simple(item) for item in expected]


def test_split_receivers_donors_matches_legacy_sorting_and_eta_gate() -> None:
    contexts = {
        "low_z": {"Y_m": 1.0, "Q": 1.0, "z_m": 0.7, "eta_m": 100.0, "trs": 700.0, "theta_m": 1000.0},
        "crit_higher_z": {"Y_m": 1.0, "Q": 1.0, "z_m": 0.75, "eta_m": 100.0, "trs": 750.0, "theta_m": 1000.0},
        "low_receiver": {"Y_m": 1.0, "Q": 1.0, "z_m": 0.9, "eta_m": 100.0, "trs": 900.0, "theta_m": 1000.0},
        "idle": {"Y_m": 0.0, "Q": 0.0, "z_m": 10.0, "eta_m": 0.0, "trs": 0.0, "theta_m": 1000.0},
        "waste": {"Y_m": 1.0, "Q": 1.0, "z_m": 1.4, "eta_m": 250.0, "trs": 1400.0, "theta_m": 1000.0},
        "surplus": {"Y_m": 1.0, "Q": 1.0, "z_m": 1.6, "eta_m": 500.0, "trs": 1600.0, "theta_m": 1000.0},
        "filtered": {"Y_m": 1.0, "Q": 1.0, "z_m": 1.8, "eta_m": 100.0, "trs": 1800.0, "theta_m": 1000.0},
    }
    controls = {
        "waste": {"receiver_thrashing_eff": 200.0, "donor_waste_eff": 300.0},
        "surplus": {"receiver_thrashing_eff": 200.0, "donor_waste_eff": 300.0},
        "filtered": {"receiver_thrashing_eff": 200.0, "donor_waste_eff": 300.0},
    }

    expected_cls = legacy_classify_all_models(contexts, model_control_configs=controls)
    actual_cls = classify_all_models(contexts, model_control_configs=controls)
    expected_receivers, expected_donors = legacy_split_receivers_donors(expected_cls)
    actual_receivers, actual_donors = split_receivers_donors(actual_cls)

    assert [item.model_name for item in actual_receivers] == [item.model_name for item in expected_receivers]
    assert [item.model_name for item in actual_donors] == [item.model_name for item in expected_donors]
    assert [item.model_name for item in actual_donors] == ["idle", "waste", "surplus"]

    expected_eligible, expected_filtered = legacy_filter_donors_by_eta([item for item in expected_cls if item.role.value == "donor"])
    actual_eligible, actual_filtered = filter_donors_by_eta([item for item in actual_cls if item.role.value == "donor"])
    assert [item.model_name for item in actual_eligible] == [item.model_name for item in expected_eligible]
    assert [item.model_name for item in actual_filtered] == [item.model_name for item in expected_filtered]


def test_build_comparison_log_matches_legacy() -> None:
    legacy_cls = legacy_classify_model(
        model_name="donor",
        trs=1500.0,
        Z_m=1.5,
        eta_m=350.0,
        theta_m=1000.0,
        tau=LegacyTauThresholds.from_control(),
        eta_crit=200.0,
        eta_low=300.0,
    )
    actual_cls = classify_model(
        model_name="donor",
        trs=1500.0,
        Z_m=1.5,
        eta_m=350.0,
        theta_m=1000.0,
        tau=TauThresholds.from_control(),
        eta_crit=200.0,
        eta_low=300.0,
    )

    assert build_comparison_log(model_name="donor", legacy_type="SURPLUS", paper_cls=actual_cls) == legacy_build_comparison_log(
        model_name="donor",
        legacy_type="SURPLUS",
        paper_cls=legacy_cls,
    )


def test_idle_rps_guard_neutralizes_low_rate_nonzero_queue_signal() -> None:
    classification = classify_model(
        model_name="queue-arm",
        trs=0.0,
        Z_m=0.1,
        eta_m=None,
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
            "eta_m": 250.0,
            "trs": 700.0,
            "theta_m": 1000.0,
            "request_rate_rps": 0.8,
        },
        "healthy": {
            "Y_m": 100.0,
            "Q": 1.0,
            "z_m": 1.1,
            "eta_m": 320.0,
            "trs": 1100.0,
            "theta_m": 1000.0,
            "request_rate_rps": 1.2,
        },
        "high": {
            "Y_m": 100.0,
            "Q": 1.0,
            "z_m": 1.5,
            "eta_m": 350.0,
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
            "eta_m": 0.0,
            "trs": 0.0,
            "theta_m": 1000.0,
            "request_rate_rps": 0.0,
        }
    }

    classification = classify_all_models(contexts, signal_idle_rps_eps=0.05)[0]
    assert classification.state.value == "idle"
    assert classification.donor_tier == "idle"

def test_disable_eta_gate_keeps_all_donors_and_uses_natural_model_order() -> None:
    contexts = {
        "model-10": {
            "Y_m": 1.0,
            "Q": 1.0,
            "z_m": 1.8,
            "eta_m": 100.0,
            "trs": 1800.0,
            "theta_m": 1000.0,
        },
        "model-2": {
            "Y_m": 1.0,
            "Q": 1.0,
            "z_m": 1.4,
            "eta_m": 250.0,
            "trs": 1400.0,
            "theta_m": 1000.0,
        },
        "model-1": {
            "Y_m": 1.0,
            "Q": 1.0,
            "z_m": 1.6,
            "eta_m": 500.0,
            "trs": 1600.0,
            "theta_m": 1000.0,
        },
    }
    controls = {
        model: {"receiver_thrashing_eff": 200.0, "donor_waste_eff": 300.0}
        for model in contexts
    }
    classifications = classify_all_models(contexts, model_control_configs=controls)
    donors = [item for item in classifications if item.role.value == "donor"]

    eligible, filtered = filter_donors_by_eta(donors, disabled=True)
    assert {item.model_name for item in eligible} == set(contexts)
    assert filtered == []

    _receivers, ordered = split_receivers_donors(
        classifications, disable_eta_gate=True
    )
    assert [item.model_name for item in ordered] == ["model-1", "model-2", "model-10"]
    assert sorted(
        donors,
        key=lambda item: donor_mock_cost_key(item, disable_eta_gate=True),
    ) == ordered
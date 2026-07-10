from __future__ import annotations

import pytest
import yaml

from tre_ui.params import ParamValidationError, apply_and_validate, build_view

_REGISTRY = """
cluster:
  nodes:
    - name: node-a
      gpus: 4
      two_gpu_slots: [[0, 1], [2, 3]]
models:
  - name: m1
    weights_path: /w/m1
    tp_size: 1
    min_replicas: 1
    max_replicas: 4
    vllm_image: img:1
    slo:
      ttft_p95_ms: 500.0
      tpot_p95_ms: 75.0
      e2e_p95_ms: 12000.0
    alt_thresholds:
      queue_len: {theta: 6.5, direction: lower_is_healthier}
    trs:
      w_p: 0.08
      w_d: 1.0
      lambda_wait: 1.875
      qmin: 1.0
      ema_alpha: 0.25
      theta_m: 738.0
      tau_crit: 0.75
      tau_low: 1.0
      tau_high: 1.63
      qsat: 4.0
      epsat: 0.1
      hsat: 4
      ema_tau_ms: 20000
"""


def test_build_view_marks_editable_and_locked() -> None:
    view = build_view(_REGISTRY)["m1"]
    assert view["editable"]["trs.theta_m"]["value"] == 738.0
    assert view["editable"]["trs.theta_m"]["min_exclusive"] is True
    assert view["editable"]["slo.ttft_p95_ms"]["value"] == 500.0
    assert view["editable"]["alt_thresholds.queue_len.theta"]["value"] == 6.5
    assert view["editable"]["alt_thresholds.queue_len.direction"]["value"] == "lower_is_healthier"
    assert view["locked"]["tp_size"]["value"] == 1
    assert "frozen with W" in view["locked"]["trs.ema_tau_ms"]["reason"]


def test_apply_valid_edit_roundtrips_through_loader() -> None:
    new_yaml = apply_and_validate(
        _REGISTRY,
        {
            "m1": {
                "trs": {"theta_m": 800.0, "tau_high": 1.7},
                "alt_thresholds": {
                    "queue_len": {
                        "theta": 7.25,
                        "direction": "lower_is_healthier",
                    }
                },
                "max_replicas": 3,
            }
        },
    )
    m = yaml.safe_load(new_yaml)["models"][0]
    assert m["trs"]["theta_m"] == 800.0
    assert m["trs"]["tau_high"] == 1.7
    assert m["max_replicas"] == 3
    assert m["alt_thresholds"]["queue_len"] == {
        "theta": 7.25,
        "direction": "lower_is_healthier",
    }
    # untouched fields preserved
    assert m["trs"]["ema_tau_ms"] == 20000
    assert m["weights_path"] == "/w/m1"


def test_alt_threshold_rejects_wrong_direction() -> None:
    with pytest.raises(ParamValidationError) as exc_info:
        apply_and_validate(
            _REGISTRY,
            {
                "m1": {
                    "alt_thresholds": {
                        "queue_len": {
                            "theta": 7.0,
                            "direction": "higher_is_healthier",
                        }
                    }
                }
            },
        )
    assert exc_info.value.errors[0]["error"] == "invalid_direction"


def test_out_of_bounds_rejected() -> None:
    with pytest.raises(ParamValidationError) as ei:
        apply_and_validate(_REGISTRY, {"m1": {"trs": {"tau_crit": 5.0}}})
    assert ei.value.errors[0]["error"] == "out_of_bounds"


def test_non_finite_threshold_rejected() -> None:
    with pytest.raises(ParamValidationError) as exc_info:
        apply_and_validate(
            _REGISTRY,
            {"m1": {"alt_thresholds": {"queue_len": {"theta": "nan"}}}},
        )
    assert exc_info.value.errors[0]["error"] == "not_finite"


def test_locked_field_rejected() -> None:
    with pytest.raises(ParamValidationError) as ei:
        apply_and_validate(_REGISTRY, {"m1": {"tp_size": 2, "trs": {"ema_tau_ms": 999}}})
    kinds = {(e["field"], e["error"]) for e in ei.value.errors}
    assert ("tp_size", "locked") in kinds
    assert ("trs.ema_tau_ms", "locked") in kinds


def test_unknown_field_and_model_rejected() -> None:
    with pytest.raises(ParamValidationError) as ei:
        apply_and_validate(_REGISTRY, {"m1": {"trs": {"bogus": 1}}})
    assert ei.value.errors[0]["error"] == "unknown_field"
    with pytest.raises(ParamValidationError) as ei2:
        apply_and_validate(_REGISTRY, {"nope": {"trs": {"theta_m": 1}}})
    assert ei2.value.errors[0]["error"] == "unknown_model"


def test_cross_field_constraints() -> None:
    # tau_low must stay < tau_high
    with pytest.raises(ParamValidationError) as ei:
        apply_and_validate(_REGISTRY, {"m1": {"trs": {"tau_low": 2.0}}})  # tau_high is 1.63
    assert any(e["error"] == "constraint" for e in ei.value.errors)
    # min_replicas must be <= max_replicas
    with pytest.raises(ParamValidationError):
        apply_and_validate(_REGISTRY, {"m1": {"min_replicas": 4, "max_replicas": 1}})

"""Each alternative signal is classified on its own fitted bands (plan 6.9 item 3).

classify.py used to apply the TSS tau_crit/tau_high to every signal source. The registry
now carries delta_crit/delta_high per alt_thresholds entry; a missing pair is an error
(the controller refuses such an alt arm at start), never a default band.
"""
from __future__ import annotations

import pytest

from tre_common.registry import _parse_registry
from tre_controller.planning.classify import (
    ModelState,
    classify_all_models,
    model_control_configs_from_registry,
)


def _registry(alt: dict) -> object:
    return _parse_registry({
        "cluster": {"nodes": []},
        "models": [{
            "name": "m", "weights_path": "/w", "tp_size": 1, "min_replicas": 1,
            "max_replicas": 4, "vllm_image": "img",
            "slo": {"ttft_p95_ms": 500.0, "tpot_p95_ms": 75.0, "e2e_p95_ms": 1.0},
            "trs": {"w_p": 0.02, "w_d": 1.0, "lambda_wait": 3.0, "qmin": 1.0,
                    "ema_alpha": 0.0, "theta_m": 40.0, "tau_crit": 0.9, "tau_low": 1.0,
                    "tau_high": 1.1, "qsat": 4.0, "epsat": 0.05, "hsat": 1},
            "alt_thresholds": alt,
        }],
    })


def test_an_alt_signal_uses_its_own_fitted_bands_not_the_tss_ones() -> None:
    reg = _registry({"queue_len": {"theta": 20.0, "direction": "lower_is_healthier",
                                   "delta_crit": 0.35, "delta_high": 0.6}})
    assert reg.validate() == []
    tss = model_control_configs_from_registry(reg)["m"]
    alt = model_control_configs_from_registry(reg, "queue_len")["m"]
    assert tss == {"delta_crit": pytest.approx(0.1), "delta_high": pytest.approx(0.1)}
    assert alt == {"delta_crit": 0.35, "delta_high": 0.6}
    # z = 0.7 is CRITICAL on the TSS band (tau_crit 0.9) but only LOW on queue_len's (0.65).
    ctx = {"m": {"z_m": 0.7, "trs": 1.0, "signal_source": "queue_len", "Y_m": 1.0, "Q": 1.0}}
    [on_alt] = classify_all_models(ctx, model_control_configs=alt and {"m": alt})
    [on_tss] = classify_all_models(ctx, model_control_configs={"m": tss})
    assert on_alt.state == ModelState.LOW
    assert on_tss.state == ModelState.CRITICAL


def test_missing_bands_are_an_error_not_a_default_band() -> None:
    reg = _registry({"decode_tps": {"theta": 900.0, "direction": "lower_is_healthier"}})
    assert reg.validate() == []
    with pytest.raises(ValueError, match="no fitted delta_crit/delta_high"):
        model_control_configs_from_registry(reg, "decode_tps")
    # an unfitted source (no alt_thresholds entry) keeps the TSS bands
    assert model_control_configs_from_registry(reg, "latency_p95") == model_control_configs_from_registry(reg)


def test_band_validation() -> None:
    bad = _registry({"queue_len": {"theta": 20.0, "direction": "lower_is_healthier",
                                   "delta_crit": 1.2, "delta_high": -0.1}})
    errors = bad.validate()
    assert any("delta_crit" in e for e in errors) and any("delta_high" in e for e in errors)

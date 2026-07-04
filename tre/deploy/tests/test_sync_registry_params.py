from __future__ import annotations

from sync_registry_params import sync_registry_params


def test_sync_registry_params_merges_old_profiles_and_seed_values() -> None:
    registry = {
        "cluster": {"nodes": []},
        "models": [
            {
                "name": "m1",
                "weights_path": "/m1",
                "tp_size": 1,
                "min_replicas": 0,
                "max_replicas": 2,
                "vllm_image": "image",
                "slo": {"ttft_p95_ms": 1200, "tpot_p95_ms": 100, "e2e_p95_ms": 10000},
                "trs": {
                    "w_p": 0.04,
                    "w_d": 1.0,
                    "lambda_wait": 2.625,
                    "qmin": 1.0,
                    "ema_alpha": 0.5,
                    "theta_m": 0.0,
                    "tau_crit": 0.8,
                    "tau_low": 1.0,
                    "tau_high": 1.25,
                    "qsat": 4.0,
                    "epsat": 0.05,
                    "hsat": 3,
                },
            }
        ],
    }
    profiles = {
        "defaults": {
            "weights": {"w_p": 0.04, "w_d": 1.0, "lambda_wait": 3.5},
            "control": {
                "trs_ema_alpha": 0.393,
                "qmin": 1.0,
                "qsat": 4.0,
                "epsat": 0.05,
                "Hsat": 3,
                "delta_crit": 0.2,
                "delta_high": 0.25,
            },
            "latency_slo_ms": {"ttft_p95": 500.0, "tpot_p95": 75.0, "e2e_p95": 10000.0},
        },
        "models": {
            "m1": {
                "weights": {"w_p": 0.08, "lambda_wait": 1.875},
                "control": {"trs_ema_alpha": 0.2485, "delta_crit": 0.2515, "delta_high": 0.6296},
                "latency_slo_ms": {"ttft_p95": 500.0, "tpot_p95": 75.0, "e2e_p95": 12000.0},
            }
        },
    }
    seed = {"m1": {"theta_m": 738.67}}

    updated, changes = sync_registry_params(registry, profiles, seed)

    model = updated["models"][0]
    assert model["slo"] == {"ttft_p95_ms": 500.0, "tpot_p95_ms": 75.0, "e2e_p95_ms": 12000.0}
    assert model["trs"]["w_p"] == 0.08
    assert model["trs"]["w_d"] == 1.0
    assert model["trs"]["lambda_wait"] == 1.875
    assert model["trs"]["ema_alpha"] == 0.2485
    assert model["trs"]["theta_m"] == 738.67
    assert model["trs"]["tau_crit"] == 0.7485
    assert model["trs"]["tau_low"] == 1.0
    assert model["trs"]["tau_high"] == 1.6296
    assert "m1.slo.ttft_p95_ms: 1200 -> 500.0" in changes
    assert "m1.trs.theta_m: 0.0 -> 738.67" in changes


def test_sync_registry_params_leaves_models_without_legacy_sources_unchanged() -> None:
    registry = {
        "models": [
            {
                "name": "unknown",
                "slo": {"ttft_p95_ms": 1200, "tpot_p95_ms": 100, "e2e_p95_ms": 10000},
                "trs": {"theta_m": 0.0},
            }
        ]
    }

    updated, changes = sync_registry_params(registry, {"defaults": {}, "models": {}}, {})

    assert updated == registry
    assert changes == []


def test_sync_registry_params_breaks_yaml_aliases_between_model_trs_buckets() -> None:
    shared_trs = {"theta_m": 0.0, "tau_crit": 0.8, "tau_low": 1.0, "tau_high": 1.25}
    registry = {
        "models": [
            {"name": "m1", "slo": {}, "trs": shared_trs},
            {"name": "m2", "slo": {}, "trs": shared_trs},
        ]
    }
    profiles = {"defaults": {}, "models": {}}
    seed = {"m1": {"theta_m": 111.0}, "m2": {"theta_m": 222.0}}

    updated, changes = sync_registry_params(registry, profiles, seed)

    assert updated["models"][0]["trs"]["theta_m"] == 111.0
    assert updated["models"][1]["trs"]["theta_m"] == 222.0
    assert updated["models"][0]["trs"] is not updated["models"][1]["trs"]
    assert changes == ["m1.trs.theta_m: 0.0 -> 111.0", "m2.trs.theta_m: 0.0 -> 222.0"]


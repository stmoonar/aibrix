from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROFILES_PATH = Path("/root/aibrix-main/python/tre/configs/model_slo_profiles.json")
DEFAULT_SEED_PATH = Path("/root/aibrix-main/python/tre/configs/seed_calibration.json")


def sync_registry_params(
    registry: dict[str, Any],
    profiles: dict[str, Any],
    seed: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    updated = copy.deepcopy(registry)
    changes: list[str] = []
    defaults = profiles.get("defaults") or {}
    model_profiles = profiles.get("models") or {}

    for model in updated.get("models", []):
        name = str(model.get("name", ""))
        model_profile = model_profiles.get(name)
        model_seed = seed.get(name)
        if not model_profile and not model_seed:
            continue

        merged_weights = _merged(defaults.get("weights"), (model_profile or {}).get("weights"))
        merged_control = _merged(defaults.get("control"), (model_profile or {}).get("control"))
        merged_slo = _merged(defaults.get("latency_slo_ms"), (model_profile or {}).get("latency_slo_ms"))

        slo = dict(model.get("slo") or {})
        model["slo"] = slo
        _set_change(changes, name, slo, "ttft_p95_ms", merged_slo.get("ttft_p95"))
        _set_change(changes, name, slo, "tpot_p95_ms", merged_slo.get("tpot_p95"))
        _set_change(changes, name, slo, "e2e_p95_ms", merged_slo.get("e2e_p95"))

        trs = dict(model.get("trs") or {})
        model["trs"] = trs
        _set_change(changes, name, trs, "w_p", merged_weights.get("w_p"), prefix="trs")
        _set_change(changes, name, trs, "w_d", merged_weights.get("w_d"), prefix="trs")
        _set_change(changes, name, trs, "lambda_wait", merged_weights.get("lambda_wait"), prefix="trs")
        _set_change(changes, name, trs, "qmin", merged_control.get("qmin"), prefix="trs")
        _set_change(changes, name, trs, "ema_alpha", merged_control.get("trs_ema_alpha"), prefix="trs")
        _set_change(changes, name, trs, "qsat", merged_control.get("qsat"), prefix="trs")
        _set_change(changes, name, trs, "epsat", merged_control.get("epsat"), prefix="trs")
        _set_change(changes, name, trs, "hsat", merged_control.get("Hsat"), prefix="trs")
        if merged_control.get("delta_crit") is not None:
            _set_change(changes, name, trs, "tau_crit", round(1.0 - float(merged_control["delta_crit"]), 6), prefix="trs")
        _set_change(changes, name, trs, "tau_low", 1.0, prefix="trs")
        if merged_control.get("delta_high") is not None:
            _set_change(changes, name, trs, "tau_high", round(1.0 + float(merged_control["delta_high"]), 6), prefix="trs")
        if model_seed and model_seed.get("theta_m") is not None:
            _set_change(changes, name, trs, "theta_m", model_seed.get("theta_m"), prefix="trs")

    return updated, changes


def _merged(default: object, override: object) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(default, dict):
        merged.update(default)
    if isinstance(override, dict):
        merged.update(override)
    return merged


def _set_change(
    changes: list[str],
    model_name: str,
    bucket: dict[str, Any],
    key: str,
    value: object,
    *,
    prefix: str = "slo",
) -> None:
    if value is None:
        return
    old = bucket.get(key)
    if old == value:
        return
    bucket[key] = value
    changes.append(f"{model_name}.{prefix}.{key}: {old} -> {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="registry.yaml")
    parser.add_argument("--profiles", default=str(DEFAULT_PROFILES_PATH))
    parser.add_argument("--seed", default=str(DEFAULT_SEED_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    seed = json.loads(Path(args.seed).read_text(encoding="utf-8"))
    updated, changes = sync_registry_params(registry, profiles, seed)

    for change in changes:
        print(change)
    if not args.dry_run:
        registry_path.write_text(yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()

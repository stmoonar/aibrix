from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tre_common.registry import Registry, load_registry
from sync_registry_params import DEFAULT_PROFILES_PATH


def registry_warnings(registry: Registry, *, profiles: dict[str, Any] | None = None) -> list[str]:
    warnings: list[str] = []
    profile_models = (profiles or {}).get("models") or {}
    for model in registry.models():
        if model.trs.theta_m == 0.0:
            warnings.append(f"WARNING {model.name}.trs.theta_m is 0.0")
        profile = profile_models.get(model.name) or {}
        slo = profile.get("latency_slo_ms") or {}
        _compare_slo(warnings, model.name, "ttft_p95_ms", model.slo.ttft_p95_ms, slo.get("ttft_p95"))
        _compare_slo(warnings, model.name, "tpot_p95_ms", model.slo.tpot_p95_ms, slo.get("tpot_p95"))
        _compare_slo(warnings, model.name, "e2e_p95_ms", model.slo.e2e_p95_ms, slo.get("e2e_p95"))
    return warnings


def _compare_slo(warnings: list[str], model: str, field: str, current: float, expected: object) -> None:
    if expected is None:
        return
    expected_float = float(expected)
    if current != expected_float:
        warnings.append(f"WARNING {model}.slo.{field} differs from profile: {current} != {expected_float}")


def _load_profiles(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="deploy/registry.yaml")
    parser.add_argument("--profiles", default=str(DEFAULT_PROFILES_PATH))
    args = parser.parse_args()

    registry = load_registry(args.registry)
    errors = registry.validate()
    if errors:
        raise SystemExit(errors)
    for warning in registry_warnings(registry, profiles=_load_profiles(Path(args.profiles))):
        print(warning)
    print("tre smoke ok")


if __name__ == "__main__":
    main()

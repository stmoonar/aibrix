"""Merge a release registry with the LIVE registry (ConfigMap tre-v2-registry).

params_set.py (deploy/scripts/release) edits only the per-model tunables: trs.*, slo.*,
alt_thresholds.*, min_replicas / max_awake_replicas. Everything else
(max_replicas = the GPU layout, images, engine args, vllm / gateway / reissue / service_manager /
placement sections) changes only
with a release and has no params_set.py path. This script takes the release registry and
lays the live tunables over it, key by key, so a release never resets calibrated values
(e.g. theta_m) and never drops a key the release added.

    kubectl -n tre-v2 get cm tre-v2-registry -o jsonpath='{.data.registry\\.yaml}' > live.yaml
    python3 deploy/scripts/merge_live_registry.py --live live.yaml \\
        --release deploy/registry.yaml --out merged.yaml
    # review the printed report and `diff live.yaml merged.yaml`, then write merged.yaml
    # into the ConfigMap (see deploy/RELEASE-*.md).

The output is validated with tre_common.registry (the service-manager / controller
parser). Exit 1 on validation errors; the report lists every live tunable that differs
from the release file and warns when max_replicas (the GPU layout the release manifests
were rendered from) differs; the release value is used.
"""
from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from tre_common.tss import NUMERATOR_COMPLETION_HISTOGRAMS

#: Per-model keys params_set.py may edit (deploy/scripts/registry_params.py).
TUNABLE_SECTIONS = ("trs", "slo")
TUNABLE_KEYS = ("min_replicas", "max_awake_replicas")

#: The theta basis stamps (tre_common.tss.theta_basis_errors) are part of theta: a theta is a
#: window token total of one numerator source. A live theta_m without stamps predates them
#: (2026-10-28) and was fitted on the completion histograms over 30 s windows. Its stamps are
#: that truthful basis, never the release file's, so a controller computing another basis
#: refuses to start until a theta fitted on its basis is published.
THETA_BASIS_STAMPS = ("numerator", "theta_window_ms")
LEGACY_THETA_BASIS = {"numerator": NUMERATOR_COMPLETION_HISTOGRAMS, "theta_window_ms": 30000}


def merge(live: dict[str, Any], release: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    merged = copy.deepcopy(release)
    report: list[str] = []
    live_models = {m.get("name"): m for m in live.get("models") or []}
    for model in merged.get("models") or []:
        name = model.get("name")
        src = live_models.get(name)
        if src is None:
            report.append(f"{name}: not in the live registry (release values used)")
            continue
        for section in TUNABLE_SECTIONS:
            for key, value in (src.get(section) or {}).items():
                target = model.setdefault(section, {})
                if target.get(key) != value:
                    report.append(f"{name}: {section}.{key} live {value!r} kept (release {target.get(key)!r})")
                target[key] = value
        live_trs = src.get("trs") or {}
        if "theta_m" in live_trs:
            target = model.setdefault("trs", {})
            for key in THETA_BASIS_STAMPS:
                if key not in live_trs:
                    value = LEGACY_THETA_BASIS[key]
                    if target.get(key) != value:
                        report.append(
                            f"WARN {name}: trs.{key} {value!r} (live theta_m has no stamp: fitted before the "
                            f"stamps, on that basis; release {target.get(key)!r} not applied to it)"
                        )
                    target[key] = value
        for signal, fields in (src.get("alt_thresholds") or {}).items():
            target = model.setdefault("alt_thresholds", {}).setdefault(signal, {})
            for key, value in (fields or {}).items():
                if target.get(key) != value:
                    report.append(
                        f"{name}: alt_thresholds.{signal}.{key} live {value!r} kept (release {target.get(key)!r})"
                    )
                target[key] = value
        for key in TUNABLE_KEYS:
            if key in src:
                if model.get(key) != src[key]:
                    report.append(f"{name}: {key} live {src[key]!r} kept (release {model.get(key)!r})")
                model[key] = src[key]
        if "max_replicas" in src and src["max_replicas"] != model.get("max_replicas"):
            report.append(
                f"WARN {name}: max_replicas live {src['max_replicas']!r} replaced by the release "
                f"{model.get('max_replicas')!r} (the GPU layout the release manifests were rendered with)"
            )
    for name in sorted(set(live_models) - {m.get("name") for m in merged.get("models") or []}):
        report.append(f"WARN {name}: in the live registry only (dropped by the release)")
    return merged, report


def _validate(merged: dict[str, Any]) -> list[str]:
    from tre_common.registry import load_registry

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(merged, handle, sort_keys=False)
        path = handle.name
    try:
        return load_registry(path).validate()
    finally:
        Path(path).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", required=True, help="live registry.yaml (from the ConfigMap)")
    parser.add_argument("--release", required=True, help="release registry.yaml (deploy/registry.yaml)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    live = yaml.safe_load(Path(args.live).read_text(encoding="utf-8")) or {}
    release = yaml.safe_load(Path(args.release).read_text(encoding="utf-8")) or {}
    merged, report = merge(live, release)
    errors = _validate(merged)
    for line in report or ["no live tunable differs from the release file"]:
        print(line)
    if errors:
        print("registry validation failed:\n" + "\n".join(errors), file=sys.stderr)
        return 1
    Path(args.out).write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

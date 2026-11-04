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

from tre_common import slo as slo_def
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


#: One alternative-signal threshold is ONE fit (review M1, 2026-10-28): theta, its bands
#: and the basis stamp (definition, window, EMA tau) belong together. A key-by-key merge
#: combined a stale live theta (07-10 fit, no bands, no stamp) with the release file's bands
#: and basis, and that mix passed the controller's start check.
ALT_THRESHOLD_GROUP = ("theta", "direction", "delta_crit", "delta_high", "basis")


def _merge_alt_threshold(name: str, signal: str, live: dict, model: dict, report: list[str]) -> None:
    """``alt_thresholds.<signal>``: the whole live fit (when live has a theta) or the whole
    release fit, never a mix. A live theta without a basis stays without one, so an alt arm
    on it refuses to start until a stamped fit is published."""
    target = model.setdefault("alt_thresholds", {}).setdefault(signal, {})
    if "theta" not in live:
        if any(key in live for key in ALT_THRESHOLD_GROUP):
            report.append(
                f"WARN {name}: alt_thresholds.{signal} live entry has no theta "
                f"({sorted(k for k in live if k in ALT_THRESHOLD_GROUP)}): the release fit is used whole"
            )
        return
    release_fit = {key: target[key] for key in ALT_THRESHOLD_GROUP if key in target}
    live_fit = {key: copy.deepcopy(live[key]) for key in ALT_THRESHOLD_GROUP if key in live}
    for key in ALT_THRESHOLD_GROUP:
        target.pop(key, None)
    target.update(live_fit)
    for key, value in live.items():  # any non-fit key: live wins, as for every tunable
        if key not in ALT_THRESHOLD_GROUP:
            target[key] = copy.deepcopy(value)
    if live_fit != release_fit:
        report.append(
            f"{name}: alt_thresholds.{signal} live fit {live_fit!r} kept whole (release {release_fit!r})"
        )
    if "basis" not in live_fit:
        report.append(
            f"WARN {name}: alt_thresholds.{signal} live theta has no basis stamp: kept without one "
            f"(an alt arm on {signal} refuses to start until a stamped fit is published)"
        )


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
                if section == "slo" and key == slo_def.LEGACY_MODEL_KEY:
                    continue  # the retired per-model switch: see _merge_slo_mode
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
            _merge_alt_threshold(name, signal, fields or {}, model, report)
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
    _merge_slo_mode(live, merged, report)
    return merged, report


def _merge_slo_mode(live: dict[str, Any], merged: dict[str, Any], report: list[str]) -> None:
    """The SLO switch is a tunable (params_set.py --set slo_mode=...): a live ``slo_mode``
    is kept. A live registry from before it (2026-11-04) has the two retired keys instead;
    they are dropped (one switch, tre_common.slo) and the release's ``slo_mode`` applies,
    reported with what the live keys meant."""
    key = slo_def.SLO_MODE_KEY
    if key in live:
        if merged.get(key) != live[key]:
            report.append(f"registry: {key} live {live[key]!r} kept (release {merged.get(key)!r})")
        merged[key] = live[key]
    else:
        try:
            legacy = slo_def.resolve_slo_mode(live)
        except ValueError as exc:
            legacy = None
            report.append(f"WARN registry: live SLO keys unresolvable ({exc})")
        if legacy is not None and legacy.source != "default":
            safescale = (live.get("safescale") or {}).get(slo_def.LEGACY_SAFESCALE_KEY)
            report.append(
                f"WARN registry: live has no {key} (legacy {legacy.source} -> {legacy.mode}; safescale.slo_mode "
                f"{safescale!r}); the release {key} {merged.get(key)!r} applies to every consumer and the retired "
                "keys are dropped"
            )
    safescale = merged.get("safescale")
    if isinstance(safescale, dict):
        safescale.pop(slo_def.LEGACY_SAFESCALE_KEY, None)
    for model in merged.get("models") or []:
        (model.get("slo") or {}).pop(slo_def.LEGACY_MODEL_KEY, None)


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

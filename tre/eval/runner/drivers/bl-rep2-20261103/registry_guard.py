#!/usr/bin/env python3
# [repo] Repo copy: only the fallback path to the tre/ tree changed (TRE_DIR still wins).
"""READ-ONLY guard: is the baseline-relevant part of the live registry the one of the reference file?

Why: the three baseline arms (APA, TokenScale, PreServe) read from the registry only the per-model bounds
(min_replicas, max_awake_replicas / max_replicas, tp_size), the vLLM args (--max-num-seqs), slo.ttft_p95_ms /
slo.tpot_p95_ms, and the service_manager / gateway / cluster / placement sections (SM + gateway behave the same for
every arm). They read NO trs.* and NO alt_thresholds.* value. So a calibration refit that republishes TRE parameters
(trs, alt_thresholds, the c/b and slowdown parts of slo, safescale, scaling) must NOT stop this campaign, while a change
anywhere else must.

  registry_guard.py REFERENCE.yaml            exit 0 = baseline part identical (controller-only differences are printed
                                              as INFO), exit 1 = a baseline-relevant key differs, exit 2 = cannot read.
  registry_guard.py REFERENCE.yaml --id       print the content id of the live registry (arm_registry.registry_id).

The live registry is read with `kubectl get cm tre-v2-registry` (read only).
"""
import copy
import hashlib
import json
import os
import subprocess
import sys

import yaml

CONTROLLER_ONLY_TOP = ("safescale", "scaling")                 # controller config (SafeScale probe, rescue / breakpoint)
SLO_BASELINE_KEYS = ("ttft_p95_ms", "tpot_p95_ms", "e2e_p95_ms")   # the other slo keys are the TRE label fit (c / b / k)


def live_text() -> str:
    ns = os.environ.get("TRE_NS", "tre-v2")
    cm = os.environ.get("REGISTRY_CM", "tre-v2-registry")
    r = subprocess.run(["kubectl", "-n", ns, "get", "cm", cm, "-o", r"jsonpath={.data.registry\.yaml}"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"cannot read the live registry: {r.stderr.strip()[:200]}")
    return r.stdout


def content_id(text: str) -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        tre = os.environ.get("TRE_DIR") or os.path.join(here, "..", "..", "..", "..")
        sys.path.insert(0, os.path.join(tre, "eval", "runner"))
        import arm_registry as AR  # noqa: PLC0415
        return AR.registry_id(text)
    except Exception:  # noqa: BLE001  fall back to a plain hash of the parsed content
        data = yaml.safe_load(text) or {}
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:16] + "(plain)"


def flat(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flat(v, f"{p}.{k}" if p else str(k)))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(flat(v, f"{p}[{i}]"))
    else:
        out[p] = o
    return out


def split(reg: dict):
    """(baseline-relevant part, controller-only part) as flat dicts."""
    base = copy.deepcopy(reg)
    ctl = {}
    for k in CONTROLLER_ONLY_TOP:
        if k in base:
            ctl[k] = base.pop(k)
    for i, m in enumerate(base.get("models") or []):
        name = m.get("name", i)
        for k in ("trs", "alt_thresholds"):
            if k in m:
                ctl[f"models[{name}].{k}"] = m.pop(k)
        slo = m.get("slo")
        if isinstance(slo, dict):
            ctl[f"models[{name}].slo_extra"] = {k: slo.pop(k) for k in list(slo) if k not in SLO_BASELINE_KEYS}
    return flat(base), flat(ctl)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    try:
        live = live_text()
    except Exception as exc:  # noqa: BLE001
        print(f"registry_guard: {exc}")
        return 2
    if "--id" in sys.argv[2:]:
        print(content_id(live))
        return 0
    try:
        ref = open(sys.argv[1]).read()
    except OSError as exc:
        print(f"registry_guard: cannot read the reference {sys.argv[1]}: {exc}")
        return 2
    lb, lc = split(yaml.safe_load(live) or {})
    rb, rc = split(yaml.safe_load(ref) or {})
    diff = sorted(k for k in set(lb) | set(rb) if lb.get(k, "<absent>") != rb.get(k, "<absent>"))
    cdiff = sorted(k for k in set(lc) | set(rc) if lc.get(k, "<absent>") != rc.get(k, "<absent>"))
    print(f"live registry id {content_id(live)}; reference id {content_id(ref)}")
    print(f"baseline-relevant keys: {len(lb)} live / {len(rb)} reference; differing: {len(diff)}")
    for k in diff[:25]:
        print(f"  DIFF {k}: live={lb.get(k, '<absent>')!r} reference={rb.get(k, '<absent>')!r}")
    print(f"INFO controller-only keys differing from the reference (trs / alt_thresholds / slo c,b / safescale / scaling): "
          f"{len(cdiff)}{' e.g. ' + ', '.join(cdiff[:4]) if cdiff else ''}")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())

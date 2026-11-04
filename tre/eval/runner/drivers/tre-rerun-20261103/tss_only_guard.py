#!/usr/bin/env python3
"""READ-ONLY guard for the TRE rerun: does the live registry differ from the reference ONLY in the TSS parameters?

The rerun must change nothing but the recalibrated TSS parameters (theta_m, tau_crit, tau_high, possibly lambda_wait /
w_p) against the registry of the baselines rep 2 (reference 0ab5352747bdd70e = tune F + the 11-02 alt_thresholds).
Everything else (trs basis: numerator / theta_window_ms / ema, alt_thresholds, slo, safescale, scaling, replicas,
engine args, ...) must be identical, so that TRE (new parameters) stays comparable with both baseline samples.

  tss_only_guard.py REFERENCE.yaml [--live-file F]
      exit 0 = only models[*].trs.{theta_m,tau_crit,tau_high,lambda_wait,w_p} differ and at least one does;
      exit 1 = another key differs, or nothing differs (new parameters not published; ALLOW_SAME=1 accepts that);
      exit 2 = cannot read.
  TSS_GUARD_EXTRA=<regex>   also accept keys matching it (e.g. 'models\\[[^]]+\\]\\.slo\\.ttft_idle_(c_ms|b_ms_per_token)'
                            if c / b are refit too); say so in the spec header.

The live registry is read with `kubectl get cm tre-v2-registry` (read only) unless --live-file is given.
"""
import os
import re
import subprocess
import sys

import yaml

TSS_KEYS = ("theta_m", "tau_crit", "tau_high", "lambda_wait", "w_p")
ALLOWED = re.compile(r"models\[[^]]+\]\.trs\.(" + "|".join(TSS_KEYS) + r")")


def live_text() -> str:
    ns = os.environ.get("TRE_NS", "tre-v2")
    cm = os.environ.get("REGISTRY_CM", "tre-v2-registry")
    r = subprocess.run(["kubectl", "-n", ns, "get", "cm", cm, "-o", r"jsonpath={.data.registry\.yaml}"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"cannot read the live registry: {r.stderr.strip()[:200]}")
    return r.stdout


def flat(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flat(v, f"{p}.{k}" if p else str(k)))
    elif isinstance(o, list):
        for i, v in enumerate(o):   # models by name (robust to order), other lists by index
            out.update(flat(v, f"{p}[{v['name']}]" if isinstance(v, dict) and v.get("name") else f"{p}[{i}]"))
    else:
        out[p] = o
    return out


def same(a, b) -> bool:
    try:
        return float(a) == float(b)   # 20 == 20.0 (params_set re-renders numbers)
    except (TypeError, ValueError):
        return a == b


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        print(__doc__)
        return 2
    try:
        ref = yaml.safe_load(open(args[0]).read()) or {}
        if "--live-file" in args:
            live = yaml.safe_load(open(args[args.index("--live-file") + 1]).read()) or {}
        else:
            live = yaml.safe_load(live_text()) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"tss_only_guard: {exc}")
        return 2
    extra = re.compile(os.environ["TSS_GUARD_EXTRA"]) if os.environ.get("TSS_GUARD_EXTRA") else None
    fl, fr = flat(live), flat(ref)
    diff = sorted(k for k in set(fl) | set(fr) if not same(fl.get(k, "<absent>"), fr.get(k, "<absent>")))
    tss = [k for k in diff if ALLOWED.fullmatch(k) or (extra and extra.fullmatch(k))]
    other = [k for k in diff if k not in tss]
    print(f"keys: {len(fl)} live / {len(fr)} reference; differing: {len(diff)} ({len(tss)} allowed TSS keys, {len(other)} other)")
    models = sorted({m.get("name") for m in (live.get("models") or []) + (ref.get("models") or []) if isinstance(m, dict)})
    print(f"{'model':<12} " + " ".join(f"{k:>22}" for k in TSS_KEYS) + "   (reference -> live)")
    for m in models:
        cells = []
        for k in TSS_KEYS:
            key = f"models[{m}].trs.{k}"
            a, b = fr.get(key, "-"), fl.get(key, "-")
            cells.append(f"{a}" if same(a, b) else f"{a}->{b}")
        print(f"{m:<12} " + " ".join(f"{c:>22}" for c in cells))
    for k in other[:40]:
        print(f"  NOT-TSS DIFF {k}: reference={fr.get(k, '<absent>')!r} live={fl.get(k, '<absent>')!r}")
    if other:
        print(f"FAIL {len(other)} non-TSS key(s) differ from the reference (only {', '.join(TSS_KEYS)} may change)")
        return 1
    if not tss:
        if os.environ.get("ALLOW_SAME") == "1":
            print("WARN live registry = reference in every TSS key (ALLOW_SAME=1: accepted)")
            return 0
        print("FAIL live registry = reference: the recalibrated parameters are not published (ALLOW_SAME=1 to accept)")
        return 1
    print(f"OK only TSS parameters differ ({len(tss)} keys)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

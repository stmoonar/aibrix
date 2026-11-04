#!/usr/bin/env python3
"""Build the W=5 s registry from the LIVE ConfigMap (read-only: kubectl get) - no cluster writes.

  PYTHONPATH=$TRE/common:$TRE/deploy python3 make_win5_registry.py --out-dir DIR [--tau-ms 0] [--window-ms 5000]

Edits, every model, in one rendered file:
  trs.theta_m          x window/20000 (theta is a window token total; offline W5/W20 TSS level ratio 0.997-1.014)
  trs.theta_window_ms  -> window (the controller refuses to start unless == TRE_METRICS_WINDOW_MS)
  trs.ema_tau_ms       -> tau (LOCKED in params_set, "window-coupled; frozen with W": we change W, so it is set
                         here directly; tau <= 0 = raw passthrough, controller/signals/trs.py TRSComputer)
theta_m / theta_window_ms go through registry_params.apply_and_validate (bounds, cross-field, parser);
the whole result is re-validated with Registry.validate() (no new errors vs live).
Writes: live-registry.yaml (as read), registry-win5.yaml, diff.txt, meta.json (resourceVersion read,
params-hash, arm_registry id of both).
"""
import argparse, json, subprocess, sys, os
from pathlib import Path

import yaml

TRE = Path(os.environ.get("TRE_DIR", "/data/nfs_shared_data/xxy/aibrix/tre"))
sys.path[:0] = [str(TRE / "common"), str(TRE / "deploy"), str(TRE / "eval" / "runner")]
from scripts import registry_params as rp  # noqa: E402
import arm_registry as AR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window-ms", type=int, default=5000)
    ap.add_argument("--tau-ms", type=float, default=0.0)
    ap.add_argument("--registry-file", help="offline: use this file instead of the live ConfigMap")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    if a.registry_file:
        text = Path(a.registry_file).read_text(); rv = None
    else:
        cm = json.loads(subprocess.run(["kubectl", "-n", "tre-v2", "get", "cm", "tre-v2-registry", "-o", "json"],
                                       check=True, capture_output=True, text=True).stdout)
        text = cm["data"]["registry.yaml"]; rv = cm["metadata"]["resourceVersion"]
        if (cm["metadata"].get("annotations") or {}).get(AR.ANNOTATION):
            sys.exit("REFUSED: an arm registry perturbation is in effect (annotation %s)" % AR.ANNOTATION)
    data = yaml.safe_load(text)
    sets = []
    for m in data["models"]:
        t = m["trs"]
        if int(t["theta_window_ms"]) == a.window_ms:
            sys.exit("REFUSED: %s already stamped theta_window_ms=%s" % (m["name"], t["theta_window_ms"]))
        f = a.window_ms / float(t["theta_window_ms"])
        sets += ["%s.trs.theta_m=%r" % (m["name"], round(float(t["theta_m"]) * f, 6)),
                 "%s.trs.theta_window_ms=%d" % (m["name"], a.window_ms)]
    names = [m["name"] for m in data["models"]]
    try:
        mid = rp.apply_and_validate(text, rp.parse_sets(sets, names))
    except rp.ParamValidationError as exc:
        sys.exit("REJECTED: %s" % exc.errors)
    d2 = yaml.safe_load(mid)
    for m in d2["models"]:
        m["trs"]["ema_tau_ms"] = a.tau_ms
    new = rp.render(d2)
    errs = rp.new_validation_errors(text, new)
    if errs:
        sys.exit("REJECTED by Registry.validate(): %s" % errs)
    (out / "live-registry.yaml").write_text(text)
    (out / "registry-win5.yaml").write_text(new)
    diff = rp.unified_diff(text, new)
    (out / "diff.txt").write_text(diff)
    ch = rp.changed_fields(text, new)
    meta = dict(read_resource_version=rv, window_ms=a.window_ms, tau_ms=a.tau_ms, sets=sets,
                changes=[list(map(str, c)) for c in ch],
                params_hash_live=rp.params_hash(text), params_hash_new=rp.params_hash(new),
                registry_id_live=AR.registry_id(text), registry_id_new=AR.registry_id(new))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    for c in ch:
        print("%s.%s: %s -> %s" % c)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("sets", "changes")}, indent=1))


if __name__ == "__main__":
    main()

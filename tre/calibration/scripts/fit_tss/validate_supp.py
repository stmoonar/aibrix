#!/usr/bin/env python3
"""validate_supp.py - prereg-calib-supplement-20261103.md §4 acceptance metrics of ONE candidate parameter set
(theta, tau_crit, tau_high, lambda, w_p; r*, window, attribution) on ONE dataset.  CPU only, read-only.

Per model, on the kept windows (same filter as the fit: valid, not warm-up, not contaminated, TSS > 0, first window per
cell dropped), TSS at the candidate's lambda / w_p, Z = TSS / theta, requests attributed by the candidate's attribution
(first token by default), label = TTFT > TTFT_slo or TPOT > TPOT_slo (the registry's, supp_lib "SLO") or not served:
  1. Z in [0.9, 1.1): per-request violation rate, cell-bootstrap CI90 (--nboot 2000); PASS if the CI90 meets
     [0.5, 2] x r*  (prereg §4.1)
  2. per-request violation rate over the Z bands  < tau_crit | tau_crit-0.9 | 0.9-1.1 | 1.1-tau_high | >= tau_high
     non-increasing (point estimates, empty bands skipped; prereg §4.2); the bootstrap share of monotone resamples is
     reported, not judged
  per shape: local p-hat at theta (local logistic h 0.15, cell-bootstrap CI90) and the observed rate in [0.9, 1.1)
  (§4.3 is judged on the TRAINING set by refit_supp.py; here it is a diagnostic)
  --fit-json (default: the candidate's fit.json) copies the training-side criteria 3-5 into the report.

DEV mode (default): for development on TRAINING data only.  Sealed roots (M = calib_20261003/M, the E set, any path
  naming a sealed / holdout / validation set) are refused BEFORE any file is opened; only split train / auxiliary rows
  and training shapes are used.  The numbers are in-sample (the candidate was fitted on these cells).
FINAL mode (--i-am-the-final-evaluation --set-name E|M): the one-time evaluation of prereg §4 on a sealed set; all
  splits; refused if this candidate was already evaluated on this set (ledger, --ledger).

Usage:
  dev  : nice -n 19 python3 -B validate_supp.py --candidate FIT/primary/candidate.json --data old --only-src run2,run2b \\
             --sample-cells 40 --out DIR
  final: nice -n 19 python3 -B validate_supp.py --candidate CAND --data E=E_ROOT --out OUT \\
             --i-am-the-final-evaluation --set-name E
"""
import argparse
import hashlib
import json
import os
import re
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import supp_lib as L  # noqa: E402
from supp_lib import r  # noqa: E402

BANDS = ("Z<tau_crit", "tau_crit<=Z<0.9", "0.9<=Z<1.1", "1.1<=Z<tau_high", "Z>=tau_high")
DEFAULT_LEDGER = os.environ.get("TRE_FIT_LEDGER")  # the one-time ledger (11-03: <data root>/calib_supp_20261103_fit_tools/)


def candidate_params(a):
    """{model: params} from --candidate or the explicit flags."""
    if a.candidate:
        c = json.load(open(a.candidate))
        base = {"window_ms": int(c["window_ms"]), "attribution": c["attribution"], "r_star": float(c["r_star"]),
                "variant": c.get("variant"), "candidate_file": os.path.abspath(a.candidate),
                "candidate_sha256": hashlib.sha256(open(a.candidate, "rb").read()).hexdigest(),
                "fit_json": c.get("fit_json")}
        out = {}
        for m, p in c["models"].items():
            if a.models and m not in a.models:
                continue
            out[m] = {**base, **{kk: p[kk] for kk in ("theta_m", "tau_crit", "tau_high", "lambda_wait", "w_p")}}
        return out
    need = ("model", "theta", "tau_crit", "tau_high", "lam", "wp", "r_star")
    miss = [x for x in need if getattr(a, x) is None]
    if miss:
        raise SystemExit(f"without --candidate give {miss}")
    return {a.model: {"window_ms": a.window_ms, "attribution": a.attribution or "first_token", "r_star": a.r_star,
                      "variant": "explicit", "candidate_file": None, "candidate_sha256": None, "fit_json": None,
                      "theta_m": a.theta, "tau_crit": a.tau_crit, "tau_high": a.tau_high, "lambda_wait": a.lam,
                      "w_p": a.wp}}


def ledger_key(set_name, model, p, attribution):
    return json.dumps([set_name, model, p["theta_m"], p["tau_crit"], p["tau_high"], p["lambda_wait"], p["w_p"],
                       p["r_star"], p["window_ms"], attribution])


def band_metrics(K, n, k, p, rng, nboot):
    th, tc, thi = p["theta_m"], p["tau_crit"], p["tau_high"]
    assert 0 < tc < 0.9 and 1.1 < thi, ("band edges out of order", tc, thi)
    Z = K.tss.values / th
    b = np.digitize(Z, [tc, 0.9, 1.1, thi])
    codes, uniq = pd.factorize(K.key.values)
    nc = len(uniq)
    Nc = np.zeros((nc, 5))
    Kc = np.zeros((nc, 5))
    np.add.at(Nc, (codes, b), n)
    np.add.at(Kc, (codes, b), k)
    tot_n, tot_k = Nc.sum(0), Kc.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(tot_n > 0, tot_k / tot_n, np.nan)
    boot = np.full((nboot, 5), np.nan)
    for i in range(nboot):
        w = np.bincount(rng.integers(0, nc, size=nc), minlength=nc).astype(float)
        nb, kb = w @ Nc, w @ Kc
        with np.errstate(invalid="ignore", divide="ignore"):
            boot[i] = np.where(nb > 0, kb / nb, np.nan)
    bands = []
    for j, name in enumerate(BANDS):
        ci, ff = L.pct_ci(boot[:, j], nboot)
        m = b == j
        bands.append({"band": name, "n_req": int(tot_n[j]), "n_viol": int(tot_k[j]), "rate": rate[j], "ci90": ci,
                      "boot_undefined": ff, "n_windows": int((m & (n > 0)).sum()),
                      "n_cells": int(len(set(codes[m & (n > 0)])))})
    nonempty = [j for j in range(5) if tot_n[j] > 0]
    viol_pairs = [(BANDS[i], BANDS[j]) for i, j in zip(nonempty, nonempty[1:]) if rate[j] > rate[i]]
    mono_boot = []
    for row in boot:
        v = [row[j] for j in nonempty if np.isfinite(row[j])]
        mono_boot.append(all(y <= x for x, y in zip(v, v[1:])))
    lo, hi = 0.5 * p["r_star"], 2.0 * p["r_star"]
    c = bands[2]
    if c["n_req"] == 0 or c["ci90"][0] is None:
        c1 = {"pass": False, "why": "no (or too few) requests in Z in [0.9, 1.1)"}
    else:
        c1 = {"pass": bool(c["ci90"][1] >= lo and c["ci90"][0] <= hi), "point_in_band": bool(lo <= c["rate"] <= hi)}
    c1.update(rule="Z in [0.9, 1.1): cell-bootstrap CI90 of the per-request violation rate meets [0.5, 2] x r*",
              band=[lo, hi], rate=c["rate"], ci90=c["ci90"], n_req=c["n_req"], n_cells=c["n_cells"])
    c2 = {"rule": "per-request violation rate non-increasing over the Z bands (point estimates, empty bands skipped)",
          "pass": not viol_pairs, "increasing_pairs": viol_pairs, "rates": [bd["rate"] for bd in bands],
          "bands_used": [BANDS[j] for j in nonempty], "boot_share_monotone": float(np.mean(mono_boot))}
    return bands, c1, c2, Z


def per_shape(K, n, k, Z, p, rng, nboot):
    x = np.log(K.tss.values)
    x0 = np.log(p["theta_m"])
    out = []
    for s in sorted(K["shape"].unique()):
        m = K["shape"].values == s
        xs, ks, ns = x[m], k[m], n[m]
        codes, uniq = pd.factorize(K.key.values[m])
        nc = len(uniq)
        bs = []
        for _ in range(nboot):
            wt = np.bincount(rng.integers(0, nc, size=nc), minlength=nc)[codes].astype(float)
            bs.append(L.p_at(xs, ks, ns, x0, wt=wt))
        ci, ff = L.pct_ci(bs, nboot)
        mb = m & (Z >= 0.9) & (Z < 1.1)
        out.append({"shape": s, "class": L.shape_class(s), "n_cells": nc, "n_req": int(ns.sum()),
                    "p_at_theta": L.p_at(xs, ks, ns, x0), "p_at_theta_ci90": ci, "p_at_theta_bootfail": ff,
                    "n_eff_near_theta": L.neff_at(xs, ns, x0),
                    "rate_Z_0.9_1.1": float(k[mb].sum() / n[mb].sum()) if n[mb].sum() > 0 else None,
                    "n_req_Z_0.9_1.1": int(n[mb].sum())})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", default=None, help="candidate.json from refit_supp.py")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--theta", type=float)
    ap.add_argument("--tau-crit", type=float)
    ap.add_argument("--tau-high", type=float)
    ap.add_argument("--lam", type=float)
    ap.add_argument("--wp", type=float)
    ap.add_argument("--r-star", type=float)
    ap.add_argument("--window-ms", type=int, default=10000)
    ap.add_argument("--attribution", choices=tuple(L.TCOL), default=None, help="default: the candidate's")
    ap.add_argument("--data", action="append", required=True, help="'old', 'old=ROOT', NAME=ROOT (repeat)")
    ap.add_argument("--only-src", default=None, help="comma list of source names to keep")
    ap.add_argument("--sample-cells", type=int, default=0, help="dev: keep N random cells per model")
    ap.add_argument("--sample-seed", type=int, default=20261103)
    ap.add_argument("--cell-regex", default=None, help="dev: keep cells whose key matches")
    ap.add_argument("--shapes", default=None, help="comma list of shapes to keep")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--nboot-shape", type=int, default=200)
    ap.add_argument("--fit-json", default=None, help="fit.json whose training-side criteria 3-5 are copied")
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-am-the-final-evaluation", dest="final", action="store_true")
    ap.add_argument("--set-name", default=None, help="final mode: E or M")
    ap.add_argument("--ledger", default=DEFAULT_LEDGER, help="final mode: the one-time ledger (default $TRE_FIT_LEDGER)")
    ap.add_argument("--final-splits", default="all", help="final mode: 'all' or a comma list of split values")
    ap.add_argument("--sealed-pattern", action="append", default=[])
    ap.add_argument("--allow-contract-mismatch", action="store_true")
    ap.add_argument("--no-hash", action="store_true")
    a = ap.parse_args(argv)
    L.marker_check()
    params = candidate_params(a)
    if not params:
        raise SystemExit("no model to evaluate")
    W_ms = {p["window_ms"] for p in params.values()}
    assert len(W_ms) == 1
    W_ms = W_ms.pop()
    L.configure(W_ms)
    if a.final:
        if a.set_name not in ("E", "M"):
            raise SystemExit("--i-am-the-final-evaluation needs --set-name E or M")
        if not a.ledger:
            raise SystemExit("--i-am-the-final-evaluation needs --ledger (or TRE_FIT_LEDGER): the one-time ledger")
        if a.sample_cells or a.cell_regex or a.shapes or a.only_src:
            raise SystemExit("final mode evaluates the whole set: no --sample-cells / --cell-regex / --shapes / --only-src")
        splits = None if a.final_splits == "all" else tuple(a.final_splits.split(","))
        print("=" * 100 + f"\nFINAL one-time evaluation on the sealed set {a.set_name} (prereg §4)\n" + "=" * 100,
              file=sys.stderr, flush=True)
        done_keys = set()
        if os.path.isfile(a.ledger):
            for line in open(a.ledger):
                if line.strip():
                    done_keys.add(json.loads(line)["ledger_key"])
    else:
        splits = L.TRAIN_SPLITS
        if a.set_name:
            raise SystemExit("--set-name is for the final evaluation only")
    fit_json = a.fit_json or next(iter(params.values())).get("fit_json")
    fitc = json.load(open(fit_json))["models"] if fit_json and os.path.isfile(fit_json) else {}
    only = set(a.only_src.split(",")) if a.only_src else None
    report = {"mode": "FINAL" if a.final else "DEV (training data only; in-sample)", "set_name": a.set_name,
              "argv": sys.argv if argv is None else argv, "host": os.uname().nodename, "tool_sha256": L.tool_hashes(),
              "time": pd.Timestamp.now().isoformat(timespec="seconds"), "data": a.data, "fit_json": fit_json,
              "models": {}}
    lines = [f"validate_supp {report['mode']}  set {a.set_name or '-'}  data {a.data}  window {W_ms} ms"]
    for model, p in params.items():
        attribution = a.attribution or p["attribution"]
        if a.final:
            lk = ledger_key(a.set_name, model, p, attribution)
            if lk in done_keys:
                raise SystemExit(f"{model}: this candidate was already evaluated on {a.set_name} ({a.ledger}); "
                                 "prereg §4: each candidate is evaluated once")
        srcs = L.resolve_specs(a.data, model, W_ms, final=a.final, extra=a.sealed_pattern, only_src=only)
        if not srcs:
            lines.append(f"{model}: no data source - skipped")
            continue
        md = L.ModelData(model, W_ms, srcs, splits, final=a.final, extra=a.sealed_pattern,
                         hash_files=not a.no_hash, allow_contract=a.allow_contract_mismatch)
        lam, wp = p["lambda_wait"], p["w_p"]
        t = L.tss_vec(md.Wall, lam, wp)
        L.assert_tss(md.Wall, lam, wp, t)
        K = L.kept(md.Wall, t, W_ms, splits)
        sub = []
        if a.cell_regex:
            K = K[K.key.str.contains(a.cell_regex, regex=True)]
            sub.append(f"cell regex {a.cell_regex!r}")
        if a.shapes:
            K = K[K["shape"].isin(a.shapes.split(","))]
            sub.append(f"shapes {a.shapes}")
        if a.sample_cells:
            keys = np.array(sorted(K.key.unique()))
            rs = np.random.default_rng(a.sample_seed + L.zlib.crc32(model.encode()) % 1000)
            pick = set(rs.choice(keys, size=min(a.sample_cells, len(keys)), replace=False))
            K = K[K.key.isin(pick)]
            sub.append(f"{len(pick)} random cells (seed {a.sample_seed})")
        K = K.reset_index(drop=True)
        if not len(K):
            raise SystemExit(f"{model}: no kept window left")
        C = L.attr_counts(K, md.Q, L.TCOL[attribution], ["v"])
        n, k = C[:, 0], C[:, 1]
        rng = np.random.default_rng(L.model_seed(model) + 7)
        bands, c1, c2, Z = band_metrics(K, n, k, p, rng, a.nboot)
        shp = per_shape(K, n, k, Z, p, rng, a.nboot_shape)
        fm = fitc.get(model, {}).get("criteria") if fitc else None
        mrep = {"params": p, "attribution": attribution, "subset": sub, "n_windows_kept": int(len(K)),
                "n_windows": int((n > 0).sum()), "n_cells": int(K.key.nunique()), "n_requests": int(n.sum()),
                "n_viol": int(k.sum()), "sources": md.prov, "notes": md.notes, "bands": bands,
                "criteria": {"c1_rate_Z_0.9_1.1": c1, "c2_monotone_bands": c2}, "per_shape": shp,
                "training_side_criteria_from_fit": fm}
        if a.final:
            mrep["sealed_reasons"] = {s["name"]: L.sealed_reason(s["ddir"], a.sealed_pattern) for s in srcs}
        report["models"][model] = mrep
        lines.append(f"-- {model}: theta {p['theta_m']} tau_crit {p['tau_crit']} tau_high {p['tau_high']} lambda {lam} "
                     f"w_p {wp} r* {p['r_star']} ({p.get('variant')}), attribution {attribution}; {mrep['n_cells']} "
                     f"cells, {mrep['n_windows']} windows, {mrep['n_requests']} requests {('; ' + '; '.join(sub)) if sub else ''}")
        for bd in bands:
            ci = bd["ci90"]
            rate = "  -   " if not np.isfinite(bd["rate"]) else f"{100 * bd['rate']:6.2f}%"
            cis = "[NaN]" if ci[0] is None else f"[{100 * ci[0]:.2f}, {100 * ci[1]:.2f}]"
            lines.append(f"   {bd['band']:16s} {rate} {cis:18s} n_req {bd['n_req']:7d}  windows {bd['n_windows']:5d}  "
                         f"cells {bd['n_cells']:4d}")
        c1ci = c1.get("ci90") or [None, None]
        c1s = "[NaN]" if c1ci[0] is None else f"[{100 * c1ci[0]:.2f}, {100 * c1ci[1]:.2f}] %"
        lines.append(f"   c1 {'PASS' if c1['pass'] else 'FAIL'}: band [{100 * c1['band'][0]:.2f}, {100 * c1['band'][1]:.2f}] %"
                     f" vs rate CI90 {c1s}  |  c2 {'PASS' if c2['pass'] else 'FAIL'} (increasing pairs "
                     f"{c2['increasing_pairs']}; bootstrap share monotone {c2['boot_share_monotone']:.2f})")
        lines.append("   per shape p@theta [CI90] (rate in 0.9-1.1, n): " + "; ".join(
            f"{x['shape']} " + ("NaN" if x["p_at_theta"] is None or not np.isfinite(x["p_at_theta"]) else f"{100 * x['p_at_theta']:.2f}%")
            + ("" if x["p_at_theta_ci90"][0] is None else f"[{100 * x['p_at_theta_ci90'][0]:.2f},{100 * x['p_at_theta_ci90'][1]:.2f}]")
            + (f" ({100 * x['rate_Z_0.9_1.1']:.2f}%, {x['n_req_Z_0.9_1.1']})" if x["rate_Z_0.9_1.1"] is not None else " (-, 0)")
            for x in shp))
        if fm:
            c3, c4, c5 = (fm.get("c3_new_shapes_p_at_theta", {}), fm.get("c4_tau_high_interpolated", {}),
                          fm.get("c5_theta_ci", {}))
            lines.append(f"   training side (fit.json): c3 {c3.get('status') or ('PASS' if c3.get('pass') else 'FAIL')} | c4 "
                         f"{'PASS' if c4.get('pass') else 'FAIL'} {c4.get('n_interpolated')}/{c4.get('n_old_shapes_in_ladder')} | "
                         f"c5 {'PASS' if c5.get('pass') else 'FAIL'} (hw {c5.get('halfwidth_rel')}, bootfail {c5.get('bootfail')})")
        del md
    os.makedirs(a.out, exist_ok=True)
    tag = (a.set_name or "dev")
    with open(os.path.join(a.out, f"validate_{tag}.json"), "w") as f:
        json.dump(L.clean(report), f, indent=1, default=L.json_default)
    with open(os.path.join(a.out, f"validate_{tag}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    if a.final:
        with open(a.ledger, "a") as f:
            for model, mrep in report["models"].items():
                p = mrep["params"]
                f.write(json.dumps({"ledger_key": ledger_key(a.set_name, model, p, mrep["attribution"]),
                                    "time": report["time"], "out": os.path.abspath(a.out),
                                    "c1": mrep["criteria"]["c1_rate_Z_0.9_1.1"]["pass"],
                                    "c2": mrep["criteria"]["c2_monotone_bands"]["pass"],
                                    "candidate_sha256": p.get("candidate_sha256"),
                                    "inputs": {s["name"]: {fn: v.get("sha256") for fn, v in s["files"].items()}
                                               for s in mrep["sources"]}}) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

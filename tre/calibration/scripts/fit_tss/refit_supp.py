#!/usr/bin/env python3
"""refit_supp.py - per-model refit of theta / tau_crit / tau_high / lambda / w_p for the supplementary TSS calibration
(prereg-calib-supplement-20261103.md §2 primary analysis, §3 S1-S5, §4 criteria 3-5).  CPU only; read-only on the
datasets; writes only under --out.  Sealed held-out sets (M, E) are refused before any file is opened (no override).

Rules (primary, prereg §2):
  attribution  first_token: a request belongs to the window whose (end - 10 s, end] holds send_ts + ttft
               (failures without a first token: done / abort time); S4 = completion time
  label        TTFT > 500 ms or TPOT > 75 ms or not served (outcome != ok / http != 200: failed, timed out, truncated)
  lambda, w_p  formula on the training set: lambda = 0.15 L_out, w_p = 6.667 / L_in (request-weighted means of the
               SERVED requests attributed to the kept training windows); fitted with lambda rounded to 0.01 and w_p to
               1e-6 (= the published values).  S2 = fixed 20 / 0.01
  estimator    local logistic on log TSS (h 0.15, refit_fixedslo.local_root), cluster bootstrap over cells (--nboot 200)
  theta        TSS at r* = 1.5 % (S1: 5 %), published to 0.1;  tau_crit = theta(17 %) / theta_pub, to 0.01
  tau_high     min(round_.01(median over ladder shapes of the n = 3 value), floor_.01(0.85 C_m / theta_pub)); width
               tau_high - 1 >= 0.2 required.  Ladder = hold / static cells (TSS cell median) of run2 / run2b + every added
               source (--ladder-src).  C_m = C_m^D (W / 10 s) q99(TSS) / q99(TSS at lambda_D, w_p 0), same windows
  shapes       all (S3: S1, S2, S4, S5, T9 + new shapes with input <= 1100); window 10 s (S5: --window-ms 20000 / 30000)
Data: --data old (the 10-03 training roots at this window) and --data NAME=ROOT for the new campaign (ROOT laid out by
calibration_dataset.py: ROOT/<model>/dataset_hybrid/, or ROOT/<src>/<model>/dataset_hybrid/ for several sources).
Only split train / auxiliary rows; only S1-S5, T8, T9 and G<in>x<out> shapes (an unknown shape stops the fit).

Usage:
  nice -n 19 python3 -B refit_supp.py --out DIR [--variant primary --variant S1 ... | --variant all10]
          [--window-ms 10000] [--data old --data A=ROOT_A --data B=ROOT_B] [--models dsqwen-7b ...]
  S5:     ... --variant S5 --window-ms 20000 --data old --data A=ROOT_A_W20 ...
  custom: ... --variant custom --attribution completion --lam-wp fixed --lam 20 --wp 0.01 --r-star 0.05 --tag X
Outputs per variant: OUT/<tag>/{fit.json, candidate.json, summary.txt, per_shape.csv, tau_high_by_shape.csv}
"""
import argparse
import copy
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import supp_lib as L  # noqa: E402
from supp_lib import r  # noqa: E402

VARIANT_ALL10 = ("primary", "S1", "S2", "S3", "S4")


def fit_one(md, cfg, nboot, rng, ladder_src, prims=L.LADDER_PRIMITIVES, rounding="half_up"):
    """One model, one analysis configuration -> result dict.  cfg: attribution, lamwp (formula|fixed), lam, wp,
    r_star, shapes.  rng: one integer draw per bootstrap resample (= refit.cluster_weights)."""
    model, W_ms = md.model, md.window_ms
    tcol = L.TCOL[cfg["attribution"]]
    r_star = float(cfg["r_star"])
    # 1. kept training windows at a reference positive w_p (any w_p > 0 gives the same set: asserted in 3.)
    K0 = L.kept(md.Wall, L.tss_vec(md.Wall, 20.0, 0.01), W_ms, md.splits)
    K0, shapes_used = L.shape_subset(K0, cfg["shapes"])
    C = L.attr_counts(K0, md.Q, tcol, ["v", "served", "lin", "lout", "fail"])
    # 2. lambda / w_p
    L_in, L_out, n_served, lam_x, wp_x = L.lam_wp_from(C)
    if cfg["lamwp"] == "formula":
        lam, wp = round(lam_x, 2), round(wp_x, 6)
    else:
        lam, wp = float(cfg["lam"]), float(cfg["wp"])
    # 3. TSS on the basis (asserted = recompute_tss_rows), kept set unchanged
    t_all = L.tss_vec(md.Wall, lam, wp)
    L.assert_tss(md.Wall, lam, wp, t_all)
    K = L.kept(md.Wall, t_all, W_ms, md.splits)
    K = K[K["shape"].isin(shapes_used)].reset_index(drop=True)
    assert np.array_equal(K._row.values, K0._row.values), "kept window set depends on w_p"
    tss = K.tss.values
    x, n, k = np.log(tss), C[:, 0], C[:, 1]
    res = {"model": model, "window_ms": W_ms, "attribution": cfg["attribution"], "r_star": r_star,
           "lam_wp_rule": cfg["lamwp"], "shapes_used": shapes_used, "lambda_wait": lam, "w_p": wp,
           "L_in": L_in, "L_out": L_out, "n_served": n_served, "lambda_formula_exact": lam_x, "w_p_formula_exact": wp_x,
           "n_windows_kept": int(len(K)), "n_windows": int((n > 0).sum()), "n_cells": int(K.key.nunique()),
           "n_requests": int(n.sum()), "n_viol": int(k.sum()), "viol_rate": float(k.sum() / max(n.sum(), 1)),
           "n_fail": int(C[:, 5].sum()), "sources": sorted(K.src.unique()),
           "windows_by_source": {s: int(v) for s, v in K.src.value_counts().items()}}
    # 4. theta, theta17 (local logistic), bootstrap
    xr, slope_r = r.local_root(x, k, n, r_star)
    xc, _ = r.local_root(x, k, n, L.R_CRIT)
    theta, theta17 = float(np.exp(xr)), float(np.exp(xc))
    ok_theta = bool(np.isfinite(theta))
    theta_pub = round(theta, 1) if ok_theta else None
    xt = np.log(theta_pub) if ok_theta else np.nan
    shapes = sorted(K["shape"].unique())
    masks = {s: K["shape"].values == s for s in shapes}
    draw, n_clusters = L.cluster_weight_fn(K.key.values)
    br, bc, bp, bps = [], [], [], {s: [] for s in shapes}
    for _ in range(nboot):
        wt = draw(rng)
        br.append(r.local_root(x, k, n, r_star, wt=wt, x0=xr)[0])
        bc.append(r.local_root(x, k, n, L.R_CRIT, wt=wt, x0=xc)[0])
        if ok_theta:
            bp.append(L.p_at(x, k, n, xt, wt=wt))
            for s in shapes:
                m = masks[s]
                bps[s].append(L.p_at(x[m], k[m], n[m], xt, wt=wt[m]))
    br, bc = np.array(br, float), np.array(bc, float)
    lo, hi, bf = r.boot_ci(np.exp(br), nboot)
    lo17, hi17, bf17 = r.boot_ci(np.exp(bc), nboot)
    tclo, tchi, tcbf = r.boot_ci(np.exp(bc - br), nboot)
    hw = (hi - lo) / (2 * theta) if np.isfinite(lo) and np.isfinite(hi) and ok_theta else np.nan
    res.update(theta=theta, theta_pub=theta_pub, theta_ci90=[lo, hi], theta_ci90_halfwidth_rel=hw,
               theta_ci90_rel=[lo / theta - 1, hi / theta - 1] if ok_theta else [np.nan, np.nan],
               theta_bootfail=bf, local_slope_at_theta=slope_r, n_eff_near_theta=L.neff_at(x, n, xt) if ok_theta else 0,
               theta17=theta17, theta17_ci90=[lo17, hi17], theta17_bootfail=bf17, nboot=nboot, n_clusters=n_clusters)
    if not ok_theta:
        res["failed"] = f"no local-logistic root at r* {r_star}"
        return res
    tau_crit = round(theta17 / theta_pub, 2) if np.isfinite(theta17) else None
    res.update(tau_crit=tau_crit, tau_crit_exact=theta17 / theta_pub, tau_crit_ci90=[tclo, tchi], tau_crit_bootfail=tcbf)
    pci, pbf = L.pct_ci(bp, nboot)
    res.update(p_at_theta=L.p_at(x, k, n, xt), p_at_theta_ci90=pci, p_at_theta_bootfail=pbf,
               p_at_tau_crit=L.p_at(x, k, n, np.log(tau_crit * theta_pub)) if tau_crit else None)
    Z = tss / theta_pub
    mz = (Z >= 0.5) & (Z <= 2.0)
    res.update(auroc_global=r.auroc(-tss, k, n), auroc_local=r.auroc(-tss[mz], k[mz], n[mz]),
               n_req_local=int(n[mz].sum()))
    # 5. C_m (q99 ratio, rstar_F) and tau_high (ladder n = 3)
    q99 = float(np.quantile(tss, .99))
    tD = L.tss_vec(K, L.LAM_D[model], 0.0)
    q99D = float(np.quantile(tD[np.isfinite(tD) & (tD > 0)], .99))
    C_m_D = L.CM_D_10S[model] * W_ms / 10000.0
    C_m = round(C_m_D * q99 / q99D, 1)
    cell, lnotes = L.ladder_cells(K, ladder_src, prims)
    lad = L.ladder(cell)
    th = L.tau_high_rule(lad, theta_pub, q99, C_m, rounding)
    linfo = L.ladder_info(cell, lad)
    res.update(C_m_D=C_m_D, C_m=C_m, tss_q99=q99, tss_q99_D=q99D, lambda_D=L.LAM_D[model],
               ladder_sources=sorted(set(ladder_src) & set(K.src.unique())), ladder_notes=lnotes,
               tau_high=th["value"], tau_high_detail={kk: v for kk, v in th.items() if kk != "rows"})
    # 6. per shape
    rows_by = {row["shape"]: row for row in th["rows"]}
    per = []
    for s in shapes:
        m = masks[s]
        ps = L.p_at(x[m], k[m], n[m], xt)
        ci, pf = L.pct_ci(bps[s], nboot)
        zr = r.local_root(x[m], k[m], n[m], r_star, h=0.2)[0]
        g = K[m]
        row = {"model": model, "shape": s, "class": L.shape_class(s), "n_cells": int(g.key.nunique()),
               "n_windows": int((n[m] > 0).sum()), "n_requests": int(n[m].sum()),
               "viol_rate": float(k[m].sum() / max(n[m].sum(), 1)), "p_at_theta": ps, "p_at_theta_ci90": ci,
               "p_at_theta_bootfail": pf, "n_eff_near_theta": L.neff_at(x[m], n[m], xt),
               "Z_at_rstar_shape_h0.2": float(np.exp(zr) / theta_pub) if np.isfinite(zr) else None,
               "Z_max": float(Z[m].max()), "Z_min": float(Z[m].min())}
        lr = rows_by.get(s, {})
        row.update({kk: lr.get(kk) for kk in ("bracketed", "rho1", "tau_high_n2", "tau_high_n3", "tau_high_n4",
                                               "in_ladder_n2", "in_ladder_n3", "in_ladder_n4", "note")})
        li = linfo.get(s)
        if li:
            row.update({"ladder_" + kk: v for kk, v in li.items()})
        per.append(row)
    res["per_shape"] = per
    # 7. prereg §4 criteria 3-5 (training side) and the §2.7 width rule
    blo, bhi = 0.5 * r_star, 2.0 * r_star
    new = [row for row in per if row["class"] == "new"]
    c3 = {"rule": "each new (B) shape: local p-hat at theta within [0.5, 2] x r* (point estimate)",
          "band": [blo, bhi], "shapes": {}}
    for row in new:
        p = row["p_at_theta"]
        c3["shapes"][row["shape"]] = {"p": p, "ci90": row["p_at_theta_ci90"], "n_eff": row["n_eff_near_theta"],
                                      "in_band": bool(np.isfinite(p) and blo <= p <= bhi) if p is not None else False}
    c3["pass"] = all(v["in_band"] for v in c3["shapes"].values()) if new else None
    if not new:
        c3["status"] = "n/a: no new shape in the training set"
    old_rows = [row for row in per if row["class"] == "old7" and row.get("tau_high_n3") is not None]
    n_int = sum(1 for row in old_rows if row.get("in_ladder_n3") is True)
    c4 = {"rule": ">= 5 of the old 7 shapes have tau_high (n = 3) interpolated", "n_interpolated": n_int,
          "n_old_shapes_in_ladder": len(old_rows), "pass": n_int >= 5,
          "interpolated": [row["shape"] for row in old_rows if row.get("in_ladder_n3") is True],
          "extrapolated": [row["shape"] for row in old_rows if row.get("in_ladder_n3") is False]}
    c5 = {"rule": "theta CI90 half-width <= 15 % and bootstrap failure <= 0.05", "halfwidth_rel": hw,
          "bootfail": bf, "pass": bool(np.isfinite(hw) and hw <= 0.15 and bf <= 0.05)}
    res["criteria"] = {"c3_new_shapes_p_at_theta": c3, "c4_tau_high_interpolated": c4, "c5_theta_ci": c5,
                       "width_rule_tau_high": {"tau_high_minus_1": th["value"] - 1.0, "pass": th["width_ok"]}}
    res["publish"] = {"theta_m": theta_pub, "tau_crit": tau_crit, "tau_high": th["value"], "lambda_wait": lam, "w_p": wp}
    # 8. prereg §5: against live tune F (10 s only)
    if W_ms == 10000:
        live = L.LIVE_F[model]
        tin = bool(np.isfinite(lo) and np.isfinite(hi) and lo <= live["theta_m"] <= hi)
        dtc = (tau_crit - live["tau_crit"]) if tau_crit is not None else None
        dth = th["value"] - live["tau_high"]
        unchanged = bool(tin and dtc is not None and abs(dtc) <= 0.03 + 1e-9 and abs(dth) <= 0.03 + 1e-9)
        res["vs_live_F"] = {"live": live, "theta_live_in_new_ci90": tin, "d_tau_crit": dtc, "d_tau_high": dth,
                            "theta_ratio_new_over_live": theta_pub / live["theta_m"],
                            "basis_differs": bool(abs(lam - live["lambda_wait"]) > 1e-9 or abs(wp - live["w_p"]) > 1e-12),
                            "rule_outcome": "live unchanged" if unchanged else "publish atomically + re-run TRE arms"}
    return res


def fmt_ci(ci, nd=1):
    if ci is None or ci[0] is None or not np.isfinite(ci[0]):
        return "[NaN]"
    return f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def summary_lines(tag, results, header):
    out = [f"== {tag}  {header}"]
    out.append("model       | theta_pub [CI90] hw bf | th17 | tau_crit [CI] | tau_high (ladder n3 med -> rnd | cap) by, "
               "width | lambda w_p (L_in L_out) | AUROC loc/glob | p@theta [CI] | n_win n_req viol")
    for m, x in results.items():
        if "failed" in x:
            out.append(f"{m:11s} | FAILED: {x['failed']}")
            continue
        td = x["tau_high_detail"]
        out.append(
            f"{m:11s} | {x['theta_pub']:.1f} {fmt_ci(x['theta_ci90'])} {100 * (x['theta_ci90_halfwidth_rel'] or np.nan):.1f}% "
            f"{x['theta_bootfail']:.3f} | {x['theta17']:.1f} | {x['tau_crit']:.2f} {fmt_ci(x['tau_crit_ci90'], 3)} | "
            f"{x['tau_high']:.2f} ({(td['ladder_median_n3'] or np.nan):.4f} -> {td['ladder_rounded']} | {td['cap']:.2f}) "
            f"{td['bound_by']}, {'ok' if td['width_ok'] else 'WIDTH<0.2'} | {x['lambda_wait']:g} {x['w_p']:g} "
            f"({x['L_in']:.1f} {x['L_out']:.1f}) | {x['auroc_local']:.3f}/{x['auroc_global']:.3f} | "
            f"{100 * x['p_at_theta']:.2f}% {fmt_ci([100 * v if v is not None else None for v in x['p_at_theta_ci90']], 2)} | "
            f"{x['n_windows']} {x['n_requests']} {100 * x['viol_rate']:.1f}%")
    out.append("  tau_high n = 3 per shape (I = interpolated in the hold ladder, E = extrapolated; * = theta outside the "
               "ladder's TSS range) and p-hat at theta per shape [CI90]:")
    for m, x in results.items():
        if "failed" in x:
            continue
        parts = []
        for row in x["per_shape"]:
            t3 = row.get("tau_high_n3")
            tag3 = ("I" if row.get("in_ladder_n3") else "E") if t3 is not None else "-"
            br = "" if row.get("bracketed") in (True, None) else "*"
            pc = row["p_at_theta_ci90"]
            pstr = f"{100 * row['p_at_theta']:.2f}%" if row["p_at_theta"] is not None and np.isfinite(row["p_at_theta"]) else "NaN"
            cstr = f"[{100 * pc[0]:.2f},{100 * pc[1]:.2f}]" if pc[0] is not None else "[NaN]"
            parts.append(f"{row['shape']} {('%.2f' % t3) if t3 is not None else '-'}{tag3}{br} p {pstr}{cstr}")
        out.append(f"  {m:11s} " + "; ".join(parts))
    out.append("  criteria (training side):  c3 new shapes p@theta in [.5, 2] r* | c4 >= 5 of 7 old interpolated | "
               "c5 CI90 hw <= 15 % & bootfail <= .05 | width tau_high - 1 >= .2 | vs live F (prereg §5)")
    for m, x in results.items():
        if "failed" in x:
            continue
        c = x["criteria"]
        c3 = c["c3_new_shapes_p_at_theta"]
        c3s = c3.get("status") or ("PASS" if c3["pass"] else "FAIL") + " " + ", ".join(
            f"{s} {100 * v['p']:.2f}%" if v["p"] is not None and np.isfinite(v["p"]) else f"{s} NaN"
            for s, v in c3["shapes"].items())
        c4 = c["c4_tau_high_interpolated"]
        c5 = c["c5_theta_ci"]
        vl = x.get("vs_live_F")
        vls = (f"{vl['rule_outcome']} (theta {x['theta_pub']:.1f} vs {vl['live']['theta_m']}, in CI {vl['theta_live_in_new_ci90']}, "
               f"d tau_c {vl['d_tau_crit']:+.2f}, d tau_h {vl['d_tau_high']:+.2f}, basis differs {vl['basis_differs']})"
               if vl else "n/a (window != 10 s)")
        out.append(f"  {m:11s} c3 {c3s} | c4 {'PASS' if c4['pass'] else 'FAIL'} {c4['n_interpolated']}/"
                   f"{c4['n_old_shapes_in_ladder']} | c5 {'PASS' if c5['pass'] else 'FAIL'} | width "
                   f"{'PASS' if c['width_rule_tau_high']['pass'] else 'FAIL'} | {vls}")
        for note in x.get("ladder_notes", []):
            out.append(f"  {m:11s} ladder note: {note}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", action="append", default=None,
                    help="primary | S1 | S2 | S3 | S4 | S5 (needs --window-ms 20000/30000) | all10 (= primary,S1-S4) | custom")
    ap.add_argument("--window-ms", type=int, default=10000)
    ap.add_argument("--data", action="append", default=None, help="'old' (default), 'old=ROOT', NAME=ROOT (repeat)")
    ap.add_argument("--ladder-src", action="append", default=None,
                    help="sources whose hold cells form the tau_high ladder (default run2, run2b + every added source)")
    ap.add_argument("--ladder-primitives", default=",".join(L.LADDER_PRIMITIVES))
    ap.add_argument("--models", nargs="+", default=list(L.MODELS))
    ap.add_argument("--nboot", type=int, default=L.NBOOT)
    ap.add_argument("--seed", type=int, default=None, help="bootstrap seed base (default per model: 20261103 + crc32)")
    ap.add_argument("--tau-high-round", choices=("half_up", "ceil"), default="half_up",
                    help="rounding of the ladder median (half_up = tune F make_tuneF.py = prereg §2.7, round half up)")
    ap.add_argument("--attribution", choices=tuple(L.TCOL), default=None)
    ap.add_argument("--lam-wp", choices=("formula", "fixed"), default=None)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--wp", type=float, default=None)
    ap.add_argument("--r-star", type=float, default=None)
    ap.add_argument("--shapes", default=None, help="all | le1100 | comma list")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--allow-shapes", default="", help="comma list of extra training shape names")
    ap.add_argument("--allow-contract-mismatch", action="store_true")
    ap.add_argument("--warmup-fallback-s", type=float, default=0.0)
    ap.add_argument("--sealed-pattern", action="append", default=[])
    ap.add_argument("--no-hash", action="store_true", help="skip sha256 of the input files")
    a = ap.parse_args(argv)
    L.marker_check()
    L.configure(a.window_ms)
    variants = a.variant or ["primary"]
    if "all10" in variants:
        variants = [v for v in variants if v != "all10"] + [v for v in VARIANT_ALL10 if v not in variants]
    cfgs = {}
    for v in variants:
        if v == "custom":
            cfg = dict(L.PRESETS["primary"])
            for kk, val in (("attribution", a.attribution), ("lamwp", a.lam_wp), ("lam", a.lam), ("wp", a.wp),
                            ("r_star", a.r_star), ("shapes", a.shapes)):
                if val is not None:
                    cfg[kk] = val
            if cfg["lamwp"] == "fixed" and (cfg.get("lam") is None or cfg.get("wp") is None):
                raise SystemExit("--lam-wp fixed needs --lam and --wp")
            tag = a.tag or "custom"
        else:
            if v not in L.PRESETS:
                raise SystemExit(f"unknown variant {v}")
            if any(x is not None for x in (a.attribution, a.lam_wp, a.lam, a.wp, a.r_star, a.shapes)):
                raise SystemExit("--attribution/--lam-wp/--lam/--wp/--r-star/--shapes apply to --variant custom only")
            if v == "S5" and a.window_ms == 10000:
                raise SystemExit("S5 = the primary rules at --window-ms 20000 or 30000")
            cfg = copy.deepcopy(L.PRESETS[v])
            tag = v if a.window_ms == 10000 else f"{v}_w{a.window_ms}"
        cfgs[tag] = cfg
    data = a.data or ["old"]
    added_names_all = set()
    prims = tuple(x for x in a.ladder_primitives.split(",") if x)
    results = {tag: {} for tag in cfgs}
    data_prov = {}
    for model in a.models:
        L.log(f"{model}: loading {data}")
        srcs = L.resolve_specs(data, model, a.window_ms, extra=a.sealed_pattern)
        md = L.ModelData(model, a.window_ms, srcs, L.TRAIN_SPLITS, extra=a.sealed_pattern,
                         allow_shapes=tuple(x for x in a.allow_shapes.split(",") if x), hash_files=not a.no_hash,
                         allow_contract=a.allow_contract_mismatch, warmup_fallback_s=a.warmup_fallback_s)
        added = [s["name"] for s in srcs if s["origin"] != "old"]
        added_names_all |= set(added)
        lsrc = a.ladder_src or (list(L.LADDER_SRC_OLD) + added)
        data_prov[model] = {"sources": md.prov, "notes": md.notes, "ladder_src": lsrc}
        for note in md.notes:
            L.log(f"{model}: {note}")
        L.log(f"{model}: {len(md.Wall)} window rows, {len(md.Q)} requests from {[s['name'] for s in srcs]}")
        for tag, cfg in cfgs.items():
            seed = (a.seed if a.seed is not None else L.model_seed(model))
            rng = np.random.default_rng(seed)
            L.log(f"{model} {tag}: fitting {cfg}")
            res = fit_one(md, cfg, a.nboot, rng, lsrc, prims, a.tau_high_round)
            res["bootstrap_seed"] = seed
            results[tag][model] = res
            if "failed" in res:
                L.log(f"{model} {tag}: FAILED {res['failed']}")
            else:
                L.log(f"{model} {tag}: theta {res['theta_pub']} {fmt_ci(res['theta_ci90'])} bf {res['theta_bootfail']:.3f} "
                      f"tau_crit {res['tau_crit']} tau_high {res['tau_high']} lambda {res['lambda_wait']} w_p {res['w_p']}")
        del md
    os.makedirs(a.out, exist_ok=True)
    for tag, cfg in cfgs.items():
        d = os.path.join(a.out, tag)
        os.makedirs(d, exist_ok=True)
        meta = {"tool": os.path.abspath(__file__), "tool_sha256": L.tool_hashes(),
                "argv": sys.argv if argv is None else argv, "variant": tag,
                "config": cfg, "window_ms": a.window_ms, "nboot": a.nboot, "tau_high_round": a.tau_high_round,
                "ladder_primitives": prims, "data": data, "host": os.uname().nodename,
                "time": pd.Timestamp.now().isoformat(timespec="seconds"),
                "prereg": "tre_local_workspace/docs/prereg-calib-supplement-20261103.md",
                "constants": {"TTFT_SLO_MS": L.TTFT_SLO_MS, "TPOT_SLO_MS": L.TPOT_SLO_MS, "H_LOCAL": L.H_LOCAL,
                              "R_CRIT": L.R_CRIT, "CAP": L.CAP, "MIN_W": L.MIN_W, "CM_D_10S": L.CM_D_10S,
                              "LAM_D": L.LAM_D, "TRAIN_SPLITS": L.TRAIN_SPLITS, "STEP_MS": L.STEP_MS, "TRIM": L.TRIM}}
        fit = L.clean({"meta": meta, "data": data_prov, "models": results[tag]})
        fp = os.path.join(d, "fit.json")
        with open(fp, "w") as f:
            json.dump(fit, f, indent=1, default=L.json_default)
        sha = hashlib.sha256(open(fp, "rb").read()).hexdigest()
        cand = {"variant": tag, "window_ms": a.window_ms, "attribution": cfg["attribution"], "r_star": cfg["r_star"],
                "fit_json": os.path.abspath(fp), "fit_json_sha256": sha,
                "models": {m: {**x["publish"], "theta_ci90": x["theta_ci90"], "theta_bootfail": x["theta_bootfail"]}
                           for m, x in results[tag].items() if "publish" in x}}
        with open(os.path.join(d, "candidate.json"), "w") as f:
            json.dump(L.clean(cand), f, indent=1, default=L.json_default)
        rows = [{kk: (json.dumps(v) if isinstance(v, (list, dict)) else v) for kk, v in row.items()}
                for x in results[tag].values() for row in x.get("per_shape", [])]
        pd.DataFrame(rows).to_csv(os.path.join(d, "per_shape.csv"), index=False)
        trows = []
        for m, x in results[tag].items():
            for row in x.get("per_shape", []):
                trows.append({kk: row.get(kk) for kk in ("model", "shape", "class", "bracketed", "rho1", "tau_high_n2",
                                                         "tau_high_n3", "tau_high_n4", "in_ladder_n2", "in_ladder_n3",
                                                         "in_ladder_n4", "ladder_n_cells", "ladder_n_rho_levels",
                                                         "ladder_rho_min", "ladder_rho_max")})
        pd.DataFrame(trows).to_csv(os.path.join(d, "tau_high_by_shape.csv"), index=False)
        hdr = (f"window {a.window_ms} ms, attribution {cfg['attribution']}, lambda/w_p {cfg['lamwp']}"
               f"{'' if cfg['lamwp'] == 'formula' else ' %g/%g' % (cfg['lam'], cfg['wp'])}, r* {cfg['r_star']}, "
               f"shapes {cfg['shapes']}, nboot {a.nboot}, data {data}")
        lines = summary_lines(tag, results[tag], hdr)
        with open(os.path.join(d, "summary.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
    return results


if __name__ == "__main__":
    main()

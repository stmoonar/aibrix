#!/usr/bin/env python3
"""selfcheck.py - the fit tools must reproduce known numbers on the OLD training data (10 s).  CPU only, read-only.

 0. internals: vectorised TSS == recompute_tss_rows; kept windows / completion counts == refit_fixedslo.kept_windows
    (n1 / k1); window and request counts == the tune F fit set; the generalised ladder == rstar_F's ladder (run2/run2b).
 a. completion attribution, lambda 20, w_p .01, r* 5 %, all shapes  -> tune F (fit/rstarF/rstar_F.json):
    theta 404.94 / 363.58 / 443.93, tau_crit .65 / .52 / .48, tau_high 1.39 / 1.59 / 1.19 (14b at the live 439.4: 1.20),
    C_m 716.1 / 726.6 / 623.7.  Bootstrap = rstar_F's RNG stream (seed 20261029, models in order) -> its CIs exactly.
 b. first-token attribution, lambda 20, w_p .01, r* 1.5 % -> review §8 (A/followup2/grid_*.json): 414.6 / 364.5 / 421.6.
 c. lambda / w_p formula on the old training set (completion attribution = analysis C) -> C/lam_wp.json:
    lambda 32.19 / 31.12 / 32.46, w_p .009180 / .009253 / .009109; plus the first-token version (primary).
Usage: nice -n 19 python3 -B selfcheck.py --out DIR [--nboot 200]"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import supp_lib as L  # noqa: E402
from supp_lib import r  # noqa: E402
import refit_supp as RS  # noqa: E402

REF_A = {  # campaign-icse-tune-20261031/fit/rstarF/rstar_F.json, scan row r* = 5 %
    "dsqwen-7b": dict(theta=404.94, theta_ci90=[369.8, 462.2], bootfail=0.0, theta17=264.63,
                      theta17_ci90=[246.9, 283.3], tau_crit=0.65, C_m=716.1, q99=743.8629784208496,
                      q99D=727.0936894906267, n3_med=1.3932091138727671, cap=1.50, tau_high=1.39),
    "dsllama-8b": dict(theta=363.58, theta_ci90=[None, None], bootfail=0.235, theta17=190.55,
                       theta17_ci90=[176.2, 204.7], tau_crit=0.52, C_m=726.6, q99=702.6942115101092,
                       q99D=667.2671454553117, n3_med=1.592204743803711, cap=1.69, tau_high=1.59),
    "dsqwen-14b": dict(theta=443.93, theta_ci90=[394.1, 487.8], bootfail=0.0, theta17=211.44,
                       theta17_ci90=[189.1, 244.3], tau_crit=0.48, C_m=623.7, q99=678.4521985400661,
                       q99D=663.5595818315804, n3_med=1.5283897241272049, cap=1.19, tau_high=1.19)}
REF_A14_LIVE = dict(theta=439.4, n3_med=1.5440423271280517, tau_high=1.20, tau_crit=0.48)  # tune F 14b (r* 5.25 %)
REF_B = {  # analysis-train-slo-20261102/A/followup2/grid_<model>.json, cells all, label a, attribution ft
    "dsqwen-7b": dict(theta=414.6328711245861, theta17=239.91955304251547, n_req=354862),
    "dsllama-8b": dict(theta=364.4773015019552, theta17=179.8574623890031, n_req=295199),
    "dsqwen-14b": dict(theta=421.64007870586596, theta17=190.68425174234426, n_req=231242)}
REF_C = {  # analysis-train-slo-20261102/C/lam_wp.json, subset A (fit set, completion attribution, served requests)
    "dsqwen-7b": dict(n_windows=3644, n_req=360456, n_served=341147, L_in=726.2164374888245, L_out=214.62120141757072,
                      lam=32.1931802126356, w_p=0.00917999968400503),
    "dsllama-8b": dict(n_windows=3657, n_req=300459, n_served=284815, L_in=720.473654828573, L_out=207.49242490739604,
                       lam=31.123863736109406, w_p=0.009253172023691707),
    "dsqwen-14b": dict(n_windows=3586, n_req=237308, n_served=224911, L_in=731.8445118291236, L_out=216.42824495022475,
                       lam=32.464236742533714, w_p=0.00910940310258588)}


def chk(rows, model, item, mine, ref, tol, rel=False):
    if ref is None or mine is None or (isinstance(mine, float) and not np.isfinite(mine)):
        ok = (ref is None) == (mine is None or (isinstance(mine, float) and not np.isfinite(mine)))
        d = None
    else:
        d = float(mine) - float(ref)
        ok = abs(d) <= (tol * abs(float(ref)) if rel else tol)
    rows.append(dict(model=model, item=item, mine=mine, ref=ref, diff=d, ok=bool(ok)))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nboot", type=int, default=200)
    a = ap.parse_args()
    L.marker_check()
    L.configure(10000)
    os.makedirs(a.out, exist_ok=True)
    rows, extra = [], {}
    rng_a = np.random.default_rng(20261029)  # = refit_fixedslo.RNG at import; rstar_F drew 7b, 8b, 14b in this order
    for model in L.MODELS:
        L.log(f"{model}: loading (mine)")
        md = L.ModelData(model, 10000, L.resolve_specs(["old"], model, 10000), L.TRAIN_SPLITS, hash_files=False)
        # ---- 0. internals vs refit_fixedslo
        L.log(f"{model}: loading (refit_fixedslo reference path)")
        srcs = r.load_model(model)
        Wref = r.kept_windows(srcs, r.tss_all(srcs, 20.0, 0.01))
        t = L.tss_vec(md.Wall, 20.0, 0.01)
        L.assert_tss(md.Wall, 20.0, 0.01, t)
        K = L.kept(md.Wall, t, 10000)
        same_rows = (len(K) == len(Wref) and np.array_equal(K.key.values, Wref.key.values)
                     and np.array_equal(K.window_start_ms.values, Wref.window_start_ms.values))
        chk(rows, model, "0 kept windows == refit.kept_windows (rows, order)", float(same_rows), 1.0, 0)
        if same_rows:
            chk(rows, model, "0 TSS == refit.tss_all (max rel diff)", float(np.max(np.abs(K.tss.values / Wref.tss.values - 1))), 0.0, 1e-12)
        Cc = L.attr_counts(K, md.Q, "done", ["v", "served", "lin", "lout"])
        if same_rows:
            chk(rows, model, "0 completion n == refit n1 (all windows)", float(np.array_equal(Cc[:, 0], Wref.n1.values)), 1.0, 0)
            chk(rows, model, "0 completion k == refit k1 (all windows)", float(np.array_equal(Cc[:, 1], Wref.k1.values)), 1.0, 0)
        chk(rows, model, "0 n windows with requests (comp)", int((Cc[:, 0] > 0).sum()), REF_C[model]["n_windows"], 0)
        chk(rows, model, "0 n requests (completion)", int(Cc[:, 0].sum()), REF_C[model]["n_req"], 0)
        Cf = L.attr_counts(K, md.Q, "tft", ["v", "served", "lin", "lout"])
        chk(rows, model, "0 n requests (first token)", int(Cf[:, 0].sum()), REF_B[model]["n_req"], 0)
        # generalised ladder (sources run2/run2b, primitives hold/static, rho_factor rule) == rstar_F.ladder
        h = K[(K.src.isin(["run2", "run2b"])) & (K.primitive == "hold") & K.rho.notna() & K.rho_factor.notna()]
        cref = h.groupby(["shape", "key"]).agg(rho=("rho", "first"), tss=("tss", "median"), nw=("tss", "size")).reset_index()
        cref = cref[cref.nw >= 3]
        lad_ref = L.ladder(cref)
        cell, _ = L.ladder_cells(K, L.LADDER_SRC_OLD)
        lad = L.ladder(cell)
        same_lad = (set(lad) == set(lad_ref) and all(np.array_equal(lad[s][0], lad_ref[s][0])
                                                     and np.allclose(lad[s][1], lad_ref[s][1]) for s in lad))
        chk(rows, model, "0 ladder == rstar_F.ladder", float(same_lad), 1.0, 0)
        # ---- a. tune F at r* 5 % (completion, lambda 20, w_p .01), rstar_F's RNG stream
        L.log(f"{model}: check a")
        cfg_a = dict(attribution="completion", lamwp="fixed", lam=20.0, wp=0.01, r_star=0.05, shapes="all")
        ra = RS.fit_one(md, cfg_a, a.nboot, rng_a, list(L.LADDER_SRC_OLD))
        ref = REF_A[model]
        chk(rows, model, "a theta (r* 5 %)", round(ra["theta"], 2), ref["theta"], 0.005)
        ci = ra["theta_ci90"]
        chk(rows, model, "a theta CI90 lo", None if not np.isfinite(ci[0]) else round(ci[0], 1), ref["theta_ci90"][0], 0.05)
        chk(rows, model, "a theta CI90 hi", None if not np.isfinite(ci[1]) else round(ci[1], 1), ref["theta_ci90"][1], 0.05)
        chk(rows, model, "a theta bootfail", ra["theta_bootfail"], ref["bootfail"], 1e-9)
        chk(rows, model, "a theta17", round(ra["theta17"], 2), ref["theta17"], 0.005)
        c17 = ra["theta17_ci90"]
        chk(rows, model, "a theta17 CI90 lo", round(c17[0], 1), ref["theta17_ci90"][0], 0.05)
        chk(rows, model, "a theta17 CI90 hi", round(c17[1], 1), ref["theta17_ci90"][1], 0.05)
        chk(rows, model, "a tau_crit", ra["tau_crit"], ref["tau_crit"], 1e-9)
        chk(rows, model, "a C_m", ra["C_m"], ref["C_m"], 1e-9)
        chk(rows, model, "a TSS q99", ra["tss_q99"], ref["q99"], 1e-9, rel=True)
        chk(rows, model, "a TSS q99 (tune D basis)", ra["tss_q99_D"], ref["q99D"], 1e-9, rel=True)
        chk(rows, model, "a tau_high ladder n3 median", ra["tau_high_detail"]["ladder_median_n3"], ref["n3_med"], 1e-9, rel=True)
        chk(rows, model, "a tau_high cap", ra["tau_high_detail"]["cap"], ref["cap"], 1e-9)
        chk(rows, model, "a tau_high", ra["tau_high"], ref["tau_high"], 1e-9)
        extra[f"a_{model}"] = {kk: ra.get(kk) for kk in ("theta", "theta_ci90", "theta_bootfail", "theta17", "tau_crit",
                                                         "tau_high", "tau_high_detail", "C_m", "auroc_local")}
        if model == "dsqwen-14b":  # the live 14b theta (r* 5.25 %)
            th = L.tau_high_rule(lad, REF_A14_LIVE["theta"], ra["tss_q99"], ra["C_m"])
            chk(rows, model, "a tau_high at the live theta 439.4", th["value"], REF_A14_LIVE["tau_high"], 1e-9)
            chk(rows, model, "a ladder n3 median at 439.4", th["ladder_median_n3"], REF_A14_LIVE["n3_med"], 1e-9, rel=True)
            chk(rows, model, "a tau_crit at 439.4", round(ra["theta17"] / 439.4, 2), REF_A14_LIVE["tau_crit"], 1e-9)
        # ---- b. first token, lambda 20, w_p .01, r* 1.5 %
        L.log(f"{model}: check b")
        cfg_b = dict(attribution="first_token", lamwp="fixed", lam=20.0, wp=0.01, r_star=0.015, shapes="all")
        rb = RS.fit_one(md, cfg_b, a.nboot, np.random.default_rng(L.model_seed(model)), list(L.LADDER_SRC_OLD))
        chk(rows, model, "b theta (first token, r* 1.5 %)", rb["theta"], REF_B[model]["theta"], 1e-6, rel=True)
        chk(rows, model, "b theta17 (first token)", rb["theta17"], REF_B[model]["theta17"], 1e-6, rel=True)
        extra[f"b_{model}"] = {kk: rb.get(kk) for kk in ("theta", "theta_pub", "theta_ci90", "theta_bootfail",
                                                         "theta17", "tau_crit", "tau_high", "tau_high_detail",
                                                         "auroc_local", "auroc_global")}
        # ---- c. lambda / w_p formula (completion = analysis C; first token = primary)
        L_in, L_out, ns, lam, wp = L.lam_wp_from(Cc, model)
        rc = REF_C[model]
        chk(rows, model, "c n served (completion)", ns, rc["n_served"], 0)
        chk(rows, model, "c L_in (completion)", L_in, rc["L_in"], 1e-12, rel=True)
        chk(rows, model, "c L_out (completion)", L_out, rc["L_out"], 1e-12, rel=True)
        chk(rows, model, "c lambda = 0.15 L_out", lam, rc["lam"], 1e-12, rel=True)
        chk(rows, model, "c w_p = 6.667 / L_in", wp, rc["w_p"], 1e-12, rel=True)
        fL_in, fL_out, fns, flam, fwp = L.lam_wp_from(Cf, model)
        extra[f"c_{model}_first_token"] = dict(n_served=fns, L_in=fL_in, L_out=fL_out, lam=flam, w_p=fwp)
        L.log(f"{model}: done")
        del md, srcs, Wref
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "selfcheck.csv"), index=False)
    json.dump(L.clean({"checks": rows, "extra": extra}), open(os.path.join(a.out, "selfcheck.json"), "w"), indent=1,
              default=L.json_default)
    lines = [f"selfcheck: {int(R.ok.sum())}/{len(R)} ok (nboot {a.nboot})"]
    for _, x in R.iterrows():
        lines.append(f"{'ok ' if x.ok else 'MISMATCH'} {x.model:11s} {x['item']:48s} mine {x.mine!s:>22s}  ref {x.ref!s:>22s}"
                     + ("" if x['diff'] is None or (isinstance(x['diff'], float) and not np.isfinite(x['diff'])) else f"  diff {x['diff']:+.3g}"))
    for m in L.MODELS:
        e = extra.get(f"c_{m}_first_token")
        if e:
            lines.append(f"info {m:11s} c (first-token attribution, = primary): n_served {e['n_served']} L_in {e['L_in']:.2f} "
                         f"L_out {e['L_out']:.2f} lambda {e['lam']:.3f} w_p {e['w_p']:.6f}")
        b = extra.get(f"b_{m}")
        if b:
            lines.append(f"info {m:11s} b: theta_pub {b['theta_pub']} CI90 {b['theta_ci90']} bf {b['theta_bootfail']} "
                         f"tau_crit {b['tau_crit']} tau_high {b['tau_high']} (n3 med {b['tau_high_detail']['ladder_median_n3']}, "
                         f"cap {b['tau_high_detail']['cap']}) AUROC loc {b['auroc_local']:.4f} glob {b['auroc_global']:.4f}")
    open(os.path.join(a.out, "selfcheck.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

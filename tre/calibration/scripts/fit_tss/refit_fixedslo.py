#!/usr/bin/env python3
"""Offline refit of theta / tau_crit / tau_high (and a lambda x w_p grid) under the registry's SLO
(tre_common.slo; written for the FIXED SLO of 2026-10-29): a request violates if not served ok (error /
unserved / client timeout) OR client TTFT > TTFT_slo(L) OR TPOT = (e2e - ttft) / (out - 1) > TPOT_slo.
theta_m = the TSS at which the per-request violation probability reaches r* = 1 %.

SLO (2026-11-04): every threshold is the registry's, judged by tre_common.slo.violated_array (the one
definition). --slo-registry (default $TRE_FIT_SLO_REGISTRY, else this repo's deploy/registry.yaml) names the
registry or run snapshot, --slo-mode (default $TRE_FIT_SLO_MODE, else the registry's slo_mode) the mode; a
legacy registry whose two retired keys disagree needs --slo-mode. fit_summary.json records both (config.slo).
The 10-29 .. 11-03 fits used fixed 500 / 75 ms: --slo-mode fixed with any registry of that time (all carry
ttft_p95_ms 500 / tpot_p95_ms 75) gives the same labels, bit for bit.

Read-only on freeze-ctr20-20261028 datasets; TSS recomputed with the freeze build tree
(rel-e891d29b, the code the freeze ran); writes only to OUT.  CPU only.
2026-11-03 repo copy (tre/calibration/scripts/fit_tss/).  Changes, all of them:
  * paths are configuration: the data roots are named relative to TRE_FIT_DATA_ROOT, --out has no default;
  * recompute_tss_rows comes from this repo's tre/ tree instead of the build tree rel-e891d29b (= commit e891d29b).
    Its raw-TSS path (ema_tau_ms None: tss_terms, replica_factor) is unchanged since that commit; the EMA is not
    (b8bd62d1, 10-29: idle = nothing in flight, a zero TSS advances the EMA);
  * therefore raw TSS only: --ema-tau-ms defaults to 0 and a value > 0 is refused.  The 10-29 refit (EMA tau 10 s)
    is reproducible only with the original against the freeze tree (backup: <data root>/backups/fit-tools-20261103/).
The fit tools (supp_lib.configure) always pass --ema-tau-ms 0.
Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 python3 refit_fixedslo.py --no-stage2 \
       --ds-root DATASETS --window-ms 10000 --ema-tau-ms 0 --out OUTDIR
2026-10-31 copy (refit-fixedslo-w10-20261031): window / EMA tau / dataset root / output dir are CLI
parameters (defaults then = the 10-29 run: freeze datasets, 20 s, tau 10 s; here tau 0, see above).
--ema-tau-ms 0 = no EMA (raw TSS, ema_tau_ms=None in TssRecompute = the controller's tau <= 0 passthrough).
A guard asserts every kept window is exactly --window-ms long (the dataset must be built at that window).
Data loading / per-request label / PAVA crossing follow analysis-vz-by-shape-20261029/vz_by_shape.py.
"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

DATA_ROOT = os.environ.get("TRE_FIT_DATA_ROOT") or None
FREEZE = os.path.join(DATA_ROOT, "freeze-ctr20-20261028") if DATA_ROOT else None
DS = FREEZE + "/datasets" if FREEZE else None
TRE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # this repo's tre/
OUT = None  # main() needs --out (the 10-29 run wrote <data root>/refit-fixedslo-20261029)
WINDOW_MS = 20000.0
EMA_TAU_MS = None  # raw TSS only (the 10-31 default 10000 needs the freeze tree's EMA, see the header)
sys.path[:0] = [TRE + "/common", TRE + "/calibration"]
from tre_calibration.dataset import TssRecompute, recompute_tss_rows  # noqa: E402
from tre_common import slo as slo_def  # noqa: E402

SOURCES = {
    "dsqwen-7b": ["run1", "run2", "supp", "p1r2"],
    "dsllama-8b": ["run1", "run2", "supp", "p1"],
    "dsqwen-14b": ["run1", "run2b", "supp", "p1"],
}
TRAIN_SPLITS = ("train", "auxiliary")  # = the freeze fitting CSV (holdout = sealed H2/M)
SLO = None  # tre_common.slo.RegistrySlo of --slo-registry / --slo-mode (configure())
R_STAR, R_CRIT = 0.01, 0.10
STEP_MS = 10000.0
TRIM = 1  # dline_refit.TRIM_RAMP_WINDOWS: earliest kept window per scenario_id dropped
MIN_REQ_RAW = 10  # raw window-crossing: windows with >= 10 requests (as vz_by_shape)
N_BOOT = 500
N_BOOT_RAW = 200
H_LOCAL = 0.15  # local-logistic Gaussian kernel bandwidth in log TSS (~ +-15 %, the Z noise scale)
H_SENS = (0.10, 0.25)
RNG = np.random.default_rng(20261029)
LOGIT = lambda p: np.log(p / (1 - p))  # noqa: E731


# ---------------------------------------------------------------- data
def request_violated(q, model):
    """Per-request SLO verdict of a requests.csv frame of one model (tre_common.slo, the registry's thresholds):
    not served ok (outcome ok and http 200) | TTFT > TTFT_slo(L) | TPOT = (e2e - ttft) / (out - 1) > TPOT_slo."""
    th = SLO.of(model)
    ok = (q.outcome == "ok") & (q.http_status == 200)
    tt = q.ttft_ms
    out = q.output_tokens
    tpot = np.where(out > 1, (q.e2e_ms - tt) / (out - 1), np.nan)
    return ok, slo_def.violated_array(th, completed=ok, prompt_tokens=q.input_tokens if th.per_request_ttft else None,
                                      ttft_ms=tt, tpot_ms=tpot)


def per_request(q, model):
    _, v = request_violated(q, model)
    done = q.done_ts_ms.fillna(q.send_ts_ms + q.e2e_ms.fillna(q.request_timeout_s * 1000))
    return pd.DataFrame({"key": q.key.values, "v": v.astype(float), "done": done.values,
                         "send": q.send_ts_ms.values.astype(float)})


def count(win, req, ts, lo_col, hi_col):
    """n, k of requests with ts in (lo, hi] per window (same key)."""
    res = np.zeros((len(win), 2))
    rg = {k: g for k, g in req.groupby("key")}
    for key, wg in win.groupby("key"):
        g = rg.get(key)
        if g is None:
            continue
        g = g.sort_values(ts)
        t = g[ts].values
        cv = np.concatenate([[0], np.cumsum(g.v.values)])
        lo = np.searchsorted(t, wg[lo_col].values, side="right")
        hi = np.searchsorted(t, wg[hi_col].values, side="right")
        idx = win.index.get_indexer(wg.index)
        res[idx, 0] = hi - lo
        res[idx, 1] = cv[hi] - cv[lo]
    return res


def load_model(model):
    """All rows per source (for the EMA recompute) + per-request labels."""
    srcs = []
    for src in SOURCES[model]:
        p = f"{DS}/{src}/{model}/dataset_hybrid"
        w = pd.read_csv(p + "/windows.csv")
        q = pd.read_csv(p + "/requests.csv", usecols=[
            "cell_id", "attempt", "send_ts_ms", "first_token_ts_ms", "done_ts_ms", "ttft_ms", "e2e_ms",
            "output_tokens", "http_status", "outcome", "request_timeout_s"]
            + (["input_tokens"] if SLO.of(model).per_request_ttft else []))
        w["src"] = src
        w["key"] = src + "|" + w.cell_id + "|" + w.attempt.astype(str)
        q["key"] = src + "|" + q.cell_id + "|" + q.attempt.astype(str)
        srcs.append((src, w, w.to_dict("records"), per_request(q, model)))
    return srcs


def tss_all(srcs, lam, wp):
    out = []
    for _, w, rows, _ in srcs:
        vals = recompute_tss_rows(rows, TssRecompute(w_p=wp, lambda_wait=lam, qmin=1.0, ema_tau_ms=EMA_TAU_MS))
        out.append(np.array([np.nan if v is None else v for v in vals], float))
    return out


def kept_windows(srcs, tss_list):
    """Freeze window filter: split train/auxiliary, valid, not in_warmup (run1 has no flag = kept),
    not possibly_contaminated, finite TSS, tokens > 0; then trim 1 earliest window per scenario_id."""
    parts = []
    for (src, w, _, req), tss in zip(srcs, tss_list):
        w = w.copy()
        w["tss"] = tss
        m = (w.split.isin(TRAIN_SPLITS) & (w.cell_status == "valid") & (w.in_warmup != True)  # noqa: E712
             & (w.possibly_contaminated != True) & np.isfinite(w.tss) & (w.tss > 0)  # noqa: E712
             & ((w.prompt_tokens_total.fillna(0) + w.generation_tokens_total.fillna(0)) > 0))
        w = w[m].reset_index(drop=True)
        dur = (w.window_end_ms - w.window_start_ms).unique()
        assert len(dur) <= 1 and all(abs(d - WINDOW_MS) < 1e-6 for d in dur), (src, dur[:5], WINDOW_MS)
        w["lo1"] = w.window_end_ms - STEP_MS
        c1 = count(w, req, "done", "lo1", "window_end_ms")  # single assignment (primary)
        c2 = count(w, req, "done", "window_start_ms", "window_end_ms")  # full 20 s (raw crossing)
        c3 = count(w, req, "send", "lo1", "window_end_ms")  # sensitivity: send attribution
        w["n1"], w["k1"] = c1[:, 0], c1[:, 1]
        w["n2"], w["k2"] = c2[:, 0], c2[:, 1]
        w["n3"], w["k3"] = c3[:, 0], c3[:, 1]
        parts.append(w)
    W = pd.concat(parts, ignore_index=True)
    W = W.sort_values(["scenario_id", "window_start_ms"], kind="stable")
    first = W.groupby("scenario_id").cumcount() < TRIM
    W = W[~first].sort_index().reset_index(drop=True)
    return W[["src", "key", "shape", "primitive", "stage", "rho", "rho_factor", "scenario_id", "slo_label",
              "window_start_ms", "window_end_ms", "tss", "n1", "k1", "n2", "k2", "n3", "k3"]]


# ---------------------------------------------------------------- logistic (binomial GLM on log TSS)
def glm(x, k, n, wt=None, it=50):
    m = n > 0
    x, k, n = x[m], k[m], n[m]
    wt = np.ones_like(n) if wt is None else wt[m]
    X = np.column_stack([np.ones_like(x), x])
    p0 = np.clip(k.sum() / n.sum(), 1e-4, 0.5)
    beta = np.array([LOGIT(p0), 0.0])
    for _ in range(it):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        var = np.maximum(mu * (1 - mu), 1e-12)
        W = wt * n * var
        z = eta + (k - n * mu) / (n * var)
        new = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * z))
        if np.max(np.abs(new - beta)) < 1e-9:
            beta = new
            break
        beta = new
    return beta


def tss_at(beta, r):
    a, b = beta
    return float(np.exp((LOGIT(r) - a) / b)) if b < 0 else np.nan


def deviance(beta, x, k, n):
    m = n > 0
    x, k, n = x[m], k[m], n[m]
    mu = np.clip(1 / (1 + np.exp(-(beta[0] + beta[1] * x))), 1e-12, 1 - 1e-12)
    return -2 * np.sum(k * np.log(mu) + (n - k) * np.log(1 - mu))


def auroc(score_bad, k, n):
    """AUROC of a score (higher = more likely violating) vs per-request labels, window-level counts."""
    df = pd.DataFrame({"s": score_bad, "pos": k, "neg": n - k}).groupby("s").sum().sort_index()
    pos, neg = df.pos.values, df.neg.values
    neg_below = np.concatenate([[0], np.cumsum(neg)[:-1]])
    num = np.sum(pos * (neg_below + 0.5 * neg))
    return float(num / (pos.sum() * neg.sum()))


def cluster_weights(keys):
    codes, uniq = pd.factorize(keys)
    pick = RNG.integers(0, len(uniq), size=len(uniq))
    return np.bincount(pick, minlength=len(uniq))[codes].astype(float)


# ---------------------------------------------------------------- PAVA crossing (vz_by_shape)
def antitonic(z, v, w=None):
    o = np.argsort(z, kind="stable")
    zs, y = z[o], -v[o]
    ww = np.ones_like(y) if w is None else w[o]
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, ww):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            tot = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / tot if tot > 0 else v1); wts.append(tot); cnt.append(c1 + c2)
    return zs, -np.repeat(vals, cnt)


def crossing(z, v, rs, w=None):
    """largest z where the non-increasing fit is still >= r; -inf never, +inf floor."""
    if len(z) < 5:
        return [np.nan] * len(rs)
    zs, f = antitonic(z, v, w)
    out = []
    for r in rs:
        above = np.nonzero(f >= r)[0]
        if len(above) == 0:
            out.append(-np.inf)
        elif above[-1] == len(zs) - 1:
            out.append(np.inf)
        else:
            i = above[-1]
            out.append(0.5 * (zs[i] + zs[i + 1]))
    return out


def boot_ci(vals, n_boot):
    vals = np.asarray(vals, float)
    fin = vals[np.isfinite(vals)]
    if len(fin) < 0.8 * n_boot:
        return np.nan, np.nan, float(np.mean(~np.isfinite(vals)))
    lo, hi = np.percentile(fin, [5, 95])
    return lo, hi, float(np.mean(~np.isfinite(vals)))


def wilson(k, n, z=1.645):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ---------------------------------------------------------------- tau_high proposal
def tau_high_rows(model, W, theta):
    """Hold cells of the 2nd-round ladder (run2 / run2b, rho_factor set): per shape, cell-median TSS
    vs offered load rho; non-increasing interpolation in log-log.  rho1 = load where TSS = theta.
    Losing 1 of n replicas multiplies per-replica load by n/(n-1), so the model stays at Z >= 1
    iff the current load <= rho1 (n-1)/n; tau_high(n) = Z(rho1 (n-1)/n)."""
    rows = []
    h = W[(W.src.isin(["run2", "run2b"])) & (W.primitive == "hold") & W.rho.notna() & W.rho_factor.notna()]
    cell = h.groupby(["shape", "key"]).agg(rho=("rho", "first"), tss=("tss", "median"), nw=("tss", "size")).reset_index()
    cell = cell[cell.nw >= 3]
    for shape, g in cell.groupby("shape"):
        lv = g.groupby("rho").tss.apply(lambda s: np.exp(np.log(s).mean())).sort_index()
        lr, lt = np.log(lv.index.values), np.log(lv.values)
        _, fit = antitonic(lr, lt)  # non-increasing log TSS in log rho
        lz = fit - np.log(theta)
        row = {"model": model, "shape": shape, "n_cells": len(g), "n_rho_levels": len(lv),
               "rho_min": float(lv.index.min()), "rho_max": float(lv.index.max())}
        if not (lz.max() >= 0 >= lz.min()):
            row.update({"rho_at_Z1": np.nan, "note": "Z=1 not bracketed by hold ladder"})
            rows.append(row)
            continue
        # rho where lz crosses 0 (last level with lz >= 0 and the next one)
        i = np.nonzero(lz >= 0)[0][-1]
        if i == len(lz) - 1:
            l1 = lr[i]
        else:
            l1 = lr[i] + (lr[i + 1] - lr[i]) * (lz[i] - 0) / max(lz[i] - lz[i + 1], 1e-12)
        row["rho_at_Z1"] = float(np.exp(l1))
        for n in (2, 3, 4):
            lt_ = l1 + np.log((n - 1) / n)
            if lt_ >= lr[0]:
                z = np.exp(np.interp(lt_, lr, lz))
                ext = False
            else:  # below the ladder: extrapolate with the slope of the two lowest levels
                s = (lz[1] - lz[0]) / (lr[1] - lr[0]) if len(lr) > 1 else 0.0
                z = np.exp(lz[0] + s * (lt_ - lr[0]))
                ext = True
            row[f"tau_high_n{n}"] = float(z)
            row[f"extrap_n{n}"] = ext
        row["note"] = ""
        rows.append(row)
    return rows


# ---------------------------------------------------------------- main

# ---------------------------------------------------------------- local logistic (primary estimator)
def local_root(x, k, n, r, h=H_LOCAL, wt=None, x0=None, it=60):
    """log TSS where the per-request violation probability equals r, from a kernel-weighted
    binomial GLM on log TSS centred on the answer itself (local likelihood, fixed point):
    weights exp(-0.5 ((x - x0)/h)^2); x0 <- (logit r - a)/b until it stops moving.
    The global 2-parameter logistic is misspecified in the tail (heavier than logistic), so
    the curve is fitted where it is read.  Returns (x_root, local slope b) or (nan, nan)."""
    base = np.ones_like(x) if wt is None else wt
    if x0 is None:
        b = glm(x, k, n, base)
        if not b[1] < 0:
            return np.nan, np.nan
        x0 = (LOGIT(r) - b[0]) / b[1]
    for _ in range(it):
        w = base * np.exp(-0.5 * ((x - x0) / h) ** 2)
        if (w * n).sum() < 200:
            return np.nan, np.nan
        b = glm(x, k, n, w)
        if not b[1] < 0:
            return np.nan, np.nan
        x1 = (LOGIT(r) - b[0]) / b[1]
        x1 = np.clip(x1, x0 - 2 * h, x0 + 2 * h)  # damped step
        if abs(x1 - x0) < 1e-5:
            return float(x1), float(b[1])
        x0 = 0.5 * (x0 + x1)
    return float(x0), float(b[1])


def local_curve(x, k, n, grid, h=H_LOCAL):
    """p-hat(x0) of the local logistic at each grid point (for the plot)."""
    out = []
    for x0 in grid:
        w = np.exp(-0.5 * ((x - x0) / h) ** 2)
        if (w * n).sum() < 200:
            out.append(np.nan)
            continue
        b = glm(x, k, n, w)
        out.append(1 / (1 + np.exp(-(b[0] + b[1] * x0))))
    return np.array(out)


# ---------------------------------------------------------------- main
def fit_model(model, W, theta_old):
    x = np.log(W.tss.values)
    k, n = W.k1.values, W.n1.values
    beta = glm(x, k, n)  # global logistic (reported, misspecified in the tail)
    x1, s1 = local_root(x, k, n, R_STAR)
    x10, s10 = local_root(x, k, n, R_CRIT)
    th, th10 = np.exp(x1), np.exp(x10)
    boots = []
    for _ in range(N_BOOT):
        wt = cluster_weights(W.key.values)
        a1, _ = local_root(x, k, n, R_STAR, wt=wt, x0=x1)
        a10, _ = local_root(x, k, n, R_CRIT, wt=wt, x0=x10)
        bg = glm(x, k, n, wt)
        boots.append((np.exp(a1), np.exp(a10 - a1), tss_at(bg, R_STAR)))
    boots = np.array(boots)
    sens = {}
    for h in H_SENS:
        a1, _ = local_root(x, k, n, R_STAR, h=h)
        a10, _ = local_root(x, k, n, R_CRIT, h=h)
        sens[f"theta_local_h{h}"] = np.exp(a1)
        sens[f"tau_crit_local_h{h}"] = np.exp(a10 - a1)
    # raw window crossing (vz_by_shape method, fixed SLO, 20 s done attribution, n >= 10)
    R = W[W.n2 >= MIN_REQ_RAW]
    zr = R.tss.values
    vr = R.k2.values / R.n2.values
    raw = crossing(zr, vr, (R_STAR, R_CRIT))
    codes, uniq = pd.factorize(R.key)
    mem = [np.nonzero(codes == j)[0] for j in range(len(uniq))]
    rb = []
    for _ in range(N_BOOT_RAW):
        idx = np.concatenate([mem[j] for j in RNG.integers(0, len(uniq), size=len(uniq))])
        rb.append(crossing(zr[idx], vr[idx], (R_STAR, R_CRIT)))
    rb = np.array(rb)
    # request-weighted PAVA (pooled, per-request monotone MLE) with cell bootstrap
    vv = np.where(n > 0, k / np.maximum(n, 1), 0)
    pav = crossing(x, vv, (R_STAR, R_CRIT), w=n)
    codes, uniq = pd.factorize(W.key)
    mem = [np.nonzero(codes == j)[0] for j in range(len(uniq))]
    pb = []
    for _ in range(N_BOOT_RAW):
        idx = np.concatenate([mem[j] for j in RNG.integers(0, len(uniq), size=len(uniq))])
        pb.append(crossing(x[idx], vv[idx], (R_STAR,), w=n[idx])[0])
    plo, phi, _ = boot_ci(np.exp(np.array(pb)), N_BOOT_RAW)
    # attribution sensitivity: send-time attribution, same local estimator
    a3, _ = local_root(x, W.k3.values, W.n3.values, R_STAR)
    lo, hi, nf = boot_ci(boots[:, 0], N_BOOT)
    tclo, tchi, _ = boot_ci(boots[:, 1], N_BOOT)
    glo, ghi, _ = boot_ci(boots[:, 2], N_BOOT)
    rlo, rhi, rnf = boot_ci(rb[:, 0], N_BOOT_RAW)
    r10lo, r10hi, _ = boot_ci(rb[:, 1], N_BOOT_RAW)
    # calibration of the global logistic at the new theta (shows the misfit)
    near = np.abs(x - x1) < 0.1
    obs_near = k[near].sum() / max(n[near].sum(), 1)
    glob_near = float(np.sum(n[near] / (1 + np.exp(-(beta[0] + beta[1] * x[near])))) / max(n[near].sum(), 1))
    return beta, {
        "model": model, "theta_old": theta_old,
        "theta_new": th, "theta_ci90_lo": lo, "theta_ci90_hi": hi, "theta_boot_fail_frac": nf,
        "theta_new_over_old": th / theta_old, "local_slope_at_theta": s1,
        "tau_crit_Z10": th10 / th, "tau_crit_ci90_lo": tclo, "tau_crit_ci90_hi": tchi,
        "tau_crit_dist_from_1": 1 - th10 / th,
        **sens,
        "theta_global_logit": tss_at(beta, R_STAR), "theta_global_logit_ci90_lo": glo,
        "theta_global_logit_ci90_hi": ghi, "tau_crit_global_logit": tss_at(beta, R_CRIT) / tss_at(beta, R_STAR),
        "logit_a": beta[0], "logit_b_per_logTSS": beta[1],
        "p_obs_within10pct_of_theta": obs_near, "p_global_logit_within10pct_of_theta": glob_near,
        "theta_raw_cross1": raw[0], "theta_raw_cross1_ci90_lo": rlo, "theta_raw_cross1_ci90_hi": rhi,
        "raw_cross1_boot_nonfinite": rnf,
        "tss_raw_cross10": raw[1], "tss_raw_cross10_ci90_lo": r10lo, "tss_raw_cross10_ci90_hi": r10hi,
        "theta_pava_req": float(np.exp(pav[0])) if np.isfinite(pav[0]) else pav[0],
        "theta_pava_req_ci90_lo": plo, "theta_pava_req_ci90_hi": phi,
        "tss10_pava_req": float(np.exp(pav[1])) if np.isfinite(pav[1]) else pav[1],
        "theta_send_attr": float(np.exp(a3)),
        "n_requests": int(n.sum()), "n_viol": int(k.sum()), "viol_rate": k.sum() / n.sum(),
        "n_windows": int((n > 0).sum()), "n_cells": int(W.key.nunique()),
        "n_req_Z_0.7_1.5": int(n[(x > x1 + np.log(0.7)) & (x < x1 + np.log(1.5))].sum()),
        "auroc": auroc(-W.tss.values, k, n), "deviance_global_logit": deviance(beta, x, k, n),
    }


def per_shape(model, W, theta):
    rows = []
    for shape, g in W.groupby("shape"):
        xs, k, n = np.log(g.tss.values), g.k1.values, g.n1.values
        l1, _ = local_root(xs, k, n, R_STAR, h=0.2)
        l10, _ = local_root(xs, k, n, R_CRIT, h=0.2)
        pav = crossing(xs, np.where(n > 0, k / np.maximum(n, 1), 0), (R_STAR, R_CRIT), w=n)
        z = g.tss.values / theta
        row = {"model": model, "shape": shape, "n_cells": g.key.nunique(), "n_windows": int((n > 0).sum()),
               "n_requests": int(n.sum()), "viol_rate": k.sum() / max(n.sum(), 1),
               "Z1_local": np.exp(l1) / theta, "Z10_local": np.exp(l10) / theta,
               "Z1_pava": np.exp(pav[0]) / theta if np.isfinite(pav[0]) else pav[0],
               "Z10_pava": np.exp(pav[1]) / theta if np.isfinite(pav[1]) else pav[1],
               "Zmax": float(z.max()), "n_req_Zge1": int(n[z >= 1].sum())}
        for zc in (1.0, 1.25, 1.5, 2.0):
            m = z >= zc
            kk, nn = k[m].sum(), n[m].sum()
            lo, hi = wilson(kk, nn)
            row[f"p_Zge{zc}"] = kk / nn if nn else np.nan
            row[f"p_Zge{zc}_lo90"] = lo
            row[f"p_Zge{zc}_hi90"] = hi
            row[f"n_Zge{zc}"] = int(nn)
        # floor: violation rate at high Z still significantly > r*;  late: shape crosses 1 % above Z = 1.25
        row["floor_flag"] = bool(row["n_Zge1.5"] > 0 and row["p_Zge1.5_lo90"] > R_STAR)
        row["late_flag"] = bool(np.nanmax([row["Z1_local"], row["Z1_pava"]]) > 1.25)
        row["sparse_flag"] = bool(row["n_req_Zge1"] < 2000)
        rows.append(row)
    return rows


def binned(model, W, theta):
    rows = []
    edges = np.exp(np.arange(np.log(0.05), np.log(8.0) + 1e-9, np.log(8.0 / 0.05) / 28))
    for shape in ["ALL"] + sorted(W["shape"].unique()):
        g = W if shape == "ALL" else W[W["shape"] == shape]
        z = g.tss.values / theta
        b = np.digitize(z, edges)
        for bi in np.unique(b):
            m = b == bi
            kk, nn = g.k1.values[m].sum(), g.n1.values[m].sum()
            if nn < 50:
                continue
            lo, hi = wilson(kk, nn)
            rows.append({"model": model, "shape": shape, "Z_mid": float(np.exp(np.log(z[m]).mean())),
                         "n_req": int(nn), "n_windows": int(m.sum()), "n_cells": int(g.key.values[m].size and
                                                                                         pd.Series(g.key.values[m]).nunique()),
                         "p": kk / nn, "p_ci90_lo": lo, "p_ci90_hi": hi})
    return rows


def stage2(model, srcs, lam0, wp0):
    lams = sorted(set(np.round(np.arange(max(0.5, lam0 - 1.0), lam0 + 1.0001, 0.25), 4)) | {lam0})
    wps = sorted(set([0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]) | {wp0})
    rows, cache = [], {}
    for lam in lams:
        for wp in wps:
            W = kept_windows(srcs, tss_all(srcs, lam, wp))
            x, k, n = np.log(W.tss.values), W.k1.values, W.n1.values
            b = glm(x, k, n)
            l1, _ = local_root(x, k, n, R_STAR)
            l10, _ = local_root(x, k, n, R_CRIT)
            r = {"model": model, "lambda": lam, "w_p": wp, "deviance_global_logit": deviance(b, x, k, n),
                 "auroc": auroc(-W.tss.values, k, n), "theta_local": np.exp(l1), "tau_crit_local": np.exp(l10 - l1),
                 "theta_global_logit": tss_at(b, R_STAR), "n_requests": int(n.sum()), "n_windows": int((n > 0).sum()),
                 "is_freeze": bool(abs(lam - lam0) < 1e-9 and abs(wp - wp0) < 1e-9)}
            rows.append(r)
            cache[(lam, wp)] = W
            print(f"  [{model}] lam {lam} wp {wp} dev {r['deviance_global_logit']:.1f} auc {r['auroc']:.4f} "
                  f"theta {r['theta_local']:.1f}", file=sys.stderr, flush=True)
    G = pd.DataFrame(rows)
    f0 = G[G.is_freeze].iloc[0]
    G["d_dev_vs_freeze"] = G.deviance_global_logit - f0.deviance_global_logit
    G["d_auroc_vs_freeze"] = G.auroc - f0.auroc
    best = G.loc[G.deviance_global_logit.idxmin()]
    besta = G.loc[G.auroc.idxmax()]
    # paired cluster bootstrap: deviance(best) - deviance(freeze) and AUROC difference, params refit per resample
    key = ["src", "key", "window_start_ms"]
    Wf, Wb = cache[(lam0, wp0)], cache[(best["lambda"], best["w_p"])]
    J = Wf.merge(Wb[key + ["tss"]], on=key, suffixes=("_f", "_b"))
    xf, xb, k, n = np.log(J.tss_f.values), np.log(J.tss_b.values), J.k1.values, J.n1.values
    dd, da = [], []

    def dev_w(bb_, xx, wt):
        mu = np.clip(1 / (1 + np.exp(-(bb_[0] + bb_[1] * xx))), 1e-12, 1 - 1e-12)
        return -2 * np.sum(wt * (k * np.log(mu) + (n - k) * np.log(1 - mu)))

    codes, uniq = pd.factorize(J.key)
    for _ in range(200):
        wt = cluster_weights(J.key.values)
        dd.append(dev_w(glm(xb, k, n, wt), xb, wt) - dev_w(glm(xf, k, n, wt), xf, wt))
        sel = wt > 0
        rep = wt[sel].astype(int)
        da.append(auroc(-np.repeat(J.tss_b.values[sel], rep), np.repeat(k[sel], rep), np.repeat(n[sel], rep))
                  - auroc(-np.repeat(J.tss_f.values[sel], rep), np.repeat(k[sel], rep), np.repeat(n[sel], rep)))
    summary = {"model": model, "freeze_lambda": lam0, "freeze_w_p": wp0,
               "grid_lambda": [float(v) for v in lams], "grid_w_p": [float(v) for v in wps],
               "best_dev_lambda": float(best["lambda"]), "best_dev_w_p": float(best["w_p"]),
               "d_dev_best_minus_freeze": float(best.d_dev_vs_freeze),
               "d_dev_best_minus_freeze_ci90": [float(np.percentile(dd, 5)), float(np.percentile(dd, 95))],
               "d_auroc_best_minus_freeze_ci90": [float(np.percentile(da, 5)), float(np.percentile(da, 95))],
               "freeze_auroc": float(f0.auroc), "best_dev_auroc": float(best.auroc),
               "best_auroc_lambda": float(besta["lambda"]), "best_auroc_w_p": float(besta["w_p"]),
               "best_auroc": float(besta.auroc),
               "freeze_theta_local": float(f0.theta_local), "best_dev_theta_local": float(best.theta_local),
               "freeze_tau_crit": float(f0.tau_crit_local), "best_dev_tau_crit": float(best.tau_crit_local),
               "n_aligned_windows": int(len(J))}
    return G, summary


def plot(curves, fits, shapes, locs, betas, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = pd.DataFrame(curves)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, m in zip(axes, SOURCES):
        f = fits[m]
        th = f["theta_new"]
        cm = C[C.model == m]
        shs = sorted(s for s in cm["shape"].unique() if s != "ALL")
        flags = {r["shape"]: r for r in shapes[m]}
        for i, s in enumerate(shs):
            g = cm[cm["shape"] == s]
            tag = (" floor" if flags[s]["floor_flag"] else "") + (" late" if flags[s]["late_flag"] else "")
            ax.plot(g.Z_mid, np.maximum(g.p, 3e-4), "o-", ms=3, lw=0.8, alpha=0.7, color=cmap(i), label=s + tag)
        g = cm[cm["shape"] == "ALL"]
        ax.errorbar(g.Z_mid, np.maximum(g.p, 3e-4),
                    yerr=[np.clip(g.p - g.p_ci90_lo, 0, None), np.clip(g.p_ci90_hi - g.p, 0, None)],
                    fmt="s", color="k", ms=5, label="pooled (90% Wilson)")
        zz, pl = locs[m]
        ax.plot(zz, pl, "k-", lw=2.2, label="local logistic (h=0.15)")
        a, b = betas[m]
        ax.plot(zz, 1 / (1 + np.exp(-(a + b * np.log(zz * th)))), "k--", lw=1, label="global logistic")
        ax.axhline(R_STAR, color="r", ls="--", lw=1)
        ax.axhline(R_CRIT, color="orange", ls="--", lw=1)
        ax.axvline(1.0, color="r", ls=":", lw=1.2)
        ax.axvspan(f["theta_ci90_lo"] / th, f["theta_ci90_hi"] / th, color="r", alpha=0.08)
        ax.axvline(f["tau_crit_Z10"], color="orange", ls=":", lw=1.2)
        ax.axvline(f["theta_old"] / th, color="gray", ls="-.", lw=1, label="old theta")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(3e-4, 1.05); ax.set_xlim(0.2, 3)
        ax.set_title(f"{m}: theta {th:.0f} [{f['theta_ci90_lo']:.0f},{f['theta_ci90_hi']:.0f}] (old {f['theta_old']:.0f})"
                     f"\ntau_crit {f['tau_crit_Z10']:.2f}  (lambda {f['lambda']}, w_p {f['w_p']})", fontsize=10)
        ax.set_xlabel("Z = TSS / theta_new"); ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel(f"per-request violation probability (SLO {SLO.mode}, fail)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def repro_check(model, W, fz):
    """Is the recomputed TSS the freeze's signal?  The freeze theta sits at a known healthy-window
    quantile (plateau entry closest to the published theta, under the freeze's own label slo_label);
    compare with the same quantile on our recomputed TSS."""
    inside = fz["theta_selection"]["plateau"]["inside"]
    th = fz["published"]["theta"]
    e = min(inside, key=lambda d: abs(d["theta"] - th))
    h = W[W.slo_label == "healthy"]
    return {"freeze_theta": th, "plateau_entry_theta": e["theta"], "freeze_healthy_quantile": e["healthy_quantile"],
            "recomputed_healthy_quantile_at_entry": float((h.tss < e["theta"]).mean()), "n_healthy": int(len(h))}


def load_slo(path, mode):
    """The registry's SLO (tre_common.slo.load_registry_slo); refuses (SystemExit) a registry it cannot build."""
    try:
        return slo_def.load_registry_slo(path, mode=mode or None)
    except (OSError, slo_def.SloConfigError) as exc:
        raise SystemExit(f"--slo-registry {path} / --slo-mode {mode}: {exc}") from None


def configure(argv=None):
    global DS, OUT, WINDOW_MS, EMA_TAU_MS, SLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-stage2", action="store_true")
    ap.add_argument("--ds-root", default=DS)
    ap.add_argument("--window-ms", type=float, default=WINDOW_MS)
    ap.add_argument("--ema-tau-ms", type=float, default=0.0, help="0 only (raw TSS); > 0 is refused")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--slo-registry", default=os.environ.get("TRE_FIT_SLO_REGISTRY") or TRE + "/deploy/registry.yaml",
                    help="registry / run snapshot whose models[].slo judges the requests (default $TRE_FIT_SLO_REGISTRY, "
                         "else this repo's deploy/registry.yaml)")
    ap.add_argument("--slo-mode", default=os.environ.get("TRE_FIT_SLO_MODE") or None, choices=slo_def.SLO_MODES,
                    help="default $TRE_FIT_SLO_MODE, else the registry's slo_mode")
    args = ap.parse_args(argv)
    DS, OUT, WINDOW_MS = args.ds_root, args.out, args.window_ms
    SLO = load_slo(args.slo_registry, args.slo_mode)
    if args.ema_tau_ms and args.ema_tau_ms > 0:
        raise SystemExit(f"--ema-tau-ms {args.ema_tau_ms:g}: this repo copy computes raw TSS only - the EMA of the "
                         "repo's recompute_tss_rows changed after the freeze tree e891d29b (b8bd62d1), so an EMA refit "
                         "would not be the 10-29 one; use the original refit_fixedslo.py with the freeze tree")
    EMA_TAU_MS = None
    print(f"config: ds={DS} window_ms={WINDOW_MS} ema_tau_ms={EMA_TAU_MS} out={OUT} slo={SLO.mode} "
          f"({SLO.source}, {SLO.path} sha256 {SLO.sha256[:12]})", file=sys.stderr, flush=True)
    return args


def main():
    args = configure()
    if not OUT or not FREEZE:
        raise SystemExit("refit_fixedslo.py needs --out and TRE_FIT_DATA_ROOT (the freeze parameters)")
    os.makedirs(OUT, exist_ok=True)
    fz = json.load(open(FREEZE + "/freeze/params_freeze.json"))["models"]
    fits, shapes, curves, th_rows, betas, locs, s2rows, s2sum, repro = {}, {}, [], [], {}, {}, [], [], {}
    allW = []
    for model in SOURCES:
        pub = fz[model]["published"]
        lam0, wp0, th_old = pub["lambda_wait"], pub["w_p"], pub["theta"]
        srcs = load_model(model)
        W = kept_windows(srcs, tss_all(srcs, lam0, wp0))
        repro[model] = repro_check(model, W, fz[model])
        W["model"] = model
        allW.append(W)
        beta, f = fit_model(model, W, th_old)
        f.update({"lambda": lam0, "w_p": wp0, "tau_crit_old": pub["tau_crit"], "tau_high_old": pub["tau_high"]})
        fits[model], betas[model] = f, beta
        th = f["theta_new"]
        shapes[model] = per_shape(model, W, th)
        fl = [r["shape"] for r in shapes[model] if r["floor_flag"] or r["late_flag"]]
        Wn = W[~W["shape"].isin(fl)]
        xn = np.log(Wn.tss.values)
        a1, _ = local_root(xn, Wn.k1.values, Wn.n1.values, R_STAR)
        a10, _ = local_root(xn, Wn.k1.values, Wn.n1.values, R_CRIT)
        f["floor_or_late_shapes"] = ",".join(fl)
        f["theta_excl_floor_late"] = float(np.exp(a1))
        f["tau_crit_excl_floor_late"] = float(np.exp(a10 - a1))
        x = np.log(W.tss.values)
        zz = np.exp(np.linspace(np.log(0.2), np.log(3.0), 60))
        locs[model] = (zz, local_curve(x, W.k1.values, W.n1.values, np.log(zz * th)))
        curves += binned(model, W, th)
        tr = tau_high_rows(model, W, th)
        th_rows += tr
        T = pd.DataFrame(tr)
        for nrep in (2, 3, 4):
            c = f"tau_high_n{nrep}"
            if c in T and T[c].notna().any():
                f[f"tau_high_n{nrep}_max_shape"] = float(T[c].max())
                f[f"tau_high_n{nrep}_median_shape"] = float(T[c].median())
                f[f"tau_high_n{nrep}_extrap_shapes"] = ",".join(T.loc[T[f"extrap_n{nrep}"].astype(object).eq(True), "shape"])
        print(model, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()}),
              file=sys.stderr, flush=True)
        if not args.no_stage2:
            G, s = stage2(model, srcs, lam0, wp0)
            s2rows.append(G)
            s2sum.append(s)
    pd.DataFrame(fits.values()).to_csv(OUT + "/fit_summary.csv", index=False)
    pd.DataFrame([r for m in shapes for r in shapes[m]]).to_csv(OUT + "/per_shape.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT + "/binned_curves.csv", index=False)
    pd.DataFrame(th_rows).to_csv(OUT + "/tau_high_by_shape.csv", index=False)
    pd.concat(allW).to_csv(OUT + "/windows_used.csv.gz", index=False)
    if s2rows:
        pd.concat(s2rows).to_csv(OUT + "/stage2_grid.csv", index=False)
        json.dump(s2sum, open(OUT + "/stage2_summary.json", "w"), indent=1)
    json.dump({"config": {"ds_root": DS, "window_ms": WINDOW_MS, "ema_tau_ms": EMA_TAU_MS, "tre": TRE,
                          "slo": SLO.as_record()},
               "tss_repro_check": repro, "fits": fits}, open(OUT + "/fit_summary.json", "w"), indent=1, default=float)
    plot(curves, fits, shapes, locs, betas, OUT + "/p_vs_Z.png")


if __name__ == "__main__":
    main()

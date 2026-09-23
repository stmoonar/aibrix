#!/usr/bin/env python3
"""The D-line offline refit (plan 2026-09-21 §6.11 step 4): alpha, w_p, final.

Until 2026-09-23 this pipeline existed only as ``/tmp/refit_final_1790081928/refit.py``
(archived under ``archive_tmp_20260922/refit_final_1790081928``): every number the D-line
decisions rest on - the chosen EMA alpha, the D3 w_p, theta / delta and the M hold-out
acceptance, for the primary (D6') and the fixed label - came out of a script outside the
repository with hard-coded paths. This module is that script, formalised: same stages,
same rules, same seeds; paths and the per-model inputs are arguments.

Inputs are the re-windowed CSVs of a fit plan (``rewindow_from_raw --window-align grid
--step-ms 10000``, i.e. ``calibration_campaign.fit_plan``'s ``rewindow`` step) in one
directory, ``--fit-dir``: ``<model>_fitting.csv``, ``<model>_validation.csv`` (the
held-out set), ``<model>_fitting_decode_heavy.csv``, ``<model>_fitting_prefill_heavy.csv``.
Outputs go to ``--out-dir/<model>/<arm>/<stage>.json``; each stage reads the previous
one's output from there.

Stages (``python -m scripts.dline_refit STAGE --model M --arm primary|fixed|k3 ...``):

``alpha`` (D4')
    the TSS EMA time constant. ``--alpha-rule d4prime`` (default) is
    :mod:`scripts.alpha_fit` - same-window LOSO balanced accuracy of the deployed
    classifier (tau-EMA + dwell 2), healthy false alarm <= 5 %, within 1 SE the fewest
    spurious CRITICAL episodes per hour on steady healthy cells, then the larger alpha.
    ``--alpha-rule refit0922`` is the archived stage the 2026-09-22 numbers came from: LOSO
    BA of dwell-2 CRITICAL at t against the label at t + 30 s, FA <= 5 %, the most
    responsive tau within 1 SE of the best. D4' replaced it because a t + 30 s label
    mechanically favours tau = 0 (TSS has no lead, plan §6.9c E-B); it is kept to
    reproduce the archive. Both run at the alpha-stage w_p (``--alpha-w-p``; default
    :data:`ALPHA_STAGE_W_P`, the D3 values current on 2026-09-22) and lambda_wait = 1.
``wp`` (D3, the constrained 1-SE rule)
    over :data:`WP_GRID` at the chosen tau and lambda_wait = 1, the verdict
    (``theta_verdict.verdict_report``) of every w_p; a w_p is admissible when
    (c1) its training BA is within one SE of the w_p = 0 BA (cell bootstrap of the BA at
    the w_p = 0 theta), (c2) the family gap ``|theta_P - theta_D| / theta`` is within the
    merged fit's CI half width fraction, and (c3) the family rule publishes the merged
    theta. w_p* is the LARGEST admissible w_p (:func:`d3_select`), 0 when none is. Then the
    lambda check: lambda_wait in :data:`LAMBDAS` at w_p*; lambda moves off 1 only if the
    best BA beats lambda = 1 by >= 0.02.
``final`` (D5 + hold-out)
    the verdict at (tau, w_p*, lambda*) with 1000 / 200 resamples; D5: the merged theta is
    published whatever the family rule says (the family theta is kept as diagnostic);
    then the hold-out report on the validation CSV (``theta_verdict.holdout_report``,
    dwell 2), the M balanced-accuracy CI (cell bootstrap) and the per-prompt-length
    attainment of the TTFT SLO on non-overlapping 30 s tiles.
``summary``
    the table of every model and arm under ``--out-dir`` plus the boundary-band window
    counts per shape / family and what hold cells a family short of
    ``MIN_FAMILY_WINDOWS`` band windows would need (``summary.json``).

Labels are ``tre_common.slo_labels``: ``primary`` is the D6' slowdown label of the
registry profile (``max(500 ms, 5 * idle TTFT(L))``, TPOT 75 ms, >= 20 completions),
``fixed`` the 500 / 75 ms comparison, ``k3`` the k = 3 / 150 ms ablation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from tre_common import slo_labels

#: The archived stage's grid and constants (refit.py, 2026-09-22), unchanged.
TAUS_S: tuple[float, ...] = (0, 5, 10, 15, 20, 30, 40, 60)
DT_REF_S = 10.0
WP_GRID: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2)
LAMBDAS: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)
LAMBDA_WAIT = 1.0
#: lambda moves off LAMBDA_WAIT only when the best lambda beats it by this much BA.
LAMBDA_MIN_GAIN = 0.02
#: w_p the alpha stage runs at: the D3 values current on 2026-09-22 (plan 6.11 D3,
#: fable_acc_wp/wp_rule.log). ``--alpha-w-p`` overrides.
ALPHA_STAGE_W_P: dict[str, float] = {"dsqwen-7b": 0.01, "dsllama-8b": 0.005, "dsqwen-14b": 0.005}
HORIZON_MS = 30_000
FA_MAX = 0.05
SEED = 20260922
DWELL_WINDOWS = 2
TRIM_RAMP_WINDOWS = 1
#: Verdict resamples: the w_p grid, the lambda check, the final verdict.
WP_RESAMPLES = (1000, 200)
LAMBDA_RESAMPLES = (300, 100)
FINAL_RESAMPLES = (1000, 200)
BA_SE_RESAMPLES = 300
M_CI_RESAMPLES = 1000
#: summary: band windows a family needs, and the fallback yield of one hold cell.
MIN_FAMILY_WINDOWS = 30

ARMS = ("primary", "fixed", "k3")
ALPHA_RULES = ("d4prime", "refit0922")
FAMILY_FILES = ("decode_heavy", "prefill_heavy")


# ------------------------------------------------------------------------ inputs


def paths(fit_dir: Path, model: str) -> dict[str, Any]:
    f = Path(fit_dir)
    return {
        "fitting": f / f"{model}_fitting.csv",
        "validation": f / f"{model}_validation.csv",
        "families": {fam: f / f"{model}_fitting_{fam}.csv" for fam in FAMILY_FILES},
    }


def label_for(model: str, arm: str, registry: Optional[str] = None) -> slo_labels.LabelDefinition:
    """The label of one arm: the registry profile's D6' primary, or an arm of it."""
    primary = slo_labels.label_def_for_model(
        model, ttft_p95_ms=500.0, tpot_p95_ms=75.0, mode=None, registry=registry)
    if arm == "primary":
        return primary
    arms = slo_labels.label_arms(primary)
    return {"fixed": arms[slo_labels.ARM_FIXED], "k3": arms[slo_labels.ARM_K3]}[arm]


def shape_fn() -> Callable[[str], str]:
    from scripts import alpha_fit

    table = alpha_fit.shape_table()
    return lambda sid: alpha_fit.shape_of(sid, table)


def alpha_of(tau_s: float) -> float:
    return 1.0 if tau_s <= 0 else 1.0 - math.exp(-DT_REF_S / tau_s)


def step90_s(tau_s: float) -> float:
    a = alpha_of(tau_s)
    if a >= 1.0:
        return 0.0
    return DT_REF_S * math.ceil(math.log(0.1) / math.log(1.0 - a))


def spec_for(tau_s: float, w_p: float, lam: float):
    from scripts import theta_verdict as tv

    return tv.build_signal_spec("tss", w_p=w_p, lambda_wait=lam, qmin=1.0,
                                ema_tau_ms=(tau_s * 1000.0 if tau_s > 0 else None))


# ------------------------------------------------------------------ scoring helpers


def rates(pred: Sequence[bool], truth_violated: Sequence[bool]) -> tuple:
    tp = sum(1 for p, v in zip(pred, truth_violated) if p and v)
    fn = sum(1 for p, v in zip(pred, truth_violated) if not p and v)
    fp = sum(1 for p, v in zip(pred, truth_violated) if p and not v)
    tn = sum(1 for p, v in zip(pred, truth_violated) if not p and not v)
    rec = tp / (tp + fn) if tp + fn else float("nan")
    fa = fp / (fp + tn) if fp + tn else float("nan")
    return rec, fa, (rec + 1.0 - fa) / 2.0, tp + fn, fp + tn


def future_pairs(windows, crit) -> list[tuple]:
    """(cell, crit flag at t, violated at t+30 s) for every window whose +30 s window is labelled."""
    idx = {(w.scenario_id, int(w.window_start_ms)): i for i, w in enumerate(windows)}
    out = []
    for i, w in enumerate(windows):
        j = idx.get((w.scenario_id, int(w.window_start_ms) + HORIZON_MS))
        if j is not None:
            out.append((w.scenario_id, crit[i], not windows[j].slo_met))
    return out


def detection_lags(windows, crit) -> list[Optional[float]]:
    """Per violation episode (run of violated windows in a cell), seconds from its first
    window to the first dwell-confirmed CRITICAL within [start-30 s, end]; None = missed."""
    by = defaultdict(list)
    for i, w in enumerate(windows):
        by[w.scenario_id].append(i)
    lags: list[Optional[float]] = []
    for idx in by.values():
        idx.sort(key=lambda i: windows[i].window_start_ms)
        k = 0
        while k < len(idx):
            if windows[idx[k]].slo_met:
                k += 1
                continue
            s = k
            while k < len(idx) and not windows[idx[k]].slo_met:
                k += 1
            t0 = windows[idx[s]].window_start_ms
            t1 = windows[idx[k - 1]].window_start_ms
            hits = [windows[i].window_start_ms for i in idx
                    if crit[i] and t0 - HORIZON_MS <= windows[i].window_start_ms <= t1]
            lags.append((min(hits) - t0) / 1000.0 if hits else None)
    return lags


def fit_theta_delta(windows, spec):
    from tre_calibration.fit import fit_delta_margins

    cfg = spec.default_config()
    fit = cfg.fit(windows)
    if not fit.publish or fit.theta is None:
        return None
    theta = float(fit.theta)
    d = fit_delta_margins(windows, theta=theta, direction=spec.direction)
    return theta, d.crit.tau, d.crit.delta, d.high.delta, fit


def boot_ba_se(pairs, n: int = 300, seed: int = SEED) -> float:
    cells = sorted({c for c, _, _ in pairs})
    by = defaultdict(list)
    for c, p, v in pairs:
        by[c].append((p, v))
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        smp = [x for c in (rng.choice(cells) for _ in cells) for x in by[c]]
        _, _, ba, npos, nneg = rates([p for p, _ in smp], [v for _, v in smp])
        if npos and nneg:
            vals.append(ba)
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


# ------------------------------------------------------------------------- alpha


def stage_alpha_refit0922(model: str, label, p: Mapping[str, Any], *, w_p: float) -> dict:
    """The archived alpha stage (t + 30 s label; kept to reproduce 2026-09-22)."""
    from scripts import theta_verdict as tv

    shape_of = shape_fn()
    lam = LAMBDA_WAIT
    curve = []
    for tau in TAUS_S:
        spec = spec_for(tau, w_p, lam)
        windows = spec.load(p["fitting"], label, TRIM_RAMP_WINDOWS)
        shapes = sorted({shape_of(w.scenario_id) for w in windows})
        pairs, lags, folds = [], [], {}
        for s in shapes:
            train = [w for w in windows if shape_of(w.scenario_id) != s]
            test = [w for w in windows if shape_of(w.scenario_id) == s]
            fd = fit_theta_delta(train, spec)
            if fd is None:
                folds[s] = None
                continue
            theta, tau_crit = fd[0], fd[1]
            crit = tv.critical_dwell_flags(test, theta=theta, tau_crit=tau_crit,
                                           direction=spec.direction, dwell_windows=DWELL_WINDOWS)
            pairs += future_pairs(test, crit)
            lags += detection_lags(test, crit)
            folds[s] = {"theta": theta, "tau_crit": tau_crit}
        rec, fa, ba, npos, nneg = rates([q for _, q, _ in pairs], [v for _, _, v in pairs])
        se = boot_ba_se(pairs)
        full = fit_theta_delta(windows, spec)
        hit = sorted(x for x in lags if x is not None)
        curve.append({
            "tau_s": tau, "alpha": alpha_of(tau), "loso_ba": ba, "loso_ba_se": se, "recall": rec,
            "false_alarm": fa, "n_pos": npos, "n_neg": nneg, "feasible_fa": fa <= FA_MAX,
            "step90_ema_s": step90_s(tau), "step90_with_dwell_s": step90_s(tau) + DT_REF_S,
            "episodes": len(lags), "episodes_detected": len(hit),
            "detect_lag_median_s": hit[len(hit) // 2] if hit else None,
            "full_fit": ({"theta": full[0], "tau_crit": full[1], "delta_crit": full[2], "delta_high": full[3]}
                         if full else None),
            "folds": folds,
        })
        print(model, tau, f"BA={ba:.3f}+-{se:.3f} rec={rec:.3f} fa={fa:.3f}", flush=True)
    feas = [c for c in curve if c["feasible_fa"]] or curve
    best = max(feas, key=lambda c: c["loso_ba"])
    ok = [c for c in feas if c["loso_ba"] >= best["loso_ba"] - best["loso_ba_se"]]
    chosen = min(ok, key=lambda c: c["tau_s"])  # most responsive = smallest tau
    return {"rule": "refit0922", "w_p": w_p, "lambda_wait": lam,
            "objective": "LOSO (leave-one-shape-out) BA of dwell-2 CRITICAL at t vs label at t+30 s, "
                         "healthy false alarm <= 0.05, most responsive tau within 1 SE of the best",
            "no_feasible_tau": not any(c["feasible_fa"] for c in curve),
            "best_tau_s": best["tau_s"], "chosen_tau_s": chosen["tau_s"], "chosen_alpha": chosen["alpha"],
            "curve": curve}


def stage_alpha_d4prime(model: str, label, p: Mapping[str, Any], *, w_p: float,
                        ledgers: Sequence[str] = (), bootstrap: int = 1000,
                        registry: Optional[str] = None) -> dict:
    """D4' (:mod:`scripts.alpha_fit`) on the same fitting CSV and label."""
    from scripts import alpha_fit

    ns = argparse.Namespace(
        model=model, fitting_csv=str(p["fitting"]), w_p=w_p, lambda_wait=LAMBDA_WAIT, qmin=1.0,
        trim_ramp_windows=TRIM_RAMP_WINDOWS, tau_grid_s=[float(t) for t in TAUS_S],
        dt_ref_s=DT_REF_S, dwell_windows=DWELL_WINDOWS, fa_max=FA_MAX,
        step_ms=alpha_fit.DEFAULT_STEP_MS, se_resamples=alpha_fit.DEFAULT_SE_RESAMPLES,
        bootstrap=bootstrap, bootstrap_refit=False, seed=alpha_fit.DEFAULT_SEED,
        ledger=list(ledgers),
        # the label: this arm's definition, passed field by field
        ttft_p95_ms=label.ttft_p95_ms, tpot_p95_ms=label.tpot_p95_ms,
        ttft_slo_mode=label.ttft_slo_mode, ttft_slowdown_k=label.ttft_slowdown_k,
        ttft_floor_ms=label.ttft_floor_ms, ttft_idle_c_ms=label.ttft_idle_c_ms,
        ttft_idle_b_ms_per_token=label.ttft_idle_b_ms_per_token,
        min_completed_requests=label.min_completed_requests, label_registry=registry,
    )
    rep = alpha_fit.run(ns)
    sel = rep.get("selection") or {}
    return {"rule": "d4prime", "w_p": w_p, "lambda_wait": LAMBDA_WAIT,
            "chosen_tau_s": sel.get("chosen_tau_s"), "chosen_alpha": sel.get("chosen_alpha"),
            "alpha_fit": rep}


# ---------------------------------------------------------------------------- w_p


def verdict(model: str, label, p: Mapping[str, Any], tau: float, w_p: float, lam: float,
            n_res: int = 1000, fam_res: int = 200) -> dict:
    from scripts import theta_verdict as tv

    spec = spec_for(tau, w_p, lam)
    try:
        return tv.verdict_report(model=model, fitting_csv=p["fitting"], families=p["families"], spec=spec,
                                 label=label, trim_ramp_windows=TRIM_RAMP_WINDOWS, n_resamples=n_res,
                                 family_resamples=fam_res, seed=SEED)
    except tv.VerdictError as exc:
        return {"error": str(exc)}


def summarize(v: Mapping[str, Any]) -> dict:
    if "error" in v:
        return dict(v)
    m, fams = v["merged"], v["families"]
    tP = fams.get("prefill_heavy", {}).get("theta")
    tD = fams.get("decode_heavy", {}).get("theta")
    theta = m["theta"]
    half_frac = m["bootstrap"]["ci_half_width_fraction"]
    gap = abs(tP - tD) / theta if tP and tD else None
    return {
        "theta_merged": theta, "theta_published": v["published"]["theta_m"], "source": v["published"]["source"],
        "train_ba": m["fit"].get("balanced_accuracy"), "ci_half_frac": half_frac,
        "publish_rate": m["bootstrap"]["publish_rate"], "theta_P": tP, "theta_D": tD,
        "family_ratio_P_over_D": (tP / tD) if tP and tD else None, "family_gap_frac": gap,
        "family_merged": v["published"]["source"] == "merged",
        "delta_crit": v["published"]["delta_crit"], "delta_high": v["published"]["delta_high"],
        "tau_crit": v["published"]["tau_crit"],
        "delta_crit_ci": [m["delta_crit"]["bootstrap"].get(k) for k in ("delta_p2_5", "delta_p97_5")],
        "stop_rule": v["stop_rule"], "near_band_windows_merged": m["near_theta"]["windows"],
        "family_near_band": {k: f.get("near_merged_theta") for k, f in fams.items()},
    }


def ba_se_at(label, p: Mapping[str, Any], tau: float, w_p: float, lam: float, theta: float,
             n: int = BA_SE_RESAMPLES) -> float:
    """Cell-bootstrap SE of the training BA at a fixed theta (the D3 1-SE width)."""
    from tre_calibration.fit import threshold_balanced_accuracy

    ws = spec_for(tau, w_p, lam).load(p["fitting"], label, TRIM_RAMP_WINDOWS)
    by = defaultdict(list)
    for w in ws:
        by[w.scenario_id].append(w)
    cells = sorted(by)
    rng = random.Random(SEED)
    vals = []
    for _ in range(n):
        smp = [w for c in (rng.choice(cells) for _ in cells) for w in by[c]]
        vals.append(threshold_balanced_accuracy(smp, theta=theta, direction="higher_is_healthier")["balanced_accuracy"])
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def d3_select(rows: list[dict], se: Optional[float]) -> Optional[float]:
    """D3, in place on ``rows`` (``summarize`` rows over the w_p grid, the first at w_p = 0):
    mark c1 (BA within one SE of the w_p = 0 BA), c2 (family gap within the CI half
    width), c3 (the family rule publishes the merged theta) and ``admissible``; return the
    largest admissible w_p, or None when none is (or the w_p = 0 fit failed)."""
    base = rows[0]
    for r in rows:
        if "error" in r or se is None or "error" in base:
            r["admissible"] = False
            continue
        r["c1_1se"] = r["train_ba"] >= base["train_ba"] - se
        r["c2_gap"] = r["family_gap_frac"] is not None and r["family_gap_frac"] <= r["ci_half_frac"]
        r["c3_merged"] = r["family_merged"]
        r["admissible"] = r["c1_1se"] and r["c2_gap"] and r["c3_merged"]
    adm = [r["w_p"] for r in rows if r.get("admissible")]
    return max(adm) if adm else None


def lambda_select(rows: list[dict]) -> float:
    """lambda_wait: stays at LAMBDA_WAIT unless the best lambda beats it by >= LAMBDA_MIN_GAIN BA."""
    ba1 = next((r for r in rows if r["lambda_wait"] == LAMBDA_WAIT), {}).get("train_ba")
    if ba1 is None:
        return LAMBDA_WAIT
    ok = [r for r in rows if "error" not in r]
    best = max(ok, key=lambda r: r["train_ba"])
    return best["lambda_wait"] if best["train_ba"] - ba1 >= LAMBDA_MIN_GAIN else LAMBDA_WAIT


def stage_wp(model: str, label, p: Mapping[str, Any], alpha_doc: Mapping[str, Any]) -> dict:
    tau = alpha_doc["chosen_tau_s"]
    if tau is None:
        raise SystemExit(f"{model}: the alpha stage chose no tau ({alpha_doc.get('rule')})")
    rows = []
    for wp in WP_GRID:
        s = summarize(verdict(model, label, p, tau, wp, LAMBDA_WAIT, *WP_RESAMPLES))
        s["w_p"] = wp
        rows.append(s)
        print(model, "wp", wp, {k: s.get(k) for k in ("theta_merged", "train_ba", "family_gap_frac", "source")},
              flush=True)
    base = rows[0]
    se = ba_se_at(label, p, tau, 0.0, LAMBDA_WAIT, base["theta_merged"]) if "error" not in base else None
    wp_star = d3_select(rows, se)
    wp_l = wp_star if wp_star is not None else 0.0
    lam_rows = []
    for lam in LAMBDAS:
        s = summarize(verdict(model, label, p, tau, wp_l, lam, *LAMBDA_RESAMPLES))
        s["lambda_wait"] = lam
        lam_rows.append(s)
    return {"model": model, "tau_s": tau, "ba0_se": se, "grid": rows,
            "admissible": [r["w_p"] for r in rows if r.get("admissible")],
            "w_p_star": wp_star, "w_p_used": wp_l, "lambda_rows": lam_rows,
            "lambda_star": lambda_select(lam_rows)}


# -------------------------------------------------------------------------- final


def bucket(length: float) -> str:
    return ("<=256" if length <= 256 else "257-1024" if length <= 1024
            else "1025-2048" if length <= 2048 else ">2048")


def stage_final(model: str, label, p: Mapping[str, Any], wp_doc: Mapping[str, Any], out_dir: Path) -> dict:
    from tre_calibration.fit import threshold_balanced_accuracy

    from scripts import theta_verdict as tv

    tau, wp, lam = wp_doc["tau_s"], wp_doc["w_p_used"], wp_doc["lambda_star"]
    v = verdict(model, label, p, tau, wp, lam, *FINAL_RESAMPLES)
    out_dir.mkdir(parents=True, exist_ok=True)
    if "error" in v:
        (out_dir / "verdict_final.json").write_text(json.dumps(v, indent=1, default=str))
        return {"model": model, "error": v["error"]}
    # D5: the merged theta is published whatever the family verdict says (diagnostic only)
    v["published"]["family_rule_theta"] = v["published"]["theta_m"]
    v["published"]["theta_m"] = v["merged"]["theta"]
    v["published"]["d5_merged_published"] = True
    (out_dir / "verdict_final.json").write_text(json.dumps(v, indent=1, default=str))
    h = tv.holdout_report(v, p["validation"], dwell_windows=DWELL_WINDOWS)
    (out_dir / "holdout_final.json").write_text(json.dumps(h, indent=1, default=str))
    # M BA CI (cell bootstrap; few M cells -> wide, reported as such)
    spec = spec_for(tau, wp, lam)
    mw = spec.load(p["validation"], label, TRIM_RAMP_WINDOWS)
    theta = v["published"]["theta_m"]
    by = defaultdict(list)
    for x in mw:
        by[x.scenario_id].append(x)
    cells = sorted(by)
    rng = random.Random(SEED)
    bas = []
    for _ in range(M_CI_RESAMPLES):
        smp = [x for c in (rng.choice(cells) for _ in cells) for x in by[c]]
        r = threshold_balanced_accuracy(smp, theta=theta, direction="higher_is_healthier")
        if r["balanced_accuracy"] == r["balanced_accuracy"]:
            bas.append(r["balanced_accuracy"])
    bas.sort()
    ci = [bas[int(0.025 * len(bas))], bas[int(0.975 * len(bas)) - 1]] if bas else [None, None]
    # per-length-bucket request attainment on M (non-overlapping 30 s tiles only)
    att = defaultdict(lambda: [0, 0])
    first: dict[str, int] = {}
    with open(p["validation"], newline="") as fh:
        for row in csv.DictReader(fh):
            c, s = row["scenario_id"], int(row["window_start_ms"])
            first.setdefault(c, s)
            if (s - first[c]) % HORIZON_MS:
                continue
            for ttft, length in slo_labels.parse_ttft_len_samples(row.get("ttft_len_samples") or ""):
                b = bucket(length)
                att[b][1] += 1
                att[b][0] += ttft <= label.ttft_slo_ms(length)
    wd = h["with_dwell"]
    s = summarize(v)
    s["theta_family_rule"] = v["published"]["family_rule_theta"]
    return {
        "model": model, "tau_s": tau, "alpha": alpha_of(tau), "w_p": wp, "lambda_wait": lam,
        **s, "stop_rule_15": v["stop_rule"]["satisfied"],
        "appendix_10_met": v["stop_rule"].get("appendix_ci_target_met"),
        "train_ba_at_published": threshold_balanced_accuracy(
            spec.load(p["fitting"], label, TRIM_RAMP_WINDOWS), theta=theta,
            direction="higher_is_healthier")["balanced_accuracy"],
        "M_windows": h["windows"], "M_cells": h["cells"], "M_violating": h["violating"],
        "M_ba": h["at_published_theta"]["balanced_accuracy"], "M_ba_ci95": ci,
        "M_recall_both_tpot_dwell2": wd["critical_recall_both_tpot"], "M_both_tpot_n": wd["both_tpot_windows"],
        "M_false_alarm_dwell2": wd["critical_false_alarm_on_healthy"], "M_healthy_n": wd["healthy_windows"],
        "M_recall_all_dwell2": wd["critical_recall_of_violating"],
        "M_ttft_only_recall_dwell2": wd["violation_classes"]["ttft_only"]["critical_recall"],
        "M_ttft_only_n": wd["violation_classes"]["ttft_only"]["windows"],
        "M_classes_dwell2": wd["violation_classes"],
        "M_bucket_attainment": {b: {"met": a[0], "n": a[1], "rate": a[0] / a[1] if a[1] else None}
                                for b, a in sorted(att.items())},
    }


# ------------------------------------------------------------------------ summary


def band_counts(model: str, arm: str, final: Mapping[str, Any], p: Mapping[str, Any], *,
                registry: Optional[str] = None, ledger: Optional[Mapping[str, Any]] = None) -> dict:
    """Boundary-band windows (|Z - 1| <= theta_verdict.BOUNDARY_BAND at the published theta)
    per shape and family, and the hold cells a family short of MIN_FAMILY_WINDOWS needs.

    A "dwell" cell (the yield estimate) is a steady cell of >= 20 windows -
    ``alpha_fit.is_steady_cell``, i.e. from the run's ledger when there is one."""
    from scripts import alpha_fit
    from scripts import gen_calibration_schedules as gen
    from scripts import theta_verdict as tv

    families = gen.families()
    family_of = {s: fam for fam, shapes in families.items() for s in shapes}
    shape_of = shape_fn()
    spec = spec_for(final["tau_s"], final["w_p"], final["lambda_wait"])
    ws = spec.load(p["fitting"], label_for(model, arm, registry), TRIM_RAMP_WINDOWS)
    theta = final["theta_published"]
    band: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    cellband: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for x in ws:
        z = x.signal / theta if theta else float("nan")
        inside = int(math.isfinite(z) and abs(z - 1.0) <= tv.BOUNDARY_BAND)
        cellband[x.scenario_id][1] += 1
        cellband[x.scenario_id][0] += inside
        s = shape_of(x.scenario_id)
        band[s][1] += 1
        band[s][0] += inside
    fam: dict[str, int] = defaultdict(int)
    for s, (n, _total) in band.items():
        if s in family_of:
            fam[family_of[s]] += n
    need = []
    for famname, shapes in sorted(families.items()):
        have = fam.get(famname, 0)
        deficit = max(0, MIN_FAMILY_WINDOWS - have)
        dw = [v[0] for c, v in cellband.items()
              if shape_of(c) in shapes and v[1] >= 20 and alpha_fit.is_steady_cell(c, ledger)]
        yield_per_cell = (sum(dw) / len(dw)) if dw and sum(dw) > 0 else 28 * 0.3
        need.append({"family": famname, "band_windows": have, "deficit": deficit, "shapes": list(shapes),
                     "hold_cells": math.ceil(deficit / yield_per_cell) if deficit else 0,
                     "yield_band_windows_per_hold_cell": round(yield_per_cell, 1),
                     "observed_dwell_cells": len(dw)})
    half = final.get("ci_half_frac")
    return {
        "band_by_shape": {s: {"band": v[0], "independent": round(v[0] / 3, 1), "windows": v[1]}
                          for s, v in sorted(band.items())},
        "band_by_family": {k: {"band": v, "independent": round(v / 3, 1)} for k, v in fam.items()},
        "needs": {"family_holds": need, "ci_half_frac": half,
                  "ci_window_factor_for_15pct": round((half / 0.15) ** 2, 2) if half and half > 0.15 else 1.0,
                  "ci_window_factor_for_10pct": round((half / 0.10) ** 2, 2) if half else None},
    }


def stage_summary(out_root: Path, fit_dirs: Mapping[str, Path], *, registry: Optional[str] = None,
                  ledger: Optional[Mapping[str, Any]] = None) -> dict:
    out: dict[str, Any] = {"root": str(out_root), "models": {}}
    for model in sorted(fit_dirs):
        for arm in ARMS:
            d = out_root / model / arm
            if not (d / "final.json").exists():
                continue
            a = json.loads((d / "alpha.json").read_text())
            w = json.loads((d / "wp.json").read_text())
            fin = json.loads((d / "final.json").read_text())
            rec = {
                "alpha": {k: a.get(k) for k in ("rule", "chosen_tau_s", "chosen_alpha", "best_tau_s", "w_p")},
                "wp_rule": {"tau_s": w["tau_s"], "ba0_se": w["ba0_se"], "admissible": w["admissible"],
                            "w_p_star": w["w_p_star"], "w_p_used": w["w_p_used"],
                            "lambda_star": w["lambda_star"]},
                "final": fin,
            }
            if "error" not in fin:
                rec.update(band_counts(model, arm, fin, paths(fit_dirs[model], model),
                                       registry=registry, ledger=ledger))
            out["models"].setdefault(model, {})[arm] = rec
    return out


# ---------------------------------------------------------------------------- CLI


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} is missing: run the previous stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["alpha", "wp", "final", "summary"])
    ap.add_argument("--model", action="append", required=True,
                    help="model (repeatable for summary)")
    ap.add_argument("--arm", choices=ARMS, default="primary")
    ap.add_argument("--fit-dir", type=Path, required=True,
                    help="directory of the re-windowed CSVs (<model>_fitting.csv ...); for several "
                         "models a template with {model}, e.g. /r/{model}/fit/fit")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--alpha-rule", choices=ALPHA_RULES, default="d4prime")
    ap.add_argument("--alpha-w-p", type=float, default=None,
                    help=f"w_p of the alpha stage (default: {ALPHA_STAGE_W_P})")
    ap.add_argument("--alpha-bootstrap", type=int, default=1000,
                    help="whole-rule bootstrap resamples of the d4prime alpha rule")
    ap.add_argument("--ledger", action="append", default=[],
                    help="cells.jsonl of a ladder-design run (steady cells for D4' and the summary)")
    ap.add_argument("--registry", default=None, help="registry the label's idle TTFT fit is read from")
    args = ap.parse_args(argv)

    def fit_dir(model: str) -> Path:
        text = str(args.fit_dir)
        return Path(text.format(model=model)) if "{model}" in text else args.fit_dir

    ledger = None
    if args.ledger:
        from scripts.rewindow_from_raw import load_ledgers

        ledger = load_ledgers(args.ledger)
    if args.stage == "summary":
        doc = stage_summary(args.out_dir, {m: fit_dir(m) for m in args.model},
                            registry=args.registry, ledger=ledger)
        (args.out_dir / "summary.json").write_text(json.dumps(doc, indent=1, default=str))
        print(f"wrote {args.out_dir / 'summary.json'}")
        return 0
    if len(args.model) != 1:
        ap.error(f"stage {args.stage} takes one --model")
    model = args.model[0]
    label = label_for(model, args.arm, args.registry)
    p = paths(fit_dir(model), model)
    out = args.out_dir / model / args.arm
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "alpha":
        w_p = args.alpha_w_p if args.alpha_w_p is not None else ALPHA_STAGE_W_P.get(model)
        if w_p is None:
            ap.error(f"no alpha-stage w_p for {model}: pass --alpha-w-p")
        if args.alpha_rule == "refit0922":
            doc = stage_alpha_refit0922(model, label, p, w_p=w_p)
        else:
            doc = stage_alpha_d4prime(model, label, p, w_p=w_p, ledgers=args.ledger,
                                      bootstrap=args.alpha_bootstrap, registry=args.registry)
    elif args.stage == "wp":
        doc = stage_wp(model, label, p, _read_json(out / "alpha.json"))
    else:
        doc = stage_final(model, label, p, _read_json(out / "wp.json"), out)
    doc.update({"model": model, "arm": args.arm, "label_def": label.as_dict(),
                "inputs": {k: str(v) for k, v in p.items() if k != "families"}
                | {f"family_{k}": str(v) for k, v in p["families"].items()}})
    (out / f"{args.stage}.json").write_text(json.dumps(doc, indent=1, default=str))
    print(f"wrote {out / (args.stage + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

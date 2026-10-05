"""POST-HOC diagnosis of the 2026-10-04 stage-H acceptance failure (frozen theta on M).

POST-HOC. NOT ACCEPTANCE EVIDENCE. NOT USED FOR ANY DECISION OF THIS ROUND.
The frozen ``dline_refit accept`` result (freeze/params_freeze.accept.json, FAILED) is the
only verdict. Everything here is read after that verdict, on the same held-out windows, to
inform the design of the NEXT calibration round.

What it computes (per model, from accept's own validation CSVs and the frozen loader):

* Q1 anatomy: per-cell table (windows, severe, CRITICAL hits, misses, FA), severe-violation
  episodes and their CRITICAL lag, miss classification (onset vs later, Z band), and a
  per-window timeline of every bursts cell (offered load, running / waiting, TSS raw vs EMA,
  Z, label, CRITICAL; L3 Z next to it);
* lag split: the same episodes scored with (i) the frozen signal, (ii) the gateway
  numerator without EMA, (iii) the L3 numerator without EMA, (iv) L3 with EMA, so
  EMA share = (i) - (ii), completion-numerator share = (ii) - (iii), residual = (iii)
  (window length / step + queue and label timing);
* Q2: 14b confusion at Z = 1 (BA) per cell, M vs T14;
* Q3 counterfactuals with the accept gate arithmetic (b_prime.series_point / b_prime_boot /
  b_prime.criteria, acceptance_bootstrap for the BA CI, seed 20260922, 1000 resamples):
  L3 numerator (training L3 v1-lambda refit), EMA tau 5 s and 0 s (theta frozen), and two
  queue overrides whose thresholds come from TRAINING healthy windows only (99th
  percentile; rule fixed before M is read by this tool);
* Q4 power: nested cell bootstrap - how many independent severe cells M needs for the B'
  recall CI95 low to reach 0.70.

Usage (from tre/deploy, PYTHONPATH as the RUN plan):
  python3 -m scripts.analysis.posthoc_hfail_20261005 --calib-root $C --out-dir $C/posthoc-20261005
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

MODELS = ("dsqwen-7b", "dsllama-8b", "dsqwen-14b")
LABEL = ("POST-HOC diagnosis 2026-10-05 - NOT acceptance evidence, not used for any decision of this round; "
         "the frozen accept result (FAILED) is the only verdict. Informs the next calibration round only.")
STEP_MS = 10_000
WINDOW_MS = 30_000
#: Leading-indicator thresholds: this quantile of the indicator over TRAINING healthy windows
#: (so the override alone fires on <= 1 % of training healthy windows). Fixed before M is read.
OVERRIDE_TRAIN_HEALTHY_QUANTILE = 0.99


# ------------------------------------------------------------------ loading


def key_of(scenario_id: str, start: Any) -> tuple[str, int]:
    return (str(scenario_id), int(round(float(start))))


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def spec_with_ema(spec, ema_tau_ms: Optional[float]):
    return dataclasses.replace(spec, tss=dataclasses.replace(spec.tss, ema_tau_ms=ema_tau_ms),
                               ema_tau_ms=ema_tau_ms)


def l3_entry(calib: Path, model: str, gw_entry: dict) -> dict:
    from scripts import dline_refit as dl

    base = calib / "fit/l3/refit-v1lambda" / model / "primary"
    vf = json.loads((base / "verdict_final.json").read_text())
    fin = json.loads((base / "final.json").read_text())
    return {"verdict_for_holdout": dl.verdict_for_holdout(vf), "windowing": gw_entry.get("windowing"),
            "train_ba_at_published": fin["train_ba_at_published"],
            "training_csv": str(calib / "fit/l3" / model / "fit" / f"{model}_fitting.csv"),
            "source": str(base)}


def build_l3_validation_csv(calib: Path, out_dir: Path, model: str) -> Path:
    """accept's validation-CSV rule (collect_m_rows over the M manifest cells) on the L3 datasets:
    M from the post-hoc build, run1 (retained cells) from the E1 build."""
    from scripts import dline_refit as dl

    path = out_dir / "validation_l3" / f"{model}_validation_l3.csv"
    if path.exists():
        return path
    man = json.loads((calib / "M" / model / "M_manifest.json").read_text())
    sources = [dl.DatasetSource.parse(f"M.{model}={out_dir}/M_l3/{model}/dataset_l3", sealed_to_h2=False),
               dl.DatasetSource.parse(f"run1.{model}={calib}/run1/{model}/dataset_l3", sealed_to_h2=False)]
    m, problems = dl.collect_m_rows(sources, {model: man})
    if problems:
        raise SystemExit(f"{model}: L3 validation rows: {problems}")
    path.parent.mkdir(parents=True, exist_ok=True)
    dl._write_validation_csv(path, m["header"][model], m["rows"][model])
    return path


def offered_rps(calib: Path, rows: Sequence[dict]) -> dict[tuple[str, int], float]:
    """Requests SENT in each window (start, end] / window seconds, from the datasets' requests.csv."""
    from collections import Counter

    cells = {(r["cell_id"], str(r["attempt"])): r["scenario_id"] for r in rows}
    wins: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in rows:
        wins[r["scenario_id"]].append((int(float(r["window_start_ms"])), int(float(r["window_end_ms"]))))
    model = rows[0]["model"]
    sends: dict[str, list[int]] = defaultdict(list)
    done: dict[str, list[tuple[int, int, Optional[float]]]] = defaultdict(list)
    for p in (calib / "M" / model / "dataset" / "requests.csv", calib / "run1" / model / "dataset" / "requests.csv"):
        with open(p, newline="", encoding="utf-8") as fh:
            for q in csv.DictReader(fh):
                sid = cells.get((q["cell_id"], str(q["attempt"])))
                if sid is not None and q.get("send_ts_ms"):
                    sends[sid].append(int(float(q["send_ts_ms"])))
                    if q.get("done_ts_ms") and q.get("outcome") == "ok":
                        done[sid].append((int(float(q["done_ts_ms"])), int(float(q["send_ts_ms"])), fnum(q.get("ttft_ms"))))
    out = {}
    import bisect

    for sid, ws in wins.items():
        s = sorted(sends.get(sid, []))
        for a, b in ws:
            n = bisect.bisect_right(s, b) - bisect.bisect_right(s, a)
            out[(sid, a)] = n / ((b - a) / 1000.0)
    return out, done


class Var:
    """One CRITICAL rule on one window set."""

    def __init__(self, name, windows, theta, tau_crit, direction, crit=None, cut=None, note="",
                 train_ba=None, ba_override=None):
        from scripts import theta_verdict as tv

        self.name, self.windows, self.theta, self.tau_crit, self.direction = name, windows, theta, tau_crit, direction
        self.crit = crit if crit is not None else tv.critical_dwell_flags(
            windows, theta=theta, tau_crit=tau_crit, direction=direction, dwell_windows=1, window_ms=WINDOW_MS)
        self.cut, self.note, self.train_ba, self.ba_override = cut, note, train_ba, ba_override
        self.by_key = {key_of(w.scenario_id, w.window_start_ms): i for i, w in enumerate(windows)}

    def z(self, i):
        return self.windows[i].signal / self.theta


# ------------------------------------------------------------------ scoring (accept arithmetic)


def gate_eval(v: Var, cut: float, gate: dict, *, n: int, seed: int, a_train_ba: Optional[float]) -> dict:
    from scripts import b_prime
    from scripts import dline_refit as dl
    from tre_calibration.fit import threshold_balanced_accuracy

    point = b_prime.series_point(v.windows, theta=v.theta, cut=cut, crit=v.crit)
    ci = b_prime.b_prime_boot(v.windows, cut=cut, crit=v.crit, n=n, seed=seed)
    crits = b_prime.criteria(point, ci, gate)
    shares = b_prime.band_shares(v.windows, theta=v.theta, tau_crit=v.tau_crit, cut=cut)
    ba = threshold_balanced_accuracy(v.windows, theta=v.theta, direction=v.direction)["balanced_accuracy"]
    boot = dl.acceptance_bootstrap(v.windows, v.crit, theta=v.theta, direction=v.direction, n_resamples=n, seed=seed)
    ba_ci = boot["metrics"]["balanced_accuracy"]["ci95"]
    a = None
    if a_train_ba is not None:
        a = {"ba_ge_0.80": ba >= dl.A_BA_MIN, "ci_low_ge_0.75": ba_ci[0] is not None and ba_ci[0] >= dl.A_BA_CI_LOW_MIN,
             "drop_ge_-0.08": ba >= a_train_ba - dl.A_MAX_DROP_FROM_TRAINING, "train_ba": a_train_ba}
        a["passed"] = all(a[k] for k in ("ba_ge_0.80", "ci_low_ge_0.75", "drop_ge_-0.08"))
    return {"recall_severe": point["recall_severe"], "recall_severe_ci95": ci["recall_severe_ci95"],
            "false_alarm": point["false_alarm"], "false_alarm_ci95": ci["false_alarm_ci95"],
            "recall_all": point["recall_all"], "b_prime_passed": all(c["met"] for c in crits),
            "b_prime_criteria": crits, "severe_windows": shares["severe_windows"], "healthy": shares["healthy"],
            "violating": shares["violating"], "windows": len(v.windows),
            "balanced_accuracy": ba, "ba_ci95": ba_ci, "A": a, "severity_cut": cut}


# ------------------------------------------------------------------ episodes and lag


def episodes(base: Var, cut: float) -> list[dict]:
    """Runs of consecutive (10 s step) violating windows per cell that hold >= 1 severe window
    (labels of the frozen window set)."""
    from scripts import b_prime

    by: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(base.windows):
        by[w.scenario_id].append(i)
    eps = []
    for sid, idx in by.items():
        idx.sort(key=lambda i: base.windows[i].window_start_ms)
        k = 0
        while k < len(idx):
            if base.windows[idx[k]].slo_met:
                k += 1
                continue
            s = k
            while (k + 1 < len(idx) and not base.windows[idx[k + 1]].slo_met
                   and base.windows[idx[k + 1]].window_start_ms - base.windows[idx[k]].window_start_ms <= STEP_MS):
                k += 1
            run = idx[s:k + 1]
            k += 1
            sev = [i for i in run if b_prime.severity(base.windows[i]) >= cut]
            if not sev:
                continue
            eps.append({"cell": sid, "t_first_viol": base.windows[run[0]].window_start_ms,
                        "t_first_sev": base.windows[sev[0]].window_start_ms,
                        "t_end": base.windows[run[-1]].window_start_ms,
                        "n_viol": len(run), "n_sev": len(sev), "sev_keys": [key_of(sid, base.windows[i].window_start_ms) for i in sev]})
    return eps


def first_crit(v: Var, cell: str, lo: float, hi: float) -> Optional[float]:
    ts = [v.windows[i].window_start_ms for i, w in enumerate(v.windows)
          if w.scenario_id == cell and v.crit[i] and lo <= w.window_start_ms <= hi]
    return min(ts) if ts else None


def lag_of(v: Var, ep: dict) -> Optional[float]:
    t = first_crit(v, ep["cell"], ep["t_first_sev"] - WINDOW_MS, ep["t_end"])
    return None if t is None else (t - ep["t_first_sev"]) / 1000.0


def lag_summary(lags: Sequence[Optional[float]]) -> dict:
    hit = sorted(x for x in lags if x is not None)
    return {"episodes": len(lags), "detected": len(hit), "missed": len(lags) - len(hit),
            "median_s": statistics.median(hit) if hit else None, "max_s": max(hit) if hit else None,
            "lags_s": [x for x in lags]}


# ------------------------------------------------------------------ power


def power_sim(cells: list[tuple[int, int]], ns: Sequence[int], *, outer: int, inner: int, seed: int,
              homogeneous_p: Optional[float] = None) -> list[dict]:
    """Nested cell bootstrap. Each simulated M has N severe cells drawn with replacement from
    ``cells`` = [(hits, severe windows)] (or, with ``homogeneous_p``, sizes drawn from ``cells``
    and hits ~ Binomial(size, p)); its recall CI95 is the accept rule (cell bootstrap, percentile).
    Returns P(CI low >= 0.70), P(point >= 0.80 and CI low >= 0.70), median CI low."""
    import numpy as np

    rng = np.random.default_rng(seed)
    h = np.array([c[0] for c in cells], float)
    s = np.array([c[1] for c in cells], float)
    out = []
    for n in ns:
        lows, both = [], 0
        for _ in range(outer):
            pick = rng.integers(0, len(cells), n)
            sz = s[pick]
            hits = rng.binomial(sz.astype(int), homogeneous_p).astype(float) if homogeneous_p is not None else h[pick]
            point = hits.sum() / sz.sum()
            ib = rng.integers(0, n, (inner, n))
            rec = np.sort(hits[ib].sum(1) / sz[ib].sum(1))
            low = rec[int(0.025 * inner)]
            lows.append(low)
            both += int(point >= 0.80 and low >= 0.70)
        lows = np.array(lows)
        out.append({"n_cells": n, "p_ci_low_ge_0.70": float((lows >= 0.70).mean()),
                    "p_point_and_ci_pass": both / outer, "median_ci_low": float(np.median(lows))})
    return out


# ------------------------------------------------------------------ per model


def training_override_thresholds(entry: dict) -> dict:
    """99th percentile over TRAINING healthy windows (frozen label, every non-dropped row) of
    q1 = avg_waiting / max(avg_running, 1) and q2 = avg_waiting(t) - avg_waiting(t - 10 s)."""
    from scripts.analysis import h_dropped_windows as hd
    from tre_calibration.dataset import calibration_window_from_row

    spec, label, trim = hd.frozen_spec_and_label(entry)
    path = Path(entry["b_prime"]["training_csv"])
    rows = read_rows(path)
    prev = {}
    for r in rows:
        prev[(r.get("run", ""), r["scenario_id"], int(float(r["window_start_ms"])))] = fnum(r.get("avg_waiting"))
    q1, q2 = [], []
    for r in rows:
        w = calibration_window_from_row(r, latency_slo_ms=label, signal=1.0)
        if w is None or not w.slo_met:
            continue
        wt, rn = fnum(r.get("avg_waiting")) or 0.0, fnum(r.get("avg_running")) or 0.0
        q1.append(wt / max(rn, 1.0))
        p = prev.get((r.get("run", ""), r["scenario_id"], int(float(r["window_start_ms"])) - STEP_MS))
        if p is not None:
            q2.append(wt - p)

    def q(v):
        v = sorted(v)
        return v[min(len(v) - 1, int(OVERRIDE_TRAIN_HEALTHY_QUANTILE * len(v)))]

    return {"q1_wait_per_running": q(q1), "q2_wait_growth_10s": q(q2), "training_healthy_windows": len(q1),
            "training_csv": str(path), "quantile": OVERRIDE_TRAIN_HEALTHY_QUANTILE}


P1_RUNS = ("p1", "p1r2")
#: The 2026-09-24 B' cuts (D16 training set of the 09-23 round; other c/b label and engine
#: state, so only a reference): b_prime_thresholds.json severity_cut_train.
D22_CUTS = {"dsqwen-7b": 14.066, "dsllama-8b": 15.276, "dsqwen-14b": 7.971}


def severity_cut_hypothesis(model: str, entry: dict, base: "Var", gate: dict, out_dir: Path, *, n: int,
                            seed: int) -> dict:
    """H-cut (coordinator 2026-10-05): the P1 deep-overload holds (1.5-3 x rho*) in this round's
    training pool raised the .65 severity quantile, so few M windows reach 'severe'.
    Cut with / without the P1 rows (run p1 / p1r2, whole cells dropped, so the per-cell EMA of the
    other cells is unchanged), M severe windows / cells under each cut, and B' of the frozen
    CRITICAL flags under each cut; the 09-24 (D22-round) cut as a reference."""
    from scripts import b_prime
    from scripts.analysis import h_dropped_windows as hd

    spec, label, trim = hd.frozen_spec_and_label(entry)
    src = Path(entry["b_prime"]["training_csv"])
    with open(src, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        head, rows = reader.fieldnames, list(reader)
    cut_dir = out_dir / "cuts"
    cut_dir.mkdir(parents=True, exist_ok=True)
    pools = {}
    for name, keep in (("no_p1", lambda r: r.get("run") not in P1_RUNS), ("p1_only", lambda r: r.get("run") in P1_RUNS)):
        p = cut_dir / f"{model}_fitting_{name}.posthoc.csv"
        with open(p, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=head)
            wr.writeheader()
            wr.writerows(r for r in rows if keep(r))
        pools[name] = spec.load(p, label, trim)
    pools["all"] = spec.load(src, label, trim)
    sev_dist = {}
    for name, ws in pools.items():
        s = sorted(b_prime.violating_severities(ws))
        sev_dist[name] = {"violating": len(s), "cut_q65": (b_prime.severity_cut(ws) if s else None),
                          "median": statistics.median(s) if s else None}
    cuts = {"frozen_all": float(entry["b_prime"]["severity_cut"]), "no_p1": sev_dist["no_p1"]["cut_q65"],
            "d22_0924_reference": D22_CUTS[model]}
    out = {"training_pools": sev_dist, "recomputed_all_equals_frozen":
           abs(sev_dist["all"]["cut_q65"] - cuts["frozen_all"]) < 1e-6, "by_cut": {}}
    for name, c in cuts.items():
        ev = gate_eval(base, c, gate, n=n, seed=seed, a_train_ba=None)
        sev_cells = defaultdict(lambda: [0, 0])
        for i, w in enumerate(base.windows):
            if not w.slo_met and b_prime.severity(w) >= c:
                sev_cells[w.scenario_id][1] += 1
                sev_cells[w.scenario_id][0] += int(base.crit[i])
        out["by_cut"][name] = {"cut": c, "m_severe_windows": ev["severe_windows"], "m_severe_cells": len(sev_cells),
                               "m_severe_by_cell": {k: v for k, v in sorted(sev_cells.items())},
                               "recall_severe": ev["recall_severe"], "recall_severe_ci95": ev["recall_severe_ci95"],
                               "false_alarm": ev["false_alarm"], "false_alarm_ci95": ev["false_alarm_ci95"],
                               "b_prime_passed": ev["b_prime_passed"]}
    return out


def analyse_model(model: str, calib: Path, out_dir: Path, doc: dict, cfg: dict, *, n: int, seed: int) -> dict:
    from scripts import b_prime
    from scripts import dline_refit as dl
    from scripts.analysis import h_dropped_windows as hd
    from tre_calibration.fit import threshold_balanced_accuracy

    entry = doc["models"][model]
    vh = entry["verdict_for_holdout"]
    spec, label, trim = hd.frozen_spec_and_label(entry)
    theta, tau = float(vh["published"]["theta_m"]), float(vh["published"]["tau_crit"])
    direction = vh["fit_config"]["direction"]
    cut, gate = float(cfg["severity_cut"]), cfg["gate"]
    acc_csv = calib / "freeze/params_freeze.accept.d" / f"{model}_validation.csv"
    rows = read_rows(acc_csv)
    row_of = {key_of(r["scenario_id"], r["window_start_ms"]): r for r in rows}
    train_csv = Path(entry["b_prime"]["training_csv"])
    train_ba_frozen = float(entry["train_ba_at_published"])

    def load(sp, path=acc_csv):
        return sp.load(path, label, trim)

    base = Var("frozen", load(spec), theta, tau, direction, cut=cut, train_ba=train_ba_frozen,
               note="gateway numerator, EMA 10 s, frozen theta / tau_crit")
    raw = Var("gw_ema0", load(spec_with_ema(spec, None)), theta, tau, direction, cut=cut,
              note="gateway numerator, no EMA, frozen theta / tau_crit")
    ema5 = Var("gw_ema5", load(spec_with_ema(spec, 5000.0)), theta, tau, direction, cut=cut,
               note="gateway numerator, EMA 5 s, frozen theta / tau_crit")

    def train_ba(sp):
        tw = sp.load(train_csv, label, trim)
        return threshold_balanced_accuracy(tw, theta=theta, direction=direction)["balanced_accuracy"]

    tb = {"frozen_recomputed": train_ba(spec), "frozen_sealed": train_ba_frozen,
          "ema0": train_ba(spec_with_ema(spec, None)), "ema5": train_ba(spec_with_ema(spec, 5000.0))}
    raw.train_ba, ema5.train_ba = tb["ema0"], tb["ema5"]

    # L3
    le = l3_entry(calib, model, entry)
    lvh = le["verdict_for_holdout"]
    lspec, llabel, ltrim = hd.frozen_spec_and_label(le)
    l3_csv = build_l3_validation_csv(calib, out_dir, model)
    lth, ltau = float(lvh["published"]["theta_m"]), float(lvh["published"]["tau_crit"])
    l3_cut_rec = dl.b_prime_freeze_record(lvh, Path(le["training_csv"]))
    l3 = Var("l3_ema10", lspec.load(l3_csv, llabel, ltrim), lth, ltau, direction, cut=l3_cut_rec["severity_cut"],
             train_ba=float(le["train_ba_at_published"]),
             note="L3 numerator (vLLM counters), training L3 v1-lambda refit theta / tau_crit / w_p / lambda, EMA 10 s")
    l3raw = Var("l3_ema0", spec_with_ema(lspec, None).load(l3_csv, llabel, ltrim), lth, ltau, direction,
                cut=l3_cut_rec["severity_cut"], note="L3 numerator, no EMA, L3 theta / tau_crit (decomposition only)")
    common = set(base.by_key) & set(l3.by_key)
    base_c = Var("frozen_on_l3_windows", [base.windows[base.by_key[k]] for k in sorted(common, key=lambda k: base.by_key[k])],
                 theta, tau, direction, note="frozen rule restricted to the windows L3 keeps (void-free)")

    # queue overrides (thresholds from training healthy windows)
    thr = training_override_thresholds(entry)

    def ind(i, which):
        r = row_of[key_of(base.windows[i].scenario_id, base.windows[i].window_start_ms)]
        wt, rn = fnum(r.get("avg_waiting")) or 0.0, fnum(r.get("avg_running")) or 0.0
        if which == "q1":
            return wt / max(rn, 1.0)
        pk = key_of(r["scenario_id"], float(r["window_start_ms"]) - STEP_MS)
        p = row_of.get(pk)
        return None if p is None else wt - (fnum(p.get("avg_waiting")) or 0.0)

    q1v = [ind(i, "q1") for i in range(len(base.windows))]
    q2v = [ind(i, "q2") for i in range(len(base.windows))]
    ov1 = Var("override_q1", base.windows, theta, tau, direction,
              crit=[c or (x is not None and x > thr["q1_wait_per_running"]) for c, x in zip(base.crit, q1v)],
              note=f"frozen CRITICAL OR waiting/running > {thr['q1_wait_per_running']:.3f} (train healthy p99)",
              train_ba=train_ba_frozen)
    ov2 = Var("override_q2", base.windows, theta, tau, direction,
              crit=[c or (x is not None and x > thr["q2_wait_growth_10s"]) for c, x in zip(base.crit, q2v)],
              note=f"frozen CRITICAL OR waiting growth over 10 s > {thr['q2_wait_growth_10s']:.3f} (train healthy p99)",
              train_ba=train_ba_frozen)

    variants = [base, base_c, l3, ema5, raw, ov1, ov2]
    results = {}
    for v in variants:
        results[v.name] = {"note": v.note, **gate_eval(v, v.cut if v.cut is not None else cut, gate, n=n, seed=seed,
                                                       a_train_ba=v.train_ba)}
    # L3 with the gateway (frozen) cut too: same label, the cut only differs by the finite-signal set
    results["l3_ema10_gwcut"] = {"note": "L3 as above, scored with the frozen (gateway training) severity cut",
                                 **gate_eval(l3, cut, gate, n=n, seed=seed, a_train_ba=l3.train_ba)}

    # hypothesis H-cut: the P1 deep-overload holds pushed the training severity cut up
    cut_h = severity_cut_hypothesis(model, entry, base, gate, out_dir, n=n, seed=seed)

    # sanity: frozen reproduces accept
    acc = json.loads((calib / "freeze/params_freeze.accept.json").read_text())["models"][model]
    acc_d1 = acc["criteria"]["B_prime"]["by_dwell"]["1"]
    sanity = {"accept_recall_severe": acc_d1["recall_severe"], "ours": results["frozen"]["recall_severe"],
              "accept_ci": acc_d1["recall_severe_ci95"], "ours_ci": results["frozen"]["recall_severe_ci95"],
              "accept_ba": acc["holdout_report"]["at_published_theta"]["balanced_accuracy"],
              "ours_ba": results["frozen"]["balanced_accuracy"],
              "accept_ba_ci": acc["bootstrap"]["metrics"]["balanced_accuracy"]["ci95"],
              "ours_ba_ci": results["frozen"]["ba_ci95"], "train_ba": tb}
    sanity["matches"] = (abs(sanity["accept_recall_severe"] - sanity["ours"]) < 1e-12
                         and sanity["accept_ci"] == sanity["ours_ci"]
                         and abs(sanity["accept_ba"] - sanity["ours_ba"]) < 1e-12)

    # per-cell table (frozen)
    meta = {}
    for r in rows:
        meta.setdefault(r["scenario_id"], {"shape": r["shape"], "primitive": r["primitive"], "rho": r["rho"],
                                           "rho_factor": r["rho_factor"], "cell_id": r["cell_id"]})
    man = json.loads((calib / "M" / model / "M_manifest.json").read_text())
    origin = {c["cell_id"]: c["origin"] for c in man["cells"]}
    cells = []
    by: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(base.windows):
        by[w.scenario_id].append(i)
    for sid in sorted(by, key=lambda s: (meta[s]["primitive"], s)):
        idx = by[sid]
        sev = [i for i in idx if not base.windows[i].slo_met and b_prime.severity(base.windows[i]) >= cut]
        ok = [i for i in idx if base.windows[i].slo_met]
        viol = [i for i in idx if not base.windows[i].slo_met]
        zlt1 = lambda i: base.z(i) < 1.0  # noqa: E731
        cells.append({"cell": sid, **meta[sid], "origin": origin.get(meta[sid]["cell_id"]),
                      "windows": len(idx), "violating": len(viol), "severe": len(sev),
                      "crit_hits_severe": sum(base.crit[i] for i in sev),
                      "misses_severe": sum(not base.crit[i] for i in sev),
                      "healthy": len(ok), "fa": sum(base.crit[i] for i in ok),
                      "fn_z1": sum(not zlt1(i) for i in viol), "fp_z1": sum(zlt1(i) for i in ok),
                      "z_viol_median": statistics.median(base.z(i) for i in viol) if viol else None,
                      "sev_viol_median": statistics.median(b_prime.severity(base.windows[i]) for i in viol) if viol else None,
                      "z_healthy_median": statistics.median(base.z(i) for i in ok) if ok else None,
                      "l3_misses_severe": sum(1 for i in sev if (k := key_of(sid, base.windows[i].window_start_ms)) in l3.by_key
                                              and not l3.crit[l3.by_key[k]]),
                      "l3_void_severe": sum(1 for i in sev if key_of(sid, base.windows[i].window_start_ms) not in l3.by_key)})

    # episodes / lag / decomposition
    eps = episodes(base, cut)
    for ep in eps:
        for v in (base, raw, ema5, l3, l3raw, ov1, ov2):
            ep[f"lag_{v.name}"] = lag_of(v, ep)
        tfc = first_crit(base, ep["cell"], ep["t_first_sev"] - WINDOW_MS, ep["t_end"])
        miss_on = miss_late = 0
        bands = {"critical": 0, "low": 0, "z_ge_1": 0}
        for k in ep["sev_keys"]:
            i = base.by_key[k]
            if base.crit[i]:
                continue
            if tfc is None or k[1] < tfc:
                miss_on += 1
            else:
                miss_late += 1
            z = base.z(i)
            bands["critical" if z < tau else ("low" if z < 1 else "z_ge_1")] += 1
        ep.update({"sev_missed_before_first_crit": miss_on, "sev_missed_after_first_crit": miss_late,
                   "missed_z_band": bands, "primitive": meta[ep["cell"]]["primitive"], "shape": meta[ep["cell"]]["shape"]})
        del ep["sev_keys"]
    lags = {v: lag_summary([ep[f"lag_{v}"] for ep in eps]) for v in
            ("frozen", "gw_ema0", "gw_ema5", "l3_ema10", "l3_ema0", "override_q1", "override_q2")}

    # timelines for bursts cells (and every cell with a severe miss)
    off, done = offered_rps(calib, rows)
    # where the missed severe windows sit: after the episode's last CRITICAL (drain tail) or not,
    # the queue there, and how old the requests completing there are (the label is by completion)
    tail_stats = []
    for ep in eps:
        lo, hi = ep["t_first_sev"] - WINDOW_MS, ep["t_end"]
        crits = [base.windows[i].window_start_ms for i, w in enumerate(base.windows)
                 if w.scenario_id == ep["cell"] and base.crit[i] and lo <= w.window_start_ms <= hi]
        last = max(crits) if crits else None
        for i, w in enumerate(base.windows):
            if (w.scenario_id != ep["cell"] or w.slo_met or not (ep["t_first_viol"] <= w.window_start_ms <= hi)
                    or b_prime.severity(w) < cut or base.crit[i]):
                continue
            r = row_of[key_of(w.scenario_id, w.window_start_ms)]
            a, b = int(float(r["window_start_ms"])), int(float(r["window_end_ms"]))
            comp = [(d, s_, t) for d, s_, t in done.get(w.scenario_id, []) if a < d <= b]
            tail_stats.append({"cell": w.scenario_id, "t_start": a, "after_last_crit": last is not None and a > last,
                               "avg_waiting": fnum(r["avg_waiting"]), "avg_running": fnum(r["avg_running"]),
                               "z": base.z(i), "severity": b_prime.severity(w), "completions": len(comp),
                               "share_sent_before_window": (sum(1 for _, s_, _ in comp if s_ <= a) / len(comp)) if comp else None,
                               "median_ttft_ms": statistics.median([t for _, _, t in comp if t is not None]) if comp else None})
    n_tail = sum(1 for t in tail_stats if t["after_last_crit"] and (t["avg_waiting"] or 0) <= 15)
    sev_total = sum(c["severe"] for c in cells)
    hits_total = sum(c["crit_hits_severe"] for c in cells)
    miss_anatomy = {"missed_severe": len(tail_stats), "drain_tail_waiting_le_15": n_tail,
                    "recall_excluding_drain_tail": hits_total / (sev_total - n_tail) if sev_total - n_tail else None,
                    "episodes_caught": sum(1 for ep in eps if ep["lag_frozen"] is not None), "episodes": len(eps),
                    "windows": tail_stats}
    tl_dir = out_dir / "timelines"
    tl_dir.mkdir(parents=True, exist_ok=True)
    tl_cells = sorted({c["cell"] for c in cells if c["primitive"] == "bursts" or c["misses_severe"] > 0})
    timeline_files = {}
    for sid in tl_cells:
        path = tl_dir / f"{model}_{sid}.posthoc.csv"
        cell_rows = sorted((r for r in rows if r["scenario_id"] == sid), key=lambda r: float(r["window_start_ms"]))
        t0 = float(cell_rows[0]["window_start_ms"])
        with open(path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["# " + LABEL])
            wr.writerow(["t_s", "in_warmup", "offered_rps", "completed", "avg_running", "avg_waiting", "q1", "q2",
                         "tss_raw_gw", "tss_ema_gw", "z_raw_gw", "z_ema_gw", "z_raw_l3", "z_ema_l3", "label",
                         "severity", "crit_frozen", "crit_l3", "crit_ov1", "crit_ov2"])
            for r in cell_rows:
                k = key_of(sid, r["window_start_ms"])
                i = base.by_key.get(k)
                ir, il, ilr = raw.by_key.get(k), l3.by_key.get(k), l3raw.by_key.get(k)
                w = base.windows[i] if i is not None else None
                sev = b_prime.severity(w) if w is not None else None
                lab = "-" if w is None else ("H" if w.slo_met else ("S" if sev >= cut else "V"))
                fmt = lambda x, p=3: "" if x is None else f"{x:.{p}f}"  # noqa: E731
                wr.writerow([f"{(float(r['window_start_ms']) - t0) / 1000:.0f}", r["in_warmup"], fmt(off.get(k), 2),
                             r["completed_requests"], fmt(fnum(r["avg_running"]), 1), fmt(fnum(r["avg_waiting"]), 1),
                             fmt(q1v[i]) if i is not None else "", fmt(q2v[i], 1) if i is not None and q2v[i] is not None else "",
                             fmt(raw.windows[ir].signal, 1) if ir is not None else "",
                             fmt(w.signal, 1) if w is not None else "",
                             fmt(raw.z(ir)) if ir is not None else "", fmt(base.z(i)) if i is not None else "",
                             fmt(l3raw.z(ilr)) if ilr is not None else "", fmt(l3.z(il)) if il is not None else "",
                             lab, fmt(sev, 2), int(base.crit[i]) if i is not None else "",
                             int(l3.crit[il]) if il is not None else "", int(ov1.crit[i]) if i is not None else "",
                             int(ov2.crit[i]) if i is not None else ""])
        timeline_files[sid] = str(path)

    # power
    sev_cells = [(c["crit_hits_severe"], c["severe"]) for c in cells if c["severe"] > 0]
    pooled = sum(h for h, _ in sev_cells) / sum(s for _, s in sev_cells)
    ns = (3, 5, 8, 10, 15, 20, 30, 40, 60)
    power = {"severe_cells": sev_cells, "pooled_recall": pooled,
             "empirical": power_sim(sev_cells, ns, outer=400, inner=500, seed=seed),
             "homogeneous_at_pooled": power_sim(sev_cells, ns, outer=400, inner=500, seed=seed + 1,
                                                homogeneous_p=pooled)}

    return {"model": model, "theta": theta, "tau_crit": tau, "severity_cut": cut, "sanity_vs_accept": sanity,
            "severity_cut_hypothesis": cut_h, "miss_anatomy": miss_anatomy,
            "override_thresholds": thr, "l3": {"source": le["source"], "theta": lth, "tau_crit": ltau,
                                                "w_p": lvh["signal_spec"]["tss"]["w_p"],
                                                "lambda": lvh["signal_spec"]["tss"]["lambda_wait"],
                                                "validation_csv": str(l3_csv), "cut_record": l3_cut_rec,
                                                "windows_kept": len(l3.windows), "frozen_windows": len(base.windows)},
            "variants": results, "cells": cells, "episodes": eps, "lag": lags, "timelines": timeline_files,
            "power": power}


def analyse_t14(calib: Path, doc: dict, cfg: dict) -> dict:
    """14b on T14: per-cell confusion at Z = 1 and severe windows / hits (frozen rule)."""
    from scripts import b_prime
    from scripts.analysis import h_dropped_windows as hd

    entry = doc["models"]["dsqwen-14b"]
    vh = entry["verdict_for_holdout"]
    spec, label, trim = hd.frozen_spec_and_label(entry)
    theta, tau = float(vh["published"]["theta_m"]), float(vh["published"]["tau_crit"])
    cut = float(cfg["severity_cut"])
    path = calib / "eval/T14_score.json.d/dsqwen-14b_t14_validation.csv"
    v = Var("t14", spec.load(path, label, trim), theta, tau, vh["fit_config"]["direction"], cut=cut)
    rows = read_rows(path)
    shape = {r["scenario_id"]: (r["shape"], r["rho_factor"]) for r in rows}
    by = defaultdict(list)
    for i, w in enumerate(v.windows):
        by[w.scenario_id].append(i)
    cells = []
    for sid in sorted(by):
        idx = by[sid]
        viol = [i for i in idx if not v.windows[i].slo_met]
        ok = [i for i in idx if v.windows[i].slo_met]
        sev = [i for i in viol if b_prime.severity(v.windows[i]) >= cut]
        cells.append({"cell": sid, "shape": shape[sid][0], "rho_factor": shape[sid][1], "windows": len(idx),
                      "violating": len(viol), "severe": len(sev), "crit_hits_severe": sum(v.crit[i] for i in sev),
                      "fn_z1": sum(v.z(i) >= 1 for i in viol), "fp_z1": sum(v.z(i) < 1 for i in ok),
                      "z_viol_median": statistics.median(v.z(i) for i in viol) if viol else None,
                      "sev_viol_median": statistics.median(b_prime.severity(v.windows[i]) for i in viol) if viol else None,
                      "healthy": len(ok)})
    sev_cells = [(c["crit_hits_severe"], c["severe"]) for c in cells if c["severe"] > 0]
    return {"cells": cells, "severe_cells": sev_cells,
            "z_violating_median": statistics.median(v.z(i) for i, w in enumerate(v.windows) if not w.slo_met)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    from scripts import dline_refit as dl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--resamples", type=int, default=dl.ACCEPT_RESAMPLES)
    args = ap.parse_args(argv)
    calib, out = args.calib_root, args.out_dir
    out_json = out / "posthoc_diagnosis.posthoc.json"
    if out_json.exists():
        ap.error(f"{out_json} exists (write once)")
    freeze = calib / "freeze/params_freeze.json"
    doc = dl.verify_freeze(freeze)
    cfgs, bp_summary, problems = dl.b_prime_inputs(doc, None, dl.ONLINE_DWELL_WINDOWS, "--dwell-windows")
    if problems:
        ap.error("; ".join(problems))
    result: dict[str, Any] = {"what": LABEL, "code": dl.code_state(),
                              "freeze": {"path": str(freeze), "sha256": dl.sha256_file(freeze)},
                              "bootstrap": {"n_resamples": args.resamples, "seed": dl.SEED},
                              "override_rule": f"threshold = {OVERRIDE_TRAIN_HEALTHY_QUANTILE} quantile over training "
                                               "healthy windows; fixed before M was read by this tool",
                              "models": {}}
    for m in (args.model or MODELS):
        result["models"][m] = analyse_model(m, calib, out, doc, cfgs[m], n=args.resamples, seed=dl.SEED)
        print(f"[{m}] done", flush=True)
    result["t14_dsqwen-14b"] = analyse_t14(calib, doc, cfgs["dsqwen-14b"])
    t14c = result["t14_dsqwen-14b"]["severe_cells"]
    result["t14_dsqwen-14b"]["power_reference"] = power_sim(
        t14c, (3, 5, 8, 10, 15, 20), outer=400, inner=500, seed=dl.SEED)
    out.mkdir(parents=True, exist_ok=True)
    with open(out_json, "x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

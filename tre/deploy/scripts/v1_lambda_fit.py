#!/usr/bin/env python3
"""v1's lambda_wait / w_p selection, ported onto the v2 training windows (user 2026-09-24).

The user decided (2026-09-24) that lambda_wait is to be fitted the way v1 fitted it.
This module is a line-by-line port of the selection part of v1's
``/root/aibrix-main/python/tre/calibration/fit_tre_parameters_from_runs.py``
(``_frange`` / ``_rankdata`` / ``_spearman`` / ``_auc`` :186-235, the window health
:366-393, the signal :507-530, the objective :533-587, the tie-break :590-607 and the
three stages :1280-1344). Everything after the selection - tau, theta, delta, the labels -
stays on the v2 path (D16 / D17 / D18, ``scripts.dline_refit``): ``dline_refit wp
--lambda-method v1`` takes lambda_wait AND w_p from stage C here (v1 refined the two
jointly; the D17 w_p rule is not applied - user 2026-09-24: when v1 and D17 disagree the
v1 joint refinement wins, and the disagreement is reported).

What v1 did, item by item, and how it maps onto v2 data
--------------------------------------------------------
* unit of the Spearman: the WINDOW, pooled over every window of the model (v1 pooled the
  model's windows of every run into one list, :1255-1265). Here: the windows the v2 fit
  itself uses (``SignalSpec.load`` of the D16 fitting CSV, ramp trim 1) - same set as the
  theta fit, so the selection and the fit see the same data.
* score: the RAW signal, no EMA, no qmin floor (v1 ``trs_no_floor``, :528). Here
  ``tre_common.tss.tss_terms(..., qmin=0)`` - the v2 TSS (window token totals, no swap
  term, the replica factor is 1 on a single-replica calibration cell). A window v2 calls
  idle (nothing in flight) has no TSS and is not in the set (v1 kept it at +inf).
* p95 health: v1 ``1 / (1 + max_k p95_k / SLO_k)`` over its active SLOs (:366-393). Here
  the v2 label's own ``health_score`` = ``1 / (1 + ratio_max)`` with the v2 SLOs (D6'
  slowdown TTFT, TPOT 75 ms, e2e excluded, an unserved window graded at ratio >= 2) -
  i.e. v1's formula on v2's labels. v1 also graded e2e (12 / 15 s) and dropped windows
  without a latency sample; both differences are v2's label definition (disclosed).
* average health: v1 ``1 / (1 + mean_k avg_k / SLO_k)`` (a mean over the SLOs, not a max).
  v2 windows carry no average latency, so it is rebuilt from the requests: TTFT = mean
  over the window's served requests of ``ttft_i / ttft_slo(L_i)`` (the
  ``ttft_len_samples`` column, the same requests the p95 is taken over), TPOT = mean
  ``tpot_ms`` of the served requests completed in the window (``requests.csv`` of the
  window's dataset, same membership as the rewindow) / 75 ms. A window without either
  has no average health and - as in v1 (:541-546) - only drops out of the average term.
* objective: ``0.8 * (rho_s(score, p95 health) + 1) / 2 + 0.2 * (rho_s(score, avg
  health) + 1) / 2`` minus ``0.002 * ((w_p - 0.04) / 0.04)^2`` minus ``0.0005 *
  max(0, lambda - 1)^2`` (:533-587); ties: higher AUC of the score against the p95 label,
  then smaller lambda, then w_p closer to 0.04 (:590-607).
* stages (:1280-1344): A - lambda over 1.0 .. 4.0 step 0.25 at w_p = 0.04; B - w_p over
  0.01 .. 0.08 step 0.005 at lambda_A; C - the joint grid lambda_A +- 0.25 step 0.125 x
  w_p_B +- 0.005 step 0.0025 (clamped to the stage-A/B ranges); the published pair is
  stage C's best.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

V1_SOURCE = "/root/aibrix-main/python/tre/calibration/fit_tre_parameters_from_runs.py"
P95_WEIGHT = 0.8
AVG_WEIGHT = 0.2
LAMBDA_MIN, LAMBDA_MAX, LAMBDA_STEP = 1.0, 4.0, 0.25
W_P_MIN, W_P_MAX, W_P_STEP = 0.01, 0.08, 0.005
W_P_PRIOR_CENTER = 0.04
W_P_PRIOR_STRENGTH = 0.002
LAMBDA_PENALTY_STRENGTH = 0.0005
#: v1 scores the unfloored signal (``trs_no_floor``).
SCORE_QMIN = 0.0
V1_CONSTANTS = {
    "p95_weight": P95_WEIGHT, "avg_weight": AVG_WEIGHT,
    "lambda_grid": {"min": LAMBDA_MIN, "max": LAMBDA_MAX, "step": LAMBDA_STEP},
    "w_p_grid": {"min": W_P_MIN, "max": W_P_MAX, "step": W_P_STEP},
    "w_p_prior_center": W_P_PRIOR_CENTER, "w_p_prior_strength": W_P_PRIOR_STRENGTH,
    "lambda_wait_penalty_strength": LAMBDA_PENALTY_STRENGTH, "score_qmin": SCORE_QMIN,
    "stage_c": "lambda_A +- lambda_step at lambda_step/2 x w_p_B +- w_p_step at w_p_step/2, clamped",
}


# ------------------------------------------------------------- v1 helpers, verbatim


def frange(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    out: list[float] = []
    value = start
    while value <= stop + (step / 10.0):
        out.append(round(value, 10))
        value += step
    return out


def rankdata(values: Sequence[float]) -> list[float]:
    pairs = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(pairs):
        j = idx + 1
        while j < len(pairs) and pairs[j][1] == pairs[idx][1]:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        for k in range(idx, j):
            ranks[pairs[k][0]] = avg_rank
        idx = j
    return ranks


def spearman(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return 0.0
    rank_x = rankdata(values_x)
    rank_y = rankdata(values_y)
    mean_x = sum(rank_x) / len(rank_x)
    mean_y = sum(rank_y) / len(rank_y)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(rank_x, rank_y))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in rank_x))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in rank_y))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    positives = sum(int(v) for v in labels)
    negatives = len(labels) - positives
    if positives <= 0 or negatives <= 0:
        return 0.5
    ranks = rankdata(scores)
    pos_rank_sum = sum(rank for rank, label in zip(ranks, labels) if int(label) == 1)
    return (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def pick_better(current: Optional[dict], candidate: dict) -> dict:
    """v1 ``_pick_better_candidate`` (:590-607)."""
    if current is None:
        return candidate
    if candidate["objective_adjusted"] > current["objective_adjusted"] + 1e-12:
        return candidate
    if current["objective_adjusted"] > candidate["objective_adjusted"] + 1e-12:
        return current
    if candidate["hard_auc"] > current["hard_auc"] + 1e-12:
        return candidate
    if current["hard_auc"] > candidate["hard_auc"] + 1e-12:
        return current
    if candidate["lambda_wait"] < current["lambda_wait"] - 1e-12:
        return candidate
    if current["lambda_wait"] < candidate["lambda_wait"] - 1e-12:
        return current
    if abs(candidate["w_p"] - W_P_PRIOR_CENTER) < abs(current["w_p"] - W_P_PRIOR_CENTER) - 1e-12:
        return candidate
    return current


# ---------------------------------------------------------------------- the windows


@dataclass(frozen=True)
class V1Window:
    scenario_id: str
    window_start_ms: float
    prompt_tokens: float
    generation_tokens: float
    avg_running: float
    avg_waiting: float
    slo_met: bool
    p95_health: float
    avg_health: Optional[float]


def health(ratio: float) -> float:
    return 1.0 / (1.0 + ratio)


def score(w: V1Window, *, w_p: float, lambda_wait: float, qmin: float = SCORE_QMIN) -> float:
    """The raw v2 TSS of one window (no EMA; ``qmin`` 0 = v1's no-floor score)."""
    from tre_common.tss import tss_terms

    raw = tss_terms(prompt_tokens=w.prompt_tokens, generation_tokens=w.generation_tokens,
                    avg_running=w.avg_running, avg_waiting=w.avg_waiting, w_p=w_p,
                    lambda_wait=lambda_wait, qmin=qmin).raw
    return math.inf if raw is None else float(raw)


class RequestIndex:
    """Served requests' (done_ts_ms, tpot_ms) per (model, cell_id, attempt), from the
    ``requests.csv`` of standard datasets (one per training source)."""

    def __init__(self) -> None:
        self._by: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
        self.sources: dict[str, str] = {}

    def add_dataset(self, run: str, directory: Path, model: Optional[str] = None) -> int:
        from scripts.rewindow_from_raw import OUTCOME_OK

        path = Path(directory) / "requests.csv"
        n = 0
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if model is not None and r.get("model") != model:
                    continue
                # the dataset's own classification (rewindow_from_raw.is_served on the raw)
                if r.get("outcome") != OUTCOME_OK:
                    continue
                done, tpot = r.get("done_ts_ms"), r.get("tpot_ms")
                if not done or not tpot:
                    continue
                self._by[(run, r["model"], r["cell_id"], str(r.get("attempt") or "1"))].append(
                    (float(done), float(tpot)))
                n += 1
        for v in self._by.values():
            v.sort()
        self.sources[run] = str(path)
        return n

    def tpots(self, run: str, model: str, cell_id: str, attempt: str, start_ms: float, end_ms: float,
              *, closed_right: bool) -> list[float]:
        rows = self._by.get((run, model, cell_id, str(attempt or "1")), [])
        keys = [d for d, _ in rows]
        if closed_right:  # start < done <= end
            lo, hi = bisect.bisect_right(keys, start_ms), bisect.bisect_right(keys, end_ms)
        else:  # start <= done < end
            lo, hi = bisect.bisect_left(keys, start_ms), bisect.bisect_left(keys, end_ms)
        return [t for _, t in rows[lo:hi]]

    def count(self, *a, **kw) -> int:
        return len(self.tpots(*a, **kw))


def sources_from_trainset(fit_dir: Path) -> dict[str, Path]:
    """run -> dataset directory, from the fit dir's ``trainset.json`` (D16 provenance)."""
    man = Path(fit_dir) / "trainset.json"
    if not man.exists():
        return {}
    doc = json.loads(man.read_text(encoding="utf-8"))
    return {s["run"]: Path(s["directory"]) for s in doc.get("sources", [])}


def load_windows(model: str, fitting_csv: Path, label, *, trim: int,
                 requests: Optional[RequestIndex]) -> tuple[list[V1Window], dict]:
    """The v2 fit's windows of ``fitting_csv`` as v1 records (see the module docstring)."""
    from scripts import theta_verdict as tv
    from tre_common import slo_labels

    spec = tv.build_signal_spec("tss", w_p=W_P_PRIOR_CENTER, lambda_wait=1.0, qmin=SCORE_QMIN,
                                ema_tau_ms=None)
    base = spec.load(fitting_csv, label, trim)
    rows: dict[tuple, dict] = {}
    with open(fitting_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r["scenario_id"], float(r["window_start_ms"]))
            if key in rows:
                raise ValueError(f"{fitting_csv}: two rows for window {key}")
            rows[key] = r
    stats = {"windows": 0, "avg_health": 0, "avg_ttft_only": 0, "unserved_no_latency": 0,
             "tpot_membership": {"closed_right_match": 0, "closed_left_match": 0, "checked": 0},
             "requests_missing": 0}
    out: list[V1Window] = []
    for w in base:
        r = rows[(w.scenario_id, float(w.window_start_ms))]
        start, end = float(r["window_start_ms"]), float(r["window_end_ms"])
        pairs = slo_labels.parse_ttft_len_samples(r.get(slo_labels.TTFT_LEN_SAMPLES_COLUMN))
        ttft_avg = (sum(t / label.ttft_slo_ms(L) for t, L in pairs) / len(pairs)) if pairs else None
        tpot_avg = None
        if requests is not None:
            args = (r.get("run", ""), model, r["cell_id"], r.get("attempt") or "1", start, end)
            right = requests.tpots(*args, closed_right=True)
            n_done = slo_labels.completed_requests(r)
            if n_done is not None:
                stats["tpot_membership"]["checked"] += 1
                stats["tpot_membership"]["closed_right_match"] += int(len(right) == n_done)
                stats["tpot_membership"]["closed_left_match"] += int(
                    requests.count(*args, closed_right=False) == n_done)
            if right:
                tpot_avg = sum(right) / len(right)
            elif n_done:
                stats["requests_missing"] += 1
        ratios = [x for x in (ttft_avg, None if tpot_avg is None else tpot_avg / float(label.tpot_p95_ms))
                  if x is not None]
        avg_h = health(sum(ratios) / len(ratios)) if ratios else None
        stats["avg_health"] += avg_h is not None
        stats["avg_ttft_only"] += (ttft_avg is not None and tpot_avg is None)
        stats["unserved_no_latency"] += (w.violation_class == "unserved" and not pairs)
        out.append(V1Window(
            scenario_id=w.scenario_id, window_start_ms=float(w.window_start_ms),
            prompt_tokens=float(r.get("prompt_tokens_total") or 0.0),
            generation_tokens=float(r.get("generation_tokens_total") or 0.0),
            avg_running=float(r.get("avg_running") or 0.0), avg_waiting=float(r.get("avg_waiting") or 0.0),
            slo_met=bool(w.slo_met), p95_health=float(w.health_score), avg_health=avg_h))
    stats["windows"] = len(out)
    stats["violating"] = sum(1 for w in out if not w.slo_met)
    # self-check: the recomputed score is the loader's signal
    for w, b in zip(out, base):
        s = score(w, w_p=W_P_PRIOR_CENTER, lambda_wait=1.0)
        if not math.isclose(s, float(b.signal), rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError(f"score recompute disagrees with the loader at {w.scenario_id} "
                                 f"{w.window_start_ms}: {s} != {b.signal}")
    return out, stats


# ------------------------------------------------------------------- the objective


def evaluate(windows: Sequence[V1Window], *, w_p: float, lambda_wait: float) -> dict:
    """v1 ``_evaluate_raw_parameters`` + ``_score_signal`` (:533-587)."""
    scores = [score(w, w_p=w_p, lambda_wait=lambda_wait) for w in windows]
    p95 = (spearman(scores, [w.p95_health for w in windows]) + 1.0) / 2.0
    pairs = [(s, w.avg_health) for s, w in zip(scores, windows) if w.avg_health is not None]
    avg = 0.5
    if len(pairs) >= 2:
        avg = (spearman([s for s, _ in pairs], [h for _, h in pairs]) + 1.0) / 2.0
    hard_auc = auc(scores, [1 if w.slo_met else 0 for w in windows])
    objective = P95_WEIGHT * p95 + AVG_WEIGHT * avg
    w_p_pen = W_P_PRIOR_STRENGTH * ((w_p - W_P_PRIOR_CENTER) / max(W_P_PRIOR_CENTER, 1e-9)) ** 2
    lam_pen = LAMBDA_PENALTY_STRENGTH * max(0.0, lambda_wait - 1.0) ** 2
    return {"w_p": w_p, "lambda_wait": lambda_wait, "objective": objective,
            "objective_adjusted": objective - w_p_pen - lam_pen, "p95_score": p95, "avg_score": avg,
            "hard_auc": hard_auc, "w_p_penalty": w_p_pen, "lambda_wait_penalty": lam_pen,
            "avg_pairs": len(pairs)}


def search(windows: Sequence[V1Window]) -> dict:
    """v1 stages A, B, C (:1280-1344)."""
    stage_a, best_a = [], None
    for lam in frange(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_STEP):
        c = evaluate(windows, w_p=W_P_PRIOR_CENTER, lambda_wait=lam)
        stage_a.append(c)
        best_a = pick_better(best_a, c)
    stage_b, best_b = [], None
    for wp in frange(W_P_MIN, W_P_MAX, W_P_STEP):
        c = evaluate(windows, w_p=wp, lambda_wait=float(best_a["lambda_wait"]))
        stage_b.append(c)
        best_b = pick_better(best_b, c)
    lam_lo = max(LAMBDA_MIN, float(best_a["lambda_wait"]) - LAMBDA_STEP)
    lam_hi = min(LAMBDA_MAX, float(best_a["lambda_wait"]) + LAMBDA_STEP)
    wp_lo = max(W_P_MIN, float(best_b["w_p"]) - W_P_STEP)
    wp_hi = min(W_P_MAX, float(best_b["w_p"]) + W_P_STEP)
    stage_c, best_c = [], None
    for lam in frange(lam_lo, lam_hi, LAMBDA_STEP / 2.0):
        for wp in frange(wp_lo, wp_hi, W_P_STEP / 2.0):
            c = evaluate(windows, w_p=wp, lambda_wait=lam)
            stage_c.append(c)
            best_c = pick_better(best_c, c)
    best = best_c or best_b
    a_obj = [c["objective"] for c in stage_a]
    return {
        "stage_a": stage_a, "best_a": best_a, "stage_b": stage_b, "best_b": best_b,
        "stage_c": stage_c, "best_c": best_c,
        "lambda_wait": float(best["lambda_wait"]), "w_p": float(best["w_p"]),
        "flatness": {
            "stage_a_objective_range": max(a_obj) - min(a_obj),
            "stage_a_penalty_at_lambda_4": LAMBDA_PENALTY_STRENGTH * 9.0,
            "note": ("when the unpenalised stage-A objective varies less than the lambda "
                     "penalty over the grid, lambda is set by the penalty (the prior), not the data"),
        },
    }


def fit_model(model: str, label, fitting_csv: Path, *, trim: int,
              sources: Mapping[str, Path]) -> dict:
    """The v1 selection on one model's D16 fitting CSV; ``sources`` run -> dataset dir
    (the requests.csv the average TPOT is rebuilt from)."""
    req = RequestIndex()
    for run, d in sorted(sources.items()):
        if (Path(d) / "requests.csv").exists():
            req.add_dataset(run, Path(d), model)
    windows, stats = load_windows(model, Path(fitting_csv), label, trim=trim, requests=req)
    res = search(windows)
    mem = stats["tpot_membership"]
    if mem["checked"] and mem["closed_right_match"] < 0.99 * mem["checked"]:
        raise AssertionError(f"{model}: average-TPOT window membership disagrees with "
                             f"completed_requests ({mem})")
    return {
        "method": "v1 (fit_tre_parameters_from_runs.py stages A-C), ported onto v2 windows",
        "v1_source": V1_SOURCE, "constants": V1_CONSTANTS, "fitting_csv": str(fitting_csv),
        "trim_ramp_windows": trim, "request_sources": req.sources, "window_stats": stats,
        **res,
    }


# ------------------------------------------------------- B' (severity-aligned CRITICAL)


def severity(w) -> float:
    """Severity of one v2 window, as ``tre_calibration.fit.fit_delta_margins`` grades it:
    0.8 * p95 ratio + 0.2 * average ratio, the average falling back to the p95 ratio
    (always so under the slowdown label, whose windows carry no average ratio)."""
    p95 = w.latency_ratio_p95 if w.latency_ratio_p95 is not None else (1.0 / w.health_score) - 1.0
    avg = w.latency_ratio_avg if w.latency_ratio_avg is not None else p95
    return 0.8 * p95 + 0.2 * avg


def severity_cut(train_windows: Sequence[Any], q: float = 0.65) -> float:
    """The training set's severity quantile of the violating windows - the cut
    ``fit_delta_margins`` labels its critical windows with (same ``_quantile``)."""
    from tre_calibration.fit import _quantile

    sev = [severity(w) for w in train_windows if math.isfinite(w.signal) and not w.slo_met]
    cut = _quantile(sev, q)
    if cut is None:
        raise ValueError("no violating training window")
    return float(cut)


def b_prime_point(windows: Sequence[Any], *, theta: float, tau_crit: float, cut: float,
                  crit0: Sequence[bool], crit2: Sequence[bool]) -> dict:
    """B' on one window list: CRITICAL recall of violations with severity >= ``cut``
    (no dwell / dwell 2), healthy false alarm, all-violation recall, and where the
    violations fall (CRITICAL / LOW band tau_crit <= Z < 1 / Z >= 1)."""
    def rate(sel, flags):
        return (sum(1 for i in sel if flags[i]) / len(sel)) if sel else None

    idx = [i for i, w in enumerate(windows) if math.isfinite(w.signal)]
    viol = [i for i in idx if not windows[i].slo_met]
    ok = [i for i in idx if windows[i].slo_met]
    sev = [i for i in viol if severity(windows[i]) >= cut]
    z = {i: windows[i].signal / theta for i in idx}
    missed = [i for i in viol if not crit0[i]]
    low = [i for i in viol if tau_crit <= z[i] < 1.0]
    return {
        "violating": len(viol), "healthy": len(ok), "severe_windows": len(sev), "severity_cut": cut,
        "nodwell": {"recall_severe": rate(sev, crit0), "false_alarm": rate(ok, crit0),
                    "recall_all": rate(viol, crit0)},
        "dwell2": {"recall_severe": rate(sev, crit2), "false_alarm": rate(ok, crit2),
                   "recall_all": rate(viol, crit2)},
        "violations_by_band": {
            "critical": (sum(1 for i in viol if z[i] < tau_crit) / len(viol)) if viol else None,
            "low": (len(low) / len(viol)) if viol else None,
            "healthy_side_z_ge_1": (sum(1 for i in viol if z[i] >= 1.0) / len(viol)) if viol else None,
        },
        "missed_caught_by_slow_loop": (sum(1 for i in missed if z[i] < 1.0) / len(missed)) if missed else None,
    }


def b_prime_boot(windows: Sequence[Any], *, theta: float, tau_crit: float, cut: float,
                 crit: Sequence[bool], n: int = 1000, seed: int = 20260922) -> dict:
    """Cell bootstrap (scenario ids with replacement) CI95 of the B' recall / false alarm
    for one flag series."""
    import random

    by: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(windows):
        if math.isfinite(w.signal):
            by[w.scenario_id].append(i)
    cells = sorted(by)
    cnt = {}
    for c in cells:
        sev = [i for i in by[c] if not windows[i].slo_met and severity(windows[i]) >= cut]
        ok = [i for i in by[c] if windows[i].slo_met]
        cnt[c] = (sum(crit[i] for i in sev), len(sev), sum(crit[i] for i in ok), len(ok))
    rec, fa = [], []
    rng = random.Random(seed)
    for _ in range(n if cells else 0):
        pick = [rng.choice(cells) for _ in cells]
        a = [sum(cnt[c][k] for c in pick) for k in range(4)]
        if a[1]:
            rec.append(a[0] / a[1])
        if a[3]:
            fa.append(a[2] / a[3])

    def ci(v):
        v = sorted(v)
        return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]] if v else [None, None]

    return {"recall_severe_ci95": ci(rec), "false_alarm_ci95": ci(fa), "n": n, "seed": seed}


def main(argv: Optional[Sequence[str]] = None) -> int:
    from scripts import dline_refit as dr

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fit-dir", type=Path, required=True)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--arm", default="primary")
    ap.add_argument("--requests-dataset", action="append", default=[], metavar="RUN=DIR")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    label = dr.label_for(args.model, args.arm, args.registry)
    sources = sources_from_trainset(args.fit_dir)
    for t in args.requests_dataset:
        run, _, d = t.partition("=")
        sources[run] = Path(d)
    doc = fit_model(args.model, label, dr.paths(args.fit_dir, args.model)["fitting"],
                    trim=dr.TRIM_RAMP_WINDOWS, sources=sources)
    args.out.write_text(json.dumps(doc, indent=1, default=str))
    print(f"{args.model}: v1 lambda_wait={doc['lambda_wait']:g} w_p={doc['w_p']:g} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

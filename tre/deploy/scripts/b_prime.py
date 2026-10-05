"""B' - the severity-aligned CRITICAL criterion (user 2026-09-24; the acceptance gate since
2026-10-03).

B' replaces the old criterion B (CRITICAL recall of both / TPOT-only violations) as the
gate of ``dline_refit accept``: the CRITICAL recall of the violating windows whose
*severity* is at least the model's TRAINING .65 quantile of violating-window severity,
and the CRITICAL false alarm on healthy windows, both with their cell-bootstrap CI95.

* severity of a window = ``0.8 * p95 ratio + 0.2 * average ratio`` - the grading of
  ``tre_calibration.fit.fit_delta_margins`` (its own critical cut is the same .65
  quantile); the average falls back to the p95 ratio (always so under the D6' slowdown
  label, whose windows carry no average ratio);
* the cut comes from the training windows only, and is sealed before any test window is
  read: ``dline_refit freeze`` computes it from the freeze's training CSV and writes it
  into the freeze (format revision 3); a freeze that predates that takes it from an
  explicit thresholds file (``--b-prime-thresholds``, e.g. the 2026-09-24
  ``b_prime_thresholds.json``: ``severity_cut_train`` + ``gate``);
* CRITICAL = ``Z < tau_crit`` dwell-confirmed over ``dwell_windows`` new windows
  (``theta_verdict.critical_dwell_flags`` = the controller's ``tre_common.dwell``). The
  gate is judged at the dwell the controller runs with (``TRE_DWELL_WINDOWS``, 1 since the
  v1 alignment); dwell 2 and no dwell are disclosed next to it.

Used by ``scripts.dline_refit`` (accept: the gate; freeze: the cut) and by
``scripts.v1_lambda_compare`` (disclosure tables, through ``scripts.v1_lambda_fit``).
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

#: The training quantile of violating-window severity B' recalls above.
SEVERITY_QUANTILE = 0.65
SEVERITY_RULE = ("0.8 * p95 ratio + 0.2 * average ratio (tre_calibration.fit.fit_delta_margins "
                 "grading; the average falls back to the p95 ratio)")
#: The gate (b_prime_thresholds.json of 2026-09-24, decided before any test number of B').
GATE_KEYS = ("recall_severe_min", "recall_severe_ci95_low_min", "false_alarm_max", "false_alarm_ci95_high_max")
DEFAULT_GATE = {"recall_severe_min": 0.80, "recall_severe_ci95_low_min": 0.70,
                "false_alarm_max": 0.05, "false_alarm_ci95_high_max": 0.08}
#: Dwell always disclosed next to the gate's own (the D22 deployed form).
DISCLOSED_DWELL_WINDOWS = (1, 2)


def severity(w) -> float:
    """Severity of one v2 window (:data:`SEVERITY_RULE`)."""
    p95 = w.latency_ratio_p95 if w.latency_ratio_p95 is not None else (1.0 / w.health_score) - 1.0
    avg = w.latency_ratio_avg if w.latency_ratio_avg is not None else p95
    return 0.8 * p95 + 0.2 * avg


def violating_severities(train_windows: Sequence[Any]) -> list[float]:
    return [severity(w) for w in train_windows if math.isfinite(w.signal) and not w.slo_met]


def severity_cut(train_windows: Sequence[Any], q: float = SEVERITY_QUANTILE) -> float:
    """The training set's severity quantile of the violating windows - the cut
    ``fit_delta_margins`` labels its critical windows with (same ``_quantile``)."""
    from tre_calibration.fit import _quantile

    cut = _quantile(violating_severities(train_windows), q)
    if cut is None:
        raise ValueError("no violating training window: the severity cut is undefined")
    return float(cut)


def _rate(sel: Sequence[int], flags: Sequence[bool]) -> Optional[float]:
    return (sum(1 for i in sel if flags[i]) / len(sel)) if sel else None


def _sets(windows: Sequence[Any], cut: float):
    idx = [i for i, w in enumerate(windows) if math.isfinite(w.signal)]
    viol = [i for i in idx if not windows[i].slo_met]
    ok = [i for i in idx if windows[i].slo_met]
    sev = [i for i in viol if severity(windows[i]) >= cut]
    return idx, viol, ok, sev


def series_point(windows: Sequence[Any], *, theta: float, cut: float, crit: Sequence[bool]) -> dict:
    """One CRITICAL flag series: recall of the severe violations, healthy false alarm,
    all-violation recall, and the share of missed violations the slow loop sees (Z < 1)."""
    idx, viol, ok, sev = _sets(windows, cut)
    missed = [i for i in viol if not crit[i]]
    return {"recall_severe": _rate(sev, crit), "false_alarm": _rate(ok, crit), "recall_all": _rate(viol, crit),
            "missed_caught_by_slow_loop": (sum(1 for i in missed if windows[i].signal / theta < 1.0) / len(missed))
            if missed else None}


def band_shares(windows: Sequence[Any], *, theta: float, tau_crit: float, cut: float) -> dict:
    """Window counts and where the violations fall (CRITICAL / LOW band tau_crit <= Z < 1 /
    Z >= 1); independent of the dwell."""
    idx, viol, ok, sev = _sets(windows, cut)
    z = {i: windows[i].signal / theta for i in idx}
    share = (lambda n: (n / len(viol)) if viol else None)
    return {"violating": len(viol), "healthy": len(ok), "severe_windows": len(sev), "severity_cut": cut,
            "violations_by_band": {"critical": share(sum(1 for i in viol if z[i] < tau_crit)),
                                   "low": share(sum(1 for i in viol if tau_crit <= z[i] < 1.0)),
                                   "healthy_side_z_ge_1": share(sum(1 for i in viol if z[i] >= 1.0))}}


def b_prime_point(windows: Sequence[Any], *, theta: float, tau_crit: float, cut: float,
                  crit0: Sequence[bool], crit2: Sequence[bool]) -> dict:
    """B' on one window list, no dwell and dwell 2 (the disclosure shape of
    ``v1_lambda_compare``)."""
    p0 = series_point(windows, theta=theta, cut=cut, crit=crit0)
    p2 = series_point(windows, theta=theta, cut=cut, crit=crit2)
    keys = ("recall_severe", "false_alarm", "recall_all")
    return {**{k: v for k, v in band_shares(windows, theta=theta, tau_crit=tau_crit, cut=cut).items()},
            "nodwell": {k: p0[k] for k in keys}, "dwell2": {k: p2[k] for k in keys},
            "missed_caught_by_slow_loop": p0["missed_caught_by_slow_loop"]}


def b_prime_boot(windows: Sequence[Any], *, theta: float = 0.0, tau_crit: float = 0.0, cut: float,
                 crit: Sequence[bool], n: int = 1000, seed: int = 20260922) -> dict:
    """Cell bootstrap (scenario ids with replacement) CI95 of the B' recall / false alarm
    for one flag series (``theta`` / ``tau_crit`` unused, kept for the old call sites)."""
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


def check_gate(gate: Mapping[str, Any]) -> dict:
    """``gate`` with exactly the :data:`GATE_KEYS`, each a finite number in [0, 1]; raises
    ``ValueError`` otherwise."""
    out = {}
    for k in GATE_KEYS:
        v = gate.get(k) if isinstance(gate, Mapping) else None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
            raise ValueError(f"B' gate {k!r} is {v!r}: needs a number in [0, 1]")
        out[k] = float(v)
    return out


def load_thresholds(path: Path) -> dict:
    """An explicit B' thresholds file (``severity_cut_train`` {model: cut} + ``gate``),
    for a freeze that predates the sealed cut. Raises ``ValueError`` on anything else."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: unreadable ({exc})")
    cuts = doc.get("severity_cut_train") if isinstance(doc, dict) else None
    if not isinstance(cuts, dict) or not cuts:
        raise ValueError(f"{path}: no severity_cut_train {{model: cut}}")
    for m, c in cuts.items():
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or c <= 0:
            raise ValueError(f"{path}: severity_cut_train[{m!r}] = {c!r} is not a positive number")
    return {"severity_cut_train": {m: float(c) for m, c in cuts.items()}, "gate": check_gate(doc.get("gate") or {}),
            "doc": doc}


def criteria(point: Mapping[str, Any], ci: Mapping[str, Any], gate: Mapping[str, float]) -> list[dict]:
    """The four B' criteria of one flag series (``point`` from :func:`series_point`,
    ``ci`` from :func:`b_prime_boot`)."""
    def crit(name, value, op, threshold):
        v = value if isinstance(value, (int, float)) and math.isfinite(value) else None
        met = v is not None and (v >= threshold if op == ">=" else v <= threshold)
        return {"name": name, "value": v, "op": op, "threshold": threshold, "met": met}

    return [crit("CRITICAL recall of severe violations", point["recall_severe"], ">=", gate["recall_severe_min"]),
            crit("its CI95 lower bound", ci["recall_severe_ci95"][0], ">=", gate["recall_severe_ci95_low_min"]),
            crit("CRITICAL false alarm on healthy windows", point["false_alarm"], "<=", gate["false_alarm_max"]),
            crit("its CI95 upper bound", ci["false_alarm_ci95"][1], "<=", gate["false_alarm_ci95_high_max"])]


# ------------------------------------------------------- the onset gate (2026-10-05)
#
# The next-round accept gate (design docs/calib-next-round-design-20261005.md item 3, user
# 2026-10-05). B' scored every severe WINDOW; under completion attribution the drain tail
# after a burst kept windows "severe" after the engine had recovered, so B' measured the
# label, not the controller. The onset gate scores overload EPISODES instead:
#
# * episode = a maximal run of consecutive (<= one re-window step apart) violating windows
#   of one cell holding >= 1 severe window (severity >= the sealed training cut);
# * detected = a dwell-confirmed CRITICAL window inside the episode's detection window
#   (t_lo, t_end], t_lo = max(t_first_severe - lookback, t_end_prev) with t_end_prev the last
#   window of the previous episode of the same cell, EXCLUDED (coordinator 2026-10-05: a
#   CRITICAL of an earlier episode is never credited to the next; a CRITICAL run that simply
#   continues counts only if it is still active after t_end_prev); lag = t_first_crit -
#   t_first_severe (negative = CRITICAL led the label);
# * success = detected with lag <= the lag budget (2 ticks = 20 s, derived from the EMA
#   alpha .632, tau_crit and window filling - not read off M);
# * gated on the onset episodes of the DYNAMIC cells (steps / ramp / bursts), whose
#   overload has an onset; hold-cell episodes are disclosed;
# * at most miss_tolerance episodes may fail, and the Clopper-Pearson lower bound of the
#   success rate on the effective number of episodes n / (1 + (m - 1) ICC), ICC = the
#   one-way ICC(1) of the per-cell lags (challenge_checks.posthoc.json's estimator;
#   undefined -> 1, the conservative end; negative -> 0), must reach detection_ci_low_min;
#   a cell-cluster bootstrap bound is disclosed (it never decides with miss_tolerance 0).
#
# Every parameter lives in :data:`ONSET_GATE` and is sealed in the freeze (``accept_gate``).

#: Primitives whose overload has an onset: the episodes the onset gate is judged on.
ONSET_DYNAMIC_PRIMITIVES = ("steps", "ramp", "bursts")
ONSET_GATE = {
    "lag_budget_s": 20.0,             # 2 controller ticks (10 s refresh)
    "lookback_s": 30.0,               # one window before the first severe window
    "episode_step_ms": 10_000.0,      # consecutive = <= one re-window step apart
    "detection_ci_low_min": 0.80,
    "ci_alpha_one_sided": 0.05,       # both lower bounds are one-sided 95 % (design: n_eff 14, 0 misses -> .81)
    "miss_tolerance": 0,              # design item 3 (i): every onset is caught
}
ONSET_GATE_KEYS = tuple(ONSET_GATE)
#: Window false alarm (design item 3 (ii)): CRITICAL on healthy windows at the gate's dwell.
WINDOW_FA_GATE = {"false_alarm_max": 0.05, "false_alarm_ci95_high_max": 0.08}
ICC_UNDEFINED_VALUE = 1.0
ONSET_RULE = ("episode = maximal run of consecutive violating windows (<= episode_step_ms apart) of one cell "
              "with >= 1 window of severity >= the sealed training cut; detection window = window starts t with "
              "t_first_severe - lookback_s <= t <= t_end and t > t_end_prev, t_end_prev = the last window of the "
              "previous episode of the same cell (none: no such bound) - a CRITICAL of an earlier episode is "
              "never credited to the next one, a CRITICAL run that continues counts only if still active after "
              "t_end_prev; success = a dwell-confirmed CRITICAL in the detection window with t_first_crit - "
              "t_first_severe <= lag_budget_s; gated on the episodes of cells whose primitive is in "
              "ONSET_DYNAMIC_PRIMITIVES")
#: Sealed with the onset parameters: how the look-back is clipped (coordinator 2026-10-05).
ONSET_LOOKBACK_CLIP = "previous_episode_end_exclusive"
ONSET_CI_RULE = ("decides: at most miss_tolerance onset episodes missed (or later than the lag budget) AND the "
                 "one-sided (ci_alpha_one_sided) Clopper-Pearson lower bound of the success rate on n_eff = "
                 "n / (1 + (m - 1) ICC) reaches detection_ci_low_min (m = episodes per cell, ICC = one-way ICC(1) "
                 "of the per-cell lags of detected episodes, undefined -> 1, negative -> 0; x_eff = rate * "
                 "n_eff). The cell-cluster bootstrap percentile lower bound is disclosed, not deciding (with "
                 "miss_tolerance 0 it is 1 whenever the miss criterion passes)")


def check_onset_gate(gate: Mapping[str, Any]) -> dict:
    """``gate`` with exactly :data:`ONSET_GATE_KEYS`, each a finite number in range;
    raises ``ValueError`` otherwise."""
    out = {}
    for k in ONSET_GATE_KEYS:
        v = gate.get(k) if isinstance(gate, Mapping) else None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"onset gate {k!r} is {v!r}: needs a finite number >= 0")
        out[k] = float(v)
    if not 0 < out["ci_alpha_one_sided"] < 0.5 or out["detection_ci_low_min"] > 1:
        raise ValueError(f"onset gate {out}: ci_alpha_one_sided in (0, .5), detection_ci_low_min <= 1")
    out["miss_tolerance"] = int(out["miss_tolerance"])
    return out


def episodes(windows: Sequence[Any], cut: float, *, step_ms: float = ONSET_GATE["episode_step_ms"]) -> list[dict]:
    """Severe-violation episodes per cell (the rule of :data:`ONSET_RULE`; the post-hoc
    ``posthoc_hfail_20261005.episodes``). Window instants are ``window_start_ms``."""
    by: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(windows):
        by[w.scenario_id].append(i)
    out = []
    for sid, idx in sorted(by.items()):
        idx.sort(key=lambda i: windows[i].window_start_ms)
        k = 0
        while k < len(idx):
            if windows[idx[k]].slo_met:
                k += 1
                continue
            s = k
            while (k + 1 < len(idx) and not windows[idx[k + 1]].slo_met
                   and windows[idx[k + 1]].window_start_ms - windows[idx[k]].window_start_ms <= step_ms):
                k += 1
            run = idx[s:k + 1]
            k += 1
            sev = [i for i in run if severity(windows[i]) >= cut]
            if sev:
                prev = next((e["t_end"] for e in reversed(out) if e["cell"] == sid), None)
                out.append({"cell": sid, "t_end_prev": prev, "t_first_viol": windows[run[0]].window_start_ms,
                            "t_first_severe": windows[sev[0]].window_start_ms,
                            "t_end": windows[run[-1]].window_start_ms, "n_violating": len(run), "n_severe": len(sev)})
    return out


def episode_lag_s(windows: Sequence[Any], crit: Sequence[bool], ep: Mapping[str, Any], *,
                  lookback_s: float = ONSET_GATE["lookback_s"]) -> Optional[float]:
    """Seconds from the episode's first severe window to the first dwell-confirmed CRITICAL
    window of its cell in the detection window of :data:`ONSET_RULE` (``ep["t_end_prev"]``:
    the previous episode's last window, None = none); None = missed."""
    lo, hi, prev = ep["t_first_severe"] - lookback_s * 1000.0, ep["t_end"], ep.get("t_end_prev")
    ts = [w.window_start_ms for i, w in enumerate(windows)
          if crit[i] and w.scenario_id == ep["cell"] and lo <= w.window_start_ms <= hi
          and (prev is None or w.window_start_ms > prev)]
    return (min(ts) - ep["t_first_severe"]) / 1000.0 if ts else None


def lag_icc(groups: Sequence[Sequence[float]]) -> Optional[float]:
    """One-way ICC(1) of the lags grouped by cell (the challenge_checks estimator, unbalanced
    n0); None when undefined (one group, no within-group replicate, or zero variance)."""
    groups = [list(g) for g in groups if g]
    allv = [x for g in groups for x in g]
    if len(groups) < 2 or len(allv) <= len(groups):
        return None
    gm = sum(allv) / len(allv)
    means = [sum(g) / len(g) for g in groups]
    msb = sum(len(g) * (m - gm) ** 2 for g, m in zip(groups, means)) / (len(groups) - 1)
    msw = sum((x - m) ** 2 for g, m in zip(groups, means) for x in g) / (len(allv) - len(groups))
    n0 = (len(allv) - sum(len(g) ** 2 for g in groups) / len(allv)) / (len(groups) - 1)
    den = msb + (n0 - 1) * msw
    return (msb - msw) / den if den > 0 else None


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularised incomplete beta (modified Lentz)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def regularized_beta(x: float, a: float, b: float) -> float:
    """I_x(a, b) for a, b > 0 (no scipy on the test hosts' import path)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def clopper_pearson_lower(x: float, n: float, alpha: float) -> Optional[float]:
    """One-sided (1 - alpha) Clopper-Pearson lower bound of a rate with ``x`` successes in
    ``n`` trials, non-integer (effective) counts allowed: the p with I_p(x, n - x + 1) =
    alpha; 0 for x = 0, alpha ** (1 / n) for x = n. None for n <= 0."""
    if not n > 0:
        return None
    x = min(max(float(x), 0.0), float(n))
    if x <= 0.0:
        return 0.0
    if x >= n:
        return alpha ** (1.0 / n)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if regularized_beta(mid, x, n - x + 1.0) > alpha:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def onset_detection(windows: Sequence[Any], crit: Sequence[bool], *, cut: float,
                    primitive_of: Mapping[str, str], gate: Mapping[str, Any], n_resamples: int = 1000,
                    seed: int = 20260922) -> dict:
    """The onset gate (:data:`ONSET_RULE`, :data:`ONSET_CI_RULE`) of one flag series.
    ``primitive_of`` maps a scenario id to its cell's primitive. Not evaluable (and not
    passed) without an episode in a dynamic cell."""
    g = check_onset_gate(gate)
    budget, alpha = g["lag_budget_s"], g["ci_alpha_one_sided"]
    eps = episodes(windows, cut, step_ms=g["episode_step_ms"])
    starts: dict[str, set] = defaultdict(set)
    for w in windows:
        starts[w.scenario_id].add(w.window_start_ms)
    for ep in eps:
        ep["primitive"] = primitive_of.get(ep["cell"], "")
        ep["lag_s"] = episode_lag_s(windows, crit, ep, lookback_s=g["lookback_s"])
        ep["success"] = ep["lag_s"] is not None and ep["lag_s"] <= budget
        # disclosure (review 2026-10-05 P3-7): the lag from the first VIOLATING window, and
        # whether the window right before the episode is missing from the labelled set (an
        # unlabelled / dropped window: the min-n guard can delay where an onset is seen)
        ep["lag_from_first_violating_s"] = (None if ep["lag_s"] is None else
                                            ep["lag_s"] + (ep["t_first_severe"] - ep["t_first_viol"]) / 1000.0)
        before = ep["t_first_viol"] - g["episode_step_ms"]
        ep["preceded_by_unlabelled_window"] = (before >= min(starts[ep["cell"]])
                                               and before not in starts[ep["cell"]])
    onset = [ep for ep in eps if ep["primitive"] in ONSET_DYNAMIC_PRIMITIVES]
    hold = [ep for ep in eps if ep["primitive"] not in ONSET_DYNAMIC_PRIMITIVES]

    def summary(sel):
        lags = [ep["lag_s"] for ep in sel if ep["lag_s"] is not None]
        return {"episodes": len(sel), "detected": len(lags), "within_budget": sum(ep["success"] for ep in sel),
                "lag_max_s": max(lags) if lags else None,
                "lag_median_s": sorted(lags)[len(lags) // 2] if lags else None}

    out: dict[str, Any] = {"rule": ONSET_RULE, "ci_rule": ONSET_CI_RULE, "gate": g, "severity_cut": cut,
                           "dynamic_primitives": list(ONSET_DYNAMIC_PRIMITIVES),
                           "onset": summary(onset), "hold_disclosed": summary(hold),
                           "lookback_clip": ONSET_LOOKBACK_CLIP,
                           "episodes": [{k: ep[k] for k in ("cell", "primitive", "t_end_prev", "t_first_viol",
                                                            "t_first_severe",
                                                            "t_end", "n_violating", "n_severe", "lag_s", "success",
                                                            "lag_from_first_violating_s",
                                                            "preceded_by_unlabelled_window")}
                                        for ep in eps]}
    n = len(onset)
    if not n:
        out.update({"evaluable": False, "passed": False, "criteria": [],
                    "reason": "no severe-violation episode in a dynamic cell"})
        return out
    by: dict[str, list[dict]] = defaultdict(list)
    for ep in onset:
        by[ep["cell"]].append(ep)
    cells = sorted(by)
    succ = sum(ep["success"] for ep in onset)
    rate = succ / n
    rng = random.Random(seed)
    vals = []
    for _ in range(n_resamples):
        pick = [rng.choice(cells) for _ in cells]
        den = sum(len(by[c]) for c in pick)
        vals.append(sum(ep["success"] for c in pick for ep in by[c]) / den)
    vals.sort()
    boot_low = vals[int(alpha * len(vals))] if vals else None
    icc_raw = lag_icc([[ep["lag_s"] for ep in by[c] if ep["lag_s"] is not None] for c in cells])
    icc = ICC_UNDEFINED_VALUE if icc_raw is None else min(1.0, max(0.0, icc_raw))
    m_bar = n / len(cells)
    n_eff = n / (1.0 + (m_bar - 1.0) * icc)
    cp_low = clopper_pearson_lower(rate * n_eff, n_eff, alpha)

    def crit_(name, value, op, threshold):
        met = value is not None and (value >= threshold if op == ">=" else value <= threshold)
        return {"name": name, "value": value, "op": op, "threshold": threshold, "met": met}

    criteria = [crit_("onset episodes missed or later than the lag budget", n - succ, "<=", g["miss_tolerance"]),
                crit_("success rate, Clopper-Pearson lower bound on n_eff", cp_low, ">=", g["detection_ci_low_min"])]
    out.update({"evaluable": True, "passed": all(c["met"] for c in criteria), "criteria": criteria,
                "success_rate": rate, "cells": len(cells), "episodes_per_cell": m_bar,
                "icc": icc, "icc_raw": icc_raw, "n_eff": n_eff, "x_eff": rate * n_eff,
                "bootstrap": {"n_resamples": n_resamples, "seed": seed, "lower": boot_low, "deciding": False},
                "clopper_pearson_lower": cp_low})
    return out


def window_fa(point: Mapping[str, Any], ci: Mapping[str, Any], gate: Mapping[str, float] = WINDOW_FA_GATE) -> dict:
    """Design item 3 (ii): CRITICAL false alarm on healthy windows (``point`` /
    ``ci`` of :func:`series_point` / :func:`b_prime_boot` at the gate's dwell)."""
    def crit_(name, value, threshold):
        v = value if isinstance(value, (int, float)) and math.isfinite(value) else None
        return {"name": name, "value": v, "op": "<=", "threshold": threshold, "met": v is not None and v <= threshold}

    cs = [crit_("CRITICAL false alarm on healthy windows", point.get("false_alarm"), gate["false_alarm_max"]),
          crit_("its CI95 upper bound", (ci.get("false_alarm_ci95") or [None, None])[1],
                gate["false_alarm_ci95_high_max"])]
    evaluable = point.get("false_alarm") is not None
    return {"criteria": cs, "evaluable": evaluable, "passed": evaluable and all(c["met"] for c in cs),
            "gate": dict(gate)}


# ---------------------------------------------- the A dead band (disclosure only)

#: s0 = this quantile, over training steady-hold windows whose observed p95 TTFT ratio is
#: near 1, of (bootstrap 97.5 % p95 ratio) / (observed p95 ratio): the label noise of a
#: window at the SLO line (challenge_checks.posthoc.json; 1.14 / 1.16 / 1.17 on the hybrid
#: training sets). The A dead band drops violating windows of severity < s0 (disclosed).
DEADBAND_RULE = {"quantile": 0.90, "bootstrap": 300, "seed": 20261005, "min_samples": 20,
                 "near_one": [0.8, 1.25], "primitive": "hold", "upper": 0.975, "min_latency_samples": 10}


def label_noise_s0(rows: Sequence[Mapping[str, str]], label: Any, rule: Mapping[str, Any] = DEADBAND_RULE) -> dict:
    """The A dead band's s0 from training CSV rows (``DEADBAND_RULE``): non-warm-up windows
    of ``rule["primitive"]`` cells with >= min_samples TTFT samples and an observed p95
    TTFT / SLO ratio in near_one; per window the bootstrap upper (``upper``) p95 ratio over
    the observed one; s0 = their ``quantile``."""
    from scripts.rewindow_from_raw import _guarded_p95
    from tre_common import slo_labels

    rng = random.Random(int(rule["seed"]))
    lo_r, hi_r = rule["near_one"]
    mode, min_lat = label.percentile_mode, int(rule["min_latency_samples"])
    ups, ns = [], []
    for r in rows:
        if r.get("primitive") != rule["primitive"] or r.get("in_warmup") == "True":
            continue
        pairs = slo_labels.parse_ttft_len_samples(r.get("ttft_len_samples") or "")
        n = len(pairs)
        if n < int(rule["min_samples"]):
            continue
        ratios = [t / label.ttft_slo_ms(L) for t, L in pairs]
        obs = _guarded_p95(ratios, mode, min_lat)
        if obs is None or not (lo_r <= obs <= hi_r):
            continue
        boots = sorted(_guarded_p95([ratios[rng.randrange(n)] for _ in range(n)], mode, min_lat)
                       for _ in range(int(rule["bootstrap"])))
        ups.append(boots[int(float(rule["upper"]) * len(boots))] / obs)
        ns.append(n)
    ups.sort()
    s0 = ups[int(float(rule["quantile"]) * len(ups))] if ups else None
    return {"s0": s0, "windows_used": len(ups), "n_median": sorted(ns)[len(ns) // 2] if ns else None,
            "rule": dict(rule)}


def deadband_ba(windows: Sequence[Any], *, theta: float, s0: float, direction: str) -> dict:
    """BA at ``theta`` with the violating windows of severity < ``s0`` left out (disclosed)."""
    from tre_calibration.fit import threshold_balanced_accuracy

    keep = [w for w in windows if math.isfinite(w.signal) and (w.slo_met or severity(w) >= s0)]
    two = any(w.slo_met for w in keep) and any(not w.slo_met for w in keep)
    ba = threshold_balanced_accuracy(keep, theta=theta, direction=direction)["balanced_accuracy"] if two else None
    return {"s0": s0, "balanced_accuracy": ba, "excluded_violating": sum(1 for w in windows if math.isfinite(w.signal))
            - len(keep), "windows": len(keep)}

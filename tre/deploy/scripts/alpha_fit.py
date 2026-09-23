#!/usr/bin/env python3
"""Fit the TSS EMA constant alpha (plan 2026-09-21 §6.11 D4'): the ``alpha`` stage of the
campaign fit plan, run after the re-window and before theta is fitted at the chosen alpha.

Definition (D4): ``alpha = 1 - exp(-dt_ref / tau)`` with ``dt_ref = 10 s`` (the phase-aligned
refresh). The controller keeps the dt-scaled update ``alpha_k = 1 - (1 - alpha)^(dt_k/dt_ref)``,
which is exactly the tau-EMA of ``tre_common.tss``; tau = 0 is alpha = 1 (no smoothing).

Rule, per model, over the grid ``tau in {0,5,10,15,20,30,40,60} s`` (alpha 1 .. 0.15):

1. for every alpha the signal is recomputed with that tau-EMA and theta / tau_crit are
   refitted in each leave-one-shape-out (LOSO) fold on the other shapes;
2. the held-out shape is classified in the *deployed* form: CRITICAL iff Z < tau_crit,
   confirmed by the shared dwell (``tre_common.dwell``, 2 new windows by default);
3. it is scored against the SAME-window label (TSS has no lead - plan §6.9c E-B - so a
   t+30 s label mechanically favours tau = 0);
4. feasibility: healthy-window false alarm <= 5 %;
5. score: LOSO balanced accuracy (BA); every feasible alpha within 1 SE (cell bootstrap)
   of the best BA is a candidate;
6. tie 1: fewest spurious CRITICAL episodes per hour on steady healthy cells (hold /
   static cells with no violating window; an episode is a dwell-confirmed CRITICAL run
   with no violating window within +-30 s of it);
   tie 2: the larger alpha (the more responsive EMA).

Detection lag (per violation episode: first confirmed CRITICAL from 30 s before its first
window to its last window), the 90 % step response and the recall / FA / spurious curve
are reported, not optimised. The whole rule is re-applied on ``--bootstrap`` cell
resamples (default 1000) and the selection frequency of every grid point is reported.
By default the bootstrap holds the LOSO fold fits and the 1-SE width at the full sample
and only moves which cells - and so which BA / FA / spurious counts - the rule sees;
``--bootstrap-refit`` also refits theta / tau_crit in every fold of every resample (about
64 fits per resample). The 1-SE width stays the full-sample one in both modes (a nested
bootstrap per resample is not run).

The chosen alpha / tau are written as the registry fields they deploy as
(``trs.ema_tau_ms``, ``trs.ema_alpha``).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

#: tau grid (s) and the alpha each maps to at dt_ref = 10 s: 1, .86, .63, .49, .39, .28, .22, .15
TAU_GRID_S: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0)
DT_REF_S = 10.0
FA_MAX = 0.05
DEFAULT_DWELL_WINDOWS = 2
#: +- margin (ms) within which a violating window makes a CRITICAL episode non-spurious;
#: also how early before a violation episode a confirmation counts as detecting it.
EPISODE_MARGIN_MS = 30_000
DEFAULT_WINDOW_MS = 30_000.0
DEFAULT_STEP_MS = 10_000.0
DEFAULT_BOOTSTRAP = 1000
DEFAULT_SE_RESAMPLES = 1000
DEFAULT_SEED = 20260922
#: load codes of the fixed dynamic primitives; any code >= HOLD_CODE_MIN is a constant-rho
#: cell (boundary hold 1000+, static grid 2000+, gen_calibration_schedules).
HOLD_CODE_MIN = 1000


def alpha_of_tau(tau_s: float, dt_ref_s: float = DT_REF_S) -> float:
    return 1.0 if tau_s <= 0 else 1.0 - math.exp(-float(dt_ref_s) / float(tau_s))


def tau_of_alpha(alpha: float, dt_ref_s: float = DT_REF_S) -> float:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    return 0.0 if alpha >= 1.0 else -float(dt_ref_s) / math.log(1.0 - alpha)


def step90_s(tau_s: float, *, dt_ref_s: float = DT_REF_S, dwell_windows: int = 1) -> float:
    """Time for the EMA to cover 90 % of a step, in refreshes of dt_ref (the first refresh
    after the step counts), plus the extra windows the dwell waits for confirmation."""
    a = alpha_of_tau(tau_s, dt_ref_s)
    n = 1 if a >= 1.0 else max(1, math.ceil(math.log(0.1) / math.log(1.0 - a) - 1e-12))
    return dt_ref_s * (n + max(0, int(dwell_windows) - 1))


#: suffix a bootstrap copy of a cell carries (``<cell>#<k>``), so a cell drawn twice stays
#: two cells to the dwell and the per-cell counts.
COPY_SEP = "#"


def base_cell(scenario_id: str) -> str:
    return scenario_id.split(COPY_SEP, 1)[0]


def cell_load_code(scenario_id: str) -> Optional[int]:
    tail = base_cell(scenario_id).rsplit("_c", 1)
    if len(tail) != 2:
        return None
    try:
        return int(tail[1])
    except ValueError:
        return None


def is_steady_cell(scenario_id: str) -> bool:
    """Constant-rho cell (boundary hold / probe, static grid), not a steps/ramp/bursts cell."""
    code = cell_load_code(scenario_id)
    return code is not None and code >= HOLD_CODE_MIN


def shape_table() -> dict[tuple[int, int], str]:
    """(input, output) of a cell id -> shape name, from the schedule generator."""
    from scripts import gen_calibration_schedules as gen

    out = {(int(i), int(o)): name for name, (i, o) in gen.SHAPES.items()}
    for name, (i, o) in gen.SAMPLED_SHAPES.items():
        out[(gen._length_nominal(i), gen._length_nominal(o))] = name
    for name, (i, o) in gen.STATIC_GRID_SHAPES.items():
        out[(int(i), int(o))] = name
    return out


def shape_of(scenario_id: str, table: Optional[Mapping[tuple[int, int], str]] = None) -> str:
    """Shape of a window's ``scenario_id`` (``i<in>_o<out>_c<code>``); unknown lengths keep
    their ``i<in>_o<out>`` stem as the shape so LOSO still holds them out together."""
    stem = base_cell(scenario_id).rsplit("_c", 1)[0]
    parts = stem.split("_")
    if table is not None and len(parts) == 2 and parts[0][:1] == "i" and parts[1][:1] == "o":
        try:
            key = (int(parts[0][1:]), int(parts[1][1:]))
        except ValueError:
            key = None
        if key in table:
            return table[key]
    return stem


def _runs(flags: Sequence[bool]) -> list[tuple[int, int]]:
    out, i = [], 0
    while i < len(flags):
        if flags[i]:
            s = i
            while i < len(flags) and flags[i]:
                i += 1
            out.append((s, i - 1))
        else:
            i += 1
    return out


@dataclass
class CellStats:
    """One held-out cell under one alpha: counts the rule and its bootstrap need."""

    cell: str
    steady: bool
    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0
    hours: float = 0.0
    spurious: int = 0
    lags_s: list = field(default_factory=list)  # per violation episode, None = missed

    @property
    def healthy(self) -> bool:
        return self.tp + self.fn == 0

    @property
    def steady_healthy(self) -> bool:
        return self.steady and self.healthy


def cell_stats(
    cell: str, starts_ms: Sequence[float], crit: Sequence[bool], violated: Sequence[bool], *,
    step_ms: float = DEFAULT_STEP_MS, margin_ms: float = EPISODE_MARGIN_MS,
) -> CellStats:
    """Same-window confusion counts, spurious CRITICAL episodes and detection lags of one
    cell (rows in window order)."""
    st = CellStats(cell=cell, steady=is_steady_cell(cell))
    for c, v in zip(crit, violated):
        if v:
            st.tp += bool(c)
            st.fn += not c
        else:
            st.fp += bool(c)
            st.tn += not c
    st.hours = len(starts_ms) * step_ms / 3.6e6
    for s, e in _runs(list(crit)):
        lo, hi = starts_ms[s] - margin_ms, starts_ms[e] + margin_ms
        if not any(v for t, v in zip(starts_ms, violated) if lo <= t <= hi):
            st.spurious += 1
    for s, e in _runs(list(violated)):
        t0, t1 = starts_ms[s], starts_ms[e]
        hits = [t for t, c in zip(starts_ms, crit) if c and t0 - margin_ms <= t <= t1]
        st.lags_s.append((min(hits) - t0) / 1000.0 if hits else None)
    return st


@dataclass
class Aggregate:
    ba: float
    recall: float
    fa: float
    n_pos: int
    n_neg: int
    spurious_steady_healthy: int
    hours_steady_healthy: float
    spurious_all: int
    hours_all: float

    @property
    def spurious_per_h(self) -> Optional[float]:
        return self.spurious_steady_healthy / self.hours_steady_healthy if self.hours_steady_healthy else None

    @property
    def spurious_per_h_all(self) -> Optional[float]:
        return self.spurious_all / self.hours_all if self.hours_all else None


def aggregate(stats: Sequence[CellStats]) -> Aggregate:
    tp = sum(s.tp for s in stats)
    fn = sum(s.fn for s in stats)
    fp = sum(s.fp for s in stats)
    tn = sum(s.tn for s in stats)
    rec = tp / (tp + fn) if tp + fn else float("nan")
    fa = fp / (fp + tn) if fp + tn else float("nan")
    sh = [s for s in stats if s.steady_healthy]
    return Aggregate(
        ba=(rec + 1.0 - fa) / 2.0, recall=rec, fa=fa, n_pos=tp + fn, n_neg=fp + tn,
        spurious_steady_healthy=sum(s.spurious for s in sh), hours_steady_healthy=sum(s.hours for s in sh),
        spurious_all=sum(s.spurious for s in stats), hours_all=sum(s.hours for s in stats),
    )


def _resample(cells: Sequence[str], rng: random.Random) -> list[str]:
    return [rng.choice(cells) for _ in cells]


def ba_se(stats: Mapping[str, CellStats], *, n: int = DEFAULT_SE_RESAMPLES, seed: int = DEFAULT_SEED) -> float:
    """Cell-bootstrap SE of the LOSO BA (resamples with one class missing are skipped)."""
    cells = sorted(stats)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        agg = aggregate([stats[c] for c in _resample(cells, rng)])
        if agg.n_pos and agg.n_neg:
            vals.append(agg.ba)
    if len(vals) < 2:
        return float("nan")
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def select(points: Sequence[Mapping[str, Any]], *, fa_max: float = FA_MAX) -> dict[str, Any]:
    """The D4' rule on per-alpha points ``{tau_s, alpha, ba, se, fa, spurious_per_h}``.

    Returns the chosen point's tau, the BA leader, the 1-SE candidate set and why the
    winner won. An undefined spurious rate (no steady healthy hours) ranks as 0 for every
    point, so the rule falls through to the larger alpha."""
    valid = [p for p in points if math.isfinite(p["ba"])]
    feasible = [p for p in valid if math.isfinite(p["fa"]) and p["fa"] <= fa_max]
    no_feasible = not feasible
    pool = feasible or valid
    if not pool:
        return {"chosen_tau_s": None, "no_feasible_alpha": True, "reason": "no scorable alpha"}
    best = max(pool, key=lambda p: (p["ba"], p["alpha"]))
    se = best["se"] if math.isfinite(best["se"]) else 0.0
    within = [p for p in pool if p["ba"] >= best["ba"] - se]

    def spur(p):
        v = p.get("spurious_per_h")
        return 0.0 if v is None else v

    chosen = min(within, key=lambda p: (spur(p), -p["alpha"]))
    fewest = min(spur(p) for p in within)
    tied = [p for p in within if spur(p) == fewest]
    if len(within) == 1:
        reason = "only alpha within 1 SE of the best LOSO BA"
    elif len(tied) == 1:
        reason = "fewest spurious CRITICAL episodes/h on steady healthy cells within 1 SE"
    else:
        reason = "tied on spurious episodes within 1 SE -> larger alpha"
    return {
        "chosen_tau_s": chosen["tau_s"],
        "chosen_alpha": chosen["alpha"],
        "best_ba_tau_s": best["tau_s"],
        "best_ba": best["ba"],
        "one_se": se,
        "within_1se_tau_s": [p["tau_s"] for p in within],
        "feasible_tau_s": [p["tau_s"] for p in feasible],
        "no_feasible_alpha": no_feasible,
        "reason": reason + (" (no alpha met FA <= %.2f; rule applied to all)" % fa_max if no_feasible else ""),
    }


#: fit(train_windows) -> (theta, tau_crit) or None when the fit does not publish
FitFn = Callable[[Sequence[Any]], Optional[tuple[float, float]]]
#: crit(test_windows, theta, tau_crit) -> dwell-confirmed CRITICAL per window
CritFn = Callable[[Sequence[Any], float, float], list[bool]]


def loso_stats(
    windows: Sequence[Any], fit: FitFn, crit_fn: CritFn, *,
    shape_fn: Callable[[str], str], step_ms: float = DEFAULT_STEP_MS,
) -> tuple[dict[str, CellStats], dict[str, Any]]:
    """Per-cell stats of the LOSO classifier under one alpha, and the per-fold fits."""
    shapes = sorted({shape_fn(w.scenario_id) for w in windows})
    out: dict[str, CellStats] = {}
    folds: dict[str, Any] = {}
    for s in shapes:
        train = [w for w in windows if shape_fn(w.scenario_id) != s]
        test = [w for w in windows if shape_fn(w.scenario_id) == s]
        fd = fit(train)
        if fd is None:
            folds[s] = None
            continue
        theta, tau_crit = fd
        folds[s] = {"theta": theta, "tau_crit": tau_crit, "test_windows": len(test)}
        crit = crit_fn(test, theta, tau_crit)
        by: dict[str, list[int]] = {}
        for i, w in enumerate(test):
            by.setdefault(w.scenario_id, []).append(i)
        for cell, idx in by.items():
            idx.sort(key=lambda i: test[i].window_start_ms or 0.0)
            out[cell] = cell_stats(
                cell, [float(test[i].window_start_ms or 0.0) for i in idx], [crit[i] for i in idx],
                [not test[i].slo_met for i in idx], step_ms=step_ms,
            )
    return out, folds


def _pct(values: Sequence[float], q: float) -> Optional[float]:
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


def alpha_rule(
    load: Callable[[float], Sequence[Any]],
    fit: FitFn,
    crit_fn: CritFn,
    *,
    tau_grid_s: Sequence[float] = TAU_GRID_S,
    dt_ref_s: float = DT_REF_S,
    dwell_windows: int = DEFAULT_DWELL_WINDOWS,
    fa_max: float = FA_MAX,
    shape_fn: Optional[Callable[[str], str]] = None,
    step_ms: float = DEFAULT_STEP_MS,
    se_resamples: int = DEFAULT_SE_RESAMPLES,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    full_fit: Optional[Callable[[Sequence[Any]], Optional[Mapping[str, Any]]]] = None,
    bootstrap_refit: bool = False,
    log: Callable[[str], None] = lambda _m: None,
) -> dict[str, Any]:
    """Apply the D4' rule. ``load(tau_s)`` returns the fitting windows with the tau-EMA
    applied; ``fit`` / ``crit_fn`` are the theta/delta fit and the deployed classifier."""
    if shape_fn is None:
        table = shape_table()
        shape_fn = lambda sid: shape_of(sid, table)  # noqa: E731
    curve: list[dict[str, Any]] = []
    per_alpha: dict[float, dict[str, CellStats]] = {}
    loaded: dict[float, Sequence[Any]] = {}
    for tau in tau_grid_s:
        windows = load(tau)
        if bootstrap_refit:
            loaded[tau] = windows
        stats, folds = loso_stats(windows, fit, crit_fn, shape_fn=shape_fn, step_ms=step_ms)
        per_alpha[tau] = stats
        agg = aggregate(list(stats.values()))
        se = ba_se(stats, n=se_resamples, seed=seed)
        lags = [x for s in stats.values() for x in s.lags_s]
        hit = [x for x in lags if x is not None]
        point = {
            "tau_s": tau, "alpha": alpha_of_tau(tau, dt_ref_s),
            "ba": agg.ba, "se": se, "recall": agg.recall, "fa": agg.fa,
            "feasible": math.isfinite(agg.fa) and agg.fa <= fa_max,
            "n_pos": agg.n_pos, "n_neg": agg.n_neg, "cells": len(stats),
            "spurious_per_h": agg.spurious_per_h, "spurious_n": agg.spurious_steady_healthy,
            "steady_healthy_hours": agg.hours_steady_healthy,
            "steady_healthy_cells": sum(1 for s in stats.values() if s.steady_healthy),
            "spurious_per_h_all_cells": agg.spurious_per_h_all, "spurious_n_all_cells": agg.spurious_all,
            "episodes": len(lags), "episodes_detected": len(hit),
            "detect_lag_median_s": _pct(hit, 0.5), "detect_lag_p75_s": _pct(hit, 0.75),
            "step90_ema_s": step90_s(tau, dt_ref_s=dt_ref_s),
            "step90_with_dwell_s": step90_s(tau, dt_ref_s=dt_ref_s, dwell_windows=dwell_windows),
            "folds": folds,
        }
        if full_fit is not None:
            point["full_fit"] = full_fit(windows)
        curve.append(point)
        log(f"tau={tau:g} alpha={point['alpha']:.2f} BA={agg.ba:.3f}+-{se:.3f} rec={agg.recall:.3f} "
            f"fa={agg.fa:.3f} spur/h={point['spurious_per_h']} lag med={point['detect_lag_median_s']}")
    sel = select(curve, fa_max=fa_max)

    # whole-rule bootstrap: resample cells, re-score every alpha, re-apply the rule
    cells = sorted({c for st in per_alpha.values() for c in st})
    rng = random.Random(seed + 1)
    counts = {tau: 0 for tau in tau_grid_s}
    used = 0
    se_by_tau = {p["tau_s"]: p["se"] for p in curve}
    by_cell: dict[float, dict[str, list]] = {}
    for tau, ws in loaded.items():
        d: dict[str, list] = {}
        for w in ws:
            d.setdefault(w.scenario_id, []).append(w)
        by_cell[tau] = d
    for b in range(bootstrap):
        smp = _resample(cells, rng)
        pts = []
        for tau in tau_grid_s:
            if bootstrap_refit:
                ws = [replace(w, scenario_id=f"{c}{COPY_SEP}{k}")
                      for k, c in enumerate(smp) for w in by_cell[tau].get(c, ())]
                st, _ = loso_stats(ws, fit, crit_fn, shape_fn=shape_fn, step_ms=step_ms)
                agg = aggregate(list(st.values()))
            else:
                st = per_alpha[tau]
                agg = aggregate([st[c] for c in smp if c in st])
            if not (agg.n_pos and agg.n_neg):
                continue
            pts.append({"tau_s": tau, "alpha": alpha_of_tau(tau, dt_ref_s), "ba": agg.ba,
                        "se": se_by_tau[tau], "fa": agg.fa, "spurious_per_h": agg.spurious_per_h})
        r = select(pts, fa_max=fa_max) if pts else {"chosen_tau_s": None}
        if r["chosen_tau_s"] is not None:
            counts[r["chosen_tau_s"]] += 1
            used += 1
        if bootstrap_refit and (b + 1) % 50 == 0:
            log(f"bootstrap {b + 1}/{bootstrap}")
    freq = {str(t): (counts[t] / used if used else None) for t in tau_grid_s}
    chosen_tau = sel.get("chosen_tau_s")
    out = {
        "rule": "D4-prime: FA<=%.2f feasibility; same-window LOSO BA; within 1 SE -> fewest spurious "
                "CRITICAL episodes/h on steady healthy cells -> larger alpha" % fa_max,
        "dt_ref_s": dt_ref_s, "dwell_windows": dwell_windows, "fa_max": fa_max,
        "episode_margin_ms": EPISODE_MARGIN_MS, "tau_grid_s": list(tau_grid_s),
        "label_horizon": "same window",
        "selection": sel,
        "bootstrap": {"resamples": bootstrap, "used": used, "seed": seed + 1,
                      "selection_frequency_by_tau_s": freq,
                      "chosen_frequency": freq.get(str(chosen_tau)) if chosen_tau is not None else None,
                      "refit": bootstrap_refit,
                      "note": ("cells resampled, LOSO theta/tau_crit refitted per resample; 1-SE width "
                               "held at the full sample" if bootstrap_refit else
                               "cells resampled; LOSO fold fits and the 1-SE width held at the full sample")},
        "curve": curve,
    }
    if chosen_tau is not None:
        out["chosen"] = {"tau_s": chosen_tau, "alpha": alpha_of_tau(chosen_tau, dt_ref_s)}
        out["registry_fields"] = registry_fields(chosen_tau, dt_ref_s)
    return out


def registry_fields(tau_s: float, dt_ref_s: float = DT_REF_S) -> dict[str, Any]:
    """The ``trs`` registry block the chosen tau deploys as. tau = 0 is ``ema_tau_ms: 0``
    with ``ema_alpha: 1.0`` - the controller's fixed-alpha branch at alpha 1 is the identity,
    i.e. no smoothing; ``ema_alpha`` otherwise records alpha at dt_ref (plan D4)."""
    return {"trs": {"ema_tau_ms": float(tau_s) * 1000.0, "ema_alpha": round(alpha_of_tau(tau_s, dt_ref_s), 6)},
            "dt_ref_s": dt_ref_s}


# --------------------------------------------------------------------- repo wiring


def run(args: argparse.Namespace) -> dict[str, Any]:
    from tre_calibration.fit import fit_delta_margins
    from tre_common.slo_labels import label_def_from_args

    from scripts import theta_verdict as tv

    label = label_def_from_args(args, args.model)

    def spec(tau_s: float):
        return tv.build_signal_spec("tss", w_p=args.w_p, lambda_wait=args.lambda_wait, qmin=args.qmin,
                                    ema_tau_ms=(tau_s * 1000.0 if tau_s > 0 else None))

    cur = {"spec": spec(0.0)}

    def load(tau_s: float):
        cur["spec"] = spec(tau_s)
        return cur["spec"].load(args.fitting_csv, label, args.trim_ramp_windows)

    def fit_full(windows):
        sp = cur["spec"]
        f = sp.default_config().fit(windows)
        if not f.publish or f.theta is None:
            return None
        d = fit_delta_margins(windows, theta=float(f.theta), direction=sp.direction)
        return {"theta": float(f.theta), "tau_crit": d.crit.tau, "delta_crit": d.crit.delta,
                "delta_high": d.high.delta}

    def fit(windows):
        r = fit_full(windows)
        return None if r is None else (r["theta"], r["tau_crit"])

    def crit(test, theta, tau_crit):
        return tv.critical_dwell_flags(test, theta=theta, tau_crit=tau_crit, direction=cur["spec"].direction,
                                       dwell_windows=args.dwell_windows)

    grid = tuple(float(x) for x in args.tau_grid_s) if args.tau_grid_s else TAU_GRID_S
    rep = alpha_rule(
        load, fit, crit, tau_grid_s=grid, dt_ref_s=args.dt_ref_s, dwell_windows=args.dwell_windows,
        fa_max=args.fa_max, step_ms=args.step_ms, se_resamples=args.se_resamples,
        bootstrap=args.bootstrap, seed=args.seed, full_fit=fit_full, bootstrap_refit=args.bootstrap_refit,
        log=lambda m: print(f"[{args.model}] {m}", flush=True),
    )
    chosen = rep.get("chosen")
    if chosen is not None:
        pt = next(p for p in rep["curve"] if p["tau_s"] == chosen["tau_s"])
        rep["chosen"]["full_fit"] = pt.get("full_fit")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "fitting_csv": str(args.fitting_csv),
        "label_def": label.as_dict(),
        "signal": {"signal": "tss", "w_p": args.w_p, "lambda_wait": args.lambda_wait, "qmin": args.qmin},
        "trim_ramp_windows": args.trim_ramp_windows,
        **rep,
    }


def _parse(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    from tre_common.slo_labels import add_label_arguments

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--fitting-csv", required=True)
    p.add_argument("--w-p", type=float, required=True)
    p.add_argument("--lambda-wait", type=float, required=True)
    p.add_argument("--qmin", type=float, default=1.0)
    add_label_arguments(p, require_fixed=False)
    p.add_argument("--trim-ramp-windows", type=int, default=1)
    p.add_argument("--tau-grid-s", type=float, nargs="*", default=None)
    p.add_argument("--dt-ref-s", type=float, default=DT_REF_S)
    p.add_argument("--dwell-windows", type=int, default=DEFAULT_DWELL_WINDOWS)
    p.add_argument("--fa-max", type=float, default=FA_MAX)
    p.add_argument("--step-ms", type=float, default=DEFAULT_STEP_MS,
                   help="re-window step (hours per window for the spurious rate)")
    p.add_argument("--se-resamples", type=int, default=DEFAULT_SE_RESAMPLES)
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--bootstrap-refit", action="store_true",
                   help="refit the LOSO folds on every bootstrap resample (slow)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", required=True)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse(argv)
    rep = run(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    sel = rep["selection"]
    print(f"[{args.model}] chosen tau={sel.get('chosen_tau_s')} s alpha={sel.get('chosen_alpha')} "
          f"({sel.get('reason')}); bootstrap freq={rep['bootstrap']['chosen_frequency']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

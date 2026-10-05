#!/usr/bin/env python3
"""Power of the next round's acceptance gates on M2, by Monte Carlo (design aid only).

Sizes the number of burst cells of the M2 composition (``calibration_acceptance``
``--composition m2-20261005``). Nothing here reads M2; the DEV inputs (detection, lags,
within-cell ICC, healthy windows per primitive) are the post-hoc tables of 2026-10-05 and
are only used to set the simulation's plausible ranges.

Gate (i), the episode gate (owned by ``dline_refit accept``):
  every onset episode of the dynamic cells (bursts, ramps, steps) is caught by CRITICAL
  within 20 s, AND the detection rate's lower confidence bound >= .80, the bound being a
  Clopper-Pearson bound on the effective n = N / DE, DE = 1 + (m_w - 1) * ICC (m_w the
  size-weighted episodes per cell). With every episode detected the cell-cluster
  bootstrap of the rate is degenerate (all 1.0), so the CP bound on n_eff is what binds:
  it needs n_eff >= 14 (one-sided 95 %, .05^(1/n) >= .80) or >= 17 (two-sided 95 %,
  .025^(1/n) >= .80).

The miss of an episode (not caught, or caught later than 20 s) is drawn from a
beta-binomial per cell (cell miss probability ~ Beta with mean p and ICC rho), so misses
cluster in cells like the DEV lags do. The ICC that enters DE is either the design value
("known") or estimated from simulated quantised lags with the same rho ("estimated":
one-way ANOVA ICC(1) over cells with >= 2 episodes, clamped to [0, 1]; undefined -> 1.0,
as the DEV table reports 14b's all-zero lags).

Gate (ii), window false alarms: FA = false CRITICAL windows / healthy windows <= .05 with
the CI upper bound <= .08 (cell-cluster percentile bootstrap). False alarms arrive as runs
of 3 overlapping windows (30 s windows on a 10 s step) at a per-cell rate that varies
between cells (gamma, CV 1).

Usage: python3 m2_power_sim.py --out DIR [--sims 4000] [--seed 20261005]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

EPISODES_PER_BURST = 4      # one onset per spike
RAMPS, STEPS = 2, 2          # ramp: one onset; steps top out at .95 rho*: assumed none
LAG_TICK_S = 10.0
LAG_SIGMA_TICKS = 1.0        # DEV lags: mostly 0 / 10 s, range -30..20 s
LAG_MEAN_TICKS = 0.4
RATE_LOW = 0.80

#: Healthy windows per cell (DEV per-cell tables of M scaled to the M2 cell lengths).
HEALTHY = {"bursts": 25, "steps": 32, "ramp": 16, "hold_0.9": 21, "hold_deep": 2}


def cp_lower(x: float, n: float, alpha: float) -> float:
    if n <= 0:
        return 0.0
    if x <= 0:
        return 0.0
    return float(beta_dist.ppf(alpha, x, n - x + 1)) if x < n else float(alpha ** (1.0 / n))


def episode_layout(bursts: int) -> list[int]:
    return [EPISODES_PER_BURST] * bursts + [1] * RAMPS


def design_effect(sizes: list[int], icc: float) -> float:
    s = np.asarray(sizes, float)
    m_w = float((s * s).sum() / s.sum())
    return 1.0 + (m_w - 1.0) * icc


def anova_icc(lags_by_cell: list[np.ndarray]) -> float:
    groups = [g for g in lags_by_cell if len(g) >= 2]
    if len(groups) < 2:
        return 1.0
    k = len(groups)
    n_i = np.array([len(g) for g in groups], float)
    n = n_i.sum()
    grand = np.concatenate(groups).mean()
    means = np.array([g.mean() for g in groups])
    ssb = float((n_i * (means - grand) ** 2).sum())
    ssw = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    msb = ssb / (k - 1)
    msw = ssw / (n - k)
    n0 = (n - (n_i ** 2).sum() / n) / (k - 1)
    denom = msb + (n0 - 1) * msw
    if denom <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (msb - msw) / denom)))


def sim_episode_gate(bursts: int, p: float, rho: float, sims: int, rng: np.random.Generator) -> dict:
    sizes = episode_layout(bursts)
    n_total = sum(sizes)
    out = {"bursts": bursts, "p_miss": p, "icc": rho, "episodes": n_total}
    for alpha, tag in ((0.05, "one_sided"), (0.025, "two_sided")):
        out[f"n_eff_known_{tag}"] = None
    de_known = design_effect(sizes, rho)
    out["n_eff_known"] = round(n_total / de_known, 2)
    out["ci_low_known_one_sided"] = round(cp_lower(n_total / de_known, n_total / de_known, 0.05), 3)
    out["ci_low_known_two_sided"] = round(cp_lower(n_total / de_known, n_total / de_known, 0.025), 3)
    zero = 0
    pass_known = {0.05: 0, 0.025: 0}
    pass_est = {0.05: 0, 0.025: 0}
    icc_hats = []
    for _ in range(sims):
        # misses: beta-binomial per cell
        if p <= 0:
            misses = 0
        elif rho <= 0:
            misses = int(rng.binomial(n_total, p))
        else:
            ab = (1.0 - rho) / rho
            q = rng.beta(p * ab, (1.0 - p) * ab, size=len(sizes))
            misses = int(sum(rng.binomial(m, qq) for m, qq in zip(sizes, q)))
        all_caught = misses == 0
        zero += all_caught
        # lags (for the estimated ICC)
        a = rng.normal(0.0, LAG_SIGMA_TICKS * math.sqrt(rho), size=len(sizes))
        lags = [np.round(LAG_MEAN_TICKS + a_i + rng.normal(0.0, LAG_SIGMA_TICKS * math.sqrt(1 - rho), m))
                * LAG_TICK_S for a_i, m in zip(a, sizes)]
        icc_hat = anova_icc(lags)
        icc_hats.append(icc_hat)
        n_eff_hat = n_total / design_effect(sizes, icc_hat)
        for alpha in (0.05, 0.025):
            if all_caught and cp_lower(n_total / de_known, n_total / de_known, alpha) >= RATE_LOW:
                pass_known[alpha] += 1
            if all_caught and cp_lower(n_eff_hat, n_eff_hat, alpha) >= RATE_LOW:
                pass_est[alpha] += 1
    out.update({
        "p_all_caught": round(zero / sims, 3),
        "p_pass_known_one_sided": round(pass_known[0.05] / sims, 3),
        "p_pass_known_two_sided": round(pass_known[0.025] / sims, 3),
        "p_pass_est_one_sided": round(pass_est[0.05] / sims, 3),
        "p_pass_est_two_sided": round(pass_est[0.025] / sims, 3),
        "icc_hat_median": round(float(np.median(icc_hats)), 3),
        "icc_hat_p90": round(float(np.quantile(icc_hats, 0.9)), 3),
    })
    for k in [k for k in out if k.startswith("n_eff_known_")]:
        del out[k]
    return out


def healthy_layout(bursts: int) -> np.ndarray:
    cells = ([HEALTHY["bursts"]] * bursts + [HEALTHY["steps"]] * STEPS + [HEALTHY["ramp"]] * RAMPS
             + [HEALTHY["hold_0.9"]] + [HEALTHY["hold_deep"]] * 3)
    return np.asarray(cells, int)


def sim_fa_gate(healthy: np.ndarray, f: float, sims: int, boots: int,
                rng: np.random.Generator, run_len: int = 3) -> dict:
    h_total = int(healthy.sum())
    ok = 0
    highs = []
    for _ in range(sims):
        # per-cell rate: gamma with mean f, CV 1; events are runs of `run_len` windows
        rate = rng.gamma(1.0, f, size=len(healthy)) if f > 0 else np.zeros(len(healthy))
        runs = rng.poisson(rate * healthy / run_len)
        fa = np.minimum(runs * run_len, healthy)
        point = fa.sum() / h_total
        idx = rng.integers(0, len(healthy), size=(boots, len(healthy)))
        boot = fa[idx].sum(axis=1) / np.maximum(healthy[idx].sum(axis=1), 1)
        high = float(np.quantile(boot, 0.975))
        highs.append(high)
        ok += (point <= 0.05) and (high <= 0.08)
    return {"healthy_windows": h_total, "cells": int(len(healthy)), "true_fa": f,
            "p_pass": round(ok / sims, 3), "ci_high_median": round(float(np.median(highs)), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--fa-sims", type=int, default=1500)
    ap.add_argument("--boots", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    episode = [sim_episode_gate(b, p, rho, args.sims, rng)
               for b in (6, 8, 10, 12, 14, 16, 18)
               for rho in (0.3, 0.8)
               for p in (0.0, 0.01, 0.03)]
    fa = []
    for b in (8, 12, 16):
        layout = healthy_layout(b)
        for f in (0.01, 0.02, 0.03):
            fa.append({"bursts": b, **sim_fa_gate(layout, f, args.fa_sims, args.boots, rng)})
    # FA alone vs healthy-window count (cells of 25 healthy windows)
    fa_n = []
    for n_cells in (8, 12, 16, 20, 24, 32):
        layout = np.full(n_cells, 25, int)
        for f in (0.01, 0.02, 0.03):
            fa_n.append(sim_fa_gate(layout, f, args.fa_sims, args.boots, rng))
    doc = {"what": "M2 power simulation (design aid; DEV-informed ranges, not evidence)",
           "seed": args.seed, "sims": args.sims, "fa_sims": args.fa_sims, "boots": args.boots,
           "assumptions": {"episodes_per_burst": EPISODES_PER_BURST, "ramps": RAMPS,
                           "steps_episodes": 0, "lag_tick_s": LAG_TICK_S,
                           "lag_sigma_ticks": LAG_SIGMA_TICKS, "lag_mean_ticks": LAG_MEAN_TICKS,
                           "healthy_windows_per_cell": HEALTHY, "fa_run_windows": 3,
                           "fa_rate_between_cells": "gamma, CV 1"},
           "episode_gate": episode, "fa_gate_composition": fa, "fa_gate_vs_healthy": fa_n}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "m2_power.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    lines = ["bursts icc  p_miss episodes n_eff(known) all_caught pass_known(1s/2s) pass_est(1s/2s) icc_hat_med/p90"]
    for r in episode:
        lines.append(f"{r['bursts']:6d} {r['icc']:.1f} {r['p_miss']:6.2f} {r['episodes']:8d} "
                     f"{r['n_eff_known']:12.1f} {r['p_all_caught']:10.3f} "
                     f"{r['p_pass_known_one_sided']:.3f}/{r['p_pass_known_two_sided']:.3f}      "
                     f"{r['p_pass_est_one_sided']:.3f}/{r['p_pass_est_two_sided']:.3f}    "
                     f"{r['icc_hat_median']:.2f}/{r['icc_hat_p90']:.2f}")
    lines.append("")
    lines.append("FA gate on the composition: bursts healthy_windows true_fa p_pass ci_high_median")
    for r in fa:
        lines.append(f"  {r['bursts']:3d} {r['healthy_windows']:5d} {r['true_fa']:.2f} {r['p_pass']:.3f} {r['ci_high_median']:.4f}")
    lines.append("FA gate vs healthy windows (25 per cell): cells healthy true_fa p_pass ci_high_median")
    for r in fa_n:
        lines.append(f"  {r['cells']:3d} {r['healthy_windows']:5d} {r['true_fa']:.2f} {r['p_pass']:.3f} {r['ci_high_median']:.4f}")
    text = "\n".join(lines) + "\n"
    (args.out / "m2_power.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

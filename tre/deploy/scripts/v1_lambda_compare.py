#!/usr/bin/env python3
"""The v1-lambda refits next to the D22 freeze, on data that already exists (2026-09-24).

Selects nothing: every number here is disclosure. The v1-lambda parameters were chosen by
``dline_refit wp --lambda-method v1`` + ``final --no-holdout`` on the freeze's own D16
training set before this script read any evaluation data; T14 (not collected yet) is the
only unseen test set and is evaluated by the pre-registered rule, not here.

Stages
``resplit-fit``  one resplit seed x model: the v1-lambda selection + v2 theta/delta on that
                 seed's TRAIN part (``calibration_resplit_20260924/seed_S/fit``, read only),
                 written under ``--out/resplit/seed_S`` - the resplit protocol with the
                 lambda rule swapped, so it can sit next to the resplit's own lambda = 1 refit.
                 ``--wp-grid`` / ``--max-ci-half-width-fraction`` as ``dline_refit``.
``compare``      the tables, for the freeze and every ``--arm NAME=<refit root>``: parameters;
                 D13 CI half width (and the 20 % / 15 % verdicts); training BA / AUROC /
                 Spearman; B' (CRITICAL recall of violations with severity >= the TRAINING
                 .65 quantile, false alarm, all-violation recall, LOW-band share) on training,
                 on every resplit seed's test part (all cells / M-origin cells only - the only
                 cells none of the full-set fits saw) and on M (the accept stage's validation
                 CSVs, already used once); the resplit protocol (refit on each seed's train
                 part) for the freeze, the resplit's lambda = 1 refit and every
                 ``--protocol NAME=<resplit-fit out root>``; cross-model pooled-Z AUROC.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts import calibration_resplit as rs
from scripts import dline_refit as dr
from scripts import v1_lambda_fit as v1

MODELS = ("dsqwen-7b", "dsllama-8b", "dsqwen-14b")
X = Path("/data/nfs_shared_data/xxy")
FIT = X / "calibration_refit_final_20260923" / "fit"
FREEZE = X / "calibration_freeze_20260923" / "params_freeze.json"
ACCEPT_D = X / "calibration_freeze_20260923" / "params_freeze.accept.d"
RESPLIT = X / "calibration_resplit_20260924"
SEEDS = (20260924, 20260925, 20260926, 20260927, 20260928)
M_DATASET = X / "calibration_M_20260923" / "dataset"
SEVERITY_Q = 0.65
D13_GATE = 0.20
D13_LEGACY = 0.15


def _vh_from_verdict(path: Path) -> dict:
    return dr.verdict_for_holdout(json.loads(Path(path).read_text()))


def _origin(csv_path: Path) -> dict[str, str]:
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["scenario_id"]] = r.get("pool_origin") or "train_final"
    return out


def score_set(vh: Mapping[str, Any], csv_path: Path, *, cut: float, subset: Optional[str] = None,
              origin: Optional[Mapping[str, str]] = None, boot: bool = True) -> dict:
    """BA / AUROC / Spearman(health) + both/TPOT recall (resplit's point metrics) + B'."""
    s = rs.Scored(vh, csv_path)
    idx = list(range(len(s.windows)))
    if subset == "M":
        idx = [i for i in idx if origin[s.windows[i].scenario_id].startswith("M_")]
    if not idx:
        return {"windows": 0}
    pm = rs.point_metrics(s, idx)
    ws = [s.windows[i] for i in idx]
    c0 = [s.crit0[i] for i in idx]
    c2 = [s.crit2[i] for i in idx]
    bp = v1.b_prime_point(ws, theta=s.theta, tau_crit=s.tau_crit, cut=cut, crit0=c0, crit2=c2)
    if boot:
        bp["dwell2_ci"] = v1.b_prime_boot(ws, theta=s.theta, tau_crit=s.tau_crit, cut=cut, crit=c2)
        bp["nodwell_ci"] = v1.b_prime_boot(ws, theta=s.theta, tau_crit=s.tau_crit, cut=cut, crit=c0)
    return {"windows": pm["windows"], "cells": pm["cells"], "violating": pm["violating"],
            "ba": pm["ba"], "auroc": pm["auroc"], "spearman_health": pm["spearman_health"],
            "B_old_dwell2": pm["dwell2"], "B_old_nodwell": pm["nodwell"], "B_prime": bp,
            "theta": s.theta, "tau_crit": s.tau_crit, "z": [s.z[i] for i in idx],
            "slo_met": [s.windows[i].slo_met for i in idx]}


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("z", "slo_met")}


def stage_resplit_fit(out: Path, seed: int, model: str, registry: str, *, wp_grid: str = "with_zero",
                      max_ci: Optional[float] = None) -> None:
    fit = RESPLIT / f"seed_{seed}" / "fit"
    p = dr.paths(fit, model)
    lab = dr.label_for(model, "primary", registry)
    od = out / "resplit" / f"seed_{seed}" / "refit" / model / "primary"
    od.mkdir(parents=True, exist_ok=True)
    dr.D13_MAX_CI_FRACTION = max_ci
    alpha_doc = {"published_tau_s": 10.0, "rule": "fixed (D18, not swept)", "published_alpha": dr.alpha_of(10.0)}
    (od / "alpha.json").write_text(json.dumps(alpha_doc, indent=1))
    sources = v1.sources_from_trainset(FIT)
    sources["M"] = M_DATASET
    wp = dr.stage_wp_v1(model, lab, p, alpha_doc, sources, wp_grid=wp_grid)
    (od / "wp.json").write_text(json.dumps(wp, indent=1, default=str))
    fin = dr.stage_final(model, lab, p, wp, od, holdout=False)
    (od / "final.json").write_text(json.dumps(fin, indent=1, default=str))
    print(f"seed {seed} {model}: lambda {wp['lambda_star']} w_p {wp['w_p_used']} theta {fin.get('theta_published')}")


def _d13(ci_frac: Optional[float], publish_rate: Optional[float]) -> dict:
    ok = publish_rate is not None and publish_rate >= 0.9
    return {"ci_half_frac": ci_frac, "publish_rate": publish_rate,
            f"pass_{int(D13_GATE * 100)}": None if ci_frac is None else bool(ok and ci_frac < D13_GATE),
            f"pass_{int(D13_LEGACY * 100)}": None if ci_frac is None else bool(ok and ci_frac < D13_LEGACY)}


def stage_compare(out: Path, arms_roots: Mapping[str, Path], protocol_roots: Mapping[str, Path]) -> dict:
    """``arms_roots``: NAME -> a dline_refit out root (<root>/<model>/primary/...) of a fit on
    the full D16 training set; ``protocol_roots``: NAME -> a resplit-fit out root."""
    freeze = json.loads(FREEZE.read_text())
    res: dict[str, Any] = {"what": __doc__, "severity_quantile": SEVERITY_Q, "arms": ["freeze", *arms_roots],
                           "arm_roots": {k: str(v) for k, v in arms_roots.items()},
                           "protocol_roots": {k: str(v) for k, v in protocol_roots.items()}, "models": {}}
    pooled: dict[str, dict[str, tuple[list, list]]] = {}
    for m in MODELS:
        fe = freeze["models"][m]
        arms = {"freeze": fe["verdict_for_holdout"]}
        d13 = {"freeze": _d13(fe.get("ci_half_frac"), fe.get("publish_rate"))}
        for name, root in arms_roots.items():
            d = Path(root) / m / "primary"
            arms[name] = _vh_from_verdict(d / "verdict_final.json")
            fin = json.loads((d / "final.json").read_text())
            d13[name] = _d13(fin.get("ci_half_frac"), fin.get("publish_rate"))
        train_csv = dr.paths(FIT, m)["fitting"]
        cut = v1.severity_cut(rs.Scored(arms["freeze"], train_csv).windows, SEVERITY_Q)
        r: dict[str, Any] = {"severity_cut_train": cut, "params": {}, "d13": d13, "train": {}, "M_accept": {},
                             "resplit_test_fullfit": {}, "resplit_protocol": {}}
        for a, vh in arms.items():
            pub = vh["published"]
            spec = vh["signal_spec"]["tss"]
            r["params"][a] = {"lambda_wait": spec["lambda_wait"], "w_p": spec["w_p"], "theta": pub["theta_m"],
                              "tau_crit": pub["tau_crit"], "tau_high": pub["tau_high"],
                              "delta_crit": pub.get("delta_crit"), "delta_high": pub.get("delta_high")}
            r["train"][a] = _strip(score_set(vh, train_csv, cut=cut))
            r["M_accept"][a] = _strip(score_set(vh, ACCEPT_D / f"{m}_validation.csv", cut=cut))
        for sd in SEEDS:
            test = RESPLIT / f"seed_{sd}" / "fit" / f"{m}_test.csv"
            org = _origin(test)
            row = {}
            for a, vh in arms.items():
                full = score_set(vh, test, cut=cut, boot=False)
                mo = score_set(vh, test, cut=cut, subset="M", origin=org, boot=False)
                row[a] = {"all": _strip(full), "M_origin": _strip(mo)}
                pooled.setdefault(f"fullfit_{a}_seed{sd}", {}).setdefault(m, (full["z"], full["slo_met"]))
            r["resplit_test_fullfit"][str(sd)] = row
            prot = {"freeze": arms["freeze"],
                    "lambda1_refit": _vh_from_verdict(RESPLIT / f"seed_{sd}" / "refit" / m / "verdict_final.json")}
            for name, root in protocol_roots.items():
                p = Path(root) / "resplit" / f"seed_{sd}" / "refit" / m / "primary" / "verdict_final.json"
                if p.exists():
                    prot[name] = _vh_from_verdict(p)
            prow = {}
            for a, vh in prot.items():
                sc = score_set(vh, test, cut=cut, boot=False)
                spec = vh["signal_spec"]["tss"]
                prow[a] = {"params": {"lambda_wait": spec["lambda_wait"], "w_p": spec["w_p"],
                                      "theta": vh["published"]["theta_m"], "tau_crit": vh["published"]["tau_crit"]},
                           **_strip(sc)}
                pooled.setdefault(f"protocol_{a}_seed{sd}", {}).setdefault(m, (sc["z"], sc["slo_met"]))
            r["resplit_protocol"][str(sd)] = prow
        res["models"][m] = r
    cross = {}
    for key, per in pooled.items():
        if set(per) == set(MODELS):
            z = [x for mm in MODELS for x in per[mm][0]]
            y = [x for mm in MODELS for x in per[mm][1]]
            cross[key] = rs.auroc(z, y)
    res["cross_model_pooled_z_auroc"] = cross
    summ = {}
    for key in {k.rsplit("_seed", 1)[0] for k in cross}:
        vals = [v for k, v in cross.items() if k.rsplit("_seed", 1)[0] == key and v is not None]
        summ[key] = statistics.median(vals) if vals else None
    res["cross_model_pooled_z_auroc_median"] = summ
    (out / "compare.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def _f(x, d=3):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def _med(vs):
    vs = [v for v in vs if v is not None]
    return statistics.median(vs) if vs else None


def print_tables(res: Mapping[str, Any]) -> None:
    for m, r in res["models"].items():
        print(f"\n== {m}  (severity cut, training .65 quantile of violations: {r['severity_cut_train']:.3f})")
        for a, p in r["params"].items():
            t, x, d = r["train"][a], r["M_accept"][a], r["d13"][a]
            rows = [row[a] for row in r["resplit_test_fullfit"].values()]
            print(f"  {a:12} lam {p['lambda_wait']:<6g} w_p {p['w_p']:<7g} theta {p['theta']:7.1f} tc {p['tau_crit']:.3f} "
                  f"th {p['tau_high']:.3f} | D13 {_f(d['ci_half_frac'])} (20%:{d['pass_20']} 15%:{d['pass_15']}) "
                  f"| train BA {_f(t['ba'])} | resplit BA {_f(_med([q['all']['ba'] for q in rows]))} "
                  f"M-orig {_f(_med([q['M_origin'].get('ba') for q in rows]))} | M BA {_f(x['ba'])} "
                  f"| B' train {_f(t['B_prime']['dwell2']['recall_severe'])}/{_f(t['B_prime']['dwell2']['false_alarm'])} "
                  f"M {_f(x['B_prime']['dwell2']['recall_severe'])}/{_f(x['B_prime']['dwell2']['false_alarm'])} "
                  f"resplit {_f(_med([q['all']['B_prime']['dwell2']['recall_severe'] for q in rows]))}"
                  f"/{_f(_med([q['all']['B_prime']['dwell2']['false_alarm'] for q in rows]))} "
                  f"| all-viol M {_f(x['B_prime']['dwell2']['recall_all'])}")
        names = sorted({a for row in r["resplit_protocol"].values() for a in row})
        for a in names:
            rows = [row[a] for row in r["resplit_protocol"].values() if a in row]
            lam = sorted({q["params"]["lambda_wait"] for q in rows})
            wps = sorted({q["params"]["w_p"] for q in rows})
            print(f"  protocol {a:18} lam {lam} w_p {wps} BA {_f(_med([q['ba'] for q in rows]))} "
                  f"AUC {_f(_med([q['auroc'] for q in rows]))}")
    print("\ncross-model pooled-Z AUROC (median over 5 seeds):")
    for k, v in sorted(res["cross_model_pooled_z_auroc_median"].items()):
        print(f"  {k:32} {_f(v, 4)}")


def _named(items: Sequence[str]) -> dict[str, Path]:
    out = {}
    for t in items:
        name, sep, root = t.partition("=")
        if not sep:
            raise SystemExit(f"{t!r}: expected NAME=ROOT")
        out[name] = Path(root)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["resplit-fit", "compare"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--wp-grid", choices=v1.WP_GRIDS, default="with_zero")
    ap.add_argument("--max-ci-half-width-fraction", type=float, default=None)
    ap.add_argument("--v1-refit", type=Path, default=None,
                    help="compare: <v1lambda root>/refit, as the arm 'v1lambda' (first round)")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=ROOT")
    ap.add_argument("--protocol", action="append", default=[], metavar="NAME=ROOT",
                    help="compare: a resplit-fit --out root (default: --out itself as 'v1lambda_refit')")
    args = ap.parse_args(argv)
    if args.stage == "resplit-fit":
        stage_resplit_fit(args.out, args.seed, args.model, args.registry, wp_grid=args.wp_grid,
                          max_ci=args.max_ci_half_width_fraction)
        return 0
    arms = _named(args.arm)
    if args.v1_refit is not None:
        arms = {"v1lambda": args.v1_refit, **arms}
    protocol = _named(args.protocol) or {"v1lambda_refit": args.out}
    res = stage_compare(args.out, arms, protocol)
    print_tables(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

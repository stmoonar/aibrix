#!/usr/bin/env python3
"""Re-split training set + M (steady hold cells only), refit, evaluate (2026-09-24).

After the D22 once-only acceptance on M failed (params_freeze.accept.json), the user asked
for the existing data to be re-split into train / test, the parameters refitted on the
train part with the unchanged pipeline, and the ranking separation checked on the test
part. M has been used once: the test part of this re-split is NOT unseen data and this is
not a confirmatory acceptance. Everything is fixed in a pre-registration written before any
fit (``prereg``); ``run`` does one seed; ``aggregate`` builds the comparison tables.

Stages
``prereg``  pool = the D16 training CSVs of the final refit (verified against their
            trainset.json) + the constant-load (hold) cells of the M dataset; dynamic
            cells stay disclosure-only. Unit = (model, scenario_id). Stratum = model x
            shape (nested in the load-shape family). 30 % of every stratum to test, cells
            ordered by sha256(seed|model|scenario_id). Writes split_manifest.json (0444) +
            .sha256. No label, no signal and no latency is read to split.
``run``     one seed: writes train / test CSVs, fits with dline_refit.stage_wp (tau 10 s,
            the D17 w_p rule) and stage_final(holdout=False) with lambda_wait forced to 1,
            then scores the refit and the frozen parameters on the same test cells.
``aggregate`` the side-by-side tables and the seed spread.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import dline_refit as dr

MODELS = ("dsqwen-7b", "dsllama-8b", "dsqwen-14b")
X = Path("/data/nfs_shared_data/xxy")
REFIT = X / "calibration_refit_final_20260923"
MROOT = X / "calibration_M_20260923"
FREEZE = X / "calibration_freeze_20260923" / "params_freeze.json"
ACCEPT = X / "calibration_freeze_20260923" / "params_freeze.accept.json"
TEST_FRACTION = 0.30
PRIMARY_SEED = 20260924
ROBUST_SEEDS = (20260925, 20260926, 20260927, 20260928)
BOOT_SEED = dr.SEED  # 20260922, the accept stage's bootstrap seed
BOOT_N = 1000
FIX_TAU_S = 10.0
FIX_LAMBDA = 1.0
#: stratification family of a shape (reporting / nesting only; the fit's own family CSVs
#: stay gen.families() = S3,T8 / S4,S5, unchanged)
STRAT_FAMILY = {"S3": "prefill_heavy", "T8": "prefill_heavy", "MP": "prefill_heavy",
                "S4": "decode_heavy", "S5": "decode_heavy", "MD": "decode_heavy"}
TAU_B_GRID = tuple(round(0.50 + 0.01 * i, 2) for i in range(101))  # 0.50 .. 1.50


def sha(p: Path) -> str:
    return dr.sha256_file(Path(p))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def label(model: str, registry: str):
    return dr.label_for(model, "primary", registry)


# ----------------------------------------------------------------------------- pool


def read_pool(models: Sequence[str] = MODELS):
    """(header, rows per model in source order, cell table per model). Rows get the
    columns ``pool_origin`` and ``orig_split``; M rows get ``run = M``."""
    from scripts import gen_calibration_schedules as gen  # noqa: F401  (cell_kind imports it)

    tman = json.loads((REFIT / "fit" / dr.TRAINSET_MANIFEST).read_text())
    header: list[str] = []
    rows: dict[str, list[dict]] = {m: [] for m in models}
    cells: dict[str, OrderedDict] = {m: OrderedDict() for m in models}

    def add(model, row, origin):
        for c in row:
            if c not in header:
                header.append(c)
        row = dict(row)
        row["orig_split"] = row.get("split", "")
        row["pool_origin"] = origin
        rows[model].append(row)
        sid = row["scenario_id"]
        c = cells[model].get(sid)
        if c is None:
            c = cells[model][sid] = {"model": model, "scenario_id": sid, "origin": origin, "shape": row["shape"],
                                     "strat_family": STRAT_FAMILY.get(row["shape"], "other"),
                                     "primitive": row["primitive"], "role": row["role"],
                                     "run": row.get("run", ""), "attempts": set(), "windows": 0}
        if c["origin"] != origin or c["shape"] != row["shape"]:
            raise SystemExit(f"{model} {sid}: two origins / shapes")
        c["attempts"].add(row.get("attempt", ""))
        c["windows"] += 1

    for m in models:
        f = REFIT / "fit" / f"{m}_fitting.csv"
        want = tman["models"][m]["files"]["fitting"]["sha256"]
        if sha(f) != want:
            raise SystemExit(f"{f} changed since trainset.json")
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if dr.cell_kind(r["primitive"], r["role"], r["split"], r["shape"]) != dr.KIND_CONSTANT:
                    raise SystemExit(f"{f}: non-constant-load row {r['scenario_id']}")
                add(m, r, "train_final")
    mw = MROOT / "dataset" / "windows.csv"
    with open(mw, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = r["model"]
            if m not in rows:
                continue
            # D16 on M: the row's own primitive / role; split is holdout for every M row and
            # the M shapes are acceptance shapes, so both are ignored for the kind here.
            if dr.cell_kind(r["primitive"], r["role"], "train", "") != dr.KIND_CONSTANT:
                continue
            r = dict(r)
            r["run"] = "M"
            add(m, r, "M_acceptance_hold" if r["role"] == "acceptance" else "M_boundary_probe")
    for m in models:
        for c in cells[m].values():
            c["attempts"] = sorted(c["attempts"])
    if "pool_origin" not in header:
        header += ["orig_split", "pool_origin"]
    return header, rows, cells


def split_cells(cells: Mapping[str, Mapping[str, Any]], seed: int, model: str) -> dict[str, str]:
    by = defaultdict(list)
    for sid, c in cells.items():
        by[c["shape"]].append(sid)
    out = {}
    for shape, sids in sorted(by.items()):
        n = len(sids)
        k = int(math.floor(TEST_FRACTION * n + 0.5))
        if n >= 2:
            k = min(max(k, 1), n - 1)
        order = sorted(sids, key=lambda s: hashlib.sha256(f"{seed}|{model}|{s}".encode()).hexdigest())
        for i, s in enumerate(order):
            out[s] = "test" if i < k else "train"
    return out


# --------------------------------------------------------------------------- prereg


def stage_prereg(out: Path, registry: str) -> None:
    man_path = out / "split_manifest.json"
    if man_path.exists():
        raise SystemExit(f"{man_path} exists (pre-registration is written once)")
    header, rows, cells = read_pool()
    freeze = json.loads(FREEZE.read_text())
    labels = {}
    for m in MODELS:
        ld = label(m, registry).as_dict()
        s = dr.canonical_sha256(ld)
        mm = json.loads((MROOT / m / "M_manifest.json").read_text())
        if s != mm["label_def_sha256"] or s != freeze["models"][m]["label_def_sha256"]:
            raise SystemExit(f"{m}: label_def sha {s} != M / freeze")
        labels[m] = {"label_def": ld, "label_def_sha256": s,
                     "matches": {"M_manifest": mm["label_def_sha256"], "freeze": freeze["models"][m]["label_def_sha256"]}}
    seeds = (PRIMARY_SEED, *ROBUST_SEEDS)
    splits = {}
    for seed in seeds:
        sp = {}
        for m in MODELS:
            a = split_cells(cells[m], seed, m)
            strata = defaultdict(lambda: {"train": 0, "test": 0})
            fam = defaultdict(lambda: {"train": 0, "test": 0})
            org = defaultdict(lambda: {"train": 0, "test": 0})
            for sid, side in a.items():
                c = cells[m][sid]
                strata[c["shape"]][side] += 1
                fam[c["strat_family"]][side] += 1
                org[c["origin"]][side] += 1
            sp[m] = {"test": sorted(s for s, v in a.items() if v == "test"),
                     "train": sorted(s for s, v in a.items() if v == "train"),
                     "cells_per_stratum": dict(sorted(strata.items())),
                     "cells_per_family": dict(sorted(fam.items())),
                     "cells_per_origin": dict(sorted(org.items()))}
        splits[str(seed)] = sp
    pool_cells = {m: [{k: v for k, v in c.items()} for c in cells[m].values()] for m in MODELS}
    doc = {
        "what": ("Pre-registration of the 2026-09-24 re-split: train / test re-drawn from the D16 "
                 "training set U the steady hold cells of M; refit with the unchanged pipeline; "
                 "evaluate refit and frozen parameters on the same test cells. Written before any fit."),
        "written_at_utc": now(),
        "status_of_M": ("M (calibration_M_20260923) HAS BEEN USED ONCE: the D22 once-only acceptance "
                        "(params_freeze.accept.json, 2026-09-23) scored 13 manifest cells per model and "
                        "failed (A: 8b/14b BA CI95 low .736/.732 < .75; B: CRITICAL recall .53-.63 < .85). "
                        "This is a re-split of data already looked at: the test part is NOT unseen data "
                        "and nothing here is a confirmatory acceptance. Of the M hold cells pooled here, "
                        "the 5 acceptance holds per model were scored in that acceptance; the boundary "
                        "probes (role boundary) were sealed and never scored."),
        "accept_result_sha256": sha(ACCEPT),
        "freeze": {"path": str(FREEZE), "sha256": sha(FREEZE), "freeze_sha256": freeze["freeze_sha256"]},
        "code": {**dr.code_state(), "script": "tre/deploy/scripts/calibration_resplit.py"},
        "pool": {
            "rule": ("D16: steady hold cells only. (1) every row of the final refit's training CSVs "
                     "<model>_fitting.csv (the D16 cut: primitive hold/static, role not ramp; smoke "
                     "holds and sentinels train), sha256 checked against trainset.json; (2) every M "
                     "dataset row with primitive in STEADY_PRIMITIVES and role not in UNSTEADY_ROLES "
                     "(M acceptance holds + M boundary probes, all cell_status as in the training cut). "
                     "Excluded, disclosure only: M dynamic cells (steps / ramp / bursts), the run1 "
                     "retained M cells (dynamic), H2."),
            "sources": {
                "training_csvs": {m: {"path": str(REFIT / "fit" / f"{m}_fitting.csv"),
                                      "sha256": sha(REFIT / "fit" / f"{m}_fitting.csv")} for m in MODELS},
                "trainset_manifest": {"path": str(REFIT / "fit" / dr.TRAINSET_MANIFEST),
                                      "sha256": sha(REFIT / "fit" / dr.TRAINSET_MANIFEST)},
                "M_windows_csv": {"path": str(MROOT / "dataset" / "windows.csv"),
                                  "sha256": sha(MROOT / "dataset" / "windows.csv")},
                "M_manifests": {m: {"path": str(MROOT / m / "M_manifest.json"),
                                    "sha256": sha(MROOT / m / "M_manifest.json")} for m in MODELS},
            },
            "counts": {m: {"cells": len(cells[m]), "windows_rows": len(rows[m]),
                           "cells_by_origin": {o: sum(1 for c in cells[m].values() if c["origin"] == o)
                                               for o in ("train_final", "M_acceptance_hold", "M_boundary_probe")}}
                       for m in MODELS},
            "cells": pool_cells,
        },
        "split": {
            "unit": "cell = (model, scenario_id); attempts of one scenario_id stay together (the loaders EMA and bootstrap by scenario_id)",
            "strata": ("model x shape; shapes nest in the load-shape families prefill_heavy {S3,T8,MP}, "
                       "decode_heavy {S4,S5,MD}, other {S1,S2,T9,G800x240,U512x512}, so the split is "
                       "also stratified by model x family"),
            "test_fraction": TEST_FRACTION,
            "allocation": ("per stratum n_test = floor(0.3 n + 0.5), clamped to [1, n-1] when n >= 2; cells "
                           "ordered by sha256('<seed>|<model>|<scenario_id>') hex, the first n_test are test"),
            "inputs_read_to_split": "model, scenario_id, shape only (no label, signal or latency)",
            "primary_seed": PRIMARY_SEED, "robustness_seeds": list(ROBUST_SEEDS),
            "assignments": splits,
        },
        "labels": labels,
        "fit": {
            "pipeline": ("scripts.dline_refit at the recorded commit, algorithms unchanged: stage_wp (w_p grid "
                         f"{list(dr.WP_GRID)}, D17: largest w_p meeting c1 (BA within 1 SE of w_p=0) and c2 "
                         "(family gap <= CI half width)), then stage_final(holdout=False): D5 merged theta, "
                         "delta_crit / delta_high by fit_delta_margins, tau_crit = 1 - delta_crit, tau_high = 1 + delta_high"),
            "fixed": {"tau_s": FIX_TAU_S, "alpha": dr.alpha_of(FIX_TAU_S), "lambda_wait": FIX_LAMBDA,
                      "note": ("tau not fitted (D18): the alpha sweep is not run, stage_wp gets published_tau_s = 10; "
                               "stage_wp's lambda check still runs and is reported, but lambda_wait is forced to 1 "
                               "for the final fit")},
            "family_csvs": "gen.families() = prefill_heavy {S3,T8}, decode_heavy {S4,S5} (unchanged; MP / MD are not added)",
            "arm": "primary (D6' label)", "trim_ramp_windows": dr.TRIM_RAMP_WINDOWS,
        },
        "evaluation": {
            "on": "the test cells of the split (steady hold only)",
            "parameter_sets": ["resplit refit (this seed)", "freeze (params_freeze.json, unchanged)"],
            "point": "theta_verdict.holdout_report (the accept stage's scorer)",
            "bootstrap": {"unit": "cell (scenario_id) with replacement", "n": BOOT_N, "seed": BOOT_SEED,
                          "interval": "95 % percentile (dline_refit._ci95)"},
            "metrics": ["BA at the published theta (+CI)", "AUROC and Spearman(health) of the signal - the "
                        "'ranking separation' of theta_verdict._ranking / eval_ranking_separation (+CI)",
                        "CRITICAL recall of both/TPOT-only violations and healthy false alarm, no dwell (gating) "
                        "and dwell 2 (reported)", "CRITICAL recall of all violations", "per violation class recall "
                        "(ttft_only / tpot_only / both / unserved)", "cross-model: AUROC of pooled Z = TSS/theta_m "
                        "over the three models' test windows (secondary)"],
            "gates": {
                "A": {"ba_min": dr.A_BA_MIN, "ba_ci_low_min": dr.A_BA_CI_LOW_MIN,
                      "max_drop_from_training": dr.A_MAX_DROP_FROM_TRAINING},
                "B": {"dwell": "none (gating; the controller will default dwell off); dwell 2 reported",
                      "recall_min": dr.B_RECALL_MIN, "recall_ci_low_min": dr.B_RECALL_CI_LOW_MIN,
                      "false_alarm_max": dr.B_FALSE_ALARM_MAX, "false_alarm_ci_high_max": dr.B_FALSE_ALARM_CI_HIGH_MAX},
                "C": "TTFT-only recall disclosed, no gate",
                "D": "training stop rule (D13): CI half width <= 15 %, publish_rate >= 0.9; family gap reported",
            },
            "comparison": ("freeze vs refit on the identical test cells, plus a paired cell bootstrap of "
                           "delta BA / delta recall (refit - freeze). Caveat fixed in advance: test cells of origin "
                           "train_final were in the freeze's training set (in-sample for freeze), so the comparison "
                           "is also reported on the M-origin test cells only (out-of-sample for both). Decision "
                           "rule: 'refit better' only if the paired delta-BA CI95 excludes 0 on the primary seed; "
                           "otherwise the difference is attributed to the split."),
            "robustness": "all 5 seeds: min / median / max and relative range of theta, w_p, tau_crit, tau_high and of the test metrics",
            "secondary_exploratory": [
                "recall ceiling: both/TPOT recall of Z < 1 (tau = 1) on test",
                ("B-aligned tau_crit chosen on TRAINING only: smallest tau on "
                 f"{TAU_B_GRID[0]}..{TAU_B_GRID[-1]} step .01 with training no-dwell both/TPOT recall >= .85; "
                 "report its training false alarm and its test recall / false alarm. Not a published parameter."),
            ],
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, indent=1, default=list) + "\n").encode()
    dr._write_once(man_path, data)
    digest = hashlib.sha256(data).hexdigest()
    dr._write_once(out / "split_manifest.json.sha256", f"{digest}  split_manifest.json\n".encode())
    print(f"wrote {man_path} sha256 {digest}")
    for m in MODELS:
        print(m, doc["pool"]["counts"][m], splits[str(PRIMARY_SEED)][m]["cells_per_origin"])


# ------------------------------------------------------------------------------ run


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(header))
        for r in rows:
            w.writerow([r.get(c, "") for c in header])


def rank_avg(v: Sequence[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def auroc(scores: Sequence[float], pos: Sequence[bool]) -> float | None:
    n1 = sum(pos)
    n0 = len(pos) - n1
    if not n1 or not n0:
        return None
    r = rank_avg(scores)
    return (sum(ri for ri, p in zip(r, pos) if p) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2:
        return None
    rx, ry = rank_avg(x), rank_avg(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else None


def health(w) -> float:
    return w.health_score if w.health_score is not None else (1.0 if w.slo_met else 0.0)


class Scored:
    """One parameter set on one window list: z, no-dwell and dwell-2 CRITICAL flags."""

    def __init__(self, vh: Mapping[str, Any], csv_path: Path):
        from scripts import theta_verdict as tv

        self.vh = vh
        self.spec = tv.SignalSpec.from_dict(vh["signal_spec"])
        self.label = dr.slo_labels.LabelDefinition.from_dict(vh["label_def"])
        self.windows = [w for w in self.spec.load(csv_path, self.label, int(vh["trim_ramp_windows"]))]
        self.theta = float(vh["published"]["theta_m"])
        self.tau_crit = float(vh["published"]["tau_crit"])
        self.direction = vh["fit_config"]["direction"]
        self.z = [w.signal / self.theta for w in self.windows]
        self.crit0 = tv.critical_dwell_flags(self.windows, theta=self.theta, tau_crit=self.tau_crit,
                                             direction=self.direction, dwell_windows=1)
        self.crit2 = tv.critical_dwell_flags(self.windows, theta=self.theta, tau_crit=self.tau_crit,
                                             direction=self.direction, dwell_windows=2)


def rate(sel, flags):
    return (sum(1 for i in sel if flags[i]) / len(sel)) if sel else None


def point_metrics(s: Scored, idx: Sequence[int]) -> dict:
    from scripts import theta_verdict as tv
    from tre_calibration.fit import threshold_balanced_accuracy

    ws = [s.windows[i] for i in idx]
    viol = [i for i in idx if not s.windows[i].slo_met]
    ok = [i for i in idx if s.windows[i].slo_met]
    b = [i for i in viol if s.windows[i].violation_class in tv.CRITERION_B_CLASSES]
    two = 0 < len(viol) < len(idx)
    ba = threshold_balanced_accuracy(ws, theta=s.theta, direction=s.direction) if two else None
    fin = [i for i in idx if math.isfinite(s.windows[i].signal)]
    out = {
        "windows": len(idx), "cells": len({s.windows[i].scenario_id for i in idx}), "violating": len(viol),
        "healthy": len(ok), "both_tpot_windows": len(b),
        "ba": ba["balanced_accuracy"] if ba else None,
        "recall_good": ba["recall_good"] if ba else None, "specificity_bad": ba["specificity_bad"] if ba else None,
        "auroc": auroc([s.windows[i].signal for i in fin], [s.windows[i].slo_met for i in fin]),
        "spearman_health": spearman([s.windows[i].signal for i in fin], [health(s.windows[i]) for i in fin]),
        "nodwell": {"recall_both_tpot": rate(b, s.crit0), "false_alarm": rate(ok, s.crit0),
                    "recall_all": rate(viol, s.crit0)},
        "dwell2": {"recall_both_tpot": rate(b, s.crit2), "false_alarm": rate(ok, s.crit2),
                   "recall_all": rate(viol, s.crit2)},
        "classes_nodwell": {}, "classes_dwell2": {},
        "recall_ceiling_z_lt_1_both_tpot": rate(b, [z < 1.0 for z in s.z]),
    }
    for cls in tv.VIOLATION_CLASSES:
        sel = [i for i in viol if s.windows[i].violation_class == cls]
        out["classes_nodwell"][cls] = {"windows": len(sel), "recall": rate(sel, s.crit0)}
        out["classes_dwell2"][cls] = {"windows": len(sel), "recall": rate(sel, s.crit2)}
    return out


def boot_metrics(s: Scored, idx: Sequence[int], *, n: int = BOOT_N, seed: int = BOOT_SEED,
                 other: "Scored | None" = None) -> dict:
    """Cell bootstrap of BA / AUROC / Spearman / B rates; with ``other`` (same windows, other
    parameters) also the paired differences s - other."""
    from scripts import theta_verdict as tv
    from tre_calibration.fit import threshold_balanced_accuracy

    by = defaultdict(list)
    for i in idx:
        by[s.windows[i].scenario_id].append(i)
    cells = sorted(by)
    if other is not None:
        okey = {(w.scenario_id, w.window_start_ms): j for j, w in enumerate(other.windows)}
    vals = defaultdict(list)
    rng = random.Random(seed)
    bset = frozenset(tv.CRITERION_B_CLASSES)

    def rates_of(sc, ii):
        viol = [i for i in ii if not sc.windows[i].slo_met]
        ok = [i for i in ii if sc.windows[i].slo_met]
        b = [i for i in viol if sc.windows[i].violation_class in bset]
        return {"recall_both_tpot_nodwell": rate(b, sc.crit0), "false_alarm_nodwell": rate(ok, sc.crit0),
                "recall_both_tpot_dwell2": rate(b, sc.crit2), "false_alarm_dwell2": rate(ok, sc.crit2),
                "recall_all_nodwell": rate(viol, sc.crit0)}

    for _ in range(n if cells else 0):
        pick = [rng.choice(cells) for _ in cells]
        ii = [i for c in pick for i in by[c]]
        ws = [s.windows[i] for i in ii]
        two = any(w.slo_met for w in ws) and any(not w.slo_met for w in ws)
        r = rates_of(s, ii)
        if two:
            ba = threshold_balanced_accuracy(ws, theta=s.theta, direction=s.direction)["balanced_accuracy"]
            vals["ba"].append(ba)
            vals["auroc"].append(auroc([w.signal for w in ws], [w.slo_met for w in ws]))
            sp = spearman([w.signal for w in ws], [health(w) for w in ws])
            if sp is not None:
                vals["spearman_health"].append(sp)
        for k, v in r.items():
            if v is not None:
                vals[k].append(v)
        if other is not None:
            jj = [okey[(s.windows[i].scenario_id, s.windows[i].window_start_ms)] for i in ii
                  if (s.windows[i].scenario_id, s.windows[i].window_start_ms) in okey]
            ow = [other.windows[j] for j in jj]
            if two and ow:
                oba = threshold_balanced_accuracy(ow, theta=other.theta, direction=other.direction)["balanced_accuracy"]
                vals["delta_ba"].append(ba - oba)
            ro = rates_of(other, jj)
            for k in ("recall_both_tpot_nodwell", "false_alarm_nodwell"):
                if r[k] is not None and ro[k] is not None:
                    vals["delta_" + k].append(r[k] - ro[k])
    return {k: {"ci95": dr._ci95(v), "used": len(v)} for k, v in sorted(vals.items())}


def gates(pm: Mapping[str, Any], bm: Mapping[str, Any], train_ba: float | None, stop: Mapping[str, Any]) -> dict:
    c = dr._criterion
    a = [c("BA", pm["ba"], ">=", dr.A_BA_MIN),
         c("BA CI95 low", bm.get("ba", {}).get("ci95", [None])[0], ">=", dr.A_BA_CI_LOW_MIN),
         c("BA >= train BA - 0.08", pm["ba"], ">=", None if train_ba is None else train_ba - dr.A_MAX_DROP_FROM_TRAINING)]
    b = [c("recall both/TPOT (no dwell)", pm["nodwell"]["recall_both_tpot"], ">=", dr.B_RECALL_MIN),
         c("its CI95 low", bm.get("recall_both_tpot_nodwell", {}).get("ci95", [None])[0], ">=", dr.B_RECALL_CI_LOW_MIN),
         c("false alarm (no dwell)", pm["nodwell"]["false_alarm"], "<=", dr.B_FALSE_ALARM_MAX),
         c("its CI95 high", bm.get("false_alarm_nodwell", {}).get("ci95", [None, None])[1], "<=", dr.B_FALSE_ALARM_CI_HIGH_MAX)]
    b2 = [c("recall both/TPOT (dwell 2)", pm["dwell2"]["recall_both_tpot"], ">=", dr.B_RECALL_MIN),
          c("its CI95 low", bm.get("recall_both_tpot_dwell2", {}).get("ci95", [None])[0], ">=", dr.B_RECALL_CI_LOW_MIN),
          c("false alarm (dwell 2)", pm["dwell2"]["false_alarm"], "<=", dr.B_FALSE_ALARM_MAX),
          c("its CI95 high", bm.get("false_alarm_dwell2", {}).get("ci95", [None, None])[1], "<=", dr.B_FALSE_ALARM_CI_HIGH_MAX)]
    return {"A": {"criteria": a, "passed": all(x["met"] for x in a)},
            "B_nodwell": {"criteria": b, "passed": all(x["met"] for x in b)},
            "B_dwell2_reported": {"criteria": b2, "passed": all(x["met"] for x in b2)},
            "D": {"passed": stop.get("satisfied") is True, "ci_half_frac": stop.get("ci_half_width_fraction"),
                  "reasons": stop.get("reasons")}}


def tau_b_on_train(vh: Mapping[str, Any], train_csv: Path) -> dict:
    """Secondary: tau chosen on TRAINING only so that no-dwell both/TPOT recall >= .85."""
    from scripts import theta_verdict as tv

    s = Scored(vh, train_csv)
    viol = [i for i, w in enumerate(s.windows) if not w.slo_met]
    ok = [i for i, w in enumerate(s.windows) if w.slo_met]
    b = [i for i in viol if s.windows[i].violation_class in tv.CRITERION_B_CLASSES]
    curve = []
    pick = None
    for t in TAU_B_GRID:
        rec = rate(b, [z < t for z in s.z])
        fa = rate(ok, [z < t for z in s.z])
        curve.append({"tau": t, "train_recall_both_tpot": rec, "train_false_alarm": fa})
        if pick is None and rec is not None and rec >= dr.B_RECALL_MIN:
            pick = {"tau": t, "train_recall_both_tpot": rec, "train_false_alarm": fa}
    return {"tau_B": pick, "train_at_tau_crit": rate(b, s.crit0), "train_fa_at_tau_crit": rate(ok, s.crit0),
            "train_recall_at_1": rate(b, [z < 1.0 for z in s.z]), "train_fa_at_1": rate(ok, [z < 1.0 for z in s.z]),
            "curve": curve[::5]}


def evaluate(vh, csv_path: Path, origin_of: Mapping[str, str], train_ba, stop, *, other_vh=None) -> dict:
    s = Scored(vh, csv_path)
    o = Scored(other_vh, csv_path) if other_vh is not None else None
    all_idx = list(range(len(s.windows)))
    subsets = {"all": all_idx,
               "origin_M": [i for i in all_idx if origin_of[s.windows[i].scenario_id].startswith("M_")],
               "origin_train_final": [i for i in all_idx if origin_of[s.windows[i].scenario_id] == "train_final"]}
    res = {}
    for name, idx in subsets.items():
        if not idx:
            res[name] = None
            continue
        pm = point_metrics(s, idx)
        bm = boot_metrics(s, idx, other=o)
        res[name] = {"point": pm, "boot": bm, "gates": gates(pm, bm, train_ba, stop) if name == "all" else None}
    return {"theta": s.theta, "tau_crit": s.tau_crit, "published": vh["published"], "subsets": res}


def stage_run(out: Path, seed: int, model: str, registry: str) -> None:
    man = json.loads((out / "split_manifest.json").read_text())
    want = (out / "split_manifest.json.sha256").read_text().split()[0]
    if sha(out / "split_manifest.json") != want:
        raise SystemExit("split_manifest.json does not match its sha256")
    assign = man["split"]["assignments"][str(seed)][model]
    side = {**{s: "test" for s in assign["test"]}, **{s: "train" for s in assign["train"]}}
    header, rows, cells = read_pool([model])
    if set(side) != set(cells[model]):
        raise SystemExit(f"{model}: pool cells differ from the manifest")
    origin_of = {sid: c["origin"] for sid, c in cells[model].items()}
    from scripts import gen_calibration_schedules as gen

    family_of = {s: f for f, shapes in gen.families().items() for s in shapes}
    d = out / f"seed_{seed}"
    fit = d / "fit"
    fit.mkdir(parents=True, exist_ok=True)
    tr = [r for r in rows[model] if side[r["scenario_id"]] == "train"]
    te = [r for r in rows[model] if side[r["scenario_id"]] == "test"]
    p = dr.paths(fit, model)
    write_csv(p["fitting"], header, tr)
    for fam in dr.FAMILY_FILES:
        write_csv(p["families"][fam], header, [r for r in tr if family_of.get(r["shape"]) == fam])
    test_csv = fit / f"{model}_test.csv"
    write_csv(test_csv, header, te)
    lab = label(model, registry)
    if dr.canonical_sha256(lab.as_dict()) != man["labels"][model]["label_def_sha256"]:
        raise SystemExit("label drift")
    od = d / "refit" / model
    od.mkdir(parents=True, exist_ok=True)
    alpha_doc = {"published_tau_s": FIX_TAU_S, "rule": "fixed (D18, not swept)", "published_alpha": dr.alpha_of(FIX_TAU_S)}
    (od / "alpha.json").write_text(json.dumps(alpha_doc, indent=1))
    wp = dr.stage_wp(model, lab, p, alpha_doc)
    wp["lambda_star_rule"] = wp["lambda_star"]
    wp["lambda_star"] = FIX_LAMBDA
    (od / "wp.json").write_text(json.dumps(wp, indent=1, default=str))
    fin = dr.stage_final(model, lab, p, wp, od, holdout=False)
    (od / "final.json").write_text(json.dumps(fin, indent=1, default=str))
    if "error" in fin:
        raise SystemExit(f"{model} seed {seed}: final failed {fin['error']}")
    v = json.loads((od / "verdict_final.json").read_text())
    vh = dr.verdict_for_holdout(v)
    freeze = json.loads(FREEZE.read_text())["models"][model]
    fvh = freeze["verdict_for_holdout"]
    ev_refit = evaluate(vh, test_csv, origin_of, fin["train_ba_at_published"], v["stop_rule"], other_vh=fvh)
    ev_freeze = evaluate(fvh, test_csv, origin_of, freeze["train_ba_at_published"], freeze["stop_rule"])
    tb = tau_b_on_train(vh, p["fitting"])
    if tb["tau_B"] is not None:
        vb = json.loads(json.dumps(vh))
        vb["published"]["tau_crit"] = tb["tau_B"]["tau"]
        sb = Scored(vb, test_csv)
        tb["test_at_tau_B"] = {k: point_metrics(sb, list(range(len(sb.windows))))[k] for k in ("nodwell", "dwell2")}
    doc = {"model": model, "seed": seed, "split_manifest_sha256": want, "generated_at_utc": now(),
           "code": dr.code_state(),
           "train": {"cells": len(assign["train"]), "rows": len(tr)}, "test": {"cells": len(assign["test"]), "rows": len(te),
                                                                             "csv": str(test_csv), "csv_sha256": sha(test_csv)},
           "refit": {"published": {"theta": fin["theta_published"], "w_p": fin["w_p"], "lambda_wait": fin["lambda_wait"],
                                   "tau_s": fin["tau_s"], "alpha": fin["alpha"], "delta_crit": fin["delta_crit"],
                                   "delta_high": fin["delta_high"], "tau_crit": 1.0 - fin["delta_crit"],
                                   "tau_high": 1.0 + fin["delta_high"]},
                     "train_ba_at_published": fin["train_ba_at_published"], "ci_half_frac": fin["ci_half_frac"],
                     "publish_rate": fin["publish_rate"], "family_gap_frac": fin["family_gap_frac"],
                     "theta_P": fin["theta_P"], "theta_D": fin["theta_D"], "stop_rule": v["stop_rule"],
                     "wp_admissible": wp["admissible"], "wp_star": wp["w_p_star"],
                     "lambda_rule_would_pick": wp["lambda_star_rule"],
                     "eval": ev_refit},
           "freeze": {"published": freeze["published"], "train_ba_at_published": freeze["train_ba_at_published"],
                      "eval": ev_freeze},
           "secondary_tau_B": tb}
    (d / f"eval_{model}.json").write_text(json.dumps(doc, indent=1, default=str))
    print(f"wrote {d / f'eval_{model}.json'}")


# ------------------------------------------------------------------------ aggregate


def stage_aggregate(out: Path, registry: str) -> None:
    from scripts import theta_verdict as tv

    man = json.loads((out / "split_manifest.json").read_text())
    seeds = [man["split"]["primary_seed"], *man["split"]["robustness_seeds"]]
    ev = {}
    for sd in seeds:
        for m in MODELS:
            f = out / f"seed_{sd}" / f"eval_{m}.json"
            if f.exists():
                ev[(sd, m)] = json.loads(f.read_text())
    # cross-model pooled Z AUROC on each seed's test windows
    cross = {}
    for sd in seeds:
        if not all((sd, m) in ev for m in MODELS):
            continue
        zs = {"refit": ([], []), "freeze": ([], [])}
        for m in MODELS:
            e = ev[(sd, m)]
            csvp = Path(e["test"]["csv"])
            for arm in ("refit", "freeze"):
                if arm == "refit":
                    v = json.loads((out / f"seed_{sd}" / "refit" / m / "verdict_final.json").read_text())
                    vh = dr.verdict_for_holdout(v)
                else:
                    vh = json.loads(FREEZE.read_text())["models"][m]["verdict_for_holdout"]
                s = Scored(vh, csvp)
                zs[arm][0].extend(s.z)
                zs[arm][1].extend(w.slo_met for w in s.windows)
        cross[str(sd)] = {arm: auroc(z, y) for arm, (z, y) in zs.items()}
    summary = {"generated_at_utc": now(), "seeds": seeds, "cross_model_pooled_z_auroc": cross, "per_model": {}}
    for m in MODELS:
        rows = [ev[(sd, m)] for sd in seeds if (sd, m) in ev]
        if not rows:
            continue
        params = {k: [r["refit"]["published"][k] for r in rows] for k in ("theta", "w_p", "tau_crit", "tau_high", "delta_crit", "delta_high")}
        spread = {}
        for k, vs in params.items():
            med = statistics.median(vs)
            spread[k] = {"values": vs, "min": min(vs), "median": med, "max": max(vs),
                         "rel_range": ((max(vs) - min(vs)) / med) if med else None}
        mets = defaultdict(list)
        for r in rows:
            for arm in ("refit", "freeze"):
                pm = r[arm]["eval"]["subsets"]["all"]["point"]
                mets[f"{arm}.ba"].append(pm["ba"])
                mets[f"{arm}.auroc"].append(pm["auroc"])
                mets[f"{arm}.spearman"].append(pm["spearman_health"])
                mets[f"{arm}.rec_nodwell"].append(pm["nodwell"]["recall_both_tpot"])
                mets[f"{arm}.fa_nodwell"].append(pm["nodwell"]["false_alarm"])
                mets[f"{arm}.rec_dwell2"].append(pm["dwell2"]["recall_both_tpot"])
                mets[f"{arm}.fa_dwell2"].append(pm["dwell2"]["false_alarm"])
                mets[f"{arm}.ceiling_z1"].append(pm["recall_ceiling_z_lt_1_both_tpot"])
                g = r[arm]["eval"]["subsets"]["all"]["gates"]
                mets[f"{arm}.A_pass"].append(g["A"]["passed"])
                mets[f"{arm}.B_pass"].append(g["B_nodwell"]["passed"])
            mets["delta_ba_ci"].append(r["refit"]["eval"]["subsets"]["all"]["boot"].get("delta_ba", {}).get("ci95"))
        summary["per_model"][m] = {
            "param_spread": spread,
            "median_params": {k: spread[k]["median"] for k in spread},
            "metrics_by_seed": {k: v for k, v in mets.items()},
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({"cross": cross, **{m: {"spread": {k: (round(v["min"], 3), round(v["median"], 3), round(v["max"], 3))
                                                       for k, v in s["param_spread"].items()}}
                                        for m, s in summary["per_model"].items()}}, indent=1))


def stage_mdecomp(out: Path) -> None:
    """Post-hoc disclosure (added after the pre-registration, selects nothing): the freeze
    parameters on the accept stage's own M validation CSVs, split by primitive (hold vs
    steps / ramp / bursts) - where the M acceptance failure came from."""
    acc = json.loads(ACCEPT.read_text())
    freeze = json.loads(FREEZE.read_text())
    res = {}
    for m in MODELS:
        csvp = Path(acc["models"][m]["validation_csv"])
        prim = {}
        with open(csvp, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                prim[r["scenario_id"]] = r["primitive"]
        s = Scored(freeze["models"][m]["verdict_for_holdout"], csvp)
        idx_all = list(range(len(s.windows)))
        groups = {"all": idx_all,
                  "hold": [i for i in idx_all if prim[s.windows[i].scenario_id] == "hold"],
                  "dynamic": [i for i in idx_all if prim[s.windows[i].scenario_id] != "hold"]}
        for p in sorted(set(prim.values())):
            groups[p] = [i for i in idx_all if prim[s.windows[i].scenario_id] == p]
        res[m] = {g: point_metrics(s, ii) for g, ii in groups.items() if ii}
    (out / "M_decomposition_freeze.json").write_text(json.dumps(
        {"what": stage_mdecomp.__doc__, "generated_at_utc": now(), "accept_sha256": sha(ACCEPT), "models": res},
        indent=1, default=str))
    for m, g in res.items():
        for k, p in g.items():
            print(m, k, p["cells"], p["windows"], p["violating"], p["both_tpot_windows"],
                  "BA", None if p["ba"] is None else round(p["ba"], 3), "AUC", None if p["auroc"] is None else round(p["auroc"], 3),
                  "rec0", None if p["nodwell"]["recall_both_tpot"] is None else round(p["nodwell"]["recall_both_tpot"], 3),
                  "fa0", None if p["nodwell"]["false_alarm"] is None else round(p["nodwell"]["false_alarm"], 3),
                  "ceilZ1", None if p["recall_ceiling_z_lt_1_both_tpot"] is None else round(p["recall_ceiling_z_lt_1_both_tpot"], 3))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prereg", "run", "aggregate", "mdecomp"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=PRIMARY_SEED)
    ap.add_argument("--model", default=None)
    ap.add_argument("--registry", required=True)
    a = ap.parse_args(argv)
    if a.stage == "prereg":
        stage_prereg(a.out, a.registry)
    elif a.stage == "run":
        stage_run(a.out, a.seed, a.model, a.registry)
    elif a.stage == "mdecomp":
        stage_mdecomp(a.out)
    else:
        stage_aggregate(a.out, a.registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

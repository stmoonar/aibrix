#!/usr/bin/env python3
"""The frozen accept metrics next to a conservative variant that counts the dropped
zero-token windows (stage H preparation, 2026-10-04). DRAFT - disclosure only.

The gate does not change: ``dline_refit accept`` (A, B' at dwell 1, D) is the only verdict.
This tool reports, per model and for one windows CSV, three variants side by side:

* ``frozen`` - the windows the loader keeps (``SignalSpec.load`` under the frozen spec,
  label and ramp trim): the accept numbers. As a self-check the same numbers are also taken
  from ``dline_refit.evaluate_model`` (the accept code path itself) and must agree;
* ``conservative_failure`` - frozen + every dropped zero-token window with failure evidence
  (an unserved request sent in it; the frozen label calls it violated, class ``unserved``);
* ``conservative`` - frozen + every dropped zero-token window with a backlog
  (``avg_running + avg_waiting > 0``) or failure evidence.

An added window is violating and never CRITICAL (Z >= 1, the stand-in for the controller's
Z = None on a zero-token window, :mod:`scripts.analysis.h_dropped_windows`), so it is a miss
for BA, for every recall, and for B' if its severity reaches the cut. Note: a window without
any latency sample has severity ``UNSERVED_MIN_RATIO`` (2.0) under the label, below every
model's sealed B' cut (8.8-12.6 on this freeze), so B' is unchanged by construction - the
conservative variant moves A and the all-violation recall, not B'.

Metrics are composed from the accept functions (``threshold_balanced_accuracy``,
``theta_verdict.dwell_acceptance`` / ``critical_dwell_flags``, ``dline_refit.acceptance_bootstrap``
/ ``acceptance_criteria`` / ``b_prime_evaluation``) on an explicit window list; nothing of the
labelling or the criteria is re-implemented.

Inputs: the freeze file and, per model, ONE windows CSV - at H the validation CSV accept
wrote (``<freeze stem>.accept.d/<model>_validation.csv``, exactly the rows accept scored);
for the dry run a training fitting CSV (``fit/gateway/<model>/fit/<model>_fitting.csv``,
whose frozen BA must equal the freeze's ``train_ba_at_published``). Output: one JSON,
written once (refuses an existing file).

    cd tre/deploy && PYTHONPATH=../common:.:../controller:../service-manager:../calibration:../replayer:../ui \\
      python3 -m scripts.analysis.h_conservative_score --freeze-file <params_freeze.json> \\
        --csv dsqwen-7b=<csv> --csv dsllama-8b=<csv> --csv dsqwen-14b=<csv> --out <file.json>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.analysis import h_dropped_windows as hd

VARIANTS = ("frozen", "conservative_failure", "conservative")


def _num(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def score_windows(entry: Mapping[str, Any], windows: Sequence[Any], *, b_prime_cfg: Mapping[str, Any],
                  n_resamples: int, seed: int) -> dict:
    """A, B (disclosed), B' and the all-violation recall of one window list, through the
    accept functions (``dline_refit.evaluate_model`` minus its CSV loading)."""
    from tre_calibration.fit import threshold_balanced_accuracy

    from scripts import dline_refit as dl
    from scripts import theta_verdict as tv

    vh = entry["verdict_for_holdout"]
    win = dl.windowing_of(entry)
    dwell, window_ms = int(win["dwell_windows"]), float(win["window_ms"])
    theta, tau_crit = float(vh["published"]["theta_m"]), float(vh["published"]["tau_crit"])
    direction = vh["fit_config"]["direction"]
    violating = sum(1 for w in windows if not w.slo_met)
    h = {"windows": len(windows), "violating": violating,
         "at_published_theta": threshold_balanced_accuracy(windows, theta=theta, direction=direction),
         "with_dwell": tv.dwell_acceptance(windows, theta=theta, tau_crit=tau_crit, direction=direction,
                                           dwell_windows=dwell, window_ms=window_ms)}
    crit = tv.critical_dwell_flags(windows, theta=theta, tau_crit=tau_crit, direction=direction,
                                   dwell_windows=dwell, window_ms=window_ms)
    boot = dl.acceptance_bootstrap(windows, crit, theta=theta, direction=direction,
                                   n_resamples=n_resamples, seed=seed)
    criteria = dl.acceptance_criteria(entry, h, boot)
    criteria["B_prime"] = dl.b_prime_evaluation(windows, theta=theta, tau_crit=tau_crit, direction=direction,
                                                window_ms=window_ms, cfg=b_prime_cfg, n_resamples=n_resamples,
                                                seed=seed)
    return summarise(h, criteria)


def summarise(h: Mapping[str, Any], criteria: Mapping[str, Any]) -> dict:
    """The numbers the report compares (one flat dict per variant)."""
    a, bp = criteria["A"], criteria["B_prime"]
    by = bp.get("by_dwell") or {}
    gate_dwell = str(bp.get("dwell_windows"))
    g = by.get(gate_dwell) or {}
    return {
        "windows": h["windows"], "violating": h["violating"],
        "violation_classes": {k: v["windows"] for k, v in h["with_dwell"]["violation_classes"].items()},
        "A_ba": _num(a["criteria"][0]["value"]), "A_ba_ci95_low": _num(a["criteria"][1]["value"]),
        "A_passed": bool(a["passed"]),
        "B_prime_dwell": bp.get("dwell_windows"),
        "B_prime_recall_severe": _num(g.get("recall_severe")),
        "B_prime_recall_severe_ci95": g.get("recall_severe_ci95"),
        "B_prime_false_alarm": _num(g.get("false_alarm")),
        "B_prime_false_alarm_ci95": g.get("false_alarm_ci95"),
        "B_prime_severe_windows": bp.get("severe_windows"),
        "B_prime_passed": bool(bp.get("passed")),
        "recall_all_violations_gate_dwell": _num(g.get("recall_all")),
        "recall_all_violations_dwell2": _num((by.get("2") or {}).get("recall_all")),
        "old_B_recall_both_tpot_dwell2": _num(criteria["B"]["criteria"][0]["value"]),
        "violations_by_band": bp.get("violations_by_band"),
    }


def score_model(entry: Mapping[str, Any], csv_path: Path, *, b_prime_cfg: Mapping[str, Any], n_resamples: int,
                seed: int, check_accept_path: bool = True) -> dict:
    """The three variants of one model on one CSV (+ the accept-path self-check)."""
    from scripts import dline_refit as dl

    spec, label, trim = hd.frozen_spec_and_label(entry)
    theta = float(entry["verdict_for_holdout"]["published"]["theta_m"])
    frozen = spec.load(csv_path, label, trim)
    rows = hd.read_rows(csv_path)
    signals = hd.row_signals(rows, spec)
    phantoms, tiers, added = hd.phantom_windows(rows, signals, label, theta=theta, trim=trim)
    failure_only = [w for w, t in zip(phantoms, tiers) if t == hd.TIER_FAILURE]
    out: dict[str, Any] = {"csv": str(csv_path), "csv_sha256": dl.sha256_file(csv_path), "added": added,
                           "variants": {}}
    for name, extra in (("frozen", []), ("conservative_failure", failure_only), ("conservative", phantoms)):
        out["variants"][name] = score_windows(entry, [*frozen, *extra], b_prime_cfg=b_prime_cfg,
                                              n_resamples=n_resamples, seed=seed)
    if check_accept_path:
        ev = dl.evaluate_model(entry, csv_path, n_resamples=n_resamples, seed=seed, b_prime_cfg=b_prime_cfg)
        ref = summarise(ev["holdout_report"], ev["criteria"])
        mine = out["variants"]["frozen"]
        diffs = sorted(k for k in ref if ref[k] != mine[k])
        out["accept_path_check"] = {"matches_evaluate_model": not diffs, "differences": diffs}
    train = _num(entry.get("train_ba_at_published"))
    out["train_ba_at_published"] = train
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    from scripts import dline_refit as dl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze-file", type=Path, required=True)
    ap.add_argument("--csv", action="append", default=[], metavar="MODEL=PATH", required=True,
                    help="one windows CSV per model (accept's validation CSV at H; a fitting CSV for the dry run)")
    ap.add_argument("--dwell-windows", type=int, default=dl.ONLINE_DWELL_WINDOWS,
                    help="B' gate dwell (default: the controller's, as accept)")
    ap.add_argument("--resamples", type=int, default=dl.ACCEPT_RESAMPLES)
    ap.add_argument("--label", default="", help="free text recorded in the output (e.g. 'DRY RUN on training')")
    ap.add_argument("--out", type=Path, required=True, help="JSON output (refuses to overwrite)")
    args = ap.parse_args(argv)
    if args.out.exists():
        ap.error(f"{args.out} exists (write once)")
    doc = dl.verify_freeze(args.freeze_file)
    cfgs, bp_summary, problems = dl.b_prime_inputs(doc, None, args.dwell_windows, "--dwell-windows")
    if problems:
        ap.error("; ".join(problems))
    result: dict[str, Any] = {
        "what": ("frozen accept metrics next to the conservative variant (dropped zero-token windows with "
                 "backlog / failure evidence counted as non-CRITICAL misses). DRAFT, disclosure only: the gate "
                 "is dline_refit accept, unchanged"),
        "label": args.label,
        "freeze": {"path": str(args.freeze_file), "sha256": dl.sha256_file(args.freeze_file),
                   "freeze_sha256": doc["freeze_sha256"]},
        "code": dl.code_state(), "bootstrap": {"n_resamples": args.resamples, "seed": dl.SEED},
        "b_prime": bp_summary, "phantom_z": hd.PHANTOM_Z, "models": {},
    }
    for text in args.csv:
        model, _, path = text.partition("=")
        if model not in doc["models"]:
            ap.error(f"{model} is not in the freeze")
        result["models"][model] = score_model(doc["models"][model], Path(path), b_prime_cfg=cfgs[model],
                                              n_resamples=args.resamples, seed=dl.SEED)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
    for model, r in result["models"].items():
        line = [f"[{model}] added {r['added']}"]
        for v in VARIANTS:
            s = r["variants"][v]
            line.append(f"  {v:21s} windows {s['windows']:5d} viol {s['violating']:5d}  BA {s['A_ba']:.4f} "
                        f"(CI low {s['A_ba_ci95_low']:.4f}) A {'pass' if s['A_passed'] else 'FAIL'}  "
                        f"B' rec {s['B_prime_recall_severe']} fa {s['B_prime_false_alarm']} "
                        f"{'pass' if s['B_prime_passed'] else 'FAIL'}  recall_all d1 "
                        f"{s['recall_all_violations_gate_dwell']}")
        line.append(f"  accept-path check {r.get('accept_path_check')}  train BA {r['train_ba_at_published']}")
        print("\n".join(line))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

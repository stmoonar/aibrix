"""POST-HOC disclosure: how CRITICAL behaves around tau_crit on a held-out window set.

NOT USED FOR ANY DECISION. The frozen tau_crit (1 - delta_crit of the freeze) is the only
one that gates (``dline_refit accept``, RUN plan H1). This tool exists because the fitted
delta_crit sat at or near the edge of its grid [0, 0.50] (2026-10-03 round: 7b 0.50, 8b
0.48, 14b 0.47), so the owner asked what CRITICAL looks like on M around the frozen
tau_crit and what a lower tau_crit would have given.

Same windows as accept (the frozen loader on accept's validation CSV: SignalSpec,
LabelDefinition, ramp trim of the freeze), same theta, same B' severity cut and
bootstrap (cell bootstrap, seed of accept); only tau_crit varies. Per tau_crit and dwell
(1 = the controller's, 2 disclosed): B' recall of severe violations, false alarm (share of
healthy windows CRITICAL), all-violation recall, each with the B' CI95.

Usage (from tre/deploy, PYTHONPATH as the RUN plan):
  python3 -m scripts.analysis.h_tau_crit_sweep --freeze-file $C/freeze/params_freeze.json \
    --csv dsqwen-7b=$C/freeze/params_freeze.accept.d/dsqwen-7b_validation.csv ... \
    --out $C/eval/H_tau_crit_sweep_M.posthoc.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

DEFAULT_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


def sweep_model(entry, csv_path: Path, *, cfg, grid: Sequence[float], n_resamples: int, seed: int) -> dict:
    from scripts import dline_refit as dl
    from scripts.analysis import h_dropped_windows as hd

    spec, label, trim = hd.frozen_spec_and_label(entry)
    vh = entry["verdict_for_holdout"]
    win = dl.windowing_of(entry)
    window_ms = float(win["window_ms"])
    theta, frozen_tau = float(vh["published"]["theta_m"]), float(vh["published"]["tau_crit"])
    direction = vh["fit_config"]["direction"]
    windows = spec.load(csv_path, label, trim)
    taus = sorted({round(float(t), 4) for t in grid} | {round(frozen_tau, 4)})
    rows = []
    for tau in taus:
        ev = dl.b_prime_evaluation(windows, theta=theta, tau_crit=tau, direction=direction, window_ms=window_ms,
                                   cfg=cfg, n_resamples=n_resamples, seed=seed)
        row: dict[str, Any] = {"tau_crit": tau, "is_frozen": abs(tau - frozen_tau) < 1e-9,
                               "gate_met_at_this_tau": bool(ev.get("passed")),
                               "violations_by_band": ev.get("violations_by_band")}
        for d, g in (ev.get("by_dwell") or {}).items():
            row[f"dwell{d}"] = {k: g.get(k) for k in ("recall_severe", "recall_severe_ci95", "false_alarm",
                                                      "false_alarm_ci95", "recall_all")}
        rows.append(row)
    return {"csv": str(csv_path), "csv_sha256": dl.sha256_file(csv_path), "theta": theta,
            "frozen_tau_crit": frozen_tau, "windows": len(windows),
            "violating": sum(1 for w in windows if not w.slo_met),
            "severity_cut": cfg.get("severity_cut"), "sweep": rows}


def main(argv: Optional[Sequence[str]] = None) -> int:
    from scripts import dline_refit as dl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze-file", type=Path, required=True)
    ap.add_argument("--csv", action="append", default=[], metavar="MODEL=PATH", required=True)
    ap.add_argument("--grid", default=",".join(f"{t:.2f}" for t in DEFAULT_GRID))
    ap.add_argument("--resamples", type=int, default=dl.ACCEPT_RESAMPLES)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, required=True, help="JSON output (refuses to overwrite)")
    args = ap.parse_args(argv)
    if args.out.exists():
        ap.error(f"{args.out} exists (write once)")
    doc = dl.verify_freeze(args.freeze_file)
    cfgs, bp_summary, problems = dl.b_prime_inputs(doc, None, dl.ONLINE_DWELL_WINDOWS, "--dwell-windows")
    if problems:
        ap.error("; ".join(problems))
    grid = [float(x) for x in args.grid.split(",") if x.strip()]
    result: dict[str, Any] = {
        "what": "POST-HOC tau_crit sweep on held-out windows. Disclosure only, NOT USED FOR ANY DECISION; "
                "the frozen tau_crit and dline_refit accept are the only gate.",
        "label": args.label,
        "freeze": {"path": str(args.freeze_file), "sha256": dl.sha256_file(args.freeze_file),
                   "freeze_sha256": doc["freeze_sha256"]},
        "code": dl.code_state(), "bootstrap": {"n_resamples": args.resamples, "seed": dl.SEED},
        "b_prime": bp_summary, "grid": grid, "models": {},
    }
    for text in args.csv:
        model, _, path = text.partition("=")
        if model not in doc["models"]:
            ap.error(f"{model} is not in the freeze")
        result["models"][model] = sweep_model(doc["models"][model], Path(path), cfg=cfgs[model], grid=grid,
                                              n_resamples=args.resamples, seed=dl.SEED)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")

    def f(x):
        return "  n/a" if x is None else f"{x:.3f}"

    for model, r in result["models"].items():
        print(f"[{model}] POST-HOC  theta {r['theta']:.1f}  frozen tau_crit {r['frozen_tau_crit']}  "
              f"windows {r['windows']} violating {r['violating']}")
        for row in r["sweep"]:
            d1, d2 = row.get("dwell1") or {}, row.get("dwell2") or {}
            print(f"  tau {row['tau_crit']:.2f}{'*' if row['is_frozen'] else ' '} d1 rec_sev {f(d1.get('recall_severe'))} "
                  f"{d1.get('recall_severe_ci95')} fa {f(d1.get('false_alarm'))} {d1.get('false_alarm_ci95')} "
                  f"rec_all {f(d1.get('recall_all'))} | d2 rec_sev {f(d2.get('recall_severe'))} "
                  f"fa {f(d2.get('false_alarm'))}  gate {'met' if row['gate_met_at_this_tau'] else 'not met'}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

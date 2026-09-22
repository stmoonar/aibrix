#!/usr/bin/env python3
"""Publishability verdict for one model's theta, and its hold-out check (plan §6.3 B5).

Two subcommands, run in this order by the campaign's fit plan:

``verdict``
    Fits theta and delta_crit on the merged fitting CSV exactly as ``tre_calibration.cli``
    does (same loader, same shared label, same TSS recompute, same ``ThetaFitConfig``),
    runs the cell-level bootstrap for both, fits each family CSV, and applies the two
    campaign rules to the result:

    * the stop rule (``adaptive_boundary.stop_rule``: bootstrap publish rate, CI half-width
      as a fraction of theta, windows per family near the boundary);
    * the family rule (``calibration_campaign.family_theta_verdict``: the merged theta
      when every family theta sits inside its CI, otherwise the LARGEST family theta).

    It writes the number that would be published and why. It never touches a registry.

``holdout``
    Scores the published theta / tau_crit on the held-out validation CSV (shape M) with
    the verdict's own label, TSS parameters and trim - nothing is refitted on it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from tre_calibration.bootstrap import bootstrap_theta, bootstrap_theta_and_delta_crit
from tre_calibration.dataset import CalibrationWindow, TssRecompute, load_windows_from_csv
from tre_calibration.fit import ThetaFitConfig, fit_delta_margins, threshold_balanced_accuracy
from tre_calibration.labels import LabelDefinition
from tre_calibration.profile import theta_fit_block
from tre_common.tss import DEFAULT_EMA_TAU_MS, TSS_UNITS

from scripts import adaptive_boundary as boundary

#: A window is "near the boundary" when its signal is within this fraction of theta.
BOUNDARY_BAND = 0.20


def _load(path: str | Path, label: LabelDefinition, tss: TssRecompute, trim: int, lambda_wait: float) -> list[CalibrationWindow]:
    return load_windows_from_csv(
        path,
        latency_slo_ms=label.latency_slo_ms(),
        trim_ramp_windows=trim,
        lambda_wait=lambda_wait,
        tss=tss,
    )


def _band(windows: Sequence[CalibrationWindow], theta: float) -> tuple[int, int]:
    near = [w for w in windows if abs(w.signal / theta - 1.0) <= BOUNDARY_BAND]
    return len(near), sum(1 for w in near if not w.slo_met)


def _waiting_stats(path: str | Path, label: LabelDefinition) -> dict[str, Any]:
    """Share of windows with a non-zero waiting queue, overall and among violated ones -
    how identifiable lambda_wait is on this data."""
    from tre_calibration.labels import label_window

    total = nonzero = viol = viol_nonzero = 0
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lab = label_window(row, label.latency_slo_ms())
            if lab is None:
                continue
            waiting = float(row.get("avg_waiting") or 0.0)
            total += 1
            nonzero += waiting > 0
            if not lab.slo_met:
                viol += 1
                viol_nonzero += waiting > 0
    return {
        "labelled_windows": total,
        "waiting_nonzero_fraction": nonzero / total if total else None,
        "violating_windows": viol,
        "waiting_nonzero_in_violating": viol_nonzero / viol if viol else None,
    }


def cmd_verdict(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.calibration_campaign import family_theta_verdict

    label = LabelDefinition(args.ttft_p95_ms, args.tpot_p95_ms)
    tss = TssRecompute(
        w_p=args.w_p, lambda_wait=args.lambda_wait, qmin=args.qmin,
        ema_tau_ms=args.ema_tau_ms if args.ema_tau_ms > 0 else None,
    )
    config = ThetaFitConfig()
    windows = _load(args.fitting_csv, label, tss, args.trim_ramp_windows, args.lambda_wait)
    fit = config.fit(windows)
    if not fit.publish or fit.theta is None:
        raise SystemExit(f"{args.model}: merged fit did not publish ({fit.reject_reason})")
    theta = float(fit.theta)
    delta = fit_delta_margins(windows, theta=theta)
    boot, dboot = bootstrap_theta_and_delta_crit(
        windows, n_resamples=args.n_resamples, seed=args.seed, config=config
    )
    half = (
        (boot.theta_p97_5 - boot.theta_p2_5) / 2.0
        if boot.theta_p2_5 is not None and boot.theta_p97_5 is not None
        else math.inf
    )
    band_n, band_viol = _band(windows, theta)

    families: dict[str, Any] = {}
    family_windows_near: dict[str, int] = {}
    for spec in args.family or []:
        name, _, path = spec.partition("=")
        fw = _load(path, label, tss, args.trim_ramp_windows, args.lambda_wait)
        ff = config.fit(fw)
        fb = bootstrap_theta(fw, n_resamples=args.family_resamples, seed=args.seed, config=config)
        near, near_viol = _band(fw, theta)
        family_windows_near[name] = near
        families[name] = {
            "csv": path,
            "theta": ff.theta,
            "publish": ff.publish,
            "reject_reason": ff.reject_reason,
            "windows": len(fw),
            "cells": len({w.scenario_id for w in fw}),
            "violating": sum(1 for w in fw if not w.slo_met),
            "theta_ci95": [fb.theta_p2_5, fb.theta_p97_5],
            "publish_rate": fb.publish_rate,
            "near_merged_theta": near,
            "near_merged_theta_violating": near_viol,
        }
    family_thetas = {n: f["theta"] for n, f in families.items() if f["publish"] and f["theta"]}
    fam = family_theta_verdict(theta, half, family_thetas)
    stop = boundary.stop_rule(
        publish_rate=boot.publish_rate,
        theta=theta,
        ci_half_width=half,
        family_boundary_windows=family_windows_near,
    )
    published_theta = float(fam["theta"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "label_def": label.as_dict(),
        "tss": {**tss.as_dict(), "units": TSS_UNITS},
        "trim_ramp_windows": args.trim_ramp_windows,
        "fit_config": config.as_dict(),
        "merged": {
            "csv": args.fitting_csv,
            "windows": len(windows),
            "cells": len({w.scenario_id for w in windows}),
            "violating": sum(1 for w in windows if not w.slo_met),
            "theta": theta,
            "fit": theta_fit_block(fit),
            "bootstrap": {
                "n_resamples": boot.n_resamples,
                "seed": args.seed,
                "publish_rate": boot.publish_rate,
                "theta_ci95": [boot.theta_p2_5, boot.theta_p97_5],
                "theta_p50": boot.theta_p50,
                "ci_half_width": half,
                "ci_half_width_fraction": half / theta,
            },
            "near_theta": {"band": BOUNDARY_BAND, "windows": band_n, "violating": band_viol},
            "delta_crit": {
                "method": delta.crit.method,
                "delta": delta.crit.delta,
                "tau_crit": delta.crit.tau,
                "balanced_accuracy": delta.crit.balanced_accuracy,
                "recall_pos": delta.crit.recall_pos,
                "clamped": delta.crit.clamped,
                "clamp_reason": delta.crit.clamp_reason,
                "used_fallback": delta.crit.used_fallback,
                "critical_windows": delta.labels.critical_positive_count,
                "bootstrap": dboot.as_dict(),
            },
            "delta_high": {"delta": delta.high.delta, "tau_high": delta.high.tau, "method": delta.high.method},
            "waiting": _waiting_stats(args.fitting_csv, label),
        },
        "families": families,
        "family_verdict": fam,
        "stop_rule": stop.as_dict(),
        "published": {
            "theta_m": published_theta,
            "tau_crit": 1.0 - delta.crit.delta,
            "delta_crit": delta.crit.delta,
            "source": fam["publish"],
            "stop_rule_satisfied": stop.satisfied,
        },
    }


def cmd_holdout(args: argparse.Namespace) -> dict[str, Any]:
    verdict = json.loads(Path(args.verdict).read_text(encoding="utf-8"))
    label = LabelDefinition(verdict["label_def"]["ttft_p95_ms"], verdict["label_def"]["tpot_p95_ms"])
    t = verdict["tss"]
    tss = TssRecompute(w_p=t["w_p"], lambda_wait=t["lambda_wait"], qmin=t["qmin"], ema_tau_ms=t["ema_tau_ms"])
    windows = _load(args.validation_csv, label, tss, int(verdict["trim_ramp_windows"]), t["lambda_wait"])
    theta = float(verdict["published"]["theta_m"])
    tau_crit = float(verdict["published"]["tau_crit"])
    direction = verdict["fit_config"]["direction"]
    at_theta = threshold_balanced_accuracy(windows, theta=theta, direction=direction)
    violated = [w for w in windows if not w.slo_met]
    healthy = [w for w in windows if w.slo_met]
    crit_recall = (sum(1 for w in violated if w.signal / theta < tau_crit) / len(violated)) if violated else None
    crit_false_alarm = (sum(1 for w in healthy if w.signal / theta < tau_crit) / len(healthy)) if healthy else None
    merged_theta = float(verdict["merged"]["theta"])
    at_merged = threshold_balanced_accuracy(windows, theta=merged_theta, direction=direction)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": verdict["model"],
        "validation_csv": args.validation_csv,
        "label_def": verdict["label_def"],
        "windows": len(windows),
        "cells": len({w.scenario_id for w in windows}),
        "violating": len(violated),
        "published_theta": theta,
        "tau_crit": tau_crit,
        "at_published_theta": at_theta,
        "at_merged_theta": at_merged,
        "critical_recall_of_violating": crit_recall,
        "critical_false_alarm_on_healthy": crit_false_alarm,
        "note": "raw windows, no dwell; the plan's acceptance adds EMA+dwell on the live path",
    }


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verdict")
    v.add_argument("--model", required=True)
    v.add_argument("--fitting-csv", required=True)
    v.add_argument("--family", action="append", default=[], help="NAME=CSV, repeatable")
    v.add_argument("--w-p", type=float, required=True)
    v.add_argument("--lambda-wait", type=float, required=True)
    v.add_argument("--qmin", type=float, default=1.0)
    v.add_argument("--ema-tau-ms", type=float, default=DEFAULT_EMA_TAU_MS)
    v.add_argument("--ttft-p95-ms", type=float, required=True)
    v.add_argument("--tpot-p95-ms", type=float, required=True)
    v.add_argument("--trim-ramp-windows", type=int, default=1)
    v.add_argument("--n-resamples", type=int, default=1000)
    v.add_argument("--family-resamples", type=int, default=200)
    v.add_argument("--seed", type=int, default=20260922)
    v.add_argument("--output", required=True)
    h = sub.add_parser("holdout")
    h.add_argument("--verdict", required=True)
    h.add_argument("--validation-csv", required=True)
    h.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    report = cmd_verdict(args) if args.command == "verdict" else cmd_holdout(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.command == "verdict":
        p = report["published"]
        print(
            f"[{report['model']}] theta={report['merged']['theta']:.4g} "
            f"CI={report['merged']['bootstrap']['theta_ci95']} -> publish {p['theta_m']:.4g} "
            f"({p['source']}), delta_crit={p['delta_crit']:.3f}, stop_rule={p['stop_rule_satisfied']}"
        )
    else:
        print(f"[{report['model']}] hold-out BA={report['at_published_theta']['balanced_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Publishability verdict for one model's threshold, and its hold-out check (plan §6.3 B5).

One pipeline for every signal (plan §6.9 item 5): TSS and each alternative signal of the
ablation (``--signal queue_len|decode_tps|prefill_tps``) go through the same loader, the
same shared label, the same ``ThetaFitConfig`` fit, the same cell bootstrap of theta and
both band margins, the same stop rule, family rule and hold-out scoring. The signal only
decides how a window's value is computed (TSS recompute, or ``tre_common.alt_signals``
through the same tau-EMA), its orientation and its theta candidate grid.
``calibration/scripts/fit_alt_thresholds.py`` is a driver over :func:`verdict_report`.

Two subcommands, run in this order by the campaign's fit plan:

``verdict``
    Fits theta and delta_crit / delta_high on the merged fitting CSV exactly as
    ``tre_calibration.cli`` does, runs the cell-level bootstrap for all three, fits each
    family CSV, and applies the two campaign rules to the result:

    * the stop rule (``adaptive_boundary.stop_rule``: bootstrap publish rate, CI half-width
      as a fraction of theta, windows per family near the boundary);
    * the family rule (``calibration_campaign.family_theta_verdict``: the merged theta
      when every family theta sits inside its CI, otherwise the conservative family theta -
      the largest for TSS, the smallest for a lower_is_healthier signal).

    It also records the ranking quality of the signal (AUROC in its prior direction, flag
    ``inert`` below 0.6) and the balanced accuracy of the fit in BOTH directions. It writes
    the numbers that would be published and why. It never touches a registry.

``holdout``
    Scores the published theta / bands on the held-out validation CSV (shape M) with the
    verdict's own label, signal definition and trim - nothing is refitted on it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tre_calibration.alt_signals import (
    INERT_AUROC,
    alt_signal_candidate_grid,
    alt_signal_direction,
    alt_signal_names,
    alt_signal_transform,
)
from tre_calibration.bootstrap import bootstrap_theta, bootstrap_theta_and_delta_crit
from tre_calibration.dataset import CalibrationWindow, TssRecompute, load_windows_from_csv
from tre_calibration.evaluate import evaluate_signal_direction
from tre_calibration.fit import (
    ThetaFitConfig,
    fit_delta_margins,
    signal_z,
    threshold_balanced_accuracy,
)
from tre_calibration.labels import LabelDefinition, add_label_arguments, label_def_from_args
from tre_calibration.profile import theta_fit_block
from tre_common.tss import DEFAULT_EMA_TAU_MS, TSS_UNITS

from scripts import adaptive_boundary as boundary

#: A window is "near the boundary" when its z is within this distance of 1.
BOUNDARY_BAND = 0.20

SIGNALS = ("tss", *alt_signal_names())

_OPPOSITE = {"higher_is_healthier": "lower_is_healthier", "lower_is_healthier": "higher_is_healthier"}


class VerdictError(RuntimeError):
    """The merged fit did not publish a threshold."""


@dataclass(frozen=True)
class SignalSpec:
    """How one window's signal value is computed - the only per-signal part of the fit.

    ``label_lambda_wait`` only feeds the surplus label of the delta_high fit (queue depth
    of a healthy window); it is recorded so every arm of the ablation can be checked to
    have used the same label.
    """

    signal: str
    label_lambda_wait: float
    tss: TssRecompute | None = None
    ema_tau_ms: float | None = DEFAULT_EMA_TAU_MS

    @property
    def direction(self) -> str:
        return "higher_is_healthier" if self.signal == "tss" else alt_signal_direction(self.signal)

    @property
    def candidate_grid(self) -> str:
        return "quantile" if self.signal == "tss" else alt_signal_candidate_grid(self.signal)

    def load(self, path: str | Path, label: LabelDefinition, trim: int) -> list[CalibrationWindow]:
        if self.signal == "tss":
            return load_windows_from_csv(
                path, latency_slo_ms=label, trim_ramp_windows=trim,
                lambda_wait=self.label_lambda_wait, tss=self.tss,
            )
        return load_windows_from_csv(
            path, latency_slo_ms=label, trim_ramp_windows=trim,
            lambda_wait=self.label_lambda_wait,
            signal_transform=alt_signal_transform(self.signal),
            ema_tau_ms=self.ema_tau_ms,
        )

    def default_config(self) -> ThetaFitConfig:
        return ThetaFitConfig(direction=self.direction, candidate_grid=self.candidate_grid)

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "direction": self.direction,
            "candidate_grid": self.candidate_grid,
            "label_lambda_wait": self.label_lambda_wait,
            "ema_tau_ms": self.tss.ema_tau_ms if self.tss is not None else self.ema_tau_ms,
            "tss": self.tss.as_dict() if self.tss is not None else None,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SignalSpec":
        t = raw.get("tss")
        return cls(
            signal=str(raw["signal"]),
            label_lambda_wait=float(raw["label_lambda_wait"]),
            tss=(
                TssRecompute(w_p=t["w_p"], lambda_wait=t["lambda_wait"], qmin=t["qmin"], ema_tau_ms=t["ema_tau_ms"])
                if t else None
            ),
            ema_tau_ms=raw.get("ema_tau_ms"),
        )


def build_signal_spec(
    signal: str,
    *,
    w_p: float | None = None,
    lambda_wait: float | None = None,
    qmin: float = 1.0,
    ema_tau_ms: float | None = DEFAULT_EMA_TAU_MS,
    label_lambda_wait: float | None = None,
) -> SignalSpec:
    if signal not in SIGNALS:
        raise ValueError(f"signal must be one of {SIGNALS}")
    tau = ema_tau_ms if ema_tau_ms is not None and ema_tau_ms > 0 else None
    if signal == "tss":
        if w_p is None or lambda_wait is None:
            raise ValueError("--signal tss needs --w-p and --lambda-wait")
        return SignalSpec(
            signal="tss",
            label_lambda_wait=float(lambda_wait if label_lambda_wait is None else label_lambda_wait),
            tss=TssRecompute(w_p=w_p, lambda_wait=lambda_wait, qmin=qmin, ema_tau_ms=tau),
        )
    if label_lambda_wait is None:
        label_lambda_wait = lambda_wait
    if label_lambda_wait is None:
        raise ValueError(f"--signal {signal} needs --label-lambda-wait (the surplus label's queue weight)")
    return SignalSpec(signal=signal, label_lambda_wait=float(label_lambda_wait), ema_tau_ms=tau)


def _z(windows: Sequence[CalibrationWindow], theta: float, direction: str) -> list[float]:
    return [signal_z(w.signal, theta, direction) for w in windows]


def _band(windows: Sequence[CalibrationWindow], theta: float, direction: str) -> tuple[int, int]:
    near = [w for w, z in zip(windows, _z(windows, theta, direction)) if math.isfinite(z) and abs(z - 1.0) <= BOUNDARY_BAND]
    return len(near), sum(1 for w in near if not w.slo_met)


def _waiting_stats(path: str | Path, label: LabelDefinition) -> dict[str, Any]:
    """Share of windows with a non-zero waiting queue, overall and among violated ones -
    how identifiable lambda_wait is on this data."""
    from tre_calibration.labels import label_window

    total = nonzero = viol = viol_nonzero = 0
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lab = label_window(row, label)
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


def _ranking(windows: Sequence[CalibrationWindow], direction: str) -> dict[str, Any]:
    finite = [w for w in windows if math.isfinite(w.signal)]
    ev = evaluate_signal_direction(finite, direction=direction)
    return {
        "direction": direction,
        "auroc": ev.auroc,
        "spearman_health": ev.spearman_health,
        "inert": ev.auroc < INERT_AUROC,
        "inert_below": INERT_AUROC,
        "zero_value_fraction": (sum(1 for w in finite if w.signal == 0.0) / len(finite)) if finite else None,
        "distinct_values": len({w.signal for w in finite}),
    }


VIOLATION_CLASSES = ("ttft_only", "tpot_only", "both", "unserved")


def violation_class_breakdown(
    windows: Sequence[CalibrationWindow], *, theta: float, tau_crit: float | None, direction: str,
) -> dict[str, Any]:
    """Violating windows per label class (TTFT-only / TPOT-only / both / unserved) and,
    given tau_crit, the CRITICAL recall of each class - plan 6.9f criteria B/C."""
    out: dict[str, Any] = {}
    for cls in VIOLATION_CLASSES:
        members = [w for w in windows if not w.slo_met and w.violation_class == cls]
        entry: dict[str, Any] = {"windows": len(members)}
        if tau_crit is not None:
            z = _z(members, theta, direction)
            entry["critical_recall"] = (sum(1 for v in z if v < tau_crit) / len(members)) if members else None
        out[cls] = entry
    return out


#: Plan 6.9f item B / D8: CRITICAL counts only after this many consecutive new windows.
DEFAULT_HOLDOUT_DWELL_WINDOWS = 2
#: Classes criterion B gates on (TSS-observable); TTFT-only is disclosure only (item C).
CRITERION_B_CLASSES = ("both", "tpot_only")


def critical_dwell_flags(
    windows: Sequence[CalibrationWindow], *, theta: float, tau_crit: float, direction: str,
    dwell_windows: int = DEFAULT_HOLDOUT_DWELL_WINDOWS, window_ms: float = 30_000.0,
    max_gap_ms: float | None = None,
) -> list[bool]:
    """Per window, whether CRITICAL (Z < tau_crit) is *dwell-confirmed*, with the shared
    ``tre_common.dwell`` counter the controller uses: each cell is run in window order and
    a window confirms only after ``dwell_windows`` consecutive CRITICAL windows.

    Windows the label dropped (thin / missing latency) are not in ``windows``; with
    ``max_gap_ms=None`` their neighbours count as consecutive (the controller saw those
    windows too, with their signal), a finite ``max_gap_ms`` makes the gap reset the run.
    """
    from tre_common.dwell import dwell_confirmed_series

    z = _z(windows, theta, direction)
    out: list[bool] = [False] * len(windows)
    by_cell: dict[str, list[int]] = {}
    for i, w in enumerate(windows):
        by_cell.setdefault(w.scenario_id, []).append(i)
    for idx in by_cell.values():
        idx.sort(key=lambda i: (windows[i].window_start_ms if windows[i].window_start_ms is not None else i))
        ends = [
            int((windows[i].window_start_ms if windows[i].window_start_ms is not None else i * window_ms) + window_ms)
            for i in idx
        ]
        flags = [math.isfinite(z[i]) and z[i] < tau_crit for i in idx]
        for i, ok in zip(idx, dwell_confirmed_series(flags, ends, required=dwell_windows, max_gap_ms=max_gap_ms)):
            out[i] = ok
    return out


def dwell_acceptance(
    windows: Sequence[CalibrationWindow], *, theta: float, tau_crit: float, direction: str,
    dwell_windows: int = DEFAULT_HOLDOUT_DWELL_WINDOWS, window_ms: float = 30_000.0,
) -> dict[str, Any]:
    """Plan 6.9f criterion B on the deployed form (tau-EMA already in the signal + dwell):
    CRITICAL recall of both/TPOT-only violations, healthy false alarm, overall recall, and
    the per-class recall (TTFT-only is reported for disclosure, criterion C)."""
    crit = critical_dwell_flags(windows, theta=theta, tau_crit=tau_crit, direction=direction,
                                dwell_windows=dwell_windows, window_ms=window_ms)

    def rate(sel: list[int]) -> float | None:
        return (sum(1 for i in sel if crit[i]) / len(sel)) if sel else None

    viol = [i for i, w in enumerate(windows) if not w.slo_met]
    ok = [i for i, w in enumerate(windows) if w.slo_met]
    b_sel = [i for i in viol if windows[i].violation_class in CRITERION_B_CLASSES]
    classes = {
        cls: {"windows": len(sel), "critical_recall": rate(sel)}
        for cls in VIOLATION_CLASSES
        for sel in [[i for i in viol if windows[i].violation_class == cls]]
    }
    return {
        "dwell_windows": int(dwell_windows),
        "dwell_impl": "tre_common.dwell.dwell_confirmed_series (per cell, window order)",
        "critical_recall_both_tpot": rate(b_sel),
        "both_tpot_windows": len(b_sel),
        "critical_recall_of_violating": rate(viol),
        "critical_false_alarm_on_healthy": rate(ok),
        "healthy_windows": len(ok),
        "violation_classes": classes,
    }


def verdict_report(
    *,
    model: str,
    fitting_csv: str | Path,
    families: Mapping[str, str | Path],
    spec: SignalSpec,
    label: LabelDefinition,
    trim_ramp_windows: int = 1,
    config: ThetaFitConfig | None = None,
    n_resamples: int = 1000,
    family_resamples: int = 200,
    seed: int = 20260922,
) -> dict[str, Any]:
    """The verdict for ``spec``'s signal on one model - the one fit path of every arm."""
    from scripts.calibration_campaign import family_theta_verdict

    config = config or spec.default_config()
    if config.direction != spec.direction:
        raise ValueError(f"config direction {config.direction} != signal direction {spec.direction}")
    direction = spec.direction
    windows = spec.load(fitting_csv, label, trim_ramp_windows)
    fit = config.fit(windows)
    if not fit.publish or fit.theta is None:
        raise VerdictError(f"{model}/{spec.signal}: merged fit did not publish ({fit.reject_reason})")
    theta = float(fit.theta)
    delta = fit_delta_margins(windows, theta=theta, direction=direction)
    boot, dboot = bootstrap_theta_and_delta_crit(
        windows, n_resamples=n_resamples, seed=seed, config=config
    )
    half = (
        (boot.theta_p97_5 - boot.theta_p2_5) / 2.0
        if boot.theta_p2_5 is not None and boot.theta_p97_5 is not None
        else math.inf
    )
    band_n, band_viol = _band(windows, theta, direction)

    # Same criterion, orientation flipped: a signal whose wrong-way fit separates as well
    # has not demonstrated a direction (plan §6.9: direction stays the prior one, BA is
    # reported both ways).
    opposite = _OPPOSITE[direction]
    opposite_config = ThetaFitConfig(**{**config.__dict__, "direction": opposite})
    opposite_fit = opposite_config.fit(windows)

    fam_reports: dict[str, Any] = {}
    family_windows_near: dict[str, int] = {}
    for name, path in families.items():
        fw = spec.load(path, label, trim_ramp_windows)
        ff = config.fit(fw)
        fb = bootstrap_theta(fw, n_resamples=family_resamples, seed=seed, config=config)
        near, near_viol = _band(fw, theta, direction)
        family_windows_near[name] = near
        fam_reports[name] = {
            "csv": str(path),
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
    family_thetas = {n: f["theta"] for n, f in fam_reports.items() if f["publish"] and f["theta"]}
    fam = family_theta_verdict(theta, half, family_thetas, direction=direction)
    stop = boundary.stop_rule(
        publish_rate=boot.publish_rate,
        theta=theta,
        ci_half_width=half,
        family_boundary_windows=family_windows_near,
    )
    published_theta = float(fam["theta"])
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "signal": spec.signal,
        "direction": direction,
        "signal_spec": spec.as_dict(),
        "label_def": label.as_dict(),
        "trim_ramp_windows": trim_ramp_windows,
        "fit_config": config.as_dict(),
        "merged": {
            "csv": str(fitting_csv),
            "windows": len(windows),
            "cells": len({w.scenario_id for w in windows}),
            "violating": sum(1 for w in windows if not w.slo_met),
            "theta": theta,
            "fit": theta_fit_block(fit),
            "ranking": _ranking(windows, direction),
            "opposite_direction": {
                "direction": opposite,
                "publish": opposite_fit.publish,
                "theta": opposite_fit.theta,
                "fit": theta_fit_block(opposite_fit),
            },
            "bootstrap": {
                "n_resamples": boot.n_resamples,
                "seed": seed,
                "publish_rate": boot.publish_rate,
                "theta_ci95": [boot.theta_p2_5, boot.theta_p97_5],
                "theta_p50": boot.theta_p50,
                "ci_half_width": half,
                "ci_half_width_fraction": half / theta,
            },
            "near_theta": {"band": BOUNDARY_BAND, "windows": band_n, "violating": band_viol},
            "violation_classes": violation_class_breakdown(
                windows, theta=theta, tau_crit=delta.crit.tau, direction=direction,
            ),
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
            "delta_high": {
                "method": delta.high.method,
                "delta": delta.high.delta,
                "tau_high": delta.high.tau,
                "balanced_accuracy": delta.high.balanced_accuracy,
                "precision_pos": delta.high.precision_pos,
                "clamped": delta.high.clamped,
                "clamp_reason": delta.high.clamp_reason,
                "used_fallback": delta.high.used_fallback,
                "surplus_windows": delta.labels.surplus_positive_count,
            },
            "waiting": _waiting_stats(fitting_csv, label),
        },
        "families": fam_reports,
        "family_verdict": fam,
        "stop_rule": stop.as_dict(),
        "published": {
            "signal": spec.signal,
            "direction": direction,
            "theta_m": published_theta,
            "tau_crit": 1.0 - delta.crit.delta,
            "delta_crit": delta.crit.delta,
            "tau_high": 1.0 + delta.high.delta,
            "delta_high": delta.high.delta,
            "source": fam["publish"],
            "stop_rule_satisfied": stop.satisfied,
        },
    }
    if spec.tss is not None:
        report["tss"] = {**spec.tss.as_dict(), "units": TSS_UNITS}
    return report


def cmd_verdict(args: argparse.Namespace) -> dict[str, Any]:
    label = label_def_from_args(args, args.model)
    try:
        spec = build_signal_spec(
            args.signal, w_p=args.w_p, lambda_wait=args.lambda_wait, qmin=args.qmin,
            ema_tau_ms=args.ema_tau_ms, label_lambda_wait=args.label_lambda_wait,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    families: dict[str, str] = {}
    for item in args.family or []:
        name, _, path = item.partition("=")
        families[name] = path
    try:
        return verdict_report(
            model=args.model, fitting_csv=args.fitting_csv, families=families, spec=spec,
            label=label, trim_ramp_windows=args.trim_ramp_windows,
            n_resamples=args.n_resamples, family_resamples=args.family_resamples, seed=args.seed,
        )
    except VerdictError as exc:
        raise SystemExit(str(exc)) from exc


def holdout_report(
    verdict: Mapping[str, Any], validation_csv: str | Path, *,
    dwell_windows: int = DEFAULT_HOLDOUT_DWELL_WINDOWS,
) -> dict[str, Any]:
    label = LabelDefinition.from_dict(verdict["label_def"])
    if "signal_spec" in verdict:
        spec = SignalSpec.from_dict(verdict["signal_spec"])
    else:  # verdicts written before the signal parameter
        t = verdict["tss"]
        spec = SignalSpec(
            signal="tss", label_lambda_wait=t["lambda_wait"],
            tss=TssRecompute(w_p=t["w_p"], lambda_wait=t["lambda_wait"], qmin=t["qmin"], ema_tau_ms=t["ema_tau_ms"]),
        )
    windows = spec.load(validation_csv, label, int(verdict["trim_ramp_windows"]))
    direction = verdict["fit_config"]["direction"]
    theta = float(verdict["published"]["theta_m"])
    tau_crit = float(verdict["published"]["tau_crit"])
    at_theta = threshold_balanced_accuracy(windows, theta=theta, direction=direction)
    violated = [w for w in windows if not w.slo_met]
    healthy = [w for w in windows if w.slo_met]
    z_viol = _z(violated, theta, direction)
    z_ok = _z(healthy, theta, direction)
    crit_recall = (sum(1 for z in z_viol if z < tau_crit) / len(violated)) if violated else None
    crit_false_alarm = (sum(1 for z in z_ok if z < tau_crit) / len(healthy)) if healthy else None
    merged_theta = float(verdict["merged"]["theta"])
    at_merged = threshold_balanced_accuracy(windows, theta=merged_theta, direction=direction)
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": verdict["model"],
        "signal": spec.signal,
        "direction": direction,
        "validation_csv": str(validation_csv),
        "label_def": verdict["label_def"],
        "windows": len(windows),
        "cells": len({w.scenario_id for w in windows}),
        "violating": len(violated),
        "published_theta": theta,
        "tau_crit": tau_crit,
        "at_published_theta": at_theta,
        "at_merged_theta": at_merged,
        "ranking": _ranking(windows, direction) if windows else None,
        "critical_recall_of_violating": crit_recall,
        "critical_false_alarm_on_healthy": crit_false_alarm,
        "violation_classes": violation_class_breakdown(
            windows, theta=theta, tau_crit=tau_crit, direction=direction,
        ),
        "with_dwell": dwell_acceptance(
            windows, theta=theta, tau_crit=tau_crit, direction=direction, dwell_windows=dwell_windows,
        ),
        "note": (
            "windows carry the fit's EMA (TSS recompute / signal_ema); the top-level CRITICAL "
            "numbers are per window without dwell, with_dwell applies the shared "
            "tre_common.dwell counter (plan 6.9f item B)"
        ),
    }
    opposite = (verdict.get("merged") or {}).get("opposite_direction") or {}
    if opposite.get("theta") is not None:
        report["opposite_direction"] = {
            "direction": opposite["direction"],
            "theta": opposite["theta"],
            **threshold_balanced_accuracy(windows, theta=float(opposite["theta"]), direction=opposite["direction"]),
        }
    return report


def cmd_holdout(args: argparse.Namespace) -> dict[str, Any]:
    verdict = json.loads(Path(args.verdict).read_text(encoding="utf-8"))
    return holdout_report(verdict, args.validation_csv, dwell_windows=args.dwell_windows)


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verdict")
    v.add_argument("--model", required=True)
    v.add_argument("--fitting-csv", required=True)
    v.add_argument("--family", action="append", default=[], help="NAME=CSV, repeatable")
    v.add_argument("--signal", choices=SIGNALS, default="tss")
    v.add_argument("--w-p", type=float, help="TSS prefill weight (--signal tss)")
    v.add_argument("--lambda-wait", type=float, help="TSS waiting weight (--signal tss)")
    v.add_argument(
        "--label-lambda-wait", type=float,
        help="queue weight of the delta_high surplus label; defaults to --lambda-wait. Pass "
             "the primary TSS value for every ablation arm so all arms share one label",
    )
    v.add_argument("--qmin", type=float, default=1.0)
    v.add_argument("--ema-tau-ms", type=float, default=DEFAULT_EMA_TAU_MS)
    add_label_arguments(v)
    v.add_argument("--trim-ramp-windows", type=int, default=1)
    v.add_argument("--n-resamples", type=int, default=1000)
    v.add_argument("--family-resamples", type=int, default=200)
    v.add_argument("--seed", type=int, default=20260922)
    v.add_argument("--output", required=True)
    h = sub.add_parser("holdout")
    h.add_argument("--verdict", required=True)
    h.add_argument("--validation-csv", required=True)
    h.add_argument("--dwell-windows", type=int, default=DEFAULT_HOLDOUT_DWELL_WINDOWS,
                   help="CRITICAL dwell for acceptance item B (1 = no dwell)")
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
            f"[{report['model']}/{report['signal']}] theta={report['merged']['theta']:.4g} "
            f"CI={report['merged']['bootstrap']['theta_ci95']} -> publish {p['theta_m']:.4g} "
            f"({p['source']}), delta_crit={p['delta_crit']:.3f}, delta_high={p['delta_high']:.3f}, "
            f"auroc={report['merged']['ranking']['auroc']:.3f}, stop_rule={p['stop_rule_satisfied']}"
        )
    else:
        print(f"[{report['model']}/{report['signal']}] hold-out BA={report['at_published_theta']['balanced_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fit model-specific alternative-signal thresholds and bands from R3 window CSVs.

The signal ablation compares TSS against queue length and the per-replica completed-token
rates. For that comparison to be about the signals, every arm has to go through the same
fit, so this driver has no fitting code of its own (plan §6.9 item 5): each model is
fitted by ``scripts.theta_verdict.verdict_report`` - the function the TSS verdict step
runs - parameterised by the signal. That gives every alternative signal the same
criterion and knobs (``ThetaFitConfig``), the same cell bootstrap of theta, delta_crit and
delta_high, the same stop rule, the same family rule and a verdict JSON that
``theta_verdict holdout`` scores on the held-out shape M.

What differs per signal lives in ``tre_calibration.alt_signals`` and nowhere else: the
value (``tre_common.alt_signals``, the controller's own functions, smoothed by the same
tau-EMA), the prior orientation (``lower_is_healthier``, hardcoded; the balanced accuracy
of the opposite orientation is reported next to it) and the candidate grid (distinct
values for queue_len). A signal whose AUROC is below 0.6 is flagged ``inert``.

Labels come from :mod:`tre_calibration.labels` (p95 TTFT/TPOT + unserved) with the SLOs
given on the command line; ``label_def`` is written into the report. Thresholds stay in raw
signal units; the registry-shaped block per signal carries theta, direction, delta_crit
and delta_high.
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from xml.etree import ElementTree as ET

import yaml

from tre_calibration.alt_signals import (
    alt_signal_direction,
    alt_signal_names,
    threshold_curve,
)
from tre_calibration.fit import (
    DEFAULT_HEALTHY_QUANTILE_CANDIDATES,
    DEFAULT_MIN_HEALTHY_RECALL,
    DEFAULT_THETA_CRITERION,
    THETA_CRITERIA,
    ThetaFitConfig,
)
from tre_calibration.labels import LabelDefinition
from tre_common.tss import DEFAULT_EMA_TAU_MS

from scripts.theta_verdict import build_signal_spec, verdict_report

#: Queue weight of the delta_high surplus label when the caller names none: the primary
#: TSS lambda_wait, so the alternative arms share the label of the TSS fit.
DEFAULT_LABEL_LAMBDA_WAIT = 3.0

#: Curve column plotted for each criterion, with the reference level drawn across it.
_PLOT_METRIC = {
    "balanced_accuracy": ("balanced_accuracy", 0.5),
    "reliability": ("attainment", 0.9),
}


def parse_model_input(raw: str) -> tuple[str, Path]:
    model, separator, path = raw.partition("=")
    if not separator or not model.strip() or not path.strip():
        raise argparse.ArgumentTypeError("model input must be MODEL=CSV_PATH")
    return model.strip(), Path(path.strip())


def fit_model(
    model_name: str,
    input_path: Path,
    *,
    label_def: LabelDefinition,
    signal: str,
    trim_ramp_windows: int,
    criterion: str = DEFAULT_THETA_CRITERION,
    healthy_quantile_candidates: Sequence[float] = DEFAULT_HEALTHY_QUANTILE_CANDIDATES,
    min_healthy_recall: float = DEFAULT_MIN_HEALTHY_RECALL,
    reliability_target: float = 0.9,
    min_support: int = 3,
    min_confidence: float = 0.9,
    min_scenario_families: int = 2,
    max_single_scenario_ratio: float = 0.7,
    families: dict[str, Path] | None = None,
    label_lambda_wait: float = DEFAULT_LABEL_LAMBDA_WAIT,
    ema_tau_ms: float | None = DEFAULT_EMA_TAU_MS,
    n_resamples: int = 200,
    family_resamples: int = 100,
    seed: int = 20260922,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One model's alt threshold through :func:`scripts.theta_verdict.verdict_report`."""
    spec = build_signal_spec(signal, ema_tau_ms=ema_tau_ms, label_lambda_wait=label_lambda_wait)
    config = ThetaFitConfig(
        criterion=criterion,
        direction=spec.direction,
        healthy_quantile_candidates=tuple(healthy_quantile_candidates),
        min_healthy_recall=min_healthy_recall,
        reliability_target=reliability_target,
        min_support=min_support,
        min_confidence=min_confidence,
        min_scenario_families=min_scenario_families,
        max_single_scenario_ratio=max_single_scenario_ratio,
        candidate_grid=spec.candidate_grid,
    )
    verdict = verdict_report(
        model=model_name, fitting_csv=input_path, families=dict(families or {}), spec=spec,
        label=label_def, trim_ramp_windows=trim_ramp_windows, config=config,
        n_resamples=n_resamples, family_resamples=family_resamples, seed=seed,
    )
    windows = spec.load(input_path, label_def, trim_ramp_windows)
    published = verdict["published"]
    theta = float(published["theta_m"])
    # Windows sitting exactly on the threshold: with ignore_eos every request has the
    # same token count, so rate signals live on a lattice and two models can land on the
    # same lattice point. Recorded so a coincidence can be told apart from a bug.
    on_theta = sorted({window.scenario_id for window in windows if window.signal == theta})
    merged = verdict["merged"]
    payload = {
        "input_csv": str(input_path),
        "label_def": label_def.as_dict(),
        "windows_at_theta": sum(1 for window in windows if window.signal == theta),
        "cells_at_theta": on_theta,
        "window_count": len(windows),
        "cell_count": len({window.scenario_id for window in windows}),
        "alt_thresholds": {
            signal: {
                "theta": theta,
                "direction": spec.direction,
                "delta_crit": float(published["delta_crit"]),
                "delta_high": float(published["delta_high"]),
            }
        },
        "theta_criterion": criterion,
        "fit": merged["fit"],
        "ranking": merged["ranking"],
        "opposite_direction_diagnostic": merged["opposite_direction"],
        "verdict": verdict,
    }
    return payload, threshold_curve(windows, direction=spec.direction)


def write_threshold_svg(
    path: str | Path,
    curves: dict[str, list[dict[str, Any]]],
    selected_thetas: dict[str, float],
    *,
    signal: str,
    direction: str,
    criterion: str,
) -> None:
    metric, reference = _PLOT_METRIC[criterion]
    width, height = 900, 520
    left, right, top, bottom = 72, 24, 32, 64
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_theta = max(row["theta"] for rows in curves.values() for row in rows)
    colors = ["#006d77", "#d1495b", "#6a4c93"]

    def x(value: float) -> float:
        return left + (value / max_theta) * plot_width

    def y(value: float) -> float:
        return top + (1.0 - value) * plot_height

    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": f"0 0 {width} {height}",
            "role": "img",
            "aria-label": f"{signal} threshold {metric} curves",
        },
    )
    ET.SubElement(svg, "rect", {"width": str(width), "height": str(height), "fill": "white"})
    ET.SubElement(
        svg,
        "line",
        {"x1": str(left), "y1": str(y(reference)), "x2": str(width - right), "y2": str(y(reference)), "stroke": "#777", "stroke-dasharray": "6 5"},
    )
    for tick in sorted({0.0, 0.5, reference, 1.0}):
        ET.SubElement(
            svg,
            "text",
            {"x": str(left - 10), "y": str(y(tick) + 5), "text-anchor": "end", "font-size": "13", "fill": "#222"},
        ).text = f"{tick:.1f}"
    for tick in range(5):
        value = max_theta * tick / 4
        ET.SubElement(
            svg,
            "text",
            {"x": str(x(value)), "y": str(height - bottom + 24), "text-anchor": "middle", "font-size": "13", "fill": "#222"},
        ).text = f"{value:.0f}"
    ET.SubElement(svg, "line", {"x1": str(left), "y1": str(top), "x2": str(left), "y2": str(height - bottom), "stroke": "#222"})
    ET.SubElement(svg, "line", {"x1": str(left), "y1": str(height - bottom), "x2": str(width - right), "y2": str(height - bottom), "stroke": "#222"})

    for index, (model, rows) in enumerate(sorted(curves.items())):
        color = colors[index % len(colors)]
        points = " ".join(f"{x(row['theta']):.2f},{y(row[metric]):.2f}" for row in rows)
        ET.SubElement(svg, "polyline", {"points": points, "fill": "none", "stroke": color, "stroke-width": "2"})
        selected = selected_thetas[model]
        selected_row = min(rows, key=lambda row: abs(row["theta"] - selected))
        ET.SubElement(
            svg,
            "circle",
            {"cx": f"{x(selected):.2f}", "cy": f"{y(selected_row[metric]):.2f}", "r": "5", "fill": color, "stroke": "white", "stroke-width": "1.5"},
        )
        legend_y = top + 18 * index
        ET.SubElement(svg, "line", {"x1": str(width - 260), "y1": str(legend_y), "x2": str(width - 230), "y2": str(legend_y), "stroke": color, "stroke-width": "3"})
        ET.SubElement(svg, "text", {"x": str(width - 220), "y": str(legend_y + 5), "font-size": "13", "fill": "#222"}).text = f"{model} theta={selected:.2f}"

    ET.SubElement(svg, "text", {"x": str(width / 2), "y": str(height - 12), "text-anchor": "middle", "font-size": "15", "fill": "#111"}).text = f"{signal} theta (raw per-replica units)"
    healthy_side = "<= theta" if direction == "lower_is_healthier" else ">= theta"
    ET.SubElement(svg, "text", {"x": "18", "y": str(height / 2), "text-anchor": "middle", "font-size": "15", "fill": "#111", "transform": f"rotate(-90 18 {height / 2})"}).text = (
        f"{metric} of 'value {healthy_side} => SLO met'"
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg).write(output, encoding="utf-8", xml_declaration=True)


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-input", action="append", type=parse_model_input, required=True)
    parser.add_argument("--ttft-p95-ms", type=float, required=True)
    parser.add_argument("--tpot-p95-ms", type=float, required=True)
    parser.add_argument("--signal", choices=alt_signal_names(), default="queue_len")
    parser.add_argument("--output", required=True)
    parser.add_argument("--curve-dir")
    parser.add_argument("--plot-output")
    parser.add_argument("--trim-ramp-windows", type=int, default=1)
    parser.add_argument(
        "--theta-criterion",
        choices=THETA_CRITERIA,
        default=DEFAULT_THETA_CRITERION,
        help=(
            "same knob, same default as tre_calibration.cli: balanced_accuracy maximises "
            "balanced accuracy of 'healthy side of theta => SLO met' over healthy-score "
            "quantiles; reliability is the cumulative-attainment containment rule"
        ),
    )
    parser.add_argument(
        "--min-healthy-recall",
        type=float,
        default=DEFAULT_MIN_HEALTHY_RECALL,
        help="floor on recall of healthy windows for the balanced-accuracy criterion",
    )
    parser.add_argument(
        "--healthy-quantiles",
        default="",
        help="comma-separated healthy-score quantiles to search (default 0.05..0.50 step 0.05)",
    )
    parser.add_argument("--reliability-target", type=float, default=0.9)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.9)
    parser.add_argument("--min-scenario-families", type=int, default=2)
    parser.add_argument("--max-single-scenario-ratio", type=float, default=0.7)
    parser.add_argument(
        "--family", action="append", default=[],
        help="MODEL:NAME=CSV, repeatable - per-family CSVs for the family rule and stop rule",
    )
    parser.add_argument("--label-lambda-wait", type=float, default=DEFAULT_LABEL_LAMBDA_WAIT)
    parser.add_argument("--ema-tau-ms", type=float, default=DEFAULT_EMA_TAU_MS)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--family-resamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--verdict-dir",
        help="write each model's verdict JSON here as <model>_verdict_<signal>.json "
             "(input of theta_verdict holdout)",
    )
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)

    families: dict[str, dict[str, Path]] = {}
    for item in args.family:
        head, sep, path = item.partition("=")
        model, colon, name = head.partition(":")
        if not sep or not colon or not model or not name or not path:
            parser.error("--family must be MODEL:NAME=CSV")
        families.setdefault(model, {})[name] = Path(path)

    if len({model for model, _path in args.model_input}) != len(args.model_input):
        parser.error("each model may appear only once")

    healthy_quantiles = (
        tuple(float(part) for part in args.healthy_quantiles.split(","))
        if args.healthy_quantiles
        else DEFAULT_HEALTHY_QUANTILE_CANDIDATES
    )

    label_def = LabelDefinition(args.ttft_p95_ms, args.tpot_p95_ms)
    models: dict[str, Any] = {}
    curves: dict[str, list[dict[str, Any]]] = {}
    for model, input_path in sorted(args.model_input):
        models[model], curves[model] = fit_model(
            model,
            input_path,
            label_def=label_def,
            signal=args.signal,
            trim_ramp_windows=args.trim_ramp_windows,
            criterion=args.theta_criterion,
            healthy_quantile_candidates=healthy_quantiles,
            min_healthy_recall=args.min_healthy_recall,
            reliability_target=args.reliability_target,
            min_support=args.min_support,
            min_confidence=args.min_confidence,
            min_scenario_families=args.min_scenario_families,
            max_single_scenario_ratio=args.max_single_scenario_ratio,
            families=families.get(model),
            label_lambda_wait=args.label_lambda_wait,
            ema_tau_ms=args.ema_tau_ms if args.ema_tau_ms > 0 else None,
            n_resamples=args.n_resamples,
            family_resamples=args.family_resamples,
            seed=args.seed,
        )
        if args.verdict_dir:
            vdir = Path(args.verdict_dir)
            vdir.mkdir(parents=True, exist_ok=True)
            (vdir / f"{model}_verdict_{args.signal}.json").write_text(
                json.dumps(models[model]["verdict"], indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )

    # Two models on the same threshold is not an error: token-rate signals sit on a
    # lattice under ignore_eos (e.g. 1446.4 = 339 x 128 / 30), so coinciding thresholds
    # happen. Warn and record which windows/cells carry the shared value (plan 6.9).
    by_theta: dict[float, list[str]] = {}
    for model, payload in models.items():
        by_theta.setdefault(payload["alt_thresholds"][args.signal]["theta"], []).append(model)
    warnings: list[dict[str, Any]] = []
    for theta, same in sorted(by_theta.items()):
        if len(same) < 2:
            continue
        entry = {
            "kind": "coinciding_threshold",
            "theta": theta,
            "models": sorted(same),
            "windows_at_theta": {m: models[m]["windows_at_theta"] for m in sorted(same)},
            "cells_at_theta": {m: models[m]["cells_at_theta"] for m in sorted(same)},
        }
        warnings.append(entry)
        print(f"WARNING: {args.signal} threshold {theta} is shared by {sorted(same)}", file=sys.stderr)

    report = {
        "generated_at": args.generated_at or datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "command": shlex.join([sys.argv[0], *(argv if argv is not None else sys.argv[1:])]),
        "signal": args.signal,
        "direction": alt_signal_direction(args.signal),
        "theta_criterion": args.theta_criterion,
        "trim_ramp_windows": args.trim_ramp_windows,
        "label_def": label_def.as_dict(),
        "warnings": warnings,
        "fit_config": {
            "pipeline": "scripts.theta_verdict.verdict_report",
            "label_lambda_wait": args.label_lambda_wait,
            "ema_tau_ms": args.ema_tau_ms,
            "n_resamples": args.n_resamples,
            "healthy_quantile_candidates": list(healthy_quantiles),
            "min_healthy_recall": args.min_healthy_recall,
            "reliability_target": args.reliability_target,
            "min_support": args.min_support,
            "min_confidence": args.min_confidence,
            "min_scenario_families": args.min_scenario_families,
            "max_single_scenario_ratio": args.max_single_scenario_ratio,
        },
        "models": models,
    }
    report["models"] = {
        model: {k: v for k, v in payload.items() if k != "verdict"}
        | {"published": payload["verdict"]["published"], "stop_rule": payload["verdict"]["stop_rule"],
           "family_verdict": payload["verdict"]["family_verdict"],
           "bootstrap": payload["verdict"]["merged"]["bootstrap"]}
        for model, payload in models.items()
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(json.loads(json.dumps(report, default=str)), sort_keys=False), encoding="utf-8")

    if args.curve_dir:
        curve_dir = Path(args.curve_dir)
        curve_dir.mkdir(parents=True, exist_ok=True)
        for model, rows in curves.items():
            with (curve_dir / f"{model}_{args.signal}_curve.csv").open(
                "w", encoding="utf-8", newline=""
            ) as destination:
                writer = csv.DictWriter(
                    destination, fieldnames=list(rows[0]), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)

    if args.plot_output:
        write_threshold_svg(
            args.plot_output,
            curves,
            {
                model: payload["alt_thresholds"][args.signal]["theta"]
                for model, payload in models.items()
            },
            signal=args.signal,
            direction=alt_signal_direction(args.signal),
            criterion=args.theta_criterion,
        )

    for model, payload in models.items():
        threshold = payload["alt_thresholds"][args.signal]
        print(
            f"{model}: theta={threshold['theta']:.6f} delta_crit={threshold['delta_crit']:.3f} "
            f"delta_high={threshold['delta_high']:.3f} direction={threshold['direction']} "
            f"criterion={args.theta_criterion} auroc={payload['ranking']['auroc']:.3f} "
            f"inert={payload['ranking']['inert']} windows={payload['window_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

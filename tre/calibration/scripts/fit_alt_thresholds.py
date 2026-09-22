#!/usr/bin/env python3
"""Fit model-specific alternative-signal thresholds from existing R3 window CSVs.

The signal ablation compares TSS against queue length and the per-replica completed-token
rates. For that comparison to be about the signals, every arm has to be thresholded by
the same criterion, so this driver goes through :func:`tre_calibration.fit.fit_theta` --
the same entry point ``tre_calibration.cli`` uses for TSS -- and defaults to the same
criterion (``balanced_accuracy``) and the same knobs. ``--theta-criterion`` can select
the cumulative-attainment containment rule instead, but then it applies to whichever
signal is being fitted, never to one side of a comparison only.

Orientation is the one thing the alternative signals do not share with TSS: they are
pressure signals (``lower_is_healthier``), recorded per signal in
``tre_calibration.alt_signals``. Thresholds stay in raw signal units; no reciprocal
transform is written into the registry. The first ramp window of every R3 cell is
trimmed by default, matching the experiment scorer.

Labels come from :mod:`tre_calibration.labels` - the same p95 TTFT/TPOT + unserved label
the TSS theta fit uses - with the SLOs given on the command line (``--ttft-p95-ms`` /
``--tpot-p95-ms``), never read from a registry: the registry's e2e SLO made this driver
fit a different label than the TSS fit (plan §6.3 B4). ``label_def`` is written into the
report.
"""
from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from xml.etree import ElementTree as ET

import yaml

from tre_calibration.alt_signals import (
    alt_signal_column,
    alt_signal_direction,
    alt_signal_names,
    fit_report,
    per_replica_token_rate_transform,
    threshold_curve,
)
from tre_calibration.dataset import load_windows_from_csv
from tre_calibration.fit import (
    DEFAULT_HEALTHY_QUANTILE_CANDIDATES,
    DEFAULT_MIN_HEALTHY_RECALL,
    DEFAULT_THETA_CRITERION,
    THETA_CRITERIA,
    fit_theta,
)
from tre_calibration.labels import LabelDefinition

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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    direction = alt_signal_direction(signal)
    signal_column = alt_signal_column(signal)
    windows = load_windows_from_csv(
        input_path,
        latency_slo_ms=label_def.latency_slo_ms(),
        signal_column=signal_column or signal,
        signal_transform=(
            per_replica_token_rate_transform(signal) if signal_column is None else None
        ),
        trim_ramp_windows=trim_ramp_windows,
    )
    knobs: dict[str, Any] = {
        "criterion": criterion,
        "healthy_quantile_candidates": healthy_quantile_candidates,
        "min_healthy_recall": min_healthy_recall,
        "reliability_target": reliability_target,
        "min_support": min_support,
        "min_confidence": min_confidence,
        "min_scenario_families": min_scenario_families,
        "max_single_scenario_ratio": max_single_scenario_ratio,
    }
    fit = fit_theta(windows, direction=direction, **knobs)
    # Same criterion, orientation flipped: a signal whose wrong-way fit also publishes
    # has not demonstrated a direction, and the artifact has to say so.
    opposite_direction = (
        "higher_is_healthier"
        if direction == "lower_is_healthier"
        else "lower_is_healthier"
    )
    opposite_fit = fit_theta(windows, direction=opposite_direction, **knobs)
    if not fit.publish or fit.theta is None:
        raise RuntimeError(
            f"{model_name}/{signal} did not publish under criterion={criterion}: "
            f"{fit.reject_reason}"
        )
    # Windows sitting exactly on the threshold: with ignore_eos every request has the
    # same token count, so rate signals live on a lattice and two models can land on the
    # same lattice point. Recorded so a coincidence can be told apart from a bug.
    on_theta = sorted(
        {window.scenario_id for window in windows if window.signal == fit.theta}
    )
    payload = {
        "input_csv": str(input_path),
        "label_def": label_def.as_dict(),
        "windows_at_theta": sum(1 for window in windows if window.signal == fit.theta),
        "cells_at_theta": on_theta,
        "window_count": len(windows),
        "cell_count": len({window.scenario_id for window in windows}),
        "alt_thresholds": {
            signal: {
                "theta": fit.theta,
                "direction": direction,
            }
        },
        "theta_criterion": criterion,
        "fit": fit_report(fit, windows, direction=direction),
        "opposite_direction_diagnostic": {
            "direction": opposite_direction,
            "publish": opposite_fit.publish,
            "theta": opposite_fit.theta,
            **fit_report(opposite_fit, windows, direction=opposite_direction),
        },
    }
    return payload, threshold_curve(windows, direction=direction)


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
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)

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
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")

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
            f"{model}: theta={threshold['theta']:.6f} "
            f"direction={threshold['direction']} criterion={args.theta_criterion} "
            f"windows={payload['window_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fit model-specific alternative-signal thresholds from existing R3 window CSVs.

Queue length uses the native lower-is-healthier branch in
``fit_theta_by_reliability``. The reported theta therefore stays in raw queue units; no
reciprocal transform is written into the registry. The first ramp window of every R3 cell
is trimmed by default, matching the experiment scorer.
"""
from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from xml.etree import ElementTree as ET

import yaml

from tre_calibration.dataset import CalibrationWindow, load_windows_from_csv
from tre_calibration.fit import fit_theta_by_reliability
from tre_common.registry import load_registry

_SIGNAL_CONFIG = {
    "queue_len": ("queue_control", "lower_is_healthier"),
}


def parse_model_input(raw: str) -> tuple[str, Path]:
    model, separator, path = raw.partition("=")
    if not separator or not model.strip() or not path.strip():
        raise argparse.ArgumentTypeError("model input must be MODEL=CSV_PATH")
    return model.strip(), Path(path.strip())


def reliability_curve(
    windows: Sequence[CalibrationWindow],
    *,
    direction: str,
) -> list[dict[str, Any]]:
    lower = direction == "lower_is_healthier"
    rows: list[dict[str, Any]] = []
    for theta in sorted({window.signal for window in windows}):
        subset = [
            window
            for window in windows
            if (window.signal <= theta if lower else window.signal >= theta)
        ]
        families = Counter(window.scenario_family for window in subset)
        rows.append(
            {
                "theta": theta,
                "support": len(subset),
                "attainment": (
                    sum(1 for window in subset if window.slo_met) / len(subset)
                    if subset
                    else 0.0
                ),
                "healthy": sum(1 for window in subset if window.slo_met),
                "violations": sum(1 for window in subset if not window.slo_met),
                "scenario_families": len(families),
                "max_family_ratio": (
                    max(families.values()) / len(subset) if subset else 0.0
                ),
            }
        )
    return rows


def fit_model(
    model_name: str,
    input_path: Path,
    *,
    registry_path: str,
    signal: str,
    trim_ramp_windows: int,
    reliability_target: float,
    min_support: int,
    min_confidence: float,
    min_scenario_families: int,
    max_single_scenario_ratio: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    signal_column, direction = _SIGNAL_CONFIG[signal]
    spec = load_registry(registry_path).model(model_name)
    windows = load_windows_from_csv(
        input_path,
        latency_slo_ms={
            "ttft_p95": spec.slo.ttft_p95_ms,
            "tpot_p95": spec.slo.tpot_p95_ms,
            "e2e_p95": spec.slo.e2e_p95_ms,
        },
        signal_column=signal_column,
        trim_ramp_windows=trim_ramp_windows,
    )
    fit = fit_theta_by_reliability(
        windows,
        reliability_target=reliability_target,
        min_support=min_support,
        min_confidence=min_confidence,
        min_scenario_families=min_scenario_families,
        max_single_scenario_ratio=max_single_scenario_ratio,
        direction=direction,
    )
    if not fit.publish or fit.theta is None:
        raise RuntimeError(
            f"{model_name}/{signal} did not publish: {fit.reject_reason} "
            f"(support={fit.support}, attainment={fit.attainment})"
        )
    payload = {
        "input_csv": str(input_path),
        "window_count": len(windows),
        "cell_count": len({window.scenario_id for window in windows}),
        "alt_thresholds": {
            signal: {
                "theta": fit.theta,
                "direction": direction,
            }
        },
        "fit": {
            "support": fit.support,
            "attainment": fit.attainment,
            "confidence": fit.confidence,
            "coverage_pass": fit.coverage_pass,
            "family_counts": fit.family_counts,
            "candidate_count": fit.candidate_count,
        },
    }
    return payload, reliability_curve(windows, direction=direction)


def write_reliability_svg(
    path: str | Path,
    curves: dict[str, list[dict[str, Any]]],
    selected_thetas: dict[str, float],
) -> None:
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
            "aria-label": "Queue threshold reliability curves",
        },
    )
    ET.SubElement(svg, "rect", {"width": str(width), "height": str(height), "fill": "white"})
    ET.SubElement(
        svg,
        "line",
        {"x1": str(left), "y1": str(y(0.9)), "x2": str(width - right), "y2": str(y(0.9)), "stroke": "#777", "stroke-dasharray": "6 5"},
    )
    for tick in (0.0, 0.5, 0.9, 1.0):
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
        points = " ".join(f"{x(row['theta']):.2f},{y(row['attainment']):.2f}" for row in rows)
        ET.SubElement(svg, "polyline", {"points": points, "fill": "none", "stroke": color, "stroke-width": "2"})
        selected = selected_thetas[model]
        selected_row = min(rows, key=lambda row: abs(row["theta"] - selected))
        ET.SubElement(
            svg,
            "circle",
            {"cx": f"{x(selected):.2f}", "cy": f"{y(selected_row['attainment']):.2f}", "r": "5", "fill": color, "stroke": "white", "stroke-width": "1.5"},
        )
        legend_y = top + 18 * index
        ET.SubElement(svg, "line", {"x1": str(width - 260), "y1": str(legend_y), "x2": str(width - 230), "y2": str(legend_y), "stroke": color, "stroke-width": "3"})
        ET.SubElement(svg, "text", {"x": str(width - 220), "y": str(legend_y + 5), "font-size": "13", "fill": "#222"}).text = f"{model} theta={selected:.2f}"

    ET.SubElement(svg, "text", {"x": str(width / 2), "y": str(height - 12), "text-anchor": "middle", "font-size": "15", "fill": "#111"}).text = "queue_len theta (raw queue units)"
    ET.SubElement(svg, "text", {"x": "18", "y": str(height / 2), "text-anchor": "middle", "font-size": "15", "fill": "#111", "transform": f"rotate(-90 18 {height / 2})"}).text = "healthy attainment for queue <= theta"
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
    parser.add_argument("--registry", required=True)
    parser.add_argument("--signal", choices=sorted(_SIGNAL_CONFIG), default="queue_len")
    parser.add_argument("--output", required=True)
    parser.add_argument("--curve-dir")
    parser.add_argument("--plot-output")
    parser.add_argument("--trim-ramp-windows", type=int, default=1)
    parser.add_argument("--reliability-target", type=float, default=0.9)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.9)
    parser.add_argument("--min-scenario-families", type=int, default=2)
    parser.add_argument("--max-single-scenario-ratio", type=float, default=0.7)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)

    if len({model for model, _path in args.model_input}) != len(args.model_input):
        parser.error("each model may appear only once")

    models: dict[str, Any] = {}
    curves: dict[str, list[dict[str, Any]]] = {}
    for model, input_path in sorted(args.model_input):
        models[model], curves[model] = fit_model(
            model,
            input_path,
            registry_path=args.registry,
            signal=args.signal,
            trim_ramp_windows=args.trim_ramp_windows,
            reliability_target=args.reliability_target,
            min_support=args.min_support,
            min_confidence=args.min_confidence,
            min_scenario_families=args.min_scenario_families,
            max_single_scenario_ratio=args.max_single_scenario_ratio,
        )

    theta_values = {
        payload["alt_thresholds"][args.signal]["theta"] for payload in models.values()
    }
    if len(theta_values) != len(models):
        raise RuntimeError("fitted thresholds must be model-distinct")

    report = {
        "generated_at": args.generated_at or datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "command": shlex.join([sys.argv[0], *(argv if argv is not None else sys.argv[1:])]),
        "signal": args.signal,
        "trim_ramp_windows": args.trim_ramp_windows,
        "fit_config": {
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
        write_reliability_svg(
            args.plot_output,
            curves,
            {
                model: payload["alt_thresholds"][args.signal]["theta"]
                for model, payload in models.items()
            },
        )

    for model, payload in models.items():
        threshold = payload["alt_thresholds"][args.signal]
        print(
            f"{model}: theta={threshold['theta']:.6f} "
            f"direction={threshold['direction']} windows={payload['window_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

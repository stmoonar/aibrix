#!/usr/bin/env python3
"""Cell-level bootstrap CI driver for the reliability theta_m fit (calibration QA).

Thin wrapper around :func:`tre_calibration.bootstrap.bootstrap_theta`: it loads a window CSV
exactly the way the production fit CLI (``tre_calibration.cli``) does
(``load_windows_from_csv``, same warmup/contaminated/missing-latency/zero-token filtering),
runs the production point fit for reference, then runs the cell-level bootstrap under the SAME
fit-config, and emits a JSON report plus a terse one-line summary.

The resampling unit is the distinct ``scenario_id`` (load-scan grid cell), not the individual
sliding window -- see ``bootstrap.py`` for why. Report-only: it never touches ``registry.yaml``.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from tre_calibration.bootstrap import bootstrap_theta
from tre_calibration.dataset import load_windows_from_csv
from tre_calibration.fit import fit_theta_by_reliability


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    latency_slo_ms: dict[str, float] = {
        "ttft_p95": args.ttft_p95_ms,
        "tpot_p95": args.tpot_p95_ms,
    }
    if args.e2e_p95_ms is not None:
        latency_slo_ms["e2e_p95"] = args.e2e_p95_ms

    windows = load_windows_from_csv(
        args.input,
        latency_slo_ms=latency_slo_ms,
        signal_column=args.signal_column,
        trim_ramp_windows=args.trim_ramp_windows,
    )
    n_cells = len({w.scenario_id for w in windows})

    point = fit_theta_by_reliability(
        windows,
        reliability_target=args.reliability_target,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_scenario_families=args.min_scenario_families,
        max_single_scenario_ratio=args.max_single_scenario_ratio,
    )
    result = bootstrap_theta(
        windows,
        n_resamples=args.n_resamples,
        seed=args.seed,
        reliability_target=args.reliability_target,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_scenario_families=args.min_scenario_families,
        max_single_scenario_ratio=args.max_single_scenario_ratio,
    )

    report = {
        "generated_at": args.generated_at or datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "signal_column": args.signal_column,
        "trim_ramp_windows": args.trim_ramp_windows,
        "slo": dict(latency_slo_ms),
        "fit_config": {
            "reliability_target": args.reliability_target,
            "min_support": args.min_support,
            "min_confidence": args.min_confidence,
            "min_scenario_families": args.min_scenario_families,
            "max_single_scenario_ratio": args.max_single_scenario_ratio,
        },
        "window": {"count": len(windows), "n_cells": n_cells},
        "point_fit": {
            "publish": point.publish,
            "theta": point.theta,
            "support": point.support,
            "attainment": point.attainment,
            "confidence": point.confidence,
            "coverage_pass": point.coverage_pass,
            "family_counts": point.family_counts,
            "reject_reason": point.reject_reason,
            "candidate_count": point.candidate_count,
        },
        "bootstrap": {
            "n_resamples": result.n_resamples,
            "seed": args.seed,
            "n_cells_resampled": n_cells,
            "n_published": result.n_published,
            "publish_rate": result.publish_rate,
            "theta_p2_5": result.theta_p2_5,
            "theta_p50": result.theta_p50,
            "theta_p97_5": result.theta_p97_5,
            "theta_mean": result.theta_mean,
            "theta_std": result.theta_std,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    b = report["bootstrap"]
    ci = (
        f"[{b['theta_p2_5']:.3f}, {b['theta_p97_5']:.3f}]"
        if b["n_published"]
        else "[n/a, n/a]"
    )
    median = f"{b['theta_p50']:.3f}" if b["n_published"] else "n/a"
    point_theta = f"{point.theta:.3f}" if point.theta is not None else "n/a"
    print(
        f"[{args.model_name}] windows={len(windows)} cells={n_cells} "
        f"point_theta={point_theta}(publish={point.publish}) "
        f"boot_median={median} CI95={ci} publish_rate={b['publish_rate']:.3f} "
        f"(n_resamples={b['n_resamples']} seed={args.seed})"
    )
    print(f"wrote report to {out}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cell-level bootstrap CI for the reliability theta_m fit (report only; "
            "does not touch registry.yaml)"
        )
    )
    parser.add_argument("--input", required=True, help="R3 window CSV (one model)")
    parser.add_argument("--output", required=True, help="Path to write the JSON bootstrap report")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--signal-column", default="trs")
    parser.add_argument("--trim-ramp-windows", type=int, default=1)
    parser.add_argument("--ttft-p95-ms", type=float, required=True)
    parser.add_argument("--tpot-p95-ms", type=float, required=True)
    parser.add_argument("--e2e-p95-ms", type=float)
    parser.add_argument("--reliability-target", type=float, default=0.9)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.9)
    parser.add_argument("--min-scenario-families", type=int, default=2)
    parser.add_argument("--max-single-scenario-ratio", type=float, default=0.7)
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generated-at")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

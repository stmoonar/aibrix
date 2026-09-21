#!/usr/bin/env python3
"""Cell-level bootstrap CI driver for the ``theta_m`` fit (calibration QA).

Thin wrapper around :func:`tre_calibration.bootstrap.bootstrap_theta`: it loads a window CSV
exactly the way the production fit CLI (``tre_calibration.cli``) does
(``load_windows_from_csv``, same warmup/contaminated/missing-latency/zero-token filtering),
builds one :class:`~tre_calibration.fit.ThetaFitConfig` from the command line, runs the point
fit with it, runs the cell-level bootstrap with the *same object*, and emits a JSON report plus
a terse one-line summary.

One configuration for both is the point: the interval only brackets the published ``theta_m``
if the point estimate and every resample went through the same criterion, orientation and
acceptance gates. The criterion and orientation default to ``tre_calibration.cli``'s defaults
(:data:`~tre_calibration.fit.DEFAULT_THETA_CRITERION` /
:data:`~tre_calibration.fit.DEFAULT_SIGNAL_DIRECTION`), and the report records what was used, so
a stop rule reading ``publish_rate`` or the CI half-width can check it is reading the interval
of the threshold it actually publishes.

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
from tre_calibration.fit import (
    DEFAULT_HEALTHY_QUANTILE_CANDIDATES,
    DEFAULT_MAX_SINGLE_SCENARIO_RATIO,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_HEALTHY_RECALL,
    DEFAULT_MIN_SCENARIO_FAMILIES,
    DEFAULT_MIN_SUPPORT,
    DEFAULT_RELIABILITY_TARGET,
    DEFAULT_SIGNAL_DIRECTION,
    DEFAULT_THETA_CRITERION,
    SIGNAL_DIRECTIONS,
    THETA_CRITERIA,
    ThetaFitConfig,
)
from tre_calibration.profile import theta_fit_block, theta_method_of


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

    healthy_quantiles = (
        tuple(float(part) for part in args.healthy_quantiles.split(","))
        if args.healthy_quantiles
        else DEFAULT_HEALTHY_QUANTILE_CANDIDATES
    )
    config = ThetaFitConfig(
        criterion=args.theta_criterion,
        direction=args.direction,
        healthy_quantile_candidates=healthy_quantiles,
        min_healthy_recall=args.min_healthy_recall,
        reliability_target=args.reliability_target,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_scenario_families=args.min_scenario_families,
        max_single_scenario_ratio=args.max_single_scenario_ratio,
    )

    # Same `config` object for the point estimate and for every resample: the interval is
    # only an interval for *this* theta if nothing about the fit changed between them.
    point = config.fit(windows)
    result = bootstrap_theta(
        windows,
        n_resamples=args.n_resamples,
        seed=args.seed,
        config=config,
    )

    fit_config = dict(config.as_dict())
    fit_config.update(
        {
            "latency_slo_ms": dict(sorted(latency_slo_ms.items())),
            "signal_column": args.signal_column,
            "trim_ramp_windows": args.trim_ramp_windows,
        }
    )
    report = {
        "generated_at": args.generated_at or datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "signal_column": args.signal_column,
        "trim_ramp_windows": args.trim_ramp_windows,
        "slo": dict(latency_slo_ms),
        "method": {"theta_m_method": theta_method_of(point)},
        "fit_config": dict(sorted(fit_config.items())),
        "window": {"count": len(windows), "n_cells": n_cells},
        "point_fit": {
            "publish": point.publish,
            "theta": point.theta,
            **theta_fit_block(point),
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
            # Recorded from the result, not from the argv, so the report states the
            # configuration the resamples actually ran under.
            "fit_config": dict(sorted(result.config.as_dict().items())),
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
        f"criterion={config.criterion} direction={config.direction} "
        f"point_theta={point_theta}(publish={point.publish}) "
        f"boot_median={median} CI95={ci} publish_rate={b['publish_rate']:.3f} "
        f"(n_resamples={b['n_resamples']} seed={args.seed})"
    )
    print(f"wrote report to {out}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cell-level bootstrap CI for the theta_m fit (report only; "
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
    parser.add_argument(
        "--theta-criterion",
        choices=THETA_CRITERIA,
        default=DEFAULT_THETA_CRITERION,
        help=(
            "criterion the point fit and every resample are fitted under. Defaults to the "
            "calibration CLI's default, so the interval brackets the theta that CLI publishes"
        ),
    )
    parser.add_argument(
        "--direction",
        choices=list(SIGNAL_DIRECTIONS),
        default=DEFAULT_SIGNAL_DIRECTION,
        help="orientation of --signal-column (TSS/TRS is higher_is_healthier, the default)",
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
        help="Comma-separated healthy-score quantiles to search (default 0.05..0.50 step 0.05)",
    )
    parser.add_argument("--reliability-target", type=float, default=DEFAULT_RELIABILITY_TARGET)
    parser.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument(
        "--min-scenario-families", type=int, default=DEFAULT_MIN_SCENARIO_FAMILIES
    )
    parser.add_argument(
        "--max-single-scenario-ratio", type=float, default=DEFAULT_MAX_SINGLE_SCENARIO_RATIO
    )
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generated-at")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from tre_calibration.dataset import load_windows_from_csv
from tre_calibration.evaluate import evaluate_signal_direction
from tre_calibration.fit import (
    DEFAULT_CRITICAL_VIOLATION_QUANTILE,
    DEFAULT_HEALTHY_QUANTILE_CANDIDATES,
    DEFAULT_DELTA_FLOOR_MODE,
    DEFAULT_MIN_CRITICAL_RECALL,
    DEFAULT_MIN_HEALTHY_RECALL,
    DEFAULT_MIN_SURPLUS_PRECISION,
    DEFAULT_SIGNAL_DIRECTION,
    FLOOR_MODES,
    DEFAULT_SURPLUS_LATENCY_QUANTILE,
    DEFAULT_SURPLUS_QUEUE_QUANTILE,
    DEFAULT_THETA_CRITERION,
    SIGNAL_DIRECTIONS,
    THETA_CRITERIA,
    fit_delta_margins,
    fit_theta,
)
from tre_calibration.profile import build_profile_patch
from tre_calibration.signals import ParameterCandidateScore


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    latency_slo_ms = {
        "ttft_p95": args.ttft_p95_ms,
        "tpot_p95": args.tpot_p95_ms,
    }
    if args.e2e_p95_ms is not None:
        latency_slo_ms["e2e_p95"] = args.e2e_p95_ms

    healthy_quantiles = (
        tuple(float(part) for part in args.healthy_quantiles.split(","))
        if args.healthy_quantiles
        else DEFAULT_HEALTHY_QUANTILE_CANDIDATES
    )

    windows = load_windows_from_csv(
        args.input,
        latency_slo_ms=latency_slo_ms,
        signal_column=args.signal_column,
        trim_ramp_windows=args.trim_ramp_windows,
        lambda_wait=args.lambda_wait,
    )
    theta_fit = fit_theta(
        windows,
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

    delta_fit = None
    if args.fit_delta and theta_fit.theta is not None and theta_fit.theta > 0.0:
        delta_fit = fit_delta_margins(
            windows,
            theta=theta_fit.theta,
            critical_violation_quantile=args.critical_violation_quantile,
            surplus_latency_quantile=args.surplus_latency_quantile,
            surplus_queue_quantile=args.surplus_queue_quantile,
            min_critical_recall=args.min_critical_recall,
            min_surplus_precision=args.min_surplus_precision,
            floor_mode=args.delta_floor_mode,
        )

    direction = evaluate_signal_direction(windows)
    parameter_score = ParameterCandidateScore(
        w_p=args.w_p,
        lambda_wait=args.lambda_wait,
        qmin=args.qmin,
        objective=(direction.spearman_health + 1.0) / 2.0,
        spearman_health=direction.spearman_health,
        auroc=direction.auroc,
        scored_windows=windows,
    )
    fit_config = {
        "critical_violation_quantile": args.critical_violation_quantile,
        "direction": args.direction,
        "fit_delta": bool(args.fit_delta),
        "healthy_quantile_candidates": list(healthy_quantiles),
        "latency_slo_ms": dict(sorted(latency_slo_ms.items())),
        "max_single_scenario_ratio": args.max_single_scenario_ratio,
        "min_confidence": args.min_confidence,
        "delta_floor_mode": args.delta_floor_mode,
        "min_critical_recall": args.min_critical_recall,
        "min_healthy_recall": args.min_healthy_recall,
        "min_scenario_families": args.min_scenario_families,
        "min_support": args.min_support,
        "min_surplus_precision": args.min_surplus_precision,
        "reliability_target": args.reliability_target,
        "signal_column": args.signal_column,
        "surplus_latency_quantile": args.surplus_latency_quantile,
        "surplus_queue_quantile": args.surplus_queue_quantile,
        "theta_criterion": args.theta_criterion,
        "trim_ramp_windows": args.trim_ramp_windows,
    }
    inputs = {
        "csv_path": str(args.input),
        "csv_sha256": _sha256(args.input),
        "window_count": len(windows),
        "scenario_count": len({window.scenario_id for window in windows}),
    }
    patch = build_profile_patch(
        args.model_name,
        theta_fit=theta_fit,
        parameter_score=parameter_score,
        generated_at=args.generated_at or datetime.now(timezone.utc).isoformat(),
        fit_config=fit_config,
        delta_fit=delta_fit,
        inputs=inputs,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(patch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit TRE calibration parameters from a window CSV")
    parser.add_argument("--input", required=True, help="CSV with per-window metrics and a TRS/signal column")
    parser.add_argument("--output", required=True, help="Path to write the calibration profile patch JSON")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--signal-column", default="trs")
    parser.add_argument(
        "--trim-ramp-windows",
        type=int,
        default=1,
        help=(
            "Earliest windows dropped per scenario. This moves theta by a few percent, so "
            "the value used is always recorded in the artifact's fit_config."
        ),
    )
    parser.add_argument("--ttft-p95-ms", type=float, required=True)
    parser.add_argument("--tpot-p95-ms", type=float, required=True)
    parser.add_argument("--e2e-p95-ms", type=float)
    parser.add_argument(
        "--theta-criterion",
        choices=THETA_CRITERIA,
        default=DEFAULT_THETA_CRITERION,
        help=(
            "balanced_accuracy: theta maximises balanced accuracy of 'signal >= theta => SLO met' "
            "over healthy-score quantiles (default). reliability: the cumulative-attainment "
            "containment rule, kept as a comparison baseline."
        ),
    )
    parser.add_argument(
        "--direction",
        choices=list(SIGNAL_DIRECTIONS),
        default=DEFAULT_SIGNAL_DIRECTION,
        help=(
            "orientation of --signal-column. TSS/TRS is higher_is_healthier (the "
            "default); the pressure signals it is compared against in the ablation "
            "(queue length, per-replica token rates) are lower_is_healthier"
        ),
    )
    parser.add_argument(
        "--min-healthy-recall",
        type=float,
        default=DEFAULT_MIN_HEALTHY_RECALL,
        help=(
            "Floor on recall of healthy windows for the balanced-accuracy criterion. Off (0.0) "
            "by default: a 0.90 floor vetoes the balanced-accuracy optimum and re-introduces the "
            "low-theta bias."
        ),
    )
    parser.add_argument(
        "--healthy-quantiles",
        default="",
        help="Comma-separated healthy-score quantiles to search (default 0.05..0.50 step 0.05)",
    )
    parser.add_argument("--reliability-target", type=float, default=0.9)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.9)
    parser.add_argument("--min-scenario-families", type=int, default=2)
    parser.add_argument("--max-single-scenario-ratio", type=float, default=0.7)
    parser.add_argument(
        "--no-fit-delta",
        dest="fit_delta",
        action="store_false",
        help="Skip the delta_crit/delta_high margin fit (fitted by default)",
    )
    parser.set_defaults(fit_delta=True)
    parser.add_argument("--critical-violation-quantile", type=float, default=DEFAULT_CRITICAL_VIOLATION_QUANTILE)
    parser.add_argument("--surplus-latency-quantile", type=float, default=DEFAULT_SURPLUS_LATENCY_QUANTILE)
    parser.add_argument("--surplus-queue-quantile", type=float, default=DEFAULT_SURPLUS_QUEUE_QUANTILE)
    parser.add_argument("--min-critical-recall", type=float, default=DEFAULT_MIN_CRITICAL_RECALL)
    parser.add_argument("--min-surplus-precision", type=float, default=DEFAULT_MIN_SURPLUS_PRECISION)
    parser.add_argument(
        "--delta-floor-mode",
        choices=list(FLOOR_MODES),
        default=DEFAULT_DELTA_FLOOR_MODE,
        help=(
            "how the acceptance floors rank candidates: soft (default) selects on "
            "balanced accuracy and uses the floor only as a tie-break; strict keeps "
            "the older behaviour where a candidate missing the floor always loses"
        ),
    )
    parser.add_argument("--w-p", type=float, default=0.04)
    parser.add_argument("--lambda-wait", type=float, default=2.625)
    parser.add_argument("--qmin", type=float, default=1.0)
    parser.add_argument("--generated-at")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

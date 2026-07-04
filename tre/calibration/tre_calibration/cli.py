from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from tre_calibration.dataset import load_windows_from_csv
from tre_calibration.evaluate import evaluate_signal_direction
from tre_calibration.fit import fit_theta_by_reliability
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

    windows = load_windows_from_csv(args.input, latency_slo_ms=latency_slo_ms, signal_column=args.signal_column)
    theta_fit = fit_theta_by_reliability(
        windows,
        reliability_target=args.reliability_target,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_scenario_families=args.min_scenario_families,
        max_single_scenario_ratio=args.max_single_scenario_ratio,
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
    patch = build_profile_patch(
        args.model_name,
        theta_fit=theta_fit,
        parameter_score=parameter_score,
        generated_at=args.generated_at or datetime.now(timezone.utc).isoformat(),
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(patch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit TRE calibration parameters from a window CSV")
    parser.add_argument("--input", required=True, help="CSV with per-window metrics and a TRS/signal column")
    parser.add_argument("--output", required=True, help="Path to write the calibration profile patch JSON")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--signal-column", default="trs")
    parser.add_argument("--ttft-p95-ms", type=float, required=True)
    parser.add_argument("--tpot-p95-ms", type=float, required=True)
    parser.add_argument("--e2e-p95-ms", type=float)
    parser.add_argument("--reliability-target", type=float, default=0.9)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.9)
    parser.add_argument("--min-scenario-families", type=int, default=2)
    parser.add_argument("--max-single-scenario-ratio", type=float, default=0.7)
    parser.add_argument("--w-p", type=float, default=0.04)
    parser.add_argument("--lambda-wait", type=float, default=2.625)
    parser.add_argument("--qmin", type=float, default=1.0)
    parser.add_argument("--generated-at")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Ranking-separation train/test evaluation driver (doc15 s1.4 acceptance, experiment 2).

The R3 refit acceptance (``docs/refactor/15_signal_and_window_plan.md`` s1.4) asks the
``theta_m`` fit report to hit AUROC / coverage targets on the frozen control window W. To
show that separation *generalises* rather than overfits, this driver:

  1. loads an R3 window CSV (``tre_calibration.dataset.load_windows_from_csv``);
  2. splits **whole scenarios** into train/test with a deterministic scenario-id hash
     holdout (``select_test_scenarios`` -> ``split_by_scenario``), so no grid cell leaks
     across the two sets;
  3. fits ``theta_m`` on the **train** set only, using the same reliability/coverage gate
     as the production fit (``fit_theta_by_reliability``);
  4. reports how the raw signal direction and the train-fitted threshold hold up on the
     **held-out test** scenarios (AUROC, Spearman health, threshold confusion counts).

Output is an artifact-only JSON report: per-set metrics plus a reproducible split manifest
(seed, fraction, exact train/test scenario lists). Like ``tre_calibration.cli`` it does NOT
mutate ``registry.yaml`` -- applying/rejecting the calibration remains an operator step.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from tre_calibration.dataset import (
    CalibrationWindow,
    load_windows_from_csv,
    select_test_scenarios,
    split_by_scenario,
)
from tre_calibration.evaluate import evaluate_signal_direction, evaluate_threshold
from tre_calibration.fit import fit_theta_by_reliability


def _scenarios(windows: Sequence[CalibrationWindow]) -> list[str]:
    return sorted({window.scenario_id for window in windows})


def _direction_payload(windows: Sequence[CalibrationWindow]) -> dict[str, Any]:
    direction = evaluate_signal_direction(windows)
    healthy = sum(1 for window in windows if window.slo_met)
    return {
        "window_count": len(windows),
        "healthy_windows": healthy,
        "violation_windows": len(windows) - healthy,
        "auroc": direction.auroc,
        "spearman_health": direction.spearman_health,
    }


def _threshold_payload(windows: Sequence[CalibrationWindow], *, theta: float) -> dict[str, Any]:
    metrics = evaluate_threshold(windows, theta=theta)
    return {
        "theta": theta,
        "auroc": metrics.auroc,
        "spearman_health": metrics.spearman_health,
        "balanced_accuracy": metrics.balanced_accuracy,
        "true_healthy": metrics.true_healthy,
        "false_healthy": metrics.false_healthy,
        "true_violation": metrics.true_violation,
        "false_violation": metrics.false_violation,
    }


def run_ranking_separation(
    windows: Sequence[CalibrationWindow],
    *,
    model_name: str,
    test_fraction: float = 0.2,
    split_seed: str = "tre-v2-ranking",
    reliability_target: float = 0.9,
    min_support: int = 3,
    min_confidence: float = 0.9,
    min_scenario_families: int = 2,
    max_single_scenario_ratio: float = 0.7,
    signal_column: str = "trs",
    trim_ramp_windows: int = 0,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Fit theta on a deterministic train split, report generalisation on the test split."""
    windows = list(windows)
    test_scenarios = select_test_scenarios(
        windows, test_fraction=test_fraction, seed=split_seed
    )
    train, test = split_by_scenario(windows, test_scenarios=test_scenarios)

    theta_fit = fit_theta_by_reliability(
        train,
        reliability_target=reliability_target,
        min_support=min_support,
        min_confidence=min_confidence,
        min_scenario_families=min_scenario_families,
        max_single_scenario_ratio=max_single_scenario_ratio,
    )

    train_report = {
        "fit": {
            "publish": theta_fit.publish,
            "theta_m": theta_fit.theta,
            "support": theta_fit.support,
            "attainment": theta_fit.attainment,
            "confidence": theta_fit.confidence,
            "coverage_pass": theta_fit.coverage_pass,
            "family_counts": dict(sorted(theta_fit.family_counts.items())),
            "reject_reason": theta_fit.reject_reason,
            "candidate_count": theta_fit.candidate_count,
        },
        "direction": _direction_payload(train),
    }

    test_report: dict[str, Any] = {"direction": _direction_payload(test)}
    if theta_fit.theta is not None and test:
        test_report["threshold"] = _threshold_payload(test, theta=theta_fit.theta)
    else:
        test_report["threshold"] = None

    train_ids = _scenarios(train)
    test_ids = _scenarios(test)
    # Convenience gate: train publishes a theta AND that theta, applied to unseen test
    # scenarios, never labels a real SLO violation as healthy (the operationally unsafe
    # error). The full metrics below let a human apply a stricter bar.
    generalizes = bool(
        theta_fit.publish
        and test
        and test_report["threshold"] is not None
        and test_report["threshold"]["false_healthy"] == 0
    )

    return {
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "signal_column": signal_column,
        "split": {
            "method": "scenario_hash_holdout",
            "seed": split_seed,
            "trim_ramp_windows": trim_ramp_windows,
            "test_fraction": test_fraction,
            "total_windows": len(windows),
            "total_scenarios": len(_scenarios(windows)),
            "train_scenarios": train_ids,
            "test_scenarios": test_ids,
            "train_window_count": len(train),
            "test_window_count": len(test),
            "leakage_free": not (set(train_ids) & set(test_ids)),
        },
        "train": train_report,
        "test": test_report,
        "generalizes": generalizes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    latency_slo_ms = {
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
    report = run_ranking_separation(
        windows,
        model_name=args.model_name,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        reliability_target=args.reliability_target,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_scenario_families=args.min_scenario_families,
        max_single_scenario_ratio=args.max_single_scenario_ratio,
        signal_column=args.signal_column,
        trim_ramp_windows=args.trim_ramp_windows,
        generated_at=args.generated_at,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    split = report["split"]
    train = report["train"]
    test = report["test"]
    print(
        f"[{report['model_name']}] scenarios={split['total_scenarios']} "
        f"train={split['train_window_count']}w/{len(split['train_scenarios'])}s "
        f"test={split['test_window_count']}w/{len(split['test_scenarios'])}s "
        f"| train theta_m={train['fit']['theta_m']} publish={train['fit']['publish']} "
        f"reject={train['fit']['reject_reason']} train_auroc={train['direction']['auroc']:.3f}"
    )
    if test["threshold"] is not None:
        print(
            f"    test_auroc={test['direction']['auroc']:.3f} "
            f"test_spearman={test['direction']['spearman_health']:.3f} "
            f"test_threshold_bal_acc={test['threshold']['balanced_accuracy']:.3f} "
            f"false_healthy={test['threshold']['false_healthy']} "
            f"generalizes={report['generalizes']}"
        )
    else:
        print(
            f"    test set has no train-fitted theta to apply "
            f"(test_windows={test['direction']['window_count']}, "
            f"train_published={train['fit']['publish']})"
        )
    print(f"wrote report to {out}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TRE signal ranking separation with a scenario-level train/test split"
    )
    parser.add_argument("--input", required=True, help="R3 window CSV with a TRS/signal column")
    parser.add_argument("--output", required=True, help="Path to write the JSON separation report")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--signal-column", default="trs")
    parser.add_argument("--trim-ramp-windows", type=int, default=1)
    parser.add_argument("--ttft-p95-ms", type=float, required=True)
    parser.add_argument("--tpot-p95-ms", type=float, required=True)
    parser.add_argument("--e2e-p95-ms", type=float)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", default="tre-v2-ranking")
    parser.add_argument("--reliability-target", type=float, default=0.9)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.9)
    parser.add_argument("--min-scenario-families", type=int, default=2)
    parser.add_argument("--max-single-scenario-ratio", type=float, default=0.7)
    parser.add_argument("--generated-at")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

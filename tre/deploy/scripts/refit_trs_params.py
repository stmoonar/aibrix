#!/usr/bin/env python3
"""S2 parameter-refit control driver (doc15 s2, "标定范围补齐").

``docs/refactor/15_signal_and_window_plan.md`` s2 asks that, alongside the R3
``theta_m`` fit, we re-run ``tre_calibration.signals.grid_search_parameters`` over the
``w_p / lambda_wait / qmin`` grid on the *same* R3 window CSV and produce a "refit value
vs inherited value" comparison, so the paper cannot be accused of simply reusing the old
0.4.0 system's hand-tuned parameters.

This is a thin wrapper: it only assembles data and emits a report. It does NOT touch
``registry.yaml`` -- the keep/adopt decision remains an operator/architect step (like
``tre_calibration.cli`` and ``eval_ranking_separation.py``).

What it does, per model (grid search is per-model, so run one model at a time):

  1. load the R3 window CSV into aligned ``(CalibrationWindow, SignalInputs)`` sequences
     using the *same* row filtering as ``tre_calibration.dataset.load_windows_from_csv``
     (warmup/contaminated/missing-latency/zero-token rows dropped, ``slo_met`` and
     ``health_score`` computed identically), so the refit sees exactly the windows the
     ``theta_m`` fit saw;
  2. score the *inherited* triple (from ``registry.yaml`` for that model, or CLI
     overrides) with ``score_parameter_candidate``;
  3. run ``grid_search_parameters`` over the grid and take ``best``;
  4. compare best vs inherited on AUROC and Spearman-health and emit a
     ``recommendation`` per doc15 s2 step 3's 5% rule.

The 5% rule (doc15 s2 step 3): keep the inherited triple iff the best candidate improves
BOTH auroc AND spearman by <= ``threshold_pct`` (default 5%) relative to the inherited
score; if EITHER metric improves by more than the threshold, recommend adopting the
refit. Improvements are signed (``best - inherited``); because ``grid_search_parameters``
maximises the spearman-based objective, a metric the optimum happens to *lower* does not
by itself force adoption. When an inherited score is ~0 the relative percent is undefined
(reported as ``null``) and any real improvement counts as exceeding the threshold.

Grid (doc15 s2 step 1, verbatim):
  w_p_candidates      = [0.02, 0.04, 0.06, 0.08, 0.10]
  lambda_wait_candidates = [1.5, 1.875, 2.25, 2.625, 3.0]
  qmin_candidates     = [1.0]
These bracket every model's inherited value (7b/8b w_p=0.08, lambda_wait=1.875;
14b w_p=0.0575 sits between 0.04 and 0.06, lambda_wait=3.0) plus an order-of-magnitude
sweep of w_p, and are overridable via CLI for ad-hoc sweeps.

The ``signals.py`` scoring口径 is deliberate (no_floor / no EMA / no ``w_d``; it mirrors
the old system's parameter-search objective -- see ``06_calibration_design.md``). This
driver must NOT change that formula; it only assembles inputs and reports.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from tre_calibration.dataset import (
    CalibrationWindow,
    _as_float,
    _resolve_latency_columns,
    _skip_row,
)
from tre_calibration.signals import (
    ParameterCandidateScore,
    SignalInputs,
    _candidate_key,
    grid_search_parameters,
    score_parameter_candidate,
)

# doc15 s2 step 1, verbatim.
DEFAULT_W_P_CANDIDATES: tuple[float, ...] = (0.02, 0.04, 0.06, 0.08, 0.10)
DEFAULT_LAMBDA_WAIT_CANDIDATES: tuple[float, ...] = (1.5, 1.875, 2.25, 2.625, 3.0)
DEFAULT_QMIN_CANDIDATES: tuple[float, ...] = (1.0,)
DEFAULT_THRESHOLD_PCT: float = 5.0

_EPS = 1e-12


def load_windows_and_inputs(
    path: str | Path,
    *,
    latency_slo_ms: dict[str, float],
    signal_column: str = "trs",
) -> tuple[list[CalibrationWindow], list[SignalInputs]]:
    """Load aligned ``(windows, inputs)`` from an R3 window CSV.

    Mirrors ``tre_calibration.dataset.load_windows_from_csv`` row-for-row (same filters,
    same ``slo_met``/``health_score`` construction) but ALSO emits the ``SignalInputs``
    that ``grid_search_parameters`` needs, so the two lists are index-aligned by
    construction. ``grid_search_parameters`` recomputes the signal from ``inputs`` and
    ignores ``window.signal``; the ``signal_column`` is still required so the refit
    operates on exactly the same window set as the ``theta_m`` fit.
    """
    active_columns = _resolve_latency_columns(latency_slo_ms)
    if not active_columns:
        raise ValueError("latency_slo_ms must contain at least one active SLO")

    windows: list[CalibrationWindow] = []
    inputs: list[SignalInputs] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if _skip_row(row):
                continue
            signal = _as_float(row.get(signal_column))
            if signal is None:
                continue
            prompt_tokens = _as_float(row.get("prompt_tokens_total"), 0.0) or 0.0
            generation_tokens = _as_float(row.get("generation_tokens_total"), 0.0) or 0.0
            if prompt_tokens + generation_tokens <= 0.0:
                continue

            ratios: list[float] = []
            missing_latency = False
            for slo_key, column in active_columns.items():
                value = _as_float(row.get(column))
                if value is None:
                    missing_latency = True
                    break
                ratios.append(value / float(latency_slo_ms[slo_key]))
            if missing_latency or not ratios:
                continue

            p95_ratio_max = max(ratios)
            windows.append(
                CalibrationWindow(
                    scenario_id=(row.get("scenario_id") or "unknown").strip() or "unknown",
                    scenario_family=(row.get("scenario_family") or "unknown").strip() or "unknown",
                    signal=signal,
                    slo_met=all(ratio <= 1.0 for ratio in ratios),
                    health_score=1.0 / (1.0 + p95_ratio_max),
                )
            )
            inputs.append(
                SignalInputs(
                    prompt_tokens_total=prompt_tokens,
                    generation_tokens_total=generation_tokens,
                    avg_waiting=_as_float(row.get("avg_waiting"), 0.0) or 0.0,
                    avg_running=_as_float(row.get("avg_running"), 0.0) or 0.0,
                    avg_swapping=_as_float(row.get("avg_swapping"), 0.0) or 0.0,
                    assigned_replicas=_as_float(row.get("assigned_replicas"), 1.0) or 1.0,
                    routable_pods=_as_float(row.get("routable_pods"), 1.0) or 1.0,
                    kv_cache_hit_rate=_as_float(row.get("kv_cache_hit_rate"), 0.0) or 0.0,
                )
            )
    return windows, inputs


def resolve_inherited(
    model_name: str,
    *,
    w_p: float | None,
    lambda_wait: float | None,
    qmin: float | None,
    registry_path: str | None = None,
) -> tuple[tuple[float, float, float], str]:
    """Resolve the inherited ``(w_p, lambda_wait, qmin)`` triple.

    Any value left as ``None`` is filled from ``registry.yaml`` for ``model_name``
    (needs ``tre_common`` on ``PYTHONPATH``). Returns the triple and a source tag
    (``"cli"`` when every value was supplied on the CLI, else ``"registry"``).
    """
    if w_p is not None and lambda_wait is not None and qmin is not None:
        return (w_p, lambda_wait, qmin), "cli"

    from tre_common.registry import load_registry

    trs = load_registry(registry_path).model(model_name).trs
    resolved = (
        trs.w_p if w_p is None else w_p,
        trs.lambda_wait if lambda_wait is None else lambda_wait,
        trs.qmin if qmin is None else qmin,
    )
    return resolved, "registry"


def _candidate_payload(candidate: ParameterCandidateScore) -> dict[str, Any]:
    return {
        "w_p": candidate.w_p,
        "lambda_wait": candidate.lambda_wait,
        "qmin": candidate.qmin,
        "objective": candidate.objective,
        "auroc": candidate.auroc,
        "spearman_health": candidate.spearman_health,
    }


def _relative_pct(best_value: float, inherited_value: float) -> float | None:
    if abs(inherited_value) < _EPS:
        return None
    return (best_value - inherited_value) / abs(inherited_value) * 100.0


def _metric_exceeds(delta: float, delta_pct: float | None, threshold_pct: float) -> bool:
    # inherited ~0 -> percent is undefined; any real positive improvement exceeds.
    if delta_pct is None:
        return delta > _EPS
    return delta_pct > threshold_pct


def build_report(
    windows: Sequence[CalibrationWindow],
    inputs: Sequence[SignalInputs],
    *,
    model_name: str,
    inherited_w_p: float,
    inherited_lambda_wait: float,
    inherited_qmin: float,
    inherited_source: str = "cli",
    w_p_candidates: Sequence[float] = DEFAULT_W_P_CANDIDATES,
    lambda_wait_candidates: Sequence[float] = DEFAULT_LAMBDA_WAIT_CANDIDATES,
    qmin_candidates: Sequence[float] = DEFAULT_QMIN_CANDIDATES,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    signal_column: str = "trs",
    slo: dict[str, float] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Score inherited vs grid-best and assemble the comparison report."""
    if not windows:
        raise ValueError("no windows survived filtering; cannot refit")
    if len(windows) != len(inputs):
        raise ValueError("windows and inputs must have the same length")

    inherited = score_parameter_candidate(
        windows,
        inputs,
        w_p=inherited_w_p,
        lambda_wait=inherited_lambda_wait,
        qmin=inherited_qmin,
    )
    search = grid_search_parameters(
        windows,
        inputs,
        w_p_candidates=list(w_p_candidates),
        lambda_wait_candidates=list(lambda_wait_candidates),
        qmin_candidates=list(qmin_candidates),
    )
    best = search.best

    top5 = sorted(search.candidates, key=_candidate_key, reverse=True)[:5]

    auroc_delta = best.auroc - inherited.auroc
    spearman_delta = best.spearman_health - inherited.spearman_health
    auroc_delta_pct = _relative_pct(best.auroc, inherited.auroc)
    spearman_delta_pct = _relative_pct(best.spearman_health, inherited.spearman_health)
    auroc_exceeds = _metric_exceeds(auroc_delta, auroc_delta_pct, threshold_pct)
    spearman_exceeds = _metric_exceeds(spearman_delta, spearman_delta_pct, threshold_pct)
    recommendation = "adopt_refit" if (auroc_exceeds or spearman_exceeds) else "keep_inherited"

    healthy = sum(1 for window in windows if window.slo_met)
    return {
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "signal_column": signal_column,
        "slo": dict(slo) if slo else None,
        "window": {
            "count": len(windows),
            "healthy": healthy,
            "violation": len(windows) - healthy,
        },
        "grid": {
            "w_p_candidates": list(w_p_candidates),
            "lambda_wait_candidates": list(lambda_wait_candidates),
            "qmin_candidates": list(qmin_candidates),
            "candidate_count": len(search.candidates),
        },
        "inherited": {"source": inherited_source, **_candidate_payload(inherited)},
        "best": _candidate_payload(best),
        "top5": [_candidate_payload(candidate) for candidate in top5],
        "comparison": {
            "threshold_pct": threshold_pct,
            "auroc_delta": auroc_delta,
            "auroc_delta_pct": auroc_delta_pct,
            "auroc_exceeds": auroc_exceeds,
            "spearman_delta": spearman_delta,
            "spearman_delta_pct": spearman_delta_pct,
            "spearman_exceeds": spearman_exceeds,
        },
        "recommendation": recommendation,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    latency_slo_ms: dict[str, float] = {
        "ttft_p95": args.ttft_p95_ms,
        "tpot_p95": args.tpot_p95_ms,
    }
    if args.e2e_p95_ms is not None:
        latency_slo_ms["e2e_p95"] = args.e2e_p95_ms

    windows, inputs = load_windows_and_inputs(
        args.input, latency_slo_ms=latency_slo_ms, signal_column=args.signal_column
    )

    (inherited_w_p, inherited_lambda_wait, inherited_qmin), inherited_source = resolve_inherited(
        args.model_name,
        w_p=args.inherited_w_p,
        lambda_wait=args.inherited_lambda_wait,
        qmin=args.inherited_qmin,
        registry_path=args.registry,
    )

    report = build_report(
        windows,
        inputs,
        model_name=args.model_name,
        inherited_w_p=inherited_w_p,
        inherited_lambda_wait=inherited_lambda_wait,
        inherited_qmin=inherited_qmin,
        inherited_source=inherited_source,
        w_p_candidates=args.w_p_candidates,
        lambda_wait_candidates=args.lambda_wait_candidates,
        qmin_candidates=args.qmin_candidates,
        threshold_pct=args.threshold_pct,
        signal_column=args.signal_column,
        slo=dict(latency_slo_ms),
        generated_at=args.generated_at,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    inh = report["inherited"]
    best = report["best"]
    cmp = report["comparison"]
    print(
        f"[{report['model_name']}] windows={report['window']['count']} "
        f"(healthy={report['window']['healthy']}/violation={report['window']['violation']}) "
        f"inherited({inherited_source}) w_p={inh['w_p']} lambda_wait={inh['lambda_wait']} "
        f"qmin={inh['qmin']} objective={inh['objective']:.4f} auroc={inh['auroc']:.4f} "
        f"spearman={inh['spearman_health']:.4f}"
    )
    print(
        f"    best w_p={best['w_p']} lambda_wait={best['lambda_wait']} qmin={best['qmin']} "
        f"objective={best['objective']:.4f} auroc={best['auroc']:.4f} "
        f"spearman={best['spearman_health']:.4f}"
    )
    print(
        f"    delta auroc={cmp['auroc_delta']:+.4f} ({cmp['auroc_delta_pct']}) "
        f"spearman={cmp['spearman_delta']:+.4f} ({cmp['spearman_delta_pct']}) "
        f"threshold={cmp['threshold_pct']}% -> {report['recommendation']}"
    )
    print(f"wrote report to {out}")
    return 0


def _float_list(raw: str) -> list[float]:
    return [float(part) for part in raw.split(",") if part.strip()]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "S2 refit control: grid-search w_p/lambda_wait/qmin on an R3 window CSV and "
            "compare best vs inherited (report only; does not touch registry.yaml)"
        )
    )
    parser.add_argument("--input", required=True, help="R3 window CSV (one model)")
    parser.add_argument("--output", required=True, help="Path to write the JSON comparison report")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--signal-column", default="trs")
    parser.add_argument("--ttft-p95-ms", type=float, required=True)
    parser.add_argument("--tpot-p95-ms", type=float, required=True)
    parser.add_argument("--e2e-p95-ms", type=float)
    parser.add_argument(
        "--inherited-w-p", type=float, help="Inherited w_p (default: registry value for the model)"
    )
    parser.add_argument(
        "--inherited-lambda-wait",
        type=float,
        help="Inherited lambda_wait (default: registry value for the model)",
    )
    parser.add_argument(
        "--inherited-qmin", type=float, help="Inherited qmin (default: registry value for the model)"
    )
    parser.add_argument(
        "--registry", help="registry.yaml path for inherited defaults (default: repo registry)"
    )
    parser.add_argument(
        "--w-p-candidates",
        type=_float_list,
        default=list(DEFAULT_W_P_CANDIDATES),
        dest="w_p_candidates",
        help="Comma-separated w_p grid (default doc15 s2: 0.02,0.04,0.06,0.08,0.10)",
    )
    parser.add_argument(
        "--lambda-wait-candidates",
        type=_float_list,
        default=list(DEFAULT_LAMBDA_WAIT_CANDIDATES),
        dest="lambda_wait_candidates",
        help="Comma-separated lambda_wait grid (default doc15 s2: 1.5,1.875,2.25,2.625,3.0)",
    )
    parser.add_argument(
        "--qmin-candidates",
        type=_float_list,
        default=list(DEFAULT_QMIN_CANDIDATES),
        dest="qmin_candidates",
        help="Comma-separated qmin grid (default doc15 s2: 1.0)",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=DEFAULT_THRESHOLD_PCT,
        help="Keep/adopt relative-improvement threshold in percent (default 5)",
    )
    parser.add_argument("--generated-at")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tre_calibration.dataset import CalibrationWindow
from tre_calibration.evaluate import evaluate_signal_direction


@dataclass(frozen=True)
class SignalInputs:
    prompt_tokens_total: float
    generation_tokens_total: float
    avg_waiting: float
    avg_running: float
    avg_swapping: float
    assigned_replicas: float = 1.0
    routable_pods: float = 1.0
    kv_cache_hit_rate: float = 0.0


@dataclass(frozen=True)
class TrsBreakdown:
    total_tokens: float
    queue_raw: float
    queue_floor: float
    trs_floor: float
    trs_no_floor: float


@dataclass(frozen=True)
class ParameterCandidateScore:
    w_p: float
    lambda_wait: float
    qmin: float
    objective: float
    spearman_health: float
    auroc: float
    scored_windows: list[CalibrationWindow]


@dataclass(frozen=True)
class ParameterSearchResult:
    best: ParameterCandidateScore
    candidates: list[ParameterCandidateScore]


def compute_trs(inputs: SignalInputs, *, w_p: float, lambda_wait: float, qmin: float) -> TrsBreakdown:
    total_tokens = inputs.prompt_tokens_total * (1.0 - inputs.kv_cache_hit_rate) * w_p + inputs.generation_tokens_total
    queue_raw = lambda_wait * inputs.avg_waiting + inputs.avg_running + inputs.avg_swapping
    queue_floor = max(queue_raw, qmin)
    replica_factor = inputs.assigned_replicas / max(1.0, inputs.routable_pods)

    if total_tokens <= 0.0:
        trs_floor = 0.0
        trs_no_floor = 0.0
    else:
        trs_floor = (total_tokens / queue_floor) * replica_factor
        trs_no_floor = (total_tokens / queue_raw) * replica_factor if queue_raw > 0.0 else float("inf")

    return TrsBreakdown(
        total_tokens=total_tokens,
        queue_raw=queue_raw,
        queue_floor=queue_floor,
        trs_floor=trs_floor,
        trs_no_floor=trs_no_floor,
    )


def score_parameter_candidate(
    windows: Sequence[CalibrationWindow],
    inputs: Sequence[SignalInputs],
    *,
    w_p: float,
    lambda_wait: float,
    qmin: float,
) -> ParameterCandidateScore:
    if len(windows) != len(inputs):
        raise ValueError("windows and inputs must have the same length")

    scored_windows: list[CalibrationWindow] = []
    for window, signal_inputs in zip(windows, inputs):
        breakdown = compute_trs(signal_inputs, w_p=w_p, lambda_wait=lambda_wait, qmin=qmin)
        scored_windows.append(
            CalibrationWindow(
                scenario_id=window.scenario_id,
                scenario_family=window.scenario_family,
                signal=breakdown.trs_no_floor,
                slo_met=window.slo_met,
                health_score=window.health_score,
            )
        )

    metrics = evaluate_signal_direction(scored_windows)
    objective = (metrics.spearman_health + 1.0) / 2.0
    return ParameterCandidateScore(
        w_p=w_p,
        lambda_wait=lambda_wait,
        qmin=qmin,
        objective=objective,
        spearman_health=metrics.spearman_health,
        auroc=metrics.auroc,
        scored_windows=scored_windows,
    )


def grid_search_parameters(
    windows: Sequence[CalibrationWindow],
    inputs: Sequence[SignalInputs],
    *,
    w_p_candidates: Sequence[float],
    lambda_wait_candidates: Sequence[float],
    qmin_candidates: Sequence[float],
) -> ParameterSearchResult:
    candidates: list[ParameterCandidateScore] = []
    best: ParameterCandidateScore | None = None
    for w_p in w_p_candidates:
        for lambda_wait in lambda_wait_candidates:
            for qmin in qmin_candidates:
                candidate = score_parameter_candidate(
                    windows,
                    inputs,
                    w_p=w_p,
                    lambda_wait=lambda_wait,
                    qmin=qmin,
                )
                candidates.append(candidate)
                if best is None or _candidate_key(candidate) > _candidate_key(best):
                    best = candidate

    if best is None:
        raise ValueError("parameter search requires at least one candidate")
    return ParameterSearchResult(best=best, candidates=candidates)


def _candidate_key(candidate: ParameterCandidateScore) -> tuple[float, float, float, float, float, float]:
    return (
        candidate.objective,
        candidate.auroc,
        candidate.spearman_health,
        -candidate.w_p,
        -candidate.lambda_wait,
        -candidate.qmin,
    )

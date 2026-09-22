"""Offline TSS scoring over a (w_p, lambda_wait, qmin) grid.

The formula is NOT implemented here: every value comes from :func:`tre_common.tss.tss_terms`
and :class:`tre_common.tss.TssEma`, the same functions the online controller uses, so an
offline candidate and the live signal at the same parameters are the same number
(plan 2026-09-21 §6.4). The former "no floor / no EMA / window-total" scoring口径 of the
0.4.0 parameter search is gone with the unified definition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from tre_calibration.dataset import CalibrationWindow
from tre_calibration.evaluate import evaluate_signal_direction
from tre_common.tss import TssEma, replica_factor, tss_terms, window_is_idle


@dataclass(frozen=True)
class SignalInputs:
    """One window's raw observables.

    ``window_ms`` turns token totals into rates and is required by :func:`compute_trs`.
    ``window_end_ms`` + ``cell_id`` place the window on its cell's EMA timeline; they are
    needed only when scoring with ``ema_tau_ms``. ``preceding`` holds the cell's windows
    that sit between the previous scored window and this one but were filtered out of the
    fit (missing latency, trimmed ramp, ...): the online EMA saw them, so the offline EMA
    must advance over them too.
    """

    prompt_tokens_total: float
    generation_tokens_total: float
    avg_waiting: float
    avg_running: float
    avg_swapping: float
    assigned_replicas: float = 1.0
    routable_pods: float = 1.0
    kv_cache_hit_rate: float = 0.0
    window_ms: Optional[float] = None
    window_end_ms: Optional[float] = None
    cell_id: Optional[str] = None
    preceding: tuple["SignalInputs", ...] = ()


@dataclass(frozen=True)
class TrsBreakdown:
    #: ``w_p * r_p + r_d`` (tokens/s).
    numerator_rate: float
    queue_raw: float
    queue_floor: float
    #: The unified raw TSS (qmin-guarded, replica factor applied); NaN when undefined (idle).
    trs_floor: float
    #: Diagnostic only: the same without the qmin guard.
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
    """Raw TSS of one window through the shared definition (``tre_common.tss``)."""
    if inputs.window_ms is None:
        raise ValueError("SignalInputs.window_ms is required: TSS is a rate (tokens / window duration)")
    terms = tss_terms(
        prompt_tokens=inputs.prompt_tokens_total,
        generation_tokens=inputs.generation_tokens_total,
        window_ms=inputs.window_ms,
        avg_running=inputs.avg_running,
        avg_waiting=inputs.avg_waiting,
        w_p=w_p,
        lambda_wait=lambda_wait,
        qmin=qmin,
        kv_cache_hit_rate=inputs.kv_cache_hit_rate,
        avg_swapping=inputs.avg_swapping,
        factor=replica_factor(inputs.assigned_replicas, inputs.routable_pods),
    )
    factor = replica_factor(inputs.assigned_replicas, inputs.routable_pods)
    if terms.numerator_rate <= 0.0 or terms.raw is None:
        no_floor = 0.0 if terms.numerator_rate <= 0.0 else float("nan")
    else:
        no_floor = (terms.numerator_rate / terms.queue) * factor if terms.queue > 0.0 else float("inf")
    return TrsBreakdown(
        numerator_rate=terms.numerator_rate,
        queue_raw=terms.queue,
        queue_floor=terms.queue_ctl,
        trs_floor=terms.raw if terms.raw is not None else float("nan"),
        trs_no_floor=no_floor,
    )


def tss_series(
    inputs: Sequence[SignalInputs],
    *,
    w_p: float,
    lambda_wait: float,
    qmin: float,
    ema_tau_ms: Optional[float] = None,
) -> list[Optional[float]]:
    """The signal a fit scores, one value per input (``None`` = undefined / idle).

    Without ``ema_tau_ms`` this is the raw TSS. With it, each ``cell_id``'s windows are run
    through :class:`~tre_common.tss.TssEma` in input order, each window's ``preceding``
    windows first - exactly the sequence the online per-model EMA consumes.
    """

    def raw_of(item: SignalInputs) -> Optional[float]:
        value = compute_trs(item, w_p=w_p, lambda_wait=lambda_wait, qmin=qmin).trs_floor
        return None if math.isnan(value) else value

    if ema_tau_ms is None:
        return [raw_of(item) for item in inputs]
    emas: dict[Optional[str], TssEma] = {}
    out: list[Optional[float]] = []
    for item in inputs:
        ema = emas.get(item.cell_id)
        if ema is None:
            ema = emas[item.cell_id] = TssEma(ema_tau_ms)
        for earlier in item.preceding:
            if earlier.window_end_ms is None:
                raise ValueError("EMA scoring needs window_end_ms on every window")
            ema.update(
                raw_of(earlier), earlier.window_end_ms, earlier.window_ms,
                window_is_idle(earlier.prompt_tokens_total, earlier.generation_tokens_total),
            )
        if item.window_end_ms is None:
            raise ValueError("EMA scoring needs window_end_ms on every window")
        out.append(ema.update(
            raw_of(item), item.window_end_ms, item.window_ms,
            window_is_idle(item.prompt_tokens_total, item.generation_tokens_total),
        ))
    return out


def score_parameter_candidate(
    windows: Sequence[CalibrationWindow],
    inputs: Sequence[SignalInputs],
    *,
    w_p: float,
    lambda_wait: float,
    qmin: float,
    ema_tau_ms: Optional[float] = None,
) -> ParameterCandidateScore:
    if len(windows) != len(inputs):
        raise ValueError("windows and inputs must have the same length")

    values = tss_series(inputs, w_p=w_p, lambda_wait=lambda_wait, qmin=qmin, ema_tau_ms=ema_tau_ms)
    scored_windows: list[CalibrationWindow] = []
    for window, value in zip(windows, values):
        if value is None:
            continue  # undefined (idle) windows carry no TSS - nothing to rank
        scored_windows.append(
            CalibrationWindow(
                scenario_id=window.scenario_id,
                scenario_family=window.scenario_family,
                signal=value,
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
    ema_tau_ms: Optional[float] = None,
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
                    ema_tau_ms=ema_tau_ms,
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

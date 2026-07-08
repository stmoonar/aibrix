from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from tre_calibration.capacity import CapacitySurface
from tre_replayer.engine.schedule import RpsSegment


@dataclass(frozen=True)
class DemandPhase:
    name: str
    start_s: float
    end_s: float
    rho_by_model: Mapping[str, float]
    period_s: float | None = None
    # Per-phase, per-model (input_tokens, output_tokens) override. When a model is
    # absent here design_trace_segments falls back to the trace-wide token_shapes.
    # This is what lets a single model change its i/o shape across phases (axis A3).
    token_shapes: Mapping[str, tuple[int, int]] | None = None
    # Intentional sub-minimum phases (e.g. the A4 narrow spike shorter than the EMA
    # time constant) opt out of the min-duration guard; the resonance guard still runs.
    allow_short: bool = False


def validate_phase_plan(
    phases: Sequence[DemandPhase],
    *,
    slow_loop_s: float = 10.0,
    control_periods_s: Sequence[float] = (5.0, 10.0, 20.0),
) -> None:
    min_phase_s = 5.0 * slow_loop_s
    for phase in phases:
        duration_s = phase.end_s - phase.start_s
        if duration_s < min_phase_s and not phase.allow_short:
            raise ValueError(f"phase too short: {phase.name} duration={duration_s:g}s min={min_phase_s:g}s")
        if phase.period_s is not None and _is_resonant_period(phase.period_s, control_periods_s):
            raise ValueError(f"resonant period: {phase.name} period={phase.period_s:g}s")


def design_trace_segments(
    phases: Sequence[DemandPhase],
    capacity: CapacitySurface,
    *,
    token_shapes: Mapping[str, tuple[int, int]],
) -> list[RpsSegment]:
    validate_phase_plan(phases)
    segments: list[RpsSegment] = []
    for phase in phases:
        overrides = phase.token_shapes or {}
        for model, rho in phase.rho_by_model.items():
            if rho <= 0.0:
                continue
            input_tokens, output_tokens = overrides.get(model, token_shapes[model])
            point = capacity.capacity_at(model, input_tokens=input_tokens, output_tokens=output_tokens)
            segments.append(
                RpsSegment(
                    model=model,
                    start_s=phase.start_s,
                    end_s=phase.end_s,
                    rps=rho * point.rps,
                    input_tokens=input_tokens,
                    max_output_tokens=output_tokens,
                )
            )
    return segments


def _is_resonant_period(period_s: float, control_periods_s: Sequence[float]) -> bool:
    for control_period_s in control_periods_s:
        ratio = period_s / control_period_s
        if abs(ratio - round(ratio)) < 1e-9:
            return True
    return False

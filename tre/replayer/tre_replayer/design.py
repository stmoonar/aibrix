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


def validate_phase_plan(
    phases: Sequence[DemandPhase],
    *,
    slow_loop_s: float = 10.0,
    control_periods_s: Sequence[float] = (5.0, 10.0, 20.0),
) -> None:
    min_phase_s = 5.0 * slow_loop_s
    for phase in phases:
        duration_s = phase.end_s - phase.start_s
        if duration_s < min_phase_s:
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
        for model, rho in phase.rho_by_model.items():
            if rho <= 0.0:
                continue
            input_tokens, output_tokens = token_shapes[model]
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

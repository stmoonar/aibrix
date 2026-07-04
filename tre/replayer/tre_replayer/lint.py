from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from tre_calibration.capacity import CapacitySurface
from tre_replayer.engine.schedule import RpsSegment


_HEADROOM_TARGETS = {
    "loose": 0.60,
    "medium": 0.75,
    "tight": 0.90,
}


@dataclass(frozen=True)
class TraceLintReport:
    passed: bool
    failed_constraints: list[str]
    max_headroom: float
    static_violation_duration_s: float
    low_confidence_capacity: bool


def lint_trace(
    segments: Sequence[RpsSegment],
    capacity: CapacitySurface,
    *,
    model_slot_widths: Mapping[str, float],
    total_slots: float,
    slow_loop_s: float = 10.0,
    headroom_tier: str = "medium",
) -> TraceLintReport:
    intervals = _constant_intervals(segments)
    max_occupancy = 0.0
    static_violation_duration_s = 0.0
    low_confidence = False

    for start_s, end_s in intervals:
        active = [segment for segment in segments if segment.start_s <= start_s and segment.end_s >= end_s]
        occupancy = 0.0
        any_static_violation = False
        for segment in active:
            point = capacity.capacity_at(
                segment.model,
                input_tokens=segment.input_tokens or 0,
                output_tokens=segment.max_output_tokens or 0,
            )
            low_confidence = low_confidence or point.low_confidence
            rho = segment.rps / point.rps if point.rps > 0.0 else float("inf")
            occupancy += rho * float(model_slot_widths[segment.model])
            any_static_violation = any_static_violation or rho > 1.2
        max_occupancy = max(max_occupancy, occupancy)
        if any_static_violation:
            static_violation_duration_s += end_s - start_s

    max_headroom = max_occupancy / total_slots if total_slots > 0.0 else float("inf")
    failed: list[str] = []
    if max_headroom > 0.95:
        failed.append("C1")
    if static_violation_duration_s < 3.0 * slow_loop_s:
        failed.append("C2")
    target = _HEADROOM_TARGETS[headroom_tier]
    if abs(max_headroom - target) > 0.05:
        failed.append("C3")

    return TraceLintReport(
        passed=not failed,
        failed_constraints=failed,
        max_headroom=round(max_headroom, 6),
        static_violation_duration_s=static_violation_duration_s,
        low_confidence_capacity=low_confidence,
    )


def _constant_intervals(segments: Sequence[RpsSegment]) -> list[tuple[float, float]]:
    boundaries = sorted({time for segment in segments for time in (segment.start_s, segment.end_s)})
    return [(start, end) for start, end in zip(boundaries, boundaries[1:]) if end > start]

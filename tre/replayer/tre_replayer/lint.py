from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from tre_calibration.capacity import CapacitySurface
from tre_replayer.engine.schedule import RpsSegment
from tre_replayer.oracle import compute_oracle_lower_bound


# Headroom tiers are expressed as *integer GPU occupancy* / total_slots (traceset-v2):
#   loose  = 4/8 (resting serving floor: one replica per model, 14b is tp2)
#   medium = 6/8 (25% GPU headroom)
#   tight  = 8/8 (hugging the physical cluster limit, still integer-feasible)
# See traceset-v2 README for why v1's fractional-rho occupancy was physically infeasible.
_HEADROOM_TARGETS = {
    "loose": 0.50,
    "medium": 0.75,
    "tight": 1.00,
}

# rho -> replica count: real deployments run an integer number of pods and every model
# that receives traffic needs at least one awake replica (serving floor). A small epsilon
# absorbs float division noise so a design rho that is exactly an integer (e.g. rps = 2*C)
# does not round up to an extra replica.
_CEIL_EPS = 1e-9


def replicas_for_rho(rho: float) -> int:
    """Integer replicas needed to serve load `rho` (in single-pod-capacity units)."""
    if rho <= 0.0:
        return 0
    return max(1, math.ceil(rho - _CEIL_EPS))


@dataclass(frozen=True)
class TraceLintReport:
    passed: bool
    failed_constraints: list[str]
    max_headroom: float  # peak INTEGER GPU occupancy / total_slots (the feasibility metric)
    max_fractional_headroom: float  # peak sum(rho*width) / total_slots (aggregate lower bound)
    static_violation_duration_s: float
    oracle_violation_fraction: float
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
    max_integer_occupancy = 0.0
    max_fractional_occupancy = 0.0
    static_violation_duration_s = 0.0
    low_confidence = False

    for start_s, end_s in intervals:
        active = [segment for segment in segments if segment.start_s <= start_s and segment.end_s >= end_s]
        integer_occupancy = 0.0
        fractional_occupancy = 0.0
        any_static_violation = False
        for segment in active:
            point = capacity.capacity_at(
                segment.model,
                input_tokens=segment.input_tokens or 0,
                output_tokens=segment.max_output_tokens or 0,
            )
            low_confidence = low_confidence or point.low_confidence
            rho = segment.rps / point.rps if point.rps > 0.0 else float("inf")
            width = float(model_slot_widths[segment.model])
            fractional_occupancy += rho * width
            integer_occupancy += replicas_for_rho(rho) * width
            any_static_violation = any_static_violation or rho > 1.2
        max_integer_occupancy = max(max_integer_occupancy, integer_occupancy)
        max_fractional_occupancy = max(max_fractional_occupancy, fractional_occupancy)
        if any_static_violation:
            static_violation_duration_s += end_s - start_s

    max_headroom = max_integer_occupancy / total_slots if total_slots > 0.0 else float("inf")
    max_fractional_headroom = max_fractional_occupancy / total_slots if total_slots > 0.0 else float("inf")
    oracle = compute_oracle_lower_bound(
        segments,
        capacity,
        model_slot_widths=model_slot_widths,
        total_slots=total_slots,
    )

    failed: list[str] = []
    # C1 feasibility: the integer GPU requirement must fit the cluster (<= total_slots, i.e.
    # headroom <= 1.0 exactly is feasible). The fractional oracle stays a secondary guard.
    if max_headroom > 1.0 + _CEIL_EPS or oracle.violation_fraction >= 0.01:
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
        max_fractional_headroom=round(max_fractional_headroom, 6),
        static_violation_duration_s=static_violation_duration_s,
        oracle_violation_fraction=oracle.violation_fraction,
        low_confidence_capacity=low_confidence,
    )


def _constant_intervals(segments: Sequence[RpsSegment]) -> list[tuple[float, float]]:
    boundaries = sorted({time for segment in segments for time in (segment.start_s, segment.end_s)})
    return [(start, end) for start, end in zip(boundaries, boundaries[1:]) if end > start]

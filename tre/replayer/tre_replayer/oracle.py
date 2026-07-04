from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from tre_calibration.capacity import CapacitySurface
from tre_replayer.engine.schedule import RpsSegment


@dataclass(frozen=True)
class OracleLowerBoundReport:
    total_duration_s: float
    violation_duration_s: float
    violation_fraction: float
    max_required_slots: float


def compute_oracle_lower_bound(
    segments: Sequence[RpsSegment],
    capacity: CapacitySurface,
    *,
    model_slot_widths: Mapping[str, float],
    total_slots: float,
) -> OracleLowerBoundReport:
    intervals = _constant_intervals(segments)
    total_duration_s = 0.0
    violation_duration_s = 0.0
    max_required_slots = 0.0

    for start_s, end_s in intervals:
        duration_s = end_s - start_s
        total_duration_s += duration_s
        required_slots = 0.0
        for segment in segments:
            if segment.start_s <= start_s and segment.end_s >= end_s:
                point = capacity.capacity_at(
                    segment.model,
                    input_tokens=segment.input_tokens or 0,
                    output_tokens=segment.max_output_tokens or 0,
                )
                rho = segment.rps / point.rps if point.rps > 0.0 else float("inf")
                required_slots += rho * float(model_slot_widths[segment.model])
        max_required_slots = max(max_required_slots, required_slots)
        if required_slots > total_slots:
            violation_duration_s += duration_s

    violation_fraction = violation_duration_s / total_duration_s if total_duration_s > 0.0 else 0.0
    return OracleLowerBoundReport(
        total_duration_s=total_duration_s,
        violation_duration_s=violation_duration_s,
        violation_fraction=violation_fraction,
        max_required_slots=round(max_required_slots, 6),
    )


def _constant_intervals(segments: Sequence[RpsSegment]) -> list[tuple[float, float]]:
    boundaries = sorted({time for segment in segments for time in (segment.start_s, segment.end_s)})
    return [(start, end) for start, end in zip(boundaries, boundaries[1:]) if end > start]

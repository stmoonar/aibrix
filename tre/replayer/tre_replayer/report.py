from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from tre_calibration.capacity import CapacitySample, CapacitySurface, fit_capacity_surface
from tre_replayer.engine.schedule import RpsSegment
from tre_replayer.lint import lint_trace
from tre_replayer.traces.loader import TraceCase


def build_placeholder_capacity_surface(segments: Sequence[RpsSegment]) -> CapacitySurface:
    best: dict[tuple[str, int, int], float] = {}
    for segment in segments:
        key = (segment.model, segment.input_tokens or 0, segment.max_output_tokens or 0)
        best[key] = max(best.get(key, 0.0), segment.rps)
    samples = [
        CapacitySample(model, input_tokens, output_tokens, rps, True)
        for (model, input_tokens, output_tokens), rps in best.items()
    ]
    return fit_capacity_surface(samples)


def lint_trace_case(
    case: TraceCase,
    capacity: CapacitySurface,
    *,
    model_slot_widths: Mapping[str, float],
    total_slots: float,
    capacity_source: str,
    headroom_tier: str = "medium",
) -> dict[str, Any]:
    report = lint_trace(
        case.segments,
        capacity,
        model_slot_widths=model_slot_widths,
        total_slots=total_slots,
        headroom_tier=headroom_tier,
    )
    return {
        "trace": case.name,
        "path": str(case.path),
        "indexed": case.indexed,
        "passed": report.passed,
        "failed_constraints": report.failed_constraints,
        "max_headroom": report.max_headroom,
        "static_violation_duration_s": report.static_violation_duration_s,
        "oracle_violation_fraction": report.oracle_violation_fraction,
        "capacity_low_confidence": report.low_confidence_capacity or capacity_source.startswith("placeholder"),
        "capacity_source": capacity_source,
    }


def write_trace_report(path: str | Path, summaries: Sequence[dict[str, Any]]) -> None:
    import json

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(list(summaries), indent=2, sort_keys=True) + "\n", encoding="utf-8")

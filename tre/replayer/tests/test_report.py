from __future__ import annotations

from tre_replayer.engine.schedule import RpsSegment
from tre_replayer.report import build_placeholder_capacity_surface, lint_trace_case
from tre_replayer.traces.loader import TraceCase


def test_build_placeholder_capacity_surface_uses_max_trace_rps_per_shape() -> None:
    segments = [
        RpsSegment("m1", 0.0, 10.0, 4.0, input_tokens=100, max_output_tokens=50),
        RpsSegment("m1", 10.0, 20.0, 7.0, input_tokens=100, max_output_tokens=50),
        RpsSegment("m1", 20.0, 30.0, 3.0, input_tokens=200, max_output_tokens=50),
    ]

    capacity = build_placeholder_capacity_surface(segments)

    assert capacity.capacity_at("m1", input_tokens=100, output_tokens=50).rps == 7.0
    assert capacity.capacity_at("m1", input_tokens=200, output_tokens=50).rps == 3.0


def test_lint_trace_case_returns_json_ready_summary_with_capacity_source() -> None:
    segments = [RpsSegment("m1", 0.0, 60.0, 4.0, input_tokens=100, max_output_tokens=50)]
    case = TraceCase("trace_a", path="trace.json", indexed=True, segments=segments)
    capacity = build_placeholder_capacity_surface(segments)

    summary = lint_trace_case(
        case,
        capacity,
        model_slot_widths={"m1": 1.0},
        total_slots=4.0,
        capacity_source="placeholder_from_trace_max_rps",
    )

    assert summary["trace"] == "trace_a"
    assert summary["indexed"] is True
    assert summary["capacity_source"] == "placeholder_from_trace_max_rps"
    assert summary["capacity_low_confidence"] is True
    assert summary["oracle_violation_fraction"] == 0.0
    assert isinstance(summary["failed_constraints"], list)

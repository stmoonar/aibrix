from __future__ import annotations

from tre_calibration.capacity import CapacitySample, fit_capacity_surface
from tre_replayer.engine.schedule import RpsSegment
from tre_replayer.lint import lint_trace


def test_lint_trace_rejects_overcapacity_trace_with_c1() -> None:
    capacity = fit_capacity_surface([
        CapacitySample("m1", 100, 50, 10.0, True),
        CapacitySample("m2", 100, 50, 10.0, True),
    ])
    segments = [
        RpsSegment("m1", 0.0, 60.0, 30.0, input_tokens=100, max_output_tokens=50),
        RpsSegment("m2", 0.0, 60.0, 30.0, input_tokens=100, max_output_tokens=50),
    ]

    report = lint_trace(segments, capacity, model_slot_widths={"m1": 1.0, "m2": 1.0}, total_slots=4.0)

    assert report.passed is False
    assert "C1" in report.failed_constraints
    assert report.max_headroom == 1.5


def test_lint_trace_rejects_trace_that_never_triggers_scaling_with_c2() -> None:
    capacity = fit_capacity_surface([CapacitySample("m1", 100, 50, 10.0, True)])
    segments = [RpsSegment("m1", 0.0, 60.0, 8.0, input_tokens=100, max_output_tokens=50)]

    report = lint_trace(segments, capacity, model_slot_widths={"m1": 1.0}, total_slots=4.0, slow_loop_s=10.0)

    assert report.passed is False
    assert "C2" in report.failed_constraints
    assert report.static_violation_duration_s == 0.0

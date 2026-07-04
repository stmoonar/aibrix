from __future__ import annotations

import pytest

from tre_calibration.capacity import CapacitySample, fit_capacity_surface
from tre_replayer.design import DemandPhase, design_trace_segments, validate_phase_plan


def test_validate_phase_plan_rejects_short_phase_and_resonant_period() -> None:
    with pytest.raises(ValueError, match="phase too short"):
        validate_phase_plan([DemandPhase("warmup", 0.0, 40.0, {"m1": 0.5})], slow_loop_s=10.0)

    with pytest.raises(ValueError, match="resonant period"):
        validate_phase_plan([DemandPhase("cycle", 0.0, 60.0, {"m1": 0.5}, period_s=20.0)], slow_loop_s=10.0)


def test_design_trace_segments_maps_rho_to_rps_with_capacity_surface() -> None:
    capacity = fit_capacity_surface([
        CapacitySample("m1", 100, 50, 20.0, True),
        CapacitySample("m2", 200, 100, 5.0, True),
    ])
    phases = [
        DemandPhase("warmup", 0.0, 50.0, {"m1": 0.5, "m2": 1.2}),
        DemandPhase("burst", 50.0, 120.0, {"m1": 2.0}),
    ]

    segments = design_trace_segments(
        phases,
        capacity,
        token_shapes={"m1": (100, 50), "m2": (200, 100)},
    )

    assert [(segment.model, segment.start_s, segment.end_s, segment.rps, segment.input_tokens, segment.max_output_tokens) for segment in segments] == [
        ("m1", 0.0, 50.0, 10.0, 100, 50),
        ("m2", 0.0, 50.0, 6.0, 200, 100),
        ("m1", 50.0, 120.0, 40.0, 100, 50),
    ]

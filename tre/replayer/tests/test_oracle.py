from __future__ import annotations

from tre_calibration.capacity import CapacitySample, fit_capacity_surface
from tre_replayer.engine.schedule import RpsSegment
from tre_replayer.oracle import compute_oracle_lower_bound


def test_oracle_lower_bound_counts_only_unavoidable_overcapacity_intervals() -> None:
    capacity = fit_capacity_surface([
        CapacitySample("m1", 100, 50, 10.0, True),
        CapacitySample("m2", 100, 50, 10.0, True),
    ])
    segments = [
        RpsSegment("m1", 0.0, 20.0, 8.0, input_tokens=100, max_output_tokens=50),
        RpsSegment("m2", 0.0, 10.0, 8.0, input_tokens=100, max_output_tokens=50),
    ]

    report = compute_oracle_lower_bound(
        segments,
        capacity,
        model_slot_widths={"m1": 1.0, "m2": 1.0},
        total_slots=1.0,
    )

    assert report.total_duration_s == 20.0
    assert report.violation_duration_s == 10.0
    assert report.violation_fraction == 0.5
    assert report.max_required_slots == 1.6

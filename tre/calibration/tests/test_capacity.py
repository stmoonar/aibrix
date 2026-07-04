from __future__ import annotations

from tre_calibration.capacity import CapacitySample, fit_capacity_surface


def test_fit_capacity_surface_uses_max_slo_safe_rps_per_grid_point() -> None:
    surface = fit_capacity_surface([
        CapacitySample("dsqwen-7b", input_tokens=100, output_tokens=50, rps=4.0, slo_met=True),
        CapacitySample("dsqwen-7b", input_tokens=100, output_tokens=50, rps=8.0, slo_met=True),
        CapacitySample("dsqwen-7b", input_tokens=100, output_tokens=50, rps=10.0, slo_met=False),
        CapacitySample("dsqwen-7b", input_tokens=200, output_tokens=50, rps=5.0, slo_met=True),
    ])

    point = surface.capacity_at("dsqwen-7b", input_tokens=100, output_tokens=50)

    assert point.rps == 8.0
    assert point.low_confidence is False
    assert point.reason == "exact"


def test_capacity_surface_marks_out_of_grid_lookup_low_confidence() -> None:
    surface = fit_capacity_surface([
        CapacitySample("dsqwen-7b", input_tokens=100, output_tokens=50, rps=8.0, slo_met=True),
        CapacitySample("dsqwen-7b", input_tokens=200, output_tokens=50, rps=5.0, slo_met=True),
    ])

    point = surface.capacity_at("dsqwen-7b", input_tokens=300, output_tokens=50)

    assert point.rps == 5.0
    assert point.low_confidence is True
    assert point.reason == "nearest_extrapolated"

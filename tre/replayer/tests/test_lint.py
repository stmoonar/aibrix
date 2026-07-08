from __future__ import annotations

from tre_calibration.capacity import CapacitySample, fit_capacity_surface
from tre_replayer.engine.schedule import RpsSegment
from tre_replayer.lint import lint_trace, replicas_for_rho


def test_replicas_for_rho_rounds_up_with_serving_floor() -> None:
    assert replicas_for_rho(0.0) == 0  # no traffic -> no replica
    assert replicas_for_rho(0.4) == 1  # any traffic needs at least one awake replica
    assert replicas_for_rho(1.0) == 1  # exactly one pod of load (no float overshoot)
    assert replicas_for_rho(2.0) == 2
    assert replicas_for_rho(2.8) == 3
    assert replicas_for_rho(6.0) == 6


def test_lint_trace_rejects_integer_infeasible_but_fractionally_ok_trace_with_c1() -> None:
    """traceset-v1 A1 bug: fractional occupancy 7.2/8 <= 0.95 but integer requirement 9 > 8."""
    capacity = fit_capacity_surface([
        CapacitySample("m7b", 512, 512, 10.0, True),
        CapacitySample("m8b", 512, 512, 10.0, True),
        CapacitySample("m14b", 512, 512, 10.0, True),
    ])
    segments = [
        RpsSegment("m7b", 0.0, 300.0, 60.0, input_tokens=512, max_output_tokens=512),   # rho 6.0
        RpsSegment("m8b", 0.0, 300.0, 4.0, input_tokens=512, max_output_tokens=512),     # rho 0.4
        RpsSegment("m14b", 0.0, 300.0, 4.0, input_tokens=512, max_output_tokens=512),    # rho 0.4, tp2
    ]
    widths = {"m7b": 1.0, "m8b": 1.0, "m14b": 2.0}

    report = lint_trace(segments, capacity, model_slot_widths=widths, total_slots=8.0, headroom_tier="tight")

    # Fractional occupancy fits (6.0 + 0.4 + 0.8 = 7.2 -> 0.9) ...
    assert report.max_fractional_headroom == 0.9
    # ... but the integer requirement (6 + 1 + 2 = 9 GPU) does not, so C1 fails.
    assert report.max_headroom == round(9.0 / 8.0, 6)
    assert "C1" in report.failed_constraints
    assert report.passed is False


def test_lint_trace_passes_c1_for_full_but_integer_feasible_tight_trace() -> None:
    capacity = fit_capacity_surface([
        CapacitySample("m7b", 512, 512, 10.0, True),
        CapacitySample("m8b", 512, 512, 10.0, True),
        CapacitySample("m14b", 512, 512, 10.0, True),
    ])
    segments = [
        RpsSegment("m7b", 0.0, 300.0, 48.0, input_tokens=512, max_output_tokens=512),   # rho 4.8 -> 5 pods
        RpsSegment("m8b", 0.0, 300.0, 4.0, input_tokens=512, max_output_tokens=512),     # rho 0.4 -> 1 pod
        RpsSegment("m14b", 0.0, 300.0, 4.0, input_tokens=512, max_output_tokens=512),    # rho 0.4 -> 1 pod x2
    ]
    widths = {"m7b": 1.0, "m8b": 1.0, "m14b": 2.0}

    report = lint_trace(segments, capacity, model_slot_widths=widths, total_slots=8.0, headroom_tier="tight")

    assert report.max_headroom == 1.0  # 5 + 1 + 2 = 8 GPU exactly (tight)
    assert "C1" not in report.failed_constraints
    assert "C3" not in report.failed_constraints
    assert report.passed is True


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
    assert report.oracle_violation_fraction == 1.0


def test_lint_trace_rejects_trace_that_never_triggers_scaling_with_c2() -> None:
    capacity = fit_capacity_surface([CapacitySample("m1", 100, 50, 10.0, True)])
    segments = [RpsSegment("m1", 0.0, 60.0, 8.0, input_tokens=100, max_output_tokens=50)]

    report = lint_trace(segments, capacity, model_slot_widths={"m1": 1.0}, total_slots=4.0, slow_loop_s=10.0)

    assert report.passed is False
    assert "C2" in report.failed_constraints
    assert report.static_violation_duration_s == 0.0


def test_lint_trace_reports_oracle_violation_fraction_for_short_instant_spike() -> None:
    capacity = fit_capacity_surface([CapacitySample("m1", 100, 50, 10.0, True)])
    segments = [
        RpsSegment("m1", 0.0, 1.0, 11.0, input_tokens=100, max_output_tokens=50),
        RpsSegment("m1", 1.0, 200.0, 7.5, input_tokens=100, max_output_tokens=50),
    ]

    report = lint_trace(segments, capacity, model_slot_widths={"m1": 1.0}, total_slots=1.0, headroom_tier="medium")

    assert report.oracle_violation_fraction == 0.005
    assert "C1" in report.failed_constraints

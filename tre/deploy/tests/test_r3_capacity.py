from __future__ import annotations

from scripts import r3_capacity


def _row(**kw):
    base = {
        "scenario_id": "i128_o64_c4", "input_tokens": "128", "output_tokens": "64",
        "generation_tokens_total": "8192", "window_start_ms": "0", "window_end_ms": "60000",
        "p95_ttft": "60.0", "p95_tpot": "25.0",
    }
    base.update(kw)
    return base


def test_sample_rps_and_slo_met() -> None:
    # 8192 gen / 64 out = 128 reqs over 60s => ~2.133 rps; ttft/tpot within SLO
    s = r3_capacity.sample_from_row(_row(), ttft_slo_ms=500, tpot_slo_ms=75)
    assert abs(s.rps - (8192 / 64 / 60.0)) < 1e-6
    assert s.slo_met is True
    assert s.input_tokens == 128 and s.output_tokens == 64


def test_sample_slo_violation_marks_not_met() -> None:
    s = r3_capacity.sample_from_row(_row(p95_ttft="900.0"), ttft_slo_ms=500, tpot_slo_ms=75)
    assert s.slo_met is False


def test_sample_none_latency_not_met() -> None:
    s = r3_capacity.sample_from_row(_row(p95_ttft=""), ttft_slo_ms=500, tpot_slo_ms=75)
    assert s.slo_met is False


def test_surface_keeps_max_slo_met_rps() -> None:
    from tre_calibration.capacity import CapacitySample, fit_capacity_surface
    samples = [
        CapacitySample("m", 128, 64, 5.0, True),
        CapacitySample("m", 128, 64, 8.0, True),
        CapacitySample("m", 128, 64, 20.0, False),  # violating: ignored
    ]
    surface = fit_capacity_surface(samples)
    assert surface.points[("m", 128, 64)] == 8.0
    js = r3_capacity.surface_to_json("m", surface)
    assert js["capacity"][0] == {"input_tokens": 128, "output_tokens": 64, "rps": 8.0}

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import gen_calibration_schedules as gen


# 7b's measured prior: capacity falls with both input and output length.
HEALTHY_POINTS = [
    (128, 128, 16.0),
    (128, 512, 4.2667),
    (512, 128, 11.7333),
    (512, 512, 6.4),
    (1024, 128, 6.1),
    (1024, 512, 4.2),
]

# 14b's 2026-07-09 prior: capacity *rises* from input 128 -> 512 (14.9 -> 32.97), the
# signature of a prefix-cached identical-prompt sender. The fit must refuse it rather
# than silently produce an inverted capacity surface.
CONTAMINATED_POINTS = [
    (128, 128, 14.9),
    (128, 512, 4.2667),
    (512, 128, 32.9667),
    (512, 512, 12.8),
    (1024, 128, 32.0),
    (1024, 512, 12.8),
]


def test_capacity_model_is_monotone_decreasing_in_both_axes() -> None:
    model = gen.fit_capacity_model("dsqwen-7b", HEALTHY_POINTS)
    assert model.prefill_tps > 0 and model.decode_tps > 0
    assert model.rps(256, 128) > model.rps(1024, 128)
    assert model.rps(256, 128) > model.rps(256, 448)
    # the fit should track the measurement it was built from
    assert model.rms_rel_error < 0.35


def test_capacity_model_rejects_a_contaminated_prior() -> None:
    with pytest.raises(ValueError, match="unphysical"):
        gen.fit_capacity_model("dsqwen-14b", CONTAMINATED_POINTS)


def test_mixture_capacity_is_the_weighted_harmonic_combination() -> None:
    model = gen.fit_capacity_model("dsqwen-7b", HEALTHY_POINTS)
    c_m = model.mixture_rps(gen.MIXTURE)
    singles = [model.rps(i, o) for _w, i, o in gen.MIXTURE]
    # a blend saturates between its cheapest and its most expensive component
    assert min(singles) < c_m < max(singles)
    expected = 1.0 / sum(w / model.rps(i, o) for w, i, o in gen.MIXTURE)
    assert abs(c_m - expected) < 1e-9


def test_ramp_holds_then_drains() -> None:
    segs = gen.ramp_segments(10.0)
    assert segs[0]["start_time"] == 0
    assert max(s["end_time"] for s in segs) == 540
    rising = [s["rps"] for s in segs[: int(gen.RAMP_DURATION_S / gen.RAMP_SEGMENT_S)]]
    assert rising == sorted(rising)
    assert rising[0] > gen.RAMP_RHO_START * 10.0  # midpoint of the first 5 s segment
    assert segs[-2]["rps"] == pytest.approx(gen.RAMP_RHO_END * 10.0)
    assert segs[-1]["rps"] == pytest.approx(gen.RAMP_DRAIN_RHO * 10.0)


def test_steps_are_monotone_and_carry_discard_boundaries() -> None:
    segs = gen.step_segments(10.0)
    rates = [s["rps"] for s in segs]
    assert rates == sorted(rates)
    assert max(s["end_time"] for s in segs) == 450
    model = gen.fit_capacity_model("dsqwen-7b", HEALTHY_POINTS)
    _body, meta = gen.build_schedule("dsqwen-7b", "S1", "steps", model)
    assert meta["discard_after_s"] == [0, 90, 210]


def test_bursts_superpose_a_spike_on_the_base_rate() -> None:
    segs = gen.burst_segments(10.0)
    base = [s for s in segs if s["end_time"] - s["start_time"] > 10]
    spikes = [s for s in segs if s["end_time"] - s["start_time"] == gen.BURST_WIDTH_S]
    assert len(base) == 1 and base[0]["rps"] == pytest.approx(6.0)
    assert len(spikes) == gen.BURST_COUNT
    # B = 2 * C_s * T_s requests in T_s seconds -> a 2*C_s segment on top of the base
    assert all(s["rps"] == pytest.approx(20.0) for s in spikes)
    assert [s["start_time"] for s in spikes] == [60, 150, 240, 330]
    requests_per_burst = gen.BURST_MULTIPLIER * 10.0 * gen.BURST_WIDTH_S
    assert requests_per_burst == 40.0


def test_schedule_files_use_the_replayer_trace_schema(tmp_path: Path) -> None:
    from tre_replayer.traces.loader import load_trace_segments

    model = gen.fit_capacity_model("dsqwen-7b", HEALTHY_POINTS)
    body, meta = gen.build_schedule("dsqwen-7b", "S4", "bursts", model)
    path = tmp_path / "S4_bursts.json"
    path.write_text(json.dumps(body), encoding="utf-8")

    segments = load_trace_segments(path)
    assert segments and all(s.model == "dsqwen-7b" for s in segments)
    assert {(s.input_tokens, s.max_output_tokens) for s in segments} == {(256, 448)}
    assert meta["cell_id"] == "i256_o448_c60"
    assert meta["held_out"] is False


def test_mixture_schedule_is_held_out_and_runs_shapes_in_parallel(tmp_path: Path) -> None:
    from tre_replayer.traces.loader import load_trace_segments

    model = gen.fit_capacity_model("dsqwen-7b", HEALTHY_POINTS)
    body, meta = gen.build_schedule("dsqwen-7b", gen.MIXTURE_NAME, "steps", model)
    path = tmp_path / "M_steps.json"
    path.write_text(json.dumps(body), encoding="utf-8")

    segments = load_trace_segments(path)
    shapes = {(s.input_tokens, s.max_output_tokens) for s in segments}
    assert shapes == {(i, o) for _w, i, o in gen.MIXTURE}
    assert meta["held_out"] is True
    # i0_o0 keeps the id parseable while telling r3_capacity not to fit a capacity point
    assert meta["cell_id"] == "i0_o0_c95"
    from scripts.r3_capacity import sample_from_row

    row = {
        "scenario_id": meta["cell_id"], "input_tokens": 0, "output_tokens": 0,
        "generation_tokens_total": "100", "window_start_ms": "0", "window_end_ms": "30000",
        "p95_ttft": "100", "p95_tpot": "10",
    }
    assert sample_from_row(row, ttft_slo_ms=500.0, tpot_slo_ms=75.0) is None


def test_every_cell_id_round_trips_through_the_grid_parser() -> None:
    # rewindow_from_raw silently SKIPS a raw file whose name is not a GridCell id, so an
    # unparseable cell id would throw away a whole cell's capture without an error.
    from scripts.r3_grid import GridCell

    model = gen.fit_capacity_model("dsqwen-7b", HEALTHY_POINTS)
    ids = set()
    for shape in [*gen.SHAPES, gen.MIXTURE_NAME]:
        for primitive in gen.PRIMITIVES:
            _body, meta = gen.build_schedule("dsqwen-7b", shape, primitive, model)
            GridCell.from_scenario_id(meta["cell_id"])
            ids.add(meta["cell_id"])
    assert len(ids) == len(gen.SHAPES) * len(gen.PRIMITIVES) + len(gen.PRIMITIVES)


def test_generate_writes_every_shape_and_primitive(tmp_path: Path) -> None:
    out = tmp_path / "calibration"
    index = gen.generate(
        tmp_path, out, ["dsqwen-7b"],
        capacity_overrides={"dsqwen-7b": ("dsqwen-7b", HEALTHY_POINTS)},
    )
    expected = (len(gen.SHAPES) + 1) * len(gen.PRIMITIVES)
    assert len(index["schedules"]) == expected
    assert len(list((out / "dsqwen-7b").glob("*.json"))) == expected
    assert index["capacity_models"]["dsqwen-7b"]["prefill_tokens_per_s"] > 0
    for meta in index["schedules"]:
        assert meta["planned_requests"] > 0
        assert meta["capacity_prior_rps"] > 0

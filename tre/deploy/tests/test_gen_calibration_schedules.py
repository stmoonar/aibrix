from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts import gen_calibration_schedules as gen
from scripts.admission_cap import (
    DEFAULT_CAP_NAME,
    ENGINE_CAPPED,
    GATEWAY_CAPPED,
    get_cap,
    ramp_backlog_coefficient,
)
from tre_replayer.engine.schedule import TokenRange

#: What the cluster runs, and therefore what an unqualified call generates against.
DEPLOYED = get_cap(DEFAULT_CAP_NAME)


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

# Measured GPU KV cache sizes (vLLM startup log, 2026-09-20). The 8b engine has the
# smallest cache, so under the deployed policy it is the only model where a
# gateway-admissible burst can push the engine past its running limit.
KV_7B = 349232
KV_8B = 147536


@pytest.fixture()
def capacity_model() -> gen.CapacityModel:
    return gen.fit_capacity_model("dsqwen-7b", HEALTHY_POINTS)


def test_capacity_model_is_monotone_decreasing_in_both_axes(capacity_model) -> None:
    model = capacity_model
    assert model.prefill_tps > 0 and model.decode_tps > 0
    assert model.rps(256, 128) > model.rps(1024, 128)
    assert model.rps(256, 128) > model.rps(256, 448)
    # the fit should track the measurement it was built from
    assert model.rms_rel_error < 0.35


def test_capacity_model_rejects_a_contaminated_prior() -> None:
    with pytest.raises(ValueError, match="unphysical"):
        gen.fit_capacity_model("dsqwen-14b", CONTAMINATED_POINTS)


def test_mixture_capacity_is_the_weighted_harmonic_combination(capacity_model) -> None:
    c_m = capacity_model.mixture_rps(gen.MIXTURE)
    singles = [capacity_model.rps(i, o) for _w, i, o in gen.MIXTURE]
    # a blend saturates between its cheapest and its most expensive component
    assert min(singles) < c_m < max(singles)
    expected = 1.0 / sum(w / capacity_model.rps(i, o) for w, i, o in gen.MIXTURE)
    assert abs(c_m - expected) < 1e-9


# ------------------------------------------------------------------ KV cache prior


def test_load_kv_cache_tokens_reads_the_recorded_measurement(tmp_path: Path) -> None:
    path = tmp_path / "capacity_dsqwen-7b.json"
    path.write_text(
        json.dumps({"model": "dsqwen-7b", "kv_cache_tokens": KV_7B, "capacity": []}),
        encoding="utf-8",
    )
    assert gen.load_kv_cache_tokens(path) == KV_7B


def test_missing_kv_cache_tokens_fails_loudly(tmp_path: Path) -> None:
    # Guessing it would silently produce bursts that only measure the Envoy gateway.
    path = tmp_path / "capacity_dsqwen-7b.json"
    path.write_text(json.dumps({"model": "dsqwen-7b", "capacity": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="kv_cache_tokens"):
        gen.load_kv_cache_tokens(path)


def test_tokens_per_request_is_i_plus_o_and_load_weighted_for_the_mixture() -> None:
    assert gen.tokens_per_request("S3") == 2048 + 96
    assert gen.tokens_per_request(gen.MIXTURE_NAME) == pytest.approx(
        sum(w * (i + o) for w, i, o in gen.MIXTURE)
    )


# ------------------------------------------------------------------ ramp


def test_ramp_backlog_coefficient_is_derived_from_the_rho_endpoints() -> None:
    # It must not be restated as a literal anywhere, or a change to the rho endpoints
    # would silently leave the budget arithmetic behind.
    assert gen.RAMP_BACKLOG_COEFFICIENT == pytest.approx(
        ramp_backlog_coefficient(gen.RAMP_RHO_START, gen.RAMP_RHO_END, gen.RAMP_HOLD_FRACTION)
    )
    assert gen.RAMP_BACKLOG_COEFFICIENT == pytest.approx(0.075)


def test_ramp_duration_spends_the_caps_excess_budget() -> None:
    # T_r solves coefficient * C_s * T_r = budget, clamped at the cap's ramp_max_s.
    budget = GATEWAY_CAPPED.ramp_excess_budget
    assert gen.ramp_duration_s(1.0, GATEWAY_CAPPED) == GATEWAY_CAPPED.ramp_max_s
    assert gen.ramp_duration_s(20.0, GATEWAY_CAPPED) == pytest.approx(
        budget / (gen.RAMP_BACKLOG_COEFFICIENT * 20.0)
    )
    # a cap that admits far more lets every shape reach the clamp instead
    assert gen.ramp_duration_s(20.0, ENGINE_CAPPED) == ENGINE_CAPPED.ramp_max_s
    assert gen.ramp_duration_s(20.0, ENGINE_CAPPED) > gen.ramp_duration_s(20.0, GATEWAY_CAPPED)


def test_ramp_climbs_to_rho_1_2_then_holds_and_drains() -> None:
    segs = gen.ramp_segments(10.0, 100.0)
    assert segs[0]["start_time"] == 0
    rising = [s["rps"] for s in segs[:-2]]
    assert rising == sorted(rising)
    assert len(rising) == round(100.0 / gen.RAMP_SEGMENT_S)
    # midpoint rho, so the first segment is above rho_start and the last below rho_end
    assert gen.RAMP_RHO_START * 10.0 < rising[0]
    assert rising[-1] < gen.RAMP_RHO_END * 10.0
    hold, drain = segs[-2], segs[-1]
    assert hold["start_time"] == 100 and hold["end_time"] == 125  # T_r / 4
    assert hold["rps"] == pytest.approx(gen.RAMP_RHO_END * 10.0)
    assert drain["start_time"] == 125
    assert drain["rps"] == pytest.approx(gen.RAMP_DRAIN_RHO * 10.0)
    assert max(s["end_time"] for s in segs) == 125 + gen.RAMP_DRAIN_S


def test_ramp_metadata_records_the_per_shape_timeline(capacity_model) -> None:
    _body, meta = gen.build_schedule("dsqwen-7b", "S1", "ramp", capacity_model)
    c_s = capacity_model.rps(*gen.SHAPES["S1"])
    expected = DEPLOYED.ramp_seconds(c_s, gen.RAMP_BACKLOG_COEFFICIENT)
    assert meta["ramp_s"] == pytest.approx(expected, abs=1e-3)
    assert meta["hold_s"] == pytest.approx(expected / 4.0, abs=1e-3)
    assert meta["drain_s"] == gen.RAMP_DRAIN_S
    # a cell truncated on the first proxy 503 jumps straight to this offset
    assert meta["drain_start_s"] == pytest.approx(expected * 1.25, abs=1e-3)
    assert (meta["rho_start"], meta["rho_end"]) == (0.4, 1.2)
    assert meta["drain_rho"] == 0.5
    assert meta["admission_cap"] == DEPLOYED.name


def test_ramp_load_code_tracks_its_characteristic_rho() -> None:
    from scripts.r3_grid import GridCell

    assert gen.LOAD_CODE["ramp"] == round(100 * gen.RAMP_RHO_END) == 120
    GridCell.from_scenario_id("i256_o128_c120")


# ------------------------------------------------------------------ steps


def test_steps_are_monotone_and_carry_discard_boundaries(capacity_model) -> None:
    segs = gen.step_segments(10.0)
    rates = [s["rps"] for s in segs]
    assert rates == sorted(rates)
    assert max(s["end_time"] for s in segs) == 450
    _body, meta = gen.build_schedule("dsqwen-7b", "S1", "steps", capacity_model)
    assert meta["discard_after_s"] == [0, 90, 210]


# ------------------------------------------------------------------ bursts


def test_burst_size_overshoots_the_engine_running_limit() -> None:
    # Only the engine queues, and it only queues past min(max_num_seqs, KV capacity).
    sizing = GATEWAY_CAPPED.burst_sizing(KV_8B, 2144.0)
    assert sizing.kv_request_limit == math.floor(KV_8B / 2144.0) == 68
    assert sizing.engine_running_limit == 68 and sizing.binding_limit == "kv_cache"
    assert sizing.requests_needed == math.ceil(1.5 * 68) == 102
    assert sizing.reachable and sizing.requests == 102


def test_bursts_superpose_a_spike_on_the_base_rate() -> None:
    segs = gen.burst_segments(10.0, 102)
    base = [s for s in segs if s["end_time"] - s["start_time"] > 10]
    spikes = [s for s in segs if s["end_time"] - s["start_time"] == gen.BURST_WIDTH_S]
    assert len(base) == 1 and base[0]["rps"] == pytest.approx(6.0)
    assert len(spikes) == gen.BURST_COUNT
    # B requests inside BURST_WIDTH_S seconds -> a B / T_s segment on top of the base
    assert all(s["rps"] == pytest.approx(102 / gen.BURST_WIDTH_S) for s in spikes)
    assert [s["start_time"] for s in spikes] == [60, 150, 240, 330]
    assert all(s["start_time"] % gen.BURST_PERIOD_S == 60 for s in spikes)


def test_a_shape_whose_burst_would_be_shed_is_skipped(capacity_model) -> None:
    # Only reachable under the SUPERSEDED gateway-capped policy, which is why the cap has
    # to be named: under the deployed one the admission budget is 3840 and nothing skips.
    body, meta = gen.build_schedule(
        "dsqwen-7b", "S1", "bursts", capacity_model, KV_7B, cap=GATEWAY_CAPPED
    )
    assert body is None
    assert meta["skipped"] is True
    assert meta["reachable"] is False
    assert "path" not in meta
    assert meta["kv_cache_tokens"] == KV_7B
    assert meta["tokens_per_request"] == 384.0
    assert meta["engine_running_limit"] == math.floor(KV_7B / 384)
    assert meta["burst_requests_needed"] == 1364
    assert meta["burst_request_cap"] == GATEWAY_CAPPED.burst_request_cap == 240
    assert "shed by the gateway" in meta["reason"]


def test_a_shape_whose_burst_is_admissible_keeps_its_schedule(capacity_model) -> None:
    body, meta = gen.build_schedule(
        "dsqwen-7b", "S3", "bursts", capacity_model, KV_8B, cap=GATEWAY_CAPPED
    )
    assert body is not None
    assert meta["skipped"] is False and meta["reachable"] is True
    assert meta["burst_requests"] == 102
    assert meta["burst_segment_rps"] == pytest.approx(102 / gen.BURST_WIDTH_S)
    assert meta["peak_offered_rps"] > meta["capacity_rps"]
    assert meta["reason"]


def test_raising_the_admission_cap_makes_a_skipped_shape_viable(capacity_model) -> None:
    # The same (model, shape) that the deployed policy cannot observe becomes
    # observable once the binding limit moves into the engine.
    _shed, shed_meta = gen.build_schedule(
        "dsqwen-7b", "S1", "bursts", capacity_model, KV_7B, cap=GATEWAY_CAPPED
    )
    body, meta = gen.build_schedule(
        "dsqwen-7b", "S1", "bursts", capacity_model, KV_7B, cap=ENGINE_CAPPED
    )
    assert shed_meta["skipped"] is True
    assert body is not None and meta["skipped"] is False
    assert meta["binding_limit"] == "max_num_seqs"
    assert meta["engine_running_limit"] == ENGINE_CAPPED.sequence_limit == 256
    assert meta["burst_requests"] == 384
    assert meta["admission_cap"] == ENGINE_CAPPED.name


def test_bursts_without_a_kv_cache_size_raise(capacity_model) -> None:
    with pytest.raises(ValueError, match="kv_cache_tokens"):
        gen.build_schedule("dsqwen-7b", "S1", "bursts", capacity_model)


# ------------------------------------------------- explicit-capacity entry point


def test_build_schedule_from_capacity_rps_uses_the_supplied_capacity() -> None:
    body, meta = gen.build_schedule_from_capacity_rps(
        "dsqwen-7b", "S1", "ramp", 20.0, capacity_source="measured_steps"
    )
    assert body is not None
    assert meta["capacity_rps"] == 20.0
    assert meta["capacity_source"] == "measured_steps"
    expected = DEPLOYED.ramp_seconds(20.0, gen.RAMP_BACKLOG_COEFFICIENT)
    assert meta["ramp_s"] == pytest.approx(expected, abs=1e-3)
    assert meta["drain_start_s"] == pytest.approx(expected * 1.25, abs=1e-3)
    peak = max(s["rps"] for s in body["dsqwen-7b"])
    assert peak == pytest.approx(gen.RAMP_RHO_END * 20.0)


def test_build_schedule_delegates_to_the_explicit_capacity_entry_point(capacity_model) -> None:
    c_s = capacity_model.rps(*gen.SHAPES["S2"])
    body_a, meta_a = gen.build_schedule("dsqwen-7b", "S2", "ramp", capacity_model)
    body_b, meta_b = gen.build_schedule_from_capacity_rps("dsqwen-7b", "S2", "ramp", c_s)
    assert body_a == body_b
    assert meta_a == meta_b
    assert meta_a["capacity_source"] == "prior_fit"


def test_explicit_capacity_carries_the_cap_through() -> None:
    body, meta = gen.build_schedule_from_capacity_rps(
        "dsqwen-7b", "S3", "bursts", 4.0, kv_cache_tokens=KV_7B, cap=ENGINE_CAPPED
    )
    assert body is not None
    assert meta["admission_cap"] == ENGINE_CAPPED.name
    assert meta["burst_requests"] == 243
    with pytest.raises(ValueError, match="kv_cache_tokens"):
        gen.build_schedule_from_capacity_rps("dsqwen-7b", "S3", "bursts", 4.0)


# ------------------------------------------------------------------ whole schedules


def test_schedule_files_use_the_replayer_trace_schema(capacity_model, tmp_path: Path) -> None:
    from tre_replayer.traces.loader import load_trace_segments

    body, meta = gen.build_schedule("dsqwen-7b", "S3", "bursts", capacity_model, KV_8B)
    path = tmp_path / "S3_bursts.json"
    path.write_text(json.dumps(body), encoding="utf-8")

    segments = load_trace_segments(path)
    assert segments and all(s.model == "dsqwen-7b" for s in segments)
    assert {(s.input_tokens, s.max_output_tokens) for s in segments} == {(2048, 96)}
    assert meta["cell_id"] == "i2048_o96_c60"
    assert meta["held_out"] is False


def test_mixture_schedule_is_held_out_and_runs_shapes_in_parallel(
    capacity_model, tmp_path: Path
) -> None:
    from tre_replayer.traces.loader import load_trace_segments

    body, meta = gen.build_schedule("dsqwen-7b", gen.MIXTURE_NAME, "steps", capacity_model)
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


def test_mixture_burst_splits_its_requests_across_the_component_streams(
    capacity_model,
) -> None:
    body, meta = gen.build_schedule(
        "dsqwen-7b", gen.MIXTURE_NAME, "bursts", capacity_model, KV_8B
    )
    assert body is not None
    spikes = [
        s for s in body["dsqwen-7b"]
        if s["end_time"] - s["start_time"] == gen.BURST_WIDTH_S
        and s["start_time"] == gen.BURST_FIRST_S
    ]
    assert len(spikes) == len(gen.MIXTURE)
    total = sum(s["rps"] for s in spikes) * gen.BURST_WIDTH_S
    assert total == pytest.approx(meta["burst_requests"], abs=1e-3)


def test_every_cell_id_round_trips_through_the_grid_parser(capacity_model) -> None:
    # rewindow_from_raw silently SKIPS a raw file whose name is not a GridCell id, so an
    # unparseable cell id would throw away a whole cell's capture without an error.
    from scripts.r3_grid import GridCell

    ids = set()
    for shape in gen.ALL_SHAPES:
        for primitive in gen.PRIMITIVES:
            _body, meta = gen.build_schedule(
                "dsqwen-7b", shape, primitive, capacity_model, KV_7B
            )
            GridCell.from_scenario_id(meta["cell_id"])
            ids.add(meta["cell_id"])
    # One id per (shape, primitive): a collision would make two cells share a raw file.
    assert len(ids) == len(gen.ALL_SHAPES) * len(gen.PRIMITIVES)


def test_generate_indexes_every_cell_and_writes_only_the_feasible_ones(
    tmp_path: Path,
) -> None:
    out = tmp_path / "calibration"
    index = gen.generate(
        tmp_path, out, ["dsqwen-7b"],
        capacity_overrides={"dsqwen-7b": ("dsqwen-7b", HEALTHY_POINTS)},
        kv_cache_overrides={"dsqwen-7b": KV_8B},
        cap=GATEWAY_CAPPED,
    )
    cells = len(gen.ALL_SHAPES) * len(gen.PRIMITIVES)
    assert len(index["schedules"]) == cells
    assert index["admission_cap"]["name"] == GATEWAY_CAPPED.name

    written = [m for m in index["schedules"] if not m["skipped"]]
    skipped = [m for m in index["schedules"] if m["skipped"]]
    # Under the superseded policy only 240 requests were admitted, so the light shapes'
    # bursts could not overshoot the engine and were skipped.
    assert {m["shape"] for m in skipped} == {"S1", "S4"}
    assert all(m["primitive"] == "bursts" for m in skipped)
    assert all(m["reason"] for m in skipped)
    assert len(list((out / "dsqwen-7b").glob("*.json"))) == len(written)

    assert index["capacity_models"]["dsqwen-7b"]["prefill_tokens_per_s"] > 0
    assert index["capacity_models"]["dsqwen-7b"]["kv_cache_tokens"] == KV_8B
    for meta in written:
        assert (out / meta["path"]).exists()
        assert meta["planned_requests"] > 0
        assert meta["capacity_rps"] > 0
    for meta in skipped:
        assert "path" not in meta
        assert not (out / "dsqwen-7b" / f"{meta['shape']}_bursts.json").exists()


def test_generating_for_the_deployed_policy_skips_nothing(tmp_path: Path) -> None:
    # The regression that proves the cap is a real parameter, and the answer to "did
    # raising the ceiling make the previously-unreachable bursts viable": it did. Every
    # burst cell is now observable, so num_requests_waiting - and with it lambda_wait -
    # is identifiable for every shape instead of only the handful that fitted under 240.
    out = tmp_path / "calibration-engine"
    index = gen.generate(
        tmp_path, out, list(gen.MODELS),
        capacity_overrides={m: (m, HEALTHY_POINTS) for m in gen.MODELS},
        kv_cache_overrides={m: KV_7B for m in gen.MODELS},
        cap=ENGINE_CAPPED,
    )
    assert index["admission_cap"]["name"] == ENGINE_CAPPED.name == DEPLOYED.name
    assert index["admission_cap"]["admission_controller"] == "engine"
    bursts = [m for m in index["schedules"] if m["primitive"] == "bursts"]
    assert len(bursts) == len(gen.MODELS) * len(gen.ALL_SHAPES) == 24
    assert not [m for m in index["schedules"] if m["skipped"]]
    assert all((out / m["path"]).exists() for m in bursts)
    ramps = [m for m in index["schedules"] if m["primitive"] == "ramp"]
    assert {m["ramp_s"] for m in ramps} == {ENGINE_CAPPED.ramp_max_s}
    # Every one of them overshoots the ENGINE ceiling, not a gateway number.
    for meta in bursts:
        assert meta["burst_requests_needed"] <= meta["burst_request_cap"]
        assert meta["engine_running_limit"] <= ENGINE_CAPPED.sequence_limit


def test_the_shapes_that_the_old_cap_excluded_are_now_all_viable(capacity_model) -> None:
    # Named explicitly rather than counted: these are the (shape) cells the superseded
    # 320-in-flight ceiling declared unreachable for the 7b engine's 349232-token cache.
    for shape in gen.ALL_SHAPES:
        _old_body, old_meta = gen.build_schedule(
            "dsqwen-7b", shape, "bursts", capacity_model, KV_7B, cap=GATEWAY_CAPPED
        )
        new_body, new_meta = gen.build_schedule(
            "dsqwen-7b", shape, "bursts", capacity_model, KV_7B, cap=DEPLOYED
        )
        assert old_meta["skipped"] is True, shape
        assert new_body is not None and new_meta["skipped"] is False, shape
        # The ceiling that now decides is the engine's, per pod and scaling with
        # replicas: min(max_num_seqs, what the KV cache holds).
        assert new_meta["engine_running_limit"] <= DEPLOYED.sequence_limit == 256
        assert new_meta["binding_limit"] in ("max_num_seqs", "kv_cache")


# ------------------------------------------------------------------ T8 and T9


def test_t8_sits_on_the_real_trace_working_point() -> None:
    # t8 replays Azure conversation inputs whose median is 1628 tokens. A theta fitted
    # only on the synthetic corners would be extrapolating to the point the experiments
    # are actually run at.
    assert gen.SHAPES["T8"] == (1600, 112)
    assert gen.tokens_per_request("T8") == 1600 + 112


def test_t9_lengths_are_sampled_and_their_means_drive_the_sizing() -> None:
    low_in, low_out = gen.SAMPLED_SHAPES["T9"]
    assert (low_in.low, low_in.high) == (300, 2200)
    assert (low_out.low, low_out.high) == (100, 580)
    # E[X] for log-uniform is (high - low) / ln(high / low), NOT the midpoint: capacity
    # and KV footprint are linear in length, so the mean is what sets them.
    assert low_in.mean == pytest.approx((2200 - 300) / math.log(2200 / 300))
    assert gen.tokens_per_request("T9") == pytest.approx(low_in.mean + low_out.mean)
    # ... while the cell id uses the geometric midpoint, which is a name, not a claim
    assert low_in.median == round(math.sqrt(300 * 2200))


def test_t9_schedule_carries_the_distribution_not_a_fixed_length(
    capacity_model, tmp_path: Path
) -> None:
    from tre_replayer.traces.loader import load_trace_segments

    body, meta = gen.build_schedule("dsqwen-7b", "T9", "steps", capacity_model)
    path = tmp_path / "T9_steps.json"
    path.write_text(json.dumps(body), encoding="utf-8")

    segments = load_trace_segments(path)
    assert segments
    for segment in segments:
        assert segment.input_tokens is None and segment.max_output_tokens is None
        assert segment.input_tokens_range == TokenRange(300, 2200)
        assert segment.max_output_tokens_range == TokenRange(100, 580)
        assert segment.sampled
    assert meta["sampled"] is True and meta["held_out"] is False
    assert meta["cell_id"] == "i812_o241_c95"


def test_t9_capacity_prior_is_the_cost_at_the_mean_lengths(capacity_model) -> None:
    low_in, low_out = gen.SAMPLED_SHAPES["T9"]
    assert gen.capacity_rps_for_shape(capacity_model, "T9") == pytest.approx(
        capacity_model.rps(low_in.mean, low_out.mean)
    )


def test_the_held_out_shape_is_not_a_training_shape() -> None:
    # The one structural guarantee that stops the validation set reaching the fit.
    assert gen.MIXTURE_NAME not in gen.TRAINING_SHAPES
    assert gen.MIXTURE_NAME in gen.ALL_SHAPES
    assert gen.is_held_out(gen.MIXTURE_NAME)
    assert not any(gen.is_held_out(s) for s in gen.TRAINING_SHAPES)
    # and no family may quietly include it
    for members in gen.FAMILIES.values():
        assert gen.MIXTURE_NAME not in members
        assert all(m in gen.TRAINING_SHAPES for m in members)


def test_the_index_marks_the_held_out_shape_on_every_one_of_its_cells(
    tmp_path: Path,
) -> None:
    index = gen.generate(
        tmp_path, tmp_path / "calibration", ["dsqwen-7b"],
        capacity_overrides={"dsqwen-7b": ("dsqwen-7b", HEALTHY_POINTS)},
        kv_cache_overrides={"dsqwen-7b": KV_8B},
    )
    assert index["held_out_shapes"] == [gen.MIXTURE_NAME]
    assert index["training_shapes"] == list(gen.TRAINING_SHAPES)
    held = {m["shape"] for m in index["schedules"] if m["held_out"]}
    assert held == {gen.MIXTURE_NAME}
    assert all(m["held_out"] for m in index["schedules"] if m["shape"] == gen.MIXTURE_NAME)


# ------------------------------------------------------------------ hold primitive


def test_hold_is_one_flat_segment_at_the_requested_rho() -> None:
    segs = gen.hold_segments(10.0, 0.85, 120.0)
    assert len(segs) == 1
    assert segs[0]["rps"] == pytest.approx(8.5)
    assert (segs[0]["start_time"], segs[0]["end_time"]) == (0, 120)


def test_hold_cell_ids_encode_their_own_rho() -> None:
    from scripts.r3_grid import GridCell

    body, meta = gen.build_hold_schedule("dsqwen-7b", "S1", 10.0, 0.87, 120.0, stage="bisect")
    assert meta["cell_id"] == f"i256_o128_c{gen.hold_load_code(0.87)}" == "i256_o128_c1087"
    GridCell.from_scenario_id(meta["cell_id"])
    assert meta["rho"] == 0.87 and meta["stage"] == "bisect"
    assert meta["offered_rps"] == pytest.approx(8.7)
    assert max(s["end_time"] for s in body["dsqwen-7b"]) == 120


def test_a_hold_can_never_collide_with_a_committed_primitives_cell_id() -> None:
    # The raw tree is keyed by cell id, so a collision silently merges two different
    # cells' windows. A probe at rho 0.95 / 0.6 / 1.2 would otherwise land on the steps /
    # bursts / ramp code of the same shape.
    fixed = set(gen.LOAD_CODE.values())
    codes = {gen.hold_load_code(r / 100.0) for r in range(1, 300)}
    assert not (codes & fixed)
    assert min(codes) > max(fixed)
    ids = {
        gen.build_hold_schedule("dsqwen-7b", "S1", 10.0, r / 100.0, 60.0)[1]["cell_id"]
        for r in (60, 95, 120)
    }
    committed = {
        gen.build_schedule_from_capacity_rps(
            "dsqwen-7b", "S1", p, 10.0, kv_cache_tokens=KV_7B
        )[1]["cell_id"]
        for p in gen.PRIMITIVES
    }
    assert not (ids & committed)


def test_a_hold_without_a_rho_is_a_loud_failure() -> None:
    with pytest.raises(ValueError, match="hold_rho"):
        gen.build_schedule_from_capacity_rps("dsqwen-7b", "S1", gen.HOLD_PRIMITIVE, 10.0)


def test_generate_refuses_a_capacity_override_without_a_kv_cache_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="kv_cache_overrides"):
        gen.generate(
            tmp_path, tmp_path / "calibration", ["dsqwen-7b"],
            capacity_overrides={"dsqwen-7b": ("dsqwen-7b", HEALTHY_POINTS)},
        )


def test_the_primitive_blocks_only_carry_shape_independent_constants(
    tmp_path: Path,
) -> None:
    # T_r, B and the hold all depend on C_s, the shape and the cap, so a global value
    # for any of them here would be a lie.
    index = gen.generate(
        tmp_path, tmp_path / "calibration", ["dsqwen-7b"],
        capacity_overrides={"dsqwen-7b": ("dsqwen-7b", HEALTHY_POINTS)},
        kv_cache_overrides={"dsqwen-7b": KV_8B},
    )
    ramp = index["primitives"]["ramp"]
    assert "ramp_s" not in ramp and "hold_s" not in ramp
    assert ramp["ramp_s_formula"] == gen.RAMP_DURATION_FORMULA
    assert ramp["backlog_coefficient"] == pytest.approx(0.075)
    assert (ramp["rho_start"], ramp["rho_end"]) == (0.4, 1.2)
    bursts = index["primitives"]["bursts"]
    assert "multiplier" not in bursts and "burst_requests" not in bursts
    assert "kv_cache_tokens" in bursts["burst_requests_formula"]

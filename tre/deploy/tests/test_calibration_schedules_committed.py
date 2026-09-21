"""Guards on the committed calibration schedule set (traces_v2/calibration/)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import gen_calibration_schedules as gen
from scripts.admission_cap import DEFAULT_CAP_NAME, get_cap
from scripts.r3_grid import GridCell
from tre_replayer.engine.schedule import TokenRange
from tre_replayer.traces.loader import load_trace_segments

ROOT = Path(__file__).resolve().parents[2]
CALIB = ROOT / "replayer" / "traces_v2" / "calibration"
INDEX = json.loads((CALIB / "INDEX.json").read_text(encoding="utf-8"))
CAP = get_cap(DEFAULT_CAP_NAME)


def _written() -> list[dict]:
    return [m for m in INDEX["schedules"] if not m["skipped"]]


def _skipped() -> list[dict]:
    return [m for m in INDEX["schedules"] if m["skipped"]]


def test_every_committed_prior_is_physically_monotone() -> None:
    # A prior whose capacity rises with token count means the sender was serving the same
    # prompt out of the prefix cache; every rho built on it would be meaningless.
    for model in gen.MODELS:
        _name, points = gen.load_capacity_points(CALIB / "capacity" / f"capacity_{model}.json")
        fitted = gen.fit_capacity_model(model, points)
        assert fitted.rps(256, 128) > fitted.rps(2048, 128)
        assert fitted.rps(256, 128) > fitted.rps(256, 448)


def test_every_committed_prior_records_its_kv_cache_with_provenance() -> None:
    # Burst sizing is driven by the engine's KV cache, so an unattributed number here
    # would be an unfalsifiable input to the whole campaign.
    for model in gen.MODELS:
        path = CALIB / "capacity" / f"capacity_{model}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert gen.load_kv_cache_tokens(path) == data["kv_cache_tokens"] > 0
        prov = data["kv_cache_provenance"]
        assert "GPU KV cache size" in prov["source"]
        assert prov["measured"] and prov["pod"].startswith(model)
        assert INDEX["capacity_models"][model]["kv_cache_tokens"] == data["kv_cache_tokens"]


def test_frozen_experiment_priors_are_not_the_campaign_priors() -> None:
    # traceset-v2 must stay byte-unchanged; the campaign keeps its own dated copy.
    frozen = ROOT / "replayer" / "traces_v2" / "capacity" / "capacity_dsqwen-14b.json"
    campaign = CALIB / "capacity" / "capacity_dsqwen-14b.json"
    assert json.loads(frozen.read_text()) != json.loads(campaign.read_text())


def test_the_committed_set_is_generated_for_the_deployed_admission_policy() -> None:
    # A schedule set sized for a policy the cluster does not run would measure the proxy.
    # Since 2026-09-21 the deployed BackendTrafficPolicy is 4096 + 1024 per cluster and
    # every model runs --max-num-seqs 256, so the binding limit is the ENGINE's and it
    # scales with replicas. The superseded 320 ceiling is in admission_cap.GATEWAY_CAPPED.
    assert INDEX["admission_cap"]["name"] == DEFAULT_CAP_NAME == "engine-capped"
    assert INDEX["admission_cap"]["max_parallel_requests"] == 4096
    assert INDEX["admission_cap"]["max_pending_requests"] == 1024
    assert INDEX["admission_cap"]["shed_ceiling"] == 5120
    assert INDEX["admission_cap"]["admission_controller"] == "engine"
    assert INDEX["admission_cap"]["sequence_limit"] == 256
    assert INDEX["admission_cap"]["burst_request_cap"] == 3840


def test_the_committed_cap_matches_what_the_registry_launches_the_pods_with() -> None:
    # The admission ceiling is max_num_seqs * replicas, so hard-coding 256 anywhere would
    # go silently wrong the moment the manifests changed.
    import yaml

    from scripts.admission_cap import max_num_seqs_from_registry

    registry = yaml.safe_load((ROOT / "deploy" / "registry.yaml").read_text(encoding="utf-8"))
    assert max_num_seqs_from_registry(registry, gen.MODELS) == CAP.sequence_limit == 256


def test_index_covers_every_model_shape_and_primitive() -> None:
    expected = len(gen.MODELS) * len(gen.ALL_SHAPES) * len(gen.PRIMITIVES)
    assert len(INDEX["schedules"]) == expected == 72
    combos = {(m["model"], m["shape"], m["primitive"]) for m in INDEX["schedules"]}
    assert len(combos) == expected


def test_nothing_is_skipped_under_the_deployed_policy() -> None:
    # Under the superseded 320 ceiling 13 of 18 burst cells were unreachable, which left
    # num_requests_waiting unobservable for almost every shape and lambda_wait
    # unidentifiable. Raising the ceiling to the engine's own limit removes the skip list.
    assert _skipped() == []
    assert all("path" in m for m in INDEX["schedules"])


def test_every_burst_size_reproduces_the_caps_sizing_rule() -> None:
    for meta in INDEX["schedules"]:
        if meta["primitive"] != "bursts":
            continue
        sizing = CAP.burst_sizing(meta["kv_cache_tokens"], meta["tokens_per_request"])
        assert meta["tokens_per_request"] == pytest.approx(
            gen.tokens_per_request(meta["shape"])
        )
        assert meta["burst_requests_needed"] == sizing.requests_needed
        assert meta["engine_running_limit"] == sizing.engine_running_limit
        assert meta["engine_running_limit"] <= CAP.sequence_limit
        assert meta["skipped"] is not sizing.reachable
        assert meta["burst_requests"] == sizing.requests
        assert meta["burst_segment_rps"] == pytest.approx(
            sizing.requests / gen.BURST_WIDTH_S
        )
        # if the spike does not exceed C_s the primitive cannot create a transient
        assert meta["peak_offered_rps"] > meta["capacity_rps"]


def test_every_ramp_reproduces_the_caps_duration_rule() -> None:
    ramps = [m for m in INDEX["schedules"] if m["primitive"] == "ramp"]
    assert len(ramps) == len(gen.MODELS) * len(gen.ALL_SHAPES)
    for meta in ramps:
        expected = CAP.ramp_seconds(meta["capacity_rps"], gen.RAMP_BACKLOG_COEFFICIENT)
        # Under the deployed cap the admission headroom is 3980 requests, far more than
        # any campaign ramp accumulates, so every ramp is pinned at the ramp_max_s clamp.
        assert expected == CAP.ramp_max_s
        assert meta["ramp_s"] == pytest.approx(expected, abs=1e-2)
        assert meta["hold_s"] == pytest.approx(meta["ramp_s"] / 4.0, abs=1e-2)
        assert meta["drain_start_s"] == pytest.approx(
            meta["ramp_s"] + meta["hold_s"], abs=1e-2
        )
        assert meta["drain_s"] == 60.0
        assert (meta["rho_start"], meta["rho_end"], meta["drain_rho"]) == (0.4, 1.2, 0.5)
        assert meta["cell_id"].endswith("_c120")


def test_the_held_out_shape_is_marked_on_every_one_of_its_cells() -> None:
    # The fit excludes held-out cells BY ID, so an unmarked cell is one that reaches the
    # training set.
    assert INDEX["held_out_shapes"] == [gen.MIXTURE_NAME]
    assert INDEX["training_shapes"] == list(gen.TRAINING_SHAPES)
    held = {m["cell_id"] for m in INDEX["schedules"] if m["held_out"]}
    shapes = {m["shape"] for m in INDEX["schedules"] if m["held_out"]}
    assert shapes == {gen.MIXTURE_NAME}
    assert held and all(cid.startswith("i0_o0_") for cid in held)
    assert not any(
        m["held_out"] for m in INDEX["schedules"] if m["shape"] in gen.TRAINING_SHAPES
    )


def test_the_families_name_real_committed_shapes() -> None:
    covered = {m["shape"] for m in INDEX["schedules"]}
    for family, members in INDEX["families"].items():
        assert members, family
        assert set(members) <= covered


def test_every_committed_schedule_loads_and_matches_its_index_entry() -> None:
    for meta in _written():
        path = CALIB / meta["path"]
        segments = load_trace_segments(path)
        assert segments, path
        assert {s.model for s in segments} == {meta["model"]}
        GridCell.from_scenario_id(meta["cell_id"])
        assert abs(max(s.end_s for s in segments) - meta["duration_s"]) < 1e-6
        if meta["shape"] == gen.MIXTURE_NAME:
            shapes = {(s.input_tokens, s.max_output_tokens) for s in segments}
            assert shapes == {(i, o) for _w, i, o in gen.MIXTURE}
        elif meta["shape"] in gen.SAMPLED_SHAPES:
            want_in, want_out = gen.SAMPLED_SHAPES[meta["shape"]]
            for segment in segments:
                assert segment.input_tokens is None and segment.max_output_tokens is None
                assert segment.input_tokens_range == want_in
                assert segment.max_output_tokens_range == want_out
        else:
            shapes = {(s.input_tokens, s.max_output_tokens) for s in segments}
            assert shapes == {tuple(gen.SHAPES[meta["shape"]])}


def test_a_sampled_schedule_produces_reproducible_per_request_lengths() -> None:
    # The whole point of T9 is variance in the generation length. If the schedule builder
    # collapsed it to one value the shape would be a duplicate of a fixed one, and if it
    # were not reproducible the campaign could not be re-run.
    from tre_replayer.engine.schedule import build_poisson_schedule

    meta = next(
        m for m in _written()
        if m["shape"] in gen.SAMPLED_SHAPES and m["primitive"] == "steps"
    )
    segments = load_trace_segments(CALIB / meta["path"])
    first = build_poisson_schedule(segments, seed=1234)
    again = build_poisson_schedule(segments, seed=1234)
    assert [(e.request_id, e.prompt_tokens, e.max_output_tokens) for e in first] == [
        (e.request_id, e.prompt_tokens, e.max_output_tokens) for e in again
    ]
    want_in, want_out = gen.SAMPLED_SHAPES[meta["shape"]]
    assert len({e.max_output_tokens for e in first}) > 5
    assert all(want_in.low <= e.prompt_tokens <= want_in.high for e in first)
    assert all(want_out.low <= e.max_output_tokens <= want_out.high for e in first)
    # and every request has its own prompt seed key, so no two share a prefix
    assert len({e.request_id for e in first}) == len(first)


def test_the_index_and_the_tree_agree_on_which_files_exist() -> None:
    # A schedule left behind by an earlier, larger set would be silently picked up by a
    # campaign that globs the directory.
    on_disk = {
        str(p.relative_to(CALIB)).replace("\\", "/")
        for p in CALIB.glob("*/*.json")
        if p.parent.name in gen.MODELS
    }
    indexed = {m["path"] for m in _written()}
    assert on_disk == indexed
    assert len(indexed) == 72
    for path in indexed:
        assert (CALIB / path).exists()


def test_the_sampled_shape_declares_its_distribution_in_the_index() -> None:
    entry = INDEX["shapes"]["T9"]
    assert entry["sampled"] is True and entry["held_out"] is False
    assert entry["input_tokens_dist"] == TokenRange(300, 2200).as_dict()
    assert entry["max_tokens_dist"] == TokenRange(100, 580).as_dict()
    assert entry["input_tokens_nominal"] == TokenRange(300, 2200).median

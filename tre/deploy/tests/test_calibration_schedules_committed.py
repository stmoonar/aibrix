"""Guards on the committed calibration schedule set (traces_v2/calibration/)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import gen_calibration_schedules as gen
from scripts.admission_cap import DEFAULT_CAP_NAME, get_cap
from scripts.r3_grid import GridCell
from tre_replayer.traces.loader import load_trace_segments

ROOT = Path(__file__).resolve().parents[2]
CALIB = ROOT / "replayer" / "traces_v2" / "calibration"
INDEX = json.loads((CALIB / "INDEX.json").read_text(encoding="utf-8"))
CAP = get_cap(DEFAULT_CAP_NAME)

#: The bursts cells that survive under the DEPLOYED admission policy. Everything else is
#: skipped because the spike needed to push the engine past its running limit is larger
#: than the gateway will admit, so the cell could only ever record gateway shedding.
SURVIVING_BURSTS = {
    ("dsllama-8b", "S2"): 230,
    ("dsllama-8b", "S3"): 102,
    ("dsllama-8b", "S5"): 192,
    ("dsllama-8b", "M"): 231,
    ("dsqwen-14b", "S3"): 141,
}


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
    assert INDEX["admission_cap"]["name"] == DEFAULT_CAP_NAME == "gateway-capped"
    assert INDEX["admission_cap"]["shed_ceiling"] == 320
    assert INDEX["admission_cap"]["admission_controller"] == "gateway"
    assert INDEX["admission_cap"]["burst_request_cap"] == 240


def test_index_covers_every_model_shape_and_primitive() -> None:
    expected = len(gen.MODELS) * (len(gen.SHAPES) + 1) * len(gen.PRIMITIVES)
    assert len(INDEX["schedules"]) == expected
    combos = {(m["model"], m["shape"], m["primitive"]) for m in INDEX["schedules"]}
    assert len(combos) == expected


def test_only_unreachable_bursts_are_skipped_and_each_says_why() -> None:
    skipped = _skipped()
    assert len(skipped) == 13
    assert {m["primitive"] for m in skipped} == {"bursts"}
    # all six dsqwen-7b shapes: its 349232-token KV cache is far too large to overcommit
    # with 240 admitted requests at any of the campaign's token shapes
    assert {m["shape"] for m in skipped if m["model"] == "dsqwen-7b"} == {
        *gen.SHAPES, gen.MIXTURE_NAME
    }
    for meta in skipped:
        assert meta["reachable"] is False
        assert meta["burst_requests_needed"] > meta["burst_request_cap"]
        assert "shed by the gateway" in meta["reason"]
        assert meta["kv_cache_tokens"] > 0 and meta["tokens_per_request"] > 0
        assert "path" not in meta


def test_surviving_bursts_are_exactly_the_reachable_cells() -> None:
    kept = {
        (m["model"], m["shape"]): m["burst_requests"]
        for m in _written() if m["primitive"] == "bursts"
    }
    assert kept == SURVIVING_BURSTS


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
        assert meta["skipped"] is not sizing.reachable
        if sizing.reachable:
            assert meta["burst_requests"] == sizing.requests
            assert meta["burst_segment_rps"] == pytest.approx(
                sizing.requests / gen.BURST_WIDTH_S
            )
            # if the spike does not exceed C_s the primitive cannot create a transient
            assert meta["peak_offered_rps"] > meta["capacity_rps"]


def test_every_ramp_reproduces_the_caps_duration_rule() -> None:
    ramps = [m for m in INDEX["schedules"] if m["primitive"] == "ramp"]
    assert len(ramps) == len(gen.MODELS) * (len(gen.SHAPES) + 1)
    for meta in ramps:
        expected = CAP.ramp_seconds(meta["capacity_rps"], gen.RAMP_BACKLOG_COEFFICIENT)
        assert meta["ramp_s"] == pytest.approx(expected, abs=1e-2)
        assert meta["hold_s"] == pytest.approx(meta["ramp_s"] / 4.0, abs=1e-2)
        assert meta["drain_start_s"] == pytest.approx(
            meta["ramp_s"] + meta["hold_s"], abs=1e-2
        )
        assert meta["drain_s"] == 60.0
        assert (meta["rho_start"], meta["rho_end"], meta["drain_rho"]) == (0.4, 1.2, 0.5)
        assert meta["cell_id"].endswith("_c120")


def test_every_committed_schedule_loads_and_matches_its_index_entry() -> None:
    for meta in _written():
        path = CALIB / meta["path"]
        segments = load_trace_segments(path)
        assert segments, path
        assert {s.model for s in segments} == {meta["model"]}
        GridCell.from_scenario_id(meta["cell_id"])
        assert abs(max(s.end_s for s in segments) - meta["duration_s"]) < 1e-6
        shapes = {(s.input_tokens, s.max_output_tokens) for s in segments}
        if meta["shape"] == gen.MIXTURE_NAME:
            assert shapes == {(i, o) for _w, i, o in gen.MIXTURE}
        else:
            assert shapes == {tuple(gen.SHAPES[meta["shape"]])}


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
    assert len(indexed) == 41
    for path in indexed:
        assert (CALIB / path).exists()

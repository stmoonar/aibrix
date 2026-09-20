"""Guards on the committed calibration schedule set (traces_v2/calibration/)."""
from __future__ import annotations

import json
from pathlib import Path

from scripts import gen_calibration_schedules as gen
from scripts.r3_grid import GridCell
from tre_replayer.traces.loader import load_trace_segments

ROOT = Path(__file__).resolve().parents[2]
CALIB = ROOT / "replayer" / "traces_v2" / "calibration"
INDEX = json.loads((CALIB / "INDEX.json").read_text(encoding="utf-8"))


def test_every_committed_prior_is_physically_monotone() -> None:
    # A prior whose capacity rises with token count means the sender was serving the same
    # prompt out of the prefix cache; every rho built on it would be meaningless.
    for model in gen.MODELS:
        _name, points = gen.load_capacity_points(CALIB / "capacity" / f"capacity_{model}.json")
        fitted = gen.fit_capacity_model(model, points)
        assert fitted.rps(256, 128) > fitted.rps(2048, 128)
        assert fitted.rps(256, 128) > fitted.rps(256, 448)


def test_frozen_experiment_priors_are_not_the_campaign_priors() -> None:
    # traceset-v2 must stay byte-unchanged; the campaign keeps its own dated copy.
    frozen = ROOT / "replayer" / "traces_v2" / "capacity" / "capacity_dsqwen-14b.json"
    campaign = CALIB / "capacity" / "capacity_dsqwen-14b.json"
    assert json.loads(frozen.read_text()) != json.loads(campaign.read_text())


def test_index_covers_every_model_shape_and_primitive() -> None:
    expected = len(gen.MODELS) * (len(gen.SHAPES) + 1) * len(gen.PRIMITIVES)
    assert len(INDEX["schedules"]) == expected
    combos = {(m["model"], m["shape"], m["primitive"]) for m in INDEX["schedules"]}
    assert len(combos) == expected


def test_every_committed_schedule_loads_and_matches_its_index_entry() -> None:
    for meta in INDEX["schedules"]:
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


def test_only_the_mixture_is_held_out() -> None:
    held = {m["shape"] for m in INDEX["schedules"] if m["held_out"]}
    assert held == {gen.MIXTURE_NAME}


def test_bursts_offer_more_than_capacity_at_their_peak() -> None:
    # If the spike does not exceed C_s the primitive cannot create a transient at all.
    for meta in INDEX["schedules"]:
        if meta["primitive"] != "bursts":
            continue
        assert meta["peak_offered_rps"] > meta["capacity_prior_rps"] * gen.BURST_MULTIPLIER

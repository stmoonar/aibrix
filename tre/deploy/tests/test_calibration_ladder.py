"""The preregistered calibration design (docs/preregistration-20260923-calibration-run2.md).

What these pin: the ladder is the preregistered table, laid out interleaved and
randomised; every cell - replicates included - is its own sample (id, arrivals, prompts,
files); stage 0 finishes for every shape before the ladder starts and starts from the
priors; cells are separated by a bounded drain whose failure marks the next cell; the
priors and regime groups refuse to start when absent or malformed; the run manifest is
written once, read-only, with every preregistered parameter; and the standard dataset
carries role, replicate, contamination and the warm-up mark.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from tre_replayer.engine.prompt_store import prompt_specs
from tre_replayer.engine.prompts import MODE_TOKEN_IDS, build_prompt
from tre_replayer.engine.schedule import build_poisson_schedule
from tre_replayer.traces.loader import load_trace_segments

from scripts import adaptive_boundary as boundary
from scripts import calibration_campaign as campaign
from scripts import calibration_dataset as dataset
from scripts import calibration_design as design
from scripts import calibration_ladder as ladder
from scripts import gen_calibration_schedules as gen
from scripts import openloop, r3_grid
from scripts.analysis import calibration_decision as decision

TRE_ROOT = Path(__file__).resolve().parents[2]
PREREG = TRE_ROOT / design.PREREGISTRATION_DOC
MODEL = "dsqwen-7b"

#: A priors file in the layout the loader documents: S1 and S4 never violated in the
#: first round, so they carry a widened search range instead of a boundary.
PRIORS_DOC = {
    "models": {
        model: {
            **{shape: {"capacity_rps": 5.0, "rho_star": 1.2, "boundary_found": True}
               for shape in gen.ALL_SHAPES},
            "S1": {"capacity_rps": 16.6, "boundary_found": False,
                   "search_start": 1.8, "search_max": 2.6},
            "S4": {"capacity_rps": 6.4, "boundary_found": False,
                   "search": {"start": 1.8, "max": 2.4}},
        }
        for model in gen.MODELS
    }
}
GROUPS_DOC = {"groups": {"decode": ["S4", "S5"], "prefill": ["S3", "T8"],
                         "middle": ["S1", "S2", "T9"]}}


def _write(path: Path, doc) -> Path:
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _priors(tmp_path, doc=PRIORS_DOC, models=(MODEL,)) -> design.RhoPriors:
    return design.load_rho_priors(_write(tmp_path / "rho_priors.json", doc), models=models)


def _plan(tmp_path, seed=20260923, model=MODEL):
    factory = design.CellFactory(model, seed)
    return factory, design.build_static_plan(model, _priors(tmp_path, models=(model,)), factory)


# ------------------------------------------------------------------ the ladder itself


def test_the_ladder_is_the_preregistered_table() -> None:
    items = design.ladder_items()
    assert len(items) == 12
    counts = {}
    for factor, _rep, seconds in items:
        counts[factor] = counts.get(factor, 0) + 1
        assert seconds == (240.0 if factor in (0.95, 1.00, 1.05) else 150.0)
    assert counts == {0.70: 1, 0.85: 2, 0.95: 2, 1.00: 2, 1.05: 2, 1.15: 2, 1.30: 1}
    assert design.WARMUP_S == 60.0 and design.DRAIN_LIMIT_S == 90.0
    assert (design.RAMP_RHO_START, design.RAMP_RHO_END, design.RAMP_SECONDS) == (0.6, 1.4, 360.0)
    assert design.SUPPLEMENT_BAND == (0.85, 1.15)
    assert (design.SUPPLEMENT_MIN_CELLS, design.SUPPLEMENT_CELLS) == (4, 2)
    assert design.MIN_WINDOW_REQUESTS == 10
    assert design.W_P_GRID == (0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)


def test_the_ladder_interleaves_shapes_and_randomises_each_shape_s_rho_order(tmp_path) -> None:
    _factory, plan = _plan(tmp_path)
    assert len(plan.ladder_rounds) == 12
    for rnd in plan.ladder_rounds:
        assert sorted(c.shape for c in rnd) == sorted(gen.ALL_SHAPES)  # one cell per shape
    ladder_cells = plan.ladder
    assert len(ladder_cells) == 96
    for a, b in zip(ladder_cells, ladder_cells[1:]):
        assert a.shape != b.shape  # never the same shape twice in a row
    orders = {}
    for cell in ladder_cells:
        orders.setdefault(cell.shape, []).append(cell.rho_factor)
    for shape, factors in orders.items():
        assert sorted(factors) == sorted(f for f, _r, _s in design.ladder_items())
    # random, not sorted, for (at least) most shapes
    assert sum(factors != sorted(factors) for factors in orders.values()) >= 6
    # reproducible from the seed, different under another seed
    _f2, again = _plan(tmp_path)
    assert [(c.shape, c.rho_factor, c.replicate) for c in again.ladder] == \
        [(c.shape, c.rho_factor, c.replicate) for c in ladder_cells]
    _f3, other = _plan(tmp_path, seed=7)
    assert [(c.shape, c.rho_factor) for c in other.ladder] != \
        [(c.shape, c.rho_factor) for c in ladder_cells]


# ---------------------------------------------------------------- cell independence


def _events(cell: design.DesignCell, tmp_path: Path, *, anchor=1.0, capacity=5.0,
            seed=None, key="use-cell"):
    design.anchor_cells([cell], {cell.shape: anchor})
    body, _meta = design.cell_schedule(cell, capacity, anchor_rho=anchor)
    path = tmp_path / f"{cell.cell_id}.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    segments = load_trace_segments(path)
    events = build_poisson_schedule(segments, seed=cell.arrival_seed if seed is None else seed)
    return openloop.namespace_request_ids(
        events, cell.prompt_key if key == "use-cell" else key)


def test_two_replicates_of_one_shape_and_rho_are_two_samples(tmp_path) -> None:
    _factory, plan = _plan(tmp_path)
    a, b = [c for c in plan.ladder if c.shape == "S2" and c.rho_factor == 1.00]
    # identity: different ids, both parseable by the re-window, different files
    assert a.cell_id != b.cell_id
    for cell in (a, b):
        r3_grid.GridCell.from_scenario_id(cell.cell_id)
    assert a.stem(1) != b.stem(1) and a.schedule_stem != b.schedule_stem
    # arrivals: different seeds, different instants
    assert a.arrival_seed != b.arrival_seed
    ea, eb = _events(a, tmp_path), _events(b, tmp_path)
    assert [e.scheduled_offset_s for e in ea[:50]] != [e.scheduled_offset_s for e in eb[:50]]
    # prompts: disjoint seed keys, disjoint prompts
    keys_a = {s.seed_key for s in prompt_specs(ea)}
    keys_b = {s.seed_key for s in prompt_specs(eb)}
    assert keys_a and keys_b and not keys_a & keys_b
    prompts_a = {tuple(build_prompt(s.token_count, s.seed_key, mode=MODE_TOKEN_IDS))
                 for s in prompt_specs(ea)}
    prompts_b = {tuple(build_prompt(s.token_count, s.seed_key, mode=MODE_TOKEN_IDS))
                 for s in prompt_specs(eb)}
    assert not prompts_a & prompts_b


def test_without_a_per_cell_key_two_schedules_send_the_same_prompts(tmp_path) -> None:
    # The defect the prompt key exists for: the replayer seeds prompt k from
    # "<model>|<model>-<k>", so the same shape at the same seed sends the same bytes.
    _factory, plan = _plan(tmp_path)
    a, b = [c for c in plan.ladder if c.shape == "S2" and c.rho_factor == 1.00]
    ea = _events(a, tmp_path, seed=1234, key=None)
    eb = _events(b, tmp_path, seed=1234, key=None)
    assert [e.scheduled_offset_s for e in ea] == [e.scheduled_offset_s for e in eb]
    assert {s.seed_key for s in prompt_specs(ea)} == {s.seed_key for s in prompt_specs(eb)}


def test_every_cell_of_the_run_has_its_own_id_seed_key_and_files(tmp_path) -> None:
    ids, seeds, keys, stems = set(), set(), set(), set()
    for model in gen.MODELS:
        factory = design.CellFactory(model, 20260923)
        plan = design.build_static_plan(
            model, _priors(tmp_path, models=(model,)), factory)
        extra = [factory.new("S1", design.ROLE_BOUNDARY, 150.0, rho=1.2, stage="coarse")
                 for _ in range(10)]
        for cell in plan.all_cells() + extra:
            ids.add(cell.cell_id)
            seeds.add(cell.arrival_seed)
            keys.add(cell.prompt_key)
            stems.add(cell.stem(1))
    total = 3 * (107 + 10)
    assert len(ids) == len(seeds) == len(keys) == len(stems) == total


def test_a_ladder_cell_at_a_probe_s_rho_never_shares_its_capture(tmp_path) -> None:
    # The first round's dwell and a probe at the same rho got one cell id and one raw
    # directory; here load and identity are unrelated.
    factory = design.CellFactory(MODEL, 1)
    probe = factory.new("S3", design.ROLE_BOUNDARY, 150.0, rho=1.0, stage="coarse")
    rung = factory.new("S3", design.ROLE_LADDER, 240.0, rho_factor=1.0)
    design.anchor_cells([rung], {"S3": 1.0})
    assert probe.rho == rung.rho
    assert probe.cell_id != rung.cell_id
    assert probe.stem(1) != rung.stem(1)
    assert probe.arrival_seed != rung.arrival_seed and probe.prompt_key != rung.prompt_key


def test_cell_codes_stay_clear_of_every_earlier_campaign_s_codes() -> None:
    codes = [design.cell_code(m, s) for m in gen.MODELS for s in (1, 99_999)]
    assert min(codes) > 2000  # hold probes of the first round were 1001-1999
    with pytest.raises(ValueError):
        design.cell_code("not-a-model", 1)


# ------------------------------------------------------------------------- priors


def test_priors_load_in_either_layout(tmp_path) -> None:
    priors = _priors(tmp_path)
    s1 = priors.get(MODEL, "S1")
    assert not s1.boundary_found and s1.anchor_rho == 1.8 and s1.search_max == 2.6
    assert s1.search_step == design.STEP_NOT_FOUND
    s4 = priors.get(MODEL, "S4")
    assert (s4.search_start, s4.search_max) == (1.8, 2.4)
    s2 = priors.get(MODEL, "S2")
    assert s2.boundary_found and s2.anchor_rho == 1.2
    assert s2.search_max == pytest.approx(1.2 * design.FOUND_SEARCH_SPAN)

    records = [{"model": MODEL, "shape": s, "C_s": 3.0, "empirical_boundary": 1.1}
               for s in gen.ALL_SHAPES]
    listed = design.load_rho_priors(_write(tmp_path / "list.json", records), models=[MODEL])
    assert listed.get(MODEL, "T9").rho_star == 1.1 and listed.get(MODEL, "T9").capacity_rps == 3.0


def test_priors_load_in_the_layout_the_first_round_analysis_writes(tmp_path) -> None:
    # scripts.analysis.calibration_priors: C_s_rps, boundary_rho, and a search block
    # whose upper_rho is a ceiling for a shape that never violated but only the first
    # bracket for one that did.
    def entry(found, b=None, start=None, lower=None, upper=None):
        return {"C_s_rps": 4.0, "boundary_found": found, "boundary_rho": b,
                "max_healthy_rho": lower, "windows": 80, "violating_windows": 9,
                "search": {"start_rho": start, "lower_rho": lower, "upper_rho": upper,
                           "expected_flip_rho": None, "rule": "..."}}
    doc = {"provenance": {}, "label": "client", "models": {MODEL: {
        **{s: entry(False, start=1.6, lower=1.45, upper=6.0) for s in gen.ALL_SHAPES},
        "S3": entry(True, b=0.9, start=0.9, lower=0.77, upper=1.04),
    }}}
    priors = _priors(tmp_path, doc)
    s3 = priors.get(MODEL, "S3")
    assert s3.boundary_found and s3.rho_star == 0.9 and s3.capacity_rps == 4.0
    assert s3.search_max == pytest.approx(0.9 * design.FOUND_SEARCH_SPAN)
    s2 = priors.get(MODEL, "S2")
    assert not s2.boundary_found and (s2.search_start, s2.search_max) == (1.6, 6.0)


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["models"][MODEL].pop("T9"), "no entry for dsqwen-7b/T9"),
    (lambda d: d["models"][MODEL]["S1"].pop("search_max"), "widened upward search"),
    (lambda d: d["models"][MODEL]["S2"].update(capacity_rps=-1), "capacity"),
    (lambda d: d["models"][MODEL]["S2"].update(boundary_found="yes"), "true/false"),
    (lambda d: d["models"][MODEL]["S3"].update(rho_star=None), "no rho*"),
])
def test_a_malformed_prior_refuses_to_start(tmp_path, mutate, message) -> None:
    doc = json.loads(json.dumps(PRIORS_DOC))
    mutate(doc)
    with pytest.raises(SystemExit, match=message):
        _priors(tmp_path, doc)


def test_a_missing_or_unreadable_priors_file_refuses_to_start(tmp_path) -> None:
    with pytest.raises(SystemExit, match="does not exist"):
        design.load_rho_priors(tmp_path / "absent.json", models=[MODEL])
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="not valid JSON"):
        design.load_rho_priors(tmp_path / "bad.json", models=[MODEL])


def test_regime_groups_are_validated_against_section_5_1(tmp_path) -> None:
    groups = design.load_regime_groups(_write(tmp_path / "g.json", GROUPS_DOC), models=[MODEL])
    assert groups.by_model[MODEL]["middle"] == ("S1", "S2", "T9")
    per_model = {"models": {MODEL: GROUPS_DOC}}
    assert design.load_regime_groups(
        _write(tmp_path / "pm.json", per_model), models=[MODEL]).by_model[MODEL]
    bad = [
        ({"groups": {"a": ["S4", "S5", "M"], "b": ["S3", "T8"], "c": ["S1", "S2", "T9"]}},
         "held out"),
        ({"groups": {"a": ["S4", "S5"], "b": ["S3", "T8", "S1", "S2", "T9"]}}, "exactly 3"),
        ({"groups": {"a": ["S4"], "b": ["S3", "T8", "S5"], "c": ["S1", "S2", "T9"]}}, ">= 2"),
        ({"groups": {"a": ["S4", "S5"], "b": ["S3", "T8"], "c": ["S1", "S2", "S5"]}},
         "more than one group"),
        ({"groups": {"a": ["S4", "S5"], "b": ["S3", "T8"], "c": ["S1", "S2"]}}, "missing"),
        ({"something": 1}, "expected"),
    ]
    for i, (doc, message) in enumerate(bad):
        with pytest.raises(SystemExit, match=message):
            design.load_regime_groups(_write(tmp_path / f"bad{i}.json", doc), models=[MODEL])


# ---------------------------------------------------------------- stage 0 (search)


def _drive_search(search, boundary_rho, *, log=None, inconclusive=()):
    while True:
        probe = search.next_probe()
        if probe is None:
            return search
        if log is not None:
            log.append((probe.stage, probe.rho, probe.attempt, probe.duration_s))
        if probe.rho in inconclusive and probe.attempt == 1:
            verdict = boundary.VERDICT_INCONCLUSIVE
        else:
            verdict = (boundary.VERDICT_VIOLATED if probe.rho >= boundary_rho
                       else boundary.VERDICT_HEALTHY)
        search.record(boundary.ProbeResult(probe=probe, verdict=verdict))


def test_the_search_starts_at_the_prior_brackets_then_bisects(tmp_path) -> None:
    prior = _priors(tmp_path).get(MODEL, "S2")  # found, rho* 1.2
    log = []
    search = _drive_search(design.PriorGuidedSearch.from_prior(prior), 1.3, log=log)
    assert log[0][:2] == ("coarse", 1.2)
    assert log[1][:2] == ("coarse", pytest.approx(1.38))
    assert [s for s, *_ in log[2:]] == ["bisect", "bisect"]
    assert search.boundary_found and search.healthy_rho < 1.3 <= search.violating_rho
    assert search.anchor_rho == pytest.approx((search.healthy_rho + search.violating_rho) / 2)
    assert all(d == design.BRACKET_SECONDS for *_x, d in log)


def test_a_shape_that_never_violated_is_searched_up_to_the_prior_s_ceiling(tmp_path) -> None:
    prior = _priors(tmp_path).get(MODEL, "S1")  # not found: 1.8 -> 2.6
    log = []
    search = _drive_search(design.PriorGuidedSearch.from_prior(prior), 99.0, log=log)
    rhos = [r for _s, r, *_ in log]
    assert rhos[0] == 1.8 and rhos[-1] == 2.6 and rhos == sorted(rhos)
    assert not search.boundary_found and search.anchor_rho == 2.6
    assert "ceiling" in search.exhausted
    assert "highest driven" in search.anchor_rule
    # ... and when it does flip inside the widened range, it is found
    found = _drive_search(design.PriorGuidedSearch.from_prior(prior), 2.3)
    assert found.boundary_found and found.healthy_rho < 2.3 <= found.violating_rho


def test_a_prior_that_already_violates_steps_down(tmp_path) -> None:
    prior = _priors(tmp_path).get(MODEL, "T8")
    log = []
    search = _drive_search(design.PriorGuidedSearch.from_prior(prior), 0.9, log=log)
    assert log[0][1] == 1.2 and log[1][1] == pytest.approx(1.2 / 1.15)
    assert search.healthy_rho < 0.9 <= search.violating_rho


def test_an_inconclusive_probe_is_re_driven_longer_then_stops(tmp_path) -> None:
    prior = _priors(tmp_path).get(MODEL, "S2")
    log = []
    _drive_search(design.PriorGuidedSearch.from_prior(prior), 1.3, log=log, inconclusive=(1.2,))
    assert log[0] == ("coarse", 1.2, 1, 150.0) and log[1] == ("coarse", 1.2, 2, 300.0)
    stuck = design.PriorGuidedSearch.from_prior(prior)
    for _ in range(2):
        probe = stuck.next_probe()
        stuck.record(boundary.ProbeResult(probe=probe, verdict=boundary.VERDICT_INCONCLUSIVE))
    assert stuck.next_probe() is None and "inconclusive" in stuck.stopped_reason
    assert stuck.anchor_rho is None


def _windows(start_ms, seconds, *, tpot=10.0, ttft=100.0, step_ms=5000):
    rows, w = [], start_ms
    while w + 30000 <= start_ms + int(seconds * 1000):
        rows.append({"window_start_ms": w, "window_end_ms": w + 30000,
                     "p95_ttft_client_ms": ttft, "p95_tpot_client_ms": tpot,
                     "completed_requests": 50})
        w += step_ms
    return rows


def test_a_hold_cell_is_labelled_after_its_warm_up_and_a_backlog_stop_is_violated() -> None:
    start = 1_790_000_000_000
    healthy_then_bad = _windows(start, 60) + _windows(start + 60_000, 90, tpot=200.0)
    guard = {"start_ms": start, "void_reasons": []}
    verdict = design.hold_cell_verdict(healthy_then_bad, guard, warmup_s=60.0)
    assert verdict["verdict"] == boundary.VERDICT_VIOLATED  # the healthy warm-up is ignored
    assert verdict["independent_windows"] == 3
    short = design.hold_cell_verdict(_windows(start, 70), {
        "start_ms": start, "void_reasons": [], "truncated": True,
        "truncation_cause": openloop.TRUNCATION_BACKLOG}, warmup_s=60.0)
    assert short["verdict"] == boundary.VERDICT_VIOLATED and short["backlog_stopped"]
    void = design.hold_cell_verdict([], {"void_reasons": ["envoy pending overflow"]},
                                    warmup_s=60.0)
    assert void["verdict"] == boundary.VERDICT_VOID


# ---------------------------------------------------------------- stage 3 / sentinels


def test_supplementary_cells_follow_measured_labels_not_rho() -> None:
    V, H = boundary.VERDICT_VIOLATED, boundary.VERDICT_HEALTHY
    factors = [f for f, _r, _s in design.ladder_items()]
    # S1: the boundary is far above rho* - every band cell is healthy by measurement.
    s1 = [design.CellOutcome("S1", f, H if f < 1.30 else V) for f in factors]
    # S2: split cleanly at 1.0 - four and more on each side of the band.
    s2 = [design.CellOutcome("S2", f, V if f >= 1.0 else H) for f in factors]
    plan = {d["shape"]: d for d in design.supplement_plan({"S1": s1, "S2": s2})}
    assert plan["S2"]["supplement"] == []
    assert plan["S1"]["band_violated_cells"] == 0 and plan["S1"]["band_healthy_cells"] == 10
    b = plan["S1"]["empirical_boundary_factor"]
    assert 1.15 < b < 1.30
    assert plan["S1"]["supplement"] == [pytest.approx(b * 1.04), pytest.approx(b * 1.08)]


def test_sentinel_drift_is_judged_against_the_first_sentinel() -> None:
    start = 1_790_000_000_000
    guard = {"start_ms": start, "void_reasons": []}
    same = design.sentinel_summary(_windows(start, 240), guard, warmup_s=60.0)
    slower = design.sentinel_summary(_windows(start, 240, tpot=13.0), guard, warmup_s=60.0)
    assert same["measured"] and same["median_p95_tpot_client_ms"] == 10.0
    assert design.sentinel_drift([same, same, same])["flagged"] is False
    drift = design.sentinel_drift([same, same, slower])
    assert drift["flagged"] is True and drift["checks"][1]["median_p95_tpot_relative"] == 0.3
    unmeasured = design.sentinel_summary([], {"void_reasons": ["x"]}, warmup_s=60.0)
    assert design.sentinel_drift([same, unmeasured, same])["flagged"] is None


# --------------------------------------------------------------------------- drain


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += float(s)


def test_the_drain_waits_for_an_idle_engine() -> None:
    clock = _Clock()
    seq = iter([{"running": 3, "waiting": 2}, {"running": 1, "waiting": 0},
                {"running": 0, "waiting": 0}])

    def sample():
        clock.t += 0.1
        return {**next(seq), "pods_scraped": 1, "scrape_errors": 0}

    out = ladder.wait_for_drain(sample, clock=clock, sleep=clock.sleep)
    assert out["drained"] and out["polls"] == 3 and out["waited_s"] < 10


def test_a_drain_that_never_finishes_stops_at_the_limit_and_says_so() -> None:
    clock = _Clock()
    out = ladder.wait_for_drain(
        lambda: {"running": 0, "waiting": 7, "pods_scraped": 1, "scrape_errors": 0},
        clock=clock, sleep=clock.sleep)
    assert not out["drained"] and out["waited_s"] >= design.DRAIN_LIMIT_S
    assert out["last"]["waiting"] == 7
    # a pod that did not answer is not an idle pod, nor is a sampler that raised
    clock = _Clock()
    silent = ladder.wait_for_drain(
        lambda: {"running": 0, "waiting": 0, "pods_scraped": 0, "scrape_errors": 1},
        clock=clock, sleep=clock.sleep)
    assert not silent["drained"]

    def broken():
        raise RuntimeError("no routable pods")

    clock = _Clock()
    assert not ladder.wait_for_drain(broken, clock=clock, sleep=clock.sleep)["drained"]


# ------------------------------------------------------------------- the whole run


def _args(tmp_path, **over):
    base = dict(
        models=MODEL, out_dir=tmp_path / "out", raw_dir=tmp_path / "out" / "raw",
        window_ms=30000, fit_step_ms=5000, instant_sample_ms=1000,
        ttft_slo_ms=500.0, tpot_slo_ms=75.0,
        rho_priors=_write(tmp_path / "rho_priors.json", PRIORS_DOC),
        regime_groups=_write(tmp_path / "regime_groups.json", GROUPS_DOC),
        index=None, cap=None, design_seed=20260923, cooldown_s=45.0, dry_run=False,
        preregistration=PREREG, stop_on_failure=True, controller_namespace="tre-v2",
        model_namespace="default", registry=None, redis_url=None,
        gateway_url="http://gw/v1/completions", guard_mode="warn", min_slo_windows=3,
        max_model_error_rate=0.05, envoy_stats_url=None, envoy_cluster_filter=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


#: Where each shape really flips, in rho units of its prior capacity. S1 never does
#: within its widened range; S4 does, inside it.
TRUE_BOUNDARY = {**{s: 1.25 for s in gen.ALL_SHAPES}, "S1": 99.0, "S4": 2.1}


class _FakeCluster:
    def __init__(self, clock, stuck_gaps=(4,)):
        self.clock = clock
        self.stuck_gaps = set(stuck_gaps)
        self.gaps = 0
        self.driven = []
        self.t_ms = 1_790_000_000_000

    def sampler(self, _model):
        state = {"polls_in_gap": 0}

        def sample():
            gap = self.gaps
            waiting = 5 if gap in self.stuck_gaps else 0
            return {"running": 0, "waiting": waiting, "pods_scraped": 1, "scrape_errors": 0}

        return sample

    def drive(self, cell, attempt, schedule_path, output, prompt_dir):
        self.gaps += 1
        raw = output.parent / "raw" / output.stem
        raw.mkdir(parents=True, exist_ok=True)
        capture = raw / f"{cell.cell_id}.jsonl"
        assert not capture.exists(), f"{capture} would receive a second capture"
        capture.write_text("", encoding="utf-8")
        assert prompt_dir.name == output.stem
        self.driven.append((cell.role, cell.shape, cell.cell_id, attempt, cell.rho))
        start = self.t_ms
        self.t_ms += int(cell.duration_s * 1000) + 60_000
        if cell.profile == design.PROFILE_RAMP:
            rows = _windows(start, cell.duration_s)
        else:
            bad = cell.rho >= TRUE_BOUNDARY[cell.shape]
            rows = _windows(start, cell.duration_s, tpot=200.0 if bad else 10.0)
        guard = {"start_ms": start, "end_ms": start + int(cell.duration_s * 1000),
                 "void_reasons": [], "sent": 10, "completed": 10}
        return rows, guard, 0


def _run(tmp_path, **over):
    clock = _Clock()
    fake = _FakeCluster(clock, **over)
    code = ladder.run_ladder_campaign(
        _args(tmp_path), drive=fake.drive, sample_factory=fake.sampler,
        sleep=clock.sleep, clock=clock, check_controller=False)
    return code, fake


def _ledger(tmp_path):
    lines = (tmp_path / "out" / ladder.LEDGER).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_the_run_follows_the_preregistered_order(tmp_path) -> None:
    code, fake = _run(tmp_path)
    assert code == 0
    roles = [r for r, *_ in fake.driven]
    assert roles[0] == "sentinel" and roles[-1] == "sentinel"
    first_ladder = roles.index("ladder")
    assert set(roles[1:first_ladder]) == {"boundary"}  # every search done before the ladder
    assert "boundary" not in roles[first_ladder:]
    middle = roles.index("sentinel", 1)
    assert roles[first_ladder:middle].count("ladder") == 48  # half of the rounds
    last_ladder = max(i for i, r in enumerate(roles) if r == "ladder")
    ramps = [i for i, r in enumerate(roles) if r == "ramp"]
    assert len(ramps) == 8 and min(ramps) > last_ladder
    supplement = [i for i, r in enumerate(roles) if r == "adaptive"]
    assert supplement and min(supplement) > max(ramps)
    assert roles.count("ladder") == 96 and roles.count("sentinel") == 3
    # every shape searched, and stage 0 interleaves them rather than finishing one first
    boundary_shapes = [s for r, s, *_ in fake.driven if r == "boundary"]
    assert set(boundary_shapes) == set(gen.ALL_SHAPES)
    assert len(set(boundary_shapes[:8])) == 8  # round-robin: every shape's first probe first


def test_the_run_anchors_the_ladder_on_what_stage_0_found(tmp_path) -> None:
    _run(tmp_path)
    result = json.loads((tmp_path / "out" / ladder.DESIGN_RESULT).read_text())["models"][0]
    assert result["status"] == "complete"
    assert result["boundary_found"]["S1"] is False and result["anchors"]["S1"] == 2.6
    assert result["boundary_found"]["S4"] is True
    assert 1.8 < result["anchors"]["S4"] < 2.4
    assert 1.1 < result["anchors"]["S2"] < 1.4
    records = _ledger(tmp_path)
    s2_rungs = [r for r in records if r["role"] == "ladder" and r["shape"] == "S2"]
    for r in s2_rungs:
        assert r["rho"] == pytest.approx(r["rho_factor"] * result["anchors"]["S2"], abs=1e-6)
    searched = json.loads((tmp_path / "out" / "boundary" / f"{MODEL}_S1.json").read_text())
    assert searched["boundary_found"] is False and "highest driven" in searched["anchor_source"]
    # S1's ladder all measured healthy (its boundary is out of reach): it gets supplements
    decisions = {d["shape"]: d for d in result["supplement"]}
    assert decisions["S1"]["supplement"]


def test_every_attempt_of_the_run_is_its_own_capture(tmp_path) -> None:
    _run(tmp_path)
    records = _ledger(tmp_path)
    assert len({r["stem"] for r in records}) == len(records)
    assert len({(r["cell_id"], r["attempt"]) for r in records}) == len(records)
    cells = {r["cell_id"]: r for r in records}
    assert len({r["arrival_seed"] for r in cells.values()}) == len(cells)
    assert len({r["prompt_key"] for r in cells.values()}) == len(cells)
    assert {r["split"] for r in records if r["role"] == "ladder" and r["shape"] != "M"} == {"train"}
    assert {r["split"] for r in records if r["shape"] == "M" and r["role"] == "ladder"} == {"holdout"}
    assert {r["split"] for r in records if r["role"] == "ramp"} == {"holdout"}
    assert {r["split"] for r in records
            if r["role"] in ("boundary", "sentinel") and r["shape"] != "M"} == {"auxiliary"}
    # "M shape 的全部 cell" (§5.3): M's probes are held out too
    assert {r["split"] for r in records if r["shape"] == "M"} == {"holdout"}


def test_every_stage_the_run_writes_is_one_the_analysis_partitions_on(tmp_path) -> None:
    # One vocabulary on both sides: the analysis raises on a hold stage it does not
    # know, so a renamed stage here would stop it rather than quietly empty a split.
    policy = decision.PREREGISTERED
    assert policy.train_hold_stages == design.TRAINING_HOLD_STAGES
    assert design.EXCLUDED_HOLD_STAGES <= policy.excluded_hold_stages
    assert policy.hold_warmup_s == design.WARMUP_S
    _run(tmp_path)
    expected = {design.SPLIT_TRAIN: decision.ROLE_TRAIN,
                design.SPLIT_HOLDOUT: decision.ROLE_HOLDOUT,
                design.SPLIT_AUXILIARY: decision.ROLE_EXCLUDED}
    seen = set()
    for r in _ledger(tmp_path):
        row = {"shape": r["shape"], "primitive": r["primitive"], "stage": r["stage"],
               "cell_id": r["cell_id"], "model": r["model"]}
        assert policy.role(row) == expected[r["split"]], r
        seen.add((r["role"], r["primitive"], r["stage"]))
    assert seen == {
        ("sentinel", "hold", "sentinel"), ("boundary", "hold", "coarse"),
        ("boundary", "hold", "bisect"), ("ladder", "hold", "ladder"),
        ("ramp", "ramp", ""), ("adaptive", "hold", "adaptive"),
    }


def test_an_engine_that_did_not_drain_marks_the_next_cell(tmp_path) -> None:
    _run(tmp_path, stuck_gaps=(4,))
    records = _ledger(tmp_path)
    flagged = [r for r in records if r["possibly_contaminated"]]
    assert len(flagged) == 1 and flagged[0] is records[4]
    assert flagged[0]["drain_before"]["drained"] is False
    assert flagged[0]["drain_before"]["waited_s"] >= design.DRAIN_LIMIT_S
    assert "not drained" in flagged[0]["contamination_reason"]
    ok = records[3]["drain_before"]
    assert ok["drained"] and ok["gap_s"] == pytest.approx(45.0)  # the cooldown is the floor
    result = json.loads((tmp_path / "out" / ladder.DESIGN_RESULT).read_text())["models"][0]
    assert [c["cell_id"] for c in result["possibly_contaminated_cells"]] == [flagged[0]["cell_id"]]


def test_the_run_manifest_freezes_every_preregistered_parameter(tmp_path) -> None:
    _run(tmp_path)
    path = tmp_path / "out" / ladder.RUN_MANIFEST
    assert not os.stat(path).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    pre = manifest["preregistered"]
    assert pre["w_p_grid"] == list(design.W_P_GRID)
    assert pre["lambda_wait"] == {"primary": 3.0, "sensitivity": [3.0, 10.0]}
    assert pre["decision_rule"]["loro_min_gain_ba_points"] == 3.0
    assert pre["decision_rule"]["bootstrap_interval"] == 0.90
    assert pre["decision_rule"]["pooled_holdout_paired_difference_min"] == 0.0
    assert pre["label"]["min_window_requests"] == 10
    assert pre["sentinel"]["drift_thresholds"] == design.SENTINEL_DRIFT_THRESHOLDS
    assert manifest["design_seed"] == 20260923
    assert manifest["regime_groups"]["groups"][MODEL]["decode"] == ["S4", "S5"]
    assert manifest["rho_priors"]["parsed"][f"{MODEL}/S1"]["search_max"] == 2.6
    assert len(manifest["preregistration"]["commit"]) == 40
    plan = json.loads((tmp_path / "out" / "plan.json").read_text())
    assert plan["run_manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="already exists"):
        ladder.write_frozen(path, {})


def test_the_run_refuses_to_start_without_its_inputs(tmp_path) -> None:
    with pytest.raises(SystemExit, match="--rho-priors"):
        ladder.run_ladder_campaign(_args(tmp_path, rho_priors=None), check_controller=False)
    with pytest.raises(SystemExit, match="does not exist"):
        ladder.run_ladder_campaign(_args(tmp_path, regime_groups=tmp_path / "nope.json"),
                                   check_controller=False)
    with pytest.raises(SystemExit, match="--tpot-slo-ms"):
        ladder.run_ladder_campaign(_args(tmp_path, tpot_slo_ms=100.0), check_controller=False)
    with pytest.raises(SystemExit, match="preregistration"):
        ladder.run_ladder_campaign(_args(tmp_path, preregistration=tmp_path / "x.md"),
                                   check_controller=False)
    assert not (tmp_path / "out" / ladder.RUN_MANIFEST).exists()


def test_the_dry_run_reports_each_model_s_duration(tmp_path, capsys) -> None:
    args = [
        "--models", MODEL, "--out-dir", str(tmp_path / "out"), "--dry-run",
        "--rho-priors", str(_write(tmp_path / "p.json", PRIORS_DOC)),
        "--regime-groups", str(_write(tmp_path / "g.json", GROUPS_DOC)),
        "--index", str(tmp_path / "absent-index.json"),
    ]
    assert campaign.main(args) == 0
    out = capsys.readouterr().out
    assert f"{MODEL}: expected" in out and "ladder" in out and "boundary" in out
    plan = json.loads((tmp_path / "out" / "plan.json").read_text())
    est = plan["estimate"][MODEL]
    assert est["stages"]["ladder"]["cells_expected"] == 96
    assert est["stages"]["sentinel"]["cells_expected"] == 3
    assert est["seconds_upper"] > est["seconds_expected"] > 8 * 3600
    assert not (tmp_path / "out" / ladder.RUN_MANIFEST).exists()
    with pytest.raises(SystemExit, match="--rho-priors"):
        campaign.main([a for a in args if a not in ("--rho-priors", str(tmp_path / "p.json"))])


# ------------------------------------------------------------------- the driver side


def test_design_cells_are_driven_with_their_own_seed_key_prompts_and_valve(tmp_path) -> None:
    factory = design.CellFactory(MODEL, 5)
    probe = factory.new("S2", design.ROLE_BOUNDARY, 150.0, rho=1.2, stage="bisect")
    rung = factory.new("S2", design.ROLE_LADDER, 150.0, rho_factor=1.0)
    args = _args(tmp_path)
    for cell in (probe, rung):
        grid = campaign.Cell(MODEL, cell.shape, cell.primitive, cell.cell_id, "s.json", 150.0, 0.0)
        command = ladder.design_cell_command(grid, cell, args, Path("s.json"),
                                             Path("o.csv"), Path("/p") / cell.stem(1))
        assert command[command.index("--schedule-seed") + 1] == str(cell.arrival_seed)
        assert command[command.index("--prompt-key") + 1] == cell.prompt_key
        assert command[command.index("--prompt-dir") + 1] == str(Path("/p") / cell.stem(1))
        assert command[command.index("--cell-id") + 1] == cell.cell_id
        assert command[command.index("--shed-policy") + 1] == openloop.SHED_POLICY_VOID
        assert ("--max-backlog" in command) == (cell.role == design.ROLE_BOUNDARY)
    parsed = r3_grid.parse_args(["--model", MODEL, "--gateway-url", "g", "--output", "o.csv",
                                 "--schedule", "s.json", "--prompt-key", "k",
                                 "--max-backlog", "7"])
    assert parsed.prompt_key == "k" and parsed.max_backlog == 7


def test_the_backlog_valve_stops_sending_and_says_why() -> None:
    release = None

    class Sender:
        records: list = []

        async def __call__(self, request, scheduled_ts, actual_ts):
            await release.wait()

    class Req:
        def __init__(self, k):
            self.scheduled_offset_s = float(k)

    async def main():
        nonlocal release
        release = asyncio.Event()
        valve = openloop.StopOnBacklog(Sender(), max_backlog=3)
        tasks = [asyncio.create_task(valve(Req(k), 0.0, 0.0)) for k in range(6)]
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(*tasks)
        return valve

    valve = asyncio.run(main())
    assert valve.truncated and valve.censored == 3 and valve.peak_outstanding == 3
    assert valve.truncated_at_offset_s == 3.0
    guard = openloop.check_cell("i1_o1_c1", scheduled=6, records=[{"http_status": 200, "e2e_ms": 50.0}] * 3,
                                p99_delay_ms=1.0, truncated=True, censored=3,
                                truncation_cause=openloop.TRUNCATION_BACKLOG,
                                backlog_limit=3)
    body = guard.as_dict()
    assert body["truncation_cause"] == openloop.TRUNCATION_BACKLOG and body["backlog_limit"] == 3
    assert not guard.voided


# ------------------------------------------------------------------ the dataset side


def _capture(raw_dir: Path, cell_id: str, *, start: int, seconds: int, tpot: float) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    senders = [{
        "request_id": f"k-{i:06d}", "actual_send_ts_ms": start + 300 + 250 * i,
        "on_wire_delay_ms": 2.0, "ttft_ms": 150.0, "e2e_ms": 150.0 + tpot * 127,
        "prompt_tokens": 768, "completion_tokens": 128, "http_status": 200, "error": None,
        "error_body": None, "error_headers": None, "target_pod": None,
        "client_timeout": False, "request_timeout_s": 32.0, "in_flight_at_send": 4,
    } for i in range(seconds * 4)]
    raw = [openloop._raw_from_sender_record(cell_id, r) for r in senders]
    openloop._append_jsonl(raw_dir / f"{cell_id}.jsonl", raw)
    openloop._append_jsonl(raw_dir / f"{cell_id}.instant.jsonl", openloop.mark_live_grid([
        {"ts_ms": start + 1000 * s, "waiting": 0.0, "running": 3.0, "swapping": 0.0}
        for s in range(seconds + 1)
    ]))
    guard = {"cell_id": cell_id, "start_ms": start, "end_ms": start + seconds * 1000 + 300,
             "truncated_at_ts_ms": None, "void_reasons": [], "sent": len(senders),
             "ttft_slo_ms": 500.0, "tpot_slo_ms": 75.0}
    (raw_dir / f"{cell_id}.guard.json").write_text(json.dumps(guard), encoding="utf-8")


def test_the_dataset_reads_a_ladder_run_from_its_ledger_and_marks_the_warm_up(tmp_path) -> None:
    out = tmp_path / "run" / MODEL
    factory = design.CellFactory(MODEL, 3)
    rung = factory.new("S2", design.ROLE_LADDER, 150.0, rho_factor=1.05, replicate=2, round=4)
    planned = factory.new("S2", design.ROLE_LADDER, 150.0, rho_factor=0.85)
    design.anchor_cells([rung, planned], {"S2": 1.2})
    start = 1_790_000_000_000
    out.mkdir(parents=True)
    (out / "plan.json").write_text(json.dumps({
        "design": "ladder", "models": [MODEL],
        "static_cells": {MODEL: [rung.as_dict(), planned.as_dict()]},
        "provenance": {"window_ms": 30000, "step_ms": 5000},
    }), encoding="utf-8")
    _capture(out / "raw" / rung.stem(1), rung.cell_id, start=start, seconds=150, tpot=20.0)
    (out / "raw" / "stray_dir").mkdir()
    record = {**rung.as_dict(), "attempt": 1, "stem": rung.stem(1), "verdict": "healthy",
              "void_reasons": [], "capacity_rps": 5.0, "offered_rps": 6.3,
              "possibly_contaminated": True,
              "drain_before": {"drained": False, "waited_s": 90.0}}
    (out / "cells.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    built = dataset.build_dataset(tmp_path / "run")
    with (built / "windows.csv").open(newline="", encoding="utf-8") as fh:
        windows = list(csv.DictReader(fh))
    assert windows and {w["role"] for w in windows} == {"ladder"}
    assert {w["replicate"] for w in windows} == {"2"} and {w["rho_factor"] for w in windows} == {"1.05"}
    assert {w["possibly_contaminated"] for w in windows} == {"True"}
    for w in windows:
        expect = int(w["window_start_ms"]) < start + 60_000
        assert w["in_warmup"] == str(expect)
        # the analysis drops exactly these (calibration_decision.build_cells)
        dropped = (float(w["window_start_ms"]) - start
                   < decision.PREREGISTERED.hold_warmup_s * 1000.0)
        assert w["in_warmup"] == str(dropped)
    assert any(w["in_warmup"] == "True" for w in windows)
    assert any(w["in_warmup"] == "False" for w in windows)
    with (built / "cells.csv").open(newline="", encoding="utf-8") as fh:
        cells = {c["cell_id"]: c for c in csv.DictReader(fh)}
    assert cells[rung.cell_id]["status"] == "valid" and cells[rung.cell_id]["warmup_s"] == "60.0"
    assert cells[rung.cell_id]["drained_before"] == "False"
    assert cells[planned.cell_id]["status"] == "missing"
    with (built / "requests.csv").open(newline="", encoding="utf-8") as fh:
        requests = list(csv.DictReader(fh))
    assert {r["in_warmup"] for r in requests} == {"True", "False"}
    manifest = json.loads((built / "manifest.json").read_text())
    assert any("stray_dir" in d and "not in cells.jsonl" in d for d in manifest["discrepancies"])
    assert any("planned, never driven" in d for d in manifest["discrepancies"])

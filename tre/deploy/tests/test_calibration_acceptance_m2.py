"""The next round's M2 (``calibration_acceptance --composition m2-20261005``).

Invariants: the composition (24 cells: 16 heterogeneous-spike bursts, 2 ramps, 2 steps,
three deep holds and one 0.9 x rho* hold; every spike its own seed, all three heights in
every burst cell, room to drain between spikes); every cell code and seed new (unique, and
in no earlier ledger); rho* reused from M's sealed boundary records, recorded with its
source paths and hashes, and refused when the source differs from the design constants; a
complete run seals an M that ``dline_refit accept`` reads as it is.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from tre_common import slo_labels

from scripts import calibration_acceptance as ac
from scripts import calibration_design as design
from scripts import dline_refit as dl
from scripts import gen_calibration_schedules as gen

MODEL = "dsqwen-7b"
SEED = ac.M2_DESIGN_SEED


def _write(path: Path, doc) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _m_root(root: Path, model: str = MODEL, override=None) -> Path:
    """A sealed M run root with the rho* M2 reuses (``override``: shape -> anchor_rho)."""
    d = root / "M" / model
    files = []
    for shape, (anchor, cs) in ac.M2_RHO_STAR[model].items():
        files.append(_write(d / "boundary" / f"{model}_{shape}.json", {
            "anchor_rho": (override or {}).get(shape, anchor), "capacity_rps": cs,
            "rho_star_status": "measured",
            "anchor_rule": "midpoint of the final (healthy, violated) bracket"}))
    _write(d / ac.M_MANIFEST, {"rho_star": {s: {"anchor_rho": a, "capacity_rps": c, "status": "measured"}
                                            for s, (a, c) in ac.M2_RHO_STAR[model].items()}})
    (d / ac.M_SHA256SUMS).write_text("".join(
        f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.resolve()}\n" for f in files), encoding="utf-8")
    return root / "M"


def _windows(start_ms, seconds, *, ttft):
    rows, w = [], start_ms
    while w + 30000 <= start_ms + int(seconds * 1000):
        rows.append({"window_start_ms": w, "window_end_ms": w + 30000, "p95_ttft_client_ms": ttft,
                     "p95_tpot_client_ms": 20.0, "avg_running": 5.0, "completed_requests": 50,
                     "ttft_len_samples": slo_labels.format_ttft_len_samples([(ttft, 512)] * 50)})
        w += 10000
    return rows


class _Fake:
    def __init__(self):
        self.t, self.t_ms, self.driven = 0.0, 1_790_000_000_000, []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += float(s)

    def sample(self):
        return {"running": 0, "waiting": 0, "pods_scraped": 1, "scrape_errors": 0}

    def drive(self, cell, attempt, schedule_path, output, prompt_dir):
        body = json.loads(Path(schedule_path).read_text())
        self.driven.append((cell, body))
        raw = Path(output).parent / "raw" / cell.stem(attempt)
        raw.mkdir(parents=True, exist_ok=True)
        (raw / f"{cell.cell_id}.jsonl").write_text(f"{cell.cell_id}\n", encoding="utf-8")
        start = self.t_ms
        self.t_ms += int(cell.duration_s * 1000) + 60_000
        rows = _windows(start, cell.duration_s, ttft=100.0)
        guard = {"start_ms": start, "end_ms": start + int(cell.duration_s * 1000),
                 "void_reasons": [], "sent": 10, "completed": 10}
        return rows, guard, 0


def _args(tmp_path, **over):
    base = dict(
        models=MODEL, out_dir=tmp_path / "out", raw_dir=tmp_path / "out" / "raw",
        window_ms=30000, fit_step_ms=10000, instant_sample_ms=1000, ttft_slo_ms=500.0,
        tpot_slo_ms=75.0, index=None, cap=None, design_seed=SEED, cooldown_s=45.0,
        dry_run=False, stop_on_failure=True, controller_namespace="tre-v2",
        model_namespace="default", registry=None, redis_url=None,
        gateway_url="http://gw/v1/completions", guard_mode="warn", min_slo_windows=3,
        max_model_error_rate=0.05, envoy_stats_url=None, envoy_cluster_filter=None,
        freeze_file=tmp_path / "freeze.json", composition=ac.COMPOSITION_M2,
        rho_star_run=_m_root(tmp_path), ledger_root=[],
        corpus_lang="en", zh_ratio=0.0, routing_strategy=None, api="completions")
    base.update(over)
    return argparse.Namespace(**base)


def _freeze(tmp_path, monkeypatch, label) -> Path:
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps({"models": {MODEL: {"verdict_for_holdout": {"label_def": label.as_dict()}}},
                                "freeze_sha256": "cd" * 32}), encoding="utf-8")
    monkeypatch.setattr(dl, "verify_freeze", lambda p: json.loads(Path(p).read_text()))
    return path


# ------------------------------------------------------------------- composition


def test_m2_composition_kinds_and_heterogeneous_spikes() -> None:
    items = ac.M2_COMPOSITION
    assert len(items) == 24
    assert Counter(i.kind for i in items) == {"bursts": 16, "ramp": 2, "steps": 2, "hold": 4}
    assert sorted(i.factor for i in items if i.kind == "hold") == [0.9, 1.15, 1.25, 1.4]
    assert Counter(i.shape for i in items if i.kind == "bursts") == {s: 4 for s in ac.M2_SHAPES}
    cap = gen.get_cap(gen.DEFAULT_CAP_NAME)
    drain_s = ac.M2_SPIKE_EXCESS_S / (1.0 - ac.M2_BURST_BASE_RHO)  # backlog served at rho* only
    heights = Counter()
    for model, table in ac.M2_RHO_STAR.items():
        cells, spikes = ac.m2_cells(model, SEED)
        assert len(cells) == 24 and len(spikes) == 16
        assert all(c.split == design.SPLIT_HOLDOUT and c.role == design.ROLE_ACCEPTANCE for c in cells)
        anchors = {s: a for s, (a, _c) in table.items()}
        capacity = {s: c for s, (_a, c) in table.items()}
        ac.place(cells, anchors, capacity, cap, {}, items=items, spikes=spikes)  # checks durations
        for cell in cells:
            if cell.profile != design.PROFILE_BURSTS:
                continue
            plan = spikes[cell.cell_id]
            assert len(plan) == ac.M2_SPIKES and {s["height_x_rho_star"] for s in plan} == {2.0, 3.0, 4.0}
            heights.update(s["height_x_rho_star"] for s in plan)
            body, _meta = design.cell_schedule(cell, capacity[cell.shape], anchor_rho=anchors[cell.shape])
            (segs,) = body.values()
            rate = lambda t: sum(s["rps"] for s in segs if s["start_time"] <= t < s["end_time"])  # noqa: E731
            unit_rps = anchors[cell.shape] * capacity[cell.shape]
            assert rate(30.0) == pytest.approx(ac.M2_BURST_BASE_RHO * unit_rps, rel=1e-3)
            ends = []
            for s in plan:
                mid = s["start_s"] + s["width_s"] / 2
                assert rate(mid) == pytest.approx(s["height_x_rho_star"] * unit_rps, rel=1e-3)
                assert (s["height_x_rho_star"] - 1.0) * s["width_s"] == pytest.approx(ac.M2_SPIKE_EXCESS_S, abs=0.01)
                ends.append(s["start_s"] + s["width_s"])
            # each spike drains (at rho* alone) and leaves a healthy 30 s window before the next
            starts = [s["start_s"] for s in plan]
            assert all(nxt - end >= drain_s + 30.0 for end, nxt in zip(ends, starts[1:]))
            assert cell.duration_s - ends[-1] >= drain_s + 30.0
    # 3 models x 16 cells x (all three heights + one rotating) -> balanced within one cell
    assert max(heights.values()) - min(heights.values()) <= 3


# -------------------------------------------------------------------- ids, seeds


def test_m2_ids_and_seeds_are_new(tmp_path) -> None:
    per_model = {m: ac.m2_cells(m, SEED) for m in ac.M2_RHO_STAR}
    cells = {m: v[0] for m, v in per_model.items()}
    spikes = {m: v[1] for m, v in per_model.items()}
    serials = [c.serial for v in cells.values() for c in v]
    assert all(ac.M2_CELL_SERIAL_BASE < s < ac.M2_CELL_SERIAL_BASE + 100 for s in serials)
    # an earlier ledger (M's own ids) shares nothing
    old = tmp_path / "ledgers" / "M" / MODEL / "cells.jsonl"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"cell_id": "i0_o0_c1070501", "arrival_seed": 380691728}) + "\n")
    report = ac.check_new_ids(cells, spikes, [tmp_path / "ledgers"])
    assert report["problems"] == [] and report["ledgers_scanned"] == 1
    assert report["m2_seeds"] == 3 * (24 + 16 * 4)  # arrival seeds + one per spike, all distinct
    # a ledger that already used an M2 cell id, or an M2 spike seed as an arrival seed
    victim = cells[MODEL][3]
    spike_seed = spikes["dsllama-8b"][next(iter(spikes["dsllama-8b"]))][2]["seed"]
    bad = tmp_path / "ledgers" / "other" / "cells.csv"
    bad.parent.mkdir(parents=True)
    bad.write_text(f"cell_id,arrival_seed\n{victim.cell_id},1\ni0_o0_c60,{spike_seed}\n")
    problems = ac.check_new_ids(cells, spikes, [tmp_path / "ledgers"])["problems"]
    assert any(str(victim.code) in p for p in problems)
    assert any(str(spike_seed) in p for p in problems)
    # the run's own out-dir is not an earlier ledger
    assert ac.check_new_ids(cells, spikes, [tmp_path / "ledgers"],
                            skip=[tmp_path / "ledgers" / "other"])["problems"] == []


# --------------------------------------------------------- rho* source and the seal


@pytest.mark.parametrize("attribution", ["completion", "hybrid"])
def test_m2_run_records_the_rho_star_source_and_seals(tmp_path, monkeypatch, attribution) -> None:
    if attribution == "hybrid" and "attribution" not in getattr(slo_labels.LabelDefinition,
                                                                "__dataclass_fields__", {}):
        pytest.skip("label v2 (hybrid attribution) is not in this tree")
    args = _args(tmp_path)
    args.fit_label_attribution = attribution
    frozen_label = ac.fit_label(args, MODEL)
    del args.fit_label_attribution  # M2 takes it from the freeze
    freeze = _freeze(tmp_path, monkeypatch, frozen_label)
    fake = _Fake()
    code = ac.run_acceptance_set(args, drive=fake.drive, sample_factory=lambda _m: fake.sample,
                                 sleep=fake.sleep, clock=fake.clock, check_controller=False)
    assert code == 0
    assert len(fake.driven) == 24 and {c.role for c, _b in fake.driven} == {design.ROLE_ACCEPTANCE}
    man_path = tmp_path / "out" / ac.M_MANIFEST
    man = json.loads(man_path.read_text())
    assert man["composition_name"] == ac.COMPOSITION_M2 == man["composition"]["name"]
    assert man["label_attribution"] == attribution
    assert man["label_def"] == frozen_label.as_dict()
    plan = json.loads((tmp_path / "out" / "plan.json").read_text())
    assert plan["label"]["label_def"] == frozen_label.as_dict()  # what finalize_run reads
    assert len(man["cells"]) == 24 and all(c["origin"] == "collected" for c in man["cells"])
    assert "retained_source" not in man and man["sealed_probes"] == []
    src = man["rho_star_source"]
    for shape, (anchor, cs) in ac.M2_RHO_STAR[MODEL].items():
        assert src["shapes"][shape]["anchor_rho"] == anchor and src["shapes"][shape]["capacity_rps"] == cs
        assert man["rho_star"][shape]["anchor_rho"] == anchor
    for s in src["sources"]:
        assert s["sha256"] and hashlib.sha256(Path(s["path"]).read_bytes()).hexdigest() == s["sha256"]
    assert len(src["sources"]) == 6  # the M manifest, its M_SHA256SUMS, four boundary records
    assert set(man["seeds"]) == {c["cell_id"] for c in man["cells"]}
    assert man["design_seed"] == SEED
    _man, problems = dl.check_m_manifest(man_path, json.loads(freeze.read_text()),
                                         hashlib.sha256(freeze.read_bytes()).hexdigest())
    assert problems == []
    # a rho* source that is not M's sealed record is refused before anything runs
    boundary = args.rho_star_run / MODEL / "boundary" / f"{MODEL}_MP.json"
    boundary.write_text(json.dumps({**json.loads(boundary.read_text()), "pad": 1}))
    with pytest.raises(ValueError, match="digest M sealed"):
        ac.m2_rho_star(MODEL, args.rho_star_run)
    with pytest.raises(ValueError, match="design constant"):
        ac.m2_rho_star(MODEL, _m_root(tmp_path / "other", override={"MD": 1.2}))
    unsealed = _m_root(tmp_path / "unsealed")
    (unsealed / MODEL / ac.M_SHA256SUMS).write_text("")
    with pytest.raises(ValueError, match="sealed M run"):
        ac.m2_rho_star(MODEL, unsealed)


def test_m2_three_models_in_sequence_under_one_root_and_a_restart_with_an_offset(tmp_path) -> None:
    """Review 2026-10-05 P1-1: the three models launch one after another into one root (each
    run's ledger holds only its own codes and is a sibling, not an earlier ledger); an
    aborted model's re-run into a new root collides with its own old ledger unless it takes a
    new --m2-serial-offset, and then passes."""
    root = tmp_path / "next-20261005" / "M2"

    def launch(model: str, out: Path, offset: int = 0) -> dict:
        cells, spikes, ids = ac.m2_id_check(model, SEED, offset, [tmp_path], out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "plan.json").write_text(json.dumps({"composition_name": ac.COMPOSITION_M2,
                                                   "design_seed": SEED, "models": [model]}))
        (out / "cells.jsonl").write_text("".join(
            json.dumps({"cell_id": c.cell_id, "arrival_seed": c.arrival_seed}) + "\n" for c in cells))
        return ids

    for model in ac.M2_RHO_STAR:
        assert launch(model, root / model)["problems"] == []
    # dsqwen-14b aborted; its re-run into a new root, same codes -> refused ...
    again = ac.m2_id_check("dsqwen-14b", SEED, 0, [tmp_path], root.parent / "M2-rerun" / "dsqwen-14b")[2]
    assert any("cell codes already in a ledger" in p for p in again["problems"])
    # ... with the next offset: new codes, new seeds -> passes, and the offset is recorded
    ids = launch("dsqwen-14b", root.parent / "M2-rerun" / "dsqwen-14b", offset=ac.M2_SERIAL_OFFSET_STEP)
    assert ids["problems"] == [] and ids["serial_offset"] == ac.M2_SERIAL_OFFSET_STEP
    assert ids["cell_serial_base"] == ac.M2_CELL_SERIAL_BASE + ac.M2_SERIAL_OFFSET_STEP
    with pytest.raises(ValueError):
        ac.m2_serial_base(50)

"""Step ④ of plan 2026-09-21 §6.11 - the acceptance set M (``scripts.calibration_acceptance``).

What these pin: M's new shapes are held out and in no default shape list; the ten
collected cells are the fixed composition with the first round's primitives on the
measured rho* (steps 450 s, bursts 360 s, ramp 435 s, holds 300 / 600 s); the new shapes'
probes start at a boundary prior fitted through the training shapes' D6' boundaries and are
judged like every probe; a real run refuses without a verifying freeze file or under a
label other than the frozen one (D22); a rho* that is not measured stops the run before
any M cell; a complete run seals M - M_SHA256SUMS over every file and the retained cells'
raw captures, M_manifest.json (read-only) with the 13 evaluated cells (3 retained, marked
seen before), the probes, the freeze hash and the label hash - and every ledger line is in
the holdout split, which the D16 training cut never trains on.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from tre_common import slo_labels

from scripts import adaptive_boundary as boundary
from scripts import calibration_acceptance as ac
from scripts import calibration_campaign as campaign
from scripts import calibration_design as design
from scripts import calibration_ladder as ladder
from scripts import calibration_supplement as supplement
from scripts import dline_refit as dl
from scripts import gen_calibration_schedules as gen

MODEL = "dsqwen-7b"
A, B = 1.0e-4, 1.5e-4           # the synthetic boundary cost per token (s)
C_S = 2.0                       # every training shape's run-2 C_s
FLIP = 1.1                      # every new shape violates from 1.1 x its predicted boundary


def _write(path: Path, doc) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _rate(shape):
    return 1.0 / sum(w * (A * gen._length_mean(i) + B * gen._length_mean(o))
                     for w, i, o in gen.shape_components(shape))


def _inputs(root: Path) -> dict:
    shapes = ac.PRIOR_SHAPES
    d = root / "base" / MODEL
    _write(d / ladder.DESIGN_RESULT, {"models": [{
        "model": MODEL, "status": "complete", "anchors": {s: 1.0 for s in shapes},
        "anchor_sources": {s: "midpoint" for s in shapes}}]})
    for s in shapes:
        _write(d / "boundary" / f"{MODEL}_{s}.json", {"anchor_rho": 1.0, "prior": {"capacity_rps": C_S}})
    _write(d / ladder.RUN_MANIFEST, {"rho_priors": {"path": "/x", "sha256": "ab" * 32, "parsed": {
        f"{MODEL}/{s}": {"capacity_rps": C_S} for s in shapes}}})
    _write(root / "supp" / MODEL / "boundary" / f"{MODEL}_S3.json", {
        "mode": supplement.MODE, "rho_star_status": boundary.RHO_STAR_MEASURED,
        "anchor_rho": _rate("S3") / C_S, "base": {"capacity_rps": C_S}})
    table = root / "table.csv"
    with open(table, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "shape", "rho_star_fixed", "P_b50_rf"])
        for s in shapes:
            w.writerow([MODEL, s, 1.0, "" if s == "S3" else _rate(s) / C_S])
    run1 = root / "run1"
    ds = run1 / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    _write(ds / "manifest.json", {"run_root": str(run1)})
    (ds / "windows.csv").write_text("model\n", encoding="utf-8")
    with open(ds / "cells.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "shape", "primitive", "cell_id", "attempt",
                                           "split", "status", "windows", "raw_path",
                                           "guard_path", "role"])
        w.writeheader()
        for prim, code in (("bursts", 60), ("ramp", 120), ("steps", 95)):
            raw = f"{MODEL}/raw/{MODEL}_M_{prim}/i0_o0_c{code}.jsonl"
            guard = f"{MODEL}/raw/{MODEL}_M_{prim}/i0_o0_c{code}.guard.json"
            for rel in (raw, guard):
                (run1 / rel).parent.mkdir(parents=True, exist_ok=True)
                (run1 / rel).write_text(f"{prim}\n", encoding="utf-8")
            w.writerow({"model": MODEL, "shape": "M", "primitive": prim, "cell_id": f"i0_o0_c{code}",
                        "attempt": 1, "split": "holdout", "status": "valid", "windows": 40,
                        "raw_path": raw, "guard_path": guard, "role": ""})
    return {"base_run": root / "base", "boundary_supplement_run": root / "supp",
            "boundary_table": table, "retained_dataset": ds}


def _args(tmp_path, **over):
    base = dict(
        models=MODEL, out_dir=tmp_path / "out", raw_dir=tmp_path / "out" / "raw",
        window_ms=30000, fit_step_ms=10000, instant_sample_ms=1000,
        ttft_slo_ms=500.0, tpot_slo_ms=75.0, index=None, cap=None, design_seed=20260923,
        cooldown_s=45.0, dry_run=False, stop_on_failure=True, controller_namespace="tre-v2",
        model_namespace="default", registry=None, redis_url=None,
        gateway_url="http://gw/v1/completions", guard_mode="warn", min_slo_windows=3,
        max_model_error_rate=0.05, envoy_stats_url=None, envoy_cluster_filter=None,
        freeze_file=tmp_path / "freeze.json", **_inputs(tmp_path),
    )
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """A freeze file that verifies, frozen under the model's primary label."""
    path = tmp_path / "freeze.json"
    doc = {"models": {MODEL: {"verdict_for_holdout": {
        "label_def": dl.label_for(MODEL, "primary", None).as_dict()}}}, "freeze_sha256": "cd" * 32}
    path.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(dl, "verify_freeze", lambda p: json.loads(Path(p).read_text()))
    return path


def _windows(start_ms, seconds, *, ttft, tpot=20.0):
    rows, w = [], start_ms
    while w + 30000 <= start_ms + int(seconds * 1000):
        rows.append({"window_start_ms": w, "window_end_ms": w + 30000,
                     "p95_ttft_client_ms": ttft, "p95_tpot_client_ms": tpot,
                     "avg_running": 5.0, "completed_requests": 50,
                     "ttft_len_samples": slo_labels.format_ttft_len_samples([(ttft, 512)] * 50)})
        w += 10000
    return rows


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += float(s)


class _FakeCluster:
    def __init__(self, flip):
        self.flip = flip
        self.driven = []
        self.t_ms = 1_790_000_000_000

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
        bad = cell.rho is not None and cell.rho >= self.flip.get(cell.shape, FLIP)
        rows = _windows(start, cell.duration_s, ttft=20000.0 if bad else 100.0)
        guard = {"start_ms": start, "end_ms": start + int(cell.duration_s * 1000),
                 "void_reasons": [], "sent": 10, "completed": 10}
        return rows, guard, 0


def _run(tmp_path, flip=None, **over):
    clock = _Clock()
    fake = _FakeCluster(flip or {})
    code = ac.run_acceptance_set(_args(tmp_path, **over), drive=fake.drive,
                                 sample_factory=lambda _m: fake.sample, sleep=clock.sleep,
                                 clock=clock, check_controller=False)
    return code, fake


# ------------------------------------------------------------------------ shapes


def test_the_new_shapes_are_held_out_and_in_no_default_list() -> None:
    assert set(ac.NEW_SHAPES) == {"MP", "MD", "G800x240", "U512x512"}
    for shape in ac.NEW_SHAPES:
        assert gen.is_held_out(shape)
        assert shape not in gen.ALL_SHAPES and shape not in gen.TRAINING_SHAPES
        assert "_" not in shape
    assert gen.is_mixture("MP") and gen.is_mixture("MD") and gen.is_mixture("M")
    assert not gen.is_mixture("U512x512")
    mp = gen.shape_components("MP")
    assert [w for w, _i, _o in mp] == [0.7, 0.3]
    assert (mp[0][1].low, mp[0][1].high, mp[0][2].low, mp[0][2].high) == (2048, 3072, 64, 96)
    assert mp[1][1:] == (256, 128)
    assert [w for w, _i, _o in gen.shape_components("MD")] == [0.3, 0.7]
    f = design.CellFactory(MODEL, 1, serial_base=ac.CELL_SERIAL_BASE)
    assert f.new("MP", design.ROLE_ACCEPTANCE, 60).cell_id.startswith("i0_o0_c")
    u = f.new("U512x512", design.ROLE_ACCEPTANCE, 60)
    assert u.cell_id.startswith("i512_o512_c") and u.split == design.SPLIT_HOLDOUT
    assert f.new("G800x240", design.ROLE_BOUNDARY, 60, rho=1.0, stage="coarse").split == design.SPLIT_HOLDOUT
    # the first-round M still names its cells i0_o0 and the training shapes are unchanged
    assert f.new("M", design.ROLE_LADDER, 60).cell_id.startswith("i0_o0_c")
    assert f.new("S3", design.ROLE_LADDER, 60).cell_id.startswith("i2048_o96_c")


def test_the_primitives_on_rho_star() -> None:
    steps = ac.steps_profile(2.0)
    assert steps == [(0.0, 90.0, 1.0), (90.0, 210.0, 1.6), (210.0, 450.0, 1.9)]
    ramp = ac.ramp_profile(2.0, 300.0)
    assert ac.profile_seconds(ramp) == 435.0
    assert ramp[0][2] == pytest.approx(2.0 * (0.4 + 0.8 * 0.5 / 60))
    assert ramp[-2] == (300.0, 375.0, 2.4) and ramp[-1] == (375.0, 435.0, 1.0)
    bursts = ac.bursts_profile(2.0, 50.0)
    assert bursts[0] == (0.0, 360.0, 1.2)
    assert [(a, b) for a, b, _r in bursts[1:]] == [(60.0, 62.0), (150.0, 152.0), (240.0, 242.0), (330.0, 332.0)]
    assert [(i.shape, i.kind, i.factor, i.seconds) for i in ac.COMPOSITION] == [
        ("MP", "steps", None, 450.0), ("MP", "bursts", None, 360.0), ("MP", "hold", 1.05, 600.0),
        ("MP", "hold", 1.15, 600.0), ("MD", "steps", None, 450.0), ("MD", "ramp", None, 435.0),
        ("G800x240", "hold", 0.9, 300.0), ("G800x240", "hold", 1.05, 300.0),
        ("U512x512", "hold", 1.0, 300.0), ("U512x512", "steps", None, 450.0)]


def test_the_boundary_prior_recovers_the_rate_model(tmp_path) -> None:
    ins = _inputs(tmp_path)
    prior = ac.boundary_prior(MODEL, ins["base_run"], ins["boundary_supplement_run"], ins["boundary_table"])
    assert prior["a_s_per_token"] == pytest.approx(A) and prior["b_s_per_token"] == pytest.approx(B)
    assert all(abs(x["loo_error"]) < 1e-6 for x in prior["leave_one_out"])
    for shape in ac.NEW_SHAPES:
        assert prior["predicted_rps"][shape] == pytest.approx(_rate(shape), rel=1e-5)
    # a mixture meets its boundary at the load-share harmonic rate of its streams
    single = {k: 1.0 / (A * gen._length_mean(i) + B * gen._length_mean(o))
              for k, (w, i, o) in enumerate(gen.shape_components("MP"))}
    assert _rate("MP") == pytest.approx(1.0 / (0.7 / single[0] + 0.3 / single[1]))


# ----------------------------------------------------------------------------- run


def test_a_real_run_needs_the_freeze_and_the_frozen_label(tmp_path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="D22"):
        ac.run_acceptance_set(_args(tmp_path), check_controller=False)
    path = tmp_path / "freeze.json"
    other = dl.label_for(MODEL, "fixed", None).as_dict()
    path.write_text(json.dumps({"models": {MODEL: {"verdict_for_holdout": {"label_def": other}}}}))
    monkeypatch.setattr(dl, "verify_freeze", lambda p: json.loads(Path(p).read_text()))
    with pytest.raises(ValueError, match="another label"):
        ac.run_acceptance_set(_args(tmp_path), check_controller=False)


def test_a_rho_star_that_is_not_measured_stops_before_any_m_cell(tmp_path, frozen, capsys) -> None:
    code, fake = _run(tmp_path, flip={"G800x240": 99.0})
    assert code == ac.EXIT_CHECK_FAILED
    assert {c.role for c, _b in fake.driven} == {design.ROLE_BOUNDARY}
    assert "G800x240: rho* is lower_bound" in capsys.readouterr().out
    assert not (tmp_path / "out" / ac.M_MANIFEST).exists()


def test_a_complete_run_places_the_cells_on_rho_star_and_seals_m(tmp_path, frozen) -> None:
    code, fake = _run(tmp_path)
    assert code == 0
    probes = [c for c, _b in fake.driven if c.role == design.ROLE_BOUNDARY]
    cells = [(c, b) for c, b in fake.driven if c.role == design.ROLE_ACCEPTANCE]
    assert {c.shape for c in probes} == set(ac.NEW_SHAPES)
    assert probes[0].rho == ac.PROBE_START_RHO
    assert len(cells) == 10
    result = json.loads((tmp_path / "out" / ladder.DESIGN_RESULT).read_text())["models"][0]
    anchors = {s: v["anchor_rho"] for s, v in result["rho_star"].items()}
    assert all(1.0 < a < FLIP for a in anchors.values())
    for cell, body in cells:
        (segs,) = body.values()
        cs = _rate(cell.shape)
        total = lambda t: sum(s["rps"] for s in segs if s["start_time"] <= t < s["end_time"])  # noqa: E731
        if cell.profile == design.PROFILE_HOLD:
            assert cell.rho == pytest.approx(cell.rho_factor * anchors[cell.shape])
            assert total(100) == pytest.approx(cell.rho * cs, rel=1e-3)
            assert cell.warmup_s == design.WARMUP_S
        else:
            assert cell.warmup_s == 0.0 and cell.primitive == cell.profile
        assert max(s["end_time"] for s in segs) == cell.duration_s
        if cell.profile == design.PROFILE_STEPS:
            assert total(300) == pytest.approx(0.95 * anchors[cell.shape] * cs, rel=1e-3)
    records = [json.loads(x) for x in (tmp_path / "out" / ladder.LEDGER).read_text().splitlines()]
    assert {r["split"] for r in records} == {design.SPLIT_HOLDOUT}
    for r in records:
        row = {k: r[k] for k in ("model", "cell_id", "attempt", "shape", "primitive", "role",
                                 "split", "stage")}
        assert dl.assign_set(row, sealed_to_h2=False, sentinels=True) == dl.SET_M
    # the seal
    man_path = tmp_path / "out" / ac.M_MANIFEST
    man = json.loads(man_path.read_text())
    assert not os.access(man_path, os.W_OK) or os.geteuid() == 0
    assert oct(man_path.stat().st_mode & 0o777) == "0o444"
    assert man["model"] == MODEL and man["format_revision"] == 1
    assert man["freeze"]["sha256"] == hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert man["label_def_sha256"] == dl.canonical_sha256(man["label_def"])
    assert man["label_def"] == dl.label_for(MODEL, "primary", None).as_dict()
    evaluated = man["cells"]
    assert len(evaluated) == 13
    retained = [c for c in evaluated if c["origin"] == "retained"]
    assert len(retained) == 3 and all(c["seen_before"] and "2026-09-22" in c["note"] for c in retained)
    assert {c["primitive"] for c in retained} == {"steps", "ramp", "bursts"}
    collected = [c for c in evaluated if c["origin"] == "collected"]
    assert all(not c["seen_before"] and c["raw_files"] for c in collected)
    assert {c["cell_id"] for c in man["sealed_probes"]} == {c.cell_id for c in probes}
    sums = (tmp_path / "out" / man["sha256sums_file"]).read_text()
    assert hashlib.sha256(sums.encode()).hexdigest() == man["sha256sums_sha256"]
    listed = {}
    for line in sums.splitlines():
        digest, path = line.split("  ", 1)
        listed[path] = digest
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    for c in [*collected, *retained]:
        assert all(str(Path(f).resolve()) in listed for f in c["raw_files"])
    assert str((tmp_path / "out" / ladder.LEDGER).resolve()) in listed
    # ... and dline_refit accept takes this manifest under this freeze as it is
    freeze_doc = json.loads(frozen.read_text())
    _man, problems = dl.check_m_manifest(man_path, freeze_doc,
                                         hashlib.sha256(frozen.read_bytes()).hexdigest())
    assert problems == []
    # a raw file changed after the seal is caught there
    victim = Path(collected[0]["raw_files"][0])
    victim.write_text("tampered", encoding="utf-8")
    _man, problems = dl.check_m_manifest(man_path, freeze_doc,
                                         hashlib.sha256(frozen.read_bytes()).hexdigest())
    assert any("changed after it was sealed" in x for x in problems)


def test_the_cli_dry_run_lists_m_without_a_freeze(tmp_path, capsys) -> None:
    ins = _inputs(tmp_path)
    out = tmp_path / "dry"
    argv = ["--acceptance-set", "--models", MODEL, "--base-run", str(ins["base_run"]),
            "--boundary-supplement-run", str(ins["boundary_supplement_run"]),
            "--boundary-table", str(ins["boundary_table"]),
            "--retained-dataset", str(ins["retained_dataset"]),
            "--freeze-file", str(tmp_path / "absent.json"),
            "--out-dir", str(out), "--raw-dir", str(out / "raw"), "--dry-run",
            "--index", str(tmp_path / "absent-index.json")]
    assert campaign.main(argv) == 0
    text = capsys.readouterr().out
    assert "refuses to start without one (D22)" in text
    assert "retained" in text and "MP bursts" in text and "U512x512" in text
    plan = json.loads((out / "plan.json").read_text())
    assert len(plan["static_cells"][MODEL]) == 10 and len(plan["retained"]["cells"]) == 3
    assert plan["composition"]["evaluated_cells"] == 13
    assert not (out / ladder.RUN_MANIFEST).exists() and not (out / ac.M_MANIFEST).exists()

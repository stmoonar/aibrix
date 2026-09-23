"""Step ③ of plan 2026-09-21 §6.11 - the training supplement
(``scripts.calibration_training_supplement``).

What these pin: the cells per model (③a S3 ladder on the supplement's D6' rho*, 14b also
at 0.75 x; ③b 14b's twelve T8 cells on 1.006 x rho*_run2; the 7b T8 top-up; three S2
sentinels at 0.72 x rho*_run2 first / middle / last); every cell is a 300 s hold (240 s
sentinel) with a run-unique id above the supplement's, its own seeds, a ledger line the
D16 training cut classifies as a constant-load training row; the holds are interleaved
by shape in a seeded order; the run refuses a supplement rho* that is not measured or sits
on another C_s, and a non-D6' label; it ends with exit 3 and a banner when a ladder shape
has no violated (or no healthy) cell or the sentinels drift; the CLI dry run writes only
the plan.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tre_common import slo_labels

from scripts import adaptive_boundary as boundary
from scripts import calibration_campaign as campaign
from scripts import calibration_design as design
from scripts import calibration_ladder as ladder
from scripts import calibration_supplement as supplement
from scripts import calibration_training_supplement as ts
from scripts import dline_refit as dl

ANCHORS = {"S2": 1.4, "S3": 1.5, "T8": 0.9}      # rho*_run2 (x C_s)
CAPACITY = {"S2": 7.5, "S3": 2.0, "T8": 6.0}      # C_s (rps)
SUPP_S3 = 2.1                                     # rho*_D6'(S3) (x C_s)
LENGTH = {"S2": 768, "S3": 2048, "T8": 1600}


def _write(path: Path, doc) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _base_run(root: Path, model: str) -> Path:
    d = root / model
    _write(d / ladder.DESIGN_RESULT, {"models": [{
        "model": model, "status": "complete", "anchors": dict(ANCHORS),
        "anchor_sources": {s: "midpoint of the final (healthy, violated) bracket" for s in ANCHORS}}]})
    for shape, anchor in ANCHORS.items():
        _write(d / "boundary" / f"{model}_{shape}.json",
               {"anchor_rho": anchor, "prior": {"capacity_rps": CAPACITY[shape]}})
    _write(d / ladder.RUN_MANIFEST, {"rho_priors": {
        "path": "/x/rho_priors.json", "sha256": "ab" * 32,
        "parsed": {f"{model}/{s}": {"capacity_rps": c} for s, c in CAPACITY.items()}}})
    return root


def _supp_run(root: Path, model: str, *, status=boundary.RHO_STAR_MEASURED,
              capacity=CAPACITY["S3"]) -> Path:
    _write(root / model / "boundary" / f"{model}_S3.json", {
        "mode": supplement.MODE, "rho_star_status": status, "anchor_rho": SUPP_S3,
        "rho_star_bracket": [SUPP_S3 * 0.98, SUPP_S3 * 1.02],
        "base": {"capacity_rps": capacity, "anchor_rho": ANCHORS["S3"]}})
    return root


def _args(tmp_path, model="dsqwen-14b", **over):
    base = dict(
        models=model, out_dir=tmp_path / "out", raw_dir=tmp_path / "out" / "raw",
        window_ms=30000, fit_step_ms=10000, instant_sample_ms=1000,
        ttft_slo_ms=500.0, tpot_slo_ms=75.0, index=None, cap=None, design_seed=20260923,
        cooldown_s=45.0, dry_run=False, stop_on_failure=True, controller_namespace="tre-v2",
        model_namespace="default", registry=None, redis_url=None,
        gateway_url="http://gw/v1/completions", guard_mode="warn", min_slo_windows=3,
        max_model_error_rate=0.05, envoy_stats_url=None, envoy_cluster_filter=None,
        base_run=_base_run(tmp_path / "base", model),
        boundary_supplement_run=_supp_run(tmp_path / "supp", model),
    )
    base.update(over)
    return argparse.Namespace(**base)


def _windows(start_ms, seconds, *, ttft, length, tpot=20.0, running=6.0):
    rows, w = [], start_ms
    while w + 30000 <= start_ms + int(seconds * 1000):
        rows.append({"window_start_ms": w, "window_end_ms": w + 30000,
                     "p95_ttft_client_ms": ttft, "p95_tpot_client_ms": tpot,
                     "avg_running": running, "completed_requests": 50,
                     "ttft_len_samples": slo_labels.format_ttft_len_samples([(ttft, length)] * 50)})
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
    """A hold at rho >= flip[shape] answers at 20 s TTFT (violated under any label), below
    at 100 ms; the sentinels' TPOT / running are ``sentinel_tpot`` in turn."""

    def __init__(self, flip: dict, sentinel_tpot=(20.0, 20.0, 20.0)):
        self.flip = flip
        self.sentinel_tpot = list(sentinel_tpot)
        self.driven = []
        self.t_ms = 1_790_000_000_000

    def sample(self):
        return {"running": 0, "waiting": 0, "pods_scraped": 1, "scrape_errors": 0}

    def drive(self, cell, attempt, schedule_path, output, prompt_dir):
        body = json.loads(Path(schedule_path).read_text())
        self.driven.append((cell.role, cell.shape, cell.rho, cell.duration_s, cell.cell_id, body))
        start = self.t_ms
        self.t_ms += int(cell.duration_s * 1000) + 60_000
        tpot = 20.0
        if cell.role == design.ROLE_SENTINEL:
            tpot = self.sentinel_tpot.pop(0)
            ttft = 100.0
        else:
            ttft = 20000.0 if cell.rho >= self.flip[cell.shape] else 100.0
        rows = _windows(start, cell.duration_s, ttft=ttft, length=LENGTH[cell.shape], tpot=tpot)
        guard = {"start_ms": start, "end_ms": start + int(cell.duration_s * 1000),
                 "void_reasons": [], "sent": 10, "completed": 10}
        return rows, guard, 0


def _run(tmp_path, model="dsqwen-14b", flip=None, **kw):
    clock = _Clock()
    flip = flip or {"S3": SUPP_S3, "T8": ANCHORS["T8"] * 1.006, "S2": 99.0}
    if model == "dsqwen-7b":
        flip = {**flip, "T8": ANCHORS["T8"] * 1.13}
    fake = _FakeCluster(flip, **kw)
    code = ts.run_training_supplement(
        _args(tmp_path, model), drive=fake.drive, sample_factory=lambda _m: fake.sample,
        sleep=clock.sleep, clock=clock, check_controller=False)
    return code, fake


def _ledger(tmp_path):
    return [json.loads(x) for x in (tmp_path / "out" / ladder.LEDGER).read_text().splitlines()]


# ------------------------------------------------------------------------- the plan


def _sequence(tmp_path, model):
    args = _args(tmp_path, model)
    resolved = ts.resolve_units(model, args.base_run, args.boundary_supplement_run)
    factory = design.CellFactory(model, 20260923, serial_base=ts.SERIAL_BASE)
    return ts.build_cells(model, factory, resolved, 20260923)[0]


@pytest.mark.parametrize("model,n_holds", [("dsqwen-7b", 13), ("dsllama-8b", 10), ("dsqwen-14b", 24)])
def test_the_cells_of_each_model(tmp_path, model, n_holds) -> None:
    seq = _sequence(tmp_path, model)
    holds = [c for c in seq if c.role == design.ROLE_LADDER]
    sentinels = [c for c in seq if c.role == design.ROLE_SENTINEL]
    assert len(holds) == n_holds and len(sentinels) == 3
    s3 = sorted(c.rho_factor for c in holds if c.shape == "S3")
    ladder_factors = ts.S3_LADDER_14B if model == "dsqwen-14b" else ts.S3_LADDER
    assert s3 == sorted([f for f in ladder_factors for _ in (1, 2)])
    for c in holds:
        assert c.duration_s == ts.HOLD_SECONDS and c.warmup_s == design.WARMUP_S
        if c.shape == "S3":
            assert c.rho == pytest.approx(c.rho_factor * SUPP_S3)
    t8 = [c for c in holds if c.shape == "T8"]
    if model == "dsqwen-14b":
        assert sorted(c.rho_factor for c in t8) == sorted([f for f in ts.T8_CI_14B for _ in (1, 2)])
        assert all(c.rho == pytest.approx(c.rho_factor * 1.006 * ANCHORS["T8"]) for c in t8)
        assert 0.75 in {c.rho_factor for c in holds if c.shape == "S3"}
    elif model == "dsqwen-7b":
        assert sorted(c.rho_factor for c in t8) == [1.1, 1.15, 1.2]
        assert all(c.rho == pytest.approx(c.rho_factor * ANCHORS["T8"]) for c in t8)
    else:
        assert not t8
    # sentinels: first, middle, last; S2 at 0.72 x rho*_run2
    assert seq[0].role == seq[-1].role == design.ROLE_SENTINEL
    assert seq[len(holds) // 2 + 1].role == design.ROLE_SENTINEL
    assert [c.position for c in sentinels] == list(design.SENTINEL_POSITIONS)
    assert all(c.shape == "S2" and c.duration_s == 240.0
               and c.rho == pytest.approx(0.72 * ANCHORS["S2"]) for c in sentinels)
    # identity: run-unique ids above the supplement's, own seeds and prompt keys
    assert len({c.cell_id for c in seq}) == len(seq)
    assert all(c.serial > ts.SERIAL_BASE > supplement.SUPPLEMENT_SERIAL_BASE for c in seq)
    assert len({c.arrival_seed for c in seq}) == len(seq)
    assert len({c.prompt_key for c in seq}) == len(seq)


def test_the_holds_are_interleaved_and_the_order_is_seeded(tmp_path) -> None:
    seq = _sequence(tmp_path, "dsqwen-14b")
    shapes = [c.shape for c in seq if c.role == design.ROLE_LADDER]
    # 12 S3 and 12 T8: an interleaving never repeats a shape while the other has cells
    assert all(a != b for a, b in zip(shapes, shapes[1:]))
    again = _sequence(tmp_path / "again", "dsqwen-14b")
    assert [c.cell_id for c in seq] == [c.cell_id for c in again]
    other = ts.interleave([c for c in seq if c.role == design.ROLE_LADDER],
                          __import__("random").Random(1))
    assert sorted(c.cell_id for c in other) == sorted(c.cell_id for c in seq if c.role == design.ROLE_LADDER)


# ----------------------------------------------------------------------------- run


def test_the_run_drives_every_cell_and_its_ledger_trains(tmp_path) -> None:
    code, fake = _run(tmp_path)
    assert code == 0
    assert len(fake.driven) == 27
    records = _ledger(tmp_path)
    assert {r["role"] for r in records} == {design.ROLE_LADDER, design.ROLE_SENTINEL}
    assert {r["split"] for r in records if r["role"] == design.ROLE_LADDER} == {design.SPLIT_TRAIN}
    # every schedule is a constant-rate hold at rho x C_s of its shape
    for role, shape, rho, seconds, _cid, body in fake.driven:
        (segs,) = body.values()
        assert len(segs) == 1 and segs[0]["rps"] == pytest.approx(rho * CAPACITY[shape], rel=1e-3)
        assert segs[0]["end_time"] == seconds
    # D16: the training cut classifies every line as a constant-load training row
    for r in records:
        row = {"model": r["model"], "cell_id": r["cell_id"], "attempt": r["attempt"],
               "shape": r["shape"], "primitive": r["primitive"], "role": r["role"],
               "split": r["split"], "stage": r["stage"]}
        assert dl.assign_set(row, sealed_to_h2=False, sentinels=True) == dl.SET_TRAINING
    result = json.loads((tmp_path / "out" / ladder.DESIGN_RESULT).read_text())["models"][0]
    assert result["check_failures"] == [] and result["sentinel_drift"]["flagged"] is False
    plan = json.loads((tmp_path / "out" / "plan.json").read_text())
    assert plan["design"] == ladder.DESIGN_NAME and plan["mode"] == ts.MODE
    manifest = json.loads((tmp_path / "out" / ladder.RUN_MANIFEST).read_text())
    assert manifest["sequence"] == [r["cell_id"] for r in records]
    assert manifest["design_seed"] == 20260923


def test_a_ladder_without_a_violated_cell_fails_the_check(tmp_path, capsys) -> None:
    code, _fake = _run(tmp_path, flip={"S3": 99.0, "T8": ANCHORS["T8"] * 1.006, "S2": 99.0})
    assert code == ts.EXIT_CHECK_FAILED
    out = capsys.readouterr().out
    assert "CHECK FAILED" in out and "S3 part 3a: 12 healthy and 0 violated" in out
    status = json.loads((tmp_path / "out" / campaign.CAMPAIGN_STATUS_FILE).read_text())
    assert status["exit_code"] == ts.EXIT_CHECK_FAILED


def test_sentinel_drift_fails_the_check(tmp_path, capsys) -> None:
    code, _fake = _run(tmp_path, sentinel_tpot=(20.0, 20.0, 30.0))  # +50 % TPOT at the end
    assert code == ts.EXIT_CHECK_FAILED
    assert "sentinel drift flagged" in capsys.readouterr().out


def test_the_run_refuses_bad_inputs(tmp_path) -> None:
    with pytest.raises(ValueError, match="not measured"):
        ts.run_training_supplement(_args(tmp_path, boundary_supplement_run=_supp_run(
            tmp_path / "s1", "dsqwen-14b", status=boundary.RHO_STAR_LOWER_BOUND)), check_controller=False)
    with pytest.raises(ValueError, match="C_s"):
        ts.run_training_supplement(_args(tmp_path, boundary_supplement_run=_supp_run(
            tmp_path / "s2", "dsqwen-14b", capacity=9.0)), check_controller=False)
    with pytest.raises(ValueError, match="D6'"):
        ts.run_training_supplement(_args(tmp_path, fit_ttft_slo_mode="fixed"), check_controller=False)
    with pytest.raises(ValueError, match="one model"):
        ts.run_training_supplement(_args(tmp_path, models="dsqwen-7b,dsqwen-14b"), check_controller=False)
    with pytest.raises(SystemExit, match="--tpot-slo-ms"):
        ts.run_training_supplement(_args(tmp_path, tpot_slo_ms=100.0), check_controller=False)


def test_the_cli_dry_run_writes_only_the_plan(tmp_path, capsys) -> None:
    model = "dsqwen-7b"
    base = _base_run(tmp_path / "base", model)
    supp = _supp_run(tmp_path / "supp", model)
    out = tmp_path / "dry"
    argv = ["--training-supplement", "--models", model, "--base-run", str(base),
            "--boundary-supplement-run", str(supp), "--out-dir", str(out),
            "--raw-dir", str(out / "raw"), "--dry-run", "--index", str(tmp_path / "absent.json")]
    assert campaign.main(argv) == 0
    text = capsys.readouterr().out
    assert "16 cells" in text and "part 3a" in text and "part t8" in text and "part 3d" in text
    plan = json.loads((out / "plan.json").read_text())
    assert len(plan["static_cells"][model]) == 16 and plan["estimate"]["seconds_expected"] > 0
    assert not (out / ladder.RUN_MANIFEST).exists() and not (out / ladder.LEDGER).exists()


def test_the_cli_keeps_the_collection_flags_together(tmp_path) -> None:
    for argv in (
        ["--training-supplement", "--acceptance-set"],
        ["--training-supplement", "--reprobe-base", str(tmp_path)],
        ["--base-run", str(tmp_path)],
        ["--freeze-file", str(tmp_path / "f.json")],
    ):
        with pytest.raises(SystemExit):
            campaign.main([*argv, "--models", "dsqwen-7b", "--out-dir", str(tmp_path / "o"),
                           "--dry-run"])

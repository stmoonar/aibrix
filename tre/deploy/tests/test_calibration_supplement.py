"""The boundary supplement (plan 2026-09-21 §6.11 D19): S3 re-probed above the second
round's rho* on a per-model grid, judged with the fit's ruler, then one smoke hold.

What these pin: the unit is the base run's own anchor, read three ways that must agree;
the search walks the grid upwards from its lowest point, stops at the first violation and
bisects, reports a bound (never a number) when the grid never flips, and reaches the top
of the grid (14b: 2.8 x); every probe is judged on the post-warm-up windows under the D6'
primary label - a window the fixed 500 ms label would call violated is healthy when D6'
says so - and a fixed-label run refuses to start; the smoke hold runs only on a measured
rho*, at the bracket midpoint, and a fraction outside 20-70 % (or no smoke at all) ends
the run with exit 3 and a banner; every cell is a ladder-design ledger line with its own
id above the base run's, excluded from training, which the analysis partitions.
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
from scripts.analysis import calibration_decision as decision

MODEL = "dsqwen-7b"
SHAPE = "S3"
ANCHOR = 1.5      # rho*_base, in rho of the base run's C_s
CAPACITY = 2.0    # C_s (rps)
GRID = "1.3,1.45,1.6,1.8"
#: S3 is i2048_o96: the 7b D6' TTFT SLO there is max(500, 5 * (36.4 + 0.0527 * 2048)) ms.
S3_INPUT = 2048
D6_TTFT_SLO_MS = 5 * (36.4 + 0.0527 * S3_INPUT)


def _write(path: Path, doc) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _base_run(root: Path, *, model=MODEL, anchor=ANCHOR, capacity=CAPACITY,
              boundary_anchor=None, manifest_capacity=None, status="complete") -> Path:
    d = root / model
    _write(d / ladder.DESIGN_RESULT, {"models": [{
        "model": model, "status": status, "anchors": {SHAPE: anchor},
        "anchor_sources": {SHAPE: "midpoint of the final (healthy, violated) bracket"}}]})
    _write(d / "boundary" / f"{model}_{SHAPE}.json", {
        "anchor_rho": anchor if boundary_anchor is None else boundary_anchor,
        "prior": {"capacity_rps": capacity}})
    _write(d / ladder.RUN_MANIFEST, {"rho_priors": {
        "path": "/x/rho_priors.json", "sha256": "ab" * 32,
        "parsed": {f"{model}/{SHAPE}": {
            "capacity_rps": capacity if manifest_capacity is None else manifest_capacity}}}})
    return root


def _args(tmp_path, **over):
    base = dict(
        models="", out_dir=tmp_path / "out", raw_dir=tmp_path / "out" / "raw",
        window_ms=30000, fit_step_ms=10000, instant_sample_ms=1000,
        ttft_slo_ms=500.0, tpot_slo_ms=75.0, index=None, cap=None, design_seed=20260923,
        cooldown_s=45.0, dry_run=False, stop_on_failure=True, controller_namespace="tre-v2",
        model_namespace="default", registry=None, redis_url=None,
        gateway_url="http://gw/v1/completions", guard_mode="warn", min_slo_windows=3,
        max_model_error_rate=0.05, envoy_stats_url=None, envoy_cluster_filter=None,
        reprobe_base=_base_run(tmp_path / "base"), reprobe_grid=[f"{MODEL}:{GRID}"],
        smoke_at_rho_star=True,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _windows(start_ms, seconds, *, ttft, bad_from=None, tpot=20.0, running=6.0):
    """10 s-step 30 s windows of S3 requests (2048 prompt tokens) at ``ttft`` ms; windows
    starting at or after ``bad_from`` (ms) get 3 x the TTFT."""
    rows, w = [], start_ms
    while w + 30000 <= start_ms + int(seconds * 1000):
        t = ttft * 3 if bad_from is not None and w >= bad_from else ttft
        rows.append({"window_start_ms": w, "window_end_ms": w + 30000,
                     "p95_ttft_client_ms": t, "p95_tpot_client_ms": tpot,
                     "avg_running": running, "completed_requests": 50,
                     "ttft_len_samples": slo_labels.format_ttft_len_samples(
                         [(t, S3_INPUT)] * 50)})
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
    """S3 flips at ``flip`` (rho). Healthy cells answer at 600 ms - over the fixed 500 ms
    SLO, under D6' (~722 ms) - with a violated warm-up; violated ones at 3 x that. The
    smoke hold violates in its last ``smoke_bad`` of its post-warm-up windows."""

    def __init__(self, flip: float, smoke_bad: float = 0.4):
        self.flip = flip
        self.smoke_bad = smoke_bad
        self.driven = []
        self.t_ms = 1_790_000_000_000

    def sample(self):
        return {"running": 0, "waiting": 0, "pods_scraped": 1, "scrape_errors": 0}

    def drive(self, cell, attempt, schedule_path, output, prompt_dir):
        self.driven.append((cell.role, cell.stage, cell.rho, cell.duration_s, cell.cell_id))
        start = self.t_ms
        self.t_ms += int(cell.duration_s * 1000) + 60_000
        warm_end = start + int(design.WARMUP_S * 1000)
        if cell.role == design.ROLE_SMOKE:
            post = _windows(warm_end, cell.duration_s - design.WARMUP_S, ttft=600.0)
            cut = post[int(len(post) * (1 - self.smoke_bad))]["window_start_ms"] \
                if self.smoke_bad > 0 else None
            rows = (_windows(start, design.WARMUP_S, ttft=5000.0)
                    + _windows(warm_end, cell.duration_s - design.WARMUP_S, ttft=600.0,
                               bad_from=cut))
        else:
            ttft = 1800.0 if cell.rho >= self.flip else 600.0
            rows = (_windows(start, design.WARMUP_S, ttft=5000.0)  # the queue building up
                    + _windows(warm_end, cell.duration_s - design.WARMUP_S, ttft=ttft))
        guard = {"start_ms": start, "end_ms": start + int(cell.duration_s * 1000),
                 "void_reasons": [], "sent": 10, "completed": 10}
        return rows, guard, 0


def _run(tmp_path, flip, *, smoke_bad=0.4, **over):
    clock = _Clock()
    fake = _FakeCluster(flip, smoke_bad)
    args = _args(tmp_path, **over)
    code = supplement.run_boundary_supplement(
        args, {MODEL: [SHAPE]}, drive=fake.drive, sample_factory=lambda _m: fake.sample,
        sleep=clock.sleep, clock=clock, check_controller=False)
    return code, fake


def _ledger(tmp_path):
    lines = (tmp_path / "out" / ladder.LEDGER).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _result(tmp_path):
    return json.loads((tmp_path / "out" / ladder.DESIGN_RESULT).read_text())["models"][0]


# ---------------------------------------------------------------------- the base run


def test_the_unit_is_the_base_run_s_anchor_read_three_ways(tmp_path) -> None:
    base = supplement.load_base_anchor(_base_run(tmp_path / "b"), MODEL, SHAPE)
    assert (base.anchor_rho, base.capacity_rps) == (ANCHOR, CAPACITY)
    assert base.anchor_rps == pytest.approx(3.0)
    assert base.rho_of(1.8) == pytest.approx(2.7) and base.factor_of(2.7) == 1.8
    with pytest.raises(ValueError, match="disagrees with its boundary JSON"):
        supplement.load_base_anchor(_base_run(tmp_path / "c", boundary_anchor=1.4), MODEL, SHAPE)
    with pytest.raises(ValueError, match="capacity in the run manifest"):
        supplement.load_base_anchor(_base_run(tmp_path / "d", manifest_capacity=2.5),
                                    MODEL, SHAPE)
    with pytest.raises(ValueError, match="not complete"):
        supplement.load_base_anchor(_base_run(tmp_path / "e", status="stopped: x"), MODEL, SHAPE)
    with pytest.raises(ValueError, match="does not exist"):
        supplement.load_base_anchor(tmp_path / "nowhere", MODEL, SHAPE)


def test_the_grid_is_per_model_and_ascending() -> None:
    grids = supplement.parse_grid(["dsqwen-7b:1.3,1.45", "dsqwen-14b:1.8,2.1,2.4,2.8"])
    assert grids == {"dsqwen-7b": (1.3, 1.45), "dsqwen-14b": (1.8, 2.1, 2.4, 2.8)}
    for bad in ("dsqwen-7b", "dsqwen-7b:", "dsqwen-7b:1.5,1.3", "dsqwen-7b:1.3,1.3",
                "dsqwen-7b:0,1", "dsqwen-7b:a"):
        with pytest.raises(ValueError):
            supplement.parse_grid([bad])
    with pytest.raises(ValueError, match="twice"):
        supplement.parse_grid(["dsqwen-7b:1.3", "dsqwen-7b:1.4"])


# ------------------------------------------------------------------------- the search


def _base(anchor=0.821962, capacity=2.3644, model="dsqwen-14b"):
    return supplement.BaseAnchor(model=model, shape=SHAPE, anchor_rho=anchor,
                                 capacity_rps=capacity, anchor_rule="", base_root="",
                                 rho_priors_path="", rho_priors_sha256="")


def test_the_search_walks_the_grid_up_stops_at_the_first_violation_then_bisects() -> None:
    base = _base()
    case = supplement.simulate(base, (1.8, 2.1, 2.4, 2.8), 2.25, smoke=True)
    walk = [(p["stage"], p["factor_of_base"], p["verdict"][0]) for p in case["probes"]]
    assert walk[:3] == [("coarse", 1.8, "h"), ("coarse", 2.1, "h"), ("coarse", 2.4, "v")]
    assert [s for s, *_ in walk[3:]] == ["bisect", "bisect"]
    assert case["rho_star_status"] == boundary.RHO_STAR_MEASURED
    assert 2.1 < case["anchor_factor_of_base"] < 2.4
    assert case["smoke"]["factor_of_base"] == case["anchor_factor_of_base"]
    assert case["smoke"]["seconds"] == design.SMOKE_SECONDS
    assert all(p["seconds"] == design.BRACKET_SECONDS for p in case["probes"])


def test_the_14b_search_reaches_past_2_6x_and_reports_a_bound_above_the_grid() -> None:
    # D19: 14b must be probed up to 2.0-2.6 x rho*_run2 and beyond.
    base = _base()
    case = supplement.simulate(base, (1.8, 2.1, 2.4, 2.8), 9.0, smoke=True)
    factors = [p["factor_of_base"] for p in case["probes"]]
    assert factors == [1.8, 2.1, 2.4, 2.8] and max(factors) >= 2.6
    assert case["rho_star_status"] == boundary.RHO_STAR_LOWER_BOUND
    assert case["smoke"] is None  # no smoke on a bound


def test_below_the_grid_the_search_steps_down() -> None:
    case = supplement.simulate(_base(1.545116, 2.3556, "dsqwen-7b"), (1.3, 1.45, 1.6, 1.8),
                               1.0, smoke=True)
    coarse = [p["factor_of_base"] for p in case["probes"] if p["stage"] == "coarse"]
    assert coarse[0] == 1.3 and coarse[1] == pytest.approx(1.3 / design.STEP_FOUND, abs=1e-3)
    assert case["rho_star_status"] == boundary.RHO_STAR_MEASURED


def test_every_scenario_is_listed_with_its_wall_clock() -> None:
    cases = supplement.scenarios(_base(), (1.8, 2.1, 2.4, 2.8), smoke=True)
    assert len(cases) == 5  # below, three intervals, above
    assert [c["rho_star_status"] for c in cases[:4]] == [boundary.RHO_STAR_MEASURED] * 4
    assert cases[-1]["rho_star_status"] == boundary.RHO_STAR_LOWER_BOUND
    seconds = supplement.scenario_seconds(cases[2], SHAPE, cooldown_s=45.0)
    # 3 coarse + 2 bisect + smoke; violated probes pay the request deadline
    assert seconds["cells"] == 6
    assert seconds["offered_load_s"] == 5 * design.BRACKET_SECONDS + design.SMOKE_SECONDS
    assert seconds["seconds_upper"] > seconds["seconds_expected"] > seconds["offered_load_s"]


# ------------------------------------------------------------------------ the run


def test_the_run_locates_rho_star_and_smokes_it_on_the_fit_s_ruler(tmp_path) -> None:
    flip = ANCHOR * 1.5  # between the 1.45 and 1.6 grid points
    code, fake = _run(tmp_path, flip)
    assert code == 0
    roles = [r for r, *_ in fake.driven]
    assert roles[-1] == design.ROLE_SMOKE and set(roles[:-1]) == {design.ROLE_BOUNDARY}
    coarse = [rho for r, st, rho, *_ in fake.driven if st == "coarse"]
    assert coarse == [pytest.approx(ANCHOR * f) for f in (1.3, 1.45, 1.6)]
    result = _result(tmp_path)
    shape = result["shapes"][SHAPE]
    assert shape["rho_star_status"] == boundary.RHO_STAR_MEASURED
    assert 1.45 < shape["anchor_factor_of_base"] < 1.6
    # the probes were judged on post-warm-up windows by D6': at 600 ms every one of them
    # violates the fixed 500 ms SLO, and the warm-up windows violate D6' too
    assert 600.0 > 500.0 and 600.0 < D6_TTFT_SLO_MS
    search = json.loads((tmp_path / "out" / "boundary" / f"{MODEL}_{SHAPE}.json").read_text())
    healthy = [p for p in search["probes"] if p["verdict"] == boundary.VERDICT_HEALTHY]
    assert healthy and all(p["violating_windows"] == 0 for p in healthy)
    assert search["rho_star_status"] == boundary.RHO_STAR_MEASURED
    assert search["base"]["anchor_rho"] == ANCHOR and search["grid"]
    assert all("factor_of_base" in p for p in search["probes"])
    smoke = shape["smoke"]
    assert smoke["driven"] and smoke["in_band"] is True
    assert 0.2 <= smoke["violating_fraction"] <= 0.7
    assert smoke["rho"] == pytest.approx(shape["anchor_rho"])
    status = json.loads((tmp_path / "out" / campaign.CAMPAIGN_STATUS_FILE).read_text())
    assert status["status"] == "complete" and status["exit_code"] == 0


def test_every_cell_is_a_ladder_ledger_line_above_the_base_run_s_ids(tmp_path) -> None:
    _run(tmp_path, ANCHOR * 1.5)
    records = _ledger(tmp_path)
    assert {r["role"] for r in records} == {design.ROLE_BOUNDARY, design.ROLE_SMOKE}
    assert {r["split"] for r in records} == {design.SPLIT_AUXILIARY}
    assert all(r["serial"] > supplement.SUPPLEMENT_SERIAL_BASE for r in records)
    assert len({r["cell_id"] for r in records}) == len(records)
    assert all(r["capacity_rps"] == CAPACITY and r["warmup_s"] == design.WARMUP_S
               for r in records)
    smoke = [r for r in records if r["role"] == design.ROLE_SMOKE]
    assert len(smoke) == 1 and smoke[0]["stage"] == design.STAGE_DWELL
    assert smoke[0]["duration_s"] == design.SMOKE_SECONDS
    # the analysis knows every stage written: the probes are not analysed, the smoke hold
    # trains by its role (D21), as it does in dline_refit
    policy = decision.PREREGISTERED
    for r in records:
        row = {"shape": r["shape"], "primitive": r["primitive"], "stage": r["stage"],
               "role": r["role"], "cell_id": r["cell_id"], "model": r["model"]}
        want = decision.ROLE_TRAIN if r["role"] == design.ROLE_SMOKE else decision.ROLE_EXCLUDED
        assert policy.role(row) == want
    plan = json.loads((tmp_path / "out" / "plan.json").read_text())
    assert plan["design"] == ladder.DESIGN_NAME and plan["mode"] == supplement.MODE
    manifest = json.loads((tmp_path / "out" / ladder.RUN_MANIFEST).read_text())
    assert manifest["bases"][SHAPE]["anchor_rho"] == ANCHOR


def test_a_smoke_outside_the_band_ends_the_run_loudly(tmp_path, capsys) -> None:
    code, _fake = _run(tmp_path, ANCHOR * 1.5, smoke_bad=0.9)
    assert code == supplement.EXIT_CHECK_FAILED
    out = capsys.readouterr().out
    assert "CHECK FAILED" in out and "outside 20%-70%" in out
    result = _result(tmp_path)
    assert result["shapes"][SHAPE]["smoke"]["in_band"] is False
    assert result["check_failures"]
    status = json.loads((tmp_path / "out" / campaign.CAMPAIGN_STATUS_FILE).read_text())
    assert status["exit_code"] == supplement.EXIT_CHECK_FAILED


def test_no_smoke_runs_on_a_bound_and_that_fails_the_check(tmp_path, capsys) -> None:
    code, fake = _run(tmp_path, 99.0)  # healthy up to the top of the grid
    assert code == supplement.EXIT_CHECK_FAILED
    assert design.ROLE_SMOKE not in {r for r, *_ in fake.driven}
    assert max(rho for _r, _s, rho, _d, _c in fake.driven) == pytest.approx(ANCHOR * 1.8)
    result = _result(tmp_path)
    assert result["shapes"][SHAPE]["rho_star_status"] == boundary.RHO_STAR_LOWER_BOUND
    assert result["shapes"][SHAPE]["smoke"]["driven"] is False
    assert "lower_bound" in capsys.readouterr().out


def test_the_run_refuses_a_label_other_than_d6_prime_and_other_mistakes(tmp_path) -> None:
    with pytest.raises(ValueError, match="D6'"):
        supplement.run_boundary_supplement(
            _args(tmp_path, fit_ttft_slo_mode="fixed"), {MODEL: [SHAPE]},
            check_controller=False)
    with pytest.raises(SystemExit, match="--tpot-slo-ms"):
        supplement.run_boundary_supplement(_args(tmp_path, tpot_slo_ms=100.0),
                                           {MODEL: [SHAPE]}, check_controller=False)
    with pytest.raises(ValueError, match="no grid"):
        supplement.run_boundary_supplement(_args(tmp_path, reprobe_grid=["dsllama-8b:1.5"]),
                                           {MODEL: [SHAPE]}, check_controller=False)
    with pytest.raises(ValueError, match="one model"):
        supplement.run_boundary_supplement(_args(tmp_path), {MODEL: [SHAPE], "dsllama-8b": [SHAPE]},
                                           check_controller=False)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "x").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        supplement.run_boundary_supplement(_args(tmp_path), {MODEL: [SHAPE]},
                                           check_controller=False)


def test_the_cli_dry_run_lists_the_probe_sequences(tmp_path, capsys) -> None:
    base = _base_run(tmp_path / "base")
    out = tmp_path / "dry"
    argv = ["--reprobe-shapes", f"{MODEL}:{SHAPE}", "--reprobe-base", str(base),
            "--reprobe-grid", f"{MODEL}:{GRID}", "--smoke-at-rho-star",
            "--out-dir", str(out), "--raw-dir", str(out / "raw"), "--dry-run",
            "--index", str(tmp_path / "absent-index.json")]
    assert campaign.main(argv) == 0
    text = capsys.readouterr().out
    assert "grid (x rho*_base): 1.3" in text and "smoke" in text
    assert f"TTFT SLO at L={S3_INPUT}: {round(D6_TTFT_SLO_MS, 1):g} ms" in text
    plan = json.loads((out / "plan.json").read_text())
    target = plan["targets"][0]
    assert target["grid_factors_of_base"] == [1.3, 1.45, 1.6, 1.8]
    assert len(target["scenarios"]) == 5
    assert plan["label"]["ttft_slo_mode"] == "slowdown" and plan["label"]["ttft_slowdown_k"] == 5.0
    assert not (out / ladder.RUN_MANIFEST).exists() and not (out / ladder.LEDGER).exists()


def test_the_cli_keeps_the_supplement_flags_together(tmp_path, capsys) -> None:
    base = str(_base_run(tmp_path / "base"))
    for argv in (
        ["--reprobe-grid", f"{MODEL}:1.3", "--reprobe-shapes", f"{MODEL}:{SHAPE}"],
        ["--smoke-at-rho-star", "--reprobe-shapes", f"{MODEL}:{SHAPE}"],
        ["--reprobe-base", base],
        ["--reprobe-base", base, "--reprobe-shapes", f"{MODEL}:{SHAPE}",
         "--design", "primitives"],
    ):
        with pytest.raises(SystemExit):
            campaign.main([*argv, "--out-dir", str(tmp_path / "o"), "--dry-run"])

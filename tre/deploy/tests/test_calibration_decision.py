"""The pre-registered decision rules and the hold-out isolation of
``scripts.analysis.calibration_decision``.

Every branch of the preregistration's decision rule (docs/preregistration-20260923-
calibration-run2.md §4) has a test here, because each one is a place where a quiet bug
would flip a published verdict: a candidate that should have lost wins, or the hold-out
set leaks into the choice it is meant to check.
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import pytest

from scripts.analysis import calibration_decision as cd
from tre_common.registry import load_registry

REGISTRY = Path(__file__).resolve().parents[1] / "registry.yaml"
MODEL = "dsqwen-7b"
GROUPS = {"decode": ["S4", "S5"], "mixed": ["S1", "S2", "T9"], "prefill": ["S3", "T8"]}
SHAPE_TOKENS = {"S1": (256, 128), "S2": (768, 192), "S3": (2048, 96), "S4": (256, 448),
                "S5": (768, 384), "T8": (1600, 112), "T9": (812, 241), "M": (0, 0)}


# ------------------------------------------------------------- decision rule branches


def test_a_candidate_passes_only_when_all_three_conditions_hold() -> None:
    ok = cd.candidate_conditions(gain_points=3.5, ci_low_points=0.4, ci_high_points=7.0,
                                 holdout_diff_points=0.0)
    assert ok["loro_gain_ge_3"] and ok["ci90_excludes_0"] and ok["holdout_not_worse"]
    assert ok["passes_selection"] and ok["passes"]


def test_condition_1_needs_three_full_points_of_loro_min() -> None:
    c = cd.candidate_conditions(gain_points=2.99, ci_low_points=0.4, ci_high_points=7.0,
                                holdout_diff_points=1.0)
    assert not c["loro_gain_ge_3"] and not c["passes_selection"] and not c["passes"]


def test_condition_2_fails_when_the_interval_touches_or_crosses_zero() -> None:
    for lo in (-0.1, 0.0):
        c = cd.candidate_conditions(gain_points=5.0, ci_low_points=lo, ci_high_points=9.0,
                                    holdout_diff_points=1.0)
        assert not c["ci90_excludes_0"] and not c["passes"]


def test_condition_2_is_about_the_candidate_being_better_not_worse() -> None:
    # An interval entirely below 0 excludes 0 too, but it says the candidate is worse.
    c = cd.candidate_conditions(gain_points=5.0, ci_low_points=-9.0, ci_high_points=-1.0)
    assert not c["ci90_excludes_0"]


def test_condition_3_fails_on_a_negative_holdout_difference_and_accepts_a_tie() -> None:
    worse = cd.candidate_conditions(gain_points=5.0, ci_low_points=1.0, ci_high_points=9.0,
                                    holdout_diff_points=-0.01)
    tie = cd.candidate_conditions(gain_points=5.0, ci_low_points=1.0, ci_high_points=9.0,
                                  holdout_diff_points=0.0)
    assert worse["passes_selection"] and not worse["passes"]
    assert tie["passes"]


def test_condition_3_unevaluated_is_not_a_pass() -> None:
    c = cd.candidate_conditions(gain_points=5.0, ci_low_points=1.0, ci_high_points=9.0)
    assert c["holdout_not_worse"] is None and not c["passes"]


def _cand(value, loro, passes):
    return {"value": value, "loro_min": loro,
            "conditions": {"passes_selection": passes}}


def test_no_candidate_passing_selection_means_no_winner() -> None:
    assert cd.pick_selection_winner([_cand(0.01, 0.9, False), _cand(0.02, 0.95, False)]) is None


def test_the_selection_winner_is_the_best_passing_candidate_not_the_best_overall() -> None:
    cands = [_cand(0.01, 0.80, True), _cand(0.02, 0.84, True), _cand(0.04, 0.99, False)]
    assert cd.pick_selection_winner(cands) == 0.02


def test_a_selection_tie_goes_to_the_smaller_w_p() -> None:
    assert cd.pick_selection_winner([_cand(0.04, 0.8, True), _cand(0.01, 0.8, True)]) == 0.01


def test_no_winner_leaves_the_baseline_as_a_verdict() -> None:
    c = cd.final_choice(0.0, None, None)
    assert c["value"] == 0.0 and not c["replaced_baseline"]
    assert "no difference detected" in c["verdict"]


def test_a_holdout_veto_reverts_to_the_baseline_not_to_the_runner_up() -> None:
    c = cd.final_choice(0.0, 0.02, -0.5)
    assert c["value"] == 0.0 and not c["replaced_baseline"] and "condition 3" in c["verdict"]


def test_a_winner_that_survives_the_holdout_replaces_the_baseline() -> None:
    c = cd.final_choice(0.0, 0.02, 0.0)
    assert c["value"] == 0.02 and c["replaced_baseline"]


@pytest.mark.parametrize(
    "passes, adopted",
    [({"a": True, "b": True, "c": False}, True),
     ({"a": True, "b": True, "c": True}, True),
     ({"a": True, "b": False, "c": False}, False),
     ({"a": False, "b": False, "c": False}, False)],
)
def test_regime_aware_needs_two_of_three_models(passes, adopted) -> None:
    assert cd.regime_aware_adopted(passes) is adopted


def test_all_models_on_the_baseline_share_it() -> None:
    loro = {m: {0.0: 0.8, 0.01: 0.82} for m in "abc"}
    s = cd.shared_w_p({"a": 0.0, "b": 0.0, "c": 0.0}, loro)
    assert s["adopt"] and s["value"] == 0.0 and s["worst_loss_points"] == 0.0


def test_a_shared_value_costing_under_one_point_is_adopted() -> None:
    loro = {"a": {0.0: 0.800, 0.01: 0.805}, "b": {0.0: 0.80, 0.01: 0.83}, "c": {0.0: 0.80, 0.01: 0.80}}
    s = cd.shared_w_p({"a": 0.0, "b": 0.01, "c": 0.0}, loro)
    # sharing 0.01 costs a -0.5 and c 0.0; sharing 0.0 costs b 3.0 -> 0.01 wins, adopted
    assert s["value"] == 0.01 and s["adopt"]
    assert math.isclose(s["worst_loss_points"], 0.0, abs_tol=1e-9)


def test_a_shared_value_costing_one_point_or_more_is_not_adopted() -> None:
    loro = {"a": {0.0: 0.80, 0.02: 0.79}, "b": {0.0: 0.78, 0.02: 0.80}}
    s = cd.shared_w_p({"a": 0.0, "b": 0.02}, loro)
    assert not s["adopt"] and s["worst_loss_points"] >= 1.0


def test_a_shared_loss_of_exactly_one_point_is_not_under_one_point() -> None:
    loro = {"a": {0.0: 0.80, 0.02: 0.79}, "b": {0.0: 0.79, 0.02: 0.80}}
    s = cd.shared_w_p({"a": 0.0, "b": 0.02}, loro)
    assert math.isclose(s["worst_loss_points"], 1.0) and not s["adopt"]


@pytest.mark.parametrize(
    "diff, lo, hi, verdict",
    [(3.2, 0.5, 6.0, "A better"), (-3.5, -7.0, -0.2, "B better"),
     (2.0, 0.5, 3.0, "equivalent"), (4.0, -1.0, 8.0, "equivalent"), (None, None, None, "undefined")],
)
def test_holdout_comparisons_name_a_winner_only_for_three_points_and_a_clear_interval(diff, lo, hi, verdict) -> None:
    assert cd.paired_verdict(diff, lo, hi) == verdict


def test_the_preregistered_constants() -> None:
    assert cd.W_P_GRID == (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)
    assert cd.BASELINE_W_P == 0.0 and cd.MAIN_LAMBDA == 3.0 and cd.SENSITIVITY_LAMBDA == 10.0
    assert cd.MIN_GAIN_POINTS == 3.0 and cd.CI_LEVEL == 0.90 and cd.SHARE_LOSS_POINTS == 1.0
    assert cd.REGIME_AWARE_MIN_MODELS == 2
    assert cd.THETA_CONFIG.criterion == "balanced_accuracy"


# ------------------------------------------------------------------ cell policy


def test_m_cells_and_ramps_are_held_out_under_every_policy() -> None:
    for policy in cd.POLICIES.values():
        assert policy.role({"shape": "M", "primitive": "hold", "stage": "ladder"}) == cd.ROLE_HOLDOUT
        assert policy.role({"shape": "S1", "primitive": "ramp", "stage": ""}) == cd.ROLE_HOLDOUT


def test_the_preregistered_policy_trains_on_ladder_and_adaptive_hold_cells_only() -> None:
    p = cd.PREREGISTERED
    assert p.role({"shape": "S1", "primitive": "hold", "stage": "ladder"}) == cd.ROLE_TRAIN
    assert p.role({"shape": "S1", "primitive": "hold", "stage": "adaptive"}) == cd.ROLE_TRAIN
    for stage in ("coarse", "bisect", "dwell", "sentinel"):
        assert p.role({"shape": "S1", "primitive": "hold", "stage": stage}) == cd.ROLE_EXCLUDED
    assert p.role({"shape": "S1", "primitive": "steps", "stage": ""}) == cd.ROLE_EXCLUDED


def test_an_unknown_hold_stage_is_an_error_not_a_silent_drop() -> None:
    with pytest.raises(ValueError, match="stage"):
        cd.PREREGISTERED.role({"shape": "S1", "primitive": "hold", "stage": "hold_ladder"})


def test_an_unknown_primitive_is_an_error() -> None:
    with pytest.raises(ValueError, match="primitive"):
        cd.PREREGISTERED.role({"shape": "S1", "primitive": "sweep", "stage": ""})


# --------------------------------------------------------------- synthetic dataset


def _window_rows(shape, primitive, stage, cell_id, load, start_ms, n, rng, flip=False):
    """Windows of one cell at a fixed load: queue, tokens and latency all rise with load,
    and the p95s cross the SLO a little above load 1 (noisily), so every regime group
    carries both labels."""
    i_tok, o_tok = SHAPE_TOKENS[shape]
    rows = []
    for k in range(n):
        ws = start_ms + 5000 * k
        noise = rng.uniform(-0.15, 0.15)
        eff = load + noise
        running = 8.0 * eff
        waiting = max(0.0, 30.0 * (eff - 1.0))
        gen = 900.0 * min(eff, 1.1) * (1.0 + 0.1 * (o_tok > 300))
        prompt = gen * (max(i_tok, 200) / max(o_tok, 100))
        ttft = 150.0 + 1500.0 * max(0.0, eff - 0.9)
        tpot = 30.0 + 80.0 * max(0.0, eff - 0.95)
        violated = ttft > 500.0 or tpot > 75.0
        if flip:
            violated = not violated
            ttft = 900.0 if violated else 100.0
            tpot = 20.0
        rows.append({
            "model": MODEL, "shape": shape, "primitive": primitive, "stage": stage, "rho": load,
            "cell_id": cell_id, "attempt": 1, "split": "holdout" if shape == "M" else "train",
            "cell_status": "valid", "scenario_id": cell_id, "scenario_family": f"i{i_tok}_o{o_tok}",
            "input_tokens": i_tok, "output_tokens": o_tok, "concurrency": 0,
            "window_start_ms": ws, "window_end_ms": ws + 30000,
            "prompt_tokens_total": prompt, "generation_tokens_total": gen,
            "avg_waiting": waiting, "avg_running": running, "avg_swapping": 0.0,
            "queue_control": "", "trs": "", "completed_requests": 40,
            "p95_ttft_client_ms": ttft, "p95_tpot_client_ms": tpot, "p95_e2e_client_ms": ttft + 100 * tpot,
            "model_errors": 0, "proxy_transient_errors": 0, "client_timeouts": 0,
            "slo_label": "violated" if violated else "healthy", "slo_violated": violated,
        })
    return rows


def _synthetic(tmp_path: Path, *, flip_holdout: bool = False, seed: int = 7) -> tuple[Path, Path]:
    rng = random.Random(seed)
    windows, cells = [], []
    t = 1_000_000
    for shape in list(SHAPE_TOKENS):
        loads = (0.6, 0.9, 1.0, 1.1, 1.3) if shape != "M" else (0.8, 1.2)
        for j, load in enumerate(loads):
            cid = f"i{SHAPE_TOKENS[shape][0]}_o{SHAPE_TOKENS[shape][1]}_c{1000 + j}"
            windows += _window_rows(shape, "hold", "ladder", cid, load, t, 22, rng,
                                    flip=flip_holdout and shape == "M")
            cells.append({"model": MODEL, "shape": shape, "primitive": "hold", "cell_id": cid,
                          "attempt": 1, "start_ms": t})
            t += 200_000
        if shape != "M":
            cid = f"i{SHAPE_TOKENS[shape][0]}_o{SHAPE_TOKENS[shape][1]}_c120"
            ramp = []
            for k, load in enumerate((0.7, 0.9, 1.1, 1.3)):
                ramp += _window_rows(shape, "ramp", "", cid, load, t + 20000 * k, 4, rng, flip=flip_holdout)
            windows += ramp
            cells.append({"model": MODEL, "shape": shape, "primitive": "ramp", "cell_id": cid,
                          "attempt": 1, "start_ms": t})
            t += 200_000
        # a boundary probe: excluded under the preregistered policy
        cid = f"i{SHAPE_TOKENS[shape][0]}_o{SHAPE_TOKENS[shape][1]}_c1143"
        windows += _window_rows(shape, "hold", "bisect", cid, 1.4, t, 10, rng)
        cells.append({"model": MODEL, "shape": shape, "primitive": "hold", "cell_id": cid,
                      "attempt": 1, "start_ms": t})
        t += 200_000
    ds = tmp_path / ("dataset_flip" if flip_holdout else "dataset")
    ds.mkdir()
    for name, rows in (("windows.csv", windows), ("cells.csv", cells)):
        with (ds / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (ds / "manifest.json").write_text(json.dumps({
        "label": {"slo_ms": {"p95_ttft_client_ms": 500.0, "p95_tpot_client_ms": 75.0}},
        "registry_used_for_signal_columns": {"sha256": "not-this-registry"},
    }))
    groups = tmp_path / "regime_groups.json"
    groups.write_text(json.dumps({"consistent_across_models": True, "groups": GROUPS}))
    return ds, groups


def _cells(ds: Path, groups: Path, policy=cd.PREREGISTERED):
    windows, cell_rows, manifest = cd.load_dataset(ds)
    starts = {(c["model"], c["shape"], c["primitive"], c["cell_id"], str(c["attempt"])): float(c["start_ms"])
              for c in cell_rows}
    return cd.build_cells(windows, slo_ms=cd.slo_from_manifest(manifest), registry=load_registry(str(REGISTRY)),
                          groups=cd.load_groups(groups, [MODEL]), cell_start_ms=starts,
                          hold_warmup_s=policy.hold_warmup_s)


# ------------------------------------------------------------------ isolation


def test_the_split_puts_every_m_cell_and_every_ramp_in_the_holdout(tmp_path) -> None:
    ds, groups = _synthetic(tmp_path)
    training, holdout, counts = cd.split_dataset(_cells(ds, groups), cd.PREREGISTERED)
    shapes_prims = {(c.shape, c.primitive) for c in training.cells(MODEL)}
    assert not any(s == "M" or p == "ramp" for s, p in shapes_prims)
    assert all(p == "hold" for _, p in shapes_prims)
    assert all(c.stage == "ladder" for c in training.cells(MODEL))
    # hold-out: M's 2 ladder cells + M's probe (shape M is held out whole) + 7 ramps;
    # excluded: the 7 training shapes' boundary probes
    assert counts["holdout"] == 2 + 1 + 7 and counts["excluded"] == 7


def test_the_holdout_set_exposes_no_windows(tmp_path) -> None:
    ds, groups = _synthetic(tmp_path)
    _, holdout, _ = cd.split_dataset(_cells(ds, groups), cd.PREREGISTERED)
    public = [n for n in dir(holdout) if not n.startswith("_")]
    assert sorted(public) == ["evaluate", "summary"]
    with pytest.raises(TypeError, match="FrozenSelection"):
        holdout.evaluate({"fits": {}}, lambda by_model, fz: by_model)


def test_the_selection_stage_refuses_the_holdout_set(tmp_path) -> None:
    ds, groups = _synthetic(tmp_path)
    _, holdout, _ = cd.split_dataset(_cells(ds, groups), cd.PREREGISTERED)
    for call in (lambda: cd.select(holdout, n_boot=2, seed=1, processes=1),
                 lambda: cd.select_model(holdout, MODEL, lambda_wait=3.0, n_boot=2, seed=1, processes=1),
                 lambda: cd.bootstrap_loro(holdout, MODEL, [], n=1, seed="x")):
        with pytest.raises(TypeError, match="TrainingSet"):
            call()


def test_changing_the_holdout_cannot_change_the_selection(tmp_path) -> None:
    # Same training cells, hold-out labels inverted: every number the selection stage
    # produces - LORO curves, bootstrap intervals, fitted thresholds - must be identical.
    ds, groups = _synthetic(tmp_path)
    ds_flip, _ = _synthetic(tmp_path, flip_holdout=True)
    results = []
    for d in (ds, ds_flip):
        training, _, _ = cd.split_dataset(_cells(d, groups), cd.PREREGISTERED)
        frozen = cd.select(training, n_boot=4, seed=11, processes=1)
        results.append(json.dumps(frozen.selection, sort_keys=True, default=str))
    assert results[0] == results[1]


def test_the_full_analysis_runs_and_writes_a_verdict(tmp_path) -> None:
    ds, groups = _synthetic(tmp_path)
    result = cd.analyse(ds, groups, policy=cd.PREREGISTERED, registry_path=REGISTRY, n_boot=4, seed=3)
    per = result["decision"]["per_model"][MODEL]
    assert per["w_p"] in cd.W_P_GRID
    assert per["w_p_verdict"]
    assert result["signal_column_check"]["checked"] is False  # different registry: skipped, said so
    hold = result["holdout"][MODEL]
    assert hold["cells"] == 9 and hold["trs_vs_queue_per_replica"]["windows"] == hold["windows"]
    text = cd.render_markdown(result)
    assert "## Decision" in text and MODEL in text


def test_trs_comes_from_the_controller_computer_with_its_ema(tmp_path) -> None:
    ds, groups = _synthetic(tmp_path)
    cells = _cells(ds, groups)
    cell = next(c for c in cells.values() if c.primitive == "hold" and c.stage == "ladder")
    from tre_controller.signals.trs import TRSComputer, TRSInput

    p = cell.params
    comp = TRSComputer(ema_alpha=p.ema_alpha, ema_tau_ms=p.ema_tau_ms)
    expected = []
    for r in cell.rows:
        inp = TRSInput(prompt_tokens_total=float(r["prompt_tokens_total"]),
                       generation_tokens_total=float(r["generation_tokens_total"]),
                       avg_waiting=float(r["avg_waiting"]), avg_running=float(r["avg_running"]),
                       avg_swapping=float(r["avg_swapping"]), routable_pods=1, assigned_replicas=1,
                       w_p=0.01, w_d=p.w_d, lambda_wait=3.0, qmin=p.qmin)
        expected.append(comp.compute(inp, window_end_ms=int(float(r["window_end_ms"]))).TRS)
    assert cell.series(("trs", 0.01, 3.0)) == expected


def test_hold_warmup_windows_feed_the_ema_but_are_not_scored(tmp_path) -> None:
    ds, groups = _synthetic(tmp_path)
    cells = _cells(ds, groups)
    cell = next(c for c in cells.values() if c.primitive == "hold" and c.stage == "ladder")
    start = min(float(r["window_start_ms"]) for r in cell.rows)
    kept_starts = [float(cell.rows[i]["window_start_ms"]) for i in cell.kept]
    assert kept_starts and min(kept_starts) - start >= 60_000
    assert len(cell.rows) > len(cell.kept)


def test_a_fold_without_violations_has_no_ba_rather_than_a_wrong_one() -> None:
    from tre_calibration.dataset import CalibrationWindow

    healthy = [CalibrationWindow("c", "f", 2.0, True) for _ in range(5)]
    assert cd.balanced_accuracy(healthy) is None

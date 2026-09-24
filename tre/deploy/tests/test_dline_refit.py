"""The D-line refit driver (scripts.dline_refit) and the ledger-based steady cells of
scripts.alpha_fit (D4')."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest

from scripts import alpha_fit as af
from scripts import dline_refit as dl
from scripts import r3_grid
from tre_common import slo_labels

TRE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = TRE_ROOT / "deploy" / "registry.yaml"
MODEL = "dsqwen-7b"


# ----------------------------------------------------------------- the D3 w_p rule


def _row(w_p, ba, gap, half, source="merged"):
    return {"w_p": w_p, "train_ba": ba, "family_gap_frac": gap, "ci_half_frac": half,
            "source": source, "family_merged": source == "merged"}


def test_d3_takes_the_largest_w_p_meeting_c1_and_c2() -> None:
    rows = [
        _row(0.0, 0.85, 0.05, 0.15),
        _row(0.005, 0.845, 0.10, 0.15),               # c1, c2 hold
        _row(0.01, 0.842, 0.14, 0.15),                # c1, c2 hold
        _row(0.02, 0.80, 0.10, 0.15),                 # c1 fails (more than 1 SE below)
        _row(0.05, 0.845, 0.30, 0.15),                # c2 fails (family gap > CI half width)
        _row(0.1, 0.849, 0.10, 0.15, "max_family"),   # c3 fails: diagnostic only (D17)
        _row(0.2, 0.83, 0.20, 0.15, "max_family"),    # c1 and c2 fail
    ]
    assert dl.d3_select(rows, se=0.01) == 0.1
    flags = {r["w_p"]: (r["c1_1se"], r["c2_gap"], r["c3_merged"]) for r in rows}
    assert flags[0.02] == (False, True, True)
    assert flags[0.05] == (True, False, True)
    assert flags[0.1] == (True, True, False)
    assert [r["w_p"] for r in rows if r["admissible"]] == [0.0, 0.005, 0.01, 0.1]
    assert dl.D3_CONDITIONS == ("c1_1se", "c2_gap") and dl.D3_DIAGNOSTIC == ("c3_merged",)


def test_d17_c3_alone_no_longer_forces_w_p_to_zero() -> None:
    """On data holding shapes outside both families the family rule never publishes the
    merged theta (c3 false for every w_p); under D3 that pinned w_p* to None -> 0."""
    rows = [_row(0.0, 0.93, 0.35, 0.16, "max_family"), _row(0.0025, 0.927, 0.06, 0.11, "max_family"),
            _row(0.005, 0.928, 0.09, 0.11, "max_family"), _row(0.01, 0.926, 0.14, 0.11, "max_family")]
    assert dl.d3_select(rows, se=0.012) == 0.005
    assert not any(r["c3_merged"] for r in rows)


def test_stop_rule_15_is_the_legacy_d13_verdict() -> None:
    """final.json's ``stop_rule_15``: the D13 verdict at the pre-2026-09-24 15 % gate."""
    ok = {"satisfied": True, "reasons": [], "ci_half_width_fraction": 0.12}
    between = {"satisfied": True, "reasons": [], "ci_half_width_fraction": 0.17}
    thin = {"satisfied": False, "reasons": ["bootstrap publish rate 0.50 < 0.90"],
            "ci_half_width_fraction": 0.05}
    assert dl._legacy_stop(ok) is True and dl._legacy_stop(between) is False
    assert dl._legacy_stop(thin) is False and dl._legacy_stop({"reasons": []}) is None


def test_the_d13_gate_is_a_cli_parameter(tmp_path, monkeypatch) -> None:
    seen = {}

    def fake_verdict_report(**kw):
        seen["max_ci_fraction"] = kw.get("max_ci_fraction")
        raise RuntimeError("stop here")

    from scripts import theta_verdict as tv
    monkeypatch.setattr(tv, "verdict_report", fake_verdict_report)
    monkeypatch.setattr(dl, "D13_MAX_CI_FRACTION", 0.15)
    with pytest.raises(RuntimeError):
        dl.verdict(MODEL, None, {"fitting": "x", "families": {}}, 10.0, 0.0, 1.0)
    assert seen["max_ci_fraction"] == 0.15


def test_d3_admits_nothing_without_a_baseline_fit() -> None:
    rows = [{"w_p": 0.0, "error": "no theta"}, _row(0.01, 0.9, 0.0, 0.2)]
    assert dl.d3_select(rows, se=None) is None
    assert not any(r["admissible"] for r in rows)


def test_lambda_moves_off_one_only_for_two_points_of_ba() -> None:
    rows = [{"lambda_wait": lam, "train_ba": ba} for lam, ba in ((0.0, 0.80), (1.0, 0.81), (2.0, 0.825), (3.0, 0.82))]
    assert dl.lambda_select(rows) == 1.0          # +0.015 is not enough
    rows[3]["train_ba"] = 0.835
    assert dl.lambda_select(rows) == 3.0          # +0.025 is


def test_the_arms_are_the_label_arms_of_the_registry_profile() -> None:
    primary = dl.label_for(MODEL, "primary", str(REGISTRY))
    assert primary.slowdown and primary.ttft_slowdown_k == 5.0 and primary.ttft_floor_ms == 500.0
    assert primary.min_completed_requests == 20
    fixed = dl.label_for(MODEL, "fixed", str(REGISTRY))
    assert not fixed.slowdown and fixed.latency_slo_ms() == {"ttft_p95": 500.0, "tpot_p95": 75.0}
    k3 = dl.label_for(MODEL, "k3", str(REGISTRY))
    assert (k3.ttft_slowdown_k, k3.ttft_floor_ms) == (3.0, 150.0)
    assert dl.alpha_of(0) == 1.0 and dl.step90_s(0) == 0.0


# ---------------------------------------------------------------- end to end (tiny)


#: The dataset columns that say what a cell was (the rows the fit stages accept carry them).
_IDENTITY = {"primitive": "hold", "role": "ladder", "split": "train"}


def _fit_csv(path: Path, *, seed: int, shapes=("i2048_o96", "i256_o448", "i256_o128"),
             identity: dict | None = _IDENTITY) -> Path:
    rng = random.Random(seed)
    extra = list(identity) if identity else []
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=r3_grid.CSV_COLUMNS + extra)
        writer.writeheader()
        for cell in range(18):
            load = 0.5 + 0.06 * cell
            shape = shapes[cell % len(shapes)]
            start = 1_790_000_000_000 + cell * 1_000_000
            for k in range(16):
                running = 20.0 * load
                gen = 30_000.0 * min(load, 1.0) * rng.uniform(0.95, 1.05)
                ratio = load * rng.uniform(0.85, 1.15)
                writer.writerow({
                    **(identity or {}),
                    "scenario_id": f"{shape}_c{1000 + cell}", "scenario_family": shape,
                    "input_tokens": 0, "output_tokens": 0, "concurrency": 1000 + cell,
                    "window_start_ms": start, "window_end_ms": start + 30_000,
                    "prompt_tokens_total": gen, "generation_tokens_total": gen,
                    "avg_waiting": 0.0, "avg_running": running, "avg_swapping": 0.0,
                    "queue_control": running, "p95_ttft_client_ms": 400.0 * ratio,
                    "p95_tpot_client_ms": 30.0, "p95_e2e_client_ms": 1.0, "trs": "",
                    "model_errors": 0, "proxy_transient_errors": 0, "client_timeouts": 0,
                })
                start += 10_000
    return path


def test_the_three_stages_run_end_to_end_and_chain_through_their_outputs(tmp_path, monkeypatch) -> None:
    fit = tmp_path / "fit"
    fit.mkdir()
    _fit_csv(fit / f"{MODEL}_fitting.csv", seed=1)
    _fit_csv(fit / f"{MODEL}_validation.csv", seed=2, shapes=("i0_o0",))
    _fit_csv(fit / f"{MODEL}_fitting_decode_heavy.csv", seed=3, shapes=("i256_o448",))
    _fit_csv(fit / f"{MODEL}_fitting_prefill_heavy.csv", seed=4, shapes=("i2048_o96",))
    # small grids and resample counts: this pins the plumbing, not the statistics
    monkeypatch.setattr(dl, "TAUS_S", (0, 10))
    monkeypatch.setattr(dl, "WP_GRID", (0.0, 0.02))
    monkeypatch.setattr(dl, "LAMBDAS", (0.0, 1.0))
    for name in ("WP_RESAMPLES", "LAMBDA_RESAMPLES", "FINAL_RESAMPLES"):
        monkeypatch.setattr(dl, name, (20, 10))
    monkeypatch.setattr(dl, "BA_SE_RESAMPLES", 20)
    monkeypatch.setattr(dl, "M_CI_RESAMPLES", 20)
    out = tmp_path / "out"
    common = ["--model", MODEL, "--arm", "fixed", "--fit-dir", str(fit), "--out-dir", str(out),
              "--registry", str(REGISTRY)]
    assert dl.main(["alpha", *common, "--alpha-rule", "refit0922", "--alpha-w-p", "0.01"]) == 0
    alpha = json.loads((out / MODEL / "fixed" / "alpha.json").read_text())
    assert alpha["rule"] == "refit0922" and alpha["chosen_tau_s"] in (0, 10)
    assert alpha["label_def"]["mode"] == "fixed" and alpha["w_p"] == 0.01
    # D18: the common tau is published whatever the rule picked; the sweep is disclosed
    assert alpha["published_tau_s"] == dl.PUBLISH_TAU_S == 10.0
    assert alpha["published_registry_fields"] == {"ema_tau_ms": 10_000.0, "ema_alpha": round(dl.alpha_of(10), 6)}
    assert alpha["disclosure"]["curves"]["tau_s"] == [0.0, 10.0]
    assert alpha["training_set"]["rows"]["fitting"][dl.KIND_CONSTANT] == 18 * 16
    assert dl.main(["wp", *common]) == 0
    wp = json.loads((out / MODEL / "fixed" / "wp.json").read_text())
    assert wp["tau_s"] == alpha["published_tau_s"] and [r["w_p"] for r in wp["grid"]] == [0.0, 0.02]
    assert wp["d3_rule"]["conditions"] == ["c1_1se", "c2_gap"]
    assert wp["w_p_used"] in (0.0, 0.02) and wp["lambda_star"] in (0.0, 1.0)
    assert dl.main(["final", *common]) == 0
    final = json.loads((out / MODEL / "fixed" / "final.json").read_text())
    assert "error" not in final, final
    assert final["w_p"] == wp["w_p_used"] and final["theta_published"] == final["theta_merged"]  # D5
    assert (out / MODEL / "fixed" / "holdout_final.json").exists()
    assert final["holdout_evaluated"] is True and "M_ba" in final
    assert dl.main(["summary", "--model", MODEL, "--fit-dir", str(fit), "--out-dir", str(out),
                    "--registry", str(REGISTRY)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    rec = summary["models"][MODEL]["fixed"]
    assert rec["wp_rule"]["w_p_used"] == wp["w_p_used"]
    assert set(rec["band_by_family"]) <= {"prefill_heavy", "decode_heavy"}


def test_no_holdout_never_opens_the_validation_csv(tmp_path, monkeypatch) -> None:
    """Plan §6.11 note 9: M is read once, after it is frozen. ``--no-holdout`` must fit and
    publish without the validation CSV existing, and never call the hold-out report."""
    from scripts import theta_verdict as tv

    fit = tmp_path / "fit"
    fit.mkdir()
    _fit_csv(fit / f"{MODEL}_fitting.csv", seed=1)
    _fit_csv(fit / f"{MODEL}_fitting_decode_heavy.csv", seed=3, shapes=("i256_o448",))
    _fit_csv(fit / f"{MODEL}_fitting_prefill_heavy.csv", seed=4, shapes=("i2048_o96",))
    assert not (fit / f"{MODEL}_validation.csv").exists()
    monkeypatch.setattr(dl, "TAUS_S", (0, 10))
    monkeypatch.setattr(dl, "WP_GRID", (0.0, 0.02))
    monkeypatch.setattr(dl, "LAMBDAS", (0.0, 1.0))
    for name in ("WP_RESAMPLES", "LAMBDA_RESAMPLES", "FINAL_RESAMPLES"):
        monkeypatch.setattr(dl, name, (20, 10))
    monkeypatch.setattr(dl, "BA_SE_RESAMPLES", 20)

    def _no_m(*_a, **_k):
        raise AssertionError("the hold-out report ran under --no-holdout")

    monkeypatch.setattr(tv, "holdout_report", _no_m)
    out = tmp_path / "out"
    common = ["--model", MODEL, "--arm", "fixed", "--fit-dir", str(fit), "--out-dir", str(out),
              "--registry", str(REGISTRY)]
    assert dl.main(["alpha", *common, "--alpha-rule", "refit0922", "--alpha-w-p", "0.01"]) == 0
    assert dl.main(["wp", *common]) == 0
    assert dl.main(["final", *common, "--no-holdout"]) == 0
    final = json.loads((out / MODEL / "fixed" / "final.json").read_text())
    assert "error" not in final, final
    assert final["holdout_evaluated"] is False and final["holdout"] == dl.HOLDOUT_SKIPPED
    assert not any(k.startswith("M_") for k in final)
    assert final["inputs"]["validation"] == dl.HOLDOUT_SKIPPED
    assert final["theta_published"] == final["theta_merged"]  # D5 still applies
    assert not (out / MODEL / "fixed" / "holdout_final.json").exists()
    assert (out / MODEL / "fixed" / "verdict_final.json").exists()
    assert dl.main(["summary", "--model", MODEL, "--fit-dir", str(fit), "--out-dir", str(out),
                    "--registry", str(REGISTRY)]) == 0


def test_a_stage_refuses_to_run_before_its_predecessor(tmp_path) -> None:
    with pytest.raises(SystemExit):
        dl.main(["wp", "--model", MODEL, "--arm", "fixed", "--fit-dir", str(tmp_path),
                 "--out-dir", str(tmp_path / "out"), "--registry", str(REGISTRY)])


# ------------------------------------------------ D4': steady cells from the ledger


def test_first_round_cell_ids_keep_the_load_code_rule() -> None:
    assert af.is_steady_cell("i256_o128_c1177") and af.is_steady_cell("i400_o160_c2085")
    assert not af.is_steady_cell("i256_o128_c120")


def test_a_ladder_cell_is_steady_by_its_ledger_line_not_its_code() -> None:
    ledger = {
        "i256_o128_c1000056": {"role": "ladder", "primitive": "hold"},
        "i256_o128_c1000090": {"role": "ramp", "primitive": "ramp"},
        "i768_o192_c1000001": {"role": "sentinel", "primitive": "hold"},
        "i768_o192_c1000110": {"role": "boundary", "primitive": "hold"},
    }
    assert af.is_steady_cell("i256_o128_c1000056", ledger)
    assert af.is_steady_cell("i256_o128_c1000056#3", ledger)          # a bootstrap copy
    assert not af.is_steady_cell("i256_o128_c1000090", ledger)        # >= 1e6, but a ramp
    assert af.is_steady_cell("i768_o192_c1000001", ledger)
    assert af.is_steady_cell("i768_o192_c1000110", ledger)
    # the old rule read every second-round code (>= 1e6) as a steady hold, ramps included
    with pytest.raises(af.LedgerMissing):
        af.is_steady_cell("i256_o128_c1000090")
    with pytest.raises(af.LedgerMissing):
        af.is_steady_cell("i256_o128_c1000999", ledger)               # not in the ledger


def test_a_ramp_no_longer_feeds_the_spurious_episode_tie_break() -> None:
    t = [i * 10_000.0 for i in range(12)]
    crit = [False, True, True] + [False] * 9
    healthy = [False] * 12
    ledger = {"i256_o128_c1000090": {"role": "ramp", "primitive": "ramp"}}
    ramp = af.cell_stats("i256_o128_c1000090", t, crit, healthy,
                         steady=af.is_steady_cell("i256_o128_c1000090", ledger))
    assert ramp.spurious == 1 and not ramp.steady_healthy
    agg = af.aggregate([ramp])
    assert agg.spurious_steady_healthy == 0 and agg.spurious_per_h is None


def test_alpha_fit_reads_the_ledger_it_is_given(tmp_path) -> None:
    ledger = tmp_path / "cells.jsonl"
    ledger.write_text("\n".join(json.dumps(r) for r in (
        {"cell_id": "i256_o128_c1000056", "attempt": 1, "role": "ladder", "primitive": "hold"},
        {"cell_id": "i256_o128_c1000090", "attempt": 1, "role": "ramp", "primitive": "ramp"},
    )) + "\n")
    args = af._parse(["--model", MODEL, "--fitting-csv", "x.csv", "--w-p", "0.01",
                      "--lambda-wait", "1", "--ledger", str(ledger), "--output", "o.json"])
    assert args.ledger == [str(ledger)]
    from scripts.rewindow_from_raw import load_ledgers

    loaded = load_ledgers(args.ledger)
    assert not af.is_steady_cell("i256_o128_c1000090", loaded)
    assert af.is_steady_cell("i256_o128_c1000056", loaded)


def test_the_label_module_guard_holds_for_the_driver() -> None:
    """The primary label every stage uses reads unserved evidence from the counts only."""
    label = dl.label_for(MODEL, "primary", str(REGISTRY))
    row = {"completed_requests": 25, "p95_tpot_client_ms": 20.0,
           "ttft_len_samples": slo_labels.format_ttft_len_samples([(100.0, 256)] * 25),
           "slo_violated": "True", "model_errors": 0, "proxy_transient_errors": 0, "client_timeouts": 0}
    assert slo_labels.window_slo_label(row, label) == slo_labels.LABEL_HEALTHY
    row["client_timeouts"] = 1
    assert slo_labels.window_slo_label(row, label) == slo_labels.LABEL_VIOLATED


# ------------------------------------------------------------ the fit plan wiring


class _Args:
    out_dir = Path("/out")
    window_ms = 30000
    fit_step_ms = 10000
    fit_window_align = "grid"
    instant_sample_ms = 1000
    ttft_slo_ms = 500.0
    tpot_slo_ms = 75.0
    registry = None


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_a_ladder_fit_plan_selects_its_cells_from_the_ledger_and_runs_the_dline() -> None:
    from scripts import calibration_campaign as campaign

    ledger = Path("/out/cells.jsonl")
    plan = campaign.fit_plan([MODEL], Path("/out"), Path("/raw"), _Args(), {}, ledgers=[ledger])
    assert plan["selection"]["source"] == "ledger"
    by_purpose = {e["purpose"].split(" ")[0]: e["command"] for e in plan["rewindow"]}
    fitting = by_purpose["fitting"]
    assert _flag(fitting, "--ledger") == str(ledger) and _flag(fitting, "--only-split") == "train"
    assert _flag(fitting, "--window-align") == "grid" and _flag(fitting, "--step-ms") == "10000"
    assert _flag(fitting, "--ttft-slo-mode") == "slowdown"          # the rows carry the three arms
    assert _flag(by_purpose["held-out"], "--only-split") == "holdout"
    assert "--exclude-cell-id" not in fitting                        # no INDEX.json list
    assert _flag(plan["alpha"][0]["command"], "--ledger") == str(ledger)   # D4' steady cells
    stages = [(e.get("model"), e.get("arm"), e["stage"]) for e in plan["dline"]]
    assert (MODEL, "primary", "alpha") in stages and (MODEL, "fixed", "final") in stages
    assert stages[-1] == (None, None, "summary")
    assert plan["order"].index("dline") < plan["order"].index("theta")


def test_a_primitives_fit_plan_still_uses_the_schedule_index() -> None:
    from scripts import calibration_campaign as campaign

    index = {"schedules": [{"cell_id": "i0_o0_c95", "held_out": True}]}
    plan = campaign.fit_plan([MODEL], Path("/out"), Path("/raw"), _Args(), index)
    assert plan["selection"]["source"] == "schedule index"
    fitting = next(e["command"] for e in plan["rewindow"] if e["purpose"].startswith("fitting"))
    assert _flag(fitting, "--exclude-cell-id") == "i0_o0_c95" and "--ledger" not in fitting


# ------------------------------------------------ D16: the training set is cut from datasets

_DATASET_IDENTITY = ["model", "shape", "primitive", "stage", "rho", "cell_id", "attempt", "split",
                     "cell_status", "role", "rho_factor", "replicate", "possibly_contaminated", "in_warmup"]
_SHAPE_IO = {"S1": (256, 128), "S3": (2048, 96), "S4": (256, 448), "S5": (768, 384), "T8": (1600, 112),
             "M": (0, 0)}


def _dataset(root: Path, name: str, cells: list[tuple], *, windows: int = 12, seed: int = 7) -> Path:
    """A standard dataset (windows.csv + manifest.json) of ``cells``:
    (shape, primitive, role, stage, split, code, load)."""
    d = root / name / "dataset"
    d.mkdir(parents=True)
    rng = random.Random(seed)
    with (d / "windows.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_DATASET_IDENTITY + r3_grid.CSV_COLUMNS)
        writer.writeheader()
        for n, (shape, primitive, role, stage, split, code, load) in enumerate(cells):
            i, o = _SHAPE_IO[shape]
            sid = f"i{i}_o{o}_c{code}"
            start = 1_790_000_000_000 + n * 1_000_000
            for _ in range(windows):
                running = 20.0 * load
                gen = 30_000.0 * min(load, 1.0) * rng.uniform(0.95, 1.05)
                ratio = load * rng.uniform(0.85, 1.15)
                writer.writerow({
                    "model": MODEL, "shape": shape, "primitive": primitive, "stage": stage, "rho": load,
                    "cell_id": sid, "attempt": 1, "split": split, "cell_status": "valid", "role": role,
                    "rho_factor": "", "replicate": "", "possibly_contaminated": "False", "in_warmup": "False",
                    "scenario_id": sid, "scenario_family": shape,
                    "input_tokens": i, "output_tokens": o, "concurrency": code,
                    "window_start_ms": start, "window_end_ms": start + 30_000,
                    "prompt_tokens_total": gen, "generation_tokens_total": gen,
                    "avg_waiting": 0.0, "avg_running": running, "avg_swapping": 0.0,
                    "queue_control": running, "p95_ttft_client_ms": 400.0 * ratio,
                    "p95_tpot_client_ms": 30.0, "p95_e2e_client_ms": 1.0, "trs": "",
                    "model_errors": 0, "proxy_transient_errors": 0, "client_timeouts": 0,
                })
                start += 10_000
    (d / "manifest.json").write_text(json.dumps({"format_revision": 2}))
    return d


#: run 1: first-round primitives (hold stages, steps, bursts) and its held-out M cells.
_RUN1 = [
    ("S1", "hold", "", "coarse", "train", 1001, 0.6),
    ("S3", "hold", "", "dwell", "train", 1002, 1.1),
    ("S4", "steps", "", "", "train", 60, 0.9),
    ("S5", "bursts", "", "", "train", 70, 0.9),
    ("M", "ramp", "", "", "holdout", 90, 0.9),
    ("T8", "hold", "", "bisect", "train", 1003, 1.4),
]
#: run 2: the ladder design (roles), its ramps and its sealed M-shape cells.
_RUN2 = [
    ("S1", "hold", "ladder", "ladder", "train", 1100001, 1.3),
    ("S3", "hold", "boundary", "bisect", "auxiliary", 1100002, 0.8),
    ("S4", "hold", "sentinel", "sentinel", "auxiliary", 1100003, 0.7),
    ("S5", "hold", "adaptive", "adaptive", "train", 1100004, 1.2),
    ("S1", "ramp", "ramp", "", "holdout", 1100005, 1.0),
    ("M", "hold", "ladder", "ladder", "holdout", 1100006, 1.0),
]


def _ids(path: Path) -> list[str]:
    with path.open(newline="") as fh:
        return list(dict.fromkeys(r["scenario_id"] for r in csv.DictReader(fh)))


def _trainset(tmp_path: Path, fit_name: str = "fit", *extra: str, run2_as: str = "--h2-dataset") -> Path:
    fit = tmp_path / fit_name
    assert dl.main(["trainset", "--fit-dir", str(fit), "--h2-dataset", str(tmp_path / "run1"),
                    run2_as, str(tmp_path / "run2"), *extra]) == 0
    return fit


def test_trainset_keeps_constant_load_cells_and_seals_the_rest_as_h2(tmp_path) -> None:
    _dataset(tmp_path, "run1", _RUN1)
    _dataset(tmp_path, "run2", _RUN2, seed=8)
    fit = _trainset(tmp_path)
    # training: constant-load cells only, dataset order, run1 before run2 (CLI order)
    assert _ids(fit / f"{MODEL}_fitting.csv") == [
        "i256_o128_c1001", "i2048_o96_c1002", "i1600_o112_c1003",
        "i256_o128_c1100001", "i2048_o96_c1100002", "i256_o448_c1100003", "i768_o384_c1100004"]
    assert _ids(fit / f"{MODEL}_fitting_prefill_heavy.csv") == [
        "i2048_o96_c1002", "i1600_o112_c1003", "i2048_o96_c1100002"]
    assert _ids(fit / f"{MODEL}_fitting_decode_heavy.csv") == ["i256_o448_c1100003", "i768_o384_c1100004"]
    with (fit / f"{MODEL}_fitting.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["run"] for r in rows} == {"run1", "run2"} and {r["primitive"] for r in rows} == {"hold"}
    assert not list(fit.glob("*_validation.csv"))
    # H2: every dynamic cell + the sealed split of both runs, listed and hashed
    h2 = json.loads((fit / dl.H2_MANIFEST).read_text())
    got = {(c["run"], c["scenario_id"], c["why"]) for c in h2["cells"]}
    assert got == {("run1", "i256_o448_c60", "dynamic"), ("run1", "i768_o384_c70", "dynamic"),
                   ("run1", "i0_o0_c90", "sealed"), ("run2", "i256_o128_c1100005", "sealed"),
                   ("run2", "i0_o0_c1100006", "sealed")}
    assert all(c["windows"] == 12 for c in h2["cells"])
    assert len(h2["rows_sha256"]) == 64 and set(h2["rows_sha256_by_run"]) == {"run1", "run2"}
    man = json.loads((fit / dl.TRAINSET_MANIFEST).read_text())
    assert man["h2"]["rows_sha256"] == h2["rows_sha256"] and man["m_unread"] == {}
    assert man["models"][MODEL]["cells"] == 7 and man["models"][MODEL]["windows"] == 7 * 12
    assert man["sentinels"] is True
    ledger = [json.loads(x) for x in (fit / dl.TRAINING_LEDGER).read_text().splitlines()]
    assert len(ledger) == 7 and {x["primitive"] for x in ledger} == {"hold"}


def test_the_h2_hash_ignores_the_sentinel_switch_and_moves_with_any_h2_row(tmp_path) -> None:
    _dataset(tmp_path, "run1", _RUN1)
    _dataset(tmp_path, "run2", _RUN2, seed=8)
    with_s = json.loads((_trainset(tmp_path, "a") / dl.TRAINSET_MANIFEST).read_text())
    without = json.loads((_trainset(tmp_path, "b", "--no-sentinels") / dl.TRAINSET_MANIFEST).read_text())
    assert without["sentinels"] is False and without["models"][MODEL]["cells"] == 6
    assert without["models"][MODEL]["excluded"] == {"sentinel_windows_excluded": 12}
    assert "i256_o448_c1100003" not in _ids(tmp_path / "b" / f"{MODEL}_fitting.csv")
    assert without["h2"]["rows_sha256"] == with_s["h2"]["rows_sha256"]   # a sentinel is not H2
    # one H2 value changed -> a different hash (the disclosure can prove H2 unchanged)
    w = tmp_path / "run1" / "dataset" / "windows.csv"
    text = w.read_text()
    lines = text.splitlines(keepends=True)
    k = next(i for i, x in enumerate(lines) if ",steps," in x)
    lines[k] = lines[k].replace(",0.0,", ",0.5,", 1)
    w.write_text("".join(lines))
    moved = json.loads((_trainset(tmp_path, "c") / dl.TRAINSET_MANIFEST).read_text())
    assert moved["h2"]["rows_sha256"] != with_s["h2"]["rows_sha256"]
    assert moved["h2"]["cells_sha256"] == with_s["h2"]["cells_sha256"]
    assert moved["models"][MODEL]["files"] == {
        k: {**v, "path": v["path"].replace("/a/", "/c/")} for k, v in with_s["models"][MODEL]["files"].items()}


def test_the_sealed_split_of_a_dataset_is_m_and_is_skipped_unread(tmp_path) -> None:
    _dataset(tmp_path, "run1", _RUN1)
    _dataset(tmp_path, "run2", _RUN2, seed=8)
    fit = _trainset(tmp_path, "fit", run2_as="--dataset")
    man = json.loads((fit / dl.TRAINSET_MANIFEST).read_text())
    assert man["m_unread"] == {"run2": {MODEL: {"rows": 24, "cells": 2}}}
    h2 = json.loads((fit / dl.H2_MANIFEST).read_text())
    assert {c["scenario_id"] for c in h2["cells"] if c["run"] == "run2"} == set()
    # whatever an M row holds changes nothing but the source file's own hash
    w = tmp_path / "run2" / "dataset" / "windows.csv"
    lines = w.read_text().splitlines(keepends=True)
    lines = [x.replace(",0.0,", ",999.0,") if ",holdout," in x else x for x in lines]
    w.write_text("".join(lines))
    again = json.loads((_trainset(tmp_path, "fit2", run2_as="--dataset") / dl.TRAINSET_MANIFEST).read_text())
    assert again["h2"]["rows_sha256"] == man["h2"]["rows_sha256"]
    assert [v["sha256"] for v in again["models"][MODEL]["files"].values()] == \
        [v["sha256"] for v in man["models"][MODEL]["files"].values()]


def test_d21_a_smoke_hold_trains_by_its_role(tmp_path) -> None:
    """The boundary supplement's smoke hold (role smoke, stage dwell, split auxiliary,
    primitive hold) is a training cell - keyed on the role, as in calibration_decision."""
    from scripts import calibration_design as design

    assert design.TRAINING_HOLD_ROLES == {design.ROLE_SMOKE}
    row = {"model": MODEL, "cell_id": "i2048_o96_c2000001", "attempt": "1", "shape": "S3",
           "primitive": "hold", "role": design.ROLE_SMOKE, "stage": design.STAGE_DWELL,
           "split": design.SPLIT_AUXILIARY}
    assert dl.cell_kind(row["primitive"], row["role"], row["split"], row["shape"]) == dl.KIND_CONSTANT
    for sentinels in (True, False):
        assert dl.assign_set(row, sealed_to_h2=True, sentinels=sentinels) == dl.SET_TRAINING
    # the role does not unseal anything, nor make a dynamic primitive train
    assert dl.assign_set({**row, "split": "holdout"}, sealed_to_h2=False, sentinels=True) == dl.SET_M
    assert dl.assign_set({**row, "primitive": "steps"}, sealed_to_h2=False, sentinels=True) == dl.SET_H2
    # end to end: the smoke hold of a supplement run lands in the fitting CSV and its family
    _dataset(tmp_path, "sup", [("S3", "hold", "smoke", "dwell", "auxiliary", 2000001, 1.0),
                               ("S3", "hold", "boundary", "bisect", "auxiliary", 2000002, 1.1)])
    fit = tmp_path / "fit"
    assert dl.main(["trainset", "--fit-dir", str(fit), "--h2-dataset", str(tmp_path / "sup")]) == 0
    assert _ids(fit / f"{MODEL}_fitting.csv") == ["i2048_o96_c2000001", "i2048_o96_c2000002"]
    assert _ids(fit / f"{MODEL}_fitting_prefill_heavy.csv") == ["i2048_o96_c2000001", "i2048_o96_c2000002"]
    by_role = json.loads((fit / dl.TRAINSET_MANIFEST).read_text())["models"][MODEL]["by_run_role"]
    assert by_role == {"sup|smoke": 12, "sup|boundary": 12}


def test_trainset_never_guesses_what_a_cell_was(tmp_path) -> None:
    _dataset(tmp_path, "run1", [("S1", "", "", "", "train", 1001, 0.6)])
    with pytest.raises(SystemExit, match="no primitive"):
        dl.main(["trainset", "--fit-dir", str(tmp_path / "fit"), "--h2-dataset", str(tmp_path / "run1")])
    _dataset(tmp_path, "run2", [("M", "hold", "ladder", "ladder", "train", 1100001, 0.6)])
    with pytest.raises(SystemExit, match="held-out shape"):
        dl.main(["trainset", "--fit-dir", str(tmp_path / "fit2"), "--h2-dataset", str(tmp_path / "run2")])


def test_a_fit_stage_refuses_rows_that_are_not_constant_load_training_rows(tmp_path) -> None:
    fit = tmp_path / "fit"
    fit.mkdir()
    _fit_csv(fit / f"{MODEL}_fitting.csv", seed=1, identity={"primitive": "steps", "role": "", "split": "train"})
    common = ["--model", MODEL, "--arm", "fixed", "--fit-dir", str(fit), "--out-dir", str(tmp_path / "o"),
              "--registry", str(REGISTRY), "--alpha-rule", "refit0922", "--alpha-w-p", "0.01"]
    with pytest.raises(SystemExit, match="D16"):
        dl.main(["alpha", *common])
    _fit_csv(fit / f"{MODEL}_fitting.csv", seed=1, identity={"primitive": "hold", "role": "ladder",
                                                             "split": "holdout"})
    with pytest.raises(SystemExit, match="sealed"):
        dl.main(["alpha", *common])
    # no identity column and no ledger: unknown, refused (a cell id is not read)
    _fit_csv(fit / f"{MODEL}_fitting.csv", seed=1, identity=None)
    p = dl.paths(fit, MODEL)
    with pytest.raises(SystemExit, match="unknown"):
        dl.check_training_inputs(MODEL, p)
    # ... unless a ledger says what every cell was
    ledger = {f"{s}_c{1000 + c}": {"primitive": "hold", "role": "ladder", "split": "train"}
              for c, s in enumerate(("i2048_o96", "i256_o448", "i256_o128") * 6)}
    got = dl.check_training_inputs(MODEL, p, ledger=ledger)
    assert got["rows"]["fitting"][dl.KIND_CONSTANT] == 18 * 16 and got["trainset_manifest"] is None


def test_a_fit_stage_refuses_a_training_csv_changed_after_trainset(tmp_path) -> None:
    _dataset(tmp_path, "run1", _RUN1)
    _dataset(tmp_path, "run2", _RUN2, seed=8)
    fit = _trainset(tmp_path)
    p = dl.paths(fit, MODEL)
    prov = dl.check_training_inputs(MODEL, p)
    assert prov["h2_rows_sha256"] == json.loads((fit / dl.H2_MANIFEST).read_text())["rows_sha256"]
    with p["fitting"].open("a") as fh:
        fh.write(p["fitting"].read_text().splitlines()[-1] + "\n")
    with pytest.raises(SystemExit, match="not the file the trainset stage wrote"):
        dl.check_training_inputs(MODEL, p)


def test_the_chain_runs_on_a_trainset_directory_without_m(tmp_path, monkeypatch) -> None:
    shapes = ("S1", "S3", "S4", "S5", "T8")
    ladder = [(shapes[n % 5], "hold", "ladder", "ladder", "train", 1_100_000 + n, 0.5 + 0.05 * n)
              for n in range(20)]
    _dataset(tmp_path, "run2", ladder + [("S1", "ramp", "ramp", "", "holdout", 1_100_900, 1.0)], windows=16)
    fit = tmp_path / "fit"
    assert dl.main(["trainset", "--fit-dir", str(fit), "--h2-dataset", str(tmp_path / "run2")]) == 0
    monkeypatch.setattr(dl, "TAUS_S", (0, 10))
    monkeypatch.setattr(dl, "WP_GRID", (0.0, 0.02))
    monkeypatch.setattr(dl, "LAMBDAS", (0.0, 1.0))
    for name in ("WP_RESAMPLES", "LAMBDA_RESAMPLES", "FINAL_RESAMPLES"):
        monkeypatch.setattr(dl, name, (20, 10))
    monkeypatch.setattr(dl, "BA_SE_RESAMPLES", 20)
    out = tmp_path / "out"
    common = ["--model", MODEL, "--arm", "fixed", "--fit-dir", str(fit), "--out-dir", str(out),
              "--registry", str(REGISTRY)]
    assert dl.main(["alpha", *common, "--alpha-rule", "refit0922", "--alpha-w-p", "0.01"]) == 0
    assert dl.main(["wp", *common]) == 0
    with pytest.raises(SystemExit, match="no M"):
        dl.main(["final", *common])
    assert dl.main(["final", *common, "--no-holdout"]) == 0
    final = json.loads((out / MODEL / "fixed" / "final.json").read_text())
    h2 = json.loads((fit / dl.H2_MANIFEST).read_text())
    assert final["training_set"]["h2_rows_sha256"] == h2["rows_sha256"]
    assert final["tau_s"] == 10.0 and final["holdout_evaluated"] is False
    assert dl.main(["summary", "--model", MODEL, "--fit-dir", str(fit), "--out-dir", str(out),
                    "--registry", str(REGISTRY)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["models"][MODEL]["fixed"]["alpha"]["published_tau_s"] == 10.0


# --------------------------------------------------------- D18: the published tau


def _d4prime_doc(bas, *, chosen=10.0, fa=0.01) -> dict:
    taus = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0)
    return {"rule": "d4prime", "chosen_tau_s": chosen, "alpha_fit": {
        "curve": [{"tau_s": t, "alpha": dl.alpha_of(t), "ba": b, "se": 0.01, "fa": fa, "recall": 0.7,
                   "feasible": fa <= 0.05, "step90_ema_s": dl.step90_s(t), "step90_with_dwell_s": dl.step90_s(t) + 10}
                  for t, b in zip(taus, bas)],
        "bootstrap": {"used": 100, "selection_frequency_by_tau_s": {"10.0": 0.6, "0.0": 0.4}}}}


def test_the_disclosure_reports_the_flat_interval_around_the_published_tau() -> None:
    doc = dl.publish_alpha(_d4prime_doc((0.82, 0.83, 0.84, 0.835, 0.80, 0.79), chosen=5.0), 10.0)
    assert doc["chosen_tau_s"] == 5.0 and doc["published_tau_s"] == 10.0     # the rule's pick is kept
    d = doc["disclosure"]
    assert d["within_1se_tau_s"] == [5.0, 10.0, 15.0] and d["flat_interval_s"] == [5.0, 15.0]
    assert d["published_within_1se_of_best"] and d["best_ba_tau_s"] == 10.0
    assert d["range_s"] == [0.0, 15.0] and d["range_ba_spread"] == pytest.approx(0.02)
    assert d["range_ba_spread_within_1se"] is False and d["range_all_within_1se_of_best"] is False
    assert d["curves"]["step90_with_dwell_s"][2] == dl.step90_s(10.0) + 10
    assert d["selection_frequency_by_tau_s"] == {"10.0": 0.6, "0.0": 0.4}
    off = dl.publish_alpha(_d4prime_doc((0.82, 0.83, 0.84, 0.835, 0.80, 0.79)), 20.0)["disclosure"]
    assert off["flat_interval_s"] is None and not off["published_within_1se_of_best"]


def test_publish_tau_rule_publishes_the_rule_pick() -> None:
    doc = dl.publish_alpha(_d4prime_doc((0.82,) * 6, chosen=0.0), None)
    assert doc["published_tau_s"] == 0.0 and doc["published_alpha"] == 1.0
    assert dl.publish_tau_arg("rule") is None and dl.publish_tau_arg("10") == 10.0
    with pytest.raises(Exception):
        dl.publish_tau_arg("-1")
    # an alpha.json from before D18 still chains through its rule pick
    assert dl.published_tau({"chosen_tau_s": 15.0}) == 15.0

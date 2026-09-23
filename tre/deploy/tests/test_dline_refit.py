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


def test_d3_takes_the_largest_admissible_w_p() -> None:
    rows = [
        _row(0.0, 0.85, 0.05, 0.15),
        _row(0.005, 0.845, 0.10, 0.15),               # all three hold
        _row(0.01, 0.842, 0.14, 0.15),                # all three hold: largest admissible
        _row(0.02, 0.80, 0.10, 0.15),                 # c1 fails (more than 1 SE below)
        _row(0.05, 0.845, 0.30, 0.15),                # c2 fails (family gap > CI half width)
        _row(0.1, 0.849, 0.10, 0.15, "max_family"),   # c3 fails (families disagree)
    ]
    assert dl.d3_select(rows, se=0.01) == 0.01
    flags = {r["w_p"]: (r["c1_1se"], r["c2_gap"], r["c3_merged"]) for r in rows}
    assert flags[0.02] == (False, True, True)
    assert flags[0.05] == (True, False, True)
    assert flags[0.1] == (True, True, False)
    assert [r["w_p"] for r in rows if r["admissible"]] == [0.0, 0.005, 0.01]


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


def _fit_csv(path: Path, *, seed: int, shapes=("i2048_o96", "i256_o448", "i256_o128")) -> Path:
    rng = random.Random(seed)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=r3_grid.CSV_COLUMNS)
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
    assert dl.main(["wp", *common]) == 0
    wp = json.loads((out / MODEL / "fixed" / "wp.json").read_text())
    assert wp["tau_s"] == alpha["chosen_tau_s"] and [r["w_p"] for r in wp["grid"]] == [0.0, 0.02]
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

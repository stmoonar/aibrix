"""v1's lambda_wait / w_p selection on v2 windows (scripts.v1_lambda_fit) and its wiring
into ``dline_refit wp --lambda-method v1`` (user 2026-09-24)."""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import pytest

from scripts import dline_refit as dl
from scripts import r3_grid
from scripts import v1_lambda_fit as v1
from tre_calibration.dataset import CalibrationWindow

TRE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = TRE_ROOT / "deploy" / "registry.yaml"
MODEL = "dsqwen-7b"


# ------------------------------------------------------------- v1 helpers, verbatim


def test_the_grids_are_v1s() -> None:
    lam = v1.frange(v1.LAMBDA_MIN, v1.LAMBDA_MAX, v1.LAMBDA_STEP)
    assert lam[0] == 1.0 and lam[-1] == 4.0 and len(lam) == 13
    wp = v1.frange(v1.W_P_MIN, v1.W_P_MAX, v1.W_P_STEP)
    assert wp[0] == 0.01 and wp[-1] == 0.08 and len(wp) == 15
    assert (v1.P95_WEIGHT, v1.AVG_WEIGHT) == (0.8, 0.2)
    assert (v1.W_P_PRIOR_CENTER, v1.W_P_PRIOR_STRENGTH, v1.LAMBDA_PENALTY_STRENGTH) == (0.04, 0.002, 0.0005)
    assert v1.SCORE_QMIN == 0.0


def test_rank_helpers_behave_like_v1() -> None:
    assert v1.rankdata([3.0, 1.0, 1.0, 2.0]) == [4.0, 1.5, 1.5, 3.0]
    assert v1.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert v1.spearman([1, 1, 1], [1, 2, 3]) == 0.0          # degenerate -> 0, as v1
    assert v1.spearman([1.0], [2.0]) == 0.0
    assert v1.auc([0.1, 0.2, 0.9], [0, 0, 1]) == 1.0
    assert v1.auc([0.1, 0.2], [1, 1]) == 0.5                 # one class -> 0.5, as v1


def _cand(adj, auc=0.9, lam=1.0, wp=0.04):
    return {"objective_adjusted": adj, "hard_auc": auc, "lambda_wait": lam, "w_p": wp}


def test_the_tie_break_is_v1s() -> None:
    assert v1.pick_better(_cand(0.9), _cand(0.91))["objective_adjusted"] == 0.91
    assert v1.pick_better(_cand(0.9, auc=0.8), _cand(0.9, auc=0.85))["hard_auc"] == 0.85
    assert v1.pick_better(_cand(0.9, lam=2.0), _cand(0.9, lam=1.5))["lambda_wait"] == 1.5
    assert v1.pick_better(_cand(0.9, wp=0.06), _cand(0.9, wp=0.035))["w_p"] == 0.035
    first = _cand(0.9, wp=0.03)
    assert v1.pick_better(first, _cand(0.9, wp=0.05)) is first   # equal distance: keep current


def _win(n, gen, running, waiting, health, avg=None, met=True):
    return v1.V1Window(scenario_id=f"c{n}", window_start_ms=float(n), prompt_tokens=gen,
                       generation_tokens=gen, avg_running=running, avg_waiting=waiting,
                       slo_met=met, p95_health=health, avg_health=avg)


def test_the_penalties_are_v1s() -> None:
    ws = [_win(i, 1000.0 + i, 10.0, 1.0 + i, 1.0 / (1 + i)) for i in range(5)]
    c = v1.evaluate(ws, w_p=0.04, lambda_wait=1.0)
    assert c["w_p_penalty"] == 0.0 and c["lambda_wait_penalty"] == 0.0
    c = v1.evaluate(ws, w_p=0.01, lambda_wait=3.0)
    assert c["w_p_penalty"] == pytest.approx(0.002 * 0.75 ** 2)
    assert c["lambda_wait_penalty"] == pytest.approx(0.0005 * 4.0)
    assert c["objective_adjusted"] == pytest.approx(c["objective"] - c["w_p_penalty"] - c["lambda_wait_penalty"])
    assert c["avg_score"] == 0.5 and c["avg_pairs"] == 0   # no average health -> v1's 0.5


def test_the_score_is_the_unfloored_raw_tss() -> None:
    w = _win(0, 1000.0, 0.2, 0.1, 0.5)
    # (w_p * prompt + gen) / (running + lambda * waiting), no qmin floor (v1 trs_no_floor)
    assert v1.score(w, w_p=0.04, lambda_wait=3.0) == pytest.approx(1040.0 / 0.5)


def _synthetic(lambda_true: float, n: int = 240, seed: int = 3) -> list[v1.V1Window]:
    """Health is a monotone function of the TSS at ``lambda_true`` (w_p = 0.04)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        gen, prompt = rng.uniform(1e4, 3e4), rng.uniform(1e4, 6e4)
        running, waiting = rng.uniform(5, 40), rng.uniform(0, 40)
        tss = (0.04 * prompt + gen) / (running + lambda_true * waiting)
        h = 1.0 / (1.0 + 500.0 / tss)
        out.append(v1.V1Window(scenario_id=f"c{i // 10}", window_start_ms=float(i), prompt_tokens=prompt,
                               generation_tokens=gen, avg_running=running, avg_waiting=waiting,
                               slo_met=h > 0.5, p95_health=h, avg_health=h))
    return out


def test_the_three_stages_find_a_planted_lambda_and_refine_jointly() -> None:
    res = v1.search(_synthetic(3.0))
    lam_a = res["best_a"]["lambda_wait"]
    assert lam_a in (2.75, 3.0)                            # the penalty may pull one step down
    assert res["best_b"]["w_p"] == 0.04
    lams = sorted({c["lambda_wait"] for c in res["stage_c"]})
    wps = sorted({c["w_p"] for c in res["stage_c"]})
    assert lams == [lam_a - 0.25, lam_a - 0.125, lam_a, lam_a + 0.125, lam_a + 0.25]
    assert wps == [0.035, 0.0375, 0.04, 0.0425, 0.045]
    assert len(res["stage_c"]) == 25 and len(res["stage_a"]) == 13 and len(res["stage_b"]) == 16
    assert (res["lambda_wait"], res["w_p"]) == (res["best_c"]["lambda_wait"], res["best_c"]["w_p"])
    assert 2.5 <= res["lambda_wait"] <= 3.25


def _synthetic_wp(w_p: float, n: int = 240, seed: int = 5) -> list[v1.V1Window]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        # prompts large enough that every w_p > 0 adds rank noise the prior cannot outweigh
        gen, prompt = rng.uniform(1e4, 3e4), rng.uniform(0.0, 3e6)
        running = rng.uniform(5, 40)
        h = 1.0 / (1.0 + 500.0 * running / (w_p * prompt + gen))
        out.append(v1.V1Window(scenario_id=f"c{i}", window_start_ms=float(i), prompt_tokens=prompt,
                               generation_tokens=gen, avg_running=running, avg_waiting=0.0,
                               slo_met=h > 0.5, p95_health=h, avg_health=h))
    return out


def test_the_w_p_grids() -> None:
    v1_grid = v1.w_p_grid("v1")
    assert v1_grid == v1.frange(0.01, 0.08, 0.005) and len(v1_grid) == 15
    assert v1.w_p_grid("with_zero") == [0.0, *v1_grid] and v1.w_p_grid() == v1.w_p_grid("with_zero")
    with pytest.raises(ValueError):
        v1.w_p_grid("nope")


def test_with_zero_reaches_w_p_zero_and_refines_down_to_it() -> None:
    res = v1.search(_synthetic_wp(0.0))           # default grid: with_zero
    assert res["best_b"]["w_p"] == 0.0 and res["w_p_grid"]["name"] == "with_zero"
    assert res["w_p_grid"]["stage_b"][0] == 0.0 and len(res["w_p_grid"]["stage_b"]) == 16
    assert sorted({c["w_p"] for c in res["stage_c"]}) == [0.0, 0.0025, 0.005]
    assert res["w_p"] in (0.0, 0.0025, 0.005)
    # the prior penalty still applies at 0: 0.002 * ((0 - 0.04) / 0.04)^2
    c0 = next(c for c in res["stage_b"] if c["w_p"] == 0.0)
    assert c0["w_p_penalty"] == pytest.approx(0.002)


def test_stage_c_is_clamped_to_the_grid() -> None:
    res = v1.search(_synthetic_wp(0.0), wp_grid="v1")
    assert res["best_a"]["lambda_wait"] == 1.0 and res["best_b"]["w_p"] == 0.01
    assert sorted({c["lambda_wait"] for c in res["stage_c"]}) == [1.0, 1.125, 1.25]
    assert sorted({c["w_p"] for c in res["stage_c"]}) == [0.01, 0.0125, 0.015]


def test_a_flat_objective_leaves_lambda_to_the_penalty() -> None:
    """No waiting anywhere: lambda does not move the score, the penalty picks lambda = 1."""
    ws = [_win(i, 1000.0 + 7 * i, 10.0 + (i % 5), 0.0, 1.0 / (1 + (i % 7))) for i in range(60)]
    res = v1.search(ws)
    assert res["best_a"]["lambda_wait"] == 1.0 and res["lambda_wait"] == 1.0
    assert res["flatness"]["stage_a_objective_range"] == pytest.approx(0.0)


# ------------------------------------------------------ the windows (v2 data -> v1)


def test_request_membership_is_closed_right(tmp_path) -> None:
    d = tmp_path / "ds"
    d.mkdir()
    with (d / "requests.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "cell_id", "attempt", "done_ts_ms", "tpot_ms", "outcome"])
        w.writerow([MODEL, "c1", 1, 1000, 10, "ok"])     # == start: out when closed right
        w.writerow([MODEL, "c1", 1, 1500, 20, "ok"])
        w.writerow([MODEL, "c1", 1, 2000, 30, "ok"])     # == end: in when closed right
        w.writerow([MODEL, "c1", 1, 1600, 99, "client_timeout"])   # not served
        w.writerow(["other", "c1", 1, 1600, 99, "ok"])
    idx = v1.RequestIndex()
    assert idx.add_dataset("run", d, MODEL) == 3
    assert idx.tpots("run", MODEL, "c1", "1", 1000, 2000, closed_right=True) == [20.0, 30.0]
    assert idx.tpots("run", MODEL, "c1", "1", 1000, 2000, closed_right=False) == [10.0, 20.0]
    assert idx.tpots("other-run", MODEL, "c1", "1", 1000, 2000, closed_right=True) == []


_IDENTITY = {"primitive": "hold", "role": "ladder", "split": "train", "run": "r1", "cell_id": "", "attempt": 1}


def _fit_csv_with_requests(fit: Path, ds: Path, *, seed: int = 1) -> None:
    """A fitting CSV (+ family CSVs) whose windows carry ttft_len_samples, and the
    requests.csv the average TPOT is rebuilt from."""
    rng = random.Random(seed)
    fit.mkdir(parents=True, exist_ok=True)
    ds.mkdir(parents=True, exist_ok=True)
    cols = r3_grid.CSV_COLUMNS + [c for c in _IDENTITY if c not in r3_grid.CSV_COLUMNS]
    for c in ("ttft_len_samples", "completed_requests"):
        if c not in cols:
            cols.append(c)
    reqs = []

    def served_in(sid, start, end):  # closed right, as the grid-aligned rewindow
        return sum(1 for r in reqs if r[1] == sid and start < r[3] <= end)

    files = {name: (fit / f"{MODEL}_{name}.csv").open("w", newline="")
             for name in ("fitting", "fitting_decode_heavy", "fitting_prefill_heavy")}
    writers = {k: csv.DictWriter(f, fieldnames=cols) for k, f in files.items()}
    for w in writers.values():
        w.writeheader()
    shapes = ("i2048_o96", "i256_o448", "i256_o128")
    for cell in range(18):
        load = 0.5 + 0.06 * cell
        shape = shapes[cell % 3]
        sid = f"{shape}_c{1000 + cell}"
        start = 1_790_000_000_000 + cell * 1_000_000
        # one served request every 500 ms over the cell (16 windows on a 10 s step, 30 s wide)
        for j in range(360):
            reqs.append([MODEL, sid, 1, start + 250 + 500 * j, 20.0 * load, "ok"])
        for k in range(16):
            running = 20.0 * load
            waiting = max(0.0, 30.0 * (load - 1.0))
            gen = 30_000.0 * min(load, 1.0) * rng.uniform(0.95, 1.05)
            ratio = load * rng.uniform(0.85, 1.15)
            ttfts = [400.0 * ratio * rng.uniform(0.6, 1.0) for _ in range(20)]
            row = {**_IDENTITY, "cell_id": sid,
                   "scenario_id": sid, "scenario_family": shape, "input_tokens": 0, "output_tokens": 0,
                   "concurrency": 1000 + cell, "window_start_ms": start, "window_end_ms": start + 30_000,
                   "prompt_tokens_total": gen, "generation_tokens_total": gen, "avg_waiting": waiting,
                   "avg_running": running, "avg_swapping": 0.0, "queue_control": running,
                   "p95_ttft_client_ms": 400.0 * ratio, "p95_tpot_client_ms": 30.0, "p95_e2e_client_ms": 1.0,
                   "trs": "", "model_errors": 0, "proxy_transient_errors": 0, "client_timeouts": 0,
                   "ttft_len_samples": ";".join(f"{t:.2f}:256" for t in ttfts),
                   "completed_requests": served_in(sid, start, start + 30_000)}
            for name, w in writers.items():
                fam = {"fitting_decode_heavy": "i256_o448", "fitting_prefill_heavy": "i2048_o96"}.get(name)
                if fam is None or fam == shape:
                    w.writerow(row)
            start += 10_000
    for f in files.values():
        f.close()
    with (ds / "requests.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "cell_id", "attempt", "done_ts_ms", "tpot_ms", "outcome"])
        w.writerows(reqs)


def test_load_windows_rebuilds_average_health_and_checks_the_score(tmp_path) -> None:
    _fit_csv_with_requests(tmp_path / "fit", tmp_path / "ds")
    label = dl.label_for(MODEL, "fixed", str(REGISTRY))
    req = v1.RequestIndex()
    req.add_dataset("r1", tmp_path / "ds", MODEL)
    ws, stats = v1.load_windows(MODEL, tmp_path / "fit" / f"{MODEL}_fitting.csv", label, trim=1, requests=req)
    assert stats["windows"] == len(ws) == 18 * 15            # ramp trim 1 per cell
    assert stats["avg_health"] == len(ws) and stats["avg_ttft_only"] == 0
    assert all(w.avg_health is not None and 0 < w.avg_health < 1 for w in ws)
    # the p95 health is the v2 label's health score (1 / (1 + ratio_max))
    assert all(0 < w.p95_health <= 1 for w in ws)


def test_wp_lambda_method_v1_runs_the_chain(tmp_path, monkeypatch) -> None:
    fit, ds = tmp_path / "fit", tmp_path / "ds"
    _fit_csv_with_requests(fit, ds)
    for name in ("WP_RESAMPLES", "LAMBDA_RESAMPLES", "FINAL_RESAMPLES"):
        monkeypatch.setattr(dl, name, (20, 10))
    monkeypatch.setattr(dl, "BA_SE_RESAMPLES", 20)
    out = tmp_path / "out"
    (out / MODEL / "fixed").mkdir(parents=True)
    (out / MODEL / "fixed" / "alpha.json").write_text(json.dumps(
        {"published_tau_s": 10.0, "rule": "d4prime", "published_registry_fields": {}}))
    common = ["--model", MODEL, "--arm", "fixed", "--fit-dir", str(fit), "--out-dir", str(out),
              "--registry", str(REGISTRY)]
    with pytest.raises(SystemExit):
        dl.main(["wp", *common, "--lambda-method", "v1", "--requests-dataset", "no-equals-sign"])
    assert dl.main(["wp", *common, "--lambda-method", "v1", "--requests-dataset", f"r1={ds}"]) == 0
    wp = json.loads((out / MODEL / "fixed" / "wp.json").read_text())
    assert wp["lambda_method"] == "v1" and wp["grid"] == [] and "not applied" in wp["d3_rule"]["statement"]
    sel = wp["v1_selection"]
    assert wp["lambda_star"] == sel["lambda_wait"] and wp["w_p_used"] == wp["w_p_star"] == sel["w_p"]
    assert sel["lambda_wait"] in v1.frange(1.0, 4.0, 0.125) and 0.01 <= sel["w_p"] <= 0.08
    assert sel["window_stats"]["tpot_membership"]["closed_right_match"] == sel["window_stats"]["tpot_membership"]["checked"]
    assert dl.main(["final", *common, "--no-holdout"]) == 0
    final = json.loads((out / MODEL / "fixed" / "final.json").read_text())
    assert "error" not in final, final
    assert final["lambda_wait"] == wp["lambda_star"] and final["w_p"] == wp["w_p_used"]


def test_the_default_lambda_method_is_still_v2() -> None:
    assert dl.LAMBDA_METHODS == ("v2", "v1")


# ------------------------------------------------------------------------------ B'


def _cw(sid, t, signal, met, ratio, cls=None):
    return CalibrationWindow(scenario_id=sid, scenario_family="f", signal=signal, slo_met=met,
                             health_score=1.0 / (1.0 + ratio), window_start_ms=float(t),
                             latency_ratio_p95=ratio, violation_class=cls)


def test_severity_cut_is_the_training_violations_quantile() -> None:
    train = [_cw("a", i, 100.0, False, 1.0 + 0.1 * i, "both") for i in range(21)]
    train += [_cw("b", i, 900.0, True, 0.5) for i in range(10)]
    from tre_calibration.fit import _quantile
    assert v1.severity_cut(train) == pytest.approx(_quantile([1.0 + 0.1 * i for i in range(21)], 0.65))
    # slowdown windows carry no average ratio: severity = the p95 ratio
    assert v1.severity(train[3]) == pytest.approx(1.3)


def test_b_prime_counts_only_severe_violations_and_reports_the_bands() -> None:
    theta, tau = 100.0, 0.8
    ws = [
        _cw("a", 0, 50.0, False, 2.0, "both"),     # severe, Z .5 < tau: CRITICAL
        _cw("a", 1, 90.0, False, 2.0, "both"),     # severe, Z .9: LOW band, missed
        _cw("a", 2, 70.0, False, 1.1, "tpot_only"),  # mild, CRITICAL
        _cw("a", 3, 150.0, False, 1.05, "ttft_only"),  # mild, Z 1.5: healthy side
        _cw("b", 0, 200.0, True, 0.5),
        _cw("b", 1, 60.0, True, 0.5),              # false alarm
    ]
    crit0 = [w.signal / theta < tau for w in ws]
    res = v1.b_prime_point(ws, theta=theta, tau_crit=tau, cut=1.5, crit0=crit0, crit2=[False] * len(ws))
    assert res["severe_windows"] == 2
    assert res["nodwell"]["recall_severe"] == 0.5 and res["nodwell"]["false_alarm"] == 0.5
    assert res["nodwell"]["recall_all"] == 0.5
    assert res["violations_by_band"] == {"critical": 0.5, "low": 0.25, "healthy_side_z_ge_1": 0.25}
    assert res["missed_caught_by_slow_loop"] == 0.5
    boot = v1.b_prime_boot(ws, theta=theta, tau_crit=tau, cut=1.5, crit=crit0, n=50)
    assert boot["recall_severe_ci95"][0] is not None

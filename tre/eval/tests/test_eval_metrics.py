"""Metric definitions of tre_eval on a tiny synthetic run (no cluster, no network)."""

import json
import os

import pytest

pytest.importorskip("numpy")  # tre/eval/requirements.txt; the TRE service images do not need it

from tre_eval import load as L  # noqa: E402
from tre_eval import metrics as M  # noqa: E402

SLO = {"ttft_p95_ms": 500.0, "tpot_p95_ms": 75.0, "e2e_p95_ms": 12000.0, "ttft_idle_c_ms": 40.0,
       "ttft_idle_b_ms_per_token": 0.1, "ttft_slo_mode": "slowdown", "ttft_slowdown_k": 5.0, "ttft_floor_ms": 500.0}
T0 = 1_000_000.0


def req(i, model="m-a", t=40.0, ttft=0.1, e2e=5.0, out=400, inp=500, ok=True, cont=None):
    return {"request_id": f"r{i}", "model_name": model, "timestamp": t, "start_time": T0 + t,
            "end_time": T0 + t + e2e, "e2e_latency": e2e, "ttft": ttft, "tpot": None, "input_tokens": inp,
            "output_tokens": out, "success": ok, "http_status": 200 if ok else 503, "tre_continued": cont}


# ---------------------------------------------------------------- SLO / V_req
def test_ttft_threshold_is_max_of_floor_and_slowdown():
    assert M.ttft_threshold_ms(SLO, 500) == 500.0          # 5 * (40 + 50) = 450 < floor
    assert M.ttft_threshold_ms(SLO, 2000) == pytest.approx(1200.0)  # 5 * (40 + 200)
    assert M.ttft_threshold_ms({**SLO, "ttft_slo_mode": "fixed"}, 2000) == 500.0


@pytest.mark.parametrize("kw,viol,what", [
    ({}, False, "nominal"),
    ({"ttft": 0.6}, True, "v_ttft"),                        # 600 ms > 500 ms
    ({"ttft": 0.1, "e2e": 0.1 + 0.076 * 399}, True, "v_tpot"),  # (e2e - ttft)/(out - 1) = 76 ms > 75
    ({"e2e": 149.0, "ttft": 0.1}, True, "censored"),         # route-timeout cut, recorded as success
    ({"ok": False, "ttft": None}, True, "fail"),
])
def test_violation_components(kw, viol, what):
    s = M.score_request(req(0, **kw), SLO)
    assert s["viol"] is viol
    if what != "nominal" and what != "fail":
        assert s[what]


def test_e2e_slo_only_in_with_e2e_variant():
    s = M.score_request(req(0, e2e=20.0, ttft=0.1, out=400), SLO)  # TPOT 49.9 ms, e2e 20 s > 12 s
    assert not s["viol"] and s["viol_with_e2e"]


def test_trim_and_summary_counts():
    rows = [req(0, t=0.0), req(1, t=10.0, ttft=0.9), req(2, t=40.0), req(3, t=41.0, ttft=0.9), req(4, t=42.0, ok=False, ttft=None)]
    recs, summ = M.score_requests(rows, {"m-a": SLO}, trim_s=30.0)
    a = summ["ALL"]
    assert a["trimmed"] == 2 and a["n"] == 3 and a["fail"] == 1 and a["viol"] == 2
    assert a["V_req_pct"] == pytest.approx(66.667)
    assert a["good"] == 1 and a["slo_attainment_pct"] == pytest.approx(33.333)
    assert sum(1 for r in recs if r["trimmed"]) == 2


def test_percentile_matches_score_pilot_interpolation():
    xs = [1, 2, 3, 4, 10]
    assert M.pct(xs, 50) == 3
    assert M.pct(xs, 95) == pytest.approx(8.8)   # k = 3.8 -> 4 + 0.8 * 6
    assert M.pct([None, 5], 99) == 5
    assert M.pct([], 50) is None


# ---------------------------------------------------------------- synthetic arm directory
def write_arm(d, layout, requests, traces=None, decisions=None, signal=None, registry_models=("m-a", "m-b")):
    os.makedirs(os.path.join(d, "client"), exist_ok=True)
    with open(os.path.join(d, "client", "performance_metrics.json"), "w") as fh:
        for r in requests:
            fh.write(json.dumps(r) + "\n")
    if traces is not None:
        with open(os.path.join(d, "client", "traces.json"), "w") as fh:
            json.dump(traces, fh)
    with open(os.path.join(d, "load_start_epoch"), "w") as fh:
        fh.write(str(int(T0)))
    with open(os.path.join(d, "load_end_epoch"), "w") as fh:
        fh.write(str(int(T0 + 100)))
    reg = "models:\n" + "".join(
        f"- name: {m}\n  tp_size: 1\n  slo: {json.dumps(SLO)}\n  trs: {{tau_crit: 0.6, tau_low: 1.0, tau_high: 1.9}}\n"
        for m in registry_models)
    with open(os.path.join(d, "live-registry.yaml"), "w") as fh:
        fh.write(reg)
    with open(os.path.join(d, "layout.jsonl"), "w") as fh:
        for ts, models in layout:
            fh.write(json.dumps({"ts": T0 + ts, "models": models}) + "\n")
    if decisions:
        p = os.path.join(d, "baseline", "pod-x")
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, "decisions-x-1.jsonl"), "w") as fh:
            for x in decisions:
                fh.write(json.dumps(x) + "\n")
    if signal:
        with open(os.path.join(d, "signal_log.jsonl"), "w") as fh:
            for x in signal:
                fh.write(json.dumps(x) + "\n")


def lay(a_awake, b_awake, b_hidden=()):
    return {"m-a": {"awake": list(a_awake), "hidden": []}, "m-b": {"awake": list(b_awake), "hidden": list(b_hidden)}}


A0, A1 = "m-a/n1/0", "m-a/n1/1"
B0, B23 = "m-b/n2/0", "m-b/n2/2,3"


@pytest.fixture
def arm_dir(tmp_path):
    """m-a: 2 replicas until t=28 (donor), then 1. m-b: rate steps 1 -> 10 req/s at t=20, decision at t=22,
    a TP-2 replica wakes at t=30 hidden until t=35 (SafeScale-style), so routable lags awake by 5 s."""
    layout = []
    for t in range(-5, 101):
        a = [A0, A1] if t < 28 else [A0]
        b = [B0] if t < 30 else [B0, B23]
        layout.append((float(t), lay(a, b, b_hidden=[B23] if 30 <= t < 35 else [])))
    traces, rows = [], []
    i = 0
    for t10 in range(0, 1000):  # 0.1 s grid over 100 s
        t = t10 / 10
        rate = 1 if t < 20 else 10
        if (t10 % (10 // rate)) == 0:
            traces.append({"request_id": f"r{i}", "timestamp": t, "model_name": "m-b", "max_output_tokens": 400})
            rows.append(req(i, model="m-b", t=t, ttft=0.9 if 20 <= t < 35 else 0.1))
            i += 1
    dec = [{"ts_ms": int((T0 + 22) * 1000), "model": "m-b", "action": "up", "clamped": 2, "raw_desired": 2, "awake": 1},
           {"ts_ms": int((T0 + 10) * 1000), "model": "m-b", "action": "none", "clamped": 1, "awake": 1}]
    d = str(tmp_path / "trace-x" / "chiron")
    write_arm(d, layout, rows, traces=traces, decisions=dec)
    return d


def test_gpu_seconds_and_routable(arm_dir):
    arm = L.load_arm(arm_dir)
    _, summ = M.score_requests(arm.requests, arm.slo, t_ref=arm.t_load)
    g = M.gpu_accounting(arm, summ)
    # m-a: 2 GPUs over [0, 28) + 1 GPU over [28, 100] = 56 + 72; m-b: 1 GPU, + 2 GPUs (TP 2) from t=30
    assert g["m-a"]["gpu_s"] == pytest.approx(128.0)
    assert g["m-b"]["gpu_s"] == pytest.approx(100 + 2 * 70)
    assert g["m-b"]["hidden_gpu_s"] == pytest.approx(10.0)       # 2 GPUs hidden for 5 s
    assert g["ALL"]["gpu_s"] == pytest.approx(368.0)
    assert g["ALL"]["mean_gpus"] == pytest.approx(3.68)
    assert g["ALL"]["gpu_s_per_good_req"] == pytest.approx(368.0 / summ["ALL"]["good"], rel=1e-3)


def test_onset_latency_breakdown(arm_dir):
    arm = L.load_arm(arm_dir)
    recs, _ = M.score_requests(arm.requests, arm.slo, trim_s=0, t_ref=arm.t_load)
    ons = M.detect_onsets(arm)
    b = [o for o in ons if o["model"] == "m-b"]
    assert len(b) == 1 and abs(b[0]["t_on"] - 20) <= 2
    row = next(r for r in M.onset_latencies(arm, recs, ons) if r["model"] == "m-b")
    t_on = row["t_on"]
    assert row["t_decision"] == pytest.approx(22.0)
    assert row["t_awake"] == pytest.approx(30.0)
    assert row["t_routable"] == pytest.approx(35.0)
    assert row["t_donor"] == pytest.approx(28.0) and row["donor_model"] == "m-a"
    assert row["lat_awake_s"] == pytest.approx(30.0 - t_on)
    assert row["decision_to_awake_s"] == pytest.approx(8.0)
    assert row["t_slo_ok"] == pytest.approx(40.0)   # [30, 40) still has 0.9 s TTFTs until 35; [40, 50) is clean
    assert M.decision_source(arm) == "baseline"
    assert [d["dir"] for d in M.decisions(arm)] == [1]  # 'none' rows are not decisions


def test_decision_points_t7_style():
    rows = [{"model": "m-b", "t_on": 20, "donor_model": "m-a", "max_awake_in_phase": 2, "awake_at_onset": 1},
            {"model": "m-c", "t_on": 80, "donor_model": None, "max_awake_in_phase": 2, "awake_at_onset": 2}]
    pts = [{"surge_model": "m-b", "expect": {"donor": "m-x"}}, {"surge_model": "m-c", "expect": {"no_move": True}}]
    res = M.decision_points(rows, pts)
    assert [(r["observed"], r["pass"]) for r in res] == [("m-a", False), ("no_move", True)]


def test_paired_bootstrap_identical_runs_is_zero():
    rows = [req(i, t=40 + i, ttft=0.9 if i % 3 == 0 else 0.1) for i in range(120)]
    recs, _ = M.score_requests(rows, {"m-a": SLO}, trim_s=0)
    p = M.paired_bootstrap(recs, recs, "ALL", reps=50)
    assert p["n_pairs"] == 120 and p["d_vreq_pp"] == 0 and p["d_vreq_ci"] == (0, 0)
    ci = M.block_bootstrap(recs, "ALL", {"v": M.vreq_stat}, reps=50)["v"]
    assert ci[0] <= 33.34 and ci[1] >= 33.33


def test_seed_ci():
    assert M.seed_ci([1.0])["ci"] == (None, None)
    r = M.seed_ci([1.0, 3.0])
    assert r["mean"] == 2.0 and r["ci"][0] < 2.0 < r["ci"][1]


def test_clock_offset_shifts_sidecar_events(arm_dir):
    sc = os.path.join(arm_dir, "sidecar")
    os.makedirs(sc)
    with open(os.path.join(sc, "m-b-n2-gpu-0-abc.log"), "w") as fh:
        fh.write(json.dumps({"event": "tre_reissue", "ts": T0 + 200.0, "kind": "continue", "gap_ms": 900}) + "\n")
    arm = L.load_arm(arm_dir, clock_offsets={"n2": 160.0})
    assert arm.sidecar[0]["ts"] == pytest.approx(T0 + 40.0)


def test_report_end_to_end_with_missing_optional_inputs(arm_dir, tmp_path):
    pytest.importorskip("matplotlib")
    from tre_eval import report
    out = str(tmp_path / "rep")
    rc = report.main([arm_dir, "--out", out, "--reps", "20"])
    assert rc == 0
    assert os.path.exists(os.path.join(out, "index.html"))
    assert os.path.exists(os.path.join(out, "arms", "chiron", "fig_gpu_map.png"))
    s = json.load(open(os.path.join(out, "arms", "chiron", "summary.json")))
    assert any("pod_gauges" in w for w in s["warnings"])  # absent input -> warning, not a crash

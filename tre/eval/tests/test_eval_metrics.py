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
    with open(os.path.join(out, "arms", "chiron", "timeseries_1s.csv")) as fh:
        assert "kv_mean_all" not in fh.readline()  # no newer collector -> old columns only


# ---------------------------------------------------------------- newer collectors
def _utc(ts):
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts)) + f",{int(round(ts % 1 * 1000)):03d}"


def _jl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write((r if isinstance(r, str) else json.dumps(r)) + "\n")


def add_new_collectors(d):
    """n2 runs 10 s ahead; SM and Redis run on n2; sidecar pod 'pod-b23' (no node in its name) on n2."""
    with open(os.path.join(d, "clock_offsets.json"), "w") as fh:
        json.dump({"ref_host": "runner", "method": "ssh-date", "offsets_s": {"n1": 0.0, "n2": 10.0},
                   "start": {"ts": T0, "nodes": {"n2": {"offset_s": 10.0, "rtt_s": 0.01}}},
                   "end": {"ts": T0 + 100, "nodes": {"n2": {"offset_s": 10.2, "rtt_s": 0.01}}}}, fh)
    pods = [{"namespace": "tre-v2", "name": "sm-0", "component": "service-manager", "node": "n2", "phase": "Running",
             "containers": [{"name": "sm", "image": "tre-v2-sm:x", "image_id": "docker://sha256:aa"}]},
            {"namespace": "default", "name": "pod-b23", "component": "model", "node": "n2", "phase": "Running", "containers": []}]
    with open(os.path.join(d, "components.json"), "w") as fh:
        json.dump({"component_nodes": {"service-manager": "n2", "redis": "n2", "controller": "n2"},
                   "image_ids_by_node": {"n2": {"tre-v2-sm:x": "sha256:aa"}, "n1": {"vllm:x": "sha256:bb"}},
                   "start": {"ts": T0, "pods": pods}, "end": {"ts": T0 + 100, "pods": pods}}, fh)
    # B23 hidden over [30, 35): still decoding, label routable=false
    pm = []
    for t in range(25, 40):
        p = {"pod-b0": {"model": "m-b", "node": "n2", "routable_label": "true", "routable": True, "sm_awake": True,
                        "sm_hidden": False, "kv_cache_usage_perc": 0.2, "num_requests_running": 2,
                        "num_requests_waiting": 0, "generation_tokens_total": 100.0 * t, "num_preemptions_total": 0}}
        if t >= 30:
            p["pod-b23"] = {"model": "m-b", "node": "n2", "routable_label": "false" if t < 35 else "true",
                            "routable": t >= 35, "sm_awake": True, "sm_hidden": t < 35, "kv_cache_usage_perc": 0.8,
                            "num_requests_running": 5, "num_requests_waiting": 1, "generation_tokens_total": 0.0,
                            "num_preemptions_total": 0}
        pm.append({"ts": T0 + t, "pods": p})
    _jl(os.path.join(d, "pod_metrics_1s.jsonl"), pm)
    _jl(os.path.join(d, "sm.log"), [f'{_utc(T0 + 60)} INFO tre_sm.ops: wake_done {{"binding_id": "m-b/n2/0", "phases_ms": {{"wake_up": 1500}}}}'])
    _jl(os.path.join(d, "sm.ts.log"), ['1970-01-12T13:47:40.500000000Z INFO:     10.0.0.1:5 - "PUT /v2/models/m-b/target HTTP/1.1" 200 OK',
                                       '1970-01-12T13:47:41.000000000Z INFO:     10.0.0.1:5 - "GET /v2/state HTTP/1.1" 200 OK'])
    _jl(os.path.join(d, "sidecar", "pod-b23.log"),
        [{"event": "tre_reissue", "ts": T0 + 10 + 48.0, "abort_ts": T0 + 10 + 46.0, "request_id": "r250",
          "kind": "continue", "reason": "sleep", "gap_ms": 700}])
    _jl(os.path.join(d, "baseline", "bl_req_events.m-b.jsonl"),
        [{"id": f"{int((T0 + 10 + 43.0) * 1000)}-0", "kind": "arr", "req_id": "r250", "pod": "pod-b0"},
         {"id": f"{int((T0 + 10 + 44.0) * 1000)}-0", "kind": "done", "req_id": "r250", "pod": "pod-b0"},
         {"id": f"{int((T0 + 10 + 44.0) * 1000)}-1", "kind": "arr", "req_id": "a-gateway-uuid", "pod": "pod-b0"}])
    _jl(os.path.join(d, "gpu_map.jsonl"),  # t=50 agrees with the layout; t=60 claims m-a on n1/1 (asleep since 28)
        [{"ts": T0 + 50, "map": {"n1/0": {"awake": [A0], "hidden": []}, "n2/0": {"awake": [B0], "hidden": []},
                                 "n2/2": {"awake": [B23], "hidden": []}, "n2/3": {"awake": [B23], "hidden": []}}},
         {"ts": T0 + 60, "map": {"n1/0": {"awake": [A0]}, "n1/1": {"awake": [A1]}, "n2/0": {"awake": [B0]},
                                 "n2/2": {"awake": [B23]}, "n2/3": {"awake": [B23]}}}])
    _jl(os.path.join(d, "gpu_truth.jsonl"),  # n1 gpu 3: 30 GB used, no binding there (gpu 0 holds m-a)
        [{"ts": T0 + t, "nodes": {"n1": {"node": "n1", "gpus": [{"used_mib": 30000}, {"used_mib": 400}, {"used_mib": 400}, {"used_mib": 30000}]}}}
         for t in range(40, 45)])


def test_pod_metrics_hidden_pods_only_in_all_columns(arm_dir):
    from tre_eval import timeseries as T
    add_new_collectors(arm_dir)
    arm = L.load_arm(arm_dir)
    ts = T.build(arm, [])
    j = list(ts["_t"]["t"]).index(32.0)
    b = ts["m-b"]
    assert (b["kv_mean"][j], b["running"][j], b["pods_sampled"][j]) == (0.2, 2, 1)   # routable pods only
    assert b["kv_mean_all"][j] == pytest.approx(0.5) and b["running_all"][j] == 7   # + the hidden pod
    assert b["pods_hidden_sampled"][j] == 1 and b["routable_label"][j] == 1
    assert b["gen_tok_s_engine"][j] == pytest.approx(100.0)                     # counter delta / 1 s


def test_components_v2_sets_node_offsets_and_image_ids(arm_dir):
    from tre_eval import report
    add_new_collectors(arm_dir)
    arm = L.load_arm(arm_dir)
    wake = next(e for e in arm.sm_events if e["kind"] == "wake_done")
    assert wake["ts"] == pytest.approx(T0 + 50.0)                 # SM node n2 is 10 s ahead
    assert arm.sm_access[0]["ts"] == pytest.approx(T0 + 50.5)     # kubelet stamp, same correction
    assert arm.sidecar[0]["abort_ts"] == pytest.approx(T0 + 46.0)  # pod-b23 -> n2 via components.json pods
    man = report.manifest(arm)
    assert man["image_ids_by_node"]["n1"]["vllm:x"] == "sha256:bb"
    assert man["clock"]["drift_s"]["n2"] == pytest.approx(0.2)


def test_trs_now_ms_is_the_tre_decision_time(tmp_path):
    from tre_eval import timeseries as T
    d = str(tmp_path / "trace-y" / "tre")
    write_arm(d, [(float(t), lay([A0], [B0])) for t in range(-5, 101)], [req(i, model="m-b", t=float(i)) for i in range(90)])
    msg = {"event": "trs_calc_result", "ts_ms": str(int((T0 + 20) * 1000)), "now_ms": str(int((T0 + 22.4) * 1000)),
           "loop": "rescue", "submitted": 1, "actions": json.dumps([{"model": "m-b", "delta": 1, "reason": "crit"}])}
    _jl(os.path.join(d, "controller.log"), [{"message": json.dumps(msg)}])
    arm = L.load_arm(d)
    assert arm.ctrl_ticks[0]["ts"] == pytest.approx(T0 + 20)   # window boundary kept for signal columns
    assert [x["ts"] for x in M.decisions(arm)] == [pytest.approx(T0 + 22.4)]
    ev = [e for e in T.events(arm) if e["kind"] == "decision_up"]
    assert ev[0]["t"] == pytest.approx(22.4)


def test_report_joins_request_ids_from_sidecar_and_gateway(arm_dir, tmp_path):
    pytest.importorskip("matplotlib")
    import csv
    from tre_eval import report
    add_new_collectors(arm_dir)
    out = str(tmp_path / "rep")
    assert report.main([arm_dir, "--out", out, "--reps", "20"]) == 0
    a = os.path.join(out, "arms", "chiron")
    rows = {r["request_id"]: r for r in csv.DictReader(open(os.path.join(a, "requests.csv")))}
    assert rows["r250"]["sidecar_continuations"] == "1" and rows["r251"]["sidecar_continuations"] == "0"
    assert float(rows["r250"]["gw_arr_t"]) == pytest.approx(43.0)   # Redis on n2, 10 s ahead
    v = json.load(open(os.path.join(out, "validation.json")))["chiron"]
    assert v["gateway_join"]["joined"] == 1 and v["gateway_join"]["gw_unmatched"] == 1
    assert v["gpu_truth_vs_layout"]["gpu_busy_without_awake_binding_s"] == {"n1/3": 5}
    assert (v["gpu_map_vs_layout"]["snapshots_compared"], v["gpu_map_vs_layout"]["mismatched"]) == (2, 1)
    s = json.load(open(os.path.join(a, "summary.json")))
    assert s["interruption"]["added_latency_method"] == "exact_request_id_join"
    ev = list(csv.DictReader(open(os.path.join(a, "events.csv"))))
    assert [e["kind"] for e in ev if e["source"] == "sm_api"] == ["PUT /v2/models/{model}/target"]

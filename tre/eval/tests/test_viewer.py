"""tre_eval.viewer: the embedded data follows the report files, and missing inputs degrade to warnings."""

import csv
import json
import os
import shutil
import subprocess

import pytest

pytest.importorskip("numpy")

from tre_eval import viewer as V  # noqa: E402
from test_eval_metrics import arm_dir, T0  # noqa: E402,F401  (fixture reuse)

ASSETS = os.path.join(os.path.dirname(V.__file__), "viewer_assets")


def _page_data(path):
    with open(path, encoding="utf-8") as fh:
        page = fh.read()
    head = '<script id="data" type="application/json">'
    i = page.index(head) + len(head)
    blob = page[i:page.index("</script>", i)]
    for ph in ("/*__", "__DATA_JSON__", "__TITLE__"):
        assert ph not in page, f"placeholder {ph} left in the page"
    return json.loads(blob), page


def _set_manifest(report_dir, sub, **kw):
    p = os.path.join(report_dir, "arms", sub, "manifest.json")
    man = json.load(open(p))
    man.update(kw)
    json.dump(man, open(p, "w"))


@pytest.fixture
def report_dir(arm_dir, tmp_path):
    pytest.importorskip("matplotlib")
    from tre_eval import report
    # a SafeScale probe on the hidden TP-2 binding, opened at the scale-up decision (t = 22), and one probe
    # left over from an earlier arm (old safescale.json files hold every probe of the campaign)
    probe = {"model": "m-b", "start_wall_ms": (T0 + 22.0) * 1000, "resolved_ts": T0 + 35.0, "resolution": "commit",
             "status": "resolved", "pods": ["m-b-n2-gpu-2-3-abc-xyz"], "terminal_reason": "formal_commit_gate_passed",
             "hide_anchor": {"controller_ts_ms": (T0 + 30.0) * 1000, "ts_ms": (T0 + 40.0) * 1000},
             "terminal_details": {"idle_commit": True}}
    old = {**probe, "start_wall_ms": (T0 - 900.0) * 1000, "resolved_ts": T0 - 880.0}
    with open(os.path.join(arm_dir, "safescale.json"), "w") as fh:
        json.dump({"probes": {"m-b-1": json.dumps(probe), "m-b-0": json.dumps(old)}}, fh)
    out = str(tmp_path / "rep")
    assert report.main([arm_dir, "--out", out, "--reps", "20"]) == 0
    return out


def test_report_writes_viewer_consistent_with_its_csvs(report_dir):
    data, _ = _page_data(os.path.join(report_dir, "viewer.html"))  # written by the report.py hook
    assert "viewer.html" in open(os.path.join(report_dir, "index.html")).read()
    arm = data["arms"][0]
    a_dir = os.path.join(report_dir, "arms", arm["sub"])
    ts = arm["ts"]
    rows = list(csv.DictReader(open(os.path.join(a_dir, "timeseries_1s.csv"))))
    assert rows
    for r in rows:  # replica step series == timeseries awake / routable, second by second
        f = ts["models"][r["model"]]
        i = int(r["t"]) - ts["t0"]
        for col in ("awake", "routable"):
            want = float(r[col]) if r[col] != "" else None
            assert f[col][i] == want, (r["model"], r["t"], col)
    gpu_rows = list(csv.DictReader(open(os.path.join(a_dir, "gpu_intervals.csv"))))
    assert gpu_rows
    for r in gpu_rows:  # every GPU interval is in the Gantt data
        ivs = arm["gpus"][r["gpu"]]
        assert any(iv[3] == r["binding"] and iv[0] == pytest.approx(float(r["t0"]), abs=1e-3)
                   and iv[1] == pytest.approx(float(r["t1"]), abs=1e-3) for iv in ivs), r
    assert arm["decision_source"] == "baseline" and arm["probes"] == []  # probes belong to TRE arms only
    assert {e["kind"] for e in arm["events"]} <= {"decision_up", "decision_down", "wake", "sleep", "refusal", "counterfactual"}


def test_fragmentation_is_layout_only_and_equal_for_equal_layouts(report_dir):
    nodes = [{"name": "n", "gpus": 4, "pairs": [[0, 1], [2, 3]], "short": "n"}]
    # GPUs 1 and 2 busy for t < 10: 0 and 3 are free but not a slot; from t = 10 GPU 2 frees -> slot (2, 3) free
    gpus = {"n/1": [[0, 20, "small", "small/n/1", [], False]], "n/2": [[0, 10, "small", "small/n/2", [], False]]}
    assert V.fragmentation(0, 20, gpus, nodes, [1, 2]) == [{"tp": 2, "t0": 0, "t1": 10, "free": 2}]
    # a second arm with the same layout but a different target series: same stranded seconds
    src = os.path.join(report_dir, "arms", "chiron")
    dst = os.path.join(report_dir, "arms", "twin")
    shutil.copytree(src, dst)
    rows = list(csv.DictReader(open(os.path.join(dst, "timeseries_1s.csv"))))
    for r in rows:
        r["target"] = "4"
    with open(os.path.join(dst, "timeseries_1s.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    _set_manifest(report_dir, "twin", arm="twin", decision_source="tre")
    reg = os.path.join(json.load(open(os.path.join(src, "manifest.json")))["dir"], "live-registry.yaml")
    with open(reg) as fh:
        text = fh.read()
    with open(reg, "w") as fh:  # m-b binds two GPUs (m-b/n2/2,3): TP 2
        fh.write(text.replace("- name: m-b\n  tp_size: 1", "- name: m-b\n  tp_size: 2"))
    data = V.build(report_dir)
    secs = {a["name"]: sum(f["t1"] - f["t0"] for f in a["frag"]) for a in data["arms"]}
    assert secs["chiron"] == secs["twin"] > 0  # m-b (TP 2) is stranded once n2/2,3 are taken and only n1/1, n2/1 are free
    assert any("not in comparison.csv" in w for w in data["warnings"])


def test_probes_tre_only_in_window_on_the_controller_clock(report_dir):
    # controller node 10 s ahead: load.py does not correct controller / scaler stamps, so the viewer must not either
    _set_manifest(report_dir, "chiron", decision_source="tre", clock_offsets_s={"n1": 0.0, "n2": 10.0},
                  component_nodes={"service-manager": "n2", "redis": "n2", "controller": "n2"})
    arm = V.build(report_dir)["arms"][0]
    assert len(arm["probes"]) == 1  # the earlier arm's probe is dropped ...
    assert any("1 probe(s) outside" in w for w in arm["warnings"])  # ... and counted
    assert any(w.startswith("controller runs on n2") for w in arm["warnings"])
    p = arm["probes"][0]
    assert (p["model"], p["outcome"], p["reason"], p["binding"]) == ("m-b", "commit", "idle_commit", "m-b/n2/2,3")
    up = next(e for e in arm["events"] if e["kind"] == "decision_up" and e["model"] == "m-b")
    assert p["t0"] == pytest.approx(up["t"])  # the probe starts at the decision that opened it
    assert p["t_hide"] == pytest.approx(30.0) and p["t1"] == pytest.approx(35.0)


def test_embedded_json_cannot_break_out_of_the_script_element():
    hostile = "</script><!--<script>alert(1)</script> & \u2028"
    blob = V.embed_json({"s": hostile, "nan": float("nan"), "deep": [{"inf": float("inf")}]})
    assert not set("<>&") & set(blob)
    assert json.loads(blob) == {"s": hostile, "nan": None, "deep": [{"inf": None}]}


def test_missing_optional_inputs_give_warnings_not_errors(report_dir, tmp_path):
    rep = str(tmp_path / "bare")
    shutil.copytree(report_dir, rep)
    a_dir = os.path.join(rep, "arms", os.listdir(os.path.join(rep, "arms"))[0])
    for f in os.listdir(a_dir):
        if f != "timeseries_1s.csv":
            os.remove(os.path.join(a_dir, f))
    for f in ("comparison.csv", "validation.json", "onsets.csv", "decision_points.csv"):
        os.remove(os.path.join(rep, f))
    out = V.write(rep, os.path.join(rep, "v.html"), raw_root=str(tmp_path / "nowhere"))
    data, _ = _page_data(out)
    warn = " ".join(data["warnings"] + data["arms"][0]["warnings"])
    for what in ("comparison.csv", "events.csv", "gpu_intervals.csv", "requests.csv"):
        assert what in warn
    assert data["arms"][0]["ts"]["n"] > 0 and data["arms"][0]["events"] == []


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_page_js_parses_and_core_matches_the_export(report_dir, tmp_path):
    for name in ("viewer_core.js", "viewer_app.js"):
        r = subprocess.run(["node", "--check", os.path.join(ASSETS, name)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    data, _ = _page_data(os.path.join(report_dir, "viewer.html"))
    blob = tmp_path / "data.json"
    blob.write_text(json.dumps(data))
    harness = tmp_path / "h.js"
    harness.write_text("""
const assert = require('assert');
const C = require(process.argv[2]);
const D = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));
const st = {v: 3, m: 'm-b', a: ['x', 'y'], x: [10, 42.5], log: false, th: 'dark', k: ['wake']};
assert.deepStrictEqual(C.parseHash(C.serializeHash(st)), st);
const g = C.grid(D.arms);
for (const a of D.arms) for (const m of Object.keys(a.ts.models)) {
  const al = C.aligned(a, m, 'awake', g), src = a.ts.models[m].awake;
  for (let i = 0; i < src.length; i++) assert.strictEqual(al[i + a.ts.t0 - g.lo], src[i]);
}
const bars = C.gpuBars(D.arms[0].gpus);
assert.ok(bars.some((b) => b.gpus.length === 2), 'a TP-2 binding is one bar over two GPUs');
console.log('core ok');
""")
    r = subprocess.run(["node", str(harness), os.path.join(ASSETS, "viewer_core.js"), str(blob)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_scaler_refusals_trust_an_inferred_clock_shift_only_with_a_clear_margin(tmp_path):
    import time
    clock = V._Clock({"load_start_epoch": T0, "clock_offsets_s": {"n1": 0.0, "n2": 10.0}})  # scaler node unknown

    def log(ts):  # scaler on n2: its stamps run 10 s ahead; 0.5 s after the SM answered
        with open(tmp_path / "baseline-scaler.log", "w") as fh:
            for t in ts:
                e = T0 + t + 10.0 + 0.5
                fh.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(e)) + f",{int(e % 1 * 1000):03d} WARNING x.loop "
                         f"SM refused m-b up->2: {{'code': 409, 'reason': 'slot_occupied', 'elapsed_s': 0.1}}\n")
    sm = [{"t": t, "model": "m-b"} for t in (20.0, 40.0, 60.0)]
    log([20.0, 40.0, 60.0])
    warns = []
    ref, sure = V._refusals(str(tmp_path), clock, sm, warns.append)
    assert sure and [r["t"] for r in ref] == [20.0, 40.0, 60.0]  # drawn at the SM time
    assert any("3/3 at -10.0 s" in w and "0/3 at +0.0 s" in w for w in warns)
    # only 2 pairs under the best shift: not a clear margin -> time uncertain, kept out of the time axes
    log([20.0, 40.0])
    ev = V.build_events([{"t": str(t), "source": "sm_api", "kind": "wake", "model": "m-b", "value": "409"} for t in (20, 40)],
                        str(tmp_path), clock)
    ref = [e for e in ev if e["src"] == "baseline"]
    assert len(ref) == 2 and all(e.get("uncertain") for e in ref)
    assert sorted(e["t"] for e in ev if e["src"] == "sm_api") == [20, 40]  # the SM 409 lines are the markers


def test_apa_window_value_follows_the_log_else_the_trailing_mean_and_cr_thresholds(tmp_path):
    import time
    with open(tmp_path / "apa_crs_before.yaml", "w") as fh:  # recorded CR: not the defaults
        fh.write("items:\n- kind: PodAutoscaler\n  metadata:\n    name: m-a-apa\n    labels: {model.aibrix.ai/name: m-a}\n"
                 "    annotations: {apa.autoscaling.aibrix.ai/window: 10s, autoscaling.aibrix.ai/scale-up-tolerance: '0.2',\n"
                 "                  autoscaling.aibrix.ai/scale-down-tolerance: '0.3'}\n"
                 "  spec:\n    metricsSources: [{targetMetric: kv_cache_usage_perc, targetValue: '0.6'}]\n")
    evals = [(10.2, 0.7), (25.5, 0.3), (40.0, 0.5)]
    with open(tmp_path / "aibrix-controller-manager.log", "w") as fh:
        for t, v in evals:
            e = T0 + t
            fh.write(time.strftime("I%m%d %H:%M:%S", time.gmtime(e)) + f".{int(e % 1 * 1e6):06d}       1 client.go:217] "
                     f'"Metrics window aggregation" metricKey="default/m-a-apa/kv_cache_usage_perc" stableAvg={v} panicAvg=0\n')
    kv = [None if i % 7 == 0 else (i % 10) / 10 for i in range(60)]

    def run():
        ts = {"t0": 0, "n": 60, "models": {"m-a": {"kv_mean": list(kv)}}}
        warns = []
        out = V.build_apa_kv(str(tmp_path), V._Clock({"load_start_epoch": T0}), ts, [], warns.append)
        return out, ts["models"]["m-a"]["apa_kv"], warns
    out, series, warns = run()
    c = out["models"]["m-a"]
    assert (out["source"], c["target"], c["up_tol"], c["down_tol"], c["window_s"], c["from"]) == ("logged", 0.6, 0.2, 0.3, 10.0, "CR")
    assert not warns
    held = [None] * 10 + [0.7] * 15 + [0.3] * 15 + [0.5] * 20  # each evaluation held until the next
    assert series == held
    os.remove(tmp_path / "aibrix-controller-manager.log")
    out, series, warns = run()
    assert out["source"] == "approx" and any("approximated" in w for w in warns)
    for i in range(60):  # trailing 10 s mean of the routable cross-pod series
        xs = [x for x in kv[max(0, i - 9):i + 1] if x is not None]
        assert series[i] == (pytest.approx(sum(xs) / len(xs), rel=1e-3) if xs else None)

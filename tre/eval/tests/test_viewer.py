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
    return json.loads(blob.replace("<\\/", "</")), page


@pytest.fixture
def report_dir(arm_dir, tmp_path):
    pytest.importorskip("matplotlib")
    from tre_eval import report
    # a SafeScale probe on the hidden TP-2 binding (controller on the reference clock)
    probe = {"model": "m-b", "start_wall_ms": (T0 + 29.0) * 1000, "resolved_ts": T0 + 35.0, "resolution": "commit",
             "status": "resolved", "pods": ["m-b-n2-gpu-2-3-abc-xyz"], "terminal_details": {"idle_commit": True}}
    with open(os.path.join(arm_dir, "safescale.json"), "w") as fh:
        json.dump({"probes": {"m-b-1": json.dumps(probe)}}, fh)
    out = str(tmp_path / "rep")
    assert report.main([arm_dir, "--out", out, "--reps", "20"]) == 0
    return out


def test_report_writes_viewer_consistent_with_its_csvs(report_dir):
    data, _ = _page_data(os.path.join(report_dir, "viewer.html"))  # written by the report.py hook
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
    p = arm["probes"][0]
    assert (p["model"], p["outcome"], p["reason"], p["binding"]) == ("m-b", "commit", "idle_commit", "m-b/n2/2,3")
    assert p["t0"] == pytest.approx(29.0) and p["t1"] == pytest.approx(35.0)
    assert {e["kind"] for e in arm["events"]} <= {"decision_up", "decision_down", "wake", "sleep", "refusal", "counterfactual"}


def test_fragmentation_needs_free_gpus_but_no_aligned_slot():
    nodes = [{"name": "n", "gpus": 4, "pairs": [[0, 1], [2, 3]], "short": "n"}]
    n = 20
    ts = {"t0": 0, "n": n, "models": {"big": {"awake": [0] * n, "target": [0] * 5 + [1] * 15}}}
    # GPUs 1 and 2 busy for t < 10: 0 and 3 are free but not a slot; from t = 10 GPU 2 frees -> slot (2, 3) free
    gpus = {"n/1": [[0, 20, "small", "small/n/1", [], False]], "n/2": [[0, 10, "small", "small/n/2", [], False]]}
    assert V.fragmentation(ts, gpus, nodes, {"big": 2, "small": 1}) == [{"model": "big", "t0": 5, "t1": 10, "free": 2}]


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

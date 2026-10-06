"""Campaign runner (tre/eval/runner/campaign.py): plan, order, resume, retry, stop, marker.

No cluster: the loop's side effects (pre-checks, reset, the arm runner, reports) are a fake that
writes the files a finished arm leaves behind. The invariants tested:

* each trace's arms run back to back, traces in file order; the in-trace order is a
  Latin-square row (every arm in every position, carry-over balanced) and is recorded;
* resume never reruns an arm with a valid done marker; an interrupted directory is moved aside
  and counts as an attempt;
* a failed / invalid arm is retried once after a full reset, then marked failed, and the
  campaign goes on;
* STOP stops after the current arm (no retry started), a blocked pre-check stops the campaign
  without consuming an attempt;
* the exclusive-window marker only ever gets a later end time on line 1; nothing else changes.
"""
from __future__ import annotations

import io
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "runner"
sys.path.insert(0, str(RUNNER))

import campaign as C  # noqa: E402

yaml = pytest.importorskip("yaml")
TRACES = ["T7_tight_pool", "Alternating", "Simul_spike"]


# ------------------------------------------------------------------ fixtures
def make_trace(root: Path, name: str, seed: int = 1, n: int = 100, duration: float = 200.0) -> None:
    d = root / name / f"seed{seed}"
    d.mkdir(parents=True)
    recs = [{"request_id": f"req_{i:06d}", "timestamp": round(i * duration / n, 6), "model_name": "m7",
             "prompt": f"p{i}", "prompt_length": 10, "phase_type": "tracegen-v2", "max_output_tokens": 5} for i in range(n)]
    (d / "traces_tre.effective.json").write_text(json.dumps(recs))
    (d / "manifest.json").write_text(json.dumps({"trace": name, "seed": seed, "duration_s": duration, "out_cap": {"x": 1},
                                                 "design": {"requests": n, "sha256": "d" * 64}, "generator": {"git_sha": "abc"}}))
    cfg = root / "_cfg" / name / "config.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("custom_load_test:\n  duration_seconds: 1200\n  models: []\n")


BL_CM = """apiVersion: v1
kind: ConfigMap
metadata: {name: tre-v2-baseline-tokenscale, namespace: tre-v2}
data:
  tokenscale.yaml: |
    velocity: {m7: {buckets: [1]}}
---
apiVersion: v1
kind: ConfigMap
metadata: {name: tre-v2-baseline-preserve, namespace: tre-v2}
data:
  preserve.yaml: |
    trace_path: /etc/tre-baselines-traces/Old/traces_tre.effective.json
    trace_match_parts: 2
    window_s: 30
    tier1: oracle_noisy
    noise_sigma: 0.0772
"""


def write_campaign(tmp: Path, traces=TRACES, arms=("tre", "apa", "tokenscale"), extra: dict | None = None) -> Path:
    troot = tmp / "traces"
    for t in set(traces):
        if not (troot / t).exists():
            make_trace(troot, t)
    (tmp / "bl.yaml").write_text(BL_CM)
    (tmp / "bt").mkdir(exist_ok=True)
    doc = {"name": "camp-test", "round": "camp-test", "results_root": str(tmp / "results"), "trace_root": str(troot),
           "loadgen_config_root": str(troot / "_cfg"), "arms": list(arms), "eta": {"overhead_s": 100, "report_s": 10},
           "env": {"MARKER": str(tmp / "MARKER"), "BL_CM_FILE": str(tmp / "bl.yaml")},
           "preserve": {"trace_host_root": str(tmp / "bt"), "trace_pod_root": "/etc/tre-baselines-traces", "match_parts": 3},
           "report": {"enabled": True}, "traces": [{"trace": t} for t in traces]}
    doc = C.deep_merge(doc, extra or {})
    p = tmp / "campaign.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


class FakeDeps:
    """Records calls; ``script`` maps an arm key to a list of per-attempt outcomes
    (``ok`` | ``rc`` | ``emfile``); ``on_run`` is called after each arm (e.g. to touch STOP)."""

    def __init__(self, script=None, precheck_ok=True, on_run=None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.precheck_ok = precheck_ok
        self.on_run = on_run
        self.calls: list = []

    def precheck(self, need_baseline):
        self.calls.append(("precheck",))
        if self.precheck_ok:
            return True, []
        return False, [{"name": "marker", "ok": False, "detail": "absent", "kind": "hard"}]

    def reset(self, force):
        self.calls.append(("reset", force))
        return 0

    def run_arm(self, item, env, attempt):
        self.calls.append(("run", item.key, attempt, env["ARM_OUT_DIR"], env.get("BL_CM_FILE")))
        outcome = (self.script.get(item.key) or ["ok"]).pop(0) if self.script.get(item.key) else "ok"
        d = Path(item.arm_dir)
        d.mkdir(parents=True, exist_ok=True)
        rc = 0
        if outcome != "rc":
            (d / "client").mkdir(exist_ok=True)
            (d / "client" / "performance_metrics.json").write_text("[]")
            (d / "score.json").write_text(json.dumps({"models": {"ALL": {"n": 90, "trimmed": 10, "fail": 0,
                                                                         "V_req_pct": 1.0, "max_tokens_hit_frac": 1.0}}}))
            for f in C.COLLECTOR_FILES:
                (d / f).write_text("x\n")
            (d / "EMFILE_PODS").write_text("sidecar/pod-a.log\n" if outcome == "emfile" else "")
            if item.base_arm in C.BASELINE_ARMS:
                (d / "baseline").mkdir(exist_ok=True)
                (d / "baseline" / "run_validity.json").write_text(json.dumps({"events_valid": True}))
        else:
            rc = 4
        if self.on_run:
            self.on_run(item)
        return rc

    def report(self, item, dirs):
        self.calls.append(("report", item.key, tuple(Path(x).name for x in dirs)))

    def sleep(self, s):
        self.calls.append(("sleep", s))

    def runs(self):
        return [c[1] for c in self.calls if c[0] == "run"]


def campaign(tmp, deps, **kw):
    c = C.load_campaign(write_campaign(tmp, **kw.pop("cfg", {})))
    return C.Campaign(c, deps=deps, rec=C.Recorder(c, quiet=True), **kw)


# ------------------------------------------------------------------ order
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_williams_square_is_position_and_carryover_balanced(n):
    rows = C.williams_rows(n)
    assert len(rows) == (n if n % 2 == 0 else 2 * n)
    assert all(sorted(r) == list(range(n)) for r in rows)
    for pos in range(n):
        assert len(set(Counter(r[pos] for r in rows).values())) == 1
    pairs = Counter((r[i], r[i + 1]) for r in rows for i in range(n - 1))
    assert set(pairs) == {(a, b) for a, b in itertools.permutations(range(n), 2)}
    assert len(set(pairs.values())) == 1


def test_final_campaign_plan_order_and_rotation(tmp_path):
    """The shipped campaign-final.yaml: 9 traces x 5 arms, then 3 PreServe variants on Alternating;
    arms of a trace contiguous, traces in the file's order, every arm in every position."""
    path = RUNNER / "campaigns" / "campaign-final.yaml"
    c = C.load_campaign(path, environ={"TRE_REPO": str(tmp_path)})
    plan = C.build_plan(c)
    names = [t["trace"] for t in yaml.safe_load(path.read_text())["traces"]]
    assert names[:9] == ["T7_tight_pool", "Alternating", "Simul_spike", "Sinusoidal", "Decode_burst", "Prefill_mix",
                         "Real_code", "Real_conv", "Steady"] and names[9] == "Alternating"
    assert len(plan) == 9 * 5 + 3
    blocks = [list(g) for _, g in itertools.groupby(plan, key=lambda i: i.entry)]
    assert [b[0].trace for b in blocks] == names
    for b in blocks[:9]:
        assert sorted(i.arm for i in b) == sorted(C.BASE_ARMS)
        assert [i.position for i in b] == [1, 2, 3, 4, 5] and all(i.order == [x.arm for x in b] for i in b)
    for arm in C.BASE_ARMS:
        assert {i.position for i in plan[:45] if i.arm == arm} == {1, 2, 3, 4, 5}
    assert [i.base_arm for i in blocks[9]] == ["preserve"] * 3 and blocks[9][0].out_name == "Alternating_s1"
    assert {i.label for i in blocks[9]} == {"PreServe-noisy-0.30", "PreServe-phase-shift", "PreServe-last-window"}
    assert all(i.arm_dir.endswith(f"/{i.out_name}/{i.arm}") for i in plan)


def test_smoke_campaign_is_a_t7_slice_with_two_arms(tmp_path):
    c = C.load_campaign(RUNNER / "campaigns" / "campaign-smoke.yaml", environ={"TRE_REPO": str(tmp_path)})
    (e,) = C.resolve_traces(c)
    assert e.trace == "T7_tight_pool" and e.slice == {"start_s": 100.0, "duration_s": 120.0} and e.duration_s == 120
    assert sorted(i.arm for i in C.build_plan(c)) == ["tokenscale", "tre"]
    assert e.client_dir.startswith(c["results_root"])        # the slice is a copy, never the frozen trace


def test_duplicate_arm_of_a_trace_is_refused(tmp_path):
    p = write_campaign(tmp_path, traces=["T7_tight_pool", "T7_tight_pool"])
    with pytest.raises(C.CampaignError, match="planned twice"):
        C.build_plan(C.load_campaign(p))


# ------------------------------------------------------------------ run / resume / retry / stop
def test_full_run_then_resume_skips_done_arms(tmp_path):
    deps = FakeDeps()
    camp = campaign(tmp_path, deps)
    assert camp.run() == 0
    keys = [i.key for i in camp.plan]
    assert deps.runs() == keys                                       # back to back, in plan order
    st = json.loads((Path(camp.root) / "status.json").read_text())
    assert st["state"] == "done" and st["counts"]["done"] == len(keys)
    man = json.loads((Path(camp.plan[1].arm_dir) / "campaign_manifest.json").read_text())
    assert man["order_position"] == 2 and man["order"] == camp.plan[1].order and man["validity"]["verdict"] == "valid"
    # after every arm: post-arm reset (not forced) then the reports of that arm + its trace
    assert ("reset", False) in deps.calls and ("report", keys[0], (camp.plan[0].arm,)) in deps.calls
    deps2 = FakeDeps()
    assert campaign(tmp_path, deps2).run() == 0
    assert deps2.runs() == []


def test_invalid_arm_retried_once_after_full_reset_then_failed_and_campaign_continues(tmp_path):
    deps = FakeDeps()
    camp = campaign(tmp_path, deps, cfg={"arms": ["tre", "apa"]})
    first_key = camp.plan[0].key
    deps.script = {first_key: ["emfile", "rc"]}
    assert camp.run() == 0
    runs = deps.runs()
    assert runs[:2] == [first_key, first_key] and runs[2:] == [i.key for i in camp.plan[1:]]
    i_retry = next(n for n, c in enumerate(deps.calls) if c[0] == "run" and c[2] == 2)
    assert ("reset", True) in deps.calls[:i_retry]                   # full reset before the retry
    assert C.item_status(camp.plan[0], "camp-test") == "failed"
    parent = Path(camp.plan[0].arm_dir).parent
    assert sorted(p.name.split("-2")[0] for p in parent.glob(f"{camp.plan[0].arm}.failed-a*")) == \
        [f"{camp.plan[0].arm}.failed-a1", f"{camp.plan[0].arm}.failed-a2"]
    # resume: the failed arm is not rerun; --retry-failed runs it again with fresh attempts
    deps2 = FakeDeps()
    assert campaign(tmp_path, deps2, cfg={"arms": ["tre", "apa"]}).run() == 0 and deps2.runs() == []
    deps3 = FakeDeps()
    assert campaign(tmp_path, deps3, cfg={"arms": ["tre", "apa"]}, retry_failed=True).run() == 0
    assert deps3.runs() == [first_key]


def test_interrupted_directory_moved_aside_and_counted_as_an_attempt(tmp_path):
    deps = FakeDeps()
    camp = campaign(tmp_path, deps, cfg={"traces": ["T7_tight_pool"], "arms": ["tre"]})
    item = camp.plan[0]
    Path(item.arm_dir).mkdir(parents=True)
    (Path(item.arm_dir) / "runner.log").write_text("killed mid-arm\n")
    camp.state["attempts"][item.key] = 1                              # the attempt the crash used
    camp._save_state()
    assert camp.run() == 0
    assert [c[2] for c in deps.calls if c[0] == "run"] == [2]
    assert ("reset", True) in deps.calls                             # retry = full reset first
    assert list(Path(item.arm_dir).parent.glob("tre.interrupted-*"))
    assert C.item_status(item, "camp-test") == "done"


def test_stop_file_stops_after_the_current_arm_and_resume_continues(tmp_path):
    deps = FakeDeps(on_run=lambda item: (Path(item.arm_dir).parents[1] / "STOP").write_text(""))
    camp = campaign(tmp_path, deps)
    assert camp.run() == 0
    assert deps.runs() == [camp.plan[0].key]
    assert json.loads((camp.root / "status.json").read_text())["state"] == "stopped"
    assert C.item_status(camp.plan[1], "camp-test") == "pending"
    (camp.root / "STOP").unlink()
    deps2 = FakeDeps()
    assert campaign(tmp_path, deps2).run() == 0
    assert deps2.runs() == [i.key for i in camp.plan[1:]]


def test_stop_prevents_the_retry_but_keeps_it_for_the_resume(tmp_path):
    deps = FakeDeps(on_run=lambda item: (Path(item.arm_dir).parents[1] / "STOP").write_text(""))
    camp = campaign(tmp_path, deps, cfg={"traces": ["T7_tight_pool"], "arms": ["tre"]})
    deps.script = {camp.plan[0].key: ["rc"]}
    assert camp.run() == 0 and deps.runs() == [camp.plan[0].key]
    assert C.item_status(camp.plan[0], "camp-test") == "pending"      # attempt 1 kept aside, not failed
    (camp.root / "STOP").unlink()
    deps2 = FakeDeps()
    assert campaign(tmp_path, deps2, cfg={"traces": ["T7_tight_pool"], "arms": ["tre"]}).run() == 0
    assert [c[2] for c in deps2.calls if c[0] == "run"] == [2]


def test_blocked_precheck_stops_without_using_an_attempt(tmp_path):
    deps = FakeDeps(precheck_ok=False)
    camp = campaign(tmp_path, deps)
    assert camp.run() == 3 and deps.runs() == []
    assert camp.state["attempts"] == {}
    assert json.loads((camp.root / "status.json").read_text())["state"] == "blocked"


def test_preserve_arm_gets_its_oracle_copy_and_rendered_policy(tmp_path):
    extra = {"arm_defs": {"preserve-phase": {"base": "preserve", "label": "PreServe-phase-shift", "policy": {"tier1": "oracle"},
                                             "oracle_shift": {"period_s": 100, "frac": 0.25, "seed": 1}}},
             "traces": [{"trace": "Alternating", "arms": ["preserve", "preserve-phase"], "preserve_policy": {"window_s": 120}}],
             "arms": ["preserve"]}
    deps = FakeDeps()
    camp = campaign(tmp_path, deps, cfg={"extra": extra})
    assert camp.run() == 0
    pol_files = [c[4] for c in deps.calls if c[0] == "run"]
    assert all(p and p.endswith(".policy-configmaps.yaml") for p in pol_files)
    for p, item in zip(pol_files, camp.plan):
        docs = [d for d in yaml.safe_load_all(Path(p).read_text()) if d]
        assert {d["metadata"]["name"] for d in docs} == {"tre-v2-baseline-tokenscale", "tre-v2-baseline-preserve"}
        params = yaml.safe_load(next(d for d in docs if d["metadata"]["name"].endswith("preserve"))["data"]["preserve.yaml"])
        client = str(tmp_path / "traces" / "Alternating" / "seed1" / "traces_tre.effective.json")
        assert C._tail(params["trace_path"], 3) == C._tail(client, 3) and params["trace_match_parts"] == 3
        assert params["window_s"] == 120 and params["noise_sigma"] == 0.0772
        host = Path(str(params["trace_path"]).replace("/etc/tre-baselines-traces", str(tmp_path / "bt")))
        oracle = json.loads(host.read_text())
        assert len(oracle) == 100 and all("prompt" not in r for r in oracle)
        if item.arm == "preserve-phase":
            assert params["tier1"] == "oracle"
            man = json.loads((Path(item.arm_dir) / "campaign_manifest.json").read_text())
            delta = man["preserve"]["oracle_shift"]["delta_s"]
            assert -25 <= delta <= 25 and delta != 0
            assert oracle[0]["timestamp"] != 0.0


# ------------------------------------------------------------------ pieces
def test_slice_and_shift_keep_ids_and_counts():
    recs = [{"request_id": f"r{i}", "timestamp": float(i)} for i in range(10)]
    sl = C.slice_trace(recs, 3, 4)
    assert [r["request_id"] for r in sl] == ["r3", "r4", "r5", "r6"] and sl[0]["timestamp"] == 0.0
    sh = C.shift_trace(recs, 2.5, 10)
    assert len(sh) == 10 and {r["request_id"] for r in sh} == {r["request_id"] for r in recs}
    assert all(0 <= r["timestamp"] < 10 for r in sh) and sh == sorted(sh, key=lambda r: r["timestamp"])
    assert C.shift_delta("c", "k", 1, {"period_s": 540, "frac": 0.25, "seed": 1}) == \
        C.shift_delta("c", "k", 1, {"period_s": 540, "frac": 0.25, "seed": 1})


def test_marker_end_only_moves_later_and_nothing_else_changes(tmp_path):
    m = tmp_path / "MARKER"
    text = ("validation 2026-10-08T01:00:00+08:00 2026-10-08T05:00:00+08:00 round=r1 owner=me\n"
            "phase=final; notes keep\n")
    m.write_text(text)
    end = C.parse_marker(text)["end_epoch"]
    assert C.extend_marker(m, "r1", end - 3600, 600) is None and m.read_text() == text      # earlier: untouched
    msg = C.extend_marker(m, "r1", end + 3600, 600)
    new = m.read_text()
    assert msg and new.splitlines()[1] == "phase=final; notes keep" and new.endswith("\n")
    f = new.splitlines()[0].split(" ")
    assert f[0:2] == ["validation", "2026-10-08T01:00:00+08:00"] and f[3:] == ["round=r1", "owner=me"]
    assert f[2] == "2026-10-08T06:10:00+08:00"
    assert C.extend_marker(m, "other-round", end + 99999, 0) is None and m.read_text() == new


def test_precheck_recovery_paths():
    c = {"precheck": {"wait_s": 100, "poll_s": 30}}
    log = []
    t = [0.0]
    seq = iter([[{"name": "layout", "ok": False, "detail": "x", "kind": "reset"}], [{"name": "layout", "ok": True}]])
    resets = []
    ok, _ = C.precheck_until_ready(lambda: next(seq), lambda f: resets.append(f), c, log.append)
    assert ok and resets == [True]
    resets.clear()
    ok, _ = C.precheck_until_ready(lambda: [{"name": "disk", "ok": False, "detail": "x", "kind": "hard"}],
                                   lambda f: resets.append(f), c, log.append)
    assert not ok and resets == []
    calls = []

    def stale():
        calls.append(1)
        return [{"name": "gpu_truth", "ok": False, "detail": "old", "kind": "wait"}]
    ok, _ = C.precheck_until_ready(stale, lambda f: None, c, log.append,
                                   sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0])
    assert not ok and len(calls) == 5                                  # 0, 30, 60, 90, 120 s


def test_assess_arm_verdicts(tmp_path):
    thr = C.DEFAULTS["validity"]
    d = tmp_path / "arm"
    (d / "client").mkdir(parents=True)
    (d / "client" / "performance_metrics.json").write_text("[]")
    for f in C.COLLECTOR_FILES:
        (d / f).write_text("x")

    def score(n, trimmed, fail):
        (d / "score.json").write_text(json.dumps({"models": {"ALL": {"n": n, "trimmed": trimmed, "fail": fail,
                                                                     "max_tokens_hit_frac": 1.0}}}))
    score(90, 10, 0)
    assert C.assess_arm(d, 0, 100, thr)["verdict"] == "valid"
    score(90, 10, 3)
    assert C.assess_arm(d, 0, 100, thr)["verdict"] == "suspect"
    score(90, 10, 40)
    assert C.assess_arm(d, 0, 100, thr)["verdict"] == "invalid"
    score(50, 10, 0)
    assert "requests" in C.assess_arm(d, 0, 100, thr)["invalid"][0]
    score(90, 10, 0)
    (d / "EMFILE_PODS").write_text("sidecar/p.log\n")
    assert C.assess_arm(d, 0, 100, thr)["verdict"] == "invalid"
    (d / "EMFILE_PODS").write_text("")
    (d / "sm.ts.log").write_text('2026 INFO: 10.0.0.1 - "PUT /v2/models/m/target HTTP/1.1" 409 Conflict\n')
    v = C.assess_arm(d, 0, 100, thr)
    assert v["metrics"]["sm_409"] == 1 and v["verdict"] == "valid"
    assert C.assess_arm(d, 4, 100, thr)["verdict"] == "invalid"
    assert C.assess_arm(d, 0, 100, thr, baseline=True)["verdict"] == "invalid"      # no run_validity.json


def test_dry_run_prints_plan_and_flags_missing_inputs(tmp_path, monkeypatch):
    p = write_campaign(tmp_path)
    c = C.load_campaign(p)
    monkeypatch.setattr(C, "validate_inputs", lambda c, e, plan: ([], []))
    out = io.StringIO()
    assert C.dry_run(c, out=out) == 0
    txt = out.getvalue()
    assert "total: 9 arms over 3 trace entries" in txt and "T7_tight_pool_s1" in txt
    assert not (tmp_path / "results").exists()                         # nothing written
    monkeypatch.undo()
    (tmp_path / "traces" / "Simul_spike" / "seed1" / "traces_tre.effective.json").unlink()
    probs, _ = C.validate_inputs(c, C.resolve_traces(c), C.build_plan(c))
    assert any("Simul_spike" in x and "missing trace" in x for x in probs)

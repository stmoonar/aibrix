#!/usr/bin/env python3
"""Offline self-test of bl_tools.py (analysis rules, policy check, capacity, trace) and of
score_pilot.py --validity. No cluster, no Redis. Run: python3 selftest_bl_tools.py"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bl_tools as bt  # noqa: E402

T0 = 1_000_000


def line(ts_s, awake, clamped, action="none", model="m", reason="r", **kw):
    d = {"ts_ms": T0 + int(ts_s * 1000), "tick": int(ts_s), "policy": "p", "model": model, "awake": awake,
         "clamped": clamped, "action": action, "reason": reason, "dry_run": False}
    d.update(kw)
    return d


def meta(phases, **kw):
    m = {"part": "X", "policy": kw.pop("policy", "tokenscale"), "t_load_redis_ms": T0, "tick_s": 2, "settle_s": 10,
         "max_changes": 1, "cap_awake": 4, "phases": phases}
    m.update(kw)
    return m


def ph(check, start, end, **kw):
    return {"name": check, "model": kw.pop("model", "m"), "start_s": start, "end_s": end, "check": check, **kw}


fails = []


def expect(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


# steady: constant -> pass; flip-flop -> fail; ratchet -> fail; range
dec = [line(t, 1, 2) for t in range(0, 60, 2)]
r = bt.analyze(meta([ph("steady", 0, 60, expect_range=[2, 2])]), dec, None, "act")
expect("steady constant in range passes", r["pass"])
r = bt.analyze(meta([ph("steady", 0, 60, expect_range=[1, 1])]), dec, None, "act")
expect("steady out of range fails", not r["pass"])
dec = [line(t, 1, 2 if (t // 2) % 2 else 1) for t in range(0, 60, 2)]
r = bt.analyze(meta([ph("steady", 0, 60)]), dec, None, "act")
expect("steady +-1 flip-flop fails", not r["pass"] and r["phases"][0]["flipflops"] > 0)
dec = [line(t, 1, min(4, 1 + t // 12)) for t in range(0, 60, 2)]
r = bt.analyze(meta([ph("steady", 0, 60)], policy="chiron"), dec, None, "act")
expect("steady ratchet to cap fails", not r["pass"] and r["phases"][0]["ratchet_to_cap"])
r = bt.analyze(meta([ph("steady", 0, 60)]), [], None, "act")
expect("no decisions fails", not r["pass"])

# up: target rises at 6 s, awake follows at 10 s
dec = [line(t, 1 if t < 10 else 3, 1 if t < 6 else 3, action="up" if t == 6 else "none") for t in range(0, 40, 2)]
r = bt.analyze(meta([ph("up", 0, 40, deadline_s=8)]), dec, None, "act")
p = r["phases"][0]
expect("up within deadline passes + time_to_wake", r["pass"] and p["first_up_s"] == 6 and p["time_to_wake_s"] == 4)
r = bt.analyze(meta([ph("up", 0, 40, deadline_s=4)]), dec, None, "act")
expect("up after deadline fails", not r["pass"])

# down (act): awake 3 -> 1 with one down action; once-per-window
dec = [line(t, 3 if t < 12 else 1, 3 if t < 10 else 1, action="down" if t == 10 else "none") for t in range(0, 60, 2)]
r = bt.analyze(meta([ph("down", 0, 60, deadline_s=20, once_per_window_s=30)], replay_t0_ms=T0), dec, None, "act")
expect("down within deadline passes", r["pass"] and r["phases"][0]["first_down_s"] == 10)
dec2 = dec + [line(15, 2, 1, action="down", reason="again")]
r = bt.analyze(meta([ph("down", 0, 60, deadline_s=20, once_per_window_s=30)], replay_t0_ms=T0), dec2, None, "act")
expect("two downs in one window fail", not r["pass"])
dec = [line(t, 3, 3) for t in range(0, 60, 2)]
r = bt.analyze(meta([ph("down", 0, 60, deadline_s=20)]), dec, None, "act")
expect("no down fails", not r["pass"])

# hold
dec = [line(t, 3, 3, reason="incomplete", inputs={"policy_clamped": 1}) for t in range(0, 60, 2)]
r = bt.analyze(meta([ph("hold", 0, 60)]), dec, None, "act")
expect("hold passes when no down", r["pass"] and r["phases"][0]["incomplete"] == 30)
dec[10] = line(20, 3, 2, action="down")
r = bt.analyze(meta([ph("hold", 0, 60)]), dec, None, "act")
expect("hold fails on a down", not r["pass"])

# contend + release: m refused (409, then 200 unfilled), donor d sleeps at 40 s, m reaches 4 at 44 s
dec = []
for t in range(0, 80, 2):
    sm = None
    if t in (10, 14):
        sm = {"direction": "up", "code": 409, "ok": False}
    if t == 18:
        sm = {"direction": "up", "code": 200, "ok": True, "unfilled": 1}
    dec.append(line(t, 3 if t < 44 else 4, 4, action="up" if sm else "backoff", sm_result=sm))
    dec.append(line(t, 3 if t < 40 else 1, 3 if t < 30 else 1, model="d"))
dec.sort(key=lambda d: (d["ts_ms"], d["model"]))
r = bt.analyze(meta([ph("contend", 0, 30), ph("release", 30, 80, donor="d", deadline_s=10)]), dec, None, "act")
expect("contend counts 409 + unfilled", r["phases"][0]["refusals"] == 3 and r["sm_refusals"] == 3)
expect("release latency measured and passes", r["pass"] and r["phases"][1]["release_latency_s"] == 4)
r = bt.analyze(meta([ph("contend", 0, 30), ph("release", 30, 80, donor="d", deadline_s=2)]), dec, None, "act")
expect("slow release fails", not r["pass"])

# dry: overloaded phases are informational
dec = [line(t, 1, min(4, 1 + t // 12), dry_run=True) for t in range(0, 60, 2)]
r = bt.analyze(meta([{**ph("steady", 0, 60), "mult": 1.5}]), dec, None, "dry")
expect("dry 1.5x steady is informational", r["pass"] and r["phases"][0]["pass"] is None)
r = bt.analyze(meta([{**ph("steady", 0, 60), "mult": 1.0}]), dec, None, "dry")
expect("dry 1.0x steady still gates", not r["pass"])

# tick stall
dec = [line(t, 1, 1) for t in (0, 2, 4, 40, 42)]
r = bt.analyze(meta([ph("steady", 0, 60)]), dec, None, "act")
expect("tick stall fails", not r["pass"] and not r["checks"]["no_tick_stall"])

# guard
dec = [line(t, 1, 2, action="guard_controller_active") for t in range(0, 30, 2)]
r = bt.analyze(meta([ph("steady", 0, 30)]), dec, None, "act")
expect("guard_controller_active fails", not r["pass"])

# ---- file-based commands
tmp = tempfile.mkdtemp()
reg = os.path.join(tmp, "reg.yaml")
open(reg, "w").write("models:\n  - {name: a, max_awake_replicas: 4}\n  - {name: b, max_awake_replicas: 4}\n")


def run(*args):
    p = subprocess.run([sys.executable, os.path.join(HERE, "bl_tools.py"), *args], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


live = os.path.join(tmp, "chiron.yaml")
open(live, "w").write("{}\n")
rc, out = run("check-policy", "--policy", "chiron", "--live", live, "--registry", reg)
expect("check-policy refuses {}", rc == 2)
open(live, "w").write("theta:\n  a: 0.46\n  b: null\nalpha: 0.5\n")
rc, out = run("check-policy", "--policy", "chiron", "--live", live, "--registry", reg)
expect("check-policy refuses theta null", rc == 2 and "theta[b]" in out)
open(live, "w").write("theta:\n  a: 0.46\n  b: 0.53\nalpha: 0.5\n")
cm = os.path.join(tmp, "cms.yaml")
open(cm, "w").write("""apiVersion: v1
kind: ConfigMap
metadata: {name: tre-v2-baseline-chiron, namespace: tre-v2}
data:
  chiron.yaml: |
    alpha: 0.5
    theta: {a: 0.46, b: 0.53}
---
apiVersion: v1
kind: ConfigMap
metadata: {name: tre-v2-baseline-preserve, namespace: tre-v2}
data:
  preserve.yaml: |
    trace_path: /etc/tre-baselines-traces/Alt/traces_tre.effective.json
    mu: {a: {p: 1, d: 1, t: 8920}, b: {p: 1, d: 1, t: 7136}}
""")
rc, out = run("check-policy", "--policy", "chiron", "--live", live, "--registry", reg, "--frozen", cm)
expect("check-policy accepts frozen-equal", rc == 0 and '"frozen_equal": true' in out)
open(live, "w").write("theta:\n  a: 0.46\n  b: 0.60\n")
rc, out = run("check-policy", "--policy", "chiron", "--live", live, "--registry", reg, "--frozen", cm)
expect("check-policy refuses frozen-different", rc == 2 and '"frozen_equal": false' in out)
pl = os.path.join(tmp, "preserve.yaml")
open(pl, "w").write("trace_path: /etc/tre-baselines-traces/Alt/traces_tre.effective.json\nmu: {a: {p: 1, d: 1, t: 8920}, b: {p: 1, d: 1, t: 7136}}\n")
rc, out = run("check-policy", "--policy", "preserve", "--live", pl, "--registry", reg, "--frozen", cm,
              "--trace", "/root/x/icse_final/Alt/traces_tre.effective.json")
expect("preserve trace tail match", rc == 0)
rc, out = run("check-policy", "--policy", "preserve", "--live", pl, "--registry", reg,
              "--trace", "/root/x/icse_final/Other/traces_tre.effective.json")
expect("preserve trace tail mismatch refused", rc == 2 and "Tier-1" in out)

pdir = os.path.join(tmp, "pol")
os.makedirs(pdir)
open(os.path.join(pdir, "preserve.yaml"), "w").write("mu: {a: {p: 1, d: 1, t: 8920}}\n")
open(os.path.join(pdir, "tokenscale.yaml"), "w").write(
    "bucket_edges: {'*': {in: [489, 494], out: [400, 400]}}\nvelocity: {a: {buckets: [[1,2,3],[4,5,6],[7,8,9000]], v_prefill: 9}}\n")
rc, out = run("capacity", "--policy-dir", pdir, "--model", "a", "--src", "mu")
expect("capacity from mu_t", rc == 0 and abs(json.loads(out)["rps"] - 10.0) < 1e-9)
rc, out = run("capacity", "--policy-dir", pdir, "--model", "a", "--src", "vb", "--in", "492", "--out", "400")
expect("capacity from V_b bucket [1][0] (492 in, 400 out)", rc == 0 and json.loads(out)["src"] == "tokenscale V_b[1][0]")
rc, out = run("capacity", "--policy-dir", pdir, "--model", "a", "--src", "vb", "--in", "600", "--out", "500")
expect("capacity V_b clamps to the last bucket", rc == 0 and json.loads(out)["tok_s"] == 9000)
rc, out = run("capacity", "--policy-dir", pdir, "--model", "zz", "--src", "mu")
expect("capacity missing refuses", rc != 0)
caps = os.path.join(tmp, "caps.json")
json.dump({"a": {"rps": 10.0}, "b": {"rps": 8.0}}, open(caps, "w"))
tr = os.path.join(tmp, "t", "trace.json")
rc, out = run("mktrace", "--out", tr, "--cap-json", caps, "--seg", "a:0:60:1.0", "--seg", "a:60:120:3.0", "--seg", "b:0:120:0.5")
t = json.load(open(tr))
expect("mktrace segments", rc == 0 and t["a"][1]["rps"] == 30.0 and t["b"][0]["rps"] == 4.0
       and t["a"][0]["input_tokens"] == 492 and t["a"][0]["max_tokens"] == 400)

# score_pilot --validity on a tiny metrics file
cdir = os.path.join(tmp, "client")
os.makedirs(cdir)
rows = [{"request_id": f"r{i}", "model_name": "a", "start_time": 100.0 + i, "success": True, "ttft": 0.1,
         "e2e_latency": 10.0, "output_tokens": 400, "input_tokens": 492} for i in range(50)]
with open(os.path.join(cdir, "performance_metrics.json"), "w") as fh:
    for r_ in rows:
        fh.write(json.dumps(r_) + "\n")
sreg = os.path.join(tmp, "sreg.yaml")
open(sreg, "w").write("models:\n  - name: a\n    slo: {ttft_p95_ms: 2000, tpot_p95_ms: 75, e2e_p95_ms: 15000}\n")
sp = os.path.join(HERE, "score_pilot.py")
base = subprocess.run([sys.executable, sp, cdir, "--registry", sreg], capture_output=True, text=True)
b = json.loads(base.stdout)
expect("score_pilot without --validity has no new keys", base.returncode == 0 and "valid" not in b and "run_validity" not in b)
vf = os.path.join(tmp, "run_validity.json")
json.dump({"events_valid": False, "invalid_because": ["gateway dropped events: 3.0"]}, open(vf, "w"))
v = json.loads(subprocess.run([sys.executable, sp, cdir, "--registry", sreg, "--validity", vf, "--arm-label", "Chiron-global"],
                              capture_output=True, text=True).stdout)
expect("score_pilot invalid run marked", v["valid"] is False and v["arm_label"] == "Chiron-global"
       and "dropped" in v["invalid_because"][0])
v = json.loads(subprocess.run([sys.executable, sp, cdir, "--registry", sreg, "--validity", vf + ".missing"],
                              capture_output=True, text=True).stdout)
expect("score_pilot missing validity marked invalid", v["valid"] is False and v["run_validity"] is None)
json.dump({"events_valid": True, "invalid_because": []}, open(vf, "w"))
v = json.loads(subprocess.run([sys.executable, sp, cdir, "--registry", sreg, "--validity", vf], capture_output=True, text=True).stdout)
expect("score_pilot valid run", v["valid"] is True and v["models"] == b["models"])

print(f"\n{len(fails)} failure(s)")
sys.exit(1 if fails else 0)

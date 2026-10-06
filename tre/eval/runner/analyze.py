#!/usr/bin/env python3
"""analyze.py <arm_dir>: summary.json + stdout digest for one smoke-E1 arm."""
import collections
import json
import math
import re
import sys
from pathlib import Path

D = Path(sys.argv[1])
T_LOAD = int((D / "load_start_epoch").read_text())
T_END = int((D / "load_end_epoch").read_text())
S = {"arm": D.name, "load_s": T_END - T_LOAD}


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    f = math.floor(k)
    c = min(f + 1, len(xs) - 1)
    return round(xs[f] + (xs[c] - xs[f]) * (k - f), 3)


# ---- client
recs = [json.loads(l) for l in open(D / "client/performance_metrics.json") if l.strip()]
cl = {}
for m in ["ALL", "dsqwen-7b", "dsllama-8b", "dsqwen-14b"]:
    rs = [r for r in recs if m == "ALL" or r["model_name"] == m]
    ok = [r for r in rs if r["success"]]
    cl[m] = {
        "n": len(rs), "ok": len(ok), "err": len(rs) - len(ok),
        "interrupted": sum(bool(r.get("stream_interrupted")) for r in rs),
        "finish": dict(collections.Counter(r.get("finish_reason") for r in rs)),
        "ttft_p95": pct([r["ttft"] for r in ok], 95), "ttft_p99": pct([r["ttft"] for r in ok], 99),
        "tpot_p95": pct([r["tpot"] for r in ok], 95), "tpot_p99": pct([r["tpot"] for r in ok], 99),
        "e2e_p50": pct([r["e2e_latency"] for r in ok], 50),
        "e2e_p95": pct([r["e2e_latency"] for r in ok], 95), "e2e_p99": pct([r["e2e_latency"] for r in ok], 99),
        "short_output": sum(1 for r in ok if (r.get("output_tokens") or 0) < 400 and r.get("finish_reason") != "stop"),
    }
    # Strict basis (unified client, 2026-09-30; the columns above are the v1 basis): TTFT on
    # the first content token, TPOT = (E2E - TTFT)/(n - 1), cut / error-chunk / zero-output
    # streams fail, retry waits excluded. Records written before the unified client have no
    # *_strict fields: every strict column is then None.
    if any("success_strict" in r for r in rs):
        oks = [r for r in rs if r.get("success_strict")]
        cl[m].update({
            "ok_strict": len(oks), "err_strict": len(rs) - len(oks),
            "failures_strict": dict(collections.Counter(r.get("failure_strict") for r in rs
                                                        if not r.get("success_strict"))),
            "ttft_missing_strict": sum(1 for r in oks if r.get("ttft_missing_strict")),
            "ttft_strict_p95": pct([r.get("ttft_strict_s") for r in oks], 95),
            "ttft_strict_p99": pct([r.get("ttft_strict_s") for r in oks], 99),
            "tpot_strict_p95": pct([r.get("tpot_strict_s") for r in oks], 95),
            "tpot_strict_p99": pct([r.get("tpot_strict_s") for r in oks], 99),
            "e2e_strict_p50": pct([r.get("e2e_strict_s") for r in oks], 50),
            "e2e_strict_p95": pct([r.get("e2e_strict_s") for r in oks], 95),
            "e2e_strict_p99": pct([r.get("e2e_strict_s") for r in oks], 99),
            "retried": sum(1 for r in rs if (r.get("retries") or 0) > 0),
            "tre_continued": sum(1 for r in rs if r.get("tre_continued")),
            "send_lateness_p99_ms": pct([r.get("send_lateness_ms") for r in rs], 99),
            "send_lateness_max_ms": max((r.get("send_lateness_ms") or 0.0 for r in rs), default=None),
        })
    else:
        cl[m].update({k: None for k in (
            "ok_strict", "err_strict", "failures_strict", "ttft_missing_strict", "ttft_strict_p95",
            "ttft_strict_p99", "tpot_strict_p95", "tpot_strict_p99", "e2e_strict_p50", "e2e_strict_p95",
            "e2e_strict_p99", "retried", "tre_continued", "send_lateness_p99_ms", "send_lateness_max_ms")})
cl["errors"] = dict(collections.Counter((r.get("http_status"), (r.get("error_message") or "")[:80])
                                       for r in recs if not r["success"]).most_common(8))
cl["errors"] = {str(k): v for k, v in cl["errors"].items()}
S["client"] = cl

# ---- layout: awake / routable per model, replica-seconds, changes, floor
rows = [json.loads(l) for l in open(D / "layout.jsonl") if l.strip()]
rows = [r for r in rows if "models" in r]
lay = {}
for m in ["dsqwen-7b", "dsllama-8b", "dsqwen-14b"]:
    seq = [(r["ts"], len(r["models"][m]["awake"]), len(set(r["models"][m]["awake"]) - set(r["models"][m]["hidden"])))
           for r in rows if m in r["models"]]
    win = [x for x in seq if T_LOAD <= x[0] <= T_END + 30]
    rs = 0.0
    for (t0, a, _), (t1, _, _) in zip(win, win[1:]):
        rs += a * (t1 - t0)
    changes = []
    prev = None
    for t, a, _ in win:
        if prev is not None and a != prev:
            changes.append((round(t - T_LOAD), prev, a))
        prev = a
    # flap = direction reversal within 120 s
    flaps = 0
    for (t1, p1, a1), (t2, p2, a2) in zip(changes, changes[1:]):
        if (a1 - p1) * (a2 - p2) < 0 and t2 - t1 <= 120:
            flaps += 1
    lay[m] = {"max_awake": max((a for _, a, _ in win), default=None),
              "min_routable": min((r for _, _, r in win), default=None),
              "replica_s": round(rs), "n_changes": len(changes), "flaps_120s": flaps,
              "changes": changes[:60]}
S["layout"] = lay

# ---- sidecar + SM sleep stats diffs
b, a = json.load(open(D / "snap_before.json")), json.load(open(D / "snap_after.json"))
side = collections.Counter()
for pod, ctr in a["pods"].items():
    for k, v in ctr.items():
        if k == "error":
            continue
        base = b["pods"].get(pod, {}).get(k, 0.0)
        name = re.sub(r'model="[^"]*",?', "", k).replace("{}", "")
        side[name] += v - base
S["sidecar_delta"] = {k: v for k, v in side.items() if v}
sb, sa = b["sleep"]["stats"], a["sleep"]["stats"]
S["sm_sleep_delta"] = {k: sa[k] - sb.get(k, 0) for k in sa if sa[k] != sb.get(k, 0)}

# ---- controller decisions
states = collections.defaultdict(collections.Counter)
events = collections.Counter()
actions = []
evlines = collections.Counter()
for line in open(D / "controller.log"):
    try:
        j = json.loads(line)
    except ValueError:
        continue
    msg = j.get("message", "")
    try:
        m = json.loads(msg)
    except ValueError:
        m = None
    if isinstance(m, dict) and m.get("event") == "trs_calc_result":
        ts = int(m["ts_ms"]) / 1000
        if not (T_LOAD - 5 <= ts <= T_END + 30):
            continue
        for e in json.loads(m.get("events") or "[]"):
            events[e.split(":")[0]] += 1
        acts = json.loads(m.get("actions") or "[]")
        if acts:
            actions.append({"t": round(ts - T_LOAD), "loop": m.get("loop"), "submitted": m.get("submitted"),
                            "actions": acts})
        if m.get("loop") == "rescue":
            for mod, st in json.loads(m["model_states"]).items():
                states[mod][st.get("state")] += 1
    else:
        ev = m.get("event") if isinstance(m, dict) else msg.split(":")[0][:60]
        evlines[(j.get("logger"), ev)] += 1
S["ctrl_states_rescue_ticks"] = {k: dict(v) for k, v in states.items()}
S["ctrl_events"] = dict(events.most_common(40))
S["ctrl_other_log_events"] = {f"{k[0]}|{k[1]}": v for k, v in evlines.most_common(40)}
S["ctrl_actions"] = actions
act_types = collections.Counter()
for x in actions:
    for ac in x["actions"]:
        if isinstance(ac, dict):
            act_types[(ac.get("type") or ac.get("kind") or ac.get("action"), ac.get("reason"))] += 1
        else:
            act_types[str(ac)[:60]] += 1
S["ctrl_action_types"] = {str(k): v for k, v in act_types.items()}

# ---- pod gauges (kv) and APA
kv = collections.defaultdict(list)
for l in open(D / "pod_gauges.jsonl"):
    r = json.loads(l)
    if "pods" not in r or not (T_LOAD <= r["ts"] <= T_END):
        continue
    per = collections.defaultdict(list)
    for p, g in r["pods"].items():
        if "kv_cache_usage_perc" in g:
            per[g["model"]].append(g["kv_cache_usage_perc"])
    for m, v in per.items():
        kv[m].append(sum(v) / len(v))
S["kv_mean_awake"] = {m: {"n": len(v), "nonzero_frac": round(sum(x > 0 for x in v) / len(v), 3),
                          "mean": round(sum(v) / len(v), 3), "max": round(max(v), 3)} for m, v in kv.items()}
apa = collections.defaultdict(list)
for l in open(D / "apa_status.jsonl"):
    r = json.loads(l)
    for n, st in (r.get("pa") or {}).items():
        apa[n].append((round(r["ts"] - T_LOAD), st.get("desiredScale"), st.get("actualScale")))
S["apa"] = {n: {"max_desired": max((d or 0) for _, d, _ in v),
                "changes": [x for i, x in enumerate(v) if i == 0 or x[1:] != v[i - 1][1:]][:40]}
            for n, v in apa.items()}
json.dump(S, open(D / "summary.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in S.items() if k not in ("ctrl_actions",)}, indent=1, default=str)[:12000])

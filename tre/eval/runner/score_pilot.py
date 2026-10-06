#!/usr/bin/env python3
"""score_pilot.py <client_dir_or_performance_metrics.json> [--registry R] [--trim-s 30] [--out summary.json]

Pilot-only V_req scorer for tre/loadgen_v1 output (v1 performance_metrics.json lines).
A request violates the SLO when it failed, or TTFT > max(floor, k*(c + b*input_tokens)),
or e2e >= 149 s (cut by the 150 s route timeout), or TPOT > tpot_p95_ms (TPOT = (e2e - ttft) / (output_tokens - 1), the v2 scorer's definition).
Reported twice: without the e2e SLO (primary for the ICSE traces: 300-700-token outputs make the
12/15 s e2e SLO fire on almost every request) and with it (the v2 E1 scorer's definition).
Requests are timed by start_time; the first --trim-s seconds after the first send are dropped
(E1 used trim = 1 window of 30 s). Read-only; prints JSON.

Output-length check (2026-10-05, pilot sends ignore_eos): per model and ALL,
  output_tokens_sum        sum of output_tokens over the scored (post-trim) requests
  output_tokens_sum_all    same over every row (trimmed ones too) - the arm's total work
  max_tokens_hit / _frac   successful scored requests whose output_tokens == their max_tokens
                           (max_tokens = the request's max_output_tokens in traces.json next to
                           performance_metrics.json, else --traces; with ignore_eos this should be 1.0)
  max_tokens_unknown       successful scored requests whose max_tokens could not be found

Baseline arms (2026-10-06): --validity <run_validity.json> (written by `arm disable --collect-dir`)
adds the keys `arm_label`, `run_validity` (the file, or null when missing) and `valid` (false when
the file is missing or its events_valid is false; `invalid_because` says why). Without --validity
the output is unchanged (tre / apa).
"""
import argparse, collections, json, math, os, sys
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--registry", default="/data/nfs_shared_data/xxy/aibrix-wt/calib-theta-20261003/tre/deploy/registry.yaml",
                help="registry whose slo block (c/b, k, floor, tpot, e2e) is used; default = the calib branch (new c/b)")
ap.add_argument("--trim-s", type=float, default=30.0)
ap.add_argument("--out")
ap.add_argument("--traces", help="traces.json with per-request max_output_tokens (default: next to the metrics file)")
ap.add_argument("--validity", help="run_validity.json of a baseline arm (adds run_validity / valid)")
ap.add_argument("--arm-label", help="arm label recorded with --validity (e.g. Chiron-global)")
a = ap.parse_args()
src = a.src if a.src.endswith(".json") else os.path.join(a.src, "performance_metrics.json")
traces_path = a.traces or os.path.join(os.path.dirname(os.path.abspath(src)), "traces.json")
max_tok = {}
if os.path.exists(traces_path):
    max_tok = {t["request_id"]: t.get("max_output_tokens") for t in json.load(open(traces_path))}
reg = yaml.safe_load(open(a.registry))
slo = {m["name"]: m["slo"] for m in reg["models"]}
rows = [json.loads(l) for l in open(src) if l.strip()]
t_first = min(r["start_time"] for r in rows if r.get("start_time"))

def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs: return None
    k = (len(xs) - 1) * p / 100; f = math.floor(k); c = min(f + 1, len(xs) - 1)
    return round(xs[f] + (xs[c] - xs[f]) * (k - f), 3)

acc = collections.defaultdict(lambda: collections.Counter())
lat = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    for key in (r["model_name"], "ALL"): acc[key]["output_tokens_sum_all"] += (r.get("output_tokens") or 0)
    if r.get("start_time") and r["start_time"] - t_first < a.trim_s:
        for key in (r["model_name"], "ALL"): acc[key]["trimmed"] += 1
        continue
    s = slo[r["model_name"]]
    L = r.get("input_tokens") or 0
    ttft_slo = max(s.get("ttft_floor_ms", 500.0), s.get("ttft_slowdown_k", 5.0) * (s["ttft_idle_c_ms"] + s["ttft_idle_b_ms_per_token"] * L)) \
        if s.get("ttft_slo_mode") == "slowdown" else s["ttft_p95_ms"]
    ok = bool(r.get("success"))
    ttft = r["ttft"] * 1000 if r.get("ttft") is not None else None
    e2e = r["e2e_latency"] * 1000 if r.get("e2e_latency") is not None else None
    out = r.get("output_tokens") or 0
    tpot = (e2e - ttft) / (out - 1) if (ok and ttft is not None and e2e is not None and out > 1) else None
    v_tt = ttft is not None and ttft > ttft_slo
    v_tp = tpot is not None and tpot > s["tpot_p95_ms"]
    v_e2e = e2e is not None and e2e > s["e2e_p95_ms"]
    # route-timeout censoring: loadgen_v1 records a stream cut by the 150 s route timeout as
    # success (memory tre-client-diff-v1-v2-20260930); count it as a violation.
    censored = e2e is not None and e2e >= 149000
    v = (not ok) or v_tt or v_tp or censored
    mt = max_tok.get(r.get("request_id"))
    for key in (r["model_name"], "ALL"):
        c = acc[key]; c["n"] += 1; c["fail"] += (not ok); c["v_ttft"] += v_tt; c["v_tpot"] += v_tp
        c["viol"] += v; c["viol_with_e2e"] += (v or v_e2e)
        c["censored_ge_149s"] += censored
        c["http_502"] += (r.get("http_status") == 502 or "502" in str(r.get("error_message") or ""))
        c["output_tokens_sum"] += out
        if ok:
            c["max_tokens_unknown"] += mt is None
            c["max_tokens_hit"] += (mt is not None and out == mt)
        if ok:
            lat[key]["ttft"].append(ttft); lat[key]["tpot"].append(tpot); lat[key]["e2e"].append(e2e)
res = {"source": os.path.abspath(src), "registry": a.registry, "trim_s": a.trim_s,
       "traces": traces_path if max_tok else None, "models": {}}
for key in sorted(acc, key=lambda k: (k != "ALL", k)):
    c = acc[key]; n = max(c["n"], 1)
    ok_known = c["n"] - c["fail"] - c["max_tokens_unknown"]
    res["models"][key] = {**c, "V_req_pct": round(100 * c["viol"] / n, 3),
                          "max_tokens_hit_frac": round(c["max_tokens_hit"] / ok_known, 5) if ok_known > 0 else None,
                          "V_req_with_e2e_pct": round(100 * c["viol_with_e2e"] / n, 3),
                          **{f"{m}_p{p}_ms": pct(lat[key][m], p) for m in ("ttft", "tpot", "e2e") for p in (50, 95, 99)}}
if a.validity:
    rv = json.load(open(a.validity)) if os.path.exists(a.validity) else None
    res["arm_label"] = a.arm_label
    res["run_validity"] = rv
    why = ["run_validity.json missing: " + a.validity] if rv is None else \
        ([] if rv.get("events_valid") else ["events_valid false: " + "; ".join(rv.get("invalid_because") or ["?"])])
    res["valid"] = not why
    res["invalid_because"] = why
txt = json.dumps(res, indent=1, default=int)
if a.out: open(a.out, "w").write(txt + "\n")
print(txt)

#!/usr/bin/env python3
"""bl_tools.py - helpers of the baseline arms of the pilot runner and the sanity script.

Read-only against the cluster except `del-key` (one baseline-only Redis key). Redis is reached
directly (redis-py, `--redis host[:port]`, the tre-v2-redis ClusterIP from 76).

  redis-ms        --redis H                         print Redis TIME in ms
  del-key         --redis H --key K                 DEL K (the stale replay marker before an arm)
  wait-first-arr  --redis H --since-ms T --models a,b [--timeout-s 180]
                                                    block until the first `arr` event with id >= T in
                                                    tre:v2:bl:req:<m>; print its id ms (exit 1 on timeout)
  dump-streams    --redis H --since-ms T --out-dir D --models a,b
                                                    tre:v2:bl:decisions -> D/bl_decisions.jsonl,
                                                    tre:v2:bl:req:<m>   -> D/bl_req_events.<m>.jsonl
  check-policy    --policy P --live F --registry R [--frozen CMFILE] [--trace PATH]
                                                    refuse (exit 2) an empty / placeholder policy file, a
                                                    live file != the frozen ConfigMap file, or (preserve) a
                                                    trace_path whose last trace_match_parts parts differ
                                                    from PATH; prints a JSON record
  client-header   --meta loadgen_run_meta.json      print the x-tre-bl-in-tokens precount summary
                                                    (exit 1 when any request was sent without the header)
Sanity (baseline_sanity.sh):
  capacity        --policy-dir D --model m [--src auto|mu|vb] [--in 492] [--out 400] [--rps X]
                                                    print JSON {rps, tok_s, src} of 1.0x one-replica capacity
  mktrace         --out F --cap-json J --seg model:start_s:end_s:mult [...] [--in 492] [--max 400]
                                                    replayer segment trace.json (rps = mult x capacity)
  analyze         --part P --policy P --mode dry|act --decisions DIR_OR_FILES --meta part_meta.json
                  [--validity run_validity.json] [--records records.jsonl] --out result.json
                  (steady: settle = min(settle_s, 0.4 x phase length); Chiron S2: see _chiron_up;
                  --records defaults to records.jsonl next to the meta file)
  summary         --root DIR --arm A --out summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

REQ_PREFIX = "tre:v2:bl:req:"
DECISIONS = "tre:v2:bl:decisions"


def _redis(spec: str):
    import redis as redis_lib

    host, _, port = spec.partition(":")
    return redis_lib.Redis(host=host, port=int(port or 6379), decode_responses=True, socket_timeout=5.0)


def _redis_ms(r) -> int:
    s, us = r.time()
    return int(s) * 1000 + int(us) // 1000


def _id_ms(entry_id: str) -> int:
    return int(entry_id.split("-", 1)[0])


# ------------------------------------------------------------------ runner helpers


def cmd_redis_ms(a) -> int:
    print(_redis_ms(_redis(a.redis)))
    return 0


def cmd_del_key(a) -> int:
    print(_redis(a.redis).delete(a.key))
    return 0


def cmd_wait_first_arr(a) -> int:
    r = _redis(a.redis)
    models = [m for m in a.models.split(",") if m]
    deadline = time.monotonic() + a.timeout_s
    while True:
        best = None
        for m in models:
            for eid, fields in r.xrange(REQ_PREFIX + m, min=f"{a.since_ms}-0", max="+", count=200):
                if fields.get("kind") == "arr":
                    ms = _id_ms(eid)
                    best = ms if best is None else min(best, ms)
                    break
        if best is not None:
            print(best)
            return 0
        if time.monotonic() >= deadline:
            print("timeout", file=sys.stderr)
            return 1
        time.sleep(a.poll_s)


def cmd_dump_streams(a) -> int:
    r = _redis(a.redis)
    os.makedirs(a.out_dir, exist_ok=True)
    counts = {}
    with open(os.path.join(a.out_dir, "bl_decisions.jsonl"), "w") as fh:
        n = 0
        for eid, fields in r.xrange(DECISIONS, min=f"{a.since_ms}-0", max="+"):
            fh.write(json.dumps({"id": eid, **fields}) + "\n")
            n += 1
        counts["decisions"] = n
    for m in [m for m in a.models.split(",") if m]:
        n = 0
        with open(os.path.join(a.out_dir, f"bl_req_events.{m}.jsonl"), "w") as fh:
            for eid, fields in r.xrange(REQ_PREFIX + m, min=f"{a.since_ms}-0", max="+"):
                fh.write(json.dumps({"id": eid, **fields}) + "\n")
                n += 1
        counts[m] = n
    print(json.dumps(counts))
    return 0


def _registry_models(path: str) -> list[str]:
    import yaml

    reg = yaml.safe_load(open(path))
    return [m["name"] for m in reg.get("models", [])]


def _tail(path: str, parts: int) -> tuple:
    bits = [b for b in str(path).replace("\\", "/").split("/") if b]
    return tuple(bits[-parts:]) if parts > 0 else ()


def policy_problems(policy: str, params: dict, models: list[str]) -> list[str]:
    """Placeholder / missing parameters that the policy itself would refuse (checked early)."""
    out = []
    if not isinstance(params, dict) or not params:
        return ["policy file is empty ({}): parameters not frozen/applied yet"]
    if policy == "chiron":
        theta = params.get("theta") or {}
        for m in models:
            v = theta.get(m, theta.get("*")) if isinstance(theta, dict) else theta
            if v is None or float(v) <= 0:
                out.append(f"theta[{m}] missing / null")
    elif policy == "tokenscale":
        vel = params.get("velocity") or {}
        for m in models:
            v = vel.get(m) if isinstance(vel, dict) else None
            if not v or not v.get("buckets") or not v.get("v_prefill"):
                out.append(f"velocity[{m}] missing")
        if "model-a" in (params.get("models") or []) or "model-a" in vel:
            out.append("example placeholder model-a present")
    elif policy == "preserve":
        mu = params.get("mu") or {}
        for m in models:
            v = mu.get(m) or {}
            if not v or float(v.get("t") or 0) <= 0:
                out.append(f"mu[{m}] missing / 0")
        if not params.get("trace_path"):
            out.append("trace_path missing")
    return out


def cmd_check_policy(a) -> int:
    import yaml

    live_txt = open(a.live).read()
    try:
        live = yaml.safe_load(live_txt) or {}
    except yaml.YAMLError as exc:
        live, problems = {}, [f"live policy file is not YAML: {exc}"]
    else:
        problems = []
    models = _registry_models(a.registry)
    problems += policy_problems(a.policy, live, models)
    rec = {"policy": a.policy, "live_file": a.live, "models": models, "frozen_file": a.frozen,
           "frozen_equal": None, "trace_check": None}
    if a.frozen:
        want = None
        for doc in yaml.safe_load_all(open(a.frozen)):
            if doc and doc.get("kind") == "ConfigMap" and \
                    (doc.get("metadata") or {}).get("name") == f"tre-v2-baseline-{a.policy}":
                want = yaml.safe_load((doc.get("data") or {}).get(f"{a.policy}.yaml") or "{}") or {}
        if want is None:
            problems.append(f"{a.frozen} has no ConfigMap tre-v2-baseline-{a.policy}")
        else:
            rec["frozen_equal"] = (want == live)
            if want != live:
                problems.append("live ConfigMap differs from the frozen file (apply it: BL_CM_APPLY=1)")
    if a.policy == "preserve" and a.trace:
        parts = int(live.get("trace_match_parts", 2)) if isinstance(live, dict) else 2
        tp = live.get("trace_path") if isinstance(live, dict) else None
        ok = bool(tp) and (parts <= 0 or _tail(tp, parts) == _tail(a.trace, parts))
        rec["trace_check"] = {"trace_path": tp, "marker_trace": a.trace, "match_parts": parts, "ok": ok}
        if not ok:
            problems.append(f"preserve trace_path {tp!r} does not end like the replayed trace {a.trace!r} "
                            f"(last {parts} parts): Tier-1 would stay inactive")
    rec["problems"] = problems
    rec["ok"] = not problems
    print(json.dumps(rec, indent=1))
    return 0 if not problems else 2


def cmd_client_header(a) -> int:
    meta = json.load(open(a.meta))
    s = meta.get("in_tokens_header")
    print(json.dumps(s))
    if not s:
        return 1
    return 0 if int(s.get("omitted") or 0) == 0 and int(s.get("counted") or 0) > 0 else 1


# ------------------------------------------------------------------ sanity: capacity + traces


def _bucket(edges, x) -> int:
    i = 0
    for e in edges or []:
        if x > e:
            i += 1
    return i


def capacity(policy_dir: str, model: str, src: str, tin: int, tout: int, rps: float | None) -> dict:
    import yaml

    if rps:
        return {"rps": float(rps), "tok_s": float(rps) * (tin + tout), "src": "rps"}
    tried = []
    if src in ("auto", "mu"):
        p = os.path.join(policy_dir, "preserve.yaml")
        if os.path.exists(p):
            mu = ((yaml.safe_load(open(p)) or {}).get("mu") or {}).get(model) or {}
            t = float(mu.get("t") or 0)
            if t > 0:
                return {"rps": t / (tin + tout), "tok_s": t, "src": "preserve mu_t"}
        tried.append("preserve mu_t")
    if src in ("auto", "vb"):
        p = os.path.join(policy_dir, "tokenscale.yaml")
        if os.path.exists(p):
            par = yaml.safe_load(open(p)) or {}
            vel = (par.get("velocity") or {}).get(model) or {}
            b = vel.get("buckets")
            if b:
                edges = (par.get("bucket_edges") or {})
                e = edges.get(model) or edges.get("*") or {}
                i = min(_bucket(e.get("in"), tin), len(b) - 1)
                row = b[i] if isinstance(b[i], list) else b
                j = min(_bucket(e.get("out"), tout), len(row) - 1)
                v = float(row[j])
                if v > 0:
                    return {"rps": v / (tin + tout), "tok_s": v, "src": f"tokenscale V_b[{i}][{j}]"}
        tried.append("tokenscale V_b")
    raise SystemExit(f"no capacity for {model} ({', '.join(tried)} unavailable): set CAP_RPS_<model> or CAP_RPS")


def cmd_capacity(a) -> int:
    print(json.dumps(capacity(a.policy_dir, a.model, a.src, a.tin, a.tout, a.rps)))
    return 0


def cmd_mktrace(a) -> int:
    caps = json.load(open(a.cap_json))
    trace: dict = {}
    for s in a.seg:
        model, start, end, mult = s.split(":")
        trace.setdefault(model, []).append({
            "start_time": float(start), "end_time": float(end),
            "rps": round(float(mult) * float(caps[model]["rps"]), 4),
            "input_tokens": a.tin, "max_tokens": a.tmax})
    for segs in trace.values():
        segs.sort(key=lambda x: x["start_time"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(trace, open(a.out, "w"), indent=1)
    print(json.dumps(trace))
    return 0


# ------------------------------------------------------------------ sanity: analysis


def load_decisions(paths: list[str]) -> list[dict]:
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "decisions-*.jsonl"), recursive=True))
        elif os.path.exists(p):
            files.append(p)
    seen, out = set(), []
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = (d.get("ts_ms"), d.get("model"), d.get("tick"), d.get("policy"))
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
    out.sort(key=lambda d: (d.get("ts_ms") or 0, d.get("model") or ""))
    return out


def _sm_refused(d: dict) -> bool:
    r = d.get("sm_result") or {}
    if not r:
        return False
    if r.get("code") == 409:
        return True
    return bool(r.get("ok")) and bool(r.get("unfilled") or r.get("refusals"))


def _series(dec: list[dict], model: str, t0: int, t1: int) -> list[dict]:
    return [d for d in dec if d.get("model") == model and t0 <= (d.get("ts_ms") or 0) < t1]


def _changes(vals: list[int]) -> tuple[int, int]:
    """(number of value changes, number of direction reversals = +-1 flip-flops)."""
    ch, rev, last_dir = 0, 0, 0
    for a, b in zip(vals, vals[1:]):
        if b != a:
            ch += 1
            d = 1 if b > a else -1
            if last_dir and d != last_dir:
                rev += 1
            last_dir = d
    return ch, rev


def _first(series, pred):
    for d in series:
        if pred(d):
            return d
    return None


def _target(d: dict, mode: str) -> int:
    return int(d.get("clamped") if d.get("clamped") is not None else d.get("awake") or 0)


#: Chiron S2 (methodology 10-06): <= this many direction reversals over the high plateau and
#: < this fraction of the phase's requests interrupted by a sleep (continued / retried).
CHIRON_MAX_REVERSALS = 1
CHIRON_MAX_ABORT_FRAC = 0.02
#: steady phases: settle = min(settle_s, SETTLE_FRAC x phase length), so a short dry phase
#: (DRY_FRAC 0.3) still has samples.
SETTLE_FRAC = 0.4


def _chiron_up(info: dict, s: list[dict], mode: str, tick_s: float, t0: int, t1: int,
               records: list[dict] | None, t_load: int) -> bool:
    """Chiron S2: the scale-up is timed from the evidence, not from the step.

    evidence = first tick whose sum_q > busy0 x B_mean (busy0 = the busy count at the phase
    start, i.e. sum_q crossed (k-1) x B for the next k); decision within one tick of it
    (<= 1.5 tick_s), act: the wake follows within deadline_s. Plateau: target direction
    reversals <= CHIRON_MAX_REVERSALS; act: requests sent in the phase that were interrupted
    (tre_continued / tre_retried) < CHIRON_MAX_ABORT_FRAC."""
    base = s[0].get("awake") if s else None
    inp0 = (s[0].get("inputs") or {}) if s else {}
    busy0 = int(inp0.get("busy") or 0)

    def crossed(d):
        i = d.get("inputs") or {}
        q, b = i.get("queued"), i.get("B_mean")
        return q is not None and b and float(q) > busy0 * float(b) + 1e-6

    ev = _first(s, crossed)
    info["busy0"] = busy0
    info["evidence_s"] = None if not ev else round((ev["ts_ms"] - t0) / 1000.0, 2)
    up = None if not ev else _first(s, lambda d: d["ts_ms"] >= ev["ts_ms"] and _target(d, mode) > (base or 0))
    info["first_up_s"] = None if not up else round((up["ts_ms"] - t0) / 1000.0, 2)
    info["decision_after_evidence_s"] = None if not up else round((up["ts_ms"] - ev["ts_ms"]) / 1000.0, 2)
    ok = up is not None and info["decision_after_evidence_s"] <= 1.5 * tick_s
    dl = float(info.get("deadline_s") or 10)
    if mode == "act" and up is not None:
        woke = _first(s, lambda d: d["ts_ms"] >= up["ts_ms"] and (d.get("awake") or 0) > (base or 0))
        info["time_to_wake_s"] = None if not woke else round((woke["ts_ms"] - up["ts_ms"]) / 1000.0, 2)
        ok = ok and woke is not None and info["time_to_wake_s"] <= dl
    ch, rev = _changes([_target(d, mode) for d in s])
    info.update(target_changes=ch, reversals=rev, max_reversals=CHIRON_MAX_REVERSALS)
    ok = ok and rev <= CHIRON_MAX_REVERSALS
    if mode == "act":
        if records:
            # align the client clock to the Redis clock: t_load = the first arrival = the first send
            off = t_load - min(int(r["actual_send_ts_ms"]) for r in records if r.get("actual_send_ts_ms"))
            sent = [r for r in records if r.get("actual_send_ts_ms") and t0 <= int(r["actual_send_ts_ms"]) + off < t1]
            ab = sum(1 for r in sent if r.get("tre_continued") or r.get("tre_retried"))
            info.update(requests=len(sent), aborted=ab, abort_frac=round(ab / len(sent), 4) if sent else None,
                        max_abort_frac=CHIRON_MAX_ABORT_FRAC)
            ok = ok and bool(sent) and ab / len(sent) < CHIRON_MAX_ABORT_FRAC
        else:
            info["abort_frac"] = None
            info["note"] = "no records.jsonl: abort criterion not evaluated"
            ok = False
    return ok


def analyze(meta: dict, dec: list[dict], validity: dict | None, mode: str,
            records: list[dict] | None = None) -> dict:
    part, policy = meta["part"], meta["policy"]
    t_load = int(meta["t_load_redis_ms"])
    tick_s = float(meta.get("tick_s", 2))
    res: dict = {"part": part, "policy": policy, "mode": mode, "checks": {}, "notes": [], "phases": []}
    models = sorted({d["model"] for d in dec})
    res["decision_lines"] = len(dec)
    res["models_seen"] = models
    if not dec:
        res["checks"]["decisions_present"] = False
        res["pass"] = False
        return res
    res["checks"]["decisions_present"] = True
    res["actions"] = {}
    for d in dec:
        k = f"{d['model']}:{d.get('action')}"
        res["actions"][k] = res["actions"].get(k, 0) + 1
    res["checks"]["no_guard_controller_active"] = not any(d.get("action") == "guard_controller_active" for d in dec)
    if mode == "act":
        res["checks"]["actuating_not_dry"] = any(not d.get("dry_run") for d in dec)
    reasons: dict = {}
    for d in dec:
        reasons[d.get("reason")] = reasons.get(d.get("reason"), 0) + 1
    res["reasons"] = reasons
    # tick gaps per model (liveness / deadlock)
    gap_max = 0.0
    for m in models:
        ts = [d["ts_ms"] for d in dec if d["model"] == m]
        for x, y in zip(ts, ts[1:]):
            gap_max = max(gap_max, (y - x) / 1000.0)
    res["max_tick_gap_s"] = gap_max
    res["checks"]["no_tick_stall"] = gap_max <= max(5 * tick_s, 15.0)
    refused = [d for d in dec if _sm_refused(d)]
    res["sm_refusals"] = len(refused)
    res["sm_calls"] = sum(1 for d in dec if d.get("sm_result"))
    res["sm_down_calls"] = sum(1 for d in dec if (d.get("sm_result") or {}).get("direction") == "down")
    res["sm_up_elapsed_s"] = [ (d.get("sm_result") or {}).get("elapsed_s") for d in dec
                               if (d.get("sm_result") or {}).get("direction") == "up"]
    cap = meta.get("cap_awake")
    for ph in meta["phases"]:
        model = ph["model"]
        t0 = t_load + int(ph["start_s"] * 1000)
        t1 = t_load + int(ph["end_s"] * 1000)
        s = _series(dec, model, t0, t1)
        info = {"name": ph["name"], "model": model, "mult": ph.get("mult"), "lines": len(s)}
        if s:
            info["awake_start"] = s[0].get("awake")
            info["awake_end"] = s[-1].get("awake")
            info["targets"] = sorted({_target(d, mode) for d in s})
            info["last_inputs"] = s[-1].get("inputs")
        kind = ph.get("check")
        # a dry shell never adds replicas: above 1x (and after a 3x step) the queue of the one replica
        # grows without bound, so those dry phases are informational (pass None), not gate checks
        dry_overload = mode == "dry" and (kind == "down" or (kind == "steady" and float(ph.get("mult") or 0) > 1.0))
        if kind == "steady":
            settle = int(min(float(meta.get("settle_s", 60)), SETTLE_FRAC * (ph["end_s"] - ph["start_s"])) * 1000)
            info["settle_s"] = settle / 1000.0
            ss = [d for d in s if d["ts_ms"] >= t0 + settle]
            vals = [_target(d, mode) for d in ss]
            ch, rev = _changes(vals)
            info.update(changes_after_settle=ch, flipflops=rev, target_values=sorted(set(vals)))
            ok = bool(vals) and rev == 0 and ch <= int(meta.get("max_changes", 1))
            # ratchet: rises step by step to the cap under constant load
            ratchet = bool(vals) and cap is not None and vals[-1] >= int(cap) and vals[0] < int(cap) and ch >= 2 \
                and all(b >= a for a, b in zip(vals, vals[1:]))
            info["ratchet_to_cap"] = ratchet
            ok = ok and not ratchet
            rng = ph.get("expect_range")
            if rng and vals:
                in_rng = rng[0] <= vals[-1] <= rng[1]
                info["expect_range"] = rng
                info["in_expected_range"] = in_rng
                ok = ok and in_rng
            if policy == "chiron" and s and isinstance(s[-1].get("inputs"), dict):
                info["chiron_note"] = "expect ceil(busy_eff/theta); see last_inputs"
            info["pass"] = ok
        elif kind == "up" and policy == "chiron":
            info["deadline_s"] = float(ph.get("deadline_s", 10))
            info["pass"] = _chiron_up(info, s, mode, tick_s, t0, t1, records, t_load)
        elif kind == "up":
            base = s[0].get("awake") if s else None
            first_up = _first(s, lambda d: _target(d, mode) > (base or 0))
            info["first_up_s"] = None if not first_up else round((first_up["ts_ms"] - t0) / 1000.0, 2)
            dl = float(ph.get("deadline_s", 20))
            info["deadline_s"] = dl
            ok = first_up is not None and info["first_up_s"] <= dl
            if mode == "act" and first_up is not None:
                woke = _first(s, lambda d: d["ts_ms"] >= first_up["ts_ms"] and (d.get("awake") or 0) > (base or 0))
                info["time_to_wake_s"] = None if not woke else round((woke["ts_ms"] - first_up["ts_ms"]) / 1000.0, 2)
                ok = ok and woke is not None
            info["pass"] = ok
        elif kind == "down":
            base = s[0].get("awake") if s else None
            dl = float(ph.get("deadline_s", 30))
            info["deadline_s"] = dl
            if mode == "act":
                first_dn = _first(s, lambda d: d.get("action") == "down")
                info["first_down_s"] = None if not first_dn else round((first_dn["ts_ms"] - t0) / 1000.0, 2)
                downs = [d for d in s if d.get("action") == "down"]
                info["down_actions"] = len(downs)
                ok = first_dn is not None and info["first_down_s"] <= dl and (s[-1].get("awake") or 0) < (base or 0)
                if ph.get("once_per_window_s"):
                    w = float(ph["once_per_window_s"]) * 1000
                    marker = int(meta.get("replay_t0_ms") or t_load)
                    per = {}
                    for d in downs:
                        k = int((d["ts_ms"] - marker) // w)
                        per[k] = per.get(k, 0) + 1
                    info["downs_per_window"] = per
                    ok = ok and all(v <= 1 for v in per.values())
            else:
                # dry: the policy's (unrealised) target falls back below the step's peak
                peak = max((_target(d, mode) for d in _series(dec, model, t0 - 120000, t0)), default=None)
                low = _first(s, lambda d: peak is not None and _target(d, mode) < peak)
                info["dry_peak_before"] = peak
                info["first_lower_target_s"] = None if not low else round((low["ts_ms"] - t0) / 1000.0, 2)
                ok = peak is None or peak <= (base or 1) or (low is not None and info["first_lower_target_s"] <= dl)
            info["pass"] = ok
        elif kind == "hold":
            downs = [d for d in s if d.get("action") == "down"]
            aw = [d.get("awake") or 0 for d in s]
            non_incr = any(b < a for a, b in zip(aw, aw[1:]))
            info.update(down_actions=len(downs), awake_decreased=non_incr,
                        incomplete=sum(1 for d in s if d.get("reason") == "incomplete"),
                        policy_wanted_down=sum(1 for d in s if (d.get("inputs") or {}).get("policy_clamped") is not None))
            info["pass"] = not downs and not non_incr
        elif kind == "contend":
            ref = [d for d in dec if t0 <= d["ts_ms"] < t1 and _sm_refused(d)]
            info["refusals"] = len(ref)
            info["refusal_samples"] = [d.get("sm_result") for d in ref[:3]]
            info["backoff_lines"] = sum(1 for d in dec if t0 <= d["ts_ms"] < t1 and d.get("action") == "backoff")
            if mode == "act" and not ref:
                res["notes"].append("S4: no SM refusal seen (no contention reached: raise S4 multipliers)")
            info["pass"] = True  # informational; the release phase decides
        elif kind == "release":
            other = ph.get("donor")
            so = _series(dec, other, t0, t1)
            dn = _first(so, lambda d: d.get("awake") is not None and so and d["awake"] < so[0]["awake"])
            info["donor_released_s"] = None if not dn else round((dn["ts_ms"] - t0) / 1000.0, 2)
            if mode == "act":
                if dn is None:
                    info["pass"] = None
                    res["notes"].append("S4 release: the donor model never slept; release not exercised")
                else:
                    after = [d for d in s if d["ts_ms"] >= dn["ts_ms"]]
                    want = max((_target(d, mode) for d in after), default=None)
                    got = _first(after, lambda d: want is not None and (d.get("awake") or 0) >= want)
                    info["release_latency_s"] = None if not got else round((got["ts_ms"] - dn["ts_ms"]) / 1000.0, 2)
                    dl = float(ph.get("deadline_s", 30))
                    info["deadline_s"] = dl
                    info["pass"] = got is not None and info["release_latency_s"] <= dl
            else:
                info["pass"] = True
        if dry_overload and info.get("pass") is not None:
            info["pass_if_actuated"] = info["pass"]
            info["pass"] = None
            info["note"] = "dry: overload not relieved by a dry shell; informational"
        res["phases"].append(info)
    if validity is not None:
        res["run_validity"] = {k: validity.get(k) for k in ("events_valid", "invalid_because", "gw_bl_dropped_delta",
                                                             "policy_events", "sm")}
        pe = validity.get("policy_events") or {}
        res["tier2_below_t1"] = {m: v.get("tier2_below_t1", 0) for m, v in pe.items()}
        if policy == "preserve" and mode == "act" and part.startswith("S23"):
            # 10-06 fix: tier2_below_t1 counts the windows in which Tier-2 WANTED to go below the
            # window's Tier-1 N and was clamped to N (preserve.py docstring; the gate working),
            # so it is reported, not a failure. The invariant: no decision targets below N.
            viol = [d for d in dec if isinstance(d.get("inputs"), dict)
                    and d["inputs"].get("tier1_n") is not None and d["inputs"].get("target") is not None
                    and d["inputs"]["target"] < d["inputs"]["tier1_n"]]
            res["target_below_t1"] = len(viol)
            res["checks"]["preserve_never_below_t1"] = not viol
    phase_ok = [p["pass"] for p in res["phases"] if p.get("pass") is not None]
    res["pass"] = all(res["checks"].values()) and all(phase_ok)
    return res


def cmd_analyze(a) -> int:
    meta = json.load(open(a.meta))
    dec = load_decisions(a.decisions)
    validity = json.load(open(a.validity)) if a.validity and os.path.exists(a.validity) else None
    rec_path = a.records or os.path.join(os.path.dirname(os.path.abspath(a.meta)), "records.jsonl")
    records = None
    if os.path.exists(rec_path):
        records = [json.loads(x) for x in open(rec_path) if x.strip()]
    res = analyze(meta, dec, validity, a.mode, records)
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({"part": res["part"], "mode": a.mode, "pass": res["pass"], "lines": res["decision_lines"],
                      "phases": [(p["name"], p.get("pass")) for p in res["phases"]]}))
    return 0


def cmd_summary(a) -> int:
    parts = []
    for f in sorted(glob.glob(os.path.join(a.root, "*", "*", "result.json"))):
        r = json.load(open(f))
        parts.append({"dir": os.path.dirname(f), "part": r["part"], "mode": r["mode"], "pass": r["pass"],
                      "notes": r.get("notes"), "sm_refusals": r.get("sm_refusals"),
                      "phases": [{k: p.get(k) for k in ("name", "pass", "first_up_s", "time_to_wake_s",
                                                         "evidence_s", "reversals", "abort_frac",
                                                         "first_down_s", "down_actions", "target_values",
                                                         "flipflops", "release_latency_s", "refusals")
                                  if p.get(k) is not None} for p in r["phases"]]})
    act = [p for p in parts if p["mode"] == "act"]
    doc = {"arm": a.arm, "root": a.root, "parts": parts,
           "pass": bool(act) and all(p["pass"] for p in act),
           "failed": [f"{p['part']}/{p['mode']}" for p in parts if not p["pass"]]}
    json.dump(doc, open(a.out, "w"), indent=1)
    print(json.dumps({"arm": a.arm, "pass": doc["pass"], "failed": doc["failed"]}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("redis-ms"); p.add_argument("--redis", required=True); p.set_defaults(fn=cmd_redis_ms)
    p = sub.add_parser("del-key"); p.add_argument("--redis", required=True); p.add_argument("--key", required=True)
    p.set_defaults(fn=cmd_del_key)
    p = sub.add_parser("wait-first-arr"); p.add_argument("--redis", required=True)
    p.add_argument("--since-ms", type=int, required=True); p.add_argument("--models", required=True)
    p.add_argument("--timeout-s", type=float, default=180.0); p.add_argument("--poll-s", type=float, default=0.2)
    p.set_defaults(fn=cmd_wait_first_arr)
    p = sub.add_parser("dump-streams"); p.add_argument("--redis", required=True)
    p.add_argument("--since-ms", type=int, required=True); p.add_argument("--out-dir", required=True)
    p.add_argument("--models", required=True); p.set_defaults(fn=cmd_dump_streams)
    p = sub.add_parser("check-policy"); p.add_argument("--policy", required=True, choices=["chiron", "tokenscale", "preserve"])
    p.add_argument("--live", required=True); p.add_argument("--registry", required=True)
    p.add_argument("--frozen"); p.add_argument("--trace"); p.set_defaults(fn=cmd_check_policy)
    p = sub.add_parser("client-header"); p.add_argument("--meta", required=True); p.set_defaults(fn=cmd_client_header)
    p = sub.add_parser("capacity"); p.add_argument("--policy-dir", required=True); p.add_argument("--model", required=True)
    p.add_argument("--src", default="auto", choices=["auto", "mu", "vb"]); p.add_argument("--in", dest="tin", type=int, default=492)
    p.add_argument("--out", dest="tout", type=int, default=400); p.add_argument("--rps", type=float, default=None)
    p.set_defaults(fn=cmd_capacity)
    p = sub.add_parser("mktrace"); p.add_argument("--out", required=True); p.add_argument("--cap-json", required=True)
    p.add_argument("--seg", action="append", required=True); p.add_argument("--in", dest="tin", type=int, default=492)
    p.add_argument("--max", dest="tmax", type=int, default=400); p.set_defaults(fn=cmd_mktrace)
    p = sub.add_parser("analyze"); p.add_argument("--part", required=True); p.add_argument("--policy", required=True)
    p.add_argument("--mode", required=True, choices=["dry", "act"]); p.add_argument("--decisions", nargs="+", required=True)
    p.add_argument("--meta", required=True); p.add_argument("--validity"); p.add_argument("--out", required=True)
    p.add_argument("--records"); p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("summary"); p.add_argument("--root", required=True); p.add_argument("--arm", required=True)
    p.add_argument("--out", required=True); p.set_defaults(fn=cmd_summary)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

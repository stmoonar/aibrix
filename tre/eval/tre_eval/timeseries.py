"""Time series (1 s grid, long format), the unified event list and per-GPU occupancy intervals.

All times are reference seconds relative to load start (t = 0). Empty cells = no sample
in that second (signals are never interpolated or forward-filled except the replica
counts, which are sample-and-hold by definition of the SM layout).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from .load import Arm, parse_binding
from . import metrics as M

TS_FIELDS = ("awake", "routable", "hidden", "gpus_awake", "target", "offered_rps", "rho", "sent_rps",
             "completed_rps", "gen_tok_s", "viol_frac_10s", "ttft_p95_10s_ms", "kv_mean", "kv_max",
             "running", "waiting", "pods_sampled", "z", "tss", "theta", "queue_len", "ctrl_z", "ctrl_kv",
             "ctrl_running", "ctrl_waiting", "ctrl_routable")
# columns from the newer collectors; written only when the arm has their source file (old arms: unchanged CSV)
TS_FIELDS_NEW = ("routable_sm", "routable_label", "kv_mean_all", "running_all", "waiting_all",
                 "pods_hidden_sampled", "gen_tok_s_engine", "preemptions")


def fields(ts: dict[str, dict[str, np.ndarray]]) -> list[str]:
    """Columns of this arm's time series: TS_FIELDS plus the new ones that have a source."""
    have = set()
    for m, f in ts.items():
        if m != "_t":
            have.update(k for k in TS_FIELDS_NEW if k in f)
    return list(TS_FIELDS) + [k for k in TS_FIELDS_NEW if k in have]


def _pm_routable(g: dict) -> bool:
    """Routable for the routable-only columns: the pod label (what the old pod_gauges sampler filtered on),
    else the sampler's ``routable`` flag."""
    if "routable_label" in g:
        return g.get("routable_label") == "true"
    return bool(g.get("routable"))


def grid_bounds(arm: Arm) -> tuple[int, int]:
    lo = -30
    hi = int(math.ceil((arm.t_end - arm.t_load) if arm.t_end and arm.t_load else 0)) + 90
    ends = [r["end_time"] - arm.t_load for r in arm.requests if r.get("end_time") and arm.t_load]
    if ends:
        hi = max(hi, int(math.ceil(max(ends))) + 5)
    return lo, hi


def build(arm: Arm, recs: list[dict], capacity: dict[str, float] | None = None) -> dict[str, dict[str, np.ndarray]]:
    """model -> field -> array over the 1 s grid ``t = lo .. hi-1``; also key '_t'."""
    lo, hi = grid_bounds(arm)
    n = hi - lo
    t = np.arange(lo, hi, dtype=float)
    nan = lambda: np.full(n, np.nan)  # noqa: E731
    res: dict[str, dict[str, np.ndarray]] = {m: {f: nan() for f in TS_FIELDS} for m in arm.models}
    res["_t"] = {"t": t}
    idx = lambda x: int(math.floor(x)) - lo  # noqa: E731

    # replicas (sample-and-hold from the SM layout, NaN before the first and after the last sample)
    ser = M.layout_series(arm)
    for m, s in ser.items():
        if not s:
            continue
        for i, x in enumerate(s):
            a = idx(x[0] - arm.t_load)
            b = idx(s[i + 1][0] - arm.t_load) if i + 1 < len(s) else a + 1
            a, b = max(a, 0), min(max(b, a + 1), n)
            if a >= n or b <= 0:
                continue
            res[m]["awake"][a:b] = x[1]
            res[m]["routable"][a:b] = x[2]
            res[m]["hidden"][a:b] = x[1] - x[2]
            res[m]["gpus_awake"][a:b] = x[3]
    for m, s in M.targets(arm).items():
        if m not in res:
            continue
        for i, (ts, v) in enumerate(s):
            a = idx(ts - arm.t_load)
            b = idx(s[i + 1][0] - arm.t_load) if i + 1 < len(s) else n
            a, b = max(a, 0), min(b, n)
            if b > a:
                res[m]["target"][a:b] = v

    # load
    base, rate = M.offered_rate(arm)
    for m, r in rate.items():
        for i, v in enumerate(r):
            j = i - lo
            if 0 <= j < n:
                res[m]["offered_rps"][j] = v
        if capacity and capacity.get(m):
            res[m]["rho"] = res[m]["offered_rps"] / float(capacity[m])
    for m in arm.models:
        for f in ("sent_rps", "completed_rps", "gen_tok_s"):
            res[m][f] = np.zeros(n)
    for r in arm.requests:
        m = r.get("model_name")
        if m not in res or not r.get("start_time"):
            continue
        j = idx(r["start_time"] - arm.t_load)
        if 0 <= j < n:
            res[m]["sent_rps"][j] += 1
        if r.get("end_time"):
            j = idx(r["end_time"] - arm.t_load)
            if 0 <= j < n and r.get("success"):
                res[m]["completed_rps"][j] += 1
            out = r.get("output_tokens") or 0
            if out and r.get("ttft") is not None:
                a = r["start_time"] + r["ttft"] - arm.t_load
                b = r["end_time"] - arm.t_load
                if b > a:
                    ia, ib = max(idx(a), 0), min(idx(b) + 1, n)
                    if ib > ia:
                        res[m]["gen_tok_s"][ia:ib] += out / max(ib - ia, 1)

    # SLO violations per 10 s bin (by send time), drawn as a step over the bin
    by: dict[tuple, list] = defaultdict(list)
    for r in recs:
        if "viol" in r and r.get("t_send") is not None:
            by[(r["model"], int(r["t_send"] // 10))].append(r)
    for (m, b), rs in by.items():
        if m not in res:
            continue
        a, e = max(b * 10 - lo, 0), min(b * 10 + 10 - lo, n)
        if e > a:
            res[m]["viol_frac_10s"][a:e] = sum(x["viol"] for x in rs) / len(rs)
            res[m]["ttft_p95_10s_ms"][a:e] = M.pct([x["ttft_ms"] for x in rs if x.get("ok")], 95) or np.nan

    # SM's own routable flag per binding (layout.jsonl models[m].routable), sample-and-hold
    if arm.layout_routable:
        for m in res:
            if m != "_t":
                res[m]["routable_sm"] = nan()
        for i, (ts, models) in enumerate(arm.layout_routable):
            a = idx(ts - arm.t_load)
            b = idx(arm.layout_routable[i + 1][0] - arm.t_load) if i + 1 < len(arm.layout_routable) else a + 1
            a, b = max(a, 0), min(max(b, a + 1), n)
            if a >= n or b <= 0:
                continue
            for m in arm.models:
                res[m]["routable_sm"][a:b] = len(models.get(m) or [])

    if arm.pod_metrics:
        _pod_metrics(arm, res, n, idx)

    # vLLM gauges of the routable pods (sampler, 5 s): mean / max KV, summed running / waiting
    for ts, pods in ([] if arm.pod_metrics else arm.gauges):
        j = idx(ts - arm.t_load)
        if not 0 <= j < n:
            continue
        per: dict[str, list] = defaultdict(list)
        for p, g in pods.items():
            if isinstance(g, dict) and "kv_cache_usage_perc" in g:
                per[g.get("model")].append(g)
        for m, gs in per.items():
            if m not in res:
                continue
            kv = [g["kv_cache_usage_perc"] for g in gs]
            res[m]["kv_mean"][j] = float(np.mean(kv))
            res[m]["kv_max"][j] = float(np.max(kv))
            res[m]["running"][j] = sum(g.get("num_requests_running", 0) for g in gs)
            res[m]["waiting"][j] = sum(g.get("num_requests_waiting", 0) for g in gs)
            res[m]["pods_sampled"][j] = len(gs)

    # gateway signal log (window end) and controller rescue ticks
    for s in arm.signal:
        m = s.get("model")
        j = idx(s["ts"] - arm.t_load)
        if m in res and 0 <= j < n:
            z = s.get("z_m") if s.get("z_m") is not None else s.get("z")
            res[m]["z"][j] = np.nan if z is None else z
            for f, k in (("tss", "tss"), ("theta", "theta_m"), ("queue_len", "queue_len")):
                if s.get(k) is not None:
                    res[m][f][j] = s[k]
    for tk in arm.ctrl_ticks:
        if tk.get("loop") != "rescue":
            continue
        j = idx(tk["ts"] - arm.t_load)
        if not 0 <= j < n:
            continue
        for m, st in (tk.get("model_states") or {}).items():
            if m not in res or not isinstance(st, dict):
                continue
            for f, k in (("ctrl_z", "z_m"), ("ctrl_kv", "saturation_kv"), ("ctrl_running", "saturation_running"),
                         ("ctrl_waiting", "saturation_waiting"), ("ctrl_routable", "routable_pods")):
                if st.get(k) is not None:
                    res[m][f][j] = st[k]
    return res


def _pod_metrics(arm: Arm, res: dict, n: int, idx) -> None:
    """pod_metrics_1s.jsonl (1 Hz, every awake pod incl. hidden). Routable-only columns (kv_mean, kv_max,
    running, waiting, pods_sampled) keep the old pod_gauges basis; *_all include hidden pods; engine
    rates from counter deltas between consecutive samples of a pod (a counter reset skips that pair)."""
    for m in arm.models:
        for k in ("routable_label", "kv_mean_all", "running_all", "waiting_all", "pods_hidden_sampled"):
            res[m][k] = np.full(n, np.nan)
        res[m]["gen_tok_s_engine"] = np.zeros(n)
        res[m]["preemptions"] = np.zeros(n)
    prev: dict[str, tuple] = {}
    sampled: list[int] = []
    for ts, pods in arm.pod_metrics:
        j = idx(ts - arm.t_load)
        inside = 0 <= j < n
        rout: dict[str, list] = defaultdict(list)
        alls: dict[str, list] = defaultdict(list)
        labels: dict[str, int] = defaultdict(int)
        for p, g in pods.items():
            if not isinstance(g, dict) or g.get("model") not in arm.models:
                continue
            m = g["model"]
            if g.get("routable_label") == "true":
                labels[m] += 1
            if g.get("error") is not None or "kv_cache_usage_perc" not in g or g.get("sm_awake") is False:
                continue
            alls[m].append(g)
            if _pm_routable(g):
                rout[m].append(g)
            gen, pre = g.get("generation_tokens_total"), g.get("num_preemptions_total")
            if p in prev and inside:
                t0, gen0, pre0 = prev[p]
                dt = ts - t0
                if dt > 0 and gen is not None and gen0 is not None and gen >= gen0:
                    res[m]["gen_tok_s_engine"][j] += (gen - gen0) / dt
                if pre is not None and pre0 is not None and pre >= pre0:
                    res[m]["preemptions"][j] += pre - pre0
            prev[p] = (ts, gen, pre)
        if not inside:
            continue
        sampled.append(j)
        for m in arm.models:
            res[m]["routable_label"][j] = labels.get(m, 0)
            gs = alls.get(m) or []
            res[m]["pods_hidden_sampled"][j] = sum(1 for g in gs if (g.get("sm_hidden") if g.get("sm_hidden") is not None
                                                                     else not _pm_routable(g)))
            if gs:
                res[m]["kv_mean_all"][j] = float(np.mean([g["kv_cache_usage_perc"] for g in gs]))
                res[m]["running_all"][j] = sum(g.get("num_requests_running") or 0 for g in gs)
                res[m]["waiting_all"][j] = sum(g.get("num_requests_waiting") or 0 for g in gs)
            rs = rout.get(m) or []
            if rs:
                kv = [g["kv_cache_usage_perc"] for g in rs]
                res[m]["kv_mean"][j] = float(np.mean(kv))
                res[m]["kv_max"][j] = float(np.max(kv))
                res[m]["running"][j] = sum(g.get("num_requests_running") or 0 for g in rs)
                res[m]["waiting"][j] = sum(g.get("num_requests_waiting") or 0 for g in rs)
                res[m]["pods_sampled"][j] = len(rs)
    if sampled:  # rates are 0 between samples but empty outside the sampled span
        lo_j, hi_j = min(sampled), max(sampled)
        for m in arm.models:
            for k in ("gen_tok_s_engine", "preemptions"):
                res[m][k][:lo_j] = np.nan
                res[m][k][hi_j + 1:] = np.nan


def to_rows(ts: dict[str, dict[str, np.ndarray]]):
    t = ts["_t"]["t"]
    cols = fields(ts)
    for m, f in ts.items():
        if m == "_t":
            continue
        for i, tt in enumerate(t):
            row = {"t": int(tt), "model": m}
            empty = True
            for k in cols:
                v = f[k][i] if k in f else np.nan
                if v == v:
                    row[k] = round(float(v), 5)
                    empty = False
                else:
                    row[k] = ""
            if not empty:
                yield row


# ---------------------------------------------------------------- per-GPU occupancy
def gpu_intervals(arm: Arm) -> dict[str, list[dict]]:
    """``node/gpu`` -> [{t0, t1, model, binding, hidden: [(h0, h1)], open_end}] from the 1 s layout.
    A binding occupies all its GPUs from the first sample it is awake to the first sample it is not."""
    if not arm.layout:
        return {}
    cur: dict[str, dict] = {}
    out: dict[str, list] = defaultdict(list)
    for ts, models in arm.layout:
        t = ts - arm.t_load
        awake_now: dict[str, tuple] = {}
        hidden_now = set()
        for m, (awake, hidden) in models.items():
            for b in awake:
                awake_now[b] = m
            hidden_now.update(hidden)
        for b in list(cur):
            if b not in awake_now:
                iv = cur.pop(b)
                iv["t1"] = t
                if iv.get("_h0") is not None:
                    iv["hidden"].append((iv.pop("_h0"), t))
                iv.pop("_h0", None)
                for g in iv["gpus"]:
                    out[g].append(iv)
        for b, m in awake_now.items():
            iv = cur.get(b)
            if iv is None:
                _, node, gpus = parse_binding(b)
                iv = cur[b] = {"t0": t, "t1": None, "model": m, "binding": b, "hidden": [],
                               "gpus": [f"{node}/{g}" for g in gpus], "_h0": None, "open_start": t == arm.layout[0][0] - arm.t_load}
            if b in hidden_now and iv["_h0"] is None:
                iv["_h0"] = t
            elif b not in hidden_now and iv["_h0"] is not None:
                iv["hidden"].append((iv["_h0"], t))
                iv["_h0"] = None
    t_last = arm.layout[-1][0] - arm.t_load
    for b, iv in cur.items():
        iv["t1"] = t_last
        iv["open_end"] = True
        if iv.get("_h0") is not None:
            iv["hidden"].append((iv["_h0"], t_last))
        iv.pop("_h0", None)
        for g in iv["gpus"]:
            out[g].append(iv)
    return dict(sorted(out.items(), key=lambda kv: _gpu_key(kv[0])))


def _gpu_key(g: str):
    node, _, idx = g.rpartition("/")
    return (node, int(idx) if idx.isdigit() else idx)


# ---------------------------------------------------------------- unified events
def events(arm: Arm) -> list[dict]:
    """[{t, t_epoch, source, kind, model, target, value, detail}] sorted by t."""
    ev: list[dict] = []
    rel = lambda ts: None if ts is None else round(ts - arm.t_load, 3)  # noqa: E731

    def add(ts, source, kind, model=None, target=None, value=None, /, **detail):
        ev.append({"t": rel(ts), "t_epoch": ts, "source": source, "kind": kind, "model": model,
                   "target": target, "value": value, "detail": detail or None})

    # layout transitions
    prev_awake: set = set()
    prev_hidden: set = set()
    for i, (ts, models) in enumerate(arm.layout):
        aw = {b for _, (a, _) in models.items() for b in a}
        hd = {b for _, (_, h) in models.items() for b in h}
        if i:
            for b in sorted(aw - prev_awake):
                add(ts, "layout", "awake", parse_binding(b)[0], b)
            for b in sorted(prev_awake - aw):
                add(ts, "layout", "asleep", parse_binding(b)[0], b)
            for b in sorted(hd - prev_hidden):
                add(ts, "layout", "hidden", parse_binding(b)[0], b)
            for b in sorted(prev_hidden - hd):
                add(ts, "layout", "unhidden", parse_binding(b)[0], b)
        prev_awake, prev_hidden = aw, hd
    for e in arm.sm_events:
        r = e["rec"]
        if e["kind"] == "sleep":
            add(e["ts"], "sm", "sleep", e["model"], e["binding_id"], r.get("waited_s"), path=r.get("path"),
                status=r.get("status"), forced_abort_requests=r.get("forced_abort_requests"),
                non_continuable=r.get("non_continuable_at_sleep"), ack_ms=r.get("ack_latency_ms"),
                previous_state=r.get("previous_state"))
        else:
            ph = r.get("phases_ms") or {}
            add(e["ts"], "sm", e["kind"], e["model"], e["binding_id"],
                ph.get("wake_up") / 1000 if ph.get("wake_up") is not None else None,
                operation_id=r.get("operation_id"), phases_ms=ph or None)
    for d in M.decisions(arm):
        add(d["ts"], d["source"], "decision_up" if d["dir"] > 0 else "decision_down", d["model"], None, d["dir"], **d["detail"])
    if M.decision_source(arm) != "tre":
        for t in arm.ctrl_ticks:
            for a in t["actions"]:
                if isinstance(a, dict):
                    add(M.tick_time(t), "tre_counterfactual", "action", a.get("model") or a.get("receiver"), None,
                        a.get("delta"), reason=a.get("reason"), loop=t["loop"])
    for c in arm.ctrl_events:
        add(c["ts"], "controller", c["event"], None, None, None, events=c.get("events"))
    for e in arm.sm_access:  # SM API calls with exact (kubelet) timestamps; GETs are polling, skipped
        if e["method"] != "GET":
            add(e["ts"], "sm_api", f"{e['method']} {e['template']}", e["model"], e["path"], e["status"])
    for s in arm.sidecar:
        if s.get("event") == "tre_reissue":
            extra = {}
            if s.get("request_id"):
                extra["request_id"] = s["request_id"]
            if s.get("abort_ts") is not None:
                extra["abort_t"] = rel(s["abort_ts"])
            add(s.get("ts"), "sidecar", f"reissue_{s.get('kind')}", s.get("model"), s.get("pod"), s.get("gap_ms"),
                reason=s.get("reason"), generated=s.get("generated"), depth=s.get("depth"), to=s.get("target"), **extra)
        elif s.get("event") == "tre_sleep" and s.get("ts") is not None:  # ts = sleep START (newer sidecar)
            add(s["ts"], "sidecar", "sleep_start", s.get("model"), s.get("pod"), s.get("duration_s"))
            if s.get("duration_s") is not None:
                add(s["ts"] + float(s["duration_s"]), "sidecar", "sleep_done", s.get("model"), s.get("pod"), s.get("duration_s"))
        elif s.get("event") == "tre_sleep":
            add(s.get("ts"), "sidecar", "sleep_done", s.get("model"), s.get("pod"), s.get("duration_s"))
        elif s.get("event") == "tre_sleep_failed" and s.get("ts") is not None:
            add(s["ts"], "sidecar", "sleep_failed", s.get("model"), s.get("pod"), None,
                error=s.get("error") or s.get("reason"))
    ev.sort(key=lambda e: (e["t"] is None, e["t"] if e["t"] is not None else 0))
    return ev

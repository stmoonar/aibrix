"""viewer.py <trace_report_dir> [-o viewer.html]: one self-contained interactive results page.

Reads a ``tre_eval.report --out`` directory (and, when present, the raw arm directories next to
it) and writes ONE HTML file with the data JSON, uPlot and the page code inlined, so it opens
from ``file://`` without network. Three views on one shared time axis: multi-arm timeline,
scale events, GPU layout (Gantt). Design: tre/docs/design/20261020-results-viewer.md.

Every input except ``arms/<arm>/timeseries_1s.csv`` is optional: a missing one removes the
part that needs it and adds a warning shown in the page header.

Example:
  PYTHONPATH=tre/eval python3 -m tre_eval.viewer RUN/T7_tight_pool_s1/report -o /tmp/t7.html
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as _dt
import glob
import html
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Any

from . import style as S
from .load import parse_binding, parse_rfc3339, _utc_epoch

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

SCHEMA = 1
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer_assets")
# timeseries_1s.csv column -> page field (copied unchanged except viol share in %)
TS_COPY = {"awake": "awake", "routable": "routable", "hidden": "hidden", "target": "target",
           "gpus_awake": "gpus_awake", "offered_rps": "offered_rps", "rho": "rho",
           "ttft_p95_10s_ms": "ttft_p95_ms", "running": "running", "waiting": "waiting",
           "kv_mean": "kv_mean", "z": "z", "completed_rps": "completed_rps"}
SUMMARY_KEYS = ("n", "fail", "V_req_pct", "V_req_ci_lo", "V_req_ci_hi", "ttft_p95_ms", "tpot_p95_ms",
                "mean_gpus", "gpu_s", "gpu_s_per_good_req", "goodput_rps", "max_awake", "continued", "valid")
LOG_MAX = 700          # chars of one raw log line attached to an event
DASHES = {"-": [], "--": [7, 4], ":": [2, 3], "-.": [8, 3, 2, 3]}


# ---------------------------------------------------------------- small readers
def _f(x: Any) -> float | None:
    if x is None or x == "" or x in ("None", "nan", "NaN", "null"):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _r(v: float | None, sig: int = 4) -> float | int | None:
    """Round to ``sig`` significant digits (JSON size); integers stay integers."""
    if v is None:
        return None
    if v == int(v) and abs(v) < 1e15:
        return int(v)
    return float(f"{v:.{sig}g}")


def _rt(v: float | None) -> float | None:
    return None if v is None else round(v, 3)


def _csv(path: str) -> list[dict] | None:
    try:
        with open(path, newline="", errors="replace") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return None


def _json(path: str) -> Any:
    try:
        with open(path, errors="replace") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _jcell(s: str | None) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except ValueError:
        return s


def _bool(s: Any) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


# ---------------------------------------------------------------- discovery
def discover(report_dir: str) -> dict[str, list[dict]]:
    """trace -> [{name, sub, dir, manifest}] in comparison.csv order (else directory order)."""
    out: dict[str, list[dict]] = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(report_dir, "arms", "*"))):
        if not os.path.exists(os.path.join(d, "timeseries_1s.csv")):
            continue
        man = _json(os.path.join(d, "manifest.json")) or {}
        sub = os.path.basename(d)
        name = man.get("arm") or sub.split("__", 1)[-1]
        trace = man.get("trace") or (sub.split("__", 1)[0] if "__" in sub else os.path.basename(
            os.path.dirname(os.path.abspath(report_dir))))
        out[trace].append({"name": name, "sub": sub, "dir": d, "manifest": man})
    comp = _csv(os.path.join(report_dir, "comparison.csv")) or []
    order = list(dict.fromkeys(r["arm"] for r in comp))
    for arms in out.values():
        arms.sort(key=lambda a: (order.index(a["name"]) if a["name"] in order else len(order), a["name"]))
        # one family together, its plain arm (name == family, e.g. 'tre') first: it gets the solid line
        fam = {a["name"]: S.family(a["name"], a["manifest"].get("label") or "") for a in arms}
        first: dict[str, int] = {}
        for i, a in enumerate(arms):
            first.setdefault(fam[a["name"]], i)
        pos = {a["name"]: i for i, a in enumerate(arms)}
        arms.sort(key=lambda a: (first[fam[a["name"]]], a["name"].lower() != fam[a["name"]], pos[a["name"]]))
    return dict(out)


def raw_dir(report_dir: str, arm: dict, raw_root: str | None) -> str | None:
    cands = []
    if raw_root:
        cands.append(os.path.join(raw_root, arm["name"]))
    if arm["manifest"].get("dir"):
        cands.append(arm["manifest"]["dir"])
    cands.append(os.path.join(os.path.dirname(os.path.abspath(report_dir)), arm["name"]))
    return next((c for c in cands if os.path.isdir(c)), None)


class _Clock:
    """Reference-clock correction of a raw source, the rule of load.py: subtract the clock offset of
    the node running the component (``manifest.component_nodes`` / ``clock_offsets_s``)."""

    def __init__(self, man: dict):
        self.off = {k: float(v) for k, v in (man.get("clock_offsets_s") or {}).items() if isinstance(v, (int, float))}
        self.nodes = {k: str(v) for k, v in (man.get("component_nodes") or {}).items() if v}
        self.t_load = _f(man.get("load_start_epoch"))

    def known(self, component: str) -> bool:
        return component in self.nodes

    def rel(self, epoch: float | None, component: str) -> float | None:
        if epoch is None or self.t_load is None:
            return None
        return epoch - self.off.get(self.nodes.get(component, ""), 0.0) - self.t_load


# ---------------------------------------------------------------- registry / topology
def _registry(raw: str | None) -> dict:
    if not raw or yaml is None:
        return {}
    p = os.path.join(raw, "live-registry.yaml")
    try:
        with open(p) as fh:
            return yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _short_names(names: list[str]) -> dict[str, str]:
    """Drop the longest common '-'-separated prefix of the node names (display only)."""
    if len(names) < 2:
        return {n: n for n in names}
    parts = [n.split("-") for n in names]
    k = 0
    while all(len(p) > k + 1 for p in parts) and len({p[k] for p in parts}) == 1:
        k += 1
    return {n: "-".join(p[k:]) for n, p in zip(names, parts)}


def topology(reg: dict, gpu_ids: list[str]) -> list[dict]:
    nodes: dict[str, dict] = {}
    for n in ((reg.get("cluster") or {}).get("nodes") or []):
        if isinstance(n, dict) and n.get("name"):
            nodes[n["name"]] = {"name": n["name"], "gpus": int(n.get("gpus") or 0),
                                "pairs": [list(map(int, p)) for p in (n.get("two_gpu_slots") or []) if len(p) == 2]}
    for g in gpu_ids:  # GPUs seen in the intervals but not in the registry
        node, _, i = g.rpartition("/")
        if not i.isdigit():
            continue
        e = nodes.setdefault(node, {"name": node, "gpus": 0, "pairs": []})
        e["gpus"] = max(e["gpus"], int(i) + 1)
    for e in nodes.values():
        if not e["pairs"]:
            e["pairs"] = [[i, i + 1] for i in range(0, e["gpus"] - 1, 2)]
    short = _short_names(sorted(nodes))
    return [{**nodes[n], "short": short[n]} for n in sorted(nodes, key=lambda x: (len(x), x))]


# ---------------------------------------------------------------- per-arm series
def build_ts(rows: list[dict], models: list[str]) -> dict:
    ts_all = [int(float(r["t"])) for r in rows if _f(r.get("t")) is not None]
    if not ts_all:
        return {"t0": 0, "n": 0, "models": {}}
    t0, t1 = min(ts_all), max(ts_all)
    n = t1 - t0 + 1
    out: dict[str, dict] = {}
    for r in rows:
        m, t = r.get("model"), _f(r.get("t"))
        if t is None or not m:
            continue
        f = out.get(m)
        if f is None:
            f = out[m] = {k: [None] * n for k in list(TS_COPY.values()) + ["viol_pct"]}
        i = int(t) - t0
        for src, dst in TS_COPY.items():
            v = _f(r.get(src))
            if v is not None:
                f[dst][i] = _r(v)
        v = _f(r.get("viol_frac_10s"))
        if v is not None:
            f["viol_pct"][i] = _r(100.0 * v)
    order = [m for m in models if m in out] + sorted(m for m in out if m not in models)
    return {"t0": t0, "n": n, "models": {m: out[m] for m in order}}


def add_tpot(ts: dict, req_rows: list[dict] | None) -> bool:
    """TPOT p95 per 10 s send-time bin on the scored ok requests: the report's TTFT rule
    (timeseries.build: bins of t_send // 10, metrics.pct, drawn as a step over the bin)."""
    from .metrics import pct
    if not req_rows or not ts["n"]:
        return False
    by: dict[tuple, list] = defaultdict(list)
    for r in req_rows:
        t, v = _f(r.get("t_send")), _f(r.get("tpot_ms"))
        if t is None or r.get("viol", "") == "" or not _bool(r.get("ok")) or v is None:
            continue
        by[(r.get("model"), int(t // 10))].append(v)
    t0, n = ts["t0"], ts["n"]
    for m, f in ts["models"].items():
        f["tpot_p95_ms"] = [None] * n
    for (m, b), vs in by.items():
        f = ts["models"].get(m)
        if f is None:
            continue
        p = pct(vs, 95)
        for i in range(max(b * 10 - t0, 0), min(b * 10 + 10 - t0, n)):
            f["tpot_p95_ms"][i] = _r(p)
    return True


def add_reissues(ts: dict, ev_rows: list[dict]) -> None:
    t0, n = ts["t0"], ts["n"]
    for f in ts["models"].values():
        f["reissues"] = [0] * n if n else []
    for r in ev_rows:
        if r.get("source") != "sidecar" or not str(r.get("kind", "")).startswith("reissue"):
            continue
        t, f = _f(r.get("t")), ts["models"].get(r.get("model"))
        if t is None or f is None:
            continue
        i = int(math.floor(t)) - t0
        if 0 <= i < n:
            f["reissues"][i] += 1


# ---------------------------------------------------------------- events
_SM_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) +\w+ +[\w.]+: (.*)$")
_KUBELET = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d) (.*)$")
_REFUSED = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) +\w+ +[\w.]+ SM refused (\S+) (\w+)->(\d+): (\{.*\})\s*$")


def _sm_lines(raw: str | None, clock: _Clock) -> dict[tuple, list[tuple[float, str]]]:
    """(kind, binding) -> [(t_rel, original SM line)] from sm.ts.log (wake / sleep records only)."""
    out: dict[tuple, list] = defaultdict(list)
    p = os.path.join(raw, "sm.ts.log") if raw else None
    if not p or not os.path.exists(p):
        return out
    with open(p, errors="replace") as fh:
        for line in fh:
            mk = _KUBELET.match(line.rstrip("\n"))
            body = mk.group(4) if mk else line.rstrip("\n")
            mt = _SM_LINE.match(body)
            if not mt or ('"wake_' not in body and "sleep" not in body):
                continue
            i = mt.group(3).find("{")
            if i < 0:
                continue
            try:
                rec = json.loads(mt.group(3)[i:])
            except ValueError:
                continue
            if not isinstance(rec, dict) or not rec.get("binding_id"):
                continue
            kind = rec.get("event") or (mt.group(3)[:i].strip() or None)
            t = clock.rel(_utc_epoch(mt.group(1), mt.group(2)), "service-manager")
            if t is not None:
                out[(kind, rec["binding_id"])].append((t, body[:LOG_MAX]))
    return out


def _nearest_line(index: dict, kind: str, binding: str | None, t: float | None) -> str | None:
    best = None
    for tt, line in index.get((kind, binding), []):
        if t is not None and abs(tt - t) <= 2.0 and (best is None or abs(tt - t) < best[0]):
            best = (abs(tt - t), line)
    return best[1] if best else None


def _refusals(raw: str | None, clock: _Clock, sm409: list[dict]) -> list[dict] | None:
    """baseline-scaler.log 'SM refused' lines; time taken from the matching SM 409 access line
    (reference clock) when one is within 5 s, else the scaler stamp. None when there is no log."""
    p = os.path.join(raw, "baseline-scaler.log") if raw else None
    if not p or not os.path.exists(p):
        return None
    out = []
    used: set[int] = set()
    with open(p, errors="replace") as fh:
        for line in fh:
            mt = _REFUSED.match(line.rstrip("\n"))
            if not mt:
                continue
            try:
                d = ast.literal_eval(mt.group(6))
            except (ValueError, SyntaxError):
                d = {"raw": mt.group(6)}
            if not isinstance(d, dict):
                d = {"raw": mt.group(6)}
            t = clock.rel(_utc_epoch(mt.group(1), mt.group(2)), "baseline-scaler")
            model = mt.group(3)
            j = min((k for k, e in enumerate(sm409) if k not in used and e["model"] == model and t is not None
                     and abs(e["t"] - t) <= 5.0), key=lambda k: abs(sm409[k]["t"] - t), default=None)
            if j is not None:
                used.add(j)
                t = sm409[j]["t"]
            out.append({"t": t, "model": model, "binding": d.get("binding_id"),
                        "reason": d.get("reason") or d.get("error"), "dir": mt.group(4), "to": int(mt.group(5)),
                        "detail": d, "log": line.rstrip("\n")[:LOG_MAX]})
    return out


def build_events(ev_rows: list[dict], raw: str | None, clock: _Clock) -> list[dict]:
    smidx = _sm_lines(raw, clock)
    ev: list[dict] = []
    open_wake: dict[str, dict] = {}
    sm409 = []
    for r in ev_rows:
        t, src, kind, model = _f(r.get("t")), r.get("source"), r.get("kind", ""), r.get("model") or None
        if t is None:
            continue
        det = _jcell(r.get("detail"))
        det = det if isinstance(det, dict) else ({} if det is None else {"detail": det})
        b = r.get("target") or None
        if kind in ("decision_up", "decision_down"):
            reason = det.get("reason")
            if reason is None and "desired" in det:
                reason = f"desired {det.get('prev')} -> {det.get('desired')}"
            ev.append({"t": _rt(t), "kind": kind, "model": model, "src": src, "reason": reason,
                       "value": _f(r.get("value")), "detail": det})
        elif src == "sm" and kind == "wake_start":
            e = {"t": _rt(t), "t_end": None, "kind": "wake", "model": model, "binding": b, "src": "sm",
                 "reason": "started", "detail": {"start": det}, "log": _nearest_line(smidx, "wake_start", b, t)}
            open_wake[b] = e
            ev.append(e)
        elif src == "sm" and kind in ("wake_done", "wake_failed"):
            e = open_wake.pop(b, None)
            if e is None:
                e = {"t": _rt(t), "kind": "wake", "model": model, "binding": b, "src": "sm", "detail": {}}
                ev.append(e)
            e["t_end"] = _rt(t)
            e["reason"] = "done" if kind == "wake_done" else "failed"
            e["value"] = _f(r.get("value"))
            e["detail"]["done" if kind == "wake_done" else "failed"] = det
            line = _nearest_line(smidx, kind, b, t)
            if line:
                e["log"] = (e.get("log") + "\n" if e.get("log") else "") + line
        elif src == "sm" and kind == "sleep":
            ev.append({"t": _rt(t), "kind": "sleep", "model": model, "binding": b, "src": "sm",
                       "reason": det.get("path"), "value": _f(r.get("value")), "detail": det,
                       "log": _nearest_line(smidx, "sleep", b, t)})
        elif src == "tre_counterfactual":
            ev.append({"t": _rt(t), "kind": "counterfactual", "model": model, "src": "tre (observe)",
                       "reason": det.get("reason"), "value": _f(r.get("value")), "detail": det})
        elif src == "sm_api" and str(r.get("value")) == "409":
            sm409.append({"t": t, "model": model, "path": b, "kind": kind})
    ref = _refusals(raw, clock, sm409)
    if ref is None:
        for e in sm409:
            ev.append({"t": _rt(e["t"]), "kind": "refusal", "model": e["model"], "src": "sm_api", "reason": "409",
                       "detail": {"request": e["kind"], "path": e["path"]}})
    else:
        for e in ref:
            ev.append({"t": _rt(e["t"]), "kind": "refusal", "model": e["model"], "binding": e["binding"],
                       "src": "baseline", "reason": e["reason"], "detail": {**e["detail"], "request": f"{e['dir']} -> {e['to']}"},
                       "log": e["log"]})
    for e in open_wake.values():
        e.setdefault("reason", "no wake_done")
    ev.sort(key=lambda e: (e["t"], e["kind"]))
    for i, e in enumerate(ev):
        e["id"] = i
        e.update({k: None for k in ("t_end", "binding", "value", "log") if k not in e})
    return ev


def _binding_from_pod(pod: str, model: str, nodes: list[str]) -> str | None:
    for node in sorted(nodes, key=len, reverse=True):
        mt = re.search(re.escape(node) + r"-gpu-(\d+(?:-\d+)*)-", pod)
        if mt:
            return f"{model}/{node}/{mt.group(1).replace('-', ',')}"
    return None


def build_probes(raw: str | None, clock: _Clock, nodes: list[str]) -> list[dict] | None:
    """SafeScale probe spans from safescale.json (None when the file is absent)."""
    d = _json(os.path.join(raw, "safescale.json")) if raw else None
    if not isinstance(d, dict):
        return None
    out = []
    for key, v in (d.get("probes") or {}).items():
        p = v
        if isinstance(v, str):
            try:
                p = json.loads(v)
            except ValueError:
                continue
        if not isinstance(p, dict):
            continue
        td = p.get("terminal_details") or {}
        rb = td.get("rollback_reason") if isinstance(td.get("rollback_reason"), dict) else {}
        t0 = clock.rel((_f(p.get("start_wall_ms")) or _f(p.get("start_ms")) or 0) / 1000.0 or None, "controller")
        anchor = p.get("hide_anchor") or {}
        t_hide = clock.rel(_f(anchor.get("ts_ms")) / 1000.0 if _f(anchor.get("ts_ms")) else None,
                           "redis" if anchor.get("source") == "redis_time" else "controller")
        t1 = clock.rel(_f(p.get("resolved_ts")), "controller")
        outcome = p.get("resolution") or p.get("status") or "unknown"
        if rb.get("code"):
            reason = rb["code"]
        elif p.get("terminal_reason"):
            reason = p["terminal_reason"] if isinstance(p["terminal_reason"], str) else json.dumps(p["terminal_reason"])
        elif td.get("idle_commit"):
            reason = "idle_commit"
        else:
            reason = outcome
        pods = p.get("pods") or []
        model = p.get("model")
        detail = {k: p.get(k) for k in ("request_id", "status", "resolution", "start_z_m", "start_routable",
                                        "hidden_count", "preempt_reason", "pods") if p.get(k) is not None}
        for k in ("threshold_mode", "ttft_threshold_ms", "tpot_threshold_ms", "latency_gate", "latency_skip_reason",
                  "latency_samples", "latency_violations", "evidence_source_used", "evidence_ttft_p95_ms",
                  "evidence_tpot_p95_ms", "kv_direct", "extensions", "clamped", "idle_commit", "low_sample_reason",
                  "probe_wall_clock_ms", "gate_failures", "donor_health"):
            if td.get(k) not in (None, [], {}):
                detail[k] = td[k]
        if rb:
            detail["rollback_reason"] = rb
        out.append({"model": model, "t0": _rt(t0), "t_hide": _rt(t_hide), "t1": _rt(t1), "outcome": outcome,
                    "reason": reason, "binding": _binding_from_pod(pods[0], model, nodes) if pods else None,
                    "detail": detail})
    out.sort(key=lambda p: (p["t0"] is None, p["t0"] or 0))
    for i, p in enumerate(out):
        p["id"] = i
    return out


def build_gpus(rows: list[dict] | None) -> dict[str, list]:
    out: dict[str, list] = defaultdict(list)
    for r in rows or []:
        t0, t1 = _f(r.get("t0")), _f(r.get("t1"))
        if t0 is None or t1 is None or not r.get("gpu"):
            continue
        hid = _jcell(r.get("hidden")) or []
        out[r["gpu"]].append([_rt(t0), _rt(t1), r.get("model"), r.get("binding"),
                              [[_rt(a), _rt(b)] for a, b in hid if a is not None and b is not None],
                              _bool(r.get("open_end"))])
    for v in out.values():
        v.sort(key=lambda x: x[0])
    return dict(out)


def fragmentation(ts: dict, gpus: dict[str, list], nodes: list[dict], tp: dict[str, int]) -> list[dict]:
    """Seconds where a TP>=2 model is below target, >= tp GPUs are free, but no aligned slot is free."""
    n, t0 = ts["n"], ts["t0"]
    if not n or not nodes:
        return []
    universe = [f"{nd['name']}/{i}" for nd in nodes for i in range(nd["gpus"])]
    busy = {g: [False] * n for g in universe}
    for g, ivs in gpus.items():
        if g not in busy:
            continue
        for iv in ivs:
            a = max(int(math.ceil(iv[0] - 0.5)) - t0, 0)       # second s is busy if t0 <= s + 0.5 < t1
            b = min(int(math.ceil(iv[1] - 0.5)) - t0, n)
            for i in range(a, b):
                busy[g][i] = True
    slots = [[f"{nd['name']}/{a}", f"{nd['name']}/{b}"] for nd in nodes for a, b in nd["pairs"]]
    out = []
    for m, f in ts["models"].items():
        k = tp.get(m, 1)
        if k < 2:
            continue
        cur = None
        for i in range(n):
            aw, tg = f["awake"][i], f["target"][i]
            free = sum(1 for g in universe if not busy[g][i])
            hit = (aw is not None and tg is not None and aw < tg and free >= k
                   and not any(not busy[a][i] and not busy[b][i] for a, b in slots))
            if hit and cur is None:
                cur = {"model": m, "t0": t0 + i, "t1": t0 + i + 1, "free": free}
            elif hit:
                cur["t1"] = t0 + i + 1
                cur["free"] = max(cur["free"], free)
            elif cur is not None:
                out.append(cur)
                cur = None
        if cur is not None:
            out.append(cur)
    return out


# ---------------------------------------------------------------- assemble
def _dash(ls: Any) -> list:
    if isinstance(ls, str):
        return DASHES.get(ls, [])
    if isinstance(ls, tuple) and len(ls) == 2 and isinstance(ls[1], tuple):
        return [max(1.5, x * 1.5) for x in ls[1]]
    return []


def _models_meta(reg: dict, models: list[str]) -> list[dict]:
    by = {m.get("name"): m for m in (reg.get("models") or []) if isinstance(m, dict)}
    out = []
    for i, name in enumerate(models):
        r = by.get(name) or {}
        c = S.MODEL_COLORS[i % len(S.MODEL_COLORS)]
        slo, trs = r.get("slo") or {}, r.get("trs") or {}
        tau = {k: trs.get(f"tau_{k}") for k in ("crit", "low", "high") if trs.get(f"tau_{k}") is not None}
        out.append({"name": name, "tp": int(r.get("tp_size") or 1),
                    "slo": {k: slo.get(k) for k in ("ttft_p95_ms", "tpot_p95_ms") if slo.get(k) is not None},
                    "tau": tau or None, "max_awake": r.get("max_awake_replicas"),
                    "color": {"light": c, "dark": S.DARK_STEPS.get(c, c)}})
    return out


def build(report_dir: str, trace: str | None = None, raw_root: str | None = None) -> dict:
    """The page's data object for one trace of a report directory."""
    found = discover(report_dir)
    if not found:
        raise FileNotFoundError(f"{report_dir}: no arms/<arm>/timeseries_1s.csv (not a tre_eval.report --out dir?)")
    if trace is None:
        if len(found) > 1:
            raise ValueError(f"{report_dir} holds several traces {sorted(found)}: pass --trace")
        trace = next(iter(found))
    arms_in = found[trace]
    warnings: list[str] = []
    comp = _csv(os.path.join(report_dir, "comparison.csv"))
    if comp is None:
        warnings.append("comparison.csv missing: no V_req / GPU summary in the header")
    validation = _json(os.path.join(report_dir, "validation.json"))
    if validation is None:
        warnings.append("validation.json missing: no validation status in the header")
    onsets = _csv(os.path.join(report_dir, "onsets.csv")) or []
    dps = _csv(os.path.join(report_dir, "decision_points.csv")) or []
    styles = S.ArmStyles([(a["name"], a["manifest"].get("label") or a["name"]) for a in arms_in])

    # registry: the first arm with one (all arms of a campaign share models and topology)
    reg, models = {}, []
    raws = {a["name"]: raw_dir(report_dir, a, raw_root) for a in arms_in}
    for a in arms_in:
        reg = _registry(raws[a["name"]])
        if reg:
            break
    if not reg:
        warnings.append("live-registry.yaml not found next to any arm: TP size, SLO lines, tau lines "
                        "and GPU topology inferred or left out")
    models = [m.get("name") for m in (reg.get("models") or []) if isinstance(m, dict) and m.get("name")]

    arms_out, all_gpu_ids = [], set()
    for a in arms_in:
        d, man, raw = a["dir"], a["manifest"], raws[a["name"]]
        aw: list[str] = []
        warn = aw.append
        ts_rows = _csv(os.path.join(d, "timeseries_1s.csv")) or []
        ts = build_ts(ts_rows, models)
        for m in ts["models"]:
            if m not in models:
                models.append(m)
        if not add_tpot(ts, _csv(os.path.join(d, "requests.csv"))):
            warn("requests.csv missing: no TPOT p95 series")
        ev_rows = _csv(os.path.join(d, "events.csv"))
        if ev_rows is None:
            warn("events.csv missing: no scale events")
        add_reissues(ts, ev_rows or [])
        clock = _Clock(man)
        if raw is None:
            warn("raw arm directory not found: no SafeScale probes, refusal reasons or raw log lines")
        gpus = build_gpus(_csv(os.path.join(d, "gpu_intervals.csv")))
        if not gpus:
            warn("gpu_intervals.csv missing or empty: no GPU layout")
        all_gpu_ids.update(gpus)
        node_names = sorted({g.rpartition("/")[0] for g in gpus} |
                            {n.get("name") for n in ((reg.get("cluster") or {}).get("nodes") or []) if isinstance(n, dict)})
        events = build_events(ev_rows or [], raw, clock)
        probes = build_probes(raw, clock, node_names)
        fam = S.family(a["name"], man.get("label") or "")
        if probes is None:
            probes = []
            if fam == "tre" and raw is not None:
                warn("safescale.json missing: no SafeScale probe spans")
        summ = _json(os.path.join(d, "summary.json")) or {}
        st = styles[a["name"]]
        light = st["color"]
        arms_out.append({
            "name": a["name"], "label": man.get("label") or a["name"], "sub": a["sub"], "family": fam,
            "decision_source": man.get("decision_source"),
            "color": {"light": light, "dark": S.DARK_STEPS.get(light, light)}, "dash": _dash(st["ls"]),
            "summary": {r["model"]: {k: (_f(r.get(k)) if k != "valid" else r.get(k)) for k in SUMMARY_KEYS if k in r}
                        for r in (comp or []) if r.get("arm") == a["name"]},
            "validation": (validation or {}).get(a["name"]),
            "warnings": list(summ.get("warnings") or []) + aw,
            "controller_env": (man.get("arm_meta") or {}).get("controller_env"),
            "tre_sha": man.get("tre_sha"), "load_s": man.get("load_s"),
            "ts": ts, "events": events, "probes": probes, "gpus": gpus})
    meta = _models_meta(reg, models)
    tp = {m["name"]: m["tp"] for m in meta}
    for m in meta:  # TP from the binding width when the registry is missing
        if not reg:
            widths = [len(parse_binding(iv[3])[2]) for a in arms_out for ivs in a["gpus"].values() for iv in ivs
                      if iv[2] == m["name"] and iv[3]]
            m["tp"] = tp[m["name"]] = max(widths or [1])
    nodes = topology(reg, sorted(all_gpu_ids))
    for a in arms_out:
        a["frag"] = fragmentation(a["ts"], a["gpus"], nodes, tp)
    first = arms_in[0]["manifest"]
    return {"schema": SCHEMA, "tool": "tre_eval.viewer",
            "generated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "trace": trace, "seed": first.get("trace_seed"), "report_dir": os.path.basename(os.path.abspath(report_dir)),
            "models": meta, "nodes": nodes,
            "onsets": [{k: (_f(v) if k in ("t_on", "t_off", "lat_decision_s", "lat_awake_s", "lat_donor_s") else v)
                        for k, v in r.items()} for r in onsets],
            "decision_points": dps, "warnings": warnings, "arms": arms_out}


# ---------------------------------------------------------------- html
def _asset(name: str) -> str:
    with open(os.path.join(ASSETS, name), encoding="utf-8") as fh:
        return fh.read()


def render(data: dict) -> str:
    page = _asset("viewer.html")
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False, default=str)
    blob = blob.replace("</", "<\\/")  # a JSON string must not close the <script> element
    title = f"{data['trace']} · results viewer"
    for k, v in (("/*__UPLOT_CSS__*/", _asset("uPlot.min.css")), ("/*__VIEWER_CSS__*/", _asset("viewer.css")),
                 ("/*__UPLOT_JS__*/", _asset("uPlot.iife.min.js")), ("/*__CORE_JS__*/", _asset("viewer_core.js")),
                 ("/*__APP_JS__*/", _asset("viewer_app.js")), ("__TITLE__", html.escape(title))):
        page = page.replace(k, v)
    return page.replace("__DATA_JSON__", blob)


def write(report_dir: str, out: str | None = None, trace: str | None = None, raw_root: str | None = None) -> str:
    data = build(report_dir, trace=trace, raw_root=raw_root)
    out = out or os.path.join(report_dir, "viewer.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(render(data))
    return out


def write_for_report(report_dir: str) -> list[str]:
    """report.py hook: viewer.html (one trace) or viewer-<trace>.html (several) next to index.html."""
    found = discover(report_dir)
    if len(found) == 1:
        return [write(report_dir)]
    return [write(report_dir, os.path.join(report_dir, f"viewer-{re.sub(r'[^A-Za-z0-9_.-]', '_', t)}.html"), trace=t)
            for t in sorted(found)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report_dir", help="tre_eval.report --out directory of one trace")
    ap.add_argument("-o", "--out", help="output HTML (default <report_dir>/viewer.html)")
    ap.add_argument("--trace", help="trace name when the report holds several traces")
    ap.add_argument("--raw-root", help="directory holding the raw arm directories (default: manifest dir, "
                                       "else the report's parent)")
    args = ap.parse_args(argv)
    try:
        out = write(args.report_dir, args.out, trace=args.trace, raw_root=args.raw_root)
    except (FileNotFoundError, ValueError) as e:
        print(f"viewer: {e}", file=sys.stderr)
        return 2
    print(f"viewer: {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

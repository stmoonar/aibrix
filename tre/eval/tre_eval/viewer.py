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
from .load import parse_binding, _utc_epoch

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
# what the `target` column of timeseries_1s.csv is, per decision source (metrics.targets): not comparable across arms
TARGET_SOURCE = {"tre": "signal_log replicas_target", "baseline": "scaler clamped", "apa": "APA desiredScale"}
LOG_MAX = 700          # chars of one raw log line attached to an event
PROBE_GRACE_S = 90     # probes kept up to load end + this (the report's time grid ends at load end + 90 s)
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


def _finite(o: Any) -> Any:
    """NaN / ±Inf -> None, recursively (one bad float in a detail must not abort the export)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


# ---------------------------------------------------------------- discovery
def discover(report_dir: str, subs: list[str] | None = None) -> dict[str, list[dict]]:
    """trace -> [{name, sub, dir, manifest}] in comparison.csv order (else directory order).
    ``subs``: only these ``arms/<sub>`` folders (the report.py hook passes the ones it just wrote)."""
    out: dict[str, list[dict]] = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(report_dir, "arms", "*"))):
        sub = os.path.basename(d)
        if (subs is not None and sub not in subs) or not os.path.exists(os.path.join(d, "timeseries_1s.csv")):
            continue
        man = _json(os.path.join(d, "manifest.json")) or {}
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
    """--raw-root/<arm>, else manifest.dir, else <report>/../<arm>; the first that exists."""
    cands = []
    if raw_root:
        cands.append(os.path.join(raw_root, arm["name"]))
    if arm["manifest"].get("dir"):
        cands.append(arm["manifest"]["dir"])
    cands.append(os.path.join(os.path.dirname(os.path.abspath(report_dir)), arm["name"]))
    return next((c for c in cands if os.path.isdir(c)), None)


class _Clock:
    """Time base of the raw sources, the rule of load.py exactly (so every marker lines up with the
    report's events.csv): the SM log is shifted by the clock offset of the service-manager node;
    controller and baseline-scaler timestamps are NOT corrected (load.py takes controller ticks and
    bl_decisions as written). ``skew_warnings`` says when that uncorrected rule is visibly risky."""

    def __init__(self, man: dict):
        self.off = {k: float(v) for k, v in (man.get("clock_offsets_s") or {}).items() if isinstance(v, (int, float))}
        self.nodes = {k: str(v) for k, v in (man.get("component_nodes") or {}).items() if v}
        self.t_load = _f(man.get("load_start_epoch"))

    def rel(self, epoch: float | None, correct_for: str | None = None) -> float | None:
        if epoch is None or self.t_load is None:
            return None
        off = self.off.get(self.nodes.get(correct_for, ""), 0.0) if correct_for else 0.0
        return epoch - off - self.t_load

    WHAT = {"controller": "decisions and SafeScale probes", "baseline-scaler": "decisions",
            "aibrix-controller-manager": "APA evaluation stamps (KV panel)"}
    CORRECTED = {"aibrix-controller-manager"}  # the viewer shifts this log by its node offset when the node is known

    def skew_warnings(self, components: list[str]) -> list[str]:
        out = []
        any_off = any(abs(v) > 0.5 for v in self.off.values())
        for c in components:
            node, what = self.nodes.get(c), self.WHAT.get(c, "timestamps")
            if node is not None and abs(self.off.get(node, 0.0)) > 0.5 and c not in self.CORRECTED:
                out.append(f"{c} runs on {node}, {self.off[node]:+.1f} s from the reference clock; its {what} are "
                           "not corrected (load.py rule) and may be shifted against the replica / latency series")
            elif node is None and any_off:
                out.append(f"node of {c} unknown while node clock offsets are non-zero "
                           f"({', '.join(f'{k} {v:+.1f} s' for k, v in sorted(self.off.items()))}); its {what} are "
                           "not corrected and may be shifted")
        return out


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


def topology(reg: dict, gpu_ids: list[str], warn=None) -> list[dict]:
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
    fallback = []
    for e in nodes.values():
        if not e["pairs"]:
            e["pairs"] = [[i, i + 1] for i in range(0, e["gpus"] - 1, 2)]
            fallback.append(e["name"])
    if fallback and warn:
        warn(f"no two_gpu_slots for {', '.join(sorted(fallback))}: aligned slots assumed to be (0,1), (2,3), ...")
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
    """(kind, binding) -> [(t_rel, original SM line)] from sm.ts.log (wake / sleep records only).
    Same time base as load._parse_sm: the SM's own stamp, corrected by the service-manager node offset."""
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


# Pairing window for (scaler log stamp - SM 409 access line), seconds. The scaler stamps its line after the
# HTTP call returns; the SM writes its access line when it answers. E1 10-19 data: -0.2 .. +5.0 s, the same
# for slow calls (elapsed_s up to 5.4 s), i.e. the SM line sits at the response, not at the request. The
# upper bound adds elapsed_s so a pairing still works if the SM line marks the request instead.
PAIR_LO, PAIR_HI = -1.0, 8.0


def _match(sm409: list[dict], items: list[tuple[float, str, float]]) -> list[int | None]:
    """Greedy one-to-one pairing of (scaler time, model, elapsed_s) items with SM 409 lines of the same model,
    scaler - SM in [PAIR_LO, PAIR_HI + elapsed_s], the closest first."""
    used: set[int] = set()
    out: list[int | None] = []
    for t, model, el in items:
        j = min((k for k, e in enumerate(sm409) if k not in used and e["model"] == model
                 and PAIR_LO <= t - e["t"] <= PAIR_HI + el), key=lambda k: abs(t - sm409[k]["t"]), default=None)
        if j is not None:
            used.add(j)
        out.append(j)
    return out


def _refusals(raw: str | None, clock: _Clock, sm409: list[dict], warn) -> tuple[list[dict], bool] | None:
    """baseline-scaler.log 'SM refused' lines (None when there is no log), each paired with its SM 409
    access line (events.csv, SM clock corrected by load.py) and drawn at that SM time: the same reference
    clock as the decision markers. The log is stamped by the scaler host's clock, which load.py never reads:
    we shift it by the baseline-scaler node offset when the node is known, else by the candidate node offset
    (or none) under which most lines pair with an SM 409. The inferred shift is trusted only when it is
    clearly better than the runner-up (>= 3 pairs and >= 2x the runner-up's pairs); otherwise every scaler
    line is 'time uncertain' (event table only) and the second return value is False."""
    p = os.path.join(raw, "baseline-scaler.log") if raw else None
    if not p or not os.path.exists(p):
        return None
    lines = []
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
            t = clock.rel(_utc_epoch(mt.group(1), mt.group(2)))
            if t is not None:
                lines.append((t, mt, d, line.rstrip("\n")[:LOG_MAX], max(_f(d.get("elapsed_s")) or 0.0, 0.0)))

    def pair(off: float) -> list[int | None]:
        return _match(sm409, [(t - off, mt.group(3), el) for t, mt, _, _, el in lines])

    node = clock.nodes.get("baseline-scaler")
    cands = [clock.off.get(node, 0.0)] if node is not None else \
        [0.0] + sorted({v for v in clock.off.values() if abs(v) > 0.5})
    counts = {off: sum(j is not None for j in pair(off)) for off in cands}
    ranked = sorted(cands, key=lambda off: (-counts[off], abs(off)))
    best = ranked[0]
    sure = True
    if len(cands) > 1 and lines:
        margin = ", ".join(f"{counts[o]}/{len(lines)} at {0.0 - o:+.1f} s" for o in ranked)
        sure = counts[best] >= 3 and counts[best] >= 2 * counts[ranked[1]]
        if sure:
            warn(f"baseline-scaler.log stamps shifted by {-best:+.1f} s (scaler's node unknown; pairs with SM 409 "
                 f"lines: {margin})")
        else:
            warn(f"baseline-scaler.log clock unknown (scaler's node unknown; no clearly best shift: {margin}): its "
                 f"{len(lines)} refusal(s) are listed in the event table as 'time uncertain' and not drawn; the SM 409 "
                 "access lines are drawn instead")
    pairs = pair(best) if sure else [None] * len(lines)
    out = []
    for (t, mt, d, log, _), j in zip(lines, pairs):
        if j is not None:
            d = {**d, "scaler_log_t": _rt(t - best)}
        out.append({"t": sm409[j]["t"] if j is not None else t - best, "model": mt.group(3),
                    "binding": d.get("binding_id"), "matched": j is not None, "uncertain": not sure,
                    "reason": d.get("reason") or d.get("error"), "dir": mt.group(4), "to": int(mt.group(5)),
                    "detail": d, "log": log})
    if sure:
        n_sm = len(sm409) - sum(j is not None for j in pairs)
        n_sc = sum(j is None for j in pairs)
        if n_sm:
            warn(f"{n_sm} SM 409 response(s) without a matching baseline-scaler 'SM refused' line (not drawn)")
        if n_sc:
            warn(f"{n_sc} baseline-scaler refusal(s) without a matching SM 409 access line (drawn at the shifted log stamp)")
    return out, sure


def build_events(ev_rows: list[dict], raw: str | None, clock: _Clock, warn=lambda m: None) -> list[dict]:
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
            prev = open_wake.pop(b, None)
            if prev is not None:  # a second start before any done: the first one never finished
                prev["reason"] = "no wake_done (another wake_start followed)"
            e = {"t": _rt(t), "t_end": None, "kind": "wake", "model": model, "binding": b, "src": "sm",
                 "reason": "no wake_done", "detail": {"start": det}, "log": _nearest_line(smidx, "wake_start", b, t)}
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
        elif src == "sm_api" and str(r.get("value")) == "409" and not (b == "/v2/plans" and model is None):
            # A plan's own 409 access line has no model; its refused items come as per-model
            # plan_item_refused records (load._parse_sm), which pair instead.
            sm409.append({"t": t, "model": model, "path": b, "kind": kind})
    got = _refusals(raw, clock, sm409, warn)
    ref, sure = got if got is not None else ([], False)
    if not sure:  # no scaler log, or its clock is uncertain: the SM 409 lines (reference clock) are the markers
        for e in sm409:
            ev.append({"t": _rt(e["t"]), "kind": "refusal", "model": e["model"], "src": "sm_api", "reason": "409",
                       "detail": {"request": e["kind"], "path": e["path"]}})
    for e in ref:
        x = {"t": _rt(e["t"]), "kind": "refusal", "model": e["model"], "binding": e["binding"],
             "src": "baseline", "reason": e["reason"],
             "detail": {**e["detail"], "request": f"{e['dir']} -> {e['to']}"}, "log": e["log"]}
        if e["uncertain"]:
            x["uncertain"] = True  # event table only, never on a time axis
        ev.append(x)
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


def build_probes(raw: str | None, clock: _Clock, nodes: list[str], window: tuple[float, float],
                 warn=lambda m: None) -> list[dict] | None:
    """SafeScale probe spans from safescale.json (None when the file is absent). Only probes that overlap
    ``window`` are kept: an old safescale.json without ``probes_filter`` also holds earlier arms' probes."""
    d = _json(os.path.join(raw, "safescale.json")) if raw else None
    if not isinstance(d, dict):
        return None
    out, dropped = [], 0
    for v in (d.get("probes") or {}).values():
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
        # Probe times are controller wall clock. load.py does not correct the controller clock (its ticks,
        # hence the TRE decision markers, are used as written), so neither do we: a probe then starts
        # exactly at the decision that opened it. The hide time is taken from the same clock
        # (hide_anchor.controller_ts_ms) rather than the Redis stamp, so the span stays internally
        # consistent whatever the Redis node's offset is.
        t0 = clock.rel(((_f(p.get("start_wall_ms")) or _f(p.get("start_ms"))) or 0) / 1000.0 or None)
        anchor = p.get("hide_anchor") or {}
        hide_ms = _f(anchor.get("controller_ts_ms"))
        t_hide = clock.rel(hide_ms / 1000.0) if hide_ms else None
        t1 = clock.rel(_f(p.get("resolved_ts")))
        if t0 is None or (t1 if t1 is not None else t0) < window[0] or t0 > window[1]:
            dropped += 1
            continue
        outcome = p.get("resolution") or p.get("status") or "unknown"
        if rb.get("code"):
            reason = rb["code"]
        elif td.get("idle_commit"):  # the controller writes terminal_reason too ("formal_commit_gate_passed")
            reason = "idle_commit"
        elif p.get("terminal_reason"):
            reason = p["terminal_reason"] if isinstance(p["terminal_reason"], str) else json.dumps(p["terminal_reason"])
        else:
            reason = outcome
        pods = p.get("pods") or []
        model = p.get("model")
        detail = {k: p.get(k) for k in ("request_id", "status", "resolution", "terminal_reason", "start_z_m",
                                        "start_routable", "hidden_count", "preempt_reason", "pods") if p.get(k) is not None}
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
    if dropped:
        warn(f"safescale.json: {dropped} probe(s) outside this arm's time window dropped (earlier arms' probes)")
    out.sort(key=lambda p: p["t0"])
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


def _slots(node: dict, k: int) -> list[list[int]]:
    if k == 2:
        return node["pairs"]
    return [list(range(i, i + k)) for i in range(0, node["gpus"] - k + 1, k)]  # aligned blocks of k


def fragmentation(t0: int, n: int, gpus: dict[str, list], nodes: list[dict], tps: list[int]) -> list[dict]:
    """Stranded TP slots, from the layout only (the same rule for every arm): seconds where at least k
    GPUs are free (no awake binding) for some TP size k >= 2 in use, but no aligned free k-slot exists.
    A second s counts as busy on a GPU when an interval covers s + 0.5."""
    if not n or not nodes:
        return []
    universe = [f"{nd['name']}/{i}" for nd in nodes for i in range(nd["gpus"])]
    busy = {g: [False] * n for g in universe}
    for g, ivs in gpus.items():
        if g not in busy:
            continue
        for iv in ivs:
            a = max(int(math.ceil(iv[0] - 0.5)) - t0, 0)
            b = min(int(math.ceil(iv[1] - 0.5)) - t0, n)
            for i in range(a, b):
                busy[g][i] = True
    out = []
    for k in sorted({k for k in tps if k >= 2}):
        slots = [[f"{nd['name']}/{i}" for i in s] for nd in nodes for s in _slots(nd, k)]
        cur = None
        for i in range(n):
            free = sum(1 for g in universe if not busy[g][i])
            hit = free >= k and not any(all(not busy[g][i] for g in s) for s in slots)
            if hit and cur is None:
                cur = {"tp": k, "t0": t0 + i, "t1": t0 + i + 1, "free": free}
            elif hit:
                cur["t1"] = t0 + i + 1
                cur["free"] = max(cur["free"], free)
            elif cur is not None:
                out.append(cur)
                cur = None
        if cur is not None:
            out.append(cur)
    return out


# ---------------------------------------------------------------- APA window value (KV panel)
# What APA decides on: per evaluation the cross-pod mean of kv_cache_usage_perc over the routable pods,
# averaged over the samples in its stable window; scale up when value / target > 1 + up_tol, down when
# value / target < 1 - down_tol (pkg/controller/podautoscaler/algorithm/apa.go). Window and tolerances come
# from the controller's "Effective autoscaling config" log line (what it actually used), else the recorded
# CR; the target from the CR only. A value found in neither source is left unset: no trigger line, no zone.
APA_KEYS = ("window_s", "up_tol", "down_tol")
_KLOG = re.compile(r"^[IWE](\d\d)(\d\d) (\d\d):(\d\d):(\d\d)\.(\d+)\s")
_APA_AGG = re.compile(r'"Metrics window aggregation" metricKey="[^/"]*/([^/"]+)/kv_cache_usage_perc" stableAvg=([-\d.eE+]+|NaN)')
_APA_REC = re.compile(r'"Scaling recommendation computed" source="[^/"]*/([^"]+)".*?"DesiredReplicas":(\d+)')
_APA_CFG = re.compile(r'"Effective autoscaling config" podAutoscaler="[^/"]*/([^"]+)".*?stableWindow="([\d.]+)s" '
                      r'upTolerance=([\d.]+) downTolerance=([\d.]+)')
_APA_MARKS = ("Metrics window aggregation", "Scaling recommendation computed", "Effective autoscaling config")


def _klog_epoch(mo: str, day: str, hh: str, mm: str, ss: str, frac: str, near: float) -> float | None:
    """klog stamps carry no year (UTC): take the year that puts the stamp closest to ``near``."""
    ref = _dt.datetime.fromtimestamp(near, _dt.timezone.utc)
    best = None
    for y in (ref.year - 1, ref.year, ref.year + 1):
        try:
            e = _dt.datetime(y, int(mo), int(day), int(hh), int(mm), int(ss), tzinfo=_dt.timezone.utc).timestamp()
        except ValueError:
            continue
        e += float("0." + frac)
        if best is None or abs(e - near) < abs(best - near):
            best = e
    return best


def _seconds(v: Any) -> float | None:
    mt = re.fullmatch(r"\s*([\d.]+)\s*(ms|s|m)?\s*", str(v)) if v is not None else None
    if not mt:
        return None
    return float(mt.group(1)) * {"ms": 0.001, "s": 1.0, "m": 60.0, None: 1.0}[mt.group(2)]


def _pa_model(name: str, models: list[str], by_cr: dict[str, str]) -> str | None:
    if name in by_cr:
        return by_cr[name]
    return next((m for m in sorted(models, key=len, reverse=True) if name.startswith(m)), None)


def _apa_crs(raw: str | None, fname: str) -> tuple[dict[str, dict], dict[str, str]]:
    """apa_crs_{before,after}.yaml -> ({model: {target, window_s, up_tol, down_tol}}, {PA name: model})."""
    crs: dict[str, dict] = {}
    by_cr: dict[str, str] = {}
    p = os.path.join(raw, fname) if raw else None
    if not p or yaml is None or not os.path.exists(p):
        return crs, by_cr
    try:
        with open(p) as fh:
            doc = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return crs, by_cr
    for it in (doc.get("items") or [doc]) if isinstance(doc, dict) else []:
        if not isinstance(it, dict) or it.get("kind") != "PodAutoscaler":
            continue
        md, spec = it.get("metadata") or {}, it.get("spec") or {}
        name = md.get("name")
        m = (md.get("labels") or {}).get("model.aibrix.ai/name") or (spec.get("scaleTargetRef") or {}).get("name")
        if not name or not m:
            continue
        by_cr[name] = m
        ann = md.get("annotations") or {}
        src = next((x for x in spec.get("metricsSources") or [] if x.get("targetMetric") == "kv_cache_usage_perc"), {})
        crs[m] = {"target": _f(src.get("targetValue")), "up_tol": _f(ann.get("autoscaling.aibrix.ai/scale-up-tolerance")),
                  "down_tol": _f(ann.get("autoscaling.aibrix.ai/scale-down-tolerance")),
                  "window_s": _seconds(ann.get("apa.autoscaling.aibrix.ai/window"))}
    return crs, by_cr


def apa_config(raw: str | None, models: list[str], log_cfgs: dict[str, list], warn) -> tuple[dict, dict]:
    """Per model {target, window_s, up_tol, down_tol, from}; ``log_cfgs``: model -> [(t, {window_s, up_tol,
    down_tol})] in log order. Second value: PA name -> model, from the CRs."""
    before, by_cr = _apa_crs(raw, "apa_crs_before.yaml")
    after, by_after = _apa_crs(raw, "apa_crs_after.yaml")
    by_cr = {**by_after, **by_cr}
    out, missing = {}, defaultdict(list)
    for m in models:
        logged = log_cfgs.get(m) or []
        distinct = list(dict.fromkeys(tuple(c[k] for k in APA_KEYS) for _, c in logged))
        if len(distinct) > 1:
            warn(f"APA config of {m} changed during the run (window, up, down: "
                 + " -> ".join(", ".join(f"{x:g}" for x in d) for d in distinct) + "); the first one is drawn")
        lc = dict(zip(APA_KEYS, distinct[0])) if distinct else {}
        cr = before.get(m) or after.get(m) or {}
        c, froms = {}, []
        for k in APA_KEYS:
            if lc.get(k) is not None:
                c[k] = lc[k]
            elif cr.get(k) is not None:
                c[k] = cr[k]
            else:
                c[k] = None
                missing[m].append(k)
            for name, crd in (("before", before.get(m)), ("after", after.get(m))):
                if lc.get(k) is not None and crd and crd.get(k) is not None and abs(crd[k] - lc[k]) > 1e-9:
                    warn(f"APA {k} of {m}: CR ({name}) says {crd[k]:g}, the controller log says {lc[k]:g}; the log value is drawn")
        c["target"] = cr.get("target")
        if c["target"] is None:
            missing[m].append("target")
        froms.append("window/tolerances: " + ("controller log" if lc else "CR" if cr else "none"))
        froms.append("target: " + ("CR" if c["target"] is not None else "none"))
        c["from"] = "; ".join(froms)
        out[m] = c
    if missing:
        warn("APA config not recorded (" + "; ".join(f"{m}: {', '.join(ks)}" for m, ks in missing.items())
             + "): no trigger lines or zones for " + ", ".join(missing))
    return out, by_cr


def apa_zone(v: float | None, c: dict) -> str | None:
    """The zone of a window value, by APA's own ratio test (None when the config is incomplete)."""
    if v is None or not c.get("target") or c.get("up_tol") is None or c.get("down_tol") is None:
        return None
    r = v / c["target"]
    return "above up" if r > 1 + c["up_tol"] else "below down" if r < 1 - c["down_tol"] else "band"


def _apa_log(raw: str | None, clock: _Clock):
    """aibrix-controller-manager.log -> (evals {pa: [(t, value)]}, recs {pa: [(t, desired, value)]},
    cfgs {pa: [(t, {window_s, up_tol, down_tol})]}, bad); None when the log is absent. ``bad`` counts lines
    that carry one of the three markers but do not parse. Stamps are the controller-manager host's clock,
    shifted by its node offset."""
    p = os.path.join(raw, "aibrix-controller-manager.log") if raw else None
    if not p or not os.path.exists(p) or clock.t_load is None:
        return None
    evals, recs, cfgs = defaultdict(list), defaultdict(list), defaultdict(list)
    last: dict[str, float] = {}
    bad = 0
    with open(p, errors="replace") as fh:
        for line in fh:
            if not any(k in line for k in _APA_MARKS):
                continue
            mk = _KLOG.match(line)
            t = clock.rel(_klog_epoch(*mk.groups(), near=clock.t_load), "aibrix-controller-manager") if mk else None
            ma, mr, mc = _APA_AGG.search(line), _APA_REC.search(line), _APA_CFG.search(line)
            v = _f(ma.group(2)) if ma else None
            if t is None or not (ma or mr or mc) or (ma and v is None):
                bad += 1
                continue
            if ma:
                evals[ma.group(1)].append((t, v))
                last[ma.group(1)] = v
            elif mr:
                recs[mr.group(1)].append((t, int(mr.group(2)), last.get(mr.group(1))))
            else:
                cfgs[mc.group(1)].append((t, {"window_s": float(mc.group(2)), "up_tol": float(mc.group(3)),
                                              "down_tol": float(mc.group(4))}))
    return evals, recs, cfgs, bad


def build_apa_kv(raw: str | None, clock: _Clock, ts: dict, events: list[dict], warn) -> dict | None:
    """APA arms: the KV value APA decided on, as ts field 'apa_kv' (0..1, full precision) per model, plus
    its config. Logged (per model): every evaluation's stable-window average from the controller log, held
    until the next evaluation. Approx (no evaluation of that model in the log): trailing window_s mean of the
    1 s routable cross-pod kv_mean. Each APA decision gets the window value it was taken on and its zone."""
    models = list(ts["models"])
    if not models:
        return None
    log = _apa_log(raw, clock)
    evals, recs, lcfg, bad = log if log else ({}, {}, {}, 0)
    if bad:
        warn(f"aibrix-controller-manager.log: {bad} APA line(s) that do not parse (skipped)")
    by_cr = {**_apa_crs(raw, "apa_crs_after.yaml")[1], **_apa_crs(raw, "apa_crs_before.yaml")[1]}
    to_m: dict[str, str | None] = {pa: _pa_model(pa, models, by_cr) for pa in set(evals) | set(recs) | set(lcfg)}
    lost = sorted(pa for pa, m in to_m.items() if m is None)
    if lost:
        warn("APA autoscaler(s) in the controller log that map to no model of this arm (ignored): " + ", ".join(lost))
    ev_m, rec_m, cfg_m = defaultdict(list), defaultdict(list), defaultdict(list)
    for src_d, dst in ((evals, ev_m), (recs, rec_m), (lcfg, cfg_m)):
        for pa, xs in src_d.items():
            if to_m.get(pa):
                dst[to_m[pa]] += xs
    cfg, _ = apa_config(raw, models, {m: sorted(xs, key=lambda x: x[0]) for m, xs in cfg_m.items()}, warn)
    t0, n = ts["t0"], ts["n"]
    approx = [m for m in models if not ev_m.get(m)]
    if approx:
        warn("no APA evaluation logged (aibrix-controller-manager.log 'Metrics window aggregation') for "
             + ", ".join(approx) + ": its APA window average is approximated from the 1 s routable KV mean")
    out = {"models": {}}
    for m in models:
        f, c = ts["models"][m], cfg[m]
        series: list = [None] * n
        logged = m not in approx
        if logged:
            pts = sorted((t, v) for t, v in ev_m[m] if t0 <= t < t0 + n)
            for k, (t, v) in enumerate(pts):  # hold each evaluation until the next one
                a = int(math.floor(t)) - t0
                b = int(math.floor(pts[k + 1][0])) - t0 if k + 1 < len(pts) else n
                for i in range(a, max(a + 1, b)):
                    series[i] = v
        else:
            pts = []
            w = max(1, int(round(c["window_s"] or 20)))  # unknown window: 20 s only to draw the approximation
            kv = f.get("kv_mean") or [None] * n
            for i in range(n):
                xs = [x for x in kv[max(0, i - w + 1):i + 1] if x is not None]
                series[i] = sum(xs) / len(xs) if xs else None
        f["apa_kv"] = series
        out["models"][m] = {**c, "source": "logged" if logged else "approx",
                            "label": "APA window avg (" + ("logged" if logged else "approx.") + ")",
                            "evals": len(pts) if logged else None}
    flips: dict[str, list] = defaultdict(list)  # recommendations whose desired scale differs from the previous one
    for m, xs in rec_m.items():
        prev = None
        for t, d, v in sorted(xs, key=lambda x: x[0]):
            if d != prev:
                flips[m].append((t, d, v))
            prev = d
    n_dec = n_ok = n_none = 0
    for e in events:
        if e["kind"] not in ("decision_up", "decision_down") or e.get("model") not in out["models"]:
            continue
        m, c = e["model"], cfg[e["model"]]
        want = (e.get("detail") or {}).get("desired")
        if out["models"][m]["source"] == "logged":
            # the evaluation where the recommendation turned to the new desired scale: at most 10 s before the
            # status change was polled, at most 2 s after it (poll lag)
            cand = [x for x in flips.get(m, []) if e["t"] - 10.0 <= x[0] <= e["t"] + 2.0 and (want is None or x[1] == want)]
            v = cand[-1][2] if cand else None
        else:
            # approximate: the trailing mean at the latest 1 s sample at or before e.t - 2 s (the status poll lags
            # the evaluation by up to ~2 s); real evaluations are sparse, so this is not the value APA used
            i = int(math.floor(e["t"] - 2.0)) - t0
            v = ts["models"][m]["apa_kv"][i] if 0 <= i < n else None
        z = apa_zone(v, c) if v is not None else "no matching evaluation"
        n_none += v is None
        e["detail"] = {**(e.get("detail") or {}), "apa_window_value": v, "apa_zone": z}
        n_dec += 1
        n_ok += (e["kind"] == "decision_up" and z == "above up") or (e["kind"] == "decision_down" and z == "below down")
    if n_none:
        warn(f"{n_none} APA decision(s) without a matching evaluation in [t - 10 s, t + 2 s] (no window value)")
    out["decisions"] = {"n": n_dec, "at_crossing": n_ok, "unmatched": n_none}
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
        out.append({"name": name, "tp": int(r["tp_size"]) if r.get("tp_size") else None,
                    "slo": {k: slo.get(k) for k in ("ttft_p95_ms", "tpot_p95_ms") if slo.get(k) is not None},
                    "tau": tau or None, "max_awake": r.get("max_awake_replicas"),
                    "color": {"light": c, "dark": S.DARK_STEPS.get(c, c)}})
    return out


def _num_row(r: dict) -> dict:
    return {k: (_f(v) if k.startswith(("t_", "lat_")) or k in ("lo", "hi") else v) for k, v in r.items()}


def build(report_dir: str, trace: str | None = None, raw_root: str | None = None,
          subs: list[str] | None = None) -> dict:
    """The page's data object for one trace of a report directory."""
    found = discover(report_dir, subs)
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
    else:
        comp = [r for r in comp if r.get("trace") in (None, "", trace)]  # multi-trace reports: this trace only
        if subs is None:
            listed = {r.get("arm") for r in comp}
            stale = sorted(a["sub"] for a in arms_in if a["name"] not in listed)
            if stale:
                warnings.append(f"arm folder(s) not in comparison.csv (left from an earlier report?): {', '.join(stale)}")
    styles = S.ArmStyles([(a["name"], a["manifest"].get("label") or a["name"]) for a in arms_in])

    # registry: the first arm with one (all arms of a campaign should share models and topology)
    raws = {a["name"]: raw_dir(report_dir, a, raw_root) for a in arms_in}
    reg = {}
    for a in arms_in:
        reg = _registry(raws[a["name"]])
        if reg:
            break
    if not reg:
        warnings.append("live-registry.yaml not found next to any arm: TP size, SLO lines, tau lines "
                        "and GPU topology inferred or left out")
    shas = {a["name"]: a["manifest"].get("registry_sha256") for a in arms_in if a["manifest"].get("registry_sha256")}
    if len(set(shas.values())) > 1:
        warnings.append("arms ran with different registries (registry_sha256: " +
                        ", ".join(f"{k} {v[:8]}" for k, v in shas.items()) + "); SLO / tau / topology are taken from the first")
    models = [m.get("name") for m in (reg.get("models") or []) if isinstance(m, dict) and m.get("name")]
    node_names = sorted(n.get("name") for n in ((reg.get("cluster") or {}).get("nodes") or [])
                        if isinstance(n, dict) and n.get("name"))

    arms_out, all_gpu_ids, onsets, dps = [], set(), [], []
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
        src = man.get("decision_source")
        aw.extend(clock.skew_warnings(["controller"] + (["baseline-scaler"] if src == "baseline" else [])
                                      + (["aibrix-controller-manager"] if src == "apa" else [])))
        if raw is None:
            warn("raw arm directory not found: no SafeScale probes, refusal reasons or raw log lines")
        gpus = build_gpus(_csv(os.path.join(d, "gpu_intervals.csv")))
        if not gpus:
            warn("gpu_intervals.csv missing or empty: no GPU layout")
        all_gpu_ids.update(gpus)
        nn = sorted(set(node_names) | {g.rpartition("/")[0] for g in gpus})
        events = build_events(ev_rows or [], raw, clock, warn)
        apa_kv = build_apa_kv(raw, clock, ts, events, warn) if src == "apa" else None
        probes = []
        if src == "tre":  # probes belong to the arm whose controller acts; others only hold leftovers
            load_s = _f(man.get("load_s"))
            hi = (load_s if load_s is not None else ts["t0"] + ts["n"]) + PROBE_GRACE_S
            probes = build_probes(raw, clock, nn, (ts["t0"], hi), warn)
            if probes is None:
                probes = []
                if raw is not None:
                    warn("safescale.json missing: no SafeScale probe spans")
        summ = _json(os.path.join(d, "summary.json")) or {}
        if not summ:
            warn("summary.json missing: no validation status and decision points for this arm")
        on = _csv(os.path.join(d, "onsets.csv"))
        onsets += [{"arm": a["name"], **_num_row(r)} for r in on or []]
        dps += [{"arm": a["name"], **x} for x in summ.get("decision_points") or [] if isinstance(x, dict)]
        st = styles[a["name"]]
        light = st["color"]
        arms_out.append({
            "name": a["name"], "label": man.get("label") or a["name"], "sub": a["sub"],
            "family": S.family(a["name"], man.get("label") or ""), "decision_source": src,
            "target_source": TARGET_SOURCE.get(src),
            "color": {"light": light, "dark": S.DARK_STEPS.get(light, light)}, "dash": _dash(st["ls"]),
            "summary": {r["model"]: {k: (_f(r.get(k)) if k != "valid" else r.get(k)) for k in SUMMARY_KEYS if k in r}
                        for r in (comp or []) if r.get("arm") == a["name"]},
            "validation": summ.get("validation"),
            "warnings": list(summ.get("warnings") or []) + aw,
            "controller_env": (man.get("arm_meta") or {}).get("controller_env"),
            "tre_sha": man.get("tre_sha"), "load_s": man.get("load_s"),
            "ts": ts, "events": events, "probes": probes, "gpus": gpus, "apa_kv": apa_kv})
    meta = _models_meta(reg, models)
    for m in [m for m in meta if m["tp"] is None]:  # no tp_size in the registry: the widest binding
        widths = [len(parse_binding(iv[3])[2]) for a in arms_out for ivs in a["gpus"].values() for iv in ivs
                  if iv[2] == m["name"] and iv[3]]
        m["tp"] = max(widths or [1])
    nodes = topology(reg, sorted(all_gpu_ids), warnings.append)
    tps = [m["tp"] for m in meta]
    for a in arms_out:
        a["frag"] = fragmentation(a["ts"]["t0"], a["ts"]["n"], a["gpus"], nodes, tps)
    first = arms_in[0]["manifest"]
    return _finite({"schema": SCHEMA, "tool": "tre_eval.viewer",
                    "generated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "trace": trace, "seed": first.get("trace_seed"),
                    "report_dir": os.path.basename(os.path.abspath(report_dir)),
                    "models": meta, "nodes": nodes, "onsets": onsets, "decision_points": dps,
                    "warnings": warnings, "arms": arms_out})


# ---------------------------------------------------------------- html
def _asset(name: str) -> str:
    with open(os.path.join(ASSETS, name), encoding="utf-8") as fh:
        return fh.read()


def embed_json(data: Any) -> str:
    """JSON for an inline <script> block: <, > and & as \\u escapes, so no string in the data can close
    the element or open a comment / CDATA section (JSON.parse restores them)."""
    blob = json.dumps(_finite(data), separators=(",", ":"), allow_nan=False, default=str)
    return blob.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def render(data: dict) -> str:
    page = _asset("viewer.html")
    title = f"{data['trace']} · results viewer"
    for k, v in (("/*__UPLOT_CSS__*/", _asset("uPlot.min.css")), ("/*__VIEWER_CSS__*/", _asset("viewer.css")),
                 ("/*__UPLOT_JS__*/", _asset("uPlot.iife.min.js")), ("/*__CORE_JS__*/", _asset("viewer_core.js")),
                 ("/*__APP_JS__*/", _asset("viewer_app.js")), ("__TITLE__", html.escape(title))):
        page = page.replace(k, v)
    return page.replace("__DATA_JSON__", embed_json(data))


def write(report_dir: str, out: str | None = None, trace: str | None = None, raw_root: str | None = None,
          subs: list[str] | None = None) -> str:
    data = build(report_dir, trace=trace, raw_root=raw_root, subs=subs)
    out = out or os.path.join(report_dir, "viewer.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(render(data))
    return out


def write_for_report(report_dir: str, subs: list[str] | None = None) -> list[str]:
    """report.py hook: viewer.html (one trace) or viewer-<trace>.html (several) next to index.html,
    built from the arm folders ``subs`` the report just wrote."""
    found = discover(report_dir, subs)
    if len(found) == 1:
        return [write(report_dir, subs=subs)]
    return [write(report_dir, os.path.join(report_dir, f"viewer-{re.sub(r'[^A-Za-z0-9_.-]', '_', t)}.html"),
                  trace=t, subs=subs) for t in sorted(found)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report_dir", help="tre_eval.report --out directory of one trace")
    ap.add_argument("-o", "--out", help="output HTML (default <report_dir>/viewer.html)")
    ap.add_argument("--trace", help="trace name when the report holds several traces")
    ap.add_argument("--raw-root", help="directory holding the raw arm directories (looked up first; then the "
                                       "manifest's dir, then the report's parent)")
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

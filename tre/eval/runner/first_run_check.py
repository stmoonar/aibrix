#!/usr/bin/env python3
"""first_run_check.py <arm dir> [--json OUT] [--quiet]: post-run check of one arm (read-only).

Run it after run 1 of a (re)started batch and then periodically (AGENTS.md: first run checked at
once, then every 50 min). run_arm_pilot.sh also runs it at the end of every arm and writes
``run_check.json`` + ``run_check.txt``; campaign.py folds ``validity.suspect`` into the arm verdict.

Sections (text: one screen; JSON: the same, machine readable):

1. completeness - collector files present / size / lines; logs whose first line starts too late
   after the arm start (container log rotated or truncated);
2. signal loss - controller ticks per loop, stale snapshots, ``metrics_window_stale`` by reason,
   ``samples_incomplete`` events, ticks with ``sample_quality_hold`` /
   ``signal_unavailable_reason`` per model, gateway cadence (declared or not, pod-samples with
   fewer instant / hist docs than declared, write lag, scrape age), sampler error rows, baseline
   event validity and ``inputs.gaps``;
3. validity - the campaign verdict when present, tick coverage (controller fast loop and
   baseline scaler: ticks during the load / (load duration / tick) >= ``--coverage-min``),
   frozen-tick share of a baseline that reports frozen ticks (``frozen`` true at top level or in
   ``inputs``, or a reason containing ``frozen``; not reported -> null) <= ``--frozen-max``;
4. latency - E2E / TTFT / TPOT p50 / p90 / p95 / p99 (ms) per model and ALL, successful requests;
5. replicas - mean awake / routable replicas per model over the load window (SM layout, 1 s);
6. scale actions - by decision source, direction and reason; SM operations by kind / actor / status;
7. abnormal - timeouts (e2e >= 149 s or stream cut), HTTP 5xx, client failures, stale snapshots,
   EMFILE, SM 5xx, controller ERROR lines, truncated logs, missing collectors.

Times: the reference clock (this host); the baseline decisions' ``ts_ms`` is the Redis clock.
"""
from __future__ import annotations

import argparse
import calendar
import glob
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

E2E_CUT_S = 149.0
LOG_START_SLACK_S = 90.0
FILES = ("client/performance_metrics.json", "layout.jsonl", "pod_metrics_1s.jsonl", "gpu_map.jsonl", "gpu_truth.jsonl",
         "apa_status.jsonl", "resource_usage.jsonl", "signal_log.jsonl", "decision_hist.jsonl", "gateway_cadence.jsonl",
         "sm_operations.jsonl", "controller.log", "sm.ts.log", "gateway-plugins.log", "safescale.json",
         "clock_offsets.json", "components.json", "live-registry.yaml", "score.json")
BASELINE_FILES = ("baseline-scaler.log", "baseline/run_validity.json", "baseline/bl_decisions.jsonl")


# ------------------------------------------------------------------ small readers
def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _read_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _jsonl(p):
    out = []
    try:
        with open(p, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def _epoch_file(d: Path, name: str):
    try:
        return float((d / name).read_text().strip())
    except (OSError, ValueError):
        return None


def pct(xs, p):
    v = sorted(x for x in xs if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * p / 100
    f = math.floor(k)
    c = min(f + 1, len(v) - 1)
    return round(v[f] + (v[c] - v[f]) * (k - f), 1)


def _lines(p: Path) -> int:
    try:
        with open(p, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


# ------------------------------------------------------------------ 1. completeness
_CTRL_TIME = re.compile(r'"time":\s*"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)')
_KUBELET = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)")
_KLOG = re.compile(r"^[IWEF](\d\d)(\d\d) (\d\d:\d\d:\d\d)")


def first_log_epoch(p: Path, kind: str, year: int):
    """Epoch s (UTC) of the first timestamped line: controller JSON ``time``, kubelet RFC3339, or klog."""
    try:
        with open(p, errors="replace") as f:
            for i, line in enumerate(f):
                if i > 5000:
                    return None
                if kind == "controller":
                    mt = _CTRL_TIME.search(line)
                    if mt:
                        return calendar.timegm(time.strptime(mt.group(1), "%Y-%m-%d %H:%M:%S"))
                elif kind == "kubelet":
                    mt = _KUBELET.match(line)
                    if mt:
                        return calendar.timegm(time.strptime(mt.group(1), "%Y-%m-%dT%H:%M:%S"))
                else:
                    mt = _KLOG.match(line)
                    if mt:
                        return calendar.timegm(time.strptime(f"{year}-{mt.group(1)}-{mt.group(2)} {mt.group(3)}",
                                                             "%Y-%m-%d %H:%M:%S"))
    except OSError:
        return None
    return None


def completeness(d: Path, baseline: bool, t_restart, t_first_send) -> dict:
    files = {}
    for f in FILES + (BASELINE_FILES if baseline else ()):
        p = d / f
        files[f] = {"bytes": p.stat().st_size if p.is_file() else None,
                    "lines": _lines(p) if p.is_file() and f.endswith((".jsonl", ".log")) else None}
    if baseline:
        dec = sorted(glob.glob(str(d / "baseline" / "*" / "decisions-*.jsonl")))
        files["baseline/<pod>/decisions-*.jsonl"] = {"bytes": sum(os.path.getsize(x) for x in dec) or None,
                                                     "lines": sum(_lines(Path(x)) for x in dec)}
    missing = [f for f, v in files.items() if not v["bytes"]]
    truncated = {}
    year = time.gmtime(t_restart or time.time()).tm_year
    for f, kind, ref in (("controller.log", "controller", t_restart), ("sm.ts.log", "kubelet", t_restart),
                         ("gateway-plugins.log", "klog", t_first_send)):
        p = d / f
        if ref is None or not p.is_file() or not p.stat().st_size:
            continue
        t = first_log_epoch(p, kind, year)
        if t is None:
            truncated[f] = "no timestamped line"
        elif t > ref + LOG_START_SLACK_S:
            truncated[f] = f"first line {t - ref:+.0f} s after the arm reference (rotated / truncated?)"
    return {"files": files, "missing": missing, "log_truncated": truncated}


# ------------------------------------------------------------------ 2. signal loss
_STALE = re.compile(r"metrics_window_stale: window_end=\d+ reason=(\S+)")


def controller_scan(p: Path, t0, t1) -> dict:
    """trs_calc_result ticks (all and during [t0, t1] by wall clock now_ms / window ts), stale,
    hold reasons per model, events, actions by reason, ERROR lines, stale-window warnings."""
    ticks, ticks_load = Counter(), Counter()
    stale_ticks, errors = 0, 0
    stale_reasons, events = Counter(), Counter()
    holds = defaultdict(Counter)
    actions = Counter()
    first_err = []
    try:
        fh = open(p, errors="replace")
    except OSError:
        return {}
    with fh:
        for line in fh:
            mt = _STALE.search(line)
            if mt:
                stale_reasons[mt.group(1).rstrip(";,")] += 1
            if '"level": "ERROR"' in line or '"level": "CRITICAL"' in line or line.startswith("Traceback"):
                errors += 1
                if len(first_err) < 3:
                    first_err.append(line.strip()[:240])
            if "trs_calc_result" not in line or not line.startswith("{"):
                continue
            try:
                m = json.loads(json.loads(line).get("message", ""))
            except (ValueError, TypeError, AttributeError):
                continue
            if not isinstance(m, dict) or m.get("event") != "trs_calc_result":
                continue
            loop = m.get("loop") or "?"
            ticks[loop] += 1
            t = _num(m.get("now_ms")) or _num(m.get("ts_ms"))
            t = None if t is None else t / 1000.0
            if t is not None and t0 is not None and t1 is not None and t0 <= t <= t1:
                ticks_load[loop] += 1
            stale_ticks += m.get("stale") == "true"
            try:
                evs = json.loads(m.get("events") or "[]")
                acts = json.loads(m.get("actions") or "[]")
                states = json.loads(m.get("model_states") or "{}")
            except ValueError:
                continue
            for e in evs:
                events[str(e).split(":", 1)[0]] += 1
            for a in acts:
                if isinstance(a, dict):
                    actions[(loop, a.get("kind"), a.get("reason"), "+" if (a.get("delta") or 0) > 0 else
                             ("-" if (a.get("delta") or 0) < 0 else ""))] += 1
            for model, st in (states or {}).items():
                if not isinstance(st, dict):
                    continue
                for key in ("sample_quality_hold", "signal_unavailable_reason"):
                    if st.get(key):
                        holds[model][f"{key}={st[key]}"] += 1
    return {"ticks": dict(ticks), "ticks_load": dict(ticks_load), "stale_ticks": stale_ticks,
            "metrics_window_stale": dict(stale_reasons), "error_lines": errors, "first_errors": first_err,
            "events": dict(events.most_common(25)), "holds": {m: dict(c) for m, c in sorted(holds.items())},
            "actions": {" ".join(str(x) for x in k if x): v for k, v in actions.most_common()}}


def cadence_scan(p: Path) -> dict:
    rows = [r for r in _jsonl(p) if "pods" in r]
    err = sum(1 for r in _jsonl(p) if "error" in r)
    if not rows and not err:
        return {}
    declared = sum(1 for r in rows if r.get("declared"))
    short_i = short_h = active = 0
    lags, ages = [], []
    # The counts are per window ``(end - period, end]``; ``end`` is the sampler's anchor
    # (gateway ``last_hist_ms`` since 2026-10-26: complete rounds only). Rows anchored on
    # ``redis_time`` race the 500 ms grid and may read one doc short per pod.
    # A frozen ``last_hist_ms`` (sampler: more than 2 evidence periods behind written_ms).
    sources = Counter(r.get("window_end_source") or "none" for r in rows)
    anchor_lags = [r["anchor_lag_ms"] for r in rows if isinstance(r.get("anchor_lag_ms"), (int, float))]
    for r in rows:
        exp = r.get("expected") or {}
        end = r.get("window_end_ms") or 0
        for pod in (r.get("pods") or {}).values():
            last = pod.get("inst_last_ms")
            if last is None or end - last > 60_000:   # pod not written for a minute: gone / not Ready
                continue
            active += 1
            short_i += (pod.get("inst_n") or 0) < (exp.get("inst") or 0)
            short_h += (pod.get("hist_n") or 0) < (exp.get("hist") or 0)
            if pod.get("write_lag_ms") is not None:
                lags.append(pod["write_lag_ms"])
            if pod.get("scrape_age_ms") is not None:
                ages.append(pod["scrape_age_ms"])
    last_cad = next((r.get("cadence") for r in reversed(rows) if r.get("cadence")), None)
    return {"samples": len(rows), "error_rows": err, "declared_samples": declared, "cadence_last": last_cad,
            "window_end_sources": dict(sources), "hist_frozen_samples": sources.get("written_ms_hist_frozen", 0),
            "anchor_lag_ms_max": max(anchor_lags) if anchor_lags else None,
            "pod_samples_active": active, "pod_samples_inst_short": short_i, "pod_samples_hist_short": short_h,
            "write_lag_ms_p95": pct(lags, 95), "write_lag_ms_max": max(lags) if lags else None,
            "scrape_age_ms_p95": pct(ages, 95), "scrape_age_ms_max": max(ages) if ages else None}


def sampler_errors(d: Path) -> dict:
    out = {}
    for f in ("layout.jsonl", "pod_metrics_1s.jsonl", "gpu_truth.jsonl", "apa_status.jsonl"):
        n = sum(1 for r in _jsonl(d / f) if "error" in r)
        if n:
            out[f] = n
    pod_err = 0
    for r in _jsonl(d / "pod_metrics_1s.jsonl"):
        pod_err += sum(1 for v in (r.get("pods") or {}).values() if isinstance(v, dict) and "error" in v)
    if pod_err:
        out["pod_metrics_1s.jsonl pod scrape errors"] = pod_err
    return out


# ------------------------------------------------------------------ baseline decisions
def baseline_decisions(d: Path) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(d / "baseline" / "*" / "decisions-*.jsonl"))):
        rows.extend(r for r in _jsonl(Path(f)) if "ts_ms" in r)
    rows.sort(key=lambda r: (r.get("ts_ms", 0), r.get("model", "")))
    return rows


def _frozen(r: dict):
    inputs = r.get("inputs") if isinstance(r.get("inputs"), dict) else {}
    for src in (r, inputs):
        if "frozen" in src:
            return bool(src["frozen"])
    if "frozen" in str(r.get("reason") or ""):
        return True
    return None


def baseline_scan(rows: list[dict], t0, t1, tick_s: float) -> dict:
    per = defaultdict(list)
    for r in rows:
        t = r["ts_ms"] / 1000.0
        if t0 is not None and t1 is not None and t0 <= t <= t1:
            per[r.get("model")].append(r)
    dur = (t1 - t0) if (t0 is not None and t1 is not None) else None
    out = {"models": {}}
    gaps = Counter()
    for m, rs in sorted(per.items()):
        ticks = sorted({r.get("tick") for r in rs if r.get("tick") is not None})
        times = sorted(r["ts_ms"] / 1000.0 for r in rs)
        dts = [b - a for a, b in zip(times, times[1:]) if b > a]
        fz = [_frozen(r) for r in rs]
        rep = [x for x in fz if x is not None]
        for r in rs:
            for g in ((r.get("inputs") or {}).get("gaps") or []) if isinstance(r.get("inputs"), dict) else []:
                gaps[str(g)] += 1
        out["models"][m] = {
            "ticks": len(ticks) or len(rs),
            "coverage": round((len(ticks) or len(rs)) / (dur / tick_s), 3) if dur else None,
            "median_dt_s": round(sorted(dts)[len(dts) // 2], 2) if dts else None,
            "max_gap_s": round(max(dts), 1) if dts else None,
            "frozen_share": round(sum(rep) / len(rep), 4) if rep else None,
            "reasons": dict(Counter(str(r.get("reason")) for r in rs).most_common(8)),
            "actions": dict(Counter(str(r.get("action")) for r in rs).most_common(8)),
            # queued_up / queued_down (scaler 2026-10-28) = a decision waiting behind the plan in flight
            "scale_by_reason": dict(Counter(f"{r.get('action')} {r.get('reason')}" for r in rs
                                            if r.get("action") in ("up", "down", "queued_up", "queued_down"))
                                    .most_common(10)),
        }
    out["gaps"] = dict(gaps.most_common(10))
    return out


# ------------------------------------------------------------------ 4. latency
def latency(reqs: list[dict]) -> dict:
    by = defaultdict(list)
    for r in reqs:
        by[r.get("model_name") or "?"].append(r)
        by["ALL"].append(r)
    out = {}
    for m, rs in sorted(by.items(), key=lambda x: (x[0] == "ALL", x[0])):
        ok = [r for r in rs if r.get("success")]
        e2e = [r["e2e_latency"] * 1000 for r in ok if _num(r.get("e2e_latency")) is not None]
        ttft = [r["ttft"] * 1000 for r in ok if _num(r.get("ttft")) is not None]
        tpot = []
        for r in ok:
            e, t, n = _num(r.get("e2e_latency")), _num(r.get("ttft")), r.get("output_tokens") or 0
            if e is not None and t is not None and n > 1:
                tpot.append((e - t) * 1000 / (n - 1))
        status = Counter(str(r.get("http_status")) for r in rs if r.get("http_status") not in (200, None))
        cut = sum(1 for r in rs if (_num(r.get("e2e_latency")) or 0) >= E2E_CUT_S or r.get("stream_interrupted")
                  or "RemoteProtocolError" in str(r.get("error_message") or r.get("stream_error") or ""))
        out[m] = {"n": len(rs), "fail": len(rs) - len(ok), "timeouts_or_cuts": cut,
                  "http_5xx": sum(v for k, v in status.items() if k.startswith("5")), "http_other": dict(status),
                  **{f"{name}_p{p}": pct(xs, p) for name, xs in (("e2e", e2e), ("ttft", ttft), ("tpot", tpot))
                     for p in (50, 90, 95, 99)}}
    return out


# ------------------------------------------------------------------ 5. replicas
def replicas(d: Path, t0, t1) -> dict:
    rows = [r for r in _jsonl(d / "layout.jsonl") if isinstance(r.get("models"), dict)]
    rows.sort(key=lambda r: r.get("ts", 0))
    if not rows or t0 is None or t1 is None or t1 <= t0:
        return {}
    acc = defaultdict(lambda: [0.0, 0.0, 0, 0])   # awake-s, routable-s, max awake, n samples
    for a, b in zip(rows, rows[1:] + [None]):
        ta = float(a["ts"])
        tb = float(b["ts"]) if b else ta + 1.0
        lo, hi = max(ta, t0), min(tb, t1)
        for m, v in a["models"].items():
            aw = len(v.get("awake") or [])
            ro = len([x for x in (v.get("awake") or []) if x not in set(v.get("hidden") or [])])
            s = acc[m]
            if hi > lo:
                s[0] += aw * (hi - lo)
                s[1] += ro * (hi - lo)
            if t0 <= ta <= t1:
                s[2] = max(s[2], aw)
                s[3] += 1
    dur = t1 - t0
    return {m: {"mean_awake": round(s[0] / dur, 3), "mean_routable": round(s[1] / dur, 3), "max_awake": s[2],
                "samples": s[3], "coverage": round(s[3] / dur, 3)} for m, s in sorted(acc.items())}


# ------------------------------------------------------------------ 6. actions
def apa_actions(d: Path, t0, t1) -> dict:
    prev, out = {}, Counter()
    for r in _jsonl(d / "apa_status.jsonl"):
        for name, st in (r.get("pa") or {}).items():
            v = (st or {}).get("desiredScale")
            if v is None:
                continue
            if name in prev and v != prev[name] and (t0 is None or t0 <= r.get("ts", 0) <= (t1 or 1e18)):
                out[f"{name} {'up' if v > prev[name] else 'down'}"] += 1
            prev[name] = v
    return dict(out)


def sm_ops(d: Path, t0_iso_s) -> dict:
    out = Counter()
    for r in _jsonl(d / "sm_operations.jsonl"):
        req = r.get("request") or {}
        out[f"{r.get('kind')} actor={req.get('actor') or '-'} status={r.get('status')}"] += 1
    return dict(out.most_common(30))


# ------------------------------------------------------------------ the check
def expected_rescue_interval_s(rescue_s: float = 5.0, rebalance_s: float = 10.0, snapshot_s: float = 1.0) -> float:
    """Mean spacing of the controller's rescue decisions (single decision loop, design
    20261028-single-loop): each period rounds UP to the snapshot grid (5 s with 2 s
    snapshots: 6 s; with the 1 s snapshots of the overlay since 2026-10-28: 5 s) and every
    rebalance call (10 s) also runs the rescue section, so per rebalance interval there
    are ceil(rebalance / rescue) rescue calls. Defaults: the overlay's
    (TRE_RESCUE_INTERVAL_SECONDS, TRE_FAIRNESS_INTERVAL_SECONDS, TRE_EVIDENCE_PERIOD_MS);
    a run on another period sets RUN_CHECK_CONTROLLER_TICK_S."""
    def grid(x: float) -> float:
        return math.ceil(x / snapshot_s - 1e-9) * snapshot_s
    rescue, rebalance = grid(rescue_s), grid(rebalance_s)
    return rebalance / math.ceil(rebalance / rescue - 1e-9)


def check(d: Path, controller_tick_s: float | None = None, baseline_tick_s: float = 5.0, coverage_min: float = 0.9,
          frozen_max: float = 0.05) -> dict:
    controller_tick_s = expected_rescue_interval_s() if controller_tick_s is None else controller_tick_s
    meta = _read_json(d / "arm_meta.json") or {}
    arm = meta.get("arm") or d.name
    baseline = arm in ("chiron", "tokenscale", "preserve") or (d / "baseline").is_dir()
    t_restart = _epoch_file(d, "controller_restart_epoch")
    t_load, t_end = _epoch_file(d, "load_start_epoch"), _epoch_file(d, "load_end_epoch")
    reqs = _read_json(d / "client" / "performance_metrics.json")
    if not isinstance(reqs, list):
        reqs = _jsonl(d / "client" / "performance_metrics.json")
    starts = [r["start_time"] for r in reqs if _num(r.get("start_time"))]
    t_first = min(starts) if starts else t_load
    if t_end is None and starts:
        t_end = max(starts)
    res: dict = {"arm_dir": str(d), "arm": arm, "label": (d / "arm_label").read_text().strip() if (d / "arm_label").is_file() else None,
                 "baseline": baseline, "load_s": round(t_end - t_load, 1) if t_load and t_end else None}
    comp = completeness(d, baseline, t_restart, t_first)
    res["completeness"] = comp
    ctrl = controller_scan(d / "controller.log", t_load, t_end)
    cad = cadence_scan(d / "gateway_cadence.jsonl")
    rv = _read_json(d / "baseline" / "run_validity.json") if baseline else None
    bl = baseline_scan(baseline_decisions(d), t_load, t_end, baseline_tick_s) if baseline else None
    res["signal_loss"] = {"controller": ctrl, "gateway_cadence": cad, "sampler_errors": sampler_errors(d),
                          "baseline_events": None if rv is None else {k: rv.get(k) for k in (
                              "events_valid", "invalid_because", "gw_bl_dropped_delta") if k in rv},
                          "baseline_gaps": (bl or {}).get("gaps")}
    # ---- validity
    sus, info = [], []
    load_s = (t_end - t_load) if (t_load and t_end) else None
    cov = {}
    if load_s and ctrl.get("ticks_load") is not None:
        n_fast = ctrl["ticks_load"].get("rescue", 0)
        cov["controller_fast"] = round(n_fast / (load_s / controller_tick_s), 3)
        if cov["controller_fast"] < coverage_min:
            sus.append(f"controller fast-loop ticks {n_fast} = {cov['controller_fast']:.2f} x load/{controller_tick_s:g}s "
                       f"< {coverage_min}")
    frozen = {}
    if bl:
        for m, v in bl["models"].items():
            cov[f"baseline:{m}"] = v["coverage"]
            if v["coverage"] is not None and v["coverage"] < coverage_min:
                sus.append(f"baseline {m}: {v['ticks']} ticks = {v['coverage']:.2f} x load/{baseline_tick_s:g}s < {coverage_min}")
            if v["median_dt_s"] and abs(v["median_dt_s"] - baseline_tick_s) > 0.25 * baseline_tick_s:
                sus.append(f"baseline {m}: median tick spacing {v['median_dt_s']} s, expected {baseline_tick_s:g} s")
            if v["frozen_share"] is not None:
                frozen[m] = v["frozen_share"]
                if v["frozen_share"] > frozen_max:
                    sus.append(f"baseline {m}: frozen-tick share {v['frozen_share']:.3f} > {frozen_max}")
        if not bl["models"]:
            sus.append("baseline arm without decisions in the load window")
    for f, why in comp["log_truncated"].items():
        sus.append(f"{f}: {why}")
    loss = {"stale_ticks": ctrl.get("stale_ticks"), "metrics_window_stale": sum((ctrl.get("metrics_window_stale") or {}).values()),
            "samples_incomplete_events": (ctrl.get("events") or {}).get("samples_incomplete", 0),
            "sample_quality_hold_ticks": sum(v for h in (ctrl.get("holds") or {}).values() for k, v in h.items()
                                             if k.startswith("sample_quality_hold")),
            "cadence_inst_short": cad.get("pod_samples_inst_short"), "cadence_hist_short": cad.get("pod_samples_hist_short")}
    if any(loss.values()):
        info.append("signal loss: " + ", ".join(f"{k}={v}" for k, v in loss.items() if v))
    camp = _read_json(d / "CAMPAIGN_DONE.json") or _read_json(d / "campaign_manifest.json") or {}
    res["validity"] = {"campaign_verdict": (camp.get("validity") or {}).get("verdict"),
                       "campaign_reasons": (camp.get("validity") or {}).get("invalid", []) + (camp.get("validity") or {}).get("suspect", []),
                       "score_valid": (_read_json(d / "score.json") or {}).get("valid"),
                       "tick_coverage": cov or None, "frozen_tick_share": frozen or None,
                       "signal_loss": loss, "log_truncated": comp["log_truncated"] or None,
                       "suspect": sus, "info": info,
                       "thresholds": {"controller_tick_s": controller_tick_s, "baseline_tick_s": baseline_tick_s,
                                      "coverage_min": coverage_min, "frozen_max": frozen_max}}
    res["latency_ms"] = latency(reqs)
    res["replicas"] = replicas(d, t_load, t_end)
    acts = {"controller": ctrl.get("actions"), "sm_operations": sm_ops(d, t_restart)}
    if bl:
        acts["baseline_scale"] = {m: v["scale_by_reason"] for m, v in bl["models"].items()}
        acts["baseline"] = {m: v["actions"] for m, v in bl["models"].items()}
        acts["baseline_reasons"] = {m: v["reasons"] for m, v in bl["models"].items()}
    if arm == "apa":
        acts["apa"] = apa_actions(d, t_load, t_end)
    res["actions"] = acts
    # ---- abnormal
    ab = []
    a = res["latency_ms"].get("ALL") or {}
    if not reqs:
        ab.append("no client records")
    if a.get("timeouts_or_cuts"):
        ab.append(f"{a['timeouts_or_cuts']} request(s) timed out / cut (e2e >= {E2E_CUT_S:g} s or stream error)")
    if a.get("http_5xx"):
        ab.append(f"{a['http_5xx']} HTTP 5xx at the client")
    if a.get("fail"):
        ab.append(f"{a['fail']} failed request(s)")
    if ctrl.get("stale_ticks"):
        ab.append(f"{ctrl['stale_ticks']} controller tick(s) on a stale snapshot")
    if ctrl.get("metrics_window_stale"):
        ab.append(f"metrics_window_stale {ctrl['metrics_window_stale']}")
    if ctrl.get("error_lines"):
        ab.append(f"{ctrl['error_lines']} controller ERROR line(s): {ctrl['first_errors'][:1]}")
    emf = (d / "EMFILE_PODS").read_text().split() if (d / "EMFILE_PODS").is_file() else []
    if emf:
        ab.append(f"EMFILE in {len(emf)} sidecar log(s)")
    sm5 = 0
    try:
        with open(d / "sm.ts.log", errors="replace") as f:
            sm5 = sum(1 for line in f if re.search(r'HTTP/1\.[01]" 5\d\d\b', line))
    except OSError:
        pass
    if sm5:
        ab.append(f"SM answered {sm5} request(s) with 5xx")
    if rv is not None and rv.get("events_valid") is False:
        ab.append(f"baseline events invalid: {rv.get('invalid_because')}")
    if comp["missing"]:
        ab.append(f"missing / empty: {comp['missing']}")
    if cad and cad.get("hist_frozen_samples"):
        sus.append(f"gateway last_hist_ms frozen in {cad['hist_frozen_samples']} cadence sample(s) "
                   f"(written_ms - last_hist_ms max {cad.get('anchor_lag_ms_max')} ms): histogram rounds stalled")
    if cad and cad.get("samples") and not cad.get("declared_samples"):
        info.append("gateway declares no cadence (image before 2026-10-25): doc counts judged on the 10 s grid")
    res["abnormal"] = ab + sus
    return res


# ------------------------------------------------------------------ text
def render(r: dict) -> str:
    L = []
    w = L.append
    v = r["validity"]
    w(f"== {r['arm_dir']}  arm={r['arm']} label={r['label']} load={r['load_s']} s")
    w(f"verdict: campaign={v['campaign_verdict']} score_valid={v['score_valid']} "
      f"run-check suspects={len(v['suspect'])}")
    c = r["completeness"]
    w(f"[1] completeness: {len(c['files']) - len(c['missing'])}/{len(c['files'])} files; missing={c['missing'] or '-'}; "
      f"truncated logs={c['log_truncated'] or '-'}")
    s = r["signal_loss"]
    ct = s["controller"] or {}
    w(f"[2] controller ticks {ct.get('ticks')} (load window {ct.get('ticks_load')}), stale ticks {ct.get('stale_ticks')}, "
      f"metrics_window_stale {ct.get('metrics_window_stale') or '-'}, ERROR lines {ct.get('error_lines')}")
    for m, h in (ct.get("holds") or {}).items():
        w(f"    holds {m}: {h}")
    gc = s["gateway_cadence"] or {}
    if gc:
        w(f"    gateway cadence: {gc.get('samples')} samples, declared {gc.get('declared_samples')}, last {gc.get('cadence_last')}; "
          f"pod-samples {gc.get('pod_samples_active')} (inst short {gc.get('pod_samples_inst_short')}, hist short "
          f"{gc.get('pod_samples_hist_short')}); write lag p95/max {gc.get('write_lag_ms_p95')}/{gc.get('write_lag_ms_max')} ms; "
          f"scrape age p95/max {gc.get('scrape_age_ms_p95')}/{gc.get('scrape_age_ms_max')} ms")
    if s["sampler_errors"]:
        w(f"    sampler errors: {s['sampler_errors']}")
    if s["baseline_events"] is not None:
        w(f"    baseline events: {s['baseline_events']}; gaps {s['baseline_gaps']}")
    w(f"[3] tick coverage {v['tick_coverage']}; frozen-tick share {v['frozen_tick_share']}; signal loss "
      f"{ {k: x for k, x in v['signal_loss'].items() if x} or '-'}")
    w("[4] latency ms (ok requests)     n   fail  cut   E2E p50/p90/p95/p99            TTFT p50/p90/p95/p99         TPOT p50/p90/p95/p99")
    for m, x in r["latency_ms"].items():
        q = lambda k: "/".join(str(x.get(f"{k}_p{p}")) for p in (50, 90, 95, 99))  # noqa: E731
        w(f"    {m:<26} {x['n']:>6} {x['fail']:>5} {x['timeouts_or_cuts']:>4}   {q('e2e'):<30} {q('ttft'):<28} {q('tpot')}")
    w("[5] replicas (load window): " + "; ".join(f"{m} awake {x['mean_awake']} routable {x['mean_routable']} max {x['max_awake']}"
                                               for m, x in r["replicas"].items()))
    a = r["actions"]
    w(f"[6] controller actions: {a.get('controller') or '-'}")
    for k in ("baseline_scale", "baseline", "baseline_reasons", "apa"):
        if a.get(k):
            w(f"    {k}: {a[k]}")
    w(f"    SM operations: {a.get('sm_operations') or '-'}")
    w("[7] abnormal: " + ("none" if not r["abnormal"] else ""))
    for x in r["abnormal"]:
        w(f"    - {x}")
    for x in v["info"]:
        w(f"    (info) {x}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm_dir", nargs="+")
    ap.add_argument("--json", help="write the result here (one arm dir only)")
    ap.add_argument("--text", help="also write the text here (one arm dir only)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--controller-tick-s", type=float,
                    default=float(os.environ.get("RUN_CHECK_CONTROLLER_TICK_S") or expected_rescue_interval_s()))
    ap.add_argument("--baseline-tick-s", type=float, default=float(os.environ.get("RUN_CHECK_BASELINE_TICK_S") or 5))
    ap.add_argument("--coverage-min", type=float, default=float(os.environ.get("RUN_CHECK_COVERAGE_MIN") or 0.9))
    ap.add_argument("--frozen-max", type=float, default=float(os.environ.get("RUN_CHECK_FROZEN_MAX") or 0.05))
    a = ap.parse_args(argv)
    for p in a.arm_dir:
        r = check(Path(p), a.controller_tick_s, a.baseline_tick_s, a.coverage_min, a.frozen_max)
        txt = render(r)
        if a.json:
            with open(a.json, "w") as f:
                json.dump(r, f, indent=1, default=str)
        if a.text:
            with open(a.text, "w") as f:
                f.write(txt + "\n")
        if not a.quiet:
            print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Metric definitions (the spec's "derived metrics" section is the normative text).

Request scoring reproduces ``pilot-e1-20261005/tools/score_pilot.py`` exactly (same trim,
same thresholds, same percentile interpolation, same rounding) so the numbers can be
checked against every existing ``score.json``. Everything else here is additive.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable

import numpy as np

from .load import Arm, parse_binding

E2E_CENSOR_MS = 149000.0  # streams cut by the 150 s route timeout are recorded as success by the client
ALL = "ALL"


# ---------------------------------------------------------------- percentiles (score_pilot's)
def pct(xs: Iterable[float | None], p: float) -> float | None:
    """Linear-interpolated percentile, None ignored, rounded to 3 decimals (score_pilot.pct)."""
    v = sorted(x for x in xs if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * p / 100
    f = math.floor(k)
    c = min(f + 1, len(v) - 1)
    return round(v[f] + (v[c] - v[f]) * (k - f), 3)


# ---------------------------------------------------------------- SLO thresholds
def ttft_threshold_ms(slo: dict, input_tokens: float) -> float:
    """TTFT SLO of one request: ``max(floor, k * (c + b * L))`` in slowdown mode, else ttft_p95_ms."""
    if slo.get("ttft_slo_mode") == "slowdown":
        return max(slo.get("ttft_floor_ms", 500.0),
                   slo.get("ttft_slowdown_k", 5.0) * (slo["ttft_idle_c_ms"] + slo["ttft_idle_b_ms_per_token"] * input_tokens))
    return slo["ttft_p95_ms"]


def score_request(r: dict, slo: dict) -> dict:
    """Per-request scored record (times in ms). V_req = fail or TTFT>thr or TPOT>tpot_p95 or e2e>=149 s."""
    L = r.get("input_tokens") or 0
    ok = bool(r.get("success"))
    ttft = r["ttft"] * 1000 if r.get("ttft") is not None else None
    e2e = r["e2e_latency"] * 1000 if r.get("e2e_latency") is not None else None
    out = r.get("output_tokens") or 0
    tpot = (e2e - ttft) / (out - 1) if (ok and ttft is not None and e2e is not None and out > 1) else None
    thr = ttft_threshold_ms(slo, L)
    v_tt = ttft is not None and ttft > thr
    v_tp = tpot is not None and tpot > slo["tpot_p95_ms"]
    v_e2e = e2e is not None and e2e > slo["e2e_p95_ms"]
    censored = e2e is not None and e2e >= E2E_CENSOR_MS
    viol = (not ok) or v_tt or v_tp or censored
    return {"ok": ok, "ttft_ms": ttft, "tpot_ms": tpot, "e2e_ms": e2e, "out": out, "in": L,
            "ttft_thr_ms": thr, "v_ttft": v_tt, "v_tpot": v_tp, "v_e2e": v_e2e, "censored": censored,
            "viol": viol, "viol_with_e2e": viol or v_e2e,
            "tpot_v1_ms": r["tpot"] * 1000 if (ok and r.get("tpot") is not None) else None,
            "http_502": (r.get("http_status") == 502 or "502" in str(r.get("error_message") or ""))}


def score_requests(rows: list[dict], slo_by_model: dict[str, dict], trim_s: float = 30.0,
                   max_tok: dict[str, Any] | None = None, t_ref: float | None = None) -> tuple[list[dict], dict]:
    """Score every client row. Returns (records, summary_by_model) with summary keys == score_pilot's
    plus extras (``*_mean_ms``, ``*_p90_ms``, ``*_p999_ms``, ``*_max_ms``, v1-basis TPOT, goodput).

    ``t_ref`` (epoch s) is t = 0 for the records' ``t_send`` / ``t_end`` (default: first send).
    """
    max_tok = max_tok or {}
    starts = [r["start_time"] for r in rows if r.get("start_time")]
    t_first = min(starts) if starts else 0.0
    t_ref = t_first if t_ref is None else t_ref
    recs: list[dict] = []
    acc: dict[str, Counter] = defaultdict(Counter)
    lat: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        model = r.get("model_name")
        keys = (model, ALL)
        for k in keys:
            acc[k]["output_tokens_sum_all"] += (r.get("output_tokens") or 0)
        st = r.get("start_time")
        base = {"request_id": r.get("request_id"), "model": model,
                "t_send": (st - t_ref) if st else None,
                "t_end": (r["end_time"] - t_ref) if r.get("end_time") else None,
                "continued": int(r.get("tre_continued") or 0), "interrupted": bool(r.get("stream_interrupted")),
                "target_pod": r.get("target_pod"), "finish_reason": r.get("finish_reason"),
                "http_status": r.get("http_status"), "send_lateness_ms": r.get("send_lateness_ms")}
        if st and st - t_first < trim_s:
            for k in keys:
                acc[k]["trimmed"] += 1
            recs.append({**base, "trimmed": True})
            continue
        if model not in slo_by_model:
            recs.append({**base, "trimmed": False, "unscored": True})
            continue
        s = score_request(r, slo_by_model[model])
        mt = max_tok.get(r.get("request_id"))
        rec = {**base, "trimmed": False, **s, "max_tokens": mt}
        recs.append(rec)
        for k in keys:
            c = acc[k]
            c["n"] += 1
            c["fail"] += (not s["ok"])
            c["v_ttft"] += s["v_ttft"]
            c["v_tpot"] += s["v_tpot"]
            c["viol"] += s["viol"]
            c["viol_with_e2e"] += s["viol_with_e2e"]
            c["censored_ge_149s"] += s["censored"]
            c["http_502"] += s["http_502"]
            c["output_tokens_sum"] += s["out"]
            if s["ok"]:
                c["max_tokens_unknown"] += mt is None
                c["max_tokens_hit"] += (mt is not None and s["out"] == mt)
                lat[k]["ttft"].append(s["ttft_ms"])
                lat[k]["tpot"].append(s["tpot_ms"])
                lat[k]["e2e"].append(s["e2e_ms"])
                lat[k]["tpot_v1"].append(s["tpot_v1_ms"])
            if not s["viol"]:
                c["good"] += 1
                c["good_output_tokens"] += s["out"]
    scored_t = [x["t_send"] for x in recs if not x.get("trimmed") and x.get("t_send") is not None and "viol" in x]
    window = (max(scored_t) - (t_first - t_ref + trim_s)) if scored_t else 0.0
    out: dict[str, dict] = {}
    for key in sorted(acc, key=lambda k: (k != ALL, str(k))):
        c = acc[key]
        n = max(c["n"], 1)
        ok_known = c["n"] - c["fail"] - c["max_tokens_unknown"]
        row: dict[str, Any] = {**c, "V_req_pct": round(100 * c["viol"] / n, 3),
                               "max_tokens_hit_frac": round(c["max_tokens_hit"] / ok_known, 5) if ok_known > 0 else None,
                               "V_req_with_e2e_pct": round(100 * c["viol_with_e2e"] / n, 3),
                               **{f"{m}_p{p}_ms": pct(lat[key][m], p) for m in ("ttft", "tpot", "e2e") for p in (50, 95, 99)}}
        for m in ("ttft", "tpot", "e2e"):
            v = [x for x in lat[key][m] if x is not None]
            row[f"{m}_mean_ms"] = round(sum(v) / len(v), 3) if v else None
            row[f"{m}_p90_ms"] = pct(v, 90)
            row[f"{m}_p999_ms"] = pct(v, 99.9)
            row[f"{m}_max_ms"] = round(max(v), 3) if v else None
        for p in (50, 95, 99):
            row[f"tpot_v1_p{p}_ms"] = pct(lat[key]["tpot_v1"], p)
        row["slo_attainment_pct"] = round(100 - row["V_req_pct"], 3) if c["n"] else None
        row["scored_window_s"] = round(window, 3)
        row["goodput_rps"] = round(c["good"] / window, 4) if window > 0 else None
        row["good_output_tokens_per_s"] = round(c["good_output_tokens"] / window, 2) if window > 0 else None
        out[key] = row
    return recs, out


# ---------------------------------------------------------------- replica / GPU accounting
def layout_series(arm: Arm) -> dict[str, list[tuple[float, int, int, int, int]]]:
    """model -> [(ts, awake, routable, gpus_awake, gpus_routable)] (routable = awake - hidden)."""
    out: dict[str, list] = defaultdict(list)
    for ts, models in arm.layout:
        for m in arm.models:
            awake, hidden = models.get(m, ([], []))
            rout = [b for b in awake if b not in set(hidden)]
            g = lambda bs: sum(len(parse_binding(b)[2]) for b in bs)  # noqa: E731
            out[m].append((ts, len(awake), len(rout), g(awake), g(rout)))
    return out


def integrate_step(series: list[tuple], idx: int, t_a: float, t_b: float) -> float:
    """∫ value dt over [t_a, t_b], value held from one sample to the next (no extrapolation)."""
    tot = 0.0
    for a, b in zip(series, series[1:]):
        lo, hi = max(a[0], t_a), min(b[0], t_b)
        if hi > lo:
            tot += a[idx] * (hi - lo)
    return tot


def gpu_accounting(arm: Arm, recs_summary: dict[str, dict]) -> dict[str, dict]:
    """GPU-seconds over the load window [load_start, load_end] (awake GPUs, routable GPUs),
    mean GPUs, GPU-s per good request, and the compare_arms.py-compatible fields."""
    ser = layout_series(arm)
    out: dict[str, dict] = {}
    if not ser or arm.t_load is None or arm.t_end is None:
        return out
    load_s = arm.t_end - arm.t_load
    tot_awake = tot_rout = 0.0
    compat_gpu_s = 0.0
    compat_changes = 0
    for m, s in ser.items():
        ga = integrate_step(s, 3, arm.t_load, arm.t_end)
        gr = integrate_step(s, 4, arm.t_load, arm.t_end)
        ra = integrate_step(s, 1, arm.t_load, arm.t_end)
        # analyze.py basis: rows with ts in [T_LOAD, T_END + 30], replicas (not GPUs), rounded
        win = [x for x in s if arm.t_load <= x[0] <= arm.t_end + 30]
        rs = sum(a[1] * (b[0] - a[0]) for a, b in zip(win, win[1:]))
        changes = sum(1 for a, b in zip(win, win[1:]) if a[1] != b[1])
        compat_gpu_s += round(rs) * arm.tp.get(m, 1)
        compat_changes += changes
        good = (recs_summary.get(m) or {}).get("good")
        out[m] = {"gpu_s": round(ga, 1), "routable_gpu_s": round(gr, 1), "hidden_gpu_s": round(ga - gr, 1),
                  "replica_s": round(ra, 1), "mean_gpus": round(ga / load_s, 3) if load_s else None,
                  "gpu_s_per_good_req": round(ga / good, 4) if good else None,
                  "max_awake": max((x[1] for x in win), default=None),
                  "min_routable": min((x[2] for x in win), default=None),
                  "compat_replica_s": round(rs), "compat_layout_changes": changes,
                  "compat_mean_gpus": round(round(rs) * arm.tp.get(m, 1) / load_s, 2) if load_s else None}
        tot_awake += ga
        tot_rout += gr
    good_all = (recs_summary.get(ALL) or {}).get("good")
    out[ALL] = {"gpu_s": round(tot_awake, 1), "routable_gpu_s": round(tot_rout, 1),
                "hidden_gpu_s": round(tot_awake - tot_rout, 1), "load_s": load_s,
                "mean_gpus": round(tot_awake / load_s, 3) if load_s else None,
                "gpu_s_per_good_req": round(tot_awake / good_all, 4) if good_all else None,
                "compat_mean_gpus": round(compat_gpu_s / load_s, 2) if load_s else None,
                "compat_layout_changes": compat_changes}
    return out


def step_value_at(series: list[tuple], idx: int, t: float):
    """Value of a sample-and-hold series at time t (None before the first sample)."""
    v = None
    for x in series:
        if x[0] > t:
            break
        v = x[idx]
    return v


# ---------------------------------------------------------------- decisions (per decision source)
def decision_source(arm: Arm) -> str:
    """Who actuates in this arm: baseline shell, AIBrix APA, or the TRE controller."""
    if arm.bl_decisions:
        return "baseline"
    if any(pa for _, pa in arm.apa_status):
        return "apa"
    return "tre"


def decisions(arm: Arm) -> list[dict]:
    """Scale decisions of the arm's own decision source, as {ts, model, dir (+1/-1), source, detail}.
    TRE controller actions in a non-TRE arm (counterfactual logging) are excluded."""
    src = decision_source(arm)
    out: list[dict] = []
    if src == "tre":
        for t in arm.ctrl_ticks:
            for a in t["actions"]:
                if not isinstance(a, dict):
                    continue
                det = {"loop": t["loop"], "submitted": t["submitted"], "reason": a.get("reason"),
                       "kind": a.get("kind"), "delta": a.get("delta")}
                donor, recv = a.get("donor"), a.get("receiver")
                if donor and recv:
                    out.append({"ts": t["ts"], "model": recv, "dir": 1, "source": "tre", "detail": det})
                    out.append({"ts": t["ts"], "model": donor, "dir": -1, "source": "tre", "detail": det})
                    continue
                delta = a.get("delta")
                m = a.get("model") or recv or donor
                if m is None or not delta:
                    continue
                out.append({"ts": t["ts"], "model": m, "dir": 1 if delta > 0 else -1, "source": "tre", "detail": det})
    elif src == "baseline":
        for r in arm.bl_decisions:
            if r.get("action") in ("up", "down"):
                out.append({"ts": r["ts_ms"] / 1000.0, "model": r.get("model"), "dir": 1 if r["action"] == "up" else -1,
                            "source": "baseline", "detail": {"reason": r.get("reason"), "raw_desired": r.get("raw_desired"),
                                                             "clamped": r.get("clamped"), "awake": r.get("awake")}})
    else:
        prev: dict[str, Any] = {}
        for ts, pa in arm.apa_status:
            for name, st in pa.items():
                m = next((x for x in sorted(arm.models, key=len, reverse=True) if name.startswith(x)), name)
                d = (st or {}).get("desiredScale")
                if d is None:
                    continue
                if m in prev and d != prev[m]:
                    out.append({"ts": ts, "model": m, "dir": 1 if d > prev[m] else -1, "source": "apa",
                                "detail": {"desired": d, "prev": prev[m]}})
                prev[m] = d
    out.sort(key=lambda x: x["ts"])
    return out


def targets(arm: Arm) -> dict[str, list[tuple[float, float]]]:
    """Target replicas per model over time from the arm's decision source."""
    src = decision_source(arm)
    out: dict[str, list] = defaultdict(list)
    if src == "tre":
        for s in arm.signal:
            if s.get("replicas_target") is not None:
                out[s["model"]].append((s["ts"], s["replicas_target"]))
    elif src == "baseline":
        for r in arm.bl_decisions:
            if r.get("clamped") is not None:
                out[r["model"]].append((r["ts_ms"] / 1000.0, float(r["clamped"])))
    else:
        for ts, pa in arm.apa_status:
            for name, st in pa.items():
                m = next((x for x in sorted(arm.models, key=len, reverse=True) if name.startswith(x)), name)
                if (st or {}).get("desiredScale") is not None:
                    out[m].append((ts, float(st["desiredScale"])))
    for v in out.values():
        v.sort()
    return out


# ---------------------------------------------------------------- onsets
def offered_rate(arm: Arm, bin_s: float = 1.0) -> tuple[float, dict[str, np.ndarray]]:
    """Planned arrivals per model per bin from traces.json, on the reference time base
    (t = 0 = load start). Returns (t_origin_rel, {model: counts/s})."""
    if not arm.traces or arm.t_load is None:
        return 0.0, {}
    sends = {r["request_id"]: r["start_time"] for r in arm.requests if r.get("start_time")}
    deltas = [sends[t["request_id"]] - t["timestamp"] for t in arm.traces if t["request_id"] in sends]
    base = (statistics.median(deltas) - arm.t_load) if deltas else 0.0
    t_max = max(t["timestamp"] for t in arm.traces) + base
    n = int(math.ceil((t_max + 1) / bin_s)) + 1
    out = {m: np.zeros(n) for m in arm.models}
    for t in arm.traces:
        i = int((t["timestamp"] + base) // bin_s)
        if 0 <= i < n and t["model_name"] in out:
            out[t["model_name"]][i] += 1.0 / bin_s
    return base, out


def detect_onsets(arm: Arm, smooth_s: int = 15, min_gap_s: float = 30.0, min_ratio: float = 1.5) -> list[dict]:
    """Hot-phase onsets per model: [{model, t_on, t_off, lo, hi, source}].

    Segments (trace phase table) when available: a segment whose rps exceeds the previous one by
    >= 25 % starts an onset (also the first segment when it is above the model's median rps); the
    onset ends where the rps drops again. Otherwise from planned arrivals: centred moving average
    over ``smooth_s`` seconds, threshold midway between its p10 and p90 inside the load (from the first
    to the last arrival, minus half a window at each end), upward crossings
    (debounced by ``min_gap_s``); no onsets when p90 < ``min_ratio`` * p10.
    """
    base, _ = offered_rate(arm) if arm.traces else (0.0, {})
    out: list[dict] = []
    if arm.trace_segments:
        for m, segs in arm.trace_segments.items():
            if m not in arm.models or not isinstance(segs, list):
                continue
            segs = sorted(segs, key=lambda s: s["start_time"])
            rates = [float(s.get("rps") or 0) for s in segs]
            med = statistics.median(rates) if rates else 0
            i = 0
            while i < len(segs):
                prev = rates[i - 1] if i else None
                hot = (prev is not None and rates[i] >= 1.25 * prev and rates[i] > 0) or (prev is None and rates[i] > med)
                if not hot:
                    i += 1
                    continue
                j = i
                while j + 1 < len(segs) and rates[j + 1] >= rates[i] * 0.8:
                    j += 1
                out.append({"model": m, "t_on": segs[i]["start_time"] + base, "t_off": segs[j]["end_time"] + base,
                            "lo": prev if prev is not None else min(rates), "hi": rates[i], "source": "segments",
                            "at_start": prev is None})
                i = j + 1
        return sorted(out, key=lambda x: x["t_on"])
    _, rate = offered_rate(arm)
    for m, r in rate.items():
        if len(r) < 3 * smooth_s:
            continue
        k = np.ones(smooth_s) / smooth_s
        sm = np.convolve(r, k, mode="same")
        nz = np.nonzero(r)[0]
        if not len(nz):
            continue
        a, b = nz[0] + smooth_s // 2, nz[-1] - smooth_s // 2  # inside the load, away from the edge effects
        core = sm[a:b + 1] if b > a else sm
        lo, hi = np.percentile(core, 10), np.percentile(core, 90)
        if hi < min_ratio * max(lo, 1e-9):
            continue
        thr = (lo + hi) / 2
        above = sm > thr
        ons, offs = [], []
        if above[0]:
            ons.append(0)
        for i in range(1, len(sm)):
            if above[i] and not above[i - 1]:
                if not ons or i - ons[-1] >= min_gap_s:
                    ons.append(i)
            if not above[i] and above[i - 1]:
                offs.append(i)
        for i in ons:
            off = next((o for o in offs if o > i), len(sm) - 1)
            out.append({"model": m, "t_on": float(i), "t_off": float(off), "lo": round(float(lo), 3),
                        "hi": round(float(hi), 3), "source": "arrivals", "at_start": i == 0})
    return sorted(out, key=lambda x: x["t_on"])


def onset_latencies(arm: Arm, recs: list[dict], onsets: list[dict] | None = None) -> list[dict]:
    """Per onset of model m at t_on (reference seconds, t=0 = load start):
    decision  first scale-up decision for m in [t_on, t_off) by the arm's decision source
    awake     first layout sample with awake(m) > awake(m) at t_on
    routable  first layout sample with routable(m) > routable(m) at t_on
    donor     first layout sample with awake(x) < awake(x) at t_on for some other model x (GPU released)
    slo_ok    start of the first 10 s bin after `awake` whose V_req(m) <= 5 % (requests by send time)
    Latencies are the stage time minus t_on (None = did not happen inside the hot phase)."""
    onsets = detect_onsets(arm) if onsets is None else onsets
    ser = layout_series(arm)
    dec = decisions(arm)
    rel = lambda ts: ts - arm.t_load  # noqa: E731
    out = []
    for o in onsets:
        m, t_on, t_off = o["model"], o["t_on"], o["t_off"]
        s = [(rel(x[0]),) + tuple(x[1:]) for x in ser.get(m, [])]
        a0 = step_value_at(s, 1, t_on)
        r0 = step_value_at(s, 2, t_on)
        row = {**o, "awake_at_onset": a0, "routable_at_onset": r0}
        row["t_decision"] = next((rel(d["ts"]) for d in dec if d["model"] == m and d["dir"] > 0
                                  and t_on <= rel(d["ts"]) < t_off), None)
        row["t_awake"] = next((x[0] for x in s if t_on <= x[0] < t_off and a0 is not None and x[1] > a0), None)
        row["t_routable"] = next((x[0] for x in s if t_on <= x[0] < t_off and r0 is not None and x[2] > r0), None)
        donor_t, donor_m = None, None
        for x, sx in ser.items():
            if x == m:
                continue
            sxr = [(rel(y[0]),) + tuple(y[1:]) for y in sx]
            ax0 = step_value_at(sxr, 1, t_on)
            t = next((y[0] for y in sxr if t_on <= y[0] < t_off and ax0 is not None and y[1] < ax0), None)
            if t is not None and (donor_t is None or t < donor_t):
                donor_t, donor_m = t, x
        row["t_donor"], row["donor_model"] = donor_t, donor_m
        row["max_awake_in_phase"] = max((x[1] for x in s if t_on <= x[0] < t_off), default=None)
        mine = [r for r in recs if r.get("model") == m and "viol" in r and r.get("t_send") is not None
                and t_on <= r["t_send"] < t_off]
        row["n_req_phase"] = len(mine)
        row["V_req_phase_pct"] = round(100 * sum(r["viol"] for r in mine) / len(mine), 3) if mine else None
        row["ttft_p95_phase_ms"] = pct([r["ttft_ms"] for r in mine if r.get("ok")], 95)
        row["t_slo_ok"] = None
        if row["t_awake"] is not None:
            b0 = int(row["t_awake"] // 10) * 10
            for b in range(b0, int(t_off) + 10, 10):
                inb = [r["viol"] for r in mine if b <= r["t_send"] < b + 10]
                if inb and sum(inb) / len(inb) <= 0.05:
                    row["t_slo_ok"] = float(b)
                    break
        for st in ("decision", "awake", "routable", "donor", "slo_ok"):
            v = row[f"t_{st}"]
            row[f"lat_{st}_s"] = None if v is None else round(v - t_on, 3)
        row["decision_to_awake_s"] = (None if row["t_decision"] is None or row["t_awake"] is None
                                      else round(row["t_awake"] - row["t_decision"], 3))
        out.append(row)
    return out


def decision_points(onset_rows: list[dict], points: list[dict]) -> list[dict]:
    """Pre-registered decision points, e.g. T7: [{surge_model, expect: {donor: X} | {no_move: true}}].
    For each onset of ``surge_model``: observed = donor model (first other model released), 'free'
    (receiver grew without a donor), or 'no_move' (no growth, no donor)."""
    res = []
    for p in points or []:
        for o in onset_rows:
            if o["model"] != p.get("surge_model") or o.get("at_start"):
                continue
            grew = (o.get("max_awake_in_phase") or 0) > (o.get("awake_at_onset") or 0)
            obs = o.get("donor_model") or ("free" if grew else "no_move")
            exp = p.get("expect") or {}
            if "donor" in exp:
                ok = obs == exp["donor"]
                want = exp["donor"]
            elif exp.get("no_move"):
                ok = obs == "no_move"
                want = "no_move"
            else:
                ok, want = None, None
            res.append({"name": p.get("name") or f"{p.get('surge_model')} surge", "surge_model": o["model"],
                        "t_on": o["t_on"], "observed": obs, "expected": want, "pass": ok,
                        "lat_donor_s": o.get("lat_donor_s"), "lat_awake_s": o.get("lat_awake_s")})
    return res


# ---------------------------------------------------------------- interruption cost
def interruption(arm: Arm, recs: list[dict]) -> dict:
    """Aborts (SM sleeps), continuations (sidecar), client-visible effects, and the added latency of
    continued requests vs non-continued requests of the same model sent in the same 30 s bin."""
    t_a, t_b = (arm.t_load or 0) - 5, (arm.t_end or 0) + 60
    sleeps = [e for e in arm.sm_events if e["kind"] == "sleep" and t_a <= e["ts"] <= t_b]
    paths = Counter(e["rec"].get("path") for e in sleeps)
    res: dict[str, Any] = {
        "sm_sleeps": len(sleeps), "sm_sleeps_by_path": dict(paths),
        "sm_sleep_status": dict(Counter(e["rec"].get("status") for e in sleeps)),
        "sm_forced_abort_requests": sum(int(e["rec"].get("forced_abort_requests") or 0) for e in sleeps),
        "sm_non_continuable_at_sleep": sum(int(e["rec"].get("non_continuable_at_sleep") or 0) for e in sleeps),
        "sm_wakes": sum(1 for e in arm.sm_events if e["kind"] == "wake_done" and t_a <= e["ts"] <= t_b),
    }
    re_ = [e for e in arm.sidecar if e.get("event") == "tre_reissue"]
    in_win = [e for e in re_ if e.get("ts") is None or t_a <= e["ts"] <= t_b]
    gaps = [float(e["gap_ms"]) for e in in_win if e.get("gap_ms") is not None]
    res.update({"sidecar_reissue_events": len(in_win),
                "sidecar_by_kind_reason": {f"{k}:{r}": n for (k, r), n in
                                           Counter((e.get("kind"), e.get("reason")) for e in in_win).items()},
                "continuation_gap_ms_p50": pct(gaps, 50), "continuation_gap_ms_p95": pct(gaps, 95),
                "continuation_gap_ms_max": max(gaps) if gaps else None,
                "continuation_gap_ms_sum": round(sum(gaps), 1) if gaps else 0.0,
                "sidecar_sleep_durations_s": [e.get("duration_s") for e in arm.sidecar if e.get("event") == "tre_sleep"]})
    res["client_continued_requests"] = sum(1 for r in recs if r.get("continued"))
    res["client_continuations_total"] = sum(int(r.get("continued") or 0) for r in recs)
    res["client_interrupted"] = sum(1 for r in recs if r.get("interrupted"))
    res["client_failed"] = sum(1 for r in recs if r.get("ok") is False)
    # added latency (scored requests only)
    sc = [r for r in recs if "viol" in r and r.get("ok") and r.get("t_send") is not None]
    by: dict[tuple, list] = defaultdict(list)
    for r in sc:
        if not r.get("continued"):
            by[(r["model"], int(r["t_send"] // 30))].append(r)
    d_e2e, d_ttft = [], []
    for r in sc:
        if r.get("continued"):
            peers = by.get((r["model"], int(r["t_send"] // 30)))
            if peers:
                d_e2e.append(r["e2e_ms"] - statistics.median(p["e2e_ms"] for p in peers))
                d_ttft.append(r["ttft_ms"] - statistics.median(p["ttft_ms"] for p in peers))
    res["added_e2e_ms_p50"] = pct(d_e2e, 50)
    res["added_e2e_ms_p95"] = pct(d_e2e, 95)
    res["added_ttft_ms_p50"] = pct(d_ttft, 50)
    res["continued_scored"] = len(d_e2e)
    res["continued_viol_pct"] = (round(100 * sum(r["viol"] for r in sc if r.get("continued")) /
                                       max(1, sum(1 for r in sc if r.get("continued"))), 3)
                                 if any(r.get("continued") for r in sc) else None)
    return res


# ---------------------------------------------------------------- wake / sleep durations
def switch_durations(arm: Arm) -> dict:
    wakes = []
    starts: dict[str, float] = {}
    for e in arm.sm_events:
        key = e.get("binding_id") or e.get("serve_id")
        if e["kind"] == "wake_start":
            starts[key] = e["ts"]
        elif e["kind"] == "wake_done":
            ph = e["rec"].get("phases_ms") or {}
            wakes.append({"ts": e["ts"], "model": e["model"], "binding_id": key,
                          "total_s": round(e["ts"] - starts.pop(key), 3) if key in starts else None,
                          "wake_up_s": ph.get("wake_up", 0) / 1000 if ph.get("wake_up") is not None else None})
    sleeps = [float(e["duration_s"]) for e in arm.sidecar if e.get("event") == "tre_sleep" and e.get("duration_s") is not None]
    wu = [w["wake_up_s"] for w in wakes if w["wake_up_s"] is not None]
    wt = [w["total_s"] for w in wakes if w["total_s"] is not None]
    return {"wakes": wakes, "wake_up_s_p50": pct(wu, 50), "wake_up_s_p95": pct(wu, 95),
            "wake_total_s_p50": pct(wt, 50), "wake_total_s_p95": pct(wt, 95),
            "sleep_s_p50": pct(sleeps, 50), "sleep_s_p95": pct(sleeps, 95), "n_sleep_durations": len(sleeps)}


# ---------------------------------------------------------------- control overhead
def control_overhead(arm: Arm) -> dict:
    res: dict[str, Any] = {}
    a = (arm.t_load or 0) * 1000
    b = (arm.t_end or 0) * 1000
    if arm.profile_rows:
        cpu, rss, tick = [], [], defaultdict(list)
        for r in arm.profile_rows:
            ts = float(r.get("ts_ms") or 0)
            if not (a <= ts <= b):
                continue
            if r.get("kind") == "proc":
                cpu.append(float(r["cpu_percent"]))
                rss.append(float(r["rss_mib"]))
            elif r.get("kind") == "tick" and r.get("tick_total_ns"):
                tick[r.get("loop") or "?"].append(float(r["tick_total_ns"]) / 1e6)
        res["controller"] = {"cpu_pct_p50": pct(cpu, 50), "cpu_pct_p95": pct(cpu, 95), "cpu_pct_max": max(cpu) if cpu else None,
                             "rss_mib_p50": pct(rss, 50), "rss_mib_max": max(rss) if rss else None,
                             **{f"tick_{k}_ms_p50": pct(v, 50) for k, v in tick.items()},
                             **{f"tick_{k}_ms_p99": pct(v, 99) for k, v in tick.items()},
                             "source": "controller_profile.csv"}
    if arm.resource_rows:  # spec'd collector: one row per sample {ts, component, cpu_cores, rss_mib}
        by = defaultdict(lambda: ([], []))
        for r in arm.resource_rows:
            ts = float(r.get("ts") or 0)
            if not (a / 1000 <= ts <= b / 1000):
                continue
            c = by[r.get("component")]
            if r.get("cpu_cores") not in (None, ""):
                c[0].append(float(r["cpu_cores"]))
            if r.get("rss_mib") not in (None, ""):
                c[1].append(float(r["rss_mib"]))
        for comp, (cpu, rss) in by.items():
            res[comp] = {"cpu_cores_p50": pct(cpu, 50), "cpu_cores_p95": pct(cpu, 95),
                         "rss_mib_p50": pct(rss, 50), "rss_mib_max": max(rss) if rss else None,
                         "source": "resource_usage"}
    return res


# ---------------------------------------------------------------- confidence intervals
def _blocks(recs: list[dict], model: str, block_s: float):
    rows = [r for r in recs if "viol" in r and r.get("t_send") is not None and (model == ALL or r["model"] == model)]
    by: dict[int, list] = defaultdict(list)
    for r in rows:
        by[int(r["t_send"] // block_s)].append(r)
    return [by[k] for k in sorted(by)]


def _quantile(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if len(a) else float("nan")


def block_bootstrap(recs: list[dict], model: str, stats: dict[str, Callable[[list[dict]], float]],
                    block_s: float = 30.0, reps: int = 1000, seed: int = 0) -> dict[str, tuple]:
    """95 % CI of each statistic by resampling ``block_s``-second blocks of requests (by send time)
    with replacement - requests inside a block are correlated through the shared system state."""
    blocks = _blocks(recs, model, block_s)
    if len(blocks) < 2:
        return {k: (None, None) for k in stats}
    rng = np.random.default_rng(seed)
    res = {k: [] for k in stats}
    for _ in range(reps):
        idx = rng.integers(0, len(blocks), len(blocks))
        sample = [r for i in idx for r in blocks[i]]
        for k, f in stats.items():
            res[k].append(f(sample))
    return {k: (round(_quantile(np.array([x for x in v if x == x]), 2.5), 3),
                round(_quantile(np.array([x for x in v if x == x]), 97.5), 3)) for k, v in res.items()}


def vreq_stat(rs: list[dict]) -> float:
    return 100.0 * sum(r["viol"] for r in rs) / len(rs) if rs else float("nan")


def ttft_p95_stat(rs: list[dict]) -> float:
    v = [r["ttft_ms"] for r in rs if r.get("ok") and r.get("ttft_ms") is not None]
    return float(np.percentile(v, 95)) if v else float("nan")


def paired_bootstrap(recs_a: list[dict], recs_b: list[dict], model: str, block_s: float = 30.0,
                     reps: int = 1000, seed: int = 0) -> dict:
    """A - B on the same request plan (paired by request_id, blocks by B's send time):
    ΔV_req (percentage points) and ΔP95 TTFT (ms) with block-bootstrap 95 % CIs."""
    ia = {r["request_id"]: r for r in recs_a if "viol" in r}
    pairs = [(ia[r["request_id"]], r) for r in recs_b if "viol" in r and r["request_id"] in ia
             and r.get("t_send") is not None and (model == ALL or r["model"] == model)]
    if not pairs:
        return {"n_pairs": 0}
    by: dict[int, list] = defaultdict(list)
    for p in pairs:
        by[int(p[1]["t_send"] // block_s)].append(p)
    blocks = [by[k] for k in sorted(by)]

    def stat(ps):
        a = [x[0] for x in ps]
        b = [x[1] for x in ps]
        return vreq_stat(a) - vreq_stat(b), ttft_p95_stat(a) - ttft_p95_stat(b)

    point = stat(pairs)
    rng = np.random.default_rng(seed)
    dv, dt = [], []
    for _ in range(reps):
        idx = rng.integers(0, len(blocks), len(blocks))
        s = stat([p for i in idx for p in blocks[i]])
        dv.append(s[0])
        dt.append(s[1])
    dv_a, dt_a = np.array(dv), np.array([x for x in dt if x == x])
    return {"n_pairs": len(pairs), "d_vreq_pp": round(point[0], 3),
            "d_vreq_ci": (round(_quantile(dv_a, 2.5), 3), round(_quantile(dv_a, 97.5), 3)),
            "d_ttft_p95_ms": round(point[1], 3),
            "d_ttft_p95_ci": (round(_quantile(dt_a, 2.5), 3), round(_quantile(dt_a, 97.5), 3))}


# two-sided 95 % Student t critical values, df 1..30
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
        2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def seed_ci(diffs: list[float]) -> dict:
    """Mean of per-seed paired differences with a t-based 95 % CI (None for < 2 seeds)."""
    n = len(diffs)
    if n == 0:
        return {"n_seeds": 0, "mean": None, "ci": (None, None)}
    mu = sum(diffs) / n
    if n < 2:
        return {"n_seeds": 1, "mean": round(mu, 3), "ci": (None, None)}
    sd = statistics.stdev(diffs)
    t = _T95[min(n - 1, len(_T95)) - 1] if n - 1 <= len(_T95) else 1.96
    h = t * sd / math.sqrt(n)
    return {"n_seeds": n, "mean": round(mu, 3), "ci": (round(mu - h, 3), round(mu + h, 3))}

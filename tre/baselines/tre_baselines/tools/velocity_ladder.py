"""TokenScale V_b (3x3 cells) and PreServe mu, measured direct to each model's single awake pod.

Port of the TRE v1 tool (``tre_code`` dev ``python/tre_tokenscale/tools/profile_velocity.py`` @
``19544cb``, the 2026-10-25 run; v1 decisions D46 / D49 / D50 / D51 / D53 / D58) to v2 and vLLM 0.30
(decision 2026-10-26, ``docs/campaign-v2-20261026-plan.md``). Subcommands:

* ``ladder`` (V): per model, on its **single awake, routable pod, sent to directly**
  (``http://<pod IP>:<port>/v1``; V is a property of the engine instance and the gateway stays out
  of it); the models run **in parallel**, one process each, optionally pinned (``--cpus``).
  One open-loop Poisson rate ladder per cell, at the cell's calibration point
  (``bucket_centers`` of the buckets report, ``tools/tokenscale_buckets.py``). Ladder centre =
  the rps the service-time fit ``1/rps = a*in + b*out + c`` (``--fit``) predicts there, times the
  model's ``--anchor`` scale; rates = centre x ``--factors`` (0.8 ... 1.3). The fit only places
  the ladder; V is measured.

  - throughput of a step = increase of the pod's own counters ``vllm:prompt_tokens_total`` +
    ``vllm:generation_tokens_total`` over the **last ``--measure-s`` (60) s** of the step; the step
    lasts ``max(--step-s, measure_s + --warmup-lifetimes x max_num_seqs / rate)`` so the warm-up
    covers >= 1.5 saturated request lifetimes (running is capped at ``max_num_seqs``);
  - a step is **effective** when the pod set is unchanged and the pod awake at both ends, at most
    ``--max-fail-frac`` (5 %) of its window's requests failed, no request reached
    ``--route-timeout-s`` (150, what the experiment's route would cut), the sender kept the
    schedule (p99 lateness and p99 header-send lag <= 1 s) and the window has >= ``--min-samples``
    usable /metrics samples and >= ``--min-completed`` completed requests;
  - a step is **full** (D53) when the least-squares slope of ``vllm:num_requests_waiting`` over the
    window is above ``--knee-slope-frac`` (2 %) x the offered req/s: arrivals exceed what the
    engine serves. Running, KV usage and preemptions are recorded, not judged;
  - **V_b = the highest tok/s of an effective full step**; **stop** after two consecutive
    effective full steps (``saturated``); otherwise the ladder extends by ``--extend-factor``
    (1.1) up to ``--max-steps`` (10). If the first step is already full or ineffective, downward
    steps (0.7, 0.6 x centre) run until one is effective and not full. No V_P (D51).
  - the v2 knee (highest bounded step) is kept for comparison only; D52 (disclosure only): about
    ``--stream-frac`` of the requests stream, giving p95 TTFT / TPOT and the best step meeting the
    TokenScale paper SLO.
  - every request is logged (``raw/<model>_<cell>_<rate>.jsonl``: send, done, ok, prompt and
    completion tokens), so ``mu`` can re-judge the same steps.

* ``isolated``: one request at a time per model (pod idle before each), the mix shape and the
  nine cell centres, ``--runs`` each: the isolated normalized latency NL = e2e / output tokens,
  the base of PreServe's SLO (3 x isolated NL of the mix shape, paper section 5.1; v1 D58).

* ``mu`` (offline): PreServe Alg. 1 lines 8-11 from the ladder: a step passes when it is
  effective and **every** request sent in its window completed with NL <= SLO; mu_p / mu_d / mu_t
  = per term the highest engine rate over passing steps. Variant B (v1 D58, used) keeps only the
  cells whose P share in / (in + out) lies in the campaign traces' window share range
  (``p_share.pooled`` [min, max] of the buckets report); variant A (all cells) is reported.

* ``summarize`` (offline): V / knees again from step logs (logs of runs over disjoint ``--cells``
  merge into one grid).

Validity (state, not timers; all cluster access read only): a real run needs
``--i-have-user-approval`` and the exclusive-window marker (``calibration`` type, start / end with a
time zone, now inside, ``--marker-tag`` a token of its note), re-read before and after every
step; before sending: TRE controller and SM actuation both ``observe``, no AIBrix PodAutoscaler on
a profiled model, the baseline scaler Deployment at 0 replicas with no pod, the SM's
``/v2/state`` shows exactly one awake replica per model, exactly one routable
(``tre.aibrix.io/routable=true``), Ready pod per model and it reports ``/is_sleeping`` false. At
both ends of every step: the same pod (name, IP, node, every container's restartCount and
startedAt) still awake; a token counter that falls means the engine restarted. Either makes the
step invalid and **aborts that model**. After every step the pod drains to nothing in flight.
/metrics is sampled once a second by a **separate process** (one CPU when ``--cpus`` is given).
Prompts are natural text fitted to the exact ``usage.prompt_tokens`` with the chat template
(``tre_replayer.engine.prompts``, the campaign's own prompt builder), built before each step.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import datetime as _dt
import json
import math
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

import yaml

PROMPT_COUNTER = "vllm:prompt_tokens_total"
GEN_COUNTER = "vllm:generation_tokens_total"
WAITING_GAUGE = "vllm:num_requests_waiting"
RUNNING_GAUGE = "vllm:num_requests_running"
PREEMPT_COUNTER = "vllm:num_preemptions_total"
KV_GAUGE = "vllm:kv_cache_usage_perc"
ROUTABLE_LABEL = "tre.aibrix.io/routable=true"
ENV_EXCLUSIVE_MARKER = "TRE_EXCLUSIVE_WINDOW_FILE"
DEFAULT_MARKER = "/data/nfs_shared_data/xxy/TRE_EXCLUSIVE_WINDOW"
DEFAULT_MAX_FAIL_FRAC = 0.05
DEFAULT_KNEE_SLOPE_FRAC = 0.02
DEFAULT_FACTORS = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3)
DEFAULT_DOWN_FACTORS = (0.7, 0.6)
DEFAULT_MIN_SAMPLES = 30
DEFAULT_STREAM_FRAC = 0.05
DEFAULT_MIN_STREAM = 20
DEFAULT_ROUTE_TIMEOUT_S = 150.0
DEFAULT_MAX_NUM_SEQS = 256
#: TokenScale paper SLO (D52 disclosure): (input upper bound, TTFT p95 ms); TPOT p95 100 ms
PAPER_TTFT_SLO_MS = ((256, 250.0), (1024, 400.0), (math.inf, 2000.0))
PAPER_TPOT_SLO_MS = 100.0
SAMPLER_JOIN_S = 20.0
#: HTTP clients per model process (request k uses client k mod CLIENT_SHARDS): one big httpcore pool
#: stalls the event loop (v1 2026-10-21, py-spy).
CLIENT_SHARDS = 64
CLIENT_KEEPALIVE = 32
#: below vLLM/uvicorn's 5 s idle keep-alive timeout (v1 D23: equal timeouts race -> ReadError)
CLIENT_KEEPALIVE_EXPIRY_S = 2.0
EXIT_ABORTED = 5
CELLS = tuple(f"b{i}{j}" for i in range(3) for j in range(3))
BASELINE_SCALER = ("tre-v2", "tre-v2-baseline-scaler")


class ProfileAbort(RuntimeError):
    pass


# ------------------------------------------------------------------ pure pieces

@dataclass
class Step:
    model: str
    kind: str                 # "bucket"
    cell: str                 # "b<i><j>"
    in_tokens: int
    out_tokens: int
    rate_rps: float
    step_s: float = 0.0
    window_s: float = 0.0
    #: wall-clock bounds (unix s) of the measured window: requests sent in [lo, hi] count
    window_lo: Optional[float] = None
    window_hi: Optional[float] = None
    arrived: int = 0
    completed: int = 0
    failed: int = 0
    prefill_tok_s: Optional[float] = None
    decode_tok_s: Optional[float] = None
    ttft_p50_ms: Optional[float] = None
    ttft_p95_ms: Optional[float] = None
    tpot_p50_ms: Optional[float] = None
    tpot_p95_ms: Optional[float] = None
    stream_n: Optional[int] = None
    waiting_slope: Optional[float] = None
    waiting_max: Optional[float] = None
    preemptions: Optional[float] = None
    mean_prompt_tokens: Optional[float] = None
    mean_completion_tokens: Optional[float] = None
    sender_p99_late_s: Optional[float] = None
    sender_late: bool = False
    send_p99_lag_s: Optional[float] = None
    send_late: bool = False
    max_latency_s: Optional[float] = None
    errors: Optional[Dict[str, int]] = None
    valid: bool = True
    note: Optional[str] = None
    ineffective: Optional[str] = None
    pod: Optional[str] = None
    node: Optional[str] = None
    window_samples: Optional[int] = None
    running_p50: Optional[float] = None
    running_p90: Optional[float] = None
    kv_p50: Optional[float] = None
    kv_p90: Optional[float] = None
    samples_bad: Optional[int] = None
    sampler_exit: Optional[int] = None

    @property
    def tok_s(self) -> Optional[float]:
        if self.prefill_tok_s is None or self.decode_tok_s is None:
            return None
        return self.prefill_tok_s + self.decode_tok_s


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    """Nearest-rank quantile (None when empty; inf counts as a value)."""
    vals = sorted(values)
    if not vals:
        return None
    return vals[min(len(vals) - 1, max(0, math.ceil(q * len(vals)) - 1))]


def slope(points: Sequence[Tuple[float, float]]) -> Optional[float]:
    """Least-squares slope of (t_s, y) points (None with fewer than 3 or no time spread)."""
    pts = [(float(t), float(y)) for t, y in points if y is not None and math.isfinite(y)]
    if len(pts) < 3:
        return None
    mt = sum(t for t, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    den = sum((t - mt) ** 2 for t, _ in pts)
    return None if den <= 0 else sum((t - mt) * (y - my) for t, y in pts) / den


def queue_bounded(s: Step, slope_frac: float = DEFAULT_KNEE_SLOPE_FRAC) -> Optional[bool]:
    if s.waiting_slope is None:
        return None
    return s.waiting_slope <= slope_frac * float(s.rate_rps)


def has_timeout(errors: Optional[Dict[str, int]]) -> bool:
    return any("timeout" in e.lower() or "timed out" in e.lower() for e in (errors or {}))


@dataclass(frozen=True)
class Rules:
    """The effective-step criteria and the stop rule (one place, also used offline)."""

    max_fail_frac: float = DEFAULT_MAX_FAIL_FRAC
    route_timeout_s: float = DEFAULT_ROUTE_TIMEOUT_S
    min_completed: int = 1
    min_samples: int = DEFAULT_MIN_SAMPLES
    min_stream: int = DEFAULT_MIN_STREAM
    knee_slope_frac: float = DEFAULT_KNEE_SLOPE_FRAC


def ineffective_why(s: Step, r: Rules) -> Optional[str]:
    """None when the step is effective, else the first reason it is not."""
    if not s.valid:
        return f"invalid: {s.note}"
    if s.sampler_exit not in (None, 0):
        return f"metrics_gap (sampler exit {s.sampler_exit})"
    if (s.window_samples or 0) < r.min_samples:
        return f"metrics_gap ({s.window_samples or 0} usable /metrics samples in the window < {r.min_samples})"
    if s.sender_late:
        return f"sender_late (p99 {s.sender_p99_late_s} s)"
    if s.send_late:
        return f"send_late (httpx p99 {s.send_p99_lag_s} s)"
    if s.arrived > 0 and s.failed / s.arrived > r.max_fail_frac:
        return f"failures {s.failed}/{s.arrived}"
    if has_timeout(s.errors) or (s.max_latency_s is not None and s.max_latency_s >= r.route_timeout_s):
        return f"route_timeout (max e2e {s.max_latency_s} s)"
    if s.tok_s is None or s.completed < max(1, r.min_completed):
        return f"no_throughput (completed {s.completed})"
    return None


def effective_steps(steps: Sequence[Step], r: Rules) -> List[Step]:
    """Effective steps in ascending rate (the downward steps run last but sort first)."""
    return sorted((s for s in steps if ineffective_why(s, r) is None), key=lambda s: s.rate_rps)


def is_full(s: Step, r: Rules) -> bool:
    """D53: the waiting queue kept growing over the window (slope > knee_slope_frac x offered req/s)."""
    return queue_bounded(s, r.knee_slope_frac) is False


def saturation(steps: Sequence[Step], r: Rules) -> Optional[Dict[str, Any]]:
    """The stop rule: two consecutive effective steps (by ascending rate) are both full."""
    eff = effective_steps(steps, r)
    for a, b in zip(eff, eff[1:]):
        if is_full(a, r) and is_full(b, r):
            return {"full_rps": [round(a.rate_rps, 3), round(b.rate_rps, 3)]}
    return None


def peak(steps: Sequence[Step], r: Rules) -> Dict[str, Any]:
    """V = the highest tok/s of an effective full step (None when no step was full: the cell stays
    uncalibrated and the policy refuses to start); ``saturated`` = two consecutive full steps."""
    full = [s for s in effective_steps(steps, r) if is_full(s, r)]
    top = max(steps, key=lambda s: s.rate_rps)
    sat = saturation(steps, r)
    out: Dict[str, Any] = {"value": None, "steps": len(steps), "full_steps": len(full),
                           "top_rps": round(top.rate_rps, 3), "min_rps": round(min(s.rate_rps for s in steps), 3),
                           "saturated": sat is not None}
    out["full"] = [{"rate_rps": round(s.rate_rps, 3), "tok_s": round(s.tok_s, 1), "kv_p50": s.kv_p50,
                    "running_p50": s.running_p50, "preemptions": s.preemptions} for s in full]
    if full:
        best = max(full, key=lambda s: s.tok_s)
        out.update(value=round(best.tok_s, 1), rate_rps=round(best.rate_rps, 3), waiting_slope=best.waiting_slope,
                   running_p50=best.running_p50, kv_p50=best.kv_p50, preemptions=best.preemptions)
    if sat is None:
        why = ineffective_why(top, r)
        out["not_saturated"] = (f"stopped at {top.rate_rps:.3f} req/s: {why}" if why else "step cap reached") + \
            f" ({len(full)} full step(s), not 2 in a row)"
    else:
        out.update(sat)
    return out


def knee(steps: Sequence[Step], r: Rules) -> Optional[Dict[str, Any]]:
    """v2 rule (2026-10-06), comparison only: the highest tok/s of an effective, queue-bounded step."""
    ok = [s for s in effective_steps(steps, r) if queue_bounded(s, r.knee_slope_frac)]
    if not ok:
        return None
    best = max(ok, key=lambda s: s.tok_s)
    return {"value": round(best.tok_s, 1), "rate_rps": round(best.rate_rps, 3), "waiting_slope": best.waiting_slope}


def paper_ttft_slo_ms(in_tokens: float) -> float:
    return next(ms for bound, ms in PAPER_TTFT_SLO_MS if in_tokens < bound)


def slo_capacity(steps: Sequence[Step], r: Rules, ttft_ms: float, tpot_ms: float) -> Optional[Dict[str, Any]]:
    """D52, disclosure only: the effective step with the highest tok/s whose streamed p95 TTFT and TPOT
    meet the SLO (at least ``min_stream`` streamed requests in the window). Not V."""
    ok = [s for s in effective_steps(steps, r) if (s.stream_n or 0) >= r.min_stream
          and s.ttft_p95_ms is not None and s.tpot_p95_ms is not None
          and s.ttft_p95_ms <= ttft_ms and s.tpot_p95_ms <= tpot_ms]
    if not ok:
        return None
    best = max(ok, key=lambda s: s.tok_s)
    return {"value": round(best.tok_s, 1), "rate_rps": round(best.rate_rps, 3), "ttft_p95_ms": best.ttft_p95_ms,
            "tpot_p95_ms": best.tpot_p95_ms, "stream_n": best.stream_n}


def summarize(steps: Sequence[Step], r: Rules) -> Dict[str, Any]:
    """One model's steps -> V grid (peaks; a cell without an effective full step is None), knees,
    per-cell details."""
    cells: Dict[str, List[Step]] = {}
    for s in steps:
        cells.setdefault(s.cell, []).append(s)
    grid: List[List[Optional[float]]] = [[None] * 3 for _ in range(3)]
    peaks: Dict[str, Any] = {}
    knees: Dict[str, Any] = {}
    slo: Dict[str, Any] = {}
    for cell in CELLS:
        st = cells.get(cell)
        if not st:
            peaks[cell] = knees[cell] = slo[cell] = None
            continue
        shape = {"in": st[0].in_tokens, "out": st[0].out_tokens}
        p, k = peak(st, r), knee(st, r)
        if p.get("value") is not None and k:
            p["v_minus_knee"] = round(p["value"] - k["value"], 1)
            p["v_over_knee"] = round(p["value"] / k["value"], 4)
        peaks[cell] = {**shape, **p}
        knees[cell] = {**shape, **(k or {"value": None})}
        t_paper = paper_ttft_slo_ms(st[0].in_tokens)
        slo[cell] = {**shape, "paper": {"ttft_p95_ms_max": t_paper, "tpot_p95_ms_max": PAPER_TPOT_SLO_MS,
                                        **(slo_capacity(st, r, t_paper, PAPER_TPOT_SLO_MS) or {"value": None})}}
        grid[int(cell[1])][int(cell[2])] = p["value"]
    return {"velocity": {"buckets": grid}, "peaks": peaks, "knees": knees, "slo_capacity": slo,
            "not_saturated": sorted(c for c, p in peaks.items() if p and not p.get("saturated"))}


def counter_sum(text: str, name: str) -> Optional[float]:
    """Sum of every sample of the exact metric ``name`` (labels ignored); None when absent."""
    total, seen = 0.0, False
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        head, _, val = line.rpartition(" ")
        if head.split("{", 1)[0].strip() == name:
            try:
                v = float(val)
            except ValueError:
                continue
            if math.isfinite(v):
                total += v
                seen = True
    return total if seen else None


# ------------------------------------------------------------------ plan and one ladder

#: measure(model, cell, in_tokens, out_tokens, rate_rps) -> Step
Measure = Callable[[str, str, int, int, float], Step]


def ladder_centre(fit: Sequence[float], tin: int, tout: int, scale: float = 1.0) -> float:
    t = fit[0] * tin + fit[1] * tout + fit[2]
    if not t > 0:
        raise ProfileAbort(f"fit predicts a non-positive service time at ({tin}, {tout})")
    return scale / t


def anchor_scale(fit: Sequence[float], tin: int, tout: int, rps: float) -> float:
    """The factor that makes the fit predict ``rps`` at (tin, tout) (a measured point on this engine)."""
    return float(rps) / ladder_centre(fit, tin, tout)


def ladder_rates(centre: float, factors: Sequence[float], extend_factor: float, max_steps: int) -> List[float]:
    """Every rate a ladder may run, in order: centre x factors, then x extend_factor up to max_steps."""
    rates = [centre * f for f in factors][:max_steps]
    while len(rates) < max_steps:
        rates.append(rates[-1] * extend_factor)
    return rates


def step_length(rate: float, step_s: float, measure_s: float, max_num_seqs: int, lifetimes: float) -> float:
    """Step duration: at least ``step_s``, and a warm-up of >= ``lifetimes`` saturated request lifetimes
    (``max_num_seqs / rate``: running is capped at max_num_seqs) before the measured window."""
    return max(float(step_s), float(measure_s) + float(lifetimes) * float(max_num_seqs) / float(rate))


def run_ladder(measure: Measure, model: str, cell: str, in_tokens: int, out_tokens: int,
               rates: Sequence[float], rules: Rules, log: Callable[[str], None],
               check: Optional[Callable[[], None]] = None, down_rates: Sequence[float] = ()) -> List[Step]:
    """Up the ``rates`` until two consecutive effective steps are full, an ineffective step or the
    cap; then, when the first step was already full (or ineffective), down the ``down_rates`` until
    a step is effective and not full. ``check`` runs before and after every step."""
    steps: List[Step] = []

    def one(rate: float) -> Step:
        if check is not None:
            check()
        st = measure(model, cell, in_tokens, out_tokens, rate)
        st.kind = "bucket"
        steps.append(st)
        st.ineffective = ineffective_why(st, rules)
        bounded = queue_bounded(st, rules.knee_slope_frac)
        log(json.dumps({"event": "step", **asdict(st), "tok_s": st.tok_s, "bounded": bounded}))
        if not st.valid:
            raise ProfileAbort(f"{model} {cell}: {st.note}")
        fail_frac = st.failed / st.arrived if st.arrived > 0 else 0.0
        if bounded is not False and fail_frac > rules.max_fail_frac:
            # a bounded (or unknown) step must serve its load: failures there are a fault
            raise ProfileAbort(f"{model} {cell}: {st.failed}/{st.arrived} requests failed at {st.rate_rps:.3f} req/s "
                               f"(> {rules.max_fail_frac:.0%})")
        if check is not None:
            check()
        return st

    for rate in rates:
        st = one(rate)
        if st.ineffective is not None:
            break  # a higher rate only overloads more (or the client cannot offer it)
        if saturation(steps, rules) is not None:
            break
    first = steps[0]
    if first.ineffective is not None or is_full(first, rules):
        for rate in down_rates:
            st = one(rate)
            if st.ineffective is None and not is_full(st, rules):
                break
    return steps


def profile_model(measure: Measure, model: str, centers: Sequence[Sequence[Sequence[int]]], fit: Sequence[float],
                  scale: float, factors: Sequence[float], *, extend_factor: float, max_steps: int, rules: Rules,
                  log: Callable[[str], None], check: Optional[Callable[[], None]] = None,
                  down_factors: Sequence[float] = DEFAULT_DOWN_FACTORS,
                  cells: Sequence[str] = CELLS) -> Tuple[Dict[str, Any], List[Step]]:
    all_steps: List[Step] = []
    for cell in cells:
        tin, tout = (int(x) for x in centers[int(cell[1])][int(cell[2])])
        centre = ladder_centre(fit, tin, tout, scale)
        rates = ladder_rates(centre, factors, extend_factor, max_steps)
        all_steps += run_ladder(measure, model, cell, tin, tout, rates, rules, log, check,
                                down_rates=[centre * f for f in down_factors])
    return summarize(all_steps, rules), all_steps


def plan(models: Sequence[str], centers: Dict[str, Any], fits: Dict[str, Sequence[float]], scales: Dict[str, float],
         factors: Sequence[float], *, extend_factor: float, max_steps: int, step_s: float, measure_s: float,
         max_num_seqs: int, lifetimes: float, overhead_s: float,
         down_factors: Sequence[float] = DEFAULT_DOWN_FACTORS,
         cells: Optional[Dict[str, Sequence[str]]] = None) -> Dict[str, Any]:
    """The ladders and the wall-time estimate (drain of an overloaded step's backlog included):
    min = 2 steps per ladder; nominal = up to the second rate above the centre; max = every rate to
    the cap plus the downward steps. Models in parallel -> wall = slowest."""

    def t_step(rate: float, centre: float) -> float:
        L = step_length(rate, step_s, measure_s, max_num_seqs, lifetimes)
        return L + overhead_s + max(0.0, rate / centre - 1.0) * L

    out: Dict[str, Any] = {"models": {}, "step_s": step_s, "measure_s": measure_s, "overhead_s": overhead_s}
    for m in models:
        mc = {}
        est = {"min": 0.0, "nominal": 0.0, "max": 0.0}
        for cell in (cells or {}).get(m, CELLS):
            tin, tout = (int(x) for x in centers[m][int(cell[1])][int(cell[2])])
            c = ladder_centre(fits[m], tin, tout, scales.get(m, 1.0))
            rates = ladder_rates(c, factors, extend_factor, max_steps)
            down = [c * f for f in down_factors]
            t = [t_step(r, c) for r in rates]
            above = [i for i, r in enumerate(rates) if r > c * (1 + 1e-9)]
            n_nom = above[1] + 1 if len(above) >= 2 else len(rates)
            est["min"] += sum(t[:2])
            est["nominal"] += sum(t[:n_nom])
            est["max"] += sum(t) + sum(t_step(r, c) for r in down)
            mc[cell] = {"in": tin, "out": tout, "centre_rps": round(c, 3),
                        "rates": [round(x, 3) for x in rates[:len(factors)]],
                        "extensions": [round(x, 3) for x in rates[len(factors):]],
                        "down": [round(x, 3) for x in down],
                        "step_s_at_centre": round(step_length(c, step_s, measure_s, max_num_seqs, lifetimes), 1)}
        out["models"][m] = {"scale": round(scales.get(m, 1.0), 4), "cells": mc,
                            "est_s": {k: round(v) for k, v in est.items()}}
    out["wall_s"] = {k: max(v["est_s"][k] for v in out["models"].values()) for k in ("min", "nominal", "max")}
    return out


def parse_cells(models: Sequence[str], spec: Optional[str]) -> Dict[str, List[str]]:
    """``--cells``: comma list of ``<cell>`` (every model) or ``<model>:<cell>``; default all 27."""
    if not spec:
        return {m: list(CELLS) for m in models}
    out: Dict[str, List[str]] = {m: [] for m in models}
    for item in filter(None, (x.strip() for x in spec.split(","))):
        m, _, cell = item.rpartition(":")
        if cell not in CELLS or (m and m not in out):
            raise ValueError(f"--cells: {item!r} is not <cell> or <model>:<cell> (cells {', '.join(CELLS)})")
        for mm in ([m] if m else models):
            if cell not in out[mm]:
                out[mm].append(cell)
    return {m: [c for c in CELLS if c in out[m]] for m in models}


# ------------------------------------------------------------------ mu (PreServe, offline)

def isolated_nl(records: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, float]], int]:
    """shape -> {mean, sd, n} of the isolated NL (s) of the clean measured runs; also the number of
    records left out (warm-up excluded): failed, wrong token counts or a busy pod before sending."""
    by: Dict[str, List[float]] = {}
    bad = 0
    for r in records:
        if r.get("run", -1) < 0:
            continue
        if (not r.get("ok") or r.get("completion_tokens") != r.get("out") or r.get("prompt_tokens") != r.get("in")
                or r.get("queue_before")):
            bad += 1
            continue
        by.setdefault(r["shape"], []).append(float(r["e2e_s"]) / float(r["completion_tokens"]))
    return {k: {"mean": statistics.mean(v), "sd": statistics.pstdev(v), "n": len(v)} for k, v in by.items()}, bad


def judge_step(step: Step, records: Sequence[Dict[str, Any]], slo_s: float, rules: Rules) -> Dict[str, Any]:
    """PreServe Alg. 1 line 8 on one ladder step: it passes when the step is effective and every
    request sent inside the measured window completed with NL = e2e / output tokens <= SLO."""
    lo, hi = step.window_lo, step.window_hi
    win = [x for x in records if lo is not None and hi is not None and lo <= x["send"] <= hi]
    nl = []
    for x in win:
        if x.get("ok"):
            out = x.get("completion_tokens") or step.out_tokens
            nl.append((x["done"] - x["send"]) / float(out))
    n, ok_n = len(win), len(nl)
    within = sum(1 for v in nl if v <= slo_s)
    effective = ineffective_why(step, rules) is None
    return {"model": step.model, "cell": step.cell, "in_tokens": step.in_tokens, "out_tokens": step.out_tokens,
            "p_share": round(step.in_tokens / (step.in_tokens + step.out_tokens), 4),
            "rate_rps": round(step.rate_rps, 3), "n": n, "failed": n - ok_n, "within": within,
            "attain": round(within / n, 4) if n else None,
            "nl_p95_ms": round(1000 * quantile(nl, 0.95), 2) if nl else None,
            "nl_max_ms": round(1000 * max(nl), 2) if nl else None,
            "prefill_tok_s": step.prefill_tok_s, "decode_tok_s": step.decode_tok_s, "total_tok_s": step.tok_s,
            "effective": effective, "passes": bool(n) and effective and within == n}


def mu_from(judged: Sequence[Dict[str, Any]], keep: Callable[[Dict[str, Any]], bool]) -> Dict[str, Any]:
    """Alg. 1 lines 9-11: per term the highest engine rate over passing steps (each term may come
    from a different step); ``bracketed`` = a faster step of the same cell failed."""
    res: Dict[str, Any] = {}
    for key, col in (("p", "prefill_tok_s"), ("d", "decode_tok_s"), ("t", "total_tok_s")):
        ok = [s for s in judged if keep(s) and s["passes"] and s[col] is not None]
        if not ok:
            res[key] = None
            continue
        best = max(ok, key=lambda s: s[col])
        above = [s for s in judged if s["cell"] == best["cell"] and s["rate_rps"] > best["rate_rps"]]
        res[key] = {"value": round(best[col], 1), "cell": best["cell"], "rate_rps": best["rate_rps"],
                    "p_share": best["p_share"], "bracketed": any(not s["passes"] for s in above)}
    return res


# ------------------------------------------------------------------ cluster access (read only)

def kubectl_json(args: Sequence[str], kubectl: str = "kubectl") -> dict:
    p = subprocess.run([kubectl, *args, "-o", "json"], capture_output=True, text=True, timeout=30)
    if p.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)}: {p.stderr.strip()[:300]}")
    return json.loads(p.stdout)


#: (pod name, pod IP, node, ((container, restartCount, running startedAt), ...))
PodId = Tuple[str, str, str, Tuple[Tuple[str, int, str], ...]]


def _containers(p: dict) -> Tuple[Tuple[str, int, str], ...]:
    return tuple(sorted((str(c.get("name")), int(c.get("restartCount") or 0),
                         str(((c.get("state") or {}).get("running") or {}).get("startedAt") or "not running"))
                        for c in (p.get("status") or {}).get("containerStatuses") or []))


def routable_pods(model: str, namespace: str, kubectl: str = "kubectl") -> List[PodId]:
    """[PodId] of the model's Ready pods the SM marks routable (label ``tre.aibrix.io/routable=true``)."""
    items = kubectl_json(["-n", namespace, "get", "pods", "-l", f"model.aibrix.ai/name={model},{ROUTABLE_LABEL}"],
                         kubectl)["items"]
    out = []
    for p in items:
        conds = {c.get("type"): c.get("status") for c in (p.get("status") or {}).get("conditions") or []}
        if conds.get("Ready") != "True" or (p.get("metadata") or {}).get("deletionTimestamp"):
            continue
        out.append((p["metadata"]["name"], p["status"].get("podIP"), (p.get("spec") or {}).get("nodeName") or "?",
                    _containers(p)))
    return sorted(out)


def pod_awake(ip: str, port: int) -> Optional[bool]:
    """vLLM ``/is_sleeping``: True awake, False asleep, None unknown."""
    try:
        with urlopen(f"http://{ip}:{port}/is_sleeping", timeout=5) as r:
            v = json.loads(r.read().decode("utf-8")).get("is_sleeping")
    except Exception:  # noqa: BLE001 - unknown is not awake
        return None
    return None if not isinstance(v, bool) else not v


def sm_awake(sm_url: str, models: Sequence[str]) -> Dict[str, int]:
    with urlopen(Request(sm_url.rstrip("/") + "/v2/state", headers={"accept": "application/json"}), timeout=10) as r:
        state = json.loads(r.read().decode("utf-8"))
    return {m: int(((state.get("models") or {}).get(m) or {}).get("awake", 0)) for m in models}


def scaler_problems(kubectl: str = "kubectl", target: Tuple[str, str] = BASELINE_SCALER) -> List[str]:
    """The baseline scaler must be at 0 replicas with no pod left (it would scale the models)."""
    ns, name = target
    try:
        d = kubectl_json(["-n", ns, "get", "deploy", name], kubectl)
    except RuntimeError as exc:
        if "NotFound" in str(exc):
            return []
        return [f"baseline scaler unreadable: {exc}"]
    out = []
    if int((d.get("spec") or {}).get("replicas") or 0) != 0:
        out.append(f"{ns}/{name} has {d['spec'].get('replicas')} replicas (must be 0)")
    sel = ((d.get("spec") or {}).get("selector") or {}).get("matchLabels") or {}
    if sel:
        pods = kubectl_json(["-n", ns, "get", "pods", "-l", ",".join(f"{k}={v}" for k, v in sel.items())],
                            kubectl)["items"]
        if pods:
            out.append(f"{ns}/{name} still has pods: {[p['metadata']['name'] for p in pods]}")
    return out


def run_mode_problems(models: Sequence[str], redis_get: Callable[[str], Optional[str]],
                      list_apa: Callable[[], list]) -> List[str]:
    from tre_baselines.tools.tokenscale_profile import check_preconditions
    return check_preconditions(models, redis_get, list_apa)


def check_preconditions(models: Sequence[str], sm_url: str, namespace: str, port: int, redis_get: Callable,
                        list_apa: Callable, kubectl: str = "kubectl") -> Dict[str, Dict[str, Any]]:
    """model -> {pod, ip, node, containers} of its single awake, routable pod; ProfileAbort otherwise."""
    problems = run_mode_problems(models, redis_get, list_apa) + scaler_problems(kubectl)
    awake = sm_awake(sm_url, models)
    fleets = {}
    for m in models:
        if awake.get(m) != 1:
            problems.append(f"{m}: {awake.get(m)} awake replicas per the SM (need exactly 1)")
        pods = routable_pods(m, namespace, kubectl)
        if len(pods) != 1:
            problems.append(f"{m}: {len(pods)} routable pods (need exactly 1): {pods}")
            continue
        name, ip, node, containers = pods[0]
        if pod_awake(ip, port) is not True:
            problems.append(f"{m}: routable pod {name} does not report awake (/is_sleeping)")
        fleets[m] = {"pod": name, "ip": ip, "node": node, "containers": [list(c) for c in containers]}
    if problems:
        raise ProfileAbort("preconditions not met:\n  " + "\n  ".join(problems))
    return fleets


def check_marker(path: Optional[str], tag: Optional[str] = None, now: Optional[float] = None
                 ) -> Tuple[Optional[str], str]:
    """(error or None, content) of the exclusive-window marker: one line
    ``calibration <start ISO> <end ISO> [note ...]``; both times with a time zone; now within
    start..end; ``tag`` = one whitespace-separated token of the note (exact match)."""
    if not path or not Path(path).is_file():
        return f"marker file {path or '(unset)'} missing", ""
    content = Path(path).read_text(encoding="utf-8").strip()
    parts = content.split()
    if len(parts) < 3 or parts[0] != "calibration":
        return f"marker {path}: expected 'calibration <start ISO> <end ISO>', got {content!r}", content
    try:
        start, end = (_dt.datetime.fromisoformat(x.replace("Z", "+00:00")) for x in parts[1:3])
    except ValueError:
        return f"marker {path}: start / end {parts[1:3]!r} not ISO 8601", content
    if start.tzinfo is None or end.tzinfo is None:
        return f"marker {path}: start / end {parts[1:3]!r} need a time zone (e.g. +08:00)", content
    t = time.time() if now is None else now
    if end.timestamp() < t:
        return f"marker {path}: the window ended at {parts[2]}", content
    if start.timestamp() > t:
        return f"marker {path}: the window starts at {parts[1]}", content
    if tag and tag not in parts[3:]:
        return f"marker {path}: not this window (note must have the token {tag!r}): {content!r}", content
    return None, content


def marker_identity(content: str) -> Tuple[str, ...]:
    """What must stay the same during a run: type, start and note (the end may be extended)."""
    parts = content.split()
    return tuple(parts[:2] + parts[3:])


# ------------------------------------------------------------------ real measurement

def build_prompts(model: str, n_tokens: int, keys: Sequence[str], processes: int) -> List[str]:
    """Natural prompts of exactly ``n_tokens`` engine prompt tokens with the chat template, one per key
    (the campaign's prompt builder, off the send path)."""
    from tre_replayer.engine.prompt_store import PromptSpec
    from tre_replayer.engine.prompt_store import build_prompts as _build

    specs = [PromptSpec(request_id=k, model=model, token_count=int(n_tokens), seed_key=f"bl-vel|{model}|{k}")
             for k in keys]
    got = _build(specs, mode="natural", processes=processes, api="chat")
    return [got[k] for k in keys]


def send_stamp_transport(httpx_mod: Any, sent: Dict[str, float], **kw: Any) -> Any:
    """httpx transport that records when each request's headers reached the socket (httpcore trace
    ``http11.send_request_headers.complete``), keyed by the ``x-ts-k`` header."""

    class SendStamp(httpx_mod.AsyncHTTPTransport):
        async def handle_async_request(self, request: Any) -> Any:
            k = request.headers.get("x-ts-k")

            async def trace(name: str, info: Dict[str, Any]) -> None:
                if k is not None and name == "http11.send_request_headers.complete" and k not in sent:
                    sent[k] = time.time()

            request.extensions = {**request.extensions, "trace": trace}
            return await super().handle_async_request(request)

    return SendStamp(**kw)


MAX_SCRAPE_S = 0.5
SAMPLE_FIELDS = (("running", RUNNING_GAUGE), ("waiting", WAITING_GAUGE), ("kv", KV_GAUGE),
                 ("prompt", PROMPT_COUNTER), ("gen", GEN_COUNTER), ("preempt", PREEMPT_COUNTER))


def parse_sample(text: str) -> Dict[str, Optional[float]]:
    return {k: counter_sum(text, name) for k, name in SAMPLE_FIELDS}


def http_text(url: str, timeout_s: float = 5.0) -> str:
    with urlopen(url, timeout=timeout_s) as r:
        return r.read().decode("utf-8", "replace")


def die_with_parent(watchdog: bool = True) -> None:
    """End this (forked) process when its parent dies (PR_SET_PDEATHSIG + a watchdog thread)."""
    import signal
    ppid = os.getppid()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, int(signal.SIGTERM))  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 - not Linux: the watchdog alone
        pass
    if not watchdog:
        return

    def watch() -> None:
        while os.getppid() == ppid:
            time.sleep(0.5)
        os._exit(EXIT_ABORTED)
    threading.Thread(target=watch, name="parent-watchdog", daemon=True).start()


def sample_loop(url: str, out_path: str, stop: Any, *, period_s: float = 1.0, max_scrape_s: float = MAX_SCRAPE_S,
                cpu: Optional[int] = None, fetch: Callable[[str], str] = http_text) -> None:
    """The /metrics sampler, its own process: every ``period_s`` one scrape stamped at its midpoint,
    one JSON line with its duration and the parsed fields; a slow scrape is retried (twice) and kept
    with ``ok: false``; a failed one with ``error``. Ends when ``stop`` is set or the parent is gone."""
    ppid = os.getppid()
    die_with_parent(watchdog=False)
    if cpu is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
    with open(out_path, "w", encoding="utf-8") as fh:
        while not stop.is_set() and os.getppid() == ppid:
            for attempt in range(3):
                a0 = time.time()
                try:
                    text = fetch(url)
                except Exception as e:  # noqa: BLE001 - a gap, recorded
                    fh.write(json.dumps({"t": a0, "dur": round(time.time() - a0, 4), "ok": False,
                                         "attempt": attempt, "error": str(e)[:120]}) + "\n")
                    break
                a1 = time.time()
                ok = a1 - a0 <= max_scrape_s
                fh.write(json.dumps({"t": (a0 + a1) / 2.0, "dur": round(a1 - a0, 4), "ok": ok, "attempt": attempt,
                                     **parse_sample(text)}) + "\n")
                if ok:
                    break
            fh.flush()
            stop.wait(period_s)


def read_samples(path: str, ok: bool = True) -> List[Dict[str, Any]]:
    """The usable (``ok``) samples of a sampler file in time order; ``ok=False``: the unused ones."""
    out = []
    p = Path(path)
    for line in (p.read_text(encoding="utf-8", errors="replace").splitlines() if p.exists() else []):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            if not isinstance(d, dict) or not isinstance(d.get("t"), (int, float)):
                raise ValueError("no time")
        except ValueError:
            d = {"t": None, "ok": False, "corrupt": line[:80]}
        if bool(d.get("ok")) == ok:
            out.append(d)
    return sorted(out, key=lambda d: (d["t"] is None, d["t"] or 0.0))


def counters_monotonic(samples: Sequence[Dict[str, Any]]) -> bool:
    """False when a token counter fell between two samples (the engine restarted)."""
    for key in ("prompt", "gen"):
        vals = [d[key] for d in samples if d.get(key) is not None]
        if any(b < a for a, b in zip(vals, vals[1:])):
            return False
    return True


def window_stats(samples: Sequence[Dict[str, Any]], lo: float, hi: float) -> Dict[str, Any]:
    """Over the usable samples in [lo, hi]: count, waiting slope (judged, D53) and max, running / KV
    median and p90 (recorded only)."""
    win = [d for d in samples if lo <= d["t"] <= hi]
    waits = [(d["t"] - lo, d["waiting"]) for d in win if d.get("waiting") is not None]
    out: Dict[str, Any] = {"window_samples": len(win), "waiting_slope": slope(waits),
                           "waiting_max": max((w for _, w in waits), default=None)}
    for key in ("running", "kv"):
        vals = [d[key] for d in win if d.get(key) is not None]
        out[f"{key}_p50"], out[f"{key}_p90"] = quantile(vals, 0.5), quantile(vals, 0.9)
    return out


def fill_step(st: Step, records: List[Dict[str, Any]], samples: List[Dict[str, Any]], t0: float, lo: float,
              hi: float, sent: Dict[str, float]) -> None:
    """Everything a step reports, from its request records and /metrics samples (pure)."""
    st.window_lo, st.window_hi, st.window_s = lo, hi, hi - lo

    def nearest(target: float) -> Optional[Dict[str, Any]]:
        best = min(samples, key=lambda d: abs(d["t"] - target), default=None)
        return best if best is not None and abs(best["t"] - target) <= 2.5 else None

    for r in records:
        r["sent"] = sent.get(str(r["k"]))
    lags = sorted(r["sent"] - r["planned"] for r in records if r.get("sent") is not None)
    st.send_p99_lag_s = round(quantile(lags, 0.99), 3) if lags else None
    st.send_late = st.send_p99_lag_s is not None and st.send_p99_lag_s > 1.0
    a, b = nearest(lo), nearest(hi)
    if not counters_monotonic(samples):
        st.valid, st.note = False, "token counters fell during the step: the engine restarted"
    if a and b and b["t"] > a["t"]:
        dt = b["t"] - a["t"]
        for key, attr in (("prompt", "prefill_tok_s"), ("gen", "decode_tok_s")):
            x, y = a.get(key), b.get(key)
            setattr(st, attr, None if x is None or y is None or y < x else (y - x) / dt)
        pa, pb = a.get("preempt"), b.get("preempt")
        st.preemptions = None if pa is None or pb is None or pb < pa else pb - pa
    for k, v in window_stats(samples, lo, hi).items():
        setattr(st, k, v)
    win = [r for r in records if lo <= r["send"] <= hi]
    ok = [r for r in win if r["ok"]]
    st.arrived, st.completed, st.failed = len(win), len(ok), len(win) - len(ok)
    st.max_latency_s = round(max((r["end"] - r["send"] for r in records), default=0.0), 3)
    errs: Dict[str, int] = {}
    for r in records:
        if not r["ok"]:
            key = str(r.get("error", "?"))[:120]
            errs[key] = errs.get(key, 0) + 1
    st.errors = errs
    sw = [r for r in win if r["stream"]]
    ttft = [((r["first"] or r["done"]) - r["send"]) * 1000 if r["ok"] else math.inf for r in sw]
    tpot = [(r["end"] - r["first"]) * 1000 / r["completion_tokens"]
            if r["ok"] and r["first"] and r.get("completion_tokens") else math.inf for r in sw]
    st.stream_n = len(sw)
    st.ttft_p50_ms, st.ttft_p95_ms = (None if v is None else round(v, 1) for v in
                                      (quantile(ttft, 0.5), quantile(ttft, 0.95)))
    st.tpot_p50_ms, st.tpot_p95_ms = (None if v is None else round(v, 2) for v in
                                      (quantile(tpot, 0.5), quantile(tpot, 0.95)))
    lates = sorted(r["late"] for r in records)
    st.sender_p99_late_s = round(quantile(lates, 0.99), 3) if lates else None
    if st.sender_p99_late_s is not None and st.sender_p99_late_s > 1.0:
        st.sender_late = True
        st.note = f"sender late (p99 {st.sender_p99_late_s:.2f} s): offered rate not met"
    pts = [r["prompt_tokens"] for r in ok if r.get("prompt_tokens") is not None]
    st.mean_prompt_tokens = round(sum(pts) / len(pts), 1) if pts else None
    cts = [r["completion_tokens"] for r in ok if r.get("completion_tokens") is not None]
    st.mean_completion_tokens = round(sum(cts) / len(cts), 1) if cts else None


def make_measure(model: str, fleet: Dict[str, Any], namespace: str, *, port: int, step_s: float, measure_s: float,
                 max_num_seqs: int, lifetimes: float, seed: int, drain_cap_s: float, raw_dir: Path,
                 prompt_processes: int = 4, stream_frac: float = DEFAULT_STREAM_FRAC, kubectl: str = "kubectl",
                 sampler_cpu: Optional[int] = None) -> Measure:
    import httpx
    import openai

    expected = [(fleet["pod"], fleet["ip"], fleet["node"], tuple(tuple(c) for c in fleet["containers"]))]
    base = f"http://{fleet['ip']}:{port}"

    def scrape() -> str:
        return http_text(f"{base}/metrics")

    def same_awake_pod() -> Tuple[bool, str]:
        pods = routable_pods(model, namespace, kubectl)
        if pods != expected:
            return False, f"routable pods {pods}, expected {expected}"
        if pod_awake(fleet["ip"], port) is not True:
            return False, f"pod {fleet['pod']} not reporting awake"
        return True, ""

    def measure(model_: str, cell: str, in_tokens: int, out_tokens: int, rate_rps: float) -> Step:
        length = step_length(rate_rps, step_s, measure_s, max_num_seqs, lifetimes)
        st = Step(model, "bucket", cell, in_tokens, out_tokens, rate_rps, step_s=round(length, 1),
                  pod=fleet["pod"], node=fleet["node"])
        ok_before, why_before = same_awake_pod()
        rng = random.Random(f"{seed}:{model}:{cell}:{rate_rps:.4f}")
        sched, t = [], rng.expovariate(rate_rps)
        while t < length:
            sched.append(t)
            t += rng.expovariate(rate_rps)
        streamed = [rng.random() < stream_frac for _ in sched]
        tag = f"{cell}-{rate_rps:.3f}"
        prompts = build_prompts(model, in_tokens, [f"{tag}-{k}" for k in range(len(sched))], prompt_processes)
        records: List[Dict[str, Any]] = []
        sent: Dict[str, float] = {}
        raw_dir.mkdir(parents=True, exist_ok=True)
        sample_path = str(raw_dir / f"{model}_{cell}_{rate_rps:.3f}.metrics.jsonl")
        import multiprocessing
        mp = multiprocessing.get_context("fork")
        stop = mp.Event()
        sampler = mp.Process(target=sample_loop, args=(f"{base}/metrics", sample_path, stop),
                             kwargs={"cpu": sampler_cpu}, name=f"sampler-{model}", daemon=True)

        async def one(client: Any, k: int, t0: float) -> None:
            await asyncio.sleep(max(0.0, t0 + sched[k] - time.time()))
            rec: Dict[str, Any] = {"k": k, "planned": t0 + sched[k], "send": time.time(), "first": None, "done": None,
                                   "end": None, "ok": False, "prompt_tokens": None, "completion_tokens": None,
                                   "stream": streamed[k]}
            rec["late"] = rec["send"] - rec["planned"]
            msgs = [{"role": "user", "content": prompts[k]}]
            hdr = {"x-ts-k": str(k)}
            try:
                if streamed[k]:
                    resp = await client.chat.completions.create(
                        model=model, messages=msgs, max_tokens=out_tokens, temperature=0.0, stream=True,
                        stream_options={"include_usage": True}, extra_body={"ignore_eos": True}, extra_headers=hdr)
                    async for ch in resp:
                        if ch.choices and ch.choices[0].delta.content is not None and rec["first"] is None:
                            rec["first"] = time.time()
                        if getattr(ch, "usage", None) is not None:
                            rec["prompt_tokens"] = ch.usage.prompt_tokens
                            rec["completion_tokens"] = ch.usage.completion_tokens
                else:
                    resp = await client.chat.completions.create(
                        model=model, messages=msgs, max_tokens=out_tokens, temperature=0.0,
                        extra_body={"ignore_eos": True}, extra_headers=hdr)
                    if resp.usage:
                        rec["prompt_tokens"] = resp.usage.prompt_tokens
                        rec["completion_tokens"] = resp.usage.completion_tokens
                rec["done"] = time.time()
                rec["ok"] = True
            except Exception as e:  # noqa: BLE001
                rec["error"] = str(e)[:200]
            rec["end"] = time.time()
            records.append(rec)

        async def drive() -> float:
            clients = [openai.AsyncOpenAI(
                api_key="dummy", base_url=f"{base}/v1", max_retries=0, timeout=600.0,
                http_client=httpx.AsyncClient(timeout=600.0, transport=send_stamp_transport(httpx, sent, limits=httpx.Limits(
                    max_connections=None, max_keepalive_connections=CLIENT_KEEPALIVE,
                    keepalive_expiry=CLIENT_KEEPALIVE_EXPIRY_S))))
                for _ in range(CLIENT_SHARDS)]
            try:
                t0 = time.time() + 1.0
                await asyncio.gather(*(one(clients[k % CLIENT_SHARDS], k, t0) for k in range(len(sched))))
            finally:
                for c in clients:
                    await c.close()
            return t0

        sampler.start()
        try:
            t0 = asyncio.run(drive())
        finally:
            stop.set()
            sampler.join(SAMPLER_JOIN_S)
            if sampler.is_alive():
                sampler.terminate()
                sampler.join(5)
        st.sampler_exit = sampler.exitcode
        samples = read_samples(sample_path)
        hi = t0 + length
        lo = hi - measure_s
        st.samples_bad = sum(1 for d in read_samples(sample_path, ok=False) if d["t"] is None or lo <= d["t"] <= hi)
        fill_step(st, records, samples, t0, lo, hi, sent)
        with open(raw_dir / f"{model}_{cell}_{rate_rps:.3f}.jsonl", "w", encoding="utf-8") as fh:
            for r in sorted(records, key=lambda r: r["k"]):
                fh.write(json.dumps(r) + "\n")
        deadline = time.time() + drain_cap_s
        while True:  # drain: the next step must not inherit this one's queue
            try:
                txt = scrape()
                q = (counter_sum(txt, RUNNING_GAUGE) or 0) + (counter_sum(txt, WAITING_GAUGE) or 0)
            except Exception:  # noqa: BLE001
                q = None
            if q == 0:
                break
            if time.time() > deadline:
                raise ProfileAbort(f"{model}: pod not idle {drain_cap_s:g}s after the step ({q} in flight)")
            time.sleep(1.0)
        ok_after, why_after = same_awake_pod()
        if not (ok_before and ok_after):
            st.valid = False
            st.note = f"fleet changed: before: {why_before or 'ok'}; after: {why_after or 'ok'}"
        return st

    return measure


# ------------------------------------------------------------------ isolated NL

def isolated_plan(shapes: Sequence[Tuple[str, int, int]], runs: int, warmup: int) -> List[Tuple[str, int, int, int]]:
    """Warm-up requests (run -1, the first shape) then ``runs`` rounds over every shape."""
    plan_ = [(shapes[0][0], shapes[0][1], shapes[0][2], -1) for _ in range(warmup)]
    for r in range(runs):
        plan_ += [(s[0], s[1], s[2], r) for s in shapes]
    return plan_


def run_isolated_model(model: str, fleet: Dict[str, Any], port: int, shapes: Sequence[Tuple[str, int, int]],
                       runs: int, warmup: int, out_path: Path, prompt_processes: int, gap_s: float = 0.5) -> int:
    """One request at a time to the model's pod, non-streaming, ``ignore_eos``; the pod must report
    nothing running or waiting before each. One JSON line per request."""
    import httpx

    base = f"http://{fleet['ip']}:{port}"
    plan_ = isolated_plan(shapes, runs, warmup)
    keys = [f"iso-{name}-{run}-{k}" for k, (name, _i, _o, run) in enumerate(plan_)]
    prompts: Dict[str, str] = {}
    for tin in sorted({p[1] for p in plan_}):
        ks = [keys[k] for k, p in enumerate(plan_) if p[1] == tin]
        prompts.update(zip(ks, build_prompts(model, tin, ks, prompt_processes)))
    client = httpx.Client(timeout=600.0, limits=httpx.Limits(keepalive_expiry=CLIENT_KEEPALIVE_EXPIRY_S))
    bad = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for k, (name, tin, tout, run) in enumerate(plan_):
            txt = http_text(f"{base}/metrics", 10)
            q = (counter_sum(txt, RUNNING_GAUGE) or 0) + (counter_sum(txt, WAITING_GAUGE) or 0)
            body = {"model": model, "messages": [{"role": "user", "content": prompts[keys[k]]}],
                    "max_tokens": tout, "temperature": 0.0, "ignore_eos": True}
            rec: Dict[str, Any] = {"model": model, "pod": fleet["pod"], "shape": name, "in": tin, "out": tout,
                                   "run": run, "queue_before": q, "send": time.time()}
            try:
                resp = client.post(f"{base}/v1/chat/completions", json=body)
                rec["done"] = time.time()
                rec["status"] = resp.status_code
                j = resp.json()
                u = j.get("usage") or {}
                rec["prompt_tokens"], rec["completion_tokens"] = u.get("prompt_tokens"), u.get("completion_tokens")
                rec["ok"] = resp.status_code == 200
            except Exception as e:  # noqa: BLE001
                rec["done"], rec["ok"], rec["error"] = time.time(), False, str(e)[:200]
            rec["e2e_s"] = rec["done"] - rec["send"]
            if rec.get("completion_tokens"):
                rec["norm_lat_s"] = rec["e2e_s"] / rec["completion_tokens"]
            if run >= 0 and (not rec["ok"] or q or rec.get("completion_tokens") != tout or rec.get("prompt_tokens") != tin):
                bad += 1
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            time.sleep(gap_s)
    return bad


# ------------------------------------------------------------------ CLI helpers

def steps_from_log(path: Path) -> Dict[str, List[Step]]:
    """model -> the ``event: step`` lines of a step log, as Steps."""
    names = set(Step.__dataclass_fields__)
    out: Dict[str, List[Step]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line) if line.strip() else {}
        if d.get("event") == "step":
            out.setdefault(d["model"], []).append(Step(**{k: v for k, v in d.items() if k in names}))
    return out


def _floats(text: str) -> Tuple[float, ...]:
    vals = tuple(float(x) for x in text.split(",") if x.strip())
    if not vals or any(not v > 0 for v in vals):
        raise argparse.ArgumentTypeError("factors must be > 0")
    return vals


def _anchors(text: str) -> Dict[str, Tuple[int, int, float]]:
    out = {}
    for part in filter(None, (x.strip() for x in text.split(","))):
        m, sep, v = part.partition("=")
        try:
            i, o, r = v.split(":")
            out[m] = (int(i), int(o), float(r))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"--anchor {part!r} is not model=in:out:rps") from exc
        if not sep or not m:
            raise argparse.ArgumentTypeError(f"--anchor {part!r} is not model=in:out:rps")
    return out


def _cpus(text: Optional[str], models: Sequence[str]) -> Dict[str, List[int]]:
    """``--cpus model=a-b,model2=c-d`` -> per model CPU list (its last CPU runs the sampler)."""
    if not text:
        return {}
    out: Dict[str, List[int]] = {}
    for part in filter(None, (x.strip() for x in text.split(","))):
        m, _, rng = part.partition("=")
        lo, _, hi = rng.partition("-")
        out[m] = list(range(int(lo), int(hi or lo) + 1))
    missing = [m for m in models if m not in out or len(out[m]) < 2]
    if missing:
        raise ValueError(f"--cpus needs >= 2 CPUs for {missing}")
    return out


def load_report(path: str) -> Dict[str, Any]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(doc.get("bucket_centers"), dict):
        raise ProfileAbort(f"{path}: no bucket_centers (write it with tools/tokenscale_buckets.py)")
    return doc


def _sha256(path: str) -> str:
    from tre_baselines.tools.tokenscale_buckets import sha256_file
    return sha256_file(path)


def _git_sha() -> Optional[str]:
    try:
        here = Path(__file__).resolve().parent
        sha = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", "."], capture_output=True,
                               text=True, timeout=10).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" if sha else None
    except Exception:  # noqa: BLE001
        return None


def _cluster_readers(a: argparse.Namespace) -> Tuple[Callable, Callable]:
    from tre_baselines.tools import arm
    redis = (arm.DirectRedis(a.redis_url) if a.redis_url else
             arm.KubectlRedis(arm.subprocess_runner, a.kubectl, "tre-v2", "tre-v2-redis"))

    def list_apa() -> list:
        p = subprocess.run([a.kubectl, "get", arm.APA_RESOURCE, "-A", "-o", "json"], capture_output=True, text=True,
                           check=True, timeout=30)
        return (json.loads(p.stdout or "{}")).get("items") or []

    return redis.get, list_apa


def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--models", default="dsqwen-7b,dsllama-8b,dsqwen-14b")
    ap.add_argument("--report", required=True, help="buckets report (tools/tokenscale_buckets.py)")
    ap.add_argument("--port", type=int, default=8000, help="vLLM port of the model pods (API and /metrics)")
    ap.add_argument("--sm-url", default=os.environ.get("TRE_SM_URL"), help="SM base URL (GET /v2/state)")
    ap.add_argument("--redis-url", default=None, help="run modes here (default: kubectl exec redis-cli)")
    ap.add_argument("--namespace", default="default", help="namespace of the model pods")
    ap.add_argument("--kubectl", default="kubectl")
    ap.add_argument("--exclusive-marker", default=os.environ.get(ENV_EXCLUSIVE_MARKER, DEFAULT_MARKER))
    ap.add_argument("--marker-tag", default="baselines", help="a token the marker's note must have (exact)")
    ap.add_argument("--prompt-processes", type=int, default=4, help="prompt builder processes per model")
    ap.add_argument("--out-dir", default=None, help="required for a real run")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; nothing is sent")
    ap.add_argument("--i-have-user-approval", action="store_true", help="required to send real requests")


def _run_parallel(models: Sequence[str], one: Callable[[str], Any]) -> Dict[str, Tuple[str, Any]]:
    from tre_baselines.tools.tokenscale_profile import run_isolated
    return run_isolated(models, one)


def ladder_main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="velocity_ladder ladder", description="TokenScale V_b, direct to the pods")
    _common(ap)
    ap.add_argument("--fit", required=True, help="service-time fit json {model: [a, b, c]} (1/rps = a*in + b*out + c)")
    ap.add_argument("--anchor", type=_anchors, default={},
                    help="model=in:out:rps,... a rate known on THIS engine at that shape; scales the fit")
    ap.add_argument("--factors", type=_floats, default=DEFAULT_FACTORS)
    ap.add_argument("--down-factors", type=_floats, default=DEFAULT_DOWN_FACTORS)
    ap.add_argument("--extend-factor", type=float, default=1.1)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--cells", default=None, help="comma list of <cell> or <model>:<cell> (default all 27)")
    ap.add_argument("--step-s", type=float, default=110.0)
    ap.add_argument("--measure-s", type=float, default=60.0, help="statistics over the last this many s of a step")
    ap.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS, help="engine --max-num-seqs")
    ap.add_argument("--warmup-lifetimes", type=float, default=1.5)
    ap.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    ap.add_argument("--min-completed", type=int, default=150)
    ap.add_argument("--knee-slope-frac", type=float, default=DEFAULT_KNEE_SLOPE_FRAC)
    ap.add_argument("--route-timeout-s", type=float, default=DEFAULT_ROUTE_TIMEOUT_S)
    ap.add_argument("--max-fail-frac", type=float, default=DEFAULT_MAX_FAIL_FRAC)
    ap.add_argument("--drain-cap-s", type=float, default=600.0)
    ap.add_argument("--est-overhead-s", type=float, default=15.0, help="plan only: per-step overhead (prompts, checks)")
    ap.add_argument("--cpus", default=None, help="model=a-b,... CPUs per model process (the last one: sampler)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--stream-frac", type=float, default=DEFAULT_STREAM_FRAC)
    a = ap.parse_args(argv)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    try:
        rep = load_report(a.report)
        centers = {m: rep["bucket_centers"][m] for m in models}
        fits_doc = json.loads(Path(a.fit).read_text(encoding="utf-8"))
        fits = {m: [float(x) for x in fits_doc[m]] for m in models}
        scales = {m: anchor_scale(fits[m], *a.anchor[m]) if m in a.anchor else 1.0 for m in models}
        cells = parse_cells(models, a.cells)
        cpus = _cpus(a.cpus, models)
    except (ProfileAbort, KeyError, ValueError, OSError) as exc:
        print(f"error: {exc!r}", file=sys.stderr)
        return 2
    pl = plan(models, centers, fits, scales, a.factors, extend_factor=a.extend_factor, max_steps=a.max_steps,
              step_s=a.step_s, measure_s=a.measure_s, max_num_seqs=a.max_num_seqs, lifetimes=a.warmup_lifetimes,
              overhead_s=a.est_overhead_s, down_factors=a.down_factors, cells=cells)
    print(yaml.safe_dump({"plan": pl}, sort_keys=False, default_flow_style=None))
    w = pl["wall_s"]
    print(f"estimated wall time (models in parallel): min {w['min'] / 60:.0f} / nominal {w['nominal'] / 60:.0f} / "
          f"max {w['max'] / 60:.0f} min", file=sys.stderr)
    if a.dry_run:
        return 0
    if not a.i_have_user_approval:
        print("refusing to send requests without --i-have-user-approval (use --dry-run)", file=sys.stderr)
        return 3
    if not a.out_dir or not a.sm_url:
        print("error: a real run needs --out-dir and --sm-url", file=sys.stderr)
        return 2
    err, marker = check_marker(a.exclusive_marker, a.marker_tag)
    if err:
        print(f"error: exclusive window: {err}", file=sys.stderr)
        return 4
    redis_get, list_apa = _cluster_readers(a)
    try:
        fleets = check_preconditions(models, a.sm_url, a.namespace, a.port, redis_get, list_apa, a.kubectl)
    except ProfileAbort as exc:
        print(f"refusing to profile: {exc}", file=sys.stderr)
        return 4
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rules = Rules(max_fail_frac=a.max_fail_frac, route_timeout_s=a.route_timeout_s, min_completed=a.min_completed,
                  min_samples=a.min_samples, knee_slope_frac=a.knee_slope_frac)
    ident = marker_identity(marker)
    meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tool_sha": _git_sha(), "argv": argv,
            "report": a.report, "report_sha256": _sha256(a.report), "fit": a.fit, "fit_sha256": _sha256(a.fit),
            "scales": scales, "fleets": fleets, "marker": marker, "rules": asdict(rules), "plan": pl}
    (out / "run-meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")

    def one(model: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        die_with_parent()
        mine = cpus.get(model)
        if mine and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(mine[:-1]))
        logf = open(out / f"profile-{model}.log.jsonl", "a", encoding="utf-8")

        def log(line: str) -> None:
            logf.write(line + "\n")
            logf.flush()

        def check() -> None:
            e, content = check_marker(a.exclusive_marker, a.marker_tag)
            if e or marker_identity(content) != ident:
                raise ProfileAbort(f"exclusive window changed or ended: {e or content!r}")
            probs = run_mode_problems([model], redis_get, list_apa) + scaler_problems(a.kubectl)
            if probs:
                raise ProfileAbort("; ".join(probs))

        measure = make_measure(model, fleets[model], a.namespace, port=a.port, step_s=a.step_s,
                               measure_s=a.measure_s, max_num_seqs=a.max_num_seqs, lifetimes=a.warmup_lifetimes,
                               seed=a.seed, drain_cap_s=a.drain_cap_s, raw_dir=out / "raw",
                               prompt_processes=a.prompt_processes, stream_frac=a.stream_frac, kubectl=a.kubectl,
                               sampler_cpu=mine[-1] if mine else None)
        try:
            res, steps = profile_model(measure, model, centers[model], fits[model], scales[model], a.factors,
                                       extend_factor=a.extend_factor, max_steps=a.max_steps, rules=rules, log=log,
                                       check=check, down_factors=a.down_factors, cells=cells[model])
        finally:
            logf.close()
        return res, [{**asdict(s), "tok_s": s.tok_s} for s in steps]

    outcomes = _run_parallel(models, one)
    results = {m: p for m, (st, p) in outcomes.items() if st == "ok"}
    errors = {m: p for m, (st, p) in outcomes.items() if st != "ok"}
    doc = {"velocity": {m: r["velocity"] for m, (r, _) in results.items()},
           "peaks": {m: r["peaks"] for m, (r, _) in results.items()},
           "knees": {m: r["knees"] for m, (r, _) in results.items()},
           "slo_capacity": {m: r["slo_capacity"] for m, (r, _) in results.items()},
           "not_saturated": {m: r["not_saturated"] for m, (r, _) in results.items() if r["not_saturated"]},
           "aborted": errors, "provenance": {k: meta[k] for k in ("started", "tool_sha", "report_sha256",
                                                                  "fit_sha256", "scales", "fleets", "rules")}}
    (out / "profile.yaml").write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=None), encoding="utf-8")
    rows = [row for _, (_, rs) in results.items() for row in rs]
    if rows:
        with (out / "profile_raw.csv").open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()} for r in rows])
    print(f"wrote {out / 'profile.yaml'}")
    if errors:
        print(f"error: aborted: {errors}", file=sys.stderr)
        return EXIT_ABORTED
    if doc["not_saturated"]:
        print(f"warning: cells not saturated: {doc['not_saturated']}", file=sys.stderr)
    return 0


def summarize_main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="velocity_ladder summarize", description="V / knees again from step logs")
    ap.add_argument("--log", action="append", required=True, help="step log (repeatable)")
    ap.add_argument("--min-completed", type=int, default=150)
    ap.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    ap.add_argument("--knee-slope-frac", type=float, default=DEFAULT_KNEE_SLOPE_FRAC)
    ap.add_argument("--route-timeout-s", type=float, default=DEFAULT_ROUTE_TIMEOUT_S)
    ap.add_argument("--max-fail-frac", type=float, default=DEFAULT_MAX_FAIL_FRAC)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rules = Rules(max_fail_frac=a.max_fail_frac, route_timeout_s=a.route_timeout_s, min_completed=a.min_completed,
                  min_samples=a.min_samples, knee_slope_frac=a.knee_slope_frac)
    steps: Dict[str, List[Step]] = {}
    for p in a.log:
        for m, s in steps_from_log(Path(p)).items():
            steps.setdefault(m, []).extend(s)
    res = {m: summarize(s, rules) for m, s in sorted(steps.items())}
    doc = {"velocity": {m: r["velocity"] for m, r in res.items()}, "peaks": {m: r["peaks"] for m, r in res.items()},
           "knees": {m: r["knees"] for m, r in res.items()},
           "not_saturated": {m: r["not_saturated"] for m, r in res.items() if r["not_saturated"]},
           "logs": {p: _sha256(p) for p in a.log}}
    Path(a.out).write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=None), encoding="utf-8")
    print(yaml.safe_dump(doc["velocity"], default_flow_style=None))
    return 0


def isolated_main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="velocity_ladder isolated", description="isolated normalized latency per model")
    _common(ap)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--gap-s", type=float, default=0.5)
    a = ap.parse_args(argv)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    try:
        rep = load_report(a.report)
        shapes = {m: [("mix", int(rep["mix_shape"][m]["in"]), int(rep["mix_shape"][m]["out"]))] +
                  [(f"b{i}{j}", int(c[0]), int(c[1])) for i, row in enumerate(rep["bucket_centers"][m])
                   for j, c in enumerate(row)] for m in models}
    except (ProfileAbort, KeyError, TypeError) as exc:
        print(f"error: {exc!r}", file=sys.stderr)
        return 2
    n = {m: len(isolated_plan(shapes[m], a.runs, a.warmup)) for m in models}
    print(json.dumps({"requests": n, "shapes": shapes}), file=sys.stderr)
    if a.dry_run:
        return 0
    if not a.i_have_user_approval:
        print("refusing to send requests without --i-have-user-approval (use --dry-run)", file=sys.stderr)
        return 3
    if not a.out_dir or not a.sm_url:
        print("error: a real run needs --out-dir and --sm-url", file=sys.stderr)
        return 2
    err, _marker = check_marker(a.exclusive_marker, a.marker_tag)
    if err:
        print(f"error: exclusive window: {err}", file=sys.stderr)
        return 4
    redis_get, list_apa = _cluster_readers(a)
    try:
        fleets = check_preconditions(models, a.sm_url, a.namespace, a.port, redis_get, list_apa, a.kubectl)
    except ProfileAbort as exc:
        print(f"refusing to measure: {exc}", file=sys.stderr)
        return 4
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "isolated-meta.json").write_text(json.dumps({"tool_sha": _git_sha(), "argv": argv, "fleets": fleets,
                                                         "report_sha256": _sha256(a.report), "shapes": shapes},
                                                        indent=1), encoding="utf-8")

    def one(model: str) -> int:
        die_with_parent()
        return run_isolated_model(model, fleets[model], a.port, shapes[model], a.runs, a.warmup,
                                  out / f"{model}.jsonl", a.prompt_processes, a.gap_s)

    outcomes = _run_parallel(models, one)
    print(json.dumps({m: {"status": st, "bad_records": p} for m, (st, p) in outcomes.items()}))
    after = check_preconditions(models, a.sm_url, a.namespace, a.port, redis_get, list_apa, a.kubectl)
    if after != fleets:
        print(f"error: the pods changed during the run: {fleets} -> {after}", file=sys.stderr)
        return EXIT_ABORTED
    return 0 if all(st == "ok" and p == 0 for st, p in outcomes.values()) else EXIT_ABORTED


def mu_main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="velocity_ladder mu", description="PreServe mu by normalized latency (D58)")
    ap.add_argument("--ladder-dir", required=True, help="out-dir of the ladder run (step logs + raw/)")
    ap.add_argument("--isolated-dir", required=True, help="out-dir of the isolated run")
    ap.add_argument("--report", required=True, help="buckets report (p_share range)")
    ap.add_argument("--models", default="dsqwen-7b,dsllama-8b,dsqwen-14b")
    ap.add_argument("--slo-factor", type=float, default=3.0, help="SLO = this x isolated NL of the mix shape")
    ap.add_argument("--min-completed", type=int, default=150)
    ap.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    ap.add_argument("--route-timeout-s", type=float, default=DEFAULT_ROUTE_TIMEOUT_S)
    ap.add_argument("--max-fail-frac", type=float, default=DEFAULT_MAX_FAIL_FRAC)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    rules = Rules(max_fail_frac=a.max_fail_frac, route_timeout_s=a.route_timeout_s, min_completed=a.min_completed,
                  min_samples=a.min_samples)
    rep = load_report(a.report)
    pooled = (rep.get("p_share") or {}).get("pooled") or {}
    lo, hi = pooled.get("min"), pooled.get("max")
    if lo is None or hi is None:
        print("error: the report has no p_share.pooled range", file=sys.stderr)
        return 2
    ladder, iso = Path(a.ladder_dir), Path(a.isolated_dir)
    summary: Dict[str, Any] = {"p_share_range": [lo, hi], "slo_factor": a.slo_factor, "models": {}}
    all_rows: List[Dict[str, Any]] = []
    mu_b: Dict[str, Any] = {}
    for m in models:
        recs = [json.loads(x) for x in (iso / f"{m}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        nl, bad = isolated_nl(recs)
        if "mix" not in nl:
            print(f"error: {m}: no clean isolated mix-shape run", file=sys.stderr)
            return 2
        slo_s = a.slo_factor * nl["mix"]["mean"]
        steps = steps_from_log(ladder / f"profile-{m}.log.jsonl").get(m, [])
        judged = []
        for s in steps:
            raw = ladder / "raw" / f"{m}_{s.cell}_{s.rate_rps:.3f}.jsonl"
            rr = [json.loads(x) for x in raw.read_text(encoding="utf-8").splitlines() if x.strip()]
            judged.append(judge_step(s, rr, slo_s, rules))
        all_rows += judged
        mu_a = mu_from(judged, lambda s: True)
        mb = mu_from(judged, lambda s: lo <= s["p_share"] <= hi)
        summary["models"][m] = {"isolated_nl_ms": {k: {"mean": round(1000 * v["mean"], 3), "sd": round(1000 * v["sd"], 3),
                                                       "n": v["n"]} for k, v in nl.items()},
                                "isolated_bad_records": bad, "slo_ms": round(1000 * slo_s, 3),
                                "mu": {"A": mu_a, "B": mb}}
        if all(mb.get(k) for k in ("p", "d", "t")):
            mu_b[m] = {k: mb[k]["value"] for k in ("p", "d", "t")}
    summary["mu_B"] = mu_b
    summary["sources"] = {"report_sha256": _sha256(a.report)}
    Path(a.out).write_text(yaml.safe_dump(summary, sort_keys=False, default_flow_style=None), encoding="utf-8")
    if all_rows:
        with Path(a.out).with_suffix(".steps.csv").open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
            wr.writeheader()
            wr.writerows(all_rows)
    print(yaml.safe_dump({"mu_B": mu_b}, default_flow_style=None))
    missing = [m for m in models if m not in mu_b]
    if missing:
        print(f"error: no passing step for some term of {missing} (variant B)", file=sys.stderr)
        return EXIT_ABORTED
    return 0


COMMANDS = {"ladder": ladder_main, "summarize": summarize_main, "isolated": isolated_main, "mu": mu_main}


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(f"usage: python3 -m tre_baselines.tools.velocity_ladder {{{'|'.join(COMMANDS)}}} ...", file=sys.stderr)
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

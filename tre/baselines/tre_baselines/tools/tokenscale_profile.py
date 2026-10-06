"""Baseline parameter profiling: TokenScale-colocated V_b / V_P and PreServe-oracle mu in
one run (decision 2026-10-06, "参数测量").

The hot alt trace has one request shape (input ~492, output 400), so TokenScale's 3x3
buckets collapse into one cell: one shape is measured and written to all nine cells
(lookups clamp to the nearest bucket; disclose "buckets degenerate to one cell").

Per model, on its **single awake replica** (``--sm-url`` checks exactly one is awake; the
models run in parallel, each on its own replica), a closed-loop load at each concurrency
of the ladder (default 1, 2, 4, 8, 12, 16, 24, 32, 48, 64; extended by x1.5 steps while
the last step still gains more than ``--extend-gain`` (5 %), up to ``--max-concurrency``,
with a warning if it still gains there). Each step lasts ``--step-s`` (60) of which the
first ``--warmup-s`` (15) are dropped. Requests are the calibration chat sender's
(:func:`scripts.r3_grid.drive_cell`): ``/v1/chat/completions``, a distinct natural prompt of
``in`` tokens after the chat template per request, ``max_tokens = out``, ``ignore_eos``,
temperature 0, streamed with usage. Per step:

* prefill / decode tok/s from the replica's own counters ``vllm:prompt_tokens_total`` /
  ``vllm:generation_tokens_total``, scraped at the end of the warm-up and at the end of
  the step (not from completed requests);
* p95 TTFT and p95 TPOT (client side) of the requests that finished inside the window.

Outputs (``--out-dir``): ``profile.yaml`` and ``profile_raw.csv``:

* TokenScale ``velocity``: ``V_b`` = the highest (prefill + decode) tok/s of any step (a
  saturation indicator; no SLO), ``V_P`` = the highest prefill tok/s of a separate ladder
  with ``out = 1``;
* PreServe ``mu`` (``p`` / ``d`` / ``t``): prefill / decode / total tok/s of the highest
  step whose p95 TTFT <= the model's TTFT SLO at this input length
  (``max(floor, k (c + b L))`` from the registry, the SLO labels use) and p95 TPOT <= its
  TPOT SLO (75 ms). mu_p : mu_d is fixed by the shape (one free quantity; disclose).
  Measured 2026-10-06, the closed-loop mu came out 0.16-0.34 x the calibration rho* x
  token mix (lockstep bursts, see "Open loop" below): measure mu with ``--open-loop``.

Validity gates (state, not timers):

* before sending: the TRE controller and the SM actuation are both ``observe``
  (``tre:v2:controller:mode`` / ``tre:v2:sm:actuation``, missing = observe), no AIBrix APA
  ``PodAutoscaler`` targets a profiled model, each model has exactly one awake replica and
  one routable pod (all read only);
* at both marks of every step the model's awake bindings and routable pods are read
  again; a change (a wake, a sleep, a restart) makes the step invalid (rates None) and
  aborts the model's run (exit 5);
* after every step, before the next one (also between the two ladders), the tool waits
  until the replica reports nothing running or waiting, so a step never inherits the
  previous step's queue (bounded by ``--drain-cap-s``, beyond any request's lifetime).

The SLO and c/b come from ``--registry`` or, by default, the live ConfigMap
``tre-v2/tre-v2-registry`` (read with ``kubectl get``); the source and the c/b used are
written into ``profile.yaml``. Each model's ladders run in their own process.

Open loop (``--open-loop``, decisions 2026-10-06): a closed loop with a
constant output length runs in lockstep (every worker resends at once, so p95 TTFT measures
a burst of ``c`` prefills) and biased mu low. With ``--open-loop`` each step instead offers
Poisson arrivals at ``--base-rps[model] x factor`` (``--rate-factors``, default 0.3 ... 1.3)
through the calibration's open-loop sender (``openloop.drive_cell_schedule``) for
``--step-s`` (110; the first ``--warmup-s`` 20 dropped). Token rates are the engine
counters between the end of the warm-up and the step end; p95 TTFT / TPOT are those of
the requests *sent* in that window (a failed request counts as infinite). mu = the
highest rate passing both SLOs with at least ``--min-completed`` (150) such requests; a
failing rate below a passing one is reported (``non_monotone_rates``: lengthen the steps).
TokenScale velocity at the knee: per step the tool also records the least-squares slope
of the engine's ``vllm:num_requests_waiting`` over the measured window (1 Hz scrapes), its
last / max value, the ``vllm:num_preemptions_total`` delta and the slope of the client-side
queue (sent, no first token). A step's queue is bounded iff that waiting slope is <=
``--knee-slope-frac`` (2 %) x the offered req/s; the knee is the highest-throughput bounded
step (``bracketed`` says whether a higher rate was seen unbounded). V_b = (in+out) tok/s at
the knee of the mixed ladder; V_P = prefill tok/s at the knee of an out=1 ladder
(``--prefill-base-rps`` x ``--prefill-rate-factors``). ``--model-rate-factors`` overrides the
mixed factors per model (``model=`` = none). The closed-loop maxima are a sensitivity row.

Cost: (len(ladder) x 2) x step_s per model, models in parallel (default ~20 min).
Sending needs ``--i-have-user-approval``; ``--dry-run`` uses a synthetic stub.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from urllib.request import Request, urlopen

import yaml

#: Closed-loop concurrency ladder (decision 2026-10-06).
DEFAULT_LADDER = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
DEFAULT_EXTEND_GAIN = 0.05
DEFAULT_MAX_CONCURRENCY = 256
PROMPT_COUNTER = "vllm:prompt_tokens_total"
GEN_COUNTER = "vllm:generation_tokens_total"
#: Upper bound of the wait for an idle replica after a step: no request outlives the
#: gateway route timeout (150 s); twice that means something is wrong -> abort.
DEFAULT_DRAIN_CAP_S = 300.0
CONTROLLER_MODE_KEY = "tre:v2:controller:mode"
SM_ACTUATION_KEY = "tre:v2:sm:actuation"
EXIT_ABORTED = 5


class ProfileAbort(RuntimeError):
    """The measurement cannot go on validly (fleet changed, replica never idle)."""


@dataclass(frozen=True)
class StepMeasure:
    concurrency: int
    window_s: float
    completed: int
    #: Engine counters' rates over the measured window (None when unreadable / reset).
    prefill_tok_s: Optional[float]
    decode_tok_s: Optional[float]
    #: Client-side p95 of the requests that finished inside the window (ms).
    ttft_p95_ms: Optional[float]
    tpot_p95_ms: Optional[float]
    #: False when the fleet changed during the step (its rates are then None).
    valid: bool = True
    note: Optional[str] = None
    #: Open loop only: the offered Poisson rate, the requests that arrived inside the
    #: measured window (their achieved rate) and how many of those did not complete.
    rate_rps: Optional[float] = None
    arrived: Optional[int] = None
    achieved_rps: Optional[float] = None
    failed: Optional[int] = None
    #: Open loop only, over the measured window: least-squares slope (req/s) of the
    #: engine's vllm:num_requests_waiting (1 Hz scrapes), its last / max value, the
    #: vllm:num_preemptions_total delta, and the slope of the client-side queue (sent,
    #: no first token yet; a cross-check that needs no engine gauge).
    waiting_slope: Optional[float] = None
    waiting_last: Optional[float] = None
    waiting_max: Optional[float] = None
    preemptions: Optional[float] = None
    client_queue_slope: Optional[float] = None

    @property
    def tok_s(self) -> Optional[float]:
        if self.prefill_tok_s is None or self.decode_tok_s is None:
            return None
        return self.prefill_tok_s + self.decode_tok_s


#: measure(model, in_tokens, out_tokens, concurrency, step_s, warmup_s) -> StepMeasure
Measure = Callable[[str, int, int, int, float, float], StepMeasure]


# ------------------------------------------------------------------ pieces


def p95(values: Sequence[float]) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return vals[min(len(vals) - 1, max(0, math.ceil(0.95 * len(vals)) - 1))]


def counter_sum(text: str, name: str) -> Optional[float]:
    from tre_baselines.sources import parse_prometheus_text

    vals = [s.value for s in parse_prometheus_text(text) if s.name == name and math.isfinite(s.value)]
    return sum(vals) if vals else None


def queued(text: str) -> Optional[float]:
    """running + waiting of a replica's ``/metrics`` (None when a gauge is missing)."""
    from tre_common.vllm_metrics import VLLM_METRICS

    from tre_baselines.sources import parse_prometheus_text

    samples = parse_prometheus_text(text)
    total = 0.0
    for key in ("num_requests_running", "num_requests_waiting"):
        names = set(VLLM_METRICS[key])
        vals = [x.value for x in samples if x.name in names and math.isfinite(x.value)]
        if not vals:
            return None
        total += sum(vals)
    return total


def engine_rates(before: str, after: str, dt_s: float) -> tuple[Optional[float], Optional[float]]:
    """(prefill tok/s, decode tok/s) from two scrapes of the replica; None on a missing
    counter, a reset (engine restart) or a zero-length window."""
    out = []
    for name in (PROMPT_COUNTER, GEN_COUNTER):
        a, b = counter_sum(before, name), counter_sum(after, name)
        out.append(None if a is None or b is None or b < a or dt_s <= 0 else (b - a) / dt_s)
    return out[0], out[1]


def window_latencies(records: Sequence[dict], lo_ms: int, hi_ms: int) -> tuple[int, Optional[float], Optional[float]]:
    """(completed, p95 TTFT ms, p95 TPOT ms) of the requests that finished with 200 and no
    stream error inside ``[lo, hi]`` (raw records of ``r3_grid.drive_cell``)."""
    ok = [r for r in records
          if r.get("http_status") == 200 and not r.get("stream_error") and r.get("done_ts_ms") is not None
          and lo_ms <= r["done_ts_ms"] <= hi_ms]
    return len(ok), p95([r.get("ttft_ms") for r in ok]), p95([r.get("tpot_ms") for r in ok])


def http_get_text(url: str, timeout_s: float = 5.0) -> str:
    with urlopen(Request(url, headers={"accept": "text/plain"}), timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", "replace")


def make_http_measure(gateway_url: str, metrics_urls: Callable[[str], Sequence[str]], *, api: str = "chat",
                      prompt_mode: str = "natural", routing_strategy: Optional[str] = None,
                      run_key: str = "bl-profile", stream_call: Optional[Callable] = None,
                      fetch: Callable[[str], str] = http_get_text, raw_dir: Optional[Path] = None,
                      now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
                      fleet_view: Optional[Callable[[str], Any]] = None,
                      expected: Optional[dict[str, Any]] = None,
                      drain_poll_s: float = 0.5, drain_cap_s: float = DEFAULT_DRAIN_CAP_S,
                      sleep: Callable[[float], None] = time.sleep) -> Measure:
    """One step = ``r3_grid.drive_cell`` (the calibration chat sender) for ``step_s`` with
    ``concurrency`` closed-loop workers, the replica's ``/metrics`` (``metrics_urls(model)``)
    scraped when the warm-up ends and when the step ends. ``fleet_view(model)`` (awake
    bindings + routable pods) is read at both marks and must equal ``expected[model]``;
    after the step the replica must drain to nothing in flight before the call returns."""
    from scripts import r3_grid  # lazy: tre/deploy on PYTHONPATH (as for the calibration tools)

    def scrape(model: str) -> str:
        return "\n".join(fetch(url) for url in metrics_urls(model))

    def drain(model: str) -> None:
        deadline = time.monotonic() + drain_cap_s
        while True:
            try:
                q = queued(scrape(model))
            except Exception:  # noqa: BLE001 - unreadable: not idle
                q = None
            if q == 0:
                return
            if time.monotonic() >= deadline:
                raise ProfileAbort(f"{model}: replica still has {q} requests in flight {drain_cap_s:g}s after the step")
            sleep(drain_poll_s)

    def measure(model: str, in_tokens: int, out_tokens: int, concurrency: int, step_s: float,
                warmup_s: float) -> StepMeasure:
        cell = r3_grid.GridCell(int(in_tokens), int(out_tokens), int(concurrency))
        marks: dict[str, tuple[float, int, Optional[str]]] = {}

        views: dict[str, Any] = {}

        def mark(name: str) -> None:
            try:
                text: Optional[str] = scrape(model)
            except Exception:  # noqa: BLE001 - an unreadable scrape is unknown, not 0
                text = None
            marks[name] = (time.monotonic(), now_ms(), text)
            if fleet_view is not None:
                try:
                    views[name] = fleet_view(model)
                except Exception as exc:  # noqa: BLE001 - unknown fleet: not valid
                    views[name] = f"<unreadable: {exc!r}>"

        timers = [threading.Timer(warmup_s, mark, ("a",)), threading.Timer(step_s, mark, ("b",))]
        with tempfile.TemporaryDirectory(prefix="bl-profile-") as tmp:
            base = Path(raw_dir) if raw_dir is not None else Path(tmp)
            base.mkdir(parents=True, exist_ok=True)
            raw = base / f"{model}_{cell.scenario_id}.jsonl"
            raw.unlink(missing_ok=True)
            for t in timers:
                t.start()
            r3_grid.drive_cell(gateway_url, model, cell, step_s, raw_path=raw, stream_call=stream_call,
                               prompt_mode=prompt_mode, routing_strategy=routing_strategy,
                               run_key=f"{run_key}|{model}", api=api)
            for t in timers:
                t.join()
            records = [json.loads(ln) for ln in raw.read_text(encoding="utf-8").splitlines() if ln.strip()] \
                if raw.exists() else []
        (ta, wa, sa), (tb, wb, sb) = marks["a"], marks["b"]
        prefill, decode = engine_rates(sa, sb, tb - ta) if sa is not None and sb is not None else (None, None)
        completed, ttft, tpot = window_latencies(records, wa, wb)
        drain(model)  # the next step must not inherit this step's queue
        if fleet_view is not None:
            want = (expected or {}).get(model)
            if not views.get("a") == views.get("b") == want:
                return StepMeasure(int(concurrency), round(tb - ta, 3), completed, None, None, ttft, tpot,
                                   valid=False, note=f"fleet changed: expected {want}, a={views.get('a')}, "
                                                     f"b={views.get('b')}")
        return StepMeasure(int(concurrency), round(tb - ta, 3), completed, prefill, decode, ttft, tpot)

    return measure


def steady_latencies(records: Sequence[dict], lo_ms: int, hi_ms: int) -> tuple[int, int, Optional[float], Optional[float]]:
    """(arrived, completed, p95 TTFT ms, p95 TPOT ms) of the requests SENT inside
    ``[lo, hi]`` (steady state of an open-loop step). A request that did not complete (non
    200, stream error, no first token) counts as an infinite TTFT and TPOT, so failures
    can only make the step fail."""
    arrived = [r for r in records if r.get("send_ts_ms") is not None and lo_ms <= r["send_ts_ms"] <= hi_ms]
    ok = [r for r in arrived if r.get("http_status") == 200 and not r.get("stream_error")
          and r.get("ttft_ms") is not None and r.get("done_ts_ms") is not None]
    failed = len(arrived) - len(ok)
    ttft = p95([r["ttft_ms"] for r in ok] + [math.inf] * failed)
    tpot = p95([r.get("tpot_ms") for r in ok] + [math.inf] * failed)
    return len(arrived), len(ok), ttft, tpot


PREEMPT_COUNTER = "vllm:num_preemptions_total"
WAITING_GAUGE = "vllm:num_requests_waiting"


def slope(points: Sequence[tuple[float, float]]) -> Optional[float]:
    """Least-squares slope of (t_s, y) points (None with fewer than 3 or no time spread)."""
    pts = [(float(t), float(y)) for t, y in points if y is not None and math.isfinite(y)]
    if len(pts) < 3:
        return None
    mt = sum(t for t, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    den = sum((t - mt) ** 2 for t, _ in pts)
    return None if den <= 0 else sum((t - mt) * (y - my) for t, y in pts) / den


def client_queue_series(records: Sequence[dict], lo_ms: int, hi_ms: int, step_ms: int = 1000) -> list[tuple[float, float]]:
    """(t_s, requests sent but without a first token yet) every ``step_ms`` in [lo, hi]."""
    out = []
    for t in range(int(lo_ms), int(hi_ms) + 1, int(step_ms)):
        q = sum(1 for r in records if r.get("send_ts_ms") is not None and r["send_ts_ms"] <= t
                and (r.get("recv_first_token_ts_ms") is None or r["recv_first_token_ts_ms"] > t))
        out.append(((t - lo_ms) / 1000.0, float(q)))
    return out


#: Knee rule (decision 2026-10-06, TokenScale V_b / V_P): a step's waiting queue is
#: BOUNDED when the least-squares slope of vllm:num_requests_waiting over the measured
#: window is at most this fraction of the offered rate (req/s per req/s). An overloaded
#: step's queue grows at about (offered - served) req/s; on the 10-06 open-loop run the
#: client-side queue grew at <= 0.2 % of the rate on every step below the knee and at
#: >= 3.7 % above it, so 2 % separates them.
DEFAULT_KNEE_SLOPE_FRAC = 0.02


def queue_bounded(s: StepMeasure, slope_frac: float = DEFAULT_KNEE_SLOPE_FRAC) -> Optional[bool]:
    """None when the step has no waiting series (no evidence either way)."""
    if s.waiting_slope is None or s.rate_rps is None:
        return None
    return s.waiting_slope <= slope_frac * float(s.rate_rps)


def knee(steps: Sequence[StepMeasure], rate: str = "tok_s", slope_frac: float = DEFAULT_KNEE_SLOPE_FRAC,
         min_completed: int = 1) -> Optional[dict]:
    """The highest ``rate`` (tok_s | prefill_tok_s) of a valid step whose waiting queue is
    bounded, plus whether a higher offered rate was seen unbounded (``bracketed``; if not,
    the knee may lie above the ladder)."""
    ok = [s for s in steps if s.valid and getattr(s, rate) is not None and s.completed >= max(1, min_completed)
          and queue_bounded(s, slope_frac)]
    if not ok:
        return None
    best = max(ok, key=lambda s: getattr(s, rate))
    above = [s for s in steps if s.rate_rps is not None and s.rate_rps > best.rate_rps
             and queue_bounded(s, slope_frac) is False]
    return {"value": round(getattr(best, rate), 1), "rate_rps": round(best.rate_rps, 3),
            "achieved_rps": best.achieved_rps, "waiting_slope": best.waiting_slope,
            "preemptions": best.preemptions, "bracketed": bool(above),
            "first_unbounded_rps": round(min(s.rate_rps for s in above), 3) if above else None}


#: Open-loop step: measure(model, in_tokens, out_tokens, rate_rps, step_s, warmup_s)
OpenMeasure = Callable[[str, int, int, float, float, float], StepMeasure]


def make_openloop_measure(gateway_url: str, metrics_urls: Callable[[str], Sequence[str]], *,
                          api: str = "chat", prompt_mode: str = "natural", routing_strategy: Optional[str] = None,
                          run_key: str = "bl-profile-ol", seed: int = 1234, stream_call: Optional[Callable] = None,
                          fetch: Callable[[str], str] = http_get_text, raw_dir: Optional[Path] = None,
                          sender_processes: Optional[int] = None, prompt_dir_enabled: bool = True,
                          fleet_view: Optional[Callable[[str], Any]] = None,
                          expected: Optional[dict[str, Any]] = None, sample_s: float = 1.0,
                          drain_poll_s: float = 0.5, drain_cap_s: float = DEFAULT_DRAIN_CAP_S,
                          sleep: Callable[[float], None] = time.sleep) -> OpenMeasure:
    """One step = Poisson arrivals at ``rate_rps`` for ``step_s`` through the calibration's
    open-loop sender (``openloop.drive_cell_schedule``: one ``RpsSegment``, prompts
    materialised before the step, chat, ``ignore_eos``, temperature 0). The replica's
    ``/metrics`` is scraped every ``sample_s`` by the sender's sidecar; the token rates
    come from the samples nearest to the end of the warm-up and to the step end. The
    latencies are those of the requests sent between those two instants. The call returns
    after every request finished and the replica reports nothing in flight."""
    from scripts import openloop  # lazy: tre/deploy on PYTHONPATH
    from tre_replayer.engine.schedule import RpsSegment

    def scrape(model: str) -> str:
        return "\n".join(fetch(url) for url in metrics_urls(model))

    def drain(model: str) -> None:
        deadline = time.monotonic() + drain_cap_s
        while True:
            try:
                q = queued(scrape(model))
            except Exception:  # noqa: BLE001 - unreadable: not idle
                q = None
            if q == 0:
                return
            if time.monotonic() >= deadline:
                raise ProfileAbort(f"{model}: replica still has {q} requests in flight {drain_cap_s:g}s after the step")
            sleep(drain_poll_s)

    def view(model: str) -> Any:
        try:
            return fleet_view(model) if fleet_view is not None else None
        except Exception as exc:  # noqa: BLE001 - unknown fleet: not valid
            return f"<unreadable: {exc!r}>"

    def measure(model: str, in_tokens: int, out_tokens: int, rate_rps: float, step_s: float,
                warmup_s: float) -> StepMeasure:
        cell_id = f"i{int(in_tokens)}_o{int(out_tokens)}_r{rate_rps:.3f}"
        samples: list[tuple[int, str]] = []

        def sampler(now: int) -> dict:
            text = scrape(model)
            samples.append((int(now), text))
            # persisted 1 Hz to <raw>/<model>_<cell>.engine.jsonl (with raw_dir)
            return {"running": counter_sum(text, "vllm:num_requests_running"),
                    "waiting": counter_sum(text, WAITING_GAUGE),
                    "prompt_tokens_total": counter_sum(text, PROMPT_COUNTER),
                    "generation_tokens_total": counter_sum(text, GEN_COUNTER),
                    "preemptions_total": counter_sum(text, PREEMPT_COUNTER)}

        seg = RpsSegment(model=model, start_s=0.0, end_s=float(step_s), rps=float(rate_rps),
                         input_tokens=int(in_tokens), max_output_tokens=int(out_tokens))
        view_a = view(model)
        with tempfile.TemporaryDirectory(prefix="bl-profile-ol-") as tmp:
            base = Path(raw_dir) if raw_dir is not None else Path(tmp)
            base.mkdir(parents=True, exist_ok=True)
            raw = base / f"{model}_{cell_id}.jsonl"
            raw.unlink(missing_ok=True)
            (base / f"{model}_{cell_id}.engine.jsonl").unlink(missing_ok=True)
            start_ms, _end_ms, guard = openloop.drive_cell_schedule(
                gateway_url, model, cell_id, [seg], seed=seed, raw_path=raw, instant_sampler=sampler,
                instant_path=base / f"{model}_{cell_id}.engine.jsonl" if raw_dir is not None else None,
                instant_interval_s=sample_s, prompt_mode=prompt_mode,
                prompt_dir=Path(tmp) / "prompts" if prompt_dir_enabled else None,
                routing_strategy=routing_strategy, stream_call=stream_call, request_key=f"{run_key}|{model}|{cell_id}",
                api=api, sender_processes=sender_processes)
            records = [json.loads(ln) for ln in raw.read_text(encoding="utf-8").splitlines() if ln.strip()] \
                if raw.exists() else []
        drain(model)
        view_b = view(model)
        lo, hi = int(start_ms + warmup_s * 1000), int(start_ms + step_s * 1000)

        def nearest(target: int) -> Optional[tuple[int, str]]:
            best = min(samples, key=lambda s: abs(s[0] - target), default=None)
            return best if best is not None and abs(best[0] - target) <= max(2000, 2 * sample_s * 1000) else None

        a, b = nearest(lo), nearest(hi)
        prefill = decode = None
        window = (hi - lo) / 1000.0
        if a is not None and b is not None and b[0] > a[0]:
            window = (b[0] - a[0]) / 1000.0
            prefill, decode = engine_rates(a[1], b[1], window)
        arrived, completed, ttft, tpot = steady_latencies(records, lo, hi)
        span = (hi - lo) / 1000.0
        waits = []
        for ts, text in samples:
            if lo <= ts <= hi:
                w = counter_sum(text, WAITING_GAUGE)
                if w is not None:
                    waits.append(((ts - lo) / 1000.0, w))
        w_slope = slope(waits)
        preempt = None
        if a is not None and b is not None:
            pa, pb = counter_sum(a[1], PREEMPT_COUNTER), counter_sum(b[1], PREEMPT_COUNTER)
            preempt = None if pa is None or pb is None or pb < pa else pb - pa
        cq = slope(client_queue_series(records, lo, hi))
        queue = dict(waiting_slope=None if w_slope is None else round(w_slope, 4),
                     waiting_last=waits[-1][1] if waits else None,
                     waiting_max=max((w for _, w in waits), default=None), preemptions=preempt,
                     client_queue_slope=None if cq is None else round(cq, 4))
        notes = []
        notes.append(f"sender p99 lateness {guard.p99_delay_ms:.0f} ms")
        if guard.issues or guard.void_reasons:
            notes.append(f"sender guard: issues={list(guard.issues)} void={list(guard.void_reasons)}")
        if fleet_view is not None:
            want = (expected or {}).get(model)
            if not view_a == view_b == want:
                return StepMeasure(0, round(window, 3), completed, None, None, ttft, tpot, valid=False,
                                   note=f"fleet changed: expected {want}, a={view_a}, b={view_b}",
                                   rate_rps=float(rate_rps), arrived=arrived, failed=arrived - completed,
                                   achieved_rps=round(arrived / span, 3) if span > 0 else None, **queue)
        return StepMeasure(0, round(window, 3), completed, prefill, decode, ttft, tpot,
                           note="; ".join(notes) or None, rate_rps=float(rate_rps), arrived=arrived,
                           failed=arrived - completed, achieved_rps=round(arrived / span, 3) if span > 0 else None,
                           **queue)

    return measure


#: Open-loop rate ladder, x each model's base rate (decision 2026-10-06, PreServe mu).
DEFAULT_RATE_FACTORS = (0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3)
#: A step counts for mu only with at least this many completed steady-state requests.
DEFAULT_OPEN_MIN_COMPLETED = 150


def profile_model_open(measure: OpenMeasure, model: str, in_tokens: int, out_tokens: int, slo: Any, *,
                       base_rps: float, factors: Sequence[float], step_s: float, warmup_s: float,
                       min_completed: int = DEFAULT_OPEN_MIN_COMPLETED, prefill_base_rps: Optional[float] = None,
                       prefill_factors: Sequence[float] = (),
                       slope_frac: float = DEFAULT_KNEE_SLOPE_FRAC) -> tuple[dict, list[dict]]:
    """The open-loop ladders of one model -> ({mu, knee, velocity, slo, ...}, raw rows).

    Mixed ladder (``out_tokens``, rates ``base_rps x factors``): PreServe mu (SLO rule)
    and TokenScale V_b = (in+out) tok/s at the knee. Prefill ladder (out = 1, rates
    ``prefill_base_rps x prefill_factors``): V_P = prefill tok/s at the knee. Either
    ladder may be empty (its outputs are then None)."""
    ladders = [("mixed", int(out_tokens), float(base_rps or 0.0), tuple(factors))]
    if prefill_base_rps and prefill_factors:
        ladders.append(("prefill", 1, float(prefill_base_rps), tuple(prefill_factors)))
    runs: dict[str, list[StepMeasure]] = {"mixed": [], "prefill": []}
    aborted = None
    for kind, out, base, fs in ladders:
        for f in fs:
            if aborted:
                break
            st = measure(model, in_tokens, out, base * float(f), step_s, warmup_s)
            runs[kind].append(st)
            if not st.valid:
                aborted = st.note
    mixed, prefill = runs["mixed"], runs["prefill"]
    ttft_slo, tpot_slo = float(slo.ttft_slo_ms(in_tokens)), float(slo.tpot_p95_ms)
    v_b = None if aborted else knee(mixed, "tok_s", slope_frac, min_completed)
    v_p = None if aborted else knee(prefill, "prefill_tok_s", slope_frac, min_completed)
    result = {
        "mu": None if aborted else preserve_mu(mixed, ttft_slo, tpot_slo, min_completed),
        "knee": {"v_b": v_b, "v_p": v_p, "slope_frac": slope_frac},
        "velocity": None if v_b is None or v_p is None else
        {"buckets": [[v_b["value"]] * 3 for _ in range(3)], "v_prefill": v_p["value"]},
        "base_rps": base_rps, "prefill_base_rps": prefill_base_rps,
        "non_monotone_rates": [round(x, 3) for x in non_monotone(mixed, ttft_slo, tpot_slo, min_completed)],
        "slo": {"ttft_p95_ms": round(ttft_slo, 1), "tpot_p95_ms": tpot_slo, "in_tokens": in_tokens,
                "ttft_idle_c_ms": getattr(slo, "ttft_idle_c_ms", None),
                "ttft_idle_b_ms_per_token": getattr(slo, "ttft_idle_b_ms_per_token", None),
                "ttft_slowdown_k": getattr(slo, "ttft_slowdown_k", None),
                "ttft_floor_ms": getattr(slo, "ttft_floor_ms", None)},
        "aborted": aborted,
    }
    rows = [dict(model=model, kind=f"open_{kind}", in_tokens=in_tokens, out_tokens=out, **asdict(s), tok_s=s.tok_s,
                 passes=step_passes(s, ttft_slo, tpot_slo, min_completed) if kind == "mixed" else None,
                 queue_bounded=queue_bounded(s, slope_frac))
            for kind, out, steps in (("mixed", out_tokens, mixed), ("prefill", 1, prefill)) for s in steps]
    return result, rows


def make_stub_open_measure(capacity_rps: float = 10.0, prefill_capacity_rps: float = 30.0) -> OpenMeasure:
    """Synthetic open-loop steps: served rate saturates at the capacity, TTFT explodes and
    the waiting queue grows at (offered - served) above it."""

    def measure(model: str, in_tokens: int, out_tokens: int, rate_rps: float, step_s: float,
                warmup_s: float) -> StepMeasure:
        cap = prefill_capacity_rps if out_tokens <= 1 else capacity_rps
        served = min(rate_rps, cap)
        window = step_s - warmup_s
        rho = rate_rps / cap
        ttft = 100.0 / max(1e-3, 1.0 - rho) if rho < 1 else 1e5
        n = int(rate_rps * window)
        return StepMeasure(0, window, n, served * in_tokens, served * out_tokens, ttft,
                           None if out_tokens <= 1 else 20.0 + 20 * rho, rate_rps=rate_rps, arrived=n,
                           achieved_rps=rate_rps, failed=0, waiting_slope=max(0.0, rate_rps - cap),
                           preemptions=0.0)

    return measure


def make_stub_measure(vmax_tok_s: float = 10000.0, c_half: float = 4.0, vmax_prefill_tok_s: float = 30000.0,
                      ttft_ms_per_c: float = 30.0, tpot_ms_per_c: float = 2.0) -> Measure:
    """Synthetic saturating throughput and latencies that grow with concurrency."""

    def measure(model: str, in_tokens: int, out_tokens: int, concurrency: int, step_s: float,
                warmup_s: float) -> StepMeasure:
        vmax = vmax_prefill_tok_s if out_tokens <= 1 else vmax_tok_s
        rate = vmax * concurrency / (concurrency + c_half)
        share = in_tokens / (in_tokens + out_tokens)
        window = step_s - warmup_s
        return StepMeasure(concurrency, window, int(rate * window / (in_tokens + out_tokens)), rate * share,
                           rate * (1 - share), 100.0 + ttft_ms_per_c * concurrency, 20.0 + tpot_ms_per_c * concurrency)

    return measure


# ------------------------------------------------------------------ the ladder


def run_ladder(measure: Measure, model: str, in_tokens: int, out_tokens: int, step_s: float, warmup_s: float,
               ladder: Sequence[int] = DEFAULT_LADDER, *, extend_gain: float = DEFAULT_EXTEND_GAIN,
               max_concurrency: int = DEFAULT_MAX_CONCURRENCY, rate: str = "tok_s",
               warn: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr)) -> list[StepMeasure]:
    """Every step of the ladder; then, while the last step still gained more than
    ``extend_gain`` over the one before, further steps at x1.5 up to ``max_concurrency``."""
    steps: list[StepMeasure] = []
    for c in ladder:
        steps.append(measure(model, in_tokens, out_tokens, int(c), step_s, warmup_s))
        if not steps[-1].valid:
            return steps  # aborted: the caller reports it


    def gain() -> float:
        if len(steps) < 2:
            return 0.0
        a, b = getattr(steps[-2], rate), getattr(steps[-1], rate)
        return 0.0 if not a or b is None else (b - a) / a

    while gain() > extend_gain:
        nxt = int(math.ceil(steps[-1].concurrency * 1.5))
        if nxt > max_concurrency:
            warn(f"warning: {model} in={in_tokens} out={out_tokens}: still +{gain():.0%} at concurrency "
                 f"{steps[-1].concurrency} (max {max_concurrency}); the peak may be higher")
            break
        steps.append(measure(model, in_tokens, out_tokens, nxt, step_s, warmup_s))
        if not steps[-1].valid:
            break
    return steps


def _best(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def _load(s: StepMeasure) -> float:
    """The step's load level: the offered rate (open loop) or the concurrency (closed)."""
    return float(s.rate_rps) if s.rate_rps is not None else float(s.concurrency)


def step_passes(s: StepMeasure, ttft_slo_ms: float, tpot_slo_ms: float, min_completed: int = 1) -> bool:
    return (s.valid and s.completed >= max(1, min_completed) and s.tok_s is not None
            and s.ttft_p95_ms is not None and s.tpot_p95_ms is not None
            and s.ttft_p95_ms <= ttft_slo_ms and s.tpot_p95_ms <= tpot_slo_ms)


def preserve_mu(steps: Sequence[StepMeasure], ttft_slo_ms: float, tpot_slo_ms: float,
                min_completed: int = 1) -> Optional[dict[str, float]]:
    """mu from the highest-load step meeting both p95 SLOs (None if none does)."""
    ok = [s for s in steps if step_passes(s, ttft_slo_ms, tpot_slo_ms, min_completed)]
    if not ok:
        return None
    best = max(ok, key=_load)
    out = {"p": round(best.prefill_tok_s, 1), "d": round(best.decode_tok_s, 1), "t": round(best.tok_s, 1)}
    if best.rate_rps is not None:
        out.update(rate_rps=round(best.rate_rps, 3), achieved_rps=best.achieved_rps)
    else:
        out["concurrency"] = best.concurrency
    return out


def non_monotone(steps: Sequence[StepMeasure], ttft_slo_ms: float, tpot_slo_ms: float,
                 min_completed: int = 1) -> list[float]:
    """Load levels that fail an SLO although a higher one passes (the verdict is not
    monotone). A step with too few completed requests is no evidence either way."""
    passing = [_load(s) for s in steps if step_passes(s, ttft_slo_ms, tpot_slo_ms, min_completed)]
    top = max(passing, default=None)
    return [] if top is None else [_load(s) for s in steps if _load(s) < top and s.completed >= min_completed
                                   and not step_passes(s, ttft_slo_ms, tpot_slo_ms, min_completed)]


def profile_model(measure: Measure, model: str, in_tokens: int, out_tokens: int, slo: Any, *, step_s: float,
                  warmup_s: float, ladder: Sequence[int], extend_gain: float = DEFAULT_EXTEND_GAIN,
                  max_concurrency: int = DEFAULT_MAX_CONCURRENCY) -> tuple[dict, list[dict]]:
    """Both ladders of one model -> ({velocity, mu, slo}, raw rows)."""
    kw = dict(extend_gain=extend_gain, max_concurrency=max_concurrency)
    mixed = run_ladder(measure, model, in_tokens, out_tokens, step_s, warmup_s, ladder, rate="tok_s", **kw)
    aborted = next((s.note for s in mixed if not s.valid), None)
    prefill = [] if aborted else run_ladder(measure, model, in_tokens, 1, step_s, warmup_s, ladder,
                                            rate="prefill_tok_s", **kw)
    aborted = aborted or next((s.note for s in prefill if not s.valid), None)
    ttft_slo = float(slo.ttft_slo_ms(in_tokens))
    tpot_slo = float(slo.tpot_p95_ms)
    v_b, v_p = _best([s.tok_s for s in mixed]), _best([s.prefill_tok_s for s in prefill])
    result = {
        "velocity": None if aborted or v_b is None or v_p is None else
        {"buckets": [[round(v_b, 1)] * 3 for _ in range(3)], "v_prefill": round(v_p, 1)},
        "mu": None if aborted else preserve_mu(mixed, ttft_slo, tpot_slo),
        "slo": {"ttft_p95_ms": round(ttft_slo, 1), "tpot_p95_ms": tpot_slo, "in_tokens": in_tokens,
                "ttft_idle_c_ms": getattr(slo, "ttft_idle_c_ms", None),
                "ttft_idle_b_ms_per_token": getattr(slo, "ttft_idle_b_ms_per_token", None),
                "ttft_slowdown_k": getattr(slo, "ttft_slowdown_k", None),
                "ttft_floor_ms": getattr(slo, "ttft_floor_ms", None)},
        "aborted": aborted,
    }
    rows = [dict(model=model, kind=kind, in_tokens=in_tokens, out_tokens=out, **asdict(s), tok_s=s.tok_s)
            for kind, out, steps in (("mixed", out_tokens, mixed), ("prefill", 1, prefill)) for s in steps]
    return result, rows


# ------------------------------------------------------------------ cluster checks (read only)


def get_state(sm_url: str, timeout_s: float = 5.0) -> dict:
    with urlopen(Request(sm_url.rstrip("/") + "/v2/state", headers={"accept": "application/json"}),
                 timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_one_awake(sm_url: str, model: str, timeout_s: float = 5.0) -> int:
    """Awake replicas of ``model`` per the SM's ``GET /v2/state`` (read only)."""
    state = get_state(sm_url, timeout_s)
    return int(((state.get("models") or {}).get(model) or {}).get("awake", 0))


def awake_bindings(state: dict, model: str) -> frozenset:
    return frozenset(str(b.get("serve_id")) for b in state.get("bindings") or ()
                     if b.get("model") == model and b.get("awake") and not b.get("hidden"))


def check_preconditions(models: Sequence[str], redis_get: Callable[[str], Optional[str]],
                        list_apa: Callable[[], list]) -> list[str]:
    """Why profiling must not start (empty = ok): both run modes observe, no APA CR on a
    profiled model. ``redis_get`` / ``list_apa`` are read-only."""
    problems = []
    for key in (CONTROLLER_MODE_KEY, SM_ACTUATION_KEY):
        value = redis_get(key)
        if value not in (None, "", "observe"):
            problems.append(f"{key} is {value!r}, must be observe (deploy/scripts/set_run_mode.sh observe observe)")
    for item in list_apa():
        target = ((item.get("spec") or {}).get("scaleTargetRef") or {}).get("name")
        if target in set(models):
            meta = item.get("metadata") or {}
            problems.append(f"APA PodAutoscaler {meta.get('namespace')}/{meta.get('name')} targets {target}")
    return problems


def _kubectl(argv: Sequence[str]) -> str:
    return subprocess.run(list(argv), capture_output=True, text=True, check=True).stdout


def live_registry(namespace: str = "tre-v2", name: str = "tre-v2-registry", kubectl: str = "kubectl") -> str:
    """The live registry ConfigMap (``kubectl get``, read only) written to a temp file."""
    text = _kubectl([kubectl, "-n", namespace, "get", "configmap", name, "-o", "jsonpath={.data.registry\\.yaml}"])
    if not text.strip():
        raise ValueError(f"ConfigMap {namespace}/{name} has no registry.yaml")
    fh = tempfile.NamedTemporaryFile("w", suffix="-registry.yaml", delete=False, encoding="utf-8")
    with fh:
        fh.write(text)
    return fh.name


def model_slo(model: str, registry_path: Optional[str]) -> Any:
    """The model's SLO definition from the registry with its fitted c/b (strict: refuses a
    registry without them, as an actuating shell does)."""
    from tre_common.registry import load_registry

    from tre_baselines.config import _slo_definition

    registry = load_registry(registry_path)
    return _slo_definition(model, registry.model(model), registry, strict=True)


def run_isolated(models: Sequence[str], one: Callable[[str], tuple]) -> dict[str, tuple]:
    """Each model's ladders in their own (forked) process; results come back by a queue."""
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()

    def child(model: str) -> None:
        try:
            queue.put((model, "ok", one(model)))
        except BaseException as exc:  # noqa: BLE001 - reported by the parent
            queue.put((model, "error", repr(exc)))

    procs = [ctx.Process(target=child, args=(m,), name=f"bl-profile-{m}") for m in models]
    for p in procs:
        p.start()
    out: dict[str, tuple] = {}
    for _ in procs:
        model, status, payload = queue.get()
        out[model] = (status, payload)
    for p in procs:
        p.join()
    return out


def _ladder(text: str) -> tuple[int, ...]:
    try:
        ladder = tuple(int(x) for x in text.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not ladder or any(c < 1 for c in ladder):
        raise argparse.ArgumentTypeError("concurrencies must be >= 1")
    return ladder


def _base_rps(text: str) -> dict[str, float]:
    out = {}
    for part in (x.strip() for x in text.split(",")):
        if not part:
            continue
        name, sep, value = part.partition("=")
        try:
            rate = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"--base-rps: {part!r} is not model=req/s") from exc
        if not sep or not name or not rate > 0:
            raise argparse.ArgumentTypeError(f"--base-rps: {part!r} is not model=req/s with req/s > 0")
        out[name.strip()] = rate
    return out


def _factors(text: str) -> tuple[float, ...]:
    try:
        vals = tuple(float(x) for x in text.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not vals or any(not v > 0 for v in vals):
        raise argparse.ArgumentTypeError("rate factors must be > 0")
    return tuple(sorted(vals))


def _model_factors(text: str) -> dict[str, tuple[float, ...]]:
    """``model=f1:f2,model2=`` -> per-model rate factors (empty = no steps for that model)."""
    out: dict[str, tuple[float, ...]] = {}
    for part in (x.strip() for x in text.split(",")):
        if not part:
            continue
        name, sep, value = part.partition("=")
        if not sep or not name.strip():
            raise argparse.ArgumentTypeError(f"{part!r} is not model=f1:f2")
        out[name.strip()] = _factors(value.replace(":", ",")) if value.strip() else ()
    return out


def main(argv: Optional[list[str]] = None, *, measure: Optional[Measure] = None,
         slo_for: Optional[Callable[[str], Any]] = None) -> int:
    ap = argparse.ArgumentParser(description="TokenScale V_b/V_P + PreServe mu profiling (one awake replica per model)")
    ap.add_argument("--models", required=True, help="comma list; profiled in parallel, one awake replica each")
    ap.add_argument("--in-tokens", type=int, default=492, help="request input length after the chat template")
    ap.add_argument("--out-tokens", type=int, default=400, help="max_tokens (ignore_eos)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gateway-url", default=None, help="the chat endpoint URL (.../v1/chat/completions)")
    ap.add_argument("--sm-url", default=None, help="required for a real run: checks exactly one awake replica")
    ap.add_argument("--redis-url", default=None, help="read the run modes here (default: kubectl exec redis-cli)")
    ap.add_argument("--namespace", default="tre-v2", help="namespace of Redis and the registry ConfigMap")
    ap.add_argument("--redis-deploy", default="tre-v2-redis")
    ap.add_argument("--kubectl", default="kubectl")
    ap.add_argument("--model-namespace", default="default", help="namespace of the model pods")
    ap.add_argument("--metrics-port", type=int, default=8000, help="pod port serving vLLM /metrics")
    ap.add_argument("--registry", default=None,
                    help="registry with the fitted c/b (default: the live ConfigMap; dry-run: the shared file)")
    ap.add_argument("--routing-strategy", default=None)
    ap.add_argument("--raw-dir", default=None, help="keep the per-request raw JSONL here")
    ap.add_argument("--dry-run", action="store_true", help="synthetic stub; nothing is sent")
    ap.add_argument("--i-have-user-approval", action="store_true", help="required to send real requests")
    ap.add_argument("--step-s", type=float, default=None, help="default 60 (closed loop) / 110 (open loop)")
    ap.add_argument("--warmup-s", type=float, default=None, help="default 15 (closed loop) / 20 (open loop)")
    ap.add_argument("--drain-cap-s", type=float, default=DEFAULT_DRAIN_CAP_S)
    ap.add_argument("--concurrency", type=_ladder, default=DEFAULT_LADDER)
    ap.add_argument("--extend-gain", type=float, default=DEFAULT_EXTEND_GAIN)
    ap.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    ap.add_argument("--open-loop", action="store_true",
                    help="Poisson rate ladders: PreServe mu + TokenScale V_b at the knee (mixed, --base-rps x "
                         "--rate-factors) and V_P at the knee (out=1, --prefill-base-rps x --prefill-rate-factors)")
    ap.add_argument("--base-rps", type=_base_rps, default={},
                    help="open loop: model=req/s,... (1.0x of each model's mixed ladder)")
    ap.add_argument("--rate-factors", type=_factors, default=DEFAULT_RATE_FACTORS)
    ap.add_argument("--model-rate-factors", type=_model_factors, default={},
                    help="open loop: per-model override of --rate-factors, model=f1:f2,... (model= : no mixed steps)")
    ap.add_argument("--prefill-base-rps", type=_base_rps, default={},
                    help="open loop: model=req/s,... (1.0x of each model's out=1 ladder; omit = no V_P ladder)")
    ap.add_argument("--prefill-rate-factors", type=_factors, default=None)
    ap.add_argument("--knee-slope-frac", type=float, default=DEFAULT_KNEE_SLOPE_FRAC,
                    help="queue bounded iff slope(num_requests_waiting) <= this x offered req/s")
    ap.add_argument("--min-completed", type=int, default=DEFAULT_OPEN_MIN_COMPLETED,
                    help="open loop: completed steady-state requests a step needs to count for mu")
    ap.add_argument("--schedule-seed", type=int, default=1234, help="open loop: Poisson arrival seed")
    args = ap.parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.step_s is None:
        args.step_s = 110.0 if args.open_loop else 60.0
    if args.warmup_s is None:
        args.warmup_s = 20.0 if args.open_loop else 15.0
    def mixed_factors(m: str) -> tuple[float, ...]:
        return args.model_rate_factors.get(m, args.rate_factors)

    def prefill_factors(m: str) -> tuple[float, ...]:
        return tuple(args.prefill_rate_factors or ()) if m in args.prefill_base_rps else ()

    if args.open_loop:
        no_base = [m for m in models if mixed_factors(m) and m not in args.base_rps]
        if no_base:
            print(f"error: --open-loop needs --base-rps for {no_base} (or --model-rate-factors {no_base[0]}=)",
                  file=sys.stderr)
            return 2
        if args.prefill_base_rps and not args.prefill_rate_factors:
            print("error: --prefill-base-rps needs --prefill-rate-factors", file=sys.stderr)
            return 2
    if args.warmup_s >= args.step_s:
        print("error: --warmup-s must be < --step-s", file=sys.stderr)
        return 2
    real = measure is None and not args.dry_run
    if real and not args.i_have_user_approval:
        print("refusing to send requests without --i-have-user-approval (use --dry-run)", file=sys.stderr)
        return 3
    if real and (not args.gateway_url or not args.sm_url):
        print("error: a real run needs --gateway-url and --sm-url", file=sys.stderr)
        return 2
    if args.open_loop:
        steps = {m: len(mixed_factors(m)) + len(prefill_factors(m)) for m in models}
        minutes = max(steps.values(), default=0) * args.step_s / 60
        print(f"estimated time: ~{minutes:.0f} min + drains (models in parallel; steps per model {steps}), open loop: "
              f"mixed x {({m: list(mixed_factors(m)) for m in models})} of {args.base_rps}; "
              f"out=1 x {({m: list(prefill_factors(m)) for m in models})} of {args.prefill_base_rps}")
    else:
        minutes = 2 * len(args.concurrency) * args.step_s / 60
        print(f"estimated time: ~{minutes:.0f} min per model (models in parallel), ladder {list(args.concurrency)}")
    registry_source = f"file:{args.registry}" if args.registry else "shared registry file (dry-run)"
    registry_path = args.registry
    if slo_for is None:
        if registry_path is None and real:
            try:
                registry_path = live_registry(args.namespace, kubectl=args.kubectl)
            except (subprocess.CalledProcessError, OSError, ValueError) as exc:
                print(f"error: cannot read the live registry ConfigMap: {exc}", file=sys.stderr)
                return 2
            registry_source = f"configmap {args.namespace}/tre-v2-registry (live, read with kubectl get)"
        slo_for = lambda m: model_slo(m, registry_path)  # noqa: E731
    try:
        slos = {m: slo_for(m) for m in models}
    except (ValueError, KeyError, SystemExit) as exc:
        print(f"error: SLO: {exc}", file=sys.stderr)
        return 2
    if measure is None:
        if args.dry_run:
            measure = make_stub_open_measure() if args.open_loop else make_stub_measure()
        else:
            from tre_baselines.tools import arm

            redis = (arm.DirectRedis(args.redis_url) if args.redis_url else
                     arm.KubectlRedis(arm.subprocess_runner, args.kubectl, args.namespace, args.redis_deploy))

            def list_apa() -> list:
                doc = json.loads(_kubectl([args.kubectl, "get", arm.APA_RESOURCE, "-A", "-o", "json"]) or "{}")
                return doc.get("items") or []

            problems = check_preconditions(models, redis.get, list_apa)
            wrong = {m: n for m in models if (n := check_one_awake(args.sm_url, m)) != 1}
            if wrong:
                problems.append(f"profile with exactly one awake replica per model, got {wrong}")
            from scripts.r3_grid import discover_pod_metrics_endpoints

            def routable(m: str) -> tuple:
                return tuple(sorted(discover_pod_metrics_endpoints(m, args.model_namespace, args.metrics_port)))

            urls = {m: routable(m) for m in models}
            if any(len(u) != 1 for u in urls.values()):
                problems.append(f"expected one routable pod per model, got {urls}")
            if problems:
                print("refusing to profile:\n  " + "\n  ".join(problems), file=sys.stderr)
                return 4

            def fleet_view(m: str) -> tuple:
                return (tuple(sorted(awake_bindings(get_state(args.sm_url), m))), routable(m))

            expected = {m: fleet_view(m) for m in models}
            if args.open_loop:
                measure = make_openloop_measure(args.gateway_url, lambda m: urls[m],
                                                routing_strategy=args.routing_strategy, seed=args.schedule_seed,
                                                raw_dir=Path(args.raw_dir) if args.raw_dir else None,
                                                fleet_view=fleet_view, expected=expected, drain_cap_s=args.drain_cap_s)
            else:
                measure = make_http_measure(args.gateway_url, lambda m: urls[m], routing_strategy=args.routing_strategy,
                                            raw_dir=Path(args.raw_dir) if args.raw_dir else None,
                                            fleet_view=fleet_view, expected=expected, drain_cap_s=args.drain_cap_s)

    def one(model: str) -> tuple[dict, list[dict]]:
        if args.open_loop:
            return profile_model_open(measure, model, args.in_tokens, args.out_tokens, slos[model],
                                      base_rps=args.base_rps.get(model), factors=mixed_factors(model),
                                      step_s=args.step_s, warmup_s=args.warmup_s, min_completed=args.min_completed,
                                      prefill_base_rps=args.prefill_base_rps.get(model),
                                      prefill_factors=prefill_factors(model), slope_frac=args.knee_slope_frac)
        return profile_model(measure, model, args.in_tokens, args.out_tokens, slos[model], step_s=args.step_s,
                             warmup_s=args.warmup_s, ladder=args.concurrency, extend_gain=args.extend_gain,
                             max_concurrency=args.max_concurrency)

    outcomes = run_isolated(models, one)
    errors = {m: p for m, (st, p) in outcomes.items() if st != "ok"}
    results = {m: p for m, (st, p) in outcomes.items() if st == "ok"}
    doc = {
        "shape": {"in_tokens": args.in_tokens, "out_tokens": args.out_tokens,
                  "note": "TokenScale buckets degenerate to one cell (all nine = V_b); disclose"},
        "registry": registry_source,
        "mode": "open_loop" if args.open_loop else "closed_loop",
        "step_s": args.step_s, "warmup_s": args.warmup_s,
    }
    if args.open_loop:
        doc.update({
            "schedule_seed": args.schedule_seed, "min_completed": args.min_completed,
            "rate_factors": {m: list(mixed_factors(m)) for m in models},
            "prefill_rate_factors": {m: list(prefill_factors(m)) for m in models},
            "base_rps": {m: r["base_rps"] for m, (r, _) in results.items()},
            "prefill_base_rps": {m: r["prefill_base_rps"] for m, (r, _) in results.items()},
            "non_monotone_rates": {m: r["non_monotone_rates"] for m, (r, _) in results.items()},
            "knee_rule": f"queue bounded iff least-squares slope of vllm:num_requests_waiting over the measured "
                         f"window <= {args.knee_slope_frac:g} x offered req/s; knee = highest tok/s of a bounded step",
            "knee": {m: r["knee"] for m, (r, _) in results.items()},
            "velocity": {m: r["velocity"] for m, (r, _) in results.items()},
        })
    else:
        doc["velocity"] = {m: r["velocity"] for m, (r, _) in results.items()}
    doc.update({
        "mu": {m: r["mu"] for m, (r, _) in results.items()},
        "slo": {m: r["slo"] for m, (r, _) in results.items()},
        "aborted": {**{m: r["aborted"] for m, (r, _) in results.items() if r["aborted"]}, **errors},
    })
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(doc, sort_keys=False, default_flow_style=None)
    (out / "profile.yaml").write_text(text, encoding="utf-8")
    rows = [row for _, (_, rs) in results.items() for row in rs]
    if rows:
        with (out / "profile_raw.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    sys.stdout.write(text)
    if doc["aborted"]:
        print(f"error: aborted / failed: {doc['aborted']}", file=sys.stderr)
        return EXIT_ABORTED
    missing = [m for m, (r, _) in results.items() if (r["mu"] is None and not args.open_loop)
               or (not args.open_loop and r["velocity"] is None)
               or (args.open_loop and mixed_factors(m) and r["mu"] is None)]
    unbracketed = {m: [k for k, v in (r.get("knee") or {}).items() if isinstance(v, dict) and not v["bracketed"]]
                   for m, (r, _) in results.items()}
    unbracketed = {m: v for m, v in unbracketed.items() if v}
    if unbracketed:
        print(f"warning: knee not bracketed (no unbounded step above it; it may lie higher): {unbracketed}",
              file=sys.stderr)
    bumpy = {m: r["non_monotone_rates"] for m, (r, _) in results.items() if r.get("non_monotone_rates")}
    if bumpy:
        print(f"warning: SLO verdict not monotone in the rate (failing below a passing rate): {bumpy}; "
              "lengthen the steps (--step-s)", file=sys.stderr)
    if missing:
        print(f"warning: no SLO-meeting step / unreadable counters for {missing}: see profile_raw.csv", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

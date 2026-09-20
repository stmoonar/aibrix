#!/usr/bin/env python3
"""Open-loop load primitives for the calibration campaign (schedule-driven R3).

Why this exists
---------------
The closed-loop grid driver in :mod:`r3_grid` holds a fixed number of worker threads
busy: every worker sends its next request only once the previous one came back. That is
a *self-limiting* load source. Measured against the fleet it never produced a
TTFT-bound SLO violation (R3 TTFT ratio ~1 while production violations run 26-74x SLO)
and it left ``avg_waiting`` at exactly zero in 81.5 / 90.7 / 100 % of windows for
7b / 8b / 14b, which makes ``lambda_wait`` unidentifiable: the queue term of TSS is
multiplied by a regressor that is always 0.

An *open-loop* source fires at a wall-clock schedule regardless of whether the previous
request finished, so offered load can exceed capacity and a real waiting queue forms.
This module provides that source, expressed as replayer schedules so the primitives,
the trace format and the sender are all shared with the replayer
(:mod:`tre_replayer.engine.schedule` / ``dispatcher`` / ``http_sender``).

What is in here
---------------
* :func:`drive_cell_schedule` - the open-loop counterpart of ``r3_grid.drive_cell``. It
  emits the SAME per-request raw JSONL schema (``r3_grid.RAW_COLUMNS``) and the same
  instant sidecar, so ``rewindow_from_raw.py`` and ``tre_calibration`` consume it
  unchanged.
* :func:`check_cell` - a fail-loud guard. The closed-loop worker swallows every send
  exception (``except Exception: continue``), so a misconfigured sender produces an
  empty raw file and a silently-zero row rather than an error. Here a cell that sent
  nothing, completed nothing, errored too often, or could not keep its schedule is a
  hard failure.
* :func:`make_pod_metrics_sampler` - a 1 Hz sidecar that scrapes the model pods'
  ``/metrics`` directly instead of reading the gateway's redis buckets. The gateway
  writes instantaneous gauges on a 10 s boundary-aligned ticker
  (:data:`tre_common.rediskeys.SCRAPE_INTERVAL_MS`), i.e. 3 samples per 30 s control
  window. A burst that saturates the engine for a few seconds is therefore very likely
  to fall entirely between two gateway samples and be recorded as ``avg_waiting == 0``.
  Sampling at 1 s and *additionally* marking which samples would have landed on the live
  10 s grid (:func:`mark_live_grid`) lets the campaign compare ground truth against
  "what the controller would have seen" from the same capture. This changes only the
  campaign sidecar; the production gateway cadence is untouched.
"""
from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from tre_common.rediskeys import SCRAPE_INTERVAL_MS

#: Sidecar cadence for the campaign. One sample per second; see the module docstring.
DEFAULT_SIDECAR_INTERVAL_S = 1.0

#: The cadence the live control path actually observes. Samples are tagged against this
#: grid so a capture can be replayed at either resolution.
LIVE_GRID_MS = SCRAPE_INTERVAL_MS

#: Guard defaults. ``p99_delay`` is how late the dispatcher fired a request relative to
#: its scheduled time; ``pool_wait`` is how long a fired request then waited for a sender
#: thread. Either one growing means the "open loop" has quietly become a closed loop
#: bounded by the driver, which is the exact failure this whole module exists to avoid.
DEFAULT_MAX_P99_DELAY_MS = 250.0
DEFAULT_MAX_P99_POOL_WAIT_MS = 250.0
DEFAULT_MAX_ERROR_RATE = 0.05
#: Sender threads. An open-loop cell at rho>1 accumulates backlog for its whole duration,
#: so the pool must be sized for the peak in-flight, not for the offered rate.
DEFAULT_MAX_IN_FLIGHT = 4096

#: vLLM instantaneous gauges, keyed by the name the TRE metrics schema uses.
#: Mirrors ``tre_controller.store.metrics_store.INSTANT_METRICS`` (which reads the same
#: gauges out of redis after the gateway has scraped them).
POD_INSTANT_GAUGES = {
    "waiting": "vllm:num_requests_waiting",
    "running": "vllm:num_requests_running",
    "swapping": "vllm:num_requests_swapped",
}


class CellGuardError(RuntimeError):
    """A cell's load was not actually delivered as specified."""


@dataclass(frozen=True)
class CellGuard:
    """Post-hoc verdict on one cell's dispatch."""

    cell_id: str
    scheduled: int
    sent: int
    completed: int
    errors: int
    p99_delay_ms: float
    p99_pool_wait_ms: float
    issues: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def error_rate(self) -> float:
        return 0.0 if self.sent == 0 else self.errors / self.sent

    def as_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "scheduled": self.scheduled,
            "sent": self.sent,
            "completed": self.completed,
            "errors": self.errors,
            "error_rate": round(self.error_rate, 6),
            "p99_delay_ms": round(self.p99_delay_ms, 3),
            "p99_pool_wait_ms": round(self.p99_pool_wait_ms, 3),
            "issues": list(self.issues),
            "ok": self.ok,
        }


def _p99(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    index = max(0, math.ceil(0.99 * len(ordered)) - 1)
    return ordered[index]


def check_cell(
    cell_id: str,
    *,
    scheduled: int,
    records: Sequence[dict],
    p99_delay_ms: float,
    max_p99_delay_ms: float = DEFAULT_MAX_P99_DELAY_MS,
    max_p99_pool_wait_ms: float = DEFAULT_MAX_P99_POOL_WAIT_MS,
    max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
) -> CellGuard:
    """Verdict on a dispatched cell. Pure - takes the sender's records, no network.

    ``records`` are :class:`tre_replayer.engine.http_sender.StreamingHttpSender` rows.
    A record counts as an error when the HTTP status is not 2xx (status 0 is the
    sender's "transport failed" code) or no end-to-end time was measured.
    """
    sent = len(records)
    errors = sum(1 for r in records if not _record_ok(r))
    completed = sent - errors
    pool_waits = [float(r.get("pool_wait_ms", 0.0) or 0.0) for r in records]
    p99_pool_wait_ms = _p99(pool_waits)

    issues: list[str] = []
    if scheduled <= 0:
        issues.append("schedule produced 0 requests (empty or mis-filtered segments)")
    if sent == 0:
        issues.append("0 requests were sent")
    elif sent < scheduled:
        issues.append(f"only {sent}/{scheduled} scheduled requests were sent")
    if sent > 0 and completed == 0:
        issues.append(f"0/{sent} requests completed (every send failed)")
    if sent > 0 and errors / sent > max_error_rate:
        issues.append(
            f"error rate {errors / sent:.1%} > {max_error_rate:.1%} ({errors}/{sent})"
        )
    if p99_delay_ms > max_p99_delay_ms:
        issues.append(
            f"p99 dispatch delay {p99_delay_ms:.1f}ms > {max_p99_delay_ms:.1f}ms "
            "(the driver could not keep the schedule: offered load was under-delivered)"
        )
    if p99_pool_wait_ms > max_p99_pool_wait_ms:
        issues.append(
            f"p99 sender pool wait {p99_pool_wait_ms:.1f}ms > {max_p99_pool_wait_ms:.1f}ms "
            "(sender threads starved: the open loop degenerated into a closed loop)"
        )
    return CellGuard(
        cell_id=cell_id,
        scheduled=scheduled,
        sent=sent,
        completed=completed,
        errors=errors,
        p99_delay_ms=float(p99_delay_ms),
        p99_pool_wait_ms=p99_pool_wait_ms,
        issues=tuple(issues),
    )


def _record_ok(record: dict) -> bool:
    status = record.get("http_status")
    if status is None or not (200 <= int(status) < 300):
        return False
    return record.get("e2e_ms") is not None


def raise_on_guard(guard: CellGuard) -> None:
    if guard.ok:
        return
    raise CellGuardError(
        f"cell {guard.cell_id} did not deliver its load:\n  - "
        + "\n  - ".join(guard.issues)
        + f"\n  stats: {json.dumps(guard.as_dict(), sort_keys=True)}"
    )


# --------------------------------------------------------------------------- sidecar


def parse_pod_gauges(text: str, gauges: dict[str, str] | None = None) -> dict[str, float]:
    """Sum the named Prometheus gauges over all label sets in one ``/metrics`` body.

    vLLM labels its gauges with ``model_name`` (and, on some builds, an engine index),
    so a name can appear on several lines; summing mirrors ``MetricsStore``'s per-pod
    handling. A gauge the build does not export (``num_requests_swapped`` is absent on
    the V1 engine) yields 0.0 rather than raising - it is genuinely zero there.
    """
    names = gauges or POD_INSTANT_GAUGES
    totals = {key: 0.0 for key in names}
    wanted = {metric: key for key, metric in names.items()}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        if not head:
            continue
        metric = head.split("{", 1)[0].strip()
        key = wanted.get(metric)
        if key is None:
            continue
        try:
            totals[key] += float(value)
        except ValueError:
            continue
    return totals


def _default_fetch(url: str, timeout_s: float = 2.0) -> str:
    from urllib.request import urlopen

    with urlopen(url, timeout=timeout_s) as response:
        return response.read().decode("utf-8", errors="replace")


def make_pod_metrics_sampler(
    endpoints: Sequence[str],
    *,
    fetch: Callable[[str], str] = _default_fetch,
    gauges: dict[str, str] | None = None,
) -> Callable[[int], dict]:
    """Instant queue snapshot read straight from the pods, summed across ``endpoints``.

    ``endpoints`` are full ``/metrics`` URLs. Summing across pods matches
    ``MetricsStore._aggregate_model`` (per-pod sum for the queue observables). A pod that
    fails to answer contributes nothing for that tick and is counted in ``scrape_errors``
    so a silently half-observed queue is visible in the capture.
    """
    if not endpoints:
        raise ValueError("make_pod_metrics_sampler needs at least one /metrics endpoint")

    def sample(_now_ms: int) -> dict:
        totals = {key: 0.0 for key in (gauges or POD_INSTANT_GAUGES)}
        errors = 0
        for url in endpoints:
            try:
                body = fetch(url)
            except Exception:  # noqa: BLE001 - a dead pod must not kill the sidecar
                errors += 1
                continue
            for key, value in parse_pod_gauges(body, gauges).items():
                totals[key] += value
        totals["scrape_errors"] = float(errors)
        totals["pods_scraped"] = float(len(endpoints) - errors)
        return totals

    return sample


def mark_live_grid(samples: Iterable[dict], *, grid_ms: int = LIVE_GRID_MS) -> list[dict]:
    """Tag each sample with ``on_live_grid``: the first sample inside each ``grid_ms``
    bucket, i.e. the one the gateway's boundary-aligned ticker would have written.

    With this flag a single 1 Hz capture answers both questions - what the queue really
    did, and what the controller would have seen - without a second run.
    """
    seen: set[int] = set()
    out: list[dict] = []
    for sample in sorted(samples, key=lambda s: int(s["ts_ms"])):
        bucket = int(sample["ts_ms"]) // grid_ms
        tagged = dict(sample)
        tagged["on_live_grid"] = bucket not in seen
        seen.add(bucket)
        out.append(tagged)
    return out


def windows_observing(
    samples: Sequence[dict],
    *,
    window_ms: int,
    key: str = "waiting",
    grid_only: bool = False,
    threshold: float = 0.0,
) -> tuple[int, int]:
    """(windows where ``key`` exceeded ``threshold``, total windows) over tumbling
    ``window_ms`` windows spanning the capture.

    ``grid_only`` restricts the evidence to the samples tagged ``on_live_grid``, which is
    what the live 10 s cadence would have delivered. The gap between the two counts is
    the aliasing the bursts primitive is designed to expose.
    """
    usable = [s for s in samples if s.get("ts_ms") is not None]
    if not usable:
        return 0, 0
    start = min(int(s["ts_ms"]) for s in usable)
    end = max(int(s["ts_ms"]) for s in usable) + 1
    total = 0
    hits = 0
    w = start
    while w + window_ms <= end:
        total += 1
        for s in usable:
            if grid_only and not s.get("on_live_grid", True):
                continue
            ts = int(s["ts_ms"])
            if w <= ts < w + window_ms and float(s.get(key, 0.0)) > threshold:
                hits += 1
                break
        w += window_ms
    return hits, total


# ---------------------------------------------------------------------------- driver


@dataclass
class _Sidecar:
    """1 Hz instant sampler thread."""

    sampler: Callable[[int], dict]
    interval_s: float
    now_ms: Callable[[], int]
    samples: list = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ts = self.now_ms()
            try:
                snap = self.sampler(ts)
            except Exception:  # noqa: BLE001
                snap = None
            if snap is not None:
                row = {"ts_ms": int(ts)}
                row.update({k: float(v) for k, v in snap.items()})
                self.samples.append(row)
            self._stop.wait(self.interval_s)

    def stop(self) -> list:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return mark_live_grid(self.samples)


def drive_cell_schedule(
    gateway_url: str,
    model: str,
    cell_id: str,
    segments: Sequence,
    *,
    seed: int = 1234,
    raw_path: Optional[Path] = None,
    instant_path: Optional[Path] = None,
    instant_sampler: Optional[Callable[[int], dict]] = None,
    instant_interval_s: float = DEFAULT_SIDECAR_INTERVAL_S,
    prompt_mode: str = "token_ids",
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
    stream_call: Optional[Callable] = None,
    now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    guard_kwargs: Optional[dict] = None,
) -> tuple:
    """Drive one open-loop cell from ``segments``; returns (start_ms, end_ms, guard).

    Requests fire at their Poisson-sampled wall-clock offsets via ``dispatch_open_loop``,
    NOT when a previous request returns, so offered load can exceed capacity. Overlapping
    segments superpose (each is sampled independently), which is how the bursts primitive
    lays a 2 s spike on top of its base rate and how the mixture shape runs four parallel
    token-shape streams.

    The per-request raw lines use ``r3_grid.RAW_COLUMNS`` and the instant sidecar uses the
    ``r3_grid`` sidecar schema plus ``on_live_grid``, so the offline re-windowing path is
    unchanged (pass ``--instant-sample-ms 1000`` to ``rewindow_from_raw`` to match this
    cadence).
    """
    from tre_replayer.engine.dispatcher import dispatch_open_loop
    from tre_replayer.engine.http_sender import StreamingHttpSender
    from tre_replayer.engine.schedule import build_poisson_schedule

    events = [e for e in build_poisson_schedule(segments, seed=seed) if e.model == model]
    scheduled = len(events)

    sender = StreamingHttpSender(
        gateway_url,
        stream_call=stream_call,
        max_in_flight=max_in_flight,
        prompt_mode=prompt_mode,
        now_ms=now_ms,
    )
    sidecar = None
    if instant_sampler is not None:
        sidecar = _Sidecar(sampler=instant_sampler, interval_s=instant_interval_s, now_ms=now_ms)

    start_ms = now_ms()
    instants: list = []
    if sidecar is not None:
        sidecar.start()
    try:
        report = asyncio.run(dispatch_open_loop(events, sender))
    finally:
        sender.close()
        if sidecar is not None:
            instants = sidecar.stop()
    end_ms = now_ms()

    guard = check_cell(
        cell_id,
        scheduled=scheduled,
        records=sender.records,
        p99_delay_ms=report.p99_delay_ms,
        **(guard_kwargs or {}),
    )

    if raw_path is not None:
        raw = [_raw_from_sender_record(cell_id, rec) for rec in sender.records]
        _append_jsonl(raw_path, raw)
    if instant_path is not None and instants:
        _append_jsonl(instant_path, instants)
    return start_ms, end_ms, guard


def _raw_from_sender_record(cell_id: str, record: dict) -> dict:
    """Sender row -> ``r3_grid.RAW_COLUMNS``, reusing ``r3_grid.build_raw_record`` so the
    derived fields (tpot, absolute timestamps) have exactly one implementation."""
    from types import SimpleNamespace

    from scripts import r3_grid

    res = SimpleNamespace(
        status=record.get("http_status"),
        first_token_ms=record.get("ttft_ms"),
        done_ms=record.get("e2e_ms"),
        prompt_tokens=record.get("prompt_tokens"),
        completion_tokens=record.get("completion_tokens"),
    )
    return r3_grid.build_raw_record(cell_id, int(record["actual_send_ts_ms"]), res)


def _append_jsonl(path: Path, records: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")

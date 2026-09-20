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
from dataclasses import dataclass, field, replace
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
#: Only MODEL errors count against this. A gateway shed is not the model failing, it is
#: the campaign hitting the admission ceiling, and it is handled by truncation instead.
DEFAULT_MAX_MODEL_ERROR_RATE = 0.05
#: Windows above the SLO a truncated cell must already have collected for its evidence
#: to be usable. Below that the cell was cut short before it had said anything.
DEFAULT_MIN_SLO_WINDOWS = 3
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


# ------------------------------------------------------------------------- classifier

#: A request that was served.
FAILURE_NONE = "ok"
#: The gateway rejected the request at its circuit breaker; it never reached vLLM. This
#: says nothing about the model and everything about the admission policy.
FAILURE_PROXY = "proxy"
#: The model itself failed or was too slow. vLLM queues rather than shedding, so a
#: non-2xx that carries an engine body, and any timeout, is a statement about the model.
FAILURE_MODEL = "model"

#: Statuses only the proxy can produce: they all mean "no usable answer from upstream",
#: which an engine that queues its work never needs to say. A 503 is the circuit
#: breaker; 502 and 504 are the same shed wearing different numbers.
PROXY_STATUSES = frozenset({502, 503, 504})

#: Headers Envoy sets when it is the one rejecting. Definitive when present - but the
#: measured circuit-breaker response carries NONE of them, so they can never be the only
#: test (see the pre-check signature in the module docstring of the campaign runner).
PROXY_HEADER_MARKERS = ("x-envoy-overloaded", "x-envoy-ratelimited", "x-envoy-upstream-service-time")

#: Phrases in Envoy's plain-text rejection bodies. The circuit-breaker body measured on
#: 2026-09-20 was exactly:
#:   upstream connect error or disconnect/reset before headers. reset reason: overflow
PROXY_BODY_MARKERS = (
    "upstream connect error",
    "upstream request timeout",
    "no healthy upstream",
    "reset reason",
    "overflow",
    "connection termination",
)


def _has_structured_body(body: Optional[str]) -> bool:
    """True when the body is a JSON document, i.e. the upstream answered with its own
    structured error rather than a proxy writing a plain-text rejection over it."""
    if not body:
        return False
    text = body.strip()
    if not text or text[0] not in "{[":
        return False
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def classify_failure(record: dict) -> str:
    """Attribute one request record to the model or to the gateway.

    The distinction decides what the campaign does about it, so it must not be guessed
    from the status code alone. A gateway shed means offered load exceeded the admission
    ceiling: the requests after it measure the circuit breaker, not the engine, so the
    cell is truncated and its later windows censored. A model error means the engine
    failed under the offered load, which is a real finding and fails the cell past a
    threshold.

    Rules, in order:

    * 2xx with a measured end-to-end time -> served.
    * an explicit Envoy marker header -> proxy.
    * no status at all (transport failure, timeout) -> model. vLLM queues rather than
      shedding, so a request that hung was waiting on the engine.
    * a proxy status carrying a JSON body -> model: the engine answered for itself and
      the proxy only relayed it.
    * a proxy status with a plain-text body matching a known Envoy phrase, or with no
      body at all (the connection was reset before the body could be read) -> proxy.
    * anything else -> model, because attributing an unknown failure to the proxy would
      silently exempt it from the error budget.
    """
    status = record.get("http_status")
    if status is not None and 200 <= int(status) < 300 and record.get("e2e_ms") is not None:
        return FAILURE_NONE

    raw_headers = record.get("error_headers") or {}
    headers = {str(k).lower(): str(v) for k, v in raw_headers.items()}
    if any(name in headers for name in PROXY_HEADER_MARKERS):
        return FAILURE_PROXY

    if not status:
        return FAILURE_MODEL

    if int(status) in PROXY_STATUSES:
        body = record.get("error_body") or ""
        if _has_structured_body(body):
            return FAILURE_MODEL
        lowered = body.lower()
        if not body.strip() or any(marker in lowered for marker in PROXY_BODY_MARKERS):
            return FAILURE_PROXY
    return FAILURE_MODEL


def failure_signature(record: dict) -> dict:
    """The evidence behind one classification, for the cell artifact. Keeping the body
    and headers verbatim is what lets a later reader re-judge a call this made."""
    return {
        "request_id": record.get("request_id"),
        "send_ts_ms": record.get("actual_send_ts_ms"),
        "http_status": record.get("http_status"),
        "error": record.get("error"),
        "error_body": record.get("error_body"),
        "error_headers": record.get("error_headers"),
        "e2e_ms": record.get("e2e_ms"),
        "failure_class": classify_failure(record),
    }


def count_failures(records: Sequence[dict]) -> tuple[int, int, int]:
    """(served, model errors, proxy errors) over a cell's sender records."""
    served = model_errors = proxy_errors = 0
    for record in records:
        verdict = classify_failure(record)
        if verdict == FAILURE_NONE:
            served += 1
        elif verdict == FAILURE_PROXY:
            proxy_errors += 1
        else:
            model_errors += 1
    return served, model_errors, proxy_errors


class CellGuardError(RuntimeError):
    """A cell's load was not actually delivered as specified."""


#: Default ceiling on a cell's per-pod request imbalance (max/min). None means "report
#: the balance in the artifact but never fail on it", which is what the campaign uses:
#: the number is new evidence, and turning it into a gate before anyone has looked at a
#: run's worth of it would fail cells for a reason nobody has calibrated yet.
DEFAULT_MAX_ROUTING_IMBALANCE: Optional[float] = None


def routing_balance(records: Sequence[dict]) -> dict:
    """How this cell's requests were spread over the pods that served them.

    The capacity signal the calibration fits is an *aggregate* over a model's pods, so an
    imbalanced router hides inside it: aggregate Z looks fine while one pod carries twice
    its share and its p95 explodes. This is the check that makes that visible - but only
    where the answers name a pod. On the per-model HTTPRoute path (the campaign default)
    nothing does, so ``attributed`` is 0 and ``ratio`` is None; a reader must treat that
    as "not measured", not as "balanced". See
    :func:`tre_replayer.engine.http_sender.build_request_headers`.
    """
    counts: dict[str, int] = {}
    for record in records:
        pod = record.get("target_pod")
        if pod:
            counts[str(pod)] = counts.get(str(pod), 0) + 1
    attributed = sum(counts.values())
    values = sorted(counts.values())
    ratio = None
    if values and values[0] > 0:
        ratio = values[-1] / values[0]
    return {
        "pods": len(counts),
        "attributed": attributed,
        "unattributed": max(0, len(records) - attributed),
        "per_pod": dict(sorted(counts.items())),
        "max_requests": values[-1] if values else None,
        "min_requests": values[0] if values else None,
        "imbalance_ratio": None if ratio is None else round(ratio, 4),
    }


#: Issue text prefix for the truncation-evidence verdict, so re-deciding it later can
#: find and replace exactly that issue without disturbing the dispatch-level ones.
TRUNCATION_EVIDENCE_ISSUE = "truncated before collecting enough evidence"


@dataclass(frozen=True)
class CellGuard:
    """Post-hoc verdict on one cell's dispatch."""

    cell_id: str
    scheduled: int
    sent: int
    completed: int
    model_errors: int
    proxy_errors: int
    p99_delay_ms: float
    p99_pool_wait_ms: float
    #: Set when a gateway shed cut the cell short and it jumped to its drain segment.
    truncated: bool = False
    truncated_at_offset_s: Optional[float] = None
    truncated_at_ts_ms: Optional[int] = None
    #: Scheduled requests deliberately not sent after truncation.
    censored: int = 0
    #: Windows above the SLO collected before truncation; None when not yet counted.
    slo_windows: Optional[int] = None
    min_slo_windows: int = DEFAULT_MIN_SLO_WINDOWS
    #: :func:`routing_balance` over this cell's records; None only for a guard built
    #: before the balance was computed.
    routing: Optional[dict] = None
    issues: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def errors(self) -> int:
        return self.model_errors + self.proxy_errors

    @property
    def error_rate(self) -> float:
        return 0.0 if self.sent == 0 else self.errors / self.sent

    @property
    def model_error_rate(self) -> float:
        return 0.0 if self.sent == 0 else self.model_errors / self.sent

    @property
    def proxy_error_rate(self) -> float:
        return 0.0 if self.sent == 0 else self.proxy_errors / self.sent

    def as_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "scheduled": self.scheduled,
            "sent": self.sent,
            "completed": self.completed,
            "model_errors": self.model_errors,
            "proxy_errors": self.proxy_errors,
            "model_error_rate": round(self.model_error_rate, 6),
            "proxy_error_rate": round(self.proxy_error_rate, 6),
            "p99_delay_ms": round(self.p99_delay_ms, 3),
            "p99_pool_wait_ms": round(self.p99_pool_wait_ms, 3),
            "truncated": self.truncated,
            "truncated_at_offset_s": self.truncated_at_offset_s,
            "truncated_at_ts_ms": self.truncated_at_ts_ms,
            "censored": self.censored,
            "slo_windows": self.slo_windows,
            "min_slo_windows": self.min_slo_windows,
            "routing_balance": self.routing,
            "issues": list(self.issues),
            "ok": self.ok,
        }

    def with_slo_windows(self, count: int, *, min_slo_windows: Optional[int] = None) -> "CellGuard":
        """Fold the window-level evidence count into the verdict.

        A truncated cell still passes when it had already collected at least
        ``min_slo_windows`` windows above the SLO before the gateway cut it off: the
        violation boundary was crossed and observed, which is the whole point of the
        cell. Below that it was cut off before saying anything and must be re-run.
        """
        floor_windows = self.min_slo_windows if min_slo_windows is None else int(min_slo_windows)
        issues = [issue for issue in self.issues if not issue.startswith(TRUNCATION_EVIDENCE_ISSUE)]
        if self.truncated and int(count) < floor_windows:
            issues.append(
                f"{TRUNCATION_EVIDENCE_ISSUE}: only {int(count)} window(s) above the SLO "
                f"before the gateway shed at offset {self.truncated_at_offset_s}s, "
                f"need >= {floor_windows}"
            )
        return replace(
            self,
            slo_windows=int(count),
            min_slo_windows=floor_windows,
            issues=tuple(issues),
        )


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
    max_model_error_rate: float = DEFAULT_MAX_MODEL_ERROR_RATE,
    truncated: bool = False,
    truncated_at_offset_s: Optional[float] = None,
    truncated_at_ts_ms: Optional[int] = None,
    censored: int = 0,
    slo_windows: Optional[int] = None,
    min_slo_windows: int = DEFAULT_MIN_SLO_WINDOWS,
    max_routing_imbalance: Optional[float] = DEFAULT_MAX_ROUTING_IMBALANCE,
) -> CellGuard:
    """Verdict on a dispatched cell. Pure - takes the sender's records, no network.

    ``records`` are :class:`tre_replayer.engine.http_sender.StreamingHttpSender` rows.
    Failures are attributed by :func:`classify_failure` rather than counted together:
    a model error is the engine failing under load and fails the cell past
    ``max_model_error_rate``, while a proxy shed is the campaign hitting the admission
    ceiling and is handled by truncation, because failing on it would throw away a cell
    that had already measured everything it was built to measure.
    """
    sent = len(records)
    served, model_errors, proxy_errors = count_failures(records)
    pool_waits = [float(r.get("pool_wait_ms", 0.0) or 0.0) for r in records]
    p99_pool_wait_ms = _p99(pool_waits)
    expected_sent = max(0, scheduled - int(censored))

    issues: list[str] = []
    if scheduled <= 0:
        issues.append("schedule produced 0 requests (empty or mis-filtered segments)")
    if sent == 0:
        issues.append("0 requests were sent")
    elif sent < expected_sent:
        detail = f"only {sent}/{expected_sent} scheduled requests were sent"
        if censored:
            detail += f" ({censored} more were censored by truncation)"
        issues.append(detail)
    if sent > 0 and served == 0:
        issues.append(f"0/{sent} requests completed (every send failed)")
    if sent > 0 and model_errors / sent > max_model_error_rate:
        issues.append(
            f"model error rate {model_errors / sent:.1%} > {max_model_error_rate:.1%} "
            f"({model_errors}/{sent}); these reached vLLM and failed there"
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
    routing = routing_balance(records)
    imbalance = routing["imbalance_ratio"]
    if (
        max_routing_imbalance is not None
        and imbalance is not None
        and routing["pods"] > 1
        and imbalance > max_routing_imbalance
    ):
        issues.append(
            f"per-pod request imbalance {imbalance:.2f}x > {max_routing_imbalance:.2f}x "
            f"across {routing['pods']} pod(s) ({routing['max_requests']} vs "
            f"{routing['min_requests']}); the aggregate capacity signal averages over "
            "pods, so one overloaded pod's p95 hides inside a healthy-looking Z"
        )
    guard = CellGuard(
        cell_id=cell_id,
        scheduled=scheduled,
        sent=sent,
        completed=served,
        model_errors=model_errors,
        proxy_errors=proxy_errors,
        p99_delay_ms=float(p99_delay_ms),
        p99_pool_wait_ms=p99_pool_wait_ms,
        truncated=bool(truncated),
        truncated_at_offset_s=truncated_at_offset_s,
        truncated_at_ts_ms=truncated_at_ts_ms,
        censored=int(censored),
        min_slo_windows=int(min_slo_windows),
        routing=routing,
        issues=tuple(issues),
    )
    if slo_windows is not None:
        guard = guard.with_slo_windows(slo_windows)
    return guard



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


class TruncateOnProxyShed:
    """Sender wrapper that cuts a cell short at the first gateway shed.

    Once the circuit breaker rejects, offered load is above what the gateway will admit,
    so every subsequent request measures the admission policy rather than the engine -
    and the engine, starved of exactly the surplus that would have queued, reports a
    queue that never formed. Continuing would fill the raw log with that. Instead the
    cell jumps to its drain segment: requests scheduled before ``drain_start_s`` are
    dropped and counted as censored, while the drain itself still fires so the recovery
    tail is captured. A primitive with no drain segment (``drain_start_s`` None) simply
    stops sending.

    The scan over the wrapped sender's records is safe without a lock: every wrapper
    coroutine runs on one asyncio loop and only yields at its own await, so no record can
    be appended between the await returning and the scan finishing.
    """

    def __init__(self, sender, *, drain_start_s: Optional[float] = None) -> None:
        self._sender = sender
        self._drain_start_s = drain_start_s
        self._cursor = 0
        self.truncated = False
        self.truncated_at_offset_s: Optional[float] = None
        self.truncated_at_ts_ms: Optional[int] = None
        self.first_proxy_record: Optional[dict] = None
        self.censored = 0

    @property
    def records(self) -> list:
        return self._sender.records

    async def __call__(self, request, scheduled_ts: float, actual_ts: float) -> None:
        offset_s = float(getattr(request, "scheduled_offset_s", 0.0))
        if self.truncated and (self._drain_start_s is None or offset_s < self._drain_start_s):
            self.censored += 1
            return
        await self._sender(request, scheduled_ts, actual_ts)
        self._scan(offset_s)

    def _scan(self, offset_s: float) -> None:
        if self.truncated:
            return
        records = self._sender.records
        while self._cursor < len(records):
            record = records[self._cursor]
            if classify_failure(record) == FAILURE_PROXY:
                self.truncated = True
                self.truncated_at_offset_s = offset_s
                self.truncated_at_ts_ms = record.get("actual_send_ts_ms")
                self.first_proxy_record = record
                return
            self._cursor += 1



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
    prompt_mode: Optional[str] = None,
    routing_strategy: Optional[str] = None,
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
    stream_call: Optional[Callable] = None,
    now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    guard_kwargs: Optional[dict] = None,
    truncate_on_proxy_shed: bool = False,
    drain_start_s: Optional[float] = None,
    failures_path: Optional[Path] = None,
) -> tuple:
    """Drive one open-loop cell from ``segments``; returns (start_ms, end_ms, guard).

    Requests fire at their Poisson-sampled wall-clock offsets via ``dispatch_open_loop``,
    NOT when a previous request returns, so offered load can exceed capacity. Overlapping
    segments superpose (each is sampled independently), which is how the bursts primitive
    lays a 2 s spike on top of its base rate and how the mixture shape runs four parallel
    token-shape streams.

    With ``truncate_on_proxy_shed`` the first gateway shed cuts the cell short and it
    jumps to ``drain_start_s``; see :class:`TruncateOnProxyShed`. Classified failures are
    written verbatim to ``failures_path`` so a later reader can re-judge the attribution.

    ``prompt_mode`` None means "whatever the sender defaults to"
    (:data:`tre_replayer.engine.prompts.DEFAULT_MODE`), so the default lives in exactly
    one place. ``routing_strategy`` moves the requests onto the AIBrix-routed path, which
    is the only path that reports a serving pod and is therefore the only way
    :func:`routing_balance` sees anything - at the cost of changing who picks the pod.

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

    sender_kwargs = {} if prompt_mode is None else {"prompt_mode": prompt_mode}
    sender = StreamingHttpSender(
        gateway_url,
        stream_call=stream_call,
        max_in_flight=max_in_flight,
        routing_strategy=routing_strategy,
        now_ms=now_ms,
        **sender_kwargs,
    )
    sidecar = None
    if instant_sampler is not None:
        sidecar = _Sidecar(sampler=instant_sampler, interval_s=instant_interval_s, now_ms=now_ms)

    truncator = (
        TruncateOnProxyShed(sender, drain_start_s=drain_start_s)
        if truncate_on_proxy_shed
        else None
    )

    start_ms = now_ms()
    instants: list = []
    if sidecar is not None:
        sidecar.start()
    try:
        report = asyncio.run(dispatch_open_loop(events, truncator or sender))
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
        truncated=bool(truncator and truncator.truncated),
        truncated_at_offset_s=truncator.truncated_at_offset_s if truncator else None,
        truncated_at_ts_ms=truncator.truncated_at_ts_ms if truncator else None,
        censored=truncator.censored if truncator else 0,
        **(guard_kwargs or {}),
    )
    if failures_path is not None:
        failures = [
            failure_signature(record)
            for record in sender.records
            if classify_failure(record) != FAILURE_NONE
        ]
        if failures:
            _append_jsonl(failures_path, failures)

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
        target_pod=record.get("target_pod"),
    )
    return r3_grid.build_raw_record(cell_id, int(record["actual_send_ts_ms"]), res)


def _append_jsonl(path: Path, records: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")

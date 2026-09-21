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
* :func:`count_outcomes` / :func:`goodput` - the per-cell accounting. Every request is
  one of four outcomes (ok / shed / model_error / client_timeout) and every cell reports
  offered / admitted / completed, because the campaign's headline metric is
  ``G = #(admitted AND met every SLO) / #offered``: a denominator of *offered* makes a
  rejection a loss instead of an absence, which is the only way a boundary search can
  tell "served less" apart from "refused more".
* :class:`PendingOverflowSentinel` - reads Envoy's own ``upstream_rq_pending_overflow``
  around a cell. A non-zero delta means the capture was shaped by the proxy and the run
  is invalid. It is a validity check only and never enters a control law.
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
#: The ceiling a *calibration* cell is held to, ten times tighter than the replay
#: default. theta is fitted on the relationship between offered load and latency, so a
#: generator that fires 250 ms late at the 99th percentile has not offered the load the
#: schedule says it did - the cell is then measuring the driver, and a calibration point
#: that measures the driver is worse than no point at all.
CALIBRATION_MAX_P99_DELAY_MS = 50.0
#: Only MODEL errors count against this. A gateway shed is not the model failing, it is
#: the campaign hitting the admission ceiling; a client timeout is the driver giving up.
DEFAULT_MAX_MODEL_ERROR_RATE = 0.05

#: What a cell does when the gateway sheds.
#:
#: ``truncate`` is the replay behaviour: stop offering, censor the windows after the
#: shed, and keep the earlier ones as evidence.
#:
#: ``void`` is what a calibration cell must do. Keeping only the windows from before the
#: shed keeps exactly the *healthy* part of the cell and throws away the overloaded part,
#: which biases every theta fitted on it towards health - the same mechanism that
#: produced the superseded 1718 / 1494 / 1414. A shed also means the cell never offered
#: the load it was asked to, so there is nothing to salvage: it has to be re-run.
SHED_POLICY_TRUNCATE = "truncate"
SHED_POLICY_VOID = "void"
SHED_POLICIES = (SHED_POLICY_TRUNCATE, SHED_POLICY_VOID)
DEFAULT_SHED_POLICY = SHED_POLICY_TRUNCATE
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
#: The model itself failed. vLLM queues rather than shedding, so a non-2xx that carries
#: an engine body, and any transport failure that is not the client's own deadline, is a
#: statement about the model.
FAILURE_MODEL = "model"
#: The *client* gave up before the upstream answered. Its own class: nothing is known
#: about what the engine did with the request, so counting it against the model's error
#: budget would read as an engine fault it may not be, and counting it as a shed would
#: read as an admission decision nobody made. It is recorded, never budgeted.
FAILURE_CLIENT_TIMEOUT = "client_timeout"

#: The four outcomes one request can have, under the names the cell artifacts use.
#: ``FAILURE_PROXY`` is spelled ``shed`` outward because that is what it is; the constant
#: keeps its original value so captures written before this split still read back.
FAILURE_CLASSES = (FAILURE_NONE, FAILURE_PROXY, FAILURE_MODEL, FAILURE_CLIENT_TIMEOUT)
OUTCOME_NAMES = {
    FAILURE_NONE: "ok",
    FAILURE_PROXY: "shed",
    FAILURE_MODEL: "model_error",
    FAILURE_CLIENT_TIMEOUT: "client_timeout",
}

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
    * the sender flagged ``client_timeout`` -> the client's own deadline fired. Checked
      before everything else because it is the only class the *sender* can attest to;
      every other rule is an inference from what came back, and nothing came back.
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

    if record.get("client_timeout"):
        return FAILURE_CLIENT_TIMEOUT

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
    and headers verbatim is what lets a later reader re-judge a call this made.

    ``in_flight_at_send`` travels with it: a failure is only interpretable next to how
    loaded the path was when the request left, and it is the one quantity that cannot be
    reconstructed from the raw log afterwards.
    """
    verdict = classify_failure(record)
    return {
        "request_id": record.get("request_id"),
        "send_ts_ms": record.get("actual_send_ts_ms"),
        "http_status": record.get("http_status"),
        "error": record.get("error"),
        "error_body": record.get("error_body"),
        "error_headers": record.get("error_headers"),
        "e2e_ms": record.get("e2e_ms"),
        "in_flight_at_send": record.get("in_flight_at_send"),
        "request_timeout_s": record.get("request_timeout_s"),
        "failure_class": verdict,
        "outcome": OUTCOME_NAMES[verdict],
    }


@dataclass(frozen=True)
class CellOutcomes:
    """Per-cell request accounting, in the three counts a calibration point needs.

    * ``offered`` - requests the driver actually emitted. This is the denominator of
      goodput: a request the gateway refused was still load this experiment asked the
      system to carry, and dividing by ``admitted`` instead would let a system that
      rejects half its traffic score the same as one that serves it.
    * ``admitted`` - requests that reached vLLM, i.e. everything not shed at the proxy.
    * ``completed`` - requests that came back 2xx with a measured end-to-end time.
    """

    offered: int
    admitted: int
    completed: int
    ok: int
    shed: int
    model_error: int
    client_timeout: int

    def as_dict(self) -> dict:
        return {
            "offered": self.offered,
            "admitted": self.admitted,
            "completed": self.completed,
            "ok": self.ok,
            "shed": self.shed,
            "model_error": self.model_error,
            "client_timeout": self.client_timeout,
        }


def count_outcomes(records: Sequence[dict]) -> CellOutcomes:
    """Four-way classification of a cell's sender records, plus offered/admitted/completed."""
    counts = {name: 0 for name in OUTCOME_NAMES.values()}
    for record in records:
        counts[OUTCOME_NAMES[classify_failure(record)]] += 1
    offered = len(records)
    return CellOutcomes(
        offered=offered,
        admitted=offered - counts["shed"],
        completed=counts["ok"],
        ok=counts["ok"],
        shed=counts["shed"],
        model_error=counts["model_error"],
        client_timeout=counts["client_timeout"],
    )


def count_failures(records: Sequence[dict]) -> tuple[int, int, int]:
    """(served, model errors, proxy errors) over a cell's sender records.

    Client timeouts are in none of the three: they are counted on their own in
    :func:`count_outcomes` and deliberately kept out of the model's error budget.
    """
    outcomes = count_outcomes(records)
    return outcomes.ok, outcomes.model_error, outcomes.shed


def request_meets_slo(record: dict, *, ttft_slo_ms: float, tpot_slo_ms: float) -> bool:
    """True when one served request met every latency SLO.

    A served request with no measurable TTFT or TPOT does not count as meeting the SLO:
    the campaign is fitting a threshold on latency, and "no evidence" must never be
    scored as "evidence of health".
    """
    if classify_failure(record) != FAILURE_NONE:
        return False
    ttft = record.get("ttft_ms")
    tpot = record.get("tpot_ms")
    if tpot is None:
        e2e = record.get("e2e_ms")
        completion = record.get("completion_tokens")
        if ttft is not None and e2e is not None and completion is not None and completion > 1:
            tpot = (float(e2e) - float(ttft)) / (float(completion) - 1.0)
    if ttft is None or tpot is None:
        return False
    return float(ttft) <= ttft_slo_ms and float(tpot) <= tpot_slo_ms


@dataclass(frozen=True)
class Goodput:
    """``G = #(admitted AND met every SLO) / #offered``.

    The denominator is what was offered, not what was admitted, so a rejection is a loss
    rather than an absence. That is the whole reason this replaces raw throughput as the
    campaign's headline metric: at the boundary the two diverge, and it is exactly the
    boundary the campaign is trying to locate.
    """

    offered: int
    admitted: int
    completed: int
    good: int
    ttft_slo_ms: float
    tpot_slo_ms: float

    @property
    def goodput(self) -> float:
        return 0.0 if self.offered == 0 else self.good / self.offered

    @property
    def admission_rate(self) -> float:
        return 0.0 if self.offered == 0 else self.admitted / self.offered

    def as_dict(self) -> dict:
        return {
            "offered": self.offered,
            "admitted": self.admitted,
            "completed": self.completed,
            "good": self.good,
            "goodput": round(self.goodput, 6),
            "admission_rate": round(self.admission_rate, 6),
            "ttft_slo_ms": self.ttft_slo_ms,
            "tpot_slo_ms": self.tpot_slo_ms,
        }


def goodput(
    records: Sequence[dict], *, ttft_slo_ms: float, tpot_slo_ms: float
) -> Goodput:
    outcomes = count_outcomes(records)
    good = sum(
        1
        for record in records
        if request_meets_slo(record, ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms)
    )
    return Goodput(
        offered=outcomes.offered,
        admitted=outcomes.admitted,
        completed=outcomes.completed,
        good=good,
        ttft_slo_ms=float(ttft_slo_ms),
        tpot_slo_ms=float(tpot_slo_ms),
    )


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

#: Why a cell's evidence was thrown away. Each one is a separate rule with a separate
#: test, because a void rule that silently stops firing does not fail anything - it just
#: lets a polluted cell into the fit, and theta moves without anyone seeing why.
VOID_SHED = "gateway shed"
VOID_DISPATCH_DELAY = "client dispatch delay"
VOID_MODEL_ERRORS = "model error rate"
VOID_PENDING_OVERFLOW = "envoy pending overflow"
VOID_REASONS = (VOID_SHED, VOID_DISPATCH_DELAY, VOID_MODEL_ERRORS, VOID_PENDING_OVERFLOW)


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
    #: Requests the client abandoned at its own deadline. Recorded, never budgeted.
    client_timeouts: int = 0
    #: What this cell does about a shed; see :data:`SHED_POLICIES`.
    shed_policy: str = DEFAULT_SHED_POLICY
    #: :func:`count_outcomes` for this cell, as a dict.
    outcomes: Optional[dict] = None
    #: :func:`goodput` for this cell, as a dict; None when no SLO was supplied.
    goodput: Optional[dict] = None
    #: Increase in Envoy's ``upstream_rq_pending_overflow`` across the cell. A validity
    #: sentinel only - it never enters a control law, it just says the capture is not
    #: measuring the engine.
    pending_overflow_delta: Optional[int] = None
    #: Which :data:`VOID_REASONS` fired. Non-empty means the cell's evidence must not
    #: reach the fit and the cell has to be re-run.
    void_reasons: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def voided(self) -> bool:
        """The cell produced no usable evidence, whatever else it did produce."""
        return bool(self.void_reasons)

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
            "client_timeouts": self.client_timeouts,
            "shed_policy": self.shed_policy,
            "outcomes": self.outcomes,
            "goodput": self.goodput,
            "pending_overflow_delta": self.pending_overflow_delta,
            "void_reasons": list(self.void_reasons),
            "voided": self.voided,
            "issues": list(self.issues),
            "ok": self.ok,
        }

    def with_slo_windows(self, count: int, *, min_slo_windows: Optional[int] = None) -> "CellGuard":
        """Fold the window-level evidence count into the verdict.

        Under :data:`SHED_POLICY_TRUNCATE` a truncated cell still passes when it had
        already collected at least ``min_slo_windows`` windows above the SLO before the
        gateway cut it off: the violation boundary was crossed and observed, which is the
        whole point of the cell. Below that it was cut off before saying anything.

        Under :data:`SHED_POLICY_VOID` this escape does not exist and this method cannot
        create one: a voided cell stays voided however many windows it collected, because
        the windows it collected are precisely the healthy ones.
        """
        floor_windows = self.min_slo_windows if min_slo_windows is None else int(min_slo_windows)
        issues = [issue for issue in self.issues if not issue.startswith(TRUNCATION_EVIDENCE_ISSUE)]
        if self.shed_policy == SHED_POLICY_TRUNCATE and self.truncated and int(count) < floor_windows:
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
    shed_policy: str = DEFAULT_SHED_POLICY,
    pending_overflow_delta: Optional[int] = None,
    ttft_slo_ms: Optional[float] = None,
    tpot_slo_ms: Optional[float] = None,
) -> CellGuard:
    """Verdict on a dispatched cell. Pure - takes the sender's records, no network.

    ``records`` are :class:`tre_replayer.engine.http_sender.StreamingHttpSender` rows.
    Failures are attributed by :func:`classify_failure` rather than counted together, and
    each verdict below is its own rule:

    * a **model error** is the engine failing under load. The window it lands in is a
      violation and is kept; past ``max_model_error_rate`` of the cell, the cell is void.
    * a **gateway shed** means the load never reached the engine. Under
      ``SHED_POLICY_TRUNCATE`` the cell is cut short and its earlier windows are kept;
      under ``SHED_POLICY_VOID`` the whole cell is void.
    * a **client timeout** is the driver's own deadline. Counted on its own and budgeted
      against nothing.
    * a **dispatch delay** above ``max_p99_delay_ms`` means the schedule was not offered,
      so the cell is void whatever it recorded.
    * a non-zero **pending-overflow delta** means Envoy was queueing and refusing behind
      the scenes; the capture is not of the engine and the run is void.
    """
    if shed_policy not in SHED_POLICIES:
        raise ValueError(f"unknown shed policy {shed_policy!r} (expected {SHED_POLICIES})")
    sent = len(records)
    outcomes = count_outcomes(records)
    served, model_errors, proxy_errors = outcomes.ok, outcomes.model_error, outcomes.shed
    pool_waits = [float(r.get("pool_wait_ms", 0.0) or 0.0) for r in records]
    p99_pool_wait_ms = _p99(pool_waits)
    expected_sent = max(0, scheduled - int(censored))

    issues: list[str] = []
    void_reasons: list[str] = []
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
            f"{VOID_MODEL_ERRORS} {model_errors / sent:.1%} > {max_model_error_rate:.1%} "
            f"({model_errors}/{sent}); these reached vLLM and failed there"
        )
        void_reasons.append(VOID_MODEL_ERRORS)
    if shed_policy == SHED_POLICY_VOID and proxy_errors > 0:
        issues.append(
            f"{VOID_SHED}: {proxy_errors}/{sent} request(s) were refused at the Envoy "
            "circuit breaker, so the offered load never reached the engine. Keeping the "
            "windows from before the shed would keep only the healthy part of the cell "
            "and bias theta towards health - the whole cell is void and must be re-run"
        )
        void_reasons.append(VOID_SHED)
    if p99_delay_ms > max_p99_delay_ms:
        issues.append(
            f"{VOID_DISPATCH_DELAY} p99 {p99_delay_ms:.1f}ms > {max_p99_delay_ms:.1f}ms "
            "(the driver could not keep the schedule: offered load was under-delivered, "
            "so this is not an open loop)"
        )
        void_reasons.append(VOID_DISPATCH_DELAY)
    if pending_overflow_delta is not None and int(pending_overflow_delta) > 0:
        issues.append(
            f"{VOID_PENDING_OVERFLOW}: Envoy's upstream_rq_pending_overflow rose by "
            f"{int(pending_overflow_delta)} during this cell, so requests were queued and "
            "dropped at the proxy; the capture is not of the engine"
        )
        void_reasons.append(VOID_PENDING_OVERFLOW)
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
    cell_goodput = None
    if ttft_slo_ms is not None and tpot_slo_ms is not None:
        cell_goodput = goodput(
            records, ttft_slo_ms=float(ttft_slo_ms), tpot_slo_ms=float(tpot_slo_ms)
        ).as_dict()
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
        client_timeouts=outcomes.client_timeout,
        shed_policy=shed_policy,
        outcomes=outcomes.as_dict(),
        goodput=cell_goodput,
        pending_overflow_delta=(
            None if pending_overflow_delta is None else int(pending_overflow_delta)
        ),
        void_reasons=tuple(void_reasons),
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


# ----------------------------------------------------------------- overflow sentinel

#: Envoy's per-cluster counter of requests dropped because the pending queue was full.
#: It is the proxy's own account of having refused work, independent of anything the
#: client saw, which is why it is the sentinel: a cell can look clean from the client
#: side and still have been shaped by the proxy.
PENDING_OVERFLOW_COUNTER = "upstream_rq_pending_overflow"


def parse_envoy_counters(text: str, counter: str, *, cluster_filter: str = "") -> int:
    """Sum one Envoy admin counter over the clusters whose name contains ``cluster_filter``.

    The admin ``/stats`` body is ``<name>: <value>`` per line. Summing rather than
    picking one line is deliberate: a model is served by one cluster today, but a name
    change or a second listener would otherwise make the sentinel silently read zero.
    """
    total = 0
    for line in text.splitlines():
        name, sep, value = line.partition(":")
        if not sep:
            continue
        name = name.strip()
        if not name.endswith("." + counter) and name != counter:
            continue
        if cluster_filter and cluster_filter not in name:
            continue
        try:
            total += int(value.strip())
        except ValueError:
            continue
    return total


@dataclass
class PendingOverflowSentinel:
    """Reads :data:`PENDING_OVERFLOW_COUNTER` around a cell and reports the delta.

    Strictly a validity check. It says "this capture was shaped by the proxy, throw it
    away", and it is never fed to a controller or a fit: the counter is a property of the
    admission policy, and a control law that reacted to it would be steering on the proxy
    rather than on the model.

    ``read`` is injected (``() -> str``, the admin ``/stats`` body) so tests and dry runs
    never touch the network. A read that fails yields ``None``, which is reported as
    "not measured" rather than as zero - an unread sentinel must not look like a clean one.
    """

    read: Callable[[], str]
    cluster_filter: str = ""
    counter: str = PENDING_OVERFLOW_COUNTER
    baseline: Optional[int] = None

    def sample(self) -> Optional[int]:
        try:
            body = self.read()
        except Exception:  # noqa: BLE001 - an unreachable admin port is not a cell failure
            return None
        return parse_envoy_counters(body, self.counter, cluster_filter=self.cluster_filter)

    def start(self) -> Optional[int]:
        self.baseline = self.sample()
        return self.baseline

    def delta(self) -> Optional[int]:
        """Increase since :meth:`start`, or None when either read was unavailable."""
        if self.baseline is None:
            return None
        after = self.sample()
        if after is None:
            return None
        return max(0, int(after) - int(self.baseline))


def make_envoy_stats_reader(url: str, *, fetch: Callable[[str], str] = _default_fetch) -> Callable[[], str]:
    """``() -> /stats body`` for an Envoy admin endpoint."""

    def read() -> str:
        return fetch(url)

    return read


# ------------------------------------------------------------- model-error windowing


def mark_model_error_windows(
    rows: Sequence[dict], records: Sequence[dict]
) -> list[dict]:
    """Label every window that contains a model error as an SLO violation, and keep it.

    A request the engine failed is evidence about the engine at that operating point -
    arguably the strongest evidence a window can carry - so dropping those windows would
    remove exactly the overloaded ones and pull theta towards health. It must also not be
    scored as healthy just because the failed request contributed no latency sample,
    which is what happens if nothing marks it: a window whose slowest requests all
    errored out can otherwise show a comfortable p95.

    A request is attributed to a window by its send time, because that is the operating
    point that produced the failure; a failed request often has no completion time at all.
    """
    errors: list[int] = []
    for record in records:
        if classify_failure(record) != FAILURE_MODEL:
            continue
        ts = record.get("actual_send_ts_ms", record.get("send_ts_ms"))
        if ts is not None:
            errors.append(int(ts))
    marked: list[dict] = []
    for row in rows:
        out = dict(row)
        start = int(row["window_start_ms"])
        end = int(row["window_end_ms"])
        count = sum(1 for ts in errors if start <= ts < end)
        out["model_errors"] = count
        out["slo_violated"] = bool(row.get("slo_violated")) or count > 0
        marked.append(out)
    return marked


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
    overflow_sentinel: Optional["PendingOverflowSentinel"] = None,
    records_out: Optional[list] = None,
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
    if overflow_sentinel is not None:
        overflow_sentinel.start()
    try:
        report = asyncio.run(dispatch_open_loop(events, truncator or sender))
    finally:
        sender.close()
        if sidecar is not None:
            instants = sidecar.stop()
    end_ms = now_ms()
    overflow_delta = overflow_sentinel.delta() if overflow_sentinel is not None else None

    guard = check_cell(
        cell_id,
        scheduled=scheduled,
        records=sender.records,
        p99_delay_ms=report.p99_delay_ms,
        truncated=bool(truncator and truncator.truncated),
        truncated_at_offset_s=truncator.truncated_at_offset_s if truncator else None,
        truncated_at_ts_ms=truncator.truncated_at_ts_ms if truncator else None,
        censored=truncator.censored if truncator else 0,
        pending_overflow_delta=overflow_delta,
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
    if records_out is not None:
        # The sender rows, for a caller that needs to attribute per-request outcomes to
        # windows. The raw JSONL cannot serve that: it carries neither the failure class
        # nor the send-side concurrency, by design (it is the metrics schema, not a log
        # of what the driver did).
        records_out.extend(sender.records)
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

"""SafeScale direct evidence (2026-09-29, plan B+D): the controller scrapes the vLLM
``/metrics`` of the probe's remaining pods itself instead of waiting for the gateway's
10 s Redis docs.

* **baseline** - ``safescale.baseline_delay_ms`` (1 s) after the SM confirmed the hide
  (ActionQueue ``on_hide_done`` -> :meth:`DirectEvidenceCollector.on_hide_done`; the
  delay lets the gateway's pod watch apply the hide first), every remaining pod (the
  model's awake pods minus the probe pods) is scraped once, concurrently; the cumulative TTFT /
  inter-token-latency histograms and the prompt-token sum / count are kept in the probe
  record (compact: only the bucket steps, see :func:`compact_hist`), so a restarted
  controller continues from it;
* **poll** - every ``evidence_poll_s`` (one SafeScale tick) the same pods are scraped
  again; the difference to the baseline is the evidence of ``[baseline, now]``: TTFT /
  TPOT p95 per pod (percentile mode and per-pod minimum samples of the metrics store),
  model p95 = max over the pods, ``n`` = TTFT count, mean prompt length ``L``, and the
  KV-cache fill = mean ``kv_cache_usage_perc`` of the pods of the latest scrape (the
  KV gate's aggregation, ``remaining_pods_kv_cache``);
* **failures** - a pod whose scrape fails is left out of that tick (recorded);
* **evidence gaps never commit** (2026-09-29 reviews) - a pod whose baseline is late
  (pending at the baseline scrape, the whole probe's baseline later than the planned
  time + one scrape timeout, a restart - counters backwards, a family gone or a new
  ``process_start_time_seconds``: its baseline is reset to zero, everything since is
  post-hide - or a remaining pod the cluster view lists that joined after the baseline)
  has a hole in its evidence: it is ``late`` for the rest of the probe. Its data still
  drives the immediate rollback; the probe can no longer commit. The Redis evidence is
  never used for a commit on this path (the gateway re-writes cached docs with fresh
  stamps, so their freshness cannot be trusted). The deadline commit needs every live
  pod scraped successfully in that very poll, a fresh cluster view listing no pod
  outside the evidence, and every pod's evidence reaching the deadline (see
  ``SafeScaleStateMachine._direct_outcome``).
* **coverage** - a scrape is stamped twice: when the read is sent (``started_ms``) and
  when its answer is parsed (``ts_ms``). A baseline counts from ``ts_ms`` (the latest
  moment the counters can have been read), a poll covers up to ``started_ms`` (the
  earliest), so the evidence of every pod covers ``[its baseline, the poll]`` for sure.

Scrapes run in a dedicated thread pool (stdlib ``urllib``, as the controller's other
HTTP reads; no new dependency) and are awaited asynchronously: the event loop and the
fast loop never block on them. ``scrape_timeout_s`` counts from the moment a pool thread
starts the read (time queued behind other scrapes is not a pod timeout); the queue wait
itself is bounded by one more timeout (``pool_saturated``), and a busy pool is logged.
Only samples labelled with the probe's ``model_name`` are read (a pod IP reused by
another model's pod yields ``model_mismatch``).

Metric names (vLLM 0.30, verified on a live pod): ``vllm:time_to_first_token_seconds``,
``vllm:inter_token_latency_seconds`` (old name ``vllm:time_per_output_token_seconds``;
the gateway's TPOT is the same per-token histogram), ``vllm:request_prompt_tokens``,
``vllm:kv_cache_usage_perc`` (old ``vllm:gpu_cache_usage_perc``), all with an
``engine`` label (summed / averaged over engines), and ``vllm:engine_sleep_state``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping

from tre_common.percentile import histogram_percentile
from tre_common.vllm_metrics import vllm_candidates
from tre_common.window_pods import pooled_p95_ms

LOG = logging.getLogger("tre_controller.safescale")

#: evidence_source_used values (audit / summary).
SOURCE_DIRECT = "direct"
SOURCE_REDIS_FALLBACK = "redis_fallback"
SOURCE_REDIS = "redis"

_TTFT = "time_to_first_token_seconds"
_TPOT = "inter_token_latency_seconds"
_PROMPT = "request_prompt_tokens"
_KV = "kv_cache_usage_perc"
_SLEEP_STATE = "vllm:engine_sleep_state"
#: Process start time (restart detection next to the counter reset); vLLM 0.30 exposes
#: the unprefixed prometheus_client gauge, the prefixed name is taken first if present.
_PROCESS_START = ("vllm:process_start_time_seconds", "process_start_time_seconds")
#: Requests in flight (running + waiting): "traffic" for the stalled check.
_IN_FLIGHT = ("vllm:num_requests_running", "vllm:num_requests_waiting")
#: Scrape timestamps kept in the probe record (audit), newest last.
MAX_SCRAPE_TS = 64

_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


# ------------------------------------------------------------------ parsing
@dataclass(frozen=True)
class Hist:
    """A cumulative Prometheus histogram: ``count`` and ``((le, cumulative), ...)``
    sorted by le, ``+Inf`` included."""

    count: float
    buckets: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class PodScrape:
    """What one ``/metrics`` read of a pod yields for the evidence."""

    ts_ms: int
    ttft: Hist | None = None
    tpot: Hist | None = None
    #: request_prompt_tokens (sum, count).
    prompt: tuple[float, float] | None = None
    kv_cache: float | None = None
    #: False when vllm:engine_sleep_state reports the engine not awake; None = unknown.
    awake: bool | None = None
    #: The raw TPOT family found (0.30: inter_token_latency_seconds).
    tpot_name: str | None = None
    #: ``model_name`` labels of evidence samples that were skipped (another model).
    other_models: tuple[str, ...] = ()
    #: When the read was sent (controller clock); ``ts_ms`` = when its answer was
    #: parsed. None = unknown (then ``ts_ms`` stands for both).
    started_ms: int | None = None
    #: process_start_time_seconds (None = not exposed): a change = the pod restarted.
    process_start_s: float | None = None
    #: num_requests_running + num_requests_waiting (None = not exposed).
    in_flight: float | None = None

    @property
    def coverage_end_ms(self) -> int:
        """The latest moment this read's counters surely include: the request start."""
        return int(self.started_ms) if self.started_ms is not None else int(self.ts_ms)


def parse_vllm_metrics(
    text: str, *, ts_ms: int, model_name: str | None = None, started_ms: int | None = None
) -> PodScrape:
    """The evidence families of a vLLM ``/metrics`` body (Prometheus text format).
    Label sets (engines) are summed; the KV gauge is averaged over engines. With
    ``model_name``, samples labelled with another ``model_name`` are skipped (a pod IP
    reused by another model); unlabelled samples are kept. ``started_ms`` = when the
    read was sent (``ts_ms`` = when it was answered)."""
    wanted: dict[str, tuple[str, str]] = {}
    families: dict[str, tuple[str, ...]] = {}
    for canonical in (_TTFT, _TPOT, _PROMPT):
        families[canonical] = vllm_candidates(canonical)
        for raw in families[canonical]:
            for suffix in ("_bucket", "_count", "_sum"):
                wanted[raw + suffix] = (raw, suffix)
    kv_names = vllm_candidates(_KV)
    for raw in (*kv_names, *_PROCESS_START, *_IN_FLIGHT):
        wanted[raw] = (raw, "")
    wanted[_SLEEP_STATE] = (_SLEEP_STATE, "")

    buckets: dict[str, dict[float, float]] = {}
    counts: dict[str, float] = {}
    sums: dict[str, float] = {}
    gauges: dict[str, list[float]] = {}
    awake_values: list[float] = []
    other_models: set[str] = set()
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        brace = line.find("{")
        space = line.find(" ")
        name_end = brace if 0 <= brace < space or (brace >= 0 and space < 0) else space
        if name_end <= 0:
            continue
        name = line[:name_end]
        target = wanted.get(name)
        if target is None:
            continue
        if brace == name_end:
            close = line.rfind("}")
            if close < 0:
                continue
            labels = dict(_LABEL_RE.findall(line[brace + 1 : close]))
            rest = line[close + 1 :]
        else:
            labels = {}
            rest = line[name_end:]
        if model_name is not None:
            labelled = labels.get("model_name")
            if labelled is not None and labelled != model_name:
                other_models.add(labelled)
                continue
        parts = rest.split()
        if not parts:
            continue
        try:
            value = float(parts[0])
        except ValueError:
            continue
        family, suffix = target
        if family == _SLEEP_STATE:
            if labels.get("sleep_state") == "awake":
                awake_values.append(value)
            continue
        if suffix == "_bucket":
            le_raw = labels.get("le")
            if le_raw is None:
                continue
            le = math.inf if le_raw in ("+Inf", "Inf", "inf") else float(le_raw)
            per = buckets.setdefault(family, {})
            per[le] = per.get(le, 0.0) + value
        elif suffix == "_count":
            counts[family] = counts.get(family, 0.0) + value
        elif suffix == "_sum":
            sums[family] = sums.get(family, 0.0) + value
        else:
            gauges.setdefault(family, []).append(value)

    def hist(canonical: str) -> tuple[Hist | None, str | None]:
        for raw in families[canonical]:  # newest name first
            if raw in counts and raw in buckets:
                return Hist(counts[raw], tuple(sorted(buckets[raw].items()))), raw
        return None, None

    ttft, _ = hist(_TTFT)
    tpot, tpot_name = hist(_TPOT)
    prompt = None
    for raw in families[_PROMPT]:
        if raw in counts and raw in sums:
            prompt = (sums[raw], counts[raw])
            break
    kv = None
    for raw in kv_names:
        values = gauges.get(raw)
        if values:
            kv = sum(values) / len(values)
            break
    awake = None if not awake_values else all(value >= 1.0 for value in awake_values)
    process_start = None
    for raw in _PROCESS_START:
        if gauges.get(raw):
            process_start = max(gauges[raw])
            break
    flight = [value for raw in _IN_FLIGHT for value in gauges.get(raw, ())]
    return PodScrape(ts_ms=int(ts_ms), ttft=ttft, tpot=tpot, prompt=prompt, kv_cache=kv, awake=awake,
                     tpot_name=tpot_name, other_models=tuple(sorted(other_models)),
                     started_ms=int(started_ms) if started_ms is not None else None,
                     process_start_s=process_start, in_flight=sum(flight) if flight else None)


# ------------------------------------------------------------------ compact baseline
def compact_hist(hist: Hist) -> list[Any]:
    """``[count, [[le, cumulative], ...]]`` with only the finite les where the cumulative
    count changes (the step points): the step function - and so any delta against it -
    is unchanged (``cumulative(x)`` = the last step at or below x, 0 before the first,
    ``count`` at +Inf). Keeps the baseline small and JSON-safe (no Infinity)."""
    steps: list[list[float]] = []
    previous = 0.0
    for le, cumulative in hist.buckets:
        if math.isinf(le):
            continue
        if cumulative != previous:
            steps.append([le, cumulative])
            previous = cumulative
    return [hist.count, steps]


def expand_hist(raw: Any) -> Hist | None:
    try:
        count = float(raw[0])
        steps = [(float(le), float(cumulative)) for le, cumulative in raw[1]]
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return Hist(count, tuple(sorted(steps)) + ((math.inf, count),))


def _cumulative_at(buckets: Iterable[tuple[float, float]], upper: float) -> float:
    """Cumulative count at ``upper`` of a (possibly compacted) cumulative histogram."""
    best = 0.0
    for le, cumulative in buckets:
        if le <= upper and cumulative > best:
            best = cumulative
    return best


def hist_delta(base: Hist, current: Hist) -> tuple[float, tuple[tuple[float, float], ...]] | None:
    """(count delta, cumulative bucket delta) of ``current`` since ``base``; None when a
    counter went backwards (the pod restarted: its baseline is void). Bucket delta rule =
    the metrics store's ``_bucket_delta`` (union of les, running max)."""
    if current.count < base.count:
        return None
    les = sorted({le for le, _ in base.buckets} | {le for le, _ in current.buckets})
    delta: list[tuple[float, float]] = []
    running = 0.0
    for le in les:
        now, then = _cumulative_at(current.buckets, le), _cumulative_at(base.buckets, le)
        if now < then:
            return None
        running = max(running, now - then)
        delta.append((le, running))
    return current.count - base.count, tuple(delta)


# ------------------------------------------------------------------ probe state
@dataclass(frozen=True)
class PodBaseline:
    #: When the baseline read was answered (the latest moment its counters were read).
    ts_ms: int
    ttft: Hist | None
    tpot: Hist | None
    prompt: tuple[float, float] | None
    #: process_start_time_seconds at the baseline (None = not exposed).
    process_start_s: float | None = None

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"ts_ms": self.ts_ms}
        if self.ttft is not None:
            record["ttft"] = compact_hist(self.ttft)
        if self.tpot is not None:
            record["tpot"] = compact_hist(self.tpot)
        if self.prompt is not None:
            record["prompt"] = [self.prompt[0], self.prompt[1]]
        if self.process_start_s is not None:
            record["process_start_s"] = self.process_start_s
        return record

    @classmethod
    def from_record(cls, raw: Any) -> "PodBaseline | None":
        if not isinstance(raw, Mapping):
            return None
        try:
            ts_ms = int(float(raw["ts_ms"]))
        except (KeyError, TypeError, ValueError):
            return None
        prompt = None
        if isinstance(raw.get("prompt"), (list, tuple)) and len(raw["prompt"]) == 2:
            try:
                prompt = (float(raw["prompt"][0]), float(raw["prompt"][1]))
            except (TypeError, ValueError):
                prompt = None
        process_start = None
        if raw.get("process_start_s") is not None:
            try:
                process_start = float(raw["process_start_s"])
            except (TypeError, ValueError):
                return None  # unreadable: the whole record is refused (retaken, late)
        return cls(
            ts_ms=ts_ms,
            ttft=expand_hist(raw["ttft"]) if raw.get("ttft") is not None else None,
            tpot=expand_hist(raw["tpot"]) if raw.get("tpot") is not None else None,
            prompt=prompt,
            process_start_s=process_start,
        )

    @classmethod
    def from_scrape(cls, scrape: PodScrape) -> "PodBaseline":
        return cls(ts_ms=scrape.ts_ms, ttft=scrape.ttft, tpot=scrape.tpot, prompt=scrape.prompt,
                   process_start_s=scrape.process_start_s)

    @classmethod
    def zero(cls, scrape: PodScrape, *, ts_ms: int) -> "PodBaseline":
        """The baseline of a pod whose counters went backwards (restarted after its
        baseline, i.e. after the hide): zero, so everything it counted since the restart
        is evidence. The families it no longer reports stay absent."""
        empty = Hist(0.0, ())
        return cls(
            ts_ms=int(ts_ms),
            ttft=empty if scrape.ttft is not None else None,
            tpot=empty if scrape.tpot is not None else None,
            prompt=(0.0, 0.0) if scrape.prompt is not None else None,
            process_start_s=scrape.process_start_s,
        )


@dataclass(frozen=True)
class PodDelta:
    """One pod's evidence since its own baseline, from its latest successful scrape
    (cumulative, so it stays valid when a later scrape of the pod fails)."""

    #: Coverage end: when the read was sent (its counters include everything before).
    ts_ms: int
    n: float
    ttft_p95_ms: float | None
    tpot_p95_ms: float | None
    prompt: tuple[float, float] | None
    #: The delta histograms (cumulative buckets) and counts, for the pooled p95.
    ttft_hist: tuple[tuple[float, float], ...] = ()
    tpot_hist: tuple[tuple[float, float], ...] = ()
    tpot_n: float = 0.0

    def audit(self) -> dict[str, Any]:
        return {"n": self.n, "ttft_p95_ms": self.ttft_p95_ms, "tpot_p95_ms": self.tpot_p95_ms, "ts_ms": self.ts_ms}


@dataclass(frozen=True)
class DirectWindow:
    """The remaining pods' evidence: every live pod's latest delta (baseline -> its
    latest successful scrape), ``end_ms`` = this poll."""

    start_ms: int
    end_ms: int
    pods: tuple[str, ...]
    #: pod -> why it gave no data in this poll (scrape failure, counter reset, ...).
    excluded: dict[str, str]
    #: Live pods without complete evidence: pending baseline, late baseline
    #: (``late_baseline``, for good: no commit), no successful scrape yet, or the latest
    #: older than the freshness bound.
    missing: dict[str, str]
    ttft_p95_ms: float | None
    tpot_p95_ms: float | None
    ttft_count: float
    judged_count: float
    prompt_tokens: float | None
    prompt_count: float | None
    #: Mean kv_cache_usage_perc over the pods scraped in this poll; ``kv_tail_max`` =
    #: max of the per-poll means over the hq tail of the polls (the KV gate's value).
    kv_cache: float | None
    kv_tail_max: float | None = None
    #: pod -> {"n", "ttft_p95_ms", "tpot_p95_ms", "ts_ms"} (audit).
    per_pod: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Live pods NOT scraped successfully in this very poll (pod -> reason): a deadline
    #: commit needs every live pod answered in the deciding poll.
    unanswered: dict[str, str] = field(default_factory=dict)
    #: pod -> {"cause", "lag_ms", "ts_ms"}: pods with a late baseline (see DirectState.late),
    #: live or not (a pending pod that went to sleep stays here).
    late: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: p95 of the pods' delta histograms pooled first (folded into the p95s above).
    pooled_ttft_p95_ms: float | None = None
    pooled_tpot_p95_ms: float | None = None
    #: The remaining pods of a FRESH cluster view at this poll (awake, not hidden, not
    #: the probe's); None = no fresh view (a commit then waits for one).
    view_pods: tuple[str, ...] | None = None
    #: Actual coverage of the evidence of EVERY live pod: from the latest baseline read
    #: (answered) to the earliest latest read (sent). None = a live pod without either.
    coverage_start_ms: int | None = None
    coverage_end_ms: int | None = None
    #: p95 without any minimum-samples rule (max of per pod and pooled): the ceiling's
    #: low-sample evaluation.
    low_ttft_p95_ms: float | None = None
    low_tpot_p95_ms: float | None = None
    #: num_requests_running + waiting over the pods of this poll (None = not exposed).
    in_flight: float | None = None

    @property
    def mean_prompt_tokens(self) -> float | None:
        if self.prompt_tokens is None or not self.prompt_count or self.prompt_count <= 0:
            return None
        return float(self.prompt_tokens) / float(self.prompt_count)

    @property
    def p95_available(self) -> bool:
        return self.ttft_p95_ms is not None or self.tpot_p95_ms is not None

    def audit(self) -> dict[str, Any]:
        return {
            "direct_window_start_ms": self.start_ms,
            "direct_window_end_ms": self.end_ms,
            "direct_pods": {pod: dict(values) for pod, values in sorted(self.per_pod.items())},
            "direct_excluded_pods": dict(sorted(self.excluded.items())),
            "direct_missing_pods": dict(sorted(self.missing.items())),
            "direct_unanswered_pods": dict(sorted(self.unanswered.items())),
            "direct_view_pods": list(self.view_pods) if self.view_pods is not None else None,
            "direct_in_flight": self.in_flight,
        }

    def latency_audit(self) -> dict[str, Any]:
        return {
            "evidence_start_ms": self.start_ms,
            "evidence_end_ms": self.end_ms,
            "evidence_coverage_start_ms": self.coverage_start_ms,
            "evidence_coverage_end_ms": self.coverage_end_ms,
            "latency_samples": self.ttft_count,
            "latency_samples_judged": self.judged_count,
            "mean_prompt_tokens": self.mean_prompt_tokens,
            "evidence_ttft_p95_ms": self.ttft_p95_ms,
            "evidence_tpot_p95_ms": self.tpot_p95_ms,
            "evidence_pooled_ttft_p95_ms": self.pooled_ttft_p95_ms,
            "evidence_pooled_tpot_p95_ms": self.pooled_tpot_p95_ms,
        }


@dataclass(frozen=True)
class DirectState:
    """Per-probe direct-evidence state, persisted in the probe record
    (``direct_evidence``)."""

    #: pod -> baseline; None = no pod answered the baseline scrape yet.
    baseline: dict[str, PodBaseline] | None = None
    baseline_ts_ms: int | None = None
    #: Remaining pods whose baseline scrape failed (transiently): their baseline is the
    #: first successful later scrape (their own start). They count as missing evidence.
    pending: tuple[str, ...] = ()
    #: pod -> {"reason", "ts_ms"}: out of the evidence for good (engine asleep while its
    #: baseline was pending / at the baseline: it serves nothing).
    dropped: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: pod -> ms from the hide confirmation to its baseline scrape (audit).
    baseline_lag_ms: dict[str, int] = field(default_factory=dict)
    #: pod -> {"cause", "lag_ms", "ts_ms"}: its evidence has a hole after the hide (cause
    #: ``pending_baseline`` / ``probe_baseline`` = baseline later than planned + one scrape
    #: timeout, ``counter_reset`` / ``family_vanished`` / ``process_restarted`` =
    #: restarted, baseline reset to zero, ``asleep_before_baseline`` = pending, then
    #: asleep: dropped from the polls, ``joined_after_baseline`` = listed by the cluster
    #: view but not in the evidence). For good: its data still drives the immediate
    #: rollback, never a commit.
    late: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Wall-clock ms of every poll (newest last, capped at MAX_SCRAPE_TS).
    scrapes: tuple[int, ...] = ()
    #: Consecutive scrapes (baseline attempts included) in which no pod answered.
    failed_polls: int = 0
    #: (poll ts, mean kv_cache_usage_perc of the pods of that poll), newest last.
    kv_history: tuple[tuple[int, float], ...] = ()
    #: Legacy only: the Redis-fallback switch of the previous release's records
    #: ({"reason", "ts_ms"}). Such a probe is rolled back (the Redis evidence never
    #: decides a direct-mode commit); nothing sets it any more.
    fallback: dict[str, Any] | None = None
    #: pod -> latest delta (not persisted: the next poll recomputes it).
    latest: dict[str, PodDelta] = field(default_factory=dict)
    #: The latest window (not persisted: recomputed by the next poll).
    last: DirectWindow | None = None

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {}
        if self.baseline is not None:
            record["baseline"] = {pod: base.as_record() for pod, base in sorted(self.baseline.items())}
            record["baseline_ts_ms"] = self.baseline_ts_ms
        if self.pending:
            record["pending"] = list(self.pending)
        if self.dropped:
            record["dropped"] = {pod: dict(value) for pod, value in sorted(self.dropped.items())}
        if self.baseline_lag_ms:
            record["baseline_lag_ms"] = dict(sorted(self.baseline_lag_ms.items()))
        if self.late:
            record["late"] = {pod: dict(value) for pod, value in sorted(self.late.items())}
        if self.scrapes:
            record["scrapes"] = list(self.scrapes)
        if self.failed_polls:
            record["failed_polls"] = self.failed_polls
        if self.kv_history:
            record["kv_history"] = [[ts, value] for ts, value in self.kv_history]
        if self.fallback is not None:
            record["fallback"] = dict(self.fallback)
        return record

    @classmethod
    def from_record(cls, raw: Any) -> "DirectState | None":
        """Tolerant restore: a malformed record yields None (the baseline is then taken
        again at the next tick - still post-hide), never an exception at startup."""
        if not isinstance(raw, Mapping):
            return None
        try:
            baseline = None
            if isinstance(raw.get("baseline"), Mapping):
                baseline = {}
                for pod, value in raw["baseline"].items():
                    parsed = PodBaseline.from_record(value)
                    if parsed is not None:
                        baseline[str(pod)] = parsed
            baseline_ts = int(float(raw["baseline_ts_ms"])) if raw.get("baseline_ts_ms") is not None else None
            dropped_raw = raw.get("dropped")
            dropped = {
                str(pod): dict(value) for pod, value in (dropped_raw.items() if isinstance(dropped_raw, Mapping) else ())
                if isinstance(value, Mapping)
            }
            scrapes = tuple(
                int(float(ts)) for ts in (raw.get("scrapes") if isinstance(raw.get("scrapes"), list) else ())
                if isinstance(ts, (int, float))
            )
            pending = tuple(str(pod) for pod in (raw.get("pending") if isinstance(raw.get("pending"), list) else ()))
            kv_history = tuple(
                (int(float(item[0])), float(item[1]))
                for item in (raw.get("kv_history") if isinstance(raw.get("kv_history"), list) else ())
                if isinstance(item, (list, tuple)) and len(item) == 2
            )
            fallback = dict(raw["fallback"]) if isinstance(raw.get("fallback"), Mapping) else None
            failed = int(float(raw.get("failed_polls") or 0))
            lag_raw = raw.get("baseline_lag_ms")
            lags = {
                str(pod): int(float(value))
                for pod, value in (lag_raw.items() if isinstance(lag_raw, Mapping) else ())
            }
            late_raw = raw.get("late")
            if late_raw is not None and not isinstance(late_raw, Mapping):
                raise ValueError("late")
            late = {str(pod): dict(value) if isinstance(value, Mapping) else {"cause": "unreadable"}
                    for pod, value in (late_raw or {}).items()}
        except (TypeError, ValueError, AttributeError, OverflowError):
            LOG.warning("safescale direct evidence: unreadable record ignored: %r", raw)
            return None
        if baseline is not None and isinstance(raw.get("baseline"), Mapping) \
                and len(baseline) != len(raw["baseline"]):
            # A baseline entry that does not parse: that pod would silently leave the
            # evidence. Refuse the record (the baseline is retaken, late -> no commit).
            LOG.warning("safescale direct evidence: unreadable baseline entry, record ignored: %r", raw)
            return None
        return cls(baseline=baseline, baseline_ts_ms=baseline_ts, pending=pending, dropped=dropped,
                   baseline_lag_ms=lags, late=late,
                   scrapes=scrapes, failed_polls=failed, kv_history=kv_history, fallback=fallback)

    def live_pods(self) -> tuple[str, ...]:
        """Remaining pods still in the evidence (polled every tick): baseline pods and
        pods waiting for their baseline, minus the dropped ones."""
        pods = set(self.baseline or ()) | set(self.pending)
        return tuple(sorted(pod for pod in pods if pod not in self.dropped))


@dataclass(frozen=True)
class DirectPoll:
    """One tick's scrape of a probe's live pods: pod -> PodScrape, or an error text."""

    request_id: str
    ts_ms: int
    results: Mapping[str, "PodScrape | str"]
    #: The model's remaining pods in a FRESH cluster view at this poll (awake, not
    #: hidden, not the probe's); None = no fresh view.
    view_pods: tuple[str, ...] | None = None
    #: Timer cleanup (early commit): the same poll's scrape of the probe's HIDDEN pods
    #: (vLLM running + waiting), never part of the evidence. None = not scraped.
    hidden: Mapping[str, "PodScrape | str"] | None = None
    #: Gateway in-flight requests on the hidden pods (``tre:v2:gw:inflight:<pod>``
    #: totals); None = unknown (not read, no live gateway instance, read error).
    gateway_inflight: float | None = None


def _late_entry(cause: str, *, lag_ms: int | None, ts_ms: int) -> dict[str, Any]:
    return {"cause": cause, "lag_ms": lag_ms, "ts_ms": int(ts_ms)}


def take_baseline(
    results: Mapping[str, "PodScrape | str"],
    *,
    ts_ms: int,
    previous: DirectState | None = None,
    hide_confirm_ms: int | None = None,
    late_after_ms: float | None = None,
) -> DirectState:
    """The probe's direct state from a baseline scrape: pods that answered (awake, with
    a TTFT histogram) form the baseline, pods reporting their engine asleep are dropped,
    the others (timeout, error, no IP) are pending - their baseline is their first later
    successful scrape, and they are ``late`` then. No pod answered: ``baseline`` stays
    None and ``failed_polls`` counts the attempt.

    ``hide_confirm_ms`` / ``late_after_ms``: each baseline pod's lag (scrape time - hide
    confirmation) is recorded; a lag above ``late_after_ms`` (the planned baseline time
    plus one scrape timeout: a retried / restarted-controller baseline) makes the pod
    ``late`` (``probe_baseline``) - requests it completed in [confirmation, its baseline]
    are not in the evidence."""
    state = previous or DirectState()
    baseline: dict[str, PodBaseline] = {}
    dropped = dict(state.dropped)
    pending: list[str] = []
    for pod, result in sorted(results.items()):
        reason = _unusable(result)
        if reason == "asleep":
            dropped[pod] = {"reason": "baseline_asleep", "ts_ms": int(ts_ms)}
        elif reason is not None:
            pending.append(pod)
        else:
            baseline[pod] = PodBaseline.from_scrape(result)  # type: ignore[arg-type]
    if not baseline:
        return replace(state, dropped=dropped, pending=tuple(pending), failed_polls=state.failed_polls + 1)
    lags = dict(state.baseline_lag_ms)
    late = dict(state.late)
    if hide_confirm_ms is not None:
        for pod, base in baseline.items():
            lag = int(base.ts_ms) - int(hide_confirm_ms)
            lags[pod] = lag
            if late_after_ms is not None and lag > late_after_ms:
                late[pod] = _late_entry("probe_baseline", lag_ms=lag, ts_ms=base.ts_ms)
    start = min(base.ts_ms for base in baseline.values())
    return replace(state, baseline=baseline, baseline_ts_ms=start, dropped=dropped, pending=tuple(pending),
                   failed_polls=0, baseline_lag_ms=lags, late=late)


def _unusable(result: "PodScrape | str") -> str | None:
    if isinstance(result, str):
        return result or "scrape_failed"
    if result.awake is False:
        return "asleep"
    if result.ttft is None:
        if result.other_models:
            return "model_mismatch:" + ",".join(result.other_models)
        return "no_ttft_histogram"
    return None


def _pod_delta(
    base: PodBaseline, result: PodScrape, *, percentile_mode: str, min_latency_samples: int
) -> "PodDelta | str":
    """The pod's evidence since its baseline; a string = why it cannot be differenced
    (``counter_reset``: a counter went backwards; ``family_vanished``: a family of the
    baseline is gone; ``process_restarted``: process_start_time_seconds changed - a
    restart whose counters already passed the baseline's) - the pod restarted after its
    baseline."""
    if base.process_start_s is not None:
        if result.process_start_s is None:
            return "family_vanished"
        if abs(float(result.process_start_s) - float(base.process_start_s)) > 1e-3:
            return "process_restarted"
    if base.ttft is None or result.ttft is None:
        return "family_vanished"
    ttft = hist_delta(base.ttft, result.ttft)
    if ttft is None:
        return "counter_reset"
    if base.tpot is not None and result.tpot is not None:
        tpot = hist_delta(base.tpot, result.tpot)
        if tpot is None:
            return "counter_reset"
    elif base.tpot is not None:
        return "family_vanished"
    else:
        tpot = (0.0, ())
    prompt = None
    if base.prompt is not None and result.prompt is not None:
        prompt = (result.prompt[0] - base.prompt[0], result.prompt[1] - base.prompt[1])
        if min(prompt) < 0:
            return "counter_reset"
    elif base.prompt is not None:
        return "family_vanished"
    return PodDelta(
        ts_ms=result.coverage_end_ms,
        n=ttft[0],
        ttft_p95_ms=_p95_ms(ttft[1], ttft[0], percentile_mode, min_latency_samples),
        tpot_p95_ms=_p95_ms(tpot[1], tpot[0], percentile_mode, min_latency_samples),
        prompt=prompt,
        ttft_hist=tuple(ttft[1]),
        tpot_hist=tuple(tpot[1]),
        tpot_n=float(tpot[0]),
    )


def evaluate_poll(
    state: DirectState,
    poll: DirectPoll,
    *,
    percentile_mode: str,
    min_latency_samples: int,
    fresh_ms: float,
    hq: float = 0.25,
    hide_confirm_ms: int | None = None,
) -> tuple[DirectState, DirectWindow | None]:
    """Apply one poll: pending pods that answered get their baseline (and are ``late``:
    their requests before it are lost), baseline pods that answered get a new delta, a
    pod whose counters went backwards (restarted after its baseline, so after the hide)
    gets a zero baseline - everything it counted since is evidence - and is ``late`` (the
    requests before its restart are lost); failures keep the pod's previous delta and are
    recorded. The window aggregates every live pod's latest delta (p95 = max over pods,
    n = sum, judged = n of the pods with a p95); ``missing`` lists the live pods whose
    evidence is incomplete (pending, late for good, no delta, or none newer than
    ``fresh_ms``), ``unanswered`` the live pods without a usable scrape in THIS poll.
    Returns the updated state and the window (None while no live pod has any delta)."""
    now = int(poll.ts_ms)
    baseline = dict(state.baseline or {})
    pending = list(state.pending)
    dropped = dict(state.dropped)
    latest = dict(state.latest)
    lags = dict(state.baseline_lag_ms)
    late = dict(state.late)
    excluded: dict[str, str] = {}
    unanswered: dict[str, str] = {}
    kv_values: list[float] = []
    flight: list[float] = []
    answered = False
    for pod in state.live_pods():
        result = poll.results.get(pod, "not_polled")
        reason = _unusable(result)
        if reason is not None:
            excluded[pod] = reason
            unanswered[pod] = reason
            if reason == "asleep" and pod in pending:
                # Asleep now, but whatever it served between the hide and its sleep is
                # not in any delta: out of the polls, and late (no direct commit).
                pending.remove(pod)
                dropped[pod] = {"reason": "asleep", "ts_ms": now}
                late[pod] = _late_entry(
                    "asleep_before_baseline",
                    lag_ms=(now - int(hide_confirm_ms)) if hide_confirm_ms is not None else None, ts_ms=now,
                )
            continue
        assert isinstance(result, PodScrape)
        answered = True
        if result.kv_cache is not None:
            kv_values.append(float(result.kv_cache))
        if result.in_flight is not None:
            flight.append(float(result.in_flight))
        if pod in pending:
            pending.remove(pod)
            baseline[pod] = PodBaseline.from_scrape(result)
            lag = int(result.ts_ms) - int(hide_confirm_ms) if hide_confirm_ms is not None else None
            if lag is not None:
                lags[pod] = lag
            if pod not in late:  # a joined pod keeps its cause
                late[pod] = _late_entry("pending_baseline", lag_ms=lag, ts_ms=result.ts_ms)
            excluded[pod] = "baseline_taken"
            continue
        delta = _pod_delta(baseline[pod], result, percentile_mode=percentile_mode,
                           min_latency_samples=min_latency_samples)
        if isinstance(delta, str):
            # Restarted after its baseline: reset to zero (all of it is post-hide) and
            # keep it in the evidence - never a subset commit without it.
            late[pod] = _late_entry(
                delta, lag_ms=(now - int(hide_confirm_ms)) if hide_confirm_ms is not None else None, ts_ms=now
            )
            baseline[pod] = PodBaseline.zero(result, ts_ms=baseline[pod].ts_ms)
            excluded[pod] = delta
            reset = _pod_delta(baseline[pod], result, percentile_mode=percentile_mode,
                               min_latency_samples=min_latency_samples)
            if isinstance(reset, str):  # cannot happen with a zero baseline; fail closed
                latest.pop(pod, None)
                unanswered[pod] = delta
                continue
            delta = reset
        latest[pod] = delta
    if poll.view_pods is not None:
        # 2026-09-29 review P2-c: the remaining-pod set was fixed at the baseline. A pod
        # the fresh cluster view lists now but the evidence does not know (a replica that
        # joined, a dropped sleeper that woke up) served requests outside the evidence
        # since the hide: it is polled from the next tick on (its own baseline) and late
        # for good. A dropped pod that is already late is not re-added.
        known = set(baseline) | set(pending)
        for pod in sorted(set(poll.view_pods) - known):
            if pod in dropped and pod in late:
                continue
            dropped.pop(pod, None)
            pending.append(pod)
            late[pod] = _late_entry(
                "joined_after_baseline",
                lag_ms=(now - int(hide_confirm_ms)) if hide_confirm_ms is not None else None, ts_ms=now,
            )
            excluded[pod] = "joined_after_baseline"
            unanswered[pod] = "joined_after_baseline"
    kv_history = state.kv_history
    if kv_values:
        kv_history = (kv_history + ((now, sum(kv_values) / len(kv_values)),))[-MAX_SCRAPE_TS:]
    updated = replace(
        state,
        baseline=baseline,
        pending=tuple(pending),
        dropped=dropped,
        latest=latest,
        kv_history=kv_history,
        scrapes=(state.scrapes + (now,))[-MAX_SCRAPE_TS:],
        failed_polls=0 if answered else state.failed_polls + 1,
        baseline_lag_ms=lags,
        late=late,
    )
    live = updated.live_pods()
    deltas = {pod: latest[pod] for pod in live if pod in latest}
    missing: dict[str, str] = {}
    for pod in live:
        if pod in pending:
            missing[pod] = "pending_baseline"
        elif pod in late:
            missing[pod] = "late_baseline"
        elif pod not in deltas:
            missing[pod] = f"no_data:{excluded.get(pod, 'not_polled')}"
        elif now - deltas[pod].ts_ms > fresh_ms:
            missing[pod] = f"stale:{excluded.get(pod, 'not_polled')}"
    if not deltas:
        return replace(updated, last=None), None
    ttft_p95 = [d.ttft_p95_ms for d in deltas.values() if d.ttft_p95_ms is not None]
    tpot_p95 = [d.tpot_p95_ms for d in deltas.values() if d.tpot_p95_ms is not None]
    prompts = [d.prompt for d in deltas.values() if d.prompt is not None]
    # Pooled p95 next to the per-pod maximum (2026-09-29 review F1): a pod below the
    # per-pod minimum samples - an overloaded pod completes few requests - still
    # weighs in; the minimum is applied to the pooled count.
    rule = (percentile_mode, int(min_latency_samples))
    pooled_ttft = pooled_p95_ms(((d.ttft_hist, d.n) for d in deltas.values()), rule)
    pooled_tpot = pooled_p95_ms(((d.tpot_hist, d.tpot_n) for d in deltas.values()), rule)
    ttft_p95 += [pooled_ttft] if pooled_ttft is not None else []
    tpot_p95 += [pooled_tpot] if pooled_tpot is not None else []
    total_n = float(sum(d.n for d in deltas.values()))
    judged_n = float(sum(d.n for d in deltas.values() if d.ttft_p95_ms is not None or d.tpot_p95_ms is not None))
    if pooled_ttft is not None or pooled_tpot is not None:
        judged_n = total_n  # every request is in the pooled p95
    # The ceiling's low-sample evaluation: the same max(per pod, pooled) without any
    # minimum-samples rule (every request counts, however few).
    rule0 = (percentile_mode, 0)
    low_ttft = _max_present(
        [_p95_ms(d.ttft_hist, d.n, percentile_mode, 0) for d in deltas.values()]
        + [pooled_p95_ms(((d.ttft_hist, d.n) for d in deltas.values()), rule0)]
    )
    low_tpot = _max_present(
        [_p95_ms(d.tpot_hist, d.tpot_n, percentile_mode, 0) for d in deltas.values()]
        + [pooled_p95_ms(((d.tpot_hist, d.tpot_n) for d in deltas.values()), rule0)]
    )
    # Coverage of every live pod: [latest baseline, earliest coverage end].
    coverage_start = max(baseline[pod].ts_ms for pod in live) if all(pod in baseline for pod in live) else None
    coverage_end = min(deltas[pod].ts_ms for pod in live) if all(pod in deltas for pod in live) else None
    window = DirectWindow(
        start_ms=min(baseline[pod].ts_ms for pod in deltas),
        end_ms=now,
        pods=tuple(sorted(deltas)),
        excluded={**{pod: str(value.get("reason")) for pod, value in dropped.items()}, **excluded},
        missing=missing,
        ttft_p95_ms=max(ttft_p95) if ttft_p95 else None,
        tpot_p95_ms=max(tpot_p95) if tpot_p95 else None,
        ttft_count=total_n,
        judged_count=judged_n,
        prompt_tokens=sum(p[0] for p in prompts) if prompts else None,
        prompt_count=sum(p[1] for p in prompts) if prompts else None,
        kv_cache=(sum(kv_values) / len(kv_values)) if kv_values else None,
        kv_tail_max=_kv_tail_max(kv_history, hq),
        per_pod={pod: delta.audit() for pod, delta in deltas.items()},
        unanswered={pod: reason for pod, reason in unanswered.items() if pod in live},
        late={pod: dict(value) for pod, value in late.items()},  # dropped ones included
        pooled_ttft_p95_ms=pooled_ttft,
        pooled_tpot_p95_ms=pooled_tpot,
        view_pods=tuple(sorted(poll.view_pods)) if poll.view_pods is not None else None,
        coverage_start_ms=int(coverage_start) if coverage_start is not None else None,
        coverage_end_ms=int(coverage_end) if coverage_end is not None else None,
        low_ttft_p95_ms=low_ttft,
        low_tpot_p95_ms=low_tpot,
        in_flight=sum(flight) if flight else None,
    )
    return replace(updated, last=window), window


def _max_present(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    return max(present) if present else None


def _kv_tail_max(history: tuple[tuple[int, float], ...], hq: float) -> float | None:
    """Max of the per-poll pod means over the hq tail of the polls: the KV gate's
    existing aggregation (mean over the remaining pods, max over the tail)."""
    if not history:
        return None
    hq_value = hq if hq > 0 else 0.25
    size = max(2, int(math.ceil(len(history) * hq_value))) if hq_value < 1.0 else max(2, int(hq_value))
    return max(value for _, value in history[-size:])


def _p95_ms(buckets, count: float, mode: str, min_samples: int) -> float | None:
    """Per-pod p95 (ms) with the metrics store's rule: None below ``min_samples``. A p95
    in the +Inf bucket is reported as the largest finite bucket bound (a lower bound,
    still above any threshold; keeps the records valid JSON)."""
    if count <= 0 or (min_samples > 0 and count < min_samples):
        return None
    value = histogram_percentile(buckets, 0.95, mode=mode)
    if value is None:
        return None
    if math.isinf(value):
        finite = [le for le, _ in buckets if not math.isinf(le)]
        if not finite:
            return None
        value = max(finite)
    return float(value) * 1000.0


# ------------------------------------------------------------------ scraping
#: Upper bound of a /metrics body (a vLLM pod serves ~60 KB).
MAX_METRICS_BYTES = 8 * 1024 * 1024
#: No proxy for pod IPs, whatever HTTP(S)_PROXY the controller's environment sets.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_get_text(url: str, timeout_s: float) -> str:
    with _OPENER.open(url, timeout=timeout_s) as response:  # noqa: S310 - cluster-internal
        body = response.read(MAX_METRICS_BYTES + 1)
    if len(body) > MAX_METRICS_BYTES:
        raise ValueError(f"/metrics body above {MAX_METRICS_BYTES} bytes")
    return body.decode("utf-8", errors="replace")


def metrics_url(ip: str, port: int) -> str:
    host = f"[{ip}]" if ":" in ip and not ip.startswith("[") else ip
    return f"http://{host}:{int(port)}/metrics"


class PodMetricsScraper:
    """Concurrent, bounded ``/metrics`` reads in a dedicated thread pool, never blocking
    the event loop.

    Timing (2026-09-29 review P2): ``timeout_s`` counts from the moment a pool thread
    starts the read, not from the submission - a read queued behind others (more targets
    than threads, or threads still stuck in reads that already timed out) is not a pod
    timeout. The queue wait itself is bounded by ``queue_timeout_s`` (default: one more
    ``timeout_s``): a read that never got a thread is ``pool_saturated`` (a failure like a
    timeout - it never lets a probe commit - but not blamed on the pod). A caller waits at
    most ``queue_timeout_s + timeout_s``. When the reads in flight (queued + running,
    stuck ones included) exceed ``busy_ratio`` of the pool, a WARNING
    ``safescale_scrape_pool_busy`` is logged (at most once per ``alert_interval_s``)."""

    def __init__(
        self,
        *,
        timeout_s: float,
        fetch: Callable[[str, float], str] = http_get_text,
        clock_ms: Callable[[], int] | None = None,
        max_workers: int = 16,
        queue_timeout_s: float | None = None,
        busy_ratio: float = 0.75,
        alert_interval_s: float = 60.0,
    ) -> None:
        self._timeout_s = float(timeout_s)
        self._queue_timeout_s = float(timeout_s if queue_timeout_s is None else queue_timeout_s)
        self._fetch = fetch
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._max_workers = int(max_workers)
        self._busy_ratio = float(busy_ratio)
        self._alert_interval_s = float(alert_interval_s)
        self._last_alert: float | None = None
        self._lock = threading.Lock()
        self._occupied = 0
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="safescale-scrape")

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def occupancy(self) -> int:
        """Reads submitted and not finished yet (queued + running)."""
        with self._lock:
            return self._occupied

    def _release(self, _future: Any) -> None:
        with self._lock:
            self._occupied -= 1

    def _read(self, url: str, model_name: str | None) -> PodScrape:
        # Stamped when the request is sent AND when it is answered: a poll's evidence
        # surely covers up to the send time, a baseline counts from the answer time.
        started = int(self._clock_ms())
        text = self._fetch(url, self._timeout_s)
        return parse_vllm_metrics(text, ts_ms=int(self._clock_ms()), model_name=model_name, started_ms=started)

    def _check_busy(self, incoming: int) -> None:
        occupied = self.occupancy()
        if occupied + incoming <= self._busy_ratio * self._max_workers:
            return
        now = time.monotonic()
        if self._last_alert is not None and now - self._last_alert < self._alert_interval_s:
            return
        self._last_alert = now
        LOG.warning(
            '{"event": "safescale_scrape_pool_busy", "in_flight": %d, "incoming": %d, "max_workers": %d, '
            '"busy_ratio": %s}', occupied, incoming, self._max_workers, self._busy_ratio,
        )

    async def scrape(
        self, targets: Mapping[str, str | None], *, model_name: str | None = None
    ) -> dict[str, "PodScrape | str"]:
        loop = asyncio.get_running_loop()
        self._check_busy(sum(1 for url in targets.values() if url))

        def started_now(started: asyncio.Future, ts: float) -> None:
            if not started.done():
                started.set_result(ts)

        async def one(pod: str, url: str | None) -> tuple[str, "PodScrape | str"]:
            if not url:
                return pod, "no_endpoint"
            started: asyncio.Future = loop.create_future()

            def run() -> PodScrape:
                loop.call_soon_threadsafe(started_now, started, time.monotonic())
                return self._read(url, model_name)

            with self._lock:
                self._occupied += 1
            try:
                job = self._executor.submit(run)
            except RuntimeError:  # the pool is shut down (app closing)
                with self._lock:
                    self._occupied -= 1
                return pod, "scraper_closed"
            job.add_done_callback(self._release)
            try:
                began = await asyncio.wait_for(asyncio.shield(started), timeout=self._queue_timeout_s)
            except asyncio.TimeoutError:
                if job.cancel():
                    return pod, "pool_saturated"
                began = time.monotonic()  # it just started: time it from now
            remaining = max(0.0, self._timeout_s - (time.monotonic() - began))
            try:
                return pod, await asyncio.wait_for(asyncio.wrap_future(job), timeout=remaining)
            except asyncio.TimeoutError:
                return pod, "timeout"
            except Exception as exc:  # noqa: BLE001 - recorded per pod
                return pod, f"error:{type(exc).__name__}"

        pairs = await asyncio.gather(*(one(pod, url) for pod, url in sorted(targets.items())))
        return dict(pairs)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class DirectEvidenceCollector:
    """Takes the baselines (``baseline_delay_ms`` after the hide confirmation, else at
    the next tick) and the per-tick polls of the probes on the direct path.

    ``targets(model, exclude)`` -> ``{pod: url | None}`` of the model's remaining pods
    (awake, not hidden, not in ``exclude``); ``urls(model, pods)`` -> the URL of each
    named pod (None = unknown). The app builds both from the SM cluster view."""

    def __init__(
        self,
        safescale: Any,
        scraper: PodMetricsScraper,
        targets: Callable[[str, tuple[str, ...]], Mapping[str, str | None]],
        *,
        poll_ms: float,
        urls: Callable[[str, tuple[str, ...]], Mapping[str, str | None]] | None = None,
        clock_ms: Callable[[], int] | None = None,
        hidden_scrape: bool = False,
        gateway_inflight: Callable[[tuple[str, ...]], float | None] | None = None,
    ) -> None:
        """``hidden_scrape`` / ``gateway_inflight`` (timer cleanup, early commit): each
        poll also reads the probe's hidden pods' ``/metrics`` (running + waiting) and
        their gateway in-flight count (``gateway_inflight(pods)``, None = unknown)."""
        self._safescale = safescale
        self._hidden_scrape = bool(hidden_scrape)
        self._gateway_inflight = gateway_inflight
        self._scraper = scraper
        self._targets = targets
        self._urls = urls or (lambda model, pods: {pod: dict(targets(model, ())).get(pod) for pod in pods})
        self._poll_ms = float(poll_ms)
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._inflight: set[str] = set()
        self._tasks: set[asyncio.Task] = set()

    def on_hide_done(self, model: str, pods: tuple[str, ...]) -> bool:
        """ActionQueue ``on_hide_done``: anchor the probe (``mark_hidden``) and schedule
        its baseline scrape at the planned time (confirmation + ``baseline_delay_ms``,
        asynchronously, on the running loop)."""
        anchored = self._safescale.mark_hidden(model, pods=pods)
        if anchored:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return anchored  # no loop (sync callers / tests): the next tick takes it
            task = loop.create_task(self._baseline_when_planned(model))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        return anchored

    async def _baseline_when_planned(self, model: str) -> int:
        planned_of = getattr(self._safescale, "direct_baseline_planned_ms", None)
        for _ in range(64):  # bounded: a clock jumping backwards cannot pin the task
            planned = planned_of(model) if callable(planned_of) else None
            if planned is None:
                break
            wait_ms = int(planned) - int(self._clock_ms())
            if wait_ms <= 0:
                break
            await asyncio.sleep(wait_ms / 1000.0)
        return await self.take_baselines(only_model=model)

    async def take_baselines(self, *, only_model: str | None = None) -> int:
        """Baseline scrape of every due probe (planned time reached), concurrently. A
        probe without targets (no fresh cluster view, no pod IPs) is retried at the next
        tick - its pods are then ``late`` (no commit); one that never gets a baseline is
        rolled back at its deadline (``evidence_incomplete:no_baseline``)."""
        due = [
            probe for probe in self._safescale.direct_baseline_due(now_ms=int(self._clock_ms()))
            if (only_model is None or probe.model == only_model) and probe.request_id not in self._inflight
        ]
        if not due:
            return 0
        for probe in due:
            self._inflight.add(probe.request_id)
        try:
            jobs = []
            for probe in due:
                try:
                    targets = dict(self._targets(probe.model, tuple(probe.pods)))
                except Exception:  # noqa: BLE001 - retried next tick
                    LOG.exception("safescale direct targets of %s unavailable", probe.model)
                    targets = {}
                if targets:
                    jobs.append((probe, targets))
            results = await asyncio.gather(
                *(self._scraper.scrape(targets, model_name=probe.model) for probe, targets in jobs)
            )
            ts_ms = int(self._clock_ms())
            taken = 0
            for (probe, _), result in zip(jobs, results):
                if self._safescale.set_direct_baseline(probe.model, request_id=probe.request_id,
                                                       results=result, ts_ms=ts_ms):
                    taken += 1
            return taken
        except Exception:  # noqa: BLE001 - the next tick retries; the deadline falls back
            LOG.exception("safescale direct baselines failed")
            return 0
        finally:
            for probe in due:
                self._inflight.discard(probe.request_id)

    async def poll(self) -> dict[str, DirectPoll]:
        """One tick: missing baselines first, then one scrape of every probe's live pods
        (baseline + pending) whose last poll (or baseline) is at least ~half a poll
        period old. Each poll carries the remaining pods of the fresh cluster view
        (``view_pods``; None without a fresh view): a pod outside the evidence is late."""
        await self.take_baselines()
        polls: dict[str, DirectPoll] = {}
        now = int(self._clock_ms())
        jobs = []
        for probe in self._safescale.direct_poll_due():
            state = probe.direct
            last = max([state.baseline_ts_ms or 0, *state.scrapes]) if state is not None else 0
            if now - last < 0.5 * self._poll_ms:
                continue  # a baseline / poll this young adds nothing (and n ~ 0)
            pods = state.live_pods()
            try:
                urls = dict(self._urls(probe.model, pods))
            except Exception:  # noqa: BLE001 - recorded per pod as no_endpoint
                LOG.exception("safescale direct pod URLs of %s unavailable", probe.model)
                urls = {}
            try:
                # {} = no fresh view (or no remaining pod): unknown, a commit waits.
                view = tuple(sorted(self._targets(probe.model, tuple(probe.pods)))) or None
            except Exception:  # noqa: BLE001 - no view this tick
                LOG.exception("safescale direct remaining pods of %s unavailable", probe.model)
                view = None
            hidden: dict[str, str | None] = {}
            if self._hidden_scrape:
                try:
                    hidden = {pod: url for pod, url in dict(self._urls(probe.model, tuple(probe.pods))).items()
                              if pod not in pods}
                except Exception:  # noqa: BLE001 - no hidden-pod data: no early commit
                    LOG.exception("safescale hidden pod URLs of %s unavailable", probe.model)
                    hidden = {}
            jobs.append((probe, {pod: urls.get(pod) for pod in pods}, view, hidden))
        if not jobs:
            return polls
        # One scrape per probe: the remaining pods (evidence) and the hidden pods (early
        # commit drain check) read concurrently, split again below.
        results = await asyncio.gather(
            *(self._scraper.scrape({**targets, **hidden}, model_name=probe.model)
              for probe, targets, _, hidden in jobs)
        )
        gateway = await asyncio.gather(*(self._read_gateway_inflight(probe, hidden) for probe, _, _, hidden in jobs))
        ts_ms = int(self._clock_ms())
        for (probe, targets, view, hidden), result, inflight in zip(jobs, results, gateway):
            result = dict(result)
            polls[probe.model] = DirectPoll(
                request_id=probe.request_id, ts_ms=ts_ms,
                results={pod: value for pod, value in result.items() if pod in targets},
                view_pods=view,
                hidden={pod: result[pod] for pod in hidden if pod in result} if self._hidden_scrape else None,
                gateway_inflight=inflight,
            )
        return polls

    async def _read_gateway_inflight(self, probe: Any, hidden: Mapping[str, str | None]) -> float | None:
        if not self._hidden_scrape or self._gateway_inflight is None or not hidden:
            return None
        try:
            value = await asyncio.to_thread(self._gateway_inflight, tuple(probe.pods))
        except Exception:  # noqa: BLE001 - unknown: no early commit
            LOG.warning("safescale gateway in-flight of %s unavailable", probe.model, exc_info=True)
            return None
        return None if value is None else float(value)

    def close(self) -> None:
        """App shutdown: cancel the scheduled baselines and shut the scrape pool down."""
        for task in list(self._tasks):
            task.cancel()
        close = getattr(self._scraper, "close", None)
        if callable(close):
            close()


class GatewayInflightReader:
    """Gateway in-flight requests on a set of pods (timer cleanup, early commit): the
    sum of the ``total`` fields of ``tre:v2:gw:inflight:<pod>`` (one field per gateway
    plugin instance, written by the transparent-sleep coordination). Conservative: a
    field of any instance counts (also a stale one), and the answer is unknown (None)
    when no plugin instance is registered (``tre:v2:gw:instances`` empty: nothing
    writes the counts), on a read error or an unparsable field."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    def __call__(self, pods: tuple[str, ...]) -> float | None:
        from tre_common import rediskeys

        try:
            if int(self._redis.zcard(rediskeys.GW_INSTANCES_KEY) or 0) <= 0:
                return None
            total = 0.0
            for pod in pods:
                for raw in (self._redis.hgetall(rediskeys.gw_inflight_key(pod)) or {}).values():
                    text = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
                    payload = json.loads(text)
                    value = float(payload["total"])
                    if not math.isfinite(value) or value < 0:
                        return None
                    total += value
            return total
        except Exception:  # noqa: BLE001 - unknown: no early commit
            LOG.warning("gateway in-flight read failed", exc_info=True)
            return None


def cluster_view_targets(view_getter: Callable[[], Any], *, port: int):
    """``targets(model, exclude)`` over the SM cluster view: the model's awake, not
    hidden bindings minus ``exclude`` (the probe pods), URL from the fleet state's pod
    IP (None when the view has no IP for the pod). ``view_getter`` should return only a
    FRESH view (the app passes ``ClusterViewBox.fresh``): the remaining-pod set decides
    whose evidence a commit needs. No (fresh) view -> {} (retried)."""

    def targets(model: str, exclude: tuple[str, ...]) -> dict[str, str | None]:
        view = view_getter()
        if view is None:
            return {}
        ips = getattr(view, "pod_ips", None) or {}
        out: dict[str, str | None] = {}
        for pod in _remaining_of(view, model, exclude):
            ip = ips.get(pod)
            out[pod] = metrics_url(ip, port) if ip else None
        return out

    return targets


def _remaining_of(view: Any, model: str, exclude: Iterable[str]) -> tuple[str, ...]:
    skip = set(exclude)
    return tuple(sorted(
        binding.serve_id for binding in getattr(view, "bindings", ()) or ()
        if binding.model == model and binding.awake and not getattr(binding, "hidden", False)
        and binding.serve_id not in skip
    ))


def cluster_view_remaining(view_getter: Callable[[], Any]):
    """``remaining(model, exclude)`` -> the model's awake, not hidden pods minus
    ``exclude`` (the probe pods) of a FRESH view, None without one: the pods whose
    evidence a Redis-mode commit needs (``view_getter`` = ``ClusterViewBox.fresh``)."""

    def remaining(model: str, exclude: tuple[str, ...]) -> tuple[str, ...] | None:
        view = view_getter()
        return None if view is None else _remaining_of(view, model, exclude)

    return remaining


def cluster_view_urls(view_getter: Callable[[], Any], *, port: int):
    """``urls(model, pods)``: the URL of each named pod from the cluster view's pod IPs
    (whatever its state; None = unknown)."""

    def urls(model: str, pods: tuple[str, ...]) -> dict[str, str | None]:
        view = view_getter()
        ips = (getattr(view, "pod_ips", None) or {}) if view is not None else {}
        return {pod: (metrics_url(ips[pod], port) if ips.get(pod) else None) for pod in pods}

    return urls

"""What a baseline policy sees on one tick: the frozen snapshot contract.

The shell (``tre_baselines.loop``) builds one :class:`ClusterSnapshot` per tick from
three sources and hands it to ``Policy.decide``:

* the service manager's ``GET /v2/state`` (awake count per model, which bindings are
  awake and not hidden, their node and GPUs),
* each awake, routable pod's own ``/metrics`` (scraped directly, not the gateway's 10 s
  Redis docs), mapped onto the standardized :data:`COUNTER_KEYS`,
* the gateway request-event stream ``tre:v2:bl:req:<model>`` (events new since the
  previous tick) and the replay start marker ``tre:v2:bl:replay_t0``.

Unknown is not idle. A missing gauge is ``None`` (never 0); a pod the SM calls serving
that could not be scraped is in ``unscraped``; the event stream says since when it has
been read without a gap (``events_since_ms``), how many requests it shows in flight on
each pod (``tracked_inflight``) and whether that covers everything the engine runs since
the last gap (``events_cover``). :func:`evidence_gaps` turns these into the reasons a
snapshot cannot prove that a model's load is low; a policy (and the shell, as a
backstop) never scales a model down while that list is non-empty.

Every timestamp is on the **Redis server clock** (``TIME`` for ``now_ms`` /
``scraped_at_ms``, the millisecond part of the stream entry ID for ``ts_ms``). The two
nodes' wall clocks differ by minutes, so nothing here uses a local clock.

All types are frozen and hold only plain data, so a policy can keep references to them
across ticks and a recorded snapshot sequence can be replayed deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional

#: Standardized cumulative counter keys of :attr:`PodSnapshot.counters`. The shell maps
#: the vLLM family names onto them (``tre_baselines.sources.COUNTER_SOURCES``); a key is
#: absent when the pod does not export it. Latencies are in seconds (vLLM units).
COUNTER_KEYS = (
    "gen_tokens",      # vllm:generation_tokens_total
    "prompt_tokens",   # vllm:prompt_tokens_total
    "itl_sum",         # vllm:inter_token_latency_seconds_sum
    "itl_count",       # vllm:inter_token_latency_seconds_count
    "ttft_sum",        # vllm:time_to_first_token_seconds_sum
    "ttft_count",      # vllm:time_to_first_token_seconds_count
    "e2e_sum",         # vllm:e2e_request_latency_seconds_sum
    "e2e_count",       # vllm:e2e_request_latency_seconds_count
    "req_success",     # vllm:request_success_total, summed over finished_reason
)

EventKind = Literal["arr", "ft", "done"]
REISSUE_KINDS = ("none", "continued", "retried")


@dataclass(frozen=True)
class PodSnapshot:
    """One awake, routable pod as scraped on this tick."""

    pod: str
    model: str
    node: Optional[str]
    gpu_ids: tuple[int, ...]
    #: vllm:num_requests_running / vllm:num_requests_waiting (gauges); None when the pod
    #: did not export a finite value (unknown, not 0).
    running: Optional[float]
    waiting: Optional[float]
    #: vllm:kv_cache_usage_perc in [0, 1]; None when the pod does not export it.
    kv_usage: Optional[float]
    #: Cumulative counters keyed by :data:`COUNTER_KEYS`. They reset when the engine
    #: restarts: a policy differencing them must treat a decrease as a reset.
    counters: Mapping[str, float]
    #: From vllm:cache_config_info labels; None when absent.
    num_gpu_blocks: Optional[int]
    block_size: Optional[int]
    #: Redis server clock (ms) when the scrape round started.
    scraped_at_ms: int
    #: Requests the gateway event stream shows on this pod that may have been in flight
    #: during the scrape; None when the source does not track events.
    tracked_inflight: Optional[int] = None
    #: True when every request in flight here is known from the event stream: the pod was
    #: verified (running + waiting covered by tracked requests) since the model's last
    #: event gap, so the cohort that predates the gap has drained. None = not tracked.
    events_cover: Optional[bool] = None

    @property
    def queued(self) -> Optional[float]:
        """running + waiting, or None when either gauge is unknown."""
        if self.running is None or self.waiting is None:
            return None
        return self.running + self.waiting


@dataclass(frozen=True)
class RequestEvent:
    """One gateway request event (stream ``tre:v2:bl:req:<model>``)."""

    kind: EventKind
    model: str
    pod: Optional[str]
    req_id: str
    #: Millisecond part of the stream entry ID: the authoritative (Redis) clock.
    ts_ms: int
    in_tokens: Optional[int]
    #: Where in_tokens came from ("header" | "estimate" | ...); None when unknown.
    in_src: Optional[str]
    max_tokens: Optional[int]
    out_tokens: Optional[int]
    status: Optional[str]
    #: "none" | "continued" | "retried" (a reissued request; policies usually skip it).
    reissue: str = "none"
    #: The stream entry ID, kept for tie-breaking and debugging.
    entry_id: str = ""


@dataclass(frozen=True)
class ModelSnapshot:
    """One model on this tick. Bounds and SLOs come from the registry."""

    model: str
    #: Awake bindings according to the service manager (the actuator's own count).
    awake: int
    #: Clamp range the shell applies to every decision: registry ``min_replicas`` and
    #: the scaling cap ``max_awake_replicas`` (else ``max_replicas``).
    min_replicas: int
    max_replicas: int
    #: GPUs one replica occupies (registry ``tp_size``).
    gpus_per_replica: int
    #: Fixed-arm SLOs (registry ``slo.ttft_p95_ms`` / ``slo.tpot_p95_ms``).
    ttft_slo_ms: float
    tpot_slo_ms: float
    #: vLLM ``--max-num-seqs`` from the registry; None = engine default.
    max_num_seqs: Optional[int]
    #: Awake + routable pods that were scraped successfully.
    pods: tuple[PodSnapshot, ...]
    #: Events new since the previous tick, in stream order.
    events: tuple[RequestEvent, ...]
    #: Awake + routable pods whose scrape failed this tick (omitted from ``pods``).
    unscraped: tuple[str, ...] = ()
    #: The per-request SLO definition (``tre_common.slo_labels.LabelDefinition``), for a
    #: policy that wants the length-dependent TTFT SLO; None when not available.
    slo: Any = None
    #: Redis-clock ms since which this model's event stream has been read without a known
    #: gap (shell start, a trimmed / recreated stream, a read backlog). None = no event
    #: history (e.g. the source does not read events, or the gap is this tick).
    events_since_ms: Optional[int] = None
    #: Sleeping bindings of this model the SM could wake right now without a donor: all
    #: their GPUs ``wakeable`` in ``/v2/state`` ``gpus[]`` (GPU-disjoint count). None =
    #: unknown (the SM state has no ``gpus[]``).
    wakeable_slots: Optional[int] = None


@dataclass(frozen=True)
class ReplayInfo:
    """Replay start marker written by the campaign (``tre:v2:bl:replay_t0``)."""

    t0_ms: int
    trace_path: str
    #: The replay's ``--seed`` (segment traces are re-scheduled from it); None = not given.
    seed: Optional[int] = None


@dataclass(frozen=True)
class ClusterSnapshot:
    #: Redis server clock (ms) at the start of the tick.
    now_ms: int
    #: Nominal tick period (s).
    tick_s: float
    models: Mapping[str, ModelSnapshot]
    replay: Optional[ReplayInfo] = None
    #: Tick sequence number (0-based, per shell process).
    tick: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)


#: A scrape older than this many ticks is stale (the pod's state is unknown now).
STALE_SCRAPE_TICKS = 2.0


def evidence_gaps(
    ms: ModelSnapshot,
    now_ms: int,
    tick_s: float,
    *,
    events: bool = False,
    history_s: float = 0.0,
) -> tuple[str, ...]:
    """Why ``ms`` cannot prove that the model's load is low (empty = evidence complete).

    * ``unscraped``: a pod the SM calls serving was not scraped;
    * ``missing_gauges``: a scraped pod lacks running or waiting;
    * ``stale_scrape``: a pod's scrape is older than :data:`STALE_SCRAPE_TICKS` ticks;
    * with ``events`` (a policy that reads the request-event stream):
      ``no_event_history``: the stream has not been read gap-free for ``history_s``
      (shell start, trimmed stream, backlog); ``event_gap``: a pod may still run requests
      from before the last gap (``PodSnapshot.events_cover`` not True). The requests in
      flight at the gap are the cohort; it has drained when the engine's running + waiting
      is covered by tracked requests (state, not a timer).

    Scale-up may still use whatever evidence is present; only a scale-down needs this
    list to be empty.
    """
    gaps: list[str] = []
    if ms.unscraped:
        gaps.append("unscraped")
    if any(p.queued is None for p in ms.pods):
        gaps.append("missing_gauges")
    stale_ms = STALE_SCRAPE_TICKS * float(tick_s) * 1000.0
    if any(now_ms - p.scraped_at_ms > stale_ms for p in ms.pods):
        gaps.append("stale_scrape")
    if events:
        since = ms.events_since_ms
        if since is None or now_ms - since < float(history_s) * 1000.0:
            gaps.append("no_event_history")
        if any(p.events_cover is not True for p in ms.pods):
            gaps.append("event_gap")
    return tuple(gaps)

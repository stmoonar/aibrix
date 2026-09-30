"""What a baseline policy sees on one tick: the frozen snapshot contract.

The shell (``tre_baselines.loop``) builds one :class:`ClusterSnapshot` per tick from
three sources and hands it to ``Policy.decide``:

* the service manager's ``GET /v2/state`` (awake count per model, which bindings are
  awake and not hidden, their node and GPUs),
* each awake, routable pod's own ``/metrics`` (scraped directly, not the gateway's 10 s
  Redis docs), mapped onto the standardized :data:`COUNTER_KEYS`,
* the gateway request-event stream ``tre:v2:bl:req:<model>`` (events new since the
  previous tick) and the replay start marker ``tre:v2:bl:replay_t0``.

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
    #: vllm:num_requests_running / vllm:num_requests_waiting (gauges).
    running: float
    waiting: float
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


@dataclass(frozen=True)
class ReplayInfo:
    """Replay start marker written by the campaign (``tre:v2:bl:replay_t0``)."""

    t0_ms: int
    trace_path: str


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

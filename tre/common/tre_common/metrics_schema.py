from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PodWindowMetrics:
    pod: str
    prompt_tokens: float | None
    generation_tokens: float | None
    avg_waiting: float
    avg_running: float
    avg_swapping: float
    kv_cache_hit_rate: float
    ttft_p95_ms: float | None
    tpot_p95_ms: float | None
    e2e_p95_ms: float | None
    request_count: float | None = None
    token_counter_reset: bool = False
    #: Timestamps (ms) of the gateway instant samples read for this window. Provenance for
    #: the phase-aligned sampler's freshness check (a window must hold every
    #: SCRAPE_INTERVAL_MS tick, the newest one at window_end). Excluded from equality.
    instant_ticks_ms: tuple[int, ...] = field(default=(), compare=False)
    #: Window average of vLLM ``gpu_cache_usage_perc`` (KV-cache fill, 0..1) from the
    #: gateway instant docs; None when no doc carries it. Read only by the SafeScale
    #: KV-cache guard (v1 avg_gpu_cache_norm); excluded from equality so the existing
    #: window/golden comparisons are unaffected.
    gpu_cache_usage: float | None = field(default=None, compare=False)
    #: Window means (ms) and sample counts of TTFT / TPOT from the histogram sum/count
    #: deltas; the SafeScale window falls back to them when a p95 is missing (v1
    #: start_hidden_probe). Excluded from equality like the fields above.
    ttft_avg_ms: float | None = field(default=None, compare=False)
    ttft_count: float | None = field(default=None, compare=False)
    tpot_avg_ms: float | None = field(default=None, compare=False)
    tpot_count: float | None = field(default=None, compare=False)


@dataclass(frozen=True)
class ModelWindowMetrics:
    model: str
    window_start_ms: int
    window_end_ms: int
    prompt_tokens: float | None
    generation_tokens: float | None
    avg_waiting: float
    avg_running: float
    avg_swapping: float
    kv_cache_hit_rate: float
    ttft_p95_ms: float | None
    tpot_p95_ms: float | None
    e2e_p95_ms: float | None
    routable_pods: int
    assigned_replicas: int
    per_pod: dict[str, PodWindowMetrics]
    request_count: float | None = None
    token_counter_reset: bool = False
    #: Union of the per-pod ``instant_ticks_ms`` (see :class:`PodWindowMetrics`).
    instant_ticks_ms: tuple[int, ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class MetricsSnapshot:
    ts_ms: int
    models: dict[str, ModelWindowMetrics]
    stale: bool

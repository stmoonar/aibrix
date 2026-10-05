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
    #: Window average of vLLM ``kv_cache_usage_perc`` (``gpu_cache_usage_perc`` before
    #: vLLM 0.11; KV-cache fill, 0..1) from the gateway instant docs; None when no doc
    #: carries it. Read only by the SafeScale
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
    #: The window's e2e-latency histogram delta, UNGATED: cumulative buckets
    #: ``((upper_s, count), ...)`` ascending, and its observation count. The model
    #: e2e p95 is computed from the pods' histograms merged BEFORE the
    #: minimum-samples gate (``tre_common.window_pods``), so a model whose pods each
    #: see a few requests at low load still gets a p95. None = no histogram.
    e2e_hist: tuple[tuple[float, float], ...] | None = field(default=None, compare=False)
    e2e_hist_count: float | None = field(default=None, compare=False)
    #: Timestamp (ms, the gateway's boundary stamp) of the first histogram doc the TTFT
    #: delta was taken from: the baseline doc before the window when there is one, else
    #: the window's first doc. None = no TTFT histogram. Provenance for the SafeScale
    #: evidence clock check; excluded from equality.
    hist_first_ts_ms: int | None = field(default=None, compare=False)
    #: Timestamp (ms) of the last histogram doc the TTFT delta ends at (SafeScale
    #: evidence completeness: a pod whose docs stop early has a hole at the end).
    hist_last_ts_ms: int | None = field(default=None, compare=False)
    #: The window's TTFT / TPOT histogram deltas, UNGATED (like ``e2e_hist``): the
    #: SafeScale evidence pools them across the remaining pods, so a pod below the
    #: per-pod minimum samples still weighs in.
    ttft_hist: tuple[tuple[float, float], ...] | None = field(default=None, compare=False)
    ttft_hist_count: float | None = field(default=None, compare=False)
    tpot_hist: tuple[tuple[float, float], ...] | None = field(default=None, compare=False)
    tpot_hist_count: float | None = field(default=None, compare=False)
    #: The pod's NEWEST gateway instant sample in the window (not a window average):
    #: vLLM ``num_requests_waiting`` / ``num_requests_running`` / KV-cache fill (0..1,
    #: None when the doc does not carry it) and the sample's timestamp (ms). Read by the
    #: controller's onset saturation rescue (design 20261002-saturation-onset-rescue),
    #: which judges "engine full" on the latest grid, not on the 30 s mean. None = no
    #: instant doc in the window. Excluded from equality.
    latest_waiting: float | None = field(default=None, compare=False)
    latest_running: float | None = field(default=None, compare=False)
    latest_gpu_cache: float | None = field(default=None, compare=False)
    latest_instant_ms: int | None = field(default=None, compare=False)


#: How a model-level p95 is formed from the pods' merged histograms:
#: (percentile mode, minimum observations; 0 = no gate). See ``aggregate_pods``.
P95Rule = tuple[str, int]


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
    #: The rule the model e2e p95 was formed with (merged pod histograms, gated at
    #: the model level); carried so a re-aggregation (``restrict_to_serving``)
    #: applies the same one. None = max of the per-pod p95s (no histograms).
    p95_rule: P95Rule | None = field(default=None, compare=False)
    #: O1 (breakpoint-aware window, 2026-10-01): the same model window read over its
    #: grid-aligned suffixes ``(s, window_end_ms]``, one per gateway boundary ``s``
    #: strictly inside the window, ascending by ``s`` (for a 30 s window on the 10 s
    #: grid: the last 20 s and the last 10 s). Built from the docs the full read
    #: already fetched (``MetricsStore(suffix_period_ms=...)``); empty when off, for
    #: unaligned windows and for offline / synthetic windows. The controller decides on
    #: a suffix only while a breakpoint (traffic onset, routable-count change) lies
    #: inside the window. Excluded from equality.
    suffix_windows: tuple["ModelWindowMetrics", ...] = field(default=(), compare=False)
    #: I3 (2026-10-04): pods left out of this window because the gateway's last
    #: successful scrape of them (doc ``scraped_ms``) lies before the window - their
    #: docs repeat frozen values. Diagnostics only; excluded from equality.
    scrape_stale_pods: tuple[str, ...] = field(default=(), compare=False)
    #: Some pod of this model (sleeping ones included) has a successful gateway scrape
    #: inside the window: the gateway scraper is alive. False for an old gateway.
    scrape_fresh: bool = field(default=False, compare=False)
    #: ``scrape_stale_pods`` -> routed, unfinished requests on the pod (gateway in-flight
    #: mirror, live gateway instances only; ``tre_common.gateway_inflight``).
    scrape_stale_inflight: dict[str, int] = field(default_factory=dict, compare=False)
    #: Review 2026-10-06 P2-2: pods whose newest successful gateway scrape
    #: (``scraped_ms``) lies in the window's last grid ``[window_end - grid, window_end]``
    #: (gateway clock). An idle window needs every serving pod here.
    scrape_current_pods: frozenset[str] = field(default=frozenset(), compare=False)
    #: P2-2: the gateway in-flight count of each pod with docs in the window (live
    #: gateway instances; ``tre_common.gateway_inflight``), read only when the window has
    #: no token and no running / waiting request. None = not read or unknown.
    gateway_inflight: dict[str, int] | None = field(default=None, compare=False)


@dataclass(frozen=True)
class MetricsSnapshot:
    ts_ms: int
    models: dict[str, ModelWindowMetrics]
    stale: bool

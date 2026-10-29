from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class Observed:
    """What a window actually observed, on the observation clock (round 4,
    docs/design/20261025-instant-sampling.md "Evidence, not ticks").

    Every time here is a gateway ``scraped_ms`` (the start of the /metrics fetch whose
    values a doc carries; gateway wall clock), never a doc stamp (the write grid):

    * ``t_first_ms``: where the window's evidence starts - the observation of the token
      baseline (the histogram doc the deltas start from), else the first observation;
    * ``t_last_ms``: the newest observation in the window.

    (Round 5: only what a decision reads. Observation counts, request counts and flags
    live on the window itself - ``request_count_observed``, ``sample_incomplete_pods``.)

    The evidence period is a storage detail: a decision rule reads spans of these
    times (``t_last - t_ref``), never a count of windows. None = unknown."""

    t_first_ms: int | None = None
    t_last_ms: int | None = None

    @property
    def span_ms(self) -> int:
        """``t_last - t_first`` (0 when unknown)."""
        if self.t_first_ms is None or self.t_last_ms is None:
            return 0
        return max(0, int(self.t_last_ms) - int(self.t_first_ms))

    def since(self, ref_ms: int | None) -> int:
        """The observed span after the reference event ``ref_ms`` (a breakpoint, a hide,
        a rollback; same clock): ``t_last - max(t_first, ref)``, 0 when unknown."""
        if self.t_last_ms is None:
            return 0
        start = self.t_first_ms if ref_ms is None else (
            int(ref_ms) if self.t_first_ms is None else max(int(self.t_first_ms), int(ref_ms))
        )
        if start is None:
            return 0
        return max(0, int(self.t_last_ms) - int(start))

    @classmethod
    def merge(cls, items: Iterable["Observed | None"]) -> "Observed | None":
        """The observation of several pods together: earliest start, newest end.
        Order-independent; None when no item is known."""
        known = [item for item in items if item is not None]
        if not known:
            return None
        firsts = [item.t_first_ms for item in known if item.t_first_ms is not None]
        lasts = [item.t_last_ms for item in known if item.t_last_ms is not None]
        return cls(
            t_first_ms=min(firsts) if firsts else None,
            t_last_ms=max(lasts) if lasts else None,
        )


@dataclass(frozen=True)
class PodWindowMetrics:
    pod: str
    #: The TSS numerator's token totals over the window (2026-10-28, design
    #: docs/design/20261028-tss-step-counters.md): vLLM's per-step counters
    #: ``prompt_tokens_total`` / ``generation_tokens_total`` - their delta between the
    #: baseline and the newest histogram doc, divided by the two docs' ``scraped_ms``
    #: span and multiplied by the window length. Tokens the engine PROCESSED in the
    #: window, not the tokens of the requests that completed in it. None = unknown.
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
    #: Review round 3 (L5): the completed requests actually observed (the histogram
    #: count delta, not scaled to the window by the token span like
    #: ``request_count``). The evidence-count gates (``min_evidence_requests``, the
    #: low-evidence rescue step) count these. None = not read / no histogram.
    request_count_observed: float | None = field(default=None, compare=False)
    #: Timestamps (ms) of the gateway instant docs read for this window (as written,
    #: stale ones included). Provenance for the phase-aligned sampler, which waits until
    #: the gateway has written the window's end tick (2026-10-25: per-pod coverage is
    #: the store's ``sample_incomplete_pods``). Excluded from equality.
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
    #: False when some histogram doc of the read lacks the TTFT series, so ``ttft_count``
    #: (a delta of what the docs carry) is not a trustworthy first-token count. Read only
    #: by the alt ``ttft_mean`` signal; SafeScale and TSS ignore it.
    ttft_count_complete: bool = field(default=True, compare=False)
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
    #: Round 4: the pod's observation of the window (:class:`Observed`); None = not
    #: read on the observation clock (v1 schema, offline / synthetic windows).
    observed: Observed | None = field(default=None, compare=False)
    #: 2026-10-28: prompt tokens of the requests that COMPLETED in the window (the
    #: ``request_prompt_tokens`` histogram sum delta, not scaled), the numerator of the
    #: window's mean prompt length with ``request_count_observed`` (SafeScale's
    #: length-dependent TTFT threshold). Not a TSS input. None = no histogram.
    completed_prompt_tokens: float | None = field(default=None, compare=False)


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
    #: Review round 3 (L5): the completed requests actually observed (the histogram
    #: count delta, not scaled to the window by the token span like
    #: ``request_count``). The evidence-count gates (``min_evidence_requests``, the
    #: low-evidence rescue step) count these. None = not read / no histogram.
    request_count_observed: float | None = field(default=None, compare=False)
    #: Union of the per-pod ``instant_ticks_ms`` (see :class:`PodWindowMetrics`).
    instant_ticks_ms: tuple[int, ...] = field(default=(), compare=False)
    #: Sum of the pods' ``completed_prompt_tokens`` (see :class:`PodWindowMetrics`).
    completed_prompt_tokens: float | None = field(default=None, compare=False)
    #: Round 4: the model's observation of the window (:class:`Observed`, merged over
    #: the pods in ``per_pod``; the store's flags included). None = not read on the
    #: observation clock - use :func:`observed_of`.
    observed: Observed | None = field(default=None, compare=False)
    #: The rule the model e2e p95 was formed with (merged pod histograms, gated at
    #: the model level); carried so a re-aggregation (``restrict_to_serving``)
    #: applies the same one. None = max of the per-pod p95s (no histograms).
    p95_rule: P95Rule | None = field(default=None, compare=False)
    #: The same model window read over grid-aligned suffixes ``(s, window_end_ms]``, one
    #: per gateway boundary ``s`` strictly inside the window, ascending by ``s``. Built
    #: from the docs the full read already fetched (``MetricsStore(suffix_period_ms=...)``);
    #: live, only the SafeScale commit gate's post-hide tail of a probing model (2026-10-29:
    #: the decision signal always uses the whole window). Empty when off, for unaligned
    #: windows and for offline / synthetic windows. Excluded from equality.
    suffix_windows: tuple["ModelWindowMetrics", ...] = field(default=(), compare=False)
    #: I3 (2026-10-04; 2026-10-25 the sample rule): pods left out of this window
    #: because no sample of them was observed in it (no ``scraped_ms`` in the window -
    #: their docs repeat frozen values). Also in ``sample_incomplete_pods`` (a donor-gate
    #: flag). Excluded from equality.
    scrape_stale_pods: tuple[str, ...] = field(default=(), compare=False)
    #: 2026-10-25 (review round 1 D2): flagged pods - samples (distinct ``scraped_ms``
    #: in the window) below ``scaling.sample_min_ratio`` x (window / instant period), a
    #: token span over 1.5 windows, or left out (0 samples / no token span). A
    #: donor-gate flag only: a pod with samples stays in the window; while a flagged
    #: pod serves, the controller takes no donor / shrink / release from the model.
    #: Excluded from equality.
    sample_incomplete_pods: tuple[str, ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class MetricsSnapshot:
    ts_ms: int
    models: dict[str, ModelWindowMetrics]
    stale: bool
    #: Why the store marked the snapshot stale (e.g. a gateway cadence mismatch).
    stale_reason: str | None = field(default=None, compare=False)
    #: Round 5: the newest histogram boundary the gateway wrote completely
    #: (``last_hist_ms`` of ``tre:v2:gateway:cadence``, in the round's MULTI). The sampler's
    #: freshness rule: a window ending at ``E`` is complete once this is >= ``E``.
    #: None = not read (no cadence check, v1 schema, offline).
    gateway_last_hist_ms: int | None = field(default=None, compare=False)
    #: Controller clock (ms) when the sampler read this snapshot (its last fetch
    #: attempt); None = not from the sampler. TSS suffix (2026-10-29): a pod-set change
    #: is dated against it, never against a gateway stamp (``SignalState.note_snapshot``).
    read_ms: int | None = field(default=None, compare=False)


def mean_completed_prompt_tokens(metrics: Any) -> float | None:
    """Mean prompt length of the requests completed in a window: ``completed_prompt_tokens``
    / the observed completed-request count. None when either is unknown or no request
    completed. (``prompt_tokens`` is the step-counter total since 2026-10-28: tokens
    processed, not tokens of completed requests - never a mean-length numerator.)"""
    tokens = getattr(metrics, "completed_prompt_tokens", None)
    count = observed_requests(metrics)
    if tokens is None or not count or float(count) <= 0:
        return None
    return float(tokens) / float(count)


def observed_requests(metrics: Any) -> float | None:
    """The completed requests observed in a window (``request_count_observed``, not
    scaled by the token span; review round 3, L5), else ``request_count`` (synthetic /
    offline windows, which carry only that)."""
    observed = getattr(metrics, "request_count_observed", None)
    return getattr(metrics, "request_count", None) if observed is None else observed


def observed_of(metrics: Any) -> Observed:
    """``metrics.observed``, else - a window not read on the observation clock (v1,
    offline, synthetic) - the window bounds as its observation span."""
    observed = getattr(metrics, "observed", None)
    if observed is not None:
        return observed
    return Observed(t_first_ms=int(metrics.window_start_ms), t_last_ms=int(metrics.window_end_ms))

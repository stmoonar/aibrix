"""Per-pod -> per-model window aggregation, and the restriction to serving pods.

The gateway writes instant/histogram docs for *every* pod of a model, sleeping ones
included (zero gauges, flat counters), and ``MetricsStore`` sees them all: the raw
``ModelWindowMetrics.routable_pods`` counts every pod with a doc in the window (live on
09-22: 8 for dsqwen-7b with 1 awake + 7 asleep). The pod count is only trustworthy from
the service manager's fleet state, which the controller already holds (``ClusterView``).

:func:`aggregate_pods` is the one aggregation rule (sums for tokens / queue gauges, max for
the TTFT / TPOT p95, the merged-histogram e2e p95, mean of the non-zero kv hit rates);
``MetricsStore`` uses it for the raw window and
:func:`restrict_to_serving` re-applies it to the awake pods, so the decision path and the
safescale observation see exactly the window of the pods that can serve - the same window
the single-awake-pod calibration capture produced offline.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Collection, Mapping, Optional

from tre_common.metrics_schema import ModelWindowMetrics, P95Rule, PodWindowMetrics
from tre_common.percentile import histogram_percentile


def _sum_optional(values: list[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _max_optional(values: list[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def merged_hist_p95_ms(pods: list[PodWindowMetrics], rule: P95Rule) -> Optional[float]:
    """Model e2e p95 (ms) from the pods' window histograms merged FIRST, then gated.

    ``rule`` = (percentile mode, minimum observations). The gate applies to the
    merged count: at low load every pod sees fewer than the minimum, yet together
    they may have enough (a per-pod gate dropped them all). Buckets are summed on
    the union of the pods' bucket bounds (cumulative count at or below each)."""
    mode, min_samples = rule
    hists = [(pod.e2e_hist, float(pod.e2e_hist_count or 0.0)) for pod in pods if pod.e2e_hist]
    if not hists:
        return None
    count = sum(item_count for _, item_count in hists)
    if min_samples > 0 and count < min_samples:
        return None
    uppers = sorted({upper for buckets, _ in hists for upper, _ in buckets})
    merged = [
        (upper, sum(_cumulative_at(buckets, upper) for buckets, _ in hists)) for upper in uppers
    ]
    p95_s = histogram_percentile(merged, 0.95, mode=mode)
    return None if p95_s is None else p95_s * 1000.0


def pooled_p95_ms(hists, rule: P95Rule) -> Optional[float]:
    """p95 (ms) of several cumulative histograms ``((upper_s, cumulative), ...)`` with
    their observation counts, merged FIRST and then gated (``rule`` = (percentile mode,
    minimum observations) applied to the merged count), like :func:`merged_hist_p95_ms`.
    SafeScale judges it next to the per-pod maximum: a pod below the per-pod minimum
    (an overloaded pod completes few requests) still weighs in. A p95 in the ``+Inf``
    bucket is reported as the largest finite bound (a lower bound; keeps JSON strict)."""
    mode, min_samples = rule
    present = [(tuple(buckets), float(count or 0.0)) for buckets, count in hists if buckets]
    if not present:
        return None
    count = sum(item for _, item in present)
    if count <= 0 or (min_samples > 0 and count < min_samples):
        return None
    uppers = sorted({upper for buckets, _ in present for upper, _ in buckets})
    merged = [(upper, sum(_cumulative_at(buckets, upper) for buckets, _ in present)) for upper in uppers]
    p95_s = histogram_percentile(merged, 0.95, mode=mode)
    if p95_s is None:
        return None
    if p95_s == float("inf"):
        finite = [upper for upper in uppers if upper != float("inf")]
        if not finite:
            return None
        p95_s = max(finite)
    return float(p95_s) * 1000.0


def _cumulative_at(buckets: tuple[tuple[float, float], ...], upper: float) -> float:
    counts = [count for bucket_upper, count in buckets if bucket_upper <= upper]
    return max(counts) if counts else 0.0


def aggregate_pods(
    model: str,
    window_start_ms: int,
    window_end_ms: int,
    per_pod: Mapping[str, PodWindowMetrics],
    *,
    p95_rule: P95Rule | None = None,
) -> ModelWindowMetrics:
    """``p95_rule`` set (the live MetricsStore): the e2e p95 comes from the pods'
    merged histograms, gated on the merged count (:func:`merged_hist_p95_ms`); with no
    pod histogram at all it is the max of the per-pod p95s. None: max of the per-pod
    p95s (offline / synthetic windows)."""
    pods = list(per_pod.values())
    e2e_p95_ms = _max_optional([pod.e2e_p95_ms for pod in pods])
    if p95_rule is not None and any(pod.e2e_hist for pod in pods):
        e2e_p95_ms = merged_hist_p95_ms(pods, p95_rule)
    routable_pods = len(pods)
    kv_values = [pod.kv_cache_hit_rate for pod in pods if pod.kv_cache_hit_rate > 0.0]
    return ModelWindowMetrics(
        model=model,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        prompt_tokens=_sum_optional([pod.prompt_tokens for pod in pods]),
        generation_tokens=_sum_optional([pod.generation_tokens for pod in pods]),
        avg_waiting=sum(pod.avg_waiting for pod in pods),
        avg_running=sum(pod.avg_running for pod in pods),
        avg_swapping=sum(pod.avg_swapping for pod in pods),
        kv_cache_hit_rate=(sum(kv_values) / len(kv_values)) if kv_values else 0.0,
        ttft_p95_ms=_max_optional([pod.ttft_p95_ms for pod in pods]),
        tpot_p95_ms=_max_optional([pod.tpot_p95_ms for pod in pods]),
        e2e_p95_ms=e2e_p95_ms,
        routable_pods=routable_pods,
        assigned_replicas=routable_pods,
        per_pod=dict(per_pod),
        request_count=_sum_optional([pod.request_count for pod in pods]),
        token_counter_reset=any(pod.token_counter_reset for pod in pods),
        instant_ticks_ms=tuple(sorted({tick for pod in pods for tick in pod.instant_ticks_ms})),
        p95_rule=p95_rule,
    )


def restrict_to_serving(
    metrics: ModelWindowMetrics,
    *,
    sleeping_pods: Collection[str],
    routable_pods: int,
) -> ModelWindowMetrics:
    """``metrics`` without the docs of pods the fleet state reports asleep.

    * ``sleeping_pods``: pod names (== SM ``serve_id``) of this model's bindings with
      ``awake=False``. Only positively-known sleepers are dropped; a pod the fleet state
      does not list (a just-replaced pod whose docs are still in the window) is kept, so
      real traffic is never discarded on a stale view. Hidden-but-awake pods (a safescale
      probe) are kept too: they are still draining the requests they hold.
    * ``routable_pods``: the authoritative serving count (awake and not hidden); it
      replaces both ``routable_pods`` and ``assigned_replicas`` (TSS replica factor 1,
      per-replica alt signals divide by the serving count).

    A window with no per-pod breakdown (synthetic / offline) only gets the counts.
    """
    per_pod = metrics.per_pod
    if per_pod and sleeping_pods:
        asleep = set(sleeping_pods)
        kept = {key: pod for key, pod in per_pod.items() if pod.pod not in asleep and key not in asleep}
        if len(kept) != len(per_pod):
            metrics = replace(
                aggregate_pods(
                    metrics.model, metrics.window_start_ms, metrics.window_end_ms, kept,
                    p95_rule=metrics.p95_rule,
                ),
                # Ticks are gateway-wide provenance (freshness), not a per-pod quantity.
                instant_ticks_ms=metrics.instant_ticks_ms,
            )
    return replace(metrics, routable_pods=int(routable_pods), assigned_replicas=int(routable_pods))

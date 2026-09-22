"""Per-pod -> per-model window aggregation, and the restriction to serving pods.

The gateway writes instant/histogram docs for *every* pod of a model, sleeping ones
included (zero gauges, flat counters), and ``MetricsStore`` sees them all: the raw
``ModelWindowMetrics.routable_pods`` counts every pod with a doc in the window (live on
09-22: 8 for dsqwen-7b with 1 awake + 7 asleep). The pod count is only trustworthy from
the service manager's fleet state, which the controller already holds (``ClusterView``).

:func:`aggregate_pods` is the one aggregation rule (sums for tokens / queue gauges, max for
p95, mean of the non-zero kv hit rates); ``MetricsStore`` uses it for the raw window and
:func:`restrict_to_serving` re-applies it to the awake pods, so the decision path and the
safescale observation see exactly the window of the pods that can serve - the same window
the single-awake-pod calibration capture produced offline.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Collection, Mapping, Optional

from tre_common.metrics_schema import ModelWindowMetrics, PodWindowMetrics


def _sum_optional(values: list[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _max_optional(values: list[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def aggregate_pods(
    model: str,
    window_start_ms: int,
    window_end_ms: int,
    per_pod: Mapping[str, PodWindowMetrics],
) -> ModelWindowMetrics:
    pods = list(per_pod.values())
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
        e2e_p95_ms=_max_optional([pod.e2e_p95_ms for pod in pods]),
        routable_pods=routable_pods,
        assigned_replicas=routable_pods,
        per_pod=dict(per_pod),
        request_count=_sum_optional([pod.request_count for pod in pods]),
        token_counter_reset=any(pod.token_counter_reset for pod in pods),
        instant_ticks_ms=tuple(sorted({tick for pod in pods for tick in pod.instant_ticks_ms})),
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
                aggregate_pods(metrics.model, metrics.window_start_ms, metrics.window_end_ms, kept),
                # Ticks are gateway-wide provenance (freshness), not a per-pod quantity.
                instant_ticks_ms=metrics.instant_ticks_ms,
            )
    return replace(metrics, routable_pods=int(routable_pods), assigned_replicas=int(routable_pods))

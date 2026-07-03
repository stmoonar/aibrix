
from __future__ import annotations

import json
from typing import Any

from tre_common.metrics_schema import ModelWindowMetrics, PodWindowMetrics


def legacy_collect_model_window(redis, model: str, window_start_ms: int, window_end_ms: int, *, instant_sample_interval_ms: int) -> ModelWindowMetrics:
    pods = sorted(str(pod) for pod in redis.smembers(f"tre:v2:pods:{model}"))
    per_pod: dict[str, PodWindowMetrics] = {}
    for pod_key in pods:
        hist_docs = _read_docs(redis, f"tre:v2:hist:{pod_key}", window_start_ms, window_end_ms)
        inst_docs = _read_docs(redis, f"tre:v2:inst:{pod_key}", window_start_ms, window_end_ms)
        if not hist_docs and not inst_docs:
            continue
        pod_name = _pod_name(hist_docs, inst_docs, pod_key)
        metrics = PodWindowMetrics(
            pod=pod_name,
            prompt_tokens=_hist_delta_sum(model, "request_prompt_tokens", hist_docs),
            generation_tokens=_hist_delta_sum(model, "request_generation_tokens", hist_docs),
            avg_waiting=_instant_avg(model, "num_requests_waiting", inst_docs, window_start_ms, window_end_ms, instant_sample_interval_ms),
            avg_running=_instant_avg(model, "num_requests_running", inst_docs, window_start_ms, window_end_ms, instant_sample_interval_ms),
            avg_swapping=_instant_avg(model, "num_requests_swapped", inst_docs, window_start_ms, window_end_ms, instant_sample_interval_ms),
            kv_cache_hit_rate=_instant_avg(model, "kv_cache_hit_rate", inst_docs, window_start_ms, window_end_ms, instant_sample_interval_ms),
            ttft_p95_ms=_seconds_to_ms(_hist_bucket_upper_p95(model, "time_to_first_token_seconds", hist_docs)),
            tpot_p95_ms=_seconds_to_ms(_hist_bucket_upper_p95(model, "time_per_output_token_seconds", hist_docs)),
            e2e_p95_ms=_seconds_to_ms(_hist_bucket_upper_p95(model, "e2e_request_latency_seconds", hist_docs)),
        )
        per_pod[pod_name] = metrics

    pods_metrics = list(per_pod.values())
    kv_values = [pod.kv_cache_hit_rate for pod in pods_metrics if pod.kv_cache_hit_rate > 0.0]
    return ModelWindowMetrics(
        model=model,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        prompt_tokens=sum(pod.prompt_tokens for pod in pods_metrics),
        generation_tokens=sum(pod.generation_tokens for pod in pods_metrics),
        avg_waiting=sum(pod.avg_waiting for pod in pods_metrics),
        avg_running=sum(pod.avg_running for pod in pods_metrics),
        avg_swapping=sum(pod.avg_swapping for pod in pods_metrics),
        kv_cache_hit_rate=(sum(kv_values) / len(kv_values)) if kv_values else 0.0,
        ttft_p95_ms=_max_optional([pod.ttft_p95_ms for pod in pods_metrics]),
        tpot_p95_ms=_max_optional([pod.tpot_p95_ms for pod in pods_metrics]),
        e2e_p95_ms=_max_optional([pod.e2e_p95_ms for pod in pods_metrics]),
        routable_pods=len(pods_metrics),
        assigned_replicas=len(pods_metrics),
        per_pod=per_pod,
    )


def _read_docs(redis, key: str, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    docs = []
    for raw in redis.zrangebyscore(key, start_ms, end_ms):
        try:
            doc = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(doc, dict):
            docs.append(doc)
    return sorted(docs, key=lambda doc: _number(doc.get("timestamp"), 0.0))


def _pod_name(hist_docs: list[dict[str, Any]], inst_docs: list[dict[str, Any]], fallback: str) -> str:
    for doc in hist_docs + inst_docs:
        name = doc.get("pod_name")
        if isinstance(name, str) and name:
            return name
    return fallback


def _hist_entry(model: str, metric: str, doc: dict[str, Any]) -> dict[str, Any] | None:
    metrics = doc.get("model_histogram_metrics")
    if not isinstance(metrics, dict):
        return None
    entry = metrics.get(f"{model}/{metric}")
    return entry if isinstance(entry, dict) else None


def _first_last(model: str, metric: str, docs: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    entries = [_hist_entry(model, metric, doc) for doc in docs]
    entries = [entry for entry in entries if entry is not None]
    if not entries:
        return None, None
    return entries[0], entries[-1]


def _hist_delta_sum(model: str, metric: str, docs: list[dict[str, Any]]) -> float:
    first, last = _first_last(model, metric, docs)
    if first is None or last is None:
        return 0.0
    return max(0.0, _number(last.get("sum"), 0.0) - _number(first.get("sum"), 0.0))


def _hist_bucket_upper_p95(model: str, metric: str, docs: list[dict[str, Any]]) -> float | None:
    first, last = _first_last(model, metric, docs)
    if first is None or last is None:
        return None
    first_buckets = _buckets(first.get("buckets"))
    last_buckets = _buckets(last.get("buckets"))
    if not first_buckets or not last_buckets:
        return None
    delta = {}
    running = 0.0
    for upper in sorted(set(first_buckets) | set(last_buckets)):
        value = max(0.0, _cumulative_at(last_buckets, upper) - _cumulative_at(first_buckets, upper))
        running = max(running, value)
        delta[upper] = running
    total = list(delta.values())[-1]
    target = 0.95 * total
    for upper, cumulative in sorted(delta.items()):
        if cumulative >= target:
            return upper
    return None


def _instant_avg(model: str, metric: str, docs: list[dict[str, Any]], start_ms: int, end_ms: int, sample_ms: int) -> float:
    total = 0.0
    key = f"{model}/{metric}"
    for doc in docs:
        metrics = doc.get("model_metrics")
        if isinstance(metrics, dict):
            total += _number(metrics.get(key), 0.0)
    expected = max(1, int((end_ms - start_ms) / sample_ms))
    return total / expected


def _buckets(raw: Any) -> dict[float, float]:
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key, value in raw.items():
        try:
            out[float(key)] = _number(value, 0.0)
        except ValueError:
            continue
    return out


def _cumulative_at(buckets: dict[float, float], upper: float) -> float:
    candidates = [count for bucket_upper, count in buckets.items() if bucket_upper <= upper]
    return max(candidates) if candidates else 0.0


def _number(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _seconds_to_ms(value: float | None) -> float | None:
    return None if value is None else value * 1000.0


def _max_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None

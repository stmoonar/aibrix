
from __future__ import annotations

import json
from typing import Any

from tre_common.metrics_schema import ModelWindowMetrics, PodWindowMetrics
from tre_common.percentile import histogram_percentile
from tre_common.rediskeys import hist_key, inst_key, pods_key

HISTOGRAM_METRICS = {
    "prompt_tokens": "request_prompt_tokens",
    "generation_tokens": "request_generation_tokens",
    "ttft": "time_to_first_token_seconds",
    "tpot": "time_per_output_token_seconds",
    "e2e": "e2e_request_latency_seconds",
}

INSTANT_METRICS = {
    "waiting": "num_requests_waiting",
    "running": "num_requests_running",
    "swapping": "num_requests_swapped",
    "kv_hit": "kv_cache_hit_rate",
}


class MetricsStore:
    def __init__(
        self,
        redis_client: Any,
        registry: Any,
        *,
        instant_sample_interval_ms: int,
        percentile_mode: str = "bucket_upper",
    ) -> None:
        if instant_sample_interval_ms <= 0:
            raise ValueError("instant_sample_interval_ms must be positive")
        self._redis = redis_client
        self._registry = registry
        self._instant_sample_interval_ms = instant_sample_interval_ms
        self._percentile_mode = percentile_mode
        self._window_cache: dict[tuple[str, int, int], ModelWindowMetrics] = {}

    def read_model_window(self, model: str, window_start_ms: int, window_end_ms: int) -> ModelWindowMetrics:
        cache_key = (model, int(window_start_ms), int(window_end_ms))
        if cache_key in self._window_cache:
            return self._window_cache[cache_key]

        pods = sorted(_decode_text(pod) for pod in self._redis.smembers(pods_key(model)))
        per_pod: dict[str, PodWindowMetrics] = {}
        for pod_key in pods:
            hist_docs = self._read_zset_docs(hist_key(pod_key), window_start_ms, window_end_ms)
            inst_docs = self._read_zset_docs(inst_key(pod_key), window_start_ms, window_end_ms)
            pod_metrics = self._aggregate_pod(model, pod_key, hist_docs, inst_docs, window_start_ms, window_end_ms)
            if pod_metrics is not None:
                per_pod[pod_metrics.pod] = pod_metrics

        model_metrics = self._aggregate_model(model, window_start_ms, window_end_ms, per_pod)
        self._window_cache[cache_key] = model_metrics
        return model_metrics

    def _read_zset_docs(self, key: str, window_start_ms: int, window_end_ms: int) -> list[dict[str, Any]]:
        raw_members = self._redis.zrangebyscore(key, window_start_ms, window_end_ms)
        docs: list[dict[str, Any]] = []
        for raw in raw_members:
            try:
                doc = json.loads(_decode_text(raw))
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(doc, dict):
                docs.append(doc)
        docs.sort(key=lambda doc: _number(doc.get("timestamp"), 0.0))
        return docs

    def _aggregate_pod(
        self,
        model: str,
        pod_key: str,
        hist_docs: list[dict[str, Any]],
        inst_docs: list[dict[str, Any]],
        window_start_ms: int,
        window_end_ms: int,
    ) -> PodWindowMetrics | None:
        if not hist_docs and not inst_docs:
            return None
        pod_name = _pod_name(hist_docs, inst_docs, pod_key)

        prompt_tokens = self._hist_sum_delta(model, HISTOGRAM_METRICS["prompt_tokens"], hist_docs)
        generation_tokens = self._hist_sum_delta(model, HISTOGRAM_METRICS["generation_tokens"], hist_docs)
        ttft_avg_s = self._hist_avg_delta(model, HISTOGRAM_METRICS["ttft"], hist_docs)
        tpot_avg_s = self._hist_avg_delta(model, HISTOGRAM_METRICS["tpot"], hist_docs)
        e2e_avg_s = self._hist_avg_delta(model, HISTOGRAM_METRICS["e2e"], hist_docs)

        ttft_p95_s = self._hist_percentile(model, HISTOGRAM_METRICS["ttft"], hist_docs)
        tpot_p95_s = self._hist_percentile(model, HISTOGRAM_METRICS["tpot"], hist_docs)
        e2e_p95_s = self._hist_percentile(model, HISTOGRAM_METRICS["e2e"], hist_docs)

        return PodWindowMetrics(
            pod=pod_name,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            avg_waiting=self._instant_avg(model, INSTANT_METRICS["waiting"], inst_docs, window_start_ms, window_end_ms),
            avg_running=self._instant_avg(model, INSTANT_METRICS["running"], inst_docs, window_start_ms, window_end_ms),
            avg_swapping=self._instant_avg(model, INSTANT_METRICS["swapping"], inst_docs, window_start_ms, window_end_ms),
            kv_cache_hit_rate=self._instant_avg(model, INSTANT_METRICS["kv_hit"], inst_docs, window_start_ms, window_end_ms),
            ttft_p95_ms=_seconds_to_ms(ttft_p95_s),
            tpot_p95_ms=_seconds_to_ms(tpot_p95_s),
            e2e_p95_ms=_seconds_to_ms(e2e_p95_s),
        )

    def _aggregate_model(
        self,
        model: str,
        window_start_ms: int,
        window_end_ms: int,
        per_pod: dict[str, PodWindowMetrics],
    ) -> ModelWindowMetrics:
        pods = list(per_pod.values())
        routable_pods = len(pods)
        kv_values = [pod.kv_cache_hit_rate for pod in pods if pod.kv_cache_hit_rate > 0.0]
        return ModelWindowMetrics(
            model=model,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            prompt_tokens=sum(pod.prompt_tokens for pod in pods),
            generation_tokens=sum(pod.generation_tokens for pod in pods),
            avg_waiting=sum(pod.avg_waiting for pod in pods),
            avg_running=sum(pod.avg_running for pod in pods),
            avg_swapping=sum(pod.avg_swapping for pod in pods),
            kv_cache_hit_rate=(sum(kv_values) / len(kv_values)) if kv_values else 0.0,
            ttft_p95_ms=_max_optional([pod.ttft_p95_ms for pod in pods]),
            tpot_p95_ms=_max_optional([pod.tpot_p95_ms for pod in pods]),
            e2e_p95_ms=_max_optional([pod.e2e_p95_ms for pod in pods]),
            routable_pods=routable_pods,
            assigned_replicas=routable_pods,
            per_pod=per_pod,
        )

    def _hist_sum_delta(self, model: str, metric: str, docs: list[dict[str, Any]]) -> float:
        first, last = _first_last_metric(model, metric, docs)
        if first is None or last is None:
            return 0.0
        return max(0.0, _number(last.get("sum"), 0.0) - _number(first.get("sum"), 0.0))

    def _hist_avg_delta(self, model: str, metric: str, docs: list[dict[str, Any]]) -> float | None:
        first, last = _first_last_metric(model, metric, docs)
        if first is None or last is None:
            return None
        sum_delta = max(0.0, _number(last.get("sum"), 0.0) - _number(first.get("sum"), 0.0))
        count_delta = max(0.0, _number(last.get("count"), 0.0) - _number(first.get("count"), 0.0))
        if count_delta <= 0.0:
            return None
        return sum_delta / count_delta

    def _hist_percentile(self, model: str, metric: str, docs: list[dict[str, Any]]) -> float | None:
        first, last = _first_last_metric(model, metric, docs)
        if first is None or last is None:
            return None
        first_buckets = _normal_buckets(first.get("buckets"))
        last_buckets = _normal_buckets(last.get("buckets"))
        if not first_buckets or not last_buckets:
            return None
        delta = _bucket_delta(first_buckets, last_buckets)
        return histogram_percentile(delta.items(), 0.95, mode=self._percentile_mode)

    def _instant_avg(
        self,
        model: str,
        metric: str,
        docs: list[dict[str, Any]],
        window_start_ms: int,
        window_end_ms: int,
    ) -> float:
        metric_key = f"{model}/{metric}"
        total = 0.0
        for doc in docs:
            metrics = doc.get("model_metrics")
            if isinstance(metrics, dict):
                total += _number(metrics.get(metric_key), 0.0)
        expected_samples = max(1, int((window_end_ms - window_start_ms) / self._instant_sample_interval_ms))
        return total / expected_samples


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _number(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pod_name(hist_docs: list[dict[str, Any]], inst_docs: list[dict[str, Any]], pod_key: str) -> str:
    for doc in hist_docs + inst_docs:
        name = doc.get("pod_name")
        if isinstance(name, str) and name:
            return name
    return pod_key


def _metric_entry(model: str, metric: str, doc: dict[str, Any]) -> dict[str, Any] | None:
    metrics = doc.get("model_histogram_metrics")
    if not isinstance(metrics, dict):
        return None
    entry = metrics.get(f"{model}/{metric}")
    return entry if isinstance(entry, dict) else None


def _first_last_metric(model: str, metric: str, docs: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    entries = [_metric_entry(model, metric, doc) for doc in docs]
    entries = [entry for entry in entries if entry is not None]
    if not entries:
        return None, None
    return entries[0], entries[-1]


def _normal_buckets(raw: Any) -> dict[float, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[float, float] = {}
    for key, value in raw.items():
        try:
            upper = float("inf") if str(key) in {"+Inf", "Inf", "inf"} else float(key)
            out[upper] = _number(value, 0.0)
        except ValueError:
            continue
    return out


def _cumulative_at(buckets: dict[float, float], upper: float) -> float:
    candidates = [count for bucket_upper, count in buckets.items() if bucket_upper <= upper]
    return max(candidates) if candidates else 0.0


def _bucket_delta(first: dict[float, float], last: dict[float, float]) -> dict[float, float]:
    result: dict[float, float] = {}
    running = 0.0
    for upper in sorted(set(first) | set(last)):
        delta = max(0.0, _cumulative_at(last, upper) - _cumulative_at(first, upper))
        running = max(running, delta)
        result[upper] = running
    return result


def _seconds_to_ms(value: float | None) -> float | None:
    return None if value is None else value * 1000.0


def _max_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None

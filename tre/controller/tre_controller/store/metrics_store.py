
from __future__ import annotations

import json
from typing import Any

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics, PodWindowMetrics
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

LEGACY_HIST_PREFIX = "aibrix:pod_histogram_metrics_"
LEGACY_INST_PREFIX = "aibrix:pod_instant_metrics_"


class MetricsStore:
    def __init__(
        self,
        redis_client: Any,
        registry: Any,
        *,
        instant_sample_interval_ms: int,
        percentile_mode: str = "bucket_upper",
        schema: str = "v2",
        histogram_lookback_ms: int = 90_000,
        min_latency_samples: int = 0,
    ) -> None:
        if instant_sample_interval_ms <= 0:
            raise ValueError("instant_sample_interval_ms must be positive")
        if histogram_lookback_ms < 0:
            raise ValueError("histogram_lookback_ms must be non-negative")
        if min_latency_samples < 0:
            raise ValueError("min_latency_samples must be non-negative")
        if schema not in {"v1", "v2"}:
            raise ValueError("schema must be v1 or v2")
        self._redis = redis_client
        self._registry = registry
        self._instant_sample_interval_ms = instant_sample_interval_ms
        self._percentile_mode = percentile_mode
        self._schema = schema
        self._histogram_lookback_ms = histogram_lookback_ms
        # N1: below this many latency observations in a window, a p95 estimate is too
        # noisy to decide on (short windows + low QPS can have single-digit samples).
        # 0 disables the guard (default; the live controller sets it from config).
        self._min_latency_samples = min_latency_samples
        self._window_cache: dict[tuple[str, str, int, int], ModelWindowMetrics] = {}

    def read_snapshot(
        self, window_start_ms: int, window_end_ms: int, *, use_cache: bool = True
    ) -> MetricsSnapshot:
        models = {
            spec.name: self.read_model_window(spec.name, window_start_ms, window_end_ms, use_cache=use_cache)
            for spec in self._registry.models()
        }
        return MetricsSnapshot(ts_ms=int(window_end_ms), models=models, stale=False)

    def read_model_window(
        self, model: str, window_start_ms: int, window_end_ms: int, *, use_cache: bool = True
    ) -> ModelWindowMetrics:
        cache_key = (self._schema, model, int(window_start_ms), int(window_end_ms))
        # Sliding windows (S1.1) pass use_cache=False: every window is unique, so the
        # per-window cache never hits and would grow without bound. Only tumbling reads
        # (repeated identical [start, end] within a block) benefit from caching.
        if use_cache and cache_key in self._window_cache:
            return self._window_cache[cache_key]

        if self._schema == "v1":
            per_pod = self._read_v1_model_window(model, window_start_ms, window_end_ms)
        else:
            pods = sorted(_decode_text(pod) for pod in self._redis.smembers(pods_key(model)))
            per_pod: dict[str, PodWindowMetrics] = {}
            for pod_key in pods:
                hist_docs = self._read_zset_docs(
                    hist_key(pod_key),
                    window_start_ms,
                    window_end_ms,
                    lookback_ms=self._histogram_lookback_ms,
                )
                inst_docs = self._read_zset_docs(inst_key(pod_key), window_start_ms, window_end_ms)
                pod_metrics = self._aggregate_pod(model, pod_key, hist_docs, inst_docs, window_start_ms, window_end_ms)
                if pod_metrics is not None:
                    per_pod[pod_metrics.pod] = pod_metrics

        model_metrics = self._aggregate_model(model, window_start_ms, window_end_ms, per_pod)
        if use_cache:
            self._window_cache[cache_key] = model_metrics
        return model_metrics

    def read_latest_instant(self, model: str, now_ms: int, lookback_ms: int) -> dict[str, float]:
        """Latest instant queue snapshot (waiting/running/swapping), summed across pods.

        Unlike ``read_model_window`` this returns the single most-recent scrape value per
        pod (not a windowed average divided by expected_samples). The gateway writes the
        instant buckets on a boundary-aligned ~10s ticker (SCRAPE_INTERVAL_MS); at real
        time ``now`` the current bucket may not be written yet, so a narrow read can miss
        it and record 0 (r3 SMOKE_FINDINGS defect 1: 34/34 zero samples). ``lookback_ms``
        should be >= ~2x SCRAPE_INTERVAL_MS so the last-written bucket is always in range;
        taking the latest doc (not an average) avoids the halving that a windowed read
        would suffer when only one bucket is present. Used by the offline r3 sidecar
        sampler to capture a queue snapshot that reconciles with the online average.
        """
        start_ms = max(0, now_ms - lookback_ms)
        if self._schema == "v1":
            inst_by_pod = self._read_legacy_docs(LEGACY_INST_PREFIX, model, start_ms, now_ms)
        else:
            inst_by_pod = {
                _decode_text(pod): self._read_zset_docs(inst_key(_decode_text(pod)), start_ms, now_ms)
                for pod in self._redis.smembers(pods_key(model))
            }
        totals = {"waiting": 0.0, "running": 0.0, "swapping": 0.0}
        for docs in inst_by_pod.values():
            if not docs:
                continue
            latest = docs[-1]  # _read_* sort ascending by timestamp -> last is freshest
            metrics = latest.get("model_metrics")
            if not isinstance(metrics, dict):
                continue
            for out_key, metric in (
                ("waiting", INSTANT_METRICS["waiting"]),
                ("running", INSTANT_METRICS["running"]),
                ("swapping", INSTANT_METRICS["swapping"]),
            ):
                totals[out_key] += _number(metrics.get(f"{model}/{metric}"), 0.0)
        return totals

    def _read_zset_docs(
        self,
        key: str,
        window_start_ms: int,
        window_end_ms: int,
        *,
        lookback_ms: int = 0,
    ) -> list[dict[str, Any]]:
        raw_members = self._redis.zrangebyscore(key, max(0, window_start_ms - lookback_ms), window_end_ms)
        docs: list[dict[str, Any]] = []
        for raw in raw_members:
            try:
                doc = json.loads(_decode_text(raw))
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(doc, dict):
                docs.append(doc)
        docs.sort(key=lambda doc: _number(doc.get("timestamp"), 0.0))
        return _with_baseline_doc(docs, window_start_ms) if lookback_ms else docs

    def _read_v1_model_window(
        self,
        model: str,
        window_start_ms: int,
        window_end_ms: int,
    ) -> dict[str, PodWindowMetrics]:
        hist_by_pod = self._read_legacy_docs(
            LEGACY_HIST_PREFIX,
            model,
            window_start_ms,
            window_end_ms,
            lookback_ms=self._histogram_lookback_ms,
        )
        inst_by_pod = self._read_legacy_docs(LEGACY_INST_PREFIX, model, window_start_ms, window_end_ms)
        per_pod: dict[str, PodWindowMetrics] = {}
        for pod_key in sorted(set(hist_by_pod) | set(inst_by_pod)):
            pod_metrics = self._aggregate_pod(
                model,
                pod_key,
                hist_by_pod.get(pod_key, []),
                inst_by_pod.get(pod_key, []),
                window_start_ms,
                window_end_ms,
            )
            if pod_metrics is not None:
                per_pod[pod_metrics.pod] = pod_metrics
        return per_pod

    def _read_legacy_docs(
        self,
        prefix: str,
        model: str,
        window_start_ms: int,
        window_end_ms: int,
        *,
        lookback_ms: int = 0,
    ) -> dict[str, list[dict[str, Any]]]:
        keys: list[str] = []
        for raw_key in self._redis.scan_iter(prefix + "*"):
            key = _decode_text(raw_key)
            parsed = _parse_legacy_key(prefix, key)
            if parsed is None:
                continue
            _, ts_ms = parsed
            if max(0, window_start_ms - lookback_ms) <= ts_ms <= window_end_ms:
                keys.append(key)
        values = self._redis.mget(keys) if keys else []
        docs_by_pod: dict[str, list[dict[str, Any]]] = {}
        for key, raw_value in zip(keys, values):
            if raw_value is None:
                continue
            parsed = _parse_legacy_key(prefix, key)
            if parsed is None:
                continue
            pod_key, ts_ms = parsed
            try:
                doc = json.loads(_decode_text(raw_value))
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(doc, dict) or not _doc_has_model(doc, model):
                continue
            doc.setdefault("timestamp", ts_ms)
            docs_by_pod.setdefault(pod_key, []).append(doc)
        for docs in docs_by_pod.values():
            docs.sort(key=lambda doc: _number(doc.get("timestamp"), 0.0))
        if lookback_ms:
            return {pod: _with_baseline_doc(docs, window_start_ms) for pod, docs in docs_by_pod.items()}
        return docs_by_pod

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

        prompt_tokens = self._hist_sum_delta(model, HISTOGRAM_METRICS["prompt_tokens"], hist_docs, window_start_ms)
        generation_tokens = self._hist_sum_delta(model, HISTOGRAM_METRICS["generation_tokens"], hist_docs, window_start_ms)
        ttft_avg_s = self._hist_avg_delta(model, HISTOGRAM_METRICS["ttft"], hist_docs, window_start_ms)
        tpot_avg_s = self._hist_avg_delta(model, HISTOGRAM_METRICS["tpot"], hist_docs, window_start_ms)
        e2e_avg_s = self._hist_avg_delta(model, HISTOGRAM_METRICS["e2e"], hist_docs, window_start_ms)

        ttft_p95_s = self._hist_percentile(model, HISTOGRAM_METRICS["ttft"], hist_docs, window_start_ms)
        tpot_p95_s = self._hist_percentile(model, HISTOGRAM_METRICS["tpot"], hist_docs, window_start_ms)
        e2e_p95_s = self._hist_percentile(model, HISTOGRAM_METRICS["e2e"], hist_docs, window_start_ms)

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
            request_count=self._hist_count_delta(
                model, HISTOGRAM_METRICS["prompt_tokens"], hist_docs, window_start_ms
            ),
            token_counter_reset=(
                self._hist_counter_reset(
                    model, HISTOGRAM_METRICS["prompt_tokens"], hist_docs
                )
                or self._hist_counter_reset(
                    model, HISTOGRAM_METRICS["generation_tokens"], hist_docs
                )
            ),
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
            per_pod=per_pod,
            request_count=_sum_optional([pod.request_count for pod in pods]),
            token_counter_reset=any(pod.token_counter_reset for pod in pods),
        )

    def _hist_sum_delta(self, model: str, metric: str, docs: list[dict[str, Any]], window_start_ms: int) -> float | None:
        if not _has_window_hist_doc(docs, window_start_ms):
            return None
        first, last = _first_last_metric(model, metric, docs)
        if first is None or last is None:
            return 0.0
        delta = _number(last.get("sum"), 0.0) - _number(first.get("sum"), 0.0)
        return delta if delta >= 0.0 else None

    def _hist_counter_reset(
        self, model: str, metric: str, docs: list[dict[str, Any]]
    ) -> bool:
        first, last = _first_last_metric(model, metric, docs)
        if first is None or last is None:
            return False
        return _number(last.get("sum"), 0.0) < _number(first.get("sum"), 0.0)

    def _hist_count_delta(
        self, model: str, metric: str, docs: list[dict[str, Any]], window_start_ms: int
    ) -> float | None:
        if not _has_window_hist_doc(docs, window_start_ms):
            return None
        first, last = _first_last_metric(model, metric, docs)
        if first is None or last is None:
            return 0.0
        return max(
            0.0,
            _number(last.get("count"), 0.0) - _number(first.get("count"), 0.0),
        )

    def _hist_avg_delta(self, model: str, metric: str, docs: list[dict[str, Any]], window_start_ms: int) -> float | None:
        if not _has_window_metric(model, metric, docs, window_start_ms):
            return None
        first, last = _first_last_metric(model, metric, docs)
        if first is None or last is None:
            return None
        sum_delta = max(0.0, _number(last.get("sum"), 0.0) - _number(first.get("sum"), 0.0))
        count_delta = max(0.0, _number(last.get("count"), 0.0) - _number(first.get("count"), 0.0))
        if count_delta <= 0.0:
            return None
        return sum_delta / count_delta

    def _hist_percentile(self, model: str, metric: str, docs: list[dict[str, Any]], window_start_ms: int) -> float | None:
        if not _has_window_metric(model, metric, docs, window_start_ms):
            return None
        first, last = _first_last_metric(model, metric, docs)
        if first is None or last is None:
            return None
        if self._min_latency_samples > 0:
            count_delta = max(0.0, _number(last.get("count"), 0.0) - _number(first.get("count"), 0.0))
            if count_delta < self._min_latency_samples:
                # N1: too few observations for a stable p95 -> None (not 0), so the
                # signal/safescale layer treats the latency metric as unavailable rather
                # than deciding on noise.
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


def _parse_legacy_key(prefix: str, key: str) -> tuple[str, int] | None:
    if not key.startswith(prefix):
        return None
    body = key[len(prefix):]
    try:
        pod_key, raw_ts = body.rsplit("_", 1)
        return pod_key, _timestamp_to_ms(int(raw_ts))
    except (ValueError, TypeError):
        return None


def _timestamp_to_ms(raw: int) -> int:
    if raw > 1_000_000_000_000_000:
        return raw // 1_000_000
    if raw < 100_000_000_000:
        return raw * 1000
    return raw


def _doc_has_model(doc: dict[str, Any], model: str) -> bool:
    prefix = model + "/"
    for field in ("model_histogram_metrics", "model_metrics"):
        metrics = doc.get(field)
        if isinstance(metrics, dict) and any(isinstance(key, str) and key.startswith(prefix) for key in metrics):
            return True
    return False


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


def _has_window_metric(model: str, metric: str, docs: list[dict[str, Any]], window_start_ms: int) -> bool:
    for doc in docs:
        if _number(doc.get("timestamp"), 0.0) < window_start_ms:
            continue
        if _metric_entry(model, metric, doc) is not None:
            return True
    return False


def _has_window_hist_doc(docs: list[dict[str, Any]], window_start_ms: int) -> bool:
    return any(
        _number(doc.get("timestamp"), 0.0) >= window_start_ms
        and isinstance(doc.get("model_histogram_metrics"), dict)
        for doc in docs
    )


def _first_last_metric(model: str, metric: str, docs: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    entries = [_metric_entry(model, metric, doc) for doc in docs]
    entries = [entry for entry in entries if entry is not None]
    if not entries:
        return None, None
    return entries[0], entries[-1]


def _with_baseline_doc(docs: list[dict[str, Any]], window_start_ms: int) -> list[dict[str, Any]]:
    baseline = None
    window_docs: list[dict[str, Any]] = []
    for doc in docs:
        if _number(doc.get("timestamp"), 0.0) < window_start_ms:
            baseline = doc
        else:
            window_docs.append(doc)
    if baseline is None:
        return window_docs
    return [baseline, *window_docs]


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



def _sum_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _max_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None

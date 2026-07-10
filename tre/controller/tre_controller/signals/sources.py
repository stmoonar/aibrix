from __future__ import annotations

import math
from dataclasses import dataclass

from tre_common.metrics_schema import ModelWindowMetrics
from tre_common.registry import AltThreshold, ModelSpec

SignalSource = str

EPS = 1e-6
Z_MAX = 10.0


@dataclass(frozen=True)
class SignalValue:
    source: SignalSource
    raw_value: float | None
    z_m: float | None
    unavailable_reason: str | None = None


def get_signal(
    metrics: ModelWindowMetrics,
    spec: ModelSpec,
    source: SignalSource,
    *,
    trs_z_m: float | None,
) -> SignalValue:
    if source == "zm":
        return SignalValue(source=source, raw_value=trs_z_m, z_m=trs_z_m)
    if source == "latency_p95":
        return _latency_signal(metrics, spec)
    if source == "queue_len":
        return _queue_signal(metrics, spec)
    if source == "decode_tps":
        return _token_rate_signal(
            metrics, spec, source, metrics.generation_tokens
        )
    if source == "prefill_tps":
        return _token_rate_signal(metrics, spec, source, metrics.prompt_tokens)
    if source == "kv_cache":
        return _kv_cache_signal(metrics)
    raise ValueError(f"unsupported signal source: {source}")


def _latency_signal(metrics: ModelWindowMetrics, spec: ModelSpec) -> SignalValue:
    pairs = (
        (metrics.ttft_p95_ms, spec.slo.ttft_p95_ms),
        (metrics.tpot_p95_ms, spec.slo.tpot_p95_ms),
        (metrics.e2e_p95_ms, spec.slo.e2e_p95_ms),
    )
    samples: list[tuple[float, float]] = []
    for observed, slo in pairs:
        observed_value = _positive_float(observed)
        slo_value = _positive_float(slo)
        if observed_value is not None and slo_value is not None:
            samples.append((observed_value, slo_value))
    if not samples:
        return SignalValue("latency_p95", raw_value=None, z_m=None, unavailable_reason="latency_p95_missing")
    health = min(slo / observed for observed, slo in samples)
    return SignalValue("latency_p95", raw_value=max(observed for observed, _slo in samples), z_m=health)


def _queue_signal(metrics: ModelWindowMetrics, spec: ModelSpec) -> SignalValue:
    control_queue = max(
        0.0,
        metrics.avg_running + metrics.avg_swapping + metrics.avg_waiting * spec.trs.lambda_wait,
    )
    threshold = spec.alt_thresholds.get("queue_len")
    z_m = normalize_signal(control_queue, threshold)
    if z_m is None:
        return SignalValue(
            "queue_len",
            raw_value=control_queue,
            z_m=None,
            unavailable_reason="queue_threshold_missing",
        )
    return SignalValue("queue_len", raw_value=control_queue, z_m=z_m)


def per_replica_token_rate(
    metrics: ModelWindowMetrics, token_total: float | int | None
) -> float | None:
    if (
        token_total is None
        or metrics.routable_pods <= 0
        or metrics.token_counter_reset
    ):
        return None
    total = float(token_total)
    duration_s = (metrics.window_end_ms - metrics.window_start_ms) / 1000.0
    if not math.isfinite(total) or total < 0.0 or duration_s <= 0.0:
        return None
    return total / duration_s / metrics.routable_pods


def _token_rate_signal(
    metrics: ModelWindowMetrics,
    spec: ModelSpec,
    source: str,
    token_total: float | int | None,
) -> SignalValue:
    raw_value = per_replica_token_rate(metrics, token_total)
    if raw_value is None:
        return SignalValue(
            source,
            raw_value=None,
            z_m=None,
            unavailable_reason=f"{source}_counter_missing",
        )
    z_m = normalize_signal(raw_value, spec.alt_thresholds.get(source))
    if z_m is None:
        return SignalValue(
            source,
            raw_value=raw_value,
            z_m=None,
            unavailable_reason=f"{source}_threshold_missing",
        )
    return SignalValue(source, raw_value=raw_value, z_m=z_m)


def normalize_signal(value: float | int | None, threshold: AltThreshold | None) -> float | None:
    if threshold is None or threshold.theta <= 0.0 or value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        return None
    if threshold.direction == "higher_is_healthier":
        normalized = parsed / threshold.theta
    elif threshold.direction == "lower_is_healthier":
        normalized = threshold.theta / max(parsed, EPS)
    else:
        return None
    return min(Z_MAX, normalized)


def _kv_cache_signal(metrics: ModelWindowMetrics) -> SignalValue:
    hit_rate = _bounded_unit(metrics.kv_cache_hit_rate)
    if hit_rate is None:
        return SignalValue("kv_cache", raw_value=None, z_m=None, unavailable_reason="kv_cache_missing")
    return SignalValue("kv_cache", raw_value=hit_rate, z_m=hit_rate / 0.5)


def _positive_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if math.isnan(parsed) or math.isinf(parsed) or parsed <= 0.0:
        return None
    return parsed


def _bounded_unit(value: float | int | None) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return min(1.0, max(0.0, parsed))

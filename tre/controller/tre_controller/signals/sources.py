from __future__ import annotations

import math
from dataclasses import dataclass

from tre_common.metrics_schema import ModelWindowMetrics
from tre_common.registry import ModelSpec

SignalSource = str


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
        spec.trs.qmin,
        metrics.avg_running + metrics.avg_swapping + metrics.avg_waiting * spec.trs.lambda_wait,
    )
    qsat = _positive_float(spec.trs.qsat)
    if qsat is None or control_queue <= 0.0:
        return SignalValue("queue_len", raw_value=control_queue, z_m=None, unavailable_reason="queue_threshold_missing")
    return SignalValue("queue_len", raw_value=control_queue, z_m=qsat / control_queue)


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

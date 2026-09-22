"""Control signals the classifier can run on: TSS (``zm``) and the ablation arms.

Every alternative signal goes through :func:`_thresholded_signal`: its raw value comes
from :mod:`tre_common.alt_signals` (the same function the offline fit uses), is smoothed
there by the shared wall-clock EMA (:func:`tre_common.tss.signal_ema`, the registry's
``ema_tau_ms``, i.e. the TSS tau) when a :class:`~tre_controller.signals.trs.SignalState`
is supplied, and is only then normalised to ``z``. TSS itself is smoothed by the same
``ema_step`` inside ``TRSComputer`` before ``Z = TSS / theta`` (plan §6.9 item 4).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tre_common.alt_signals import (
    EPS,
    Z_MAX,
    normalize_signal_value,
    queue_len_per_replica,
    token_rate_per_replica,
)
from tre_common.metrics_schema import ModelWindowMetrics
from tre_common.registry import AltThreshold, ModelSpec
from tre_common.tss import DEFAULT_EMA_TAU_MS

if TYPE_CHECKING:  # pragma: no cover
    from tre_controller.signals.trs import SignalState

SignalSource = str

__all__ = ["EPS", "Z_MAX", "SignalValue", "get_signal", "normalize_signal", "per_replica_token_rate"]


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
    signal_state: "SignalState | None" = None,
) -> SignalValue:
    """``source``'s value and ``z`` for this window.

    ``signal_state`` carries the per-(model, signal) EMA across ticks; without it the
    alternative signals are scored raw (a fresh EMA seeds at the raw value), exactly as
    a fresh ``TRSComputer`` does for TSS.
    """
    if source == "zm":
        return SignalValue(source=source, raw_value=trs_z_m, z_m=trs_z_m)
    if source == "latency_p95":
        return _latency_signal(metrics, spec)
    if source == "queue_len":
        raw = queue_len_per_replica(metrics.avg_running, metrics.avg_waiting, metrics.routable_pods)
        return _thresholded_signal(metrics, spec, source, raw, "queue_len_missing", "queue_threshold_missing", signal_state)
    if source == "decode_tps":
        return _thresholded_signal(
            metrics, spec, source, per_replica_token_rate(metrics, metrics.generation_tokens),
            f"{source}_counter_missing", f"{source}_threshold_missing", signal_state,
        )
    if source == "prefill_tps":
        return _thresholded_signal(
            metrics, spec, source, per_replica_token_rate(metrics, metrics.prompt_tokens),
            f"{source}_counter_missing", f"{source}_threshold_missing", signal_state,
        )
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


def per_replica_token_rate(
    metrics: ModelWindowMetrics, token_total: float | int | None
) -> float | None:
    if token_total is None or metrics.token_counter_reset:
        return None
    return token_rate_per_replica(
        token_total, metrics.window_end_ms - metrics.window_start_ms, metrics.routable_pods
    )


def _thresholded_signal(
    metrics: ModelWindowMetrics,
    spec: ModelSpec,
    source: str,
    raw_value: float | None,
    missing_reason: str,
    threshold_missing_reason: str,
    signal_state: "SignalState | None",
) -> SignalValue:
    """The one place every alternative signal is smoothed and normalised."""
    if raw_value is None:
        return SignalValue(source, raw_value=None, z_m=None, unavailable_reason=missing_reason)
    if signal_state is not None:
        tau_ms = spec.trs.ema_tau_ms if spec.trs.ema_tau_ms else DEFAULT_EMA_TAU_MS
        raw_value = signal_state.smooth_signal(
            spec.name, source, raw_value, window_end_ms=metrics.window_end_ms, tau_ms=tau_ms,
            window_ms=float(metrics.window_end_ms - metrics.window_start_ms),
        )
    z_m = normalize_signal(raw_value, spec.alt_thresholds.get(source))
    if z_m is None:
        return SignalValue(
            source,
            raw_value=raw_value,
            z_m=None,
            unavailable_reason=threshold_missing_reason,
        )
    return SignalValue(source, raw_value=raw_value, z_m=z_m)


def normalize_signal(value: float | int | None, threshold: AltThreshold | None) -> float | None:
    if threshold is None:
        return None
    return normalize_signal_value(value, threshold.theta, threshold.direction)


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

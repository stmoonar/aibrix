"""Control signals the classifier can run on: TSS (``zm``) and the ablation arms.

Sources (``TRE_SIGNAL_SOURCE``): ``zm`` (TSS), the v1 ablation signals ``queue_len`` /
``ttft_mean`` / ``kv_usage`` (definitions in :mod:`tre_common.alt_signals`), the plan
section 6.9 rates ``decode_tps`` / ``prefill_tps``, and ``latency_p95`` (worst p95/SLO
ratio; not fitted, not v1's latency signal - not used by the ablation). ``ttft_mean`` and
``kv_usage`` are built from the per-pod window data and are unknown unless every routable
replica reported (:func:`_serving_pods`).

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
    kv_headroom,
    normalize_signal_value,
    queue_len_per_replica,
    token_rate_per_replica,
    ttft_mean_value,
)
from tre_common.metrics_schema import ModelWindowMetrics, PodWindowMetrics
from tre_common.registry import AltThreshold, ModelSpec
from tre_common.tss import DEFAULT_EMA_TAU_MS, window_is_idle

if TYPE_CHECKING:  # pragma: no cover
    from tre_controller.signals.trs import SignalState

SignalSource = str

__all__ = [
    "EPS", "Z_MAX", "SignalValue", "get_signal", "model_kv_usage_sum", "model_mean_ttft_ms",
    "normalize_signal", "per_replica_token_rate",
]


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
    if source == "ttft_mean":
        pods = _serving_pods(metrics)
        if pods is None:
            return SignalValue(source, raw_value=None, z_m=None, unavailable_reason="ttft_mean_missing")
        mean_ms = model_mean_ttft_ms(pods)
        if mean_ms is _MISSING:
            return SignalValue(source, raw_value=None, z_m=None, unavailable_reason="ttft_mean_missing")
        raw = ttft_mean_value(mean_ms)
        if raw is None:
            # Fresh pods, no first token in the window: hold the last EMA (v1), bounded
            # by the EMA's own gap / idle rules (SignalState.held_signal).
            return _held_signal(metrics, spec, source, "ttft_mean_no_sample", signal_state)
        return _thresholded_signal(metrics, spec, source, raw, "ttft_mean_missing", "ttft_mean_threshold_missing", signal_state)
    if source == "kv_usage":
        pods = _serving_pods(metrics)
        usage_sum = None if pods is None else model_kv_usage_sum(pods)
        raw = kv_headroom(usage_sum, metrics.routable_pods)
        return _thresholded_signal(metrics, spec, source, raw, "kv_usage_missing", "kv_usage_threshold_missing", signal_state)
    raise ValueError(f"unsupported signal source: {source}")


#: ``model_mean_ttft_ms`` result when some serving pod's TTFT histogram was not read.
_MISSING = object()


def _serving_pods(metrics: ModelWindowMetrics) -> list[PodWindowMetrics] | None:
    """The pods of a serving window whose data the model-level ttft_mean / kv_usage are
    built from, or None when the window cannot speak for every routable replica (fail
    closed: unknown, never healthy or idle).

    ``metrics`` is the serving window (``restrict_to_serving``: sleeping pods dropped,
    ``routable_pods`` = the fleet state's serving count). The store leaves a pod without
    a sample in the window out of ``per_pod`` (``scrape_stale_pods``), so fewer pods than
    routable replicas means a serving replica is unknown. Hidden-but-awake pods (a
    SafeScale probe) stay in the window, as in the queue sum.
    """
    routable = int(metrics.routable_pods or 0)
    pods = list((metrics.per_pod or {}).values())
    if routable < 1 or len(pods) < routable:
        return None
    return pods


def model_mean_ttft_ms(pods: list[PodWindowMetrics]) -> float | None | object:
    """Model window mean TTFT (ms): the pods' histogram means pooled by their first-token
    counts (sum of sums / sum of counts, v1 D38). None = no first token in the window;
    :data:`_MISSING` = some pod's TTFT histogram was not read."""
    total_ms = 0.0
    count = 0.0
    for pod in pods:
        n = pod.ttft_count
        if n is None or not math.isfinite(float(n)) or float(n) < 0.0:
            return _MISSING
        if float(n) == 0.0:
            continue
        avg = pod.ttft_avg_ms
        if avg is None or not math.isfinite(float(avg)):
            return _MISSING
        total_ms += float(avg) * float(n)
        count += float(n)
    return total_ms / count if count > 0.0 else None


def model_kv_usage_sum(pods: list[PodWindowMetrics]) -> float | None:
    """Sum of the pods' window-mean KV-cache fill (v1 ``avg_gpu_cache``); None when a pod
    did not report it (unknown, never 0)."""
    total = 0.0
    for pod in pods:
        usage = pod.gpu_cache_usage
        if usage is None or not math.isfinite(float(usage)):
            return None
        total += float(usage)
    return total


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


def alt_signal_tau_ms(spec: ModelSpec) -> float | None:
    """The time constant an alternative signal is smoothed with: the registry's TSS tau.

    ``ema_tau_ms`` unset (None) keeps :data:`DEFAULT_EMA_TAU_MS`. ``ema_tau_ms <= 0`` is
    tau = 0, i.e. alpha = 1: no smoothing, the raw value is used - as offline
    (``theta_verdict.build_signal_spec`` maps tau <= 0 to no EMA). It used to be read as
    "unset" and fall back to the 20 s default, so a registry publishing tau = 0 smoothed
    the ablation signals online with a tau the fit never saw.
    """
    tau = spec.trs.ema_tau_ms
    if tau is None:
        return DEFAULT_EMA_TAU_MS
    tau = float(tau)
    return tau if tau > 0 else None


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
    tau_ms = alt_signal_tau_ms(spec)
    if signal_state is not None and tau_ms is not None:
        raw_value = signal_state.smooth_signal(
            spec.name, source, raw_value, window_end_ms=metrics.window_end_ms, tau_ms=tau_ms,
            window_ms=float(metrics.window_end_ms - metrics.window_start_ms),
            idle=window_is_idle(metrics.prompt_tokens, metrics.generation_tokens),
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


def _held_signal(
    metrics: ModelWindowMetrics,
    spec: ModelSpec,
    source: str,
    no_sample_reason: str,
    signal_state: "SignalState | None",
) -> SignalValue:
    """A window without a sample of ``source``: the last smoothed value, normalised, or
    unavailable when there is none (no state, no smoothing, or the EMA was cleared)."""
    tau_ms = alt_signal_tau_ms(spec)
    held = None
    if signal_state is not None and tau_ms is not None:
        held = signal_state.held_signal(
            spec.name, source, window_end_ms=metrics.window_end_ms, tau_ms=tau_ms,
            window_ms=float(metrics.window_end_ms - metrics.window_start_ms),
            idle=window_is_idle(metrics.prompt_tokens, metrics.generation_tokens),
        )
    if held is None:
        return SignalValue(source, raw_value=None, z_m=None, unavailable_reason=no_sample_reason)
    z_m = normalize_signal(held, spec.alt_thresholds.get(source))
    if z_m is None:
        return SignalValue(source, raw_value=held, z_m=None, unavailable_reason=f"{source}_threshold_missing")
    return SignalValue(source, raw_value=held, z_m=z_m)


def normalize_signal(value: float | int | None, threshold: AltThreshold | None) -> float | None:
    if threshold is None:
        return None
    return normalize_signal_value(value, threshold.theta, threshold.direction)


def _positive_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if math.isnan(parsed) or math.isinf(parsed) or parsed <= 0.0:
        return None
    return parsed


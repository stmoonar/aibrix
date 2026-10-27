"""Values of the alternative (ablation) signals, shared by the controller and every fit.

Plan 2026-09-21 §6.9 and the v1 signal ablation (10-23, D30-D40): TSS is compared against
single-metric signals. That comparison is only about the signals if each alternative means
the same thing online (``tre_controller.signals.sources``) and offline
(``tre_calibration.alt_signals`` -> ``fit_alt_thresholds`` / ``theta_verdict``), so the
quantity and its normalisation to ``z`` live here, once:

* ``queue_len`` is the **queue length per routable replica**,
  ``max((running + waiting) / max(1, routable_pods), 1)``: no ``lambda_wait`` (the
  threshold is fitted in raw units and does not depend on the TSS weight), no swapping
  (vLLM v1 never swaps). The floor of one request per replica is v1's
  (``alt_signals.QUEUE_FLOOR_PER_POD``, the TSS ``qmin``): below one request per replica
  the queue says nothing more about pressure, and the floor keeps an idle model at a
  finite ``z = theta`` instead of the ``Z_MAX`` cap. The metrics store sums
  running/waiting over pods, so the fleet sum used before shrank ``z`` roughly N-fold
  after scaling out to N replicas: that arm then read CRITICAL forever and never scaled in;
* ``ttft_mean`` is the model's **server-side mean TTFT** of the window in ms (v1 D38: the
  vLLM TTFT histogram ``sum / count`` deltas pooled over the pods, i.e. weighted by each
  pod's first-token count), floored at 1 ms (v1 ``LATENCY_FLOOR_MS``). A window without a
  first token has no value;
* ``kv_usage`` is the model's **KV-cache headroom** ``max(1 - u, 0.001)``,
  ``u = min(1, sum of the pods' window-mean KV-cache fill / routable_pods)`` (v1
  ``SIGNAL_KV``: vLLM ``kv_cache_usage_perc``, not the prefix-cache hit rate). Higher is
  healthier; the floor is v1's ``KV_HEADROOM_FLOOR``;
* ``decode_tps`` / ``prefill_tps`` are processed tokens per second per routable replica;
* ``z`` = ``value / theta`` for ``higher_is_healthier``, ``theta / max(value, EPS)`` for
  ``lower_is_healthier``, capped at :data:`Z_MAX`.

Smoothing is :func:`tre_common.tss.signal_ema` (same tau-EMA as TSS) over the value
above, online and offline. Nothing here may import from the controller or calibration
packages: both import it.
"""
from __future__ import annotations

import math
from typing import Optional

EPS = 1e-6
Z_MAX = 10.0

#: v1 floors (``tre_code`` ``controller/alt_signals.py`` FLOORS).
QUEUE_FLOOR_PER_REPLICA = 1.0
TTFT_FLOOR_MS = 1.0
KV_HEADROOM_FLOOR = 1e-3

#: What each fitted alternative signal's value IS (the formulas in this module). A theta
#: is only valid for the definition and the metrics window it was fitted on, so the fit
#: stamps ``alt_thresholds.<signal>.basis = {definition, window_ms}`` and the controller
#: refuses to start an alt arm on a missing or different stamp (:func:`alt_basis_errors`,
#: the alternative-signal twin of ``tre_common.tss.theta_basis_errors``). Change an id
#: whenever its value changes (floor, pooling, counter source, ...).
ALT_SIGNAL_DEFINITIONS = {
    "queue_len": "queue_len:max((running+waiting)/max(1,routable),1):20261028",
    "ttft_mean": "ttft_mean:max(sum_pods(ttft_sum)/sum_pods(ttft_count),1ms):20261028",
    "kv_usage": "kv_usage:max(1-min(1,sum_pods(kv_fill)/routable),0.001):20261028",
    "decode_tps": "decode_tps:step_counter_generation_tokens/s/routable:20261028",
    "prefill_tps": "prefill_tps:step_counter_prompt_tokens/s/routable:20261028",
}

#: Per-signal orientation (mirrors tre_common.registry.EXPECTED_SIGNAL_DIRECTIONS).
LOWER_IS_HEALTHIER = "lower_is_healthier"
HIGHER_IS_HEALTHIER = "higher_is_healthier"


def _finite(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def queue_len_per_replica(
    avg_running: object, avg_waiting: object, routable_pods: object
) -> Optional[float]:
    """``max((running + waiting) / max(1, routable_pods), 1)`` - the queue_len signal.

    ``running`` / ``waiting`` are the fleet totals the metrics store reports (sums over
    the model's pods); a missing waiting count reads as 0, a missing running count makes
    the value unavailable.
    """
    running = _finite(avg_running)
    if running is None:
        return None
    waiting = _finite(avg_waiting) or 0.0
    pods = _finite(routable_pods)
    replicas = max(1.0, pods if pods is not None else 1.0)
    return max(max(0.0, running + waiting) / replicas, QUEUE_FLOOR_PER_REPLICA)


def ttft_mean_value(mean_ttft_ms: object) -> Optional[float]:
    """The ttft_mean signal from the model's window mean TTFT (ms): floored at
    :data:`TTFT_FLOOR_MS`; None when there is no first token in the window (missing,
    non-finite or <= 0 - v1 ``window_value``)."""
    mean = _finite(mean_ttft_ms)
    if mean is None or mean <= 0.0:
        return None
    return max(mean, TTFT_FLOOR_MS)


def kv_headroom(usage_sum: object, pods: object) -> Optional[float]:
    """The kv_usage signal: ``max(1 - min(1, usage_sum / pods), KV_HEADROOM_FLOOR)``.

    ``usage_sum`` is the sum over the serving pods of their window-mean KV-cache fill
    (0..1 each); ``pods`` the routable replica count. None when either is unknown or no
    replica serves (never an idle-looking headroom of 1)."""
    total = _finite(usage_sum)
    count = _finite(pods)
    if total is None or count is None or count < 1.0:
        return None
    usage = min(1.0, max(0.0, total) / count)
    return max(1.0 - usage, KV_HEADROOM_FLOOR)


def token_rate_per_replica(
    token_total: object, window_ms: object, routable_pods: object
) -> Optional[float]:
    """Processed tokens per second per routable replica, or ``None`` when not computable
    (missing counter, non-positive window, no routable replica)."""
    total = _finite(token_total)
    duration_ms = _finite(window_ms)
    pods = _finite(routable_pods)
    if total is None or total < 0.0 or duration_ms is None or duration_ms <= 0.0:
        return None
    if pods is None or pods <= 0.0:
        return None
    return total / (duration_ms / 1000.0) / pods


def normalize_signal_value(
    value: object, theta: Optional[float], direction: str
) -> Optional[float]:
    """``z`` of one signal value against its threshold (``None`` when undefined)."""
    if theta is None or not (float(theta) > 0.0) or value is None:
        return None
    parsed = _finite(value)
    if parsed is None or parsed < 0.0:
        return None
    if direction == HIGHER_IS_HEALTHIER:
        normalized = parsed / float(theta)
    elif direction == LOWER_IS_HEALTHIER:
        normalized = float(theta) / max(parsed, EPS)
    else:
        return None
    return min(Z_MAX, normalized)


def alt_basis_errors(models, signal: str, *, window_ms: float) -> list[str]:
    """Why the registry's ``alt_thresholds.<signal>`` do not fit this controller, one
    line per model (empty: they fit). Each model's threshold must carry
    ``basis.definition == ALT_SIGNAL_DEFINITIONS[signal]`` and ``basis.window_ms ==
    window_ms`` (TRE_METRICS_WINDOW_MS). A missing stamp is a mismatch."""
    want = ALT_SIGNAL_DEFINITIONS.get(signal)
    errors: list[str] = []
    for spec in models:
        name = getattr(spec, "name", "?")
        threshold = (getattr(spec, "alt_thresholds", None) or {}).get(signal)
        definition = getattr(threshold, "basis_definition", None)
        window = getattr(threshold, "basis_window_ms", None)
        if want is None or definition != want or window is None or float(window) != float(window_ms):
            errors.append(
                f"{name}: alt_thresholds.{signal}.basis definition={definition!r} window_ms={window!r}, "
                f"controller computes definition={want!r} on TRE_METRICS_WINDOW_MS={int(window_ms)}"
            )
    return errors


__all__ = (
    "ALT_SIGNAL_DEFINITIONS",
    "EPS",
    "HIGHER_IS_HEALTHIER",
    "KV_HEADROOM_FLOOR",
    "LOWER_IS_HEALTHIER",
    "QUEUE_FLOOR_PER_REPLICA",
    "TTFT_FLOOR_MS",
    "Z_MAX",
    "alt_basis_errors",
    "kv_headroom",
    "normalize_signal_value",
    "queue_len_per_replica",
    "token_rate_per_replica",
    "ttft_mean_value",
)

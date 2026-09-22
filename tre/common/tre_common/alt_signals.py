"""Values of the alternative (ablation) signals, shared by the controller and every fit.

Plan 2026-09-21 §6.9: TSS is compared against queue length and the per-replica completed
token rates. That comparison is only about the signals if each alternative means the same
thing online (``tre_controller.signals.sources``) and offline
(``tre_calibration.alt_signals`` -> ``fit_alt_thresholds`` / ``theta_verdict``), so the
quantity and its normalisation to ``z`` live here, once:

* ``queue_len`` is the **raw queue length per routable replica**,
  ``(running + waiting) / max(1, routable_pods)``: no ``lambda_wait`` (the threshold is
  fitted in raw units and does not depend on the TSS weight), no swapping (vLLM v1 never
  swaps). The metrics store sums running/waiting over pods, so the fleet sum used before
  shrank ``z`` roughly N-fold after scaling out to N replicas: that arm then read CRITICAL
  forever and never scaled in;
* ``decode_tps`` / ``prefill_tps`` are completed tokens per second per routable replica;
* ``z`` = ``value / theta`` for ``higher_is_healthier``, ``theta / max(value, EPS)`` for
  ``lower_is_healthier``, capped at :data:`Z_MAX`.

Smoothing is :func:`tre_common.tss.signal_ema` (same tau-EMA as TSS). Nothing here may
import from the controller or calibration packages: both import it.
"""
from __future__ import annotations

import math
from typing import Optional

EPS = 1e-6
Z_MAX = 10.0

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
    """``(running + waiting) / max(1, routable_pods)`` - the queue_len ablation signal.

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
    return max(0.0, running + waiting) / replicas


def token_rate_per_replica(
    token_total: object, window_ms: object, routable_pods: object
) -> Optional[float]:
    """Completed tokens per second per routable replica, or ``None`` when not computable
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


__all__ = (
    "EPS",
    "HIGHER_IS_HEALTHIER",
    "LOWER_IS_HEALTHIER",
    "Z_MAX",
    "normalize_signal_value",
    "queue_len_per_replica",
    "token_rate_per_replica",
)

"""The alternative signals TSS is compared against in the signal ablation.

The ablation asks whether TSS separates healthy from violating windows better than the
obvious cheaper signals: queue length and the per-replica completed-token rates. That
question is only answerable if every signal was thresholded by the *same* criterion --
otherwise the comparison measures the criteria. So this module owns nothing but the
per-signal facts (which column, which orientation) and hands the actual fitting to
:func:`tre_calibration.fit.fit_theta`, the same entry point ``tre_calibration.cli`` uses
for TSS.

Orientation matters here and is the one place the signals genuinely differ from TSS.
TSS is ``higher_is_healthier``: a large score means headroom. Queue length and the token
rates are pressure signals -- across an R3 load scan a larger value means the replica is
closer to saturation -- so they are ``lower_is_healthier``. Thresholds stay in raw signal
units; no reciprocal transform is written into the registry.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

from tre_calibration.dataset import CalibrationWindow
from tre_common.alt_signals import queue_len_per_replica, token_rate_per_replica
from tre_calibration.fit import (
    BalancedAccuracyThetaFit,
    ReliabilityThetaFit,
    threshold_balanced_accuracy,
)

#: Per signal: the CSV column it is read from (``None`` means it is derived by
#: :func:`alt_signal_transform` through :mod:`tre_common.alt_signals`, the functions the
#: controller uses online) and its orientation. The orientation is the prior one and stays
#: hardcoded; the fit reports balanced accuracy in both directions (plan §6.9).
ALT_SIGNALS: Mapping[str, tuple[str | None, str]] = {
    "queue_len": (None, "lower_is_healthier"),
    "decode_tps": (None, "lower_is_healthier"),
    "prefill_tps": (None, "lower_is_healthier"),
}

#: Which token counter each rate signal is computed from.
_TOKEN_COLUMN = {
    "decode_tps": "generation_tokens_total",
    "prefill_tps": "prompt_tokens_total",
}

#: Theta candidate grid per signal (``ThetaFitConfig.candidate_grid``). queue_len is a
#: small-integer-valued signal with a heavy mass at a few values, so fixed healthy
#: quantiles collapse onto the same threshold; it searches the distinct observed values
#: inside the same quantile range instead (plan §6.9).
_CANDIDATE_GRID = {"queue_len": "unique"}

#: A signal whose AUROC (in its prior direction) is below this does not rank windows by
#: health well enough to be a baseline and is reported "inert" (plan §6.9).
INERT_AUROC = 0.6


def alt_signal_names() -> list[str]:
    return sorted(ALT_SIGNALS)


def alt_signal_direction(signal: str) -> str:
    """Orientation of ``signal`` -- the fact a fair comparison must not get wrong."""
    return _entry(signal)[1]


def alt_signal_column(signal: str) -> str | None:
    """CSV column for ``signal``, or ``None`` when it has to be derived."""
    return _entry(signal)[0]


def alt_signal_candidate_grid(signal: str) -> str:
    _entry(signal)
    return _CANDIDATE_GRID.get(signal, "quantile")


def _entry(signal: str) -> tuple[str | None, str]:
    try:
        return ALT_SIGNALS[signal]
    except KeyError as exc:
        raise KeyError(f"unknown alternative signal: {signal}") from exc


def _row_float(row: Mapping[str, Any], column: str, default: float | None = None) -> float | None:
    value = row.get(column)
    if value is None or str(value).strip() == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def alt_signal_transform(signal: str) -> Callable[[Mapping[str, Any]], float | None]:
    """Row transform computing ``signal`` from a window CSV row exactly as the controller
    computes it from a metrics window (:mod:`tre_common.alt_signals`). A CSV without a
    ``routable_pods`` column is a single-replica capture (routable_pods = 1)."""
    _entry(signal)
    if signal == "queue_len":

        def queue(row: Mapping[str, Any]) -> float | None:
            return queue_len_per_replica(
                _row_float(row, "avg_running"),
                _row_float(row, "avg_waiting", 0.0),
                _row_float(row, "routable_pods", 1.0),
            )

        return queue
    return per_replica_token_rate_transform(signal)


def per_replica_token_rate_transform(signal: str) -> Callable[[Mapping[str, Any]], float | None]:
    """Row transform turning a token counter into a per-replica completed-token rate."""
    token_column = _TOKEN_COLUMN[signal]

    def transform(row: Mapping[str, Any]) -> float | None:
        start = _row_float(row, "window_start_ms")
        end = _row_float(row, "window_end_ms")
        if start is None or end is None:
            return None
        return token_rate_per_replica(
            _row_float(row, token_column), end - start, _row_float(row, "routable_pods", 1.0)
        )

    return transform


def threshold_curve(
    windows: Sequence[CalibrationWindow],
    *,
    direction: str,
) -> list[dict[str, Any]]:
    """Per-candidate diagnostic curve over every distinct observed signal value.

    Carries both scores so the artifact shows *why* the published threshold won:
    ``balanced_accuracy`` is the criterion that selects it, ``attainment`` is the
    one-sided containment score the criterion replaced. Reading them side by side is
    how the low-theta bias of the containment rule stays visible after the fact.
    """
    lower = direction == "lower_is_healthier"
    rows: list[dict[str, Any]] = []
    finite = [window for window in windows if math.isfinite(window.signal)]
    for theta in sorted({window.signal for window in finite}):
        subset = [
            window
            for window in finite
            if (window.signal <= theta if lower else window.signal >= theta)
        ]
        families = Counter(window.scenario_family for window in subset)
        metrics = threshold_balanced_accuracy(finite, theta=theta, direction=direction)
        rows.append(
            {
                "theta": theta,
                "support": len(subset),
                "attainment": (
                    sum(1 for window in subset if window.slo_met) / len(subset)
                    if subset
                    else 0.0
                ),
                "healthy": sum(1 for window in subset if window.slo_met),
                "violations": sum(1 for window in subset if not window.slo_met),
                "scenario_families": len(families),
                "max_family_ratio": (
                    max(families.values()) / len(subset) if subset else 0.0
                ),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "recall_good": metrics["recall_good"],
                "specificity_bad": metrics["specificity_bad"],
            }
        )
    return rows


def fit_report(
    fit: ReliabilityThetaFit | BalancedAccuracyThetaFit,
    windows: Sequence[CalibrationWindow],
    *,
    direction: str,
) -> dict[str, Any]:
    """Acceptance numbers for ``fit``, in the shape the alt-threshold artifact records."""
    report: dict[str, Any] = {
        "coverage_pass": fit.coverage_pass,
        "family_counts": dict(sorted(fit.family_counts.items())),
        "candidate_count": fit.candidate_count,
        "reject_reason": fit.reject_reason,
    }
    if isinstance(fit, BalancedAccuracyThetaFit):
        report.update(
            {
                "balanced_accuracy": fit.balanced_accuracy,
                "healthy_quantile": fit.healthy_quantile,
                "healthy_window_count": fit.healthy_window_count,
                "min_healthy_recall": fit.min_healthy_recall,
                "precision_good": fit.precision_good,
                "recall_good": fit.recall_good,
                "specificity_bad": fit.specificity_bad,
                "violating_window_count": fit.violating_window_count,
            }
        )
    else:
        report.update(
            {
                "attainment": fit.attainment,
                "confidence": fit.confidence,
                "support": fit.support,
            }
        )
        if fit.theta is not None:
            report["balanced_accuracy"] = threshold_balanced_accuracy(
                windows, theta=fit.theta, direction=direction
            )["balanced_accuracy"]
    return report

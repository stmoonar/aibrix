from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from tre_calibration.dataset import CalibrationWindow


@dataclass(frozen=True)
class FittedTheta:
    signal_name: str
    theta: float
    violation_max: float
    healthy_min: float
    sample_count: int


@dataclass(frozen=True)
class ReliabilityThetaFit:
    publish: bool
    theta: float | None
    support: int
    attainment: float
    confidence: float
    coverage_pass: bool
    family_counts: dict[str, int]
    reject_reason: str | None
    candidate_count: int


def fit_theta_from_health(
    windows: Iterable[CalibrationWindow],
    *,
    signal_name: str = "signal",
) -> FittedTheta:
    rows = list(windows)
    healthy = [row.signal for row in rows if row.slo_met and math.isfinite(row.signal)]
    violations = [row.signal for row in rows if not row.slo_met and math.isfinite(row.signal)]
    if not healthy:
        raise ValueError("cannot fit theta without healthy windows")
    if not violations:
        raise ValueError("cannot fit theta without violating windows")

    healthy_min = min(healthy)
    violation_max = max(violations)
    if violation_max >= healthy_min:
        raise ValueError("healthy and violating windows are not separable by a higher-is-healthier threshold")

    return FittedTheta(
        signal_name=signal_name,
        theta=(violation_max + healthy_min) / 2.0,
        violation_max=violation_max,
        healthy_min=healthy_min,
        sample_count=len(rows),
    )


def fit_theta_by_reliability(
    windows: Iterable[CalibrationWindow],
    *,
    reliability_target: float,
    min_support: int,
    min_confidence: float,
    min_scenario_families: int,
    max_single_scenario_ratio: float,
    direction: str = "higher_is_healthier",
) -> ReliabilityThetaFit:
    if direction not in {"higher_is_healthier", "lower_is_healthier"}:
        raise ValueError("direction must be higher_is_healthier or lower_is_healthier")
    rows = [row for row in windows if math.isfinite(row.signal)]
    candidates = sorted(
        {row.signal for row in rows},
        reverse=direction == "lower_is_healthier",
    )
    selected_subset: list[CalibrationWindow] = []
    selected_theta: float | None = None
    selected_attainment = 0.0

    for theta in candidates:
        if direction == "higher_is_healthier":
            subset = [row for row in rows if row.signal >= theta]
        else:
            subset = [row for row in rows if row.signal <= theta]
        support = len(subset)
        if support == 0:
            continue
        attainment = sum(1 for row in subset if row.slo_met) / support
        if support >= min_support and attainment >= reliability_target:
            selected_theta = theta
            selected_subset = subset
            selected_attainment = attainment
            break

    coverage_pass, family_counts = _coverage_stats(
        selected_subset,
        min_scenario_families=min_scenario_families,
        max_single_scenario_ratio=max_single_scenario_ratio,
    )
    confidence = selected_attainment
    reject_reason: str | None = None
    publish = False
    if not rows:
        reject_reason = "no_valid_windows"
    elif selected_theta is None:
        reject_reason = "insufficient_support_or_attainment"
    elif not coverage_pass:
        reject_reason = "insufficient_coverage"
    elif confidence < min_confidence:
        reject_reason = "insufficient_confidence"
    else:
        publish = True

    return ReliabilityThetaFit(
        publish=publish,
        theta=selected_theta,
        support=len(selected_subset),
        attainment=selected_attainment,
        confidence=confidence,
        coverage_pass=coverage_pass,
        family_counts=family_counts,
        reject_reason=reject_reason,
        candidate_count=len(candidates),
    )


def _coverage_stats(
    rows: list[CalibrationWindow],
    *,
    min_scenario_families: int,
    max_single_scenario_ratio: float,
) -> tuple[bool, dict[str, int]]:
    if not rows:
        return False, {}
    counter = Counter(row.scenario_family for row in rows)
    max_ratio = max(counter.values()) / len(rows)
    passed = len(counter) >= min_scenario_families and max_ratio <= max_single_scenario_ratio
    return passed, dict(sorted(counter.items()))


# ---------------------------------------------------------------------------
# Balanced-accuracy theta_m fit and the two-sided delta margin fit.
#
# Criterion names, never era names: `fit_theta_by_reliability` (above) selects the
# lowest score whose *cumulative upper set* attains `reliability_target`; it is a
# one-sided containment rule and it is kept because published calibration rounds
# used it and it is a useful comparison baseline. `fit_theta_by_balanced_accuracy`
# (below) selects the healthy-score quantile that maximises balanced accuracy of
# the rule "signal >= theta ==> SLO met". On the R3 load scans the containment rule
# puts theta far below the empirical healthy/violating boundary (local attainment at
# theta is 0.09-0.23), while the balanced-accuracy criterion lands on it.
# ---------------------------------------------------------------------------

#: Identifier written into calibration artifacts for the balanced-accuracy theta fit.
THETA_METHOD_BALANCED_ACCURACY = "healthy_quantile_balanced_accuracy"
#: Identifier written into calibration artifacts for the cumulative attainment fit.
THETA_METHOD_RELIABILITY = "cumulative_reliability_attainment"
#: Identifier written into calibration artifacts for the delta margin fit.
DELTA_METHOD = "severity_quantile_balanced_accuracy"

#: Accepted values of the explicit ``theta_criterion`` knob.
THETA_CRITERIA = ("balanced_accuracy", "reliability")
#: Default criterion for *new* fits.
DEFAULT_THETA_CRITERION = "balanced_accuracy"

#: Healthy-score quantiles searched by the balanced-accuracy fit.
DEFAULT_HEALTHY_QUANTILE_CANDIDATES: tuple[float, ...] = (
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
)

#: Default floor on ``recall_good`` for the balanced-accuracy fit.
#:
#: **Off by default, deliberately.** A floor of 0.90 vetoes the balanced-accuracy
#: optimum on every R3 model measured so far (the optimum's ``recall_good`` is
#: 0.79-0.86), dragging theta back to the 5th-10th healthy percentile and
#: reproducing exactly the bias the balanced-accuracy criterion exists to remove.
#: Raise it only with evidence that the resulting theta still separates.
DEFAULT_MIN_HEALTHY_RECALL = 0.0

#: Quantiles of the positive-label z distribution searched by the delta fit.
DEFAULT_DELTA_CANDIDATE_QUANTILES: tuple[float, ...] = tuple(
    round(0.05 + 0.05 * i, 10) for i in range(19)
)

#: Label-construction quantiles (severity of violations / comfort of healthy windows).
DEFAULT_CRITICAL_VIOLATION_QUANTILE = 0.65
DEFAULT_SURPLUS_LATENCY_QUANTILE = 0.35
DEFAULT_SURPLUS_QUEUE_QUANTILE = 0.50

#: Acceptance floors: delta_crit must recall the critical windows, delta_high must be
#: precise about surplus windows.
DEFAULT_MIN_CRITICAL_RECALL = 0.85
DEFAULT_MIN_SURPLUS_PRECISION = 0.80

#: Weights blending the p95 and average latency-ratio into one severity score.
DEFAULT_P95_WEIGHT = 0.8
DEFAULT_AVG_WEIGHT = 0.2

#: Margins used when a model's labels cannot support a fit. These are the same numbers
#: :class:`tre_controller.planning.classify.TauThresholds` falls back to, so a rejected
#: fit leaves behaviour unchanged rather than silently moving the bands.
FALLBACK_DELTA_CRIT = 0.2
FALLBACK_DELTA_HIGH = 0.25


@dataclass(frozen=True)
class BalancedAccuracyThetaFit:
    """Result of :func:`fit_theta_by_balanced_accuracy`."""

    publish: bool
    theta: float | None
    healthy_quantile: float | None
    balanced_accuracy: float
    recall_good: float
    specificity_bad: float
    precision_good: float
    healthy_window_count: int
    violating_window_count: int
    candidate_count: int
    min_healthy_recall: float
    family_counts: dict[str, int]
    coverage_pass: bool
    reject_reason: str | None


@dataclass(frozen=True)
class DeltaMarginFit:
    """One side (crit or high) of the two-sided margin fit."""

    delta: float
    tau: float
    balanced_accuracy: float | None
    recall_pos: float | None
    precision_pos: float | None
    specificity_neg: float | None
    support_pos: int
    candidate_quantile: float | None
    candidate_count: int
    meets_target_floor: bool
    used_fallback: bool
    reject_reason: str | None


@dataclass(frozen=True)
class DeltaLabelSummary:
    critical_cut: float | None
    surplus_latency_cut: float | None
    surplus_queue_cut: float | None
    critical_positive_count: int
    surplus_positive_count: int
    queue_available: bool


@dataclass(frozen=True)
class DeltaMarginsFit:
    """Per-model ``delta_crit`` / ``delta_high`` around ``tau_low``."""

    theta: float
    tau_low: float
    crit: DeltaMarginFit
    high: DeltaMarginFit
    labels: DeltaLabelSummary


def fit_theta_by_balanced_accuracy(
    windows: Iterable[CalibrationWindow],
    *,
    healthy_quantile_candidates: Sequence[float] = DEFAULT_HEALTHY_QUANTILE_CANDIDATES,
    min_healthy_recall: float = DEFAULT_MIN_HEALTHY_RECALL,
    min_scenario_families: int = 2,
    max_single_scenario_ratio: float = 0.7,
) -> BalancedAccuracyThetaFit:
    """Pick ``theta`` maximising balanced accuracy of ``signal >= theta ==> slo_met``.

    Candidates are quantiles of the *healthy* windows' signal distribution, which keeps
    the search on the scale the data actually occupies. Ties break on specificity first,
    then on the larger theta. ``min_healthy_recall`` is a hard filter applied before the
    balanced-accuracy comparison: candidates meeting it always beat candidates that do
    not. See :data:`DEFAULT_MIN_HEALTHY_RECALL` for why it defaults to off.
    """
    rows = [row for row in windows if math.isfinite(row.signal)]
    labels = [1 if row.slo_met else 0 for row in rows]
    scores = [row.signal for row in rows]
    healthy_scores = [score for score, label in zip(scores, labels) if label == 1]
    violating_count = sum(1 for label in labels if label == 0)

    best: dict[str, float] | None = None
    best_quantile: float | None = None
    candidate_count = 0
    for quantile in healthy_quantile_candidates:
        theta = _quantile(healthy_scores, quantile)
        if theta is None:
            continue
        metrics = _balanced_accuracy_at(scores, labels, theta)
        candidate_count += 1
        candidate = {"theta": theta, **metrics}
        if best is None:
            best, best_quantile = candidate, quantile
            continue
        best_ok = best["recall_good"] >= min_healthy_recall
        cand_ok = candidate["recall_good"] >= min_healthy_recall
        if cand_ok and not best_ok:
            best, best_quantile = candidate, quantile
            continue
        if best_ok and not cand_ok:
            continue
        if candidate["balanced_accuracy"] > best["balanced_accuracy"] + 1e-12:
            best, best_quantile = candidate, quantile
            continue
        if best["balanced_accuracy"] > candidate["balanced_accuracy"] + 1e-12:
            continue
        if candidate["specificity_bad"] > best["specificity_bad"] + 1e-12:
            best, best_quantile = candidate, quantile
            continue
        if best["specificity_bad"] > candidate["specificity_bad"] + 1e-12:
            continue
        if candidate["theta"] > best["theta"] + 1e-12:
            best, best_quantile = candidate, quantile

    if best is None:
        reject_reason = "no_valid_windows" if not rows else "no_healthy_windows"
        return BalancedAccuracyThetaFit(
            publish=False,
            theta=None,
            healthy_quantile=None,
            balanced_accuracy=0.0,
            recall_good=0.0,
            specificity_bad=0.0,
            precision_good=0.0,
            healthy_window_count=len(healthy_scores),
            violating_window_count=violating_count,
            candidate_count=candidate_count,
            min_healthy_recall=min_healthy_recall,
            family_counts={},
            coverage_pass=False,
            reject_reason=reject_reason,
        )

    theta = float(best["theta"])
    selected = [row for row in rows if row.signal >= theta]
    coverage_pass, family_counts = _coverage_stats(
        selected,
        min_scenario_families=min_scenario_families,
        max_single_scenario_ratio=max_single_scenario_ratio,
    )
    reject_reason: str | None = None
    if violating_count == 0:
        reject_reason = "no_violating_windows"
    elif not coverage_pass:
        reject_reason = "insufficient_coverage"

    return BalancedAccuracyThetaFit(
        publish=reject_reason is None,
        theta=theta,
        healthy_quantile=best_quantile,
        balanced_accuracy=float(best["balanced_accuracy"]),
        recall_good=float(best["recall_good"]),
        specificity_bad=float(best["specificity_bad"]),
        precision_good=float(best["precision_good"]),
        healthy_window_count=len(healthy_scores),
        violating_window_count=violating_count,
        candidate_count=candidate_count,
        min_healthy_recall=min_healthy_recall,
        family_counts=family_counts,
        coverage_pass=coverage_pass,
        reject_reason=reject_reason,
    )


def fit_delta_margins(
    windows: Iterable[CalibrationWindow],
    *,
    theta: float,
    tau_low: float = 1.0,
    p95_weight: float = DEFAULT_P95_WEIGHT,
    avg_weight: float = DEFAULT_AVG_WEIGHT,
    critical_violation_quantile: float = DEFAULT_CRITICAL_VIOLATION_QUANTILE,
    surplus_latency_quantile: float = DEFAULT_SURPLUS_LATENCY_QUANTILE,
    surplus_queue_quantile: float = DEFAULT_SURPLUS_QUEUE_QUANTILE,
    candidate_quantiles: Sequence[float] = DEFAULT_DELTA_CANDIDATE_QUANTILES,
    min_critical_recall: float = DEFAULT_MIN_CRITICAL_RECALL,
    min_surplus_precision: float = DEFAULT_MIN_SURPLUS_PRECISION,
    fallback_delta_crit: float = FALLBACK_DELTA_CRIT,
    fallback_delta_high: float = FALLBACK_DELTA_HIGH,
) -> DeltaMarginsFit:
    """Fit the per-model control margins ``tau_crit = tau_low - delta_crit`` and
    ``tau_high = tau_low + delta_high`` on ``z = signal / theta``.

    Two independent one-sided thresholds are fitted against labels derived from the
    same windows: *critical* = a violating window whose severity is at or above the
    ``critical_violation_quantile`` of all violations; *surplus* = a healthy window
    that is both comfortable on latency and short on queue. ``surplus`` labels require
    ``queue_raw``; without it the high side falls back to ``fallback_delta_high``.
    """
    if not math.isfinite(theta) or theta <= 0.0:
        raise ValueError("theta must be finite and positive")

    rows = [row for row in windows if math.isfinite(row.signal)]
    severity: list[float] = []
    for row in rows:
        p95_ratio = row.latency_ratio_p95
        if p95_ratio is None and row.health_score:
            p95_ratio = (1.0 / row.health_score) - 1.0
        if p95_ratio is None:
            raise ValueError(
                "delta fit needs latency_ratio_p95 (or health_score) on every window"
            )
        avg_ratio = row.latency_ratio_avg if row.latency_ratio_avg is not None else p95_ratio
        severity.append(p95_weight * p95_ratio + avg_weight * avg_ratio)

    queue_raw = [row.queue_raw for row in rows]
    queue_available = bool(rows) and all(
        q is not None and math.isfinite(q) for q in queue_raw
    )

    unhealthy_severity = [s for s, row in zip(severity, rows) if not row.slo_met]
    healthy_comfort = [s for s, row in zip(severity, rows) if row.slo_met]
    healthy_queue = (
        [float(q) for q, row in zip(queue_raw, rows) if row.slo_met and q is not None]
        if queue_available
        else []
    )

    critical_cut = _quantile(unhealthy_severity, critical_violation_quantile)
    surplus_latency_cut = _quantile(healthy_comfort, surplus_latency_quantile)
    surplus_queue_cut = _quantile(healthy_queue, surplus_queue_quantile) if queue_available else None

    critical_labels: list[int] = []
    surplus_labels: list[int] = []
    for row, score, q in zip(rows, severity, queue_raw):
        critical_labels.append(
            1 if (not row.slo_met and critical_cut is not None and score >= critical_cut) else 0
        )
        surplus_labels.append(
            1
            if (
                row.slo_met
                and surplus_latency_cut is not None
                and surplus_queue_cut is not None
                and q is not None
                and score <= surplus_latency_cut
                and float(q) <= surplus_queue_cut
            )
            else 0
        )

    z = [row.signal / theta for row in rows]
    crit = _fit_one_delta_margin(
        z,
        critical_labels,
        direction="low",
        tau_low=tau_low,
        candidate_quantiles=candidate_quantiles,
        target_floor=min_critical_recall,
        fallback_delta=fallback_delta_crit,
    )
    high = _fit_one_delta_margin(
        z,
        surplus_labels,
        direction="high",
        tau_low=tau_low,
        candidate_quantiles=candidate_quantiles,
        target_floor=min_surplus_precision,
        fallback_delta=fallback_delta_high,
    )

    return DeltaMarginsFit(
        theta=float(theta),
        tau_low=float(tau_low),
        crit=crit,
        high=high,
        labels=DeltaLabelSummary(
            critical_cut=critical_cut,
            surplus_latency_cut=surplus_latency_cut,
            surplus_queue_cut=surplus_queue_cut,
            critical_positive_count=sum(critical_labels),
            surplus_positive_count=sum(surplus_labels),
            queue_available=queue_available,
        ),
    )


def _fit_one_delta_margin(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    direction: str,
    tau_low: float,
    candidate_quantiles: Sequence[float],
    target_floor: float,
    fallback_delta: float,
) -> DeltaMarginFit:
    if direction not in {"low", "high"}:
        raise ValueError("direction must be low or high")

    finite = [(float(s), int(l)) for s, l in zip(scores, labels) if math.isfinite(s)]
    positive_scores = [s for s, l in finite if l == 1]
    if len(positive_scores) < 2 or len(positive_scores) == len(finite):
        return _delta_fallback(
            direction=direction,
            tau_low=tau_low,
            fallback_delta=fallback_delta,
            reject_reason="insufficient_label_separation",
            candidate_count=0,
        )

    best: dict[str, float] | None = None
    best_quantile: float | None = None
    candidate_count = 0
    for quantile in candidate_quantiles:
        tau = _quantile(positive_scores, quantile)
        if tau is None or not math.isfinite(tau):
            continue
        tau = min(tau, tau_low - 1e-6) if direction == "low" else max(tau, tau_low + 1e-6)
        metrics = _threshold_metrics_at(scores, labels, tau, direction=direction)
        meets = (
            metrics["recall_pos"] >= target_floor
            if direction == "low"
            else metrics["precision_pos"] >= target_floor
        )
        candidate = {"tau": tau, "meets_target_floor": 1.0 if meets else 0.0, **metrics}
        candidate_count += 1
        if best is None:
            best, best_quantile = candidate, quantile
            continue
        best_ok = bool(best["meets_target_floor"])
        cand_ok = bool(candidate["meets_target_floor"])
        if cand_ok and not best_ok:
            best, best_quantile = candidate, quantile
            continue
        if best_ok and not cand_ok:
            continue
        if candidate["balanced_accuracy"] > best["balanced_accuracy"] + 1e-12:
            best, best_quantile = candidate, quantile
            continue
        if best["balanced_accuracy"] > candidate["balanced_accuracy"] + 1e-12:
            continue
        secondary = "precision_pos" if direction == "low" else "recall_pos"
        if candidate[secondary] > best[secondary] + 1e-12:
            best, best_quantile = candidate, quantile
            continue
        if best[secondary] > candidate[secondary] + 1e-12:
            continue
        if candidate["tau"] > best["tau"] + 1e-12:
            best, best_quantile = candidate, quantile

    if best is None:
        return _delta_fallback(
            direction=direction,
            tau_low=tau_low,
            fallback_delta=fallback_delta,
            reject_reason="no_valid_candidates",
            candidate_count=candidate_count,
        )

    tau = float(best["tau"])
    return DeltaMarginFit(
        delta=abs(tau - tau_low),
        tau=tau,
        balanced_accuracy=float(best["balanced_accuracy"]),
        recall_pos=float(best["recall_pos"]),
        precision_pos=float(best["precision_pos"]),
        specificity_neg=float(best["specificity_neg"]),
        support_pos=int(best["support_pos"]),
        candidate_quantile=best_quantile,
        candidate_count=candidate_count,
        meets_target_floor=bool(best["meets_target_floor"]),
        used_fallback=False,
        reject_reason=None,
    )


def _delta_fallback(
    *,
    direction: str,
    tau_low: float,
    fallback_delta: float,
    reject_reason: str,
    candidate_count: int,
) -> DeltaMarginFit:
    tau = tau_low - fallback_delta if direction == "low" else tau_low + fallback_delta
    return DeltaMarginFit(
        delta=abs(tau - tau_low),
        tau=tau,
        balanced_accuracy=None,
        recall_pos=None,
        precision_pos=None,
        specificity_neg=None,
        support_pos=0,
        candidate_quantile=None,
        candidate_count=candidate_count,
        meets_target_floor=False,
        used_fallback=True,
        reject_reason=reject_reason,
    )


def _balanced_accuracy_at(
    scores: Sequence[float], labels: Sequence[int], theta: float
) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        pred_good = score >= theta
        if label == 1 and pred_good:
            tp += 1
        elif label == 1:
            fn += 1
        elif pred_good:
            fp += 1
        else:
            tn += 1
    recall_good = tp / (tp + fn) if (tp + fn) else 0.0
    specificity_bad = tn / (tn + fp) if (tn + fp) else 0.0
    precision_good = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "balanced_accuracy": 0.5 * (recall_good + specificity_bad),
        "recall_good": recall_good,
        "specificity_bad": specificity_bad,
        "precision_good": precision_good,
    }


def _threshold_metrics_at(
    scores: Sequence[float], labels: Sequence[int], threshold: float, *, direction: str
) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        if not math.isfinite(score):
            continue
        pred_pos = score <= threshold if direction == "low" else score >= threshold
        if label == 1 and pred_pos:
            tp += 1
        elif label == 1:
            fn += 1
        elif pred_pos:
            fp += 1
        else:
            tn += 1
    recall_pos = tp / (tp + fn) if (tp + fn) else 0.0
    specificity_neg = tn / (tn + fp) if (tn + fp) else 0.0
    precision_pos = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "balanced_accuracy": 0.5 * (recall_pos + specificity_neg),
        "recall_pos": recall_pos,
        "specificity_neg": specificity_neg,
        "precision_pos": precision_pos,
        "support_pos": float(tp + fn),
        "support_neg": float(tn + fp),
    }


def _quantile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation quantile over the finite entries of ``values``."""
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return None
    if q <= 0:
        return finite[0]
    if q >= 1:
        return finite[-1]
    pos = (len(finite) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return finite[lo]
    frac = pos - lo
    return finite[lo] * (1.0 - frac) + finite[hi] * frac

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from tre_calibration.dataset import CalibrationWindow

#: Signal orientations understood by every theta fit in this module.
#:
#: ``higher_is_healthier`` is the TSS/TRS convention: a large score means headroom.
#: ``lower_is_healthier`` is the convention of the pressure signals TSS is compared
#: against in the signal ablation -- queue length and the per-replica token rates --
#: where a large value means the replica is saturated. Every fit takes the orientation
#: as an explicit argument, so a signal comparison can never come out of two different
#: criteria merely because one of them could not express the direction.
SIGNAL_DIRECTIONS = ("higher_is_healthier", "lower_is_healthier")

#: Orientation assumed when a caller does not name one (the TSS/TRS convention).
DEFAULT_SIGNAL_DIRECTION = "higher_is_healthier"


def signal_orientation(direction: str) -> float:
    """``+1`` / ``-1`` multiplier that turns ``direction`` into "larger is healthier"."""
    if direction not in SIGNAL_DIRECTIONS:
        raise ValueError(f"direction must be one of {SIGNAL_DIRECTIONS}")
    return -1.0 if direction == "lower_is_healthier" else 1.0


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
    #: Which orientation the fit ran under. Recorded for the same reason as on
    #: :class:`BalancedAccuracyThetaFit`: a published threshold without its orientation
    #: is not a rule, and the ablation compares signals of both orientations.
    direction: str = DEFAULT_SIGNAL_DIRECTION


def fit_theta_by_reliability(
    windows: Iterable[CalibrationWindow],
    *,
    reliability_target: float,
    min_support: int,
    min_confidence: float,
    min_scenario_families: int,
    max_single_scenario_ratio: float,
    direction: str = DEFAULT_SIGNAL_DIRECTION,
) -> ReliabilityThetaFit:
    if direction not in SIGNAL_DIRECTIONS:
        raise ValueError(f"direction must be one of {SIGNAL_DIRECTIONS}")
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
        direction=direction,
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
#
# Both criteria accept both entries of `SIGNAL_DIRECTIONS`, and `fit_theta` dispatches
# between them, so the main signal and the alternative signals it is compared against in
# the ablation go through one criterion under one set of defaults. Fitting TSS by one
# criterion and queue_len by another would make the ablation measure the criteria
# rather than the signals.
# ---------------------------------------------------------------------------

#: Identifier written into calibration artifacts for the balanced-accuracy theta fit.
THETA_METHOD_BALANCED_ACCURACY = "healthy_quantile_balanced_accuracy"
#: Identifier written into calibration artifacts for the cumulative attainment fit.
THETA_METHOD_RELIABILITY = "cumulative_reliability_attainment"
#: Identifier written into calibration artifacts for the delta margin fit.
DELTA_METHOD = "crit:ba_grid_delta_0_0.5+high:severity_quantile_balanced_accuracy"

#: How ``delta_crit`` is searched. ``ba_grid`` (default, plan §6.3 B6): tau_crit =
#: tau_low - delta over :data:`DEFAULT_DELTA_CRIT_GRID`, maximising balanced accuracy of
#: "Z < tau_crit => critical" - the same predicate ``classify_model`` applies. ``quantile``
#: is the former rule (candidates = quantiles of the critical windows' z, clipped to
#: tau_low), kept only as a comparison baseline: on a sharp boundary every quantile lands
#: on the wrong side of tau_low and the fit clamps (7b's delta_crit = 0 in the dry run).
CRIT_METHOD_BA_GRID = "ba_grid"
CRIT_METHOD_QUANTILE = "quantile"
CRIT_METHODS = (CRIT_METHOD_BA_GRID, CRIT_METHOD_QUANTILE)
DEFAULT_CRIT_METHOD = CRIT_METHOD_BA_GRID
#: delta_crit candidates: 0.00 .. 0.50 in 0.01 steps.
DEFAULT_DELTA_CRIT_GRID: tuple[float, ...] = tuple(round(0.01 * i, 10) for i in range(51))

#: Accepted values of the explicit ``theta_criterion`` knob.
THETA_CRITERIA = ("balanced_accuracy", "reliability")
#: Criterion used by every fit that does not name one.
DEFAULT_THETA_CRITERION = "balanced_accuracy"

#: Acceptance-gate defaults shared by :func:`fit_theta` and :class:`ThetaFitConfig`.
#: They are named constants rather than literals in a signature so that a threshold, its
#: bootstrap confidence interval and its train/test acceptance cannot drift into three
#: slightly different gates.
DEFAULT_RELIABILITY_TARGET = 0.9
DEFAULT_MIN_SUPPORT = 3
DEFAULT_MIN_CONFIDENCE = 0.9
DEFAULT_MIN_SCENARIO_FAMILIES = 2
DEFAULT_MAX_SINGLE_SCENARIO_RATIO = 0.7

#: Healthy-score quantiles searched by the balanced-accuracy fit: 0.01 steps up to 0.05,
#: then 0.05 steps to 0.50. The lower edge used to be 0.05, and the 2026-09-21 dry run put
#: two of three models' theta on it (plan §6.3 B6) - a fit on the grid edge is a fit the
#: grid chose, not the data.
DEFAULT_HEALTHY_QUANTILE_CANDIDATES: tuple[float, ...] = (
    0.01, 0.02, 0.03, 0.04,
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

#: How the acceptance floor enters the candidate choice.
#:
#: ``strict`` is lexicographic: a candidate that misses the floor loses to any candidate
#: that meets it, whatever their balanced accuracies. That is how dsqwen-14b's LOW band
#: collapsed - its best-BA candidate recalled 116 of 137 critical windows against a floor
#: needing 117, so only candidates clipped to the ``tau_low`` bound survived and
#: ``tau_crit`` came out at ``tau_low - 1e-6``.
#:
#: ``soft`` (the default) selects on balanced accuracy and uses the floor only to break
#: ties, so one window either side of the floor can no longer flip the fit. The mode that
#: ran is recorded on the result and therefore in the artifact.
FLOOR_MODE_SOFT = "soft"
FLOOR_MODE_STRICT = "strict"
FLOOR_MODES = (FLOOR_MODE_SOFT, FLOOR_MODE_STRICT)
DEFAULT_DELTA_FLOOR_MODE = FLOOR_MODE_SOFT

#: A fitted tau this close to a bound is reported as ``clamped`` - the band it opens is
#: degenerate, and the number must never look like an ordinary fit.
CLAMP_TOLERANCE = 1e-5

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
    #: Which orientation the fit ran under. ``healthy_quantile`` is always measured from
    #: the *unhealthy* end of the healthy windows, so the same quantile means the same
    #: thing under either orientation.
    direction: str = DEFAULT_SIGNAL_DIRECTION


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
    #: Which role the acceptance floor played (see :data:`DEFAULT_DELTA_FLOOR_MODE`).
    floor_mode: str = DEFAULT_DELTA_FLOOR_MODE
    #: True when ``tau`` sits on (or within :data:`CLAMP_TOLERANCE` of) a bound rather
    #: than where the data put it. A clamped side opens a degenerate band.
    clamped: bool = False
    clamp_reason: str | None = None
    #: How the candidates were generated (:data:`CRIT_METHODS` for the crit side;
    #: ``quantile`` for the high side).
    method: str = "quantile"


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
    direction: str = DEFAULT_SIGNAL_DIRECTION,
) -> BalancedAccuracyThetaFit:
    """Pick ``theta`` maximising balanced accuracy of "healthy side of theta => slo_met".

    Under ``higher_is_healthier`` the rule scored is ``signal >= theta ==> slo_met``;
    under ``lower_is_healthier`` it is ``signal <= theta ==> slo_met``. The two are the
    same problem on a reflected axis, so the search runs on ``orientation * signal`` and
    the winning threshold is reflected back into raw signal units before it is returned.
    Criterion, candidate set, tie-breaks and acceptance gates are therefore identical for
    both orientations by construction rather than by a parallel code path.

    Candidates are quantiles of the *healthy* windows' signal distribution measured from
    the unhealthy end, which keeps the search on the scale the data actually occupies.
    Ties break on specificity first, then on the threshold that admits fewer windows as
    healthy. ``min_healthy_recall`` is a hard filter applied before the balanced-accuracy
    comparison: candidates meeting it always beat candidates that do not. See
    :data:`DEFAULT_MIN_HEALTHY_RECALL` for why it defaults to off.
    """
    orientation = signal_orientation(direction)
    rows = [row for row in windows if math.isfinite(row.signal)]
    labels = [1 if row.slo_met else 0 for row in rows]
    # Oriented scores: "larger is healthier" holds for both orientations from here on.
    scores = [orientation * row.signal for row in rows]
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
            direction=direction,
        )

    oriented_theta = float(best["theta"])
    theta = orientation * oriented_theta
    selected = [row for row in rows if orientation * row.signal >= oriented_theta]
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
        direction=direction,
    )


def threshold_balanced_accuracy(
    windows: Iterable[CalibrationWindow],
    *,
    theta: float,
    direction: str = DEFAULT_SIGNAL_DIRECTION,
) -> dict[str, float]:
    """Confusion metrics of "healthy side of ``theta`` ==> SLO met" on ``windows``.

    Exactly the scoring :func:`fit_theta_by_balanced_accuracy` optimises, exposed so that
    a diagnostic curve or a cross-criterion comparison cannot drift from the criterion
    that picked the threshold. ``direction`` selects which side of ``theta`` counts as
    the healthy prediction.
    """
    orientation = signal_orientation(direction)
    rows = [row for row in windows if math.isfinite(row.signal)]
    scores = [orientation * row.signal for row in rows]
    labels = [1 if row.slo_met else 0 for row in rows]
    return _balanced_accuracy_at(scores, labels, orientation * theta)


def fit_theta(
    windows: Iterable[CalibrationWindow],
    *,
    criterion: str = DEFAULT_THETA_CRITERION,
    direction: str = DEFAULT_SIGNAL_DIRECTION,
    healthy_quantile_candidates: Sequence[float] = DEFAULT_HEALTHY_QUANTILE_CANDIDATES,
    min_healthy_recall: float = DEFAULT_MIN_HEALTHY_RECALL,
    reliability_target: float = DEFAULT_RELIABILITY_TARGET,
    min_support: int = DEFAULT_MIN_SUPPORT,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_scenario_families: int = DEFAULT_MIN_SCENARIO_FAMILIES,
    max_single_scenario_ratio: float = DEFAULT_MAX_SINGLE_SCENARIO_RATIO,
) -> ReliabilityThetaFit | BalancedAccuracyThetaFit:
    """Fit ``theta`` under the named criterion -- the entry point every fit goes through.

    Having a single dispatcher is the point. The signal ablation compares TSS against
    queue length and the per-replica token rates, and that comparison is only about the
    signals if every arm went through the same criterion with the same defaults. Callers
    name the criterion and the orientation; they do not pick a fit function.
    """
    if criterion not in THETA_CRITERIA:
        raise ValueError(f"criterion must be one of {THETA_CRITERIA}")
    if criterion == "balanced_accuracy":
        return fit_theta_by_balanced_accuracy(
            windows,
            healthy_quantile_candidates=healthy_quantile_candidates,
            min_healthy_recall=min_healthy_recall,
            min_scenario_families=min_scenario_families,
            max_single_scenario_ratio=max_single_scenario_ratio,
            direction=direction,
        )
    return fit_theta_by_reliability(
        windows,
        reliability_target=reliability_target,
        min_support=min_support,
        min_confidence=min_confidence,
        min_scenario_families=min_scenario_families,
        max_single_scenario_ratio=max_single_scenario_ratio,
        direction=direction,
    )


@dataclass(frozen=True)
class ThetaFitConfig:
    """Every knob :func:`fit_theta` takes, carried as one value.

    A threshold only means something together with the criterion, orientation and
    acceptance gates that produced it. Tools that fit ``theta`` more than once -- the
    bootstrap CI (a point estimate plus one fit per resample) and the train/test ranking
    separation report -- must use one configuration for every one of those fits, or the
    interval they publish describes a different quantity than the threshold it is meant to
    bracket. Passing this object instead of a bag of keyword arguments makes that sameness
    structural: there is one configuration to thread through, and its defaults are
    :func:`fit_theta`'s own defaults, which are the calibration CLI's defaults.

    :meth:`as_dict` renders the configuration with the key names the calibration CLI writes
    into an artifact's ``fit_config``, so a report and the artifact whose theta it describes
    can be compared field by field.
    """

    criterion: str = DEFAULT_THETA_CRITERION
    direction: str = DEFAULT_SIGNAL_DIRECTION
    healthy_quantile_candidates: tuple[float, ...] = DEFAULT_HEALTHY_QUANTILE_CANDIDATES
    min_healthy_recall: float = DEFAULT_MIN_HEALTHY_RECALL
    reliability_target: float = DEFAULT_RELIABILITY_TARGET
    min_support: int = DEFAULT_MIN_SUPPORT
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    min_scenario_families: int = DEFAULT_MIN_SCENARIO_FAMILIES
    max_single_scenario_ratio: float = DEFAULT_MAX_SINGLE_SCENARIO_RATIO

    def __post_init__(self) -> None:
        if self.criterion not in THETA_CRITERIA:
            raise ValueError(f"criterion must be one of {THETA_CRITERIA}")
        if self.direction not in SIGNAL_DIRECTIONS:
            raise ValueError(f"direction must be one of {SIGNAL_DIRECTIONS}")
        object.__setattr__(
            self, "healthy_quantile_candidates", tuple(self.healthy_quantile_candidates)
        )

    def fit(
        self, windows: Iterable[CalibrationWindow]
    ) -> ReliabilityThetaFit | BalancedAccuracyThetaFit:
        """Run :func:`fit_theta` on ``windows`` under exactly this configuration."""
        return fit_theta(
            windows,
            criterion=self.criterion,
            direction=self.direction,
            healthy_quantile_candidates=self.healthy_quantile_candidates,
            min_healthy_recall=self.min_healthy_recall,
            reliability_target=self.reliability_target,
            min_support=self.min_support,
            min_confidence=self.min_confidence,
            min_scenario_families=self.min_scenario_families,
            max_single_scenario_ratio=self.max_single_scenario_ratio,
        )

    def as_dict(self) -> dict[str, object]:
        """JSON-ready record of the configuration, keyed as the CLI keys ``fit_config``."""
        return {
            "direction": self.direction,
            "healthy_quantile_candidates": list(self.healthy_quantile_candidates),
            "max_single_scenario_ratio": self.max_single_scenario_ratio,
            "min_confidence": self.min_confidence,
            "min_healthy_recall": self.min_healthy_recall,
            "min_scenario_families": self.min_scenario_families,
            "min_support": self.min_support,
            "reliability_target": self.reliability_target,
            "theta_criterion": self.criterion,
        }


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
    floor_mode: str = DEFAULT_DELTA_FLOOR_MODE,
    fallback_delta_crit: float = FALLBACK_DELTA_CRIT,
    fallback_delta_high: float = FALLBACK_DELTA_HIGH,
    crit_method: str = DEFAULT_CRIT_METHOD,
    delta_crit_grid: Sequence[float] = DEFAULT_DELTA_CRIT_GRID,
) -> DeltaMarginsFit:
    """Fit the per-model control margins ``tau_crit = tau_low - delta_crit`` and
    ``tau_high = tau_low + delta_high`` on ``z = signal / theta``.

    Two independent one-sided thresholds are fitted against labels derived from the
    same windows: *critical* = a violating window whose severity is at or above the
    ``critical_violation_quantile`` of all violations; *surplus* = a healthy window
    that is both comfortable on latency and short on queue. ``surplus`` labels require
    ``queue_raw``; without it the high side falls back to ``fallback_delta_high``.

    ``min_critical_recall`` / ``min_surplus_precision`` are acceptance floors whose role
    is set by ``floor_mode``: ``soft`` (default) ranks candidates by balanced accuracy and
    only breaks ties with the floor, ``strict`` keeps the older lexicographic filter. Each
    side reports the mode it ran under and whether its ``tau`` ended up clamped to a bound.

    ``crit_method`` selects how the critical side is searched (:data:`CRIT_METHODS`); the
    default is the balanced-accuracy grid over ``delta_crit_grid``.
    """
    if not math.isfinite(theta) or theta <= 0.0:
        raise ValueError("theta must be finite and positive")
    if crit_method not in CRIT_METHODS:
        raise ValueError(f"crit_method must be one of {CRIT_METHODS}, got {crit_method!r}")

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
    if crit_method == CRIT_METHOD_BA_GRID:
        crit = _fit_delta_crit_grid(
            z,
            critical_labels,
            tau_low=tau_low,
            delta_grid=delta_crit_grid,
            target_floor=min_critical_recall,
            floor_mode=floor_mode,
            fallback_delta=fallback_delta_crit,
        )
    else:
        crit = _fit_one_delta_margin(
            z,
            critical_labels,
            direction="low",
            tau_low=tau_low,
            candidate_quantiles=candidate_quantiles,
            target_floor=min_critical_recall,
            floor_mode=floor_mode,
            fallback_delta=fallback_delta_crit,
        )
    high = _fit_one_delta_margin(
        z,
        surplus_labels,
        direction="high",
        tau_low=tau_low,
        candidate_quantiles=candidate_quantiles,
        target_floor=min_surplus_precision,
        floor_mode=floor_mode,
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
    floor_mode: str = DEFAULT_DELTA_FLOOR_MODE,
    fallback_delta: float,
) -> DeltaMarginFit:
    if direction not in {"low", "high"}:
        raise ValueError("direction must be low or high")
    if floor_mode not in FLOOR_MODES:
        raise ValueError(f"floor_mode must be one of {FLOOR_MODES}, got {floor_mode!r}")

    finite = [(float(s), int(l)) for s, l in zip(scores, labels) if math.isfinite(s)]
    positive_scores = [s for s, l in finite if l == 1]
    if len(positive_scores) < 2 or len(positive_scores) == len(finite):
        return _delta_fallback(
            direction=direction,
            tau_low=tau_low,
            fallback_delta=fallback_delta,
            reject_reason="insufficient_label_separation",
            candidate_count=0,
            floor_mode=floor_mode,
        )

    best: dict[str, float] | None = None
    best_quantile: float | None = None
    candidate_count = 0
    for quantile in candidate_quantiles:
        raw_tau = _quantile(positive_scores, quantile)
        if raw_tau is None or not math.isfinite(raw_tau):
            continue
        bound = tau_low - 1e-6 if direction == "low" else tau_low + 1e-6
        tau = min(raw_tau, bound) if direction == "low" else max(raw_tau, bound)
        metrics = _threshold_metrics_at(scores, labels, tau, direction=direction)
        meets = (
            metrics["recall_pos"] >= target_floor
            if direction == "low"
            else metrics["precision_pos"] >= target_floor
        )
        candidate = {
            "tau": tau,
            "meets_target_floor": 1.0 if meets else 0.0,
            "clipped_to_bound": 1.0 if tau != raw_tau else 0.0,
            **metrics,
        }
        candidate_count += 1
        if best is None:
            best, best_quantile = candidate, quantile
            continue
        best_ok = bool(best["meets_target_floor"])
        cand_ok = bool(candidate["meets_target_floor"])
        if floor_mode == FLOOR_MODE_STRICT:
            # Lexicographic: the floor outranks the objective. Kept for reproducing
            # older fits; see DEFAULT_DELTA_FLOOR_MODE for why it is not the default.
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
        # Equal balanced accuracy: now the acceptance floor speaks (soft mode's only say).
        if cand_ok and not best_ok:
            best, best_quantile = candidate, quantile
            continue
        if best_ok and not cand_ok:
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
            floor_mode=floor_mode,
        )

    tau = float(best["tau"])
    clamped, clamp_reason = _clamp_state(
        tau, tau_low, clipped=bool(best["clipped_to_bound"])
    )
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
        floor_mode=floor_mode,
        clamped=clamped,
        clamp_reason=clamp_reason,
    )


def _fit_delta_crit_grid(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    tau_low: float,
    delta_grid: Sequence[float],
    target_floor: float,
    floor_mode: str = DEFAULT_DELTA_FLOOR_MODE,
    fallback_delta: float,
) -> DeltaMarginFit:
    """delta_crit by balanced accuracy over an explicit grid (plan §6.3 B6).

    ``tau = tau_low - delta`` and the prediction is ``z < tau`` - strictly, exactly as
    ``classify_model`` decides CRITICAL. The candidates no longer come from the critical
    windows' own z quantiles, so a sharp boundary (every critical window just under
    tau_low) is a legitimate small delta instead of a forced clamp. Ties break on the
    acceptance floor, then on recall of the critical windows, then on the smaller delta
    (the more sensitive band). A delta on either grid edge is reported ``clamped``.
    """
    if floor_mode not in FLOOR_MODES:
        raise ValueError(f"floor_mode must be one of {FLOOR_MODES}, got {floor_mode!r}")
    grid = sorted({float(d) for d in delta_grid if math.isfinite(float(d)) and float(d) >= 0.0})
    finite = [(float(s), int(l)) for s, l in zip(scores, labels) if math.isfinite(s)]
    n_pos = sum(l for _s, l in finite)
    n_neg = len(finite) - n_pos
    if not grid or n_pos < 2 or n_neg == 0:
        fallback = _delta_fallback(
            direction="low",
            tau_low=tau_low,
            fallback_delta=fallback_delta,
            reject_reason="insufficient_label_separation" if grid else "empty_delta_grid",
            candidate_count=0,
            floor_mode=floor_mode,
        )
        return DeltaMarginFit(**{**fallback.__dict__, "method": CRIT_METHOD_BA_GRID})

    # Sorted z with cumulative positive counts: each candidate is one bisect.
    import bisect

    ordered = sorted(finite)
    zs = [s for s, _l in ordered]
    cum_pos = [0]
    for _s, label in ordered:
        cum_pos.append(cum_pos[-1] + label)

    best: dict[str, float] | None = None
    for delta in grid:
        tau = tau_low - delta
        k = bisect.bisect_left(zs, tau)  # count of z < tau
        tp = cum_pos[k]
        fp = k - tp
        recall = tp / n_pos
        specificity = (n_neg - fp) / n_neg
        precision = tp / k if k else 0.0
        candidate = {
            "delta": delta,
            "tau": tau,
            "balanced_accuracy": 0.5 * (recall + specificity),
            "recall_pos": recall,
            "precision_pos": precision,
            "specificity_neg": specificity,
            "support_pos": float(n_pos),
            "meets": 1.0 if recall >= target_floor else 0.0,
        }
        if best is None:
            best = candidate
            continue
        if floor_mode == FLOOR_MODE_STRICT and candidate["meets"] != best["meets"]:
            if candidate["meets"] > best["meets"]:
                best = candidate
            continue
        key_c = (candidate["balanced_accuracy"], candidate["meets"], candidate["recall_pos"], -candidate["delta"])
        key_b = (best["balanced_accuracy"], best["meets"], best["recall_pos"], -best["delta"])
        if _key_greater(key_c, key_b):
            best = candidate

    assert best is not None
    delta = float(best["delta"])
    clamped = delta <= grid[0] + 1e-12 or delta >= grid[-1] - 1e-12
    clamp_reason = None
    if clamped:
        clamp_reason = "delta_at_grid_lower_edge" if delta <= grid[0] + 1e-12 else "delta_at_grid_upper_edge"
    return DeltaMarginFit(
        delta=delta,
        tau=float(best["tau"]),
        balanced_accuracy=float(best["balanced_accuracy"]),
        recall_pos=float(best["recall_pos"]),
        precision_pos=float(best["precision_pos"]),
        specificity_neg=float(best["specificity_neg"]),
        support_pos=int(best["support_pos"]),
        candidate_quantile=None,
        candidate_count=len(grid),
        meets_target_floor=bool(best["meets"]),
        used_fallback=False,
        reject_reason=None,
        floor_mode=floor_mode,
        clamped=clamped,
        clamp_reason=clamp_reason,
        method=CRIT_METHOD_BA_GRID,
    )


def _key_greater(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """Lexicographic ``a > b`` with a 1e-12 tolerance per component."""
    for x, y in zip(a, b):
        if x > y + 1e-12:
            return True
        if y > x + 1e-12:
            return False
    return False


def _clamp_state(tau: float, tau_low: float, *, clipped: bool) -> tuple[bool, str | None]:
    """Was ``tau`` put where the data wanted it, or on a bound?

    ``clipped`` means the candidate quantile fell on the wrong side of ``tau_low`` and was
    pushed back to the bound. Either way, a ``tau`` within :data:`CLAMP_TOLERANCE` of
    ``tau_low`` opens an empty band, so it is reported rather than returned silently.
    """
    if abs(tau - tau_low) <= CLAMP_TOLERANCE:
        return True, (
            "quantile_on_wrong_side_of_tau_low_clipped_to_bound"
            if clipped
            else "tau_within_clamp_tolerance_of_tau_low"
        )
    if clipped:
        return True, "quantile_on_wrong_side_of_tau_low_clipped_to_bound"
    return False, None


def _delta_fallback(
    *,
    direction: str,
    tau_low: float,
    fallback_delta: float,
    reject_reason: str,
    candidate_count: int,
    floor_mode: str = DEFAULT_DELTA_FLOOR_MODE,
) -> DeltaMarginFit:
    tau = tau_low - fallback_delta if direction == "low" else tau_low + fallback_delta
    clamped, clamp_reason = _clamp_state(tau, tau_low, clipped=False)
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
        floor_mode=floor_mode,
        clamped=clamped,
        clamp_reason=clamp_reason,
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

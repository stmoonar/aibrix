from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

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
) -> ReliabilityThetaFit:
    rows = [row for row in windows if math.isfinite(row.signal)]
    candidates = sorted({row.signal for row in rows})
    selected_subset: list[CalibrationWindow] = []
    selected_theta: float | None = None
    selected_attainment = 0.0

    for theta in candidates:
        subset = [row for row in rows if row.signal >= theta]
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

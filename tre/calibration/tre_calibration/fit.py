from __future__ import annotations

import math
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

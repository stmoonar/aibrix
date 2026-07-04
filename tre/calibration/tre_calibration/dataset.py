from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CalibrationWindow:
    scenario_id: str
    scenario_family: str
    signal: float
    slo_met: bool
    health_score: float | None = None


def split_by_scenario(
    windows: Iterable[CalibrationWindow],
    *,
    test_scenarios: set[str],
) -> tuple[list[CalibrationWindow], list[CalibrationWindow]]:
    train: list[CalibrationWindow] = []
    test: list[CalibrationWindow] = []
    for window in windows:
        if window.scenario_id in test_scenarios:
            test.append(window)
        else:
            train.append(window)
    return train, test

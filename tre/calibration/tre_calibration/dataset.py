from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


_LATENCY_COLUMNS = {
    "ttft_p95": "p95_ttft",
    "tpot_p95": "p95_tpot",
    "e2e_p95": "p95_e2e",
}


@dataclass(frozen=True)
class CalibrationWindow:
    scenario_id: str
    scenario_family: str
    signal: float
    slo_met: bool
    health_score: float | None = None


def load_windows_from_csv(
    path: str | Path,
    *,
    latency_slo_ms: Mapping[str, float],
    signal_column: str = "trs",
) -> list[CalibrationWindow]:
    active_columns = _resolve_latency_columns(latency_slo_ms)
    if not active_columns:
        raise ValueError("latency_slo_ms must contain at least one active SLO")

    windows: list[CalibrationWindow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if _skip_row(row):
                continue

            signal = _as_float(row.get(signal_column))
            if signal is None:
                continue
            prompt_tokens = _as_float(row.get("prompt_tokens_total"), 0.0) or 0.0
            generation_tokens = _as_float(row.get("generation_tokens_total"), 0.0) or 0.0
            if prompt_tokens + generation_tokens <= 0.0:
                continue

            ratios: list[float] = []
            missing_latency = False
            for slo_key, column in active_columns.items():
                value = _as_float(row.get(column))
                if value is None:
                    missing_latency = True
                    break
                ratios.append(value / float(latency_slo_ms[slo_key]))
            if missing_latency or not ratios:
                continue

            p95_ratio_max = max(ratios)
            windows.append(
                CalibrationWindow(
                    scenario_id=(row.get("scenario_id") or "unknown").strip() or "unknown",
                    scenario_family=(row.get("scenario_family") or "unknown").strip() or "unknown",
                    signal=signal,
                    slo_met=all(ratio <= 1.0 for ratio in ratios),
                    health_score=1.0 / (1.0 + p95_ratio_max),
                )
            )
    return windows


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


def _resolve_latency_columns(latency_slo_ms: Mapping[str, float]) -> dict[str, str]:
    out: dict[str, str] = {}
    for slo_key in latency_slo_ms:
        out[slo_key] = _LATENCY_COLUMNS.get(slo_key, slo_key)
    return out


def _skip_row(row: Mapping[str, Any]) -> bool:
    if row.get("metric_scope") and str(row.get("metric_scope")).strip() != "model":
        return True
    if _as_bool(row.get("is_warmup")) or _as_bool(row.get("is_contaminated")):
        return True
    return bool(str(row.get("filter_reason") or "").strip())


def _as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out

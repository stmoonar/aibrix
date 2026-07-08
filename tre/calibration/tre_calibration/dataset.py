from __future__ import annotations

import csv
import hashlib
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


def select_test_scenarios(
    windows: Iterable[CalibrationWindow],
    *,
    test_fraction: float = 0.2,
    seed: str = "tre-v2-ranking",
) -> set[str]:
    """Deterministically pick whole scenarios for the held-out test set.

    ``docs/refactor/06_calibration_design.md`` ("Split Target") assigns
    scenario-level train/test splitting to this module "so scenario IDs never
    leak across sets". The plan does not pin a ratio or an RNG, so this uses a
    reproducible scenario-id hash split (default 80/20 train/test):

      * the split unit is the whole scenario (grid cell) -- every window of a
        scenario lands in the same set, so no cell can leak across train/test;
      * scenarios are ranked by ``sha256(seed:scenario_id)`` using ``hashlib``
        (NOT the builtin ``hash``, which is salted per-process via
        ``PYTHONHASHSEED``), so the split is byte-identical across processes and
        machines for the same CSV, ``seed`` and ``test_fraction``;
      * the ``round(n * test_fraction)`` lowest-hash scenarios become test,
        clamped to ``1 <= n_test <= n - 1`` whenever there are >= 2 scenarios so
        that neither train nor test is empty.

    A single-scenario CSV cannot be split without leaking, so this returns an
    empty test set; the caller is expected to report that (and skip test-set
    evaluation) rather than fit theta on an empty train set.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in the open interval (0, 1)")
    scenario_ids = sorted({window.scenario_id for window in windows})
    n = len(scenario_ids)
    if n < 2:
        return set()
    n_test = round(n * test_fraction)
    n_test = max(1, min(n_test, n - 1))
    ranked = sorted(scenario_ids, key=lambda sid: (_scenario_hash(sid, seed), sid))
    return set(ranked[:n_test])


def _scenario_hash(scenario_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{scenario_id}".encode("utf-8")).hexdigest()


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

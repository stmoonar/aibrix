from __future__ import annotations

import csv
import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from tre_calibration.labels import LabelDefinition, LabelSpec, label_window
from tre_common.tss import (
    DEFAULT_EMA_TAU_MS,
    TssEma,
    replica_factor,
    signal_ema,
    tss_queue,
    tss_terms,
    window_is_idle,
)


LOG = logging.getLogger(__name__)

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
    window_start_ms: float | None = None
    # Extra columns the delta-margin fit needs (see tre_calibration.fit.fit_delta_margins).
    # ``latency_ratio_p95`` is max(p95_metric / its SLO); ``latency_ratio_avg`` is the
    # same over average latencies and is None whenever the sweep did not record them
    # (the fit then falls back to the p95 ratio). ``queue_raw`` is the unfloored control
    # TSS queue running + lambda_wait*waiting (tre_common.tss.tss_queue - no swapping
    # term), populated only when the loader is given ``lambda_wait``.
    latency_ratio_p95: float | None = None
    latency_ratio_avg: float | None = None
    queue_raw: float | None = None
    #: None for a healthy window, else unserved / both / ttft_only / tpot_only
    #: (tre_calibration.labels.WindowLabel.violation_class) - for per-class recall.
    violation_class: str | None = None


@dataclass(frozen=True)
class TssRecompute:
    """Recompute the TSS signal from a window CSV's raw columns instead of reading the
    ``trs`` column, under the given parameters and the shared definition
    (:mod:`tre_common.tss`: rate numerator, qmin guard, idle rule, tau-EMA).

    With ``ema_tau_ms`` set, each cell's windows are EMA'd in CSV order with the same
    wall-clock time constant the controller uses, over *every* row of the cell (a row
    later dropped by a filter still advanced the online EMA). ``ema_tau_ms=None`` scores
    the raw signal.
    """

    w_p: float
    lambda_wait: float
    qmin: float = 1.0
    ema_tau_ms: float | None = DEFAULT_EMA_TAU_MS

    def as_dict(self) -> dict[str, float | None]:
        return {
            "w_p": self.w_p,
            "lambda_wait": self.lambda_wait,
            "qmin": self.qmin,
            "ema_tau_ms": self.ema_tau_ms,
        }


def smooth_rows_by_cell(
    rows: Sequence[Mapping[str, Any]],
    raws: Sequence[float | None],
    *,
    make_ema: Callable[[], TssEma],
) -> list[float | None]:
    """EMA ``raws`` cell by cell in CSV order - the one offline smoothing path.

    Cells are contiguous blocks of rows (a new block starts when ``scenario_id`` changes
    or ``window_end_ms`` goes backwards), which is how ``rewindow_from_raw`` writes them.
    Every row of a cell advances the EMA, including rows a later filter drops, because
    the online EMA saw them too. ``make_ema`` builds a fresh EMA per cell: ``TssEma`` for
    TSS, :func:`tre_common.tss.signal_ema` for the alternative signals - the same class,
    tau and alpha the controller uses.

    Within a cell, idle periods are handled by the same ``TssEma`` rules the controller
    applies: the idle-window reset with :func:`tre_common.tss.window_is_idle` of the row's
    ``prompt_tokens_total`` / ``generation_tokens_total`` (missing columns -> not idle), and
    the idle-gap rule with the row's own window (``window_end_ms - window_start_ms``;
    without a ``window_start_ms`` column the gap rule is off). The per-cell reset assumes the online EMA was reset between cells too,
    which needs more than one window of quiet between them; a cell that starts less than
    one window after the previous one ended is logged as a warning (the online EMA may
    have carried over, so the offline value can differ from what the controller saw).
    """
    values: list[float | None] = []
    ema: TssEma | None = None
    prev_cell: str | None = None
    prev_end: float | None = None
    close_cells: list[str] = []
    for row, raw in zip(rows, raws):
        end = _as_float(row.get("window_end_ms"))
        if end is None:
            raise ValueError("EMA smoothing needs a window_end_ms column")
        start = _as_float(row.get("window_start_ms"))
        window_ms = end - start if start is not None and end > start else None
        cell = str(row.get("scenario_id") or "")
        if ema is None or cell != prev_cell or (prev_end is not None and end < prev_end):
            if (
                ema is not None
                and prev_end is not None
                and end >= prev_end
                and start is not None
                and window_ms is not None
                and start - prev_end < window_ms
            ):
                close_cells.append(f"{prev_cell}->{cell} ({start - prev_end:.0f} ms)")
            ema = make_ema()
        prev_cell, prev_end = cell, end
        idle = window_is_idle(
            _as_float(row.get("prompt_tokens_total")), _as_float(row.get("generation_tokens_total"))
        )
        values.append(ema.update(raw, end, window_ms, idle))
    if close_cells:
        LOG.warning(
            "%d cell(s) start less than one metrics window after the previous cell ended; "
            "the offline per-cell EMA reset may not match the online EMA there: %s",
            len(close_cells),
            ", ".join(close_cells[:5]) + (" ..." if len(close_cells) > 5 else ""),
        )
    return values


def recompute_tss_rows(
    rows: Sequence[Mapping[str, Any]], params: TssRecompute
) -> list[float | None]:
    """One TSS value per CSV row (``None`` = undefined or not computable).

    Raw TSS per row, then :func:`smooth_rows_by_cell` when ``params.ema_tau_ms`` is set.
    """
    raws: list[float | None] = []
    for row in rows:
        start = _as_float(row.get("window_start_ms"))
        end = _as_float(row.get("window_end_ms"))
        if start is None or end is None:
            raise ValueError("TSS recompute needs window_start_ms and window_end_ms columns")
        prompt = _as_float(row.get("prompt_tokens_total"))
        generation = _as_float(row.get("generation_tokens_total"))
        if prompt is None or generation is None:
            raws.append(None)  # tokens missing: the controller computes nothing either
            continue
        terms = tss_terms(
            prompt_tokens=prompt,
            generation_tokens=generation,
            avg_running=_as_float(row.get("avg_running"), 0.0) or 0.0,
            avg_waiting=_as_float(row.get("avg_waiting"), 0.0) or 0.0,
            w_p=params.w_p,
            lambda_wait=params.lambda_wait,
            qmin=params.qmin,
            kv_cache_hit_rate=_as_float(row.get("kv_cache_hit_rate"), 0.0) or 0.0,
            avg_swapping=_as_float(row.get("avg_swapping"), 0.0) or 0.0,
            factor=replica_factor(
                _as_float(row.get("assigned_replicas"), 1.0) or 1.0,
                _as_float(row.get("routable_pods"), 1.0) or 1.0,
            ),
        )
        raws.append(terms.raw)
    if params.ema_tau_ms is None:
        return raws
    tau = params.ema_tau_ms
    return smooth_rows_by_cell(rows, raws, make_ema=lambda: TssEma(tau))


def load_windows_from_csv(
    path: str | Path,
    *,
    latency_slo_ms: LabelSpec,
    signal_column: str = "trs",
    signal_transform: Callable[[Mapping[str, Any]], float | None] | None = None,
    trim_ramp_windows: int = 0,
    lambda_wait: float | None = None,
    tss: TssRecompute | None = None,
    ema_tau_ms: float | None = None,
) -> list[CalibrationWindow]:
    """Load per-window calibration rows from a load-scan CSV.

    ``trim_ramp_windows`` drops that many earliest windows per scenario; it shifts the
    fitted theta by a few percent, so callers record the value they used in the
    calibration artifact rather than relying on a default.

    ``lambda_wait`` is optional and only used to reconstruct ``queue_raw`` from the
    ``avg_waiting`` / ``avg_running`` / ``avg_swapping`` columns. Pass the model's
    registry value whenever the caller intends to run
    :func:`tre_calibration.fit.fit_delta_margins`, whose surplus labels need the
    queue depth; leave it None and the high-side margin falls back to its default.

    ``tss`` recomputes the signal from the raw columns (see :class:`TssRecompute`) and
    takes precedence over ``signal_column`` / ``signal_transform``.

    ``ema_tau_ms`` smooths a column / transform signal with the alternative-signal EMA
    (:func:`tre_common.tss.signal_ema`) over every row of each cell, as the controller
    does online (plan §6.9 item 4). It is ignored with ``tss``, which carries its own tau.
    """
    # The production callers pass the LabelDefinition itself (fixed or slowdown TTFT SLO,
    # min-n guard); a plain mapping keeps the historical fixed label for ad-hoc CSVs.
    label_spec: LabelSpec = latency_slo_ms
    slowdown = isinstance(label_spec, LabelDefinition) and label_spec.slowdown
    if isinstance(label_spec, LabelDefinition):
        latency_slo_ms = label_spec.latency_slo_ms()
    active_columns = _resolve_latency_columns(latency_slo_ms)
    if not active_columns:
        raise ValueError("latency_slo_ms must contain at least one active SLO")

    windows: list[CalibrationWindow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        all_rows = list(csv.DictReader(f))
    if tss is not None:
        recomputed: list[float | None] | None = recompute_tss_rows(all_rows, tss)
    elif ema_tau_ms is not None:
        raws = [
            _as_float(signal_transform(row) if signal_transform is not None else row.get(signal_column))
            for row in all_rows
        ]
        tau = float(ema_tau_ms)
        recomputed = smooth_rows_by_cell(all_rows, raws, make_ema=lambda: signal_ema(tau))
    else:
        recomputed = None
    for index, row in enumerate(all_rows):
        if _skip_row(row):
            continue

        if recomputed is not None:
            raw_signal: Any = recomputed[index]
        else:
            raw_signal = (
                signal_transform(row) if signal_transform is not None else row.get(signal_column)
            )
        signal = _as_float(raw_signal)
        if signal is None:
            continue
        prompt_tokens = _as_float(row.get("prompt_tokens_total"), 0.0) or 0.0
        generation_tokens = _as_float(row.get("generation_tokens_total"), 0.0) or 0.0
        if prompt_tokens + generation_tokens <= 0.0:
            continue

        # The shared label (tre_calibration.labels): p95 TTFT/TPOT against the SLOs, and a
        # window holding an unserved request is violated even without a latency sample.
        label = label_window(row, label_spec)
        if label is None:
            continue
        p95_ratio_max = label.ratio_max
        queue_raw: float | None = None
        if lambda_wait is not None and _as_float(row.get("avg_running")) is not None:
            queue_raw = tss_queue(
                _as_float(row.get("avg_running"), 0.0) or 0.0,
                _as_float(row.get("avg_waiting"), 0.0) or 0.0,
                float(lambda_wait),
            )
        windows.append(
            CalibrationWindow(
                scenario_id=(row.get("scenario_id") or "unknown").strip() or "unknown",
                scenario_family=(row.get("scenario_family") or "unknown").strip() or "unknown",
                signal=signal,
                slo_met=label.slo_met,
                health_score=1.0 / (1.0 + p95_ratio_max),
                window_start_ms=_as_float(row.get("window_start_ms")),
                latency_ratio_p95=p95_ratio_max,
                # avg_* columns are fixed-threshold ratios; meaningless under slowdown.
                latency_ratio_avg=None if slowdown else _avg_latency_ratio(row, latency_slo_ms),
                queue_raw=queue_raw,
                violation_class=label.violation_class,
            )
        )
    return trim_scenario_ramp_windows(windows, count=trim_ramp_windows)


def trim_scenario_ramp_windows(
    windows: Iterable[CalibrationWindow],
    *,
    count: int,
) -> list[CalibrationWindow]:
    """Drop the earliest windows of each scenario while preserving CSV row order."""
    if count < 0:
        raise ValueError("trim_ramp_windows must be non-negative")
    rows = list(windows)
    if count == 0:
        return rows

    by_scenario: dict[str, list[tuple[int, CalibrationWindow]]] = {}
    for index, window in enumerate(rows):
        by_scenario.setdefault(window.scenario_id, []).append((index, window))

    dropped: set[int] = set()
    for entries in by_scenario.values():
        ordered = sorted(
            entries,
            key=lambda item: (
                item[1].window_start_ms is None,
                item[1].window_start_ms
                if item[1].window_start_ms is not None
                else item[0],
                item[0],
            ),
        )
        dropped.update(index for index, _window in ordered[:count])
    return [window for index, window in enumerate(rows) if index not in dropped]


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


_AVG_LATENCY_COLUMNS = {
    "ttft_p95": "avg_ttft",
    "tpot_p95": "avg_tpot",
    "e2e_p95": "avg_e2e",
}


def _avg_latency_ratio(
    row: Mapping[str, Any], latency_slo_ms: Mapping[str, float]
) -> float | None:
    """Mean of the average-latency/SLO ratios, or None when the CSV lacks them."""
    ratios: list[float] = []
    for slo_key in latency_slo_ms:
        column = _AVG_LATENCY_COLUMNS.get(slo_key)
        if column is None:
            return None
        value = _as_float(row.get(column))
        if value is None:
            return None
        ratios.append(value / float(latency_slo_ms[slo_key]))
    if not ratios:
        return None
    return sum(ratios) / len(ratios)

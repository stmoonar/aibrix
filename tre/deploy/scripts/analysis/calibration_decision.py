#!/usr/bin/env python3
"""The pre-registered analysis of a calibration run (preregistration-20260923 §3-§5).

Input is one standard dataset directory (``windows.csv`` / ``cells.csv`` /
``manifest.json``, see ``docs/DATASET.md``) plus the regime grouping fixed before the run
(``regime_groups.json``, from :mod:`scripts.analysis.calibration_priors`). Output is one
structured JSON and one human-readable markdown report.

What it decides, and by which rule (all fixed in the preregistration, none tuned here):

* **w_p** per model, over the grid :data:`W_P_GRID`. The baseline is w_p = 0, one shared
  theta, lambda_wait = 3. A candidate replaces the baseline only if *all three* hold:
  (1) its LORO-min balanced accuracy is at least :data:`MIN_GAIN_POINTS` above the
  baseline's; (2) the 90 % cell-clustered bootstrap interval of that difference excludes
  0; (3) on the merged hold-out set its paired BA difference against the baseline is not
  negative. Otherwise the baseline wins - "no difference detected at this resolution" is
  a verdict, not a missing one.
* **shared w_p**: the models share one value when the worst per-model loss of the shared
  value against the model's own choice (LORO-min) is under :data:`SHARE_LOSS_POINTS`.
* **regime-aware theta(phi)** (secondary ablation only): adopted only if it passes the
  same three conditions against the single theta in at least
  :data:`REGIME_AWARE_MIN_MODELS` of the three models.
* **comparisons** on the merged hold-out set (TRS vs q-per-replica vs two signals) are
  paired: only windows the two methods label differently move the difference, and the
  interval comes from a cell-clustered bootstrap. A difference of at least
  :data:`DECISIVE_POINTS` whose 90 % interval excludes 0 names a winner; anything else is
  reported as equivalent.

LORO-min: hold out one regime group, fit one theta on the other two with the
balanced-accuracy criterion, score BA on the held-out group; the score is the minimum of
the three held-out BAs. A held-out group lacking one of the two classes has no BA and is
left out of the minimum (the count is reported).

The merged hold-out set - every cell of the held-out shape M plus the ramp cells of the
training shapes - is cut away in :func:`split_dataset`, the first thing :func:`analyse`
does. What is returned for it is a :class:`HoldoutSet`, which exposes no windows at all;
the only way to read them is :meth:`HoldoutSet.evaluate`, which demands a
:class:`FrozenSelection` - the object the selection stage returns once every fit is
done. Every selection-stage function takes a :class:`TrainingSet` and refuses anything
else. The hold-out therefore cannot reach a fit or the choice among candidates; it enters
the decision only as preregistered condition (3), a veto on the single candidate the
training data already chose (a vetoed candidate is not replaced by the runner-up, which
would let the hold-out pick).

Signals are never re-implemented here. TRS (with the controller's time-constant EMA,
one ``TRSComputer`` per cell, windows in time order - exactly
``r3_grid.compute_window_results``) comes from ``tre_controller.signals.trs``; every raw
value is cross-checked against ``tre_calibration.signals.compute_trs``. Thresholds come
from ``tre_calibration.fit.ThetaFitConfig`` (``fit_theta``, balanced accuracy) and are
scored with ``tre_calibration.fit.threshold_balanced_accuracy``. Window rows become
``CalibrationWindow`` through ``tre_calibration.dataset.calibration_window_from_row``, so
the label is ``tre_common.slo_labels.window_slo_label``.

Usage::

    python3 -m scripts.analysis.calibration_decision <dataset_dir> \\
        --regime-groups <regime_groups.json> --out-dir <dir> [--profile preregistered]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import multiprocessing
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from tre_calibration.dataset import CalibrationWindow, calibration_window_from_row
from tre_calibration.fit import ThetaFitConfig, threshold_balanced_accuracy
from tre_calibration.signals import SignalInputs, compute_trs
from tre_common import slo_labels
from tre_common.registry import load_registry
from tre_controller.signals.trs import TRSComputer, TRSInput

# ------------------------------------------------------------ preregistered constants

PREREGISTRATION = "docs/preregistration-20260923-calibration-run2.md"
W_P_GRID: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)
BASELINE_W_P = 0.0
MAIN_LAMBDA = 3.0
SENSITIVITY_LAMBDA = 10.0
MIN_GAIN_POINTS = 3.0
CI_LEVEL = 0.90
SHARE_LOSS_POINTS = 1.0
REGIME_AWARE_MIN_MODELS = 2
DECISIVE_POINTS = 3.0
DEFAULT_HOLD_WARMUP_S = 60.0
#: Window width (preregistration §6); used only to count non-overlapping windows.
WINDOW_MS = 30_000
DEFAULT_BOOTSTRAP = 1000
DEFAULT_SEED = 20260923

#: One configuration for every TRS / speed fit, one for the pressure signal (queue).
THETA_CONFIG = ThetaFitConfig()
QUEUE_CONFIG = ThetaFitConfig(direction="lower_is_healthier")
#: Stand-in for "no pressure at all" (empty queue, idle batch): finite, so every fit and
#: scorer keeps the window rather than silently dropping it as non-finite.
BIG = 1e12

ROLE_TRAIN = "train"
ROLE_HOLDOUT = "holdout"
ROLE_EXCLUDED = "excluded"
KNOWN_PRIMITIVES = frozenset({"hold", "steps", "ramp", "bursts"})


def points(x: float | None) -> float | None:
    """Balanced accuracy (0-1) -> BA points (0-100)."""
    return None if x is None else 100.0 * x


# --------------------------------------------------------------------- cell roles


@dataclass(frozen=True)
class CellPolicy:
    """Which cells train, which are held out, which are not analysed.

    Held out (preregistration §5.3): every cell of a ``holdout_shapes`` shape and every
    ``ramp`` cell. Trained on (§5.2): ``hold`` cells whose stage is in
    ``train_hold_stages`` plus any other primitive in ``train_primitives``. A ``hold``
    stage or a primitive this policy does not name is an error, never a silent drop -
    a renamed stage must not quietly empty the training set.
    """

    name: str
    train_primitives: frozenset[str]
    train_hold_stages: frozenset[str]
    excluded_hold_stages: frozenset[str]
    excluded_primitives: frozenset[str]
    holdout_shapes: frozenset[str] = frozenset({"M"})
    holdout_primitives: frozenset[str] = frozenset({"ramp"})
    hold_warmup_s: float = DEFAULT_HOLD_WARMUP_S

    def role(self, row: Mapping[str, Any]) -> str:
        shape, primitive = row["shape"], row["primitive"]
        if shape in self.holdout_shapes or primitive in self.holdout_primitives:
            return ROLE_HOLDOUT
        if primitive == "hold":
            stage = (row.get("stage") or "").strip()
            if stage in self.train_hold_stages:
                return ROLE_TRAIN
            if stage in self.excluded_hold_stages:
                return ROLE_EXCLUDED
            raise ValueError(
                f"hold cell {row.get('cell_id')} ({row.get('model')}/{shape}) has stage "
                f"{stage!r}, which policy {self.name!r} neither trains on "
                f"{sorted(self.train_hold_stages)} nor excludes {sorted(self.excluded_hold_stages)}"
            )
        if primitive in self.train_primitives:
            return ROLE_TRAIN
        if primitive in self.excluded_primitives:
            return ROLE_EXCLUDED
        raise ValueError(f"primitive {primitive!r} is not covered by policy {self.name!r}")


#: The second run as pre-registered: hold ladder + adaptive supplement cells train;
#: boundary-search probes and sentinels are not analysed; M and ramps are held out.
PREREGISTERED = CellPolicy(
    name="preregistered",
    train_primitives=frozenset({"hold"}),
    train_hold_stages=frozenset({"ladder", "adaptive"}),
    excluded_hold_stages=frozenset({"coarse", "bisect", "dwell", "sentinel"}),
    excluded_primitives=frozenset({"steps", "bursts"}),
)
#: The first run had no ladder: its training cells were the boundary probes, the
#: capacity steps and the bursts. Used only to dry-run the pipeline on that data.
FIRST_RUN = CellPolicy(
    name="first-run",
    train_primitives=frozenset({"hold", "steps", "bursts"}),
    train_hold_stages=frozenset({"coarse", "bisect", "dwell"}),
    excluded_hold_stages=frozenset(),
    excluded_primitives=frozenset(),
)
POLICIES = {PREREGISTERED.name: PREREGISTERED, FIRST_RUN.name: FIRST_RUN}


# ------------------------------------------------------------------------ cells


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


@dataclass
class Cell:
    """One driven attempt: every window row (time order) plus derived per-window series.

    ``rows`` keeps *all* windows - unlabeled ones and the warm-up included - because the
    controller's EMA runs through them; ``kept`` indexes the windows that are scored.
    """

    uid: str
    model: str
    shape: str
    primitive: str
    stage: str
    group: str | None
    rows: list[dict[str, Any]]
    kept: list[int]
    labels: list[bool]  # slo_met of each kept window
    labels_ttft: list[bool]
    labels_tpot: list[bool]
    base: list[CalibrationWindow]  # kept windows, signal placeholder
    params: Any  # TrsParams of the model
    _series: dict[tuple, list[float]] = field(default_factory=dict, repr=False)
    _windows: dict[tuple, list[CalibrationWindow]] = field(default_factory=dict, repr=False)

    @property
    def n_violating(self) -> int:
        return sum(1 for ok in self.labels if not ok)

    @property
    def independent_violating(self) -> int:
        """Violating windows that do not overlap each other (greedy in time): evidence is
        counted in independent time spans, not in 5 s sliding-window rows (§6)."""
        count, free_from = 0, -math.inf
        for w, ok in zip(self.base, self.labels):
            start = w.window_start_ms or 0.0
            if not ok and start >= free_from:
                count += 1
                free_from = start + WINDOW_MS
        return count

    # -- series over every row -------------------------------------------------------
    def series(self, key: tuple) -> list[float]:
        if key not in self._series:
            kind = key[0]
            if kind == "trs":
                self._series[key] = trs_series(self.rows, self.params, w_p=key[1], lambda_wait=key[2])
            elif kind == "q":
                self._series[key] = [
                    key[1] * _f(r["avg_waiting"]) + _f(r["avg_running"]) + _f(r["avg_swapping"])
                    for r in self.rows
                ]
            elif kind == "speed":
                self._series[key] = [decode_speed_per_sequence(r) for r in self.rows]
            elif kind == "phi":
                self._series[key] = prefill_share_series(self.rows, self.params)
            else:
                raise KeyError(key)
        return self._series[key]

    def values(self, key: tuple) -> list[float]:
        """``key``'s series on the kept (scored) windows."""
        s = self.series(key)
        return [s[i] for i in self.kept]

    def windows(self, key: tuple, label: str = "full") -> list[CalibrationWindow]:
        """Kept windows carrying ``key``'s signal and the named label."""
        ck = (key, label)
        if ck not in self._windows:
            vals = self.values(key)
            labs = {"full": self.labels, "ttft": self.labels_ttft, "tpot": self.labels_tpot}[label]
            self._windows[ck] = [
                dataclasses.replace(w, signal=v, slo_met=ok) for w, v, ok in zip(self.base, vals, labs)
            ]
        return self._windows[ck]


def trs_series(rows: Sequence[Mapping[str, Any]], params: Any, *, w_p: float, lambda_wait: float) -> list[float]:
    """EMA'd TRS per window row - ``r3_grid.compute_window_results`` with the TRS weights
    replaced. One ``TRSComputer`` for the cell, rows in time order, single replica. Every
    raw value is checked against the archived formula ``tre_calibration.signals.compute_trs``."""
    computer = TRSComputer(ema_alpha=params.ema_alpha, ema_tau_ms=params.ema_tau_ms)
    out: list[float] = []
    for r in rows:
        inp = TRSInput(
            prompt_tokens_total=_f(r["prompt_tokens_total"]),
            generation_tokens_total=_f(r["generation_tokens_total"]),
            avg_waiting=_f(r["avg_waiting"]),
            avg_running=_f(r["avg_running"]),
            avg_swapping=_f(r["avg_swapping"]),
            routable_pods=1,
            assigned_replicas=1,
            w_p=w_p,
            w_d=params.w_d,
            lambda_wait=lambda_wait,
            qmin=params.qmin,
            kv_cache_hit_rate=0.0,
        )
        res = computer.compute(inp, theta_m=None, window_end_ms=int(_f(r["window_end_ms"])))
        archived = compute_trs(
            SignalInputs(inp.prompt_tokens_total, inp.generation_tokens_total, inp.avg_waiting,
                         inp.avg_running, inp.avg_swapping),
            w_p=w_p, lambda_wait=lambda_wait, qmin=params.qmin,
        )
        if params.w_d == 1.0 and not math.isclose(archived.trs_floor, res.TRS_raw, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError(f"TRS raw mismatch: controller {res.TRS_raw} vs compute_trs {archived.trs_floor}")
        out.append(res.TRS)
    return out


def decode_speed_per_sequence(row: Mapping[str, Any]) -> float:
    """Completed decode tokens per second per running sequence (~ 1/TPOT); higher is
    healthier. An idle batch has no per-sequence speed to fall short: :data:`BIG`."""
    duration_s = (_f(row["window_end_ms"]) - _f(row["window_start_ms"])) / 1000.0
    running = _f(row["avg_running"])
    if running <= 0 or duration_s <= 0:
        return BIG
    return _f(row["generation_tokens_total"]) / duration_s / running


def prefill_share_series(rows: Sequence[Mapping[str, Any]], params: Any) -> list[float]:
    """phi = P / (P + D) per window (completed prompt vs generated tokens), smoothed by the
    controller's own time-constant EMA (``TRSComputer._update_ema``). Independent of w_p
    on purpose: at the baseline w_p = 0 a w_p-weighted share would be identically 0."""
    computer = TRSComputer(ema_alpha=params.ema_alpha, ema_tau_ms=params.ema_tau_ms)
    out = []
    for r in rows:
        p, d = _f(r["prompt_tokens_total"]), _f(r["generation_tokens_total"])
        raw = p / (p + d) if p + d > 0 else float("nan")
        out.append(computer._update_ema(raw, window_end_ms=int(_f(r["window_end_ms"]))))
    return out


def build_cells(
    rows: Iterable[Mapping[str, Any]],
    *,
    slo_ms: Mapping[str, float],
    registry: Any,
    groups: Mapping[str, Mapping[str, str]],
    cell_start_ms: Mapping[tuple, float],
    hold_warmup_s: float,
) -> dict[str, Cell]:
    """Window rows -> :class:`Cell` objects (labels checked against the dataset's own)."""
    by_cell: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["model"], row["shape"], row["primitive"], row.get("stage") or "",
               row["cell_id"], str(row["attempt"]))
        by_cell[key].append(dict(row))
    ttft_only = {"ttft_p95": slo_ms["ttft_p95"]}
    tpot_only = {"tpot_p95": slo_ms["tpot_p95"]}
    cells: dict[str, Cell] = {}
    for key, cell_rows in sorted(by_cell.items()):
        model, shape, primitive, stage, cell_id, attempt = key
        cell_rows.sort(key=lambda r: _f(r["window_end_ms"]))
        uid = f"{model}|{shape}|{primitive}|{stage}|{cell_id}|a{attempt}"
        start = cell_start_ms.get((model, shape, primitive, cell_id, attempt))
        warmup_ms = hold_warmup_s * 1000.0 if primitive == "hold" else 0.0
        kept, labels, lt, lp, base = [], [], [], [], []
        for i, r in enumerate(cell_rows):
            if warmup_ms and start is not None and _f(r["window_start_ms"]) - start < warmup_ms:
                continue
            label = slo_labels.window_slo_label(r, slo_ms)
            recorded = (r.get(slo_labels.LABEL_COLUMN) or "").strip()
            if recorded and recorded != label:
                raise AssertionError(f"{uid}: label {label} disagrees with dataset {recorded}")
            if label == slo_labels.LABEL_UNLABELED:
                continue
            w = calibration_window_from_row(r, latency_slo_ms=slo_ms, signal=0.0)
            if w is None:
                # Only a zero-token window gets here with a label; it is a violation
                # (nothing was served) and must stay one (preregistration §6).
                if label != slo_labels.LABEL_VIOLATED:
                    continue
                w = CalibrationWindow(
                    scenario_id=uid, scenario_family=r.get("scenario_family") or shape,
                    signal=0.0, slo_met=False, window_start_ms=_f(r["window_start_ms"]),
                )
            kept.append(i)
            labels.append(w.slo_met)
            lt.append(slo_labels.window_slo_label(r, ttft_only) == slo_labels.LABEL_HEALTHY)
            lp.append(slo_labels.window_slo_label(r, tpot_only) == slo_labels.LABEL_HEALTHY)
            base.append(dataclasses.replace(w, scenario_id=uid))
        group = groups.get(model, {}).get(shape)
        cells[uid] = Cell(
            uid=uid, model=model, shape=shape, primitive=primitive, stage=stage, group=group,
            rows=cell_rows, kept=kept, labels=labels, labels_ttft=lt, labels_tpot=lp, base=base,
            params=registry.model(model).trs,
        )
    return cells


# ------------------------------------------------------------ the split (isolation)


class TrainingSet:
    """Cells that may be fitted on and selected with (preregistration §5.2)."""

    def __init__(self, cells: Mapping[str, Cell]):
        self._cells = dict(cells)
        missing = sorted({c.shape for c in self._cells.values() if c.group is None})
        if missing:
            raise ValueError(f"training shapes without a regime group: {missing}")

    def models(self) -> list[str]:
        return sorted({c.model for c in self._cells.values()})

    def cells(self, model: str) -> list[Cell]:
        return [c for c in self._cells.values() if c.model == model and c.kept]

    def uids(self) -> set[str]:
        return set(self._cells)


class FrozenSelection:
    """Everything the selection stage decided, fits included. Created only by
    :func:`select`; the key to :meth:`HoldoutSet.evaluate`."""

    def __init__(self, selection: dict[str, Any], fits: dict[str, dict[str, Any]]):
        self.selection = selection
        self.fits = fits  # model -> method name -> fitted params


class HoldoutSet:
    """The merged hold-out set (§5.3). Deliberately has no window accessor: its cells are
    readable only through :meth:`evaluate`, which requires the frozen selection."""

    def __init__(self, cells: Mapping[str, Cell]):
        self.__cells = dict(cells)

    def __repr__(self) -> str:
        return f"HoldoutSet(<{len(self.__cells)} sealed cells>)"

    def summary(self) -> dict[str, Any]:
        """Counts only - what the report may say about the set before it is opened."""
        out: dict[str, Any] = {}
        for c in self.__cells.values():
            m = out.setdefault(c.model, {"cells": 0, "windows": 0})
            m["cells"] += 1
            m["windows"] += len(c.kept)
        return out

    def evaluate(self, frozen: "FrozenSelection", fn: Callable[[dict[str, list[Cell]], FrozenSelection], Any]) -> Any:
        if not isinstance(frozen, FrozenSelection):
            raise TypeError("the hold-out set opens only for a FrozenSelection")
        by_model: dict[str, list[Cell]] = defaultdict(list)
        for c in self.__cells.values():
            if c.kept:
                by_model[c.model].append(c)
        return fn(dict(by_model), frozen)


def split_dataset(cells: Mapping[str, Cell], policy: CellPolicy) -> tuple[TrainingSet, HoldoutSet, dict[str, int]]:
    train, holdout, excluded = {}, {}, {}
    for uid, cell in cells.items():
        role = policy.role({"shape": cell.shape, "primitive": cell.primitive, "stage": cell.stage,
                            "cell_id": uid, "model": cell.model})
        {ROLE_TRAIN: train, ROLE_HOLDOUT: holdout, ROLE_EXCLUDED: excluded}[role][uid] = cell
    counts = {"train": len(train), "holdout": len(holdout), "excluded": len(excluded)}
    return TrainingSet(train), HoldoutSet(holdout), counts


def _require_training(data: Any) -> None:
    if not isinstance(data, TrainingSet):
        raise TypeError(f"selection stage accepts only a TrainingSet, got {type(data).__name__}")


# ------------------------------------------------------------------------ methods
#
# A method fits parameters on a list of cells and turns cells into "margin" windows:
# signal >= 1 means the method predicts healthy. Scoring every method as a margin at 1
# lets one BA function (threshold_balanced_accuracy at theta = 1) score all of them.


def _concat(cells: Sequence[Cell], key: tuple, label: str = "full") -> list[CalibrationWindow]:
    out: list[CalibrationWindow] = []
    for c in cells:
        out.extend(c.windows(key, label))
    return out


class TrsSingle:
    def __init__(self, w_p: float, lambda_wait: float):
        self.w_p, self.lambda_wait = w_p, lambda_wait
        self.key = ("trs", w_p, lambda_wait)
        self.name = f"trs_single(w_p={w_p:g},lambda={lambda_wait:g})"

    def fit(self, cells: Sequence[Cell]) -> dict[str, Any]:
        f = THETA_CONFIG.fit(_concat(cells, self.key))
        return {"theta": f.theta, "publish": f.publish, "reject_reason": f.reject_reason}

    def margins(self, cells: Sequence[Cell], params: Mapping[str, Any]) -> list[CalibrationWindow]:
        th = params["theta"]
        return [dataclasses.replace(w, signal=(w.signal / th if th else BIG)) for w in _concat(cells, self.key)]


class QueueSingle:
    def __init__(self, lambda_wait: float):
        self.lambda_wait = lambda_wait
        self.key = ("q", lambda_wait)
        self.name = f"queue_per_replica(lambda={lambda_wait:g})"

    def fit(self, cells: Sequence[Cell]) -> dict[str, Any]:
        f = QUEUE_CONFIG.fit(_concat(cells, self.key))
        return {"theta": f.theta, "publish": f.publish, "reject_reason": f.reject_reason}

    def margins(self, cells: Sequence[Cell], params: Mapping[str, Any]) -> list[CalibrationWindow]:
        th = params["theta"]
        return [dataclasses.replace(w, signal=(th / w.signal if w.signal > 0 else BIG)) for w in _concat(cells, self.key)]


class TwoSignal:
    """In-flight count guards TTFT, per-sequence decode speed guards TPOT; a window is
    predicted violated if either crosses. Each threshold is fitted against its own SLO
    component (unserved requests count against both), then scored on the full label."""

    def __init__(self, lambda_wait: float):
        self.lambda_wait = lambda_wait
        self.qkey = ("q", lambda_wait)
        self.skey = ("speed",)
        self.name = f"two_signal(lambda={lambda_wait:g})"

    def fit(self, cells: Sequence[Cell]) -> dict[str, Any]:
        fq = QUEUE_CONFIG.fit(_concat(cells, self.qkey, "ttft"))
        fs = THETA_CONFIG.fit(_concat(cells, self.skey, "tpot"))
        return {
            "theta_q": fq.theta if fq.violating_window_count > 0 else None,
            "theta_speed": fs.theta if fs.violating_window_count > 0 else None,
            "ttft_violating": fq.violating_window_count,
            "tpot_violating": fs.violating_window_count,
        }

    def margins(self, cells: Sequence[Cell], params: Mapping[str, Any]) -> list[CalibrationWindow]:
        out = []
        for c in cells:
            qs, ss = c.values(self.qkey), c.values(self.skey)
            for w, q, s in zip(c.base, qs, ss):
                a = params["theta_q"] / q if params["theta_q"] and q > 0 else BIG
                b = s / params["theta_speed"] if params["theta_speed"] else BIG
                out.append(dataclasses.replace(w, signal=min(a, b)))
        # base windows carry the full label already
        return out


class RegimeAware:
    """theta(phi): one theta per training regime group, anchored at the group's median
    phi, linearly interpolated between anchors and clamped outside them."""

    def __init__(self, w_p: float, lambda_wait: float):
        self.w_p, self.lambda_wait = w_p, lambda_wait
        self.key = ("trs", w_p, lambda_wait)
        self.phi = ("phi",)
        self.name = f"trs_regime_aware(w_p={w_p:g},lambda={lambda_wait:g})"

    def fit(self, cells: Sequence[Cell]) -> dict[str, Any]:
        by_group: dict[str, list[Cell]] = defaultdict(list)
        for c in cells:
            by_group[c.group].append(c)
        anchors = []
        for g, gcells in sorted(by_group.items()):
            f = THETA_CONFIG.fit(_concat(gcells, self.key))
            phis = [p for c in gcells for p in c.values(self.phi) if math.isfinite(p)]
            if f.theta is None or not phis:
                continue
            anchors.append({"group": g, "phi": statistics.median(phis), "theta": f.theta})
        anchors.sort(key=lambda a: a["phi"])
        return {"anchors": anchors}

    @staticmethod
    def theta_at(anchors: Sequence[Mapping[str, float]], phi: float) -> float | None:
        if not anchors:
            return None
        if not math.isfinite(phi) or phi <= anchors[0]["phi"]:
            return anchors[0]["theta"]
        if phi >= anchors[-1]["phi"]:
            return anchors[-1]["theta"]
        for lo, hi in zip(anchors, anchors[1:]):
            if lo["phi"] <= phi <= hi["phi"]:
                span = hi["phi"] - lo["phi"]
                x = (phi - lo["phi"]) / span if span > 0 else 0.0
                return lo["theta"] + (hi["theta"] - lo["theta"]) * x
        return anchors[-1]["theta"]

    def margins(self, cells: Sequence[Cell], params: Mapping[str, Any]) -> list[CalibrationWindow]:
        out = []
        for c in cells:
            for w, phi in zip(c.windows(self.key), c.values(self.phi)):
                th = self.theta_at(params["anchors"], phi)
                out.append(dataclasses.replace(w, signal=(w.signal / th if th else BIG)))
        return out


def balanced_accuracy(margins: Sequence[CalibrationWindow]) -> float | None:
    """BA of "margin >= 1 => healthy"; None unless both classes are present."""
    healthy = sum(1 for w in margins if w.slo_met)
    if healthy == 0 or healthy == len(margins):
        return None
    return threshold_balanced_accuracy(margins, theta=1.0)["balanced_accuracy"]


def predictions(margins: Sequence[CalibrationWindow]) -> list[bool]:
    return [w.signal >= 1.0 for w in margins]


# --------------------------------------------------------------------------- LORO


def loro(cells: Sequence[Cell], method: Any) -> dict[str, Any]:
    """Leave-one-regime-out: fit on two groups, BA on the third; score = min over groups."""
    groups = sorted({c.group for c in cells})
    folds: dict[str, Any] = {}
    for g in groups:
        train = [c for c in cells if c.group != g]
        held = [c for c in cells if c.group == g]
        params = method.fit(train)
        folds[g] = {"ba": balanced_accuracy(method.margins(held, params)),
                    "windows": sum(len(c.kept) for c in held),
                    "violating": sum(c.n_violating for c in held)}
    defined = [f["ba"] for f in folds.values() if f["ba"] is not None]
    return {
        "folds": folds,
        "min": min(defined) if defined else None,
        "mean": statistics.fmean(defined) if defined else None,
        "undefined_folds": [g for g, f in folds.items() if f["ba"] is None],
    }


def resample_cells(cells: Sequence[Cell], rng: random.Random) -> list[Cell]:
    """Cell-clustered bootstrap draw, stratified by regime group (every fold survives)."""
    by_group: dict[str, list[Cell]] = defaultdict(list)
    for c in cells:
        by_group[c.group].append(c)
    out: list[Cell] = []
    for g in sorted(by_group):
        members = by_group[g]
        out.extend(rng.choice(members) for _ in range(len(members)))
    return out


_BOOT_STATE: dict[str, Any] = {}


def _boot_worker(indices: Sequence[int]) -> list[list[float | None]]:
    cells, methods, seed = _BOOT_STATE["cells"], _BOOT_STATE["methods"], _BOOT_STATE["seed"]
    rows = []
    for b in indices:
        draw = resample_cells(cells, random.Random(f"{seed}:{b}"))
        rows.append([loro(draw, m)["min"] for m in methods])
    return rows


def bootstrap_loro(
    data: TrainingSet, model: str, methods: Sequence[Any], *, n: int, seed: str, processes: int = 1
) -> list[list[float | None]]:
    """``n`` paired replicates (the same cell draw for every method) of LORO-min."""
    _require_training(data)
    cells = data.cells(model)
    _BOOT_STATE.update(cells=cells, methods=list(methods), seed=seed)
    size = max(1, n // max(1, processes * 4))
    chunks = [list(range(i, min(n, i + size))) for i in range(0, n, size)]
    if processes > 1 and n > 1:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes) as pool:
            parts = pool.map(_boot_worker, chunks)
    else:
        parts = [_boot_worker(ch) for ch in chunks]
    return [row for part in parts for row in part]


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile; None for an empty list."""
    xs = sorted(values)
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def interval(diffs: Sequence[float], level: float = CI_LEVEL) -> tuple[float | None, float | None]:
    a = (1.0 - level) / 2.0
    return percentile(diffs, a), percentile(diffs, 1.0 - a)


# ------------------------------------------------------------- decision rules (pure)


def candidate_conditions(
    *, gain_points: float | None, ci_low_points: float | None, ci_high_points: float | None,
    holdout_diff_points: float | None = None,
) -> dict[str, Any]:
    """The three preregistered conditions for one candidate against the baseline.

    ``gain_points`` is LORO-min(candidate) - LORO-min(baseline); the interval is the
    90 % bootstrap interval of that difference; ``holdout_diff_points`` is the paired BA
    difference on the merged hold-out set (None = not evaluated yet)."""
    c1 = gain_points is not None and gain_points >= MIN_GAIN_POINTS
    # The candidate must be the better one, so "does not cross 0" means the whole
    # interval lies above 0.
    c2 = ci_low_points is not None and ci_high_points is not None and ci_low_points > 0.0
    c3 = None if holdout_diff_points is None else holdout_diff_points >= 0.0
    return {
        "loro_gain_ge_3": c1,
        "ci90_excludes_0": c2,
        "holdout_not_worse": c3,
        "passes_selection": c1 and c2,
        "passes": bool(c1 and c2 and c3),
    }


def pick_selection_winner(candidates: Sequence[Mapping[str, Any]]) -> Any:
    """Among candidates passing conditions (1)+(2), the highest LORO-min (ties: the smaller
    value). None when none passes - the baseline stands. Uses training data only."""
    passing = [c for c in candidates if c["conditions"]["passes_selection"]]
    if not passing:
        return None
    return max(passing, key=lambda c: (c["loro_min"], -c["value"]))["value"]


def final_choice(baseline: Any, winner: Any, holdout_diff_points: float | None) -> dict[str, Any]:
    """Apply condition (3) to the one training-selected winner. A veto reverts to the
    baseline; the runner-up is never promoted (that would let the hold-out select)."""
    if winner is None:
        return {"value": baseline, "replaced_baseline": False,
                "verdict": "baseline wins: no difference detected at this data's resolution"}
    if holdout_diff_points is None or holdout_diff_points < 0.0:
        return {"value": baseline, "replaced_baseline": False,
                "verdict": "baseline wins: the training-selected candidate is worse than the "
                           "baseline on the merged hold-out set (condition 3)"}
    return {"value": winner, "replaced_baseline": True,
            "verdict": "candidate replaces baseline: all three preregistered conditions hold"}


def regime_aware_adopted(passes_by_model: Mapping[str, bool]) -> bool:
    """Regime-aware theta needs the three conditions in >= 2 of the 3 models."""
    return sum(1 for v in passes_by_model.values() if v) >= REGIME_AWARE_MIN_MODELS


def shared_w_p(chosen: Mapping[str, float], loro_min: Mapping[str, Mapping[float, float | None]]) -> dict[str, Any]:
    """Share one w_p across models if its worst loss vs each model's own choice is under
    :data:`SHARE_LOSS_POINTS` BA points (LORO-min). Candidates are the chosen values;
    ties prefer the baseline, then the smaller value."""
    best = None
    for cand in sorted(set(chosen.values()), key=lambda v: (v != BASELINE_W_P, v)):
        losses = {}
        for m, own in chosen.items():
            a, b = loro_min[m].get(own), loro_min[m].get(cand)
            losses[m] = None if a is None or b is None else 100.0 * (a - b)
        worst = max((v for v in losses.values() if v is not None), default=None)
        if worst is None:
            continue
        if best is None or worst < best["worst_loss_points"] - 1e-12:
            best = {"value": cand, "losses_points": losses, "worst_loss_points": worst}
    if best is None:
        return {"adopt": False, "value": None, "reason": "no LORO-min available"}
    best["adopt"] = best["worst_loss_points"] < SHARE_LOSS_POINTS
    return best


def paired_verdict(diff_points: float | None, ci_low: float | None, ci_high: float | None) -> str:
    """Hold-out comparison wording (§5.3): a winner needs >= 3 points AND an interval that
    excludes 0; otherwise "equivalent" (no winner is reported)."""
    if diff_points is None:
        return "undefined"
    if abs(diff_points) >= DECISIVE_POINTS and ci_low is not None and ci_high is not None \
            and (ci_low > 0 or ci_high < 0):
        return "A better" if diff_points > 0 else "B better"
    return "equivalent"


# -------------------------------------------------------------------- selection


def select_model(
    data: TrainingSet, model: str, *, lambda_wait: float, n_boot: int, seed: int, processes: int
) -> dict[str, Any]:
    """w_p selection for one model at one lambda (training data only)."""
    _require_training(data)
    cells = data.cells(model)
    methods = [TrsSingle(w, lambda_wait) for w in W_P_GRID]
    point = {m.w_p: loro(cells, m) for m in methods}
    boot = bootstrap_loro(data, model, methods, n=n_boot, seed=f"{seed}:{model}:wp:{lambda_wait:g}",
                          processes=processes)
    base_idx = W_P_GRID.index(BASELINE_W_P)
    candidates = []
    for j, m in enumerate(methods):
        col = [r[j] for r in boot]
        diffs = [100.0 * (r[j] - r[base_idx]) for r in boot if r[j] is not None and r[base_idx] is not None]
        lo, hi = interval(diffs)
        gain = None if point[m.w_p]["min"] is None or point[BASELINE_W_P]["min"] is None \
            else 100.0 * (point[m.w_p]["min"] - point[BASELINE_W_P]["min"])
        defined = [v for v in col if v is not None]
        candidates.append({
            "value": m.w_p,
            "loro_min": point[m.w_p]["min"],
            "loro_mean": point[m.w_p]["mean"],
            "loro_folds": {g: f["ba"] for g, f in point[m.w_p]["folds"].items()},
            "undefined_folds": point[m.w_p]["undefined_folds"],
            "gain_points": gain,
            "boot_loro_min_p05": percentile(defined, 0.05),
            "boot_loro_min_p50": percentile(defined, 0.5),
            "boot_loro_min_p95": percentile(defined, 0.95),
            "boot_loro_min_sd_points": (100.0 * statistics.pstdev(defined)) if len(defined) > 1 else None,
            "boot_diff_ci90_points": [lo, hi],
            "boot_replicates_used": len(diffs),
            "boot_frac_gain_ge_3": (sum(1 for d in diffs if d >= MIN_GAIN_POINTS) / len(diffs)) if diffs else None,
            "conditions": candidate_conditions(gain_points=gain, ci_low_points=lo, ci_high_points=hi)
            if m.w_p != BASELINE_W_P else None,
        })
    real = [c for c in candidates if c["value"] != BASELINE_W_P]
    winner = pick_selection_winner(real)
    return {"lambda_wait": lambda_wait, "candidates": candidates, "selection_winner": winner,
            "folds": sorted({c.group for c in cells})}


def select_regime_aware(
    data: TrainingSet, model: str, *, w_p: float, lambda_wait: float, n_boot: int, seed: int, processes: int
) -> dict[str, Any]:
    _require_training(data)
    cells = data.cells(model)
    single, regime = TrsSingle(w_p, lambda_wait), RegimeAware(w_p, lambda_wait)
    ps, pr = loro(cells, single), loro(cells, regime)
    boot = bootstrap_loro(data, model, [single, regime], n=n_boot,
                          seed=f"{seed}:{model}:regime:{w_p:g}:{lambda_wait:g}", processes=processes)
    diffs = [100.0 * (r[1] - r[0]) for r in boot if r[0] is not None and r[1] is not None]
    lo, hi = interval(diffs)
    gain = None if ps["min"] is None or pr["min"] is None else 100.0 * (pr["min"] - ps["min"])
    return {
        "w_p": w_p, "lambda_wait": lambda_wait,
        "single_loro_min": ps["min"], "regime_loro_min": pr["min"],
        "single_folds": {g: f["ba"] for g, f in ps["folds"].items()},
        "regime_folds": {g: f["ba"] for g, f in pr["folds"].items()},
        "gain_points": gain, "boot_diff_ci90_points": [lo, hi],
        "boot_frac_gain_ge_3": (sum(1 for d in diffs if d >= MIN_GAIN_POINTS) / len(diffs)) if diffs else None,
        "conditions": candidate_conditions(gain_points=gain, ci_low_points=lo, ci_high_points=hi),
    }


def select(data: TrainingSet, *, n_boot: int, seed: int, processes: int) -> FrozenSelection:
    """The whole selection stage. Sees only ``data``; returns the frozen selection with
    every threshold the hold-out evaluation will apply, fitted on the full training set."""
    _require_training(data)
    out: dict[str, Any] = {"models": {}}
    fits: dict[str, dict[str, Any]] = {}
    for model in data.models():
        cells = data.cells(model)
        main = select_model(data, model, lambda_wait=MAIN_LAMBDA, n_boot=n_boot, seed=seed, processes=processes)
        sens = select_model(data, model, lambda_wait=SENSITIVITY_LAMBDA, n_boot=n_boot, seed=seed, processes=processes)
        winner = main["selection_winner"]
        at_wp = winner if winner is not None else BASELINE_W_P
        regime = select_regime_aware(data, model, w_p=at_wp, lambda_wait=MAIN_LAMBDA, n_boot=n_boot,
                                     seed=seed, processes=processes)
        controls = {
            "queue_per_replica": loro(cells, QueueSingle(MAIN_LAMBDA)),
            "two_signal": loro(cells, TwoSignal(MAIN_LAMBDA)),
        }
        methods = {
            "baseline": TrsSingle(BASELINE_W_P, MAIN_LAMBDA),
            "queue_per_replica": QueueSingle(MAIN_LAMBDA),
            "two_signal": TwoSignal(MAIN_LAMBDA),
            "regime_aware": RegimeAware(at_wp, MAIN_LAMBDA),
            "trs_at_selected_w_p": TrsSingle(at_wp, MAIN_LAMBDA),
            "baseline_lambda10": TrsSingle(BASELINE_W_P, SENSITIVITY_LAMBDA),
        }
        if sens["selection_winner"] is not None:
            methods["lambda10_selected"] = TrsSingle(sens["selection_winner"], SENSITIVITY_LAMBDA)
        fits[model] = {name: {"method": m, "params": m.fit(cells)} for name, m in methods.items()}
        out["models"][model] = {
            "training": {
                "cells": len(cells),
                "windows": sum(len(c.kept) for c in cells),
                "violating_windows": sum(c.n_violating for c in cells),
                "independent_violating_windows": sum(c.independent_violating for c in cells),
                "cells_with_violation": sum(1 for c in cells if c.n_violating),
                "by_group": {
                    g: {"shapes": sorted({c.shape for c in cells if c.group == g}),
                        "cells": sum(1 for c in cells if c.group == g),
                        "windows": sum(len(c.kept) for c in cells if c.group == g),
                        "violating_windows": sum(c.n_violating for c in cells if c.group == g),
                        "independent_violating_windows": sum(c.independent_violating for c in cells if c.group == g),
                        "cells_with_violation": sum(1 for c in cells if c.group == g and c.n_violating)}
                    for g in sorted({c.group for c in cells})
                },
            },
            "w_p": main,
            "lambda_sensitivity": sens,
            "regime_aware": regime,
            "controls_loro": {k: {"min": v["min"], "folds": {g: f["ba"] for g, f in v["folds"].items()}}
                              for k, v in controls.items()},
            "fitted_on_full_training": {name: _jsonable(f["params"]) for name, f in fits[model].items()},
        }
    return FrozenSelection(out, fits)


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


# ------------------------------------------------------------ hold-out evaluation


def paired_compare(
    cells: Sequence[Cell], a: tuple[Any, Mapping[str, Any]], b: tuple[Any, Mapping[str, Any]],
    *, n_boot: int, seed: str,
) -> dict[str, Any]:
    """Paired BA comparison A - B on ``cells`` with a cell-clustered bootstrap (strata: cells
    with / without a violating window). Only discordant windows move the difference."""
    per_cell = []
    for c in cells:
        ma, mb = a[0].margins([c], a[1]), b[0].margins([c], b[1])
        pa, pb = predictions(ma), predictions(mb)
        per_cell.append((c, [w.slo_met for w in ma], pa, pb))

    def ba_of(rows, idx):
        tp = fn = tn = fp = 0
        for _, labels, pa, pb in rows:
            preds = pa if idx == 0 else pb
            for ok, p in zip(labels, preds):
                if ok and p:
                    tp += 1
                elif ok:
                    fn += 1
                elif p:
                    fp += 1
                else:
                    tn += 1
        if tp + fn == 0 or tn + fp == 0:
            return None
        return 0.5 * (tp / (tp + fn) + tn / (tn + fp))

    def diff(rows):
        x, y = ba_of(rows, 0), ba_of(rows, 1)
        return None if x is None or y is None else 100.0 * (x - y)

    point_a, point_b = ba_of(per_cell, 0), ba_of(per_cell, 1)
    d = diff(per_cell)
    disc = {"healthy_a_only": 0, "healthy_b_only": 0, "violating_a_only": 0, "violating_b_only": 0}
    disc_cells = set()
    for c, labels, pa, pb in per_cell:
        for ok, x, y in zip(labels, pa, pb):
            if x == y:
                continue
            disc_cells.add(c.uid)
            right_a = x == ok
            disc[("healthy_" if ok else "violating_") + ("a_only" if right_a else "b_only")] += 1
    strata: dict[bool, list] = defaultdict(list)
    for row in per_cell:
        strata[row[0].n_violating > 0].append(row)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        draw = [rng.choice(members) for key in sorted(strata) for members in [strata[key]] for _ in members]
        v = diff(draw)
        if v is not None:
            diffs.append(v)
    lo, hi = interval(diffs)
    return {
        "ba_a": point_a, "ba_b": point_b, "diff_points": d, "ci90_points": [lo, hi],
        "verdict": paired_verdict(d, lo, hi),
        "discordant_windows": disc, "discordant_cells": len(disc_cells),
        "windows": sum(len(r[1]) for r in per_cell),
        "violating_windows": sum(1 for r in per_cell for ok in r[1] if not ok),
        "cells": len(per_cell), "cells_with_violation": len(strata.get(True, [])),
        "boot_replicates_used": len(diffs),
    }


def evaluate_holdout(by_model: Mapping[str, list[Cell]], frozen: FrozenSelection, *, n_boot: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model, cells in sorted(by_model.items()):
        fits = frozen.fits.get(model)
        if fits is None:
            continue
        sel = frozen.selection["models"][model]

        def pair(an: str, bn: str, subset: Sequence[Cell] = cells) -> dict[str, Any]:
            return paired_compare(subset, (fits[an]["method"], fits[an]["params"]),
                                  (fits[bn]["method"], fits[bn]["params"]), n_boot=n_boot,
                                  seed=f"{seed}:{model}:holdout:{an}:{bn}")

        chosen = "trs_at_selected_w_p"
        res: dict[str, Any] = {
            "cells": len(cells),
            "windows": sum(len(c.kept) for c in cells),
            "violating_windows": sum(c.n_violating for c in cells),
            "independent_violating_windows": sum(c.independent_violating for c in cells),
            "cells_with_violation": sum(1 for c in cells if c.n_violating),
            "by_source": {
                src: {"cells": len(cs), "windows": sum(len(c.kept) for c in cs),
                      "violating_windows": sum(c.n_violating for c in cs),
                      "independent_violating_windows": sum(c.independent_violating for c in cs),
                      "cells_with_violation": sum(1 for c in cs if c.n_violating)}
                for src, cs in (("M", [c for c in cells if c.shape == "M"]),
                                ("ramp", [c for c in cells if c.shape != "M"]))
            },
        }
        winner = sel["w_p"]["selection_winner"]
        res["condition3_w_p"] = None if winner is None else pair(chosen, "baseline")
        res["condition3_regime_aware"] = pair("regime_aware", chosen)
        res["trs_vs_queue_per_replica"] = pair(chosen, "queue_per_replica")
        res["trs_vs_two_signal"] = pair(chosen, "two_signal")
        res["lambda10_vs_lambda3_baseline"] = pair("baseline_lambda10", "baseline")
        if "lambda10_selected" in fits:
            res["condition3_lambda10_w_p"] = pair("lambda10_selected", "baseline_lambda10")
        m_cells = [c for c in cells if c.shape == "M"]
        res["m_only_ba"] = {
            name: balanced_accuracy(f["method"].margins(m_cells, f["params"])) for name, f in fits.items()
        } if m_cells else None
        res["ba"] = {name: balanced_accuracy(f["method"].margins(cells, f["params"])) for name, f in fits.items()}
        out[model] = res
    return out


# ------------------------------------------------------------------------ decide


def decide(frozen: FrozenSelection, holdout: Mapping[str, Any]) -> dict[str, Any]:
    models = frozen.selection["models"]
    per_model: dict[str, Any] = {}
    for model, sel in models.items():
        winner = sel["w_p"]["selection_winner"]
        c3 = holdout.get(model, {}).get("condition3_w_p")
        d3 = None if c3 is None else c3["diff_points"]
        choice = final_choice(BASELINE_W_P, winner, d3)
        reg = sel["regime_aware"]
        reg_c3 = holdout.get(model, {}).get("condition3_regime_aware")
        reg_cond = candidate_conditions(
            gain_points=reg["gain_points"], ci_low_points=reg["boot_diff_ci90_points"][0],
            ci_high_points=reg["boot_diff_ci90_points"][1],
            holdout_diff_points=None if reg_c3 is None else reg_c3["diff_points"],
        )
        sens = sel["lambda_sensitivity"]
        sens_c3 = holdout.get(model, {}).get("condition3_lambda10_w_p")
        sens_choice = final_choice(BASELINE_W_P, sens["selection_winner"],
                                   None if sens_c3 is None else sens_c3["diff_points"])
        per_model[model] = {
            "w_p": choice["value"],
            "w_p_replaced_baseline": choice["replaced_baseline"],
            "w_p_verdict": choice["verdict"],
            "selection_winner": winner,
            "condition3_diff_points": d3,
            "regime_aware_conditions": reg_cond,
            "lambda10_w_p": sens_choice["value"],
            "lambda10_verdict": sens_choice["verdict"],
            "lambda_sensitivity_changes_decision": sens_choice["value"] != choice["value"],
        }
    loro_min = {m: {c["value"]: c["loro_min"] for c in s["w_p"]["candidates"]} for m, s in models.items()}
    share = shared_w_p({m: v["w_p"] for m, v in per_model.items()}, loro_min)
    reg_passes = {m: bool(v["regime_aware_conditions"]["passes"]) for m, v in per_model.items()}
    adopted = regime_aware_adopted(reg_passes)
    return {
        "per_model": per_model,
        "shared_w_p": share,
        "regime_aware": {"passes_by_model": reg_passes, "adopted": adopted,
                         "verdict": ("regime-aware theta adopted (>= 2 of 3 models pass)" if adopted else
                                     "single theta stands: regime-aware not detected as better in >= 2 of 3 models")},
        "any_change_from_baseline": any(v["w_p_replaced_baseline"] for v in per_model.values()) or adopted,
    }


# --------------------------------------------------------------------- the driver


def load_groups(path: Path, models: Iterable[str]) -> dict[str, dict[str, str]]:
    """``regime_groups.json`` -> ``{model: {shape: group}}``."""
    doc = json.loads(Path(path).read_text())
    out: dict[str, dict[str, str]] = {}
    for m in models:
        groups = doc["groups"] if doc.get("consistent_across_models") else doc["groups_by_model"][m]
        out[m] = {shape: g for g, shapes in groups.items() for shape in shapes}
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(dataset_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    ds = Path(dataset_dir)
    return _read_csv(ds / "windows.csv"), _read_csv(ds / "cells.csv"), json.loads((ds / "manifest.json").read_text())


def slo_from_manifest(manifest: Mapping[str, Any]) -> dict[str, float]:
    by_col = manifest["label"]["slo_ms"]
    inv = {col: key for key, col in slo_labels.SLO_COLUMNS.items()}
    return {inv[col]: float(v) for col, v in by_col.items()}


def verify_signal_column(cells: Mapping[str, Cell], manifest: Mapping[str, Any], registry_path: Path) -> dict[str, Any]:
    """Recompute the dataset's own ``trs`` column with its registry's weights. Proves the
    re-signalling path is the path that wrote the column (only when the registry matches)."""
    used = manifest.get("registry_used_for_signal_columns", {}).get("sha256")
    ours = _sha256(registry_path)
    if used != ours:
        return {"checked": False, "reason": f"dataset trs used registry {used}, analysis uses {ours}"}
    worst, n = 0.0, 0
    for c in cells.values():
        p = c.params
        series = c.series(("trs", p.w_p, p.lambda_wait))
        for r, v in zip(c.rows, series):
            rec = r.get("trs")
            if rec in (None, ""):
                continue
            rec = float(rec)
            n += 1
            if math.isfinite(rec) and math.isfinite(v):
                worst = max(worst, abs(rec - v) / max(1e-9, abs(rec)))
    return {"checked": True, "windows": n, "max_rel_diff": worst, "ok": worst < 1e-6}


def analyse(
    dataset_dir: Path, groups_path: Path, *, policy: CellPolicy, registry_path: Path,
    n_boot: int = DEFAULT_BOOTSTRAP, seed: int = DEFAULT_SEED, processes: int = 1,
) -> dict[str, Any]:
    windows, cell_rows, manifest = load_dataset(dataset_dir)
    registry = load_registry(str(registry_path))
    models = sorted({r["model"] for r in windows})
    groups = load_groups(groups_path, models)
    starts = {(c["model"], c["shape"], c["primitive"], c["cell_id"], str(c["attempt"])): _f(c["start_ms"])
              for c in cell_rows if c.get("start_ms")}
    cells = build_cells(windows, slo_ms=slo_from_manifest(manifest), registry=registry, groups=groups,
                        cell_start_ms=starts, hold_warmup_s=policy.hold_warmup_s)
    # The split comes first: nothing below this line sees a hold-out window except
    # HoldoutSet.evaluate, which only opens for the frozen selection.
    training, holdout, counts = split_dataset(cells, policy)
    signal_check = verify_signal_column(cells, manifest, registry_path)
    frozen = select(training, n_boot=n_boot, seed=seed, processes=processes)
    holdout_result = holdout.evaluate(
        frozen, lambda by_model, fz: evaluate_holdout(by_model, fz, n_boot=n_boot, seed=seed)
    )
    decision = decide(frozen, holdout_result)
    return {
        "preregistration": PREREGISTRATION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "dataset_dir": str(dataset_dir),
            "dataset_sha256": {n: _sha256(Path(dataset_dir) / n) for n in ("windows.csv", "cells.csv", "manifest.json")},
            "regime_groups": str(groups_path), "regime_groups_sha256": _sha256(groups_path),
            "registry": str(registry_path), "registry_sha256": _sha256(registry_path),
        },
        "settings": {
            "policy": dataclasses.asdict(policy) | {k: sorted(v) for k, v in dataclasses.asdict(policy).items()
                                                    if isinstance(v, (set, frozenset))},
            "w_p_grid": list(W_P_GRID), "baseline_w_p": BASELINE_W_P, "main_lambda": MAIN_LAMBDA,
            "sensitivity_lambda": SENSITIVITY_LAMBDA, "min_gain_points": MIN_GAIN_POINTS,
            "ci_level": CI_LEVEL, "share_loss_points": SHARE_LOSS_POINTS,
            "regime_aware_min_models": REGIME_AWARE_MIN_MODELS, "decisive_points": DECISIVE_POINTS,
            "bootstrap": n_boot, "seed": seed, "theta_fit": THETA_CONFIG.as_dict(),
            "queue_fit": QUEUE_CONFIG.as_dict(), "slo_ms": slo_from_manifest(manifest),
            "regime_groups": groups,
        },
        "cell_counts": counts,
        "holdout_summary_before_opening": holdout.summary(),
        "signal_column_check": signal_check,
        "selection": frozen.selection,
        "holdout": holdout_result,
        "decision": decision,
    }


# ------------------------------------------------------------------------- report


def _pt(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{100.0 * x:.{digits}f}"


def _num(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{x:+.{digits}f}"


def _frac(x: float | None) -> str:
    return "—" if x is None else f"{x:.2f}"


def render_markdown(result: Mapping[str, Any]) -> str:
    dec = result["decision"]
    sel = result["selection"]["models"]
    hold = result["holdout"]
    lines = [
        "# Calibration analysis (pre-registered rules)",
        "",
        f"- Rules: `{result['preregistration']}`; policy `{result['settings']['policy']['name']}`; "
        f"bootstrap {result['settings']['bootstrap']} (seed {result['settings']['seed']}).",
        f"- Dataset: `{result['inputs']['dataset_dir']}`; cells train/hold-out/excluded = "
        f"{result['cell_counts']['train']}/{result['cell_counts']['holdout']}/{result['cell_counts']['excluded']}.",
        f"- Signal column check: {result['signal_column_check']}",
        "",
        "## Decision",
        "",
        "| model | w_p | verdict | λ=10 w_p | regime-aware passes |",
        "|---|---|---|---|---|",
    ]
    for m, v in dec["per_model"].items():
        lines.append(f"| {m} | {v['w_p']:g} | {v['w_p_verdict']} | {v['lambda10_w_p']:g} | "
                     f"{v['regime_aware_conditions']['passes']} |")
    sh = dec["shared_w_p"]
    lines += [
        "",
        f"- Shared w_p: adopt={sh.get('adopt')} value={sh.get('value')} worst loss "
        f"{_num(sh.get('worst_loss_points'), 2)} points (rule: < {SHARE_LOSS_POINTS:g}).",
        f"- Regime-aware θ(φ): {dec['regime_aware']['verdict']}.",
        "",
        "## w_p grid (λ = 3): LORO-min BA points and bootstrap",
        "",
        "Gain and the 90 % interval are candidate − baseline (w_p = 0), in BA points.",
        "",
    ]
    for m, s in sel.items():
        tr = s["training"]
        lines += [f"### {m}", "",
                  f"Training: {tr['cells']} cells, {tr['windows']} windows, {tr['violating_windows']} violating "
                  f"({tr['independent_violating_windows']} non-overlapping; {tr['cells_with_violation']} cells "
                  "with any violation). Groups (windows/violating/non-overlapping violating): " +
                  "; ".join(f"{g} {v['shapes']} {v['windows']}/{v['violating_windows']}/"
                            f"{v['independent_violating_windows']}"
                            for g, v in tr["by_group"].items()),
                  "",
                  "| w_p | LORO-min | folds | gain | 90% CI | P(gain≥3) | (1) | (2) |",
                  "|---|---|---|---|---|---|---|---|"]
        for c in s["w_p"]["candidates"]:
            cond = c["conditions"] or {}
            ci = c["boot_diff_ci90_points"]
            folds = ", ".join(f"{g}:{_pt(b)}" for g, b in c["loro_folds"].items())
            lines.append(
                f"| {c['value']:g} | {_pt(c['loro_min'])} | {folds} | {_num(c['gain_points'])} | "
                f"[{_num(ci[0])}, {_num(ci[1])}] | "
                f"{_frac(c['boot_frac_gain_ge_3'])} | "
                f"{cond.get('loro_gain_ge_3', 'base')} | {cond.get('ci90_excludes_0', 'base')} |")
        lines += ["", f"Selection winner (conditions 1+2, training only): {s['w_p']['selection_winner']}", ""]
        h = hold.get(m, {})
        c3 = h.get("condition3_w_p")
        if c3:
            lines.append(f"Condition 3 (hold-out, selected − baseline): {_num(c3['diff_points'])} points "
                         f"[{_num(c3['ci90_points'][0])}, {_num(c3['ci90_points'][1])}]")
        r = s["regime_aware"]
        lines += [
            f"Regime-aware (w_p={r['w_p']:g}) LORO-min {_pt(r['regime_loro_min'])} vs single "
            f"{_pt(r['single_loro_min'])}: gain {_num(r['gain_points'])}, 90% CI "
            f"[{_num(r['boot_diff_ci90_points'][0])}, {_num(r['boot_diff_ci90_points'][1])}]",
            "",
        ]
        sens = s["lambda_sensitivity"]
        lines.append("λ = 10 LORO-min by w_p: " + ", ".join(
            f"{c['value']:g}:{_pt(c['loro_min'])}" for c in sens["candidates"]) +
            f"; selection winner {sens['selection_winner']}")
        ctl = s["controls_loro"]
        lines += [f"Controls LORO-min: q/replica {_pt(ctl['queue_per_replica']['min'])}, "
                  f"two-signal {_pt(ctl['two_signal']['min'])}", ""]
    lines += ["## Merged hold-out set (paired, cell-clustered bootstrap)", ""]
    for m, h in hold.items():
        lines.append(f"- {m}: {h['cells']} cells, {h['windows']} windows, {h['violating_windows']} violating "
                     f"({h['independent_violating_windows']} non-overlapping, {h['cells_with_violation']} cells); "
                     + "; ".join(f"{src} {v['cells']} cells {v['violating_windows']}/{v['independent_violating_windows']}"
                                 for src, v in h["by_source"].items()))
    lines += ["",
              "| model | windows (viol) | cells w/ viol | comparison | A BA | B BA | A−B | 90% CI | discordant | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for m, h in hold.items():
        for key, label in (("condition3_w_p", "selected vs baseline"),
                           ("condition3_regime_aware", "regime-aware vs single"),
                           ("trs_vs_queue_per_replica", "TRS vs q/replica"),
                           ("trs_vs_two_signal", "TRS vs two-signal"),
                           ("lambda10_vs_lambda3_baseline", "λ10 vs λ3 (w_p=0)")):
            p = h.get(key)
            if not p:
                continue
            dsum = sum(p["discordant_windows"].values())
            lines.append(
                f"| {m} | {p['windows']} ({p['violating_windows']}) | {p['cells_with_violation']} | {label} | "
                f"{_pt(p['ba_a'])} | {_pt(p['ba_b'])} | {_num(p['diff_points'])} | "
                f"[{_num(p['ci90_points'][0])}, {_num(p['ci90_points'][1])}] | {dsum} in {p['discordant_cells']} cells | "
                f"{p['verdict']} |")
    lines += ["", "M-only BA (secondary, not used for any verdict):", ""]
    for m, h in hold.items():
        if h.get("m_only_ba"):
            lines.append(f"- {m}: " + ", ".join(f"{k} {_pt(v)}" for k, v in h["m_only_ba"].items()))
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--regime-groups", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--profile", choices=sorted(POLICIES), default=PREREGISTERED.name)
    ap.add_argument("--train-hold-stages", default=None,
                    help="comma list overriding the policy's training hold stages")
    ap.add_argument("--hold-warmup-s", type=float, default=None)
    ap.add_argument("--registry", type=Path, default=Path(__file__).resolve().parents[2] / "registry.yaml")
    ap.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--processes", type=int, default=max(1, (multiprocessing.cpu_count() or 2) // 2))
    args = ap.parse_args(argv)
    policy = POLICIES[args.profile]
    if args.train_hold_stages is not None:
        stages = frozenset(s.strip() for s in args.train_hold_stages.split(",") if s.strip())
        policy = dataclasses.replace(policy, train_hold_stages=stages,
                                     excluded_hold_stages=policy.excluded_hold_stages - stages)
    if args.hold_warmup_s is not None:
        policy = dataclasses.replace(policy, hold_warmup_s=args.hold_warmup_s)
    result = analyse(args.dataset_dir, args.regime_groups, policy=policy, registry_path=args.registry,
                     n_boot=args.bootstrap, seed=args.seed, processes=args.processes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "decision.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    (args.out_dir / "decision.md").write_text(render_markdown(result))
    print(f"wrote {args.out_dir / 'decision.json'} and decision.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

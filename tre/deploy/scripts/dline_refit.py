#!/usr/bin/env python3
"""The D-line offline refit (plan 2026-09-21 §6.11 step 4): alpha, w_p, final.

Until 2026-09-23 this pipeline existed only as ``/tmp/refit_final_1790081928/refit.py``
(archived under ``archive_tmp_20260922/refit_final_1790081928``): every number the D-line
decisions rest on - the chosen EMA alpha, the D3 w_p, theta / delta and the M hold-out
acceptance, for the primary (D6') and the fixed label - came out of a script outside the
repository with hard-coded paths. This module is that script, formalised: same stages,
same rules, same seeds; paths and the per-model inputs are arguments.

Inputs are the re-windowed CSVs of a fit plan (``rewindow_from_raw --window-align grid
--step-ms 10000``, i.e. ``calibration_campaign.fit_plan``'s ``rewindow`` step) in one
directory, ``--fit-dir``: ``<model>_fitting.csv``, ``<model>_validation.csv`` (the
held-out set), ``<model>_fitting_decode_heavy.csv``, ``<model>_fitting_prefill_heavy.csv``.
Outputs go to ``--out-dir/<model>/<arm>/<stage>.json``; each stage reads the previous
one's output from there.

Stages (``python -m scripts.dline_refit STAGE --model M --arm primary|fixed|k3 ...``):

``trainset`` (D16)
    cuts the training set out of standard datasets (``scripts.calibration_dataset``):
    theta is fitted on constant-load cells only - a row's own ``primitive`` / ``role`` /
    ``split`` columns decide, never its cell id. Writes the fitting and family CSVs into
    ``--fit-dir`` (no validation CSV), ``training_cells.jsonl``, ``trainset.json`` and
    ``h2.json``: H2, the disclosure set (every dynamic cell, plus the sealed split of the
    ``--h2-dataset`` runs), listed and hashed but never parsed into windows. The sealed
    split of a ``--dataset`` is M and is skipped unread. Sentinels train unless
    ``--no-sentinels``. Every later stage refuses a fitting / family CSV holding a row that
    is not a constant-load training row (:func:`check_training_inputs`), whoever built it.
``alpha`` (D4', published per D18)
    the TSS EMA time constant. ``--alpha-rule d4prime`` (default) is
    :mod:`scripts.alpha_fit` - same-window LOSO balanced accuracy of the deployed
    classifier (tau-EMA + dwell 2), healthy false alarm <= 5 %, within 1 SE the fewest
    spurious CRITICAL episodes per hour on steady healthy cells, then the larger alpha.
    ``--alpha-rule refit0922`` is the archived stage the 2026-09-22 numbers came from: LOSO
    BA of dwell-2 CRITICAL at t against the label at t + 30 s, FA <= 5 %, the most
    responsive tau within 1 SE of the best. D4' replaced it because a t + 30 s label
    mechanically favours tau = 0 (TSS has no lead, plan §6.9c E-B); it is kept to
    reproduce the archive. Both run at the alpha-stage w_p (``--alpha-w-p``; default
    :data:`ALPHA_STAGE_W_P`, the D3 values current on 2026-09-22) and lambda_wait = 1.
    D18: the rule's pick is kept as ``chosen_tau_s`` and disclosed, but the published tau
    - the one the later stages fit at and the registry deploys - is ``--publish-tau-s``
    (default :data:`PUBLISH_TAU_S` = 10 s, alpha = .63, common to the three models;
    ``rule`` publishes the rule's pick). ``disclosure`` carries the BA / false-alarm / 90 %
    step-delay curves, the 1-SE set, the flat interval around the published tau, the
    flatness over :data:`DISCLOSURE_TAU_RANGE_S` and the bootstrap selection frequencies.
``wp`` (D3 as revised by D17)
    over :data:`WP_GRID` at the published tau and lambda_wait = 1, the verdict
    (``theta_verdict.verdict_report``) of every w_p; a w_p is admissible when
    (c1) its training BA is within one SE of the w_p = 0 BA (cell bootstrap of the BA at
    the w_p = 0 theta) and (c2) the family gap ``|theta_P - theta_D| / theta`` is within the
    merged fit's CI half width fraction. (c3) - the family rule publishes the merged theta -
    is still reported per w_p but no longer selects (D17: it restated D5). w_p* is the
    LARGEST admissible w_p (:func:`d3_select`), 0 when none is. Then the
    lambda check: lambda_wait in :data:`LAMBDAS` at w_p*; lambda moves off 1 only if the
    best BA beats lambda = 1 by >= 0.02.
    ``--lambda-method v1`` (user 2026-09-24) replaces both rules: lambda_wait AND w_p are
    v1's selection (:mod:`scripts.v1_lambda_fit` - v1's rank-correlation objective, lambda
    1..4 / 0.25, w_p 0.01..0.08 / 0.005, the joint refinement), ported onto the same
    training windows; the D17 w_p rule is not applied (when the two disagree, v1's joint
    refinement wins and wp.json says so). tau, theta, delta and the labels stay v2.
``final`` (D5 + hold-out)
    the verdict at (tau, w_p*, lambda*) with 1000 / 200 resamples; D5: the merged theta is
    published whatever the family rule says (the family theta is kept as diagnostic);
    then the hold-out report on the validation CSV (``theta_verdict.holdout_report``,
    dwell 2), the M balanced-accuracy CI (cell bootstrap) and the per-prompt-length
    attainment of the TTFT SLO on non-overlapping 30 s tiles. ``--no-holdout`` stops
    after the verdict and never opens the validation CSV: M is evaluated exactly once,
    after it is frozen and hashed (plan §6.11 note 9), so every refit before that runs
    without it.
``summary``
    the table of every model and arm under ``--out-dir`` plus the boundary-band window
    counts per shape / family and what hold cells a family short of
    ``MIN_FAMILY_WINDOWS`` band windows would need (``summary.json``).
``freeze`` (D22: refit -> freeze -> collect M -> A-D once)
    ``--model`` repeated, ``--freeze-file PATH``: one JSON of every model's published
    parameters (theta, w_p, tau / alpha, lambda_wait, delta_crit / delta_high, the registry
    EMA fields), the D13 stop rule, what ``holdout_report`` needs (``verdict_for_holdout``),
    sha256 of the stage outputs and of every training input, the H2 hashes and the code
    commit; self-hashed (``freeze_sha256``, :func:`canonical_sha256`), a sha256sum sidecar
    ``PATH.sha256``, mode 0444. Refuses - listing every reason, writing nothing - unless
    each model's final ran with ``--no-holdout``, meets the stop rule and still sits on
    the training inputs it was fitted on, and the freeze file is new.
``verify-freeze``
    checks the sidecar and the self hash (:func:`verify_freeze`); exit 0 = intact.
``accept`` (plan §6.9f A-D, once)
    ``--freeze-file``, ``--dataset [RUN=]DIR`` (repeatable) and one ``--m-manifest`` per
    frozen model; reads nothing but those. Refuses on a broken freeze, a training input
    changed since the freeze, an M manifest sealed under another freeze or label or whose
    raw-data sums moved, a manifest cell missing / duplicated / not ``holdout`` / not
    ``valid`` in the datasets. Scores the manifest cells with ``holdout_report`` (dwell 2)
    plus a cell bootstrap for the CIs; writes ``<stem>.accept.json`` (0444), the
    validation CSVs under ``<stem>.accept.d/`` and the marker ``PATH.accepted``. Exit 0 =
    A, B and D pass for every model, 3 = evaluated and failed, other = refused. Runs once;
    ``--recheck`` recomputes in a temp dir and compares, writing nothing.

Labels are ``tre_common.slo_labels``: ``primary`` is the D6' slowdown label of the
registry profile (``max(500 ms, 5 * idle TTFT(L))``, TPOT 75 ms, >= 20 completions),
``fixed`` the 500 / 75 ms comparison, ``k3`` the k = 3 / 150 ms ablation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from tre_common import slo_labels

#: The archived stage's grid and constants (refit.py, 2026-09-22), unchanged.
TAUS_S: tuple[float, ...] = (0, 5, 10, 15, 20, 30, 40, 60)
DT_REF_S = 10.0
WP_GRID: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2)
LAMBDAS: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)
LAMBDA_WAIT = 1.0
#: lambda moves off LAMBDA_WAIT only when the best lambda beats it by this much BA.
LAMBDA_MIN_GAIN = 0.02
#: w_p the alpha stage runs at: the D3 values current on 2026-09-22 (plan 6.11 D3,
#: fable_acc_wp/wp_rule.log). ``--alpha-w-p`` overrides.
ALPHA_STAGE_W_P: dict[str, float] = {"dsqwen-7b": 0.01, "dsllama-8b": 0.005, "dsqwen-14b": 0.005}
HORIZON_MS = 30_000
FA_MAX = 0.05
SEED = 20260922
DWELL_WINDOWS = 2
TRIM_RAMP_WINDOWS = 1
#: Verdict resamples: the w_p grid, the lambda check, the final verdict.
WP_RESAMPLES = (1000, 200)
LAMBDA_RESAMPLES = (300, 100)
FINAL_RESAMPLES = (1000, 200)
BA_SE_RESAMPLES = 300
M_CI_RESAMPLES = 1000
#: summary: band windows a family needs, and the fallback yield of one hold cell.
MIN_FAMILY_WINDOWS = 30

ARMS = ("primary", "fixed", "k3")
#: How the wp stage picks lambda_wait (and, for v1, w_p): ``v2`` = D17 + the BA lambda
#: check (the default, what the D22 freeze used); ``v1`` = v1's selection (user 2026-09-24).
LAMBDA_METHODS = ("v2", "v1")
ALPHA_RULES = ("d4prime", "refit0922")
FAMILY_FILES = ("decode_heavy", "prefill_heavy")
#: D18: the tau every model publishes (= the refresh period DT_REF_S, alpha = .63). The
#: D4' sweep still runs, for the disclosure only. ``--publish-tau-s rule`` publishes the
#: rule's own pick instead (how the pre-D18 numbers were made).
PUBLISH_TAU_S = 10.0
#: D18: the tau range whose flatness (BA / false alarm / 90 % step delay) is disclosed.
DISCLOSURE_TAU_RANGE_S = (0.0, 15.0)


# ------------------------------------------------------------------------ inputs


def paths(fit_dir: Path, model: str) -> dict[str, Any]:
    f = Path(fit_dir)
    return {
        "fitting": f / f"{model}_fitting.csv",
        "validation": f / f"{model}_validation.csv",
        "families": {fam: f / f"{model}_fitting_{fam}.csv" for fam in FAMILY_FILES},
    }


# ------------------------------------------------------------ training set (D16)
#
# D16 (plan §6.11, 2026-09-23): theta is fitted on constant-load cells only. Every other
# cell of the calibration runs is sealed away before anything is fitted:
#
# * the training set - the constant-load cells (a dataset row's ``primitive`` in
#   alpha_fit.STEADY_PRIMITIVES, ``role`` not in alpha_fit.UNSTEADY_ROLES) of the train /
#   auxiliary splits; sentinels are constant-load and train unless ``--no-sentinels``;
#   D21: so do the boundary supplement's smoke holds (role smoke, stage dwell, split
#   auxiliary, primitive hold) - by their role, calibration_design.TRAINING_HOLD_ROLES,
#   the key calibration_decision's cell policy trains them by too (the stage is not read);
# * H2, the disclosure set - the dynamic cells (steps / ramp / bursts) and the sealed split
#   (``split == holdout``: the held-out shape and the ramps) of the runs given with
#   ``--h2-dataset``. It is reported after M, never read by a selection: its rows are
#   hashed and listed (h2.json), never parsed into windows;
# * M, the acceptance set - the sealed split of a ``--dataset``: skipped unread, counted.
#
# The three are disjoint by construction (one row, one set). The fit stages then refuse a
# fitting / family CSV holding any row that is not a constant-load training row.

TRAINSET_MANIFEST = "trainset.json"
H2_MANIFEST = "h2.json"
#: Ledger-format list of the training cells (``cells.jsonl`` lines): the steady-cell
#: source of alpha_fit and of the summary when no ``--ledger`` is given.
TRAINING_LEDGER = "training_cells.jsonl"
DATASET_WINDOWS = "windows.csv"
DATASET_MANIFEST = "manifest.json"
SPLIT_HOLDOUT = "holdout"
TRAINING_SPLITS = frozenset({"train", "auxiliary"})
ROLE_SENTINEL = "sentinel"
SET_TRAINING, SET_H2, SET_M, SET_SENTINEL_OFF = "training", "h2", "m", "sentinel_excluded"
KIND_CONSTANT, KIND_DYNAMIC, KIND_SEALED, KIND_UNKNOWN = "constant_load", "dynamic", "sealed", "unknown"
#: Identity columns a dataset row must carry for the training set to be cut from it.
REQUIRED_DATASET_COLUMNS = ("model", "shape", "primitive", "stage", "split", "role",
                            "cell_id", "attempt", "scenario_id")


class TrainingSetError(ValueError):
    """A dataset row the D16 cut cannot place (never guessed)."""


def cell_kind(primitive: str, role: str, split: str, shape: str = "") -> str:
    """D16: ``constant_load`` / ``dynamic`` / ``sealed`` from a row's own fields.

    ``sealed`` - the held-out split, or the held-out shape wherever it appears;
    ``constant_load`` - a steady primitive (hold / static) that is not a ramp role; D21:
    explicitly, a steady hold whose role is in ``calibration_design.TRAINING_HOLD_ROLES``
    (the smoke hold: stage dwell, split auxiliary - neither is read here);
    ``dynamic`` - anything else with a primitive (steps / ramp / bursts);
    ``unknown`` - no primitive: nothing says what the cell was, and a cell id is not read."""
    from scripts import alpha_fit
    from scripts import calibration_design as design
    from scripts import gen_calibration_schedules as gen

    if split == SPLIT_HOLDOUT or (shape and gen.is_held_out(shape)):
        return KIND_SEALED
    if not primitive:
        return KIND_UNKNOWN
    if primitive in alpha_fit.STEADY_PRIMITIVES and role in design.TRAINING_HOLD_ROLES:
        return KIND_CONSTANT  # D21: the smoke hold trains (same key as calibration_decision)
    if primitive in alpha_fit.STEADY_PRIMITIVES and role not in alpha_fit.UNSTEADY_ROLES:
        return KIND_CONSTANT
    return KIND_DYNAMIC


def assign_set(row: Mapping[str, str], *, sealed_to_h2: bool, sentinels: bool) -> str:
    """The set one standard-dataset row belongs to (training / h2 / m / sentinel_excluded).

    Read from the row's own ``split`` / ``shape`` / ``primitive`` / ``role`` (never its
    stage or cell id): a smoke hold (role smoke, stage dwell, split auxiliary) trains - D21,
    :func:`cell_kind`."""
    from scripts import gen_calibration_schedules as gen

    split, shape = row.get("split") or "", row.get("shape") or ""
    where = f"{row.get('model')}/{row.get('cell_id')} a{row.get('attempt')}"
    if not split:
        raise TrainingSetError(f"{where}: no split")
    if gen.is_held_out(shape) and split != SPLIT_HOLDOUT:
        raise TrainingSetError(f"{where}: held-out shape {shape} outside the {SPLIT_HOLDOUT} split")
    if split != SPLIT_HOLDOUT and split not in TRAINING_SPLITS:
        raise TrainingSetError(f"{where}: unknown split {split!r}")
    kind = cell_kind(row.get("primitive") or "", row.get("role") or "", split, shape)
    if kind == KIND_SEALED:
        return SET_H2 if sealed_to_h2 else SET_M
    if kind == KIND_UNKNOWN:
        raise TrainingSetError(f"{where}: no primitive - a cell is not classified by its id")
    if kind == KIND_DYNAMIC:
        return SET_H2
    if not sentinels and (row.get("role") or "") == ROLE_SENTINEL:
        return SET_SENTINEL_OFF
    return SET_TRAINING


@dataclass(frozen=True)
class DatasetSource:
    """One standard dataset (``scripts.calibration_dataset``) the training set is cut from.

    ``sealed_to_h2``: its sealed split joins H2 (D16: run 1 and run 2, whose held-out cells
    are disclosure-only); otherwise the sealed split is M and is skipped unread."""

    name: str
    directory: Path
    sealed_to_h2: bool

    @property
    def windows(self) -> Path:
        return self.directory / DATASET_WINDOWS

    @classmethod
    def parse(cls, text: str, *, sealed_to_h2: bool) -> "DatasetSource":
        name, sep, path = text.partition("=")
        if not sep:
            name, path = "", text
        d = Path(path)
        if not (d / DATASET_WINDOWS).exists() and (d / "dataset" / DATASET_WINDOWS).exists():
            d = d / "dataset"
        if not (d / DATASET_WINDOWS).exists():
            raise TrainingSetError(f"{path}: no {DATASET_WINDOWS} (a standard dataset directory)")
        if not name:
            name = d.parent.name if d.name == "dataset" else d.name
        return cls(name=name, directory=d, sealed_to_h2=sealed_to_h2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class SealedRows:
    """H2 as the training-set cut sees it: a digest and a cell list, no window accessor.

    Rows are fed in as raw CSV values and only hashed and counted (the
    ``calibration_decision.HoldoutSet`` pattern): nothing here can hand a row to a fit."""

    def __init__(self) -> None:
        self.__digests: dict[str, Any] = {}
        self.__cells: dict[tuple, dict[str, Any]] = {}

    def add(self, source: str, header: Sequence[str], values: Sequence[str], why: str) -> None:
        h = self.__digests.get(source)
        if h is None:
            h = self.__digests[source] = hashlib.sha256()
            h.update((json.dumps(list(header)) + "\n").encode())
        h.update((json.dumps(list(values), separators=(",", ":")) + "\n").encode())
        row = dict(zip(header, values))
        key = (source, row["model"], row["cell_id"], row["attempt"], row["scenario_id"])
        cell = self.__cells.get(key)
        if cell is None:
            cell = self.__cells[key] = {
                "run": source, "model": row["model"], "cell_id": row["cell_id"],
                "attempt": row["attempt"], "scenario_id": row["scenario_id"], "shape": row["shape"],
                "primitive": row["primitive"], "stage": row["stage"], "role": row["role"],
                "split": row["split"], "why": why, "windows": 0,
            }
        cell["windows"] += 1

    def manifest(self) -> dict[str, Any]:
        cells = [self.__cells[k] for k in sorted(self.__cells)]
        per_source = {s: h.hexdigest() for s, h in sorted(self.__digests.items())}
        overall = hashlib.sha256("".join(f"{s}:{d}\n" for s, d in per_source.items()).encode()).hexdigest()
        counts: dict[str, Any] = {}
        for c in cells:
            m = counts.setdefault(c["model"], {"cells": 0, "windows": 0, "by_run_why_primitive": {}})
            m["cells"] += 1
            m["windows"] += c["windows"]
            k = f"{c['run']}|{c['why']}|{c['primitive']}"
            m["by_run_why_primitive"][k] = m["by_run_why_primitive"].get(k, 0) + c["windows"]
        return {
            "rows_sha256": overall, "rows_sha256_by_run": per_source,
            "rows_hash_rule": ("per run: sha256 of the JSON header line, then one compact JSON "
                               "array of the raw CSV values per H2 row, in file order; overall: "
                               "sha256 of the sorted '<run>:<sha256>' lines"),
            "cells_sha256": hashlib.sha256(json.dumps(cells, sort_keys=True).encode()).hexdigest(),
            "counts": counts, "cells": cells,
        }


def _dataset_header(src: DatasetSource) -> list[str]:
    with open(src.windows, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh), None)
    missing = [c for c in REQUIRED_DATASET_COLUMNS if c not in (header or [])]
    if missing:
        raise TrainingSetError(f"{src.windows}: missing columns {missing}")
    return list(header)


def build_training_set(sources: Sequence[DatasetSource], fit_dir: Path, *,
                       sentinels: bool = True, models: Optional[Sequence[str]] = None) -> dict:
    """D16: cut the training set (constant-load cells) out of standard datasets.

    Writes, into ``fit_dir``, what the fit stages read - ``<model>_fitting.csv`` and one
    ``<model>_fitting_<family>.csv`` per family, rows in dataset order (the loaders EMA a
    cell in CSV order), a ``run`` column first - plus :data:`TRAINING_LEDGER`,
    :data:`H2_MANIFEST` (the H2 cell list and row hash) and :data:`TRAINSET_MANIFEST`. No
    ``<model>_validation.csv`` is written. H2 covers every model of the sources whatever
    ``models`` restricts, so its hash does not depend on it."""
    from scripts import gen_calibration_schedules as gen

    fit_dir = Path(fit_dir)
    if len({s.name for s in sources}) != len(sources):
        raise TrainingSetError(f"two datasets share a run name: {[s.name for s in sources]}")
    families = gen.families()
    if set(families) != set(FAMILY_FILES):
        raise TrainingSetError(f"families {sorted(families)} != {sorted(FAMILY_FILES)}")
    family_of = {shape: fam for fam, shapes in families.items() for shape in shapes}
    headers = {s.name: _dataset_header(s) for s in sources}
    out_header = ["run"]
    for h in headers.values():
        out_header += [c for c in h if c not in out_header]
    fit_dir.mkdir(parents=True, exist_ok=True)
    stale = sorted(p.name for p in fit_dir.glob("*_validation.csv"))
    if stale:
        raise TrainingSetError(f"{fit_dir} holds {stale}: a training-set directory carries no M")

    h2 = SealedRows()
    handles: dict[Path, Any] = {}
    writers: dict[str, dict[str, Any]] = {}
    counts: dict[str, dict[str, Any]] = {}
    m_counts: dict[str, dict[str, Any]] = {}
    other: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    owner: dict[tuple[str, str], str] = {}
    ledger_lines: dict[tuple, dict[str, Any]] = {}

    def writer_for(model: str) -> dict[str, Any]:
        if model not in writers:
            ws = {}
            for key, path in [("fitting", fit_dir / f"{model}_fitting.csv"),
                              *[(fam, fit_dir / f"{model}_fitting_{fam}.csv") for fam in FAMILY_FILES]]:
                fh = handles[path] = open(path, "w", newline="", encoding="utf-8")
                w = csv.writer(fh, lineterminator="\n")
                w.writerow(out_header)
                ws[key] = w
            writers[model] = ws
        return writers[model]

    try:
        for src in sources:
            header = headers[src.name]
            with open(src.windows, newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                next(reader)
                for values in reader:
                    row = dict(zip(header, values))
                    target = assign_set(row, sealed_to_h2=src.sealed_to_h2, sentinels=sentinels)
                    model = row["model"]
                    if target == SET_M:
                        # M: the split column decided it; nothing else of the row is used.
                        mc = m_counts.setdefault(src.name, {}).setdefault(model, {"rows": 0, "cells": set()})
                        mc["rows"] += 1
                        mc["cells"].add((row["cell_id"], row["attempt"]))
                        continue
                    if target == SET_H2:
                        why = "sealed" if row["split"] == SPLIT_HOLDOUT else "dynamic"
                        h2.add(src.name, header, values, why)
                        continue
                    if target == SET_SENTINEL_OFF:
                        other[model]["sentinel_windows_excluded"] += 1
                        continue
                    if models and model not in models:
                        continue
                    sid = row["scenario_id"]
                    prev = owner.setdefault((model, sid), src.name)
                    if prev != src.name:
                        raise TrainingSetError(f"{model} {sid}: in both {prev} and {src.name}")
                    out = [src.name if c == "run" else row.get(c, "") for c in out_header]
                    ws = writer_for(model)
                    ws["fitting"].writerow(out)
                    fam = family_of.get(row["shape"])
                    if fam is not None:
                        ws[fam].writerow(out)
                    c = counts.setdefault(model, {"fitting": 0, "cells": set(), "by_run_role": {},
                                                  "by_cell_status": {}, **{f: 0 for f in FAMILY_FILES}})
                    c["fitting"] += 1
                    c["cells"].add((src.name, sid))
                    rk = f"{src.name}|{row['role'] or row['stage']}"
                    c["by_run_role"][rk] = c["by_run_role"].get(rk, 0) + 1
                    st = row.get("cell_status") or ""
                    c["by_cell_status"][st] = c["by_cell_status"].get(st, 0) + 1
                    if fam is not None:
                        c[fam] += 1
                    ledger_lines.setdefault((model, sid), {
                        "cell_id": sid, "attempt": int(row["attempt"] or 1), "model": model,
                        "run": src.name, "role": row["role"], "primitive": row["primitive"],
                        "stage": row["stage"], "split": row["split"], "shape": row["shape"],
                    })
    finally:
        for fh in handles.values():
            fh.close()

    with open(fit_dir / TRAINING_LEDGER, "w", encoding="utf-8") as fh:
        for key in sorted(ledger_lines):
            fh.write(json.dumps(ledger_lines[key], sort_keys=True) + "\n")
    h2_doc = {
        "what": ("H2, the D16 disclosure set: every dynamic cell and the sealed split of the "
                 "--h2-dataset runs. Reported after M; no selection (theta / w_p / alpha / "
                 "delta) reads it. Recompute rows_sha256 from the sources to show it is unchanged."),
        "sources": [{"run": s.name, "windows_csv": str(s.windows), "sealed_to_h2": s.sealed_to_h2}
                    for s in sources],
        **h2.manifest(),
    }
    (fit_dir / H2_MANIFEST).write_text(json.dumps(h2_doc, indent=1), encoding="utf-8")

    model_docs: dict[str, Any] = {}
    for model, c in sorted(counts.items()):
        files = {"fitting": fit_dir / f"{model}_fitting.csv",
                 **{f"family_{fam}": fit_dir / f"{model}_fitting_{fam}.csv" for fam in FAMILY_FILES}}
        model_docs[model] = {
            "windows": c["fitting"], "cells": len(c["cells"]),
            "by_run_role": dict(sorted(c["by_run_role"].items())),
            "by_cell_status": dict(sorted(c["by_cell_status"].items())),
            "family_windows": {fam: c[fam] for fam in FAMILY_FILES},
            **({"excluded": dict(other[model])} if model in other else {}),
            "files": {k: {"path": str(p), "sha256": sha256_file(p)} for k, p in files.items()},
        }
    source_docs = []
    for s in sources:
        man = s.directory / DATASET_MANIFEST
        rev = None
        if man.exists():
            try:
                rev = json.loads(man.read_text(encoding="utf-8")).get("format_revision")
            except ValueError:
                rev = None
        source_docs.append({
            "run": s.name, "directory": str(s.directory), "sealed_split": SET_H2 if s.sealed_to_h2 else SET_M,
            "windows_csv_sha256": sha256_file(s.windows),
            "manifest_sha256": sha256_file(man) if man.exists() else None, "format_revision": rev,
        })
    doc = {
        "what": "D16 training set: the constant-load cells of the sources (theta is fitted on these only)",
        "rules": {
            "training": ("split in {train, auxiliary}, primitive in alpha_fit.STEADY_PRIMITIVES, role "
                         "not in alpha_fit.UNSTEADY_ROLES - read from each row's own dataset columns"),
            "sentinels": "role sentinel trains" if sentinels else "role sentinel excluded (--no-sentinels)",
            "h2": "dynamic cells of every source + the sealed split (split holdout) of --h2-dataset sources",
            "m": "the sealed split of --dataset sources: skipped unread (counts only)",
            "families": {fam: list(shapes) for fam, shapes in families.items()},
        },
        "sentinels": sentinels,
        "sources": source_docs,
        "models": model_docs,
        "h2": {"manifest": str(fit_dir / H2_MANIFEST), "manifest_sha256": sha256_file(fit_dir / H2_MANIFEST),
               "rows_sha256": h2_doc["rows_sha256"], "cells_sha256": h2_doc["cells_sha256"],
               "counts": {m: {k: v for k, v in c.items() if k != "by_run_why_primitive"}
                          for m, c in h2_doc["counts"].items()}},
        "m_unread": {run: {m: {"rows": v["rows"], "cells": len(v["cells"])} for m, v in sorted(ms.items())}
                     for run, ms in sorted(m_counts.items())},
        "training_ledger": str(fit_dir / TRAINING_LEDGER),
    }
    (fit_dir / TRAINSET_MANIFEST).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def scan_training_rows(path: Path, ledger: Optional[Mapping[str, Mapping[str, Any]]] = None) -> dict[str, int]:
    """Kinds of the rows of a fitting / family CSV, from the rows' own ``primitive`` /
    ``role`` / ``split`` / ``shape`` columns, else from the ledger line of the row's cell."""
    from scripts import alpha_fit

    out = {KIND_CONSTANT: 0, KIND_DYNAMIC: 0, KIND_SEALED: 0, KIND_UNKNOWN: 0}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            primitive, role, split = row.get("primitive") or "", row.get("role") or "", row.get("split") or ""
            if not primitive and ledger:
                entry = ledger.get(alpha_fit.base_cell(row.get("scenario_id") or "")) or {}
                primitive = str(entry.get("primitive") or "")
                role = role or str(entry.get("role") or "")
                split = split or str(entry.get("split") or "")
            out[cell_kind(primitive, role, split, row.get("shape") or "")] += 1
    return out


def check_training_inputs(model: str, p: Mapping[str, Any], *,
                          ledger: Optional[Mapping[str, Mapping[str, Any]]] = None) -> dict:
    """D16 at every fit stage's entry: the fitting and family CSVs hold constant-load
    training rows only - no dynamic cell, no sealed (H2 / M) row, no row nothing
    classifies. When the directory was built by the ``trainset`` stage its files must also
    still be the ones it wrote. Returns the provenance the stage records."""
    files = {"fitting": Path(p["fitting"]),
             **{f"family_{k}": Path(v) for k, v in p["families"].items()}}
    fit_dir = files["fitting"].parent
    man_path = fit_dir / TRAINSET_MANIFEST
    prov: dict[str, Any] = {"trainset_manifest": None}
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        entry = (man.get("models") or {}).get(model)
        if entry is None:
            raise SystemExit(f"{man_path}: no training set for {model}")
        for key, path in files.items():
            want = (entry["files"].get(key) or {}).get("sha256")
            if want is not None and path.exists() and sha256_file(path) != want:
                raise SystemExit(f"{path} is not the file the trainset stage wrote ({man_path})")
        prov = {"trainset_manifest": str(man_path), "trainset_manifest_sha256": sha256_file(man_path),
                "sentinels": man.get("sentinels"), "h2_rows_sha256": man["h2"]["rows_sha256"],
                "h2_cells_sha256": man["h2"]["cells_sha256"]}
    kinds = {}
    for key, path in files.items():
        if not path.exists():
            continue
        k = kinds[key] = scan_training_rows(path, ledger)
        bad = {kind: n for kind, n in k.items() if kind != KIND_CONSTANT and n}
        if bad:
            raise SystemExit(
                f"{path}: D16 - theta is fitted on constant-load cells only, and this file holds "
                f"{bad} window rows that are not (unknown = no primitive column and no ledger line). "
                "Build the training set with `dline_refit trainset`.")
    prov["rows"] = kinds
    return prov


def label_for(model: str, arm: str, registry: Optional[str] = None) -> slo_labels.LabelDefinition:
    """The label of one arm: the registry profile's D6' primary, or an arm of it."""
    primary = slo_labels.label_def_for_model(
        model, ttft_p95_ms=500.0, tpot_p95_ms=75.0, mode=None, registry=registry)
    if arm == "primary":
        return primary
    arms = slo_labels.label_arms(primary)
    return {"fixed": arms[slo_labels.ARM_FIXED], "k3": arms[slo_labels.ARM_K3]}[arm]


def shape_fn() -> Callable[[str], str]:
    from scripts import alpha_fit

    table = alpha_fit.shape_table()
    return lambda sid: alpha_fit.shape_of(sid, table)


def alpha_of(tau_s: float) -> float:
    return 1.0 if tau_s <= 0 else 1.0 - math.exp(-DT_REF_S / tau_s)


def step90_s(tau_s: float) -> float:
    a = alpha_of(tau_s)
    if a >= 1.0:
        return 0.0
    return DT_REF_S * math.ceil(math.log(0.1) / math.log(1.0 - a))


def spec_for(tau_s: float, w_p: float, lam: float):
    from scripts import theta_verdict as tv

    return tv.build_signal_spec("tss", w_p=w_p, lambda_wait=lam, qmin=1.0,
                                ema_tau_ms=(tau_s * 1000.0 if tau_s > 0 else None))


# ------------------------------------------------------------------ scoring helpers


def rates(pred: Sequence[bool], truth_violated: Sequence[bool]) -> tuple:
    tp = sum(1 for p, v in zip(pred, truth_violated) if p and v)
    fn = sum(1 for p, v in zip(pred, truth_violated) if not p and v)
    fp = sum(1 for p, v in zip(pred, truth_violated) if p and not v)
    tn = sum(1 for p, v in zip(pred, truth_violated) if not p and not v)
    rec = tp / (tp + fn) if tp + fn else float("nan")
    fa = fp / (fp + tn) if fp + tn else float("nan")
    return rec, fa, (rec + 1.0 - fa) / 2.0, tp + fn, fp + tn


def future_pairs(windows, crit) -> list[tuple]:
    """(cell, crit flag at t, violated at t+30 s) for every window whose +30 s window is labelled."""
    idx = {(w.scenario_id, int(w.window_start_ms)): i for i, w in enumerate(windows)}
    out = []
    for i, w in enumerate(windows):
        j = idx.get((w.scenario_id, int(w.window_start_ms) + HORIZON_MS))
        if j is not None:
            out.append((w.scenario_id, crit[i], not windows[j].slo_met))
    return out


def detection_lags(windows, crit) -> list[Optional[float]]:
    """Per violation episode (run of violated windows in a cell), seconds from its first
    window to the first dwell-confirmed CRITICAL within [start-30 s, end]; None = missed."""
    by = defaultdict(list)
    for i, w in enumerate(windows):
        by[w.scenario_id].append(i)
    lags: list[Optional[float]] = []
    for idx in by.values():
        idx.sort(key=lambda i: windows[i].window_start_ms)
        k = 0
        while k < len(idx):
            if windows[idx[k]].slo_met:
                k += 1
                continue
            s = k
            while k < len(idx) and not windows[idx[k]].slo_met:
                k += 1
            t0 = windows[idx[s]].window_start_ms
            t1 = windows[idx[k - 1]].window_start_ms
            hits = [windows[i].window_start_ms for i in idx
                    if crit[i] and t0 - HORIZON_MS <= windows[i].window_start_ms <= t1]
            lags.append((min(hits) - t0) / 1000.0 if hits else None)
    return lags


def fit_theta_delta(windows, spec):
    from tre_calibration.fit import fit_delta_margins

    cfg = spec.default_config()
    fit = cfg.fit(windows)
    if not fit.publish or fit.theta is None:
        return None
    theta = float(fit.theta)
    d = fit_delta_margins(windows, theta=theta, direction=spec.direction)
    return theta, d.crit.tau, d.crit.delta, d.high.delta, fit


def boot_ba_se(pairs, n: int = 300, seed: int = SEED) -> float:
    cells = sorted({c for c, _, _ in pairs})
    by = defaultdict(list)
    for c, p, v in pairs:
        by[c].append((p, v))
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        smp = [x for c in (rng.choice(cells) for _ in cells) for x in by[c]]
        _, _, ba, npos, nneg = rates([p for p, _ in smp], [v for _, v in smp])
        if npos and nneg:
            vals.append(ba)
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


# ------------------------------------------------------------------------- alpha


def stage_alpha_refit0922(model: str, label, p: Mapping[str, Any], *, w_p: float) -> dict:
    """The archived alpha stage (t + 30 s label; kept to reproduce 2026-09-22)."""
    from scripts import theta_verdict as tv

    shape_of = shape_fn()
    lam = LAMBDA_WAIT
    curve = []
    for tau in TAUS_S:
        spec = spec_for(tau, w_p, lam)
        windows = spec.load(p["fitting"], label, TRIM_RAMP_WINDOWS)
        shapes = sorted({shape_of(w.scenario_id) for w in windows})
        pairs, lags, folds = [], [], {}
        for s in shapes:
            train = [w for w in windows if shape_of(w.scenario_id) != s]
            test = [w for w in windows if shape_of(w.scenario_id) == s]
            fd = fit_theta_delta(train, spec)
            if fd is None:
                folds[s] = None
                continue
            theta, tau_crit = fd[0], fd[1]
            crit = tv.critical_dwell_flags(test, theta=theta, tau_crit=tau_crit,
                                           direction=spec.direction, dwell_windows=DWELL_WINDOWS)
            pairs += future_pairs(test, crit)
            lags += detection_lags(test, crit)
            folds[s] = {"theta": theta, "tau_crit": tau_crit}
        rec, fa, ba, npos, nneg = rates([q for _, q, _ in pairs], [v for _, _, v in pairs])
        se = boot_ba_se(pairs)
        full = fit_theta_delta(windows, spec)
        hit = sorted(x for x in lags if x is not None)
        curve.append({
            "tau_s": tau, "alpha": alpha_of(tau), "loso_ba": ba, "loso_ba_se": se, "recall": rec,
            "false_alarm": fa, "n_pos": npos, "n_neg": nneg, "feasible_fa": fa <= FA_MAX,
            "step90_ema_s": step90_s(tau), "step90_with_dwell_s": step90_s(tau) + DT_REF_S,
            "episodes": len(lags), "episodes_detected": len(hit),
            "detect_lag_median_s": hit[len(hit) // 2] if hit else None,
            "full_fit": ({"theta": full[0], "tau_crit": full[1], "delta_crit": full[2], "delta_high": full[3]}
                         if full else None),
            "folds": folds,
        })
        print(model, tau, f"BA={ba:.3f}+-{se:.3f} rec={rec:.3f} fa={fa:.3f}", flush=True)
    feas = [c for c in curve if c["feasible_fa"]] or curve
    best = max(feas, key=lambda c: c["loso_ba"])
    ok = [c for c in feas if c["loso_ba"] >= best["loso_ba"] - best["loso_ba_se"]]
    chosen = min(ok, key=lambda c: c["tau_s"])  # most responsive = smallest tau
    return {"rule": "refit0922", "w_p": w_p, "lambda_wait": lam,
            "objective": "LOSO (leave-one-shape-out) BA of dwell-2 CRITICAL at t vs label at t+30 s, "
                         "healthy false alarm <= 0.05, most responsive tau within 1 SE of the best",
            "no_feasible_tau": not any(c["feasible_fa"] for c in curve),
            "best_tau_s": best["tau_s"], "chosen_tau_s": chosen["tau_s"], "chosen_alpha": chosen["alpha"],
            "curve": curve}


def stage_alpha_d4prime(model: str, label, p: Mapping[str, Any], *, w_p: float,
                        ledgers: Sequence[str] = (), bootstrap: int = 1000,
                        registry: Optional[str] = None) -> dict:
    """D4' (:mod:`scripts.alpha_fit`) on the same fitting CSV and label."""
    from scripts import alpha_fit

    ns = argparse.Namespace(
        model=model, fitting_csv=str(p["fitting"]), w_p=w_p, lambda_wait=LAMBDA_WAIT, qmin=1.0,
        trim_ramp_windows=TRIM_RAMP_WINDOWS, tau_grid_s=[float(t) for t in TAUS_S],
        dt_ref_s=DT_REF_S, dwell_windows=DWELL_WINDOWS, fa_max=FA_MAX,
        step_ms=alpha_fit.DEFAULT_STEP_MS, se_resamples=alpha_fit.DEFAULT_SE_RESAMPLES,
        bootstrap=bootstrap, bootstrap_refit=False, seed=alpha_fit.DEFAULT_SEED,
        ledger=list(ledgers),
        # the label: this arm's definition, passed field by field
        ttft_p95_ms=label.ttft_p95_ms, tpot_p95_ms=label.tpot_p95_ms,
        ttft_slo_mode=label.ttft_slo_mode, ttft_slowdown_k=label.ttft_slowdown_k,
        ttft_floor_ms=label.ttft_floor_ms, ttft_idle_c_ms=label.ttft_idle_c_ms,
        ttft_idle_b_ms_per_token=label.ttft_idle_b_ms_per_token,
        min_completed_requests=label.min_completed_requests, label_registry=registry,
    )
    rep = alpha_fit.run(ns)
    sel = rep.get("selection") or {}
    return {"rule": "d4prime", "w_p": w_p, "lambda_wait": LAMBDA_WAIT,
            "chosen_tau_s": sel.get("chosen_tau_s"), "chosen_alpha": sel.get("chosen_alpha"),
            "alpha_fit": rep}


# ------------------------------------------------------------- D18: tau published


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def alpha_curve_points(doc: Mapping[str, Any]) -> list[dict]:
    """The alpha sweep of either rule as ``{tau_s, alpha, ba, se, fa, recall, feasible,
    step90_ema_s, step90_with_dwell_s}`` points, grid order."""
    if doc.get("rule") == "d4prime":
        return [{"tau_s": float(c["tau_s"]), "alpha": c["alpha"], "ba": c["ba"], "se": c["se"],
                 "fa": c["fa"], "recall": c["recall"], "feasible": bool(c["feasible"]),
                 "step90_ema_s": c["step90_ema_s"], "step90_with_dwell_s": c["step90_with_dwell_s"]}
                for c in (doc.get("alpha_fit") or {}).get("curve") or []]
    return [{"tau_s": float(c["tau_s"]), "alpha": c["alpha"], "ba": c["loso_ba"], "se": c["loso_ba_se"],
             "fa": c["false_alarm"], "recall": c["recall"], "feasible": bool(c["feasible_fa"]),
             "step90_ema_s": c["step90_ema_s"], "step90_with_dwell_s": c["step90_with_dwell_s"]}
            for c in doc.get("curve") or []]


def alpha_disclosure(doc: Mapping[str, Any], published_tau_s: float) -> dict:
    """D18: what is disclosed about the alpha sweep next to the published tau.

    * the three curves over the tau grid - BA (with its SE), healthy false alarm and the
      90 % step delay (EMA + dwell) - plus recall;
    * the 1-SE set (feasible taus whose BA is within one SE of the best feasible BA - the
      D4' candidate set) and the flat interval: the contiguous run of grid taus around the
      published tau inside that set (None when the published tau is outside it);
    * over :data:`DISCLOSURE_TAU_RANGE_S`: the BA range and whether it is within one SE;
    * the bootstrap selection frequency of every grid tau (d4prime only)."""
    pts = alpha_curve_points(doc)
    valid = [q for q in pts if _finite(q["ba"])]
    pool = [q for q in valid if q["feasible"]] or valid
    out: dict[str, Any] = {
        "published_tau_s": published_tau_s,
        "curves": {k: [q[k] for q in pts] for k in
                   ("tau_s", "alpha", "ba", "se", "fa", "recall", "step90_ema_s", "step90_with_dwell_s")},
    }
    if not pool:
        return out | {"within_1se_tau_s": [], "flat_interval_s": None}
    best = max(pool, key=lambda q: (q["ba"], q["alpha"]))
    one_se = best["se"] if _finite(best["se"]) else 0.0
    within = [q["tau_s"] for q in pool if q["ba"] >= best["ba"] - one_se]
    grid = [q["tau_s"] for q in pts]
    flat = None
    if published_tau_s in within:
        i = j = grid.index(published_tau_s)
        while i > 0 and grid[i - 1] in within:
            i -= 1
        while j < len(grid) - 1 and grid[j + 1] in within:
            j += 1
        flat = [grid[i], grid[j]]
    lo, hi = DISCLOSURE_TAU_RANGE_S
    rng = [q for q in valid if lo <= q["tau_s"] <= hi]
    ba_range = (max(q["ba"] for q in rng) - min(q["ba"] for q in rng)) if rng else None
    out.update({
        "best_ba_tau_s": best["tau_s"], "best_ba": best["ba"], "one_se": one_se,
        "feasible_tau_s": [q["tau_s"] for q in pts if q["feasible"]],
        "within_1se_tau_s": within, "flat_interval_s": flat,
        "published_within_1se_of_best": published_tau_s in within,
        "published_on_grid": published_tau_s in grid,
        "published_point": next((q for q in pts if q["tau_s"] == published_tau_s), None),
        "range_s": [lo, hi],
        "range_ba_spread": ba_range,
        "range_ba_spread_within_1se": (ba_range <= one_se) if ba_range is not None else None,
        "range_all_within_1se_of_best": all(q["tau_s"] in within for q in rng) if rng else None,
        "range_max_fa": max((q["fa"] for q in rng if _finite(q["fa"])), default=None),
    })
    if doc.get("rule") == "d4prime":
        boot = (doc.get("alpha_fit") or {}).get("bootstrap") or {}
        out["selection_frequency_by_tau_s"] = boot.get("selection_frequency_by_tau_s")
        out["bootstrap_resamples"] = boot.get("used")
    return out


def publish_alpha(doc: dict, publish_tau_s: Optional[float]) -> dict:
    """D18 on an alpha-stage document: the rule's pick stays ``chosen_tau_s`` (disclosed);
    ``published_tau_s`` - what w_p / theta / delta are fitted at and what deploys - is
    ``publish_tau_s``, or the rule's pick when that is None (``--publish-tau-s rule``)."""
    rule_tau = doc.get("chosen_tau_s")
    tau = rule_tau if publish_tau_s is None else float(publish_tau_s)
    doc["published_tau_s"] = tau
    doc["published_alpha"] = alpha_of(tau) if tau is not None else None
    doc["publish_rule"] = ("the alpha rule's pick (--publish-tau-s rule)" if publish_tau_s is None
                           else f"D18: the common tau {tau:g} s, whatever the rule picks")
    doc["published_registry_fields"] = (
        {"ema_tau_ms": tau * 1000.0, "ema_alpha": round(alpha_of(tau), 6)} if tau is not None else None)
    if tau is not None:
        doc["disclosure"] = alpha_disclosure(doc, tau)
    return doc


def publish_tau_arg(text: str) -> Optional[float]:
    if text == "rule":
        return None
    tau = float(text)
    if not math.isfinite(tau) or tau < 0:
        raise argparse.ArgumentTypeError(f"--publish-tau-s: {text!r} is not a tau >= 0 or 'rule'")
    return tau


# ---------------------------------------------------------------------------- w_p


def verdict(model: str, label, p: Mapping[str, Any], tau: float, w_p: float, lam: float,
            n_res: int = 1000, fam_res: int = 200) -> dict:
    from scripts import theta_verdict as tv

    spec = spec_for(tau, w_p, lam)
    try:
        return tv.verdict_report(model=model, fitting_csv=p["fitting"], families=p["families"], spec=spec,
                                 label=label, trim_ramp_windows=TRIM_RAMP_WINDOWS, n_resamples=n_res,
                                 family_resamples=fam_res, seed=SEED)
    except tv.VerdictError as exc:
        return {"error": str(exc)}


def summarize(v: Mapping[str, Any]) -> dict:
    if "error" in v:
        return dict(v)
    m, fams = v["merged"], v["families"]
    tP = fams.get("prefill_heavy", {}).get("theta")
    tD = fams.get("decode_heavy", {}).get("theta")
    theta = m["theta"]
    half_frac = m["bootstrap"]["ci_half_width_fraction"]
    gap = abs(tP - tD) / theta if tP and tD else None
    return {
        "theta_merged": theta, "theta_published": v["published"]["theta_m"], "source": v["published"]["source"],
        "train_ba": m["fit"].get("balanced_accuracy"), "ci_half_frac": half_frac,
        "publish_rate": m["bootstrap"]["publish_rate"], "theta_P": tP, "theta_D": tD,
        "family_ratio_P_over_D": (tP / tD) if tP and tD else None, "family_gap_frac": gap,
        "family_merged": v["published"]["source"] == "merged",
        "delta_crit": v["published"]["delta_crit"], "delta_high": v["published"]["delta_high"],
        "tau_crit": v["published"]["tau_crit"],
        "delta_crit_ci": [m["delta_crit"]["bootstrap"].get(k) for k in ("delta_p2_5", "delta_p97_5")],
        "stop_rule": v["stop_rule"], "near_band_windows_merged": m["near_theta"]["windows"],
        "family_near_band": {k: f.get("near_merged_theta") for k, f in fams.items()},
    }


def ba_se_at(label, p: Mapping[str, Any], tau: float, w_p: float, lam: float, theta: float,
             n: int = BA_SE_RESAMPLES) -> float:
    """Cell-bootstrap SE of the training BA at a fixed theta (the D3 1-SE width)."""
    from tre_calibration.fit import threshold_balanced_accuracy

    ws = spec_for(tau, w_p, lam).load(p["fitting"], label, TRIM_RAMP_WINDOWS)
    by = defaultdict(list)
    for w in ws:
        by[w.scenario_id].append(w)
    cells = sorted(by)
    rng = random.Random(SEED)
    vals = []
    for _ in range(n):
        smp = [w for c in (rng.choice(cells) for _ in cells) for w in by[c]]
        vals.append(threshold_balanced_accuracy(smp, theta=theta, direction="higher_is_healthier")["balanced_accuracy"])
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


#: D17 (2026-09-23, revising D3): the conditions a w_p must meet. c3 is still computed and
#: reported, but no longer selects - it restated D5 (the family verdict is diagnostic).
D3_CONDITIONS = ("c1_1se", "c2_gap")
D3_DIAGNOSTIC = ("c3_merged",)


def d3_select(rows: list[dict], se: Optional[float]) -> Optional[float]:
    """D3 as revised by D17, in place on ``rows`` (``summarize`` rows over the w_p grid, the
    first at w_p = 0): mark c1 (BA within one SE of the w_p = 0 BA), c2 (family gap within
    the CI half width) and ``admissible`` = c1 and c2; return the largest admissible w_p, or
    None when none is (or the w_p = 0 fit failed). c3 (the family rule publishes the merged
    theta) is marked as a diagnostic only: it is redundant with D5, and on data holding
    shapes outside both families it fails for every w_p."""
    base = rows[0]
    for r in rows:
        if "error" in r or se is None or "error" in base:
            r["admissible"] = False
            continue
        r["c1_1se"] = r["train_ba"] >= base["train_ba"] - se
        r["c2_gap"] = r["family_gap_frac"] is not None and r["family_gap_frac"] <= r["ci_half_frac"]
        r["c3_merged"] = r["family_merged"]
        r["admissible"] = all(r[c] for c in D3_CONDITIONS)
    adm = [r["w_p"] for r in rows if r.get("admissible")]
    return max(adm) if adm else None


def lambda_select(rows: list[dict]) -> float:
    """lambda_wait: stays at LAMBDA_WAIT unless the best lambda beats it by >= LAMBDA_MIN_GAIN BA."""
    ba1 = next((r for r in rows if r["lambda_wait"] == LAMBDA_WAIT), {}).get("train_ba")
    if ba1 is None:
        return LAMBDA_WAIT
    ok = [r for r in rows if "error" not in r]
    best = max(ok, key=lambda r: r["train_ba"])
    return best["lambda_wait"] if best["train_ba"] - ba1 >= LAMBDA_MIN_GAIN else LAMBDA_WAIT


def published_tau(alpha_doc: Mapping[str, Any]) -> Optional[float]:
    """The tau theta / delta / w_p are fitted at: the alpha stage's published tau (D18), or,
    for an alpha.json written before D18, the rule's pick."""
    if "published_tau_s" in alpha_doc:
        return alpha_doc["published_tau_s"]
    return alpha_doc.get("chosen_tau_s")


def stage_wp(model: str, label, p: Mapping[str, Any], alpha_doc: Mapping[str, Any]) -> dict:
    tau = published_tau(alpha_doc)
    if tau is None:
        raise SystemExit(f"{model}: the alpha stage published no tau ({alpha_doc.get('rule')})")
    rows = []
    for wp in WP_GRID:
        s = summarize(verdict(model, label, p, tau, wp, LAMBDA_WAIT, *WP_RESAMPLES))
        s["w_p"] = wp
        rows.append(s)
        print(model, "wp", wp, {k: s.get(k) for k in ("theta_merged", "train_ba", "family_gap_frac", "source")},
              flush=True)
    base = rows[0]
    se = ba_se_at(label, p, tau, 0.0, LAMBDA_WAIT, base["theta_merged"]) if "error" not in base else None
    wp_star = d3_select(rows, se)
    wp_l = wp_star if wp_star is not None else 0.0
    lam_rows = []
    for lam in LAMBDAS:
        s = summarize(verdict(model, label, p, tau, wp_l, lam, *LAMBDA_RESAMPLES))
        s["lambda_wait"] = lam
        lam_rows.append(s)
    return {"model": model, "tau_s": tau, "ba0_se": se, "grid": rows,
            "d3_rule": {"conditions": list(D3_CONDITIONS), "diagnostic": list(D3_DIAGNOSTIC),
                        "statement": "largest w_p meeting c1 and c2 (D17: c3 reported, not selecting)"},
            "admissible": [r["w_p"] for r in rows if r.get("admissible")],
            "admissible_with_c3": [r["w_p"] for r in rows if r.get("admissible") and r.get("c3_merged")],
            "w_p_star": wp_star, "w_p_used": wp_l, "lambda_rows": lam_rows,
            "lambda_star": lambda_select(lam_rows)}


def stage_wp_v1(model: str, label, p: Mapping[str, Any], alpha_doc: Mapping[str, Any],
                sources: Mapping[str, Path]) -> dict:
    """``wp --lambda-method v1``: lambda_wait and w_p from v1's selection (stages A-C of
    ``fit_tre_parameters_from_runs.py``, :mod:`scripts.v1_lambda_fit`) on the D16 fitting
    windows; ``sources`` (run -> standard dataset dir) are where the average TPOT of v1's
    average-health term is rebuilt from. The D17 w_p rule is NOT applied: user 2026-09-24,
    v1's joint refinement is taken as is and a disagreement with D17 is reported."""
    from scripts import v1_lambda_fit

    tau = published_tau(alpha_doc)
    if tau is None:
        raise SystemExit(f"{model}: the alpha stage published no tau ({alpha_doc.get('rule')})")
    sel = v1_lambda_fit.fit_model(model, label, p["fitting"], trim=TRIM_RAMP_WINDOWS, sources=sources)
    lam, wp = sel["lambda_wait"], sel["w_p"]
    print(model, "v1 selection", {"lambda_wait": lam, "w_p": wp,
                                  "objective_adjusted": sel["best_c"]["objective_adjusted"]}, flush=True)
    return {"model": model, "tau_s": tau, "lambda_method": "v1", "ba0_se": None, "grid": [],
            "d3_rule": {"conditions": [], "diagnostic": [],
                        "statement": ("not applied: --lambda-method v1 takes w_p from v1's joint "
                                      "lambda x w_p refinement (user 2026-09-24)")},
            "admissible": [], "admissible_with_c3": [],
            "w_p_star": wp, "w_p_used": wp, "lambda_rows": [], "lambda_star": lam,
            "v1_selection": sel}


# -------------------------------------------------------------------------- final


def bucket(length: float) -> str:
    return ("<=256" if length <= 256 else "257-1024" if length <= 1024
            else "1025-2048" if length <= 2048 else ">2048")


#: What ``final.json`` says instead of the M numbers when the stage ran with ``--no-holdout``.
HOLDOUT_SKIPPED = ("not evaluated (--no-holdout): M is read once, after it is frozen and "
                   "hashed (plan 2026-09-21 §6.11 note 9)")


def stage_final(model: str, label, p: Mapping[str, Any], wp_doc: Mapping[str, Any], out_dir: Path,
                *, holdout: bool = True) -> dict:
    """D5 verdict at (tau, w_p*, lambda*); then, unless ``holdout`` is False, the M report.

    With ``holdout=False`` the validation CSV is never opened (not even for its size)."""
    from tre_calibration.fit import threshold_balanced_accuracy

    from scripts import theta_verdict as tv

    tau, wp, lam = wp_doc["tau_s"], wp_doc["w_p_used"], wp_doc["lambda_star"]
    v = verdict(model, label, p, tau, wp, lam, *FINAL_RESAMPLES)
    out_dir.mkdir(parents=True, exist_ok=True)
    if "error" in v:
        (out_dir / "verdict_final.json").write_text(json.dumps(v, indent=1, default=str))
        return {"model": model, "error": v["error"]}
    # D5: the merged theta is published whatever the family verdict says (diagnostic only)
    v["published"]["family_rule_theta"] = v["published"]["theta_m"]
    v["published"]["theta_m"] = v["merged"]["theta"]
    v["published"]["d5_merged_published"] = True
    (out_dir / "verdict_final.json").write_text(json.dumps(v, indent=1, default=str))
    if not holdout:
        spec = spec_for(tau, wp, lam)
        s = summarize(v)
        s["theta_family_rule"] = v["published"]["family_rule_theta"]
        return {
            "model": model, "tau_s": tau, "alpha": alpha_of(tau), "w_p": wp, "lambda_wait": lam,
            **s, "stop_rule_15": v["stop_rule"]["satisfied"],
            "appendix_10_met": v["stop_rule"].get("appendix_ci_target_met"),
            "train_ba_at_published": threshold_balanced_accuracy(
                spec.load(p["fitting"], label, TRIM_RAMP_WINDOWS), theta=v["published"]["theta_m"],
                direction="higher_is_healthier")["balanced_accuracy"],
            "holdout_evaluated": False, "holdout": HOLDOUT_SKIPPED,
        }
    if not Path(p["validation"]).exists():
        raise SystemExit(f"{p['validation']} does not exist: a D16 training-set directory carries "
                         "no M - run final with --no-holdout until M is frozen")
    h = tv.holdout_report(v, p["validation"], dwell_windows=DWELL_WINDOWS)
    (out_dir / "holdout_final.json").write_text(json.dumps(h, indent=1, default=str))
    # M BA CI (cell bootstrap; few M cells -> wide, reported as such)
    spec = spec_for(tau, wp, lam)
    mw = spec.load(p["validation"], label, TRIM_RAMP_WINDOWS)
    theta = v["published"]["theta_m"]
    by = defaultdict(list)
    for x in mw:
        by[x.scenario_id].append(x)
    cells = sorted(by)
    rng = random.Random(SEED)
    bas = []
    for _ in range(M_CI_RESAMPLES):
        smp = [x for c in (rng.choice(cells) for _ in cells) for x in by[c]]
        r = threshold_balanced_accuracy(smp, theta=theta, direction="higher_is_healthier")
        if r["balanced_accuracy"] == r["balanced_accuracy"]:
            bas.append(r["balanced_accuracy"])
    bas.sort()
    ci = [bas[int(0.025 * len(bas))], bas[int(0.975 * len(bas)) - 1]] if bas else [None, None]
    # per-length-bucket request attainment on M (non-overlapping 30 s tiles only)
    att = defaultdict(lambda: [0, 0])
    first: dict[str, int] = {}
    with open(p["validation"], newline="") as fh:
        for row in csv.DictReader(fh):
            c, s = row["scenario_id"], int(row["window_start_ms"])
            first.setdefault(c, s)
            if (s - first[c]) % HORIZON_MS:
                continue
            for ttft, length in slo_labels.parse_ttft_len_samples(row.get("ttft_len_samples") or ""):
                b = bucket(length)
                att[b][1] += 1
                att[b][0] += ttft <= label.ttft_slo_ms(length)
    wd = h["with_dwell"]
    s = summarize(v)
    s["theta_family_rule"] = v["published"]["family_rule_theta"]
    return {
        "model": model, "tau_s": tau, "alpha": alpha_of(tau), "w_p": wp, "lambda_wait": lam,
        **s, "stop_rule_15": v["stop_rule"]["satisfied"],
        "appendix_10_met": v["stop_rule"].get("appendix_ci_target_met"),
        "train_ba_at_published": threshold_balanced_accuracy(
            spec.load(p["fitting"], label, TRIM_RAMP_WINDOWS), theta=theta,
            direction="higher_is_healthier")["balanced_accuracy"],
        "holdout_evaluated": True,
        "M_windows": h["windows"], "M_cells": h["cells"], "M_violating": h["violating"],
        "M_ba": h["at_published_theta"]["balanced_accuracy"], "M_ba_ci95": ci,
        "M_recall_both_tpot_dwell2": wd["critical_recall_both_tpot"], "M_both_tpot_n": wd["both_tpot_windows"],
        "M_false_alarm_dwell2": wd["critical_false_alarm_on_healthy"], "M_healthy_n": wd["healthy_windows"],
        "M_recall_all_dwell2": wd["critical_recall_of_violating"],
        "M_ttft_only_recall_dwell2": wd["violation_classes"]["ttft_only"]["critical_recall"],
        "M_ttft_only_n": wd["violation_classes"]["ttft_only"]["windows"],
        "M_classes_dwell2": wd["violation_classes"],
        "M_bucket_attainment": {b: {"met": a[0], "n": a[1], "rate": a[0] / a[1] if a[1] else None}
                                for b, a in sorted(att.items())},
    }


# ------------------------------------------------------------------------ summary


def band_counts(model: str, arm: str, final: Mapping[str, Any], p: Mapping[str, Any], *,
                registry: Optional[str] = None, ledger: Optional[Mapping[str, Any]] = None) -> dict:
    """Boundary-band windows (|Z - 1| <= theta_verdict.BOUNDARY_BAND at the published theta)
    per shape and family, and the hold cells a family short of MIN_FAMILY_WINDOWS needs.

    A "dwell" cell (the yield estimate) is a steady cell of >= 20 windows -
    ``alpha_fit.is_steady_cell``, i.e. from the run's ledger when there is one."""
    from scripts import alpha_fit
    from scripts import gen_calibration_schedules as gen
    from scripts import theta_verdict as tv

    families = gen.families()
    family_of = {s: fam for fam, shapes in families.items() for s in shapes}
    shape_of = shape_fn()
    spec = spec_for(final["tau_s"], final["w_p"], final["lambda_wait"])
    ws = spec.load(p["fitting"], label_for(model, arm, registry), TRIM_RAMP_WINDOWS)
    theta = final["theta_published"]
    band: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    cellband: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for x in ws:
        z = x.signal / theta if theta else float("nan")
        inside = int(math.isfinite(z) and abs(z - 1.0) <= tv.BOUNDARY_BAND)
        cellband[x.scenario_id][1] += 1
        cellband[x.scenario_id][0] += inside
        s = shape_of(x.scenario_id)
        band[s][1] += 1
        band[s][0] += inside
    fam: dict[str, int] = defaultdict(int)
    for s, (n, _total) in band.items():
        if s in family_of:
            fam[family_of[s]] += n
    need = []
    for famname, shapes in sorted(families.items()):
        have = fam.get(famname, 0)
        deficit = max(0, MIN_FAMILY_WINDOWS - have)
        dw = [v[0] for c, v in cellband.items()
              if shape_of(c) in shapes and v[1] >= 20 and alpha_fit.is_steady_cell(c, ledger)]
        yield_per_cell = (sum(dw) / len(dw)) if dw and sum(dw) > 0 else 28 * 0.3
        need.append({"family": famname, "band_windows": have, "deficit": deficit, "shapes": list(shapes),
                     "hold_cells": math.ceil(deficit / yield_per_cell) if deficit else 0,
                     "yield_band_windows_per_hold_cell": round(yield_per_cell, 1),
                     "observed_dwell_cells": len(dw)})
    half = final.get("ci_half_frac")
    return {
        "band_by_shape": {s: {"band": v[0], "independent": round(v[0] / 3, 1), "windows": v[1]}
                          for s, v in sorted(band.items())},
        "band_by_family": {k: {"band": v, "independent": round(v / 3, 1)} for k, v in fam.items()},
        "needs": {"family_holds": need, "ci_half_frac": half,
                  "ci_window_factor_for_15pct": round((half / 0.15) ** 2, 2) if half and half > 0.15 else 1.0,
                  "ci_window_factor_for_10pct": round((half / 0.10) ** 2, 2) if half else None},
    }


def stage_summary(out_root: Path, fit_dirs: Mapping[str, Path], *, registry: Optional[str] = None,
                  ledger: Optional[Mapping[str, Any]] = None) -> dict:
    out: dict[str, Any] = {"root": str(out_root), "models": {}}
    for model in sorted(fit_dirs):
        for arm in ARMS:
            d = out_root / model / arm
            if not (d / "final.json").exists():
                continue
            a = json.loads((d / "alpha.json").read_text())
            w = json.loads((d / "wp.json").read_text())
            fin = json.loads((d / "final.json").read_text())
            disc = a.get("disclosure") or {}
            rec = {
                "alpha": {**{k: a.get(k) for k in ("rule", "chosen_tau_s", "chosen_alpha", "best_tau_s", "w_p",
                                                   "published_tau_s", "published_alpha", "publish_rule")},
                          "flat_interval_s": disc.get("flat_interval_s"),
                          "within_1se_tau_s": disc.get("within_1se_tau_s"),
                          "range_ba_spread_within_1se": disc.get("range_ba_spread_within_1se")},
                "training_set": fin.get("training_set"),
                "wp_rule": {"tau_s": w["tau_s"], "ba0_se": w["ba0_se"], "admissible": w["admissible"],
                            "w_p_star": w["w_p_star"], "w_p_used": w["w_p_used"],
                            "lambda_star": w["lambda_star"], "d3_rule": w.get("d3_rule"),
                            "admissible_with_c3": w.get("admissible_with_c3")},
                "final": fin,
            }
            if "error" not in fin:
                rec.update(band_counts(model, arm, fin, paths(fit_dirs[model], model),
                                       registry=registry, ledger=ledger))
            out["models"].setdefault(model, {})[arm] = rec
    return out


# ------------------------------------------------------- freeze (D22) and accept (A-D)
#
# Plan §6.11 D22: refit -> FREEZE theta / w_p / alpha / delta -> collect M -> evaluate the
# acceptance criteria A-D of plan §6.9f ONCE, on the frozen parameters and M only.
#
# ``freeze`` refuses unless every model's refit is final, never read M, meets the D13 stop
# rule and still sits on the training inputs it was fitted on; it writes one JSON for all
# models, self-hashed (``freeze_sha256``, :func:`canonical_sha256`), with a sha256sum
# sidecar, read-only. ``accept`` reads only that file and M (standard datasets + one M
# manifest per model, written by the collection side), refuses on any provenance break,
# and writes its result and a marker once; ``--recheck`` recomputes in a temp dir and
# compares, never writing next to the freeze.

FREEZE_FORMAT_REVISION = 1
M_MANIFEST_FORMAT_REVISION = 1
#: The refit stage outputs a freeze reads (``<out>/<model>/<arm>/<name>.json``).
FREEZE_STAGE_FILES = ("alpha", "wp", "final", "verdict_final")
M_MANIFEST_KEYS = ("model", "format_revision", "freeze", "label_def", "label_def_sha256", "cells",
                   "sealed_probes", "sha256sums_file", "sha256sums_sha256")
M_CELL_KEYS = ("model", "cell_id", "attempt", "shape", "primitive", "role", "origin", "seen_before", "note")
M_CELL_ORIGINS = ("collected", "retained")
M_REQUIRED_COLUMNS = ("model", "cell_id", "attempt", "split", "cell_status", "scenario_id", "shape",
                      "primitive", "role")
CELL_STATUS_VALID = "valid"
#: Plan §6.9f: the thresholds of criteria A and B, the non-gating target of all-violating
#: recall, and windows per independent window (30 s windows on a 10 s step) for C.
A_BA_MIN = 0.80
A_BA_CI_LOW_MIN = 0.75
A_MAX_DROP_FROM_TRAINING = 0.08
B_RECALL_MIN = 0.85
B_RECALL_CI_LOW_MIN = 0.75
B_FALSE_ALARM_MAX = 0.05
B_FALSE_ALARM_CI_HIGH_MAX = 0.08
ALL_VIOLATING_RECALL_TARGET = 0.70
WINDOWS_PER_INDEPENDENT = 3
ACCEPT_RESAMPLES = 1000
ACCEPT_DWELL_WINDOWS = DWELL_WINDOWS
EXIT_REFUSED = 1
EXIT_ACCEPT_FAILED = 3
EXIT_RECHECK_DIFFERS = 4
#: Keys a ``--recheck`` does not compare: timestamps, work paths, the command line, and the
#: code state of the run (provenance of the run, printed when it differs, not a result).
ACCEPT_VOLATILE_KEYS = frozenset({"generated_at", "evaluated_at_utc", "validation_csv", "work_dir",
                                  "command", "code"})


class FreezeError(RuntimeError):
    """``freeze`` / ``verify-freeze`` / ``accept`` refused; ``problems`` lists every reason."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def canonical_json(obj: Any) -> bytes:
    """The bytes :func:`canonical_sha256` hashes: sorted keys, no whitespace, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha256(obj: Any) -> str:
    """sha256 of the canonical JSON of ``obj`` (the freeze self hash; label definitions)."""
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def code_state() -> dict:
    """Commit of the tre tree this module runs from, and whether tracked files are dirty."""
    import subprocess

    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"commit": commit, "dirty": dirty, "tre_root": str(root)}


def freeze_paths(freeze_file: Path) -> dict[str, Path]:
    """The files that go with a freeze file: its sha256 sidecar, the accept result, the
    accept marker and the accept work dir (the per-model validation CSVs)."""
    f = Path(freeze_file)
    return {"freeze": f, "sidecar": Path(f"{f}.sha256"), "result": f.with_name(f"{f.stem}.accept.json"),
            "marker": Path(f"{f}.accepted"), "work": f.with_name(f"{f.stem}.accept.d")}


def _write_once(path: Path, data: bytes, mode: int = 0o444) -> None:
    with open(path, "xb") as fh:
        fh.write(data)
    os.chmod(path, mode)


def _json_bytes(doc: Any) -> bytes:
    return (json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def training_input_files(fit_dir: Path, model: str) -> dict[str, Path]:
    """Every training input of ``model`` in a fit dir that exists: the fitting and family
    CSVs, the training ledger, trainset.json and h2.json."""
    p = paths(fit_dir, model)
    files = {"fitting": p["fitting"], **{f"family_{k}": v for k, v in p["families"].items()},
             "training_ledger": Path(fit_dir) / TRAINING_LEDGER,
             "trainset_manifest": Path(fit_dir) / TRAINSET_MANIFEST, "h2_manifest": Path(fit_dir) / H2_MANIFEST}
    return {k: v for k, v in files.items() if v.exists()}


def verdict_for_holdout(verdict_doc: Mapping[str, Any]) -> dict:
    """Exactly what ``theta_verdict.holdout_report`` reads from a verdict."""
    merged: dict[str, Any] = {"theta": verdict_doc["merged"]["theta"]}
    opp = verdict_doc["merged"].get("opposite_direction")
    if opp is not None:
        merged["opposite_direction"] = {k: opp.get(k) for k in ("direction", "theta", "publish")}
    return {
        "model": verdict_doc["model"], "signal": verdict_doc.get("signal"),
        "label_def": verdict_doc["label_def"], "signal_spec": verdict_doc["signal_spec"],
        "trim_ramp_windows": verdict_doc["trim_ramp_windows"],
        "fit_config": {"direction": verdict_doc["fit_config"]["direction"]},
        "published": dict(verdict_doc["published"]), "merged": merged,
    }


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=0.0)
    return a == b


def freeze_model(out_root: Path, fit_dir: Path, model: str, arm: str) -> tuple[Optional[dict], list[str]]:
    """One model's freeze entry, or the reasons it cannot be frozen (every one found)."""
    from scripts.rewindow_from_raw import load_ledgers

    d = Path(out_root) / model / arm
    problems: list[str] = []
    docs: dict[str, dict] = {}
    for name in FREEZE_STAGE_FILES:
        f = d / f"{name}.json"
        if not f.exists():
            problems.append(f"{f} is missing")
            continue
        try:
            docs[name] = json.loads(f.read_text(encoding="utf-8"))
        except ValueError as exc:
            problems.append(f"{f}: not JSON ({exc})")
    if problems:
        return None, problems
    alpha, wp, fin, ver = docs["alpha"], docs["wp"], docs["final"], docs["verdict_final"]
    for name, doc in (("final", fin), ("verdict_final", ver)):
        if "error" in doc:
            problems.append(f"{name}: the fit failed ({doc['error']})")
    if problems:
        return None, problems

    # M must not have been read before the freeze (plan §6.11 note 9)
    if fin.get("holdout_evaluated") is not False:
        problems.append(f"final ran with the hold-out (holdout_evaluated = {fin.get('holdout_evaluated')!r}): "
                        "M was read before the freeze - rerun final with --no-holdout")
    # D13
    stop = fin.get("stop_rule") or {}
    if stop.get("satisfied") is not True:
        reasons = stop.get("reasons") or ["final records no stop rule"]
        problems.append("D13 stop rule not satisfied: " + "; ".join(str(r) for r in reasons))
    # the four stage outputs are one refit
    if fin.get("arm") not in (None, arm):
        problems.append(f"final.json is arm {fin.get('arm')!r}, not {arm!r}")
    if ver.get("model") != model or fin.get("model") not in (None, model):
        problems.append(f"the stage outputs name model {fin.get('model')!r} / {ver.get('model')!r}, not {model!r}")
    if "signal_spec" not in ver:
        problems.append("verdict_final.json has no signal_spec")
    pub = ver.get("published") or {}
    spec_tss = (ver.get("signal_spec") or {}).get("tss") or {}
    tau_s = fin.get("tau_s")
    tau_ms = None if tau_s is None or tau_s <= 0 else tau_s * 1000.0
    for what, a, b in (
        ("published theta (final / verdict_final)", fin.get("theta_published"), pub.get("theta_m")),
        ("label_def (final / verdict_final)", fin.get("label_def"), ver.get("label_def")),
        ("w_p (final / wp.w_p_used)", fin.get("w_p"), wp.get("w_p_used")),
        ("lambda_wait (final / wp.lambda_star)", fin.get("lambda_wait"), wp.get("lambda_star")),
        ("tau_s (final / alpha published)", tau_s, published_tau(alpha)),
        ("w_p (final / verdict signal_spec)", fin.get("w_p"), spec_tss.get("w_p")),
        ("lambda_wait (final / verdict signal_spec)", fin.get("lambda_wait"), spec_tss.get("lambda_wait")),
        ("EMA tau ms (final / verdict signal_spec)", tau_ms, spec_tss.get("ema_tau_ms")),
        ("tau_crit (final / verdict_final)", fin.get("tau_crit"), pub.get("tau_crit")),
    ):
        if not _same(a, b):
            problems.append(f"stage outputs disagree on {what}: {a!r} != {b!r}")
    if not alpha.get("published_registry_fields"):
        problems.append("alpha.json publishes no registry fields (ema_tau_ms / ema_alpha)")

    # training inputs: unchanged since final ran (D16 provenance)
    fit_dir = Path(fit_dir)
    ts = fin.get("training_set") or {}
    man = fit_dir / TRAINSET_MANIFEST
    man_doc: dict = {}
    p = paths(fit_dir, model)
    required = {"fitting": p["fitting"], **{f"family_{k}": v for k, v in p["families"].items()}}
    for key, path in required.items():
        if not path.exists():
            problems.append(f"training input {key} {path} does not exist")
    if not ts.get("trainset_manifest"):
        problems.append("final records no trainset manifest: a freeze needs a training set built by "
                        "the trainset stage (D16)")
    elif not man.exists():
        problems.append(f"{man} does not exist (final ran on {ts['trainset_manifest']})")
    else:
        have = sha256_file(man)
        if have != ts.get("trainset_manifest_sha256"):
            problems.append(f"{man} (sha256 {have}) is not the trainset manifest final ran on "
                            f"({ts['trainset_manifest']}, sha256 {ts.get('trainset_manifest_sha256')}): the fit "
                            "dir does not match this refit, or its training inputs changed since")
        else:
            man_doc = json.loads(man.read_text(encoding="utf-8"))
            lp = fit_dir / TRAINING_LEDGER
            try:
                check_training_inputs(model, p, ledger=load_ledgers([str(lp)]) if lp.exists() else None)
            except SystemExit as exc:
                problems.append(f"training inputs: {exc}")
    if problems:
        return None, problems

    h2_path = fit_dir / H2_MANIFEST
    stage_files = {name: d / f"{name}.json" for name in FREEZE_STAGE_FILES}
    entry = {
        "fit_dir": str(fit_dir), "refit_dir": str(d),
        "published": {
            "signal": ver.get("signal"), "direction": ver["fit_config"]["direction"],
            "theta": fin["theta_published"], "w_p": fin["w_p"], "tau_s": fin["tau_s"], "alpha": fin["alpha"],
            "lambda_wait": fin["lambda_wait"], "delta_crit": fin["delta_crit"], "delta_high": fin["delta_high"],
            "tau_crit": fin["tau_crit"], "tau_high": pub.get("tau_high"),
        },
        "registry": dict(alpha["published_registry_fields"]),
        "train_ba_at_published": fin.get("train_ba_at_published"),
        "stop_rule": stop,
        "ci_half_frac": fin.get("ci_half_frac"), "publish_rate": fin.get("publish_rate"),
        "family_gap_frac": fin.get("family_gap_frac"), "theta_P": fin.get("theta_P"), "theta_D": fin.get("theta_D"),
        "family_rule": {"source": fin.get("source"), "theta": fin.get("theta_family_rule")},
        "label_def_sha256": canonical_sha256(ver["label_def"]),
        "verdict_for_holdout": verdict_for_holdout(ver),
        "stage_files": {k: {"path": str(v), "sha256": sha256_file(v)} for k, v in stage_files.items()},
        "training_inputs": {k: {"path": str(v), "sha256": sha256_file(v)}
                            for k, v in training_input_files(fit_dir, model).items()},
        "h2": {"rows_sha256": (man_doc.get("h2") or {}).get("rows_sha256"),
               "cells_sha256": (man_doc.get("h2") or {}).get("cells_sha256"),
               "manifest_sha256": sha256_file(h2_path) if h2_path.exists() else None},
        "trainset": {"manifest_sha256": ts.get("trainset_manifest_sha256"), "sentinels": ts.get("sentinels")},
    }
    return entry, []


def stage_freeze(out_root: Path, fit_dir_of: Callable[[str], Path], models: Sequence[str], arm: str,
                 freeze_file: Path, *, command: Sequence[str] = ()) -> dict:
    """D22: freeze every model's published parameters into ``freeze_file`` (or refuse,
    raising :class:`FreezeError` with every reason, before anything is written)."""
    fp = freeze_paths(freeze_file)
    problems: list[str] = []
    for key in ("freeze", "sidecar", "result", "marker"):
        if fp[key].exists():
            problems.append(f"{fp[key]} already exists: a freeze is never overwritten")
    if len(set(models)) != len(models):
        problems.append(f"a model is given twice: {list(models)}")
    entries: dict[str, dict] = {}
    for model in models:
        entry, pr = freeze_model(out_root, fit_dir_of(model), model, arm)
        problems += [f"{model}: {x}" for x in pr]
        if entry is not None:
            entries[model] = entry
    if problems:
        raise FreezeError(problems)
    doc = {
        "what": ("D22 parameter freeze (plan §6.11): the published theta / w_p / alpha / delta of every "
                 "model, frozen before M is collected; `dline_refit accept` evaluates A-D on M once, "
                 "from this file only"),
        "format_revision": FREEZE_FORMAT_REVISION,
        "created_at_utc": _utc_now(),
        "arm": arm,
        "command": list(command),
        "code": code_state(),
        "refit_out_dir": str(out_root),
        "models": entries,
        "self_hash_rule": ("freeze_sha256 = sha256 of json.dumps(doc without freeze_sha256, sort_keys=True, "
                           "separators=(',', ':'), ensure_ascii=False) in UTF-8"),
    }
    doc["freeze_sha256"] = canonical_sha256(doc)
    data = _json_bytes(doc)
    fp["freeze"].parent.mkdir(parents=True, exist_ok=True)
    _write_once(fp["freeze"], data)
    _write_once(fp["sidecar"], f"{hashlib.sha256(data).hexdigest()}  {fp['freeze'].name}\n".encode())
    return doc


def verify_freeze(path: Path | str) -> dict:
    """The freeze document, after checking its sidecar and its embedded self hash; raises
    :class:`FreezeError` on any mismatch."""
    f = Path(path)
    side = freeze_paths(f)["sidecar"]
    if not f.exists():
        raise FreezeError([f"{f} does not exist"])
    if not side.exists():
        raise FreezeError([f"{side} is missing: the freeze file has no sha256 sidecar"])
    data = f.read_bytes()
    parts = side.read_text(encoding="utf-8").split()
    if len(parts) != 2 or len(parts[0]) != 64:
        raise FreezeError([f"{side}: not a sha256sum line ('<hex>  <name>')"])
    want, name = parts[0], parts[1].lstrip("*")
    got = hashlib.sha256(data).hexdigest()
    if name != f.name:
        raise FreezeError([f"{side} names {name!r}, not {f.name!r}"])
    if got != want:
        raise FreezeError([f"{f}: sha256 {got} != {want} in {side.name}: the freeze file changed after it "
                           "was written"])
    try:
        doc = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        raise FreezeError([f"{f}: not JSON ({exc})"])
    if not isinstance(doc, dict):
        raise FreezeError([f"{f}: not a freeze document"])
    body = {k: v for k, v in doc.items() if k != "freeze_sha256"}
    if doc.get("freeze_sha256") != canonical_sha256(body):
        raise FreezeError([f"{f}: embedded freeze_sha256 {doc.get('freeze_sha256')} does not match its content "
                           f"({canonical_sha256(body)})"])
    if doc.get("format_revision") != FREEZE_FORMAT_REVISION or not isinstance(doc.get("models"), dict):
        raise FreezeError([f"{f}: not a format revision {FREEZE_FORMAT_REVISION} freeze"])
    return doc


# ------------------------------------------------------------------------- accept


def _attempt(value: Any) -> Any:
    try:
        return int(str(value).strip())
    except ValueError:
        return str(value)


def _parse_sums_line(line: str) -> tuple[str, str]:
    """``<hex>  <path>`` (text) or ``<hex> *<path>`` (binary), sha256sum format."""
    digest, _, rest = line.partition(" ")
    rest = rest[1:] if rest.startswith((" ", "*")) else rest
    if len(digest) != 64 or not rest:
        raise ValueError(line)
    return digest, rest


def check_m_manifest(path: Path, freeze_doc: Mapping[str, Any],
                     freeze_file_sha256: str) -> tuple[Optional[dict], list[str]]:
    """An M manifest (the collection side's ``<M root>/<model>/M_manifest.json``) and every
    reason it cannot be accepted against this freeze."""
    path = Path(path)
    try:
        man = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, [f"{path}: unreadable ({exc})"]
    if not isinstance(man, dict):
        return None, [f"{path}: not a manifest"]
    missing = [k for k in M_MANIFEST_KEYS if k not in man]
    if missing:
        return None, [f"{path}: missing keys {missing}"]
    problems: list[str] = []
    model = man["model"]
    if man["format_revision"] != M_MANIFEST_FORMAT_REVISION:
        problems.append(f"format_revision {man['format_revision']!r} != {M_MANIFEST_FORMAT_REVISION}")
    entry = freeze_doc["models"].get(model)
    if entry is None:
        return None, [f"{path}: model {model!r} is not in the freeze ({sorted(freeze_doc['models'])})"]
    fr = man["freeze"] if isinstance(man["freeze"], dict) else {}
    if fr.get("sha256") != freeze_file_sha256:
        problems.append(f"sealed under another freeze: manifest freeze sha256 {fr.get('sha256')} != "
                        f"{freeze_file_sha256} (this freeze file)")
    want_label = canonical_sha256(entry["verdict_for_holdout"]["label_def"])
    if man["label_def_sha256"] != want_label:
        problems.append(f"M was sealed under another label: label_def_sha256 {man['label_def_sha256']} != "
                        f"{want_label} (the frozen label)")
    if canonical_sha256(man["label_def"]) != man["label_def_sha256"]:
        problems.append("label_def does not hash to label_def_sha256")
    sums = path.parent / str(man["sha256sums_file"])
    if not sums.exists():
        problems.append(f"{sums} (sha256sums_file) does not exist")
    elif sha256_file(sums) != man["sha256sums_sha256"]:
        problems.append(f"{sums}: sha256 {sha256_file(sums)} != sha256sums_sha256 {man['sha256sums_sha256']}")
    else:
        n = 0
        for k, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                digest, name = _parse_sums_line(line)
            except ValueError:
                problems.append(f"{sums}:{k}: not a sha256sum line")
                continue
            target = Path(name) if Path(name).is_absolute() else path.parent / name
            n += 1
            if not target.exists():
                problems.append(f"{target} (listed in {sums.name}) does not exist")
            elif sha256_file(target) != digest:
                problems.append(f"{target}: sha256 differs from {sums.name} - M data changed after it was sealed")
        if not n:
            problems.append(f"{sums} lists no file")
    cells = man["cells"]
    if not isinstance(cells, list) or not cells:
        problems.append("no cells to evaluate")
        cells = []
    seen: set = set()
    for i, c in enumerate(cells):
        miss = [k for k in M_CELL_KEYS if k not in c]
        if miss:
            problems.append(f"cells[{i}]: missing keys {miss}")
            continue
        key = (str(c["cell_id"]), _attempt(c["attempt"]))
        if c["model"] != model:
            problems.append(f"cells[{i}] {key}: model {c['model']!r} != {model!r}")
        if key in seen:
            problems.append(f"cells[{i}] {key}: listed twice")
        seen.add(key)
        if c["origin"] not in M_CELL_ORIGINS:
            problems.append(f"cells[{i}] {key}: origin {c['origin']!r} not in {M_CELL_ORIGINS}")
        if not isinstance(c["seen_before"], bool):
            problems.append(f"cells[{i}] {key}: seen_before is not a bool")
    probes = man["sealed_probes"] if isinstance(man["sealed_probes"], list) else []
    probe_keys = {(str(q.get("cell_id")), _attempt(q.get("attempt"))) for q in probes if isinstance(q, Mapping)}
    both = sorted(str(k) for k in seen & probe_keys)
    if both:
        problems.append(f"cells also listed as sealed probes: {both}")
    return man, [f"{path}: {x}" for x in problems]


def collect_m_rows(sources: Sequence[DatasetSource],
                   manifests: Mapping[str, Mapping[str, Any]]) -> tuple[dict, list[str]]:
    """The dataset rows of every manifest cell (matched on model, cell_id, attempt), in
    dataset order, per model; plus every reason they cannot be evaluated."""
    problems: list[str] = []
    wanted: dict[tuple, Mapping[str, Any]] = {}
    for model, man in manifests.items():
        for c in man["cells"]:
            wanted[(model, str(c["cell_id"]), _attempt(c["attempt"]))] = c
    header: dict[str, list[str]] = {}
    rows: dict[str, list[dict]] = defaultdict(list)
    where: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for src in sources:
        with open(src.windows, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            head = next(reader, None) or []
            miss = [c for c in M_REQUIRED_COLUMNS if c not in head]
            if miss:
                problems.append(f"{src.windows}: missing columns {miss}")
                continue
            for values in reader:
                row = dict(zip(head, values))
                key = (row["model"], row["cell_id"], _attempt(row["attempt"]))
                if key not in wanted:
                    continue
                h = header.setdefault(key[0], [])
                h += [c for c in head if c not in h]
                rows[key[0]].append(row)
                where[key][src.name].append(row)
    placed: dict[tuple, dict] = {}
    scenario_of: dict[tuple[str, str], tuple] = {}
    for key, cell in wanted.items():
        tag = f"{key[0]}/{key[1]} a{key[2]}"
        found = where.get(key) or {}
        if not found:
            problems.append(f"{tag}: not found in any dataset")
            continue
        if len(found) > 1:
            problems.append(f"{tag}: found in {len(found)} datasets ({sorted(found)})")
            continue
        (name, cell_rows), = found.items()
        bad_split = sorted({r["split"] for r in cell_rows} - {SPLIT_HOLDOUT})
        if bad_split:
            problems.append(f"{tag}: rows of split {bad_split} - M is the {SPLIT_HOLDOUT} split only")
        bad_status = sorted({r["cell_status"] for r in cell_rows} - {CELL_STATUS_VALID})
        if bad_status:
            problems.append(f"{tag}: cell_status {bad_status}, not {CELL_STATUS_VALID!r}")
        for col in ("shape", "primitive", "role"):
            have = sorted({r[col] for r in cell_rows})
            if have != [str(cell[col])]:
                problems.append(f"{tag}: dataset {col} {have} != manifest {cell[col]!r}")
        sids = sorted({r["scenario_id"] for r in cell_rows})
        if len(sids) != 1:
            problems.append(f"{tag}: rows carry scenario ids {sids} (one cell, one scenario id)")
        else:
            other = scenario_of.setdefault((key[0], sids[0]), key)
            if other != key:
                problems.append(f"{tag}: shares scenario id {sids[0]} with {other} (dwell and the cell "
                                "bootstrap group by scenario id)")
        placed[key] = {"dataset": name, "rows": len(cell_rows), "scenario_id": sids[0] if sids else None}
    return {"header": header, "rows": dict(rows), "placed": placed}, problems


def _ci95(values: Sequence[float]) -> list[Optional[float]]:
    """95 % percentile interval, the index rule of ``stage_final``'s M BA CI."""
    v = sorted(values)
    if not v:
        return [None, None]
    return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]


def _finite_or_none(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _rate_metrics() -> dict[str, Callable[[Any], bool]]:
    """The rate metrics of the cell bootstrap and the windows each is a rate over - the
    selections of ``theta_verdict.dwell_acceptance``."""
    from scripts import theta_verdict as tv

    b = frozenset(tv.CRITERION_B_CLASSES)
    return {
        "critical_recall_both_tpot": lambda w: not w.slo_met and w.violation_class in b,
        "critical_false_alarm_on_healthy": lambda w: w.slo_met,
        "critical_recall_of_violating": lambda w: not w.slo_met,
        "critical_recall_ttft_only": lambda w: not w.slo_met and w.violation_class == "ttft_only",
    }


def acceptance_bootstrap(windows: Sequence[Any], crit: Sequence[bool], *, theta: float, direction: str,
                         n_resamples: int = ACCEPT_RESAMPLES, seed: int = SEED) -> dict:
    """Cell bootstrap (cells = scenario ids, drawn with replacement) of the A-C metrics.

    The dwell flags ``crit`` are computed once per cell on the full data
    (``theta_verdict.critical_dwell_flags``) and scored on the resampled cells; BA at
    ``theta`` is ``threshold_balanced_accuracy`` on the resampled windows, and a resample
    holding only one class has no BA and is skipped (``resamples_used`` counts the rest)."""
    from tre_calibration.fit import threshold_balanced_accuracy

    metrics = _rate_metrics()
    by: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(windows):
        by[w.scenario_id].append(i)
    cells = sorted(by)
    counts = {c: {k: (sum(1 for i in by[c] if f(windows[i]) and crit[i]), sum(1 for i in by[c] if f(windows[i])))
                  for k, f in metrics.items()} for c in cells}
    values: dict[str, list[float]] = {"balanced_accuracy": [], **{k: [] for k in metrics}}
    rng = random.Random(seed)
    for _ in range(n_resamples if cells else 0):
        pick = [rng.choice(cells) for _ in cells]
        smp = [windows[i] for c in pick for i in by[c]]
        if any(w.slo_met for w in smp) and any(not w.slo_met for w in smp):
            values["balanced_accuracy"].append(
                threshold_balanced_accuracy(smp, theta=theta, direction=direction)["balanced_accuracy"])
        for k in metrics:
            den = sum(counts[c][k][1] for c in pick)
            if den:
                values[k].append(sum(counts[c][k][0] for c in pick) / den)
    return {
        "n_resamples": n_resamples, "seed": seed, "unit": "cell (scenario_id), drawn with replacement",
        "interval": "95 % percentile", "cells": len(cells),
        "metrics": {k: {"ci95": _ci95(v), "resamples_used": len(v)} for k, v in values.items()},
    }


def _criterion(name: str, value: Any, op: str, threshold: Any) -> dict:
    v, t = _finite_or_none(value), _finite_or_none(threshold)
    met = v is not None and t is not None and (v >= t if op == ">=" else v <= t)
    return {"name": name, "value": v, "op": op, "threshold": t, "met": met}


def acceptance_criteria(entry: Mapping[str, Any], h: Mapping[str, Any], boot: Mapping[str, Any]) -> dict:
    """Plan §6.9f A-D for one model from its freeze entry, hold-out report ``h`` and
    bootstrap ``boot``. A is not evaluable (and fails) when M lacks one of the classes."""
    ci = {k: v["ci95"] for k, v in boot["metrics"].items()}
    wd = h["with_dwell"]
    two_classes = 0 < h["violating"] < h["windows"]
    ba = h["at_published_theta"]["balanced_accuracy"] if two_classes else None
    train = _finite_or_none(entry.get("train_ba_at_published"))
    a = [_criterion("BA at the published theta", ba, ">=", A_BA_MIN),
         _criterion("BA CI95 lower bound", ci["balanced_accuracy"][0], ">=", A_BA_CI_LOW_MIN),
         _criterion(f"BA >= training BA at the published theta - {A_MAX_DROP_FROM_TRAINING}", ba, ">=",
                    None if train is None else train - A_MAX_DROP_FROM_TRAINING)]
    rec, fa = wd["critical_recall_both_tpot"], wd["critical_false_alarm_on_healthy"]
    b = [_criterion("CRITICAL recall of both/TPOT-only violations (dwell)", rec, ">=", B_RECALL_MIN),
         _criterion("its CI95 lower bound", ci["critical_recall_both_tpot"][0], ">=", B_RECALL_CI_LOW_MIN),
         _criterion("CRITICAL false alarm on healthy windows (dwell)", fa, "<=", B_FALSE_ALARM_MAX),
         _criterion("its CI95 upper bound", ci["critical_false_alarm_on_healthy"][1], "<=",
                    B_FALSE_ALARM_CI_HIGH_MAX)]
    b_eval = bool(wd["both_tpot_windows"]) and bool(wd["healthy_windows"])
    all_rec = _finite_or_none(wd["critical_recall_of_violating"])
    ttft = wd["violation_classes"]["ttft_only"]
    stop = entry.get("stop_rule") or {}
    gap, half = _finite_or_none(entry.get("family_gap_frac")), _finite_or_none(entry.get("ci_half_frac"))
    return {
        "A": {"criteria": a, "evaluable": ba is not None, "passed": all(c["met"] for c in a),
              "train_ba_at_published": train},
        "B": {"criteria": b, "evaluable": b_eval, "passed": b_eval and all(c["met"] for c in b),
              "both_tpot_windows": wd["both_tpot_windows"], "healthy_windows": wd["healthy_windows"],
              "dwell_windows": wd["dwell_windows"],
              "all_violating_recall": {"value": all_rec, "target": ALL_VIOLATING_RECALL_TARGET,
                                       "met": all_rec is not None and all_rec >= ALL_VIOLATING_RECALL_TARGET,
                                       "ci95": ci["critical_recall_of_violating"], "gating": False}},
        "C": {"gating": False, "critical_recall_ttft_only": ttft["critical_recall"], "windows": ttft["windows"],
              "independent_windows": ttft["windows"] / WINDOWS_PER_INDEPENDENT,
              "ci95": ci["critical_recall_ttft_only"]},
        "D": {"passed": stop.get("satisfied") is True, "stop_rule_satisfied": stop.get("satisfied"),
              "reasons": stop.get("reasons"), "ci_half_frac": half, "publish_rate": entry.get("publish_rate"),
              "family_gap_frac": gap, "theta_P": entry.get("theta_P"), "theta_D": entry.get("theta_D"),
              "family_gap_within_ci_half_width": (gap <= half) if gap is not None and half is not None else None,
              "source": "the training stop rule (D13) as recorded at freeze time"},
    }


def _write_validation_csv(path: Path, header: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(header))
        for r in rows:
            w.writerow([r.get(c, "") for c in header])


def evaluate_model(entry: Mapping[str, Any], csv_path: Path, *, n_resamples: int, seed: int) -> dict:
    """``theta_verdict.holdout_report`` for the point estimates (dwell 2), the cell
    bootstrap for the CIs, then A-D."""
    from scripts import theta_verdict as tv

    vh = entry["verdict_for_holdout"]
    h = tv.holdout_report(vh, csv_path, dwell_windows=ACCEPT_DWELL_WINDOWS)
    spec = tv.SignalSpec.from_dict(vh["signal_spec"])
    label = slo_labels.LabelDefinition.from_dict(vh["label_def"])
    windows = spec.load(csv_path, label, int(vh["trim_ramp_windows"]))
    theta, tau_crit = float(vh["published"]["theta_m"]), float(vh["published"]["tau_crit"])
    direction = vh["fit_config"]["direction"]
    crit = tv.critical_dwell_flags(windows, theta=theta, tau_crit=tau_crit, direction=direction,
                                   dwell_windows=ACCEPT_DWELL_WINDOWS)
    boot = acceptance_bootstrap(windows, crit, theta=theta, direction=direction,
                                n_resamples=n_resamples, seed=seed)
    per_cell: dict[str, int] = defaultdict(int)
    for w in windows:
        per_cell[w.scenario_id] += 1
    criteria = acceptance_criteria(entry, h, boot)
    return {"holdout_report": h, "bootstrap": boot, "criteria": criteria,
            "passed": criteria["A"]["passed"] and criteria["B"]["passed"] and criteria["D"]["passed"],
            "M": {"windows": h["windows"], "cells": h["cells"], "violating": h["violating"],
                  "violating_fraction": (h["violating"] / h["windows"]) if h["windows"] else None,
                  "cell_windows": dict(sorted(per_cell.items()))}}


def _accept_inputs(freeze_file: Path, datasets: Sequence[str],
                   m_manifests: Sequence[str]) -> tuple[dict, list[str]]:
    """Every refusal check of ``accept`` but the once-only one; the inputs when none fails."""
    problems: list[str] = []
    try:
        doc = verify_freeze(freeze_file)
    except FreezeError as exc:
        return {}, exc.problems
    freeze_sha = sha256_file(Path(freeze_file))
    for model, entry in sorted(doc["models"].items()):
        for key, rec in sorted(entry["training_inputs"].items()):
            p = Path(rec["path"])
            if not p.exists():
                problems.append(f"{model}: training input {key} {p} is missing")
            elif sha256_file(p) != rec["sha256"]:
                problems.append(f"{model}: training input {key} {p} changed after the freeze")
    manifests: dict[str, dict] = {}
    manifest_paths: dict[str, Path] = {}
    unreadable = False
    for text in m_manifests:
        man, pr = check_m_manifest(Path(text), doc, freeze_sha)
        problems += pr
        if man is None:
            unreadable = True
            continue
        if man["model"] in manifests:
            problems.append(f"{man['model']}: two M manifests ({manifest_paths[man['model']]}, {text})")
            continue
        manifests[man["model"]] = man
        manifest_paths[man["model"]] = Path(text)
    if not unreadable:
        problems += [f"{m}: no M manifest (--m-manifest)" for m in sorted(doc["models"]) if m not in manifests]
    sources: list[DatasetSource] = []
    for text in datasets:
        try:
            sources.append(DatasetSource.parse(text, sealed_to_h2=False))
        except TrainingSetError as exc:
            problems.append(f"dataset {text}: {exc}")
    if not datasets:
        problems.append("no --dataset given")
    if len({s.name for s in sources}) != len(sources):
        problems.append(f"two datasets share a run name: {[s.name for s in sources]}")
    m: dict = {"header": {}, "rows": {}, "placed": {}}
    if not problems:
        m, pr = collect_m_rows(sources, manifests)
        problems += pr
    return {"doc": doc, "freeze_sha256": freeze_sha, "manifests": manifests, "manifest_paths": manifest_paths,
            "sources": sources, "m": m}, problems


def _accept_result(freeze_file: Path, inp: Mapping[str, Any], work: Path, *, n_resamples: int, seed: int,
                   command: Sequence[str]) -> dict:
    doc, m = inp["doc"], inp["m"]
    sums_cover: set[str] = set()
    for model, man in inp["manifests"].items():
        mdir = inp["manifest_paths"][model].parent
        for line in (mdir / str(man["sha256sums_file"])).read_text(encoding="utf-8").splitlines():
            if line.strip():
                name = _parse_sums_line(line)[1]
                sums_cover.add(str(Path(name) if Path(name).is_absolute() else mdir / name))
    datasets = [{"name": s.name, "directory": str(s.directory), "windows_csv": str(s.windows),
                 "windows_csv_sha256": sha256_file(s.windows),
                 "covered_by_m_sha256sums": str(s.windows) in sums_cover}
                for s in inp["sources"]]
    models: dict[str, Any] = {}
    for model in sorted(doc["models"]):
        entry, man = doc["models"][model], inp["manifests"][model]
        mpath = inp["manifest_paths"][model]
        csv_path = work / f"{model}_validation.csv"
        _write_validation_csv(csv_path, m["header"][model], m["rows"][model])
        ev = evaluate_model(entry, csv_path, n_resamples=n_resamples, seed=seed)
        cells = []
        for c in man["cells"]:
            placed = m["placed"][(model, str(c["cell_id"]), _attempt(c["attempt"]))]
            cells.append({**{k: c[k] for k in M_CELL_KEYS}, **placed})
        ev["M"].update({
            "manifest_cells": cells, "sealed_probes_not_evaluated": len(man["sealed_probes"]),
            "seen_before": [{k: c[k] for k in ("cell_id", "attempt", "origin", "seen_before", "note")}
                            for c in cells if c["seen_before"] or c["origin"] == "retained"],
        })
        models[model] = {
            "m_manifest": {"path": str(mpath), "sha256": sha256_file(mpath),
                           "sha256sums_file": str(mpath.parent / str(man["sha256sums_file"])),
                           "sha256sums_sha256": man["sha256sums_sha256"],
                           "label_def_sha256": man["label_def_sha256"]},
            "validation_csv": str(csv_path), "validation_csv_sha256": sha256_file(csv_path),
            "validation_rows": len(m["rows"][model]),
            "published": entry["published"],
            **ev,
        }
    failed = []
    for model, r in models.items():
        for g in ("A", "B", "D"):
            crit = r["criteria"][g]
            if crit["passed"]:
                continue
            if g == "D":
                why = [str(x) for x in (crit["reasons"] or ["stop rule not satisfied"])]
            else:
                why = [f"{c['name']} {c['value']} {c['op']} {c['threshold']} not met"
                       for c in crit["criteria"] if not c["met"]]
                if not crit["evaluable"]:
                    why.insert(0, "not evaluable on this M")
            failed.append(f"{model}: {g} failed - " + "; ".join(why))
    return {
        "what": ("plan §6.9f acceptance A-D on M, evaluated once on the frozen parameters "
                 "(A, B, D gate; C and the all-violating recall are disclosed)"),
        "format_revision": 1,
        "evaluated_at_utc": _utc_now(),
        "command": list(command),
        "code": code_state(),
        "freeze": {"path": str(freeze_file), "sha256": inp["freeze_sha256"],
                   "freeze_sha256": doc["freeze_sha256"], "arm": doc.get("arm")},
        "thresholds": {"A": {"ba_min": A_BA_MIN, "ba_ci_low_min": A_BA_CI_LOW_MIN,
                             "max_drop_from_training": A_MAX_DROP_FROM_TRAINING},
                       "B": {"recall_min": B_RECALL_MIN, "recall_ci_low_min": B_RECALL_CI_LOW_MIN,
                             "false_alarm_max": B_FALSE_ALARM_MAX,
                             "false_alarm_ci_high_max": B_FALSE_ALARM_CI_HIGH_MAX,
                             "all_violating_recall_target": ALL_VIOLATING_RECALL_TARGET},
                       "C": {"windows_per_independent": WINDOWS_PER_INDEPENDENT},
                       "dwell_windows": ACCEPT_DWELL_WINDOWS},
        "bootstrap": {"n_resamples": n_resamples, "seed": seed},
        "datasets": datasets,
        "work_dir": str(work),
        "models": models,
        "passed": not failed,
        "failed": failed,
    }


def result_differences(stored: Any, recomputed: Any, *, ignore: frozenset = ACCEPT_VOLATILE_KEYS,
                       where: str = "") -> list[str]:
    """Every difference between two accept results, keys in ``ignore`` skipped at any depth."""
    if isinstance(stored, dict) and isinstance(recomputed, dict):
        out = []
        for k in sorted(set(stored) | set(recomputed), key=str):
            if k in ignore:
                continue
            at = f"{where}.{k}" if where else str(k)
            if k not in stored:
                out.append(f"{at}: only in the recomputed result")
            elif k not in recomputed:
                out.append(f"{at}: only in the stored result")
            else:
                out += result_differences(stored[k], recomputed[k], ignore=ignore, where=at)
        return out
    if isinstance(stored, list) and isinstance(recomputed, list):
        if len(stored) != len(recomputed):
            return [f"{where}: {len(stored)} items stored, {len(recomputed)} recomputed"]
        out = []
        for i, (a, b) in enumerate(zip(stored, recomputed)):
            out += result_differences(a, b, ignore=ignore, where=f"{where}[{i}]")
        return out
    both_nan = (isinstance(stored, float) and isinstance(recomputed, float)
                and math.isnan(stored) and math.isnan(recomputed))
    if (stored == recomputed and type(stored) is type(recomputed)) or both_nan:
        return []
    return [f"{where}: stored {stored!r} != recomputed {recomputed!r}"]


def stage_accept(freeze_file: Path, datasets: Sequence[str], m_manifests: Sequence[str], *,
                 recheck: bool = False, n_resamples: int = ACCEPT_RESAMPLES,
                 command: Sequence[str] = ()) -> int:
    """Plan §6.9f A-D, once. Returns 0 = evaluated and passed, :data:`EXIT_ACCEPT_FAILED`
    = evaluated and failed, :data:`EXIT_REFUSED` = refused (nothing written); with
    ``recheck``: 0 = the stored result reproduces, :data:`EXIT_RECHECK_DIFFERS` = not."""
    import shutil
    import tempfile

    freeze_file = Path(freeze_file)
    fp = freeze_paths(freeze_file)
    problems: list[str] = []
    if recheck:
        if not fp["result"].exists():
            problems.append(f"{fp['result']} does not exist: nothing to recheck")
    else:
        problems += [f"{fp[k]} already exists: M is evaluated once (--recheck reproduces it)"
                     for k in ("result", "marker", "work") if fp[k].exists()]
    inp, pr = _accept_inputs(freeze_file, datasets, m_manifests)
    problems += pr
    if problems:
        print("accept REFUSED - nothing was written:")
        for x in problems:
            print(f"  - {x}")
        return EXIT_REFUSED
    if recheck:
        stored_bytes = fp["result"].read_bytes()
        stored = json.loads(stored_bytes.decode("utf-8"))
        n_resamples, seed = int(stored["bootstrap"]["n_resamples"]), int(stored["bootstrap"]["seed"])
        with tempfile.TemporaryDirectory(prefix="dline_accept_recheck_") as tmp:
            new = _accept_result(freeze_file, inp, Path(tmp), n_resamples=n_resamples, seed=seed, command=command)
        new = json.loads(_json_bytes(new).decode("utf-8"))
        diffs = result_differences(stored, new)
        try:
            marker = json.loads(fp["marker"].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            marker = {}
        if marker.get("result_sha256") != hashlib.sha256(stored_bytes).hexdigest():
            diffs.insert(0, f"{fp['marker']}: missing, or not the marker of the stored result's bytes")
        if stored.get("code") != new.get("code"):
            print(f"note: code state differs (stored {stored.get('code')}, now {new.get('code')}) - not compared")
        if diffs:
            print(f"recheck: {len(diffs)} difference(s) from {fp['result']}:")
            for d in diffs:
                print(f"  - {d}")
            return EXIT_RECHECK_DIFFERS
        print(f"recheck: identical to {fp['result']} (not compared: {sorted(ACCEPT_VOLATILE_KEYS)})")
        return 0
    fp["work"].mkdir()
    try:
        result = _accept_result(freeze_file, inp, fp["work"], n_resamples=n_resamples, seed=SEED, command=command)
        for f in fp["work"].iterdir():
            os.chmod(f, 0o444)
        data = _json_bytes(result)
        _write_once(fp["result"], data)
    except BaseException:
        shutil.rmtree(fp["work"], ignore_errors=True)
        raise
    _write_once(fp["marker"], _json_bytes({"result": str(fp["result"]),
                                           "result_sha256": hashlib.sha256(data).hexdigest(),
                                           "accepted_at_utc": _utc_now()}))
    for model, r in result["models"].items():
        c = r["criteria"]
        print(f"[{model}] M {r['M']['windows']} windows / {r['M']['cells']} cells: "
              + " ".join(f"{g}={'pass' if c[g]['passed'] else 'FAIL'}" for g in ("A", "B", "D"))
              + f" (C, disclosed: TTFT-only recall {c['C']['critical_recall_ttft_only']} "
                f"on {c['C']['windows']} windows)")
    print(f"wrote {fp['result']} and {fp['marker']}")
    if not result["passed"]:
        print("acceptance FAILED:")
        for x in result["failed"]:
            print(f"  - {x}")
        return EXIT_ACCEPT_FAILED
    print("acceptance passed")
    return 0


# ---------------------------------------------------------------------------- CLI


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} is missing: run the previous stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["trainset", "alpha", "wp", "final", "summary", "freeze", "verify-freeze",
                                      "accept"])
    ap.add_argument("--model", action="append", default=[],
                    help="model (repeatable for summary; trainset: restrict the training CSVs written)")
    ap.add_argument("--arm", choices=ARMS, default="primary")
    ap.add_argument("--fit-dir", type=Path, default=None,
                    help="directory of the training CSVs (<model>_fitting.csv ...; the trainset stage "
                         "writes them); for several models a template with {model}, e.g. /r/{model}/fit/fit")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--dataset", action="append", default=[], metavar="[RUN=]DIR",
                    help="trainset: a standard dataset (calibration_dataset) to cut the training set "
                         "from; its sealed split is M and is skipped unread. accept: a dataset holding M")
    ap.add_argument("--h2-dataset", action="append", default=[], metavar="[RUN=]DIR",
                    help="trainset: a standard dataset whose sealed split joins H2 (D16: run 1 and run 2); "
                         "its constant-load train / auxiliary cells still train")
    ap.add_argument("--no-sentinels", action="store_true",
                    help="trainset: leave the sentinel cells out of the training set (default: they train)")
    ap.add_argument("--alpha-rule", choices=ALPHA_RULES, default="d4prime")
    ap.add_argument("--lambda-method", choices=LAMBDA_METHODS, default="v2",
                    help="wp: v2 (default) = D17 w_p + the BA lambda check; v1 = lambda_wait and w_p "
                         "from v1's selection (scripts.v1_lambda_fit, user 2026-09-24)")
    ap.add_argument("--requests-dataset", action="append", default=[], metavar="RUN=DIR",
                    help="wp --lambda-method v1: a standard dataset whose requests.csv rebuilds the "
                         "average TPOT of rows whose run column is RUN (default: the fit dir's "
                         f"{TRAINSET_MANIFEST} sources; repeatable, overrides)")
    ap.add_argument("--alpha-w-p", type=float, default=None,
                    help=f"w_p of the alpha stage (default: {ALPHA_STAGE_W_P})")
    ap.add_argument("--alpha-bootstrap", type=int, default=1000,
                    help="whole-rule bootstrap resamples of the d4prime alpha rule")
    ap.add_argument("--publish-tau-s", type=publish_tau_arg, default=PUBLISH_TAU_S,
                    help=f"alpha: the tau w_p / theta / delta are fitted at and that deploys (D18, default "
                         f"{PUBLISH_TAU_S:g}); 'rule' publishes the alpha rule's own pick")
    ap.add_argument("--ledger", action="append", default=[],
                    help="cells.jsonl of a ladder-design run (steady cells for D4', the summary and the "
                         f"training-row check); default: the fit dir's {TRAINING_LEDGER}")
    ap.add_argument("--registry", default=None, help="registry the label's idle TTFT fit is read from")
    ap.add_argument("--no-holdout", action="store_true",
                    help="final: stop after the verdict; never open <model>_validation.csv (M is "
                         "evaluated once, after it is frozen - plan §6.11 note 9)")
    ap.add_argument("--freeze-file", type=Path, default=None,
                    help="freeze: the parameter freeze to write; verify-freeze / accept: the one to read")
    ap.add_argument("--m-manifest", action="append", default=[],
                    help="accept: an M manifest (<M root>/<model>/M_manifest.json), one per frozen model")
    ap.add_argument("--recheck", action="store_true",
                    help="accept: recompute in a temp dir and compare with the stored result; writes nothing")
    ap.add_argument("--accept-resamples", type=int, default=ACCEPT_RESAMPLES,
                    help="accept: cell-bootstrap resamples of the A-C intervals")
    args = ap.parse_args(argv)
    command = ["python", "-m", "scripts.dline_refit", *(sys.argv[1:] if argv is None else argv)]

    if args.stage in ("verify-freeze", "accept"):
        if args.freeze_file is None:
            ap.error(f"{args.stage} needs --freeze-file")
        if args.stage == "accept":
            return stage_accept(args.freeze_file, args.dataset, args.m_manifest, recheck=args.recheck,
                                n_resamples=args.accept_resamples, command=command)
        try:
            doc = verify_freeze(args.freeze_file)
        except FreezeError as exc:
            print(f"verify-freeze REFUSED: {exc}")
            return EXIT_REFUSED
        print(f"OK {sha256_file(args.freeze_file)}  {args.freeze_file} (freeze_sha256 {doc['freeze_sha256']}, "
              f"models {sorted(doc['models'])})")
        return 0
    if args.fit_dir is None:
        ap.error(f"stage {args.stage} needs --fit-dir")

    def fit_dir(model: str) -> Path:
        text = str(args.fit_dir)
        return Path(text.format(model=model)) if "{model}" in text else args.fit_dir

    if args.stage == "trainset":
        if "{model}" in str(args.fit_dir):
            ap.error("trainset writes every model into one --fit-dir (no {model} template)")
        try:
            sources = ([DatasetSource.parse(t, sealed_to_h2=True) for t in args.h2_dataset]
                       + [DatasetSource.parse(t, sealed_to_h2=False) for t in args.dataset])
        except TrainingSetError as exc:
            ap.error(str(exc))
        if not sources:
            ap.error("trainset needs --dataset and / or --h2-dataset")
        try:
            doc = build_training_set(sources, args.fit_dir, sentinels=not args.no_sentinels,
                                     models=args.model or None)
        except TrainingSetError as exc:
            raise SystemExit(f"trainset: {exc}")
        print(json.dumps({"models": {m: {k: v[k] for k in ("windows", "cells", "family_windows")}
                                     for m, v in doc["models"].items()},
                          "h2_rows_sha256": doc["h2"]["rows_sha256"], "m_unread": doc["m_unread"]}, indent=1))
        print(f"wrote {args.fit_dir / TRAINSET_MANIFEST} and {args.fit_dir / H2_MANIFEST}")
        return 0
    if not args.model:
        ap.error(f"stage {args.stage} needs --model")
    if args.out_dir is None:
        ap.error(f"stage {args.stage} needs --out-dir")

    from scripts.rewindow_from_raw import load_ledgers

    def ledger_paths(models: Iterable[str]) -> list[str]:
        if args.ledger:
            return list(args.ledger)
        own = {fit_dir(m) / TRAINING_LEDGER for m in models}
        return [str(q) for q in sorted(own) if q.exists()]

    if args.stage == "freeze":
        if args.freeze_file is None:
            ap.error("freeze needs --freeze-file")
        try:
            doc = stage_freeze(args.out_dir, fit_dir, args.model, args.arm, args.freeze_file, command=command)
        except FreezeError as exc:
            print(f"freeze REFUSED - nothing was written ({len(exc.problems)} problem(s)):")
            for x in exc.problems:
                print(f"  - {x}")
            return EXIT_REFUSED
        for model, e in doc["models"].items():
            q = e["published"]
            print(f"[{model}] theta={q['theta']:.6g} w_p={q['w_p']:g} tau_s={q['tau_s']:g} "
                  f"lambda_wait={q['lambda_wait']:g} delta_crit={q['delta_crit']:g} delta_high={q['delta_high']:g}")
        print(f"wrote {args.freeze_file} (freeze_sha256 {doc['freeze_sha256']}) and "
              f"{freeze_paths(args.freeze_file)['sidecar']}")
        return 0
    if args.stage == "summary":
        lp = ledger_paths(args.model)
        doc = stage_summary(args.out_dir, {m: fit_dir(m) for m in args.model},
                            registry=args.registry, ledger=load_ledgers(lp) if lp else None)
        (args.out_dir / "summary.json").write_text(json.dumps(doc, indent=1, default=str))
        print(f"wrote {args.out_dir / 'summary.json'}")
        return 0
    if len(args.model) != 1:
        ap.error(f"stage {args.stage} takes one --model")
    model = args.model[0]
    label = label_for(model, args.arm, args.registry)
    p = paths(fit_dir(model), model)
    lp = ledger_paths([model])
    # D16: nothing but constant-load training rows reaches a fit (H2 / M cut at the entry)
    training = check_training_inputs(model, p, ledger=load_ledgers(lp) if lp else None)
    out = args.out_dir / model / args.arm
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "alpha":
        w_p = args.alpha_w_p if args.alpha_w_p is not None else ALPHA_STAGE_W_P.get(model)
        if w_p is None:
            ap.error(f"no alpha-stage w_p for {model}: pass --alpha-w-p")
        if args.alpha_rule == "refit0922":
            doc = stage_alpha_refit0922(model, label, p, w_p=w_p)
        else:
            doc = stage_alpha_d4prime(model, label, p, w_p=w_p, ledgers=lp,
                                      bootstrap=args.alpha_bootstrap, registry=args.registry)
        doc = publish_alpha(doc, args.publish_tau_s)
    elif args.stage == "wp" and args.lambda_method == "v1":
        from scripts import v1_lambda_fit

        sources = v1_lambda_fit.sources_from_trainset(fit_dir(model))
        for text in args.requests_dataset:
            run, sep, d = text.partition("=")
            if not sep or not run or not d:
                ap.error(f"--requests-dataset {text!r}: expected RUN=DIR")
            sources[run] = Path(d)
        doc = stage_wp_v1(model, label, p, _read_json(out / "alpha.json"), sources)
    elif args.stage == "wp":
        doc = stage_wp(model, label, p, _read_json(out / "alpha.json"))
    else:
        doc = stage_final(model, label, p, _read_json(out / "wp.json"), out, holdout=not args.no_holdout)
    inputs = {k: str(v) for k, v in p.items() if k != "families"}
    if args.stage == "final" and args.no_holdout:
        inputs["validation"] = HOLDOUT_SKIPPED
    doc.update({"model": model, "arm": args.arm, "label_def": label.as_dict(), "training_set": training,
                "inputs": inputs | {f"family_{k}": str(v) for k, v in p["families"].items()}})
    (out / f"{args.stage}.json").write_text(json.dumps(doc, indent=1, default=str))
    print(f"wrote {out / (args.stage + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

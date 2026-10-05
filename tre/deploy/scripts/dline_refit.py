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
    ``--train-dynamic`` (next round, design 2026-10-05 item 2, cancels D16's "constant load
    only"): the TRAIN-split dynamic cells (steps / ramp / bursts; run 1's) train too, pooled
    with the holds at equal weight per window; the sealed split (the run-2 ramps) stays H2.
    The flag is recorded in trainset.json (``dynamic_training``) and the fit stages accept
    dynamic rows only from a training set that records it. Sources of two label
    attributions (completion / hybrid, the dataset manifest's ``attribution``) are refused,
    like two numerators; the attribution is recorded and the fit stages label with it.
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
    training windows; the D17 w_p rule is neither applied nor computed (wp.json records
    ``d3_rule.statement`` "not applied"; no D17 w_p is reported next to v1's). tau, theta,
    delta and the labels stay v2.
``final`` (D5 + hold-out)
    the verdict at (tau, w_p*, lambda*) with 1000 / 200 resamples; D5: the merged theta is
    published whatever the family rule says (the family theta is kept as diagnostic);
    then the hold-out report on the validation CSV (``theta_verdict.holdout_report``,
    dwell 2), the M balanced-accuracy CI (cell bootstrap) and the per-prompt-length
    attainment of the TTFT SLO on non-overlapping 30 s tiles. ``--no-holdout`` stops
    after the verdict and never opens the validation CSV: M is evaluated exactly once,
    after it is frozen and hashed (plan §6.11 note 9), so every refit before that runs
    without it. 2026-10-05: theta stays the BA-argmax candidate (exact ties: the fit's
    existing deterministic rule, :data:`THETA_SELECTION_RULE`; no midpoint rule); recorded
    beside it are the theta uncertainty band - the fit's own candidates within
    :data:`PLATEAU_BA_TOLERANCE` BA of the best (``theta_selection.plateau``) - and the
    absolute thresholds theta * tau_crit / theta * tau_high (``absolute_thresholds``).
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
    the training inputs it was fitted on, and the freeze file is new. Format revision 3
    (2026-10-03) also seals each model's B' severity cut (``b_prime``: the .65 quantile of
    the TRAINING violating windows' severity, :mod:`scripts.b_prime`) and the B' gate
    (``b_prime_gate``), so the cut exists before M / T14 is collected. Revision 4
    (2026-10-05) also seals the accept gate (``accept_gate``: always the onset gate - one gate, no fork;
    accept refuses a sealed gate that differs from the running code's) with every onset-gate
    parameter, refuses a tau other than 10 s (the lag budget's base), each model's theta selection
    plateau (uncertainty band), absolute thresholds, label attribution, dynamic-training record and
    the A dead band's s0 (training label noise, :func:`scripts.b_prime.label_noise_s0`).
``verify-freeze``
    checks the sidecar and the self hash (:func:`verify_freeze`); exit 0 = intact.
``accept`` (plan §6.9f A-D, once)
    ``--freeze-file``, ``--dataset [RUN=]DIR`` (repeatable) and one ``--m-manifest`` per
    frozen model; reads nothing but those. Refuses on a broken freeze, a training input
    changed since the freeze, an M manifest sealed under another freeze or label or whose
    raw-data sums moved, a manifest cell missing / duplicated / not ``holdout`` / not
    ``valid`` in the datasets. Scores the manifest cells with ``holdout_report`` (the
    freeze's dwell / window length; 2 x 30 s for a revision-1 freeze) plus a cell bootstrap
    for the CIs, and discloses - never gates - the ranking metrics of pressure = -Z
    (AUROC, Kendall tau-b per model / pooled / cross-model, ``tre_calibration.ranking``,
    docs/design/20260930-ranking-metrics.md); writes ``<stem>.accept.json`` (0444), the
    validation CSVs under ``<stem>.accept.d/`` and the marker ``PATH.accepted``.
    The gate is A, B' and D (user 2026-10-03; :mod:`scripts.b_prime`): B' is judged at
    ``--dwell-windows`` (default :data:`ONLINE_DWELL_WINDOWS` = the controller's
    ``TRE_DWELL_WINDOWS``), dwell 1 and 2 are disclosed next to it; the old B (both /
    TPOT-only recall at the freeze's dwell) is disclosed, no longer gating. The severity
    cut comes from the freeze (revision 3) or, for an older freeze, from an explicit
    ``--b-prime-thresholds FILE`` - never silently from anywhere else. Exit 0 =
    A, B' and D pass for every model, 3 = evaluated and failed, other = refused. Runs once;
    ``--recheck`` recomputes in a temp dir and compares, writing nothing (a stored result
    of an older format revision is compared without the keys added since).
    Revision 4 (2026-10-05, design item 3, user decisions 2): under a freeze whose
    ``accept_gate.rule`` is ``onset`` the gate is (i) the onset episodes of the dynamic M
    cells caught within the lag budget (:func:`scripts.b_prime.onset_detection`), (ii) the
    window false alarm, (iii) A and (iv) D, and each model gets a three-way ``verdict``:
    ``pass`` (all four), ``pass_a_disclosed`` (only A fails: go-live allowed, A disclosed as
    a limitation) or ``fail``; B', the old B, dwell 2 and the A dead band are disclosed.
    Under an older freeze the gate stays A, B', D and the onset-gate fields are added
    beside it as a disclosure. A dataset whose label attribution differs from the frozen
    label's is refused.

Windowing: ``--window-ms`` / ``--step-ms`` / ``--dt-ref-s`` / ``--horizon-ms`` /
``--dwell-windows`` (defaults 30 s / 10 s / 10 s / 30 s / 2) are threaded through alpha,
wp and final, recorded as ``windowing`` in their outputs and in the freeze (revision 2),
and accept reads them from the freeze - except ``--dwell-windows``, which for accept is
the B' gate's dwell (default :data:`ONLINE_DWELL_WINDOWS`).

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
#: The D13 CI gate every verdict is judged against (``--max-ci-half-width-fraction``;
#: None = adaptive_boundary.MAX_CI_HALF_WIDTH_FRACTION, 20 % since 2026-09-24, the old
#: 15 % is reported as ``stop_rule_15``).
D13_MAX_CI_FRACTION: Optional[float] = None
#: lambda moves off LAMBDA_WAIT only when the best lambda beats it by this much BA.
LAMBDA_MIN_GAIN = 0.02
#: w_p the alpha stage runs at: the D3 values current on 2026-09-22 (plan 6.11 D3,
#: fable_acc_wp/wp_rule.log). ``--alpha-w-p`` overrides.
ALPHA_STAGE_W_P: dict[str, float] = {"dsqwen-7b": 0.01, "dsllama-8b": 0.005, "dsqwen-14b": 0.005}
HORIZON_MS = 30_000
FA_MAX = 0.05
SEED = 20260922
DWELL_WINDOWS = 2
#: The CRITICAL dwell the controller runs with: ``TRE_DWELL_WINDOWS`` of
#: deploy/overlays/tre-v2/controller.yaml (= tre_controller.config default, 1 = off since
#: the v1 alignment A5). accept judges B' at this dwell unless ``--dwell-windows`` says
#: otherwise (a guard test keeps it equal to the overlay).
ONLINE_DWELL_WINDOWS = 1
TRIM_RAMP_WINDOWS = 1
#: Windowing (2026-09-30): the window length and re-window step the fit CSVs were cut
#: with. Every windowing constant - WINDOW_MS, STEP_MS, DT_REF_S, HORIZON_MS,
#: DWELL_WINDOWS - is only a default: the stages take ``--window-ms`` / ``--step-ms`` /
#: ``--dt-ref-s`` / ``--horizon-ms`` / ``--dwell-windows``, thread the values explicitly
#: (no module state changes), record them as ``windowing`` in their outputs, and the
#: freeze carries them to ``accept``: a different window only needs an offline re-run.
WINDOW_MS = 30_000.0
STEP_MS = 10_000.0
WINDOWING_KEYS = ("window_ms", "step_ms", "dt_ref_s", "horizon_ms", "dwell_windows")
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
#: ``--train-dynamic``: the only split whose dynamic cells train (run 1's steps / ramp / bursts).
SPLIT_TRAIN_DYNAMIC = "train"
DYNAMIC_TRAINING_RULE = ("--train-dynamic (design 2026-10-05 item 2): dynamic cells (steps / ramp / bursts) "
                         "of the train split train with the constant-load cells; windows pooled with equal "
                         "weight (one window, one vote, as the BA fit always counted); the sealed split "
                         "(split holdout: run-2 ramps, held-out shape) stays H2")
#: Label attributions (the labels branch's tre_common.slo_labels.ATTRIBUTIONS; mirrored so
#: this module runs before that branch is merged). A manifest / label without one = completion.
ATTRIBUTION_COMPLETION, ATTRIBUTION_HYBRID = "completion", "hybrid"
ATTRIBUTIONS = (ATTRIBUTION_COMPLETION, ATTRIBUTION_HYBRID)
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


def assign_set(row: Mapping[str, str], *, sealed_to_h2: bool, sentinels: bool,
               train_dynamic: bool = False) -> str:
    """The set one standard-dataset row belongs to (training / h2 / m / sentinel_excluded).

    Read from the row's own ``split`` / ``shape`` / ``primitive`` / ``role`` (never its
    stage or cell id): a smoke hold (role smoke, stage dwell, split auxiliary) trains - D21,
    :func:`cell_kind`. ``train_dynamic`` (design 2026-10-05 item 2): a dynamic cell of the
    ``train`` split trains instead of joining H2 (the sealed split never trains)."""
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
        return SET_TRAINING if train_dynamic and split == SPLIT_TRAIN_DYNAMIC else SET_H2
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
        # Absolute, so a freeze run from another directory still finds the dataset.
        return cls(name=name, directory=d.resolve(), sealed_to_h2=sealed_to_h2)


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


def dataset_numerator(directory: Path) -> str:
    """The TSS numerator a standard dataset's signal columns were built with (its
    manifest's ``numerator.source``, :mod:`scripts.l3_numerator`); a manifest from before
    the record is ``gateway``, the only numerator there was."""
    from scripts import l3_numerator as l3

    man = Path(directory) / DATASET_MANIFEST
    try:
        doc = json.loads(man.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return l3.NUMERATOR_GATEWAY
    return str((doc.get("numerator") or {}).get("source") or l3.NUMERATOR_GATEWAY)


def dataset_attribution(directory: Path) -> str:
    """The label attribution a standard dataset was built with (its manifest's
    ``attribution.value``; ``calibration_dataset.dataset_attribution`` when the labels
    branch provides it). A manifest without the record - or no manifest - is completion."""
    try:
        from scripts.calibration_dataset import dataset_attribution as _da  # the labels branch
    except ImportError:
        _da = None
    if _da is not None:
        return str(_da(Path(directory)))
    d = Path(directory)
    man = d / DATASET_MANIFEST
    if not man.exists() and (d / "dataset" / DATASET_MANIFEST).exists():
        man = d / "dataset" / DATASET_MANIFEST
    try:
        doc = json.loads(man.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ATTRIBUTION_COMPLETION
    rec = doc.get("attribution") if isinstance(doc, dict) else None
    value = rec.get("value") if isinstance(rec, dict) else rec
    return str(value or ATTRIBUTION_COMPLETION)


def label_attribution(label_def: Optional[Mapping[str, Any]]) -> str:
    """The attribution of a label record (``LabelDefinition.as_dict()``); absent = completion
    (every v1 record)."""
    return str((label_def or {}).get("attribution") or ATTRIBUTION_COMPLETION)


def attribution_of_inputs(*, fit_dir: Optional[Path] = None, datasets: Iterable[Path] = (),
                          label_def: Optional[Mapping[str, Any]] = None, what: str = "inputs") -> str:
    """The one label attribution of a standalone tool's inputs (review 2026-10-05 P3-11):
    every record that states one - the fit dir's trainset.json, each dataset manifest, a
    label record - must agree; the agreed value is returned (completion when none of them
    states another, i.e. inputs made before label v2). Two different ones are a refusal
    (SystemExit), never a silent default."""
    found: dict[str, str] = {}
    if fit_dir is not None and (Path(fit_dir) / TRAINSET_MANIFEST).exists():
        found[f"{Path(fit_dir) / TRAINSET_MANIFEST}"] = trainset_attribution(Path(fit_dir))
    for d in datasets:
        if d is not None and Path(d).exists():
            found[str(d)] = dataset_attribution(Path(d))
    if label_def is not None:
        found["label record"] = label_attribution(label_def)
    values = set(found.values())
    if len(values) > 1:
        raise SystemExit(f"{what}: label attributions disagree {found}: refusing rather than mixing label v1 / v2")
    return values.pop() if values else ATTRIBUTION_COMPLETION


def trainset_attribution(fit_dir: Path) -> str:
    """The attribution trainset.json of ``fit_dir`` recorded; completion without one (a
    hand-built fit dir, or one written before the record)."""
    man = Path(fit_dir) / TRAINSET_MANIFEST
    try:
        doc = json.loads(man.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ATTRIBUTION_COMPLETION
    return str(doc.get("attribution") or ATTRIBUTION_COMPLETION)


def prompt_token_mismatched_cells(src: DatasetSource) -> set[tuple[str, str]]:
    """``(cell_id, attempt)`` of the source's cells whose served requests were off their
    prompt length (``cells.csv`` ``prompt_tokens_mismatched > 0``; a dataset built with
    ``--allow-prompt-token-mismatch``). Empty for a dataset without the column."""
    path = src.directory / "cells.csv"
    if not path.is_file():
        return set()
    with open(path, newline="", encoding="utf-8") as fh:
        return {(row.get("cell_id", ""), row.get("attempt", "")) for row in csv.DictReader(fh)
                if (row.get("prompt_tokens_mismatched") or "0").strip() not in ("", "0")}


def build_training_set(sources: Sequence[DatasetSource], fit_dir: Path, *,
                       sentinels: bool = True, models: Optional[Sequence[str]] = None,
                       train_dynamic: bool = False) -> dict:
    """D16: cut the training set (constant-load cells) out of standard datasets;
    ``train_dynamic``: plus the train-split dynamic cells (:data:`DYNAMIC_TRAINING_RULE`).

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
    numerators = {s.name: dataset_numerator(s.directory) for s in sources}
    if len(set(numerators.values())) > 1:
        # one theta, one numerator: gateway and L3 totals of a window differ by design
        raise TrainingSetError(f"the datasets were built with different TSS numerators {numerators}; "
                               "rebuild them with one calibration_dataset --numerator")
    attributions = {s.name: dataset_attribution(s.directory) for s in sources}
    if len(set(attributions.values())) > 1:
        # one theta, one label: completion and hybrid label a window's requests differently
        raise TrainingSetError(f"the datasets were built with different label attributions {attributions}; "
                               "rebuild them with one calibration_dataset --attribution")
    bad_attr = sorted({a for a in attributions.values() if a not in ATTRIBUTIONS})
    if bad_attr:
        raise TrainingSetError(f"unknown label attribution(s) {bad_attr} (known: {list(ATTRIBUTIONS)})")
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

    off_length = {s.name: prompt_token_mismatched_cells(s) for s in sources}
    try:
        for src in sources:
            header = headers[src.name]
            with open(src.windows, newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                next(reader)
                for values in reader:
                    row = dict(zip(header, values))
                    target = assign_set(row, sealed_to_h2=src.sealed_to_h2, sentinels=sentinels,
                                        train_dynamic=train_dynamic)
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
                    if (row["cell_id"], row["attempt"]) in off_length[src.name]:
                        raise TrainingSetError(
                            f"{src.name}: training cell {row['cell_id']} attempt {row['attempt']} served "
                            f"requests off their prompt length (cells.csv prompt_tokens_mismatched > 0; all "
                            f"such cells of the source: {sorted(off_length[src.name])}); a theta is never "
                            "fitted on windows indexed by a length the engine did not prefill")
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
                                                  "by_cell_status": {}, "dynamic_windows": 0,
                                                  "dynamic_cells": set(), "dynamic_by_run_primitive": {},
                                                  **{f: 0 for f in FAMILY_FILES}})
                    c["fitting"] += 1
                    c["cells"].add((src.name, sid))
                    if cell_kind(row["primitive"], row["role"], row["split"], row["shape"]) == KIND_DYNAMIC:
                        c["dynamic_windows"] += 1
                        c["dynamic_cells"].add((src.name, sid))
                        dk = f"{src.name}|{row['primitive']}"
                        c["dynamic_by_run_primitive"][dk] = c["dynamic_by_run_primitive"].get(dk, 0) + 1
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
            "dynamic": {"windows": c["dynamic_windows"], "cells": len(c["dynamic_cells"]),
                        "by_run_primitive": dict(sorted(c["dynamic_by_run_primitive"].items()))},
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
            "numerator": numerators[s.name], "attribution": attributions[s.name],
        })
    doc = {
        "what": ("D16 training set: the constant-load cells of the sources (theta is fitted on these only)"
                 if not train_dynamic else
                 "training set: the constant-load cells and the train-split dynamic cells of the sources "
                 "(--train-dynamic, design 2026-10-05 item 2)"),
        "rules": {
            "training": ("split in {train, auxiliary}, primitive in alpha_fit.STEADY_PRIMITIVES, role "
                         "not in alpha_fit.UNSTEADY_ROLES - read from each row's own dataset columns"
                         + ("; plus every dynamic cell (steps / ramp / bursts) of split train" if train_dynamic
                            else "")),
            "sentinels": "role sentinel trains" if sentinels else "role sentinel excluded (--no-sentinels)",
            "h2": ("the sealed split (split holdout) of --h2-dataset sources + dynamic cells outside split train"
                   if train_dynamic else
                   "dynamic cells of every source + the sealed split (split holdout) of --h2-dataset sources"),
            "m": "the sealed split of --dataset sources: skipped unread (counts only)",
            "families": {fam: list(shapes) for fam, shapes in families.items()},
        },
        "sentinels": sentinels,
        # The TSS numerator of every source (scripts.l3_numerator; one by construction).
        "numerator": next(iter(numerators.values()), None),
        # The label attribution of every source (one by construction); the fit stages label with it.
        "attribution": next(iter(attributions.values()), ATTRIBUTION_COMPLETION),
        "dynamic_training": {"admitted": bool(train_dynamic),
                             "rule": DYNAMIC_TRAINING_RULE if train_dynamic else
                             "off: D16 - constant-load cells only (no --train-dynamic)",
                             "splits": [SPLIT_TRAIN_DYNAMIC] if train_dynamic else [],
                             "weights": "equal per window"},
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
    allowed = {KIND_CONSTANT}
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        entry = (man.get("models") or {}).get(model)
        if entry is None:
            raise SystemExit(f"{man_path}: no training set for {model}")
        for key, path in files.items():
            want = (entry["files"].get(key) or {}).get("sha256")
            if want is not None and path.exists() and sha256_file(path) != want:
                raise SystemExit(f"{path} is not the file the trainset stage wrote ({man_path})")
        dyn = (man.get("dynamic_training") or {}).get("admitted") is True
        if dyn:
            allowed.add(KIND_DYNAMIC)
        prov = {"trainset_manifest": str(man_path), "trainset_manifest_sha256": sha256_file(man_path),
                "sentinels": man.get("sentinels"), "h2_rows_sha256": man["h2"]["rows_sha256"],
                "h2_cells_sha256": man["h2"]["cells_sha256"], "dynamic_training": dyn,
                "attribution": str(man.get("attribution") or ATTRIBUTION_COMPLETION)}
    kinds = {}
    for key, path in files.items():
        if not path.exists():
            continue
        k = kinds[key] = scan_training_rows(path, ledger)
        bad = {kind: n for kind, n in k.items() if kind not in allowed and n}
        if bad:
            raise SystemExit(
                f"{path}: D16 - theta is fitted on constant-load cells only"
                + (" (and the train-split dynamic cells the trainset manifest admits)" if KIND_DYNAMIC in allowed
                   else "")
                + f", and this file holds {bad} window rows that are not (unknown = no primitive column and no "
                "ledger line). Build the training set with `dline_refit trainset` (--train-dynamic admits the "
                "train-split dynamic cells).")
    prov["rows"] = kinds
    return prov


def label_for(model: str, arm: str, registry: Optional[str] = None,
              attribution: Optional[str] = None) -> slo_labels.LabelDefinition:
    """The label of one arm: the registry profile's D6' primary, or an arm of it.

    ``attribution`` (completion / hybrid; None = completion): the attribution of the
    datasets it labels - trainset.json's for the fit stages, the freeze label's
    (``label_def["attribution"]``) for anything compared with a freeze. Completion builds
    the v1 label exactly as before."""
    attribution = attribution or ATTRIBUTION_COMPLETION
    if attribution not in ATTRIBUTIONS:
        raise SystemExit(f"label attribution {attribution!r} is not one of {list(ATTRIBUTIONS)}")
    extra: dict[str, Any] = {}
    if attribution != ATTRIBUTION_COMPLETION:
        if "attribution" not in getattr(slo_labels.LabelDefinition, "__dataclass_fields__", {}):
            raise SystemExit(f"a {attribution} label needs tre_common.slo_labels with label attributions "
                             "(branch calib/next-20261005)")
        extra["attribution"] = attribution
    primary = slo_labels.label_def_for_model(
        model, ttft_p95_ms=500.0, tpot_p95_ms=75.0, mode=None, registry=registry, **extra)
    if arm == "primary":
        return primary
    arms = slo_labels.label_arms(primary)
    return {"fixed": arms[slo_labels.ARM_FIXED], "k3": arms[slo_labels.ARM_K3]}[arm]


def shape_fn() -> Callable[[str], str]:
    from scripts import alpha_fit

    table = alpha_fit.shape_table()
    return lambda sid: alpha_fit.shape_of(sid, table)


def windowing(*, window_ms: float = WINDOW_MS, step_ms: float = STEP_MS, dt_ref_s: float = DT_REF_S,
              horizon_ms: int = HORIZON_MS, dwell_windows: int = DWELL_WINDOWS) -> dict[str, Any]:
    """The windowing of one run of the stages (:data:`WINDOWING_KEYS`), validated."""
    win = {"window_ms": float(window_ms), "step_ms": float(step_ms), "dt_ref_s": float(dt_ref_s),
           "horizon_ms": int(horizon_ms), "dwell_windows": int(dwell_windows)}
    bad = [k for k in ("window_ms", "step_ms", "dt_ref_s", "horizon_ms") if not win[k] > 0]
    bad += ["dwell_windows"] if win["dwell_windows"] < 1 else []
    if bad or not all(math.isfinite(float(v)) for v in win.values()):
        raise ValueError(f"windowing {win}: {bad or 'non-finite'} out of range")
    return win


DEFAULT_WINDOWING = windowing()


def windowing_of(doc: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """The windowing a stage output / freeze entry recorded; the defaults for keys (or a
    whole block) an older document does not carry - the D22 freeze predates the record."""
    rec = (doc or {}).get("windowing") or {}
    return {k: rec.get(k, DEFAULT_WINDOWING[k]) for k in WINDOWING_KEYS}


def alpha_of(tau_s: float, dt_ref_s: float = DT_REF_S) -> float:
    return 1.0 if tau_s <= 0 else 1.0 - math.exp(-dt_ref_s / tau_s)


def step90_s(tau_s: float, dt_ref_s: float = DT_REF_S) -> float:
    a = alpha_of(tau_s, dt_ref_s)
    if a >= 1.0:
        return 0.0
    return dt_ref_s * math.ceil(math.log(0.1) / math.log(1.0 - a))


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


def future_pairs(windows, crit, *, horizon_ms: int = HORIZON_MS) -> list[tuple]:
    """(cell, crit flag at t, violated at t + horizon) for every window whose t + horizon
    window (default 30 s) is labelled."""
    idx = {(w.scenario_id, int(w.window_start_ms)): i for i, w in enumerate(windows)}
    out = []
    for i, w in enumerate(windows):
        j = idx.get((w.scenario_id, int(w.window_start_ms) + int(horizon_ms)))
        if j is not None:
            out.append((w.scenario_id, crit[i], not windows[j].slo_met))
    return out


def detection_lags(windows, crit, *, horizon_ms: int = HORIZON_MS) -> list[Optional[float]]:
    """Per violation episode (run of violated windows in a cell), seconds from its first
    window to the first dwell-confirmed CRITICAL within [start - horizon, end] (default
    30 s); None = missed."""
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
                    if crit[i] and t0 - horizon_ms <= windows[i].window_start_ms <= t1]
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


def stage_alpha_refit0922(model: str, label, p: Mapping[str, Any], *, w_p: float,
                          win: Optional[Mapping[str, Any]] = None) -> dict:
    """The archived alpha stage (t + 30 s label; kept to reproduce 2026-09-22). ``win``:
    the windowing (:func:`windowing`; default :data:`DEFAULT_WINDOWING`)."""
    from scripts import theta_verdict as tv

    win = dict(DEFAULT_WINDOWING if win is None else win)
    dt_ref, horizon = win["dt_ref_s"], win["horizon_ms"]

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
            crit = tv.critical_dwell_flags(test, theta=theta, tau_crit=tau_crit, direction=spec.direction,
                                           dwell_windows=win["dwell_windows"], window_ms=win["window_ms"])
            pairs += future_pairs(test, crit, horizon_ms=horizon)
            lags += detection_lags(test, crit, horizon_ms=horizon)
            folds[s] = {"theta": theta, "tau_crit": tau_crit}
        rec, fa, ba, npos, nneg = rates([q for _, q, _ in pairs], [v for _, _, v in pairs])
        se = boot_ba_se(pairs)
        full = fit_theta_delta(windows, spec)
        hit = sorted(x for x in lags if x is not None)
        curve.append({
            "tau_s": tau, "alpha": alpha_of(tau, dt_ref), "loso_ba": ba, "loso_ba_se": se, "recall": rec,
            "false_alarm": fa, "n_pos": npos, "n_neg": nneg, "feasible_fa": fa <= FA_MAX,
            "step90_ema_s": step90_s(tau, dt_ref), "step90_with_dwell_s": step90_s(tau, dt_ref) + dt_ref,
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
                        registry: Optional[str] = None, win: Optional[Mapping[str, Any]] = None) -> dict:
    """D4' (:mod:`scripts.alpha_fit`) on the same fitting CSV and label, at the windowing
    ``win`` (dt_ref, dwell, re-window step and window length are passed through)."""
    from scripts import alpha_fit

    win = dict(DEFAULT_WINDOWING if win is None else win)
    ns = argparse.Namespace(
        model=model, fitting_csv=str(p["fitting"]), w_p=w_p, lambda_wait=LAMBDA_WAIT, qmin=1.0,
        trim_ramp_windows=TRIM_RAMP_WINDOWS, tau_grid_s=[float(t) for t in TAUS_S],
        dt_ref_s=win["dt_ref_s"], dwell_windows=win["dwell_windows"], fa_max=FA_MAX,
        step_ms=win["step_ms"], window_ms=win["window_ms"], episode_margin_ms=None,
        se_resamples=alpha_fit.DEFAULT_SE_RESAMPLES,
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


def publish_alpha(doc: dict, publish_tau_s: Optional[float], *, dt_ref_s: float = DT_REF_S) -> dict:
    """D18 on an alpha-stage document: the rule's pick stays ``chosen_tau_s`` (disclosed);
    ``published_tau_s`` - what w_p / theta / delta are fitted at and what deploys - is
    ``publish_tau_s``, or the rule's pick when that is None (``--publish-tau-s rule``)."""
    rule_tau = doc.get("chosen_tau_s")
    tau = rule_tau if publish_tau_s is None else float(publish_tau_s)
    doc["published_tau_s"] = tau
    doc["published_alpha"] = alpha_of(tau, dt_ref_s) if tau is not None else None
    doc["publish_rule"] = ("the alpha rule's pick (--publish-tau-s rule)" if publish_tau_s is None
                           else f"D18: the common tau {tau:g} s, whatever the rule picks")
    doc["published_registry_fields"] = (
        {"ema_tau_ms": tau * 1000.0, "ema_alpha": round(alpha_of(tau, dt_ref_s), 6)} if tau is not None else None)
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
                                 family_resamples=fam_res, seed=SEED, max_ci_fraction=D13_MAX_CI_FRACTION)
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
                sources: Mapping[str, Path], *, wp_grid: str = "with_zero") -> dict:
    """``wp --lambda-method v1``: lambda_wait and w_p from v1's selection (stages A-C of
    ``fit_tre_parameters_from_runs.py``, :mod:`scripts.v1_lambda_fit`) on the D16 fitting
    windows; ``sources`` (run -> standard dataset dir) are where the average TPOT of v1's
    average-health term is rebuilt from. The D17 w_p rule is NOT applied and NOT computed:
    user 2026-09-24, v1's joint lambda x w_p refinement is taken as is (no D17 comparison is
    made or reported)."""
    from scripts import v1_lambda_fit

    tau = published_tau(alpha_doc)
    if tau is None:
        raise SystemExit(f"{model}: the alpha stage published no tau ({alpha_doc.get('rule')})")
    sel = v1_lambda_fit.fit_model(model, label, p["fitting"], trim=TRIM_RAMP_WINDOWS, sources=sources,
                                  wp_grid=wp_grid)
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


def _legacy_stop(stop: Mapping[str, Any]) -> Optional[bool]:
    """The D13 verdict at the pre-2026-09-24 CI gate (15 %): every other condition as
    judged, the CI half width below 15 % (reported, never gating)."""
    from scripts import adaptive_boundary as boundary

    frac = stop.get("ci_half_width_fraction")
    if frac is None:
        return None
    others = [r for r in stop.get("reasons") or [] if not str(r).startswith("CI half width")]
    return bool(not others and frac < boundary.LEGACY_CI_HALF_WIDTH_FRACTION)


#: The theta uncertainty band disclosed next to D (coordinator 2026-10-05, replacing the
#: dropped plateau-midpoint rule): the fit's own candidates within this BA of the best.
PLATEAU_BA_TOLERANCE = 0.005
THETA_SELECTION_RULE = ("theta = the BA-argmax candidate of tre_calibration.fit.fit_theta_by_balanced_accuracy "
                        "(candidates: the fit config's healthy-quantile grid); exact BA ties (1e-12) break on the "
                        "higher violating-window specificity, then on the larger oriented theta (the threshold "
                        "admitting fewer windows as healthy) - the existing pipeline's rule, unchanged. No "
                        "midpoint rule (dropped 2026-10-05).")
PLATEAU_RULE = ("disclosed, not selecting: the fit's own candidates whose training BA is within "
                "PLATEAU_BA_TOLERANCE of the best; [min, max] of their theta is the theta uncertainty band "
                "next to D (the CI half width <= 20 %, which stays the identifiability gate)")


def theta_plateau(windows: Sequence[Any], config: Any, *, fit_theta: Optional[float] = None,
                  tol: float = PLATEAU_BA_TOLERANCE) -> dict:
    """The BA plateau over the fit's OWN candidate grid (:data:`PLATEAU_RULE`) - the same
    candidates ``config.fit`` searches, scored with ``threshold_balanced_accuracy``."""
    from tre_calibration import fit as tf

    o = tf.signal_orientation(config.direction)
    rows = [w for w in windows if math.isfinite(w.signal)]
    healthy = [o * w.signal for w in rows if w.slo_met]
    if not healthy or all(w.slo_met for w in rows):
        return {"rule": PLATEAU_RULE, "tolerance": tol, "candidates": 0, "ba_max": None}
    if config.candidate_grid == "unique":
        cands = tf._unique_candidates(healthy, config.healthy_quantile_candidates, o)
    else:
        cands = [(q, tf._quantile(healthy, q)) for q in config.healthy_quantile_candidates]
    pts = []
    for q, ot in cands:
        if ot is None:
            continue
        theta = o * float(ot)
        ba = tf.threshold_balanced_accuracy(rows, theta=theta, direction=config.direction)["balanced_accuracy"]
        pts.append({"healthy_quantile": q, "theta": theta, "ba": ba})
    best = max(p["ba"] for p in pts)
    inside = [p for p in pts if p["ba"] >= best - tol - 1e-12]
    lo, hi = min(p["theta"] for p in inside), max(p["theta"] for p in inside)
    return {"rule": PLATEAU_RULE, "tolerance": tol, "candidates": len(pts), "ba_max": best,
            "inside": inside, "theta_lo": lo, "theta_hi": hi,
            "band_frac_of_fit_theta": ([lo / fit_theta - 1.0, hi / fit_theta - 1.0]
                                       if fit_theta else None),
            "argmax_unique_within_tolerance": len(inside) == 1}


def absolute_thresholds(theta: Optional[float], tau_crit: Optional[float], tau_high: Optional[float]) -> dict:
    """Disclosure (design item 2): the absolute CRITICAL / HIGH lines theta * tau."""
    ok = all(isinstance(x, (int, float)) and math.isfinite(x) for x in (theta, tau_crit, tau_high))
    return {"theta": theta, "tau_crit": tau_crit, "tau_high": tau_high,
            "critical_abs": theta * tau_crit if ok else None, "high_abs": theta * tau_high if ok else None}


#: What ``final.json`` says instead of the M numbers when the stage ran with ``--no-holdout``.
HOLDOUT_SKIPPED = ("not evaluated (--no-holdout): M is read once, after it is frozen and "
                   "hashed (plan 2026-09-21 §6.11 note 9)")


def stage_final(model: str, label, p: Mapping[str, Any], wp_doc: Mapping[str, Any], out_dir: Path,
                *, holdout: bool = True, win: Optional[Mapping[str, Any]] = None) -> dict:
    """D5 verdict at (tau, w_p*, lambda*) with the theta plateau disclosed
    (:func:`theta_plateau`); then, unless ``holdout`` is False, the M report (dwell and
    window length from ``win``; the attainment tiles are one window long).

    With ``holdout=False`` the validation CSV is never opened (not even for its size)."""
    from tre_calibration.fit import threshold_balanced_accuracy

    from scripts import theta_verdict as tv

    win = dict(DEFAULT_WINDOWING if win is None else win)
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
    # the theta uncertainty band (disclosed; theta stays the fit's argmax), on the training windows
    spec_f = spec_for(tau, wp, lam)
    selection = {"rule": THETA_SELECTION_RULE, "theta": v["published"]["theta_m"],
                 "plateau": theta_plateau(spec_f.load(p["fitting"], label, TRIM_RAMP_WINDOWS),
                                          spec_f.default_config(), fit_theta=v["published"]["theta_m"])}
    (out_dir / "verdict_final.json").write_text(json.dumps(v, indent=1, default=str))
    pub = v["published"]
    extra = {"theta_selection": selection,
             "absolute_thresholds": absolute_thresholds(pub["theta_m"], pub["tau_crit"], pub.get("tau_high"))}
    if not holdout:
        spec = spec_for(tau, wp, lam)
        s = summarize(v)
        s["theta_family_rule"] = v["published"]["family_rule_theta"]
        return {
            "model": model, "tau_s": tau, "alpha": alpha_of(tau, win["dt_ref_s"]), "w_p": wp, "lambda_wait": lam,
            **s, **extra, "stop_rule_d13": v["stop_rule"]["satisfied"],
            "d13_max_ci_half_width_fraction": v["stop_rule"].get("max_ci_half_width_fraction"),
            "stop_rule_15": _legacy_stop(v["stop_rule"]),
            "appendix_10_met": v["stop_rule"].get("appendix_ci_target_met"),
            "train_ba_at_published": threshold_balanced_accuracy(
                spec.load(p["fitting"], label, TRIM_RAMP_WINDOWS), theta=v["published"]["theta_m"],
                direction="higher_is_healthier")["balanced_accuracy"],
            "holdout_evaluated": False, "holdout": HOLDOUT_SKIPPED,
        }
    if not Path(p["validation"]).exists():
        raise SystemExit(f"{p['validation']} does not exist: a D16 training-set directory carries "
                         "no M - run final with --no-holdout until M is frozen")
    h = tv.holdout_report(v, p["validation"], dwell_windows=win["dwell_windows"], window_ms=win["window_ms"])
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
    # per-length-bucket request attainment on M (non-overlapping one-window tiles only)
    tile_ms = int(win["window_ms"])
    att = defaultdict(lambda: [0, 0])
    first: dict[str, int] = {}
    with open(p["validation"], newline="") as fh:
        for row in csv.DictReader(fh):
            c, s = row["scenario_id"], int(row["window_start_ms"])
            first.setdefault(c, s)
            if (s - first[c]) % tile_ms:
                continue
            for ttft, length in slo_labels.parse_ttft_len_samples(row.get("ttft_len_samples") or ""):
                b = bucket(length)
                att[b][1] += 1
                att[b][0] += ttft <= label.ttft_slo_ms(length)
    wd = h["with_dwell"]
    s = summarize(v)
    s["theta_family_rule"] = v["published"]["family_rule_theta"]
    return {
        "model": model, "tau_s": tau, "alpha": alpha_of(tau, win["dt_ref_s"]), "w_p": wp, "lambda_wait": lam,
        **s, **extra, "stop_rule_d13": v["stop_rule"]["satisfied"],
        "d13_max_ci_half_width_fraction": v["stop_rule"].get("max_ci_half_width_fraction"),
        "stop_rule_15": _legacy_stop(v["stop_rule"]),
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
    ws = spec.load(p["fitting"], label_for(model, arm, registry, trainset_attribution(Path(p["fitting"]).parent)),
                   TRIM_RAMP_WINDOWS)
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

#: 2 (2026-09-30): each model entry records its ``windowing``. Revision 1 freezes (the
#: D22 one) still verify and accept: their windowing is :data:`DEFAULT_WINDOWING`.
#: 3 (2026-10-03): each model entry seals its B' severity cut (``b_prime``) and the
#: document the B' gate (``b_prime_gate``). Older freezes accept only with an explicit
#: ``--b-prime-thresholds`` file.
#: 4 (2026-10-05): the accept gate (``accept_gate``: onset or b_prime) and, per model, the
#: theta selection / plateau, absolute thresholds, label attribution, dynamic-training
#: record and the A dead band's s0. A revision <= 3 freeze accepts under the A, B', D gate.
FREEZE_FORMAT_REVISION = 4
FREEZE_READABLE_REVISIONS = (1, 2, 3, 4)
#: The accept gate of a NEW (revision 4) freeze is always onset (design 2026-10-05 item 3;
#: review 2026-10-05: one gate, no fork). b_prime names the 2026-10-03 gate (A, B', D) that a
#: freeze of revision <= 3 still accepts under; a revision-4 freeze sealing anything else
#: than the running code's onset gate is refused (:func:`accept_gate_problems`).
ACCEPT_GATE_ONSET, ACCEPT_GATE_B_PRIME = "onset", "b_prime"
ACCEPT_GATES = (ACCEPT_GATE_ONSET, ACCEPT_GATE_B_PRIME)
#: The onset gate's lag budget (2 ticks, 20 s) is derived from the EMA alpha .632 of
#: tau = 10 s (design item 3 (i)): a freeze of another tau is refused under it.
ONSET_TAU_S = 10.0
#: Three-way verdict of the onset gate (user 2026-10-05, decision 2; one rule for every model).
VERDICT_PASS, VERDICT_PASS_A_DISCLOSED, VERDICT_FAIL = "pass", "pass_a_disclosed", "fail"
ONSET_GATING_CRITERIA = ("onset", "window_fa", "A", "D")
VERDICT_RULE = ("pass = onset, window_fa, A and D all pass; pass_a_disclosed = only A fails (every other gate "
                "passes): go-live allowed, A disclosed as a limitation; fail = anything else. Same rule for "
                "every model (user 2026-10-05)")
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
#: The dwell accept applies when the freeze records no windowing (revision 1, e.g. D22).
ACCEPT_DWELL_WINDOWS = DWELL_WINDOWS
EXIT_REFUSED = 1
EXIT_ACCEPT_FAILED = 3
EXIT_RECHECK_DIFFERS = 4
#: Keys a ``--recheck`` does not compare: timestamps, work paths, the command line, and the
#: code state of the run (provenance of the run, printed when it differs, not a result).
ACCEPT_VOLATILE_KEYS = frozenset({"generated_at", "evaluated_at_utc", "validation_csv", "work_dir",
                                  "command", "code"})
#: 2 (2026-09-30): the ranking disclosure (per model and pooled, never gating) and each
#: model's windowing. A ``--recheck`` of a revision-1 result compares everything else.
#: 3 (2026-10-03): the gate is A, B' and D (``criteria.B_prime``, ``thresholds.B_prime``);
#: the old B is disclosed. A ``--recheck`` of an older result compares it under its own
#: gate (A, B, D) without the B' keys (:func:`as_revision`).
#: 4 (2026-10-05): the accept gate rule (``gate_rule``), the onset gate / window false alarm
#: / A dead band (``criteria.onset``, ``criteria.window_fa``, ``criteria.A_deadband``), the
#: per-model and overall ``verdict``, ``disclosed_limitations``, the datasets' attribution
#: and ``thresholds.accept_gate``. Under an older freeze the gate is still A, B', D, so a
#: ``--recheck`` of a revision-3 result compares it without these keys.
ACCEPT_FORMAT_REVISION = 4
#: Keys a revision added, top level and per model: absent from an older stored result.
ACCEPT_REVISION_KEYS = {2: {"top": ("ranking_disclosure",), "model": ("ranking_disclosure", "windowing")},
                        3: {"top": (), "model": ()},
                        4: {"top": ("gate_rule", "verdict", "disclosed_limitations", "attribution"),
                            "model": ("gate_rule", "verdict")}}
#: Revision 4 keys one level down: under ``criteria`` and ``thresholds``.
ACCEPT_REV4_CRITERIA = ("onset", "window_fa", "A_deadband")
ACCEPT_REV4_THRESHOLDS = ("accept_gate",)
LEGACY_ACCEPT_WHAT = ("plan §6.9f acceptance A-D on M, evaluated once on the frozen parameters "
                      "(A, B, D gate; C and the all-violating recall are disclosed)")
#: The criteria that gate, by accept format revision.
GATING_CRITERIA = ("A", "B_prime", "D")
LEGACY_GATING_CRITERIA = ("A", "B", "D")


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
    # one windowing for the whole refit (stage outputs written before the record carry none)
    win = windowing_of(fin)
    for name, doc in (("alpha", alpha), ("wp", wp)):
        if "windowing" in doc and windowing_of(doc) != win:
            problems.append(f"stage outputs disagree on the windowing: {name} {windowing_of(doc)} != final {win}")

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
            load_paths, lp_problems = training_load_paths(man_doc)
            problems += lp_problems
            if len(load_paths) > 1:
                problems.append(f"the training set mixes load paths {sorted(load_paths)}: a theta "
                                "fitted across prompt corpora / routing / APIs describes neither")
            lp = fit_dir / TRAINING_LEDGER
            try:
                check_training_inputs(model, p, ledger=load_ledgers([str(lp)]) if lp.exists() else None)
            except SystemExit as exc:
                problems.append(f"training inputs: {exc}")
            ts_attr = str(man_doc.get("attribution") or ATTRIBUTION_COMPLETION)
            if label_attribution(ver.get("label_def")) != ts_attr:
                problems.append(f"the fitted label's attribution {label_attribution(ver.get('label_def'))!r} != the "
                                f"training datasets' {ts_attr!r} (trainset.json): rerun the fit stages")
    if problems:
        return None, problems
    # B' (2026-10-03): the severity cut, from these training windows only, sealed now -
    # before any M / T14 window exists
    try:
        b_prime_rec = b_prime_freeze_record(verdict_for_holdout(ver), p["fitting"])
    except (ValueError, KeyError) as exc:
        return None, [f"B' severity cut: {exc}"]
    a_deadband = deadband_freeze_record(verdict_for_holdout(ver), p["fitting"])

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
        "windowing": {**win, "source": ("final.json" if "windowing" in fin
                                        else "defaults: final.json predates the windowing record")},
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
        # The training load path (scripts.prompt_corpus): what the prompts were written in
        # and how they were routed. M and T14 refuse to run under this freeze with another.
        # Absent in older freezes = English prompts, no routing header.
        "prompt_corpus": next(iter(load_paths.values()))["prompt"],
        "routing_strategy": next(iter(load_paths.values()))["routing_strategy"],
        # ... and through which API (absent in older freezes = completions). The idle-TTFT
        # fit of the D6' label and every length in ttft_len_samples count the prompt the
        # way this API does (chat: template included).
        "api": next(iter(load_paths.values()))["api"],
        # The TSS numerator the training windows carried (scripts.l3_numerator; absent in
        # older freezes = gateway). accept refuses an M dataset built with another.
        "numerator": man_doc.get("numerator") or "gateway",
        "b_prime": b_prime_rec,
        # revision 4 (2026-10-05)
        "label_attribution": label_attribution(ver["label_def"]),
        "dynamic_training": man_doc.get("dynamic_training") or {"admitted": False,
                                                                 "rule": "trainset.json predates the record"},
        "theta_selection": fin.get("theta_selection") or {"rule": "final.json predates the record: the BA argmax"},
        "absolute_thresholds": absolute_thresholds(fin["theta_published"], fin["tau_crit"], pub.get("tau_high")),
        "a_deadband": a_deadband,
    }
    return entry, []


def deadband_freeze_record(vh: Mapping[str, Any], training_csv: Path) -> dict:
    """The A dead band's s0 (disclosure only) from the freeze's training CSV:
    :func:`scripts.b_prime.label_noise_s0` under the frozen label."""
    from scripts import b_prime

    label = slo_labels.LabelDefinition.from_dict(vh["label_def"])
    with open(training_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rec = b_prime.label_noise_s0(rows, label)
    return {**rec, "training_csv": str(training_csv), "training_csv_sha256": sha256_file(Path(training_csv)),
            "gating": False, "use": "A dead band: BA with violating windows of severity < s0 left out (disclosed)"}


def b_prime_freeze_record(vh: Mapping[str, Any], training_csv: Path) -> dict:
    """The sealed B' cut of one model: the :data:`scripts.b_prime.SEVERITY_QUANTILE` of
    the violating windows' severity in ``training_csv`` (the freeze's fitting CSV), loaded
    with the frozen signal spec, label and ramp trim. Raises ``ValueError`` without a
    violating training window."""
    from scripts import b_prime
    from scripts import theta_verdict as tv

    spec = tv.SignalSpec.from_dict(vh["signal_spec"])
    label = slo_labels.LabelDefinition.from_dict(vh["label_def"])
    windows = spec.load(Path(training_csv), label, int(vh["trim_ramp_windows"]))
    cut = b_prime.severity_cut(windows, b_prime.SEVERITY_QUANTILE)
    return {"severity_cut": cut, "severity_quantile": b_prime.SEVERITY_QUANTILE,
            "severity": b_prime.SEVERITY_RULE,
            "violating_training_windows": len(b_prime.violating_severities(windows)),
            "training_csv": str(training_csv), "training_csv_sha256": sha256_file(Path(training_csv)),
            "windows": "the training fitting CSV, frozen signal spec / label / ramp trim; violating, finite signal"}


def training_load_paths(trainset_manifest: Mapping) -> tuple[dict[str, dict], list[str]]:
    """``({description: load path}, problems)`` over the standard datasets a training set
    (the trainset stage's manifest, ``sources[].directory``) was cut from, each read from
    the dataset's own manifest - whose sha256 must still be the one the trainset stage
    recorded. A manifest that is missing, unreadable or changed is a problem, never a
    silent default; a readable one without a record predates the options (English
    prompts, no routing header)."""
    from scripts import prompt_corpus as corpus_record

    found: dict[str, dict] = {}
    problems: list[str] = []
    sources = trainset_manifest.get("sources") or []
    if not sources:
        problems.append("the trainset manifest lists no sources: the training load path is unknown")
    for source in sources:
        directory = source.get("directory")
        manifest = Path(directory) / DATASET_MANIFEST if directory else None
        if manifest is None or not manifest.is_file():
            problems.append(f"training source {source.get('run')!r}: no dataset manifest at "
                            f"{manifest}: its load path (prompt corpus, routing) is unknown")
            continue
        recorded = source.get("manifest_sha256")
        if not recorded:
            problems.append(f"{manifest}: the trainset stage saw no dataset manifest here, so what "
                            "is there now cannot be the training data's")
            continue
        if sha256_file(manifest) != recorded:
            problems.append(f"{manifest} changed since the trainset stage read it "
                            f"(sha256 {recorded})")
            continue
        try:
            doc = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{manifest}: unreadable ({exc})")
            continue
        path = corpus_record.dataset_load_path(doc)
        found[corpus_record.describe_load_path(path)] = path
    return found, problems


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
    accept_gate = ACCEPT_GATE_ONSET
    entries: dict[str, dict] = {}
    for model in models:
        entry, pr = freeze_model(out_root, fit_dir_of(model), model, arm)
        problems += [f"{model}: {x}" for x in pr]
        if entry is not None:
            entries[model] = entry
            tau_s = (entry.get("published") or {}).get("tau_s")
            if not _same(tau_s, ONSET_TAU_S):
                # review 2026-10-05 P2-3: the 20 s lag budget is alpha .632 of tau 10 s
                problems.append(f"{model}: tau_s {tau_s!r} != {ONSET_TAU_S:g}: the onset gate's lag budget "
                                "(2 ticks) is derived from the EMA of tau 10 s")
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
        "b_prime_gate": {**b_prime_gate_defaults(),
                         "decided": "user 2026-10-03: B' replaces B as the accept gate (b_prime_thresholds.json "
                                    "of 2026-09-24); judged at the controller's dwell, dwell 2 disclosed"},
        "accept_gate": accept_gate_record(accept_gate),
        "self_hash_rule": ("freeze_sha256 = sha256 of json.dumps(doc without freeze_sha256, sort_keys=True, "
                           "separators=(',', ':'), ensure_ascii=False) in UTF-8"),
    }
    doc["freeze_sha256"] = canonical_sha256(doc)
    data = _json_bytes(doc)
    fp["freeze"].parent.mkdir(parents=True, exist_ok=True)
    _write_once(fp["freeze"], data)
    _write_once(fp["sidecar"], f"{hashlib.sha256(data).hexdigest()}  {fp['freeze'].name}\n".encode())
    return doc


def accept_gate_record(rule: str) -> dict:
    """The accept gate a freeze seals (revision 4): which rule gates, and every parameter
    of the onset gate, window false alarm and verdict (also sealed under ``b_prime``, where
    they are disclosed)."""
    from scripts import b_prime

    return {"rule": rule, "gating_criteria": list(ONSET_GATING_CRITERIA if rule == ACCEPT_GATE_ONSET
                                                  else GATING_CRITERIA),
            "onset": {**b_prime.ONSET_GATE, "dynamic_primitives": list(b_prime.ONSET_DYNAMIC_PRIMITIVES),
                      "rule": b_prime.ONSET_RULE, "ci_rule": b_prime.ONSET_CI_RULE,
                      "lookback_clip": b_prime.ONSET_LOOKBACK_CLIP,
                      "icc_undefined_value": b_prime.ICC_UNDEFINED_VALUE},
            "window_fa": dict(b_prime.WINDOW_FA_GATE),
            "severity_cut": "each model's b_prime.severity_cut (training .65 quantile under the frozen label)",
            "dwell_windows": ONLINE_DWELL_WINDOWS, "verdict_rule": VERDICT_RULE,
            "disclosed": ["B' (window severe recall)", "old B", "dwell 2", "A dead band at s0",
                          "theta plateau", "absolute thresholds"],
            "decided": ("design docs/calib-next-round-design-20261005.md item 3 and user decision 2 (2026-10-05)"
                        if rule == ACCEPT_GATE_ONSET else "user 2026-10-03: A, B', D")}


def b_prime_gate_defaults() -> dict:
    from scripts import b_prime

    return {**b_prime.DEFAULT_GATE, "severity_quantile": b_prime.SEVERITY_QUANTILE,
            "online_dwell_windows": ONLINE_DWELL_WINDOWS}


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
    if doc.get("format_revision") not in FREEZE_READABLE_REVISIONS or not isinstance(doc.get("models"), dict):
        raise FreezeError([f"{f}: not a format revision {' / '.join(map(str, FREEZE_READABLE_REVISIONS))} freeze"])
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
              "gating": False, "note": "disclosed since 2026-10-03; B_prime gates",
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


def failures(models: Mapping[str, Any], gates: Sequence[str]) -> list[str]:
    """One line per model and failed gating criterion, with every unmet threshold."""
    failed = []
    for model, r in models.items():
        for g in gates:
            crit = r["criteria"][g]
            if crit["passed"]:
                continue
            if g == "D":
                why = [str(x) for x in (crit["reasons"] or ["stop rule not satisfied"])]
            else:
                why = [f"{c['name']} {c['value']} {c['op']} {c['threshold']} not met"
                       for c in crit["criteria"] if not c["met"]]
                if not crit["evaluable"]:
                    why.insert(0, str(crit.get("reason") or "not evaluable on this M"))
            failed.append(f"{model}: {g} failed - " + "; ".join(why))
    return failed


def _write_validation_csv(path: Path, header: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(header))
        for r in rows:
            w.writerow([r.get(c, "") for c in header])


def window_end_index(csv_path: Path) -> dict[tuple[str, float], float]:
    """(scenario id, window start ms) -> window end ms of a window CSV (the instant the
    cross-model ranking pairs windows by); rows without the two columns are skipped."""
    out: dict[tuple[str, float], float] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                start, end = float(row["window_start_ms"]), float(row["window_end_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            out[((row.get("scenario_id") or "unknown").strip() or "unknown", start)] = end
    return out


def ranking_records(model: str, windows: Sequence[Any], csv_path: Path, *, theta: float, direction: str,
                    window_ms: float) -> list:
    """The windows as ``tre_calibration.ranking`` records: Z at ``theta``, severity = the
    label's ratio_max, instant = the CSV's window end (start + ``window_ms`` without one)."""
    from tre_calibration import ranking

    ends = window_end_index(csv_path)
    instants = []
    for w in windows:
        start = w.window_start_ms
        end = ends.get((w.scenario_id, float(start))) if start is not None else None
        instants.append(end if end is not None else (None if start is None else float(start) + window_ms))
    return ranking.records_from_windows(model, windows, theta=theta, direction=direction, instants=instants)


def b_prime_evaluation(windows: Sequence[Any], *, theta: float, tau_crit: float, direction: str,
                       window_ms: float, cfg: Mapping[str, Any], n_resamples: int, seed: int) -> dict:
    """Criterion B' (:mod:`scripts.b_prime`) of one model's windows: the gate at
    ``cfg["dwell_windows"]``, every dwell of :data:`scripts.b_prime.DISCLOSED_DWELL_WINDOWS`
    disclosed with its CI95. No cut (``cfg["severity_cut"]`` None) = not evaluable = not
    passed."""
    from scripts import b_prime
    from scripts import theta_verdict as tv

    dwell, cut, gate = int(cfg["dwell_windows"]), cfg.get("severity_cut"), cfg["gate"]
    out: dict[str, Any] = {"gating": True, "dwell_windows": dwell, "dwell_source": cfg.get("dwell_source"),
                           "severity_cut": cut, "cut_source": cfg.get("cut_source"),
                           "severity_quantile": b_prime.SEVERITY_QUANTILE, "gate": dict(gate)}
    if cut is None:
        out.update({"criteria": [], "evaluable": False, "passed": False,
                    "reason": cfg.get("missing") or "no severity cut"})
        return out
    by_dwell: dict[str, Any] = {}
    for d in sorted({dwell, *b_prime.DISCLOSED_DWELL_WINDOWS}):
        crit = tv.critical_dwell_flags(windows, theta=theta, tau_crit=tau_crit, direction=direction,
                                       dwell_windows=d, window_ms=window_ms)
        point = b_prime.series_point(windows, theta=theta, cut=cut, crit=crit)
        ci = b_prime.b_prime_boot(windows, cut=cut, crit=crit, n=n_resamples, seed=seed)
        by_dwell[str(d)] = {**point, "recall_severe_ci95": ci["recall_severe_ci95"],
                            "false_alarm_ci95": ci["false_alarm_ci95"]}
    g = by_dwell[str(dwell)]
    crits = b_prime.criteria(g, g, gate)
    shares = b_prime.band_shares(windows, theta=theta, cut=cut, tau_crit=tau_crit)
    evaluable = shares["severe_windows"] > 0 and shares["healthy"] > 0
    out.update({"criteria": crits, "evaluable": evaluable, "passed": evaluable and all(c["met"] for c in crits),
                **{k: shares[k] for k in ("violating", "healthy", "severe_windows", "violations_by_band")},
                "by_dwell": by_dwell,
                "disclosed_not_gating": ["recall_all (every violation)", "violations_by_band (LOW band = slow loop)",
                                         "missed_caught_by_slow_loop", "the dwells other than dwell_windows"]})
    return out


def accept_gate_problems(doc: Mapping[str, Any]) -> list[str]:
    """Review 2026-10-05 P2-2: a revision-4 freeze's sealed accept gate must be the running
    code's - the onset rule (no other gate for a new freeze), every numeric parameter, the
    episode / CI / look-back rules, the dynamic primitives, the undefined-ICC value, the
    window false alarm and the verdict rule. Any difference is a refusal (the gate decides
    with code, and the code must be the one that was sealed)."""
    rec = doc.get("accept_gate")
    if not isinstance(rec, Mapping):
        return []
    want = accept_gate_record(ACCEPT_GATE_ONSET)
    out = []
    if rec.get("rule") != ACCEPT_GATE_ONSET:
        out.append(f"the freeze seals accept gate {rec.get('rule')!r}; a revision-4 freeze has only {ACCEPT_GATE_ONSET!r}")
    for key in ("gating_criteria", "window_fa", "dwell_windows", "verdict_rule"):
        if not _same_doc(rec.get(key), want[key]):
            out.append(f"sealed accept_gate.{key} {rec.get(key)!r} != this code's {want[key]!r}")
    for key, value in want["onset"].items():
        if not _same_doc((rec.get("onset") or {}).get(key), value):
            out.append(f"sealed accept_gate.onset.{key} {(rec.get('onset') or {}).get(key)!r} != this code's {value!r}")
    return out


def _same_doc(a: Any, b: Any) -> bool:
    return canonical_json(a) == canonical_json(b) if not isinstance(a, (int, float)) else _same(a, b)


def accept_gate_of(doc: Mapping[str, Any]) -> dict:
    """The accept gate of a freeze: its sealed ``accept_gate`` (revision 4), else - an older
    freeze - the A, B', D gate with the onset-gate defaults for the disclosure."""
    from scripts import b_prime

    rec = doc.get("accept_gate")
    if isinstance(rec, Mapping):
        onset = b_prime.check_onset_gate(rec.get("onset") or {})
        fa = {k: float((rec.get("window_fa") or {})[k]) for k in b_prime.WINDOW_FA_GATE}
        return {"rule": rec.get("rule"), "onset": onset, "window_fa": fa, "source": "freeze accept_gate"}
    return {"rule": ACCEPT_GATE_B_PRIME, "onset": dict(b_prime.ONSET_GATE), "window_fa": dict(b_prime.WINDOW_FA_GATE),
            "source": f"defaults: the freeze (format revision {doc.get('format_revision')}) predates accept_gate; "
                      "the gate is A, B', D and the onset gate is disclosed"}


def verdict_of(criteria: Mapping[str, Any]) -> str:
    """:data:`VERDICT_RULE` on one model's criteria."""
    others = all(criteria[g]["passed"] for g in ONSET_GATING_CRITERIA if g != "A")
    if others and criteria["A"]["passed"]:
        return VERDICT_PASS
    return VERDICT_PASS_A_DISCLOSED if others else VERDICT_FAIL


def primitives_by_scenario(csv_path: Path) -> dict[str, str]:
    """scenario id -> the cell's primitive, from a validation CSV."""
    out: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.setdefault((row.get("scenario_id") or "unknown").strip() or "unknown", row.get("primitive") or "")
    return out


def evaluate_model(entry: Mapping[str, Any], csv_path: Path, *, n_resamples: int, seed: int,
                   records_sink: Optional[list] = None, b_prime_cfg: Optional[Mapping[str, Any]] = None,
                   gate_cfg: Optional[Mapping[str, Any]] = None) -> dict:
    """``theta_verdict.holdout_report`` for the point estimates (the freeze's dwell and
    window length - 2 x 30 s for a freeze that predates the record), the cell bootstrap for
    the CIs, then A-D; plus the ranking disclosure (AUROC, Kendall tau-b; never gating).
    ``records_sink`` receives the ranking records (for the pooled disclosure)."""
    from tre_calibration import ranking

    from scripts import theta_verdict as tv

    vh = entry["verdict_for_holdout"]
    win = windowing_of(entry)
    dwell, window_ms = int(win["dwell_windows"]), float(win["window_ms"])
    h = tv.holdout_report(vh, csv_path, dwell_windows=dwell, window_ms=window_ms)
    spec = tv.SignalSpec.from_dict(vh["signal_spec"])
    label = slo_labels.LabelDefinition.from_dict(vh["label_def"])
    windows = spec.load(csv_path, label, int(vh["trim_ramp_windows"]))
    theta, tau_crit = float(vh["published"]["theta_m"]), float(vh["published"]["tau_crit"])
    direction = vh["fit_config"]["direction"]
    crit = tv.critical_dwell_flags(windows, theta=theta, tau_crit=tau_crit, direction=direction,
                                   dwell_windows=dwell, window_ms=window_ms)
    boot = acceptance_bootstrap(windows, crit, theta=theta, direction=direction,
                                n_resamples=n_resamples, seed=seed)
    records = ranking_records(vh["model"], windows, csv_path, theta=theta, direction=direction,
                              window_ms=window_ms)
    if records_sink is not None:
        records_sink.extend(records)
    per_cell: dict[str, int] = defaultdict(int)
    for w in windows:
        per_cell[w.scenario_id] += 1
    criteria = acceptance_criteria(entry, h, boot)
    criteria["B_prime"] = b_prime_evaluation(
        windows, theta=theta, tau_crit=tau_crit, direction=direction, window_ms=window_ms,
        cfg=b_prime_cfg or {"dwell_windows": ONLINE_DWELL_WINDOWS, "severity_cut": None,
                            "gate": b_prime_gate_defaults(), "missing": "no B' configuration"},
        n_resamples=n_resamples, seed=seed)
    # 2026-10-05: the onset gate, the window false alarm and the A dead band - gating under an
    # onset freeze (verdict), disclosed under an older one
    from scripts import b_prime

    gate = dict(gate_cfg or {"rule": ACCEPT_GATE_B_PRIME, "onset": dict(b_prime.ONSET_GATE),
                             "window_fa": dict(b_prime.WINDOW_FA_GATE)})
    onset_rule = gate["rule"] == ACCEPT_GATE_ONSET
    bp = criteria["B_prime"]
    bp["gating"] = not onset_rule
    bp_dwell, cut = int(bp["dwell_windows"]), bp.get("severity_cut")
    prim = primitives_by_scenario(csv_path)
    if cut is None:
        criteria["onset"] = {"evaluable": False, "passed": False, "criteria": [], "reason": "no severity cut"}
        criteria["window_fa"] = {"evaluable": False, "passed": False, "criteria": [], "reason": "no severity cut"}
    else:
        onset_by_dwell = {}
        for d in sorted({bp_dwell, *b_prime.DISCLOSED_DWELL_WINDOWS}):
            cd = tv.critical_dwell_flags(windows, theta=theta, tau_crit=tau_crit, direction=direction,
                                         dwell_windows=d, window_ms=window_ms)
            onset_by_dwell[d] = b_prime.onset_detection(windows, cd, cut=cut, primitive_of=prim, gate=gate["onset"],
                                                        n_resamples=n_resamples, seed=seed)
        criteria["onset"] = {**onset_by_dwell[bp_dwell], "gating": onset_rule, "dwell_windows": bp_dwell,
                             "disclosed_dwells": {str(d): {k: o.get(k) for k in ("onset", "passed", "success_rate",
                                                                                   "n_eff", "icc")}
                                                  for d, o in onset_by_dwell.items() if d != bp_dwell}}
        g = bp["by_dwell"][str(bp_dwell)]
        criteria["window_fa"] = {**b_prime.window_fa(g, g, gate["window_fa"]), "gating": onset_rule,
                                 "dwell_windows": bp_dwell}
    dead = gate.get("a_deadband") or entry.get("a_deadband") or {}
    s0 = _finite_or_none(dead.get("s0"))
    criteria["A_deadband"] = ({**b_prime.deadband_ba(windows, theta=theta, s0=s0, direction=direction),
                               "gating": False, "s0_source": dead.get("source", "freeze a_deadband")}
                              if s0 is not None else {"gating": False, "s0": None,
                                                      "reason": "no s0 (freeze has no a_deadband)"})
    verdict = verdict_of(criteria)
    return {"holdout_report": h, "bootstrap": boot, "criteria": criteria,
            "passed": (verdict != VERDICT_FAIL) if onset_rule else all(criteria[g]["passed"] for g in GATING_CRITERIA),
            "gate_rule": gate["rule"], "verdict": verdict if onset_rule else {"disclosed": verdict, "gating": False},
            "windowing": win,
            "ranking_disclosure": ranking.ranking_disclosure(records, n_resamples=n_resamples, seed=seed),
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
    frozen_numerators = {m_: str(e.get("numerator") or "gateway") for m_, e in sorted(doc["models"].items())}
    for s in sources:
        have = dataset_numerator(s.directory)
        wrong = sorted(m_ for m_, n in frozen_numerators.items() if n != have)
        if wrong:
            problems.append(f"dataset {s.name} ({s.directory}) was built with the {have!r} TSS numerator; the "
                            f"freeze fitted {wrong} with {sorted({frozen_numerators[m_] for m_ in wrong})} "
                            "(rebuild it with calibration_dataset --numerator)")
    frozen_attr = {m_: label_attribution((e.get("verdict_for_holdout") or {}).get("label_def"))
                   for m_, e in sorted(doc["models"].items())}
    for s in sources:
        have = dataset_attribution(s.directory)
        wrong = sorted(m_ for m_, a in frozen_attr.items() if a != have)
        if wrong:
            problems.append(f"dataset {s.name} ({s.directory}) carries {have!r} label attribution; the freeze's "
                            f"label for {wrong} is {sorted({frozen_attr[m_] for m_ in wrong})} (use the matching "
                            "dataset directory, e.g. dataset_hybrid/, or rebuild with calibration_dataset --attribution)")
    problems += [f"freeze: {x}" for x in accept_gate_problems(doc)]
    for model_, entry_ in sorted(doc["models"].items()):
        bp_ = entry_.get("b_prime")
        if not isinstance(entry_.get("a_deadband"), Mapping) and isinstance(bp_, Mapping):
            # review 2026-10-05 P3-10: an older freeze's s0 is computed at accept from its training
            # CSV - only from the very file the freeze sealed
            tcsv = Path(bp_.get("training_csv") or "")
            if not tcsv.is_file():
                problems.append(f"{model_}: the freeze's training CSV {tcsv} (A dead band s0) is missing")
            elif sha256_file(tcsv) != bp_.get("training_csv_sha256"):
                problems.append(f"{model_}: the freeze's training CSV {tcsv} changed after the freeze "
                                "(the A dead band's s0 is computed from it)")
    m: dict = {"header": {}, "rows": {}, "placed": {}}
    if not problems:
        m, pr = collect_m_rows(sources, manifests)
        problems += pr
    cuts: dict[str, dict] = {}
    if not problems:
        cuts, pr = m2_stream_cut_audits(manifests, sources, m)
        problems += pr
    return {"doc": doc, "freeze_sha256": freeze_sha, "manifests": manifests, "manifest_paths": manifest_paths,
            "sources": sources, "m": m, "stream_cut": cuts}, problems


#: The M2 composition whose manifests carry the stream-cut rule (calibration_acceptance).
M2_COMPOSITION_NAME = "m2-20261005"


def m2_stream_cut_audits(manifests: Mapping[str, Mapping[str, Any]], sources: Sequence[DatasetSource],
                         m: Mapping[str, Any]) -> tuple[dict, list[str]]:
    """User 2026-10-05: M2 is judged under T14's stream-cut rule (:mod:`scripts.stream_cut`,
    the T14 scorer's own function). For each M2 manifest (``composition_name``
    :data:`M2_COMPOSITION_NAME`): the manifest must bind the 0.10 runtime limit; every cell's
    evaluated attempt is audited over its dataset's ``requests.csv`` (route-timeout cuts are
    censored, not errors - their windows stay violations through the unserved counts); a cell
    whose non-cut errors exceed 0.05 of its requests is void at audit, excluded from the
    evaluation and listed. Other manifests: nothing (M of 2026-10-03 and older)."""
    from scripts import stream_cut

    out: dict[str, dict] = {}
    problems: list[str] = []
    by_name = {s.name: s for s in sources}
    for model, man in sorted(manifests.items()):
        if man.get("composition_name") != M2_COMPOSITION_NAME:
            continue
        rec = man.get("stream_cut") or {}
        if not _same(rec.get("max_model_error_rate"), stream_cut.RUNTIME_MODEL_ERROR_LIMIT):
            problems.append(f"{model}: the M2 manifest binds max_model_error_rate {rec.get('max_model_error_rate')!r}, "
                            f"not the stream-cut rule's {stream_cut.RUNTIME_MODEL_ERROR_LIMIT:g}")
            continue
        per_ds: dict[str, dict] = defaultdict(dict)
        for c in man["cells"]:
            key = (model, str(c["cell_id"]), _attempt(c["attempt"]))
            placed = (m.get("placed") or {}).get(key) or {}
            per_ds[placed.get("dataset", "")][(str(c["cell_id"]), int(_attempt(c["attempt"])))] = {
                "shape": c.get("shape"), "primitive": c.get("primitive")}
        cells, rule = [], None
        for name, keys in sorted(per_ds.items()):
            src = by_name.get(name)
            req = (src.directory / "requests.csv") if src else None
            if req is None or not req.is_file():
                problems.append(f"{model}: no requests.csv for the M2 cells of dataset {name!r} (stream-cut audit)")
                continue
            a = stream_cut.audit(req, keys)
            rule = a["rule"]
            cells += [{**c, "dataset": name} for c in a["cells"]]
        void = sorted((c["cell_id"], c["attempt"]) for c in cells if c["void_at_audit"])
        out[model] = {"rule": rule, "manifest_record": rec, "cells": cells,
                      "audit_void_cells": [{"cell_id": c, "attempt": a} for c, a in void],
                      "excluded_from_evaluation": [c for c, _a in void],
                      "totals": {k: sum(c[k] for c in cells) for k in ("sent", "model_error", "cut", "non_cut")}}
    return out, problems


def b_prime_inputs(doc: Mapping[str, Any], thresholds_file: Optional[Path], dwell_windows: int,
                   dwell_source: str) -> tuple[dict, dict, list[str]]:
    """``({model: B' config}, summary, problems)`` for accept. The cut of each model
    comes from its freeze entry (revision 3, sealed at freeze time) or - for a freeze
    that predates it, and only then - from ``thresholds_file``; a model with neither is a
    problem (B' is never judged against a cut taken from the data it judges)."""
    from scripts import b_prime

    problems: list[str] = []
    models = doc.get("models") or {}
    sealed = {m: e["b_prime"] for m, e in models.items() if isinstance(e.get("b_prime"), dict)}
    file_rec, file_info = None, None
    if thresholds_file is not None:
        if sealed:
            problems.append(f"--b-prime-thresholds {thresholds_file}: the freeze seals its own B' cuts "
                            f"({sorted(sealed)}); the file is only for a freeze that predates them")
        else:
            try:
                file_rec = b_prime.load_thresholds(Path(thresholds_file))
                file_info = {"path": str(thresholds_file), "sha256": sha256_file(Path(thresholds_file))}
            except ValueError as exc:
                problems.append(f"--b-prime-thresholds: {exc}")
    gate = None
    if sealed:
        try:
            gate = b_prime.check_gate(doc.get("b_prime_gate") or {})
        except ValueError as exc:
            problems.append(f"the freeze's b_prime_gate: {exc}")
    elif file_rec is not None:
        gate = file_rec["gate"]
    frozen_dwell = (doc.get("b_prime_gate") or {}).get("online_dwell_windows")
    cfgs: dict[str, dict] = {}
    for m in sorted(models):
        cfg: dict[str, Any] = {"dwell_windows": int(dwell_windows), "dwell_source": dwell_source,
                               "gate": gate or dict(b_prime.DEFAULT_GATE), "severity_cut": None}
        if m in sealed:
            cfg["severity_cut"] = _finite_or_none(sealed[m].get("severity_cut"))
            cfg["cut_source"] = {"source": "freeze", "freeze_sha256": doc.get("freeze_sha256"),
                                 **{k: sealed[m].get(k) for k in ("severity_quantile", "training_csv",
                                                                   "training_csv_sha256",
                                                                   "violating_training_windows")}}
            if cfg["severity_cut"] is None:
                problems.append(f"{m}: the freeze's B' severity cut {sealed[m].get('severity_cut')!r} is not a number")
        elif file_rec is not None and m in file_rec["severity_cut_train"]:
            cfg["severity_cut"] = file_rec["severity_cut_train"][m]
            cfg["cut_source"] = {"source": "thresholds_file", **file_info}
        elif file_rec is not None:
            problems.append(f"{m}: --b-prime-thresholds {thresholds_file} has no severity_cut_train for it")
        else:
            cfg["missing"] = (f"the freeze (format revision {doc.get('format_revision')}) seals no B' severity cut "
                              f"for {m}")
            problems.append(f"{m}: {cfg['missing']}: pass --b-prime-thresholds <file decided before M> "
                            "(severity_cut_train + gate), or accept under a revision-3 freeze")
        cfgs[m] = cfg
    summary = {**(gate or dict(b_prime.DEFAULT_GATE)), "dwell_windows": int(dwell_windows),
               "dwell_source": dwell_source, "disclosed_dwell_windows": list(b_prime.DISCLOSED_DWELL_WINDOWS),
               "severity_quantile": b_prime.SEVERITY_QUANTILE, "severity": b_prime.SEVERITY_RULE,
               "cut_source": "freeze" if sealed else ("thresholds_file" if file_rec else None),
               "thresholds_file": file_info, "freeze_online_dwell_windows": frozen_dwell}
    return cfgs, summary, problems


def _accept_result(freeze_file: Path, inp: Mapping[str, Any], work: Path, *, n_resamples: int, seed: int,
                   command: Sequence[str]) -> dict:
    doc, m = inp["doc"], inp["m"]
    bp_cfgs, bp_summary = inp["b_prime"], inp["b_prime_summary"]
    sums_cover: set[str] = set()
    for model, man in inp["manifests"].items():
        mdir = inp["manifest_paths"][model].parent
        for line in (mdir / str(man["sha256sums_file"])).read_text(encoding="utf-8").splitlines():
            if line.strip():
                name = _parse_sums_line(line)[1]
                sums_cover.add(str((Path(name) if Path(name).is_absolute() else mdir / name).resolve()))
    datasets = [{"name": s.name, "directory": str(s.directory), "windows_csv": str(s.windows),
                 "windows_csv_sha256": sha256_file(s.windows),
                 "covered_by_m_sha256sums": str(Path(s.windows).resolve()) in sums_cover}
                for s in inp["sources"]]
    from tre_calibration import ranking

    gate = accept_gate_of(doc)
    onset_rule = gate["rule"] == ACCEPT_GATE_ONSET
    models: dict[str, Any] = {}
    records: list = []
    for model in sorted(doc["models"]):
        entry, man = doc["models"][model], inp["manifests"][model]
        mpath = inp["manifest_paths"][model]
        csv_path = work / f"{model}_validation.csv"
        cut = (inp.get("stream_cut") or {}).get(model)
        rows = m["rows"][model]
        if cut:
            # M2 stream-cut audit: audit-void cells are excluded (and listed in the result)
            gone = {(d["cell_id"], int(d["attempt"])) for d in cut["audit_void_cells"]}
            rows = [r for r in rows if (r["cell_id"], int(_attempt(r["attempt"]))) not in gone]
        _write_validation_csv(csv_path, m["header"][model], rows)
        gcfg = dict(gate)
        if not isinstance(entry.get("a_deadband"), Mapping) and isinstance(entry.get("b_prime"), Mapping):
            # an older freeze: s0 from its own (hash-checked) training CSV, never from M
            gcfg["a_deadband"] = {**deadband_freeze_record(entry["verdict_for_holdout"],
                                                           Path(entry["b_prime"]["training_csv"])),
                                  "source": "computed at accept from the frozen training CSV (the freeze predates "
                                            "a_deadband)"}
        ev = evaluate_model(entry, csv_path, n_resamples=n_resamples, seed=seed, records_sink=records,
                            b_prime_cfg=bp_cfgs.get(model), gate_cfg=gcfg)
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
            "validation_rows": len(rows),
            "published": entry["published"],
            **ev,
        }
        if cut:
            models[model]["stream_cut_audit"] = cut
    if onset_rule:
        failed = failures({k: r for k, r in models.items() if r["verdict"] == VERDICT_FAIL}, ONSET_GATING_CRITERIA)
        limitations = [x for x in failures({k: r for k, r in models.items()
                                            if r["verdict"] == VERDICT_PASS_A_DISCLOSED}, ("A",))]
        verdicts = {k: r["verdict"] for k, r in models.items()}
        overall = (VERDICT_FAIL if VERDICT_FAIL in verdicts.values() else
                   VERDICT_PASS_A_DISCLOSED if VERDICT_PASS_A_DISCLOSED in verdicts.values() else VERDICT_PASS)
    else:
        failed = failures(models, GATING_CRITERIA)
        limitations, overall = [], {"disclosed": {k: r["verdict"]["disclosed"] for k, r in models.items()},
                                    "gating": False}
    wins = {model: windowing_of(e) for model, e in doc["models"].items()}
    dwells = {model: w["dwell_windows"] for model, w in wins.items()}
    steps = {w["step_ms"] for w in wins.values()}
    step_bin = steps.pop() if len(steps) == 1 else STEP_MS
    pooled = ranking.ranking_disclosure(records, n_resamples=n_resamples, seed=seed,
                                        cross_model_bins=(None, step_bin))
    pooled["note"] = ("pooled over the models' M windows (cells stratified by model); the cross-model "
                      "tau_b is disclosed exact and with window ends rounded to the re-window step")
    return {
        "what": (("acceptance on M, evaluated once on the frozen parameters: onset episodes, window false "
                  "alarm, A and D gate with a three-way verdict (design 2026-10-05 item 3, user decision 2); "
                  "B', the old B, C, dwell 2 and the A dead band are disclosed") if onset_rule else
                 ("plan §6.9f acceptance A-D on M, evaluated once on the frozen parameters "
                  "(A, B' and D gate - B' since 2026-10-03; the old B, C and the all-violating "
                  "recall are disclosed)")),
        "gate_rule": {"rule": gate["rule"], "source": gate["source"],
                      "gating_criteria": list(ONSET_GATING_CRITERIA if onset_rule else GATING_CRITERIA)},
        "verdict": overall,
        "disclosed_limitations": limitations,
        "attribution": {"datasets": {s.name: dataset_attribution(s.directory) for s in inp["sources"]},
                        "frozen_labels": {k: label_attribution(e["verdict_for_holdout"]["label_def"])
                                          for k, e in sorted(doc["models"].items())}},
        "format_revision": ACCEPT_FORMAT_REVISION,
        "evaluated_at_utc": _utc_now(),
        "command": list(command),
        "code": code_state(),
        "freeze": {"path": str(freeze_file), "sha256": inp["freeze_sha256"],
                   "freeze_sha256": doc["freeze_sha256"], "arm": doc.get("arm")},
        "thresholds": {"A": {"ba_min": A_BA_MIN, "ba_ci_low_min": A_BA_CI_LOW_MIN,
                             "max_drop_from_training": A_MAX_DROP_FROM_TRAINING},
                       "B_prime": bp_summary,
                       "B": {"gating": False, "recall_min": B_RECALL_MIN, "recall_ci_low_min": B_RECALL_CI_LOW_MIN,
                             "false_alarm_max": B_FALSE_ALARM_MAX,
                             "false_alarm_ci_high_max": B_FALSE_ALARM_CI_HIGH_MAX,
                             "all_violating_recall_target": ALL_VIOLATING_RECALL_TARGET},
                       "C": {"windows_per_independent": WINDOWS_PER_INDEPENDENT},
                       "accept_gate": {**gate, "verdict_rule": VERDICT_RULE},
                       "dwell_windows": (next(iter(set(dwells.values()))) if len(set(dwells.values())) == 1
                                         else dict(sorted(dwells.items())))},
        "bootstrap": {"n_resamples": n_resamples, "seed": seed},
        "datasets": datasets,
        "work_dir": str(work),
        "models": models,
        "ranking_disclosure": pooled,
        "passed": not failed,
        "failed": failed,
    }


def as_revision(result: Mapping[str, Any], revision: Any) -> dict:
    """``result`` without the keys revisions after ``revision`` added
    (:data:`ACCEPT_REVISION_KEYS`) and with its ``format_revision``: what a result of that
    older revision holds, so a ``--recheck`` of it compares everything it has."""
    out = json.loads(json.dumps(result))
    for rev, keys in sorted(ACCEPT_REVISION_KEYS.items()):
        if not isinstance(revision, int) or revision >= rev:
            continue
        for k in keys["top"]:
            out.pop(k, None)
        for r in (out.get("models") or {}).values():
            for k in keys["model"]:
                r.pop(k, None)
    if not isinstance(revision, int) or revision < 4:
        for r in (out.get("models") or {}).values():
            for k in ACCEPT_REV4_CRITERIA:
                (r.get("criteria") or {}).pop(k, None)
        for k in ACCEPT_REV4_THRESHOLDS:
            (out.get("thresholds") or {}).pop(k, None)
    if not isinstance(revision, int) or revision < 3:
        # before 2026-10-03 the gate was A, B, D and B' did not exist
        for r in (out.get("models") or {}).values():
            crit = r.get("criteria") or {}
            crit.pop("B_prime", None)
            for k in ("gating", "note"):
                (crit.get("B") or {}).pop(k, None)
            if all(g in crit for g in LEGACY_GATING_CRITERIA):
                r["passed"] = all(crit[g]["passed"] for g in LEGACY_GATING_CRITERIA)
        th = out.get("thresholds") or {}
        th.pop("B_prime", None)
        (th.get("B") or {}).pop("gating", None)
        out["what"] = LEGACY_ACCEPT_WHAT
        out["failed"] = failures(out.get("models") or {}, LEGACY_GATING_CRITERIA)
        out["passed"] = not out["failed"]
    out["format_revision"] = revision
    return out


def print_ranking_table(result: Mapping[str, Any]) -> None:
    """The ranking disclosure of an accept result as a small table (not gating)."""
    from tre_calibration import ranking

    blocks = {m: r["ranking_disclosure"] for m, r in (result.get("models") or {}).items()
              if r.get("ranking_disclosure")}
    if result.get("ranking_disclosure"):
        blocks["pooled"] = result["ranking_disclosure"]
    if not blocks:
        return
    print("ranking disclosure (pressure = -Z; not gating):")
    for row in ranking.disclosure_table(blocks):
        print(f"  {row}")


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
                 command: Sequence[str] = (), b_prime_thresholds: Optional[Path] = None,
                 dwell_windows: Optional[int] = None) -> int:
    """Plan §6.9f A-D, once. Returns 0 = evaluated and passed, :data:`EXIT_ACCEPT_FAILED`
    = evaluated and failed, :data:`EXIT_REFUSED` = refused (nothing written); with
    ``recheck``: 0 = the stored result reproduces, :data:`EXIT_RECHECK_DIFFERS` = not."""
    import shutil
    import tempfile

    freeze_file = Path(freeze_file)
    fp = freeze_paths(freeze_file)
    problems: list[str] = []
    stored_bytes, stored = b"", {}
    if recheck:
        if not fp["result"].exists():
            problems.append(f"{fp['result']} does not exist: nothing to recheck")
        else:
            stored_bytes = fp["result"].read_bytes()
            try:
                stored = json.loads(stored_bytes.decode("utf-8"))
            except ValueError as exc:
                problems.append(f"{fp['result']}: not JSON ({exc})")
    else:
        problems += [f"{fp[k]} already exists: M is evaluated once (--recheck reproduces it)"
                     for k in ("result", "marker", "work") if fp[k].exists()]
    stored_rev = stored.get("format_revision") if stored else None
    stored_bp = ((stored.get("thresholds") or {}).get("B_prime") or {}) if stored else {}
    if recheck and stored_bp.get("dwell_windows") is not None:
        if dwell_windows is not None and int(dwell_windows) != int(stored_bp["dwell_windows"]):
            print(f"note: --recheck uses the stored B' dwell {stored_bp['dwell_windows']}, not --dwell-windows "
                  f"{dwell_windows}")
        dwell, dwell_source = int(stored_bp["dwell_windows"]), stored_bp.get("dwell_source")
    elif dwell_windows is not None:
        dwell, dwell_source = int(dwell_windows), "--dwell-windows"
    else:
        dwell, dwell_source = ONLINE_DWELL_WINDOWS, ("default: ONLINE_DWELL_WINDOWS = the controller's "
                                                     "TRE_DWELL_WINDOWS (deploy/overlays/tre-v2/controller.yaml)")
    inp, pr = _accept_inputs(freeze_file, datasets, m_manifests)
    problems += pr
    if "doc" in inp:
        bp_cfgs, bp_summary, bp_problems = b_prime_inputs(inp["doc"], b_prime_thresholds, dwell, dwell_source)
        # a recheck of a result from before B' existed does not need a cut
        if not recheck or not isinstance(stored_rev, int) or stored_rev >= 3:
            problems += bp_problems
        inp["b_prime"], inp["b_prime_summary"] = bp_cfgs, bp_summary
        frozen = bp_summary.get("freeze_online_dwell_windows")
        if frozen is not None and int(frozen) != dwell:
            print(f"WARNING: B' is judged at dwell {dwell} ({dwell_source}); the freeze recorded the "
                  f"controller's dwell as {frozen}")
    if problems:
        print("accept REFUSED - nothing was written:")
        for x in problems:
            print(f"  - {x}")
        return EXIT_REFUSED
    if recheck:
        n_resamples, seed = int(stored["bootstrap"]["n_resamples"]), int(stored["bootstrap"]["seed"])
        with tempfile.TemporaryDirectory(prefix="dline_accept_recheck_") as tmp:
            new = _accept_result(freeze_file, inp, Path(tmp), n_resamples=n_resamples, seed=seed, command=command)
        new = json.loads(_json_bytes(new).decode("utf-8"))
        print_ranking_table(new)
        if stored_rev != new.get("format_revision"):
            print(f"note: the stored result is format revision {stored_rev}; keys added since "
                  f"({sorted({k for r, ks in ACCEPT_REVISION_KEYS.items() if not isinstance(stored_rev, int) or r > stored_rev for k in ks['top'] + ks['model']})}) "
                  "are not compared")
            new = as_revision(new, stored_rev)
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
    bp = result["thresholds"]["B_prime"]
    onset_rule = result["gate_rule"]["rule"] == ACCEPT_GATE_ONSET
    print(f"B' ({'disclosed' if onset_rule else 'gating'}) judged at dwell {bp['dwell_windows']} "
          f"({bp['dwell_source']}); severity cut from "
          f"{bp['cut_source']}; dwell {', '.join(map(str, bp['disclosed_dwell_windows']))} disclosed")
    gates = ONSET_GATING_CRITERIA if onset_rule else GATING_CRITERIA
    print(f"gate: {result['gate_rule']['rule']} ({', '.join(gates)})")
    for model, r in result["models"].items():
        c = r["criteria"]
        g2 = (c["B_prime"].get("by_dwell") or {}).get("2") or {}
        on = (c.get("onset") or {}).get("onset") or {}
        print(f"[{model}] M {r['M']['windows']} windows / {r['M']['cells']} cells: "
              + " ".join(f"{g}={'pass' if c[g]['passed'] else 'FAIL'}" for g in gates)
              + (f" -> verdict {r['verdict']}" if onset_rule else "")
              + f" (onset episodes {on.get('within_budget')}/{on.get('episodes')} within budget, "
                f"n_eff {c.get('onset', {}).get('n_eff')}; B' {'pass' if c['B_prime']['passed'] else 'fail'})")
        print(f"[{model}]"
              + f" (disclosed: B' dwell 2 recall {g2.get('recall_severe')} / false alarm {g2.get('false_alarm')}; "
                f"old B {'pass' if c['B']['passed'] else 'fail'}; "
                f"C TTFT-only recall {c['C']['critical_recall_ttft_only']} on {c['C']['windows']} windows)")
    print_ranking_table(result)
    print(f"wrote {fp['result']} and {fp['marker']}")
    for x in result.get("disclosed_limitations") or []:
        print(f"  limitation (disclosed, not failing): {x}")
    if not result["passed"]:
        print("acceptance FAILED:")
        for x in result["failed"]:
            print(f"  - {x}")
        return EXIT_ACCEPT_FAILED
    print("acceptance passed" + (f" (verdict {result['verdict']})" if onset_rule else ""))
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
    ap.add_argument("--train-dynamic", action="store_true",
                    help="trainset: the train-split dynamic cells (steps / ramp / bursts) train with the "
                         "constant-load cells, equal weight per window (design 2026-10-05 item 2; recorded "
                         "in trainset.json); default: D16, constant-load cells only")
    ap.add_argument("--no-sentinels", action="store_true",
                    help="trainset: leave the sentinel cells out of the training set (default: they train)")
    ap.add_argument("--alpha-rule", choices=ALPHA_RULES, default="d4prime")
    ap.add_argument("--max-ci-half-width-fraction", type=float, default=None,
                    help="the D13 stop rule's CI gate (fraction of theta; default "
                         "adaptive_boundary.MAX_CI_HALF_WIDTH_FRACTION = 0.20 since 2026-09-24; "
                         "0.15 reproduces the rule before that)")
    ap.add_argument("--v1-wp-grid", choices=("with_zero", "v1"), default="with_zero",
                    help="wp --lambda-method v1: w_p grid of stages B / C - with_zero (default, user "
                         "2026-09-24): v1's 0.01..0.08 / 0.005 plus 0; v1: v1's grid as is")
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
    ap.add_argument("--b-prime-thresholds", type=Path, default=None,
                    help="accept, only under a freeze older than format revision 3 (no sealed B' cut): "
                         "the B' thresholds decided before M (severity_cut_train {model: cut} + gate, "
                         "e.g. b_prime_thresholds.json of 2026-09-24); refused with a revision-3 freeze")
    win_group = ap.add_argument_group(
        "windowing (alpha / wp / final; recorded in the stage outputs and the freeze, which accept reads)")
    win_group.add_argument("--window-ms", type=float, default=WINDOW_MS,
                           help=f"window length of the fit CSVs (default {WINDOW_MS:g})")
    win_group.add_argument("--step-ms", type=float, default=STEP_MS,
                           help=f"re-window step (default {STEP_MS:g})")
    win_group.add_argument("--dt-ref-s", type=float, default=DT_REF_S,
                           help=f"EMA reference period: alpha = 1 - exp(-dt_ref / tau) (default {DT_REF_S:g})")
    win_group.add_argument("--horizon-ms", type=int, default=HORIZON_MS,
                           help=f"refit0922 label horizon / detection look-back (default {HORIZON_MS})")
    win_group.add_argument("--dwell-windows", type=int, default=None,
                           help=f"alpha / wp / final: CRITICAL dwell in new windows (default {DWELL_WINDOWS}); "
                                f"accept: the dwell B' is judged at (default {ONLINE_DWELL_WINDOWS} = the "
                                "controller's TRE_DWELL_WINDOWS; dwell 1 and 2 are always disclosed)")
    args = ap.parse_args(argv)
    if args.dwell_windows is not None and args.dwell_windows < 1:
        ap.error("--dwell-windows must be >= 1")
    try:
        win = windowing(window_ms=args.window_ms, step_ms=args.step_ms, dt_ref_s=args.dt_ref_s,
                        horizon_ms=args.horizon_ms,
                        dwell_windows=DWELL_WINDOWS if args.dwell_windows is None else args.dwell_windows)
    except ValueError as exc:
        ap.error(str(exc))
    command = ["python", "-m", "scripts.dline_refit", *(sys.argv[1:] if argv is None else argv)]
    global D13_MAX_CI_FRACTION
    D13_MAX_CI_FRACTION = args.max_ci_half_width_fraction

    if args.stage in ("verify-freeze", "accept"):
        if args.freeze_file is None:
            ap.error(f"{args.stage} needs --freeze-file")
        if args.stage == "accept":
            if {k: v for k, v in win.items() if k != "dwell_windows"} != {
                    k: v for k, v in DEFAULT_WINDOWING.items() if k != "dwell_windows"}:
                # accept scores with the windowing recorded in the freeze, never the command line's
                print(f"WARNING: accept ignores the windowing flags ({win}); it uses the windowing "
                      "recorded in the freeze (--dwell-windows is the B' gate's dwell)")
            return stage_accept(args.freeze_file, args.dataset, args.m_manifest, recheck=args.recheck,
                                n_resamples=args.accept_resamples, command=command,
                                b_prime_thresholds=args.b_prime_thresholds, dwell_windows=args.dwell_windows)
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
                                     models=args.model or None, train_dynamic=args.train_dynamic)
        except TrainingSetError as exc:
            raise SystemExit(f"trainset: {exc}")
        print(json.dumps({"attribution": doc["attribution"], "dynamic_training": doc["dynamic_training"]["admitted"],
                          "models": {m: {k: v[k] for k in ("windows", "cells", "family_windows", "dynamic")}
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
            ab, pl = e["absolute_thresholds"], (e["theta_selection"].get("plateau") or {})
            print(f"[{model}] theta={q['theta']:.6g} w_p={q['w_p']:g} tau_s={q['tau_s']:g} "
                  f"lambda_wait={q['lambda_wait']:g} delta_crit={q['delta_crit']:g} delta_high={q['delta_high']:g} "
                  f"CRIT abs={ab['critical_abs']} HIGH abs={ab['high_abs']} "
                  f"plateau=[{pl.get('theta_lo')}, {pl.get('theta_hi')}] s0={e['a_deadband'].get('s0')} "
                  f"attribution={e['label_attribution']}")
        print(f"accept gate: {doc['accept_gate']['rule']}")
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
    # the label of the training datasets' attribution (trainset.json; completion without one)
    label = label_for(model, args.arm, args.registry, trainset_attribution(fit_dir(model)))
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
            doc = stage_alpha_refit0922(model, label, p, w_p=w_p, win=win)
        else:
            doc = stage_alpha_d4prime(model, label, p, w_p=w_p, ledgers=lp,
                                      bootstrap=args.alpha_bootstrap, registry=args.registry, win=win)
        doc = publish_alpha(doc, args.publish_tau_s, dt_ref_s=win["dt_ref_s"])
    elif args.stage == "wp" and args.lambda_method == "v1":
        from scripts import v1_lambda_fit

        sources = v1_lambda_fit.sources_from_trainset(fit_dir(model))
        for text in args.requests_dataset:
            run, sep, d = text.partition("=")
            if not sep or not run or not d:
                ap.error(f"--requests-dataset {text!r}: expected RUN=DIR")
            sources[run] = Path(d)
        doc = stage_wp_v1(model, label, p, _read_json(out / "alpha.json"), sources, wp_grid=args.v1_wp_grid)
    elif args.stage == "wp":
        doc = stage_wp(model, label, p, _read_json(out / "alpha.json"))
    else:
        doc = stage_final(model, label, p, _read_json(out / "wp.json"), out, holdout=not args.no_holdout, win=win)
    inputs = {k: str(v) for k, v in p.items() if k != "families"}
    if args.stage == "final" and args.no_holdout:
        inputs["validation"] = HOLDOUT_SKIPPED
    doc.update({"model": model, "arm": args.arm, "label_def": label.as_dict(), "training_set": training,
                "windowing": win,
                "inputs": inputs | {f"family_{k}": str(v) for k, v in p["families"].items()}})
    (out / f"{args.stage}.json").write_text(json.dumps(doc, indent=1, default=str))
    print(f"wrote {out / (args.stage + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""T14 scorer - the preregistered evaluation of the held-out 14b set.

Two rule versions, chosen by the preregistration itself (never by a flag):

* **v1** - round 2026-10-03 (``$CALIB_ROOT/t14/preregistration.json`` + the stream-cut and
  cross-shape addenda; scoring decisions D1-D5 of ``ADDENDUM-T14-scoring``). Gates A and
  B' (dwell 1). Kept byte-for-byte in behaviour so the sealed ``eval/T14_score.json``
  reproduces (a v1 output gains only ``label_attribution`` and ``rule_version``).
* **v2** - the next round (design 2026-10-05, ``schema`` = :data:`PREREG_SCHEMA_V2`): the
  stream-cut rule, the cross-shape claim rule and D1-D5 are INLINE in the preregistration
  (``t14.stream_cut``, ``evaluation.cross_shape``, ``evaluation.scoring``); no addendum is
  read (one is refused). The gate set follows the new acceptance on steady holds:

  - A (BA >= .80, CI95 low >= .75, drop from training <= .08) - a gate; the verdict is the
    accept's three-way one: ``pass`` (A and FA), ``pass_a_disclosed`` (only A fails - the
    pre-declared consequence ``evaluation.outcome_statements.pass_a_disclosed``: go-live
    allowed, A disclosed as a limitation), ``fail`` (anything else);
  - FA: CRITICAL false alarm on healthy windows at dwell 1 <= .05, CI95 high <= .08;
  - onset episodes: NOT APPLICABLE - T14 is 24 steady holds and has no dynamic cell, so it
    has no overload onset to catch (the onset gate is judged on M2's dynamic cells);
  - D (theta CI half width) is a property of the training fit, not of T14 (as in v1);
  - window B' (severe recall at the freeze's sealed cut) is DISCLOSED, not gating;
  - the cross-shape claim (D1, D2) is sealed as a claim rule, not a gate.

Label attribution (both versions): the window label's request-to-window attribution
(``completion`` = every label so far; ``hybrid`` = label v2, TTFT by first token) is read
from the frozen label (``verdict_for_holdout.label_def["attribution"]``, absent =
completion), the dataset manifest (``attribution.value``, absent = completion), the T14
manifest's label, the v2 preregistration (``t14.conditions.label_attribution``) and the
dry-run record. They must all agree, or the scorer refuses: T14 is scored under ONE
attribution, the one the dataset carries, and never mixes them.

v1 rule text follows (unchanged).

The rule, as preregistered (``$CALIB_ROOT/t14/preregistration.json``, sha256 59608505...,
key ``evaluation``) - quoted:

    "times": "once on T14, with the freeze parameters; nothing is tuned on T14"
    "data": "the sealed T14 dataset (T14_manifest.json cells only, the valid attempt),
             dsqwen-14b, gateway-numerator dataset/"
    "A": {"metric": "window balanced accuracy at the published theta (no dwell)",
          "ci": "cell bootstrap, 1000 resamples, seed 20260922, 95 % percentile",
          "gates_reported": {"ba_min": 0.8, "ba_ci95_low_min": 0.75,
                             "max_drop_from_training_ba": 0.08}}
    "B_prime": {"cut": "the freeze's sealed b_prime.severity_cut (training .65 quantile)",
                "dsqwen-14b_cut": 8.80764, "gate": {... "false_alarm_ci95_high_max": 0.08,
                "false_alarm_max": 0.05, "online_dwell_windows": 1,
                "recall_severe_ci95_low_min": 0.7, "recall_severe_min": 0.8, ...},
                "dwell": "judged at dwell 1 (the controller's TRE_DWELL_WINDOWS); dwell 2 disclosed"}
    "also_reported": ["AUROC of Z vs SLO label", "Kendall tau-b ranking disclosure",
                      "C: TTFT-only recall (disclosure)",
                      "per-shape BA, interpolation and extrapolation separately"]
    "outcome_statements": {"pass": "A and B-prime gates all met", "fail": "reported as is"}
    t14.conditions.windows: "30 s windows on the 10 s grid, first 60 s of each cell dropped,
                             ramp trim 1 (as the training fit)"

Addendum ``prereg/ADDENDUM-T14-streamcut.json`` (sha256 3a6f71a2...), ``audit_rule``:

    "cut": "a request with outcome model_error and e2e_ms >= 150000 (the route timeout) is a
            route-timeout cut: censored, not a model_error"
    "rule": "every model_error in an accepted T14 cell must be a route-timeout cut; any other
             model_error is a real engine error, reported per cell and judged against the
             original 0.05 limit (a cell above it is void at audit)"
    "label": "the SLO label is unchanged: a cut request ... stays an unserved request
              (violated window) ...; the rule only changes the void audit and the error accounting"
    "report": "per cell: model_error count, cuts, non-cut errors, cut share of sent"
    runtime_limit.max_model_error_rate: 0.1

Addendum ``prereg/ADDENDUM-T14-crossshape.json`` (sha256 2848c705...), reported, not gating:

    cross_shape.what: "fixed Z = 1 (the published theta of the freeze): BA and AUROC per T14
        shape, interpolation and extrapolation in separate tables, cell-bootstrap CI per shape
        (1000 resamples, seed 20260922)"
    cross_shape.claim_rule: "'one theta transfers across shapes' iff the SD of the per-shape
        BA <= the median per-shape BA CI95 half width; stated separately for interpolation
        and extrapolation"

Prereg ``t14.void_rule`` - quoted:

    "a void cell is re-driven once; a second void stops the run; a stopped run is never
     evaluated - re-run all 24 cells into a new root"

(the stream-cut addendum's ``runtime_limit.unchanged`` names the same guard: "void re-drive
once then stop"; the campaign machinery it refers to, ``calibration_ladder.drive_cell`` /
``adaptive_boundary.next_void_attempt``, counts the second void PER CELL).

Decisions (design discussion 2026-10-04; each is a rule below and a line of the H addendum):

D1 cross-shape SD: the claim uses the SAMPLE standard deviation (n - 1) of the per-shape BA;
   the population SD is reported for disclosure only.
D2 single-class shape: a shape whose windows are all one class has no BA and no AUROC
   (``undefined``); the claim of its kind is then ``not_evaluable`` - the shape is NOT
   dropped to claim on the rest; the SD over the remaining shapes is disclosure only. The
   pooled A is computed on all windows as usual.
D3 void at audit (interpretation decided 2026-10-04 evening +08:00, before any M or T14 data
   was opened; owner approval). A cell is void at audit when its non-cut model errors /
   sent > 0.05. Voids of a cell = its void attempts at run time (the T14 manifest's
   ``attempts`` with ``void_reasons``; in a dry run the dataset's ``cells.csv`` rows with
   status ``void``) + 1 if it is void at audit.
   PRIMARY reading (decides status and verdict) - per cell, as ``calibration_ladder.drive_cell``
   counts "a second void": a cell with 2 voids -> ``run_void`` (stopped, never evaluated;
   re-run all 24 into a new root); else a cell void at audit -> ``void_redrive_required:<cells>``
   (re-drive each once, then evaluate); else ``evaluated``. Rationale: the prereg text refers
   to this registered machinery, and T14 is collected under it.
   SENSITIVITY reading (disclosed with equal visibility, never decides) - run level: 2 or more
   voids anywhere in the run (2 voided cells, or one cell voided twice) -> ``run_void``; one
   void that is an audit void -> ``void_redrive_required:<cell>``; else ``evaluated``.
   The output carries ``voided_cells``, ``void_status_primary`` and
   ``void_status_run_level_sensitivity`` side by side. Without a verdict every metric is
   written under ``disclosure_not_an_evaluation``.
D4 a model_error without ``e2e_ms`` is non-cut.
D5 A's training BA is the freeze's pooled ``train_ba_at_published`` (as accept).

Further rules (also in the H addendum):

* audit scope: only the evaluated (valid) attempt of each manifest cell - a voided earlier
  attempt is excluded; every request of that attempt counts, warm-up INCLUDED. The prereg
  does not name warm-up; the stream-cut addendum judges "every model_error in an accepted
  T14 cell" and its runtime limit is ``openloop.check_cell``'s ``model_errors / sent`` over
  the whole cell, so the audit uses the same whole-cell denominator.
* zero-token windows: the dropped-window counts of the T14 validation CSV
  (:mod:`scripts.analysis.h_dropped_windows`) and the conservative variant (dropped windows
  with backlog / failure evidence counted as non-CRITICAL misses) - disclosure, as at H.
* per-shape AUROC follows D2.
* Fig 2.1 KV usage: the scorer does not draw Fig 2.1; it discloses, per cell, the share of
  missing 1 Hz samples of the instant sidecar (``<cell>.instant.jsonl``, the
  ``kv_cache_usage`` source the cross-shape addendum names; it falls behind under overload)
  over the whole cell and after the warm-up, plus the largest gap.
* B' unit: ``recall_severe`` (``b_prime.series_point``) is a rate over WINDOWS (severe
  violating windows that are CRITICAL at dwell 1 / severe violating windows), not over
  episodes; its CI resamples cells.
* scorer identity: the output records this file's path and sha256 and the commit
  (``dline_refit.code_state``); the rule is "dry run on the frozen / training set first": a
  real run refuses unless ``--dry-run-result`` is a dry-run output of the same commit and
  the same scorer sha256, with a clean tree.

How it is implemented (every number through the accept code path):

* inputs checked first, every problem listed, nothing written on a refusal: the prereg and
  both addenda against their sha256 sidecars and the addenda's ``amends.sha256``; the freeze
  (``dline_refit.verify_freeze``; file sha256 and self hash = the prereg's
  ``parameter_sets.freeze``; the 14b published theta / lambda / w_p / tau_crit = the prereg's;
  sealed B' cut = ``dsqwen-14b_cut``; B' gate = the prereg's); the T14 manifest
  (``dline_refit.check_m_manifest``: T14_SHA256SUMS, freeze, label) and its cells = the
  prereg's 24 (cell id, shape, kind);
* rows: ``dline_refit.collect_m_rows`` (manifest cells, the listed attempt, split holdout,
  cell_status valid) -> a validation CSV; windows: the frozen spec / label / ramp trim
  (warm-up rows ``in_warmup`` dropped by the loader);
* A, B' (dwell 1, the freeze's sealed cut and gate), old B / C / D and the ranking
  disclosure: ``dline_refit.evaluate_model`` with ``b_prime_inputs`` - the accept function.
  Verdict (only with status ``evaluated``) = A passed and B' passed (D is not in the T14 rule);
* per shape and per kind: BA at the published theta with ``dline_refit.acceptance_bootstrap``
  (1000, seed 20260922) and AUROC with ``tre_calibration.ranking.ranking_disclosure``;
  the cross-shape claim per kind by D1 / D2;
* censoring audit from the T14 dataset's ``requests.csv``: model_error with
  ``e2e_ms >= 150000`` = cut, any other model_error = non-cut (D4); non-cut / sent > 0.05 =
  void at audit -> D3.

Not implemented (disclosure only in the cross-shape addendum): the Fig 2.1 drawing and the
lambda disclosure.

Runs once: ``--out`` and ``<out>.d/`` must not exist; both are made read-only.

    cd tre/deploy && PYTHONPATH=../common:.:../controller:../service-manager:../calibration:../replayer:../ui \\
      python3 -m scripts.analysis.t14_score --prereg $C/t14/preregistration.json \\
        --addendum $C/prereg/ADDENDUM-T14-streamcut.json --addendum $C/prereg/ADDENDUM-T14-crossshape.json \\
        --freeze-file $C/freeze/params_freeze.json --t14-manifest $C/T14/dsqwen-14b/T14_manifest.json \\
        --dataset $C/T14/dsqwen-14b/dataset --dry-run-result <dry-run output of this commit> \\
        --out $C/eval/T14_score.json

``--dry-run-dataset DIR`` replaces the manifest and the T14 dataset with a TRAINING dataset
(its training-set hold cells, fake interpolation / extrapolation kinds by shape) to prove
the code runs; the output says DRY RUN and carries no verdict.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PREREG_SEED = 20260922
PREREG_RESAMPLES = 1000
#: The stream-cut rule (route-timeout cut, the 0.05 non-cut limit, the 0.10 runtime limit) is
#: scripts.stream_cut's - one definition, shared with dline_refit accept (M2).
from scripts.stream_cut import (NON_CUT_ERROR_LIMIT, OUTCOME_MODEL_ERROR,  # noqa: E402
                                ROUTE_TIMEOUT_CUT_MS, RUNTIME_MODEL_ERROR_LIMIT)
KINDS = ("interpolation", "extrapolation")
#: Dry run only: fake kinds for the training shapes (no T14 shape is read).
DRY_RUN_KINDS = {"S1": "interpolation", "S2": "interpolation", "S3": "interpolation", "S4": "interpolation",
                 "S5": "extrapolation", "T8": "extrapolation", "T9": "extrapolation"}

DECISIONS = {
    "D1_cross_shape_sd": "the claim uses the sample SD (n-1) of the per-shape BA; the population SD is disclosure only",
    "D2_single_class_shape": "a single-class shape has no BA and no AUROC; its kind's claim is not_evaluable (the "
                             "shape is not dropped); the SD over the remaining shapes is disclosure only; pooled A "
                             "unaffected",
    "D3_void_rule_primary": "per cell (calibration_ladder.drive_cell): a cell with 2 voids (run-time void attempts + "
                            "audit void) -> run_void; else an audit-void cell -> void_redrive_required:<cells>; decides",
    "D3_void_rule_sensitivity": "run level: 2 or more voids in the run (2 voided cells or one cell twice) -> run_void; "
                                "one audit void -> void_redrive_required:<cell>; disclosed side by side, never decides",
    "D3_decided": "2026-10-04 evening +08:00, owner approval, before any M or T14 data was opened",
    "D4_model_error_without_e2e": "non-cut",
    "D5_training_ba": "the freeze's pooled train_ba_at_published",
    "audit_scope": "the evaluated (valid) attempt only, voided earlier attempts excluded; warm-up included "
                   "(whole-cell denominator of openloop.check_cell, which the stream-cut addendum names)",
    "b_prime_unit": "recall_severe at dwell 1 is per window (b_prime.series_point), not per episode",
    "kv_source": "1 Hz instant sidecar <cell>.instant.jsonl (kv_cache_usage); missing-sample share per cell disclosed",
    "scorer_identity": "path, sha256 and commit recorded; dry run on the frozen/training set first (enforced)",
}
STATUS_EVALUATED = "evaluated"
STATUS_REDRIVE = "void_redrive_required"
STATUS_RUN_VOID = "run_void"
CLAIM_NOT_EVALUABLE = "not_evaluable"

#: The ``schema`` of a next-round (v2) preregistration; a preregistration without it is v1.
PREREG_SCHEMA_V2 = "t14-prereg-v2"
#: The per-shape BA CI of the cross-shape claim (user 2026-10-05, CI method only; the claim
#: rule is unchanged). v1 (2026-10-03) used the accept's cell bootstrap, which with 3 cells
#: per shape degenerates (half width 0). v2: a moving-block bootstrap over the windows of the
#: shape, blocks of CROSS_SHAPE_BLOCK_WINDOWS consecutive windows kept inside one cell.
CROSS_SHAPE_CI_METHOD = {
    "method": "moving_block_bootstrap",
    "block_windows": 6,
    "blocks_within": "cell",
    "resamples": 1000,
    "seed": 20260922,
    "interval": "95 % percentile (dline_refit._ci95 index rule)",
    "resample": ("draw blocks uniformly (with replacement) from every block of block_windows consecutive "
                 "windows of each cell of the shape (a cell shorter than a block is one block), concatenate "
                 "until the shape's window count is reached, truncate; BA at the published theta; a "
                 "single-class resample is skipped"),
    "block_length_why": ("the method docs' 'independent samples ~ windows / 6' (theta-recalibration.md section 3; "
                         "preregistration-20261003 interval note); that wording came from the earlier 5 s step: "
                         "with 30 s windows on a 10 s step the overlap spans 3 windows "
                         "(dline_refit.WINDOWS_PER_INDEPENDENT = 3), so a 6-window block is conservative"),
}
#: Rule v2's noise yardstick of the cross-shape claim (coordinator 2026-10-05, decided before
#: sealing on a TRAINING-only check): the median per-shape CI95 half width is taken only over
#: shapes whose CI is non-degenerate; the claim rule itself is unchanged.
CROSS_SHAPE_YARDSTICK = {
    "median_over": "shapes of the kind with both classes AND a per-shape CI95 half width > 0",
    "min_qualifying_shapes": 2,
    "fewer_than_min": "not_evaluable (never false)",
    "sd_over": "every shape of the kind with both classes (D1 sample SD); D2 single-class handling unchanged",
    "why": ("a zero-width CI at BA = 1 or .5 reflects perfect separation or a one-sided classification in 3 "
            "cells, not zero sampling uncertainty, so it must not set the noise yardstick"),
}
#: Label attributions (``tre_common.slo_labels.ATTRIBUTIONS`` of label v2; absent = completion).
ATTRIBUTION_COMPLETION, ATTRIBUTION_HYBRID = "completion", "hybrid"
ATTRIBUTIONS = (ATTRIBUTION_COMPLETION, ATTRIBUTION_HYBRID)
#: v2: the D1-D5 keys the preregistration's ``evaluation.scoring.decisions`` must carry.
V2_DECISION_KEYS = ("D1_cross_shape_sd", "D2_single_class_shape", "D3_void_at_audit",
                    "D4_model_error_without_e2e", "D5_training_ba")
ONSET_NOT_APPLICABLE = {
    "status": "not_applicable",
    "why": ("T14 is 24 steady holds (0.9 / 1.0 / 1.1 x C^_s, 240 s) and has no dynamic cell: there is no "
            "overload onset to catch. The onset-episode gate of the new acceptance is judged on M2's "
            "dynamic cells only."),
}
V2_VERDICT_RULE = ("only with status 'evaluated', the accept vocabulary (dline_refit.VERDICT_*): pass iff A "
                   "(BA >= .80, CI95 low >= .75, drop from training <= .08) and FA (CRITICAL false alarm on "
                   "healthy windows at dwell 1 <= .05, CI95 high <= .08) are all met; pass_a_disclosed iff only "
                   "A fails and FA passes (the pre-declared consequence: go-live allowed, A disclosed as a "
                   "limitation); fail = anything else, reported as is; onset episodes not applicable (steady "
                   "holds); window B' and D disclosed")


class Refused(RuntimeError):
    def __init__(self, problems: Sequence[str]):
        super().__init__("; ".join(problems))
        self.problems = list(problems)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sidecar_ok(path: Path, problems: list[str]) -> Optional[str]:
    """sha256 of ``path`` if it matches ``<path>.sha256`` (sha256sum format), else a problem."""
    side = Path(f"{path}.sha256")
    if not path.is_file():
        problems.append(f"{path}: missing")
        return None
    sha = _sha256(path)
    if not side.is_file():
        problems.append(f"{side}: missing sidecar")
    elif side.read_text(encoding="utf-8").split()[0] != sha:
        problems.append(f"{path}: sha256 {sha} != its sidecar {side}")
    return sha


def _close(a: Any, b: Any, tol: float = 1e-9) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    except (TypeError, ValueError):
        return False


def label_attribution(label_def: Optional[Mapping[str, Any]]) -> str:
    """The request-to-window attribution of a label record (absent = completion, label v1)."""
    return str((label_def or {}).get("attribution") or ATTRIBUTION_COMPLETION)


def freeze_attribution(doc: Mapping[str, Any], model: str) -> str:
    """The attribution of the label ``model`` was frozen under (the label the scorer labels
    every window with: ``verdict_for_holdout.label_def``)."""
    entry = (doc.get("models") or {}).get(model) or {}
    return label_attribution((entry.get("verdict_for_holdout") or {}).get("label_def") or entry.get("label_def"))


def dataset_attribution(directory: Path) -> str:
    """The attribution a standard dataset was labelled under: its manifest's
    ``attribution.value`` (``calibration_dataset.dataset_attribution`` when that helper
    exists - label v2 - else the manifest read by the same contract); absent = completion."""
    try:
        from scripts import calibration_dataset as cd
        helper = getattr(cd, "dataset_attribution", None)
    except ImportError:            # pragma: no cover - the module is always there in the repo
        helper = None
    if helper is not None:
        try:
            return str(helper(Path(directory)))
        except ValueError as exc:          # an unknown attribution in the manifest
            raise Refused([f"dataset {directory}: {exc}"])
    d = Path(directory)
    man = d / "manifest.json" if (d / "manifest.json").exists() else d / "dataset" / "manifest.json"
    try:
        doc = json.loads(man.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ATTRIBUTION_COMPLETION
    att = doc.get("attribution")
    if isinstance(att, Mapping):
        att = att.get("value")
    return str(att or ATTRIBUTION_COMPLETION)


def check_attributions(sources: Mapping[str, Any]) -> tuple[Optional[str], list[str]]:
    """``(the one attribution, problems)``: every source (``{what: attribution}``) must name
    the same known attribution - T14 is scored under one attribution and never mixes them."""
    problems = [f"{what}: unknown label attribution {a!r} (known: {list(ATTRIBUTIONS)})"
                for what, a in sources.items() if a not in ATTRIBUTIONS]
    values = sorted({str(a) for a in sources.values()})
    if len(values) > 1:
        problems.append("mixed label attributions - T14 is scored under ONE attribution, the one the dataset "
                        "carries, and never mixes them: " + ", ".join(f"{w} = {a}" for w, a in sources.items()))
    return (values[0] if len(values) == 1 and not problems else None), problems


def _v2_rule_checks(prereg: Mapping[str, Any]) -> list[str]:
    """The v2 preregistration's inline rule blocks the scorer implements (beyond the shared
    A / stream-cut / cross-shape checks)."""
    from scripts import dline_refit as dl

    problems: list[str] = []
    t14 = prereg.get("t14") or {}
    ev = prereg.get("evaluation") or {}
    fa = ev.get("FA") or {}
    from scripts import b_prime as _bp
    want_fa = dict(_bp.WINDOW_FA_GATE)  # the accept window_fa gate (one definition)
    gate = fa.get("gate") or {}
    if any(not _close(gate.get(k), v) for k, v in want_fa.items()):
        problems.append(f"prereg evaluation.FA.gate {gate} != {want_fa}")
    if fa.get("dwell_windows") != dl.ONLINE_DWELL_WINDOWS:
        problems.append(f"prereg evaluation.FA.dwell_windows {fa.get('dwell_windows')!r} != "
                        f"dline_refit.ONLINE_DWELL_WINDOWS {dl.ONLINE_DWELL_WINDOWS}")
    if (ev.get("onset_episodes") or {}).get("status") != ONSET_NOT_APPLICABLE["status"]:
        problems.append("prereg evaluation.onset_episodes.status is not 'not_applicable' (T14 has no dynamic cell)")
    missing = [k for k in V2_DECISION_KEYS if k not in ((ev.get("scoring") or {}).get("decisions") or {})]
    if missing:
        problems.append(f"prereg evaluation.scoring.decisions misses {missing}")
    ci_m = (ev.get("cross_shape") or {}).get("ci_method")
    if json.dumps(ci_m, sort_keys=True) != json.dumps(CROSS_SHAPE_CI_METHOD, sort_keys=True):
        problems.append(f"prereg evaluation.cross_shape.ci_method {ci_m!r} is not the implemented per-shape CI "
                        f"{CROSS_SHAPE_CI_METHOD!r}")
    ys = (ev.get("cross_shape") or {}).get("yardstick")
    if json.dumps(ys, sort_keys=True) != json.dumps(CROSS_SHAPE_YARDSTICK, sort_keys=True):
        problems.append(f"prereg evaluation.cross_shape.yardstick {ys!r} is not the implemented one "
                        f"{CROSS_SHAPE_YARDSTICK!r}")
    if t14.get("void_rule") != VOID_RULE_TEXT:
        problems.append(f"prereg t14.void_rule {t14.get('void_rule')!r} is not the implemented rule")
    if "pass_a_disclosed" not in (ev.get("outcome_statements") or {}):
        problems.append("prereg evaluation.outcome_statements has no pre-declared 'pass_a_disclosed' consequence")
    if (t14.get("conditions") or {}).get("label_attribution") not in ATTRIBUTIONS:
        problems.append(f"prereg t14.conditions.label_attribution "
                        f"{(t14.get('conditions') or {}).get('label_attribution')!r} not in {list(ATTRIBUTIONS)}")
    return problems


def check_inputs(prereg_path: Path, addenda: Sequence[Path], freeze_file: Path) -> dict:
    """Every check on the prereg, the addenda (v1) and the freeze; raises :class:`Refused`."""
    from scripts import b_prime
    from scripts import dline_refit as dl

    problems: list[str] = []
    prereg_sha = _sidecar_ok(prereg_path, problems)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8")) if prereg_path.is_file() else {}
    v2 = prereg.get("schema") == PREREG_SCHEMA_V2
    ev = prereg.get("evaluation") or {}
    a_ci = str((ev.get("A") or {}).get("ci") or "")
    if str(PREREG_RESAMPLES) not in a_ci or str(PREREG_SEED) not in a_ci:
        problems.append(f"prereg evaluation.A.ci {a_ci!r} does not name {PREREG_RESAMPLES} resamples / seed {PREREG_SEED}")
    if (dl.ACCEPT_RESAMPLES, dl.SEED) != (PREREG_RESAMPLES, PREREG_SEED):
        problems.append(f"dline_refit bootstrap ({dl.ACCEPT_RESAMPLES}, {dl.SEED}) != the prereg's")
    gates = (ev.get("A") or {}).get("gates_reported") or {}
    want_a = {"ba_min": dl.A_BA_MIN, "ba_ci95_low_min": dl.A_BA_CI_LOW_MIN,
              "max_drop_from_training_ba": dl.A_MAX_DROP_FROM_TRAINING}
    if any(not _close(gates.get(k), v) for k, v in want_a.items()):
        problems.append(f"prereg A gates {gates} != the accept constants {want_a}")
    adds: dict[str, dict] = {}
    if v2:
        if addenda:
            problems.append(f"a {PREREG_SCHEMA_V2} preregistration carries its rules inline (t14.stream_cut, "
                            f"evaluation.cross_shape, evaluation.scoring); no --addendum is read: {list(map(str, addenda))}")
        problems += _v2_rule_checks(prereg)
        sc = (prereg.get("t14") or {}).get("stream_cut") or {}
        cs = ev.get("cross_shape") or {}
        where_sc, where_cs = "prereg t14.stream_cut", "prereg evaluation.cross_shape"
    else:
        for p in addenda:
            sha = _sidecar_ok(p, problems)
            doc = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
            if (doc.get("amends") or {}).get("sha256") != prereg_sha:
                problems.append(f"{p}: amends {(doc.get('amends') or {}).get('sha256')} != the prereg sha256 {prereg_sha}")
            kind = "streamcut" if "audit_rule" in doc else ("crossshape" if "chapter2_outputs" in doc else None)
            if kind is None:
                problems.append(f"{p}: neither the stream-cut nor the cross-shape addendum")
                continue
            adds[kind] = {"path": str(p), "sha256": sha, "doc": doc}
        for kind in ("streamcut", "crossshape"):
            if kind not in adds:
                problems.append(f"the {kind} addendum is not given (--addendum)")
        sc = (adds.get("streamcut") or {}).get("doc") or {}
        cs = (((adds.get("crossshape") or {}).get("doc") or {}).get("chapter2_outputs") or {}).get("cross_shape") or {}
        where_sc, where_cs = "stream-cut addendum", "cross-shape"
    rule = sc.get("audit_rule") or {}
    if "e2e_ms >= 150000" not in str(rule.get("cut") or ""):
        problems.append(f"{where_sc} cut {rule.get('cut')!r} is not 'e2e_ms >= 150000'")
    if "0.05" not in str(rule.get("rule") or ""):
        problems.append(f"{where_sc} rule does not name the 0.05 limit")
    if not _close((sc.get("runtime_limit") or {}).get("max_model_error_rate"), RUNTIME_MODEL_ERROR_LIMIT):
        problems.append(f"{where_sc} runtime_limit.max_model_error_rate != 0.10")
    if "SD of the per-shape BA <= the median per-shape BA CI95 half width" not in str(cs.get("claim_rule") or ""):
        problems.append(f"{where_cs} claim rule {cs.get('claim_rule')!r} is not the one implemented")

    model = str((prereg.get("t14") or {}).get("model") or "")
    ps = (prereg.get("parameter_sets") or {}).get("freeze") or {}
    doc: dict = {}
    try:
        doc = dl.verify_freeze(freeze_file)
    except dl.FreezeError as exc:
        problems += exc.problems
    freeze_sha = _sha256(freeze_file) if freeze_file.is_file() else None
    bp_key = "B_prime_disclosure" if v2 else "B_prime"
    if doc:
        if freeze_sha != ps.get("sha256"):
            problems.append(f"freeze file sha256 {freeze_sha} != prereg parameter_sets.freeze.sha256 {ps.get('sha256')}")
        if doc.get("freeze_sha256") != ps.get("freeze_sha256"):
            problems.append("freeze self hash != prereg parameter_sets.freeze.freeze_sha256")
        entry = (doc.get("models") or {}).get(model)
        if entry is None:
            problems.append(f"{model!r} (prereg t14.model) is not in the freeze")
        else:
            want = ps.get(model) or {}
            pub = entry.get("published") or {}
            for k in ("theta", "lambda_wait", "w_p", "tau_crit", "tau_s"):
                if not _close(pub.get(k), want.get(k)):
                    problems.append(f"freeze {model} published {k} {pub.get(k)} != prereg {want.get(k)}")
            cut = (entry.get("b_prime") or {}).get("severity_cut")
            pre_cut = ((ev.get(bp_key) or {}).get(f"{model}_cut"))
            if not _close(round(float(cut or 0), 5), pre_cut, 1e-12):
                problems.append(f"freeze {model} B' cut {cut} != prereg {pre_cut}")
            if str(entry.get("numerator") or "gateway") != "gateway":
                problems.append(f"freeze {model} numerator {entry.get('numerator')} != gateway (prereg data)")
            if v2:
                label = (entry.get("verdict_for_holdout") or {}).get("label_def") or {}
                want_label = (prereg.get("t14") or {}).get("conditions", {}).get("label_def_sha256")
                if dl.canonical_sha256(label) != want_label:
                    problems.append(f"freeze {model} label sha256 {dl.canonical_sha256(label)} != prereg "
                                    f"t14.conditions.label_def_sha256 {want_label}")
        try:
            gate = b_prime.check_gate(doc.get("b_prime_gate") or {})
            if not v2:
                pre_gate = (ev.get("B_prime") or {}).get("gate") or {}
                if any(not _close(gate[k], pre_gate.get(k)) for k in b_prime.GATE_KEYS):
                    problems.append(f"freeze B' gate {gate} != prereg {pre_gate}")
                if int(pre_gate.get("online_dwell_windows", -1)) != dl.ONLINE_DWELL_WINDOWS:
                    problems.append("prereg B' dwell != dline_refit.ONLINE_DWELL_WINDOWS")
        except ValueError as exc:
            problems.append(f"freeze B' gate: {exc}")
    if problems:
        raise Refused(problems)
    return {"prereg": prereg, "prereg_sha256": prereg_sha, "addenda": adds, "doc": doc,
            "freeze_sha256": freeze_sha, "model": model, "schema": "v2" if v2 else "v1",
            "stream_cut": sc, "cross_shape": cs, "freeze_attribution": freeze_attribution(doc, model)}


def prereg_cells(prereg: Mapping[str, Any]) -> dict[str, dict]:
    return {str(c["cell_id"]): {"shape": c["shape"], "kind": c["kind"], "factor": c.get("factor")}
            for c in (prereg.get("t14") or {}).get("cells") or []}


def manifest_rows(inp: Mapping[str, Any], manifest_path: Path, dataset_dir: Path) -> tuple[dict, dict, Path]:
    """(manifest, {(cell_id, attempt): kind/shape}, the dataset dir) after the manifest checks."""
    from scripts import dline_refit as dl

    problems: list[str] = []
    man, pr = dl.check_m_manifest(manifest_path, inp["doc"], inp["freeze_sha256"])
    problems += pr
    if man is None:
        raise Refused(problems)
    if man.get("model") != inp["model"]:
        problems.append(f"manifest model {man.get('model')} != prereg {inp['model']}")
    want = prereg_cells(inp["prereg"])
    have = {str(c["cell_id"]): {"shape": c.get("shape"), "kind": c.get("kind"), "factor": c.get("factor")}
            for c in man["cells"]}
    if sorted(have) != sorted(want):
        problems.append(f"manifest cells {sorted(set(have) ^ set(want))} differ from the prereg's 24")
    for cid in sorted(set(have) & set(want)):
        if (have[cid]["shape"], have[cid]["kind"]) != (want[cid]["shape"], want[cid]["kind"]):
            problems.append(f"{cid}: manifest shape/kind {have[cid]} != prereg {want[cid]}")
    roots = [Path(r) for r in ((inp["prereg"].get("t14") or {}).get("forbidden_roots") or [])]
    if any(Path(dataset_dir).resolve().is_relative_to(r) for r in roots):
        problems.append(f"{dataset_dir} lies under a forbidden root of the prereg")
    if problems:
        raise Refused(problems)
    void_attempts: dict[str, int] = defaultdict(int)
    for a in man.get("attempts") or []:
        if a.get("void_reasons"):
            void_attempts[str(a["cell_id"])] += 1
    cells = {}
    for c in man["cells"]:
        raw = [str(f) for f in c.get("raw_files") or []]
        cells[(str(c["cell_id"]), int(c["attempt"]))] = {
            "shape": c["shape"], "kind": c["kind"], "warmup_s": c.get("warmup_s"),
            "runtime_voids": void_attempts.get(str(c["cell_id"]), 0),
            "instant": next((f for f in raw if f.endswith(".instant.jsonl")), None),
            "guard": next((f for f in raw if f.endswith(".guard.json")), None)}
    return man, cells, Path(dataset_dir)


def censoring_audit(requests_csv: Path, cells: Mapping[tuple, Mapping[str, Any]]) -> dict:
    """The stream-cut audit per cell (:func:`scripts.stream_cut.audit`, the one definition):
    every request of the cell's EVALUATED attempt (the manifest's; voided earlier attempts are
    other keys and are not read), warm-up included (the whole-cell ``model_errors / sent`` of
    ``openloop.check_cell``)."""
    from scripts import stream_cut

    return stream_cut.audit(requests_csv, cells)


def void_status(audit: Mapping[str, Any]) -> dict:
    """D3: the prereg's void_rule under the primary (per cell) and the sensitivity (run
    level) reading, side by side. Run-time voids of a cell = ``runtime_voids`` of the audit
    cell (from the manifest's attempt records), else its evaluated attempt - 1 (the T14
    machinery re-drives only a void attempt)."""
    voids = {c["cell_id"]: int(c.get("runtime_voids", int(c["attempt"]) - 1)) + int(bool(c["void_at_audit"]))
             for c in audit["cells"]}
    audit_void = sorted(c["cell_id"] for c in audit["cells"] if c["void_at_audit"])
    voided = sorted(cid for cid, n in voids.items() if n >= 1)
    twice = sorted(cid for cid, n in voids.items() if n >= 2)
    if twice:
        primary = (STATUS_RUN_VOID, f"cell(s) {twice} voided twice (per-cell second void): the run is stopped, "
                                    "never evaluated - re-run all 24 cells into a new root")
    elif audit_void:
        primary = (f"{STATUS_REDRIVE}:{','.join(audit_void)}",
                   f"cell(s) {audit_void} void at audit, first void each: re-drive each once, then evaluate")
    else:
        primary = (STATUS_EVALUATED, "no cell void at audit")
    total = sum(voids.values())
    if total >= 2:
        sens = (STATUS_RUN_VOID, f"{total} voids in the run (cells {voided}): a second void stops the run")
    elif audit_void:
        sens = (f"{STATUS_REDRIVE}:{','.join(audit_void)}", "one void in the run, at audit: re-drive it once")
    else:
        sens = (STATUS_EVALUATED, "at most one void in the run, none at audit")
    return {"void_rule": VOID_RULE_TEXT, "voided_cells": len(voided), "voided_cell_ids": voided,
            "voids_in_run": total, "audit_void_cells": audit_void,
            "void_status_primary": {"status": primary[0], "why": primary[1], "reading": "per cell (decides)"},
            "void_status_run_level_sensitivity": {"status": sens[0], "why": sens[1],
                                                  "reading": "run level (disclosure, never decides)"},
            "decided": "2026-10-04 evening +08:00, before any M or T14 data was opened"}


VOID_RULE_TEXT = ("a void cell is re-driven once; a second void stops the run; a stopped run is never evaluated - "
                  "re-run all 24 cells into a new root")


def kv_missing(cells: Mapping[tuple, Mapping[str, Any]]) -> dict:
    """Per cell, the share of missing 1 Hz instant-sidecar samples (``kv_cache_usage``),
    over the cell [guard start_ms, end_ms] and after the warm-up, and the largest gap. A
    sample counts when it has a finite ``kv_cache_usage``, ``scrape_errors`` 0 and at least
    one pod scraped. ``cells`` values carry ``instant`` / ``guard`` paths and ``warmup_s``."""
    out = []
    for (cid, att), meta in sorted(cells.items()):
        rec: dict[str, Any] = {"cell_id": cid, "attempt": att, "instant": meta.get("instant")}
        try:
            g = json.loads(Path(meta["guard"]).read_text(encoding="utf-8"))
            start, end = float(g["start_ms"]), float(g["end_ms"])
            step = float(g.get("instant_sample_ms") or 1000.0)
            ts = []
            with open(meta["instant"], encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    x = json.loads(line)
                    kv = x.get("kv_cache_usage")
                    ok = (isinstance(kv, (int, float)) and math.isfinite(kv) and not (x.get("scrape_errors") or 0)
                          and (x.get("pods_scraped") or 0) >= 1)
                    if ok and start <= float(x["ts_ms"]) <= end:
                        ts.append(float(x["ts_ms"]))
        except (KeyError, OSError, TypeError, ValueError) as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            out.append(rec)
            continue
        ts.sort()
        warm = start + 1000.0 * float(meta.get("warmup_s") or 0.0)

        def share(lo: float) -> Optional[float]:
            expected = int((end - lo) // step) + 1
            got = sum(1 for t in ts if t >= lo)
            return max(0.0, 1.0 - got / expected) if expected > 0 else None

        edges = [start, *ts, end]
        rec.update({"sample_ms": step, "samples": len(ts), "missing_share_cell": share(start),
                    "missing_share_after_warmup": share(warm),
                    "max_gap_ms": max(b - a for a, b in zip(edges, edges[1:]))})
        out.append(rec)
    shares = [c["missing_share_cell"] for c in out if c.get("missing_share_cell") is not None]
    return {"source": "1 Hz instant sidecar <cell>.instant.jsonl, kv_cache_usage (the Fig 2.1 KV source); "
                      "disclosure only", "cells": out,
            "max_missing_share_cell": max(shares) if shares else None,
            "cells_unreadable": [c["cell_id"] for c in out if "error" in c]}


def moving_block_ba_ci(windows: Sequence[Any], *, theta: float, direction: str,
                       method: Mapping[str, Any] = CROSS_SHAPE_CI_METHOD) -> dict:
    """The BA CI95 of :data:`CROSS_SHAPE_CI_METHOD`: a moving-block bootstrap over the
    windows, blocks of ``block_windows`` consecutive windows (by window start) kept within a
    cell."""
    import random

    from tre_calibration.fit import threshold_balanced_accuracy

    from scripts import dline_refit as dl

    length = int(method["block_windows"])
    by: dict[str, list] = defaultdict(list)
    for w in windows:
        by[w.scenario_id].append(w)
    blocks = []
    for cell in sorted(by):
        ws = sorted(by[cell], key=lambda w: w.window_start_ms)
        if len(ws) <= length:
            blocks.append(ws)
        else:
            blocks += [ws[i:i + length] for i in range(len(ws) - length + 1)]
    n = len(windows)
    rng = random.Random(int(method["seed"]))
    vals = []
    for _ in range(int(method["resamples"])):
        smp: list = []
        while len(smp) < n:
            smp += rng.choice(blocks)
        smp = smp[:n]
        if any(w.slo_met for w in smp) and any(not w.slo_met for w in smp):
            vals.append(threshold_balanced_accuracy(smp, theta=theta, direction=direction)["balanced_accuracy"])
    ci = dl._ci95(vals)
    return {"ci95": ci, "half_width": ((ci[1] - ci[0]) / 2.0) if None not in ci else None,
            "resamples_used": len(vals), "blocks": len(blocks), "method": dict(method)}


def _ba_ci_auroc(entry: Mapping[str, Any], windows: Sequence[Any], *, n_resamples: int, seed: int,
                 ci_method: Optional[Mapping[str, Any]] = None) -> dict:
    """BA at the published theta (+ CI95) and AUROC (+ CI95, ``ranking.ranking_disclosure``)
    of one window subset; both ``undefined`` (None) when the subset holds one class only
    (D2). The BA CI: ``ci_method`` (rule v2, :data:`CROSS_SHAPE_CI_METHOD`, moving blocks) or,
    None, the accept's cell bootstrap (rule v1)."""
    from tre_calibration import ranking
    from tre_calibration.fit import threshold_balanced_accuracy

    from scripts import dline_refit as dl
    from scripts import theta_verdict as tv

    vh = entry["verdict_for_holdout"]
    theta, tau_crit = float(vh["published"]["theta_m"]), float(vh["published"]["tau_crit"])
    direction = vh["fit_config"]["direction"]
    viol = sum(1 for w in windows if not w.slo_met)
    cells = len({w.scenario_id for w in windows})
    single = not 0 < viol < len(windows)
    base = {"windows": len(windows), "violating": viol, "cells": cells, "single_class": single}
    if single:
        return {**base, "ba": None, "ba_ci95": [None, None], "ba_ci95_half_width": None, "auroc": None,
                "auroc_ci95": [None, None], "undefined": "one class only: BA and AUROC undefined (D2)"}
    ba = threshold_balanced_accuracy(windows, theta=theta, direction=direction)["balanced_accuracy"]
    crit = tv.critical_dwell_flags(windows, theta=theta, tau_crit=tau_crit, direction=direction,
                                   dwell_windows=dl.ONLINE_DWELL_WINDOWS)
    if ci_method is not None:
        mb = moving_block_ba_ci(windows, theta=theta, direction=direction, method=ci_method)
        ci, used, how = mb["ci95"], mb["resamples_used"], f"{ci_method['method']} ({mb['blocks']} blocks)"
    else:
        boot = dl.acceptance_bootstrap(windows, crit, theta=theta, direction=direction, n_resamples=n_resamples,
                                       seed=seed)
        ci, used, how = (boot["metrics"]["balanced_accuracy"]["ci95"],
                         boot["metrics"]["balanced_accuracy"]["resamples_used"], "cell bootstrap (rule v1)")
    recs = ranking.records_from_windows(vh["model"], windows, theta=theta, direction=direction)
    au = ranking.ranking_disclosure(recs, n_resamples=n_resamples, seed=seed).get("auroc") or {}
    return {**base, "ba": ba, "ba_ci95": ci,
            "ba_ci95_half_width": ((ci[1] - ci[0]) / 2.0) if None not in ci else None,
            "ba_ci_method": how, "ba_resamples_used": used,
            "auroc": au.get("value"), "auroc_ci95": au.get("ci95"), "auroc_resamples_used": au.get("resamples_used")}


def claim(table: Mapping[str, Mapping[str, Any]], yardstick: Optional[Mapping[str, Any]] = None) -> dict:
    """The cross-shape claim of one kind (D1, D2). ``yardstick`` (rule v2,
    :data:`CROSS_SHAPE_YARDSTICK`): the median half width only over the non-degenerate shapes
    (both classes, half width > 0), not evaluable below ``min_qualifying_shapes``; None =
    rule v1 (every shape with a CI)."""
    single = sorted(s for s, t in table.items() if t["single_class"])
    bas = [t["ba"] for s, t in sorted(table.items()) if not t["single_class"]]
    halves = [t["ba_ci95_half_width"] for s, t in sorted(table.items())
              if not t["single_class"] and t["ba_ci95_half_width"] is not None]
    if yardstick is not None:
        degenerate = sorted(s for s, t in table.items() if not t["single_class"]
                            and not (t["ba_ci95_half_width"] or 0) > 0)
        halves = [h for h in halves if h > 0]
        sd_s = statistics.stdev(bas) if len(bas) >= 2 else None
        sd_p = statistics.pstdev(bas) if bas else None
        med = statistics.median(halves) if halves else None
        out = {"rule": "'one theta transfers across shapes' iff the SD (sample, n-1) of the per-shape BA <= the "
                       "median per-shape BA CI95 half width over the non-degenerate shapes",
               "yardstick": dict(yardstick), "shapes": len(table), "single_class_shapes": single,
               "degenerate_ci_shapes": degenerate, "yardstick_shapes": len(halves),
               "role": "claim rule for the text; not an acceptance gate"}
        need = int(yardstick["min_qualifying_shapes"])
        if not table or single or sd_s is None or len(halves) < need:
            out.update({"one_theta_transfers": CLAIM_NOT_EVALUABLE,
                        "why": ("single-class shape(s) " + str(single) + ": BA undefined, the shape is not dropped (D2)"
                                if single else f"fewer than {need} shapes with a non-degenerate CI (or with a BA)"),
                        "disclosure_only": {"sd_sample": sd_s, "sd_population": sd_p, "median_ci95_half_width": med,
                                            "shapes_with_ba": len(bas)}})
        else:
            out.update({"one_theta_transfers": sd_s <= med, "sd_sample": sd_s, "median_ci95_half_width": med,
                        "disclosure_population_sd": {"sd_population": sd_p, "would_claim": sd_p <= med}})
        return out
    sd_s = statistics.stdev(bas) if len(bas) >= 2 else None
    sd_p = statistics.pstdev(bas) if bas else None
    med = statistics.median(halves) if halves else None
    out = {"rule": "'one theta transfers across shapes' iff the SD (sample, n-1) of the per-shape BA <= the "
                   "median per-shape BA CI95 half width",
           "shapes": len(table), "single_class_shapes": single,
           "role": "claim rule for the text; not an acceptance gate"}
    if not table or single or sd_s is None or med is None or len(halves) != len(table):
        out.update({"one_theta_transfers": CLAIM_NOT_EVALUABLE,
                    "why": ("single-class shape(s) " + str(single) + ": BA undefined, the shape is not dropped (D2)"
                            if single else "fewer than 2 shapes with a BA and a CI"),
                    "disclosure_only_over_remaining_shapes": {"sd_sample": sd_s, "sd_population": sd_p,
                                                              "median_ci95_half_width": med,
                                                              "shapes": len(bas)}})
    else:
        out.update({"one_theta_transfers": sd_s <= med, "sd_sample": sd_s, "median_ci95_half_width": med,
                    "disclosure_population_sd": {"sd_population": sd_p, "would_claim": sd_p <= med}})
    return out


def cross_shape(entry: Mapping[str, Any], windows: Sequence[Any], shape_of: Mapping[str, str],
                kind_of: Mapping[str, str], *, n_resamples: int, seed: int,
                ci_method: Optional[Mapping[str, Any]] = None) -> dict:
    """Per-shape tables split by kind, per-kind pooled numbers (disclosure), the claim per kind;
    ``ci_method``: the per-shape BA CI (rule v2's moving blocks; None = rule v1's cell bootstrap)."""
    by_shape: dict[str, list] = defaultdict(list)
    for w in windows:
        by_shape[shape_of[w.scenario_id]].append(w)
    out: dict[str, Any] = {}
    for kind in KINDS:
        shapes = sorted(s for s in by_shape if kind_of[s] == kind)
        table = {s: _ba_ci_auroc(entry, by_shape[s], n_resamples=n_resamples, seed=seed, ci_method=ci_method)
                 for s in shapes}
        pooled = [w for s in shapes for w in by_shape[s]]
        out[kind] = {"per_shape": table,
                     "pooled_disclosure": (_ba_ci_auroc(entry, pooled, n_resamples=n_resamples, seed=seed,
                                                        ci_method=ci_method) if pooled else None),
                     "claim": claim(table, CROSS_SHAPE_YARDSTICK if ci_method is not None else None)}
    return out


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    with open(path, "x", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(header))
        for r in rows:
            w.writerow([r.get(c, "") for c in header])


def dry_run_rows(dataset_dir: Path, model: str) -> tuple[list[str], list[dict], dict]:
    """A TRAINING dataset's training-set rows of ``model`` (dry run only), and its cells with
    their raw sidecar paths (``cells.csv`` raw_path / guard_path under the manifest's run_root)."""
    from scripts import dline_refit as dl

    with open(dataset_dir / "windows.csv", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows = [r for r in reader if r.get("model") == model and (r.get("cell_status") or "valid") == "valid"
                and (r.get("split") or "") != "holdout"
                and dl.assign_set(r, sealed_to_h2=True, sentinels=True) == dl.SET_TRAINING
                and r.get("shape") in DRY_RUN_KINDS]
    root = Path(json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8")).get("run_root") or dataset_dir.parent)
    meta: dict[tuple, dict] = {}
    void_attempts: dict[str, int] = defaultdict(int)
    with open(dataset_dir / "cells.csv", newline="", encoding="utf-8") as fh:
        crow = list(csv.DictReader(fh))
    for c in crow:
        if (c.get("status") or "").strip() == "void" or (c.get("void_reasons") or "").strip():
            void_attempts[c["cell_id"]] += 1
    for c in crow:
        raw = c.get("raw_path") or ""
        meta[(c["cell_id"], int(float(c["attempt"] or 1)))] = {
            "warmup_s": c.get("warmup_s"),
            "instant": str(root / (raw[:-len(".jsonl")] + ".instant.jsonl")) if raw.endswith(".jsonl") else None,
            "guard": str(root / c["guard_path"]) if c.get("guard_path") else None}
    cells = {}
    for r in rows:
        key = (r["cell_id"], int(float(r["attempt"] or 1)))
        cells[key] = {"shape": r["shape"], "kind": DRY_RUN_KINDS[r["shape"]], **meta.get(key, {}),
                      "runtime_voids": void_attempts.get(r["cell_id"], 0)}
    return header, rows, cells


def v2_verdict(a_passed: bool, fa_passed: bool) -> str:
    """The v2 T14 verdict in the accept's three-way vocabulary (``dline_refit.verdict_of``):
    A and FA pass -> pass; only A fails -> pass_a_disclosed; otherwise fail."""
    from scripts import dline_refit as dl

    if not fa_passed:
        return dl.VERDICT_FAIL
    return dl.VERDICT_PASS if a_passed else dl.VERDICT_PASS_A_DISCLOSED


def fa_gate(entry: Mapping[str, Any], windows: Sequence[Any], *, gate: Mapping[str, float], n_resamples: int,
            seed: int, b_prime_block: Optional[Mapping[str, Any]] = None) -> dict:
    """v2 gate FA: the CRITICAL false alarm on healthy windows at the controller's dwell
    (``dline_refit.ONLINE_DWELL_WINDOWS``) and its cell-bootstrap CI95 - the same flag
    series, point and bootstrap as B' (``b_prime.series_point`` / ``b_prime_boot``), which
    do not depend on the severity cut; so FA needs no cut and equals B''s ``false_alarm``
    at that dwell whenever B' is evaluable (``same_as_b_prime`` records the check)."""
    from scripts import b_prime
    from scripts import dline_refit as dl
    from scripts import theta_verdict as tv

    vh = entry["verdict_for_holdout"]
    theta, tau_crit = float(vh["published"]["theta_m"]), float(vh["published"]["tau_crit"])
    direction = vh["fit_config"]["direction"]
    window_ms = float(dl.windowing_of(entry)["window_ms"])
    dwell = dl.ONLINE_DWELL_WINDOWS
    crit = tv.critical_dwell_flags(windows, theta=theta, tau_crit=tau_crit, direction=direction,
                                   dwell_windows=dwell, window_ms=window_ms)
    point = b_prime.series_point(windows, theta=theta, cut=math.inf, crit=crit)["false_alarm"]
    ci = b_prime.b_prime_boot(windows, cut=math.inf, crit=crit, n=n_resamples, seed=seed)["false_alarm_ci95"]
    healthy = sum(1 for w in windows if math.isfinite(w.signal) and w.slo_met)
    criteria = [dl._criterion("CRITICAL false alarm on healthy windows (dwell 1)", point, "<=",
                              gate["false_alarm_max"]),
                dl._criterion("its CI95 upper bound", ci[1], "<=", gate["false_alarm_ci95_high_max"])]
    same = None
    bd = ((b_prime_block or {}).get("by_dwell") or {}).get(str(dwell))
    if bd:
        same = _close(bd.get("false_alarm"), point) and all(_close(a, b) for a, b in zip(bd.get("false_alarm_ci95")
                                                                                        or [], ci))
    return {"criteria": criteria, "evaluable": healthy > 0, "passed": healthy > 0 and all(c["met"] for c in criteria),
            "dwell_windows": dwell, "healthy_windows": healthy, "false_alarm": point, "false_alarm_ci95": ci,
            "gate": dict(gate), "same_as_b_prime": same,
            "unit": "healthy WINDOWS (slo_met, finite signal) CRITICAL at dwell 1 / healthy windows; CI resamples cells"}


def evaluate(inp: Mapping[str, Any], csv_path: Path, cells: Mapping[tuple, Mapping[str, Any]],
             requests_csv: Path, *, n_resamples: int, dry_run: bool) -> dict:
    from scripts import dline_refit as dl
    from scripts.analysis import h_conservative_score as hcs
    from scripts.analysis import h_dropped_windows as hd

    model, doc = inp["model"], inp["doc"]
    entry = doc["models"][model]
    cfgs, bp_summary, problems = dl.b_prime_inputs(doc, None, dl.ONLINE_DWELL_WINDOWS,
                                                   "prereg: dwell 1 (the controller's TRE_DWELL_WINDOWS)")
    if problems:
        raise Refused(problems)
    ev = dl.evaluate_model(entry, csv_path, n_resamples=n_resamples, seed=PREREG_SEED, b_prime_cfg=cfgs[model])
    crit = ev["criteria"]
    a_ok, bp_ok = bool(crit["A"]["passed"]), bool(crit["B_prime"]["passed"])
    spec, label, trim = hd.frozen_spec_and_label(entry)
    windows = spec.load(csv_path, label, trim)
    shape_of, kind_of = {}, {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sid = (r.get("scenario_id") or "unknown").strip() or "unknown"
            meta = cells[(r["cell_id"], int(float(r["attempt"] or 1)))]
            shape_of[sid], kind_of[meta["shape"]] = meta["shape"], meta["kind"]
    audit = censoring_audit(requests_csv, {k: {"shape": v["shape"], "kind": v["kind"],
                                               "runtime_voids": int(v.get("runtime_voids") or 0)}
                                           for k, v in cells.items()})
    vs = void_status(audit)
    v2 = inp.get("schema") == "v2"
    if v2:
        fa = fa_gate(entry, windows, gate=(inp["prereg"]["evaluation"]["FA"]["gate"]), n_resamples=n_resamples,
                     seed=PREREG_SEED, b_prime_block=crit["B_prime"])
        gate_ok = a_ok and bool(fa["passed"])
    else:
        gate_ok = a_ok and bp_ok
    if dry_run:
        status, verdict = "dry_run", None
    elif vs["void_status_primary"]["status"] != STATUS_EVALUATED:
        status, verdict = vs["void_status_primary"]["status"], None
    else:
        status, verdict = STATUS_EVALUATED, (v2_verdict(a_ok, fa["passed"]) if v2 else (
            dl.VERDICT_PASS if gate_ok else dl.VERDICT_FAIL))
    dropped = hd.audit_csv(csv_path, entry, model=model, sealed_to_h2=False, read_holdout=True)
    bp_unit = ("recall_severe = severe violating WINDOWS CRITICAL at dwell 1 / severe violating windows "
               "(b_prime.series_point; per window, not per episode); CI resamples cells")
    if v2:
        head = {"A": crit["A"], "FA": fa, "onset_episodes": dict(ONSET_NOT_APPLICABLE),
                "B_prime_disclosure": {**crit["B_prime"], "gating": False,
                                       "note": "window B' is disclosed, not gating (v2 rule)"},
                "B_prime_unit": bp_unit,
                "pass_a_disclosed_consequence": inp["prereg"]["evaluation"]["outcome_statements"]["pass_a_disclosed"]}
    else:
        head = {"A": crit["A"], "B_prime": crit["B_prime"], "B_prime_unit": bp_unit}
    metrics = {
        **head,
        "disclosed": {"B_old": crit["B"], "C": crit["C"], "D": crit["D"],
                      "ranking_disclosure": ev["ranking_disclosure"], "holdout_report": ev["holdout_report"]},
        "b_prime_config": bp_summary,
        "counts": ev["M"],
        "cross_shape": cross_shape(entry, windows, shape_of, kind_of, n_resamples=n_resamples, seed=PREREG_SEED,
                                   ci_method=CROSS_SHAPE_CI_METHOD if v2 else None),
        "zero_token_dropped_windows": {"what": "disclosure (same as H)", "total": dropped["total"]},
        "conservative_disclosure": hcs.score_model(entry, csv_path, b_prime_cfg=cfgs[model],
                                                   n_resamples=n_resamples, seed=PREREG_SEED,
                                                   check_accept_path=False),
        "kv_missing_samples": kv_missing(cells),
    }
    out = {"status": status, "verdict": verdict,
           "verdict_rule": V2_VERDICT_RULE if v2 else (
               "only with status 'evaluated': pass iff A (BA >= .80, CI95 low >= .75, drop from "
               "training <= .08) and B' (dwell 1) all met; fail = reported as is"),
           "void_status": vs, "censoring_audit": audit}
    if status == STATUS_EVALUATED:
        out["evaluation"] = metrics
    else:
        out["disclosure_not_an_evaluation"] = {
            "note": ("DRY RUN on training data" if dry_run else
                     f"status {status}: no verdict; every metric below is disclosure, not an evaluation"),
            **metrics}
    return out


def check_dry_run_record(path: Optional[Path], scorer_sha: str, code: Mapping[str, Any], *,
                         prereg_sha256: Optional[str] = None, attribution: Optional[str] = None) -> list[str]:
    """'dry run on the frozen / training set first': a real run needs a dry-run output of
    this commit and this scorer file, from a clean tree - and, when given, of the same
    preregistration and the same label attribution (a v1 dry-run record has no attribution:
    completion)."""
    if path is None:
        return ["--dry-run-result is required for a real run (dry run on the frozen / training set first)"]
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"--dry-run-result {path}: unreadable ({exc})"]
    problems = []
    if d.get("dry_run") is not True:
        problems.append(f"{path}: not a dry-run output")
    if not code.get("commit") or code.get("dirty") is not False:
        problems.append(f"code state {code}: a real run needs a known commit and a clean tree")
    if (d.get("code") or {}).get("commit") != code.get("commit"):
        problems.append(f"{path}: dry run of commit {(d.get('code') or {}).get('commit')}, not {code.get('commit')}")
    if (d.get("scorer") or {}).get("sha256") != scorer_sha:
        problems.append(f"{path}: dry run of another scorer file")
    if prereg_sha256 is not None and (d.get("prereg") or {}).get("sha256") != prereg_sha256:
        problems.append(f"{path}: dry run under another preregistration ({(d.get('prereg') or {}).get('sha256')})")
    if attribution is not None:
        had = (d.get("label_attribution") or {}).get("value") or ATTRIBUTION_COMPLETION
        if had != attribution:
            problems.append(f"mixed label attributions: {path} is a dry run under {had!r}, this run is {attribution!r}")
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    from scripts import dline_refit as dl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prereg", type=Path, required=True)
    ap.add_argument("--addendum", type=Path, action="append", default=[],
                    help="v1 only: both T14 addenda (a v2 preregistration carries its rules inline)")
    ap.add_argument("--freeze-file", type=Path, required=True)
    ap.add_argument("--t14-manifest", type=Path, default=None)
    ap.add_argument("--dataset", type=Path, default=None,
                    help="the T14 standard dataset directory of the frozen label's attribution "
                         "(<run>/dataset = completion, <run>/dataset_hybrid = hybrid); a mismatch is refused")
    ap.add_argument("--dry-run-dataset", type=Path, default=None,
                    help="a TRAINING dataset instead of T14 (proves the code runs; no verdict)")
    ap.add_argument("--dry-run-result", type=Path, default=None,
                    help="real run: the dry-run output of this commit and scorer (required)")
    ap.add_argument("--resamples", type=int, default=PREREG_RESAMPLES,
                    help="dry run only; a real run refuses anything but the prereg's 1000")
    ap.add_argument("--out", type=Path, required=True, help="result JSON (write once; <out>.d/ holds the CSV)")
    args = ap.parse_args(argv)
    dry = args.dry_run_dataset is not None
    if dry == bool(args.t14_manifest or args.dataset):
        ap.error("give --t14-manifest and --dataset, or --dry-run-dataset")
    if not dry and args.resamples != PREREG_RESAMPLES:
        ap.error(f"--resamples is fixed at {PREREG_RESAMPLES} by the prereg")
    work = Path(f"{args.out}.d")
    if args.out.exists() or work.exists():
        ap.error(f"{args.out} or {work} exists: T14 is scored once")
    scorer = {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve()),
              "rule": "dry run on the frozen / training set first (a real run refuses without a dry-run output "
                      "of the same commit and scorer sha256)"}
    code = dl.code_state()
    try:
        inp = check_inputs(args.prereg, args.addendum, args.freeze_file)
        att_sources: dict[str, str] = {"frozen label (freeze verdict_for_holdout.label_def)": inp["freeze_attribution"]}
        if inp["schema"] == "v2":
            att_sources["prereg t14.conditions.label_attribution"] = str(
                inp["prereg"]["t14"]["conditions"]["label_attribution"])
        if dry:
            ds = args.dry_run_dataset
            ds = ds / "dataset" if not (ds / "windows.csv").exists() and (ds / "dataset" / "windows.csv").exists() else ds
            if any(part in ("M", "T14") for part in ds.resolve().parts):
                raise Refused([f"{ds}: the dry run never reads M / T14"])
            att_sources[f"dataset {ds}"] = dataset_attribution(ds)
            attribution, pr = check_attributions(att_sources)
            if pr:
                raise Refused(pr)
            header, rows, cells = dry_run_rows(ds, inp["model"])
            manifest_info = {"dry_run_dataset": str(ds)}
        else:
            pr = check_dry_run_record(args.dry_run_result, scorer["sha256"], code,
                                      prereg_sha256=inp["prereg_sha256"], attribution=inp["freeze_attribution"])
            if pr:
                raise Refused(pr)
            man, cells, ds = manifest_rows(inp, args.t14_manifest, args.dataset)
            src = dl.DatasetSource.parse(f"T14={ds}", sealed_to_h2=False)
            att_sources["T14 manifest label_def"] = label_attribution(man.get("label_def"))
            att_sources[f"dataset {src.directory}"] = dataset_attribution(src.directory)
            attribution, pr = check_attributions(att_sources)
            if pr:
                raise Refused(pr)
            if dl.dataset_numerator(src.directory) != "gateway":
                raise Refused([f"{ds}: not a gateway-numerator dataset"])
            m, pr = dl.collect_m_rows([src], {inp["model"]: man})
            if pr:
                raise Refused(pr)
            header, rows, ds = m["header"][inp["model"]], m["rows"][inp["model"]], src.directory
            manifest_info = {"path": str(args.t14_manifest), "sha256": _sha256(args.t14_manifest),
                             "sha256sums_sha256": man["sha256sums_sha256"],
                             "dry_run_result": {"path": str(args.dry_run_result),
                                                "sha256": _sha256(args.dry_run_result)}}
    except Refused as exc:
        print("t14_score REFUSED - nothing was written:")
        for p in exc.problems:
            print(f"  - {p}")
        return 1
    work.mkdir(parents=True)
    csv_path = work / f"{inp['model']}_t14_validation.csv"
    _write_csv(csv_path, header, rows)
    os.chmod(csv_path, 0o444)
    v2 = inp["schema"] == "v2"
    result = {
        "what": (("T14 evaluation by the preregistered rule v2 (next round: rules inline in the preregistration; "
                  "gates A and FA, onset not applicable, window B' disclosed; decisions D1-D5)") if v2 else
                 ("T14 evaluation by the preregistered rule (DRAFT scorer; prereg t14/preregistration.json + "
                  "ADDENDUM-T14-streamcut + ADDENDUM-T14-crossshape; decisions D1-D5 of 2026-10-04)")),
        "rule_version": inp["schema"],
        "label_attribution": {"value": attribution, "sources": att_sources,
                              "rule": "every source names the same attribution, or the scorer refuses"},
        "dry_run": dry,
        "dry_run_note": ("DRY RUN on a TRAINING dataset with fake kinds - not T14, no verdict" if dry else None),
        "scorer": scorer,
        "code": code,
        "prereg": {"path": str(args.prereg), "sha256": inp["prereg_sha256"]},
        "addenda": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in inp["addenda"].items()},
        "freeze": {"path": str(args.freeze_file), "sha256": inp["freeze_sha256"],
                   "freeze_sha256": inp["doc"]["freeze_sha256"],
                   "published": inp["doc"]["models"][inp["model"]]["published"]},
        "manifest": manifest_info,
        "dataset": str(ds),
        "validation_csv": {"path": str(csv_path), "sha256": _sha256(csv_path), "rows": len(rows)},
        "bootstrap": {"n_resamples": args.resamples, "seed": PREREG_SEED},
        "decisions": DECISIONS,
        "model": inp["model"],
        **evaluate(inp, csv_path, cells, Path(ds) / "requests.csv", n_resamples=args.resamples, dry_run=dry),
    }
    with open(args.out, "x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
    os.chmod(args.out, 0o444)
    body = result.get("evaluation") or result["disclosure_not_an_evaluation"]
    a, bp = body["A"], (body.get("FA") or body["B_prime"])
    gate_name = "FA" if v2 else "B'"
    print(f"[{inp['model']}] rule {inp['schema']} attribution {attribution} status {result['status']} verdict {result['verdict']} "
          f"(voided cells {result['void_status']['voided_cells']}; primary "
          f"{result['void_status']['void_status_primary']['status']}; run-level sensitivity "
          f"{result['void_status']['void_status_run_level_sensitivity']['status']}): "
          f"A {'pass' if a['passed'] else 'FAIL'} (BA {a['criteria'][0]['value']}, CI low {a['criteria'][1]['value']}); "
          f"{gate_name} {'pass' if bp['passed'] else 'FAIL'} "
          f"({[(c['name'], c['value']) for c in bp.get('criteria', [])]})")
    if v2:
        bd = body["B_prime_disclosure"]
        print(f"  window B' (disclosed): {[(c['name'], c['value']) for c in bd.get('criteria', [])]}; "
              f"onset episodes {body['onset_episodes']['status']}")
    for kind, block in body["cross_shape"].items():
        cl = block["claim"]
        print(f"  {kind}: per-shape BA " + ", ".join(f"{s} {t['ba']}" for s, t in block["per_shape"].items())
              + f"; claim {cl['one_theta_transfers']} (sample sd {cl.get('sd_sample')} vs median half "
                f"{cl.get('median_ci95_half_width')})")
    z = body["zero_token_dropped_windows"]["total"]["zero_token"]
    kv = body["kv_missing_samples"]
    print(f"  censoring audit totals {result['censoring_audit']['totals']}; void at audit "
          f"{result['censoring_audit']['void_at_audit']}; zero-token dropped {z['total']} (evidence "
          f"{z['backlog_or_failure']}); conservative added {body['conservative_disclosure']['added']}; "
          f"KV max missing share {kv['max_missing_share_cell']} unreadable {len(kv['cells_unreadable'])}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

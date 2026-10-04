#!/usr/bin/env python3
"""T14 scorer - the preregistered evaluation of the held-out 14b set, round 2026-10-03.
DRAFT for the owner's review (2026-10-04); not sealed, not run on T14.

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
#: The route timeout of the stream-cut addendum (``audit_rule.cut``: "e2e_ms >= 150000").
ROUTE_TIMEOUT_CUT_MS = 150_000.0
#: The original model-error limit the non-cut errors are judged against (``audit_rule.rule``).
NON_CUT_ERROR_LIMIT = 0.05
#: The runtime void limit of the addendum (``runtime_limit.max_model_error_rate``).
RUNTIME_MODEL_ERROR_LIMIT = 0.10
OUTCOME_MODEL_ERROR = "model_error"
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


def check_inputs(prereg_path: Path, addenda: Sequence[Path], freeze_file: Path) -> dict:
    """Every check on the prereg, the addenda and the freeze; raises :class:`Refused`."""
    from scripts import b_prime
    from scripts import dline_refit as dl

    problems: list[str] = []
    prereg_sha = _sidecar_ok(prereg_path, problems)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8")) if prereg_path.is_file() else {}
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
    rule = sc.get("audit_rule") or {}
    if "e2e_ms >= 150000" not in str(rule.get("cut") or ""):
        problems.append(f"stream-cut addendum cut {rule.get('cut')!r} is not 'e2e_ms >= 150000'")
    if "0.05" not in str(rule.get("rule") or ""):
        problems.append("stream-cut addendum rule does not name the 0.05 limit")
    if not _close((sc.get("runtime_limit") or {}).get("max_model_error_rate"), RUNTIME_MODEL_ERROR_LIMIT):
        problems.append("stream-cut addendum runtime_limit.max_model_error_rate != 0.10")
    cs = (((adds.get("crossshape") or {}).get("doc") or {}).get("chapter2_outputs") or {}).get("cross_shape") or {}
    if "SD of the per-shape BA <= the median per-shape BA CI95 half width" not in str(cs.get("claim_rule") or ""):
        problems.append(f"cross-shape claim rule {cs.get('claim_rule')!r} is not the one implemented")

    model = str((prereg.get("t14") or {}).get("model") or "")
    ps = (prereg.get("parameter_sets") or {}).get("freeze") or {}
    doc: dict = {}
    try:
        doc = dl.verify_freeze(freeze_file)
    except dl.FreezeError as exc:
        problems += exc.problems
    freeze_sha = _sha256(freeze_file) if freeze_file.is_file() else None
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
            pre_cut = ((ev.get("B_prime") or {}).get(f"{model}_cut"))
            if not _close(round(float(cut or 0), 5), pre_cut, 1e-12):
                problems.append(f"freeze {model} B' cut {cut} != prereg {pre_cut}")
            if str(entry.get("numerator") or "gateway") != "gateway":
                problems.append(f"freeze {model} numerator {entry.get('numerator')} != gateway (prereg data)")
        try:
            gate = b_prime.check_gate(doc.get("b_prime_gate") or {})
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
            "freeze_sha256": freeze_sha, "model": model}


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
    """The stream-cut audit per cell: every request of the cell's EVALUATED attempt (the
    manifest's; voided earlier attempts are other keys and are not read), warm-up included
    (the whole-cell ``model_errors / sent`` of ``openloop.check_cell``)."""
    per: dict[tuple, dict] = {k: {"sent": 0, "sent_in_warmup": 0, "model_error": 0, "cut": 0, "non_cut": 0,
                                  "non_cut_no_e2e": 0} for k in cells}
    with open(requests_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                key = (str(r["cell_id"]), int(float(r["attempt"] or 1)))
            except (KeyError, ValueError):
                continue
            c = per.get(key)
            if c is None:
                continue
            c["sent"] += 1
            c["sent_in_warmup"] += str(r.get("in_warmup") or "").strip().lower() in ("1", "true", "yes")
            if (r.get("outcome") or "").strip() != OUTCOME_MODEL_ERROR:
                continue
            c["model_error"] += 1
            try:
                e2e = float(r.get("e2e_ms") or "nan")
            except ValueError:
                e2e = math.nan
            if math.isfinite(e2e) and e2e >= ROUTE_TIMEOUT_CUT_MS:
                c["cut"] += 1
            else:
                c["non_cut"] += 1
                c["non_cut_no_e2e"] += not math.isfinite(e2e)
    out = []
    for (cid, att), c in sorted(per.items()):
        sent = c["sent"]
        rate = (c["non_cut"] / sent) if sent else None
        out.append({"cell_id": cid, "attempt": att, **cells[(cid, att)], **c,
                    "cut_share_of_sent": (c["cut"] / sent) if sent else None,
                    "non_cut_rate": rate,
                    "model_error_rate": (c["model_error"] / sent) if sent else None,
                    "void_at_audit": bool(sent == 0 or (rate is not None and rate > NON_CUT_ERROR_LIMIT)),
                    "runtime_limit_exceeded": bool(sent and c["model_error"] / sent > RUNTIME_MODEL_ERROR_LIMIT)})
    return {"rule": {"cut": f"outcome {OUTCOME_MODEL_ERROR} and e2e_ms >= {ROUTE_TIMEOUT_CUT_MS:.0f}",
                     "non_cut": "any other model_error, e2e_ms missing included (D4)",
                     "non_cut_limit": NON_CUT_ERROR_LIMIT, "runtime_limit": RUNTIME_MODEL_ERROR_LIMIT,
                     "scope": "the evaluated (valid) attempt only; voided earlier attempts excluded",
                     "denominator": "every request of that attempt in requests.csv, warm-up included"},
            "requests_csv": str(requests_csv), "cells": out,
            "void_at_audit": [c["cell_id"] for c in out if c["void_at_audit"]],
            "totals": {k: sum(c[k] for c in out) for k in ("sent", "model_error", "cut", "non_cut")}}


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


def _ba_ci_auroc(entry: Mapping[str, Any], windows: Sequence[Any], *, n_resamples: int, seed: int) -> dict:
    """BA at the published theta (+ cell-bootstrap CI95, accept's bootstrap) and AUROC
    (+ CI95, ``ranking.ranking_disclosure``) of one window subset; both ``undefined`` (None)
    when the subset holds one class only (D2)."""
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
    boot = dl.acceptance_bootstrap(windows, crit, theta=theta, direction=direction, n_resamples=n_resamples,
                                   seed=seed)
    ci = boot["metrics"]["balanced_accuracy"]["ci95"]
    recs = ranking.records_from_windows(vh["model"], windows, theta=theta, direction=direction)
    au = ranking.ranking_disclosure(recs, n_resamples=n_resamples, seed=seed).get("auroc") or {}
    return {**base, "ba": ba, "ba_ci95": ci,
            "ba_ci95_half_width": ((ci[1] - ci[0]) / 2.0) if None not in ci else None,
            "ba_resamples_used": boot["metrics"]["balanced_accuracy"]["resamples_used"],
            "auroc": au.get("value"), "auroc_ci95": au.get("ci95"), "auroc_resamples_used": au.get("resamples_used")}


def claim(table: Mapping[str, Mapping[str, Any]]) -> dict:
    """The cross-shape claim of one kind (D1, D2)."""
    single = sorted(s for s, t in table.items() if t["single_class"])
    bas = [t["ba"] for s, t in sorted(table.items()) if not t["single_class"]]
    halves = [t["ba_ci95_half_width"] for s, t in sorted(table.items())
              if not t["single_class"] and t["ba_ci95_half_width"] is not None]
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
                kind_of: Mapping[str, str], *, n_resamples: int, seed: int) -> dict:
    """Per-shape tables split by kind, per-kind pooled numbers (disclosure), the claim per kind."""
    by_shape: dict[str, list] = defaultdict(list)
    for w in windows:
        by_shape[shape_of[w.scenario_id]].append(w)
    out: dict[str, Any] = {}
    for kind in KINDS:
        shapes = sorted(s for s in by_shape if kind_of[s] == kind)
        table = {s: _ba_ci_auroc(entry, by_shape[s], n_resamples=n_resamples, seed=seed) for s in shapes}
        pooled = [w for s in shapes for w in by_shape[s]]
        out[kind] = {"per_shape": table,
                     "pooled_disclosure": (_ba_ci_auroc(entry, pooled, n_resamples=n_resamples, seed=seed)
                                           if pooled else None),
                     "claim": claim(table)}
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
    if dry_run:
        status, verdict = "dry_run", None
    elif vs["void_status_primary"]["status"] != STATUS_EVALUATED:
        status, verdict = vs["void_status_primary"]["status"], None
    else:
        status, verdict = STATUS_EVALUATED, ("pass" if a_ok and bp_ok else "fail")
    dropped = hd.audit_csv(csv_path, entry, model=model, sealed_to_h2=False, read_holdout=True)
    metrics = {
        "A": crit["A"], "B_prime": crit["B_prime"],
        "B_prime_unit": "recall_severe = severe violating WINDOWS CRITICAL at dwell 1 / severe violating windows "
                        "(b_prime.series_point; per window, not per episode); CI resamples cells",
        "disclosed": {"B_old": crit["B"], "C": crit["C"], "D": crit["D"],
                      "ranking_disclosure": ev["ranking_disclosure"], "holdout_report": ev["holdout_report"]},
        "b_prime_config": bp_summary,
        "counts": ev["M"],
        "cross_shape": cross_shape(entry, windows, shape_of, kind_of, n_resamples=n_resamples, seed=PREREG_SEED),
        "zero_token_dropped_windows": {"what": "disclosure (same as H)", "total": dropped["total"]},
        "conservative_disclosure": hcs.score_model(entry, csv_path, b_prime_cfg=cfgs[model],
                                                   n_resamples=n_resamples, seed=PREREG_SEED,
                                                   check_accept_path=False),
        "kv_missing_samples": kv_missing(cells),
    }
    out = {"status": status, "verdict": verdict,
           "verdict_rule": "only with status 'evaluated': pass iff A (BA >= .80, CI95 low >= .75, drop from "
                           "training <= .08) and B' (dwell 1) all met; fail = reported as is",
           "void_status": vs, "censoring_audit": audit}
    if status == STATUS_EVALUATED:
        out["evaluation"] = metrics
    else:
        out["disclosure_not_an_evaluation"] = {
            "note": ("DRY RUN on training data" if dry_run else
                     f"status {status}: no verdict; every metric below is disclosure, not an evaluation"),
            **metrics}
    return out


def check_dry_run_record(path: Optional[Path], scorer_sha: str, code: Mapping[str, Any]) -> list[str]:
    """'dry run on the frozen / training set first': a real run needs a dry-run output of
    this commit and this scorer file, from a clean tree."""
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
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    from scripts import dline_refit as dl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prereg", type=Path, required=True)
    ap.add_argument("--addendum", type=Path, action="append", default=[], help="both T14 addenda")
    ap.add_argument("--freeze-file", type=Path, required=True)
    ap.add_argument("--t14-manifest", type=Path, default=None)
    ap.add_argument("--dataset", type=Path, default=None, help="the T14 standard dataset directory")
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
        if dry:
            ds = args.dry_run_dataset
            ds = ds / "dataset" if not (ds / "windows.csv").exists() and (ds / "dataset" / "windows.csv").exists() else ds
            if any(part in ("M", "T14") for part in ds.resolve().parts):
                raise Refused([f"{ds}: the dry run never reads M / T14"])
            header, rows, cells = dry_run_rows(ds, inp["model"])
            manifest_info = {"dry_run_dataset": str(ds)}
        else:
            pr = check_dry_run_record(args.dry_run_result, scorer["sha256"], code)
            if pr:
                raise Refused(pr)
            man, cells, ds = manifest_rows(inp, args.t14_manifest, args.dataset)
            src = dl.DatasetSource.parse(f"T14={ds}", sealed_to_h2=False)
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
    result = {
        "what": ("T14 evaluation by the preregistered rule (DRAFT scorer; prereg t14/preregistration.json + "
                 "ADDENDUM-T14-streamcut + ADDENDUM-T14-crossshape; decisions D1-D5 of 2026-10-04)"),
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
    a, bp = body["A"], body["B_prime"]
    print(f"[{inp['model']}] status {result['status']} verdict {result['verdict']} "
          f"(voided cells {result['void_status']['voided_cells']}; primary "
          f"{result['void_status']['void_status_primary']['status']}; run-level sensitivity "
          f"{result['void_status']['void_status_run_level_sensitivity']['status']}): "
          f"A {'pass' if a['passed'] else 'FAIL'} (BA {a['criteria'][0]['value']}, CI low {a['criteria'][1]['value']}); "
          f"B' {'pass' if bp['passed'] else 'FAIL'} ({[(c['name'], c['value']) for c in bp.get('criteria', [])]})")
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

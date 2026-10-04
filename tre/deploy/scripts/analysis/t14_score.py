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
  Verdict = A passed and B' passed (D is not part of the T14 rule);
* per shape and per kind (interpolation / extrapolation): BA at the published theta with
  ``dline_refit.acceptance_bootstrap`` (1000, seed 20260922) and AUROC with
  ``tre_calibration.ranking.ranking_disclosure`` (same resamples); the cross-shape claim per
  kind with the SAMPLE standard deviation (n - 1) - the population SD is reported next to it
  (the addendum does not say which; see ``ambiguities`` in the output);
* censoring audit from the T14 dataset's ``requests.csv`` (every request of the cell's
  attempt, warm-up included - the runtime guard counts the same): model_error with
  ``e2e_ms >= 150000`` = cut; any other model_error (an ``e2e_ms`` missing included) = non-cut;
  a cell whose non-cut errors / sent > 0.05 is ``void_at_audit``. With such a cell the
  verdict is ``undetermined_audit_void`` (the prereg does not say whether the run is then
  evaluated without the cell or re-run; owner decision);
* also disclosed (draft H addendum, not preregistered): the conservative variant of
  :mod:`scripts.analysis.h_conservative_score` on the same validation CSV.

Not implemented (disclosure only in the cross-shape addendum): Fig 2.1 (needs the 1 Hz KV
sidecar), the lambda disclosure.

Runs once: ``--out`` and ``<out>.d/`` must not exist; both are made read-only.

    cd tre/deploy && PYTHONPATH=../common:.:../controller:../service-manager:../calibration:../replayer:../ui \\
      python3 -m scripts.analysis.t14_score --prereg $C/t14/preregistration.json \\
        --addendum $C/prereg/ADDENDUM-T14-streamcut.json --addendum $C/prereg/ADDENDUM-T14-crossshape.json \\
        --freeze-file $C/freeze/params_freeze.json --t14-manifest $C/T14/dsqwen-14b/T14_manifest.json \\
        --dataset $C/T14/dsqwen-14b/dataset --out $C/eval/T14_score.json

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

AMBIGUITIES = [
    "cross-shape claim rule: 'the SD of the per-shape BA' does not say sample (n-1) or population (n); "
    "this draft uses the sample SD and reports both (with 4 shapes they differ by x1.155)",
    "cross-shape claim rule: a shape whose windows hold one class only has no BA; this draft leaves it out "
    "of the SD and the median and reports it (the addendum is silent)",
    "stream-cut audit: what happens to the evaluation when a cell is void at audit (non-cut model errors "
    "> 0.05) is not stated (drop the cell, or re-run all 24 as the void_rule does for a stopped run); this "
    "draft then reports the metrics but no verdict ('undetermined_audit_void')",
    "stream-cut audit: a model_error without e2e_ms is counted as non-cut (cannot be shown to be a cut)",
    "A's 'max_drop_from_training_ba': the training BA is the freeze's train_ba_at_published (all training "
    "shapes pooled), as accept uses it",
    "per-kind pooled BA / AUROC are reported in addition to the per-shape tables (the prereg asks only for "
    "per-shape BA split by kind)",
]


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
    cells = {(str(c["cell_id"]), int(c["attempt"])): {"shape": c["shape"], "kind": c["kind"]} for c in man["cells"]}
    return man, cells, Path(dataset_dir)


def censoring_audit(requests_csv: Path, cells: Mapping[tuple, Mapping[str, Any]]) -> dict:
    """The stream-cut audit per cell (every request of the cell's listed attempt)."""
    per: dict[tuple, dict] = {k: {"sent": 0, "model_error": 0, "cut": 0, "non_cut": 0, "non_cut_no_e2e": 0}
                              for k in cells}
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
                     "non_cut_limit": NON_CUT_ERROR_LIMIT, "runtime_limit": RUNTIME_MODEL_ERROR_LIMIT,
                     "denominator": "every request of the cell attempt in requests.csv (warm-up included)"},
            "requests_csv": str(requests_csv), "cells": out,
            "void_at_audit": [c["cell_id"] for c in out if c["void_at_audit"]],
            "totals": {k: sum(c[k] for c in out) for k in ("sent", "model_error", "cut", "non_cut")}}


def _ba_ci_auroc(entry: Mapping[str, Any], windows: Sequence[Any], *, n_resamples: int, seed: int) -> dict:
    """BA at the published theta (+ cell-bootstrap CI95, accept's bootstrap) and AUROC
    (+ CI95, ``ranking.ranking_disclosure``) of one window subset."""
    from tre_calibration import ranking
    from tre_calibration.fit import threshold_balanced_accuracy

    from scripts import dline_refit as dl
    from scripts import theta_verdict as tv

    vh = entry["verdict_for_holdout"]
    theta, tau_crit = float(vh["published"]["theta_m"]), float(vh["published"]["tau_crit"])
    direction = vh["fit_config"]["direction"]
    viol = sum(1 for w in windows if not w.slo_met)
    cells = len({w.scenario_id for w in windows})
    ba = (threshold_balanced_accuracy(windows, theta=theta, direction=direction)["balanced_accuracy"]
          if 0 < viol < len(windows) else None)
    crit = tv.critical_dwell_flags(windows, theta=theta, tau_crit=tau_crit, direction=direction,
                                   dwell_windows=dl.ONLINE_DWELL_WINDOWS)
    boot = dl.acceptance_bootstrap(windows, crit, theta=theta, direction=direction, n_resamples=n_resamples,
                                   seed=seed)
    ci = boot["metrics"]["balanced_accuracy"]["ci95"]
    recs = ranking.records_from_windows(vh["model"], windows, theta=theta, direction=direction)
    rd = ranking.ranking_disclosure(recs, n_resamples=n_resamples, seed=seed)
    return {"windows": len(windows), "violating": viol, "cells": cells, "ba": ba, "ba_ci95": ci,
            "ba_ci95_half_width": ((ci[1] - ci[0]) / 2.0) if None not in ci and ba is not None else None,
            "ba_resamples_used": boot["metrics"]["balanced_accuracy"]["resamples_used"],
            "auroc": (rd.get("auroc") or {}), "single_class": ba is None}


def cross_shape(entry: Mapping[str, Any], windows: Sequence[Any], shape_of: Mapping[str, str],
                kind_of: Mapping[str, str], *, n_resamples: int, seed: int) -> dict:
    """Per-shape tables split by kind, per-kind pooled numbers, and the claim rule per kind."""
    by_shape: dict[str, list] = defaultdict(list)
    for w in windows:
        by_shape[shape_of[w.scenario_id]].append(w)
    out: dict[str, Any] = {}
    for kind in KINDS:
        shapes = sorted(s for s in by_shape if kind_of[s] == kind)
        table = {s: _ba_ci_auroc(entry, by_shape[s], n_resamples=n_resamples, seed=seed) for s in shapes}
        bas = [t["ba"] for t in table.values() if t["ba"] is not None]
        halves = [t["ba_ci95_half_width"] for t in table.values() if t["ba_ci95_half_width"] is not None]
        sd_s = statistics.stdev(bas) if len(bas) >= 2 else None
        sd_p = statistics.pstdev(bas) if len(bas) >= 1 else None
        med = statistics.median(halves) if halves else None
        pooled = [w for s in shapes for w in by_shape[s]]
        out[kind] = {
            "per_shape": table,
            "pooled": _ba_ci_auroc(entry, pooled, n_resamples=n_resamples, seed=seed) if pooled else None,
            "claim": {
                "rule": "SD of the per-shape BA <= the median per-shape BA CI95 half width",
                "shapes_with_ba": len(bas), "shapes_single_class": [s for s, t in table.items() if t["ba"] is None],
                "sd_sample": sd_s, "sd_population": sd_p, "median_ci95_half_width": med,
                "one_theta_transfers": (sd_s <= med) if sd_s is not None and med is not None else None,
                "one_theta_transfers_population_sd": (sd_p <= med) if sd_p is not None and med is not None else None,
                "role": "claim rule for the text; not an acceptance gate",
            },
        }
    return out


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    with open(path, "x", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(header))
        for r in rows:
            w.writerow([r.get(c, "") for c in header])


def dry_run_rows(dataset_dir: Path, model: str) -> tuple[list[str], list[dict], dict]:
    """A TRAINING dataset's training-set rows of ``model`` (dry run only)."""
    from scripts import dline_refit as dl

    with open(dataset_dir / "windows.csv", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows = [r for r in reader if r.get("model") == model and (r.get("cell_status") or "valid") == "valid"
                and (r.get("split") or "") != "holdout"
                and dl.assign_set(r, sealed_to_h2=True, sentinels=True) == dl.SET_TRAINING
                and r.get("shape") in DRY_RUN_KINDS]
    cells = {(r["cell_id"], int(float(r["attempt"] or 1))): {"shape": r["shape"], "kind": DRY_RUN_KINDS[r["shape"]]}
             for r in rows}
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
    audit = censoring_audit(requests_csv, cells)
    if dry_run:
        verdict = "dry_run_no_verdict"
    elif audit["void_at_audit"]:
        verdict = "undetermined_audit_void"
    else:
        verdict = "pass" if a_ok and bp_ok else "fail"
    return {
        "verdict": verdict,
        "verdict_rule": "pass iff A (BA >= .80, CI95 low >= .75, drop from training <= .08) and B' (dwell 1) all "
                        "met; fail = reported as is",
        "A": crit["A"], "B_prime": crit["B_prime"],
        "disclosed": {"B_old": crit["B"], "C": crit["C"], "D": crit["D"],
                      "ranking_disclosure": ev["ranking_disclosure"], "holdout_report": ev["holdout_report"]},
        "b_prime_config": bp_summary,
        "M_like_counts": ev["M"],
        "cross_shape": cross_shape(entry, windows, shape_of, kind_of, n_resamples=n_resamples, seed=PREREG_SEED),
        "censoring_audit": audit,
        "conservative_disclosure": hcs.score_model(entry, csv_path, b_prime_cfg=cfgs[model],
                                                   n_resamples=n_resamples, seed=PREREG_SEED,
                                                   check_accept_path=False),
    }


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
            man, cells, ds = manifest_rows(inp, args.t14_manifest, args.dataset)
            src = dl.DatasetSource.parse(f"T14={ds}", sealed_to_h2=False)
            if dl.dataset_numerator(src.directory) != "gateway":
                raise Refused([f"{ds}: not a gateway-numerator dataset"])
            m, pr = dl.collect_m_rows([src], {inp["model"]: man})
            if pr:
                raise Refused(pr)
            header, rows, ds = m["header"][inp["model"]], m["rows"][inp["model"]], src.directory
            manifest_info = {"path": str(args.t14_manifest), "sha256": _sha256(args.t14_manifest),
                             "sha256sums_sha256": man["sha256sums_sha256"]}
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
                 "ADDENDUM-T14-streamcut + ADDENDUM-T14-crossshape)"),
        "dry_run": dry,
        "dry_run_note": ("DRY RUN on a TRAINING dataset with fake kinds - not T14, no verdict" if dry else None),
        "prereg": {"path": str(args.prereg), "sha256": inp["prereg_sha256"]},
        "addenda": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in inp["addenda"].items()},
        "freeze": {"path": str(args.freeze_file), "sha256": inp["freeze_sha256"],
                   "freeze_sha256": inp["doc"]["freeze_sha256"],
                   "published": inp["doc"]["models"][inp["model"]]["published"]},
        "manifest": manifest_info,
        "dataset": str(ds),
        "validation_csv": {"path": str(csv_path), "sha256": _sha256(csv_path), "rows": len(rows)},
        "code": dl.code_state(),
        "bootstrap": {"n_resamples": args.resamples, "seed": PREREG_SEED},
        "ambiguities": AMBIGUITIES,
        "model": inp["model"],
        **evaluate(inp, csv_path, cells, Path(ds) / "requests.csv", n_resamples=args.resamples, dry_run=dry),
    }
    with open(args.out, "x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
    os.chmod(args.out, 0o444)
    a, bp = result["A"], result["B_prime"]
    print(f"[{inp['model']}] {'DRY RUN ' if dry else ''}verdict {result['verdict']}: "
          f"A {'pass' if a['passed'] else 'FAIL'} (BA {a['criteria'][0]['value']}, CI low {a['criteria'][1]['value']}); "
          f"B' {'pass' if bp['passed'] else 'FAIL'} ({[(c['name'], c['value']) for c in bp.get('criteria', [])]})")
    for kind, block in result["cross_shape"].items():
        cl = block["claim"]
        print(f"  {kind}: per-shape BA " + ", ".join(f"{s} {t['ba']}" for s, t in block["per_shape"].items())
              + f"; claim {cl['one_theta_transfers']} (sd {cl['sd_sample']} vs median half {cl['median_ci95_half_width']})")
    print(f"  censoring audit totals {result['censoring_audit']['totals']}; void at audit "
          f"{result['censoring_audit']['void_at_audit']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

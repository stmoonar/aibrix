#!/usr/bin/env python3
"""Write the T14 preregistration of the next calibration round (schema ``t14-prereg-v2``).

Design 2026-10-05 (user decision 4): T14 is re-collected under label v2 (hybrid
request-to-window attribution) - the same design as the 2026-10-03 round (dsqwen-14b, the
8 held-out shapes x {0.9, 1.0, 1.1} x C^_s of the capacity prior, 240 s holds), with a new
design seed and a fresh cell serial base, after 14b's M2 in the same exclusive window. The
document binds what ``calibration_t14.check_preregistration`` checks at collection time and
what ``scripts.analysis.t14_score`` (rule v2) checks at scoring time; the rules that were
addenda in the 2026-10-03 round (stream cut, cross-shape claim, scoring decisions D1-D5)
are inline.

    cd tre/deploy && python3 -m scripts.t14_prereg \\
        --out <dir>/preregistration.json --design-seed <seed> --cell-serial-base <base> \\
        --capacity-prior <prior.json> --boundary-table <b50.csv> \\
        --freeze-file <freeze> | --draft \\
        --gateway-url http://<gateway>/v1/chat/completions --engine-image <image> \\
        --forbidden-root <root> ... --dev-root <opened T14 / M root> ... \\
        --ledger-glob '<root>/**/cells.jsonl' ...

``--draft`` writes a document marked DRAFT with placeholders for what does not exist yet
(the freeze, the label sha when the freeze is not given, the scorer commit); it gets no
sha256 sidecar, so the collection refuses it. A final document (no ``--draft``) refuses
any placeholder, gets its sidecar, and both files are made read-only. Written once: an
existing ``--out`` is refused.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts import calibration_campaign as campaign
from scripts import calibration_design as design
from scripts import calibration_t14 as t14
from scripts.analysis import t14_score

SCHEMA = "t14-prereg-v2"
PLACEHOLDER = "PLACEHOLDER"
LABEL_NAME_V2 = "p95_ttft_slowdown_tpot_plus_unserved_v2"
ATTRIBUTION = "hybrid"
SCORE_SEED, SCORE_RESAMPLES = 20260922, 1000
CROSS_SHAPE_CLAIM_RULE = ("'one theta transfers across shapes' iff the SD of the per-shape BA <= the median "
                          "per-shape BA CI95 half width; stated separately for interpolation and extrapolation")
VOID_RULE = ("a void cell is re-driven once; a second void stops the run; a stopped run is never "
             "evaluated - re-run all 24 cells into a new root")
DECISIONS = {
    "D1_cross_shape_sd": "the cross-shape claim uses the sample SD (n-1) of the per-shape BA; the population SD is "
                         "disclosure only",
    "D2_single_class_shape": "a shape whose windows are all one class has BA and AUROC undefined; the claim of its "
                             "kind is 'not_evaluable'; the shape is never dropped to claim on the remaining shapes; "
                             "the SD over the remaining shapes is disclosure only; the pooled A (all windows) is "
                             "unaffected",
    "D3_void_at_audit": "a cell is void at audit when its non-cut model errors / sent > 0.05; voids of a cell = its "
                        "run-time void attempts (T14_manifest attempts with void_reasons) + 1 if void at audit. "
                        "PRIMARY (decides): a cell with 2 voids -> run_void (never evaluated; re-run all 24 into a "
                        "new root); else any audit-void cell -> void_redrive_required:<cells> (re-drive each once, "
                        "then score); else evaluated. SENSITIVITY (disclosed side by side, never decides): run "
                        "level, 2 or more voids anywhere -> run_void",
    "D4_model_error_without_e2e": "a model_error without e2e_ms is non-cut",
    "D5_training_ba": "A's max-drop-from-training check uses the freeze's pooled train_ba_at_published (as accept)",
}
DISCLOSURE_RULES = {
    "audit_scope": "the censoring audit counts the evaluated (valid) attempt of each manifest cell only; voided "
                   "earlier attempts are excluded; every request of that attempt counts, warm-up included "
                   "(openloop.check_cell's whole-cell model_errors / sent)",
    "zero_token_windows": "dropped zero-token windows and the conservative variant (counted as non-CRITICAL misses) "
                          "are disclosed",
    "per_shape_auroc": "a single-class shape has AUROC undefined (D2)",
    "kv_missing_samples": "per cell, the share of missing 1 Hz instant-sidecar kv_cache_usage samples and the largest "
                          "gap are disclosed",
    "b_prime_unit": "window B' recall_severe at dwell 1 is a rate over windows, not episodes; disclosed, not gating",
    "attribution_disclosure": "the completion-attribution score of the same T14 data may be reported next to the "
                              "hybrid one, labelled as a disclosure; it never decides",
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state(repo: Path) -> dict:
    def run(*a: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *a], text=True).strip()
    try:
        return {"branch": run("rev-parse", "--abbrev-ref", "HEAD"), "commit": run("rev-parse", "HEAD"),
                "dirty": bool(run("status", "--short", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"branch": None, "commit": None, "dirty": None}


def disjointness(rows: Sequence[Mapping[str, Any]], ledger_globs: Sequence[str]) -> dict:
    """Arrival seeds and cell ids of ``rows`` against every ledger line (``cells.jsonl``)."""
    seeds, ids = set(), set()
    ledgers = sorted({f for g in ledger_globs for f in glob.glob(g, recursive=True)})
    for f in ledgers:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("arrival_seed") is not None:
                    seeds.add(int(d["arrival_seed"]))
                if d.get("cell_id"):
                    ids.add(str(d["cell_id"]))
    mine_seeds = {int(r["arrival_seed"]) for r in rows}
    mine_ids = {str(r["cell_id"]) for r in rows}
    return {"checked_against": list(ledger_globs), "ledgers": len(ledgers), "existing_arrival_seeds": len(seeds),
            "existing_cell_ids": len(ids), "t14_arrival_seed_overlap": sorted(mine_seeds & seeds),
            "t14_cell_id_overlap": sorted(mine_ids & ids)}


def cell_rows(design_seed: int, serial_base: int, prior: Mapping[str, Any]) -> tuple[list[dict], list]:
    cells = t14.new_cells(t14.MODEL, design_seed, serial_base)
    t14.check_held_out(cells)
    order = t14.interleaved_order(cells, t14.MODEL, design_seed)
    rows = []
    for i, c in enumerate(order, 1):
        cap = float(prior["predicted_rps"][c.shape])
        rows.append({"order": i, "cell_id": c.cell_id, "shape": c.shape, "kind": t14.shape_kind(c.shape),
                     "factor": c.rho_factor, "rps": round(float(c.rho_factor) * cap, 4), "hold_s": c.duration_s,
                     "warmup_s": c.warmup_s, "arrival_seed": c.arrival_seed, "prompt_key": c.prompt_key,
                     "split": c.split, "role": c.role})
    return rows, order


def build(args) -> dict:
    prior = t14.load_capacity_prior(Path(args.capacity_prior), t14.MODEL)
    rows, order = cell_rows(args.design_seed, args.cell_serial_base, prior)
    disjoint = disjointness(rows, args.ledger_glob)
    est = t14.estimate(order, args.cooldown_s)
    mml = t14.check_max_model_len(t14.MODEL, t14.SHAPES, args.registry)
    freeze_block: dict[str, Any]
    label_sha, label_name = args.label_def_sha256 or PLACEHOLDER, LABEL_NAME_V2
    cut: Any = PLACEHOLDER
    if args.freeze_file:
        from scripts import dline_refit as dl

        fz = dl.verify_freeze(Path(args.freeze_file))
        fm = fz["models"][t14.MODEL]
        label = (fm.get("verdict_for_holdout") or {}).get("label_def") or {}
        if str(label.get("attribution") or "completion") != ATTRIBUTION:
            raise ValueError(f"{args.freeze_file}: {t14.MODEL} was frozen under the "
                             f"{label.get('attribution') or 'completion'!r} attribution, T14 v2 is {ATTRIBUTION!r}")
        label_sha, label_name = dl.canonical_sha256(label), str(label.get("name") or LABEL_NAME_V2)
        cut = round(float(fm["b_prime"]["severity_cut"]), 5)
        freeze_block = {"path": str(Path(args.freeze_file).resolve()), "sha256": _sha(Path(args.freeze_file)),
                        "freeze_sha256": fz["freeze_sha256"], t14.MODEL: fm["published"]}
    else:
        freeze_block = {"path": PLACEHOLDER, "sha256": PLACEHOLDER, "freeze_sha256": PLACEHOLDER,
                        t14.MODEL: PLACEHOLDER,
                        "note": "the next-round freeze does not exist yet (hybrid refit + run_E + freeze)"}
    roots = list(dict.fromkeys([*args.forbidden_root, *args.dev_root]))
    code = git_state(Path(args.code_repo)) if args.code_repo else {"branch": None, "commit": None, "dirty": None}
    interp, extrap = list(t14.INTERPOLATION_SHAPES), list(t14.EXTRAPOLATION_SHAPES)
    doc = {
        "schema": SCHEMA,
        "status": "DRAFT - not sealed; placeholders marked PLACEHOLDER" if args.draft else "sealed",
        "what": ("T14 re-collection under label v2 (hybrid attribution), next calibration round - preregistration: "
                 "design, capacity prior, label binding, stream-cut rule, evaluation rule (gates A and FA; onset "
                 "not applicable; window B' disclosed; cross-shape claim rule). Written before any data of this "
                 "T14 exists; the collection refuses to start unless this file matches its sha256 sidecar and the "
                 "bound keys (calibration_t14.check_preregistration)."),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "written_by": args.written_by,
        "design_doc": "local workspace docs/calib-next-round-design-20261005.md (user decisions 2026-10-05, item 4)",
        "code": {**code, "modules": ["deploy/scripts/calibration_t14.py", "deploy/scripts/t14_prereg.py",
                                     "deploy/scripts/analysis/t14_score.py (rule v2)",
                                     "deploy/scripts/dline_refit.py (accept A, B' disclosure)",
                                     "deploy/scripts/b_prime.py (FA, B' disclosure)"]},
        "t14": {
            "model": t14.MODEL, "api": "chat",
            "shapes": {"interpolation": interp, "extrapolation": extrap},
            "factors": list(t14.FACTORS), "hold_s": t14.HOLD_S, "warmup_dropped_s": design.WARMUP_S,
            "cells_total": len(rows),
            "design_seed": args.design_seed, "cell_serial_base": args.cell_serial_base,
            "prompt_key_prefix": f"p{args.design_seed}.",
            "arrival_seed_rule": f"calibration_design.derived_seed({args.design_seed}, '{t14.MODEL}', cell_id, 'arrivals')",
            "order_rule": f"shapes interleaved, derived_seed({args.design_seed}, '{t14.MODEL}', 't14-order')",
            "cells": rows,
            "seed_disjointness": disjoint,
            "capacity_prior": {
                "path": prior["path"], "sha256": prior["sha256"], "form": prior["form"],
                "coefficients": prior["coefficients"],
                "fit_points": prior.get("fit_points"),
                "boundary_table": ({"path": str(Path(args.boundary_table).resolve()),
                                    "sha256": _sha(Path(args.boundary_table))} if args.boundary_table else None),
                "leave_one_out": prior.get("leave_one_out"), "predicted_rps": prior["predicted_rps"],
                "decision": args.capacity_prior_decision,
            },
            "held_out_assertion": "calibration_t14 refuses unless gen.is_held_out(shape) and split == holdout for every cell",
            "max_model_len_check": mml,
            "gateway": {"url": args.gateway_url, "headers": {"model": t14.MODEL, "routing-strategy": t14.ROUTING_STRATEGY}},
            "engine_image": args.engine_image,
            "schedule": ("in the same exclusive window, right after dsqwen-14b's M2 (and its one-shot acceptance "
                         "inputs sealed); the other models' M2 may still run on other GPUs; no other 14b load"),
            "conditions": {
                "controller_mode": "observe", "sm_mode": "observe", "routable_replicas": 1,
                "label": ("label v2: D6' primary (slowdown TTFT k=5 floor 500 ms over the 2026-10-03 idle c/b, TPOT "
                          "75 ms, min 20 requests, unserved = violated) with HYBRID attribution: TTFT samples (and "
                          "the min-n count) in the window of the first token, TPOT / e2e in the window of completion, "
                          "unserved requests in the window of sending"),
                "label_def_name": label_name,
                "label_def_sha256": label_sha,
                "label_attribution": ATTRIBUTION,
                "dataset": "<T14 root>/<model>/dataset_hybrid (calibration_dataset --attribution hybrid)",
                "windows": "30 s windows on the 10 s grid, first 60 s of each cell dropped, ramp trim 1 (as the training fit)",
            },
            "no_adaptive_points": True,
            "void_rule": VOID_RULE,
            "stream_cut": {
                "why": ("the tre-v2 route total timeout (150 s) cuts streams mid-decode under overload; the client "
                        "records a model_error with e2e about 151 s (2026-10-03 round: 7b P1 voided twice on it)"),
                "runtime_limit": {"max_model_error_rate": 0.10, "default_was": 0.05,
                                  "how": ("calibration_campaign --max-model-error-rate 0.10: run_T14.sh reads this key "
                                          "and passes it; calibration_t14.check_preregistration refuses another value"),
                                  "unchanged": ("every other guard (shed void, proxy transient budget, 50 ms lateness "
                                                "p99, reissue contamination, drain gate, void re-drive once then stop)")},
                "audit_rule": {
                    "cut": ("a request with outcome model_error and e2e_ms >= 150000 (the route timeout) is a "
                            "route-timeout cut: censored, not a model_error"),
                    "rule": ("every model_error in an accepted T14 cell must be a route-timeout cut; any other "
                             "model_error is a real engine error, reported per cell and judged against the original "
                             "0.05 limit (a cell above it is void at audit, D3)"),
                    "label": ("the SLO label is unchanged: a cut request stays an unserved request (violated window); "
                              "the rule only changes the void audit and the error accounting"),
                    "report": "per cell: model_error count, cuts, non-cut errors, cut share of sent"},
                "status": "a preregistered rule of this document (was an addendum in the 2026-10-03 round)",
            },
            "forbidden_roots": roots,
            "dev_roots": {"roots": list(args.dev_root),
                          "note": ("opened in an earlier round (the 2026-10-03 T14 and M): DEV only - never evidence "
                                   "for this T14, never pooled with it, and the output must stay out of them")},
            "expected_wall_clock_h": {"expected": round(est["seconds_expected"] / 3600, 2),
                                      "upper": round(est["seconds_upper"] / 3600, 2),
                                      "estimate": est, "cooldown_s": args.cooldown_s,
                                      "reference": ("the 2026-10-03 T14 (same design) ran 2.18 h against 2.15 "
                                                    "expected / 2.88 upper")},
        },
        "parameter_sets": {
            "freeze": freeze_block,
            "v1lambda": {**{k: freeze_block[k] for k in ("path", "sha256", "freeze_sha256")},
                         "note": ("ONE frozen set: the campaign needs two parameter files, so the freeze is named "
                                  "twice. There is no second set and no replacement rule.")},
        },
        "evaluation": {
            "times": "once on T14, with the freeze parameters; nothing is tuned on T14",
            "data": ("the sealed T14 dataset (T14_manifest.json cells only, the valid attempt), dsqwen-14b, "
                     "gateway numerator, hybrid attribution: <T14 root>/dsqwen-14b/dataset_hybrid"),
            "label_attribution": {"value": ATTRIBUTION,
                                  "rule": ("the scorer reads the attribution from the frozen label, the dataset "
                                           "manifest, the T14 manifest's label and this document, and refuses unless "
                                           "they all agree - never mixes attributions")},
            "A": {"metric": "window balanced accuracy at the published theta (no dwell)",
                  "ci": f"cell bootstrap, {SCORE_RESAMPLES} resamples, seed {SCORE_SEED}, 95 % percentile",
                  "gates_reported": {"ba_min": 0.80, "ba_ci95_low_min": 0.75, "max_drop_from_training_ba": 0.08},
                  "role": "gate"},
            "FA": {"metric": "CRITICAL false alarm on healthy windows at dwell 1 (the controller's TRE_DWELL_WINDOWS)",
                   "ci": f"cell bootstrap, {SCORE_RESAMPLES} resamples, seed {SCORE_SEED}, 95 % percentile",
                   "gate": {"false_alarm_max": 0.05, "false_alarm_ci95_high_max": 0.08}, "dwell_windows": 1,
                   "role": "gate"},
            "onset_episodes": {"status": "not_applicable",
                               "why": ("T14 is 24 steady holds and has no dynamic cell: there is no overload onset "
                                       "to catch. The onset-episode gate is judged on M2's dynamic cells only.")},
            "D": {"status": "not in the T14 rule",
                  "why": "theta CI half width is a property of the training fit, judged at freeze time"},
            "B_prime_disclosure": {"role": "disclosed, not gating",
                                   "cut": "the freeze's sealed b_prime.severity_cut (new-label training .65 quantile)",
                                   f"{t14.MODEL}_cut": cut,
                                   "dwell": "dwell 1 (and dwell 2) disclosed"},
            "cross_shape": {
                "what": ("fixed Z = 1 (the published theta of the freeze): BA and AUROC per T14 shape, interpolation "
                         "and extrapolation in separate tables; per-shape BA CI95 by ci_method (moving-block "
                         "bootstrap over windows, blocks within a cell - user 2026-10-05: the 2026-10-03 cell "
                         "bootstrap degenerates with 3 cells per shape); AUROC CI as ranking_disclosure"),
                "claim_rule": CROSS_SHAPE_CLAIM_RULE,
                "ci_method": dict(t14_score.CROSS_SHAPE_CI_METHOD),
                "yardstick": dict(t14_score.CROSS_SHAPE_YARDSTICK),
                "ci_notes": ("history (see dev_disclosure): the 2026-10-03 per-shape CI (the accept's cell bootstrap) "
                             "degenerates with 3 cells per shape (half width 0); the defect was noticed while scoring "
                             "the 2026-10-03 T14 as DEV under the hybrid label, and the CI method was then changed to "
                             "the moving-block bootstrap on design grounds (the claim rule is unchanged). The 6-window "
                             "block is conservative relative to dline_refit.WINDOWS_PER_INDEPENDENT = 3 (30 s windows "
                             "on a 10 s step); the 'windows / 6' wording of the method docs came from the earlier 5 s "
                             "step. The yardstick refinement came from a TRAINING-only check (next-20261005/"
                             "t14_ci_check/: run2 / run2b holds, 3-cell subsets), not from DEV: a zero-width per-shape "
                             "CI at BA = 1 or .5 reflects perfect separation or a one-sided classification in 3 cells, "
                             "not zero sampling uncertainty, so the median half width is taken over the non-degenerate "
                             "shapes only (fewer than 2 in a kind: not_evaluable). No T14 cross-shape claim was "
                             "computed under the new CI method or yardstick."),
                "role": "claim rule for the text, sealed under the hybrid label; not an acceptance gate"},
            "scoring": {"decisions": DECISIONS, "disclosure_rules": DISCLOSURE_RULES,
                        "scorer": {"module": "scripts.analysis.t14_score (rule v2)",
                                   "commit": args.scorer_commit or PLACEHOLDER,
                                   "identity_rule": ("the output records the scorer path, sha256 and commit; a real "
                                                     "run refuses without a dry-run output of the same commit, "
                                                     "scorer sha256, preregistration and attribution")}},
            "also_reported": ["AUROC of Z vs SLO label", "Kendall tau-b ranking disclosure",
                              "C: TTFT-only recall (disclosure)", "old B (disclosure)",
                              "per-shape BA, interpolation and extrapolation separately"],
            "outcome_statements": {
                "pass": "A and FA gates all met (onset not applicable)",
                "fail": "reported as is",
                "pass_a_disclosed": ("if only A fails and FA passes, theta may still go online and A is disclosed as a "
                                "limitation (user decision 2026-10-05 item 2: urgent scale-up and GPU release act on "
                                "the CRIT / HIGH lines, not on Z = 1); same rule as M2, all three models"),
            },
        },
        "dev_disclosure": ("the 2026-10-03 T14 (same design, other seed) is DEV for this round. (1) It was opened and "
                           "scored under the completion label in its own round (pass). (2) It was scored again as DEV "
                           "under the hybrid label with this round's freeze (next-20261005/dev/dev_score.json): rule "
                           "v2 verdict pass_a_disclosed (A: BA .820, CI95 low .749 < .75; FA 0). (3) That DEV scoring "
                           "is where the defect of the per-shape CI was noticed: the cell bootstrap degenerates to a "
                           "half width of 0 with 3 cells per shape; the CI method was then changed to the moving-block "
                           "bootstrap on design grounds (evaluation.cross_shape.ci_method; claim rule unchanged). "
                           "(4) The yardstick refinement (median half width over non-degenerate shapes only) came "
                           "from a TRAINING-only check, not from DEV. (5) The cross-shape result of the 2026-10-03 T14 "
                           "under the new CI method and yardstick was deliberately NOT computed. Beyond these points "
                           "and keeping the design unchanged, the old T14 informed no choice in this document."),
        "known_risks": [
            "the capacity prior's leave-one-out of S1 is far off (the short-prompt S1 dominates the intercept); the 8 "
            "T14 predictions are all positive",
            "T14 violations were deep in the 2026-10-03 round (Z median .32): A on T14 says little about the knee "
            "region where 14b's A failed on M (DEV)",
        ],
    }
    return doc


def placeholders(doc: Any, path: str = "") -> list[str]:
    if isinstance(doc, dict):
        return [p for k, v in doc.items() for p in placeholders(v, f"{path}.{k}" if path else str(k))]
    if isinstance(doc, list):
        return [p for i, v in enumerate(doc) for p in placeholders(v, f"{path}[{i}]")]
    return [path] if isinstance(doc, str) and PLACEHOLDER in doc else []


def write(doc: Mapping[str, Any], out: Path, *, draft: bool) -> Optional[str]:
    if out.exists():
        raise ValueError(f"{out} exists: a preregistration is written once")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, indent=1) + "\n")
    if draft:
        return None
    h = _sha(out)
    side = Path(f"{out}.sha256")
    side.write_text(f"{h}  {out.name}\n", encoding="utf-8")
    for p in (out, side):
        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return h


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draft", action="store_true", help="placeholders allowed; marked DRAFT; no sidecar")
    ap.add_argument("--design-seed", type=int, required=True)
    ap.add_argument("--cell-serial-base", type=int, required=True)
    ap.add_argument("--capacity-prior", type=Path, required=True)
    ap.add_argument("--capacity-prior-decision", default="reused",
                    help="recorded as t14.capacity_prior.decision (why it is reused / rebuilt)")
    ap.add_argument("--boundary-table", type=Path, default=None)
    ap.add_argument("--freeze-file", type=Path, default=None)
    ap.add_argument("--label-def-sha256", default=None, help="without --freeze-file: the label v2 sha (else placeholder)")
    ap.add_argument("--scorer-commit", default=None)
    ap.add_argument("--gateway-url", required=True)
    ap.add_argument("--engine-image", required=True)
    ap.add_argument("--forbidden-root", action="append", default=[])
    ap.add_argument("--dev-root", action="append", default=[])
    ap.add_argument("--ledger-glob", action="append", default=[])
    ap.add_argument("--registry", default=None)
    ap.add_argument("--code-repo", type=Path, default=None)
    ap.add_argument("--cooldown-s", type=float, default=campaign.DEFAULT_COOLDOWN_S)
    ap.add_argument("--written-by", default="main session (Claude); owner review pending")
    args = ap.parse_args(argv)
    try:
        t14.cell_serial_base(argparse.Namespace(t14_cell_serial_base=args.cell_serial_base))
        if not args.ledger_glob or not args.forbidden_root:
            raise ValueError("--ledger-glob and --forbidden-root are required (seed / id disjointness, isolation)")
        doc = build(args)
        dj = doc["t14"]["seed_disjointness"]
        if dj["t14_arrival_seed_overlap"] or dj["t14_cell_id_overlap"]:
            raise ValueError(f"seed / cell id overlap with a ledger: {dj}")
        holes = placeholders(doc)
        if holes and not args.draft:
            raise ValueError(f"placeholders left in a final preregistration: {holes}")
        if doc["code"].get("dirty") and not args.draft:
            raise ValueError(f"code tree {doc['code']} is dirty: a final preregistration records a clean commit")
        h = write(doc, args.out, draft=args.draft)
    except ValueError as exc:
        ap.error(str(exc))
    print(f"wrote {args.out} ({'DRAFT, no sidecar' if args.draft else 'sha256 ' + str(h)}); "
          f"{len(doc['t14']['cells'])} cells, serials {args.cell_serial_base + 1}-{args.cell_serial_base + 24}, "
          f"seed {args.design_seed}; ledgers {dj['ledgers']} (seed overlap {len(dj['t14_arrival_seed_overlap'])}, "
          f"id overlap {len(dj['t14_cell_id_overlap'])}); wall clock {doc['t14']['expected_wall_clock_h']['expected']} h "
          f"expected, {doc['t14']['expected_wall_clock_h']['upper']} h upper; placeholders {placeholders(doc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

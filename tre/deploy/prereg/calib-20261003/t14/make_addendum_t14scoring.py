# T14 scoring addendum for the 2026-10-03 round (decisions D1-D5 + the void_rule interpretation), sealed before any M or T14 data was opened.
import hashlib, json, os, stat, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

C = Path("/data/nfs_shared_data/xxy/calib_20261003")
HP = Path("/data/nfs_shared_data/xxy/aibrix-wt/calib-h-prep-20261004")
TOOL_COMMIT = "1c2eb6ab336fee901bbe2f1f71164dcb135d98d7"
TOOL_FILES = {
    "tre/deploy/scripts/analysis/t14_score.py": "47cbb04a966fcbd02af778139d8e3d92d810b0edfebdd6622ba4793674941553",
    "tre/deploy/scripts/analysis/h_conservative_score.py": "e965ab66c497547bf66d760b2ecc086ebf767ab1f4d418e97aa5a9fcfef3d788",
    "tre/deploy/scripts/analysis/h_dropped_windows.py": "95e73d65cc5dc81715190eb3de94799fd3552f83d78ed63e27d768c2d008782a",
}
PREREG = C / "t14" / "preregistration.json"
FREEZE = C / "freeze" / "params_freeze.json"
CROSS = C / "prereg" / "ADDENDUM-T14-crossshape.json"
STREAM = C / "prereg" / "ADDENDUM-T14-streamcut.json"
VOID_RULE = ("a void cell is re-driven once; a second void stops the run; a stopped run is never evaluated - "
             "re-run all 24 cells into a new root")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_tools():
    git = ["git", "-c", "safe.directory=*", "-C", str(HP)]
    head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(git + ["status", "--short"], capture_output=True, text=True, check=True).stdout.strip()
    if head != TOOL_COMMIT or dirty:
        sys.exit(f"h-prep worktree at {head} (dirty: {bool(dirty)}), expected {TOOL_COMMIT} clean")
    for rel, want in TOOL_FILES.items():
        blob = subprocess.run(git + ["show", f"{TOOL_COMMIT}:{rel}"], capture_output=True, check=True).stdout
        if hashlib.sha256(blob).hexdigest() != want:
            sys.exit(f"{rel} at {TOOL_COMMIT} does not hash to {want}")


def collection_state():
    """From CHAIN_STATUS.jsonl only: nothing under M/ or T14/ is listed or read."""
    lines = [json.loads(l) for l in (C / "CHAIN_STATUS.jsonl").read_text().splitlines() if l.strip()]
    g = [f"{x['ts']} {x['model']} {x['step']} {x['event']} rc={x['rc']}" for x in lines if str(x.get("step", "")).startswith("G_")]
    return {"source": "CHAIN_STATUS.jsonl (no file under M/ or T14/ was listed, opened or read)", "G_lines": g,
            "t14_started": any("G_T14" in s and "start" in s for s in g)}


def write_once(out, doc):
    if out.exists():
        sys.exit(f"{out} exists: an addendum is written once")
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    h = sha(out)
    Path(str(out) + ".sha256").write_text(f"{h}  {out.name}\n")
    for p in (out, Path(str(out) + ".sha256")):
        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(h)

OUT = C / "prereg" / "ADDENDUM-T14-scoring.json"
check_tools()
state = collection_state()
now = datetime.now().astimezone().isoformat(timespec="seconds")
fz = json.load(open(FREEZE))
SCRATCH = "/data/nfs_shared_data/xxy/calib_20261003_review/h_prep_20261004"
doc = {
    "what": ("Addendum to the T14 preregistration of the 2026-10-03 round: how T14 is scored - the five scoring "
             "decisions D1-D5, the interpretation of the prereg void_rule, and the disclosure rules - with the scorer "
             "that implements them. Decided and sealed before any M or T14 data was opened (owner approval 2026-10-04)."),
    "written_at": now,
    "amends": {"path": str(PREREG), "sha256": sha(PREREG),
               "note": "adds scoring rules only; changes no bound key of the preregistration, no collection parameter and no gate"},
    "parameter_set": {"path": str(FREEZE), "sha256": sha(FREEZE), "freeze_sha256": fz["freeze_sha256"]},
    "sibling_addenda": {"crossshape": {"path": str(CROSS), "sha256": sha(CROSS)},
                        "streamcut": {"path": str(STREAM), "sha256": sha(STREAM)},
                        "h_dropped_windows": {"path": str(C / "prereg" / "ADDENDUM-H-dropped-windows.json"),
                                              "note": "sealed right after this one; it records this file's sha256"}},
    "collection_state_at_sealing": state,
    "void_rule_interpretation": {
        "void_rule_quoted": VOID_RULE + " (t14/preregistration.json t14.void_rule)",
        "primary_reading": ("per cell: 'a second void' is a second void of the SAME cell, as calibration_ladder.drive_cell / "
                            "adaptive_boundary.next_void_attempt count it. Voids of a cell = its void attempts at run time "
                            "(T14_manifest.json attempts with void_reasons) + 1 if it is void at audit (non-cut model "
                            "errors / sent > 0.05, stream-cut addendum). A cell with 2 voids -> status run_void (never "
                            "evaluated; re-run all 24 cells into a new root); else any cell void at audit -> status "
                            "void_redrive_required:<cell ids> (re-drive each once, then score); else evaluated. The "
                            "primary reading alone decides the status and the verdict."),
        "sensitivity_reading": ("run level: 2 or more voids anywhere in the run (2 voided cells, or one cell voided twice) "
                                "-> run_void; exactly one void and it is an audit void -> void_redrive_required:<cell>; else "
                                "evaluated. Disclosed with equal visibility; never decides."),
        "reporting": ("the T14 output and the report state the number of voided cells (voided_cells) and both statuses "
                      "side by side (void_status.void_status_primary, void_status.void_status_run_level_sensitivity)"),
        "rationale": ("the prereg text refers to the registered collection machinery, and T14 is collected under it; that "
                      "machinery counts a second void per cell"),
        "decided": f"2026-10-04 evening +08:00 (owner approval; sealed {now}), before any M or T14 data was opened",
    },
    "decisions": {
        "D1_cross_shape_sd": "the cross-shape claim uses the sample SD (n-1) of the per-shape BA; the population SD is disclosure only",
        "D2_single_class_shape": ("a shape whose windows are all one class has BA and AUROC undefined; the claim of its kind "
                                  "is 'not_evaluable'; the shape is never dropped to claim on the remaining shapes; the SD over "
                                  "the remaining shapes is disclosure only; the pooled A (all windows) is unaffected"),
        "D3_void_at_audit": "see void_rule_interpretation (primary per cell decides; run-level sensitivity disclosed)",
        "D4_model_error_without_e2e": "a model_error without e2e_ms is non-cut",
        "D5_training_ba": "A's max-drop-from-training check uses the freeze's pooled train_ba_at_published (as accept)",
        "no_verdict_output": "without status 'evaluated' there is no verdict; every metric is written under 'disclosure_not_an_evaluation'",
    },
    "disclosure_rules": {
        "audit_scope": ("the censoring audit counts the evaluated (valid) attempt of each manifest cell only; voided earlier "
                        "attempts are excluded; every request of that attempt counts, warm-up INCLUDED - the prereg does not "
                        "name warm-up, and the stream-cut addendum judges 'every model_error in an accepted T14 cell' "
                        "against openloop.check_cell's whole-cell model_errors / sent"),
        "zero_token_windows": ("the T14 validation CSV's dropped zero-token windows (with backlog / failure evidence / "
                               "unserved-violated) and the conservative variant (those windows counted as non-CRITICAL misses) "
                               "are reported as disclosure (ADDENDUM-H-dropped-windows)"),
        "per_shape_auroc": "a single-class shape has AUROC undefined (D2)",
        "fig_2_1_kv": ("KV usage comes from the 1 Hz instant sidecar (<cell>.instant.jsonl, kv_cache_usage), which falls "
                       "behind under overload: per cell the share of missing samples against the nominal 1000 ms step "
                       "(whole cell and after the 60 s warm-up) and the largest gap are disclosed; the ~1017 ms real cadence "
                       "alone shows as about 2 % missing. The Fig 2.1 drawing and the lambda disclosure are not in the scorer"),
        "b_prime_unit": ("B' recall_severe at dwell 1 is a rate over WINDOWS (b_prime.series_point: severe violating windows "
                         "CRITICAL at dwell 1 / severe violating windows), not over episodes; its CI resamples cells"),
    },
    "scorer": {
        "path": "tre/deploy/scripts/analysis/t14_score.py (module scripts.analysis.t14_score)",
        "branch": "calib/h-prep-20261004", "commit": TOOL_COMMIT, "files_sha256": TOOL_FILES,
        "identity_rule": ("the T14 output records the scorer path, its sha256 and the commit; dry run on the frozen / "
                          "training set first: a real run refuses unless --dry-run-result is a dry-run output of the same "
                          "commit and scorer sha256 from a clean tree"),
        "sha_rule": "this commit is referenced by a sealed record: never rebase, squash or cherry-pick it; merge into main by fast-forward or merge commit",
        "dry_runs": {
            "run2b_14b": {"path": f"{SCRATCH}/t14_score.v4.dryrun_run2b_14b.json", "sha256": sha(f"{SCRATCH}/t14_score.v4.dryrun_run2b_14b.json")},
            "p1_14b": {"path": f"{SCRATCH}/t14_score.v4.dryrun_p1_14b.json", "sha256": sha(f"{SCRATCH}/t14_score.v4.dryrun_p1_14b.json")},
        },
        "command": ("cd <h-prep worktree>/tre/deploy; python3 -m scripts.analysis.t14_score --prereg $C/t14/preregistration.json "
                    "--addendum $C/prereg/ADDENDUM-T14-streamcut.json --addendum $C/prereg/ADDENDUM-T14-crossshape.json "
                    "--freeze-file $C/freeze/params_freeze.json --t14-manifest $C/T14/dsqwen-14b/T14_manifest.json "
                    "--dataset $C/T14/dsqwen-14b/dataset --dry-run-result $C/eval/T14_score.dryrun_run2b.json "
                    "--out $C/eval/T14_score.json (RUN plan H4, after the dry run of the same commit)"),
    },
    "does_not_change_acceptance": "the T14 gates (A, B' at dwell 1) and RUN plan H1 (accept on M) are unchanged",
}
write_once(OUT, doc)

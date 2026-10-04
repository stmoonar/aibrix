# H dropped-window disclosure addendum for the 2026-10-03 round, sealed before any M or T14 data was opened.
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

BODY = '{"what": "Addendum to the 2026-10-03 calibration preregistration and the T14 preregistration: disclosure of the windows the zero-token rule drops, with a conservative variant reported next to the frozen acceptance metrics on M (RUN plan H3) and T14 (RUN plan H4). Decided and sealed before any M or T14 data was opened (owner approval 2026-10-04).", "amends": [{"path": "tre/docs/preregistration-20261003-calibration-v030.md", "note": "the round\'s preregistration (draft on the calib branch); acceptance section"}, {"path": "/data/nfs_shared_data/xxy/calib_20261003/t14/preregistration.json", "sha256": "59608505fc055bc2d36cbb64aadb8038112f32c4eb0cae20aed42fbccb1daf30"}], "parameter_set": {"path": "/data/nfs_shared_data/xxy/calib_20261003/freeze/params_freeze.json", "sha256": "eb2d59e10a84bb3563e7872cb5252b1142855f662b0549c9172efeafe322e531", "freeze_sha256": "ca7fe45235da713c4faf8d8b8d766343dd96405439eeb3fa185cc80bac5a1f60"}, "drop_rule": {"code": "tre/calibration/tre_calibration/dataset.py calibration_window_from_row (commit da0ff2cc, the freeze\'s code): `if prompt_tokens + generation_tokens <= 0.0: return None`, before label_window", "effect": "With the gateway numerator the window token totals count completed requests only. A window in which the engine completed nothing (queue > 0; requests timing out or cut at the 150 s route timeout) has 0 tokens and is dropped before it is labelled, although its unserved requests make it a violation (class unserved). theta is not biased (the windows are far from the boundary). The acceptance metrics (A, B\', all-violation recall) are computed without these windows and are therefore optimistic. No count of them was reported before.", "online_semantics": "The controller sees the same window as idle: tre_common.tss.window_is_idle (no token) resets the EMA and the dwell runs, and TSS 0 maps to Z = None (compute_z_m). Such a window is never CRITICAL online.", "training_extent_known_before_M": {"source": "h_dropped_windows audit on the training datasets (run1, run2 / run2b for 14b, supp, p1 / p1r2 for 7b), produced at commit 84843e08 (the counting logic is unchanged at the sealed tool commit; it only gained a read_holdout switch); output sha256 e379ecbb1ae747290a26d036d32a86777b638e08cc9dc4cf7914e6c44aef0e03", "dsqwen-7b": {"dropped_zero_token": 41, "with_backlog": 41, "with_failure_evidence": 34, "unserved_violated": 34}, "dsllama-8b": {"dropped_zero_token": 44, "with_backlog": 44, "with_failure_evidence": 41, "unserved_violated": 41}, "dsqwen-14b": {"dropped_zero_token": 37, "with_backlog": 37, "with_failure_evidence": 32, "unserved_violated": 32}, "where": "all in the P1 deep-overload cells (p1 / p1r2); none in run1, run2 / run2b or supp"}}, "gate_unchanged": "The acceptance gate is unchanged: dline_refit accept (A, B\' at dwell 1 with the freeze\'s sealed cuts and gate, D) on M, run once; T14 by its preregistered rule (A and B\' at dwell 1). The conservative variant never gates, tunes, selects or replaces anything; it is reported side by side.", "conservative_variant": {"variants": {"frozen": "the windows the loader keeps (= the accept numbers; checked equal to dline_refit.evaluate_model)", "conservative_failure": "frozen + every dropped zero-token window with failure evidence (model_errors / proxy_transient_errors / client_timeouts > 0; the frozen label calls it violated, class unserved)", "conservative": "frozen + every dropped zero-token window with a backlog (avg_running + avg_waiting > 0) or failure evidence"}, "added_window": "violating, never CRITICAL: Z set to 1e6 (finite, >= 1: predicted healthy for BA, outside the CRITICAL and LOW bands) - the stand-in for the controller\'s Z = None", "label_of_added_window": "the frozen label\'s verdict when violated (ratio UNSERVED_MIN_RATIO = 2.0 without latency samples); a backlog-only window (no unserved request, label None by min-n) is counted violated with the same no-sample convention (class unserved, ratio 2.0)", "ramp_trim": "a dropped window among the first trim_ramp_windows (1) non-filtered rows of its cell is not added (onset window); the frozen windows are never changed", "warmup": "rows in the 60 s warm-up stay excluded (in_warmup)", "b_prime_note": "an added window has severity 2.0, below every sealed B\' cut (7b 7.85, 8b 12.60, 14b 8.81), so B\' is unchanged by construction; the variant moves A (BA) and the all-violation recall", "bootstrap": "same as accept: cell bootstrap, 1000 resamples, seed 20260922; B\' at dwell 1, dwell 2 disclosed"}, "reported_side_by_side": ["windows and violating windows per variant", "A: BA at the published theta, its CI95 lower bound, A pass/fail", "B\' recall of severe violations and false alarm with CI95 (dwell 1), pass/fail", "all-violation CRITICAL recall at dwell 1 and dwell 2", "old B recall (dwell 2)", "violations by band", "the added counts per tier"], "commands": {"M": "cd $HP/deploy && python3 -m scripts.analysis.h_conservative_score --freeze-file $C/freeze/params_freeze.json --csv dsqwen-7b=$C/freeze/params_freeze.accept.d/dsqwen-7b_validation.csv --csv dsllama-8b=$C/freeze/params_freeze.accept.d/dsllama-8b_validation.csv --csv dsqwen-14b=$C/freeze/params_freeze.accept.d/dsqwen-14b_validation.csv --label \'H3 on M (accept validation CSVs)\' --out $C/eval/H_conservative_M.json  (RUN plan H3, after H1)", "T14": "the conservative_disclosure and zero_token_dropped_windows blocks of scripts.analysis.t14_score (RUN plan H4; ADDENDUM-T14-scoring)", "HP": "/data/nfs_shared_data/xxy/aibrix-wt/calib-h-prep-20261004/tre", "C": "/data/nfs_shared_data/xxy/calib_20261003"}, "code": {"branch": "calib/h-prep-20261004", "commit": "84843e08c2f148209ca26a33ce2af905078012b1", "files_sha256": {"deploy/scripts/analysis/h_dropped_windows.py": "0d0fd25f8bb9d6ecadb3ff4e483509c9585b5e9c73ba5afc0f3e105b922397d7", "deploy/scripts/analysis/h_conservative_score.py": "e965ab66c497547bf66d760b2ecc086ebf767ab1f4d418e97aa5a9fcfef3d788", "deploy/scripts/analysis/t14_score.py": "0509bb07043900c0f9abca9f371870acaa508336425736167f30537701877225"}, "sha_rule": "once registered, this commit is referenced by data records: merge it into main by fast-forward or merge commit only (no rebase / squash / cherry-pick); if the owner changes the tools, a new commit and new hashes go into this file before it is registered"}, "dry_run_on_training": {"output": "/data/nfs_shared_data/xxy/calib_20261003_review/h_prep_20261004/conservative_score.dryrun_training.json", "sha256": "813a9cd4b1aaea82bbd62ba53ab170beff3028b2da31f5586b39b069950abb80", "BA_frozen_vs_conservative": {"dsqwen-7b": [0.8541, 0.8407], "dsllama-8b": [0.8486, 0.8326], "dsqwen-14b": [0.8405, 0.8248]}, "note": "training fitting CSVs only; frozen BA equals the freeze\'s train_ba_at_published exactly; produced at commit 84843e08 - h_conservative_score.py is byte-identical at the sealed tool commit"}}'
OUT = C / "prereg" / "ADDENDUM-H-dropped-windows.json"
SCORING = C / "prereg" / "ADDENDUM-T14-scoring.json"
if not SCORING.exists():
    sys.exit(f"{SCORING} is sealed first")
check_tools()
now = datetime.now().astimezone().isoformat(timespec="seconds")
doc = json.loads(BODY)
doc = {"what": doc.pop("what"), "written_at": now, **doc}
doc["amends"][1]["sha256"] = sha(PREREG)
doc["parameter_set"] = {"path": str(FREEZE), "sha256": sha(FREEZE), "freeze_sha256": json.load(open(FREEZE))["freeze_sha256"]}
doc["sibling_addenda"] = {"crossshape": {"path": str(CROSS), "sha256": sha(CROSS)},
                          "streamcut": {"path": str(STREAM), "sha256": sha(STREAM)},
                          "t14_scoring": {"path": str(SCORING), "sha256": sha(SCORING)}}
doc["collection_state_at_sealing"] = collection_state()
doc["void_rule_interpretation"] = {
    "void_rule_quoted": VOID_RULE + " (t14/preregistration.json t14.void_rule)",
    "primary_reading": "per cell, as calibration_ladder.drive_cell counts a second void; decides the T14 status and verdict",
    "sensitivity_reading": "run level: 2 or more voids anywhere in the run stop it; disclosed with equal visibility, never decides",
    "rationale": "the prereg text refers to the registered collection machinery, and T14 is collected under it",
    "decided": f"2026-10-04 evening +08:00 (owner approval; sealed {now}), before any M or T14 data was opened",
    "scope_here": ("the conservative variant does not depend on it; on M, accept has no audit-time void rule (M's "
                   "run-time voids follow the same per-cell machinery). Full rule: ADDENDUM-T14-scoring.json"),
}
doc["code"] = {"branch": "calib/h-prep-20261004", "commit": TOOL_COMMIT, "files_sha256": TOOL_FILES,
               "sha_rule": "this commit is referenced by a sealed record: never rebase, squash or cherry-pick it; merge into main by fast-forward or merge commit"}
write_once(OUT, doc)

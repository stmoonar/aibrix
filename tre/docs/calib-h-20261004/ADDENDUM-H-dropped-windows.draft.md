# Addendum (DRAFT): windows the zero-token rule drops

Status: **draft, not registered.** Register it (copy to `$CALIB_ROOT/prereg/ADDENDUM-H-dropped-windows.json` + sha256 sidecar) **before** RUN plan H1 (accept) reads M. The JSON next to this file is the text to register.

## The rule

- `calibration_window_from_row` (the single row-to-window rule of every fit and of accept) drops a window with `prompt_tokens_total + generation_tokens_total <= 0` before it is labelled.
- The gateway numerator counts only completed requests. So a fully stalled window (queue > 0, nothing completed, requests timing out or cut at 150 s) has 0 tokens and is dropped, although it is a violation.
- θ is not biased. The accept metrics (A, B′, all-violation recall) are optimistic.
- Online the controller sees the same window as idle (EMA and dwell reset, Z = None): never CRITICAL.

## What changes

- **The gate does not change.** Accept (A, B′ at dwell 1, D) on M, and the preregistered T14 rule, are the verdicts.
- A conservative variant is reported next to them. It adds the dropped zero-token windows with a backlog or failure evidence as violating, non-CRITICAL windows (two tiers: failure evidence only; backlog or failure).
- B′ cannot move: the added windows have severity 2.0, below every sealed cut (7.85 / 12.60 / 8.81). The variant moves BA and the all-violation recall.

## Known before M (training data only)

| model | dropped (zero tokens) | with backlog | with failure evidence | unserved-violated | training BA frozen → conservative |
|---|---|---|---|---|---|
| dsqwen-7b | 41 | 41 | 34 | 34 | .8541 → .8407 |
| dsllama-8b | 44 | 44 | 41 | 41 | .8486 → .8326 |
| dsqwen-14b | 37 | 37 | 32 | 32 | .8405 → .8248 |

All of them are in the P1 deep-overload cells.

## How it is produced

- Code: branch `calib/h-prep-20261004`, commit `d947da76` (file hashes in the JSON).
- M: RUN plan H3 (`scripts.analysis.h_conservative_score` on accept's validation CSVs), after H1.
- T14: the `conservative_disclosure` block of `scripts.analysis.t14_score` (RUN plan H4).

## T14 scoring decisions (2026-10-04)

Prereg void_rule, quoted: "a void cell is re-driven once; a second void stops the run; a stopped run is never evaluated - re-run all 24 cells into a new root".

1. **Cross-shape SD:** the claim uses the sample SD (n−1). The population SD is disclosure only.
2. **Single-class shape:** BA and AUROC are undefined, and the claim of its kind is `not_evaluable`. The shape is never dropped. The SD over the other shapes is disclosure only. The pooled A is unaffected.
3. **Void at audit:** the void_rule is applied per cell.
   - A void on attempt 1 gives `void_redrive_required:<cells>`: re-drive the cell once, then score.
   - A void on a re-driven attempt (its second void) gives `run_void`: re-run all 24 cells.
   - In both cases there is no verdict; the metrics go under `disclosure_not_an_evaluation`.
   - **Open point for the owner:** "two or more void cells → run stopped" is not the per-cell reading. It is reported only, as `status_if_two_void_cells_stop_the_run`.
4. **model_error without e2e_ms:** counts as non-cut.
5. **Training BA for A:** the pooled `train_ba_at_published`.
6. **Audit scope:** the valid attempt only (voided attempts excluded), warm-up included (the whole-cell `model_errors / sent` of `check_cell`).
7. **Zero-token windows in T14:** the drop counts and the conservative variant are reported as disclosure.
8. **Per-shape AUROC:** follows rule 2.
9. **Fig 2.1 KV:** the share of missing 1 Hz sidecar samples per cell, plus the largest gap. In the dry runs the maximum was 19 % on run2b and 41 % under P1 overload.
10. **B′ unit:** `recall_severe` is counted per window, not per episode.
11. **Scorer identity:** the output records the path, sha256 and commit. The real run requires a dry-run output of the same commit.

Code: commit `d947da76`; the file hashes are in the JSON.

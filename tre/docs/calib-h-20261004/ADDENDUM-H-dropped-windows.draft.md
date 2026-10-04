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

- Code: branch `calib/h-prep-20261004`, commit `84843e08` (file hashes in the JSON).
- M: RUN plan H3 (`scripts.analysis.h_conservative_score` on accept's validation CSVs), after H1.
- T14: the `conservative_disclosure` block of `scripts.analysis.t14_score` (RUN plan H4).

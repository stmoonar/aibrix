# TRE v2 Architect Handoff - 2026-07-05

This note is the current handoff snapshot for architect review. The authoritative
workspace is `/data/nfs_shared_data/xxy/aibrix` on `A100_76`.

## Source Of Truth

- Active execution plan: `docs/refactor/14_endgame_plan.md`.
- Running log: `docs/refactor/WORKLOG.md`.
- Evidence tree: `docs/refactor/p11_evidence/`.
- This handoff file is intentionally a current-status summary. Some newest
  evidence below is still uncommitted because D8 hygiene is in progress.

## Completed And Committed

1. F1 full topology restore and live smoke are complete.
   - Commit: `4b692fbc [Endgame] restore full F1 topology evidence`.
   - Final verified topology at that point: `dsqwen-7b`, `dsllama-8b`,
     `dsqwen-14b` each `awake=1`, `bound=4`.
   - Reconcile returned `warnings=[]`; gateway smoke passed 20/20 per model.
   - Evidence: `docs/refactor/p11_evidence/f1_restore_20260705/`.

2. F1 GPU truth and D8/D10 service-manager guardrails are implemented and rolled.
   - Commits:
     - `ba88b1b0 [Endgame] add GPU truth headroom checks`.
     - `1ab54ede [Endgame] roll GPU truth service manager`.
   - `RedisGpuTruth` reads `tre:gpu_truth:<node>` from tre-v2 Redis.
   - GPU truth agents run on node9/node10 and write local logs under `/tmp`.
   - Reconcile emits `sleep_leak:<serve_id>` for sleeping-only GPUs over threshold.
   - Create paths reject GPUs above `TRE_CREATE_MAX_USED_MIB` before Deployment create.
   - Synthetic Redis truth injection produced expected leak warnings and recovered
     after agent refresh.
   - Evidence: `docs/refactor/p11_evidence/f1_gpu_truth_20260705/`.

3. F2.1 metrics baseline lookback is complete.
   - Commit: `3e5fb36b [Endgame] add metrics baseline lookback`.
   - Production AIBrix Redis histogram interval p95 was measured at `5000 ms`.
   - Selected controller histogram lookback: `90000 ms`.
   - Token fields are now `float | None`; no in-window histogram doc yields
     `None`, not zero.
   - Full verification at commit: `cd tre && make check` -> `250 passed`.

4. F2.2 stale paper-state hold is complete.
   - Commit: `a7694d30 [Endgame] hold stale paper state`.
   - Added `PaperStateCache` and per-task persistent caches.
   - Token-missing windows can hold the last valid per-model paper state for a
     bounded number of windows.
   - Events: `paper_state_stale_hold:<model>` and
     `paper_state_stale_unknown:<model>`.
   - Config: `TRE_PAPER_STALE_MAX_WINDOWS`.
   - Full verification: `cd tre && make check` -> `253 passed`.

5. F2.3 per-model incomplete drop policy is complete.
   - Commit: `7bfb0709 [Endgame] drop incomplete paper state by model`.
   - Default policy changed to `drop_model`; compatibility policy `drop_all`
     remains available.
   - Config: `TRE_INCOMPLETE_POLICY`.
   - Events default to `paper_state_incomplete_drop:<model>`.
   - Full verification: `cd tre && make check` -> `256 passed`.

6. F2.4 live Z-state observability and 15-minute precheck are complete.
   - Commits:
     - `987672ce [Endgame] pin controller image and precheck script`.
     - `699bbb6d [Endgame] fix precheck result output`.
     - `bb37a230 [Endgame] expose model z state in decisions`.
     - `b04d8810 [Endgame] roll controller decision state image`.
     - `0a8a5c28 [Endgame] keep precheck models active for z evidence`.
     - `0554391c [Endgame] validate zm precheck with model states`.
   - Controller decision snapshots now include per-model `model_states` with
     `z_m`, `trs_z_m`, `signal_source`, and `signal_unavailable_reason`.
   - Final precheck command:
     `python3 tre/deploy/scripts/n4b_three_model_precheck.py --duration-seconds 900 --phase-seconds 60 --workers 4 --baseline-workers-per-model 1 --sample-seconds 30 --max-tokens 96`.
   - Final result: duration `900.9s`, gateway errors `{}`, restarts `0`,
     reconcile `warnings=[]`.
   - Z-state evidence after warm-up stayed populated for all three models;
     each model had `255/259` non-null samples, with the nulls only during
     initial warm-up.
   - Evidence: `docs/refactor/p11_evidence/f2_zm_precheck_20260705/`.

7. F2.5 ActionQueue inflight bug is fixed and controller image is rolled.
   - Commits:
     - `d795a715 [Endgame] release inflight actions after dispatch failure`.
     - `0907a89c [Endgame] roll controller inflight fix image`.
   - Root cause of first high-load no-scale run:
     `ActionQueue.drain_once()` cleared `_inflight` only on successful dispatch.
     A failed service-manager dispatch could leave a model permanently inflight,
     so planner skipped it even while `Z_m` was critical.
   - Fix: `_inflight` is released after every dispatch attempt; failure is still
     recorded in `DispatchResult`.
   - Verification before roll: full `cd tre && make check` -> `260 passed`.
   - Controller image: `tre-v2-controller:20260705-d795a715`, image id
     `sha256:6b722a12a4aadb01dd3b485d5d537196deb337c0d4ebd7d63b54269b5eb118d3`.

## Latest Uncommitted Evidence

After rolling the inflight fix, F2.5 high-load validation was rerun:

- Command:
  `python3 tre/deploy/scripts/n4b_three_model_precheck.py --models dsqwen-7b --duration-seconds 300 --phase-seconds 300 --workers 16 --sample-seconds 15 --max-tokens 96`.
- Output file:
  `docs/refactor/p11_evidence/f2_zm_precheck_20260705/dsqwen7b_highload_after_inflight_fix.json`.
- Controller log:
  `docs/refactor/p11_evidence/f2_zm_precheck_20260705/controller_since_dsqwen7b_highload_after_inflight_fix.log`.
- Post-run reconcile:
  `docs/refactor/p11_evidence/f2_zm_precheck_20260705/post_dsqwen7b_highload_after_inflight_fix_reconcile.json`.

Observed result:

- Duration `301.6s`.
- `dsqwen-7b` ok requests: `3808`.
- Gateway errors: one HTTP `503`.
- Component restarts: `0`.
- `dsqwen-7b` final state grew from `awake=1` to `awake=3`, `bound=4`.
- `dsqwen-7b` decision `z_m`: count `81`, min `0.539`, p50 `0.700`, max `0.703`.
- Controller emitted `54` scale actions, all for
  `ScaleAction dsqwen-7b delta=1 reason=critical_sleeping_capacity` from rescue.

Conclusion: the F2.5 scale-action path now works. Critical `Z_m` produced scale
commands and additional `dsqwen-7b` replicas woke. The remaining issue is not
planner action generation; it is post-run GPU hygiene after scaling back down.

## Current Live State

As of the latest check:

- `tre-v2-service-manager` is healthy on image
  `tre-v2-service-manager:20260705-ba88b1b0`.
- `tre-v2-controller` is back at replicas `1` after hygiene. Its pinned image is
  `tre-v2-controller:20260705-d795a715`; current pod is
  `tre-v2-controller-65bb7cdf46-7tnmp`.
- Reconcile currently returns `warnings=[]`.
- Full topology is restored:
  - `dsqwen-7b`: `awake=1`, `bound=4`.
  - `dsllama-8b`: `awake=1`, `bound=4`.
  - `dsqwen-14b`: `awake=1`, `bound=4`.
- `dsqwen-7b` node9 GPU2/GPU3 and `dsqwen-14b` node9 GPU2-3 were recreated and
  all report `is_sleeping=true`.

Node9 GPU memory after completed hygiene:

- GPU0/GPU1 remain high and expected because they host the intentionally awake
  qwen/llama replicas and sleeping 14B node9 GPU0-1 shards.
- GPU2/GPU3 are now around `4070 MiB` each, matching three sleeping processes
  per GPU: llama about `1090 MiB`, qwen about `1054 MiB`, and 14B shard about
  `1766 MiB`.

## Current Problem / Architect Review Target

High-load scale-up worked, but when reducing `dsqwen-7b` back to target
`wake_replicas=1`, the sleeping `dsqwen-7b` replicas on node9 GPU2/GPU3 leaked
about `36.9 GiB` each. D8/manual node9 GPU truth inspection confirmed a true
sleep leak, not stale Redis truth.

Manual hygiene path executed:

1. Pause controller: `kubectl -n tre-v2 scale deploy/tre-v2-controller --replicas=0`.
2. Save pre-delete Deployment/reconcile evidence under
   `docs/refactor/p11_evidence/f2_hygiene_after_highload_20260705/`.
3. Delete leaked `dsqwen-7b` GPU2/GPU3 Deployments and the co-resident
   `dsqwen-14b` node9 GPU2-3 Deployment.
4. Recreate them one by one, wait for HTTP readiness, reconcile, set target
   `wake_replicas=1`, verify `/is_sleeping` and node9 GPU memory.
5. Save post-hygiene evidence, restore full `awake=1/bound=4` topology, then
   resume controller.

Evidence now present:

- `post_hygiene_reconcile.json`: `warnings=[]`.
- `post_hygiene_state.json`: all three models `awake=1`, `bound=4`.
- `post_hygiene_sleep_probes.json`: all recreated pods sleeping.
- `node9_post_hygiene_nvidia.txt`: GPU2/GPU3 both explainable at about
  `4070 MiB`.

## Architecture Questions

1. Should D8 sleep-leak remediation remain an operator-run hygiene playbook, or
   should service-manager/controller get an explicit automated recreate path for
   sleeping endpoints whose GPU truth remains above threshold?

2. Is the observed vLLM sleep leak after scale-down considered an expected
   operational hazard that the system should tolerate through delete/recreate,
   or should F3/F4 be blocked until the root cause is narrowed further?

3. The inflight fix deliberately does not retry a failed dispatch immediately;
   it releases inflight so the next planner tick can decide again. Is that the
   desired semantics, or should ActionQueue grow an explicit retry/backoff queue?

4. The second F2.5 high-load run had one gateway `503` while still producing
   correct Z-state and scale actions with no component restarts. Should that be
   accepted as live-load noise, or should F2.5 require a clean zero-error rerun
   after hygiene?

5. Should F3 live defrag proceed after topology is restored and reconcile is
   clean, or should the plan insert an additional 5-minute scale-up/scale-down
   hygiene gate first?

## Recommended Next Steps

1. Run full `cd tre && make check`.
2. Commit the F2.5 high-load evidence plus hygiene notes.
3. Continue to F3 live defrag unless architect review says to add an extra
   scale-up/scale-down hygiene gate first.

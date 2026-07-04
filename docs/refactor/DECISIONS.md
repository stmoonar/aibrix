# Refactor Decisions

## ADR-0001: Remote server 76 is authoritative

- Date: 2026-07-04
- Status: accepted

### Context

`REFACTOR_PLAN.md` states that all new code changes and tests happen in `/data/nfs_shared_data/xxy/aibrix` on server 76. The local Windows checkout is not authoritative.

### Decision

All implementation and verification must happen in `/data/nfs_shared_data/xxy/aibrix` on server 76. Local test/build output is ignored. Local work may only be used as a disposable draft source and must be re-applied/verified remotely.

### Consequences

P1 work drafted locally is not considered complete until created and verified on server 76. P0 inventory and subsequent docs in this directory are authoritative.

## ADR-0002: Record baseline commit instead of creating `baseline-v0` tag immediately

- Date: 2026-07-04
- Status: accepted

### Context

P0 allows either tagging `baseline-v0` or recording the baseline commit in WORKLOG. The remote workspace has an untracked `REFACTOR_PLAN.md` at the start of work.

### Decision

Record baseline commit `adfe6f8373afe5a90a2e93687474f07a0d4aed26` in P0 docs and WORKLOG. Defer creating phase tags until the first clean phase commit.

### Consequences

This avoids tagging a state before the refactor plan/documentation commit exists. The baseline is still recoverable by commit hash.

## ADR-0003: Same-slot shrink probes start in loop tick, not safescale_task

- Date: 2026-07-04
- Status: accepted

### Context

`docs/refactor/10_next_steps.md` N1.2 says the new same-slot HIGH shrink action is consumed by `safescale_task`. The current P5/P9 controller architecture already has a narrower boundary: planner actions are converted to SafeScale probes in `loops/tick.py` through `_apply_safescale()`, while `safescale_task.py` only observes active probes and emits commit/rollback actions.

### Decision

Keep the existing boundary. `ShrinkForSlotAction` is emitted by the pure planner and consumed by `loops/tick.py`, which starts the SafeScale probe with the concrete donor serve id and records the TP=2 beneficiary as a pending upscale. `safescale_task.py` remains the observer for active probes.

### Consequences

This avoids routing planner output into the observation task and preserves the existing SafeScale lifecycle. The behavior still satisfies N1.2's intent: same-slot shrink is SafeScale-gated, and beneficiary expansion is delayed until after donor shrink commits.

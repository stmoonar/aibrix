# Single decision loop (2026-10-28)

Status: approved (lead + design review 2026-10-28). Branch `feat/single-loop-20261028`.

## Motivation

The paper (§5) describes **one main polling loop with two logical cadences**: every
poll classifies the models, the fast cadence plans the rescue of CRITICAL models, the
slow cadence plans the rebalance toward LOW models. v1 did the same:
`_build_paper_scaling_plan` was called once per tick and planned both parts against one
ledger of free capacity.

v2 (up to main `633b080f`) runs two asyncio tasks instead:

- `rescue_task` ticks on a fixed `sleep(5)` after each tick (the period drifts with the
  tick time and has no phase relation to the snapshots, published every 2 s);
- `fairness_task` ticks every 10 s with `rescue_due=False`;
- each task has its own `PaperStateCache`.

Defects seen in the 10-28 campaign and review:

1. **Two ledgers.** The fairness tick does not see CRITICAL models. It gave half of a
   TP2 pair (freed for a CRITICAL 14b) to a LOW 7b.
2. **Registry order.** Every ordinary CRITICAL receiver sorts as `(0, 0.0)`, so two
   CRITICAL models compete for a free GPU in registry order, not by need. LOW receivers
   are not sorted at all.
3. **Deficit steps on stale evidence.** The rescue target `ceil(n·τ_crit/Z)` (up to
   `2n`) is computed from the window that still describes the old replica count. The
   partial-window and low-evidence caps patch this case by case.
4. **TP2 supply.** A CRITICAL TP2 receiver stops after the same-slot shrink / empty
   slot / defrag attempts (an unconditional `continue`): it never reaches the IDLE or
   HIGH / middle-zone relays. The same-slot shrink takes HIGH donors only. Under
   `TRE_SAFESCALE_SCOPE=receiverless_shrink` a same-slot release is a bare donor sleep;
   "the next tick wakes the receiver" - and another receiver can take the pair first.

## Design

### L1 - one loop, two sections

- One task (`loops/decision_task.py`) wakes when the metrics task publishes a snapshot
  (`SnapshotBox.wait_newer`; wakes coalesce, the newest snapshot is used).
- Observation time of a snapshot = the newest `scraped_ms` of its windows
  (`Observed.t_last_ms`; `ts_ms` for a snapshot without windows).
- The **rescue section** runs when `TRE_RESCUE_INTERVAL_SECONDS` (5 s) of observation
  time passed since its last run; the **rebalance section** (fairness, LOW receivers)
  when `TRE_FAIRNESS_INTERVAL_SECONDS` (10 s) passed. Tolerance: the fetch jitter,
  bounded by half the evidence period (`SPAN_TOLERANCE_MS` at >= 2 s).
- Both sections run in the SAME `run_planner_tick` call, rescue first. They share one
  ledger (planned deltas, claimed GPUs, released donor pods) and one `PaperStateCache`.
  One plan goes out through the ActionQueue (the tick never waits for the SM).
- Stale snapshot: the sampler marks it stale after `TRE_METRICS_STALE_HOLD_MS` (20 s)
  of stale windows; the tick then decides nothing (`snapshot_stale`). If no snapshot is
  published at all for that hold plus one evidence period, the loop logs
  `decision_snapshot_stale` (once, `decision_snapshot_recovered` when one arrives).
- Deleted: `fairness_task.py`, the fixed rescue timer, H1 (the rescue section woke LOW
  receivers on free GPUs). LOW is served only in the rebalance section. IDLE release and
  the receiver-less HIGH shrink stay in the rescue section.
- Existing guards stay: in-flight models, O1 view-pending, relay holds, rollback holds.

### L2 - receiver order

- CRITICAL: key `(tier, Z ascending)`. Tiers as before: TSS-confirmed CRITICAL, then
  onset-saturation receivers (no Z: most waiting per replica first), then O1-exempt
  receivers. `Z = None` sorts as `+inf`. Raw Z, not `Z / τ_crit`.
- LOW: Z ascending (`classify.receiver_sort_key`). Donor order is unchanged.

### L3 - one step per plan

- Every receiver gets at most `max(1, ceil(scale_step_ratio · n))` replicas per plan
  (v1 `scale_step`; ratio 0.1, so +1 below 10 replicas) in rescue and rebalance. This
  also holds for the onset saturation rescue (it was a bounded doubling).
- The `rescue_targets` ledger stays ("one deficit, one target"): the target is
  `base + step`; a target the window does not reflect yet gives the same target again,
  so nothing is added until the window moves (or the target landed nothing).
- The H3 O1 exemption stays (it is a hold exemption, not a step cap). O1 is unchanged.
- Deleted: `rescue_desired`, `scaling.rescue_max_step_ratio`, `rescue_max_step_pods`,
  `breakpoint_partial_max_step`, `breakpoint_lowevidence_requests`,
  `saturation_max_step_factor`. A registry that still carries them loads; the loader
  logs one deprecation line and ignores them.

### L4 - TP2 supply (minimal)

- The unconditional `continue` after the TP2 slot attempts is gone: the remaining need
  of a TP2 receiver goes to the IDLE relay and the HIGH / middle-zone relay, which count
  only aligned pairs whose every GPU one donor model holds (`_SlotOccupancy.pairable`).
  No single-GPU counting for a TP receiver.
- Same-slot shrink: any donor candidate (IDLE, HIGH, middle zone; floor headroom, not
  probing, not in flight, not rollback-held) whose slot mate is free; IDLE first, then
  HIGH, then middle zone, lowest Z within a tier.
- A same-slot release is always ONE plan: the donor pod's sleep and the receiver's wake
  with `hints` = its binding on that pair.
  - scope `all`: probe first; the SafeScale commit is that plan (receiver target hints
    from the probe pods' slot).
  - scope `receiverless_shrink` / `off`: `_release_without_safescale` emits both items
    as one `PlanBatch` (the receiver keeps its rescue target).
- Mixed-donor pairs (7b on one GPU, 8b on the other) stay unsupported: nothing is
  planned, `capacity_blocked:<model>` is recorded.

## Ablation

`TRE_ABLATION_DISABLE_SLOW_LOOP=true`: the rebalance section never runs (LOW receivers
get nothing); the rescue section runs as usual. The old name
`TRE_ABLATION_DISABLE_FAST_LOOP` still makes the controller refuse to start.

## Invariants (tests)

1. A same-slot release plan contains the donor sleep and the receiver wake with hints =
   that pair; no bare-sleep ShrinkForSlot under any scope.
2. Two CRITICAL models, one free GPU: the lower Z wins whatever the registry order; LOW
   is never planned before CRITICAL.
3. Each receiver gets at most +1 per plan; no second +1 before the previous target is
   reflected.
4. TP2: a pair (free GPU, IDLE 7b) is obtained in one plan; a mixed-donor pair gives
   nothing and `capacity_blocked`.
5. One ledger: rescue and rebalance of one tick never assign one free GPU twice; with
   `DISABLE_SLOW_LOOP` LOW gets no action.
6. Period invariance: the same observation stream gives the same decisions at 1 s and
   2 s snapshot cadence; a stale snapshot gives no decision and an alert.

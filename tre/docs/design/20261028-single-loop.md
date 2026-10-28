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
  when `TRE_FAIRNESS_INTERVAL_SECONDS` (10 s) passed, and only together with the rescue
  section (v1: the slow beat is always a fast beat; a LOW receiver is never planned on
  a call that skipped the CRITICAL ones). The span since a section's last run is
  snapped to whole snapshot (producer) periods - that absorbs a scrape jitter below half
  a period and nothing else - so the cadence is deterministic and rounds UP to the
  snapshot grid: with 2 s snapshots the rescue section runs every 6 s, and at every
  10 s rebalance call (mean rescue spacing 5 s; `first_run_check` expects that).
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

Superseded for TP>1 receivers by L4 rev 2 below (same day); kept as the record of what
ran between the two commits.

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

### L4 rev 2 - block-level supply for TP>1 receivers (user decisions 2026-10-28)

Replaces, for TP>1 receivers, the same-slot shrink, the TP empty-slot / capacity paths
and the relays (`pairable` pairs) of L4. TP1 receivers are unchanged for now.

- **Unit**: a buddy block (`tre_common/gpu_placement`): the aligned order-k slot a
  TP=2^k replica needs. `planner.cheapest_block(...)` (pure) chooses one block per
  receiver per plan (+1 replica, the step of L3).
- **Candidates**: the slot of each sleeping, not hidden binding of the receiver (a wake,
  hints = that binding), and - only while the receiver has no unplanned sleeping
  binding, as the SM creates only then - every fully free aligned block (a cold create;
  a wake wins over a create at equal key). Every GPU of a candidate is free (not awake
  in the view, not claimed by this plan's ledger, not blocked) or held by exactly one
  awake, not hidden pod of an **eligible donor** whose slot lies inside the block (a
  donor pod spanning the whole block counts once). Any other GPU: not a candidate.
- **Eligible donor**: state IDLE, HIGH or HEALTHY (never LOW / CRITICAL); keeps its
  floor after this plan's releases; not probing, not in flight, not down-blocked, no
  probe of it in this plan; a pod not released by this plan. A non-IDLE donor gives
  at most ONE replica per plan, to all receivers together (no release of it planned
  yet in this plan) - so a block whose two GPUs are two pods of one HIGH model is not
  a candidate. IDLE donors may give more, down to their floor. Mixed blocks (two
  different donor models, one pod each) are allowed.
- **Choice** (lexicographic, SLO risk first; not a cost sum):
  1. the worst donor tier in the block: free GPU < IDLE < HIGH < HEALTHY;
  2. fewer donor pods to sleep;
  3. higher donor Z (most surplus first; two donors: the lower Z, then the higher);
  4. wake before create, then the placement score of `gpu_placement.choose_placement`
     (reserve violation, split cost, node load, address) for a deterministic tie.
  A HEALTHY donor is touched only when no block of free / IDLE / HIGH GPUs exists.
- **Release**: IDLE donors sleep at once in every scope. A block with a HIGH /
  HEALTHY donor follows the scope: `receiverless_shrink` / `off` - at once; `all` - a
  SafeScale probe of that pod whose commit wakes the receiver (hints = the block), the
  IDLE pods of the block sleep at once. Under `all` a block that needs probes of TWO
  donor models is not a candidate (one probe per model, no joint commit).
- **One plan**: the immediate form is one `PlanBatch` (donor sleeps + receiver wake)
  that joins the decision's fused SM plan as one unit; the SM runs the wake as wave 2
  after the sleeps.
- The plan claims the block's GPUs and releases the donor pods in its ledger; the
  receiver's rescue target counts the +1 (`receiver_upscales`). No block:
  `capacity_blocked:<model>` once per receiver per plan; defrag (off by default) is
  the fallback as before.

### Implementation notes (2026-10-28)

- L1 (review 2026-10-28): the rebalance section runs only with the rescue section; the
  span is snapped to the snapshot grid (`decision_task.grid_span_ms`); a call without a
  cluster view decides nothing and leaves the sections due.
- L3: `saturation_max_step_factor` (the bounded doubling of the onset saturation rescue)
  is retired with the other step keys.
- L4: a non-IDLE donor gives a TP receiver one step (one replica) per decision; a slot
  of two single-GPU replicas of one HIGH / middle-zone donor in one decision was rejected
  (user, 2026-10-28). TP supply is to be redesigned on buddy blocks
  (`tre_common/gpu_placement`).
- L4: an IDLE donor's same-slot release is immediate under every scope, scope `all`
  included (only an IDLE donor is released at once): one `PlanBatch`, no probe. HIGH /
  middle-zone donors keep the scope semantics (probe under `all`; the commit's
  receiver target carries the hints - the receiver's sleeping bindings on the GPUs of
  the probe pods).
- L4: `capacity_blocked:<model>` is recorded once at the end of a TP receiver's
  rescue, when the slot path and the relays together left it short (no longer by the
  slot path alone, which the relays may still fill).
- L4 (review P1-1): a free slot mate is judged on the plan's ledger (awake, claimed by
  this plan's wakes / creates / slots, blocked), never on the view alone.
- L3 (review P2-3): a receiver an unresolved SafeScale probe promises replicas to (its
  commit wakes them) holds that step like a queued rescue target until the probe
  resolves (`tick._with_probe_promises`); the probe placements carry the rescue tag.
- L4 rev 2 (implementation): `planner.cheapest_block` / `BlockSupplyAction` replace
  `_try_plan_same_slot_shrink` / `ShrinkForSlotAction`, the TP empty-slot loop and the
  TP use of the IDLE / middle relays (`pairable` stays for TP1). Kept: the joint
  sleeping-binding wakes of all receivers up front (a free block wake, tier free, is
  what `cheapest_block` would choose first anyway); defrag as the fallback of a
  CRITICAL TP receiver (not of a LOW one, as before). A donor block needs a sleeping
  binding of the receiver on it (the SM creates only into free GPUs). A TP1 receiver's
  HIGH / middle-zone relay or probe skips a donor that gave a block this plan, and a
  block skips a non-IDLE donor any earlier item of the plan took from (one replica per
  plan, both directions). Events: `block_supply:<receiver>:<block>:<tier>:<pods>`;
  reasons `critical_block_<tier>` / `low_fairness_block_<tier>` (`_nosafescale` when a
  HIGH / HEALTHY pod sleeps at once).
- One decision, one SM plan: the SM's `POST /v2/plans` runs several sleeps and wakes in
  one writer-lock hold, a wake that needs GPUs the plan's own sleeps free as wave 2
  after them, hints first (`20261027-sm-plans.md` 4.1). So a same-slot release joins
  the decision's fused plan as one unit (`ActionQueue._fuse_into_plans`). What still
  splits a decision: two items of one model (the SM allows one item per model; the
  next plan then serializes on that model) and SafeScale one-shot actions (a commit
  keeps its own plan and retries).
- Plan groups (2026-10-28, `20261027-sm-plans.md` section 10, review A-1 / A-2): every
  transfer - a TP block or a TP1 relay pair - is named and is one plan group; several
  named items of one model share a plan, so a block and a relay taking two pods of one
  IDLE donor are one SM plan (only a pod named twice splits it).

## Ablation

`TRE_ABLATION_DISABLE_SLOW_LOOP=true`: the rebalance section never runs (LOW receivers
get nothing); the rescue section runs as usual. **The name is kept, the semantics
changed on 2026-10-28**: from 2026-10-08 to 2026-10-27 it removed the fairness task and
every rescue tick also planned fairness (LOW receivers were served). No-slow-loop data
recorded before 2026-10-28 is not comparable with runs after it. The old name
`TRE_ABLATION_DISABLE_FAST_LOOP` still makes the controller refuse to start.

## Invariants (tests)

1. A TP block plan contains the donor sleeps and the receiver wake with hints = its
   binding on the block; no bare donor sleep under any scope.
2. Two CRITICAL models, one free GPU: the lower Z wins whatever the registry order; LOW
   is never planned before CRITICAL.
3. Each receiver gets at most +1 per plan; no second +1 before the previous target is
   reflected.
4. TP2 (L4 rev 2): a free block is woken into, no donor touched; the worst tier decides
   (free + HEALTHY loses to HIGH + HIGH: two sleeps and the wake in one plan); a block
   with a LOW / CRITICAL pod is not used; a non-IDLE donor gives one replica per plan
   (two receivers; two pods of one HIGH model on a block); scope `all`: one HIGH donor
   is a probe carrying the receiver, a block needing two probed donors is skipped.
5. One ledger: rescue and rebalance of one tick never assign one free GPU twice; with
   `DISABLE_SLOW_LOOP` LOW gets no action.
6. Period invariance: the same observation stream gives the same decisions at 1 s and
   2 s snapshot cadence; a stale snapshot gives no decision and an alert.

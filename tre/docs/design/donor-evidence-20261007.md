# Donor evidence fixes (2026-10-07)

Status: implemented on `fix/donor-evidence-20261007` (controller only; no SM change).
Background: diagnosis "TP2 donor asymmetry" (T7 / Alternating, 2026-10-07): HIGH donors
released at once in the rescue and fairness loops were wrong 6/6 times at n = 2 with Z
just above tau_high; a split TP2 pair took 57-678 s to come back. User decision
(2026-10-07): F1-B + F2 + slot reservation + F4. This overrides the 2026-10-02 rule
"the CRITICAL rescue keeps immediate HIGH donors".

Paths are relative to `tre/controller/tre_controller/`.

## 1. F1-B: a HIGH donor gives capacity only through a SafeScale probe

Why: "Z > tau_high now" says nothing about Z with one replica less (n = 2: half the
capacity; the queue term compounds, Z fell about twice the linear estimate). A probe
(hide, observe, commit or roll back) is cheap; undoing a wrong release is not
(a TP2 pair). IDLE stays immediate: an idle window is evidence at any replica count.

Changes (`planning/planner.py`):

- `build_plan` :642 `probe_donors` = HIGH donors (cost order of `paper_donors`) +
  the middle zone. The rescue loop (:1032) and the fairness loop (:1322) probe from
  this list with the unchanged middle-zone code (`_plan_middle_zone_probe`); reasons
  `critical_high_donor_safescale` / `low_fairness_high_donor_safescale` (labels only).
- The immediate donor loops (`critical_donor_immediate` :1005, `low_fairness_donor_immediate`
  :1296) take IDLE donors only; IDLE still gives its whole surplus (Q3).
- `probe_held` / `rollback_held` (review round 2026-10-07): (a) one rule for every
  donor probe - the existing rollback evidence hold (`probe_backoff_models`,
  `SafeScaleStateMachine.rollback_retry_holds`, `same_evidence` / `no_new_window`) holds
  HIGH donors, middle-zone (HEALTHY / LOW) donors and the same-slot shrink (its call
  site adds held models to `inflight_models`, event `safescale_rollback_hold`), not only
  the receiver-less HIGH probe. No new gate: the existing one, applied everywhere.
  (b) one probe per donor model per plan (the state machine is keyed by model): a donor
  already probed in this plan gives no other receiver anything; a second receiver,
  CRITICAL ones included, waits until that probe resolves (serialized).
- The fairness piggyback (`_piggyback_probe`) is removed. It handed a probe's
  unclaimed donor replicas to LOW receivers as receiver replicas (a TP2 receiver of two
  TP1 donors left one phantom "unclaimed"), without slot geometry. In practice it only
  served receiver-less HIGH probes; such a probe now promises nothing, and the GPU its
  commit frees is planned from free capacity on the next tick (F2 makes it visible at
  once) - with real geometry, at most one tick later.
- Removed: `_donor_give`, `PlanConfig.donor_surplus_release`, the registry key
  (`tre_common/registry.py`; an old registry carrying it still loads with a warning),
  `deploy/registry.yaml`, `deploy/overlays/tre-v2/params.yaml`, `loops/tick.py`
  `_scaling_options`, and the tests of the switch.
- Unchanged: `_try_plan_same_slot_high_shrink` (already SafeScale; now reads the
  rollback hold, see above), HIGH proactive
  shrink (receiver-less SafeScale), IDLE proactive shrink, the SafeScale gates.

Behaviour change / risk:

- A CRITICAL receiver whose only capacity is a HIGH donor waits for the probe's commit
  (evidence floor about 2 post-hide grids + `min_commit_samples`, i.e. ~20-30 s with F4,
  up to W otherwise) instead of one SM transfer (~5-7 s). Diagnosis data: 2 of 4
  CRITICAL receivers left CRITICAL within 20 s of the immediate GPU; the other 2 stayed
  CRITICAL 110-130 s anyway.
- A HIGH donor with an in-flight probe serves no second receiver until the probe
  resolves (before: an immediate relay on another pod in the same tick).
- A donor whose probe rolled back for capacity is not probed for a receiver until its
  routable count changes or Z rises by `rollback_retry_z_margin` (before: a HIGH donor
  was released at once, ignoring the rollback). When the only donors are HIGH and all of
  them are held this way, the receiver waits until a donor's routable count changes or
  its Z rises. This is the semantics the user accepted (evidence, not time).

Invariant tests (`tests/test_donor_evidence_20261007.py`): a HIGH donor gets no relay
and no `urgent` sleep, only a probe (rescue and fairness); an IDLE donor still relays;
a held HIGH donor gets nothing.

## 2. F2: event-driven cluster-view refresh

Why: the planner reads `ClusterViewBox`, refreshed every `fairness_interval_s` (10 s).
A committed receiver-less scale-down freed a GPU the next tick could not see for up to
10 s (Alternating: 10 s lost). A fixed polling period is a cold-start-style timer.

Changes: `ClusterViewBox.request_refresh` (`loops/cluster_view_task.py` :76) sets an
`asyncio.Event`; `cluster_view_task` clears it before each GET and waits for the event
or the period, whichever first (`wait_next_refresh` :291). `ActionQueue` takes
`on_fleet_change` (`loops/action_queue.py` :340) and calls it wherever it stamps an
O1 / view-pending change (:1545) - every SM call that changed the fleet: SafeScale
commit sleeps, relays, scales, hides/unhides. App wiring `app.py` :426. The period
stays as the fallback; no sleep is added. A request during a GET triggers one more GET
(that answer may predate the change). Requests are coalesced by the event.

Risk: more `GET /v2/state` (at most one per fleet change, coalesced). Test: a commit
calls the hook; a request refreshes the view while the period never ends.

## 3. Reserve the slot an in-flight probe counts on

Why (diagnosis section 3.6): a same-slot shrink probe for a CRITICAL TP2 receiver
relies on the donor pod's free slot mate. `_SlotOccupancy` is rebuilt every tick from
awake bindings + SM `blocked_gpus`, so the mate looked free: H1 `low_rescue` (it prefers
half-used pairs) or any wake / create could take it, and the commit's wake would 409.

Changes:

- `probe_reserved_gpus` (`planning/planner.py` :2058): for every unresolved probe
  (probing or committing) and every receiver in its `pending_upscales`, the aligned
  `tp_size(receiver)` slot around each probe-pod GPU. While the donor pod is hidden its
  own GPU is occupied anyway; this adds the mate, and between a commit's donor sleep and
  the receiver's wake also the freed GPU itself.
- `loops/tick.py` :191 adds them to the view's `blocked_gpus` for the whole plan (event
  `probe_slot_reserved:<node>/<gpu>,...`). Released when the probe resolves: state,
  not time.
- Same tick: when the rescue plans a same-slot shrink, the slot is claimed in
  `_SlotOccupancy` right away (:910), so the LOW rescue / fairness of the same plan
  cannot take it either.
- `_try_plan_same_slot_high_shrink` (:1980) treats a blocked GPU (SM not wakeable, or
  reserved) as no free mate.

Risk: a probe for a TP2 receiver blocks one more GPU for the probe's duration. The
reservation is local to the controller: the SM does not know it, so the SM's floor
repair or self-healing may still wake into a reserved GPU. It is best effort - it
removes the controller's own conflicting wakes, not every possible 409. Tests: same-plan claim; cross-tick reservation (unit + tick wiring);
control cases free again.

## 4. F4: a CRITICAL model is an early-commit trigger

Finding: the existing early commit (`planning/safescale.py` `_try_early_commit`) does
not depend on receivers at all. It fires on the first poll where (a)
`min_commit_samples` requests are judged, (b) the formal gates pass on the evidence so
far, (c) the hidden pods have nothing in flight, (d) the newest snapshot window holds
`early_commit_min_grids` (= `scaling.min_evidence_grids` = 2, 10 s grid) post-hide grids
AND one donor p95 e2e has passed since the hide confirmation. In Alternating_s1 the
7b probe (hide 944.3 s) committed at 978.9 s with `min_elapsed_ms` 30 000: the "22 s
after 14b turned CRITICAL" were this evidence floor.

Change (review round 2026-10-07): `observe(..., critical_receivers=)` and
`_try_early_commit`: when a model other than the donor is CRITICAL in the latest planner
tick, only (c) is waived. The hidden pods' in-flight requests are a cost, not evidence:
the sleep aborts them and the sidecar re-issues them, as on an immediate release.
(a), (b) and all of (d) stay. The p95 e2e term is evidence completeness: SLO samples
count at completion, so less than one e2e after the hide they are biased towards short
requests. Immediate rollbacks are unchanged. Any CRITICAL model (not "a receiver of this
donor's GPU"): the commit is still fully evidence-backed, the freed GPU goes to the
planner's free pool (CRITICAL first, next tick via F2), and slot matching in SafeScale
would duplicate the planner. Source: `ModelStateBox` (`app.py`,
`loops/safescale_task.py`); no new state, no "upgrade to preemption" state.

Benefit: only when the hidden pods still serve long requests at the evidence floor
(about 0 s on the Alternating case, where they had drained by then). Audit:
`early_commit.critical_receivers`, `hidden_drained: false`. Tests (driven through the
state machine's public `observe`): a busy hidden pod does not block the commit with a
CRITICAL model; it does without one (or when only the donor is CRITICAL); one p95 e2e
and `min_commit_samples` are never waived.

## Points for the main session

1. F1-B: rescue latency from a HIGH donor rises from one transfer to one probe
   (evidence floor ~20-30 s after the hide, up to W); one probe per donor at a time.
2. The live ConfigMap still carries `scaling.donor_surplus_release: false`: harmless
   (warning). Remove it with `merge_live_registry.py` at the next structural update.
3. `docs/donor-release-policy.md` (local workspace) must be rewritten (F1-B).

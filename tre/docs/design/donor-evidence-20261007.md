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
- `probe_held` :644: (a) the existing rollback evidence hold (`probe_backoff_models`,
  `SafeScaleStateMachine.rollback_retry_holds`, `same_evidence` / `no_new_window`) now
  applies to every HIGH donor probe, not only the receiver-less one - this is what makes
  "SafeScale's rollback gate constrains HIGH donors" true; no new gate. (b) one probe per
  donor model per plan (the state machine is keyed by model): a donor already probed in
  this plan serves another receiver only through the existing piggyback.
- Removed: `_donor_give`, `PlanConfig.donor_surplus_release`, the registry key
  (`tre_common/registry.py`; an old registry carrying it still loads with a warning),
  `deploy/registry.yaml`, `deploy/overlays/tre-v2/params.yaml`, `loops/tick.py`
  `_scaling_options`, and the tests of the switch.
- Unchanged: `_try_plan_same_slot_high_shrink` (already SafeScale), HIGH proactive
  shrink (receiver-less SafeScale), IDLE proactive shrink, the SafeScale gates.

Behaviour change / risk:

- A CRITICAL receiver whose only capacity is a HIGH donor waits for the probe's commit
  (evidence floor about 2 post-hide grids + `min_commit_samples`, i.e. ~20-30 s with F4,
  up to W otherwise) instead of one SM transfer (~5-7 s). Diagnosis data: 2 of 4
  CRITICAL receivers left CRITICAL within 20 s of the immediate GPU; the other 2 stayed
  CRITICAL 110-130 s anyway.
- A HIGH donor with an in-flight probe serves no second receiver until the probe
  resolves (before: an immediate relay on another pod in the same tick).
- A HIGH donor whose probe rolled back for capacity is not probed for a receiver until
  its routable count changes or Z rises by `rollback_retry_z_margin` (before: released
  at once, ignoring the rollback).

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

Risk: a probe for a TP2 receiver blocks one more GPU for the probe's duration. Only the
controller's own plans honour it (the SM does not know it); in TRE mode nothing else
wakes models. Tests: same-plan claim; cross-tick reservation (unit + tick wiring);
control cases free again.

## 4. F4: a CRITICAL model is an early-commit trigger

Finding: the existing early commit (`planning/safescale.py` `_try_early_commit` :1366)
does not depend on receivers at all. It fires on the first poll where (a)
`min_commit_samples` requests are judged, (b) the formal gates pass on the evidence so
far, (c) the hidden pods have nothing in flight, (d) the newest snapshot window holds
`early_commit_min_grids` (= `scaling.min_evidence_grids` = 2, 10 s grid) post-hide grids
AND one donor p95 e2e has passed since the hide confirmation. In Alternating_s1 the
7b probe (hide 944.3 s) committed at 978.9 s with `min_elapsed_ms` 30 000 (p95 e2e,
a histogram bucket edge): the "22 s after 14b turned CRITICAL" were the evidence floor
(two grids, ~973-975 s) plus ~4-6 s of the e2e settle term.

Change: `observe(..., critical_receivers=)` (:749) and `_try_early_commit` (:1412): when
a model other than the donor is CRITICAL in the latest planner tick, the settle
conditions are waived - (c) the hidden pods' drain (their requests are aborted at the
sleep and re-issued by the sidecar, as on an immediate release) and the p95 e2e part of
(d). The evidence floor stays: (a), the post-hide grids of (d), and every gate of (b)
(SLO on the evidence so far, KV, Z tail, completeness); immediate rollbacks are
unchanged. Any CRITICAL model (not "a receiver of this donor's GPU") because the commit
is still evidence-backed, the freed GPU goes to the planner's free pool (CRITICAL first,
woken on the next tick via F2), and matching slot geometry in SafeScale would duplicate
the planner. Source: `ModelStateBox` (`app.py` :173, `loops/safescale_task.py` :308);
no new state, no "upgrade to preemption" state.

Benefit (estimate): ~4-6 s on the Alternating case; more for long-e2e donors or busy
hidden pods. Audit: `early_commit.critical_receivers`, `hidden_drained: false`,
`min_elapsed_ms: 0`. Tests: commits on the evidence floor with a CRITICAL model; no
other CRITICAL model (or only the donor) keeps the settle conditions; too few samples
never commit.

## Points for the main session

1. F4 waives the p95-e2e settle term for the CRITICAL case. It is not a fixed timer
   (it is one donor p95 e2e, Review P2-3: the remaining pods' concurrency needs about
   one e2e to settle), but it is a minimum observation time. If it must stay, revert the
   `min_elapsed` line in `_try_early_commit`; F4 then only waives the drain and gains
   ~0 s on Alternating. The 2-grid floor (O1 warm rule) is unchanged and is now the
   dominant term (~20-30 s after the hide).
2. F1-B: rescue latency from a HIGH donor rises from one transfer to one probe; one
   probe per donor; rollback hold now applies to receiver-driven HIGH probes.
3. Not changed: `_try_plan_same_slot_high_shrink` does not read the rollback hold (T7:
   4 rolled-back same-slot probes 7b -> 14b at 266/316/386/476 s); HEALTHY/LOW middle
   zone donors do not read it either.
4. The live ConfigMap still carries `scaling.donor_surplus_release: false`: harmless
   (warning). Remove it with `merge_live_registry.py` at the next structural update.
5. `docs/donor-release-policy.md` (local workspace) must be rewritten (F1-B).

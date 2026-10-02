# Timer cleanup: state and evidence gates instead of fixed waits (2026-10-02)

Branch `feat/timer-cleanup-20261002` (based on the onset saturation rescue branch).
One commit per item, so each can be rebased on its own.

## Premise

TRE switches models in place: a vLLM wake / sleep takes 2-3 s and no Pod is
created. Its real costs are two:

* a sleep interrupts the requests in flight (the reissue sidecar continues them);
* waking a model may displace another model resident on the same GPU.

Several timers of the controller were sized for systems whose scale-up starts a
Pod (tens of seconds to minutes). The rule applied here:

| Timer protects ... | Treatment |
|---|---|
| correctness (no double wake on one GPU, books vs. physical state, crash recovery, leases) | kept |
| decisions on signals that lag the change | replaced by a state / evidence gate |
| debouncing only | removed |

Timers kept untouched: the S3 wake-refusal GPU cooldown, gpu-truth admission,
SM leases / reservations, the B8 commit max age, the floor-violation hold
(`TRE_FLOOR_VIOLATION_COOLDOWN_TICKS`, handled separately), the C1 rescue
target bookkeeping and the O1 breakpoint window itself.

## 1. F4 action cooldown -> O1 breakpoint gate

### Before

Review F4 (`loops/tick._action_cooldowns`, `TRE_ACTION_COOLDOWN`, default on):
after a model's executed action, its next action waits until a metrics window
**starts** after that action completed (W = 30 s plus one window step plus a
tick). Same direction held; after a scale-up a scale-down is held too; after a
scale-down only a CRITICAL receiver may scale up. It compared the completion time
with `snapshot.models[m].window_start_ms`, the start of the **whole** 30 s
window, not of the O1 effective window.

### Why it is not needed with O1

O1 (`scaling.breakpoint_window`, design `20261001-o1-breakpoint-window.md`)
already makes every routable-count change a breakpoint:

* `note_routable` (`signals/trs.py`) records a change of the routable count
  between two fleet views, dated by the controller's own SM-call completion
  stamp (`ActionQueue.routable_changes`) or the view's fetch time, plus
  `breakpoint_margin_ms`;
* `effective_window` then restricts the window to complete grids after the
  breakpoint and restarts the EMA there;
* the planner holds a donor / middle-zone model (any scale-down) until a whole
  window follows the breakpoint (`signal_full_window`) and a receiver until
  `min_evidence_grids` grids of post-change evidence exist (`signal_warm`).

So an executed action that changed the routable count holds the model until the
signal describes the new count. The F4 rule only added a fixed wait on top
(about one more window for receivers).

Gaps found while checking the coverage, and how they are closed:

| Case | O1 alone | Gate now |
|---|---|---|
| The fleet view of this tick was fetched before the action completed (views refresh every fairness interval, 10 s) | no breakpoint yet: the model would be planned again on the old count | **O1 view-pending gate**: models whose last action completed after `ClusterView.fetched_ms` are held (F4 direction rules, event `o1_view_pending_hold:<m>`) until a newer view exists |
| O1 suspended (`breakpoint_window_suspended`, gateway clock check) or off | whole windows only | F4 (unchanged rule) |
| The model has no fleet view this tick (`note_routable` is not called) | change not seen | F4 |
| Stale-held context (`paper_state_stale_hold`, tokens missing) | context not recomputed | F4 |
| O1 hold fallback (`breakpoint_hold_max_windows` reached) | the receiver decides on the whole window again | F4 |
| The action did not change the routable count (failed) | nothing to reflect | nothing to wait for |

Each tick's context carries `o1_routable_tracked` (O1 active, model in the fleet
view, signal computed this tick, no hold fallback). F4 applies exactly to the
models where it is not true; the view-pending gate exactly to the models where
it is. `TRE_ACTION_COOLDOWN` stays the switch of the F4 fallback (pinned to `1`
in the overlay).

LOW receivers (slow loop) are covered the same way: after a scale-up the
receiver is held by the view-pending gate, then by the post-breakpoint evidence
requirement; a second step needs `min_evidence_grids` grids measured at the new
replica count. The C1 rescue-target bookkeeping was not extended to LOW
receivers.

### Behaviour change

* A receiver acts again once two post-change grids exist (about 20-30 s after
  the change) instead of after the next whole window (about 40-60 s).
* After a scale-down, a LOW receiver may scale up once the post-change evidence
  says LOW (F4 allowed only a CRITICAL receiver). The evidence is measured at the
  new replica count, so this follows the measured state rather than the window
  that still contained the old count.
* Scale-downs: unchanged in effect (O1 already needed a whole window after the
  breakpoint, which is at least the F4 wait).

### Fallback and risks

* While O1 is suspended the F4 rule is active again (tested).
* A change the SM applies later than its answer (routable label written after
  the call returned) is dated at the view that shows it; the grid holding it is
  excluded, so the effective window still starts after it.
* `fetched_ms` and the action completion time are both controller clock. A view
  without a fetch time (synthetic / offline) gives no view-pending hold; O1 then
  dates a change at the window end it is first seen in.

Tests: `controller/tests/test_timer_cleanup_f4_o1_20261002.py`.

## 2. SafeScale rollback backoff -> rollback evidence gate

### Before

A13 (commit 90c18c73, `TRE_SAFESCALE_ROLLBACK_BACKOFF_MS` = 60000): after a probe
of a model rolled back (any reason except a preemption for the model's own
scale-up), no receiver-less HIGH proactive probe of that model for 60 s. Purpose:
stop hide / unhide flapping. Demand-driven releases were never held.

### Why a fixed time is the wrong measure

A rollback says "with the load seen at the probe's start, the model could not
spare these pods". Sixty seconds later the same load gives the same answer, so
under steady load the probe repeated every 60 s + W; after a real load drop the
model still waited the full 60 s.

### Gate

At the rollback the state machine keeps `RollbackEvidence`: the Z and routable
count of the planner decision that **started** the probe (`ProbeWindowInputs`,
stored on the probe as `start_z_m` / `start_routable` and in its record), the
rollback time and code. The Z observed during the probe is not used: it was
measured with the probe pods hidden, and the unhide alone raises it again.

The next receiver-less HIGH probe of the model is planned only when

1. a metrics window ends after the rollback (new evidence; hold reason
   `no_new_window`), and
2. for a capacity rollback (`slo_violation`, `slo_violation_direct`,
   `formal_commit_gate_failed`, `donor_health`): the routable count differs from
   the starting one, or Z >= starting Z + `safescale.rollback_retry_z_margin`
   (registry, default 0.25 in Z units; hold reason `same_evidence`).

Other rollbacks (evidence gaps, hide failures, SM maintenance, observe mode,
pods gone) say nothing about capacity and only need step 1. A model without a
signal this tick is held (`no_signal`). Event: `safescale_rollback_hold:<m>:<reason>`
(was `safescale_rollback_backoff:<m>`). The evidence is cleared by the model's
next commit and replaced by its next rollback; a preemption records nothing.

With O1 active the unhide of a rollback is itself a routable-count change, so the
donor is additionally held until a whole window follows it (O1 donor rule).

### Configuration

* `safescale.rollback_retry_z_margin: 0.25` (registry and its params mirror).
* `TRE_SAFESCALE_ROLLBACK_BACKOFF_MS` still parses (invalid values still refuse the
  start) and is logged as deprecated; the overlay keeps `60000` for image rollback
  (older images read it).

### Risks

* Under steady load a model whose probe failed is not probed again until its
  Z or replica count changes - by design. Z is not monotonic in load; a model
  whose load falls to IDLE releases through the idle path, which this gate does
  not hold (it never held demand-driven or idle releases).
* A probe restored after a controller restart has no starting Z: the first window
  after its rollback becomes the reference (one extra window of hold).
* Repeated non-capacity rollbacks (for example a persistent scrape failure) can
  now repeat once per new window plus W; each such probe only hides pods (no
  request is cut) and is visible in the rollback-reason events.

Tests: `controller/tests/test_timer_cleanup_rollback_gate_20261002.py`.

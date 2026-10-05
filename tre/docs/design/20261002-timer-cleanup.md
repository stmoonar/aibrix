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
* `TRE_SAFESCALE_ROLLBACK_BACKOFF_MS` is ignored (a set value is logged, not
  validated) and no longer in the overlay (2026-10-06; an image rollback restores
  the backed-up Deployment with its env).

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

## 3. SafeScale early commit

### Before

The probe window W = min(max(2 x p95_e2e, 20 s), 60 s) after the hide
confirmation. A judged violation rolled back at once; otherwise the probe always
waited for the deadline, even when the evidence was complete and the hidden pods
were already idle - the probe pods stayed awake (and their GPUs unavailable to a
receiver) for the rest of W.

### Gate

Direct evidence path only (`safescale.evidence_source: direct`; the Redis path
keeps the deadline). On every poll before the deadline, after all rollback checks
of that tick ran unchanged, the probe commits when all of these hold on the same
poll:

| | Condition |
|---|---|
| (a) | at least `min_commit_samples` requests of the remaining pods judged, p95 available |
| (b) | the formal commit gates pass on the evidence covered so far (`_judge` in early mode: latency SLO, complete evidence of every remaining pod in this poll, fresh cluster view, KV-cache ceiling, Z tail >= tau_low); the evidence must cover up to the poll's read time instead of the deadline |
| (c) | the hidden pods have nothing in flight: vLLM `num_requests_running + num_requests_waiting` of each hidden pod (scraped in the same poll, never part of the evidence) known and 0, and the gateway in-flight count (`tre:v2:gw:inflight:<pod>` totals of the **live** plugin instances only - heartbeat in `tre:v2:gw:instances` at most `service_manager.sleep.instance_staleness_s` old, Redis TIME - read with the shared `tre_common.gateway_inflight` reader (2026-10-06: a field a dead instance left behind no longer blocks until its TTL); unknown without a live instance) known and 0 |
| (d) | see "Review fixes" (P2-3): `early_commit_min_grids` complete post-hide grids in the newest snapshot window (default 2) and max(those grids, p95 e2e, W / 2) since the hide confirmation |

In early mode nothing but a commit is acted on: an outcome that would extend,
wait, roll back or fail a gate leaves the probe probing, and the deadline decides
as before. The commit is the same `scale_down` of the hidden pods with the same
follow-up upscales; its decision carries `early_commit` {`elapsed_ms`, `samples`,
`planned_deadline_ms`, in-flight counts}, logged as the JSON event
`safescale_early_commit` and as the tick event
`safescale_early_commit:<m>:elapsed_ms=..:samples=..`.

### Configuration

Registry `safescale.early_commit: true`, `safescale.early_commit_min_grids: 2`
(and the params mirror). `false` = deadline only. The controller wires the hidden
pod scrape and the gateway reader only when enabled.

### Why rollback is not weakened

The rollback checks (preemption, hide failure, direct SLO violation, donor
health) run before the early check on every tick, exactly as before. An early
commit needs at least the evidence volume the deadline commit needs
(`min_commit_samples`) and passes the same gates.

### Risks

* Less elapsed time means fewer samples of slow phases (for example long decodes
  that complete later); (a) and (d) bound this, and min grids can be raised.
* The Z gate reads the snapshot tail, whose 30 s windows still include pre-hide
  grids early in the probe (as at a 20 s deadline); (d) requires one whole
  post-hide grid in the newest window.
* The gateway count includes fields of instances that stopped without clearing
  them; that only blocks an early commit (the deadline still decides).
* Each poll reads the hidden pods too (a few more scrapes per probe).

Tests: `controller/tests/test_timer_cleanup_early_commit_20261002.py`.

## 4. Dead timers removed

### CRIT scale-up cooldown (`scaling.scale_up_cooldown_enabled`)

C1 made the F4 hold of a CRITICAL receiver's scale-up opt-in; every shipped
registry had it off, because the C1 rescue-target bookkeeping (`rescue_bases`)
already keeps an unreflected scale-up from being repeated. The switch and its
`PlanConfig` field are removed. The F4 / O1 view-pending hold of a CRITICAL
scale-up remains only for the legacy one-step rescue (`rescue_max_step_ratio: 0`,
C1 off), which has no target bookkeeping. The registry key is no longer parsed
(2026-10-06): an old registry that still has it loads with the usual "unknown
key" warning; it is not in the shipped registry / params mirror.

### Band dwell (`TRE_DWELL_WINDOWS`, `TRE_DWELL_STATES`)

The D8 dwell (a band acts only after N consecutive new windows) had been off
(`1`) since the v1/paper alignment A5. `SignalState.apply_dwell`, its counters,
the `dwell_confirmed` receiver suppression in the planner and the model-state box
are removed. The environment variables are ignored (a set one is logged, not
validated, 2026-10-06); the overlay no longer sets `TRE_DWELL_WINDOWS` (images
before the cleanup default to `1` without it). `tre_common.dwell` stays: the
offline calibration and hold-out tools use it.

### Risks

None in behaviour with the shipped configuration (both were off). A registry or
environment that turned either on loses that hold; with O1 active, receivers
still need post-breakpoint evidence.

Tests: `controller/tests/test_band_dwell.py` (offline helper + "no controller
dwell"), `controller/tests/test_c1_deficit_scaleup_20261001.py`
(`test_scale_up_cooldown_key_is_ignored`), `controller/tests/test_action_cooldown.py`.

## Review fixes (independent review, 2026-10-02)

* **View-pending sources (P2-1).** The O1 view-pending gate read only
  `last_actions`, which holds completed scaling decisions: a probe-rollback unhide
  (no direction) and failed or partial SM calls were missing. Between a rollback
  unhide and the next view refresh, a LOW receiver could scale up again on the
  hidden-pod count. The gate now also reads `ActionQueue.view_changes()`: every
  routable-changing SM call (stamped after the answer, ok or not) with a hold rule -
  a wake / scale-up holds like "up", a sleep / hide / unhide like "down" (a CRITICAL
  receiver passes, LOW receivers and scale-downs wait). The later of the two sources
  per model decides.
* **View time is a lower bound (P2-2).** `ClusterView.fetched_ms` (response
  received) is an upper bound of the state's time, right for dating O1
  breakpoints but not for "the view shows this action". The view now also carries
  `state_ms`: the time the controller sent the request. Timestamps are never
  compared across machines: the SM's own `/v2/state` `fetched_ms` (SM clock; one
  cluster node runs about 160 s ahead of another) is kept on the view as
  `sm_fetched_ms` for reference only and decides nothing. The view-pending gate compares the
  action's completion with `state_ms`; O1 dating keeps `fetched_ms`.
* **No fresh view (P3-6): alert only, holds kept on purpose.** A view that stops
  refreshing happens only on failures (SM restart / crash, Redis down, network,
  slow Kubernetes API timing out `/v2/state`). The controller then stays
  conservative by design: the view-pending gate keeps holding LOW scale-ups and
  scale-downs of a model with a pending change (CRITICAL receivers pass); it does
  not fall back to F4, which would resume normal control without data. The cluster
  view task logs `cluster_view_stale` (age from the view's state time - the controller's
  request time, never the SM clock - and the last refresh error) once when the
  age exceeds `TRE_VIEW_STALE_PERIODS` (default 3) refresh periods, and
  `cluster_view_recovered` once a fresh view arrives. No control change.
* **Early commit evidence (P2-3).** Condition (d) of item 3 is now: the newest
  snapshot window holds `early_commit_min_grids` complete post-hide grids (default
  2, at least `scaling.min_evidence_grids`: the O1 warm rule), and the time since
  the hide confirmation is at least max(those grids, the donor's p95 end-to-end
  latency, W / 2). The remaining pods' concurrency needs about one end-to-end
  latency to reach its new level; an early commit at most halves W.
* **Stalled probe (P2-4).** `insufficient_evidence:stalled` (traffic in flight, not
  one request completed within W) is a capacity rollback for the item-2 gate.
* **Permanent hold under unchanged evidence (P3-5).** After a capacity rollback, a
  model whose Z and routable count stay the same is not probed again - by design:
  the same load gives the same answer. The unhide restarts the O1 EMA, so the first
  post-rollback windows are noisier than the steady EMA; a Z excursion can cross
  the 0.25 margin through noise alone and allow one more probe (which is then
  judged by the full SafeScale gates).
* **Oscillation (P3-7).** Two models trading a replica back and forth: each hop
  needs the receiver's post-change evidence (min_evidence_grids) and the donor's
  whole post-change window, so a reverse hop waits at least one whole window after
  the forward one (tested).

Tests: `controller/tests/test_timer_cleanup_review_20261002.py`.

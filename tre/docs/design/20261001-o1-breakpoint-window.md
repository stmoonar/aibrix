# O1: breakpoint-aware decision window (2026-10-01)

User decision (2026-10-01): replace the ADR-0013 onset warmup guard with a
**breakpoint-aware effective window**. Branch `feat/o1-breakpoint-window-20261001`
(on top of C1, `feat/c1-deficit-scaleup-20261001` @ 57510356). ADR-0015 in
`docs/refactor/DECISIONS.md`.

## Problem

TSS = window token total / Q, Q = running + lambda * waiting, over a 30 s window that
the gateway fills in 10 s grids. After a breakpoint - load onset, or a change of the
model's routable replica count - the window mixes two regimes:

* at **onset** the numerator holds 1/3, 2/3 of a window's tokens while Q sits at qmin
  (light load): Z dips by the fill fraction and reads CRITICAL (ADR-0013: a 1 req/s
  trickle took dsqwen-7b 1 -> 3 in 20 s). The guard fixed that by ignoring receivers
  until the whole window lies after the onset: every idle -> loaded scale-up waited a
  full window (30 s after the onset window, 40-50 s after the load began);
* after a **scale-up** the window still describes the old replica count, and the EMA
  carries older windows on top; C1 bridged that with `rescue_settle_ema_k * ema_tau`
  (20 s) after the window started past the scale-up, i.e. ~50 s.

## O1

**Breakpoint** `t_break = max(onset, last routable-count change)`.

* onset: unchanged (`SignalState.observe_traffic`): the `window_end` of the first
  window with tokens after an idle window; an idle window clears it.
* routable change (`SignalState.note_routable`): the count is the SM fleet view's
  (awake and not hidden). A change is dated by `ActionQueue.routable_changes()` - the time
  the controller's last SM call that can change the model's routable set (scale, wake,
  sleep, hide, unhide, receiver target) *returned*, ok or not - when that falls after the
  previous view showing the old count; else by the fetch time of the view showing the new
  count (`ClusterView.fetched_ms`, stamped after the response). Both are at or after the
  real change. A C1 rescue target's `done_ms` is never used: a target covered by a probe
  preemption is stamped when planned, before its unhide ran (review P1). Plus
  `breakpoint_margin_ms` (1 s) before rounding up to the grid: the SM writes the routable
  label (and route generation) before it answers - checked in `_commit_one_wake` /
  `write_binding_annotations` - so the margin only covers the gateway's pod-informer
  propagation. A view older than one already seen is ignored. A first observation (start /
  restart) takes the latest stamp, if any, but such a guessed date never settles C1.

**Effective window** `(eff_start, window_end]`, `eff_start` = first gateway boundary at
or after `t_break` (the grid holding the breakpoint is excluded, only complete grids).
When `eff_start <= window_start` it is the whole window and the computation is exactly
the pre-O1 one (bit-identical, tested). Otherwise:

* numerator = suffix token total x `W / span` (a whole-window equivalent; Z keeps the
  calibrated theta's units);
* Q = the instant average over the suffix's ticks (expected-samples divisor = suffix
  grids) - the store builds every grid-aligned suffix of the window from the docs the
  window read already fetched (`MetricsStore(suffix_period_ms=10 s)`,
  `ModelWindowMetrics.suffix_windows`; no extra redis round trip);
* the model's EMAs (TSS and alternative signals) restart at the breakpoint (once, on
  the first tick that sees it inside the window); request / token rates of the context
  are the suffix's.

**Evidence** (`warm`): at least `min_evidence_grids` complete grids (2 = 20 s) and
`min_evidence_requests` completed requests (0 = off) after the breakpoint. Until then
the model decides nothing this tick: it is neither receiver nor donor (planner), its
SafeScale observation Z is the whole window's raw value (no EMA advance), and its
`ModelStateBox` state is `unconfirmed`.

**Scale-down stays cautious**: a model whose window still holds a breakpoint
(`signal_full_window` False) is never a donor (HIGH, IDLE, middle zone); scale-downs wait
for a whole window after the breakpoint, as before (the F4 cooldown and SafeScale are
unchanged). SafeScale's Z after a hide (a breakpoint) is the raw whole window until the
post-hide evidence is warm, then the post-hide window's - both at least as strict as the
pre-O1 EMA, which weighed the pre-hide windows.

**IDLE donors (Q3, 2026-10-06)**: the exception to the rule above. After its own
scale-down an IDLE model used to wait a whole window after that breakpoint before its next
step, so an IDLE surplus left about one step per 30-40 s. An idle window - no prefill or
decode token and no running / waiting request in the whole current window - is the same
evidence at any replica count, so the breakpoint changes nothing about it. An IDLE model
whose context carries `window_idle: true` is exempt from the donor hold, and an IDLE donor
gives its **whole surplus** in one decision: an `idle_proactive_immediate` shrink goes
straight to the floor (bounded by the SM `floor_headroom`), a relay to a CRITICAL or LOW
receiver may take all of it (bounded by the receiver's need and the headroom). This is a
code rule for IDLE only; `scaling.donor_surplus_release` keeps its value (false) and now
only affects HIGH donors. HIGH donors keep the O1 hold and one step per tick (their
evidence - a throughput level - does depend on the replica count). `window_idle` is
current-window evidence only (I4): the tick sets it from this tick's serving window with
tokens known and every serving pod scraped; a held context (`tokens_missing`), a
scrape-stale context and a tokens-missing window never carry it, so those stay held.

**CRITICAL receivers on free capacity (H3, 2026-10-06)**: the receiver hold is narrowed to
what it guards against - an over-scale decided on a window that still describes the old
regime. On a free GPU such an over-scale costs one wake (slept again by a later donor
decision); waiting for 2 grids is then a timer in disguise (pilot E1 drift1: 8b held +36 ->
+68 s at Z ~ 0.3, KV 1.0, free GPUs). A CRITICAL receiver whose `signal_warm` is false is
**not held** when its queue rose since the breakpoint (`o1_queue_rise` in the tick
context; **2026-10-07: no longer required**, see below). It then takes **free capacity only** - its sleeping bindings on free GPUs, free
slot groups (no donor, no middle-zone SafeScale probe, no TP same-slot shrink, no defrag) -
and **one replica per decision** (`max(1, breakpoint_partial_max_step)`: Z is the held
window's old-regime value). An earlier C1 rescue target O1 has not settled yet (the C1
basis of `_rescue_bases`, landed or not), or a fleet view older than the last action, holds
it (`receiver_o1_exempt_pending`): one stale window buys one wake (2026-10-07; before, a
landed target released it and the next step re-asked the target from the new count with
the old window's Z, climbing to `max_awake`). LOW sleeping-capacity wakes carry no rescue
target and do not block it. Only the O1 evidence holds are exempt (`no_complete_grid`,
`no_suffix`, `evidence_grids`, `evidence_tokens`, `evidence_requests`); a held context
(`tokens_missing`), `scrape_stale` and a hold fallback stay held. Exempt receivers are planned
after warm TSS and saturation receivers. Everything else keeps O1: steps from donors, every
scale-down, LOW receivers.

**2026-10-07: queue rise dropped from the exemption.** `queue_rise` compares `waiting` on
the pods present in both samples only; after a wake the old pods' waiting drains to 0 while
their running stays ~150/pod, so it could never hold after a wake and the receiver sat out
the full 2 grids (~22 s) with free GPUs. At the hot-segment onset the window has no
post-breakpoint sample (`no_complete_grid`), so it was None there too. A CRITICAL receiver
on free capacity is now exempt whenever its previous step has settled. `o1_queue_rise` is
still computed and logged (decision snapshot, the exempt event's `queue_rise=yes|no`); the
paragraph below describes that measurement.

*Queue rising* (state only, no timer): every O1-tracked read records the model's queue
sample - the routable pods' (serving window minus hidden probe pods) newest gateway instant
samples of the window's last grid: `q = sum(running) + lambda_wait * sum(waiting)` (the TSS
queue term on instant values, no qmin) and `waiting`, stamped with the newest pod stamp;
the last 8 stamps are kept. On a held window the current sample takes only pods stamped
**after** the breakpoint; it rises when `q` or `waiting` is strictly higher than the newest
recorded sample stamped **at or before** the breakpoint. No post-breakpoint sample (the
window predates the change), no baseline (restart, history gone) or not higher: held. A
held context (tokens missing) never carries the evidence. The gauges do not wait for
completions, so the first post-breakpoint sample is evidence the token window cannot give
for another 1-2 grids. Onset breakpoints count too (idle -> hot: the onset window's sample
is the baseline, the next window's the evidence; the saturation rescue is unchanged and
takes precedence). Known false positive: right after a scale-up the new pods' running
count grows to its steady level, so `q` can rise while the fleet already copes; the cost
is one wake on a free GPU, bounded by the one-step cap and `max_awake_replicas`.

Events: `receiver_o1_exempt_free_gpu:<model>:<hold reason>:planned=<n>:bp=<ms>:q=<base>-><now>:waiting=<base>-><now>:sample_ms=<base>-><now>:queue_rise=<yes|no>`
(queue fields `none` without a rise)
when the exemption planned a step; otherwise (no free capacity, nothing needed, no rescue
this tick) the usual `receiver_held_breakpoint_window:<model>:<reason>`. Decision snapshot
`model_states.<model>.o1_queue_rise` (when present).

**C1 settle**: a rescue target counts as reflected once the model is warm and its
settle breakpoint (`signal_settle_ms`: the onset, or a count change seen between two
views of this process - not a first-observation date, review P2-a: after a restart the
restored in-flight target is stamped done=now while the SM may still be waking) is at or
after the target's `done_ms`: its Z then describes the new replica count with a fresh EMA, so the
`rescue_settle_ema_k` extension is not needed. A target that changed nothing (all parts
failed) never moves the breakpoint and settles by the old window-start rule, which stays
as the fallback. **Q2 (2026-10-06)**: while O1 tracks the model (`o1_routable_tracked`)
the window-start rule runs with an effective `k = 0` for a target this process dispatched
(the window must only start after `done_ms`): the breakpoint restarts the EMA and the
evidence gate holds the receiver, so the EMA-lag extension only delays a decision on
evidence that is already new. `rescue_settle_ema_k * ema_tau` applies only as the
fallback: O1 does not track the model (O1 off / suspended, no fleet view, stale-held
context, hold fallback), or the target's `done_ms` is not an observed completion (restored
after a restart - review P2-a; covered by a probe preemption whose unhide is still to
come). The registry key stays as that fallback. In-flight protection and the base / covered bookkeeping are unchanged.

**Review round 2 (2026-10-01)**:

* *low-QPS evidence* (P2-1): tokens count at request completion, so a 20 s suffix with one
  short request done and long ones running can read Z ~ 5 %. `min_evidence_requests`
  defaults to 3 (can the window decide at all), and a C1 rescue decided on a partial
  window **with fewer than `breakpoint_lowevidence_requests` (10) completed requests**
  adds at most `breakpoint_partial_max_step` (1) replica (event
  `rescue_low_evidence_step`); with 10+ requests the whole deficit at once (ratio /
  step_pods caps). At 0.1 rps a 20 s suffix holds ~2 completions (< 3: no decision); a
  partial-window misjudgment needs 3-9 completions and then adds at most +1 replica per
  breakpoint (instead of C1's +4).
* *starvation* (P2-2): after `breakpoint_hold_max_windows` (6 windows = 60 s; a single
  change holds at most 3, an onset 2) consecutive held windows a receiver decides on the
  whole window again (EMA advanced, event `breakpoint_hold_fallback`); donors still need
  a clean window.
* *one clock* (P2-3): O1 compares the controller clock (SM-call returns, view fetch
  times) with the gateway's 10 s doc stamps (its clock rounded to the boundary) - it
  **assumes NTP-synchronised nodes**. `tre_controller.gateway_clock` checks at start and
  every `gateway_clock_check_s` (60 s): `lag = now - newest instant stamp` must lie in
  `[-tolerance, 2 * period + tolerance]` (tolerance `gateway_clock_tolerance_ms`, 2 s).
  The gateway's write phase is unknown, so only an offset beyond that phase window is
  visible that way (75's +160 s is). Review round 3: gateway images from 2026-10-01 write
  `written_ms` (their wall clock at the write) into every doc and keep one doc per
  boundary; the check then watches one pod's newest doc until a new one appears and
  measures `offset = written_ms - write time on the controller clock` to ~0.2 s, phase
  independent (the stamp-lag bounds stay the fallback for older gateways). **This needs
  the gateway-plugins image rebuilt in the same release.** A violation suspends O1 -
  whole windows, the onset guard, C1's own settle rule, event
  `breakpoint_window_suspended:<reason>` every tick - until 3 good checks.
* O1 off or suspended, or a receiver on the hold fallback: C1 settles by its window-start
  rule only (`signal_settle_ms` None), as without O1 (review round 3 P1 / P2-1).
* A persisting low-QPS misjudgment can add +1 per breakpoint, i.e. about every 30 s
  (each +1 is a new breakpoint: 20 s evidence + the wake). Not gated further (a "two
  consecutive low-evidence CRITICAL" rule would land the second decision on a whole
  window, which is uncapped by design); bounded by the scaling cap.
* a hint dates a change only in its own direction (`routable_changes()` carries +1 / -1);
  a held receiver's event names the reason (`receiver_held_breakpoint_window:<model>:<reason>`).
* **restart under load**: the controller's first window with tokens is an onset (as with
  the ADR-0013 guard): EMA restarted, receivers held 20 s and donors 30 s after a restart.

## Timeline (10 s grid, decision on the window's rescue tick ~ end + 8 s)

| event | pre-O1 | O1 |
|---|---|---|
| idle -> load, first CRITICAL decision | onset + 30 s | onset + 20 s (+1 only on < 10 requests) |
| scale-up done -> next decision on the new count | window start >= done + 20 s (~done + 60-70 s) | ceil(done) + 20 s (~done + 30-40 s) |
| scale-down eligibility after any breakpoint | F4: window start >= done | whole window after the breakpoint |

Replay (decision offsets from the logs, wake 3 s from `sm.log`), time after the load
start. These loads complete hundreds of requests per grid, so the evidence gate never
caps them: 1 -> 2 and 1 -> 4 happen in the same step.

| run | pre-O1 + C1: 1 -> 4 | O1 (evidence-gated cap): 1 -> 4 | an always-on partial cap: 1 -> 2 / 3 / 4 |
|---|---|---|---|
| C-crit | 53.7 s | 43.7 s | 43.7 / 83.7 / 123.7 s |
| F4-161915 | 55.7 s | 45.7 s | 45.7 / 85.7 / 125.7 s |
| F4-161559 | 50.5 s | 40.5 s | 40.5 / 80.5 / 120.5 s |

Replay on the 2026-10-01 verify evidence (ctrl_ticks deconvolved per grid; the pre-O1
replay reproduces the logged first scale-up of all 6 episodes that logged one): first
CRITICAL decision after the onset window 30 s -> 20 s in all 7 loaded episodes (C-crit,
F4 x2, E-14b, D-safescale-n1, A-smoke-tre / A-smoke-apa 8b), i.e. 50.7 -> 40.7 s after
the load start for C-crit (C1 target 4 at once, Z = 0.017). Light models at onset (A-smoke-tre 7b): the raw filling window dipped to
Z = 0.20 (< tau_crit 0.56, the ADR-0013 false CRITICAL); O1's first decided Z = 3.78.

## Configuration (registry `scaling:`, restart-to-apply)

| key | default | meaning |
|---|---|---|
| `breakpoint_window` | true | O1 on; false = pre-O1 (the onset guard then always applies) |
| `onset_warmup_guard` | false | also apply the ADR-0013 guard on top of O1 |
| `breakpoint_margin_ms` | 1000 | added to a routable change time before grid rounding |
| `breakpoint_partial_max_step` | 1 | C1 step cap on a low-evidence partial window (0 = none) |
| `breakpoint_lowevidence_requests` | 10 | below this many completed requests a partial window is capped |
| `breakpoint_hold_max_windows` | 6 | held windows before a receiver falls back to the whole window (0 = never) |
| `gateway_clock_tolerance_ms` / `gateway_clock_check_s` | 2000 / 60 | same-clock check (0 s = off) |
| `min_evidence_grids` | 2 | complete grids after the breakpoint before a scale-up |
| `min_evidence_requests` | 3 | completed requests the post-breakpoint window needs |

Pre-O1 behaviour: `breakpoint_window: false` (review P2-b: turning O1 off never leaves
both guards off). Controller
images before O1 ignore the keys (C1 images warn "unknown keys"). The schema `v1` store
and unaligned (free-running) windows have no suffixes: O1 then waits for a whole clean
window (reason `no_suffix`).

Observability: decision snapshot `model_states` gains `signal_full_window`,
`signal_breakpoint_ms`, `signal_window_start_ms`, `signal_evidence_grids`,
`signal_hold_reason` (`no_complete_grid` / `evidence_grids` / `evidence_tokens` /
`evidence_requests` / `no_suffix`); planner event `donor_suppressed_breakpoint_window`;
startup log `breakpoint_window_config`.

## Signal validity (2026-10-04, I3 / I4)

I3: every observation carries its own validity; data older than the window is unknown,
not 0 and not the last value. I4: a model is a donor (scale-down or release) only on a
level computed from tokens observed in the current window. Both reuse the O1 donor gate
(`signal_full_window: false` + `signal_hold_reason`), no new timer:

* **held context** (`signal_hold_reason: tokens_missing`): the window has no token data,
  `PaperStateCache` holds the last context (at most `paper_stale_max_windows`, 3);
* **stale scrape** (`scrape_stale`): the gateway writes `scraped_ms` (its wall clock at
  the pod's last successful /metrics fetch). `MetricsStore.read_model_window` leaves out
  a pod whose newest in-window `scraped_ms` is before the window's (or O1 suffix's) read
  start - both values are gateway clock. A serving pod left out puts the model on the
  donor gate: the remaining pods may be the light ones. No valid pod left: tokens None
  (the held / UNKNOWN path). An old gateway (no `scraped_ms` on any doc of the model
  in the window) keeps every pod. Event `scrape_stale:<model>:<pods>`.

**Deliberate exception (scale-up stays aggressive):** a receiver on such a level still
acts - a held CRITICAL on its last value, a scrape-stale CRITICAL on its valid pods. The
rescue step is capped like a thin partial window: `breakpoint_partial_max_step` (1)
replica per decision (`rescue_low_evidence_step`). A wrong +1 costs one wake and is
slept again by the next donor decision on full evidence; withholding it from a model
that may be overloaded costs SLO. The ModelStateBox reports these models UNCONFIRMED
(commit revalidation keeps the plan).

## Not affected

The TSS definition (`tre_common.tss`), the offline calibration (`calibration/`,
`deploy/scripts/rewindow_from_raw.py`, `calibration_decision.py`: they call
`TRSComputer.compute` on whole windows, unchanged) and theta. Steady state (no breakpoint
inside the window) is bit-identical.

## Risks

* A model whose routable count keeps changing gets no decision for up to
  `breakpoint_hold_max_windows` windows (then the whole-window fallback).
* A partial window with 3-9 completions still steps +1 at a time (low-QPS loads only).
* Clock offsets inside the gateway's write phase are invisible to the same-clock check.
* The onset is still "first window with a completed request": a long prefill (14b, E-14b)
  shows queue 30 s before its first completion. An activity onset (Q > 0) would gain
  another 10-20 s (replay: C-crit 40.7 -> 30.7 s) but changes the shared idle predicate;
  not done.
* 2 grids is less evidence than 3: noisier first decisions at very low request rates.
  `min_evidence_requests` is the knob.
* A view without `fetched_ms` (synthetic / offline) dates a change at the snapshot's
  window end (tests only; the live view always carries it).

## Follow-up (not done): activity onset

The replay shows another ~10 s (C-crit 40.7 -> 30.7 s after the load start; E-14b up to
20 s) if the onset is the first window with anything in flight rather than the first
window with a completed request: long prefills keep Q > 0 for 10-30 s before the first
token total appears. Concretely:

* add `window_is_active(prompt, generation, running, waiting)` = tokens > 0 or
  running + waiting > 0 in `tre_common.tss` next to `window_is_idle`;
* `SignalState` keeps a separate O1 onset: recorded at the first *active* window, cleared
  only by a window that is neither active nor carrying tokens; `breakpoint_ms` uses it;
* leave `window_is_idle` itself unchanged. Its callers and what they would see:
  `TssEma.update(idle=...)` (online `TRSComputer`, offline `smooth_series` /
  calibration recompute - the EMA reset and theta stay as calibrated),
  `SignalState.observe_traffic` (the ADR-0013 onset, the idle EMA / dwell reset - a
  Q-only window keeps resetting them, so the first active window's Z stays undefined as
  today), `sources._thresholded_signal` (alt-signal EMA reset),
  `tick.compute_model_signal` (`has_traffic`). Only the O1 breakpoint would move.
* Risks: a request stuck in flight keeps a model "active" (no new onset after it really
  idles); the first post-onset grids hold queue but few completions, so the numerator
  over 2 grids is low - needs `min_evidence_requests` or a "first completion inside the
  effective window" condition, plus a replay of light-load onsets (A-smoke 7b / 14b) to
  rule out a false CRITICAL.

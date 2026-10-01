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

**C1 settle**: a rescue target counts as reflected once the model is warm and its
settle breakpoint (`signal_settle_ms`: the onset, or a count change seen between two
views of this process - not a first-observation date, review P2-a: after a restart the
restored in-flight target is stamped done=now while the SM may still be waking) is at or
after the target's `done_ms`: its Z then describes the new replica count with a fresh EMA, so the
`rescue_settle_ema_k` extension is not needed. A target that changed nothing (all parts
failed) never moves the breakpoint and settles by the old window-start rule, which stays
as the fallback. In-flight protection and the base / covered bookkeeping are unchanged.

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
  visible (75's +160 s is). A violation suspends O1 - whole windows and the onset guard,
  event `breakpoint_window_suspended:<reason>` every tick - until 3 good checks.
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

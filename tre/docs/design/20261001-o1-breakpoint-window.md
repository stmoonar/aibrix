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
  (awake and not hidden). A change is dated by the latest completion time of the
  controller's own actions on the model (last scale / hide done, C1 rescue target done)
  that falls after the previous view showing the old count, else by the fetch time of
  the view showing the new count (`ClusterView.fetched_ms`, stamped after the response).
  Both are at or after the real change, never before. A first observation (start /
  restart) takes the latest done time, if any (the restored C1 scale memory).

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
breakpoint is at or after the target's `done_ms` (the change carries that time, or a
later one): its Z then describes the new replica count with a fresh EMA, so the
`rescue_settle_ema_k` extension is not needed. A target that changed nothing (all parts
failed) never moves the breakpoint and settles by the old window-start rule, which stays
as the fallback. In-flight protection and the base / covered bookkeeping are unchanged.

## Timeline (10 s grid, decision on the window's rescue tick ~ end + 8 s)

| event | pre-O1 | O1 |
|---|---|---|
| idle -> load, first CRITICAL decision | onset + 30 s | onset + 20 s |
| scale-up done -> next decision on the new count | window start >= done + 20 s (~done + 60-70 s) | ceil(done) + 20 s (~done + 30-40 s) |
| scale-down eligibility after any breakpoint | F4: window start >= done | whole window after the breakpoint |

Replay on the 2026-10-01 verify evidence (ctrl_ticks deconvolved per grid; the pre-O1
replay reproduces the logged first scale-up of all 6 episodes that logged one): first
CRITICAL decision after the onset window 30 s -> 20 s in all 7 loaded episodes (C-crit,
F4 x2, E-14b, D-safescale-n1, A-smoke-tre / A-smoke-apa 8b), i.e. 50.7 -> 40.7 s after
the load start for C-crit (C1 target 4 at once, Z = 0.017). Light models at onset (A-smoke-tre 7b): the raw filling window dipped to
Z = 0.20 (< tau_crit 0.56, the ADR-0013 false CRITICAL); O1's first decided Z = 3.78.

## Configuration (registry `scaling:`, restart-to-apply)

| key | default | meaning |
|---|---|---|
| `breakpoint_window` | true | O1 on |
| `onset_warmup_guard` | false | also apply the ADR-0013 guard (`TRE_SIGNAL_WARMUP_MS`) |
| `min_evidence_grids` | 2 | complete grids after the breakpoint before a scale-up |
| `min_evidence_requests` | 0 | completed requests the post-breakpoint window needs |

Pre-O1 behaviour: `breakpoint_window: false`, `onset_warmup_guard: true`. Controller
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

* A model whose routable count keeps changing (a crash-looping pod, repeated probes /
  rollbacks) gets no decisions while every window holds a breakpoint.
* The onset is still "first window with a completed request": a long prefill (14b, E-14b)
  shows queue 30 s before its first completion. An activity onset (Q > 0) would gain
  another 10-20 s (replay: C-crit 40.7 -> 30.7 s) but changes the shared idle predicate;
  not done.
* 2 grids is less evidence than 3: noisier first decisions at very low request rates.
  `min_evidence_requests` is the knob.
* A view without `fetched_ms` (synthetic / offline) dates a change at the snapshot's
  window end (tests only; the live view always carries it).

# P5 Paper vs Implementation Notes

Date: 2026-07-04
Environment: remote server 76, `/data/nfs_shared_data/xxy/aibrix`

## TRS Signal Migration

The code keeps the legacy controller name `TRS`; the plan notes this is the same metric as paper `TSS`.

The migrated `TRSComputer` intentionally preserves the frozen upstream implementation's replica correction:

```python
TRS_raw = TRS_raw * assigned_replicas / routable_pods
```

This correction is applied after `Y_m / Q_ctl` and is documented in the frozen upstream `trs.py` as matching the existing `main.py` behavior. It may not appear as a separate term in the paper derivation, so it is recorded here as an implementation contract rather than changed during migration.

The saturation guard formula is preserved as `Gamma_m = (Y_m(t) - Y_m(t-1)) / (Q_ctl(t) - Q_ctl(t-1))`, with saturation only when `Q_ctl >= qsat` and `abs(Gamma_m) <= epsat` for `Hsat` consecutive windows.

### TRS EMA is a shared, per-model, wall-clock time-constant filter (S1.3 / ADR-0011, 2026-07-06)

Contract: the TRS EMA is **not** a fixed-alpha-per-step filter. It is a wall-clock
time-constant filter — `decay = exp(-dt_ms / ema_tau_ms)`, `ema = decay*prev + (1-decay)*raw`
— where `dt_ms` is the delta of the metrics window's `window_end_ms` (data time, not
scheduler wall-clock). Smoothing strength is set by `ema_tau_ms` alone and is decoupled
from the refresh cadence, so shortening the window / speeding refresh (S1.2) does not
change the smoothing. The EMA advances **at most once per distinct `window_end_ms`**
(a per-window dedup): rescue (5s), fairness (10s), and safescale all read the same shared
snapshot between metrics refreshes, and a single shared `TRSComputer` per model
(`SignalState`) must not over-advance on duplicate reads. **One EMA per model** — rescue
and fairness share it (one window, one theta, one EMA).

Correction to the migration record: prior to S1.3 the live path constructed a fresh
`TRSComputer` every tick with no EMA restore, so live `TRS == TRS_raw` and `ema_alpha`
had no live effect (the only smoothing was the 60s tumbling window). See ADR-0011.
`ema_alpha` is retained only for the offline legacy path (when `ema_tau_ms` is unset and
no `window_end_ms` is supplied, the fixed-alpha branch is byte-identical to the frozen
`legacy_trs` golden). EMA state is in-process only (no Redis persistence); on restart it
re-seeds from raw and reconverges within ~tau.

Still-divergent (recorded, not yet reconciled): `SaturationGuard`/gamma is dead in the
live control path (tick.py uses the direct `Q_ctl >= qsat` threshold, never the guard);
and `r3_grid.py`'s `trs` CSV column is within-cell EMA'd while the live signal was raw —
R3/S1.4 must make r3_grid replicate the live EMA semantics before refitting theta.

## Planner Migration

The frozen planner had two branches: the newer `paper_state` branch and a legacy raw-TRS fallback. P5 keeps the paper-state path and deliberately drops the legacy raw-TRS fallback required by `REFACTOR_PLAN.md`.

When migrated planner input is incomplete (`UNKNOWN` state or missing `Z_m` outside IDLE), `PlanResult.dropped_legacy_raw_trs` is set and no fallback action is emitted. This makes the removed behavior explicit for logs/tests instead of silently preserving the old branch.

The first planner slice preserves the paper-path delta semantics and SafeScale probe metadata. The TP-aware planner-defrag slice now layers cached cluster-view input onto the pure planner for 2-card CRITICAL receivers: complete empty slot, allocator `plan_defrag`, or explicit `capacity_blocked` event.

The plan also calls for shrinking HIGH same-slot halves before allocator defrag. That branch remains pending because it needs slot-aware donor selection to be integrated with SafeScale and concrete model occupancy; this slice does not silently approximate it with model-level HIGH shrink decisions.

## SafeScale Migration

The migrated SafeScale slice intentionally separates state-machine decisions from the old thread, Redis, and service-manager side effects. It preserves the frozen implementation's key commit guard semantics: immediate rollback on TTFT/TPOT SLO violation, deadline-based commit only when tail latency is OK including observations restored from probe journal, tail `z_m` is not below `tau_low` for traffic-bearing probes, and observed normalized GPU cache does not exceed 0.8.

The old implementation combined probe monitoring, HTTP scale calls, runtime observation capture, and file-backed persistence in one module. P5 records this as an implementation split rather than a formula change: the new state machine emits data-only commands for later queue wiring and accepts an injected store protocol for restart recovery.

### SafeScale window vs the shared metrics window (S1 N2, architect-ruled 2026-07-06)

There are **two distinct, orthogonal windows** and they must not be conflated:

1. `metrics_window_ms` (30s, sliding, ends at now) — the per-observation aggregation
   history. Each `ProbeObservation` carries ttft/tpot p95 and z_m computed from the
   single shared `snapshot_box` (same window / theta / EMA as rescue and fairness, ADR-0011).
2. `SafeScaleConfig.default_window_ms` (60s) — the wall-clock **probe deadline**:
   `deadline = hide_start + default_window_ms`. safescale_task polls the shared snapshot
   every `probe_poll_seconds` (2s) and appends an observation each tick.

**Invariant (hard, enforced at startup):**
`default_window_ms · (1 − hq) ≥ metrics_window_ms`. The commit gate (`_summarize_tail`)
inspects only the tail (`hq`=0.25 fraction) of observations; those tail observations'
metrics windows must be **fully post-hide**. At defaults 60000·0.75 = 45000 ≥ 30000
(margin 15s; refined form including one refresh interval is 45s ≥ 35s). This is enforced
in `ControllerConfig.from_env` — setting e.g. `SAFE_SCALE_DEFAULT_WINDOW_MS=15000` with a
30s metrics window now raises, instead of silently diluting the commit gate with pre-hide
traffic. For `hq ≥ 1` (absolute tail count) the tail span is `hq · probe_poll_seconds`.

**Direction-of-error note:** per-observation `_violates_slo` on the early *blended* windows
(right after a hide, the sliding 30s window still contains mostly pre-hide traffic) and the
count-based tail under sparse observations can only ever cause **extra rollbacks, never a
false commit** (latency is an AND, z_m a min over the tail). Shortening `metrics_window_ms`
strictly **speeds up** rollback detection — a benefit, not a conflict. No SafeScale logic
or warmup-discount change was needed for S1; the tail-only gate already excludes the blended
early observations from the commit decision.

**Dead-config landmine:** `SafeScaleConfig.min_window_ms` (15000) / `max_window_ms` (300000)
are currently **unused** — only a `min ≤ max` parse check exists; no code path sets a probe
deadline from them (they are a placeholder for the paper's adaptive probe window). They are
intentionally NOT guarded at startup (guarding would reject valid current defaults). Whoever
later wires the adaptive window must make the effective window satisfy the invariant above —
with current parameters the floor is `metrics_window_ms / (1 − hq)` = 40s, so `min_window_ms`
= 15000 must be raised (or the clamp re-expressed in terms of `metrics_window_ms`) at that time.

## Service Manager Client Gap

The controller client now targets service-manager v2 for state, target replicas, and routable hidden pods. Defrag remains an explicit integration gap: the planner and ActionQueue can represent `DefragAction`, but service-manager v2 currently has no `/v2/defrag` endpoint. Until P4/P5 adds that endpoint, controller dispatch returns a structured unsupported result instead of silently pretending the migration ran.

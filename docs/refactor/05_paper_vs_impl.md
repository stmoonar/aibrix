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

## Planner Migration

The frozen planner had two branches: the newer `paper_state` branch and a legacy raw-TRS fallback. P5 keeps the paper-state path and deliberately drops the legacy raw-TRS fallback required by `REFACTOR_PLAN.md`.

When migrated planner input is incomplete (`UNKNOWN` state or missing `Z_m` outside IDLE), `PlanResult.dropped_legacy_raw_trs` is set and no fallback action is emitted. This makes the removed behavior explicit for logs/tests instead of silently preserving the old branch.

The first planner slice preserves the paper-path delta semantics and SafeScale probe metadata. The TP-aware planner-defrag slice now layers cached cluster-view input onto the pure planner for 2-card CRITICAL receivers: complete empty slot, allocator `plan_defrag`, or explicit `capacity_blocked` event.

The plan also calls for shrinking HIGH same-slot halves before allocator defrag. That branch remains pending because it needs slot-aware donor selection to be integrated with SafeScale and concrete model occupancy; this slice does not silently approximate it with model-level HIGH shrink decisions.

## SafeScale Migration

The migrated SafeScale slice intentionally separates state-machine decisions from the old thread, Redis, and service-manager side effects. It preserves the frozen implementation's key commit guard semantics: immediate rollback on TTFT/TPOT SLO violation, deadline-based commit only when tail latency is OK including observations restored from probe journal, tail `z_m` is not below `tau_low` for traffic-bearing probes, and observed normalized GPU cache does not exceed 0.8.

The old implementation combined probe monitoring, HTTP scale calls, runtime observation capture, and file-backed persistence in one module. P5 records this as an implementation split rather than a formula change: the new state machine emits data-only commands for later queue wiring and accepts an injected store protocol for restart recovery.

## Service Manager Client Gap

The controller client now targets service-manager v2 for state, target replicas, and routable hidden pods. Defrag remains an explicit integration gap: the planner and ActionQueue can represent `DefragAction`, but service-manager v2 currently has no `/v2/defrag` endpoint. Until P4/P5 adds that endpoint, controller dispatch returns a structured unsupported result instead of silently pretending the migration ran.

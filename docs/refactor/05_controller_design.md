# P5 Controller Design

Date: 2026-07-04
Environment: remote server 76, `/data/nfs_shared_data/xxy/aibrix`

## Config Contract

The first P5 slice introduces `tre_controller.config` as the single environment parsing boundary for the Python controller migration. Later P5 modules should receive a `ControllerConfig` instance instead of calling `os.getenv` directly.

Centralized values:

- Redis and service-manager endpoints: `TRE_REDIS_URL`, `TRE_SERVICE_MANAGER_URL`.
- Registry and runtime state paths: `TRE_REGISTRY_PATH`, `TRE_RUNTIME_STATE_DIR`.
- Loop cadence: `TRE_MONITOR_INTERVAL_SECONDS`, `TRE_RESCUE_INTERVAL_SECONDS`, `TRE_FAIRNESS_INTERVAL_SECONDS`.
- Metrics windowing: `TRE_METRICS_WINDOW_MS`, `TRE_INSTANT_SAMPLE_INTERVAL_MS`, `TRE_PERCENTILE_MODE`.
- P5 ablation switches: `ENABLE_TRE_SCALING`, `TRE_ABLATION_DISABLE_FAST_LOOP`, `TRE_ABLATION_DISABLE_SAFESCALE`.
- Signal source switch: `TRE_SIGNAL_SOURCE=zm|latency_p95|queue_len|kv_cache`.
- Legacy controller constants found in the frozen upstream controller: `PROACTIVE_RELEASE_MIN_TRS` and all `SAFE_SCALE_*` knobs.

Validation rules:

- Boolean env values accept `1/0`, `true/false`, `yes/no`, and `on/off`.
- Loop intervals, metric windows, SafeScale windows, and timing constants must be positive.
- `TRE_PERCENTILE_MODE` is restricted to `bucket_upper` or `interpolated`; the P5 default is `bucket_upper`.
- `TRE_SIGNAL_SOURCE` is restricted to the four plan-approved values and defaults to `zm`.
- SafeScale minimum window must not exceed the maximum window.


## TRS Signal Contract

The second P5 slice migrates the frozen upstream `python/tre/controller/trs.py` formulas into `tre_controller.signals.trs` without changing behavior.

Implemented pieces:

- `TRSInput`, `TRSResult`, and `TRSComputer` preserve `Y_m`, `y_m`, `Q`, `Q_ctl`, `TRS_raw`, EMA-smoothed `TRS`, `eta_m`, and `Z_m` outputs.
- `TRSInput.from_metrics()` adapts the P3 `ModelWindowMetrics` schema plus registry `TrsParams` into the migrated formula input.
- `SaturationGuard` preserves the legacy Gamma calculation and consecutive-window saturation counter.
- `compute_eta_m()` and `compute_z_m()` keep the old unavailable-value behavior for zero, NaN, infinity, and missing theta.

Golden coverage lives in `tre/controller/tests/golden/legacy_trs.py` and compares migrated output against the frozen implementation on multi-tick EMA, restore/snapshot, saturation, and helper edge cases.

### P5-CTRL-002 TRS signals

RED:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests python3 -m pytest -q tre/controller/tests/test_trs_signals.py
```

Result: failed during collection because `tre_controller.signals` did not exist.

GREEN:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests python3 -m pytest -q tre/controller/tests/test_trs_signals.py
```

Result: focused TRS signal tests passed with 10 tests on server 76.

Full slice verification:

```bash
cd tre && make check && make smoke
```

Result: passed with 67 tests and `tre smoke ok` on server 76.


## Classification Contract

The third P5 slice migrates the frozen upstream `paper_state.py` behavior into `tre_controller.planning.classify` as pure functions. This slice covers the paper-state path only; legacy raw-TRS planner classification is intentionally not migrated and will be recorded when the planner slice replaces the old branch.

Implemented pieces:

- `ModelState` and `ModelRole` preserve CRITICAL / LOW / HEALTHY / HIGH plus IDLE and UNKNOWN bypass states.
- `TauThresholds.from_control()` preserves the legacy `delta_crit` / `delta_high` conversion into `tau_crit`, `tau_low`, and `tau_high`.
- `classify_model()` preserves Z-threshold behavior and donor tiering by `eta_low`.
- `classify_all_models()` preserves per-model control overrides and the raw-observation zero-load IDLE donor bypass.
- `filter_donors_by_eta()` preserves the donor eta gate while allowing IDLE donors through.
- `split_receivers_donors()` preserves receiver priority sorting and donor mock-cost ordering.
- `build_comparison_log()` preserves the old transition log shape for migration diagnostics.

Golden coverage lives in `tre/controller/tests/golden/legacy_classify.py` and compares migrated output against the frozen implementation on boundary states, per-model controls, zero-load bypass, donor filtering/sorting, and comparison logs.

### P5-CTRL-003 classification

RED:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests python3 -m pytest -q tre/controller/tests/test_classify.py
```

Result: failed during collection because `tre_controller.planning` did not exist.

GREEN:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests python3 -m pytest -q tre/controller/tests/test_classify.py
```

Result: focused classification tests passed with 11 tests on server 76.

Full slice verification:

```bash
cd tre && make check && make smoke
```

Result: passed with 78 tests and `tre smoke ok` on server 76.


## Planner Contract

The fourth P5 slice migrates the frozen upstream planner's paper-state path into `tre_controller.planning.planner` as pure functions returning typed actions.

Implemented pieces:

- `PlanConfig` carries replica bounds, scale-step ratio, and fast/slow cadence due flags.
- `build_plan()` accepts already-built classifications plus metric contexts and returns only data, with no Redis, HTTP, Kubernetes, or service-manager calls.
- `ScaleAction`, `HideAction`, `UnhideAction`, and `DefragAction` define the action vocabulary required by the target P5 architecture. This slice wires scale actions; SafeScale and TP-aware defrag will use the same action model in the next sub-slices.
- CRITICAL receivers follow the frozen paper path: idle capacity, then IDLE/HIGH immediate donors, then HEALTHY/LOW middle-zone SafeScale-gated donors.
- LOW receivers follow the frozen fairness path and require saturation before donor transfer.
- IDLE donors shrink immediately; HIGH proactive shrink is SafeScale-gated.
- The legacy raw-TRS fallback is intentionally dropped. If paper-state input is incomplete, `PlanResult.dropped_legacy_raw_trs` is set and no legacy plan is produced.

Golden coverage lives in `tre/controller/tests/golden/legacy_planner.py` and compares migrated deltas, delayed-down models, and probe-upscale plans against the frozen paper path for rescue, middle-zone SafeScale, and fairness scenarios.

### P5-CTRL-004 planner paper path

RED:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests python3 -m pytest -q tre/controller/tests/test_planner.py
```

Result: failed during collection because `tre_controller.planning.planner` did not exist.

GREEN:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests python3 -m pytest -q tre/controller/tests/test_planner.py
```

Result: focused planner tests passed with 4 tests on server 76.

Full slice verification:

```bash
cd tre && make check && make smoke
```

Result: passed with 82 tests and `tre smoke ok` on server 76.

## Verification Log

### P5-CTRL-001 centralized config

RED:

```bash
PYTHONPATH=tre/common:tre/controller python3 -m pytest -q tre/controller/tests/test_config.py
```

Result: failed during collection because `tre_controller.config` did not exist.

GREEN:

```bash
PYTHONPATH=tre/common:tre/controller python3 -m pytest -q tre/controller/tests/test_config.py
```

Result: focused config tests passed with 14 tests on server 76.

Full slice verification:

```bash
cd tre && make check && make smoke
```

Result: passed with 57 tests and `tre smoke ok` on server 76.

## TP-Aware Planner Contract

The fifth P5 controller slice adds the first TP-aware capacity layer to the pure planner. The planner still performs no Redis, HTTP, Kubernetes, or service-manager calls; callers pass a cached `ClusterView` derived from service-manager `/v2/state`.

Implemented pieces:

- `PlanConfig.model_tp_sizes` declares per-model tensor parallelism for planner decisions.
- `ClusterView` carries service-manager topology and bindings as immutable planner input.
- CRITICAL receivers with `tp_size > 1` now use the service-manager allocator semantics before generic GPU-count fallback.
- A complete empty two-GPU slot emits a `ScaleAction` with reason `critical_empty_slot`.
- Fragmented two-GPU capacity emits a `DefragAction` with allocator migrations plus the receiver `ScaleAction`, both tied to reason `critical_tp_defrag`.
- If no complete slot or defrag plan exists, no scale action is emitted and `PlanResult.events` records `capacity_blocked:<model>`.

This slice covers the complete-slot, allocator-defrag, and capacity-blocked branches from the P5 TP-aware contract. The explicit "shrink HIGH same-slot halves" branch remains pending because it requires connecting SafeScale donor selection to concrete slot occupancy rather than only model-level classifications.

### P5-CTRL-005 TP-aware planner defrag

RED:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests:tre/service-manager python3 -m pytest -q tre/controller/tests/test_planner.py
```

Result: failed on the three TP-aware tests with `NameError: name '_try_plan_tp_capacity' is not defined`, proving the new tests exercised missing planner behavior.

GREEN:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests:tre/service-manager python3 -m pytest -q tre/controller/tests/test_planner.py
```

Result: focused planner tests passed with 7 tests on server 76.

Full slice verification:

```bash
cd tre && make check && make smoke
```

Result: passed with 85 tests and `tre smoke ok` on server 76.

## SafeScale State Machine

The SafeScale migration is split from loop and queue wiring. This slice defines a pure controller-side state machine that can be driven by future loops and can restore unresolved probes from the controller state store.

```text
START
  -> start_probe(model, pods, deadline)
  -> PROBING(pods, deadline, observations)

PROBING
  -> ROLLBACK when any observation violates TTFT/TPOT SLO
  -> PROBING while now < deadline and no violation
  -> COMMIT when now >= deadline and tail guard passes
  -> ROLLBACK when now >= deadline and tail guard fails

COMMIT
  -> emit scale_down donor command and pending receiver scale_up commands
  -> delete unresolved probe from state store

ROLLBACK
  -> emit unhide donor pods command
  -> delete unresolved probe from state store
```

Tail guard contract: latency must remain OK, tail `z_m` must not fall below `tau_low` when traffic is present, and normalized GPU cache must not exceed 0.8 when observed. No-traffic probes may commit if latency remains OK, matching the frozen implementation's conservative idle handling.

The state machine emits data-only commands (`hide`, `unhide`, `scale_down`, `scale_up`) and performs no Redis, HTTP, Kubernetes, or service-manager calls itself. Persistence is an injected store protocol with `save_probe`, `delete_probe`, `list_unresolved_probes`, `append_probe_journal`, and `load_probe_journal`, keeping restart recovery testable without live infrastructure.

### P5-CTRL-006 SafeScale state machine

RED:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests python3 -m pytest -q tre/controller/tests/test_safescale.py
```

Result: failed during collection because `tre_controller.planning.safescale` did not exist.

GREEN:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests python3 -m pytest -q tre/controller/tests/test_safescale.py
```

Result: focused SafeScale state-machine tests passed with 6 tests on server 76.

Full slice verification:

```bash
cd tre && make check && make smoke
```

Result: passed with 91 tests and `tre smoke ok` on server 76.

## ActionQueue Contract

The ActionQueue slice introduces the controller boundary that will own service-manager HTTP calls. It is intentionally driven by an injected client in tests, so loop code can submit typed planner/SafeScale actions without depending on live infrastructure.

Implemented pieces:

- `ActionQueue.submit()` accepts typed `ScaleAction`, `HideAction`, `UnhideAction`, and `DefragAction` values.
- The queue tracks in-flight models so later fairness actions for the same model are dropped until the current action succeeds.
- Rescue actions may replace pending fairness actions for the same model, matching the P5 rescue-priority rule.
- `drain_once()` dispatches to an injected service-manager client and releases a model from in-flight only after a successful response.
- Failed dispatches keep the model in-flight so later loop ticks do not stack conflicting actions before retry/recovery handling is added.

This slice does not yet implement the long-running `run()` coroutine, real HTTP client, loop tick wiring, or JSON logging; those remain for the next P5 slices.

### P5-CTRL-007 ActionQueue arbitration

RED:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests:tre/service-manager python3 -m pytest -q tre/controller/tests/test_action_queue.py
```

Result: failed during collection because `tre_controller.loops` did not exist.

GREEN:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests:tre/service-manager python3 -m pytest -q tre/controller/tests/test_action_queue.py
```

Result: focused ActionQueue tests passed with 4 tests on server 76.

Full slice verification:

```bash
cd tre && make check && make smoke
```

Result: passed with 95 tests and `tre smoke ok` on server 76.

## Service Manager Client Contract

The controller `sm_client` slice provides the HTTP boundary used by `ActionQueue`. Tests use an injected async transport so no live service-manager calls are required.

Implemented pieces:

- `get_state()` calls `GET /v2/state` and returns the JSON object for future cluster-view construction.
- `scale_model(model, delta)` reads current awake replicas from `/v2/state`, converts the delta into an absolute `wake_replicas` target, clamps downscales at zero, and calls `PUT /v2/models/{model}/target`.
- `set_routable(model, hidden_pods)` calls `PUT /v2/models/{model}/routable` with the hidden serve IDs.
- Transport failures are normalized into `{"ok": False, "error": ...}` for queue dispatch.
- `defrag()` currently returns an explicit unsupported result because service-manager v2 does not yet expose a defrag endpoint.

### P5-CTRL-008 service-manager client

RED:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests python3 -m pytest -q tre/controller/tests/test_sm_client.py
```

Result: failed during collection because `tre_controller.sm_client` did not exist.

GREEN:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests python3 -m pytest -q tre/controller/tests/test_sm_client.py
```

Result: focused service-manager client tests passed with 6 tests on server 76.

Full slice verification:

```bash
cd tre && make check && make smoke
```

Result: passed with 101 tests and `tre smoke ok` on server 76.

## Rescue/Fairness Tick Contract

The loop tick slice connects metrics snapshots to the migrated signal, classification, planner, and ActionQueue layers without starting infinite async loops. It keeps the P5 loop invariant testable: tick functions read only the supplied snapshot, registry, cached cluster view, active probes, and queue in-flight state; they perform no Redis or HTTP calls.

Implemented pieces:

- `run_rescue_tick()` calls the shared planner tick with `rescue_due=True` and `fairness_due=False`.
- `run_fairness_tick()` calls the shared planner tick with `rescue_due=False` and `fairness_due=True`.
- Stale snapshots are skipped and return a `snapshot_stale` event without submitting actions.
- Model contexts are derived from `MetricsSnapshot` using migrated TRS computation and paper-state classification.
- Planner in-flight filtering uses `queue.inflight_models()` so repeated loop ticks do not stack conflicting actions.
- Idle GPU capacity is derived from registry topology and assigned replicas for offline planner tests.

This slice intentionally avoids the long-running `while True` sleep loops and `SnapshotBox`; those are app-wiring work once the single-tick behavior is stable.

### P5-CTRL-009 loop tick wiring

RED:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests:tre/service-manager python3 -m pytest -q tre/controller/tests/test_loop_ticks.py
```

Result: failed during collection because `tre_controller.loops.fairness_task` did not exist.

GREEN:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests:tre/service-manager python3 -m pytest -q tre/controller/tests/test_loop_ticks.py
```

Result: focused loop tick tests passed with 3 tests on server 76.

Full slice verification:

```bash
cd tre && make check && make smoke
```

Result: passed with 104 tests and `tre smoke ok` on server 76.


## SnapshotBox / Metrics Task Contract

The metrics task slice introduces the controller-side snapshot boundary required by the P5 three-task loop design. `MetricsStore` remains the only Redis-facing component; `metrics_task` asks it for the last complete aligned metrics window and atomically replaces the in-process `SnapshotBox` value. Rescue and fairness ticks consume only this snapshot and retain their no-Redis/no-HTTP invariant.

Implemented pieces:

- `SnapshotBox` starts empty and supports atomic whole-snapshot replacement through `get()` and `set()`.
- `refresh_metrics_once()` aligns `now_ms` to the last complete `metrics_window_ms` boundary, calls `store.read_snapshot(start, end)`, and stores the result.
- Store read failures do not escape the loop boundary. If a previous snapshot exists, the same data is retained with `stale=True`; without previous data, an empty stale snapshot is published at the attempted window end.
- `metrics_task()` provides the long-running async wrapper using `ControllerConfig.metrics_window_ms` and `monitor_interval_s`, but tests target the deterministic single-refresh function.

### P5-CTRL-010 metrics snapshot task

RED:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests:tre/service-manager python3 -m pytest -q tre/controller/tests/test_metrics_task.py
```

Result: failed during collection because `tre_controller.loops.metrics_task` did not exist.

GREEN:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests:tre/service-manager python3 -m pytest -q tre/controller/tests/test_metrics_task.py
```

Result: focused metrics task tests passed with 4 tests on server 76.

Full slice verification:

```bash
cd tre && make check && make smoke
```

Result: passed with 108 tests and `tre smoke ok` on server 76.

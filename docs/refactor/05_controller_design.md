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

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

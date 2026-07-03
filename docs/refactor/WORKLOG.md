# Refactor Worklog

## 2026-07-04

### Done

- Confirmed active execution environment is server 76: `nscc-ds-4a100-node10`.
- Copied `REFACTOR_PLAN.md` into `/data/nfs_shared_data/xxy/aibrix` and read it through section 12.8 before starting remote work.
- Recorded new workspace baseline commit: `adfe6f8373afe5a90a2e93687474f07a0d4aed26`.
- Added upstream remote/tag reference and fetched official `v0.4.0` as `upstream-v0.4.0`.
- Inspected frozen old system at `/root/aibrix-main` without modifying it.
- Created P0 custom diff and interface inventory: `docs/refactor/00_custom_diff_inventory.md`.

### Current State

- P0 inventory is started and contains the major Go/config/Python v1 interface migration surface.
- No local tests/builds are counted.
- No Kubernetes write operations were performed.

### Next

- Capture read-only `kubectl get all -A -o yaml` and `nvidia-smi` snapshots if available.
- Add a simple inventory coverage checker or note exact diff coverage limits.
- Start P1 skeleton/common implementation on server 76 only, using tests as remote verification.

### P0 Snapshot Update

- Captured `nvidia-smi` to `docs/refactor/p0_snapshots/nvidia-smi.txt`.
- Captured `kubectl get pods -A -o wide` to `docs/refactor/p0_snapshots/kubectl_pods_wide.txt` with rc 0.
- Captured `kubectl get all -A -o yaml` to `docs/refactor/p0_snapshots/kubectl_all.yaml` with rc 0.
- These were read-only inspections; no cluster resources were modified.

### Next After P0

- Add/keep a lightweight coverage note for the broad new-workspace upstream drift.
- Commit P0 docs and snapshots.
- Start P1 in the remote workspace only.

### P0 Verification Update

- Added `docs/refactor/00_p0_verification.md`.
- P0 is ready to commit as documentation/snapshot work.
- No local tests or local build outputs were used.


### P1 Common Skeleton

- Added remote-only P1 `tre/common` package with registry, Redis key, metrics schema, percentile, and JSON logging helpers.
- Added `tre/deploy/registry.yaml` from frozen old manifests/profile defaults for `dsqwen-7b`, `dsllama-8b`, and `dsqwen-14b`.
- Added `tre/deploy/gen_model_manifests.py` and generated per-slot model Deployment manifests.
- Added `tre/Makefile` with `check`, `smoke`, and `manifests` targets.
- Verification recorded in `docs/refactor/01_p1_verification.md`.


### P2 Inspection Start

- Inspected new gateway queue-router, SLO router construction, request-body availability check, and old frozen queue wake-up code.
- Found target-version behavior difference: new gateway rejects zero-routable models before queue-router wake-up can run.
- Created `docs/refactor/02_upstream_patches.md` with the P2 gateway/APA patch map.


### P2 Gateway Wake-Up Dispatcher

- Added RED tests for `callWakeUpService`: missing `SERVEMENT_URL` must fail, configured URL must receive POST `/wake_up` with `model_name`, normalized `kind`, and `queue_len`.
- RED result used `/usr/local/go/bin/go` and `GOPROXY=https://goproxy.cn,direct`; the test failed because `callWakeUpService` was undefined.
- Added `pkg/plugins/gateway/algorithms/wakeup.go` for `TRE-PATCH(P2-GW-001)` with service-manager URL sourced only from `SERVEMENT_URL` and no hard-coded fallback.
- Verified `go test ./pkg/plugins/gateway/... -count=1` passed on server 76 with the regional Go proxy.
- Next P2 slice: wire zero-routable request handling in `gateway_req_body.go` to submit a wake-up request before returning 503.


### P2 Gateway Zero-Routable Hook

- Added RED test `TestValidateModelAvailabilitySubmitsWakeupWhenNoRoutablePods`; it failed because no wake-up request was observed before 503.
- Added `routingalgorithms.SubmitWakeUpIfEnabled()` and called it from `validateModelAvailability()` when the model exists but has zero routable pods.
- Verified the targeted test and `go test ./pkg/plugins/gateway/... -count=1` passed on server 76 using `/usr/local/go/bin/go` and `GOPROXY=https://goproxy.cn,direct`.
- Next P2 slice: decide whether queue-router retry needs a separate hook after the early availability hook, then migrate APA sleep-mode service-manager behavior.


### P2 TRE Redis Schema Writer

- Found the old frozen pod metric Redis writer in `/root/aibrix-main/pkg/cache/cache_trace.go`; the new target had request trace only, so this is a reintroduction into `pkg/cache`.
- Added RED tests with `miniredis` for `TRE_REDIS_SCHEMA=v2` and default dual mode. RED failed on undefined writer/schema helpers.
- Added `pkg/cache/cache_tre_redis.go`, writing `tre:v2:hist:{pod}` and `tre:v2:inst:{pod}` sorted sets, `tre:v2:pods:{model}` sets, and legacy v1 keys when mode is `v1` or `dual`.
- Wired the writer into cache initialization when Redis is configured, using millisecond timestamps aligned to `RequestTraceWriteInterval`.
- Verified `go test ./pkg/cache -count=1` and `go test ./pkg/plugins/gateway/... -count=1` passed on server 76 with `GOPROXY=https://goproxy.cn,direct`.
- Next P2 slice: APA sleep-mode service-manager adapter and podautoscaler tests.

### P2 APA Sleep-Mode Service-Manager Adapter

- Inspected frozen old APA behavior and the new target `WorkloadScale` seam.
- Added RED tests for APA sleep mode reading wake replicas from service-manager `/models_replicas`, applying replica deltas through `/scale_service`, and requiring `SERVICE_MANAGE_URL` when sleep mode is enabled.
- Added `NewWorkloadScaleFromEnv` startup validation with `APA_SCALE_SLEEP_MODE` default enabled and no hard-coded service-manager URL fallback.
- Routed APA sleep-mode `GetCurrentReplicasFromScale` and `SetDesiredReplicas` through service-manager, while leaving KPA, non-APA, and sleep-disabled APA on the existing Kubernetes scaling path.
- Verified `go test ./pkg/controller/podautoscaler -run TestAPASleepMode -count=1`, `go test ./pkg/controller/podautoscaler -count=1`, and `go test ./pkg/controller/podautoscaler/... -count=1` passed on server 76 with `GOPROXY=https://goproxy.cn,direct`.
- Final P2 verification for this slice passed: combined `go test ./pkg/plugins/gateway/... ./pkg/controller/podautoscaler/... -count=1`, `go test ./pkg/cache -count=1`, and `go build ./...` on server 76.

### P3 Metrics Store v2 Slice

- Read the P3 plan, current `tre/common` schema, new Go Redis writer, and frozen old collector formulas.
- Added `docs/refactor/03_metrics_pipeline.md` with v2 writer keys, field-level units, window semantics, and old-formula compatibility notes.
- Added RED tests for `MetricsStore` v2 sorted-set reads, histogram first/last deltas, instant expected-sample averaging, bucket-upper p95, and completed-window caching; RED failed because `tre_controller.store.metrics_store` did not exist.
- Implemented the first `tre/controller` package slice with `MetricsStore.read_model_window()` for v2 Redis keys.
- Updated `tre/Makefile` so `make check` includes controller tests.
- Verified remotely: targeted metrics-store tests passed, `cd tre && make check` passed with 12 tests, and `cd tre && make smoke` passed.
- Next P3 work: v1 compatibility mode, fixture generator edge cases, full snapshot reads, old/new collector comparison, and fixture benchmark.

### P3 Metrics Store v1 Compatibility

- Added RED test for reading legacy `aibrix:pod_histogram_metrics_*` and `aibrix:pod_instant_metrics_*` keys without a v2 pod set; RED failed on missing `schema="v1"` support.
- Implemented `MetricsStore(schema="v1")` with legacy prefix scans, timestamp suffix normalization, model-key filtering, and the same window aggregation path used by v2.
- Verified remotely: `PYTHONPATH=tre/common:tre/controller python3 -m pytest -q tre/controller/tests/test_metrics_store.py` passed with 3 tests, `cd tre && make check` passed with 13 tests, and `cd tre && make smoke` passed.
- Remaining P3 work: fixture generator edge cases, full `MetricsSnapshot` multi-model reads, old/new collector comparison, and fixture benchmark.

### P3 Fixture Edge Cases and Snapshot Reads

- Added `tre/controller/tests/make_redis_fixture.py` with a fake Redis fixture covering out-of-order v2 sorted-set writes, missing instant samples, and histogram counter resets.
- Added RED test for `MetricsStore.read_snapshot()` over the fixture; RED failed because the snapshot API did not exist.
- Implemented `read_snapshot(window_start_ms, window_end_ms)` to return a `MetricsSnapshot` for every registry model using the existing per-model window cache.
- Updated `tre/Makefile` test path so controller test helpers are importable under `make check`.
- Verified remotely: focused snapshot test passed, all metrics-store tests passed with 4 tests, `cd tre && make check` passed with 14 tests, and `cd tre && make smoke` passed.
- Remaining P3 work: multi-window fixture/benchmark, old/new collector comparison, and documented differences.

### P3 Golden Comparison and Benchmark

- Added a test-only golden collector helper that mirrors the frozen collector formulas on v2 fixture data.
- Added a comparison test proving the current `MetricsStore` matches the golden helper on the edge fixture.
- Extended `make_redis_fixture.py` with a 3 model x 8 pod x 30 minute synthetic fixture and added `benchmark_metrics_store.py`.
- Verified remotely: metrics-store tests passed with 5 tests, `cd tre && make check` passed with 15 tests, `cd tre && make smoke` passed, and the benchmark completed in 87.293 ms for 3 models / 24 pods / 30 minutes.
- Remaining P3 follow-up: optional real Redis dump if accessible and downstream P5 integration of `MetricsSnapshot`.

### P3 Closure Check

- Attempted read-only discovery for a real Redis dump. `kubectl get svc -A` found `aibrix-system/aibrix-redis-master` at `10.111.75.152:6379`, but the host-side Python Redis probe timed out, so no real dump was captured.
- P3 synthetic verification remains complete: edge fixture, golden collector comparison, and 3 model x 8 pod x 30 minute benchmark under 100 ms.
- Next phase: P4 service manager rewrite, starting with pure slot allocator tests.

### P4 Slot Allocator Slice

- Read the P4 service-manager target contract and frozen old service-manager resource code.
- Added RED tests for the required slot allocator behavior: 1-GPU allocations fill a split 2-GPU slot before opening a new slot, and the START.md fragmentation counterexample produces a minimal defrag migration.
- Implemented pure `tre_sm.allocator.slots` with `Slot`, `Binding`, `Migration`, and `SlotAllocator`.
- Updated `tre/Makefile` so `make check` includes `service-manager/tests`.
- Verified remotely: focused slot tests passed, combined Python tests passed with 17 tests, and `cd tre && make check` passed with 17 tests.
- Next P4 work: allocator property tests, then topology/state/reconcile with fake Redis and fake Kubernetes clients.

### P4 State Store Slice

- Added RED tests for service-manager state persistence: empty load defaults to version 0, saved bindings round-trip through Redis-style bytes, and stale expected versions fail without overwriting existing bindings.
- Implemented `tre_sm.state.store.StateStore` backed by `tre:v2:sm:state` and `tre:v2:sm:version` from the shared Redis key schema.
- Kept the store behind a small Redis protocol so unit tests use a fake client and never touch live Redis.
- Verified remotely: focused state-store tests passed with 2 tests, and `tre/service-manager/tests` passed with 4 tests.
- Next P4 work: allocator property tests or reconcile using fake Redis plus fake Kubernetes pod state.

### P4 Allocator Property Slice

- Added RED coverage for cross-node fragmentation: two free half-slots on different nodes now require a defrag migration instead of returning `None`.
- Added a seeded 1000-step allocation/release property test that asserts bindings never overlap and free capacity is either directly allocatable or defraggable.
- Updated `SlotAllocator.plan_defrag(2)` to allow one-GPU migrations into a free half-slot on another node.
- Verified remotely: focused slot tests passed with 4 tests, `cd tre && make check` passed with 21 tests, and `cd tre && make smoke` passed.
- Next P4 work: reconcile using fake Redis plus fake Kubernetes pod state.

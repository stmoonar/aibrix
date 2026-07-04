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

### P4 Reconcile Slice

- Added RED tests for startup reconciliation: stale Redis bindings are overwritten by existing pod `CUDA_VISIBLE_DEVICES`, while persisted bindings with no pod observation are retained with a warning.
- Added a state-store regression test for Redis clients that return strings instead of bytes, then fixed `_to_text` to support both forms.
- Implemented `tre_sm.state.reconcile` with `PodRecord`, `ReconcileResult`, and `reconcile_state()` returning an in-memory `SlotAllocator` after persisting changed merged state.
- Verified remotely: focused state/reconcile/slot tests passed with 9 tests.
- Next P4 work: topology builder/discovery adapter, then ops wrappers and API v2 idempotency.

### P4 Topology Adapter Slice

- Added RED tests for Kubernetes pod snapshot normalization: `CUDA_VISIBLE_DEVICES` beats stale GPU annotations, state annotations map through, unknown nodes fail, and invalid slot shapes are rejected.
- Implemented `tre_sm.allocator.topology` with `K8sPodSnapshot`, TRE annotation constants, and `pod_records_from_snapshots()`.
- Reused `SlotAllocator` validation instead of duplicating GPU slot rules in the discovery adapter.
- Verified remotely: focused topology tests passed with 2 tests, and `tre/service-manager/tests` passed with 11 tests.
- Next P4 work: vLLM/Kubernetes ops wrappers, then API v2 idempotency.

### P4 vLLM Ops Slice

- Added RED tests for vLLM `/sleep` and `/wake_up` operations with fake HTTP transport: retries, timeout propagation, idempotent 409 handling, and structured failure after exhausted attempts.
- Implemented `tre_sm.ops.vllm_ops.VllmOps` and `VllmOpResult` with injectable transport and lazy `requests` import for real use.
- Verified remotely: focused vLLM ops tests passed with 3 tests, and `tre/service-manager/tests` passed with 14 tests.
- Next P4 work: Kubernetes annotation/discovery ops wrapper, then API v2 idempotency.

### P4 Kubernetes Ops Slice

- Added RED tests for Kubernetes pod discovery normalization and TRE annotation patching with a fake API object.
- Implemented `tre_sm.ops.k8s_ops.K8sOps`, including model label selection, running/non-deleting pod filtering, `K8sPodSnapshot` mapping, and binding state annotation patch bodies.
- Kept Kubernetes API construction out of this slice so unit tests remain offline and future deployment wiring can choose incluster or kubeconfig loading.
- Verified remotely: focused Kubernetes ops tests passed with 2 tests, and `tre/service-manager/tests` passed with 16 tests.
- Next P4 work: API v2 idempotent target endpoint/service logic, then v1 compatibility adapters.

### P4 API v2 State/Target Slice

- Added RED tests for v2 state serialization, idempotent `PUT /v2/models/{model}/target`, bound-pool validation, and FastAPI route delegation.
- Implemented `tre_sm.api.v2.ServiceManagerV2` with deterministic state output and optimistic persistence only when target changes produce wake/sleep actions.
- Added `create_app(service)` exposing `/healthz`, `GET /v2/state`, and `PUT /v2/models/{model}/target` as thin FastAPI routes.
- Verified remotely: focused API v2 tests passed with 4 tests, and `tre/service-manager/tests` passed with 20 tests.
- Next P4 work: routable/reconcile endpoints, app wiring, and v1 compatibility adapters.

### P4 API v2 Routable Slice

- Added RED tests for idempotent `PUT /v2/models/{model}/routable` and direct `ServiceManagerV2.put_model_routable()` behavior.
- Extended `Binding` and `StateStore` with backward-compatible hidden route state persistence; existing records default to `hidden=False`.
- Updated reconcile so pod state `hidden` maps to an awake-but-hidden binding, matching the plan's route-hidden SafeScale state.
- Verified remotely: focused API v2 tests passed with 6 tests, and `tre/service-manager/tests` passed with 22 tests.
- Next P4 work: manual reconcile endpoint/app wiring, then v1 compatibility adapters.

### P4 API v2 Reconcile/App Slice

- Added RED tests for `ServiceManagerV2.reconcile()`, `POST /v2/reconcile`, and `tre_sm.app.create_service_app()`.
- Wired manual reconcile through the existing `reconcile_state()` function using an injected Kubernetes pod client, so tests remain offline and no live cluster calls are made.
- Added `tre_sm.app.create_service_app()` as the FastAPI app factory over registry, state store, and optional pod client.
- Verified remotely: focused app/API tests passed with 9 tests, and `tre/service-manager/tests` passed with 25 tests.
- Next P4 work: v1 compatibility adapters and final P4 verification/tagging.

### P4 v1 Compatibility Slice

- Added RED tests for legacy `/models_replicas`, `/scale_service`, and `/wake_up` endpoints used by the migrated APA and gateway patches.
- Implemented `tre_sm.api.v1_compat` as a route-only adapter over `ServiceManagerV2` state/target methods.
- Registered the v1 adapter in the FastAPI app factory while keeping state mutation centralized in v2 service logic.
- Verified remotely: focused v1 compatibility tests passed with 3 tests, and `tre/service-manager/tests` passed with 28 tests.
- Next P4 work: final P4 verification and tag if all P4 acceptance items are covered.

### P4 Closure Audit

- Audited P4 against REFACTOR_PLAN 5.3/6: slots, topology/state/reconcile, ops, API v2, v1 compatibility, idempotent target calls, restart consistency, and allocator property tests all have focused tests.
- Final verification passed remotely: `cd tre && make check && make smoke` completed with 43 tests and `tre smoke ok`; closure note will be committed and tagged `p4-done`.

### P5 Controller Config Slice

- Re-read `REFACTOR_PLAN.md` before starting P5 and began with the required `controller/config.py` step.
- Added RED tests for centralized controller env parsing, plan ablation switches, signal-source validation, percentile-mode validation, loop interval validation, and legacy SafeScale/state env values.
- Implemented `tre_controller.config.ControllerConfig` plus `SafeScaleConfig` as the single env parsing boundary for later P5 controller modules.
- Verified remotely: focused config tests passed with 14 tests; `cd tre && make check && make smoke` passed with 57 tests and `tre smoke ok`.
- Next P5 work: migrate `trs.py` signal formulas unchanged with golden comparisons, then pure classifier/planner paths.

### P5 TRS Signal Slice

- Re-read `REFACTOR_PLAN.md` completely before starting the signal migration slice.
- Read frozen upstream `/root/aibrix-main/python/tre/controller/trs.py` and migrated formulas into `tre_controller.signals.trs` without changing behavior.
- Added golden tests under `tre/controller/tests/golden/legacy_trs.py` covering TRS EMA sequence behavior, restore/snapshot state, saturation guard, helper edge cases, and the P3 metrics-to-TRS input adapter.
- Verified RED remotely: `tre_controller.signals` was missing. Verified GREEN remotely: focused TRS signal tests passed with 10 tests; `cd tre && make check && make smoke` passed with 67 tests and `tre smoke ok`.
- Recorded the legacy replica correction as an implementation-vs-paper note in `docs/refactor/05_paper_vs_impl.md`.
- Next P5 work: migrate classification/planner pure functions with golden comparisons, starting from the paper path and recording discarded legacy paths.

### P5 Classification Slice

- Re-read `REFACTOR_PLAN.md` completely before starting the classify/planner segment.
- Read frozen upstream `paper_state.py` and the planner's paper-state shadow path.
- Added golden tests under `tre/controller/tests/golden/legacy_classify.py` covering Z-threshold boundaries, zero-load IDLE bypass, per-model control overrides, donor eta filtering/sorting, and comparison logs.
- Implemented `tre_controller.planning.classify` as pure functions matching the frozen paper-state path.
- Verified RED remotely: `tre_controller.planning` was missing. Verified GREEN remotely: focused classification tests passed with 11 tests; `cd tre && make check && make smoke` passed with 78 tests and `tre smoke ok`.
- Next P5 work: migrate `planning/planner.py` as a pure `build_plan()` path, dropping the frozen legacy raw-TRS branch and recording that removal.

### P5 Planner Paper Path Slice

- Re-read `REFACTOR_PLAN.md` completely before starting the planner segment.
- Read the frozen upstream `planner.py` paper-state branch and `dual_cadence.py` step/cadence helpers.
- Added golden tests under `tre/controller/tests/golden/legacy_planner.py` covering CRITICAL rescue from IDLE/HIGH donors, middle-zone SafeScale probe plans, LOW fairness saturation gating, and explicit legacy raw-TRS fallback removal.
- Implemented `tre_controller.planning.planner` as a pure action-producing paper path with `ScaleAction`, `HideAction`, `UnhideAction`, and `DefragAction` types.
- Verified RED remotely: `tre_controller.planning.planner` was missing. Verified GREEN remotely: focused planner tests passed with 4 tests; `cd tre && make check && make smoke` passed with 82 tests and `tre smoke ok`.
- Next P5 work: add TP-aware cluster-view/defrag planning on top of `DefragAction`, then SafeScale state machine.

### P5 TP-Aware Planner Defrag Slice

- Re-read `REFACTOR_PLAN.md` completely on remote server 76 before continuing the planner segment.
- Added TP-aware planner tests for complete two-GPU slots, allocator defrag migrations, and explicit capacity-blocked events for 2-card CRITICAL receivers.
- Verified RED remotely: focused planner tests failed on the TP-aware cases because `_try_plan_tp_capacity` was missing.
- Implemented `ClusterView`, `PlanConfig.model_tp_sizes`, and pure planner use of `SlotAllocator.find_slot()` / `plan_defrag()` to emit `ScaleAction`, `DefragAction`, or `capacity_blocked` without direct service-manager calls.
- Verified GREEN remotely: focused planner tests passed with 7 tests; `cd tre && make check && make smoke` passed with 85 tests and `tre smoke ok`.
- Scope note: HIGH same-slot shrink remains for the next SafeScale/slot-aware donor slice rather than being approximated at model level.
- Next P5 work: proceed to SafeScale state machine.

### P5 SafeScale State Machine Slice

- Re-read `REFACTOR_PLAN.md` completely on remote server 76 before starting the SafeScale segment.
- Read frozen upstream `safescale.py` start/restore/commit/rollback/tail-guard logic and current centralized `SafeScaleConfig`.
- Added the SafeScale state diagram to `docs/refactor/05_controller_design.md` before production code, per the P5 plan.
- Added RED tests for probe start persistence, immediate SLO rollback, deadline commit with follow-up upscales, deadline rollback on failed tail health, and restoring unresolved probes from store journal.
- Implemented `tre_controller.planning.safescale` as a data-only state machine with injected persistence and no Redis/HTTP/Kubernetes calls.
- Verified RED remotely: `tre_controller.planning.safescale` was missing. Verified GREEN remotely: focused SafeScale tests passed with 6 tests, including restored-journal latency guard coverage.
- Full quality gates for this slice were completed before commit; next work moved to loops/action queue wiring.

### P5 ActionQueue Arbitration Slice

- Re-read `REFACTOR_PLAN.md` completely on remote server 76 before starting the loops/queue segment.
- Added RED tests for ActionQueue model in-flight arbitration, rescue replacement of pending fairness actions, typed action dispatch to an injected service-manager client, and failed dispatch retaining in-flight state.
- Implemented `tre_controller.loops.action_queue` with data-only queueing, rescue-priority replacement, and `drain_once()` dispatch for `ScaleAction`, `HideAction`, `UnhideAction`, and `DefragAction`.
- Verified RED remotely: `tre_controller.loops` was missing. Verified GREEN remotely: focused ActionQueue tests passed with 4 tests; `cd tre && make check && make smoke` passed with 95 tests and `tre smoke ok`.
- Scope note: long-running queue loop, real HTTP client, JSON action logs, and rescue cancellation of already-dispatched actions remain for later P5 wiring slices.
- Next P5 work: service-manager client wrapper or rescue/fairness loop wiring on top of this queue boundary.

### P5 Service Manager Client Slice

- Re-read `REFACTOR_PLAN.md` completely on remote server 76 before starting the sm_client segment.
- Read service-manager v2 implementation and tests to align with `/v2/state`, `/v2/models/{model}/target`, and `/v2/models/{model}/routable` contracts.
- Added RED tests for controller `ServiceManagerClient` delta-to-target conversion, downscale clamping, routable hidden pods, state reads, error normalization, and explicit unsupported defrag dispatch.
- Implemented `tre_controller.sm_client` with injectable async transport and a standard-library urllib transport for real use.
- Verified RED remotely: `tre_controller.sm_client` was missing. Verified GREEN remotely: focused sm_client tests passed with 6 tests; `cd tre && make check && make smoke` passed with 101 tests and `tre smoke ok`.
- Scope note: service-manager v2 has no defrag endpoint yet, so defrag dispatch is reported as unsupported until the API is added.
- Next P5 work: wire rescue/fairness loop ticks to planner, sm_client state, and ActionQueue.

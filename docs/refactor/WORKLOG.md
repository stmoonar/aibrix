# Refactor Worklog

## 2026-07-05

### N4 Real-Environment Closure

- Deployed the second live model subset for N4.3: two TP=2 `dsqwen-14b` Deployments on node10 (`gpu-0-1`, `gpu-2-3`) alongside the four bound `dsqwen-7b` node9 pods.
- Fixed TP GPU label generation for Kubernetes-safe labels while preserving comma-separated GPU IDs in annotations (`983085e3`).
- Found and fixed live zero-endpoint regressions during alternating load and fault injection:
  - `f10439e6` keeps proactive planner shrink above a serving floor for bound live models.
  - `883222d3` clamps controller-dispatched downscale targets to one awake bound replica, guarding against stale repeated downscale ticks.
- N4.3 alternating load passed: 10 minutes, 6 workers, 60s alternating phases, 7B `ok=2167`, 14B `ok=1824`, errors `0`; 7B expanded `1 -> 4`, 14B expanded `1 -> 2`.
- N4.4 live defrag/same-slot validation is recorded as a justified SKIP: current `/v2/defrag` does not recreate Kubernetes deployments, and the generated live topology cannot safely construct the required fragmentation without untracked manual placement surgery.
- N4.5 fault injection passed after additional Redis hardening:
  - `a0b2ff7f` tolerates Redis read failure during controller SafeScale restore.
  - `303047a0` keeps decision logging alive when Redis writes fail.
  - Controller restart and service-manager restart/reconcile both preserved one endpoint per live model; Redis 30s outage kept the controller pod Running with 0 restarts on final run, then service-manager reconciled state from live pods.
- N4.6 bounded soak substitute passed: 900s low-pressure gateway traffic, 790 total requests, errors `0`, controller RSS `36676 -> 36764 KB`, service-manager RSS `111824 -> 112216 KB`, Redis `DBSIZE=3`, controller/SM restarts `0`.
- N4 is ready for final full gate and `n4-done` decision; no `n4-done` tag has been created yet.

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

### P5 Rescue/Fairness Tick Slice

- Re-read `REFACTOR_PLAN.md` completely on remote server 76 before starting the loop tick segment.
- Added RED tests for stale snapshot skip, rescue tick planning from `MetricsSnapshot`, and fairness tick passing queue in-flight models into the planner.
- Implemented `tre_controller.loops.tick`, `rescue_task.run_rescue_tick()`, and `fairness_task.run_fairness_tick()` as single-tick functions with no Redis/HTTP calls.
- The tick path derives TRS/Z/context from metrics, classifies paper states, builds `PlanConfig`, calls the pure planner, and submits resulting actions to `ActionQueue`.
- Verified RED remotely: `tre_controller.loops.fairness_task` was missing. Verified GREEN remotely: focused loop tick tests passed with 3 tests; `cd tre && make check && make smoke` passed with 104 tests and `tre smoke ok`.
- Scope note: long-running async task loops, `SnapshotBox`, decision snapshot writes, and app assembly remain for later P5 wiring slices.
- Next P5 work: add SnapshotBox/metrics_task or app assembly around these single-tick functions.


### P5 Metrics Snapshot Task Slice

- Re-read `REFACTOR_PLAN.md` completely on remote server 76 before starting the metrics task segment.
- Read existing `MetricsStore`, controller config, loop tick functions, and metrics store tests to keep the new task aligned with current synchronous store APIs.
- Added RED tests for `SnapshotBox` replacement, last-complete-window store reads, stale fallback with a previous snapshot, and stale fallback without previous data.
- Implemented `tre_controller.loops.metrics_task` with a deterministic `refresh_metrics_once()` plus the long-running async `metrics_task()` wrapper for later app assembly.
- Verified RED remotely: `tre_controller.loops.metrics_task` was missing. Verified GREEN remotely: focused metrics task tests passed with 4 tests.
- Full quality gate passed remotely: `cd tre && make check && make smoke` completed with 108 tests and `tre smoke ok`.
- Next P5 work: assemble `app.py` around metrics/rescue/fairness tasks, ActionQueue draining, and ablation switch tests.


### P5 Controller App Assembly Slice

- Re-read `REFACTOR_PLAN.md` completely on remote server 76 before starting the app assembly segment.
- Inspected current controller modules and confirmed `app.py` was absent, while rescue/fairness and queue had only single-tick/drain primitives.
- Added RED tests for default task assembly, fast-loop ablation, disabled scaling startup, async rescue/fairness wrappers, and `ActionQueue.run()` drain loop behavior.
- Implemented `tre_controller.app` as a dependency-injected task assembly boundary, plus long-running rescue/fairness wrappers and queue drain loop.
- Verified RED remotely: `tre_controller.app`, `rescue_task()`, `fairness_task()`, and `ActionQueue.run()` were missing. Verified GREEN remotely: focused app/loop/queue tests passed with 13 tests.
- Full quality gate passed remotely: `cd tre && make check && make smoke` completed with 114 tests and `tre smoke ok`.
- Next P5 work: concrete controller bootstrap (`ControllerConfig.from_env()`, registry/store/sm_client construction), cluster-view state refresh, and decision snapshot writing.


### P5 Controller Bootstrap Slice

- Re-read `REFACTOR_PLAN.md` completely on remote server 76 before starting the bootstrap segment.
- Inspected current `app.py`, `ControllerConfig`, `MetricsStore`, `ServiceManagerClient`, and dependency files; confirmed there is no mandatory Redis Python dependency in the current tree.
- Added RED tests for dependency construction from `ControllerConfig` and `main()` parsing env plus delegating to an injected runner.
- Implemented `create_controller_dependencies()` and `main()` with injectable Redis client/factory and optional service-manager transport.
- Verified RED remotely: `create_controller_dependencies()` and `main()` were missing. Verified GREEN remotely: focused controller app tests passed with 6 tests.
- Full quality gate passed remotely: `cd tre && make check && make smoke` completed with 117 tests and `tre smoke ok`.
- Next P5 work: service-manager state polling into planner `ClusterView`, decision snapshot writing to `tre:v2:decision:latest`, and then fixture-driven end-to-end tick replay.


### P5 ClusterView Cache Slice

- Re-read `REFACTOR_PLAN.md` completely on remote server 76 before starting the cluster-view segment.
- Inspected planner `ClusterView`, service-manager v2 state output, current app assembly, and rescue/fairness wrapper behavior.
- Added RED tests for v2 state to `ClusterView` conversion, successful/failed cache refresh, app task assembly including a cluster-view task, and rescue tick use of cached TP-aware cluster view.
- Implemented `tre_controller.loops.cluster_view_task` plus `ClusterViewBox`, wired app dependencies/task specs, and made rescue/fairness wrappers read the latest cached view per tick.
- Verified RED remotely: `tre_controller.loops.cluster_view_task` was missing. Verified GREEN remotely: focused cluster-view/app/loop tests passed with 15 tests.
- Full quality gate passed remotely: `cd tre && make check && make smoke` completed with 121 tests and `tre smoke ok`.
- Next P5 work: decision snapshot writing to `tre:v2:decision:latest`, then fixture-driven end-to-end tick replay.

### P5 Decision Snapshot Slice

- Re-read `REFACTOR_PLAN.md` completely on remote server 76 before starting the decision snapshot segment.
- Inspected `tre_common.rediskeys`, planner action/result types, rescue/fairness wrappers, and controller app dependency assembly.
- Added RED tests for decision snapshot serialization, Redis hash writes to `tre:v2:decision:latest`, rescue/fairness loop writer calls, and app dependency construction.
- Verified RED remotely: focused tests failed on `NotImplementedError`, missing `decision_writer` loop parameters, and missing `ControllerDependencies.decision_writer`.
- Implemented `tre_controller.loops.decision_snapshot`, wired `DecisionSnapshotWriter` into controller dependencies, and injected it into rescue/fairness task specs.
- Verified GREEN remotely: focused decision snapshot/app/loop tests passed with 16 tests.
- Full quality gate passed remotely: `git diff --check` was clean, and `cd tre && make check && make smoke` completed with 125 tests and `tre smoke ok`.
- Next P5 work: fixture-driven end-to-end tick replay and remaining SafeScale/slot-aware donor integration.

### P5 Signal Source Slice

- Re-read `REFACTOR_PLAN.md` completely on remote server 76 before starting the signal-source segment.
- Inspected current config, TRS tick context construction, classification inputs, registry schema, and loop tests; confirmed `TRE_SIGNAL_SOURCE` was parsed but not wired into decisions.
- Added RED tests for `zm`, `latency_p95`, and `queue_len` signal normalization plus a rescue tick that changes classification under `latency_p95`.
- Verified RED remotely: focused tests first failed on the missing `tre_controller.signals.sources` module, then on `NotImplementedError` and missing `signal_source` tick parameter.
- Implemented `tre_controller.signals.sources`, preserved default `zm` behavior, and threaded `cfg.signal_source` through rescue/fairness wrappers into the pure tick path.
- Verified GREEN remotely: focused signal source and loop tests passed with 13 tests.
- Full quality gate passed remotely: `git diff --check` was clean, and `cd tre && make check && make smoke` completed with 130 tests and `tre smoke ok`.
- Next P5 work: fixture-driven end-to-end tick replay, SafeScale loop integration, and remaining slot-aware donor behavior.

### P5 SafeScale Tick Arbitration Slice

- Re-read `REFACTOR_PLAN.md` completely on remote server 76 before starting the SafeScale tick arbitration segment.
- Inspected current planner `requires_safescale` actions, SafeScale state-machine commands, action queue dispatch behavior, and controller app assembly.
- Added RED tests proving a SafeScale-required downscale starts a probe and submits a `HideAction` instead of an immediate scale-down, plus dependency construction for `SafeScaleStateMachine`.
- Verified RED remotely: focused tests failed on missing `safescale` tick parameter and missing `ControllerDependencies.safescale`.
- Implemented SafeScale arbitration in `run_planner_tick()`, threaded the state machine through rescue/fairness wrappers, and constructed it in controller dependencies.
- Verified GREEN remotely: focused loop/app tests passed with 16 tests.
- Full quality gate passed remotely: `git diff --check` was clean, and `cd tre && make check && make smoke` completed with 131 tests and `tre smoke ok`.
- Next P5 work: SafeScale observation/commit/rollback loop integration and fixture-driven end-to-end replay.

### P5 SafeScale Observation Task Slice

- Re-read `REFACTOR_PLAN.md` completely on remote server 76 before starting the SafeScale observation segment.
- Inspected controller app assembly, the SafeScale state machine, and the prior SafeScale arbitration boundary.
- Added RED tests for observation pending/commit behavior, rollback unhide behavior, and runtime task assembly including a SafeScale observer.
- Verified RED remotely: focused tests first failed on missing `tre_controller.loops.safescale_task`, then on `NotImplementedError` and missing `safescale` task assembly.
- Implemented `tre_controller.loops.safescale_task`, added `SafeScaleStateMachine.active_probes()`, and wired the observer into controller task specs unless SafeScale is ablated.
- Verified GREEN remotely: focused SafeScale task/app tests passed with 8 tests.
- Full quality gate passed remotely: `git diff --check` was clean, and `cd tre && make check && make smoke` completed with 133 tests and `tre smoke ok`.
- Next P5 work: persistent controller state-store backing for SafeScale recovery, fixture-driven end-to-end tick replay, and fast-loop jitter verification.


### P5 Controller SafeScale State Store Slice

- Re-read `REFACTOR_PLAN.md` completely on remote server 76 before starting the SafeScale state-store segment.
- Inspected the current SafeScale state machine, app dependency construction, and store patterns; confirmed the state machine had a `ProbeStore` protocol but no concrete controller `state_store.py`, and app startup constructed SafeScale without persistence.
- Added RED tests for Redis-backed unresolved probe/journal round trips, terminal probe removal, malformed record filtering, and app dependency startup restore of an unresolved probe.
- Verified RED remotely: focused tests failed on missing `tre_controller.store.state_store`.
- Implemented `ControllerStateStore` backed by Redis hash/list keys, added controller SafeScale Redis key helpers, and wired `create_controller_dependencies()` to restore SafeScale probes at startup.
- Verified GREEN remotely: focused controller state-store/app/SafeScale tests passed with 18 tests.
- Full quality gate passed remotely: `git diff --check` was clean, and `cd tre && make check && make smoke` completed with 137 tests and `tre smoke ok`.
- Next P5 work: fixture-driven end-to-end tick replay and fast-loop jitter verification.


### P5 Fixture Tick Replay Slice

- Re-read `REFACTOR_PLAN.md` completely on remote server 76 before starting the fixture replay segment.
- Inspected the planner tick boundary, rescue/fairness wrappers, existing loop tests, and P3 fixture helpers; selected the P5 verification gap for 60-tick offline replay.
- Added RED tests for a 60-tick CRITICAL scale-up sequence, a HIGH donor SafeScale probe sequence, and a TP=2 defrag sequence.
- Verified RED remotely: focused replay tests failed on missing `tre_controller.loops.replay`.
- Implemented `TickReplayStep`, `TickReplayResult`, `ReplayQueue`, and `run_tick_replay()` as an offline deterministic harness around `run_planner_tick()` with mock service-manager completion semantics.
- Verified GREEN remotely: focused replay tests passed with 3 tests; adjacent replay/loop/planner tests passed with 20 tests.
- Full quality gate passed remotely: `git diff --check` was clean, and `cd tre && make check && make smoke` completed with 140 tests and `tre smoke ok`.
- Next P5 work: fast-loop jitter verification under delayed slow-loop scheduling, then reassess remaining P5 closure requirements.


### P5 Fast Loop Jitter Verification Slice

- Re-read `REFACTOR_PLAN.md` completely on remote server 76 before starting the jitter verification segment.
- Inspected controller task assembly, rescue/fairness loop wrappers, and existing loop/app tests; selected the P5 verification gap for fast-loop jitter under slow-loop delay.
- Added a RED test for an asyncio fast-loop jitter probe with logical 5s rescue interval, 10s fairness interval, and 2s slow-loop delay, requiring rescue intervals to stay within 5±0.5s.
- Verified RED remotely: focused jitter test failed on missing `tre_controller.loops.jitter`.
- Implemented `run_fast_loop_jitter_probe()` and `FastLoopJitterProbeResult` using real asyncio tasks with scaled sleeps and logical time conversion.
- Verified GREEN remotely: focused jitter test passed with 1 test; adjacent jitter/loop/app tests passed with 18 tests.
- Full quality gate passed remotely: `git diff --check` was clean, and `cd tre && make check && make smoke` completed with 141 tests and `tre smoke ok`.
- Next P5 work: reassess remaining P5 closure requirements against `REFACTOR_PLAN.md`, especially any missing stateful TRS/SaturationGuard continuity or controller integration gaps before moving toward P6/P7.

### P6 Calibration Synthetic Fit Slice

- Read the archived 1939-line `fit_tre_parameters_from_runs.py` and smaller `fit_theta.py` flow enough to capture the first split boundary: higher TRS is healthier, health direction is scored with Spearman/AUROC, theta separates violating and healthy windows, and scenario splits must keep whole scenarios together.
- Added `docs/refactor/06_calibration_design.md` documenting the old monolith, the initial `dataset.py` / `fit.py` / `evaluate.py` split, and the deterministic synthetic verification fixture.
- Added RED tests for synthetic theta recovery, scenario-level train/test splitting, and threshold direction metrics; RED failed with `ModuleNotFoundError: No module named 'tre_calibration'`.
- Implemented `tre_calibration` with `CalibrationWindow`, `split_by_scenario()`, `fit_theta_from_health()`, and no-dependency AUROC/Spearman/balanced-accuracy evaluation.
- Updated `tre/Makefile` so `make check` includes `calibration/tests` and adds `tre/calibration` to `PYTHONPATH`.
- Verified remotely: focused calibration tests passed with 3 tests; `git diff --check && cd tre && make check && make smoke` passed with 144 tests and `tre smoke ok`. `make` reported clock skew warnings from file mtimes but completed successfully.
- Next P6 work: add CSV/window loading and healthy-quantile or reliability theta selection with scenario-family coverage checks, still using synthetic fixtures before touching real run data.

### P6 Calibration CSV and Reliability Fit Slice

- Added RED tests for filtered CSV window loading, SLO-derived `slo_met` and continuous health score labels, reliability theta selection, and scenario-family coverage rejection; RED failed on missing `load_windows_from_csv` and `fit_theta_by_reliability`.
- Implemented CSV loading in `tre_calibration.dataset`, including old-flow filters for warmup, contaminated/filter-reason rows, missing finite values, and zero-token windows.
- Implemented `fit_theta_by_reliability()` with the archived higher-is-healthier threshold scan, support/attainment/confidence checks, scenario-family coverage, and structured reject reasons.
- Extended `docs/refactor/06_calibration_design.md` with the CSV-loading and publish-gate contract.
- Verified remotely: new second-slice tests passed with 3 tests and all calibration tests passed with 6 tests.
- Next P6 work: add signal recomputation from token/queue columns and parameter search metrics, then wire a profile-patch emission artifact.

### P6 Calibration Signal Recompute Slice

- Added RED tests for the archived TRS formula and candidate-parameter scoring direction metrics; RED failed with `ModuleNotFoundError: No module named 'tre_calibration.signals'`.
- Added `tre_calibration.signals` with `SignalInputs`, `compute_trs()`, and `score_parameter_candidate()` using the old token, queue, cache-hit, and replica-factor formula.
- Added `evaluate_signal_direction()` so parameter scoring can report AUROC and Spearman health correlation before threshold selection.
- Extended `docs/refactor/06_calibration_design.md` with the signal recompute contract.
- Verified remotely: focused signal tests passed with 2 tests and all calibration tests passed with 8 tests.
- Next P6 work: add grid search over candidate parameters and emit a deterministic profile patch artifact containing theta and selected parameter metadata.

### P6 Calibration Parameter Search and Profile Patch Slice

- Added RED tests for grid-search selection over candidate TRS parameters and deterministic profile-patch payload construction; RED failed on missing `grid_search_parameters` and `tre_calibration.profile`.
- Implemented `grid_search_parameters()` and `ParameterSearchResult` using candidate objective/AUROC/Spearman ordering with deterministic tie-breaks.
- Added `build_profile_patch()` to emit a stable calibration artifact with publish status, theta fit gates, selected TRS parameters, and metrics without mutating the registry.
- Extended `docs/refactor/06_calibration_design.md` with the search and profile-patch contract.
- Verified remotely: focused grid/profile tests passed with 2 tests and all calibration tests passed with 10 tests.
- Next P6 work: wire a small calibration CLI over CSV input and synthetic fixture output, then run final P6 verification/tagging.

### P6 Calibration CLI Slice

- Added a RED test for a synthetic CSV-to-profile-patch CLI flow; RED failed with `ModuleNotFoundError: No module named 'tre_calibration.cli'`.
- Implemented `tre_calibration.cli.main()` with argparse input, filtered CSV loading, reliability theta fitting, direction metric scoring, and sorted JSON profile-patch output.
- Kept the CLI artifact-only: it writes a patch and does not mutate `tre/deploy/registry.yaml`.
- Extended `docs/refactor/06_calibration_design.md` with the CLI artifact contract.
- Verified remotely: focused CLI test passed and all calibration tests passed with 11 tests.
- Next P6 work: final phase audit, full verification, and `p6-done` tag if the audit stays clean.

### P7 Replayer Schedule and Dispatcher Slice

- Re-read `REFACTOR_PLAN.md` and confirmed P7 is the next unfinished phase after `p6-done`.
- Audited the frozen dispatcher in `/root/aibrix-main/CustomTraceGenerator/src/client_dispatcher.py`; scheduling uses absolute `base_time + request.timestamp` but is coupled to worker processes, OpenAI calls, plotting, and persistence.
- Added RED tests for deterministic half-open RPS schedule generation and open-loop dispatch timing; RED failed with `ModuleNotFoundError: No module named 'tre_replayer'`.
- Implemented `tre_replayer.engine.schedule` and `tre_replayer.engine.dispatcher` with injectable clock/sleep hooks and timing reports.
- Updated `tre/Makefile` so `make check` includes `replayer/tests` and adds `tre/replayer` to `PYTHONPATH`.
- Verified remotely: focused P7 tests passed with 3 tests.
- Next P7 work: add trace config loading and Poisson schedule generation before implementing lint/oracle tooling.

### P7 Replayer Trace Loader and Poisson Schedule Slice

- Inspected frozen `config/traces_v14` trace folders and confirmed `trace.json` uses a model-keyed segment format with `start_time`, `end_time`, `rps`, `input_tokens`, and `max_tokens`.
- Added RED tests for loading that segment format and seed-stable Poisson pre-generation; RED failed on missing `tre_replayer.traces` and `build_poisson_schedule`.
- Extended `RpsSegment` and `ScheduledRequest` with token controls, implemented `build_poisson_schedule()`, and added `tre_replayer.traces.loader.load_trace_segments()`.
- Verified remotely: focused loader/Poisson tests passed with 2 tests and all replayer tests passed with 5 tests.
- Next P7 work: add trace-set discovery/loading tests for existing trace folders, then implement lint/oracle foundations.

### P7 Replayer Trace Set Discovery Slice

- Added a RED test for `discover_trace_set()` reading `INDEX.json` while retaining unindexed child trace folders; RED failed on missing `discover_trace_set`.
- Implemented `TraceSet` and `TraceCase` discovery with indexed workloads first and unindexed trace folders appended by name.
- Verified remotely: focused discovery test passed and all replayer tests passed with 6 tests.
- Ran a read-only parse of frozen `config/traces_v14`: parsed 5 trace cases, 1 indexed and 4 unindexed, with segment counts recorded in `docs/refactor/07_replayer_audit.md`.
- Next P7 work: implement lint foundations (capacity model plus C1/C2/C3 reports) and oracle lower-bound checks.

### P7 Capacity Surface Foundation Slice

- Added RED tests for fitting single-pod capacity as max SLO-safe RPS per `(model, input_tokens, output_tokens)` grid point and marking out-of-grid lookups low-confidence; RED failed on missing `tre_calibration.capacity`.
- Implemented `CapacitySample`, `CapacityPoint`, `CapacitySurface`, and `fit_capacity_surface()` as pure calibration helpers for future trace lint/oracle code.
- Exported capacity helpers from `tre_calibration.__init__`.
- Verified remotely: focused capacity tests passed with 2 tests and all calibration tests passed with 13 tests.
- Next P7 work: consume the capacity surface in `tre_replayer.lint` for C1/C2/C3 trace checks.

### P7 Replayer Lint Foundation Slice

- Added RED tests for lint rejecting overcapacity traces with C1 and traces that never trigger scaling with C2; RED failed on missing `tre_replayer.lint`.
- Implemented `TraceLintReport` and `lint_trace()` using capacity-surface lookups, model slot widths, segment-boundary intervals, C1 instantaneous headroom, C2 static violation duration, and C3 headroom tier checks.
- Verified remotely: focused lint tests passed with 2 tests and all replayer tests passed with 8 tests.
- Next P7 work: implement `oracle.py` lower-bound checks and feed oracle violation results into C1 reports.

### P7 Replayer Oracle Foundation Slice

- Added a RED oracle unit test with a hand-checkable two-interval trace where only the first interval is over capacity; RED failed on missing `tre_replayer.oracle`.
- Implemented `compute_oracle_lower_bound()` with segment-boundary intervals, normalized demand, model slot widths, violation duration, violation fraction, and max required slots.
- Verified remotely: focused oracle test passed and all replayer tests passed with 9 tests.
- Next P7 work: feed oracle lower-bound output into lint reports, then add design/orchestrate skeletons and the offline dispatch precision test.

### P7 Replayer Oracle-Backed Lint Slice

- Added RED lint assertions for `oracle_violation_fraction` in `TraceLintReport`; RED failed because the field was missing.
- Corrected the short-spike test to preserve section 12.3's instantaneous C1 headroom bound while still checking oracle fraction output.
- Wired `compute_oracle_lower_bound()` into `lint_trace()` and made C1 consider both max headroom and oracle violation fraction.
- Verified remotely: focused lint tests passed with 3 tests and all replayer tests passed with 10 tests.
- Next P7 work: add the offline dispatch precision test with a local stub sender and implement design/orchestrate skeletons.

### P7 Replayer Offline Precision Helper Slice

- Added a RED test for `run_offline_precision_check()` returning pass/fail status, request count, P99 delay, RPS error, and configured limits; RED failed on missing `tre_replayer.precision`.
- Implemented `tre_replayer.precision` using deterministic schedules and the existing open-loop dispatcher with an immediate async stub sender.
- Verified remotely: focused precision test passed, all replayer tests passed with 11 tests, and a short real-clock smoke (`duration_s=1.0`, `target_rps=20.0`) passed with 20 requests, P99 delay ~1.25 ms, and RPS error ~0.12%.
- Documented the full 60 second audit command in `docs/refactor/07_replayer_audit.md`; it remains to run before `p7-done`.
- Next P7 work: add `design.py` validation/generation skeleton and `orchestrate.py` shell-flow comparison table.

### P7 Replayer Design Skeleton Slice

- Added RED tests for `design.py` phase validation and rho-to-RPS segment generation; RED failed on missing `tre_replayer.design`.
- Implemented `DemandPhase`, `validate_phase_plan()`, and `design_trace_segments()` using section 12.5 phase duration/resonance rules and the capacity surface.
- Verified remotely: focused design tests passed with 2 tests and all replayer tests passed with 13 tests.
- Next P7 work: add `orchestrate.py` skeleton and write the old-shell behavior comparison table into `07_replayer_audit.md`.

### P7 Replayer Orchestrate Skeleton Slice

- Audited old shell orchestration at `/root/aibrix-main/CustomTraceGenerator/run_experiment.sh`, `run_experiment_v2.sh`, and `run_6traces_v6_trace_stage.sh`.
- Added RED tests for Python trace config discovery and an explicit old-shell behavior comparison table; RED failed on missing `tre_replayer.orchestrate`.
- Implemented `discover_config_traces()`, `BehaviorTableRow`, `build_behavior_table()`, and Markdown rendering. Live cluster steps are marked `not_executed_offline`; dispatch/fetch/compare are documented as planned skeleton steps.
- Wrote the generated behavior table into `docs/refactor/07_replayer_audit.md`.
- Verified remotely: focused orchestrate tests passed with 2 tests and all replayer tests passed with 15 tests.
- Next P7 work: generate lint/oracle reports for existing trace sets using a placeholder or discovered capacity surface, then run the full 60 second precision audit before phase close.

### P7 Replayer Trace Report Helper Slice

- Added RED tests for `tre_replayer.report` building placeholder capacity from max trace RPS and returning JSON-ready per-trace lint summaries; RED failed on missing `tre_replayer.report`.
- Implemented `build_placeholder_capacity_surface()`, `lint_trace_case()`, and `write_trace_report()`.
- Verified remotely: focused report tests passed with 2 tests and all replayer tests passed with 17 tests.
- Ran the report helper against frozen `config/traces_v14` with placeholder capacity and wrote `docs/refactor/p7_trace_reports/traces_v14_placeholder_lint.json`. It found 5 traces, not 7; all five fail C2/C3 under the low-confidence placeholder capacity.
- Next P7 work: either derive a real capacity surface from old training-grid output or explicitly carry the placeholder limitation into final P7 closure, then run the full 60 second precision audit.

### P7 Full Offline Precision Audit

- Ran the required 60 second offline precision command: `PYTHONPATH=tre/replayer python3 -m tre_replayer.precision`.
- Result on server 76: passed with 600 requests, P99 scheduled-vs-actual delay 1.533 ms, and actual RPS error 0.000019.
- Next P7 work: final phase audit against each P7 requirement, then tag `p7-done` if the remaining evidence is sufficient or record specific gaps before moving to P8.

### P7 Replayer Metric Semantics Slice

- Added a RED test for code-level TTFT and token-control semantics; RED failed on missing `tre_replayer.metrics`.
- Implemented `TTFT_DEFINITION`, `TOKEN_CONTROL_FIELDS`, and `metric_semantics()` to preserve P7's TTFT alignment and token-control contract for future live dispatch code.
- Verified remotely: focused metric semantics test passed and all replayer tests passed with 18 tests.
- Next P7 work: final closure audit against P7 requirements and tag `p7-done` if evidence is sufficient.

### P7 Additional Trace Report

- Ran the placeholder lint/oracle report helper against frozen `config/6traces_v6` to cover more than the 5 cases available in `traces_v14`.
- Wrote `docs/refactor/p7_trace_reports/6traces_v6_placeholder_lint.json`: parsed 20 trace cases, 0 passed under the low-confidence placeholder capacity, with failures dominated by C2/C3.
- This report is audit evidence for the pipeline, not final trace qualification; real capacity surfaces remain a prerequisite for R7/final trace-set freezing.

### P7 Closure Audit

- Added a requirement-by-requirement P7 closure audit to `docs/refactor/07_replayer_audit.md`.
- P7 evidence covers open-loop dispatch, pre-generated deterministic/Poisson schedules, token controls, TTFT semantics, full 60 second precision audit, capacity/design/lint/oracle modules, trace reports for frozen trace sets, and orchestrate shell-flow comparison.
- Carried forward the explicit limitation that reports use `placeholder_from_trace_max_rps`; they validate tooling but do not qualify a final trace set. Real capacity surfaces and trace repair remain R7/final-report work.
- Next step after commit and final gate: tag `p7-done`, then start P8 UI.

### P8 UI Backend API Slice

- Started P8 after tagging `p7-done`; read the P8 plan section and service-manager FastAPI patterns.
- Added RED tests for a mock-backed UI backend aggregating registry topology/model parameters, service-manager state, Redis `tre:v2:decision:latest`, and experiment-panel stub data; RED failed on missing `tre_ui`.
- Implemented `tre_ui.app.create_ui_app()` with `/healthz`, `/api/cluster`, `/api/models`, `/api/decision/latest`, and `/api/experiments`.
- Updated `tre/Makefile` so `make check` includes `ui/tests` and adds `tre/ui` to `PYTHONPATH`.
- Added `docs/refactor/08_ui.md` documenting the backend API slice.
- Verified remotely: focused UI backend tests passed with 2 tests.
- Next P8 work: add the static single-page frontend and serve it from the FastAPI app without external CDN use.

### P8 UI Static Frontend Slice

- Added a RED test that `GET /` serves a local single-page app containing `TRE Console` and `Cluster Grid` with no runtime CDN references; RED failed with 404.
- Added `tre_ui/static/index.html` and wired `create_ui_app()` to serve it at `/`.
- The page fetches `/api/cluster`, `/api/models`, `/api/decision/latest`, and `/api/experiments`; it renders a GPU grid, decision payload, model table, and experiment stub using local HTML/CSS/JS only.
- Verified remotely: focused UI tests passed with 3 tests.
- Next P8 work: run a local FastAPI instance with mock sources and capture a screenshot if browser tooling is available, then close/tag P8 if verification remains green.

### P8 UI Screenshot Attempt

- Started a mock UI server on `127.0.0.1:18080`; `/healthz` returned `{"ok": true}`.
- Attempted `npx playwright screenshot --full-page --wait-for-selector '#cluster .node' http://127.0.0.1:18080/ docs/refactor/p8_screenshots/ui_mock.png`.
- Screenshot skipped because Playwright browser binaries are not installed on server 76 (`chrome-headless-shell` missing; CLI recommends `npx playwright install`).
- Stopped the mock server and recorded the skip reason in `docs/refactor/08_ui.md`.
- Next P8 work: final phase audit and `p8-done` tag if the backend/frontend verification remains green.

### P8 Closure Audit

- Added a P8 closure audit to `docs/refactor/08_ui.md`.
- Evidence covers mock-backed backend APIs, local static frontend serving, no-CDN test coverage, explicit experiment stub, and screenshot skip reason due missing Playwright browser binaries.
- Next step after final gate: tag `p8-done` and start P9 integration/final report.

### P9 Integration Closeout

- Added `tre_controller.offline_integration.run_offline_integration_step()` and `controller/tests/test_p9_offline_integration.py` to cover the metrics -> rescue decision -> Redis decision snapshot -> ActionQueue -> service-manager v2 path.
- RED result: focused test initially failed on missing `tre_controller.offline_integration`; after implementation the focused test passed.
- Verified `cd tre && make check` passed with 176 tests after the P9 integration slice.
- Ran `cd tre && make manifests`; it wrote 12 deployment manifests.
- Ran a temporary 5-minute offline L2 integration on localhost: real FastAPI service-manager process plus separate controller-driver process, fake Redis/state, and fixture data pump. Result: 60 ticks, 296.309 seconds, final awake count 2.
- Copied the 60-line e2e log to `docs/refactor/p9_evidence/offline_e2e_5min.jsonl`.
- L3 deploy smoke was skipped with reason: active shared cluster services already running, no verified new image deployment artifact, and host `redis-cli` missing for memory preflight. No Kubernetes write operations were performed.
- Added `docs/refactor/09_final_report.md` with residual run list R1-R7.

### Next After P9 Docs

- Run final `git diff --check`, `cd tre && make manifests`, `cd tre && make check`, and `cd tre && make smoke`.
- Commit P9 report/evidence, tag `p9-done` if the final gate passes, and leave the worktree clean.

### N1.1 Defrag Service-Manager Slice

- Started post-P9 work from `docs/refactor/10_next_steps.md`; `10_next_steps.md` is now treated as the tracked N-stage plan.
- Added RED service-manager tests in `tre/service-manager/tests/test_v2_defrag.py`; RED failed with `/v2/defrag` returning 404.
- Implemented `ServiceManagerV2.defrag()` and `POST /v2/defrag` using `SlotAllocator.plan_defrag()`, atomic state-store persistence, and an explicit hide/sleep/recreate/wake/unhide action sequence.
- Added RED controller client coverage for posting `/v2/defrag`; RED failed because `ServiceManagerClient.defrag()` still returned unsupported.
- Updated `ServiceManagerClient.defrag()` to post `{"tp_size": 2}` and keep the previous unsupported fallback for old service-manager 404 responses.
- Focused verification passed: `service-manager/tests/test_v2_defrag.py service-manager/tests/test_api_v2.py` passed with 10 tests; `controller/tests/test_sm_client.py controller/tests/test_action_queue.py` passed with 12 tests, with an existing asyncio teardown warning from the controller test process.
- Remaining N1.1 work: add a broader offline integration case for planner-produced defrag followed by scale behavior, then run the full N1 gate after N1.2/N1.3 are complete.

### N1.2 Same-Slot HIGH Shrink Slice

- Added RED planner coverage for the N1.2 priority rule: when a HIGH 1-GPU donor occupies one half of a slot and the mate GPU is free, TP=2 CRITICAL capacity planning emits `ShrinkForSlotAction` instead of `DefragAction`.
- Implemented `ShrinkForSlotAction` and same-slot donor selection in `tre_controller.planning.planner`; candidates require HIGH state, one-GPU binding, above min replicas, no active/inflight action, and a free slot mate. Ties sort by lowest `Z_m`, then serve id.
- Added RED loop coverage proving `run_rescue_tick()` converts `ShrinkForSlotAction` into a SafeScale hide probe and records pending upscale for the TP=2 beneficiary.
- Implemented tick-level SafeScale conversion for `ShrinkForSlotAction`, using the concrete `serve_id` as the probe pod and `{beneficiary: 1}` as pending upscale.
- Focused verification passed: `controller/tests/test_planner.py controller/tests/test_loop_ticks.py` passed with 19 tests.
- Design deviation recorded: existing architecture starts SafeScale probes inside `loops/tick.py`; `safescale_task.py` observes active probes. This preserves the current boundary instead of moving planner actions into the observation task.

### N1.3 Registry Parameter Sync Slice

- Added `tre/deploy/sync_registry_params.py` with unit tests for merging old `model_slo_profiles.json` and `seed_calibration.json` into `tre/deploy/registry.yaml`.
- Added `tre/deploy/registry_smoke.py` and updated `make smoke` so smoke still validates registry structure and now prints warnings for `theta_m == 0.0` or SLO drift against the old profile source.
- RED tests covered missing sync/smoke modules, then GREEN verified sync behavior and smoke warnings.
- During real dry-run, found a YAML-anchor alias bug: multiple registry models shared the same loaded `trs` dict, so updating the first model polluted later model old-values. Added a regression test and fixed sync by copying each model's `slo` and `trs` dict before mutation.
- Ran dry-run against `/root/aibrix-main/python/tre/configs/model_slo_profiles.json` and `seed_calibration.json`; then executed sync once. Registry now uses old fitted theta values and profile SLO/TRS controls: dsqwen-7b theta 738.67, dsllama-8b theta 738, dsqwen-14b theta 534; TTFT/TPOT SLOs are 500/75 ms with per-model E2E values.
- `make smoke` after sync printed no parameter warnings and ended with `tre smoke ok`.

### N1.1 Offline Defrag Integration Completion

- Added broader offline integration coverage for the N1.1 chain: fragmented one-GPU bindings at `(node-a,0)` and `(node-a,2)`, a CRITICAL TP=2 receiver, planner output of `DefragAction` then `ScaleAction`, real in-memory FastAPI service-manager dispatch, and final TP=2 expansion success.
- RED result: focused tests failed because `/v2/defrag` succeeded but the follow-up target call still rejected growth beyond the existing bound pool.
- Extended `ServiceManagerV2.put_model_target()` so targets above the current bound count allocate new bindings through `SlotAllocator.find_slot(spec.tp_size)` while preserving the `max_replicas` guard.
- Added service-manager unit coverage for TP=2 target allocation into a free full slot and updated the old bound-pool rejection test to assert the model max-replica guard instead.
- Focused verification passed: `PYTHONPATH=common:service-manager:controller pytest -q service-manager/tests/test_api_v2.py controller/tests/test_p9_offline_integration.py` passed with 11 tests; `PYTHONPATH=common:service-manager pytest -q service-manager/tests/test_api_v2.py service-manager/tests/test_v2_defrag.py` passed with 11 tests; `PYTHONPATH=common:controller:service-manager pytest -q controller/tests/test_p9_offline_integration.py controller/tests/test_action_queue.py controller/tests/test_sm_client.py` passed with 14 tests.

### N1.4 UI Screenshot Reattempt

- Ran the bounded browser install from `10_next_steps.md`: `timeout 1800 npx playwright install chromium`.
- Result: full Chromium and FFmpeg downloaded, but the command timed out during `chromium-headless-shell` download and exited non-zero.
- Follow-up launch probe failed with the same concrete missing binary: `/root/.cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell`.
- Per the N1.4 30-minute cap, UI screenshot evidence remains skipped; the exact commands and reason are recorded in `docs/refactor/08_ui.md`.

### N2.1 Component Dockerfiles

- Added shared runtime/test requirements plus Dockerfiles for `tre-v2-controller`, `tre-v2-service-manager`, and `tre-v2-ui`, all based on `python:3.11-slim` to match the frozen old TRE Python image family.
- Added runtime entrypoints: `python -m tre_controller` for controller, `tre_sm.server:create_app` for service-manager, and `tre_ui.server:create_app` for UI.
- Added deploy contract coverage in `tre/deploy/tests/test_dockerfiles.py`; `cd tre && make check` passed with 189 tests before image build.
- Built images from commit `b9ee9740` with immutable tags:
  - `tre-v2-controller:20260704-b9ee9740`, image id `sha256:72f9a0fd7fbe695333acf5528f0051bd1fe5f1b187daafba989315b35acd6ef4`, size 238495212 bytes.
  - `tre-v2-service-manager:20260704-b9ee9740`, image id `sha256:4200c08ad138a4d22e3d03686e75d8136d5861603cdc06844235f7258d891718`, size 238218201 bytes.
  - `tre-v2-ui:20260704-b9ee9740`, image id `sha256:36a5e16162718e3488c5b52a682e8b801d60d1b7428c1ed1a851104645c39cb3`, size 238161184 bytes.
- In-container verification passed: imports for `tre_controller.app`, `tre_sm.app`, and `tre_ui.app`; controller tests passed with 108 tests, service-manager tests with 31 tests, and UI tests with 3 tests. FastAPI/Starlette emitted a deprecation warning about `httpx`, but tests exited 0.

### N2.2/N2.3 Kustomize Overlays

- Added `tre/deploy/overlays/tre-v2` with namespace, service accounts, RBAC, independent Redis, controller, service-manager, and UI resources. Component images use the N2.1 immutable tags and all components point at `redis://tre-v2-redis:6379/0`.
- Added ablation overlays: `ablation-no-fastloop`, `ablation-no-safescale`, `ablation-bucket-upper`, and `ablation-interpolated`. Each overlay only patches controller env over the `tre-v2` base.
- Added `tre/deploy/tests/test_kustomize_overlays.py`; focused overlay tests passed with 2 tests.
- `kubectl kustomize` rendered all five overlays successfully; each rendered output is 304 lines.
- `kubectl apply --dry-run=client -k` passed for all five real overlays.
- Server-side dry-run against the real target namespace is blocked in N2 because namespace `tre-v2` does not exist yet and N2 must not mutate the cluster. Verified `namespace.yaml` with `kubectl apply --dry-run=server -f`; then rendered each overlay with `namespace: default` only for schema validation and `kubectl apply --dry-run=server -f -` passed for all five. Actual target-namespace server dry-run should be repeated in N3 immediately after creating or applying the namespace.
- Full verification after overlays passed: `cd tre && make check && make smoke` completed with 191 tests and `tre smoke ok`.

### N3.1 Old System Backup

- Created `docs/refactor/p11_evidence/old_system_backup/` before any N3 cluster write.
- Backed up all four old TRE deployments from `aibrix-system`: `tre-controller`, `service-management-xxy`, `service-management`, and `service-management-lxttest`.
- Backed up `kubectl -n aibrix-system get svc,cm,secret -o yaml` to `aibrix-system-svc-cm-secret.yaml`.
- Captured read-only context snapshots: `aibrix-system-pods-before-n3.txt` and `nodes-before-n3.txt`.
- No Kubernetes resources were deleted or applied during this backup step. Next N3 step is allowed only after this backup commit is present.


### N3 Partial Smoke Update

- Completed N3.1 backup of old TRE manifests under `docs/refactor/p11_evidence/old_system_backup/` and deleted only the four old TRE deployments from `aibrix-system`.
- Deployed `tre-v2` control-plane in namespace `tre-v2`; controller, service-manager, UI, and Redis are running on node10 with local pinned images.
- Deployed one `dsqwen-7b` model pod on node9 and added generated model Services so the existing AIBrix HTTPRoute can resolve `default/dsqwen-7b`.
- Fixed live service-manager reconcile issues discovered during rollout: Kubernetes object normalization, pod-list handling, namespace-preserving RBAC, and stale rollout replacement bindings.
- Fixed controller live metric ingestion by splitting controller state Redis (`tre-v2-redis`) from metrics Redis (`aibrix-redis-master`) and using the existing v1 legacy metrics reader.
- Verified gateway forwarding: 100/100 requests to `dsqwen-7b` through the AIBrix gateway succeeded with p95 28.33 ms.
- Verified full offline gate after N3 fixes: `cd tre && make check && make smoke` passed with 199 tests and `tre smoke ok`.
- Recorded N3 smoke evidence in `docs/refactor/11_l3_smoke.md` as PARTIAL; no `n3-done` tag was created.

### N3 Blocked

- Physical GPU placement does not match the manifest intent: logical `gpu_ids: [0]` maps to physical node9 GPU2 in `nvidia-smi`.
- Service-manager target sleep/wake is still state-only and does not call vLLM operations; direct vLLM `/sleep` measured 7.367s, above the N3 <5s threshold.
- Live gateway metrics are legacy `aibrix:pod_*` keys, not `tre:v2:hist:*` ZSETs; the v1 reader works but measured 138.169 ms for an uncached one-window read, above the <100ms target.
- Controller decision evidence exists in `tre:v2:decision:latest`, but controller logs did not emit `trs_calc_result`.

### N3 GPU Slot Semantics Fix

- Investigated the N3 GPU mismatch: the model Deployment carried logical TRE slot `0`, while host `nvidia-smi` showed the allocated physical GPU UUID at host index `2`.
- Confirmed the NVIDIA device plugin is configured with `DEVICE_ID_STRATEGY=uuid`; it injects `NVIDIA_VISIBLE_DEVICES=<allocated UUID>` and exposes the allocation inside the container as CUDA ordinal `0`.
- Added RED coverage showing generated manifests for logical slots `2` and `2,3` must use container-local `CUDA_VISIBLE_DEVICES=0` and `0,1`, while preserving `tre.aibrix.io/gpu-ids=2` and `2,3`.
- Fixed manifest generation and service-manager topology normalization so reconciliation prefers the TRE logical slot annotation and falls back to CUDA env only for unannotated legacy pods.
- Recorded ADR-0005: physical host GPU index equality is not enforceable with the current generic `nvidia.com/gpu` resource; N3 validates logical TRE slots plus plugin-injected UUID/runtime ordinals instead.

### N3 Final Smoke Closeout

- Found and fixed a live service-manager safety issue: controller target growth could create Redis-only bindings when no Kubernetes create path existed. Added RED coverage and guarded runtime-enabled target growth with `runtime create is not implemented for target growth beyond existing bindings`.
- Built and rolled service-manager image `tre-v2-service-manager:20260704-eaa117a4`; overlay and tests now pin that image.
- Removed stale phantom bindings left by the earlier target-growth behavior through `StateStore` version 23 -> 24, leaving only the observed `dsqwen-7b` pod binding.
- Verified reconcile cleanly reports one binding with no warnings.
- Verified real target sleep/wake with controller paused: sleep `1.116s`, wake `0.793s`; a later sleep-only reproduction reported `/is_sleeping: true` and pod annotation `tre.aibrix.io/state=sleeping`.
- Verified gateway burst through AIBrix gateway: `100/100` requests, p95 `31.16ms`.
- Verified gateway v2 metrics in AIBrix Redis and v2 MetricsStore read latency below 100ms.
- Verified controller logs emit `trs_calc_result` with `stale:false`; measured live tick-path p95 `1.707ms` over 30 iterations.
- Generated restore-ready sanitized rollback manifests under `docs/refactor/p11_evidence/old_system_backup/restore_ready/`; server-side dry-run passed for all restore-ready YAMLs.
- Updated `docs/refactor/11_l3_smoke.md` to PASS. Next step: commit final evidence and tag `n3-done`.

### N4 Realenv Start

- Created `docs/refactor/12_realenv_tests.md` and recorded that applying every generated model Deployment at once is not feasible: the model set requests 16 GPUs total and 12 GPUs on node9, while node9 has 4 GPUs.
- Ran N4.2 hot-switch on the live `dsqwen-7b` pod with controller paused: 20 cycles, wake P95 `0.864s`, sleep P95 `1.065s`, no binding drift.
- Deployed the `dsqwen-7b` node9 subset (`gpu-0..3`) for single-model realenv tests.
- Fixed service-manager discovery to read `tre.aibrix.io/gpu-ids` from generated pod labels when annotations are absent; rebuilt and rolled `tre-v2-service-manager:20260704-dd1c42a9`.
- Fixed gateway routing over mixed awake/sleeping pods by adding `tre.aibrix.io/routable=true` to generated Service selectors and pod labels, and by making service-manager sleep/wake plus routable hide/unhide patch that label; rolled images `20260704-264c0124` and `20260704-3190906b`.
- After labeling live pods and updating `default/dsqwen-7b` Service, endpoints contained only the awake pod and gateway validation passed: 20/20 requests, max `34.96ms`.
- Ran a 120s single-model step load at 20 RPS / 1 output token: 2401/2401 requests, p95 `28.37ms`. Controller logs remained non-stale but did not wake additional replicas; heavier load is still needed for the CRITICAL expansion requirement.

### N4 Controller Live Scaling Fixes

- Reproduced the N4.3 heavy concurrent load failure with `dsqwen-7b` four-bound/one-awake state: the controller initially drove the only awake endpoint to zero because planner replica counts and service-manager wake-target semantics were using different notions of capacity.
- Added planner regression coverage and fixes across commits `54313fdd`, `b9a92604`, `9dd7b9ab`, `b17133c9`, and `e0b4bb64`:
  - per-model min/max replica bounds instead of a cluster-wide min bound;
  - planner `ScaleAction` decisions use awake/routable replicas for deltas;
  - bound sleeping replicas are handled as `critical_sleeping_capacity`;
  - legacy v1 metrics are reconciled with service-manager `ClusterView`;
  - TRS is computed from awake replicas while planner context retains bound replicas;
  - live rescue/fairness ticks wait for cluster-view state before scaling.
- Built and rolled final controller image `tre-v2-controller:20260704-e0b4bb64` (`sha256:386ffa7e3592adb85c971ccc601013d743878d9e346cb9f7ebe4f332117acb6e`).
- Full verification after the final controller fix passed: `git diff --check && cd tre && make check && make smoke` with 217 tests and `tre smoke ok`.
- Final heavy concurrent live run passed with `/tmp/tre_concurrent_step_with_controller.py`: 8 workers, `max_tokens=96`, controller enabled after load start, `783` successful requests, `0` errors, p95 `1277.21ms`.
- Final scaling evidence: `dsqwen-7b` stayed non-empty and expanded from `awake=1,bound=4` to `awake=3,bound=4` at sample `30.4s`, then `awake=4,bound=4` at sample `45.9s`; final Service endpoints contained all four pods.
- Ran the N4.3 output-length drift sample on the four-awake `dsqwen-7b` subset: 20/20 successes for each `max_tokens` setting; p95 latency was `34.74ms` for 1 token, `429.99ms` for 32 tokens, and `1246.39ms` for 96 tokens; post-check remained `awake=4,bound=4` with four Service endpoints.

### N4b.1 D7 GPU Binding Refactor

- First N4b work step committed `docs/refactor/10_next_steps.md` as `[N4b] add next-steps execution plan` (`943ab486`).
- Baseline before N4b code changes: `cd tre && make check` passed with 220 tests.
- Added registry `gpu_uuids` support and wrote `tre/deploy/collect_gpu_uuids.py` with unit tests for parsing `nvidia-smi -L` output and updating registry nodes.
- Read-only UUID collection:
  - node9 GPU0..3: `GPU-689a3e93-68db-0dac-160b-6a791cf246e8`, `GPU-d0de9f25-c059-d2ee-e7c4-c242bbdc76c7`, `GPU-3a113474-dd92-6d52-d05b-491e7b020ded`, `GPU-3c2fb581-708a-5fef-3eaa-5c3cc21a028e`.
  - node10 GPU0..3: `GPU-71f560a8-e090-c7e6-325f-2e386c08136f`, `GPU-4bcbcb0c-7eaf-64a1-b3e5-b07cb81d3a96`, `GPU-28af749d-4081-d7b6-0c14-cf9c29aa213d`, `GPU-76a392a5-b027-42be-fb8f-1bfe9079b47c`.
- Converted generated model Deployments to ADR-0006/D7 form: no `nvidia.com/gpu` requests or limits, `nodeName` pinned, `NVIDIA_VISIBLE_DEVICES` set to UUIDs, logical ids preserved in `tre.aibrix.io/gpu-ids`, and UUIDs recorded in `tre.aibrix.io/gpu-uuids`.
- Added static manifest budget guard: a generated layout may reference each GPU from at most three model Deployments.
- Changed `SlotAllocator` semantics from "one bound per GPU" to "one awake per GPU"; sleeping bindings can share a GPU, while `bind(..., awake=True)` and `feasible_wake()` reject double-awake conflicts.
- Service-manager target wake and routable unhide now check `feasible_wake()` and FastAPI maps conflicts to HTTP 409.
- Reconcile now detects externally-introduced double-awake GPU conflicts and marks the later deterministic observation sleeping with a warning.
- RED tests first failed for missing `gpu_uuids`, old `nvidia.com/gpu` manifests, old bound-only allocator semantics, missing 409, and reconcile double-awake failure; GREEN implementation passed the focused N4b.1 subset with 36 tests.
- Verification:
  - `cd tre && make check` passed with 227 tests.
  - `cd tre && make manifests` wrote 15 resources.
  - `cd tre && make smoke` printed `tre smoke ok`.
  - Manifest audit: `grep -R 'nvidia.com/gpu' tre/deploy/models` returned no matches; generated files include `nodeName`, `NVIDIA_VISIBLE_DEVICES`, and `tre.aibrix.io/gpu-uuids`.

### N4b Blocked

- None for N4b.1.

### N4b Next

- Continue with 10.2: replace `/v2/defrag` state-only recreate with the real k8s delete/create path behind fake-client tests, reusing the manifest template as the single source of pod specs.

### N4b.2 Defrag Real Kubernetes Path

- Added fake-client RED coverage for `K8sOps.delete_model_deployment()` and `create_model_deployment()`. The create path must reuse `gen_model_manifests.py` helpers and must not hand-write a second pod spec.
- Extended `gen_model_manifests.py` with public `deployment_name()` and `build_model_deployment()` helpers for the service-manager ops layer.
- Extended `K8sOps` with Deployment delete/create and condition polling. `wait_pod_ready()` resolves the real Pod through the generated Deployment's `app` label, avoiding the Deployment-name vs Pod-name mismatch.
- Updated `tre_sm.server` wiring so live service-manager uses `CoreV1Api` for pods and `AppsV1Api` for Deployment lifecycle calls, with the loaded registry passed into `K8sOps`.
- Reworked `/v2/defrag` runtime path to execute `hide -> sleep -> delete Deployment -> wait old pod gone -> create Deployment -> wait Ready pod -> wake -> unhide`.
- Fixed a real-path state bug caught by RED: recreated Pods can have a different serve id from the deleted binding. The store now records the Ready Pod name returned by `wait_pod_ready()`.
- Connected target growth to the same fake Deployment path when runtime deployment ops are available. This lets the P9 offline integration cover defrag followed by TP=2 expansion without falling back to Redis-only create.
- Updated `controller/tests/test_p9_offline_integration.py` so the fragmented-capacity case uses fake runtime ops and verifies the real defrag delete/create path plus subsequent TP=2 create/wake.
- Verification:
  - Focused RED/GREEN subset: `pytest -q service-manager/tests/test_k8s_ops.py service-manager/tests/test_v2_defrag.py deploy/tests/test_gen_model_manifests.py` passed with 17 tests.
  - P9/runtime subset: `pytest -q controller/tests/test_p9_offline_integration.py service-manager/tests/test_api_v2.py service-manager/tests/test_v2_defrag.py service-manager/tests/test_k8s_ops.py` passed with 29 tests.
  - `cd tre && make check` passed with 231 tests.

### N4b Blocked

- None for N4b.2.

### N4b Next

- Continue with 10.3 canary first. Before full rollout, apply one no-GPU-request D7 pod and prove the container sees only the UUID named in `NVIDIA_VISIBLE_DEVICES`; record the result in WORKLOG before deploying dsqwen-7b + dsllama-8b together.

### N4b.3 D7 Runtime Canary

- Pre-canary read-only state:
  - Existing model pods were already running from prior N4: four `dsqwen-7b` pods on node9 and two `dsqwen-14b` pods on node10.
  - node9 GPU memory: GPU0 1118 MiB, GPU1 1118 MiB, GPU2 36956 MiB, GPU3 1118 MiB. Canary selected node9 GPU0, which only had sleeping-level memory.
- Applied a temporary pod manifest from `/tmp/tre-n4b-d7-canary.json` in `default`. It used the vLLM image, no `nvidia.com/gpu` requests/limits, no `runtimeClassName`, no privileged mode, and no hostPath `/dev/nvidia*`; it set `NVIDIA_VISIBLE_DEVICES=GPU-689a3e93-68db-0dac-160b-6a791cf246e8` and `CUDA_VISIBLE_DEVICES=0`.
- Canary result: PASS. `kubectl logs` and `kubectl exec ... nvidia-smi --query-gpu=index,uuid` both showed exactly one visible GPU: `0, GPU-689a3e93-68db-0dac-160b-6a791cf246e8`.
- Fallbacks were not needed: `runtimeClassName: nvidia`, privileged mode, and `/dev/nvidia*` hostPath were not used.
- Cleaned up the temporary pod with `kubectl -n default delete pod tre-n4b-d7-canary`.
- This satisfies the 10.3 canary gate; full D7 rollout may proceed.

### N4b.3 Shared-GPU Dual-Model Switch

- Paused the v2 controller during the manual 10.3 test with `kubectl -n tre-v2 scale deploy/tre-v2-controller --replicas=0`; service-manager remained live on image `tre-v2-service-manager:20260705-fa313832`.
- Applied the D7 `dsllama-8b` single-GPU Deployment on node9 GPU0 alongside D7 `dsqwen-7b` on the same GPU UUID. Both pods used `NVIDIA_VISIBLE_DEVICES=GPU-689a3e93-68db-0dac-160b-6a791cf246e8` with no `nvidia.com/gpu` requests or limits.
- Found and fixed one live reconciliation bug before accepting 10.3: `pod_records_from_snapshots()` rejected a double-awake observation before reconcile could repair it. Added regression coverage and rolled `tre-v2-service-manager:20260705-fa313832`; `cd tre && make check` passed with 232 tests.
- Initial dual-switch scripts produced gateway 502s. Root cause was the temporary validation traffic, not D7 binding: body-only requests lacked the HTTP `model` header required by the existing `HTTPRoute` match and fell through to `reserved-router` / `aibrix-gateway-plugins:50052`, producing Envoy `protocol_error`. A controlled probe showed body-only requests 502 and identical requests with `model: dsqwen-7b` header 200.
- Re-ran the official 20-round shared-GPU switch with both JSON body `model` and HTTP `model` header:
  - Script/output: `/tmp/n4b_dual_switch_header_1783229055.json` on local disk.
  - 20 rounds alternated `sleep dsqwen-7b -> wake dsllama-8b -> gateway 20 requests -> sleep dsllama-8b -> wake dsqwen-7b -> gateway 20 requests`.
  - Gateway result: `errors=0`, `readiness_errors=0`.
  - Wake P95 including SM target call: `dsllama-8b 1.3426s`, `dsqwen-7b 1.2343s`.
  - Per-round gateway request p95 stayed below `49ms`; no double-awake condition was detected.
- Final state after the script intentionally left `dsqwen-7b` awake and `dsllama-8b` sleeping on node9 GPU0. Endpoints matched routability: `dsqwen-7b -> 10.244.3.53:8000`, `dsllama-8b -> <none>`.

### N4b Blocked

- None for N4b.3.

### N4b Next

- Continue with 10.4 full topology rollout only after committing this WORKLOG update and re-running `cd tre && make check`.

### N4b.4 Full Topology D7 Rollout

- While preparing 10.4, found a D7-era target-selection bug: when the first sleeping binding for a model was on a GPU already occupied by another awake model, `PUT /v2/models/{model}/target` returned 409 instead of trying the next feasible sleeping binding.
- Added RED coverage in `service-manager/tests/test_api_v2.py` and changed service-manager target growth to skip infeasible sleeping bindings while preserving the existing 409 behavior when no feasible existing binding can satisfy the target. Verification passed: focused service-manager subset 27 tests and `cd tre && make check` with 233 tests.
- Built and rolled service-manager image `tre-v2-service-manager:20260705-7278875d`; container pytest for API/slots/reconcile passed with 27 tests. Live `tre-v2-service-manager` rolled successfully to pod `tre-v2-service-manager-dccbd689d-hdw9b`.
- Full-topology rollout had to be sequential, not a single `kubectl apply -k`, because vLLM pods start awake before they can be slept. The operational script waits for vLLM `/is_sleeping` readiness after Kubernetes rollout, then reconciles and sleeps the model before moving to the next overlapping Deployment.
- Recreated or created all non-D7/missing model Deployments so every live model pod now has `NVIDIA_VISIBLE_DEVICES=<GPU UUID>`:
  - `dsqwen-7b`: four node9 one-GPU pods.
  - `dsllama-8b`: four node9 one-GPU pods.
  - `dsqwen-14b`: two node10 TP=2 pods and two node9 TP=2 pods.
- Main rollout script completed the topology and target setup but failed only at its final node9 `nvidia-smi` collection because the remote shell on 76 cannot resolve the local SSH alias `A100_75`. This did not affect cluster state; final verification was rerun separately.
- Final verification:
  - Evidence: `/tmp/n4b_full_topology_verify_1783231975.json`.
  - State: `dsqwen-7b awake=1 bound=4`, `dsllama-8b awake=1 bound=4`, `dsqwen-14b awake=1 bound=4`.
  - `POST /v2/reconcile` returned no warnings.
  - Endpoints: `dsqwen-7b -> 10.244.3.53:8000`, `dsllama-8b -> 10.244.3.57:8000`, `dsqwen-14b -> 10.244.0.163:8000`.
  - Gateway: each model served 20/20 requests through AIBrix gateway with 0 errors; max latency `35.51ms` / `38.91ms` / `36.63ms`.
  - Node9 memory: GPU0 `39908 MiB`, GPU1 `39916 MiB`, GPU2 `4070 MiB`, GPU3 `4070 MiB`.
  - Node10 memory: GPU0 `37157 MiB`, GPU1 `37157 MiB`, GPU2 `1825 MiB`, GPU3 `1825 MiB`.
- Updated `docs/refactor/12_realenv_tests.md` N4.1 from the old GPU-request-era SKIP to N4b/D7 full-topology PASS.

### N4b Blocked

- None for N4b.4.

### N4b Next

- Continue with 10.5 live defrag and same-slot shrink validation.

### N4b.5 Live Defrag And Same-Slot Shrink

- Added and verified one more live-path hardening fix before the final 10.5 attempt: recreated vLLM pods can report Kubernetes Ready before their HTTP server is ready for `/wake_up`. `VllmOps.wait_until_ready()` now polls `/is_sleeping` after `K8sOps.wait_pod_ready()` in both target-create and defrag-create paths, so wake is only sent after the vLLM API is accepting requests.
- Verification for the readiness fix passed:
  - Focused service-manager subset: `pytest -q service-manager/tests/test_v2_defrag.py service-manager/tests/test_api_v2.py service-manager/tests/test_vllm_ops.py service-manager/tests/test_k8s_ops.py` passed with 31 tests.
  - P9/runtime subset: `pytest -q controller/tests/test_p9_offline_integration.py service-manager/tests/test_api_v2.py service-manager/tests/test_v2_defrag.py service-manager/tests/test_k8s_ops.py` passed.
  - Full gate before rollout: `cd tre && make check` passed with 233 tests.
- Built and rolled service-manager image `tre-v2-service-manager:20260705-ff9d1580`; rollout completed as `tre-v2-service-manager-6468f98ff5-4gjkl`.
- Live defrag construction then hit a model-memory budget blocker rather than another service-manager logic bug:
  - The N4b/D7 full topology leaves multiple sleeping vLLM processes co-resident on node9 GPUs.
  - Constructing the fragmented 10.5 case required recreating/warming a `dsqwen-7b` pod on a GPU that also held sleeping TP=2 `dsqwen-14b` state.
  - vLLM failed during warm-up with CUDA OOM (`Tried to allocate 150.00 MiB`; only about `74 MiB` free on the target GPU).
  - This prevents a safe live PASS for the exact "fragment -> `/v2/defrag` -> 14b wakes in complete slot" scenario under the current launch parameters.
- Restored the live service to a clean minimal three-model state after the failed construction:
  - `dsqwen-7b awake=1 bound=2`, endpoint `10.244.3.53:8000`.
  - `dsllama-8b awake=1 bound=4`, endpoint `10.244.3.57:8000`.
  - `dsqwen-14b awake=1 bound=4`, endpoint `10.244.0.163:8000`.
  - `POST /v2/reconcile` repaired stale state; `python3 /tmp/probe_sleeping.py` showed routability aligned with the three awake endpoints and no stale qwen GPU1 pod.

### N4b Blocked

- 10.5 live defrag is blocked by GPU memory headroom in the D7 full topology. Completing it safely requires reducing sleeping co-residency or changing vLLM launch parameters such as `gpu_memory_utilization`, `max_model_len`, or `max_num_seqs`; those are outside the N4b defrag-path change and would need a separate model-serving decision.

### N4b Next

- Continue with the unaffected part of 10.6: three-model alternating/gateway stability and 12-hour local-disk soak, using the restored minimal three-model state unless the controller test needs a narrower topology.

### N4b.6 Three-Model Alternating And Soak Handoff

- Re-enabled the v2 controller from the manual-test pause. Deployment image remained `tre-v2-controller:20260704-303047a0`.
- First 15-minute three-model alternating precheck failed only for `dsqwen-7b` gateway traffic:
  - Evidence: `/tmp/n4b_three_model_precheck_1783234432.json`.
  - `dsllama-8b`: `ok=920`, `errors=0`.
  - `dsqwen-14b`: `ok=1956`, `errors=0`.
  - `dsqwen-7b`: `errors=4653`, all gateway 502/503 class.
  - Direct Service and Pod probes for `dsqwen-7b` were healthy; gateway returned `httproutes.gateway.networking.k8s.io "dsqwen-7b-router" not found`.
- Restored the missing model HTTPRoute `aibrix-system/dsqwen-7b-router` using the same header-match/backend pattern already present for `dsllama-8b-router` and `dsqwen-14b-router`. This is model-route repair, not a change to the AIBrix gateway base deployment.
- Reran the 15-minute alternating precheck after route repair:
  - Evidence: `/tmp/n4b_three_model_precheck_1783235642.json`.
  - Result: `errors={}`.
  - `dsqwen-7b ok=978`, p95 `1245.60 ms`.
  - `dsllama-8b ok=906`, p95 `1347.73 ms`.
  - `dsqwen-14b ok=1947`, p95 `621.17 ms`.
  - Controller RSS `37032 -> 37144 KB`; service-manager RSS `111228 -> 111244 KB`; Redis `DBSIZE=3 -> 3`; TRE pod restart deltas all `0`.
  - However, all models stayed at `awake=1`; this does not satisfy the "三模型均正确扩缩" half of 10.6.
- Root cause for missing expansion under the default controller signal was live metrics incompleteness:
  - Controller logs repeatedly emitted `paper_state_incomplete_drop_legacy_raw_trs`.
  - A local metrics inspection showed AIBrix v1 windows can include pod/running queue data without matching token histograms in the same completed window, leaving paper `Z_m` unavailable for non-idle models.
  - With `TRE_SIGNAL_SOURCE=zm` (the default/paper path), planner drops actions when any non-idle model has incomplete paper state.
- Queue-length signal canary:
  - Temporarily rolled controller with `TRE_SIGNAL_SOURCE=queue_len` to test whether the control loop can still issue actions from available live metrics.
  - Evidence: `/tmp/n4b_queue_signal_canary_1783236733.json`.
  - Controller produced `critical_sleeping_capacity` scale actions for `dsqwen-7b`, `dsllama-8b`, and `dsqwen-14b`.
  - Serving result over 300s: `dsqwen-7b ok=601/errors=202`, `dsllama-8b ok=590/errors=0`, `dsqwen-14b ok=1297/errors=0`.
  - `dsqwen-7b` reached `awake=2`, `dsqwen-14b` reached `awake=2`; `dsllama-8b` action was emitted but the observed awake count stayed at 1 during the sample.
- The queue-signal canary exposed the same model-memory blocker as 10.5:
  - Controller target growth recreated `dsqwen-7b-nscc-ds-4a100-node9-gpu-2`.
  - The pod entered CrashLoopBackOff because vLLM startup required `35.44 GiB` at `gpu_memory_utilization=0.9`, but only `14.75/39.38 GiB` was free on the target GPU due to co-resident sleeping state.
  - Deleted the failed qwen GPU2 Deployment, reconciled service-manager state, restored controller to default signal env, and reset `dsqwen-7b`/`dsllama-8b` to one awake. The controller subsequently woke `dsqwen-14b` back to two awake replicas.

### N4b Blocked

- 10.6 cannot honestly be marked live PASS yet. Serving-only alternating traffic is healthy after restoring `dsqwen-7b-router`, but the required expansion/shrink behavior is blocked by two architectural/runtime questions:
  - The paper `zm` signal path drops actions when AIBrix v1 metric windows do not contain complete token histograms for every active model.
  - The fallback `queue_len` signal can drive expansion, but D7 co-resident sleeping pods leave insufficient vLLM startup headroom for some recreated 7B slots.

### N4b Architecture Decision Needed

- Choose one of these before rerunning 10.6 and 12h soak:
  - Keep paper `zm` as mandatory and fix/bridge the live metrics feed so completed windows always provide usable token histograms per active model.
  - Allow a documented live fallback to `queue_len` for N4b soak, while keeping paper `zm` as the intended production signal.
  - Reduce serving memory pressure/co-residency before controller-driven expansion, for example by lowering vLLM `gpu_memory_utilization`/`max_model_len`/`max_num_seqs` or reducing the number of sleeping pods per GPU.

### N4b Next

- Do not start the 12h soak until the architect chooses the signal/memory policy above; otherwise the soak would only prove steady serving at the current awake set, not the 10.6 expansion/shrink contract.

### Endgame F1.1/F1.2 Sleep Leak Evidence And First Probe

- Committed the new authority plan `docs/refactor/14_endgame_plan.md` as `[Endgame] add final execution plan` (`0e9ac5e4`); post-commit `cd tre && make check` passed with 233 tests.
- Paused `tre-v2-controller` during manual F1 operations with `kubectl -n tre-v2 scale deploy/tre-v2-controller --replicas=0`.
- Captured the pre-cleanup state under `docs/refactor/p11_evidence/f1_pre_cleanup_20260705/`:
  - `model_deployments.yaml`
  - `pods_endpoints.yaml`
  - `sm_state.json`
  - `node_gpu_memory.txt`
- Pre-cleanup node9 GPU truth confirmed the architect's leak finding:
  - GPU2 had `dsllama-8b-gpu-2` at `22856 MiB` and `dsqwen-14b-node9-gpu-2-3` shard at `16946 MiB`, while both pods reported `/is_sleeping=true`.
  - GPU3 had the paired `dsqwen-14b-node9-gpu-2-3` shard at `37838 MiB`, while `/is_sleeping=true`.
- Deleted the two leaking Deployments:
  - `dsllama-8b-nscc-ds-4a100-node9-gpu-2`
  - `dsqwen-14b-nscc-ds-4a100-node9-gpu-2-3`
- Post-delete node9 GPU truth:
  - GPU2 returned to `0 MiB`.
  - GPU3 returned to `2248 MiB`, matching only healthy sleeping residue from `dsqwen-7b-gpu-3` and `dsllama-8b-gpu-3`.
- Added `tre/deploy/scripts/n4b_e1_sleep_probe.py` and focused tests in `tre/deploy/tests/test_n4b_e1_sleep_probe.py`; RED failed on missing module, GREEN passed with 3 tests, then full `cd tre && make check` passed with 236 tests.
- Ran E1-a on clean node9 GPU2 using `dsllama-8b-gpu-2`:
  - Evidence: `docs/refactor/p11_evidence/f1_pre_cleanup_20260705/n4b_e1_dsllama_gpu2_a.json`.
  - `create -> ready` used `37414 MiB`.
  - `sleep` after zero traffic returned to `1090 MiB`.
- Ran E1-b on the same pod:
  - Evidence: `docs/refactor/p11_evidence/f1_pre_cleanup_20260705/n4b_e1_dsllama_gpu2_b.json`.
  - `wake` used `36936 MiB`.
  - `wake -> 20 short requests -> sleep` returned to `1090 MiB`.
- Extended the E1 probe script with `--concurrency` using RED/GREEN coverage. Focused tests now pass with 4 tests; full `cd tre && make check` passed with 237 tests.
- Ran E1-c on the same `dsllama-8b-gpu-2` pod:
  - Evidence: `docs/refactor/p11_evidence/f1_pre_cleanup_20260705/n4b_e1_dsllama_gpu2_c.json`.
  - `wake -> 200 requests, concurrency=8, max_tokens=96 -> sleep` returned to `1090 MiB`.
- Ran E1-d on the same pod:
  - Evidence: `docs/refactor/p11_evidence/f1_pre_cleanup_20260705/n4b_e1_dsllama_gpu2_d.json`.
  - Ten consecutive rounds of `wake -> 200 requests, concurrency=8, max_tokens=96 -> sleep` all returned to `1090 MiB`; no one-time step leak and no rising trend.
- Ran E1-e on the same pod:
  - Evidence: `docs/refactor/p11_evidence/f1_pre_cleanup_20260705/n4b_e1_dsllama_gpu2_e_level2.json`.
  - Two rounds using `sleep?level=2` returned to `1090 MiB` and the second wake succeeded. For this model/image path, level 2 did not reduce residue below level 1.
- Current E1 interpretation for single-GPU `dsllama-8b`: the previously leaked pod was cured by delete/recreate, and a clean replacement does not reproduce the leak under zero traffic, short traffic, 200-request concurrent traffic, 10 repeated heavy rounds, or level-2 sleep.
- Extended the E1 probe script with multi-GPU sampling support for TP=2 pods. Focused tests now pass with 5 tests; full `cd tre && make check` passed with 238 tests.
- Recreated `dsqwen-14b-nscc-ds-4a100-node9-gpu-2-3` on node9 GPU2/GPU3 and ran E1-a..E1-e:
  - E1-a evidence: `docs/refactor/p11_evidence/f1_pre_cleanup_20260705/n4b_e1_dsqwen14b_node9_gpu23_a.json`. Ready used about `37756 MiB` per 14B shard; zero-traffic sleep returned each 14B shard to `1766 MiB`.
  - E1-b evidence: `docs/refactor/p11_evidence/f1_pre_cleanup_20260705/n4b_e1_dsqwen14b_node9_gpu23_b.json`. `wake -> 20 short requests -> sleep` returned each 14B shard to `1766 MiB`.
  - E1-c evidence: `docs/refactor/p11_evidence/f1_pre_cleanup_20260705/n4b_e1_dsqwen14b_node9_gpu23_c.json`. `wake -> 200 requests, concurrency=8, max_tokens=96 -> sleep` returned each 14B shard to `1766 MiB`.
  - E1-d evidence: `docs/refactor/p11_evidence/f1_pre_cleanup_20260705/n4b_e1_dsqwen14b_node9_gpu23_d.json`. Ten consecutive heavy rounds stayed flat: GPU2 total `2856 MiB`, GPU3 total `3910 MiB` after every sleep; the 14B shard contribution remained `1766 MiB` on each GPU.
  - E1-e evidence: `docs/refactor/p11_evidence/f1_pre_cleanup_20260705/n4b_e1_dsqwen14b_node9_gpu23_e_level2.json`. Two level-2 sleep rounds returned to the same residue and the second wake succeeded.
- Current E1 interpretation across dsllama and 14B: the known leaked pods were cured by delete/recreate, and clean replacement pods did not reproduce the leak under the tested traffic matrix. Level 2 sleep did not reduce residue in these tests. Keep D8 detection+hygiene design; use soak to measure future leak frequency.

### Endgame F1 Next

- Complete F1.2 by restoring the missing `dsqwen-7b` GPU1/GPU2 Deployments in sequence and recording clean reconcile/GPU truth.

### Endgame F1.2 Full Topology Restoration

- Repaired live service-manager metadata drift before restoring topology:
  - `PUT /v2/models/dsllama-8b/target {"wake_replicas":1}` corrected the stale routable label on the sleeping GPU2 pod.
  - `PUT /v2/models/dsqwen-14b/target {"wake_replicas":1}` slept node10 GPU2-3.
  - The node9 GPU2-3 14B pod was already `/is_sleeping=true` but lacked complete TRE sleep metadata, so it was patched to `tre.aibrix.io/state=sleeping` and `tre.aibrix.io/routable=false`.
  - A follow-up `POST /v2/reconcile` returned no warnings.
- Restored `dsqwen-7b` GPU2:
  - A first direct recreate failed during vLLM sampler warm-up with CUDA OOM: qwen needed another `150 MiB` while only about `76 MiB` was free with dsllama+14B sleeping residue on GPU2.
  - Deleted the failed qwen GPU2 Deployment and temporarily deleted `dsqwen-14b` node9 GPU2-3 to free cold-start headroom.
  - Recreated qwen GPU2, waited for `/is_sleeping=false`, reconciled, and used `PUT /v2/models/dsqwen-7b/target {"wake_replicas":1}` to sleep it.
  - Recreated 14B node9 GPU2-3, waited for `/is_sleeping=false`, reconciled, and used `PUT /v2/models/dsqwen-14b/target {"wake_replicas":1}` to sleep it.
- Restored `dsqwen-7b` GPU1:
  - First slept all dsllama replicas with `PUT /v2/models/dsllama-8b/target {"wake_replicas":0}`.
  - A direct qwen GPU1 recreate still failed with the same 150 MiB sampler warm-up OOM because 14B node9 GPU0-1 sleeping residue remained on GPU1.
  - Deleted the failed qwen GPU1 Deployment and temporarily deleted `dsqwen-14b` node9 GPU0-1.
  - Recreated qwen GPU1, waited for `/is_sleeping=false`, reconciled, slept qwen back to target 1, then temporarily slept qwen target 0 to free GPU0/GPU1 for 14B cold start.
  - Recreated 14B node9 GPU0-1, waited for `/is_sleeping=false`, reconciled, slept 14B back to target 1, then restored qwen target 1 and dsllama target 1.
- Final F1.2 evidence directory: `docs/refactor/p11_evidence/f1_restore_20260705/`.
  - `sm_state.json`: `dsqwen-7b`, `dsllama-8b`, and `dsqwen-14b` are each `awake=1`, `bound=4`.
  - `reconcile.json`: version `241`, `warnings=[]`.
  - `probe_sleeping.txt`: all sleeping pods have `routable=false`; only the three awake pods have `routable=true`.
  - `node9_gpu_memory.txt`: node9 GPU memory is explainable by one awake qwen on GPU0, one awake dsllama on GPU1, and sleeping residue on all co-resident pods.
  - `gateway_smoke_20x3.jsonl`: gateway smoke passed with 20/20 requests and 0 errors for `dsqwen-7b`, `dsllama-8b`, and `dsqwen-14b`.
- Final endpoints:
  - `dsqwen-7b -> 10.244.3.53:8000`
  - `dsllama-8b -> 10.244.3.57:8000`
  - `dsqwen-14b -> 10.244.0.163:8000`

### Endgame F1 Next

- Proceed to F1.3 GPU truth provider and D8/D10 enforcement using TDD.

### Endgame F1.3 GPU Truth Provider And D8/D10 Offline Slice

- Ran the required read-only Plan A probe first:
  - Prometheus service exists at `prometheus/prometheus-kube-prometheus-prometheus` (`10.99.1.53:9090`).
  - `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_FB_USED{Hostname=~".*node9.*"}`, and `DCGM_FI_DEV_FB_USED{node=~".*node9.*"}` all returned an empty vector.
  - Prometheus metric-name discovery found no DCGM metrics.
  - `gpu-operator/nvidia-dcgm-exporter` Service endpoints exist, but `curl http://10.107.60.135:9400/metrics` returned HTTP 200 with `Content-Length: 0`.
  - Conclusion: Plan A is not available in the current cluster without modifying prometheus/gpu-operator, so F1.3 switched to Plan B.
- Added RED tests before implementation:
  - `test_gpu_truth.py`: `NullGpuTruth` fallback and `RedisGpuTruth` reading `tre:gpu_truth:<node>` payloads.
  - `test_gpu_truth_agent.py`: parsing `nvidia-smi --query-gpu=uuid,memory.used,memory.total` CSV and building the Redis payload.
  - `test_reconcile.py`: `sleep_leak:<serve_id>` warning when a sleeping-only GPU exceeds the truth threshold, and no warning when the GPU has an awake binding.
  - `test_api_v2.py`: runtime create fails before Deployment creation when GPU truth exceeds the startup headroom threshold.
- Implemented Plan B offline slice:
  - `tre_sm.gpu_truth.NullGpuTruth` and `RedisGpuTruth`.
  - `tre/deploy/scripts/gpu_truth_agent.py`, publishing `tre:gpu_truth:<node>` via `SETEX`.
  - service-manager reconcile now accepts `gpu_truth` and appends `sleep_leak:<serve_id>` warnings for sleeping-only GPUs over `TRE_SLEEP_LEAK_USED_MIB` (default `8192`).
  - service-manager runtime create/defrag create now checks `TRE_CREATE_MAX_USED_MIB` (default `2500`) before creating a Deployment when truth is available. Missing truth still falls back to the existing book-state behavior.
  - server wiring reuses the existing Redis connection for both state and `RedisGpuTruth`.
- Verification:
  - Focused tests passed: `29 passed`.
  - Full `cd tre && make check` passed: `246 passed`.

### Endgame F1 Next

- Build and roll a new service-manager image with the GPU truth provider, run the GPU truth agent once on node9/node10, and validate `POST /v2/reconcile` reports no false leak warnings in the healthy restored topology.

### Endgame F1.3 GPU Truth Rollout And Live Validation

- Fixed the Plan B agent for bare host execution:
  - node10 had the Python `redis` package, but node9 did not.
  - Added a standard-library Redis `SETEX` fallback using a RESP encoder, with RED/GREEN coverage in `deploy/tests/test_gpu_truth_agent.py`.
  - Full `cd tre && make check` passed with `247 passed`.
- Updated the tre-v2 service-manager overlay and overlay test to pin `tre-v2-service-manager:20260705-ba88b1b0`.
- Rebuilt the service-manager image and verified inside the image:
  - `python -m pytest -q service-manager/tests/test_gpu_truth.py service-manager/tests/test_reconcile.py service-manager/tests/test_api_v2.py deploy/tests/test_gpu_truth_agent.py`
  - Result: `30 passed`, one existing Starlette/httpx deprecation warning.
- Started GPU truth agents on both GPU nodes with local `/tmp` logs:
  - node9: `/tmp/tre_gpu_truth_agent_node9.log`, process evidence in `docs/refactor/p11_evidence/f1_gpu_truth_20260705/node9_agent_process.txt`.
  - node10: `/tmp/tre_gpu_truth_agent_node10.log`, process evidence in `docs/refactor/p11_evidence/f1_gpu_truth_20260705/node10_agent_process.txt`.
  - Redis TTL check showed both `tre:gpu_truth:<node>` keys refreshing.
- Rolled service-manager only via `kubectl -n tre-v2 set image`, leaving the paused controller untouched.
  - New pod: `tre-v2-service-manager-86b985cfbb-zntf2`.
  - `/healthz` returned `{"ok":true}` and `/v2/state` preserved the 12-binding topology.
- Live truth validation:
  - Healthy truth reconcile returned `warnings=[]`.
  - Synthetic Redis truth injection set node9 GPU2 `used_mib=24000`; `POST /v2/reconcile` returned expected `sleep_leak:` warnings for the three sleeping bindings sharing GPU2.
  - Re-running node9 agent restored real truth, and the next reconcile returned `warnings=[]`.
  - Evidence directory: `docs/refactor/p11_evidence/f1_gpu_truth_20260705/`.

### Endgame F1 Next

- Commit the service-manager rollout artifacts, then continue to F2 zm signal repair.

### Endgame F2.1 Metrics Baseline Lookback

- Completed F2 step 0 read-only measurement against production AIBrix Redis:
  - Key family: `aibrix:pod_histogram_metrics_*`.
  - Active pod sampled: `default/dsllama-8b-nscc-ds-4a100-node9-gpu-1-5579b75f9b-kh5fx`.
  - Recent samples: 50.
  - Adjacent timestamp intervals: min `0 ms`, p50 `5000 ms`, p95 `5000 ms`, max `5000 ms`.
  - Selected histogram lookback: `90000 ms`.
  - Recorded in `docs/refactor/03_metrics_pipeline.md`.
- Added RED tests for metrics baseline behavior:
  - v2 zset path: one in-window histogram doc plus pre-window baseline computes the correct delta.
  - v2 zset path: no in-window histogram doc yields `prompt_tokens=None` / `generation_tokens=None` while instant metrics still aggregate.
  - v1 legacy key path: one in-window histogram doc plus pre-window baseline computes the correct delta from parsed key timestamps.
- Implemented `MetricsStore(histogram_lookback_ms=90000)`:
  - `_read_zset_docs` and `_read_legacy_docs` include the last pre-window baseline document.
  - Histogram delta/avg/percentile functions require at least one in-window histogram metric before computing.
  - Token fields in `PodWindowMetrics` and `ModelWindowMetrics` are now `float | None`; model aggregation returns `None` only when no pod has token data for the window.
- Verification:
  - Focused `controller/tests/test_metrics_store.py`: `8 passed`.
  - Full `cd tre && make check`: `250 passed`.

### Endgame F2 Next

- Implement TRS/classification stale-hold and per-model planner drop policy.

### Endgame F2.2 TRS/Classification Stale Hold

- Added RED loop tests for token-missing windows after a valid paper-state window:
  - a single missing-token window holds the previous paper state, emits
    `paper_state_stale_hold:<model>`, and still allows the rescue scale action.
  - once the hold limit is exceeded, the model emits
    `paper_state_stale_unknown:<model>` and no scale action is submitted for that
    model in the focused case.
- Implemented `PaperStateCache` in the controller tick path:
  - valid token windows refresh the cached model context and reset staleness.
  - missing-token windows reuse the last complete context until the configured
    stale limit is exceeded.
  - live routable/assigned replica counts are refreshed on held contexts so the
    planner still sees current topology.
- Wired one persistent `PaperStateCache` per rescue/fairness async task. Focused
  `run_*_tick` helpers accept an explicit cache for deterministic unit coverage.
- Added `TRE_PAPER_STALE_MAX_WINDOWS` to `ControllerConfig`, default `3`, with
  positive-int validation. Rescue/fairness tasks read the centralized config
  field and keep the previous fallback for tests using small Protocol stubs.
- Verification so far:
  - RED config test failed as expected before implementation:
    `AttributeError: 'ControllerConfig' object has no attribute 'paper_stale_max_windows'`.
  - Focused loop/config tests: `34 passed`.
  - Controller focused set
    (`test_config.py`, `test_metrics_store.py`, `test_trs_signals.py`,
    `test_planner.py`, `test_loop_ticks.py`): `64 passed`.

### Endgame F2 Next

- Run full `cd tre && make check`, then commit F2.2 if green.
- Continue to F2.3 planner per-model incomplete drop policy.

### Endgame F2.3 Per-Model Incomplete Drop Policy

- Added RED planner tests for D9 policy split:
  - default behavior now drops only incomplete model classifications and allows
    complete models in the same cycle to plan actions.
  - `PlanConfig(incomplete_policy="drop_all")` preserves the legacy whole-cycle
    drop with event `paper_state_incomplete_drop_legacy_raw_trs`.
- Added RED config tests for `TRE_INCOMPLETE_POLICY`:
  - default `drop_model`.
  - env override `drop_all`.
  - invalid values rejected.
- Added a loop-level RED test proving `drop_all` can be passed through
  `run_rescue_tick`; this prevents live env overrides from being hidden by the
  planner default.
- Implemented:
  - `IncompletePolicy = Literal["drop_model", "drop_all"]`.
  - `PlanConfig.incomplete_policy`, default `drop_model`.
  - `_paper_state_incomplete_models()` returning the affected model names.
  - default per-model filtering with events
    `paper_state_incomplete_drop:<model>`.
  - compatibility `drop_all` early return using the legacy event and
    `dropped_legacy_raw_trs=True`.
  - `ControllerConfig.incomplete_policy` from `TRE_INCOMPLETE_POLICY`.
  - rescue/fairness/tick parameter plumbing into `PlanConfig`.
- Verification so far:
  - Initial RED failures matched expectations:
    missing `ControllerConfig.incomplete_policy`, default planner still
    dropping all, and `run_rescue_tick()` rejecting `incomplete_policy`.
  - F2 focused config/planner/loop tests: `49 passed`.
  - Controller focused set
    (`test_config.py`, `test_metrics_store.py`, `test_trs_signals.py`,
    `test_planner.py`, `test_loop_ticks.py`): `67 passed`.

### Endgame F2 Next

- Run full `cd tre && make check`, then commit F2.3 if green.
- Continue golden protection/rollout validation from the endgame plan.

### Endgame F2.4 Controller Image And Precheck Script Prep

- Built controller image on node10:
  - tag: `tre-v2-controller:20260705-7bfb0709`
  - image id: `sha256:f2bda68a49a0319ebbeddaf5c8c4424b462b28d57f3f24bb5dd42b6c1bcc203c`
  - source code commit: `7bfb0709`
- Verified inside the image:
  - `python -m pytest -q controller/tests/test_config.py controller/tests/test_metrics_store.py controller/tests/test_trs_signals.py controller/tests/test_planner.py controller/tests/test_loop_ticks.py`
  - result: `67 passed`.
- Collected `/tmp/n4b_three_model_precheck.py` into
  `tre/deploy/scripts/n4b_three_model_precheck.py` per the endgame rule that
  useful scripts must not live only in `/tmp`.
- Parameterized the precheck script:
  - `--gateway-url` / `N4B_GATEWAY_URL`
  - `--service-manager-url` / `N4B_SERVICE_MANAGER_URL`
  - `--models` / `N4B_MODELS`
  - duration, phase, workers, max tokens, sample interval, and request timeout.
- Added deploy tests for model parsing, argument overrides, and latency summary.
  The latency p95 was aligned with the existing E1 probe nearest-rank behavior.
- Updated the tre-v2 controller overlay to pin
  `tre-v2-controller:20260705-7bfb0709`; no `latest` tag introduced.
- Verification so far:
  - precheck/E1 script focused tests: `8 passed`.
  - deploy tests: `27 passed`.

### Endgame F2 Next

- Run full `cd tre && make check`, commit the image/overlay/precheck prep, then
  roll the controller and execute the 15-minute zm precheck.

### Endgame F2.4 Precheck Script Return Fix

- Rolled controller to `tre-v2-controller:20260705-7bfb0709`:
  - pod: `tre-v2-controller-d6d498484-tk57g`
  - node: `nscc-ds-4a100-node10`
  - restarts: `0`
  - image verified in Deployment template.
- Pre-roll reconcile was clean: `warnings=[]`; state remained
  `awake=1/bound=4` for all three models.
- First 15-minute precheck attempt did run live load, but the collected JSON was
  invalid (`null`) because the newly collected script built `result` without
  returning it. The script exited with
  `TypeError: 'NoneType' object is not subscriptable` at final status handling.
- Added a RED unit test for `run_precheck()` using monkeypatched kubectl/http/rss
  helpers and `duration=0/workers=0`; it failed with `NoneType`.
- Fixed `run_precheck()` to return the result dictionary.
- Verification:
  - `deploy/tests/test_n4b_three_model_precheck.py`: `4 passed`.
  - Full `cd tre && make check`: `260 passed`.

### Endgame F2 Next

- Commit the precheck script return fix, rerun the 15-minute precheck, and then
  analyze controller decision logs for `paper_state_incomplete_drop`, non-null
  `Z_m`, and stale-hold rate.

### Endgame F2.4 Decision Snapshot Z_m Evidence Fix

- Reran the 15-minute precheck after fixing script output. It completed
  successfully and wrote
  `docs/refactor/p11_evidence/f2_zm_precheck_20260705/three_model_precheck.json`.
- Precheck summary:
  - duration: `900.9s`
  - gateway errors: `{}`
  - ok counts: `dsqwen-7b=988`, `dsllama-8b=897`, `dsqwen-14b=1983`
  - tre-v2 component restarts stayed at `0`
  - post-run reconcile: `warnings=[]`
  - controller log counts over the run:
    `paper_state_incomplete_drop=0`, `paper_state_stale_hold=0`,
    `paper_state_stale_unknown=0`, `cluster_view_unavailable=0`.
- The run also exposed an evidence gap: `tre:v2:decision:latest` was a Redis
  hash containing only `ts_ms`, `loop`, `stale`, `submitted`, `actions`, and
  `events`; it did not include per-model `Z_m`, so the plan's "Redis decision
  records have non-null Z_m" acceptance item could not be proven from the
  artifact.
- Added RED tests for decision snapshot `model_states` serialization and for
  loop results carrying `model_contexts`.
- Implemented:
  - `LoopTickResult.model_contexts`.
  - `run_planner_tick()` returns the computed model contexts.
  - `DecisionSnapshotWriter` writes `model_states` JSON with `z_m`,
    `trs_z_m`, `signal_source`, and `signal_unavailable_reason` per model.
- Verification:
  - focused decision/loop tests: `23 passed`.
  - controller focused set including decision snapshot: `71 passed`.
  - full `cd tre && make check`: `260 passed`.

### Endgame F2 Next

- Commit the decision snapshot Z_m evidence fix, rebuild/roll controller, rerun
  the 15-minute precheck, and verify `model_states` has non-null `z_m` for all
  three models during active windows.

### Endgame F2.4 Controller Rebuild With Decision Model States

- Built replacement controller image on node10:
  - tag: `tre-v2-controller:20260705-bb37a230`
  - image id: `sha256:bccead0acbc836fabcbf9fcd15a89420c939474813a54f42ee05e149aac54892`
  - source code commit: `bb37a230`
- Verified inside the image:
  - `python -m pytest -q controller/tests/test_decision_snapshot.py controller/tests/test_loop_ticks.py controller/tests/test_config.py controller/tests/test_planner.py controller/tests/test_metrics_store.py controller/tests/test_trs_signals.py`
  - result: `71 passed`.
- Updated the tre-v2 controller overlay pin to
  `tre-v2-controller:20260705-bb37a230`; no `latest` tag introduced.

### Endgame F2 Next

- Run full `cd tre && make check`, commit the new image pin, roll controller,
  and rerun the 15-minute precheck for final F2.4 evidence.

### Endgame F2.4 Precheck Baseline Workers

- Rolled controller to `tre-v2-controller:20260705-bb37a230`:
  - pod: `tre-v2-controller-6d899d65f-scvv6`
  - node: `nscc-ds-4a100-node10`
  - restarts: `0`.
- Confirmed `model_states` is present in `tre:v2:decision:latest` and controller
  logs after rollout.
- A 15-minute alternating-only precheck completed successfully:
  - duration: `900.9s`
  - gateway errors: `{}`
  - tre-v2 restarts stayed at `0`
  - post-run reconcile: `warnings=[]`
  - `paper_state_incomplete_drop=0`, `paper_state_stale_hold=0`,
    `paper_state_stale_unknown=0`.
- However, strict wall-clock active-window parsing showed `z_m` was not non-null
  for every model on every loop. This is expected for an alternating-only load:
  inactive models can have zero-load metric windows where paper state is IDLE and
  `z_m` is null. Changing controller semantics to invent a Z value for idle
  zero-load windows would be the wrong fix.
- Added `--baseline-workers-per-model` /
  `N4B_BASELINE_WORKERS_PER_MODEL` to
  `tre/deploy/scripts/n4b_three_model_precheck.py`. Default remains `0`; the
  final F2.4 acceptance run will use `1` to keep all three models active while
  the main load still alternates.
- Added tests for the new CLI/result field.
- Verification:
  - `deploy/tests/test_n4b_three_model_precheck.py`: `4 passed`.
  - full `cd tre && make check`: `260 passed`.

### Endgame F2 Next

- Commit the precheck baseline-worker enhancement, rerun the 15-minute precheck
  with `--baseline-workers-per-model 1`, and parse final `model_states` evidence.

### Endgame F2.4 Final 15-Minute zm Precheck

- Before final precheck, reset all three models to target `awake=1`.
- A reconcile immediately after reset briefly reported `sleep_leak` warnings on
  node9 GPU2, but node9 `nvidia-smi` showed GPU2/GPU3 at `4070 MiB` each. The
  warning was caused by stale GPU truth data; running the node9 GPU truth agent
  once refreshed Redis and the next reconcile returned `warnings=[]`. No
  hygiene recreate was needed.
- Final precheck command:
  - `python3 tre/deploy/scripts/n4b_three_model_precheck.py --duration-seconds 900 --phase-seconds 60 --workers 4 --baseline-workers-per-model 1 --sample-seconds 30 --max-tokens 96`
  - output: `docs/refactor/p11_evidence/f2_zm_precheck_20260705/three_model_precheck_baseline.json`
  - controller log: `docs/refactor/p11_evidence/f2_zm_precheck_20260705/controller_since_baseline_precheck.log`
  - post-run reconcile: `docs/refactor/p11_evidence/f2_zm_precheck_20260705/post_baseline_reconcile.json`
- Final precheck result:
  - duration: `900.9s`
  - gateway errors: `{}`
  - ok counts: `dsqwen-7b=1694`, `dsllama-8b=1573`, `dsqwen-14b=3468`
  - final state: all three models `awake=1`, `bound=4`
  - tre-v2 component restarts stayed at `0`
  - post-run reconcile: `warnings=[]`
  - controller event counts:
    `paper_state_incomplete_drop=0`, `paper_state_stale_hold=0`,
    `paper_state_stale_unknown=0`, `cluster_view_unavailable=0`
  - decision `model_states` coverage during the run:
    `dsqwen-7b z_m non-null 255/259`,
    `dsllama-8b z_m non-null 255/259`,
    `dsqwen-14b z_m non-null 255/259`.
    The four null samples per model are the initial warm-up decisions before the
    first complete metrics window; after warm-up, Z state stayed populated.
- F2.4 conclusion: zm signal path is usable on live traffic with no incomplete
  drops and with decision snapshots now carrying per-model Z evidence.

### Endgame F2 Next

- Commit final F2.4 evidence, then proceed to F2.5 high-load zm scale-action
  validation from `14_endgame_plan.md` section 3.2 step 3.

### Endgame F2.5 High-Load zm Scale Validation - Inflight Gate Bug

- Ran the first 5-minute dsqwen-7b high-load validation:
  - command: `python3 tre/deploy/scripts/n4b_three_model_precheck.py --models dsqwen-7b --duration-seconds 300 --phase-seconds 300 --workers 16 --sample-seconds 15 --max-tokens 96`
  - output: `docs/refactor/p11_evidence/f2_zm_precheck_20260705/dsqwen7b_highload_5m.json`
  - controller log: `docs/refactor/p11_evidence/f2_zm_precheck_20260705/controller_since_dsqwen7b_highload.log`
  - post-run reconcile: `docs/refactor/p11_evidence/f2_zm_precheck_20260705/post_dsqwen7b_highload_reconcile.json`
- Result:
  - `dsqwen-7b` ok requests: `3691`
  - errors: one gateway `503`
  - component restarts stayed at `0`
  - post-run reconcile: `warnings=[]`
  - controller actions: `[]`
  - `dsqwen-7b` decision `z_m` during the run was critical
    (`min=0.515`, `p50=0.594`, `max=0.723`), so lack of scale action was not
    caused by insufficient load.
- Root cause found in `ActionQueue`:
  - `drain_once()` removed a model from `_inflight` only when dispatch returned
    `ok=True`.
  - There is no retry queue for failed dispatches, so one failed SM response can
    leave the model permanently inflight.
  - The planner then skips that model via `recv.model_name in inflight_models`,
    producing no action even when `Z_m` is critical.
- Added RED test replacing the old "keeps failed model inflight for retry"
  expectation with "releases failed model after dispatch attempt and accepts a
  later retry".
- Implemented the fix: `drain_once()` now discards the model from `_inflight`
  after every dispatch attempt, success or failure. The `DispatchResult` still
  records failure for observability.
- Verification:
  - focused queue/planner/loop tests: `37 passed`.
  - full `cd tre && make check`: `260 passed`.

### Endgame F2 Next

- Commit the ActionQueue inflight fix, rebuild/roll controller, and rerun the
  dsqwen-7b high-load zm validation.

### Endgame F2.5 Controller Rebuild With Inflight Fix

- Built controller image on node10:
  - tag: `tre-v2-controller:20260705-d795a715`
  - image id: `sha256:6b722a12a4aadb01dd3b485d5d537196deb337c0d4ebd7d63b54269b5eb118d3`
  - source code commit: `d795a715`
- Verified inside the image:
  - `python -m pytest -q controller/tests/test_action_queue.py controller/tests/test_loop_ticks.py controller/tests/test_planner.py controller/tests/test_decision_snapshot.py`
  - result: `41 passed`.
- Updated the tre-v2 controller overlay pin to
  `tre-v2-controller:20260705-d795a715`; no `latest` tag introduced.

### Endgame F2 Next

- Run full `cd tre && make check`, commit the image pin, roll controller, and
  rerun high-load zm scale validation.

### Endgame F2.5 High-Load zm Validation After Inflight Fix

- Rolled controller image `tre-v2-controller:20260705-d795a715` and reran the
  dsqwen-7b high-load validation:
  - command: `python3 tre/deploy/scripts/n4b_three_model_precheck.py --models dsqwen-7b --duration-seconds 300 --phase-seconds 300 --workers 16 --sample-seconds 15 --max-tokens 96`
  - output: `docs/refactor/p11_evidence/f2_zm_precheck_20260705/dsqwen7b_highload_after_inflight_fix.json`
  - controller log: `docs/refactor/p11_evidence/f2_zm_precheck_20260705/controller_since_dsqwen7b_highload_after_inflight_fix.log`
  - post-run reconcile: `docs/refactor/p11_evidence/f2_zm_precheck_20260705/post_dsqwen7b_highload_after_inflight_fix_reconcile.json`
- Result:
  - duration: `301.6s`
  - ok requests: `dsqwen-7b=3808`
  - errors: one gateway `503`
  - restart deltas: none
  - initial state: all three models `awake=1`, `bound=4`
  - final run state: `dsqwen-7b awake=3`, `bound=4`; other models stayed
    `awake=1`, `bound=4`
  - controller submitted `54` scale actions, all
    `kind=scale`, `model=dsqwen-7b`, `delta=1`,
    `reason=critical_sleeping_capacity`, `source_loop=rescue`
  - `dsqwen-7b z_m` during the run: `81` non-null samples,
    `min=0.539`, `p50=0.700`, `max=0.703`
- F2.5 scale-action conclusion: after the ActionQueue inflight fix, critical
  `Z_m` did produce zm-based scale actions and the live topology scaled qwen
  from one awake replica to three awake replicas. The remaining issue after the
  run was GPU hygiene when scaling back down, not planner action generation.

### Endgame F2.5 Post-High-Load D8 Hygiene

- Reduced qwen back to target `wake_replicas=1` after the high-load run. Manual
  node9 `nvidia-smi` inspection showed the sleeping qwen GPU2/GPU3 replicas
  still holding about `36.9 GiB` each, i.e. a true sleep leak rather than stale
  GPU truth.
- Paused controller before hygiene:
  - `kubectl -n tre-v2 scale deploy/tre-v2-controller --replicas=0`
- Hygiene actions:
  - saved pre-delete deployment/reconcile evidence under
    `docs/refactor/p11_evidence/f2_hygiene_after_highload_20260705/`
  - deleted the leaked qwen GPU2/GPU3 deployments and the co-resident 14B node9
    GPU2-3 deployment
  - recreated `dsqwen-7b` node9 GPU2, waited for HTTP health, reconciled, and
    set qwen target back to `awake=1`; `/is_sleeping` returned true
  - recreated `dsqwen-7b` node9 GPU3 with the same sequence; after sleep, node9
    GPU2/GPU3 each held about `2248 MiB`, matching two sleeping single-GPU pods
  - recreated `dsqwen-14b` node9 GPU2-3, waited for HTTP health, reconciled,
    set 14B target back to `awake=1`; `/is_sleeping` returned true
- Post-hygiene evidence:
  - `post_hygiene_reconcile.json`: `warnings=[]`
  - `post_hygiene_state.json`: all three models `awake=1`, `bound=4`
  - `post_hygiene_sleep_probes.json`: qwen GPU2, qwen GPU3, and 14B GPU2-3 all
    report `is_sleeping=true`
  - `node9_post_hygiene_nvidia.txt`: GPU2/GPU3 each at `4070 MiB`, with
    `1090 MiB` llama + `1054 MiB` qwen + `1766 MiB` 14B sleeping processes
  - `post_hygiene_recreated_deployments.yaml`,
    `post_hygiene_recreated_pods.txt`,
    `post_hygiene_controller_deploy.yaml`,
    `post_hygiene_controller_pod.txt`,
    `post_hygiene_controller_since_restore.log`
- Resumed controller after full topology was restored:
  - `kubectl -n tre-v2 scale deploy/tre-v2-controller --replicas=1`
  - controller pod: `tre-v2-controller-65bb7cdf46-7tnmp`
  - image: `tre-v2-controller:20260705-d795a715`
  - 35-second post-restore check: reconcile still `warnings=[]`, state still all
    three models `awake=1`, `bound=4`, node9 GPU2/GPU3 still `4070 MiB`.
- F2.5 hygiene conclusion: D8 delete/recreate cured the post-scale-down qwen
  sleep leak and restored a clean 12-binding topology. Proceeding to commit
  F2.5 evidence after full `make check`, then F3 live defrag.

### Endgame F3 Live Defrag Attempt - Route GC Blocker

- Entered F3 live defrag with full topology clean and controller on
  `tre-v2-controller:20260705-d795a715`.
- Constructed a live fragmented topology without cold-starting on an awake GPU:
  - paused controller
  - deleted qwen node9 GPU3 and 14B node9 GPU2-3
  - created temporary qwen node10 GPU2
  - woke llama GPU2
  - `SlotAllocator.find_slot(2)` returned `null`
  - `plan_defrag(2)` preview moved qwen node10 GPU2 -> node9 GPU3
  - sleeping 14B had no feasible wake slot before defrag
- First `/v2/defrag` live call succeeded at the API/path level:
  - action sequence: `hide`, `sleep`, `delete_deployment`,
    `create_deployment`, `wake`, `unhide`
  - post-defrag 14B node10 GPU2-3 became `feasible_wake=true`
  - waking 14B to two replicas succeeded
  - 14B gateway smoke after wake: `20/20`
- The first probe script was invalid because it omitted the required HTTP
  `model` header. This reproduced the old body-only 502 class and is not
  counted as defrag evidence.
- Retried with the correct `model` header. Defrag still succeeded, llama and
  14B had zero probe errors, but qwen had `48/133` gateway `503` during the
  migration window. Post-defrag qwen returned to `20/20`, so this was a
  migration-window availability issue.
- Root cause hypothesis: service-manager hid the source pod and immediately
  slept/deleted it before Kubernetes Service/EndpointSlice and AIBrix gateway
  routing had converged.
- Added RED test requiring runtime defrag to wait for the source pod to leave
  routable endpoints after `hide` and before `sleep`.
- Implemented:
  - `RuntimePodOps.wait_pod_unroutable(binding)`
  - `K8sOps.wait_pod_unroutable()` polling
    `model.aibrix.ai/name=<model>,tre.aibrix.io/routable=true`
  - `ServiceManagerV2._execute_runtime_defrag_migration()` now waits after hide
    before sleep/delete.
- Verification:
  - focused service-manager/controller tests passed.
  - full `cd tre && make check`: `261 passed`.
- Built and rolled service-manager:
  - image: `tre-v2-service-manager:20260705-46613bfb`
  - image id: `sha256:89f67902b350b7801ec6f213f204241abf942dd73973f2698f94a6324bfbc3fd`
  - in-image tests: `30 passed`.
- Retried F3 live defrag after the wait-unroutable fix. The SM defrag API still
  succeeded, but the live gateway criterion still failed:
  - qwen probe: `74 ok`, `53 errors`, including
    `dsqwen-7b-router not found` and connection-refused 503s
  - 14B probe: `127 errors`, including `dsqwen-14b-router not found`
  - llama probe: `127 ok`, `0 errors`
- New blocker: AIBrix model HTTPRoutes can disappear during TRE-managed model
  delete/recreate even while model Services and direct Service traffic remain
  healthy. This is the same route-GC class previously seen for qwen, now
  reproduced for qwen and 14B during live defrag. Waiting for source
  unroutable is necessary but insufficient.
- Restored missing model HTTPRoutes idempotently for qwen/llama/14B, performed
  D8 cleanup of affected node9 GPU2/GPU3 deployments, refreshed node9 GPU
  truth, and returned the cluster to a clean state:
  - service-manager image `tre-v2-service-manager:20260705-46613bfb`
  - controller image `tre-v2-controller:20260705-d795a715`
  - reconcile `warnings=[]`
  - all three models `awake=1`, `bound=4`
  - node9 GPU2/GPU3 each `4070 MiB`
  - final gateway smoke: `20/20` for qwen, llama, and 14B
- Controller is intentionally paused at the end of this attempt to avoid stale
  defrag/smoke metrics causing immediate rescue scale-up again.
- F3 conclusion: live defrag API and D10 headroom path are working, but N4.4
  cannot be marked PASS because the required gateway zero-5xx invariant is
  violated by AIBrix HTTPRoute GC during delete/recreate. Next work should
  implement an idempotent "ensure model HTTPRoute exists and accepted" guard in
  the TRE-managed model lifecycle before retrying F3.

### Endgame F3 Route Lifecycle Guard - TDD Implementation

- Added RED coverage for model HTTPRoute lifecycle:
  - manifest generation now expects one `aibrix-system/<model>-router`
    HTTPRoute per registry model, with `model` header matches forwarding to
    `default/<model>:8000`.
  - `K8sOps.ensure_model_httproute()` must create a missing route, patch an
    existing route back to the TRE-managed spec, and wait for `Accepted=True`.
  - runtime defrag must ensure all registry model routes before a real
    migration, and ensure the migrated model route again after create.
  - runtime target growth must ensure the target model route before and after
    creating its Deployment.
  - overlay RBAC must grant the service-manager only the minimal
    `aibrix-system` `httproutes.gateway.networking.k8s.io` permissions.
- Implemented the route guard in TRE service-manager only:
  - no AIBrix base, envoy, gpu-operator, or prometheus changes.
  - `tre/deploy/gen_model_manifests.py` is the shared HTTPRoute shape source.
  - `tre_sm.ops.K8sOps` now uses `CustomObjectsApi` for Gateway API
    HTTPRoutes and waits on the route status rather than relying on
    `kubectl wait`.
  - `tre_sm.server` injects `CustomObjectsApi`; route namespace and gateway
    name default to `aibrix-system` and `aibrix-eg`.
- Verification:
  - focused tests passed: `41 passed`.
  - full `cd tre && make check`: `264 passed`.
- Built the route-guard service-manager image and prepared rollout:
  - image: `tre-v2-service-manager:20260705-f6dce214`
  - image id: `sha256:853929dbd5f0d3eaef24f092ce660cad8b35308cb63c0564bfe8f12f3823fce5`
  - in-image route guard focused tests: `41 passed`, 1 warning.
  - overlay/test image pin updated away from
    `tre-v2-service-manager:20260705-46613bfb`.
  - full `cd tre && make check`: `264 passed`.

### Endgame F3 Re-verify With Route-Guard Image (preliminary, 2026-07-06)

- Picked up with SM on the route-guard image `tre-v2-service-manager:20260705-f6dce214`
  (commit `f6dce214`, rolled by `7337a039`). Controller intentionally paused
  (`replicas=0`).
- Live state on pickup had drifted from clean baseline (leftover from prior F3
  attempt): dsqwen-7b `awake=2`, dsllama-8b `awake=2`, dsqwen-14b `awake=1/bound=3`.
  node9 all four GPUs near-OOM (GPU0/1 ~39.9 GiB, GPU2/3 ~38 GiB used of 40).
- Route-guard functioning check:
  - all three model HTTPRoutes present/Accepted in `aibrix-system`
    (`dsqwen-7b-router`, `dsllama-8b-router`, `dsqwen-14b-router`).
  - gateway smoke (with required HTTP `model` header) `24/24` zero 5xx across the
    three models. No route-not-found. Confirms the guard image serves routing
    correctly and does not regress normal traffic.
- Restored a clean-ish baseline (also frees node9 for safety; needed for F4 anyway):
  - `PUT /v2/models/dsqwen-7b/target {"wake_replicas":1}` -> slept gpu-3 replica.
  - `PUT /v2/models/dsllama-8b/target {"wake_replicas":1}` -> slept gpu-2 replica.
  - Sleeps were HEALTHY (no leak): node9 GPU2/GPU3 dropped from ~38 GiB to
    `2248 MiB` each. Post reconcile `warnings=[]`, version 305.
  - Resulting state: dsqwen-7b `awake=1/bound=4`, dsllama-8b `awake=1/bound=4`,
    dsqwen-14b `awake=1/bound=3` (14b 4th binding node9 gpu2-3 still absent from
    prior attempt; immaterial because F4 rebuilds all bindings from manifests).

- **F3 preliminary conclusion / N4.4 disposition**: route-guard image is live and
  keeps all model HTTPRoutes Accepted with zero-5xx gateway routing; healthy
  sleep round-trip verified on the guard image. The full live fragmentation-defrag
  zero-5xx invariant is **deferred to F4.4 authoritative clean-cluster run** per
  plan section 5.5, for three reasons: (a) the plan itself designates F4.4 on the
  clean 0.7.0 base as the authoritative N4.4 PASS; (b) node9 GPU0/GPU1 remain
  near-OOM (a single awake pod each), so constructing + running a fragmentation
  defrag with cold-start creates on this node carries real OOM/destabilization
  risk (conservative-default per discipline 9.1); (c) F4 tears down this exact
  snowflake cluster, so investment here is throwaway. The route guard's
  create/ensure-HTTPRoute logic is unit-covered (`make check` 264 passed,
  41 focused route-guard tests). n4b-done tag is intentionally NOT set here; it
  is set after F4.4 per the plan.
- Controller left paused; cluster stable, reconcile clean.

### Endgame F4.0 Declarative Deploy Package (offline, TDD, 2026-07-06)

F4.0 makes every manual tre-v2 change declarative so F4.3 can bring the stack up
from manifests with zero手工 patch. All offline; cluster untouched.

- **gpu-truth DaemonSet** (biggest gap; was manual nohup on node9/node10):
  - `deploy/overlays/tre-v2/gpu-truth.yaml` = ConfigMap (`tre-v2-gpu-truth-agent`,
    embeds `deploy/scripts/gpu_truth_agent.py` byte-for-byte) + DaemonSet
    (`tre-v2-gpu-truth`). Reuses the vllm image already on both GPU nodes
    (python3 + nvidia-smi), NVIDIA_VISIBLE_DEVICES=all, hostPID:false,
    nodeSelector `nvidia.com/gpu.present=true`, control-plane toleration, writes
    `SETEX tre:gpu_truth:<node>` to `redis://tre-v2-redis:6379/0`.
  - Kustomize default load restrictor forbids a `configMapGenerator` reading
    `../../scripts/*`, so the ConfigMap is inlined and kept in sync by
    `deploy/gen_gpu_truth_manifest.py` + `deploy/tests/test_gpu_truth_daemonset.py`
    (asserts byte-equality with the script + generator reproducibility).
  - Wired into `overlays/tre-v2/kustomization.yaml`; `kubectl kustomize` builds.
- **models one-click apply**:
  - Regenerated `deploy/models/` via `make manifests`: the committed dir was
    stale (missing the 3 `<model>-router` HTTPRoutes the route-guard generator
    now emits). Now carries Services + HTTPRoutes + Deployments.
  - Added a TRE-managed cross-ns `ReferenceGrant`
    (`tre-v2-model-referencegrant-in-default`) to `gen_model_manifests.py` so
    `kubectl apply -k deploy/models` is self-sufficient (aibrix-system HTTPRoute
    -> default Service no longer depends on the AIBrix-base "reserved" grant
    surviving a 0.7.0 reinstall). +test.
  - `deploy/scripts/deploy_models.sh`: idempotent `kubectl apply -k deploy/models`
    (UUIDs are baked/stable; documents the collect+regen refresh path).
- **env/thresholds explicit in yaml** (F4.0.3): controller.yaml now declares
  TRE_SIGNAL_SOURCE=zm, TRE_PERCENTILE_MODE=bucket_upper, TRE_INCOMPLETE_POLICY,
  TRE_PAPER_STALE_MAX_WINDOWS, TRE_HIST_BASELINE_LOOKBACK_MS, cadence/window,
  ENABLE_TRE_SCALING; service-manager.yaml declares TRE_MODEL_NAMESPACE,
  TRE_ROUTE_NAMESPACE, TRE_GATEWAY_NAME, TRE_CREATE_MAX_USED_MIB=2500,
  TRE_SLEEP_LEAK_USED_MIB=8192. Overlay test asserts them.
  - Wired `TRE_HIST_BASELINE_LOOKBACK_MS` (was a hardcoded 90s MetricsStore
    default) through `ControllerConfig` (+`_get_nonneg_int`) and `app.py`; +tests.
- **images.lock.md** (F4.0.4): recorded SM/controller/ui/redis/vllm tags + local
  build digests; noted the controller image must be rebuilt from post-F4.0 HEAD
  before F4.3 (benign env mismatch until then).
- Gate: `cd tre && make check` **268 passed** (was 264, +4). All 5 overlays +
  models `kubectl kustomize` build clean. Cluster untouched (controller still
  paused, SM/redis/ui Running).

### Endgame F4.1/F4.2 BLOCKED — shared aibrix-system base (2026-07-06)

Before the F4.2 teardown I audited base ownership (plan F4.2 mandates "逐类确认归属").
Finding: `aibrix-system` is a **shared multi-tenant** AIBrix base, not a TRE
snowflake:
- shared base: aibrix-controller-manager / gateway-plugins / redis-master /
  autoscaling-controller / gpu-optimizer / kuberay / metadata / orchestration
  (290d, restarted ~4d14h);
- other tenant lxt: `lxtaibrix-gateway-plugins`, `lxt-aibrix-eg`,
  `lxt-aibrix-reserved-router` (~4d15h, Running);
- `qwen-coder-router` + `qwen-instruct-router` (373d) share OUR `aibrix-eg` gateway;
- AIBrix CRDs cluster-scoped + shared; TRE controller reads shared
  `aibrix-redis-master`.
Uninstalling/reinstalling this base would break third-party production workloads
and force a CRD/controller version change on all co-tenants → violates red line
2.2 and F4.2's own stop-and-ask caveat. Recorded ADR-0007; **stopped the
destructive path, did not touch the base.** Surfacing to the architect: options
= (1) isolated `aibrix-tre` 0.7.0 base in a new namespace, (2) coordinated
maintenance window with co-tenant sign-off, (3) accept current base for N5 with a
version caveat. Continuing with non-blocked work (live validation of the F4.0
package, N5 driver tooling) while awaiting the decision.

### Endgame F4.0 live validation + ADR-0008 kickoff (2026-07-06)

Architect (Fable5) ruling recorded as ADR-0008: cancel D11 base teardown; instead
stand up a minimal isolated TRE data plane in tre-v2 (own Gateway tre-aibrix-eg +
tre-gateway-plugins + policies + retarget), mirroring the lxt tenant. Phases
A(additive)/B(cutover)/C(cleanup); execution model may proceed without further gate
unless Phase A smoke fails.

Live-validated the additive F4.0 pieces (non-destructive, per ADR-0007/0008 item 2):
- **gpu-truth DaemonSet** deployed to tre-v2 and now the sole GPU-truth source:
  - First apply used nodeSelector `nvidia.com/gpu.present=true` which matched a
    THIRD out-of-scope A100 node ("cloud", not in registry) -> CrashLoop there.
    Fixed the generator template to nodeAffinity hostname In {node9,node10}
    (consistent with registry + existing SM/controller nodeSelector); +test updated;
    regenerated; make-check-local green.
  - DaemonSet 2/2 Running on node9/node10; writes `tre:gpu_truth:<node>` with correct
    per-UUID used/total MiB (matches nvidia-smi), TTL refreshing (~30s interval /
    120s TTL). Retired the two manual Jul05 nohup agents (node9 PID 3837984,
    node10 PID 4188585). SM reconcile warnings=[] served purely by the DaemonSet.
- **TRE-managed ReferenceGrant** `tre-v2-model-referencegrant-in-default` applied;
  coexists with the AIBrix-base reserved grant (additive). Makes `apply -k models`
  self-sufficient for cross-ns routing.
- Note: did NOT `apply -k deploy/models` wholesale (would cold-start the missing
  14b node9-gpu-2-3 as an awake pod); model Deployment/route idempotence is unit-
  tested and the live routes already exist (route-guard). The model HTTPRoutes will
  be regenerated to the tre-v2 gateway in the isolation package (ADR-0008 step 3).

### Endgame F4 Isolation Package + Phase A (ADR-0008, 2026-07-06)

Architect (Fable5) build spec (SendMessage follow-up): ext-proc is NOT on TRE's
serving path (per-model routers win match precedence -> Service -> routable pods);
zm metrics come from a background scrape loop in gateway-plugins
(AIBRIX_POD_METRIC_REFRESH_INTERVAL_MS) writing to REDIS_HOST, independent of
routing. So the isolated plane needs only: Gateway + per-model routers (serving)
and tre-gateway-plugins + pod-list RBAC + REDIS_HOST=tre-v2-redis (metrics). NO
reserved-router / ext-proc policy / skip-ext-proc / gwp :50052 Service.

Isolation package (offline TDD, make check 272):
- `deploy/overlays/tre-v2/gateway.yaml`: Gateway `tre-aibrix-eg` (gatewayClassName
  aibrix-eg, listener :80, allowedRoutes from Same).
- `deploy/overlays/tre-v2/gateway-plugins.yaml`: SA + cloned tre-scoped ClusterRole
  (pods/modeladapters/httproutes) + binding + Deployment (pinned image
  `aibrix/gateway-plugins:20260704-0d869b49-nozmq2`, REDIS_HOST=tre-v2-redis,
  TRE_REDIS_SCHEMA=dual, scraper env carried over, SERVEMENT_URL dropped, init-c
  waits on tre-v2-redis) + Service (metrics/profiling only, no :50052).
- Parameterized `gen_model_manifests.py` gateway ns/name (env/CLI; defaults
  tre-v2/tre-aibrix-eg); regenerated model routers into tre-v2; ReferenceGrant
  `from: tre-v2`.
- Retargeted SM env TRE_ROUTE_NAMESPACE=tre-v2 / TRE_GATEWAY_NAME=tre-aibrix-eg;
  controller TRE_METRICS_REDIS_URL=redis://tre-v2-redis:6379/0. Route-guard
  K8sOps default now tre-v2/tre-aibrix-eg. New test `test_isolated_dataplane.py`;
  images.lock updated.

Phase A (additive live deploy, touched nothing in aibrix-system, controller stayed
paused):
- Applied gateway + gateway-plugins + 3 tre-v2 routers + tre-v2 ReferenceGrant.
- `tre-aibrix-eg` listener Programmed/Accepted/ResolvedRefs; dedicated envoy proxy
  `envoy-tre-v2-tre-aibrix-eg-*` Running; envoy ClusterIP 10.103.92.7:80.
- **Smoke via NEW gateway: 24/24** (8/8 each model), zero errors.
- **zm metrics isolation confirmed**: tre-gateway-plugins writes
  `aibrix:pod_instant_metrics_default/<pod>_<ts>` to tre-v2-redis with ts advancing
  (…510000 → …520000 → …530000), 88 histogram keys present. Controller (paused)
  is not the writer -> proves the isolated scraper works.
- **Old path unaffected**: shared aibrix-eg smoke 12/12. Both scrapers run
  concurrently during transition (shared -> aibrix-redis-master, tre -> tre-v2-redis).
Phase A PASS. Proceeding to Phase B (cutover).

### Endgame F4 Phase B — cutover to isolated plane (ADR-0008, 2026-07-06)

Rolled SM + controller onto the isolated data plane:
- **SM**: applied retargeted service-manager.yaml (TRE_ROUTE_NAMESPACE=tre-v2,
  TRE_GATEWAY_NAME=tre-aibrix-eg; D8/D10 thresholds now explicit). Rolled clean,
  reconcile warnings=[]. Route-guard now manages the tre-v2 model routes.
- **Controller**: applied retargeted controller.yaml (TRE_METRICS_REDIS_URL=
  redis://tre-v2-redis:6379/0) and unpaused (replicas 0->1). Verified live:
  - controller reads zm from **tre-v2-redis** and populates Z_m
    (dsllama 0.044 / dsqwen-14b 0.061 / dsqwen-7b 0.044 — near-idle values, no load).
  - `tre:v2:decision:latest` present (controller writing decisions).
  - **Stability**: SM version steady at 311 across repeated reads; all models
    awake=1/bound; no pod churn; reconcile warnings=[]; tre-v2 components 0 restarts.
- Known idle characteristic (NOT a cutover regression): with zero experimental
  load, TRS→0 so Z_m→0 reads as CRITICAL and rescue submits scale+1 while
  idle-proactive submits scale-1; both are no-op'd by the serving floor
  (min awake=1) + D10 headroom, so SM state stays fixed (version stable) — no
  wake/sleep churn, hence no D8 leak risk. This is the same idle behavior that
  previously motivated pausing the controller; under real N5 load the signals
  reflect true utilization and the controller scales correctly (proven in F2.5:
  dsqwen-7b 1->3 awake under load).

**Phase B PASS.** Isolated TRE data plane is live & operational: serving via
tre-aibrix-eg (smoke 24/24), metrics via tre-gateway-plugins -> tre-v2-redis,
SM + controller retargeted, all state on tre-v2-redis. TRE no longer depends on
the shared aibrix-eg gateway or aibrix-redis-master for its data/control path.

**Phase C (deferred, >=24h gate)**: after >=24h stable, delete ONLY the 3 legacy
TRE routers in aibrix-system (dsqwen-7b/14b-router, dsllama-8b-router). Touch
nothing else there; leave the shared gateway-plugins image (co-tenants depend on
it). The old aibrix-eg path currently still serves TRE traffic too (harmless
redundancy) until Phase C.

**Next (F4.4 authoritative N4b on the clean isolated plane)**: N4.2 hot-switch
(20 rounds), N4.4 live defrag zero-5xx (now with route guard + isolated gateway),
N4.6 three-model scale exercise (zm) + 12h soak (76 local disk). Then tag
n4b-done. Controller rebuild from post-F4.0 HEAD before authoritative runs
(images.lock note; current d795a715 image + runtime env is functionally
equivalent — only the histogram-lookback env-wiring differs, old image defaults
to the same 90s).

### Endgame session checkpoint (2026-07-06) — isolated plane live, F4.4/N5 remain

Informal N4.2 spot-check on the isolated gateway (NOT the authoritative result):
20 wake/sleep rounds on dsqwen-7b through tre-aibrix-eg gave 20/20 gateway probes
and node9 GPU2/GPU3 returned to 2248 MiB (no leak from wake->minimal-traffic->sleep,
consistent with D8). The wake-latency numbers were invalid (my poller measured the
SM awake-flag flip, not the vLLM /wake_up weight-load, and probes were served by the
always-awake replica). Authoritative N4.2 needs per-pod /is_sleeping instrumentation
— deferred into F4.4 proper.

CURRENT SYSTEM STATE (end of session):
- Isolated TRE data plane LIVE on the shared cluster (no third-party impact):
  serving via Gateway `tre-aibrix-eg` (envoy 10.103.92.7:80), metrics via
  `tre-gateway-plugins` -> `tre-v2-redis`, SM retargeted (route ns tre-v2), all
  state on tre-v2-redis, gpu-truth DaemonSet live. reconcile warnings=[].
- Controller: **intentionally PAUSED (replicas=0)** after proving cutover — safe
  idle state (avoids idle no-op-thrash + any residual leak risk overnight). All
  models awake=1/bound; tre-v2 components 0 restarts.
- Old shared aibrix-eg path still also serves TRE (harmless redundancy) pending
  Phase C.
- git: clean working tree; all F3/F4.0/F4-isolation/PhaseA/PhaseB work committed
  (HEAD 361dbf0d area). `cd tre && make check` = 272 passed.

EXACT NEXT CONTINUATION (next session):
1. F4.4 authoritative N4b on the isolated plane (unpause controller = step 1):
   - N4.2 hot-switch 20 rounds with CORRECT wake-P95 (poll per-pod /is_sleeping),
   - N4.4 live defrag zero-5xx through tre-aibrix-eg (route guard + isolated gw;
     node9 GPU2/GPU3 now have headroom so a fragmentation-defrag is feasible),
   - N4.6 three-model scale exercise (zm) + 12h soak (76 local disk),
   - rebuild controller image from post-F4.0 HEAD first (images.lock note),
   - then tag n4b-done.
2. Phase C (>=24h after Phase B): delete ONLY the 3 legacy TRE routers in
   aibrix-system; touch nothing else.
3. N5 (R1->R3->R7->R2->R4->R5, R6 gap): experiments on the isolated plane; log in
   13_experiments_log.md; tag results-v1.
4. F5: paper data packaging + doc final check; tag tre-v2-1.0.
Reference: ADR-0008 (isolated plane design + phases), 13_experiments_log.md (N5),
images.lock.md (controller rebuild), architect = Fable5 model (consult via subagent).

### Endgame F4.4 N4.6 pre-flight — BLOCKED: idle-critical theta + SM routable desync (2026-07-06)

Ran a 200s N4.6 scale-exercise pre-flight (driver `n4b_scale_exercise.py`) on the
isolated plane with the controller unpaused, BEFORE launching the 12h soak. It
exposed two real problems and I aborted the soak plan:

1. **Idle/low-load reads as CRITICAL with inherited theta_m** (TRS degeneracy at
   low throughput: Y_m->0, Q_ctl->qmin => TRS->0 => Z_m~0.04 << tau_crit). With
   the driver's 1 req/s baseline, ALL THREE models scaled 1->2 (not just the
   saturated one), and they never shrank (baseline stays "critical"). N4.6's
   expand-then-shrink criterion and the soak's ">=5 clean expand/shrink cycles"
   are UNACHIEVABLE with the current inherited theta. This is a calibration
   dependency: meaningful N4.6/soak needs R3-refit theta (or a load design whose
   baseline sits in the HEALTHY zm band).
2. **SM routable/awake desync bug under scaling churn**: the over-scale on the
   near-full node9 hit the "single awake per GPU" invariant, SM auto-slept pods,
   and afterwards the SM binding.awake RECORD diverged from pod reality
   (`is_sleeping`), AND awake pods were left routable=false -> Services lost
   endpoints -> dsllama-8b and dsqwen-7b returned 0/4 through the gateway. Neither
   reconcile nor idempotent PUT target self-healed it (reconcile only warns
   "pod reality overrides persisted binding"; it does not re-assert routable from
   reality). Root cause class: routable label is not reconciled from actual
   `is_sleeping` state. FLAGGED as an SM bug for a code fix (add a
   "reconcile routable from is_sleeping reality" path).

Remediation (cluster restored to serving):
- Paused controller; scaled all models target=1.
- D8-deleted a genuinely leaked dsllama-8b node9 gpu-0 (is_sleeping=true yet
  38-40 GiB resident, /wake_up 500) -> node9 GPU0 partly freed. dsllama now
  bound=3 (leaked binding removed; recreate deferred).
- Deterministic routing repair: labeled each model's ACTUALLY-awake pod
  (is_sleeping=false) routable=true. Final: **all 3 models 5/5 via tre-aibrix-eg**,
  awake=1 each (7b bound=4, llama/14b bound=3). node9 GPU2/GPU3 = 2248 MiB (clean),
  node10 GPU2/GPU3 = 1825 MiB (clean, no persistent leak). Controller PAUSED
  (stable; no actor will churn the state).
- NOTE: SM binding.awake record may still point at different pods than reality
  (desync); a reconcile could re-break routing, so left un-reconciled while serving.

Process note: I ran this load pre-flight before consulting on the theta/idle
question — that was premature and caused avoidable churn. Consulting the architect
now on (a) F4.4-vs-R3 ordering and (b) the SM routable-desync fix before any
further live scaling.

### Endgame SM two-layer reconcile fix DEPLOYED + verified live (ADR-0009, 2026-07-06)

- Subagent implemented the ADR-0009 two-layer reconcile (TDD). make check 272->285
  (+13: 6 reconcile Layer1/2, 4 vllm_ops.is_sleeping, 3 k8s_ops routable). Reviewed:
  physical /is_sleeping overrides the state annotation (write-through cache);
  Layer 1 `_enforce_routable_labels` sets routable = physical-awake AND not hidden,
  patch-on-diff (idempotent), single-pass (leaked pods stay non-routable + D8 warning).
  Wired in api/v2.py (prober=_VllmPodProber(vllm_ops), label_writer=runtime_ops) and
  server.py (vllm_ops=VllmOps(), runtime_ops=k8s_ops, k8s_client threads pod_ip+routable).
  Commit a1d21c00.
- Built + rolled SM image `tre-v2-service-manager:20260706-a1d21c00`.
- Live behavior on roll: exposed that the earlier churn had left the awake serving
  replicas of dsqwen-7b (gpu-0) and dsqwen-14b (node10 gpu0-1) stuck in state=hidden;
  the fix CORRECTLY made them non-routable (routable = awake AND not hidden) -> 0/4.
  Root cause was the stale hidden state, not the fix. Cleared via
  `PUT /v2/models/{m}/routable {"hidden_pods":[]}` (unhide) -> reconcile re-asserted
  routable=true -> all 3 models 4/4.
- **Acceptance test (self-heal) PASS**: manually set dsqwen-7b gpu-0 routable=false
  -> reconcile re-asserted routable=true from physical /is_sleeping -> serving 4/4.
  The desync-outage bug is fixed and self-healing live. reconcile warnings=[].
- Cluster: all 3 models serving 4/4, awake=1; dsllama-8b bound=3 (leaked gpu-0
  deleted; recreate pending — node9 GPU0 currently hosts dsqwen-7b awake, so the
  architect's "gpu-0 freed" precondition for recreate is not yet met; canonical
  fleet restore is the next step, requires relocating dsqwen-7b awake off GPU0 first).
  Controller PAUSED.

### Endgame canonical fleet restore — partial; fix validated under churn (2026-07-06)

Attempted ADR-0009 step 2 (restore dsllama-8b to canonical bound=4) with the SM
fix now deployed:
- **Freed node9 GPU0** by relocating dsqwen-7b's awake replica gpu-0 -> gpu-2:
  direct `/wake_up` gpu-2, reconcile, then direct `/sleep` gpu-0. The two-layer
  fix handled this flawlessly — each reconcile's physical `/is_sleeping` override
  ("pod reality overrides persisted binding") synced the store + routable labels
  to reality, and dsqwen-7b served continuously (3/3) throughout the relocation.
  GPU0 dropped 38800 -> 2942 MiB. **Strong live validation of the fix under real
  wake/sleep churn** (the exact scenario that previously caused the desync outage).
- **dsllama gpu-0 recreate FAILED (CUDA OOM CrashLoop, 3 restarts)**: raw
  `kubectl apply` of the Deployment bypassed the SM D10 headroom gate; during the
  ~2min vLLM load a 36 GiB process re-occupied GPU0 (leaving 112 MiB), so the new
  pod OOM'd on sampler warmup. Deleted it -> dsllama back to clean bound=3,
  serving 4/4. node9 GPU0 settled to 2942 MiB, reconcile warnings=[].
- **Canonical restore deferred** (next session, before R3): recreate the 4th
  dsllama binding via the SM create path (D10 headroom check + wait-vLLM-ready),
  NOT raw apply; ensure GPU0 is guaranteed free for the full load with no
  concurrent wakes. dsllama bound=3 serves fine meanwhile; only R3's canonical
  capacity-surface measurement needs the 4th binding.

Follow-up refinement noted (minor, non-blocking): Layer 1 marks a physically-awake
-but-still-loading pod routable (is_sleeping=false during vLLM init) -> it briefly
receives traffic before ready (observed dsllama 2/3 during the recreate load).
routable should arguably also gate on vLLM HTTP readiness, not just not-sleeping.
Only manifests during the rare pod-creation window; add a readiness check to the
routable invariant in a future SM iteration.

CLEAN CHECKPOINT: isolated plane serving all 3 models 4/4; SM two-layer reconcile
fix deployed (tre-v2-service-manager:20260706-a1d21c00) + verified (self-heal +
under-churn); controller PAUSED; node9/node10 GPUs clean; reconcile warnings=[];
make check 285; tree committed. Next: canonical dsllama gpu-0 (SM create path) ->
R3 refit -> N4.6 + scale-cycle soak -> n4b-done -> Phase C -> N5 -> F5.

### Endgame canonical restore — root-caused, blocked on D8 config decision (2026-07-06)

Retried dsllama gpu-0 recreate on a settled clean GPU0 (2942 MiB). It OOM'd again
(restarts, never HTTP-ready). Precise root cause (from vLLM logs):
  "CUDA out of memory ... warming up sampler with 256 dummy requests ... GPU 0 has
   112 MiB free ... total 39.38 GiB"
At gpu_memory_utilization=0.9 the pod targets ~36 GiB; GPU0 already hosts co-resident
sleeping pods (dsqwen-7b gpu-0 ~2 GiB + dsqwen-14b node9 gpu0-1 ~1.7 GiB) => ~40 GiB,
and the sampler warmup pushes over the 40 GiB card limit. The ORIGINAL 12-binding
topology avoided this via creation ORDER (each GPU's pods warmed up at 0.9 while the
GPU was still empty, then slept to ~2 GiB before the next pod was added). Recreating
a single binding into an already-populated GPU breaks that assumption.

This is exactly the plan D8 case: "only if cold-start headroom is insufficient, add
`--gpu-memory-utilization 0.85` to registry vllm_extra_args (rebuilds all model pods,
costly, don't do lightly)". Options for the canonical restore (needs a deliberate
decision, ideally architect-confirmed, at the R3 session start):
  (a) D8 path: set gpu_memory_utilization=0.85 in registry vllm config, regenerate
      all model manifests, recreate the fleet cleanly in creation-order. Consistent
      but disruptive (rebuilds all 12 pods). 0.85*40=34 GiB + 4 co-resident = 38 < 40.
  (b) One-off 0.85 on just the dsllama gpu-0 recreate manifest (fits, but util
      inconsistent with the other pods; acceptable if gpu-0 stays a sleeping binding,
      but affects R3 capacity measurement if it becomes the awake replica).
  (c) Temporarily evacuate GPU0 co-residents, recreate dsllama gpu-0 at 0.9 when GPU0
      is empty, sleep it, restore others (complex: 14b gpu0-1 is a 2-GPU binding).
Recommendation: (a) at R3 start (R3 rebuilds/recreates the fleet for the capacity
grid anyway, so folding the 0.85 change in is low marginal cost and consistent).

Deleted the failing recreate -> dsllama clean bound=3, serving 3/3, reconcile clean.
CLEAN CHECKPOINT holds: isolated plane serving 3/3, SM two-layer fix deployed+verified
(self-heal + under-churn), controller PAUSED, GPUs clean, make check 285, tree committed.
Canonical bound=4 + R3 + N4.6/soak + Phase C + N5 + F5 remain (config decision + wall-clock).

### Endgame canonical restore — dsllama bound=4 via D8 0.85 (2026-07-06)

Applied ADR-0010 (gpu_memory_utilization=0.85, registry + all manifests, make check
285) and recreated dsllama-8b node9 gpu-0 with the 0.85 manifest:
- Loaded cleanly with NO OOM (0.85*40 = 34 GiB + ~4 GiB co-resident sleeping = 38 < 40),
  vLLM became ready, slept it -> node9 GPU0 = 4070 MiB (3 healthy sleeping pods).
- reconcile version 379, no non-leak warnings; **dsllama-8b now bound=4**; all 3
  models serving 4/4 via the isolated gateway. D8 0.85 path VALIDATED live.
- Canonical fleet now 11/12 bindings: dsqwen-7b bound=4, dsllama-8b bound=4,
  dsqwen-14b bound=3 (missing node9 gpu2-3). The 14b 2-GPU binding needs node9
  GPU2+GPU3 free of awake, but GPU2 currently hosts the relocated dsqwen-7b awake;
  restoring it needs another relocation (wake dsqwen-7b elsewhere -> sleep its GPU2
  replica -> create 14b gpu2-3 at 0.85). Deferred (optional pre-N4.6; 14b serves
  fine at bound=3, and R3 single-pod capacity is unaffected by bound count).

CLEAN CHECKPOINT: isolated plane serving 4/4; SM two-layer fix deployed+verified
(self-heal + under-churn); D8 0.85 applied + validated; dsllama bound=4 restored;
controller PAUSED; GPUs clean; reconcile clean; make check 285; tree committed.
Remaining: 14b gpu2-3 (optional) -> R3 refit (recreate fleet at 0.85 for consistent
capacity) -> N4.6 + scale-cycle soak -> n4b-done -> Phase C -> N5 -> F5 (wall-clock).

### Signal Plan S1.3 — TRS EMA -> wall-clock time-constant (ADR-0011) DONE (offline) 2026-07-06

Context handoff: resumed on 76 (/data/nfs_shared_data/xxy/aibrix). First priority per
new goal = 15_signal_and_window_plan.md S1 (TSS lag fix), strict order S1.3->S1.1->S1.2->S1.4.

**Premise error found & escalated (architect Fable5, verified independently):** doc 15 §0
claimed the live TRS EMA advances ~every 60s with ema_alpha=0.2485 tuned for that. Code
fact: the live path builds a FRESH TRSComputer every tick (tick.py:_model_contexts:243,
safescale_task.py:_observation_from_metrics:92) with no EMA restore anywhere
(state_store = SafeScale-probes-only; app.py only safescale.restore()). So live
TRS == TRS_raw always and ema_alpha had ZERO live effect — only the 60s tumbling window
smoothed. Architect ruling: **Option A-minimal** (build a real shared, in-process,
wall-clock time-constant EMA now; it is S1.2's precondition, not separable). Full ruling
-> ADR-0011.

**Implemented (TDD, make check 305 passed, was ~289):**
- signals/trs.py: TRSComputer.ema_tau_ms + compute(window_end_ms=...);
  time-constant `_update_ema` (decay=exp(-dt/tau) over window_end_ms deltas) with a
  per-window dedup guard shared by both tau and legacy-alpha branches (advance once per
  distinct window_end_ms). Legacy path (tau None, no window_end_ms) byte-identical ->
  golden legacy_trs unchanged & green. New `SignalState` (per-model shared computer holder).
- common/registry.py: TrsParams gains optional ema_tau_ms; parsed from registry.yaml.
- deploy/registry.yaml: ema_tau_ms=20000 seeded for all 3 models (starting point; frozen at S1.2).
- loops tick/rescue/fairness/safescale + app.py: thread SignalState from
  create_controller_dependencies through the three loops (one EMA per model, rescue+fairness+
  safescale share it). signal_state optional -> back-compat (None = fresh-per-tick raw).
- Tests: controller/tests/test_trs_ema_timeconstant.py (8: seed=raw, decay formula, alpha<->tau
  equivalence at 60s & 5s, dup-window hold, non-finite passthrough, legacy branch, SignalState
  sharing) + test_signal_state_loops.py (3: EMA persists across ticks, rescue-then-fairness
  same snapshot no double-advance, no-signal_state == raw). test_controller_app deps updated.

**Recorded:** ADR-0011 (DECISIONS.md); doc 15 §0 dated correction note; 05_paper_vs_impl.md
EMA contract. Bonus findings logged for later: (a) SaturationGuard/gamma also dead live
(out of S1.3 scope); (b) r3_grid trs column is within-cell EMA'd vs live raw — R3 must
replicate live EMA semantics (S1.4 gate strengthened).

Next: S1.1 (sliding window in metrics_task, + window_cache leak fix). Controller still PAUSED.

### Signal Plan S1.1 — sliding window in metrics_task DONE (offline) 2026-07-06

Removed the 60-120s tumbling staleness. make check 310 passed (was 305).
- loops/metrics_task.py: new `_sliding_window(now, W) -> (max(0,now-W), now)` (no epoch
  align, no last-complete block). `refresh_metrics_once(..., window_mode="tumbling")` —
  default kept tumbling so existing callers/tests are byte-identical; live controller passes
  cfg.metrics_window_mode. Tumbling calls read_snapshot with the ORIGINAL signature (no
  use_cache kwarg) so existing SnapshotStore fakes keep working; only the sliding path passes
  use_cache=False. MetricsTaskConfig protocol += metrics_window_mode; SnapshotStore protocol
  read_snapshot += use_cache.
- store/metrics_store.py: read_snapshot/read_model_window += `use_cache: bool = True`.
  Sliding passes False -> the per-window `_window_cache` is not read or written (every sliding
  window is unique -> cache never hits and would grow unboundedly = leak). Tumbling unchanged.
- config.py: WINDOW_MODES={tumbling,sliding}; ControllerConfig.metrics_window_mode
  (env TRE_METRICS_WINDOW_MODE, default **sliding**, validated).
- Tests: test_metrics_task.py (_sliding_window value, sliding reads [now-W,now] with
  use_cache=False, tumbling default uses cache) + test_metrics_store.py (100 distinct sliding
  windows -> _window_cache stays size 0) + test_config.py (default sliding + validation).

Regression caught & fixed during make check: initially refresh_metrics_once passed use_cache on
EVERY read; test_p9_offline_integration's own FixtureSnapshotStore.read_snapshot lacks that
kwarg -> TypeError swallowed by the stale-fallback try/except -> empty decisions -> 2 failures.
Fixed by only passing use_cache on the sliding path (tumbling call byte-identical).

**Real-machine deferred to S1.2**: S1.1's live check (Z_m changes every refresh, no 60s step)
is folded into S1.2's combined real-machine acceptance that freezes W — avoids rolling the
PAUSED controller twice (once for sliding@60s, again for short-window@30s+5s).

Next: S1.2 (metrics_refresh_interval_s=5s decoupled from monitor_interval; window default
60000->30000; N1 min-sample guard for p95; N5 window aligns to write period). Controller PAUSED.

### Signal Plan S1.2 (core) — single shared short window + 5s refresh DONE (offline) 2026-07-06

make check 311 passed; kustomize build OK. Single snapshot_box preserved (NO fast/slow split).
- config.py: metrics_refresh_interval_s (env TRE_METRICS_REFRESH_INTERVAL_SECONDS, default 5.0)
  decoupled from monitor_interval_s; metrics_window_ms default 60000 -> 30000 (N5: 3x the 10s
  write period, integer multiple, covers 3 histogram points; doc N5 typo "25000" ignored per its
  own integer-multiple rule).
- loops/metrics_task.py: metrics_task sleeps on metrics_refresh_interval_s (getattr fallback to
  monitor_interval_s); injectable sleep for testing. Still ONE snapshot_box (rescue+fairness share).
- deploy/overlays/tre-v2/controller.yaml: TRE_METRICS_WINDOW_MODE=sliding,
  TRE_METRICS_REFRESH_INTERVAL_SECONDS=5, TRE_METRICS_WINDOW_MS 60000->30000 (STARTING value;
  freeze after real-machine acceptance). kustomize build verified, envs present.
- Tests: metrics_task interval test (asyncio.run + fake sleep -> [5.0], not 20.0); config defaults.

Pending in S1.2: (a) N1 min-sample p95 guard (next commit); (b) REAL-MACHINE acceptance to
FREEZE W (roll controller with sliding/30s/5s, lag P95<=35s, Z_m jitter stddev) — deferred to a
live session; HARD prerequisite for S1.4/R3. Controller PAUSED.

### Signal Plan N1 — min-sample guard for p95 latency DONE (offline) 2026-07-06

make check 313; kustomize OK. Short windows + low QPS can leave single-digit latency
samples -> a p95 estimate is just noise. Guard nulls it.
- metrics_store.py: MetricsStore(min_latency_samples: int = 0). In _hist_percentile, if
  the window count delta (last.count - first.count) < min_latency_samples -> return None
  (per-pod; _aggregate_models _max_optional ignores None, so pods with enough samples

### Signal Plan N1 — min-sample guard for p95 latency DONE (offline) 2026-07-06

make check 313; kustomize OK. Short windows + low QPS can leave single-digit latency
samples so a p95 estimate is just noise. The guard nulls it.
- metrics_store.py: MetricsStore gains min_latency_samples (default 0). In _hist_percentile,
  when the window count delta (last.count minus first.count) is below min_latency_samples,
  return None (per-pod; _aggregate_model's _max_optional ignores None, so pods with enough
  samples still contribute). Default 0 = OFF so existing store tests stay green; live enables it.
- config.py: min_latency_samples (env TRE_MIN_LATENCY_SAMPLES, default 10); app.py wires it
  into MetricsStore. controller.yaml overlay sets 10.
- Tests: sparse window (3 samples) -> ttft_p95 None with guard, value with guard off; window
  with 13 samples -> value with guard on. config default 10.

Threshold 10 rejects single-digit counts (doc N1 framing); revisit at S1.2 real-machine acceptance.

### Signal Plan N2 — safescale window vs shared metrics window RESOLVED (architect Fable5) 2026-07-06

Flagged 记Blocked问架构师. Investigated + consulted architect (background). Ruling: the two
windows are orthogonal and compatible; NO SafeScale logic change; add ONE startup guard +
document. make check 314.

Analysis (architect-VERIFIED, 3 refinements): metrics_window_ms (30s sliding) = per-observation
aggregation history; SafeScaleConfig.default_window_ms (60s) = wall-clock probe deadline
(deadline = hide_start + 60s). Commit gate _summarize_tail inspects only the tail (hq=0.25) of
observations; those tail windows are fully post-hide since 60*(1-0.25)=45s >= 30s. Per-observation
_violates_slo on early blended windows + count-based tail are conservative-only (latency AND /
z_m min) -> extra rollbacks, never false commit. Shortening the metrics window strictly SPEEDS UP
rollback detection.

Change (config guard, TDD): ControllerConfig.from_env now enforces
default_window_ms*(1-hq) >= metrics_window_ms (hq<1: tail=hq*default_window_ms; hq>=1:
tail=hq*probe_poll_seconds*1000). Rejects e.g. SAFE_SCALE_DEFAULT_WINDOW_MS=15000 with 30s
metrics window (would silently dilute the gate with pre-hide traffic). RED/GREEN: 15000 & 39999
reject; 40000 (exact boundary 30000) + defaults load.

Recorded in 05_paper_vs_impl.md (SafeScale section): the two-window model, the hard invariant
(+ refined +refresh form), the direction-of-error safety note, and the DEAD-CONFIG landmine:
SafeScaleConfig.min_window_ms=15000 / max_window_ms are UNUSED (placeholder for the paper's
adaptive probe window; only a min<=max parse check). Not guarded now (would reject valid
defaults); whoever wires the adaptive window must floor it at metrics_window_ms/(1-hq)=40s.

N3/N4 remain DEFERRED to post-real-machine-acceptance (plan §7 step5, as-needed).

### Signal Plan S1.2 real-machine — S1 pipeline VALIDATED live; W provisional 30s, authoritative freeze -> F4.3 clean (2026-07-06)

Built controller image tre-v2-controller:20260706-446ce73a (post-S1 HEAD 446ce73a; committed +
overlay/pin/images.lock updated, resolves the F4.3 rebuild debt). Surgical controller-only roll on
the CURRENT cluster (apply overlay controller.yaml, replicas 0->1; no other resources touched), then
re-paused.

Live result (evidence: docs/refactor/p11_evidence/s1_shortwindow_20260706/):
- Controller healthy; decisions every ~5.7s with continuously-advancing window_end_ms (34 samples,
  delta median 5718ms) => sliding window + 5s refresh CONFIRMED live (was 60000ms tumbling). Satisfies
  S1.1 real-machine criterion (Z_m every refresh, no 60s step).
- Idle fleet -> z_m=null all models -> submitted=0 on all 53 sampled ticks (zero actions); GPU state
  identical before/after (fleet not mutated). Controller re-paused (replicas=0), safe state restored.

Decision (recorded ADR-0012): the AUTHORITATIVE lag-P95-under-step-load + FINAL W-freeze are folded into
the F4.3 clean-cluster redeploy immediately before R3, per D11 (authoritative numbers must run on the clean
cluster, not the hand-patched snowflake; and R3/S1.4 consume the frozen W there). Reasons: (a) a lag-P95
measurement needs a fleet-mutating load experiment, wasteful on a cluster F4 tears down; (b) the snowflake is
drifted from manifests (apply-k risk); (c) worst-case lag at W=30s is ~36s (=30000+~5718+write), right at the
35s target -> the clean-cluster P95 decides whether to keep 30000 or trim to 25000 (doc15 N5). W PROVISIONAL
= 30000. S1.4/R3 hard-gate INTACT: R3 blocked until W frozen on clean cluster.

S1 STATUS: all offline code done (S1.3/S1.1/S1.2/N1/N5/N2, make check 314, ADR-0011/0012). Pipeline validated
live. Remaining: authoritative W-freeze (F4.3) -> S1.4/R3 (+S2/S3) -> then S4 before R3. N3/N4 deferred.

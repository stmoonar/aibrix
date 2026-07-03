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

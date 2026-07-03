# P2 Upstream Patch Plan

Date: 2026-07-04
Environment: remote server 76, `/data/nfs_shared_data/xxy/aibrix`

## Target-Version Inspection

### Gateway queue wake-up

Old TRE source:
- `/root/aibrix-main/pkg/plugins/gateway/algorithms/queue_router.go`
- `/root/aibrix-main/pkg/plugins/gateway/algorithms/wakeup.go`

New target files inspected:
- `pkg/plugins/gateway/algorithms/queue_router.go`
- `pkg/plugins/gateway/algorithms/slo.go`
- `pkg/plugins/gateway/gateway_req_body.go`
- `pkg/plugins/gateway/gateway.go`

Finding: the new gateway rejects a request in `validateModelAvailability()` when a model has no routable pods before `selectTargetPod()` reaches the queue router. A literal queue-router-only port would not wake sleeping warm-pool pods on the zero-routable path. The migration therefore needs two compatible pieces:

1. Add the wake-up dispatcher and service-manager client in `pkg/plugins/gateway/algorithms/wakeup.go`, with `SERVEMENT_URL` required from env and no hard-coded fallback.
2. Wire the wake-up trigger where the new target can observe zero routable pods: first in `validateModelAvailability()`, then queue-router retry can be considered for queued requests that are already inside the router.

### APA sleep-mode scaling

Old TRE source:
- `/root/aibrix-main/pkg/controller/podautoscaler/podautoscaler_controller.go`

New target files inspected:
- `pkg/controller/podautoscaler/podautoscaler_controller.go`
- `pkg/controller/podautoscaler/autoscaler.go`
- `pkg/controller/podautoscaler/workload_scale.go`

Status: migrated in `WorkloadScale` as `TRE-PATCH(P2-APA-001)`. APA sleep mode is enabled by default (`APA_SCALE_SLEEP_MODE != "0"`), requires `SERVICE_MANAGE_URL`, reads wake replicas from `/models_replicas`, and applies desired-replica deltas via `/scale_service`.

## Patch / Commit Map

| Patch | Commit | Scope | Verification |
| --- | --- | --- | --- |
| TRE-PATCH(P2-GW-001) | `eeed0601` | Gateway wake-up dispatcher and env-validated service-manager client. | RED: missing `callWakeUpService`; GREEN: `GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/plugins/gateway/... -count=1` |
| TRE-PATCH(P2-GW-002) | `[P2] gateway: trigger wake-up for zero routable pods` | New-target zero-routable hook in `validateModelAvailability()`. | RED: no wake-up request observed; GREEN: `GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/plugins/gateway/... -count=1` |
| TRE-PATCH(P2-GW-003) | `[P2] gateway: dual-write TRE Redis metrics` | TRE Redis pod-metrics writer with `TRE_REDIS_SCHEMA=v1|v2|dual`, default dual. | RED: undefined writer/schema helpers; GREEN: `go test ./pkg/cache -count=1` and `go test ./pkg/plugins/gateway/... -count=1` |
| TRE-PATCH(P2-APA-001) | `[P2] podautoscaler: add APA sleep-mode service-manager adapter` | APA sleep-mode service-manager adapter in `WorkloadScale`, with pure-env `SERVICE_MANAGE_URL` startup validation. | RED: missing service-manager fields/config; GREEN: `go test ./pkg/controller/podautoscaler/... -count=1` |

## Notes

- `kustomize` binary is absent on server 76; use `kubectl kustomize` for manifest verification where needed.
- No local tests are authoritative for P2.


## Verification Log

### TRE-PATCH(P2-GW-001)

RED:

```bash
GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/plugins/gateway/algorithms -run TestCallWakeUpService -count=1
```

Result: build failed because `callWakeUpService` was undefined.

GREEN:

```bash
GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/plugins/gateway/algorithms -run TestCallWakeUpService -count=1
GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/plugins/gateway/algorithms -count=1
GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/plugins/gateway/... -count=1
```

Result: all passed on server 76. `/usr/local/go/bin/go` is required because `go` is not on the SSH PATH. `GOPROXY=https://goproxy.cn,direct` is required because `proxy.golang.org` timed out from the remote host.


### TRE-PATCH(P2-GW-002)

RED:

```bash
GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/plugins/gateway -run TestValidateModelAvailabilitySubmitsWakeupWhenNoRoutablePods -count=1
```

Result: failed after 2 seconds because no wake-up request was sent for a model with only non-routable pods.

GREEN:

```bash
GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/plugins/gateway -run TestValidateModelAvailabilitySubmitsWakeupWhenNoRoutablePods -count=1
GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/plugins/gateway/... -count=1
```

Result: all passed on server 76.


### TRE-PATCH(P2-GW-003)

RED:

```bash
GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/cache -run TestWriteTREPodMetricsToRedis -count=1
```

Result: build failed because `writeTREPodMetricsToRedis` and `treMetricSchemaMode` were undefined.

GREEN:

```bash
GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/cache -run TestWriteTREPodMetricsToRedis -count=1
GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/cache -count=1
GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/plugins/gateway/... -count=1
```

Result: all passed on server 76. Tests use `miniredis` and verify `ZRANGEBYSCORE` can read back `tre:v2:hist:{pod}` / `tre:v2:inst:{pod}` entries. The default mode writes both v1 legacy keys and v2 sorted sets; `TRE_REDIS_SCHEMA=v2` writes only v2.

### TRE-PATCH(P2-APA-001)

RED:

```bash
GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/controller/podautoscaler -run TestAPASleepMode -count=1
```

Result: build failed because `workloadScale` did not yet expose the APA service-manager configuration and `newServiceManageConfigFromEnv` was undefined.

GREEN:

```bash
GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/controller/podautoscaler -run TestAPASleepMode -count=1
GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/controller/podautoscaler -count=1
GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/controller/podautoscaler/... -count=1
```

Result: all passed on server 76. The adapter preserves the old sleep/wake control plane behavior on the new `WorkloadScale` seam: APA reads current wake replicas from service-manager and sends only the up/down delta to `/scale_service`, while non-APA and `APA_SCALE_SLEEP_MODE=0` continue to use Kubernetes resource scaling.

## P2 Combined Verification

```bash
GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/plugins/gateway/... ./pkg/controller/podautoscaler/... -count=1
GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/cache -count=1
GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go build ./...
```

Result: all passed on server 76 after `TRE-PATCH(P2-APA-001)`.

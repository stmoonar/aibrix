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

Status: not migrated yet. Keep for a separate P2 commit after gateway wake-up is isolated.

## Patch / Commit Map

| Patch | Commit | Scope | Verification |
| --- | --- | --- | --- |
| TRE-PATCH(P2-GW-001) | pending | Gateway wake-up dispatcher and env-validated service-manager client. | RED: missing `callWakeUpService`; GREEN: `GOPROXY=https://goproxy.cn,direct /usr/local/go/bin/go test ./pkg/plugins/gateway/... -count=1` |
| TRE-PATCH(P2-GW-002) | pending | New-target zero-routable hook in `validateModelAvailability()`. | Gateway request-body tests plus algorithms tests |
| TRE-PATCH(P2-APA-001) | pending | APA sleep-mode service-manager adapter. | Podautoscaler tests |

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

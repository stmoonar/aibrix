# N3 L3 Smoke Report

Date: 2026-07-04
Host: `nscc-ds-4a100-node10` / workspace `/data/nfs_shared_data/xxy/aibrix`
Status: **PARTIAL - do not tag `n3-done`**

## Summary

N3.1 backup and N3.2 old TRE removal are complete. The `tre-v2` control-plane is deployed and healthy, one `dsqwen-7b` model pod is running, gateway forwarding works for 100 requests, UI state works, and controller decisions are now non-stale after reading live gateway metrics from AIBrix Redis.

N3 is not complete because several required smoke checks failed or are only partially implemented:

- The model manifest requested logical GPU `0`, but Kubernetes/NVIDIA device allocation placed the process on physical GPU `2` (`nvidia-smi` on node9 shows GPU2 using ~37 GiB, GPU0 0 MiB).
- `PUT /v2/models/dsqwen-7b/target` is state-only. It records sleep/wake actions but does not call vLLM `/sleep` or `/wake_up`.
- Direct vLLM sleep/wake works, but sleep took `7.367s`, above the `<5s` N3 target; wake took `0.812s`.
- Gateway metrics are present only as legacy `aibrix:pod_*` keys in `aibrix-redis-master`, not as `tre:v2:hist:*` ZSET keys. The controller can read them through the v1 reader, but an uncached one-window read measured `138.169ms`, above the `<100ms` target.
- Controller logs did not contain `trs_calc_result`; evidence is currently from `tre:v2:decision:latest`, not structured controller log lines.

## Checks

| Check | Result | Evidence |
| --- | --- | --- |
| N3.1 old TRE backup committed | PASS | `docs/refactor/p11_evidence/old_system_backup/` contains old deployments, svc/cm/secret snapshot, pods, and nodes. Commit `e97a60e8`. |
| Old TRE deployments deleted | PASS | `tre-controller`, `service-management-xxy`, `service-management`, and `service-management-lxttest` no longer run in `aibrix-system`. AIBrix base pods remain running. |
| `tre-v2` overlay deployed | PASS | `tre-v2-controller`, `tre-v2-service-manager`, `tre-v2-ui`, and `tre-v2-redis` all `1/1 Running` on node10. |
| Model Service for gateway route | PASS after fix | Generated and applied `default/dsqwen-7b` Service. `dsqwen-7b-router` resolves refs after Service creation. |
| One `dsqwen-7b` model pod Running | PASS | Pod `dsqwen-7b-nscc-ds-4a100-node9-gpu-0-858d467d84-98mbp` is `1/1 Running` on node9. |
| GPU memory on expected physical GPU | FAIL | Expected manifest logical GPU `0`; node9 `nvidia-smi` shows physical GPU2 using ~37 GiB and GPU0 using 0 MiB. This is likely NVIDIA device-plugin remapping. |
| `/v2/reconcile` | PASS after fix | Reconcile dropped stale rollout pod binding and persisted current pod at version 4; later reconcile restored pod reality after state-only sleep. |
| `PUT /v2/models/dsqwen-7b/target` sleep/wake | PARTIAL | API returns quickly (`0.008s`, `0.003s`) but is state-only and does not call vLLM. Controller idle scale-down also only changed SM state; vLLM remained awake. |
| Direct vLLM sleep/wake | FAIL threshold | `/sleep` took `7.367s`, `/wake_up` took `0.812s`; N3 target is `<5s`. |
| Gateway 100 requests | PASS | `100/100` requests through `http://10.99.21.145/v1/completions` with header `model: dsqwen-7b`; p95 `28.33ms`, no errors. |
| Metrics written after gateway traffic | PARTIAL | Legacy keys exist in `aibrix-redis-master`: `aibrix:pod_histogram_metrics_default/dsqwen-7b...`; no `tre:v2:hist:*` keys in `tre-v2-redis` or AIBrix Redis. |
| Metrics window read | FAIL threshold | Controller v1 reader can read live metrics, but uncached read measured `138.169ms`, above `<100ms`. |
| Controller decision | PARTIAL | `tre:v2:decision:latest` hash has `stale=false`, `loop=rescue`, and submitted idle scale actions. Controller logs are empty; no `trs_calc_result` log evidence. |
| UI `/api/cluster` | PASS | `/healthz` returns `{"ok":true}`; `/api/cluster` shows topology and SM state version 8 with the current `dsqwen-7b` binding. |
| Rollback dry-run | PARTIAL | Server dry-run succeeded for `tre-controller.deploy.yaml` and `service-management-xxy.deploy.yaml`; remaining backup manifests still need dry-run before any real rollback. |
| Offline gate after N3 fixes | PASS | `cd tre && make check && make smoke`: `199 passed`, `tre smoke ok`. |

## Live Images

- `tre-v2-controller:20260704-a3d756b4`
- `tre-v2-service-manager:20260704-8af70fe4`
- `tre-v2-ui:20260704-669f0381`

## Fixes Made During N3 Smoke

- Preserved explicit namespaces in the `tre-v2` overlay so default-namespace RBAC is not rewritten into `tre-v2`.
- Fixed service-manager Kubernetes pod list/object handling and rollout replacement reconciliation.
- Fixed `dsqwen-7b` model path.
- Added generated per-model Services so AIBrix HTTPRoutes can resolve model backends.
- Split controller Redis configuration so state/decisions stay in `tre-v2-redis` while live metrics can be read from `aibrix-redis-master` using the legacy v1 reader.

## Required Before `n3-done`

1. Decide and implement deterministic physical GPU binding, or explicitly revise the acceptance criterion to treat logical CUDA device `0` inside the container as sufficient.
2. Wire service-manager target sleep/wake to real vLLM operations and reconcile pod annotations/state after those operations.
3. Bring direct vLLM sleep latency under 5s or revise the threshold with evidence.
4. Either deploy a v2 gateway metrics writer producing `tre:v2:hist:*` ZSETs, or optimize and formally accept the legacy v1 metrics path for N3.
5. Add/enable structured controller logs containing `trs_calc_result`, or update the checklist to use `tre:v2:decision:latest` as the authoritative decision evidence.
6. Complete rollback dry-run for all old-system backup manifests.

No `n3-done` tag was created.

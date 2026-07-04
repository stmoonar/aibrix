# N3 L3 Smoke Report

Date: 2026-07-04
Host: `nscc-ds-4a100-node10` / workspace `/data/nfs_shared_data/xxy/aibrix`
Status: **PASS - ready for `n3-done`**

## Summary

N3.1 backup, N3.2 old TRE removal, and N3.3 `tre-v2` deployment are complete. The live system has one `dsqwen-7b` model pod, the service-manager target path performs real vLLM sleep/wake operations, the gateway serves requests and writes v2 Redis metrics, controller decision logs include `trs_calc_result`, UI state reflects the live service-manager state, and restore-ready rollback manifests pass server-side dry-run.

GPU acceptance follows ADR-0005: TRE `gpu_ids` are logical scheduler slots. The current NVIDIA device plugin allocates a physical GPU UUID and exposes it inside the container as CUDA ordinal `0`; host physical GPU index equality is not enforceable with the generic `nvidia.com/gpu` resource.

## Checks

| Check | Result | Evidence |
| --- | --- | --- |
| N3.1 old TRE backup committed | PASS | Raw backups are under `docs/refactor/p11_evidence/old_system_backup/`; restore-ready sanitized copies are under `old_system_backup/restore_ready/`. |
| Old TRE deployments deleted | PASS | `tre-controller`, `service-management-xxy`, `service-management`, and `service-management-lxttest` are no longer running in `aibrix-system`; AIBrix base services remain. |
| `tre-v2` overlay deployed | PASS | Controller image `tre-v2-controller:20260704-51e6cde3`; service-manager image `tre-v2-service-manager:20260704-eaa117a4`; UI image `tre-v2-ui:20260704-669f0381`. |
| Model Service for gateway route | PASS | `default/dsqwen-7b` Service exists and the AIBrix HTTPRoute resolves refs. |
| One `dsqwen-7b` model pod Running | PASS | Pod `dsqwen-7b-nscc-ds-4a100-node9-gpu-0-858d467d84-98mbp` is `1/1 Running` on node9. |
| GPU binding contract | PASS | Pod annotation `tre.aibrix.io/gpu-ids=0`; container env `CUDA_VISIBLE_DEVICES=0`, `NVIDIA_VISIBLE_DEVICES=GPU-3a113474-dd92-6d52-d05b-491e7b020ded`; container `nvidia-smi` sees that UUID as index 0. Host node9 shows the same UUID at physical index 2, which is expected device-plugin remapping. |
| `/v2/reconcile` | PASS | State version 24 after cleanup had one binding and `POST /v2/reconcile` returned `warnings: []`. |
| `PUT /v2/models/dsqwen-7b/target` sleep/wake | PASS | With controller paused: target 1 idempotent `0.008s`; target 0 real sleep `1.116s`; target 1 real wake `0.793s`; final `/is_sleeping` false and pod annotation `state=awake`. A later sleep-only reproduction measured `1.078s` and produced `/is_sleeping: true`, annotation `state=sleeping`. |
| Runtime target growth guard | PASS | Live service-manager now rejects target growth beyond existing bindings when runtime ops are enabled, preventing state-only phantom bindings. Stale phantoms from earlier smoke were removed through `StateStore` version 23 -> 24; subsequent state stayed one binding. |
| Gateway 100 requests | PASS | With controller paused and model awake: `ok 100 errors 0`, latency min/avg/p95/max `19.46/25.06/31.16/39.33 ms`. |
| Gateway v2 metrics written | PASS | AIBrix Redis has `tre:v2:pods:dsqwen-7b`, `tre:v2:hist:default/dsqwen-7b...`, and `tre:v2:inst:default/dsqwen-7b...`; after the gateway burst, hist/inst ZCARD were `141/141`. |
| Metrics window read | PASS | v2 `MetricsStore.read_snapshot` / model window read against AIBrix Redis measured below target; model window read examples were `2.800 ms`, `0.791 ms`, `0.744 ms`, `0.726 ms`, `0.733 ms`. |
| Controller decision logs | PASS | Live controller logs contain JSON `trs_calc_result` entries with `stale:false` for rescue/fairness loops. |
| Control-loop tick timing | PASS | Direct live tick-path benchmark over metrics snapshot read + SM state fetch + cluster-view construction + rescue planning, 30 iterations: min/avg/p95/max `1.121/1.503/1.707/8.877 ms`. |
| UI `/api/cluster` | PASS | UI `/healthz` returns `{"ok":true}`; `/api/cluster` shows topology and service-manager version 29 with one bound `dsqwen-7b` pod. |
| Rollback dry-run | PASS | `kubectl apply --dry-run=server -f docs/refactor/p11_evidence/old_system_backup/restore_ready/*.yaml` passed for gateway, svc/cm/secret, and all old TRE deployments. Warnings were only missing last-applied annotations. |
| Offline gate | PASS | `git diff --check && cd tre && make check && make smoke`: `205 passed`, `tre smoke ok`. |

## Live Images

- `tre-v2-controller:20260704-51e6cde3`
- `tre-v2-service-manager:20260704-eaa117a4`
- `tre-v2-ui:20260704-669f0381`
- `aibrix/gateway-plugins:20260704-0d869b49-nozmq2` with `TRE_REDIS_SCHEMA=dual`

## Notes

- The gateway v2 metrics rollout is captured in `docs/refactor/p11_evidence/gateway-v2-metrics.deploy.yaml`.
- Raw rollback captures are retained for audit. Restore commands should use the sanitized `restore_ready/` manifests to avoid Kubernetes `resourceVersion` conflicts.
- During idle operation the controller may sleep the model. Gateway throughput smoke was therefore run with the controller paused, then the controller was restored.

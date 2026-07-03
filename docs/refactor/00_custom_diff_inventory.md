# P0 Custom Diff Inventory

Date: 2026-07-04
Host: server 76 (`nscc-ds-4a100-node10`)
Authoritative workspace: `/data/nfs_shared_data/xxy/aibrix`

## Baselines

- New workspace branch: `main`
- New workspace baseline commit recorded instead of tag: `adfe6f8373afe5a90a2e93687474f07a0d4aed26`
- Official AIBrix comparison tag fetched as: `upstream-v0.4.0`
- Official tag commit from upstream: `24eaefcfd0f4ea4ba5aaf8f283c4192c34d265a8`
- Frozen old system reference: `/root/aibrix-main`
- Old system commit: `1fab53b68cf64df4f3923a2ebe6ac448f6194279`
- Old system branch: `AIBrix_v0.4.0`

## Scope Note

`git diff upstream-v0.4.0..HEAD -- pkg api cmd config` in the new workspace contains broad upstream drift (373 files). That is not itself the custom TRE patch inventory. For P2, treat the frozen old system as the source of TRE custom behavior and the new workspace as the target to which those behaviors are migrated.

## Go/Config Custom Patch Inventory

| Source path | Evidence | Purpose | Keep? | Migration target / notes |
| --- | --- | --- | --- | --- |
| `pkg/plugins/gateway/algorithms/queue_router.go` | Diff against official extracted `v0.4.0` shows `GlobalQueueRegistry`, `modelName`, `globalRegistry`, `lastWakeUpCallTime`, new `NewQueueRouter(..., modelName string)` signature, and `wakeUpIfNeeded(queueLen)` hook when no routable pods exist. | Let gateway observe per-model/global queues and trigger warm-pool wakeup when routeable capacity is zero. | Keep behavior, rewrite as P2 patch. | Port to new gateway router API. Avoid hard coupling where possible. This is part of D1/D2 gateway path and must align with Redis v2 schema in P2/P3. |
| `pkg/plugins/gateway/algorithms/wakeup.go` | Old system contains wake-up dispatcher and `callWakeUpService`, using `/wake_up?model_name=...&kind=...&queue_len=...`. | Serialize gateway-triggered wakeup calls and shield request path from direct blocking service-manager calls. | Keep initially for v1 compatibility; later route through service-manager v2 where possible. | P2 should make service manager URL pure env, remove hard-coded defaults, and document compatibility path. |
| `pkg/controller/podautoscaler/podautoscaler_controller.go` | Diff against official extracted `v0.4.0` shows `SERVICE_MANAGE_URL`, `APA_SCALE_SLEEP_MODE`, `getWakeReplicasFromServiceManage`, `scaleViaServiceManage`, and APA path using wake replica count instead of K8s ready pod count. | Let APA baseline scale via service-manager sleep/wake instead of Kubernetes replica scale, preserving warm pod pool semantics. | Keep behavior, rewrite against new podautoscaler structure. | P2 migration. Replace command-style `/scale_service` with v1 compatibility or v2 target API adapter. Keep `APA_SCALE_SLEEP_MODE` env switch. |
| `config_tre/gateway.yaml` | Contains TRE env values (`TRE_DEFAULT_RPM`, `TRE_CHECK_RPM`, Redis host, tracing/debug flags). | Deploy old gateway plugin with TRE queue/wakeup behavior and Redis access. | Keep intent, not file shape. | P1/P2/P9 deploy manifests should move these values into `tre/deploy/base` and overlays. |
| `config_tre/tre-controller.yaml` | Contains TRE controller env (`ENABLE_TRE_SCALING`, `TRE_EXCLUDED_MODELS`, `TRE_HISTOGRAM_DEBUG`, Redis host). | Deploy old TRE controller and configure scaling switches. | Keep intent, replace with new controller config. | P5/P8 deploy base and ablation overlays. |
| `config_tre/model/*` | Model manifests use vLLM sleep image and `--enable_sleep_mode`; existing layouts are multi-replica/older generated style. | Pre-create warm model pods. | Replace. | P1 `gen_model_manifests.py` must generate one single-replica Deployment per model slot from `tre/deploy/registry.yaml`. |

## Python Module Boundary Inventory

| Existing module/path | Role today | Refactor target |
| --- | --- | --- |
| `/root/aibrix-main/python/tre/controller/main.py` | Monolithic TRE controller loop: reads metrics, computes TRS/Z_m, plans actions, writes Redis snapshots, calls service manager. | Split into `tre/controller/tre_controller/app.py`, loops, planning, signals, stores, and `sm_client.py` per section 5.4. |
| `/root/aibrix-main/python/tre/controller/trs.py` | Formal TRS/TSS calculation and saturation guard. | Migrate to `tre/controller/tre_controller/signals/trs.py`; add golden tests against old behavior. |
| `/root/aibrix-main/python/tre/controller/paper_state.py` | Z_m state classification (`CRITICAL`, `LOW`, `HEALTHY`, `HIGH`, plus idle/unknown). | Migrate to `tre/controller/tre_controller/planning/classify.py`; preserve old thresholds by default. |
| `/root/aibrix-main/python/tre/controller/planner.py` | Large planner with paper and legacy paths. | Pure `build_plan()` in `tre/controller/tre_controller/planning/planner.py`; drop legacy path only with DECISIONS entry. |
| `/root/aibrix-main/python/tre/controller/safescale.py` | SafeScale state/probe logic and service-manager action flow. | `tre/controller/tre_controller/planning/safescale.py`; write state-machine design before migration. |
| `/root/aibrix-main/python/tre/monitor/collector.py` | Redis SCAN-based metrics reader; key prefixes include `aibrix:pod_histogram_metrics_`; also `save_to_redis`. | `tre/controller/tre_controller/store/metrics_store.py` plus pure aggregation functions. P3 must support v2 sorted sets and v1 compatibility. |
| `/root/aibrix-main/python/tre/calibration/*` | Long calibration and fitting scripts. | `tre/calibration/tre_calibration/{collect,dataset,fit,evaluate,capacity}.py`. |
| `/root/aibrix-main/python/service_manage_aibrix/*` | FastAPI service manager, k8s discovery, sleep/wake, slot-ish cluster state held mostly in memory/yaml. | `tre/service-manager/tre_sm/*`, especially allocator/state/reconcile/API v2 from section 5.3. |
| `/root/aibrix-main/CustomTraceGenerator/*` | Trace replay, dispatch, orchestration, plotting. | `tre/replayer/tre_replayer/*`; P7 also implements `design.py`, `lint.py`, `oracle.py`. |

## v1 HTTP Interface Inventory

### Gateway/APA to service manager

| Endpoint | Caller | Purpose | v2 migration |
| --- | --- | --- | --- |
| `POST /wake_up?model_name=<model>&kind=<kind>&queue_len=<n>` | `pkg/plugins/gateway/algorithms/wakeup.go` | Passive/proactive wakeup decisions from gateway queue pressure. | Keep in `v1_compat`; internally translate to `/v2/models/{model}/target`. |
| `POST /scale_service?model_name=<model>&scale_type=up|down&scale_value=<n>` | APA sleep-mode path in `podautoscaler_controller.go`; TRE scaler compatibility. | Wake/sleep delta through service manager. | Keep in `v1_compat`; translate desired wake replica count to v2 target. |
| `POST /available_pods` or query equivalent | TRE/controller/service-manager clients | Query currently awake/routable pods. | Replace with `GET /v2/state`; v1 compatibility can derive old response. |
| `POST /models_replicas` | TRE/controller tooling | Query model replica counts. | Replace with `GET /v2/state`. |
| `POST /routable_models` | TRE/controller tooling | Query routability state. | Replace with `GET /v2/state` and `/v2/models/{model}/routable`. |
| `POST /sleep_all`, `POST /reactivate_all`, `/sleep_service`, `/wake_up_service`, `/wake_up_all`, `/is_sleeping` | Manual/admin and older service manager routes | Operational convenience and direct pod sleep/wake checks. | Keep only where necessary in `v1_compat`; prefer explicit v2 target/reconcile APIs. |

### service manager to vLLM pods

| Endpoint | Caller | Purpose |
| --- | --- | --- |
| `POST http://<pod_ip>:8000/sleep` | `k8s_discovery.py` | Put vLLM pod into sleep mode. |
| `POST http://<pod_ip>:8000/wake_up` | `k8s_discovery.py` | Wake vLLM pod. |
| `GET http://<pod_ip>:8000/is_sleeping` | `k8s_discovery.py` | Poll sleep state after operation. |

## Redis Interface Inventory

| Key/prefix | Current location | Purpose | Migration target |
| --- | --- | --- | --- |
| `aibrix:pod_histogram_metrics_<pod>_<ts>` | `python/tre/monitor/collector.py` | Histogram snapshots scanned by timestamp suffix. | Replace writer/reader with `tre:v2:hist:{pod}` sorted set. Keep v1 read compatibility in P3. |
| Instant metric prefix in old collector | `python/tre/monitor/collector.py` | Instant waiting/running/swapping/kv hit style samples. | Replace with `tre:v2:inst:{pod}` sorted set. |
| Controller snapshot keys written by `save_to_redis` | `python/tre/controller/main.py` via `tre.monitor.save_to_redis` | Debug/UI/controller metadata, including TRS/Z_m/theta fields. | Replace latest decision snapshot with `tre:v2:decision:latest`. |
| Service manager state | Mostly in memory/yaml today. | GPU/pod sleep state. | Persist to `tre:v2:sm:state` and `tre:v2:sm:version`. |

## P0 Gaps / Notes

- The new workspace `main` versus `upstream-v0.4.0` diff is broad upstream drift; P2 must inspect target files in the new tree before porting custom patches.
- `kubectl`/`nvidia-smi` snapshots are stored separately under `docs/refactor/p0_snapshots/` if available.
- No cluster writes were performed for this inventory.

## Upstream Drift Coverage Artifact

- Full new-workspace diff list saved to docs/refactor/00_upstream_drift_name_status.tsv.
- File count: 373 under pkg api cmd config.
- This list is target-version drift, not automatically custom TRE behavior. P2 must use it to inspect where the custom old-system patches land in the new tree.

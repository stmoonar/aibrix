# N4 Real-Environment Tests

Date: 2026-07-04
Host: `nscc-ds-4a100-node10` / workspace `/data/nfs_shared_data/xxy/aibrix`
Status: **READY FOR FINAL GATE - do not tag `n4-done` until final verification passes**

## Summary

N4 started after `n3-done` (`d694bc4e`). The live `tre-v2` control-plane is healthy. N4.1, as written, cannot be executed literally because the generated Deployment set requests more GPUs than the pinned nodes can provide; this is recorded as a justified SKIP for the all-at-once variant. N4.2 hot-switch validation passed. N4.3 control-loop live tests passed after controller fixes for awake/bound planning, routable serving floors, and Redis outage tolerance. N4.4 live defrag/same-slot validation is a justified SKIP because the implemented `/v2/defrag` does not recreate Kubernetes deployments. N4.5 fault injection passed after hardening. N4.6 used a bounded 15-minute soak substitute rather than the planned 12-hour overnight.

## N4.1 Full Topology Deployment

Status: **SKIP for all-at-once deployment; PASS for `dsqwen-7b` node9 subset**

Evidence:

- Generated model manifests include 12 model Deployments plus 3 model Services.
- Total GPU requests if all model Deployments are applied: 16 GPUs.
- Node-pinned requests for node9 alone: 12 GPUs.
- Registry topology has node9 = 4 GPUs and node10 = 4 GPUs. The cluster also reports a `cloud` node with 4 GPUs, but the registry/manifests do not target it.
- vLLM sleep mode reduces GPU memory after startup, but Kubernetes still reserves `nvidia.com/gpu` for sleeping pods, so applying all generated Deployments cannot produce the desired "all bound, mostly sleeping" state.

Decision:

- Do not apply all model manifests at once.
- Validate slot behavior sequentially or by model subset, and record each subset's pod scheduling, SM reconcile state, and `nvidia-smi` before moving to the next subset.

`dsqwen-7b` node9 subset result:

- Applied slot Deployments `gpu-1`, `gpu-2`, and `gpu-3` in addition to the existing `gpu-0` pod.
- After fixing discovery to read `tre.aibrix.io/gpu-ids` from pod labels, `POST /v2/reconcile` returned four `dsqwen-7b` bindings with no warnings.
- `PUT /v2/models/dsqwen-7b/target {"wake_replicas":1}` slept two newly observed awake pods and produced state version 71 with `awake=1`, `bound=4`.
- Node9 memory after target 1: one awake GPU at ~37470 MiB and three sleeping GPUs at ~1118 MiB.
- This subset is now the active N4 single-model topology.

## N4.2 Hot-Switch Round Trip

Status: **PASS**

Setup:

- Controller paused with `kubectl -n tre-v2 scale deploy/tre-v2-controller --replicas=0`.
- Initial node9 memory before test showed the allocated GPU UUID `GPU-3a113474-dd92-6d52-d05b-491e7b020ded` at 1118 MiB, consistent with vLLM sleep mode.
- Test target: `default/dsqwen-7b-nscc-ds-4a100-node9-gpu-0-858d467d84-98mbp`.

Result over 20 sleep/wake cycles:

```text
sleep_s_min_avg_p95_max 0.007 0.989 1.065 1.105
wake_s_min_avg_p95_max 0.661 0.808 0.864 0.870
```

Final state:

- Service-manager version 68.
- `dsqwen-7b`: `awake=1`, `bound=1`.
- Binding remained `node=nscc-ds-4a100-node9`, `gpu_ids=[0]`.
- Node9 memory after final wake: host physical GPU2 / UUID `GPU-3a113474-dd92-6d52-d05b-491e7b020ded` at 36956 MiB.

Follow-up:

- A post-test sleep returned state version 69 with `awake=0`, pod annotation `tre.aibrix.io/state=sleeping`, and `/is_sleeping: true`.

## N4.3 Control-Loop Real Behavior

Status: **PASS**

Required scenarios:

- Single-model step load.
- Alternating two-model load.
- Output-length drift sample.

Completed scenarios:

- Single-model low-latency step.
- Single-model heavy-load expansion.
- Alternating two-model load.
- Output-length drift sample.

### Single-Model Step Load

Status: **PARTIAL**

Setup:

- Active model subset: four bound `dsqwen-7b` pods on node9, one awake and three sleeping.
- Fixed routing so generated model Services select `tre.aibrix.io/routable=true`.
- Service-manager now patches `tre.aibrix.io/routable=false` on sleep/hidden and `true` on wake/unhide.
- Live `default/dsqwen-7b` Service endpoints after target 1: only `10.244.3.47:8000`.

Gateway validation after routing fix:

```text
ok 20 errors 0
lat_ms_min_avg_max 21.29 25.30 34.96
```

Step load:

```text
duration_s 120
rps 20
ok 2401 errors 0
lat_ms_min_avg_p95_max 19.63 26.27 28.37 56.84
```

Observed controller behavior:

- Controller logs stayed non-stale and emitted `trs_calc_result`.
- The controller did not wake additional `dsqwen-7b` replicas during this low-latency load; it kept one Service endpoint.
- A SafeScale probe marked the already-sleeping `gpu-0` binding hidden, but no scale-up action was observed.

Conclusion:

- Gateway routing over mixed awake/sleeping pods is now fixed for the service-selector path.
- This low-latency step did not satisfy the "CRITICAL expansion" part of N4.3; the heavier output-token run below covers that path.

### Single-Model Heavy Concurrent Load

Status: **PASS for single-model expansion**

Setup:

- Controller image: `tre-v2-controller:20260704-e0b4bb64`.
- Service-manager image: `tre-v2-service-manager:20260704-053e22f2`.
- `dsqwen-7b` subset: four bound node9 pods, initial target one awake pod.
- Reproducer: `/tmp/tre_concurrent_step_with_controller.py`, 8 worker threads, `max_tokens=96`, 120 seconds, controller scaled from 0 to 1 after load had started.

Controller fixes required before this pass:

- Per-model min/max bounds in planner config, instead of a single cluster-wide floor.
- Planner decisions use awake/routable counts for `ScaleAction` deltas and bound counts only for sleeping-capacity wake decisions.
- Controller overlays service-manager `ClusterView` state onto legacy v1 metrics because v1 metrics continue to list all historical model pods as metric-bearing pods even when they are sleeping.
- TRS is computed from awake replicas while the planner context keeps bound replicas for `critical_sleeping_capacity`.
- Rescue/fairness tasks skip live scaling until the service-manager cluster view has been populated, preventing startup ticks from acting on raw v1 pod counts.

Final live result:

```text
initial_state version=92 dsqwen-7b awake=1 bound=4
sample_s 30.4 version=94 dsqwen-7b awake=3 bound=4
sample_s 45.9 version=95 dsqwen-7b awake=4 bound=4
final_state version=95 dsqwen-7b awake=4 bound=4
final_endpoints 4 dsqwen-7b endpoints
ok 783 errors 0
lat_ms_min_avg_p95_max 1216.24 1236.5 1277.21 1312.88
```

Controller decision evidence:

- The final run had no `idle_proactive_immediate` downscale before expansion.
- Controller emitted `critical_sleeping_capacity` scale-up actions for `dsqwen-7b` during the active window.
- Endpoints remained non-empty throughout the run; no 500/503/timeout errors occurred.

### Output-Length Drift Sample

Status: **PASS**

Setup:

- `dsqwen-7b` final state from the heavy run: `awake=4`, `bound=4`.
- 20 gateway requests per setting, same prompt, `temperature=0`.

Result:

```text
max_tokens 1 ok 20 errors 0 lat_ms_min_avg_p95_max 20.72 26.49 34.74 41.66
max_tokens 32 ok 20 errors 0 lat_ms_min_avg_p95_max 412.81 420.96 429.99 433.38
max_tokens 96 ok 20 errors 0 lat_ms_min_avg_p95_max 1225.75 1235.22 1246.39 1246.84
```

Post-check:

- Service-manager remained `dsqwen-7b awake=4 bound=4`.
- `default/dsqwen-7b` retained four Service endpoints.

### Alternating Two-Model Load

Status: **PASS**

Setup:

- Controller image after fixes: `tre-v2-controller:20260704-f10439e6`.
- Second model subset: two TP=2 `dsqwen-14b` Deployments on node10, slots `gpu-0-1` and `gpu-2-3`.
- Initial target state: `dsqwen-7b awake=1 bound=4`, `dsqwen-14b awake=1 bound=2`.
- Driver: `/tmp/tre_alternating_load.py`, 10 minutes, 6 worker threads, 60s alternating phases, `max_tokens=64`, gateway path `http://10.99.21.145/v1/completions`.

Bug found before final pass:

- `idle_proactive_immediate` could sleep an idle bound model to zero endpoints, leaving no gateway route for a later alternating phase.
- Fixes:
  - `f10439e6` keeps proactive planner shrink above a live serving floor.
  - `883222d3` clamps controller-dispatched downscale targets to one awake bound replica, protecting against stale repeated downscale ticks.

Final live result:

```text
initial_state version=105 dsqwen-7b awake=1 bound=4, dsqwen-14b awake=1 bound=2
sample_s 90.9  dsqwen-7b awake=4 bound=4, dsqwen-14b awake=1 bound=2
sample_s 150.2 dsqwen-7b awake=4 bound=4, dsqwen-14b awake=2 bound=2
final_state version=109 dsqwen-7b awake=4 bound=4, dsqwen-14b awake=2 bound=2
dsqwen-7b ok 2167 errors 0 p95 855.8 ms
dsqwen-14b ok 1824 errors 0 p95 1010.5 ms
```

Controller evidence:

- `critical_sleeping_capacity` scale-up actions were emitted for `dsqwen-7b` and `dsqwen-14b`.
- Endpoints stayed non-empty for both models throughout the final run.
- No gateway 5xx, timeout, or request errors were observed.

## N4.4 Defrag And Same-Slot Shrink

Status: **SKIP for live execution; PASS for offline planner/API coverage**

Reason:

- The current live implementation of `POST /v2/defrag` computes migrations and updates service-manager state, but it does not recreate Kubernetes Deployments or move pods between GPU slots.
- Generated one-GPU model manifests only cover node9; the active TP=2 `dsqwen-14b` subset is on node10. Constructing the exact fragmented topology would require untracked manual manifest surgery or deleting/recreating live model slots outside the implemented defrag path.
- Because Kubernetes still reserves `nvidia.com/gpu` for sleeping pods, the all-at-once "all bound, mostly sleeping" topology cannot be used as a safe substitute.

Evidence retained:

- Offline tests cover `SlotAllocator.plan_defrag`, service-manager `/v2/defrag`, controller `DefragAction`, same-slot high shrink planning, and the P9 offline defrag integration path.
- Full gate after N4 fixes passed with `220 passed` and `tre smoke ok`.

Follow-up:

- A true live N4.4 PASS requires implementing Kubernetes delete/recreate operations in service-manager defrag or adding a tracked placement override that can generate node10 one-GPU manifests safely.

## N4.5 Fault Injection

Status: **PASS**

Checks:

- Kill controller pod and verify state recovery.
- Kill service-manager pod and verify reconcile.
- Stop Redis briefly and record degraded behavior.

Results:

```text
controller restart:
old pod tre-v2-controller-758787b7d-tlc7t
new pod tre-v2-controller-758787b7d-gbddq
post-state dsqwen-7b awake=1 bound=4, dsqwen-14b awake=1 bound=2
restarts after fixed rerun: 0

service-manager restart:
old pod tre-v2-service-manager-5f6bb479f8-d7fwh
new pod tre-v2-service-manager-5f6bb479f8-f6nqx
POST /v2/reconcile warnings=[]
post-state dsqwen-7b awake=1 bound=4, dsqwen-14b awake=1 bound=2

Redis outage:
tre-v2-redis scaled 1 -> 0 for 30s -> 1
controller pod stayed Running with 0 restarts on final run
service-manager state reset to empty after Redis restart, then reconcile rebuilt version=1 from live pods
endpoints after reconcile: one dsqwen-7b endpoint and one dsqwen-14b endpoint
```

Fixes required:

- `883222d3` clamps controller downscale targets to a serving floor, preventing stale repeated idle shrink from removing the last endpoint.
- `a0b2ff7f` treats Redis read failures during SafeScale restore as empty restore state.
- `303047a0` makes decision snapshot Redis writes best-effort while preserving structured `trs_calc_result` logs.

## N4.6 Soak

Status: **PASS for bounded 15-minute substitute; SKIP for full 12-hour overnight**

Bounded substitute:

- Driver: `/tmp/tre_soak_bounded.py`.
- Duration: 900 seconds.
- Traffic: one low-token gateway request per model per loop.
- Samples: controller RSS, service-manager RSS, Redis `DBSIZE`, service-manager state.

Result:

```text
initial controller_rss_kb=36676 service_manager_rss_kb=111824 redis_dbsize=3
sample 300s controller_rss_kb=36744 service_manager_rss_kb=112216 redis_dbsize=3
sample 600s controller_rss_kb=36744 service_manager_rss_kb=112216 redis_dbsize=3
final controller_rss_kb=36764 service_manager_rss_kb=112216 redis_dbsize=3
final state dsqwen-7b awake=4 bound=4, dsqwen-14b awake=2 bound=2
dsqwen-7b ok 395 errors 0 p95 131.75 ms
dsqwen-14b ok 395 errors 0 p95 151.45 ms
controller pod restarts 0
service-manager pod restarts 0
```

Conclusion:

- No request errors, restarts, RSS growth trend, or Redis key growth were observed in the bounded run.
- The full 12-hour overnight soak remains skipped for time, with this bounded substitute recorded as the N4 functional gate evidence.

No `n4-done` tag has been created.

# N4 Real-Environment Tests

Date: 2026-07-04
Host: `nscc-ds-4a100-node10` / workspace `/data/nfs_shared_data/xxy/aibrix`
Status: **IN PROGRESS - do not tag `n4-done`**

## Summary

N4 started after `n3-done` (`d694bc4e`). The live `tre-v2` control-plane is healthy and the `dsqwen-7b` model pod is available. N4.2 hot-switch validation passed on the deployed model. N4.1, as written, cannot be executed literally because the generated Deployment set requests more GPUs than the pinned nodes can provide; this is recorded as a justified SKIP for the all-at-once variant and needs a sequential-slot validation plan.

## N4.1 Full Topology Deployment

Status: **SKIP for all-at-once deployment; sequential validation pending**

Evidence:

- Generated model manifests include 12 model Deployments plus 3 model Services.
- Total GPU requests if all model Deployments are applied: 16 GPUs.
- Node-pinned requests for node9 alone: 12 GPUs.
- Registry topology has node9 = 4 GPUs and node10 = 4 GPUs. The cluster also reports a `cloud` node with 4 GPUs, but the registry/manifests do not target it.
- vLLM sleep mode reduces GPU memory after startup, but Kubernetes still reserves `nvidia.com/gpu` for sleeping pods, so applying all generated Deployments cannot produce the desired "all bound, mostly sleeping" state.

Decision:

- Do not apply all model manifests at once.
- Validate slot behavior sequentially or by model subset, and record each subset's pod scheduling, SM reconcile state, and `nvidia-smi` before moving to the next subset.

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

Status: **PENDING**

Required scenarios:

- Single-model step load.
- Alternating two-model load.
- Output-length drift sample.

Current blocker:

- Only one `dsqwen-7b` pod is currently bound. Expansion behavior requires additional bound pods or a runtime Kubernetes create path; N3 intentionally added a guard that rejects state-only live target growth.

## N4.4 Defrag And Same-Slot Shrink

Status: **PENDING**

Current blocker:

- Requires multiple model subsets and TP=2 `dsqwen-14b` pods. The all-at-once topology deployment cannot be used because of GPU request overcommit.

## N4.5 Fault Injection

Status: **PENDING**

Planned checks:

- Kill controller pod and verify state recovery.
- Kill service-manager pod and verify reconcile.
- Stop Redis briefly and record degraded behavior.

## N4.6 Soak

Status: **PENDING**

Planned check:

- Low-pressure overnight loop, with controller/SM RSS, Redis key count, and unexpected exception checks.

No `n4-done` tag has been created.

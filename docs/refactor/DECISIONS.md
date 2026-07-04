# Refactor Decisions

## ADR-0001: Remote server 76 is authoritative

- Date: 2026-07-04
- Status: accepted

### Context

`REFACTOR_PLAN.md` states that all new code changes and tests happen in `/data/nfs_shared_data/xxy/aibrix` on server 76. The local Windows checkout is not authoritative.

### Decision

All implementation and verification must happen in `/data/nfs_shared_data/xxy/aibrix` on server 76. Local test/build output is ignored. Local work may only be used as a disposable draft source and must be re-applied/verified remotely.

### Consequences

P1 work drafted locally is not considered complete until created and verified on server 76. P0 inventory and subsequent docs in this directory are authoritative.

## ADR-0002: Record baseline commit instead of creating `baseline-v0` tag immediately

- Date: 2026-07-04
- Status: accepted

### Context

P0 allows either tagging `baseline-v0` or recording the baseline commit in WORKLOG. The remote workspace has an untracked `REFACTOR_PLAN.md` at the start of work.

### Decision

Record baseline commit `adfe6f8373afe5a90a2e93687474f07a0d4aed26` in P0 docs and WORKLOG. Defer creating phase tags until the first clean phase commit.

### Consequences

This avoids tagging a state before the refactor plan/documentation commit exists. The baseline is still recoverable by commit hash.

## ADR-0003: Same-slot shrink probes start in loop tick, not safescale_task

- Date: 2026-07-04
- Status: accepted

### Context

`docs/refactor/10_next_steps.md` N1.2 says the new same-slot HIGH shrink action is consumed by `safescale_task`. The current P5/P9 controller architecture already has a narrower boundary: planner actions are converted to SafeScale probes in `loops/tick.py` through `_apply_safescale()`, while `safescale_task.py` only observes active probes and emits commit/rollback actions.

### Decision

Keep the existing boundary. `ShrinkForSlotAction` is emitted by the pure planner and consumed by `loops/tick.py`, which starts the SafeScale probe with the concrete donor serve id and records the TP=2 beneficiary as a pending upscale. `safescale_task.py` remains the observer for active probes.

### Consequences

This avoids routing planner output into the observation task and preserves the existing SafeScale lifecycle. The behavior still satisfies N1.2's intent: same-slot shrink is SafeScale-gated, and beneficiary expansion is delayed until after donor shrink commits.

## ADR-0004: TRE v2 images use immutable date-plus-git tags

- Date: 2026-07-04
- Status: accepted

### Context

N2 introduces build artifacts for the controller, service-manager, and UI. The next deploy phases need image references that can be traced back to source without relying on a mutable `latest` tag.

### Decision

TRE v2 component images use `tre-v2-<component>:<yyyymmdd>-<git-short-sha>`. `latest` is not used in Dockerfiles, manifests, or overlays. When images are pushed to an external registry, the pushed digest should be recorded beside this tag in WORKLOG.

### Consequences

Local build artifacts and cluster deployments can be tied to a specific source commit. Documentation-only follow-up commits may record build evidence, but deployable image tags still point at the source commit used for the build.

## ADR-0005: TRE GPU ids are logical slots under Kubernetes device-plugin allocation

- Date: 2026-07-04
- Status: accepted

### Context

N3 live smoke showed a `dsqwen-7b` Deployment labeled as TRE GPU `0`, while host `nvidia-smi` on node9 showed memory on physical GPU `2`. The installed NVIDIA device plugin advertises generic `nvidia.com/gpu`, uses `DEVICE_ID_STRATEGY=uuid`, and injects `NVIDIA_VISIBLE_DEVICES=<allocated GPU UUID>`. Inside the container, the allocated device is exposed as local CUDA ordinal `0`.

The original generated manifests set `CUDA_VISIBLE_DEVICES` to the TRE slot id. That works for logical slot `0`, but a one-GPU pod labeled as slot `2` would receive only one plugin-allocated GPU while also setting `CUDA_VISIBLE_DEVICES=2`, which is not a valid container-local ordinal.

### Decision

TRE slot ids in labels and annotations (`tre.aibrix.io/gpu-ids`) are logical scheduler slots used by the allocator and service-manager state. Generated model manifests set `CUDA_VISIBLE_DEVICES` to container-local ordinals (`0` for one GPU, `0,1` for two GPUs), while preserving the logical slot ids in names, labels, and annotations.

Service-manager reconciliation prefers `tre.aibrix.io/gpu-ids` when present and falls back to `CUDA_VISIBLE_DEVICES` only for unannotated legacy pods.

### Consequences

N3 acceptance checks validate the logical TRE slot, pod annotation, plugin-injected `NVIDIA_VISIBLE_DEVICES` UUID, and container-local CUDA ordinal. Host physical GPU index equality is not an enforceable property with the current generic `nvidia.com/gpu` resource and NVIDIA device-plugin configuration. Deterministic host physical GPU placement would require a separate device-plugin/resource model or scheduler integration and is outside the N3 deployment contract.

## ADR-0006: Model pods bind GPUs through NVIDIA_VISIBLE_DEVICES UUIDs

- Date: 2026-07-05
- Status: accepted

### Context

N4 showed that Kubernetes `nvidia.com/gpu` requests prevent TRE's intended warm-pool multiplexing: sleeping pods still reserve GPU quota, so the cluster cannot bind multiple sleeping model pods to one physical GPU and wake at most one at a time. The old system supported pods with zero GPU requests and recovered GPU binding from `NVIDIA_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES`, which is the resource model TRE needs.

### Decision

Generated model Deployments no longer request or limit `nvidia.com/gpu`. They pin `nodeName`, set `NVIDIA_VISIBLE_DEVICES` to the selected GPU UUIDs from `tre/deploy/registry.yaml`, keep logical GPU ids in `tre.aibrix.io/gpu-ids`, and add `tre.aibrix.io/gpu-uuids` for audit.

The Kubernetes scheduler no longer owns GPU exclusivity for TRE model pods. `SlotAllocator` is the source of truth: multiple sleeping bindings may share a GPU, but a GPU may have at most one awake binding. Service-manager wake and unhide paths check this invariant and reject conflicts with HTTP 409. Reconcile detects externally-created double-awake conflicts and conservatively marks the later observed binding sleeping.

Manifest generation enforces a static bound budget of at most three generated Deployments per GPU, matching the N4 measured 40GB budget of one awake pod plus up to two sleeping pods.

### Consequences

The D7 canary in N4b.3 is mandatory before broad rollout: first prove that a no-GPU-request pod can see only the UUID named by `NVIDIA_VISIBLE_DEVICES` in the current gpu-operator/runtime environment. If it cannot, the fallback chain is `runtimeClassName: nvidia`, then privileged plus `/dev/nvidia*` hostPath. The canary conclusion must be recorded before full topology deployment.

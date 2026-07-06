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

## ADR-0007: F4 base teardown (D11) BLOCKED — aibrix-system is a shared multi-tenant base

- Status: **Resolved by ADR-0008** (re-scoped to isolated TRE data plane). Original: Blocked / needed architect decision. Date: 2026-07-06.
- Context: Endgame plan §5 (D11) directs F4.2 to uninstall the AIBrix application
  base in `aibrix-system` and F4.3 to reinstall a clean AIBrix 0.7.0, on the
  premise that "the whole cluster base is a TRE snowflake". Live inspection on
  2026-07-06 contradicts that premise:
  - `aibrix-system` hosts ONE shared AIBrix base (controller-manager,
    gateway-plugins, redis-master, autoscaling-controller, gpu-optimizer,
    kuberay-operator, metadata-service, orchestration-controller, visualizers;
    290 days old, restarted ~4d14h ago).
  - A SEPARATE tenant runs on the same base: `lxtaibrix-gateway-plugins` (Running),
    `lxt-aibrix-eg` gateway + `lxt-aibrix-reserved-router` (all ~4d15h).
  - `qwen-coder-router` and `qwen-instruct-router` (373d) parent to the SAME
    `aibrix-eg` gateway that the TRE model routers (dsqwen-7b/14b, dsllama-8b) use.
  - AIBrix CRDs (`model.aibrix.ai`, `autoscaling.aibrix.ai`,
    `orchestration.aibrix.ai`, `ray.io`) are cluster-scoped and shared by all tenants.
  - The TRE controller itself reads metrics from the shared
    `aibrix-redis-master.aibrix-system`.
- Decision: **Do NOT execute F4.2/F4.3 autonomously.** Uninstalling the shared base
  or reinstalling it as 0.7.0 would (a) break the lxt tenant, (b) break
  qwen-coder/qwen-instruct serving via the shared `aibrix-eg` gateway, (c) force a
  CRD/controller version change (0.4→0.7) on all co-tenants, (d) delete the shared
  redis other components depend on. This violates red line 2.2 ("禁止改动破坏机器环境
  /其它工作负载") and the plan's own F4.2 caveat ("删任何底座对象前逐类确认归属，
  不确定就记 Blocked 停手问架构师；宁可留残余也不误删底座").
- Consequences: F4.4 (authoritative N4b on a clean 0.7.0 base) and the paper-grade
  N5 numbers that the plan gates behind it are ON HOLD pending one of the options
  below, to be chosen by the architect/human:
  1. **Isolated base for TRE**: install a second AIBrix 0.7.0 base in a NEW namespace
     (e.g. `aibrix-tre`) with its own gateway/redis, leaving `aibrix-system`
     untouched; retarget tre-v2 controller/SM/routes at the new base. (Cleanest;
     no third-party impact. Requires confirming 0.7.0 supports side-by-side install
     and the GPU nodes can host both bases' pods.)
  2. **Coordinated maintenance window**: get explicit sign-off from the lxt /
     qwen-coder / qwen-instruct owners, snapshot+restore their resources, then do
     the in-place 0.7.0 upgrade of the shared base during an agreed downtime.
  3. **Accept the current base for N5**: run N5 on the existing shared base with a
     documented version caveat (base is mixed 0.4-era images, not clean 0.7.0),
     skip F4.2/F4.3, and note the reproducibility limitation in the paper. Numbers
     would not be on a pristine 0.7.0 base.
- Recommendation: Option 1 (isolated `aibrix-tre` base) if 0.7.0 supports it;
  else Option 3 with a clear caveat. Option 2 only with explicit co-tenant sign-off.
- What proceeds regardless (not blocked): F4.0 declarative package (done); live
  NON-destructive validation of that package on the current cluster (gpu-truth
  DaemonSet, ReferenceGrant, regenerated model routes — all additive/idempotent);
  N5 driver tooling (r3_grid.py, reset scripts, 13_experiments_log scaffold).

## ADR-0008: D11 re-scoped — minimal isolated TRE data plane (supersedes F4.2/F4.3 teardown)

- Status: **Accepted** (architect ruling, 2026-07-06). Resolves ADR-0007.
- Ruling: The D11 "snowflake teardown + AIBrix 0.7.0 reinstall" is **cancelled**.
  A full side-by-side 0.7.0 base is infeasible (cluster-scoped AIBrix CRDs would
  schema-upgrade under co-tenants; duplicate controller/webhooks fight over shared
  CRs) AND unnecessary — TRE consumes NO AIBrix CRs (models are plain Deployments
  in `default`; the only PodAutoscaler `dsqwen14b-apa` is a dead 49d leftover),
  keeps its own state on `tre-v2-redis`, and replaces the AIBrix autoscaler. TRE's
  only couplings to the shared base are (a) the `aibrix-eg` Gateway + shared
  `aibrix-gateway-plugins` ext-proc (routing + zm metrics) and (b)
  `TRE_METRICS_REDIS_URL=aibrix-redis-master`.
- Decision: stand up a **minimal isolated TRE data plane inside the existing
  `tre-v2` namespace**, mirroring the proven lxt-tenant pattern (own Gateway + own
  gateway-plugins + own Envoy policies; no CRDs/controller/webhooks/base-redis):
  1. Gateway `tre-aibrix-eg` (tre-v2, `gatewayClassName: aibrix-eg`,
     `allowedRoutes.from: Same`).
  2. `tre-gateway-plugins` Deployment (tre-v2) = pinned TRE gateway-plugins image
     (`aibrix/gateway-plugins:20260704-0d869b49-nozmq2`), env REDIS_HOST=tre-v2-redis,
     TRE_REDIS_SCHEMA=dual (+RPM/tracing env); cloned SA/RBAC (list pods/services in default).
  3. EnvoyExtensionPolicy `tre-gateway-plugins-extension-policy` -> tre-aibrix-eg,
     ext-proc backend tre-gateway-plugins; clone lxt's epp + circuit-breaker patch
     policies with `tre-` prefix.
  4. Parameterize `gen_model_manifests.py` gateway ns/name (defaults
     tre-v2/tre-aibrix-eg); regenerate model HTTPRoutes into tre-v2; ReferenceGrant
     in default allows `from: tre-v2`.
  5. Retarget SM/controller env: TRE_ROUTE_NAMESPACE=tre-v2,
     TRE_GATEWAY_NAME=tre-aibrix-eg, TRE_METRICS_REDIS_URL=redis://tre-v2-redis:6379/0.
  6. Skip metadata-service (models-endpoint route only; add later if a driver needs /v1/models).
- Migration (each phase reversible):
  - **Phase A** additive: deploy Gateway + gateway-plugins + policies + new routes;
    old path keeps serving; smoke 20/20 per model through the new envoy + zm keys in tre-v2-redis.
  - **Phase B** cutover: roll SM/controller env to new targets; verify sleep round-trip,
    reconcile warnings=[], route-guard on new routes; point N5 drivers at new gateway.
  - **Phase C** cleanup (after >=24h stable): delete ONLY the 3 TRE model HTTPRoutes
    in aibrix-system. Touch nothing else there; do NOT revert the shared
    gateway-plugins image (co-tenants now depend on its current behavior; documented here).
- Fallback: if tre-gateway-plugins hard-requires an un-clonable base component
  (no evidence — lxt runs standalone), fall back to ADR-0007 Option 2 (shared base
  + caveat). Option 3 (maintenance window) rejected.
- N5 impact: "clean base" claim re-scoped to TRE's own data path (model pods +
  tre-aibrix-eg envoy + tre-gateway-plugins@digest + tre-v2-redis + controller/SM),
  a STRONGER reproducibility + traffic-isolation claim. F4.4 authoritative N4b =
  rerun N4.2/N4.4/N4.6 + 12h soak on the new path after Phase B, then tag n4b-done.
  V_static runs through the identical data plane (only controller policy differs).
  R1 old-system stays prior-work/secondary; old env now survives (no teardown).

## ADR-0009: SM reconcile two-layer model + reorder R3 before F4.4 scale-cycle

- Status: **Accepted** (architect ruling, 2026-07-06). Prompted by N4.6 pre-flight.
- Root cause (verified in code): `reconcile.py` reads the `tre.aibrix.io/state`
  annotation (a write-through cache of the last value SM wrote) as "reality" and
  never consults the physical vLLM `/is_sleeping`. Two gaps: (A) stale-truth —
  a physically-awake-but-annotated-sleeping leaked pod is invisible; (B) no
  enforcement — reconcile computes desired bindings + store.save but never writes
  the `routable` label back to pods, and `_auto_sleep_awake_conflicts` flips only
  the store record. Under scaling churn the routable label desyncs from physical
  reality → empty Service endpoints → models 0/4.
- Decision (SM fix, TDD before any further live scaling):
  - Physical `/is_sleeping` (+ gpu_truth used_mib) = OBSERVED ground truth; store
    `binding.awake` = DESIRED; annotation = write-through cache only (never read as reality).
  - **Layer 1 (safety invariant, every reconcile, idempotent, patch-on-diff):**
    `routable = true  iff  physical /is_sleeping == false  AND  not hidden`.
    Reconcile probes each pod and re-asserts the routable label from physical reality.
  - **Layer 2 (convergence):** where observed physical awake-set != desired store
    awake-set, re-issue wake/sleep (subject to single-awake-per-GPU + capacity).
    Non-converging pods (/wake_up 500 or sleep-leak) stay non-routable, emit the
    leak warning, surface as D8 candidate — no infinite wake loop.
  - 6 TDD cases: (1) awake+routable=false -> true; (2) sleeping+routable=true -> false;
    (3) annotation vs /is_sleeping mismatch -> physical wins, both corrected;
    (4) leaked pod stays non-routable + D8 candidate, no loop; (5) idempotence -> 0 patches;
    (6) auto-slept binding's pod label driven false.
- Decision (reorder): F4.4-before-R3 assumed realistic theta; the pre-flight
  falsified it (inherited theta 738.67/738/534 -> idle Z_m~0.04 << tau_crit -> all
  models CRITICAL at baseline -> over-provision, never shrink). New order:
  1. Fix SM desync (Q2) via TDD [blocks all].
  2. Restore canonical fleet (4 bindings/model) declaratively.
  3. R3 refit -> real theta_m + capacity surface.
  4. N4.6 expand/shrink + scale-cycle soak on real theta -> tag n4b-done.
  - Soak split: **Endurance** (restarts/RSS/redis-growth/zero-5xx at steady state;
    theta-independent; may run early, NOT the authoritative gate) vs **Scale-cycle**
    (>=5 clean expand/shrink; theta-dependent; step 4; certifies n4b-done).
  - Rejected: hand-tuning baseline into a "healthy" zm band (indefensible in paper).
- Recreate leaked dsllama gpu-0: YES, but only AFTER the SM fix lands AND
  nvidia-smi confirms gpu-0 memory freed, via the declarative path (deploy_models.sh),
  never manual kubectl.
- Plan patch: §5.5/§6 note F4.4's scale-cycle gate now consumes R3 output.

## ADR-0010: Apply D8 gpu_memory_utilization=0.85 (co-resident cold-start headroom)

- Status: **Accepted** (2026-07-06). Authorized by plan D8, whose precondition
  ("only if E1 proves healthy-path cold-start headroom insufficient") is now met.
- Evidence: recreating a single model binding into a GPU already hosting co-resident
  sleeping pods OOMs at the default 0.9 util — vLLM sampler warmup needs >40 GiB
  (0.9*40 = 36 GiB target + ~4 GiB co-resident sleeping pods + warmup). The original
  12-binding topology only avoided this via creation ORDER (each pod warmed up at 0.9
  while its GPU was still empty, then slept to ~2 GiB before the next was added).
- Decision: add `--gpu-memory-utilization 0.85` to every model's `vllm_extra_args` in
  `deploy/registry.yaml`; regenerated all model manifests (make check 285). 0.85*40 =
  34 GiB + ~4 co-resident = 38 < 40, so a binding can be (re)created into a populated
  GPU without OOM.
- Consequences:
  1. **R3 capacity baseline is now 0.85** (less KV cache than 0.9). This is a
     system-parameter choice that propagates into all N5 capacity/theta numbers — the
     paper must report util=0.85. Kept consistent by regenerating all manifests.
  2. LIVE serving pods still run at 0.9 (their pods predate this change); they take
     0.85 only when recreated. **R3 setup must recreate the fleet at 0.85** for a
     consistent capacity measurement (R3 already prepares the fleet, so low marginal cost).
  3. Does NOT fix concurrent fresh bring-up: 3 pods/GPU loading at 0.85 = 102 GiB > 40.
     `deploy_models.sh` still needs a STAGGERED create->wait-ready->sleep path for
     fresh clusters (warned in-script; TODO before any F4.3-style fresh deploy).
- Reversible: drop the two args from registry.yaml + regenerate to return to 0.9.

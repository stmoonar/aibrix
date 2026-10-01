# Release 2026-10-01: APA stable window from the PodAutoscaler annotation - DRAFT

Plan only: the image is built and loaded on both nodes, nothing is applied. Confirm with
the owner before any step that touches the cluster (a parallel session may be using it).
This release replaces the image of `aibrix-system/aibrix-controller-manager`; the owner
approved that single change as an exception to ADR-0008 (no other `aibrix-system` object
is touched).

## What changes

Branch `feat/apa-window-20261001` from 7994da76, source commit 94a0d82c.

- The APA stable metric window was a constant 180 s
  (`pkg/controller/podautoscaler/metrics/client.go`); the CR annotation
  `apa.autoscaling.aibrix.ai/window` had no reader. It is now read per PodAutoscaler
  (positive duration; absent = 180 s, unchanged).
- The APA baseline CRs (`tre/deploy/baselines/apa/*-apa.yaml`): window `30s` -> `20s`;
  tolerance keys renamed to the names the controller reads. Effective values:

  | | before (live) | after |
  |---|---|---|
  | stable window | 180 s (annotation ignored) | 20 s |
  | scale-up tolerance | 0.1 (default; key ignored) | 0.1 |
  | scale-down tolerance | 0.1 (default; key ignored) | **0.2** |
  | max scale-up / down rate | 2 / 2 (default) | 2 / 2 (default) |
  | scale-up / down cooldown | 0 s / 300 s (default) | 0 s / 300 s (default) |
  | min / max replicas, target | 1 / 4, `kv_cache_usage_perc` 0.5 | unchanged |

- Every evaluation (every 10 s per PodAutoscaler and metric) logs
  `"Effective autoscaling config"` with `stableWindow`, `upTolerance`, `downTolerance`,
  `maxScaleUpRate`, `maxScaleDownRate`, `scaleUpCooldown`, `scaleDownCooldown`,
  `minReplicas`, `maxReplicas`. Unrecognized `autoscaling.aibrix.ai/`, `apa.` or `kpa.`
  annotation keys are logged as `Ignoring unrecognized autoscaling annotation`.
- The binary also carries the non-autoscaler changes between the live image's source
  (7d3535b1) and 7994da76: the vLLM metric-name fallback in `pkg/metrics`
  (`kv_cache_usage_perc` is still read as `vllm:kv_cache_usage_perc` first) and the
  routable-label gate in `pkg/utils` (off unless `TRE_ROUTABLE_LABEL_FILTER=true`, which
  this Deployment does not set). `pkg/controller` and `cmd/controllers` have no other change.

## Image

| | |
|---|---|
| tag | `aibrix/controller-manager:20261001-94a0d82c` |
| ID (76 and 75) | `sha256:91e8f06127181d0e893da14c07fa86bf15633ac65297779bc26497d5e64defed` |
| build | `tre/deploy/scripts/build_controller_manager.sh` from a clean clone at 94a0d82c (go1.22.12, CGO off, GOPROXY=off; same distroless base and entrypoint as `build/container/Dockerfile`); binary stamped `vcs.revision=94a0d82c..., vcs.modified=false` |
| live (rollback) | `aibrix/controller-manager:nightly` = `:7d3535b111e3...`, ID `4c658acb5caf` |

The Deployment is pinned to one node by its existing `nodeSelector`; the image is on both
nodes anyway.

## Tests

`go test ./pkg/controller/podautoscaler/... ./pkg/metrics/... ./pkg/utils/...` all ok.
New: annotation `20s` -> 20 s, absent -> 180 s, invalid / non-positive rejected; the
baseline annotation set takes effect by name; the old `*-fluctuation-tolerance` keys keep
the defaults; per-PA window sizes in `MetricsClient`; a 25 s old sample drops out of a 20 s
window but not of the default one; the pipeline sizes each PA's window from its own CR.

## Steps

Run on the control-plane node. `NS=aibrix-system`, `D=aibrix-controller-manager`,
`B=<backups dir>/apa-window-20261001`, `IMG=aibrix/controller-manager:20261001-94a0d82c`.

0. Preconditions: no experiment running; the cluster is on the TRE arm (no
   `podautoscalers.autoscaling.aibrix.ai` objects; `kubectl get podautoscalers -A` empty),
   so the swap cannot move any pod. `docker image inspect $IMG` on the node in the
   Deployment's `nodeSelector` returns the ID above.
1. Backup:
   ```bash
   mkdir -p $B
   kubectl -n $NS get deploy $D -o yaml > $B/deploy.yaml
   kubectl -n $NS get deploy $D -o jsonpath='{.spec.template.spec.containers[0].image}' > $B/image.txt
   kubectl -n $NS get deploy $D -o jsonpath='{.spec.template.spec.containers[0].env}{"\n"}{.spec.template.spec.containers[0].args}{"\n"}{.spec.template.spec.nodeSelector}' > $B/env-args-nodeselector.txt
   kubectl -n $NS logs deploy/$D --tail=2000 > $B/logs-before.txt
   cat > $B/rollback.sh <<EOF
   #!/usr/bin/env bash
   set -euo pipefail
   kubectl -n $NS set image deploy/$D manager=$(cat $B/image.txt)
   kubectl -n $NS rollout status deploy/$D --timeout=180s
   EOF
   chmod +x $B/rollback.sh
   ```
2. Swap only the image (env, args, nodeSelector, probes stay as they are):
   ```bash
   kubectl -n $NS set image deploy/$D manager=$IMG
   kubectl -n $NS rollout status deploy/$D --timeout=180s
   ```
3. Check nothing else changed:
   `kubectl -n $NS get deploy $D -o yaml > $B/deploy-after.yaml && diff $B/deploy.yaml $B/deploy-after.yaml`
   shows only the image, `generation`, `resourceVersion`, the revision annotation and status.
   The new pod becomes leader of lease `aibrix-controller-manager` (`kubectl -n $NS get lease aibrix-controller-manager`).
4. Verify the window is in force. It only shows while APA CRs exist, so do it inside the
   canary (step 5) or briefly apply the CRs on an idle cluster in `observe observe`:
   ```bash
   kubectl -n $NS logs deploy/$D --since=2m | grep "Effective autoscaling config"
   # expect per PA: stableWindow="20s" upTolerance=0.1 downTolerance=0.2 maxScaleUpRate=2
   #                scaleDownCooldown="5m0s" minReplicas=1 maxReplicas=4
   kubectl -n $NS logs deploy/$D --since=2m | grep "Ignoring unrecognized autoscaling annotation"
   # expect nothing
   ```
   The `Metrics window aggregation` lines list `stableWindowValues`: at most 3 samples
   (20 s at the 10 s evaluation period) instead of up to 19.
5. Canary (about 12 min): the toggle applies the CRs from `APA_DIR`. Until this branch is
   on `main`, point it at the branch's CRs, otherwise the old 30 s / ignored-key CRs are applied:
   ```bash
   export APA_DIR=<worktree>/tre/deploy/baselines/apa
   <verify dir>/scripts/B_apa_canary.sh            # plan
   <verify dir>/scripts/B_apa_canary.sh --execute
   ```
   Expect: `AbleToScale=True` x3; on the 8b KV-heavy load the first 1 -> 2 decision
   (`desiredScale>1`, then the service-manager wake) about 20-30 s after the load starts,
   down from about 65 s with the 180 s window; after the load the scale-down still waits
   for the 300 s cooldown and is now gated by the 0.2 down tolerance. Keep the canary's
   `aibrix-controller-manager.log` and `apa_crs_after.yaml` as evidence, and check that it
   shows `stableWindow="20s"`.
6. Rollback (any failure in 2-5): `$B/rollback.sh`; then confirm the pod runs the image
   in `$B/image.txt` and the lease is held. The CR files are plain manifests: re-apply the
   old ones from `main` if the window or tolerance keys need to go back.

## Open points for the owner (differences from v1, not changed here)

| | v1 (AIBrix 0.4 config) | v2 after this release |
|---|---|---|
| window | 20 s | 20 s |
| up / down tolerance | 0.2 / 0.8 | 0.1 / 0.2 |
| max scale-up rate | 2 | 2 (default) |
| min / max replicas | 1 / 4 | 1 / 4 |
| target | `gpu_cache_usage_perc` 0.5 | `kv_cache_usage_perc` 0.5 (renamed metric) |
| scale-down cooldown | none | 300 s (default; `autoscaling.aibrix.ai/scale-down-cooldown-window` would set it) |

Aligning the remaining rows means editing the CR annotations only
(`scale-up-tolerance`, `scale-down-tolerance`, `scale-down-cooldown-window: 0s`); no
image change is needed.

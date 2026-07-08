# APA (KVCache) baseline for experiment 3

The control arm of experiment 3 (TRE vs APA). APA is AIBrix's built-in Pod Autoscaling
Algorithm driving on the vLLM pod metric `gpu_cache_usage_perc`. These manifests wire APA to
the tre-v2 models through the same service-manager seam TRE uses, so the two arms move the
exact same pods and the comparison is apples-to-apples.

## Files

| File | What |
| --- | --- |
| `dsqwen-7b-apa.yaml`, `dsllama-8b-apa.yaml`, `dsqwen-14b-apa.yaml` | one `PodAutoscaler` (`scalingStrategy: APA`) per model |
| `dsqwen-7b-apa-anchor.yaml`, `dsllama-8b-apa-anchor.yaml`, `dsqwen-14b-apa-anchor.yaml` | 0-replica scale-anchor `Deployment` per model |

## How the seam works (evidence)

The aibrix podautoscaler controller is patched (`TRE-PATCH(P2-APA-001)`,
`pkg/controller/podautoscaler/workload_scale.go`) so that when `spec.scalingStrategy == APA`
and `APA_SCALE_SLEEP_MODE != 0`, scaling is applied through service-manager instead of k8s
`spec.replicas`:

- `shouldUseAPASleepMode` — `workload_scale.go:334` (`sleepModeEnabled && ScalingStrategy == APA`).
- current replicas read from service-manager `POST /models_replicas?models=<name>` —
  `workload_scale.go:179,357`.
- desired replicas applied via service-manager `POST /scale_service?model_name=<name>&scale_type=up|down&scale_value=<delta>` —
  `workload_scale.go:243,409`.

In every one of those calls the model name is **`pa.Spec.ScaleTargetRef.Name`**. That is why
each CR sets `scaleTargetRef.name` to the exact registry model name (`dsqwen-7b`,
`dsllama-8b`, `dsqwen-14b`) — service-manager keys models by that name, and min/max replicas
mirror `deploy/registry.yaml` (7b/8b `1..8`, 14b `0..4`).

### Why the anchor Deployment exists

Even in sleep mode the reconcile still resolves the scale target to read the pod label
selector for metric scraping (`getScaleResource` → `GetPodSelectorFromScale`,
`workload_scale.go:497`). tre-v2 model pods are per-GPU Deployments
(`dsqwen-7b-<node>-gpu-N`) with no aggregate Deployment, so each anchor is a **0-replica**
Deployment named after the model whose `spec.selector` is `model.aibrix.ai/name: <model>`.
That selector matches all awake pods of the model, so APA averages `gpu_cache_usage_perc`
across them. Sleep mode never writes the anchor's replicas; real scaling goes to
service-manager. Apply the anchor **before** the PodAutoscaler.

## Usage

Do not apply these by hand during an experiment — always go through the toggle so the mutual
exclusion is enforced:

```bash
# switch the cluster to the APA arm (stops TRE first, verifies, then applies these CRs)
tre/deploy/scripts/toggle_tre_apa.sh apa

# switch back to the TRE arm (deletes these CRs, verifies none remain, then enables TRE)
tre/deploy/scripts/toggle_tre_apa.sh tre

# report the active decision source
tre/deploy/scripts/toggle_tre_apa.sh status
```

If you must stage manually: `kubectl -n default apply -f dsqwen-7b-apa-anchor.yaml` then
`... -f dsqwen-7b-apa.yaml` (anchor first), and delete in the reverse order.

## Mutual exclusion (critical)

Both the TRE controller and the patched APA controller push scaling through
service-manager. Running both at once makes them fight over the same pods. Exactly one arm
may be live:

- **APA arm**: `ENABLE_TRE_SCALING=false` on `tre-v2-controller` **and** these PA CRs applied.
- **TRE arm**: PA CRs deleted **and** `ENABLE_TRE_SCALING=true`.

`toggle_tre_apa.sh` always stops the old source and verifies it is gone before starting the
new one.

## Leftover assumptions (verify on the live cluster after R3)

1. **`SERVICE_MANAGE_URL` / `APA_SCALE_SLEEP_MODE` on the aibrix-system podautoscaler
   controller** must point at the tre-v2 service-manager for these CRs to actuate the tre-v2
   pods. Confirm the aibrix-system controller env (MEMORY notes a kubectl-layer drift on
   `SERVICE_MANAGE_URL`). This dir does not touch aibrix-system (ADR-0008).
2. **Anchor selector overlap**: the 0-replica anchor RS shares the `model.aibrix.ai/name`
   label with real model pods. Real pods have their own controller owner refs, so the anchor
   cannot adopt them and (replicas 0) never deletes them, but confirm no selector-overlap
   surprises and that the anchor does not schedule a pause pod onto a GPU node.
3. **Metric availability**: confirm `gpu_cache_usage_perc` is exposed on `:8000/metrics` for
   the tre-v2 vLLM image and that `targetValue: 0.5` gives sane replica counts; tune if not.
4. The PA controller must be watching namespace `default` (where the models and these CRs
   live).

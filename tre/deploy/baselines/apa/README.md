# APA (KVCache) baseline for experiment 3

The control arm of experiment 3 (TRE vs APA). APA is AIBrix's built-in Pod Autoscaling
Algorithm driving on the vLLM pod metric `kv_cache_usage_perc` (named `gpu_cache_usage_perc`
before vLLM 0.11; the 0.30 fork exports only the new name). These manifests wire APA to
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
`dsllama-8b`, `dsqwen-14b`) — service-manager keys models by that name. minReplicas mirrors
`deploy/registry.yaml` `min_replicas`; maxReplicas mirrors the registry scaling cap
`max_awake_replicas` (4 for every model, same cap as TRE; v1/paper alignment A1), not the
GPU layout size `max_replicas` (7b/8b 8 bindings, 14b 4). Service-manager enforces the same
cap on `/scale_service`, so every model is `1..4` (14b raised from 0 to 1 on 2026-09-24, as in v1).

### Why the anchor Deployment exists

Even in sleep mode the reconcile still resolves the scale target to read the pod label
selector for metric scraping (`getScaleResource` → `GetPodSelectorFromScale`,
`workload_scale.go:497`). tre-v2 model pods are per-GPU Deployments
(`dsqwen-7b-<node>-gpu-N`) with no aggregate Deployment, so each anchor is a **0-replica**
Deployment named after the model whose `spec.selector` is `model.aibrix.ai/name: <model>`.
That selector matches all awake pods of the model, so APA averages `kv_cache_usage_perc`
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

## Annotations (keys the controller actually reads)

The v2 controller only applies annotation keys it has a parser for
(`pkg/controller/podautoscaler/context/context.go` `annotationParsers`); any other
`autoscaling.aibrix.ai/`, `apa.autoscaling.aibrix.ai/` or `kpa.autoscaling.aibrix.ai/` key is
ignored and logged as `Ignoring unrecognized autoscaling annotation`. The values follow the
v1 APA configuration, `/root/aibrix-main/config_tre/autoscaler_hot/<model>_APA.yaml` (same values in `config_tre/autoscaler/`):

| Key | Value | v1 annotation (value) | Notes |
| --- | --- | --- | --- |
| `autoscaling.aibrix.ai/scale-up-tolerance` | `0.2` | `autoscaling.aibrix.ai/up-fluctuation-tolerance` (`0.2`) | scale up when usage > target x 1.2 |
| `autoscaling.aibrix.ai/scale-down-tolerance` | `0.8` | `autoscaling.aibrix.ai/down-fluctuation-tolerance` (`0.8`) | scale down when usage < target x 0.2 |
| `autoscaling.aibrix.ai/scale-down-cooldown-window` | `0s` | none (v1 APA has no cooldown) | v2 default is 300 s |
| `apa.autoscaling.aibrix.ai/window` | `20s` | `apa.autoscaling.aibrix.ai/window` (`20s`) | stable metric window, per PodAutoscaler; APA only, at least 1 s; default 180 s |

Not set, so the controller defaults apply, equal to v1: `autoscaling.aibrix.ai/max-scale-up-rate` 2,
`max-scale-down-rate` 2 (v1 files: `apa.autoscaling.aibrix.ai/max-scale-{up,down}-rate: '2.0'`),
`scale-up-cooldown-window` 0 s. `minReplicas` 1 / `maxReplicas` 4 and `targetValue` 0.5 are
as in the v1 `autoscaler_hot` files (v1 metric name `gpu_cache_usage_perc`, renamed
`kv_cache_usage_perc` in vLLM 0.11+).

Note on the v1 tolerance keys: the v1 controller (`pkg/controller/podautoscaler/scaler/apa.go`
in `/root/aibrix-main`) reads tolerances under the `apa.autoscaling.aibrix.ai/` prefix, so the
v1 files' `autoscaling.aibrix.ai/{up,down}-fluctuation-tolerance` keys were not parsed and the
v1 run used its defaults, up 0.1 / down 0.2. The CRs here follow the values written in the v1
files (owner decision 2026-10-01).
The window keeps one sample per second bucket (a later sample in the same second overwrites)
and averages the buckets of the last 20 s with equal weight. The sample count is not fixed:
besides the 10 s resync (`DefaultResyncInterval`), the controller watches PodAutoscaler
objects without an event filter, so its own status write-back and every scaling step trigger
an immediate re-evaluation that adds a sample. Under load this chains decisions: one canary
went 1 -> 2 -> 3 -> 4 within 5.5 s (`max-scale-up-rate` caps each step, not the chain;
scale-up cooldown is 0 s). This is the stock controller behaviour and is kept as the baseline.
Each evaluation logs the values in force, and the window itself:
`kubectl -n aibrix-system logs deploy/aibrix-controller-manager | grep -E "Effective autoscaling config|Metrics window aggregation"`
(`stableWindow`, `stableWindowSpan` = time from the oldest to the newest sample, at most the window).

## Leftover assumptions (verify on the live cluster after R3)

1. **`SERVICE_MANAGE_URL` / `APA_SCALE_SLEEP_MODE` on the aibrix-system podautoscaler
   controller** must point at the tre-v2 service-manager for these CRs to actuate the tre-v2
   pods. Confirm the aibrix-system controller env (MEMORY notes a kubectl-layer drift on
   `SERVICE_MANAGE_URL`). This dir does not touch aibrix-system (ADR-0008).
2. **Anchor selector overlap**: the 0-replica anchor RS shares the `model.aibrix.ai/name`
   label with real model pods. Real pods have their own controller owner refs, so the anchor
   cannot adopt them and (replicas 0) never deletes them, but confirm no selector-overlap
   surprises and that the anchor does not schedule a pause pod onto a GPU node.
3. **Metric availability**: confirm `kv_cache_usage_perc` is exposed on `:8000/metrics` for
   the tre-v2 vLLM image (vLLM 0.10.1 and the 0.30 fork both export it; 0.30 no longer
   exports `gpu_cache_usage_perc`, and the AIBrix APA fetcher returns 0 for a missing
   metric, i.e. never scales - `tre/docs/design/20260928-vllm-030-metric-names.md`) and that `targetValue: 0.5` gives sane replica counts; tune if not.
4. The PA controller must be watching namespace `default` (where the models and these CRs
   live).

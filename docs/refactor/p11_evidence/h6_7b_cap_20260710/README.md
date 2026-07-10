# H6 dsqwen-7b replica-cap diagnosis (2026-07-10)

## Finding

The t1/t7 plateau at four awake 7b replicas is service-manager-side, but it is
not a wake-feasibility or GPU-capacity failure. The deployed service-manager
registry capped `dsqwen-7b` at four replicas.

The controller did request the next scale step. During the post-theta t1 run,
three successive 7b target requests returned 200 and the next request returned
400 at `2026-07-09T17:20:57Z`; see
`historical_t1_target_requests.txt`. The committed timeline simultaneously
records another `delta=+1` action while the aggregate state remains at four.

## Live reproduction

The cluster was in controller `observe` mode and the normal 1/1/1 baseline.
Before the request, service-manager reported 7b as `awake=1, bound=8`. The
following request was issued against the deployed service-manager image:

```bash
curl -X PUT -H 'Content-Type: application/json' \
  --data '{"wake_replicas":5}' \
  http://tre-v2-service-manager:8000/v2/models/dsqwen-7b/target
```

It returned HTTP 400 with the committed response in
`live_target5_response.json`:

```json
{"detail":"wake_replicas exceeds max_replicas for dsqwen-7b"}
```

The SM state version stayed 94 and no vLLM power transition occurred. A
follow-up target=1 request was a no-op, confirming the cluster remained at the
baseline throughout the diagnosis.

## Root cause

`tre/deploy/registry.yaml` already specifies `max_replicas: 8` for both 7b and
llama. Two runtime inputs were older:

- the live `tre-v2-registry` ConfigMap, bootstrapped from
  `deploy/overlays/tre-v2/params.yaml`, still specified four; and
- the deployed service-manager uses the registry baked into its old
  `20260707-07717371` image (`TRE_REGISTRY_PATH=/app/tre/deploy/registry.yaml`),
  which also predates the eight-replica update.

The raw comparison is in `registry_source_comparison.txt`.

## Fix and guard

- `deploy/overlays/tre-v2/params.yaml` is regenerated from the current
  `deploy/registry.yaml`, including max replicas 8/8/4 and current calibrated
  theta values.
- `test_kustomize_overlays.py` now requires the embedded ConfigMap registry to
  equal `deploy/registry.yaml` structurally, preventing another silent drift.
- `test_wake_order.py::test_7b_target_five_uses_all_four_node9_slots`
  reconstructs the 1/1/1 baseline and asserts target=5 wakes all four node9 7b
  bindings, reaching five awake replicas.

The service-manager image built in Phase 3 will contain the corrected registry
and the H1 node-order fix. The mandatory live t1 tight-segment spot-check remains
pending until that batched rollout; no intermediate image is built for H6.

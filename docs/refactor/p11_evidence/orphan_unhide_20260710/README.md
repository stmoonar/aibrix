# Orphaned hidden pod repair (2026-07-10)

## Summary

Two sleeping bindings left by a safescale run were still marked hidden:

- `dsllama-8b-nscc-ds-4a100-node10-gpu-0-5f474d7f85-7mns7`
- `dsqwen-7b-nscc-ds-4a100-node10-gpu-0-78588bb7d8-b4p5p`

The safescale probe hash was empty, so neither binding belonged to an active
probe. Both bindings are sleeping and share node10 GPU 0 with the awake 14b
binding. The repair changed only hidden-state metadata; it did not wake or
sleep any model.

## Evidence

- `sm_state_before.txt`: Redis state before repair; both target bindings have
  `"awake":false` and `"hidden":true`.
- `safescale_probes_before.txt`: empty probe hash before repair.
- `sm_state_after.txt`: Redis state after repair; both target bindings have
  `"awake":false` and `"hidden":false`.
- `pod_metadata_after.txt`: both pod annotations are `sleeping` and both
  routable labels remain `false`, as required for sleeping pods.
- `reconcile_response.json`: service-manager reconciliation response (state
  version 94) showing both corrected bindings.

After one controller slow-loop interval, controller logs contained no warning,
error, hidden, or orphan complaint. `GET /api/snapshot` showed both bindings as
normal sleeping pool members (`awake:false`, `hidden:false`). The controller
remained in `observe` mode throughout.

## Endpoint defect found during repair

The planned `PUT /v2/models/{model}/routable` request returned HTTP 409,
captured in `unhide_dsllama-8b_status.txt` and
`unhide_dsllama-8b_response.json`. The current implementation incorrectly runs
wake-feasibility checks while unhiding a sleeping binding, even though unhide
does not wake it. Stopping the only awake 14b binding to satisfy that check
would have caused avoidable data-plane impact.

The zero-data-plane-impact recovery therefore used Kubernetes metadata as the
service-manager's observed state and then invoked its reconciliation API:

```bash
kubectl -n default annotate pod <serve-id> tre.aibrix.io/state=sleeping --overwrite
curl -X POST http://tre-v2-service-manager:8000/v2/reconcile
```

The physical `/is_sleeping` probe confirmed the state and reconciliation saved
the corrected bindings with the normal StateStore compare-and-set path. A
regression fix for `/routable` is included in the Phase 1 service-manager work.
# P4 Service Manager

Date: 2026-07-04
Environment: remote server 76, `/data/nfs_shared_data/xxy/aibrix`

## Slot Allocator Contract

The first P4 slice implements the pure, IO-free `tre_sm.allocator.slots` module required by REFACTOR_PLAN section 5.3.

Types:

- `Slot(node, gpu_ids)` identifies one 1-GPU half-slot or one complete 2-GPU slot.
- `Binding(serve_id, model, slot, awake)` separates slot binding from awake state; sleeping serves still occupy their slot.
- `Migration(serve_id, from_slot, to_slot)` describes a defrag move.
- `SlotAllocator(topology, bindings)` tracks only topology and bindings, with no Kubernetes, Redis, or vLLM IO.

Implemented rules:

- `find_slot(1)` fills the free half of an already split 2-GPU slot before splitting a new 2-GPU slot.
- `find_slot(2)` only returns a completely empty declared 2-GPU slot.
- `plan_defrag(2)` handles the required counterexample: two 1-GPU serves on GPUs 0 and 2 leave total free capacity but no complete slot; the plan migrates the serve on GPU 2 to GPU 1, freeing slot `(2,3)`.
- `plan_defrag(2)` can also consolidate split free halves across nodes when no single node has an intact 2-GPU slot.

## State Store Contract

The second P4 slice adds `tre_sm.state.store`, the persistence boundary that later reconcile/API code will use.

Rules:

- Redis hash `tre:v2:sm:state` stores one JSON payload per `serve_id`.
- Redis key `tre:v2:sm:version` stores the optimistic version number.
- `StateStore.load()` returns a deterministic `StateSnapshot(version, bindings)` sorted by `serve_id`.
- `StateStore.save(bindings, expected_version=...)` rejects stale writers with `StateConflict` before changing state.
- The store is tested with a fake Redis client and does not contact live Redis during unit tests.
- Redis byte and string response forms are accepted by the loader.

## Topology Adapter Contract

The fifth P4 slice adds `tre_sm.allocator.topology`, the normalized boundary between Kubernetes pod discovery and allocator/reconcile code.

Rules:

- `K8sPodSnapshot` is a small DTO for future real Kubernetes discovery and current fake tests.
- `pod_records_from_snapshots()` turns snapshots into `PodRecord`s sorted by pod name.
- `CUDA_VISIBLE_DEVICES` is required and wins over `tre.aibrix.io/gpu-ids` annotations during discovery normalization.
- `tre.aibrix.io/state` supplies awake/sleeping/hidden state, defaulting to awake when absent.
- Unknown nodes and invalid GPU slot shapes fail before reconcile by reusing `SlotAllocator` validation.

## Reconcile Contract

The fourth P4 slice adds `tre_sm.state.reconcile`, an IO-free startup reconciliation boundary for fake or real Kubernetes pod clients.

Rules:

- `PodRecord` is the normalized pod observation used by tests and future `ops.k8s_ops`.
- `CUDA_VISIBLE_DEVICES` is parsed into the binding slot and validated by `SlotAllocator`.
- Pod reality overrides stale Redis state for the same `serve_id`, matching REFACTOR_PLAN section 5.3.
- Persisted bindings without a current pod observation are retained conservatively and reported as warnings.
- Reconcile persists the merged result only when bindings change, preserving the state version on no-op restart.

## v1 Compatibility Contract

The eleventh P4 slice adds `tre_sm.api.v1_compat`, a thin compatibility adapter for migrated Go callers.

Rules:

- `POST /models_replicas?models=m1[,m2]` returns current awake, non-hidden replicas per model for APA sleep mode.
- `POST /scale_service?model_name=m&scale_type=up|down&scale_value=n` delegates to v2 target state and returns `{requested, actual}`.
- `POST /wake_up?model_name=m&kind=...&queue_len=...` wakes one sleeping replica for gateway zero-routable requests and returns the legacy success/delayed/strategy shape.
- The adapter is route-only: all state mutation stays in `ServiceManagerV2`.

## API v2 Contract

The eighth P4 slice adds `tre_sm.api.v2`, the declarative API surface required by REFACTOR_PLAN section 5.3.

Rules:

- `ServiceManagerV2.get_state()` returns current version, per-model awake/bound counts, and deterministic binding rows including hidden state.
- `ServiceManagerV2.put_model_target(model, wake_replicas=n)` validates registry limits and the already-bound warm pool.
- Target changes are persisted optimistically only when the desired target differs from current awake state.
- Repeating the same target is idempotent: no actions and no version bump.
- `ServiceManagerV2.put_model_routable(model, hidden_pods=[...])` persists SafeScale route-hidden state and is idempotent.
- `ServiceManagerV2.reconcile()` triggers manual startup reconciliation when a Kubernetes pod client is configured.
- `create_app(service)` exposes `/healthz`, `GET /v2/state`, `PUT /v2/models/{model}/target`, `PUT /v2/models/{model}/routable`, and `POST /v2/reconcile` as thin FastAPI routes over the service layer.
- `tre_sm.app.create_service_app()` wires registry, state store, and optional Kubernetes pod client into the FastAPI app without constructing live clients in tests.

## Kubernetes Ops Contract

The seventh P4 slice adds `tre_sm.ops.k8s_ops`, the Kubernetes client boundary for pod discovery and TRE annotations.

Rules:

- The Kubernetes API object is injected; unit tests use a fake object and never contact the cluster.
- `list_pod_snapshots()` lists running, non-deleting pods and maps them into `K8sPodSnapshot` for the topology adapter.
- The model selector uses `model.aibrix.ai/name=<model>`.
- `write_binding_annotations()` writes `tre.aibrix.io/gpu-ids` and `tre.aibrix.io/state` in Kubernetes patch body form.
- Unknown pod states are rejected before any patch call.

## vLLM Ops Contract

The sixth P4 slice adds `tre_sm.ops.vllm_ops`, the retrying boundary for vLLM `/sleep` and `/wake_up` calls.

Rules:

- The HTTP transport is injectable, so unit tests use fake transport and do not touch the network.
- `sleep()` posts `/sleep`; `wake_up()` posts `/wake_up`; both default to port 8000 and propagate configured timeout.
- Transient transport exceptions and non-success HTTP responses are retried up to `max_attempts`.
- HTTP 2xx is success; HTTP 409 is treated as idempotent success for repeated target-state operations.
- Exhausted retries return a structured `VllmOpResult` instead of raising transport-specific exceptions.

## Verification Log

### P4-SM-001 slot allocator

RED:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_slots.py
```

Result: failed during collection because `tre_sm.allocator.slots` did not exist.

GREEN:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_slots.py
PYTHONPATH=tre/common:tre/deploy:tre/controller:tre/controller/tests:tre/service-manager python3 -m pytest -q tre/common/tests tre/deploy/tests tre/controller/tests tre/service-manager/tests
cd tre && make check
```

Result: all passed on server 76. `make check` now includes `service-manager/tests` and passed with 17 tests.

### P4-SM-002 state store

RED:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_state_store.py
```

Result: failed during collection because `tre_sm.state.store` did not exist.

GREEN:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_state_store.py
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests
```

Result: focused state-store tests passed with 2 tests; service-manager tests passed with 4 tests.

### P4-SM-003 allocator property coverage

RED:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_slots.py
```

Result: failed because cross-node split halves could not produce a defrag migration.

GREEN:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_slots.py
cd tre && make check && make smoke
```

Result: slot tests passed with 4 tests; `make check` passed with 21 tests; smoke passed.

### P4-SM-004 reconcile startup state

RED:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_reconcile.py
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_state_store.py
```

Result: reconcile failed during collection because `tre_sm.state.reconcile` did not exist; the state-store regression failed because string Redis responses decoded to `None`.

GREEN:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_state_store.py tre/service-manager/tests/test_reconcile.py tre/service-manager/tests/test_slots.py
```

Result: focused service-manager tests passed with 9 tests.

### P4-SM-005 topology adapter

RED:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_topology.py
```

Result: failed during collection because `tre_sm.allocator.topology` did not exist.

GREEN:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_topology.py
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests
```

Result: focused topology tests passed with 2 tests; service-manager tests passed with 11 tests.

### P4-SM-006 vLLM ops wrapper

RED:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_vllm_ops.py
```

Result: failed during collection because `tre_sm.ops.vllm_ops` did not exist.

GREEN:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_vllm_ops.py
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests
```

Result: focused vLLM ops tests passed with 3 tests; service-manager tests passed with 14 tests.

### P4-SM-007 Kubernetes ops wrapper

RED:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_k8s_ops.py
```

Result: failed during collection because `tre_sm.ops.k8s_ops` did not exist.

GREEN:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_k8s_ops.py
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests
```

Result: focused Kubernetes ops tests passed with 2 tests; service-manager tests passed with 16 tests.

### P4-SM-008 API v2 state and target

RED:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_api_v2.py
```

Result: first failed during collection because `tre_sm.api.v2` did not exist; route coverage then failed because `create_app` did not exist.

GREEN:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_api_v2.py
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests
```

Result: focused API v2 tests passed with 4 tests; service-manager tests passed with 20 tests.

### P4-SM-009 API v2 routable hidden pods

RED:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_api_v2.py
```

Result: failed because `ServiceManagerV2.put_model_routable()` and `/v2/models/{model}/routable` did not exist.

GREEN:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_api_v2.py
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests
```

Result: focused API v2 tests passed with 6 tests; service-manager tests passed with 22 tests.

### P4-SM-010 API v2 reconcile and app wiring

RED:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_api_v2.py
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_app.py
```

Result: API tests failed because `ServiceManagerV2` did not accept a Kubernetes pod client for manual reconcile; app tests failed because `tre_sm.app` did not exist.

GREEN:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_app.py tre/service-manager/tests/test_api_v2.py
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests
```

Result: focused app/API tests passed with 9 tests; service-manager tests passed with 25 tests.

### P4-SM-011 v1 compatibility adapters

RED:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_v1_compat.py
```

Result: failed with 404s because `/models_replicas`, `/scale_service`, and `/wake_up` were not registered.

GREEN:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests/test_v1_compat.py
PYTHONPATH=tre/common:tre/service-manager python3 -m pytest -q tre/service-manager/tests
```

Result: focused v1 compatibility tests passed with 3 tests; service-manager tests passed with 28 tests.

## P4 Closure Audit

P4 acceptance coverage:

- Slot allocator: `test_slots.py` covers the required 0/2 fragmentation counterexample, split-slot fill order, cross-node defrag, and a seeded 1000-step allocation/release invariant test.
- State persistence: `test_state_store.py` covers versioned round-trips, stale writer rejection, and Redis byte/string response compatibility.
- Restart reconciliation: `test_reconcile.py` constructs persisted state plus fake pod reality and verifies pod `CUDA_VISIBLE_DEVICES` wins, merged state persists, and missing pod observations are retained conservatively.
- Topology/discovery: `test_topology.py` and `test_k8s_ops.py` cover snapshot normalization, annotation patch bodies, model selectors, and invalid slot rejection with fake Kubernetes clients.
- Ops: `test_vllm_ops.py` covers sleep/wake endpoint selection, retries, timeout propagation, idempotent 409 handling, and structured failures without network calls.
- API v2: `test_api_v2.py` and `test_app.py` cover `/healthz`, `GET /v2/state`, `PUT /v2/models/{model}/target`, `PUT /v2/models/{model}/routable`, `POST /v2/reconcile`, app wiring, and target idempotency.
- v1 compatibility: `test_v1_compat.py` covers `/models_replicas`, `/scale_service`, and `/wake_up` for migrated APA/gateway callers.

Final remote verification command:

```bash
cd /data/nfs_shared_data/xxy/aibrix/tre && make check && make smoke
```

Result: passed on server 76 with 43 tests and `tre smoke ok`.

## Remaining P4 Work

- None. P4 is closed by commit/tag after final verification.

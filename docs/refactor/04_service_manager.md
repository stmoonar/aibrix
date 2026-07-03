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

## Remaining P4 Work

- Implement Kubernetes ops wrapper, API v2, and v1 compatibility adapters.

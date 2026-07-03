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

## Remaining P4 Work

- Add allocator property/random sequence tests.
- Implement topology/state store/reconcile with fake Redis and fake Kubernetes clients.
- Implement vLLM/Kubernetes ops wrappers, API v2, and v1 compatibility adapters.

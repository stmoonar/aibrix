from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tre_common.registry import Registry
from tre_sm.allocator.slots import Binding, Migration, Slot, SlotAllocator
from tre_sm.allocator.topology import K8sPodSnapshot
from tre_sm.state.reconcile import K8sPodClient, POD_STATE_AWAKE, POD_STATE_HIDDEN, POD_STATE_SLEEPING, reconcile_state
from tre_sm.state.store import StateConflict, StateStore
from tre_sm.api.v1_compat import create_v1_compat_router




class RuntimePodOps(Protocol):
    def list_pod_snapshots(self, *, model: str | None = None) -> list[K8sPodSnapshot]: ...

    def write_binding_annotations(self, binding: Binding, *, state: str) -> None: ...


class VllmRuntimeOps(Protocol):
    def sleep(self, pod_ip: str, *, port: int | None = None): ...

    def wake_up(self, pod_ip: str, *, port: int | None = None): ...

class ServiceManagerV2:
    def __init__(
        self,
        registry: Registry,
        store: StateStore,
        *,
        k8s_client: K8sPodClient | None = None,
        runtime_ops: RuntimePodOps | None = None,
        vllm_ops: VllmRuntimeOps | None = None,
    ) -> None:
        self._registry = registry
        self._store = store
        self._k8s_client = k8s_client
        self._runtime_ops = runtime_ops
        self._vllm_ops = vllm_ops

    def get_state(self) -> dict:
        snapshot = self._store.load()
        return {
            "version": snapshot.version,
            "models": self._model_counts(snapshot.bindings),
            "bindings": [self._binding_dict(binding) for binding in snapshot.bindings],
        }

    def put_model_target(self, model: str, *, wake_replicas: int) -> dict:
        spec = self._registry.model(model)
        if wake_replicas < 0:
            raise ValueError("wake_replicas must be non-negative")
        if wake_replicas > spec.max_replicas:
            raise ValueError(f"wake_replicas exceeds max_replicas for {model}")

        snapshot = self._store.load()
        model_bindings = [binding for binding in snapshot.bindings if binding.model == model]
        if self._runtime_ops is not None and wake_replicas > len(model_bindings):
            raise ValueError("runtime create is not implemented for target growth beyond existing bindings")
        awake = [binding for binding in model_bindings if binding.awake]
        actions: list[dict] = []
        updated_by_serve = {binding.serve_id: binding for binding in snapshot.bindings}

        if len(awake) < wake_replicas:
            sleeping = [binding for binding in model_bindings if not binding.awake]
            wake_existing = min(wake_replicas - len(awake), len(sleeping))
            for binding in sleeping[:wake_existing]:
                self._ensure_feasible_wake(binding, list(updated_by_serve.values()))
                self._apply_runtime_power_action(binding, action="wake")
                updated_by_serve[binding.serve_id] = replace(binding, awake=True, hidden=False)
                actions.append({"action": "wake", "serve_id": binding.serve_id})
            create_count = wake_replicas - len(awake) - wake_existing
            if create_count > 0:
                allocator = SlotAllocator(self._registry.topology(), list(updated_by_serve.values()))
                existing_serve_ids = set(updated_by_serve)
                for _ in range(create_count):
                    slot = allocator.find_slot(spec.tp_size)
                    if slot is None:
                        raise ValueError(f"no free slot for {model} tp_size={spec.tp_size}")
                    serve_id = _next_serve_id(model, existing_serve_ids)
                    allocator.bind(serve_id, model, slot, awake=True)
                    existing_serve_ids.add(serve_id)
                    updated_by_serve[serve_id] = Binding(serve_id, model, slot, awake=True)
                    actions.append(
                        {
                            "action": "create",
                            "serve_id": serve_id,
                            "node": slot.node,
                            "gpu_ids": list(slot.gpu_ids),
                        }
                    )
        elif len(awake) > wake_replicas:
            for binding in reversed(awake[wake_replicas:]):
                self._apply_runtime_power_action(binding, action="sleep")
                updated_by_serve[binding.serve_id] = replace(binding, awake=False, hidden=False)
                actions.append({"action": "sleep", "serve_id": binding.serve_id})

        version = snapshot.version
        if actions:
            updated = list(updated_by_serve.values())
            try:
                version = self._store.save(updated, expected_version=snapshot.version)
            except StateConflict:
                current = self._store.load()
                current_counts = self._model_counts(current.bindings).get(model, {"awake": 0})
                if current_counts["awake"] != wake_replicas:
                    raise
                version = current.version

        return {
            "model": model,
            "wake_replicas": wake_replicas,
            "version": version,
            "actions": actions,
        }


    def put_model_routable(self, model: str, *, hidden_pods: list[str]) -> dict:
        self._registry.model(model)
        snapshot = self._store.load()
        model_bindings = [binding for binding in snapshot.bindings if binding.model == model]
        model_serve_ids = {binding.serve_id for binding in model_bindings}
        requested_hidden = set(hidden_pods)
        unknown = requested_hidden - model_serve_ids
        if unknown:
            raise ValueError(f"unknown pods for {model}: {sorted(unknown)}")

        actions: list[dict] = []
        updated_by_serve = {binding.serve_id: binding for binding in snapshot.bindings}
        for binding in model_bindings:
            should_hide = binding.serve_id in requested_hidden
            if binding.hidden == should_hide:
                continue
            if not should_hide:
                self._ensure_feasible_wake(binding, list(updated_by_serve.values()))
            if self._runtime_ops is not None:
                self._runtime_ops.write_binding_annotations(
                    binding,
                    state=POD_STATE_HIDDEN if should_hide else POD_STATE_AWAKE,
                )
            updated_by_serve[binding.serve_id] = replace(binding, hidden=should_hide)
            actions.append({"action": "hide" if should_hide else "unhide", "serve_id": binding.serve_id})

        version = snapshot.version
        if actions:
            updated = [updated_by_serve[binding.serve_id] for binding in snapshot.bindings]
            version = self._store.save(updated, expected_version=snapshot.version)

        return {
            "model": model,
            "hidden_pods": sorted(requested_hidden),
            "version": version,
            "actions": actions,
        }


    def defrag(self, *, tp_size: int) -> dict:
        snapshot = self._store.load()
        allocator = SlotAllocator(self._registry.topology(), snapshot.bindings)
        migrations = allocator.plan_defrag(tp_size)
        if migrations is None:
            raise DefragUnavailable("no_feasible_defrag")

        actions: list[dict] = []
        updated_by_serve = {binding.serve_id: binding for binding in snapshot.bindings}
        for migration in migrations:
            binding = updated_by_serve[migration.serve_id]
            actions.extend(
                [
                    {"action": "hide", "serve_id": migration.serve_id},
                    {"action": "sleep", "serve_id": migration.serve_id},
                    {
                        "action": "recreate",
                        "serve_id": migration.serve_id,
                        "node": migration.to_slot.node,
                        "gpu_ids": list(migration.to_slot.gpu_ids),
                    },
                    {"action": "wake", "serve_id": migration.serve_id},
                    {"action": "unhide", "serve_id": migration.serve_id},
                ]
            )
            updated_by_serve[migration.serve_id] = replace(binding, slot=migration.to_slot, awake=True, hidden=False)

        version = snapshot.version
        if migrations:
            updated = [updated_by_serve[binding.serve_id] for binding in snapshot.bindings]
            version = self._store.save(updated, expected_version=snapshot.version)
        return {
            "version": version,
            "migrations": [_migration_dict(migration) for migration in migrations],
            "actions": actions,
        }


    def reconcile(self) -> dict:
        if self._k8s_client is None:
            raise ValueError("k8s_client is required for reconcile")
        result = reconcile_state(self._registry.topology(), self._store, self._k8s_client)
        return {
            "version": result.version,
            "warnings": result.warnings,
            "bindings": [self._binding_dict(binding) for binding in result.bindings],
        }

    def _apply_runtime_power_action(self, binding: Binding, *, action: str) -> None:
        if self._runtime_ops is None or self._vllm_ops is None:
            return
        snapshot = self._snapshot_for_binding(binding)
        if not snapshot.pod_ip:
            raise ValueError(f"pod {binding.serve_id} has no pod IP for {action}")

        if action == "sleep":
            result = self._vllm_ops.sleep(snapshot.pod_ip, port=8000)
            state = POD_STATE_SLEEPING
        elif action == "wake":
            result = self._vllm_ops.wake_up(snapshot.pod_ip, port=8000)
            state = POD_STATE_AWAKE
        else:
            raise ValueError(f"unknown runtime action: {action}")

        if not bool(getattr(result, "success", False)):
            message = getattr(result, "message", "") or "operation failed"
            raise ValueError(f"vLLM {action} failed for {binding.serve_id}: {message}")
        self._runtime_ops.write_binding_annotations(binding, state=state)

    def _ensure_feasible_wake(self, binding: Binding, bindings: list[Binding]) -> None:
        try:
            allocator = SlotAllocator(self._registry.topology(), bindings)
        except ValueError as exc:
            raise WakeConflict(str(exc)) from exc
        if not allocator.feasible_wake(binding.serve_id):
            raise WakeConflict(f"{binding.serve_id}: slot already has awake binding")

    def _snapshot_for_binding(self, binding: Binding) -> K8sPodSnapshot:
        snapshots = self._runtime_ops.list_pod_snapshots(model=binding.model) if self._runtime_ops else []
        for snapshot in snapshots:
            if snapshot.name == binding.serve_id:
                return snapshot
        raise ValueError(f"pod {binding.serve_id} not found for runtime operation")

    def _model_counts(self, bindings: list[Binding]) -> dict[str, dict[str, int]]:
        counts = {model.name: {"awake": 0, "bound": 0} for model in self._registry.models()}
        for binding in bindings:
            bucket = counts.setdefault(binding.model, {"awake": 0, "bound": 0})
            bucket["bound"] += 1
            if binding.awake:
                bucket["awake"] += 1
        return counts

    def _binding_dict(self, binding: Binding) -> dict:
        return {
            "serve_id": binding.serve_id,
            "model": binding.model,
            "node": binding.slot.node,
            "gpu_ids": list(binding.slot.gpu_ids),
            "awake": binding.awake,
            "hidden": binding.hidden,
        }


class DefragUnavailable(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class WakeConflict(ValueError):
    pass


class TargetRequest(BaseModel):
    wake_replicas: int

class DefragRequest(BaseModel):
    tp_size: int




class RoutableRequest(BaseModel):
    hidden_pods: list[str]

def create_app(service: ServiceManagerV2) -> FastAPI:
    app = FastAPI()
    app.include_router(create_v1_compat_router(service))

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}


    @app.post("/v2/reconcile")
    def reconcile() -> dict:
        try:
            return service.reconcile()
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v2/state")
    def get_state() -> dict:
        return service.get_state()


    @app.post("/v2/defrag")
    def defrag(request: DefragRequest) -> dict:
        try:
            return service.defrag(tp_size=request.tp_size)
        except DefragUnavailable as exc:
            raise HTTPException(status_code=409, detail={"reason": exc.reason}) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @app.put("/v2/models/{model}/routable")
    def put_model_routable(model: str, request: RoutableRequest) -> dict:
        try:
            return service.put_model_routable(model, hidden_pods=request.hidden_pods)
        except WakeConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/v2/models/{model}/target")
    def put_model_target(model: str, request: TargetRequest) -> dict:
        try:
            return service.put_model_target(model, wake_replicas=request.wake_replicas)
        except WakeConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app

def _migration_dict(migration: Migration) -> dict:
    return {
        "serve_id": migration.serve_id,
        "from_slot": _slot_dict(migration.from_slot),
        "to_slot": _slot_dict(migration.to_slot),
    }


def _slot_dict(slot: Slot) -> dict:
    return {"node": slot.node, "gpu_ids": list(slot.gpu_ids)}


def _next_serve_id(model: str, existing_serve_ids: set[str]) -> str:
    base = "".join(char if char.isalnum() else "-" for char in model).strip("-") or "serve"
    index = 1
    while f"{base}-{index}" in existing_serve_ids:
        index += 1
    return f"{base}-{index}"

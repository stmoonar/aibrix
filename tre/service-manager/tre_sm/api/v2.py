from __future__ import annotations

from dataclasses import replace

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tre_common.registry import Registry
from tre_sm.allocator.slots import Binding, Migration, Slot, SlotAllocator
from tre_sm.state.reconcile import K8sPodClient, reconcile_state
from tre_sm.state.store import StateStore
from tre_sm.api.v1_compat import create_v1_compat_router


class ServiceManagerV2:
    def __init__(
        self,
        registry: Registry,
        store: StateStore,
        *,
        k8s_client: K8sPodClient | None = None,
    ) -> None:
        self._registry = registry
        self._store = store
        self._k8s_client = k8s_client

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
        awake = [binding for binding in model_bindings if binding.awake]
        actions: list[dict] = []
        updated_by_serve = {binding.serve_id: binding for binding in snapshot.bindings}

        if len(awake) < wake_replicas:
            sleeping = [binding for binding in model_bindings if not binding.awake]
            wake_existing = min(wake_replicas - len(awake), len(sleeping))
            for binding in sleeping[:wake_existing]:
                updated_by_serve[binding.serve_id] = replace(binding, awake=True)
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
                updated_by_serve[binding.serve_id] = replace(binding, awake=False)
                actions.append({"action": "sleep", "serve_id": binding.serve_id})

        version = snapshot.version
        if actions:
            updated = list(updated_by_serve.values())
            version = self._store.save(updated, expected_version=snapshot.version)

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
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/v2/models/{model}/target")
    def put_model_target(model: str, request: TargetRequest) -> dict:
        try:
            return service.put_model_target(model, wake_replicas=request.wake_replicas)
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

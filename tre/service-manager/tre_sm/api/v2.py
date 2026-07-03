from __future__ import annotations

from dataclasses import replace

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tre_common.registry import Registry
from tre_sm.allocator.slots import Binding
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
        if wake_replicas > len(model_bindings):
            raise ValueError(f"wake_replicas exceeds bound pool for {model}")

        awake = [binding for binding in model_bindings if binding.awake]
        actions: list[dict] = []
        updated_by_serve = {binding.serve_id: binding for binding in snapshot.bindings}

        if len(awake) < wake_replicas:
            sleeping = [binding for binding in model_bindings if not binding.awake]
            for binding in sleeping[: wake_replicas - len(awake)]:
                updated_by_serve[binding.serve_id] = replace(binding, awake=True)
                actions.append({"action": "wake", "serve_id": binding.serve_id})
        elif len(awake) > wake_replicas:
            for binding in reversed(awake[wake_replicas:]):
                updated_by_serve[binding.serve_id] = replace(binding, awake=False)
                actions.append({"action": "sleep", "serve_id": binding.serve_id})

        version = snapshot.version
        if actions:
            updated = [updated_by_serve[binding.serve_id] for binding in snapshot.bindings]
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


class TargetRequest(BaseModel):
    wake_replicas: int



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

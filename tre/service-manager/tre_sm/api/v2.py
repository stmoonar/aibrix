from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from typing import Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tre_common.registry import Registry
from tre_common.registry import NodeSpec
from tre_sm.allocator.slots import Binding, Migration, Slot, SlotAllocator
from tre_sm.allocator.topology import K8sPodSnapshot
from tre_sm.gpu_truth import GpuTruthProvider
from tre_sm.state.reconcile import K8sPodClient, POD_STATE_AWAKE, POD_STATE_HIDDEN, POD_STATE_SLEEPING, reconcile_state
from tre_sm.state.store import StateConflict, StateStore
from tre_sm.api.v1_compat import create_v1_compat_router


_NAT_SPLIT = re.compile(r"(\d+)")


class RuntimePodOps(Protocol):
    def list_pod_snapshots(self, *, model: str | None = None) -> list[K8sPodSnapshot]: ...

    def write_binding_annotations(self, binding: Binding, *, state: str) -> None: ...

    def set_pod_routable(self, serve_id: str, *, routable: bool) -> None: ...

    def wait_pod_unroutable(self, binding: Binding): ...

    def ensure_model_httproute(self, model: str): ...

    def delete_model_deployment(self, binding: Binding) -> str: ...

    def create_model_deployment(self, model: str, slot: Slot) -> str: ...

    def wait_pod_deleted(self, serve_id: str): ...

    def wait_pod_ready(self, serve_id: str) -> K8sPodSnapshot: ...


class VllmRuntimeOps(Protocol):
    def sleep(self, pod_ip: str, *, port: int | None = None): ...

    def wake_up(self, pod_ip: str, *, port: int | None = None): ...

    def is_sleeping(self, pod_ip: str, *, port: int | None = None) -> bool | None: ...


class _VllmPodProber:
    """Adapt VllmRuntimeOps.is_sleeping to the reconcile PodPhysicalProber."""

    def __init__(self, vllm_ops: VllmRuntimeOps) -> None:
        self._vllm_ops = vllm_ops

    def is_sleeping(self, pod) -> bool | None:
        pod_ip = getattr(pod, "pod_ip", None)
        if not pod_ip:
            return None
        return self._vllm_ops.is_sleeping(pod_ip, port=8000)

class ServiceManagerV2:
    def __init__(
        self,
        registry: Registry,
        store: StateStore,
        *,
        k8s_client: K8sPodClient | None = None,
        runtime_ops: RuntimePodOps | None = None,
        vllm_ops: VllmRuntimeOps | None = None,
        gpu_truth: GpuTruthProvider | None = None,
        create_max_used_mib: int = 2500,
        sleep_leak_used_mib: int = 8192,
    ) -> None:
        self._registry = registry
        self._store = store
        self._k8s_client = k8s_client
        self._runtime_ops = runtime_ops
        self._vllm_ops = vllm_ops
        self._gpu_truth = gpu_truth
        self._create_max_used_mib = create_max_used_mib
        self._sleep_leak_used_mib = sleep_leak_used_mib

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
        if self._runtime_ops is not None and wake_replicas > len(model_bindings) and not self._has_deployment_ops():
            raise ValueError("runtime create is not implemented for target growth beyond existing bindings")
        awake = [binding for binding in model_bindings if binding.awake]
        actions: list[dict] = []
        updated_by_serve = {binding.serve_id: binding for binding in snapshot.bindings}

        if len(awake) < wake_replicas:
            sleeping = [binding for binding in model_bindings if not binding.awake]
            awake_by_node = Counter(
                binding.slot.node for binding in snapshot.bindings if binding.awake
            )
            sleeping.sort(
                key=lambda binding: (
                    awake_by_node[binding.slot.node], _natural_key(binding.serve_id)
                )
            )
            wake_existing = 0
            wake_needed = wake_replicas - len(awake)
            skipped_conflict: str | None = None
            for binding in sleeping:
                if wake_existing >= wake_needed:
                    break
                if not self._feasible_wake(binding, list(updated_by_serve.values())):
                    if skipped_conflict is None:
                        skipped_conflict = f"{binding.serve_id}: slot already has awake binding"
                    continue
                self._apply_runtime_power_action(binding, action="wake")
                updated_by_serve[binding.serve_id] = replace(binding, awake=True, hidden=False)
                actions.append({"action": "wake", "serve_id": binding.serve_id})
                wake_existing += 1
            create_count = max(0, wake_replicas - len(model_bindings))
            if len(awake) + wake_existing + create_count < wake_replicas:
                raise WakeConflict(skipped_conflict or f"{model}: no feasible sleeping binding for target")
            if create_count > 0:
                allocator = SlotAllocator(self._registry.topology(), list(updated_by_serve.values()))
                existing_serve_ids = set(updated_by_serve)
                for _ in range(create_count):
                    slot = allocator.find_slot(spec.tp_size)
                    if slot is None:
                        raise ValueError(f"no free slot for {model} tp_size={spec.tp_size}")
                    serve_id = _next_serve_id(model, existing_serve_ids)
                    allocator.bind(serve_id, model, slot, awake=True)
                    binding = Binding(serve_id, model, slot, awake=True)
                    if self._has_deployment_ops():
                        binding = self._create_and_wake_runtime_binding(model, slot)
                    existing_serve_ids.add(binding.serve_id)
                    updated_by_serve[binding.serve_id] = binding
                    actions.append(
                        {
                            "action": "create",
                            "serve_id": binding.serve_id,
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
            if not should_hide and binding.awake:
                self._ensure_feasible_wake(binding, list(updated_by_serve.values()))
            if self._runtime_ops is not None:
                state = POD_STATE_HIDDEN if should_hide else (
                    POD_STATE_AWAKE if binding.awake else POD_STATE_SLEEPING
                )
                self._runtime_ops.write_binding_annotations(binding, state=state)
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
        if migrations:
            self._ensure_all_model_routes()

        actions: list[dict] = []
        updated_by_serve = {binding.serve_id: binding for binding in snapshot.bindings}
        for migration in migrations:
            binding = updated_by_serve[migration.serve_id]
            if self._has_deployment_ops():
                migration_actions, moved_binding = self._execute_runtime_defrag_migration(binding, migration)
                actions.extend(migration_actions)
                updated_by_serve.pop(binding.serve_id, None)
                updated_by_serve[moved_binding.serve_id] = moved_binding
            else:
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
            updated = [updated_by_serve[serve_id] for serve_id in sorted(updated_by_serve)]
            version = self._store.save(updated, expected_version=snapshot.version)
        return {
            "version": version,
            "migrations": [_migration_dict(migration) for migration in migrations],
            "actions": actions,
        }


    def reconcile(self) -> dict:
        if self._k8s_client is None:
            raise ValueError("k8s_client is required for reconcile")
        prober = None
        if self._vllm_ops is not None and hasattr(self._vllm_ops, "is_sleeping"):
            prober = _VllmPodProber(self._vllm_ops)
        label_writer = None
        if self._runtime_ops is not None and hasattr(self._runtime_ops, "set_pod_routable"):
            label_writer = self._runtime_ops
        result = reconcile_state(
            self._registry.topology(),
            self._store,
            self._k8s_client,
            gpu_truth=self._gpu_truth,
            sleep_leak_used_mib=self._sleep_leak_used_mib,
            prober=prober,
            label_writer=label_writer,
        )
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

    def _has_deployment_ops(self) -> bool:
        return self._runtime_ops is not None and all(
            hasattr(self._runtime_ops, name)
            for name in (
                "delete_model_deployment",
                "create_model_deployment",
                "wait_pod_deleted",
                "wait_pod_ready",
            )
        )

    def _create_and_wake_runtime_binding(self, model: str, slot: Slot) -> Binding:
        if self._runtime_ops is None or self._vllm_ops is None:
            raise ValueError("runtime_ops and vllm_ops are required for runtime create")
        self._ensure_create_headroom(slot)
        self._ensure_model_route(model)
        deployment_id = self._runtime_ops.create_model_deployment(model, slot)
        self._ensure_model_route(model)
        ready = self._runtime_ops.wait_pod_ready(deployment_id)
        if not ready.pod_ip:
            raise ValueError(f"pod {ready.name} has no pod IP for wake")
        ready_result = self._vllm_ops.wait_until_ready(ready.pod_ip, port=8000)
        if not bool(getattr(ready_result, "success", False)):
            message = getattr(ready_result, "message", "") or "operation failed"
            raise ValueError(f"vLLM readiness failed for {ready.name}: {message}")
        result = self._vllm_ops.wake_up(ready.pod_ip, port=8000)
        if not bool(getattr(result, "success", False)):
            message = getattr(result, "message", "") or "operation failed"
            raise ValueError(f"vLLM wake failed for {ready.name}: {message}")
        binding = Binding(ready.name, model, slot, awake=True, hidden=False)
        self._runtime_ops.write_binding_annotations(binding, state=POD_STATE_AWAKE)
        return binding

    def _execute_runtime_defrag_migration(self, binding: Binding, migration: Migration) -> tuple[list[dict], Binding]:
        if self._runtime_ops is None or self._vllm_ops is None:
            raise ValueError("runtime_ops and vllm_ops are required for runtime defrag")

        actions: list[dict] = []
        self._runtime_ops.write_binding_annotations(binding, state=POD_STATE_HIDDEN)
        actions.append({"action": "hide", "serve_id": binding.serve_id})
        self._runtime_ops.wait_pod_unroutable(binding)

        self._apply_runtime_power_action(binding, action="sleep")
        actions.append({"action": "sleep", "serve_id": binding.serve_id})

        self._runtime_ops.delete_model_deployment(binding)
        actions.append({"action": "delete_deployment", "serve_id": binding.serve_id})

        self._runtime_ops.wait_pod_deleted(binding.serve_id)
        self._ensure_create_headroom(migration.to_slot)
        deployment_id = self._runtime_ops.create_model_deployment(binding.model, migration.to_slot)
        self._ensure_model_route(binding.model)
        ready = self._runtime_ops.wait_pod_ready(deployment_id)
        new_serve_id = ready.name
        actions.append(
            {
                "action": "create_deployment",
                "serve_id": new_serve_id,
                "node": migration.to_slot.node,
                "gpu_ids": list(migration.to_slot.gpu_ids),
            }
        )
        if not ready.pod_ip:
            raise ValueError(f"pod {new_serve_id} has no pod IP for wake")
        ready_result = self._vllm_ops.wait_until_ready(ready.pod_ip, port=8000)
        if not bool(getattr(ready_result, "success", False)):
            message = getattr(ready_result, "message", "") or "operation failed"
            raise ValueError(f"vLLM readiness failed for {new_serve_id}: {message}")
        result = self._vllm_ops.wake_up(ready.pod_ip, port=8000)
        if not bool(getattr(result, "success", False)):
            message = getattr(result, "message", "") or "operation failed"
            raise ValueError(f"vLLM wake failed for {new_serve_id}: {message}")

        moved = Binding(new_serve_id, binding.model, migration.to_slot, awake=True, hidden=False)
        self._runtime_ops.write_binding_annotations(moved, state=POD_STATE_AWAKE)
        actions.append({"action": "wake", "serve_id": new_serve_id})
        actions.append({"action": "unhide", "serve_id": new_serve_id})
        return actions, moved

    def _ensure_feasible_wake(self, binding: Binding, bindings: list[Binding]) -> None:
        if not self._feasible_wake(binding, bindings):
            raise WakeConflict(f"{binding.serve_id}: slot already has awake binding")

    def _ensure_model_route(self, model: str) -> None:
        if self._runtime_ops is not None and hasattr(self._runtime_ops, "ensure_model_httproute"):
            self._runtime_ops.ensure_model_httproute(model)

    def _ensure_all_model_routes(self) -> None:
        for model in self._registry.models():
            self._ensure_model_route(model.name)

    def _feasible_wake(self, binding: Binding, bindings: list[Binding]) -> bool:
        try:
            allocator = SlotAllocator(self._registry.topology(), bindings)
        except ValueError as exc:
            raise WakeConflict(str(exc)) from exc
        return allocator.feasible_wake(binding.serve_id)

    def _ensure_create_headroom(self, slot: Slot) -> None:
        if self._gpu_truth is None:
            return
        nodes = {node.name: node for node in self._registry.topology().nodes}
        node = nodes.get(slot.node)
        for gpu_id in slot.gpu_ids:
            gpu_uuid = _gpu_uuid(node, gpu_id)
            if gpu_uuid is None:
                continue
            used_mib = self._gpu_truth.used_mib(node=slot.node, gpu_id=gpu_id, gpu_uuid=gpu_uuid)
            if used_mib is None or used_mib <= self._create_max_used_mib:
                continue
            raise ValueError(
                "insufficient startup headroom: "
                f"{slot.node}/{gpu_uuid} used_mib={used_mib} max_used_mib={self._create_max_used_mib}"
            )

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

def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part for part in _NAT_SPLIT.split(value))


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


def _gpu_uuid(node: NodeSpec | None, gpu_id: int) -> str | None:
    if node is None:
        return None
    if gpu_id < 0 or gpu_id >= len(node.gpu_uuids):
        return None
    return node.gpu_uuids[gpu_id]

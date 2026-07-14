from __future__ import annotations

import re
import json
from collections import Counter
from dataclasses import replace
from dataclasses import asdict
from functools import wraps
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tre_common.registry import Registry
from tre_common.registry import NodeSpec
from tre_sm.allocator.slots import Binding, Migration, Slot, SlotAllocator
from tre_sm.allocator.topology import K8sPodSnapshot
from tre_sm.gpu_truth import GpuTruthProvider
from tre_sm.ops.k8s_ops import StartupPodRecord
from tre_sm.state.reconcile import K8sPodClient, POD_STATE_AWAKE, POD_STATE_HIDDEN, POD_STATE_SLEEPING, audit_state, reconcile_state
from tre_sm.state.operations import OperationBusy, OperationCoordinator
from tre_sm.state.fleet_repair import FleetRepairExecutor
from tre_sm.state.fleet_store import DesiredBinding, FleetStateStore, ObservedBinding
from tre_sm.state.safety import ClusterSafetyGate, ControllerNotPaused, NodePressureActive
from tre_sm.state.gpu_leases import GpuLeaseConflict, GpuLeaseStore
from tre_sm.state.store import StateConflict, StateFenceError, StateStore
from tre_sm.api.v1_compat import create_v1_compat_router


_NAT_SPLIT = re.compile(r"(\d+)")


def serialized_operation(kind: str):
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            if self._operation_coordinator is None:
                return method(self, *args, **kwargs)
            with self._operation_coordinator.operation(kind) as operation:
                operation.advance("executing")
                return method(self, *args, **kwargs)

        return wrapped

    return decorate


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

    def get_startup_pod(self, name: str) -> StartupPodRecord: ...

    def admit_startup_pod(
        self,
        name: str,
        *,
        pod_uid: str,
        suspended_binding_ids: list[str],
        operation_id: str,
    ) -> None: ...

    def clear_startup_admission(self, name: str) -> None: ...

    def list_admitted_startup_pods(self) -> list[StartupPodRecord]: ...

    def list_startup_resident_snapshots(self) -> list[K8sPodSnapshot]: ...


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
        operation_coordinator: OperationCoordinator | None = None,
        safety_gate: ClusterSafetyGate | None = None,
        fleet_store: FleetStateStore | None = None,
        gpu_leases: GpuLeaseStore | None = None,
    ) -> None:
        self._registry = registry
        self._store = store
        self._k8s_client = k8s_client
        self._runtime_ops = runtime_ops
        self._vllm_ops = vllm_ops
        self._gpu_truth = gpu_truth
        self._create_max_used_mib = create_max_used_mib
        self._sleep_leak_used_mib = sleep_leak_used_mib
        self._operation_coordinator = operation_coordinator
        self._safety_gate = safety_gate
        self._fleet_store = fleet_store
        self._gpu_leases = gpu_leases
        self._supervisor = None
        self._fleet_repair = None
        if (
            runtime_ops is not None
            and vllm_ops is not None
            and safety_gate is not None
            and all(
                hasattr(runtime_ops, name)
                for name in (
                    "list_model_deployments",
                    "scale_model_deployment",
                    "wait_deployment_pods_deleted",
                )
            )
        ):
            self._fleet_repair = FleetRepairExecutor(
                runtime_ops=runtime_ops,
                vllm_ops=vllm_ops,
                safety_gate=safety_gate,
                gpu_leases=gpu_leases,
            )

    def set_supervisor(self, supervisor) -> None:
        self._supervisor = supervisor

    def get_supervisor_state(self) -> dict:
        if self._supervisor is None:
            return {"running": False, "enabled": False}
        return {"enabled": True, **asdict(self._supervisor.snapshot())}

    def get_state(self) -> dict:
        snapshot = self._store.load()
        state = {
            "version": snapshot.version,
            "models": self._model_counts(snapshot.bindings),
            "bindings": [self._binding_dict(binding) for binding in snapshot.bindings],
        }
        if self._fleet_store is not None:
            state["fleet"] = self.get_fleet_state()
        return state

    @serialized_operation("put_model_target")
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
        plan = self._plan_model_target(
            model=model,
            wake_replicas=wake_replicas,
            bindings=snapshot.bindings,
            tp_size=spec.tp_size,
        )
        self._set_model_desired_target(
            model=model,
            target_bindings=plan["target_bindings"],
            reason="model_target_request",
        )
        actions: list[dict] = []
        updated_by_serve = {binding.serve_id: binding for binding in snapshot.bindings}

        for binding in plan["sleep"]:
            self._apply_runtime_power_action(binding, action="sleep")
            updated_by_serve[binding.serve_id] = replace(
                binding, awake=False, hidden=False
            )
            actions.append({"action": "sleep", "serve_id": binding.serve_id})

        for binding in plan["wake"]:
            self._apply_runtime_power_action(binding, action="wake")
            updated_by_serve[binding.serve_id] = replace(
                binding, awake=True, hidden=False
            )
            actions.append({"action": "wake", "serve_id": binding.serve_id})

        for planned in plan["create"]:
            binding = planned
            if self._has_deployment_ops():
                binding = self._create_and_wake_runtime_binding(
                    model, planned.slot
                )
            updated_by_serve[binding.serve_id] = binding
            actions.append(
                {
                    "action": "create",
                    "serve_id": binding.serve_id,
                    "node": binding.slot.node,
                    "gpu_ids": list(binding.slot.gpu_ids),
                }
            )

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

    def _plan_model_target(
        self,
        *,
        model: str,
        wake_replicas: int,
        bindings: list[Binding],
        tp_size: int,
    ) -> dict[str, list[Binding]]:
        model_bindings = [binding for binding in bindings if binding.model == model]
        awake = [binding for binding in model_bindings if binding.awake]
        planning = {binding.serve_id: binding for binding in bindings}
        if len(awake) >= wake_replicas:
            sleeping = list(reversed(awake[wake_replicas:]))
            target = awake[:wake_replicas]
            return {
                "sleep": sleeping,
                "wake": [],
                "create": [],
                "target_bindings": target,
            }

        target = list(awake)
        sleeping = [binding for binding in model_bindings if not binding.awake]
        awake_by_node = Counter(
            binding.slot.node for binding in bindings if binding.awake
        )
        wakes: list[Binding] = []
        existing_target = min(wake_replicas, len(model_bindings))
        while sleeping and len(target) < existing_target:
            feasible = [
                binding
                for binding in sleeping
                if self._feasible_wake(binding, list(planning.values()))
            ]
            if not feasible:
                raise WakeConflict(
                    f"{sleeping[0].serve_id}: slot already has awake binding"
                )
            binding = min(
                feasible,
                key=lambda item: (
                    awake_by_node[item.slot.node], _natural_key(item.serve_id)
                ),
            )
            sleeping.remove(binding)
            planning[binding.serve_id] = replace(
                binding, awake=True, hidden=False
            )
            wakes.append(binding)
            target.append(planning[binding.serve_id])
            awake_by_node[binding.slot.node] += 1

        creates: list[Binding] = []
        existing_ids = set(planning)
        allocator = SlotAllocator(
            self._registry.topology(), list(planning.values())
        )
        while len(target) + len(creates) < wake_replicas:
            slot = allocator.find_slot(tp_size)
            if slot is None:
                raise ValueError(f"no free slot for {model} tp_size={tp_size}")
            serve_id = _next_serve_id(model, existing_ids)
            allocator.bind(serve_id, model, slot, awake=True)
            planned = Binding(serve_id, model, slot, awake=True)
            planning[serve_id] = planned
            existing_ids.add(serve_id)
            creates.append(planned)

        return {
            "sleep": [],
            "wake": wakes,
            "create": creates,
            "target_bindings": target + creates,
        }

    @serialized_operation("put_binding_power")
    def put_binding_power(self, serve_id: str, *, awake: bool) -> dict:
        return self._put_binding_power_unlocked(serve_id, awake=awake)

    def _put_binding_power_unlocked(self, serve_id: str, *, awake: bool) -> dict:
        snapshot = self._store.load()
        binding = next(
            (item for item in snapshot.bindings if item.serve_id == serve_id),
            None,
        )
        if binding is None:
            raise ValueError(f"unknown binding: {serve_id}")

        self._update_desired(
            {binding.binding_id: {"power": "awake" if awake else "sleeping"}},
            updated_by="service-manager-api",
            reason="binding_power_request",
        )

        actions: list[dict] = []
        version = snapshot.version
        updated_binding = binding
        if binding.awake != awake:
            action = "wake" if awake else "sleep"
            if awake:
                self._ensure_feasible_wake(binding, snapshot.bindings)
            self._apply_runtime_power_action(binding, action=action)
            updated_binding = replace(binding, awake=awake, hidden=False)
            updated = [
                updated_binding if item.serve_id == serve_id else item
                for item in snapshot.bindings
            ]
            version = self._store.save(updated, expected_version=snapshot.version)
            actions.append({"action": action, "serve_id": serve_id})

        return {
            "serve_id": serve_id,
            "awake": awake,
            "version": version,
            "actions": actions,
            "binding": self._binding_dict(updated_binding),
        }


    @serialized_operation("put_model_routable")
    def put_model_routable(self, model: str, *, hidden_pods: list[str]) -> dict:
        self._registry.model(model)
        snapshot = self._store.load()
        model_bindings = [binding for binding in snapshot.bindings if binding.model == model]
        model_serve_ids = {binding.serve_id for binding in model_bindings}
        requested_hidden = set(hidden_pods)
        unknown = requested_hidden - model_serve_ids
        if unknown:
            raise ValueError(f"unknown pods for {model}: {sorted(unknown)}")

        self._update_desired(
            {
                binding.binding_id: {
                    "hidden": binding.serve_id in requested_hidden
                }
                for binding in model_bindings
            },
            updated_by="service-manager-api",
            reason="model_routable_request",
        )

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


    @serialized_operation("defrag")
    def defrag(self, *, tp_size: int) -> dict:
        snapshot = self._store.load()
        allocator = SlotAllocator(self._registry.topology(), snapshot.bindings)
        migrations = allocator.plan_defrag(tp_size)
        if migrations is None:
            raise DefragUnavailable("no_feasible_defrag")
        self._set_defrag_desired(snapshot.bindings, migrations)
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


    def audit(self) -> dict:
        if self._k8s_client is None:
            raise ValueError("k8s_client is required for audit")
        prober = None
        if self._vllm_ops is not None and hasattr(self._vllm_ops, "is_sleeping"):
            prober = _VllmPodProber(self._vllm_ops)
        result = audit_state(self._store, self._k8s_client, prober=prober)
        issues = list(result.issues)
        if self._fleet_store is not None:
            issues.extend(self._fleet_mismatches())
        return {
            "healthy": not issues,
            "version": result.version,
            "issues": issues,
        }

    @serialized_operation("reconcile")
    def reconcile(self, *, drop_missing: bool = False) -> dict:
        return self._reconcile_unlocked(drop_missing=drop_missing)

    def _reconcile_unlocked(self, *, drop_missing: bool = False) -> dict:
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
            drop_missing=drop_missing,
        )
        self._sync_observed(result.observations)
        return {
            "version": result.version,
            "warnings": result.warnings,
            "bindings": [self._binding_dict(binding) for binding in result.bindings],
        }

    def start_fleet_repair(
        self,
        *,
        awake_binding_ids: list[str] | None = None,
        recovered_from: list[str] | None = None,
    ) -> dict:
        if self._operation_coordinator is None or self._fleet_repair is None:
            raise ValueError("fleet repair runtime is not configured")
        if self._safety_gate is None:
            raise ValueError("fleet repair safety gate is not configured")
        self._safety_gate.assert_controller_observe()
        snapshot = self._store.load()
        targets = (
            sorted(set(awake_binding_ids))
            if awake_binding_ids is not None
            else self._desired_awake_binding_ids(snapshot.bindings)
        )

        def run(operation) -> None:
            for stale_operation_id in recovered_from or []:
                operation.supersede(stale_operation_id)
            self._fleet_repair.run(
                operation,
                awake_binding_ids=targets,
                reconcile=lambda strict: self._reconcile_unlocked(
                    drop_missing=strict
                ),
                set_binding_power=self._set_binding_power_by_id_unlocked,
                audit=self.audit,
            )

        operation_request = {"awake_binding_ids": targets}
        if recovered_from:
            operation_request["recovered_from"] = recovered_from
        operation_id = self._operation_coordinator.submit(
            "fleet_repair",
            run,
            request=operation_request,
        )
        response = {
            "operation_id": operation_id,
            "status": "accepted",
            "awake_binding_ids": targets,
        }
        if recovered_from:
            response["recovered_from"] = recovered_from
        return response

    def recover_stale_fleet_repairs(self) -> dict | None:
        if self._operation_coordinator is None:
            return None
        if self._operation_coordinator.active_operation() is not None:
            return None
        stale = self._operation_coordinator.stale_running_operations(
            kind="fleet_repair"
        )
        if not stale:
            return None
        newest = stale[0]
        request = newest.get("request") or {}
        return self.start_fleet_repair(
            awake_binding_ids=[
                str(item) for item in request.get("awake_binding_ids", [])
            ],
            recovered_from=[
                str(record["operation_id"]) for record in stale
            ],
        )

    def detect_fleet_drift(self) -> list[dict]:
        if self._runtime_ops is None or self._fleet_store is None:
            return []
        deployments = {
            item.binding_id: item
            for item in self._runtime_ops.list_model_deployments()
        }
        snapshots: dict[str, list[K8sPodSnapshot]] = {}
        for pod in self._runtime_ops.list_pod_snapshots():
            binding = _binding_from_snapshot(pod)
            snapshots.setdefault(binding.binding_id, []).append(pod)
        admitted_startups = {
            pod.binding_id
            for pod in self._runtime_ops.list_admitted_startup_pods()
        }
        observed = {
            item.binding_id: item
            for item in self._fleet_store.load_observed().bindings
        }
        issues: list[dict] = []
        for binding_id, desired in {
            item.binding_id: item
            for item in self._fleet_store.load_desired().bindings
            if item.lifecycle == "resident"
        }.items():
            if binding_id not in deployments:
                issues.append(
                    {"code": "deployment_missing", "binding_id": binding_id}
                )
                continue
            pods = snapshots.get(binding_id, [])
            if len(pods) != 1:
                if binding_id in admitted_startups:
                    continue
                issues.append(
                    {
                        "code": "pod_cardinality",
                        "binding_id": binding_id,
                        "count": len(pods),
                    }
                )
                continue
            pod = pods[0]
            if not pod.ready and not pod.annotations.get(
                "tre.aibrix.io/startup-admitted-uid"
            ):
                issues.append(
                    {"code": "pod_not_ready", "binding_id": binding_id}
                )
            previous = observed.get(binding_id)
            if (
                previous is not None
                and previous.pod_uid
                and pod.pod_uid
                and previous.pod_uid != pod.pod_uid
                and not pod.annotations.get("tre.aibrix.io/startup-admitted-uid")
            ):
                issues.append(
                    {
                        "code": "pod_uid_changed_without_admission",
                        "binding_id": binding_id,
                        "old_uid": previous.pod_uid,
                        "new_uid": pod.pod_uid,
                    }
                )
        return issues

    def admit_startup(self, *, pod_name: str, pod_uid: str) -> dict:
        """Authorize a Pod to start vLLM only after its GPU slot is safe."""
        if (
            self._operation_coordinator is None
            or self._runtime_ops is None
            or self._vllm_ops is None
            or self._gpu_leases is None
            or self._fleet_store is None
            or self._safety_gate is None
        ):
            raise ValueError("startup admission runtime is not configured")
        pod = self._runtime_ops.get_startup_pod(pod_name)
        if pod.uid != pod_uid:
            raise ValueError(f"Pod UID changed for {pod_name}")
        admitted_uid = pod.annotations.get("tre.aibrix.io/startup-admitted-uid")
        if admitted_uid == pod_uid:
            return {
                "status": "admitted",
                "binding_id": pod.binding_id,
                "operation_id": pod.annotations.get(
                    "tre.aibrix.io/startup-operation-id"
                ),
                "idempotent": True,
            }

        # A fleet repair already owns the global writer fence and has acquired
        # the target lease before scaling the Deployment. Reuse that prepared
        # phase instead of deadlocking the init gate on a second writer.
        active = self._operation_coordinator.active_operation(kind="fleet_repair")
        if active is not None:
            details = active.get("details") or {}
            if (
                active.get("phase") == "starting_binding"
                and details.get("binding_id") == pod.binding_id
                and self._lease_matches(pod.binding_id, phase="starting")
            ):
                self._assert_startup_overlaps_sleeping(pod)
                self._runtime_ops.admit_startup_pod(
                    pod.name,
                    pod_uid=pod.uid,
                    suspended_binding_ids=[],
                    operation_id=str(active["operation_id"]),
                )
                return {
                    "status": "admitted",
                    "binding_id": pod.binding_id,
                    "operation_id": active["operation_id"],
                    "prepared_by_fleet_repair": True,
                }
            raise OperationBusy(
                f"{active.get('owner')}:{active.get('fencing_token')}"
            )

        request = {"pod_name": pod_name, "pod_uid": pod_uid}
        with self._operation_coordinator.operation(
            "startup_admit", request=request
        ) as operation:
            operation.advance("validating_startup", details={"binding_id": pod.binding_id})
            self._safety_gate.assert_no_pressure()
            desired = self._desired_binding(pod.binding_id)
            if desired.lifecycle != "resident":
                raise ValueError(
                    f"startup denied for non-resident desired binding {pod.binding_id}"
                )

            suspended: list[str] = []
            legacy = self._store.load()
            updated = {binding.binding_id: binding for binding in legacy.bindings}
            target_gpus = set(pod.gpu_ids)
            for snapshot in self._runtime_ops.list_startup_resident_snapshots():
                if snapshot.name == pod.name or snapshot.node != pod.node:
                    continue
                binding = _binding_from_snapshot(snapshot)
                if not target_gpus.intersection(binding.slot.gpu_ids):
                    continue
                if not snapshot.pod_ip or not snapshot.ready:
                    raise ValueError(
                        f"overlapping resident {binding.binding_id} is not Ready"
                    )
                physical_sleeping = self._vllm_ops.is_sleeping(
                    snapshot.pod_ip, port=8000
                )
                if physical_sleeping is None:
                    raise ValueError(
                        f"cannot verify overlapping resident {binding.binding_id}"
                    )
                wanted = self._desired_binding(binding.binding_id)
                if not physical_sleeping:
                    self._runtime_ops.write_binding_annotations(
                        binding, state=POD_STATE_HIDDEN
                    )
                    self._apply_runtime_power_action(binding, action="sleep")
                    updated[binding.binding_id] = replace(
                        binding, awake=False, hidden=False
                    )
                if wanted.power == "awake" and binding.binding_id != pod.binding_id:
                    suspended.append(binding.binding_id)

            if list(updated.values()) != legacy.bindings:
                self._store.save(
                    list(updated.values()), expected_version=legacy.version
                )
            planned = Binding(
                pod.name,
                pod.model,
                Slot(pod.node, pod.gpu_ids),
                awake=False,
                hidden=True,
            )
            self._gpu_leases.acquire(planned, phase="starting")
            self._runtime_ops.admit_startup_pod(
                pod.name,
                pod_uid=pod.uid,
                suspended_binding_ids=suspended,
                operation_id=operation.operation_id,
            )
            operation.advance(
                "startup_admitted",
                details={
                    "binding_id": pod.binding_id,
                    "suspended_binding_ids": suspended,
                },
            )
            return {
                "status": "admitted",
                "binding_id": pod.binding_id,
                "operation_id": operation.operation_id,
                "suspended_binding_ids": suspended,
            }

    def converge_startups(self) -> dict:
        """Converge admitted Pods after vLLM becomes reachable."""
        if self._runtime_ops is None or self._vllm_ops is None:
            return {"converged": [], "pending": []}
        converged: list[str] = []
        pending: list[str] = []
        for snapshot in self._runtime_ops.list_startup_resident_snapshots():
            admitted_uid = snapshot.annotations.get(
                "tre.aibrix.io/startup-admitted-uid"
            )
            if not admitted_uid or admitted_uid != snapshot.pod_uid:
                continue
            if not snapshot.pod_ip:
                pending.append(snapshot.name)
                continue
            physical_sleeping = self._vllm_ops.is_sleeping(
                snapshot.pod_ip, port=8000
            )
            if physical_sleeping is None:
                pending.append(snapshot.name)
                continue
            try:
                self._converge_startup(snapshot, physical_sleeping)
            except OperationBusy:
                pending.append(snapshot.name)
                continue
            converged.append(snapshot.name)
        return {"converged": converged, "pending": pending}

    @serialized_operation("startup_converge")
    def _converge_startup(
        self, snapshot: K8sPodSnapshot, physical_sleeping: bool
    ) -> None:
        if self._fleet_store is None or self._gpu_leases is None:
            raise ValueError("startup convergence state is not configured")
        binding = _binding_from_snapshot(snapshot)
        desired = self._desired_binding(binding.binding_id)
        if desired.power == "sleeping":
            if not physical_sleeping:
                self._apply_runtime_power_action(binding, action="sleep")
            else:
                self._runtime_ops.write_binding_annotations(
                    binding, state=POD_STATE_SLEEPING
                )
                self._gpu_leases.release(binding)
            binding = replace(binding, awake=False, hidden=False)
        else:
            if physical_sleeping:
                self._apply_runtime_power_action(binding, action="wake")
            else:
                self._runtime_ops.write_binding_annotations(
                    binding,
                    state=POD_STATE_HIDDEN if desired.hidden else POD_STATE_AWAKE,
                )
                self._gpu_leases.acquire(binding, phase="awake")
            binding = replace(
                binding, awake=True, hidden=desired.hidden
            )

        legacy = self._store.load()
        by_id = {item.binding_id: item for item in legacy.bindings}
        by_id[binding.binding_id] = binding
        self._store.save(list(by_id.values()), expected_version=legacy.version)

        suspended_raw = snapshot.annotations.get(
            "tre.aibrix.io/startup-suspended-bindings", "[]"
        )
        suspended = [str(item) for item in json.loads(suspended_raw)]
        if desired.power == "awake" and suspended:
            raise ValueError(
                f"awake startup {binding.binding_id} conflicts with suspended {suspended}"
            )
        for binding_id in suspended:
            wanted = self._desired_binding(binding_id)
            if wanted.lifecycle == "resident" and wanted.power == "awake":
                self._restore_desired_awake(binding_id)
        self._runtime_ops.clear_startup_admission(snapshot.name)
        self._reconcile_unlocked(drop_missing=False)

    def _restore_desired_awake(self, binding_id: str) -> None:
        snapshot = self._store.load()
        binding = next(
            (item for item in snapshot.bindings if item.binding_id == binding_id),
            None,
        )
        if binding is None or binding.awake:
            return
        self._ensure_feasible_wake(binding, snapshot.bindings)
        self._apply_runtime_power_action(binding, action="wake")
        updated = [
            replace(item, awake=True, hidden=False)
            if item.binding_id == binding_id
            else item
            for item in snapshot.bindings
        ]
        self._store.save(updated, expected_version=snapshot.version)

    def _desired_binding(self, binding_id: str) -> DesiredBinding:
        if self._fleet_store is None:
            raise ValueError("desired fleet state is not configured")
        matches = [
            item
            for item in self._fleet_store.load_desired().bindings
            if item.binding_id == binding_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"stable binding {binding_id} resolved to {len(matches)} desired records"
            )
        return matches[0]

    def _lease_matches(self, binding_id: str, *, phase: str) -> bool:
        if self._gpu_leases is None:
            return False
        return any(
            lease.binding_id == binding_id and lease.phase == phase
            for lease in self._gpu_leases.load()
        )

    def _assert_startup_overlaps_sleeping(self, pod: StartupPodRecord) -> None:
        target_gpus = set(pod.gpu_ids)
        for snapshot in self._runtime_ops.list_pod_snapshots():
            if snapshot.name == pod.name or snapshot.node != pod.node:
                continue
            binding = _binding_from_snapshot(snapshot)
            if not target_gpus.intersection(binding.slot.gpu_ids):
                continue
            if not snapshot.pod_ip or self._vllm_ops.is_sleeping(
                snapshot.pod_ip, port=8000
            ) is not True:
                raise ValueError(
                    f"overlapping resident {binding.binding_id} is not physically sleeping"
                )

    def _set_binding_power_by_id_unlocked(
        self, binding_id: str, awake: bool
    ) -> dict:
        snapshot = self._store.load()
        matches = [
            binding
            for binding in snapshot.bindings
            if binding.binding_id == binding_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"stable binding {binding_id} resolved to {len(matches)} instances"
            )
        return self._put_binding_power_unlocked(
            matches[0].serve_id, awake=awake
        )

    def list_operations(self, *, limit: int = 100) -> list[dict]:
        if self._operation_coordinator is None:
            return []
        return self._operation_coordinator.list_operations(limit=limit)

    def get_operation(self, operation_id: str) -> dict:
        if self._operation_coordinator is None:
            raise KeyError(operation_id)
        operation = self._operation_coordinator.get_operation(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        return operation

    def get_fleet_state(self) -> dict:
        if self._fleet_store is None:
            return {
                "desired_version": 0,
                "observed_version": 0,
                "desired": [],
                "observed": [],
                "mismatches": [],
            }
        desired = self._fleet_store.load_desired()
        observed = self._fleet_store.load_observed()
        result = {
            "desired_version": desired.version,
            "observed_version": observed.version,
            "desired": [asdict(binding) for binding in desired.bindings],
            "observed": [asdict(binding) for binding in observed.bindings],
            "mismatches": self._fleet_mismatches(
                desired_bindings=desired.bindings,
                observed_bindings=observed.bindings,
            ),
        }
        if self._gpu_leases is not None:
            result["gpu_leases"] = [
                asdict(lease) for lease in self._gpu_leases.load()
            ]
        return result

    def _desired_awake_binding_ids(
        self, legacy_bindings: list[Binding]
    ) -> list[str]:
        if self._fleet_store is None:
            return sorted(
                binding.binding_id for binding in legacy_bindings if binding.awake
            )
        return sorted(
            binding.binding_id
            for binding in self._fleet_store.load_desired().bindings
            if binding.lifecycle == "resident" and binding.power == "awake"
        )

    def _update_desired(
        self,
        updates: dict[str, dict[str, object]],
        *,
        updated_by: str,
        reason: str,
    ) -> None:
        if self._fleet_store is None or not updates:
            return
        snapshot = self._fleet_store.load_desired()
        by_id = {binding.binding_id: binding for binding in snapshot.bindings}
        unknown = sorted(set(updates) - set(by_id))
        if unknown:
            raise ValueError(f"desired state missing stable binding(s): {unknown}")
        changed = False
        for binding_id, fields in updates.items():
            current = by_id[binding_id]
            updated = current.with_intent(
                power=fields.get("power"),
                hidden=fields.get("hidden"),
                lifecycle=fields.get("lifecycle"),
                updated_by=updated_by,
                reason=reason,
            )
            by_id[binding_id] = updated
            changed = changed or updated is not current
        if changed:
            self._fleet_store.save_desired(
                list(by_id.values()), expected_version=snapshot.version
            )

    def _set_model_desired_target(
        self,
        *,
        model: str,
        target_bindings: list[Binding],
        reason: str,
    ) -> None:
        if self._fleet_store is None:
            return
        snapshot = self._fleet_store.load_desired()
        by_id = {binding.binding_id: binding for binding in snapshot.bindings}
        target_by_id = {
            binding.binding_id: binding for binding in target_bindings
        }
        changed = False
        for binding_id, desired in list(by_id.items()):
            if desired.model != model:
                continue
            updated = desired.with_intent(
                lifecycle="resident",
                power="awake" if binding_id in target_by_id else "sleeping",
                hidden=False,
                updated_by="service-manager-api",
                reason=reason,
            )
            by_id[binding_id] = updated
            changed = changed or updated is not desired
        for binding_id, planned in target_by_id.items():
            if binding_id in by_id:
                continue
            by_id[binding_id] = DesiredBinding.from_binding(
                planned,
                updated_by="service-manager-api",
                reason=reason,
            )
            changed = True
        if changed:
            self._fleet_store.save_desired(
                list(by_id.values()), expected_version=snapshot.version
            )

    def _set_defrag_desired(
        self, bindings: list[Binding], migrations: list[Migration]
    ) -> None:
        if self._fleet_store is None or not migrations:
            return
        desired_snapshot = self._fleet_store.load_desired()
        by_id = {
            binding.binding_id: binding
            for binding in desired_snapshot.bindings
        }
        actual_by_serve = {binding.serve_id: binding for binding in bindings}
        for migration in migrations:
            actual = actual_by_serve[migration.serve_id]
            old_id = actual.binding_id
            old = by_id.get(old_id)
            if old is None:
                raise ValueError(f"desired state missing stable binding: {old_id}")
            by_id[old_id] = old.with_intent(
                lifecycle="absent",
                power="sleeping",
                hidden=True,
                updated_by="service-manager-api",
                reason="defrag_source_removed",
            )
            planned = Binding(
                serve_id=actual.serve_id,
                model=actual.model,
                slot=migration.to_slot,
                awake=True,
            )
            existing = by_id.get(planned.binding_id)
            if existing is None:
                by_id[planned.binding_id] = DesiredBinding.from_binding(
                    planned,
                    updated_by="service-manager-api",
                    reason="defrag_destination",
                )
            else:
                by_id[planned.binding_id] = existing.with_intent(
                    lifecycle="resident",
                    power="awake",
                    hidden=False,
                    updated_by="service-manager-api",
                    reason="defrag_destination",
                )
        self._fleet_store.save_desired(
            list(by_id.values()), expected_version=desired_snapshot.version
        )

    def _sync_observed(self, observations) -> None:
        if self._fleet_store is None:
            return
        snapshot = self._fleet_store.load_observed()
        records = []
        for observation in observations:
            binding = observation.binding
            pod = observation.pod
            physical_power = (
                "unknown"
                if observation.physical_awake is None
                else "awake" if observation.physical_awake else "sleeping"
            )
            records.append(
                ObservedBinding(
                    binding_id=binding.binding_id,
                    model=binding.model,
                    node=binding.slot.node,
                    gpu_ids=binding.slot.gpu_ids,
                    pod_name=binding.serve_id,
                    pod_uid=pod.pod_uid,
                    pod_ip=pod.pod_ip,
                    phase=pod.phase,
                    ready=pod.ready,
                    restart_count=pod.restart_count,
                    physical_power=physical_power,
                    routable=pod.routable,
                    hidden=binding.hidden,
                    error=(
                        "physical_probe_unreachable"
                        if observation.physical_awake is None
                        else None
                    ),
                )
            )
        records.sort(key=lambda item: item.binding_id)
        if records != snapshot.bindings:
            self._fleet_store.save_observed(
                records, expected_version=snapshot.version
            )

    def _fleet_mismatches(
        self,
        *,
        desired_bindings: list[DesiredBinding] | None = None,
        observed_bindings: list[ObservedBinding] | None = None,
    ) -> list[dict]:
        if self._fleet_store is None:
            return []
        desired_bindings = (
            self._fleet_store.load_desired().bindings
            if desired_bindings is None
            else desired_bindings
        )
        observed_bindings = (
            self._fleet_store.load_observed().bindings
            if observed_bindings is None
            else observed_bindings
        )
        desired = {binding.binding_id: binding for binding in desired_bindings}
        observed = {binding.binding_id: binding for binding in observed_bindings}
        issues: list[dict] = []
        for binding_id in sorted(set(desired) | set(observed)):
            wanted = desired.get(binding_id)
            actual = observed.get(binding_id)
            if wanted is None:
                issues.append({"code": "observed_without_desired", "binding_id": binding_id})
                continue
            if actual is None:
                if wanted.lifecycle == "resident":
                    issues.append({"code": "desired_binding_missing", "binding_id": binding_id})
                continue
            if wanted.lifecycle == "absent":
                issues.append({"code": "desired_absent_but_observed", "binding_id": binding_id})
                continue
            if actual.physical_power != wanted.power:
                issues.append(
                    {
                        "code": "desired_power_mismatch",
                        "binding_id": binding_id,
                        "desired_power": wanted.power,
                        "observed_power": actual.physical_power,
                    }
                )
            if actual.hidden != wanted.hidden:
                issues.append(
                    {
                        "code": "desired_hidden_mismatch",
                        "binding_id": binding_id,
                        "desired_hidden": wanted.hidden,
                        "observed_hidden": actual.hidden,
                    }
                )
        if self._gpu_leases is not None:
            leases = {
                lease.binding_id: lease for lease in self._gpu_leases.load()
            }
            for binding_id, actual in observed.items():
                has_lease = binding_id in leases
                if actual.physical_power == "awake" and not has_lease:
                    issues.append(
                        {
                            "code": "awake_without_gpu_lease",
                            "binding_id": binding_id,
                        }
                    )
                if actual.physical_power == "sleeping" and has_lease:
                    issues.append(
                        {
                            "code": "gpu_lease_without_awake",
                            "binding_id": binding_id,
                        }
                    )
        return issues

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
            if self._gpu_leases is not None:
                self._gpu_leases.acquire(binding, phase="waking")
            result = self._vllm_ops.wake_up(snapshot.pod_ip, port=8000)
            state = POD_STATE_AWAKE
        else:
            raise ValueError(f"unknown runtime action: {action}")

        if not bool(getattr(result, "success", False)):
            message = getattr(result, "message", "") or "operation failed"
            raise ValueError(f"vLLM {action} failed for {binding.serve_id}: {message}")
        if hasattr(self._vllm_ops, "is_sleeping"):
            physical = self._vllm_ops.is_sleeping(snapshot.pod_ip, port=8000)
            expected_sleeping = action == "sleep"
            if physical is not expected_sleeping:
                raise ValueError(
                    f"vLLM {action} did not physically converge for {binding.serve_id}"
                )
        self._runtime_ops.write_binding_annotations(binding, state=state)
        if self._gpu_leases is not None:
            if action == "sleep":
                self._gpu_leases.release(binding)
            else:
                self._gpu_leases.acquire(binding, phase="awake")

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
        planned = Binding("startup", model, slot, awake=False)
        if self._gpu_leases is not None:
            self._gpu_leases.acquire(planned, phase="starting")
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
        if self._gpu_leases is not None:
            self._gpu_leases.acquire(binding, phase="awake")
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
        planned = Binding(
            "defrag-startup", binding.model, migration.to_slot, awake=False
        )
        if self._gpu_leases is not None:
            self._gpu_leases.acquire(planned, phase="starting")
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
        if self._gpu_leases is not None:
            self._gpu_leases.acquire(moved, phase="awake")
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
            "binding_id": binding.binding_id,
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


class BindingPowerRequest(BaseModel):
    awake: bool


class DefragRequest(BaseModel):
    tp_size: int




class RoutableRequest(BaseModel):
    hidden_pods: list[str]


class ReconcileRequest(BaseModel):
    drop_missing: bool = False


class FleetRepairRequest(BaseModel):
    awake_binding_ids: list[str] | None = None


class StartupAdmissionRequest(BaseModel):
    pod_name: str
    pod_uid: str

def create_app(service: ServiceManagerV2) -> FastAPI:
    app = FastAPI()
    app.include_router(create_v1_compat_router(service))

    @app.exception_handler(OperationBusy)
    async def operation_busy_handler(
        _request: Request, exc: OperationBusy
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "current_writer": exc.current_writer},
        )

    @app.exception_handler(StateFenceError)
    async def state_fence_handler(
        _request: Request, exc: StateFenceError
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ControllerNotPaused)
    async def controller_not_paused_handler(
        _request: Request, exc: ControllerNotPaused
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(NodePressureActive)
    async def node_pressure_handler(
        _request: Request, exc: NodePressureActive
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(GpuLeaseConflict)
    async def gpu_lease_conflict_handler(
        _request: Request, exc: GpuLeaseConflict
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}


    @app.get("/v2/audit")
    def audit() -> dict:
        try:
            return service.audit()
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @app.post("/v2/reconcile")
    def reconcile(request: ReconcileRequest | None = None) -> dict:
        try:
            return service.reconcile(
                drop_missing=request.drop_missing if request is not None else False
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @app.post("/v2/fleet/repair", status_code=202)
    def start_fleet_repair(request: FleetRepairRequest) -> dict:
        try:
            return service.start_fleet_repair(
                awake_binding_ids=request.awake_binding_ids
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v2/startup/admit")
    def admit_startup(request: StartupAdmissionRequest) -> dict:
        try:
            return service.admit_startup(
                pod_name=request.pod_name, pod_uid=request.pod_uid
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v2/startup/converge")
    def converge_startups() -> dict:
        return service.converge_startups()

    @app.get("/v2/state")
    def get_state() -> dict:
        return service.get_state()


    @app.get("/v2/fleet/state")
    def get_fleet_state() -> dict:
        return service.get_fleet_state()


    @app.get("/v2/operations")
    def list_operations(limit: int = 100) -> dict:
        if limit < 1 or limit > 1000:
            raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
        return {"operations": service.list_operations(limit=limit)}


    @app.get("/v2/operations/{operation_id}")
    def get_operation(operation_id: str) -> dict:
        try:
            return service.get_operation(operation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="operation not found") from exc

    @app.get("/v2/supervisor")
    def get_supervisor() -> dict:
        return service.get_supervisor_state()


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

    @app.put("/v2/bindings/{serve_id}/power")
    def put_binding_power(serve_id: str, request: BindingPowerRequest) -> dict:
        try:
            return service.put_binding_power(serve_id, awake=request.awake)
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


def _binding_from_snapshot(snapshot: K8sPodSnapshot) -> Binding:
    gpu_text = snapshot.annotations.get("tre.aibrix.io/gpu-ids")
    if not gpu_text:
        raise ValueError(f"Pod {snapshot.name} has no stable GPU annotation")
    state = snapshot.annotations.get("tre.aibrix.io/state", POD_STATE_AWAKE)
    return Binding(
        serve_id=snapshot.name,
        model=snapshot.model,
        slot=Slot(
            snapshot.node,
            tuple(int(part) for part in str(gpu_text).split(",")),
        ),
        awake=state != POD_STATE_SLEEPING,
        hidden=state == POD_STATE_HIDDEN,
    )


def _gpu_uuid(node: NodeSpec | None, gpu_id: int) -> str | None:
    if node is None:
        return None
    if gpu_id < 0 or gpu_id >= len(node.gpu_uuids):
        return None
    return node.gpu_uuids[gpu_id]

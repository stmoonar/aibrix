from __future__ import annotations

import time
from typing import Callable, Protocol

from tre_sm.allocator.slots import Binding, Slot
from tre_sm.allocator.topology import GPU_IDS_ANNOTATION, K8sPodSnapshot
from tre_sm.ops.k8s_ops import ModelDeploymentRecord
from tre_sm.state.operations import OperationHandle
from tre_sm.state.reconcile import (
    POD_STATE_AWAKE,
    POD_STATE_HIDDEN,
    POD_STATE_SLEEPING,
)
from tre_sm.state.safety import ClusterSafetyGate
from tre_sm.state.gpu_leases import GpuLeaseStore


class FleetRuntimeOps(Protocol):
    def list_model_deployments(self) -> list[ModelDeploymentRecord]: ...
    def list_pod_snapshots(self, *, model: str | None = None) -> list[K8sPodSnapshot]: ...
    def write_binding_annotations(self, binding: Binding, *, state: str) -> None: ...
    def scale_model_deployment(self, name: str, *, replicas: int) -> None: ...
    def wait_deployment_pods_deleted(self, deployment_name: str): ...
    def wait_pod_ready(self, serve_id: str) -> K8sPodSnapshot: ...


class FleetVllmOps(Protocol):
    def sleep(self, pod_ip: str, *, port: int | None = None): ...
    def is_sleeping(self, pod_ip: str, *, port: int | None = None) -> bool | None: ...
    def wait_until_ready(self, pod_ip: str, *, port: int | None = None): ...


class FleetRepairExecutor:
    def __init__(
        self,
        *,
        runtime_ops: FleetRuntimeOps,
        vllm_ops: FleetVllmOps,
        safety_gate: ClusterSafetyGate,
        gpu_leases: GpuLeaseStore | None = None,
        physical_timeout_s: float = 120.0,
        poll_interval_s: float = 2.0,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._runtime = runtime_ops
        self._vllm = vllm_ops
        self._safety = safety_gate
        self._gpu_leases = gpu_leases
        self._physical_timeout_s = physical_timeout_s
        self._poll_interval_s = poll_interval_s
        self._monotonic = monotonic
        self._sleep = sleep

    def run(
        self,
        operation: OperationHandle,
        *,
        awake_binding_ids: list[str],
        reconcile: Callable[[bool], dict],
        set_binding_power: Callable[[str, bool], dict],
        audit: Callable[[], dict],
    ) -> None:
        deployments = self._runtime.list_model_deployments()
        if not deployments:
            raise RuntimeError("fleet repair found no managed model Deployments")
        by_id = {deployment.binding_id: deployment for deployment in deployments}
        if len(by_id) != len(deployments):
            raise RuntimeError("duplicate stable binding_id in managed Deployments")
        unknown = sorted(set(awake_binding_ids) - set(by_id))
        if unknown:
            raise ValueError(f"unknown awake binding_id(s): {unknown}")
        self._validate_awake_targets(awake_binding_ids, by_id)
        operation.advance(
            "inventory",
            details={
                "deployments": len(deployments),
                "awake_binding_ids": awake_binding_ids,
            },
        )
        self._safety.wait_until_healthy(operation)

        repair_ids = self._quarantine_and_sleep_residents(operation, by_id)
        operation.advance(
            "residents_quarantined",
            details={"repair_binding_ids": sorted(repair_ids)},
        )

        # Remove every unhealthy/unknown instance first. This guarantees that
        # sequential cold starts never encounter an overlapping unknown Pod.
        for binding_id in sorted(repair_ids):
            operation.assert_active()
            deployment = by_id[binding_id]
            self._runtime.scale_model_deployment(deployment.name, replicas=0)
        for binding_id in sorted(repair_ids):
            self._runtime.wait_deployment_pods_deleted(by_id[binding_id].name)

        completed: list[str] = []
        for binding_id in sorted(repair_ids):
            operation.assert_active()
            self._safety.wait_until_healthy(operation)
            deployment = by_id[binding_id]
            self._assert_overlapping_residents_sleeping(deployment)
            operation.advance(
                "starting_binding",
                details={
                    "binding_id": binding_id,
                    "completed_binding_ids": completed,
                },
            )
            planned = Binding(
                serve_id=deployment.name,
                model=deployment.model,
                slot=Slot(deployment.node, deployment.gpu_ids),
                awake=False,
            )
            if self._gpu_leases is not None:
                self._gpu_leases.acquire(planned, phase="starting")
            self._runtime.scale_model_deployment(deployment.name, replicas=1)
            pod = self._runtime.wait_pod_ready(deployment.name)
            if not pod.pod_ip:
                raise RuntimeError(f"new Pod for {binding_id} has no IP")
            ready = self._vllm.wait_until_ready(pod.pod_ip, port=8000)
            if not bool(getattr(ready, "success", False)):
                raise RuntimeError(
                    f"vLLM readiness failed for {binding_id}: "
                    f"{getattr(ready, 'message', '')}"
                )
            binding = _binding_from_snapshot(pod)
            self._runtime.write_binding_annotations(binding, state=POD_STATE_HIDDEN)
            self._sleep_binding(binding, pod.pod_ip)
            completed.append(binding_id)
            operation.advance(
                "binding_resident_sleeping",
                details={
                    "binding_id": binding_id,
                    "completed_binding_ids": completed,
                },
            )

        operation.advance("reconciling_resident_pool")
        reconcile(True)
        for binding_id in awake_binding_ids:
            operation.assert_active()
            self._safety.wait_until_healthy(operation)
            operation.advance("waking_target", details={"binding_id": binding_id})
            set_binding_power(binding_id, True)

        operation.advance("final_reconcile")
        reconcile(True)
        result = audit()
        if not result.get("healthy"):
            raise RuntimeError(f"fleet repair final audit failed: {result.get('issues')}")
        operation.advance(
            "verified",
            details={
                "deployments": len(deployments),
                "repaired": len(repair_ids),
                "awake_binding_ids": awake_binding_ids,
                "audit_version": result.get("version"),
            },
        )

    def _quarantine_and_sleep_residents(
        self,
        operation: OperationHandle,
        deployments: dict[str, ModelDeploymentRecord],
    ) -> set[str]:
        snapshots_by_id: dict[str, list[K8sPodSnapshot]] = {}
        for snapshot in self._runtime.list_pod_snapshots():
            try:
                binding_id = _binding_from_snapshot(snapshot).binding_id
            except (KeyError, ValueError):
                continue
            snapshots_by_id.setdefault(binding_id, []).append(snapshot)

        repair_ids: set[str] = set()
        for binding_id, deployment in deployments.items():
            operation.assert_active()
            snapshots = snapshots_by_id.get(binding_id, [])
            if len(snapshots) != 1:
                repair_ids.add(binding_id)
                continue
            snapshot = snapshots[0]
            binding = _binding_from_snapshot(snapshot)
            self._runtime.write_binding_annotations(binding, state=POD_STATE_HIDDEN)
            if not snapshot.ready or not snapshot.pod_ip:
                repair_ids.add(binding_id)
                continue
            sleeping = self._vllm.is_sleeping(snapshot.pod_ip, port=8000)
            if sleeping is None:
                repair_ids.add(binding_id)
                continue
            if not sleeping:
                self._sleep_binding(binding, snapshot.pod_ip)
            else:
                self._runtime.write_binding_annotations(
                    binding, state=POD_STATE_SLEEPING
                )
        return repair_ids

    def _sleep_binding(self, binding: Binding, pod_ip: str) -> None:
        result = self._vllm.sleep(pod_ip, port=8000)
        if not bool(getattr(result, "success", False)):
            raise RuntimeError(
                f"vLLM sleep failed for {binding.binding_id}: "
                f"{getattr(result, 'message', '')}"
            )
        self._wait_physical(pod_ip, sleeping=True)
        self._runtime.write_binding_annotations(binding, state=POD_STATE_SLEEPING)
        if self._gpu_leases is not None:
            self._gpu_leases.release(binding)

    def _wait_physical(self, pod_ip: str, *, sleeping: bool) -> None:
        deadline = self._monotonic() + self._physical_timeout_s
        while self._monotonic() < deadline:
            if self._vllm.is_sleeping(pod_ip, port=8000) is sleeping:
                return
            self._sleep(self._poll_interval_s)
        raise TimeoutError(
            f"vLLM {pod_ip} did not converge to sleeping={sleeping}"
        )

    def _assert_overlapping_residents_sleeping(
        self, target: ModelDeploymentRecord
    ) -> None:
        target_gpus = set(target.gpu_ids)
        for snapshot in self._runtime.list_pod_snapshots():
            binding = _binding_from_snapshot(snapshot)
            if binding.slot.node != target.node:
                continue
            if not target_gpus.intersection(binding.slot.gpu_ids):
                continue
            if not snapshot.ready or not snapshot.pod_ip:
                raise RuntimeError(
                    f"overlapping resident {binding.binding_id} is not Ready"
                )
            if self._vllm.is_sleeping(snapshot.pod_ip, port=8000) is not True:
                raise RuntimeError(
                    f"overlapping resident {binding.binding_id} is not physically sleeping"
                )

    @staticmethod
    def _validate_awake_targets(
        awake_binding_ids: list[str],
        deployments: dict[str, ModelDeploymentRecord],
    ) -> None:
        occupied: dict[tuple[str, int], str] = {}
        for binding_id in awake_binding_ids:
            deployment = deployments[binding_id]
            for gpu_id in deployment.gpu_ids:
                key = (deployment.node, gpu_id)
                previous = occupied.get(key)
                if previous is not None:
                    raise ValueError(
                        f"awake targets overlap {key[0]}/{key[1]}: "
                        f"{previous} and {binding_id}"
                    )
                occupied[key] = binding_id


def _binding_from_snapshot(snapshot: K8sPodSnapshot) -> Binding:
    gpu_text = snapshot.annotations.get(GPU_IDS_ANNOTATION)
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

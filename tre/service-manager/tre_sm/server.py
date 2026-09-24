from __future__ import annotations

import os
from typing import Protocol

from fastapi import FastAPI

from tre_common.registry import ClusterTopology
from tre_common.registry import load_registry
from tre_sm.allocator.topology import K8sPodSnapshot, pod_records_from_snapshots
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.app import create_service_app
from tre_sm.gpu_truth import RedisGpuTruth
from tre_sm.ops.drain import DrainConfig, check_reissue_coupling
from tre_sm.ops.k8s_ops import K8sOps
from tre_sm.ops.vllm_ops import VllmOps
from tre_sm.state.async_ops import AsyncOpsConfig
from tre_sm.state.drain_markers import DrainMarkerStore
from tre_sm.state.reconcile import PodRecord
from tre_sm.state.operations import OperationCoordinator
from tre_sm.state.safety import ClusterSafetyGate
from tre_sm.state.fleet_store import FleetStateStore
from tre_sm.state.gpu_leases import GpuLeaseStore
from tre_sm.state.store import StateStore


class PodSnapshotOps(Protocol):
    def list_pod_snapshots(self) -> list[K8sPodSnapshot]: ...


class K8sPodClientFromOps:
    def __init__(self, topology: ClusterTopology, ops: PodSnapshotOps) -> None:
        self._topology = topology
        self._ops = ops

    def list_pods(self) -> list[PodRecord]:
        return pod_records_from_snapshots(self._topology, self._ops.list_pod_snapshots())


def create_app() -> FastAPI:
    try:
        import redis  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("redis package is required for the service-manager server") from exc

    registry = load_registry(os.environ.get("TRE_REGISTRY_PATH"))
    # Fail closed before touching Redis/Kubernetes: DRAIN without HIDE, or
    # the reissue sidecar without HIDE, is a startup error.
    drain_config = DrainConfig.from_env(os.environ)
    check_reissue_coupling(registry, drain_config)
    # TRE_SM_ASYNC_OPS (default off): 202 + operation id for target/power calls.
    async_config = AsyncOpsConfig.from_env(os.environ)
    redis_url = os.environ.get("TRE_REDIS_URL", "redis://aibrix-redis-master:6379/0")
    redis_client = redis.Redis.from_url(redis_url)
    k8s_ops = _create_k8s_ops(registry)
    operation_coordinator = OperationCoordinator(
        redis_client,
        owner=os.environ.get("HOSTNAME", "tre-v2-service-manager"),
        lease_ttl_ms=int(os.environ.get("TRE_SM_WRITER_LEASE_TTL_MS", "30000")),
    )
    legacy_store = StateStore(redis_client, require_fence=True)
    fleet_store = FleetStateStore(redis_client)
    gpu_leases = GpuLeaseStore(redis_client)
    starting_bindings = [
        Binding(
            pod.name,
            pod.model,
            Slot(pod.node, pod.gpu_ids),
            awake=False,
            hidden=True,
        )
        for pod in k8s_ops.list_admitted_startup_pods()
    ]
    with operation_coordinator.operation("bootstrap_fleet_state"):
        fleet_store.bootstrap(legacy_store.load().bindings)
        gpu_leases.rebuild_awake(
            legacy_store.load().bindings,
            starting_bindings=starting_bindings,
        )
    safety_gate = ClusterSafetyGate(
        redis_client,
        k8s_ops,
        clear_hysteresis_s=float(
            os.environ.get("TRE_SM_PRESSURE_CLEAR_HYSTERESIS_S", "60")
        ),
        pressure_timeout_s=float(
            os.environ.get("TRE_SM_PRESSURE_TIMEOUT_S", "3600")
        ),
    )
    return create_service_app(
        registry,
        legacy_store,
        k8s_client=K8sPodClientFromOps(registry.topology(), k8s_ops),
        runtime_ops=k8s_ops,
        vllm_ops=VllmOps(),
        gpu_truth=RedisGpuTruth(redis_client),
        create_max_used_mib=int(os.environ.get("TRE_CREATE_MAX_USED_MIB", "2500")),
        sleep_leak_used_mib=int(os.environ.get("TRE_SLEEP_LEAK_USED_MIB", "8192")),
        require_gpu_truth=gpu_truth_required_from_env(os.environ),
        operation_coordinator=operation_coordinator,
        safety_gate=safety_gate,
        fleet_store=fleet_store,
        gpu_leases=gpu_leases,
        supervisor_enabled=os.environ.get(
            "TRE_SM_SUPERVISOR_ENABLED", "true"
        ).lower() in {"1", "true", "yes"},
        supervisor_interval_s=float(
            os.environ.get("TRE_SM_SUPERVISOR_INTERVAL_S", "5")
        ),
        drain_config=drain_config,
        drain_markers=DrainMarkerStore(redis_client, require_fence=True),
        async_config=async_config,
    )


def _create_k8s_pod_client(topology: ClusterTopology) -> K8sPodClientFromOps:
    return K8sPodClientFromOps(topology, _create_k8s_ops())


def _create_k8s_ops(registry=None) -> K8sOps:
    try:
        from kubernetes import client, config  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("kubernetes package is required for service-manager reconcile") from exc

    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()

    namespace = os.environ.get("TRE_MODEL_NAMESPACE", os.environ.get("TARGET_NAMESPACE", "default"))
    return K8sOps(
        api=client.CoreV1Api(),
        apps_api=client.AppsV1Api(),
        route_api=client.CustomObjectsApi(),
        namespace=namespace,
        route_namespace=os.environ.get("TRE_ROUTE_NAMESPACE", "aibrix-system"),
        gateway_name=os.environ.get("TRE_GATEWAY_NAME", "aibrix-eg"),
        registry=registry,
    )


def gpu_truth_required_from_env(environ) -> bool:
    """Whether a missing GPU truth payload must block cold starts.

    Defaults to True (fail closed). Operators can set
    TRE_GPU_TRUTH_REQUIRED=false to fall back to the permissive behaviour if
    the gpu-truth DaemonSet is unavailable during an emergency.
    """
    return str(environ.get("TRE_GPU_TRUTH_REQUIRED", "true")).strip().lower() not in {
        "0",
        "false",
        "no",
    }

from __future__ import annotations

import os
from typing import Protocol

from fastapi import FastAPI

from tre_common.registry import ClusterTopology
from tre_common.registry import load_registry
from tre_sm.allocator.topology import K8sPodSnapshot, pod_records_from_snapshots
from tre_sm.app import create_service_app
from tre_sm.gpu_truth import RedisGpuTruth
from tre_sm.ops.k8s_ops import K8sOps
from tre_sm.ops.vllm_ops import VllmOps
from tre_sm.state.reconcile import PodRecord
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
    redis_url = os.environ.get("TRE_REDIS_URL", "redis://aibrix-redis-master:6379/0")
    redis_client = redis.Redis.from_url(redis_url)
    k8s_ops = _create_k8s_ops(registry)
    return create_service_app(
        registry,
        StateStore(redis_client),
        k8s_client=K8sPodClientFromOps(registry.topology(), k8s_ops),
        runtime_ops=k8s_ops,
        vllm_ops=VllmOps(),
        gpu_truth=RedisGpuTruth(redis_client),
        create_max_used_mib=int(os.environ.get("TRE_CREATE_MAX_USED_MIB", "2500")),
        sleep_leak_used_mib=int(os.environ.get("TRE_SLEEP_LEAK_USED_MIB", "8192")),
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

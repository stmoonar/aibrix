from __future__ import annotations

import os
from typing import Protocol

from fastapi import FastAPI

from tre_common.registry import ClusterTopology
from tre_common.registry import load_registry
from tre_sm.allocator.topology import K8sPodSnapshot, pod_records_from_snapshots
from tre_sm.app import create_service_app
from tre_sm.ops.k8s_ops import K8sOps
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
    return create_service_app(
        registry,
        StateStore(redis.Redis.from_url(redis_url)),
        k8s_client=_create_k8s_pod_client(registry.topology()),
    )


def _create_k8s_pod_client(topology: ClusterTopology) -> K8sPodClientFromOps:
    try:
        from kubernetes import client, config  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("kubernetes package is required for service-manager reconcile") from exc

    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()

    namespace = os.environ.get("TRE_MODEL_NAMESPACE", os.environ.get("TARGET_NAMESPACE", "default"))
    return K8sPodClientFromOps(topology, K8sOps(api=client.CoreV1Api(), namespace=namespace))

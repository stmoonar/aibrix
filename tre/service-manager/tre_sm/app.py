from __future__ import annotations

from fastapi import FastAPI

from tre_common.registry import Registry
from tre_sm.api.v2 import ServiceManagerV2, create_app
from tre_sm.state.reconcile import K8sPodClient
from tre_sm.state.store import StateStore


def create_service_app(
    registry: Registry,
    store: StateStore,
    *,
    k8s_client: K8sPodClient | None = None,
) -> FastAPI:
    return create_app(ServiceManagerV2(registry, store, k8s_client=k8s_client))

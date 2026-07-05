from __future__ import annotations

from fastapi import FastAPI

from tre_common.registry import Registry
from tre_sm.api.v2 import RuntimePodOps, ServiceManagerV2, VllmRuntimeOps, create_app
from tre_sm.gpu_truth import GpuTruthProvider
from tre_sm.state.reconcile import K8sPodClient
from tre_sm.state.store import StateStore


def create_service_app(
    registry: Registry,
    store: StateStore,
    *,
    k8s_client: K8sPodClient | None = None,
    runtime_ops: RuntimePodOps | None = None,
    vllm_ops: VllmRuntimeOps | None = None,
    gpu_truth: GpuTruthProvider | None = None,
    create_max_used_mib: int = 2500,
    sleep_leak_used_mib: int = 8192,
) -> FastAPI:
    return create_app(
        ServiceManagerV2(
            registry,
            store,
            k8s_client=k8s_client,
            runtime_ops=runtime_ops,
            vllm_ops=vllm_ops,
            gpu_truth=gpu_truth,
            create_max_used_mib=create_max_used_mib,
            sleep_leak_used_mib=sleep_leak_used_mib,
        )
    )

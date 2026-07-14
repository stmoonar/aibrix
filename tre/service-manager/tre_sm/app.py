from __future__ import annotations

from fastapi import FastAPI

from tre_common.registry import Registry
from tre_sm.api.v2 import RuntimePodOps, ServiceManagerV2, VllmRuntimeOps, create_app
from tre_sm.gpu_truth import GpuTruthProvider
from tre_sm.state.reconcile import K8sPodClient
from tre_sm.state.operations import OperationCoordinator
from tre_sm.state.safety import ClusterSafetyGate
from tre_sm.state.fleet_store import FleetStateStore
from tre_sm.state.gpu_leases import GpuLeaseStore
from tre_sm.state.store import StateStore
from tre_sm.state.supervisor import FleetSupervisor


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
    operation_coordinator: OperationCoordinator | None = None,
    safety_gate: ClusterSafetyGate | None = None,
    fleet_store: FleetStateStore | None = None,
    gpu_leases: GpuLeaseStore | None = None,
    supervisor_enabled: bool = False,
    supervisor_interval_s: float = 5.0,
) -> FastAPI:
    service = ServiceManagerV2(
            registry,
            store,
            k8s_client=k8s_client,
            runtime_ops=runtime_ops,
            vllm_ops=vllm_ops,
            gpu_truth=gpu_truth,
            create_max_used_mib=create_max_used_mib,
            sleep_leak_used_mib=sleep_leak_used_mib,
            operation_coordinator=operation_coordinator,
            safety_gate=safety_gate,
            fleet_store=fleet_store,
            gpu_leases=gpu_leases,
        )
    app = create_app(service)
    if supervisor_enabled:
        supervisor = FleetSupervisor(
            service, interval_s=supervisor_interval_s
        )
        service.set_supervisor(supervisor)
        # FastAPI 0.12x removed the application-level convenience method;
        # Starlette's router lifecycle API remains stable across our dev and
        # runtime versions.
        app.router.add_event_handler("startup", supervisor.start)
        app.router.add_event_handler("shutdown", supervisor.stop)
    return app

from __future__ import annotations

import logging
import signal

from fastapi import FastAPI

from tre_common.registry import Registry
from tre_sm.api.v2 import RuntimePodOps, ServiceManagerV2, VllmRuntimeOps, create_app
from tre_sm.gpu_truth import GpuTruthProvider
from tre_sm.ops.sleep_primitive import GatewayState, SleepJournal
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
    create_max_used_mib: int | None = None,
    sleep_leak_used_mib: int = 8192,
    require_gpu_truth: bool = True,
    operation_coordinator: OperationCoordinator | None = None,
    safety_gate: ClusterSafetyGate | None = None,
    fleet_store: FleetStateStore | None = None,
    gpu_leases: GpuLeaseStore | None = None,
    gateway_state: GatewayState | None = None,
    sleep_journal: SleepJournal | None = None,
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
            require_gpu_truth=require_gpu_truth,
            operation_coordinator=operation_coordinator,
            safety_gate=safety_gate,
            fleet_store=fleet_store,
            gpu_leases=gpu_leases,
            gateway_state=gateway_state,
            sleep_journal=sleep_journal,
        )
    app = create_app(service)
    install_lifecycle(app, service)
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


LOG = logging.getLogger(__name__)


def install_lifecycle(app: FastAPI, service: ServiceManagerV2) -> None:
    """Startup: resolve sleep journal entries a dead SM left behind (review P1-2)
    and chain a SIGTERM hook. Shutdown: wait (bounded) for sleeps to finish.

    On SIGTERM the hook stops new sleeps at once and makes every drain that has
    not reached /sleep roll back at its next poll, so uvicorn's graceful shutdown
    (which waits for in-flight requests) is not held up by minutes of draining.
    The Deployment's terminationGracePeriodSeconds must cover the rest (see the
    tre-v2 overlay).
    """

    def on_startup() -> None:
        try:
            service.recover_sleep_journal()
        except Exception:  # the supervisor retries; the audit shows what is left
            LOG.exception("sleep journal recovery at startup failed")
        install_sigterm_hook(service)

    def on_shutdown() -> None:
        if not service.shutdown(timeout_s=service.shutdown_timeout_s()):
            LOG.warning("service-manager shutdown: sleeps still in progress after the wait")

    app.router.add_event_handler("startup", on_startup)
    app.router.add_event_handler("shutdown", on_shutdown)


def install_sigterm_hook(service: ServiceManagerV2) -> bool:
    """Chain ``service.begin_shutdown`` in front of the current SIGTERM handler
    (uvicorn's). Must run in the main thread (FastAPI startup does)."""
    previous = signal.getsignal(signal.SIGTERM)

    def handler(signum, frame):
        service.begin_shutdown()
        if callable(previous):
            previous(signum, frame)

    try:
        signal.signal(signal.SIGTERM, handler)
    except ValueError:  # not the main thread (embedded / tests)
        return False
    return True

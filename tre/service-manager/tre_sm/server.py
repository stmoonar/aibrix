from __future__ import annotations

import logging
import os
from typing import Protocol

from fastapi import FastAPI

from tre_common.registry import ClusterTopology
from tre_common.registry import load_registry
from tre_sm.allocator.topology import K8sPodSnapshot, pod_records_from_snapshots
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.app import create_service_app
from tre_sm.clock_check import check_clock_skew
from tre_sm.gpu_truth import RedisGpuTruth
from tre_sm.ops.k8s_ops import K8sOps
from tre_sm.ops.sleep_primitive import GatewayState, SleepJournal
from tre_sm.ops.vllm_ops import VllmOps
from tre_sm.state.reconcile import PodRecord
from tre_sm.state.operations import OperationCoordinator
from tre_sm.state.safety import MAINTENANCE_RENEW_S, MAINTENANCE_TTL_S, ClusterSafetyGate
from tre_sm.state.fleet_seed import seed_desired
from tre_sm.state.fleet_store import FleetStateStore
from tre_sm.state.gpu_leases import GpuLeaseStore
from tre_sm.state.store import StateStore
from tre_sm.state.wake_journal import WakeJournal


LOG = logging.getLogger(__name__)

#: Loggers of the service-manager's own code; set to the configured level.
SM_LOGGERS = ("tre_sm", "tre_common")
LOG_LEVEL_ENV = "TRE_SM_LOG_LEVEL"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_HANDLER_NAME = "tre-sm-log"


def resolve_log_level(environ, registry_level: str | None = None) -> int:
    """TRE_SM_LOG_LEVEL, else registry ``service_manager.log_level``, else INFO.
    An unknown env value falls back to INFO (with a warning) rather than keep
    the SM from starting; the registry value is validated at start."""
    raw = environ.get(LOG_LEVEL_ENV)
    source = LOG_LEVEL_ENV
    if raw is None or not str(raw).strip():
        raw, source = registry_level, "service_manager.log_level"
    if raw is None or not str(raw).strip():
        return logging.INFO
    level = logging.getLevelName(str(raw).strip().upper())
    if isinstance(level, int):
        return level
    LOG.warning("unknown log level %r from %s; using INFO", raw, source)
    return logging.INFO


def configure_logging(level: int) -> None:
    """Make the SM's own log lines (seed, clock skew, sleep primitive, ...) visible.

    uvicorn configures only its own loggers (``uvicorn``, ``uvicorn.error``,
    ``uvicorn.access``: own handlers, no propagation), so records of other
    loggers reach the root logger, which has no handler and WARNING level. This
    adds ONE timestamped stderr handler to the root logger (idempotent) and sets
    the ``tre_sm`` / ``tre_common`` loggers to ``level``; the root level stays
    as it is, so third-party libraries (kubernetes, urllib3) keep logging at
    WARNING and uvicorn's lines are not duplicated.
    """
    root = logging.getLogger()
    if not any(handler.get_name() == _HANDLER_NAME for handler in root.handlers):
        handler = logging.StreamHandler()
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
    for name in SM_LOGGERS:
        logging.getLogger(name).setLevel(level)


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

    # Env level first (registry errors are logged too), then the registry's.
    configure_logging(resolve_log_level(os.environ))
    registry = load_registry(os.environ.get("TRE_REGISTRY_PATH"))
    check_service_manager_config(registry)
    configure_logging(resolve_log_level(os.environ, registry.service_manager().log_level))
    redis_url = os.environ.get("TRE_REDIS_URL", "redis://aibrix-redis-master:6379/0")
    redis_client = redis.Redis.from_url(redis_url)
    sm_config = registry.service_manager()
    check_clock_skew(
        redis_client,
        warn_s=sm_config.clock_skew_warn_s,
        fail_s=sm_config.clock_skew_fail_s,
    )
    k8s_ops = _create_k8s_ops(registry)
    operation_coordinator = OperationCoordinator(
        redis_client,
        owner=os.environ.get("HOSTNAME", "tre-v2-service-manager"),
        lease_ttl_ms=int(os.environ.get("TRE_SM_WRITER_LEASE_TTL_MS", "30000")),
        max_records=sm_config.operations_max_records,
    )
    # Every vLLM probe of a sleep uses the registry's probe timeout: it is part of
    # the worst-case call duration the SM and the controller validate.
    vllm_ops = VllmOps(timeout_s=sm_config.sleep.probe_timeout_s)
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
        # D7: every registry binding (and TRE-managed Deployment) gets a desired
        # record (append-only), so the startup gate admits the Pods of a fresh
        # deployment on an empty Redis; existing pods are seeded with their
        # actual power (review P2-8).
        seeded = seed_desired(registry, fleet_store, runtime_ops=k8s_ops, vllm_ops=vllm_ops)
        if seeded["added"]:
            LOG.info("seeded desired state from registry: %s", seeded)
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
        maintenance_ttl_s=float(
            os.environ.get("TRE_SM_MAINTENANCE_TTL_S", str(MAINTENANCE_TTL_S))
        ),
        maintenance_renew_s=float(
            os.environ.get("TRE_SM_MAINTENANCE_RENEW_S", str(MAINTENANCE_RENEW_S))
        ),
    )
    return create_service_app(
        registry,
        legacy_store,
        k8s_client=K8sPodClientFromOps(registry.topology(), k8s_ops),
        runtime_ops=k8s_ops,
        vllm_ops=vllm_ops,
        gpu_truth=RedisGpuTruth(redis_client),
        # Explicit override only (B9): unset = service_manager.create in the registry
        # (absolute max_used_mib, else derived from each model's
        # --gpu-memory-utilization and the GPU's total memory).
        create_max_used_mib=(
            int(os.environ["TRE_CREATE_MAX_USED_MIB"])
            if os.environ.get("TRE_CREATE_MAX_USED_MIB", "").strip()
            else None
        ),
        sleep_leak_used_mib=int(os.environ.get("TRE_SLEEP_LEAK_USED_MIB", "8192")),
        require_gpu_truth=gpu_truth_required_from_env(os.environ),
        operation_coordinator=operation_coordinator,
        safety_gate=safety_gate,
        fleet_store=fleet_store,
        gpu_leases=gpu_leases,
        gateway_state=GatewayState(
            redis_client,
            plugin_pods=_plugin_pod_lister(k8s_ops, sm_config.sleep),
        ),
        sleep_journal=SleepJournal(redis_client),
        wake_journal=WakeJournal(redis_client),
        # Read only while registry service_manager.test_hooks is true.
        fault_redis=redis_client,
        supervisor_enabled=os.environ.get(
            "TRE_SM_SUPERVISOR_ENABLED", "true"
        ).lower() in {"1", "true", "yes"},
        supervisor_interval_s=float(
            os.environ.get("TRE_SM_SUPERVISOR_INTERVAL_S", "5")
        ),
    )


def check_service_manager_config(registry) -> None:
    """Refuse to start on an invalid service_manager: / gateway: section, e.g. a
    worst-case sleeping call that outlasts the controller's call timeout."""
    errors = registry.validate_service_manager()
    if errors:
        raise RuntimeError(
            "invalid registry service_manager/gateway configuration: " + "; ".join(errors)
        )


def _plugin_pod_lister(k8s_ops: K8sOps, policy):
    selector = policy.plugin_label_selector
    if not selector:
        return None
    return lambda: k8s_ops.list_ready_pod_names(
        namespace=policy.plugin_namespace, label_selector=selector
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

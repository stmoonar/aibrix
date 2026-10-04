from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
from contextlib import contextmanager
import contextvars
import logging
import threading
import time
import re
import json
from dataclasses import asdict, dataclass, field, replace
from functools import wraps
from typing import Callable, Protocol
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tre_common.registry import (
    SLEEP_PATHS,
    Registry,
    ServiceManagerConfig,
    gpu_memory_utilization,
    scale_max_replicas,
)
from tre_common.registry import NodeSpec
from tre_common import rediskeys
from tre_common.gpu_placement import (
    PlacementPolicy,
    choose_placement,
    placement_policy_from_registry,
)
from tre_sm.allocator.slots import (
    Binding,
    Migration,
    Slot,
    SlotAllocator,
    awake_gpus,
    awake_model_counts,
    is_buddy_aligned,
    model_awake_gpus,
    node_gpu_counts,
    release_order,
    slot_block,
)
from tre_sm.allocator.topology import K8sPodSnapshot
from tre_sm.gpu_truth import GpuTruthProvider
from tre_sm.ops.k8s_ops import StartupPodRecord
from tre_sm.ops.transfer import (
    SKIP_FLOOR,
    TRANSFER_DONE,
    TRANSFER_DONOR_FAILED,
    TRANSFER_PENDING,
    TRANSFER_RECEIVER_FAILED,
    TransferPair,
    TransferSelection,
    busy_binding_ids,
    select_transfer_pairs,
)
from tre_sm.ops.sleep_primitive import (
    PHASE_SLEPT,
    STATUS_SLEPT,
    Clock,
    GatewayState,
    ServiceShuttingDown,
    SleepFailed,
    SleepJournal,
    SleepPrimitive,
    SleepTarget,
    _parallel,
)
from tre_sm.ops.vllm_ops import NO_SIDECAR
from tre_sm.state.replica_floor import (
    EXEMPT_PATHS,
    HIDE_PATH,
    MAKEUP_PATHS,
    TRANSIENT_LEASE_PHASES,
    FloorCheck,
    FloorRecorder,
    FloorViolation,
    check_floor,
)
from tre_sm.state.reconcile import K8sPodClient, POD_STATE_AWAKE, POD_STATE_HIDDEN, POD_STATE_SLEEPING, audit_state, observe_bindings, reconcile_state
from tre_sm.state.operations import (
    OperationBusy,
    OperationCoordinator,
    current_operation,
    reset_current_actor,
    set_current_actor,
)
from tre_sm.state.fleet_repair import FleetRepairExecutor
from tre_sm.state.fleet_seed import registry_binding_ids, seed_binding_ids, seed_desired
from tre_sm.state.fleet_store import DesiredBinding, FleetStateConflict, FleetStateStore, ObservedBinding
from tre_sm.state.safety import ClusterSafetyGate, MaintenanceLockLost, NodePressureActive
from tre_sm.state.gpu_leases import GpuLeaseConflict, GpuLeaseStore
from tre_sm.state.store import StateConflict, StateFenceError, StateStore
from tre_sm.state.wake_journal import RestartLedger, WakeJournal
from tre_sm.api.v1_compat import create_v1_compat_router


_NAT_SPLIT = re.compile(r"(\d+)")
LOG = logging.getLogger(__name__)

#: Journal field of a request-initiated sleep: record desired power "sleeping"
#: once the pod is confirmed asleep (also by crash recovery), never before
#: (review 2 P2-3).
DESIRED_ON_SLEEP = {"desired_on_sleep": "sleeping"}

#: Poll interval of a GPU headroom gate waiting for a fresh gpu-truth sample.
TRUTH_POLL_S = 0.2


def serialized_operation(kind: str):
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            if self._operation_coordinator is None:
                return method(self, *args, **kwargs)
            with self._operation_coordinator.operation(
                kind, wait_s=self._sm_config.writer_lock_wait_s
            ) as operation:
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
    # Only tre_sm.ops.sleep_primitive may call sleep() (plan D2).
    def sleep(
        self,
        pod_ip: str,
        *,
        port: int | None = None,
        mode: str | None = None,
        timeout_s: float | None = None,
        hidden: bool = False,
    ): ...

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
        create_max_used_mib: int | None = None,
        sleep_leak_used_mib: int = 8192,
        require_gpu_truth: bool = True,
        operation_coordinator: OperationCoordinator | None = None,
        safety_gate: ClusterSafetyGate | None = None,
        fleet_store: FleetStateStore | None = None,
        gpu_leases: GpuLeaseStore | None = None,
        gateway_state: GatewayState | None = None,
        sleep_journal: SleepJournal | None = None,
        sleep_clock: Clock | None = None,
        wake_journal: WakeJournal | None = None,
        fault_redis=None,
        restart_ledger: RestartLedger | None = None,
        restored_placeholders: list[tuple[str, str, tuple[int, ...], str]] | None = None,
        restored_suspects: list[tuple[str, str, tuple[int, ...], str]] | None = None,
        placeholder_min_hold_s: float = 120.0,
        placeholder_release_reasons=None,
    ) -> None:
        self._registry = registry
        # Registry placement policy shared with the controller planner (design
        # 20260928-placement-node-balance); None = plain buddy best-fit (registry
        # stubs without models).
        self._placement = _registry_placement_policy(registry)
        config = getattr(registry, "service_manager", None)
        self._sm_config: ServiceManagerConfig = (
            config() if callable(config) else ServiceManagerConfig()
        )
        self._sleep_clock = sleep_clock
        # Replica floor (2026-09-29): the check and the hide / sleep it guards run
        # under the one writer lock (whole-lock, 2026-10-02); counters and recent
        # events of refusals / clamps / exemptions / make-up wakes.
        self._floor_recorder = FloorRecorder(
            log_interval_s=float(getattr(self._sm_config, "replica_floor_log_interval_s", 0.0) or 0.0)
        )
        self._sleep_primitive: SleepPrimitive | None = None
        if runtime_ops is not None and vllm_ops is not None:
            self._sleep_primitive = SleepPrimitive(
                runtime_ops=runtime_ops,
                vllm_ops=vllm_ops,
                policy=self._sm_config.sleep,
                gateway=gateway_state,
                journal=sleep_journal or SleepJournal(),
                clock=sleep_clock,
                owner=getattr(operation_coordinator, "owner", "service-manager"),
                operation_id=_current_operation_id,
                floor_guard=self._sleep_floor_guard,
            )
            # One set of floor counters (review 2026-09-29 P3): written to and read
            # from the sleep journal's stats (Redis when configured), so
            # ``floor.counts`` and ``stats`` of GET /v2/sleep agree and survive a
            # restart.
            self._floor_recorder.incr = self._sleep_primitive.journal.incr
            self._floor_recorder.read_counts = self._sleep_primitive.journal.stats
        self._store = store
        self._k8s_client = k8s_client
        self._runtime_ops = runtime_ops
        self._vllm_ops = vllm_ops
        self._gpu_truth = gpu_truth
        # Wake gate trust (S1, 2026-09-30): (node, gpu) -> the gpu-truth sample a
        # wake gate may trust again after the last local power change on that GPU
        # (sleep commit, wake, start, deletion). Process-local: the SM is the only
        # writer of power changes; after a restart every TTL-valid sample is trusted
        # (the account and the GPU leases still fence every wake).
        self._power_marks: dict[tuple[str, int], _PowerMark] = {}
        self._power_marks_lock = threading.Lock()
        #: node -> (sample identity, monotonic time this SM first saw it): truth_age_s.
        self._truth_seen: dict[str, tuple[tuple, float]] = {}
        # Explicit absolute cold-start limit (env TRE_CREATE_MAX_USED_MIB); None =
        # service_manager.create.max_used_mib, else derived per GPU and model (B9).
        self._create_max_used_mib = create_max_used_mib
        self._sleep_leak_used_mib = sleep_leak_used_mib
        self._require_gpu_truth = require_gpu_truth
        self._operation_coordinator = operation_coordinator
        self._safety_gate = safety_gate
        self._fleet_store = fleet_store
        self._gpu_leases = gpu_leases
        # Wake journal (Redis when configured): the crash evidence of a wake. Every
        # wake runs under the writer lock from its checks to its commit
        # (whole-lock, 2026-10-02); an entry outlives the lock only when the
        # service-manager died mid-wake or the engine's state could not be read -
        # the journal recovery settles it from the physical state.
        self._wake_journal = wake_journal or WakeJournal()
        #: Redis of the acceptance-test fault hooks; read only while registry
        #: service_manager.test_hooks is true (off by default: never read).
        self._fault_redis = fault_redis
        #: /wake_up of the bindings of ONE request run concurrently on it.
        self._wake_executor = ThreadPoolExecutor(
            max_workers=WAKE_WORKERS, thread_name_prefix="wake"
        )
        #: P2-8 / P1-3: starting lease binding id -> (monotonic time this SM first
        #: saw it, restart count of its Pod then).
        self._placeholder_seen: dict[str, tuple[float, int]] = {}
        #: P1-4: pod UID -> restart count last seen (container restarts in place);
        #: persisted (review P2-4) and loaded at the first guard pass.
        self._restart_ledger = restart_ledger or RestartLedger()
        self._restarts_seen: dict[str, int] | None = None
        #: Released placeholders (review P2-3): binding id -> (node, gpu ids, pod).
        #: Their GPUs are never trusted from gpu-truth (the probe path decides) until
        #: the pod reads asleep, is gone, or is converged.
        self._suspects: dict[str, tuple[str, tuple[int, ...], str]] = {}
        #: Placeholders already alerted for exceeding placeholder_max_s.
        self._placeholder_alerted: set[str] = set()
        #: A placeholder is never released before this SM has seen it this long
        #: (second guard against releasing a Pod that is still starting; env
        #: TRE_SM_PLACEHOLDER_MIN_HOLD_S).
        self._placeholder_min_hold_s = max(0.0, float(placeholder_min_hold_s))
        #: Waiting reasons of the engine container that mean "not running, holds
        #: no GPU memory" (env TRE_SM_PLACEHOLDER_RELEASE_REASONS, comma list).
        self._placeholder_release_reasons = frozenset(
            PLACEHOLDER_RELEASE_REASONS if placeholder_release_reasons is None else placeholder_release_reasons
        )
        #: P1-4: binding ids holding a restart placeholder (starting lease) that the
        #: restart guard converges.
        self._restart_placeholders: dict[str, str] = {}
        # Placeholders re-derived at bootstrap (restart_placeholder_candidates):
        # (binding id, node, gpu ids, pod) - converged by the restart guard and
        # suspects until then.
        for binding_id, node, gpus, pod_name in restored_placeholders or ():
            self._restart_placeholders[binding_id] = pod_name
            self._suspects[binding_id] = (node, tuple(int(g) for g in gpus), pod_name)
        # Engines reloading at bootstrap on GPUs another binding is recorded awake
        # on (restart_placeholder_candidates ``conflicts``): no placeholder, but
        # suspects until the restart guard converges them (integration 2026-10-01).
        for binding_id, node, gpus, pod_name in restored_suspects or ():
            self._suspects.setdefault(binding_id, (node, tuple(int(g) for g in gpus), pod_name))
        # P2-3: after a restart the power marks are empty; the GPUs of wakes still
        # journaled may hold an engine the last gpu-truth sample does not show.
        self._mark_journaled_wakes()
        self._supervisor = None
        self._fleet_repair = None
        # Background startup admissions (review 3 P2-5): (pod, uid) -> (job, start).
        self._admission_executor = ThreadPoolExecutor(
            max_workers=ADMISSION_WORKERS, thread_name_prefix="startup-admit"
        )
        self._admission_lock = threading.Lock()
        # Startup gates seen polling (review 2026-09-29 P1-1): pod name -> (UID,
        # first request, last request) on this SM's monotonic clock. A Pod waiting
        # in its gate is not fleet drift (detect_fleet_drift) for up to
        # startup_admission.drift_grace_s; cleared once the Pod is admitted.
        self._startup_gate_seen: dict[str, tuple[str, float, float]] = {}
        self._admission_jobs: dict[tuple[str, str], tuple] = {}
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
                sleep_binding=self._repair_sleep_binding,
                on_quarantine=self._record_repair_floor_exemptions,
            )

    def set_supervisor(self, supervisor) -> None:
        self._supervisor = supervisor

    def get_supervisor_state(self) -> dict:
        if self._supervisor is None:
            state = {"running": False, "enabled": False}
        else:
            state = {"enabled": True, **asdict(self._supervisor.snapshot())}
        return {**state, **self._actuation_payload()}

    def _actuation_payload(self) -> dict:
        """SM actuation switch + maintenance lock, for /v2/supervisor (console)."""
        gate = self._safety_gate
        payload: dict = {}
        reader = getattr(gate, "actuation_state", None)
        if callable(reader):
            try:
                payload["actuation"] = reader()
            except Exception as exc:  # display only
                payload["actuation"] = {"error": str(exc)}
        maintenance = getattr(gate, "maintenance", None)
        if callable(maintenance):
            try:
                payload["maintenance"] = maintenance()
            except Exception as exc:  # display only
                payload["maintenance"] = {"error": str(exc)}
        return payload

    # ------------------------------------------------------ actuation switch
    def actuation_observe(self) -> bool:
        """SM actuation observe (``tre:v2:sm:actuation``, user decision
        2026-09-28): the supervisor takes no capacity-changing action and a
        startup admission nobody requested sleeps no resident. Without a safety
        gate (embedded / unit-test wiring) the SM is active."""
        reader = getattr(self._safety_gate, "actuation_mode", None)
        if not callable(reader):
            return False
        return reader() == "observe"

    def record_suppressed(self, action: str, detail: dict) -> None:
        record = getattr(self._safety_gate, "record_suppressed", None)
        if callable(record):
            record(action, detail)
        else:
            LOG.warning(json.dumps({"event": "sm_supervisor_action_suppressed", "action": action,
                                    "detail": detail}, sort_keys=True, default=str))

    def get_state(self, *, routable: bool = False) -> dict:
        """The fleet state. ``fetched_ms``: when this SM built it (its wall clock,
        ms). ``routable`` (only ``GET /v2/state``): per binding ``routable`` and
        per model ``routable`` / ``floor`` / ``floor_headroom``, plus the
        top-level ``floor_enforced`` - the replica-floor check's own count
        (:meth:`_routable_binding_ids`) on the same store snapshot: awake and not
        hidden in the store, no unexpired ``starting`` / ``waking`` GPU lease, Pod
        Ready with the routable label. It costs one listing of every model's Pods
        and one lease read; the v1 API (APA arm) and internal callers keep the
        light path without them. ``routable`` is None (``routable_error`` set)
        when that view cannot be read."""
        fetched_ms = int(time.time() * 1000)
        snapshot = self._store.load()
        state = {"version": snapshot.version, "fetched_ms": fetched_ms}
        if not routable:
            state["models"] = self._model_counts(snapshot.bindings)
            state["bindings"] = [self._binding_dict(binding) for binding in snapshot.bindings]
        else:
            view, routable_error = self._routable_by_model(snapshot.bindings)
            state["floor_enforced"] = self._floor_enforced()
            state["models"] = self._model_counts(snapshot.bindings, routable=view)
            state["bindings"] = [
                {
                    **self._binding_dict(binding),
                    "routable": None if view is None else binding.binding_id in view.get(binding.model, ()),
                }
                for binding in snapshot.bindings
            ]
            if routable_error is not None:
                state["routable_error"] = routable_error
        gpus, nodes = self._gpu_states(snapshot.bindings)
        state["gpus"] = gpus
        state["nodes"] = nodes
        if self._fleet_store is not None:
            state["fleet"] = self.get_fleet_state()
        return state

    #: Reasons a GPU is not wakeable in ``/v2/state`` ``gpus[]``, most binding first.
    GPU_BLOCK_REASONS = ("awake", "draining", "loading", "waking", "gpu_truth_used")

    def _gpu_states(self, bindings: list[Binding]) -> tuple[list[dict], dict]:
        """``gpus[]`` and ``nodes{}`` of ``/v2/state`` (S5): per GPU whether a wake
        could go there now and why not (an awake binding, a sleep left unconfirmed
        there - ``draining`` -, a Pod loading, a wake left journaled - ``waking`` -,
        gpu-truth showing it in use), the occupants and how the wake gate would
        judge it; per node the gpu-truth health. Read only (no refresh request, no
        probe); never fails the state call. Under the whole writer lock
        (2026-10-02) no sleep or wake is ever half-way outside it: ``draining`` /
        ``waking`` only show the journal entries a crash or an unreadable engine
        left for the recovery."""
        try:
            leases = self._active_leases()
        except Exception:  # noqa: BLE001 - shown as unknown
            leases = []
        journals: dict[str, list[dict]] = {"draining": [], "waking": []}
        try:
            if self._sleep_primitive is not None:
                journals["draining"] = list(self._sleep_primitive.journal.entries().values())
            journals["waking"] = list(self._wake_journal.entries().values())
        except Exception:  # noqa: BLE001
            pass
        gpus: list[dict] = []
        nodes: dict[str, dict] = {}
        for node in self._registry.topology().nodes:
            truth = None
            if self._gpu_truth is not None:
                try:
                    truth = self._gpu_truth.node_truth(node=node.name)
                except Exception:  # noqa: BLE001
                    truth = None
            age = self._truth_age_s(node.name, truth)
            nodes[node.name] = {
                "gpus": int(node.gpus),
                "truth_configured": self._gpu_truth is not None,
                "truth_available": truth is not None,
                "truth_age_s": age,
                "truth_seq": getattr(truth, "seq", None),
                "truth_refresh_seq": getattr(truth, "refresh_seq", None),
                "truth_timestamp": getattr(truth, "timestamp", None),
            }
            for gpu in range(int(node.gpus)):
                gpus.append(self._gpu_state(node, gpu, bindings, leases, journals, truth, age))
        return gpus, nodes

    def _gpu_state(self, node, gpu: int, bindings, leases, journals, truth, age) -> dict:
        def holder(phase: str) -> str | None:
            return next(
                (
                    lease.binding_id
                    for lease in leases
                    if lease.node == node.name
                    and gpu in {int(item) for item in lease.gpu_ids}
                    and str(getattr(lease, "phase", "")) == phase
                ),
                None,
            )

        def journaled(kind: str) -> str | None:
            for entry in journals.get(kind, ()):
                try:
                    if entry.get("node") == node.name and gpu in {int(g) for g in entry.get("gpu_ids") or ()}:
                        return str(entry.get("binding_id"))
                except (TypeError, ValueError, AttributeError):
                    continue
            return None

        awake = next(
            (b.binding_id for b in bindings if b.awake and b.slot.node == node.name and gpu in b.slot.gpu_ids),
            None,
        )
        draining = journaled("draining")
        loading = holder("starting")
        waking = holder("waking") or journaled("waking")
        used = limit = None
        truth_source = "none"
        over = False
        if self._gpu_truth is not None:
            gpu_uuid = _gpu_uuid(node, gpu)
            trusted = truth is not None and not self._untrusted_gpus(node.name, (gpu,), truth)
            if truth is not None and gpu_uuid is not None:
                used = truth.used_mib(gpu_uuid)
                total = getattr(truth, "total_mib", None)
                limit = self._sm_config.wake_limit_mib(total(gpu_uuid) if callable(total) else None)
            if trusted and used is not None and limit is not None:
                truth_source = "gpu_truth"
                over = used > limit
            else:
                truth_source = "is_sleeping_probe"
        reasons = {
            "awake": awake is not None,
            "draining": draining is not None,
            "loading": loading is not None,
            "waking": waking is not None,
            # An awake binding explains the memory; only unexplained use blocks.
            "gpu_truth_used": over and awake is None,
        }
        reason = next((name for name in self.GPU_BLOCK_REASONS if reasons[name]), None)
        return {
            "node": node.name,
            "gpu": gpu,
            "wakeable": reason is None,
            "reason": reason,
            "awake_binding_id": awake,
            "loading_binding_id": loading,
            "waking_binding_id": waking,
            "draining_binding_id": draining,
            "used_mib": used,
            "limit_mib": limit,
            "truth_source": truth_source,
            "truth_age_s": age,
        }

    def put_model_target(
        self,
        model: str,
        *,
        wake_replicas: int,
        within_bindings: bool = False,
        sleep_path: str = "scale_down",
        drain_budget_s: float | None = None,
        at_least: bool = False,
        hints: list[str] | tuple[str, ...] | None = None,
        avoid_gpus: list[str] | tuple[str, ...] | None = None,
    ) -> dict:
        """Scale a model to ``wake_replicas`` awake bindings - in ONE writer-lock
        hold from the plan to the books (whole-lock, 2026-10-02): a shrink's
        hide, gateway ack, /sleep and physical confirmation, a growth's account,
        /wake_up (every binding of the request concurrently) and commit. A
        concurrent request waits for the lock (``writer_lock_wait_s``, then 409
        ``writer_busy``); it never sees this one half-way.

        ``wake_replicas`` is an absolute target (I2, 2026-10-04): a replayed or
        duplicated request is harmless. Relative callers (v1 /scale_service,
        /wake_up) convert at arrival, outside the lock - see v1_compat.
        ``within_bindings``: a growth past the model's bindings is a no-op
        answered with ``at_bindings_limit`` (no cold create; v1 /wake_up).

        A shrink is clamped at the replica floor on EVERY path (2026-10-02):
        200 with ``taken`` (bindings put to sleep) and ``clamped_by_floor``
        (the floor kept replicas awake); never 409 floor_violation. A routable
        view that cannot be read refuses the shrink with 409
        ``routable_unknown`` before anything is hidden.

        ``avoid_gpus`` (``"node/gpu"``): GPUs the SM must not pick for a wake of this
        request. ``hints`` (S5): serve ids of sleeping bindings the caller would
        like woken; the service-manager picks the GPUs (registry placement
        policy), taking the feasible hints first; ``picked`` in the response says
        which bindings woke where (``hinted`` false = substituted).

        ``at_least``: grow-only (review 3 P2-1) - a model that already has
        ``wake_replicas`` or more awake bindings is left as it is (no-op).

        Desired state follows the outcome (review 2 P2-3): only the bindings that
        slept go "sleeping"; a failed wake's desired record is restored.
        ``drain_budget_s`` is deprecated and ignored (the SM never drains).
        """
        request = {"model": model, "wake_replicas": wake_replicas, "sleep_path": sleep_path}
        with self._writer("put_model_target", request=request):
            return self._put_model_target_locked(
                model,
                wake_replicas=wake_replicas,
                within_bindings=within_bindings,
                sleep_path=sleep_path,
                at_least=at_least,
                hints=tuple(hints or ()),
                avoid_gpus=tuple(avoid_gpus or ()),
            )

    def _put_model_target_locked(
        self,
        model: str,
        *,
        wake_replicas: int,
        sleep_path: str,
        within_bindings: bool = False,
        at_least: bool = False,
        hints: tuple[str, ...] = (),
        avoid_gpus: tuple[str, ...] = (),
    ) -> dict:
        spec = self._registry.model(model)
        if wake_replicas < 0:
            raise ValueError("wake_replicas must be non-negative")
        snapshot = self._store.load()
        model_bindings = [binding for binding in snapshot.bindings if binding.model == model]
        awake = [binding for binding in model_bindings if binding.awake]
        response = {
            "model": model,
            "wake_replicas": wake_replicas,
            "version": snapshot.version,
            "actions": [],
        }
        if within_bindings and wake_replicas > len(model_bindings):
            # v1 /wake_up: no cold create - answered "delayed" by the caller.
            response["at_bindings_limit"] = True
            return response
        if at_least and len(awake) >= wake_replicas:
            response.update({"awake": len(awake), "at_least": True})
            return response
        # Scaling cap (registry max_awake_replicas, v1/paper alignment A1); max_replicas
        # is the GPU layout size (bindings), not how many may be awake. Only GROWTH past
        # the cap is refused: shrinking (or holding) a model that is above the cap - e.g.
        # after the cap was lowered - must go through.
        self._ensure_target_within_cap(model, spec, wake_replicas, snapshot.bindings)
        if self._runtime_ops is not None and wake_replicas > len(model_bindings) and not self._has_deployment_ops():
            raise ValueError("runtime create is not implemented for target growth beyond existing bindings")
        plan = self._plan_model_target(
            model=model,
            wake_replicas=wake_replicas,
            bindings=snapshot.bindings,
            tp_size=spec.tp_size,
        )
        if plan["sleep"]:
            response["taken"] = 0
            response["clamped_by_floor"] = False
            if self._floor_enforced():
                plan = self._clamp_sleep_plan_to_floor(
                    model, plan, snapshot.bindings, response, path=sleep_path
                )
                response["clamped_by_floor"] = bool((response.get("floor") or {}).get("kept_awake"))
        if plan["sleep"] and self._sleep_primitive is not None and self._runtime_ops is not None:
            # Every binding of one scale-down goes through the sleep primitive
            # together (in parallel); the desired power of exactly those that
            # slept becomes "sleeping" (review 2 P2-3).
            targets = self._sleep_targets_for(plan["sleep"])
            outcomes = self._sleep_targets(
                targets,
                sleep_path=sleep_path,
                journal_extra=DESIRED_ON_SLEEP,
                desired_sleeping=True,
                update_store=True,
            )
            # Every target slept: the model's desired target is what is awake now.
            current = self._store.load()
            self._set_model_desired_target(
                model=model,
                target_bindings=[b for b in current.bindings if b.model == model and b.awake],
                reason="model_target_request",
            )
            response["version"] = current.version
            response["taken"] = sum(1 for item in outcomes if item.get("status") == STATUS_SLEPT)
            response["actions"] = [{"action": "sleep", "serve_id": item["serve_id"]} for item in outcomes]
            return response
        if plan["wake"] and not plan["create"] and self._physical_wake_capable():
            before = self._desired_records()
            tickets, refusals = self._prepare_target_wakes(
                model, len(plan["wake"]), snapshot.bindings, hints, before, avoid_gpus=avoid_gpus
            )
            try:
                with self._desired_guard(reason="model_target_request"):
                    self._set_model_desired_target(
                        model=model,
                        target_bindings=awake + [ticket.binding for ticket in tickets],
                        reason="model_target_request",
                    )
            except BaseException:
                self._abort_prepared_wakes(tickets)
                raise
            # /wake_up of every ticket concurrently, then the commit; the first
            # failed wake's error is raised once all are recorded.
            self._wake_tickets(tickets)
            response["version"] = self._store.load().version
            response["actions"] = [_wake_action(ticket) for ticket in tickets]
            response["picked"] = [_picked(ticket) for ticket in tickets]
            unfilled = len(plan["wake"]) - len(tickets)
            if unfilled:
                response["unfilled"] = unfilled
                # Where the SM could not wake (the controller cools those GPUs down).
                response["refusals"] = [
                    exc.body(retry_after_s=self.wake_retry_after_s(exc.scope))
                    for exc in refusals
                    if isinstance(exc, WakeConflict)
                ]
                if not at_least:
                    # P2-7: an exact target (APA /scale_service) that could not be met
                    # is not a success; what did wake stays awake (committed).
                    first = response["refusals"][0] if response["refusals"] else {}
                    raise WakeConflict(
                        f"model target of {model}: {unfilled} of {len(tickets) + unfilled} wakes could not "
                        f"be placed (woke {[t.binding.serve_id for t in tickets]}); refusals: "
                        f"{[r.get('reason') for r in response['refusals']]}",
                        reason="partial", node=first.get("node"), gpus=first.get("gpu_ids") or (),
                        blocking_binding_id=first.get("blocking_binding_id"),
                    )
            return response
        with self._desired_guard(reason="model_target_request") as guard:
            self._set_model_desired_target(
                model=model,
                target_bindings=plan["target_bindings"],
                reason="model_target_request",
            )
            return self._apply_model_target_plan(
                model, wake_replicas, snapshot, plan, response, guard=guard
            )

    def _physical_wake_capable(self) -> bool:
        """Physical wakes are possible (runtime + vLLM ops)."""
        return self._runtime_ops is not None and self._vllm_ops is not None

    def _desired_records(self) -> dict[str, tuple[str, bool]]:
        """binding id -> (desired power, hidden), empty without a fleet store."""
        if self._fleet_store is None:
            return {}
        return {
            item.binding_id: (item.power, bool(item.hidden))
            for item in self._fleet_store.load_desired().bindings
        }

    def _prepare_target_wakes(
        self,
        model: str,
        need: int,
        bindings: list[Binding],
        hints: tuple[str, ...],
        before: dict[str, tuple[str, bool]],
        *,
        avoid_gpus: tuple[str, ...] = (),
    ) -> tuple[list["_WakeTicket"], list[BaseException]]:
        """The wake prepare of a model's growth by ``need`` wakes (writer lock held; S5). The
        service-manager picks: among the sleeping bindings the account allows, the
        caller's hints first, else the best by the registry placement policy
        (``_wake_pick`` = ``tre_common.gpu_placement.choose_placement``, each pick
        scored against the earlier ones). A candidate the wake gate / a lease /
        a journaled wake refuses is skipped (``placement_substituted`` event) and the
        next best tried. Fewer than ``need`` -> what could be prepared (none ->
        the first refusal is raised). Returns (tickets, refusals)."""
        hinted = [str(hint) for hint in hints]
        planning = list(bindings)
        leases = self._active_leases()
        journal = self._wake_journal.entries()
        topology = self._registry.topology()
        tickets: list[_WakeTicket] = []
        refusals: list[BaseException] = []
        tried: set[str] = set()
        avoid = {str(item) for item in avoid_gpus}
        try:
            while len(tickets) < need:
                sleeping = [
                    b for b in planning
                    if b.model == model and not b.awake and b.serve_id not in tried
                    and not any(f"{b.slot.node}/{gpu}" in avoid for gpu in b.slot.gpu_ids)
                ]
                blockers = {b.serve_id: self._wake_blocker(b, planning, leases, journal) for b in sleeping}
                feasible = [b for b in sleeping if blockers[b.serve_id] is None]
                if not feasible:
                    refusals.extend(blockers[b.serve_id] for b in sleeping[:1])
                    break
                preferred = [b for b in feasible if b.serve_id in hinted]
                binding = _wake_pick(preferred or feasible, planning, topology, self._placement)
                tried.add(binding.serve_id)
                open_hints = [
                    hint for hint in hinted
                    if hint != binding.serve_id and hint not in {t.binding.serve_id for t in tickets}
                ]
                placement = {
                    "hint_binding_id": binding.binding_id if binding.serve_id in hinted else (
                        self._binding_id_of(open_hints[0], planning) if open_hints else None
                    ),
                    "chosen_binding_id": binding.binding_id,
                    "source": "hint" if binding.serve_id in hinted else "sm_choose",
                }
                try:
                    ticket = self._prepare_wake(
                        binding, planning, leases=leases, journal=journal,
                        previous_desired=before.get(binding.binding_id), placement=placement,
                    )
                except (WakeConflict, GpuLeaseConflict, ValueError) as exc:
                    refusals.append(exc)
                    _log_event(
                        "placement_substituted",
                        level=logging.WARNING,
                        model=model, refused_binding_id=binding.binding_id,
                        error_code=_error_code(exc), detail=str(exc),
                    )
                    continue
                tickets.append(ticket)
                planning = [
                    replace(item, awake=True, hidden=False) if item.serve_id == binding.serve_id else item
                    for item in planning
                ]
        except BaseException:
            self._abort_prepared_wakes(tickets)
            raise
        if not tickets:
            raise refusals[0] if refusals else WakeConflict(
                f"no wakeable sleeping binding of {model}", node=None, gpus=()
            )
        self._note_wake_details(tickets)
        return tickets, refusals

    @staticmethod
    def _binding_id_of(serve_id: str, bindings: list[Binding]) -> str | None:
        return next((b.binding_id for b in bindings if b.serve_id == serve_id), None)

    def _apply_model_target_plan(
        self, model: str, wake_replicas: int, snapshot, plan: dict, response: dict, *, guard=None
    ) -> dict:
        actions: list[dict] = []
        updated_by_serve = {binding.serve_id: binding for binding in snapshot.bindings}
        for binding in plan["sleep"]:  # no runtime: record the intent only
            updated_by_serve[binding.serve_id] = replace(binding, awake=False, hidden=False)
            actions.append({"action": "sleep", "serve_id": binding.serve_id})

        for binding in plan["wake"]:
            self._apply_runtime_power_action(binding, action="wake")
            if guard is not None:
                guard.settle([binding.binding_id])  # awake now: keep its desired
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
            if guard is not None:
                guard.settle([binding.binding_id])
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
        response["version"] = version
        response["actions"] = actions
        return response

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
            # Review F2: when shrinking, sleep hidden (safescale-probed, unroutable)
            # bindings first so a serving pod is never slept while the hidden one stays
            # awake as an orphan. Without hidden bindings this is exactly the legacy
            # "sleep the tail of awake" order.
            shrink = len(awake) - wake_replicas
            hidden = sorted(
                (binding for binding in awake if binding.hidden),
                key=lambda item: _natural_key(item.serve_id),
            )
            # Hidden first (they are already drained); the rest in buddy-release
            # order so shrinking off four GPUs hands back an aligned pair instead
            # of two orphaned singles.
            serving = release_order(
                [binding for binding in awake if not binding.hidden],
                bindings=bindings,
                topology=self._registry.topology(),
                already_released=hidden[:shrink],
                policy=self._placement,
            )
            candidates = hidden + serving
            sleeping = candidates[:shrink]
            sleeping_ids = {binding.serve_id for binding in sleeping}
            target = [
                binding for binding in awake if binding.serve_id not in sleeping_ids
            ]
            return {
                "sleep": sleeping,
                "wake": [],
                "create": [],
                "target_bindings": target,
            }

        target = list(awake)
        sleeping = [binding for binding in model_bindings if not binding.awake]
        topology = self._registry.topology()
        wakes: list[Binding] = []
        existing_target = min(wake_replicas, len(model_bindings))
        leases = self._active_leases()
        journal = self._wake_journal.entries()
        while sleeping and len(target) < existing_target:
            blockers = {
                binding.serve_id: self._wake_blocker(binding, list(planning.values()), leases, journal)
                for binding in sleeping
            }
            feasible = [binding for binding in sleeping if blockers[binding.serve_id] is None]
            if not feasible:
                raise blockers[sleeping[0].serve_id]
            binding = _wake_pick(feasible, planning.values(), topology, self._placement)
            sleeping.remove(binding)
            planning[binding.serve_id] = replace(
                binding, awake=True, hidden=False
            )
            wakes.append(binding)
            target.append(planning[binding.serve_id])

        creates: list[Binding] = []
        existing_ids = set(planning)
        allocator = SlotAllocator(
            self._registry.topology(), list(planning.values()), policy=self._placement
        )
        while len(target) + len(creates) < wake_replicas:
            slot = allocator.find_slot(tp_size, model)
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

    def put_binding_power(
        self,
        serve_id: str,
        *,
        awake: bool,
        sleep_path: str = "scale_down",
        drain_budget_s: float | None = None,
    ) -> dict:
        """Wake or sleep one binding, in ONE writer-lock hold (whole-lock,
        2026-10-02): a wake's account, /wake_up and commit, or a sleep's floor
        check (409 ``floor_violation`` / ``routable_unknown``: a named binding is
        never clamped), hide, gateway ack, /sleep and confirmation.
        ``drain_budget_s`` is deprecated and ignored."""
        request = {"serve_id": serve_id, "awake": awake, "sleep_path": sleep_path}
        with self._writer("put_binding_power", request=request):
            snapshot = self._store.load()
            binding = next((item for item in snapshot.bindings if item.serve_id == serve_id), None)
            if awake:
                if binding is not None and not binding.awake:
                    # Controller-requested wake: same scaling cap as put_model_target.
                    # Fleet repair (_set_binding_power_by_id_unlocked) restores recorded
                    # desired state and is deliberately not capped here.
                    self._ensure_wake_within_cap(binding, snapshot.bindings)
                    if self._physical_wake_capable():
                        return self._wake_binding_locked(binding, snapshot.bindings)
                return self._put_binding_power_unlocked(serve_id, awake=True, sleep_path=sleep_path)
            if self._sleep_primitive is None or self._runtime_ops is None:
                return self._put_binding_power_unlocked(serve_id, awake=False, sleep_path=sleep_path)
            if binding is None:
                raise ValueError(f"unknown binding: {serve_id}")
            if self._wake_journal.get(binding.binding_id) is not None:
                # A wake of it was left journaled (a crash or an unreadable engine):
                # the journal recovery settles it first.
                raise WakeConflict(
                    f"power change of {serve_id}: a wake of {binding.binding_id} is journaled "
                    "(left for the wake journal recovery)",
                    reason="wake_in_progress", node=binding.slot.node, gpus=binding.slot.gpu_ids,
                    binding_id=binding.binding_id, blocking_binding_id=binding.binding_id,
                )
            if not binding.awake:
                self._update_desired(
                    {binding.binding_id: {"power": "sleeping", "hidden": False}},
                    updated_by="service-manager-api",
                    reason="binding_power_request",
                )
                return {
                    "serve_id": serve_id,
                    "awake": False,
                    "version": snapshot.version,
                    "actions": [],
                    "binding": self._binding_dict(binding),
                }
            # Desired power "sleeping" is recorded once the pod is confirmed
            # asleep (review 2 P2-3), never before.
            self._sleep_targets(
                self._sleep_targets_for([binding]),
                sleep_path=sleep_path,
                journal_extra=DESIRED_ON_SLEEP,
                desired_sleeping=True,
                update_store=True,
            )
            return {
                "serve_id": serve_id,
                "awake": False,
                "version": self._store.load().version,
                "actions": [{"action": "sleep", "serve_id": serve_id}],
                "binding": self._binding_dict(replace(binding, awake=False, hidden=False)),
            }

    def _wake_binding_locked(self, binding: Binding, bindings: list[Binding]) -> dict:
        """A binding-level wake (writer lock held): the checks and the GPU lease,
        the desired intent (power awake; restored if the wake fails), /wake_up,
        the commit."""
        ticket = self._prepare_wake(
            binding, bindings, previous_desired=self._desired_power_of(binding.binding_id)
        )
        try:
            self._update_desired(
                {binding.binding_id: {"power": "awake"}},
                updated_by="service-manager-api",
                reason="binding_power_request",
            )
        except BaseException:
            self._abort_prepared_wakes([ticket])
            raise
        self._note_wake_details([ticket])
        self._wake_tickets([ticket])
        return {
            "serve_id": binding.serve_id,
            "awake": True,
            "version": self._store.load().version,
            "actions": [_wake_action(ticket)],
            "picked": [_picked(ticket)],
            "binding": self._binding_dict(replace(binding, awake=True, hidden=False)),
        }

    @contextmanager
    def _writer(self, kind: str, *, request: dict | None = None, wait_s: float | None = None):
        """Hold the SM writer lock for one phase, waiting up to ``wait_s``
        (default ``service_manager.writer_lock_wait_s``) for it."""
        if self._operation_coordinator is None:
            yield None
            return
        with self._operation_coordinator.operation(
            kind,
            request=request,
            wait_s=self._sm_config.writer_lock_wait_s if wait_s is None else wait_s,
        ) as operation:
            operation.advance("executing")
            yield operation

    @contextmanager
    def _desired_guard(self, binding_ids=None, *, reason: str):
        """Desired state changes only stick when the body succeeds (review 2
        P2-3): on ANY exception the desired records of ``binding_ids`` (None =
        all) are put back to what they were on entry; a record the body added is
        marked ``absent``. Needs the writer fence (like every desired write).

        Yields a :class:`_DesiredGuard`: the body calls ``settle(ids)`` once the
        PHYSICAL change of those bindings is done (a wake succeeded, a defrag
        migration completed) - a later failure (e.g. the legacy store save) no
        longer rolls their desired state back to something the cluster is not
        (review 3 P2-4 / P3)."""
        guard = _DesiredGuard()
        if self._fleet_store is None:
            yield guard
            return
        try:
            before = {item.binding_id: item for item in self._fleet_store.load_desired().bindings}
        except Exception:
            before = None
        try:
            yield guard
        except BaseException:
            if before is not None and not guard.all_settled:
                self._restore_desired(before, binding_ids, reason=reason, keep=guard.settled)
            raise

    def _restore_desired(self, before: dict, binding_ids, *, reason: str, keep=frozenset()) -> None:
        try:
            snapshot = self._fleet_store.load_desired()
            by_id = {item.binding_id: item for item in snapshot.bindings}
            ids = set(by_id) if binding_ids is None else set(binding_ids) & set(by_id)
            ids -= set(keep)
            changed = False
            for binding_id in ids:
                current = by_id[binding_id]
                previous = before.get(binding_id)
                if previous is None:
                    restored = current.with_intent(
                        lifecycle="absent",
                        power="sleeping",
                        hidden=False,
                        updated_by="service-manager-rollback",
                        reason=f"{reason}_failed",
                    )
                else:
                    restored = current.with_intent(
                        lifecycle=previous.lifecycle,
                        power=previous.power,
                        hidden=previous.hidden,
                        updated_by="service-manager-rollback",
                        reason=f"{reason}_failed",
                    )
                by_id[binding_id] = restored
                changed = changed or restored is not current
            if changed:
                self._fleet_store.save_desired(
                    list(by_id.values()), expected_version=snapshot.version
                )
        except Exception:  # the audit reports the desired/observed mismatch
            LOG.exception("restoring desired state after a failed %s failed", reason)

    def _locked_sleep(
        self,
        bindings: list[Binding],
        *,
        sleep_path: str,
        kind: str,
        desired_sleeping: bool = False,
        before_prepare: Callable[[], None] | None = None,
    ) -> list[dict]:
        """Sleep ``bindings`` in ONE writer-lock hold (whole-lock, 2026-10-02),
        e.g. for startup admission / convergence. The desired state is not
        touched unless ``desired_sleeping``. ``before_prepare`` runs under the
        lock right before anything is hidden or slept; raising aborts the sleep
        with nothing changed."""
        with self._writer(kind):
            if before_prepare is not None:
                before_prepare()
            return self._sleep_targets(
                self._sleep_targets_for(bindings),
                sleep_path=sleep_path,
                journal_extra=DESIRED_ON_SLEEP if desired_sleeping else None,
                desired_sleeping=desired_sleeping,
                update_store=True,
            )

    def _put_binding_power_unlocked(
        self,
        serve_id: str,
        *,
        awake: bool,
        sleep_path: str = "default",
        drain_budget_s: float | None = None,
    ) -> dict:
        snapshot = self._store.load()
        binding = next(
            (item for item in snapshot.bindings if item.serve_id == serve_id),
            None,
        )
        if binding is None:
            raise ValueError(f"unknown binding: {serve_id}")
        intent: dict[str, object] = {"power": "awake" if awake else "sleeping"}
        if not awake:
            # Sleeping clears the hidden flag in the legacy store below; keep the
            # desired state consistent (a safescale commit sleeps a hidden binding).
            intent["hidden"] = False

        actions: list[dict] = []
        version = snapshot.version
        updated_binding = binding
        # A failed wake / sleep must not leave the new desired power behind
        # (review 2 P1-1 / P2-3): the guard restores it on any exception.
        with self._desired_guard([binding.binding_id], reason="binding_power_request") as guard:
            if awake and binding.awake != awake:
                self._ensure_feasible_wake(binding, snapshot.bindings)
            self._update_desired(
                {binding.binding_id: intent},
                updated_by="service-manager-api",
                reason="binding_power_request",
            )
            if binding.awake != awake:
                action = "wake" if awake else "sleep"
                self._apply_runtime_power_action(
                    binding,
                    action=action,
                    sleep_path=sleep_path,
                    drain_budget_s=drain_budget_s,
                )
                # Physically done: desired now matches the cluster; a failing
                # legacy store save below must not roll it back (review 3 P3).
                guard.settle([binding.binding_id])
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
        # The floor check below and the store save share one snapshot and run
        # under the writer lock (serialized_operation): two concurrent hides can
        # never both pass the check.
        return self._put_model_routable_locked(model, hidden_pods=hidden_pods)

    def _put_model_routable_locked(self, model: str, *, hidden_pods: list[str]) -> dict:
        self._registry.model(model)
        snapshot = self._store.load()
        model_bindings = [binding for binding in snapshot.bindings if binding.model == model]
        model_serve_ids = {binding.serve_id for binding in model_bindings}
        requested_hidden = set(hidden_pods)
        unknown = requested_hidden - model_serve_ids
        if unknown:
            raise ValueError(f"unknown pods for {model}: {sorted(unknown)}")

        # Validate first, then write desired (restored on any failure, review 2 P2-3).
        for binding in model_bindings:
            if binding.hidden and binding.serve_id not in requested_hidden and binding.awake:
                self._assert_confirmed_awake_for_unhide(binding)
        newly_hidden = {
            binding.binding_id
            for binding in model_bindings
            if not binding.hidden and binding.serve_id in requested_hidden
        }
        if newly_hidden and self._floor_enforced():
            # Replica floor (SafeScale hide): pods this request unhides are not
            # counted (they route again only once the gateway picks them up).
            check = self._floor_check(model, newly_hidden, snapshot.bindings)
            if not check.ok:
                self._on_floor_violation(check, HIDE_PATH)
        actions: list[dict] = []
        updated_by_serve = {binding.serve_id: binding for binding in snapshot.bindings}
        with self._desired_guard(
            [binding.binding_id for binding in model_bindings], reason="model_routable_request"
        ):
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
            for binding in model_bindings:
                should_hide = binding.serve_id in requested_hidden
                if binding.hidden == should_hide:
                    continue
                if not should_hide and binding.awake and not self._feasible_wake(
                    binding, list(updated_by_serve.values())
                ):
                    # Unhiding an awake binding: only the account matters (its own
                    # awake lease is on these GPUs by definition).
                    raise WakeConflict(
                        f"{binding.serve_id}: slot already has awake binding",
                        node=binding.slot.node, gpus=binding.slot.gpu_ids,
                        binding_id=binding.binding_id,
                    )
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
                self._refresh_observed(
                    updated_by_serve[item["serve_id"]].binding_id for item in actions
                )

        return {
            "model": model,
            "hidden_pods": sorted(requested_hidden),
            "version": version,
            "actions": actions,
        }


    def _assert_confirmed_awake_for_unhide(self, binding: Binding) -> None:
        """Routing is reopened only on a pod confirmed awake (review 4 P2-2): a
        pod with a sleep journal entry (a sleep in progress, ``sleep_unconfirmed``
        or a failed rollback) or whose ``/is_sleeping`` is not a clear "awake"
        stays hidden - crash recovery resolves it - and the unhide is refused
        (409, retriable)."""
        if self._sleep_primitive is not None:
            entry = self._sleep_primitive.journal.get(binding.serve_id)
            if entry is not None:
                raise RetryLater(
                    f"unhide of {binding.serve_id} refused: sleep journal phase "
                    f"{entry.get('phase')!r} (not confirmed awake)"
                )
        if self._runtime_ops is None or self._vllm_ops is None:
            return
        probe = getattr(self._vllm_ops, "is_sleeping", None)
        if not callable(probe):
            return
        snapshot = self._snapshot_for_binding(binding)
        physical = probe(snapshot.pod_ip, port=8000) if snapshot.pod_ip else None
        if physical is not False:
            raise RetryLater(
                f"unhide of {binding.serve_id} refused: /is_sleeping answered "
                f"{physical!r} (not confirmed awake)"
            )

    def defrag_enabled(self) -> bool:
        """Registry ``placement.defrag.enabled`` (default false, v1 parity)."""
        placement = getattr(self._registry, "placement", None)
        if not callable(placement):
            return False
        return bool(getattr(placement(), "defrag_enabled", False))

    def defrag(self, *, tp_size: int, force: bool = False) -> dict:
        """Manual defragmentation (``POST /v2/defrag``).

        While registry ``placement.defrag.enabled`` is false (the default, as in v1)
        this refuses with :class:`DefragDisabled` (HTTP 409, reason
        ``defrag_disabled``) before taking the writer lock, unless the caller passes
        ``force=True`` (an operator's explicit override; logged).  The controller
        never plans a defrag while it is disabled.
        """
        enabled = self.defrag_enabled()
        if not enabled and not force:
            LOG.warning(
                "defrag refused: registry placement.defrag.enabled is false "
                "(tp_size=%s; pass force=true to override)",
                tp_size,
            )
            raise DefragDisabled()
        if not enabled:
            LOG.warning(
                "defrag forced while registry placement.defrag.enabled is false (tp_size=%s)",
                tp_size,
            )
        if self._wake_journal.entries():
            raise DefragUnavailable("wake_in_progress")
        return self._defrag_serialized(tp_size=tp_size)

    @serialized_operation("defrag")
    def _defrag_serialized(self, *, tp_size: int) -> dict:
        snapshot = self._store.load()
        allocator = SlotAllocator(
            self._registry.topology(), snapshot.bindings, policy=self._placement
        )
        migrations = allocator.plan_defrag(tp_size)
        if migrations is None:
            raise DefragUnavailable("no_feasible_defrag")
        by_serve = {binding.serve_id: binding for binding in snapshot.bindings}
        deployed = self._deployed_binding_ids() if migrations else set()
        for migration in migrations:
            source = by_serve.get(migration.serve_id)
            if source is not None:
                if (
                    _defrag_destination(source, migration, snapshot.bindings) is None
                    and replace(source, slot=migration.to_slot).binding_id in deployed
                ):
                    # B6: the delete/create path would delete the source and then
                    # fail with AlreadyExists; refuse before anything changes.
                    raise DefragUnavailable("destination_deployment_without_binding")
        # The whole migration runs under the writer lock (every sleep and wake
        # does, 2026-10-02); the create + readiness wait holds it for minutes.
        return self._defrag_locked(snapshot, migrations)

    def _defrag_locked(self, snapshot, migrations) -> dict:
        """Run the migrations one by one. Desired state (and the legacy store) is
        committed per migration as it completes (review 3 P2-4): a failure
        restores only the desired records of the migration that failed, never
        those of migrations that already moved their pod."""
        if migrations:
            self._ensure_all_model_routes()

        actions: list[dict] = []
        updated_by_serve = {binding.serve_id: binding for binding in snapshot.bindings}
        version = snapshot.version
        for migration in migrations:
            binding = updated_by_serve[migration.serve_id]
            destination_id = replace(binding, slot=migration.to_slot).binding_id
            with self._desired_guard(
                [binding.binding_id, destination_id], reason="defrag"
            ) as guard:
                self._set_defrag_desired(list(updated_by_serve.values()), [migration])
                try:
                    self._run_defrag_migration(binding, migration, updated_by_serve, actions)
                except BaseException as exc:
                    if getattr(exc, "defrag_source_slept", False):
                        # P2-5: the source slept and the destination serves - the
                        # move happened; its desired state stays (the cleanup step
                        # that failed is left to the audit / supervisor).
                        guard.settle()
                    raise
                guard.settle()  # moved: its desired state stays, whatever follows
                updated = [updated_by_serve[serve_id] for serve_id in sorted(updated_by_serve)]
                version = self._store.save(updated, expected_version=version)
        return {
            "version": version,
            "migrations": [_migration_dict(migration) for migration in migrations],
            "actions": actions,
        }

    def _run_defrag_migration(self, binding, migration, updated_by_serve: dict, actions: list) -> None:
        destination = _defrag_destination(binding, migration, updated_by_serve.values())
        if destination is not None:
            # Full layout (B6): relocate by power only - no Deployment is
            # deleted or created.
            actions.extend(self._execute_power_defrag_migration(binding, destination))
            updated_by_serve[binding.serve_id] = replace(binding, awake=False, hidden=False)
            updated_by_serve[destination.serve_id] = replace(destination, awake=True, hidden=False)
            return
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

    def audit(self) -> dict:
        if self._k8s_client is None:
            raise ValueError("k8s_client is required for audit")
        result = audit_state(self._store, self._k8s_client, prober=self._pod_prober())
        issues = list(result.issues)
        if self._fleet_store is not None:
            issues.extend(self._fleet_mismatches())
            issues.extend(self._desired_coverage_issues())
        issues.extend(self._sleep_journal_issues())
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
        before = {binding.binding_id for binding in self._store.load().bindings}
        prober = self._pod_prober()
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
            # A binding being woken (journaled) is neither re-labelled routable nor
            # recorded awake by a reconcile: the wake's commit (or its recovery)
            # decides - a failed wake's compensating sleep must never hit a pod
            # the reconcile opened to traffic.
            frozen_serve_ids={
                str(entry.get("serve_id")) for entry in self._wake_journal.entries().values()
            },
        )
        self._sync_observed(result.observations)
        response = {
            "version": result.version,
            "warnings": result.warnings,
            "bindings": [self._binding_dict(binding) for binding in result.bindings],
        }
        if drop_missing and self._gpu_leases is not None:
            # A binding dropped from the store takes its GPU leases with it (in
            # this lock hold), unless a Pod of it still exists (2026-10-02: a
            # dropped binding's awake lease used to block its GPUs forever).
            dropped = before - {binding.binding_id for binding in result.bindings}
            if dropped:
                response["released_leases"] = self._release_orphan_leases(dropped)
        return response

    @serialized_operation("seed_desired")
    def seed_desired(self) -> dict:
        """Append-only desired seeding from the registry (plan D7)."""
        return self._seed_desired_unlocked()

    def _seed_desired_unlocked(self) -> dict:
        """Append-only seeding (plan D7, review P2-8): registry bindings UNION
        TRE-managed Deployments, power observed from each binding's pod."""
        if self._fleet_store is None:
            raise ValueError("desired fleet state is not configured")
        return seed_desired(
            self._registry,
            self._fleet_store,
            runtime_ops=self._runtime_ops,
            vllm_ops=self._vllm_ops,
        )

    def _seed_binding_ids(self) -> set[str]:
        return seed_binding_ids(self._registry, self._runtime_ops)

    def ensure_desired_seeded(self) -> dict | None:
        """Supervisor pass: re-seed desired state when bindings lack a record
        (e.g. the Redis holding it was wiped while the SM kept running). Seeds
        each binding from its pod's actual power, so an awake fleet stays desired
        awake and no repair starts to put it to sleep."""
        if self._fleet_store is None:
            return None
        missing = self._seed_binding_ids() - self._desired_binding_ids()
        if not missing:
            return None
        with self._writer("seed_desired"):
            result = self._seed_desired_unlocked()
        if result.get("added"):
            LOG.warning("desired state re-seeded (missing records): %s", result)
        return result

    def _desired_binding_ids(self) -> set[str]:
        if self._fleet_store is None:
            return set()
        return {item.binding_id for item in self._fleet_store.load_desired().bindings}

    def _desired_coverage_issues(self) -> list[dict]:
        """D7 audit: registry, desired state and Deployments must agree."""
        issues: list[dict] = []
        desired = {
            item.binding_id: item for item in self._fleet_store.load_desired().bindings
        }
        for binding_id in sorted(registry_binding_ids(self._registry) - set(desired)):
            issues.append(
                {"code": "registry_binding_without_desired", "binding_id": binding_id}
            )
        if self._runtime_ops is not None and hasattr(
            self._runtime_ops, "list_model_deployments"
        ):
            deployed = {
                item.binding_id for item in self._runtime_ops.list_model_deployments()
            }
            for binding_id, wanted in sorted(desired.items()):
                if wanted.lifecycle == "resident" and binding_id not in deployed:
                    issues.append(
                        {"code": "desired_without_deployment", "binding_id": binding_id}
                    )
        return issues

    def _sleep_journal_issues(self) -> list[dict]:
        """Crash evidence of the sleep primitive (plan D2).

        * ``sleep_unconfirmed``: /sleep was sent but the pod was never confirmed
          asleep; it stays hidden until :meth:`recover_sleep_journal` resolves it.
        * ``sleep_operation_orphaned``: a sleep journal entry whose operation is
          not the live writer operation. Every sleep runs under the writer lock
          (whole-lock, 2026-10-02), so such an entry was left by a dead
          service-manager (or a failed rollback); the recovery settles it.
        * ``hidden_without_operation``: a TRE pod in the ``hidden`` state that no
          sleep in progress, no SafeScale probe (desired hidden) and no startup
          admission accounts for - hidden and then abandoned. A hidden flag in the
          legacy store alone does not justify it (it is only a cache).

        One consistent snapshot: the journal is read before and after the active
        operation, and only entries present in both reads are judged (a sleep
        finishing between the reads is not an orphan).
        """
        if self._sleep_primitive is None or self._runtime_ops is None:
            return []
        primitive = self._sleep_primitive
        first = primitive.journal.entries()
        active_id = None
        if self._operation_coordinator is not None and hasattr(
            self._operation_coordinator, "active_operation"
        ):
            active = self._operation_coordinator.active_operation()
            active_id = None if active is None else active.get("operation_id")
        journal = primitive.journal.entries()
        stable = {
            pod_name: record
            for pod_name, record in journal.items()
            if pod_name in first
            and first[pod_name].get("started_at_ms") == record.get("started_at_ms")
        }
        issues: list[dict] = []
        for pod_name, record in sorted(stable.items()):
            base = {
                "serve_id": pod_name,
                "binding_id": record.get("binding_id"),
                "phase": record.get("phase"),
                "operation_id": record.get("operation_id"),
            }
            if record.get("phase") == "sleep_unconfirmed":
                issues.append({"code": "sleep_unconfirmed", **base, "reason": record.get("reason")})
                continue
            owner_op = record.get("operation_id")
            if owner_op is None or owner_op != active_id:
                issues.append({"code": "sleep_operation_orphaned", **base})
        desired_hidden: set[str] = set()
        if self._fleet_store is not None:
            desired_hidden = {
                item.binding_id
                for item in self._fleet_store.load_desired().bindings
                if item.hidden
            }
            legacy_hidden: set[str] = set()
        else:
            # Without desired state the legacy store is the only record of intent.
            legacy_hidden = {
                binding.serve_id for binding in self._store.load().bindings if binding.hidden
            }
        for snapshot in self._runtime_ops.list_pod_snapshots():
            if snapshot.annotations.get("tre.aibrix.io/state") != POD_STATE_HIDDEN:
                continue
            if snapshot.name in journal or snapshot.name in first or snapshot.name in legacy_hidden:
                continue
            if snapshot.annotations.get("tre.aibrix.io/startup-admitted-uid"):
                continue
            try:
                binding_id = _binding_from_snapshot(snapshot).binding_id
            except ValueError:
                binding_id = None
            if binding_id is not None and binding_id in desired_hidden:
                continue
            issues.append(
                {
                    "code": "hidden_without_operation",
                    "serve_id": snapshot.name,
                    "binding_id": binding_id,
                }
            )
        return issues

    def recover_sleep_journal(self) -> dict:
        """Resolve sleep journal entries left by a dead or failed sleep (review P1-2).

        Runs at SM start and on every supervisor pass, under the writer lock:
        since every sleep holds that lock from its hide to its bookkeeping
        (whole-lock, 2026-10-02), an entry seen here belongs to no sleep in
        progress - its service-manager died, or the sleep was left
        ``sleep_unconfirmed`` (no /sleep answer, no confirmation, an unreadable
        engine). The physical state decides:

        * pod gone -> nothing to undo, entry removed;
        * asleep -> ``sleeping`` annotation, GPU lease released, store and (when
          the request asked for it) desired power recorded;
        * awake -> once /sleep cannot be running any more (it was never sent, or
          the pod read awake twice ``sleep_call_timeout_s`` apart) and the engine
          is confirmed un-paused: routing restored per the entry's
          ``previous_state`` under a new route-gen (a SafeScale probe pod stays
          hidden), desired power restored;
        * unknown / unreachable -> the pod stays hidden and the entry stays
          (audit ``sleep_operation_orphaned`` / ``sleep_unconfirmed``).
        """
        if self._sleep_primitive is None or self._runtime_ops is None:
            return {"resolved": [], "kept": []}
        primitive = self._sleep_primitive
        if not primitive.journal.entries():
            return {"resolved": [], "kept": []}
        with self._writer("sleep_recovery"):
            return self._recover_sleep_journal_locked()

    def _recover_sleep_journal_locked(self) -> dict:
        primitive = self._sleep_primitive
        snapshots = {item.name: item for item in self._runtime_ops.list_pod_snapshots()}
        resolved: list[dict] = []
        kept: list[dict] = []
        touched: set[str] = set()
        for pod_name, record in sorted(primitive.journal.entries().items()):
            if record.get("binding_id"):
                touched.add(str(record["binding_id"]))
            snapshot = snapshots.get(pod_name)
            if snapshot is None:
                primitive.journal.end(pod_name)
                primitive.journal.incr("recovery_pod_gone_total")
                resolved.append({"serve_id": pod_name, "result": "pod_gone"})
                continue
            try:
                binding = _binding_from_snapshot(snapshot)
            except ValueError:
                kept.append({"serve_id": pod_name, "result": "no_binding_annotation"})
                continue
            pod_ip = snapshot.pod_ip or record.get("pod_ip")
            physical = None
            if pod_ip and self._vllm_ops is not None and hasattr(self._vllm_ops, "is_sleeping"):
                try:
                    physical = self._vllm_ops.is_sleeping(pod_ip, port=8000)
                except Exception:
                    physical = None
            if physical is True:
                self._runtime_ops.write_binding_annotations(binding, state=POD_STATE_SLEEPING)
                if self._gpu_leases is not None:
                    self._gpu_leases.release(binding)
                self._set_store_power(binding.binding_id, awake=False, hidden=False)
                if (
                    record.get("desired_on_sleep") == "sleeping"
                    and self._fleet_store is not None
                    and binding.binding_id in self._desired_binding_ids()
                ):
                    # The request's intent, recorded now that the pod IS asleep.
                    self._update_desired(
                        {binding.binding_id: {"power": "sleeping", "hidden": False}},
                        updated_by="service-manager-recovery",
                        reason="sleep_recovered_asleep",
                    )
                self._note_binding_power_change(binding)
                primitive.journal.end(pod_name)
                primitive.journal.incr("recovery_slept_total")
                resolved.append({"serve_id": pod_name, "result": "slept"})
            elif physical is False and not self._awake_long_enough(pod_name, record):
                # /sleep may still be running (no answer within its timeout): one
                # awake read is not enough to re-open routing (review 2 P3).
                kept.append({"serve_id": pod_name, "result": "awake_but_sleep_may_be_running"})
            elif physical is False and not primitive.confirm_unpaused(
                SleepTarget(binding, str(pod_ip), getattr(snapshot, "pod_uid", None))
            ):
                # Awake but possibly paused (/sleep paused the scheduler, then the
                # offload failed): never re-open routing on a frozen engine.
                kept.append({"serve_id": pod_name, "result": "awake_but_paused"})
            elif physical is False:
                previous = record.get("previous_state") or POD_STATE_AWAKE
                if previous not in (POD_STATE_AWAKE, POD_STATE_HIDDEN):
                    previous = POD_STATE_HIDDEN
                hidden = previous == POD_STATE_HIDDEN
                self._runtime_ops.write_binding_annotations(binding, state=previous)
                self._set_store_power(binding.binding_id, awake=True, hidden=hidden)
                if self._fleet_store is not None and binding.binding_id in self._desired_binding_ids():
                    self._update_desired(
                        {binding.binding_id: {"power": "awake", "hidden": hidden}},
                        updated_by="service-manager-recovery",
                        reason="sleep_recovered_awake",
                    )
                primitive.journal.end(pod_name)
                primitive.journal.incr("recovery_rolled_back_total")
                resolved.append({"serve_id": pod_name, "result": f"rolled_back_to_{previous}"})
            else:
                primitive.journal.update(
                    pod_name,
                    recovery="physical_state_unknown",
                    recovery_attempts=int(record.get("recovery_attempts") or 0) + 1,
                )
                kept.append({"serve_id": pod_name, "result": "physical_state_unknown"})
        if resolved or kept:
            LOG.warning("sleep journal recovery: resolved=%s kept=%s", resolved, kept)
        if touched:
            self._refresh_observed(touched)
        return {"resolved": resolved, "kept": kept}

    def _awake_long_enough(self, pod_name: str, record: dict) -> bool:
        """True when rolling back an awake pod is safe: /sleep was never sent
        (phase before "sleeping"), or the pod has read awake twice, more than
        ``sleep.sleep_call_timeout_s`` apart (a /sleep still running then has
        timed out). The first awake read is recorded in the journal entry."""
        if record.get("phase") not in ("sleeping", "sleep_unconfirmed", PHASE_SLEPT):
            return True
        primitive = self._sleep_primitive
        now_ms = primitive.journal.now_ms()
        first = record.get("awake_seen_at_ms")
        if first is None:
            primitive.journal.update(
                pod_name, awake_seen_at_ms=now_ms, recovery="awake_once_waiting"
            )
            return False
        return now_ms - int(first) > primitive.policy.sleep_call_timeout_s * 1000

    def _set_store_power(self, binding_id: str, *, awake: bool, hidden: bool) -> None:
        for attempt in range(3):
            snapshot = self._store.load()
            updated = [
                replace(item, awake=awake, hidden=hidden) if item.binding_id == binding_id else item
                for item in snapshot.bindings
            ]
            if updated == snapshot.bindings:
                return
            try:
                self._store.save(updated, expected_version=snapshot.version)
                return
            except StateConflict:
                if attempt == 2:
                    raise

    def begin_shutdown(self) -> None:
        """SIGTERM: accept no new sleep; a sleep waiting for its gateway ack rolls
        back at its next poll. Safe to call from a signal handler."""
        if self._sleep_primitive is not None:
            self._sleep_primitive.begin_shutdown()
        supervisor = self._supervisor
        stop_flag = getattr(supervisor, "request_stop", None)
        if callable(stop_flag):
            stop_flag()

    def shutdown_timeout_s(self) -> float:
        """Bound of the shutdown wait: the longest writer-lock hold
        (``ServiceManagerConfig.shutdown_timeout_s``) - a sleep waiting for its
        ack rolls back at its next poll, one past /sleep finishes."""
        return self._sm_config.shutdown_timeout_s()

    def shutdown(self, *, timeout_s: float) -> bool:
        """begin_shutdown, then wait (bounded) until every sleep has finished or
        rolled back. True when idle."""
        self.begin_shutdown()
        self._admission_executor.shutdown(wait=False, cancel_futures=True)
        self._wake_executor.shutdown(wait=False, cancel_futures=False)
        if self._sleep_primitive is None:
            return True
        return self._sleep_primitive.wait_idle(timeout_s)

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
        snapshot = self._store.load()
        targets = (
            sorted(set(awake_binding_ids))
            if awake_binding_ids is not None
            else self._desired_awake_binding_ids(snapshot.bindings)
        )

        def repair(operation) -> None:
            for stale_operation_id in recovered_from or []:
                operation.supersede(stale_operation_id)
            if self._fleet_store is not None:
                # D7: an empty/partial desired state would leave repaired Pods
                # behind a startup gate that rejects them; seed it first.
                seeded = self._seed_desired_unlocked()
                operation.advance("desired_seeded", details=seeded)
            self._fleet_repair.run(
                operation,
                awake_binding_ids=targets,
                desired_binding_ids=(
                    self._desired_binding_ids if self._fleet_store is not None else None
                ),
                required_binding_ids=(
                    self._seed_binding_ids if self._fleet_store is not None else None
                ),
                recreate_bindings=self._missing_registry_deployments,
                reconcile=lambda strict: self._reconcile_unlocked(
                    drop_missing=strict
                ),
                set_binding_power=self._set_binding_power_by_id_unlocked,
                audit=self.audit,
            )

        def run(operation) -> None:
            # The SM maintenance lock (not the controller mode) marks the repair
            # for its whole run; clearing it aborts the repair (2026-09-28).
            # A repair recovering the stale repairs of a dead SM takes over
            # their lock (if it has not expired yet); any other live holder
            # refuses it (MaintenanceLockBusy).
            self._safety_gate.acquire_maintenance(
                operation.operation_id, kind="fleet_repair",
                owner=str(getattr(self._operation_coordinator, "owner", "")),
                takeover_operation_ids=list(recovered_from or []),
            )
            try:
                repair(operation)
            finally:
                try:
                    self._safety_gate.release_maintenance(operation.operation_id)
                except Exception:  # a stale lock is taken over by the next repair
                    LOG.exception("releasing the SM maintenance lock of %s failed", operation.operation_id)

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

    def supersede_stale_operations(self) -> list[str]:
        """Supervisor pass: operation records left ``running`` by a writer whose
        lock lease expired (its service-manager died mid-operation) are marked
        ``superseded`` - fleet repairs excepted (:meth:`recover_stale_fleet_repairs`
        resumes them). What such an operation left half-done is settled by the
        sleep / wake journal recoveries; the record only stops claiming to run."""
        coordinator = self._operation_coordinator
        lister = getattr(coordinator, "stale_running_operations", None)
        if coordinator is None or not callable(lister):
            return []
        if not [record for record in lister() if record.get("kind") != "fleet_repair"]:
            return []
        superseded: list[str] = []
        with self._writer("supersede_stale_operations", wait_s=0.0) as operation:
            for record in lister():
                if record.get("kind") == "fleet_repair" or not record.get("operation_id"):
                    continue
                if operation is not None:
                    operation.supersede(str(record["operation_id"]))
                superseded.append(str(record["operation_id"]))
        if superseded:
            _log_event("stale_operations_superseded", level=logging.WARNING, operation_ids=superseded)
        return superseded

    def recover_stale_fleet_repairs(self, *, actuate: bool = True) -> dict | None:
        """Supervisor pass: resume a fleet repair a dead SM left running. With
        ``actuate=False`` (SM actuation observe) it is only recorded."""
        if self._operation_coordinator is None:
            return None
        if self._operation_coordinator.active_operation() is not None:
            return None
        stale = self._operation_coordinator.stale_running_operations(
            kind="fleet_repair"
        )
        if not stale:
            return None
        if not actuate:
            self.record_suppressed(
                "fleet_repair_recovery",
                {"stale_operation_ids": [str(record.get("operation_id")) for record in stale]},
            )
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

    def _missing_registry_deployments(self, *, power: str | None = None) -> list[Binding]:
        """B7: registry bindings desired ``resident`` (and ``power``, if given)
        whose Deployment is gone. Only the registry rendering can be recreated
        faithfully; fleet repair creates these again instead of failing its
        final audit with ``desired_without_deployment`` forever."""
        if self._fleet_store is None or not callable(
            getattr(self._runtime_ops, "create_model_deployment", None)
        ):
            return []
        deployed = self._deployed_binding_ids()
        registry_ids = registry_binding_ids(self._registry)
        return [
            Binding(item.binding_id, item.model, Slot(item.node, tuple(item.gpu_ids)), awake=False)
            for item in sorted(
                self._fleet_store.load_desired().bindings, key=lambda item: item.binding_id
            )
            if item.lifecycle == "resident"
            and (power is None or item.power == power)
            and item.binding_id in registry_ids
            and item.binding_id not in deployed
        ]

    def repair_missing_deployments(self, binding_ids, *, actuate: bool = True) -> dict | None:
        """Supervisor pass (B7): drift that is ONLY ``deployment_missing`` of
        registry bindings desired resident + sleeping is repaired by creating
        just those Deployments again from the registry - no fleet-wide
        quarantine. Their Pods go through the normal startup gate (which sleeps
        the overlapping residents for the start, and wakes them again once the
        new Pod converged asleep), like a Pod k8s restarted. Returns None when
        the drift does not qualify: the caller then runs a full fleet repair."""
        wanted = set(binding_ids)
        if not wanted:
            return None
        if not wanted <= {item.binding_id for item in self._missing_registry_deployments(power="sleeping")}:
            return None
        if not actuate:
            # SM actuation observe (2026-09-28): recreating a workload is not a
            # state-consistency action - record what would have been done.
            self.record_suppressed("recreate_missing_deployments", {"binding_ids": sorted(wanted)})
            return {"binding_ids": sorted(wanted), "created": [], "suppressed": True}
        created: list[str] = []
        with self._writer("deployment_repair", wait_s=0.0):
            # Again under the lock: another writer may have changed them meanwhile.
            for planned in self._missing_registry_deployments(power="sleeping"):
                if planned.binding_id not in wanted:
                    continue
                created.append(
                    str(self._runtime_ops.create_model_deployment(planned.model, planned.slot))
                )
                LOG.warning(
                    "recreated the missing Deployment of %s (desired resident, sleeping)",
                    planned.binding_id,
                )
        return {"binding_ids": sorted(wanted), "created": created}

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
        admitted_startups = self._runtime_ops.list_admitted_startup_pods()
        admitted_ids = {pod.binding_id for pod in admitted_startups}
        gates = self._startup_gates_waiting()
        grace_s = float(getattr(self._sm_config, "startup_gate_drift_grace_s", 0.0) or 0.0)
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
            # Review 2026-09-29 P1-1: a Pod whose startup gate is polling right now
            # (same UID, asked within startup_admission.gate_seen_s) is waiting for
            # its admission, not drifted - a fleet repair would only restart the
            # wait. Informational (the supervisor never repairs on it) for up to
            # drift_grace_s since its first request; a gate waiting longer is
            # reported as drift again, so a stuck admission is not masked.
            gated = [
                (pod, gates[pod.name][1])
                for pod in pods
                if pod.name in gates
                and pod.pod_uid
                and gates[pod.name][0] == pod.pod_uid
                and not pod.annotations.get("tre.aibrix.io/startup-admitted-uid")
            ]
            if gated:
                waited_s = max(waited for _pod, waited in gated)
                if waited_s <= grace_s:
                    issues.append(
                        {
                            "code": "startup_admission_pending",
                            "binding_id": binding_id,
                            "pods": sorted(pod.name for pod, _waited in gated),
                            "waited_s": round(waited_s, 1),
                            "informational": True,
                        }
                    )
                    continue
            if len(pods) != 1:
                if binding_id in admitted_ids or any(
                    pod.node == desired.node
                    and set(pod.gpu_ids).intersection(desired.gpu_ids)
                    for pod in admitted_startups
                ):
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

        # Pre-authorized admission (review 4 P1): a writer that creates a gated
        # Pod itself (fleet repair, a defrag migration, a cold start) holds the
        # writer lock until that Pod is ready, so the admission below could never
        # get the lock - the init gate and the creator would wait on each other
        # until the creator timed out. The creator records the operation phase
        # ``starting_binding`` (binding id, later the Pod UID) and holds the
        # binding's ``starting`` GPU lease BEFORE it creates the Pod; exactly that
        # binding is admitted here without the writer lock.
        active = self._operation_coordinator.active_operation()
        if active is not None and active.get("phase") == STARTING_BINDING_PHASE:
            details = active.get("details") or {}
            expected_uid = details.get("pod_uid")
            if (
                details.get("binding_id") == pod.binding_id
                and (not expected_uid or expected_uid == pod.uid)
                and self._lease_matches(pod.binding_id, phase="starting")
            ):
                return self._admit_pre_authorized(pod, active)
            raise OperationBusy(
                f"{active.get('owner')}:{active.get('fencing_token')}"
            )
        if active is not None and active.get("kind") == "fleet_repair":
            # A repair converging the fleet: no other startup until it is done.
            raise OperationBusy(
                f"{active.get('owner')}:{active.get('fencing_token')}"
            )

        # Awake residents on the Pod's GPUs are put to sleep first, in their own
        # writer-lock hold (the whole sleep under the lock, 2026-10-02); the
        # admission below only verifies that they are asleep. Checks that would
        # refuse the admission anyway run first, WITHOUT the lock (review 3 P2-5:
        # pressure, desired lifecycle, a starting lease or a journaled wake on the
        # Pod's GPUs), so nothing is slept for a Pod that cannot start now.
        self._safety_gate.assert_no_pressure()
        if pod.binding_id in self._desired_binding_ids():
            if self._desired_binding(pod.binding_id).lifecycle != "resident":
                raise ValueError(
                    f"startup denied for non-resident desired binding {pod.binding_id}"
                )
        self._assert_startup_slot_free(pod)
        self._assert_startup_owner_live(pod)
        self._assert_unrequested_startup_allowed(pod)
        slept: list[Binding] = []
        try:
            slept = self._sleep_overlapping_residents(pod)
            return self._admit_startup_locked(pod)
        except BaseException as exc:
            if not slept and isinstance(exc, SleepFailed):
                slept = [
                    _binding_from_outcome(outcome, pod)
                    for outcome in (exc.outcomes or [])
                    if outcome.get("status") == STATUS_SLEPT
                ]
            if slept:
                # The admission failed after residents were put to sleep for it:
                # wake them back so desired (awake) and physical agree again,
                # instead of leaving them asleep for a Pod that did not start.
                self._restore_failed_admission_residents(pod, slept)
            raise

    def _admit_pre_authorized(self, pod: StartupPodRecord, active: dict) -> dict:
        """Admit the Pod a writer-lock holder is starting (review 4 P1), without
        the writer lock: the holder already owns the binding's ``starting``
        lease and fences every other writer. The checks that need no lock still
        run: node pressure, the desired lifecycle, other starting leases /
        journaled wakes on the Pod's GPUs, and overlapping residents must be
        physically asleep. Nothing is slept here."""
        self._safety_gate.assert_no_pressure()
        if pod.binding_id in self._desired_binding_ids():
            if self._desired_binding(pod.binding_id).lifecycle != "resident":
                raise ValueError(
                    f"startup denied for non-resident desired binding {pod.binding_id}"
                )
        self._assert_startup_slot_free(pod)
        self._assert_startup_owner_live(pod)
        self._assert_startup_overlaps_sleeping(pod)
        self._runtime_ops.admit_startup_pod(
            pod.name,
            pod_uid=pod.uid,
            suspended_binding_ids=[],
            operation_id=str(active["operation_id"]),
        )
        result = {
            "status": "admitted",
            "binding_id": pod.binding_id,
            "operation_id": active["operation_id"],
            "pre_authorized": True,
        }
        if active.get("kind") == "fleet_repair":
            result["prepared_by_fleet_repair"] = True
        return result

    @contextmanager
    def _starting_binding(self, planned: Binding):
        """Pre-authorize the gated Pod of ``planned`` (review 4 P1): the current
        writer operation enters phase ``starting_binding`` (the caller holds the
        binding's ``starting`` lease) so the Pod's init gate is admitted without
        the writer lock; left again when the start finished or failed. Yields a
        callable that records the Pod UID once known (then only that Pod is
        admitted)."""
        operation = current_operation()
        details = {"binding_id": planned.binding_id}

        def note_pod(pod_uid: str | None) -> None:
            if operation is not None and pod_uid:
                operation.advance(
                    STARTING_BINDING_PHASE, details={**details, "pod_uid": pod_uid}
                )

        if operation is not None:
            operation.advance(STARTING_BINDING_PHASE, details=details)
        try:
            yield note_pod
        finally:
            if operation is not None:
                try:
                    operation.advance("executing", details={"started_binding_id": planned.binding_id})
                except Exception:  # a lost fence surfaces at the operation's end
                    LOG.warning("leaving phase starting_binding of %s failed", planned.binding_id)

    def _discard_failed_start(self, planned: Binding, deployment_created: bool) -> None:
        """A cold start / defrag destination failed after its Deployment was
        created (review 4 P1): its desired record is rolled back (absent), so a
        Deployment left behind would only make its Pod's init gate loop on 400
        "non-resident desired binding". Delete it (best effort; the supervisor's
        ``reap_rejected_deployments`` deletes what is left).

        The ``starting`` lease is NOT released here (I1, 2026-10-04): an accepted
        delete is no evidence - the Pod may still be loading or awake on those
        GPUs (or its /wake_up may not have answered). The binding becomes a
        suspect (its GPUs never trusted from gpu-truth; the caller requests a
        fresh sample): the suspect convergence releases the lease once the
        engine reads asleep with no wake in flight (or sleeps it per desired),
        ``reap_orphan_leases`` once no Pod of the binding exists."""
        if deployment_created and hasattr(self._runtime_ops, "delete_model_deployment"):
            try:
                self._runtime_ops.delete_model_deployment(planned)
            except Exception:
                LOG.exception(
                    "deleting the Deployment of the failed start %s failed; "
                    "the supervisor reaps it", planned.binding_id,
                )
        if self._gpu_leases is not None:
            self._suspects[planned.binding_id] = (
                planned.slot.node, tuple(int(g) for g in planned.slot.gpu_ids), ""
            )
            _log_event(
                "failed_start_lease_kept", level=logging.WARNING,
                binding_id=planned.binding_id, node=planned.slot.node,
                gpu_ids=list(planned.slot.gpu_ids), deployment_created=deployment_created,
                detail="starting lease kept until the Pod is gone or reads asleep; binding now a suspect",
            )

    def _finish_admitted_start(self, pod_name: str) -> None:
        """The creator converged its own Pod (awake, annotated, leased): clear
        the startup admission so the supervisor does not converge it again."""
        clear = getattr(self._runtime_ops, "clear_startup_admission", None)
        if not callable(clear):
            return
        try:
            clear(pod_name)
        except Exception:  # the supervisor's convergence clears it later
            LOG.warning("clearing the startup admission of %s failed", pod_name)

    def reap_rejected_deployments(self, *, actuate: bool = True) -> list[str]:
        """Supervisor pass (review 4 P1): delete model Deployments whose binding
        is desired ``absent`` and that have no Running Pod - e.g. left by a
        failed cold start / defrag whose cleanup did not go through. Their Pods
        sit in the init gate, refused forever. Under the writer lock (no start
        is in progress then); a busy lock skips the pass."""
        if (
            self._runtime_ops is None
            or self._fleet_store is None
            or not hasattr(self._runtime_ops, "list_model_deployments")
            or not hasattr(self._runtime_ops, "delete_model_deployment")
        ):
            return []
        desired = {item.binding_id: item for item in self._fleet_store.load_desired().bindings}
        candidates = [
            item
            for item in self._runtime_ops.list_model_deployments()
            if int(getattr(item, "replicas", 1) or 0) > 0
            and item.binding_id in desired
            and desired[item.binding_id].lifecycle == "absent"
        ]
        if not candidates:
            return []
        if not actuate:
            # SM actuation observe (2026-09-28): deleting workloads only logged.
            self.record_suppressed(
                "reap_rejected_deployments",
                {"deployments": sorted(item.name for item in candidates)},
            )
            return []
        reaped: list[str] = []
        with self._writer("reap_rejected_deployments", wait_s=0.0):
            desired = {item.binding_id: item for item in self._fleet_store.load_desired().bindings}
            running = set()
            for snapshot in self._runtime_ops.list_pod_snapshots():
                try:
                    running.add(_binding_from_snapshot(snapshot).binding_id)
                except ValueError:
                    continue
            for item in candidates:
                wanted = desired.get(item.binding_id)
                if wanted is None or wanted.lifecycle != "absent" or item.binding_id in running:
                    continue
                binding = Binding(item.name, item.model, Slot(item.node, tuple(item.gpu_ids)), awake=False)
                self._runtime_ops.delete_model_deployment(binding)
                LOG.warning(
                    "reaped Deployment %s: binding %s is desired absent and has no Running Pod",
                    item.name, item.binding_id,
                )
                reaped.append(item.name)
        return reaped

    def _assert_startup_slot_free(self, pod: StartupPodRecord) -> None:
        """Refuse a startup whose GPUs are leased by another starting binding or
        hold a journaled wake (checked again under the writer lock)."""
        conflict = self._conflicting_transient_lease(pod)
        if conflict is not None:
            gpu_id, occupant = conflict
            raise GpuLeaseConflict(gpu=f"{pod.node}/{gpu_id}", occupant=occupant)
        for binding_id, entry in self._wake_journal.entries().items():
            if binding_id == pod.binding_id or entry.get("node") != pod.node:
                continue
            try:
                overlap = sorted(set(pod.gpu_ids) & {int(gpu) for gpu in entry.get("gpu_ids") or ()})
            except (TypeError, ValueError):
                continue
            if overlap:
                raise GpuLeaseConflict(gpu=f"{pod.node}/{overlap[0]}", occupant=binding_id)

    def _assert_startup_owner_live(self, pod: StartupPodRecord) -> None:
        """Refuse (retriable 409, the gate polls again) a Pod whose owner chain
        is not live (B11): the Pod is being deleted, or its ReplicaSet /
        Deployment is gone, being deleted or was replaced. A failed cold start
        deletes its Deployment, but the ReplicaSet may already have created a
        replacement Pod; admitting it would take a ``starting`` lease for a Pod
        that dies with its Deployment. Fails closed when the owner chain cannot
        be read."""
        check = getattr(self._runtime_ops, "startup_owner_problem", None)
        if not callable(check):
            return
        try:
            problem = check(pod.name, pod.uid)
        except Exception as exc:
            raise RetryLater(
                f"cannot verify the owning Deployment of {pod.name}: {exc}; "
                "retry the admission"
            ) from exc
        if problem:
            raise RetryLater(f"startup denied for {pod.name}: {problem}")

    def reap_orphan_leases(self) -> list[str]:
        """Supervisor pass: release every GPU lease - ``starting`` (B11) or
        ``awake`` (2026-10-02) - whose binding has no Pod object any more.
        Outside a writer operation no write is half-way (whole-lock: a creator
        holds the lock from before its Deployment exists until its Pod
        converged), so such a lease is an orphan that would refuse every start
        and wake on its GPUs forever (409 lease_conflict naming a Pod that is
        gone): a starting lease of a Pod that died in its gate, an awake lease
        of a Pod deleted with its Deployment. The Pod list unreadable -> nothing
        released (fail closed). Under the writer lock (wait 0, re-checked
        there); a busy lock skips the pass. One event per release:
        ``orphan_awake_lease_released`` / ``orphan_starting_lease_released``."""
        if self._gpu_leases is None or not self._gpu_leases.load():
            return []
        if not self._orphan_leases():
            return []
        with self._writer("reap_orphan_leases", wait_s=0.0):
            return self._release_orphan_leases()

    def _orphan_leases(self, binding_ids=None) -> list:
        """Leases (of ``binding_ids``, None = all) whose binding has no live Pod
        object; [] when the Pod list cannot be read (fail closed)."""
        lister = getattr(self._runtime_ops, "list_live_model_pod_binding_ids", None)
        if self._gpu_leases is None or not callable(lister):
            return []
        leases = [
            lease for lease in self._gpu_leases.load()
            if binding_ids is None or lease.binding_id in binding_ids
        ]
        if not leases:
            return []
        try:
            live = set(lister())
        except Exception:  # noqa: BLE001 - unreadable: keep every lease
            LOG.warning("listing the model Pods failed; no orphan GPU lease released", exc_info=True)
            return []
        return [lease for lease in leases if lease.binding_id not in live]

    def _release_orphan_leases(self, binding_ids=None) -> list[str]:
        """Release the orphan leases (writer lock held); returns their binding ids."""
        stored = {binding.binding_id for binding in self._store.load().bindings}
        released: list[str] = []
        for lease in self._orphan_leases(binding_ids):
            model = lease.binding_id.split("/", 1)[0]
            self._gpu_leases.release(
                Binding("orphan-lease", model, Slot(lease.node, tuple(lease.gpu_ids)), awake=False)
            )
            self._note_power_change(lease.node, lease.gpu_ids)
            _log_event(
                "orphan_awake_lease_released" if lease.phase == "awake" else f"orphan_{lease.phase}_lease_released",
                level=logging.WARNING,
                binding_id=lease.binding_id, node=lease.node, gpu_ids=list(lease.gpu_ids),
                phase=lease.phase, in_store=lease.binding_id in stored,
                detail="no Pod object of the binding exists",
            )
            released.append(lease.binding_id)
        return released

    def _restore_failed_admission_residents(
        self, pod: StartupPodRecord, slept: list[Binding]
    ) -> None:
        """Best effort, under the writer lock (queued): wake every resident this
        failed admission put to sleep that is still desired awake. Whatever
        cannot be restored stays visible to the audit (desired awake, asleep)."""
        try:
            with self._writer("startup_admit_restore"):
                for binding in slept:
                    try:
                        wanted = self._desired_binding(binding.binding_id)
                        if wanted.lifecycle == "resident" and wanted.power == "awake":
                            # Replica floor swap (P2-3): a make-up replica woken for
                            # it stays awake instead of the resident waking back.
                            if self._swap_with_floor_makeup(binding.binding_id, pod.uid):
                                continue
                            self._restore_desired_awake(binding.binding_id)
                    except Exception:
                        LOG.exception(
                            "waking %s back after the failed startup admission of %s failed",
                            binding.binding_id, pod.name,
                        )
        except Exception:
            LOG.exception(
                "could not restore the residents %s after the failed startup admission of %s",
                [binding.binding_id for binding in slept], pod.name,
            )

    def _admit_startup_locked(self, pod: StartupPodRecord) -> dict:
        pod_name = pod.name
        request = {"pod_name": pod_name, "pod_uid": pod.uid}
        # Queue for the writer lock like any other writer (review 3 P2-5): a
        # zero wait made the admission fail (after the residents slept) whenever
        # another short writer phase happened to hold it.
        with self._operation_coordinator.operation(
            "startup_admit", request=request, wait_s=self._sm_config.writer_lock_wait_s
        ) as operation:
            operation.advance("validating_startup", details={"binding_id": pod.binding_id})
            # Checked again under the writer lock (review 2 P3): no sleep can start
            # and no transient lease can appear between the check and the admission.
            self._assert_startup_slot_free(pod)
            self._safety_gate.assert_no_pressure()
            # B11: checked again under the lock - a writer that failed a cold
            # start deletes the Deployment under this lock, so an admission that
            # queued behind it sees the Deployment gone here, before any lease.
            self._assert_startup_owner_live(pod)
            if pod.binding_id not in self._desired_binding_ids():
                # Redis lost desired state while the SM kept running: re-seed
                # from the registry (append-only). A Pod whose binding is not in
                # the registry still has no record and is still refused below.
                self._seed_desired_unlocked()
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
                    raise RetryLater(
                        f"overlapping resident {binding.binding_id} is awake again; "
                        "retry the admission"
                    )
                current = updated.get(binding.binding_id)
                if current is not None and current.awake:
                    updated[binding.binding_id] = replace(current, awake=False, hidden=False)
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
            # S2: the placeholder holds the GPUs until the Pod converged or is gone.
            _log_event(
                "startup_placeholder",
                binding_id=planned.binding_id, pod=pod.name, node=pod.node,
                gpu_ids=list(pod.gpu_ids), since=int(time.time() * 1000),
                operation_id=operation.operation_id,
            )
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

    def _assert_unrequested_startup_allowed(self, pod: StartupPodRecord) -> None:
        """Startup admission WITHOUT an owning operation (no SM operation or API
        caller is starting this Pod - k8s restarted it, a Deployment was
        applied or scaled by hand): with SM actuation observe (2026-09-28) it
        must not sleep awake residents on the Pod's GPUs - that changes the
        awake count nobody asked for. Refused (retriable 409; the init gate
        polls again, so the Pod is admitted once the residents are asleep or
        the actuation is active again) and recorded. Without an awake
        overlapping resident the admission sleeps nothing and proceeds. The
        admission of a Pod an operation creates itself (cold start, defrag
        migration, fleet repair: phase ``starting_binding``) is pre-authorized
        and never reaches this check."""
        if not self.actuation_observe():
            return
        self._refuse_unrequested_startup_sleep(pod, self._awake_overlapping_residents(pod))

    def _refuse_unrequested_startup_sleep(self, pod: StartupPodRecord, awake: list[Binding]) -> None:
        """The observe refusal of :meth:`_assert_unrequested_startup_allowed`
        for residents about to be slept (retriable 409, recorded)."""
        if not awake or not self.actuation_observe():
            return
        ids = sorted(binding.binding_id for binding in awake)
        self.record_suppressed(
            "startup_admission_sleep",
            {"pod": pod.name, "binding_id": pod.binding_id, "would_sleep": ids},
        )
        raise RetryLater(
            f"startup of {pod.name} would sleep awake resident(s) {ids}: SM actuation is observe "
            "and no operation requested this Pod; retry the admission"
        )

    def _awake_overlapping_residents(self, pod: StartupPodRecord) -> list[Binding]:
        """Ready residents on the startup Pod's GPUs that vLLM reports awake."""
        awake: list[Binding] = []
        target_gpus = set(pod.gpu_ids)
        for snapshot in self._runtime_ops.list_startup_resident_snapshots():
            if snapshot.name == pod.name or snapshot.node != pod.node:
                continue
            binding = _binding_from_snapshot(snapshot)
            if not target_gpus.intersection(binding.slot.gpu_ids):
                continue
            if not snapshot.pod_ip or not snapshot.ready:
                continue  # the admission refuses it under the lock
            if self._vllm_ops.is_sleeping(snapshot.pod_ip, port=8000) is False:
                awake.append(binding)
        return awake

    def _sleep_overlapping_residents(self, pod: StartupPodRecord) -> list[Binding]:
        """Sleep every awake resident on the startup Pod's GPUs in one writer-lock
        hold; returns the bindings put to sleep. Desired power is untouched: a
        resident desired awake is recorded as suspended by the admission and
        woken after convergence."""
        if self._sleep_primitive is None:
            return []
        awake = self._awake_overlapping_residents(pod)
        if not awake:
            return []
        # The observe check is repeated under the sleep's writer lock, right
        # before the residents are hidden: the actuation may have been switched
        # to observe since the unlocked pre-check in admit_startup (TOCTOU).
        # Only unrequested admissions get here - an owned (pre-authorized)
        # admission never sleeps residents through this path.
        #
        # Replica floor (review 2026-09-29 P1-1 / P2-4): another replica of a
        # resident's model is woken first, best effort, in its OWN writer phase
        # (never under the sleep's lock below). The sleep's floor check then
        # only re-checks: a model still below its floor is exempt and recorded
        # (path startup) - the admission never waits on the floor.
        notes = self._startup_floor_makeup(pod, awake)
        token = _FLOOR_EVENT_NOTES.set(notes)
        try:
            outcomes = self._locked_sleep(
                awake,
                sleep_path="startup",
                kind="startup_admit_sleep",
                before_prepare=lambda: self._refuse_unrequested_startup_sleep(pod, awake),
            )
        finally:
            _FLOOR_EVENT_NOTES.reset(token)
        slept_ids = {item.get("binding_id") for item in outcomes if item.get("status") == STATUS_SLEPT}
        return [binding for binding in awake if binding.binding_id in slept_ids]

    # ------------------------------------------------ async startup admission
    def request_startup_admission(self, *, pod_name: str, pod_uid: str) -> tuple[int, dict]:
        """Init-gate entry point (review 3 P2-5). An admission may sleep an
        overlapping resident and queue for the writer lock for longer than the
        gate's 15 s HTTP timeout, so it runs as a background job keyed by (pod, UID): the first
        call starts it and answers 202 "in_progress" unless it finishes within
        ``ADMISSION_SYNC_WAIT_S``; later calls answer 202 while it runs, then
        its result (200) or its error (raised: 409 / 503 / 400 like the
        synchronous call) exactly once. The existing gate script retries until
        it gets a 200, so it needs no change."""
        key = (pod_name, pod_uid)
        now = time.monotonic()
        self._note_startup_gate(pod_name, pod_uid, now)
        with self._admission_lock:
            for stale_key, (_job, started) in list(self._admission_jobs.items()):
                if _job.done() and now - started > ADMISSION_RESULT_TTL_S:
                    self._admission_jobs.pop(stale_key, None)
            entry = self._admission_jobs.get(key)
            if entry is None:
                try:
                    job = self._admission_executor.submit(
                        self.admit_startup, pod_name=pod_name, pod_uid=pod_uid
                    )
                except RuntimeError as exc:  # executor shut down (SIGTERM)
                    raise ServiceShuttingDown(
                        "service-manager is shutting down; retry the startup admission"
                    ) from exc
                self._admission_jobs[key] = (job, now)
            else:
                job = entry[0]
        if entry is None:
            futures_wait([job], timeout=ADMISSION_SYNC_WAIT_S)
        if not job.done():
            return 202, {"status": "in_progress", "pod_name": pod_name, "pod_uid": pod_uid}
        with self._admission_lock:
            self._admission_jobs.pop(key, None)
        if job.cancelled():  # cancelled by the shutdown: retriable (503), not a 500
            raise ServiceShuttingDown(
                "service-manager is shutting down; retry the startup admission"
            )
        result = job.result()  # raises the admission's error (409 / 503 / 400)
        with self._admission_lock:
            seen = self._startup_gate_seen.get(pod_name)
            if seen is not None and seen[0] == pod_uid:
                self._startup_gate_seen.pop(pod_name, None)  # admitted: no longer waiting
        return 200, result

    def _note_startup_gate(self, pod_name: str, pod_uid: str, now: float) -> None:
        """Record that ``pod_name``'s startup gate asked for admission (first and
        last request per Pod UID); prunes gates not seen for a while."""
        seen_s = float(getattr(self._sm_config, "startup_gate_seen_s", 0.0) or 0.0)
        keep_s = max(4.0 * seen_s, 300.0)
        with self._admission_lock:
            for name, (_uid, _first, last) in list(self._startup_gate_seen.items()):
                if now - last > keep_s:
                    self._startup_gate_seen.pop(name, None)
            entry = self._startup_gate_seen.get(pod_name)
            first = entry[1] if entry is not None and entry[0] == pod_uid else now
            self._startup_gate_seen[pod_name] = (pod_uid, first, now)

    def _startup_gates_waiting(self) -> dict[str, tuple[str, float]]:
        """Pods whose startup gate is polling now: pod name -> (UID, seconds since
        its first admission request). A gate counts only while it asked within
        ``startup_admission.gate_seen_s`` (a Pod that stopped polling - crashed,
        deleted, admitted elsewhere - is not waiting)."""
        seen_s = float(getattr(self._sm_config, "startup_gate_seen_s", 0.0) or 0.0)
        now = time.monotonic()
        with self._admission_lock:
            return {
                name: (uid, now - first)
                for name, (uid, first, last) in self._startup_gate_seen.items()
                if now - last <= seen_s
            }

    def converge_startups(self) -> dict:
        """Converge admitted Pods after vLLM becomes reachable.

        A Pod desired asleep is slept first (its own writer-lock hold). A Pod
        that cannot be converged right now (writer busy, a failed sleep) stays
        pending for the next pass; it never aborts the other Pods of this pass
        (review 2 P3)."""
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
                if physical_sleeping is False and self._startup_wants_sleep(snapshot):
                    self._locked_sleep(
                        [_binding_from_snapshot(snapshot)],
                        sleep_path="startup",
                        kind="startup_converge_sleep",
                    )
                    physical_sleeping = self._vllm_ops.is_sleeping(
                        snapshot.pod_ip, port=8000
                    )
                    if physical_sleeping is not True:
                        pending.append(snapshot.name)
                        continue
                self._converge_startup(snapshot, physical_sleeping)
            except (OperationBusy, RetryLater, SleepFailed, ServiceShuttingDown):
                pending.append(snapshot.name)
                continue
            except ValueError as exc:  # e.g. no desired record: report, keep going
                LOG.warning("startup convergence of %s deferred: %s", snapshot.name, exc)
                pending.append(snapshot.name)
                continue
            converged.append(snapshot.name)
        return {"converged": converged, "pending": pending}

    def _startup_wants_sleep(self, snapshot: K8sPodSnapshot) -> bool:
        if self._fleet_store is None:
            return False
        return self._desired_binding(_binding_from_snapshot(snapshot).binding_id).power == "sleeping"

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
                # converge_startups sleeps it first (its own writer-lock hold).
                raise RetryLater(f"startup {binding.binding_id} is still awake")
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

        self._note_binding_power_change(binding)
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
                # Replica floor swap (P2-3): its make-up replica stays awake instead.
                if self._swap_with_floor_makeup(binding_id, snapshot.pod_uid):
                    continue
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

    def _conflicting_transient_lease(
        self, pod: StartupPodRecord
    ) -> tuple[int, str] | None:
        if self._gpu_leases is None:
            return None
        target_gpus = set(pod.gpu_ids)
        for lease in self._gpu_leases.load():
            if (
                lease.binding_id != pod.binding_id
                and lease.node == pod.node
                and lease.phase in TRANSIENT_LEASE_PHASES
            ):
                overlap = sorted(target_gpus.intersection(lease.gpu_ids))
                if overlap:
                    return overlap[0], lease.binding_id
        return None

    def _assert_startup_overlaps_sleeping(self, pod: StartupPodRecord) -> None:
        target_gpus = set(pod.gpu_ids)
        for snapshot in self._runtime_ops.list_startup_resident_snapshots():
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
            matches[0].serve_id, awake=awake, sleep_path="repair"
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
            if _defrag_destination(actual, migration, bindings) is not None:
                # Full layout (B6): the destination binding (and its Deployment)
                # already exists, the source only goes to sleep - it stays
                # resident, so the registry layout does not shrink.
                by_id[old_id] = old.with_intent(
                    lifecycle="resident",
                    power="sleeping",
                    hidden=False,
                    updated_by="service-manager-api",
                    reason="defrag_source_slept",
                )
            else:
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

    def _sync_observed(self, observations, *, only=None) -> None:
        """Write the observed fleet state (read by the audit). ``only=None``
        (reconcile): the observations replace the whole snapshot. ``only`` = a
        set of binding ids (after a power operation): only those records are
        replaced by ``observations`` - a binding without an observation (its
        pod is gone) loses its record - and every other record is kept."""
        if self._fleet_store is None:
            return
        fresh = [_observed_record(observation) for observation in observations]
        for attempt in range(3):
            snapshot = self._fleet_store.load_observed()
            if only is None:
                records = list(fresh)
            else:
                records = [item for item in snapshot.bindings if item.binding_id not in only]
                records.extend(item for item in fresh if item.binding_id in only)
            records.sort(key=lambda item: item.binding_id)
            if records == snapshot.bindings:
                return
            try:
                self._fleet_store.save_observed(records, expected_version=snapshot.version)
                return
            except FleetStateConflict:
                if only is None or attempt == 2:
                    raise

    def _refresh_observed(self, binding_ids) -> None:
        """Targeted observe after a power operation (B2): re-read the pods of
        ``binding_ids`` (k8s + physical /is_sleeping, the reconcile's own
        observation code) and replace only their observed records, so the
        audit does not report the pre-operation power (desired_power_mismatch,
        gpu_lease_*) until the next full reconcile. Best effort: a failure is
        logged and left to the next reconcile; it never fails the operation."""
        ids = {binding_id for binding_id in binding_ids if binding_id}
        if not ids or self._fleet_store is None or self._k8s_client is None:
            return
        try:
            observations = observe_bindings(self._k8s_client, ids, prober=self._pod_prober())
            self._sync_observed(observations, only=ids)
        except Exception:
            LOG.warning(
                "refreshing the observed state of %s failed; the next reconcile does it",
                sorted(ids), exc_info=True,
            )

    def _pod_prober(self):
        if self._vllm_ops is not None and hasattr(self._vllm_ops, "is_sleeping"):
            return _VllmPodProber(self._vllm_ops)
        return None

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

    def _apply_runtime_power_action(
        self,
        binding: Binding,
        *,
        action: str,
        sleep_path: str = "default",
        drain_budget_s: float | None = None,
    ) -> None:
        if self._runtime_ops is None or self._vllm_ops is None:
            return
        if action == "sleep":
            # Plan D2: the ordering hide -> gateway ack -> /sleep -> confirmation
            # lives in the primitive; there is no other way to sleep a pod.
            self._sleep_bindings(
                [binding], sleep_path=sleep_path, drain_budget_s=drain_budget_s
            )
            return
        if action != "wake":
            raise ValueError(f"unknown runtime action: {action}")
        self._wake_inline(binding)

    # ------------------------------------------------------------------- wake
    # A wake runs under the writer lock from its first check to its commit
    # (whole-lock, 2026-10-02; the S6 split of 2026-09-30 is gone):
    #   1. prepare: the account (store + GPU leases + journaled wakes), the wake
    #      gate (S1), then the binding's ``awake`` GPU lease (the GPUs are taken
    #      from here on) and a wake journal entry (the crash evidence);
    #   2. run: /wake_up + /is_sleeping, concurrently for every binding of the
    #      request (different GPUs: the leases say so);
    #   3. commit: the awake annotation and the store; a failed wake is settled
    #      instead - its lease, its desired power, and (S4) a compensating sleep
    #      when the engine woke after all - and its journal entry ended. An entry
    #      outlives the lock only when the engine's state cannot be read (or the
    #      service-manager died): the wake journal recovery settles it.
    # Observability (verification plan 20261001 section 7): JSON log events
    # wake_start / wake_done / wake_failed, counters in GET /v2/wake, and the
    # operation record's details (binding, placement, truth_source, phases_ms,
    # error_code, compensating_sleep).

    def _wake_inline(self, binding: Binding) -> None:
        """A whole wake under the writer lock the caller holds, for callers that
        own the desired state (their desired guard restores it) and the legacy
        store (fleet repair, startup convergence, defrag, the floor make-up)."""
        ticket = self._prepare_wake(binding, None, check_account=False)
        self._run_wakes([ticket])
        self._commit_wakes([ticket], restore_desired=False, update_store=False)
        self._note_wake_details([ticket])
        if not ticket.woke:
            raise ticket.exception or WakeFailed(
                f"wake of {binding.serve_id} failed", node=binding.slot.node,
                gpus=binding.slot.gpu_ids, binding_id=binding.binding_id,
            )

    def _wake_tickets(self, tickets: list["_WakeTicket"]) -> None:
        """/wake_up of every prepared ticket concurrently, then the commit (writer
        lock held throughout). Raises the first failed wake's error once every
        ticket is recorded (what woke stays awake)."""
        operation = current_operation()
        if operation is not None and tickets:
            operation.advance("waking", details={"pods": [t.binding.serve_id for t in tickets]})
        self._run_wakes(tickets)
        self._commit_wakes(tickets, restore_desired=True, update_store=True)
        self._note_wake_details(tickets)
        failed = [ticket for ticket in tickets if not ticket.woke]
        if failed:
            first = failed[0]
            raise first.exception or WakeFailed(
                f"wake of {first.binding.serve_id} failed", node=first.binding.slot.node,
                gpus=first.binding.slot.gpu_ids, binding_id=first.binding.binding_id,
            )
        unsaved = [ticket for ticket in tickets if ticket.store_error is not None]
        if unsaved:
            raise unsaved[0].store_error

    def _prepare_wake(
        self,
        binding: Binding,
        bindings: list[Binding] | None,
        *,
        check_account: bool = True,
        leases=None,
        journal=None,
        previous_desired: tuple[str, bool] | None = None,
        placement: dict | None = None,
    ) -> "_WakeTicket":
        """The prepare (writer lock held): every check that can refuse the wake,
        then the journal entry and the binding's ``awake`` GPU lease. Raises
        without side effects."""
        started = time.monotonic()
        if self._wake_journal.get(binding.binding_id) is not None:
            raise WakeConflict(
                f"{binding.serve_id}: a wake of {binding.binding_id} is already in progress",
                reason="wake_in_progress", node=binding.slot.node, gpus=binding.slot.gpu_ids,
                binding_id=binding.binding_id, blocking_binding_id=binding.binding_id,
            )
        if check_account:
            self._ensure_feasible_wake(binding, bindings or [], leases, journal)
        self._check_fault_hooks(binding)
        snapshot = self._snapshot_for_binding(binding)
        if not snapshot.pod_ip:
            raise ValueError(f"pod {binding.serve_id} has no pod IP for wake")
        gate = self._ensure_wake_headroom(binding) or {}
        ticket = _WakeTicket(
            binding=binding,
            pod_ip=str(snapshot.pod_ip),
            previous_desired=previous_desired,
            placement=placement,
            truth_source=str(gate.get("truth_source") or "none"),
            truth_age_s=gate.get("truth_age_s"),
            hinted=bool(placement and placement.get("source") == "hint"),
        )
        operation = current_operation()
        self._wake_journal.begin(
            binding.binding_id,
            {
                "binding_id": binding.binding_id,
                "serve_id": binding.serve_id,
                "model": binding.model,
                "node": binding.slot.node,
                "gpu_ids": list(binding.slot.gpu_ids),
                "pod_ip": ticket.pod_ip,
                "pod_uid": getattr(snapshot, "pod_uid", None),
                "previous_power": None if previous_desired is None else previous_desired[0],
                "previous_hidden": None if previous_desired is None else previous_desired[1],
                "owner": str(getattr(self._operation_coordinator, "owner", "service-manager")),
                "operation_id": None if operation is None else operation.operation_id,
                "started_ms": int(time.time() * 1000),
            },
        )
        try:
            # The GPUs are the binding's from here on (an ``awake`` lease: it never
            # expires); a failed wake's settlement or the journal recovery
            # releases it once the engine is known asleep.
            if self._gpu_leases is not None:
                self._gpu_leases.acquire(binding, phase="awake")
        except BaseException:
            self._wake_journal.end(binding.binding_id)
            raise
        # From here the engine may wake: a gpu-truth sample taken before this no
        # longer describes these GPUs (P1-2, S1).
        self._note_binding_power_change(binding)
        ticket.phases_ms["reserve"] = _elapsed_ms(started)
        _log_event(
            "wake_start",
            binding_id=binding.binding_id, serve_id=binding.serve_id, model=binding.model,
            node=binding.slot.node, gpu_ids=list(binding.slot.gpu_ids),
            truth_source=ticket.truth_source, placement=placement,
            operation_id=None if operation is None else operation.operation_id,
        )
        return ticket

    def _abort_prepared_wakes(self, tickets: list["_WakeTicket"]) -> None:
        """Undo the prepare of wakes that never ran (writer lock held): release
        the lease and end the journal entry. Best effort."""
        for ticket in tickets:
            try:
                if self._gpu_leases is not None:
                    self._gpu_leases.release(ticket.binding)
            except Exception:  # noqa: BLE001 - kept for the recovery (review P2-1)
                # The lease does not expire: keep the journal entry so the
                # recovery releases it (the engine was never woken).
                LOG.exception("releasing the lease of %s failed; left to the recovery",
                              ticket.binding.binding_id)
                continue
            self._wake_journal.end(ticket.binding.binding_id)

    def _fault_active(self, kind: str, binding: Binding) -> bool:
        """A test hook key ``tre:v2:sm:fault:<kind>:<node>/<gpu>`` is set for one of
        ``binding``'s GPUs. Always False unless registry service_manager.test_hooks
        is true (the keys are not even read then)."""
        if not getattr(self._sm_config, "test_hooks", False) or self._fault_redis is None:
            return False
        for gpu in binding.slot.gpu_ids:
            try:
                if self._fault_redis.get(rediskeys.sm_fault_key(kind, binding.slot.node, gpu)) is not None:
                    return True
            except Exception:  # noqa: BLE001 - a broken hook read injects nothing
                LOG.warning("reading the %s test hook failed", kind, exc_info=True)
                return False
        return False

    def _check_fault_hooks(self, binding: Binding) -> None:
        if self._fault_active("refuse_wake", binding):
            raise WakeConflict(
                f"{binding.serve_id}: wake refused by the test hook "
                f"{rediskeys.SM_FAULT_KEY_PREFIX}refuse_wake (service_manager.test_hooks)",
                reason="fault_injected", node=binding.slot.node, gpus=binding.slot.gpu_ids,
                binding_id=binding.binding_id,
            )

    def _mark_journaled_wakes(self) -> None:
        """Power marks for the GPUs of every journaled wake (startup): a sample
        taken before that wake is not trusted."""
        if self._gpu_truth is None:
            return
        try:
            entries = self._wake_journal.entries()
        except Exception:  # noqa: BLE001 - the recovery handles them anyway
            return
        for entry in entries.values():
            try:
                self._note_power_change(str(entry["node"]), [int(g) for g in entry["gpu_ids"]])
            except (KeyError, TypeError, ValueError):
                continue

    def reap_stale_startup_placeholders(self, *, now: float | None = None) -> list[str]:
        """Supervisor pass (P2-8 / P1-3, review P2-2 / P2-3): a ``starting`` lease
        (the placeholder of a Pod admitted at its startup gate, S2, or of a container
        restart, P1-4) is released only while its engine container holds no GPU
        memory - the Pod is gone, or the ``vllm-openai`` container is waiting /
        terminated (CrashLoopBackOff between attempts) and /is_sleeping does not
        read awake. A running but not Ready engine keeps its placeholder; past
        ``startup_admission.placeholder_max_s`` that is only alerted
        (``startup_placeholder_overdue``). Every load of a crash-looping engine is
        covered: the restart guard runs first in the supervisor pass and places a
        new placeholder when the container starts again. A released binding becomes
        a suspect (its GPUs are never trusted from gpu-truth: the resident probe
        decides) until it reads asleep or is converged. Under the writer lock
        (wait 0); an alert per release."""
        if self._gpu_leases is None or self._runtime_ops is None:
            return []
        now = time.monotonic() if now is None else now
        limit = float(getattr(self._sm_config, "startup_placeholder_max_s", 900.0))
        try:
            starting = [lease for lease in self._gpu_leases.load() if lease.phase == "starting"]
        except Exception:  # noqa: BLE001 - next pass
            return []
        live = {lease.binding_id for lease in starting}
        for binding_id in list(self._placeholder_seen):
            if binding_id not in live:
                del self._placeholder_seen[binding_id]
        restarts = self._restart_counts_by_binding()

        def placeholder(lease) -> Binding:
            return Binding(
                "startup-placeholder", lease.binding_id.split("/", 1)[0],
                Slot(lease.node, tuple(int(g) for g in lease.gpu_ids)), awake=False,
            )

        releasable = []
        for lease in starting:
            first, _restarts_then = self._placeholder_seen.setdefault(
                lease.binding_id, (now, restarts.get(lease.binding_id, 0))
            )
            if now - first < self._placeholder_min_hold_s:
                continue  # 2026-10-01: never right after the placeholder appeared
            verdict = self._placeholder_verdict(placeholder(lease))
            if verdict is None:
                releasable.append(lease)
            elif verdict == "engine_running" and now - first > limit and lease.binding_id not in self._placeholder_alerted:
                self._placeholder_alerted.add(lease.binding_id)
                _log_event(
                    "startup_placeholder_overdue", level=logging.ERROR,
                    binding_id=lease.binding_id, held_s=round(now - first, 1),
                    detail="engine running but not Ready past placeholder_max_s; placeholder kept",
                )
        if not releasable:
            return []
        released: list[str] = []
        with self._writer("reap_stale_startup_placeholders", wait_s=0.0):
            for lease in releasable:
                binding = placeholder(lease)
                if self._placeholder_verdict(binding) is not None:  # changed meanwhile
                    continue
                first, restarts_then = self._placeholder_seen.get(lease.binding_id, (now, 0))
                self._gpu_leases.release(binding)
                self._placeholder_seen.pop(lease.binding_id, None)
                self._placeholder_alerted.discard(lease.binding_id)
                pod_name = self._restart_placeholders.pop(lease.binding_id, None) or ""
                self._suspects[lease.binding_id] = (
                    lease.node, tuple(int(g) for g in lease.gpu_ids), pod_name
                )
                self._note_binding_power_change(binding)
                _log_event(
                    "startup_placeholder_released",
                    level=logging.ERROR,
                    binding_id=lease.binding_id, node=lease.node, gpu_ids=list(lease.gpu_ids),
                    held_s=round(now - first, 1),
                    restarts=restarts.get(lease.binding_id, 0), restarts_when_seen=restarts_then,
                    detail="engine container not running and not verifiably awake; binding now a suspect",
                )
                released.append(lease.binding_id)
        return released

    def _restart_counts_by_binding(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        lister = getattr(self._runtime_ops, "list_pod_snapshots", None)
        if not callable(lister):
            return counts
        try:
            for snapshot in lister():
                try:
                    binding_id = _binding_from_snapshot(snapshot).binding_id
                except (KeyError, ValueError):
                    continue
                counts[binding_id] = max(counts.get(binding_id, 0), int(getattr(snapshot, "restart_count", 0) or 0))
        except Exception:  # noqa: BLE001 - unknown: no crashloop verdict
            return {}
        return counts

    def guard_container_restarts(self) -> dict:
        """Supervisor pass (P1-4): a vLLM container restarted IN PLACE (restart count
        up, same Pod) loads its weights and comes up AWAKE - the startup gate does not
        run again and the store may record it asleep. The binding gets a
        ``starting`` placeholder on its GPUs at once (same mechanism and bound as
        S2; alert ``container_restart_placeholder``). SM actuation active: once
        /is_sleeping answers, it is converged - desired sleeping and awake -> slept
        through the sleep primitive; desired awake and awake -> the ``awake`` lease,
        annotation and store; asleep -> placeholder released. Observe: placeholder
        and alert only. The first pass only records the counts (a restart while
        the SM was down is not seen: the root fix is an admission on every
        container start, see the design note)."""
        lister = getattr(self._runtime_ops, "list_pod_snapshots", None)
        if self._gpu_leases is None or not callable(lister):
            return {"placed": [], "converged": []}
        try:
            snapshots = list(lister())
        except Exception:  # noqa: BLE001 - next pass
            return {"placed": [], "converged": []}
        if self._restarts_seen is None:
            # First pass: the persisted counts (review P2-4) - a restart while this
            # SM was down is detected now; a pod never recorded is a baseline.
            try:
                self._restarts_seen = dict(self._restart_ledger.load())
            except Exception:  # noqa: BLE001 - start from scratch
                self._restarts_seen = {}
        restarted = []
        for snapshot in snapshots:
            uid = snapshot.pod_uid or snapshot.name
            count = int(getattr(snapshot, "restart_count", 0) or 0)
            previous = self._restarts_seen.get(uid)
            if previous is not None and count > previous:
                restarted.append(snapshot)
            elif previous != count:
                self._record_restart_count(uid, count)
        live = {snapshot.pod_uid or snapshot.name for snapshot in snapshots}
        for uid in list(self._restarts_seen):
            if uid not in live:
                del self._restarts_seen[uid]
                try:
                    self._restart_ledger.drop(uid)
                except Exception:  # noqa: BLE001 - stale field, harmless
                    pass
        placed: list[str] = []
        converged: list[str] = []
        if not restarted and not self._restart_placeholders and not self._suspects:
            return {"placed": placed, "converged": converged}
        with self._writer("guard_container_restarts", wait_s=0.0):
            for snapshot in restarted:
                try:
                    binding = replace(_binding_from_snapshot(snapshot), awake=False, hidden=False)
                except (KeyError, ValueError):
                    self._restarts_seen[snapshot.pod_uid or snapshot.name] = int(snapshot.restart_count or 0)
                    continue
                try:
                    self._gpu_leases.acquire(binding, phase="starting")
                except GpuLeaseConflict as exc:
                    # No placeholder: the binding stays (or becomes) a suspect, so
                    # its GPUs are never trusted from gpu-truth and, should the
                    # engine read awake, the suspect convergence below settles it
                    # to its desired power (integration 2026-10-01).
                    self._suspects[binding.binding_id] = (
                        binding.slot.node, tuple(int(g) for g in binding.slot.gpu_ids), snapshot.name
                    )
                    _log_event(
                        "container_restart_conflict", level=logging.ERROR,
                        binding_id=binding.binding_id, pod=snapshot.name, occupant=exc.occupant,
                        detail="restarted engine on GPUs another binding holds - possible double occupancy; binding kept a suspect",
                    )
                else:
                    self._restart_placeholders[binding.binding_id] = snapshot.name
                    # The placeholder now holds its GPUs: no longer a suspect.
                    self._suspects.pop(binding.binding_id, None)
                    placed.append(binding.binding_id)
                    _log_event(
                        "container_restart_placeholder", level=logging.WARNING,
                        binding_id=binding.binding_id, pod=snapshot.name, node=binding.slot.node,
                        gpu_ids=list(binding.slot.gpu_ids), restart_count=int(snapshot.restart_count or 0),
                        since=int(time.time() * 1000),
                    )
                self._note_binding_power_change(binding)
                self._record_restart_count(snapshot.pod_uid or snapshot.name, int(snapshot.restart_count or 0))
            # Converge placeholders and suspects once /is_sleeping answers; observe
            # (review P2-5) records what it finds (leases, store) but sleeps nothing.
            observe = self.actuation_observe()
            by_name = {snapshot.name: snapshot for snapshot in snapshots}
            for binding_id, pod_name in list(self._restart_placeholders.items()):
                snapshot = by_name.get(pod_name)
                if snapshot is None:
                    self._restart_placeholders.pop(binding_id, None)
                    continue
                if self._converge_isolated(snapshot, binding_id, observe=observe):
                    self._restart_placeholders.pop(binding_id, None)
                    converged.append(binding_id)
            for binding_id, (node, gpus, pod_name) in list(self._suspects.items()):
                snapshot = by_name.get(pod_name) or next(
                    (s for s in snapshots if self._snapshot_binding_id(s) == binding_id), None
                )
                if snapshot is None:
                    self._suspects.pop(binding_id, None)
                    continue
                if self._converge_isolated(snapshot, binding_id, observe=observe, suspect=True):
                    self._suspects.pop(binding_id, None)
                    converged.append(binding_id)
        return {"placed": placed, "converged": converged}

    def _converge_isolated(self, snapshot, binding_id: str, **kwargs) -> bool:
        """One placeholder / suspect never starves the others (an error = retried
        next pass)."""
        try:
            return self._converge_restart(snapshot, **kwargs)
        except Exception:  # noqa: BLE001
            LOG.exception("converging the restarted binding %s failed; next pass", binding_id)
            return False

    def _record_restart_count(self, uid: str, count: int) -> None:
        self._restarts_seen[uid] = int(count)
        try:
            self._restart_ledger.set(uid, int(count))
        except Exception:  # noqa: BLE001 - next pass writes it again
            LOG.warning("persisting the restart count of %s failed", uid, exc_info=True)

    @staticmethod
    def _snapshot_binding_id(snapshot) -> str | None:
        try:
            return _binding_from_snapshot(snapshot).binding_id
        except (KeyError, ValueError):
            return None

    def _converge_restart(self, snapshot, *, observe: bool = False, suspect: bool = False) -> bool:
        """Converge a restarted container (or a suspect, review P2-3) once
        /is_sleeping answers (writer lock held). True when done. ``observe``
        (review P2-5): bookkeeping only - an awake engine desired asleep keeps an
        ``awake`` lease and is recorded awake (alert), nothing is slept."""
        if not snapshot.pod_ip or self._vllm_ops is None:
            return False
        # A suspect may still hold its GPU lease (a gave-up wake recovery, a failed
        # cold start, a lease carried over a restart; I1, 2026-10-04): it is
        # released on "asleep" only when no wake is in flight - the sidecar's count
        # is read BEFORE /is_sleeping (no answer: next pass, no second probe).
        waking = self._sidecar_waking(snapshot.pod_ip) if suspect else None
        if suspect and waking is None:
            return False
        physical = self._vllm_ops.is_sleeping(snapshot.pod_ip, port=8000)
        if physical is None:
            return False
        binding = replace(_binding_from_snapshot(snapshot), awake=False, hidden=False)
        desired = self._desired_power_of(binding.binding_id)
        if physical is True:
            if suspect:
                if self._holds_gpu_lease(binding.binding_id):
                    if not self._no_wake_in_flight(waking, binding.binding_id, "suspect"):
                        return False  # a wake is in flight: next pass
                    self._gpu_leases.release(binding)
                    _log_event(
                        "suspect_lease_released", level=logging.WARNING,
                        binding_id=binding.binding_id, pod=snapshot.name,
                        detail="engine reads asleep and the sidecar reports no wake in flight",
                    )
                self._note_binding_power_change(binding)
                return True  # asleep: nothing held, nothing to record
            self._gpu_leases.release(binding)
            self._runtime_ops.write_binding_annotations(binding, state=POD_STATE_SLEEPING)
            self._set_store_power(binding.binding_id, awake=False, hidden=False)
        elif desired is not None and desired[0] == "sleeping" and observe:
            self._gpu_leases.acquire(binding, phase="awake")
            self._set_store_power(binding.binding_id, awake=True, hidden=True)
            _log_event(
                "restart_awake_desired_asleep_observe", level=logging.ERROR,
                binding_id=binding.binding_id,
                detail="SM actuation observe: awake engine recorded (awake lease), not slept",
            )
        elif desired is not None and desired[0] == "sleeping":
            if self._sleep_primitive is None:
                return False
            awake = replace(binding, awake=True, hidden=True)
            outcomes = self._sleep_targets([SleepTarget(awake, snapshot.pod_ip)], sleep_path="repair")
            if not any(item.get("status") == STATUS_SLEPT for item in outcomes):
                return False
            self._set_store_power(binding.binding_id, awake=False, hidden=False)
        else:
            hidden = bool(desired[1]) if desired is not None else False
            self._runtime_ops.write_binding_annotations(
                binding, state=POD_STATE_HIDDEN if hidden else POD_STATE_AWAKE
            )
            self._gpu_leases.acquire(binding, phase="awake")
            self._set_store_power(binding.binding_id, awake=True, hidden=hidden)
        self._note_binding_power_change(binding)
        self._refresh_observed([binding.binding_id])
        _log_event("container_restart_converged", binding_id=binding.binding_id,
                   physically_awake=physical is False, desired=desired)
        return True

    def _holds_gpu_lease(self, binding_id: str) -> bool:
        if self._gpu_leases is None:
            return False
        return any(lease.binding_id == binding_id for lease in self._gpu_leases.load())

    def _placeholder_verdict(self, binding: Binding) -> str | None:
        """None = the placeholder may be released; else why it is kept.

        2026-10-01 (deployment finding): a Pod admitted at its startup gate is
        still initialising (phase Pending, the gate init container running, the
        engine container Waiting PodInitializing / ContainerCreating) and is NOT
        in the Running-only Pod snapshots - it must never count as "gone" nor as
        "not holding GPU memory". With the runtime's ``startup_pod_states`` (every
        phase) a placeholder is releasable only when every Pod of the binding has
        finished initialising and its engine container is terminated, or waiting
        with a reason in the release set (CrashLoopBackOff, Error, image pull /
        create errors), and /is_sleeping does not read awake. No Pod at all keeps
        it too (the orphan starting-lease reaper owns that case)."""
        lister = getattr(self._runtime_ops, "startup_pod_states", None)
        if callable(lister):
            return self._placeholder_verdict_from_states(binding, lister)
        pods = []
        try:
            for snapshot in self._runtime_ops.list_pod_snapshots(model=binding.model):
                try:
                    if _binding_from_snapshot(snapshot).binding_id == binding.binding_id:
                        pods.append(snapshot)
                except (KeyError, ValueError):
                    continue
        except Exception:  # noqa: BLE001 - cannot tell: keep it
            return "pods_unreadable"
        for snapshot in pods:
            if snapshot.ready:
                return "pod_ready"  # the convergence's job
            if snapshot.pod_ip and self._vllm_ops is not None:
                try:
                    if self._vllm_ops.is_sleeping(snapshot.pod_ip, port=8000) is False:
                        return "pod_awake"
                except Exception:  # noqa: BLE001 - unreadable
                    pass
            if getattr(snapshot, "engine_running", None) is not False:
                # Running (loading) or unknown: it may hold GPU memory (review P2-3).
                return "engine_running"
        if not pods:
            return "no_running_pod"  # may be initialising (Pending): keep
        return None

    def _placeholder_verdict_from_states(self, binding: Binding, lister) -> str | None:
        try:
            states = [
                state for state in lister(model=binding.model)
                if state.get("binding_id") == binding.binding_id
            ]
        except Exception:  # noqa: BLE001 - cannot tell: keep it
            return "pods_unreadable"
        if not states:
            return "no_pod"
        for state in states:
            if not state.get("init_done"):
                return "initializing"
            if state.get("ready"):
                return "pod_ready"
            engine = state.get("engine_state")
            if engine == "running":
                if state.get("pod_ip") and self._vllm_ops is not None:
                    try:
                        if self._vllm_ops.is_sleeping(state["pod_ip"], port=8000) is False:
                            return "pod_awake"
                    except Exception:  # noqa: BLE001
                        pass
                return "engine_running"
            if engine == "waiting" and state.get("engine_reason") not in self._placeholder_release_reasons:
                return "engine_starting"
            if engine not in ("waiting", "terminated"):
                return "engine_unknown"
            if state.get("pod_ip") and self._vllm_ops is not None:
                try:
                    if self._vllm_ops.is_sleeping(state["pod_ip"], port=8000) is False:
                        return "pod_awake"
                except Exception:  # noqa: BLE001 - unreadable: not serving
                    pass
        return None

    def _run_wakes(self, tickets: list["_WakeTicket"]) -> None:
        """/wake_up then /is_sleeping of every ticket, concurrently when there are
        several (writer lock held). Never raises: the outcome is in each ticket
        (``woke`` / ``exception``)."""
        for ticket in tickets:
            ticket.parallel_inflight = len(tickets)
        self._wake_journal.record_max("wake_parallel_max", len(tickets))
        if len(tickets) <= 1:
            for ticket in tickets:
                self._run_one_wake(ticket)
            return
        try:
            futures = [self._wake_executor.submit(self._run_one_wake, ticket) for ticket in tickets]
        except RuntimeError:  # executor shut down (SIGTERM): run them one by one
            for ticket in tickets:
                self._run_one_wake(ticket)
            return
        futures_wait(futures)
        for ticket, future in zip(tickets, futures):
            error = future.exception()
            if error is not None and ticket.exception is None:
                ticket.woke = False
                ticket.exception = error

    def _run_one_wake(self, ticket: "_WakeTicket") -> None:
        binding = ticket.binding
        started = time.monotonic()

        def failed(message: str, reason: str) -> WakeFailed:
            return WakeFailed(
                message, reason=reason, node=binding.slot.node, gpus=binding.slot.gpu_ids,
                binding_id=binding.binding_id,
            )

        try:
            result = self._vllm_ops.wake_up(ticket.pod_ip, port=8000)
            if not bool(getattr(result, "success", False)):
                message = getattr(result, "message", "") or "operation failed"
                if _unanswered(result):
                    # No HTTP answer (VllmOps folds a timeout / connection error
                    # into status_code None): the engine may still be waking - the
                    # outcome is uncertain, not failed (I1, 2026-10-04).
                    ticket.uncertain = True
                raise failed(f"vLLM wake failed for {binding.serve_id}: {message}", "vllm_wake_failed")
            if self._fault_active("fail_wake", binding):
                raise failed(f"vLLM wake failed for {binding.serve_id}: fault injected", "fault_injected")
            if hasattr(self._vllm_ops, "is_sleeping"):
                physical = self._vllm_ops.is_sleeping(ticket.pod_ip, port=8000)
                if physical is not False:
                    raise failed(
                        f"vLLM wake did not physically converge for {binding.serve_id}", "not_converged"
                    )
            ticket.woke = True
        except WakeConflict as exc:
            ticket.woke = False
            ticket.exception = exc
        except Exception as exc:  # noqa: BLE001 - a transport error: a failed wake (P3-12)
            ticket.woke = False
            # The request may still be running on the server (a timeout): the
            # outcome is uncertain - kept for a delayed recheck by the recovery.
            ticket.uncertain = True
            failure = failed(
                f"vLLM wake failed for {binding.serve_id}: {type(exc).__name__}: {exc}", "vllm_wake_failed"
            )
            failure.__cause__ = exc
            ticket.exception = failure
        finally:
            ticket.phases_ms["wake_up"] = _elapsed_ms(started)

    def _commit_wakes(
        self,
        tickets: list["_WakeTicket"],
        *,
        restore_desired: bool,
        update_store: bool,
    ) -> None:
        """The commit (writer lock held). Woken: awake annotation, ``awake`` lease
        and (``update_store``) the store. Failed - or woken but not recordable (an
        annotation patch refused): the wake is settled
        (:meth:`_settle_failed_wake`, S4) and (``restore_desired``) its desired
        power restored from before the wake."""
        for ticket in tickets:
            binding = ticket.binding
            started = time.monotonic()
            try:
                self._commit_one_wake(ticket, restore_desired=restore_desired, update_store=update_store)
            except Exception as exc:  # noqa: BLE001 - never stop the other tickets (P1-1)
                # Left journaled (the entry is only ended once settled): the journal
                # recovery resolves it.
                LOG.exception("committing the wake of %s failed", binding.binding_id)
                ticket.commit_error = exc
            finally:
                # Woken, or a failed wake in whatever state it left: an earlier
                # gpu-truth sample no longer describes these GPUs (S1), and
                # observed follows (B2).
                self._note_binding_power_change(binding)
                self._refresh_observed([binding.binding_id])
                ticket.phases_ms["commit"] = _elapsed_ms(started)
                self._log_wake_outcome(ticket)

    def _commit_one_wake(self, ticket: "_WakeTicket", *, restore_desired: bool, update_store: bool) -> None:
        binding = ticket.binding
        if ticket.woke:
            try:
                self._runtime_ops.write_binding_annotations(binding, state=POD_STATE_AWAKE)
                if self._gpu_leases is not None:
                    self._gpu_leases.acquire(binding, phase="awake")
            except Exception as exc:  # noqa: BLE001 - settled below
                LOG.exception("recording the wake of %s failed", binding.binding_id)
                ticket.woke = False
                failure = WakeFailed(
                    f"recording the wake of {binding.serve_id} failed: {type(exc).__name__}: {exc}",
                    reason="not_recorded", node=binding.slot.node, gpus=binding.slot.gpu_ids,
                    binding_id=binding.binding_id,
                )
                failure.__cause__ = exc
                ticket.exception = failure
        if ticket.woke and update_store:
            try:
                self._set_store_power(binding.binding_id, awake=True, hidden=False)
            except Exception as exc:  # noqa: BLE001 - raised after the commit
                # Physically done (review 3 P3): desired stays awake; the
                # reconcile catches the legacy store up.
                LOG.exception("saving the wake of %s in the store failed", binding.binding_id)
                ticket.store_error = exc
        if not ticket.woke and ticket.uncertain:
            # A /wake_up without an answer may still wake the engine after we look:
            # keep the lease + journal entry (``uncertain``). The recovery looks
            # again no earlier than service_manager.wake.transport_recheck_s (a
            # floor, not the proof) and releases only on the sidecar's waking == 0
            # AND /is_sleeping true in one pass (desired restored then).
            recheck_s = float(getattr(self._sm_config, "wake_transport_recheck_s", 30.0))
            self._wake_journal.update(
                binding.binding_id, recover_after_ms=int(time.time() * 1000 + recheck_s * 1000),
                uncertain=str(ticket.exception),
            )
            ticket.left_to_recovery = True
            return
        if not ticket.woke:
            self._settle_failed_wake(ticket)
            if ticket.lease_unsettled:
                # The lease could not be released / converted (a lost fence, Redis):
                # the recovery settles it (review P2-1).
                ticket.left_to_recovery = True
                return
            if ticket.physical is False:
                # The compensating sleep did not put it to sleep: the engine is awake
                # (awake lease held). Keep the entry: the recovery completes it as a
                # wake (desired stays awake) instead of recording a failure the
                # cluster does not show.
                ticket.left_to_recovery = True
                return
            if ticket.physical is None:
                # State unknown: the lease (it does not expire) and the
                # journal entry stay; the recovery decides once the pod can
                # be read - or gives up after a bound (P2-5).
                ticket.left_to_recovery = True
                return
            if restore_desired:
                self._restore_wake_desired(ticket)
        self._wake_journal.end(binding.binding_id)

    def _settle_failed_wake(self, ticket: "_WakeTicket") -> None:
        """A failed wake must not leave its lease behind (review 2026-09-29) -
        the GPUs would stay taken for an engine that sleeps - nor an engine that
        woke after all (S4): /is_sleeping decides.

        * asleep -> the lease is released (a sleeping binding holds none);
        * awake (the wake did happen: /wake_up timed out or failed late, the
          annotation / lease could not be recorded) -> a compensating sleep
          through the sleep primitive (path ``repair``: floor-exempt; the pod was
          never made routable), confirmed by /is_sleeping; slept -> lease
          released, else the ``awake`` lease (its GPUs are in use) and an alert;
        * unknown -> the lease and the journal entry are kept (the lease does not
          expire): the journal recovery decides once the pod can be read,
          or hands the binding to the suspect convergence (lease kept) after
          ``service_manager.wake.recovery_unknown_attempts`` / when the pod is
          not Ready (P2-5). A warning is logged.
        Best effort; never raises."""
        binding = ticket.binding
        try:
            physical = self._vllm_ops.is_sleeping(ticket.pod_ip, port=8000)
        except Exception:  # noqa: BLE001 - unknown
            physical = None
        ticket.physical = physical
        if isinstance(ticket.exception, WakeFailed):
            ticket.exception.physically_awake = None if physical is None else not physical
        if physical is False:
            ticket.compensating_sleep = self._compensating_sleep(ticket)
            if isinstance(ticket.exception, WakeFailed):
                ticket.exception.compensating_sleep = ticket.compensating_sleep
            if ticket.compensating_sleep.get("done"):
                return
        if self._gpu_leases is None:
            return
        try:
            if physical is True:
                self._gpu_leases.release(binding)
            elif physical is False:
                self._gpu_leases.acquire(binding, phase="awake")
                LOG.error(
                    json.dumps(
                        {"event": "wake_failed_left_awake", "binding_id": binding.binding_id,
                         "compensating_sleep": ticket.compensating_sleep},
                        sort_keys=True,
                    )
                )
            else:
                LOG.warning(
                    json.dumps(
                        {"event": "wake_failed_state_unknown", "binding_id": binding.binding_id,
                         "detail": "GPU lease and wake journal entry kept for the recovery"},
                        sort_keys=True,
                    )
                )
        except Exception:  # noqa: BLE001 - kept for the recovery (review P2-1)
            ticket.lease_unsettled = True
            LOG.exception("settling the GPU lease of the failed wake of %s failed", binding.binding_id)

    def _compensating_sleep(self, ticket: "_WakeTicket") -> dict:
        """S4: put an engine that woke during a failed wake back to sleep (writer
        lock held). Returns {"done": bool, "result": str}."""
        binding = ticket.binding
        self._wake_journal.incr("wake_compensating_sleep_total")
        if self._sleep_primitive is None:
            return {"done": False, "result": "no_sleep_primitive"}
        # Hidden + awake: the primitive's rollback keeps it unroutable.
        target = SleepTarget(replace(binding, awake=True, hidden=True), ticket.pod_ip)
        try:
            outcomes = self._sleep_targets([target], sleep_path="repair")
        except Exception as exc:  # noqa: BLE001 - reported
            result = {"done": False, "result": f"sleep_failed: {type(exc).__name__}: {exc}"}
        else:
            slept = any(item.get("status") == STATUS_SLEPT for item in outcomes)
            confirmed = slept and self._vllm_ops.is_sleeping(ticket.pod_ip, port=8000) is True
            result = {"done": bool(confirmed), "result": "slept" if confirmed else "not_confirmed"}
        if result["done"]:
            ticket.physical = True
        else:
            self._wake_journal.incr("wake_compensating_sleep_failed_total")
        return result

    def _log_wake_outcome(self, ticket: "_WakeTicket") -> None:
        binding = ticket.binding
        if ticket.woke:
            self._wake_journal.incr("wake_done_total")
            _log_event(
                "wake_done",
                binding_id=binding.binding_id, serve_id=binding.serve_id, model=binding.model,
                node=binding.slot.node, gpu_ids=list(binding.slot.gpu_ids),
                phases_ms=dict(ticket.phases_ms), parallel_inflight=ticket.parallel_inflight,
                truth_source=ticket.truth_source, placement=ticket.placement,
            )
            return
        self._wake_journal.incr("wake_failed_total")
        _log_event(
            "wake_failed",
            level=logging.WARNING,
            binding_id=binding.binding_id, serve_id=binding.serve_id, model=binding.model,
            node=binding.slot.node, gpu_ids=list(binding.slot.gpu_ids),
            error_code=_error_code(ticket.exception), error=str(ticket.exception),
            physically_awake=None if ticket.physical is None else not ticket.physical,
            compensating_sleep=ticket.compensating_sleep, phases_ms=dict(ticket.phases_ms),
        )

    def _note_wake_details(self, tickets: list["_WakeTicket"], *, error: BaseException | None = None) -> None:
        """The current operation record's details (verification plan section 7-1):
        which bindings, where, how the gate decided, phase durations, errors."""
        operation = current_operation()
        note = getattr(operation, "note", None)
        if not callable(note) or not tickets:
            return
        first = tickets[0]
        failed = next((ticket for ticket in tickets if not ticket.woke and ticket.exception is not None), None)
        details = {
            "binding_id": first.binding.binding_id,
            "binding_ids": [ticket.binding.binding_id for ticket in tickets],
            "model": first.binding.model,
            "node": first.binding.slot.node,
            "gpu_ids": list(first.binding.slot.gpu_ids),
            "placement": first.placement if len(tickets) == 1 else [ticket.placement for ticket in tickets],
            "truth_source": first.truth_source,
            "truth_age_s": first.truth_age_s,
            "phases_ms": dict(first.phases_ms)
            if len(tickets) == 1
            else {ticket.binding.binding_id: dict(ticket.phases_ms) for ticket in tickets},
            "wake_attempts": len(tickets),
            "parallel_inflight": max(ticket.parallel_inflight for ticket in tickets),
        }
        code = _error_code(error or (failed.exception if failed is not None else None))
        if code is not None:
            details["error_code"] = code
        compensating = [ticket.compensating_sleep for ticket in tickets if ticket.compensating_sleep]
        if compensating:
            details["compensating_sleep"] = compensating[0] if len(compensating) == 1 else compensating
        try:
            note(**details)
        except Exception:  # noqa: BLE001 - observability only
            LOG.warning("recording the wake details in the operation record failed", exc_info=True)

    def _restore_wake_desired(self, ticket: "_WakeTicket") -> None:
        previous = ticket.previous_desired
        if previous is None or self._fleet_store is None:
            return
        try:
            self._update_desired(
                {ticket.binding.binding_id: {"power": previous[0], "hidden": previous[1]}},
                updated_by="service-manager-rollback",
                reason="wake_failed",
            )
        except Exception:  # noqa: BLE001 - the audit reports the mismatch
            LOG.exception("restoring the desired power of %s after a failed wake failed",
                          ticket.binding.binding_id)

    def _desired_power_of(self, binding_id: str) -> tuple[str, bool] | None:
        """(power, hidden) of ``binding_id``'s desired record, None without one."""
        if self._fleet_store is None:
            return None
        for item in self._fleet_store.load_desired().bindings:
            if item.binding_id == binding_id:
                return item.power, bool(item.hidden)
        return None

    def recover_wake_journal(self) -> dict:
        """Resolve wake journal entries, at SM start and on every supervisor pass.
        Every wake holds the writer lock from its prepare to its commit
        (whole-lock, 2026-10-02), so an entry seen under the lock belongs to no
        wake in progress: its service-manager died mid-wake, or the wake was left
        for a recheck (a /wake_up without an answer, an unreadable engine). The
        pod's physical state decides: awake -> the wake is completed
        (annotation, ``awake`` lease, store); asleep or gone -> rolled back (lease
        released, desired power restored from the entry); unknown -> kept for
        the next pass, at most ``service_manager.wake.recovery_unknown_attempts``
        passes (then, or at once when the pod is not Ready, the desired power is
        restored and the binding becomes a suspect that KEEPS its lease, with an
        alert, P2-5 / I1); a pod replaced since the wake (other UID) is rolled back
        without touching the new pod (P2-4). An unanswered /wake_up (entry
        ``uncertain``) is rolled back only when the sidecar reports no wake in
        flight AND the engine reads asleep in the same pass; else kept
        (``wake_unsettled``)."""

        def stale() -> dict[str, dict]:
            now_ms = time.time() * 1000
            return {
                binding_id: entry
                for binding_id, entry in self._wake_journal.entries().items()
                if float(entry.get("recover_after_ms") or 0) <= now_ms
            }

        if not stale() or self._runtime_ops is None or self._vllm_ops is None:
            return {"resolved": [], "kept": []}
        resolved: list[dict] = []
        kept: list[dict] = []
        with self._writer("wake_journal_recovery"):
            for binding_id, entry in sorted(stale().items()):
                try:
                    result = self._recover_wake_entry(binding_id, entry)
                except Exception as exc:  # noqa: BLE001 - one entry never stops the pass
                    LOG.exception("recovering the wake of %s failed", binding_id)
                    kept.append({"binding_id": binding_id, "result": f"error: {type(exc).__name__}"})
                    continue
                (kept if result in _WAKE_RECOVERY_KEPT else resolved).append(
                    {"binding_id": binding_id, "result": result}
                )
        if resolved or kept:
            LOG.warning("wake journal recovery: resolved=%s kept=%s", resolved, kept)
        return {"resolved": resolved, "kept": kept}

    def _recover_wake_entry(self, binding_id: str, entry: dict) -> str:
        try:
            binding = Binding(
                str(entry["serve_id"]),
                str(entry["model"]),
                Slot(str(entry["node"]), tuple(int(gpu) for gpu in entry["gpu_ids"])),
                awake=False,
            )
        except (KeyError, TypeError, ValueError):
            LOG.error("dropping corrupt wake journal entry %s: %s", binding_id, entry)
            self._wake_journal.end(binding_id)
            return "corrupt"
        previous = entry.get("previous_power")
        ticket = _WakeTicket(
            binding=binding,
            pod_ip=str(entry.get("pod_ip") or ""),
            previous_desired=None if previous is None else (str(previous), bool(entry.get("previous_hidden"))),
        )
        try:
            snapshot = self._snapshot_for_binding(binding)
        except ValueError:
            snapshot = None
        journaled_uid = entry.get("pod_uid")
        # P2-4: the pod was replaced since the wake; the new one (loading, its own
        # startup gate) is not the one we woke: never "complete" it.
        replaced = bool(
            snapshot is not None and journaled_uid and snapshot.pod_uid
            and str(snapshot.pod_uid) != str(journaled_uid)
        )
        if snapshot is None or not snapshot.pod_ip or replaced:
            # Not among the Running, non-terminating Pods - but a terminating Pod's
            # engine may still be awake in its grace period (I1, 2026-10-04): rolled
            # back only once no Pod object of it exists any more.
            if not self._journaled_pod_gone(binding, journaled_uid):
                return "pod_still_present"
            self._roll_back_journaled_wake(ticket, binding_id)
            return "pod_replaced" if replaced else "pod_gone"
        ticket.pod_ip = str(snapshot.pod_ip)
        uncertain = bool(entry.get("uncertain"))
        # Every entry (an unanswered /wake_up, or one a dead SM left mid-wake): the
        # sidecar's in-flight count is read BEFORE /is_sleeping - a wake still
        # running then shows waking > 0; one that ended before it is reflected by
        # /is_sleeping. A sidecar without an answer: the engine behind the same
        # port is not probed (unknown this pass).
        waking = self._sidecar_waking(ticket.pod_ip)
        physical = None if waking is None else self._vllm_ops.is_sleeping(ticket.pod_ip, port=8000)
        if physical is None:
            attempts = int(entry.get("recovery_attempts") or 0) + 1
            limit = int(getattr(self._sm_config, "wake_recovery_unknown_attempts", 12))
            if attempts >= limit or not snapshot.ready:
                # P2-5: do not freeze the model forever - but never release the
                # lease without evidence (I1, 2026-10-04): the binding becomes a
                # suspect (GPUs never trusted from gpu-truth); the suspect
                # convergence settles it once the pod reads asleep, or the orphan
                # lease reaper once the Pod is gone.
                _log_event(
                    "wake_recovery_gave_up",
                    level=logging.ERROR,
                    binding_id=binding_id, attempts=attempts, pod_ready=bool(snapshot.ready),
                    detail="physical state unreadable; GPU lease kept, binding now a suspect, desired power restored",
                )
                self._restore_wake_desired(ticket)
                self._suspects[binding_id] = (
                    binding.slot.node, tuple(int(g) for g in binding.slot.gpu_ids), str(snapshot.name or "")
                )
                self._wake_journal.end(binding_id)
                self._note_binding_power_change(binding)
                return "gave_up"
            self._wake_journal.update(binding_id, recovery_attempts=attempts)
            return "physical_state_unknown"
        if physical is True and not self._no_wake_in_flight(waking, binding_id, "wake_recovery"):
            # Asleep now, but a /wake_up may still be running: kept for the next pass.
            _log_event(
                "wake_recovery_wake_in_flight", level=logging.WARNING,
                binding_id=binding_id, sidecar_waking=waking, uncertain=uncertain,
                detail="engine reads asleep but a wake is in flight; GPU lease and journal entry kept",
            )
            return "wake_unsettled"
        ticket.woke = physical is False
        if not ticket.woke:
            ticket.exception = ValueError(f"{binding.serve_id} was found asleep after an interrupted wake")
        self._commit_wakes([ticket], restore_desired=True, update_store=True)
        return "completed" if ticket.woke else "rolled_back"

    def _sidecar_waking(self, pod_ip: str) -> int | str | None:
        """The reissue sidecar's count of /wake_up calls in flight;
        :data:`NO_SIDECAR` when the pod has none (also: vLLM ops that cannot
        ask); None = no answer (a wake may be in flight)."""
        reader = getattr(self._vllm_ops, "sidecar_waking", None)
        if not callable(reader):
            return NO_SIDECAR
        if not pod_ip:
            return None
        try:
            return reader(pod_ip, port=8000)
        except Exception:  # noqa: BLE001 - unreadable
            return None

    @staticmethod
    def _no_wake_in_flight(waking, binding_id: str, context: str) -> bool:
        """Whether ``waking`` (read before an /is_sleeping that said asleep) rules
        out a wake still in flight. No sidecar (404): /is_sleeping alone is the
        evidence - weaker (a wake still queued in front of the engine is not
        seen), logged as such."""
        if waking == 0:
            return True
        if waking == NO_SIDECAR:
            _log_event(
                "lease_release_weak_evidence", level=logging.WARNING,
                binding_id=binding_id, context=context,
                detail="no reissue sidecar (GET /tre-reissue/state 404): /is_sleeping alone taken as evidence",
            )
            return True
        return False

    def _journaled_pod_gone(self, binding: Binding, pod_uid) -> bool:
        """No Pod object (Pending, Running or terminating) of the journaled wake
        exists any more: by its UID, else by its binding. Unreadable - or a
        runtime that cannot tell - is not gone (fail closed)."""
        if pod_uid:
            lister = getattr(self._runtime_ops, "list_live_model_pod_uids", None)
            key = str(pod_uid)
        else:
            lister = getattr(self._runtime_ops, "list_live_model_pod_binding_ids", None)
            key = binding.binding_id
        if not callable(lister):
            return False
        try:
            return key not in set(lister())
        except Exception:  # noqa: BLE001 - unreadable: kept
            LOG.warning("listing the model Pods failed; journaled wake of %s kept", binding.binding_id, exc_info=True)
            return False

    def _roll_back_journaled_wake(self, ticket: "_WakeTicket", binding_id: str) -> None:
        if self._gpu_leases is not None:
            self._gpu_leases.release(ticket.binding)
        self._restore_wake_desired(ticket)
        self._wake_journal.end(binding_id)
        self._note_binding_power_change(ticket.binding)

    # --------------------------------------------------------------- transfer
    # POST /v2/transfers (2026-10-02): hand GPUs from a donor model to a receiver
    # model - sleep the donor binding(s) and wake the receiver binding that sleeps
    # on the same GPUs - in ONE writer-lock hold (whole-lock, design
    # docs/design/20261002-sm-wholelock.md): select the pairs (pure,
    # tre_sm.ops.transfer), the donors' sleep (hide -> ack -> /sleep ->
    # confirmation, every donor in parallel), the receivers' wake (every receiver
    # whose donors all slept, in parallel) and its commit. No other writer can
    # take the freed GPUs in between; nothing is journaled beyond the ordinary
    # sleep and wake journals - after a crash those settle each binding from its
    # physical state (a donor may have slept for nothing: the controller re-plans
    # on its next tick).

    def transfer(
        self,
        *,
        donor_model: str,
        receiver_model: str,
        count: int = 1,
        sleep_path: str = "urgent",
        donor_bindings: list[str] | tuple[str, ...] | None = None,
        avoid_gpus: list[str] | tuple[str, ...] = (),
    ) -> dict:
        """Transfer ``count`` donor replicas' GPUs to the receiver model (see the
        section comment). Returns the per-pair outcome (``pairs[].status`` done |
        donor_sleep_failed | receiver_wake_failed); raises :class:`TransferFailed`
        (409 ``partial``) when no pair completed, unless the donor's replica floor
        held the whole transfer back (200, ``clamped_by_floor``)."""
        self._registry.model(donor_model)
        self._registry.model(receiver_model)
        if donor_model == receiver_model:
            raise ValueError("donor_model and receiver_model must differ")
        count = int(count)
        if count < 1:
            raise ValueError("count must be at least 1")
        if sleep_path == "safescale_commit":
            # A SafeScale commit sleeps a HIDDEN probe pod; a transfer's donors are
            # serving (not hidden) replicas.
            raise ValueError("sleep_path safescale_commit is not a transfer path (donors are not hidden)")
        if self._sleep_primitive is None or not self._physical_wake_capable():
            raise ValueError("a transfer needs the runtime and vLLM ops")
        op = _Transfer(
            transfer_id=f"tr-{uuid4().hex[:12]}",
            donor_model=donor_model,
            receiver_model=receiver_model,
            count=count,
            sleep_path=sleep_path,
        )
        request = {
            "transfer_id": op.transfer_id, "donor_model": donor_model, "receiver_model": receiver_model,
            "count": count, "sleep_path": sleep_path,
        }
        with self._writer("transfer", request=request) as operation:
            started = time.monotonic()
            op.selection = self._select_transfer(
                op,
                donor_filter=None if donor_bindings is None else tuple(donor_bindings),
                avoid_gpus=tuple(avoid_gpus or ()),
            )
            op.phases_ms["select"] = _elapsed_ms(started)
            for item in op.selection.substituted:
                _log_event(
                    "placement_substituted", level=logging.WARNING,
                    transfer_id=op.transfer_id, model=receiver_model,
                    refused_binding_id=item["refused_binding_id"], error_code=item["reason"],
                    detail=item["detail"],
                )
            op.pairs = [_TransferPairState(pair) for pair in op.selection.pairs]
            if op.pairs:
                _log_event(
                    "transfer_start",
                    transfer_id=op.transfer_id, donor_model=donor_model, receiver_model=receiver_model,
                    count=count, sleep_path=sleep_path,
                    pairs=[_transfer_pair_brief(state.pair) for state in op.pairs],
                    unfilled=op.selection.unfilled,
                )
                phase = time.monotonic()
                self._transfer_sleep_donors(op)
                op.phases_ms["donor_sleep"] = _elapsed_ms(phase)
                phase = time.monotonic()
                self._transfer_wake_receivers(op)
                op.phases_ms["receiver_wake"] = _elapsed_ms(phase)
            op.phases_ms["total"] = _elapsed_ms(started)
            op.version = self._store.load().version
            self._note_transfer(op, operation)
            return self._finish_transfer(op)

    def _select_transfer(self, op: "_Transfer", *, donor_filter, avoid_gpus: tuple[str, ...]) -> TransferSelection:
        """The pair selection (pure, :mod:`tre_sm.ops.transfer`) on the current
        books, with the account's wake blocker, the fault hook and the resident
        probe injected. The donor model's routable view unreadable -> 409
        ``routable_unknown`` before anything is hidden."""
        bindings = self._store.load().bindings
        leases = self._active_leases()
        wake_entries = self._wake_journal.entries()
        busy = busy_binding_ids(
            wake_journal=wake_entries,
            sleep_journal=self._sleep_primitive.journal.entries(),
            transient_lease_ids=[
                lease.binding_id for lease in leases
                if str(getattr(lease, "phase", "")) in TRANSIENT_LEASE_PHASES
            ],
        )
        floor = self._model_floor(op.donor_model) if self._floor_enforced() else None
        routable = (
            set(self._floor_check(op.donor_model, set(), bindings).routable) if floor is not None else set()
        )
        spec = self._registry.model(op.receiver_model)
        awake = sum(1 for binding in bindings if binding.model == op.receiver_model and binding.awake)
        budget = max(0, scale_max_replicas(spec) - awake)
        topology = self._registry.topology()

        def blocker(receiver: Binding, planning: list[Binding], ignored: frozenset[str]):
            return self._wake_blocker(
                receiver, planning, [lease for lease in leases if lease.binding_id not in ignored], wake_entries
            )

        def pick(receivers: list[Binding], planning: list[Binding]) -> Binding:
            return _wake_pick(receivers, planning, topology, self._placement)

        return select_transfer_pairs(
            bindings,
            donor_model=op.donor_model,
            receiver_model=op.receiver_model,
            count=op.count,
            topology=topology,
            policy=self._placement,
            busy=busy,
            donor_floor=floor,
            donor_routable=routable,
            receiver_budget=budget,
            donor_filter=donor_filter,
            avoid_gpus=avoid_gpus,
            wake_blocker=blocker,
            veto=self._transfer_veto,
            receiver_pick=pick,
        )

    def _transfer_veto(self, pair: TransferPair) -> "WakeConflict | None":
        """Why the pair about to be taken must not be (None = take it): the
        ``refuse_wake`` test hook, a donor without a pod IP, or a resident of the
        receiver's GPUs (other than the donors) that is not confirmed asleep. The
        donor is still awake, so gpu-truth shows the GPU in use: only the
        resident /is_sleeping probe can tell whether a third resident sleeps."""
        receiver = pair.receiver
        node = receiver.slot.node
        gpus = tuple(receiver.slot.gpu_ids)
        if self._fault_active("refuse_wake", receiver):
            return WakeConflict(
                f"{receiver.serve_id}: wake refused by the test hook "
                f"{rediskeys.SM_FAULT_KEY_PREFIX}refuse_wake (service_manager.test_hooks)",
                reason="fault_injected", node=node, gpus=gpus, binding_id=receiver.binding_id,
            )
        for donor in pair.donors:
            try:
                pod_ip = self._snapshot_for_binding(donor).pod_ip
            except ValueError:
                pod_ip = None
            if not pod_ip:
                return WakeConflict(
                    f"{receiver.binding_id}: donor {donor.serve_id} has no pod IP",
                    reason="resident_unknown", node=node, gpus=gpus, binding_id=receiver.binding_id,
                    blocking_binding_id=donor.binding_id,
                )
        donors = set(pair.donor_ids)
        try:
            probed = self._probe_gpu_residents(receiver)
        except Exception as exc:  # noqa: BLE001 - a failed Pod LIST: unknown residents
            return WakeConflict(
                f"{receiver.binding_id}: residents cannot be listed ({type(exc).__name__}: {exc})",
                reason="resident_unknown", node=node, gpus=gpus, binding_id=receiver.binding_id,
            )
        residents = [item for item in probed if item[0] not in donors]
        awake = sorted(binding_id for binding_id, sleeping in residents if sleeping is False)
        unknown = sorted(binding_id for binding_id, sleeping in residents if sleeping is None)
        if awake:
            return WakeConflict(
                f"{receiver.binding_id}: resident(s) {awake} on {node}/{','.join(str(g) for g in gpus)} "
                "are awake (besides the donor)",
                reason="resident_awake", node=node, gpus=gpus, binding_id=receiver.binding_id,
                blocking_binding_id=awake[0],
            )
        if unknown:
            return WakeConflict(
                f"{receiver.binding_id}: cannot verify resident(s) {unknown} asleep",
                reason="resident_unknown", node=node, gpus=gpus, binding_id=receiver.binding_id,
                blocking_binding_id=unknown[0],
            )
        return None

    def _transfer_sleep_donors(self, op: "_Transfer") -> None:
        """Every donor of every pair in ONE sleep call (in parallel; writer lock
        held). A pair whose donors did not all sleep fails
        (``donor_sleep_failed``) and its receiver is not touched."""
        donors = [donor for state in op.pairs for donor in state.pair.donors]
        error: BaseException | None = None
        try:
            outcomes = self._sleep_targets(
                self._sleep_targets_for(donors),
                sleep_path=op.sleep_path,
                journal_extra={**DESIRED_ON_SLEEP, "transfer_id": op.transfer_id},
                desired_sleeping=True,
                update_store=True,
            )
        except SleepFailed as exc:
            outcomes, error = exc.outcomes, exc
        by_id = {item.get("binding_id"): item for item in outcomes or ()}
        slept = {binding_id for binding_id, item in by_id.items() if item.get("status") == STATUS_SLEPT}
        op.donors_slept |= slept
        for state in op.pairs:
            missing = [donor for donor in state.pair.donor_ids if donor not in slept]
            if missing:
                outcome = by_id.get(missing[0]) or {}
                state.status = TRANSFER_DONOR_FAILED
                state.error = SleepFailed(
                    f"donor {missing[0]} did not sleep ({outcome.get('status')}: {outcome.get('reason') or error})",
                    outcomes=[outcome] if outcome else [],
                )

    def _transfer_wake_receivers(self, op: "_Transfer") -> None:
        """The wake of every receiver whose donors all slept, in parallel, then
        the commit (writer lock held). A receiver the account / the wake gate
        refuses now, or whose wake fails (S4: an engine that woke anyway gets a
        compensating sleep), fails its pair (``receiver_wake_failed``); its donors
        stay asleep."""
        tickets: list[_WakeTicket] = []
        for state in op.pairs:
            if state.status != TRANSFER_PENDING:
                continue
            try:
                state.ticket = self._prepare_transfer_wake(op, state.pair)
            except Exception as exc:  # noqa: BLE001 - this pair fails, the others go on
                state.status = TRANSFER_RECEIVER_FAILED
                state.error = exc
                continue
            tickets.append(state.ticket)
        if not tickets:
            return
        self._note_wake_details(tickets)
        self._run_wakes(tickets)
        self._commit_wakes(tickets, restore_desired=True, update_store=True)
        for state in op.pairs:
            ticket = state.ticket
            if ticket is None:
                continue
            if ticket.woke:
                state.status = TRANSFER_DONE
            else:
                state.status = TRANSFER_RECEIVER_FAILED
                state.error = ticket.exception or WakeFailed(
                    f"wake of {ticket.binding.serve_id} failed", node=ticket.binding.slot.node,
                    gpus=ticket.binding.slot.gpu_ids, binding_id=ticket.binding.binding_id,
                )

    def _prepare_transfer_wake(self, op: "_Transfer", pair: TransferPair) -> "_WakeTicket":
        """The wake prepare of one receiver (writer lock held): the books must
        still show it asleep and not hidden; the cap, the account and the wake
        gate decide as for any wake; then its desired power is awake."""
        snapshot = self._store.load()
        receiver = next((b for b in snapshot.bindings if b.binding_id == pair.receiver.binding_id), None)
        if receiver is None:
            raise ValueError(f"transfer receiver {pair.receiver.binding_id} is no longer bound")
        if receiver.awake or receiver.hidden:
            raise WakeConflict(
                f"transfer receiver {receiver.serve_id} changed since the selection "
                f"(awake={receiver.awake}, hidden={receiver.hidden})",
                reason="wake_in_progress", node=receiver.slot.node, gpus=receiver.slot.gpu_ids,
                binding_id=receiver.binding_id, blocking_binding_id=receiver.binding_id,
            )
        self._ensure_wake_within_cap(receiver, snapshot.bindings)
        placement = {
            "source": "transfer",
            "transfer_id": op.transfer_id,
            "chosen_binding_id": receiver.binding_id,
            "hint_binding_id": None,
            "donor_binding_ids": list(pair.donor_ids),
        }
        ticket = self._prepare_wake(
            receiver, snapshot.bindings,
            previous_desired=self._desired_power_of(receiver.binding_id), placement=placement,
        )
        try:
            self._update_desired(
                {receiver.binding_id: {"power": "awake"}},
                updated_by="service-manager-transfer",
                reason="transfer",
            )
        except BaseException:
            self._abort_prepared_wakes([ticket])
            raise
        return ticket

    def _finish_transfer(self, op: "_Transfer") -> dict:
        """Counters, the transfer_done / transfer_pair_failed events and the
        response; :class:`TransferFailed` (409 partial) when no pair completed
        and the floor did not hold the whole transfer back."""
        response = self._transfer_response(op)
        done = [state for state in op.pairs if state.status == TRANSFER_DONE]
        failed = [state for state in op.pairs if state.status != TRANSFER_DONE]
        for name, amount in (
            ("transfer_pairs_done_total", len(done)),
            ("transfer_donor_sleep_failed_total",
             sum(1 for state in failed if state.status == TRANSFER_DONOR_FAILED)),
            ("transfer_receiver_wake_failed_total",
             sum(1 for state in failed if state.status == TRANSFER_RECEIVER_FAILED)),
        ):
            if amount:
                self._wake_journal.incr(name, amount)
        for item in response["pairs"]:
            if item["status"] != TRANSFER_DONE:
                _log_event(
                    "transfer_pair_failed", level=logging.WARNING,
                    transfer_id=op.transfer_id, **{k: v for k, v in item.items() if k != "transfer_id"},
                )
        _log_event(
            "transfer_done",
            level=logging.INFO if done else logging.WARNING,
            transfer_id=op.transfer_id, donor_model=op.donor_model, receiver_model=op.receiver_model,
            count=op.count, done=len(done), failed=len(failed), unfilled=response["unfilled"],
            phases_ms=dict(op.phases_ms),
        )
        if done or (not op.pairs and response["clamped_by_floor"]):
            return response
        failed_pair = next((item for item in response["pairs"] if item["status"] != TRANSFER_DONE), None)
        if failed_pair is not None:
            error = failed_pair.get("error") or {}
            first = {
                **error,
                "node": error.get("node") or failed_pair["node"],
                "gpu_ids": error.get("gpu_ids") or failed_pair["gpu_ids"],
                "binding_id": error.get("binding_id") or failed_pair["receiver_binding_id"],
            }
        else:
            first = response["refusals"][0] if response["refusals"] else {}
        raise TransferFailed(
            f"transfer {op.donor_model} -> {op.receiver_model}: no pair completed "
            f"(pairs {[(item['receiver'], item['status']) for item in response['pairs']]}, "
            f"skipped {response['skipped']}, refusals {[r.get('reason') for r in response['refusals']]})",
            response=response, first=first,
        )

    def _transfer_response(self, op: "_Transfer") -> dict:
        selection = op.selection
        pairs = []
        for state in op.pairs:
            item = {**_transfer_pair_brief(state.pair), "status": state.status}
            if state.error is not None:
                item["error"] = self._transfer_error_body(state.error)
            ticket = state.ticket
            if ticket is not None and ticket.compensating_sleep:
                item["compensating_sleep"] = ticket.compensating_sleep
            pairs.append(item)
        done = [state for state in op.pairs if state.status == TRANSFER_DONE]
        clamped = bool(selection is not None and selection.unfilled > 0 and SKIP_FLOOR in selection.skipped)
        return {
            "transfer_id": op.transfer_id,
            "donor_model": op.donor_model,
            "receiver_model": op.receiver_model,
            "count": op.count,
            "sleep_path": op.sleep_path,
            "pairs": pairs,
            "done": len(done),
            # What the transfer took from the donor model (donor replicas put to
            # sleep) and whether the donor's replica floor held part of it back
            # (same fields as a clamped /target shrink).
            "taken": len(op.donors_slept),
            "clamped_by_floor": clamped,
            "donors_slept": len(op.donors_slept),
            "receivers_woken": len(done),
            "unfilled": op.count if selection is None else selection.unfilled,
            "refusals": [] if selection is None else [self._transfer_refusal_body(r) for r in selection.refusals],
            "skipped": {} if selection is None else dict(selection.skipped),
            "picked": [_picked(state.ticket) for state in done if state.ticket is not None],
            "phases_ms": dict(op.phases_ms),
            "version": op.version,
        }

    def _transfer_error_body(self, exc: BaseException) -> dict:
        if isinstance(exc, WakeConflict):
            return exc.body(retry_after_s=self.wake_retry_after_s(exc.scope))
        body = {"detail": str(exc), "error": _error_code(exc)}
        if isinstance(exc, SleepFailed) and exc.outcomes:
            body["outcomes"] = exc.outcomes
        return body

    def _transfer_refusal_body(self, refusal) -> dict:
        if isinstance(refusal.error, WakeConflict):
            return refusal.error.body(retry_after_s=self.wake_retry_after_s(refusal.error.scope))
        return {
            "detail": refusal.detail, "error": "gpu_busy", "reason": refusal.reason,
            "binding_id": refusal.receiver_binding_id, "node": refusal.node,
            "gpu_ids": list(refusal.gpu_ids), "gpu": list(refusal.gpu_ids), "scope": "gpu",
            "blocking_binding_id": refusal.blocking_binding_id,
            "retry_after_s": self.wake_retry_after_s("gpu"),
        }

    def _note_transfer(self, op: "_Transfer", operation) -> None:
        """The transfer on its operation record (``GET /v2/operations``):
        ``details.transfer`` with the pairs and ``details.phases_ms``."""
        note = getattr(operation, "note", None)
        if not callable(note):
            return
        try:
            note(
                transfer={
                    "transfer_id": op.transfer_id,
                    "donor_model": op.donor_model,
                    "receiver_model": op.receiver_model,
                    "count": op.count,
                    "pairs": [{**_transfer_pair_brief(state.pair), "status": state.status} for state in op.pairs],
                    "unfilled": None if op.selection is None else op.selection.unfilled,
                },
                phases_ms=dict(op.phases_ms),
            )
        except Exception:  # noqa: BLE001 - observability only
            LOG.warning("recording the transfer in the operation record failed", exc_info=True)

    def _sleep_bindings(
        self,
        bindings: list[Binding],
        *,
        sleep_path: str,
        drain_budget_s: float | None = None,
    ) -> list[dict]:
        """Every sleep of the service-manager goes through here (plan D1/D2)."""
        if not bindings or self._sleep_primitive is None:
            return []
        return self._sleep_targets(
            self._sleep_targets_for(bindings), sleep_path=sleep_path, drain_budget_s=drain_budget_s
        )

    def _sleep_targets_for(self, bindings: list[Binding]) -> list[SleepTarget]:
        targets: list[SleepTarget] = []
        for binding in bindings:
            snapshot = self._snapshot_for_binding(binding)
            if not snapshot.pod_ip:
                raise ValueError(f"pod {binding.serve_id} has no pod IP for sleep")
            targets.append(SleepTarget(binding, snapshot.pod_ip, snapshot.pod_uid))
        return targets

    def _sleep_targets(
        self,
        targets: list[SleepTarget],
        *,
        sleep_path: str,
        drain_budget_s: float | None = None,
        journal_extra: dict | None = None,
        desired_sleeping: bool = False,
        update_store: bool = False,
    ) -> list[dict]:
        """One sleep call, the writer lock held throughout (whole-lock). The
        targets that slept are recorded - GPU lease released, (``update_store``)
        the legacy store, (``desired_sleeping``) desired power "sleeping" - and
        only then are their journal entries ended: a crash in between leaves the
        entries, and the journal recovery records the sleep from the physical
        state. On a partial failure the error propagates after that (the store is
        always updated then: rolled-back targets get their hidden flag back).
        Without ``update_store`` the caller saves the store itself."""
        del drain_budget_s  # deprecated and ignored: the SM never drains
        if self._sleep_primitive is None:
            raise ValueError("runtime_ops and vllm_ops are required to sleep a pod")
        primitive = self._sleep_primitive
        try:
            outcomes = primitive.sleep(
                targets, path=sleep_path, journal_extra=journal_extra, keep_journal=True
            )
        except SleepFailed as exc:
            self._record_sleep_outcomes(
                targets, exc.outcomes, update_store=True, desired_sleeping=desired_sleeping
            )
            primitive.end_journal(exc.outcomes)
            raise
        self._record_sleep_outcomes(
            targets, outcomes, update_store=update_store, desired_sleeping=desired_sleeping
        )
        primitive.end_journal(outcomes)
        return outcomes

    def _record_sleep_outcomes(
        self,
        targets: list[SleepTarget],
        outcomes: list[dict],
        *,
        update_store: bool,
        desired_sleeping: bool = False,
    ) -> None:
        """Per-target bookkeeping (review P2-7): release the GPU lease of every
        target that slept; with ``update_store`` also mark exactly those asleep in
        the legacy store (and restore the hidden flag of rolled-back ones); with
        ``desired_sleeping`` record desired power "sleeping" for exactly the
        targets that slept (review 2 P2-3: desired changes follow the outcome)."""
        try:
            self._record_sleep_bookkeeping(
                targets, outcomes, update_store=update_store, desired_sleeping=desired_sleeping
            )
        finally:
            # Slept or rolled back: the audit must see what the pods are now (B2).
            self._refresh_observed(target.binding.binding_id for target in targets)

    def _record_sleep_bookkeeping(
        self,
        targets: list[SleepTarget],
        outcomes: list[dict],
        *,
        update_store: bool,
        desired_sleeping: bool,
    ) -> None:
        by_id = {target.binding.binding_id: target.binding for target in targets}
        slept = [
            by_id[item["binding_id"]]
            for item in outcomes
            if item.get("status") == STATUS_SLEPT and item.get("binding_id") in by_id
        ]
        if self._gpu_leases is not None:
            for binding in slept:
                self._gpu_leases.release(binding)
        for binding in slept:
            self._note_binding_power_change(binding)
        if not update_store:
            return
        slept_ids = {binding.binding_id for binding in slept}
        rolled_hidden = {
            item["binding_id"]: item.get("previous_state") == POD_STATE_HIDDEN
            for item in outcomes
            if item.get("status") == "rolled_back"
        }
        for attempt in range(3 if (slept_ids or rolled_hidden) else 0):
            snapshot = self._store.load()
            updated = []
            for binding in snapshot.bindings:
                if binding.binding_id in slept_ids:
                    binding = replace(binding, awake=False, hidden=False)
                elif binding.binding_id in rolled_hidden and binding.awake:
                    # The store may record the hide (a SafeScale probe pod).
                    binding = replace(binding, hidden=rolled_hidden[binding.binding_id])
                updated.append(binding)
            if updated == snapshot.bindings:
                break
            try:
                self._store.save(updated, expected_version=snapshot.version)
                break
            except StateConflict:
                if attempt == 2:
                    raise
        if desired_sleeping and slept_ids and self._fleet_store is not None:
            known = {item.binding_id for item in self._fleet_store.load_desired().bindings}
            self._update_desired(
                {
                    binding_id: {"power": "sleeping", "hidden": False}
                    for binding_id in slept_ids
                    if binding_id in known
                },
                updated_by="service-manager-sleep",
                reason="sleep_committed",
            )

    def _repair_sleep_binding(self, binding: Binding, pod_ip: str) -> None:
        """Fleet repair's sleep: the same primitive (plan D1), path ``repair``,
        under the repair's writer lock like every sleep."""
        self._sleep_targets([SleepTarget(binding, pod_ip)], sleep_path="repair")

    def _ensure_wake_headroom(self, binding: Binding) -> dict:
        """The physical side of a wake's feasibility (S1, 2026-09-30).

        The account (store, GPU leases, journaled wakes; :meth:`_ensure_feasible_wake`)
        decides first; this gate only catches what the account cannot see - a sleep
        leak, a Pod the store does not know. It never waits for a new gpu-truth
        sample:

        * the node's sample (present = within the agent's Redis TTL) is TRUSTED
          unless it predates the last local power change on one of the binding's
          GPUs (:meth:`_note_power_change`: the agent must have answered the refresh
          request sent right after that change);
        * trusted and every GPU at most the wake limit (registry
          ``service_manager.wake.max_used_fraction`` / ``max_used_mib``) -> pass;
          trusted and above it -> refuse (409 ``gpu_truth_used``; a refresh is
          requested so the next attempt sees a new sample);
        * missing, untrusted or incomplete -> neither pass nor refuse on it: probe
          ``/is_sleeping`` of every other resident on those GPUs; all asleep -> pass,
          any awake -> refuse (``resident_awake``), any unknown -> refuse
          (``gpu_truth_unavailable`` when the node has no sample at all - a node
          scope refusal - else ``resident_unknown``).

        ``require_gpu_truth`` false keeps the permissive behaviour for a missing
        sample (pass without probing). A resident probe is logged as
        ``gpu_truth_fallback`` and counted (``truth_fallback_total:<missing|stale|
        incomplete>``). Returns how the gate decided: ``{"truth_source":
        "gpu_truth" | "is_sleeping_probe" | "none", "truth_age_s"}``."""
        if self._gpu_truth is None:
            return {"truth_source": "none", "truth_age_s": None}
        node_name = binding.slot.node
        gpus = tuple(binding.slot.gpu_ids)
        spec = {node.name: node for node in self._registry.topology().nodes}.get(node_name)
        try:
            truth = self._gpu_truth.node_truth(node=node_name)
        except Exception:  # noqa: BLE001 - an unreadable sample is a missing one
            LOG.warning("reading gpu truth of %s failed", node_name, exc_info=True)
            truth = None
        age_s = self._truth_age_s(node_name, truth)
        verdict, detail, kind = self._wake_truth_verdict(binding, spec, truth)
        if verdict == "ok":
            return {"truth_source": "gpu_truth", "truth_age_s": age_s}
        if verdict == "over":
            self._request_truth_refresh(node_name)
            raise WakeConflict(
                f"insufficient wake headroom: {detail}",
                reason="gpu_truth_used", node=node_name, gpus=gpus, binding_id=binding.binding_id,
            )
        if truth is None and not self._require_gpu_truth:
            return {"truth_source": "none", "truth_age_s": None}
        residents = self._probe_gpu_residents(binding)
        awake = sorted(binding_id for binding_id, sleeping in residents if sleeping is False)
        unknown = sorted(binding_id for binding_id, sleeping in residents if sleeping is None)
        conflict = None
        if awake:
            conflict = WakeConflict(
                f"{binding.binding_id}: {detail}; resident(s) {awake} on "
                f"{node_name}/{','.join(str(g) for g in gpus)} are awake",
                reason="resident_awake", node=node_name, gpus=gpus, binding_id=binding.binding_id,
                blocking_binding_id=awake[0],
            )
        elif unknown:
            node_scope = truth is None
            conflict = WakeConflict(
                f"{binding.binding_id}: {detail}; cannot verify resident(s) {unknown} asleep",
                reason="gpu_truth_unavailable" if node_scope else "resident_unknown",
                node=node_name, gpus=gpus, scope="node" if node_scope else "gpu",
                binding_id=binding.binding_id, blocking_binding_id=unknown[0],
            )
        self._wake_journal.incr(f"truth_fallback_total:{kind}")
        _log_event(
            "gpu_truth_fallback",
            level=logging.WARNING if conflict is not None else logging.INFO,
            node=node_name, gpu_ids=list(gpus), binding_id=binding.binding_id, reason=kind,
            detail=detail, residents_probed=[
                {"binding_id": binding_id, "is_sleeping": sleeping} for binding_id, sleeping in residents
            ],
            result="pass" if conflict is None else f"refused:{conflict.error}",
        )
        if conflict is not None:
            raise conflict
        return {"truth_source": "is_sleeping_probe", "truth_age_s": age_s}

    def _truth_age_s(self, node_name: str, node_truth) -> float | None:
        """Seconds since THIS service-manager first saw the node's current sample
        (its publish counters), a clock-skew-free lower bound of the sample's age
        (node clocks drift from the control plane's; the payload timestamp is
        not compared). None without a sample."""
        if node_truth is None:
            return None
        key = (getattr(node_truth, "seq", None), getattr(node_truth, "refresh_seq", None),
               getattr(node_truth, "timestamp", None))
        now = time.monotonic()
        with self._power_marks_lock:
            seen = self._truth_seen.get(node_name)
            if seen is None or seen[0] != key:
                self._truth_seen[node_name] = (key, now)
                return 0.0
            return round(now - seen[1], 1)

    def _wake_truth_verdict(self, binding: Binding, node, node_truth) -> tuple[str, str, str]:
        """("ok" | "over" | "unknown", detail, kind) of the gpu-truth sample for a
        wake of ``binding`` (see :meth:`_ensure_wake_headroom`); ``kind`` of an
        unknown verdict: missing | stale | incomplete."""
        node_name = binding.slot.node
        if node_truth is None:
            return "unknown", (
                f"gpu truth unavailable for node {node_name} "
                "(is the tre-v2-gpu-truth DaemonSet healthy?)"
            ), "missing"
        stale = self._untrusted_gpus(node_name, binding.slot.gpu_ids, node_truth)
        if stale:
            return "unknown", (
                f"gpu truth sample of {node_name} (refresh_seq={node_truth.refresh_seq}, "
                f"seq={node_truth.seq}) predates the last power change on gpu(s) {stale}"
            ), "stale"
        missing: list[str] = []
        for gpu_id in binding.slot.gpu_ids:
            gpu_uuid = _gpu_uuid(node, gpu_id)
            if gpu_uuid is None:
                missing.append(f"{node_name}/{gpu_id} has no GPU UUID in the registry")
                continue
            used_mib = node_truth.used_mib(gpu_uuid)
            if used_mib is None:
                missing.append(f"gpu truth for {node_name}/{gpu_uuid} missing")
                continue
            total = getattr(node_truth, "total_mib", None)
            limit = self._sm_config.wake_limit_mib(total(gpu_uuid) if callable(total) else None)
            if limit is None:
                missing.append(
                    f"gpu truth for {node_name}/{gpu_uuid} reports no total memory "
                    "(set service_manager.wake.max_used_mib to use an absolute threshold)"
                )
                continue
            if used_mib > limit:
                return "over", (
                    f"{binding.binding_id}: {node_name}/{gpu_uuid} "
                    f"used_mib={used_mib} > wake limit {limit} MiB"
                ), "over"
        if missing and self._require_gpu_truth:
            return "unknown", "; ".join(missing), "incomplete"
        return "ok", "gpu truth within the wake limit", "ok"

    def _untrusted_gpus(self, node_name: str, gpu_ids, node_truth) -> list[int]:
        """GPUs whose last local power change the sample ``node_truth`` does not
        reflect yet: it must answer the refresh request sent after the change
        (``refresh_seq``), or - an agent without refreshes - be a later publish
        (``seq``); a sample with neither is untrusted after any change."""
        suspect = {
            int(gpu)
            for node, gpus, _pod in list(self._suspects.values())
            if node == node_name
            for gpu in gpus
        }
        with self._power_marks_lock:
            marks = {gpu: self._power_marks.get((node_name, int(gpu))) for gpu in gpu_ids}
        refresh_seq = getattr(node_truth, "refresh_seq", None)
        seq = getattr(node_truth, "seq", None)
        stale: list[int] = [int(gpu) for gpu in gpu_ids if int(gpu) in suspect]
        for gpu, mark in marks.items():
            if int(gpu) in suspect:
                continue
            if mark is None:
                continue
            if mark.refresh_seq is not None and isinstance(refresh_seq, int):
                if refresh_seq < mark.refresh_seq:
                    stale.append(int(gpu))
                continue
            if mark.seq is not None and isinstance(seq, int):
                if seq <= mark.seq:
                    stale.append(int(gpu))
                continue
            stale.append(int(gpu))
        return stale

    def _note_power_change(self, node_name: str, gpu_ids) -> None:
        """A local power change on ``node_name``/``gpu_ids`` (sleep commit, wake,
        start, failed start, deletion; S1): a gpu-truth sample taken before it no
        longer describes those GPUs. Asks the node's agent for a new sample (INCR,
        never waits) and records what a sample must answer to be trusted again.
        Best effort."""
        if self._gpu_truth is None or not gpu_ids:
            return
        requested = self._request_truth_refresh(node_name)
        seq = None
        try:
            truth = self._gpu_truth.node_truth(node=node_name)
            value = getattr(truth, "seq", None) if truth is not None else None
            seq = value if isinstance(value, int) and not isinstance(value, bool) else None
        except Exception:  # noqa: BLE001 - the refresh counter alone is enough
            seq = None
        with self._power_marks_lock:
            for gpu in gpu_ids:
                key = (node_name, int(gpu))
                self._power_marks[key] = _PowerMark.merged(self._power_marks.get(key), requested, seq)

    def _note_binding_power_change(self, binding: Binding) -> None:
        self._note_power_change(binding.slot.node, binding.slot.gpu_ids)

    def _probe_gpu_residents(self, binding: Binding) -> list[tuple[str, bool | None]]:
        """(binding id, /is_sleeping) of every other Pod on ``binding``'s GPUs; a
        Pod without an IP or not Ready (e.g. still loading) is unknown (None).
        The probes run in parallel (one probe timeout under the writer lock,
        whatever the number of residents)."""
        lister = getattr(self._runtime_ops, "list_pod_snapshots", None)
        if not callable(lister) or self._vllm_ops is None:
            return [("<no runtime to probe residents>", None)]
        wanted = set(binding.slot.gpu_ids)
        found: list[tuple[str, object]] = []
        for snapshot in lister():
            if snapshot.name == binding.serve_id or snapshot.node != binding.slot.node:
                continue
            try:
                other = _binding_from_snapshot(snapshot)
            except (KeyError, ValueError):
                continue
            if wanted.intersection(other.slot.gpu_ids):
                found.append((other.binding_id, snapshot))

        def probe(item) -> bool | None:
            snapshot = item[1]
            if not (snapshot.pod_ip and snapshot.ready):
                return None
            try:
                return self._vllm_ops.is_sleeping(snapshot.pod_ip, port=8000)
            except Exception:  # noqa: BLE001 - unreachable = unknown
                return None

        states = _parallel(probe, found)
        return [(binding_id, states[id(item)]) for item in found for binding_id in [item[0]]]

    def _gpu_truth_gate(self, node_name: str, problem, *, retry_stale: bool, what: str) -> str | None:
        """Evaluate ``problem(node_truth)`` (None = pass, else the reason) on a
        gpu-truth sample taken AFTER this call started; returns the reason to
        refuse, or None. Used by the cold-start gate only (a cold start writes a
        whole model onto the GPU; the wake gate never waits, see
        :meth:`_ensure_wake_headroom`).

        The agent samples periodically, so right after a sleep (or a pod deletion)
        on the same GPU the stored sample still shows the previous occupant. The
        gate asks the node's agent for a sample (INCR
        ``tre:gpu_truth_refresh:<node>`` -> N) and waits, up to
        ``service_manager.wake.truth_wait_s``, for a payload with
        ``refresh_seq >= N``:

        * fresh and fine -> pass; fresh with a problem -> ask again (memory may
          still be being released) until the deadline, then refuse;
        * the agent does not serve refreshes (no ``refresh_seq`` in the payload,
          e.g. an agent older than this protocol during a rollout) or the request
          failed: the previous behaviour - a fine sample passes, a problem
          refuses at once, or (``retry_stale``) after re-reading until the
          deadline;
        * no fresh sample by the deadline: a fine (older, TTL-valid) sample
          passes with a warning, a problem refuses.

        Missing truth is a problem unless ``require_gpu_truth`` is off: the gate
        stays fail-closed.
        """
        clock = self._sleep_clock or _default_sleep_clock()
        deadline = clock.monotonic() + self._sm_config.wake_truth_wait_s
        requested = self._request_truth_refresh(node_name)
        while True:
            node_truth = self._gpu_truth.node_truth(node=node_name)
            issue = problem(node_truth)
            refresh_seq = getattr(node_truth, "refresh_seq", None)
            serves_refresh = requested is not None and isinstance(refresh_seq, int)
            fresh = serves_refresh and refresh_seq >= requested
            if issue is None and (fresh or not serves_refresh):
                return None
            if not serves_refresh and not retry_stale:
                return issue
            if clock.monotonic() >= deadline:
                if issue is None:
                    LOG.warning(
                        "%s: no gpu-truth sample of %s answered refresh %s within %.1fs; "
                        "using the last sample (refresh_seq=%s)",
                        what, node_name, requested, self._sm_config.wake_truth_wait_s, refresh_seq,
                    )
                return issue
            if fresh:
                requested = self._request_truth_refresh(node_name) or requested
            clock.sleep(min(TRUTH_POLL_S, max(0.0, deadline - clock.monotonic())))

    def _request_truth_refresh(self, node_name: str) -> int | None:
        request = getattr(self._gpu_truth, "request_refresh", None)
        if not callable(request):
            return None
        try:
            value = request(node=node_name)
        except Exception:
            LOG.warning("gpu-truth refresh request for %s failed", node_name, exc_info=True)
            return None
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def wake_retry_after_s(self, scope: str = "gpu") -> float:
        """How long a caller should keep off a GPU (``scope`` gpu) or a node after
        a refused / failed wake: registry ``placement.wake_cooldown`` - the same
        values the controller's cooldown uses (S3)."""
        placement = getattr(self._registry, "placement", None)
        config = placement() if callable(placement) else None
        if scope == "node":
            return float(getattr(config, "wake_cooldown_node_s", 60.0))
        return float(getattr(config, "wake_cooldown_gpu_s", 30.0))

    def wake_state(self) -> dict:
        """``GET /v2/wake``: wake counters (wake_done_total, wake_failed_total,
        wake_parallel_max, wake_compensating_sleep_total,
        truth_fallback_total:<reason>) and the wakes the journal holds (in
        progress under the writer lock, or left for the journal recovery)."""
        return {
            "stats": self._wake_journal.stats(),
            "in_progress": self._wake_journal.entries(),
            "workers": WAKE_WORKERS,
            "wake_call_timeout_s": float(getattr(self._sm_config, "wake_call_timeout_s", 10.0)),
            "retry_after_s": {"gpu": self.wake_retry_after_s("gpu"), "node": self.wake_retry_after_s("node")},
            "test_hooks": bool(getattr(self._sm_config, "test_hooks", False)),
        }

    def sleep_state(self) -> dict:
        """Counters, in-progress journal and recent outcomes of the sleep primitive."""
        if self._sleep_primitive is None:
            return {"configured": False}
        primitive = self._sleep_primitive
        latencies = sorted(primitive.journal.ack_latencies_ms())
        policy = primitive.policy
        return {
            "configured": True,
            "policy": {
                "ack_timeout_s": policy.ack_timeout_s,
                "instance_staleness_s": policy.instance_staleness_s,
                "gateway_min_instances": policy.gateway_min_instances,
                "fallback_no_plugin": policy.fallback_no_plugin,
                "no_plugin_grace_s": policy.no_plugin_grace_s,
                "sleep_call_timeout_s": policy.sleep_call_timeout_s,
                "physical_confirm_timeout_s": policy.physical_confirm_timeout_s,
                "probe_timeout_s": policy.probe_timeout_s,
                "sleep_mode_when_idle": policy.sleep_mode_when_idle,
                "writer_lock_wait_s": self._sm_config.writer_lock_wait_s,
                "shutdown_timeout_s": self._sm_config.shutdown_timeout_s(),
                "vllm_sleep_mode_param": policy.vllm_sleep_mode_param,
                "worst_case_sleep_lock_s": self._sm_config.worst_case_sleep_lock_s(),
                "worst_case_wake_lock_s": self._sm_config.worst_case_wake_lock_s(),
                "worst_case_transfer_lock_s": self._sm_config.worst_case_transfer_lock_s(),
                "worst_case_call_s": self._sm_config.worst_case_sleep_call_s(),
            },
            "shutting_down": primitive.shutting_down,
            "stats": primitive.journal.stats(),
            "in_progress": primitive.journal.entries(),
            "ack_latency_ms": {
                "count": len(latencies),
                "p50": _quantile(latencies, 0.50),
                "p99": _quantile(latencies, 0.99),
                "max": latencies[-1] if latencies else None,
            },
            "recent": primitive.recent(),
            "floor": self.floor_state(),
        }

    # ------------------------------------------------------------ replica floor
    def floor_state(self) -> dict:
        """Replica-floor switch, counters and recent events (``GET /v2/sleep``)."""
        return {
            "enforced": self._floor_enforced(),
            "counts": self._floor_recorder.counts(),
            "recent": self._floor_recorder.recent(),
        }

    def _floor_enforced(self) -> bool:
        return bool(getattr(self._sm_config, "replica_floor_enforce", True))

    def _model_floor(self, model: str) -> int:
        """The registry ``min_replicas`` of ``model`` (TRE and APA use the same)."""
        try:
            return max(0, int(self._registry.model(model).min_replicas))
        except (KeyError, AttributeError, TypeError, ValueError):
            return 0

    def _routable_binding_ids(
        self,
        model: str,
        bindings: list[Binding] | None = None,
        *,
        live: list | None | bool = False,
        transient: set[str] | None = None,
    ) -> set[str]:
        """Binding ids of ``model`` that serve traffic right now.

        A replica counts only when BOTH views agree (review 2026-09-29 P3-9):

        * the SM store (``bindings``, default a fresh load) records it awake and not
          hidden - a Pod the store does not know does not count;
        * with a Kubernetes view, its Pod is Ready and carries the routable label
          ``true`` (what the gateway routes on) - a Pod left hidden by an
          unconfirmed sleep does not count;

        and it has no unexpired transient ``starting`` (or, from before
        2026-10-02, ``waking``) GPU lease. Pods are matched by binding id, never
        by pod name (a replaced Pod keeps its binding). A make-before-break move
        counts its destination explicitly (:meth:`_floor_counting`): the store
        records it only once the move is done, but it must be Ready and routable
        in the live view.

        ``live`` / ``transient``: pre-read Pod snapshots (every model; None = no
        live view) and transient-lease ids, so ``GET /v2/state`` reads them once
        for all models (default: read here). Read errors propagate (the floor
        check turns them into 409 ``routable_unknown``)."""
        if bindings is None:
            bindings = self._store.load().bindings
        candidates = {
            binding.binding_id
            for binding in bindings
            if binding.model == model and binding.awake and not binding.hidden
        }
        candidates |= {
            binding_id for extra_model, binding_id in _FLOOR_EXTRA_ROUTABLE.get()
            if extra_model == model
        }
        candidates -= self._transient_lease_ids() if transient is None else set(transient)
        if live is False:
            lister = getattr(self._runtime_ops, "list_pod_snapshots", None)
            if self._runtime_ops is None or not callable(lister):
                return candidates
            snapshots = lister(model=model)
        elif live is None:
            return candidates
        else:
            snapshots = live
        live_ids: set[str] = set()
        for snapshot in snapshots:
            if snapshot.model != model or not snapshot.ready or snapshot.routable is not True:
                continue
            try:
                live_ids.add(_binding_from_snapshot(snapshot).binding_id)
            except (KeyError, ValueError):
                continue
        return candidates & live_ids

    def _routable_by_model(self, bindings: list[Binding]) -> tuple[dict[str, set[str]] | None, str | None]:
        """:meth:`_routable_binding_ids` of every model on one store snapshot, with
        ONE live Pod listing and one lease read shared by all models (``GET
        /v2/state`` only). (None, error) when one of them cannot be read."""
        models = {model.name for model in self._registry.models()} | {b.model for b in bindings}
        try:
            transient = self._transient_lease_ids()
            lister = getattr(self._runtime_ops, "list_pod_snapshots", None)
            live = list(lister()) if self._runtime_ops is not None and callable(lister) else None
            return {
                model: self._routable_binding_ids(model, bindings, live=live, transient=transient)
                for model in sorted(models)
            }, None
        except Exception as exc:  # noqa: BLE001 - /v2/state never fails on it
            LOG.warning("computing the routable view for /v2/state failed", exc_info=True)
            return None, f"{type(exc).__name__}: {exc}"

    def _transient_lease_ids(self) -> set[str]:
        """Bindings holding an UNEXPIRED transient (``waking`` / ``starting``) GPU
        lease on any of their GPUs (a TP>1 binding holds one record per GPU)."""
        if self._gpu_leases is None:
            return set()
        now_ms = int(time.time() * 1000)
        return {
            lease.binding_id
            for lease in self._gpu_leases.load()
            if lease.phase in TRANSIENT_LEASE_PHASES
            and not _lease_expired(lease, now_ms)
        }

    @contextmanager
    def _floor_counting(self, binding: Binding):
        """Within the block, floor checks count ``binding`` as routable when the live
        view says so although the store does not record it awake yet (the
        destination of a make-before-break move)."""
        token = _FLOOR_EXTRA_ROUTABLE.set(
            _FLOOR_EXTRA_ROUTABLE.get() | {(binding.model, binding.binding_id)}
        )
        try:
            yield
        finally:
            _FLOOR_EXTRA_ROUTABLE.reset(token)

    def _floor_check(
        self, model: str, removing, bindings: list[Binding] | None = None
    ) -> FloorCheck:
        """``removing``: binding ids the operation takes out of routing. The
        routable view unreadable (a failed Pod LIST, Redis) -> 409
        ``routable_unknown`` before anything is hidden (fail closed)."""
        try:
            routable = self._routable_binding_ids(model, bindings)
        except Exception as exc:  # noqa: BLE001 - structured refusal
            raise WakeConflict(
                f"replica floor of {model}: the routable view cannot be read "
                f"({type(exc).__name__}: {exc}); nothing changed",
                reason="routable_unknown",
            ) from exc
        return check_floor(model, self._model_floor(model), routable, removing)

    def _on_floor_violation(self, check: FloorCheck, path: str, **extra) -> None:
        """A path's answer to a floor violation: ``repair`` is exempt; ``startup``
        (whose make-up wake could not restore the floor) is exempt too - never
        RetryLater, a Pod held in its startup gate would be seen as fleet drift and
        trigger a fleet-wide repair (review 2026-09-29 P1-1); both are recorded.
        Everything else is refused."""
        if path in EXEMPT_PATHS or path in MAKEUP_PATHS:
            self._floor_recorder.record("exempt", check, path=path, **extra)
            return
        self._floor_recorder.record("rejected", check, path=path, **extra)
        raise FloorViolation(check, path=path)

    def _sleep_floor_guard(self, targets: list[SleepTarget], path: str) -> None:
        """The sleep primitive's floor check: under the writer lock, right before
        the targets are hidden (SleepPrimitive.sleep). An unreadable routable
        view refuses (409 ``routable_unknown``) except on the exempt / make-up
        paths (repair, startup: never refused, 2026-09-29 P1-1)."""
        if not self._floor_enforced():
            return
        by_model: dict[str, set[str]] = {}
        for target in targets:
            by_model.setdefault(target.binding.model, set()).add(target.binding.binding_id)
        notes = _FLOOR_EVENT_NOTES.get()
        for model, binding_ids in sorted(by_model.items()):
            try:
                check = self._floor_check(model, binding_ids)
            except WakeConflict as exc:
                if path in EXEMPT_PATHS or path in MAKEUP_PATHS:
                    LOG.warning("floor check of %s skipped for path %s: %s", model, path, exc)
                    continue
                raise
            if not check.ok:
                self._on_floor_violation(check, path, **(notes.get(model) or {}))

    def _clamp_sleep_plan_to_floor(
        self, model: str, plan: dict, bindings: list[Binding], response: dict, *, path: str
    ) -> dict:
        """A model-level shrink (``/target``, every path since 2026-10-02): keep
        awake as many of the planned sleeps as the floor needs (routable ones;
        hidden / not-yet-serving ones still sleep)."""
        removing = {binding.binding_id for binding in plan["sleep"]}
        check = self._floor_check(model, removing, bindings)
        if check.ok:
            return plan
        allowed = max(0, len(check.routable) - check.floor)
        routable = set(check.routable)
        keep: list[Binding] = []
        kept_awake: list[Binding] = []
        taken = 0
        for binding in plan["sleep"]:
            if binding.binding_id in routable:
                if taken < allowed:
                    taken += 1
                    keep.append(binding)
                else:
                    kept_awake.append(binding)
                    continue
            else:
                keep.append(binding)
        kept_ids = sorted(binding.serve_id for binding in kept_awake)
        self._floor_recorder.record(
            "clamped", check, path=path, requested_sleep=sorted(removing), kept_awake=kept_ids
        )
        response["floor"] = {"clamped": True, "kept_awake": kept_ids, **check.as_dict()}
        return {
            **plan,
            "sleep": keep,
            "target_bindings": list(plan["target_bindings"]) + kept_awake,
        }

    def _startup_floor_makeup(self, pod: StartupPodRecord, residents: list[Binding]) -> dict:
        """Before a startup admission sleeps awake residents: wake another replica of
        every model the sleep would take below its floor (not on the startup Pod's
        GPUs), best effort.

        Runs in its OWN writer-lock hold (review 2026-09-29 P2-4), before the
        sleep's; the sleep only re-checks the floor. Whatever changed in between (another
        writer hid or slept the made-up replica) is caught by that re-check, which
        exempts and records (startup never waits on the floor, P1-1). Returns the
        per-model notes (why a make-up failed) for the exemption record."""
        if not self._floor_enforced() or not residents:
            return {}
        by_model: dict[str, list[Binding]] = {}
        for binding in residents:
            by_model.setdefault(binding.model, []).append(binding)

        def short_models() -> list[tuple[str, list[Binding], FloorCheck]]:
            result = []
            for model, members in sorted(by_model.items()):
                check = self._floor_check(model, {b.binding_id for b in members})
                if check.deficit > 0:
                    result.append((model, members, check))
            return result

        if not short_models():  # nothing to make up: no extra writer phase
            return {}
        notes: dict[str, dict] = {}
        try:
            with self._writer("startup_floor_makeup"):
                # SM actuation observe: a make-up wake is actuation (same refusal
                # as the sleep it prepares).
                self._refuse_unrequested_startup_sleep(pod, residents)
                for model, members, check in short_models():  # again, under the lock
                    removing = [b for b in members if b.binding_id in set(check.removing)]
                    for resident in removing[: check.deficit]:
                        woken, reason = self._makeup_wake(model, pod, check, for_binding=resident)
                        if woken is None:
                            notes[model] = {"makeup_failed": reason}
                            break
        except (RetryLater, ServiceShuttingDown):
            raise
        except Exception as exc:  # noqa: BLE001 - best effort; the prepare exempts
            LOG.warning("floor make-up for the startup of %s failed: %s", pod.name, exc)
            for binding in residents:
                notes.setdefault(binding.model, {"makeup_failed": f"{type(exc).__name__}: {exc}"})
        return notes

    def _makeup_wake(
        self, model: str, pod: StartupPodRecord, check: FloorCheck, *, for_binding: Binding
    ) -> tuple[Binding | None, str | None]:
        """Wake one sleeping replica of ``model`` for the startup of ``pod`` (which
        suspends ``for_binding``). Obeys max_awake_replicas (review 2026-09-29
        P2-3: over the cap = no make-up). The desired record goes first, inside a
        desired guard: a binding without one is refused before anything wakes, a
        failed wake restores it. Its reason names the suspended resident and the
        Pod UID, so the admission's convergence can swap them (see
        :meth:`_swap_with_floor_makeup`). Returns (binding, None) or (None, why)."""
        snapshot = self._store.load()
        pod_gpus = set(pod.gpu_ids)
        leases = self._active_leases()
        journal = self._wake_journal.entries()
        candidates: list[Binding] = []
        for binding in snapshot.bindings:
            if binding.model != model or binding.awake:
                continue
            if binding.slot.node == pod.node and pod_gpus.intersection(binding.slot.gpu_ids):
                continue
            try:
                if self._wake_blocker(binding, snapshot.bindings, leases, journal) is not None:
                    continue
            except WakeConflict:
                continue
            candidates.append(binding)
        if not candidates:
            return None, "no_wakeable_replica"
        try:
            self._ensure_wake_within_cap(candidates[0], snapshot.bindings)
        except ValueError as exc:
            LOG.warning("floor make-up wake for %s refused: %s", model, exc)
            return None, "max_awake_replicas"
        reason = f"{FLOOR_MAKEUP_REASON}:{for_binding.binding_id}:{pod.uid}"
        last_error = "wake_failed"
        while candidates:
            binding = _wake_pick(candidates, snapshot.bindings, self._registry.topology(), self._placement)
            candidates.remove(binding)
            woke = False
            try:
                with self._desired_guard([binding.binding_id], reason=FLOOR_MAKEUP_REASON) as guard:
                    self._update_desired(
                        {binding.binding_id: {"power": "awake", "hidden": False}},
                        updated_by=FLOOR_UPDATED_BY,
                        reason=reason,
                    )
                    self._apply_runtime_power_action(binding, action="wake")
                    woke = True
                    guard.settle([binding.binding_id])  # awake now: keep its desired
                    latest = self._store.load()
                    self._store.save(
                        [
                            replace(item, awake=True, hidden=False)
                            if item.binding_id == binding.binding_id
                            else item
                            for item in latest.bindings
                        ],
                        expected_version=latest.version,
                    )
            except Exception as exc:  # noqa: BLE001 - try the next candidate
                if not woke:
                    LOG.warning("floor make-up wake of %s failed: %s", binding.serve_id, exc)
                    last_error = f"{type(exc).__name__}: {exc}"
                    continue
                # Woken (desired awake, settled); only the store save failed - the
                # reconcile catches the store up. It is a make-up all the same.
                LOG.warning("recording the floor make-up wake of %s failed: %s", binding.serve_id, exc)
            self._floor_recorder.record(
                "makeup_wake", check, path="startup", woken=binding.binding_id,
                for_binding=for_binding.binding_id, for_pod=pod.name,
            )
            return binding, None
        return None, last_error

    def _swap_with_floor_makeup(self, binding_id: str, pod_uid: str | None) -> bool:
        """Review 2026-09-29 P2-3: a resident suspended by a startup whose model got a
        make-up replica for it (desired reason ``replica_floor_makeup:<id>:<uid>``,
        still desired and stored awake) is NOT woken back: its desired power becomes
        sleeping and the make-up replica stays awake - the model ends with as many
        awake replicas as before the start (waking it back would add one, above what
        the controller asked and possibly above max_awake_replicas, and would wake
        it on a GPU the new Pod now uses). Returns True when swapped."""
        if self._fleet_store is None or not pod_uid:
            return False
        reason = f"{FLOOR_MAKEUP_REASON}:{binding_id}:{pod_uid}"
        stored_awake = {
            item.binding_id for item in self._store.load().bindings if item.awake
        }
        partner = next(
            (
                item
                for item in self._fleet_store.load_desired().bindings
                if item.reason == reason
                and item.updated_by == FLOOR_UPDATED_BY
                and item.lifecycle == "resident"
                and item.power == "awake"
                and item.binding_id in stored_awake
            ),
            None,
        )
        if partner is None:
            return False
        self._update_desired(
            {binding_id: {"power": "sleeping", "hidden": False}},
            updated_by=FLOOR_UPDATED_BY,
            reason=f"replica_floor_swap:{partner.binding_id}",
        )
        LOG.warning(
            json.dumps(
                {
                    "event": "replica_floor_makeup_swap",
                    "suspended": binding_id,
                    "kept_awake": partner.binding_id,
                    "pod_uid": pod_uid,
                },
                sort_keys=True,
            )
        )
        return True

    def _record_repair_floor_exemptions(self, snapshots: list[K8sPodSnapshot]) -> None:
        """Fleet repair quarantines (hides) every resident at once: exempt from the
        floor, but recorded per model it takes below its floor."""
        if not self._floor_enforced():
            return
        by_model: dict[str, set[str]] = {}
        for snapshot in snapshots:
            if snapshot.ready and snapshot.routable is True:
                try:
                    binding_id = _binding_from_snapshot(snapshot).binding_id
                except (KeyError, ValueError):
                    continue
                by_model.setdefault(snapshot.model, set()).add(binding_id)
        for model, binding_ids in sorted(by_model.items()):
            check = check_floor(model, self._model_floor(model), binding_ids, binding_ids)
            if not check.ok:
                self._on_floor_violation(check, "repair", reason="fleet_repair_quarantine")

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
        self._ensure_create_headroom(slot, model)
        planned = Binding("startup", model, slot, awake=False)
        if self._gpu_leases is not None:
            self._gpu_leases.acquire(planned, phase="starting")
        self._ensure_model_route(model)
        return self._start_gated_binding(planned)

    def _start_gated_binding(self, planned: Binding) -> Binding:
        """Create the Deployment of ``planned`` (its ``starting`` lease held) and
        bring its Pod up awake, with the Pod pre-authorized at the startup gate
        (review 4 P1: the writer lock is held throughout). On any failure the
        created Deployment is deleted again."""
        model, slot = planned.model, planned.slot
        created = False
        try:
            with self._starting_binding(planned) as note_pod:
                deployment_id = self._runtime_ops.create_model_deployment(model, slot)
                created = True
                self._ensure_model_route(model)
                ready = self._runtime_ops.wait_pod_ready(deployment_id)
                note_pod(getattr(ready, "pod_uid", None))
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
        except BaseException:
            self._discard_failed_start(planned, created)
            self._note_binding_power_change(planned)
            self._refresh_observed([planned.binding_id])
            raise
        self._finish_admitted_start(ready.name)
        self._note_binding_power_change(binding)
        self._refresh_observed([binding.binding_id])
        return binding

    def _deployed_binding_ids(self) -> set[str]:
        lister = getattr(self._runtime_ops, "list_model_deployments", None)
        if not callable(lister):
            return set()
        return {item.binding_id for item in lister()}

    def _execute_power_defrag_migration(self, binding: Binding, destination: Binding) -> list[dict]:
        """Full layout (B6): the destination slot already hosts a sleeping
        binding of the same model with its own Deployment. Sleep the source
        through the sleep primitive (path ``defrag``; its Deployment stays),
        then wake the destination through the normal wake path (the account,
        the wake headroom gate, GPU leases). A failed destination wake wakes
        the source again (best effort) and re-raises: the caller's desired
        guard restores both desired records."""
        if destination.awake:
            raise ValueError(f"defrag destination {destination.binding_id} is already awake")
        if self._runtime_ops is not None and self._vllm_ops is not None:
            # Refuse before the source is slept when the destination cannot wake.
            if not self._snapshot_for_binding(destination).pod_ip:
                raise ValueError(f"pod {destination.serve_id} has no pod IP for wake")
            # The destination's wake headroom gate (fresh gpu-truth sample, B3),
            # before anything changes; the wake below runs it again.
            self._ensure_wake_headroom(destination)
        if _slots_overlap(binding.slot, destination.slot):
            # The destination shares a GPU with the source: it can only wake after
            # the source slept (break-before-make; the replica floor may refuse it).
            self._apply_runtime_power_action(binding, action="sleep", sleep_path="defrag")
            try:
                self._apply_runtime_power_action(destination, action="wake")
            except BaseException:
                try:
                    self._apply_runtime_power_action(binding, action="wake")
                except Exception:  # the audit reports desired awake / physically asleep
                    LOG.exception(
                        "waking defrag source %s back after the failed wake of %s failed",
                        binding.binding_id, destination.binding_id,
                    )
                raise
            return [
                {"action": "hide", "serve_id": binding.serve_id},
                {"action": "sleep", "serve_id": binding.serve_id},
                {"action": "wake", "serve_id": destination.serve_id},
            ]
        # Make-before-break (replica floor, 2026-09-29): the destination wakes
        # first, so the model never has one routable replica fewer while it moves
        # (max_awake_replicas may be exceeded by one for the duration of the move).
        self._apply_runtime_power_action(destination, action="wake")
        try:
            with self._floor_counting(destination):
                self._apply_runtime_power_action(binding, action="sleep", sleep_path="defrag")
        except BaseException as exc:
            if not self._defrag_source_still_awake(binding, exc):
                # Review 2026-09-29 P2-5: the source slept (or its state is unknown):
                # the destination is the replica that serves now - keep it.
                _mark_defrag_source_slept(exc)
                raise
            # The source stays awake (the primitive rolled its hide back): put the
            # destination back to sleep so the layout is what it was.
            try:
                self._apply_runtime_power_action(destination, action="sleep", sleep_path="defrag")
            except Exception:  # the audit reports desired sleeping / physically awake
                LOG.exception(
                    "sleeping defrag destination %s back after the failed sleep of %s failed",
                    destination.binding_id, binding.binding_id,
                )
            raise
        return [
            {"action": "wake", "serve_id": destination.serve_id},
            {"action": "hide", "serve_id": binding.serve_id},
            {"action": "sleep", "serve_id": binding.serve_id},
        ]

    def _execute_runtime_defrag_migration(self, binding: Binding, migration: Migration) -> tuple[list[dict], Binding]:
        if self._runtime_ops is None or self._vllm_ops is None:
            raise ValueError("runtime_ops and vllm_ops are required for runtime defrag")

        actions: list[dict] = []
        if _slots_overlap(binding.slot, migration.to_slot):
            # The new slot shares a GPU with the source: break-before-make (the
            # replica floor may refuse the source's sleep).
            self._sleep_and_delete_defrag_source(binding, actions)
            moved = self._start_defrag_destination(binding, migration, actions)
            return actions, moved
        # Make-before-break (replica floor, 2026-09-29): the new replica is up and
        # routable before the source is hidden and slept.
        moved = self._start_defrag_destination(binding, migration, actions)
        try:
            with self._floor_counting(moved):
                self._sleep_and_delete_defrag_source(binding, actions)
        except BaseException as exc:
            # Review 2026-09-29 P2-5: the new replica is removed again ONLY while the
            # source has not slept (confirmed awake). Once the source slept - a later
            # step (deleting its Deployment, waiting for its Pod) failed - or its
            # state is unknown, the new replica is what serves: keep it.
            source_slept = any(
                item.get("action") == "sleep" for item in actions
            ) or not self._defrag_source_still_awake(binding, exc)
            if source_slept:
                _mark_defrag_source_slept(exc)
            else:
                # The source still serves: remove the new replica again (best effort).
                try:
                    self._runtime_ops.delete_model_deployment(moved)
                    self._runtime_ops.wait_pod_deleted(moved.serve_id)
                    if self._gpu_leases is not None:
                        self._gpu_leases.release(moved)
                except Exception:  # the audit reports the extra Deployment
                    LOG.exception(
                        "removing defrag destination %s after the failed sleep of %s failed",
                        moved.binding_id, binding.binding_id,
                    )
                finally:
                    self._refresh_observed([moved.binding_id])
            raise
        return actions, moved

    def _defrag_source_still_awake(self, binding: Binding, exc: BaseException) -> bool:
        """True only when the defrag source's sleep is confirmed NOT to have happened:
        a floor refusal (nothing was hidden), or a failed sleep whose outcome does not
        say slept and whose Pod vLLM reports awake. Unknown = False (keep the
        destination: two replicas for a while beat none)."""
        if isinstance(exc, FloorViolation):
            return True
        for outcome in getattr(exc, "outcomes", None) or []:
            if (
                isinstance(outcome, dict)
                and outcome.get("binding_id") == binding.binding_id
                and outcome.get("status") == STATUS_SLEPT
            ):
                return False
        if self._vllm_ops is None:
            return False
        try:
            pod_ip = self._snapshot_for_binding(binding).pod_ip
            if not pod_ip:
                return False
            return self._vllm_ops.is_sleeping(pod_ip, port=8000) is False
        except Exception:  # noqa: BLE001 - unknown: keep the destination
            return False

    def _sleep_and_delete_defrag_source(self, binding: Binding, actions: list[dict]) -> None:
        # The primitive hides, waits for the gateway ack and drains before /sleep.
        self._apply_runtime_power_action(binding, action="sleep", sleep_path="defrag")
        actions.append({"action": "hide", "serve_id": binding.serve_id})
        actions.append({"action": "sleep", "serve_id": binding.serve_id})

        self._runtime_ops.delete_model_deployment(binding)
        actions.append({"action": "delete_deployment", "serve_id": binding.serve_id})

        self._runtime_ops.wait_pod_deleted(binding.serve_id)
        self._note_binding_power_change(binding)  # its sleeping footprint is gone too (S1)
        self._refresh_observed([binding.binding_id])  # its pod is gone (B2)

    def _start_defrag_destination(
        self, binding: Binding, migration: Migration, actions: list[dict]
    ) -> Binding:
        self._ensure_create_headroom(migration.to_slot, binding.model)
        planned = Binding(
            "defrag-startup", binding.model, migration.to_slot, awake=False
        )
        if self._gpu_leases is not None:
            self._gpu_leases.acquire(planned, phase="starting")
        moved = self._start_gated_binding(planned)
        new_serve_id = moved.serve_id
        actions.append(
            {
                "action": "create_deployment",
                "serve_id": new_serve_id,
                "node": migration.to_slot.node,
                "gpu_ids": list(migration.to_slot.gpu_ids),
            }
        )
        actions.append({"action": "wake", "serve_id": new_serve_id})
        actions.append({"action": "unhide", "serve_id": new_serve_id})
        return moved

    def _ensure_feasible_wake(
        self, binding: Binding, bindings: list[Binding], leases=None, journal=None
    ) -> None:
        conflict = self._wake_blocker(binding, bindings, leases, journal)
        if conflict is not None:
            raise conflict

    def _wake_blocker(
        self, binding: Binding, bindings: list[Binding], leases=None, journal=None
    ) -> "WakeConflict | None":
        """Why the account refuses a wake of ``binding`` right now, or None: an awake
        binding on one of its GPUs (the store), or an active GPU lease of another
        binding there - a ``starting`` Pod still loading (S2: that lease lives until
        the Pod converged or is gone), a ``waking`` binding, an ``awake`` lease the
        store does not show yet. ``leases``: a pre-loaded lease list (planning
        loops load it once)."""
        node = binding.slot.node
        gpus = tuple(binding.slot.gpu_ids)
        if not self._feasible_wake(binding, bindings):
            occupants = sorted(
                item.binding_id
                for item in bindings
                if item.awake
                and item.serve_id != binding.serve_id
                and item.slot.node == node
                and set(item.slot.gpu_ids) & set(gpus)
            )
            return WakeConflict(
                f"{binding.serve_id}: slot already has awake binding {occupants}",
                reason="slot_occupied", node=node, gpus=gpus, binding_id=binding.binding_id,
                blocking_binding_id=occupants[0] if occupants else None,
            )
        for lease in self._active_leases(leases):
            if lease.binding_id == binding.binding_id or lease.node != node:
                continue
            overlap = sorted(set(gpus) & {int(gpu) for gpu in lease.gpu_ids})
            if not overlap:
                continue
            phase = str(getattr(lease, "phase", "") or "awake")
            reason = f"lease_{phase}" if phase in ("starting", "waking") else "slot_occupied"
            return WakeConflict(
                f"{binding.serve_id}: {node}/{overlap[0]} is held by the {phase} GPU lease of "
                f"{lease.binding_id}",
                reason=reason, node=node, gpus=gpus, binding_id=binding.binding_id,
                blocking_binding_id=lease.binding_id,
            )
        # A journaled wake (in flight, or left for the recovery) occupies its GPUs
        # whatever its lease says (P1-2): the engine may be awake already.
        entries = self._wake_journal.entries() if journal is None else journal
        for other_id, entry in entries.items():
            if other_id == binding.binding_id or entry.get("node") != node:
                continue
            try:
                overlap = sorted(set(gpus) & {int(gpu) for gpu in entry.get("gpu_ids") or ()})
            except (TypeError, ValueError):
                continue
            if overlap:
                return WakeConflict(
                    f"{binding.serve_id}: {node}/{overlap[0]} has a wake of {other_id} in flight",
                    reason="lease_waking", node=node, gpus=gpus, binding_id=binding.binding_id,
                    blocking_binding_id=other_id,
                )
        return None

    def _active_leases(self, leases=None) -> list:
        """Unexpired GPU leases (``expires_at_ms`` 0 = never expires), on the Redis
        clock the Lua scripts write expiries with."""
        if leases is None:
            if self._gpu_leases is None:
                return []
            leases = self._gpu_leases.load()
        now_reader = getattr(self._gpu_leases, "now_ms", None)
        now_ms = int(now_reader()) if callable(now_reader) else int(time.time() * 1000)
        return [lease for lease in leases if not _lease_expired(lease, now_ms)]

    def _ensure_target_within_cap(
        self, model: str, spec, target: int, bindings: list[Binding]
    ) -> None:
        """The one scaling-cap rule: refuse a target that GROWS the model's awake count
        (hidden probe pods included - they are awake, v1 assigned) past
        max_awake_replicas. Shrinks and unchanged targets always pass, even above the cap."""
        cap = scale_max_replicas(spec)
        awake = sum(1 for item in bindings if item.model == model and item.awake)
        if target > awake and target > cap:
            raise ValueError(
                f"wake_replicas {target} exceeds max_awake_replicas ({cap}) for {model} (awake {awake})"
            )

    def _ensure_wake_within_cap(self, binding: Binding, bindings: list[Binding]) -> None:
        """Binding-level wakes obey the same rule as put_model_target (a wake is the
        target awake + 1)."""
        try:
            spec = self._registry.model(binding.model)
        except KeyError:
            return
        awake = sum(1 for item in bindings if item.model == binding.model and item.awake)
        self._ensure_target_within_cap(binding.model, spec, awake + 1, bindings)

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

    def _ensure_create_headroom(self, slot: Slot, model: str) -> None:
        # Fail closed: a cold start writes model weights onto a GPU we believe
        # is free. When the gpu-truth DaemonSet is down its Redis key expires,
        # and absent truth used to silently skip this gate -- the one guard
        # standing between a stale view and two models on one GPU. Refuse
        # instead. Set TRE_GPU_TRUTH_REQUIRED=false to restore the old
        # permissive behaviour if truth is unavailable during an emergency.
        # The gate waits for a fresh gpu-truth sample (_gpu_truth_gate): a
        # defrag / scale path deletes a pod right before it and the periodic
        # sample would still show that pod's memory.
        # The limit (B9): vLLM starts only with gpu_memory_utilization x total free,
        # so used may reach total x (1 - util of ``model``) - margin; an absolute
        # override (env TRE_CREATE_MAX_USED_MIB > service_manager.create.max_used_mib)
        # replaces it.
        if self._gpu_truth is None:
            return
        util = self._create_gpu_memory_utilization(model)
        nodes = {node.name: node for node in self._registry.topology().nodes}
        node = nodes.get(slot.node)
        problem = self._gpu_truth_gate(
            slot.node,
            lambda node_truth: self._create_headroom_problem(slot, node, node_truth, util),
            retry_stale=False,
            what=f"cold start on {slot.node}/{','.join(str(g) for g in slot.gpu_ids)}",
        )
        if problem is not None:
            raise ValueError(problem)

    def _create_gpu_memory_utilization(self, model: str) -> float | None:
        """The ``--gpu-memory-utilization`` of ``model`` (None = an absolute limit is
        configured, the utilization is not needed)."""
        if self._create_max_used_mib is not None or self._sm_config.create_max_used_mib is not None:
            return None
        try:
            return gpu_memory_utilization(self._registry.model(model))
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"cold start of {model}: cannot derive the startup headroom limit: {exc}"
            ) from exc

    def _create_limit_mib(self, total_mib: int | None, util: float | None) -> tuple[int | None, str]:
        """(limit, source) of the cold-start gate: env > registry absolute > derived."""
        if self._create_max_used_mib is not None:
            return int(self._create_max_used_mib), "TRE_CREATE_MAX_USED_MIB"
        config = self._sm_config
        if config.create_max_used_mib is not None:
            return int(config.create_max_used_mib), "service_manager.create.max_used_mib"
        limit = config.create_limit_mib(total_mib, float(util))
        return limit, (
            f"total_mib={total_mib} x (1 - gpu_memory_utilization {float(util):g}) "
            f"- margin {config.create_margin_mib}"
        )

    def _create_headroom_problem(self, slot: Slot, node, node_truth, util: float | None) -> str | None:
        if node_truth is None:
            if not self._require_gpu_truth:
                return None
            return (
                f"gpu truth unavailable for node {slot.node}: refusing cold start "
                "(is the tre-v2-gpu-truth DaemonSet healthy?)"
            )
        for gpu_id in slot.gpu_ids:
            gpu_uuid = _gpu_uuid(node, gpu_id)
            if gpu_uuid is None:
                continue
            used_mib = node_truth.used_mib(gpu_uuid)
            if used_mib is None:
                if not self._require_gpu_truth:
                    continue
                return (
                    f"gpu truth unavailable for {slot.node}/{gpu_uuid}: refusing cold start "
                    "(gpu missing from the node truth payload)"
                )
            total = getattr(node_truth, "total_mib", None)
            limit, source = self._create_limit_mib(
                total(gpu_uuid) if callable(total) else None, util
            )
            if limit is None:
                if not self._require_gpu_truth:
                    continue
                return (
                    f"gpu truth for {slot.node}/{gpu_uuid} reports no total memory: refusing "
                    "cold start (set service_manager.create.max_used_mib to use an absolute "
                    "threshold)"
                )
            if used_mib > limit:
                return (
                    "insufficient startup headroom: "
                    f"{slot.node}/{gpu_uuid} used_mib={used_mib} max_used_mib={limit} ({source})"
                )
        return None

    def _snapshot_for_binding(self, binding: Binding) -> K8sPodSnapshot:
        snapshots = self._runtime_ops.list_pod_snapshots(model=binding.model) if self._runtime_ops else []
        for snapshot in snapshots:
            if snapshot.name == binding.serve_id:
                return snapshot
        raise ValueError(f"pod {binding.serve_id} not found for runtime operation")

    def _model_counts(
        self, bindings: list[Binding], *, routable: dict[str, set[str]] | None | bool = False
    ) -> dict[str, dict[str, int]]:
        """``{model: {awake, bound}}``; with ``routable`` (``_routable_by_model``,
        ``GET /v2/state``) also ``routable``, ``floor`` (the enforced
        min_replicas, 0 when the floor is off) and ``floor_headroom = routable -
        floor`` (None when the routable view could not be read)."""
        counts = {model.name: {"awake": 0, "bound": 0} for model in self._registry.models()}
        for binding in bindings:
            bucket = counts.setdefault(binding.model, {"awake": 0, "bound": 0})
            bucket["bound"] += 1
            if binding.awake:
                bucket["awake"] += 1
        if routable is not False:
            enforced = self._floor_enforced()
            for model, bucket in counts.items():
                floor = self._model_floor(model) if enforced else 0
                count = None if routable is None else len(routable.get(model, ()))
                bucket["routable"] = count
                bucket["floor"] = floor
                bucket["floor_headroom"] = None if count is None else count - floor
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


#: Startup admission jobs (review 3 P2-5): parallel admissions (they serialize
#: on the writer lock; only their resident drains overlap), how long the first
#: gate call waits for the result before answering 202 (well under the gate's
#: 15 s HTTP timeout), and how long an uncollected result is kept.
ADMISSION_WORKERS = 4
ADMISSION_SYNC_WAIT_S = 5.0
ADMISSION_RESULT_TTL_S = 600.0

#: Waiting reasons of the engine container under which a startup placeholder may
#: be released (the engine is not running and holds no GPU memory). Anything else
#: - PodInitializing, ContainerCreating, an unknown reason - keeps it.
PLACEHOLDER_RELEASE_REASONS = (
    "CrashLoopBackOff", "Error", "ImagePullBackOff", "ErrImagePull",
    "CreateContainerError", "CreateContainerConfigError",
)

#: Wakes of one request run their /wake_up + /is_sleeping concurrently on this
#: many threads (S6); different GPUs only - one binding per GPU is ever waking.
WAKE_WORKERS = 8

#: Operation phase of a writer that creates a gated Pod itself (fleet repair,
#: defrag, cold start): the Pod of ``details.binding_id`` is admitted without
#: the writer lock (review 4 P1).
STARTING_BINDING_PHASE = "starting_binding"


def restart_placeholder_candidates(
    snapshots, store_bindings, conflicts: list[Binding] | None = None
) -> list[Binding]:
    """Bootstrap re-derivation (review, 2026-09-30): pods whose engine container
    runs but is not Ready, that were not admitted at their startup gate (those get
    their starting lease from the admission) and that the store does not record
    awake - an engine reloading after an in-place restart the previous SM may have
    been converging. They get a ``starting`` placeholder and are suspects until the
    restart guard converges them; nothing of this lives only in memory. A
    candidate on GPUs another binding is recorded awake on gets no placeholder; it
    is appended to ``conflicts`` (when given) so the SM keeps it a suspect."""
    awake = {binding.binding_id for binding in store_bindings if binding.awake}
    out: list[Binding] = []
    for snapshot in snapshots:
        if getattr(snapshot, "engine_running", None) is not True or snapshot.ready:
            continue
        admitted = snapshot.annotations.get("tre.aibrix.io/startup-admitted-uid")
        if admitted and admitted == snapshot.pod_uid:
            continue
        try:
            binding = _binding_from_snapshot(snapshot)
        except (KeyError, ValueError):
            continue
        if binding.binding_id in awake:
            continue
        clash = sorted(
            other.binding_id
            for other in store_bindings
            if other.awake and other.slot.node == binding.slot.node
            and set(other.slot.gpu_ids) & set(binding.slot.gpu_ids)
        )
        if clash:
            # Another binding is recorded awake there: a placeholder would clash
            # with its lease at bootstrap. Reported (possible double occupancy);
            # the restart guard / reconcile / audit see it.
            _log_event(
                "container_restart_conflict", level=logging.ERROR,
                binding_id=binding.binding_id, pod=snapshot.name, occupants=clash,
                detail="engine reloading on GPUs another binding holds (at SM bootstrap); binding kept a suspect",
            )
            if conflicts is not None:
                conflicts.append(replace(binding, awake=False, hidden=True))
            continue
        out.append(replace(binding, awake=False, hidden=True))
    return out


def _elapsed_ms(started: float) -> int:
    return int(round((time.monotonic() - started) * 1000))


#: Wake journal recovery results that keep the entry for the next pass.
_WAKE_RECOVERY_KEPT = frozenset({"physical_state_unknown", "wake_unsettled", "pod_still_present"})


def _unanswered(result) -> bool:
    """A vLLM call that got no HTTP answer (timeout / connection error:
    ``VllmOpResult.status_code`` None) - the server may still act on it.
    Results without a ``status_code`` attribute (test doubles) never count."""
    return (
        not bool(getattr(result, "success", False))
        and hasattr(result, "status_code")
        and result.status_code is None
    )


def _log_event(event: str, *, level: int = logging.INFO, **fields) -> None:
    """One structured (JSON) log line ``{"event": ..., ...}``."""
    LOG.log(level, json.dumps({"event": event, **fields}, sort_keys=True, default=str))


def _wake_action(ticket: "_WakeTicket") -> dict:
    """``actions`` entry of a committed wake (the shape callers always got)."""
    return {"action": "wake", "serve_id": ticket.binding.serve_id}


def _picked(ticket: "_WakeTicket") -> dict:
    """``picked`` entry of a committed wake: which binding woke and where (S5)."""
    return {
        "serve_id": ticket.binding.serve_id,
        "binding_id": ticket.binding.binding_id,
        "node": ticket.binding.slot.node,
        "gpu_ids": list(ticket.binding.slot.gpu_ids),
        "hinted": bool(ticket.hinted),
        # The hint this pick stands in for (None: no hint was open).
        "hint_binding_id": (ticket.placement or {}).get("hint_binding_id"),
    }


def _observed_record(observation) -> ObservedBinding:
    """The observed-fleet record of one pod observation (reconcile / refresh)."""
    binding = observation.binding
    pod = observation.pod
    physical_power = (
        "unknown"
        if observation.physical_awake is None
        else "awake" if observation.physical_awake else "sleeping"
    )
    return ObservedBinding(
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
        error="physical_probe_unreachable" if observation.physical_awake is None else None,
    )


def _binding_from_outcome(outcome: dict, pod: StartupPodRecord) -> Binding:
    """The binding of a sleep outcome (the admission's residents are on the
    startup Pod's node)."""
    binding_id = str(outcome.get("binding_id") or "")
    model, _node, gpus = (binding_id.split("/") + ["", "", ""])[:3]
    gpu_ids = tuple(int(gpu) for gpu in gpus.split(",") if gpu.strip())
    return Binding(str(outcome.get("serve_id")), model, Slot(pod.node, gpu_ids), awake=False)


class _DesiredGuard:
    """Handle of :meth:`ServiceManagerV2._desired_guard`: bindings whose physical
    change is done are ``settle``-d and never rolled back."""

    def __init__(self) -> None:
        self.settled: set[str] = set()
        self.all_settled = False

    def settle(self, binding_ids=None) -> None:
        if binding_ids is None:
            self.all_settled = True
        else:
            self.settled.update(binding_ids)


class DefragUnavailable(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class DefragDisabled(DefragUnavailable):
    """Registry ``placement.defrag.enabled`` is false and the request did not force."""

    MESSAGE = (
        "defrag is disabled (registry placement.defrag.enabled: false); "
        "send force: true to run it anyway"
    )

    def __init__(self) -> None:
        super().__init__("defrag_disabled")


class WakeConflict(ValueError):
    """A wake refused on its GPUs (HTTP 409, structured body - S3).

    ``reason`` is the precise cause, ``error`` its stable code (verification
    plan 20261001 section 7-3):

    ====================  =====================  ================================
    reason                error                  meaning
    ====================  =====================  ================================
    slot_occupied         gpu_busy               an awake binding holds the GPU
    gpu_truth_used        gpu_busy               trusted gpu-truth: GPU in use
    fault_injected        gpu_busy               test hook (service_manager.test_hooks)
    lease_starting        resident_loading       a Pod admitted there is loading
    resident_unknown      resident_loading       a resident's state cannot be read
    lease_waking          lease_conflict         another binding is being woken there
    wake_in_progress      lease_conflict         this binding is being woken already
    resident_awake        resident_awake         the resident probe found one awake
    gpu_truth_unavailable truth_unavailable      no gpu-truth for the node and a
                                                 resident unverifiable (scope node)
    ====================  =====================  ================================

    ``node`` / ``gpus`` locate the refusal (``scope`` gpu or node), so a caller can
    avoid those GPUs (or the node) for ``retry_after_s`` instead of retrying the
    same slot; ``blocking_binding_id`` names the occupant when known."""

    ERROR_CODES = {
        "slot_occupied": "gpu_busy",
        "gpu_truth_used": "gpu_busy",
        "fault_injected": "gpu_busy",
        "lease_starting": "resident_loading",
        "resident_unknown": "resident_loading",
        "lease_waking": "lease_conflict",
        "wake_in_progress": "lease_conflict",
        "resident_awake": "resident_awake",
        "gpu_truth_unavailable": "truth_unavailable",
        "partial": "partial",
        "routable_unknown": "routable_unknown",
    }

    def __init__(
        self,
        message: str,
        *,
        reason: str = "slot_occupied",
        node: str | None = None,
        gpus=(),
        scope: str = "gpu",
        binding_id: str | None = None,
        blocking_binding_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.node = node
        self.gpus = tuple(int(gpu) for gpu in gpus)
        self.scope = scope
        self.binding_id = binding_id
        self.blocking_binding_id = blocking_binding_id

    @property
    def error(self) -> str:
        return self.ERROR_CODES.get(self.reason, "gpu_busy")

    def body(self, *, retry_after_s: float | None = None) -> dict:
        """The structured 409 body; ``detail`` keeps the plain message older
        controllers read (they treat any 409 as retriable)."""
        return {
            "detail": str(self),
            "error": self.error,
            "reason": self.reason,
            "binding_id": self.binding_id,
            "node": self.node,
            "gpu_ids": list(self.gpus),
            "gpu": list(self.gpus),
            "scope": self.scope,
            "blocking_binding_id": self.blocking_binding_id,
            "retry_after_s": retry_after_s,
        }


class WakeFailed(WakeConflict):
    """The wake itself failed (``/wake_up`` refused, the engine did not converge,
    or it could not be recorded) - HTTP 409 ``error: wake_failed``. The GPU is
    suspect for a while (a caller cools it down like a refusal).
    ``physically_awake`` / ``compensating_sleep``: what the settlement found and
    did (S4)."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("reason", "vllm_wake_failed")
        super().__init__(message, **kwargs)
        self.physically_awake: bool | None = None
        self.compensating_sleep: dict | None = None

    @property
    def error(self) -> str:
        return "wake_failed"

    def body(self, *, retry_after_s: float | None = None) -> dict:
        body = super().body(retry_after_s=retry_after_s)
        body["physically_awake"] = self.physically_awake
        body["compensating_sleep"] = self.compensating_sleep
        return body


class TransferFailed(WakeConflict):
    """No pair of a transfer completed (HTTP 409 ``error: partial``, the
    structured WakeConflict body plus the transfer response: ``pairs``,
    ``refusals``, ``unfilled``, ...). ``node`` / ``gpu_ids`` /
    ``blocking_binding_id`` are those of the first failure."""

    def __init__(self, message: str, *, response: dict, first: dict | None = None) -> None:
        first = first or {}
        super().__init__(
            message,
            reason="partial",
            node=first.get("node"),
            gpus=first.get("gpu_ids") or (),
            scope=first.get("scope") or "gpu",
            binding_id=first.get("binding_id"),
            blocking_binding_id=first.get("blocking_binding_id"),
        )
        self.response = dict(response)

    def body(self, *, retry_after_s: float | None = None) -> dict:
        return {**self.response, **super().body(retry_after_s=retry_after_s)}


def _error_code(exc: BaseException | None) -> str | None:
    """Stable error code of a failed wake (ops details / logs)."""
    if exc is None:
        return None
    if isinstance(exc, WakeConflict):
        return exc.error
    if isinstance(exc, GpuLeaseConflict):
        return "lease_conflict"
    if isinstance(exc, OperationBusy):
        return "writer_busy"
    if isinstance(exc, FloorViolation):
        return "floor_violation"
    return type(exc).__name__


@dataclass
class _WakeTicket:
    """One wake between its phases (S6): prepared under the writer lock, run
    without it, committed under it again."""

    binding: Binding
    pod_ip: str
    #: (power, hidden) of the desired record before the wake's intent was written
    #: (restored if the wake fails); None = not the wake's to restore.
    previous_desired: tuple[str, bool] | None = None
    woke: bool = False
    exception: BaseException | None = None
    #: /is_sleeping when a failed wake was settled (None = unknown / not probed).
    physical: bool | None = None
    #: The caller named this binding (S5 hint) - reported in the response.
    hinted: bool = False
    #: Woken and recorded, but the legacy store save failed (raised afterwards).
    store_error: BaseException | None = None
    #: S5: {"hint_binding_id", "chosen_binding_id", "source": "hint" | "sm_choose"}.
    placement: dict | None = None
    #: How the wake gate decided (S1): gpu_truth | is_sleeping_probe | none.
    truth_source: str = "none"
    truth_age_s: float | None = None
    #: Phase durations (ms): reserve (phase 1), wake_up (phase 2), commit (phase 3).
    phases_ms: dict = field(default_factory=dict)
    #: Wakes in flight in this process when phase 2 started (incl. this one).
    parallel_inflight: int = 1
    #: S4: {"done": bool, "result": str} of the sleep sent after a failed wake that
    #: had woken the engine anyway; None = not needed.
    compensating_sleep: dict | None = None
    #: The commit of this ticket raised (it stays journaled for the recovery).
    commit_error: BaseException | None = None
    #: Failed with an unknown physical state: journal entry + waking lease kept.
    left_to_recovery: bool = False
    #: /wake_up raised (transport error / timeout): the server may still wake it.
    uncertain: bool = False
    #: Settling the lease of a failed wake raised: journal entry kept.
    lease_unsettled: bool = False


@dataclass
class _TransferPairState:
    """One pair of a transfer."""

    pair: TransferPair
    status: str = TRANSFER_PENDING
    error: BaseException | None = None
    #: The receiver's wake.
    ticket: "_WakeTicket | None" = None


@dataclass
class _Transfer:
    """One ``POST /v2/transfers``."""

    transfer_id: str
    donor_model: str
    receiver_model: str
    count: int
    sleep_path: str
    selection: TransferSelection | None = None
    pairs: list[_TransferPairState] = field(default_factory=list)
    #: Phase durations (ms): select, donor_sleep, receiver_wake, total.
    phases_ms: dict = field(default_factory=dict)
    #: Donor binding ids recorded asleep.
    donors_slept: set = field(default_factory=set)
    version: int | None = None


def _transfer_pair_brief(pair: TransferPair) -> dict:
    """The identity of a pair in the response / events / operation record."""
    return {
        "donor": pair.donors[0].serve_id,
        "donors": [donor.serve_id for donor in pair.donors],
        "donor_binding_ids": [donor.binding_id for donor in pair.donors],
        "receiver": pair.receiver.serve_id,
        "receiver_binding_id": pair.receiver.binding_id,
        "node": pair.node,
        "gpu_ids": list(pair.gpu_ids),
    }


@dataclass(frozen=True)
class _PowerMark:
    """What a gpu-truth sample must answer to be trusted after the last local power
    change on one GPU (S1): ``refresh_seq`` = the refresh request sent right after
    the change, ``seq`` = the agent's publish counter seen at the change (for an
    agent that does not serve refreshes). None = unknown."""

    refresh_seq: int | None
    seq: int | None

    @staticmethod
    def merged(previous: "_PowerMark | None", refresh_seq: int | None, seq: int | None) -> "_PowerMark":
        if previous is None:
            return _PowerMark(refresh_seq, seq)

        def newest(old, new):
            if old is None:
                return new
            if new is None:
                return old
            return max(old, new)

        return _PowerMark(newest(previous.refresh_seq, refresh_seq), newest(previous.seq, seq))


class RetryLater(RuntimeError):
    """The request cannot proceed right now but may succeed when retried (HTTP
    409), e.g. a resident on a startup Pod's GPUs woke up again."""


class TargetRequest(BaseModel):
    #: Absolute target (I2: a replay is harmless).
    wake_replicas: int
    #: S5: serve ids of sleeping bindings to prefer (placement hints; the SM may
    #: substitute a hint it cannot wake - see ``picked`` in the response).
    hints: list[str] = []
    #: GPUs (``node/gpu``) not to wake on (relays of the caller in flight there).
    avoid_gpus: list[str] = []
    #: Grow-only: no-op when the model already has >= wake_replicas awake (review 3).
    at_least: bool = False
    #: Which sleep path this is (recorded with the outcome; floor rules).
    sleep_path: str = "scale_down"
    #: Deprecated and ignored (2026-10-02): the service-manager never drains.
    drain_budget_s: float | None = None


class TransferRequest(BaseModel):
    donor_model: str
    receiver_model: str
    #: Donor replicas to hand over (a TP=2 receiver taking two single-GPU donors
    #: counts 2).
    count: int = 1
    #: Sleep path of the donors (external paths only; safescale_commit -> 400).
    sleep_path: str = "urgent"
    #: Serve ids or binding ids the donors must be among (None = any).
    donor_bindings: list[str] | None = None
    #: GPUs (``node/gpu``) no receiver may use.
    avoid_gpus: list[str] = []


class BindingPowerRequest(BaseModel):
    awake: bool
    sleep_path: str = "scale_down"
    drain_budget_s: float | None = None


class DefragRequest(BaseModel):
    tp_size: int
    #: Run even while registry placement.defrag.enabled is false (operator override).
    force: bool = False




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
            content={
                "detail": str(exc),
                "error": "writer_busy",
                "current_writer": exc.current_writer,
                "retry_after_s": 1.0,
            },
        )

    @app.exception_handler(WakeConflict)
    async def wake_conflict_handler(_request: Request, exc: WakeConflict) -> JSONResponse:
        # S3: structured 409 (error code, binding, node, GPUs, blocking binding,
        # retry_after_s); ``detail`` keeps the plain message older callers read.
        return JSONResponse(
            status_code=409, content=exc.body(retry_after_s=service.wake_retry_after_s(exc.scope))
        )

    @app.exception_handler(StateFenceError)
    async def state_fence_handler(
        _request: Request, exc: StateFenceError
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(MaintenanceLockLost)
    async def maintenance_lock_lost_handler(
        _request: Request, exc: MaintenanceLockLost
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.middleware("http")
    async def record_actor(request: Request, call_next):
        # Who called a write API (controller, APA arm, console, operator): kept
        # in the request of every SM operation it starts (2026-09-28). The
        # X-TRE-Actor header when sent, else User-Agent + remote address.
        actor = request.headers.get("x-tre-actor")
        if not actor:
            client = request.client.host if request.client is not None else "?"
            actor = f"ua={request.headers.get('user-agent', '?')};addr={client}"
        token = set_current_actor(actor[:200])
        try:
            return await call_next(request)
        finally:
            reset_current_actor(token)

    @app.exception_handler(NodePressureActive)
    async def node_pressure_handler(
        _request: Request, exc: NodePressureActive
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(SleepFailed)
    async def sleep_failed_handler(_request: Request, exc: SleepFailed) -> JSONResponse:
        # The primitive already rolled back (or kept hidden) every target; the
        # outcomes say which pods slept.
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "error": type(exc).__name__, "outcomes": exc.outcomes},
        )

    @app.exception_handler(FloorViolation)
    async def floor_violation_handler(_request: Request, exc: FloorViolation) -> JSONResponse:
        # 409 with ``error: floor_violation``: the controller does not retry it
        # (it re-plans on its next tick).
        return JSONResponse(status_code=409, content=exc.body())

    @app.exception_handler(RetryLater)
    async def retry_later_handler(_request: Request, exc: RetryLater) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ServiceShuttingDown)
    async def shutting_down_handler(
        _request: Request, exc: ServiceShuttingDown
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(GpuLeaseConflict)
    async def gpu_lease_conflict_handler(
        _request: Request, exc: GpuLeaseConflict
    ) -> JSONResponse:
        node, _, gpu = str(exc.gpu).rpartition("/")
        gpu_ids = [int(gpu)] if gpu.isdigit() else []
        return JSONResponse(
            status_code=409,
            content={
                "detail": str(exc),
                "error": "lease_conflict",
                "reason": "lease_conflict",
                "binding_id": None,
                "node": node or None,
                "gpu_ids": gpu_ids,
                "gpu": gpu_ids,
                "scope": "gpu",
                "blocking_binding_id": exc.occupant,
                "retry_after_s": service.wake_retry_after_s("gpu"),
            },
        )

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

    @app.post("/v2/fleet/seed")
    def seed_desired() -> dict:
        try:
            return service.seed_desired()
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v2/startup/admit")
    def admit_startup(request: StartupAdmissionRequest):
        # 200 = admitted; 202 = admission in progress (the init gate polls again);
        # 409 / 503 / 400 = refused this time (review 3 P2-5).
        try:
            status, body = service.request_startup_admission(
                pod_name=request.pod_name, pod_uid=request.pod_uid
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if status == 200:
            return body
        return JSONResponse(status_code=status, content=body)

    @app.post("/v2/startup/converge")
    def converge_startups() -> dict:
        return service.converge_startups()

    @app.get("/v2/state")
    def get_state() -> dict:
        # The only caller of the routable view (one Pod LIST per call); the v1
        # API and internal callers use the light get_state().
        return service.get_state(routable=True)


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

    @app.get("/v2/sleep")
    def get_sleep_state() -> dict:
        return service.sleep_state()

    @app.get("/v2/wake")
    def get_wake_state() -> dict:
        return service.wake_state()

    @app.get("/v2/supervisor")
    def get_supervisor() -> dict:
        return service.get_supervisor_state()


    @app.post("/v2/defrag")
    def defrag(request: DefragRequest) -> dict:
        try:
            return service.defrag(tp_size=request.tp_size, force=request.force)
        except DefragDisabled as exc:
            raise HTTPException(
                status_code=409, detail={"reason": exc.reason, "message": exc.MESSAGE}
            ) from exc
        except DefragUnavailable as exc:
            raise HTTPException(status_code=409, detail={"reason": exc.reason}) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @app.put("/v2/models/{model}/routable")
    def put_model_routable(model: str, request: RoutableRequest) -> dict:
        try:
            return service.put_model_routable(model, hidden_pods=request.hidden_pods)
        except WakeConflict:
            raise  # structured 409 (wake_conflict_handler)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/v2/models/{model}/target")
    def put_model_target(model: str, request: TargetRequest) -> dict:
        try:
            return service.put_model_target(
                model,
                wake_replicas=request.wake_replicas,
                sleep_path=_sleep_path(request.sleep_path),
                drain_budget_s=request.drain_budget_s,
                at_least=request.at_least,
                hints=request.hints,
                avoid_gpus=request.avoid_gpus,
            )
        except WakeConflict:
            raise  # structured 409 (wake_conflict_handler)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v2/transfers")
    def post_transfer(request: TransferRequest) -> dict:
        # 200: at least one pair done (``pairs[].status``), or the donor's floor
        # held the whole transfer back (``clamped_by_floor``); 409 ``partial``:
        # no pair completed; 400: bad parameters.
        try:
            return service.transfer(
                donor_model=request.donor_model,
                receiver_model=request.receiver_model,
                count=request.count,
                sleep_path=_sleep_path(request.sleep_path),
                donor_bindings=request.donor_bindings,
                avoid_gpus=request.avoid_gpus,
            )
        except WakeConflict:
            raise  # structured 409 (wake_conflict_handler)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/v2/bindings/{serve_id}/power")
    def put_binding_power(serve_id: str, request: BindingPowerRequest) -> dict:
        try:
            return service.put_binding_power(
                serve_id,
                awake=request.awake,
                sleep_path=_sleep_path(request.sleep_path),
                drain_budget_s=request.drain_budget_s,
            )
        except WakeConflict:
            raise  # structured 409 (wake_conflict_handler)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app

#: Sleep paths an HTTP caller may name (review 2026-09-29 P3-8). ``repair``,
#: ``startup`` and ``defrag`` are the SM's own paths with their own floor rules
#: (exempt / make-up); ``default`` is for internal callers that name none.
EXTERNAL_SLEEP_PATHS = ("scale_down", "urgent", "safescale_commit", "apa")


def _sleep_path(value: str) -> str:
    if value not in SLEEP_PATHS:
        raise ValueError(f"unknown sleep_path {value!r} (known: {', '.join(SLEEP_PATHS)})")
    if value not in EXTERNAL_SLEEP_PATHS:
        raise ValueError(
            f"sleep_path {value!r} is internal to the service-manager "
            f"(allowed: {', '.join(EXTERNAL_SLEEP_PATHS)})"
        )
    return value


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


def _defrag_destination(binding: Binding, migration: Migration, bindings) -> Binding | None:
    """The existing binding of ``binding``'s model on the migration's destination
    slot (full static layout, B6), or None (sparse layout: create one there)."""
    destination_id = replace(binding, slot=migration.to_slot).binding_id
    for item in bindings:
        if item.binding_id == destination_id and item.serve_id != binding.serve_id:
            return item
    return None


def _registry_placement_policy(registry) -> PlacementPolicy | None:
    """The registry placement policy; None only for registry stubs without
    models (embedded / unit-test wiring). An unsupported model tp_size raises
    (ValueError) instead of silently falling back to plain best-fit."""
    models = getattr(registry, "models", None)
    if not callable(models):
        return None
    return placement_policy_from_registry(registry)


def _wake_pick(feasible, planned, topology, policy: PlacementPolicy | None = None) -> Binding:
    """Which sleeping binding to wake: the registry placement policy, same rule as
    the controller planner (tre_common.gpu_placement): keep a free aligned pair for
    a tp=2 model, then balance node load, spread the model across nodes, best fit.
    """
    nodes = node_gpu_counts(topology)
    planned = list(planned)
    scorable = [
        binding
        for binding in feasible
        if is_buddy_aligned(binding.slot, nodes)
        and len(binding.slot.gpu_ids) == len(feasible[0].slot.gpu_ids)
    ]
    if scorable:
        choice = choose_placement(
            [slot_block(binding.slot) for binding in scorable],
            nodes=nodes,
            occupied=awake_gpus(planned),
            tp_size=len(scorable[0].slot.gpu_ids),
            policy=policy.for_awake(awake_model_counts(planned)) if policy else None,
            model_occupied=model_awake_gpus(planned, scorable[0].model),
        )
        if choice is not None:
            return scorable[choice.index]
    return min(feasible, key=lambda item: _natural_key(item.serve_id))


#: Replica floor (2026-09-29): (model, binding id) pairs a floor check counts as
#: routable although the store does not record them awake yet (the destination of a
#: make-before-break move, set by ServiceManagerV2._floor_counting).
_FLOOR_EXTRA_ROUTABLE: contextvars.ContextVar[frozenset] = contextvars.ContextVar(
    "tre_sm_floor_extra_routable", default=frozenset()
)
#: Per-model notes a startup's make-up phase leaves for the exemption record of the
#: sleep's floor check (why no make-up happened). Never mutated in place.
_FLOOR_EVENT_NOTES: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "tre_sm_floor_event_notes", default={}
)
#: Desired-state reason prefix / writer of a startup's floor make-up wake; the reason
#: is ``replica_floor_makeup:<suspended binding id>:<startup Pod UID>``.
FLOOR_MAKEUP_REASON = "replica_floor_makeup"
FLOOR_UPDATED_BY = "service-manager-floor"


def _mark_defrag_source_slept(exc: BaseException) -> None:
    try:
        exc.defrag_source_slept = True
    except Exception:  # noqa: BLE001 - an exception type without a __dict__
        pass


def _lease_expired(lease, now_ms: int) -> bool:
    expires = int(getattr(lease, "expires_at_ms", 0) or 0)
    return expires != 0 and expires <= now_ms


def _slots_overlap(first: Slot, second: Slot) -> bool:
    return first.node == second.node and bool(set(first.gpu_ids) & set(second.gpu_ids))


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


def _current_operation_id() -> str | None:
    operation = current_operation()
    return None if operation is None else operation.operation_id


def _default_sleep_clock() -> Clock:
    from tre_sm.ops import sleep_primitive

    return sleep_primitive.DEFAULT_CLOCK


def _quantile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[index]

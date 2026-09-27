from __future__ import annotations

from contextlib import contextmanager
import logging
import re
import json
from dataclasses import replace
from dataclasses import asdict
from functools import wraps
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tre_common.registry import SLEEP_PATHS, Registry, ServiceManagerConfig, scale_max_replicas
from tre_common.registry import NodeSpec
from tre_common.gpu_placement import choose_placement
from tre_sm.allocator.slots import (
    Binding,
    Migration,
    Slot,
    SlotAllocator,
    awake_gpus,
    is_buddy_aligned,
    node_gpu_counts,
    release_order,
    slot_block,
)
from tre_sm.allocator.topology import K8sPodSnapshot
from tre_sm.gpu_truth import GpuTruthProvider
from tre_sm.ops.k8s_ops import StartupPodRecord
from tre_sm.ops.sleep_primitive import (
    STATUS_SLEPT,
    Clock,
    GatewayState,
    ReservationLost,
    ServiceShuttingDown,
    SleepFailed,
    SleepJournal,
    SleepPrimitive,
    SleepTarget,
)
from tre_sm.state.sleep_reservations import ReservationConflict, SleepReservations
from tre_sm.state.reconcile import K8sPodClient, POD_STATE_AWAKE, POD_STATE_HIDDEN, POD_STATE_SLEEPING, audit_state, reconcile_state
from tre_sm.state.operations import OperationBusy, OperationCoordinator, current_operation
from tre_sm.state.fleet_repair import FleetRepairExecutor
from tre_sm.state.fleet_seed import registry_binding_ids, seed_binding_ids, seed_desired
from tre_sm.state.fleet_store import DesiredBinding, FleetStateStore, ObservedBinding
from tre_sm.state.safety import ClusterSafetyGate, ControllerNotPaused, NodePressureActive
from tre_sm.state.gpu_leases import GpuLeaseConflict, GpuLeaseStore
from tre_sm.state.store import StateConflict, StateFenceError, StateStore
from tre_sm.api.v1_compat import create_v1_compat_router


_NAT_SPLIT = re.compile(r"(\d+)")
LOG = logging.getLogger(__name__)

#: Journal field of a request-initiated sleep: record desired power "sleeping"
#: once the pod is confirmed asleep (also by crash recovery), never before
#: (review 2 P2-3).
DESIRED_ON_SLEEP = {"desired_on_sleep": "sleeping"}


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
        create_max_used_mib: int = 2500,
        sleep_leak_used_mib: int = 8192,
        require_gpu_truth: bool = True,
        operation_coordinator: OperationCoordinator | None = None,
        safety_gate: ClusterSafetyGate | None = None,
        fleet_store: FleetStateStore | None = None,
        gpu_leases: GpuLeaseStore | None = None,
        gateway_state: GatewayState | None = None,
        sleep_journal: SleepJournal | None = None,
        sleep_clock: Clock | None = None,
        sleep_reservations: SleepReservations | None = None,
    ) -> None:
        self._registry = registry
        config = getattr(registry, "service_manager", None)
        self._sm_config: ServiceManagerConfig = (
            config() if callable(config) else ServiceManagerConfig()
        )
        self._sleep_clock = sleep_clock
        self._sleep_primitive: SleepPrimitive | None = None
        if runtime_ops is not None and vllm_ops is not None:
            self._sleep_primitive = SleepPrimitive(
                runtime_ops=runtime_ops,
                vllm_ops=vllm_ops,
                policy=self._sm_config.sleep,
                gateway=gateway_state,
                journal=sleep_journal or SleepJournal(),
                reservations=sleep_reservations or SleepReservations(),
                clock=sleep_clock,
                owner=getattr(operation_coordinator, "owner", "service-manager"),
                operation_id=_current_operation_id,
            )
        self._store = store
        self._k8s_client = k8s_client
        self._runtime_ops = runtime_ops
        self._vllm_ops = vllm_ops
        self._gpu_truth = gpu_truth
        self._create_max_used_mib = create_max_used_mib
        self._sleep_leak_used_mib = sleep_leak_used_mib
        self._require_gpu_truth = require_gpu_truth
        self._operation_coordinator = operation_coordinator
        self._safety_gate = safety_gate
        self._fleet_store = fleet_store
        self._gpu_leases = gpu_leases
        self._supervisor = None
        self._fleet_repair = None
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
            )

    def set_supervisor(self, supervisor) -> None:
        self._supervisor = supervisor

    def get_supervisor_state(self) -> dict:
        if self._supervisor is None:
            return {"running": False, "enabled": False}
        return {"enabled": True, **asdict(self._supervisor.snapshot())}

    def get_state(self) -> dict:
        snapshot = self._store.load()
        state = {
            "version": snapshot.version,
            "models": self._model_counts(snapshot.bindings),
            "bindings": [self._binding_dict(binding) for binding in snapshot.bindings],
        }
        if self._fleet_store is not None:
            state["fleet"] = self.get_fleet_state()
        return state

    def put_model_target(
        self,
        model: str,
        *,
        wake_replicas: int,
        sleep_path: str = "scale_down",
        drain_budget_s: float | None = None,
    ) -> dict:
        """Scale a model to ``wake_replicas`` awake bindings.

        A shrink runs in three lock phases (review P1-3): hide under the writer
        lock, gateway ack + drain WITHOUT it (the bindings are fenced by sleep
        reservations), then /sleep + store update under the writer lock again.
        Growth (wake / create) runs under the writer lock.

        Desired state follows the outcome (review 2 P2-3): a shrink records it in
        its commit phase (only the bindings that slept go "sleeping"); growth
        restores the previous desired records if anything fails.
        """
        with self._writer("put_model_target"):
            response, batch, targets = self._put_model_target_locked(
                model,
                wake_replicas=wake_replicas,
                sleep_path=sleep_path,
                drain_budget_s=drain_budget_s,
            )
        if batch is None:
            return response

        def retarget(_outcomes: list[dict]) -> None:
            # Every target slept: the model's desired target is what is awake now.
            self._set_model_desired_target(
                model=model,
                target_bindings=[
                    binding
                    for binding in self._store.load().bindings
                    if binding.model == model and binding.awake
                ],
                reason="model_target_request",
            )

        outcomes, version = self._finish_split_sleep(
            "put_model_target", batch, targets, desired_sleeping=True, on_commit=retarget
        )
        response["version"] = version
        response["actions"] = [
            {"action": "sleep", "serve_id": item["serve_id"]} for item in outcomes
        ]
        return response

    def _put_model_target_locked(
        self,
        model: str,
        *,
        wake_replicas: int,
        sleep_path: str,
        drain_budget_s: float | None,
    ) -> tuple[dict, object, list[SleepTarget]]:
        spec = self._registry.model(model)
        if wake_replicas < 0:
            raise ValueError("wake_replicas must be non-negative")
        # A model with a binding still draining for sleep has an in-flight target
        # change; the controller serializes per model, so this only fences races.
        self._assert_not_reserved(model=model, what=f"model target of {model}")

        snapshot = self._store.load()
        # Scaling cap (registry max_awake_replicas, v1/paper alignment A1); max_replicas
        # is the GPU layout size (bindings), not how many may be awake. Only GROWTH past
        # the cap is refused: shrinking (or holding) a model that is above the cap - e.g.
        # after the cap was lowered - must go through.
        self._ensure_target_within_cap(model, spec, wake_replicas, snapshot.bindings)
        model_bindings = [binding for binding in snapshot.bindings if binding.model == model]
        if self._runtime_ops is not None and wake_replicas > len(model_bindings) and not self._has_deployment_ops():
            raise ValueError("runtime create is not implemented for target growth beyond existing bindings")
        plan = self._plan_model_target(
            model=model,
            wake_replicas=wake_replicas,
            bindings=snapshot.bindings,
            tp_size=spec.tp_size,
        )
        response = {
            "model": model,
            "wake_replicas": wake_replicas,
            "version": snapshot.version,
            "actions": [],
        }
        if plan["sleep"] and self._sleep_primitive is not None and self._runtime_ops is not None:
            # All bindings of one scale-down are hidden together and drain
            # concurrently under one budget (the sleep primitive, plan D1/D2).
            # Desired state is written by the commit phase (review 2 P2-3).
            targets = self._sleep_targets_for(plan["sleep"])
            batch = self._sleep_primitive.prepare(
                targets,
                path=sleep_path,
                drain_budget_s=drain_budget_s,
                journal_extra=DESIRED_ON_SLEEP,
            )
            return response, batch, targets
        with self._desired_guard(reason="model_target_request"):
            self._set_model_desired_target(
                model=model,
                target_bindings=plan["target_bindings"],
                reason="model_target_request",
            )
            return self._apply_model_target_plan(model, wake_replicas, snapshot, plan, response)

    def _apply_model_target_plan(
        self, model: str, wake_replicas: int, snapshot, plan: dict, response: dict
    ) -> tuple[dict, object, list[SleepTarget]]:
        actions: list[dict] = []
        updated_by_serve = {binding.serve_id: binding for binding in snapshot.bindings}
        for binding in plan["sleep"]:  # no runtime: record the intent only
            updated_by_serve[binding.serve_id] = replace(binding, awake=False, hidden=False)
            actions.append({"action": "sleep", "serve_id": binding.serve_id})

        for binding in plan["wake"]:
            self._apply_runtime_power_action(binding, action="wake")
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
        return response, None, []

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
        while sleeping and len(target) < existing_target:
            feasible = [
                binding
                for binding in sleeping
                if self._feasible_wake(binding, list(planning.values()))
            ]
            if not feasible:
                raise WakeConflict(
                    f"{sleeping[0].serve_id}: slot already has awake binding"
                )
            binding = _wake_pick(feasible, planning.values(), topology)
            sleeping.remove(binding)
            planning[binding.serve_id] = replace(
                binding, awake=True, hidden=False
            )
            wakes.append(binding)
            target.append(planning[binding.serve_id])

        creates: list[Binding] = []
        existing_ids = set(planning)
        allocator = SlotAllocator(
            self._registry.topology(), list(planning.values())
        )
        while len(target) + len(creates) < wake_replicas:
            slot = allocator.find_slot(tp_size)
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
        """Wake or sleep one binding. A sleep runs in the three lock phases of
        :meth:`put_model_target` (drain outside the writer lock)."""
        with self._writer("put_binding_power"):
            if awake:
                # Controller-requested wake: same scaling cap as put_model_target. Fleet
                # repair (_set_binding_power_by_id_unlocked) restores recorded desired
                # state and is deliberately not capped here.
                snapshot = self._store.load()
                binding = next((item for item in snapshot.bindings if item.serve_id == serve_id), None)
                if binding is not None and not binding.awake:
                    self._ensure_wake_within_cap(binding, snapshot.bindings)
                return self._put_binding_power_unlocked(
                    serve_id, awake=True, sleep_path=sleep_path, drain_budget_s=drain_budget_s
                )
            if self._sleep_primitive is None or self._runtime_ops is None:
                return self._put_binding_power_unlocked(
                    serve_id, awake=False, sleep_path=sleep_path, drain_budget_s=drain_budget_s
                )
            snapshot = self._store.load()
            binding = next((item for item in snapshot.bindings if item.serve_id == serve_id), None)
            if binding is None:
                raise ValueError(f"unknown binding: {serve_id}")
            self._assert_not_reserved(binding=binding, gpus=False, what=f"power change of {serve_id}")
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
            # Desired power "sleeping" is recorded by the commit phase once the pod
            # is confirmed asleep (review 2 P2-3), never before.
            targets = self._sleep_targets_for([binding])
            batch = self._sleep_primitive.prepare(
                targets,
                path=sleep_path,
                drain_budget_s=drain_budget_s,
                journal_extra=DESIRED_ON_SLEEP,
            )
        _outcomes, version = self._finish_split_sleep(
            "put_binding_power", batch, targets, desired_sleeping=True
        )
        return {
            "serve_id": serve_id,
            "awake": False,
            "version": version,
            "actions": [{"action": "sleep", "serve_id": serve_id}],
            "binding": self._binding_dict(replace(binding, awake=False, hidden=False)),
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
        marked ``absent``. Needs the writer fence (like every desired write)."""
        if self._fleet_store is None:
            yield
            return
        try:
            before = {item.binding_id: item for item in self._fleet_store.load_desired().bindings}
        except Exception:
            before = None
        try:
            yield
        except BaseException:
            if before is not None:
                self._restore_desired(before, binding_ids, reason=reason)
            raise

    def _restore_desired(self, before: dict, binding_ids, *, reason: str) -> None:
        try:
            snapshot = self._fleet_store.load_desired()
            by_id = {item.binding_id: item for item in snapshot.bindings}
            ids = set(by_id) if binding_ids is None else set(binding_ids) & set(by_id)
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

    def _split_sleep(
        self,
        bindings: list[Binding],
        *,
        sleep_path: str,
        kind: str,
        desired_sleeping: bool = False,
    ) -> list[dict]:
        """Sleep ``bindings`` with the drain OUTSIDE the writer lock (prepare and
        commit take it briefly), e.g. for startup admission / convergence
        (review 2 P2-4). The desired state is not touched unless
        ``desired_sleeping``."""
        with self._writer(kind):
            for binding in bindings:
                self._assert_not_reserved(
                    binding=binding, gpus=False, what=f"{kind} sleep of {binding.serve_id}"
                )
            targets = self._sleep_targets_for(bindings)
            batch = self._sleep_primitive.prepare(
                targets,
                path=sleep_path,
                journal_extra=DESIRED_ON_SLEEP if desired_sleeping else None,
            )
        outcomes, _version = self._finish_split_sleep(
            kind, batch, targets, desired_sleeping=desired_sleeping
        )
        return outcomes

    def _finish_split_sleep(
        self,
        kind: str,
        batch,
        targets: list[SleepTarget],
        *,
        desired_sleeping: bool = False,
        on_commit=None,
    ) -> tuple[list[dict], int]:
        """Phase 2 (drain, no writer lock) and phase 3 (commit, writer lock) of a
        sleep prepared under the lock. Returns (outcomes, legacy store version).

        ``desired_sleeping``: record desired power "sleeping" for exactly the
        targets that slept (in the commit phase). ``on_commit(outcomes)`` runs
        under the commit lock when every target slept."""
        primitive = self._sleep_primitive
        try:
            primitive.drain(batch)
        except ReservationLost as exc:
            # Never re-acquired (review 2 P2-1): resolve under the writer lock.
            exc.outcomes = self._resolve_lost_split(kind, batch, targets)
            raise
        except BaseException:
            # Every target was rolled back by the primitive (routing restored);
            # record the store side (needs the writer lock; best effort).
            self._record_after_unlocked_failure(kind, targets, batch.outcomes())
            raise
        entered = False
        try:
            with self._writer(f"{kind}_commit", wait_s=self._sm_config.commit_wait_s) as operation:
                entered = True
                if operation is not None:
                    operation.advance("committing_sleep", details={"pods": [t.binding.serve_id for t in targets]})
                try:
                    outcomes = primitive.commit(batch)
                except SleepFailed as exc:
                    self._record_sleep_outcomes(
                        targets, exc.outcomes, update_store=True, desired_sleeping=desired_sleeping
                    )
                    raise
                self._record_sleep_outcomes(
                    targets, outcomes, update_store=True, desired_sleeping=desired_sleeping
                )
                if on_commit is not None:
                    on_commit(outcomes)
                return outcomes, self._store.load().version
        except OperationBusy:
            if entered:
                raise
            # The commit phase could not get the writer lock: nothing was slept,
            # roll the hide back (routing restored under a new route-gen).
            outcomes = primitive.abandon(
                batch, reason="writer lock unavailable for the sleep commit phase"
            )
            self._record_after_unlocked_failure(kind, targets, outcomes)
            raise

    def _resolve_lost_split(self, kind: str, batch, targets: list[SleepTarget]) -> list[dict]:
        """A drain lost its reservation: under the writer lock, roll back what no
        other sleep owns now; without the lock, leave it to crash recovery."""
        primitive = self._sleep_primitive
        try:
            with self._writer(f"{kind}_reservation_lost", wait_s=self._sm_config.commit_wait_s):
                outcomes = primitive.resolve_lost(batch)
                self._record_sleep_outcomes(targets, outcomes, update_store=True)
                return outcomes
        except Exception:
            primitive.release_unresolved(batch)
            LOG.exception(
                "resolving the lost sleep reservation of %s failed; crash recovery resolves it",
                [t.binding.serve_id for t in targets],
            )
            return batch.outcomes()

    def _record_after_unlocked_failure(
        self, kind: str, targets: list[SleepTarget], outcomes: list[dict]
    ) -> None:
        try:
            with self._writer(f"{kind}_rollback"):
                self._record_sleep_outcomes(targets, outcomes, update_store=True)
        except Exception:  # the audit reports the desired/observed mismatch
            LOG.exception("recording the rolled-back sleep of %s failed", [t.binding.serve_id for t in targets])

    def _reservations(self) -> SleepReservations | None:
        return None if self._sleep_primitive is None else self._sleep_primitive.reservations

    def _assert_not_reserved(
        self,
        *,
        what: str,
        binding: Binding | None = None,
        slot: Slot | None = None,
        model: str | None = None,
        gpus: bool = True,
    ) -> None:
        """Refuse an operation that conflicts with a binding draining for sleep:
        same binding, overlapping GPUs, or (``model``) any binding of the model."""
        reservations = self._reservations()
        if reservations is None:
            return
        if binding is not None:
            reservations.assert_free(
                node=binding.slot.node,
                gpu_ids=binding.slot.gpu_ids if gpus else (),
                binding_id=binding.binding_id,
                what=what,
            )
        if slot is not None:
            reservations.assert_free(node=slot.node, gpu_ids=slot.gpu_ids, what=what)
        if model is not None:
            reservations.assert_free(node="", gpu_ids=(), model=model, what=what)

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
        # A binding draining for sleep takes no other power / intent change.
        self._assert_not_reserved(binding=binding, gpus=False, what=f"power change of {serve_id}")

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
        with self._desired_guard([binding.binding_id], reason="binding_power_request"):
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
            if binding.hidden != (binding.serve_id in requested_hidden):
                self._assert_not_reserved(
                    binding=binding, gpus=False, what=f"hide/unhide of {binding.serve_id}"
                )
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
                if not should_hide and binding.awake:
                    self._ensure_feasible_wake(binding, list(updated_by_serve.values()))
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

        return {
            "model": model,
            "hidden_pods": sorted(requested_hidden),
            "version": version,
            "actions": actions,
        }


    @serialized_operation("defrag")
    def defrag(self, *, tp_size: int) -> dict:
        snapshot = self._store.load()
        allocator = SlotAllocator(self._registry.topology(), snapshot.bindings)
        migrations = allocator.plan_defrag(tp_size)
        if migrations is None:
            raise DefragUnavailable("no_feasible_defrag")
        by_serve = {binding.serve_id: binding for binding in snapshot.bindings}
        for migration in migrations:
            source = by_serve.get(migration.serve_id)
            if source is not None:
                self._assert_not_reserved(binding=source, what=f"defrag of {source.serve_id}")
            self._assert_not_reserved(slot=migration.to_slot, what="defrag destination")
        # The source sleeps with the writer lock held for the whole migration: the
        # create + readiness wait that follows holds it for minutes anyway, so a
        # lock-free drain would not shorten the hold (review 2 P2-4). Desired state
        # is restored if the defrag fails (review 2 P2-3).
        with self._desired_guard(reason="defrag"):
            return self._defrag_locked(snapshot, migrations)

    def _defrag_locked(self, snapshot, migrations) -> dict:
        self._set_defrag_desired(snapshot.bindings, migrations)
        if migrations:
            self._ensure_all_model_routes()

        actions: list[dict] = []
        updated_by_serve = {binding.serve_id: binding for binding in snapshot.bindings}
        for migration in migrations:
            binding = updated_by_serve[migration.serve_id]
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

        version = snapshot.version
        if migrations:
            updated = [updated_by_serve[serve_id] for serve_id in sorted(updated_by_serve)]
            version = self._store.save(updated, expected_version=snapshot.version)
        return {
            "version": version,
            "migrations": [_migration_dict(migration) for migration in migrations],
            "actions": actions,
        }


    def audit(self) -> dict:
        if self._k8s_client is None:
            raise ValueError("k8s_client is required for audit")
        prober = None
        if self._vllm_ops is not None and hasattr(self._vllm_ops, "is_sleeping"):
            prober = _VllmPodProber(self._vllm_ops)
        result = audit_state(self._store, self._k8s_client, prober=prober)
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
        prober = None
        if self._vllm_ops is not None and hasattr(self._vllm_ops, "is_sleeping"):
            prober = _VllmPodProber(self._vllm_ops)
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
        )
        self._sync_observed(result.observations)
        return {
            "version": result.version,
            "warnings": result.warnings,
            "bindings": [self._binding_dict(binding) for binding in result.bindings],
        }

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
        * ``sleep_operation_orphaned``: a sleep journal entry whose owner is gone
          (no live sleep reservation and not the live writer operation).
        * ``hidden_without_operation``: a TRE pod in the ``hidden`` state that no
          sleep in progress, no SafeScale probe (desired hidden) and no startup
          admission accounts for - hidden and then abandoned. A hidden flag in the
          legacy store alone does not justify it (it is only a cache).

        One consistent snapshot: the journal is read before and after the
        reservations / active operation, and only entries present in both reads
        are judged (a sleep finishing between the reads is not an orphan).
        """
        if self._sleep_primitive is None or self._runtime_ops is None:
            return []
        primitive = self._sleep_primitive
        first = primitive.journal.entries()
        live_tokens = {item.token for item in primitive.reservations.active().values()}
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
            and first[pod_name].get("reservation_token") == record.get("reservation_token")
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
            if record.get("reservation_token") in live_tokens:
                continue  # a drain in progress (outside the writer lock)
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
        reserved_serve_ids = {
            item.serve_id for item in primitive.reservations.active().values()
        }
        for snapshot in self._runtime_ops.list_pod_snapshots():
            if snapshot.annotations.get("tre.aibrix.io/state") != POD_STATE_HIDDEN:
                continue
            if (
                snapshot.name in journal
                or snapshot.name in first
                or snapshot.name in legacy_hidden
                or snapshot.name in reserved_serve_ids
            ):
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

        Runs at SM start and on every supervisor pass (under the writer lock). An
        entry whose sleep reservation is still live belongs to a drain in progress
        (possibly another SM replica) and is left alone; the reservation expires
        by itself when its owner died. For every other entry the physical state
        decides:

        * pod gone -> nothing to undo, entry removed;
        * asleep -> ``sleeping`` annotation, GPU lease released, store updated;
        * awake -> routing restored per the entry's ``previous_state`` under a new
          route-gen (a SafeScale probe pod stays hidden), desired power restored;
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
        live_tokens = {item.token for item in primitive.reservations.active().values()}
        snapshots = {item.name: item for item in self._runtime_ops.list_pod_snapshots()}
        resolved: list[dict] = []
        kept: list[dict] = []
        for pod_name, record in sorted(primitive.journal.entries().items()):
            if record.get("reservation_token") in live_tokens:
                continue
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
                primitive.journal.end(pod_name)
                primitive.journal.incr("recovery_slept_total")
                resolved.append({"serve_id": pod_name, "result": "slept"})
            elif physical is False and not self._awake_long_enough(pod_name, record):
                # /sleep may still be running (mode=wait waits for stragglers):
                # one awake read is not enough to re-open routing (review 2 P3).
                kept.append({"serve_id": pod_name, "result": "awake_but_sleep_may_be_running"})
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
        return {"resolved": resolved, "kept": kept}

    def _awake_long_enough(self, pod_name: str, record: dict) -> bool:
        """True when rolling back an awake pod is safe: /sleep was never sent
        (phase before "sleeping"), or the pod has read awake twice, more than
        ``sleep.sleep_call_timeout_s`` apart (a /sleep still running then has
        timed out). The first awake read is recorded in the journal entry."""
        if record.get("phase") not in ("sleeping", "sleep_unconfirmed"):
            return True
        primitive = self._sleep_primitive
        now_ms = primitive.reservations.now_ms()
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
        """SIGTERM: accept no new sleep; drains in progress (not yet /sleep'ed)
        roll back at their next poll. Safe to call from a signal handler."""
        if self._sleep_primitive is not None:
            self._sleep_primitive.begin_shutdown()
        supervisor = self._supervisor
        stop_flag = getattr(supervisor, "request_stop", None)
        if callable(stop_flag):
            stop_flag()

    def shutdown_timeout_s(self) -> float:
        """Bound of the shutdown wait, from the same time budget as the call
        timeout (review 2 P2-2): a drain rolls back at its next poll, a commit
        already past /sleep finishes."""
        return self._sm_config.shutdown_timeout_s()

    def shutdown(self, *, timeout_s: float) -> bool:
        """begin_shutdown, then wait (bounded) until every sleep has finished or
        rolled back. True when idle."""
        self.begin_shutdown()
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
        self._safety_gate.assert_controller_observe()
        reservations = self._reservations()
        active = reservations.active() if reservations is not None else {}
        if active:
            raise ReservationConflict(
                f"fleet repair waits for sleeps in progress: {sorted(active)}",
                binding_id=sorted(active)[0],
            )
        snapshot = self._store.load()
        targets = (
            sorted(set(awake_binding_ids))
            if awake_binding_ids is not None
            else self._desired_awake_binding_ids(snapshot.bindings)
        )

        def run(operation) -> None:
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
                reconcile=lambda strict: self._reconcile_unlocked(
                    drop_missing=strict
                ),
                set_binding_power=self._set_binding_power_by_id_unlocked,
                audit=self.audit,
            )

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

    def recover_stale_fleet_repairs(self) -> dict | None:
        if self._operation_coordinator is None:
            return None
        if self._operation_coordinator.active_operation() is not None:
            return None
        stale = self._operation_coordinator.stale_running_operations(
            kind="fleet_repair"
        )
        if not stale:
            return None
        self.enter_recovery_observe()
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

    def enter_recovery_observe(self) -> str:
        if self._safety_gate is None:
            raise ValueError("fleet repair safety gate is not configured")
        return self._safety_gate.enter_recovery_observe()

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

        # A fleet repair already owns the global writer fence and has acquired
        # the target lease before scaling the Deployment. Reuse that prepared
        # phase instead of deadlocking the init gate on a second writer.
        active = self._operation_coordinator.active_operation(kind="fleet_repair")
        if active is not None:
            details = active.get("details") or {}
            if (
                active.get("phase") == "starting_binding"
                and details.get("binding_id") == pod.binding_id
                and self._lease_matches(pod.binding_id, phase="starting")
            ):
                self._assert_startup_overlaps_sleeping(pod)
                self._runtime_ops.admit_startup_pod(
                    pod.name,
                    pod_uid=pod.uid,
                    suspended_binding_ids=[],
                    operation_id=str(active["operation_id"]),
                )
                return {
                    "status": "admitted",
                    "binding_id": pod.binding_id,
                    "operation_id": active["operation_id"],
                    "prepared_by_fleet_repair": True,
                }
            raise OperationBusy(
                f"{active.get('owner')}:{active.get('fencing_token')}"
            )

        # Awake residents on the Pod's GPUs are put to sleep first, with their
        # drain OUTSIDE the writer lock (review 2 P2-4); the admission below only
        # verifies that they are asleep. Checks that would refuse the admission
        # anyway run first, so nothing is slept for a Pod that cannot start.
        self._safety_gate.assert_no_pressure()
        if pod.binding_id in self._desired_binding_ids():
            if self._desired_binding(pod.binding_id).lifecycle != "resident":
                raise ValueError(
                    f"startup denied for non-resident desired binding {pod.binding_id}"
                )
        self._sleep_overlapping_residents(pod)

        request = {"pod_name": pod_name, "pod_uid": pod_uid}
        with self._operation_coordinator.operation(
            "startup_admit", request=request
        ) as operation:
            operation.advance("validating_startup", details={"binding_id": pod.binding_id})
            # Checked under the writer lock (review 2 P3): no sleep can start and
            # no transient lease can appear between the check and the admission.
            self._assert_not_reserved(
                slot=Slot(pod.node, pod.gpu_ids), what=f"startup admission of {pod_name}"
            )
            conflict = self._conflicting_transient_lease(pod)
            if conflict is not None:
                gpu_id, occupant = conflict
                raise GpuLeaseConflict(
                    gpu=f"{pod.node}/{gpu_id}", occupant=occupant
                )
            self._safety_gate.assert_no_pressure()
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

    def _sleep_overlapping_residents(self, pod: StartupPodRecord) -> None:
        """Split-sleep (drain outside the writer lock) every awake resident on the
        startup Pod's GPUs. Desired power is untouched: a resident desired awake
        is recorded as suspended by the admission and woken after convergence."""
        if self._sleep_primitive is None:
            return
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
        if awake:
            self._split_sleep(awake, sleep_path="startup", kind="startup_admit_sleep")

    def converge_startups(self) -> dict:
        """Converge admitted Pods after vLLM becomes reachable.

        A Pod desired asleep is split-slept first (drain outside the writer lock,
        review 2 P2-4). A Pod that cannot be converged right now (writer busy,
        a sleep reservation, a failed sleep) stays pending for the next pass; it
        never aborts the other Pods of this pass (review 2 P3)."""
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
                    self._split_sleep(
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
            except (OperationBusy, ReservationConflict, RetryLater, SleepFailed, ServiceShuttingDown):
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
                # converge_startups sleeps it first (drain outside the lock).
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
                and lease.phase in {"starting", "waking"}
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

    def _sync_observed(self, observations) -> None:
        if self._fleet_store is None:
            return
        snapshot = self._fleet_store.load_observed()
        records = []
        for observation in observations:
            binding = observation.binding
            pod = observation.pod
            physical_power = (
                "unknown"
                if observation.physical_awake is None
                else "awake" if observation.physical_awake else "sleeping"
            )
            records.append(
                ObservedBinding(
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
                    error=(
                        "physical_probe_unreachable"
                        if observation.physical_awake is None
                        else None
                    ),
                )
            )
        records.sort(key=lambda item: item.binding_id)
        if records != snapshot.bindings:
            self._fleet_store.save_observed(
                records, expected_version=snapshot.version
            )

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
            # Plan D2: the ordering hide -> gateway ack -> drain -> /sleep lives
            # in the primitive; there is no other way to sleep a pod.
            self._sleep_bindings(
                [binding], sleep_path=sleep_path, drain_budget_s=drain_budget_s
            )
            return
        if action != "wake":
            raise ValueError(f"unknown runtime action: {action}")
        self._assert_not_reserved(binding=binding, what=f"wake of {binding.serve_id}")
        snapshot = self._snapshot_for_binding(binding)
        if not snapshot.pod_ip:
            raise ValueError(f"pod {binding.serve_id} has no pod IP for {action}")
        self._ensure_wake_headroom(binding)
        if self._gpu_leases is not None:
            self._gpu_leases.acquire(binding, phase="waking")
        result = self._vllm_ops.wake_up(snapshot.pod_ip, port=8000)
        if not bool(getattr(result, "success", False)):
            message = getattr(result, "message", "") or "operation failed"
            raise ValueError(f"vLLM {action} failed for {binding.serve_id}: {message}")
        if hasattr(self._vllm_ops, "is_sleeping"):
            physical = self._vllm_ops.is_sleeping(snapshot.pod_ip, port=8000)
            if physical is not False:
                raise ValueError(
                    f"vLLM {action} did not physically converge for {binding.serve_id}"
                )
        self._runtime_ops.write_binding_annotations(binding, state=POD_STATE_AWAKE)
        if self._gpu_leases is not None:
            self._gpu_leases.acquire(binding, phase="awake")

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
    ) -> list[dict]:
        """One sleep call (writer lock held throughout). On a partial failure the
        targets that did sleep are recorded (lease released, store updated) and the
        rolled-back ones get their desired power back before the error propagates."""
        if self._sleep_primitive is None:
            raise ValueError("runtime_ops and vllm_ops are required to sleep a pod")
        try:
            outcomes = self._sleep_primitive.sleep(
                targets, path=sleep_path, drain_budget_s=drain_budget_s
            )
        except SleepFailed as exc:
            self._record_sleep_outcomes(targets, exc.outcomes, update_store=True)
            raise
        self._record_sleep_outcomes(targets, outcomes, update_store=False)
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
        by_id = {target.binding.binding_id: target.binding for target in targets}
        slept = [
            by_id[item["binding_id"]]
            for item in outcomes
            if item.get("status") == STATUS_SLEPT and item.get("binding_id") in by_id
        ]
        if self._gpu_leases is not None:
            for binding in slept:
                self._gpu_leases.release(binding)
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
                    # A reconcile during the drain may have recorded the hide.
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
        """Fleet repair's sleep: the same primitive (plan D1), path ``repair``.

        It keeps the writer lock through the drain (review 2 P2-4: not split):
        a repair is one fenced operation that runs only in observe mode, and
        releasing the lock mid-repair would let other writers interleave with a
        fleet the repair has not finished converging."""
        self._sleep_targets([SleepTarget(binding, pod_ip)], sleep_path="repair")

    def _ensure_wake_headroom(self, binding: Binding) -> None:
        """Fail closed unless gpu-truth shows the binding's GPUs free enough to wake.

        Sleeping residents keep only a small footprint; an awake resident on the
        same GPU (or a sleep leak) is far above the wake threshold (registry
        ``service_manager.wake.max_used_fraction`` of the GPU's total memory, or
        the absolute ``max_used_mib`` override) and would make the wake OOM or
        double-book the GPU (the sleeping-capacity deadlock). gpu-truth lags a
        sleep that just finished, so it is re-read for up to
        ``wake_truth_wait_s`` before refusing.
        """
        if self._gpu_truth is None:
            return
        clock = self._sleep_clock or _default_sleep_clock()
        deadline = clock.monotonic() + self._sm_config.wake_truth_wait_s
        nodes = {node.name: node for node in self._registry.topology().nodes}
        node = nodes.get(binding.slot.node)
        while True:
            problem = self._wake_headroom_problem(binding, node)
            if problem is None:
                return
            if clock.monotonic() >= deadline:
                raise WakeConflict(f"insufficient wake headroom: {problem}")
            clock.sleep(min(1.0, max(0.0, deadline - clock.monotonic())))

    def _wake_headroom_problem(self, binding: Binding, node) -> str | None:
        node_truth = self._gpu_truth.node_truth(node=binding.slot.node)
        if node_truth is None:
            if not self._require_gpu_truth:
                return None
            return (
                f"gpu truth unavailable for node {binding.slot.node} "
                "(is the tre-v2-gpu-truth DaemonSet healthy?)"
            )
        for gpu_id in binding.slot.gpu_ids:
            gpu_uuid = _gpu_uuid(node, gpu_id)
            if gpu_uuid is None:
                if not self._require_gpu_truth:
                    continue
                return f"{binding.slot.node}/{gpu_id} has no GPU UUID in the registry"
            used_mib = node_truth.used_mib(gpu_uuid)
            if used_mib is None:
                if not self._require_gpu_truth:
                    continue
                return f"gpu truth for {binding.slot.node}/{gpu_uuid} missing"
            total = getattr(node_truth, "total_mib", None)
            limit = self._sm_config.wake_limit_mib(total(gpu_uuid) if callable(total) else None)
            if limit is None:
                if not self._require_gpu_truth:
                    continue
                return (
                    f"gpu truth for {binding.slot.node}/{gpu_uuid} reports no total memory "
                    "(set service_manager.wake.max_used_mib to use an absolute threshold)"
                )
            if used_mib > limit:
                return (
                    f"{binding.binding_id}: {binding.slot.node}/{gpu_uuid} "
                    f"used_mib={used_mib} > wake limit {limit} MiB"
                )
        return None

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
                "hard_cap_s": policy.hard_cap_s,
                "sleep_call_timeout_s": policy.sleep_call_timeout_s,
                "physical_confirm_timeout_s": policy.physical_confirm_timeout_s,
                "reservation_ttl_s": policy.reservation_ttl_s,
                "probe_timeout_s": policy.probe_timeout_s,
                "commit_lock_wait_s": self._sm_config.commit_wait_s,
                "shutdown_timeout_s": self._sm_config.shutdown_timeout_s(),
                "vllm_sleep_mode_param": policy.vllm_sleep_mode_param,
                "budgets_s": dict(policy.budgets_s),
                "worst_case_call_s": self._sm_config.worst_case_sleep_call_s(),
            },
            "reservations": {
                binding_id: asdict(reservation)
                for binding_id, reservation in primitive.reservations.active().items()
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
        }

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
        self._assert_not_reserved(slot=slot, what=f"cold start of {model}")
        self._ensure_create_headroom(slot)
        planned = Binding("startup", model, slot, awake=False)
        if self._gpu_leases is not None:
            self._gpu_leases.acquire(planned, phase="starting")
        self._ensure_model_route(model)
        deployment_id = self._runtime_ops.create_model_deployment(model, slot)
        self._ensure_model_route(model)
        ready = self._runtime_ops.wait_pod_ready(deployment_id)
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
        return binding

    def _execute_runtime_defrag_migration(self, binding: Binding, migration: Migration) -> tuple[list[dict], Binding]:
        if self._runtime_ops is None or self._vllm_ops is None:
            raise ValueError("runtime_ops and vllm_ops are required for runtime defrag")

        actions: list[dict] = []
        # The primitive hides, waits for the gateway ack and drains before /sleep.
        self._apply_runtime_power_action(binding, action="sleep", sleep_path="defrag")
        actions.append({"action": "hide", "serve_id": binding.serve_id})
        actions.append({"action": "sleep", "serve_id": binding.serve_id})

        self._runtime_ops.delete_model_deployment(binding)
        actions.append({"action": "delete_deployment", "serve_id": binding.serve_id})

        self._runtime_ops.wait_pod_deleted(binding.serve_id)
        self._ensure_create_headroom(migration.to_slot)
        planned = Binding(
            "defrag-startup", binding.model, migration.to_slot, awake=False
        )
        if self._gpu_leases is not None:
            self._gpu_leases.acquire(planned, phase="starting")
        deployment_id = self._runtime_ops.create_model_deployment(binding.model, migration.to_slot)
        self._ensure_model_route(binding.model)
        ready = self._runtime_ops.wait_pod_ready(deployment_id)
        new_serve_id = ready.name
        actions.append(
            {
                "action": "create_deployment",
                "serve_id": new_serve_id,
                "node": migration.to_slot.node,
                "gpu_ids": list(migration.to_slot.gpu_ids),
            }
        )
        if not ready.pod_ip:
            raise ValueError(f"pod {new_serve_id} has no pod IP for wake")
        ready_result = self._vllm_ops.wait_until_ready(ready.pod_ip, port=8000)
        if not bool(getattr(ready_result, "success", False)):
            message = getattr(ready_result, "message", "") or "operation failed"
            raise ValueError(f"vLLM readiness failed for {new_serve_id}: {message}")
        result = self._vllm_ops.wake_up(ready.pod_ip, port=8000)
        if not bool(getattr(result, "success", False)):
            message = getattr(result, "message", "") or "operation failed"
            raise ValueError(f"vLLM wake failed for {new_serve_id}: {message}")

        moved = Binding(new_serve_id, binding.model, migration.to_slot, awake=True, hidden=False)
        self._runtime_ops.write_binding_annotations(moved, state=POD_STATE_AWAKE)
        if self._gpu_leases is not None:
            self._gpu_leases.acquire(moved, phase="awake")
        actions.append({"action": "wake", "serve_id": new_serve_id})
        actions.append({"action": "unhide", "serve_id": new_serve_id})
        return actions, moved

    def _ensure_feasible_wake(self, binding: Binding, bindings: list[Binding]) -> None:
        if not self._feasible_wake(binding, bindings):
            raise WakeConflict(f"{binding.serve_id}: slot already has awake binding")

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

    def _ensure_create_headroom(self, slot: Slot) -> None:
        # Fail closed: a cold start writes model weights onto a GPU we believe
        # is free. When the gpu-truth DaemonSet is down its Redis key expires,
        # and absent truth used to silently skip this gate -- the one guard
        # standing between a stale view and two models on one GPU. Refuse
        # instead. Set TRE_GPU_TRUTH_REQUIRED=false to restore the old
        # permissive behaviour if truth is unavailable during an emergency.
        if self._gpu_truth is None:
            return
        node_truth = self._gpu_truth.node_truth(node=slot.node)
        if node_truth is None:
            if not self._require_gpu_truth:
                return
            raise ValueError(
                f"gpu truth unavailable for node {slot.node}: refusing cold start "
                "(is the tre-v2-gpu-truth DaemonSet healthy?)"
            )
        nodes = {node.name: node for node in self._registry.topology().nodes}
        node = nodes.get(slot.node)
        for gpu_id in slot.gpu_ids:
            gpu_uuid = _gpu_uuid(node, gpu_id)
            if gpu_uuid is None:
                continue
            used_mib = node_truth.used_mib(gpu_uuid)
            if used_mib is None:
                if not self._require_gpu_truth:
                    continue
                raise ValueError(
                    f"gpu truth unavailable for {slot.node}/{gpu_uuid}: refusing cold start "
                    "(gpu missing from the node truth payload)"
                )
            if used_mib <= self._create_max_used_mib:
                continue
            raise ValueError(
                "insufficient startup headroom: "
                f"{slot.node}/{gpu_uuid} used_mib={used_mib} max_used_mib={self._create_max_used_mib}"
            )

    def _snapshot_for_binding(self, binding: Binding) -> K8sPodSnapshot:
        snapshots = self._runtime_ops.list_pod_snapshots(model=binding.model) if self._runtime_ops else []
        for snapshot in snapshots:
            if snapshot.name == binding.serve_id:
                return snapshot
        raise ValueError(f"pod {binding.serve_id} not found for runtime operation")

    def _model_counts(self, bindings: list[Binding]) -> dict[str, dict[str, int]]:
        counts = {model.name: {"awake": 0, "bound": 0} for model in self._registry.models()}
        for binding in bindings:
            bucket = counts.setdefault(binding.model, {"awake": 0, "bound": 0})
            bucket["bound"] += 1
            if binding.awake:
                bucket["awake"] += 1
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


class DefragUnavailable(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class WakeConflict(ValueError):
    pass


class RetryLater(RuntimeError):
    """The request cannot proceed right now but may succeed when retried (HTTP
    409), e.g. a resident on a startup Pod's GPUs woke up again."""


class TargetRequest(BaseModel):
    wake_replicas: int
    #: Which sleep path this is (registry service_manager.sleep.budgets_s key).
    sleep_path: str = "scale_down"
    #: Soft drain budget override (s); capped by the hard cap.
    drain_budget_s: float | None = None


class BindingPowerRequest(BaseModel):
    awake: bool
    sleep_path: str = "scale_down"
    drain_budget_s: float | None = None


class DefragRequest(BaseModel):
    tp_size: int




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
            content={"detail": str(exc), "current_writer": exc.current_writer},
        )

    @app.exception_handler(StateFenceError)
    async def state_fence_handler(
        _request: Request, exc: StateFenceError
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ControllerNotPaused)
    async def controller_not_paused_handler(
        _request: Request, exc: ControllerNotPaused
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

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

    @app.exception_handler(ReservationConflict)
    async def reservation_conflict_handler(
        _request: Request, exc: ReservationConflict
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409, content={"detail": str(exc), "binding_id": exc.binding_id}
        )

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
        return JSONResponse(status_code=409, content={"detail": str(exc)})

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
    def admit_startup(request: StartupAdmissionRequest) -> dict:
        try:
            return service.admit_startup(
                pod_name=request.pod_name, pod_uid=request.pod_uid
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v2/startup/converge")
    def converge_startups() -> dict:
        return service.converge_startups()

    @app.get("/v2/state")
    def get_state() -> dict:
        return service.get_state()


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

    @app.get("/v2/supervisor")
    def get_supervisor() -> dict:
        return service.get_supervisor_state()


    @app.post("/v2/defrag")
    def defrag(request: DefragRequest) -> dict:
        try:
            return service.defrag(tp_size=request.tp_size)
        except DefragUnavailable as exc:
            raise HTTPException(status_code=409, detail={"reason": exc.reason}) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @app.put("/v2/models/{model}/routable")
    def put_model_routable(model: str, request: RoutableRequest) -> dict:
        try:
            return service.put_model_routable(model, hidden_pods=request.hidden_pods)
        except WakeConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
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
            )
        except WakeConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
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
        except WakeConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app

def _sleep_path(value: str) -> str:
    if value not in SLEEP_PATHS:
        raise ValueError(f"unknown sleep_path {value!r} (known: {', '.join(SLEEP_PATHS)})")
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


def _wake_pick(feasible, planned, topology) -> Binding:
    """Which sleeping binding to wake: buddy best-fit, same rule as the planner.

    Packing beats spreading here: two single-GPU replicas on one aligned pair keep
    the other pair whole for a tp=2 model, where one replica per node would leave
    neither node able to host it.
    """
    nodes = node_gpu_counts(topology)
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
        )
        if choice is not None:
            return scorable[choice.index]
    return min(feasible, key=lambda item: _natural_key(item.serve_id))


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

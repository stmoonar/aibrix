from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import threading
import time
from typing import Protocol

from tre_sm.state.operations import OperationBusy
from tre_sm.state.safety import MaintenanceLockLost, NodePressureActive
from tre_sm.state.sleep_reservations import ReservationConflict

LOG = logging.getLogger(__name__)


class SupervisedService(Protocol):
    def converge_startups(self) -> dict: ...
    def recover_stale_fleet_repairs(self) -> dict | None: ...
    def detect_fleet_drift(self) -> list[dict]: ...
    def start_fleet_repair(self, *, awake_binding_ids=None, recovered_from=None) -> dict: ...


@dataclass(frozen=True)
class SupervisorSnapshot:
    running: bool
    last_error: str | None
    drift_observations: int
    last_drift: list[dict]
    last_recovery_operation_id: str | None
    #: Informational drift items (``informational: true``, e.g.
    #: ``startup_admission_pending``) of the last pass: reported, never repaired.
    informational: list[dict] = field(default_factory=list)


class FleetSupervisor:
    """Converge startup gates and recover persistent fleet drift.

    Drift must be identical for several observations before a repair is
    submitted. This filters normal Pod transitions while still recognizing a
    batch eviction/replacement event. Repairs remain fail-closed: the repair
    holds the SM maintenance lock and the pressure hysteresis gate.

    SM actuation ``observe`` (``tre:v2:sm:actuation``, user decision
    2026-09-28): every supervisor action that changes capacity or recreates /
    deletes workloads - B7 recreate, drift -> fleet repair, stale repair
    recovery, reaping rejected Deployments - only logs and records what it
    would have done. State-consistency passes that change no awake count keep
    running: sleep journal recovery, desired seeding, startup convergence of
    admitted Pods, orphan ``starting`` lease reaping.
    """

    def __init__(
        self,
        service: SupervisedService,
        *,
        interval_s: float = 5.0,
        drift_observations_required: int = 3,
        repair_cooldown_s: float = 60.0,
        monotonic=time.monotonic,
    ) -> None:
        self._service = service
        self._interval_s = interval_s
        self._required = drift_observations_required
        self._repair_cooldown_s = repair_cooldown_s
        self._monotonic = monotonic
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_signature: tuple | None = None
        self._drift_observations = 0
        self._last_drift: list[dict] = []
        self._last_informational: list[dict] = []
        self._last_error: str | None = None
        self._last_recovery_operation_id: str | None = None
        self._last_repair_at: float | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="tre-sm-fleet-supervisor", daemon=True
        )
        self._thread.start()

    def request_stop(self) -> None:
        """Signal-safe: stop after the current pass (no join)."""
        self._stop.set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s + 2.0)

    def snapshot(self) -> SupervisorSnapshot:
        return SupervisorSnapshot(
            running=self._thread is not None and self._thread.is_alive(),
            last_error=self._last_error,
            drift_observations=self._drift_observations,
            last_drift=list(self._last_drift),
            last_recovery_operation_id=self._last_recovery_operation_id,
            informational=list(self._last_informational),
        )

    def _actuation_observe(self) -> bool:
        """SM actuation observe? A service without the switch (embedded / unit
        tests without a safety gate) is treated as active."""
        reader = getattr(self._service, "actuation_observe", None)
        return bool(reader()) if callable(reader) else False

    def _suppress(self, action: str, detail: dict) -> None:
        record = getattr(self._service, "record_suppressed", None)
        if callable(record):
            record(action, detail)
        else:
            LOG.warning(json.dumps({"event": "sm_supervisor_action_suppressed", "action": action,
                                    "detail": detail}, sort_keys=True, default=str))

    def run_once(self) -> None:
        observe = self._actuation_observe()
        recover = getattr(self._service, "recover_sleep_journal", None)
        if callable(recover):
            try:
                recover()
            except OperationBusy:
                pass  # another writer; next pass
        recover_wakes = getattr(self._service, "recover_wake_journal", None)
        if callable(recover_wakes):
            try:
                recover_wakes()
            except OperationBusy:
                pass  # another writer; next pass
        ensure_seeded = getattr(self._service, "ensure_desired_seeded", None)
        if callable(ensure_seeded):
            try:
                ensure_seeded()
            except OperationBusy:
                pass  # another writer; next pass
        self._service.converge_startups()
        reap = getattr(self._service, "reap_rejected_deployments", None)
        if callable(reap):
            try:
                reap(actuate=False) if observe else reap()
            except OperationBusy:
                pass  # a writer (possibly starting a Pod) is active; next pass
        reap_leases = getattr(self._service, "reap_orphan_starting_leases", None)
        if callable(reap_leases):
            try:
                reap_leases()
            except OperationBusy:
                pass  # a writer (possibly starting a Pod) is active; next pass
        recovered = (
            self._service.recover_stale_fleet_repairs(actuate=False)
            if observe
            else self._service.recover_stale_fleet_repairs()
        )
        if recovered is not None:
            self._last_recovery_operation_id = str(recovered["operation_id"])
            self._reset_drift()
            return

        # Informational items (e.g. a Pod waiting in its startup gate, review
        # 2026-09-29 P1-1) are reported but never count as drift: they neither
        # start nor feed the observations that lead to a fleet repair.
        reported = self._service.detect_fleet_drift()
        informational = [item for item in reported if item.get("informational")]
        def keys(items):
            return sorted((item.get("code"), item.get("binding_id")) for item in items)

        if informational and keys(informational) != keys(self._last_informational):
            LOG.info(json.dumps({"event": "sm_supervisor_informational_drift",
                                 "items": informational}, sort_keys=True, default=str))
        self._last_informational = informational
        drift = [item for item in reported if not item.get("informational")]
        signature = tuple(
            sorted(
                (item.get("code"), item.get("binding_id"), item.get("count"))
                for item in drift
            )
        )
        if not drift:
            self._reset_drift()
            return
        if signature == self._last_signature:
            self._drift_observations += 1
        else:
            self._last_signature = signature
            self._drift_observations = 1
        self._last_drift = drift
        if self._drift_observations < self._required:
            return
        now = self._monotonic()
        if (
            self._last_repair_at is not None
            and now - self._last_repair_at < self._repair_cooldown_s
        ):
            return
        targeted = getattr(self._service, "repair_missing_deployments", None)
        if callable(targeted) and all(
            item.get("code") == "deployment_missing" for item in drift
        ):
            # B7: only Deployments of sleeping residents are gone - recreate
            # just those (their Pods pass the startup gate) instead of a
            # fleet-wide repair. None = not eligible: full repair below.
            ids = [item.get("binding_id") for item in drift]
            repaired = targeted(ids, actuate=False) if observe else targeted(ids)
            if repaired is not None:
                self._last_repair_at = now
                self._reset_drift()
                return
        if observe:
            self._suppress("fleet_repair", {"drift": drift})
            self._last_repair_at = now
            self._reset_drift()
            return
        submitted = self._service.start_fleet_repair()
        self._last_repair_at = now
        self._last_recovery_operation_id = str(submitted["operation_id"])
        self._reset_drift()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
                self._last_error = None
            except (OperationBusy, MaintenanceLockLost, NodePressureActive, ReservationConflict):
                # Expected gates: another writer is converging, the maintenance
                # lock was taken away, pressure remains, or a sleep is draining.
                # Retry without mutating intent.
                pass
            except Exception as exc:  # keep supervision alive and observable.
                self._last_error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(self._interval_s)

    def _reset_drift(self) -> None:
        self._last_signature = None
        self._drift_observations = 0
        self._last_drift = []

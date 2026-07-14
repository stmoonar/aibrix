from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Protocol

from tre_sm.state.operations import OperationBusy
from tre_sm.state.safety import ControllerNotPaused, NodePressureActive


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


class FleetSupervisor:
    """Converge startup gates and recover persistent fleet drift.

    Drift must be identical for several observations before a repair is
    submitted. This filters normal Pod transitions while still recognizing a
    batch eviction/replacement event. Repairs remain fail-closed: controller
    observe mode and the pressure hysteresis gate are enforced by the repair.
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
        )

    def run_once(self) -> None:
        self._service.converge_startups()
        recovered = self._service.recover_stale_fleet_repairs()
        if recovered is not None:
            self._last_recovery_operation_id = str(recovered["operation_id"])
            self._reset_drift()
            return

        drift = self._service.detect_fleet_drift()
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
        submitted = self._service.start_fleet_repair()
        self._last_repair_at = now
        self._last_recovery_operation_id = str(submitted["operation_id"])
        self._reset_drift()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
                self._last_error = None
            except (OperationBusy, ControllerNotPaused, NodePressureActive):
                # Expected gates: another writer is converging, controller is
                # active, or pressure remains. Retry without mutating intent.
                pass
            except Exception as exc:  # keep supervision alive and observable.
                self._last_error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(self._interval_s)

    def _reset_drift(self) -> None:
        self._last_signature = None
        self._drift_observations = 0
        self._last_drift = []

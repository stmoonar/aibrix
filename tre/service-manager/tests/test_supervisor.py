from tre_sm.state.supervisor import FleetSupervisor


class FakeService:
    def __init__(self):
        self.drift = []
        self.stale = None
        self.repairs = []
        self.converges = 0
        self.observe_handoffs = 0

    def converge_startups(self):
        self.converges += 1
        return {"converged": [], "pending": []}

    def recover_stale_fleet_repairs(self):
        result = self.stale
        self.stale = None
        return result

    def detect_fleet_drift(self):
        return list(self.drift)

    def start_fleet_repair(self, **_kwargs):
        self.repairs.append(True)
        return {"operation_id": "repair-1"}

    def enter_recovery_observe(self):
        self.observe_handoffs += 1
        return "active"


def test_supervisor_debounces_batch_drift_before_repair():
    service = FakeService()
    service.drift = [
        {"code": "pod_cardinality", "binding_id": "m/node/0", "count": 0},
        {"code": "pod_cardinality", "binding_id": "m/node/1", "count": 0},
    ]
    supervisor = FleetSupervisor(
        service, interval_s=0.01, drift_observations_required=3
    )

    supervisor.run_once()
    supervisor.run_once()
    assert service.repairs == []
    assert service.observe_handoffs == 0
    supervisor.run_once()

    assert service.repairs == [True]
    assert service.observe_handoffs == 1
    assert service.converges == 3
    assert supervisor.snapshot().last_recovery_operation_id == "repair-1"


def test_supervisor_prioritizes_stale_operation_recovery():
    service = FakeService()
    service.stale = {"operation_id": "replacement"}
    service.drift = [
        {"code": "pod_not_ready", "binding_id": "m/node/0"}
    ]
    supervisor = FleetSupervisor(service, drift_observations_required=1)

    supervisor.run_once()

    assert service.repairs == []
    assert supervisor.snapshot().last_recovery_operation_id == "replacement"

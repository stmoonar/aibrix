"""Observe = record only (user decision 2026-09-28), service-manager side.

* SM actuation observe (``tre:v2:sm:actuation``): supervisor actions that
  change capacity or recreate / delete workloads (B7 recreate, drift -> fleet
  repair, stale repair recovery, reaping rejected Deployments) only log and
  record; state-consistency passes keep running.
* A startup admission nobody requested (no owning operation) sleeps no awake
  resident in observe; a Pod an operation creates itself is still admitted.
* The SM HTTP write API is not gated (APA arm, operators); the caller is kept
  in the operation log.
* The fleet repair holds its own maintenance lock, never the controller mode.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from tre_common import rediskeys
from tre_sm.api.v2 import RetryLater, ServiceManagerV2, create_app
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.ops.k8s_ops import ModelDeploymentRecord
from tre_sm.state.operations import OperationCoordinator
from tre_sm.state.safety import ClusterSafetyGate
from tre_sm.state.store import StateStore
from tre_sm.state.supervisor import FleetSupervisor

from sm_test_fakes import FakeRedis, FakeSafety, LegacyRedis, fence, pod, registry
from test_operations import ScriptRedis, _registry as _ops_registry
from test_review2_sleep import _desired, _startup_world
from test_review4_sm import _cold_start_world, _gate_world
from test_supervisor import FakeService


# ------------------------------------------------------------------ supervisor
class ObserveAwareService(FakeService):
    """FakeService with the actuation switch and the actuate-aware passes."""

    def __init__(self, *, observe: bool):
        super().__init__()
        self.observe = observe
        self.suppressed: list[tuple[str, dict]] = []
        self.calls: list[tuple] = []

    def actuation_observe(self):
        return self.observe

    def record_suppressed(self, action, detail):
        self.suppressed.append((action, detail))

    def recover_sleep_journal(self):
        self.calls.append(("recover_sleep_journal",))

    def ensure_desired_seeded(self):
        self.calls.append(("ensure_desired_seeded",))

    def reap_orphan_leases(self):
        self.calls.append(("reap_orphan_leases",))
        return []

    def reap_rejected_deployments(self, *, actuate=True):
        self.calls.append(("reap_rejected_deployments", actuate))
        return []

    def recover_stale_fleet_repairs(self, *, actuate=True):
        self.calls.append(("recover_stale_fleet_repairs", actuate))
        return None

    def repair_missing_deployments(self, binding_ids, *, actuate=True):
        self.calls.append(("repair_missing_deployments", tuple(binding_ids), actuate))
        return {"binding_ids": list(binding_ids), "created": [], "suppressed": not actuate}


def _passes(service, drift, n=3):
    service.drift = drift
    supervisor = FleetSupervisor(service, drift_observations_required=2)
    for _ in range(n):
        supervisor.run_once()
    return supervisor


def test_supervisor_in_observe_records_a_fleet_repair_instead_of_starting_it():
    service = ObserveAwareService(observe=True)
    drift = [{"code": "pod_not_ready", "binding_id": "m1/node-a/0"}]

    _passes(service, drift)

    assert service.repairs == []
    assert service.observe_handoffs == 0
    assert service.suppressed == [("fleet_repair", {"drift": drift})]
    # consistency passes ran every pass; the capacity-changing ones as dry runs
    names = [call[0] for call in service.calls]
    for name in ("recover_sleep_journal", "ensure_desired_seeded", "reap_orphan_leases"):
        assert names.count(name) == 3
    assert service.converges == 3
    assert ("reap_rejected_deployments", False) in service.calls
    assert ("reap_rejected_deployments", True) not in service.calls
    assert ("recover_stale_fleet_repairs", False) in service.calls


def test_supervisor_in_observe_runs_b7_only_as_a_dry_run():
    service = ObserveAwareService(observe=True)
    _passes(service, [{"code": "deployment_missing", "binding_id": "m1/node-a/1"}], n=2)
    assert ("repair_missing_deployments", ("m1/node-a/1",), False) in service.calls
    assert service.repairs == []


def test_supervisor_in_active_still_repairs():
    service = ObserveAwareService(observe=False)
    _passes(service, [{"code": "pod_not_ready", "binding_id": "m1/node-a/0"}], n=2)
    assert service.repairs == [True]
    assert service.suppressed == []
    assert ("reap_rejected_deployments", True) in service.calls


# ------------------------------------------------------------------ service passes
def test_b7_recreate_in_observe_creates_nothing_and_is_recorded():
    world = _gate_world(
        [pod("pod-a", "m1", (0,), ip="10.0.0.1")],
        [_desired("m1/node-a/0", "m1", (0,), "awake"), _desired("m1/node-a/1", "m1", (1,), "sleeping")],
    )
    result = world.service.repair_missing_deployments(["m1/node-a/1"], actuate=False)

    assert result == {"binding_ids": ["m1/node-a/1"], "created": [], "suppressed": True}
    assert world.runtime.created == []
    assert world.coordinator.kinds == []
    assert world.service._safety_gate.suppressed == [
        ("recreate_missing_deployments", {"binding_ids": ["m1/node-a/1"]})
    ]


def test_reaping_rejected_deployments_in_observe_deletes_nothing():
    world = _cold_start_world()
    world.runtime.deployments.append(ModelDeploymentRecord("m1-node-a-gpu-1", "m1", "node-a", (1,), 1))
    absent = replace(_desired("m1/node-a/1", "m1", (1,), "sleeping"), lifecycle="absent")
    with fence(world.redis):
        snapshot = world.fleet.load_desired()
        world.fleet.save_desired(snapshot.bindings + [absent], expected_version=snapshot.version)

    assert world.service.reap_rejected_deployments(actuate=False) == []
    assert "m1-node-a-gpu-1" in [d.name for d in world.runtime.deployments]
    assert world.service._safety_gate.suppressed == [
        ("reap_rejected_deployments", {"deployments": ["m1-node-a-gpu-1"]})
    ]
    assert world.service.reap_rejected_deployments() == ["m1-node-a-gpu-1"]  # active: reaped


class _StaleCoordinator:
    owner = "sm-test"

    def active_operation(self, *, kind=None):
        return None

    def stale_running_operations(self, *, kind=None):
        return [{"operation_id": "old-1", "request": {"awake_binding_ids": ["m1/node-a/0"]}}]

    def submit(self, *_args, **_kwargs):
        raise AssertionError("no repair may be submitted in observe")


def test_stale_fleet_repair_recovery_in_observe_is_only_recorded():
    safety = FakeSafety("observe")
    service = ServiceManagerV2(
        registry(), StateStore(FakeRedis()), operation_coordinator=_StaleCoordinator(), safety_gate=safety
    )
    assert service.recover_stale_fleet_repairs(actuate=False) is None
    assert safety.suppressed == [("fleet_repair_recovery", {"stale_operation_ids": ["old-1"]})]


# ------------------------------------------------------------ maintenance lock
class _KV(FakeRedis):
    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value

    def delete(self, *keys):
        for key in keys:
            self.values.pop(key, None)


class _NoPressure:
    def node_pressure_reasons(self):
        return {}


class _InlineCoordinator:
    """submit() runs the operation inline with a handle named ``op-1``."""

    owner = "sm-test"

    class _Handle:
        operation_id = "op-1"

        def supersede(self, _operation_id):
            return None

        def advance(self, *_args, **_kwargs):
            return None

    def __init__(self):
        self.seen = []

    def submit(self, kind, target, *, request=None):
        target(self._Handle())
        return "op-1"


def test_the_fleet_repair_holds_the_sm_maintenance_lock_and_never_touches_the_controller_mode():
    redis = _KV()
    redis.values[rediskeys.CONTROLLER_MODE_KEY] = "active"  # no longer a precondition
    gate = ClusterSafetyGate(redis, _NoPressure())
    store = StateStore(LegacyRedis())
    store.save([Binding("pod-a", "m1", Slot("node-a", (0,)), awake=True)], expected_version=0)
    service = ServiceManagerV2(registry(), store, operation_coordinator=_InlineCoordinator(), safety_gate=gate)
    held = []

    class Executor:
        def run(self, operation, **_kwargs):
            held.append(json.loads(redis.values[rediskeys.SM_MAINTENANCE_KEY]))
            gate.wait_until_healthy(type("Op", (), {
                "operation_id": operation.operation_id,
                "assert_active": lambda self: None,
                "advance": lambda self, *a, **k: None,
            })())

    service._fleet_repair = Executor()

    assert service.start_fleet_repair()["operation_id"] == "op-1"
    assert held[0]["operation_id"] == "op-1" and held[0]["kind"] == "fleet_repair"
    assert rediskeys.SM_MAINTENANCE_KEY not in redis.values  # released at the end
    assert redis.values[rediskeys.CONTROLLER_MODE_KEY] == "active"  # untouched


# ------------------------------------------------------------ startup admission
def test_an_unrequested_startup_in_observe_sleeps_no_awake_resident():
    world = _startup_world()  # tp2 awake on GPUs 0,1; the m1 Pod starts on GPU 0
    world.service._safety_gate.actuation = "observe"

    with pytest.raises(RetryLater, match="SM actuation is observe"):
        world.service.admit_startup(pod_name="m1-new", pod_uid="new-uid")

    assert world.vllm.sleeping["10.0.0.5"] is False  # the resident keeps serving
    assert world.coordinator.kinds == []  # no sleep, no admission
    assert world.service._safety_gate.suppressed == [
        ("startup_admission_sleep",
         {"pod": "m1-new", "binding_id": "m1/node-a/0", "would_sleep": ["tp2/node-a/0,1"]})
    ]


def test_an_unrequested_startup_in_observe_is_admitted_when_nothing_must_sleep():
    world = _startup_world()
    world.vllm.sleeping["10.0.0.5"] = True  # the overlapping resident is asleep
    world.service._safety_gate.actuation = "observe"

    result = world.service.admit_startup(pod_name="m1-new", pod_uid="new-uid")

    assert result["status"] == "admitted"
    assert world.coordinator.kinds == ["startup_admit"]
    assert world.service._safety_gate.suppressed == []


def test_an_unrequested_startup_in_active_still_sleeps_the_resident():
    world = _startup_world()
    result = world.service.admit_startup(pod_name="m1-new", pod_uid="new-uid")
    assert result["suspended_binding_ids"] == ["tp2/node-a/0,1"]


def test_a_cold_start_requested_through_the_api_is_admitted_in_observe():
    # The APA arm scales through the SM API: its cold start owns the Pod
    # (phase starting_binding), so the admission is pre-authorized in observe.
    world = _cold_start_world()
    world.service._safety_gate.actuation = "observe"

    world.service.put_model_target("m1", wake_replicas=2)

    created = world.runtime.created
    assert len(created) == 1
    assert world.runtime.admitted[world.runtime.pending[created[0]][0]]["operation_id"] == "put_model_target-1"
    assert world.service._safety_gate.suppressed == []


# ------------------------------------------------------------ SM HTTP API
def _http_service():
    redis = ScriptRedis()
    service = ServiceManagerV2(
        _ops_registry(),
        StateStore(redis, require_fence=True),
        operation_coordinator=OperationCoordinator(redis, owner="sm-pod"),
        safety_gate=FakeSafety("observe"),
    )
    return TestClient(create_app(service))


def test_sm_http_writes_work_in_observe_and_record_the_actor():
    client = _http_service()

    response = client.put(
        "/v2/models/m1/target", json={"wake_replicas": 1}, headers={"X-TRE-Actor": "aibrix-apa"}
    )

    assert response.status_code == 200
    [operation] = client.get("/v2/operations").json()["operations"]
    assert operation["kind"] == "put_model_target" and operation["status"] == "succeeded"
    assert operation["request"]["actor"] == "aibrix-apa"
    supervisor = client.get("/v2/supervisor").json()
    assert supervisor["actuation"]["mode"] == "observe"


def test_the_actor_falls_back_to_user_agent_and_remote_address():
    client = _http_service()
    client.put("/v2/models/m1/target", json={"wake_replicas": 1}, headers={"User-Agent": "Go-http-client/1.1"})
    [operation] = client.get("/v2/operations").json()["operations"]
    assert operation["request"]["actor"].startswith("ua=Go-http-client/1.1;addr=")

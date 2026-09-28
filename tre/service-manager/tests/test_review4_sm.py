"""Review 4 of the transparent-sleep service-manager.

P1: a writer that creates a gated Pod itself (defrag, cold start) holds the
writer lock until the Pod is ready; the Pod's init gate must be admitted
without that lock (pre-authorized ``starting_binding``), else both wait on each
other until the creator times out. A failed start deletes its Deployment; the
supervisor reaps what is left. P2-2: no unhide of a pod not confirmed awake.
P3: startup admission during shutdown is a retriable 503.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
import threading
import time

import pytest
from fastapi.testclient import TestClient

from tre_sm.allocator.slots import Binding, Slot
from tre_sm.api.v2 import RetryLater, create_app
from tre_sm.ops.k8s_ops import ModelDeploymentRecord, StartupPodRecord
from tre_sm.state.operations import OperationBusy, _CURRENT_OPERATION
from tre_sm.state.safety import NodePressureActive

from sm_test_fakes import FakeSafety, binding_of, fence, pod
from test_review2_sleep import World, _desired


# ------------------------------------------------------------------ fakes
class PhaseHandle:
    def __init__(self, coordinator, record):
        self._coordinator = coordinator
        self.operation_id = record["operation_id"]
        self._record = record

    def advance(self, phase, *, details=None):
        with self._coordinator.lock:
            self._record["phase"] = phase
            if details:
                self._record["details"] = dict(details)
            else:
                self._record.pop("details", None)
            self._coordinator.phases.append((self._record["kind"], phase, dict(details or {})))

    def assert_active(self):
        return None

    def supersede(self, _operation_id):
        return None


class PhaseCoordinator:
    """Non-reentrant, thread-safe writer lock whose journal record (phase,
    details) is what ``active_operation`` returns - like the Redis coordinator."""

    owner = "sm-test"

    def __init__(self, redis):
        self.redis = redis
        self.lock = threading.Lock()
        self.active = None
        self.kinds = []
        self.phases = []
        self.busy = []

    @contextmanager
    def operation(self, kind, *, request=None, wait_s=0.0):
        with self.lock:
            if self.active is not None:
                self.busy.append(kind)
                raise OperationBusy(f"sm-test:{self.active['operation_id']}")
            self.kinds.append(kind)
            record = {
                "operation_id": f"{kind}-{len(self.kinds)}",
                "kind": kind,
                "owner": self.owner,
                "fencing_token": len(self.kinds),
                "status": "running",
                "phase": "acquired",
            }
            self.active = record
        handle = PhaseHandle(self, record)
        token = _CURRENT_OPERATION.set(handle)
        try:
            with fence(self.redis, record["operation_id"]):
                yield handle
        finally:
            _CURRENT_OPERATION.reset(token)
            with self.lock:
                self.active = None

    def active_operation(self, *, kind=None):
        with self.lock:
            if self.active is None:
                return None
            if kind is not None and self.active["kind"] != kind:
                return None
            return dict(self.active)

    def stale_running_operations(self, *, kind=None):
        return []

    def list_operations(self, *, limit=100):
        return []


class Leases:
    def __init__(self):
        self.held = {}
        self.calls = []

    def acquire(self, binding, *, phase):
        self.calls.append(("acquire", binding.binding_id, phase))
        self.held[binding.binding_id] = (binding, phase)

    def release(self, binding):
        self.calls.append(("release", binding.binding_id))
        self.held.pop(binding.binding_id, None)

    def load(self):
        return [
            SimpleNamespace(
                binding_id=binding.binding_id, node=binding.slot.node,
                gpu_ids=tuple(binding.slot.gpu_ids), phase=phase,
            )
            for binding, phase in list(self.held.values())
        ]


def _deployment_name(model, slot):
    return f"{model}-{slot.node}-gpu-{'-'.join(str(g) for g in slot.gpu_ids)}"


def _gate_world(snapshots, desired, *, gate_polls=20):
    """World whose runtime creates gated Pods: ``wait_pod_ready`` runs the
    Pod's init gate (``/v2/startup/admit`` polls) in its own thread and only
    returns once the Pod was admitted, like kubelet + the gate script."""
    world = World(snapshots, desired)
    service = world.service
    world.coordinator = PhaseCoordinator(world.redis)
    service._operation_coordinator = world.coordinator
    world.leases = Leases()
    service._gpu_leases = world.leases
    runtime = world.runtime
    runtime.deployments = [
        ModelDeploymentRecord(_deployment_name(s.model, binding_of(s).slot), s.model, s.node, binding_of(s).slot.gpu_ids, 1)
        for s in snapshots
    ]
    runtime.pending = {}
    runtime.admitted = {}
    runtime.created = []
    runtime.deleted = []
    runtime.cleared = []
    runtime.gate_answers = []
    runtime.fail_delete = 0
    state = {"n": 0}

    def create_model_deployment(model, slot):
        name = _deployment_name(model, slot)
        state["n"] += 1
        runtime.created.append(name)
        runtime.deployments.append(ModelDeploymentRecord(name, model, slot.node, tuple(slot.gpu_ids), 1))
        runtime.pending[name] = (f"{name}-pod{state['n']}", f"uid-new-{state['n']}", model, slot)
        return name

    def get_startup_pod(name):
        for pod_name, uid, model, slot in runtime.pending.values():
            if pod_name == name:
                return StartupPodRecord(
                    name=pod_name, uid=uid, model=model, node=slot.node, gpu_ids=tuple(slot.gpu_ids),
                    annotations={}, labels={}, pod_ip=None, phase="Pending", ready=False,
                )
        raise KeyError(name)

    def admit_startup_pod(name, **kwargs):
        runtime.admitted[name] = kwargs

    def wait_pod_ready(deployment_id):
        pod_name, uid, model, slot = runtime.pending[deployment_id]

        def gate():
            for _ in range(gate_polls):
                try:
                    status, body = service.request_startup_admission(pod_name=pod_name, pod_uid=uid)
                    runtime.gate_answers.append(status)
                    if status == 200:
                        return
                except Exception as exc:  # the gate script logs and polls again
                    runtime.gate_answers.append(type(exc).__name__)
                time.sleep(0.005)

        thread = threading.Thread(target=gate)
        thread.start()
        thread.join(10)
        if pod_name not in runtime.admitted:
            raise TimeoutError(f"pod {pod_name} was not ready before timeout (init gate refused)")
        ip = f"10.0.1.{state['n']}"
        snapshot = pod(pod_name, model, slot.gpu_ids, ip=ip, state="hidden", uid=uid)
        runtime.snapshots[pod_name] = snapshot
        world.vllm.sleeping[ip] = False
        return snapshot

    def delete_model_deployment(binding):
        if runtime.fail_delete:
            runtime.fail_delete -= 1
            raise RuntimeError("apiserver unavailable")
        name = _deployment_name(binding.model, binding.slot)
        runtime.deleted.append(name)
        runtime.deployments = [d for d in runtime.deployments if d.name != name]
        runtime.snapshots.pop(binding.serve_id, None)
        return name

    runtime.create_model_deployment = create_model_deployment
    runtime.get_startup_pod = get_startup_pod
    runtime.admit_startup_pod = admit_startup_pod
    runtime.wait_pod_ready = wait_pod_ready
    runtime.delete_model_deployment = delete_model_deployment
    runtime.wait_pod_deleted = lambda serve_id: None
    runtime.list_startup_resident_snapshots = lambda: runtime.list_pod_snapshots()
    runtime.clear_startup_admission = lambda name: runtime.cleared.append(name)
    return world


def _cold_start_world(**kwargs):
    return _gate_world(
        [pod("pod-a", "m1", (0,), ip="10.0.0.1")],
        [_desired("m1/node-a/0", "m1", (0,), "awake")],
        **kwargs,
    )


def _defrag_world(**kwargs):
    return _gate_world(
        [pod("pod-a", "m1", (0,), ip="10.0.0.1"), pod("pod-b", "m1", (2,), ip="10.0.0.2")],
        [_desired("m1/node-a/0", "m1", (0,), "awake"), _desired("m1/node-a/2", "m1", (2,), "awake")],
        **kwargs,
    )


# ------------------------------------------------------------------ P1
def test_a_cold_start_holding_the_writer_lock_admits_its_own_gated_pod():
    world = _cold_start_world()

    result = world.service.put_model_target("m1", wake_replicas=2)

    created = world.runtime.created
    assert len(created) == 1
    new_pod = world.runtime.pending[created[0]][0]
    assert world.runtime.admitted[new_pod]["operation_id"] == "put_model_target-1"
    assert 200 in world.runtime.gate_answers
    assert world.coordinator.kinds == ["put_model_target"]  # the gate never took the lock
    assert [a["action"] for a in result["actions"]] == ["create"]
    new_id = next(b for b in world.store.load().bindings if b.serve_id == new_pod).binding_id
    assert world.desired()[new_id][0] == "awake"
    assert world.leases.held[new_id][1] == "awake"
    assert world.runtime.cleared == [new_pod]  # the creator converged its own Pod
    # the operation left phase starting_binding again (and noted the UID)
    phases = [(phase, details.get("pod_uid")) for _kind, phase, details in world.coordinator.phases]
    assert ("starting_binding", None) in phases
    assert ("starting_binding", world.runtime.pending[created[0]][1]) in phases
    assert phases[-1][0] == "executing"


def test_a_defrag_migration_admits_its_destination_pod_under_the_lock():
    world = _defrag_world()

    result = world.service.defrag(tp_size=2, force=True)

    assert result["migrations"][0]["to_slot"]["gpu_ids"] == [1]
    created = world.runtime.created
    assert created == ["m1-node-a-gpu-1"]
    new_pod = world.runtime.pending[created[0]][0]
    assert world.runtime.admitted[new_pod]["operation_id"] == "defrag-1"
    assert "m1-node-a-gpu-2" in world.runtime.deleted  # the source Deployment
    assert world.desired()["m1/node-a/1"][0] == "awake"
    assert world.desired()["m1/node-a/2"][2] == "absent"
    assert world.coordinator.kinds == ["defrag"]


def test_without_pre_authorization_the_gate_and_the_creator_deadlock(monkeypatch):
    """The pre-existing failure mode (main 1f307163): the admission needs the
    writer lock the creator holds until the Pod is ready."""
    world = _cold_start_world(gate_polls=5)
    monkeypatch.setattr(world.service, "_starting_binding", _no_phase)

    with pytest.raises(TimeoutError, match="init gate refused"):
        world.service.put_model_target("m1", wake_replicas=2)

    assert "startup_admit" in world.coordinator.busy
    # the failed start cleaned up after itself: no Deployment left to 400-loop
    assert world.runtime.deleted == world.runtime.created
    new_id = "m1/node-a/1"
    assert world.desired().get(new_id, (None, None, "absent"))[2] == "absent"
    assert new_id not in world.leases.held


@contextmanager
def _no_phase(_planned):
    yield lambda _uid: None


def test_a_pod_of_another_binding_is_not_admitted_during_a_start():
    world = _cold_start_world()
    other = StartupPodRecord(
        name="tp2-x", uid="uid-x", model="tp2", node="node-a", gpu_ids=(2, 3),
        annotations={}, labels={}, pod_ip=None, phase="Pending", ready=False,
    )
    world.runtime.get_startup_pod = lambda name: other
    planned = Binding("startup", "m1", Slot("node-a", (1,)), awake=False)
    with world.coordinator.operation("put_model_target") as op:
        world.leases.acquire(planned, phase="starting")
        op.advance("starting_binding", details={"binding_id": planned.binding_id})
        with pytest.raises(OperationBusy):
            world.service.admit_startup(pod_name="tp2-x", pod_uid="uid-x")
    assert world.runtime.admitted == {}


def test_only_the_noted_pod_uid_is_admitted():
    world = _cold_start_world()
    record = StartupPodRecord(
        name="m1-new", uid="uid-replacement", model="m1", node="node-a", gpu_ids=(1,),
        annotations={}, labels={}, pod_ip=None, phase="Pending", ready=False,
    )
    world.runtime.get_startup_pod = lambda name: record
    planned = Binding("startup", "m1", Slot("node-a", (1,)), awake=False)
    with world.coordinator.operation("put_model_target") as op:
        world.leases.acquire(planned, phase="starting")
        op.advance("starting_binding", details={"binding_id": planned.binding_id, "pod_uid": "uid-first"})
        with pytest.raises(OperationBusy):
            world.service.admit_startup(pod_name="m1-new", pod_uid="uid-replacement")
        op.advance("starting_binding", details={"binding_id": planned.binding_id})
        assert world.service.admit_startup(pod_name="m1-new", pod_uid="uid-replacement")["pre_authorized"]


def test_a_pre_authorized_admission_still_checks_pressure_and_the_lease():
    world = _cold_start_world()
    record = StartupPodRecord(
        name="m1-new", uid="u", model="m1", node="node-a", gpu_ids=(1,),
        annotations={}, labels={}, pod_ip=None, phase="Pending", ready=False,
    )
    world.runtime.get_startup_pod = lambda name: record
    planned = Binding("startup", "m1", Slot("node-a", (1,)), awake=False)

    class Pressure(FakeSafety):
        def assert_no_pressure(self):
            raise NodePressureActive("node-a under memory pressure")

    with world.coordinator.operation("defrag") as op:
        op.advance("starting_binding", details={"binding_id": planned.binding_id})
        # no starting lease: not pre-authorized
        with pytest.raises(OperationBusy):
            world.service.admit_startup(pod_name="m1-new", pod_uid="u")
        world.leases.acquire(planned, phase="starting")
        world.service._safety_gate = Pressure()
        with pytest.raises(NodePressureActive):
            world.service.admit_startup(pod_name="m1-new", pod_uid="u")
    assert world.runtime.admitted == {}


def test_a_failed_cold_start_deletes_its_deployment_and_releases_the_lease():
    world = _cold_start_world()
    world.vllm.wait_until_ready = lambda pod_ip, *, port=None: SimpleNamespace(success=False, message="engine crashed")

    with pytest.raises(ValueError, match="engine crashed"):
        world.service.put_model_target("m1", wake_replicas=2)

    assert world.runtime.created and world.runtime.deleted == world.runtime.created
    assert "m1/node-a/1" not in world.leases.held
    assert world.desired().get("m1/node-a/1", (None, None, "absent"))[2] == "absent"
    assert world.service.reap_rejected_deployments() == []


def test_the_supervisor_reaps_a_deployment_whose_cleanup_failed():
    world = _cold_start_world()
    world.vllm.wait_until_ready = lambda pod_ip, *, port=None: SimpleNamespace(success=False, message="engine crashed")
    world.runtime.fail_delete = 1
    with pytest.raises(ValueError):
        world.service.put_model_target("m1", wake_replicas=2)
    assert "m1-node-a-gpu-1" in [d.name for d in world.runtime.deployments]
    # its Pod never got Running in this scenario: drop the snapshot the fake made
    world.runtime.snapshots = {k: v for k, v in world.runtime.snapshots.items() if k == "pod-a"}

    assert world.service.reap_rejected_deployments() == ["m1-node-a-gpu-1"]
    assert world.service.reap_rejected_deployments() == []
    # a resident binding's Deployment is never reaped
    assert "m1-node-a-gpu-0" in [d.name for d in world.runtime.deployments]


def test_the_reaper_skips_while_a_writer_holds_the_lock():
    world = _cold_start_world()
    world.runtime.deployments.append(ModelDeploymentRecord("m1-node-a-gpu-1", "m1", "node-a", (1,), 1))
    absent = replace(_desired("m1/node-a/1", "m1", (1,), "sleeping"), lifecycle="absent")
    with fence(world.redis):
        snapshot = world.fleet.load_desired()
        world.fleet.save_desired(snapshot.bindings + [absent], expected_version=snapshot.version)
    with world.coordinator.operation("defrag"):
        with pytest.raises(OperationBusy):
            world.service.reap_rejected_deployments()
    assert world.service.reap_rejected_deployments() == ["m1-node-a-gpu-1"]


# ------------------------------------------------------------------ P2-2
def _hidden_world():
    world = World(
        [pod("pod-a", "m1", (0,), ip="10.0.0.1"), pod("pod-b", "m1", (1,), ip="10.0.0.2", state="hidden")],
        [_desired("m1/node-a/0", "m1", (0,), "awake"), _desired("m1/node-a/1", "m1", (1,), "awake", hidden=True)],
    )
    return world


def test_unhide_is_refused_for_a_pod_whose_sleep_is_unconfirmed():
    world = _hidden_world()
    world.primitive.journal.begin("pod-b", {"phase": "sleep_unconfirmed", "binding_id": "m1/node-a/1"})

    with pytest.raises(RetryLater, match="sleep_unconfirmed"):
        world.service.put_model_routable("m1", hidden_pods=[])

    assert world.state("pod-b") == "hidden"
    assert world.desired()["m1/node-a/1"][1] is True  # desired untouched


@pytest.mark.parametrize("physical", [True, None])
def test_unhide_is_refused_unless_the_pod_is_confirmed_awake(physical):
    world = _hidden_world()
    world.vllm.physical_override["10.0.0.2"] = physical

    with pytest.raises(RetryLater, match="not confirmed awake"):
        world.service.put_model_routable("m1", hidden_pods=[])
    assert world.state("pod-b") == "hidden"

    client = TestClient(create_app(world.service))
    assert client.put("/v2/models/m1/routable", json={"hidden_pods": []}).status_code == 409


def test_unhide_of_a_pod_confirmed_awake_goes_through():
    world = _hidden_world()

    result = world.service.put_model_routable("m1", hidden_pods=[])

    assert result["actions"] == [{"action": "unhide", "serve_id": "pod-b"}]
    assert world.state("pod-b") == "awake"


def test_keeping_an_unconfirmed_pod_hidden_is_allowed():
    world = _hidden_world()
    world.primitive.journal.begin("pod-b", {"phase": "sleep_unconfirmed", "binding_id": "m1/node-a/1"})

    result = world.service.put_model_routable("m1", hidden_pods=["pod-b"])

    assert result["actions"] == []


# ------------------------------------------------------------------ P3
def test_startup_admission_during_shutdown_is_a_retriable_503():
    world = _cold_start_world()
    world.service.shutdown(timeout_s=0.1)
    client = TestClient(create_app(world.service))

    response = client.post("/v2/startup/admit", json={"pod_name": "m1-new", "pod_uid": "u"})

    assert response.status_code == 503


def test_a_cancelled_admission_job_is_a_retriable_503():
    world = _cold_start_world()
    from concurrent.futures import Future

    cancelled = Future()
    cancelled.cancel()
    world.service._admission_jobs[("m1-new", "u")] = (cancelled, time.monotonic())
    client = TestClient(create_app(world.service))

    response = client.post("/v2/startup/admit", json={"pod_name": "m1-new", "pod_uid": "u"})

    assert response.status_code == 503
    assert ("m1-new", "u") not in world.service._admission_jobs

"""B6: defrag in the full static layout relocates by power only (the
destination binding and its Deployment already exist). B7: a Deployment of a
desired-resident registry binding that was deleted is created again (targeted
repair for sleeping residents, full fleet repair otherwise)."""

from __future__ import annotations

import pytest

from tre_sm.allocator.slots import Binding, Slot
from tre_sm.api.v2 import DefragUnavailable
from tre_sm.ops.k8s_ops import ModelDeploymentRecord
from tre_sm.state.fleet_repair import FleetRepairExecutor
from tre_sm.state.supervisor import FleetSupervisor

from sm_test_fakes import Result, pod
from test_fleet_repair import (
    FakeOperation,
    FakeRuntime as RepairRuntime,
    FakeSafety as RepairSafety,
    FakeVllm as RepairVllm,
    _binding_id,
    _deployment,
    _primitive,
    _snapshot,
)
from test_review2_sleep import World, _desired
from test_review4_sm import _gate_world
from test_supervisor import FakeService


# ------------------------------------------------------------------ B6
def _full_layout_world():
    """Every GPU hosts an m1 binding (one Deployment each), two awake on
    GPUs 0 and 2 - no free TP2 pair - plus the sleeping tp2 bindings."""
    pods = [
        pod("pod-a", "m1", (0,), ip="10.0.0.1"),
        pod("pod-c", "m1", (1,), ip="10.0.0.3", state="sleeping"),
        pod("pod-b", "m1", (2,), ip="10.0.0.2"),
        pod("pod-d", "m1", (3,), ip="10.0.0.4", state="sleeping"),
        pod("pod-t", "tp2", (0, 1), ip="10.0.0.5", state="sleeping"),
    ]
    desired = [
        _desired("m1/node-a/0", "m1", (0,), "awake"),
        _desired("m1/node-a/1", "m1", (1,), "sleeping"),
        _desired("m1/node-a/2", "m1", (2,), "awake"),
        _desired("m1/node-a/3", "m1", (3,), "sleeping"),
        _desired("tp2/node-a/0,1", "tp2", (0, 1), "sleeping"),
    ]
    world = World(pods, desired)
    runtime = world.runtime
    runtime.deployments = [
        ModelDeploymentRecord(f"{p.model}-{p.name}", p.model, p.node, tuple(int(g) for g in p.annotations["tre.aibrix.io/gpu-ids"].split(",")), 1)
        for p in pods
    ]
    runtime.deployment_calls = []

    def create_model_deployment(model, slot):
        runtime.deployment_calls.append(("create", model, tuple(slot.gpu_ids)))
        raise RuntimeError("AlreadyExists")

    def delete_model_deployment(binding):
        runtime.deployment_calls.append(("delete", binding.serve_id))

    runtime.create_model_deployment = create_model_deployment
    runtime.delete_model_deployment = delete_model_deployment
    runtime.wait_pod_deleted = lambda serve_id: runtime.deployment_calls.append(("wait_deleted", serve_id))
    runtime.wait_pod_ready = lambda name: runtime.deployment_calls.append(("wait_ready", name))
    return world


def test_full_layout_defrag_sleeps_the_source_and_wakes_the_existing_destination():
    world = _full_layout_world()
    before = {d.binding_id: d for d in world.fleet.load_desired().bindings}

    result = world.service.defrag(tp_size=2, force=True)

    assert result["migrations"] == [
        {
            "serve_id": "pod-b",
            "from_slot": {"node": "node-a", "gpu_ids": [2]},
            "to_slot": {"node": "node-a", "gpu_ids": [1]},
        }
    ]
    # Make-before-break (replica floor, 2026-09-29): the destination wakes first.
    assert result["actions"] == [
        {"action": "wake", "serve_id": "pod-c"},
        {"action": "hide", "serve_id": "pod-b"},
        {"action": "sleep", "serve_id": "pod-b"},
    ]
    assert world.runtime.deployment_calls == []  # no delete, no create
    assert world.vllm.sleeping["10.0.0.2"] is True
    assert world.vllm.sleeping["10.0.0.3"] is False
    desired = world.desired()
    assert desired["m1/node-a/2"] == ("sleeping", False, "resident")  # stays resident
    assert desired["m1/node-a/1"] == ("awake", False, "resident")
    # the layout does not shrink: every record is still there and resident
    assert set(desired) == set(before)
    assert all(lifecycle == "resident" for _p, _h, lifecycle in desired.values())
    stored = {b.serve_id: b for b in world.store.load().bindings}
    assert stored["pod-b"].slot.gpu_ids == (2,) and stored["pod-b"].awake is False
    assert stored["pod-c"].slot.gpu_ids == (1,) and stored["pod-c"].awake is True
    assert len(stored) == 5
    assert world.state("pod-c") == "awake" and world.state("pod-b") == "sleeping"
    # the TP2 pair (2, 3) is free now; a second defrag has nothing to do
    assert world.service.defrag(tp_size=2, force=True)["actions"] == []


def test_full_layout_defrag_rolls_back_when_the_destination_wake_fails():
    world = _full_layout_world()
    original_wake = world.vllm.wake_up

    def wake_up(pod_ip, *, port=None):
        if pod_ip == "10.0.0.3":
            world.vllm._log("wake_up", pod_ip)
            return Result(False, "injected wake failure")
        return original_wake(pod_ip, port=port)

    world.vllm.wake_up = wake_up
    before_store = world.store.load().bindings

    with pytest.raises(ValueError, match="injected wake failure"):
        world.service.defrag(tp_size=2, force=True)

    assert world.runtime.deployment_calls == []
    assert world.vllm.sleeping["10.0.0.2"] is False  # the source was woken again
    assert world.vllm.sleeping["10.0.0.3"] is True
    desired = world.desired()
    assert desired["m1/node-a/2"] == ("awake", False, "resident")
    assert desired["m1/node-a/1"] == ("sleeping", False, "resident")
    assert world.store.load().bindings == before_store


def test_sparse_defrag_refuses_a_destination_deployment_without_a_binding():
    world = _gate_world(
        [pod("pod-a", "m1", (0,), ip="10.0.0.1"), pod("pod-b", "m1", (2,), ip="10.0.0.2")],
        [_desired("m1/node-a/0", "m1", (0,), "awake"), _desired("m1/node-a/2", "m1", (2,), "awake")],
    )
    world.runtime.deployments.append(ModelDeploymentRecord("m1-node-a-gpu-1", "m1", "node-a", (1,), 1))

    with pytest.raises(DefragUnavailable, match="destination_deployment_without_binding"):
        world.service.defrag(tp_size=2, force=True)

    assert world.runtime.deleted == [] and world.runtime.created == []
    assert world.vllm.sleeping["10.0.0.2"] is False
    assert world.desired()["m1/node-a/2"] == ("awake", False, "resident")


# ------------------------------------------------------------------ B7 executor
class CreatingRuntime(RepairRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.created = []
        self.waited = []

    def wait_deployment_pods_deleted(self, deployment_name):
        # Like K8sOps: by the ``app`` label, so it works without the Deployment.
        self.waited.append(deployment_name)
        assert not any(name.startswith(f"{deployment_name}-") for name in self.snapshots)

    def create_model_deployment(self, model, slot):
        name = f"{model}-{slot.node}-gpu-{'-'.join(str(g) for g in slot.gpu_ids)}"
        self.created.append((model, slot.node, tuple(slot.gpu_ids)))
        record = _deployment(name, model, slot.node, slot.gpu_ids)
        for snapshot in self.snapshots.values():
            if snapshot.node == slot.node and set(slot.gpu_ids) & set(
                int(g) for g in snapshot.annotations["tre.aibrix.io/gpu-ids"].split(",")
            ):
                assert self.vllm.sleeping[snapshot.pod_ip] is True
        self.deployments.append(record)
        pod_ip = f"10.0.0.{len(self.snapshots) + 10}"
        self.snapshots[f"{name}-pod"] = _snapshot(record, pod_ip=pod_ip, state="hidden")
        self.vllm.sleeping[pod_ip] = False
        return name


def _set_power(runtime, vllm):
    def set_power(binding_id, awake):
        snapshot = next(s for s in runtime.snapshots.values() if _binding_id(s) == binding_id)
        vllm.sleeping[snapshot.pod_ip] = not awake
        return {"binding_id": binding_id, "awake": awake}

    return set_power


def test_fleet_repair_recreates_a_missing_registry_deployment():
    first = _deployment("m1-node-a-gpu-0", "m1", "node-a", (0,))
    vllm = RepairVllm({"10.0.0.1": False})
    runtime = CreatingRuntime([first], [_snapshot(first, pod_ip="10.0.0.1", state="awake")], vllm)
    operation = FakeOperation()
    executor = FleetRepairExecutor(
        runtime_ops=runtime, vllm_ops=vllm, safety_gate=RepairSafety(),
        sleep_binding=_primitive(runtime, vllm), poll_interval_s=0, sleep=lambda _s: None,
    )
    missing = Binding("m2/node-a/0", "m2", Slot("node-a", (0,)), awake=False)

    executor.run(
        operation,
        awake_binding_ids=[first.binding_id],
        reconcile=lambda _strict: {},
        set_binding_power=_set_power(runtime, vllm),
        audit=lambda: {"healthy": True},
        # an entry that already has a Deployment is ignored
        recreate_bindings=lambda: [missing, Binding("m1/node-a/0", "m1", Slot("node-a", (0,)), awake=False)],
    )

    assert runtime.created == [("m2", "node-a", (0,))]
    assert runtime.scale_calls == []  # nothing to scale: the created one starts by itself
    assert runtime.waited == ["m2-node-a-gpu-0"]  # leftover Pods waited for by name
    rebuilt = next(s for s in runtime.snapshots.values() if _binding_id(s) == "m2/node-a/0")
    assert vllm.sleeping[rebuilt.pod_ip] is True  # started, then put to sleep
    assert vllm.sleeping["10.0.0.1"] is False  # the awake target woke again
    starting = [d for phase, d in operation.phases if phase == "starting_binding"]
    assert [d["binding_id"] for d in starting] == ["m2/node-a/0"]  # pre-authorized
    verified = next(d for phase, d in operation.phases if phase == "verified")
    assert verified["recreated_binding_ids"] == ["m2/node-a/0"]


def test_fleet_repair_can_recreate_when_every_deployment_is_gone_and_wake_it():
    vllm = RepairVllm({})
    runtime = CreatingRuntime([], [], vllm)
    executor = FleetRepairExecutor(
        runtime_ops=runtime, vllm_ops=vllm, safety_gate=RepairSafety(),
        sleep_binding=_primitive(runtime, vllm), poll_interval_s=0, sleep=lambda _s: None,
    )

    executor.run(
        FakeOperation(),
        awake_binding_ids=["m1/node-a/0"],  # a desired-awake binding whose Deployment is gone
        reconcile=lambda _strict: {},
        set_binding_power=_set_power(runtime, vllm),
        audit=lambda: {"healthy": True},
        recreate_bindings=lambda: [Binding("m1/node-a/0", "m1", Slot("node-a", (0,)), awake=False)],
    )

    assert runtime.created == [("m1", "node-a", (0,))]
    rebuilt = next(iter(runtime.snapshots.values()))
    assert vllm.sleeping[rebuilt.pod_ip] is False


# ------------------------------------------------------------------ B7 service
def _missing_world():
    """m1/node-a/1 (registry binding, desired sleeping) lost its Deployment."""
    world = _gate_world(
        [pod("pod-a", "m1", (0,), ip="10.0.0.1")],
        [
            _desired("m1/node-a/0", "m1", (0,), "awake"),
            _desired("m1/node-a/1", "m1", (1,), "sleeping"),
        ],
    )
    return world


def test_targeted_repair_recreates_only_missing_sleeping_registry_deployments():
    world = _missing_world()

    assert [b.binding_id for b in world.service._missing_registry_deployments()] == ["m1/node-a/1"]
    result = world.service.repair_missing_deployments(["m1/node-a/1"])

    assert result == {"binding_ids": ["m1/node-a/1"], "created": ["m1-node-a-gpu-1"]}
    assert world.runtime.created == ["m1-node-a-gpu-1"]
    assert world.runtime.deleted == []
    assert world.vllm.sleeping["10.0.0.1"] is False  # the fleet is not quarantined
    assert world.coordinator.kinds == ["deployment_repair"]
    # nothing is missing any more: a second pass is a no-op / not eligible
    assert world.service._missing_registry_deployments() == []
    assert world.service.repair_missing_deployments(["m1/node-a/1"]) is None


def test_targeted_repair_declines_awake_or_non_registry_bindings():
    world = _gate_world(
        [pod("pod-b", "m1", (1,), ip="10.0.0.2", state="sleeping")],
        [
            _desired("m1/node-a/0", "m1", (0,), "awake"),  # awake: needs the full repair
            _desired("m1/node-a/1", "m1", (1,), "sleeping"),
            _desired("m1/node-a/3", "m1", (3,), "sleeping"),  # not in the registry
        ],
    )

    assert world.service.repair_missing_deployments(["m1/node-a/0"]) is None
    assert world.service.repair_missing_deployments(["m1/node-a/3"]) is None
    assert world.runtime.created == []
    # the full repair is handed exactly the registry bindings to recreate
    assert [b.binding_id for b in world.service._missing_registry_deployments()] == ["m1/node-a/0"]


# ------------------------------------------------------------------ B7 supervisor
class TargetedService(FakeService):
    def __init__(self, answer):
        super().__init__()
        self.answer = answer
        self.targeted = []

    def repair_missing_deployments(self, binding_ids):
        self.targeted.append(list(binding_ids))
        return self.answer


def _drive(service, drift):
    service.drift = drift
    supervisor = FleetSupervisor(service, drift_observations_required=2)
    supervisor.run_once()
    supervisor.run_once()
    return supervisor


def test_supervisor_repairs_deployment_missing_only_drift_without_a_fleet_repair():
    service = TargetedService({"created": ["x"]})
    _drive(service, [{"code": "deployment_missing", "binding_id": "m1/node-a/1"}])

    assert service.targeted == [["m1/node-a/1"]]
    assert service.repairs == [] and service.observe_handoffs == 0


def test_supervisor_falls_back_to_the_fleet_repair_when_not_eligible():
    declined = TargetedService(None)
    _drive(declined, [{"code": "deployment_missing", "binding_id": "m1/node-a/0"}])
    assert declined.targeted == [["m1/node-a/0"]]
    # 2026-09-28: the supervisor no longer writes the controller mode
    assert declined.repairs == [True] and declined.observe_handoffs == 0

    mixed = TargetedService({"created": ["x"]})
    _drive(
        mixed,
        [
            {"code": "deployment_missing", "binding_id": "m1/node-a/1"},
            {"code": "pod_not_ready", "binding_id": "m1/node-a/0"},
        ],
    )
    assert mixed.targeted == []
    assert mixed.repairs == [True]


# ------------------------------------------- B6 x B3 (headroom) x B2 (observed)
class _StaticTruth:
    """gpu-truth without refresh support: fixed used MiB per GPU uuid."""

    def __init__(self, used):
        self.used = dict(used)

    def used_mib(self, *, node, gpu_id, gpu_uuid):
        return self.used.get(gpu_uuid)

    def node_truth(self, *, node):
        from tre_sm.gpu_truth import NodeGpuTruth

        return NodeGpuTruth(
            node=node,
            used_by_uuid=dict(self.used),
            total_by_uuid={uuid: 40960 for uuid in self.used},
        )


def _truth(gpu1_used):
    return _StaticTruth({"GPU-0": 30000, "GPU-1": gpu1_used, "GPU-2": 30000, "GPU-3": 500})


def test_full_layout_defrag_checks_the_destination_headroom_before_sleeping_the_source():
    from tre_sm.api.v2 import WakeConflict

    world = _full_layout_world()
    world.service._gpu_truth = _truth(33000)  # an awake resident / leak on GPU 1

    with pytest.raises(WakeConflict, match="insufficient wake headroom"):
        world.service.defrag(tp_size=2, force=True)

    assert world.vllm.sleeping["10.0.0.2"] is False  # the source was never slept
    assert not any(call[0] == "sleep" for call in world.vllm.calls)


def test_full_layout_defrag_refreshes_the_observed_state_of_both_bindings():
    from tre_sm.server import K8sPodClientFromOps

    world = _full_layout_world()
    world.service._gpu_truth = _truth(900)
    world.service._k8s_client = K8sPodClientFromOps(world.service._registry.topology(), world.runtime)
    world.service.reconcile()

    world.service.defrag(tp_size=2, force=True)

    observed = {item.binding_id: item.physical_power for item in world.fleet.load_observed().bindings}
    assert observed["m1/node-a/2"] == "sleeping"  # no reconcile needed (B2)
    assert observed["m1/node-a/1"] == "awake"
    codes = {issue["code"] for issue in world.service.audit()["issues"]}
    assert not codes & {"desired_power_mismatch", "desired_hidden_mismatch"}

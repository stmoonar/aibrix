"""B11: no startup admission (and no ``starting`` GPU lease) for a Pod whose
Deployment is gone.

Live: a cold start failed (the test deleted its Pod during the vLLM load),
the SM deleted the Deployment and rolled desired state back. The ReplicaSet had
already created a replacement Pod; ~1 s after the failed operation released the
writer lock that Pod passed the normal ``startup_admit`` path and took a
``starting`` lease on the GPUs of a Deployment that no longer existed - an
orphan lease that outlived the Pod. The admission now checks the Pod's owner
chain (Pod -> ReplicaSet -> Deployment, all live) before and under the writer
lock, and the supervisor releases ``starting`` leases whose binding has no Pod.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tre_sm.allocator.slots import Binding, Slot
from tre_sm.api.v2 import RetryLater
from tre_sm.ops.k8s_ops import MANAGED_LABEL, MODEL_LABEL, K8sOps, ModelDeploymentRecord
from tre_sm.allocator.topology import GPU_IDS_ANNOTATION
from tre_sm.state.operations import OperationBusy
from tre_sm.state.supervisor import FleetSupervisor

from sm_test_fakes import fence
from test_review2_sleep import _desired
from test_review4_sm import _deployment_name, _gate_world
from sm_test_fakes import pod


# ------------------------------------------------------------ service level
def _install_owner_check(world):
    """Like K8sOps.startup_owner_problem: the Pod's Deployment must exist."""
    runtime = world.runtime
    runtime.owner_checks = []

    def startup_owner_problem(pod_name, pod_uid):
        runtime.owner_checks.append(pod_name)
        record = runtime.get_startup_pod(pod_name)
        name = _deployment_name(record.model, Slot(record.node, record.gpu_ids))
        if name not in [item.name for item in runtime.deployments]:
            return f"Deployment {name} of Pod {pod_name} is gone"
        return None

    runtime.startup_owner_problem = startup_owner_problem
    return runtime


def _registry_slot_world():
    """m1 awake on GPU 0; m1/node-a/1 is a desired-resident (sleeping) binding
    whose Deployment is missing - so after a rolled-back cold start its desired
    record is still ``resident`` and the old admission path accepted a Pod."""
    world = _gate_world(
        [pod("pod-a", "m1", (0,), ip="10.0.0.1")],
        [
            _desired("m1/node-a/0", "m1", (0,), "awake"),
            _desired("m1/node-a/1", "m1", (1,), "sleeping"),
        ],
    )
    _install_owner_check(world)
    return world


def _fail_cold_start(world):
    world.vllm.wait_until_ready = lambda pod_ip, *, port=None: SimpleNamespace(
        success=False, message="readiness timeout"
    )
    with pytest.raises(ValueError, match="readiness timeout"):
        world.service.put_model_target("m1", wake_replicas=2)
    created = world.runtime.created
    assert created and world.runtime.deleted == created  # rolled back
    # the first Pod is gone (deleted during its vLLM load)
    world.runtime.snapshots = {k: v for k, v in world.runtime.snapshots.items() if k == "pod-a"}
    return created[0]


def _replacement_pod(world, deployment):
    """The ReplicaSet's replacement Pod, created before the Deployment deletion
    cascaded; it now polls the startup gate."""
    _pod_name, _uid, model, slot = world.runtime.pending[deployment]
    world.runtime.pending[f"{deployment}#replacement"] = (
        f"{deployment}-replacement", "uid-replacement", model, slot,
    )
    return f"{deployment}-replacement", "uid-replacement"


def test_b11_the_replacement_pod_of_a_rolled_back_cold_start_is_not_admitted():
    world = _registry_slot_world()
    deployment = _fail_cold_start(world)
    # I1 (2026-10-04): the failed start's lease stays until no Pod of the binding
    # exists (orphan lease reaper) - the replacement Pod below still does.
    assert world.leases.held["m1/node-a/1"][1] == "starting"
    # desired state was rolled back to what it was: still resident
    assert world.desired()["m1/node-a/1"][2] == "resident"
    name, uid = _replacement_pod(world, deployment)
    world.leases.calls.clear()

    with pytest.raises(RetryLater, match="is gone"):
        world.service.admit_startup(pod_name=name, pod_uid=uid)

    assert name not in world.runtime.admitted
    assert world.leases.calls == []  # no (orphan) starting lease


def test_b11_without_the_owner_check_the_replacement_pod_took_an_orphan_lease(monkeypatch):
    """The pre-B11 behaviour (root cause): the normal startup_admit path only
    looked at the desired record, which the rollback left ``resident``."""
    world = _registry_slot_world()
    deployment = _fail_cold_start(world)
    name, uid = _replacement_pod(world, deployment)
    monkeypatch.setattr(world.service, "_assert_startup_owner_live", lambda _pod: None)

    assert world.service.admit_startup(pod_name=name, pod_uid=uid)["status"] == "admitted"
    assert world.leases.held["m1/node-a/1"][1] == "starting"


def test_b11_the_async_gate_gets_a_retriable_error_not_an_admission():
    world = _registry_slot_world()
    deployment = _fail_cold_start(world)
    name, uid = _replacement_pod(world, deployment)
    world.leases.calls.clear()

    with pytest.raises(RetryLater):
        world.service.request_startup_admission(pod_name=name, pod_uid=uid)
    assert name not in world.runtime.admitted
    assert world.leases.calls == []  # no (orphan) starting lease


def test_b11_the_owner_chain_is_checked_again_under_the_writer_lock():
    """The admission queued behind the failing writer: the Deployment still
    existed at the lock-free check and is gone once the lock is acquired."""
    world = _registry_slot_world()
    runtime = world.runtime
    slot = Slot("node-a", (1,))
    name = _deployment_name("m1", slot)
    runtime.deployments.append(ModelDeploymentRecord(name, "m1", "node-a", (1,), 1))
    runtime.pending[name] = (f"{name}-pod", "uid-1", "m1", slot)
    original = world.service._admit_startup_locked

    def delete_then_lock(record):
        runtime.deployments = [item for item in runtime.deployments if item.name != name]
        return original(record)

    world.service._admit_startup_locked = delete_then_lock

    with pytest.raises(RetryLater, match="is gone"):
        world.service.admit_startup(pod_name=f"{name}-pod", pod_uid="uid-1")
    assert runtime.owner_checks == [f"{name}-pod", f"{name}-pod"]
    assert world.leases.calls == []
    assert runtime.admitted == {}


def test_b11_a_legitimate_pod_is_still_admitted_with_its_starting_lease():
    world = _registry_slot_world()
    runtime = world.runtime
    slot = Slot("node-a", (1,))
    name = _deployment_name("m1", slot)
    runtime.deployments.append(ModelDeploymentRecord(name, "m1", "node-a", (1,), 1))
    runtime.pending[name] = (f"{name}-pod", "uid-1", "m1", slot)

    result = world.service.admit_startup(pod_name=f"{name}-pod", pod_uid="uid-1")

    assert result["status"] == "admitted"
    assert world.leases.held["m1/node-a/1"][1] == "starting"
    assert runtime.admitted[f"{name}-pod"]["pod_uid"] == "uid-1"


def test_b11_a_cold_start_still_admits_its_own_pod_with_the_owner_check():
    world = _registry_slot_world()

    world.service.put_model_target("m1", wake_replicas=2)

    new_pod = world.runtime.pending[world.runtime.created[0]][0]
    assert world.runtime.admitted[new_pod]["operation_id"] == "put_model_target-1"
    assert new_pod in world.runtime.owner_checks
    assert world.leases.held["m1/node-a/1"][1] == "awake"


def test_b11_an_unreadable_owner_chain_fails_closed_and_retriable():
    world = _registry_slot_world()
    runtime = world.runtime
    slot = Slot("node-a", (1,))
    runtime.pending["x"] = ("m1-pod", "uid-1", "m1", slot)

    def broken(_name, _uid):
        raise RuntimeError("apiserver unavailable")

    runtime.startup_owner_problem = broken
    with pytest.raises(RetryLater, match="apiserver unavailable"):
        world.service.admit_startup(pod_name="m1-pod", pod_uid="uid-1")
    assert world.leases.calls == []


# ------------------------------------------------------ orphan lease reaper
def _lease_world(live):
    world = _registry_slot_world()
    world.runtime.live_binding_ids = set(live)
    world.runtime.list_live_model_pod_binding_ids = lambda: set(world.runtime.live_binding_ids)
    return world


def _hold(world, binding_id, gpus, phase):
    binding = Binding("x", binding_id.split("/")[0], Slot("node-a", tuple(gpus)), awake=False)
    with fence(world.redis):
        world.leases.acquire(binding, phase=phase)


def test_b11_the_supervisor_releases_a_starting_lease_whose_pod_is_gone():
    world = _lease_world(live={"m1/node-a/0"})
    _hold(world, "m1/node-a/0", (0,), "awake")
    _hold(world, "m1/node-a/1", (1,), "starting")

    assert world.service.reap_orphan_leases() == ["m1/node-a/1"]

    assert "m1/node-a/1" not in world.leases.held
    assert world.leases.held["m1/node-a/0"][1] == "awake"  # its Pod exists: kept
    assert world.service.reap_orphan_leases() == []


def test_b11_a_starting_lease_of_a_pod_still_in_the_gate_or_loading_is_kept():
    world = _lease_world(live={"m1/node-a/0", "m1/node-a/1"})
    _hold(world, "m1/node-a/1", (1,), "starting")

    assert world.service.reap_orphan_leases() == []
    assert world.leases.held["m1/node-a/1"][1] == "starting"


def test_b11_the_lease_reaper_skips_while_a_writer_holds_the_lock():
    """A creator holds its binding's starting lease before its Pod exists."""
    world = _lease_world(live=set())
    _hold(world, "m1/node-a/1", (1,), "starting")

    with world.coordinator.operation("put_model_target"):
        with pytest.raises(OperationBusy):
            world.service.reap_orphan_leases()
    assert "m1/node-a/1" in world.leases.held
    assert world.service.reap_orphan_leases() == ["m1/node-a/1"]


def test_b11_the_lease_reaper_rechecks_the_pods_under_the_lock():
    world = _lease_world(live=set())
    _hold(world, "m1/node-a/1", (1,), "starting")
    calls = []

    def lister():
        calls.append(True)
        # the replacement Pod appeared between the lock-free check and the lock
        return set() if len(calls) == 1 else {"m1/node-a/1"}

    world.runtime.list_live_model_pod_binding_ids = lister
    assert world.service.reap_orphan_leases() == []
    assert "m1/node-a/1" in world.leases.held


def test_b11_the_supervisor_runs_the_lease_reaper_and_tolerates_a_busy_writer():
    from test_supervisor import FakeService

    service = FakeService()
    calls = []

    def reap():
        calls.append(True)
        if len(calls) == 1:
            raise OperationBusy("other-writer:1")
        return []

    service.reap_orphan_leases = reap
    supervisor = FleetSupervisor(service, drift_observations_required=1)
    supervisor.run_once()
    supervisor.run_once()
    assert len(calls) == 2


# ------------------------------------------------------------- K8sOps level
class ApiError(Exception):
    def __init__(self, status):
        super().__init__(str(status))
        self.status = status


def _meta(name, uid, *, owner=None, deleting=False):
    metadata = {"name": name, "uid": uid, "deletionTimestamp": "2026-09-28T00:00:00Z" if deleting else None}
    if owner is not None:
        kind, owner_name, owner_uid = owner
        metadata["ownerReferences"] = [
            {"apiVersion": "apps/v1", "kind": kind, "name": owner_name, "uid": owner_uid, "controller": True}
        ]
    return metadata


class OwnerApi:
    """Pods, ReplicaSets and Deployments by name, as dicts (404 when absent)."""

    DEPLOYMENT = "m1-node-a-gpu-0-1"

    def __init__(self):
        self.pods = {
            "p": {
                "metadata": {
                    **_meta("p", "uid-p", owner=("ReplicaSet", "rs", "uid-rs")),
                    "labels": {MODEL_LABEL: "m1", MANAGED_LABEL: "true"},
                    "annotations": {GPU_IDS_ANNOTATION: "0,1"},
                },
                "spec": {"nodeName": "node-a"},
                "status": {"phase": "Pending"},
            }
        }
        self.replica_sets = {"rs": {"metadata": _meta("rs", "uid-rs", owner=("Deployment", self.DEPLOYMENT, "uid-d"))}}
        self.deployments = {self.DEPLOYMENT: {"metadata": _meta(self.DEPLOYMENT, "uid-d")}}

    def read_namespaced_pod(self, *, name, namespace):
        if name not in self.pods:
            raise ApiError(404)
        return self.pods[name]

    def list_namespaced_pod(self, *, namespace, label_selector=None):
        return list(self.pods.values())

    def read_namespaced_replica_set(self, *, name, namespace):
        if name not in self.replica_sets:
            raise ApiError(404)
        return self.replica_sets[name]

    def read_namespaced_deployment(self, *, name, namespace):
        if name not in self.deployments:
            raise ApiError(404)
        return self.deployments[name]


def _ops(api):
    return K8sOps(api=api, namespace="default")


def test_k8s_owner_chain_live_is_admissible():
    assert _ops(OwnerApi()).startup_owner_problem("p", "uid-p") is None


def test_k8s_owner_chain_deployment_gone_is_refused():
    api = OwnerApi()
    api.deployments.clear()
    assert "is gone" in _ops(api).startup_owner_problem("p", "uid-p")


def test_k8s_owner_chain_deployment_being_deleted_is_refused():
    api = OwnerApi()
    api.deployments[api.DEPLOYMENT]["metadata"]["deletionTimestamp"] = "2026-09-28T00:00:00Z"
    assert "being deleted" in _ops(api).startup_owner_problem("p", "uid-p")


def test_k8s_owner_chain_deployment_created_again_is_refused():
    api = OwnerApi()
    api.deployments[api.DEPLOYMENT]["metadata"]["uid"] = "uid-d2"
    assert "created again" in _ops(api).startup_owner_problem("p", "uid-p")


def test_k8s_owner_chain_replica_set_gone_or_deleting_is_refused():
    api = OwnerApi()
    api.replica_sets["rs"]["metadata"]["deletionTimestamp"] = "2026-09-28T00:00:00Z"
    assert "being deleted" in _ops(api).startup_owner_problem("p", "uid-p")
    api.replica_sets.clear()
    assert "is gone" in _ops(api).startup_owner_problem("p", "uid-p")


def test_k8s_owner_chain_pod_deleting_ownerless_or_of_another_binding_is_refused():
    api = OwnerApi()
    api.pods["p"]["metadata"]["deletionTimestamp"] = "2026-09-28T00:00:00Z"
    assert "being deleted" in _ops(api).startup_owner_problem("p", "uid-p")
    api = OwnerApi()
    assert "no longer UID" in _ops(api).startup_owner_problem("p", "uid-other")
    api.pods["p"]["metadata"].pop("ownerReferences")
    assert "no owning ReplicaSet" in _ops(api).startup_owner_problem("p", "uid-p")
    api = OwnerApi()
    api.pods["p"]["metadata"]["annotations"][GPU_IDS_ANNOTATION] = "2,3"
    assert "not m1-node-a-gpu-2-3" in _ops(api).startup_owner_problem("p", "uid-p")


def test_k8s_owner_chain_api_errors_propagate():
    api = OwnerApi()

    def broken(*, name, namespace):
        raise ApiError(500)

    api.read_namespaced_deployment = broken
    with pytest.raises(ApiError):
        _ops(api).startup_owner_problem("p", "uid-p")


def test_k8s_live_pod_binding_ids_include_gated_and_terminating_pods_only():
    api = OwnerApi()
    base = api.pods["p"]

    def variant(name, gpus, phase, deleting=False):
        item = {
            "metadata": {
                **base["metadata"], "name": name, "uid": f"uid-{name}",
                "annotations": {GPU_IDS_ANNOTATION: gpus},
                "deletionTimestamp": "2026-09-28T00:00:00Z" if deleting else None,
            },
            "spec": base["spec"],
            "status": {"phase": phase},
        }
        return item

    api.pods = {
        "gated": variant("gated", "0", "Pending"),
        "running": variant("running", "1", "Running"),
        "terminating": variant("terminating", "2", "Running", deleting=True),
        "failed": variant("failed", "3", "Failed"),
        "anonymous": {"metadata": {"name": "anonymous", "labels": {}}, "spec": {}, "status": {"phase": "Running"}},
    }
    assert _ops(api).list_live_model_pod_binding_ids() == {
        "m1/node-a/0", "m1/node-a/1", "m1/node-a/2",
    }
    # I1 (2026-10-04): the same Pod objects by UID (a journaled wake's Pod).
    assert {"uid-gated", "uid-running", "uid-terminating"} <= _ops(api).list_live_model_pod_uids()
    assert "uid-failed" not in _ops(api).list_live_model_pod_uids()

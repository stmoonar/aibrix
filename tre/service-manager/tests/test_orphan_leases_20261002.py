"""Orphan GPU leases (2026-10-02, seen live): an ``awake`` lease never expires, so
one whose binding has no Pod any more - its Deployment deleted and recreated,
its binding dropped by ``reconcile drop_missing`` - refused every start on its
GPUs (409 lease_conflict naming a Pod that is gone). The supervisor's lease
reaper now releases every lease (starting or awake) without a Pod, and
``drop_missing`` releases the leases of the bindings it drops; an unreadable
Pod list releases nothing."""

import json
import logging

from tre_sm.allocator.topology import GPU_IDS_ANNOTATION
from tre_sm.ops.k8s_ops import MANAGED_LABEL, MODEL_LABEL, K8sOps
from tre_sm.server import K8sPodClientFromOps
from tre_sm.state.supervisor import FleetSupervisor

from sm_test_fakes import binding_of
from test_wholelock_20261002 import World


def _world():
    world = World()
    world.runtime.list_live_model_pod_binding_ids = lambda: {
        binding_of(snapshot).binding_id for snapshot in world.runtime.snapshots.values()
    }
    return world


def _forget(world, name, *, pod=True, store=True):
    if pod:
        world.runtime.snapshots.pop(name)
    if store:
        snapshot = world.store.load()
        world.store.save([b for b in snapshot.bindings if b.serve_id != name], expected_version=snapshot.version)


def test_an_awake_lease_whose_binding_and_pod_are_gone_is_released(caplog):
    world = _world()
    assert world.lease("d-0") == "awake"
    _forget(world, "d-0")

    with caplog.at_level(logging.WARNING, logger="tre_sm.api.v2"):
        assert world.service.reap_orphan_leases() == ["d/node-a/0"]

    assert not [lease for lease in world.leases.load() if lease.binding_id == "d/node-a/0"]
    events = [json.loads(r.getMessage()) for r in caplog.records if "orphan_awake_lease_released" in r.getMessage()]
    assert events and events[0]["binding_id"] == "d/node-a/0" and events[0]["in_store"] is False


def test_an_awake_lease_is_kept_while_a_pod_of_the_binding_exists():
    world = _world()
    _forget(world, "d-0", pod=False)  # dropped from the store only

    assert world.service.reap_orphan_leases() == []
    assert world.lease("d-1") == "awake"
    assert {lease.binding_id for lease in world.leases.load()} == {"d/node-a/0", "d/node-a/1"}


def test_nothing_is_released_when_the_pod_list_cannot_be_read():
    world = _world()
    _forget(world, "d-0")
    world.runtime.list_live_model_pod_binding_ids = lambda: (_ for _ in ()).throw(ConnectionError("apiserver"))

    assert world.service.reap_orphan_leases() == []
    assert "d/node-a/0" in {lease.binding_id for lease in world.leases.load()}


def test_reconcile_drop_missing_releases_the_leases_of_the_bindings_it_drops():
    world = _world()
    world.service._k8s_client = K8sPodClientFromOps(world.registry.topology(), world.runtime)
    _forget(world, "d-0", store=False)  # its Pod is gone, the store still has it

    result = world.service.reconcile(drop_missing=True)

    assert result["released_leases"] == ["d/node-a/0"]
    assert {lease.binding_id for lease in world.leases.load()} == {"d/node-a/1"}


# ------------------------------------------------ fail closed on the real Pod list (2026-10-06)


class _PodListApi:
    """The CoreV1 Pod LIST the k8s ops read (dict Pods)."""

    def __init__(self, pods):
        self.pods = pods

    def list_namespaced_pod(self, *, namespace, label_selector=None, **_kwargs):
        return list(self.pods)


def _k8s_pod(name, model, gpus, *, gpu_annotation=True, terminating=False):
    metadata = {
        "name": name, "uid": f"uid-{name}",
        "labels": {MODEL_LABEL: model, MANAGED_LABEL: "true"},
        "annotations": {GPU_IDS_ANNOTATION: ",".join(str(g) for g in gpus)} if gpu_annotation else {},
    }
    if terminating:
        metadata["deletionTimestamp"] = "2026-10-06T00:00:00Z"
    return {"metadata": metadata, "spec": {"nodeName": "node-a"}, "status": {"phase": "Running"}}


def _k8s_world(pods):
    world = World()
    world.runtime.list_live_model_pod_binding_ids = K8sOps(
        api=_PodListApi(pods), namespace="default"
    ).list_live_model_pod_binding_ids
    return world


def test_a_managed_pod_with_an_unreadable_binding_releases_nothing():
    # d-0's Pod is gone, but another managed Pod has no GPU annotation: it may be
    # d-0's (its binding cannot be read), so the pass must not conclude anything.
    world = _k8s_world([_k8s_pod("d-1", "d", (1,)), _k8s_pod("d-0-new", "d", (0,), gpu_annotation=False)])

    assert world.service.reap_orphan_leases() == []
    assert {lease.binding_id for lease in world.leases.load()} == {"d/node-a/0", "d/node-a/1"}
    # Blocking every reap fleet-wide is visible, not just a warning.
    assert world.client.get("/v2/wake").json()["stats"]["orphan_reap_blocked_total"] == 1


def test_one_supervisor_pass_in_observe_releases_a_gone_pods_lease_and_keeps_a_terminating_one():
    # tre_models.sh down deletes the model Deployments with the SM in observe:
    # d-0's Pod is gone, d-1's still terminating. One pass frees d-0's GPU (an
    # overlapping start is no longer refused with lease_conflict) and keeps d-1's
    # (its engine may still be awake in its grace period).
    world = _k8s_world([_k8s_pod("d-1", "d", (1,), terminating=True)])
    world.service._safety_gate.actuation = "observe"
    world.service.detect_fleet_drift = lambda: []

    FleetSupervisor(world.service, drift_observations_required=1).run_once()

    assert {lease.binding_id for lease in world.leases.load()} == {"d/node-a/1"}


def test_a_restart_gives_no_awake_lease_to_a_record_whose_pod_was_replaced():
    # The store records d-0 awake, but its Pod was deleted and a NEW Pod of the
    # binding exists (asleep). The bootstrap neither rebuilds nor carries d-0's
    # awake lease (the Pod that held it is gone); the binding is a suspect
    # instead, which the suspect convergence settles from the new Pod's state.
    from tre_sm.server import rebuild_gpu_leases
    from sm_test_fakes import fence, pod

    world = World()
    old = world.runtime.snapshots.pop("d-0")
    world.runtime.snapshots["d-0-new"] = pod("d-0-new", "d", (0,), ip=old.pod_ip, state="sleeping")
    world.vllm.sleeping[old.pod_ip] = True
    world.runtime.list_live_model_pod_names = lambda: set(world.runtime.snapshots)
    world.runtime.list_live_model_pod_binding_ids = lambda: {
        binding_of(s).binding_id for s in world.runtime.snapshots.values()
    }
    with fence(world.redis):
        suspects = rebuild_gpu_leases(
            world.leases, world.runtime, world.store.load().bindings, starting_bindings=[], waking_bindings=[]
        )

    assert [item[0] for item in suspects] == ["d/node-a/0"]
    assert {lease.binding_id for lease in world.leases.load()} == {"d/node-a/1"}

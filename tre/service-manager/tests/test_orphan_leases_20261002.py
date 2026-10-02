"""Orphan GPU leases (2026-10-02, seen live): an ``awake`` lease never expires, so
one whose binding has no Pod any more - its Deployment deleted and recreated,
its binding dropped by ``reconcile drop_missing`` - refused every start on its
GPUs (409 lease_conflict naming a Pod that is gone). The supervisor's lease
reaper now releases every lease (starting or awake) without a Pod, and
``drop_missing`` releases the leases of the bindings it drops; an unreadable
Pod list releases nothing."""

import json
import logging

from tre_sm.server import K8sPodClientFromOps

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

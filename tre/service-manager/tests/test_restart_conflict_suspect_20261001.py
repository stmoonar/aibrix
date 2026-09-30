"""Integration fix (2026-10-01): a restarted / reloading engine whose GPUs another
binding holds gets no placeholder, but it stays a suspect - its GPUs are never
trusted from gpu-truth and, should it read awake, the suspect convergence settles
it to its desired power. Only a placed placeholder takes a binding off the
suspects."""

from __future__ import annotations

import dataclasses
import logging

from tre_sm.allocator.slots import Binding, Slot
from tre_sm.api.v2 import ServiceManagerV2, restart_placeholder_candidates
from tre_sm.state.store import StateStore

from sm_test_fakes import FakeRedis, fence, pod, registry
from test_review2_sleep import _desired
from test_wake_review_fixes_20260930 import _events, _leases, _world


def _conflict_world(desired_power):
    world = _world([pod("pod-a", "m1", (0,), ip="10.0.0.1", state="sleeping")],
                   [_desired("m1/node-a/0", "m1", (0,), desired_power)])
    world.service._safety_gate.actuation = "active"
    assert world.service.guard_container_restarts() == {"placed": [], "converged": []}  # baseline
    # another binding holds GPU 0 (awake lease); pod-a's engine restarts in place
    with fence(world.redis):
        world.leases.acquire(Binding("pod-x", "m2", Slot("node-a", (0,)), awake=True), phase="awake")
    world.runtime.snapshots["pod-a"] = dataclasses.replace(world.runtime.snapshots["pod-a"], restart_count=1)
    return world


def test_restart_conflict_keeps_the_binding_a_suspect_while_unreadable(caplog):
    world = _conflict_world("sleeping")
    world.service._suspects["m1/node-a/0"] = ("node-a", (0,), "pod-a")  # already a suspect
    world.vllm.physical_override["10.0.0.1"] = None  # still loading
    with caplog.at_level(logging.ERROR, logger="tre_sm.api.v2"):
        result = world.service.guard_container_restarts()
    assert result == {"placed": [], "converged": []}
    assert _events(caplog, "container_restart_conflict")
    assert world.service._suspects == {"m1/node-a/0": ("node-a", (0,), "pod-a")}  # not dropped
    assert world.service._restart_placeholders == {}


def test_restart_conflict_adds_a_suspect_and_converges_it_to_desired_asleep():
    world = _conflict_world("sleeping")
    world.vllm.physical_override["10.0.0.1"] = None
    world.service.guard_container_restarts()
    assert world.service._suspects == {"m1/node-a/0": ("node-a", (0,), "pod-a")}
    # it comes back AWAKE: the suspect convergence sleeps it (desired asleep)
    world.vllm.physical_override.pop("10.0.0.1")
    world.vllm.sleeping["10.0.0.1"] = False
    result = world.service.guard_container_restarts()
    assert result["converged"] == ["m1/node-a/0"]
    assert world.vllm.sleeping["10.0.0.1"] is True
    assert world.service._suspects == {}
    assert set(_leases(world)) == {"m2/node-a/0"}  # the occupant's lease untouched


def test_restart_placeholder_placed_takes_the_binding_off_the_suspects():
    world = _world([pod("pod-a", "m1", (0,), ip="10.0.0.1", state="sleeping")],
                   [_desired("m1/node-a/0", "m1", (0,), "sleeping")])
    world.service._safety_gate.actuation = "active"
    world.service.guard_container_restarts()
    world.service._suspects["m1/node-a/0"] = ("node-a", (0,), "pod-a")
    world.runtime.snapshots["pod-a"] = dataclasses.replace(world.runtime.snapshots["pod-a"], restart_count=1)
    world.vllm.physical_override["10.0.0.1"] = None  # loading: the placeholder is not converged yet
    result = world.service.guard_container_restarts()
    assert result == {"placed": ["m1/node-a/0"], "converged": []}
    assert world.service._suspects == {}
    assert world.service._restart_placeholders == {"m1/node-a/0": "pod-a"}


def test_bootstrap_conflict_is_restored_as_a_suspect_without_a_placeholder():
    clash = dataclasses.replace(
        pod("pod-t", "tp2", (0, 1), ip="10.0.0.4", state="sleeping"), ready=False, engine_running=True
    )
    reloading = dataclasses.replace(
        pod("pod-a", "m1", (2,), ip="10.0.0.1", state="sleeping"), ready=False, engine_running=True
    )
    store = [Binding("pod-x", "m1", Slot("node-a", (1,)), awake=True)]
    conflicts: list = []
    found = restart_placeholder_candidates([clash, reloading], store, conflicts=conflicts)
    assert [b.binding_id for b in found] == ["m1/node-a/2"]
    assert [b.binding_id for b in conflicts] == ["tp2/node-a/0,1"]
    # without the out-list the result is unchanged (backwards compatible)
    assert [b.binding_id for b in restart_placeholder_candidates([clash, reloading], store)] == ["m1/node-a/2"]

    def as_rows(bindings):
        return [(b.binding_id, b.slot.node, tuple(b.slot.gpu_ids), b.serve_id) for b in bindings]

    service = ServiceManagerV2(
        registry(), StateStore(FakeRedis()),
        restored_placeholders=as_rows(found), restored_suspects=as_rows(conflicts),
    )
    assert service._restart_placeholders == {"m1/node-a/2": "pod-a"}
    assert service._suspects == {
        "m1/node-a/2": ("node-a", (2,), "pod-a"),
        "tp2/node-a/0,1": ("node-a", (0, 1), "pod-t"),
    }

"""S5 (2026-09-30): the planner counts sleeping capacity only on GPUs the SM
reports wakeable (``/v2/state`` ``gpus[]``) - the formal fix of the sleeping-
capacity deadlock: a free-looking GPU where a Pod is loading, a wake is in
flight or gpu-truth shows unexplained memory is no capacity."""

from __future__ import annotations

from tre_controller.loops.cluster_view_task import cluster_view_from_state
from tre_controller.planning.classify import ModelRole, ModelState
from tre_controller.planning.planner import ScaleAction, build_plan

from test_gpu_cooldown_20260930 import _bindings
from test_planner_slot_occupancy import TOPOLOGY, _cfg, _cls


def _state(gpus):
    bindings = [
        {"serve_id": b.serve_id, "model": b.model, "node": b.slot.node, "gpu_ids": list(b.slot.gpu_ids),
         "awake": b.awake, "hidden": b.hidden}
        for b in _bindings()
    ]
    return {"version": 1, "bindings": bindings, "gpus": gpus}


def _gpu(node, gpu, *, wakeable, reason=None):
    return {"node": node, "gpu": gpu, "wakeable": wakeable, "reason": reason}


def test_cluster_view_parses_gpus_that_the_bindings_do_not_explain():
    view = cluster_view_from_state(
        _state([
            _gpu("node9", 0, wakeable=False, reason="awake"),
            _gpu("node9", 1, wakeable=False, reason="loading"),
            _gpu("node10", 2, wakeable=False, reason="gpu_truth_used"),
            _gpu("node10", 3, wakeable=False, reason="waking"),
            _gpu("node10", 1, wakeable=False, reason="draining"),
            _gpu("node10", 0, wakeable=True),
            {"broken": True},
        ]),
        TOPOLOGY,
    )
    assert view.blocked_gpus == {("node9", 1), ("node10", 2), ("node10", 3)}
    # an SM that does not report gpus: nothing blocked
    assert cluster_view_from_state({"bindings": []}, TOPOLOGY).blocked_gpus == frozenset()


def _critical_plan(view):
    return build_plan(
        model_contexts={
            "dsqwen-7b": {"routable_pods": 6, "assigned_replicas": 8},
            "dsllama-8b": {"routable_pods": 1, "assigned_replicas": 8},
        },
        classifications=[
            _cls("dsllama-8b", ModelState.CRITICAL, ModelRole.RECEIVER, 0.4),
            _cls("dsqwen-7b", ModelState.HEALTHY, ModelRole.NEUTRAL, 1.1),
        ],
        model_replicas={"dsqwen-7b": 8, "dsllama-8b": 8},
        idle_gpus=0,
        cfg=_cfg(),
        cluster_view=view,
    )


def test_sleeping_capacity_counts_only_wakeable_gpus():
    # The two GPUs free of 7b: node9/1 (a Pod loading) and node10/2 (wakeable).
    view = cluster_view_from_state(
        _state([_gpu("node9", 1, wakeable=False, reason="loading"), _gpu("node10", 2, wakeable=True)]),
        TOPOLOGY,
    )
    wakes = [a for a in _critical_plan(view).actions if isinstance(a, ScaleAction) and a.reason == "critical_sleeping_capacity"]
    assert [a.pods for a in wakes] == [("8b-6",)]  # node10/2, never node9/1

    both_blocked = cluster_view_from_state(
        _state([
            _gpu("node9", 1, wakeable=False, reason="loading"),
            _gpu("node10", 2, wakeable=False, reason="gpu_truth_used"),
        ]),
        TOPOLOGY,
    )
    plan = _critical_plan(both_blocked)
    assert not [a for a in plan.actions if isinstance(a, ScaleAction) and a.reason == "critical_sleeping_capacity"]
    assert "critical_sleeping_blocked:dsllama-8b" in plan.events

"""Replica floor, controller side (fix C, 2026-09-29).

The CRITICAL same-slot shrink (ShrinkForSlotAction) takes one replica of a HIGH
donor. It must count in the plan's deltas and claim the donor for the tick, so a
second CRITICAL receiver of the same tick - another TP>1 receiver's same-slot
shrink, or the donor loop of a TP=1 receiver - never takes the same donor again.
A 2-replica donor with min_replicas 1 used to be taken twice (down to 0).
"""

from __future__ import annotations

from tre_common.registry import ClusterTopology, NodeSpec
from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, TauThresholds
from tre_controller.planning.planner import (
    ClusterView,
    PlanConfig,
    ScaleAction,
    ShrinkForSlotAction,
    build_plan,
)
from tre_sm.allocator.slots import Binding, Slot
from relay_view import expand_relays, relays  # noqa: F401 - 2026-10-02 relay intents


def _cls(model: str, state: ModelState, role: ModelRole, z: float | None, tier: str | None = None):
    return ModelClassification(
        model_name=model,
        state=state,
        role=role,
        Z_m=z,
        eta_m=None,
        trs=0.0,
        theta_m=1.0,
        tau=TauThresholds.from_control(),
        donor_tier=tier,
    )


def _taken(plan, model: str) -> int:
    """Replicas the plan takes from ``model`` (same-slot shrinks + negative scales)."""
    shrinks = sum(1 for a in plan.actions if isinstance(a, ShrinkForSlotAction) and a.donor == model)
    scales = sum(-a.delta for a in expand_relays(plan.actions) if isinstance(a, ScaleAction) and a.model == model and a.delta < 0)
    return shrinks + scales


def _topology(extra_nodes=()) -> ClusterTopology:
    return ClusterTopology(
        nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),) + tuple(extra_nodes)
    )


def test_two_tp2_critical_receivers_take_a_two_replica_donor_once():
    classifications = [
        _cls("tpa", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
        _cls("tpb", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
        _cls("high", ModelState.HIGH, ModelRole.DONOR, 1.4, "surplus"),
    ]
    contexts = {
        "tpa": {"assigned_replicas": 0, "routable_pods": 0},
        "tpb": {"assigned_replicas": 0, "routable_pods": 0},
        "high": {"assigned_replicas": 2, "routable_pods": 2},
    }
    view = ClusterView(
        topology=_topology(),
        bindings=(
            Binding("high-0", "high", Slot("node-a", (0,)), awake=True),
            Binding("high-2", "high", Slot("node-a", (2,)), awake=True),
        ),
    )
    caller_inflight: set[str] = set()
    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"tpa": 0, "tpb": 0, "high": 2},
        idle_gpus=0,
        cfg=PlanConfig(
            min_replicas_per_model=1,
            max_replicas_per_model=2,
            model_tp_sizes={"tpa": 2, "tpb": 2, "high": 1},
        ),
        cluster_view=view,
        inflight_models=caller_inflight,
    )
    shrinks = [a for a in plan.actions if isinstance(a, ShrinkForSlotAction)]
    assert [(a.donor, a.beneficiary) for a in shrinks] == [("high", "tpa")]
    assert _taken(plan, "high") == 1  # 2 replicas, floor 1: one take only
    assert caller_inflight == set()  # the caller's set is not mutated


def test_same_slot_shrink_then_tp1_critical_donor_loop_skips_the_claimed_donor():
    # tp2 takes a HIGH replica through the same-slot shrink; crit1 (tp 1) cannot wake
    # its sleeping binding (high-0 is awake on its GPU) nor create (it has a blocked
    # sleeping binding), so it reaches the donor loop - which must skip "high" now.
    classifications = [
        _cls("tp2", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
        _cls("crit1", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
        _cls("high", ModelState.HIGH, ModelRole.DONOR, 1.4, "surplus"),
    ]
    contexts = {
        "tp2": {"assigned_replicas": 0, "routable_pods": 0},
        "crit1": {"assigned_replicas": 2, "routable_pods": 1, "awake_replicas": 1},
        "high": {"assigned_replicas": 2, "routable_pods": 2},
    }
    view = ClusterView(
        topology=_topology((NodeSpec(name="node-b", gpus=2, two_gpu_slots=((0, 1),)),)),
        bindings=(
            Binding("high-0", "high", Slot("node-a", (0,)), awake=True),
            Binding("high-2", "high", Slot("node-a", (2,)), awake=True),
            Binding("crit1-a0", "crit1", Slot("node-a", (0,)), awake=False),
            Binding("crit1-b0", "crit1", Slot("node-b", (0,)), awake=True),
        ),
    )
    cfg = PlanConfig(
        min_replicas_per_model=1,
        max_replicas_per_model=4,
        model_tp_sizes={"tp2": 2, "crit1": 1, "high": 1},
    )
    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"tp2": 0, "crit1": 2, "high": 2},
        idle_gpus=0,
        cfg=cfg,
        cluster_view=view,
    )
    assert [(a.donor, a.beneficiary) for a in plan.actions if isinstance(a, ShrinkForSlotAction)] == [
        ("high", "tp2")
    ]
    assert not [a for a in expand_relays(plan.actions) if isinstance(a, ScaleAction) and a.model == "high"]
    assert _taken(plan, "high") == 1

    # Without the TP=2 receiver the same crit1 does take the donor (the skip above is
    # the claim, not an unrelated gate).
    alone = build_plan(
        model_contexts=contexts,
        classifications=classifications[1:],
        model_replicas={"crit1": 2, "high": 2},
        idle_gpus=0,
        cfg=cfg,
        cluster_view=view,
    )
    assert _taken(alone, "high") == 1
    assert [a.reason for a in expand_relays(alone.actions) if isinstance(a, ScaleAction) and a.model == "high"] == [
        "critical_donor_immediate"
    ]


def test_same_slot_shrink_counts_takes_planned_by_an_earlier_receiver():
    # crit1 first takes one replica of "high" (critical_donor_immediate); the TP=2
    # receiver after it must not shrink "high" again (2 replicas, floor 1).
    classifications = [
        _cls("crit1", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
        _cls("tp2", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
        _cls("high", ModelState.HIGH, ModelRole.DONOR, 1.4, "surplus"),
    ]
    contexts = {
        "tp2": {"assigned_replicas": 0, "routable_pods": 0},
        "crit1": {"assigned_replicas": 2, "routable_pods": 1, "awake_replicas": 1},
        "high": {"assigned_replicas": 2, "routable_pods": 2},
    }
    view = ClusterView(
        topology=_topology((NodeSpec(name="node-b", gpus=2, two_gpu_slots=((0, 1),)),)),
        bindings=(
            Binding("high-0", "high", Slot("node-a", (0,)), awake=True),
            Binding("high-2", "high", Slot("node-a", (2,)), awake=True),
            Binding("crit1-a0", "crit1", Slot("node-a", (0,)), awake=False),
            Binding("crit1-b0", "crit1", Slot("node-b", (0,)), awake=True),
        ),
    )
    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"tp2": 0, "crit1": 2, "high": 2},
        idle_gpus=0,
        cfg=PlanConfig(
            min_replicas_per_model=1,
            max_replicas_per_model=4,
            model_tp_sizes={"tp2": 2, "crit1": 1, "high": 1},
        ),
        cluster_view=view,
    )
    assert _taken(plan, "high") == 1
    assert not [a for a in plan.actions if isinstance(a, ShrinkForSlotAction)]

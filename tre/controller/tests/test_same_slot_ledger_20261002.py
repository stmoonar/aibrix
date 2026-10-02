"""Same-slot shrink vs fairness piggyback (2026-10-02, approach (a)).

The same-slot preemption (ShrinkForSlotAction) is committed by the tick with exactly
``{beneficiary: 1}``. The planner's ledger now records that claim, so the fairness
piggyback sees ``unclaimed == 0`` and never promises the same shrink to a LOW receiver.
"""

from __future__ import annotations

from test_floor_plan_20260929 import _cls, _topology
from tre_common.registry import NodeSpec
from tre_controller.loops.tick import _safescale_pending_upscales
from tre_controller.planning.classify import ModelRole, ModelState
from tre_controller.planning.planner import (
    ClusterView,
    PlanConfig,
    ScaleAction,
    ShrinkForSlotAction,
    build_plan,
)
from tre_sm.allocator.slots import Binding, Slot


def _same_slot_plan(high_replicas: int, *, low_replicas: int = 1):
    classifications = [
        _cls("tp2", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
        _cls("low", ModelState.LOW, ModelRole.RECEIVER, 0.9),
        _cls("high", ModelState.HIGH, ModelRole.DONOR, 1.4, "surplus"),
    ]
    contexts = {
        "tp2": {"assigned_replicas": 0, "routable_pods": 0},
        "low": {"assigned_replicas": low_replicas, "routable_pods": low_replicas},
        "high": {"assigned_replicas": high_replicas, "routable_pods": high_replicas},
    }
    bindings = [
        Binding("high-0", "high", Slot("node-a", (0,)), awake=True),
        Binding("high-2", "high", Slot("node-a", (2,)), awake=True),
        Binding("low-3", "low", Slot("node-a", (3,)), awake=True),
        # A sleeping binding on a GPU a HIGH replica holds awake: LOW cannot wake it, and
        # the SM tries it before creating, so there is no idle capacity for LOW.
        Binding("low-s", "low", Slot("node-a", (2,)), awake=False),
    ]
    if high_replicas > 2:
        bindings.append(Binding("high-n", "high", Slot("node-b", (0,)), awake=True))
    view = ClusterView(
        topology=_topology((NodeSpec(name="node-b", gpus=2, two_gpu_slots=((0, 1),)),)),
        bindings=tuple(bindings),
    )
    cfg = PlanConfig(
        min_replicas_per_model=1,
        max_replicas_per_model=4,
        model_tp_sizes={"tp2": 2, "low": 1, "high": 1},
    )
    return build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"tp2": 0, "low": low_replicas, "high": high_replicas},
        idle_gpus=0,
        cfg=cfg,
        cluster_view=view,
    )


def test_same_slot_shrink_is_claimed_in_the_ledger_and_not_piggybacked_by_low():
    plan = _same_slot_plan(2)
    shrinks = [a for a in plan.actions if isinstance(a, ShrinkForSlotAction)]
    assert [(a.donor, a.beneficiary) for a in shrinks] == [("high", "tp2")]
    assert plan.probe_upscale_plans == {"high": {"tp2": 1}}
    assert "low" not in plan.probe_upscale_plans.get("high", {})
    assert not [a for a in plan.actions if isinstance(a, ScaleAction) and a.model in {"high", "low"}]


def test_three_replica_donor_still_feeds_low_through_an_immediate_pair_on_another_pod():
    plan = _same_slot_plan(3)
    [shrink] = [a for a in plan.actions if isinstance(a, ShrinkForSlotAction)]
    assert plan.probe_upscale_plans == {"high": {"tp2": 1}}
    scales = [a for a in plan.actions if isinstance(a, ScaleAction)]
    pair = [a for a in scales if a.reason == "low_fairness_donor_immediate"]
    assert sorted((a.model, a.delta) for a in pair) == [("high", -1), ("low", 1)]
    donor_pair = next(a for a in pair if a.model == "high")
    assert shrink.serve_id not in donor_pair.pods and donor_pair.pods


def _proactive(n_low: int, high_replicas: int):
    from test_planner import _classification

    classifications = [_classification("high", ModelState.HIGH, ModelRole.DONOR, 1.6, "surplus")]
    contexts = {"high": {"assigned_replicas": high_replicas, "routable_pods": high_replicas}}
    replicas = {"high": high_replicas}
    for i in range(n_low):
        name = f"low{i}"
        classifications.append(_classification(name, ModelState.LOW, ModelRole.RECEIVER, 0.9))
        contexts[name] = {"assigned_replicas": 1, "routable_pods": 1}
        replicas[name] = 1
    return build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas=replicas,
        idle_gpus=0,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4, suppress_hot_proactive_probe=False),
    )


def test_proactive_high_probe_still_hands_its_shrink_to_low_receivers():
    plan = _proactive(1, 2)
    [shrink] = [a for a in plan.actions if isinstance(a, ScaleAction) and a.model == "high"]
    assert shrink.reason == "high_proactive_safescale" and shrink.delta < 0
    assert plan.probe_upscale_plans["high"] == {"low0": -shrink.delta}


def test_proactive_high_probe_claims_never_exceed_the_shrink_with_two_low_receivers():
    plan = _proactive(2, 4)
    shrinks = [a for a in plan.actions if isinstance(a, ScaleAction) and a.reason == "high_proactive_safescale"]
    assert shrinks
    assert sum(plan.probe_upscale_plans["high"].values()) == sum(-a.delta for a in shrinks)


def test_middle_zone_claim_leaves_nothing_unclaimed_for_a_later_low_receiver():
    from test_planner import _classification

    plan = build_plan(
        model_contexts={
            "critical": {"assigned_replicas": 2, "routable_pods": 2},
            "low": {"assigned_replicas": 1, "routable_pods": 1},
            "healthy": {"assigned_replicas": 2, "routable_pods": 2},
        },
        classifications=[
            _classification("critical", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
            _classification("low", ModelState.LOW, ModelRole.RECEIVER, 0.9),
            _classification("healthy", ModelState.HEALTHY, ModelRole.NEUTRAL, 1.2),
        ],
        model_replicas={"critical": 2, "low": 1, "healthy": 2},
        idle_gpus=0,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4),
    )
    shrinks = [a for a in plan.actions if isinstance(a, ScaleAction) and a.model == "healthy"]
    assert [(a.reason, a.delta) for a in shrinks] == [("critical_middle_zone_safescale", -1)]
    # The CRITICAL claim covers the whole shrink: LOW gets no piggyback.
    assert plan.probe_upscale_plans == {"healthy": {"critical": 1}}


def test_tick_commits_a_same_slot_probe_with_only_the_beneficiary_upscale():
    # Hand-built input: the ShrinkForSlotAction and the ledger are given directly.
    from test_loop_ticks import _metrics_with_pods
    from tre_common.metrics_schema import MetricsSnapshot
    from tre_controller.loops.tick import _apply_safescale
    from tre_controller.planning.safescale import SafeScaleConfig, SafeScaleStateMachine

    safescale = SafeScaleStateMachine(config=SafeScaleConfig(min_window_ms=60_000.0))
    snapshot = MetricsSnapshot(
        ts_ms=20_000, stale=False,
        models={"high": _metrics_with_pods("high", generation=200.0, waiting=0.0, running=1.0, pods=("high-0",))},
    )
    shrink = ShrinkForSlotAction(
        donor="high", beneficiary="tp2", serve_id="high-0", slot=Slot("node-a", (0,)),
        reason="critical_same_slot_high_shrink", source_loop="rescue",
    )
    assert _safescale_pending_upscales(shrink, {"high": {"tp2": 1}}) == {"tp2": 1}
    out, events = _apply_safescale(snapshot, (shrink,), {"high": {"tp2": 1}}, safescale=safescale)
    probe = safescale.active_probe("high")
    assert probe is not None and probe.pending_upscales == {"tp2": 1}
    decision = safescale._commit(probe, reason="formal_commit_gate_passed")
    assert [(c.kind, c.model, c.delta) for c in decision.commands if c.kind == "scale_up"] == [("scale_up", "tp2", 1)]

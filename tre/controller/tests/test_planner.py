from __future__ import annotations

from golden.legacy_planner import (
    LegacyClassification,
    LegacyModelRole,
    LegacyModelState,
    legacy_build_paper_plan,
)
from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, TauThresholds
from tre_common.registry import ClusterTopology, NodeSpec
from tre_controller.planning.planner import ClusterView, DefragAction, PlanConfig, ScaleAction, ShrinkForSlotAction, build_plan
from tre_sm.allocator.slots import Binding, Migration, Slot


def _classification(model: str, state: ModelState, role: ModelRole, z: float | None, tier: str | None = None) -> ModelClassification:
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


def _legacy_classification(model: str, state: ModelState, role: ModelRole, z: float | None, tier: str | None = None) -> LegacyClassification:
    return LegacyClassification(
        model_name=model,
        state=LegacyModelState(state.value),
        role=LegacyModelRole(role.value),
        Z_m=z,
        donor_tier=tier,
    )


def _deltas(actions: list[ScaleAction]) -> dict[str, int]:
    out: dict[str, int] = {}
    for action in actions:
        out[action.model] = out.get(action.model, 0) + action.delta
    return out


def test_build_plan_matches_legacy_rescue_idle_and_high_donor_path() -> None:
    classifications = [
        _classification("critical", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
        _classification("idle", ModelState.IDLE, ModelRole.DONOR, 10.0, "idle"),
        _classification("high", ModelState.HIGH, ModelRole.DONOR, 1.6, "surplus"),
    ]
    legacy_classifications = [
        _legacy_classification("critical", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
        _legacy_classification("idle", ModelState.IDLE, ModelRole.DONOR, 10.0, "idle"),
        _legacy_classification("high", ModelState.HIGH, ModelRole.DONOR, 1.6, "surplus"),
    ]
    contexts = {
        "critical": {"assigned_replicas": 2, "routable_pods": 2},
        "idle": {"assigned_replicas": 3, "routable_pods": 3},
        "high": {"assigned_replicas": 2, "routable_pods": 2},
    }
    replicas = {"critical": 2, "idle": 3, "high": 2}

    expected = legacy_build_paper_plan(
        classifications=legacy_classifications,
        model_contexts=contexts,
        model_replicas=replicas,
        idle_gpus=0,
        min_replicas_per_model=1,
        max_replicas_per_model=4,
    )
    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas=replicas,
        idle_gpus=0,
        # This is a pure legacy-parity regression test for the frozen paper path. The t1
        # suppress-hot-proactive guard is a deliberate post-migration divergence, so it is
        # disabled here to keep the migration faithfulness check meaningful. The guarded
        # (default-on) behaviour is covered by the dedicated tests below.
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4, suppress_hot_proactive_probe=False),
    )

    assert _deltas([action for action in plan.actions if isinstance(action, ScaleAction)]) == expected.deltas
    assert plan.delayed_down_models == expected.delayed_down_models
    assert plan.probe_upscale_plans == expected.probe_upscale_plans
    assert all(action.source_loop == "rescue" for action in plan.actions)


def test_build_plan_matches_legacy_middle_zone_safescale_probe_path() -> None:
    classifications = [
        _classification("critical", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
        _classification("healthy", ModelState.HEALTHY, ModelRole.NEUTRAL, 1.2),
    ]
    legacy_classifications = [
        _legacy_classification("critical", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
        _legacy_classification("healthy", ModelState.HEALTHY, ModelRole.NEUTRAL, 1.2),
    ]
    contexts = {
        "critical": {"assigned_replicas": 2, "routable_pods": 2},
        "healthy": {"assigned_replicas": 3, "routable_pods": 3},
    }
    replicas = {"critical": 2, "healthy": 3}

    expected = legacy_build_paper_plan(
        classifications=legacy_classifications,
        model_contexts=contexts,
        model_replicas=replicas,
        idle_gpus=0,
        min_replicas_per_model=1,
        max_replicas_per_model=4,
    )
    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas=replicas,
        idle_gpus=0,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4),
    )

    assert _deltas([action for action in plan.actions if isinstance(action, ScaleAction)]) == expected.deltas
    assert plan.delayed_down_models == {"healthy"}
    assert plan.probe_upscale_plans == {"healthy": {"critical": 1}}
    shrink = next(action for action in plan.actions if isinstance(action, ScaleAction) and action.model == "healthy")
    assert shrink.requires_safescale is True
    assert shrink.reason == "critical_middle_zone_safescale"


def test_build_plan_low_fairness_receiver_needs_no_saturation() -> None:
    # ADR-0014 (behaviour change): fairness receiver eligibility is z_m-band only. A LOW
    # receiver that is NOT saturated now receives donor surplus. Previously the planner
    # emitted "fairness_blocked_unsaturated" and gave nothing unless is_saturated was set.
    # (No legacy parity here: the frozen legacy planner still has the removed gate and
    # would diverge -- that divergence is the whole point of ADR-0014.)
    classifications = [
        _classification("low", ModelState.LOW, ModelRole.RECEIVER, 0.9),
        _classification("high", ModelState.HIGH, ModelRole.DONOR, 1.6, "surplus"),
    ]
    contexts = {
        "low": {"assigned_replicas": 2, "routable_pods": 2},  # NOTE: no is_saturated
        "high": {"assigned_replicas": 3, "routable_pods": 3},
    }
    replicas = {"low": 2, "high": 3}

    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas=replicas,
        idle_gpus=0,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4, rescue_due=False, fairness_due=True),
    )

    upscales = [a for a in plan.actions if isinstance(a, ScaleAction) and a.delta > 0]
    assert any(a.model == "low" for a in upscales), "non-saturated LOW receiver must now receive"
    assert not any(e.startswith("fairness_blocked_unsaturated") for e in plan.events)
    assert {action.source_loop for action in plan.actions} == {"fairness"}


def test_build_plan_drops_only_incomplete_model_by_default() -> None:
    classifications = [
        _classification("critical", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
        _classification("unknown", ModelState.UNKNOWN, ModelRole.UNKNOWN, None),
    ]
    contexts = {
        "critical": {"assigned_replicas": 1, "routable_pods": 1},
        "unknown": {"assigned_replicas": 2, "routable_pods": 2, "trs": 10.0},
    }

    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"critical": 1, "unknown": 2},
        idle_gpus=1,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4),
    )

    assert plan.dropped_legacy_raw_trs is False
    assert plan.events == ["paper_state_incomplete_drop:unknown"]
    assert plan.actions == [
        ScaleAction(
            model="critical",
            delta=1,
            reason="critical_idle_capacity",
            source_loop="rescue",
            receiver="critical",
        )
    ]


def test_build_plan_can_keep_legacy_drop_all_for_incomplete_paper_state() -> None:
    classifications = [_classification("unknown", ModelState.UNKNOWN, ModelRole.UNKNOWN, None)]
    contexts = {"unknown": {"assigned_replicas": 2, "routable_pods": 2, "trs": 10.0}}

    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"unknown": 2},
        idle_gpus=4,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4, incomplete_policy="drop_all"),
    )

    assert plan.actions == []
    assert plan.dropped_legacy_raw_trs is True
    assert plan.events == ["paper_state_incomplete_drop_legacy_raw_trs"]


def test_idle_proactive_honors_per_model_min_replicas() -> None:
    classifications = [_classification("warm", ModelState.IDLE, ModelRole.DONOR, 10.0, "idle")]
    contexts = {"warm": {"assigned_replicas": 4, "routable_pods": 1}}

    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"warm": 4},
        idle_gpus=3,
        cfg=PlanConfig(
            min_replicas_per_model=0,
            max_replicas_per_model=4,
            min_replicas_by_model={"warm": 1},
            max_replicas_by_model={"warm": 4},
        ),
    )

    assert plan.actions == []


def test_idle_proactive_keeps_one_routable_endpoint_for_bound_zero_min_model() -> None:
    classifications = [_classification("warm", ModelState.IDLE, ModelRole.DONOR, 10.0, "idle")]
    contexts = {"warm": {"assigned_replicas": 2, "routable_pods": 1}}

    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"warm": 2},
        idle_gpus=2,
        cfg=PlanConfig(
            min_replicas_per_model=0,
            max_replicas_per_model=2,
            min_replicas_by_model={"warm": 0},
            max_replicas_by_model={"warm": 2},
        ),
    )

    assert plan.actions == []


def test_critical_receiver_wakes_sleeping_bound_replica_before_claiming_idle_capacity() -> None:
    classifications = [_classification("warm", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5)]
    contexts = {"warm": {"assigned_replicas": 4, "routable_pods": 1}}

    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"warm": 4},
        idle_gpus=0,
        cfg=PlanConfig(
            min_replicas_per_model=0,
            max_replicas_per_model=4,
            min_replicas_by_model={"warm": 1},
            max_replicas_by_model={"warm": 4},
        ),
    )

    assert plan.actions == [
        ScaleAction(
            model="warm",
            delta=1,
            reason="critical_sleeping_capacity",
            source_loop="rescue",
            receiver="warm",
        )
    ]


def test_critical_receiver_with_zero_assigned_can_expand_to_one_replica_max() -> None:
    classifications = [_classification("cold", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5)]
    contexts = {"cold": {"assigned_replicas": 0, "routable_pods": 0}}

    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"cold": 0},
        idle_gpus=1,
        cfg=PlanConfig(
            min_replicas_per_model=0,
            max_replicas_per_model=4,
            min_replicas_by_model={"cold": 0},
            max_replicas_by_model={"cold": 1},
        ),
    )

    assert plan.actions == [
        ScaleAction(
            model="cold",
            delta=1,
            reason="critical_idle_capacity",
            source_loop="rescue",
            receiver="cold",
        )
    ]


def _tp2_topology() -> ClusterTopology:
    return ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),))


def test_tp_aware_critical_receiver_uses_complete_two_gpu_slot() -> None:
    classifications = [_classification("tp2", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5)]
    contexts = {"tp2": {"assigned_replicas": 0, "routable_pods": 0}}
    cluster_view = ClusterView(
        topology=_tp2_topology(),
        bindings=(Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),),
    )

    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"tp2": 0},
        idle_gpus=2,
        cfg=PlanConfig(min_replicas_per_model=0, max_replicas_per_model=2, model_tp_sizes={"tp2": 2}),
        cluster_view=cluster_view,
    )

    assert [action for action in plan.actions if isinstance(action, DefragAction)] == []
    scale = next(action for action in plan.actions if isinstance(action, ScaleAction))
    assert scale.model == "tp2"
    assert scale.delta == 1
    assert scale.reason == "critical_empty_slot"


def test_tp_aware_critical_receiver_plans_defrag_for_fragmented_two_gpu_capacity() -> None:
    classifications = [_classification("tp2", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5)]
    contexts = {"tp2": {"assigned_replicas": 0, "routable_pods": 0}}
    cluster_view = ClusterView(
        topology=_tp2_topology(),
        bindings=(
            Binding("serve-0", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-2", "m1", Slot("node-a", (2,)), awake=True),
        ),
    )

    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"tp2": 0},
        idle_gpus=2,
        cfg=PlanConfig(min_replicas_per_model=0, max_replicas_per_model=2, model_tp_sizes={"tp2": 2}),
        cluster_view=cluster_view,
    )

    defrag = next(action for action in plan.actions if isinstance(action, DefragAction))
    assert defrag.reason == "critical_tp_defrag"
    assert defrag.migrations == (
        Migration(
            serve_id="serve-2",
            from_slot=Slot("node-a", (2,)),
            to_slot=Slot("node-a", (1,)),
        ),
    )
    scale = next(action for action in plan.actions if isinstance(action, ScaleAction))
    assert scale.model == "tp2"
    assert scale.delta == 1
    assert scale.reason == "critical_tp_defrag"


def test_tp_aware_critical_receiver_records_capacity_blocked_when_no_slot_or_defrag() -> None:
    classifications = [_classification("tp2", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5)]
    contexts = {"tp2": {"assigned_replicas": 0, "routable_pods": 0}}
    cluster_view = ClusterView(
        topology=_tp2_topology(),
        bindings=(
            Binding("serve-0", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-1", "m1", Slot("node-a", (1,)), awake=True),
            Binding("serve-2", "m1", Slot("node-a", (2,)), awake=True),
            Binding("serve-3", "m1", Slot("node-a", (3,)), awake=True),
        ),
    )

    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"tp2": 0},
        idle_gpus=2,
        cfg=PlanConfig(min_replicas_per_model=0, max_replicas_per_model=2, model_tp_sizes={"tp2": 2}),
        cluster_view=cluster_view,
    )

    assert [action for action in plan.actions if isinstance(action, ScaleAction)] == []
    assert [action for action in plan.actions if isinstance(action, DefragAction)] == []
    assert plan.events == ["capacity_blocked:tp2"]

def test_tp_aware_critical_receiver_prefers_high_same_slot_shrink_before_defrag() -> None:
    classifications = [
        _classification("tp2", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
        _classification("high", ModelState.HIGH, ModelRole.DONOR, 1.4, "surplus"),
        _classification("other", ModelState.HEALTHY, ModelRole.NEUTRAL, 1.1),
    ]
    contexts = {
        "tp2": {"assigned_replicas": 0, "routable_pods": 0},
        "high": {"assigned_replicas": 1, "routable_pods": 1},
        "other": {"assigned_replicas": 1, "routable_pods": 1},
    }
    cluster_view = ClusterView(
        topology=_tp2_topology(),
        bindings=(
            Binding("high-0", "high", Slot("node-a", (0,)), awake=True),
            Binding("other-2", "other", Slot("node-a", (2,)), awake=True),
        ),
    )

    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"tp2": 0, "high": 1, "other": 1},
        idle_gpus=2,
        cfg=PlanConfig(
            min_replicas_per_model=0,
            max_replicas_per_model=2,
            model_tp_sizes={"tp2": 2, "high": 1, "other": 1},
        ),
        cluster_view=cluster_view,
    )

    assert [action for action in plan.actions if isinstance(action, DefragAction)] == []
    assert [action for action in plan.actions if isinstance(action, ScaleAction)] == []
    assert plan.actions == [
        ShrinkForSlotAction(
            donor="high",
            beneficiary="tp2",
            serve_id="high-0",
            slot=Slot("node-a", (0,)),
            reason="critical_same_slot_high_shrink",
            source_loop="rescue",
        )
    ]


# --- t1 regression: suppress the receiver-less proactive scale-down probe on hot donors ---
# Diagnosis (TRE vs APA, timeline.csv): during a 7b/8b traffic spike the model is momentarily
# classified HIGH (TSS = throughput/queue spikes up), and the rescue-loop high_proactive
# block hid one of its in-service pods (routable 4->3) as a speculative surplus-reclaim probe
# with NO beneficiary -- deepening saturation right as load climbed. The guard (default on)
# must reject that probe while leaving demand-driven preemption and idle shrink untouched.


def test_high_proactive_probe_suppressed_for_hot_model_by_default() -> None:
    # A lone HIGH (hot, z_m far above tau_high) model with no receiver needing capacity.
    classifications = [_classification("hot", ModelState.HIGH, ModelRole.DONOR, 1.6, "surplus")]
    contexts = {"hot": {"assigned_replicas": 4, "routable_pods": 4}}

    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"hot": 4},
        idle_gpus=0,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4),  # guard default ON
    )

    assert [a for a in plan.actions if isinstance(a, ScaleAction) and a.requires_safescale] == []
    assert not any(a for a in plan.actions if isinstance(a, ScaleAction) and a.delta < 0)
    assert "safescale_probe_suppressed_hot:hot" in plan.events
    assert plan.delayed_down_models == set()
    assert plan.probe_upscale_plans == {}


def test_high_proactive_probe_emitted_when_guard_disabled() -> None:
    # Ablation path (TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE=0): legacy proactive release fires.
    classifications = [_classification("hot", ModelState.HIGH, ModelRole.DONOR, 1.6, "surplus")]
    contexts = {"hot": {"assigned_replicas": 4, "routable_pods": 4}}

    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"hot": 4},
        idle_gpus=0,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4, suppress_hot_proactive_probe=False),
    )

    shrink = next(a for a in plan.actions if isinstance(a, ScaleAction) and a.model == "hot")
    assert shrink.delta < 0
    assert shrink.requires_safescale is True
    assert shrink.reason == "high_proactive_safescale"
    assert plan.delayed_down_models == {"hot"}
    assert "safescale_probe_suppressed_hot:hot" not in plan.events


def test_critical_preemption_of_hot_donor_survives_guard() -> None:
    # Demand-driven preemption: a TP=2 CRITICAL receiver blocked on a fragmented slot with no
    # idle capacity legitimately shrinks a HIGH donor sharing its two-GPU slot. The guard is ON
    # (default) yet this MUST still fire, tagged with an explicit preemption reason.
    classifications = [
        _classification("tp2", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
        _classification("high", ModelState.HIGH, ModelRole.DONOR, 1.4, "surplus"),
        _classification("other", ModelState.HEALTHY, ModelRole.NEUTRAL, 1.1),
    ]
    contexts = {
        "tp2": {"assigned_replicas": 0, "routable_pods": 0},
        "high": {"assigned_replicas": 1, "routable_pods": 1},
        "other": {"assigned_replicas": 1, "routable_pods": 1},
    }
    cluster_view = ClusterView(
        topology=_tp2_topology(),
        bindings=(
            Binding("high-0", "high", Slot("node-a", (0,)), awake=True),
            Binding("other-2", "other", Slot("node-a", (2,)), awake=True),
        ),
    )

    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"tp2": 0, "high": 1, "other": 1},
        idle_gpus=2,
        cfg=PlanConfig(
            min_replicas_per_model=0,
            max_replicas_per_model=2,
            model_tp_sizes={"tp2": 2, "high": 1, "other": 1},
        ),  # guard default ON
        cluster_view=cluster_view,
    )

    assert plan.actions == [
        ShrinkForSlotAction(
            donor="high",
            beneficiary="tp2",
            serve_id="high-0",
            slot=Slot("node-a", (0,)),
            reason="critical_same_slot_high_shrink",
            source_loop="rescue",
        )
    ]
    assert "safescale_preemption:high->tp2:critical_same_slot_high_shrink" in plan.events
    assert not any(e.startswith("safescale_probe_suppressed_hot") for e in plan.events)


def test_middle_zone_safescale_donor_not_suppressed_by_hot_guard() -> None:
    # The middle-zone donor is HEALTHY (not hot) and feeds a CRITICAL receiver, so the guard
    # (default ON) must leave the demand-driven middle-zone SafeScale probe intact.
    classifications = [
        _classification("critical", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
        _classification("healthy", ModelState.HEALTHY, ModelRole.NEUTRAL, 1.2),
    ]
    contexts = {
        "critical": {"assigned_replicas": 2, "routable_pods": 2},
        "healthy": {"assigned_replicas": 3, "routable_pods": 3},
    }

    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"critical": 2, "healthy": 3},
        idle_gpus=0,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4),  # guard default ON
    )

    shrink = next(a for a in plan.actions if isinstance(a, ScaleAction) and a.model == "healthy")
    assert shrink.requires_safescale is True
    assert shrink.reason == "critical_middle_zone_safescale"
    assert plan.probe_upscale_plans == {"healthy": {"critical": 1}}
    assert not any(e.startswith("safescale_probe_suppressed_hot") for e in plan.events)


def test_idle_proactive_immediate_shrink_not_affected_by_hot_guard() -> None:
    # Genuine idle over-provisioning is reclaimed by idle_proactive_immediate (no SafeScale
    # probe). The hot guard (default ON) must not touch it.
    classifications = [_classification("idle", ModelState.IDLE, ModelRole.DONOR, 10.0, "idle")]
    contexts = {"idle": {"assigned_replicas": 4, "routable_pods": 4, "Y_m": 0.0, "Q": 0.0}}

    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"idle": 4},
        idle_gpus=0,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4),  # guard default ON
    )

    shrink = next(a for a in plan.actions if isinstance(a, ScaleAction) and a.model == "idle")
    assert shrink.delta < 0
    assert shrink.requires_safescale is False
    assert shrink.reason == "idle_proactive_immediate"
    assert not any(e.startswith("safescale_probe_suppressed_hot") for e in plan.events)


def test_disable_eta_gate_uses_signal_independent_natural_donor_order() -> None:
    classifications = [
        _classification("receiver", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
        _classification("model-10", ModelState.HIGH, ModelRole.DONOR, 2.0, "surplus"),
        _classification("model-2", ModelState.HIGH, ModelRole.DONOR, 1.5, "surplus"),
    ]
    contexts = {
        "receiver": {"assigned_replicas": 1, "routable_pods": 1},
        "model-10": {"assigned_replicas": 2, "routable_pods": 2},
        "model-2": {"assigned_replicas": 2, "routable_pods": 2},
    }

    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"receiver": 1, "model-10": 2, "model-2": 2},
        idle_gpus=0,
        cfg=PlanConfig(
            min_replicas_per_model=1,
            max_replicas_per_model=4,
            disable_eta_gate=True,
        ),
    )

    donor_shrinks = [
        action
        for action in plan.actions
        if isinstance(action, ScaleAction) and action.delta < 0
    ]
    assert donor_shrinks
    assert donor_shrinks[0].model == "model-2"
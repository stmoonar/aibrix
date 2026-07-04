from __future__ import annotations

from golden.legacy_planner import (
    LegacyClassification,
    LegacyModelRole,
    LegacyModelState,
    legacy_build_paper_plan,
)
from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, TauThresholds
from tre_common.registry import ClusterTopology, NodeSpec
from tre_controller.planning.planner import ClusterView, DefragAction, PlanConfig, ScaleAction, build_plan
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
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4),
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


def test_build_plan_matches_legacy_low_fairness_saturation_gate() -> None:
    classifications = [
        _classification("low", ModelState.LOW, ModelRole.RECEIVER, 0.9),
        _classification("high", ModelState.HIGH, ModelRole.DONOR, 1.6, "surplus"),
    ]
    legacy_classifications = [
        _legacy_classification("low", ModelState.LOW, ModelRole.RECEIVER, 0.9),
        _legacy_classification("high", ModelState.HIGH, ModelRole.DONOR, 1.6, "surplus"),
    ]
    contexts = {
        "low": {"assigned_replicas": 2, "routable_pods": 2, "is_saturated": True},
        "high": {"assigned_replicas": 3, "routable_pods": 3},
    }
    replicas = {"low": 2, "high": 3}

    expected = legacy_build_paper_plan(
        classifications=legacy_classifications,
        model_contexts=contexts,
        model_replicas=replicas,
        idle_gpus=0,
        min_replicas_per_model=1,
        max_replicas_per_model=4,
        rescue_due=False,
        fairness_due=True,
    )
    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas=replicas,
        idle_gpus=0,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4, rescue_due=False, fairness_due=True),
    )

    assert _deltas([action for action in plan.actions if isinstance(action, ScaleAction)]) == expected.deltas
    assert plan.delayed_down_models == expected.delayed_down_models
    assert plan.probe_upscale_plans == expected.probe_upscale_plans
    assert {action.source_loop for action in plan.actions} == {"fairness"}


def test_build_plan_drops_legacy_raw_trs_fallback_when_paper_state_unknown() -> None:
    classifications = [_classification("unknown", ModelState.UNKNOWN, ModelRole.UNKNOWN, None)]
    contexts = {"unknown": {"assigned_replicas": 2, "routable_pods": 2, "trs": 10.0}}

    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"unknown": 2},
        idle_gpus=4,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4),
    )

    assert plan.actions == []
    assert plan.dropped_legacy_raw_trs is True
    assert plan.events == ["paper_state_incomplete_drop_legacy_raw_trs"]


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

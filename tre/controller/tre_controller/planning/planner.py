from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from tre_common.registry import ClusterTopology
from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, donor_mock_cost_key
from tre_sm.allocator.slots import Binding, SlotAllocator

SourceLoop = Literal["rescue", "fairness"]


@dataclass(frozen=True)
class PlanConfig:
    min_replicas_per_model: int
    max_replicas_per_model: int
    scale_step_ratio: float = 0.1
    rescue_due: bool = True
    fairness_due: bool = True
    model_tp_sizes: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ClusterView:
    topology: ClusterTopology
    bindings: tuple[Binding, ...]


@dataclass(frozen=True)
class ScaleAction:
    model: str
    delta: int
    reason: str
    source_loop: SourceLoop
    requires_safescale: bool = False
    receiver: str | None = None
    donor: str | None = None


@dataclass(frozen=True)
class HideAction:
    model: str
    pods: tuple[str, ...]
    reason: str
    source_loop: SourceLoop


@dataclass(frozen=True)
class UnhideAction:
    model: str
    pods: tuple[str, ...]
    reason: str
    source_loop: SourceLoop


@dataclass(frozen=True)
class DefragAction:
    migrations: tuple[Any, ...]
    reason: str
    source_loop: SourceLoop


Action = ScaleAction | HideAction | UnhideAction | DefragAction


@dataclass(frozen=True)
class PlanResult:
    actions: list[Action]
    delayed_down_models: set[str] = field(default_factory=set)
    probe_upscale_plans: dict[str, dict[str, int]] = field(default_factory=dict)
    dropped_legacy_raw_trs: bool = False
    events: list[str] = field(default_factory=list)


def build_plan(
    *,
    model_contexts: dict[str, dict[str, Any]],
    classifications: list[ModelClassification],
    model_replicas: dict[str, int],
    idle_gpus: int,
    cfg: PlanConfig,
    active_probe_models: set[str] | None = None,
    inflight_models: set[str] | None = None,
    cluster_view: ClusterView | None = None,
) -> PlanResult:
    active_probe_models = active_probe_models or set()
    inflight_models = inflight_models or set()
    actions: list[Action] = []
    deltas: dict[str, int] = {}
    delayed_down_models: set[str] = set()
    probe_upscale_plans: dict[str, dict[str, int]] = {}
    events: list[str] = []
    remaining_idle = idle_gpus

    if _paper_state_incomplete(classifications):
        return PlanResult(
            actions=[],
            delayed_down_models=set(),
            probe_upscale_plans={},
            dropped_legacy_raw_trs=True,
            events=["paper_state_incomplete_drop_legacy_raw_trs"],
        )

    critical_receivers = [item for item in classifications if item.state == ModelState.CRITICAL]
    low_receivers = [item for item in classifications if item.state == ModelState.LOW]
    high_models = [item for item in classifications if item.state == ModelState.HIGH]
    idle_models = [item for item in classifications if item.state == ModelState.IDLE]
    paper_donors = [item for item in classifications if item.role == ModelRole.DONOR]
    paper_donors.sort(key=donor_mock_cost_key)
    middle_zone = [
        item
        for item in classifications
        if item.state in (ModelState.LOW, ModelState.HEALTHY) and item.role != ModelRole.RECEIVER
    ]
    middle_zone.sort(key=lambda item: (0 if item.state == ModelState.HEALTHY else 1, -(item.Z_m or 0.0)))

    if cfg.rescue_due:
        for recv in critical_receivers:
            if recv.model_name in inflight_models:
                continue
            recv_pods = _effective_assigned_replicas(recv.model_name, model_contexts, model_replicas)
            if recv_pods >= cfg.max_replicas_per_model:
                continue
            raw_need = min(_scale_step(recv_pods, cfg.scale_step_ratio), cfg.max_replicas_per_model - recv_pods)
            if raw_need <= 0:
                continue

            tp_size = cfg.model_tp_sizes.get(recv.model_name, 1)
            if tp_size > 1 and cluster_view is not None:
                tp_planned = _try_plan_tp_capacity(
                    actions,
                    model=recv.model_name,
                    tp_size=tp_size,
                    cluster_view=cluster_view,
                    events=events,
                    source_loop="rescue",
                )
                if tp_planned:
                    _add_scale_action(
                        actions,
                        deltas,
                        model=recv.model_name,
                        delta=1,
                        reason=tp_planned,
                        source_loop="rescue",
                        receiver=recv.model_name,
                    )
                continue

            gain_from_idle = min(raw_need, remaining_idle) if remaining_idle > 0 else 0
            if gain_from_idle > 0:
                _add_scale_action(
                    actions,
                    deltas,
                    model=recv.model_name,
                    delta=gain_from_idle,
                    reason="critical_idle_capacity",
                    source_loop="rescue",
                    receiver=recv.model_name,
                )
                remaining_idle -= gain_from_idle

            still_needed = raw_need - gain_from_idle
            for donor in paper_donors:
                if still_needed <= 0:
                    break
                if (
                    donor.model_name == recv.model_name
                    or donor.model_name in active_probe_models
                    or donor.model_name in inflight_models
                    or donor.state not in (ModelState.IDLE, ModelState.HIGH)
                ):
                    continue
                donor_pods = _effective_assigned_replicas(donor.model_name, model_contexts, model_replicas)
                if donor_pods <= cfg.min_replicas_per_model:
                    continue
                planned_take = abs(min(deltas.get(donor.model_name, 0), 0))
                transfer = min(
                    still_needed,
                    _scale_step(donor_pods, cfg.scale_step_ratio),
                    max(0, donor_pods - planned_take - cfg.min_replicas_per_model),
                )
                if transfer <= 0:
                    continue
                _add_scale_action(
                    actions,
                    deltas,
                    model=donor.model_name,
                    delta=-transfer,
                    reason="critical_donor_immediate",
                    source_loop="rescue",
                    donor=donor.model_name,
                    receiver=recv.model_name,
                )
                _add_scale_action(
                    actions,
                    deltas,
                    model=recv.model_name,
                    delta=transfer,
                    reason="critical_donor_immediate",
                    source_loop="rescue",
                    donor=donor.model_name,
                    receiver=recv.model_name,
                )
                still_needed -= transfer

            for middle in middle_zone:
                if still_needed <= 0:
                    break
                if (
                    middle.model_name == recv.model_name
                    or middle.model_name in active_probe_models
                    or middle.model_name in inflight_models
                ):
                    continue
                middle_pods = _effective_assigned_replicas(middle.model_name, model_contexts, model_replicas)
                if middle_pods <= cfg.min_replicas_per_model:
                    continue
                planned_take = abs(min(deltas.get(middle.model_name, 0), 0))
                transfer = min(
                    still_needed,
                    _scale_step(middle_pods, cfg.scale_step_ratio),
                    max(0, middle_pods - planned_take - cfg.min_replicas_per_model),
                )
                if transfer <= 0:
                    continue
                _add_scale_action(
                    actions,
                    deltas,
                    model=middle.model_name,
                    delta=-transfer,
                    reason="critical_middle_zone_safescale",
                    source_loop="rescue",
                    requires_safescale=True,
                    donor=middle.model_name,
                    receiver=recv.model_name,
                )
                delayed_down_models.add(middle.model_name)
                pending = probe_upscale_plans.setdefault(middle.model_name, {})
                pending[recv.model_name] = pending.get(recv.model_name, 0) + transfer
                still_needed -= transfer

        for idle in idle_models:
            if idle.model_name in active_probe_models or idle.model_name in inflight_models:
                continue
            if deltas.get(idle.model_name, 0) != 0:
                continue
            pods = _effective_assigned_replicas(idle.model_name, model_contexts, model_replicas)
            if pods <= cfg.min_replicas_per_model:
                continue
            shrink = min(_scale_step(pods, cfg.scale_step_ratio), pods - cfg.min_replicas_per_model)
            if shrink > 0:
                _add_scale_action(
                    actions,
                    deltas,
                    model=idle.model_name,
                    delta=-shrink,
                    reason="idle_proactive_immediate",
                    source_loop="rescue",
                    donor=idle.model_name,
                )

        for high in high_models:
            if high.model_name in active_probe_models or high.model_name in inflight_models:
                continue
            if deltas.get(high.model_name, 0) != 0:
                continue
            pods = _effective_assigned_replicas(high.model_name, model_contexts, model_replicas)
            if pods <= cfg.min_replicas_per_model:
                continue
            shrink = min(_scale_step(pods, cfg.scale_step_ratio), pods - cfg.min_replicas_per_model)
            if shrink > 0:
                _add_scale_action(
                    actions,
                    deltas,
                    model=high.model_name,
                    delta=-shrink,
                    reason="high_proactive_safescale",
                    source_loop="rescue",
                    requires_safescale=True,
                    donor=high.model_name,
                )
                delayed_down_models.add(high.model_name)
    else:
        events.append("rescue_skipped_by_cadence")

    if not cfg.fairness_due:
        events.append("fairness_skipped_by_cadence")
        return PlanResult(actions, delayed_down_models, probe_upscale_plans, events=events)

    for recv in low_receivers:
        if recv.model_name in inflight_models:
            continue
        recv_pods = _effective_assigned_replicas(recv.model_name, model_contexts, model_replicas)
        receiver_capacity = cfg.max_replicas_per_model - recv_pods - max(0, deltas.get(recv.model_name, 0))
        if receiver_capacity <= 0:
            continue
        needed = min(_scale_step(recv_pods, cfg.scale_step_ratio), receiver_capacity)

        if remaining_idle > 0:
            idle_gain = min(needed, remaining_idle)
            if idle_gain > 0:
                _add_scale_action(
                    actions,
                    deltas,
                    model=recv.model_name,
                    delta=idle_gain,
                    reason="low_fairness_idle_capacity",
                    source_loop="fairness",
                    receiver=recv.model_name,
                )
                remaining_idle -= idle_gain
                needed -= idle_gain
        if needed <= 0:
            continue
        if not bool(model_contexts.get(recv.model_name, {}).get("is_saturated", False)):
            events.append(f"fairness_blocked_unsaturated:{recv.model_name}")
            continue

        for donor in paper_donors:
            if needed <= 0:
                break
            if (
                donor.model_name == recv.model_name
                or donor.model_name in active_probe_models
                or donor.model_name in inflight_models
                or donor.state not in (ModelState.IDLE, ModelState.HIGH)
            ):
                continue
            donor_pods = _effective_assigned_replicas(donor.model_name, model_contexts, model_replicas)
            if donor_pods <= cfg.min_replicas_per_model:
                continue
            existing_shrink = abs(min(deltas.get(donor.model_name, 0), 0))
            existing_claimed = sum(probe_upscale_plans.get(donor.model_name, {}).values())
            unclaimed = existing_shrink - existing_claimed
            if unclaimed > 0 and donor.model_name in delayed_down_models:
                piggyback = min(needed, unclaimed)
                pending = probe_upscale_plans.setdefault(donor.model_name, {})
                pending[recv.model_name] = pending.get(recv.model_name, 0) + piggyback
                needed -= piggyback
                if needed <= 0:
                    continue
            planned_take = abs(min(deltas.get(donor.model_name, 0), 0))
            transfer = min(
                needed,
                _scale_step(donor_pods, cfg.scale_step_ratio),
                max(0, donor_pods - planned_take - cfg.min_replicas_per_model),
            )
            if transfer <= 0:
                continue
            _add_scale_action(
                actions,
                deltas,
                model=donor.model_name,
                delta=-transfer,
                reason="low_fairness_donor_immediate",
                source_loop="fairness",
                donor=donor.model_name,
                receiver=recv.model_name,
            )
            _add_scale_action(
                actions,
                deltas,
                model=recv.model_name,
                delta=transfer,
                reason="low_fairness_donor_immediate",
                source_loop="fairness",
                donor=donor.model_name,
                receiver=recv.model_name,
            )
            needed -= transfer

        for middle in middle_zone:
            if needed <= 0:
                break
            if middle.model_name == recv.model_name or middle.model_name in active_probe_models or middle.model_name in inflight_models:
                continue
            donor_pods = _effective_assigned_replicas(middle.model_name, model_contexts, model_replicas)
            if donor_pods <= cfg.min_replicas_per_model:
                continue
            existing_shrink = abs(min(deltas.get(middle.model_name, 0), 0))
            existing_claimed = sum(probe_upscale_plans.get(middle.model_name, {}).values())
            unclaimed = existing_shrink - existing_claimed
            if unclaimed > 0 and middle.model_name in delayed_down_models:
                piggyback = min(needed, unclaimed)
                pending = probe_upscale_plans.setdefault(middle.model_name, {})
                pending[recv.model_name] = pending.get(recv.model_name, 0) + piggyback
                needed -= piggyback
                if needed <= 0:
                    continue
            planned_take = abs(min(deltas.get(middle.model_name, 0), 0))
            transfer = min(
                needed,
                _scale_step(donor_pods, cfg.scale_step_ratio),
                max(0, donor_pods - planned_take - cfg.min_replicas_per_model),
            )
            if transfer <= 0:
                continue
            _add_scale_action(
                actions,
                deltas,
                model=middle.model_name,
                delta=-transfer,
                reason="low_fairness_middle_zone_safescale",
                source_loop="fairness",
                requires_safescale=True,
                donor=middle.model_name,
                receiver=recv.model_name,
            )
            delayed_down_models.add(middle.model_name)
            pending = probe_upscale_plans.setdefault(middle.model_name, {})
            pending[recv.model_name] = pending.get(recv.model_name, 0) + transfer
            needed -= transfer

    return PlanResult(actions, delayed_down_models, probe_upscale_plans, events=events)


def _try_plan_tp_capacity(
    actions: list[Action],
    *,
    model: str,
    tp_size: int,
    cluster_view: ClusterView,
    events: list[str],
    source_loop: SourceLoop,
) -> str | None:
    allocator = SlotAllocator(cluster_view.topology, list(cluster_view.bindings))
    if allocator.find_slot(tp_size) is not None:
        return "critical_empty_slot"

    migrations = allocator.plan_defrag(tp_size)
    if migrations:
        actions.append(
            DefragAction(
                migrations=tuple(migrations),
                reason="critical_tp_defrag",
                source_loop=source_loop,
            )
        )
        return "critical_tp_defrag"

    events.append(f"capacity_blocked:{model}")
    return None


def _paper_state_incomplete(classifications: list[ModelClassification]) -> bool:
    if not classifications:
        return True
    return any(item.state == ModelState.UNKNOWN or item.Z_m is None and item.state != ModelState.IDLE for item in classifications)


def _add_scale_action(
    actions: list[Action],
    deltas: dict[str, int],
    *,
    model: str,
    delta: int,
    reason: str,
    source_loop: SourceLoop,
    requires_safescale: bool = False,
    receiver: str | None = None,
    donor: str | None = None,
) -> None:
    if delta == 0:
        return
    deltas[model] = deltas.get(model, 0) + delta
    actions.append(
        ScaleAction(
            model=model,
            delta=delta,
            reason=reason,
            source_loop=source_loop,
            requires_safescale=requires_safescale,
            receiver=receiver,
            donor=donor,
        )
    )


def _scale_step(current_pods: int, ratio: float = 0.1) -> int:
    if current_pods <= 0:
        return 1
    return max(1, math.ceil(ratio * current_pods))


def _effective_assigned_replicas(
    model_name: str,
    model_contexts: dict[str, dict[str, Any]],
    model_replicas: dict[str, int],
) -> int:
    ctx = model_contexts.get(model_name, {})
    assigned = model_replicas.get(model_name, ctx.get("assigned_replicas", ctx.get("routable_pods", 1)))
    try:
        return max(1, int(assigned))
    except Exception:
        return 1

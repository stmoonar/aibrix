from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from tre_common.registry import ClusterTopology
from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, donor_mock_cost_key
from tre_sm.allocator.slots import Binding, Slot, SlotAllocator

SourceLoop = Literal["rescue", "fairness"]
IncompletePolicy = Literal["drop_model", "drop_all"]


@dataclass(frozen=True)
class PlanConfig:
    min_replicas_per_model: int
    max_replicas_per_model: int
    scale_step_ratio: float = 0.1
    rescue_due: bool = True
    fairness_due: bool = True
    model_tp_sizes: dict[str, int] = field(default_factory=dict)
    min_replicas_by_model: dict[str, int] = field(default_factory=dict)
    max_replicas_by_model: dict[str, int] = field(default_factory=dict)
    incomplete_policy: IncompletePolicy = "drop_model"


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


@dataclass(frozen=True)
class ShrinkForSlotAction:
    donor: str
    beneficiary: str
    serve_id: str
    slot: Slot
    reason: str
    source_loop: SourceLoop


Action = ScaleAction | HideAction | UnhideAction | DefragAction | ShrinkForSlotAction


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
    slot_shrink_donors: set[str] = set()

    incomplete_models = _paper_state_incomplete_models(classifications)
    if not classifications or (incomplete_models and cfg.incomplete_policy == "drop_all"):
        return PlanResult(
            actions=[],
            delayed_down_models=set(),
            probe_upscale_plans={},
            dropped_legacy_raw_trs=True,
            events=["paper_state_incomplete_drop_legacy_raw_trs"],
        )
    if incomplete_models:
        events.extend(f"paper_state_incomplete_drop:{model}" for model in incomplete_models)
        classifications = [item for item in classifications if item.model_name not in incomplete_models]

    # F-onset warmup guard: a receiver (CRITICAL/LOW) whose signal is not yet 'warm'
    # (the sliding window still straddles this model's traffic onset -> TRS structurally
    # low -> false CRITICAL/LOW) is suppressed for this tick, UNLESS it is genuinely
    # saturated (Q_ctl >= qsat) -- the saturation bypass keeps a real flash crowd honoured.
    # Applies to both receiver states, so it is consistent across rescue and fairness.
    warmup_suppressed: list[str] = []
    kept: list = []
    for item in classifications:
        if item.role == ModelRole.RECEIVER:
            ctx = model_contexts.get(item.model_name, {})
            if not ctx.get("signal_warm", True) and not ctx.get("is_saturated", False):
                warmup_suppressed.append(item.model_name)
                continue
        kept.append(item)
    if warmup_suppressed:
        events.extend(f"receiver_suppressed_signal_warmup:{model}" for model in warmup_suppressed)
        classifications = kept

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
            recv_pods = _effective_routable_replicas(recv.model_name, model_contexts, model_replicas)
            recv_assigned = _effective_assigned_replicas(recv.model_name, model_contexts, model_replicas)
            recv_max = _max_replicas(cfg, recv.model_name)
            if recv_pods >= recv_max:
                continue
            raw_need = min(_scale_step(recv_pods, cfg.scale_step_ratio), recv_max - recv_pods)
            if raw_need <= 0:
                continue

            gain_from_sleeping = min(raw_need, max(0, recv_assigned - recv_pods))
            if gain_from_sleeping > 0:
                _add_scale_action(
                    actions,
                    deltas,
                    model=recv.model_name,
                    delta=gain_from_sleeping,
                    reason="critical_sleeping_capacity",
                    source_loop="rescue",
                    receiver=recv.model_name,
                )
                raw_need -= gain_from_sleeping
                if raw_need <= 0:
                    continue

            tp_size = cfg.model_tp_sizes.get(recv.model_name, 1)
            if tp_size > 1 and cluster_view is not None:
                same_slot_shrink = _try_plan_same_slot_high_shrink(
                    classifications=classifications,
                    model_contexts=model_contexts,
                    model_replicas=model_replicas,
                    cfg=cfg,
                    cluster_view=cluster_view,
                    receiver=recv.model_name,
                    active_probe_models=active_probe_models,
                    inflight_models=inflight_models,
                    source_loop="rescue",
                )
                if same_slot_shrink is not None:
                    actions.append(same_slot_shrink)
                    slot_shrink_donors.add(same_slot_shrink.donor)
                    delayed_down_models.add(same_slot_shrink.donor)
                    continue

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
                donor_pods = _effective_routable_replicas(donor.model_name, model_contexts, model_replicas)
                donor_min = _min_replicas(cfg, donor.model_name)
                if donor_pods <= donor_min:
                    continue
                planned_take = abs(min(deltas.get(donor.model_name, 0), 0))
                transfer = min(
                    still_needed,
                    _scale_step(donor_pods, cfg.scale_step_ratio),
                    max(0, donor_pods - planned_take - donor_min),
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
                middle_pods = _effective_routable_replicas(middle.model_name, model_contexts, model_replicas)
                middle_min = _min_replicas(cfg, middle.model_name)
                if middle_pods <= middle_min:
                    continue
                planned_take = abs(min(deltas.get(middle.model_name, 0), 0))
                transfer = min(
                    still_needed,
                    _scale_step(middle_pods, cfg.scale_step_ratio),
                    max(0, middle_pods - planned_take - middle_min),
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
            pods = _effective_routable_replicas(idle.model_name, model_contexts, model_replicas)
            idle_min = _serving_floor(cfg, idle.model_name, model_contexts, model_replicas)
            if pods <= idle_min:
                continue
            shrink = min(_scale_step(pods, cfg.scale_step_ratio), pods - idle_min)
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
            if high.model_name in slot_shrink_donors:
                continue
            if high.model_name in active_probe_models or high.model_name in inflight_models:
                continue
            if deltas.get(high.model_name, 0) != 0:
                continue
            pods = _effective_routable_replicas(high.model_name, model_contexts, model_replicas)
            high_min = _serving_floor(cfg, high.model_name, model_contexts, model_replicas)
            if pods <= high_min:
                continue
            shrink = min(_scale_step(pods, cfg.scale_step_ratio), pods - high_min)
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
        recv_pods = _effective_routable_replicas(recv.model_name, model_contexts, model_replicas)
        recv_assigned = _effective_assigned_replicas(recv.model_name, model_contexts, model_replicas)
        receiver_capacity = _max_replicas(cfg, recv.model_name) - recv_pods - max(0, deltas.get(recv.model_name, 0))
        if receiver_capacity <= 0:
            continue
        needed = min(_scale_step(recv_pods, cfg.scale_step_ratio), receiver_capacity)

        sleeping_gain = min(needed, max(0, recv_assigned - recv_pods))
        if sleeping_gain > 0:
            _add_scale_action(
                actions,
                deltas,
                model=recv.model_name,
                delta=sleeping_gain,
                reason="low_fairness_sleeping_capacity",
                source_loop="fairness",
                receiver=recv.model_name,
            )
            needed -= sleeping_gain

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
            donor_pods = _effective_routable_replicas(donor.model_name, model_contexts, model_replicas)
            donor_min = _min_replicas(cfg, donor.model_name)
            if donor_pods <= donor_min:
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
                max(0, donor_pods - planned_take - donor_min),
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
            donor_pods = _effective_routable_replicas(middle.model_name, model_contexts, model_replicas)
            donor_min = _min_replicas(cfg, middle.model_name)
            if donor_pods <= donor_min:
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
                max(0, donor_pods - planned_take - donor_min),
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


def _try_plan_same_slot_high_shrink(
    *,
    classifications: list[ModelClassification],
    model_contexts: dict[str, dict[str, Any]],
    model_replicas: dict[str, int],
    cfg: PlanConfig,
    cluster_view: ClusterView,
    receiver: str,
    active_probe_models: set[str],
    inflight_models: set[str],
    source_loop: SourceLoop,
) -> ShrinkForSlotAction | None:
    high_by_model = {item.model_name: item for item in classifications if item.state == ModelState.HIGH}
    candidates: list[tuple[float, Binding]] = []
    occupied = {(binding.slot.node, gpu) for binding in cluster_view.bindings for gpu in binding.slot.gpu_ids}

    for binding in cluster_view.bindings:
        high = high_by_model.get(binding.model)
        if high is None or binding.model in active_probe_models or binding.model in inflight_models:
            continue
        if binding.model == receiver or len(binding.slot.gpu_ids) != 1:
            continue
        donor_pods = _effective_routable_replicas(binding.model, model_contexts, model_replicas)
        if donor_pods <= _min_replicas(cfg, binding.model):
            continue
        if not _slot_mate_is_free(cluster_view, binding.slot, occupied):
            continue
        candidates.append((high.Z_m if high.Z_m is not None else math.inf, binding))

    if not candidates:
        return None

    _, donor_binding = min(candidates, key=lambda item: (item[0], item[1].serve_id))
    return ShrinkForSlotAction(
        donor=donor_binding.model,
        beneficiary=receiver,
        serve_id=donor_binding.serve_id,
        slot=donor_binding.slot,
        reason="critical_same_slot_high_shrink",
        source_loop=source_loop,
    )


def _slot_mate_is_free(
    cluster_view: ClusterView,
    slot: Slot,
    occupied: set[tuple[str, int]],
) -> bool:
    gpu = slot.gpu_ids[0]
    for node in cluster_view.topology.nodes:
        if node.name != slot.node:
            continue
        for pair in node.two_gpu_slots:
            pair = tuple(pair)
            if gpu not in pair:
                continue
            mate = next(item for item in pair if item != gpu)
            return (slot.node, mate) not in occupied
    return False


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


def _paper_state_incomplete_models(classifications: list[ModelClassification]) -> tuple[str, ...]:
    return tuple(
        item.model_name
        for item in classifications
        if item.state == ModelState.UNKNOWN or (item.Z_m is None and item.state != ModelState.IDLE)
    )


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


def _min_replicas(cfg: PlanConfig, model_name: str) -> int:
    return cfg.min_replicas_by_model.get(model_name, cfg.min_replicas_per_model)


def _serving_floor(
    cfg: PlanConfig,
    model_name: str,
    model_contexts: Mapping[str, Mapping[str, float | int | None]],
    model_replicas: Mapping[str, int],
) -> int:
    configured_min = _min_replicas(cfg, model_name)
    context = model_contexts.get(model_name, {})
    bound_replicas = int(context.get("assigned_replicas") or model_replicas.get(model_name, 0) or 0)
    if bound_replicas > 0:
        return max(configured_min, 1)
    return configured_min


def _max_replicas(cfg: PlanConfig, model_name: str) -> int:
    return cfg.max_replicas_by_model.get(model_name, cfg.max_replicas_per_model)


def _effective_routable_replicas(
    model_name: str,
    model_contexts: dict[str, dict[str, Any]],
    model_replicas: dict[str, int],
) -> int:
    ctx = model_contexts.get(model_name, {})
    routable = ctx.get("routable_pods")
    if routable is None:
        routable = model_replicas.get(model_name, ctx.get("assigned_replicas", 1))
    try:
        return max(0, int(routable))
    except Exception:
        return 1


def _effective_assigned_replicas(
    model_name: str,
    model_contexts: dict[str, dict[str, Any]],
    model_replicas: dict[str, int],
) -> int:
    ctx = model_contexts.get(model_name, {})
    assigned = model_replicas.get(model_name, ctx.get("assigned_replicas", ctx.get("routable_pods", 1)))
    try:
        return max(0, int(assigned))
    except Exception:
        return 1

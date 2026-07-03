from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LegacyModelState(str, Enum):
    CRITICAL = "critical"
    LOW = "low"
    HEALTHY = "healthy"
    HIGH = "high"
    IDLE = "idle"
    UNKNOWN = "unknown"


class LegacyModelRole(str, Enum):
    RECEIVER = "receiver"
    DONOR = "donor"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LegacyClassification:
    model_name: str
    state: LegacyModelState
    role: LegacyModelRole
    Z_m: Optional[float]
    donor_tier: Optional[str] = None


@dataclass(frozen=True)
class LegacyPlan:
    deltas: dict[str, int]
    delayed_down_models: set[str]
    probe_upscale_plans: dict[str, dict[str, int]]


def legacy_scale_step(current_pods: int, ratio: float = 0.1) -> int:
    if current_pods <= 0:
        return 1
    return max(1, math.ceil(ratio * current_pods))


def legacy_build_paper_plan(
    *,
    classifications: list[LegacyClassification],
    model_contexts: dict[str, dict],
    model_replicas: dict[str, int],
    idle_gpus: int,
    min_replicas_per_model: int,
    max_replicas_per_model: int,
    active_probe_models: set[str] | None = None,
    scale_step_ratio: float = 0.1,
    rescue_due: bool = True,
    fairness_due: bool = True,
) -> LegacyPlan:
    active_probe_models = active_probe_models or set()
    deltas: dict[str, int] = {}
    delayed_down_models: set[str] = set()
    probe_upscale_plans: dict[str, dict[str, int]] = {}
    remaining_idle = idle_gpus

    critical_receivers = [item for item in classifications if item.state == LegacyModelState.CRITICAL]
    low_receivers = [item for item in classifications if item.state == LegacyModelState.LOW]
    high_models = [item for item in classifications if item.state == LegacyModelState.HIGH]
    idle_models = [item for item in classifications if item.state == LegacyModelState.IDLE]
    paper_donors = [item for item in classifications if item.role == LegacyModelRole.DONOR]
    middle_zone = [
        item for item in classifications
        if item.state in (LegacyModelState.LOW, LegacyModelState.HEALTHY)
        and item.role != LegacyModelRole.RECEIVER
    ]
    middle_zone.sort(key=lambda item: (0 if item.state == LegacyModelState.HEALTHY else 1, -(item.Z_m or 0.0)))

    if rescue_due:
        for recv in critical_receivers:
            recv_pods = _effective_assigned(recv.model_name, model_contexts, model_replicas)
            if recv_pods >= max_replicas_per_model:
                continue
            raw_need = min(legacy_scale_step(recv_pods, scale_step_ratio), max_replicas_per_model - recv_pods)
            if raw_need <= 0:
                continue
            gain_from_idle = min(raw_need, remaining_idle) if remaining_idle > 0 else 0
            if gain_from_idle > 0:
                deltas[recv.model_name] = deltas.get(recv.model_name, 0) + gain_from_idle
                remaining_idle -= gain_from_idle
            still_needed = raw_need - gain_from_idle
            for donor in paper_donors:
                if still_needed <= 0:
                    break
                if donor.model_name == recv.model_name or donor.model_name in active_probe_models or donor.state not in (LegacyModelState.IDLE, LegacyModelState.HIGH):
                    continue
                donor_pods = _effective_assigned(donor.model_name, model_contexts, model_replicas)
                if donor_pods <= min_replicas_per_model:
                    continue
                planned_take = abs(min(deltas.get(donor.model_name, 0), 0))
                transfer = min(still_needed, legacy_scale_step(donor_pods, scale_step_ratio), max(0, donor_pods - planned_take - min_replicas_per_model))
                if transfer <= 0:
                    continue
                deltas[donor.model_name] = deltas.get(donor.model_name, 0) - transfer
                deltas[recv.model_name] = deltas.get(recv.model_name, 0) + transfer
                still_needed -= transfer
            for middle in middle_zone:
                if still_needed <= 0:
                    break
                if middle.model_name == recv.model_name or middle.model_name in active_probe_models:
                    continue
                middle_pods = _effective_assigned(middle.model_name, model_contexts, model_replicas)
                if middle_pods <= min_replicas_per_model:
                    continue
                planned_take = abs(min(deltas.get(middle.model_name, 0), 0))
                transfer = min(still_needed, legacy_scale_step(middle_pods, scale_step_ratio), max(0, middle_pods - planned_take - min_replicas_per_model))
                if transfer <= 0:
                    continue
                deltas[middle.model_name] = deltas.get(middle.model_name, 0) - transfer
                delayed_down_models.add(middle.model_name)
                pending = probe_upscale_plans.setdefault(middle.model_name, {})
                pending[recv.model_name] = pending.get(recv.model_name, 0) + transfer
                still_needed -= transfer

        for idle in idle_models:
            if idle.model_name in active_probe_models or deltas.get(idle.model_name, 0) != 0:
                continue
            pods = _effective_assigned(idle.model_name, model_contexts, model_replicas)
            if pods <= min_replicas_per_model:
                continue
            shrink = min(legacy_scale_step(pods, scale_step_ratio), pods - min_replicas_per_model)
            if shrink > 0:
                deltas[idle.model_name] = -shrink

        for high in high_models:
            if high.model_name in active_probe_models or deltas.get(high.model_name, 0) != 0:
                continue
            pods = _effective_assigned(high.model_name, model_contexts, model_replicas)
            if pods <= min_replicas_per_model:
                continue
            shrink = min(legacy_scale_step(pods, scale_step_ratio), pods - min_replicas_per_model)
            if shrink > 0:
                deltas[high.model_name] = deltas.get(high.model_name, 0) - shrink
                delayed_down_models.add(high.model_name)

    if not fairness_due:
        return LegacyPlan(deltas, delayed_down_models, probe_upscale_plans)

    for recv in low_receivers:
        recv_pods = _effective_assigned(recv.model_name, model_contexts, model_replicas)
        receiver_capacity = max_replicas_per_model - recv_pods - max(0, deltas.get(recv.model_name, 0))
        if receiver_capacity <= 0:
            continue
        needed = min(legacy_scale_step(recv_pods, scale_step_ratio), receiver_capacity)
        if remaining_idle > 0:
            idle_gain = min(needed, remaining_idle)
            if idle_gain > 0:
                deltas[recv.model_name] = deltas.get(recv.model_name, 0) + idle_gain
                remaining_idle -= idle_gain
                needed -= idle_gain
        if needed <= 0 or not bool(model_contexts.get(recv.model_name, {}).get("is_saturated", False)):
            continue
        for donor in paper_donors:
            if needed <= 0:
                break
            if donor.model_name == recv.model_name or donor.model_name in active_probe_models or donor.state not in (LegacyModelState.IDLE, LegacyModelState.HIGH):
                continue
            donor_pods = _effective_assigned(donor.model_name, model_contexts, model_replicas)
            if donor_pods <= min_replicas_per_model:
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
            transfer = min(needed, legacy_scale_step(donor_pods, scale_step_ratio), max(0, donor_pods - planned_take - min_replicas_per_model))
            if transfer <= 0:
                continue
            deltas[donor.model_name] = deltas.get(donor.model_name, 0) - transfer
            deltas[recv.model_name] = deltas.get(recv.model_name, 0) + transfer
            needed -= transfer
        for middle in middle_zone:
            if needed <= 0:
                break
            if middle.model_name == recv.model_name or middle.model_name in active_probe_models:
                continue
            donor_pods = _effective_assigned(middle.model_name, model_contexts, model_replicas)
            if donor_pods <= min_replicas_per_model:
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
            transfer = min(needed, legacy_scale_step(donor_pods, scale_step_ratio), max(0, donor_pods - planned_take - min_replicas_per_model))
            if transfer <= 0:
                continue
            deltas[middle.model_name] = deltas.get(middle.model_name, 0) - transfer
            delayed_down_models.add(middle.model_name)
            pending = probe_upscale_plans.setdefault(middle.model_name, {})
            pending[recv.model_name] = pending.get(recv.model_name, 0) + transfer
            needed -= transfer

    return LegacyPlan(deltas, delayed_down_models, probe_upscale_plans)


def _effective_assigned(model: str, model_contexts: dict[str, dict], model_replicas: dict[str, int]) -> int:
    ctx = model_contexts.get(model, {})
    assigned = model_replicas.get(model, ctx.get("assigned_replicas", ctx.get("routable_pods", 1)))
    try:
        return max(1, int(assigned))
    except Exception:
        return 1

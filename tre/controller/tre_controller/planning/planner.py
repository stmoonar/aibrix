from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from tre_common.registry import ClusterTopology
from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, donor_mock_cost_key
from tre_controller.planning.util_scale_down import UtilWindow, util_scale_down_ready
from tre_sm.allocator.slots import Binding, Slot, SlotAllocator, natural_key

SourceLoop = Literal["rescue", "fairness", "safescale"]
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
    # t1 diagnosis (TRE vs APA): the rescue-loop "high_proactive_safescale" block used to
    # hide a serving pod of any HIGH model as a speculative surplus-reclaim probe -- with
    # NO demand-driven receiver. During a traffic spike a busy model is momentarily HIGH
    # (TSS = throughput/queue spikes up), so the probe hid a serving pod exactly as load
    # climbed, deepening saturation (routable 4->3, then CRITICAL->rescale oscillation).
    # When enabled (default) this suppresses that receiver-less proactive probe on hot
    # (HIGH/CRITICAL) donors. Demand-driven preemption (idle/HIGH immediate donors and the
    # TP critical_same_slot_high_shrink -> CRITICAL beneficiary) is a separate path and is
    # intentionally NOT gated. Ablate via TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE=0.
    suppress_hot_proactive_probe: bool = True
    disable_eta_gate: bool = False
    # Utilisation-gated scale-down (TRE_UTIL_SCALE_DOWN). Off by default here; the
    # controller enables it via run_planner_tick when a UtilScaleDown tracker is wired.
    util_scale_down: bool = False
    util_scale_down_windows: int = 6
    scale_down_q_per_replica_by_model: dict[str, float] = field(default_factory=dict)


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
    # Explicit binding identities (serve_id == pod name). For a negative delta the
    # dispatcher sleeps exactly these bindings (binding-level power) instead of letting
    # the SM pick the tail of the model; for a safescale probe they are the pods to hide.
    pods: tuple[str, ...] = ()


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
    cooldowns: Mapping[str, str] | None = None,
    util_windows: Mapping[str, tuple[UtilWindow, ...]] | None = None,
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
    # Sleeping-capacity deadlock fix: GPU-slot occupancy so a receiver's sleeping binding
    # only counts as wakeable capacity when its slot has no awake binding (of any model).
    occupancy = _SlotOccupancy(cluster_view) if cluster_view is not None else None
    # Review F4 per-model cooldown: model -> direction ("up"/"down") of its last executed
    # action whose effect the decision window does not yet fully reflect.
    cooldown = _Cooldown(cooldowns or {}, events)

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
    # low -> false CRITICAL/LOW) is suppressed for this tick. Applies to both receiver
    # states, so it is consistent across rescue and fairness.
    # ADR-0014: the former saturation bypass ("unless Q_ctl >= qsat") was removed. Warmup
    # suppression is now unconditional; a genuine flash crowd in the warmup window is
    # delayed at most one window (until the sliding window clears the traffic onset).
    warmup_suppressed: list[str] = []
    kept: list = []
    for item in classifications:
        if item.role == ModelRole.RECEIVER:
            ctx = model_contexts.get(item.model_name, {})
            if not ctx.get("signal_warm", True):
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
    paper_donors.sort(
        key=lambda item: donor_mock_cost_key(
            item, disable_eta_gate=cfg.disable_eta_gate
        )
    )
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
            if cooldown.blocks(recv.model_name, "up", critical=True):
                continue
            recv_pods = _effective_routable_replicas(recv.model_name, model_contexts, model_replicas)
            recv_assigned = _effective_assigned_replicas(recv.model_name, model_contexts, model_replicas)
            recv_max = _max_replicas(cfg, recv.model_name)
            if recv_pods >= recv_max:
                continue
            raw_need = min(_scale_step(recv_pods, cfg.scale_step_ratio), recv_max - recv_pods)
            if raw_need <= 0:
                continue

            gain_from_sleeping, wake_pods = _plan_sleeping_wakes(
                occupancy,
                receiver=recv.model_name,
                need=min(raw_need, max(0, recv_assigned - recv_pods)),
                events=events,
                blocked_event="critical_sleeping_blocked",
            )
            if gain_from_sleeping > 0:
                _add_scale_action(
                    actions,
                    deltas,
                    model=recv.model_name,
                    delta=gain_from_sleeping,
                    reason="critical_sleeping_capacity",
                    source_loop="rescue",
                    receiver=recv.model_name,
                    pods=wake_pods,
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
                    inflight_models=inflight_models | cooldown.down_blocked(),
                    source_loop="rescue",
                )
                if same_slot_shrink is not None:
                    # t1: this IS a legitimate scale-down probe on a hot (HIGH) donor, but
                    # it is demand-driven preemption -- a CRITICAL receiver blocked on a
                    # fragmented TP slot and no idle capacity. Keep it (the suppress-hot
                    # guard only gates the receiver-less proactive path) and log the
                    # preemption reason explicitly in the decision stream.
                    actions.append(same_slot_shrink)
                    slot_shrink_donors.add(same_slot_shrink.donor)
                    delayed_down_models.add(same_slot_shrink.donor)
                    events.append(
                        f"safescale_preemption:{same_slot_shrink.donor}->{recv.model_name}:{same_slot_shrink.reason}"
                    )
                    continue

                tp_planned = _try_plan_tp_capacity(
                    actions,
                    model=recv.model_name,
                    tp_size=tp_size,
                    cluster_view=cluster_view,
                    events=events,
                    source_loop="rescue",
                    occupancy=occupancy,
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

            if occupancy is not None:
                gain_from_idle = _plan_create_capacity(
                    occupancy,
                    receiver=recv.model_name,
                    tp_size=tp_size,
                    need=raw_need,
                    events=events,
                    blocked_event="critical_idle_unusable",
                )
            else:
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
            for donor in _slot_matched_first(paper_donors, occupancy, recv.model_name):
                if still_needed <= 0:
                    break
                if (
                    donor.model_name == recv.model_name
                    or donor.model_name in active_probe_models
                    or donor.model_name in inflight_models
                    or donor.state not in (ModelState.IDLE, ModelState.HIGH)
                ):
                    continue
                if cooldown.blocks(donor.model_name, "down"):
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
                transfer, donor_slot_pods, receiver_slot_pods = _slot_targeted_transfer(
                    occupancy, donor=donor.model_name, receiver=recv.model_name, transfer=transfer, events=events
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
                    pods=donor_slot_pods,
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
                    pods=receiver_slot_pods,
                )
                still_needed -= transfer

            for middle in _slot_matched_first(middle_zone, occupancy, recv.model_name):
                if still_needed <= 0:
                    break
                if (
                    middle.model_name == recv.model_name
                    or middle.model_name in active_probe_models
                    or middle.model_name in inflight_models
                ):
                    continue
                if cooldown.blocks(middle.model_name, "down"):
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
                transfer, middle_slot_pods, _ = _slot_targeted_transfer(
                    occupancy, donor=middle.model_name, receiver=recv.model_name, transfer=transfer, events=events
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
                    pods=middle_slot_pods,
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
            if cooldown.blocks(idle.model_name, "down"):
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
            if cooldown.blocks(high.model_name, "down"):
                continue
            pods = _effective_routable_replicas(high.model_name, model_contexts, model_replicas)
            high_min = _serving_floor(cfg, high.model_name, model_contexts, model_replicas)
            if pods <= high_min:
                continue
            shrink = min(_scale_step(pods, cfg.scale_step_ratio), pods - high_min)
            if shrink > 0:
                # t1 guard: never launch a receiver-less proactive scale-down probe on a
                # hot (HIGH) donor. It has no beneficiary and, during a spike, hiding a
                # serving pod is pro-cyclical (see PlanConfig.suppress_hot_proactive_probe).
                if cfg.suppress_hot_proactive_probe:
                    events.append(f"safescale_probe_suppressed_hot:{high.model_name}")
                    continue
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
        if cooldown.blocks(recv.model_name, "up"):
            continue
        recv_pods = _effective_routable_replicas(recv.model_name, model_contexts, model_replicas)
        recv_assigned = _effective_assigned_replicas(recv.model_name, model_contexts, model_replicas)
        receiver_capacity = _max_replicas(cfg, recv.model_name) - recv_pods - max(0, deltas.get(recv.model_name, 0))
        if receiver_capacity <= 0:
            continue
        needed = min(_scale_step(recv_pods, cfg.scale_step_ratio), receiver_capacity)

        sleeping_gain, wake_pods = _plan_sleeping_wakes(
            occupancy,
            receiver=recv.model_name,
            need=min(needed, max(0, recv_assigned - recv_pods)),
            events=events,
            blocked_event="low_fairness_sleeping_blocked",
        )
        if sleeping_gain > 0:
            _add_scale_action(
                actions,
                deltas,
                model=recv.model_name,
                delta=sleeping_gain,
                reason="low_fairness_sleeping_capacity",
                source_loop="fairness",
                receiver=recv.model_name,
                pods=wake_pods,
            )
            needed -= sleeping_gain

        if occupancy is not None:
            # P1-a: only slot groups the SM will really create into for this receiver
            # (TP-aware; none while it still has a sleeping binding the SM would try first).
            idle_gain = _plan_create_capacity(
                occupancy,
                receiver=recv.model_name,
                tp_size=cfg.model_tp_sizes.get(recv.model_name, 1),
                need=needed,
                events=events,
                blocked_event="low_fairness_idle_unusable",
            )
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
                needed -= idle_gain
        elif remaining_idle > 0:
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
        # ADR-0014: fairness receiver eligibility is z_m-band only (CRITICAL/LOW). The
        # former saturation gate (emit "fairness_blocked_unsaturated" unless Q_ctl >= qsat)
        # was removed -- a non-saturated LOW/CRITICAL receiver now receives donor surplus.

        for donor in _slot_matched_first(paper_donors, occupancy, recv.model_name):
            if needed <= 0:
                break
            if (
                donor.model_name == recv.model_name
                or donor.model_name in active_probe_models
                or donor.model_name in inflight_models
                or donor.state not in (ModelState.IDLE, ModelState.HIGH)
            ):
                continue
            if cooldown.blocks(donor.model_name, "down"):
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
            transfer, donor_slot_pods, receiver_slot_pods = _slot_targeted_transfer(
                occupancy, donor=donor.model_name, receiver=recv.model_name, transfer=transfer, events=events
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
                pods=donor_slot_pods,
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
                pods=receiver_slot_pods,
            )
            needed -= transfer

        for middle in _slot_matched_first(middle_zone, occupancy, recv.model_name):
            if needed <= 0:
                break
            if middle.model_name == recv.model_name or middle.model_name in active_probe_models or middle.model_name in inflight_models:
                continue
            if cooldown.blocks(middle.model_name, "down"):
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
            transfer, middle_slot_pods, _ = _slot_targeted_transfer(
                occupancy, donor=middle.model_name, receiver=recv.model_name, transfer=transfer, events=events
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
                pods=middle_slot_pods,
            )
            delayed_down_models.add(middle.model_name)
            pending = probe_upscale_plans.setdefault(middle.model_name, {})
            pending[recv.model_name] = pending.get(recv.model_name, 0) + transfer
            needed -= transfer

    if cfg.util_scale_down and util_windows is not None:
        # Receiver-less, utilisation-gated shrink: one replica, always via a safescale
        # probe (hide -> observe -> commit/rollback), never an immediate sleep.
        for item in sorted(classifications, key=lambda entry: entry.model_name):
            model = item.model_name
            if item.state not in (ModelState.HEALTHY, ModelState.HIGH):
                continue  # CRITICAL/LOW need capacity; IDLE has its own immediate path
            if model in active_probe_models or model in inflight_models or deltas.get(model, 0) != 0:
                continue
            pods = _effective_routable_replicas(model, model_contexts, model_replicas)
            if pods - 1 < _serving_floor(cfg, model, model_contexts, model_replicas):
                continue
            q_after = util_scale_down_ready(
                util_windows.get(model, ()),
                routable=pods,
                threshold=cfg.scale_down_q_per_replica_by_model.get(model, math.inf),
                windows=cfg.util_scale_down_windows,
            )
            if q_after is None or cooldown.blocks(model, "down"):
                continue
            _add_scale_action(
                actions,
                deltas,
                model=model,
                delta=-1,
                reason="util_scale_down_safescale",
                source_loop="fairness",
                requires_safescale=True,
                donor=model,
            )
            delayed_down_models.add(model)
            events.append(f"util_scale_down_proposed:{model}:q_after={q_after:.2f}")

    return PlanResult(actions, delayed_down_models, probe_upscale_plans, events=events)


class _Cooldown:
    """Review F4: hold a model's next action until a fresh metrics window reflects its
    last executed one. Same direction is held; after a scale-up a scale-down is held too;
    after a scale-down a scale-up is allowed only for a CRITICAL receiver (safety)."""

    def __init__(self, cooldowns: Mapping[str, str], events: list[str]) -> None:
        self._cooldowns = dict(cooldowns)
        self._events = events

    def blocks(self, model: str, direction: str, *, critical: bool = False) -> bool:
        last = self._cooldowns.get(model)
        if last is None:
            return False
        if direction == "up" and last == "down" and critical:
            return False
        event = f"cooldown_hold:{model}"
        if event not in self._events:
            self._events.append(event)
        return True

    def down_blocked(self) -> set[str]:
        return set(self._cooldowns)


class _SlotOccupancy:
    """GPU-slot occupancy for wake planning within one build_plan call.

    Under multi-model-per-GPU residency a receiver's sleeping binding is only real
    capacity when no other binding is awake on its GPU(s); the SM rejects any other wake
    with WakeConflict (the E1 dsllama-8b deadlock). Slots claimed by a planned wake (or
    freed by a planned slot-targeted donor sleep) are not counted twice in one tick.
    """

    def __init__(self, cluster_view: ClusterView) -> None:
        self._topology = cluster_view.topology
        self._planned_wakes: set[str] = set()
        self._bindings = tuple(cluster_view.bindings)
        self._awake: dict[tuple[str, int], Binding] = {}
        for binding in self._bindings:
            if binding.awake:
                for gpu in binding.slot.gpu_ids:
                    self._awake[(binding.slot.node, gpu)] = binding
        self._claimed: set[tuple[str, int]] = set()

    @staticmethod
    def _gpus(binding: Binding) -> set[tuple[str, int]]:
        return {(binding.slot.node, gpu) for gpu in binding.slot.gpu_ids}

    def sleeping(self, model: str) -> list[Binding]:
        return sorted(
            (
                binding
                for binding in self._bindings
                if binding.model == model and not binding.awake and not binding.hidden
            ),
            key=lambda binding: natural_key(binding.serve_id),
        )

    def wakeable(self, model: str) -> list[Binding]:
        return [
            binding
            for binding in self.sleeping(model)
            if not any(gpu in self._awake or gpu in self._claimed for gpu in self._gpus(binding))
        ]

    def claim(self, binding: Binding) -> None:
        self._claimed |= self._gpus(binding)
        self._planned_wakes.add(binding.serve_id)

    def has_unplanned_sleeping(self, model: str) -> bool:
        # The SM model-level wake tries every sleeping binding of the model before it
        # creates a new one, and raises WakeConflict on the first infeasible one.
        return any(binding.serve_id not in self._planned_wakes for binding in self.sleeping(model))

    def free_groups(self, tp_size: int) -> list[set[tuple[str, int]]]:
        groups: list[set[tuple[str, int]]] = []
        for node in self._topology.nodes:
            if tp_size > 1:
                candidates = [tuple(pair) for pair in node.two_gpu_slots]
            else:
                candidates = [(gpu,) for gpu in range(node.gpus)]
            for candidate in candidates:
                gpus = {(node.name, gpu) for gpu in candidate}
                if not any(gpu in self._awake or gpu in self._claimed for gpu in gpus):
                    groups.append(gpus)
        return groups

    def claim_gpus(self, gpus: set[tuple[str, int]]) -> None:
        self._claimed |= gpus

    def donor_slot_pods(self, donor: str, receiver: str) -> list[tuple[str, Binding]]:
        """(donor serve_id, receiver sleeping binding) pairs: sleeping that single awake
        donor binding frees exactly a slot the receiver can wake into."""
        pairs: list[tuple[str, Binding]] = []
        used: set[str] = set()
        for receiver_binding in self.sleeping(receiver):
            gpus = self._gpus(receiver_binding)
            if any(gpu in self._claimed for gpu in gpus):
                continue
            occupants = {self._awake[gpu] for gpu in gpus if gpu in self._awake}
            if len(occupants) != 1:
                continue
            occupant = next(iter(occupants))
            if occupant.model != donor or occupant.hidden or occupant.serve_id in used:
                continue
            used.add(occupant.serve_id)
            pairs.append((occupant.serve_id, receiver_binding))
        return pairs


def _plan_sleeping_wakes(
    occupancy: _SlotOccupancy | None,
    *,
    receiver: str,
    need: int,
    events: list[str],
    blocked_event: str,
) -> tuple[int, tuple[str, ...]]:
    """Return (replicas wakeable from sleeping bindings, those bindings' serve_ids).

    The serve_ids are dispatched as binding-level wakes so the SM wakes exactly the
    slot the planner claimed. Without a cluster view this is the legacy count (every
    sleeping binding counts, model-level wake)."""
    need = max(0, need)
    if occupancy is None or need <= 0:
        return need, ()
    wakeable = occupancy.wakeable(receiver)[:need]
    for binding in wakeable:
        occupancy.claim(binding)
    if len(wakeable) < need:
        events.append(f"{blocked_event}:{receiver}")
    return len(wakeable), tuple(binding.serve_id for binding in wakeable)


def _plan_create_capacity(
    occupancy: _SlotOccupancy,
    *,
    receiver: str,
    tp_size: int,
    need: int,
    events: list[str],
    blocked_event: str,
) -> int:
    """Idle capacity the SM can really use for this receiver: a cold create into a slot
    group (tp_size GPUs of one two_gpu_slot for TP>1) with no awake binding. The SM only
    creates once every sleeping binding of the model is awake, so any remaining (blocked)
    sleeping binding makes idle GPUs unusable -- the model-level wake would WakeConflict."""
    if need <= 0:
        return 0
    groups = occupancy.free_groups(tp_size)
    if not groups:
        return 0
    if occupancy.has_unplanned_sleeping(receiver):
        events.append(f"{blocked_event}:{receiver}")
        return 0
    taken = groups[:need]
    for gpus in taken:
        occupancy.claim_gpus(gpus)
    return len(taken)


def _slot_matched_first(
    candidates: list[ModelClassification],
    occupancy: _SlotOccupancy | None,
    receiver: str,
) -> list[ModelClassification]:
    """Stable re-order: donors whose awake binding sits on a receiver-sleeping slot first."""
    if occupancy is None or not occupancy.sleeping(receiver):
        return candidates
    return sorted(
        candidates,
        key=lambda item: 0 if occupancy.donor_slot_pods(item.model_name, receiver) else 1,
    )


def _slot_targeted_transfer(
    occupancy: _SlotOccupancy | None,
    *,
    donor: str,
    receiver: str,
    transfer: int,
    events: list[str],
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    """Pin a donor shrink to the bindings that free receiver-sleeping slots.

    Returns (transfer, donor pods to sleep/probe, receiver pods to wake). With a cluster
    view a donor is only paired when sleeping it frees a GPU slot the receiver holds a
    binding on; otherwise transfer=0 and ``donor_no_slot_match`` is emitted (a
    model-level shrink would sleep a tail pod the receiver cannot use). Without a cluster
    view the legacy model-level pair is kept."""
    if occupancy is None:
        return transfer, (), ()
    pairs = occupancy.donor_slot_pods(donor, receiver)[:transfer]
    if not pairs:
        event = f"donor_no_slot_match:{donor}:{receiver}"
        if event not in events:
            events.append(event)
        return 0, (), ()
    for _, receiver_binding in pairs:
        occupancy.claim(receiver_binding)
    return len(pairs), tuple(pod for pod, _ in pairs), tuple(binding.serve_id for _, binding in pairs)


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
    occupancy: _SlotOccupancy | None = None,
) -> str | None:
    if occupancy is not None and occupancy.has_unplanned_sleeping(model):
        # P1-a: the SM wakes existing sleeping bindings before any create/defrag target
        # is used; with a blocked one the +1 would WakeConflict every tick.
        events.append(f"capacity_blocked:{model}")
        return None
    allocator = SlotAllocator(cluster_view.topology, list(cluster_view.bindings))
    if occupancy is not None:
        groups = occupancy.free_groups(tp_size)
        if groups:
            occupancy.claim_gpus(groups[0])
            return "critical_empty_slot"
    elif allocator.find_slot(tp_size) is not None:
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
    pods: tuple[str, ...] = (),
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
            pods=tuple(pods),
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

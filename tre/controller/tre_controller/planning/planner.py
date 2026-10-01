from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping

from tre_common.registry import ClusterTopology, tp_size_error
from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, donor_mock_cost_key
from tre_common.gpu_placement import (
    PlacementPolicy,
    choose_placement,
    choose_release,
    plan_placements,
)
from tre_sm.allocator.slots import (
    Binding,
    Slot,
    SlotAllocator,
    awake_gpus,
    awake_model_counts,
    gpu_slot_candidates,
    is_buddy_aligned,
    natural_key,
    node_gpu_counts,
    slot_block,
)

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
    # When enabled this suppresses that receiver-less proactive probe on hot
    # (HIGH/CRITICAL) donors. Demand-driven preemption (idle/HIGH immediate donors and the
    # TP critical_same_slot_high_shrink -> CRITICAL beneficiary) is a separate path and is
    # intentionally NOT gated. Default OFF since the v1/paper alignment (A2): the path is
    # v1's paper_high_proactive_shrink (rescue tick, HIGH, replicas > floor, no active
    # probe, not moved by another path this tick -> SafeScale shrink by one step), now
    # protected by the SafeScale KV-cache / donor-health guards and the rollback backoff.
    # TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE=1 re-enables the guard.
    suppress_hot_proactive_probe: bool = False
    disable_eta_gate: bool = False
    # Registry placement.defrag.enabled: gates the critical_tp_defrag migration plan.
    # Off by default, as in v1 (design 20260928-placement-node-balance).
    defrag_enabled: bool = False
    # C1 (registry scaling.rescue_max_step_ratio): a CRITICAL receiver's rescue asks for
    # its whole deficit at once - target min(max(n+1, ceil(n*tau_crit/Z)),
    # max(n+1, floor(ratio*n)), scaling cap, capacity found). 0 = legacy one step
    # (ceil(scale_step_ratio * n)) per decision window.
    rescue_max_step_ratio: float = 2.0
    # C1 (registry scaling.scale_up_cooldown_enabled): the review F4 cooldown also holds
    # a CRITICAL receiver's scale-up. Off by default: the rescue target bookkeeping
    # (``rescue_bases``) keeps an unreflected scale-up from being repeated. Scale-down
    # holds and LOW receivers keep the cooldown either way.
    scale_up_cooldown_enabled: bool = False
    # C1 (registry scaling.rescue_max_step_pods): the rescue target may also reach
    # n + this many replicas (HPA-style "max(ratio x n, n + pods)"); 0 = ratio only.
    rescue_max_step_pods: int = 0
    # C1 review P1 (registry scaling.donor_surplus_release): an immediate IDLE / HIGH
    # donor of a CRITICAL receiver gives its whole surplus in one tick. Off (default):
    # one step per tick, as before C1 - scale-up is aggressive, scale-down cautious.
    donor_surplus_release: bool = False

    def __post_init__(self) -> None:
        # No silent fallback for a bad tp_size (the registry rejects it at load; this
        # is the controller's own guard). The same single rule as the registry, the
        # SM and the placement policy: tre_common.registry.tp_size_error (power of
        # two >= 1, at most MAX_SUPPORTED_TP_SIZE; the widest-node bound is checked
        # where the topology is known - registry load / placement policy).
        for model, tp_size in self.model_tp_sizes.items():
            problem = tp_size_error(tp_size)
            if problem:
                raise ValueError(f"PlanConfig.model_tp_sizes[{model!r}]: {problem}")


@dataclass(frozen=True)
class ClusterView:
    topology: ClusterTopology
    bindings: tuple[Binding, ...]
    #: Registry placement policy (``placement_policy_from_registry``), attached by the
    #: planner tick; None = plain buddy best-fit (tests without a registry).
    placement: PlacementPolicy | None = None
    #: serve_id (pod name) -> pod IP, from the SM fleet state's observed bindings
    #: (``/v2/state`` ``fleet.observed``); used by the SafeScale direct evidence
    #: scrape. Empty when the SM reports no fleet state.
    pod_ips: Mapping[str, str] = field(default_factory=dict)
    #: S5: GPUs the SM reports not wakeable for a reason the bindings do not show
    #: (a Pod loading there, a wake in flight, gpu-truth in use) - ``/v2/state``
    #: ``gpus[].wakeable``. Empty with an SM that does not report it.
    blocked_gpus: frozenset = frozenset()


@dataclass(frozen=True)
class RescuePlan:
    """C1 bookkeeping of a fast-loop rescue scale-up (metadata on its ScaleActions).

    ``target`` = ``covered`` + the replicas this tick planned for the receiver (an
    absolute routable count); ``desired`` = the deficit target before capacity limits;
    ``base`` = the routable count ``desired`` was computed from (the replicas the
    decision window's Z describes); ``covered`` = replicas already counted before this
    plan (the routable count, or an earlier target the window does not reflect yet)."""

    target: int
    desired: int
    base: int
    covered: int


@dataclass(frozen=True)
class RescueBasis:
    """An earlier rescue scale-up of a model whose effect its decision window does not
    reflect yet (C1): the next desired is computed from ``base`` (the replicas the
    window's Z still describes) and only the part above ``covered`` (what that
    scale-up achieved) is planned - an unrefreshed window yields the same desired, so
    nothing is added; a deeper CRITICAL (load still rising) raises the target."""

    base: int
    covered: int


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
    # Sleep path for a negative delta (SM registry service_manager.sleep.budgets_s key):
    # None = derived at dispatch ("urgent" for *_immediate reasons, else "scale_down").
    sleep_path: str | None = None
    # Soft drain budget (s) for the SM's hide -> ack -> drain -> /sleep; None = SM default.
    drain_budget_s: float | None = None
    # Donor -> receiver pair (review 2 P1-1): the donor sleep and the receiver wake of
    # one rescue transfer carry the same id; the ActionQueue executes them as ONE
    # compound action (sleep the donor, and only on success wake the receiver).
    transfer_id: str | None = None
    # S5 (2026-09-30): ``pods`` of a pure-capacity wake are placement HINTS - the SM
    # picks the GPUs itself (registry placement policy) and substitutes a hint it
    # cannot wake. False = exactly these pods (same-GPU donor / receiver relays,
    # SafeScale commits, sleeps).
    hint: bool = False
    # C1: the rescue target this scale-up belongs to (a CRITICAL receiver's wakes /
    # creates / relay wakes). Bookkeeping only: not part of the action's identity.
    rescue: RescuePlan | None = field(default=None, compare=False)


#: SM sleep path of the fast-loop "*_immediate" donors (CRIT donor, idle proactive,
#: low-fairness donor). With the default SM registry it does not drain: hide -> ack
#: -> /sleep mode=abort, the reissue sidecar continues the cut-off requests (v1).
IMMEDIATE_DONOR_SLEEP_PATH = "urgent"


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
    #: Pods that stay hidden (review 4 P2-2): a donor pod whose /sleep was sent
    #: but never confirmed may be asleep - routing is not reopened on it (the
    #: service-manager's crash recovery resolves it).
    keep_hidden: tuple[str, ...] = ()
    #: The SafeScale probe this unhide resolves (review 4 P2-4), if any
    #: (bookkeeping only: not part of the action's identity).
    request_id: str | None = field(default=None, compare=False)


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
class TransferAction:
    """A donor-sleep -> receiver-wake pair executed in order by one dispatch worker
    (review 2 P1-1). Built by :func:`fuse_transfers` from two ScaleActions that share
    a ``transfer_id``; never produced by the planner itself (its output stays two
    ScaleActions, which the decision log / tests inspect)."""

    donor: ScaleAction
    receiver: ScaleAction
    source_loop: SourceLoop = "rescue"

    @property
    def model(self) -> str:
        return self.receiver.model


@dataclass(frozen=True)
class ReceiverTarget:
    """A SafeScale follow-up upscale as an ABSOLUTE target (review 3 P2-1). A
    retry re-sends the same target, so an SM call that succeeded but timed out
    on the client is never applied twice. ``target`` None (the planned form,
    review 4 P2-1): resolved ONCE at the first dispatch from the SM's current
    awake count (never from a possibly stale cluster view) plus ``delta``,
    capped at ``cap`` (the scaling cap), then frozen."""

    model: str
    delta: int
    target: int | None = None
    cap: int | None = None


@dataclass(frozen=True)
class SafeScaleCommitAction:
    """One SafeScale commit batch (review 3 P2-1..P2-3), executed by the
    ActionQueue as an ordered one-shot unit - the SafeScale analogue of
    :class:`TransferAction`:

    1. revalidate against the CURRENT signal state (before every (re)try): the
       donor now needing capacity abandons the commit (its hidden pods are
       unhidden instead); a receiver that no longer needs capacity loses its
       upscale;
    2. sleep exactly the hidden probe pods of the donor;
    3. only then wake the receivers, each to its absolute ``target`` (dropped
       if the donor sleep failed).

    ``donor_done`` records progress across retries (a retry never re-sleeps a
    donor that already slept, and only re-sends the upscales still pending)."""

    donor: str
    pods: tuple[str, ...]
    reason: str
    upscales: tuple[ReceiverTarget, ...] = ()
    drain_budget_s: float | None = None
    request_id: str | None = None
    source_loop: SourceLoop = "safescale"
    donor_done: bool = False
    #: Donor pods whose /sleep was sent but never confirmed (from the SM's
    #: failure outcomes, review 4 P2-2): never unhidden by an abandon / preempt.
    unconfirmed_pods: tuple[str, ...] = ()
    #: Epoch ms of the SafeScale decision (the probe's ``committing_ts``), None =
    #: unknown. B8: a commit first dispatched more than ``commit_max_age_ms`` after
    #: it (held in observe mode, recovered after a restart) becomes the donor unhide.
    #: Metadata, not part of the action's identity (not compared).
    decided_ms: int | None = field(default=None, compare=False)

    @property
    def model(self) -> str:
        return self.donor

    @property
    def touched_models(self) -> tuple[str, ...]:
        """Models this commit still changes (the donor until it slept)."""
        models = () if self.donor_done else (self.donor,)
        return tuple(dict.fromkeys(models + tuple(item.model for item in self.upscales)))

    def donor_sleep(self) -> ScaleAction:
        return ScaleAction(
            self.donor,
            -len(self.pods),
            self.reason,
            self.source_loop,
            pods=self.pods,
            sleep_path="safescale_commit",
            drain_budget_s=self.drain_budget_s,
        )


def fuse_transfers(actions) -> list:
    """Replace each donor/receiver ScaleAction pair sharing a ``transfer_id`` by one
    :class:`TransferAction` (at the donor's position). A half whose partner is
    missing (e.g. dropped by a probe preemption) stays a plain ScaleAction."""
    fused: list = []
    donors: dict[str, int] = {}
    for action in actions:
        transfer_id = action.transfer_id if isinstance(action, ScaleAction) else None
        if transfer_id is None:
            fused.append(action)
            continue
        if action.delta < 0:
            donors[transfer_id] = len(fused)
            fused.append(action)
            continue
        index = donors.pop(transfer_id, None)
        if index is None:
            fused.append(action)
            continue
        fused[index] = TransferAction(
            donor=fused[index], receiver=action, source_loop=action.source_loop
        )
    return fused


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
    probe_backoff_models: set[str] | None = None,
    preemptible_models: set[str] | None = None,
    floor_holds: set[str] | None = None,
    unavailable_gpus: set[tuple[str, int]] | None = None,
    refusals: Mapping[str, str] | None = None,
    rescue_bases: Mapping[str, RescueBasis] | None = None,
) -> PlanResult:
    active_probe_models = active_probe_models or set()
    # Review 3 P2-3: models whose only in-flight work is a SafeScale commit waiting
    # out a retry backoff. A CRITICAL receiver among them is still planned: the
    # queue preempts that retry when the rescue action is submitted.
    preemptible_models = preemptible_models or set()
    # A13: models whose last SafeScale probe rolled back recently (no new HIGH proactive
    # probe until TRE_SAFESCALE_ROLLBACK_BACKOFF_MS has passed).
    probe_backoff_models = probe_backoff_models or set()
    # A local copy (the caller's set is never mutated). A donor taken earlier in this
    # tick is NOT added to it: every take is recorded in ``deltas`` and each donor
    # check subtracts the planned takes from the donor's replicas before comparing
    # with its floor (min_replicas), so a 3-replica donor with floor 1 can give one
    # replica to each of two receivers of one tick, a 2-replica one only once.
    inflight_models = set(inflight_models or ())
    actions: list[Action] = []
    deltas: dict[str, int] = {}
    delayed_down_models: set[str] = set()
    probe_upscale_plans: dict[str, dict[str, int]] = {}
    events: list[str] = []
    remaining_idle = idle_gpus
    slot_shrink_donors: set[str] = set()
    # Sleeping-capacity deadlock fix: GPU-slot occupancy so a receiver's sleeping binding
    # only counts as wakeable capacity when its slot has no awake binding (of any model).
    # S3: GPUs the SM refused a wake on recently (cooldown) are not wake / create
    # capacity this tick; S5: nor are GPUs the SM reports not wakeable.
    occupancy = (
        _SlotOccupancy(cluster_view, unavailable=unavailable_gpus)
        if cluster_view is not None
        else None
    )
    # Review F4 per-model cooldown: model -> direction ("up"/"down") of its last executed
    # action whose effect the decision window does not yet fully reflect.
    # P2-6: donors the SM refused with 409 floor_violation recently are held out of
    # every scale-down (never out of a scale-up) through the same cooldown gate.
    cooldown = _Cooldown(cooldowns or {}, events, floor_holds=floor_holds)

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
    # Band dwell (D8, SignalState.apply_dwell): a receiver whose band has not held for
    # the configured number of new metrics windows is suppressed the same way.
    warmup_suppressed: list[str] = []
    dwell_suppressed: list[str] = []
    kept: list = []
    for item in classifications:
        if item.role == ModelRole.RECEIVER:
            ctx = model_contexts.get(item.model_name, {})
            if not ctx.get("signal_warm", True):
                warmup_suppressed.append(item.model_name)
                continue
            if ctx.get("dwell_confirmed", True) is False:
                dwell_suppressed.append(item.model_name)
                continue
        kept.append(item)
    if warmup_suppressed or dwell_suppressed:
        events.extend(f"receiver_suppressed_signal_warmup:{model}" for model in warmup_suppressed)
        events.extend(f"receiver_suppressed_dwell:{model}" for model in dwell_suppressed)
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

    rescue_bases = rescue_bases or {}
    c1 = cfg.rescue_max_step_ratio > 0
    if cfg.rescue_due:
        # C1: (desired, base, covered) of each CRITICAL receiver planned this tick.
        rescue_ctx: dict[str, tuple[int, int, int]] = {}

        def critical_need(recv: ModelClassification) -> tuple[int, int] | None:
            """(replicas needed, of which wakeable from sleeping bindings), None = skip."""
            # In-flight protection: a model whose previous scale-up is still queued or
            # running (seconds; a cold create longer) is not planned again - a raise
            # would wait behind it on the model resource anyway.
            if recv.model_name in inflight_models and recv.model_name not in preemptible_models:
                return None
            if (cfg.scale_up_cooldown_enabled or not c1) and cooldown.blocks(
                recv.model_name, "up", critical=True
            ):
                return None
            recv_pods = _effective_routable_replicas(recv.model_name, model_contexts, model_replicas)
            recv_assigned = _effective_assigned_replicas(recv.model_name, model_contexts, model_replicas)
            recv_max = _max_replicas(cfg, recv.model_name)
            # Cap on the awake count incl. hidden probe pods (v1 assigned = non-sleeping,
            # draining included; the SM counts the same), not on the routable count.
            recv_awake = _awake_replicas(recv.model_name, model_contexts, model_replicas)
            if recv_awake >= recv_max:
                return None
            if not c1:
                raw_need = min(_scale_step(recv_pods, cfg.scale_step_ratio), recv_max - recv_awake)
                if raw_need <= 0:
                    return None
                return raw_need, min(raw_need, max(0, recv_assigned - recv_pods))
            # C1: one absolute target for the whole deficit. With an earlier scale-up
            # the window does not reflect yet, the target is computed from the replicas
            # the window's Z describes and only what exceeds that scale-up is planned.
            basis = rescue_bases.get(recv.model_name)
            base = basis.base if basis is not None else recv_pods
            covered = max(basis.covered, recv_pods) if basis is not None else recv_pods
            desired = rescue_desired(
                base, recv.Z_m, recv.tau.tau_crit, cfg.rescue_max_step_ratio, cfg.rescue_max_step_pods
            )
            raw_need = min(desired - covered, recv_max - max(recv_awake, covered))
            if raw_need <= 0:
                if basis is not None:
                    events.append(
                        f"rescue_target_hold:{recv.model_name}:desired={desired}:covered={covered}"
                    )
                return None
            rescue_ctx[recv.model_name] = (desired, base, covered)
            return raw_need, min(raw_need, max(0, recv_assigned - recv_pods))

        critical_needs = {recv.model_name: critical_need(recv) for recv in critical_receivers}
        # Every CRITICAL receiver's sleeping-binding wakes are assigned jointly up
        # front, so an earlier receiver never takes the one free slot a later one
        # can wake into while it had another (multi-receiver slot stealing).
        reserved_wakes = _plan_joint_wakes(
            occupancy,
            [(model, need[1]) for model, need in critical_needs.items() if need is not None],
            events=events,
        )
        for recv in critical_receivers:
            need = critical_needs[recv.model_name]
            if need is None:
                continue
            raw_need, wake_need = need
            first_action = len(actions)
            try:
                gain_from_sleeping, wake_pods = _take_reserved_wakes(
                    occupancy,
                    reserved_wakes,
                    receiver=recv.model_name,
                    need=wake_need,
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
                        hint=True,
                    )
                    raw_need -= gain_from_sleeping
                    if raw_need <= 0:
                        continue

                tp_size = _tp_size(cfg, recv.model_name)
                if tp_size > 1 and cluster_view is not None:
                    same_slot_shrink = _try_plan_same_slot_high_shrink(
                        classifications=classifications,
                        model_contexts=model_contexts,
                        model_replicas=model_replicas,
                        cfg=cfg,
                        cluster_view=cluster_view,
                        receiver=recv.model_name,
                        active_probe_models=active_probe_models,
                        # One SafeScale probe per donor model at a time (the state
                        # machine is keyed by model): a donor already shrunk for an earlier
                        # receiver this tick cannot start a second probe - its extra
                        # replicas stay available to the immediate donor loop below.
                        inflight_models=inflight_models | cooldown.down_blocked() | slot_shrink_donors,
                        planned_deltas=deltas,
                        taken_serve_ids=occupancy.donor_taken_ids() if occupancy is not None else set(),
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
                        # Replica floor (2026-09-29, fix C, relaxed): the shrink takes one
                        # replica of the donor. It is recorded in the plan's deltas only, so
                        # a later receiver's donor check counts it against the floor (two
                        # takes from a 2-replica donor with min_replicas 1 would leave 0; a
                        # 3-replica donor may still give one more). Its binding is marked
                        # taken so no later action of this tick sleeps / hides the same pod.
                        deltas[same_slot_shrink.donor] = deltas.get(same_slot_shrink.donor, 0) - 1
                        if occupancy is not None:
                            occupancy.take_donor(same_slot_shrink.serve_id)
                        events.append(
                            f"safescale_preemption:{same_slot_shrink.donor}->{recv.model_name}:{same_slot_shrink.reason}"
                        )
                        # C1: a multi-replica deficit also takes free slot pairs below.
                        raw_need -= 1
                        if raw_need <= 0:
                            continue

                    # C1: up to raw_need free slot groups (one action); a defrag plan is
                    # still one migration at a time (it is a cluster-wide action).
                    # Without a slot occupancy (no claims) the allocator would return
                    # the same free slot again: one slot per tick then (review P3).
                    empty_slots = 0
                    slot_limit = raw_need if occupancy is not None else 1
                    while slot_limit > empty_slots:
                        tp_planned = _try_plan_tp_capacity(
                            actions,
                            model=recv.model_name,
                            tp_size=tp_size,
                            cluster_view=cluster_view,
                            events=events,
                            source_loop="rescue",
                            occupancy=occupancy,
                            defrag_enabled=cfg.defrag_enabled,
                        )
                        if tp_planned == "critical_empty_slot":
                            empty_slots += 1
                            continue
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
                        break
                    if empty_slots:
                        _add_scale_action(
                            actions,
                            deltas,
                            model=recv.model_name,
                            delta=empty_slots,
                            reason="critical_empty_slot",
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
                        _donor_give(donor, donor_pods, cfg),
                        max(0, donor_pods - planned_take - donor_min),
                    )
                    if transfer <= 0:
                        continue
                    transfer, donor_slot_pods, receiver_slot_pods = _slot_targeted_transfer(
                        occupancy, donor=donor.model_name, receiver=recv.model_name, transfer=transfer, events=events
                    )
                    if transfer <= 0:
                        continue
                    # One transfer: the receiver's wake needs the GPU the donor's sleep
                    # frees, so the queue runs the pair in order as one compound action.
                    transfer_id = f"{donor.model_name}->{recv.model_name}#{len(actions)}"
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
                        transfer_id=transfer_id,
                        sleep_path=IMMEDIATE_DONOR_SLEEP_PATH,
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
                        transfer_id=transfer_id,
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
            finally:
                if recv.model_name in rescue_ctx:
                    _tag_rescue_actions(actions, first_action, recv, rescue_ctx[recv.model_name], events)

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
                    sleep_path=IMMEDIATE_DONOR_SLEEP_PATH,
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
            # A13 backoff, checked (and logged) only for a model that would otherwise be
            # probed - a model already at its floor stays silent every tick.
            if high.model_name in probe_backoff_models:
                events.append(f"safescale_rollback_backoff:{high.model_name}")
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
        _placement_retry_events(actions, cluster_view, refusals, events)
        return PlanResult(actions, delayed_down_models, probe_upscale_plans, events=events)

    def low_need(recv: ModelClassification) -> tuple[int, int] | None:
        """(replicas needed, of which wakeable from sleeping bindings), None = skip."""
        if recv.model_name in inflight_models:
            return None
        if cooldown.blocks(recv.model_name, "up"):
            return None
        recv_pods = _effective_routable_replicas(recv.model_name, model_contexts, model_replicas)
        recv_assigned = _effective_assigned_replicas(recv.model_name, model_contexts, model_replicas)
        receiver_capacity = (
            _max_replicas(cfg, recv.model_name)
            - _awake_replicas(recv.model_name, model_contexts, model_replicas)
            - max(0, deltas.get(recv.model_name, 0))
        )
        if receiver_capacity <= 0:
            return None
        needed = min(_scale_step(recv_pods, cfg.scale_step_ratio), receiver_capacity)
        return needed, min(needed, max(0, recv_assigned - recv_pods))

    # A LOW receiver is never a donor, so its need does not change while earlier
    # receivers are planned: computed (and its wakes assigned jointly) up front.
    low_needs = {recv.model_name: low_need(recv) for recv in low_receivers}
    reserved_low_wakes = _plan_joint_wakes(
        occupancy,
        [(model, need[1]) for model, need in low_needs.items() if need is not None],
        events=events,
    )
    for recv in low_receivers:
        need = low_needs[recv.model_name]
        if need is None:
            continue
        needed, wake_need = need

        sleeping_gain, wake_pods = _take_reserved_wakes(
            occupancy,
            reserved_low_wakes,
            receiver=recv.model_name,
            need=wake_need,
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
                hint=True,
            )
            needed -= sleeping_gain

        if occupancy is not None:
            # P1-a: only slot groups the SM will really create into for this receiver
            # (TP-aware; none while it still has a sleeping binding the SM would try first).
            idle_gain = _plan_create_capacity(
                occupancy,
                receiver=recv.model_name,
                tp_size=_tp_size(cfg, recv.model_name),
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
                sleep_path=IMMEDIATE_DONOR_SLEEP_PATH,
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

    _placement_retry_events(actions, cluster_view, refusals, events)
    return PlanResult(actions, delayed_down_models, probe_upscale_plans, events=events)


def _placement_retry_events(
    actions: list[Action],
    cluster_view: ClusterView | None,
    refusals: Mapping[str, str] | None,
    events: list[str],
) -> None:
    """S3: ``placement_retry:<model>:<refused gpu>-><new gpu>`` for every wake of
    a model whose last wake the SM refused (its GPU is cooling down): the plan
    moved it to another GPU."""
    if not refusals or cluster_view is None:
        return
    slots = {binding.serve_id: binding.slot for binding in cluster_view.bindings}
    for action in actions:
        if not isinstance(action, ScaleAction) or action.delta <= 0 or action.model not in refusals:
            continue
        for pod in action.pods:
            slot = slots.get(pod)
            if slot is None:
                continue
            target = f"{slot.node}/{','.join(str(gpu) for gpu in slot.gpu_ids)}"
            if target != refusals[action.model]:
                events.append(f"placement_retry:{action.model}:{refusals[action.model]}->{target}")


class _Cooldown:
    """Review F4: hold a model's next action until a fresh metrics window reflects its
    last executed one. Same direction is held; after a scale-up a scale-down is held too;
    after a scale-down a scale-up is allowed only for a CRITICAL receiver (safety)."""

    def __init__(
        self, cooldowns: Mapping[str, str], events: list[str], *, floor_holds: set[str] | None = None
    ) -> None:
        self._cooldowns = dict(cooldowns)
        self._events = events
        #: P2-6: models held out of scale-downs after an SM floor_violation refusal.
        self._floor_holds = set(floor_holds or ())

    def blocks(self, model: str, direction: str, *, critical: bool = False) -> bool:
        if direction == "down" and model in self._floor_holds:
            event = f"floor_violation_hold:{model}"
            if event not in self._events:
                self._events.append(event)
            return True
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
        return set(self._cooldowns) | self._floor_holds


class _SlotOccupancy:
    """GPU-slot occupancy for wake planning within one build_plan call.

    Under multi-model-per-GPU residency a receiver's sleeping binding is only real
    capacity when no other binding is awake on its GPU(s); the SM rejects any other wake
    with WakeConflict (the E1 dsllama-8b deadlock). Slots claimed by a planned wake (or
    freed by a planned slot-targeted donor sleep) are not counted twice in one tick.
    """

    def __init__(
        self, cluster_view: ClusterView, *, unavailable: set[tuple[str, int]] | None = None
    ) -> None:
        self._topology = cluster_view.topology
        self._nodes = node_gpu_counts(cluster_view.topology)
        #: GPUs that are no wake / create capacity this tick although no awake binding
        #: holds them: S5 the SM reports them not wakeable (a Pod loading, a wake in
        #: flight, gpu-truth in use), S3 a wake there was refused recently (cooldown).
        self._blocked: set[tuple[str, int]] = {
            (str(node), int(gpu))
            for node, gpu in set(cluster_view.blocked_gpus) | set(unavailable or ())
        }
        self._policy = cluster_view.placement
        self._planned_wakes: set[str] = set()
        self._bindings = tuple(cluster_view.bindings)
        self._awake: dict[tuple[str, int], Binding] = {}
        for binding in self._bindings:
            if binding.awake:
                for gpu in binding.slot.gpu_ids:
                    self._awake[(binding.slot.node, gpu)] = binding
        self._claimed: set[tuple[str, int]] = set()
        # Donor bindings this tick already sleeps / hides (slot-targeted donor pods, a
        # same-slot shrink's binding): never taken a second time in the same tick.
        self._donor_taken: set[str] = set()
        # Per model: GPUs claimed by this tick's planned wakes / creates, and how many.
        self._model_claimed: dict[str, set[tuple[str, int]]] = {}
        self._planned_counts: dict[str, int] = {}

    def policy(self) -> PlacementPolicy | None:
        """The placement policy with the reserve bounded by the awake counts this
        tick's plan leads to (bindings awake now + planned wakes / creates)."""
        if self._policy is None:
            return None
        counts = awake_model_counts(self._bindings)
        for model, planned in self._planned_counts.items():
            counts[model] = counts.get(model, 0) + planned
        return self._policy.for_awake(counts)

    def take_donor(self, serve_id: str) -> None:
        """Mark a donor binding as slept / hidden by this tick's plan."""
        self._donor_taken.add(serve_id)

    def donor_taken_ids(self) -> set[str]:
        return set(self._donor_taken)

    def model_gpus(self, model: str) -> set[tuple[str, int]]:
        """GPUs ``model`` holds awake or has claimed this tick."""
        held = {gpu for gpu, binding in self._awake.items() if binding.model == model}
        return held | self._model_claimed.get(model, set())

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

    def occupied(self) -> set[tuple[str, int]]:
        """Every GPU an awake binding holds, a planned wake has claimed, or that is
        blocked (not wakeable / cooling down)."""
        return set(self._awake) | set(self._claimed) | set(self._blocked)

    def wakeable(self, model: str) -> list[Binding]:
        """Sleeping bindings whose slot is free, least wasteful first."""
        return self.plan_wakes(model, len(self.sleeping(model)))

    def plan_wakes(self, model: str, need: int) -> list[Binding]:
        """Up to ``need`` sleeping bindings to wake, in placement-policy order.

        Each pick is scored against the GPUs the earlier picks take (buddy fit,
        pair reservation, node balance; tre_common.gpu_placement), so waking three
        single-GPU replicas fills a half-used pair before it breaks a free one.
        """
        need = max(0, need)
        sleeping = [
            binding
            for binding in self.sleeping(model)
            if not any(
                gpu in self._awake or gpu in self._claimed or gpu in self._blocked
                for gpu in self._gpus(binding)
            )
        ]
        if need <= 0 or not sleeping:
            return []
        aligned = [
            binding for binding in sleeping if is_buddy_aligned(binding.slot, self._nodes)
        ]
        scorable: list[Binding] = []
        if aligned:
            # One model has one tp_size; anything else cannot be compared with it.
            tp_size = len(aligned[0].slot.gpu_ids)
            scorable = [
                binding for binding in aligned if len(binding.slot.gpu_ids) == tp_size
            ]
        scorable_ids = {binding.serve_id for binding in scorable}
        ranked: list[Binding] = []
        if scorable:
            picks = plan_placements(
                [slot_block(binding.slot) for binding in scorable],
                nodes=self._nodes,
                occupied=self.occupied(),
                tp_size=len(scorable[0].slot.gpu_ids),
                count=need,
                policy=self.policy(),
                model_occupied=self.model_gpus(model),
            )
            ranked = [scorable[pick.index] for pick in picks]
        # Slots the buddy model cannot score keep the old natural order, at the tail.
        rest = [binding for binding in sleeping if binding.serve_id not in scorable_ids]
        return (ranked + rest)[:need]

    def save(self) -> tuple:
        """The claims made so far (for a trial plan, see :meth:`restore`)."""
        return (
            set(self._claimed),
            set(self._planned_wakes),
            {model: set(gpus) for model, gpus in self._model_claimed.items()},
            dict(self._planned_counts),
        )

    def restore(self, state: tuple) -> None:
        claimed, planned_wakes, model_claimed, planned_counts = state
        self._claimed = set(claimed)
        self._planned_wakes = set(planned_wakes)
        self._model_claimed = {model: set(gpus) for model, gpus in model_claimed.items()}
        self._planned_counts = dict(planned_counts)

    def claim(self, binding: Binding) -> None:
        gpus = self._gpus(binding)
        self._claimed |= gpus
        self._planned_wakes.add(binding.serve_id)
        self._model_claimed.setdefault(binding.model, set()).update(gpus)
        self._planned_counts[binding.model] = self._planned_counts.get(binding.model, 0) + 1

    def has_unplanned_sleeping(self, model: str) -> bool:
        # The SM model-level wake tries every sleeping binding of the model before it
        # creates a new one, and raises WakeConflict on the first infeasible one.
        return any(binding.serve_id not in self._planned_wakes for binding in self.sleeping(model))

    def free_groups(self, tp_size: int, model: str | None = None) -> list[set[tuple[str, int]]]:
        """Free ``tp_size``-GPU slots for ``model``, best first (placement policy).

        Ordered like :meth:`plan_wakes`, and sequentially: taking a prefix of the
        result is the same placement a one-at-a-time loop would produce.
        """
        candidates = gpu_slot_candidates(self._topology, tp_size)
        if not candidates:
            return []
        picks = plan_placements(
            [slot_block(slot) for slot in candidates],
            nodes=self._nodes,
            occupied=self.occupied(),
            tp_size=tp_size,
            count=len(candidates),
            policy=self.policy(),
            model_occupied=self.model_gpus(model) if model else (),
        )
        return [set(pick.block.keys) for pick in picks]

    def claim_gpus(self, gpus: set[tuple[str, int]], model: str | None = None) -> None:
        self._claimed |= gpus
        if model is not None:
            self._model_claimed.setdefault(model, set()).update(gpus)
            self._planned_counts[model] = self._planned_counts.get(model, 0) + 1

    def donor_slot_pods(self, donor: str, receiver: str) -> list[tuple[str, Binding]]:
        """(donor serve_id, receiver sleeping binding) pairs: sleeping that single awake
        donor binding frees exactly a slot the receiver can wake into.

        Ranked by the placement policy (the receiver slot is scored as if its donor
        had already slept), greedily, one receiver slot per donor binding; slots the
        buddy model cannot score keep the natural order at the tail."""
        matches: list[tuple[Binding, Binding]] = []
        for receiver_binding in self.sleeping(receiver):
            gpus = self._gpus(receiver_binding)
            if any(gpu in self._claimed or gpu in self._blocked for gpu in gpus):
                continue
            occupants = {self._awake[gpu] for gpu in gpus if gpu in self._awake}
            if len(occupants) != 1:
                continue
            occupant = next(iter(occupants))
            if occupant.model != donor or occupant.hidden or occupant.serve_id in self._donor_taken:
                continue
            matches.append((receiver_binding, occupant))
        policy = self.policy()
        occupied = self.occupied()
        receiver_gpus = self.model_gpus(receiver)
        pairs: list[tuple[str, Binding]] = []
        used: set[str] = set()
        while True:
            best: tuple[tuple, int] | None = None
            for index, (receiver_binding, occupant) in enumerate(matches):
                if occupant.serve_id in used or any(
                    binding.serve_id == receiver_binding.serve_id for _, binding in pairs
                ):
                    continue
                choice = None
                if is_buddy_aligned(receiver_binding.slot, self._nodes):
                    choice = choose_placement(
                        [slot_block(receiver_binding.slot)],
                        nodes=self._nodes,
                        occupied=occupied - self._gpus(occupant),
                        policy=policy,
                        model_occupied=receiver_gpus,
                    )
                key = ((0, choice.score) if choice is not None else (1, ()), index)
                if best is None or key < best[0]:
                    best = (key, index)
            if best is None:
                return pairs
            receiver_binding, occupant = matches[best[1]]
            used.add(occupant.serve_id)
            receiver_gpus = receiver_gpus | self._gpus(receiver_binding)
            pairs.append((occupant.serve_id, receiver_binding))


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
    wakeable = occupancy.plan_wakes(receiver, need)
    for binding in wakeable:
        occupancy.claim(binding)
    if len(wakeable) < need:
        events.append(f"{blocked_event}:{receiver}")
    return len(wakeable), tuple(binding.serve_id for binding in wakeable)


def _take_reserved_wakes(
    occupancy: _SlotOccupancy | None,
    reserved: Mapping[str, list[Binding]],
    *,
    receiver: str,
    need: int,
    events: list[str],
    blocked_event: str,
) -> tuple[int, tuple[str, ...]]:
    """:func:`_plan_sleeping_wakes` for wakes already assigned (and claimed) by
    :func:`_plan_joint_wakes`. Without a cluster view: the legacy count."""
    need = max(0, need)
    if occupancy is None or need <= 0:
        return need, ()
    wakes = list(reserved.get(receiver, ()))[:need]
    if len(wakes) < need:
        events.append(f"{blocked_event}:{receiver}")
    return len(wakes), tuple(binding.serve_id for binding in wakes)


def _plan_joint_wakes(
    occupancy: _SlotOccupancy | None,
    requests: list[tuple[str, int]],
    *,
    events: list[str],
) -> dict[str, list[Binding]]:
    """Assign the sleeping-binding wakes of several receivers jointly and claim them.

    ``requests`` = (receiver, wakes wanted) in priority order. Under multi-model
    residency two receivers can have sleeping bindings on the same free GPU; the
    per-receiver greedy lets the first one take a slot the second needs although
    the first had another. The greedy plan (placement-optimal, unchanged whenever
    it satisfies everyone) is kept unless a maximum matching wakes strictly more
    replicas; then the matching is used (event ``joint_wake_assignment``).
    Deterministic: receivers most constrained first (fewest free candidate slots
    minus need, then priority), candidates in placement order."""
    if occupancy is None:
        return {}
    requests = [(model, int(need)) for model, need in requests if need > 0]
    if not requests:
        return {}
    before = occupancy.save()
    greedy: dict[str, list[Binding]] = {}
    for model, need in requests:
        picks = occupancy.plan_wakes(model, need)
        for binding in picks:
            occupancy.claim(binding)
        greedy[model] = picks
    if len(requests) == 1 or all(len(greedy[model]) >= need for model, need in requests):
        return greedy
    after_greedy = occupancy.save()
    occupancy.restore(before)
    joint = _match_wakes(occupancy, requests)
    if sum(map(len, joint.values())) <= sum(map(len, greedy.values())):
        occupancy.restore(after_greedy)
        return greedy
    for model, _ in requests:
        for binding in joint.get(model, ()):
            occupancy.claim(binding)
    events.append(
        "joint_wake_assignment:"
        + ",".join(f"{model}={len(joint.get(model, ()))}" for model, _ in requests)
    )
    return joint


def _match_wakes(
    occupancy: _SlotOccupancy, requests: list[tuple[str, int]]
) -> dict[str, list[Binding]]:
    """Maximum matching (augmenting paths) of receiver wake units to free slots."""
    ranked = {model: occupancy.plan_wakes(model, len(occupancy.sleeping(model))) for model, _ in requests}
    order = sorted(
        range(len(requests)),
        key=lambda index: (len(ranked[requests[index][0]]) - requests[index][1], index),
    )
    units = [requests[index][0] for index in order for _ in range(requests[index][1])]
    holder: dict[frozenset, int] = {}
    pick: dict[int, Binding] = {}

    def slot_key(binding: Binding) -> frozenset:
        return frozenset((binding.slot.node, gpu) for gpu in binding.slot.gpu_ids)

    def augment(unit: int, seen: set) -> bool:
        for binding in ranked[units[unit]]:
            key = slot_key(binding)
            if key in seen:
                continue
            seen.add(key)
            current = holder.get(key)
            if current is None or augment(current, seen):
                holder[key] = unit
                pick[unit] = binding
                return True
        return False

    for unit in range(len(units)):
        augment(unit, set())
    # Slots that overlap without being equal (mixed tp) are not modelled by the
    # matching: keep the first of any overlapping pair, in unit order.
    used: set = set()
    chosen: dict[str, list[Binding]] = {}
    for unit in range(len(units)):
        binding = pick.get(unit)
        if binding is None or slot_key(binding) & used:
            continue
        used |= slot_key(binding)
        chosen.setdefault(units[unit], []).append(binding)
    for model, bindings in chosen.items():
        rank = {binding.serve_id: index for index, binding in enumerate(ranked[model])}
        bindings.sort(key=lambda binding: rank[binding.serve_id])
    return chosen


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
    groups = occupancy.free_groups(tp_size, receiver)
    if not groups:
        return 0
    if occupancy.has_unplanned_sleeping(receiver):
        events.append(f"{blocked_event}:{receiver}")
        return 0
    taken = groups[:need]
    for gpus in taken:
        occupancy.claim_gpus(gpus, receiver)
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
    for donor_pod, receiver_binding in pairs:
        occupancy.claim(receiver_binding)
        occupancy.take_donor(donor_pod)
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
    planned_deltas: Mapping[str, int] | None = None,
    taken_serve_ids: set[str] | None = None,
) -> ShrinkForSlotAction | None:
    high_by_model = {item.model_name: item for item in classifications if item.state == ModelState.HIGH}
    candidates: list[tuple[float, Binding]] = []
    # Only AWAKE bindings hold a GPU: a sleeping resident (multi-model residency)
    # neither occupies the slot mate nor frees anything when "shrunk".
    occupied = awake_gpus(cluster_view.bindings)

    for binding in cluster_view.bindings:
        if not binding.awake or binding.hidden:
            continue
        high = high_by_model.get(binding.model)
        if high is None or binding.model in active_probe_models or binding.model in inflight_models:
            continue
        if binding.model == receiver or len(binding.slot.gpu_ids) != 1:
            continue
        if binding.serve_id in (taken_serve_ids or ()):
            continue  # this tick already sleeps this pod for another receiver
        donor_pods = _effective_routable_replicas(binding.model, model_contexts, model_replicas)
        # Takes already planned this tick (e.g. a critical_donor_immediate of an earlier
        # receiver) count against the donor's floor too.
        planned_take = abs(min((planned_deltas or {}).get(binding.model, 0), 0))
        if donor_pods - planned_take <= _min_replicas(cfg, binding.model):
            continue
        if not _slot_mate_is_free(cluster_view, binding.slot, occupied):
            continue
        candidates.append((high.Z_m if high.Z_m is not None else math.inf, binding))

    if not candidates:
        return None

    best_z = min(z_m for z_m, _ in candidates)
    donor_binding = _release_pick(
        [binding for z_m, binding in candidates if z_m == best_z], cluster_view
    )
    return ShrinkForSlotAction(
        donor=donor_binding.model,
        beneficiary=receiver,
        serve_id=donor_binding.serve_id,
        slot=donor_binding.slot,
        reason="critical_same_slot_high_shrink",
        source_loop=source_loop,
    )


def _release_pick(bindings: list[Binding], cluster_view: ClusterView) -> Binding:
    """Among equally ranked donor bindings, the one the placement policy releases
    first (merge gain, node load, same-model spread, address; never serve_id)."""
    if len(bindings) == 1:
        return bindings[0]
    nodes = node_gpu_counts(cluster_view.topology)
    aligned = [binding for binding in bindings if is_buddy_aligned(binding.slot, nodes)]
    if aligned:
        choice = choose_release(
            [slot_block(binding.slot) for binding in aligned],
            nodes=nodes,
            occupied=awake_gpus(cluster_view.bindings)
            | {key for binding in aligned for key in slot_block(binding.slot).keys},
            policy=cluster_view.placement,
        )
        if choice is not None:
            return aligned[choice.index]
    return min(bindings, key=lambda binding: natural_key(binding.serve_id))


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
    defrag_enabled: bool = False,
) -> str | None:
    if occupancy is not None and occupancy.has_unplanned_sleeping(model):
        # P1-a: the SM wakes existing sleeping bindings before any create/defrag target
        # is used; with a blocked one the +1 would WakeConflict every tick.
        events.append(f"capacity_blocked:{model}")
        return None
    allocator = SlotAllocator(
        cluster_view.topology, list(cluster_view.bindings), policy=cluster_view.placement
    )
    if occupancy is not None:
        groups = occupancy.free_groups(tp_size, model)
        if groups:
            occupancy.claim_gpus(groups[0], model)
            return "critical_empty_slot"
    elif allocator.find_slot(tp_size, model) is not None:
        return "critical_empty_slot"

    if not defrag_enabled:
        # Registry placement.defrag.enabled is false (default, v1 parity): no
        # automatic migration is planned (design 20260928-placement-node-balance).
        events.append(f"defrag_disabled:{model}")
        events.append(f"capacity_blocked:{model}")
        return None
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
        if item.state == ModelState.UNKNOWN
        or (item.Z_m is None and item.state != ModelState.IDLE and not getattr(item, "signal_idle", False))
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
    transfer_id: str | None = None,
    sleep_path: str | None = None,
    drain_budget_s: float | None = None,
    hint: bool = False,
) -> None:
    """``sleep_path`` / ``drain_budget_s`` go to the SM sleep of a negative delta
    (None = the dispatcher / SM default). The SM registry decides whether the path
    drains at all (service_manager.sleep.no_drain_paths)."""
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
            transfer_id=transfer_id,
            sleep_path=sleep_path if delta < 0 else None,
            drain_budget_s=drain_budget_s if delta < 0 else None,
            hint=bool(hint and delta > 0 and pods),
        )
    )


def rescue_desired(
    n: int, z_m: float | None, tau_crit: float, ratio: float, step_pods: int = 0
) -> int:
    """C1 rescue target (routable replicas) of a CRITICAL receiver with ``n`` routable
    replicas and decision-window signal ``z_m``: the replicas that bring Z back to
    ``tau_crit`` under Z proportional to n at fixed load, ``ceil(n * tau_crit / Z)``,
    at least n + 1 and at most ``max(n + 1, floor(ratio * n), n + step_pods)``. Z
    missing / <= 0 or n <= 0: n + 1."""
    n = max(0, int(n))
    floor_target = n + 1
    cap = max(floor_target, math.floor(float(ratio) * n + 1e-9), n + max(0, int(step_pods)))
    if n <= 0 or z_m is None or not math.isfinite(z_m) or z_m <= 0 or tau_crit <= 0:
        return floor_target
    want = math.ceil(n * float(tau_crit) / float(z_m) - 1e-9)
    return min(max(want, floor_target), cap)


def _donor_give(donor: ModelClassification, donor_pods: int, cfg: PlanConfig) -> int:
    """Replicas an immediate (IDLE / HIGH) donor may give one CRITICAL receiver in one
    tick, before its floor. Default (and legacy): one step - the paper's bounded
    pairwise transfer moves at most one step per pair per tick, the donor side
    included. ``donor_surplus_release`` (opt-in): its surplus - an IDLE donor all of
    it; a HIGH donor the replicas above ``ceil(n * tau_high / Z)`` (its projected Z
    stays >= tau_high), never less than one step. The relay is capped by what the
    receiver still needs either way (caller)."""
    step = _scale_step(donor_pods, cfg.scale_step_ratio)
    if cfg.rescue_max_step_ratio <= 0 or not cfg.donor_surplus_release:
        return step
    if donor.state == ModelState.IDLE:
        return max(step, donor_pods)
    z_m = donor.Z_m
    tau_high = donor.tau.tau_high
    if z_m is None or not math.isfinite(z_m) or z_m <= 0 or tau_high <= 0:
        return step
    keep = math.ceil(donor_pods * float(tau_high) / float(z_m) - 1e-9)
    return max(step, donor_pods - keep)


def _tag_rescue_actions(
    actions: list[Action],
    first: int,
    recv: ModelClassification,
    ctx: tuple[int, int, int],
    events: list[str],
) -> None:
    """Attach the C1 :class:`RescuePlan` to the receiver's rescue scale-ups planned
    from ``actions[first:]`` and log the decision (``rescue_target``)."""
    desired, base, covered = ctx
    model = recv.model_name
    planned = sum(
        action.delta
        for action in actions[first:]
        if isinstance(action, ScaleAction) and action.model == model and action.delta > 0
    )
    plan = RescuePlan(target=covered + planned, desired=desired, base=base, covered=covered)
    for index in range(first, len(actions)):
        action = actions[index]
        if isinstance(action, ScaleAction) and action.model == model and action.delta > 0:
            actions[index] = replace(action, rescue=plan)
    if planned <= 0:
        return  # nothing planned: the capacity events of the paths say why
    z_text = "none" if recv.Z_m is None else f"{recv.Z_m:.4f}"
    events.append(
        f"rescue_target:{model}:n={base}:z={z_text}:desired={desired}:covered={covered}:planned={planned}"
    )


def _scale_step(current_pods: int, ratio: float = 0.1) -> int:
    if current_pods <= 0:
        return 1
    return max(1, math.ceil(ratio * current_pods))


def _tp_size(cfg: PlanConfig, model_name: str) -> int:
    """The model's tp_size. An empty map (callers without a registry) means every
    model is single-GPU; a model missing from a non-empty map is a wiring error and
    raises - never a silent tp 1 (it would plan single-GPU slots for a TP model)."""
    if not cfg.model_tp_sizes:
        return 1
    try:
        return int(cfg.model_tp_sizes[model_name])
    except KeyError:
        raise ValueError(
            f"planner: model {model_name!r} has no tp_size in PlanConfig.model_tp_sizes "
            f"(known: {sorted(cfg.model_tp_sizes)})"
        ) from None


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


def _awake_replicas(
    model_name: str,
    model_contexts: dict[str, dict[str, Any]],
    model_replicas: dict[str, int],
) -> int:
    """Awake bindings incl. hidden probe pods: what the scaling cap (max_awake_replicas)
    is checked against, matching v1 (assigned = non-sleeping, draining included) and the
    service-manager. Falls back to the routable count without a fleet view."""
    awake = model_contexts.get(model_name, {}).get("awake_replicas")
    if awake is None:
        return _effective_routable_replicas(model_name, model_contexts, model_replicas)
    try:
        return max(0, int(awake))
    except Exception:
        return _effective_routable_replicas(model_name, model_contexts, model_replicas)


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

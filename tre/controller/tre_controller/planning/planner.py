from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping

from tre_common.registry import ClusterTopology, tp_size_error
from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, donor_mock_cost_key
from tre_common.gpu_placement import (
    PlacementPolicy,
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
    # protected by the SafeScale KV-cache / donor-health guards and the rollback evidence
    # hold (a rolled-back model is re-probed only on new evidence, timer cleanup 2026-10-02).
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
    # C1 (registry scaling.rescue_max_step_pods): the rescue target may also reach
    # n + this many replicas (HPA-style "max(ratio x n, n + pods)"); 0 = ratio only.
    rescue_max_step_pods: int = 0
    # C1 review P1 (registry scaling.donor_surplus_release): an immediate HIGH donor of
    # a CRITICAL receiver gives its surplus above its tau_high level in one tick. Off
    # (default): one step per tick - scale-down stays cautious. An IDLE donor always
    # gives its whole surplus (Q3 2026-10-06, code rule): an idle window is evidence
    # that does not depend on the replica count.
    donor_surplus_release: bool = False
    #: O1 review P2-1 (evidence-gated): a C1 rescue decided on a partial
    #: (post-breakpoint) window - ``signal_full_window`` False - whose
    #: ``signal_evidence_requests`` is below ``partial_window_lowevidence_requests``
    #: adds at most ``partial_window_max_step`` replicas (0 = no cap); with more
    #: evidence the whole deficit at once.
    partial_window_max_step: int = 0
    partial_window_lowevidence_requests: int = 0
    #: Onset saturation rescue (registry scaling.saturation_max_step_factor): a
    #: ``saturation_rescue`` CRITICAL receiver's target is min(max(n + 1,
    #: floor(factor * n)), scaling cap) - bounded doubling without throughput evidence.
    saturation_max_step_factor: float = 2.0

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
    #: Epoch ms the controller received this view from the SM (``refresh_cluster_view_once``);
    #: a routable-count change it shows happened at or before it (O1 breakpoint time).
    #: None = a synthetic view (tests / offline), see ``tick.breakpoint_observation``.
    fetched_ms: int | None = field(default=None, compare=False)
    #: Timer cleanup review P2-2: epoch ms at or AFTER which the SM state of this view was
    #: produced (a lower bound): the time the controller sent the request (controller
    #: clock - timestamps are never compared across machines). An action that completed
    #: at or before it is shown by the view (O1 view-pending gate). None = unknown
    #: (synthetic views): ``fetched_ms`` is used.
    state_ms: int | None = field(default=None, compare=False)
    #: The SM's own ``/v2/state`` ``fetched_ms`` (SM clock): reference only, never
    #: compared with controller times.
    sm_fetched_ms: int | None = field(default=None, compare=False)
    #: 2026-10-02 (design 20261002-controller-transfer): serve ids the SM counts
    #: routable - the same function its replica-floor check uses (``/v2/state``
    #: ``bindings[].routable``). None = not reported / not readable: the controller's
    #: own count (awake and not hidden) is used and ``routable_error`` says why.
    routable_ids: frozenset | None = field(default=None, compare=False)
    #: model -> ``sm_client.ModelFloor`` (routable, floor, floor_headroom) of the SM.
    model_floors: Mapping[str, Any] = field(default_factory=dict, compare=False)
    #: ``/v2/state`` ``floor_enforced`` (None = not reported).
    floor_enforced: bool | None = field(default=None, compare=False)
    #: Why ``routable_ids`` is None (``routable_missing`` / ``routable_unavailable: ...``).
    routable_error: str | None = field(default=None, compare=False)
    #: ``/v2/state`` ``version``: the SM binding-store version the view was read at
    #: (every SM write - sleep, wake, hide, transfer - raises it). None = not reported.
    #: Keys the relay hold (:class:`RelayHold`).
    sm_version: int | None = field(default=None, compare=False)


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
    # Drain budget (s) passed to the SM sleep; ignored there since 2026-10-02 (the SM never
    # drains on any path: hide -> ack -> /sleep mode=abort). None = none passed.
    drain_budget_s: float | None = None
    # S5 (2026-09-30): ``pods`` of a pure-capacity wake are placement HINTS - the SM
    # picks the GPUs itself (registry placement policy) and substitutes a hint it
    # cannot wake. False = exactly these pods (SafeScale probes / commits).
    hint: bool = False
    # C1: the rescue target this scale-up belongs to (a CRITICAL receiver's wakes /
    # creates / relay wakes). Bookkeeping only: not part of the action's identity.
    rescue: RescuePlan | None = field(default=None, compare=False)


#: SM sleep path of the fast-loop "*_immediate" donors (CRIT donor, idle proactive,
#: low-fairness donor). The SM never drains (no sleep path does since 2026-10-02):
#: hide -> ack -> /sleep mode=abort, the reissue sidecar continues the cut-off requests.
IMMEDIATE_DONOR_SLEEP_PATH = "urgent"


@dataclass(frozen=True)
class RelayBasis:
    """The fleet-view inputs a relay was planned from (review P2-1, 2026-10-06): the
    SM state version of the view, the SM floor view (routable, floor, headroom) of
    both models, and the view's own time (``state_ms`` / ``fetched_ms``)."""

    sm_version: int | None
    donor_floor: Any
    receiver_floor: Any
    view_ms: int | None


def relay_basis(view: "ClusterView | None", donor: str, receiver: str) -> RelayBasis | None:
    """:class:`RelayBasis` of a donor -> receiver relay planned on ``view`` (None
    without a view: nothing to key a hold on)."""
    if view is None:
        return None
    floors = view.model_floors or {}
    return RelayBasis(
        sm_version=view.sm_version,
        donor_floor=floors.get(donor),
        receiver_floor=floors.get(receiver),
        view_ms=view.state_ms if view.state_ms is not None else view.fetched_ms,
    )


#: SM refusal codes (409) of a relay that say something about the fleet: the replica
#: floor of the SM's own view. Every other refusal - code-less (a 503 while the SM
#: shuts down, a 400), ``writer_busy``, ``routable_unknown`` - says nothing about it.
FLEET_RELAY_REFUSALS = frozenset({"floor_violation"})
#: Wake refusals inside a no-pair answer that pass without a fleet change (SM wake
#: error codes ``resident_loading`` / ``truth_unavailable`` / ``lease_conflict``, reason
#: ``resident_unknown``): a Pod still loading, a gpu-truth sample missing, a lease held.
TRANSIENT_RELAY_WAKE_REFUSALS = frozenset({"resident_loading", "truth_unavailable", "lease_conflict", "resident_unknown"})
#: SM transfer skip reasons only a fleet change undoes (``occupant_*``: the receiver's
#: GPUs hold another model / a hidden / a busy occupant; ``receiver_cap``).
FLEET_RELAY_SKIP_PREFIX = "occupant_"
FLEET_RELAY_SKIPS = frozenset({"receiver_cap"})


def relay_hold_reason(summary: Mapping[str, Any]) -> tuple[str, bool]:
    """(reason, fleet) of a relay answered with nothing done, from its transfer
    summary (review 2026-10-06 P2-1). ``fleet``: the answer follows from the fleet
    view the relay was planned from (``floor_violation``, ``clamped_by_floor``, a
    pair tried and failed, unfilled for ``occupant_*`` / ``receiver_cap`` only, an SM
    without the endpoint), so only a new SM state version or floor view releases its
    hold. Anything else is transient: any newer view releases it."""
    outcome = summary.get("outcome")
    if outcome == "unsupported":
        return "unsupported", True
    if outcome == "refused":
        code = summary.get("code")
        return str(code or "refused"), code in FLEET_RELAY_REFUSALS
    if summary.get("clamped_by_floor"):
        return "clamped_by_floor", True
    if summary.get("pairs"):
        return "no_pair_done", True
    skipped = sorted(str(key) for key in (summary.get("skipped") or {}))
    reason = "unfilled" + (f"({','.join(skipped)})" if skipped else "")
    for codes in summary.get("refusal_codes") or ():
        if set(map(str, codes)) & TRANSIENT_RELAY_WAKE_REFUSALS:
            return reason, False
    fleet = bool(skipped) and all(
        key.startswith(FLEET_RELAY_SKIP_PREFIX) or key in FLEET_RELAY_SKIPS for key in skipped
    )
    return reason, fleet


@dataclass(frozen=True)
class RelayHold:
    """A relay the SM answered with nothing done (``done == 0``: refused, vetoed,
    floor-clamped, unfilled - never an unknown outcome) is not planned again for the
    same donor -> receiver pair while the fleet view it was planned from is still
    current (review P2-1: no new writer-lock call for the same answer). A state gate,
    not a timer: for a fleet-derived answer (:func:`relay_hold_reason`) a new SM state
    version or a changed floor view of either model releases it; a transient answer,
    or a view without an SM version, is released by any newer view."""

    basis: RelayBasis
    reason: str
    fleet: bool = True

    def holds(self, current: RelayBasis | None) -> bool:
        if current is None:
            return False
        if not self.fleet or current.sm_version is None or self.basis.sm_version is None:
            return current == self.basis
        return replace(current, view_ms=None) == replace(self.basis, view_ms=None)


@dataclass(frozen=True)
class TransferIntent:
    """An immediate donor -> receiver relay expressed as a COUNT (2026-10-02, design
    20261002-controller-transfer). The controller decides the quantity - how many
    replicas, from which model, to which model, in which order; the service-manager
    (``POST /v2/transfers``) decides the placement - which donor pods, which receiver
    bindings, which GPUs - and keeps the replica floor and "one awake model per GPU"
    under its writer lock. No pod or GPU is named here.

    ``count`` = donor replicas to hand over (the SM's ``count``); ``pairs`` = receiver
    replicas the planner expects from them (equal unless a TP receiver needs several
    single-GPU donors). The SM may complete fewer: the ActionQueue accounts by the
    response's ``done`` / ``taken`` / ``unfilled``, never by ``count``."""

    donor_model: str
    receiver_model: str
    count: int
    reason: str
    source_loop: SourceLoop
    sleep_path: str = IMMEDIATE_DONOR_SLEEP_PATH
    pairs: int = 0
    #: C1: the rescue target the receiver side belongs to (bookkeeping only).
    rescue: RescuePlan | None = field(default=None, compare=False)
    #: The fleet-view inputs it was planned from (:class:`RelayHold`; None = no view).
    basis: RelayBasis | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.pairs <= 0:
            object.__setattr__(self, "pairs", int(self.count))

    @property
    def model(self) -> str:
        """The receiver (the model the relay serves; queue / rescue key)."""
        return self.receiver_model

    @property
    def donor(self) -> str:
        return self.donor_model

    @property
    def receiver(self) -> str:
        return self.receiver_model


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


Action = ScaleAction | TransferIntent | HideAction | UnhideAction | DefragAction | ShrinkForSlotAction


def upscale_of(action: object) -> tuple[str, int] | None:
    """(model, receiver replicas) a planned action adds, or None: a ScaleAction with a
    positive delta, or a TransferIntent's receiver side (its expected ``pairs``)."""
    if isinstance(action, ScaleAction) and action.delta > 0:
        return action.model, int(action.delta)
    if isinstance(action, TransferIntent):
        return action.receiver_model, int(action.pairs)
    return None


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
    ActionQueue as an ordered one-shot unit:

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
    probe_backoff_models: Mapping[str, str] | set[str] | None = None,
    preemptible_models: set[str] | None = None,
    rescue_bases: Mapping[str, RescueBasis] | None = None,
    view_pending: Mapping[str, str] | None = None,
    relay_holds: Mapping[tuple[str, str], RelayHold] | None = None,
) -> PlanResult:
    """One plan. Quantity only for the immediate relays (2026-10-02): a CRITICAL / LOW
    receiver's need is met first from free capacity (its sleeping bindings on free
    GPUs, then free slot groups), then by :class:`TransferIntent` s from IMMEDIATE
    donors (count only; the SM places them), then by middle-zone SafeScale probes. A
    donor gives at most its ``floor_headroom`` (SM count, ``_donor_headroom``) minus
    what this tick already took from it. Whatever the relays cannot cover is planned
    again on the next tick from a new view (free capacity first). ``relay_holds``
    ((donor, receiver) -> :class:`RelayHold`): pairs the SM answered with nothing
    done on the fleet view that is still current - not planned again."""
    active_probe_models = active_probe_models or set()
    # Review 3 P2-3: models whose only in-flight work is a SafeScale commit waiting
    # out a retry backoff. A CRITICAL receiver among them is still planned: the
    # queue preempts that retry when the rescue action is submitted.
    preemptible_models = preemptible_models or set()
    # Models whose last SafeScale probe rolled back and whose signal does not show new
    # evidence yet (timer cleanup 2026-10-02, replaces the A13 60 s backoff): model ->
    # hold reason (a plain set = reason "evidence"). No new HIGH proactive probe.
    probe_backoff_models = (
        dict(probe_backoff_models)
        if isinstance(probe_backoff_models, Mapping)
        else {model: "evidence" for model in probe_backoff_models or ()}
    )
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
    # S5: GPUs the SM reports not wakeable are no wake / create capacity this tick (the
    # S3 per-GPU wake cooldown was removed on 2026-10-02: a refusal is an event only).
    occupancy = _SlotOccupancy(cluster_view) if cluster_view is not None else None
    # Review F4 per-model cooldown (fallback since the timer cleanup 2026-10-02): model ->
    # direction ("up"/"down") of its last executed action whose effect the decision
    # window does not yet fully reflect - only for models O1 does not track this tick.
    # ``view_pending`` (O1 evidence gate): models O1 tracks whose last action completed
    # after the fleet view was fetched (no breakpoint can hold them yet), same rules.
    # (The P2-6 floor-violation hold was removed on 2026-10-02: the donor's SM
    # ``floor_headroom`` decides, and the SM clamps model-level shrinks.)
    cooldown = _Cooldown(cooldowns or {}, events, view_pending=view_pending)

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
    # O1 breakpoint window (signal_full_window present only with O1 on): a model whose
    # metrics window still holds a breakpoint (traffic onset, routable-count change)
    # takes part in no scale-down - neither HIGH / IDLE donor nor middle-zone donor - until
    # a whole window lies after the breakpoint (scale-down stays cautious); as a receiver
    # it acts once signal_warm (min_evidence_grids complete grids after the breakpoint).
    # Q3 (2026-10-06): an IDLE model whose whole current window is idle (``window_idle``:
    # tokens known and zero, no queue, every serving pod scraped - never a held context)
    # is exempt: no token at any replica count is the same evidence before and after
    # the breakpoint.
    # H3 (2026-10-06): O1 guards against over-scaling on a window that still describes
    # the old regime. On a free GPU an over-scale costs one wake, so a CRITICAL receiver
    # is not held: it takes free capacity only (its sleeping bindings on free GPUs, free
    # slot groups; no donor, no SafeScale probe, no TP shrink, no defrag) and one step
    # per decision, once its previous step is visible (``receiver_o1_exempt_pending``).
    # 2026-10-07: no longer gated on ``o1_queue_rise`` (still computed, logged only):
    # it compares ``waiting`` of the pods in both samples, and after a wake the old pods'
    # waiting drains to 0 while running stays high, so it never held after a wake and
    # the hold lasted 2 grids anyway. Without free capacity, or a LOW receiver: held.
    warmup_suppressed: list[str] = []
    breakpoint_held: list[str] = []
    o1_exempt: dict[str, Mapping[str, Any] | None] = {}
    kept: list = []
    for item in classifications:
        ctx = model_contexts.get(item.model_name, {})
        if getattr(item, "saturation_rescue", False):
            # Onset saturation rescue: CRITICAL exactly because the TSS is not warm yet
            # (its own consecutive-window confirmation replaces the warmup / dwell gates).
            kept.append(item)
            continue
        if item.role == ModelRole.RECEIVER:
            if not ctx.get("signal_warm", True):
                if item.state == ModelState.CRITICAL and ctx.get("signal_full_window") is False:
                    o1_exempt[item.model_name] = ctx.get("o1_queue_rise")
                    kept.append(item)
                    continue
                warmup_suppressed.append(item.model_name)
                continue
        elif ctx.get("signal_full_window", True) is False and not _idle_window_evidence(item, ctx):
            breakpoint_held.append(item.model_name)
            continue
        kept.append(item)
    if warmup_suppressed or breakpoint_held or o1_exempt:
        for model in warmup_suppressed:
            reason = model_contexts.get(model, {}).get("signal_hold_reason")
            events.append(
                f"receiver_held_breakpoint_window:{model}:{reason}"
                if reason
                else f"receiver_suppressed_signal_warmup:{model}"
            )
        events.extend(f"donor_suppressed_breakpoint_window:{model}" for model in breakpoint_held)
        classifications = kept
    # H3: exempt receivers that got a free-capacity step this tick (the others are
    # reported as held at the end of the rescue section).
    o1_exempt_planned: set[str] = set()

    critical_receivers = [item for item in classifications if item.state == ModelState.CRITICAL]
    # TSS-confirmed CRITICAL receivers first (their order unchanged); onset saturation
    # receivers after them, the most backlogged per replica (waiting / n) first; H3
    # O1-exempt receivers (the weakest evidence) last.
    critical_receivers.sort(key=lambda item: _critical_order(item, model_contexts, o1_exempt))
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

    def low_need(recv: ModelClassification) -> tuple[int, int] | None:
        """(replicas needed, of which wakeable from sleeping bindings), None = skip."""
        if recv.model_name in inflight_models:
            return None
        if deltas.get(recv.model_name, 0) > 0:
            # H1 (2026-10-06): already scaled up in this plan (the rescue section woke
            # it from sleeping capacity) - one LOW step per plan, never two.
            return None
        if cooldown.blocks(recv.model_name, "up"):
            return None
        recv_pods = _effective_routable_replicas(recv.model_name, model_contexts, model_replicas)
        recv_assigned = _effective_assigned_replicas(recv.model_name, model_contexts, model_replicas)
        receiver_capacity = (
            _max_replicas(cfg, recv.model_name)
            - _awake_replicas(recv.model_name, model_contexts, model_replicas)
        )
        if receiver_capacity <= 0:
            return None
        needed = min(_scale_step(recv_pods, cfg.scale_step_ratio), receiver_capacity)
        return needed, min(needed, max(0, recv_assigned - recv_pods))

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
            # Legacy one-step rescue (C1 off) only: without the C1 target bookkeeping the
            # F4 / O1 holds keep an unreflected scale-up from being repeated. With C1 the
            # rescue target ledger does (the opt-in scale_up_cooldown_enabled switch was
            # removed in the timer cleanup 2026-10-02).
            if not c1 and cooldown.blocks(recv.model_name, "up", critical=True):
                return None
            recv_pods = _effective_routable_replicas(recv.model_name, model_contexts, model_replicas)
            recv_assigned = _effective_assigned_replicas(recv.model_name, model_contexts, model_replicas)
            recv_max = _max_replicas(cfg, recv.model_name)
            # Cap on the awake count incl. hidden probe pods (v1 assigned = non-sleeping,
            # draining included; the SM counts the same), not on the routable count.
            recv_awake = _awake_replicas(recv.model_name, model_contexts, model_replicas)
            if recv_awake >= recv_max:
                return None
            if getattr(recv, "saturation_rescue", False):
                return saturation_need(recv, recv_pods, recv_assigned, recv_max, recv_awake)
            if not c1:
                raw_need = min(_scale_step(recv_pods, cfg.scale_step_ratio), recv_max - recv_awake)
                if raw_need <= 0:
                    return None
                return raw_need, min(raw_need, max(0, recv_assigned - recv_pods))
            # C1: one absolute target for the whole deficit. With an earlier scale-up
            # the window does not reflect yet, the target is computed from the replicas
            # the window's Z describes and only what exceeds that scale-up is planned.
            basis = rescue_bases.get(recv.model_name)
            exempt = recv.model_name in o1_exempt
            if exempt:
                # H3 review P1-2: one piece of queue evidence buys at most one wake until
                # that wake is visible - an earlier target the routable count does not
                # show yet, or a fleet view older than the last action (view-pending),
                # holds the exempt receiver (no timer: the next view / count releases it).
                if (basis is not None and basis.covered > recv_pods) or (
                    view_pending and recv.model_name in view_pending
                ):
                    events.append(
                        f"receiver_o1_exempt_pending:{recv.model_name}"
                        f":covered={basis.covered if basis is not None else None}:routable={recv_pods}"
                        f":view_pending={bool(view_pending and recv.model_name in view_pending)}"
                    )
                    return None
                # The earlier target has landed (the routable count shows it): the step
                # builds on the current count (the target is not re-asked).
                basis = None
            base = basis.base if basis is not None else recv_pods
            covered = max(basis.covered, recv_pods) if basis is not None else recv_pods
            desired = rescue_desired(
                base, recv.Z_m, recv.tau.tau_crit, cfg.rescue_max_step_ratio, cfg.rescue_max_step_pods
            )
            raw_need = min(desired - covered, recv_max - max(recv_awake, covered))
            recv_ctx = model_contexts.get(recv.model_name, {})
            evidence = recv_ctx.get("signal_evidence_requests")
            if (
                cfg.partial_window_max_step > 0
                and raw_need > cfg.partial_window_max_step
                and recv_ctx.get("signal_full_window") is False
                and (evidence is None or float(evidence) < cfg.partial_window_lowevidence_requests)
            ):
                # O1 review P2-1: a partial window with few completed requests is thin
                # evidence (tokens count at completion) - one step now; with enough
                # requests (or on a whole window) the whole deficit at once.
                events.append(
                    f"rescue_low_evidence_step:{recv.model_name}:{raw_need}->{cfg.partial_window_max_step}"
                    f":requests={evidence}"
                )
                raw_need = cfg.partial_window_max_step
            if exempt:
                # H3: Z is the held window's (old regime) value - one step per decision,
                # like a thin partial window; each step is a new breakpoint, and the next
                # waits until this one is visible (``receiver_o1_exempt_pending``).
                raw_need = min(raw_need, max(1, cfg.partial_window_max_step))
            if raw_need <= 0:
                if basis is not None:
                    events.append(
                        f"rescue_target_hold:{recv.model_name}:desired={desired}:covered={covered}"
                    )
                return None
            rescue_ctx[recv.model_name] = (desired, base, covered)
            return raw_need, min(raw_need, max(0, recv_assigned - recv_pods))

        def saturation_need(
            recv: ModelClassification, recv_pods: int, recv_assigned: int, recv_max: int, recv_awake: int
        ) -> tuple[int, int] | None:
            """Onset saturation rescue: bounded doubling of the routable count (no
            throughput evidence: never ``ceil(n * tau_crit / Z)``, which a zero Z sends
            to the cap). The O1 low-evidence step cap does not apply (it bounds a TSS
            decision on a thin partial window; this path is not a TSS decision and
            re-confirms saturation after every step instead). An earlier rescue target
            the routable count does not show yet counts as covered (C1 bookkeeping)."""
            basis = rescue_bases.get(recv.model_name)
            covered = max(basis.covered, recv_pods) if basis is not None else recv_pods
            ctx = model_contexts.get(recv.model_name, {})
            if ctx.get("saturation_reason") == "scrape_stale_inflight":
                # Frozen scrape (no engine sample, demand only): one replica per step.
                desired = min(recv_pods + 1, recv_max)
            else:
                desired = min(
                    max(recv_pods + 1, math.floor(cfg.saturation_max_step_factor * recv_pods + 1e-9)), recv_max
                )
            raw_need = min(desired - covered, recv_max - max(recv_awake, covered))
            kv = ctx.get("saturation_kv")
            events.append(
                f"saturation_rescue:{recv.model_name}:n={recv_pods}:target={desired}"
                f":waiting={_num_text(ctx.get('saturation_waiting'))}:kv={_num_text(kv, 2)}"
                f":reason={ctx.get('saturation_reason')}:ticks={ctx.get('saturation_ticks')}"
                f":planned_max={max(0, raw_need)}"
            )
            if raw_need <= 0:
                if basis is not None:
                    events.append(
                        f"rescue_target_hold:{recv.model_name}:desired={desired}:covered={covered}"
                    )
                return None
            rescue_ctx[recv.model_name] = (desired, recv_pods, covered)
            return raw_need, min(raw_need, max(0, recv_assigned - recv_pods))

        critical_needs ={recv.model_name: critical_need(recv) for recv in critical_receivers}
        # Every CRITICAL receiver's sleeping-binding wakes are assigned jointly up
        # front, so an earlier receiver never takes the one free slot a later one
        # can wake into while it had another (multi-receiver slot stealing).
        # H3 review P2-1: O1-exempt receivers (the weakest evidence) are not part of
        # it - their wakes are assigned in a second pass, after every other CRITICAL
        # receiver has planned its wakes and creates (like the LOW rescue section),
        # so they never take a free GPU a confirmed receiver needs.
        reserved_wakes = _plan_joint_wakes(
            occupancy,
            [
                (model, need[1])
                for model, need in critical_needs.items()
                if need is not None and model not in o1_exempt
            ],
            events=events,
        )
        reserved_exempt_wakes: dict[str, list[Binding]] | None = None
        for recv in critical_receivers:
            need = critical_needs[recv.model_name]
            if need is None:
                continue
            raw_need, wake_need = need
            first_action = len(actions)
            # H3: an O1-exempt receiver takes free capacity only.
            free_only = recv.model_name in o1_exempt
            if free_only and reserved_exempt_wakes is None:
                # Exempt receivers sort last (``_critical_order``): every confirmed
                # receiver is planned by now.
                reserved_exempt_wakes = _plan_joint_wakes(
                    occupancy,
                    [
                        (model, item[1])
                        for model, item in critical_needs.items()
                        if item is not None and model in o1_exempt
                    ],
                    events=events,
                )
            try:
                gain_from_sleeping, wake_pods = _take_reserved_wakes(
                    occupancy,
                    reserved_exempt_wakes if free_only else reserved_wakes,
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
                    same_slot_shrink = None if free_only else _try_plan_same_slot_high_shrink(
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
                        taken_serve_ids=occupancy.released_donor_ids() if occupancy is not None else set(),
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
                        # Ledger (2026-10-02): the tick commits this probe with exactly
                        # {beneficiary: 1} (_safescale_pending_upscales). Record it, or the
                        # fairness piggyback sees unclaimed == 1 and promises the same
                        # shrink to a LOW receiver whose upscale the tick then drops.
                        pending = probe_upscale_plans.setdefault(same_slot_shrink.donor, {})
                        pending[same_slot_shrink.beneficiary] = pending.get(same_slot_shrink.beneficiary, 0) + 1
                        if occupancy is not None:
                            occupancy.count_released((same_slot_shrink.serve_id,))
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
                            # H3: a migration moves other models - not free capacity.
                            defrag_enabled=cfg.defrag_enabled and not free_only,
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
                if free_only:
                    continue  # H3: no donor, no middle-zone probe
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
                    if cooldown.blocks(donor.model_name, "down"):
                        continue
                    headroom = _donor_headroom(cfg, donor.model_name, model_contexts, model_replicas)
                    if headroom <= 0:
                        continue
                    donor_pods = _effective_routable_replicas(donor.model_name, model_contexts, model_replicas)
                    planned_take = abs(min(deltas.get(donor.model_name, 0), 0))
                    gained = _plan_transfer_intent(
                        actions,
                        deltas,
                        occupancy,
                        donor=donor.model_name,
                        receiver=recv.model_name,
                        need=still_needed,
                        donor_limit=min(_donor_give(donor, donor_pods, cfg), headroom - planned_take),
                        reason="critical_donor_immediate",
                        source_loop="rescue",
                        events=events,
                        view=cluster_view,
                        relay_holds=relay_holds,
                    )
                    still_needed -= gained

                for middle in middle_zone:
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
                    headroom = _donor_headroom(cfg, middle.model_name, model_contexts, model_replicas)
                    if headroom <= 0:
                        continue
                    middle_pods = _effective_routable_replicas(middle.model_name, model_contexts, model_replicas)
                    planned_take = abs(min(deltas.get(middle.model_name, 0), 0))
                    gained = _plan_middle_zone_probe(
                        actions,
                        deltas,
                        occupancy,
                        donor=middle.model_name,
                        receiver=recv.model_name,
                        need=still_needed,
                        donor_limit=min(_scale_step(middle_pods, cfg.scale_step_ratio), headroom - planned_take),
                        reason="critical_middle_zone_safescale",
                        source_loop="rescue",
                        events=events,
                        delayed_down_models=delayed_down_models,
                        probe_upscale_plans=probe_upscale_plans,
                    )
                    still_needed -= gained
            finally:
                if recv.model_name in rescue_ctx:
                    _tag_rescue_actions(actions, first_action, recv, rescue_ctx[recv.model_name], events)
                if free_only:
                    planned = sum(
                        up[1]
                        for up in (upscale_of(action) for action in actions[first_action:])
                        if up is not None and up[0] == recv.model_name
                    )
                    if planned > 0:
                        o1_exempt_planned.add(recv.model_name)
                        events.append(
                            _o1_exempt_event(recv.model_name, model_contexts, o1_exempt[recv.model_name], planned)
                        )

        # H1 (2026-10-06): a LOW receiver with a sleeping binding on a free GPU is woken
        # here, right after the snapshot is published, instead of waiting for the
        # fairness loop's cadence (up to 10 s). Only free sleeping capacity, and only
        # after every CRITICAL receiver has claimed its wakes; creates, donor relays and
        # middle-zone probes for LOW stay with the fairness loop. A LOW model woken here
        # is not planned again by the fairness section of this plan (``low_need``) and
        # is in flight / O1-held for the next fairness tick, like any other scale-up.
        # Without a fleet view no GPU is known to be free: LOW stays with fairness.
        rescue_low_needs = (
            {recv.model_name: low_need(recv) for recv in low_receivers} if occupancy is not None else {}
        )
        reserved_rescue_low = _plan_joint_wakes(
            occupancy,
            [(model, need[1]) for model, need in rescue_low_needs.items() if need is not None],
            events=events,
        )
        for recv in low_receivers:
            need = rescue_low_needs.get(recv.model_name)
            if need is None:
                continue
            gain, wake_pods = _take_reserved_wakes(
                occupancy,
                reserved_rescue_low,
                receiver=recv.model_name,
                need=need[1],
                # Not reported here: the fairness loop reports a blocked LOW wake.
                events=[],
                blocked_event="low_rescue_sleeping_blocked",
            )
            if gain > 0:
                _add_scale_action(
                    actions,
                    deltas,
                    model=recv.model_name,
                    delta=gain,
                    reason="low_rescue_sleeping_capacity",
                    source_loop="rescue",
                    receiver=recv.model_name,
                    pods=wake_pods,
                    hint=True,
                )

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
            # A model-level scale-down (PUT /target, no receiver, no pod named): the SM
            # picks the replicas and clamps the shrink at the model's replica floor
            # (2026-10-02); the planner still never asks beyond the SM floor_headroom.
            # Q3 (2026-10-06): the whole surplus down to the floor in one decision.
            shrink = min(
                pods - idle_min,
                _donor_headroom(cfg, idle.model_name, model_contexts, model_replicas),
            )
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
            # Rollback evidence hold, checked (and logged) only for a model that would
            # otherwise be probed - a model already at its floor stays silent every tick.
            if high.model_name in probe_backoff_models:
                events.append(
                    f"safescale_rollback_hold:{high.model_name}:{probe_backoff_models[high.model_name]}"
                )
                continue
            shrink = min(
                _scale_step(pods, cfg.scale_step_ratio),
                pods - high_min,
                _donor_headroom(cfg, high.model_name, model_contexts, model_replicas),
            )
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
    for model in o1_exempt:
        if model not in o1_exempt_planned:
            # H3: no free-capacity step (none free, the step would need a donor, nothing
            # needed, or no rescue this tick): held, reported as before.
            reason = model_contexts.get(model, {}).get("signal_hold_reason")
            events.append(f"receiver_held_breakpoint_window:{model}:{reason}")

    if not cfg.fairness_due:
        events.append("fairness_skipped_by_cadence")
        return PlanResult(actions, delayed_down_models, probe_upscale_plans, events=events)

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
            if cooldown.blocks(donor.model_name, "down"):
                continue
            headroom = _donor_headroom(cfg, donor.model_name, model_contexts, model_replicas)
            if headroom <= 0:
                continue
            needed = _piggyback_probe(
                donor.model_name, recv.model_name, needed, deltas, delayed_down_models, probe_upscale_plans
            )
            if needed <= 0:
                continue
            donor_pods = _effective_routable_replicas(donor.model_name, model_contexts, model_replicas)
            planned_take = abs(min(deltas.get(donor.model_name, 0), 0))
            needed -= _plan_transfer_intent(
                actions,
                deltas,
                occupancy,
                donor=donor.model_name,
                receiver=recv.model_name,
                need=needed,
                # Q3: an IDLE donor gives its whole surplus, a HIGH donor one step.
                donor_limit=min(
                    donor_pods if donor.state == ModelState.IDLE else _scale_step(donor_pods, cfg.scale_step_ratio),
                    headroom - planned_take,
                ),
                reason="low_fairness_donor_immediate",
                source_loop="fairness",
                events=events,
                view=cluster_view,
                relay_holds=relay_holds,
            )

        for middle in middle_zone:
            if needed <= 0:
                break
            if middle.model_name == recv.model_name or middle.model_name in active_probe_models or middle.model_name in inflight_models:
                continue
            if cooldown.blocks(middle.model_name, "down"):
                continue
            headroom = _donor_headroom(cfg, middle.model_name, model_contexts, model_replicas)
            if headroom <= 0:
                continue
            needed = _piggyback_probe(
                middle.model_name, recv.model_name, needed, deltas, delayed_down_models, probe_upscale_plans
            )
            if needed <= 0:
                continue
            donor_pods = _effective_routable_replicas(middle.model_name, model_contexts, model_replicas)
            planned_take = abs(min(deltas.get(middle.model_name, 0), 0))
            needed -= _plan_middle_zone_probe(
                actions,
                deltas,
                occupancy,
                donor=middle.model_name,
                receiver=recv.model_name,
                need=needed,
                donor_limit=min(_scale_step(donor_pods, cfg.scale_step_ratio), headroom - planned_take),
                reason="low_fairness_middle_zone_safescale",
                source_loop="fairness",
                events=events,
                delayed_down_models=delayed_down_models,
                probe_upscale_plans=probe_upscale_plans,
            )

    return PlanResult(actions, delayed_down_models, probe_upscale_plans, events=events)


def _piggyback_probe(
    donor: str,
    receiver: str,
    needed: int,
    deltas: Mapping[str, int],
    delayed_down_models: set[str],
    probe_upscale_plans: dict[str, dict[str, int]],
) -> int:
    """A SafeScale probe this tick already plans on ``donor`` whose freed replicas no
    receiver has claimed yet serves this receiver first (piggyback). Returns what is
    still needed."""
    existing_shrink = abs(min(deltas.get(donor, 0), 0))
    existing_claimed = sum(probe_upscale_plans.get(donor, {}).values())
    unclaimed = existing_shrink - existing_claimed
    if unclaimed > 0 and donor in delayed_down_models:
        piggyback = min(needed, unclaimed)
        pending = probe_upscale_plans.setdefault(donor, {})
        pending[receiver] = pending.get(receiver, 0) + piggyback
        needed -= piggyback
    return needed


def _plan_transfer_intent(
    actions: list[Action],
    deltas: dict[str, int],
    occupancy: "_SlotOccupancy | None",
    *,
    donor: str,
    receiver: str,
    need: int,
    donor_limit: int,
    reason: str,
    source_loop: SourceLoop,
    events: list[str],
    view: ClusterView | None = None,
    relay_holds: Mapping[tuple[str, str], RelayHold] | None = None,
) -> int:
    """One immediate relay as a :class:`TransferIntent` (count only). With a cluster
    view its size is bounded by :meth:`_SlotOccupancy.pairable_count` - pairs the SM
    can form (receiver bindings whose every GPU this donor holds awake) - so no intent
    is sent that cannot be filled; without one, by ``need`` / ``donor_limit``.
    No relay while the SM reported its routable view unreadable (``routable_error``
    other than ``routable_missing``, a state without the field: the SM refuses the
    relay with 409 ``routable_unknown`` - ``relay_skipped_routable_unknown``) or while the pair is held (:class:`RelayHold`:
    ``relay_held:<donor>:<receiver>:<reason>``).
    Returns the receiver replicas it is expected to add."""
    if need <= 0 or donor_limit <= 0:
        return 0
    if view is not None and view.routable_error and view.routable_error != "routable_missing":
        _event_once(events, f"relay_skipped_routable_unknown:{donor}:{receiver}")
        return 0
    basis = relay_basis(view, donor, receiver)
    hold = (relay_holds or {}).get((donor, receiver))
    if hold is not None and hold.holds(basis):
        _event_once(events, f"relay_held:{donor}:{receiver}:{hold.reason}")
        return 0
    if occupancy is None:
        pairs = count = min(need, donor_limit)
    else:
        pairs, count = occupancy.pairable_count(donor, receiver, max_pairs=need, max_donors=donor_limit)
        if pairs <= 0:
            event = f"donor_no_slot_match:{donor}:{receiver}"
            if event not in events:
                events.append(event)
            return 0
    deltas[donor] = deltas.get(donor, 0) - count
    deltas[receiver] = deltas.get(receiver, 0) + pairs
    actions.append(
        TransferIntent(
            donor_model=donor,
            receiver_model=receiver,
            count=count,
            pairs=pairs,
            reason=reason,
            source_loop=source_loop,
            sleep_path=IMMEDIATE_DONOR_SLEEP_PATH,
            basis=basis,
        )
    )
    return pairs


def _event_once(events: list[str], event: str) -> None:
    if event not in events:
        events.append(event)


def _plan_middle_zone_probe(
    actions: list[Action],
    deltas: dict[str, int],
    occupancy: "_SlotOccupancy | None",
    *,
    donor: str,
    receiver: str,
    need: int,
    donor_limit: int,
    reason: str,
    source_loop: SourceLoop,
    events: list[str],
    delayed_down_models: set[str],
    probe_upscale_plans: dict[str, dict[str, int]],
) -> int:
    """A middle-zone donor's SafeScale probe for ``receiver`` (unchanged path: SafeScale
    hides named pods). With a cluster view the probe pods are the donor pods of the
    pairs :meth:`_SlotOccupancy.pairable` counts (sleeping them frees receiver GPUs).
    Returns the receiver replicas promised to the commit's follow-up upscale."""
    if need <= 0 or donor_limit <= 0:
        return 0
    pods: tuple[str, ...] = ()
    if occupancy is None:
        pairs = count = min(need, donor_limit)
    else:
        estimate = occupancy.pairable(donor, receiver, max_pairs=need, max_donors=donor_limit)
        if estimate.pairs <= 0:
            event = f"donor_no_slot_match:{donor}:{receiver}"
            if event not in events:
                events.append(event)
            return 0
        pairs, count, pods = estimate.pairs, estimate.donors, estimate.donor_pods
    _add_scale_action(
        actions,
        deltas,
        model=donor,
        delta=-count,
        reason=reason,
        source_loop=source_loop,
        requires_safescale=True,
        donor=donor,
        receiver=receiver,
        pods=pods,
    )
    delayed_down_models.add(donor)
    pending = probe_upscale_plans.setdefault(donor, {})
    pending[receiver] = pending.get(receiver, 0) + pairs
    return pairs


class _Cooldown:
    """Per-model holds of the planner (one gate, two sources):

    * review F4 (``cooldowns``; fallback since the timer cleanup 2026-10-02 - only for
      models O1 does not track): hold a model's next action until a fresh metrics window
      reflects its last executed one;
    * O1 evidence gate (``view_pending``): the fleet view predates the model's last
      action, so no breakpoint holds it yet - held until a newer view exists.

    (The P2-6 floor-violation hold - a timed hold after a 409 ``floor_violation`` - was
    removed on 2026-10-02: a donor gives at most its SM ``floor_headroom``.)

    F4 / view-pending direction rules: same direction is held; after a scale-up a
    scale-down is held too; after a scale-down a scale-up is allowed only for a CRITICAL
    receiver (safety)."""

    def __init__(
        self,
        cooldowns: Mapping[str, str],
        events: list[str],
        *,
        view_pending: Mapping[str, str] | None = None,
    ) -> None:
        self._cooldowns = dict(cooldowns)
        self._events = events
        self._view_pending = dict(view_pending or {})

    def blocks(self, model: str, direction: str, *, critical: bool = False) -> bool:
        for holds, name in ((self._cooldowns, "cooldown_hold"), (self._view_pending, "o1_view_pending_hold")):
            last = holds.get(model)
            if last is None:
                continue
            if direction == "up" and last == "down" and critical:
                continue
            self._event(f"{name}:{model}")
            return True
        return False

    def _event(self, event: str) -> None:
        if event not in self._events:
            self._events.append(event)

    def down_blocked(self) -> set[str]:
        return set(self._cooldowns) | set(self._view_pending)


class _SlotOccupancy:
    """GPU-slot occupancy for wake planning within one build_plan call.

    Under multi-model-per-GPU residency a receiver's sleeping binding is only real
    capacity when no other binding is awake on its GPU(s); the SM rejects any other wake
    with WakeConflict (the E1 dsllama-8b deadlock). Slots claimed by a planned wake (or
    counted for a planned relay) are not counted twice in one tick.

    Since 2026-10-02 it only COUNTS relay capacity (:meth:`pairable_count`): which donor
    pods and receiver bindings a relay uses is the service-manager's choice
    (``POST /v2/transfers``); nothing it counts here is sent to the SM.
    """

    def __init__(self, cluster_view: ClusterView) -> None:
        self._topology = cluster_view.topology
        self._nodes = node_gpu_counts(cluster_view.topology)
        #: GPUs that are no wake / create capacity this tick although no awake binding
        #: holds them: S5 the SM reports them not wakeable (a Pod loading, a wake in
        #: flight, gpu-truth in use).
        self._blocked: set[tuple[str, int]] = {
            (str(node), int(gpu)) for node, gpu in set(cluster_view.blocked_gpus)
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
        # Donor bindings this tick's plan already counts as released (relay estimates,
        # a same-slot shrink's binding): never counted a second time in the same tick.
        # Counting only - the SM picks the donors of a relay itself.
        self._released: set[str] = set()
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

    def count_released(self, serve_ids) -> None:
        """Count donor bindings as released by this tick's plan (estimate only)."""
        self._released.update(serve_ids)

    def released_donor_ids(self) -> set[str]:
        return set(self._released)

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
        TP-pair reservation ``placement.reserve_tp_pairs``, node balance;
        tre_common.gpu_placement), so waking three
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

    def pairable(
        self, donor: str, receiver: str, *, max_pairs: int, max_donors: int
    ) -> "PairEstimate":
        """How many relay pairs ``donor`` -> ``receiver`` the SM can form (the same rule
        as its pair selection, design 20261002-sm-transfer section 3): a sleeping, not
        hidden receiver binding on GPUs not claimed / blocked, EVERY one of which an
        awake, not hidden binding of ``donor`` holds (a GPU without one is
        ``uncovered_gpu``: no relay; a GPU with no occupant at all is a plain wake).
        A pair consumes all its occupants (a TP receiver may take two single-GPU donors).
        At most ``max_pairs`` pairs and ``max_donors`` donor replicas; pairs with fewer
        donors first. The pairs found are counted (receiver GPUs claimed, donors
        released) so a later relay / wake of this tick does not count them again."""
        if max_pairs <= 0 or max_donors <= 0:
            return PairEstimate(0, 0, ())
        candidates: list[tuple[Binding, tuple[Binding, ...]]] = []
        for receiver_binding in self.sleeping(receiver):
            gpus = self._gpus(receiver_binding)
            if any(gpu in self._claimed or gpu in self._blocked for gpu in gpus):
                continue
            if not all(gpu in self._awake for gpu in gpus):
                continue  # no occupant (plain wake) or uncovered_gpu (no relay)
            occupants = {self._awake[gpu].serve_id: self._awake[gpu] for gpu in gpus}
            if any(
                occupant.model != donor or occupant.hidden or occupant.serve_id in self._released
                for occupant in occupants.values()
            ):
                continue
            candidates.append(
                (receiver_binding, tuple(sorted(occupants.values(), key=lambda b: natural_key(b.serve_id))))
            )
        candidates.sort(key=lambda item: (len(item[1]), natural_key(item[0].serve_id)))
        pairs = donors = 0
        donor_pods: list[str] = []
        for receiver_binding, occupants in candidates:
            if pairs >= max_pairs:
                break
            ids = [occupant.serve_id for occupant in occupants]
            if any(serve_id in self._released for serve_id in ids) or donors + len(ids) > max_donors:
                continue
            self.claim(receiver_binding)
            self._released.update(ids)
            pairs += 1
            donors += len(ids)
            donor_pods.extend(ids)
        return PairEstimate(pairs, donors, tuple(donor_pods))

    def pairable_count(
        self, donor: str, receiver: str, *, max_pairs: int, max_donors: int
    ) -> tuple[int, int]:
        """(pairs, donor replicas) of :meth:`pairable`: the size of a relay intent that
        the SM can fill on this view (an intent larger than that would come back
        ``unfilled``)."""
        estimate = self.pairable(donor, receiver, max_pairs=max_pairs, max_donors=max_donors)
        return estimate.pairs, estimate.donors


@dataclass(frozen=True)
class PairEstimate:
    """:meth:`_SlotOccupancy.pairable`: receiver replicas (``pairs``), donor replicas
    they consume (``donors``) and those donor pods (used only to name a middle-zone
    SafeScale probe's pods; a relay intent names none)."""

    pairs: int
    donors: int
    donor_pods: tuple[str, ...]


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
            continue  # this tick already counts this pod as released for another receiver
        # Takes already planned this tick (e.g. a critical_donor_immediate of an earlier
        # receiver) count against the donor's floor headroom too.
        planned_take = abs(min((planned_deltas or {}).get(binding.model, 0), 0))
        if _donor_headroom(cfg, binding.model, model_contexts, model_replicas) - planned_take <= 0:
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


def _idle_window_evidence(item: ModelClassification, ctx: Mapping[str, Any]) -> bool:
    """Q3 (2026-10-06): an IDLE model whose whole current window is idle evidence
    (``window_idle``, set by the tick only from this tick's fully scraped window - a
    held, scrape-stale or tokens-missing context never carries it)."""
    return item.state == ModelState.IDLE and ctx.get("window_idle") is True


def _paper_state_incomplete_models(classifications: list[ModelClassification]) -> tuple[str, ...]:
    return tuple(
        item.model_name
        for item in classifications
        if item.state == ModelState.UNKNOWN
        or (
            item.Z_m is None
            and item.state != ModelState.IDLE
            and not getattr(item, "signal_idle", False)
            # Onset saturation rescue: CRITICAL on the engine gauges, Z not defined yet.
            and not getattr(item, "saturation_rescue", False)
        )
    )


def _o1_exempt_event(
    model: str, model_contexts: Mapping[str, Any], rise: Mapping[str, Any] | None, planned: int
) -> str:
    """H3: ``receiver_o1_exempt_free_gpu`` with the hold it skipped, the breakpoint and
    the queue-rise samples (log only, ``none`` when the queue did not rise)."""
    ctx = model_contexts.get(model) or {}
    queue = rise or {}
    return (
        f"receiver_o1_exempt_free_gpu:{model}:{ctx.get('signal_hold_reason')}:planned={planned}"
        f":bp={queue.get('breakpoint_ms', ctx.get('signal_breakpoint_ms'))}"
        f":q={_num_text(queue.get('base_q'), 1)}->{_num_text(queue.get('q'), 1)}"
        f":waiting={_num_text(queue.get('base_waiting'))}->{_num_text(queue.get('waiting'))}"
        f":sample_ms={queue.get('base_ms')}->{queue.get('sample_ms')}"
        f":queue_rise={'yes' if rise else 'no'}"
    )


def _critical_order(
    item: ModelClassification,
    model_contexts: Mapping[str, Any],
    o1_exempt: Mapping[str, Any] | None = None,
) -> tuple[int, float]:
    if o1_exempt and item.model_name in o1_exempt:
        return (2, 0.0)
    if not getattr(item, "saturation_rescue", False):
        return (0, 0.0)
    ctx = model_contexts.get(item.model_name) or {}
    try:
        waiting = float(ctx.get("saturation_waiting") or 0.0)
        pods = max(1, int(ctx.get("routable_pods") or 0))
    except (TypeError, ValueError):
        return (1, 0.0)
    return (1, -waiting / pods)


def _num_text(value: Any, digits: int = 0) -> str:
    if value is None:
        return "none"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "none"


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
    sleep_path: str | None = None,
    drain_budget_s: float | None = None,
    hint: bool = False,
) -> None:
    """``sleep_path`` / ``drain_budget_s`` go to the SM sleep of a negative delta
    (None = the dispatcher / SM default). The SM never drains on any path (since
    2026-10-02): ``drain_budget_s`` is passed through and ignored there."""
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
    tick, before its floor. An IDLE donor: all of it (Q3 2026-10-06, code rule - an
    idle window is evidence that does not depend on the replica count). A HIGH donor,
    default (and legacy): one step - the paper's bounded pairwise transfer moves at
    most one step per pair per tick, the donor side included;
    ``donor_surplus_release`` (opt-in): the replicas above ``ceil(n * tau_high / Z)``
    (its projected Z stays >= tau_high), never less than one step. The relay is capped
    by what the receiver still needs and the donor's floor headroom either way (caller)."""
    step = _scale_step(donor_pods, cfg.scale_step_ratio)
    if donor.state == ModelState.IDLE:
        return max(step, donor_pods)
    if cfg.rescue_max_step_ratio <= 0 or not cfg.donor_surplus_release:
        return step
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
    planned = 0
    for action in actions[first:]:
        up = upscale_of(action)
        if up is not None and up[0] == model:
            planned += up[1]
    plan = RescuePlan(target=covered + planned, desired=desired, base=base, covered=covered)
    for index in range(first, len(actions)):
        action = actions[index]
        up = upscale_of(action)
        if up is not None and up[0] == model:
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


def _donor_headroom(
    cfg: PlanConfig,
    model_name: str,
    model_contexts: Mapping[str, Mapping[str, Any]],
    model_replicas: Mapping[str, int],
) -> int:
    """Replicas ``model_name`` may give before its replica floor (2026-10-02): the SM's
    ``floor_headroom`` (``/v2/state``: routable - enforced min_replicas, the count its
    floor check uses) when the tick context carries it, never more than the routable
    count minus the registry ``min_replicas`` (the SM floor is 0 while it is not
    enforced; the planner's own floor still holds then). Without the SM value (an
    unreadable or older ``/v2/state``): routable - min_replicas, as before."""
    own = _effective_routable_replicas(model_name, model_contexts, model_replicas) - _min_replicas(cfg, model_name)
    sm = (model_contexts.get(model_name) or {}).get("floor_headroom")
    if sm is None:
        return max(0, own)
    try:
        return max(0, min(int(sm), own))
    except (TypeError, ValueError):
        return max(0, own)


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

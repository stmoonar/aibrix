from __future__ import annotations

import re

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping

from tre_common.registry import ClusterTopology, tp_size_error
from tre_controller.planning.classify import (
    ModelClassification,
    ModelRole,
    ModelState,
    donor_mock_cost_key,
    receiver_sort_key,
)
from tre_common.gpu_placement import (
    GpuBlock,
    PlacementPolicy,
    choose_placement,
    choose_release,
    plan_placements,
)
from tre_sm.allocator.slots import (
    Binding,
    Slot,
    SlotAllocator,
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
    #: L3 (design 20261028-single-loop): every receiver gets at most
    #: ``max(1, ceil(scale_step_ratio * n))`` replicas per plan (v1 ``scale_step``; +1
    #: below 10 replicas), in the rescue and the rebalance section.
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
    # (HIGH/CRITICAL) donors. Demand-driven preemption (IDLE immediate donors, HIGH donor
    # probes for a receiver and a TP receiver's block, L4 rev 2 -> CRITICAL
    # beneficiary) is a separate path and is intentionally NOT gated. Default OFF since
    # the v1/paper alignment (A2): the path is v1's paper_high_proactive_shrink (rescue
    # tick, HIGH, replicas > floor, no active probe, not moved by another path this
    # tick -> SafeScale shrink by one step), now
    # protected by the SafeScale KV-cache / donor-health guards and the rollback evidence
    # hold (a rolled-back model is re-probed only on new evidence, timer cleanup 2026-10-02).
    # TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE=1 re-enables the guard.
    suppress_hot_proactive_probe: bool = False
    disable_eta_gate: bool = False
    # No SafeScale (TRE_ABLATION_DISABLE_SAFESCALE, 2026-10-08; set by the tick when it has
    # no SafeScale controller): a middle-zone / HIGH donor for a receiver is planned as
    # an immediate relay (reason suffix ``_nosafescale``) through the same gates as an
    # IDLE donor's (relay holds, routable view, C1 tagging), instead of a probe.
    release_without_safescale: bool = False
    # Registry placement.defrag.enabled: gates the critical_tp_defrag migration plan.
    # Off by default, as in v1 (design 20260928-placement-node-balance).
    defrag_enabled: bool = False

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
    #: a routable-count change it shows happened at or before it.
    #: None = a synthetic view (tests / offline).
    fetched_ms: int | None = field(default=None, compare=False)
    #: Timer cleanup review P2-2: epoch ms at or AFTER which the SM state of this view was
    #: produced (a lower bound): the time the controller sent the request (controller
    #: clock - timestamps are never compared across machines). An action that completed
    #: at or before it is shown by the view (view-pending guard). None = unknown
    #: (synthetic views): ``fetched_ms`` is used.
    state_ms: int | None = field(default=None, compare=False)
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
    """C1 bookkeeping of a rescue scale-up (metadata on its ScaleActions).

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
    reflect yet (C1): the next target is ``base + step`` (``base`` = the replicas the
    window still describes) and only the part above ``covered`` (what that scale-up
    achieved) is planned - an unrefreshed window yields the same target, so nothing is
    added until the window moves (one step per plan, 2026-10-28)."""

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
    # S5 (2026-09-30): the planner picked ``pods`` of a pure-capacity wake by placement.
    # Since 2026-10-28 (sm-plans section 10) every wake with pods is sent NAMED (the
    # controller names every pod, the SM validates): the flag is bookkeeping only.
    hint: bool = False
    # C1: the rescue target this scale-up belongs to (a CRITICAL receiver's wakes /
    # creates / relay wakes). Bookkeeping only: not part of the action's identity.
    rescue: RescuePlan | None = field(default=None, compare=False)
    # Plan group (sm-plans section 10): the items of one transfer - its donor sleeps and
    # its receiver wake - are atomic in the SM's selection. None = no group.
    group: str | None = None
    # The fleet-view inputs a transfer's receiver wake was planned from (RelayHold).
    basis: "RelayBasis | None" = field(default=None, compare=False)


#: SM sleep path of the rescue-section "*_immediate" donors (CRIT donor, idle proactive,
#: low-fairness donor). The SM never drains (no sleep path does since 2026-10-02):
#: hide -> ack -> /sleep mode=abort, the reissue sidecar continues the cut-off requests.
IMMEDIATE_DONOR_SLEEP_PATH = "urgent"
#: Reason suffix of a shrink that would have run as a SafeScale probe and is released
#: immediately because SafeScale is off (TRE_ABLATION_DISABLE_SAFESCALE, 2026-10-08).
NO_SAFESCALE_REASON_SUFFIX = "_nosafescale"


#: Refusal codes of a NAMED transfer (sm-plans section 10) that follow from the SM's
#: own books - the replica floor, the scaling cap, another binding awake on the
#: receiver GPUs (``slot_occupied``): only a new SM state version or floor view
#: releases its hold. Everything else (code-less, ``writer_busy``,
#: ``routable_unknown``, ``donor_sleep_failed``, the transient wake refusals) passes
#: with any newer view.
FLEET_NAMED_TRANSFER_REFUSALS = frozenset({"floor_violation", "max_awake_replicas", "slot_occupied"})


def named_transfer_hold_reason(error: object, conflict: Mapping[str, Any] | None = None) -> tuple[str, bool]:
    """(reason, fleet) of a named transfer answered with nothing done, from its item
    error (``"<status>: <code> (<detail>)"``; a ``group_refused`` detail names the
    refused item's own code) and its first wake refusal."""
    text = str(error or "")
    words = {word.strip("():,;'\"") for word in text.replace(":", " ").split()}
    words |= {str(value) for value in (conflict or {}).values() if isinstance(value, str)}
    found = [code for code in re.findall(r"(?:refused|failed): ([a-z_]+)", text) if code != "group_refused"]
    reason = found[0] if found else (text.split(" (")[0].replace(": ", ":") or "refused")
    if words & TRANSIENT_RELAY_WAKE_REFUSALS:
        return reason, False
    return reason, bool(words & FLEET_NAMED_TRANSFER_REFUSALS)


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


#: Wake refusals of a transfer that pass without a fleet change (SM wake error codes
#: ``resident_loading`` / ``truth_unavailable`` / ``lease_conflict``, reason
#: ``resident_unknown``): a Pod still loading, a gpu-truth sample missing, a lease held.
TRANSIENT_RELAY_WAKE_REFUSALS = frozenset({"resident_loading", "truth_unavailable", "lease_conflict", "resident_unknown"})


@dataclass(frozen=True)
class RelayHold:
    """A relay the SM answered with nothing done (its transfer group refused - never an
    unknown outcome) is not planned again for the same donor -> receiver pair while
    the fleet view it was planned from is still current (review P2-1: no new
    writer-lock call for the same answer). A state gate, not a timer: for a
    fleet-derived answer (:func:`named_transfer_hold_reason`) a new SM state
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
class BlockSupplyAction:
    """A transfer: the receiver's binding on a block taken from donors - a TP
    receiver's buddy block (design 20261028-single-loop L4 rev 2,
    :func:`cheapest_block`) or a TP1 relay pair (:func:`_plan_transfers`). The donor
    pods on the block sleep and the receiver wakes there (``receiver_pods``: its
    binding, a named wake) - ONE plan group (design 20261027-sm-plans section 10,
    :func:`tre_controller.loops.tick._release_without_safescale`). ``sleeps`` sleep
    at once (IDLE donors, and every donor of a scope that releases without a probe);
    ``probe`` (scope ``all``: the block's one HIGH / HEALTHY donor pod) is a SafeScale
    probe whose commit sleeps it and wakes the receiver."""

    receiver: str
    block: GpuBlock
    receiver_pods: tuple[str, ...]
    #: (donor model, serve id) of the pods that sleep at once.
    sleeps: tuple[tuple[str, str], ...]
    reason: str
    source_loop: SourceLoop
    #: (donor model, serve id) of the pod a SafeScale probe hides (scope ``all``).
    probe: tuple[str, str] | None = None
    #: C1: the receiver's rescue target (bookkeeping only).
    rescue: RescuePlan | None = field(default=None, compare=False)
    #: The fleet-view inputs a relay pair was planned from (:class:`RelayHold`).
    basis: RelayBasis | None = field(default=None, compare=False)

    @property
    def donor_models(self) -> tuple[str, ...]:
        pods = (*self.sleeps, *((self.probe,) if self.probe else ()))
        return tuple(dict.fromkeys(model for model, _ in pods))


Action = ScaleAction | HideAction | UnhideAction | DefragAction | BlockSupplyAction


def upscale_of(action: object) -> tuple[str, int] | None:
    """(model, receiver replicas) a planned action adds, or None: a ScaleAction with a
    positive delta (a transfer's receiver: :func:`receiver_upscales`)."""
    if isinstance(action, ScaleAction) and action.delta > 0:
        return action.model, int(action.delta)
    return None


@dataclass(frozen=True)
class ReceiverTarget:
    """A SafeScale follow-up upscale as an ABSOLUTE target (review 3 P2-1). A
    retry re-sends the same target, so an SM call that succeeded but timed out
    on the client is never applied twice. ``target`` None (the planned form,
    review 4 P2-1): resolved ONCE at the first dispatch from the SM's current
    awake count (never from a possibly stale cluster view) plus ``delta``,
    capped at ``cap`` (the scaling cap), then frozen. ``hints``: the receiver's
    sleeping bindings on the GPUs the commit's donor pods free (L4, design
    20261028-single-loop L4 rev 2: a block probe wakes the receiver on its block)."""

    model: str
    delta: int
    target: int | None = None
    cap: int | None = None
    hints: tuple[str, ...] = ()


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
    probe_backoff_models: Mapping[str, str] | set[str] | None = None,
    preemptible_models: set[str] | None = None,
    rescue_bases: Mapping[str, RescueBasis] | None = None,
    view_pending: Mapping[str, str] | None = None,
    relay_holds: Mapping[tuple[str, str], RelayHold] | None = None,
) -> PlanResult:
    """One plan. Quantity only for the immediate relays (2026-10-02): a CRITICAL / LOW
    receiver's need is met first from free capacity (its sleeping bindings on free
    GPUs, then free slot groups), then by named relays from IDLE donors (one transfer
    per pair, the controller names the pods - sm-plans section 10), then by SafeScale probes of HIGH and middle-zone
    donors (F1-B 2026-10-07: no HIGH donor is released without a probe). A
    donor gives at most its ``floor_headroom`` (SM count, ``_donor_headroom``) minus
    what this tick already took from it. Whatever the relays cannot cover is planned
    again on the next tick from a new view (free capacity first). ``relay_holds``
    ((donor, receiver) -> :class:`RelayHold`): pairs the SM answered with nothing
    done on the fleet view that is still current - not planned again.

    ``view_pending`` (model -> direction of its last routable-changing SM call that this
    plan's fleet view does not show yet): the model takes no new action, in any role."""
    active_probe_models = active_probe_models or set()
    # Review 3 P2-3: models whose only in-flight work is a SafeScale commit waiting
    # out a retry backoff. A CRITICAL receiver among them is still planned: the
    # queue preempts that retry when the rescue action is submitted.
    preemptible_models = preemptible_models or set()
    # Models whose last SafeScale probe rolled back and whose signal does not show new
    # evidence yet (timer cleanup 2026-10-02, replaces the A13 60 s backoff): model ->
    # hold reason (a plain set = reason "evidence"). No new probe of the model as donor
    # (receiver-less or for a receiver, F1-B). Scope dependent (2026-10-26): with
    # ``release_without_safescale`` a donor for a receiver is an immediate relay, not a
    # probe, and is not held; only the receiver-less HIGH probe is.
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
    # L4 rev 2: HIGH / HEALTHY donors that gave a TP receiver's block this plan - they
    # give nothing else this plan (a non-IDLE donor gives at most one replica per plan).
    block_donor_models: set[str] = set()
    # Sleeping-capacity deadlock fix: GPU-slot occupancy so a receiver's sleeping binding
    # only counts as wakeable capacity when its slot has no awake binding (of any model).
    # S5: GPUs the SM reports not wakeable are no wake / create capacity this tick (the
    # S3 per-GPU wake cooldown was removed on 2026-10-02: a refusal is an event only).
    occupancy = _SlotOccupancy(cluster_view) if cluster_view is not None else None
    # Correctness guard (TSS only, 2026-10-29): a model whose last routable-changing SM
    # call the fleet view does not show yet takes no new action. (The F4 window cooldown
    # and the P2-6 floor-violation hold were removed: signal lag is accepted, the donor's
    # SM ``floor_headroom`` decides and the SM clamps model-level shrinks.)
    view_hold = _ViewPending(view_pending or {}, events)

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

    # I4 donor gate (2026-10-25): a model whose serving pods' samples are incomplete in
    # this window, or whose context is a held one (tokens missing), gives nothing - no
    # donor, shrink or release from it. Receivers are not affected.
    donor_held = [
        item.model_name
        for item in classifications
        if item.role != ModelRole.RECEIVER
        and model_contexts.get(item.model_name, {}).get("sample_quality_hold")
    ]
    if donor_held:
        events.extend(f"donor_suppressed_samples_incomplete:{model}" for model in donor_held)
        classifications = [item for item in classifications if item.model_name not in donor_held]

    # L2 (design 20261028-single-loop): CRITICAL receivers by Z ascending (the most
    # starved first; 2026-10-29: TSS only, no saturation / exempt tiers). Never the
    # registry order.
    critical_receivers = sorted(
        (item for item in classifications if item.state == ModelState.CRITICAL), key=receiver_sort_key
    )
    low_receivers = sorted(
        (item for item in classifications if item.state == ModelState.LOW), key=receiver_sort_key
    )
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
    # F1-B (2026-10-07, design donor-evidence-20261007): a HIGH donor gives a receiver
    # capacity only through a SafeScale probe (hide, observe, commit or roll back) - the
    # middle-zone path, HIGH donors first (cost order). Only an IDLE donor is released at
    # once: an idle window is evidence that does not depend on the replica count, while
    # "Z > tau_high now" says nothing about Z with one replica less (n = 2: half the
    # capacity).
    probe_donors = [item for item in paper_donors if item.state == ModelState.HIGH] + middle_zone

    def probe_held(donor: ModelClassification) -> bool:
        # One SafeScale probe per donor model (the state machine is keyed by model): a
        # donor this plan already probes gives no other receiver anything - a second
        # receiver (CRITICAL ones included) waits until that probe resolves.
        if donor.model_name in delayed_down_models:
            return True
        # One rule for every donor probe (HIGH, middle zone, TP block): a model
        # whose last probe rolled back is probed again only on new evidence.
        return donor_rollback_held(donor.model_name)

    def donor_rollback_held(model: str) -> bool:
        # Scope of the rollback hold (2026-10-26, Tight_pool hazard (d)): it judged one
        # decision - "a probe that hides a replica of this model is safe on this
        # evidence" - and blocks only a retry of that decision. With every probe on
        # (scope ``all``) a donor for a receiver is itself such a probe: held. When a
        # donor for a receiver is an immediate relay (``release_without_safescale``:
        # scope ``receiverless_shrink``; scope ``off`` has no holds), it is a different
        # decision - a LOW / CRITICAL receiver in need - and the hold does not apply.
        # The receiver-less HIGH probe stays held in every scope (high_models loop).
        return not cfg.release_without_safescale and rollback_held(model)

    def rollback_held(model: str) -> bool:
        if model not in probe_backoff_models:
            return False
        _event_once(events, f"safescale_rollback_hold:{model}:{probe_backoff_models[model]}")
        return True

    def gave_this_plan(model: str) -> bool:
        # User rule (2026-10-28): a non-IDLE donor gives at most ONE replica per plan,
        # across all receivers and both sections - a relay, a block or a probe of it
        # already planned (its delta, its block) ends its giving for this plan.
        return deltas.get(model, 0) < 0 or model in block_donor_models

    def note_relay_past_hold(model: str, gained: int) -> None:
        # An immediate relay took a donor whose probe rollback hold would have vetoed it
        # before 2026-10-26 (hazard (d)) - logged so a rerun shows where that happened.
        if gained > 0 and cfg.release_without_safescale and model in probe_backoff_models:
            _event_once(events, f"safescale_rollback_hold_relay_ok:{model}")

    def block_donors(receiver: str) -> dict[str, BlockDonor]:
        """L4 rev 2: the donor models a TP receiver's block may take pods of, now:
        IDLE / HIGH / HEALTHY (never LOW / CRITICAL), not the receiver, not probing, not
        in flight, not view-pending, above its floor after this plan's takes; a HIGH /
        HEALTHY donor only while nothing of it is planned yet (one replica per plan)
        and not held by a probe of this plan or a rollback hold (scope ``all``)."""
        out: dict[str, BlockDonor] = {}
        for item in classifications:
            model = item.model_name
            tier = _BLOCK_TIERS.get(item.state)
            if tier is None or item.role == ModelRole.RECEIVER or model == receiver:
                continue
            if model in active_probe_models or model in inflight_models:
                continue
            # Review P2-1 for blocks (2026-10-29): a block of this donor the SM refused
            # with nothing done is not planned again on the same fleet view.
            hold = (relay_holds or {}).get((model, receiver))
            if hold is not None and hold.holds(relay_basis(cluster_view, model, receiver)):
                _event_once(events, f"relay_held:{model}:{receiver}:{hold.reason}")
                continue
            idle = item.state == ModelState.IDLE
            if not idle and (deltas.get(model, 0) != 0 or probe_held(item)):
                continue
            if view_hold.blocks(model):
                continue
            headroom = _donor_headroom(cfg, model, model_contexts, model_replicas) - abs(
                min(deltas.get(model, 0), 0)
            )
            if headroom <= 0:
                continue
            out[model] = BlockDonor(
                tier=tier,
                z=item.Z_m,
                limit=headroom if idle else 1,
                probed=not idle and not cfg.release_without_safescale,
            )
        return out

    def plan_block(
        recv: ModelClassification,
        *,
        source_loop: SourceLoop,
        prefix: str,
        create_reason: str,
    ) -> bool:
        """L4 rev 2: one buddy block (+1 replica) for a TP receiver; False = none."""
        model = recv.model_name
        donors = block_donors(model)
        choice = cheapest_block(model, occupancy, cluster_view, donors, cfg)
        if choice is None:
            return False
        tier = BLOCK_TIER_NAMES[choice.tier]
        if not choice.sleeps:
            if choice.binding is not None:
                occupancy.claim(choice.binding)
                _add_scale_action(
                    actions,
                    deltas,
                    model=model,
                    delta=1,
                    reason=f"{prefix}_sleeping_capacity",
                    source_loop=source_loop,
                    receiver=model,
                    pods=(choice.binding.serve_id,),
                    hint=True,
                )
            else:
                occupancy.claim_gpus(set(choice.block.keys), model)
                _add_scale_action(
                    actions, deltas, model=model, delta=1, reason=create_reason,
                    source_loop=source_loop, receiver=model,
                )
            return True
        probe = next(((donor, pod) for donor, pod in choice.sleeps if donors[donor].probed), None)
        non_idle = [donor for donor, _ in choice.sleeps if donors[donor].tier > 1]
        immediate = tuple(item for item in choice.sleeps if item != probe)
        actions.append(
            BlockSupplyAction(
                receiver=model,
                block=choice.block,
                receiver_pods=(choice.binding.serve_id,),
                sleeps=immediate,
                probe=probe,
                # An IDLE-only block is released by policy, not for lack of SafeScale.
                reason=f"{prefix}_block_{tier}"
                + (NO_SAFESCALE_REASON_SUFFIX if non_idle and probe is None else ""),
                source_loop=source_loop,
                # The hold key of the transfer group (its first donor, as the tick's
                # transfer_plan names it): a refused block is held like a relay.
                basis=relay_basis(cluster_view, (*immediate, *((probe,) if probe else ()))[0][0], model),
            )
        )
        for donor, _ in choice.sleeps:
            deltas[donor] = deltas.get(donor, 0) - 1
        for donor in non_idle:
            block_donor_models.add(donor)
            note_relay_past_hold(donor, 1)
        deltas[model] = deltas.get(model, 0) + 1
        occupancy.count_released(pod for _, pod in choice.sleeps)
        occupancy.claim(choice.binding)
        if probe is not None:
            # The tick commits this probe with exactly {receiver: 1}
            # (_safescale_pending_upscales); the plan records the same promise.
            delayed_down_models.add(probe[0])
            pending = probe_upscale_plans.setdefault(probe[0], {})
            pending[model] = pending.get(model, 0) + 1
        events.append(
            f"block_supply:{model}:{choice.block}:{tier}:" + ",".join(pod for _, pod in choice.sleeps)
        )
        return True

    def low_need(recv: ModelClassification) -> tuple[int, int] | None:
        """(replicas needed, of which wakeable from sleeping bindings), None = skip."""
        if recv.model_name in inflight_models:
            return None
        if deltas.get(recv.model_name, 0) > 0:
            # Already scaled up in this plan - one step per plan, never two.
            return None
        if view_hold.blocks(recv.model_name):
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
            if view_hold.blocks(recv.model_name):
                return None
            recv_pods = _effective_routable_replicas(recv.model_name, model_contexts, model_replicas)
            recv_assigned = _effective_assigned_replicas(recv.model_name, model_contexts, model_replicas)
            recv_max = _max_replicas(cfg, recv.model_name)
            # Cap on the awake count incl. hidden probe pods (v1 assigned = non-sleeping,
            # draining included; the SM counts the same), not on the routable count.
            recv_awake = _awake_replicas(recv.model_name, model_contexts, model_replicas)
            if recv_awake >= recv_max:
                return None
            # L3 (design 20261028-single-loop): one step per plan. The rescue target ledger
            # ("one deficit, one target") keeps a step the fleet view does not show yet
            # from being repeated: the target is computed from the replicas it was
            # planned on (``base``) and only what exceeds the replicas that target added
            # (``covered``) is planned - nothing, unless it landed nothing. State only
            # (2026-10-29): once the view shows it, the next step follows the signal.
            basis = rescue_bases.get(recv.model_name)
            base = basis.base if basis is not None else recv_pods
            covered = max(basis.covered, recv_pods) if basis is not None else recv_pods
            desired = base + _scale_step(base, cfg.scale_step_ratio)
            raw_need = min(desired - covered, recv_max - max(recv_awake, covered))
            if raw_need <= 0:
                if basis is not None:
                    events.append(
                        f"rescue_target_hold:{recv.model_name}:desired={desired}:covered={covered}"
                    )
                return None
            rescue_ctx[recv.model_name] = (desired, base, covered)
            return raw_need, min(raw_need, max(0, recv_assigned - recv_pods))

        critical_needs ={recv.model_name: critical_need(recv) for recv in critical_receivers}
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
                if tp_size > 1 and occupancy is not None:
                    # L4 rev 2 (design 20261028-single-loop): a TP receiver takes one
                    # buddy block per plan (free GPUs, or the pods of eligible donors on
                    # it). No block: a defrag (registry placement.defrag, off by
                    # default), else capacity_blocked.
                    if not plan_block(
                        recv,
                        source_loop="rescue",
                        prefix="critical",
                        create_reason="critical_empty_slot",
                    ):
                        if _plan_tp_defrag(
                            actions,
                            model=recv.model_name,
                            tp_size=tp_size,
                            cluster_view=cluster_view,
                            events=events,
                            source_loop="rescue",
                            occupancy=occupancy,
                            enabled=cfg.defrag_enabled,
                        ):
                            _add_scale_action(
                                actions,
                                deltas,
                                model=recv.model_name,
                                delta=1,
                                reason="critical_tp_defrag",
                                source_loop="rescue",
                                receiver=recv.model_name,
                            )
                        else:
                            _event_once(events, f"capacity_blocked:{recv.model_name}")
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
                for donor in paper_donors:
                    if still_needed <= 0:
                        break
                    if (
                        donor.model_name == recv.model_name
                        or donor.model_name in active_probe_models
                        or donor.model_name in inflight_models
                        or donor.state != ModelState.IDLE  # F1-B: HIGH donors probe below
                    ):
                        continue
                    if view_hold.blocks(donor.model_name):
                        continue
                    headroom = _donor_headroom(cfg, donor.model_name, model_contexts, model_replicas)
                    if headroom <= 0:
                        continue
                    donor_pods = _effective_routable_replicas(donor.model_name, model_contexts, model_replicas)
                    planned_take = abs(min(deltas.get(donor.model_name, 0), 0))
                    gained = _plan_relays(
                        actions,
                        deltas,
                        occupancy,
                        donor=donor.model_name,
                        receiver=recv.model_name,
                        need=still_needed,
                        # Q3: an IDLE donor gives its whole surplus (floor headroom bounds it).
                        donor_limit=min(max(1, donor_pods), headroom - planned_take),
                        reason="critical_donor_immediate",
                        source_loop="rescue",
                        events=events,
                        view=cluster_view,
                        relay_holds=relay_holds,
                    )
                    still_needed -= gained

                for middle in probe_donors:
                    if still_needed <= 0:
                        break
                    if (
                        middle.model_name == recv.model_name
                        or middle.model_name in active_probe_models
                        or middle.model_name in inflight_models
                        or gave_this_plan(middle.model_name)  # one replica per plan (non-IDLE)
                    ):
                        continue
                    if view_hold.blocks(middle.model_name):
                        continue
                    headroom = _donor_headroom(cfg, middle.model_name, model_contexts, model_replicas)
                    if headroom <= 0 or probe_held(middle):
                        continue
                    middle_pods = _effective_routable_replicas(middle.model_name, model_contexts, model_replicas)
                    planned_take = abs(min(deltas.get(middle.model_name, 0), 0))
                    # One step of the donor (TP receivers take blocks above, L4 rev 2).
                    middle_limit = min(_scale_step(middle_pods, cfg.scale_step_ratio), headroom - planned_take)
                    middle_reason = (
                        "critical_high_donor_safescale"
                        if middle.state == ModelState.HIGH
                        else "critical_middle_zone_safescale"
                    )
                    if cfg.release_without_safescale:
                        gained = _plan_relays(
                            actions,
                            deltas,
                            occupancy,
                            donor=middle.model_name,
                            receiver=recv.model_name,
                            need=still_needed,
                            donor_limit=min(1, middle_limit),  # one replica per plan
                            reason=f"{middle_reason}{NO_SAFESCALE_REASON_SUFFIX}",
                            source_loop="rescue",
                            events=events,
                            view=cluster_view,
                            relay_holds=relay_holds,
                        )
                        note_relay_past_hold(middle.model_name, gained)
                    else:
                        gained = _plan_middle_zone_probe(
                            actions,
                            deltas,
                            occupancy,
                            donor=middle.model_name,
                            receiver=recv.model_name,
                            need=still_needed,
                            donor_limit=middle_limit,
                            reason=middle_reason,
                            source_loop="rescue",
                            events=events,
                            delayed_down_models=delayed_down_models,
                            probe_upscale_plans=probe_upscale_plans,
                        )
                    still_needed -= gained
            finally:
                if recv.model_name in rescue_ctx:
                    _tag_rescue_actions(
                        actions, first_action, recv, rescue_ctx[recv.model_name], events, probe_upscale_plans
                    )

        for idle in idle_models:
            if idle.model_name in active_probe_models or idle.model_name in inflight_models:
                continue
            if deltas.get(idle.model_name, 0) != 0:
                continue
            if view_hold.blocks(idle.model_name):
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
            if high.model_name in active_probe_models or high.model_name in inflight_models:
                continue
            if deltas.get(high.model_name, 0) != 0:
                continue
            if view_hold.blocks(high.model_name):
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

    if not cfg.fairness_due:
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

        if occupancy is not None and _tp_size(cfg, recv.model_name) > 1:
            # L4 rev 2: one buddy block per plan (no relay, no defrag for LOW).
            if needed > 0 and not plan_block(
                recv,
                source_loop="fairness",
                prefix="low_fairness",
                create_reason="low_fairness_idle_capacity",
            ):
                _event_once(events, f"capacity_blocked:{recv.model_name}")
            continue

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
                or donor.state != ModelState.IDLE  # F1-B: HIGH donors probe below
            ):
                continue
            if view_hold.blocks(donor.model_name):
                continue
            headroom = _donor_headroom(cfg, donor.model_name, model_contexts, model_replicas)
            if headroom <= 0:
                continue
            donor_pods = _effective_routable_replicas(donor.model_name, model_contexts, model_replicas)
            planned_take = abs(min(deltas.get(donor.model_name, 0), 0))
            needed -= _plan_relays(
                actions,
                deltas,
                occupancy,
                donor=donor.model_name,
                receiver=recv.model_name,
                need=needed,
                # Q3: an IDLE donor gives its whole surplus.
                donor_limit=min(donor_pods, headroom - planned_take),
                reason="low_fairness_donor_immediate",
                source_loop="fairness",
                events=events,
                view=cluster_view,
                relay_holds=relay_holds,
            )

        for middle in probe_donors:
            if needed <= 0:
                break
            if (
                middle.model_name == recv.model_name
                or middle.model_name in active_probe_models
                or middle.model_name in inflight_models
                or gave_this_plan(middle.model_name)  # one replica per plan (non-IDLE)
            ):
                continue
            if view_hold.blocks(middle.model_name):
                continue
            headroom = _donor_headroom(cfg, middle.model_name, model_contexts, model_replicas)
            if headroom <= 0 or probe_held(middle):
                continue
            donor_pods = _effective_routable_replicas(middle.model_name, model_contexts, model_replicas)
            planned_take = abs(min(deltas.get(middle.model_name, 0), 0))
            middle_limit = min(_scale_step(donor_pods, cfg.scale_step_ratio), headroom - planned_take)
            middle_reason = (
                "low_fairness_high_donor_safescale"
                if middle.state == ModelState.HIGH
                else "low_fairness_middle_zone_safescale"
            )
            if cfg.release_without_safescale:
                gained = _plan_relays(
                    actions,
                    deltas,
                    occupancy,
                    donor=middle.model_name,
                    receiver=recv.model_name,
                    need=needed,
                    donor_limit=min(1, middle_limit),  # one replica per plan
                    reason=f"{middle_reason}{NO_SAFESCALE_REASON_SUFFIX}",
                    source_loop="fairness",
                    events=events,
                    view=cluster_view,
                    relay_holds=relay_holds,
                )
                needed -= gained
                note_relay_past_hold(middle.model_name, gained)
                continue
            needed -= _plan_middle_zone_probe(
                actions,
                deltas,
                occupancy,
                donor=middle.model_name,
                receiver=recv.model_name,
                need=needed,
                donor_limit=middle_limit,
                reason=middle_reason,
                source_loop="fairness",
                events=events,
                delayed_down_models=delayed_down_models,
                probe_upscale_plans=probe_upscale_plans,
            )

    return PlanResult(actions, delayed_down_models, probe_upscale_plans, events=events)


def _plan_relays(
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
    """Immediate relays as named transfers (2026-10-28, sm-plans section 10): one
    :class:`BlockSupplyAction` per pair :meth:`_SlotOccupancy.pairable` names (a
    receiver binding whose every GPU this donor holds awake, and those donor pods - the
    shared release choice), bounded by ``need`` / ``donor_limit``. No fleet view: no
    relay (nothing to name). No relay while the SM reported its routable view unreadable (``routable_error``
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
        return 0  # no fleet view, no pod to name (production decides nothing then)
    # 2026-10-28 (sm-plans section 10): one named transfer per pair - the donor pods
    # and the receiver binding are this plan's choice, the SM validates them.
    estimate = occupancy.pairable(donor, receiver, max_pairs=need, max_donors=donor_limit)
    if estimate.pairs <= 0:
        _event_once(events, f"donor_no_slot_match:{donor}:{receiver}")
        return 0
    for receiver_binding, donor_pods in estimate.named:
        actions.append(
            BlockSupplyAction(
                receiver=receiver,
                block=slot_block(receiver_binding.slot),
                receiver_pods=(receiver_binding.serve_id,),
                sleeps=tuple((donor, pod) for pod in donor_pods),
                reason=reason,
                source_loop=source_loop,
                basis=basis,
            )
        )
    deltas[donor] = deltas.get(donor, 0) - estimate.donors
    deltas[receiver] = deltas.get(receiver, 0) + estimate.pairs
    return estimate.pairs


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


class _ViewPending:
    """The planner's one per-model hold (TSS only, 2026-10-29): ``view_pending`` - the
    fleet view predates the model's last routable-changing SM call, so the plan cannot
    see that call's effect - no new action of the model in any role until a newer view
    exists (event ``view_pending_hold:<model>:<direction of that call>``). A correctness
    guard on state, not a timer. (The review F4 window cooldown and the P2-6
    floor-violation hold were removed.)"""

    def __init__(self, view_pending: Mapping[str, str], events: list[str]) -> None:
        self._view_pending = dict(view_pending)
        self._events = events

    def blocks(self, model: str) -> bool:
        direction = self._view_pending.get(model)
        if direction is None:
            return False
        _event_once(self._events, f"view_pending_hold:{model}:{direction}")
        return True


class _SlotOccupancy:
    """GPU-slot occupancy for wake planning within one build_plan call.

    Under multi-model-per-GPU residency a receiver's sleeping binding is only real
    capacity when no other binding is awake on its GPU(s); the SM rejects any other wake
    with WakeConflict (the E1 dsllama-8b deadlock). Slots claimed by a planned wake (or
    counted for a planned relay) are not counted twice in one tick.

    Since 2026-10-28 (sm-plans section 10) it also NAMES each relay pair
    (:meth:`pairable`): the donor pods (the shared release choice) and the receiver
    binding on their GPU - the plan's choice, which the SM validates and executes.
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
        # a TP block's donor pods): never counted a second time in the same tick.
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

    def gpu_holder(self, gpu: tuple[str, int]) -> tuple[str, Binding | None]:
        """("free", None): no awake binding, not claimed, not blocked; ("pod", b): the
        awake, not hidden binding ``b`` holds it and no plan item released it yet;
        ("busy", None): anything else (claimed by this plan, blocked, a hidden pod, a
        pod this plan already releases)."""
        if gpu in self._claimed or gpu in self._blocked:
            return "busy", None
        binding = self._awake.get(gpu)
        if binding is None:
            return "free", None
        if binding.hidden or binding.serve_id in self._released:
            return "busy", None
        return "pod", binding

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
        """The relay pairs ``donor`` -> ``receiver`` this plan names (2026-10-28,
        sm-plans section 10: the controller names every pod): a sleeping, not hidden
        receiver binding on GPUs not claimed / blocked, EVERY one of which an awake,
        not hidden, not released binding of ``donor`` holds (a GPU without one is a
        plain wake, a GPU of another model no relay). A pair consumes all its
        occupants. At most ``max_pairs`` pairs and ``max_donors`` donor replicas;
        pairs with fewer donors first, then the donor unit the placement policy
        releases first (:func:`tre_common.gpu_placement.choose_release`, the release
        choice the SM uses), then the receiver's serve id. Each pair is claimed
        (receiver GPUs) and its donors released in the ledger."""
        if max_pairs <= 0 or max_donors <= 0:
            return PairEstimate(0, 0, ())
        found: list[tuple[Binding, tuple[Binding, ...]]] = []
        donors = 0
        while len(found) < max_pairs:
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
                if donors + len(occupants) > max_donors:
                    continue
                candidates.append(
                    (receiver_binding, tuple(sorted(occupants.values(), key=lambda b: natural_key(b.serve_id))))
                )
            if not candidates:
                break
            fewest = min(len(units) for _, units in candidates)
            receiver_binding, units = self._release_choice(
                [item for item in candidates if len(item[1]) == fewest]
            )
            self.claim(receiver_binding)
            self._released.update(unit.serve_id for unit in units)
            found.append((receiver_binding, units))
            donors += len(units)
        return PairEstimate(
            len(found),
            donors,
            tuple(unit.serve_id for _, units in found for unit in units),
            tuple((receiver_binding, tuple(unit.serve_id for unit in units)) for receiver_binding, units in found),
        )

    def _release_choice(
        self, candidates: list[tuple[Binding, tuple[Binding, ...]]]
    ) -> tuple[Binding, tuple[Binding, ...]]:
        """Among relay pairs, the donor unit the placement policy releases first
        (:func:`choose_release` on the block each unit frees: merge gain, node load,
        same-model spread, address), then the receiver with the lowest serve id. A
        unit that frees no aligned block ranks after the aligned ones."""
        units = list(dict.fromkeys(units for _, units in candidates))
        blocks = [_unit_block(unit, self._nodes) for unit in units]
        aligned = [(unit, block) for unit, block in zip(units, blocks) if block is not None]
        chosen = None
        if aligned:
            occupied = {gpu for gpu, binding in self._awake.items() if binding.serve_id not in self._released}
            choice = choose_release(
                [block for _, block in aligned],
                nodes=self._nodes,
                occupied=occupied | {key for _, block in aligned for key in block.keys},
                policy=self.policy(),
            )
            if choice is not None:
                chosen = aligned[choice.index][0]
        if chosen is None:
            chosen = min(units, key=lambda unit: natural_key(unit[0].serve_id))
        return min(
            (item for item in candidates if item[1] == chosen), key=lambda item: natural_key(item[0].serve_id)
        )


@dataclass(frozen=True)
class PairEstimate:
    """:meth:`_SlotOccupancy.pairable`: receiver replicas (``pairs``), donor replicas
    they consume (``donors``), those donor pods, and each pair as (receiver binding,
    its donor pods)."""

    pairs: int
    donors: int
    donor_pods: tuple[str, ...]
    named: tuple[tuple[Binding, tuple[str, ...]], ...] = ()


def _unit_block(unit: tuple[Binding, ...], nodes: Mapping[str, int]) -> GpuBlock | None:
    """The aligned GPU block a donor unit frees (one donor's slot, or the contiguous
    union of several donors' slots on one node), None when it frees no aligned block."""
    node = unit[0].slot.node
    if any(item.slot.node != node for item in unit):
        return None
    gpus = sorted({int(gpu) for item in unit for gpu in item.slot.gpu_ids})
    if gpus != list(range(gpus[0], gpus[0] + len(gpus))):
        return None
    slot = Slot(node, tuple(gpus))
    return slot_block(slot) if is_buddy_aligned(slot, nodes) else None


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


#: L4 rev 2: a block's donor tiers; the worst GPU of a block ranks it (SLO risk first).
BLOCK_TIER_NAMES = ("free", "idle", "high", "healthy")
_BLOCK_TIERS = {ModelState.IDLE: 1, ModelState.HIGH: 2, ModelState.HEALTHY: 3}


@dataclass(frozen=True)
class BlockDonor:
    """A donor model :func:`cheapest_block` may take pods of (the caller decides who is
    eligible: state, floor, probes, in flight, this plan's takes)."""

    #: Index into :data:`BLOCK_TIER_NAMES` (1 IDLE, 2 HIGH, 3 HEALTHY).
    tier: int
    z: float | None
    #: Pods it may give this plan: an IDLE donor its floor headroom, any other one.
    limit: int
    #: Its release is a SafeScale probe (a HIGH / HEALTHY donor under scope ``all``).
    probed: bool = False


@dataclass(frozen=True)
class BlockChoice:
    block: GpuBlock
    #: The receiver's sleeping binding on the block (a wake); None = a cold create.
    binding: Binding | None
    #: (donor model, serve id) of the awake pods on the block that sleep for it.
    sleeps: tuple[tuple[str, str], ...]
    #: Index into :data:`BLOCK_TIER_NAMES` of the block's worst GPU.
    tier: int
    key: tuple


def cheapest_block(
    receiver: str,
    ledger: _SlotOccupancy,
    cluster_view: ClusterView,
    donors: Mapping[str, BlockDonor],
    cfg: PlanConfig,
) -> BlockChoice | None:
    """The buddy block a TP receiver takes for +1 replica (design 20261028-single-loop
    L4 rev 2). Pure: the caller claims the block and releases the donor pods.

    Candidates: the slot of each sleeping, not hidden binding of the receiver (a wake)
    and, while the receiver has no unplanned sleeping binding (the SM creates only
    then), every aligned block whose GPUs are all free (a cold create). Each GPU of a
    candidate is free (no awake binding, not claimed by this plan, not blocked) or held
    by one awake, not hidden, not released pod of a ``donors`` model whose slot lies in
    the block (a pod spanning the block counts once). A donor gives at most its
    ``limit`` pods; at most one donor of a block is ``probed`` (one probe per model, no
    joint commit). Lexicographic choice: worst tier (free < IDLE < HIGH < HEALTHY),
    fewer donor pods, higher donor Z (the lower one first), a wake before a create,
    then the placement score (:func:`tre_common.gpu_placement.choose_placement`)."""
    tp_size = _tp_size(cfg, receiver)
    nodes = node_gpu_counts(cluster_view.topology)
    occupied = ledger.occupied()
    policy = ledger.policy()
    model_occupied = ledger.model_gpus(receiver)
    candidates: list[tuple[GpuBlock, Binding | None]] = [
        (slot_block(binding.slot), binding)
        for binding in ledger.sleeping(receiver)
        if len(binding.slot.gpu_ids) == tp_size and is_buddy_aligned(binding.slot, nodes)
    ]
    if not ledger.has_unplanned_sleeping(receiver):
        candidates += [(slot_block(slot), None) for slot in gpu_slot_candidates(cluster_view.topology, tp_size)]
    best: BlockChoice | None = None
    for block, binding in candidates:
        keys = set(block.keys)
        pods: dict[str, Binding] = {}
        usable = True
        for gpu in block.keys:
            state, pod = ledger.gpu_holder(gpu)
            if state == "free":
                continue
            if state != "pod" or binding is None or pod.model not in donors or pod.model == receiver:
                usable = False
                break
            if not {(pod.slot.node, other) for other in pod.slot.gpu_ids} <= keys:
                usable = False  # the donor pod is larger than the block
                break
            pods[pod.serve_id] = pod
        if not usable:
            continue
        per_model: dict[str, int] = {}
        for pod in pods.values():
            per_model[pod.model] = per_model.get(pod.model, 0) + 1
        if any(count > donors[model].limit for model, count in per_model.items()):
            continue
        if sum(1 for model in per_model if donors[model].probed) > 1:
            continue
        tier = max((donors[model].tier for model in per_model), default=0)
        z_values = sorted(_block_z(donors[pod.model].z) for pod in pods.values())
        z_key = (-z_values[0], -z_values[-1]) if z_values else (0.0, 0.0)
        placement = choose_placement(
            [block],
            nodes=nodes,
            occupied=occupied - keys,
            tp_size=tp_size,
            policy=policy,
            model_occupied=model_occupied,
        )
        if placement is None:
            continue
        key = (
            tier,
            len(pods),
            *z_key,
            0 if binding is not None else 1,
            placement.score[:-1],
            natural_key(binding.serve_id) if binding is not None else (),
        )
        if best is None or key < best.key:
            ordered = sorted(pods.values(), key=lambda pod: natural_key(pod.serve_id))
            best = BlockChoice(
                block=block,
                binding=binding,
                sleeps=tuple((pod.model, pod.serve_id) for pod in ordered),
                tier=tier,
                key=key,
            )
    return best


def _block_z(z: float | None) -> float:
    """A donor's Z for the block choice (higher = more surplus); an IDLE donor without
    a Z (no traffic) has the most."""
    return math.inf if z is None else float(z)


def probe_reserved_gpus(
    cluster_view: ClusterView, probes, tp_sizes: Mapping[str, int]
) -> frozenset:
    """GPUs unresolved SafeScale probes (probing or committing) count on for their
    receivers (design donor-evidence-20261007, item 3): for every receiver a probe
    promises replicas to (``pending_upscales``), the aligned ``tp_size`` slot around
    each GPU of the probe pods. While a donor pod is hidden its own GPU is occupied
    anyway; what this adds is the rest of a TP receiver's block its wake needs at commit
    (:class:`BlockSupplyAction` probes), and, between the donor's sleep and the
    receiver's wake of a commit, the freed GPU itself. The tick adds them to the view's
    ``blocked_gpus``, so no other wake / create / relay of the plan takes them; they are
    free again once the probe is resolved (state, not time)."""
    bindings = {binding.serve_id: binding for binding in cluster_view.bindings}
    reserved: set[tuple[str, int]] = set()
    for probe in probes:
        receivers = [
            model for model, count in (getattr(probe, "pending_upscales", None) or {}).items() if int(count) > 0
        ]
        if not receivers:
            continue
        probe_gpus = {
            (binding.slot.node, gpu)
            for pod in getattr(probe, "pods", ()) or ()
            if (binding := bindings.get(pod)) is not None
            for gpu in binding.slot.gpu_ids
        }
        for receiver in receivers:
            reserved |= _slots_around(cluster_view.topology, probe_gpus, int(tp_sizes.get(receiver, 1) or 1))
    return frozenset(reserved)


def _slots_around(topology: ClusterTopology, gpus: set, tp_size: int) -> set[tuple[str, int]]:
    """The GPUs of every aligned ``tp_size`` slot that contains one of ``gpus``."""
    out: set[tuple[str, int]] = set()
    for slot in gpu_slot_candidates(topology, tp_size):
        block = {(slot.node, gpu) for gpu in slot.gpu_ids}
        if block & gpus:
            out |= block
    return out


def _plan_tp_defrag(
    actions: list[Action],
    *,
    model: str,
    tp_size: int,
    cluster_view: ClusterView,
    events: list[str],
    source_loop: SourceLoop,
    occupancy: _SlotOccupancy,
    enabled: bool,
) -> bool:
    """The fallback of a CRITICAL TP receiver without a block (L4 rev 2): one defrag
    plan (registry ``placement.defrag``, off by default; a cluster-wide action)."""
    if occupancy.has_unplanned_sleeping(model):
        # P1-a: the SM wakes existing sleeping bindings before any create/defrag target
        # is used; with a blocked one the +1 would WakeConflict every tick.
        return False
    if not enabled:
        # Registry placement.defrag.enabled is false (default, v1 parity): no
        # automatic migration is planned (design 20260928-placement-node-balance).
        events.append(f"defrag_disabled:{model}")
        return False
    allocator = SlotAllocator(
        cluster_view.topology, list(cluster_view.bindings), policy=cluster_view.placement
    )
    migrations = allocator.plan_defrag(tp_size)
    if not migrations:
        return False
    actions.append(
        DefragAction(migrations=tuple(migrations), reason="critical_tp_defrag", source_loop=source_loop)
    )
    return True


def _paper_state_incomplete_models(classifications: list[ModelClassification]) -> tuple[str, ...]:
    return tuple(
        item.model_name
        for item in classifications
        if item.state == ModelState.UNKNOWN
        or (
            item.Z_m is None
            and item.state != ModelState.IDLE
            and not getattr(item, "signal_idle", False)
        )
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


def _tag_rescue_actions(
    actions: list[Action],
    first: int,
    recv: ModelClassification,
    ctx: tuple[int, int, int],
    events: list[str],
    probe_upscale_plans: Mapping[str, Mapping[str, int]] | None = None,
) -> None:
    """Attach the C1 :class:`RescuePlan` to the receiver's rescue scale-ups planned
    from ``actions[first:]`` and log the decision (``rescue_target``)."""
    desired, base, covered = ctx
    model = recv.model_name

    def receiver_part(action: Action) -> int:
        # L4 rev 2: a block wakes its receiver in the same plan (or the commit of its
        # probe); review P2-3: a donor probe for this receiver promises its commit's
        # receiver replicas (``probe_upscale_plans``, one probe per donor).
        if isinstance(action, BlockSupplyAction):
            return 1 if action.receiver == model else 0
        if (
            isinstance(action, ScaleAction) and action.requires_safescale and action.delta < 0
            and action.receiver == model
        ):
            return int((probe_upscale_plans or {}).get(action.model, {}).get(model, 0))
        up = upscale_of(action)
        return up[1] if up is not None and up[0] == model else 0

    planned = sum(receiver_part(action) for action in actions[first:])
    plan = RescuePlan(target=covered + planned, desired=desired, base=base, covered=covered)
    for index in range(first, len(actions)):
        if receiver_part(actions[index]) > 0:
            actions[index] = replace(actions[index], rescue=plan)
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

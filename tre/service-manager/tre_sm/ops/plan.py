"""Selection of a power plan ``POST /v2/plans`` (2026-10-27).

A plan is one decision of a caller: sleep items (named pods or a model-level
``at_most``) and wake items (a grow-only ``at_least``, or named pods), executed
in ONE writer-lock hold in two waves (design
``docs/design/20261027-sm-plans.md``). This module is the selection step only:
a pure function over an in-memory view of the fleet, generalizing the transfer
pair selection (:mod:`tre_sm.ops.transfer`, reused for the relay ``donors`` of a
wake item). Everything that needs I/O (the account's wake blocker, the wake
gate / third-resident veto, the placement pick, the shrink order) is injected as
a callable, so the rules can be tested without a service.

Rules:

1. sleep items -> the plan's sleep set S. Named pods: awake ones join S,
   asleep ones are a no-op; a pod with a journaled wake refuses the item
   (``wake_in_progress``); the item's pods are checked against the model's
   replica floor ONCE and a violation refuses the whole item (named pods are
   never clamped). ``at_most``: hidden replicas first, then the shrink order,
   clamped at the floor (``clamped_by_floor``). Sleep items are cumulative:
   the shrink order of an item sees the earlier items' members asleep, so its
   merge gain completes the pairs they started (as serial ``/target`` calls);
2. the planning view P = the books with S counted asleep (and S's GPU leases
   ignored by the wake blocker); R = the books with S still awake;
3. wake items widest model first (by tp size, results keep the request
   ``index``); among the items of one tp size, round-robin: one replica per
   item per round, request order breaking ties within a round, until each
   item is filled or nothing is left for it (free / hinted picks; then the
   relay donors per item; then the deferred pair-breaking picks, round-robin
   again) - contested GPUs are shared, not given to the first item.
   ``need = at_least - awake`` (<= 0: a no-op, donors untouched), at most the
   model's cap. Candidates are sleeping bindings whose
   wake the account allows on R (free GPUs: **wave 1**) or only on P with at
   least one GPU held by S members (**wave 2**, ``depends_on`` = those
   members; a TP=2 binding over one free GPU and one plan sleep is wave 2 too:
   the free GPU is marked used at once, so no later item takes it, and its
   lease is taken when wave 2 is prepared - the whole plan holds the writer
   lock, no other writer runs between the waves). The caller's hints first,
   else the placement pick on P; ``veto`` is asked for the pick about to be
   taken (wave 1: the wake gate; wave 2: the third resident on every GPU), a
   vetoed pick is replaced by the next best one. Still short and ``donors``
   given: :func:`select_transfer_pairs` per donor on P; its donors join S, its
   receivers wave 2. Free GPUs are used before any donor, except a free pick
   that would break a reserved TP pair (``reserve_violation``): such picks are
   deferred until the donors are used, then taken if still short.
   Named wakes: each feasible as above, else refused;
4. every pick updates R and P (GPU use and cap across items).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Mapping, Sequence

from tre_common.gpu_placement import PlacementPolicy
from tre_sm.allocator.slots import Binding, natural_key
from tre_sm.ops.transfer import TransferRefusal, select_transfer_pairs
from tre_sm.state.replica_floor import FloorCheck, check_floor

#: Where a wake pick came from (``woke[].source`` of the response).
SOURCE_FREE = "free"
SOURCE_PLAN_SLEEP = "plan_sleep"
SOURCE_DONOR = "donor"
SOURCE_HINT = "hint"

#: Why a candidate was not usable (``skipped`` of a wake item).
SKIP_UNCOVERED = "uncovered_gpu"
SKIP_AVOIDED = "avoided_gpu"


@dataclass(frozen=True)
class Donor:
    """Relay allowance of a wake item: up to ``max`` replicas of ``model``."""

    model: str
    max: int
    sleep_path: str = "urgent"


@dataclass(frozen=True)
class SleepItem:
    model: str
    sleep_path: str
    pods: tuple[str, ...] | None = None
    at_most: int | None = None
    reason: str = ""


@dataclass(frozen=True)
class WakeItem:
    model: str
    at_least: int | None = None
    pods: tuple[str, ...] | None = None
    hints: tuple[str, ...] = ()
    avoid_gpus: tuple[str, ...] = ()
    donors: tuple[Donor, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class Refusal:
    """A refusal the selection makes itself (shape of a WakeConflict body)."""

    reason: str
    detail: str
    binding_id: str | None = None
    node: str | None = None
    gpu_ids: tuple[int, ...] = ()
    blocking_binding_id: str | None = None


@dataclass
class SleepChoice:
    index: int
    item: SleepItem
    #: Bindings to put to sleep.
    members: list[Binding] = field(default_factory=list)
    #: Named pods already asleep (desired "sleeping" is recorded for them).
    already_asleep: list[Binding] = field(default_factory=list)
    #: ``at_most`` replicas the floor kept awake.
    kept_awake: list[Binding] = field(default_factory=list)
    floor_check: FloorCheck | None = None
    clamped: bool = False
    refusal: object | None = None

    @property
    def noop(self) -> bool:
        return self.refusal is None and not self.members and not self.clamped


@dataclass(frozen=True)
class WakePick:
    binding: Binding
    wave: int
    #: Binding ids of the S members on its GPUs (wave 2): its wake starts only
    #: once every one of them is confirmed asleep.
    depends_on: tuple[str, ...] = ()
    source: str = SOURCE_FREE
    hinted: bool = False
    #: Relay donors whose GPUs it takes (source ``donor``).
    donor_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DonorSleep:
    """A donor a wake item's relay put into S."""

    binding: Binding
    model: str
    sleep_path: str


@dataclass
class WakeChoice:
    index: int
    item: WakeItem
    awake_before: int = 0
    picks: list[WakePick] = field(default_factory=list)
    donor_sleeps: list[DonorSleep] = field(default_factory=list)
    #: Named pods already awake.
    already_awake: list[Binding] = field(default_factory=list)
    unfilled: int = 0
    refusals: list[object] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    #: Vetoed picks another candidate replaced: (binding id, refusal).
    substituted: list[tuple[str, object]] = field(default_factory=list)

    @property
    def noop(self) -> bool:
        return not self.picks and self.unfilled == 0 and not self.refusals


@dataclass
class PlanSelection:
    sleeps: list[SleepChoice] = field(default_factory=list)
    wakes: list[WakeChoice] = field(default_factory=list)

    def sleep_set(self) -> list[tuple[Binding, str]]:
        """Every binding of S with its sleep path, sleep items first."""
        out = [(member, choice.item.sleep_path) for choice in self.sleeps for member in choice.members]
        out.extend((donor.binding, donor.sleep_path) for choice in self.wakes for donor in choice.donor_sleeps)
        return out


WakeBlocker = Callable[[Binding, list[Binding], frozenset[str]], object | None]
Veto = Callable[[Binding, int, frozenset[str]], object | None]
WakePickFn = Callable[[list[Binding], list[Binding]], Binding]
ShrinkOrder = Callable[[list[Binding], list[Binding], list[Binding]], list[Binding]]
ReserveViolation = Callable[[Binding, list[Binding]], bool]


def select_plan(
    bindings: Sequence[Binding],
    *,
    sleeps: Sequence[SleepItem] = (),
    wakes: Sequence[WakeItem] = (),
    topology,
    policy: PlacementPolicy | None = None,
    floors: Mapping[str, tuple[int | None, Iterable[str]]] | None = None,
    caps: Mapping[str, int] | None = None,
    busy: Iterable[str] = (),
    wake_journaled: Iterable[str] = (),
    sleepable: Iterable[str] | None = None,
    wake_blocker: WakeBlocker,
    wake_pick: WakePickFn | None = None,
    shrink_order: ShrinkOrder | None = None,
    veto: Veto | None = None,
    reserve_violation: ReserveViolation | None = None,
) -> PlanSelection:
    """Select the sleep set and the wake picks of one plan (see the module doc).

    ``bindings``: the store's view (every model). ``floors``: model -> (replica
    floor or None = not enforced, its routable binding ids), for every sleep and
    donor model. ``caps``: model -> ``max_awake_replicas``. ``busy``: binding ids
    with something half-done (never a relay donor or receiver).
    ``wake_journaled``: binding ids with a journaled wake (a named sleep of one
    is refused). ``sleepable``: serve ids that can be put to sleep (a Pod with
    an IP; None = all). ``wake_blocker(binding, planning, ignored_ids)``: why
    the account refuses the wake on ``planning`` with the GPU leases of
    ``ignored_ids`` ignored (None = allowed). ``wake_pick(feasible, planning)``:
    the placement pick. ``shrink_order(serving, bindings, already_released)``:
    the release order of an ``at_most`` shrink. ``veto(binding, wave,
    sleeping_ids)``: why the pick about to be taken must not be (None = take it);
    ``sleeping_ids`` = the S members it may share GPUs with.
    ``reserve_violation(binding, planning)``: waking ``binding`` on ``planning``
    breaks a reserved TP pair (placement score violation > 0; None = never)."""
    floors = dict(floors or {})
    caps = dict(caps or {})
    busy_ids = frozenset(str(item) for item in busy)
    journaled = frozenset(str(item) for item in wake_journaled)
    can_sleep = None if sleepable is None else frozenset(str(item) for item in sleepable)
    pick = wake_pick or (lambda feasible, _planning: min(feasible, key=lambda b: natural_key(b.serve_id)))
    books = {binding.serve_id: binding for binding in bindings}
    selection = PlanSelection()

    # The view the shrink order sees: earlier sleep items' members asleep.
    asleep_view = dict(books)
    for index, item in enumerate(sleeps):
        choice = _select_sleep(
            index, item, books, asleep_view, floors.get(item.model), journaled, can_sleep, shrink_order
        )
        selection.sleeps.append(choice)
        for member in choice.members:
            asleep_view[member.serve_id] = replace(member, awake=False, hidden=False)
    s_ids = {member.binding_id for choice in selection.sleeps for member in choice.members}
    # R: the books plus every wave-1 wake picked so far; P: the books with S
    # asleep plus every pick. A wave-2 pick is not written into R (its GPUs still
    # carry the S members it waits for: two awake on one GPU would make every later
    # account check on R fail); its GPUs are ``reserved`` instead, so no later
    # wave-1 pick takes them (R1: the free half of a TP=2 receiver too).
    real = dict(books)
    reserved: set[tuple[str, int]] = set()
    planning = {
        serve_id: replace(binding, awake=False, hidden=False) if binding.binding_id in s_ids else binding
        for serve_id, binding in books.items()
    }
    widths: dict[str, int] = {}
    for binding in books.values():
        widths[binding.model] = max(widths.get(binding.model, 1), len(binding.slot.gpu_ids))

    def take(choice: WakeChoice, binding: Binding, wake_pick_: WakePick) -> None:
        choice.picks.append(wake_pick_)
        if wake_pick_.wave == 1:
            real[binding.serve_id] = replace(binding, awake=True, hidden=False)
        else:
            reserved.update((binding.slot.node, int(gpu)) for gpu in binding.slot.gpu_ids)
        planning[binding.serve_id] = replace(binding, awake=True, hidden=False)

    def classify(choice: WakeChoice, binding: Binding):
        """(wave, depends_on) or (None, the refusal)."""
        blocked = wake_blocker(binding, list(real.values()), frozenset())
        if blocked is None:
            taken = sorted(int(g) for g in binding.slot.gpu_ids if (binding.slot.node, int(g)) in reserved)
            if not taken:
                return 1, ()
            blocked = Refusal(
                reason="slot_occupied",
                detail=f"{binding.serve_id}: gpu(s) {taken} are taken by a wave-2 wake of this plan",
                binding_id=binding.binding_id, node=binding.slot.node, gpu_ids=tuple(binding.slot.gpu_ids),
            )
        if wake_blocker(binding, list(planning.values()), frozenset(s_ids)) is not None:
            return None, blocked
        deps = _sleepers_on(binding, books, s_ids)
        if not deps:
            choice.skipped[SKIP_UNCOVERED] = choice.skipped.get(SKIP_UNCOVERED, 0) + 1
            return None, blocked
        # A GPU of the binding not held by S is free on P (the blocker
        # passed): reserved now (not written into R), leased at the wave-2 prepare.
        return 2, tuple(dep.binding_id for dep in deps)

    def step(fill: _Fill, keep_reserve: bool) -> bool:
        """Take ONE wake for ``fill``'s item (True), or report that nothing is
        feasible for it now (False). ``keep_reserve``: leave out free (wave-1,
        unhinted) picks that break a reserved TP pair (``fill.deferred``)."""
        choice, item = fill.choice, fill.choice.item
        while fill.need > 0 and fill.budget > 0:
            feasible: dict[str, tuple[int, tuple[str, ...]]] = {}
            fill.blockers = {}
            for binding in sorted(planning.values(), key=lambda b: natural_key(b.serve_id)):
                if binding.model != item.model or binding.awake or binding.serve_id in fill.tried:
                    continue
                if any(f"{binding.slot.node}/{gpu}" in fill.avoid for gpu in binding.slot.gpu_ids):
                    fill.avoided.add(binding.binding_id)
                    continue
                wave, extra = classify(choice, binding)
                if wave is None:
                    fill.blockers[binding.serve_id] = extra
                    continue
                if (
                    keep_reserve and wave == 1 and binding.serve_id not in fill.hints
                    and reserve_violation(binding, list(planning.values()))
                ):
                    fill.deferred = True
                    continue
                feasible[binding.serve_id] = (wave, extra)
            if not feasible:
                return False
            candidates = [planning[serve_id] for serve_id in feasible]
            preferred = [binding for binding in candidates if binding.serve_id in fill.hints]
            binding = pick(preferred or candidates, list(planning.values()))
            fill.tried.add(binding.serve_id)
            wave, deps = feasible[binding.serve_id]
            refused = veto(binding, wave, frozenset(deps)) if veto is not None else None
            if refused is not None:
                choice.refusals.append(refused)
                choice.substituted.append((binding.binding_id, refused))
                continue
            hinted = binding.serve_id in fill.hints
            source = SOURCE_HINT if hinted else (SOURCE_FREE if wave == 1 else SOURCE_PLAN_SLEEP)
            take(choice, binding, WakePick(binding, wave, deps, source, hinted))
            fill.need -= 1
            fill.budget -= 1
            return True
        return False

    def round_robin(fills: list[_Fill], keep_reserve) -> None:
        """One replica per item per round, in request order, until every item is
        filled or nothing more is feasible for it (H2, 2026-10-28: contested GPUs
        are shared, not handed to the first item). Stateless and deterministic;
        another item's pick only ever takes GPUs, so an item that found nothing
        in a round finds nothing later in the same phase."""
        active = [fill for fill in fills if fill.need > 0 and fill.budget > 0]
        while active:
            active = [
                fill for fill in active
                if step(fill, keep_reserve(fill)) and fill.need > 0 and fill.budget > 0
            ]

    # Widest model first (R4): a TP=2 item takes a free pair before a TP=1 item can
    # split it. Among the items of ONE tp size: round-robin (H2), request order
    # breaking ties within a round.
    order = sorted(range(len(wakes)), key=lambda i: -widths.get(wakes[i].model, 1))
    groups: list[list[int]] = []
    for index in order:
        width = widths.get(wakes[index].model, 1)
        if groups and widths.get(wakes[groups[-1][0]].model, 1) == width:
            groups[-1].append(index)
        else:
            groups.append([index])
    for group in groups:
        fills: list[_Fill] = []
        for index in group:
            item = wakes[index]
            choice = WakeChoice(index=index, item=item)
            selection.wakes.append(choice)
            awake = [b for b in planning.values() if b.model == item.model and b.awake]
            choice.awake_before = sum(1 for b in books.values() if b.model == item.model and b.awake)
            budget = max(0, int(caps.get(item.model, 1 << 30)) - len(awake))
            if item.pods is not None:
                for name in item.pods:
                    binding = planning.get(name)
                    if binding is None:
                        choice.refusals.append(_unknown_pod(name, "wake"))
                        choice.unfilled += 1
                        continue
                    if binding.awake:
                        choice.already_awake.append(binding)
                        continue
                    if budget <= 0:
                        choice.refusals.append(_cap_refusal(binding, caps.get(item.model)))
                        choice.unfilled += 1
                        continue
                    wave, extra = classify(choice, binding)
                    if wave is None:
                        choice.refusals.append(extra)
                        choice.unfilled += 1
                        continue
                    refused = veto(binding, wave, frozenset(extra)) if veto is not None else None
                    if refused is not None:
                        choice.refusals.append(refused)
                        choice.unfilled += 1
                        continue
                    take(choice, binding, WakePick(binding, wave, extra, SOURCE_FREE if wave == 1 else SOURCE_PLAN_SLEEP))
                    budget -= 1
                continue
            need = int(item.at_least or 0) - len(awake)
            if need <= 0:
                continue
            fills.append(_Fill(
                choice=choice, need=need, budget=budget,
                hints=[str(hint) for hint in item.hints], avoid={str(gpu) for gpu in item.avoid_gpus},
            ))
        # 1. free / hinted picks; a relay keeps the TP pair reserve (its donors
        #    before a pair-breaking free pick, R2).
        round_robin(fills, lambda fill: bool(fill.choice.item.donors) and reserve_violation is not None)
        # 2. relay donors, per item as before.
        for fill in fills:
            choice, item = fill.choice, fill.choice.item
            for donor in item.donors:
                if fill.need <= 0 or fill.budget <= 0:
                    break
                floor, routable = floors.get(donor.model, (None, ()))
                pairs = select_transfer_pairs(
                    list(planning.values()),
                    donor_model=donor.model,
                    receiver_model=item.model,
                    count=donor.max,
                    topology=topology,
                    policy=policy,
                    busy=busy_ids | fill.avoided | {planning[s].binding_id for s in fill.tried},
                    donor_floor=floor,
                    donor_routable=set(routable) - s_ids,
                    receiver_budget=min(fill.need, fill.budget),
                    wake_blocker=lambda receiver, view, ignored: wake_blocker(
                        receiver, view, ignored | frozenset(s_ids)
                    ),
                    veto=lambda pair: (
                        veto(pair.receiver, 2, frozenset(s_ids) | frozenset(pair.donor_ids))
                        if veto is not None else None
                    ),
                    receiver_pick=pick,
                )
                for reason, amount in pairs.skipped.items():
                    choice.skipped[reason] = choice.skipped.get(reason, 0) + amount
                choice.refusals.extend(pairs.refusals)
                for pair in pairs.pairs:
                    for member in pair.donors:
                        choice.donor_sleeps.append(DonorSleep(member, donor.model, donor.sleep_path))
                        s_ids.add(member.binding_id)
                        planning[member.serve_id] = replace(member, awake=False, hidden=False)
                    take(choice, pair.receiver,
                         WakePick(pair.receiver, 2, pair.donor_ids, SOURCE_DONOR, donor_ids=pair.donor_ids))
                    fill.need -= 1
                    fill.budget -= 1
        # 3. the deferred pair-breaking free picks of relays still short.
        round_robin([fill for fill in fills if fill.deferred], lambda _fill: False)
        for fill in fills:
            choice, item = fill.choice, fill.choice.item
            if fill.avoided:
                choice.skipped[SKIP_AVOIDED] = len(fill.avoided)
            if fill.need > 0:
                choice.unfilled = fill.need
                if fill.budget <= 0:
                    choice.refusals.append(_cap_refusal(None, caps.get(item.model), model=item.model))
                # What is still blocking (reported, like a best-effort growth).
                choice.refusals.extend(fill.blockers.values())
                if not choice.refusals:
                    choice.refusals.append(
                        Refusal(reason="no_binding", detail=f"no sleeping binding of {item.model} left to wake")
                    )
    selection.wakes.sort(key=lambda choice: choice.index)
    return selection


@dataclass
class _Fill:
    """The growth state of one ``at_least`` wake item during the selection."""

    choice: WakeChoice
    need: int
    budget: int
    hints: list[str]
    avoid: set[str]
    tried: set[str] = field(default_factory=set)
    avoided: set[str] = field(default_factory=set)
    #: What blocked the candidates at the item's last attempt (reported when short).
    blockers: dict[str, object] = field(default_factory=dict)
    #: A free pick was left out to keep a reserved TP pair (R2).
    deferred: bool = False


def _select_sleep(
    index: int,
    item: SleepItem,
    books: dict[str, Binding],
    asleep_view: dict[str, Binding],
    floor_entry,
    journaled: frozenset[str],
    can_sleep: frozenset[str] | None,
    shrink_order: ShrinkOrder | None,
) -> SleepChoice:
    choice = SleepChoice(index=index, item=item)
    floor, routable = floor_entry if floor_entry is not None else (None, ())
    routable = set(routable)
    if item.pods is not None:
        members = []
        for name in item.pods:
            binding = books.get(name)
            if binding is None:
                choice.refusal = _unknown_pod(name, "sleep")
                return choice
            if not binding.awake:
                choice.already_asleep.append(binding)
                continue
            if binding.binding_id in journaled:
                choice.refusal = Refusal(
                    reason="wake_in_progress",
                    detail=f"power change of {name}: a wake of {binding.binding_id} is journaled "
                    "(left for the wake journal recovery)",
                    binding_id=binding.binding_id, node=binding.slot.node,
                    gpu_ids=tuple(binding.slot.gpu_ids), blocking_binding_id=binding.binding_id,
                )
                return choice
            if can_sleep is not None and name not in can_sleep:
                choice.refusal = Refusal(
                    reason="pod_not_found", detail=f"pod {name} not found (or no pod IP) for sleep",
                    binding_id=binding.binding_id, node=binding.slot.node, gpu_ids=tuple(binding.slot.gpu_ids),
                )
                return choice
            members.append(binding)
        if floor is not None and members:
            check = check_floor(item.model, floor, routable, {b.binding_id for b in members})
            choice.floor_check = check
            if not check.ok:
                choice.refusal = Refusal(reason="floor_violation", detail=f"replica floor of {item.model}")
                return choice
        choice.members = members
        return choice

    awake = sorted(
        (b for b in books.values() if b.model == item.model and b.awake),
        key=lambda b: natural_key(b.serve_id),
    )
    shrink = len(awake) - int(item.at_most or 0)
    if shrink <= 0:
        return choice
    # Hidden first (a SafeScale probe pod, review F2), then the release order.
    hidden = [b for b in awake if b.hidden]
    serving = [b for b in awake if not b.hidden]
    if shrink_order is not None:
        serving = shrink_order(serving, list(asleep_view.values()), hidden[:shrink])
    candidates = [b for b in hidden + serving if can_sleep is None or b.serve_id in can_sleep]
    members = candidates[:shrink]
    if floor is not None and members:
        check = check_floor(item.model, floor, routable, {b.binding_id for b in members})
        choice.floor_check = check
        if not check.ok:
            # Clamp: keep awake as many routable ones as the floor needs (hidden /
            # not-yet-serving ones still sleep).
            allowed = max(0, len(check.routable) - check.floor)
            kept, taken = [], 0
            for binding in members:
                if binding.binding_id in routable:
                    if taken >= allowed:
                        choice.kept_awake.append(binding)
                        continue
                    taken += 1
                kept.append(binding)
            members = kept
            choice.clamped = True
    choice.members = members
    return choice


def _sleepers_on(binding: Binding, books: dict[str, Binding], s_ids: set[str]) -> list[Binding]:
    """The S members sharing a GPU with ``binding``."""
    wanted = set(binding.slot.gpu_ids)
    return sorted(
        (
            other for other in books.values()
            if other.binding_id in s_ids
            and other.slot.node == binding.slot.node
            and wanted & set(other.slot.gpu_ids)
        ),
        key=lambda b: natural_key(b.serve_id),
    )


def _unknown_pod(name: str, what: str) -> Refusal:
    """A named pod the books do not know (replaced since the caller's view)."""
    return Refusal(reason="pod_not_found", detail=f"pod {name} is not a known binding ({what})")


def _cap_refusal(binding: Binding | None, cap, *, model: str | None = None) -> Refusal:
    name = binding.model if binding is not None else model
    return Refusal(
        reason="max_awake_replicas",
        detail=f"{name} is at max_awake_replicas ({cap})",
        binding_id=None if binding is None else binding.binding_id,
        node=None if binding is None else binding.slot.node,
        gpu_ids=() if binding is None else tuple(binding.slot.gpu_ids),
    )


def refusal_reason(refusal: object) -> str:
    """The reason code of any refusal the selection reports."""
    if isinstance(refusal, TransferRefusal):
        return refusal.reason
    return str(getattr(refusal, "reason", None) or refusal)

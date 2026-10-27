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
   clamped at the floor (``clamped_by_floor``);
2. the planning view P = the books with S counted asleep (and S's GPU leases
   ignored by the wake blocker); R = the books with S still awake;
3. wake items in request order, ``need = at_least - awake`` (<= 0: a no-op,
   donors untouched), at most the model's cap. Candidates are sleeping
   bindings whose wake the account allows on R (free GPUs: **wave 1**) or only
   on P with EVERY GPU held by S members (**wave 2**, ``depends_on`` = those
   members; a TP=2 binding over one free GPU and one plan sleep is skipped).
   The caller's hints first, else the placement pick on P; ``veto`` is asked
   for the pick about to be taken (wave 1: the wake gate; wave 2: the third
   resident), a vetoed pick is replaced by the next best one. Still short and
   ``donors`` given: :func:`select_transfer_pairs` per donor on P (free GPUs
   are used before any donor); its donors join S, its receivers wave 2.
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
    ``sleeping_ids`` = the S members it may share GPUs with."""
    floors = dict(floors or {})
    caps = dict(caps or {})
    busy_ids = frozenset(str(item) for item in busy)
    journaled = frozenset(str(item) for item in wake_journaled)
    can_sleep = None if sleepable is None else frozenset(str(item) for item in sleepable)
    pick = wake_pick or (lambda feasible, _planning: min(feasible, key=lambda b: natural_key(b.serve_id)))
    books = {binding.serve_id: binding for binding in bindings}
    selection = PlanSelection()

    for index, item in enumerate(sleeps):
        selection.sleeps.append(
            _select_sleep(index, item, books, floors.get(item.model), journaled, can_sleep, shrink_order)
        )
    s_ids = {member.binding_id for choice in selection.sleeps for member in choice.members}
    # R: the books plus every wake picked so far; P: R with S asleep.
    real = dict(books)
    planning = {
        serve_id: replace(binding, awake=False, hidden=False) if binding.binding_id in s_ids else binding
        for serve_id, binding in books.items()
    }
    for index, item in enumerate(wakes):
        choice = WakeChoice(index=index, item=item)
        selection.wakes.append(choice)
        awake = [b for b in planning.values() if b.model == item.model and b.awake]
        choice.awake_before = sum(1 for b in books.values() if b.model == item.model and b.awake)
        budget = max(0, int(caps.get(item.model, 1 << 30)) - len(awake))

        def take(binding: Binding, wake_pick_: WakePick) -> None:
            choice.picks.append(wake_pick_)
            real[binding.serve_id] = replace(binding, awake=True, hidden=False)
            planning[binding.serve_id] = replace(binding, awake=True, hidden=False)

        def classify(binding: Binding):
            """(wave, depends_on) or (None, the refusal)."""
            blocked = wake_blocker(binding, list(real.values()), frozenset())
            if blocked is None:
                return 1, ()
            if wake_blocker(binding, list(planning.values()), frozenset(s_ids)) is not None:
                return None, blocked
            deps = _sleepers_on(binding, books, s_ids)
            covered = {gpu for dep in deps for gpu in dep.slot.gpu_ids} & set(binding.slot.gpu_ids)
            if not deps or covered != set(binding.slot.gpu_ids):
                choice.skipped[SKIP_UNCOVERED] = choice.skipped.get(SKIP_UNCOVERED, 0) + 1
                return None, blocked
            return 2, tuple(dep.binding_id for dep in deps)

        if item.pods is not None:
            for name in item.pods:
                binding = planning[name]
                if binding.awake:
                    choice.already_awake.append(binding)
                    continue
                if budget <= 0:
                    choice.refusals.append(_cap_refusal(binding, caps.get(item.model)))
                    choice.unfilled += 1
                    continue
                wave, extra = classify(binding)
                if wave is None:
                    choice.refusals.append(extra)
                    choice.unfilled += 1
                    continue
                refused = veto(binding, wave, frozenset(extra)) if veto is not None else None
                if refused is not None:
                    choice.refusals.append(refused)
                    choice.unfilled += 1
                    continue
                take(binding, WakePick(binding, wave, extra, SOURCE_FREE if wave == 1 else SOURCE_PLAN_SLEEP))
                budget -= 1
            continue

        need = int(item.at_least or 0) - len(awake)
        if need <= 0:
            continue
        hints = [str(hint) for hint in item.hints]
        avoid = {str(gpu) for gpu in item.avoid_gpus}
        tried: set[str] = set()
        avoided: set[str] = set()
        blockers: dict[str, object] = {}
        while need > 0 and budget > 0:
            feasible: dict[str, tuple[int, tuple[str, ...]]] = {}
            blockers = {}
            for binding in sorted(planning.values(), key=lambda b: natural_key(b.serve_id)):
                if binding.model != item.model or binding.awake or binding.serve_id in tried:
                    continue
                if any(f"{binding.slot.node}/{gpu}" in avoid for gpu in binding.slot.gpu_ids):
                    avoided.add(binding.binding_id)
                    continue
                wave, extra = classify(binding)
                if wave is None:
                    blockers[binding.serve_id] = extra
                else:
                    feasible[binding.serve_id] = (wave, extra)
            if not feasible:
                break
            candidates = [planning[serve_id] for serve_id in feasible]
            preferred = [binding for binding in candidates if binding.serve_id in hints]
            binding = pick(preferred or candidates, list(planning.values()))
            tried.add(binding.serve_id)
            wave, deps = feasible[binding.serve_id]
            refused = veto(binding, wave, frozenset(deps)) if veto is not None else None
            if refused is not None:
                choice.refusals.append(refused)
                choice.substituted.append((binding.binding_id, refused))
                continue
            hinted = binding.serve_id in hints
            source = SOURCE_HINT if hinted else (SOURCE_FREE if wave == 1 else SOURCE_PLAN_SLEEP)
            take(binding, WakePick(binding, wave, deps, source, hinted))
            need -= 1
            budget -= 1
        if avoided:
            choice.skipped[SKIP_AVOIDED] = len(avoided)
        for donor in item.donors:
            if need <= 0 or budget <= 0:
                break
            floor, routable = floors.get(donor.model, (None, ()))
            pairs = select_transfer_pairs(
                list(planning.values()),
                donor_model=donor.model,
                receiver_model=item.model,
                count=donor.max,
                topology=topology,
                policy=policy,
                busy=busy_ids | avoided | {planning[s].binding_id for s in tried},
                donor_floor=floor,
                donor_routable=set(routable) - s_ids,
                receiver_budget=min(need, budget),
                wake_blocker=lambda receiver, view, ignored: wake_blocker(receiver, view, ignored | frozenset(s_ids)),
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
                take(pair.receiver, WakePick(pair.receiver, 2, pair.donor_ids, SOURCE_DONOR, donor_ids=pair.donor_ids))
                need -= 1
                budget -= 1
        if need > 0:
            choice.unfilled = need
            if budget <= 0:
                choice.refusals.append(_cap_refusal(None, caps.get(item.model), model=item.model))
            # What is still blocking (reported, like a best-effort growth).
            choice.refusals.extend(blockers.values())
            if not choice.refusals:
                choice.refusals.append(
                    Refusal(reason="no_binding", detail=f"no sleeping binding of {item.model} left to wake")
                )
    return selection


def _select_sleep(
    index: int,
    item: SleepItem,
    books: dict[str, Binding],
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
            binding = books[name]
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
        serving = shrink_order(serving, list(books.values()), hidden[:shrink])
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

"""Pair selection of the transfer primitive ``POST /v2/transfers`` (2026-10-02).

A transfer hands GPUs from a donor model to a receiver model: an awake donor
binding is put to sleep and the receiver binding that sleeps on the SAME GPUs is
woken right after, all in ONE writer-lock hold (design
``docs/design/20261002-sm-wholelock.md``). This module is the selection step
only: a pure function over an in-memory view of the fleet. Everything that
needs I/O (the account's wake blocker, the fault hook, the resident probe) is
injected as a callable, so the rules can be tested without a service.

Rules (one candidate pair = one receiver binding plus the donors on its GPUs):

1. receivers: the receiver model's bindings that sleep and are not hidden,
   with nothing left half-done on them (``busy``: a sleep or wake journal
   entry - a crash's or an unreadable engine's - or a transient GPU lease);
2. donors: every AWAKE binding on the receiver's GPUs. All of them must be of
   the donor model, not hidden and not busy, and together they must cover
   every GPU of the receiver (a GPU without a donor would not be fenced while
   the donor sleeps - see the design, "no extra pre-claim");
3. the account (``wake_blocker``) must allow the receiver's wake once the donors
   are counted asleep and their GPU leases ignored;
4. ``veto`` (fault hook, the probe of the third resident, which must be
   asleep) is asked only for the pair about to be taken; a vetoed pair is
   replaced by the next best one (``substituted``);
5. a TP=2 receiver may take two single-GPU donors: the pair weighs 2 and
   counts 2 against ``count`` (``count`` = donor replicas to hand over);
6. the donor model keeps at least ``donor_floor`` routable replicas;
7. order: the donor release preference of the controller planner
   (``_release_pick``: :func:`release_pick` here), then - when one donor frees
   several receivers - the receiver wake preference (``receiver_pick``, the
   service-manager's ``_wake_pick`` / ``choose_placement``);
8. up to ``count`` donor replicas are paired; what cannot be paired is
   ``unfilled``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Mapping, Sequence

from tre_common.gpu_placement import GpuBlock, PlacementPolicy, choose_release
from tre_sm.allocator.slots import (
    Binding,
    Slot,
    awake_gpus,
    is_buddy_aligned,
    natural_key,
    node_gpu_counts,
    slot_block,
)

#: Status of one pair (``pairs[].status`` of the response).
TRANSFER_PENDING = "pending"  # selected, not finished yet (never in a response)
TRANSFER_DONE = "done"
TRANSFER_DONOR_SLEEP_FAILED = "donor_sleep_failed"
TRANSFER_RECEIVER_WAKE_FAILED = "receiver_wake_failed"
#: Short aliases.
TRANSFER_DONOR_FAILED = TRANSFER_DONOR_SLEEP_FAILED
TRANSFER_RECEIVER_FAILED = TRANSFER_RECEIVER_WAKE_FAILED

#: Why a receiver binding was not a candidate (``TransferSelection.skipped``).
SKIP_NO_DONOR = "no_awake_occupant"
SKIP_FOREIGN_OCCUPANT = "occupant_not_donor_model"
SKIP_OCCUPANT_HIDDEN = "occupant_hidden"
SKIP_OCCUPANT_BUSY = "occupant_busy"
SKIP_UNCOVERED_GPU = "uncovered_gpu"
SKIP_OVER_COUNT = "over_count"
SKIP_FLOOR = "donor_floor"
SKIP_RECEIVER_BUSY = "receiver_busy"
SKIP_RECEIVER_CAP = "receiver_cap"


@dataclass(frozen=True)
class TransferPair:
    """One receiver binding and the donor binding(s) whose GPUs it takes over."""

    receiver: Binding
    donors: tuple[Binding, ...]

    @property
    def weight(self) -> int:
        """Donor replicas this pair consumes (counted against ``count``)."""
        return len(self.donors)

    @property
    def node(self) -> str:
        return self.receiver.slot.node

    @property
    def gpu_ids(self) -> tuple[int, ...]:
        return tuple(self.receiver.slot.gpu_ids)

    @property
    def donor_ids(self) -> tuple[str, ...]:
        return tuple(donor.binding_id for donor in self.donors)


@dataclass(frozen=True)
class TransferRefusal:
    """A candidate the account or a veto refused (shape of a WakeConflict)."""

    receiver_binding_id: str
    node: str
    gpu_ids: tuple[int, ...]
    reason: str
    detail: str
    blocking_binding_id: str | None = None
    #: The refusal object the callable returned (e.g. a WakeConflict).
    error: object | None = None


@dataclass
class TransferSelection:
    pairs: list[TransferPair] = field(default_factory=list)
    refusals: list[TransferRefusal] = field(default_factory=list)
    #: Donor replicas of ``count`` that could not be paired.
    unfilled: int = 0
    #: Receiver bindings that were no candidate, by reason (diagnostics).
    skipped: dict[str, int] = field(default_factory=dict)
    #: Vetoed pairs that another pair replaced: {"refused_binding_id", "reason"}.
    substituted: list[dict] = field(default_factory=list)


WakeBlocker = Callable[[Binding, list[Binding], frozenset[str]], object | None]
Veto = Callable[[TransferPair], object | None]
ReceiverPick = Callable[[list[Binding], list[Binding]], Binding]


def select_transfer_pairs(
    bindings: Sequence[Binding],
    *,
    donor_model: str,
    receiver_model: str,
    count: int,
    topology,
    policy: PlacementPolicy | None = None,
    busy: Iterable[str] = (),
    donor_floor: int | None = None,
    donor_routable: Iterable[str] = (),
    receiver_budget: int | None = None,
    wake_blocker: WakeBlocker | None = None,
    veto: Veto | None = None,
    receiver_pick: ReceiverPick | None = None,
) -> TransferSelection:
    """Choose up to ``count`` donor replicas' worth of (receiver, donors) pairs.

    ``bindings``: the store's view (every model). ``busy``: binding ids with
    something in progress (never a donor or a receiver). ``donor_floor``: the
    donor model's replica floor (None = not enforced) over ``donor_routable``
    (its routable binding ids). ``receiver_budget``: how many receivers may still
    be woken (scaling cap; None = no cap). ``wake_blocker(receiver, planning, ignored_ids)``
    returns why the account refuses the wake (None = allowed) with the donors
    asleep in ``planning`` and their leases in ``ignored_ids``. ``veto(pair)``
    returns why the pair must not be taken (None = take it).
    ``receiver_pick(receivers, planning)`` chooses among receivers one donor
    unit frees (default: lowest serve id)."""
    count = max(0, int(count))
    busy_ids = frozenset(str(item) for item in busy)
    routable = set(str(item) for item in donor_routable)
    planning: dict[str, Binding] = {binding.serve_id: binding for binding in bindings}
    selection = TransferSelection()
    remaining = count
    receivers_left = None if receiver_budget is None else max(0, int(receiver_budget))
    rejected: set[str] = set()
    taken_receivers: set[str] = set()
    skipped_once: set[tuple[str, str]] = set()

    def skip(binding: Binding, reason: str) -> None:
        key = (binding.binding_id, reason)
        if key in skipped_once:
            return
        skipped_once.add(key)
        selection.skipped[reason] = selection.skipped.get(reason, 0) + 1

    def refusal(receiver: Binding, refused: object) -> TransferRefusal:
        return TransferRefusal(
            receiver_binding_id=receiver.binding_id,
            node=receiver.slot.node,
            gpu_ids=tuple(receiver.slot.gpu_ids),
            reason=str(getattr(refused, "reason", None) or refused),
            detail=str(refused),
            blocking_binding_id=getattr(refused, "blocking_binding_id", None),
            error=refused,
        )

    while remaining > 0 and (receivers_left is None or receivers_left > 0):
        current = list(planning.values())
        candidates: list[TransferPair] = []
        for receiver in sorted(current, key=lambda item: natural_key(item.serve_id)):
            if receiver.model != receiver_model or receiver.awake or receiver.hidden:
                continue
            if receiver.binding_id in rejected or receiver.binding_id in taken_receivers:
                continue
            if receiver.binding_id in busy_ids:
                skip(receiver, SKIP_RECEIVER_BUSY)
                continue
            wanted = set(receiver.slot.gpu_ids)
            occupants = sorted(
                (
                    item
                    for item in current
                    if item.awake
                    and item.serve_id != receiver.serve_id
                    and item.slot.node == receiver.slot.node
                    and wanted & set(item.slot.gpu_ids)
                ),
                key=lambda item: natural_key(item.serve_id),
            )
            if not occupants:
                skip(receiver, SKIP_NO_DONOR)  # a plain wake, not a transfer
                continue
            reason = _occupant_problem(occupants, donor_model, busy_ids)
            if reason is not None:
                skip(receiver, reason)
                continue
            covered = {gpu for item in occupants for gpu in item.slot.gpu_ids}
            if not wanted <= covered:
                skip(receiver, SKIP_UNCOVERED_GPU)
                continue
            if len(occupants) > remaining:
                skip(receiver, SKIP_OVER_COUNT)
                continue
            if donor_floor is not None:
                consumed = sum(1 for item in occupants if item.binding_id in routable)
                if len(routable) - consumed < donor_floor:
                    skip(receiver, SKIP_FLOOR)
                    continue
            pair = TransferPair(receiver, tuple(occupants))
            if wake_blocker is not None:
                blocked = wake_blocker(
                    receiver, _with_asleep(current, pair.donors), frozenset(pair.donor_ids)
                )
                if blocked is not None:
                    rejected.add(receiver.binding_id)
                    selection.refusals.append(refusal(receiver, blocked))
                    continue
            candidates.append(pair)
        if not candidates:
            break
        units: list[tuple[Binding, ...]] = []
        for pair in candidates:
            if pair.donors not in units:
                units.append(pair.donors)
        unit = release_pick(units, current, topology, policy)
        group = [pair.receiver for pair in candidates if pair.donors == unit]
        if len(group) > 1 and receiver_pick is not None:
            chosen = receiver_pick(group, _with_asleep(current, unit))
        else:
            chosen = min(group, key=lambda item: natural_key(item.serve_id))
        pair = next(item for item in candidates if item.donors == unit and item.receiver == chosen)
        if veto is not None:
            refused = veto(pair)
            if refused is not None:
                rejected.add(chosen.binding_id)
                item = refusal(chosen, refused)
                selection.refusals.append(item)
                selection.substituted.append(
                    {"refused_binding_id": chosen.binding_id, "reason": item.reason, "detail": item.detail}
                )
                continue
        selection.pairs.append(pair)
        taken_receivers.add(chosen.binding_id)
        remaining -= pair.weight
        if receivers_left is not None:
            receivers_left -= 1
        routable -= set(pair.donor_ids)
        for donor in pair.donors:
            planning[donor.serve_id] = replace(donor, awake=False, hidden=False)
        planning[chosen.serve_id] = replace(chosen, awake=True, hidden=False)
    if remaining > 0 and receivers_left == 0:
        selection.skipped[SKIP_RECEIVER_CAP] = selection.skipped.get(SKIP_RECEIVER_CAP, 0) + 1
    selection.unfilled = remaining
    return selection


def _occupant_problem(occupants: list[Binding], donor_model: str, busy: frozenset[str]) -> str | None:
    for item in occupants:
        if item.model != donor_model:
            return SKIP_FOREIGN_OCCUPANT
        if item.hidden:
            return SKIP_OCCUPANT_HIDDEN
        if item.binding_id in busy:
            return SKIP_OCCUPANT_BUSY
    return None


def _with_asleep(bindings: list[Binding], donors: Iterable[Binding]) -> list[Binding]:
    ids = {donor.serve_id for donor in donors}
    return [
        replace(item, awake=False, hidden=False) if item.serve_id in ids else item
        for item in bindings
    ]


def release_pick(
    units: Sequence[tuple[Binding, ...]],
    bindings: Sequence[Binding],
    topology,
    policy: PlacementPolicy | None = None,
) -> tuple[Binding, ...]:
    """Among donor units (one donor, or the two single-GPU donors of a TP=2
    receiver), the one the placement policy releases first - the controller
    planner's ``_release_pick`` (merge gain, node load, same-model spread,
    address; never the serve id), applied to the block each unit frees.
    Unaligned units fall back to the lowest serve id."""
    if len(units) == 1:
        return units[0]
    nodes = node_gpu_counts(topology)
    blocks = []
    for unit in units:
        block = _unit_block(unit)
        blocks.append(block if block is not None and is_buddy_aligned(_slot(block), nodes) else None)
    aligned = [(unit, block) for unit, block in zip(units, blocks) if block is not None]
    if aligned:
        choice = choose_release(
            [block for _unit, block in aligned],
            nodes=nodes,
            occupied=awake_gpus(bindings) | {key for _unit, block in aligned for key in block.keys},
            policy=policy,
        )
        if choice is not None:
            return aligned[choice.index][0]
    return min(units, key=lambda unit: natural_key(unit[0].serve_id))


def _unit_block(unit: tuple[Binding, ...]):
    """The GPU block a donor unit frees: one donor's slot, or the union of
    several donors' slots on one node when it is a contiguous run."""
    if len(unit) == 1:
        return slot_block(unit[0].slot)
    node = unit[0].slot.node
    if any(item.slot.node != node for item in unit):
        return None
    gpus = sorted({gpu for item in unit for gpu in item.slot.gpu_ids})
    if gpus != list(range(gpus[0], gpus[0] + len(gpus))):
        return None
    return GpuBlock(node, tuple(gpus))


def _slot(block: GpuBlock) -> Slot:
    return Slot(block.node, tuple(block.gpu_ids))


def busy_binding_ids(
    *,
    wake_journal: Mapping[str, dict] | None = None,
    sleep_journal: Mapping[str, dict] | None = None,
    transient_lease_ids: Iterable[str] = (),
) -> frozenset[str]:
    """Binding ids with something left half-done: a wake or a sleep journal
    entry (the sleep journal is keyed by pod; its entries carry the binding
    id), an unexpired transient (starting) GPU lease."""
    ids = {str(item) for item in (wake_journal or {})}
    for entry in (sleep_journal or {}).values():
        binding_id = entry.get("binding_id") if isinstance(entry, Mapping) else None
        if binding_id:
            ids.add(str(binding_id))
    ids |= {str(item) for item in transient_lease_ids}
    return frozenset(ids)

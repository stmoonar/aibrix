"""Buddy-allocation GPU block placement.

Pure functions, no I/O, no redis/k8s imports: the controller (choosing which
sleeping binding to wake) and the service-manager (choosing a slot for a cold
create) share this single implementation so their anti-fragmentation behaviour
cannot drift apart.

Model
-----
Every node owns its own buddy tree over its GPUs: order-0 is one GPU, order-1 an
aligned pair ([0,1] / [2,3]), order-2 four aligned GPUs, and so on.  A replica
with ``tp_size == 2 ** k`` needs one *aligned* order-k block, and tensor
parallelism never spans nodes.

Placement is best-fit: among the fully free aligned blocks of the requested
order, pick the one whose enclosing *largest fully free* block is smallest, i.e.
the one that wastes the least future capacity ("how big a block do I have to
split to put this here").  Ties go to the lowest address.

Release is the mirror image: among the blocks a model currently holds, free the
one that merges upwards into the largest free block.  Ties go to the highest
address.  Both halves matter -- a first-fit release fragments every pair just as
effectively as a first-fit placement does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import re


__all__ = [
    "GpuBlock",
    "GpuKey",
    "PlacementChoice",
    "ReleaseChoice",
    "block_order",
    "choose_placement",
    "choose_release",
    "enumerate_blocks",
    "free_blocks",
    "plan_placements",
    "plan_releases",
]


GpuKey = tuple[str, int]

_NAT_SPLIT = re.compile(r"(\d+)")


def _natural_key(value: object) -> tuple[object, ...]:
    """``node9`` sorts before ``node10``; mirrors the service-manager helper."""
    return tuple(
        (1, int(part), "") if part.isdigit() else (0, 0, part)
        for part in _NAT_SPLIT.split(str(value))
    )


def block_order(size: int) -> int:
    """Buddy order of ``size`` GPUs; raises unless ``size`` is a power of two."""
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError(f"gpu block size must be an int, got {size!r}")
    if size < 1 or (size & (size - 1)) != 0:
        raise ValueError(f"gpu block size must be a power of two >= 1, got {size}")
    return size.bit_length() - 1


@dataclass(frozen=True)
class GpuBlock:
    """An aligned, contiguous run of GPUs on one node."""

    node: str
    gpu_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "gpu_ids", tuple(int(gpu) for gpu in self.gpu_ids))

    @property
    def size(self) -> int:
        return len(self.gpu_ids)

    @property
    def base(self) -> int:
        return self.gpu_ids[0]

    @property
    def order(self) -> int:
        return block_order(self.size)

    @property
    def keys(self) -> tuple[GpuKey, ...]:
        return tuple((self.node, gpu) for gpu in self.gpu_ids)

    @property
    def address_key(self) -> tuple[object, ...]:
        """Deterministic "low address first" ordering: node name, then base GPU."""
        return (_natural_key(self.node), self.base)

    def __str__(self) -> str:
        return f"{self.node}:{','.join(str(gpu) for gpu in self.gpu_ids)}"


@dataclass(frozen=True)
class PlacementChoice:
    """Where to put a replica, and why."""

    index: int
    """Index of the chosen entry in the caller's ``candidates`` sequence."""
    block: GpuBlock
    split_cost: int
    """Orders that have to be split to use this block; 0 == perfect fit."""
    enclosing_order: int
    """Order of the largest fully free aligned block containing ``block``."""
    considered: int
    """How many candidates were actually free and therefore comparable."""
    reason: str


@dataclass(frozen=True)
class ReleaseChoice:
    """Which replica to stop, and why."""

    index: int
    block: GpuBlock
    merge_order: int
    """Order of the free block this release merges into; >= ``block.order``."""
    merge_gain: int
    """``merge_order - block.order``; 0 == the release buys no larger block."""
    considered: int
    reason: str


def _normalise_nodes(nodes: Mapping[str, int]) -> dict[str, int]:
    if not nodes:
        raise ValueError("nodes must describe at least one node")
    out: dict[str, int] = {}
    for name, gpus in nodes.items():
        gpus = int(gpus)
        if gpus < 1:
            raise ValueError(f"node {name!r} must have at least one gpu, got {gpus}")
        out[str(name)] = gpus
    return out


def _normalise_occupied(
    occupied: Iterable[GpuKey] | None, nodes: Mapping[str, int]
) -> set[GpuKey]:
    out: set[GpuKey] = set()
    for entry in occupied or ():
        node, gpu = entry
        node = str(node)
        gpu = int(gpu)
        if node not in nodes:
            raise ValueError(f"occupied gpu on unknown node: {node}/{gpu}")
        if not 0 <= gpu < nodes[node]:
            raise ValueError(f"occupied gpu out of range: {node}/{gpu}")
        out.add((node, gpu))
    return out


def _validate_tp_size(tp_size: int, nodes: Mapping[str, int]) -> int:
    order = block_order(tp_size)
    widest = max(nodes.values())
    if tp_size > widest:
        raise ValueError(
            f"tp_size {tp_size} exceeds the widest node ({widest} gpus); "
            "tensor parallelism cannot span nodes"
        )
    return order


def _validate_block(block: GpuBlock, nodes: Mapping[str, int]) -> None:
    if not isinstance(block, GpuBlock):
        raise TypeError(f"expected GpuBlock, got {type(block).__name__}")
    if block.size == 0:
        raise ValueError("gpu block must contain at least one gpu")
    size = block.size
    block_order(size)  # power-of-two check
    if block.node not in nodes:
        raise ValueError(f"gpu block on unknown node: {block}")
    gpus = nodes[block.node]
    if block.gpu_ids != tuple(range(block.base, block.base + size)):
        raise ValueError(f"gpu block must be contiguous and ascending: {block}")
    if block.base % size != 0:
        raise ValueError(f"gpu block is not buddy-aligned: {block}")
    if block.base + size > gpus:
        raise ValueError(f"gpu block runs past the end of node {block.node}: {block}")


def _free_run_order(
    node: str, base: int, order: int, gpus: int, occupied: set[GpuKey]
) -> int:
    """Largest order j >= ``order`` whose aligned block containing ``base`` is free.

    The order-``order`` block itself is assumed free by the caller.
    """
    current = order
    while True:
        nxt = current + 1
        size = 1 << nxt
        start = (base // size) * size
        if start + size > gpus:
            return current
        if any((node, gpu) in occupied for gpu in range(start, start + size)):
            return current
        current = nxt


def enumerate_blocks(nodes: Mapping[str, int], tp_size: int) -> list[GpuBlock]:
    """Every aligned block of ``tp_size`` GPUs in the cluster, low address first."""
    node_map = _normalise_nodes(nodes)
    _validate_tp_size(tp_size, node_map)
    blocks = [
        GpuBlock(node, tuple(range(base, base + tp_size)))
        for node, gpus in node_map.items()
        for base in range(0, gpus - gpus % tp_size, tp_size)
    ]
    blocks.sort(key=lambda block: block.address_key)
    return blocks


def free_blocks(
    nodes: Mapping[str, int], occupied: Iterable[GpuKey], tp_size: int
) -> list[GpuBlock]:
    """``enumerate_blocks`` restricted to blocks with no occupied GPU."""
    node_map = _normalise_nodes(nodes)
    busy = _normalise_occupied(occupied, node_map)
    return [
        block
        for block in enumerate_blocks(node_map, tp_size)
        if not any(key in busy for key in block.keys)
    ]


def _choose_placement_indexed(
    candidates: Sequence[tuple[int, GpuBlock]],
    *,
    nodes: dict[str, int],
    occupied: set[GpuKey],
    tp_size: int,
    order: int,
) -> PlacementChoice | None:
    best_key: tuple[object, ...] | None = None
    best: PlacementChoice | None = None
    considered = 0
    for index, block in candidates:
        _validate_block(block, nodes)
        if block.size != tp_size:
            raise ValueError(
                f"candidate {block} has {block.size} gpus but tp_size is {tp_size}"
            )
        if any(key in occupied for key in block.keys):
            continue
        considered += 1
        enclosing = _free_run_order(
            block.node, block.base, order, nodes[block.node], occupied
        )
        cost = enclosing - order
        sort_key = (cost, block.address_key, index)
        if best_key is None or sort_key < best_key:
            best_key = sort_key
            best = PlacementChoice(
                index=index,
                block=block,
                split_cost=cost,
                enclosing_order=enclosing,
                considered=0,  # filled in once the scan has finished
                reason="",
            )
    if best is None:
        return None
    reason = (
        f"buddy best-fit tp={tp_size} -> {best.block} "
        f"split_cost={best.split_cost} "
        f"enclosing_order={best.enclosing_order} "
        f"free_candidates={considered}/{len(candidates)}"
    )
    return PlacementChoice(
        index=best.index,
        block=best.block,
        split_cost=best.split_cost,
        enclosing_order=best.enclosing_order,
        considered=considered,
        reason=reason,
    )


def choose_placement(
    candidates: Sequence[GpuBlock],
    *,
    nodes: Mapping[str, int],
    occupied: Iterable[GpuKey] = (),
    tp_size: int | None = None,
) -> PlacementChoice | None:
    """Pick the least wasteful free block among ``candidates``.

    ``candidates`` are the blocks the caller may use (for the controller: the
    slots of this model's sleeping bindings; for a cold create: every aligned
    block of the right size).  Candidates whose GPUs are occupied are skipped.
    ``nodes`` maps node name -> GPU count, ``occupied`` lists the ``(node, gpu)``
    pairs already taken by an awake replica.  ``tp_size`` defaults to the size of
    the first candidate.

    Returns ``None`` when no candidate is free.  Raises ``ValueError`` for a
    ``tp_size`` that is not a power of two, does not fit on any node, or does
    not match the candidates, and for candidate blocks that are unaligned,
    non-contiguous or off-node.
    """
    node_map = _normalise_nodes(nodes)
    candidates = list(candidates)
    if tp_size is None:
        if not candidates:
            raise ValueError("tp_size is required when candidates is empty")
        tp_size = candidates[0].size
    order = _validate_tp_size(tp_size, node_map)
    busy = _normalise_occupied(occupied, node_map)
    return _choose_placement_indexed(
        list(enumerate(candidates)),
        nodes=node_map,
        occupied=busy,
        tp_size=tp_size,
        order=order,
    )


def plan_placements(
    candidates: Sequence[GpuBlock],
    *,
    nodes: Mapping[str, int],
    occupied: Iterable[GpuKey] = (),
    tp_size: int | None = None,
    count: int = 1,
) -> list[PlacementChoice]:
    """Greedy sequential :func:`choose_placement` for ``count`` replicas.

    Each pick marks its GPUs occupied before the next one is scored, so a batch
    of wakes packs the way a tick-by-tick sequence of single wakes would.  The
    result is shorter than ``count`` when the candidates run out.  ``index``
    always refers to the original ``candidates`` sequence.
    """
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    node_map = _normalise_nodes(nodes)
    candidates = list(candidates)
    if tp_size is None:
        if not candidates:
            raise ValueError("tp_size is required when candidates is empty")
        tp_size = candidates[0].size
    order = _validate_tp_size(tp_size, node_map)
    busy = _normalise_occupied(occupied, node_map)
    remaining = list(enumerate(candidates))
    picks: list[PlacementChoice] = []
    for _ in range(count):
        choice = _choose_placement_indexed(
            remaining, nodes=node_map, occupied=busy, tp_size=tp_size, order=order
        )
        if choice is None:
            break
        picks.append(choice)
        busy |= set(choice.block.keys)
        remaining = [item for item in remaining if item[0] != choice.index]
    return picks


def _choose_release_indexed(
    candidates: Sequence[tuple[int, GpuBlock]],
    *,
    nodes: dict[str, int],
    occupied: set[GpuKey],
) -> ReleaseChoice | None:
    best: ReleaseChoice | None = None
    considered = 0
    for index, block in candidates:
        _validate_block(block, nodes)
        considered += 1
        after = occupied - set(block.keys)
        merge_order = _free_run_order(
            block.node, block.base, block.order, nodes[block.node], after
        )
        candidate = ReleaseChoice(
            index=index,
            block=block,
            merge_order=merge_order,
            merge_gain=merge_order - block.order,
            considered=0,
            reason="",
        )
        if best is None or (candidate.merge_order, candidate.block.address_key) > (
            best.merge_order,
            best.block.address_key,
        ):
            best = candidate
    if best is None:
        return None
    reason = (
        f"buddy release -> {best.block} merge_order={best.merge_order} "
        f"gain={best.merge_gain} candidates={considered}"
    )
    return ReleaseChoice(
        index=best.index,
        block=best.block,
        merge_order=best.merge_order,
        merge_gain=best.merge_gain,
        considered=considered,
        reason=reason,
    )


def choose_release(
    candidates: Sequence[GpuBlock],
    *,
    nodes: Mapping[str, int],
    occupied: Iterable[GpuKey] = (),
) -> ReleaseChoice | None:
    """Pick the block whose release merges into the largest free block.

    ``candidates`` are the blocks the shrinking model currently holds;
    ``occupied`` is the cluster-wide occupancy (those blocks included).  Ties on
    merge order go to the highest address, so a model shrinking off a whole node
    gives back a contiguous tail rather than every other GPU.
    """
    node_map = _normalise_nodes(nodes)
    busy = _normalise_occupied(occupied, node_map)
    return _choose_release_indexed(
        list(enumerate(candidates)), nodes=node_map, occupied=busy
    )


def plan_releases(
    candidates: Sequence[GpuBlock],
    *,
    nodes: Mapping[str, int],
    occupied: Iterable[GpuKey] = (),
    count: int = 1,
) -> list[ReleaseChoice]:
    """Greedy sequential :func:`choose_release` for ``count`` replicas.

    Each pick frees its GPUs before the next one is scored, so releasing two of
    four single-GPU replicas on one node gives back an aligned pair instead of
    two orphaned GPUs.  ``index`` refers to the original ``candidates``.
    """
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    node_map = _normalise_nodes(nodes)
    busy = _normalise_occupied(occupied, node_map)
    remaining = list(enumerate(list(candidates)))
    picks: list[ReleaseChoice] = []
    for _ in range(count):
        choice = _choose_release_indexed(remaining, nodes=node_map, occupied=busy)
        if choice is None:
            break
        picks.append(choice)
        busy -= set(choice.block.keys)
        remaining = [item for item in remaining if item[0] != choice.index]
    return picks

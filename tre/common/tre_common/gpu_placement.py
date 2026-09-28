"""Buddy-allocation GPU block placement with node balancing.

Pure functions, no I/O, no redis/k8s imports: the controller (choosing which
sleeping binding to wake, which replica to probe / shrink) and the
service-manager (choosing a slot for a cold create, which binding to wake or
sleep) share this single implementation so their placement behaviour cannot
drift apart (design note tre/docs/design/20260928-placement-node-balance.md).

Model
-----
Every node owns its own buddy tree over its GPUs: order-0 is one GPU, order-1 an
aligned pair ([0,1] / [2,3]), order-2 four aligned GPUs, and so on.  A replica
with ``tp_size == 2 ** k`` needs one *aligned* order-k block, and tensor
parallelism never spans nodes.

Policy
------
A :class:`PlacementPolicy` is built from the registry by
:func:`placement_policy_from_registry` and handed to every caller:

* ``max_order`` = log2 of the largest ``tp_size`` any registered model has.  No
  replica ever needs a block above it, so buddy climbing (split cost on
  placement, merge gain on release) is capped there: keeping a whole node free
  for a TP4 model that does not exist is not worth concentrating load.
* ``reserve_blocks`` (registry ``placement.reserve_tp_pairs``): keep this many
  fully free aligned max-order blocks (TP2 pairs today) for the widest model.
  Soft -- it never blocks a placement, it only ranks candidates.  A no-op when
  ``max_order == 0``.  With the models' awake counts known it is further bounded
  by the widest models' remaining headroom (:meth:`PlacementPolicy.for_awake`).

Placement ranks every free candidate block by (lowest first)::

    (violation, node_block_load_after, same_model_on_node,
     split_cost_capped, node_gpu_load_after, address, index)

* ``violation``: ``max(0, reserve - free max-order blocks left after placing)``.
* ``node_block_load_after``: fraction of the node's max-order-aligned groups that
  are not fully free after placing.  Load is measured at *reservation
  granularity* on purpose: filling the free half of a half-used pair adds no
  load, so single-GPU replicas pair up instead of scattering one per pair (a
  plain GPU-fraction key would spread TP1 replicas across pairs and destroy
  every pair a TP2 model could use).  With ``max_order == 0`` this is the GPU
  fraction.
* ``same_model_on_node``: GPUs the same model already holds awake on the node
  (spread a model's replicas across nodes).
* ``split_cost_capped``: order of the largest free block around the candidate,
  capped at ``max_order``, minus the candidate's order (0 == perfect fit).
* ``node_gpu_load_after``: awake GPU fraction on the node after placing.
* ``address``: natural (node, base GPU) order -- deterministic, stable.

Release is the mirror image among the blocks the shrinking model holds
(highest first)::

    (merge_gain_capped, node_block_load, same_model_on_node, node_gpu_load, address)

``policy=None`` means :data:`BEST_FIT`: plain buddy best-fit with no node
balancing, no cap below the node size and no reservation -- the reference
behaviour the brute-force tests check the buddy arithmetic against.  Production
callers always pass the registry policy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence
import re


__all__ = [
    "BEST_FIT",
    "DEFAULT_RESERVE_TP_PAIRS",
    "GpuBlock",
    "GpuKey",
    "PlacementChoice",
    "PlacementPolicy",
    "ReleaseChoice",
    "block_order",
    "choose_placement",
    "choose_release",
    "enumerate_blocks",
    "free_blocks",
    "placement_policy_from_registry",
    "plan_placements",
    "plan_releases",
]


GpuKey = tuple[str, int]

#: Default of registry ``placement.reserve_tp_pairs``.
DEFAULT_RESERVE_TP_PAIRS = 1

_NAT_SPLIT = re.compile(r"(\d+)")


def _natural_key(value: object) -> tuple[object, ...]:
    """``gpu-2`` sorts before ``gpu-10``; mirrors the service-manager helper."""
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
class PlacementPolicy:
    """How placement and release rank candidate blocks (see the module docstring).

    ``max_order``: None = no cap below the node size (reference best-fit).
    ``reserve_blocks``: free max-order blocks to keep (soft).
    ``balance_nodes``: False drops the node-balance terms (reference best-fit).
    ``reserve_caps``: ``(model, max awake replicas)`` of the models whose tp is the
    max-order size; :meth:`for_awake` bounds the reserve by their headroom.
    """

    max_order: int | None = None
    reserve_blocks: int = 0
    balance_nodes: bool = True
    reserve_caps: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.max_order is not None and int(self.max_order) < 0:
            raise ValueError(f"max_order must be >= 0, got {self.max_order}")
        if int(self.reserve_blocks) < 0:
            raise ValueError(f"reserve_blocks must be >= 0, got {self.reserve_blocks}")
        object.__setattr__(
            self,
            "reserve_caps",
            tuple((str(model), int(cap)) for model, cap in self.reserve_caps),
        )

    def effective_reserve(self, awake_counts: Mapping[str, int] | None = None) -> int:
        """The reserve after the ``max_order == 0`` no-op rule and, with
        ``awake_counts`` and ``reserve_caps`` known, ``min(reserve, sum(max_awake -
        awake))`` over the widest models: no point keeping more pairs free than the
        TP2 models could still wake into."""
        if self.max_order == 0:
            return 0
        reserve = int(self.reserve_blocks)
        if awake_counts is not None and self.reserve_caps:
            headroom = sum(
                max(0, cap - int(awake_counts.get(model, 0)))
                for model, cap in self.reserve_caps
            )
            reserve = min(reserve, headroom)
        return max(0, reserve)

    def for_awake(self, awake_counts: Mapping[str, int] | None) -> "PlacementPolicy":
        """This policy with the reserve bounded by the current awake counts."""
        if awake_counts is None:
            return self
        return replace(
            self, reserve_blocks=self.effective_reserve(awake_counts), reserve_caps=()
        )


#: Plain buddy best-fit: no node balance, no cap below the node size, no reserve.
BEST_FIT = PlacementPolicy(max_order=None, reserve_blocks=0, balance_nodes=False)


def _scale_cap(model: Any) -> int:
    for name in ("scale_max_replicas", "max_awake_replicas", "max_replicas"):
        value = getattr(model, name, None)
        if value is not None:
            return int(value)
    return 0


def placement_policy_from_registry(
    registry: Any, *, awake_counts: Mapping[str, int] | None = None
) -> PlacementPolicy:
    """The placement policy every caller uses, derived from the registry only.

    ``max_order`` = log2(max ``tp_size`` over the registered models);
    ``reserve_blocks`` = ``placement.reserve_tp_pairs``; ``reserve_caps`` = the
    scaling caps of the models whose tp is that maximum.  ``awake_counts`` (model ->
    awake replicas), when the caller knows them, bounds the reserve by those
    models' headroom (:meth:`PlacementPolicy.for_awake`).
    """
    models = list(registry.models())
    max_tp = max((int(model.tp_size) for model in models), default=1)
    placement = getattr(registry, "placement", None)
    config = placement() if callable(placement) else None
    reserve = int(getattr(config, "reserve_tp_pairs", DEFAULT_RESERVE_TP_PAIRS))
    policy = PlacementPolicy(
        max_order=block_order(max_tp),
        reserve_blocks=reserve,
        balance_nodes=True,
        reserve_caps=tuple(
            (str(model.name), _scale_cap(model))
            for model in models
            if int(model.tp_size) == max_tp
        ),
    )
    return policy.for_awake(awake_counts)


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
    """Orders that have to be split to use this block (capped at the policy's
    ``max_order``); 0 == perfect fit."""
    enclosing_order: int
    """Order of the largest fully free aligned block containing ``block`` (capped)."""
    considered: int
    """How many candidates were actually free and therefore comparable."""
    reason: str
    score: tuple = ()
    """The full sort key (lowest wins), for callers comparing single candidates."""


@dataclass(frozen=True)
class ReleaseChoice:
    """Which replica to stop, and why."""

    index: int
    block: GpuBlock
    merge_order: int
    """Order of the free block this release merges into (capped); >= ``block.order``."""
    merge_gain: int
    """``merge_order - block.order``; 0 == the release buys no larger block."""
    considered: int
    reason: str
    score: tuple = ()
    """The full sort key (highest wins)."""


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
    node: str,
    base: int,
    order: int,
    gpus: int,
    occupied: set[GpuKey],
    cap: int | None = None,
) -> int:
    """Largest order j >= ``order`` (and <= ``cap``) whose aligned block containing
    ``base`` is free.  The order-``order`` block itself is assumed free."""
    current = order
    while cap is None or current < cap:
        nxt = current + 1
        size = 1 << nxt
        start = (base // size) * size
        if start + size > gpus:
            return current
        if any((node, gpu) in occupied for gpu in range(start, start + size)):
            return current
        current = nxt
    return current


def _group_order(policy: PlacementPolicy, nodes: Mapping[str, int]) -> int:
    """Granularity of node load and reservation: the policy's ``max_order``, or
    (no cap) the largest power-of-two block the widest node holds."""
    if policy.max_order is not None:
        return int(policy.max_order)
    return max(gpus.bit_length() - 1 for gpus in nodes.values())


def _groups(gpus: int, size: int) -> list[range]:
    """The node's aligned groups of ``size`` GPUs; a ragged tail is its own group."""
    return [range(base, min(base + size, gpus)) for base in range(0, gpus, size)]


def _used_groups(node: str, gpus: int, size: int, occupied: set[GpuKey]) -> int:
    return sum(
        1
        for group in _groups(gpus, size)
        if any((node, gpu) in occupied for gpu in group)
    )


def _full_free_groups(node: str, gpus: int, size: int, occupied: set[GpuKey]) -> int:
    """Complete (not ragged) aligned groups with no occupied GPU."""
    return sum(
        1
        for group in _groups(gpus, size)
        if len(group) == size and not any((node, gpu) in occupied for gpu in group)
    )


def _node_gpu_load(node: str, gpus: int, occupied: set[GpuKey]) -> Fraction:
    return Fraction(sum(1 for gpu in range(gpus) if (node, gpu) in occupied), gpus)


def _node_block_load(node: str, gpus: int, size: int, occupied: set[GpuKey]) -> Fraction:
    return Fraction(_used_groups(node, gpus, size, occupied), len(_groups(gpus, size)))


def _node_count(keys: Iterable[GpuKey], node: str) -> int:
    return sum(1 for key_node, _ in keys if key_node == node)


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
    model_occupied: set[GpuKey],
    tp_size: int,
    order: int,
    policy: PlacementPolicy,
) -> PlacementChoice | None:
    group_order = _group_order(policy, nodes)
    group_size = 1 << group_order
    cap = max(group_order, order)
    reserve = policy.effective_reserve()
    free_by_node = {
        node: _full_free_groups(node, gpus, group_size, occupied)
        for node, gpus in nodes.items()
    }
    free_total = sum(free_by_node.values())
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
        node = block.node
        gpus = nodes[node]
        enclosing = _free_run_order(node, block.base, order, gpus, occupied, cap)
        cost = enclosing - order
        violation = 0
        if reserve:
            after = occupied | set(block.keys)
            free_after = free_total - free_by_node[node] + _full_free_groups(
                node, gpus, group_size, after
            )
            violation = max(0, reserve - free_after)
        if policy.balance_nodes:
            after = occupied | set(block.keys)
            sort_key = (
                violation,
                _node_block_load(node, gpus, group_size, after),
                _node_count(model_occupied, node),
                cost,
                _node_gpu_load(node, gpus, after),
                block.address_key,
                index,
            )
        else:
            sort_key = (violation, 0, 0, cost, 0, block.address_key, index)
        if best_key is None or sort_key < best_key:
            best_key = sort_key
            best = PlacementChoice(
                index=index,
                block=block,
                split_cost=cost,
                enclosing_order=enclosing,
                considered=0,  # filled in once the scan has finished
                reason="",
                score=sort_key,
            )
    if best is None:
        return None
    violation, node_load, same_model, _, gpu_load, _, _ = best.score
    reason = (
        f"placement tp={tp_size} -> {best.block} "
        f"violation={violation} node_block_load={node_load} "
        f"same_model_on_node={same_model} split_cost={best.split_cost} "
        f"node_gpu_load={gpu_load} enclosing_order={best.enclosing_order} "
        f"free_candidates={considered}/{len(candidates)}"
    )
    return replace(best, considered=considered, reason=reason)


def choose_placement(
    candidates: Sequence[GpuBlock],
    *,
    nodes: Mapping[str, int],
    occupied: Iterable[GpuKey] = (),
    tp_size: int | None = None,
    policy: PlacementPolicy | None = None,
    model_occupied: Iterable[GpuKey] = (),
) -> PlacementChoice | None:
    """Pick the best free block among ``candidates`` under ``policy``.

    ``candidates`` are the blocks the caller may use (for the controller: the
    slots of this model's sleeping bindings; for a cold create: every aligned
    block of the right size).  Candidates whose GPUs are occupied are skipped.
    ``nodes`` maps node name -> GPU count, ``occupied`` lists the ``(node, gpu)``
    pairs already taken by an awake replica, ``model_occupied`` the subset the
    placed model itself holds (for ``same_model_on_node``).  ``tp_size`` defaults
    to the size of the first candidate.  ``policy`` None = :data:`BEST_FIT`.

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
        model_occupied=_normalise_occupied(model_occupied, node_map),
        tp_size=tp_size,
        order=order,
        policy=policy or BEST_FIT,
    )


def plan_placements(
    candidates: Sequence[GpuBlock],
    *,
    nodes: Mapping[str, int],
    occupied: Iterable[GpuKey] = (),
    tp_size: int | None = None,
    count: int = 1,
    policy: PlacementPolicy | None = None,
    model_occupied: Iterable[GpuKey] = (),
) -> list[PlacementChoice]:
    """Greedy sequential :func:`choose_placement` for ``count`` replicas of one model.

    Each pick marks its GPUs occupied (and held by the model) before the next one
    is scored, so a batch of wakes lands the way a tick-by-tick sequence of single
    wakes would.  The result is shorter than ``count`` when the candidates run
    out.  ``index`` always refers to the original ``candidates`` sequence.
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
    model_busy = _normalise_occupied(model_occupied, node_map)
    policy = policy or BEST_FIT
    remaining = list(enumerate(candidates))
    picks: list[PlacementChoice] = []
    for _ in range(count):
        choice = _choose_placement_indexed(
            remaining,
            nodes=node_map,
            occupied=busy,
            model_occupied=model_busy,
            tp_size=tp_size,
            order=order,
            policy=policy,
        )
        if choice is None:
            break
        picks.append(choice)
        busy |= set(choice.block.keys)
        model_busy |= set(choice.block.keys)
        remaining = [item for item in remaining if item[0] != choice.index]
    return picks


def _choose_release_indexed(
    candidates: Sequence[tuple[int, GpuBlock]],
    *,
    nodes: dict[str, int],
    occupied: set[GpuKey],
    model_occupied: set[GpuKey],
    policy: PlacementPolicy,
) -> ReleaseChoice | None:
    group_order = _group_order(policy, nodes)
    group_size = 1 << group_order
    best: ReleaseChoice | None = None
    considered = 0
    for index, block in candidates:
        _validate_block(block, nodes)
        considered += 1
        node = block.node
        gpus = nodes[node]
        after = occupied - set(block.keys)
        merge_order = _free_run_order(
            node, block.base, block.order, gpus, after, max(group_order, block.order)
        )
        gain = merge_order - block.order
        if policy.balance_nodes:
            held = occupied | set(block.keys)
            sort_key = (
                gain,
                _node_block_load(node, gpus, group_size, held),
                _node_count(model_occupied, node),
                _node_gpu_load(node, gpus, held),
                block.address_key,
                -index,
            )
        else:
            # Reference best-fit ranks by the merged order itself (equal to the
            # gain ranking whenever the candidates share one size, as a model's do).
            sort_key = (merge_order, 0, 0, 0, block.address_key, -index)
        if best is None or sort_key > best.score:
            best = ReleaseChoice(
                index=index,
                block=block,
                merge_order=merge_order,
                merge_gain=gain,
                considered=0,
                reason="",
                score=sort_key,
            )
    if best is None:
        return None
    _, node_load, same_model, gpu_load, _, _ = best.score
    reason = (
        f"release -> {best.block} merge_order={best.merge_order} "
        f"gain={best.merge_gain} node_block_load={node_load} "
        f"same_model_on_node={same_model} node_gpu_load={gpu_load} "
        f"candidates={considered}"
    )
    return replace(best, considered=considered, reason=reason)


def choose_release(
    candidates: Sequence[GpuBlock],
    *,
    nodes: Mapping[str, int],
    occupied: Iterable[GpuKey] = (),
    policy: PlacementPolicy | None = None,
    model_occupied: Iterable[GpuKey] | None = None,
) -> ReleaseChoice | None:
    """Pick the block whose release gives the most back under ``policy``.

    ``candidates`` are the blocks the shrinking model currently holds;
    ``occupied`` is the cluster-wide occupancy (those blocks included);
    ``model_occupied`` the GPUs the model holds (default: the candidates' GPUs).
    Ties on merge gain go to the most loaded node, then the highest address, so a
    model shrinking off a whole node gives back a contiguous tail rather than
    every other GPU.
    """
    node_map = _normalise_nodes(nodes)
    busy = _normalise_occupied(occupied, node_map)
    candidates = list(candidates)
    if model_occupied is None:
        model_occupied = {key for block in candidates for key in block.keys}
    return _choose_release_indexed(
        list(enumerate(candidates)),
        nodes=node_map,
        occupied=busy,
        model_occupied=_normalise_occupied(model_occupied, node_map),
        policy=policy or BEST_FIT,
    )


def plan_releases(
    candidates: Sequence[GpuBlock],
    *,
    nodes: Mapping[str, int],
    occupied: Iterable[GpuKey] = (),
    count: int = 1,
    policy: PlacementPolicy | None = None,
    model_occupied: Iterable[GpuKey] | None = None,
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
    candidates = list(candidates)
    if model_occupied is None:
        model_occupied = {key for block in candidates for key in block.keys}
    model_busy = _normalise_occupied(model_occupied, node_map)
    policy = policy or BEST_FIT
    remaining = list(enumerate(candidates))
    picks: list[ReleaseChoice] = []
    for _ in range(count):
        choice = _choose_release_indexed(
            remaining,
            nodes=node_map,
            occupied=busy,
            model_occupied=model_busy,
            policy=policy,
        )
        if choice is None:
            break
        picks.append(choice)
        busy -= set(choice.block.keys)
        model_busy -= set(choice.block.keys)
        remaining = [item for item in remaining if item[0] != choice.index]
    return picks

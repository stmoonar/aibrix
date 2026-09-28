
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import re

from tre_common.gpu_placement import (
    GpuBlock,
    PlacementPolicy,
    choose_placement,
    enumerate_blocks,
    plan_releases,
)
from tre_common.registry import ClusterTopology, tp_size_error


_NAT_SPLIT = re.compile(r"(\d+)")


@dataclass(frozen=True)
class Slot:
    node: str
    gpu_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "gpu_ids", tuple(self.gpu_ids))


@dataclass(frozen=True)
class Binding:
    serve_id: str
    model: str
    slot: Slot
    awake: bool
    hidden: bool = False

    @property
    def binding_id(self) -> str:
        """Stable logical identity; pod names remain replaceable observations."""
        gpu_ids = ",".join(str(gpu_id) for gpu_id in self.slot.gpu_ids)
        return f"{self.model}/{self.slot.node}/{gpu_ids}"


def binding_sort_key(binding: Binding) -> tuple[object, ...]:
    """Canonical ordering shared by reconciliation and persistence."""
    return natural_key(binding.serve_id)


def natural_key(value: object) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part
        for part in _NAT_SPLIT.split(str(value))
    )


def node_gpu_counts(topology: ClusterTopology) -> dict[str, int]:
    """``{node name: gpu count}`` in the shape :mod:`tre_common.gpu_placement` wants."""
    return {node.name: node.gpus for node in topology.nodes}


def slot_block(slot: Slot) -> GpuBlock:
    return GpuBlock(slot.node, tuple(slot.gpu_ids))


def block_slot(block: GpuBlock) -> Slot:
    return Slot(block.node, tuple(block.gpu_ids))


def is_buddy_aligned(slot: Slot, nodes: Mapping[str, int]) -> bool:
    """True when ``slot`` is an aligned power-of-two run inside a known node.

    Placement scoring is only defined for buddy blocks; a slot that fails this
    (a hand-written registry pairing gpu 1 with gpu 2, say) is ranked last by
    natural order instead of raising out of the planner.
    """
    gpu_ids = tuple(slot.gpu_ids)
    size = len(gpu_ids)
    if size == 0 or (size & (size - 1)) != 0:
        return False
    gpus = nodes.get(slot.node)
    if gpus is None:
        return False
    if gpu_ids != tuple(range(gpu_ids[0], gpu_ids[0] + size)):
        return False
    return gpu_ids[0] % size == 0 and gpu_ids[-1] < gpus


def gpu_slot_candidates(topology: ClusterTopology, tp_size: int) -> list[Slot]:
    """Every aligned ``tp_size``-GPU slot the topology declares, low address first.

    ``two_gpu_slots`` stays the authority on which GPUs are bindable at all (it is
    what :meth:`SlotAllocator._validate_slot` enforces); this narrows that set to
    the buddy blocks of the requested width.
    """
    nodes = node_gpu_counts(topology)
    if not nodes or tp_size > max(nodes.values()):
        return []
    declared = {
        node.name: {gpu for pair in node.two_gpu_slots for gpu in pair}
        for node in topology.nodes
    }
    return [
        block_slot(block)
        for block in enumerate_blocks(nodes, tp_size)
        if declared.get(block.node, set()).issuperset(block.gpu_ids)
    ]


def awake_gpus(bindings) -> set[tuple[str, int]]:
    """``(node, gpu)`` of every GPU an awake binding holds."""
    return {
        (binding.slot.node, gpu)
        for binding in bindings
        if binding.awake
        for gpu in binding.slot.gpu_ids
    }


def awake_model_counts(bindings) -> dict[str, int]:
    """``{model: awake bindings}`` (input of ``PlacementPolicy.for_awake``)."""
    counts: dict[str, int] = {}
    for binding in bindings:
        if binding.awake:
            counts[binding.model] = counts.get(binding.model, 0) + 1
    return counts


def model_awake_gpus(bindings, models) -> set[tuple[str, int]]:
    """``(node, gpu)`` of every GPU an awake binding of one of ``models`` holds."""
    models = {models} if isinstance(models, str) else set(models)
    return awake_gpus(binding for binding in bindings if binding.model in models)


def release_order(
    candidates: list["Binding"],
    *,
    bindings: list["Binding"],
    topology: ClusterTopology,
    already_released: "list[Binding] | tuple[Binding, ...]" = (),
    policy: PlacementPolicy | None = None,
) -> list["Binding"]:
    """Awake bindings ordered by :func:`tre_common.gpu_placement.plan_releases`.

    Mirror image of placement under the same ``policy`` (the registry placement
    policy; None = plain buddy best-fit): the replica whose slot merges into the
    largest free block goes first, then the one on the most loaded node / the node
    holding most of the model, so shrinking hands back aligned pairs and the load
    stays balanced.  Shared by the service-manager shrink and the controller's
    safescale probe order.  ``already_released`` are bindings the caller stops
    first (the hidden ones), so their GPUs count as free while the rest are scored.
    """
    if len(candidates) <= 1:
        return list(candidates)
    nodes = node_gpu_counts(topology)
    scorable = [
        binding for binding in candidates if is_buddy_aligned(binding.slot, nodes)
    ]
    if not scorable:
        return list(candidates)
    scorable_ids = {binding.serve_id for binding in scorable}
    rest = [binding for binding in candidates if binding.serve_id not in scorable_ids]
    occupied = awake_gpus(bindings)
    model_occupied = model_awake_gpus(bindings, {binding.model for binding in candidates})
    for binding in already_released:
        released = {(binding.slot.node, gpu) for gpu in binding.slot.gpu_ids}
        occupied -= released
        model_occupied -= released
    picks = plan_releases(
        [slot_block(binding.slot) for binding in scorable],
        nodes=nodes,
        occupied=occupied,
        count=len(scorable),
        policy=policy,
        model_occupied=model_occupied & occupied,
    )
    return [scorable[pick.index] for pick in picks] + rest


@dataclass(frozen=True)
class Migration:
    serve_id: str
    from_slot: Slot
    to_slot: Slot


class SlotAllocator:
    def __init__(
        self,
        topology: ClusterTopology,
        bindings: list[Binding],
        *,
        allow_awake_conflicts: bool = False,
        policy: PlacementPolicy | None = None,
    ) -> None:
        self._topology = topology
        self._allow_awake_conflicts = allow_awake_conflicts
        #: Registry placement policy (None = plain buddy best-fit).
        self._policy = policy
        self._bindings: dict[str, Binding] = {}
        self._awake_gpu_to_serve: dict[tuple[str, int], str] = {}
        for binding in bindings:
            self.bind(binding.serve_id, binding.model, binding.slot, awake=binding.awake)

    def find_slot(self, tp_size: int, model: str | None = None) -> Slot | None:
        """Best free slot of ``tp_size`` GPUs for ``model`` under the placement policy.

        Shares :func:`tre_common.gpu_placement.choose_placement` with the
        controller, so a cold create lands the same way a wake does.  ``model``
        (optional) spreads the model's replicas across nodes.
        """
        self._validate_tp_size(tp_size)
        candidates = gpu_slot_candidates(self._topology, tp_size)
        if not candidates:
            return None
        bindings = list(self._bindings.values())
        policy = self._policy.for_awake(awake_model_counts(bindings)) if self._policy else None
        choice = choose_placement(
            [slot_block(slot) for slot in candidates],
            nodes=node_gpu_counts(self._topology),
            occupied=set(self._awake_gpu_to_serve),
            tp_size=tp_size,
            policy=policy,
            model_occupied=model_awake_gpus(bindings, model) if model else (),
        )
        return None if choice is None else block_slot(choice.block)

    def bind(self, serve_id: str, model: str, slot: Slot, *, awake: bool = True) -> None:
        self._validate_slot(slot)
        if serve_id in self._bindings:
            raise ValueError(f"serve already bound: {serve_id}")
        if awake:
            for gpu in slot.gpu_ids:
                occupant = self._awake_gpu_to_serve.get((slot.node, gpu))
                if occupant is not None:
                    if self._allow_awake_conflicts:
                        continue
                    raise ValueError(f"gpu already has awake binding: {slot.node}/{gpu} occupied by {occupant}")
        binding = Binding(serve_id=serve_id, model=model, slot=slot, awake=awake)
        self._bindings[serve_id] = binding
        if awake:
            for gpu in slot.gpu_ids:
                self._awake_gpu_to_serve.setdefault((slot.node, gpu), serve_id)

    def release(self, serve_id: str) -> None:
        binding = self._bindings.pop(serve_id)
        if binding.awake:
            for gpu in binding.slot.gpu_ids:
                if self._awake_gpu_to_serve.get((binding.slot.node, gpu)) == serve_id:
                    self._awake_gpu_to_serve.pop((binding.slot.node, gpu), None)

    def feasible_wake(self, serve_id: str) -> bool:
        binding = self._bindings.get(serve_id)
        if binding is None:
            return False
        for gpu in binding.slot.gpu_ids:
            occupant = self._awake_gpu_to_serve.get((binding.slot.node, gpu))
            if occupant is not None and occupant != serve_id:
                return False
        return True

    def plan_defrag(self, tp_size: int) -> list[Migration] | None:
        self._validate_tp_size(tp_size)
        if self.find_slot(tp_size) is not None:
            return []
        if tp_size != 2:
            return None

        for target_node, target_pair in self._two_gpu_slots():
            target_occupied = [gpu for gpu in target_pair if self._is_occupied(target_node, gpu)]
            if len(target_occupied) != 1:
                continue
            target_free_gpu = next(gpu for gpu in target_pair if gpu not in target_occupied)
            for source_node, source_pair in self._two_gpu_slots():
                if source_node == target_node and source_pair == target_pair:
                    continue
                source_occupied = [gpu for gpu in source_pair if self._is_occupied(source_node, gpu)]
                if len(source_occupied) != 1:
                    continue
                source_gpu = source_occupied[0]
                serve_id = self._awake_gpu_to_serve[(source_node, source_gpu)]
                return [
                    Migration(
                        serve_id=serve_id,
                        from_slot=Slot(source_node, (source_gpu,)),
                        to_slot=Slot(target_node, (target_free_gpu,)),
                    )
                ]
        return None

    def snapshot(self) -> dict:
        return {
            serve_id: {
                "model": binding.model,
                "node": binding.slot.node,
                "gpu_ids": list(binding.slot.gpu_ids),
                "awake": binding.awake,
                "hidden": binding.hidden,
            }
            for serve_id, binding in sorted(self._bindings.items())
        }

    def _two_gpu_slots(self):
        for node in self._topology.nodes:
            for pair in node.two_gpu_slots:
                yield node.name, tuple(pair)

    def _is_occupied(self, node: str, gpu: int) -> bool:
        return (node, gpu) in self._awake_gpu_to_serve

    def _validate_tp_size(self, tp_size: int) -> None:
        widest = max((node.gpus for node in self._topology.nodes), default=None)
        problem = tp_size_error(tp_size, widest_node_gpus=widest)
        if problem:
            raise ValueError(problem)

    def _validate_slot(self, slot: Slot) -> None:
        if len(slot.gpu_ids) not in (1, 2):
            raise ValueError("slot must contain one or two GPUs")
        for node, pair in self._two_gpu_slots():
            if node != slot.node:
                continue
            if len(slot.gpu_ids) == 2 and tuple(slot.gpu_ids) == pair:
                return
            if len(slot.gpu_ids) == 1 and slot.gpu_ids[0] in pair:
                return
        raise ValueError(f"invalid slot: {slot}")


from __future__ import annotations

from dataclasses import dataclass

from tre_common.registry import ClusterTopology


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


@dataclass(frozen=True)
class Migration:
    serve_id: str
    from_slot: Slot
    to_slot: Slot


class SlotAllocator:
    def __init__(self, topology: ClusterTopology, bindings: list[Binding]) -> None:
        self._topology = topology
        self._bindings: dict[str, Binding] = {}
        self._gpu_to_serve: dict[tuple[str, int], str] = {}
        for binding in bindings:
            self.bind(binding.serve_id, binding.model, binding.slot, awake=binding.awake)

    def find_slot(self, tp_size: int) -> Slot | None:
        self._validate_tp_size(tp_size)
        if tp_size == 2:
            for node, pair in self._two_gpu_slots():
                if all(not self._is_occupied(node, gpu) for gpu in pair):
                    return Slot(node, pair)
            return None

        for node, pair in self._two_gpu_slots():
            occupied = [gpu for gpu in pair if self._is_occupied(node, gpu)]
            if len(occupied) == 1:
                free_gpu = next(gpu for gpu in pair if gpu not in occupied)
                return Slot(node, (free_gpu,))
        for node, pair in self._two_gpu_slots():
            if all(not self._is_occupied(node, gpu) for gpu in pair):
                return Slot(node, (pair[0],))
        return None

    def bind(self, serve_id: str, model: str, slot: Slot, *, awake: bool = True) -> None:
        self._validate_slot(slot)
        if serve_id in self._bindings:
            raise ValueError(f"serve already bound: {serve_id}")
        for gpu in slot.gpu_ids:
            if self._is_occupied(slot.node, gpu):
                raise ValueError(f"gpu already occupied: {slot.node}/{gpu}")
        binding = Binding(serve_id=serve_id, model=model, slot=slot, awake=awake)
        self._bindings[serve_id] = binding
        for gpu in slot.gpu_ids:
            self._gpu_to_serve[(slot.node, gpu)] = serve_id

    def release(self, serve_id: str) -> None:
        binding = self._bindings.pop(serve_id)
        for gpu in binding.slot.gpu_ids:
            self._gpu_to_serve.pop((binding.slot.node, gpu), None)

    def feasible_wake(self, serve_id: str) -> bool:
        return serve_id in self._bindings

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
                if source_node != target_node or source_pair == target_pair:
                    continue
                source_occupied = [gpu for gpu in source_pair if self._is_occupied(source_node, gpu)]
                if len(source_occupied) != 1:
                    continue
                source_gpu = source_occupied[0]
                serve_id = self._gpu_to_serve[(source_node, source_gpu)]
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
            }
            for serve_id, binding in sorted(self._bindings.items())
        }

    def _two_gpu_slots(self):
        for node in self._topology.nodes:
            for pair in node.two_gpu_slots:
                yield node.name, tuple(pair)

    def _is_occupied(self, node: str, gpu: int) -> bool:
        return (node, gpu) in self._gpu_to_serve

    def _validate_tp_size(self, tp_size: int) -> None:
        if tp_size not in (1, 2):
            raise ValueError("tp_size must be 1 or 2")

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

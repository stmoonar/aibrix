"""The binding set rendered from the registry (single source of truth).

A *binding* is one model instance pinned to one GPU slot: ``model/node/gpus``.
Both the model-Deployment generator (``deploy/gen_model_manifests.py``) and the
service-manager desired-state seeding (plan 2026-09-27 D7) derive the set of
bindings from this one function, so "which Deployments exist" and "which
bindings the startup gate admits" can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

from tre_common.registry import ModelSpec, Registry

#: At most this many bindings (sleeping or awake) may share one physical GPU.
MAX_BOUND_PER_GPU = 3


@dataclass(frozen=True)
class BindingSpec:
    model: str
    node: str
    gpu_ids: tuple[int, ...]

    @property
    def binding_id(self) -> str:
        # Same format as tre_sm.allocator.slots.Binding.binding_id.
        return f"{self.model}/{self.node}/{','.join(str(gpu) for gpu in self.gpu_ids)}"


def feasible_slots(registry: Registry, model: ModelSpec) -> list[tuple[str, tuple[int, ...]]]:
    """The GPU slots a model is bound to, in registry order.

    ``max_replicas`` is the GPU layout size (how many bindings exist), not the
    scaling cap: that is ``models[].max_awake_replicas``, enforced by the
    controller planner and the service-manager, never here.
    """
    slots: list[tuple[str, tuple[int, ...]]] = []
    for node in registry.topology().nodes:
        if model.tp_size == 1:
            slots.extend((node.name, (gpu,)) for gpu in range(node.gpus))
        elif model.tp_size == 2:
            slots.extend((node.name, tuple(slot)) for slot in node.two_gpu_slots)
    return slots[: model.max_replicas]


def render_binding_set(registry: Registry) -> list[BindingSpec]:
    """Every binding the registry declares, in manifest order.

    Raises ValueError when a GPU would carry more than MAX_BOUND_PER_GPU bindings.
    """
    bindings: list[BindingSpec] = []
    bound_counts: dict[tuple[str, int], int] = {}
    for model in registry.models():
        for node_name, gpu_ids in feasible_slots(registry, model):
            for gpu in gpu_ids:
                key = (node_name, gpu)
                bound_counts[key] = bound_counts.get(key, 0) + 1
                if bound_counts[key] > MAX_BOUND_PER_GPU:
                    raise ValueError(
                        f"gpu bound budget exceeded for {node_name}/{gpu}: "
                        f"{bound_counts[key]} > {MAX_BOUND_PER_GPU}"
                    )
            bindings.append(BindingSpec(model.name, node_name, tuple(gpu_ids)))
    return bindings

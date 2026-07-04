from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Protocol

from tre_common.registry import ClusterTopology
from tre_sm.allocator.slots import Binding, Slot, SlotAllocator
from tre_sm.state.store import StateStore


POD_STATE_SLEEPING = "sleeping"
POD_STATE_AWAKE = "awake"
POD_STATE_HIDDEN = "hidden"
_VALID_POD_STATES = {POD_STATE_SLEEPING, POD_STATE_AWAKE, POD_STATE_HIDDEN}


@dataclass(frozen=True)
class PodRecord:
    serve_id: str
    model: str
    node: str
    cuda_visible_devices: str
    state: str = POD_STATE_AWAKE

    def to_binding(self) -> Binding:
        if self.state not in _VALID_POD_STATES:
            raise ValueError(f"unknown pod state for {self.serve_id}: {self.state}")
        return Binding(
            serve_id=self.serve_id,
            model=self.model,
            slot=Slot(self.node, _parse_cuda_visible_devices(self.cuda_visible_devices)),
            awake=self.state != POD_STATE_SLEEPING,
            hidden=self.state == POD_STATE_HIDDEN,
        )


class K8sPodClient(Protocol):
    def list_pods(self) -> list[PodRecord]: ...


@dataclass(frozen=True)
class ReconcileResult:
    version: int
    bindings: list[Binding]
    warnings: list[str]
    allocator: SlotAllocator


def reconcile_state(
    topology: ClusterTopology,
    store: StateStore,
    k8s_client: K8sPodClient,
) -> ReconcileResult:
    persisted = store.load()
    persisted_by_serve = {binding.serve_id: binding for binding in persisted.bindings}
    reconciled_by_serve: dict[str, Binding] = {}
    warnings: list[str] = []

    for pod in sorted(k8s_client.list_pods(), key=lambda item: item.serve_id):
        binding = pod.to_binding()
        previous = persisted_by_serve.get(binding.serve_id)
        if previous is not None and previous != binding:
            warnings.append(f"{binding.serve_id}: pod reality overrides persisted binding")
        if binding.serve_id in reconciled_by_serve:
            raise ValueError(f"duplicate pod observation: {binding.serve_id}")
        reconciled_by_serve[binding.serve_id] = binding

    observed_slots = {
        slot_key
        for binding in reconciled_by_serve.values()
        for slot_key in _slot_keys(binding.slot)
    }

    for binding in persisted.bindings:
        if binding.serve_id in reconciled_by_serve:
            continue
        if any(slot_key in observed_slots for slot_key in _slot_keys(binding.slot)):
            warnings.append(
                f"{binding.serve_id}: dropped stale persisted binding that overlaps pod observation"
            )
            continue
        warnings.append(f"{binding.serve_id}: persisted binding has no matching pod observation")
        reconciled_by_serve[binding.serve_id] = binding

    bindings = _auto_sleep_awake_conflicts([reconciled_by_serve[serve_id] for serve_id in sorted(reconciled_by_serve)], warnings)
    allocator = SlotAllocator(topology, bindings)
    if bindings == persisted.bindings:
        return ReconcileResult(
            version=persisted.version,
            bindings=bindings,
            warnings=warnings,
            allocator=allocator,
        )

    version = store.save(bindings, expected_version=persisted.version)
    return ReconcileResult(version=version, bindings=bindings, warnings=warnings, allocator=allocator)


def _parse_cuda_visible_devices(value: str) -> tuple[int, ...]:
    devices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not devices:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain at least one GPU id")
    return devices


def _slot_keys(slot: Slot) -> tuple[tuple[str, int], ...]:
    return tuple((slot.node, gpu) for gpu in slot.gpu_ids)


def _auto_sleep_awake_conflicts(bindings: list[Binding], warnings: list[str]) -> list[Binding]:
    awake_by_gpu: dict[tuple[str, int], str] = {}
    reconciled: list[Binding] = []
    for binding in bindings:
        conflict_key = None
        if binding.awake:
            for key in _slot_keys(binding.slot):
                if key in awake_by_gpu:
                    conflict_key = key
                    break
        if conflict_key is None:
            reconciled.append(binding)
            if binding.awake:
                for key in _slot_keys(binding.slot):
                    awake_by_gpu[key] = binding.serve_id
            continue

        node, gpu = conflict_key
        warnings.append(
            f"{binding.serve_id}: auto-slept to preserve single awake GPU invariant on {node}/{gpu}"
        )
        reconciled.append(replace(binding, awake=False, hidden=False))
    return reconciled

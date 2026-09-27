"""Seed desired fleet state (plan 2026-09-27 D7; review P2-8).

The startup gate admits a Pod only for a binding with a resident desired
record. On an empty Redis nothing used to create those records, so every Pod
was rejected forever. The service-manager therefore appends one ``resident``
record for every binding that has none. Append-only: an existing record
(awake, sleeping or absent) is never touched, so seeding is idempotent and
cannot undo an operator's or the controller's intent.

* **Binding set** = the registry's bindings (the same rendering that generates
  the model Deployments) UNION the TRE-managed Deployments that exist (a
  defrag-migrated binding lives on a slot the registry rendering does not
  list).
* **Power** comes from the pod's actual state only when that state can be
  trusted (review 2 P1-3): the pod passed the startup gate and its admission
  converged (no pending ``startup-admitted-uid``), it is Ready, and the physical
  ``/is_sleeping`` probe answers. Then an awake pod is seeded ``awake`` (hidden
  if its annotation says hidden), so re-seeding after a Redis loss never turns
  a serving fleet into "desired asleep" (which a fleet repair would then act
  on). Everything else - no pod, a pod waiting at the gate (its template
  annotation says ``hidden`` although vLLM never ran), not Ready, an admission
  still converging, or a probe that does not answer - is seeded ``sleeping``
  (resident), the safe default: it never claims a GPU for a pod nobody saw
  awake.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Iterable

from tre_common.bindings import BindingSpec, render_binding_set
from tre_common.registry import Registry
from tre_sm.allocator.topology import GPU_IDS_ANNOTATION, STATE_ANNOTATION
from tre_sm.state.fleet_store import DesiredBinding, FleetStateStore
from tre_sm.state.reconcile import POD_STATE_HIDDEN

#: Set by the startup gate's admission, cleared once the admission converged.
STARTUP_ADMITTED_ANNOTATION = "tre.aibrix.io/startup-admitted-uid"

SEED_UPDATED_BY = "registry-seed"
SEED_REASON = "registry_seed"

#: (power, hidden) for a binding id; None = no pod observed.
Observer = Callable[[str], "tuple[str, bool] | None"]


def seed_records(
    specs: Iterable[BindingSpec], *, observe: Observer | None = None
) -> list[DesiredBinding]:
    now = datetime.now(timezone.utc).isoformat()
    records = []
    for spec in specs:
        observed = observe(spec.binding_id) if observe is not None else None
        power, hidden = observed if observed is not None else ("sleeping", False)
        records.append(
            DesiredBinding(
                binding_id=spec.binding_id,
                model=spec.model,
                node=spec.node,
                gpu_ids=tuple(spec.gpu_ids),
                lifecycle="resident",
                power=power,
                hidden=hidden,
                generation=1,
                updated_at=now,
                updated_by=SEED_UPDATED_BY,
                reason=SEED_REASON if observed is None else f"{SEED_REASON}_observed_{power}",
            )
        )
    return records


def seed_binding_specs(registry: Registry, runtime_ops=None) -> list[BindingSpec]:
    """Registry bindings UNION TRE-managed Deployments (registry order first)."""
    specs = {spec.binding_id: spec for spec in render_binding_set(registry)}
    lister = getattr(runtime_ops, "list_model_deployments", None)
    if callable(lister):
        for deployment in lister():
            spec = BindingSpec(deployment.model, deployment.node, tuple(deployment.gpu_ids))
            specs.setdefault(spec.binding_id, spec)
    return list(specs.values())


def seed_binding_ids(registry: Registry, runtime_ops=None) -> set[str]:
    return {spec.binding_id for spec in seed_binding_specs(registry, runtime_ops)}


def pod_state_observer(runtime_ops, vllm_ops=None, *, port: int = 8000) -> Observer | None:
    """Observer of the pods' trusted power: ("awake", hidden) only for a Pod past
    the startup gate (admission converged), Ready, whose /is_sleeping probe
    answers False; ("sleeping", False) for a Pod that exists but is not trusted
    or is asleep; None when the binding has no Pod."""
    lister = getattr(runtime_ops, "list_pod_snapshots", None)
    if not callable(lister):
        return None
    by_binding: dict[str, object] = {}
    for snapshot in lister():
        gpu_text = snapshot.annotations.get(GPU_IDS_ANNOTATION)
        if not gpu_text:
            continue
        binding_id = f"{snapshot.model}/{snapshot.node}/{gpu_text}"
        by_binding.setdefault(binding_id, snapshot)
    probe = getattr(vllm_ops, "is_sleeping", None)

    def observe(binding_id: str) -> tuple[str, bool] | None:
        snapshot = by_binding.get(binding_id)
        if snapshot is None:
            return None
        if snapshot.annotations.get(STARTUP_ADMITTED_ANNOTATION):
            # Still inside the startup protocol: convergence decides its power.
            return ("sleeping", False)
        if not getattr(snapshot, "ready", False) or not getattr(snapshot, "pod_ip", None):
            # Waiting at the gate (or not serving): vLLM never ran / is not up.
            return ("sleeping", False)
        physical = None
        if callable(probe):
            try:
                physical = probe(snapshot.pod_ip, port=port)
            except Exception:
                physical = None
        if physical is not False:
            # Asleep, or unknown (the annotation is never trusted on its own).
            return ("sleeping", False)
        return ("awake", snapshot.annotations.get(STATE_ANNOTATION) == POD_STATE_HIDDEN)

    return observe


def seed_desired(
    registry: Registry,
    fleet_store: FleetStateStore,
    *,
    runtime_ops=None,
    vllm_ops=None,
) -> dict:
    """Append missing bindings to desired state; requires the writer fence."""
    specs = seed_binding_specs(registry, runtime_ops)
    existing = {item.binding_id for item in fleet_store.load_desired().bindings}
    missing = [spec for spec in specs if spec.binding_id not in existing]
    if not missing:
        return {"added": [], "desired_version": fleet_store.load_desired().version}
    observe = pod_state_observer(runtime_ops, vllm_ops) if runtime_ops is not None else None
    return fleet_store.append_missing_desired(seed_records(missing, observe=observe))


def seed_desired_from_registry(registry: Registry, fleet_store: FleetStateStore) -> dict:
    """Registry-only seeding (no pod observation): every record resident+sleeping."""
    return fleet_store.append_missing_desired(seed_records(render_binding_set(registry)))


def registry_binding_ids(registry: Registry) -> set[str]:
    return {spec.binding_id for spec in render_binding_set(registry)}

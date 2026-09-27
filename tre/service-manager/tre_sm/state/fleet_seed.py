"""Seed desired fleet state from the registry (plan 2026-09-27 D7).

The startup gate admits a Pod only for a binding with a resident desired
record. On an empty Redis nothing used to create those records, so every Pod
was rejected forever. The registry is the declaration of which bindings exist
(the same rendering that generates the model Deployments), so the
service-manager seeds one ``resident`` + ``sleeping`` record for every
registry binding that has none. Append-only: an existing record (awake,
sleeping or absent) is never touched, so seeding is idempotent and cannot undo
an operator's or the controller's intent.
"""

from __future__ import annotations

from datetime import datetime, timezone

from tre_common.bindings import BindingSpec, render_binding_set
from tre_common.registry import Registry
from tre_sm.state.fleet_store import DesiredBinding, FleetStateStore

SEED_UPDATED_BY = "registry-seed"
SEED_REASON = "registry_seed"


def seed_records(specs: list[BindingSpec]) -> list[DesiredBinding]:
    now = datetime.now(timezone.utc).isoformat()
    return [
        DesiredBinding(
            binding_id=spec.binding_id,
            model=spec.model,
            node=spec.node,
            gpu_ids=tuple(spec.gpu_ids),
            lifecycle="resident",
            power="sleeping",
            hidden=False,
            generation=1,
            updated_at=now,
            updated_by=SEED_UPDATED_BY,
            reason=SEED_REASON,
        )
        for spec in specs
    ]


def seed_desired_from_registry(registry: Registry, fleet_store: FleetStateStore) -> dict:
    """Append missing registry bindings to desired state; requires the writer fence."""
    return fleet_store.append_missing_desired(seed_records(render_binding_set(registry)))


def registry_binding_ids(registry: Registry) -> set[str]:
    return {spec.binding_id for spec in render_binding_set(registry)}

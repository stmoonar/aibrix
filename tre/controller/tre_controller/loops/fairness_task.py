from __future__ import annotations

from tre_common.metrics_schema import MetricsSnapshot
from tre_common.registry import Registry
from tre_controller.loops.tick import LoopTickResult, PlannerQueue, run_planner_tick
from tre_controller.planning.planner import ClusterView


def run_fairness_tick(
    snapshot: MetricsSnapshot,
    *,
    queue: PlannerQueue,
    registry: Registry,
    cluster_view: ClusterView | None = None,
    active_probe_models: set[str] | None = None,
) -> LoopTickResult:
    return run_planner_tick(
        snapshot,
        queue=queue,
        registry=registry,
        rescue_due=False,
        fairness_due=True,
        cluster_view=cluster_view,
        active_probe_models=active_probe_models,
    )

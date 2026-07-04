from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Protocol

from tre_common.metrics_schema import MetricsSnapshot
from tre_common.registry import Registry
from tre_controller.loops.tick import LoopTickResult, PlannerQueue, run_planner_tick
from tre_controller.planning.planner import ClusterView


class ClusterViewReader(Protocol):
    def get(self) -> ClusterView | None: ...


class SnapshotReader(Protocol):
    def get(self) -> MetricsSnapshot | None: ...


class DecisionWriter(Protocol):
    def write(self, loop_name: str, snapshot: MetricsSnapshot, result: LoopTickResult) -> None: ...


class RescueTaskConfig(Protocol):
    rescue_interval_s: float


def run_rescue_tick(
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
        rescue_due=True,
        fairness_due=False,
        cluster_view=cluster_view,
        active_probe_models=active_probe_models,
    )


async def rescue_task(
    snapshot_box: SnapshotReader,
    *,
    queue: PlannerQueue,
    registry: Registry,
    cfg: RescueTaskConfig,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    cluster_view: ClusterView | None = None,
    cluster_view_box: ClusterViewReader | None = None,
    active_probe_models: set[str] | None = None,
    decision_writer: DecisionWriter | None = None,
) -> None:
    while True:
        snapshot = snapshot_box.get()
        if snapshot is not None:
            result = run_rescue_tick(
                snapshot,
                queue=queue,
                registry=registry,
                cluster_view=_current_cluster_view(cluster_view, cluster_view_box),
                active_probe_models=active_probe_models,
            )
            if decision_writer is not None:
                decision_writer.write("rescue", snapshot, result)
        await sleep(cfg.rescue_interval_s)


def _current_cluster_view(
    cluster_view: ClusterView | None,
    cluster_view_box: ClusterViewReader | None,
) -> ClusterView | None:
    if cluster_view is not None:
        return cluster_view
    if cluster_view_box is None:
        return None
    return cluster_view_box.get()

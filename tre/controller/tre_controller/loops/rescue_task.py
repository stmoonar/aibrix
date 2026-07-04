from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Protocol

from tre_common.metrics_schema import MetricsSnapshot
from tre_common.registry import Registry
from tre_controller.loops.tick import LoopTickResult, PlannerQueue, run_planner_tick
from tre_controller.planning.planner import ClusterView


class SnapshotReader(Protocol):
    def get(self) -> MetricsSnapshot | None: ...


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
    active_probe_models: set[str] | None = None,
) -> None:
    while True:
        snapshot = snapshot_box.get()
        if snapshot is not None:
            run_rescue_tick(
                snapshot,
                queue=queue,
                registry=registry,
                cluster_view=cluster_view,
                active_probe_models=active_probe_models,
            )
        await sleep(cfg.rescue_interval_s)

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.fairness_task import fairness_task
from tre_controller.loops.metrics_task import MetricsTaskConfig, SnapshotBox, SnapshotStore, metrics_task
from tre_controller.loops.rescue_task import rescue_task

TaskFactory = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class ControllerDependencies:
    store: SnapshotStore
    snapshot_box: SnapshotBox
    queue: ActionQueue
    registry: Any


@dataclass(frozen=True)
class ControllerTaskSpec:
    name: str
    factory: TaskFactory


def build_controller_task_specs(
    deps: ControllerDependencies,
    cfg: MetricsTaskConfig,
) -> tuple[ControllerTaskSpec, ...]:
    specs: list[ControllerTaskSpec] = [
        ControllerTaskSpec("metrics", lambda: metrics_task(deps.store, deps.snapshot_box, cfg)),
    ]
    if not bool(getattr(cfg, "enable_tre_scaling", True)):
        return tuple(specs)

    if not bool(getattr(cfg, "ablation_disable_fast_loop", False)):
        specs.append(
            ControllerTaskSpec(
                "rescue",
                lambda: rescue_task(deps.snapshot_box, queue=deps.queue, registry=deps.registry, cfg=cfg),
            )
        )
    specs.append(
        ControllerTaskSpec(
            "fairness",
            lambda: fairness_task(deps.snapshot_box, queue=deps.queue, registry=deps.registry, cfg=cfg),
        )
    )
    specs.append(ControllerTaskSpec("action_queue", lambda: deps.queue.run()))
    return tuple(specs)


async def run_controller(deps: ControllerDependencies, cfg: MetricsTaskConfig) -> None:
    await asyncio.gather(*(spec.factory() for spec in build_controller_task_specs(deps, cfg)))

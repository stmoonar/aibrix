from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from tre_common.registry import Registry, load_registry
from tre_controller.config import ControllerConfig
from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.fairness_task import fairness_task
from tre_controller.loops.metrics_task import MetricsTaskConfig, SnapshotBox, SnapshotStore, metrics_task
from tre_controller.loops.rescue_task import rescue_task
from tre_controller.sm_client import AsyncTransport, ServiceManagerClient
from tre_controller.store.metrics_store import MetricsStore

TaskFactory = Callable[[], Awaitable[None]]
RedisClientFactory = Callable[[str], Any]
ControllerRunner = Callable[["ControllerDependencies", ControllerConfig], Awaitable[None]]


@dataclass(frozen=True)
class ControllerDependencies:
    store: SnapshotStore
    snapshot_box: SnapshotBox
    queue: ActionQueue
    registry: Registry


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


def create_controller_dependencies(
    cfg: ControllerConfig,
    *,
    redis_client: Any | None = None,
    redis_client_factory: RedisClientFactory | None = None,
    sm_transport: AsyncTransport | None = None,
) -> ControllerDependencies:
    registry = load_registry(cfg.registry_path)
    redis_client = redis_client if redis_client is not None else _create_redis_client(cfg.redis_url, redis_client_factory)
    store = MetricsStore(
        redis_client,
        registry,
        instant_sample_interval_ms=cfg.instant_sample_interval_ms,
        percentile_mode=cfg.percentile_mode,
    )
    sm_client = ServiceManagerClient(cfg.service_manager_url, transport=sm_transport)
    return ControllerDependencies(
        store=store,
        snapshot_box=SnapshotBox(),
        queue=ActionQueue(sm_client),
        registry=registry,
    )


async def main(
    *,
    env: Mapping[str, str] | None = None,
    redis_client_factory: RedisClientFactory | None = None,
    sm_transport: AsyncTransport | None = None,
    runner: ControllerRunner | None = None,
) -> None:
    cfg = ControllerConfig.from_env(env)
    deps = create_controller_dependencies(
        cfg,
        redis_client_factory=redis_client_factory,
        sm_transport=sm_transport,
    )
    await (runner or run_controller)(deps, cfg)


def _create_redis_client(redis_url: str, redis_client_factory: RedisClientFactory | None) -> Any:
    if redis_client_factory is not None:
        return redis_client_factory(redis_url)
    try:
        import redis  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("redis package is required unless redis_client_factory is provided") from exc
    return redis.Redis.from_url(redis_url)


async def run_controller(deps: ControllerDependencies, cfg: MetricsTaskConfig) -> None:
    await asyncio.gather(*(spec.factory() for spec in build_controller_task_specs(deps, cfg)))

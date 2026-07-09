from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from tre_common.registry import Registry, load_registry
from tre_controller.config import ControllerConfig
from tre_controller.loops.action_queue import ActionQueue
from tre_controller.mode import ObserveModeGate
from tre_controller.profiling import TickProfiler, build_profiler
from tre_controller.loops.cluster_view_task import ClusterViewBox, cluster_view_task
from tre_controller.loops.decision_snapshot import DecisionSnapshotWriter
from tre_controller.loops.fairness_task import fairness_task
from tre_controller.loops.metrics_task import MetricsTaskConfig, SnapshotBox, SnapshotStore, metrics_task
from tre_controller.loops.rescue_task import rescue_task
from tre_controller.loops.safescale_task import safescale_task
from tre_controller.planning.safescale import SafeScaleStateMachine
from tre_controller.signals.trs import SignalState
from tre_controller.sm_client import AsyncTransport, ServiceManagerClient
from tre_controller.store.metrics_store import MetricsStore
from tre_controller.store.state_store import ControllerStateStore

TaskFactory = Callable[[], Awaitable[None]]
RedisClientFactory = Callable[[str], Any]
ControllerRunner = Callable[["ControllerDependencies", ControllerConfig], Awaitable[None]]


@dataclass(frozen=True)
class ControllerDependencies:
    store: SnapshotStore
    snapshot_box: SnapshotBox
    queue: ActionQueue
    sm_client: ServiceManagerClient
    cluster_view_box: ClusterViewBox
    decision_writer: DecisionSnapshotWriter
    safescale: SafeScaleStateMachine
    registry: Registry
    signal_state: SignalState
    profiler: "TickProfiler | None" = None


@dataclass(frozen=True)
class ControllerTaskSpec:
    name: str
    factory: TaskFactory


def build_controller_task_specs(
    deps: ControllerDependencies,
    cfg: MetricsTaskConfig,
) -> tuple[ControllerTaskSpec, ...]:
    specs: list[ControllerTaskSpec] = [
        ControllerTaskSpec("metrics", lambda: metrics_task(deps.store, deps.snapshot_box, cfg, prof=deps.profiler)),
    ]
    if not bool(getattr(cfg, "enable_tre_scaling", True)):
        return tuple(specs)

    specs.append(
        ControllerTaskSpec(
            "cluster_view",
            lambda: cluster_view_task(deps.sm_client, deps.registry.topology(), deps.cluster_view_box, cfg),
        )
    )

    if not bool(getattr(cfg, "ablation_disable_fast_loop", False)):
        specs.append(
            ControllerTaskSpec(
                "rescue",
                lambda: rescue_task(
                    deps.snapshot_box,
                    queue=deps.queue,
                    registry=deps.registry,
                    cfg=cfg,
                    cluster_view_box=deps.cluster_view_box,
                    decision_writer=deps.decision_writer,
                    safescale=deps.safescale,
                    signal_state=deps.signal_state,
                    prof=deps.profiler,
                ),
            )
        )
    specs.append(
        ControllerTaskSpec(
            "fairness",
            lambda: fairness_task(
                deps.snapshot_box,
                queue=deps.queue,
                registry=deps.registry,
                cfg=cfg,
                cluster_view_box=deps.cluster_view_box,
                decision_writer=deps.decision_writer,
                safescale=deps.safescale,
                signal_state=deps.signal_state,
                prof=deps.profiler,
            ),
        )
    )
    if not bool(getattr(cfg, "ablation_disable_safescale", False)):
        specs.append(
            ControllerTaskSpec(
                "safescale",
                lambda: safescale_task(
                    deps.snapshot_box,
                    queue=deps.queue,
                    registry=deps.registry,
                    safescale=deps.safescale,
                    cfg=cfg,
                    signal_state=deps.signal_state,
                ),
            )
        )
    specs.append(ControllerTaskSpec("action_queue", lambda: deps.queue.run()))
    if deps.profiler is not None:
        specs.append(ControllerTaskSpec("profile_flush", lambda: deps.profiler.flush_loop()))
        specs.append(
            ControllerTaskSpec(
                "profile_proc_sampler",
                lambda: deps.profiler.proc_sampler_loop(
                    interval_s=getattr(cfg, "profile_proc_sample_interval_s", 5.0)
                ),
            )
        )
    return tuple(specs)


def create_controller_dependencies(
    cfg: ControllerConfig,
    *,
    redis_client: Any | None = None,
    redis_client_factory: RedisClientFactory | None = None,
    sm_transport: AsyncTransport | None = None,
) -> ControllerDependencies:
    registry = load_registry(cfg.registry_path)
    injected_redis_client = redis_client is not None
    redis_client = redis_client if redis_client is not None else _create_redis_client(cfg.redis_url, redis_client_factory)
    metrics_redis_client = (
        redis_client
        if injected_redis_client or cfg.metrics_redis_url == cfg.redis_url
        else _create_redis_client(cfg.metrics_redis_url, redis_client_factory)
    )
    store = MetricsStore(
        metrics_redis_client,
        registry,
        instant_sample_interval_ms=cfg.instant_sample_interval_ms,
        percentile_mode=cfg.percentile_mode,
        schema=cfg.metrics_schema,
        histogram_lookback_ms=cfg.histogram_lookback_ms,
        min_latency_samples=cfg.min_latency_samples,
    )
    sm_client = ServiceManagerClient(
        cfg.service_manager_url, transport=sm_transport, slow_timeout_s=cfg.sm_slow_timeout_s
    )
    safescale = SafeScaleStateMachine(config=cfg.safescale, store=ControllerStateStore(redis_client))
    safescale.restore()
    observe_gate = ObserveModeGate(redis_client)
    profiler = build_profiler(cfg, redis_client)
    return ControllerDependencies(
        store=store,
        snapshot_box=SnapshotBox(),
        queue=ActionQueue(sm_client, is_observe=observe_gate.is_observe, prof=profiler),
        sm_client=sm_client,
        cluster_view_box=ClusterViewBox(),
        decision_writer=DecisionSnapshotWriter(redis_client),
        safescale=safescale,
        registry=registry,
        signal_state=SignalState(warmup_ms=cfg.signal_warmup_ms),
        profiler=profiler,
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

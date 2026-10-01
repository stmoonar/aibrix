from __future__ import annotations

import asyncio
import json
import logging
import time
import dataclasses
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from tre_common.registry import Registry, load_registry, sleep_call_timeout_errors
from tre_controller.config import ControllerConfig
from tre_controller.gateway_cadence import check_gateway_cadence
from tre_controller.gateway_clock import GatewayClockMonitor, gateway_clock_task
from tre_controller.gateway_health import EnvoyStatsSource
from tre_controller.loops.action_queue import (
    ActionQueue,
    RetryPolicy,
    revalidate_commit_from_signals,
    revalidate_from_cluster_view,
    slot_lookup_from_cluster_view,
)
from tre_controller.maintenance import MaintenanceWatch
from tre_controller.mode import ObserveModeGate
from tre_controller.reconcile.hidden_orphans import HiddenOrphanDetector
from tre_controller.profiling import TickProfiler, build_profiler
from tre_controller.loops.cluster_view_task import ClusterViewBox, cluster_view_task
from tre_controller.loops.decision_snapshot import DecisionSnapshotWriter
from tre_controller.loops.fairness_task import fairness_task
from tre_controller.loops.model_state_box import ModelStateBox
from tre_controller.loops.metrics_task import MetricsTaskConfig, SnapshotBox, SnapshotStore, metrics_task
from tre_controller.loops.rescue_task import rescue_task
from tre_controller.loops.safescale_task import safescale_task
from tre_controller.planning.safescale import SafeScaleStateMachine
from tre_controller.planning.safescale_direct import (
    DirectEvidenceCollector,
    PodMetricsScraper,
    cluster_view_remaining,
    cluster_view_targets,
    cluster_view_urls,
)
from tre_controller.planning.safescale_evidence import MetricsEvidenceReader, RegistryThresholds
from tre_controller.signals.trs import BreakpointWindowConfig, SignalState
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
    hidden_orphan_detector: "HiddenOrphanDetector | None" = None
    # A13 donor-health guard source (None when TRE_GATEWAY_STATS_URL is unset).
    gateway_health: "EnvoyStatsSource | None" = None
    # Review 3: latest per-model signal state (planner ticks -> commit revalidation).
    model_state_box: "ModelStateBox | None" = None
    # B8: controller run mode (observe/active) - the planner loops read it too, so
    # no SafeScale probe is started while paused (the queue alone cannot stop that).
    observe_gate: "ObserveModeGate | None" = None
    # P2-3: SM maintenance lock (tre:v2:sm:maintenance) = SafeScale pause; shared by
    # the planner loops (no probe start) and the SafeScale loop (rollback), so a
    # period any of them saw counts against every probe window it overlaps.
    maintenance_watch: "MaintenanceWatch | None" = None
    # 2026-09-29 B+D: direct /metrics evidence of SafeScale probes (None when the
    # registry sets safescale.evidence_source: redis).
    direct_evidence: "DirectEvidenceCollector | None" = None


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
    if (
        bool(getattr(cfg, "orphan_scan_enabled", True))
        and deps.hidden_orphan_detector is not None
    ):
        specs.append(ControllerTaskSpec("hidden_orphans", deps.hidden_orphan_detector.run))
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
                    active_probe_models=lambda: _active_probe_models(deps.safescale),
                    decision_writer=deps.decision_writer,
                    safescale=deps.safescale,
                    signal_state=deps.signal_state,
                    prof=deps.profiler,
                    model_state_box=deps.model_state_box,
                    is_observe=_observe_reader(deps),
                    maintenance=deps.maintenance_watch,
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
                active_probe_models=lambda: _active_probe_models(deps.safescale),
                decision_writer=deps.decision_writer,
                safescale=deps.safescale,
                signal_state=deps.signal_state,
                prof=deps.profiler,
                model_state_box=deps.model_state_box,
                is_observe=_observe_reader(deps),
                maintenance=deps.maintenance_watch,
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
                    cluster_view_box=deps.cluster_view_box,
                    gateway_source=deps.gateway_health,
                    # Observe (2026-09-28): open probes are rolled back (unhide).
                    is_observe=_observe_reader(deps),
                    maintenance=deps.maintenance_watch,
                    direct=deps.direct_evidence,
                ),
            )
        )
    clock_monitor = _gateway_clock_monitor(deps, cfg)
    if clock_monitor is not None:
        interval_s = float(deps.registry.scaling().gateway_clock_check_s)
        specs.append(
            ControllerTaskSpec("gateway_clock", lambda: gateway_clock_task(clock_monitor, interval_s))
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


def _gateway_clock_monitor(deps: ControllerDependencies, cfg: Any) -> GatewayClockMonitor | None:
    """O1 same-clock check (review P2-3): only with O1 on, a redis-backed store and
    ``scaling.gateway_clock_check_s`` > 0."""
    breakpoint = getattr(deps.signal_state, "breakpoint", None)
    redis_client = getattr(deps.store, "redis_client", None)
    scaling = getattr(deps.registry, "scaling", None)
    if breakpoint is None or not breakpoint.enabled or redis_client is None or not callable(scaling):
        return None
    config = scaling()
    if int(getattr(config, "gateway_clock_check_s", 0) or 0) <= 0:
        return None
    return GatewayClockMonitor(
        redis_client,
        [spec.name for spec in deps.registry.models()],
        deps.signal_state,
        period_ms=int(getattr(cfg, "instant_sample_interval_ms", breakpoint.grid_ms)),
        tolerance_ms=int(config.gateway_clock_tolerance_ms),
    )


def _sleeping_pods(view: Any, model: str) -> set[str]:
    """Pods (SM serve_id) of ``model`` the fleet state reports asleep; empty without a view."""
    if view is None:
        return set()
    return {
        binding.serve_id
        for binding in getattr(view, "bindings", ()) or ()
        if binding.model == model and not binding.awake
    }


def _observe_reader(deps: ControllerDependencies) -> Callable[[], bool] | None:
    gate = deps.observe_gate
    return gate.is_observe if gate is not None else None


def _active_probe_models(safescale: SafeScaleStateMachine) -> set[str]:
    # Live read each tick: models with an unresolved safescale probe (hidden pod) must
    # not be picked as planner donors (review F3) - committing ones included (review 4).
    busy = getattr(safescale, "busy_models", None)
    if callable(busy):
        return set(busy())
    return {probe.model for probe in safescale.active_probes()}


def resolve_sm_call_timeout_s(cfg: ControllerConfig, registry: Registry) -> float:
    """The controller's timeout for slow SM calls (scale / binding power / defrag).

    TRE_SM_SLOW_TIMEOUT_SECONDS if set, else the registry's
    ``service_manager.api_call_timeout_s``. Refuses to start when it does not
    exceed the worst-case sleeping SM call (the SM would still be draining when
    the controller gives up and re-plans against a stale view)."""
    sm_config = registry.service_manager()
    explicit = getattr(cfg, "sm_slow_timeout_s", None)
    timeout = float(explicit) if explicit is not None else float(sm_config.api_call_timeout_s)
    errors = sleep_call_timeout_errors(
        sm_config,
        timeout,
        name="TRE_SM_SLOW_TIMEOUT_SECONDS" if explicit is not None else "service_manager.api_call_timeout_s",
    )
    if errors:
        raise ValueError("controller configuration: " + "; ".join(errors))
    return timeout


def create_controller_dependencies(
    cfg: ControllerConfig,
    *,
    redis_client: Any | None = None,
    redis_client_factory: RedisClientFactory | None = None,
    sm_transport: AsyncTransport | None = None,
) -> ControllerDependencies:
    registry = load_registry(cfg.registry_path)
    injected_redis_client = redis_client is not None
    redis_timeout_s = float(getattr(cfg, "redis_socket_timeout_s", 0.0) or 0.0)
    redis_client = (
        redis_client
        if redis_client is not None
        else _create_redis_client(cfg.redis_url, redis_client_factory, timeout_s=redis_timeout_s)
    )
    # The metrics reads (per-pod window ZRANGEBYSCOREs) get their own, longer timeout
    # than the state / scale-memory client: a separate client even on the same URL.
    metrics_timeout_s = float(getattr(cfg, "redis_metrics_socket_timeout_s", 0.0) or 0.0)
    share = injected_redis_client or (
        cfg.metrics_redis_url == cfg.redis_url
        and (redis_client_factory is not None or metrics_timeout_s == redis_timeout_s)
    )
    metrics_redis_client = (
        redis_client
        if share
        else _create_redis_client(cfg.metrics_redis_url, redis_client_factory, timeout_s=metrics_timeout_s)
    )
    breakpoint_config = BreakpointWindowConfig.from_registry(
        registry, grid_ms=cfg.instant_sample_interval_ms
    )
    logging.getLogger("tre_controller.signals").info(
        json.dumps({"event": "breakpoint_window_config", **dataclasses.asdict(breakpoint_config)}, sort_keys=True)
    )
    store = MetricsStore(
        metrics_redis_client,
        registry,
        instant_sample_interval_ms=cfg.instant_sample_interval_ms,
        percentile_mode=cfg.percentile_mode,
        schema=cfg.metrics_schema,
        histogram_lookback_ms=cfg.histogram_lookback_ms,
        min_latency_samples=cfg.min_latency_samples,
        # O1: the grid-aligned suffix windows the breakpoint window decides on.
        suffix_period_ms=cfg.instant_sample_interval_ms if breakpoint_config.enabled else 0,
    )
    sm_client = ServiceManagerClient(
        cfg.service_manager_url,
        transport=sm_transport,
        slow_timeout_s=resolve_sm_call_timeout_s(cfg, registry),
    )
    # A view older than two refresh periods is not "fresh" (review 4 P2-1).
    cluster_view_box = ClusterViewBox(max_age_s=max(5.0, 2.5 * float(getattr(cfg, "fairness_interval_s", 10.0))))
    # 2026-09-29: the commit gate's latency check reads the post-hide evidence window
    # (remaining pods: probe pods and pods the fleet state reports asleep excluded),
    # anchored on Redis TIME of the metrics store; thresholds from registry safescale.
    safescale = SafeScaleStateMachine(
        config=cfg.safescale,
        store=ControllerStateStore(redis_client),
        evidence=MetricsEvidenceReader(
            # Same rules as the controller's store, but no histogram lookback: a pod's
            # delta starts at its first doc stamped at or after the evidence start.
            MetricsStore(
                metrics_redis_client,
                registry,
                instant_sample_interval_ms=cfg.instant_sample_interval_ms,
                percentile_mode=cfg.percentile_mode,
                schema=cfg.metrics_schema,
                histogram_lookback_ms=0,
                min_latency_samples=cfg.min_latency_samples,
            ),
            redis_client=metrics_redis_client,
            sleeping_pods=lambda model: _sleeping_pods(cluster_view_box.get(), model),
        ),
        thresholds=RegistryThresholds(
            registry,
            mode=cfg.safescale.slo_mode,
            ttft_override_ms=cfg.safescale.ttft_p95_slo_ms,
            tpot_override_ms=cfg.safescale.tpot_p95_slo_ms,
        ),
        # Redis mode: the remaining pods a commit needs evidence of (fresh view only).
        remaining_pods=cluster_view_remaining(cluster_view_box.fresh),
    )
    safescale.restore()
    # 2026-09-29 B+D: the controller scrapes the probe's remaining pods itself
    # (pod IP from the SM fleet state, port = registry safescale.metrics_port).
    direct_evidence = None
    if safescale.direct_mode():
        direct_evidence = DirectEvidenceCollector(
            safescale,
            PodMetricsScraper(timeout_s=cfg.safescale.scrape_timeout_s),
            # Only a FRESH view names the remaining pods whose evidence a commit needs.
            cluster_view_targets(cluster_view_box.fresh, port=cfg.safescale.metrics_port),
            poll_ms=cfg.safescale.evidence_poll_ms,
            urls=cluster_view_urls(cluster_view_box.get, port=cfg.safescale.metrics_port),
        )
    # The effective evidence settings (unknown registry keys are only warned about).
    logging.getLogger("tre_controller.safescale").info(json.dumps({
        "event": "safescale_config",
        "evidence_source": cfg.safescale.evidence_source,
        "evidence_poll_ms": cfg.safescale.evidence_poll_ms,
        "scrape_timeout_s": cfg.safescale.scrape_timeout_s,
        "baseline_delay_ms": cfg.safescale.baseline_delay_ms,
        "metrics_port": cfg.safescale.metrics_port,
        "min_commit_samples": cfg.safescale.min_commit_samples,
        "window_ceiling_ms": cfg.safescale.window_ceiling_ms,
        "slo_mode": cfg.safescale.slo_mode,
    }, sort_keys=True))
    observe_gate = ObserveModeGate(redis_client)
    profiler = build_profiler(cfg, redis_client)
    model_state_box = ModelStateBox()
    return ControllerDependencies(
        store=store,
        snapshot_box=SnapshotBox(),
        queue=ActionQueue(
            sm_client,
            # C1 review P2-2: last scale action / rescue target survive a restart.
            scale_memory=ControllerStateStore(redis_client),
            scale_memory_max_age_ms=float(getattr(cfg, "scale_memory_max_age_s", 50.0)) * 1000.0,
            is_observe=observe_gate.is_observe,
            # Uncached re-check right before every capacity-changing SM call.
            is_observe_fresh=observe_gate.is_observe_fresh,
            prof=profiler,
            retry=RetryPolicy(
                max_attempts=int(getattr(cfg, "oneshot_retry_max_attempts", 6)),
                base_backoff_s=float(getattr(cfg, "oneshot_retry_base_s", 2.0)),
                max_backoff_s=float(getattr(cfg, "oneshot_retry_max_s", 30.0)),
            ),
            # One-shot retries re-check the latest cluster view; actions on a shared
            # GPU are serialized (review 2 P1-1 / P1-2).
            # Only a FRESH view may skip a retry (review 4 P2-1).
            revalidate=revalidate_from_cluster_view(cluster_view_box.fresh),
            # Review 3: a SafeScale commit is revalidated on the current signal state
            # (donor needing capacity -> unhide instead; receiver no longer needing it
            # -> upscale dropped) before every (re)try.
            revalidate_commit=revalidate_commit_from_signals(model_state_box.get, cluster_view_box.fresh),
            slot_of=slot_lookup_from_cluster_view(cluster_view_box.get),
            # Review 4 P2-3 / P2-4: preemption compensation and failed-commit
            # unhides use the view only while fresh; a SafeScale probe is resolved
            # when its one-shot action is finished (durable lifecycle).
            fresh_view=cluster_view_box.fresh,
            on_oneshot_done=lambda request_id, status, reason: safescale.resolve_request(
                request_id, status=status, reason=reason, now_ms=int(time.time() * 1000)
            ),
            # B8: a commit held (observe mode) or recovered past this age is
            # turned into the donor unhide instead of acting on stale evidence.
            commit_max_age_ms=cfg.safescale.commit_max_age_ms,
            # P3: a hide that did not take effect (failed / not sent in observe)
            # marks its probe for rollback instead of leaving it judged as hidden.
            on_hide_failed=lambda model, pods, reason: safescale.abort_probe(model, pods=pods, reason=reason),
            # 2026-09-29: the SM confirmed a probe's hide - anchor its evidence window
            # (and, on the direct path, start the baseline scrape of its remaining pods).
            on_hide_done=(
                direct_evidence.on_hide_done if direct_evidence is not None
                else (lambda model, pods: safescale.mark_hidden(model, pods=pods))
            ),
            # P2-6: a donor the SM refused with 409 floor_violation is not picked for
            # a scale-down again for TRE_FLOOR_VIOLATION_COOLDOWN_TICKS fast-loop ticks.
            floor_violation_hold_ms=(
                float(getattr(cfg, "floor_violation_cooldown_ticks", 6)) * float(cfg.rescue_interval_s) * 1000.0
            ),
            # S3: a GPU / node the SM refused a wake on is kept out of wake planning
            # (registry placement.wake_cooldown).
            wake_cooldown_s=(
                float(registry.placement().wake_cooldown_gpu_s),
                float(registry.placement().wake_cooldown_node_s),
            ),
        ),
        observe_gate=observe_gate,
        maintenance_watch=MaintenanceWatch(redis_client),
        direct_evidence=direct_evidence,
        model_state_box=model_state_box,
        sm_client=sm_client,
        cluster_view_box=cluster_view_box,
        decision_writer=DecisionSnapshotWriter(redis_client),
        safescale=safescale,
        registry=registry,
        signal_state=SignalState(
            warmup_ms=cfg.signal_warmup_ms,
            dwell_windows=cfg.dwell_windows,
            dwell_states=cfg.dwell_states,
            # O1 breakpoint window (registry scaling.breakpoint_window /
            # onset_warmup_guard / min_evidence_*), on the gateway grid.
            breakpoint=breakpoint_config,
        ),
        profiler=profiler,
        hidden_orphan_detector=HiddenOrphanDetector(
            redis_client, grace_s=cfg.orphan_grace_s
        ),
        gateway_health=(
            EnvoyStatsSource(
                cfg.gateway_stats_urls,
                [spec.name for spec in registry.models()],
                route_namespace=cfg.gateway_route_namespace,
                timeout_s=cfg.gateway_stats_timeout_s,
            )
            if getattr(cfg, "gateway_stats_urls", ())
            else None
        ),
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
    verify_gateway_cadence(deps, cfg)
    await (runner or run_controller)(deps, cfg)


def verify_gateway_cadence(deps: ControllerDependencies, cfg: ControllerConfig) -> None:
    """Startup assertion (D8): gateway write period == SCRAPE_INTERVAL_MS. Raises
    GatewayCadenceMismatch on a mismatch in ``fail`` mode; only warns without evidence."""
    redis_client = getattr(deps.store, "redis_client", None)
    if redis_client is None:
        return
    check_gateway_cadence(
        redis_client,
        [spec.name for spec in deps.registry.models()],
        mode=getattr(cfg, "gateway_interval_check", "fail"),
        expected_ms=cfg.instant_sample_interval_ms,
    )


def _create_redis_client(
    redis_url: str, redis_client_factory: RedisClientFactory | None, *, timeout_s: float = 0.0
) -> Any:
    if redis_client_factory is not None:
        return redis_client_factory(redis_url)
    try:
        import redis  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("redis package is required unless redis_client_factory is provided") from exc
    return redis.Redis.from_url(redis_url, **redis_timeouts(timeout_s))


def redis_timeouts(timeout_s: float) -> dict:
    """C1 review P3-2: socket / connect timeout of a controller Redis client, so a
    stalled Redis never blocks a loop (or a scale-memory write) indefinitely; a
    timeout surfaces as an error the callers already handle. 0 = none."""
    if not timeout_s or float(timeout_s) <= 0:
        return {}
    return {"socket_timeout": float(timeout_s), "socket_connect_timeout": float(timeout_s)}


async def run_controller(deps: ControllerDependencies, cfg: MetricsTaskConfig) -> None:
    try:
        await asyncio.gather(*(spec.factory() for spec in build_controller_task_specs(deps, cfg)))
    finally:
        # Shutdown (or a crashed loop): cancel SM dispatches still in flight
        # instead of leaving orphaned tasks behind (review 2 P3).
        shutdown = getattr(deps.queue, "shutdown", None)
        if callable(shutdown):
            await shutdown()
        # The direct-evidence scrape pool (and its scheduled baselines).
        direct = getattr(deps, "direct_evidence", None)
        if direct is not None:
            direct.close()

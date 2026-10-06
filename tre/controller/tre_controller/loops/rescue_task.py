from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol

from tre_common.metrics_schema import MetricsSnapshot
from tre_common.registry import Registry
from tre_controller.loops.tick import (
    ActiveProbeModels,
    LoopTickResult,
    PaperStateCache,
    PlannerQueue,
    SafeScaleController,
    resolve_active_probe_models,
    run_planner_tick,
)
from tre_controller.planning.planner import ClusterView, IncompletePolicy
from tre_controller.loops.model_state_box import ModelStateBox
from tre_controller.signals.trs import SignalState

if TYPE_CHECKING:
    from tre_controller.maintenance import MaintenanceWatch

if False:  # TYPE_CHECKING guard without importing typing symbol here
    from tre_controller.profiling import TickProfiler


class ClusterViewReader(Protocol):
    def get(self) -> ClusterView | None: ...


class SnapshotReader(Protocol):
    def get(self) -> MetricsSnapshot | None: ...


class DecisionWriter(Protocol):
    def write(self, loop_name: str, snapshot: MetricsSnapshot, result: LoopTickResult) -> None: ...


class RescueTaskConfig(Protocol):
    # Fallback wake period of the loop (TRE_RESCUE_INTERVAL_SECONDS, 5 s). H1 (2026-10-06):
    # the loop ticks right after every snapshot publish (SnapshotBox.wait_newer: the
    # sampler's learned phase offset plus the fetch), and at the latest rescue_interval_s
    # after its previous tick when nothing is published (free_running mode, stale
    # windows). A re-read of the same window_end_ms advances neither the EMA nor the band
    # dwell, so the effective decision cadence stays 10 s - now with no phase lag.
    rescue_interval_s: float


def run_rescue_tick(
    snapshot: MetricsSnapshot,
    *,
    queue: PlannerQueue,
    registry: Registry,
    cluster_view: ClusterView | None = None,
    active_probe_models: set[str] | None = None,
    signal_source: str = "zm",
    signal_idle_rps_eps: float = 0.05,
    safescale: SafeScaleController | None = None,
    paper_state_cache: PaperStateCache | None = None,
    incomplete_policy: IncompletePolicy = "drop_model",
    signal_state: SignalState | None = None,
    suppress_hot_proactive_probe: bool = False,
    disable_eta_gate: bool = False,
    prof: "TickProfiler | None" = None,
    action_cooldown: bool = False,
    observe_mode: bool = False,
    probe_block_reason: str | None = None,
) -> LoopTickResult:
    return run_planner_tick(
        snapshot,
        queue=queue,
        registry=registry,
        rescue_due=True,
        fairness_due=False,
        cluster_view=cluster_view,
        active_probe_models=active_probe_models,
        signal_source=signal_source,
        signal_idle_rps_eps=signal_idle_rps_eps,
        safescale=safescale,
        paper_state_cache=paper_state_cache,
        incomplete_policy=incomplete_policy,
        signal_state=signal_state,
        suppress_hot_proactive_probe=suppress_hot_proactive_probe,
        disable_eta_gate=disable_eta_gate,
        prof=prof,
        loop="rescue",
        action_cooldown=action_cooldown,
        observe_mode=observe_mode,
        probe_block_reason=probe_block_reason,
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
    active_probe_models: ActiveProbeModels = None,
    decision_writer: DecisionWriter | None = None,
    safescale: SafeScaleController | None = None,
    signal_state: SignalState | None = None,
    prof: "TickProfiler | None" = None,
    model_state_box: "ModelStateBox | None" = None,
    is_observe: Callable[[], bool] | None = None,
    maintenance: "MaintenanceWatch | None" = None,
) -> None:
    paper_state_cache = PaperStateCache(max_stale_windows=getattr(cfg, "paper_stale_max_windows", 3))
    wait_newer = getattr(snapshot_box, "wait_newer", None)
    while True:
        seen_version = getattr(snapshot_box, "version", None)
        snapshot = snapshot_box.get()
        if snapshot is not None:
            current_view = _current_cluster_view(cluster_view, cluster_view_box)
            if cluster_view is None and cluster_view_box is not None and current_view is None:
                result = LoopTickResult(submitted=0, events=("cluster_view_unavailable",))
            else:
                result = run_rescue_tick(
                    snapshot,
                    queue=queue,
                    registry=registry,
                    cluster_view=current_view,
                    active_probe_models=resolve_active_probe_models(active_probe_models),
                    signal_source=getattr(cfg, "signal_source", "zm"),
                    signal_idle_rps_eps=getattr(cfg, "signal_idle_rps_eps", 0.05),
                    safescale=safescale,
                    paper_state_cache=paper_state_cache,
                    incomplete_policy=getattr(cfg, "incomplete_policy", "drop_model"),
                    signal_state=signal_state,
                    suppress_hot_proactive_probe=getattr(cfg, "safescale_suppress_hot_proactive", False),
                    disable_eta_gate=getattr(cfg, "disable_eta_gate", False),
                    prof=prof,
                    action_cooldown=getattr(cfg, "action_cooldown", True),
                    # B8: controller mode, read per tick (ObserveModeGate, cached).
                    observe_mode=bool(is_observe()) if is_observe is not None else False,
                    # P2-3: no probe starts while the SM maintenance lock is held.
                    probe_block_reason=(
                        maintenance.probe_block_reason() if maintenance is not None else None
                    ),
                )
            if model_state_box is not None and result.classifications:
                # Review 3: the latest signal state, for the queue's commit revalidation.
                model_state_box.update(result.classifications, result.model_contexts, ts_ms=snapshot.ts_ms)
            if decision_writer is not None:
                if prof is not None:
                    _dw_t0 = time.perf_counter_ns()
                    decision_writer.write("rescue", snapshot, result)
                    prof.record(
                        {
                            "kind": "decision",
                            "loop": "rescue",
                            "ts_ms": prof.now_ms(),
                            "decision_write_ns": time.perf_counter_ns() - _dw_t0,
                        }
                    )
                else:
                    decision_writer.write("rescue", snapshot, result)
        if callable(wait_newer) and seen_version is not None:
            # Next tick on the next publish; rescue_interval_s is the fallback.
            await wait_newer(seen_version, cfg.rescue_interval_s, sleep=sleep)
        else:
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

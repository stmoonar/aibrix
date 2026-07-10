from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Protocol

from tre_common.metrics_schema import MetricsSnapshot
from tre_common.registry import Registry
from tre_controller.loops.tick import LoopTickResult, PaperStateCache, PlannerQueue, SafeScaleController, run_planner_tick
from tre_controller.planning.planner import ClusterView, IncompletePolicy
from tre_controller.signals.trs import SignalState

if False:  # TYPE_CHECKING guard without importing typing symbol here
    from tre_controller.profiling import TickProfiler


class ClusterViewReader(Protocol):
    def get(self) -> ClusterView | None: ...


class SnapshotReader(Protocol):
    def get(self) -> MetricsSnapshot | None: ...


class DecisionWriter(Protocol):
    def write(self, loop_name: str, snapshot: MetricsSnapshot, result: LoopTickResult) -> None: ...


class FairnessTaskConfig(Protocol):
    fairness_interval_s: float


def run_fairness_tick(
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
    suppress_hot_proactive_probe: bool = True,
    prof: "TickProfiler | None" = None,
) -> LoopTickResult:
    return run_planner_tick(
        snapshot,
        queue=queue,
        registry=registry,
        rescue_due=False,
        fairness_due=True,
        cluster_view=cluster_view,
        active_probe_models=active_probe_models,
        signal_source=signal_source,
        signal_idle_rps_eps=signal_idle_rps_eps,
        safescale=safescale,
        paper_state_cache=paper_state_cache,
        incomplete_policy=incomplete_policy,
        signal_state=signal_state,
        suppress_hot_proactive_probe=suppress_hot_proactive_probe,
        prof=prof,
        loop="fairness",
    )


async def fairness_task(
    snapshot_box: SnapshotReader,
    *,
    queue: PlannerQueue,
    registry: Registry,
    cfg: FairnessTaskConfig,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    cluster_view: ClusterView | None = None,
    cluster_view_box: ClusterViewReader | None = None,
    active_probe_models: set[str] | None = None,
    decision_writer: DecisionWriter | None = None,
    safescale: SafeScaleController | None = None,
    signal_state: SignalState | None = None,
    prof: "TickProfiler | None" = None,
) -> None:
    paper_state_cache = PaperStateCache(max_stale_windows=getattr(cfg, "paper_stale_max_windows", 3))
    while True:
        snapshot = snapshot_box.get()
        if snapshot is not None:
            current_view = _current_cluster_view(cluster_view, cluster_view_box)
            if cluster_view is None and cluster_view_box is not None and current_view is None:
                result = LoopTickResult(submitted=0, events=("cluster_view_unavailable",))
            else:
                result = run_fairness_tick(
                    snapshot,
                    queue=queue,
                    registry=registry,
                    cluster_view=current_view,
                    active_probe_models=active_probe_models,
                    signal_source=getattr(cfg, "signal_source", "zm"),
                    signal_idle_rps_eps=getattr(cfg, "signal_idle_rps_eps", 0.05),
                    safescale=safescale,
                    paper_state_cache=paper_state_cache,
                    incomplete_policy=getattr(cfg, "incomplete_policy", "drop_model"),
                    signal_state=signal_state,
                    suppress_hot_proactive_probe=getattr(cfg, "safescale_suppress_hot_proactive", True),
                    prof=prof,
                )
            if decision_writer is not None:
                if prof is not None:
                    _dw_t0 = time.perf_counter_ns()
                    decision_writer.write("fairness", snapshot, result)
                    prof.record(
                        {
                            "kind": "decision",
                            "loop": "fairness",
                            "ts_ms": prof.now_ms(),
                            "decision_write_ns": time.perf_counter_ns() - _dw_t0,
                        }
                    )
                else:
                    decision_writer.write("fairness", snapshot, result)
        await sleep(cfg.fairness_interval_s)


def _current_cluster_view(
    cluster_view: ClusterView | None,
    cluster_view_box: ClusterViewReader | None,
) -> ClusterView | None:
    if cluster_view is not None:
        return cluster_view
    if cluster_view_box is None:
        return None
    return cluster_view_box.get()

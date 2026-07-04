from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tre_common.registry import Registry
from tre_controller.loops.action_queue import DispatchResult
from tre_controller.loops.decision_snapshot import DecisionSnapshotWriter
from tre_controller.loops.metrics_task import MetricsRefreshResult, SnapshotStore, refresh_metrics_once, SnapshotBox
from tre_controller.loops.rescue_task import DecisionWriter, run_rescue_tick
from tre_controller.loops.tick import LoopTickResult, SafeScaleController
from tre_controller.planning.planner import ClusterView


class OfflineActionQueue(Protocol):
    def inflight_models(self) -> set[str]: ...

    def submit(self, actions) -> object: ...

    async def drain_once(self) -> tuple[DispatchResult, ...]: ...


@dataclass(frozen=True)
class OfflineIntegrationStepResult:
    metrics: MetricsRefreshResult
    decision: LoopTickResult
    dispatches: tuple[DispatchResult, ...]


async def run_offline_integration_step(
    *,
    store: SnapshotStore,
    queue: OfflineActionQueue,
    decision_writer: DecisionWriter | DecisionSnapshotWriter,
    registry: Registry,
    now_ms: int,
    window_ms: int,
    cluster_view: ClusterView | None = None,
    active_probe_models: set[str] | None = None,
    signal_source: str = "zm",
    safescale: SafeScaleController | None = None,
) -> OfflineIntegrationStepResult:
    snapshot_box = SnapshotBox()
    metrics = refresh_metrics_once(store, snapshot_box, now_ms=now_ms, window_ms=window_ms)
    decision = run_rescue_tick(
        metrics.snapshot,
        queue=queue,
        registry=registry,
        cluster_view=cluster_view,
        active_probe_models=active_probe_models,
        signal_source=signal_source,
        safescale=safescale,
    )
    decision_writer.write("rescue", metrics.snapshot, decision)
    dispatches = await queue.drain_once()
    return OfflineIntegrationStepResult(metrics=metrics, decision=decision, dispatches=dispatches)

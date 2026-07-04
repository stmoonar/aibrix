from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from tre_common.metrics_schema import MetricsSnapshot
from tre_common.registry import Registry
from tre_controller.loops.tick import LoopTickResult, SafeScaleController, run_planner_tick
from tre_controller.planning.planner import Action, ClusterView


class _ReplaySafeScaleController(SafeScaleController, Protocol):
    def active_probes(self) -> tuple[object, ...]: ...


@dataclass(frozen=True)
class TickReplayStep:
    snapshot: MetricsSnapshot
    rescue_due: bool
    fairness_due: bool
    cluster_view: ClusterView | None = None
    active_probe_models: frozenset[str] | None = None


@dataclass(frozen=True)
class TickReplayResult:
    results: tuple[LoopTickResult, ...]
    actions: tuple[Action, ...]
    events: tuple[str, ...]


@dataclass
class ReplayQueue:
    submitted: list[tuple[Action, ...]] = field(default_factory=list)

    def inflight_models(self) -> set[str]:
        # The replay harness models a mock service manager that completes each tick immediately.
        return set()

    def submit(self, actions) -> object:
        self.submitted.append(tuple(actions))
        return object()


def run_tick_replay(
    steps: tuple[TickReplayStep, ...] | list[TickReplayStep],
    *,
    registry: Registry,
    safescale: SafeScaleController | None = None,
    signal_source: str = "zm",
) -> TickReplayResult:
    queue = ReplayQueue()
    results: list[LoopTickResult] = []
    actions: list[Action] = []
    events: list[str] = []

    for step in steps:
        active_probe_models = step.active_probe_models
        if active_probe_models is None:
            active_probe_models = frozenset(_active_probe_models(safescale))
        result = run_planner_tick(
            step.snapshot,
            queue=queue,
            registry=registry,
            rescue_due=step.rescue_due,
            fairness_due=step.fairness_due,
            cluster_view=step.cluster_view,
            active_probe_models=set(active_probe_models),
            signal_source=signal_source,
            safescale=safescale,
        )
        results.append(result)
        actions.extend(result.actions)
        events.extend(result.events)

    return TickReplayResult(results=tuple(results), actions=tuple(actions), events=tuple(events))


def _active_probe_models(safescale: SafeScaleController | None) -> tuple[str, ...]:
    if safescale is None or not hasattr(safescale, "active_probes"):
        return ()
    models: list[str] = []
    for probe in safescale.active_probes():
        model = getattr(probe, "model", None)
        if model is not None:
            models.append(str(model))
    return tuple(models)

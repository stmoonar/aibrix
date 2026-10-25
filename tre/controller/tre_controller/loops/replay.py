from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from tre_common.metrics_schema import MetricsSnapshot
from tre_common.registry import Registry
from tre_controller.loops.tick import LoopTickResult, PaperStateCache, SafeScaleController, run_planner_tick
from tre_controller.planning.planner import Action, ClusterView
from tre_controller.signals.trs import SignalState


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
    suppress_hot_proactive_probe: bool = False,
    signal_state: SignalState | None = None,
    paper_state_cache: PaperStateCache | None = None,
    queue: object | None = None,
    safescale_scope: str = "all",
) -> TickReplayResult:
    """``safescale`` None = no SafeScale: every shrink that would start a probe is
    released immediately (urgent donor path, reason suffix ``_nosafescale``; since
    2026-10-08, before it was dropped).

    ``signal_state`` / ``paper_state_cache`` (round 4): the controller's stateful
    signal path - the O1 breakpoint window, the EMAs, the saturation tracker, the
    missing-token fallback - kept across steps (and across calls, for a closed-loop
    replay that calls this once per tick); None = stateless, as before. ``queue``: where
    the actions go (default: a :class:`ReplayQueue` that records them).
    ``safescale_scope``: TRE_SAFESCALE_SCOPE, as the live loops pass it."""
    queue = queue if queue is not None else ReplayQueue()
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
            suppress_hot_proactive_probe=suppress_hot_proactive_probe,
            safescale=safescale,
            signal_state=signal_state,
            paper_state_cache=paper_state_cache,
            safescale_scope=safescale_scope,
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

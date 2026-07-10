from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import Registry
from tre_controller.planning.planner import Action, ScaleAction, UnhideAction
from tre_controller.planning.safescale import ProbeObservation, SafeScaleCommand, SafeScaleProbe
from tre_controller.signals.sources import get_signal
from tre_controller.signals.trs import SignalState, TRSComputer, TRSInput


class SnapshotReader(Protocol):
    def get(self) -> MetricsSnapshot | None: ...


class SafeScaleObserver(Protocol):
    def active_probes(self) -> tuple[SafeScaleProbe, ...]: ...

    def observe(self, model: str, observation: ProbeObservation, *, now_ms: int): ...

    def resolve(self, model: str, *, status: str, reason: str, now_ms: int) -> bool: ...


class PlannerQueue(Protocol):
    def submit(self, actions) -> object: ...


class SafeScaleTaskConfig(Protocol):
    safescale: object


@dataclass(frozen=True)
class SafeScaleObservationResult:
    submitted: int
    actions: tuple[Action, ...] = ()
    events: tuple[str, ...] = ()


def run_safescale_observation_tick(
    snapshot: MetricsSnapshot,
    *,
    queue: PlannerQueue,
    registry: Registry,
    safescale: SafeScaleObserver,
    signal_source: str = "zm",
    signal_state: SignalState | None = None,
) -> SafeScaleObservationResult:
    if snapshot.stale:
        return SafeScaleObservationResult(submitted=0, events=("snapshot_stale",))

    accepted_actions: list[Action] = []
    events: list[str] = []
    submitted = 0
    for probe in safescale.active_probes():
        metrics = snapshot.models.get(probe.model)
        if metrics is None:
            events.append(f"safescale_observation_missing:{probe.model}")
            continue
        observation = _observation_from_metrics(
            snapshot.ts_ms, metrics, registry.model(probe.model), signal_source, signal_state=signal_state
        )
        decision = safescale.observe(probe.model, observation, now_ms=snapshot.ts_ms)
        events.append(f"safescale_{decision.reason}:{probe.model}")
        actions = _commands_to_actions(decision.commands)
        if not actions:
            continue
        try:
            submit_result = queue.submit(actions)
        except Exception as exc:
            events.append(f"safescale_enqueue_failed:{probe.model}:{type(exc).__name__}")
            continue
        accepted = int(getattr(submit_result, "accepted", 0))
        if accepted != len(actions):
            events.append(f"safescale_enqueue_rejected:{probe.model}")
            continue
        if not safescale.resolve(
            probe.model,
            status=decision.status,
            reason=decision.reason,
            now_ms=snapshot.ts_ms,
        ):
            events.append(f"safescale_resolve_missing:{probe.model}")
            continue
        accepted_actions.extend(actions)
        submitted += accepted

    return SafeScaleObservationResult(
        submitted=submitted,
        actions=tuple(accepted_actions),
        events=tuple(events),
    )

async def safescale_task(
    snapshot_box: SnapshotReader,
    *,
    queue: PlannerQueue,
    registry: Registry,
    safescale: SafeScaleObserver,
    cfg: SafeScaleTaskConfig,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    signal_state: SignalState | None = None,
) -> None:
    while True:
        snapshot = snapshot_box.get()
        if snapshot is not None:
            run_safescale_observation_tick(
                snapshot,
                queue=queue,
                registry=registry,
                safescale=safescale,
                signal_source=getattr(cfg, "signal_source", "zm"),
                signal_state=signal_state,
            )
        interval = getattr(getattr(cfg, "safescale"), "probe_poll_seconds")
        await sleep(interval)


def _observation_from_metrics(
    ts_ms: int,
    metrics: ModelWindowMetrics,
    spec,
    signal_source: str,
    signal_state: SignalState | None = None,
) -> ProbeObservation:
    if signal_state is not None:
        computer = signal_state.computer_for(
            spec.name, ema_alpha=spec.trs.ema_alpha, ema_tau_ms=spec.trs.ema_tau_ms
        )
    else:
        computer = TRSComputer(ema_alpha=spec.trs.ema_alpha, ema_tau_ms=spec.trs.ema_tau_ms)
    result = computer.compute(
        TRSInput.from_metrics(metrics, spec.trs),
        theta_m=spec.trs.theta_m,
        window_end_ms=metrics.window_end_ms,
    )
    signal = get_signal(metrics, spec, signal_source, trs_z_m=result.Z_m)
    return ProbeObservation(
        ts_ms=ts_ms,
        ttft_p95_ms=metrics.ttft_p95_ms,
        tpot_p95_ms=metrics.tpot_p95_ms,
        z_m=signal.z_m,
        q_ctl=result.Q_ctl,
        has_traffic=(result.Y_m > 0.0 or result.Q > 0.0),
        avg_gpu_cache_norm=None,
    )


def _commands_to_actions(commands: tuple[SafeScaleCommand, ...]) -> tuple[Action, ...]:
    actions: list[Action] = []
    for command in commands:
        if command.kind == "unhide":
            actions.append(UnhideAction(command.model, command.pods, command.reason, "safescale"))
        elif command.kind in {"scale_down", "scale_up"}:
            actions.append(ScaleAction(command.model, command.delta, command.reason, "safescale"))
    return tuple(actions)

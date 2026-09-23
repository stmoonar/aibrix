from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import Registry
from tre_controller.loops.tick import serving_window
from tre_controller.planning.planner import Action, ClusterView, ScaleAction, UnhideAction
from tre_controller.planning.safescale import ProbeObservation, SafeScaleCommand, SafeScaleProbe
from tre_controller.signals.sources import get_signal
from tre_controller.signals.trs import SignalState, TRSComputer, TRSInput


class SnapshotReader(Protocol):
    def get(self) -> MetricsSnapshot | None: ...


class ClusterViewReader(Protocol):
    def get(self) -> ClusterView | None: ...


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
    cluster_view: ClusterView | None = None,
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
        # Same serving-pod window as the planner tick (sleeping pods' docs and count out).
        metrics = serving_window(metrics, cluster_view)
        observation = _observation_from_metrics(
            snapshot.ts_ms,
            metrics,
            registry.model(probe.model),
            signal_source,
            signal_state=signal_state,
            hidden_pods=tuple(getattr(probe, "pods", ())),
        )
        decision = safescale.observe(probe.model, observation, now_ms=snapshot.ts_ms)
        events.append(f"safescale_{decision.reason}:{probe.model}")
        gate_failures = _gate_failures(safescale, probe.model, decision)
        if gate_failures:
            events.append(f"safescale_gate_failures:{probe.model}:{','.join(gate_failures)}")
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
    cluster_view_box: ClusterViewReader | None = None,
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
                cluster_view=cluster_view_box.get() if cluster_view_box is not None else None,
            )
        interval = getattr(getattr(cfg, "safescale"), "probe_poll_seconds")
        await sleep(interval)


def _gate_failures(safescale: SafeScaleObserver, model: str, decision) -> tuple[str, ...]:
    if getattr(decision, "reason", "") != "formal_commit_gate_failed":
        return ()
    active = getattr(safescale, "active_probe", None)
    probe = active(model) if callable(active) else None
    details = getattr(probe, "terminal_details", None) or {}
    return tuple(details.get("gate_failures") or ())


def remaining_pods_kv_cache(metrics: ModelWindowMetrics, hidden_pods: tuple[str, ...] = ()) -> float | None:
    """A12: mean KV-cache fill (0..1) over the donor's pods that still serve - the
    window's pods (sleeping ones already dropped by serving_window) minus the probe's
    hidden pods. v1 avg_gpu_cache_norm = sum of per-pod averages / routable pods."""
    hidden = set(hidden_pods)
    values = [
        float(pod.gpu_cache_usage)
        for key, pod in metrics.per_pod.items()
        if key not in hidden and pod.pod not in hidden and getattr(pod, "gpu_cache_usage", None) is not None
    ]
    return sum(values) / len(values) if values else None


def _observation_from_metrics(
    ts_ms: int,
    metrics: ModelWindowMetrics,
    spec,
    signal_source: str,
    signal_state: SignalState | None = None,
    hidden_pods: tuple[str, ...] = (),
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
    signal = get_signal(metrics, spec, signal_source, trs_z_m=result.Z_m, signal_state=signal_state)
    return ProbeObservation(
        ts_ms=ts_ms,
        ttft_p95_ms=metrics.ttft_p95_ms,
        tpot_p95_ms=metrics.tpot_p95_ms,
        z_m=signal.z_m,
        q_ctl=result.Q_ctl,
        # Idle rule (plan 6.4): tokens with nothing in flight is surplus, not traffic
        # that must prove a Z - otherwise a lightly used model could never commit a probe.
        has_traffic=(result.Q > 0.0 or (result.Y_m > 0.0 and result.defined)),
        # A12: was hard-wired None, which disabled the KV-cache commit guard.
        avg_gpu_cache_norm=remaining_pods_kv_cache(metrics, hidden_pods),
    )


def _commands_to_actions(commands: tuple[SafeScaleCommand, ...]) -> tuple[Action, ...]:
    actions: list[Action] = []
    for command in commands:
        if command.kind == "unhide":
            actions.append(UnhideAction(command.model, command.pods, command.reason, "safescale"))
        elif command.kind in {"scale_down", "scale_up"}:
            pods = command.pods if command.kind == "scale_down" else ()
            actions.append(ScaleAction(command.model, command.delta, command.reason, "safescale", pods=pods))
    return tuple(actions)

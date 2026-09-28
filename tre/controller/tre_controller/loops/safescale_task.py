from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Mapping, Protocol

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import Registry
from tre_controller.gateway_health import GatewayCounters
from tre_controller.loops.tick import serving_window
from tre_controller.planning.planner import (
    Action,
    ClusterView,
    ReceiverTarget,
    SafeScaleCommitAction,
    UnhideAction,
)
from tre_controller.planning.safescale import ProbeObservation, SafeScaleCommand, SafeScaleProbe
from tre_controller.signals.sources import get_signal
from tre_controller.signals.trs import SignalState, TRSComputer, TRSInput


LOG = logging.getLogger("tre_controller.safescale")


class GatewayCounterSource(Protocol):
    def read(self) -> Mapping[str, GatewayCounters] | None: ...


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
    gateway_counters: Mapping[str, GatewayCounters] | None = None,
    fresh_cluster_view: ClusterView | None = None,
    recovery_needs_fresh_view: bool = False,
) -> SafeScaleObservationResult:
    """One SafeScale observation tick. ``fresh_cluster_view`` is the SM view only
    while fresh (``ClusterViewBox.fresh``): with it, probes whose pods are all
    gone are resolved first, in any controller mode (B8). With
    ``recovery_needs_fresh_view`` (app wiring) a ``committing`` probe is only
    re-submitted once such a view exists, so a probe restored after a restart
    is checked for gone pods before anything reaches the SM."""
    events: list[str] = []
    # B8: before the stale-snapshot return - it needs only the cluster view.
    _resolve_probes_with_gone_pods(
        queue, safescale, fresh_cluster_view, now_ms=snapshot.ts_ms, events=events
    )
    if snapshot.stale:
        return SafeScaleObservationResult(submitted=0, events=tuple(events) + ("snapshot_stale",))

    accepted_actions: list[Action] = []
    submitted = 0
    if recovery_needs_fresh_view and fresh_cluster_view is None:
        waiting = getattr(safescale, "committing_probes", None)
        has_request = getattr(queue, "has_request", None)
        for probe in waiting() if callable(waiting) else ():
            if not (callable(has_request) and has_request(probe.request_id)):
                events.append(f"safescale_recovery_waits_for_fresh_view:{probe.model}")
    else:
        recovered = _recover_committing(queue, safescale, now_ms=snapshot.ts_ms, events=events)
        accepted_actions.extend(recovered)
        submitted += len(recovered)
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
            gateway=(gateway_counters or {}).get(probe.model),
        )
        decision = safescale.observe(probe.model, observation, now_ms=snapshot.ts_ms)
        events.append(f"safescale_{decision.reason}:{probe.model}")
        gate_failures = _gate_failures(safescale, probe.model, decision)
        if gate_failures:
            events.append(f"safescale_gate_failures:{probe.model}:{','.join(gate_failures)}")
        if getattr(decision, "reason", "") in ("formal_commit_gate_passed", "formal_commit_gate_failed") and (
            _terminal_details(safescale, probe.model).get("kv_cache") == "unavailable"
        ):
            # P2-a: the KV-cache check could not be evaluated (fail-open, as v1) - say so.
            events.append(f"safescale_kv_cache_unavailable:{probe.model}")
        if getattr(decision, "reason", "") == "donor_health":
            health = _terminal_details(safescale, probe.model).get("donor_health") or {}
            events.append(
                f"safescale_donor_health:{probe.model}:errors={health.get('errors', 0):.0f}"
                f":requests={health.get('requests', 0):.0f}:rate={health.get('error_rate', 0.0):.4f}"
            )
        actions = _commands_to_actions(
            decision.commands,
            cluster_view=cluster_view,
            registry=registry,
            request_id=_probe_request_id(safescale, probe.model),
            # = the committing_ts mark_committing records below (B8 max age).
            decided_ms=snapshot.ts_ms,
        )
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
        # Durable lifecycle (review 4 P2-4): with a queue that reports when it
        # finished the probe's one-shot action, the probe is only marked
        # ``committing`` here and resolved by the queue; a restart re-submits it.
        mark = getattr(safescale, "mark_committing", None)
        if callable(mark) and callable(getattr(queue, "has_request", None)):
            marked = mark(probe.model, status=decision.status, reason=decision.reason, now_ms=snapshot.ts_ms)
        else:
            marked = safescale.resolve(
                probe.model,
                status=decision.status,
                reason=decision.reason,
                now_ms=snapshot.ts_ms,
            )
        if not marked:
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
    gateway_source: GatewayCounterSource | None = None,
) -> None:
    while True:
        snapshot = snapshot_box.get()
        if snapshot is not None:
            counters = None
            if gateway_source is not None and safescale.active_probes():
                # A13: the donor's gateway counters, read off the event loop (HTTP).
                counters = await asyncio.to_thread(gateway_source.read)
            result = run_safescale_observation_tick(
                snapshot,
                queue=queue,
                registry=registry,
                safescale=safescale,
                signal_source=getattr(cfg, "signal_source", "zm"),
                signal_state=signal_state,
                cluster_view=cluster_view_box.get() if cluster_view_box is not None else None,
                gateway_counters=counters,
                fresh_cluster_view=_fresh_view(cluster_view_box),
                recovery_needs_fresh_view=cluster_view_box is not None,
            )
            _log_resolutions(snapshot.ts_ms, result, gateway_available=counters is not None)
        interval = getattr(getattr(cfg, "safescale"), "probe_poll_seconds")
        await sleep(interval)


def _fresh_view(cluster_view_box) -> ClusterView | None:
    fresh = getattr(cluster_view_box, "fresh", None)
    if not callable(fresh):
        return None
    try:
        return fresh()
    except Exception:  # noqa: BLE001 - no view = nothing is judged gone
        return None


def _resolve_probes_with_gone_pods(
    queue, safescale, view: ClusterView | None, *, now_ms: int, events: list[str]
) -> None:
    """B8: a probe (probing or committing) whose pods are ALL absent from a FRESH
    cluster view - deleted / replaced, e.g. by a rolling restart - can never be
    acted on: it is resolved as a rollback ``probe_pods_gone`` without any SM
    call, in every controller mode, and its queued one-shot action (held in
    observe mode, say) is dropped. Guarded: only with a fresh view (never a
    missing or stale one), and only when that view lists at least one binding of
    the probe's model (an empty / partial SM state never resolves a probe). A
    probe whose one-shot action is already running is left to finish it."""
    if view is None:
        return
    all_probes = getattr(safescale, "all_probes", None)
    resolve_request = getattr(safescale, "resolve_request", None)
    if not callable(all_probes) or not callable(resolve_request):
        return
    bindings = tuple(getattr(view, "bindings", ()) or ())
    serve_ids = {binding.serve_id for binding in bindings}
    models = {binding.model for binding in bindings}
    for probe in all_probes():
        pods = tuple(getattr(probe, "pods", ()) or ())
        if not pods or probe.model not in models or any(pod in serve_ids for pod in pods):
            continue
        cancel = getattr(queue, "cancel_request", None)
        if callable(cancel) and not cancel(probe.request_id):
            events.append(f"safescale_probe_pods_gone_deferred:{probe.model}:{probe.request_id}")
            continue
        if not resolve_request(probe.request_id, status="rollback", reason="probe_pods_gone", now_ms=now_ms):
            continue
        events.append(f"safescale_probe_pods_gone:{probe.model}:{probe.request_id}")
        LOG.warning(
            json.dumps(
                {"event": "safescale_probe_pods_gone", "model": probe.model, "request_id": probe.request_id,
                 "status": probe.status, "pods": list(pods), "resolution": "rollback"},
                sort_keys=True,
            )
        )


def _recover_committing(queue, safescale, *, now_ms: int, events: list[str]) -> list[Action]:
    """Re-submit the decision of every ``committing`` probe the queue does not
    hold (review 4 P2-4): after a controller restart, or when its action was
    lost to an unexpected error. Bounded: after a few recoveries the probe is
    resolved as a rollback (its pods are then left to the orphan detector)."""
    committing = getattr(safescale, "committing_probes", None)
    has_request = getattr(queue, "has_request", None)
    if not callable(committing) or not callable(has_request):
        return []
    submitted: list[Action] = []
    for probe in committing():
        if has_request(probe.request_id):
            continue
        if not safescale.note_recovery(probe.request_id):
            safescale.resolve_request(
                probe.request_id, status="rollback", reason="recovery_exhausted", now_ms=now_ms
            )
            events.append(f"safescale_recovery_exhausted:{probe.model}")
            continue
        actions = recovered_actions(probe)
        try:
            result = queue.submit(actions)
        except Exception as exc:  # noqa: BLE001 - retried next tick
            events.append(f"safescale_recovery_enqueue_failed:{probe.model}:{type(exc).__name__}")
            continue
        if int(getattr(result, "accepted", 0)) == len(actions):
            events.append(f"safescale_committing_recovered:{probe.model}:{probe.resolution}")
            submitted.extend(actions)
        else:
            events.append(f"safescale_recovery_deferred:{probe.model}")
    return submitted


def _log_resolutions(ts_ms: int, result: SafeScaleObservationResult, *, gateway_available: bool) -> None:
    """One JSON log line per tick that did more than wait (commit / rollback / errors)."""
    notable = [event for event in result.events if not event.startswith("safescale_probe_pending:")]
    if not notable:
        return
    LOG.info(
        json.dumps(
            {
                "event": "safescale_observation",
                "ts_ms": ts_ms,
                "events": notable,
                "submitted": result.submitted,
                "gateway_counters": gateway_available,
            },
            separators=(",", ":"),
        )
    )


def _terminal_details(safescale: SafeScaleObserver, model: str) -> dict:
    active = getattr(safescale, "active_probe", None)
    probe = active(model) if callable(active) else None
    return getattr(probe, "terminal_details", None) or {}


def _gate_failures(safescale: SafeScaleObserver, model: str, decision) -> tuple[str, ...]:
    if getattr(decision, "reason", "") != "formal_commit_gate_failed":
        return ()
    return tuple(_terminal_details(safescale, model).get("gate_failures") or ())


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
    gateway: GatewayCounters | None = None,
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
        gateway_requests=gateway.requests if gateway is not None else None,
        gateway_errors=gateway.errors if gateway is not None else None,
    )


def _probe_request_id(safescale: SafeScaleObserver, model: str) -> str | None:
    active = getattr(safescale, "active_probe", None)
    probe = active(model) if callable(active) else None
    return getattr(probe, "request_id", None)


def _commands_to_actions(
    commands: tuple[SafeScaleCommand, ...],
    *,
    cluster_view: ClusterView | None = None,
    registry: Registry | None = None,
    request_id: str | None = None,
    decided_ms: int | None = None,
) -> tuple[Action, ...]:
    """A rollback is one unhide; a commit batch is ONE :class:`SafeScaleCommitAction`
    (review 3 P2-2): the hidden donor pods sleep first, then the receivers are
    brought up to ABSOLUTE targets. The target is NOT computed here from the
    cluster view (refreshed every ~10 s, so possibly stale - review 4 P2-1): the
    queue resolves it once from the SM's awake count at the first dispatch
    (current + delta, capped at the scaling cap carried here) and freezes it, so
    a retry is still idempotent. ``cluster_view`` is kept for callers."""
    del cluster_view
    actions: list[Action] = []
    donor = next((command for command in commands if command.kind == "scale_down"), None)
    upscales = tuple(
        ReceiverTarget(command.model, int(command.delta), None, _scaling_cap(command.model, registry))
        for command in commands
        if command.kind == "scale_up" and command.delta > 0
    )
    for command in commands:
        if command.kind == "unhide":
            actions.append(
                UnhideAction(command.model, command.pods, command.reason, "safescale", request_id=request_id)
            )
    if donor is not None:
        actions.append(
            SafeScaleCommitAction(
                donor=donor.model,
                pods=donor.pods,
                reason=donor.reason,
                upscales=upscales,
                drain_budget_s=donor.drain_budget_s,
                request_id=request_id,
                decided_ms=decided_ms,
            )
        )
    elif upscales:
        actions.append(
            SafeScaleCommitAction(
                donor=upscales[0].model,
                pods=(),
                reason="safescale_followup_upscale",
                upscales=upscales,
                request_id=request_id,
                donor_done=True,
                decided_ms=decided_ms,
            )
        )
    return tuple(actions)


def _scaling_cap(model: str, registry: Registry | None) -> int | None:
    """The receiver's scaling cap (registry max_awake_replicas), or None."""
    if registry is None:
        return None
    try:
        return int(registry.model(model).scale_max_replicas)
    except (KeyError, ValueError, AttributeError, TypeError):
        return None


def recovered_actions(probe) -> tuple[Action, ...]:
    """Actions that finish a probe found ``committing`` after a controller
    restart (review 4 P2-4): its recorded resolution is submitted again and
    revalidated at dispatch like the original - a rollback is the unhide, a
    commit sleeps the hidden donor pods (a fresh cluster view showing them
    asleep makes it a no-op). The follow-up upscales are not re-sent (their
    frozen targets died with the old process; the planner re-plans receivers
    every tick)."""
    if getattr(probe, "resolution", None) == "commit":
        return (
            SafeScaleCommitAction(
                donor=probe.model,
                pods=tuple(probe.pods),
                reason=f"recovered:{probe.resolution_reason or 'commit'}",
                drain_budget_s=(float(probe.window_ms) / 1000.0 if probe.window_ms else None),
                request_id=probe.request_id,
                # B8: aged from the original decision, not from the re-submission.
                decided_ms=getattr(probe, "committing_ms", None),
            ),
        )
    return (
        UnhideAction(
            probe.model,
            tuple(probe.pods),
            f"recovered:{probe.resolution_reason or 'rollback'}",
            "safescale",
            request_id=probe.request_id,
        ),
    )

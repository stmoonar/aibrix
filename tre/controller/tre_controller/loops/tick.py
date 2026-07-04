from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import Registry, ModelSpec
from tre_controller.planning.classify import classify_all_models
from tre_controller.planning.planner import Action, ClusterView, HideAction, PlanConfig, ScaleAction, ShrinkForSlotAction, UnhideAction, build_plan
from tre_controller.planning.safescale import SafeScaleCommand, SafeScaleDecision
from tre_controller.signals.sources import get_signal
from tre_controller.signals.trs import TRSComputer, TRSInput


class PlannerQueue(Protocol):
    def inflight_models(self) -> set[str]: ...

    def submit(self, actions) -> object: ...


class SafeScaleController(Protocol):
    def start_probe(
        self,
        *,
        model: str,
        pods: tuple[str, ...],
        now_ms: int,
        pending_upscales: dict[str, int] | None = None,
    ) -> SafeScaleDecision: ...


@dataclass(frozen=True)
class LoopTickResult:
    submitted: int
    actions: tuple[Action, ...] = ()
    events: tuple[str, ...] = ()


def run_planner_tick(
    snapshot: MetricsSnapshot,
    *,
    queue: PlannerQueue,
    registry: Registry,
    rescue_due: bool,
    fairness_due: bool,
    cluster_view: ClusterView | None = None,
    active_probe_models: set[str] | None = None,
    signal_source: str = "zm",
    safescale: SafeScaleController | None = None,
) -> LoopTickResult:
    if snapshot.stale:
        return LoopTickResult(submitted=0, events=("snapshot_stale",))

    contexts = _model_contexts(snapshot, registry, signal_source=signal_source)
    classifications = classify_all_models(contexts)
    replicas = {model: int(ctx.get("assigned_replicas", 0)) for model, ctx in contexts.items()}
    cfg = PlanConfig(
        min_replicas_per_model=min((spec.min_replicas for spec in registry.models()), default=0),
        max_replicas_per_model=max((spec.max_replicas for spec in registry.models()), default=0),
        rescue_due=rescue_due,
        fairness_due=fairness_due,
        model_tp_sizes={spec.name: spec.tp_size for spec in registry.models()},
    )
    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas=replicas,
        idle_gpus=_idle_gpus(snapshot, registry),
        cfg=cfg,
        active_probe_models=active_probe_models or set(),
        inflight_models=queue.inflight_models(),
        cluster_view=cluster_view,
    )
    actions, safescale_events = _apply_safescale(snapshot, tuple(plan.actions), plan.probe_upscale_plans, safescale=safescale)
    if actions:
        queue.submit(actions)
    return LoopTickResult(submitted=len(actions), actions=actions, events=tuple(plan.events) + safescale_events)


def _apply_safescale(
    snapshot: MetricsSnapshot,
    actions: tuple[Action, ...],
    probe_upscale_plans: dict[str, dict[str, int]],
    *,
    safescale: SafeScaleController | None,
) -> tuple[tuple[Action, ...], tuple[str, ...]]:
    if safescale is None:
        return actions, ()

    converted: list[Action] = []
    events: list[str] = []
    for action in actions:
        if not _requires_safescale_probe(action):
            converted.append(action)
            continue

        probe_model = _safescale_probe_model(action)
        pods = _safescale_probe_pods(snapshot, action)
        decision = safescale.start_probe(
            model=probe_model,
            pods=pods,
            now_ms=snapshot.ts_ms,
            pending_upscales=_safescale_pending_upscales(action, probe_upscale_plans),
        )
        if decision.status == "none":
            events.append(f"safescale_probe_skipped:{probe_model}:{decision.reason}")
            continue
        events.append(f"safescale_{decision.reason}:{probe_model}")
        converted.extend(_commands_to_actions(decision.commands, source_loop=action.source_loop))
    return tuple(converted), tuple(events)


def _requires_safescale_probe(action: Action) -> bool:
    return (isinstance(action, ScaleAction) and action.requires_safescale and action.delta < 0) or isinstance(
        action, ShrinkForSlotAction
    )


def _safescale_probe_model(action: Action) -> str:
    if isinstance(action, ShrinkForSlotAction):
        return action.donor
    return action.model


def _safescale_probe_pods(snapshot: MetricsSnapshot, action: Action) -> tuple[str, ...]:
    if isinstance(action, ShrinkForSlotAction):
        return (action.serve_id,)
    return _pods_to_probe(snapshot, action.model, abs(action.delta))


def _safescale_pending_upscales(
    action: Action,
    probe_upscale_plans: dict[str, dict[str, int]],
) -> dict[str, int]:
    if isinstance(action, ShrinkForSlotAction):
        return {action.beneficiary: 1}
    return probe_upscale_plans.get(action.model, {})


def _pods_to_probe(snapshot: MetricsSnapshot, model: str, count: int) -> tuple[str, ...]:
    metrics = snapshot.models.get(model)
    if metrics is None or count <= 0:
        return ()
    if metrics.per_pod:
        pods = sorted({pod.pod for pod in metrics.per_pod.values() if pod.pod})
    else:
        pods = []
    return tuple(pods[:count])


def _commands_to_actions(commands: tuple[SafeScaleCommand, ...], *, source_loop: str) -> tuple[Action, ...]:
    actions: list[Action] = []
    for command in commands:
        if command.kind == "hide":
            actions.append(HideAction(command.model, command.pods, command.reason, source_loop))
        elif command.kind == "unhide":
            actions.append(UnhideAction(command.model, command.pods, command.reason, source_loop))
        elif command.kind in {"scale_down", "scale_up"}:
            actions.append(ScaleAction(command.model, command.delta, command.reason, source_loop))
    return tuple(actions)


def _model_contexts(snapshot: MetricsSnapshot, registry: Registry, *, signal_source: str = "zm") -> dict[str, dict]:
    contexts: dict[str, dict] = {}
    for model_name, metrics in snapshot.models.items():
        spec = registry.model(model_name)
        result = TRSComputer(ema_alpha=spec.trs.ema_alpha).compute(TRSInput.from_metrics(metrics, spec.trs), theta_m=spec.trs.theta_m)
        signal = get_signal(metrics, spec, signal_source, trs_z_m=result.Z_m)
        contexts[model_name] = {
            "trs": result.TRS,
            "z_m": signal.z_m,
            "signal_source": signal.source,
            "signal_raw_value": signal.raw_value,
            "signal_unavailable_reason": signal.unavailable_reason,
            "trs_z_m": result.Z_m,
            "eta_m": result.eta_m,
            "theta_m": spec.trs.theta_m,
            "Q": result.Q,
            "Q_ctl": result.Q_ctl,
            "Y_m": result.Y_m,
            "y_m": result.y_m,
            "routable_pods": metrics.routable_pods,
            "assigned_replicas": metrics.assigned_replicas,
            "is_saturated": result.Q_ctl >= spec.trs.qsat,
        }
    return contexts


def _idle_gpus(snapshot: MetricsSnapshot, registry: Registry) -> int:
    total_gpus = sum(node.gpus for node in registry.topology().nodes)
    used_gpus = 0
    for model_name, metrics in snapshot.models.items():
        try:
            spec: ModelSpec = registry.model(model_name)
            used_gpus += max(0, metrics.assigned_replicas) * spec.tp_size
        except KeyError:
            continue
    return max(0, total_gpus - used_gpus)

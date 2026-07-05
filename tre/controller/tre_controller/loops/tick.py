from __future__ import annotations

from dataclasses import dataclass, replace
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


@dataclass
class _PaperState:
    context: dict
    stale_windows: int = 0


class PaperStateCache:
    def __init__(self, *, max_stale_windows: int = 3) -> None:
        self.max_stale_windows = max(0, int(max_stale_windows))
        self._by_model: dict[str, _PaperState] = {}

    def apply(self, model_name: str, context: dict, *, tokens_available: bool) -> tuple[dict, tuple[str, ...]]:
        if tokens_available:
            self._by_model[model_name] = _PaperState(dict(context), stale_windows=0)
            return context, ()

        previous = self._by_model.get(model_name)
        if previous is None:
            return context, (f"paper_state_stale_unknown:{model_name}",)
        previous.stale_windows += 1
        if previous.stale_windows > self.max_stale_windows:
            return context, (f"paper_state_stale_unknown:{model_name}",)
        held = dict(previous.context)
        held.update(
            {
                "routable_pods": context.get("routable_pods", held.get("routable_pods", 0)),
                "assigned_replicas": context.get("assigned_replicas", held.get("assigned_replicas", 0)),
            }
        )
        return held, (f"paper_state_stale_hold:{model_name}",)


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
    paper_state_cache: PaperStateCache | None = None,
) -> LoopTickResult:
    if snapshot.stale:
        return LoopTickResult(submitted=0, events=("snapshot_stale",))

    contexts, paper_events = _model_contexts(
        snapshot,
        registry,
        signal_source=signal_source,
        cluster_view=cluster_view,
        paper_state_cache=paper_state_cache,
    )
    classifications = classify_all_models(contexts)
    replicas = {model: int(ctx.get("assigned_replicas", 0)) for model, ctx in contexts.items()}
    cfg = PlanConfig(
        min_replicas_per_model=min((spec.min_replicas for spec in registry.models()), default=0),
        max_replicas_per_model=max((spec.max_replicas for spec in registry.models()), default=0),
        rescue_due=rescue_due,
        fairness_due=fairness_due,
        model_tp_sizes={spec.name: spec.tp_size for spec in registry.models()},
        min_replicas_by_model={spec.name: spec.min_replicas for spec in registry.models()},
        max_replicas_by_model={spec.name: spec.max_replicas for spec in registry.models()},
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
    return LoopTickResult(submitted=len(actions), actions=actions, events=paper_events + tuple(plan.events) + safescale_events)


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


def _model_contexts(
    snapshot: MetricsSnapshot,
    registry: Registry,
    *,
    signal_source: str = "zm",
    cluster_view: ClusterView | None = None,
    paper_state_cache: PaperStateCache | None = None,
) -> tuple[dict[str, dict], tuple[str, ...]]:
    contexts: dict[str, dict] = {}
    events: list[str] = []
    cluster_counts = _cluster_view_counts(cluster_view)
    for model_name, metrics in snapshot.models.items():
        spec = registry.model(model_name)
        counts = cluster_counts.get(model_name)
        assigned_replicas = metrics.assigned_replicas
        if counts is not None:
            awake_replicas, bound_replicas = counts
            assigned_replicas = bound_replicas
            metrics = replace(metrics, routable_pods=awake_replicas, assigned_replicas=awake_replicas)
        tokens_available = metrics.prompt_tokens is not None and metrics.generation_tokens is not None
        if tokens_available:
            result = TRSComputer(ema_alpha=spec.trs.ema_alpha).compute(TRSInput.from_metrics(metrics, spec.trs), theta_m=spec.trs.theta_m)
            signal = get_signal(metrics, spec, signal_source, trs_z_m=result.Z_m)
            context = {
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
                "assigned_replicas": assigned_replicas,
                "is_saturated": result.Q_ctl >= spec.trs.qsat,
            }
        else:
            context = {
                "trs": 0.0,
                "z_m": None,
                "signal_source": signal_source,
                "signal_raw_value": None,
                "signal_unavailable_reason": "tokens_missing",
                "trs_z_m": None,
                "eta_m": None,
                "theta_m": spec.trs.theta_m,
                "Q": metrics.avg_waiting * spec.trs.lambda_wait + metrics.avg_running + metrics.avg_swapping,
                "Q_ctl": max(metrics.avg_waiting * spec.trs.lambda_wait + metrics.avg_running + metrics.avg_swapping, spec.trs.qmin),
                "Y_m": None,
                "y_m": None,
                "routable_pods": metrics.routable_pods,
                "assigned_replicas": assigned_replicas,
                "is_saturated": False,
            }
        if paper_state_cache is not None:
            context, model_events = paper_state_cache.apply(model_name, context, tokens_available=tokens_available)
            events.extend(model_events)
        contexts[model_name] = context
    return contexts, tuple(events)


def _cluster_view_counts(cluster_view: ClusterView | None) -> dict[str, tuple[int, int]]:
    if cluster_view is None:
        return {}
    counts: dict[str, list[int]] = {}
    for binding in cluster_view.bindings:
        model_counts = counts.setdefault(binding.model, [0, 0])
        if not binding.hidden:
            model_counts[1] += 1
        if binding.awake and not binding.hidden:
            model_counts[0] += 1
    return {model: (values[0], values[1]) for model, values in counts.items()}


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

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Callable, Protocol, Union

if TYPE_CHECKING:
    from tre_controller.profiling import TickProfiler

from tre_common.gpu_placement import placement_policy_from_registry
from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import Registry, ModelSpec
from tre_common.tss import window_is_idle
from tre_common.window_pods import restrict_to_serving
from tre_controller.planning.classify import (
    classify_all_models,
    model_control_configs_from_registry,
)
from tre_controller.planning.planner import (
    Action,
    ClusterView,
    HideAction,
    IncompletePolicy,
    PlanConfig,
    RescueBasis,
    ScaleAction,
    ShrinkForSlotAction,
    UnhideAction,
    build_plan,
)
from tre_controller.planning.safescale import (
    ProbeWindowInputs,
    SafeScaleCommand,
    SafeScaleDecision,
    format_window_event,
)
from tre_controller.signals.sources import get_signal, per_replica_token_rate
from tre_controller.signals.trs import SignalState, TRSComputer, TRSInput
from tre_sm.allocator.slots import natural_key, release_order


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
        window_inputs: ProbeWindowInputs | None = None,
    ) -> SafeScaleDecision: ...


# A static set (tests) or a live provider read every tick (app wiring: the models the
# SafeScaleStateMachine is currently probing), so planner donor paths exclude them.
ActiveProbeModels = Union[set[str], Callable[[], set[str]], None]


def resolve_active_probe_models(source: ActiveProbeModels) -> set[str]:
    if source is None:
        return set()
    if callable(source):
        return set(source())
    return set(source)


@dataclass(frozen=True)
class LoopTickResult:
    submitted: int
    actions: tuple[Action, ...] = ()
    events: tuple[str, ...] = ()
    model_contexts: dict[str, dict] = field(default_factory=dict)
    classifications: dict = field(default_factory=dict)  # model -> ModelClassification (S5.1)


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
                "awake_replicas": context.get("awake_replicas", held.get("awake_replicas")),
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
    signal_idle_rps_eps: float = 0.05,
    safescale: SafeScaleController | None = None,
    paper_state_cache: PaperStateCache | None = None,
    incomplete_policy: IncompletePolicy = "drop_model",
    signal_state: SignalState | None = None,
    suppress_hot_proactive_probe: bool = False,
    disable_eta_gate: bool = False,
    prof: "TickProfiler | None" = None,
    loop: str = "tick",
    action_cooldown: bool = False,
    observe_mode: bool = False,
    probe_block_reason: str | None = None,
) -> LoopTickResult:
    """One planner tick. ``observe_mode`` (controller mode ``observe``, B8): the
    plan is still computed and published, but no SafeScale probe is started or
    preempted - the probe's state would change (hide, window, commit) while its
    hide never reaches the cluster. Every other action is still submitted; the
    ActionQueue drops it in observe mode (it never reaches the SM).

    ``probe_block_reason`` (P2-3): no SafeScale probe is started this tick
    (``sm_maintenance``: the SM maintenance lock is held;
    ``sm_maintenance_unreadable``: it could not be read - fail-closed). The
    planned probe-only scale-downs are dropped with the event
    ``safescale_probe_skipped:<model>:<reason>``; every other action is kept."""
    if snapshot.stale:
        return LoopTickResult(submitted=0, events=("snapshot_stale",))
    if cluster_view is not None and cluster_view.placement is None:
        # Every placement decision of this tick (wakes, creates, donor slots, probe /
        # shrink order) ranks by the registry placement policy.
        cluster_view = replace(cluster_view, placement=placement_policy_from_registry(registry))

    _prof_on = prof is not None
    if _prof_on:
        _tick_t0 = time.perf_counter_ns()
        _ru_u0, _ru_s0 = prof.rusage_ms()
        _phase_t0 = _tick_t0

    contexts, paper_events = _model_contexts(
        snapshot,
        registry,
        signal_source=signal_source,
        cluster_view=cluster_view,
        paper_state_cache=paper_state_cache,
        signal_state=signal_state,
        queue=queue,
    )
    # Without model_control_configs every model silently falls back to the generic
    # delta_crit=0.2 / delta_high=0.25 and the fitted per-model bands in registry.yaml
    # are dead keys. An alternative signal source runs on its own fitted bands.
    classifications = classify_all_models(
        contexts,
        model_control_configs=model_control_configs_from_registry(registry, signal_source),
        signal_idle_rps_eps=signal_idle_rps_eps,
    )
    dwell_events: tuple[str, ...] = ()
    if signal_state is not None:
        # Band dwell (D8): counted per distinct window_end_ms in the shared SignalState,
        # so the rescue/fairness re-reads of one snapshot never advance it twice.
        classifications, dwell_events = signal_state.apply_dwell(classifications, contexts, snapshot.models)
    if _prof_on:
        _signals_ns = time.perf_counter_ns() - _phase_t0
        _phase_t0 = time.perf_counter_ns()
    replicas = {model: int(ctx.get("assigned_replicas", 0)) for model, ctx in contexts.items()}
    cfg = PlanConfig(
        min_replicas_per_model=min((spec.min_replicas for spec in registry.models()), default=0),
        # Scaling cap (max_awake_replicas, v1/paper alignment A1), not the GPU layout size.
        max_replicas_per_model=max((spec.scale_max_replicas for spec in registry.models()), default=0),
        rescue_due=rescue_due,
        fairness_due=fairness_due,
        model_tp_sizes={spec.name: spec.tp_size for spec in registry.models()},
        min_replicas_by_model={spec.name: spec.min_replicas for spec in registry.models()},
        max_replicas_by_model={spec.name: spec.scale_max_replicas for spec in registry.models()},
        incomplete_policy=incomplete_policy,
        suppress_hot_proactive_probe=suppress_hot_proactive_probe,
        disable_eta_gate=disable_eta_gate,
        defrag_enabled=_defrag_enabled(registry),
        **_scaling_options(registry),
    )
    plan = build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas=replicas,
        idle_gpus=_idle_gpus(snapshot, registry, cluster_view),
        cfg=cfg,
        active_probe_models=active_probe_models or set(),
        inflight_models=queue.inflight_models(),
        cluster_view=cluster_view,
        cooldowns=_action_cooldowns(snapshot, queue) if action_cooldown else None,
        # P2-6: independent of TRE_ACTION_COOLDOWN (its own switch is the tick count).
        floor_holds=_floor_held_models(queue),
        # S3: GPUs / nodes the SM recently refused a wake on (placement.wake_cooldown).
        unavailable_gpus=_cooled_gpus(queue, registry),
        refusals=_recent_refusals(queue),
        probe_backoff_models=_probe_backoff_models(safescale, snapshot.ts_ms),
        preemptible_models=_preemptible_models(queue) if rescue_due else None,
        # C1: earlier rescue targets the decision windows do not reflect yet.
        rescue_bases=_rescue_bases(snapshot, queue, registry, contexts) if rescue_due else None,
    )
    if _prof_on:
        _plan_ns = time.perf_counter_ns() - _phase_t0
        _phase_t0 = time.perf_counter_ns()
    covered_targets: dict = {}
    actions, safescale_events = _apply_safescale(
        snapshot,
        tuple(plan.actions),
        plan.probe_upscale_plans,
        safescale=safescale,
        cluster_view=cluster_view,
        contexts=contexts,
        observe_mode=observe_mode,
        probe_block_reason=probe_block_reason,
        covered_targets=covered_targets,
    )
    # C1 review P2-1: a rescue target fully covered by the pods a probe preemption gives
    # back is still a target the window does not reflect yet.
    record_covered = getattr(queue, "record_rescue_covered", None)
    if callable(record_covered):
        for model, rescue in covered_targets.items():
            record_covered(model, rescue)
    if _prof_on:
        _safescale_ns = time.perf_counter_ns() - _phase_t0
        _phase_t0 = time.perf_counter_ns()
    queue_events = _defrag_blocking_events(queue, actions) if rescue_due else ()
    # S3: gpu_cooldown / wake_refused / placement_retry recorded by the queue.
    queue_events = tuple(queue_events) + tuple(_drain_queue_events(queue))
    if actions:
        queue.submit(actions)
    if _prof_on:
        _submit_ns = time.perf_counter_ns() - _phase_t0
        _ru_u1, _ru_s1 = prof.rusage_ms()
        prof.record(
            {
                "kind": "tick",
                "loop": loop,
                "seq": prof.next_seq(loop),
                "ts_ms": prof.now_ms(),
                "n_models": len(contexts),
                "n_pods": sum(int(ctx.get("assigned_replicas", 0)) for ctx in contexts.values()),
                "n_actions": len(actions),
                "signals_ns": _signals_ns,
                "plan_ns": _plan_ns,
                "safescale_ns": _safescale_ns,
                "submit_ns": _submit_ns,
                "tick_total_ns": time.perf_counter_ns() - _tick_t0,
                "cpu_user_ms_delta": _ru_u1 - _ru_u0,
                "cpu_sys_ms_delta": _ru_s1 - _ru_s0,
            }
        )
    return LoopTickResult(
        submitted=len(actions),
        actions=actions,
        events=paper_events + dwell_events + tuple(plan.events) + safescale_events + queue_events,
        model_contexts=contexts,
        classifications={item.model_name: item for item in classifications},
    )


def _defrag_enabled(registry: Registry) -> bool:
    placement = getattr(registry, "placement", None)
    return bool(getattr(placement(), "defrag_enabled", False)) if callable(placement) else False


def _scaling_options(registry: Registry) -> dict:
    """Registry ``scaling:`` (C1) as PlanConfig keywords; a registry without the
    section (tests, older loaders) keeps the PlanConfig defaults."""
    scaling = getattr(registry, "scaling", None)
    if not callable(scaling):
        return {}
    config = scaling()
    return {
        "rescue_max_step_ratio": float(config.rescue_max_step_ratio),
        "scale_up_cooldown_enabled": bool(config.scale_up_cooldown_enabled),
        "rescue_max_step_pods": int(getattr(config, "rescue_max_step_pods", 0)),
        "donor_surplus_release": bool(getattr(config, "donor_surplus_release", False)),
        "partial_window_max_step": int(getattr(config, "breakpoint_partial_max_step", 0) or 0)
        if bool(getattr(config, "breakpoint_window", False))
        else 0,
        "partial_window_lowevidence_requests": int(getattr(config, "breakpoint_lowevidence_requests", 0) or 0),
    }


def rescue_settle_ms(registry: Registry | None, model: str) -> float:
    """C1 review P2-3: how long after a rescue target completed the decision window
    must start before it counts as reflected - ``scaling.rescue_settle_ema_k`` times
    the model's ``trs.ema_tau_ms`` (the EMA'd Z lags the raw window by about its time
    constant). 0 without a registry, a scaling section or a tau (legacy fixed-alpha)."""
    scaling = getattr(registry, "scaling", None)
    if registry is None or not callable(scaling):
        return 0.0
    k = float(getattr(scaling(), "rescue_settle_ema_k", 0.0) or 0.0)
    if k <= 0:
        return 0.0
    try:
        tau = registry.model(model).trs.ema_tau_ms
    except Exception:  # noqa: BLE001 - unknown model: no extension
        return 0.0
    return k * float(tau) if tau is not None and float(tau) > 0 else 0.0


def _rescue_bases(
    snapshot: MetricsSnapshot,
    queue: PlannerQueue,
    registry: Registry | None = None,
    contexts: dict[str, dict] | None = None,
) -> dict[str, RescueBasis]:
    """C1: per model, the last rescue target the queue issued whose effect the model's
    decision signal does not fully reflect yet: still running, or the window starts
    before it completed (the F4 rule) plus ``rescue_settle_ms`` for the EMA."""
    targets = getattr(queue, "rescue_targets", None)
    if not callable(targets):
        return {}
    check = getattr(queue, "check_restored_targets", None)
    if callable(check) and contexts:
        # P3-1: restored targets the live routable count contradicts are dropped.
        check({
            model: int(ctx["routable_pods"])
            for model, ctx in contexts.items()
            if ctx.get("routable_pods") is not None
        })
    bases: dict[str, RescueBasis] = {}
    for model, record in targets().items():
        metrics = snapshot.models.get(model)
        if metrics is None:
            continue
        if record.done_ms is not None and metrics.window_start_ms >= record.done_ms + rescue_settle_ms(
            registry, model
        ):
            continue  # settled: the signal describes the new replica count
        if record.done_ms is not None and _o1_settled((contexts or {}).get(model), int(record.done_ms)):
            continue  # O1: decided on evidence gathered after the scale-up's breakpoint
        bases[model] = RescueBasis(base=int(record.base), covered=int(record.covered))
    return bases


def _o1_settled(context: dict | None, done_ms: int) -> bool:
    """O1 (C1 settle): the routable-count change of a completed rescue target is a
    breakpoint; once the model's signal is warm on a window starting at or after it,
    that signal describes the new replica count with a freshly restarted EMA - the
    ``rescue_settle_ema_k`` extension (EMA lag) is not needed. The breakpoint carries
    the target's ``done_ms`` (or a later observation time), never an earlier one, so
    ``breakpoint >= done_ms`` means the change of this target (or a later one) was seen.
    Only breakpoints this process saw happen count (``signal_settle_ms``: the onset, or a
    count change between two views - never the first observation after a restart, whose
    date is a guess, review P2-a).
    A target that changed nothing (every part failed) never moves the breakpoint and
    settles by the window-start rule above."""
    if not context or "signal_settle_ms" not in context:
        return False
    point = context.get("signal_settle_ms")
    return point is not None and int(point) >= done_ms and bool(context.get("signal_warm"))


def _defrag_blocking_events(queue: PlannerQueue, actions) -> tuple[str, ...]:
    """Review 4 P3: a defrag in the queue conflicts with every other action, so
    a rescue scale-up planned now waits until it finished (minutes). Not
    re-planned around - made visible in the decision snapshot."""
    active = getattr(queue, "cluster_action_active", None)
    if not callable(active) or not active():
        return ()
    models = sorted(
        {
            getattr(action, "model", "")
            for action in actions
            if isinstance(action, ScaleAction) and action.delta > 0
        }
    )
    return (f"rescue_waits_for_defrag:{','.join(models)}",) if models else ()


def _preemptible_models(queue: PlannerQueue) -> set[str]:
    """Models a rescue action may preempt in the queue (review 3 P2-3)."""
    preemptible = getattr(queue, "preemptible_models", None)
    return set(preemptible()) if callable(preemptible) else set()


def _probe_backoff_models(safescale: SafeScaleController | None, now_ms: int) -> set[str]:
    backoff = getattr(safescale, "rollback_backoff_models", None)
    return set(backoff(now_ms)) if callable(backoff) else set()


def _cooled_gpus(queue: PlannerQueue, registry: Registry) -> set[tuple[str, int]]:
    """S3: GPUs whose wake the SM refused recently (ActionQueue cooldowns); a
    node-scope refusal cools every GPU of the node."""
    gpus_of = getattr(queue, "cooled_gpus", None)
    nodes_of = getattr(queue, "cooled_nodes", None)
    cooled = set(gpus_of()) if callable(gpus_of) else set()
    nodes = set(nodes_of()) if callable(nodes_of) else set()
    if nodes:
        for node in registry.topology().nodes:
            if node.name in nodes:
                cooled.update((node.name, gpu) for gpu in range(int(node.gpus)))
    return cooled


def _recent_refusals(queue: PlannerQueue) -> dict[str, str]:
    refusals = getattr(queue, "recent_refusals", None)
    return dict(refusals()) if callable(refusals) else {}


def _drain_queue_events(queue: PlannerQueue) -> list[str]:
    drain = getattr(queue, "drain_events", None)
    return list(drain()) if callable(drain) else []


def _floor_held_models(queue: PlannerQueue) -> set[str]:
    """P2-6: donors the SM recently refused with 409 floor_violation (ActionQueue)."""
    held = getattr(queue, "floor_held_models", None)
    return set(held()) if callable(held) else set()


def _action_cooldowns(snapshot: MetricsSnapshot, queue: PlannerQueue) -> dict[str, str]:
    """Models whose decision window starts before their last executed action completed
    (the window does not yet fully reflect it) -> direction of that action."""
    last_actions = getattr(queue, "last_actions", None)
    if last_actions is None:
        return {}
    cooldowns: dict[str, str] = {}
    for model, (done_ms, direction) in last_actions().items():
        metrics = snapshot.models.get(model)
        if metrics is not None and metrics.window_start_ms < done_ms:
            cooldowns[model] = direction
    return cooldowns


def _apply_safescale(
    snapshot: MetricsSnapshot,
    actions: tuple[Action, ...],
    probe_upscale_plans: dict[str, dict[str, int]],
    *,
    safescale: SafeScaleController | None,
    cluster_view: ClusterView | None = None,
    contexts: dict[str, dict] | None = None,
    observe_mode: bool = False,
    probe_block_reason: str | None = None,
    covered_targets: dict | None = None,
) -> tuple[tuple[Action, ...], tuple[str, ...]]:
    """``covered_targets`` (out): model -> its C1 rescue plan when the pods a probe
    preemption gives back cover every planned scale-up part (none is submitted)."""
    if safescale is None:
        return actions, ()

    converted: list[Action] = []
    events: list[str] = []
    # C1 review P2-1: one preemption per receiver model per tick; the pods it gives
    # back are deducted across all of the model's scale-up parts, in plan order.
    up_totals: dict[str, int] = {}
    for action in actions:
        if isinstance(action, ScaleAction) and action.delta > 0:
            up_totals[action.model] = up_totals.get(action.model, 0) + action.delta
    restore_left: dict[str, int] = {}
    restore_used: dict[str, int] = {}
    survived: set[str] = set()
    rescue_of: dict[str, object] = {}
    for action in actions:
        if observe_mode:
            # B8: never start (or preempt) a probe while paused. The planned
            # scale-down is dropped here (it only runs as a probe); anything else
            # goes to the queue, which drops it in observe mode.
            if _requires_safescale_probe(action):
                events.append(f"safescale_probe_skipped:{_safescale_probe_model(action)}:observe_mode")
            else:
                converted.append(action)
            continue
        if isinstance(action, ScaleAction) and action.delta > 0:
            model = action.model
            if model not in restore_left:
                preempt = getattr(safescale, "request_preemption", None)
                restored = preempt(model, reason="receiver_need_upscale") if callable(preempt) else 0
                restored = max(0, int(restored or 0))
                restore_left[model] = restored
                restore_used[model] = min(restored, up_totals.get(model, 0))
                if restored > 0:
                    # v1 apply_safescale_to_deltas: rollback_probe(receiver_need_upscale),
                    # then up_needed = delta - probe_hidden. The rollback (unhide) is issued
                    # by the safescale loop's next observation, one-shot and observe-mode safe.
                    events.append(
                        f"safescale_probe_preempted:{model}:restored={restored}"
                        f":up_needed={max(0, up_totals.get(model, 0) - restored)}"
                    )
            rescue = getattr(action, "rescue", None)
            if rescue is not None and restore_used.get(model, 0) > 0:
                # C1: the restored pods count toward the target already.
                rescue = replace(rescue, covered=rescue.covered + restore_used[model])
                rescue_of[model] = rescue
            take = min(restore_left[model], action.delta)
            if take > 0:
                restore_left[model] -= take
                up_needed = action.delta - take
                if up_needed > 0:
                    survived.add(model)
                    converted.append(
                        replace(
                            action,
                            delta=up_needed,
                            pods=tuple(action.pods[:up_needed]) if action.pods else (),
                            rescue=rescue,
                        )
                    )
                continue
            if rescue is not getattr(action, "rescue", None):
                action = replace(action, rescue=rescue)
            survived.add(model)
            converted.append(action)
            continue
        if not _requires_safescale_probe(action):
            converted.append(action)
            continue
        if probe_block_reason is not None:
            # P2-3: the SM maintenance lock (or an unreadable one) pauses SafeScale.
            events.append(f"safescale_probe_skipped:{_safescale_probe_model(action)}:{probe_block_reason}")
            continue

        probe_model = _safescale_probe_model(action)
        pods = _safescale_probe_pods(snapshot, action, cluster_view)
        decision = safescale.start_probe(
            model=probe_model,
            pods=pods,
            now_ms=snapshot.ts_ms,
            pending_upscales=_safescale_pending_upscales(action, probe_upscale_plans),
            window_inputs=probe_window_inputs(snapshot, probe_model, contexts, cluster_view),
        )
        if decision.status == "none":
            events.append(f"safescale_probe_skipped:{probe_model}:{decision.reason}")
            continue
        events.append(f"safescale_{decision.reason}:{probe_model}")
        if decision.reason == "probe_started" and getattr(decision, "details", None):
            events.append(format_window_event(probe_model, decision.details))
        converted.extend(_commands_to_actions(decision.commands, source_loop=action.source_loop))
    if covered_targets is not None:
        for model, rescue in rescue_of.items():
            if model not in survived:
                covered_targets[model] = rescue
    return tuple(converted), tuple(events)


def probe_window_inputs(
    snapshot: MetricsSnapshot,
    model: str,
    contexts: dict[str, dict] | None,
    cluster_view: ClusterView | None = None,
) -> ProbeWindowInputs | None:
    """The donor's adaptive-window inputs (A6) from this tick: latency p95s of its serving
    window, and Q_ctl / Y_m / y_m / Z / routable pods from its planner context - the same
    quantities v1 start_hidden_probe read from model_metrics and model_context."""
    metrics = snapshot.models.get(model)
    context = (contexts or {}).get(model) or {}
    if metrics is None and not context:
        return None
    p95_e2e = p95_tpot = interval_s = avg_ttft = avg_tpot = None
    if metrics is not None:
        serving = serving_window(metrics, cluster_view)
        p95_e2e = serving.e2e_p95_ms
        p95_tpot = serving.tpot_p95_ms
        avg_ttft = _weighted_mean(serving, "ttft")
        avg_tpot = _weighted_mean(serving, "tpot")
        interval_s = (serving.window_end_ms - serving.window_start_ms) / 1000.0
    return ProbeWindowInputs(
        p95_e2e_ms=p95_e2e,
        p95_tpot_ms=p95_tpot,
        avg_ttft_ms=avg_ttft,
        avg_tpot_ms=avg_tpot,
        q=context.get("Q_ctl"),
        y_total=context.get("Y_m"),
        y_per_pod=context.get("y_m"),
        z_m=context.get("z_m"),
        routable_pods=context.get("routable_pods"),
        interval_s=interval_s,
    )


def _weighted_mean(metrics: ModelWindowMetrics, which: str) -> float | None:
    """Model-level window mean of ``ttft`` / ``tpot`` (ms) over its pods, weighted by each
    pod's sample count = sum of sums / sum of counts (v1 get_avg_ttft_ms overall mean)."""
    total = weight = 0.0
    for pod in metrics.per_pod.values():
        avg = getattr(pod, f"{which}_avg_ms", None)
        count = getattr(pod, f"{which}_count", None)
        if avg is None or not count or count <= 0:
            continue
        total += float(avg) * float(count)
        weight += float(count)
    return total / weight if weight > 0 else None


def _requires_safescale_probe(action: Action) -> bool:
    return (isinstance(action, ScaleAction) and action.requires_safescale and action.delta < 0) or isinstance(
        action, ShrinkForSlotAction
    )


def _safescale_probe_model(action: Action) -> str:
    if isinstance(action, ShrinkForSlotAction):
        return action.donor
    return action.model


def _safescale_probe_pods(
    snapshot: MetricsSnapshot,
    action: Action,
    cluster_view: ClusterView | None = None,
) -> tuple[str, ...]:
    if isinstance(action, ShrinkForSlotAction):
        return (action.serve_id,)
    if isinstance(action, ScaleAction) and action.pods:
        return tuple(action.pods)
    return _pods_to_probe(snapshot, action.model, abs(action.delta), cluster_view=cluster_view)


def _safescale_pending_upscales(
    action: Action,
    probe_upscale_plans: dict[str, dict[str, int]],
) -> dict[str, int]:
    if isinstance(action, ShrinkForSlotAction):
        return {action.beneficiary: 1}
    return probe_upscale_plans.get(action.model, {})


def _pods_to_probe(
    snapshot: MetricsSnapshot,
    model: str,
    count: int,
    *,
    cluster_view: ClusterView | None = None,
) -> tuple[str, ...]:
    if count <= 0:
        return ()
    if cluster_view is not None:
        # Review F3: probe only serving bindings (awake and not already hidden). The
        # metrics per_pod map also lists sleeping pods, which must never be "hidden" as
        # a scale-down probe (it would remove no capacity and the commit would be a no-op).
        # Among those, release_order (registry placement policy) puts the replica whose
        # slot merges into the largest free block first, then the one on the most loaded
        # node, so a committed shrink hands back an aligned pair a tp=2 model can use and
        # keeps the nodes balanced.
        serving = sorted(
            (
                binding
                for binding in cluster_view.bindings
                if binding.model == model and binding.awake and not binding.hidden
            ),
            key=lambda binding: natural_key(binding.serve_id),
        )
        ordered = release_order(
            serving,
            bindings=list(cluster_view.bindings),
            topology=cluster_view.topology,
            policy=cluster_view.placement,
        )
        return tuple(binding.serve_id for binding in ordered[:count])
    metrics = snapshot.models.get(model)
    if metrics is None:
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
            if command.kind == "scale_down":
                actions.append(
                    ScaleAction(
                        command.model,
                        command.delta,
                        command.reason,
                        source_loop,
                        pods=command.pods,
                        sleep_path="safescale_commit",
                        drain_budget_s=command.drain_budget_s,
                    )
                )
            else:
                actions.append(ScaleAction(command.model, command.delta, command.reason, source_loop))
    return tuple(actions)


@dataclass(frozen=True)
class ModelSignal:
    """One model's decision signal of one window (planner tick and SafeScale observation
    share it, so both read the same Z and advance the shared EMA identically).

    ``window`` is the O1 effective window (None without O1); ``metrics`` the window the
    signal was computed on (the full serving window or its post-breakpoint suffix);
    ``warm`` the receiver gate (O1 evidence and/or the ADR-0013 onset guard)."""

    result: object
    signal: object
    metrics: ModelWindowMetrics
    warm: bool
    window: object | None = None
    #: O1 review P2-2: consecutive held windows reached the limit - a receiver decides
    #: on the whole window (donors still need a clean one). 0 = no fallback.
    hold_fallback: int = 0


def breakpoint_observation(
    cluster_view: ClusterView | None, queue: object | None, model: str, fallback_ms: int
) -> tuple[int, tuple[int | None, ...]]:
    """(observed_ms, done hints) for :meth:`SignalState.note_routable`: the fleet view's
    fetch time (``fallback_ms`` for a view without one - synthetic / test views) and the
    time the controller's last SM call that can change ``model``'s routable count
    returned (``ActionQueue.routable_changes``: scale / wake / sleep / hide / unhide,
    stamped after the answer, ok or not). Never a rescue target's ``done_ms``: a target
    covered by a probe preemption is stamped when planned, before its unhide ran."""
    observed = getattr(cluster_view, "fetched_ms", None)
    hints: list = []
    changes = getattr(queue, "routable_changes", None)
    if callable(changes):
        stamp = changes().get(model)
        if stamp is not None:
            hints.append(tuple(stamp) if isinstance(stamp, (tuple, list)) else int(stamp))
    return (int(observed) if observed is not None else int(fallback_ms)), tuple(hints)


def compute_model_signal(
    model_name: str,
    metrics: ModelWindowMetrics,
    spec: ModelSpec,
    *,
    signal_source: str,
    signal_state: SignalState | None,
    routable_observation: tuple[int, tuple[int | None, ...]] | None = None,
    observe_onset: bool = True,
) -> ModelSignal:
    """TSS / Z of ``metrics`` (a serving window) with the O1 breakpoint window.

    Without O1 (no ``signal_state`` or its ``breakpoint`` None) this is the pre-O1
    computation: the full window, EMA advanced, ``warm`` = the onset guard.

    With O1: the onset is recorded and the routable count noted (``routable_observation``
    = (observed_ms, done hints), None when no fleet view knows the model), then the
    effective window decides: a full window is computed exactly as before; a warm
    suffix is computed with its numerator scaled to a whole window
    (``numerator_scale = W / span``), its own queue average and the EMA restarted at
    the breakpoint; a window without enough post-breakpoint evidence reports the full
    window's raw value and leaves every EMA untouched (it decides nothing)."""
    o1 = signal_state is not None and getattr(signal_state, "breakpoint", None) is not None
    legacy_warm = True
    if signal_state is not None and observe_onset:
        # F-onset warmup guard (ADR-0013): see SignalState.observe_traffic. Called before
        # the TSS so the onset is known to O1; on an idle window it resets the EMAs the
        # TSS update would reset anyway (same state either order).
        legacy_warm = signal_state.observe_traffic(
            model_name,
            has_traffic=not window_is_idle(metrics.prompt_tokens, metrics.generation_tokens),
            window_start_ms=metrics.window_start_ms,
            window_end_ms=metrics.window_end_ms,
        )
    if signal_state is not None:
        computer = signal_state.computer_for(
            model_name, ema_alpha=spec.trs.ema_alpha, ema_tau_ms=spec.trs.ema_tau_ms
        )
    else:
        computer = TRSComputer(ema_alpha=spec.trs.ema_alpha, ema_tau_ms=spec.trs.ema_tau_ms)
    if not o1:
        result = computer.compute(
            TRSInput.from_metrics(metrics, spec.trs),
            theta_m=spec.trs.theta_m,
            window_end_ms=metrics.window_end_ms,
        )
        signal = get_signal(metrics, spec, signal_source, trs_z_m=result.Z_m, signal_state=signal_state)
        return ModelSignal(result=result, signal=signal, metrics=metrics, warm=legacy_warm)

    if routable_observation is not None:
        observed_ms, hints = routable_observation
        signal_state.note_routable(
            model_name, int(metrics.routable_pods), observed_ms=observed_ms, done_hints=hints
        )
    window = signal_state.effective_window(model_name, metrics)
    guard_warm = legacy_warm if signal_state.onset_guard_applies() else True
    # Review P2-2: consecutive held windows (a full or warm window resets the count).
    held = signal_state.note_hold(model_name, int(metrics.window_end_ms), not window.warm)
    if window.full:
        result = computer.compute(
            TRSInput.from_metrics(metrics, spec.trs),
            theta_m=spec.trs.theta_m,
            window_end_ms=metrics.window_end_ms,
        )
        signal = get_signal(metrics, spec, signal_source, trs_z_m=result.Z_m, signal_state=signal_state)
        return ModelSignal(result=result, signal=signal, metrics=metrics, warm=guard_warm, window=window)
    limit = int(getattr(signal_state.breakpoint, "hold_max_windows", 0) or 0)
    if not window.warm and limit > 0 and held >= limit:
        # Review P2-2: the routable count keeps changing (crash loop, probes): after
        # ``hold_max_windows`` held windows a receiver decides on the whole window as
        # before O1 (EMA advanced); donors keep waiting for a clean one (window.full).
        result = computer.compute(
            TRSInput.from_metrics(metrics, spec.trs),
            theta_m=spec.trs.theta_m,
            window_end_ms=metrics.window_end_ms,
        )
        signal = get_signal(metrics, spec, signal_source, trs_z_m=result.Z_m, signal_state=signal_state)
        return ModelSignal(
            result=result, signal=signal, metrics=metrics, warm=guard_warm, window=window, hold_fallback=held
        )
    if not window.warm:
        # No decision on this window: the full window's raw value for the record only.
        result = computer.compute(
            TRSInput.from_metrics(metrics, spec.trs),
            theta_m=spec.trs.theta_m,
            window_end_ms=metrics.window_end_ms,
            advance_ema=False,
        )
        signal = get_signal(metrics, spec, signal_source, trs_z_m=result.Z_m, signal_state=None)
        return ModelSignal(result=result, signal=signal, metrics=metrics, warm=False, window=window)
    suffix = window.metrics
    inp = TRSInput.from_metrics(suffix, spec.trs)
    scale = window.numerator_scale
    inp = replace(
        inp,
        prompt_tokens_total=inp.prompt_tokens_total * scale,
        generation_tokens_total=inp.generation_tokens_total * scale,
        # The EMA's idle-gap rule keeps the configured metrics window.
        window_ms=float(metrics.window_end_ms - metrics.window_start_ms),
    )
    result = computer.compute(inp, theta_m=spec.trs.theta_m, window_end_ms=suffix.window_end_ms)
    signal = get_signal(suffix, spec, signal_source, trs_z_m=result.Z_m, signal_state=signal_state)
    return ModelSignal(result=result, signal=signal, metrics=suffix, warm=guard_warm, window=window)


def _model_contexts(
    snapshot: MetricsSnapshot,
    registry: Registry,
    *,
    signal_source: str = "zm",
    cluster_view: ClusterView | None = None,
    paper_state_cache: PaperStateCache | None = None,
    signal_state: SignalState | None = None,
    queue: object | None = None,
) -> tuple[dict[str, dict], tuple[str, ...]]:
    contexts: dict[str, dict] = {}
    events: list[str] = []
    suspended = getattr(signal_state, "breakpoint_window_suspended", None)
    if suspended:
        events.append(f"breakpoint_window_suspended:{suspended}")
    cluster_counts = _cluster_view_counts(cluster_view)
    awake_counts = _awake_including_hidden(cluster_view)
    for model_name, metrics in snapshot.models.items():
        spec = registry.model(model_name)
        counts = cluster_counts.get(model_name)
        assigned_replicas = metrics.assigned_replicas
        if counts is not None:
            awake_replicas, bound_replicas = counts
            assigned_replicas = bound_replicas
            # Fleet state is the pod-count authority: sleeping pods also write gateway
            # docs, so the raw window counts them (and would carry their docs).
            metrics = serving_window(metrics, cluster_view, counts=counts)
        tokens_available = metrics.prompt_tokens is not None and metrics.generation_tokens is not None
        request_rate_rps = _request_rate_rps(metrics)
        decode_tps = per_replica_token_rate(metrics, metrics.generation_tokens)
        prefill_tps = per_replica_token_rate(metrics, metrics.prompt_tokens)
        if tokens_available:
            computed = compute_model_signal(
                model_name,
                metrics,
                spec,
                signal_source=signal_source,
                signal_state=signal_state,
                routable_observation=(
                    breakpoint_observation(cluster_view, queue, model_name, metrics.window_end_ms)
                    if counts is not None
                    else None
                ),
            )
            result, signal, signal_warm = computed.result, computed.signal, computed.warm
            window = computed.window
            if computed.hold_fallback:
                events.append(f"breakpoint_hold_fallback:{model_name}:{computed.hold_fallback}")
            if window is not None and not window.full:
                # Rates of the post-breakpoint window (the decision window).
                request_rate_rps = _request_rate_rps(computed.metrics)
                decode_tps = per_replica_token_rate(computed.metrics, computed.metrics.generation_tokens)
                prefill_tps = per_replica_token_rate(computed.metrics, computed.metrics.prompt_tokens)
            context = {
                "trs": result.TRS,
                # Pre-EMA TSS, read-only: exposed so a capture can store raw TSS, EMA and Z
                # of the same tick (the replica factor it includes is not otherwise exposed).
                "trs_raw": result.TRS_raw,
                "z_m": signal.z_m,
                "signal_source": signal.source,
                "signal_raw_value": signal.raw_value,
                "signal_unavailable_reason": signal.unavailable_reason,
                "signal_theta": _signal_theta(spec, signal.source),
                "trs_z_m": result.Z_m,
                "tss_defined": result.defined,
                "eta_m": result.eta_m,
                "theta_m": spec.trs.theta_m,
                "Q": result.Q,
                "Q_ctl": result.Q_ctl,
                "Y_m": result.Y_m,
                "y_m": result.y_m,
                "routable_pods": metrics.routable_pods,
                "assigned_replicas": assigned_replicas,
                "signal_warm": signal_warm,
                "request_rate_rps": request_rate_rps,
                "decode_tps": decode_tps,
                "prefill_tps": prefill_tps,
            }
            if window is not None:
                # O1: a scale-down needs a whole window after the breakpoint
                # (signal_full_window); a scale-up the evidence (signal_warm).
                context.update(
                    {
                        "signal_full_window": window.full,
                        "signal_breakpoint_ms": window.breakpoint_ms,
                        "signal_settle_ms": signal_state.settle_breakpoint_ms(model_name),
                        "signal_window_start_ms": window.start_ms,
                        "signal_evidence_grids": window.grids,
                        "signal_hold_reason": None if computed.hold_fallback else window.reason,
                        # Completed requests of the post-breakpoint window (the C1 step
                        # cap's evidence); None on a whole window.
                        "signal_evidence_requests": (
                            None if window.full else getattr(window.metrics, "request_count", None)
                        ),
                    }
                )
        else:
            # tokens_available=False means the metrics are MISSING (scrape gap / stale store),
            # not that the model is idle (a live idle pod reports zero-delta tokens, which is the
            # tokens branch above with Y_m<=1e-9). Do NOT touch the traffic-onset cursor here:
            # resetting on a transient scrape hiccup would re-suppress a genuinely-CRITICAL model
            # for a full window after metrics recover (review F1). Hold the cursor.
            pass
            context = {
                "trs": 0.0,
                "trs_raw": None,
                "z_m": None,
                "signal_source": signal_source,
                "signal_raw_value": None,
                "signal_unavailable_reason": "tokens_missing",
                "signal_theta": _signal_theta(spec, signal_source),
                "trs_z_m": None,
                "eta_m": None,
                "theta_m": spec.trs.theta_m,
                # Same queue term as tre_common.tss (A + w_q*W, swapping excluded).
                "Q": metrics.avg_running + spec.trs.lambda_wait * metrics.avg_waiting,
                "Q_ctl": max(metrics.avg_running + spec.trs.lambda_wait * metrics.avg_waiting, spec.trs.qmin),
                "Y_m": None,
                "y_m": None,
                "routable_pods": metrics.routable_pods,
                "assigned_replicas": assigned_replicas,
                "request_rate_rps": request_rate_rps,
                "decode_tps": decode_tps,
                "prefill_tps": prefill_tps,
            }
        # Scaling-cap count (A1/P1-2): awake bindings incl. hidden probe pods.
        context["awake_replicas"] = awake_counts.get(model_name, metrics.routable_pods)
        if paper_state_cache is not None:
            context, model_events = paper_state_cache.apply(model_name, context, tokens_available=tokens_available)
            events.extend(model_events)
        contexts[model_name] = context
    return contexts, tuple(events)


def _signal_theta(spec: ModelSpec, source: str) -> float | None:
    if source == "zm":
        return spec.trs.theta_m
    threshold = spec.alt_thresholds.get(source)
    return threshold.theta if threshold is not None else None


def _request_rate_rps(metrics: ModelWindowMetrics) -> float | None:
    if metrics.request_count is None:
        return None
    duration_s = (metrics.window_end_ms - metrics.window_start_ms) / 1000.0
    if duration_s <= 0.0:
        return None
    return max(0.0, float(metrics.request_count)) / duration_s


def serving_window(
    metrics: ModelWindowMetrics,
    cluster_view: ClusterView | None,
    *,
    counts: tuple[int, int] | None = None,
) -> ModelWindowMetrics:
    """The model's window restricted to its serving pods (``restrict_to_serving``):
    docs of pods the fleet state reports asleep are dropped and routable/assigned become
    the awake-and-not-hidden count. Without a fleet view (or a model it does not list) the
    raw window is returned unchanged. Used by the planner tick and the safescale
    observation, so both decide on the same window."""
    if cluster_view is None:
        return metrics
    if counts is None:
        counts = _cluster_view_counts(cluster_view).get(metrics.model)
        if counts is None:
            return metrics
    sleeping = {
        binding.serve_id
        for binding in cluster_view.bindings
        if binding.model == metrics.model and not binding.awake
    }
    return restrict_to_serving(metrics, sleeping_pods=sleeping, routable_pods=counts[0])


def _awake_including_hidden(cluster_view: ClusterView | None) -> dict[str, int]:
    if cluster_view is None:
        return {}
    counts: dict[str, int] = {}
    for binding in cluster_view.bindings:
        counts.setdefault(binding.model, 0)
        if binding.awake:
            counts[binding.model] += 1
    return counts


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


def _idle_gpus(
    snapshot: MetricsSnapshot,
    registry: Registry,
    cluster_view: ClusterView | None = None,
) -> int:
    if cluster_view is not None:
        # GPUs with no awake binding (hidden-but-awake still occupies its GPU). The
        # scrape-based count below treats every resident (sleeping) binding as free
        # capacity, which is wrong under multi-model-per-GPU residency.
        occupied = {
            (binding.slot.node, gpu)
            for binding in cluster_view.bindings
            if binding.awake
            for gpu in binding.slot.gpu_ids
        }
        return sum(
            1
            for node in cluster_view.topology.nodes
            for gpu in range(node.gpus)
            if (node.name, gpu) not in occupied
        )
    total_gpus = sum(node.gpus for node in registry.topology().nodes)
    used_gpus = 0
    for model_name, metrics in snapshot.models.items():
        try:
            spec: ModelSpec = registry.model(model_name)
            used_gpus += max(0, metrics.assigned_replicas) * spec.tp_size
        except KeyError:
            continue
    return max(0, total_gpus - used_gpus)

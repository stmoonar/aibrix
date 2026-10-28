from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Callable, Protocol, Union

if TYPE_CHECKING:
    from tre_controller.profiling import TickProfiler

from tre_common.gpu_placement import placement_policy_from_registry
from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics, observed_of
from tre_common.registry import Registry, ModelSpec
from tre_common.tss import window_is_idle
from tre_common.window_pods import restrict_to_serving
from tre_controller.config import SAFESCALE_SCOPE_RECEIVERLESS_SHRINK
from tre_controller.loops.action_queue import PlanBatch, block_sleeps, receiver_upscales, transfer_plan
from tre_controller.planning.classify import classify_all_models, model_control_configs_from_registry
from tre_controller.planning.planner import (
    BlockSupplyAction,
    IMMEDIATE_DONOR_SLEEP_PATH,
    NO_SAFESCALE_REASON_SUFFIX,
    Action,
    ClusterView,
    HideAction,
    IncompletePolicy,
    PlanConfig,
    RescueBasis,
    ScaleAction,
    UnhideAction,
    build_plan,
    probe_reserved_gpus,
    upscale_of,
)
from tre_controller.planning.safescale import (
    ProbeWindowInputs,
    SafeScaleCommand,
    SafeScaleDecision,
    format_window_event,
)
from tre_controller.signals.sources import get_signal, per_replica_token_rate
from tre_controller.signals.trs import SignalState, TRSComputer, TRSInput, zero_tokens_unconfirmed
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
    #: Window end (ms, gateway doc stamp) of the last window whose tokens were known.
    good_end_ms: int


class PaperStateCache:
    """A model whose tokens are missing keeps its last computed context while that
    context's window ended at most ``max_stale_ms`` before the current window
    (2026-10-25: elapsed window time, not a count of reads - 30 s = the former
    3 windows of 10 s, at any evidence period)."""

    def __init__(self, *, max_stale_ms: int = 30_000) -> None:
        self.max_stale_ms = max(0, int(max_stale_ms))
        self._by_model: dict[str, _PaperState] = {}

    def apply(
        self, model_name: str, context: dict, *, tokens_available: bool, window_end_ms: int
    ) -> tuple[dict, tuple[str, ...]]:
        if tokens_available:
            self._by_model[model_name] = _PaperState(dict(context), good_end_ms=int(window_end_ms))
            return context, ()

        previous = self._by_model.get(model_name)
        if previous is None:
            return context, (f"paper_state_stale_unknown:{model_name}",)
        if int(window_end_ms) - previous.good_end_ms > self.max_stale_ms:
            return context, (f"paper_state_stale_unknown:{model_name}",)
        held = dict(previous.context)
        held.update(
            {
                "routable_pods": context.get("routable_pods", held.get("routable_pods", 0)),
                "assigned_replicas": context.get("assigned_replicas", held.get("assigned_replicas", 0)),
                "awake_replicas": context.get("awake_replicas", held.get("awake_replicas")),
                # 2026-10-02: the SM floor headroom of THIS tick's view (never a held one).
                "floor": context.get("floor"),
                "floor_headroom": context.get("floor_headroom"),
                # I4 (2026-10-04): a held level was not computed from tokens of the
                # current window - never scale-down / release evidence (the planner's
                # donor gate), and the commit revalidation reads the model UNCONFIRMED.
                "sample_quality_hold": "tokens_missing",
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
    observe_mode: bool = False,
    probe_block_reason: str | None = None,
    leftover_probes: SafeScaleController | None = None,
    safescale_scope: str = "all",
) -> LoopTickResult:
    """One planner tick. ``safescale_scope`` (TRE_SAFESCALE_SCOPE, 2026-10-26):
    ``receiverless_shrink`` keeps a probe only for the receiver-less HIGH proactive
    shrink (:func:`_receiverless_shrink`); a donor planned for a receiver is a relay and
    a TP block sleeps its donor pods at once, as with SafeScale off. ``observe_mode`` (controller mode ``observe``, B8): the
    plan is still computed and published, but no SafeScale probe is started or
    preempted - the probe's state would change (hide, window, commit) while its
    hide never reaches the cluster. Every other action is still submitted; the
    ActionQueue drops it in observe mode (it never reaches the SM).

    ``probe_block_reason`` (P2-3): no SafeScale probe is started this tick
    (``sm_maintenance``: the SM maintenance lock is held;
    ``sm_maintenance_unreadable``: it could not be read - fail-closed). The
    planned probe-only scale-downs are dropped with the event
    ``safescale_probe_skipped:<model>:<reason>``; every other action is kept.

    ``safescale`` None (TRE_ABLATION_DISABLE_SAFESCALE): no probe; every shrink that
    would start one is released immediately (``PlanConfig.release_without_safescale``,
    :func:`_release_without_safescale`). ``leftover_probes``: the state machine that
    only finishes probes an earlier run left - their GPUs stay reserved meanwhile."""
    if snapshot.stale:
        return LoopTickResult(submitted=0, events=("snapshot_stale",))
    if cluster_view is not None and cluster_view.placement is None:
        # Every placement decision of this tick (wakes, creates, donor slots, probe /
        # shrink order) ranks by the registry placement policy.
        cluster_view = replace(cluster_view, placement=placement_policy_from_registry(registry))
    reserve_events: tuple[str, ...] = ()
    if cluster_view is not None:
        # Design donor-evidence-20261007 item 3: the GPUs in-flight probes count on for
        # their receivers are no capacity for anything else of this tick.
        reserved = probe_reserved_gpus(
            cluster_view,
            _unresolved_probes(safescale if safescale is not None else leftover_probes),
            {spec.name: spec.tp_size for spec in registry.models()}
        )
        extra = reserved - {(str(node), int(gpu)) for node, gpu in cluster_view.blocked_gpus}
        if extra:
            cluster_view = replace(cluster_view, blocked_gpus=frozenset(cluster_view.blocked_gpus) | extra)
            reserve_events = (
                "probe_slot_reserved:" + ",".join(f"{node}/{gpu}" for node, gpu in sorted(extra)),
            )

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
    )
    # Without model_control_configs every model silently falls back to the generic
    # delta_crit=0.2 / delta_high=0.25 and the fitted per-model bands in registry.yaml
    # are dead keys. An alternative signal source runs on its own fitted bands.
    classifications = classify_all_models(
        contexts,
        model_control_configs=model_control_configs_from_registry(registry, signal_source),
        signal_idle_rps_eps=signal_idle_rps_eps,
    )
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
        release_without_safescale=safescale is None or safescale_scope == SAFESCALE_SCOPE_RECEIVERLESS_SHRINK,
        defrag_enabled=_defrag_enabled(registry),
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
        # Correctness guard (TSS only, 2026-10-29): the fleet view predates the model's
        # last routable-changing SM call - no new action of the model until a newer
        # view shows it. A view that stops refreshing keeps holding (conservative on
        # missing data, review P3-6; cluster_view_task raises the cluster_view_stale
        # alert). No signal-lag hold: the window's lag is accepted.
        view_pending=_view_pending(queue, cluster_view),
        # 2026-10-02: no floor-violation hold and no GPU wake cooldown any more - a
        # donor's SM floor_headroom bounds it, refusals are events (queue events below).
        # Timer cleanup (2026-10-02): rolled-back probes wait for new evidence (no 60 s backoff).
        probe_backoff_models=_probe_backoff_models(safescale, snapshot, contexts),
        preemptible_models=_preemptible_models(queue) if rescue_due else None,
        # C1: earlier rescue targets the fleet view does not show yet (no double count).
        rescue_bases=(
            _with_probe_promises(_rescue_bases(snapshot, queue, cluster_view, contexts), safescale, contexts)
            if rescue_due else None
        ),
        # Review P2-1: relays the SM answered with nothing done on a still-current view.
        relay_holds=_relay_holds(queue),
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
        safescale_scope=safescale_scope,
    )
    # C1 review P2-1: a rescue target fully covered by the pods a probe preemption gives
    # back is still a target the window does not reflect yet.
    record_covered = getattr(queue, "record_rescue_covered", None)
    if callable(record_covered):
        for model, rescue in covered_targets.items():
            record_covered(model, rescue)
    # sm-plans section 10: the controller names every pod it sleeps.
    actions, naming_events = _name_shrink_pods(snapshot, actions, cluster_view)
    safescale_events = tuple(safescale_events) + naming_events
    if _prof_on:
        _safescale_ns = time.perf_counter_ns() - _phase_t0
        _phase_t0 = time.perf_counter_ns()
    queue_events = _defrag_blocking_events(queue, actions) if rescue_due else ()
    # Observation events of the queue (transfer outcomes, wake_refused,
    # *_clamped_by_floor, placement_substituted) - no timer follows any of them.
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
        events=(
            paper_events + reserve_events + tuple(plan.events)
            + safescale_events + queue_events
        ),
        model_contexts=contexts,
        classifications={item.model_name: item for item in classifications},
    )


def _defrag_enabled(registry: Registry) -> bool:
    placement = getattr(registry, "placement", None)
    return bool(getattr(placement(), "defrag_enabled", False)) if callable(placement) else False


def _rescue_bases(
    snapshot: MetricsSnapshot,
    queue: PlannerQueue,
    cluster_view: ClusterView | None = None,
    contexts: dict[str, dict] | None = None,
) -> dict[str, RescueBasis]:
    """C1 ledger: per model, the last rescue target the queue issued that the fleet view
    of this tick does not show yet - still running, or completed after the SM state of
    the view was produced (``state_ms``, else ``fetched_ms``). The planner counts its
    replicas as covered, so one target is never planned twice. State only (TSS only,
    2026-10-29): once the view shows the target, the model's next step follows its
    signal - no wait for the metrics window or the EMA to reflect it. A view without a
    time (synthetic / offline) shows every completed target."""
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
    bound = _view_bound(cluster_view)
    bases: dict[str, RescueBasis] = {}
    for model, record in targets().items():
        if snapshot.models.get(model) is None:
            continue
        done = record.done_ms
        if done is not None and (bound is None or int(done) <= bound):
            continue  # the fleet view shows it
        bases[model] = RescueBasis(base=int(record.base), covered=int(record.covered))
    return bases


def _view_bound(cluster_view: ClusterView | None) -> int | None:
    """When the SM state of ``cluster_view`` was produced at the latest (a lower bound:
    ``state_ms``, else ``fetched_ms``); None for a synthetic view."""
    bound = getattr(cluster_view, "state_ms", None)
    if bound is None:
        bound = getattr(cluster_view, "fetched_ms", None)
    return int(bound) if bound is not None else None


def _with_probe_promises(
    bases: dict[str, RescueBasis], safescale: SafeScaleController | None, contexts: dict[str, dict] | None
) -> dict[str, RescueBasis]:
    """Review P2-3: a receiver an unresolved SafeScale probe promises replicas to (its
    commit wakes them: ``pending_upscales``) holds that target like a queued rescue
    target - the step is planned, so the next calls add nothing until the probe
    resolves (state, not time; after a commit the rescue target rules take over)."""
    out = dict(bases)
    for probe in _unresolved_probes(safescale):
        for model, count in (getattr(probe, "pending_upscales", None) or {}).items():
            if int(count) <= 0:
                continue
            basis = out.get(model)
            if basis is None:
                routable = int(((contexts or {}).get(model) or {}).get("routable_pods") or 0)
                basis = RescueBasis(base=routable, covered=routable)
            out[model] = RescueBasis(base=basis.base, covered=basis.covered + int(count))
    return out


def _defrag_blocking_events(queue: PlannerQueue, actions) -> tuple[str, ...]:
    """Review 4 P3: a defrag in the queue conflicts with every other action, so
    a rescue scale-up planned now waits until it finished (minutes). Not
    re-planned around - made visible in the decision snapshot."""
    active = getattr(queue, "cluster_action_active", None)
    if not callable(active) or not active():
        return ()
    models = sorted({up[0] for up in map(upscale_of, actions) if up is not None})
    return (f"rescue_waits_for_defrag:{','.join(models)}",) if models else ()


def _preemptible_models(queue: PlannerQueue) -> set[str]:
    """Models a rescue action may preempt in the queue (review 3 P2-3)."""
    preemptible = getattr(queue, "preemptible_models", None)
    return set(preemptible()) if callable(preemptible) else set()


def _unresolved_probes(safescale: SafeScaleController | None) -> tuple:
    """Every unresolved SafeScale probe, probing or committing (none without SafeScale)."""
    reader = getattr(safescale, "all_probes", None)
    return tuple(reader()) if callable(reader) else ()


def _probe_backoff_models(
    safescale: SafeScaleController | None, snapshot: MetricsSnapshot, contexts: dict[str, dict]
) -> dict[str, str]:
    """Models held as a probe donor after a rollback -> why
    (``SafeScaleStateMachine.rollback_retry_holds``, timer cleanup 2026-10-02): the
    decision window's Z, routable count and newest observation (round 4: the observed
    span after the rollback is the evidence a retry needs). The planner applies it to
    every donor probe; under scope ``receiverless_shrink`` that is only the
    receiver-less HIGH probe (relays to a receiver are immediate and not held)."""
    holds = getattr(safescale, "rollback_retry_holds", None)
    if not callable(holds):
        return {}
    signals = {}
    for model, metrics in snapshot.models.items():
        context = contexts.get(model) or {}
        z = context.get("z_m")
        routable = context.get("routable_pods")
        signals[model] = (
            float(z) if z is not None else None,
            int(routable) if routable is not None else None,
            observed_of(metrics).t_last_ms,
        )
    return dict(holds(signals))


def _relay_holds(queue: PlannerQueue):
    reader = getattr(queue, "relay_holds", None)
    return reader() if callable(reader) else None


def _drain_queue_events(queue: PlannerQueue) -> list[str]:
    drain = getattr(queue, "drain_events", None)
    return list(drain()) if callable(drain) else []


def _view_pending(queue: PlannerQueue, cluster_view: ClusterView | None) -> dict[str, str]:
    """Correctness guard (timer cleanup 2026-10-02; every model since 2026-10-29): models
    whose last routable-changing SM call returned after the SM state of this tick's fleet
    view was produced -> that call's direction. The view cannot show the change yet, so
    the model takes no new action until a newer view exists (F2: every such call asks
    for a refresh).

    Sources (review P2-1): ``ActionQueue.view_changes`` - every routable-changing call,
    failed / partial ones and probe unhides included - and ``last_actions``; the later
    one per model decides. The view's time is its lower bound ``state_ms`` (review
    P2-2; ``fetched_ms`` for a view without one). A view without either (synthetic /
    offline) holds nothing."""
    bound = _view_bound(cluster_view)
    if bound is None:
        return {}
    latest: dict[str, tuple[int, str]] = {}
    for source in ("view_changes", "last_actions"):
        read = getattr(queue, source, None)
        if not callable(read):
            continue
        for model, (done_ms, direction) in read().items():
            current = latest.get(model)
            if current is None or int(done_ms) > current[0]:
                latest[model] = (int(done_ms), str(direction))
    return {model: direction for model, (done_ms, direction) in latest.items() if done_ms > bound}


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
    safescale_scope: str = "all",
) -> tuple[tuple[Action, ...], tuple[str, ...]]:
    """``covered_targets`` (out): model -> its C1 rescue plan when the pods a probe
    preemption gives back cover every planned scale-up part (none is submitted)."""
    if safescale is None:
        # No SafeScale (TRE_ABLATION_DISABLE_SAFESCALE): release immediately, as before
        # 05f489f1 / v1 (2026-10-08).
        return _release_without_safescale(actions)
    # L4 rev 2: a TP block without a probe (IDLE donors; a scope that releases at once)
    # is one plan now; a block with a probe stays one until its probe starts.
    actions = tuple(
        _release_without_safescale((action,))[0][0]
        if isinstance(action, BlockSupplyAction) and action.probe is None else action
        for action in actions
    )
    if safescale_scope == SAFESCALE_SCOPE_RECEIVERLESS_SHRINK:
        # TRE_SAFESCALE_SCOPE=receiverless_shrink (2026-10-26): only a receiver-less
        # shrink stays a probe; any other probe placement is released at once.
        actions = tuple(
            action if not _requires_safescale_probe(action) or _receiverless_shrink(action)
            else _release_without_safescale((action,))[0][0]
            for action in actions
        )

    converted: list[Action] = []
    events: list[str] = []
    # C1 review P2-1: one preemption per receiver model per tick; the pods it gives
    # back are deducted across all of the model's scale-up parts, in plan order.
    up_totals: dict[str, int] = {}
    for action in actions:
        for model, delta in receiver_upscales(action):
            up_totals[model] = up_totals.get(model, 0) + delta
    restore_left: dict[str, int] = {}
    restore_used: dict[str, int] = {}
    survived: set[str] = set()
    rescue_of: dict[str, object] = {}

    def preempt_once(model: str) -> None:
        if model in restore_left:
            return
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
        if isinstance(action, (PlanBatch, BlockSupplyAction)):
            # Review P2-2: a TP block wakes its receiver too (one plan, or the probe's
            # commit): the receiver's own probe is preempted like for any scale-up;
            # restored pods covering it make the release pointless.
            ups = receiver_upscales(action)
            if len(ups) == 1:
                model, delta = ups[0]
                preempt_once(model)
                rescue = _rescue_of(action)
                if rescue is not None and restore_used.get(model, 0) > 0:
                    rescue_of[model] = replace(rescue, covered=rescue.covered + restore_used[model])
                if restore_left[model] >= delta:
                    restore_left[model] -= delta
                    events.append(f"block_supply_covered_by_preemption:{model}")
                    continue
                survived.add(model)
            if isinstance(action, PlanBatch):
                converted.append(action)
                continue
        if upscale_of(action) is not None:
            model, delta = upscale_of(action)
            preempt_once(model)
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
        if isinstance(action, BlockSupplyAction):
            # L4 rev 2: the block's IDLE pods sleep now (their GPUs stay reserved for
            # the receiver while the probe runs, ``probe_reserved_gpus``).
            converted.extend(block_sleeps(action, action.reason))
        converted.extend(_commands_to_actions(decision.commands, source_loop=action.source_loop))
    if covered_targets is not None:
        for model, rescue in rescue_of.items():
            if model not in survived:
                covered_targets[model] = rescue
    return tuple(converted), tuple(events)


def _rescue_of(action: Action) -> object | None:
    """The rescue target a TP block's receiver wake carries (PlanBatch member or
    BlockSupplyAction), or None."""
    for member in (action.members if isinstance(action, PlanBatch) else (action,)):
        rescue = getattr(member, "rescue", None)
        if rescue is not None and (isinstance(member, BlockSupplyAction) or upscale_of(member) is not None):
            return rescue
    return None


def _release_without_safescale(actions: tuple[Action, ...]) -> tuple[tuple[Action, ...], tuple[str, ...]]:
    """Without SafeScale the probe shrinks the planner still emits take the immediate
    donor path (sleep path ``urgent``, reason suffix ``_nosafescale``); a middle-zone /
    HIGH donor for a receiver is already planned as a relay
    (``PlanConfig.release_without_safescale``). Everything else is unchanged:

    * a receiver-less shrink (HIGH proactive) -> a model-level urgent scale-down (the
      SM keeps the replica floor);
    * a transfer (:class:`BlockSupplyAction`: a TP block, a TP1 relay pair) -> ONE
      plan group (:class:`PlanBatch`, design 20261028-single-loop L4 rev 2,
      20261027-sm-plans section 10): the urgent sleeps of exactly the donor pods (its
      probe pod too) and the receiver's NAMED wake of its binding on the block (it keeps
      its rescue target). The SM selects the group atomically and wakes the receiver
      once every donor slept; no other receiver can take the block between."""
    out: list[Action] = []
    for action in actions:
        if not _requires_safescale_probe(action):
            out.append(action)
            continue
        reason = f"{action.reason}{NO_SAFESCALE_REASON_SUFFIX}"
        if isinstance(action, BlockSupplyAction):
            if action.probe is None:
                reason = action.reason  # the planner named it (IDLE: no suffix)
            else:
                action = replace(action, sleeps=(*action.sleeps, action.probe), probe=None)
            out.append(transfer_plan(action, reason))
        else:
            out.append(replace(
                action, reason=reason, requires_safescale=False, sleep_path=IMMEDIATE_DONOR_SLEEP_PATH
            ))
    return tuple(out), ()


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
    if isinstance(action, BlockSupplyAction):
        return True  # a probe, or (no probe) released at once by policy
    return isinstance(action, ScaleAction) and action.requires_safescale and action.delta < 0


def _receiverless_shrink(action: Action) -> bool:
    """A probe shrink with no receiver: the planner pairs a donor with a recipient only
    on ``ScaleAction.receiver`` (``planner._plan_middle_zone_probe``) and on a TP
    block's ``BlockSupplyAction.receiver``; the HIGH proactive shrink
    (``high_proactive_safescale``) carries neither."""
    return isinstance(action, ScaleAction) and action.requires_safescale and action.delta < 0 and (
        action.receiver is None
    )


def _safescale_probe_model(action: Action) -> str:
    if isinstance(action, BlockSupplyAction):
        return action.probe[0] if action.probe is not None else action.receiver
    return action.model


def _safescale_probe_pods(
    snapshot: MetricsSnapshot,
    action: Action,
    cluster_view: ClusterView | None = None,
) -> tuple[str, ...]:
    if isinstance(action, BlockSupplyAction):
        return (action.probe[1],) if action.probe is not None else ()
    if isinstance(action, ScaleAction) and action.pods:
        return tuple(action.pods)
    return _pods_to_probe(snapshot, action.model, abs(action.delta), cluster_view=cluster_view)


def _safescale_pending_upscales(
    action: Action,
    probe_upscale_plans: dict[str, dict[str, int]],
) -> dict[str, int]:
    if isinstance(action, BlockSupplyAction):
        return {action.receiver: 1}
    return probe_upscale_plans.get(action.model, {})


def _name_shrink_pods(
    snapshot: MetricsSnapshot, actions: tuple[Action, ...], cluster_view: ClusterView | None
) -> tuple[tuple[Action, ...], tuple[str, ...]]:
    """sm-plans section 10 (2026-10-28): one decider per choice - the controller names
    every pod it sleeps, the SM validates and executes. An immediate model-level shrink
    the planner left unnamed (IDLE proactive, a receiver-less shrink released without
    SafeScale) names its pods here by the shared release order (:func:`_pods_to_probe`,
    the SM's ``shrink_order``), skipping pods another action of this tick names. Fewer
    pods than asked: the shrink is that many; none: it is dropped
    (``shrink_unnamed:<model>``). SafeScale probes are named when they hide."""
    named = {
        pod
        for action in actions
        for member in (action.members if isinstance(action, PlanBatch) else (action,))
        for pod in (getattr(member, "pods", ()) or ())
    }
    out: list[Action] = []
    events: list[str] = []
    for action in actions:
        if not (
            isinstance(action, ScaleAction) and action.delta < 0 and not action.pods
            and not action.requires_safescale
        ):
            out.append(action)
            continue
        candidates = _pods_to_probe(snapshot, action.model, 10**6, cluster_view=cluster_view)
        pods = tuple(pod for pod in candidates if pod not in named)[: abs(action.delta)]
        if not pods:
            events.append(f"shrink_unnamed:{action.model}")
            continue
        named |= set(pods)
        out.append(replace(action, delta=-len(pods), pods=pods))
    return tuple(out), tuple(events)


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
    share it, so both read the same Z and advance the shared EMA identically)."""

    result: object
    signal: object
    metrics: ModelWindowMetrics


def compute_model_signal(
    model_name: str,
    metrics: ModelWindowMetrics,
    spec: ModelSpec,
    *,
    signal_source: str,
    signal_state: SignalState | None,
    routable_pod_ids: frozenset[str] | None = None,
) -> ModelSignal:
    """TSS / Z of ``metrics`` (a serving window): always the whole sliding window plus
    the model's shared EMA (TSS only, 2026-10-29, design docs/design/20261029-tss-only.md;
    v1 behaviour). A window that straddles a replica change mixes both regimes - TSS is a
    ratio of window totals, so that only adds lag. An idle window (nothing in flight), or a
    zero-token window whose zero may be missing data (:func:`zero_tokens_unconfirmed`),
    clears the model's EMAs (:meth:`SignalState.reset_ema`)."""
    if signal_state is not None:
        if window_is_idle(metrics.avg_running, metrics.avg_waiting) or zero_tokens_unconfirmed(metrics):
            signal_state.reset_ema(model_name)
        computer = signal_state.computer_for(
            model_name, ema_alpha=spec.trs.ema_alpha, ema_tau_ms=spec.trs.ema_tau_ms
        )
    else:
        computer = TRSComputer(ema_alpha=spec.trs.ema_alpha, ema_tau_ms=spec.trs.ema_tau_ms)
    result = computer.compute(
        TRSInput.from_metrics(metrics, spec.trs),
        theta_m=spec.trs.theta_m,
        window_end_ms=metrics.window_end_ms,
    )
    signal = get_signal(
        metrics, spec, signal_source, trs_z_m=result.Z_m, signal_state=signal_state,
        routable_pod_ids=routable_pod_ids,
    )
    return ModelSignal(result=result, signal=signal, metrics=metrics)


def _model_contexts(
    snapshot: MetricsSnapshot,
    registry: Registry,
    *,
    signal_source: str = "zm",
    cluster_view: ClusterView | None = None,
    paper_state_cache: PaperStateCache | None = None,
    signal_state: SignalState | None = None,
) -> tuple[dict[str, dict], tuple[str, ...]]:
    contexts: dict[str, dict] = {}
    events: list[str] = []
    routable_error = getattr(cluster_view, "routable_error", None)
    if cluster_view is not None and routable_error:
        # The SM did not report its routable view (older SM, or it could not read the
        # Pods / GPU leases): the controller's own count is used this tick.
        events.append(f"sm_routable_fallback:{routable_error}")
    cluster_counts = _cluster_view_counts(cluster_view)
    awake_counts = _awake_including_hidden(cluster_view)
    routable_by_model = _routable_pod_ids(cluster_view)
    for model_name, metrics in snapshot.models.items():
        raw_metrics = metrics
        spec = registry.model(model_name)
        stale_pods = getattr(metrics, "scrape_stale_pods", ())
        if stale_pods:
            # I3 (2026-10-25: merged into the sample rule): no sample of these pods was
            # observed in the window; they are left out (no pod left -> tokens missing,
            # not zero) and hold the model as a donor (I4 below).
            events.append(f"scrape_stale:{model_name}:{','.join(stale_pods)}")
        incomplete_pods = getattr(metrics, "sample_incomplete_pods", ())
        if incomplete_pods:
            # 2026-10-25: flagged pods (too few samples, a token span over 1.5 windows, or
            # left out): a donor-gate flag (I4 below), nothing else.
            events.append(f"samples_incomplete:{model_name}:{','.join(incomplete_pods)}")
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
                # Review L2: the alt signals sum the routable pods only (hidden probes out).
                routable_pod_ids=(
                    (routable_by_model or {}).get(model_name, frozenset())
                    if counts is not None and routable_by_model is not None
                    else None
                ),
            )
            result, signal = computed.result, computed.signal
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
                "request_rate_rps": request_rate_rps,
                "decode_tps": decode_tps,
                "prefill_tps": prefill_tps,
            }
        else:
            # tokens_available=False means the metrics are MISSING (scrape gap / stale store),
            # not that the model is idle (a live idle pod reports zero-delta tokens, which is the
            # tokens branch above with Y_m<=1e-9): the EMA is left alone.
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
        # 2026-10-02: the SM replica floor / headroom (donor bound of the planner).
        context.update(_sm_floor_context(cluster_view, model_name))
        serving_now = routable_by_model.get(model_name, frozenset()) if routable_by_model is not None else None
        flagged = set(incomplete_pods)
        if serving_now is None:
            held_incomplete = flagged | set(stale_pods)
        elif raw_metrics.per_pod:
            # M1: a serving pod without complete data in the window (flagged, I3-stale,
            # or no doc at all) holds as well. (An empty per_pod: no pod data at all -
            # tokens missing below, or a synthetic window without per-pod data.)
            complete = set(raw_metrics.per_pod) - flagged
            held_incomplete = set(serving_now) - complete
        else:
            held_incomplete = (flagged | set(stale_pods)) & set(serving_now)
        if tokens_available and held_incomplete:
            # I4 (review round 1): a serving pod's data is incomplete in this window -
            # no scale-down / release / donor evidence (the planner's donor gate). Only
            # that: the level and the receiver steps are unchanged.
            missing = held_incomplete - set(raw_metrics.per_pod or ())
            context["sample_quality_hold"] = (
                "scrape_stale" if missing & set(stale_pods)
                else "pod_missing" if missing
                else "samples_incomplete"
            )
        if paper_state_cache is not None:
            context, model_events = paper_state_cache.apply(
                model_name, context, tokens_available=tokens_available, window_end_ms=int(metrics.window_end_ms)
            )
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


def _routable_pod_ids(cluster_view: ClusterView | None) -> dict[str, frozenset[str]] | None:
    """model -> serve_ids (== pod names) of its routable bindings, by the same rule as
    :func:`_cluster_view_counts`; None without a fleet view."""
    if cluster_view is None:
        return None
    routable_ids = getattr(cluster_view, "routable_ids", None)
    out: dict[str, set[str]] = {}
    for binding in cluster_view.bindings:
        if _is_routable(binding, routable_ids):
            out.setdefault(binding.model, set()).add(binding.serve_id)
    return {model: frozenset(ids) for model, ids in out.items()}


def _is_routable(binding, routable_ids) -> bool:
    """The SM's routable set (``/v2/state`` ``bindings[].routable``); without it the
    controller's former rule: awake and not hidden."""
    if routable_ids is not None:
        return binding.serve_id in routable_ids
    return bool(binding.awake and not binding.hidden)


def _cluster_view_counts(cluster_view: ClusterView | None) -> dict[str, tuple[int, int]]:
    """model -> (routable, bound) of the fleet view. Routable is the SM's own count
    (``/v2/state`` ``bindings[].routable``, the function its replica-floor check uses,
    2026-10-02); without it (``routable_ids`` None) the controller's former count:
    awake and not hidden."""
    if cluster_view is None:
        return {}
    routable_ids = getattr(cluster_view, "routable_ids", None)
    counts: dict[str, list[int]] = {}
    for binding in cluster_view.bindings:
        model_counts = counts.setdefault(binding.model, [0, 0])
        if not binding.hidden:
            model_counts[1] += 1
        if _is_routable(binding, routable_ids):
            model_counts[0] += 1
    return {model: (values[0], values[1]) for model, values in counts.items()}


def _sm_floor_context(cluster_view: ClusterView | None, model: str) -> dict:
    """``floor`` / ``floor_headroom`` of ``model`` from the SM view (only while the SM
    routable view is readable: the planner then bounds every donor by it)."""
    if cluster_view is None or getattr(cluster_view, "routable_ids", None) is None:
        return {}
    entry = (getattr(cluster_view, "model_floors", None) or {}).get(model)
    headroom = getattr(entry, "floor_headroom", None)
    if headroom is None:
        return {}
    return {"floor": getattr(entry, "floor", None), "floor_headroom": int(headroom)}


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

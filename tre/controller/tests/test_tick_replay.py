from __future__ import annotations

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics, PodWindowMetrics
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_controller.config import SafeScaleConfig
from tre_controller.loops.replay import TickReplayStep, run_tick_replay
from tre_controller.planning.planner import ClusterView, DefragAction, HideAction, ScaleAction
from tre_controller.planning.safescale import SafeScaleStateMachine
from tre_sm.allocator.slots import Binding, Migration, Slot


def _registry_with_specs(*specs: ModelSpec) -> Registry:
    return Registry(
        ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)),
        list(specs),
    )


def _spec(name: str, *, tp_size: int = 1, min_replicas: int = 0, max_replicas: int = 4) -> ModelSpec:
    return ModelSpec(
        name=name,
        weights_path="/weights",
        tp_size=tp_size,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        vllm_image="image",
        slo=SloSpec(ttft_p95_ms=1200.0, tpot_p95_ms=100.0, e2e_p95_ms=10000.0),
        trs=TrsParams(
            w_p=0.04,
            w_d=1.0,
            lambda_wait=2.625,
            qmin=1.0,
            ema_alpha=0.0,
            theta_m=100.0,
            tau_crit=0.8,
            tau_low=1.0,
            tau_high=1.25,
            qsat=4.0,
            epsat=0.05,
            hsat=1,
        ),
    )


def _pod_metrics(pod: str) -> PodWindowMetrics:
    return PodWindowMetrics(
        pod=pod,
        prompt_tokens=0.0,
        generation_tokens=10.0,
        avg_waiting=0.0,
        avg_running=1.0,
        avg_swapping=0.0,
        kv_cache_hit_rate=0.0,
        ttft_p95_ms=100.0,
        tpot_p95_ms=10.0,
        e2e_p95_ms=1000.0,
    )


def _metrics(
    model: str,
    *,
    generation: float,
    waiting: float,
    running: float,
    assigned: int = 1,
    pods: tuple[str, ...] = (),
) -> ModelWindowMetrics:
    per_pod = {pod: _pod_metrics(pod) for pod in pods}
    return ModelWindowMetrics(
        model=model,
        window_start_ms=0,
        window_end_ms=60_000,
        prompt_tokens=0.0,
        generation_tokens=generation,
        avg_waiting=waiting,
        avg_running=running,
        avg_swapping=0.0,
        kv_cache_hit_rate=0.0,
        ttft_p95_ms=100.0,
        tpot_p95_ms=10.0,
        e2e_p95_ms=1000.0,
        routable_pods=assigned,
        assigned_replicas=assigned,
        per_pod=per_pod,
    )


def _steps(snapshot: MetricsSnapshot) -> tuple[TickReplayStep, ...]:
    return tuple(TickReplayStep(snapshot=snapshot, rescue_due=True, fairness_due=True) for _ in range(60))


def test_tick_replay_runs_critical_scale_sequence_for_60_ticks() -> None:
    registry = _registry_with_specs(_spec("critical"))
    snapshot = MetricsSnapshot(
        ts_ms=1_000,
        stale=False,
        models={"critical": _metrics("critical", generation=50.0, waiting=10.0, running=1.0, assigned=1)},
    )

    result = run_tick_replay(_steps(snapshot), registry=registry)

    assert len(result.results) == 60
    assert any(
        isinstance(action, ScaleAction)
        and action.model == "critical"
        and action.delta == 1
        and action.reason == "critical_idle_capacity"
        for action in result.actions
    )


def test_tick_replay_records_high_model_safescale_probe_once_across_60_ticks() -> None:
    registry = _registry_with_specs(_spec("donor", min_replicas=0))
    snapshot = MetricsSnapshot(
        ts_ms=10_000,
        stale=False,
        models={
            "donor": _metrics(
                "donor",
                generation=200.0,
                waiting=0.0,
                running=1.0,
                assigned=2,
                pods=("donor-a", "donor-b"),
            )
        },
    )
    # t1 (default guard ON): a hot HIGH model with no receiver must NEVER get a proactive
    # scale-down probe -- across all 60 ticks there is no hide and the model stays intact.
    guarded = SafeScaleStateMachine(config=SafeScaleConfig(default_window_ms=60_000.0))
    result = run_tick_replay(_steps(snapshot), registry=registry, safescale=guarded)
    assert [action for action in result.actions if isinstance(action, HideAction)] == []
    assert guarded.active_probe("donor") is None
    assert result.events.count("safescale_probe_suppressed_hot:donor") == 60

    # Ablation path (guard OFF): the legacy proactive probe fires exactly once and is not
    # re-issued while it stays active (active_probe_models idempotency across 60 ticks).
    legacy = SafeScaleStateMachine(config=SafeScaleConfig(default_window_ms=60_000.0))
    result = run_tick_replay(
        _steps(snapshot), registry=registry, safescale=legacy, suppress_hot_proactive_probe=False
    )
    hide_actions = [action for action in result.actions if isinstance(action, HideAction)]
    assert hide_actions == [HideAction("donor", ("donor-a",), "probe_started", "rescue")]
    assert legacy.active_probe("donor") is not None
    assert result.events.count("safescale_probe_started:donor") == 1


def test_tick_replay_records_tp_defrag_sequence_for_60_ticks() -> None:
    registry = _registry_with_specs(_spec("critical", tp_size=2))
    snapshot = MetricsSnapshot(
        ts_ms=1_000,
        stale=False,
        models={"critical": _metrics("critical", generation=50.0, waiting=10.0, running=1.0, assigned=0)},
    )
    cluster_view = ClusterView(
        topology=registry.topology(),
        bindings=(
            Binding("serve-0", "small", Slot("node-a", (0,)), awake=True),
            Binding("serve-2", "small", Slot("node-a", (2,)), awake=True),
        ),
    )
    steps = tuple(TickReplayStep(snapshot=snapshot, rescue_due=True, fairness_due=False, cluster_view=cluster_view) for _ in range(60))

    result = run_tick_replay(steps, registry=registry)

    defrag = next(action for action in result.actions if isinstance(action, DefragAction))
    scale = next(action for action in result.actions if isinstance(action, ScaleAction))
    assert defrag.migrations == (
        Migration("serve-2", Slot("node-a", (2,)), Slot("node-a", (1,))),
    )
    assert scale.reason == "critical_tp_defrag"

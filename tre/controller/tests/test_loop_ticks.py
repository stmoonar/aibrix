from __future__ import annotations

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics, PodWindowMetrics
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_controller.loops.fairness_task import run_fairness_tick
from tre_controller.loops.rescue_task import run_rescue_tick
from tre_controller.config import SafeScaleConfig
from tre_controller.planning.planner import HideAction, ScaleAction
from tre_controller.planning.safescale import SafeScaleStateMachine


class FakeQueue:
    def __init__(self, inflight: set[str] | None = None) -> None:
        self.submitted: list[tuple] = []
        self._inflight = inflight or set()

    def inflight_models(self) -> set[str]:
        return set(self._inflight)

    def submit(self, actions) -> object:
        self.submitted.append(tuple(actions))
        return object()


def _registry() -> Registry:
    slo = SloSpec(ttft_p95_ms=1200.0, tpot_p95_ms=100.0, e2e_p95_ms=10000.0)
    common = dict(
        weights_path="/weights",
        tp_size=1,
        min_replicas=0,
        max_replicas=4,
        vllm_image="image",
        slo=slo,
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
    return Registry(
        ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)),
        [ModelSpec(name="critical", **common)],
    )


def _metrics(
    model: str,
    *,
    generation: float,
    waiting: float,
    running: float,
    assigned: int = 1,
    routable: int | None = None,
) -> ModelWindowMetrics:
    routable = assigned if routable is None else routable
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
        routable_pods=routable,
        assigned_replicas=assigned,
        per_pod={},
    )




def _registry_with_models(*names: str) -> Registry:
    base = _registry()
    template = base.model("critical")
    return Registry(
        base.topology(),
        [
            ModelSpec(
                name=name,
                weights_path=template.weights_path,
                tp_size=template.tp_size,
                min_replicas=template.min_replicas,
                max_replicas=template.max_replicas,
                vllm_image=template.vllm_image,
                slo=template.slo,
                trs=template.trs,
            )
            for name in names
        ],
    )


def _registry_with_model_bounds(bounds: dict[str, tuple[int, int]]) -> Registry:
    base = _registry()
    template = base.model("critical")
    return Registry(
        base.topology(),
        [
            ModelSpec(
                name=name,
                weights_path=template.weights_path,
                tp_size=template.tp_size,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                vllm_image=template.vllm_image,
                slo=template.slo,
                trs=template.trs,
            )
            for name, (min_replicas, max_replicas) in bounds.items()
        ],
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


def _metrics_with_pods(model: str, *, generation: float, waiting: float, running: float, pods: tuple[str, ...]) -> ModelWindowMetrics:
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
        routable_pods=len(pods),
        assigned_replicas=len(pods),
        per_pod=per_pod,
    )


def test_rescue_tick_skips_stale_snapshot() -> None:
    queue = FakeQueue()
    snapshot = MetricsSnapshot(ts_ms=1, models={}, stale=True)

    result = run_rescue_tick(snapshot, queue=queue, registry=_registry())

    assert result.submitted == 0
    assert result.events == ("snapshot_stale",)
    assert queue.submitted == []


def test_rescue_tick_submits_critical_scale_action_from_snapshot_metrics() -> None:
    queue = FakeQueue()
    snapshot = MetricsSnapshot(
        ts_ms=1,
        stale=False,
        models={"critical": _metrics("critical", generation=50.0, waiting=10.0, running=1.0, assigned=1)},
    )

    result = run_rescue_tick(snapshot, queue=queue, registry=_registry())

    assert result.submitted == 1
    assert len(queue.submitted) == 1
    action = queue.submitted[0][0]
    assert isinstance(action, ScaleAction)
    assert action.model == "critical"
    assert action.delta == 1
    assert action.source_loop == "rescue"


def test_rescue_tick_converts_safescale_required_downscale_to_probe_hide() -> None:
    queue = FakeQueue()
    safescale = SafeScaleStateMachine(config=SafeScaleConfig(default_window_ms=60_000.0))
    registry = _registry_with_models("critical", "donor")
    snapshot = MetricsSnapshot(
        ts_ms=10_000,
        stale=False,
        models={
            "critical": _metrics_with_pods(
                "critical", generation=50.0, waiting=10.0, running=1.0, pods=("critical-a", "critical-b")
            ),
            "donor": _metrics_with_pods(
                "donor", generation=100.0, waiting=0.0, running=1.0, pods=("donor-a", "donor-b")
            ),
        },
    )

    result = run_rescue_tick(snapshot, queue=queue, registry=registry, safescale=safescale)

    assert result.submitted == 1
    assert len(queue.submitted) == 1
    assert queue.submitted[0] == (HideAction("donor", ("donor-a",), "probe_started", "rescue"),)
    assert result.actions == queue.submitted[0]
    assert safescale.active_probe("donor").pending_upscales == {"critical": 1}
    assert "safescale_probe_started:donor" in result.events


def test_rescue_tick_honors_latency_signal_source_for_classification() -> None:
    queue = FakeQueue()
    snapshot = MetricsSnapshot(
        ts_ms=1,
        stale=False,
        models={"critical": _metrics("critical", generation=50.0, waiting=10.0, running=1.0, assigned=1)},
    )

    result = run_rescue_tick(snapshot, queue=queue, registry=_registry(), signal_source="latency_p95")

    assert result.submitted == 1
    action = queue.submitted[0][0]
    assert isinstance(action, ScaleAction)
    assert action.model == "critical"
    assert action.delta == -1
    assert action.source_loop == "rescue"


def test_rescue_tick_honors_per_model_min_replicas_for_idle_model() -> None:
    queue = FakeQueue()
    registry = _registry_with_model_bounds({"warm": (1, 4), "cold": (0, 4)})
    snapshot = MetricsSnapshot(
        ts_ms=1,
        stale=False,
        models={"warm": _metrics("warm", generation=0.0, waiting=0.0, running=0.0, assigned=4, routable=1)},
    )

    result = run_rescue_tick(snapshot, queue=queue, registry=registry)

    assert result.submitted == 0
    assert queue.submitted == []


def test_fairness_tick_passes_inflight_models_to_planner() -> None:
    queue = FakeQueue(inflight={"critical"})
    snapshot = MetricsSnapshot(
        ts_ms=1,
        stale=False,
        models={"critical": _metrics("critical", generation=50.0, waiting=10.0, running=1.0, assigned=1)},
    )

    result = run_fairness_tick(snapshot, queue=queue, registry=_registry())

    assert result.submitted == 0
    assert queue.submitted == []


class StopLoop(Exception):
    pass


async def _stop_sleep(seconds: float) -> None:
    _stop_sleep.calls.append(seconds)
    raise StopLoop


_stop_sleep.calls = []


class FakeDecisionWriter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, MetricsSnapshot, object]] = []

    def write(self, loop_name: str, snapshot: MetricsSnapshot, result: object) -> None:
        self.calls.append((loop_name, snapshot, result))


def test_rescue_task_loop_reads_snapshot_and_sleeps_configured_interval() -> None:
    import asyncio

    from tre_controller.loops.metrics_task import SnapshotBox
    from tre_controller.loops.rescue_task import rescue_task

    _stop_sleep.calls = []
    queue = FakeQueue()
    snapshot = MetricsSnapshot(
        ts_ms=1,
        stale=False,
        models={"critical": _metrics("critical", generation=50.0, waiting=10.0, running=1.0, assigned=1)},
    )
    cfg = type("Cfg", (), {"rescue_interval_s": 5.0})()

    try:
        asyncio.run(
            rescue_task(
                SnapshotBox(snapshot),
                queue=queue,
                registry=_registry(),
                cfg=cfg,
                sleep=_stop_sleep,
            )
        )
    except StopLoop:
        pass

    assert _stop_sleep.calls == [5.0]
    assert len(queue.submitted) == 1


def test_rescue_task_loop_records_decision_snapshot_after_tick() -> None:
    import asyncio

    from tre_controller.loops.metrics_task import SnapshotBox
    from tre_controller.loops.rescue_task import rescue_task

    _stop_sleep.calls = []
    queue = FakeQueue()
    writer = FakeDecisionWriter()
    snapshot = MetricsSnapshot(
        ts_ms=1,
        stale=False,
        models={"critical": _metrics("critical", generation=50.0, waiting=10.0, running=1.0, assigned=1)},
    )
    cfg = type("Cfg", (), {"rescue_interval_s": 5.0})()

    try:
        asyncio.run(
            rescue_task(
                SnapshotBox(snapshot),
                queue=queue,
                registry=_registry(),
                cfg=cfg,
                sleep=_stop_sleep,
                decision_writer=writer,
            )
        )
    except StopLoop:
        pass

    assert len(writer.calls) == 1
    loop_name, seen_snapshot, result = writer.calls[0]
    assert loop_name == "rescue"
    assert seen_snapshot is snapshot
    assert result.submitted == 1


def test_fairness_task_loop_records_decision_snapshot_after_tick() -> None:
    import asyncio

    from tre_controller.loops.fairness_task import fairness_task
    from tre_controller.loops.metrics_task import SnapshotBox

    _stop_sleep.calls = []
    queue = FakeQueue()
    writer = FakeDecisionWriter()
    snapshot = MetricsSnapshot(
        ts_ms=2,
        stale=True,
        models={"critical": _metrics("critical", generation=50.0, waiting=10.0, running=1.0, assigned=1)},
    )
    cfg = type("Cfg", (), {"fairness_interval_s": 10.0})()

    try:
        asyncio.run(
            fairness_task(
                SnapshotBox(snapshot),
                queue=queue,
                registry=_registry(),
                cfg=cfg,
                sleep=_stop_sleep,
                decision_writer=writer,
            )
        )
    except StopLoop:
        pass

    assert len(writer.calls) == 1
    loop_name, seen_snapshot, result = writer.calls[0]
    assert loop_name == "fairness"
    assert seen_snapshot is snapshot
    assert result.events == ("snapshot_stale",)


def test_fairness_task_loop_skips_missing_snapshot_and_sleeps_configured_interval() -> None:
    import asyncio

    from tre_controller.loops.fairness_task import fairness_task
    from tre_controller.loops.metrics_task import SnapshotBox

    _stop_sleep.calls = []
    queue = FakeQueue()
    cfg = type("Cfg", (), {"fairness_interval_s": 10.0})()

    try:
        asyncio.run(
            fairness_task(
                SnapshotBox(),
                queue=queue,
                registry=_registry(),
                cfg=cfg,
                sleep=_stop_sleep,
            )
        )
    except StopLoop:
        pass

    assert _stop_sleep.calls == [10.0]
    assert queue.submitted == []


def test_rescue_task_loop_uses_latest_cluster_view_from_box() -> None:
    import asyncio

    from tre_controller.loops.cluster_view_task import ClusterViewBox
    from tre_controller.loops.metrics_task import SnapshotBox
    from tre_controller.loops.rescue_task import rescue_task
    from tre_controller.planning.planner import ClusterView, DefragAction, ScaleAction
    from tre_sm.allocator.slots import Binding, Migration, Slot

    base = _registry()
    tp2_spec = base.model("critical")
    tp2_spec = type(tp2_spec)(
        name=tp2_spec.name,
        weights_path=tp2_spec.weights_path,
        tp_size=2,
        min_replicas=tp2_spec.min_replicas,
        max_replicas=tp2_spec.max_replicas,
        vllm_image=tp2_spec.vllm_image,
        slo=tp2_spec.slo,
        trs=tp2_spec.trs,
    )
    registry = Registry(base.topology(), [tp2_spec])
    queue = FakeQueue()
    snapshot = MetricsSnapshot(
        ts_ms=1,
        stale=False,
        models={"critical": _metrics("critical", generation=50.0, waiting=10.0, running=1.0, assigned=0)},
    )
    view = ClusterView(
        topology=registry.topology(),
        bindings=(
            Binding("serve-0", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-2", "m1", Slot("node-a", (2,)), awake=True),
        ),
    )
    cfg = type("Cfg", (), {"rescue_interval_s": 5.0})()
    _stop_sleep.calls = []

    try:
        asyncio.run(
            rescue_task(
                SnapshotBox(snapshot),
                queue=queue,
                registry=registry,
                cfg=cfg,
                sleep=_stop_sleep,
                cluster_view_box=ClusterViewBox(view),
            )
        )
    except StopLoop:
        pass

    actions = queue.submitted[0]
    defrag = next(action for action in actions if isinstance(action, DefragAction))
    scale = next(action for action in actions if isinstance(action, ScaleAction))
    assert defrag.migrations == (
        Migration("serve-2", Slot("node-a", (2,)), Slot("node-a", (1,))),
    )
    assert scale.reason == "critical_tp_defrag"
    assert _stop_sleep.calls == [5.0]

def _registry_for_same_slot_shrink() -> Registry:
    base = _registry()
    template = base.model("critical")
    return Registry(
        base.topology(),
        [
            ModelSpec(
                name="tp2",
                weights_path=template.weights_path,
                tp_size=2,
                min_replicas=0,
                max_replicas=2,
                vllm_image=template.vllm_image,
                slo=template.slo,
                trs=template.trs,
            ),
            ModelSpec(
                name="high",
                weights_path=template.weights_path,
                tp_size=1,
                min_replicas=0,
                max_replicas=2,
                vllm_image=template.vllm_image,
                slo=template.slo,
                trs=template.trs,
            ),
        ],
    )


def test_rescue_tick_converts_same_slot_shrink_to_safescale_probe_hide() -> None:
    from tre_controller.planning.planner import ClusterView
    from tre_sm.allocator.slots import Binding, Slot

    queue = FakeQueue()
    safescale = SafeScaleStateMachine(config=SafeScaleConfig(default_window_ms=60_000.0))
    registry = _registry_for_same_slot_shrink()
    snapshot = MetricsSnapshot(
        ts_ms=20_000,
        stale=False,
        models={
            "tp2": _metrics("tp2", generation=50.0, waiting=10.0, running=1.0, assigned=0),
            "high": _metrics_with_pods("high", generation=200.0, waiting=0.0, running=1.0, pods=("high-0",)),
        },
    )
    cluster_view = ClusterView(
        topology=registry.topology(),
        bindings=(
            Binding("high-0", "high", Slot("node-a", (0,)), awake=True),
            Binding("other-2", "other", Slot("node-a", (2,)), awake=True),
        ),
    )

    result = run_rescue_tick(
        snapshot,
        queue=queue,
        registry=registry,
        cluster_view=cluster_view,
        safescale=safescale,
    )

    assert result.submitted == 1
    assert queue.submitted == [(HideAction("high", ("high-0",), "probe_started", "rescue"),)]
    assert safescale.active_probe("high").pending_upscales == {"tp2": 1}
    assert "safescale_probe_started:high" in result.events

from __future__ import annotations

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_controller.loops.fairness_task import run_fairness_tick
from tre_controller.loops.rescue_task import run_rescue_tick
from tre_controller.planning.planner import ScaleAction


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


def _metrics(model: str, *, generation: float, waiting: float, running: float, assigned: int = 1) -> ModelWindowMetrics:
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
        per_pod={},
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

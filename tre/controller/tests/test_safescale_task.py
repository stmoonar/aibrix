from __future__ import annotations

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_controller.config import SafeScaleConfig
from tre_controller.loops.action_queue import SubmitResult
from tre_controller.loops.safescale_task import run_safescale_observation_tick
from tre_controller.planning.planner import ScaleAction, UnhideAction
from tre_controller.planning.safescale import SafeScaleStateMachine


class FakeQueue:
    def __init__(self) -> None:
        self.submitted: list[tuple] = []

    def submit(self, actions) -> SubmitResult:
        self.submitted.append(tuple(actions))
        return SubmitResult(accepted=len(actions))


def _registry() -> Registry:
    spec = ModelSpec(
        name="donor",
        weights_path="/weights",
        tp_size=1,
        min_replicas=0,
        max_replicas=4,
        vllm_image="image",
        slo=SloSpec(ttft_p95_ms=1000.0, tpot_p95_ms=100.0, e2e_p95_ms=10_000.0),
        trs=TrsParams(
            w_p=0.04,
            w_d=1.0,
            lambda_wait=2.0,
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
    return Registry(ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)), [spec])


def _metrics(*, ts_ms: int, generation: float = 120.0, ttft: float = 500.0, tpot: float = 50.0) -> MetricsSnapshot:
    return MetricsSnapshot(
        ts_ms=ts_ms,
        stale=False,
        models={
            "donor": ModelWindowMetrics(
                model="donor",
                window_start_ms=0,
                window_end_ms=60_000,
                prompt_tokens=0.0,
                generation_tokens=generation,
                avg_waiting=0.0,
                avg_running=1.0,
                avg_swapping=0.0,
                kv_cache_hit_rate=0.0,
                ttft_p95_ms=ttft,
                tpot_p95_ms=tpot,
                e2e_p95_ms=1000.0,
                routable_pods=1,
                assigned_replicas=1,
                per_pod={},
            )
        },
    )


def _machine() -> SafeScaleStateMachine:
    return SafeScaleStateMachine(
        config=SafeScaleConfig(
            ttft_p95_slo_ms=1000.0,
            tpot_p95_slo_ms=100.0,
            default_window_ms=1000.0,
            hq=0.5,
            tau_low=1.0,
        )
    )


def test_safescale_observation_tick_submits_commit_actions_after_deadline() -> None:
    queue = FakeQueue()
    machine = _machine()
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=0, pending_upscales={"receiver": 1})

    pending = run_safescale_observation_tick(_metrics(ts_ms=500), queue=queue, registry=_registry(), safescale=machine)
    committed = run_safescale_observation_tick(_metrics(ts_ms=1000), queue=queue, registry=_registry(), safescale=machine)

    assert pending.submitted == 0
    assert pending.events == ("safescale_probe_pending:donor",)
    assert committed.submitted == 2
    assert queue.submitted == [
        (
            ScaleAction("donor", -1, "formal_commit_gate_passed", "safescale"),
            ScaleAction("receiver", 1, "safescale_followup_upscale", "safescale"),
        )
    ]
    assert committed.events == ("safescale_formal_commit_gate_passed:donor",)


def test_safescale_observation_tick_submits_rollback_unhide_on_slo_violation() -> None:
    queue = FakeQueue()
    machine = _machine()
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=0)

    result = run_safescale_observation_tick(
        _metrics(ts_ms=500, ttft=1500.0),
        queue=queue,
        registry=_registry(),
        safescale=machine,
    )

    assert result.submitted == 1
    assert queue.submitted == [(UnhideAction("donor", ("pod-a",), "slo_violation", "safescale"),)]
    assert result.events == ("safescale_slo_violation:donor",)

"""The fitted per-model control margins in registry.yaml must reach classification.

``classify_all_models`` accepts ``model_control_configs``; for a long time the planner
tick never passed it, so ``trs.tau_crit`` / ``trs.tau_high`` were dead keys and every
model was classified against the generic delta_crit=0.2 / delta_high=0.25 defaults.
"""
from __future__ import annotations

import pytest

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_controller.loops.tick import run_planner_tick
from tre_controller.planning.classify import (
    ModelState,
    TauThresholds,
    classify_all_models,
    model_control_configs_from_registry,
)

MODEL = "sample"


class FakeQueue:
    def inflight_models(self) -> set[str]:
        return set()

    def submit(self, actions) -> object:
        return object()


def _registry(*, tau_crit: float, tau_high: float) -> Registry:
    return Registry(
        ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)),
        [
            ModelSpec(
                name=MODEL,
                weights_path="/weights",
                tp_size=1,
                min_replicas=0,
                max_replicas=4,
                vllm_image="image",
                slo=SloSpec(ttft_p95_ms=1200.0, tpot_p95_ms=100.0, e2e_p95_ms=10000.0),
                trs=TrsParams(
                    w_p=0.04,
                    w_d=1.0,
                    lambda_wait=2.625,
                    qmin=1.0,
                    ema_alpha=0.0,
                    theta_m=100.0,
                    tau_crit=tau_crit,
                    tau_low=1.0,
                    tau_high=tau_high,
                    qsat=4.0,
                    epsat=0.05,
                    hsat=1,
                ),
            )
        ],
    )


def _snapshot() -> MetricsSnapshot:
    return MetricsSnapshot(
        ts_ms=1,
        stale=False,
        models={
            MODEL: ModelWindowMetrics(
                model=MODEL,
                window_start_ms=0,
                window_end_ms=60_000,
                prompt_tokens=0.0,
                generation_tokens=200.0,
                avg_waiting=1.0,
                avg_running=2.0,
                avg_swapping=0.0,
                kv_cache_hit_rate=0.0,
                ttft_p95_ms=100.0,
                tpot_p95_ms=10.0,
                e2e_p95_ms=1000.0,
                routable_pods=1,
                assigned_replicas=1,
                per_pod={},
            )
        },
    )


def _classify_with(registry: Registry):
    result = run_planner_tick(
        _snapshot(),
        queue=FakeQueue(),
        registry=registry,
        rescue_due=True,
        fairness_due=False,
    )
    return result.classifications[MODEL], result.model_contexts[MODEL]["z_m"]


def test_per_model_tau_crit_from_the_registry_changes_the_planner_classification() -> None:
    # Straddle whatever z_m this snapshot produces, so the test states the mechanism
    # rather than a hard-coded signal value.
    _baseline, z_m = _classify_with(_registry(tau_crit=0.8, tau_high=1.25))
    assert z_m is not None and 0.0 < z_m < 1.0

    lenient, _ = _classify_with(_registry(tau_crit=z_m - 0.05, tau_high=1.25))
    strict, _ = _classify_with(_registry(tau_crit=z_m + 0.05, tau_high=1.25))

    assert lenient.tau.tau_crit == z_m - 0.05
    assert strict.tau.tau_crit == z_m + 0.05
    assert lenient.state is ModelState.LOW
    assert strict.state is ModelState.CRITICAL


def test_per_model_tau_high_from_the_registry_reaches_the_thresholds() -> None:
    classification, _z = _classify_with(_registry(tau_crit=0.8, tau_high=1.42))

    assert classification.tau.tau_high == 1.42


def test_control_configs_are_margins_around_tau_low() -> None:
    configs = model_control_configs_from_registry(_registry(tau_crit=0.75, tau_high=1.4))

    assert configs[MODEL]["delta_crit"] == pytest.approx(0.25)
    assert configs[MODEL]["delta_high"] == pytest.approx(0.4)


def test_models_without_a_registry_entry_keep_the_documented_defaults() -> None:
    contexts = {"unregistered": {"trs": 10.0, "z_m": 0.9, "theta_m": 100.0, "eta_m": 500.0}}

    classification = classify_all_models(
        contexts,
        model_control_configs=model_control_configs_from_registry(
            _registry(tau_crit=0.75, tau_high=1.4)
        ),
    )[0]

    assert classification.tau == TauThresholds.from_control()
    assert classification.tau.tau_crit == 0.8
    assert classification.tau.tau_high == 1.25

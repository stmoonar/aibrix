"""Online side of the unified TSS (plan 2026-09-21 §6.4): the controller computes exactly
``tre_common.tss``, offline smoothing is bitwise the online EMA, and the idle rule never
yields CRITICAL."""
from __future__ import annotations

import random

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_common.tss import smooth_series, tss_terms
from tre_controller.loops.tick import _model_contexts
from tre_controller.planning.classify import ModelState, classify_all_models
from tre_controller.planning.planner import _paper_state_incomplete_models
from tre_controller.signals.trs import TRSComputer, TRSInput


def _window(end_ms: int, *, prompt: float, gen: float, running: float, waiting: float) -> ModelWindowMetrics:
    return ModelWindowMetrics(
        model="m", window_start_ms=end_ms - 30_000, window_end_ms=end_ms,
        prompt_tokens=prompt, generation_tokens=gen, avg_waiting=waiting, avg_running=running,
        avg_swapping=0.0, kv_cache_hit_rate=0.0, ttft_p95_ms=100.0, tpot_p95_ms=10.0,
        e2e_p95_ms=1000.0, routable_pods=1, assigned_replicas=1, per_pod={},
    )


def _params(theta: float = 50.0) -> TrsParams:
    return TrsParams(
        w_p=0.02, w_d=1.0, lambda_wait=3.0, qmin=1.0, ema_alpha=0.2485, theta_m=theta,
        tau_crit=0.8, tau_low=1.0, tau_high=1.25, qsat=4.0, epsat=0.05, hsat=3,
        ema_tau_ms=20_000,
    )


def test_online_raw_is_the_shared_definition() -> None:
    wm = _window(60_000, prompt=3000.0, gen=6000.0, running=4.0, waiting=1.0)
    result = TRSComputer(ema_tau_ms=20_000).compute(TRSInput.from_metrics(wm, _params()), window_end_ms=60_000)
    expected = tss_terms(
        prompt_tokens=3000.0, generation_tokens=6000.0, avg_running=4.0,
        avg_waiting=1.0, w_p=0.02, lambda_wait=3.0, qmin=1.0,
    )
    assert result.TRS_raw == expected.raw
    assert result.Y_m == expected.numerator  # window total (tokens per window)


def test_offline_smoothing_is_bitwise_the_online_ema() -> None:
    rng = random.Random(7)
    computer = TRSComputer(ema_tau_ms=20_000)
    online, raws, ends = [], [], []
    end = 30_000
    for i in range(200):
        end += rng.choice((5_000, 5_000, 10_000))
        idle = rng.random() < 0.1
        wm = _window(
            end, prompt=rng.uniform(0, 5e4), gen=rng.uniform(0, 3e4),
            running=0.0 if idle else rng.uniform(0.1, 60), waiting=0.0 if idle else rng.uniform(0, 20),
        )
        result = computer.compute(TRSInput.from_metrics(wm, _params()), window_end_ms=end)
        online.append(result.TRS if result.defined else None)
        raws.append(result.TRS_raw if result.defined else None)
        ends.append(end)
    assert smooth_series(raws, ends, tau_ms=20_000.0) == online  # exact, not approx


def _registry(theta: float) -> Registry:
    slo = SloSpec(ttft_p95_ms=500.0, tpot_p95_ms=75.0, e2e_p95_ms=12000.0)
    return Registry(
        ClusterTopology(nodes=(NodeSpec(name="n", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)),
        [ModelSpec(name="m", weights_path="/w", tp_size=1, min_replicas=1, max_replicas=4,
                   vllm_image="img", slo=slo, trs=_params(theta))],
    )


def test_idle_rule_tokens_but_nothing_in_flight_is_not_critical_and_not_dropped() -> None:
    # A light model: 3 short requests completed in the window, but every instant sample
    # saw nothing in flight. Under the old total-units formula this read as a tiny TSS and
    # therefore CRITICAL; now Z is undefined and the model is HEALTHY (surplus side).
    wm = _window(60_000, prompt=768.0, gen=384.0, running=0.0, waiting=0.0)
    wm = ModelWindowMetrics(**{**wm.__dict__, "request_count": 3.0})
    contexts, _ = _model_contexts(MetricsSnapshot(ts_ms=60_000, models={"m": wm}, stale=False), _registry(50.0))
    ctx = contexts["m"]
    assert ctx["tss_defined"] is False and ctx["z_m"] is None
    [cls] = classify_all_models(contexts)
    assert cls.state == ModelState.HEALTHY and cls.state != ModelState.CRITICAL
    assert cls.signal_idle is True
    assert _paper_state_incomplete_models([cls]) == ()


def test_in_flight_work_still_classifies_on_z() -> None:
    # 1200 tokens per 30 s window per in-flight request against theta 1500 -> Z 0.8 - LOW.
    wm = _window(60_000, prompt=0.0, gen=1200.0 * 2.0, running=2.0, waiting=0.0)
    contexts, _ = _model_contexts(MetricsSnapshot(ts_ms=60_000, models={"m": wm}, stale=False), _registry(1500.0))
    assert contexts["m"]["tss_defined"] is True
    assert abs(contexts["m"]["z_m"] - 0.8) < 1e-12

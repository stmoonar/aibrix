"""S1.3 loop-level: a shared SignalState makes the TRS EMA persist across ticks
and be shared by rescue/fairness (one EMA per model). Without a SignalState the
behaviour is unchanged (fresh computer per tick -> TRS == raw)."""

from __future__ import annotations

import math

import pytest

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_controller.loops.fairness_task import run_fairness_tick
from tre_controller.loops.rescue_task import run_rescue_tick
from tre_controller.signals.trs import SignalState


class _Queue:
    def inflight_models(self) -> set[str]:
        return set()

    def submit(self, actions) -> object:
        return object()


def _registry(tau_ms: float | None) -> Registry:
    slo = SloSpec(ttft_p95_ms=1200.0, tpot_p95_ms=100.0, e2e_p95_ms=10_000.0)
    trs = TrsParams(
        w_p=0.04,
        w_d=1.0,
        lambda_wait=2.625,
        qmin=1.0,
        ema_alpha=0.5,
        theta_m=100.0,
        tau_crit=0.8,
        tau_low=1.0,
        tau_high=1.25,
        qsat=4.0,
        epsat=0.05,
        hsat=1,
        ema_tau_ms=tau_ms,
    )
    return Registry(
        ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)),
        [
            ModelSpec(
                name="m",
                weights_path="/w",
                tp_size=1,
                min_replicas=0,
                max_replicas=4,
                vllm_image="img",
                slo=slo,
                trs=trs,
            )
        ],
    )


def _snapshot(*, window_end_ms: int, generation: float, running: float) -> MetricsSnapshot:
    metrics = ModelWindowMetrics(
        model="m",
        window_start_ms=window_end_ms - 30_000,
        window_end_ms=window_end_ms,
        prompt_tokens=0.0,
        generation_tokens=generation,
        avg_waiting=0.0,
        avg_running=running,
        avg_swapping=0.0,
        kv_cache_hit_rate=0.0,
        ttft_p95_ms=100.0,
        tpot_p95_ms=10.0,
        e2e_p95_ms=1000.0,
        routable_pods=1,
        assigned_replicas=1,
        per_pod={},
    )
    return MetricsSnapshot(ts_ms=window_end_ms, models={"m": metrics}, stale=False)


def _snapshot_tokens_missing(*, window_end_ms: int) -> MetricsSnapshot:
    metrics = ModelWindowMetrics(
        model="m", window_start_ms=window_end_ms - 30_000, window_end_ms=window_end_ms,
        prompt_tokens=None, generation_tokens=None, avg_waiting=0.0, avg_running=1.0, avg_swapping=0.0,
        kv_cache_hit_rate=0.0, ttft_p95_ms=None, tpot_p95_ms=None, e2e_p95_ms=None,
        routable_pods=1, assigned_replicas=1, per_pod={},
    )
    return MetricsSnapshot(ts_ms=window_end_ms, models={"m": metrics}, stale=False)


def test_tokens_missing_holds_traffic_onset_cursor() -> None:
    # F1: a metrics scrape gap (tokens missing) must NOT reset the traffic-onset cursor,
    # else a genuinely-loaded model is re-suppressed for a full window after metrics recover.
    from tre_controller.loops.tick import _model_contexts

    state = SignalState(warmup_ms=-1)
    registry = _registry(20_000)
    _model_contexts(_snapshot(window_end_ms=60_000, generation=200.0, running=2.0), registry, signal_state=state)
    assert state._onset_ms["m"] == 60_000
    # scrape gap: tokens missing -> onset held (not reset)
    _model_contexts(_snapshot_tokens_missing(window_end_ms=65_000), registry, signal_state=state)
    assert state._onset_ms["m"] == 60_000
    # genuine idle (tokens present, zero) DOES reset
    _model_contexts(_snapshot(window_end_ms=70_000, generation=0.0, running=0.0), registry, signal_state=state)
    assert state._onset_ms["m"] is None


def test_shared_signal_state_persists_ema_across_ticks() -> None:
    registry = _registry(20_000)
    state = SignalState()
    queue = _Queue()
    r1 = run_rescue_tick(
        _snapshot(window_end_ms=60_000, generation=200.0, running=2.0),
        queue=queue,
        registry=registry,
        signal_state=state,
    )
    assert r1.model_contexts["m"]["trs"] == pytest.approx(100.0)  # seed = raw

    r2 = run_rescue_tick(
        _snapshot(window_end_ms=65_000, generation=400.0, running=2.0),
        queue=queue,
        registry=registry,
        signal_state=state,
    )
    decay = math.exp(-5_000 / 20_000)
    expected = decay * 100.0 + (1 - decay) * 200.0
    assert r2.model_contexts["m"]["trs"] == pytest.approx(expected)
    assert r2.model_contexts["m"]["trs"] != pytest.approx(200.0)  # not raw


def test_rescue_then_fairness_same_snapshot_no_double_advance() -> None:
    registry = _registry(20_000)
    state = SignalState()
    queue = _Queue()
    run_rescue_tick(
        _snapshot(window_end_ms=60_000, generation=200.0, running=2.0),
        queue=queue,
        registry=registry,
        signal_state=state,
    )
    snap = _snapshot(window_end_ms=65_000, generation=400.0, running=2.0)
    r_rescue = run_rescue_tick(snap, queue=queue, registry=registry, signal_state=state)
    r_fair = run_fairness_tick(snap, queue=queue, registry=registry, signal_state=state)
    # Fairness re-reads the same snapshot -> same EMA, no second advance.
    assert r_fair.model_contexts["m"]["trs"] == pytest.approx(r_rescue.model_contexts["m"]["trs"])


def test_without_signal_state_trs_equals_raw_each_tick() -> None:
    # Back-compat: no signal_state -> fresh computer per tick -> no persistence.
    registry = _registry(20_000)
    queue = _Queue()
    r1 = run_rescue_tick(
        _snapshot(window_end_ms=60_000, generation=200.0, running=2.0),
        queue=queue,
        registry=registry,
    )
    r2 = run_rescue_tick(
        _snapshot(window_end_ms=65_000, generation=400.0, running=2.0),
        queue=queue,
        registry=registry,
    )
    assert r1.model_contexts["m"]["trs"] == pytest.approx(100.0)
    assert r2.model_contexts["m"]["trs"] == pytest.approx(200.0)  # raw, no persistence

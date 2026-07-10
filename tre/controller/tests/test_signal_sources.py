from __future__ import annotations

from tre_common.metrics_schema import ModelWindowMetrics
from tre_common.registry import AltThreshold, ModelSpec, SloSpec, TrsParams
from tre_controller.signals.sources import get_signal


def _spec() -> ModelSpec:
    return ModelSpec(
        name="m",
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
        alt_thresholds={
            "queue_len": AltThreshold(theta=4.0, direction="lower_is_healthier")
        },
    )


def _metrics(**overrides) -> ModelWindowMetrics:
    values = {
        "model": "m",
        "window_start_ms": 0,
        "window_end_ms": 60_000,
        "prompt_tokens": 0.0,
        "generation_tokens": 100.0,
        "avg_waiting": 0.0,
        "avg_running": 1.0,
        "avg_swapping": 0.0,
        "kv_cache_hit_rate": 0.5,
        "ttft_p95_ms": 500.0,
        "tpot_p95_ms": 50.0,
        "e2e_p95_ms": 5000.0,
        "routable_pods": 1,
        "assigned_replicas": 1,
        "per_pod": {},
    }
    values.update(overrides)
    return ModelWindowMetrics(**values)


def test_zm_signal_uses_trs_z_without_reinterpreting_metrics() -> None:
    signal = get_signal(_metrics(ttft_p95_ms=2000.0), _spec(), "zm", trs_z_m=0.75)

    assert signal.source == "zm"
    assert signal.raw_value == 0.75
    assert signal.z_m == 0.75
    assert signal.unavailable_reason is None


def test_latency_signal_uses_worst_slo_health_ratio() -> None:
    signal = get_signal(
        _metrics(ttft_p95_ms=2000.0, tpot_p95_ms=50.0, e2e_p95_ms=20_000.0),
        _spec(),
        "latency_p95",
        trs_z_m=9.0,
    )

    assert signal.source == "latency_p95"
    assert signal.raw_value == 20_000.0
    assert signal.z_m == 0.5


def test_queue_signal_uses_fitted_lower_is_healthier_threshold() -> None:
    signal = get_signal(_metrics(avg_waiting=3.0, avg_running=2.0), _spec(), "queue_len", trs_z_m=9.0)

    assert signal.source == "queue_len"
    assert signal.raw_value == 8.0
    assert signal.z_m == 0.5


def test_queue_signal_direction_boundaries_and_idle_cap() -> None:
    spec = _spec()
    at_theta = get_signal(_metrics(avg_running=4.0), spec, "queue_len", trs_z_m=9.0)
    twice_theta = get_signal(_metrics(avg_running=8.0), spec, "queue_len", trs_z_m=9.0)
    idle = get_signal(_metrics(avg_running=0.0), spec, "queue_len", trs_z_m=9.0)

    assert at_theta.z_m == 1.0
    assert twice_theta.z_m == 0.5
    assert idle.raw_value == 0.0
    assert idle.z_m == 10.0


def test_latency_signal_is_unavailable_when_no_latency_samples_exist() -> None:
    signal = get_signal(
        _metrics(ttft_p95_ms=None, tpot_p95_ms=None, e2e_p95_ms=None),
        _spec(),
        "latency_p95",
        trs_z_m=9.0,
    )

    assert signal.z_m is None
    assert signal.unavailable_reason == "latency_p95_missing"

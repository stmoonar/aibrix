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
            "queue_len": AltThreshold(theta=4.0, direction="lower_is_healthier"),
            "decode_tps": AltThreshold(theta=25.0, direction="lower_is_healthier"),
            "prefill_tps": AltThreshold(theta=50.0, direction="lower_is_healthier"),
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
    # Raw running + waiting per routable replica: lambda_wait (2.0 here) no longer
    # enters the ablation signal (plan 6.9 correction).
    assert signal.raw_value == 5.0
    assert signal.z_m == 0.8


def test_queue_signal_direction_boundaries_and_idle_cap() -> None:
    spec = _spec()
    at_theta = get_signal(_metrics(avg_running=4.0), spec, "queue_len", trs_z_m=9.0)
    twice_theta = get_signal(_metrics(avg_running=8.0), spec, "queue_len", trs_z_m=9.0)
    idle = get_signal(_metrics(avg_running=0.0), spec, "queue_len", trs_z_m=9.0)

    assert at_theta.z_m == 1.0
    assert twice_theta.z_m == 0.5
    assert idle.raw_value == 0.0
    assert idle.z_m == 10.0


def test_decode_tps_is_windowed_per_replica_pressure() -> None:
    spec = _spec()
    at_theta = get_signal(
        _metrics(
            window_end_ms=60_000,
            generation_tokens=3_000.0,
            routable_pods=2,
        ),
        spec,
        "decode_tps",
        trs_z_m=9.0,
    )
    twice_theta = get_signal(
        _metrics(generation_tokens=6_000.0, routable_pods=2),
        spec,
        "decode_tps",
        trs_z_m=9.0,
    )

    assert at_theta.raw_value == 25.0
    assert at_theta.z_m == 1.0
    assert twice_theta.raw_value == 50.0
    assert twice_theta.z_m == 0.5


def test_prefill_tps_uses_prompt_counter_delta() -> None:
    signal = get_signal(
        _metrics(prompt_tokens=6_000.0, routable_pods=2),
        _spec(),
        "prefill_tps",
        trs_z_m=9.0,
    )
    assert signal.raw_value == 50.0
    assert signal.z_m == 1.0


def test_tps_signal_is_unavailable_after_counter_reset() -> None:
    signal = get_signal(
        _metrics(generation_tokens=3_000.0, token_counter_reset=True),
        _spec(),
        "decode_tps",
        trs_z_m=9.0,
    )
    assert signal.z_m is None
    assert signal.unavailable_reason == "decode_tps_counter_missing"


def test_latency_signal_is_unavailable_when_no_latency_samples_exist() -> None:
    signal = get_signal(
        _metrics(ttft_p95_ms=None, tpot_p95_ms=None, e2e_p95_ms=None),
        _spec(),
        "latency_p95",
        trs_z_m=9.0,
    )

    assert signal.z_m is None
    assert signal.unavailable_reason == "latency_p95_missing"


def _spec_with_tau(ema_tau_ms) -> ModelSpec:
    import dataclasses

    spec = _spec()
    return dataclasses.replace(spec, trs=dataclasses.replace(spec.trs, ema_tau_ms=ema_tau_ms))


def _queue_series(spec: ModelSpec) -> list[float]:
    """queue_len raw values of three consecutive 10 s windows through one SignalState."""
    from tre_controller.signals.trs import SignalState

    state = SignalState()
    out = []
    for k, running in enumerate((2.0, 8.0, 8.0)):
        end = 60_000 + 10_000 * k
        sig = get_signal(_metrics(window_start_ms=end - 30_000, window_end_ms=end, avg_running=running),
                         spec, "queue_len", trs_z_m=9.0, signal_state=state)
        out.append(sig.raw_value)
    return out


def test_alt_signal_tau_zero_means_no_smoothing_not_the_default() -> None:
    """ema_tau_ms = 0 is tau = 0 (alpha = 1): the raw value, as offline - it used to be read
    as unset and smoothed with DEFAULT_EMA_TAU_MS (20 s)."""
    from tre_controller.signals.sources import alt_signal_tau_ms
    from tre_common.tss import DEFAULT_EMA_TAU_MS

    assert alt_signal_tau_ms(_spec_with_tau(0.0)) is None
    assert alt_signal_tau_ms(_spec_with_tau(None)) == DEFAULT_EMA_TAU_MS
    assert alt_signal_tau_ms(_spec_with_tau(10_000.0)) == 10_000.0
    assert _queue_series(_spec_with_tau(0.0)) == [2.0, 8.0, 8.0]


def test_alt_signal_tau_follows_the_registry_tau() -> None:
    import math

    smoothed = _queue_series(_spec_with_tau(10_000.0))
    alpha = 1.0 - math.exp(-1.0)          # dt 10 s, tau 10 s
    assert smoothed[0] == 2.0
    assert math.isclose(smoothed[1], 2.0 + alpha * 6.0)
    # the default tau smooths more slowly than tau = 10 s
    default = _queue_series(_spec_with_tau(None))
    assert default[1] < smoothed[1]

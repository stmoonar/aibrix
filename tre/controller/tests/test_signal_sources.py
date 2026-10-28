from __future__ import annotations

import pytest

from tre_common.metrics_schema import ModelWindowMetrics, PodWindowMetrics
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
            "ttft_mean": AltThreshold(theta=200.0, direction="lower_is_healthier",
                                      delta_crit=0.2, delta_high=0.25),
            "kv_usage": AltThreshold(theta=0.5, direction="higher_is_healthier",
                                     delta_crit=0.2, delta_high=0.25),
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
    if "per_pod" not in overrides:
        # One pod carrying the window's queue: queue_len reads the routable pods' data
        # (review L1/L2), a window without its pod is unknown.
        values["per_pod"] = {"p0": PodWindowMetrics(
            pod="p0", prompt_tokens=values["prompt_tokens"], generation_tokens=values["generation_tokens"],
            avg_waiting=values["avg_waiting"], avg_running=values["avg_running"], avg_swapping=0.0,
            kv_cache_hit_rate=0.0, ttft_p95_ms=None, tpot_p95_ms=None, e2e_p95_ms=None,
        )}
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
    # v1 floor: one request per replica, so an idle model reads z = theta / 1, not the cap.
    assert idle.raw_value == 1.0
    assert idle.z_m == 4.0


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


# --- v1 ablation signals ttft_mean / kv_usage (2026-10-28) --------------------------------


def _pod(name: str, *, kv: float | None = 0.0, ttft_ms: float | None = None,
         ttft_n: float | None = 0.0) -> PodWindowMetrics:
    return PodWindowMetrics(
        pod=name, prompt_tokens=10.0, generation_tokens=10.0, avg_waiting=0.0, avg_running=1.0,
        avg_swapping=0.0, kv_cache_hit_rate=0.0, ttft_p95_ms=None, tpot_p95_ms=None,
        e2e_p95_ms=None, gpu_cache_usage=kv, ttft_avg_ms=ttft_ms, ttft_count=ttft_n,
    )


def _pods_window(pods: list[PodWindowMetrics], *, routable: int, end: int = 60_000, **kw) -> ModelWindowMetrics:
    return _metrics(window_start_ms=end - 20_000, window_end_ms=end,
                    per_pod={p.pod: p for p in pods}, routable_pods=routable, **kw)


def test_kv_usage_is_headroom_of_the_mean_fill_per_routable_replica() -> None:
    sig = get_signal(_pods_window([_pod("a", kv=0.2), _pod("b", kv=0.6)], routable=2),
                     _spec(), "kv_usage", trs_z_m=9.0)
    assert sig.raw_value == pytest.approx(0.6)  # 1 - (0.2 + 0.6) / 2
    assert sig.z_m == pytest.approx(1.2)  # higher is healthier: 0.6 / theta 0.5
    full = get_signal(_pods_window([_pod("a", kv=1.0)], routable=1), _spec(), "kv_usage", trs_z_m=9.0)
    assert full.raw_value == pytest.approx(1e-3)  # v1 headroom floor


def test_tp2_replica_counts_once_in_the_pod_aggregates() -> None:
    """A TP2 replica is one pod with one KV gauge and one TTFT histogram (one engine):
    the model aggregates divide by replicas (pods), not by GPUs."""
    import dataclasses

    spec = dataclasses.replace(_spec(), tp_size=2)
    pods = [_pod("tp2-a", kv=0.4, ttft_ms=100.0, ttft_n=10.0), _pod("tp2-b", kv=0.8, ttft_ms=300.0, ttft_n=30.0)]
    kv = get_signal(_pods_window(pods, routable=2), spec, "kv_usage", trs_z_m=9.0)
    ttft = get_signal(_pods_window(pods, routable=2), spec, "ttft_mean", trs_z_m=9.0)
    assert kv.raw_value == pytest.approx(0.4)
    assert ttft.raw_value == pytest.approx(250.0)


def test_ttft_mean_pools_pod_means_by_first_token_count() -> None:
    pods = [_pod("a", ttft_ms=100.0, ttft_n=30.0), _pod("b", ttft_ms=400.0, ttft_n=10.0), _pod("c", ttft_n=0.0)]
    sig = get_signal(_pods_window(pods, routable=3), _spec(), "ttft_mean", trs_z_m=9.0)
    assert sig.raw_value == pytest.approx(175.0)  # (100*30 + 400*10) / 40
    assert sig.z_m == pytest.approx(200.0 / 175.0)


@pytest.mark.parametrize("source", ["queue_len", "ttft_mean", "kv_usage"])
@pytest.mark.parametrize(
    "case",
    ["pod_left_out", "no_routable", "no_pods", "kv_gauge_missing", "ttft_hist_missing"],
)
def test_unknown_pod_data_is_never_healthy(source: str, case: str) -> None:
    """Fail closed: a routable replica without data (left out as scrape-stale, or a
    gauge / histogram not read) makes the model signal unknown (z None -> state
    UNKNOWN), never a healthy or idle value read from the remaining pods."""
    good = _pod("a", kv=0.1, ttft_ms=50.0, ttft_n=20.0)
    windows = {
        "pod_left_out": _pods_window([good], routable=2, scrape_stale_pods=("b",)),
        "no_routable": _pods_window([good], routable=0),
        "no_pods": _pods_window([], routable=1),
        "kv_gauge_missing": _pods_window([good, _pod("b", kv=None, ttft_ms=50.0, ttft_n=5.0)], routable=2),
        "ttft_hist_missing": _pods_window([good, _pod("b", kv=0.1, ttft_n=None)], routable=2),
    }
    sig = get_signal(windows[case], _spec(), source, trs_z_m=9.0)
    if (source, case) in {("ttft_mean", "kv_gauge_missing"), ("kv_usage", "ttft_hist_missing"),
                          ("queue_len", "kv_gauge_missing"), ("queue_len", "ttft_hist_missing")}:
        assert sig.z_m is not None  # the other signal's input is not needed
        return
    assert sig.z_m is None
    assert sig.unavailable_reason == f"{source}_missing"


def test_ttft_mean_without_first_token_holds_one_window_then_unknown() -> None:
    """v1 held the last EMA in a window without a TTFT sample. Here it is held only while
    the shared EMA keeps it (gap <= one window, traffic flowing); then unknown."""
    from tre_controller.signals.trs import SignalState

    spec = _spec_with_tau(10_000.0)
    state = SignalState()
    seen = get_signal(_pods_window([_pod("a", ttft_ms=400.0, ttft_n=20.0)], routable=1, end=60_000),
                      spec, "ttft_mean", trs_z_m=9.0, signal_state=state)
    assert seen.raw_value == pytest.approx(400.0)
    no_token = [_pod("a", ttft_n=0.0)]
    held = get_signal(_pods_window(no_token, routable=1, end=70_000), spec, "ttft_mean",
                      trs_z_m=9.0, signal_state=state)
    assert held.raw_value == pytest.approx(400.0) and held.z_m == pytest.approx(0.5)
    stale = get_signal(_pods_window(no_token, routable=1, end=90_001), spec, "ttft_mean",
                       trs_z_m=9.0, signal_state=state)
    assert stale.z_m is None and stale.unavailable_reason == "ttft_mean_no_sample"
    # Without state (or before any sample) there is nothing to hold.
    fresh = get_signal(_pods_window(no_token, routable=1), spec, "ttft_mean", trs_z_m=9.0)
    assert fresh.z_m is None


def test_ttft_mean_idle_window_clears_the_held_value() -> None:
    from tre_controller.signals.trs import SignalState

    spec = _spec_with_tau(10_000.0)
    state = SignalState()
    get_signal(_pods_window([_pod("a", ttft_ms=50.0, ttft_n=20.0)], routable=1, end=60_000),
               spec, "ttft_mean", trs_z_m=9.0, signal_state=state)
    idle = get_signal(
        _pods_window([_pod("a", ttft_n=0.0)], routable=1, end=70_000, prompt_tokens=0.0, generation_tokens=0.0),
        spec, "ttft_mean", trs_z_m=9.0, signal_state=state,
    )
    assert idle.z_m is None


def test_the_alt_signals_sum_the_routable_pods_only() -> None:
    """Review L2: matched by identity against the fleet view's routable pods - a hidden
    SafeScale probe pod (still in the window while it drains) is left out of the sums, and
    it cannot stand in for a routable pod without data."""
    routable = _pod("a", kv=0.2, ttft_ms=100.0, ttft_n=10.0)
    hidden = _pod("probe", kv=0.9, ttft_ms=900.0, ttft_n=10.0)
    hidden = PodWindowMetrics(**{**hidden.__dict__, "avg_running": 7.0})
    window = _pods_window([routable, hidden], routable=1)
    q = get_signal(window, _spec(), "queue_len", trs_z_m=9.0, routable_pod_ids=frozenset({"a"}))
    kv = get_signal(window, _spec(), "kv_usage", trs_z_m=9.0, routable_pod_ids=frozenset({"a"}))
    ttft = get_signal(window, _spec(), "ttft_mean", trs_z_m=9.0, routable_pod_ids=frozenset({"a"}))
    assert (q.raw_value, kv.raw_value, ttft.raw_value) == (1.0, pytest.approx(0.8), pytest.approx(100.0))
    # routable b has no data, the hidden pod is there: still unknown (a count check passed it)
    gap = _pods_window([routable, hidden], routable=2, scrape_stale_pods=("b",))
    for source in ("queue_len", "ttft_mean", "kv_usage"):
        assert get_signal(gap, _spec(), source, trs_z_m=9.0, routable_pod_ids=frozenset({"a", "b"})).z_m is None
    # ids that disagree with the routable count: fail closed
    assert get_signal(window, _spec(), "queue_len", trs_z_m=9.0, routable_pod_ids=frozenset()).z_m is None

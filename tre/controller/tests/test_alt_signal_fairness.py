"""Alternative (ablation) signals mean the same thing online and offline (plan 6.9 items 2/4/7).

* queue_len is the raw queue per routable replica: scaling a model out to N replicas at
  the same per-replica load must leave its z unchanged (the fleet sum used before shrank z
  N-fold, so the queue_len arm read CRITICAL forever after a scale-out);
* the registry's lambda_wait and a non-zero swapping count do not enter it;
* every alternative signal is smoothed by the TSS tau-EMA, and the controller and the
  offline loader produce the same smoothed series from the same windows.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tre_common.alt_signals import queue_len_per_replica
from tre_common.metrics_schema import ModelWindowMetrics
from tre_common.registry import AltThreshold, ModelSpec, SloSpec, TrsParams
from tre_common.tss import DEFAULT_EMA_TAU_MS
from tre_controller.signals.sources import get_signal
from tre_controller.signals.trs import SignalState

from tre_calibration.alt_signals import alt_signal_transform
from tre_calibration.dataset import load_windows_from_csv


def _spec(lambda_wait: float = 3.0) -> ModelSpec:
    return ModelSpec(
        name="m",
        weights_path="/w",
        tp_size=1,
        min_replicas=0,
        max_replicas=8,
        vllm_image="img",
        slo=SloSpec(ttft_p95_ms=500.0, tpot_p95_ms=75.0, e2e_p95_ms=10_000.0),
        trs=TrsParams(
            w_p=0.02, w_d=1.0, lambda_wait=lambda_wait, qmin=1.0, ema_alpha=0.0,
            theta_m=40.0, tau_crit=0.8, tau_low=1.0, tau_high=1.25, qsat=4.0,
            epsat=0.05, hsat=1, ema_tau_ms=DEFAULT_EMA_TAU_MS,
        ),
        alt_thresholds={
            "queue_len": AltThreshold(theta=20.0, direction="lower_is_healthier"),
            "decode_tps": AltThreshold(theta=900.0, direction="lower_is_healthier"),
            "prefill_tps": AltThreshold(theta=2000.0, direction="lower_is_healthier"),
        },
    )


def _metrics(*, pods: int, running: float, waiting: float, end_ms: int = 30_000,
             swapping: float = 0.0, gen: float = 0.0, prompt: float = 0.0) -> ModelWindowMetrics:
    return ModelWindowMetrics(
        model="m", window_start_ms=end_ms - 30_000, window_end_ms=end_ms,
        prompt_tokens=prompt, generation_tokens=gen, avg_waiting=waiting,
        avg_running=running, avg_swapping=swapping, kv_cache_hit_rate=0.0,
        ttft_p95_ms=100.0, tpot_p95_ms=20.0, e2e_p95_ms=1000.0,
        routable_pods=pods, assigned_replicas=pods, per_pod={},
    )


@pytest.mark.parametrize("pods", [1, 2, 3, 4, 8])
def test_scaling_out_at_the_same_per_replica_load_leaves_queue_z_unchanged(pods: int) -> None:
    per_replica_running, per_replica_waiting = 18.0, 7.0
    one = get_signal(_metrics(pods=1, running=per_replica_running, waiting=per_replica_waiting),
                     _spec(), "queue_len", trs_z_m=None)
    many = get_signal(
        _metrics(pods=pods, running=pods * per_replica_running, waiting=pods * per_replica_waiting),
        _spec(), "queue_len", trs_z_m=None,
    )
    assert many.raw_value == pytest.approx(25.0)
    assert many.z_m == pytest.approx(one.z_m)
    assert one.z_m == pytest.approx(20.0 / 25.0)


@pytest.mark.parametrize("source", ["decode_tps", "prefill_tps"])
def test_scaling_out_at_the_same_per_replica_load_leaves_token_rate_z_unchanged(source: str) -> None:
    one = get_signal(_metrics(pods=1, running=10, waiting=0, gen=30_000, prompt=60_000),
                     _spec(), source, trs_z_m=None)
    four = get_signal(_metrics(pods=4, running=40, waiting=0, gen=120_000, prompt=240_000),
                      _spec(), source, trs_z_m=None)
    assert four.z_m == pytest.approx(one.z_m)


def test_queue_len_is_raw_running_plus_waiting_without_lambda_or_swapping() -> None:
    a = get_signal(_metrics(pods=2, running=20, waiting=10, swapping=5), _spec(lambda_wait=0.0),
                   "queue_len", trs_z_m=None)
    b = get_signal(_metrics(pods=2, running=20, waiting=10), _spec(lambda_wait=4.0),
                   "queue_len", trs_z_m=None)
    assert a.raw_value == b.raw_value == pytest.approx(15.0)
    # Offline goes through the very same function.
    row = {"avg_running": "20", "avg_waiting": "10", "routable_pods": "2", "avg_swapping": "5"}
    assert alt_signal_transform("queue_len")(row) == pytest.approx(15.0)
    assert queue_len_per_replica(20, 10, 0) == pytest.approx(30.0)  # max(1, pods)


_SERIES = [  # (running, waiting) per 5 s step; zeros are ordinary samples for a queue
    (0.0, 0.0), (0.0, 0.0), (12.0, 0.0), (30.0, 6.0), (30.0, 20.0), (0.0, 0.0), (8.0, 0.0),
]


def test_online_and_offline_alt_signals_share_one_tau_ema(tmp_path: Path) -> None:
    state = SignalState()
    online: list[float] = []
    for step, (running, waiting) in enumerate(_SERIES):
        end = 30_000 + 5_000 * step
        metrics = _metrics(pods=1, running=running, waiting=waiting, end_ms=end, gen=100.0)
        value = get_signal(metrics, _spec(), "queue_len", trs_z_m=None, signal_state=state).raw_value
        # A second read of the same window (rescue + fairness + safescale) must not advance it.
        again = get_signal(metrics, _spec(), "queue_len", trs_z_m=None, signal_state=state).raw_value
        assert again == value
        online.append(value)
    # The EMA really smooths (and a zero advanced it: the 6th value is below the 5th).
    assert online[3] < 36.0 and online[5] < online[4] and online[5] > 0.0

    path = tmp_path / "w.csv"
    fields = ["scenario_id", "scenario_family", "window_start_ms", "window_end_ms",
              "avg_running", "avg_waiting", "routable_pods", "prompt_tokens_total",
              "generation_tokens_total", "p95_ttft", "p95_tpot"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for step, (running, waiting) in enumerate(_SERIES):
            end = 30_000 + 5_000 * step
            writer.writerow({
                "scenario_id": "c1", "scenario_family": "f", "window_start_ms": end - 30_000,
                "window_end_ms": end, "avg_running": running, "avg_waiting": waiting,
                "routable_pods": 1, "prompt_tokens_total": 0, "generation_tokens_total": 100,
                "p95_ttft": 100, "p95_tpot": 20,
            })
    offline = [w.signal for w in load_windows_from_csv(
        path, latency_slo_ms={"ttft_p95": 500.0, "tpot_p95": 75.0},
        signal_transform=alt_signal_transform("queue_len"), ema_tau_ms=DEFAULT_EMA_TAU_MS,
    )]
    assert offline == online  # bitwise
    raw = [w.signal for w in load_windows_from_csv(
        path, latency_slo_ms={"ttft_p95": 500.0, "tpot_p95": 75.0},
        signal_transform=alt_signal_transform("queue_len"),
    )]
    assert raw == [r + w for r, w in _SERIES]

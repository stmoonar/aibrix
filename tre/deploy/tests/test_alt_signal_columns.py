"""The v1 ablation signals' window columns (scripts.alt_signal_columns): the per-pod
rules the controller applies online (pooled TTFT mean, mean KV fill of the serving pods),
replayed on the 1 Hz capture; unknown stays empty, never zero."""
from __future__ import annotations

import pytest

from scripts import alt_signal_columns as alt
from scripts import l3_numerator as l3


def _pod(name: str, *, ttft_ms: float, rps: float, kv: float | None, asleep: bool = False,
         reset_at: int | None = None, t1: int = 120_000) -> alt.PodAltSamples:
    """1 Hz samples 1 ms after each second; ``rps`` first tokens per second, each
    ``ttft_ms`` long; KV fill ``kv`` (None = gauge absent)."""
    samples = []
    for t in range(1, t1, 1000):
        base = 0 if reset_at is None or t < reset_at else reset_at
        k = (t - base) // 1000
        samples.append((t, k * rps * ttft_ms / 1000.0, float(k * rps), kv, asleep))
    return alt.PodAltSamples(name, samples)


def test_ttft_mean_pools_the_pods_by_first_token_count_and_kv_sums_awake_pods() -> None:
    src = alt.AltSignalColumns([
        _pod("a", ttft_ms=100.0, rps=3.0, kv=0.2),
        _pod("b", ttft_ms=400.0, rps=1.0, kv=0.6),
        _pod("sleeping", ttft_ms=0.0, rps=0.0, kv=0.0, asleep=True),
    ])
    row = src(40_000, 60_000)
    assert row["ttft_server_count"] == pytest.approx(80.0)  # (3 + 1) x 20 s
    assert row["ttft_server_mean_ms"] == pytest.approx(175.0)  # (100*3 + 400*1) / 4
    # the sleeping pod adds neither fill nor a replica (online: dropped from the window)
    assert row["kv_cache_usage_sum"] == pytest.approx(0.8) and row["kv_cache_pods"] == 2
    assert src.counts == {alt.STATUS_OK: 1}


def test_no_first_token_is_an_empty_mean_not_zero() -> None:
    row = alt.AltSignalColumns([_pod("a", ttft_ms=100.0, rps=0.0, kv=0.0)])(40_000, 60_000)
    assert row["ttft_server_mean_ms"] is None and row["ttft_server_count"] == 0.0
    assert row["kv_cache_usage_sum"] == 0.0  # an empty cache is a real sample


@pytest.mark.parametrize("case, status", [
    ("reset", l3.STATUS_RESET),
    ("kv_missing", alt.STATUS_NO_KV),
    ("past_capture", l3.STATUS_NO_SAMPLE),
    ("missing_pod_file", l3.STATUS_NO_METRICS),
])
def test_unknown_inputs_void_every_column(case: str, status: str) -> None:
    good = _pod("a", ttft_ms=100.0, rps=1.0, kv=0.3)
    pods, missing, window = [good], [], (40_000, 60_000)
    if case == "reset":
        pods.append(_pod("b", ttft_ms=100.0, rps=1.0, kv=0.3, reset_at=50_500))
    elif case == "kv_missing":
        pods.append(_pod("b", ttft_ms=100.0, rps=1.0, kv=None))
    elif case == "past_capture":
        window = (130_000, 150_000)
    else:
        missing = ["ns/gone"]
    src = alt.AltSignalColumns(pods, missing_pods=missing)
    assert src(*window) == {column: None for column in alt.COLUMNS}
    assert src.counts == {status: 1}


def test_columns_are_read_by_the_shared_signal_functions() -> None:
    """The dataset columns feed tre_calibration's transforms, i.e. the controller's
    tre_common.alt_signals functions."""
    from tre_calibration.alt_signals import alt_signal_transform

    row = alt.AltSignalColumns([_pod("a", ttft_ms=250.0, rps=2.0, kv=0.4)])(40_000, 60_000)
    assert alt_signal_transform("ttft_mean")(row) == pytest.approx(250.0)
    assert alt_signal_transform("kv_usage")(row) == pytest.approx(0.6)

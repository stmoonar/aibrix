"""MetricsStore reads gateway docs under either the pre-0.30 or the 0.30 metric identifier.

The gateway writes ``time_per_output_token_seconds`` / ``gpu_cache_usage_perc`` doc keys
(it resolves vLLM 0.30's renamed families itself); the reader also accepts
``inter_token_latency_seconds`` / ``kv_cache_usage_perc`` (``tre_common.vllm_metrics``),
and reads the unchanged e2e histogram the way the live docs carry it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tre_common.registry import load_registry
from tre_controller.store.metrics_store import MetricsStore
from test_metrics_store import FakeRedis, add_doc

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml"
MODEL = "dsqwen-7b"
POD = "default/pod-a"

# vLLM 0.30 bucket bounds of the three latency histograms (subset; from the live sample).
E2E_BOUNDS = ("0.3", "0.5", "0.8", "1.0", "1.5", "2.0", "2.5", "5.0", "10.0", "+Inf")


def _hist(count: float, sum_: float, bounds: tuple[str, ...], filled_from: int) -> dict:
    """Cumulative buckets: ``count`` observations all at or above ``bounds[filled_from]``."""
    return {
        "count": count,
        "sum": sum_,
        "buckets": {b: (count if i >= filled_from else 0.0) for i, b in enumerate(bounds)},
    }


def _hist_doc(tpot_key: str, n: float) -> dict:
    return {
        "pod_name": "pod-a",
        "model_histogram_metrics": {
            f"{MODEL}/request_prompt_tokens": _hist(n, 100 * n, ("100", "+Inf"), 0),
            f"{MODEL}/request_generation_tokens": _hist(n, 100 * n, ("100", "+Inf"), 0),
            f"{MODEL}/time_to_first_token_seconds": _hist(n, 0.05 * n, ("0.06", "0.1", "+Inf"), 0),
            f"{MODEL}/{tpot_key}": _hist(100 * n, 2.0 * n, ("0.01", "0.025", "0.05", "+Inf"), 1),
            f"{MODEL}/e2e_request_latency_seconds": _hist(n, 4.0 * n, E2E_BOUNDS, 7),
        },
    }


def _inst_doc(cache_key: str, usage: float) -> dict:
    return {
        "pod_name": "pod-a",
        "model_metrics": {
            f"{MODEL}/num_requests_waiting": 1,
            f"{MODEL}/num_requests_running": 2,
            f"{MODEL}/{cache_key}": usage,
        },
    }


@pytest.mark.parametrize(
    "tpot_key,cache_key",
    [
        ("time_per_output_token_seconds", "gpu_cache_usage_perc"),  # what the gateway writes
        ("inter_token_latency_seconds", "kv_cache_usage_perc"),  # vLLM 0.30 identifiers
    ],
)
def test_latency_and_cache_read_under_either_identifier(tpot_key: str, cache_key: str) -> None:
    redis = FakeRedis()
    redis.sadd(f"tre:v2:pods:{MODEL}", POD)
    for ts, n in ((0, 0.0), (10_000, 20.0), (20_000, 40.0)):
        add_doc(redis, "tre:v2:hist:" + POD, ts, _hist_doc(tpot_key, n))
        add_doc(redis, "tre:v2:inst:" + POD, ts, _inst_doc(cache_key, 0.4))
    store = MetricsStore(
        redis, load_registry(str(REGISTRY_PATH)), instant_sample_interval_ms=10_000,
        min_latency_samples=10,
    )

    window = store.read_model_window(MODEL, 0, 20_000, start_exclusive=True)
    pod = window.per_pod["pod-a"]

    assert window.e2e_p95_ms == 5000.0  # every request landed in the (2.5, 5.0] bucket
    assert window.tpot_p95_ms == 25.0
    assert window.ttft_p95_ms == 60.0
    assert pod.tpot_count == 4000.0
    assert pod.tpot_avg_ms == pytest.approx(20.0)
    assert pod.gpu_cache_usage == pytest.approx(0.4)
    latest = store.read_latest_instant(MODEL, 20_000, 20_000)
    assert latest == {"waiting": 1.0, "running": 2.0, "swapping": 0.0}


def test_new_identifier_wins_when_both_are_present() -> None:
    redis = FakeRedis()
    redis.sadd(f"tre:v2:pods:{MODEL}", POD)
    doc = _inst_doc("kv_cache_usage_perc", 0.25)
    doc["model_metrics"][f"{MODEL}/gpu_cache_usage_perc"] = 0.75
    add_doc(redis, "tre:v2:inst:" + POD, 10_000, doc)
    store = MetricsStore(redis, load_registry(str(REGISTRY_PATH)), instant_sample_interval_ms=10_000)
    window = store.read_model_window(MODEL, 0, 10_000, start_exclusive=True)
    assert window.per_pod["pod-a"].gpu_cache_usage == pytest.approx(0.25)

"""Online / offline parity of the v1 ablation signals (review L3 / L4, 2026-10-28).

The fit is only about the signal if theta is fitted on exactly the value the controller
computes. One set of raw per-pod samples goes through the controller path (gateway docs ->
MetricsStore -> restrict_to_serving -> get_signal) and through the calibration path
(``scripts.alt_signal_columns`` for ttft_mean / kv_usage, ``rewindow_from_raw.aggregate_window``
raw_mean for the queue -> ``tre_calibration.alt_signals.alt_signal_transform``): equal
window values, including a window holding a wake and a window with a missing histogram."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from tre_common.registry import load_registry
from tre_common.window_pods import restrict_to_serving
from tre_controller.signals.sources import get_signal
from tre_controller.store.metrics_store import MetricsStore
from test_metrics_store import FakeRedis

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml"
MODEL = "dsqwen-7b"
P = 2_000
T0 = 900_000
END = 1_000_000
WINDOW = 20_000
WAKE = 951_000  # pod b sleeps until here (zero gauges, flat counters), then serves
NO_TTFT = 976_000  # pod a's doc here lacks the TTFT histogram (L3)


def _sample(pod: str, ts: int) -> dict:
    """Raw state of ``pod`` at ``ts``: cumulative TTFT sum (s) / count, queue, KV fill."""
    s = (ts - T0) / 1000.0
    if pod == "a":
        return {"asleep": False, "ttft_s": 0.12 * 3 * s + 0.0004 * s * s, "ttft_n": 3 * s,
                "running": 4 + (ts // P) % 3, "waiting": (ts // P) % 2, "kv": 0.3 + 0.01 * ((ts // P) % 7)}
    if ts < WAKE:
        return {"asleep": True, "ttft_s": 0.0, "ttft_n": 0.0, "running": 0, "waiting": 0, "kv": 0.0}
    w = (ts - WAKE) / 1000.0
    return {"asleep": False, "ttft_s": 0.4 * w, "ttft_n": 2 * w, "running": 6, "waiting": 2, "kv": 0.55}


def _online_store() -> MetricsStore:
    redis = FakeRedis()
    for pod in ("a", "b"):
        key = f"default/{pod}"
        redis.sadd(f"tre:v2:pods:{MODEL}", key)
        for ts in range(T0, END + 1, P):
            x = _sample(pod, ts)
            base = {"pod_name": pod, "timestamp": ts, "scraped_ms": ts, "written_ms": ts + 5}
            hists = {f"{MODEL}/request_prompt_tokens": {"sum": 0.0, "count": 0.0, "buckets": {}}}
            if not (pod == "a" and ts == NO_TTFT):
                hists[f"{MODEL}/time_to_first_token_seconds"] = {"sum": x["ttft_s"], "count": x["ttft_n"], "buckets": {}}
            redis.zadd(f"tre:v2:hist:{key}", {json.dumps({**base, "model_histogram_metrics": hists, "model_counter_metrics": {
                f"{MODEL}/prompt_tokens_total": 10.0 * ts / 1000, f"{MODEL}/generation_tokens_total": 20.0 * ts / 1000,
            }}): ts})
            redis.zadd(f"tre:v2:inst:{key}", {json.dumps({**base, "model_metrics": {
                f"{MODEL}/num_requests_running": x["running"], f"{MODEL}/num_requests_waiting": x["waiting"],
                f"{MODEL}/gpu_cache_usage_perc": x["kv"],
            }}): ts})
    return MetricsStore(redis, load_registry(str(REGISTRY_PATH)), instant_sample_interval_ms=P, schema="v2")


def _offline_pods():
    from scripts import alt_signal_columns as alt

    pods = []
    for pod in ("a", "b"):
        rows = []
        for ts in range(T0, END + 1, P):
            x = _sample(pod, ts)
            missing = pod == "a" and ts == NO_TTFT
            rows.append((ts, None if missing else x["ttft_s"], None if missing else x["ttft_n"], x["kv"], x["asleep"]))
        pods.append(alt.PodAltSamples(f"default/{pod}", rows))
    return pods


def _offline_queue(serving: list[str], start: int, end: int) -> dict:
    """The dataset's queue columns (raw_mean of the samples in (start, end]) of the serving
    pods: the calibration capture has one serving pod, so its row is that pod's mean."""
    from scripts.rewindow_from_raw import aggregate_window

    running = waiting = 0.0
    for pod in serving:
        inst = [{"ts_ms": ts, "running": _sample(pod, ts)["running"], "waiting": _sample(pod, ts)["waiting"]}
                for ts in range(T0, END + 1, P)]
        w = aggregate_window([], inst, MODEL, start, end, percentile_mode="bucket_upper", min_latency_samples=0,
                             instant_sample_interval_ms=P, half_open_start=True)
        running += w.avg_running
        waiting += w.avg_waiting
    return {"avg_running": running, "avg_waiting": waiting, "routable_pods": len(serving)}


def test_controller_and_fit_compute_the_same_alt_signal_values():
    from scripts import alt_signal_columns as alt
    from tre_calibration.alt_signals import alt_signal_transform

    store = _online_store()
    columns = alt.AltSignalColumns(_offline_pods())
    spec = load_registry(str(REGISTRY_PATH)).model(MODEL)
    seen = {"wake": 0, "asleep": 0, "missing": 0, "clean": 0}
    for end in range(T0 + 40_000, END + 1, P):
        start = end - WINDOW
        serving = ["a"] + (["b"] if not _sample("b", end)["asleep"] else [])
        raw = store.read_model_window(MODEL, start, end, use_cache=False, start_exclusive=True)
        window = restrict_to_serving(raw, sleeping_pods=[] if "b" in serving else ["b"], routable_pods=len(serving))
        row = {**columns(start, end), **_offline_queue(serving, start, end)}
        for source in ("queue_len", "ttft_mean", "kv_usage"):
            online = get_signal(window, spec, source, trs_z_m=None, routable_pod_ids=frozenset(serving)).raw_value
            offline = alt_signal_transform(source)(row)
            assert (online is None) == (offline is None), (source, start, end, online, offline)
            if online is not None:
                assert online == pytest.approx(offline, rel=1e-9), (source, start, end)
        if start <= NO_TTFT <= end:  # the doc is the baseline or inside the window
            assert row["ttft_server_mean_ms"] is None
            seen["missing"] += 1
        elif start < WAKE <= end:
            seen["wake"] += 1
        elif "b" not in serving:
            seen["asleep"] += 1
        else:
            seen["clean"] += 1
    assert min(seen.values()) >= 1, seen


def test_a_missing_ttft_histogram_is_unknown_not_no_first_token():
    """Review L3: the TTFT count of a pod whose docs lack the histogram is unknown (the pod
    is unusable, the model's ttft_mean unknown), not 0 (silently skipped)."""
    store = _online_store()
    w = store.read_model_window(MODEL, NO_TTFT - 10_000, NO_TTFT + 10_000, use_cache=False, start_exclusive=True)
    [pod_a] = [pod for pod in w.per_pod.values() if pod.pod == "a"]
    assert pod_a.ttft_count is None
    window = restrict_to_serving(w, sleeping_pods=[], routable_pods=2)
    spec = load_registry(str(REGISTRY_PATH)).model(MODEL)
    sig = get_signal(window, spec, "ttft_mean", trs_z_m=None, routable_pod_ids=frozenset({"a", "b"}))
    assert sig.raw_value is None and sig.unavailable_reason == "ttft_mean_missing"

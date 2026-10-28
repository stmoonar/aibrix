"""The linear histogram aggregation (2026-10-28, behavior-neutral perf change) gives the
same windows as the former per-bound scan: each histogram series of a pod is now
scanned once and its bucket delta formed once (``_MetricWindow``), the bucket delta is
one pass over the sorted bounds, and the pods' window histograms are merged in one walk
per pod (``tre_common.window_pods``).

The reference below is the former code, kept verbatim in this test only. The check is
byte-identical pickles of whole windows (controller store with O1 suffixes, SafeScale
evidence store) on seeded fixture docs: unsorted bucket keys (as the gateway writes
them), bounds missing on one side, counter resets, a series missing in some docs, a
pre-window baseline."""
from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

from tre_common.metrics_schema import PodWindowMetrics
from tre_common.percentile import histogram_percentile
from tre_common.registry import load_registry
from tre_common.window_pods import _merged_cumulative, merged_hist_p95_ms
from tre_controller.store import metrics_store as ms
from tre_controller.store.metrics_store import MetricsStore
from test_metrics_store import FakeRedis

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml"
MODEL = "dsqwen-7b"
H = ms.HISTOGRAM_METRICS


# ------------------------------------------------------------------ the former code
def _ref_cumulative_at(buckets, upper):
    candidates = [count for bucket_upper, count in buckets.items() if bucket_upper <= upper]
    return max(candidates) if candidates else 0.0


def _ref_bucket_delta(first, last):
    result, running = {}, 0.0
    for upper in sorted(set(first) | set(last)):
        running = max(running, max(0.0, _ref_cumulative_at(last, upper) - _ref_cumulative_at(first, upper)))
        result[upper] = running
    return result


def _ref_first_last(model, metric, docs):
    entries = [entry for entry in (ms._metric_entry(model, metric, doc) for doc in docs) if entry is not None]
    return (entries[0], entries[-1]) if entries else (None, None)


def _ref_has_window_metric(model, metric, docs, start):
    return any(
        not ms._number(doc.get("timestamp"), 0.0) < start and ms._metric_entry(model, metric, doc) is not None
        for doc in docs
    )


def _ref_doc_ts(model, metric, docs):
    return [int(ms._number(doc.get("timestamp"), 0.0)) for doc in docs
            if ms._metric_entry(model, metric, doc) is not None and doc.get("timestamp") is not None]


class _ReferenceStore(MetricsStore):
    """MetricsStore with the former per-field histogram reads."""

    def _count(self, model, metric, docs, start):
        if not ms._has_window_hist_doc(docs, start):
            return None
        first, last = _ref_first_last(model, metric, docs)
        if first is None:
            return 0.0
        return max(0.0, ms._number(last.get("count"), 0.0) - ms._number(first.get("count"), 0.0))

    def _sum(self, model, metric, docs, start):
        if not ms._has_window_hist_doc(docs, start):
            return None
        first, last = _ref_first_last(model, metric, docs)
        if first is None:
            return 0.0
        delta = ms._number(last.get("sum"), 0.0) - ms._number(first.get("sum"), 0.0)
        return delta if delta >= 0.0 else None

    def _avg(self, model, metric, docs, start):
        if not _ref_has_window_metric(model, metric, docs, start):
            return None
        first, last = _ref_first_last(model, metric, docs)
        sum_delta = max(0.0, ms._number(last.get("sum"), 0.0) - ms._number(first.get("sum"), 0.0))
        count_delta = max(0.0, ms._number(last.get("count"), 0.0) - ms._number(first.get("count"), 0.0))
        return None if count_delta <= 0.0 else sum_delta / count_delta

    def _pct(self, model, metric, docs, start):
        if not _ref_has_window_metric(model, metric, docs, start):
            return None
        first, last = _ref_first_last(model, metric, docs)
        if self._min_latency_samples > 0:
            count_delta = max(0.0, ms._number(last.get("count"), 0.0) - ms._number(first.get("count"), 0.0))
            if count_delta < self._min_latency_samples:
                return None
        fb, lb = ms._normal_buckets(first.get("buckets")), ms._normal_buckets(last.get("buckets"))
        if not fb or not lb:
            return None
        return histogram_percentile(_ref_bucket_delta(fb, lb).items(), 0.95, mode=self._percentile_mode)

    def _buckets(self, model, metric, docs, start):
        if not _ref_has_window_metric(model, metric, docs, start):
            return None, None
        first, last = _ref_first_last(model, metric, docs)
        fb, lb = ms._normal_buckets(first.get("buckets")), ms._normal_buckets(last.get("buckets"))
        if not fb or not lb:
            return None, None
        count = max(0.0, ms._number(last.get("count"), 0.0) - ms._number(first.get("count"), 0.0))
        return tuple(sorted(_ref_bucket_delta(fb, lb).items())), count

    def _aggregate_pod(self, model, pod_key, hist_docs, inst_docs, start):
        if not hist_docs and not inst_docs:
            return None
        e2e_hist, e2e_hist_count = self._buckets(model, H["e2e"], hist_docs, start)
        ttft_hist, ttft_hist_count = self._buckets(model, H["ttft"], hist_docs, start)
        tpot_hist, tpot_hist_count = self._buckets(model, H["tpot"], hist_docs, start)
        ttft_ts = _ref_doc_ts(model, H["ttft"], hist_docs)
        return PodWindowMetrics(
            pod=ms._pod_name(hist_docs, inst_docs, pod_key),
            prompt_tokens=self._counter_delta(model, ms.COUNTER_METRICS["prompt_tokens"], hist_docs, start),
            generation_tokens=self._counter_delta(model, ms.COUNTER_METRICS["generation_tokens"], hist_docs, start),
            avg_waiting=self._instant_avg(model, ms.INSTANT_METRICS["waiting"], inst_docs),
            avg_running=self._instant_avg(model, ms.INSTANT_METRICS["running"], inst_docs),
            avg_swapping=self._instant_avg(model, ms.INSTANT_METRICS["swapping"], inst_docs),
            kv_cache_hit_rate=self._instant_avg(model, ms.INSTANT_METRICS["kv_hit"], inst_docs),
            ttft_p95_ms=ms._seconds_to_ms(self._pct(model, H["ttft"], hist_docs, start)),
            tpot_p95_ms=ms._seconds_to_ms(self._pct(model, H["tpot"], hist_docs, start)),
            e2e_p95_ms=ms._seconds_to_ms(self._pct(model, H["e2e"], hist_docs, start)),
            request_count=self._count(model, H["completed_prompt_tokens"], hist_docs, start),
            request_count_observed=self._count(model, H["completed_prompt_tokens"], hist_docs, start),
            completed_prompt_tokens=self._sum(model, H["completed_prompt_tokens"], hist_docs, start),
            token_counter_reset=ms._counter_reset_index(model, hist_docs) > 0,
            instant_ticks_ms=ms._doc_ticks(inst_docs),
            gpu_cache_usage=self._instant_avg_optional(model, ms.INSTANT_METRICS["gpu_cache"], inst_docs),
            ttft_avg_ms=ms._seconds_to_ms(self._avg(model, H["ttft"], hist_docs, start)),
            ttft_count=self._count(model, H["ttft"], hist_docs, start),
            ttft_count_complete=self._hist_series_complete(model, H["ttft"], hist_docs, start),
            tpot_avg_ms=ms._seconds_to_ms(self._avg(model, H["tpot"], hist_docs, start)),
            tpot_count=self._count(model, H["tpot"], hist_docs, start),
            e2e_hist=e2e_hist,
            e2e_hist_count=e2e_hist_count,
            hist_first_ts_ms=ttft_ts[0] if ttft_ts else None,
            hist_last_ts_ms=ttft_ts[-1] if ttft_ts else None,
            ttft_hist=ttft_hist,
            ttft_hist_count=ttft_hist_count,
            tpot_hist=tpot_hist,
            tpot_hist_count=tpot_hist_count,
            **self._latest_instant(model, inst_docs),
        )


def _ref_merged(bucket_sets):
    uppers = sorted({upper for buckets in bucket_sets for upper, _ in buckets})
    at = lambda buckets, upper: max([c for u, c in buckets if u <= upper] or [0.0])  # noqa: E731
    return [(upper, sum(at(buckets, upper) for buckets in bucket_sets)) for upper in uppers]


# ------------------------------------------------------------------ fixtures
BOUNDS = ["0.05", "0.1", "0.25", "0.5", "1.0", "2.5", "5.0", "10.0", "+Inf"]


def _hist(rng, totals, bounds):
    """A cumulative histogram whose keys come in a shuffled (gateway string) order."""
    cumulative, running = {}, 0
    for bound in bounds:
        running += totals.get(bound, 0)
        cumulative[bound] = running
    keys = list(cumulative)
    rng.shuffle(keys)
    return {key: cumulative[key] for key in keys}


def _fill(redis, rng, pod, end):
    key = f"default/{pod}"
    redis.sadd(f"tre:v2:pods:{MODEL}", key)
    series = {name: {} for name in ("ttft", "tpot", "e2e", "completed_prompt_tokens")}
    sums = {name: 0.0 for name in series}
    prompt = generation = 0.0
    reset_at = end - rng.choice((8_000, 30_000, 10**9))
    for ts in range(end - 60_000, end + 1, 2_000):
        if ts >= reset_at and ts - 2_000 < reset_at:
            series = {name: {} for name in series}
            sums = {name: 0.0 for name in series}
            prompt = generation = 0.0
        prompt += rng.randint(0, 900)
        generation += rng.randint(0, 2_000)
        doc = {"pod_name": pod, "timestamp": ts, "scraped_ms": ts - rng.randint(0, 900), "model_histogram_metrics": {},
               "model_counter_metrics": {f"{MODEL}/prompt_tokens_total": prompt,
                                         f"{MODEL}/generation_tokens_total": generation}}
        for name in series:
            for bound in BOUNDS:
                series[name][bound] = series[name].get(bound, 0) + rng.choice((0, 0, 1, 3, 7))
            count = sum(series[name].values())
            sums[name] += rng.random() * 3
            if rng.random() < 0.1:
                continue  # the series is missing in this doc
            # A doc may carry a subset of the bounds (a missing bound on one side).
            bounds = BOUNDS if rng.random() < 0.8 else [b for b in BOUNDS if rng.random() < 0.7]
            doc["model_histogram_metrics"][f"{MODEL}/{H[name]}"] = {
                "count": count, "sum": sums[name], "buckets": _hist(rng, series[name], bounds)}
        redis.zadd(f"tre:v2:hist:{key}", {json.dumps(doc): ts})
        for tick in (ts - 1_000, ts):
            redis.zadd(f"tre:v2:inst:{key}", {json.dumps({
                "pod_name": pod, "timestamp": tick, "scraped_ms": tick - 30,
                "model_metrics": {f"{MODEL}/num_requests_running": float(rng.randint(0, 8)),
                                  f"{MODEL}/num_requests_waiting": float(rng.randint(0, 4)),
                                  f"{MODEL}/gpu_cache_usage_perc": rng.random()}}): tick})


def _stores(cls, redis):
    registry = load_registry(str(REGISTRY_PATH))
    controller = cls(redis, registry, instant_sample_interval_ms=1_000, evidence_period_ms=2_000,
                     min_sample_ratio=0.5, min_latency_samples=5, suffix_period_ms=4_000)
    evidence = cls(redis, registry, instant_sample_interval_ms=1_000, evidence_period_ms=2_000,
                   read_instant=False, histogram_lookback_ms=0, min_latency_samples=5)
    return controller, evidence


def test_bucket_delta_matches_the_per_bound_scan():
    rng = random.Random(7)
    for _ in range(500):
        def side():
            keys = rng.sample([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, float("inf")], rng.randint(0, 7))
            return {key: float(rng.randint(0, 50)) for key in keys}  # also non-monotone (resets)
        first, last = side(), side()
        assert ms._bucket_delta(first, last) == _ref_bucket_delta(first, last)


def test_merged_pod_histograms_match_the_per_bound_scan():
    rng = random.Random(11)
    for _ in range(300):
        sets = [tuple((u, float(rng.randint(0, 9))) for u in rng.sample([0.1, 0.5, 1.0, 2.0, float("inf")], 3))
                for _ in range(rng.randint(1, 4))]
        uppers = sorted({u for buckets in sets for u, _ in buckets})
        assert _merged_cumulative(list(sets), uppers) == _ref_merged(sets)
        pods = [SimpleNamespace(e2e_hist=s, e2e_hist_count=9.0) for s in sets]
        expected = histogram_percentile(_ref_merged(sets), 0.95, mode="bucket_upper")
        assert merged_hist_p95_ms(pods, ("bucket_upper", 0)) == (None if expected is None else expected * 1000.0)


def test_whole_windows_are_byte_identical_to_the_former_aggregation():
    end = 1_000_000
    covered = {"pods": 0, "suffixes": 0, "p95": 0, "windows": 0}
    for seed in range(6):
        redis = FakeRedis()
        rng = random.Random(seed)
        for pod in ("a", "b", "c"):
            _fill(redis, rng, pod, end)
        new_ctl, new_ev = _stores(MetricsStore, redis)
        ref_ctl, ref_ev = _stores(_ReferenceStore, redis)
        for window_end in range(end - 30_000, end + 1, 2_000):
            for start in (window_end - 20_000, window_end - 30_000):
                new = new_ctl.read_model_window(MODEL, start, window_end, start_exclusive=True)
                ref = ref_ctl.read_model_window(MODEL, start, window_end, start_exclusive=True)
                assert pickle.dumps(new) == pickle.dumps(ref), (seed, start, window_end)
                covered["windows"] += 1
                covered["pods"] += bool(new.per_pod)
                covered["suffixes"] += bool(new.suffix_windows)
                covered["p95"] += any(pod.ttft_p95_ms is not None for pod in new.per_pod.values())
                new = new_ev.read_model_window(MODEL, start - 1, window_end, start_exclusive=True)
                ref = ref_ev.read_model_window(MODEL, start - 1, window_end, start_exclusive=True)
                assert pickle.dumps(new) == pickle.dumps(ref), (seed, start, window_end)
    # The fixtures reach every path: pods kept, O1 suffixes built, p95 formed.
    assert covered["pods"] > covered["windows"] // 2 and covered["suffixes"] > 0 and covered["p95"] > 0, covered

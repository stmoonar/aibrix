
import json

from pathlib import Path

from tre_common.registry import load_registry
from tre_controller.store.metrics_store import MetricsStore
from make_redis_fixture import FakeRedis as FixtureRedis, populate_edge_case_fixture
from golden.legacy_collector import legacy_collect_model_window


TRE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = TRE_ROOT / "deploy" / "registry.yaml"


class FakeRedis:
    def __init__(self):
        self.sets = {}
        self.zsets = {}
        self.strings = {}
        self.zrange_calls = 0
        self.scan_calls = 0

    def sadd(self, key, *values):
        self.sets.setdefault(key, set()).update(values)

    def zadd(self, key, mapping):
        self.zsets.setdefault(key, [])
        for member, score in mapping.items():
            self.zsets[key].append((float(score), member))
        self.zsets[key].sort(key=lambda item: item[0])

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def zrangebyscore(self, key, minimum, maximum):
        self.zrange_calls += 1
        lo = float(minimum)
        hi = float(maximum)
        return [member for score, member in self.zsets.get(key, []) if lo <= score <= hi]

    def set(self, key, value):
        self.strings[key] = value

    def mget(self, keys):
        return [self.strings.get(key) for key in keys]

    def scan_iter(self, match):
        self.scan_calls += 1
        if not match.endswith("*"):
            return iter(())
        prefix = match[:-1]
        return (key for key in sorted(self.strings) if key.startswith(prefix))


def add_doc(redis, key, ts_ms, doc):
    body = dict(doc)
    body["timestamp"] = ts_ms
    redis.zadd(key, {json.dumps(body, sort_keys=True): ts_ms})


def set_legacy_doc(redis, key, ts_ms, doc):
    body = dict(doc)
    body["timestamp"] = ts_ms
    redis.set(key, json.dumps(body, sort_keys=True))


def hist_doc(pod, prompt_sum, prompt_count, ttft_sum, ttft_count, ttft_buckets):
    return {
        "pod_name": pod,
        "model_histogram_metrics": {
            "dsqwen-7b/request_prompt_tokens": {"sum": prompt_sum, "count": prompt_count, "buckets": {"1": prompt_count}},
            "dsqwen-7b/time_to_first_token_seconds": {
                "sum": ttft_sum,
                "count": ttft_count,
                "buckets": ttft_buckets,
            },
        },
    }


def inst_doc(pod, waiting, running, kv_hit):
    return {
        "pod_name": pod,
        "model_metrics": {
            "dsqwen-7b/num_requests_waiting": waiting,
            "dsqwen-7b/num_requests_running": running,
            "dsqwen-7b/kv_cache_hit_rate": kv_hit,
        },
    }


def test_metrics_store_reads_v2_window_with_legacy_delta_semantics():
    redis = FakeRedis()
    pod_a = "default/pod-a"
    pod_b = "default/pod-b"
    redis.sadd("tre:v2:pods:dsqwen-7b", pod_a, pod_b)

    add_doc(redis, "tre:v2:hist:" + pod_a, 1_000, hist_doc("pod-a", 10, 1, 1.0, 10, {"0.1": 4, "0.5": 10}))
    add_doc(redis, "tre:v2:hist:" + pod_a, 11_000, hist_doc("pod-a", 70, 2, 8.0, 20, {"0.1": 6, "0.5": 20}))
    add_doc(redis, "tre:v2:hist:" + pod_b, 1_000, hist_doc("pod-b", 100, 1, 2.0, 5, {"0.2": 2, "0.7": 5}))
    add_doc(redis, "tre:v2:hist:" + pod_b, 11_000, hist_doc("pod-b", 145, 2, 7.0, 15, {"0.2": 7, "0.7": 15}))

    add_doc(redis, "tre:v2:inst:" + pod_a, 1_000, inst_doc("pod-a", waiting=2, running=1, kv_hit=0.50))
    add_doc(redis, "tre:v2:inst:" + pod_a, 6_000, inst_doc("pod-a", waiting=4, running=3, kv_hit=0.75))
    add_doc(redis, "tre:v2:inst:" + pod_b, 1_000, inst_doc("pod-b", waiting=1, running=2, kv_hit=1.00))
    add_doc(redis, "tre:v2:inst:" + pod_b, 6_000, inst_doc("pod-b", waiting=3, running=4, kv_hit=0.50))

    registry = load_registry(str(REGISTRY_PATH))
    store = MetricsStore(redis, registry, instant_sample_interval_ms=5_000, percentile_mode="bucket_upper")

    metrics = store.read_model_window("dsqwen-7b", 1_000, 11_000)

    assert metrics.model == "dsqwen-7b"
    assert metrics.prompt_tokens == 105.0
    assert metrics.avg_waiting == 5.0
    assert metrics.avg_running == 5.0
    assert metrics.kv_cache_hit_rate == 0.6875
    assert metrics.ttft_p95_ms == 700.0
    assert metrics.routable_pods == 2
    assert metrics.assigned_replicas == 2
    assert metrics.per_pod["pod-a"].prompt_tokens == 60.0
    assert metrics.per_pod["pod-b"].ttft_p95_ms == 700.0


def test_metrics_store_caches_completed_windows():
    redis = FakeRedis()
    pod = "default/pod-a"
    redis.sadd("tre:v2:pods:dsqwen-7b", pod)
    add_doc(redis, "tre:v2:hist:" + pod, 1_000, hist_doc("pod-a", 1, 1, 1.0, 1, {"0.5": 1}))
    add_doc(redis, "tre:v2:hist:" + pod, 11_000, hist_doc("pod-a", 3, 2, 2.0, 2, {"0.5": 2}))

    store = MetricsStore(redis, load_registry(str(REGISTRY_PATH)), instant_sample_interval_ms=5_000)

    first = store.read_model_window("dsqwen-7b", 1_000, 11_000)
    calls_after_first = redis.zrange_calls
    second = store.read_model_window("dsqwen-7b", 1_000, 11_000)

    assert first == second
    assert redis.zrange_calls == calls_after_first


def test_metrics_store_sliding_reads_do_not_grow_window_cache():
    # S1.1: sliding windows end at a moving `now`, so every [start, end] is unique.
    # With use_cache=False the per-window cache must not accumulate entries (no leak).
    redis = FakeRedis()
    pod = "default/pod-a"
    redis.sadd("tre:v2:pods:dsqwen-7b", pod)
    add_doc(redis, "tre:v2:hist:" + pod, 1_000, hist_doc("pod-a", 1, 1, 1.0, 1, {"0.5": 1}))
    add_doc(redis, "tre:v2:hist:" + pod, 11_000, hist_doc("pod-a", 3, 2, 2.0, 2, {"0.5": 2}))

    store = MetricsStore(redis, load_registry(str(REGISTRY_PATH)), instant_sample_interval_ms=5_000)

    for end in range(11_000, 11_100):  # 100 distinct sliding windows
        store.read_model_window("dsqwen-7b", end - 10_000, end, use_cache=False)

    assert len(store._window_cache) == 0


def test_metrics_store_reads_v1_legacy_keys_without_pod_set():
    redis = FakeRedis()
    pod = "default/pod-a"
    start_ms = 1_700_000_001_000
    mid_ms = 1_700_000_006_000
    end_ms = 1_700_000_011_000
    set_legacy_doc(redis, f"aibrix:pod_histogram_metrics_{pod}_{start_ms}", start_ms, hist_doc("pod-a", 10, 1, 1.0, 10, {"0.5": 10}))
    set_legacy_doc(redis, f"aibrix:pod_histogram_metrics_{pod}_{end_ms}", end_ms, hist_doc("pod-a", 42, 2, 4.0, 20, {"0.5": 20}))
    set_legacy_doc(redis, f"aibrix:pod_instant_metrics_{pod}_{start_ms}", start_ms, inst_doc("pod-a", waiting=2, running=3, kv_hit=0.25))
    set_legacy_doc(redis, f"aibrix:pod_instant_metrics_{pod}_{mid_ms}", mid_ms, inst_doc("pod-a", waiting=4, running=5, kv_hit=0.75))

    store = MetricsStore(
        redis,
        load_registry(str(REGISTRY_PATH)),
        instant_sample_interval_ms=5_000,
        schema="v1",
    )

    metrics = store.read_model_window("dsqwen-7b", start_ms, end_ms)

    assert metrics.prompt_tokens == 32.0
    assert metrics.avg_waiting == 3.0
    assert metrics.avg_running == 4.0
    assert metrics.kv_cache_hit_rate == 0.5
    assert metrics.routable_pods == 1
    assert redis.scan_calls == 2


def test_metrics_store_uses_pre_window_v2_baseline_for_single_hist_doc_delta():
    redis = FakeRedis()
    pod = "default/pod-a"
    redis.sadd("tre:v2:pods:dsqwen-7b", pod)
    add_doc(redis, "tre:v2:hist:" + pod, 1_000, hist_doc("pod-a", 10, 1, 1.0, 10, {"0.5": 10}))
    add_doc(redis, "tre:v2:hist:" + pod, 11_000, hist_doc("pod-a", 42, 2, 4.0, 20, {"0.5": 20}))

    store = MetricsStore(redis, load_registry(str(REGISTRY_PATH)), instant_sample_interval_ms=5_000)

    metrics = store.read_model_window("dsqwen-7b", 10_000, 20_000)

    assert metrics.prompt_tokens == 32.0
    assert metrics.ttft_p95_ms == 500.0


def test_metrics_store_returns_none_tokens_when_window_has_no_hist_docs():
    redis = FakeRedis()
    pod = "default/pod-a"
    redis.sadd("tre:v2:pods:dsqwen-7b", pod)
    add_doc(redis, "tre:v2:hist:" + pod, 1_000, hist_doc("pod-a", 10, 1, 1.0, 10, {"0.5": 10}))
    add_doc(redis, "tre:v2:inst:" + pod, 11_000, inst_doc("pod-a", waiting=2, running=1, kv_hit=0.50))

    store = MetricsStore(redis, load_registry(str(REGISTRY_PATH)), instant_sample_interval_ms=5_000)

    metrics = store.read_model_window("dsqwen-7b", 10_000, 20_000)

    assert metrics.prompt_tokens is None
    assert metrics.generation_tokens is None
    assert metrics.avg_waiting == 1.0


def test_metrics_store_uses_pre_window_v1_baseline_for_single_hist_doc_delta():
    redis = FakeRedis()
    pod = "default/pod-a"
    baseline_ms = 1_700_000_001_000
    end_ms = 1_700_000_011_000
    set_legacy_doc(redis, f"aibrix:pod_histogram_metrics_{pod}_{baseline_ms}", baseline_ms, hist_doc("pod-a", 10, 1, 1.0, 10, {"0.5": 10}))
    set_legacy_doc(redis, f"aibrix:pod_histogram_metrics_{pod}_{end_ms}", end_ms, hist_doc("pod-a", 42, 2, 4.0, 20, {"0.5": 20}))

    store = MetricsStore(redis, load_registry(str(REGISTRY_PATH)), instant_sample_interval_ms=5_000, schema="v1")

    metrics = store.read_model_window("dsqwen-7b", baseline_ms + 5_000, baseline_ms + 20_000)

    assert metrics.prompt_tokens == 32.0
    assert metrics.ttft_p95_ms == 500.0


def test_metrics_store_snapshot_reads_all_registry_models_from_fixture():
    redis = FixtureRedis()
    window = populate_edge_case_fixture(redis)
    store = MetricsStore(redis, load_registry(str(REGISTRY_PATH)), instant_sample_interval_ms=5_000)

    snapshot = store.read_snapshot(window.start_ms, window.end_ms)

    assert snapshot.ts_ms == window.end_ms
    assert snapshot.stale is False
    assert set(snapshot.models) == {"dsqwen-7b", "dsllama-8b", "dsqwen-14b"}
    assert snapshot.models["dsqwen-7b"].prompt_tokens == 32.0
    assert snapshot.models["dsqwen-7b"].avg_waiting == 3.0
    assert snapshot.models["dsllama-8b"].prompt_tokens == 0.0
    assert snapshot.models["dsllama-8b"].avg_waiting == 3.0
    assert snapshot.models["dsqwen-14b"].routable_pods == 0


def test_metrics_store_matches_legacy_formula_on_edge_fixture():
    redis = FixtureRedis()
    window = populate_edge_case_fixture(redis)
    registry = load_registry(str(REGISTRY_PATH))
    store = MetricsStore(redis, registry, instant_sample_interval_ms=5_000)

    current = store.read_model_window("dsqwen-7b", window.start_ms, window.end_ms)
    legacy = legacy_collect_model_window(redis, "dsqwen-7b", window.start_ms, window.end_ms, instant_sample_interval_ms=5_000)

    assert current.prompt_tokens == legacy.prompt_tokens
    assert current.avg_waiting == legacy.avg_waiting
    assert current.avg_running == legacy.avg_running
    assert current.kv_cache_hit_rate == legacy.kv_cache_hit_rate
    assert current.ttft_p95_ms == legacy.ttft_p95_ms


import json
from dataclasses import dataclass


@dataclass(frozen=True)
class FixtureWindow:
    start_ms: int
    mid_ms: int
    end_ms: int


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


def add_v2_doc(redis, key, ts_ms, doc):
    body = dict(doc)
    body["timestamp"] = ts_ms
    redis.zadd(key, {json.dumps(body, sort_keys=True): ts_ms})


def histogram_doc(model, pod, prompt_sum, prompt_count, ttft_sum=0.0, ttft_count=0, ttft_buckets=None):
    if ttft_buckets is None:
        ttft_buckets = {"0.5": ttft_count}
    return {
        "pod_name": pod,
        "model_histogram_metrics": {
            f"{model}/request_prompt_tokens": {"sum": prompt_sum, "count": prompt_count, "buckets": {"1": prompt_count}},
            f"{model}/time_to_first_token_seconds": {
                "sum": ttft_sum,
                "count": ttft_count,
                "buckets": ttft_buckets,
            },
        },
    }


def instant_doc(model, pod, waiting=0.0, running=0.0, kv_hit=0.0):
    return {
        "pod_name": pod,
        "model_metrics": {
            f"{model}/num_requests_waiting": waiting,
            f"{model}/num_requests_running": running,
            f"{model}/kv_cache_hit_rate": kv_hit,
        },
    }


def populate_edge_case_fixture(redis):
    window = FixtureWindow(start_ms=1_700_000_001_000, mid_ms=1_700_000_006_000, end_ms=1_700_000_011_000)

    model = "dsqwen-7b"
    pod = "default/dsqwen-7b-pod-a"
    redis.sadd(f"tre:v2:pods:{model}", pod)
    # Intentionally write out of order. The store must sort by document timestamp.
    add_v2_doc(redis, f"tre:v2:hist:{pod}", window.end_ms, histogram_doc(model, "dsqwen-7b-pod-a", 42, 2, 4.0, 20, {"0.5": 20}))
    add_v2_doc(redis, f"tre:v2:hist:{pod}", window.start_ms, histogram_doc(model, "dsqwen-7b-pod-a", 10, 1, 1.0, 10, {"0.5": 10}))
    add_v2_doc(redis, f"tre:v2:inst:{pod}", window.start_ms, instant_doc(model, "dsqwen-7b-pod-a", waiting=2, running=3, kv_hit=0.25))
    add_v2_doc(redis, f"tre:v2:inst:{pod}", window.mid_ms, instant_doc(model, "dsqwen-7b-pod-a", waiting=4, running=5, kv_hit=0.75))

    reset_model = "dsllama-8b"
    reset_pod = "default/dsllama-8b-pod-a"
    redis.sadd(f"tre:v2:pods:{reset_model}", reset_pod)
    add_v2_doc(redis, f"tre:v2:hist:{reset_pod}", window.start_ms, histogram_doc(reset_model, "dsllama-8b-pod-a", 50, 5, 1.0, 5, {"0.5": 5}))
    # Counter reset: prompt sum goes down. Token-rate signals must mark this window unavailable.
    add_v2_doc(redis, f"tre:v2:hist:{reset_pod}", window.end_ms, histogram_doc(reset_model, "dsllama-8b-pod-a", 30, 2, 0.5, 2, {"0.5": 2}))
    # Missing one instant sample: old semantics still divide by expected sample count.
    add_v2_doc(redis, f"tre:v2:inst:{reset_pod}", window.mid_ms, instant_doc(reset_model, "dsllama-8b-pod-a", waiting=6, running=2, kv_hit=1.0))
    return window


def populate_large_fixture(redis, *, start_ms=1_700_001_000_000, duration_minutes=30, sample_interval_ms=5_000, pods_per_model=8):
    models = ("dsqwen-7b", "dsllama-8b", "dsqwen-14b")
    samples = int(duration_minutes * 60_000 / sample_interval_ms) + 1
    for model_index, model in enumerate(models):
        for pod_index in range(pods_per_model):
            pod_key = f"default/{model}-pod-{pod_index}"
            redis.sadd(f"tre:v2:pods:{model}", pod_key)
            prompt_base = 1_000.0 * (model_index + 1) + pod_index * 10.0
            ttft_base = 100.0 * (model_index + 1) + pod_index
            for sample_index in range(samples):
                ts_ms = start_ms + sample_index * sample_interval_ms
                prompt_sum = prompt_base + sample_index * (model_index + 1) * (pod_index + 1)
                prompt_count = 10 + sample_index
                ttft_count = 100 + sample_index
                add_v2_doc(
                    redis,
                    f"tre:v2:hist:{pod_key}",
                    ts_ms,
                    histogram_doc(
                        model,
                        f"{model}-pod-{pod_index}",
                        prompt_sum=prompt_sum,
                        prompt_count=prompt_count,
                        ttft_sum=ttft_base + sample_index * 0.05,
                        ttft_count=ttft_count,
                        ttft_buckets={"0.1": ttft_count * 0.5, "0.5": ttft_count},
                    ),
                )
                add_v2_doc(
                    redis,
                    f"tre:v2:inst:{pod_key}",
                    ts_ms,
                    instant_doc(
                        model,
                        f"{model}-pod-{pod_index}",
                        waiting=float((sample_index + pod_index) % 7),
                        running=float(1 + (sample_index + model_index) % 5),
                        kv_hit=0.5 + 0.01 * ((sample_index + pod_index) % 10),
                    ),
                )
    return FixtureWindow(start_ms=start_ms, mid_ms=start_ms + sample_interval_ms, end_ms=start_ms + (samples - 1) * sample_interval_ms)

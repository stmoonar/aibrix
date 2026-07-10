from dataclasses import FrozenInstanceError

import pytest

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics, PodWindowMetrics
from tre_common import rediskeys


def test_redis_key_builders_match_v2_schema():
    assert rediskeys.hist_key("pod-a") == "tre:v2:hist:pod-a"
    assert rediskeys.inst_key("pod-a") == "tre:v2:inst:pod-a"
    assert rediskeys.pods_key("dsqwen-7b") == "tre:v2:pods:dsqwen-7b"
    assert rediskeys.DECISION_LATEST_KEY == "tre:v2:decision:latest"
    assert rediskeys.SM_STATE_KEY == "tre:v2:sm:state"
    assert rediskeys.SM_VERSION_KEY == "tre:v2:sm:version"
    assert rediskeys.CONTROLLER_ORPHAN_WATCH_KEY == "tre:v2:controller:orphan_watch"
    assert (
        rediskeys.CONTROLLER_HIDDEN_ORPHAN_ALERTS_KEY
        == "tre:v2:controller:alerts:hidden_orphans"
    )
    assert rediskeys.RETENTION_MS == 30 * 60 * 1000
    assert rediskeys.FALLBACK_TTL_SECONDS == 2 * 60 * 60


def test_metrics_snapshot_schema_is_immutable_and_nested_by_model_and_pod():
    pod = PodWindowMetrics(
        pod="pod-a",
        prompt_tokens=10.0,
        generation_tokens=20.0,
        avg_waiting=1.0,
        avg_running=2.0,
        avg_swapping=0.0,
        kv_cache_hit_rate=0.8,
        ttft_p95_ms=100.0,
        tpot_p95_ms=20.0,
        e2e_p95_ms=500.0,
    )
    model = ModelWindowMetrics("m", 1000, 2000, 10.0, 20.0, 1.0, 2.0, 0.0, 0.8, 100.0, 20.0, 500.0, 1, 2, {"pod-a": pod})
    snapshot = MetricsSnapshot(ts_ms=2000, models={"m": model}, stale=False)

    assert snapshot.models["m"].per_pod["pod-a"].kv_cache_hit_rate == 0.8

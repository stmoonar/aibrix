from types import SimpleNamespace

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_controller.loops.signal_log import SIGNAL_LOG_FIELDS, SignalLogWriter
from tre_controller.loops.tick import LoopTickResult
from tre_controller.planning.classify import ModelState
from tre_controller.planning.planner import ScaleAction


class FakeRedis:
    def __init__(self):
        self.entries = []

    def xadd(self, key, fields, maxlen=None, approximate=None):
        self.entries.append((key, dict(fields), maxlen, approximate))
        return f"{len(self.entries)}-0"


def _snapshot(ts_ms):
    models = {}
    for model in ("m1", "m2"):
        models[model] = ModelWindowMetrics(
            model=model,
            window_start_ms=ts_ms - 5_000,
            window_end_ms=ts_ms,
            prompt_tokens=10.0,
            generation_tokens=20.0,
            avg_waiting=1.0,
            avg_running=2.0,
            avg_swapping=0.0,
            kv_cache_hit_rate=0.5,
            ttft_p95_ms=100.0,
            tpot_p95_ms=20.0,
            e2e_p95_ms=500.0,
            routable_pods=1,
            assigned_replicas=1,
            per_pod={},
        )
    return MetricsSnapshot(ts_ms=ts_ms, models=models, stale=False)


def _result():
    contexts = {
        model: {
            "signal_source": "zm",
            "signal_raw_value": 0.8,
            "z_m": 0.8,
            "trs_z_m": 0.8,
            "trs": 125.0,
            "theta_m": 100.0,
            "Q": 4.0,
            "routable_pods": 1,
            "eta_m": 5.0,
        }
        for model in ("m1", "m2")
    }
    classifications = {
        "m1": SimpleNamespace(state=ModelState.CRITICAL),
        "m2": SimpleNamespace(state=ModelState.HEALTHY),
    }
    return LoopTickResult(
        submitted=1,
        actions=(ScaleAction("m1", 1, "critical", "rescue"),),
        model_contexts=contexts,
        classifications=classifications,
    )


def test_signal_log_writes_one_complete_row_per_model_and_window():
    redis = FakeRedis()
    writer = SignalLogWriter(redis)

    assert writer.write(_snapshot(10_000), _result()) == 2
    assert writer.write(_snapshot(10_000), _result()) == 0
    assert writer.write(_snapshot(15_000), _result()) == 2

    assert len(redis.entries) == 4
    for key, fields, maxlen, approximate in redis.entries:
        assert key == "tre:v2:controller:signal_log"
        assert tuple(fields) == SIGNAL_LOG_FIELDS
        assert all(value != "" for value in fields.values())
        assert maxlen == 200_000
        assert approximate is True
    first = redis.entries[0][1]
    assert first["raw_signal"] == "125.0"
    assert first["theta"] == "100.0"
    assert first["z"] == "0.8"
    assert first["decode_tps"] == "nan"
    assert first["replicas_target"] == "2"
    assert first["tier"] == "crit"
    assert first["action"] == "scale_up"

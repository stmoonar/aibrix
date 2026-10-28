from __future__ import annotations

import json

from tre_common.metrics_schema import MetricsSnapshot
from tre_controller.loops.decision_snapshot import DecisionSnapshotWriter, build_decision_snapshot
from tre_controller.loops.tick import LoopTickResult
from tre_controller.planning.planner import DefragAction, ScaleAction
from tre_sm.allocator.slots import Migration, Slot


class FakeRedis:
    def __init__(self) -> None:
        self.zadds: list[tuple[str, dict]] = []

    def zadd(self, name: str, mapping: dict) -> int:
        self.zadds.append((name, mapping))
        return len(mapping)

    def zremrangebyscore(self, name: str, mn, mx) -> int:
        return 0

    def expire(self, name: str, ttl: int) -> bool:
        return True


def test_build_decision_snapshot_serializes_actions_and_events() -> None:
    snapshot = MetricsSnapshot(ts_ms=1234, stale=False, models={})
    result = LoopTickResult(
        submitted=2,
        actions=(
            ScaleAction(
                model="critical",
                delta=1,
                reason="critical_tp_defrag",
                source_loop="rescue",
                receiver="critical",
            ),
            DefragAction(
                migrations=(Migration("serve-2", Slot("node-a", (2,)), Slot("node-a", (1,))),),
                reason="critical_tp_defrag",
                source_loop="rescue",
            ),
        ),
        events=("capacity_blocked:other",),
        model_contexts={
            "critical": {
                "z_m": 0.5,
                "trs_z_m": 0.5,
                "signal_source": "zm",
                "signal_unavailable_reason": None,
            }
        },
    )

    payload = build_decision_snapshot("rescue", snapshot, result)

    assert payload["ts_ms"] == "1234"
    assert payload["loop"] == "rescue"
    assert payload["stale"] == "false"
    assert payload["submitted"] == "2"
    assert json.loads(payload["events"]) == ["capacity_blocked:other"]
    assert json.loads(payload["model_states"]) == {
        "critical": {
            "z_m": 0.5,
            "trs_z_m": 0.5,
            "trs": None,
            "trs_raw": None,
            "q_ctl": None,
            "y_m": None,
            "eta_m": None,
            "theta_m": None,
            "routable_pods": None,
            "assigned_replicas": None,
            "signal_warm": None,
            "signal_full_window": None,
            "signal_breakpoint_ms": None,
            "signal_window_start_ms": None,
            "signal_evidence_ms": None,
            "signal_hold_reason": None,
            "sample_quality_hold": None,
            "state": None,
            "signal_source": "zm",
            "signal_unavailable_reason": None,
            "window_end_ms": None,
        }
    }
    assert json.loads(payload["actions"]) == [
        {
            "kind": "scale",
            "model": "critical",
            "delta": 1,
            "reason": "critical_tp_defrag",
            "source_loop": "rescue",
            "requires_safescale": False,
            "receiver": "critical",
            "donor": None,
        },
        {
            "kind": "defrag",
            "reason": "critical_tp_defrag",
            "source_loop": "rescue",
            "migrations": [
                {
                    "serve_id": "serve-2",
                    "from_slot": {"node": "node-a", "gpu_ids": [2]},
                    "to_slot": {"node": "node-a", "gpu_ids": [1]},
                }
            ],
        },
    ]


def test_decision_snapshot_writer_appends_per_model_history() -> None:
    from tre_common.rediskeys import decision_hist_key

    redis = FakeRedis()
    writer = DecisionSnapshotWriter(redis)
    snapshot = MetricsSnapshot(ts_ms=1000, stale=False, models={})
    result = LoopTickResult(submitted=0, model_contexts={"m": {"z_m": 1.2, "trs": 500.0, "trs_raw": 640.0}})

    writer.write("rescue", snapshot, result)

    # one zadd to the model's per-model hist zset, scored by ts_ms; member carries the signal.
    assert len(redis.zadds) == 1
    name, mapping = redis.zadds[0]
    assert name == decision_hist_key("m")
    member, score = next(iter(mapping.items()))
    assert score == 1000.0
    decoded = json.loads(member)
    assert decoded["model"] == "m" and decoded["z_m"] == 1.2 and decoded["trs"] == 500.0
    assert decoded["trs_raw"] == 640.0  # pre-EMA TSS next to the EMA, for the calibration capture


def test_decision_snapshot_writer_logs_even_when_redis_write_fails(caplog) -> None:
    import logging

    class FailingRedis(FakeRedis):
        def zadd(self, name: str, mapping: dict) -> int:
            raise RuntimeError("redis unavailable")

        def xadd(self, *args, **kwargs):
            raise RuntimeError("redis unavailable")

    writer = DecisionSnapshotWriter(FailingRedis())
    snapshot = MetricsSnapshot(ts_ms=42, stale=False, models={})
    result = LoopTickResult(submitted=0, events=("redis_unavailable",), model_contexts={"m": {"z_m": 1.0}})

    with caplog.at_level(logging.INFO, logger="tre_controller.decision"):
        writer.write("rescue", snapshot, result)

    assert any("trs_calc_result" in record.getMessage() for record in caplog.records)


def test_decision_snapshot_writer_logs_trs_calc_result(caplog) -> None:
    import logging

    redis = FakeRedis()
    writer = DecisionSnapshotWriter(redis)
    snapshot = MetricsSnapshot(ts_ms=99, stale=False, models={})
    result = LoopTickResult(submitted=1, actions=(ScaleAction(model="m1", delta=-1, reason="idle", source_loop="rescue"),))

    with caplog.at_level(logging.INFO, logger="tre_controller.decision"):
        writer.write("rescue", snapshot, result)

    assert any("trs_calc_result" in record.getMessage() for record in caplog.records)
    payload = json.loads(next(record.getMessage() for record in caplog.records if "trs_calc_result" in record.getMessage()))
    assert payload["event"] == "trs_calc_result"
    assert payload["loop"] == "rescue"
    assert payload["submitted"] == "1"
    assert json.loads(payload["actions"])[0]["model"] == "m1"


class RecordingRedis:
    """Every command in order; with ``pipelined`` the round trips too."""

    def __init__(self, pipelined: bool, fail: set[str] = frozenset()) -> None:
        self.commands: list[tuple] = []
        self.round_trips = 0
        self.fail = set(fail)
        if pipelined:
            self.pipeline = self._pipeline

    def _do(self, name, *args, **kwargs):
        self.commands.append((name, args, tuple(sorted(kwargs.items()))))
        if name in self.fail:
            raise RuntimeError(f"{name} failed")
        return 1

    def __getattr__(self, name):
        if name in ("zadd", "zremrangebyscore", "expire", "xadd"):
            return lambda *args, **kwargs: self._do(name, *args, **kwargs)
        raise AttributeError(name)

    def _pipeline(self, transaction=True):
        assert transaction is False
        redis, queued = self, []

        class _Pipe:
            def __getattr__(self, name):
                return lambda *args, **kwargs: queued.append((name, args, kwargs))

            def execute(self, raise_on_error=True):
                assert raise_on_error is False
                redis.round_trips += 1
                out = []
                for name, args, kwargs in queued:
                    try:
                        out.append(redis._do(name, *args, **kwargs))
                    except RuntimeError as exc:
                        out.append(exc)
                return out

        return _Pipe()


def _two_model_decision(ts_ms: int):
    from tre_common.metrics_schema import ModelWindowMetrics

    models = {
        m: ModelWindowMetrics(model=m, window_start_ms=ts_ms - 20_000, window_end_ms=ts_ms, prompt_tokens=1.0,
                              generation_tokens=2.0, avg_waiting=0.0, avg_running=1.0, avg_swapping=0.0,
                              kv_cache_hit_rate=0.0, ttft_p95_ms=None, tpot_p95_ms=None, e2e_p95_ms=None,
                              routable_pods=1, assigned_replicas=1, per_pod={})
        for m in ("m1", "m2")
    }
    contexts = {m: {"z_m": 1.0, "trs": 10.0, "signal_source": "zm", "routable_pods": 1} for m in models}
    return MetricsSnapshot(ts_ms=ts_ms, stale=False, models=models), LoopTickResult(submitted=0, model_contexts=contexts)


def test_one_decision_is_one_round_trip_with_the_same_commands() -> None:
    inline, pipelined = RecordingRedis(pipelined=False), RecordingRedis(pipelined=True)
    for redis in (inline, pipelined):
        writer = DecisionSnapshotWriter(redis)
        for ts in (20_000, 21_000):
            writer.write("rescue", *_two_model_decision(ts))
    assert pipelined.round_trips == 2
    assert pipelined.commands == inline.commands
    assert [c[0] for c in pipelined.commands[:7]] == [
        "zadd", "zremrangebyscore", "expire", "zadd", "zremrangebyscore", "expire", "xadd"]


def test_with_an_executor_redis_is_written_off_the_caller_in_decision_order() -> None:
    class QueueExecutor:
        def __init__(self) -> None:
            self.jobs: list = []

        def submit(self, fn, *args):
            self.jobs.append((fn, args))

    redis, executor = RecordingRedis(pipelined=True), QueueExecutor()
    writer = DecisionSnapshotWriter(redis, executor=executor)
    writer.write("rescue", *_two_model_decision(20_000))
    writer.write("rescue", *_two_model_decision(21_000))
    assert redis.commands == [] and len(executor.jobs) == 2  # the caller never waited for redis
    for fn, args in executor.jobs:
        fn(*args)
    reference = RecordingRedis(pipelined=True)
    inline = DecisionSnapshotWriter(reference)
    inline.write("rescue", *_two_model_decision(20_000))
    inline.write("rescue", *_two_model_decision(21_000))
    assert redis.commands == reference.commands


def test_a_failed_signal_log_entry_is_written_again_by_the_next_decision_on_its_window() -> None:
    redis = RecordingRedis(pipelined=True, fail={"xadd"})
    writer = DecisionSnapshotWriter(redis)
    writer.write("rescue", *_two_model_decision(20_000))
    redis.fail.clear()
    writer.write("rescue", *_two_model_decision(20_000))  # same windows: written now
    writer.write("rescue", *_two_model_decision(20_000))  # already written: not again
    assert sum(1 for c in redis.commands if c[0] == "xadd") == 4

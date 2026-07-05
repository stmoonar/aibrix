from __future__ import annotations

import json

from tre_common.metrics_schema import MetricsSnapshot
from tre_common.rediskeys import DECISION_LATEST_KEY
from tre_controller.loops.decision_snapshot import DecisionSnapshotWriter, build_decision_snapshot
from tre_controller.loops.tick import LoopTickResult
from tre_controller.planning.planner import DefragAction, ScaleAction
from tre_sm.allocator.slots import Migration, Slot


class FakeRedis:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []

    def hset(self, name: str, mapping: dict[str, str]) -> int:
        self.calls.append((name, mapping))
        return len(mapping)


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
            "signal_source": "zm",
            "signal_unavailable_reason": None,
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


def test_decision_snapshot_writer_writes_latest_hash() -> None:
    redis = FakeRedis()
    writer = DecisionSnapshotWriter(redis)
    snapshot = MetricsSnapshot(ts_ms=42, stale=True, models={})
    result = LoopTickResult(submitted=0, events=("snapshot_stale",))

    writer.write("fairness", snapshot, result)

    assert redis.calls == [
        (
            DECISION_LATEST_KEY,
            {
                "ts_ms": "42",
                "loop": "fairness",
                "stale": "true",
                "submitted": "0",
                "actions": "[]",
                "events": '["snapshot_stale"]',
                "model_states": "{}",
            },
        )
    ]


def test_decision_snapshot_writer_logs_even_when_redis_write_fails(caplog) -> None:
    import logging

    class FailingRedis(FakeRedis):
        def hset(self, name: str, mapping: dict[str, str]) -> int:
            raise RuntimeError("redis unavailable")

    writer = DecisionSnapshotWriter(FailingRedis())
    snapshot = MetricsSnapshot(ts_ms=42, stale=False, models={})
    result = LoopTickResult(submitted=0, events=("redis_unavailable",))

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

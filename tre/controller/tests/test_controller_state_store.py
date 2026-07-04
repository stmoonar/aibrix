from __future__ import annotations

import json

from tre_controller.store.state_store import ControllerStateStore


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.lists = {}
        self.deleted = []

    def hset(self, name, key=None, value=None, mapping=None):
        bucket = self.hashes.setdefault(name, {})
        if mapping is not None:
            for field, payload in mapping.items():
                bucket[str(field).encode("utf-8")] = str(payload).encode("utf-8")
            return len(mapping)
        bucket[str(key).encode("utf-8")] = str(value).encode("utf-8")
        return 1

    def hgetall(self, name):
        return dict(self.hashes.get(name, {}))

    def hdel(self, name, *keys):
        bucket = self.hashes.setdefault(name, {})
        removed = 0
        for key in keys:
            removed += 1 if bucket.pop(str(key).encode("utf-8"), None) is not None else 0
        return removed

    def rpush(self, name, *values):
        bucket = self.lists.setdefault(name, [])
        for value in values:
            bucket.append(str(value).encode("utf-8"))
        return len(bucket)

    def lrange(self, name, start, end):
        values = self.lists.get(name, [])
        end_index = None if int(end) == -1 else int(end) + 1
        return list(values[int(start):end_index])

    def delete(self, *names):
        for name in names:
            self.deleted.append(name)
            self.hashes.pop(name, None)
            self.lists.pop(name, None)


def test_controller_state_store_round_trips_unresolved_probe_and_journal() -> None:
    redis = FakeRedis()
    store = ControllerStateStore(redis)
    record = {
        "model": "donor",
        "request_id": "probe-1",
        "pods": ["pod-a"],
        "start_ms": 1_000,
        "deadline_ms": 61_000,
        "status": "probing",
        "pending_upscales": {"receiver": 1},
    }
    journal_entry = {
        "last_observation": {
            "ts_ms": 20_000,
            "ttft_p95_ms": 500.0,
            "tpot_p95_ms": 50.0,
            "z_m": 1.2,
            "q_ctl": 0.0,
            "has_traffic": True,
        }
    }

    store.save_probe("probe-1", record)
    store.append_probe_journal("probe-1", journal_entry)

    assert store.list_unresolved_probes() == [record]
    assert store.load_probe_journal("probe-1") == [journal_entry]


def test_controller_state_store_deletes_terminal_probe_without_deleting_journal() -> None:
    redis = FakeRedis()
    store = ControllerStateStore(redis)
    record = {
        "model": "donor",
        "request_id": "probe-1",
        "pods": ["pod-a"],
        "start_ms": 1_000,
        "deadline_ms": 61_000,
        "status": "probing",
    }
    store.save_probe("probe-1", record)
    store.append_probe_journal("probe-1", {"last_observation": {"ts_ms": 20_000}})

    store.delete_probe("probe-1")

    assert store.list_unresolved_probes() == []
    assert store.load_probe_journal("probe-1") == [{"last_observation": {"ts_ms": 20_000}}]


def test_controller_state_store_ignores_malformed_records() -> None:
    redis = FakeRedis()
    redis.hset("tre:v2:controller:safescale:probes", mapping={
        "valid": json.dumps({"request_id": "valid", "model": "m", "status": "probing"}),
        "terminal": json.dumps({"request_id": "terminal", "model": "m", "status": "commit"}),
        "broken": "{",
    })
    store = ControllerStateStore(redis)

    assert store.list_unresolved_probes() == [{"request_id": "valid", "model": "m", "status": "probing"}]


def test_controller_state_store_treats_redis_read_failure_as_empty_restore_state() -> None:
    class FailingRedis(FakeRedis):
        def hgetall(self, name):
            raise RuntimeError("redis unavailable")

        def lrange(self, name, start, end):
            raise RuntimeError("redis unavailable")

    store = ControllerStateStore(FailingRedis())

    assert store.list_unresolved_probes() == []
    assert store.load_probe_journal("probe-1") == []

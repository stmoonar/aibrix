"""tre_common.run_mode: controller mode and SM actuation are independent switches."""
from __future__ import annotations

import pytest

from tre_common.rediskeys import CONTROLLER_MODE_KEY, SM_ACTUATION_KEY
from tre_common.run_mode import (
    effective_mode,
    missing_key_warnings,
    parse_mode,
    read_run_modes,
    write_run_mode,
)


class FakePipeline:
    def __init__(self, redis, transaction):
        self.redis = redis
        self.transaction = transaction
        self.commands = []

    def set(self, key, value):
        self.commands.append((key, value))
        return self

    def execute(self):
        self.redis.transactions.append((self.transaction, list(self.commands)))
        for key, value in self.commands:
            self.redis.kv[key] = value
        return [True] * len(self.commands)


class FakeRedis:
    def __init__(self, kv=None):
        self.kv = dict(kv or {})
        self.transactions = []
        self.single_sets = []

    def set(self, key, value):
        self.single_sets.append((key, value))
        self.kv[key] = value
        return True

    def get(self, key):
        return self.kv.get(key)

    def pipeline(self, transaction=True):
        return FakePipeline(self, transaction)


def test_both_values_are_written_in_one_transaction():
    redis = FakeRedis()
    written = write_run_mode(redis, controller_mode="observe", sm_actuation="active")
    assert written == {"controller": "observe", "sm_actuation": "active"}
    assert redis.transactions == [(True, [(CONTROLLER_MODE_KEY, "observe"), (SM_ACTUATION_KEY, "active")])]
    assert redis.single_sets == []


def test_a_single_value_leaves_the_other_key_unchanged():
    redis = FakeRedis({CONTROLLER_MODE_KEY: "active", SM_ACTUATION_KEY: "active"})
    assert write_run_mode(redis, controller_mode="observe") == {"controller": "observe"}
    assert redis.kv == {CONTROLLER_MODE_KEY: "observe", SM_ACTUATION_KEY: "active"}
    assert write_run_mode(redis, sm_actuation="observe") == {"sm_actuation": "observe"}
    assert redis.kv == {CONTROLLER_MODE_KEY: "observe", SM_ACTUATION_KEY: "observe"}
    assert redis.transactions == []


def test_invalid_or_empty_writes_are_refused():
    redis = FakeRedis()
    with pytest.raises(ValueError):
        write_run_mode(redis, controller_mode="bogus")
    with pytest.raises(ValueError):
        write_run_mode(redis, controller_mode="active", sm_actuation="on")
    with pytest.raises(ValueError):
        write_run_mode(redis)
    assert redis.kv == {}


def test_read_and_missing_key_warnings():
    redis = FakeRedis({CONTROLLER_MODE_KEY: b"active"})
    raw = read_run_modes(redis)
    assert raw == {"controller": "active", "sm_actuation": None}
    assert missing_key_warnings(raw) == [f"{SM_ACTUATION_KEY} missing -> SM treats as observe"]
    assert missing_key_warnings({"controller": None, "sm_actuation": None}) == [
        f"{CONTROLLER_MODE_KEY} missing -> controller treats as observe",
        f"{SM_ACTUATION_KEY} missing -> SM treats as observe",
    ]
    assert missing_key_warnings({"controller": "observe", "sm_actuation": "active"}) == []


def test_parse_and_effective_mode_fail_closed():
    assert parse_mode(b" Active ") == "active"
    assert parse_mode("nope") is None
    assert effective_mode(None) == "observe"
    assert effective_mode("active") == "active"


def test_run_mode_view_reports_each_switch_and_missing_keys():
    from tre_common.run_mode import run_mode_view

    view = run_mode_view(FakeRedis({CONTROLLER_MODE_KEY: "active"}))
    assert (view["mode"], view["controller"], view["sm_actuation"]) == ("active", "active", "observe")
    assert view["raw"] == {"controller": "active", "sm_actuation": None}
    assert view["warnings"] == [f"{SM_ACTUATION_KEY} missing -> SM treats as observe"]

    class Broken:
        def get(self, key):
            raise ConnectionError("down")

    broken = run_mode_view(Broken())
    assert (broken["controller"], broken["sm_actuation"], broken["raw"]) == ("observe", "observe", None)
    assert "unreadable" in broken["warnings"][0]

"""SM maintenance lock mechanics (``tre:v2:sm:maintenance``, 2026-09-28).

Atomic acquire (Lua SET PX when free; take-over only of a recovered stale
operation or of a key without TTL), compare-and-pexpire renewal (background
thread + every safety check) and compare-and-delete release. The Python model
of the scripts (``sm_test_fakes.maintenance_lua``) runs always; set
TRE_TEST_REDIS_URL to also run the real Lua scripts - ``make check-redis``
does exactly that (and removes its throwaway container afterwards).
"""

import json
import os
import time

import pytest

from tre_common import rediskeys
from tre_sm.state.safety import (
    ClusterSafetyGate,
    MaintenanceLockBusy,
    MaintenanceLockLost,
)

from sm_test_fakes import FakeRedis

KEY = rediskeys.SM_MAINTENANCE_KEY


class _NoPressure:
    def node_pressure_reasons(self):
        return {}


def _backends():
    backends = [pytest.param(lambda: (FakeRedis(), False), id="lua-model")]
    url = os.environ.get("TRE_TEST_REDIS_URL")
    if url:
        backends.append(pytest.param(lambda: _real(url), id="real-redis-lua"))
    return backends


def _real(url):
    import redis as redis_lib

    client = redis_lib.Redis.from_url(url)
    client.delete(KEY)
    return client, True


def _gate(redis, *, ttl_s=60.0, renew_s=15.0):
    return ClusterSafetyGate(
        redis, _NoPressure(), maintenance_ttl_s=ttl_s, maintenance_renew_s=renew_s,
        wall_ms=lambda: 1234,
    )


def _pttl(redis, real):
    if real:
        return int(redis.pttl(KEY))
    if KEY not in redis.values:
        return -2
    return redis.ttls_ms.get(KEY, -1)


def _raw(redis):
    value = redis.get(KEY)
    return None if value is None else json.loads(value)


def _set_plain(redis, value):
    """A key without TTL (pre-TTL SM or hand-set)."""
    redis.set(KEY, json.dumps(value))


@pytest.mark.parametrize("make", _backends())
def test_acquire_sets_the_payload_with_a_ttl_and_release_deletes_it(make):
    redis, real = make()
    gate = _gate(redis)
    gate.acquire_maintenance("op-1", kind="fleet_repair", owner="sm-0")
    try:
        assert _raw(redis) == {"operation_id": "op-1", "kind": "fleet_repair", "owner": "sm-0", "since_ms": 1234}
        assert 0 < _pttl(redis, real) <= 60_000
        gate.assert_maintenance_held("op-1")
    finally:
        gate.release_maintenance("op-1")
    assert redis.get(KEY) is None


@pytest.mark.parametrize("make", _backends())
def test_a_live_lock_of_another_operation_is_not_taken_over(make):
    redis, real = make()
    holder = _gate(redis)
    other = _gate(redis)
    holder.acquire_maintenance("op-1", kind="fleet_repair", owner="sm-0")
    try:
        with pytest.raises(MaintenanceLockBusy) as busy:
            other.acquire_maintenance("op-2", kind="fleet_repair", owner="sm-1")
        assert busy.value.holder["operation_id"] == "op-1"
        assert _raw(redis)["operation_id"] == "op-1"
        holder.assert_maintenance_held("op-1")
        other.release_maintenance("op-2")  # not the holder: compare-and-delete leaves it
        assert _raw(redis)["operation_id"] == "op-1"
    finally:
        holder.release_maintenance("op-1")


@pytest.mark.parametrize("make", _backends())
def test_the_lock_of_a_recovered_stale_repair_is_taken_over_atomically(make):
    redis, real = make()
    dead = _gate(redis)
    dead.acquire_maintenance("op-dead", kind="fleet_repair", owner="sm-dead")
    dead._renewers.pop("op-dead").stop()  # the SM died: nobody renews any more
    recovering = _gate(redis)
    recovering.acquire_maintenance(
        "op-new", kind="fleet_repair", owner="sm-1", takeover_operation_ids=["op-dead"]
    )
    try:
        assert _raw(redis)["operation_id"] == "op-new"
        assert 0 < _pttl(redis, real) <= 60_000
        with pytest.raises(MaintenanceLockLost):
            dead.assert_maintenance_held("op-dead")
        dead.release_maintenance("op-dead")  # stale release: no effect
        recovering.assert_maintenance_held("op-new")
    finally:
        recovering.release_maintenance("op-new")
    assert redis.get(KEY) is None


@pytest.mark.parametrize("make", _backends())
def test_a_key_without_ttl_is_taken_over(make):
    redis, real = make()
    _set_plain(redis, {"operation_id": "op-legacy", "kind": "fleet_repair", "owner": "", "since_ms": 1})
    gate = _gate(redis)
    gate.acquire_maintenance("op-1", kind="fleet_repair")
    try:
        assert _raw(redis)["operation_id"] == "op-1"
        assert _pttl(redis, real) > 0
    finally:
        gate.release_maintenance("op-1")


@pytest.mark.parametrize("make", _backends())
def test_renewal_extends_the_ttl_and_an_operator_delete_is_noticed(make):
    redis, real = make()
    gate = _gate(redis, ttl_s=0.6, renew_s=0.1)
    gate.acquire_maintenance("op-1", kind="fleet_repair")
    try:
        time.sleep(1.0)  # longer than the TTL: only the renewal keeps it alive
        assert _raw(redis)["operation_id"] == "op-1"
        gate.assert_maintenance_held("op-1")
        redis.delete(KEY)  # operator abort
        deadline = time.monotonic() + 2.0
        while not gate._renewers["op-1"].lost.is_set() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert gate._renewers["op-1"].lost.is_set()
        with pytest.raises(MaintenanceLockLost):
            gate.assert_maintenance_held("op-1")
        # A lost lock stays lost even if someone else sets the key again.
        other = _gate(redis)
        other.acquire_maintenance("op-2", kind="fleet_repair")
        with pytest.raises(MaintenanceLockLost):
            gate.assert_maintenance_held("op-1")
        other.release_maintenance("op-2")
    finally:
        gate.release_maintenance("op-1")


@pytest.mark.parametrize("make", _backends())
def test_a_dead_sms_lock_expires_by_itself(make):
    redis, real = make()
    if not real:
        pytest.skip("expiry needs a real Redis clock")
    dead = _gate(redis, ttl_s=0.3, renew_s=0.1)
    dead.acquire_maintenance("op-dead", kind="fleet_repair")
    dead._renewers.pop("op-dead").stop()
    time.sleep(0.5)
    assert redis.get(KEY) is None
    gate = _gate(redis)
    gate.acquire_maintenance("op-1", kind="fleet_repair")
    gate.release_maintenance("op-1")


def test_the_renew_interval_must_be_at_most_half_the_ttl():
    with pytest.raises(ValueError):
        _gate(FakeRedis(), ttl_s=10, renew_s=6)
    with pytest.raises(ValueError):
        _gate(FakeRedis(), ttl_s=10, renew_s=0)

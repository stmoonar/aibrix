"""Per-binding sleep reservations (review P1-3).

The in-process backend and the FakeRedis re-implementation of the Lua scripts
run always; set TRE_TEST_REDIS_URL (a throwaway Redis, e.g.
``docker run --rm -p 127.0.0.1:16399:6379 redis:7.2-alpine``) to also run the
real Lua scripts.
"""

import os
import time

import pytest

from tre_sm.allocator.slots import Binding, Slot
from tre_sm.state.sleep_reservations import ReservationConflict, SleepReservations

from sm_test_fakes import FakeRedis


def _b(serve, model, gpus, node="node-a"):
    return Binding(serve, model, Slot(node, tuple(gpus)), awake=True)


def _backends():
    backends = [
        pytest.param(lambda: (SleepReservations(), None), id="memory"),
        pytest.param(lambda: _fake(), id="fake-redis-lua"),
    ]
    url = os.environ.get("TRE_TEST_REDIS_URL")
    if url:
        backends.append(pytest.param(lambda: _real(url), id="real-redis-lua"))
    return backends


def _fake():
    redis = FakeRedis()
    return SleepReservations(redis), redis


def _real(url):
    import redis as redis_lib

    client = redis_lib.Redis.from_url(url)
    client.delete("tre:v2:sm:sleep_reservations")
    return SleepReservations(client), client


@pytest.mark.parametrize("make", _backends())
def test_acquire_is_all_or_nothing_and_fences_binding_and_gpus(make):
    store, _redis = make()
    token = store.acquire([_b("a", "m1", (0,)), _b("t", "tp2", (2, 3))], owner="sm", operation_id="op", ttl_s=30)

    with pytest.raises(ReservationConflict) as same_binding:
        store.acquire([_b("a", "m1", (0,))], owner="sm", operation_id="op2", ttl_s=30)
    assert same_binding.value.binding_id == "m1/node-a/0"
    with pytest.raises(ReservationConflict):  # overlapping GPU 3 of the tp2 binding
        store.acquire([_b("x", "m9", (3,))], owner="sm", operation_id="op3", ttl_s=30)
    # all-or-nothing: the free GPU 1 was not reserved by the failed call
    with pytest.raises(ReservationConflict):
        store.acquire([_b("y", "m1", (1,)), _b("z", "m1", (0,))], owner="sm", operation_id="op4", ttl_s=30)
    assert set(store.active()) == {"m1/node-a/0", "tp2/node-a/2,3"}
    # another node's GPU 0 is free
    store.acquire([_b("o", "m1", (0,), node="node-b")], owner="sm", operation_id="op5", ttl_s=30)

    assert store.conflict(node="node-a", gpu_ids=(2,)).binding_id == "tp2/node-a/2,3"
    assert store.conflict(node="node-a", gpu_ids=(1,)) is None
    assert store.conflict(node="node-a", gpu_ids=(), model="m1").binding_id in {"m1/node-a/0", "m1/node-b/0"}
    assert store.conflict(node="node-a", gpu_ids=(), model="m7") is None

    assert store.renew(["m1/node-a/0", "tp2/node-a/2,3"], token, ttl_s=30)
    assert not store.renew(["m1/node-a/0"], "someone-else", ttl_s=30)
    store.release(["m1/node-a/0"], "someone-else")  # wrong token: no-op
    assert "m1/node-a/0" in store.active()
    store.release(["m1/node-a/0", "tp2/node-a/2,3"], token)
    assert set(store.active()) == {"m1/node-b/0"}


@pytest.mark.parametrize("make", _backends())
def test_expired_reservations_free_the_binding(make):
    store, redis = make()
    token = store.acquire([_b("a", "m1", (0,))], owner="sm", operation_id="op", ttl_s=0.05)
    if isinstance(redis, FakeRedis):
        redis.now_ms += 1000
    elif redis is None:
        store._wall_ms = lambda base=store._wall_ms(): base + 1000
    else:
        time.sleep(0.2)
    assert store.active() == {}
    assert not store.renew(["m1/node-a/0"], token, ttl_s=30)  # lost
    store.acquire([_b("a2", "m1", (0,))], owner="sm2", operation_id="op2", ttl_s=30)

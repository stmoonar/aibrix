"""POST /v2/plans (2026-10-27, design 20261027-sm-plans.md section 8): one
decision - sleep items and wake items - in ONE writer-lock hold, two waves.

One test per invariant / external rule: one awake engine per GPU across the
waves, a GPU is released only once its sleep is confirmed (a failed or
unanswered sleep blocks exactly its dependents), TP=2 waits for every GPU, the
replica floor, one lock acquisition per plan, routable_unknown before any hide,
a replayed plan is a no-op, the request rules. Concurrency is gated with events,
never timing (fakes of test_wholelock_20261002.py)."""

from __future__ import annotations

import threading

import pytest

from tre_common.registry import K8S_CALLS_PLAN_SELECT, K8S_CALLS_REFRESH, K8S_CALLS_SLEEP, K8S_CALLS_WAKE

from test_k8s_call_budget_20261006 import _count_sequential_k8s_calls
from test_wholelock_20261002 import Unanswered, World, _run

#: d-0 holds GPU 0 (r-0 sleeps under it); GPU 1 is free (x-1 sleeps there).
MIXED = (
    ("d-0", "d", (0,), "awake"),
    ("r-0", "r", (0,), "sleeping"),
    ("x-1", "x", (1,), "sleeping"),
)
#: Sleep d-0, wake x on the free GPU 1 (wave 1) and r on d-0's GPU 0 (wave 2).
MIXED_PLAN = {
    "sleeps": [{"model": "d", "pods": ["d-0"], "sleep_path": "urgent"}],
    "wakes": [{"model": "x", "at_least": 1}, {"model": "r", "at_least": 1}],
}


def post(world, plan):
    return world.client.post("/v2/plans", json=plan)


def _wake_calls(world, name):
    return [call for call in world.vllm.calls if call == ("wake_up", world.ip_of[name])]


def test_a_mixed_plan_keeps_one_engine_per_gpu_and_wakes_the_freed_gpu_after_the_sleep():
    world = World(pods=MIXED)

    response = post(world, MIXED_PLAN)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "done"
    assert set(body["phases_ms"]) == {"select", "wave1", "wave2", "settle", "refresh", "total"}
    [sleep] = body["sleeps"]
    assert sleep["status"] == "done" and sleep["taken"] == 1
    x_item, r_item = body["wakes"]
    assert [(w["serve_id"], w["wave"], w["source"]) for w in x_item["woke"]] == [("x-1", 1, "free")]
    assert [(w["serve_id"], w["wave"], w["source"]) for w in r_item["woke"]] == [("r-0", 2, "plan_sleep")]
    assert r_item["awake_before"] == 0 and r_item["awake_after"] == 1
    # r-0's /wake_up only after d-0's /sleep was answered (same GPU).
    calls = world.vllm.calls
    assert calls.index(("wake_up", world.ip_of["r-0"])) > max(
        i for i, call in enumerate(calls) if call[:2] == ("sleep", world.ip_of["d-0"])
    )
    assert not world.awake("d-0") and world.awake("x-1") and world.awake("r-0")
    assert world.lease("d-0") is None and world.lease("x-1") == world.lease("r-0") == "awake"
    assert world.violations == [] and world.journals() == ({}, {})


def test_an_unanswered_donor_sleep_keeps_its_gpu_and_blocks_only_its_dependent():
    world = World(pods=MIXED)
    real_sleep = world.vllm.sleep

    def hung(pod_ip, **kwargs):
        real_sleep(pod_ip, **kwargs)  # the engine does go to sleep ...
        return Unanswered()  # ... the HTTP call times out

    world.vllm.sleep = hung

    response = post(world, MIXED_PLAN)

    assert response.status_code == 200, response.text  # x-1 woke: something changed
    body = response.json()
    assert body["status"] == "partial"
    assert body["sleeps"][0]["status"] == "failed" and body["sleeps"][0]["retriable"] is True
    assert body["sleeps"][0]["outcomes"][0]["status"] == "unconfirmed"
    x_item, r_item = body["wakes"]
    assert x_item["status"] == "done"
    assert r_item["status"] == "refused" and r_item["error"]["reason"] == "donor_sleep_failed"
    assert _wake_calls(world, "r-0") == []  # never prepared
    assert world.lease("d-0") == "awake" and world.state("d-0") == "hidden"
    assert world.lease("r-0") is None and world.awake("x-1")
    sleep_journal, wake_journal = world.journals()
    assert sleep_journal["d-0"]["phase"] == "sleep_unconfirmed" and wake_journal == {}


def test_a_rolled_back_donor_blocks_its_dependent_and_reopens_routing():
    world = World(pods=MIXED)
    world.vllm.fail_sleep_for.add(world.ip_of["d-0"])

    body = post(world, MIXED_PLAN).json()

    x_item, r_item = body["wakes"]
    assert x_item["status"] == "done" and r_item["status"] == "refused"
    assert r_item["error"]["error"] == "donor_sleep_failed" and r_item["not_executed"] is True
    assert world.awake("d-0") and world.state("d-0") == "awake" and world.lease("d-0") == "awake"
    assert _wake_calls(world, "r-0") == [] and world.violations == []


#: A TP=2 receiver over two single-GPU donors.
TP2 = (("d-0", "d", (0,), "awake"), ("d-1", "d", (1,), "awake"), ("t-01", "t", (0, 1), "sleeping"))
TP2_PLAN = {"sleeps": [{"model": "d", "at_most": 0, "sleep_path": "scale_down"}], "wakes": [{"model": "t", "at_least": 1}]}


def test_tp2_receiver_waits_for_both_donors_and_never_wakes_when_one_fails():
    world = World(pods=TP2)
    entered, release = world.hold_inside("sleep", world.ip_of["d-1"])
    thread, box = _run(lambda: post(world, TP2_PLAN))
    assert entered.wait(5)
    assert _wake_calls(world, "t-01") == []  # d-1 is still in its /sleep
    release.set()
    thread.join(5)
    assert box["result"].status_code == 200, box["result"].text
    assert world.awake("t-01") and not world.awake("d-0") and not world.awake("d-1")
    assert world.violations == []

    world = World(pods=TP2)
    world.vllm.fail_sleep_for.add(world.ip_of["d-1"])
    body = post(world, TP2_PLAN).json()
    assert body["sleeps"][0]["status"] == "failed" and body["sleeps"][0]["taken"] == 1
    assert body["wakes"][0]["error"]["reason"] == "donor_sleep_failed"
    assert _wake_calls(world, "t-01") == [] and world.lease("t-01") is None
    assert world.awake("d-1") and world.violations == []


def test_floor_refuses_named_sleeps_with_nothing_hidden_and_clamps_at_most():
    world = World(d_min=1)

    refused = post(world, {"sleeps": [{"model": "d", "pods": ["d-0", "d-1"], "sleep_path": "urgent"}]})

    assert refused.status_code == 409 and refused.json()["error"] == "partial"
    [item] = refused.json()["sleeps"]
    assert item["status"] == "refused" and item["error"]["error"] == "floor_violation"
    assert not [p for p in world.runtime.patches if p[1] == "hidden"]

    clamped = post(world, {"sleeps": [{"model": "d", "at_most": 0, "sleep_path": "scale_down"}]})

    assert clamped.status_code == 200
    [item] = clamped.json()["sleeps"]
    assert item["status"] == "clamped" and item["taken"] == 1 and item["clamped_by_floor"] is True
    assert [world.awake("d-0"), world.awake("d-1")].count(True) == 1


def test_a_relay_takes_donors_only_for_what_free_gpus_cannot_fill():
    world = World()  # r-2 sits on the free GPU 2; r-0 / r-1 under d-0 / d-1

    body = post(world, {"wakes": [{"model": "r", "at_least": 2, "donors": [{"model": "d", "max": 2}]}]}).json()

    [item] = body["wakes"]
    assert item["status"] == "done"
    assert sorted((w["source"], w["wave"]) for w in item["woke"]) == [("donor", 2), ("free", 1)]
    assert len(item["donors_slept"]) == 1 and item["donors_slept"][0]["status"] == "slept"
    assert [world.awake("d-0"), world.awake("d-1")].count(True) == 1
    assert world.violations == [] and world.journals() == ({}, {})


def test_one_plan_is_one_writer_lock_acquisition():
    world = World(pods=MIXED)
    post(world, MIXED_PLAN)
    assert world.coordinator.kinds == ["plan"]
    [(_kind, handle)] = world.coordinator.handles
    assert handle.notes["plan"]["status"] == "done" and "phases_ms" in handle.notes


def test_an_unreadable_routable_view_is_409_before_any_hide_or_wake():
    world = World(pods=MIXED)
    world.runtime.list_pod_snapshots = lambda **kwargs: (_ for _ in ()).throw(ConnectionError("apiserver"))

    response = post(world, MIXED_PLAN)

    assert response.status_code == 409 and response.json()["error"] == "routable_unknown"
    assert not [p for p in world.runtime.patches if p[1] == "hidden"]
    assert world.vllm.calls == [] and world.awake("d-0")


def test_a_replayed_plan_is_a_noop():
    world = World(pods=MIXED)
    assert post(world, MIXED_PLAN).json()["status"] == "done"
    calls = list(world.vllm.calls)

    again = post(world, MIXED_PLAN)

    assert again.status_code == 200
    body = again.json()
    assert body["status"] == "noop"
    assert [i["status"] for i in body["sleeps"] + body["wakes"]] == ["noop", "noop", "noop"]
    assert [c for c in world.vllm.calls[len(calls):] if c[0] in ("sleep", "wake_up")] == []


@pytest.mark.parametrize("plan", [
    {"wakes": [{"model": "r", "at_least": 1}, {"model": "r", "at_least": 2}]},
    {"sleeps": [{"model": "r", "at_most": 0, "sleep_path": "urgent"}], "wakes": [{"model": "r", "at_least": 1}]},
    {"sleeps": [{"model": "d", "at_most": 0, "sleep_path": "urgent"}],
     "wakes": [{"model": "r", "at_least": 1, "donors": [{"model": "d", "max": 1}]}]},
    {"sleeps": [{"model": "d", "sleep_path": "urgent"}]},
    {"sleeps": [{"model": "d", "at_most": 0, "sleep_path": "repair"}]},
    {"wakes": [{"model": "r", "pods": ["d-0"]}]},
    {},
])
def test_a_bad_plan_is_400_and_changes_nothing(plan):
    world = World()
    response = post(world, plan)
    assert response.status_code == 400, response.text
    assert world.vllm.calls == []


def test_failed_wakes_of_both_waves_are_settled_with_one_compensating_sleep():
    world = World(pods=MIXED)
    real_wake = world.vllm.wake_up

    def late(pod_ip, *, port=None):
        real_wake(pod_ip, port=port)  # it woke ...
        from sm_test_fakes import Result
        return Result(False, "engine error")  # ... and reported a failure

    world.vllm.wake_up = late
    compensating = []
    original = world.service._compensating_sleeps
    world.service._compensating_sleeps = lambda tickets: compensating.append(len(tickets)) or original(tickets)

    body = post(world, MIXED_PLAN).json()

    assert compensating == [2]
    assert [i["status"] for i in body["wakes"]] == ["failed", "failed"]
    assert not world.awake("x-1") and not world.awake("r-0") and not world.awake("d-0")
    assert world.leases.load() == [] and world.journals() == ({}, {})


def test_a_plan_stays_within_its_k8s_budget_and_refreshes_observed_once():
    world = World(pods=(
        ("d-0", "d", (0,), "awake"), ("d-1", "d", (1,), "awake"),
        ("r-0", "r", (0,), "sleeping"), ("r-1", "r", (1,), "sleeping"),
    ))
    counted = _count_sequential_k8s_calls(world)
    lists = []

    class K8s:
        def list_pods(self):
            lists.append(1)
            return []

    world.service._k8s_client = K8s()
    body = post(world, {
        "sleeps": [{"model": "d", "at_most": 0, "sleep_path": "urgent"}],
        "wakes": [{"model": "r", "at_least": 2}],
    }).json()

    assert body["status"] == "done" and len(body["wakes"][0]["woke"]) == 2
    assert counted[0] <= K8S_CALLS_PLAN_SELECT + K8S_CALLS_SLEEP + K8S_CALLS_WAKE + K8S_CALLS_REFRESH
    assert lists == [1]  # ONE observed refresh for the four bindings


def test_target_and_power_are_one_item_plans_in_one_lock_acquisition_each():
    world = World(pods=(("r-2", "r", (2,), "sleeping"), ("r-3", "r", (3,), "sleeping"), ("d-0", "d", (0,), "awake")))

    grown = world.client.put("/v2/models/r/target", json={"wake_replicas": 2})
    slept = world.client.put("/v2/bindings/d-0/power", json={"awake": False})

    assert grown.status_code == 200 and sorted(a["serve_id"] for a in grown.json()["actions"]) == ["r-2", "r-3"]
    assert slept.status_code == 200 and slept.json()["actions"] == [{"action": "sleep", "serve_id": "d-0"}]
    assert world.coordinator.kinds == ["put_model_target", "put_binding_power"]
    assert world.violations == [] and world.journals() == ({}, {})


def test_the_refuse_sleep_hook_fails_a_plan_sleep_before_its_hide():
    from dataclasses import replace

    from tre_common import rediskeys

    world = World(pods=MIXED)
    world.service._sm_config = replace(world.service._sm_config, test_hooks=True)
    world.service._fault_redis = world.redis
    world.redis.set(rediskeys.sm_fault_key("refuse_sleep", "node-a", 0), "1")

    body = post(world, MIXED_PLAN).json()

    assert body["sleeps"][0]["status"] == "failed"
    assert body["wakes"][1]["error"]["reason"] == "donor_sleep_failed"
    assert world.awake("d-0") and world.state("d-0") == "awake" and world.awake("x-1")
    assert not [p for p in world.runtime.patches if p[1] == "hidden"]

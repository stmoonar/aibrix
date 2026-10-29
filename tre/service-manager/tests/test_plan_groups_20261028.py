"""Plan groups and named multi-items (2026-10-28, design 20261027-sm-plans.md section
11): one transfer - its donor sleeps and its receiver wake - is one ``group``, atomic
in the selection; the controller names every pod, the service-manager validates and
executes. The invariants:

1. a refused sleep of a group refuses the whole group (nothing slept, nothing woken),
   items outside the group are unaffected;
2. a refused wake of a group refuses its donor sleeps;
3. named items of one model are cumulative against its floor; a count item and a named
   item of one model is a 400 ``mixed_items``;
4. a donor sleep that fails at execution keeps the receiver asleep
   (``donor_sleep_failed``, plan ``partial``), the donors that slept stay asleep;
5. the same plan sent again is a no-op;
6. a plan without a group never enters the group propagation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tre_common import rediskeys
from tre_sm.ops import plan as plan_module

from test_wholelock_20261002 import World

#: TP2 receiver t on GPUs 0-1 over a pod of d (GPU 0) and a pod of a (GPU 1); x sleeps
#: on the free GPU 2.
BLOCK = (
    ("d-0", "d", (0,), "awake"),
    ("d-3", "d", (3,), "awake"),
    ("a-1", "a", (1,), "awake"),
    ("a-3", "a", (3,), "sleeping"),
    ("t-01", "t", (0, 1), "sleeping"),
    ("x-2", "x", (2,), "sleeping"),
)


def _block_plan(group="g1", *, outside=True):
    plan = {
        "plan_id": "pl-groups",
        "sleeps": [
            {"model": "d", "pods": ["d-0"], "sleep_path": "urgent", "group": group},
            {"model": "a", "pods": ["a-1"], "sleep_path": "urgent", "group": group},
        ],
        "wakes": [{"model": "t", "pods": ["t-01"], "group": group}],
    }
    if outside:
        plan["wakes"].append({"model": "x", "pods": ["x-2"]})
    return plan


def _world(**kwargs):
    return World(pods=BLOCK, extra_models=("a",), **kwargs)


def _wake_calls(world, name):
    return [call for call in world.vllm.calls if call == ("wake_up", world.ip_of[name])]


def _sleep_calls(world, name):
    return [call for call in world.vllm.calls if call[0] == "sleep" and call[1] == world.ip_of[name]]


def test_a_block_group_is_one_plan_and_its_wake_waits_for_both_donors():
    world = _world()

    response = world.client.post("/v2/plans", json=_block_plan())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "done"
    t_item = body["wakes"][0]
    assert t_item["group"] == "g1"
    [woke] = t_item["woke"]
    assert woke["wave"] == 2 and sorted(woke["donor_binding_ids"]) == ["a/node-a/1", "d/node-a/0"]
    assert world.awake("t-01") and not world.awake("d-0") and not world.awake("a-1")
    assert world.violations == []


def test_a_refused_sleep_refuses_its_whole_group_and_nothing_else():
    world = _world(d_min=2)  # d-0 and d-3 serve: d-0 may not sleep

    body = world.client.post("/v2/plans", json=_block_plan()).json()

    d_item, a_item = body["sleeps"]
    t_item, x_item = body["wakes"]
    for item in (d_item, a_item, t_item):
        assert item["status"] == "refused" and item["error"]["error"] == "group_refused"
    assert "sleep item 0 (d) refused: floor_violation" in d_item["error"]["detail"]
    assert x_item["status"] == "done"
    assert world.awake("d-0") and world.awake("a-1") and not world.awake("t-01") and world.awake("x-2")
    assert _sleep_calls(world, "a-1") == [] and not [p for p in world.runtime.patches if p[1] == "hidden"]


def test_a_refused_wake_refuses_the_donor_sleeps_of_its_group():
    world = _world()
    world.service._sm_config = replace(world.service._sm_config, test_hooks=True)
    world.service._fault_redis = world.redis
    world.redis.set(rediskeys.sm_fault_key("refuse_wake", "node-a", 0), "1")  # t-01's GPU 0

    body = world.client.post("/v2/plans", json=_block_plan(outside=False)).json()

    assert [item["status"] for item in body["sleeps"] + body["wakes"]] == ["refused"] * 3
    assert all(item["error"]["error"] == "group_refused" for item in body["sleeps"] + body["wakes"])
    assert world.awake("d-0") and world.awake("a-1") and not world.awake("t-01")
    assert [c for c in world.vllm.calls if c[0] in ("sleep", "wake_up")] == []


def test_named_sleep_items_of_one_model_are_cumulative_against_its_floor():
    pods = (("d-0", "d", (0,), "awake"), ("d-1", "d", (1,), "awake"), ("d-2", "d", (2,), "awake"))
    world = World(pods=pods, d_min=2)
    plan = {"sleeps": [
        {"model": "d", "pods": ["d-0"], "sleep_path": "urgent", "group": "g1"},
        {"model": "d", "pods": ["d-1"], "sleep_path": "urgent", "group": "g2"},
    ]}

    body = world.client.post("/v2/plans", json=plan).json()

    first, second = body["sleeps"]
    assert first["status"] == "done" and second["status"] == "refused"
    assert second["error"]["error"] == "group_refused" and "floor_violation" in second["error"]["detail"]
    assert [world.awake(name) for name in ("d-0", "d-1", "d-2")] == [False, True, True]


@pytest.mark.parametrize("plan", [
    {"sleeps": [{"model": "d", "at_most": 1, "sleep_path": "urgent"},
                {"model": "d", "pods": ["d-0"], "sleep_path": "urgent"}]},
    {"sleeps": [{"model": "d", "pods": ["d-0"], "sleep_path": "urgent"}],
     "wakes": [{"model": "d", "at_least": 2}]},
])
def test_a_count_item_and_a_named_item_of_one_model_is_400_mixed_items(plan):
    world = _world()
    response = world.client.post("/v2/plans", json=plan)
    assert response.status_code == 400 and "mixed_items" in response.json()["detail"]
    assert world.vllm.calls == []


def test_one_pod_named_in_two_items_is_400():
    world = _world()
    plan = {"sleeps": [{"model": "d", "pods": ["d-0"], "sleep_path": "urgent"},
                       {"model": "d", "pods": ["d-0"], "sleep_path": "urgent"}]}
    assert world.client.post("/v2/plans", json=plan).status_code == 400


def test_a_failed_donor_sleep_keeps_the_receiver_asleep_and_the_slept_donor_asleep():
    world = _world()
    world.vllm.fail_sleep_for.add(world.ip_of["d-0"])

    response = world.client.post("/v2/plans", json=_block_plan(outside=False))

    assert response.status_code == 200, response.text  # something changed: a-1 slept
    body = response.json()
    assert body["status"] == "partial"
    t_item = body["wakes"][0]
    assert t_item["error"]["error"] == "donor_sleep_failed" and t_item["retriable"] is True
    assert _wake_calls(world, "t-01") == [] and not world.awake("a-1") and world.awake("d-0")
    assert world.violations == []

    # The next decision names the same block again: a-1 is a no-op, d-0 sleeps now.
    world.vllm.fail_sleep_for.clear()
    again = world.client.post("/v2/plans", json=_block_plan(outside=False)).json()
    assert [item["status"] for item in again["sleeps"]] == ["done", "noop"]
    assert again["wakes"][0]["status"] == "done" and world.awake("t-01")


def test_the_same_plan_sent_again_is_all_noop():
    world = _world()
    assert world.client.post("/v2/plans", json=_block_plan()).json()["status"] == "done"
    calls = len(world.vllm.calls)

    again = world.client.post("/v2/plans", json=_block_plan())

    assert again.status_code == 200
    body = again.json()
    assert body["status"] == "noop"
    assert {item["status"] for item in body["sleeps"] + body["wakes"]} == {"noop"}
    assert [c for c in world.vllm.calls[calls:] if c[0] in ("sleep", "wake_up")] == []


def test_a_plan_without_a_group_never_enters_the_group_propagation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("group propagation entered without a group")

    monkeypatch.setattr(plan_module, "_select_grouped", forbidden)
    world = _world()
    plan = {"sleeps": [{"model": "d", "at_most": 1, "sleep_path": "urgent"}],
            "wakes": [{"model": "x", "at_least": 1}]}
    assert world.client.post("/v2/plans", json=plan).status_code == 200
    named = {"sleeps": [{"model": "d", "pods": ["d-0"], "sleep_path": "urgent"}],
             "wakes": [{"model": "t", "pods": ["t-01"]}]}
    assert world.client.post("/v2/plans", json=named).status_code in (200, 409)


def test_a_group_wake_waits_for_every_sleep_of_its_group_not_only_those_on_its_gpus():
    world = _world()
    world.vllm.fail_sleep_for.add(world.ip_of["d-3"])
    plan = {"sleeps": [{"model": "d", "pods": ["d-3"], "sleep_path": "urgent", "group": "g"}],
            "wakes": [{"model": "x", "pods": ["x-2"], "group": "g"}]}  # x-2's GPU 2 is free

    body = world.client.post("/v2/plans", json=plan).json()

    assert body["wakes"][0]["error"]["error"] == "donor_sleep_failed"
    assert _wake_calls(world, "x-2") == []

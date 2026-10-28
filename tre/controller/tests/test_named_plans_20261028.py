"""One decider per choice (2026-10-28, design 20261027-sm-plans.md section 10): the
controller names every pod it sleeps or wakes - its plans carry no ``at_most`` and no
``donors``, ``at_least`` only where no binding exists to name (a cold create); each
transfer is one plan group; the queue keeps several named items of one model in one
plan and splits only on a pod named twice."""

from __future__ import annotations

import asyncio

import pytest

from tre_common.gpu_placement import GpuBlock
from tre_common.metrics_schema import MetricsSnapshot
from tre_controller.loops.action_queue import ActionQueue, PlanBatch, transfer_plan
from tre_controller.loops.plan_items import plan_item
from tre_controller.loops.tick import _apply_safescale, _name_shrink_pods
from tre_controller.planning.planner import BlockSupplyAction, ClusterView, ScaleAction
from tre_sm.allocator.slots import release_order

from plan_shim import PlanSM
from test_single_loop_20261028 import CRIT, HEALTHY, HIGH, IDLE, TOPOLOGY, _binding, _tp2



@pytest.fixture(autouse=True)
def _tick_names_shrinks(monkeypatch):
    """``_tp2`` runs the tick's SafeScale step; the tick then names its shrinks."""
    import test_single_loop_20261028 as single_loop

    def apply(snapshot, actions, *args, cluster_view=None, **kwargs):
        out, events = _apply_safescale(snapshot, actions, *args, cluster_view=cluster_view, **kwargs)
        named, more = _name_shrink_pods(snapshot, out, cluster_view)
        return named, tuple(events) + more

    monkeypatch.setattr(single_loop, "_apply_safescale", apply)


def _items(plans):
    return [item for items in plans for item in items if item[0] in ("sleep", "wake")]


SCENARIOS = {
    "tp2_block_of_idle_pods": dict(donors=(_binding("i-0", "i", (0,), True), _binding("i-1", "i", (1,), True))),
    "tp2_block_high_released": dict(
        donors=(_binding("h-1", "h", (1,), True), _binding("s7-2", "s7", (2,), True),
                _binding("s8-3", "s8", (3,), True)),
        classes={"h": (*HEALTHY, 1.2), "s7": (*HIGH, 1.4), "s8": (*HIGH, 1.5)}, scope="receiverless_shrink"),
    "block_and_relay_share_an_idle_donor": dict(
        donors=(_binding("i-0", "i", (0,), True), _binding("i-2", "i", (2,), True)),
        classes={"tp2": (*CRIT, 0.1), "c": (*CRIT, 0.3)}, scope="receiverless_shrink",
        receivers=(_binding("tp2-4", "tp2", (4, 5), True), _binding("tp2-0", "tp2", (0, 1), False),
                   _binding("tp2-2", "tp2", (2, 3), False), _binding("c-2", "c", (2,), False))),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_controller_plans_name_every_pod(name):
    case = dict(SCENARIOS[name])
    plans, _, _ = _tp2(*case.pop("donors"), **case)
    items = _items(plans)
    assert items, plans
    for kind, model, pods, hints, donors in items:
        assert donors is None and hints is None
        assert pods, (kind, model)


def test_the_block_and_the_relay_of_one_idle_donor_are_two_groups_of_one_plan():
    case = dict(SCENARIOS["block_and_relay_share_an_idle_donor"])
    plans, _, _ = _tp2(*case.pop("donors"), **case)
    sleeps = sorted(pod for kind, _, pods, _, _ in _items(plans) if kind == "sleep" for pod in pods)
    wakes = sorted(pod for kind, _, pods, _, _ in _items(plans) if kind == "wake" for pod in pods)
    assert sleeps == ["i-0", "i-2"] and wakes == ["c-2", "tp2-0"]

    sm = PlanSM({"i": 2, "tp2": 1, "c": 0})
    queue = ActionQueue(sm, now_ms=lambda: 5_000)
    [block] = [a for a in _blocks()]
    relay = BlockSupplyAction(receiver="c", block=GpuBlock("n", (2,)), receiver_pods=("c-2",),
                              sleeps=(("i", "i-2"),), reason="critical_donor_immediate", source_loop="rescue")
    queue.submit((block, relay))
    results = asyncio.run(queue.drain_once())

    [plan] = sm.plans  # one decision, one plan: named items of one model do not split it
    assert all(result.ok for result in results)
    groups = {}
    for item in plan["sleeps"] + plan["wakes"]:
        assert not {"at_most", "at_least", "donors"} & set(item)
        groups.setdefault(item["group"], []).append(item["pods"][0])
    assert sorted(map(sorted, groups.values())) == [["c-2", "i-2"], ["i-0", "tp2-0"]]


def _blocks():
    yield BlockSupplyAction(receiver="tp2", block=GpuBlock("n", (0, 1)), receiver_pods=("tp2-0",),
                            sleeps=(("i", "i-0"),), reason="critical_block_idle", source_loop="rescue")


def test_a_pod_named_twice_starts_the_next_plan():
    sm = PlanSM({"i": 2, "tp2": 1, "c": 0})
    queue = ActionQueue(sm, now_ms=lambda: 5_000)
    [block] = list(_blocks())
    clash = transfer_plan(BlockSupplyAction(receiver="c", block=GpuBlock("n", (0,)), receiver_pods=("c-0",),
                                            sleeps=(("i", "i-0"),), reason="critical_donor_immediate",
                                            source_loop="rescue"))
    queue.submit((block, clash))  # the rescue path accepts both; the plan may not name i-0 twice
    asyncio.run(queue.drain_once())
    first, second = sm.plans
    assert [item["pods"] for item in first["wakes"]] == [["tp2-0"]]
    assert [item["pods"] for item in second["wakes"]] == [["c-0"]]


def test_an_immediate_shrink_names_its_pods_by_the_shared_release_order():
    bindings = tuple(_binding(f"i-{gpu}", "i", (gpu,), True) for gpu in range(4))
    view = ClusterView(TOPOLOGY, bindings)
    snapshot = MetricsSnapshot(ts_ms=1, models={}, stale=False)
    shrink = ScaleAction("i", -2, "idle_proactive_immediate", "rescue", donor="i", sleep_path="urgent")
    taken = ScaleAction("i", -1, "critical_block_idle", "rescue", pods=("i-1",), sleep_path="urgent")

    actions, events = _name_shrink_pods(snapshot, (taken, shrink), view)

    order = [b.serve_id for b in release_order(list(bindings), bindings=list(bindings), topology=TOPOLOGY)]
    assert actions[1].pods == tuple(pod for pod in order if pod != "i-1")[:2] and actions[1].delta == -2
    assert plan_item(actions[1])[1]["pods"] == list(actions[1].pods) and events == ()

    # No fleet view and no metrics pods: nothing to name - dropped, with an event.
    dropped, events = _name_shrink_pods(snapshot, (shrink,), None)
    assert dropped == () and events == ("shrink_unnamed:i",)

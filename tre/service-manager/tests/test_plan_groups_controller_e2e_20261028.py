"""Review finding A-1 end to end (2026-10-28, design 20261027-sm-plans.md section 10):
the controller's planner, tick and ActionQueue against the service-manager's real
``POST /v2/plans`` (the whole-lock World). One IDLE donor gives a pod to a TP1 relay
and another pod to a TP2 block in ONE decision: the controller names every pod, each
transfer is one plan group, and the SM executes exactly those pods - both receivers
wake, in one plan."""

from __future__ import annotations

import asyncio

from tre_common.metrics_schema import MetricsSnapshot
from tre_controller.config import SafeScaleConfig
from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.cluster_view_task import cluster_view_from_state
from tre_controller.loops.tick import _apply_safescale, _name_shrink_pods
from tre_controller.planning.safescale import SafeScaleStateMachine
from tre_controller.sm_client import ServiceManagerClient, ServiceManagerError

from test_single_loop_20261028 import CRIT, HEALTHY, IDLE, _plan
from test_wholelock_20261002 import World

#: GPU 0: IDLE d-0, GPU 1: free, GPU 2: IDLE d-2, GPU 3: x-3 (at its floor). The TP2
#: receiver t sleeps on 0-1 and 2-3, the TP1 receiver r on GPUs 0 and 2.
PODS = (
    ("d-0", "d", (0,), "awake"),
    ("d-2", "d", (2,), "awake"),
    ("x-3", "x", (3,), "awake"),
    ("t-01", "t", (0, 1), "sleeping"),
    ("t-23", "t", (2, 3), "sleeping"),
    ("r-0", "r", (0,), "sleeping"),
    ("r-2", "r", (2,), "sleeping"),
)


class _WorldTransport:
    """``ServiceManagerClient`` transport onto the World's TestClient."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.posts: list[dict] = []

    async def request(self, method, url, *, json=None, timeout_s):
        path = "/" + url.split("://", 1)[-1].split("/", 1)[-1]
        if method == "POST":
            self.posts.append(json)
            response = self.world.client.post(path, json=json)
        else:
            response = self.world.client.get(path)
        body = response.json()
        if response.status_code >= 400:
            raise ServiceManagerError(f"HTTP {response.status_code}", status=response.status_code,
                                      body=body.get("detail", body) if isinstance(body, dict) else body)
        return body


def _decide(world: World):
    view = cluster_view_from_state(world.client.get("/v2/state").json(), world.registry.topology())
    plan = _plan(
        view.bindings,
        {"t": (*CRIT, 0.1), "r": (*CRIT, 0.3), "d": (*IDLE, 0.0), "x": (*HEALTHY, 1.1)},
        topology=view.topology,
        model_tp_sizes={"t": 2, "r": 1, "d": 1, "x": 1},
        min_replicas_by_model={"x": 1, "d": 0},
    )
    snapshot = MetricsSnapshot(ts_ms=1, models={}, stale=False)
    actions, _ = _apply_safescale(
        snapshot, tuple(plan.actions), plan.probe_upscale_plans,
        safescale=SafeScaleStateMachine(config=SafeScaleConfig()), cluster_view=view, safescale_scope="all",
    )
    actions, _ = _name_shrink_pods(snapshot, actions, view)
    return actions, plan.events


def test_one_idle_donor_feeds_a_tp1_relay_and_a_tp2_block_in_one_plan():
    world = World(pods=PODS)
    actions, events = _decide(world)
    transport = _WorldTransport(world)
    queue = ActionQueue(ServiceManagerClient("http://sm", transport=transport))

    queue.submit(actions)
    results = asyncio.run(queue.drain_once())

    [plan] = transport.posts
    groups = {item.get("group") for item in plan["sleeps"] + plan["wakes"]}
    assert len(groups) == 2 and None not in groups, plan
    assert not any({"at_most", "at_least", "donors"} & set(item) for item in plan["sleeps"] + plan["wakes"])
    assert all(result.ok for result in results), results
    assert world.awake("t-01") or world.awake("t-23")
    assert world.awake("r-0") or world.awake("r-2")
    assert not world.awake("d-0") and not world.awake("d-2") and world.awake("x-3")
    assert world.violations == []


def test_after_a_failed_donor_sleep_the_next_decision_wakes_the_receiver_on_the_same_block():
    world = World(pods=PODS)
    world.vllm.fail_sleep_for.add(world.ip_of["d-0"])
    transport = _WorldTransport(world)
    queue = ActionQueue(ServiceManagerClient("http://sm", transport=transport))

    queue.submit(_decide(world)[0])
    asyncio.run(queue.drain_once())
    [first] = transport.posts
    block = next(item for item in first["wakes"] if item["model"] == "t")
    assert not world.awake("t-01") and not world.awake("t-23") and world.awake("d-0")
    assert world.awake("r-2")  # the relay outside the block's group is unaffected

    # The donor recovers; the next decision, on the resulting state, names the same block.
    world.vllm.fail_sleep_for.clear()
    queue.submit(_decide(world)[0])
    asyncio.run(queue.drain_once())
    second = transport.posts[-1]
    again = next(item for item in second["wakes"] if item["model"] == "t")
    assert again["pods"] == block["pods"] == ["t-01"]
    assert world.awake("t-01") and not world.awake("d-0")
    assert world.violations == []

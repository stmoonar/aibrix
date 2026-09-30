"""S3 (2026-09-30): a structured 409 wake refusal names its node / GPUs; the
controller keeps that GPU (or, node scope, that node) out of wake planning for
registry placement.wake_cooldown, so the next tick picks another GPU instead of
re-planning the refused slot every tick. A plain-text 409 of an older SM stays
what it was (retriable, no cooldown)."""

from __future__ import annotations

import asyncio
import json

from tre_controller.loops.action_queue import ActionQueue
from tre_controller.planning.classify import ModelRole, ModelState
from tre_controller.planning.planner import ClusterView, ScaleAction, build_plan
from tre_controller.sm_client import ServiceManagerError, _request_json, sm_actor
from tre_sm.allocator.slots import Binding, Slot

from test_planner_slot_occupancy import TOPOLOGY, _cfg, _cls


STRUCTURED = {
    "detail": "7b-1: slot already has awake binding",
    "error": "gpu_busy",
    "reason": "slot_occupied",
    "binding_id": "dsllama-8b/node9/1",
    "node": "node9",
    "gpu_ids": [1],
    "scope": "gpu",
    "blocking_binding_id": "other/node9/1",
    "retry_after_s": 30.0,
}


# ------------------------------------------------------------------ sm_client


def test_structured_409_is_parsed_into_a_located_wake_conflict():
    error = ServiceManagerError("HTTP 409", status=409, body=STRUCTURED)

    conflict = error.wake_conflict
    result = error.result()

    assert (conflict["node"], conflict["gpu_ids"], conflict["scope"], conflict["error"]) == (
        "node9", [1], "gpu", "gpu_busy",
    )
    assert conflict["retry_after_s"] == 30.0
    assert result["retriable"] is True and result["wake_conflict"] == conflict


def test_structured_409_legacy_shape_single_gpu_and_generic_code():
    error = ServiceManagerError(
        "HTTP 409", status=409, body={"detail": "x", "error": "wake_conflict", "node": "n", "gpu": 3}
    )
    assert error.wake_conflict["gpu_ids"] == [3]


def test_plain_text_409_stays_retriable_without_a_cooldown():
    for body in (None, {"detail": "slot already has awake binding"}):
        error = ServiceManagerError("HTTP 409: slot already has awake binding", status=409, body=body)
        assert error.wake_conflict is None
        assert error.retriable is True
        assert "wake_conflict" not in error.result()


def test_sm_calls_carry_the_actor_of_the_action(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(request, timeout):
        seen["actor"] = request.get_header("X-tre-actor")
        return Response()

    monkeypatch.setattr("tre_controller.sm_client.urlopen", fake_urlopen)
    with sm_actor("controller/rescue/critical_sleeping_capacity"):
        _request_json("GET", "http://sm/v2/state", None, 1.0)
    assert seen["actor"] == "controller/rescue/critical_sleeping_capacity"
    _request_json("GET", "http://sm/v2/state", None, 1.0)
    assert seen["actor"] == "tre-controller"


# ------------------------------------------------------------------ queue


class RefusingClient:
    """set_binding_power answers like ServiceManagerClient for a structured 409."""

    def __init__(self, body):
        self.body = body
        self.calls = []

    async def set_binding_power(self, serve_id, *, awake, **_kwargs):
        self.calls.append(serve_id)
        return ServiceManagerError("HTTP 409", status=409, body=self.body).result()

    async def scale_model(self, model, delta, **_kwargs):
        return {"ok": True}


def _queue(client, clock):
    return ActionQueue(client, now_ms=lambda: clock["now"], wake_cooldown_s=(30.0, 60.0))


def _wake(pod="8b-1"):
    return ScaleAction("dsllama-8b", 1, "critical_sleeping_capacity", "rescue", receiver="dsllama-8b", pods=(pod,))


def test_gpu_cooldown_after_a_refused_wake_then_expiry():
    clock = {"now": 1_000_000}
    queue = _queue(RefusingClient(STRUCTURED), clock)
    queue.submit([_wake()])
    (result,) = asyncio.run(queue.drain_once())

    assert result.ok is False and result.wake_conflict["node"] == "node9"
    assert queue.cooled_gpus() == {("node9", 1)}
    assert queue.recent_refusals() == {"dsllama-8b": "node9/1"}
    events = queue.drain_events()
    assert "gpu_cooldown:node9/1:1030000" in events
    assert "wake_refused:dsllama-8b:node9/1:gpu_busy" in events
    assert queue.drain_events() == []

    clock["now"] += 30_000
    assert queue.cooled_gpus() == set()
    assert queue.recent_refusals() == {}


def test_gpu_cooldown_node_scope_for_missing_gpu_truth():
    clock = {"now": 5_000}
    body = dict(STRUCTURED, error="truth_unavailable", reason="gpu_truth_unavailable", scope="node")
    queue = _queue(RefusingClient(body), clock)
    queue.submit([_wake()])
    asyncio.run(queue.drain_once())

    assert queue.cooled_gpus() == set()
    assert queue.cooled_nodes() == {"node9"}
    assert "gpu_cooldown:node9/*:65000" in queue.drain_events()
    clock["now"] += 60_000
    assert queue.cooled_nodes() == set()


def test_gpu_cooldown_is_not_set_by_a_plain_409_or_a_sleep():
    clock = {"now": 0}
    queue = _queue(RefusingClient({"detail": "busy"}), clock)
    queue.submit([_wake()])
    asyncio.run(queue.drain_once())
    assert queue.cooled_gpus() == set()

    sleep_refused = _queue(RefusingClient(STRUCTURED), clock)
    sleep_refused.submit([ScaleAction("dsllama-8b", -1, "idle_proactive_immediate", "rescue", pods=("8b-1",))])
    asyncio.run(sleep_refused.drain_once())
    assert sleep_refused.cooled_gpus() == set()


# ------------------------------------------------------------------ planner


def _bindings():
    """7b awake everywhere except node9/1 and node10/2, where 8b can wake."""
    free = {("node9", 1), ("node10", 2)}
    bindings = []
    for index, (node, gpu) in enumerate((n, g) for n in ("node9", "node10") for g in range(4)):
        bindings.append(Binding(f"7b-{index}", "dsqwen-7b", Slot(node, (gpu,)), awake=(node, gpu) not in free))
        bindings.append(Binding(f"8b-{index}", "dsllama-8b", Slot(node, (gpu,)), awake=False))
    return tuple(bindings)


def _plan(**kwargs):
    return build_plan(
        model_contexts={
            "dsqwen-7b": {"routable_pods": 6, "assigned_replicas": 8},
            "dsllama-8b": {"routable_pods": 1, "assigned_replicas": 8},
        },
        classifications=[
            _cls("dsllama-8b", ModelState.CRITICAL, ModelRole.RECEIVER, 0.4),
            _cls("dsqwen-7b", ModelState.HEALTHY, ModelRole.NEUTRAL, 1.1),
        ],
        model_replicas={"dsqwen-7b": 8, "dsllama-8b": 8},
        idle_gpus=0,
        cfg=_cfg(),
        cluster_view=ClusterView(TOPOLOGY, _bindings()),
        **kwargs,
    )


def _wake_pods(plan):
    return [
        pod
        for action in plan.actions
        if isinstance(action, ScaleAction) and action.reason == "critical_sleeping_capacity"
        for pod in action.pods
    ]


def test_gpu_cooldown_makes_the_next_plan_pick_another_gpu():
    first = _wake_pods(_plan())
    assert len(first) == 1
    slots = {b.serve_id: b.slot for b in _bindings()}
    refused = slots[first[0]]
    refused_key = (refused.node, refused.gpu_ids[0])

    retry = _plan(
        unavailable_gpus={refused_key},
        refusals={"dsllama-8b": f"{refused.node}/{refused.gpu_ids[0]}"},
    )

    second = _wake_pods(retry)
    assert len(second) == 1 and second != first
    moved = slots[second[0]]
    assert (moved.node, moved.gpu_ids[0]) != refused_key
    assert f"placement_retry:dsllama-8b:{refused.node}/{refused.gpu_ids[0]}->{moved.node}/{moved.gpu_ids[0]}" in retry.events
    action = next(a for a in retry.actions if isinstance(a, ScaleAction) and a.reason == "critical_sleeping_capacity")
    assert action.hint is True  # pure-capacity wake: the SM may substitute (S5)


def test_gpu_cooldown_of_every_free_gpu_blocks_the_sleeping_capacity():
    plan = _plan(unavailable_gpus={("node9", 1), ("node10", 2)})
    assert _wake_pods(plan) == []
    assert "critical_sleeping_blocked:dsllama-8b" in plan.events


# ------------------------------------------------------------------ hinted dispatch


def test_placement_retry_event_when_the_sm_substitutes_a_hint():
    class HintedClient:
        async def scale_model_hinted(self, model, delta, *, hints):
            return {"ok": True, "response": {"picked": [
                {"serve_id": "8b-6", "binding_id": "dsllama-8b/node10/2", "node": "node10", "gpu_ids": [2], "hinted": False},
            ]}}

    slots = {b.serve_id: (b.slot.node, b.slot.gpu_ids) for b in _bindings()}
    queue = ActionQueue(HintedClient(), slot_of=slots.get)
    queue.submit([ScaleAction(
        "dsllama-8b", 1, "critical_sleeping_capacity", "rescue", receiver="dsllama-8b", pods=("8b-1",), hint=True,
    )])
    (result,) = asyncio.run(queue.drain_once())

    assert result.ok and result.picked[0]["node"] == "node10"
    assert queue.drain_events() == ["placement_retry:dsllama-8b:node9/1->node10/2"]

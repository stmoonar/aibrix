"""S3 (2026-09-30) wake refusals, rewritten for 2026-10-02 (design
20261002-controller-transfer): a structured 409 wake refusal still names its node /
GPUs (parsed by the client), but the controller no longer cools that GPU / node down
(the 30 / 60 s ``placement.wake_cooldown`` timers were removed): the refusal is an
observation event (``wake_refused``) and the next tick re-plans from a new view, in
which the service-manager reports the GPU's state itself (``/v2/state gpus[]``,
``blocked_gpus``). A plain-text 409 of an older SM stays retriable, no event."""

from __future__ import annotations

import asyncio

import pytest

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


def test_plain_text_409_stays_retriable_without_a_located_conflict():
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
    return ActionQueue(client, now_ms=lambda: clock["now"])


def _wake(pod="8b-1"):
    return ScaleAction("dsllama-8b", 1, "critical_sleeping_capacity", "rescue", receiver="dsllama-8b", pods=(pod,))


def test_a_refused_wake_is_an_event_only_and_starts_no_cooldown():
    clock = {"now": 1_000_000}
    queue = _queue(RefusingClient(STRUCTURED), clock)
    queue.submit([_wake()])
    (result,) = asyncio.run(queue.drain_once())

    assert result.ok is False and result.wake_conflict["node"] == "node9"
    assert queue.drain_events() == ["wake_refused:dsllama-8b:node9/1:gpu_busy"]
    assert queue.drain_events() == []
    # The same wake is accepted again at once (no timer; the next tick re-plans).
    assert queue.submit([_wake()]).accepted == 1


def test_a_node_scope_refusal_is_an_event_too():
    body = dict(STRUCTURED, error="truth_unavailable", reason="gpu_truth_unavailable", scope="node")
    queue = _queue(RefusingClient(body), {"now": 5_000})
    queue.submit([_wake()])
    asyncio.run(queue.drain_once())
    assert queue.drain_events() == ["wake_refused:dsllama-8b:node9/1:truth_unavailable"]


def test_no_wake_refused_event_for_a_plain_409_or_a_sleep():
    clock = {"now": 0}
    queue = _queue(RefusingClient({"detail": "busy"}), clock)
    queue.submit([_wake()])
    asyncio.run(queue.drain_once())
    assert queue.drain_events() == []

    sleep_refused = _queue(RefusingClient(STRUCTURED), clock)
    sleep_refused.submit([ScaleAction("dsllama-8b", -1, "idle_proactive_immediate", "rescue", pods=("8b-1",))])
    asyncio.run(sleep_refused.drain_once())
    assert sleep_refused.drain_events() == []


# ------------------------------------------------------------------ planner


def _bindings():
    """7b awake everywhere except node9/1 and node10/2, where 8b can wake."""
    free = {("node9", 1), ("node10", 2)}
    bindings = []
    for index, (node, gpu) in enumerate((n, g) for n in ("node9", "node10") for g in range(4)):
        bindings.append(Binding(f"7b-{index}", "dsqwen-7b", Slot(node, (gpu,)), awake=(node, gpu) not in free))
        bindings.append(Binding(f"8b-{index}", "dsllama-8b", Slot(node, (gpu,)), awake=False))
    return tuple(bindings)


def _plan(*, blocked=frozenset(), **kwargs):
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
        cluster_view=ClusterView(TOPOLOGY, _bindings(), blocked_gpus=frozenset(blocked)),
        **kwargs,
    )


def _wake_pods(plan):
    return [
        pod
        for action in plan.actions
        if isinstance(action, ScaleAction) and action.reason == "critical_sleeping_capacity"
        for pod in action.pods
    ]


def test_the_planner_has_no_cooldown_inputs_and_replans_from_the_view():
    first = _wake_pods(_plan())
    assert len(first) == 1
    # Same view, same plan: a refusal changes nothing by itself ...
    assert _wake_pods(_plan()) == first
    # ... the SM's next view does (it reports the GPU not wakeable: S5 blocked_gpus).
    slots = {b.serve_id: b.slot for b in _bindings()}
    refused = slots[first[0]]
    retry = _plan(blocked={(refused.node, refused.gpu_ids[0])})
    second = _wake_pods(retry)
    assert len(second) == 1 and second != first
    action = next(a for a in retry.actions if isinstance(a, ScaleAction) and a.reason == "critical_sleeping_capacity")
    assert action.hint is True  # pure-capacity wake: the SM may substitute (S5)
    with pytest.raises(TypeError):
        _plan(unavailable_gpus={("node9", 1)})


def test_every_free_gpu_blocked_in_the_view_blocks_the_sleeping_capacity():
    plan = _plan(blocked={("node9", 1), ("node10", 2)})
    assert _wake_pods(plan) == []
    assert "critical_sleeping_blocked:dsllama-8b" in plan.events


# ------------------------------------------------------------------ hinted dispatch


def test_placement_substituted_event_when_the_sm_substitutes_a_hint():
    class HintedClient:
        async def scale_model_hinted(self, model, delta, *, hints):
            return {"ok": True, "response": {"picked": [
                {"serve_id": "8b-6", "binding_id": "dsllama-8b/node10/2", "node": "node10", "gpu_ids": [2],
                 "hinted": False, "hint_binding_id": "dsllama-8b/node9/1"},
            ]}}

    queue = ActionQueue(HintedClient())
    queue.submit([ScaleAction(
        "dsllama-8b", 1, "critical_sleeping_capacity", "rescue", receiver="dsllama-8b", pods=("8b-1",), hint=True,
    )])
    (result,) = asyncio.run(queue.drain_once())

    assert result.ok and result.picked[0]["node"] == "node10"
    assert queue.drain_events() == ["placement_substituted:dsllama-8b:node9/1->node10/2"]


# ------------------------------------------------------------------ review fixes


def test_wake_failed_is_not_retried_at_once():
    body = dict(STRUCTURED, error="wake_failed", reason="vllm_wake_failed")
    error = ServiceManagerError("HTTP 409", status=409, body=body)
    assert error.retriable is False
    assert error.wake_conflict["error"] == "wake_failed"


def test_refusals_of_a_partial_hinted_wake_are_events_only():
    class PartialClient:
        async def scale_model_hinted(self, model, delta, *, hints):
            return {"ok": True, "response": {"picked": [], "unfilled": 1,
                                             "actions": [{"action": "wake", "serve_id": "8b-2"}], "refusals": [
                {"error": "gpu_busy", "reason": "gpu_truth_used", "node": "node9", "gpu_ids": [1], "scope": "gpu"},
            ]}}

    queue = ActionQueue(PartialClient(), now_ms=lambda: 3)
    queue.submit([ScaleAction("dsllama-8b", 2, "critical_sleeping_capacity", "rescue", pods=("8b-1",), hint=True)])
    (result,) = asyncio.run(queue.drain_once())

    assert result.ok is False and result.error.startswith("partial")  # not a silent success
    assert result.changed == ("8b-2",) and queue.last_actions() == {"dsllama-8b": (3, "up")}  # what woke counts
    assert queue.drain_events() == ["wake_refused:dsllama-8b:node9/1:gpu_busy"]


def test_hinted_wake_sends_no_avoid_gpus():
    seen = {}

    class Client:
        async def scale_model_hinted(self, model, delta, **kwargs):
            seen.update(kwargs)
            return {"ok": True, "response": {"picked": []}}

    queue = ActionQueue(Client())
    queue.submit([ScaleAction("dsllama-8b", 1, "critical_sleeping_capacity", "rescue", pods=("8b-1",), hint=True)])
    asyncio.run(queue.drain_once())
    assert seen == {"hints": ("8b-1",)}  # GPU use is serialized by the SM, not the queue


def test_a_409_partial_growth_is_reported_with_the_replicas_that_woke():
    """SM best effort (2026-10-06): an exact growth the SM filled in part answers 409
    ``partial`` with its response; the controller counts what woke (``changed`` /
    ``picked``, an "up" last action) and never re-sends the relative call."""
    from tre_controller.sm_client import ServiceManagerClient

    body = dict(STRUCTURED, error="partial", reason="partial", unfilled=1,
                actions=[{"action": "wake", "serve_id": "8b-2"}],
                picked=[{"serve_id": "8b-2", "node": "node9", "gpu_ids": [2], "hinted": False}],
                refusals=[STRUCTURED])

    class Transport:
        calls = 0

        async def request(self, method, url, *, json=None, timeout_s):
            if method == "GET":
                return {"models": {"dsllama-8b": {"awake": 1, "bound": 4}}}
            Transport.calls += 1
            raise ServiceManagerError("HTTP 409", status=409, body=body)

    queue = ActionQueue(ServiceManagerClient("http://sm", transport=Transport()), now_ms=lambda: 5)
    queue.submit([ScaleAction("dsllama-8b", 2, "critical_sleeping_capacity", "rescue")])
    (result,) = asyncio.run(queue.drain_once())

    assert result.ok is False and result.retriable is False and Transport.calls == 1
    assert result.changed == ("8b-2",) and len(result.picked) == 1
    assert queue.last_actions() == {"dsllama-8b": (5, "up")}
    assert queue.drain_events() == ["wake_refused:dsllama-8b:node9/1:gpu_busy"]  # its refusals

"""Review P2-1 (2026-10-06): a relay the SM answered with nothing done is not re-sent
for the same donor -> receiver pair while the fleet view it was planned from is still
current (a state gate keyed on the SM state version, no timer); an unknown outcome
(timeout) leaves no hold; no relay while the SM routable view is unknown.

2026-10-28 (sm-plans section 10): a relay is a named transfer - one plan group of the
donor pod's sleep and the receiver binding's wake; a refusal of the group refuses both."""
from __future__ import annotations

import asyncio

from tre_controller.loops.cluster_view_task import cluster_view_from_state
from tre_controller.planning.classify import ModelRole, ModelState
from tre_controller.planning.planner import ClusterView, PlanConfig, build_plan
from tre_controller.sm_client import ModelFloor, ServiceManagerError

from test_controller_transfer_20261002 import D, R, _e1
from test_planner_slot_occupancy import TOPOLOGY, _cls
from plan_shim import PlanSM
from relay_view import is_relay
from tre_controller.loops.action_queue import ActionQueue

FLOORS = {D: ModelFloor(routable=8, floor=1, floor_headroom=7), R: ModelFloor(routable=1, floor=1, floor_headroom=0)}


def _view(version, *, fetched_ms=1_000, floors=FLOORS, routable_error=None):
    bindings = _e1()
    return ClusterView(
        TOPOLOGY, bindings, fetched_ms=fetched_ms, state_ms=fetched_ms, sm_version=version,
        model_floors=dict(floors), routable_error=routable_error,
        routable_ids=None if routable_error else frozenset(b.serve_id for b in bindings if b.awake),
    )


def _plan(view, queue=None):
    return build_plan(
        model_contexts={
            D: {"routable_pods": 8, "assigned_replicas": 8},
            R: {"routable_pods": 1, "assigned_replicas": 8},
        },
        classifications=[
            _cls(R, ModelState.CRITICAL, ModelRole.RECEIVER, 0.4),
            _cls(D, ModelState.IDLE, ModelRole.DONOR, 10.0, "idle"),
        ],
        model_replicas={D: 8, R: 8},
        idle_gpus=0,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=8, suppress_hot_proactive_probe=True,
                       scale_step_ratio=1.0),
        cluster_view=view,
        relay_holds=queue.relay_holds() if queue is not None else None,
    )


def _relays(plan):
    return [action for action in plan.actions if is_relay(action)]


def _run(queue, plan):
    queue.submit(tuple(_relays(plan)))
    return asyncio.run(queue.drain_once())


def _refused(code, *, kind="sleep", reason=None, retriable=False):
    """The SM's answer to a transfer group one item of which was refused ``code``."""
    def respond(request):
        model = request[f"{kind}s"][0]["model"]
        detail = f"group g: {kind} item 0 ({model}) refused: {code}: {reason or code}"
        item = {"status": "refused", "error": {"error": "group_refused", "detail": detail}, "retriable": retriable,
                "woke": [], "unfilled": 0, "refusals": []}
        body = {"plan_id": "pl", "status": "refused",
                "sleeps": [dict(item, index=i, model=s["model"]) for i, s in enumerate(request["sleeps"])],
                "wakes": [dict(item, index=i, model=w["model"]) for i, w in enumerate(request["wakes"])]}
        return {"ok": False, "partial": True, "status": 409, "response": body}
    return respond


def _sm(respond=None):
    return PlanSM({D: 8, R: 1}, respond=respond)


def test_a_refused_relay_is_held_until_the_sm_state_version_changes():
    queue = ActionQueue(_sm(_refused("floor_violation")))
    first = _plan(_view(5))
    assert len(_relays(first)) == 1
    _run(queue, first)
    # Same SM version (a newer fetch of the same state): the pair is held, with why.
    again = _plan(_view(5, fetched_ms=2_000), queue)
    assert _relays(again) == []
    assert f"relay_held:{D}:{R}:floor_violation" in again.events
    # The SM state moved on: planned again.
    assert len(_relays(_plan(_view(6, fetched_ms=3_000), queue))) == 1
    # A changed floor view of the donor (same version) also releases it.
    moved = dict(FLOORS, **{D: ModelFloor(routable=7, floor=1, floor_headroom=6)})
    assert len(_relays(_plan(_view(5, fetched_ms=3_000, floors=moved), queue))) == 1


def test_a_relay_with_a_pair_done_leaves_no_hold():
    queue = ActionQueue(_sm())
    _run(queue, _plan(_view(5)))
    assert queue.relay_holds() == {}


def test_writer_busy_is_released_by_any_newer_view():
    busy = ServiceManagerError("HTTP 409", status=409, body={"error": "writer_busy", "detail": "x"}).result()
    queue = ActionQueue(_sm(lambda request: dict(busy, retriable=False)))
    _run(queue, _plan(_view(5)))
    held = _plan(_view(5), queue)
    assert _relays(held) == [] and f"relay_held:{D}:{R}:" in " ".join(held.events)
    assert len(_relays(_plan(_view(5, fetched_ms=2_000), queue))) == 1


def test_an_unknown_relay_outcome_is_neither_accounted_as_done_nor_held():
    timeout = ServiceManagerError("request timed out", timeout=True).result()
    queue = ActionQueue(_sm(lambda request: dict(timeout, retriable=False)), now_ms=lambda: 9_000)
    plan = _plan(_view(5))
    donor, receiver = _run(queue, plan)
    assert (receiver.ok, donor.ok) == (False, False)
    assert queue.last_actions() == {}  # nothing accounted as done
    record = queue.rescue_targets().get(R)
    assert record is None or record.gained == 0
    assert queue.relay_holds() == {}
    assert len(_relays(_plan(_view(5), queue))) == 1  # not held: re-planned on the same view


def test_no_relay_while_the_sm_routable_view_is_unknown():
    plan = _plan(_view(5, routable_error="routable_unavailable: pod list failed"))
    assert _relays(plan) == []
    assert f"relay_skipped_routable_unknown:{D}:{R}" in plan.events


def test_the_view_carries_the_sm_state_version():
    view = cluster_view_from_state({"version": 42, "bindings": [], "models": {}}, TOPOLOGY)
    assert view.sm_version == 42
    assert cluster_view_from_state({"bindings": []}, TOPOLOGY).sm_version is None


def test_a_503_during_sm_shutdown_does_not_hold_the_relay():
    """Review 2026-10-06 P2-1: a code-less refusal (503 while the SM shuts down) says
    nothing about the fleet: the next tick (a newer view, same SM version and floors)
    sends the relay to the CRITICAL receiver again."""
    down = ServiceManagerError("HTTP 503", status=503).result()
    queue = ActionQueue(_sm(lambda request: dict(down, retriable=False)))
    _run(queue, _plan(_view(5)))
    assert len(_relays(_plan(_view(5, fetched_ms=2_000), queue))) == 1


def test_a_transient_wake_refusal_does_not_hold_the_relay_a_fleet_one_does():
    queue = ActionQueue(_sm(_refused("resident_loading", kind="wake", reason="lease_starting")))
    _run(queue, _plan(_view(5)))
    assert len(_relays(_plan(_view(5, fetched_ms=2_000), queue))) == 1
    # The receiver GPU holds another awake binding in the SM's books: held.
    queue = ActionQueue(_sm(_refused("gpu_busy", kind="wake", reason="slot_occupied")))
    _run(queue, _plan(_view(5)))
    assert _relays(_plan(_view(5, fetched_ms=2_000), queue)) == []


def test_the_relay_is_one_group_of_named_items():
    sm = _sm()
    queue = ActionQueue(sm)
    [relay] = _relays(_plan(_view(5)))
    _run(queue, _plan(_view(5)))
    [plan] = sm.plans
    [sleep], [wake] = plan["sleeps"], plan["wakes"]
    assert sleep["pods"] == [relay.sleeps[0][1]] and wake["pods"] == list(relay.receiver_pods)
    assert sleep["group"] == wake["group"]
    assert not {"at_most", "at_least", "donors"} & (set(sleep) | set(wake))

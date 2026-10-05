"""Review P2-1 (2026-10-06): a relay the SM answered with nothing done is not re-sent
for the same donor -> receiver pair while the fleet view it was planned from is still
current (a state gate keyed on the SM state version, no timer); an unknown outcome
(timeout) leaves no hold; no relay while the SM routable view is unknown."""
from __future__ import annotations

import asyncio

from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.cluster_view_task import cluster_view_from_state
from tre_controller.planning.classify import ModelRole, ModelState
from tre_controller.planning.planner import ClusterView, PlanConfig, TransferIntent, build_plan
from tre_controller.sm_client import ModelFloor, ServiceManagerError

from test_action_queue_review3 import ScriptedSM, transfer_body
from test_controller_transfer_20261002 import D, R, _e1
from test_planner_slot_occupancy import TOPOLOGY, _cls

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
                       rescue_max_step_ratio=2.0, scale_step_ratio=1.0),
        cluster_view=view,
        relay_holds=queue.relay_holds() if queue is not None else None,
    )


def _relays(plan):
    return [action for action in plan.actions if isinstance(action, TransferIntent)]


def _run(queue, plan):
    queue.submit(tuple(_relays(plan)))
    return asyncio.run(queue.drain_once())


def _sm(*answers):
    return ScriptedSM(results={f"transfer:{D}->{R}": list(answers)})


def test_a_refused_relay_is_held_until_the_sm_state_version_changes():
    clamped = {"ok": True, "response": transfer_body(0, clamped=True, unfilled=1, skipped={"donor_floor": 1})}
    queue = ActionQueue(_sm(clamped, clamped))
    first = _plan(_view(5))
    assert len(_relays(first)) == 1
    _run(queue, first)
    # Same SM version (a newer fetch of the same state): the pair is held, with why.
    again = _plan(_view(5, fetched_ms=2_000), queue)
    assert _relays(again) == []
    assert f"relay_held:{D}:{R}:clamped_by_floor" in again.events
    # The SM state moved on: planned again.
    assert len(_relays(_plan(_view(6, fetched_ms=3_000), queue))) == 1
    # A changed floor view of the donor (same version) also releases it.
    moved = dict(FLOORS, **{D: ModelFloor(routable=7, floor=1, floor_headroom=6)})
    assert len(_relays(_plan(_view(5, fetched_ms=3_000, floors=moved), queue))) == 1


def test_a_relay_with_a_pair_done_leaves_no_hold():
    queue = ActionQueue(_sm({"ok": True, "response": transfer_body(1)}))
    _run(queue, _plan(_view(5)))
    assert queue.relay_holds() == {}


def test_writer_busy_is_released_by_any_newer_view():
    busy = ServiceManagerError("HTTP 409", status=409, body={"error": "writer_busy", "detail": "x"}).result()
    queue = ActionQueue(_sm(dict(busy, retriable=False)))
    _run(queue, _plan(_view(5)))
    held = _plan(_view(5), queue)
    assert _relays(held) == [] and f"relay_held:{D}:{R}:writer_busy" in held.events
    assert len(_relays(_plan(_view(5, fetched_ms=2_000), queue))) == 1


def test_an_unknown_relay_outcome_is_neither_accounted_as_done_nor_held():
    timeout = ServiceManagerError("request timed out", timeout=True).result()
    queue = ActionQueue(_sm(dict(timeout, retriable=False)), now_ms=lambda: 9_000)
    plan = _plan(_view(5))
    donor, receiver = _run(queue, plan)
    assert (receiver.ok, receiver.done, donor.taken) == (False, None, None)
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

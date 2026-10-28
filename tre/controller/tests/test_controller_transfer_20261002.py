"""Controller side of the SM relay primitive (2026-10-02, docs/design/
20261002-controller-transfer.md).

* planner: an immediate donor -> receiver relay is a named transfer (2026-10-28,
  sm-plans section 10: the donor pod and the receiver binding on its GPU); free capacity
  is used first; the pairable receiver bindings bound it; a donor gives at most its SM
  ``floor_headroom``; the IDLE proactive shrink is a model-level shrink the tick names;
* ActionQueue: a transfer is one plan group (named sleeps + the named wake) - the SM
  completes or fails it under its writer lock before answering; ``writer_busy`` /
  ``routable_unknown`` count as not executed; a count shrink is accounted by ``taken``.
  The queue tests drive the old per-endpoint fakes through tests/plan_shim.py;
* SM view: ``routable`` / ``floor_headroom`` straight from ``/v2/state`` with a
  fallback (and an event) when the SM does not report them.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace

from tre_common.metrics_schema import MetricsSnapshot
from relay_view import relays
from tre_common.gpu_placement import GpuBlock
from tre_controller.loops.action_queue import PlanBatch, transfer_plan
from tre_controller.planning.planner import BlockSupplyAction
from tre_controller.loops.cluster_view_task import cluster_view_from_state
from tre_controller.loops.decision_snapshot import build_decision_snapshot
from tre_controller.loops.replay import TickReplayStep, run_tick_replay
from tre_controller.loops.tick import _model_contexts
from tre_controller.planning.classify import ModelRole, ModelState
from tre_controller.planning.planner import (
    ClusterView,
    PlanConfig,
    RescuePlan,
    ScaleAction,
    _SlotOccupancy,
    build_plan,
)
from tre_controller.sm_client import ModelFloor, ServiceManagerClient, ServiceManagerError, parse_state_routable
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.api.v2 import ServiceManagerV2
from tre_sm.state.store import StateStore

from test_action_queue_review3 import ScriptedSM, _calls
from test_tss_only_20261029 import _registry as _o1_registry
from test_tss_only_20261029 import _snap, _window
from test_planner_slot_occupancy import TOPOLOGY, _cls
from test_safescale_binding_commit import FakeRedis, InProcessServiceManager
from plan_shim import LegacyFakeQueue as ActionQueue, PlanShim  # pre-plan fakes behind POST /v2/plans

R, D = "dsllama-8b", "dsqwen-7b"
GPUS = [(node, gpu) for node in ("node9", "node10") for gpu in range(4)]


def _e1(*, free=(), no_receiver=()) -> tuple[Binding, ...]:
    """D awake on every GPU but ``free``; R sleeping on every GPU but ``no_receiver``."""
    bindings = []
    for index, (node, gpu) in enumerate(GPUS):
        if (node, gpu) not in free:
            bindings.append(Binding(f"7b-{index}", D, Slot(node, (gpu,)), awake=True))
        if (node, gpu) not in no_receiver:
            bindings.append(Binding(f"8b-{index}", R, Slot(node, (gpu,)), awake=False))
    return tuple(bindings)


def _plan(bindings, *, receiver_state=ModelState.CRITICAL, donor_state=ModelState.IDLE, need=1,
          donor_ctx=None, cfg=None):
    role = ModelRole.DONOR if donor_state in (ModelState.IDLE, ModelState.HIGH) else ModelRole.NEUTRAL
    donors = sum(1 for b in bindings if b.model == D and b.awake)
    return build_plan(
        model_contexts={
            D: {"routable_pods": donors, "assigned_replicas": 8, **(donor_ctx or {})},
            R: {"routable_pods": need, "assigned_replicas": 8},
        },
        classifications=[
            _cls(R, receiver_state, ModelRole.RECEIVER, 0.4 if receiver_state == ModelState.CRITICAL else 0.9),
            _cls(D, donor_state, role, 10.0 if donor_state == ModelState.IDLE else 2.0,
                 "idle" if donor_state == ModelState.IDLE else "surplus"),
        ],
        model_replicas={D: 8, R: 8},
        idle_gpus=0,
        # Legacy step rescue with ratio 1.0: the receiver needs exactly ``need`` replicas.
        cfg=cfg or PlanConfig(
            min_replicas_per_model=1, max_replicas_per_model=8, suppress_hot_proactive_probe=True,
            scale_step_ratio=1.0,
        ),
        cluster_view=ClusterView(TOPOLOGY, bindings),
    )


def _intents(plan):
    """The plan's relays (named transfers, summed per donor / receiver)."""
    return relays(plan.actions)


# ===================================================================== planner


def test_crit_on_full_gpus_yields_one_named_transfer():
    # 2026-10-28 (sm-plans section 10): the controller names the donor pod and the
    # receiver binding sleeping on its GPU.
    plan = _plan(_e1())
    [intent] = _intents(plan)
    assert (intent.donor_model, intent.receiver_model, intent.count, intent.pairs) == (D, R, 1, 1)
    assert (intent.reason, intent.source_loop) == ("critical_donor_immediate", "rescue")
    [donor_pod], [receiver_pod] = intent.donor_pods, intent.receiver_pods
    assert donor_pod.replace("7b", "8b") == receiver_pod
    assert intent.rescue is not None and intent.rescue.target == intent.rescue.covered + 1


def test_a_free_gpu_is_used_before_a_relay_capacity_order_unchanged():
    # need 2: the free GPU node10/3 (R sleeps there) first, then one relay.
    plan = _plan(_e1(free={("node10", 3)}), need=2)
    kinds = [(type(a).__name__, getattr(a, "reason", None)) for a in plan.actions]
    assert kinds == [("ScaleAction", "critical_sleeping_capacity"), ("BlockSupplyAction", "critical_donor_immediate")]
    wake, _ = plan.actions
    assert (wake.model, wake.delta, wake.pods) == (R, 1, ("8b-7",))
    [relay] = relays(plan.actions)
    assert (relay.count, relay.pairs) == (1, 1)


def test_pairable_receivers_bound_the_relay_size():
    # need 3, donor surplus 7, but R sleeps only under two 7b replicas.
    only_two = {gpu for gpu in GPUS if gpu not in {("node9", 0), ("node9", 1)}}
    plan = _plan(_e1(no_receiver=only_two), need=3)
    [intent] = _intents(plan)
    assert (intent.count, intent.pairs) == (2, 2)
    # Nothing pairable: no intent at all, an event says why.
    none = _plan(_e1(no_receiver=set(GPUS)), need=3)
    assert _intents(none) == [] and f"donor_no_slot_match:{D}:{R}" in none.events


def test_sm_floor_headroom_bounds_and_excludes_donors():
    bounded = _plan(_e1(), need=3, donor_ctx={"floor_headroom": 1, "floor": 7})
    assert [(i.count, i.pairs) for i in _intents(bounded)] == [(1, 1)]
    excluded = _plan(_e1(), need=3, donor_ctx={"floor_headroom": 0, "floor": 8})
    assert _intents(excluded) == [] and not any(e.startswith("donor_no_slot_match") for e in excluded.events)


def test_idle_proactive_shrink_is_a_model_level_target_call_bounded_by_headroom():
    def plan(ctx):
        return build_plan(
            model_contexts={"idle": {"routable_pods": 4, "assigned_replicas": 4, **ctx}},
            classifications=[_cls("idle", ModelState.IDLE, ModelRole.DONOR, 10.0, "idle")],
            model_replicas={"idle": 4},
            idle_gpus=0,
            cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4),
        )

    [shrink] = plan({}).actions
    assert (shrink.model, shrink.delta, shrink.reason, shrink.pods, shrink.sleep_path) == (
        "idle", -3, "idle_proactive_immediate", (), "urgent"
    )  # Q3: the whole surplus down to the floor (1) in one decision
    assert plan({"floor_headroom": 0}).actions == []


def test_tp2_receiver_pairs_with_two_single_gpu_donors_and_skips_an_uncovered_gpu():
    bindings = (
        Binding("d-0", D, Slot("node9", (0,)), awake=True),
        Binding("d-1", D, Slot("node9", (1,)), awake=True),
        Binding("d-2", D, Slot("node9", (2,)), awake=True),
        Binding("r2-a", R, Slot("node9", (0, 1)), awake=False),
        Binding("r2-b", R, Slot("node9", (2, 3)), awake=False),  # node9/3 has no donor: uncovered
    )
    occupancy = _SlotOccupancy(ClusterView(TOPOLOGY, bindings))
    none = occupancy.pairable(D, R, max_pairs=1, max_donors=1)  # one pair needs two donors
    assert (none.pairs, none.donors) == (0, 0)
    estimate = occupancy.pairable(D, R, max_pairs=5, max_donors=5)  # r2-b is not a relay
    assert (estimate.pairs, estimate.donors) == (1, 2)
    assert [(binding.serve_id, pods) for binding, pods in estimate.named] == [("r2-a", ("d-0", "d-1"))]
    assert occupancy.released_donor_ids() == {"d-0", "d-1"}


# ===================================================================== ActionQueue


def _intent(*, reason="critical_donor_immediate"):
    """A named relay pair 7b-1 -> 8b-1 as the tick sends it (one plan group)."""
    return transfer_plan(BlockSupplyAction(
        receiver="8b", block=GpuBlock("n", (1,)), receiver_pods=("8b-1",), sleeps=(("7b", "7b-1"),),
        reason=reason, source_loop="rescue",
    ))


def test_writer_busy_and_routable_unknown_are_accounted_as_not_executed():
    routable_unknown = ServiceManagerError(
        "HTTP 409", status=409, body={"error": "routable_unknown", "reason": "routable_unknown", "detail": "x"}
    ).result()
    # Plan B: requests queue on the SM writer lock; a lock wait that timed out is
    # 409 writer_busy - nothing done, retriable (the next tick re-plans).
    writer_busy = ServiceManagerError(
        "HTTP 409", status=409, body={"error": "writer_busy", "detail": "writer lock wait timed out"}
    ).result()
    assert routable_unknown["not_executed"] and writer_busy["not_executed"] and writer_busy["retriable"]
    # A 409 without an error code is not known to have changed nothing (outcome unknown).
    assert not ServiceManagerError("HTTP 409", status=409, body={"detail": "a wake of it is in progress; retry"}).not_executed
    sm = ScriptedSM(results={
        "7b-1": [dict(routable_unknown, retriable=False), dict(writer_busy, retriable=False)],
        "scale:8b": [writer_busy],
        "scale:7b": [routable_unknown],
    })
    sm.awake_counts = {"7b": 4}
    queue = ActionQueue(sm)
    queue.submit((_intent(),))
    queue.submit((ScaleAction("9b", 1, "critical_idle_capacity", "rescue"),))
    asyncio.run(queue.drain_once())
    queue.submit((ScaleAction("8b", 1, "critical_idle_capacity", "rescue"),))
    queue.submit((ScaleAction("7b", -1, "idle_proactive_immediate", "rescue", sleep_path="urgent"),))
    results = asyncio.run(queue.drain_once())
    assert all(result.not_executed for result in results)
    # Nothing was executed: only the unrelated 9b scale-up is stamped; nothing cools down.
    assert set(queue.view_changes()) == {"9b"} and set(queue.last_actions()) == {"9b"}
    assert queue.submit((_intent(),)).accepted == 1
    results = asyncio.run(queue.drain_once())  # the relay again: writer_busy this time
    assert all(result.not_executed for result in results) and set(queue.view_changes()) == {"9b"}


def test_target_scale_down_is_accounted_by_taken():
    def shrink(taken, clamped):
        return {"ok": True, "response": {"model": "7b", "actions": [], "taken": taken, "clamped_by_floor": clamped,
                                         "floor": {"clamped": clamped}}}

    sm = ScriptedSM(results={"scale:7b": [shrink(1, True), shrink(0, True)]})
    sm.awake_counts = {"7b": 4}
    queue = ActionQueue(sm, now_ms=lambda: 7_000)
    down = ScaleAction("7b", -2, "idle_proactive_immediate", "rescue", sleep_path="urgent")
    queue.submit((down,))
    [first] = asyncio.run(queue.drain_once())
    assert (first.ok, first.taken) == (True, 1)
    assert queue.last_actions() == {"7b": (7_000, "down")} and queue.view_changes() == {"7b": (7_000, "down")}
    assert queue.drain_events() == ["scale_clamped_by_floor:7b:taken=1:asked=2"]

    other = ScriptedSM(results={"scale:7b": [shrink(0, True)]})
    other.awake_counts = {"7b": 4}
    fresh = ActionQueue(other, now_ms=lambda: 8_000)
    fresh.submit((down,))
    [second] = asyncio.run(fresh.drain_once())
    assert (second.ok, second.taken) == (True, 0)
    assert fresh.last_actions() == {} and fresh.view_changes() == {}
    assert fresh.drain_events() == ["scale_clamped_by_floor:7b:taken=0:asked=2"]


def test_observe_mode_skips_a_transfer_before_its_single_call():
    sm = ScriptedSM()
    queue = ActionQueue(sm, is_observe=lambda: True)
    queue.submit((_intent(),))
    results = asyncio.run(queue.drain_once())
    assert _calls(sm) == [] and [(r.model, r.error) for r in results] == [("7b", "observe_skipped"), ("8b", "observe_skipped")]


# ===================================================================== sm_client / SM view


def _state(routable_flags, *, models=None, error=None):
    bindings = [
        {"serve_id": f"m-{i}", "model": "m", "node": "node9", "gpu_ids": [i], "awake": True, "hidden": False,
         **({} if flag == "missing" else {"routable": flag})}
        for i, flag in enumerate(routable_flags)
    ]
    state = {"bindings": bindings, "models": models or {}, "floor_enforced": True}
    if error:
        state["routable_error"] = error
    return state


def test_state_routable_and_floor_fields_are_parsed_with_a_fallback():
    models = {"m": {"awake": 2, "bound": 2, "routable": 1, "floor": 1, "floor_headroom": 0}}
    view = cluster_view_from_state(_state([True, False], models=models), TOPOLOGY)
    assert view.routable_ids == frozenset({"m-0"}) and view.routable_error is None
    assert view.model_floors["m"] == ModelFloor(routable=1, floor=1, floor_headroom=0)
    assert view.floor_enforced is True
    missing = parse_state_routable(_state([True, "missing"]))
    assert (missing.routable_ids, missing.error) == (None, "routable_missing")
    null = parse_state_routable(_state([None, None], error="pod list failed"))
    assert null.routable_ids is None and null.error == "routable_unavailable: pod list failed"


def test_tick_contexts_take_routable_and_headroom_from_the_sm():
    snapshot = _snap(_window(2_000_000, [(30.0, 0.5, 0.0, 10.0)] * 3, routable=2))
    bindings = tuple(Binding(f"m-{i}", "m", Slot("node-a", (i,)), awake=True, hidden=False) for i in range(2))
    view = ClusterView(topology=_o1_registry().topology(), bindings=bindings)
    sm_view = replace(view, routable_ids=frozenset({"m-0"}),
                      model_floors={"m": ModelFloor(routable=1, floor=1, floor_headroom=0)})
    contexts, events = _model_contexts(snapshot, _o1_registry(), cluster_view=sm_view)
    # m-1 is awake and not hidden in the store but its sleep is in progress: not routable.
    assert (contexts["m"]["routable_pods"], contexts["m"]["floor_headroom"], contexts["m"]["floor"]) == (1, 0, 1)
    assert not any(event.startswith("sm_routable_fallback") for event in events)

    fallback = replace(view, routable_error="routable_missing")
    contexts, events = _model_contexts(snapshot, _o1_registry(), cluster_view=fallback)
    assert contexts["m"]["routable_pods"] == 2 and "floor_headroom" not in contexts["m"]
    assert "sm_routable_fallback:routable_missing" in events


# ===================================================================== offline replay


def test_offline_replay_plans_a_named_transfer_the_sm_executes():
    from test_scaling_e2e import CRITICAL, IDLE_LOAD, REG_7B_8B, _registry, _window as _e2e_window

    registry = _registry(*REG_7B_8B)
    bindings = list(_e1())
    store = StateStore(FakeRedis())
    store.save(bindings, expected_version=0)
    service = ServiceManagerV2(registry, store)
    view = cluster_view_from_state(service.get_state(), registry.topology())
    snapshot = MetricsSnapshot(ts_ms=60_000, stale=False, models={
        R: _e2e_window(R, per_q=CRITICAL, running=1.0),
        D: _e2e_window(D, per_q=IDLE_LOAD[0], running=IDLE_LOAD[1]),  # F1-B: an IDLE donor relays
    })
    replay = run_tick_replay(
        [TickReplayStep(snapshot, rescue_due=True, fairness_due=False, cluster_view=view)],
        registry=registry, suppress_hot_proactive_probe=True,
    )
    [transfer] = [a for a in replay.actions if isinstance(a, PlanBatch)]
    sleep, wake = transfer.members
    assert (sleep.model, sleep.delta, wake.model, wake.delta) == (D, -1, R, 1)
    assert sleep.group == wake.group is not None
    donor_pod, receiver_pod = sleep.pods[0], wake.pods[0]
    assert donor_pod.replace("7b", "8b") == receiver_pod  # on the same GPU
    decision = build_decision_snapshot("rescue", snapshot, replay.results[0])
    payloads = [a for a in json.loads(decision["actions"]) if a["kind"] == "scale"]
    assert [(a["model"], a["delta"], a.get("pods")) for a in payloads] == [
        (D, -1, [donor_pod]), (R, 1, [receiver_pod])
    ]

    queue = ActionQueue(PlanShim(InProcessServiceManager(service), state=service.get_state))
    queue.submit((transfer,))
    results = asyncio.run(queue.drain_once())
    assert all(result.ok for result in results)
    awake = {b.serve_id for b in store.load().bindings if b.awake}
    assert receiver_pod in awake and donor_pod not in awake


def test_signal_log_records_both_sides_of_a_relay():
    from tre_controller.loops.signal_log import _summarize_actions

    from tre_common.gpu_placement import GpuBlock
    from tre_controller.loops.action_queue import transfer_plan
    from tre_controller.planning.planner import BlockSupplyAction

    transfer = BlockSupplyAction(
        receiver="8b", block=GpuBlock("n", (0,)), receiver_pods=("8b-0",), sleeps=(("7b", "7b-0"),),
        reason="critical_donor_immediate", source_loop="rescue",
    )
    labels, deltas = _summarize_actions((transfer_plan(transfer),))
    assert labels == {"7b": "transfer:7b->8b", "8b": "transfer:7b->8b"}
    assert deltas == {"7b": -1, "8b": 1}

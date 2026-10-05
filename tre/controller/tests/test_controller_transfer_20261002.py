"""Controller side of the SM relay primitive (2026-10-02, docs/design/
20261002-controller-transfer.md).

* planner: an immediate donor -> receiver relay is a TransferIntent (a count, no pod /
  GPU); free capacity is used first; ``pairable_count`` bounds the intent; a donor
  gives at most its SM ``floor_headroom``; the IDLE proactive shrink is a model-level
  ``/target`` call;
* ActionQueue: one ``POST /v2/transfers`` per intent (the SM completes or fails it
  under its global writer lock before answering), accounted by done / taken /
  unfilled (also when partial); refusals, floor clamps and ``taken: 0`` are events
  only; ``writer_busy`` / ``routable_unknown`` count as not executed; a 404 degrades
  to "not executed";
* SM view: ``routable`` / ``floor_headroom`` straight from ``/v2/state`` with a
  fallback (and an event) when the SM does not report them.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace

from tre_common.metrics_schema import MetricsSnapshot
from tre_controller.loops.action_queue import ActionQueue
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
    TransferIntent,
    _SlotOccupancy,
    build_plan,
)
from tre_controller.sm_client import ModelFloor, ServiceManagerClient, ServiceManagerError, parse_state_routable
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.api.v2 import ServiceManagerV2
from tre_sm.state.store import StateStore

from test_action_queue_review3 import ScriptedSM, _calls, _until, transfer_body
from test_o1_breakpoint_window_20261001 import _registry as _o1_registry
from test_o1_breakpoint_window_20261001 import _snap, _window
from test_planner_slot_occupancy import TOPOLOGY, _cls
from test_safescale_binding_commit import FakeRedis, InProcessServiceManager

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
            rescue_max_step_ratio=2.0, donor_surplus_release=True, scale_step_ratio=1.0,
        ),
        cluster_view=ClusterView(TOPOLOGY, bindings),
    )


def _intents(plan) -> list[TransferIntent]:
    return [a for a in plan.actions if isinstance(a, TransferIntent)]


# ===================================================================== planner


def test_crit_on_full_gpus_yields_a_transfer_intent_without_pods():
    plan = _plan(_e1())
    [intent] = _intents(plan)
    assert (intent.donor_model, intent.receiver_model, intent.count, intent.pairs) == (D, R, 1, 1)
    assert (intent.reason, intent.source_loop, intent.sleep_path) == ("critical_donor_immediate", "rescue", "urgent")
    assert not any(isinstance(a, ScaleAction) and a.pods for a in plan.actions)
    assert intent.rescue is not None and intent.rescue.target == intent.rescue.covered + 1


def test_a_free_gpu_is_used_before_a_relay_capacity_order_unchanged():
    # need 2: the free GPU node10/3 (R sleeps there) first, then one relay.
    plan = _plan(_e1(free={("node10", 3)}), need=2)
    kinds = [(type(a).__name__, getattr(a, "reason", None)) for a in plan.actions]
    assert kinds == [("ScaleAction", "critical_sleeping_capacity"), ("TransferIntent", "critical_donor_immediate")]
    wake, intent = plan.actions
    assert (wake.model, wake.delta, wake.pods) == (R, 1, ("8b-7",))
    assert (intent.count, intent.pairs) == (1, 1)


def test_pairable_count_bounds_the_intent_size():
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
        "idle", -1, "idle_proactive_immediate", (), "urgent"
    )
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
    assert occupancy.pairable_count(D, R, max_pairs=1, max_donors=1) == (0, 0)  # one pair needs two donors
    assert occupancy.pairable_count(D, R, max_pairs=5, max_donors=5) == (1, 2)  # r2-b is not a relay
    assert occupancy.released_donor_ids() == {"d-0", "d-1"}


# ===================================================================== ActionQueue


def _intent(count=1, *, rescue=None, source_loop="rescue", reason="critical_donor_immediate"):
    return TransferIntent("7b", "8b", count, reason, source_loop, rescue=rescue)


def test_partial_transfer_is_accounted_by_done_and_taken_not_count():
    rescue = RescuePlan(target=3, desired=3, base=1, covered=1)
    body = transfer_body(2, statuses=["done", "receiver_wake_failed"], unfilled=0)
    sm = ScriptedSM(results={"transfer:7b->8b": [{"ok": True, "response": body}]})
    queue = ActionQueue(sm, now_ms=lambda: 5_000)
    queue.submit((_intent(2, rescue=rescue),))
    donor, receiver = asyncio.run(queue.drain_once())
    assert (donor.model, donor.taken, donor.action_kind) == ("7b", 2, "transfer")
    assert (receiver.model, receiver.ok, receiver.done) == ("8b", True, 1)
    assert receiver.error == "partial: 1 of 2 pairs done"
    record = queue.rescue_targets()["8b"]
    assert (record.gained, record.failures, record.covered) == (1, 1, 2)
    assert queue.last_actions() == {"7b": (5_000, "down"), "8b": (5_000, "up")}
    # view_changes records BOTH sides of the relay (donor down, receiver up).
    assert queue.view_changes() == {"7b": (5_000, "down"), "8b": (5_000, "up")}
    assert queue.routable_changes() == {"7b": (5_000, -1), "8b": (5_000, 1)}
    events = queue.drain_events()
    assert "transfer_done:7b->8b:count=2:done=1:taken=2:unfilled=0:id=tr-1" in events
    assert "transfer_pair:7b->8b:7b-1->8b-1@n/1:receiver_wake_failed" in events
    assert queue.stats()["transfer_partial_total"] == 1


def test_unfilled_and_refusals_are_events_and_the_next_intent_is_accepted_at_once():
    refusal = {"error": "gpu_busy", "reason": "resident_awake", "node": "n", "gpu_ids": [3], "scope": "gpu"}
    body = transfer_body(1, unfilled=1, refusals=[refusal], skipped={"uncovered_gpu": 1})
    sm = ScriptedSM(results={"transfer:7b->8b": [{"ok": True, "response": body}]})
    queue = ActionQueue(sm)
    queue.submit((_intent(2),))
    asyncio.run(queue.drain_once())
    events = queue.drain_events()
    assert "wake_refused:8b:n/3:gpu_busy" in events
    assert "transfer_unfilled:7b->8b:1:uncovered_gpu=1" in events
    assert queue.submit((_intent(1),)).accepted == 1  # no cooldown of any kind


def test_clamped_by_floor_is_an_event_only_and_changes_nothing():
    body = transfer_body(0, clamped=True, unfilled=2, skipped={"donor_floor": 2})
    sm = ScriptedSM(results={"transfer:7b->8b": [{"ok": True, "response": body}]})
    queue = ActionQueue(sm)
    queue.submit((_intent(2),))
    donor, receiver = asyncio.run(queue.drain_once())
    assert (donor.taken, receiver.done, receiver.ok) == (0, 0, False)
    assert receiver.error == "clamped_by_floor: no donor replica taken"
    assert queue.last_actions() == {} and queue.view_changes() == {} and queue.routable_changes() == {}
    assert "transfer_clamped_by_floor:7b->8b:taken=0:count=2" in queue.drain_events()
    assert queue.stats()["transfer_clamped_by_floor_total"] == 1
    assert queue.submit((_intent(2),)).accepted == 1


def test_a_relay_is_finished_when_its_call_returns():
    """Plan B (SM global writer lock): the call returns with every pair done or failed;
    the queue frees both models at once and never looks the transfer up again."""
    body = transfer_body(2, statuses=["done", "receiver_wake_failed"])
    sm = ScriptedSM(results={"transfer:7b->8b": [{"ok": True, "response": body}]}, gated={"transfer:7b->8b"})

    async def scenario():
        queue = ActionQueue(sm)
        runner = asyncio.ensure_future(queue.run(poll_interval_s=0.001))
        queue.submit((_intent(2),))
        assert await _until(lambda: _calls(sm) == [("7b->8b", "transfer", 2)])
        assert queue.inflight_models() == {"7b", "8b"}  # held only while the call runs
        sm.gates["transfer:7b->8b"].set()
        assert await _until(lambda: queue.inflight_models() == set())
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
        return queue

    queue = asyncio.run(scenario())
    assert _calls(sm) == [("7b->8b", "transfer", 2)]
    assert queue.view_changes().keys() == {"7b", "8b"}


def test_transfer_404_degrades_to_not_executed_with_an_error_event(caplog):
    class Transport:
        async def request(self, method, url, *, json=None, timeout_s):
            raise ServiceManagerError("HTTP 404: Not Found", status=404, body={"detail": "Not Found"})

    queue = ActionQueue(ServiceManagerClient("http://sm", transport=Transport()))
    queue.submit((_intent(1),))
    with caplog.at_level("ERROR", logger="tre_controller.loops.action_queue"):
        donor, receiver = asyncio.run(queue.drain_once())
    assert (donor.ok, receiver.ok) == (False, False)
    assert receiver.error.startswith("transfer_unsupported")
    assert queue.view_changes() == {} and queue.last_actions() == {} and queue.inflight_models() == set()
    assert queue.drain_events() == ["transfer_unsupported:7b->8b"]
    assert queue.stats()["transfer_unsupported_total"] == 1
    assert any("transfer_endpoint_missing" in record.getMessage() for record in caplog.records)


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
        "transfer:7b->8b": [dict(routable_unknown, retriable=False), dict(writer_busy, retriable=False)],
        "scale:8b": [writer_busy],
        "scale:7b": [routable_unknown],
    })
    queue = ActionQueue(sm)
    queue.submit((_intent(1),))
    queue.submit((ScaleAction("9b", 1, "critical_idle_capacity", "rescue"),))
    asyncio.run(queue.drain_once())
    queue.submit((ScaleAction("8b", 1, "critical_idle_capacity", "rescue"),))
    queue.submit((ScaleAction("7b", -1, "idle_proactive_immediate", "rescue", sleep_path="urgent"),))
    results = asyncio.run(queue.drain_once())
    assert all(result.not_executed for result in results)
    # Nothing was executed: only the unrelated 9b scale-up is stamped; nothing cools down.
    assert set(queue.view_changes()) == {"9b"} and set(queue.last_actions()) == {"9b"}
    assert queue.submit((_intent(1),)).accepted == 1
    results = asyncio.run(queue.drain_once())  # the relay again: writer_busy this time
    assert all(result.not_executed for result in results) and set(queue.view_changes()) == {"9b"}


def test_target_scale_down_is_accounted_by_taken():
    def shrink(taken, clamped):
        return {"ok": True, "response": {"model": "7b", "actions": [], "taken": taken, "clamped_by_floor": clamped,
                                         "floor": {"clamped": clamped}}}

    sm = ScriptedSM(results={"scale:7b": [shrink(1, True), shrink(0, True)]})
    queue = ActionQueue(sm, now_ms=lambda: 7_000)
    down = ScaleAction("7b", -2, "idle_proactive_immediate", "rescue", sleep_path="urgent")
    queue.submit((down,))
    [first] = asyncio.run(queue.drain_once())
    assert (first.ok, first.taken) == (True, 1)
    assert queue.last_actions() == {"7b": (7_000, "down")} and queue.view_changes() == {"7b": (7_000, "down")}
    assert queue.drain_events() == ["scale_clamped_by_floor:7b:taken=1:asked=2"]

    fresh = ActionQueue(ScriptedSM(results={"scale:7b": [shrink(0, True)]}), now_ms=lambda: 8_000)
    fresh.submit((down,))
    [second] = asyncio.run(fresh.drain_once())
    assert (second.ok, second.taken) == (True, 0)
    assert fresh.last_actions() == {} and fresh.view_changes() == {} and fresh.routable_changes() == {}
    assert fresh.drain_events() == ["scale_clamped_by_floor:7b:taken=0:asked=2"]


def test_observe_mode_skips_an_intent_before_its_single_call():
    sm = ScriptedSM()
    queue = ActionQueue(sm, is_observe=lambda: True)
    queue.submit((_intent(1),))
    results = asyncio.run(queue.drain_once())
    assert _calls(sm) == [] and [(r.model, r.error) for r in results] == [("7b", "observe_skipped"), ("8b", "observe_skipped")]


# ===================================================================== sm_client / SM view


class _Transport:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    async def request(self, method, url, *, json=None, timeout_s):
        self.calls.append((method, url.replace("http://sm", ""), json))
        answer = self.answers[(method, url.replace("http://sm", ""))]
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_sm_client_transfer():
    body = transfer_body(1)
    transport = _Transport({("POST", "/v2/transfers"): body})
    client = ServiceManagerClient("http://sm", transport=transport)
    assert asyncio.run(client.transfer("7b", "8b", 1)) == {"ok": True, "response": body}
    assert transport.calls == [("POST", "/v2/transfers",
                                {"donor_model": "7b", "receiver_model": "8b", "count": 1, "sleep_path": "urgent"})]

    partial = dict(transfer_body(1, statuses=["donor_sleep_failed"], done=0, taken=0), error="partial")
    transport.answers[("POST", "/v2/transfers")] = ServiceManagerError("HTTP 409", status=409, body=partial)
    result = asyncio.run(client.transfer("7b", "8b", 1))
    assert (result["ok"], result["partial"], result["retriable"]) == (False, True, False)
    assert result["response"]["pairs"][0]["status"] == "donor_sleep_failed"
    transport.answers[("POST", "/v2/transfers")] = ServiceManagerError("HTTP 404", status=404, body={"detail": "x"})
    assert asyncio.run(client.transfer("7b", "8b", 1))["unsupported"] is True


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


def test_offline_replay_plans_an_intent_and_the_fake_sm_names_the_pods():
    from test_scaling_e2e import CRITICAL, HIGH, REG_7B_8B, _registry, _window as _e2e_window

    registry = _registry(*REG_7B_8B)
    bindings = list(_e1())
    store = StateStore(FakeRedis())
    store.save(bindings, expected_version=0)
    service = ServiceManagerV2(registry, store)
    view = cluster_view_from_state(service.get_state(), registry.topology())
    snapshot = MetricsSnapshot(ts_ms=60_000, stale=False, models={
        R: _e2e_window(R, per_q=CRITICAL, running=1.0), D: _e2e_window(D, per_q=HIGH, running=8.0),
    })
    replay = run_tick_replay(
        [TickReplayStep(snapshot, rescue_due=True, fairness_due=False, cluster_view=view)],
        registry=registry, suppress_hot_proactive_probe=True,
    )
    [intent] = [a for a in replay.actions if isinstance(a, TransferIntent)]
    decision = build_decision_snapshot("rescue", snapshot, replay.results[0])
    [payload] = [a for a in json.loads(decision["actions"]) if a["kind"] == "transfer"]
    assert (payload["donor"], payload["receiver"], payload["count"]) == (D, R, intent.count)
    assert "pods" not in payload

    queue = ActionQueue(InProcessServiceManager(service))
    queue.submit((intent,))
    results = asyncio.run(queue.drain_once())
    assert all(result.ok for result in results)
    pairs = [event for event in queue.drain_events() if event.startswith("transfer_pair:")]
    assert len(pairs) == intent.count and all(event.endswith(":done") for event in pairs)
    # pod names come from the SM response: 7b-<i> -> 8b-<i> on the same GPU
    donor_pod, receiver_pod = pairs[0].split(":")[2].split("@")[0].split("->")
    assert donor_pod.replace("7b", "8b") == receiver_pod


def test_signal_log_records_both_sides_of_a_relay():
    from tre_controller.loops.signal_log import _summarize_actions

    labels, deltas = _summarize_actions((TransferIntent("7b", "8b", 2, "critical_donor_immediate", "rescue", pairs=1),))
    assert labels == {"7b": "transfer:7b->8b", "8b": "transfer:7b->8b"}
    assert deltas == {"7b": -2, "8b": 1}

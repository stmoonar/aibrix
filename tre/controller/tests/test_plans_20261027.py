"""Controller side of POST /v2/plans (design 20261027-sm-plans §5, §8 items 10-12)."""

from __future__ import annotations

import asyncio

from tre_controller.loops.action_queue import ActionQueue
from tre_controller.planning.planner import (
    HideAction,
    ReceiverTarget,
    RescuePlan,
    SafeScaleCommitAction,
    ScaleAction,
    TransferIntent,
)

from plan_shim import PlanSM, done_plan


def _commit(*, pods=("7b-1",), upscales=(), donor_done=False):
    return SafeScaleCommitAction(
        donor="7b", pods=pods, reason="formal_commit_gate_passed", upscales=tuple(upscales),
        request_id="7b-0", donor_done=donor_done,
    )


# ------------------------------------------------------------------ (10) no-op fix
def test_a_target_already_met_makes_no_sm_call_and_stamps_nothing():
    """§5.5: a receiver whose target the SM state already meets (and a shrink the
    serving floor leaves at its count) is answered locally: no POST, ``woke_none`` /
    ``taken: 0``, no O1 / view stamp, no last action - the probe still resolves."""
    done = []
    sm = PlanSM({"8b": 3, "m": 1})
    queue = ActionQueue(sm, now_ms=lambda: 9_000, on_oneshot_done=lambda *a: done.append(a))
    queue.submit((_commit(upscales=(ReceiverTarget("8b", 1, 3),), donor_done=True),))
    queue.submit((ScaleAction("m", -1, "idle_proactive_immediate", "rescue", sleep_path="urgent"),))
    results = asyncio.run(queue.drain_once())

    assert sm.plans == [] and sm.state_reads == 2
    by_model = {result.model: result for result in results}
    assert by_model["8b"].ok and by_model["8b"].woke_none
    assert by_model["m"].ok and by_model["m"].taken == 0
    assert queue.view_changes() == {} and queue.routable_changes() == {} and queue.last_actions() == {}
    assert done == [("7b-0", "commit", "formal_commit_gate_passed")]


def test_an_sm_side_noop_wake_is_no_up_action():
    """§5.5: the SM answering a receiver item ``noop`` (met under its lock) woke nothing."""

    def noop(request):
        body = done_plan(request)
        for item in body["wakes"]:
            item.update(status="noop", woke=[])
        return {"ok": True, "status": 200, "response": body}

    sm = PlanSM({"8b": 1}, respond=noop)
    queue = ActionQueue(sm, now_ms=lambda: 9_000)
    queue.submit((_commit(upscales=(ReceiverTarget("8b", 1, 2),), donor_done=True),))
    [result] = asyncio.run(queue.drain_once())
    assert len(sm.plans) == 1 and result.ok and result.woke_none
    assert queue.last_actions() == {} and queue.view_changes() == {}


# ------------------------------------------------------------------ (11) SafeScale commit = one plan per attempt
def test_a_commit_is_one_plan_and_its_retry_resends_only_the_pending_receiver():
    answers = []

    def respond(request):
        body = done_plan(request, sm.awake)
        if not answers:
            # First attempt: the donor slept, 8b woke, 14b was refused (a Pod loading).
            body["wakes"][1].update(status="refused", woke=[], error="resident_loading", retriable=True)
        answers.append(request)
        return {"ok": True, "status": 200, "response": body}

    sm = PlanSM({"8b": 2, "14b": 1}, respond=respond)
    done = []
    queue = ActionQueue(sm, sleep=lambda _s: asyncio.sleep(0), now_ms=lambda: 7_000,
                        on_oneshot_done=lambda *a: done.append(a))
    queue.submit((_commit(upscales=(ReceiverTarget("8b", 1, None, 8), ReceiverTarget("14b", 1, 2))),))
    results = asyncio.run(queue.drain_once())

    first, retry = sm.plans
    assert first == {
        "sleeps": [{"model": "7b", "reason": "formal_commit_gate_passed", "sleep_path": "safescale_commit",
                    "pods": ["7b-1"]}],
        "wakes": [{"model": "8b", "reason": "safescale_receiver", "at_least": 3},
                  {"model": "14b", "reason": "safescale_receiver", "at_least": 2}],
    }
    # The retry: no donor sleep again, only the pending receiver with the same target.
    assert retry == {"sleeps": [], "wakes": [{"model": "14b", "reason": "safescale_receiver", "at_least": 2}]}
    assert all(result.ok for result in results) and len(results) == 3
    assert {m: d for m, (_, d) in queue.last_actions().items()} == {"7b": "down", "8b": "up", "14b": "up"}
    assert done == [("7b-0", "commit", "formal_commit_gate_passed")]


def test_a_receiver_on_a_free_gpu_may_wake_when_the_donor_sleep_fails():
    """Accepted semantics (§5.4): the SM wakes a receiver on a free GPU (wave 1) even
    when the donor sleep fails; a receiver on the donor's GPU is refused
    ``donor_sleep_failed``. The commit then fails for good: the donor pod is unhidden,
    the woken receiver is an "up", the refused one is dropped."""

    def respond(request):
        body = done_plan(request, sm.awake)
        body["sleeps"][0].update(status="failed", taken=0, error="sleep_failed", retriable=False,
                                 outcomes=[{"serve_id": "7b-1", "status": "rolled_back"}])
        body["wakes"][1].update(status="refused", woke=[], error="donor_sleep_failed")
        return {"ok": True, "status": 200, "response": body}

    sm = PlanSM({"8b": 1, "14b": 1}, respond=respond)
    done = []
    queue = ActionQueue(sm, now_ms=lambda: 7_000, on_oneshot_done=lambda *a: done.append(a))
    queue.submit((_commit(upscales=(ReceiverTarget("8b", 1, 2), ReceiverTarget("14b", 1, 2))),))
    results = asyncio.run(queue.drain_once())

    assert len(sm.plans) == 1 and sm.routable == [("7b", ())]  # the donor pod unhidden
    by_model: dict[str, list] = {}
    for result in results:
        by_model.setdefault(result.model, []).append(result)
    assert by_model["8b"][0].ok and not by_model["8b"][0].woke_none
    assert by_model["14b"][0].error.startswith("donor_sleep_failed")
    assert {m: d for m, (_, d) in queue.last_actions().items()} == {"8b": "up"}
    assert done and done[0][1] == "rollback"


def test_a_whole_plan_refusal_keeps_every_commit_member_pending():
    """409 ``writer_busy`` refuses the whole plan before any change: nothing is
    stamped and the retry re-sends the same plan (named pods, frozen targets)."""
    busy = {"ok": False, "error": "HTTP 409", "status": 409, "retriable": True, "code": "writer_busy",
            "not_executed": True}
    answers = [busy]
    seen_at_retry = []

    def respond(request):
        if answers:
            return answers.pop(0)
        seen_at_retry.append(queue.view_changes())
        return {"ok": True, "status": 200, "response": done_plan(request, sm.awake)}

    sm = PlanSM({"8b": 1}, respond=respond)
    queue = ActionQueue(sm, sleep=lambda _s: asyncio.sleep(0), now_ms=lambda: 7_000)
    queue.submit((_commit(upscales=(ReceiverTarget("8b", 1, None),)),))
    results = asyncio.run(queue.drain_once())
    first, retry = sm.plans
    assert first == retry  # same named sleep, same frozen target (1 + 1)
    assert first["wakes"] == [{"model": "8b", "reason": "safescale_receiver", "at_least": 2}]
    assert seen_at_retry == [{}]  # the refused plan stamped nothing
    assert all(result.ok for result in results) and results[0].attempts == 2


# ------------------------------------------------------------------ (12) one submit -> one plan
def test_one_submit_of_different_models_is_one_plan_with_one_item_per_action():
    """§5.2 as narrowed: [+k on R1, D->R2, shrink D2] of one submit = ONE POST with three
    items; each member keeps its own accounting (rescue credit, relay, F4 memory)."""
    rescue = RescuePlan(target=3, desired=3, base=1, covered=1)
    sm = PlanSM({"R1": 1, "R2": 1, "D": 4, "D2": 3})
    queue = ActionQueue(sm, now_ms=lambda: 5_000)
    submitted = queue.submit((
        ScaleAction("R1", 2, "critical_sleeping_capacity", "rescue", rescue=rescue),
        TransferIntent("D", "R2", 1, "critical_donor_immediate", "rescue"),
        ScaleAction("D2", -1, "idle_proactive_immediate", "rescue", sleep_path="urgent"),
    ))
    assert submitted.accepted == 3 and len(queue.pending_actions()) == 1
    results = asyncio.run(queue.drain_once())

    assert sm.plans == [{
        "sleeps": [{"model": "D2", "reason": "idle_proactive_immediate", "sleep_path": "urgent", "at_most": 2}],
        "wakes": [
            {"model": "R1", "reason": "critical_sleeping_capacity", "at_least": 3},
            {"model": "R2", "reason": "critical_donor_immediate", "at_least": 2,
             "donors": [{"model": "D", "max": 1, "sleep_path": "urgent"}]},
        ],
    }]
    assert sm.state_reads == 1
    assert [(r.model, r.action_kind, r.ok) for r in results] == [
        ("R1", "scale", True), ("D", "transfer", True), ("R2", "transfer", True), ("D2", "scale", True),
    ]
    record = queue.rescue_targets()["R1"]
    assert (record.gained, record.outstanding, record.done_ms) == (2, 0, 5_000)
    assert {m: d for m, (_, d) in queue.last_actions().items()} == {"R1": "up", "R2": "up", "D": "down", "D2": "down"}
    assert any(event.startswith("transfer_done:D->R2:count=1:done=1:taken=1") for event in queue.drain_events())
    assert queue.relay_holds() == {} and queue.inflight_models() == set()


def test_a_second_action_on_a_used_model_starts_the_next_plan():
    """Same-model actions are never merged into one item: the rescue path may submit
    a +k and a relay for one receiver; the relay starts the next plan, which runs
    after the first (they share the receiver), and both parts count for the target."""
    rescue = RescuePlan(target=4, desired=4, base=1, covered=1)
    sm = PlanSM({"R1": 1, "R2": 1, "D": 4, "D3": 4})
    queue = ActionQueue(sm, now_ms=lambda: 6_000)
    submitted = queue.submit((
        ScaleAction("R1", 1, "critical_sleeping_capacity", "rescue", rescue=rescue),
        ScaleAction("R2", 1, "critical_sleeping_capacity", "rescue"),
        TransferIntent("D3", "R1", 1, "critical_donor_immediate", "rescue", rescue=rescue),
    ))
    assert submitted.accepted == 3 and len(queue.pending_actions()) == 2
    asyncio.run(queue.drain_once())

    first, second = sm.plans
    assert [item["model"] for item in first["wakes"]] == ["R1", "R2"] and first["sleeps"] == []
    assert second == {"sleeps": [], "wakes": [{"model": "R1", "reason": "critical_donor_immediate", "at_least": 2,
                                               "donors": [{"model": "D3", "max": 1, "sleep_path": "urgent"}]}]}
    record = queue.rescue_targets()["R1"]
    assert (record.gained, record.outstanding) == (2, 0)


def test_a_member_never_jumps_over_another_action_of_its_model():
    """A hide of B between a scale-up of A and a (rescue) scale-up of B: the B scale-up is
    not fused forward into A's plan (it would run before the hide); it starts the next
    plan, which a later scale-up of C joins."""
    sm = PlanSM({"A": 1, "B": 2, "C": 1})
    queue = ActionQueue(sm)
    queue.submit((
        ScaleAction("A", 1, "critical_sleeping_capacity", "rescue"),
        HideAction("B", ("b-1",), "probe_started", "rescue"),
        ScaleAction("B", 1, "critical_sleeping_capacity", "rescue"),
        ScaleAction("C", 1, "critical_sleeping_capacity", "rescue"),
    ))
    pending = queue.pending_actions()
    assert [type(item.action).__name__ for item in pending] == ["ScaleAction", "HideAction", "PlanBatch"]
    assert [member.model for member in pending[2].action.members] == ["B", "C"]
    asyncio.run(queue.drain_once())
    assert [[item["model"] for item in plan["wakes"]] for plan in sm.plans] == [["A"], ["B", "C"]]
    assert sm.routable == [("B", ("b-1",))]


def test_a_rescue_replaces_only_the_fairness_member_of_a_pending_plan():
    queue = ActionQueue(PlanSM({"m": 2, "n": 1}))
    queue.submit((
        ScaleAction("m", -1, "high_release_immediate", "fairness", sleep_path="urgent"),
        ScaleAction("n", 1, "low_sleeping_capacity", "fairness"),
    ))
    [batch] = queue.pending_actions()
    assert [member.model for member in batch.action.members] == ["m", "n"]
    rescue = queue.submit((ScaleAction("m", 1, "critical_sleeping_capacity", "rescue"),))
    assert rescue.accepted == 1 and rescue.replaced == (("m", "fairness"),)
    assert [(item.action.model, item.action.delta) for item in queue.pending_actions()] == [("n", 1), ("m", 1)]


def test_a_rescue_member_on_a_busy_model_is_not_fused():
    """Review 2026-10-28: a rescue member whose model a running action holds (the
    rescue path skips the inflight check) made the whole fused plan wait for it; it
    stays its own queued action and the free members form the plan."""
    queue = ActionQueue(PlanSM({"8b": 1, "n": 1, "m": 1}))
    running = queue._queued(ScaleAction("8b", 1, "low_sleeping_capacity", "fairness"))
    queue._running[object()] = running  # a dispatch of 8b in progress
    queue._inflight.update(running.models)

    queue.submit((
        ScaleAction("8b", 1, "critical_sleeping_capacity", "rescue"),
        ScaleAction("n", 1, "critical_sleeping_capacity", "rescue"),
        ScaleAction("m", 1, "critical_sleeping_capacity", "rescue"),
    ))

    pending = queue.pending_actions()
    assert [type(item.action).__name__ for item in pending] == ["ScaleAction", "PlanBatch"]
    assert pending[0].action.model == "8b"
    assert [member.model for member in pending[1].action.members] == ["n", "m"]

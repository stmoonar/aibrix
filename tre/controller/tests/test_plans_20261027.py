"""Controller side of POST /v2/plans (design 20261027-sm-plans §5, §8 items 10-12)."""

from __future__ import annotations

import asyncio

from tre_controller.loops.action_queue import ActionQueue
from tre_controller.planning.planner import ReceiverTarget, SafeScaleCommitAction, ScaleAction

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

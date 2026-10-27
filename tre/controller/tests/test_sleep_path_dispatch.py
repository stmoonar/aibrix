"""Controller -> SM sleep path (plan 2026-09-27 D1), as plan items (design 20261027-sm-plans).

The drain budget is no longer sent: the SM never drains (since 2026-10-02) and a plan
item has no budget field."""

from __future__ import annotations

import asyncio

import pytest

from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.plan_items import SleepPathRefused, plan_item, sleep_path_of
from tre_controller.planning.planner import ScaleAction

from plan_shim import PlanSM


def _dispatch(*actions, awake=None):
    sm = PlanSM(awake if awake is not None else {"m": 2})
    queue = ActionQueue(sm)
    for action in actions:
        queue.submit((action,))
        asyncio.run(queue.drain_once())
    return sm


def test_immediate_donor_paths_are_urgent():
    sm = _dispatch(ScaleAction("m", -1, "critical_donor_immediate", "rescue"))
    assert sm.plans == [{"sleeps": [{"model": "m", "reason": "critical_donor_immediate", "sleep_path": "urgent",
                                     "at_most": 1}], "wakes": []}]


def test_pathless_scale_down_is_refused_not_sent_to_the_sm_default():
    # 2026-10-02: no implicit `scale_down` path; the dispatch reports a failed result and
    # nothing reaches the SM (not even the plan's GET).
    action = ScaleAction("m", -1, "high_release", "fairness")
    with pytest.raises(SleepPathRefused, match="no sleep_path"):
        sleep_path_of(action)
    sm = PlanSM({"m": 2})
    queue = ActionQueue(sm)
    queue.submit((action,))
    results = asyncio.run(queue.drain_once())
    assert sm.plans == [] and sm.state_reads == 0
    assert [r.ok for r in results] == [False]
    assert results[0].error.startswith("sleep_path_refused") and results[0].retriable is False
    assert queue.stats()["sleep_path_refused_total"] == 1 and queue.view_changes() == {}


def test_pathless_binding_scale_down_is_refused_too():
    sm = _dispatch(ScaleAction("m", -1, "high_release", "fairness", pods=("pod-a",)))
    assert sm.plans == []


def test_plan_items_keep_explicit_paths_and_scale_ups_carry_none():
    assert plan_item(ScaleAction("m", 1, "x", "rescue")) == ("wake", {"model": "m", "reason": "x", "_delta": 1})
    assert plan_item(ScaleAction("m", -1, "x_immediate", "rescue"))[1]["sleep_path"] == "urgent"
    assert plan_item(ScaleAction("m", -1, "x", "safescale", sleep_path="safescale_commit"))[1]["sleep_path"] == (
        "safescale_commit"
    )


def test_scale_up_is_an_absolute_at_least_without_sleep_fields():
    sm = _dispatch(ScaleAction("m", 2, "critical", "rescue"))
    assert sm.plans == [{"sleeps": [], "wakes": [{"model": "m", "reason": "critical", "at_least": 4}]}]


def test_safescale_commit_sends_its_path_for_the_named_pods():
    sm = _dispatch(
        ScaleAction(
            "m", -1, "formal_commit_gate_passed", "safescale", pods=("pod-a",),
            sleep_path="safescale_commit", drain_budget_s=42.0,
        )
    )
    assert sm.plans == [{"sleeps": [{"model": "m", "reason": "formal_commit_gate_passed",
                                     "sleep_path": "safescale_commit", "pods": ["pod-a"]}], "wakes": []}]
    assert sm.state_reads == 0  # named pods need no awake count

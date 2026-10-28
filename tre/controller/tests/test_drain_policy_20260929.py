"""Fast-loop donors name the SM "urgent" sleep path explicitly (2026-09-29).

With the default SM registry that path does not drain (hide -> ack -> /sleep
mode=abort, the reissue sidecar continues the cut-off requests, v1 semantics);
the SafeScale commit keeps its own path. The planner carries the sleep path /
drain budget through ``_add_scale_action`` to the dispatch.
"""

from __future__ import annotations

import asyncio

from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.plan_items import plan_item
from tre_controller.planning.classify import ModelRole, ModelState
from tre_controller.planning.planner import (
    IMMEDIATE_DONOR_SLEEP_PATH,
    PlanConfig,
    SafeScaleCommitAction,
    ScaleAction,
    _add_scale_action,
    build_plan,
)
from relay_view import expand_relays, relays, pool_view# noqa: F401 - 2026-10-02 relay intents

from test_planner import _classification


from plan_shim import PlanSM


def _plan():
    classifications = [
        _classification("critical", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
        _classification("low", ModelState.LOW, ModelRole.RECEIVER, 0.9),
        _classification("idle", ModelState.IDLE, ModelRole.DONOR, 10.0, "idle"),
    ]
    contexts = {
        "critical": {"assigned_replicas": 2, "routable_pods": 2},
        "low": {"assigned_replicas": 1, "routable_pods": 1},
        "idle": {"assigned_replicas": 4, "routable_pods": 4},
    }
    return build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"critical": 2, "low": 1, "idle": 4},
        idle_gpus=0,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4),
        cluster_view=pool_view({"critical": 2, "low": 1, "idle": 4}, receivers=("critical", "low")),
    )


def test_immediate_donors_carry_the_urgent_sleep_path_and_receivers_none():
    assert IMMEDIATE_DONOR_SLEEP_PATH == "urgent"
    plan = _plan()
    # A relay (a named transfer since 2026-10-28) carries the donor's sleep path.
    assert {(r.reason, r.sleep_path) for r in relays(plan.actions)} == {
        ("critical_donor_immediate", "urgent"), ("low_fairness_donor_immediate", "urgent"),
    }
    actions = [a for a in expand_relays(plan.actions) if isinstance(a, ScaleAction)]
    donors = [a for a in actions if a.delta < 0]
    receivers = [a for a in actions if a.delta > 0]
    assert {a.reason for a in donors} >= {"critical_donor_immediate", "low_fairness_donor_immediate"}
    assert all(a.reason.endswith("_immediate") for a in donors)
    assert all(a.sleep_path == "urgent" and a.drain_budget_s is None for a in donors)
    assert receivers and all(a.sleep_path is None for a in receivers)


def test_add_scale_action_passes_sleep_path_and_budget_only_for_a_shrink():
    actions: list = []
    deltas: dict = {}
    _add_scale_action(
        actions, deltas, model="m", delta=-1, reason="x_immediate", source_loop="rescue",
        sleep_path="urgent", drain_budget_s=5.0,
    )
    _add_scale_action(
        actions, deltas, model="n", delta=1, reason="x_immediate", source_loop="rescue",
        sleep_path="urgent", drain_budget_s=5.0,
    )
    assert (actions[0].sleep_path, actions[0].drain_budget_s) == ("urgent", 5.0)
    assert (actions[1].sleep_path, actions[1].drain_budget_s) == (None, None)
    # The plan item carries the path; the SM never drains, so no budget is sent.
    assert plan_item(actions[0])[1]["sleep_path"] == "urgent"
    assert "sleep_path" not in plan_item(actions[1])[1]


def test_idle_proactive_immediate_donor_dispatches_on_the_urgent_path():
    classifications = [_classification("idle", ModelState.IDLE, ModelRole.DONOR, 10.0, "idle")]
    plan = build_plan(
        model_contexts={"idle": {"assigned_replicas": 4, "routable_pods": 4, "Y_m": 0.0, "Q": 0.0}},
        classifications=classifications,
        model_replicas={"idle": 4},
        idle_gpus=0,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4),
    )
    shrink = next(a for a in expand_relays(plan.actions) if isinstance(a, ScaleAction) and a.model == "idle")
    assert shrink.reason == "idle_proactive_immediate" and shrink.sleep_path == "urgent"

    sm = PlanSM({"idle": 4})
    queue = ActionQueue(sm)
    queue.submit((shrink,))
    asyncio.run(queue.drain_once())
    [plan] = sm.plans
    assert plan["sleeps"] == [{"model": "idle", "reason": "idle_proactive_immediate", "sleep_path": "urgent",
                               "at_most": 4 + shrink.delta}]


def test_safescale_commit_keeps_its_own_path():
    # The SM never drains (since 2026-10-02): a plan sleep item has no drain budget.
    commit = SafeScaleCommitAction(donor="m", pods=("pod-a",), reason="gate", drain_budget_s=60.0)
    assert plan_item(commit.donor_sleep()) == (
        "sleep", {"model": "m", "reason": "gate", "sleep_path": "safescale_commit", "pods": ["pod-a"]}
    )

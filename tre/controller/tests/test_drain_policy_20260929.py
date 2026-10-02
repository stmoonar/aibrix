"""Fast-loop donors name the SM "urgent" sleep path explicitly (2026-09-29).

With the default SM registry that path does not drain (hide -> ack -> /sleep
mode=abort, the reissue sidecar continues the cut-off requests, v1 semantics);
the SafeScale commit keeps its own path. The planner carries the sleep path /
drain budget through ``_add_scale_action`` to the dispatch.
"""

from __future__ import annotations

import asyncio

from tre_controller.loops.action_queue import ActionQueue, _sleep_kwargs
from tre_controller.planning.classify import ModelRole, ModelState
from tre_controller.planning.planner import (
    IMMEDIATE_DONOR_SLEEP_PATH,
    PlanConfig,
    SafeScaleCommitAction,
    ScaleAction,
    _add_scale_action,
    build_plan,
)
from relay_view import expand_relays, relays  # noqa: F401 - 2026-10-02 relay intents

from test_planner import _classification


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def scale_model(self, model, delta, **kwargs):
        self.calls.append(("scale", model, delta, kwargs))
        return {"ok": True}

    async def set_binding_power(self, serve_id, *, awake, **kwargs):
        self.calls.append(("power", serve_id, awake, kwargs))
        return {"ok": True}


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
    )


def test_immediate_donors_carry_the_urgent_sleep_path_and_receivers_none():
    assert IMMEDIATE_DONOR_SLEEP_PATH == "urgent"
    plan = _plan()
    # 2026-10-02: a relay is one TransferIntent carrying the donor's sleep path.
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
    assert _sleep_kwargs(actions[0]) == {"sleep_path": "urgent", "drain_budget_s": 5.0}
    assert _sleep_kwargs(actions[1]) == {}


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

    client = RecordingClient()
    queue = ActionQueue(client)
    queue.submit((shrink,))
    asyncio.run(queue.drain_once())
    assert client.calls == [("scale", "idle", shrink.delta, {"sleep_path": "urgent"})]


def test_safescale_commit_keeps_its_own_path_and_the_window_budget():
    # The SM ignores the budget while safescale_commit is a no-drain path; it is
    # still sent so that no_drain_paths: [] restores the probe-window drain.
    commit = SafeScaleCommitAction(donor="m", pods=("pod-a",), reason="gate", drain_budget_s=60.0)
    assert _sleep_kwargs(commit.donor_sleep()) == {
        "sleep_path": "safescale_commit",
        "drain_budget_s": 60.0,
    }

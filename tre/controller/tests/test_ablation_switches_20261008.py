"""Controller ablation switches (2026-10-08).

TRE_ABLATION_DISABLE_SAFESCALE: SafeScale off = immediate release (as before 05f489f1 /
v1). The planner loops get no SafeScale, every shrink that would run as a probe takes
the urgent donor path, and probes an earlier run left in Redis are rolled back once.

TRE_ABLATION_DISABLE_SLOW_LOOP (2026-10-28 semantics): the single decision loop never
runs the rebalance section (test_single_loop_20261028.py).
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import tre_controller.app as app
from tre_common.metrics_schema import MetricsSnapshot
import pytest

from tre_controller.app import build_controller_task_specs
from tre_controller.config import ControllerConfig, SafeScaleConfig
from section_ticks import run_rescue_tick
from tre_controller.loops.safescale_task import rollback_left_probes_task
from tre_controller.planning.classify import ModelRole, ModelState
from tre_controller.planning.planner import (
    ClusterView,
    PlanConfig,
    RelayHold,
    ScaleAction,
    TransferIntent,
    UnhideAction,
    build_plan,
    relay_basis,
)
from tre_controller.planning.safescale import SafeScaleStateMachine
from tre_controller.store.state_store import ControllerStateStore
from tre_sm.allocator.slots import Binding, Slot

from test_b8_observe_probes import _probe_snapshot
from test_controller_app import _cfg, _deps
from test_floor_plan_20260929 import _cls, _topology
from test_loop_ticks import FakeQueue, _metrics_with_pods, _registry_with_models
from test_safescale_commit import FakeRedis


# ------------------------------------------------------------- SafeScale off
def test_safescale_off_decision_loop_gets_no_safescale_and_no_safescale_loop_runs(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(app, "DecisionLoop", lambda **kwargs: seen.update(kwargs))
    specs = {spec.name: spec for spec in build_controller_task_specs(_deps(), _cfg(ablation_disable_safescale=True))}

    assert "safescale" not in specs and "safescale_leftover_rollback" in specs and "decision" in specs
    assert seen["safescale"] is None and seen["leftover_probes"] is not None


def test_high_donor_for_a_critical_receiver_is_an_urgent_transfer():
    snapshot = _probe_snapshot()
    high = _metrics_with_pods("donor", generation=200.0, waiting=0.0, running=1.0, pods=("donor-a", "donor-b"))
    snapshot = replace(snapshot, models={**snapshot.models, "donor": high})
    queue = FakeQueue()

    result = run_rescue_tick(snapshot, queue=queue, registry=_registry_with_models("critical", "donor"))

    assert result.classifications["donor"].state.value == "high"
    assert queue.submitted == [
        (TransferIntent("donor", "critical", 1, "critical_high_donor_safescale_nosafescale", "rescue"),)
    ]
    assert queue.submitted[0][0].sleep_path == "urgent"


def test_high_proactive_shrink_is_an_urgent_model_level_scale_down():
    snapshot = MetricsSnapshot(
        ts_ms=1,
        stale=False,
        models={"high": _metrics_with_pods("high", generation=400.0, waiting=0.0, running=1.0, pods=("h-a", "h-b"))},
    )

    result = run_rescue_tick(snapshot, queue=FakeQueue(), registry=_registry_with_models("high"))

    assert result.classifications["high"].state.value == "high"
    [action] = result.actions
    assert (action.model, action.delta, action.sleep_path) == ("high", -1, "urgent")
    assert action.reason == "high_proactive_safescale_nosafescale" and not action.requires_safescale
    assert not any(event.startswith("safescale_probe") for event in result.events)


def _two_donor_plan(relay_holds=None):
    """CRITICAL tp1 receiver ``r`` (sleeping bindings on GPUs two HIGH donors hold)."""
    bindings = (
        Binding("h1-0", "h1", Slot("node-a", (0,)), awake=True),
        Binding("h1-1", "h1", Slot("node-a", (1,)), awake=True),
        Binding("h2-2", "h2", Slot("node-a", (2,)), awake=True),
        Binding("h2-3", "h2", Slot("node-a", (3,)), awake=True),
        Binding("r-0", "r", Slot("node-a", (0,)), awake=False),
        Binding("r-2", "r", Slot("node-a", (2,)), awake=False),
    )
    view = ClusterView(topology=_topology(), bindings=bindings, fetched_ms=1_000, state_ms=1_000, sm_version=7)
    plan = build_plan(
        model_contexts={
            "r": {"assigned_replicas": 0, "routable_pods": 0},
            "h1": {"assigned_replicas": 2, "routable_pods": 2},
            "h2": {"assigned_replicas": 2, "routable_pods": 2},
        },
        classifications=[
            _cls("r", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
            _cls("h1", ModelState.HIGH, ModelRole.DONOR, 1.4, "surplus"),
            _cls("h2", ModelState.HIGH, ModelRole.DONOR, 1.4, "surplus"),
        ],
        model_replicas={"r": 0, "h1": 2, "h2": 2},
        idle_gpus=0,
        cfg=PlanConfig(
            min_replicas_per_model=1, max_replicas_per_model=4, suppress_hot_proactive_probe=True,
            release_without_safescale=True,
        ),
        cluster_view=view,
        relay_holds=relay_holds,
    )
    return plan, view


def test_a_held_pair_is_not_planned_and_the_receiver_uses_the_next_donor():
    plan, view = _two_donor_plan()
    [first] = [a for a in plan.actions if isinstance(a, TransferIntent)]
    assert first.receiver == "r" and first.reason == "critical_high_donor_safescale_nosafescale"
    assert not any(isinstance(a, ScaleAction) and a.requires_safescale for a in plan.actions)
    held = {(first.donor, "r"): RelayHold(basis=relay_basis(view, first.donor, "r"), reason="refused")}

    again, _ = _two_donor_plan(held)

    relays = [a for a in again.actions if isinstance(a, TransferIntent)]
    assert [(a.donor, a.receiver) for a in relays] == [({"h1": "h2", "h2": "h1"}[first.donor], "r")]
    assert f"relay_held:{first.donor}:r:refused" in again.events


class _AcceptingQueue:
    def __init__(self) -> None:
        self.submitted: list[tuple] = []

    def submit(self, actions):
        self.submitted.append(tuple(actions))
        return SimpleNamespace(accepted=len(actions))


async def _no_sleep(_seconds: float) -> None:
    return None


def test_startup_rolls_back_a_probe_left_in_redis():
    store = ControllerStateStore(FakeRedis())
    SafeScaleStateMachine(config=SafeScaleConfig(), store=store).start_probe(
        model="donor", pods=("donor-a",), now_ms=0
    )
    restarted = SafeScaleStateMachine(config=SafeScaleConfig(), store=store)
    assert restarted.restore() == 1
    queue = _AcceptingQueue()

    asyncio.run(
        rollback_left_probes_task(queue=queue, safescale=restarted, interval_s=0.0, sleep=_no_sleep, now_ms=lambda: 5_000)
    )

    assert queue.submitted == [(UnhideAction("donor", ("donor-a",), "safescale_disabled", "safescale"),)]
    assert restarted.all_probes() == ()
    assert SafeScaleStateMachine(config=SafeScaleConfig(), store=store).restore() == 0


# ------------------------------------------------------------- slow loop off
def test_the_removed_fast_loop_switch_fails_closed():
    with pytest.raises(ValueError, match="TRE_ABLATION_DISABLE_SLOW_LOOP"):
        ControllerConfig.from_env({"TRE_ABLATION_DISABLE_FAST_LOOP": "true"})

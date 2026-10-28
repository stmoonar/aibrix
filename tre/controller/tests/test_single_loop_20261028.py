"""Single decision loop (design docs/design/20261028-single-loop.md): the invariants.

1. A same-slot release is ONE plan: the donor pod's sleep and the receiver's wake with
   its binding on that slot as the hint - never a bare donor sleep, under every scope;
   an IDLE donor's release is that plan at once even under scope ``all`` (no probe).
2. Order: two CRITICAL receivers compete for one free GPU - the lower Z wins whatever
   the registry order; a LOW receiver is never planned before a CRITICAL one.
3. One step: a receiver gets at most +1 per plan however deep its Z and however much
   capacity is free; no second +1 before the window reflects the previous target.
5. One ledger: the rescue and the rebalance section of one planner call never assign
   one free GPU twice; with TRE_ABLATION_DISABLE_SLOW_LOOP a LOW receiver gets nothing.
4. TP2 supply: a slot of one free GPU and an IDLE donor pod, or of two pods of one
   IDLE donor, is obtained in one plan; a mixed-donor slot gives nothing and
   ``capacity_blocked``.
6. A stale snapshot (or no publish at all) gives no decision and an alert. (The same
   observation stream at a 1 s and a 2 s cadence: test_period_invariance_20261025.)
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tre_common.metrics_schema import MetricsSnapshot
from tre_common.registry import ClusterTopology, NodeSpec
from tre_controller.config import SafeScaleConfig
from tre_controller.loops.action_queue import ActionQueue, PlanBatch
from tre_controller.loops.decision_task import DecisionLoop, decision_task
from tre_controller.loops.metrics_task import SnapshotBox
from tre_controller.loops.plan_items import plan_item
from tre_controller.loops.safescale_task import _commands_to_actions
from tre_controller.loops.tick import LoopTickResult, _apply_safescale, upscale_of
from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, TauThresholds
from tre_controller.planning.planner import (
    ClusterView,
    HideAction,
    PlanConfig,
    RescueBasis,
    ScaleAction,
    ShrinkForSlotAction,
    TransferIntent,
    build_plan,
)
from tre_controller.loops.decision_task import run_rescue_tick
from tre_controller.planning.safescale import SafeScaleCommand, SafeScaleStateMachine
from tre_sm.allocator.slots import Binding, Slot

from plan_shim import PlanSM
from test_b8_observe_probes import _probe_snapshot
from test_loop_ticks import FakeQueue, _registry_with_models

TOPOLOGY = ClusterTopology(nodes=(NodeSpec(name="n", gpus=4, two_gpu_slots=((0, 1), (2, 3))),))
TOPOLOGY8 = ClusterTopology(nodes=(NodeSpec(name="n", gpus=8, two_gpu_slots=((0, 1), (2, 3), (4, 5), (6, 7))),))
CRIT = ModelState.CRITICAL, ModelRole.RECEIVER
LOW = ModelState.LOW, ModelRole.RECEIVER
IDLE = ModelState.IDLE, ModelRole.DONOR
HEALTHY = ModelState.HEALTHY, ModelRole.NEUTRAL


def _cls(model: str, state: ModelState, role: ModelRole, z: float | None) -> ModelClassification:
    return ModelClassification(
        model_name=model, state=state, role=role, Z_m=z, eta_m=None, trs=0.0, theta_m=1.0,
        tau=TauThresholds.from_control(), donor_tier="idle" if state == ModelState.IDLE else None,
    )


def _binding(serve_id: str, model: str, gpus: tuple[int, ...], awake: bool) -> Binding:
    return Binding(serve_id, model, Slot("n", gpus), awake=awake)


def _plan(bindings, classes, *, rescue_due=True, fairness_due=True, bases=None, topology=TOPOLOGY, **cfg):
    """One planner call on the bindings; ``x`` fills the other GPUs at its floor."""
    awake, bound = {}, {}
    for binding in bindings:
        awake[binding.model] = awake.get(binding.model, 0) + int(binding.awake)
        bound[binding.model] = bound.get(binding.model, 0) + 1
    return build_plan(
        model_contexts={
            model: {"routable_pods": awake[model], "awake_replicas": awake[model], "assigned_replicas": bound[model]}
            for model in bound
        },
        classifications=[_cls(model, *spec) for model, spec in classes.items()],
        model_replicas=bound,
        idle_gpus=0,
        cfg=PlanConfig(
            min_replicas_per_model=0,
            max_replicas_per_model=4,
            rescue_due=rescue_due,
            fairness_due=fairness_due,
            suppress_hot_proactive_probe=True,
            **{"min_replicas_by_model": {"x": 3}, **cfg},
        ),
        cluster_view=ClusterView(topology, tuple(bindings)),
        rescue_bases=bases,
    )


def _one_free_gpu():
    """GPU 0 free; ``crit`` and ``low`` both sleep there; ``x`` holds GPUs 1-3."""
    return [
        _binding("crit-0", "crit", (0,), False),
        _binding("low-0", "low", (0,), False),
        *(_binding(f"x-{gpu}", "x", (gpu,), True) for gpu in (1, 2, 3)),
    ]


def _ups(plan) -> list[tuple[str, int, tuple[str, ...]]]:
    return [
        (action.model, action.delta, action.pods)
        for action in plan.actions
        if isinstance(action, ScaleAction) and upscale_of(action) is not None
    ]


def _tp2(*donors: Binding):
    """CRITICAL ``tp2`` (TP2): awake on GPUs 4-5, sleeping on 0-1 and 2-3; ``x`` holds
    6-7 at its floor; ``donors`` (IDLE, floor 0) sit on GPUs 0-3. Returns the plan
    items the tick sends after SafeScale (scope ``all``): (kind, model, pods, hints,
    donors) per plan, and the planner events."""
    bindings = [
        _binding("tp2-4", "tp2", (4, 5), True),
        _binding("tp2-0", "tp2", (0, 1), False),
        _binding("tp2-2", "tp2", (2, 3), False),
        *(_binding(f"x-{gpu}", "x", (gpu,), True) for gpu in (6, 7)),
        *donors,
    ]
    models = {binding.model for binding in donors} - {"x"}
    classes = {"tp2": (*CRIT, 0.3), "x": (*HEALTHY, 1.1), **{model: (*IDLE, 0.0) for model in models}}
    plan = _plan(bindings, classes, topology=TOPOLOGY8, model_tp_sizes={"tp2": 2},
                 min_replicas_by_model={"x": 2, **{model: 0 for model in models}})
    out, _ = _apply_safescale(
        MetricsSnapshot(ts_ms=1, models={}, stale=False), tuple(plan.actions), plan.probe_upscale_plans,
        safescale=SafeScaleStateMachine(config=SafeScaleConfig()),
        cluster_view=ClusterView(TOPOLOGY8, tuple(bindings)), safescale_scope="all",
    )
    plans = []
    for action in out:
        items = []
        for member in (action.members if isinstance(action, PlanBatch) else (action,)):
            kind, item = plan_item(member)
            items.append((kind, item["model"], item.get("pods"), item.get("hints"),
                          [(d["model"], d["max"]) for d in item.get("donors", ())] or None))
        plans.append(items)
    return plans, plan.events


# ------------------------------------------------------------------ 1. one plan
SHRINK = ShrinkForSlotAction(
    donor="d", beneficiary="tp2", serve_id="d-0", slot=Slot("n", (0,)),
    reason="critical_same_slot_high_shrink", source_loop="rescue", receiver_pods=("tp2-0",),
)
VIEW = ClusterView(TOPOLOGY, (
    _binding("d-0", "d", (0,), True), _binding("tp2-0", "tp2", (0, 1), False), _binding("tp2-2", "tp2", (2, 3), False),
))


@pytest.mark.parametrize(("scope", "shrink"), [
    ("receiverless_shrink", SHRINK),
    ("off", SHRINK),
    ("all", replace(SHRINK, reason="critical_same_slot_idle_shrink", immediate=True)),  # IDLE donor
])
def test_a_same_slot_release_is_one_plan_of_the_donor_sleep_and_the_receiver_wake(scope, shrink):
    machine = None if scope == "off" else SafeScaleStateMachine(config=SafeScaleConfig())

    out, _ = _apply_safescale(MetricsSnapshot(ts_ms=1, models={}, stale=False), (shrink,), {},
                              safescale=machine, cluster_view=VIEW, safescale_scope=scope)

    assert machine is None or machine.active_probe("d") is None  # no probe
    [plan] = out
    assert isinstance(plan, PlanBatch)
    items = [plan_item(member) for member in plan.members]
    assert [(kind, item["model"], item.get("pods"), item.get("hints")) for kind, item in items] == [
        ("sleep", "d", ["d-0"], None),
        ("wake", "tp2", None, ["tp2-0"]),
    ]


def _release(donor="d", receiver="tp2"):
    return PlanBatch((
        ScaleAction(donor, -1, "critical_same_slot_idle_shrink", "rescue", donor=donor, pods=(f"{donor}-0",),
                    sleep_path="urgent"),
        ScaleAction(receiver, 1, "critical_same_slot_idle_shrink", "rescue", receiver=receiver,
                    pods=(f"{receiver}-0",), hint=True),
    ))


def test_a_decision_with_a_same_slot_release_is_one_sm_plan():
    sm = PlanSM({"d": 2, "tp2": 1, "r": 1})
    queue = ActionQueue(sm, now_ms=lambda: 5_000)
    queue.submit((_release(), ScaleAction("r", 1, "critical_idle_capacity", "rescue")))
    asyncio.run(queue.drain_once())

    [plan] = sm.plans
    assert [(i["model"], i.get("pods")) for i in plan["sleeps"]] == [("d", ["d-0"])]
    assert [(i["model"], i.get("hints")) for i in plan["wakes"]] == [("tp2", ["tp2-0"]), ("r", None)]


def test_a_same_slot_release_is_never_split_across_plans():
    sm = PlanSM({"d": 3, "tp2": 1, "r": 1})
    queue = ActionQueue(sm, now_ms=lambda: 5_000)
    # The relay uses d first: the release starts the next plan, whole.
    queue.submit((TransferIntent("d", "r", 1, "critical_donor_immediate", "rescue"), _release()))
    asyncio.run(queue.drain_once())

    first, second = sm.plans
    assert [i["model"] for i in first["wakes"]] == ["r"] and first["sleeps"] == []
    assert [i["model"] for i in second["sleeps"]] == ["d"] and [i["model"] for i in second["wakes"]] == ["tp2"]


def test_with_safescale_the_release_is_a_probe_and_its_commit_is_that_one_plan():
    machine = SafeScaleStateMachine(config=SafeScaleConfig())
    snapshot = MetricsSnapshot(ts_ms=1, models={}, stale=False)
    out, _ = _apply_safescale(snapshot, (SHRINK,), {}, safescale=machine, cluster_view=VIEW, safescale_scope="all")
    assert out and all(isinstance(action, HideAction) for action in out)  # no bare sleep

    [commit] = _commands_to_actions(
        (SafeScaleCommand("scale_down", "d", pods=("d-0",)), SafeScaleCommand("scale_up", "tp2", delta=1)),
        cluster_view=VIEW,
    )
    assert commit.donor_sleep().pods == ("d-0",)
    assert [plan_item(target)[1].get("hints") for target in commit.upscales] == [["tp2-0"]]


# ------------------------------------------------------------------ 4. TP2 supply
def _wakes(plans, model):
    return [item for items in plans for item in items if item[0] == "wake" and item[1] == model]


def test_a_free_gpu_and_an_idle_donor_pod_give_the_tp2_receiver_its_slot_in_one_plan():
    plans, _ = _tp2(_binding("i-0", "i", (0,), True), _binding("i-2", "i", (2,), True), _binding("i-3", "i", (3,), True))

    # One plan (no probe, scope all): the IDLE pod beside the free GPU sleeps, tp2 wakes there.
    assert [("sleep", "i", ["i-0"], None, None), ("wake", "tp2", None, ["tp2-0"], None)] in plans


def test_two_pods_of_one_idle_donor_relay_the_tp2_slot():
    plans, events = _tp2(*(_binding(f"i-{gpu}", "i", (gpu,), True) for gpu in (0, 1)),
                         *(_binding(f"x-{gpu}", "x", (gpu,), True) for gpu in (2, 3)))

    assert _wakes(plans, "tp2") == [("wake", "tp2", None, None, [("i", 2)])]  # both pods of the slot
    assert not any(event.startswith("capacity_blocked") for event in events)


def test_a_mixed_donor_slot_gives_the_tp2_receiver_nothing():
    plans, events = _tp2(_binding("a-0", "a", (0,), True), _binding("b-1", "b", (1,), True),
                         _binding("a-2", "a", (2,), True), _binding("b-3", "b", (3,), True))

    assert _wakes(plans, "tp2") == []
    assert not [item for items in plans for item in items if item[0] == "sleep" and item[2]]  # no pod named
    assert "capacity_blocked:tp2" in events


# ------------------------------------------------------------------ 2. receiver order
@pytest.mark.parametrize("z_a, z_b, winner", [(0.5, 0.2, "b"), (0.2, 0.5, "a")])
def test_the_lower_z_wins_the_free_gpu_whatever_the_registry_order(z_a, z_b, winner):
    bindings = [
        _binding("low-0", "low", (0,), False),
        _binding("a-0", "a", (0,), False),
        _binding("b-0", "b", (0,), False),
        *(_binding(f"x-{gpu}", "x", (gpu,), True) for gpu in (1, 2, 3)),
    ]
    # Registry order: low, a, b - LOW (lower Z than neither) is listed first.
    classes = {"low": (*LOW, 0.1), "a": (*CRIT, z_a), "b": (*CRIT, z_b),
               "x": (ModelState.HEALTHY, ModelRole.NEUTRAL, 1.1)}

    plan = _plan(bindings, classes)

    assert _ups(plan) == [(winner, 1, (f"{winner}-0",))]


# ------------------------------------------------------------------ 3. one step
def test_a_receiver_gets_one_step_per_plan_and_none_before_the_window_reflects_it():
    def bindings(awake_count):
        return [_binding(f"crit-{gpu}", "crit", (gpu,), gpu < awake_count) for gpu in range(4)]

    classes = {"crit": (*CRIT, 0.05)}  # deep CRITICAL, three free GPUs it sleeps on

    first = _plan(bindings(1), classes)
    assert [(model, delta) for model, delta, _pods in _ups(first)] == [("crit", 1)]
    # The target (1 -> 2) is issued; the window still describes 1 replica.
    held = _plan(bindings(1), classes, bases={"crit": RescueBasis(base=1, covered=2)})
    assert _ups(held) == [] and "rescue_target_hold:crit:desired=2:covered=2" in held.events
    # The window moved (2 replicas): the next single step.
    reflected = _plan(bindings(2), classes)
    assert [(model, delta) for model, delta, _pods in _ups(reflected)] == [("crit", 1)]


def test_a_step_promised_by_a_running_probe_is_not_planned_again():
    """Review P2-3: a donor probe for the CRITICAL receiver is its step (the commit wakes
    the replica); while the probe runs, the next calls hold the receiver."""
    machine = SafeScaleStateMachine(config=SafeScaleConfig())
    registry = _registry_with_models("critical", "donor")
    queue = FakeQueue()

    first = run_rescue_tick(_probe_snapshot(), queue=queue, registry=registry, safescale=machine)
    assert machine.active_probe("donor") is not None
    assert "rescue_target:critical:n=2:z=0.0183:desired=3:covered=2:planned=1" in first.events

    later = run_rescue_tick(replace(_probe_snapshot(), ts_ms=15_000), queue=queue, registry=registry,
                            safescale=machine, active_probe_models={"donor"})
    assert later.actions == () and "rescue_target_hold:critical:desired=3:covered=3" in later.events


# ------------------------------------------------------------------ 5. one ledger
def test_rescue_and_rebalance_of_one_call_never_assign_one_free_gpu_twice():
    classes = {"crit": (*CRIT, 0.3), "low": (*LOW, 0.9), "x": (ModelState.HEALTHY, ModelRole.NEUTRAL, 1.1)}

    both = _plan(_one_free_gpu(), classes)

    assert _ups(both) == [("crit", 1, ("crit-0",))]
    # Alone, the rebalance section would have woken LOW on that GPU.
    alone = _plan(_one_free_gpu(), classes, rescue_due=False)
    assert _ups(alone) == [("low", 1, ("low-0",))]


def test_a_freed_tp2_half_slot_goes_to_the_critical_receiver_not_to_a_low_one():
    """The 10-28 defect: GPU 1 free, an IDLE pod on GPU 0; the CRITICAL TP2 receiver needs
    the pair, a LOW 7b sleeps on GPU 1. One call: the pair goes to the TP2 receiver."""
    bindings = [
        _binding("i-0", "i", (0,), True),
        _binding("tp2-0", "tp2", (0, 1), False),
        _binding("low-1", "low", (1,), False),
        *(_binding(f"x-{gpu}", "x", (gpu,), True) for gpu in (2, 3)),
    ]
    classes = {"tp2": (*CRIT, 0.3), "low": (*LOW, 0.9), "i": (*IDLE, 0.0), "x": (*HEALTHY, 1.1)}
    cfg = dict(model_tp_sizes={"tp2": 2, "low": 1, "i": 1, "x": 1}, min_replicas_by_model={"x": 2, "i": 0})

    both = _plan(bindings, classes, **cfg)
    assert [(a.donor, a.beneficiary) for a in both.actions if isinstance(a, ShrinkForSlotAction)] == [("i", "tp2")]
    assert _ups(both) == []  # GPU 1 is the slot's: the LOW receiver gets nothing
    # The rebalance section alone would have woken LOW there.
    assert _ups(_plan(bindings, classes, rescue_due=False, **cfg)) == [("low", 1, ("low-1",))]


def test_a_gpu_claimed_by_an_earlier_wake_is_no_free_slot_mate():
    """Review P1-1: GPU 1 is free in the view but this plan wakes the CRITICAL 7b there;
    the CRITICAL TP2 receiver must not sleep the pod on GPU 0 for a slot it cannot get."""
    bindings = [
        _binding("d-0", "d", (0,), True),
        _binding("s7-1", "s7", (1,), False),
        _binding("s7-2", "s7", (2,), True),
        _binding("tp2-0", "tp2", (0, 1), False),
        _binding("x-3", "x", (3,), True),
    ]
    classes = {"tp2": (*CRIT, 0.1), "s7": (*CRIT, 0.5), "d": (*IDLE, 0.0), "x": (*HEALTHY, 1.1)}

    plan = _plan(bindings, classes, model_tp_sizes={"tp2": 2}, min_replicas_by_model={"x": 1, "d": 0})

    assert _ups(plan) == [("s7", 1, ("s7-1",))]
    assert not [a for a in plan.actions if isinstance(a, ShrinkForSlotAction)]
    assert "capacity_blocked:tp2" in plan.events


@pytest.mark.parametrize("slow_off", [False, True])
def test_with_the_slow_loop_off_a_low_receiver_gets_nothing(slow_off):
    bindings = [b for b in _one_free_gpu() if b.model != "crit"]
    classes = {"low": (*LOW, 0.9), "x": (ModelState.HEALTHY, ModelRole.NEUTRAL, 1.1)}
    seen = []

    def tick(snapshot, *, rescue_due, fairness_due, **_kwargs):
        seen.append(fairness_due)
        plan = _plan(bindings, classes, rescue_due=rescue_due, fairness_due=fairness_due)
        return LoopTickResult(submitted=len(plan.actions), actions=tuple(plan.actions))

    cfg = SimpleNamespace(rescue_interval_s=5.0, fairness_interval_s=10.0, ablation_disable_slow_loop=slow_off)
    loop = DecisionLoop(queue=SimpleNamespace(), registry=None, cfg=cfg, tick=tick)
    results = [loop.on_snapshot(MetricsSnapshot(ts_ms=t, models={}, stale=False)) for t in range(0, 30_001, 5_000)]

    low_ups = [a for r in results if r is not None for a in r.actions if upscale_of(a) is not None]
    if slow_off:
        assert seen and not any(seen) and low_ups == []
    else:
        assert seen == [True, False, True, False, True, False, True] and low_ups


# ------------------------------------------------------------------ 6. stale snapshots
def test_a_stale_snapshot_decides_nothing_and_alerts_once(caplog):
    calls = []
    cfg = SimpleNamespace(rescue_interval_s=5.0, fairness_interval_s=10.0)
    loop = DecisionLoop(queue=SimpleNamespace(), registry=None, cfg=cfg,
                        tick=lambda snapshot, **kw: calls.append(snapshot.ts_ms) or LoopTickResult(submitted=0))

    with caplog.at_level(logging.INFO, logger="tre_controller.decision_loop"):
        first = loop.on_snapshot(MetricsSnapshot(ts_ms=2_000, models={}, stale=True))
        loop.on_snapshot(MetricsSnapshot(ts_ms=2_000, models={}, stale=True))
        loop.on_snapshot(MetricsSnapshot(ts_ms=4_000, models={}, stale=False))

    assert first.events == ("snapshot_stale",) and calls == [4_000]
    logged = [record.getMessage() for record in caplog.records]
    assert sum("decision_snapshot_stale" in line for line in logged) == 1
    assert sum("decision_snapshot_recovered" in line for line in logged) == 1


def test_no_publish_for_the_silence_limit_alerts_without_a_decision():
    class _Stop(Exception):
        pass

    class _Loop:
        silences: list[float] = []

        def on_snapshot(self, snapshot):
            raise AssertionError("no snapshot was published")

        def on_silence(self, silent_s):
            self.silences.append(silent_s)
            raise _Stop

    with pytest.raises(_Stop):
        asyncio.run(decision_task(SnapshotBox(), _Loop(), silence_alert_s=0.0))
    assert _Loop.silences == [0.0]

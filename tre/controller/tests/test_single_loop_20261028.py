"""Single decision loop (design docs/design/20261028-single-loop.md): the invariants.

2. Order: two CRITICAL receivers compete for one free GPU - the lower Z wins whatever
   the registry order; a LOW receiver is never planned before a CRITICAL one.
5. One ledger: the rescue and the rebalance section of one planner call never assign
   one free GPU twice; with TRE_ABLATION_DISABLE_SLOW_LOOP a LOW receiver gets nothing.
6. A stale snapshot (or no publish at all) gives no decision and an alert. (The same
   observation stream at a 1 s and a 2 s cadence: test_period_invariance_20261025.)
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from tre_common.metrics_schema import MetricsSnapshot
from tre_common.registry import ClusterTopology, NodeSpec
from tre_controller.loops.decision_task import DecisionLoop, decision_task
from tre_controller.loops.metrics_task import SnapshotBox
from tre_controller.loops.tick import LoopTickResult, upscale_of
from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, TauThresholds
from tre_controller.planning.planner import ClusterView, PlanConfig, ScaleAction, build_plan
from tre_sm.allocator.slots import Binding, Slot

TOPOLOGY = ClusterTopology(nodes=(NodeSpec(name="n", gpus=4, two_gpu_slots=((0, 1), (2, 3))),))
CRIT = ModelState.CRITICAL, ModelRole.RECEIVER
LOW = ModelState.LOW, ModelRole.RECEIVER


def _cls(model: str, state: ModelState, role: ModelRole, z: float | None) -> ModelClassification:
    return ModelClassification(
        model_name=model, state=state, role=role, Z_m=z, eta_m=None, trs=0.0, theta_m=1.0,
        tau=TauThresholds.from_control(), donor_tier="idle" if state == ModelState.IDLE else None,
    )


def _binding(serve_id: str, model: str, gpus: tuple[int, ...], awake: bool) -> Binding:
    return Binding(serve_id, model, Slot("n", gpus), awake=awake)


def _plan(bindings, classes, *, rescue_due=True, fairness_due=True, **cfg):
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
            min_replicas_by_model={"x": 3},
            suppress_hot_proactive_probe=True,
            **cfg,
        ),
        cluster_view=ClusterView(TOPOLOGY, tuple(bindings)),
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


# ------------------------------------------------------------------ 5. one ledger
def test_rescue_and_rebalance_of_one_call_never_assign_one_free_gpu_twice():
    classes = {"crit": (*CRIT, 0.3), "low": (*LOW, 0.9), "x": (ModelState.HEALTHY, ModelRole.NEUTRAL, 1.1)}

    both = _plan(_one_free_gpu(), classes)

    assert _ups(both) == [("crit", 1, ("crit-0",))]
    # Alone, the rebalance section would have woken LOW on that GPU.
    alone = _plan(_one_free_gpu(), classes, rescue_due=False)
    assert _ups(alone) == [("low", 1, ("low-0",))]


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

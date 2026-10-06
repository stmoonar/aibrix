"""H1 (2026-10-06, docs baselines-methodology "10-06 晚 TRE 决策延迟修复"), controller side.

* The rescue loop ticks right after a snapshot is published (not on its free-running
  5 s phase); ``rescue_interval_s`` is only the fallback when nothing is published.
* A LOW receiver with a sleeping binding on a free GPU is woken by the rescue loop in
  the same grid; donor-taking steps for LOW stay with the fairness loop.
* No model gets two scale-ups for one grid across the two loops."""
from __future__ import annotations

import asyncio

from tre_common.metrics_schema import MetricsSnapshot
from tre_controller.loops import rescue_task as rescue_mod
from tre_controller.loops.metrics_task import SnapshotBox
from tre_controller.loops.tick import LoopTickResult
from tre_controller.planning.classify import ModelRole, ModelState
from tre_controller.planning.planner import ClusterView, PlanConfig, ScaleAction, TransferIntent, build_plan
from tre_sm.allocator.slots import Binding, Slot

from test_planner_slot_occupancy import TOPOLOGY, _cls, _e1_bindings

LOW_MODEL = "dsllama-8b"


# ----------------------------------------------------------------- (a) tick on publish


def test_rescue_decides_on_publish_without_waiting_for_the_interval(monkeypatch) -> None:
    clock = {"ms": 0}
    decided: list[tuple[int, int]] = []  # (snapshot ts_ms, fake clock at decision)
    fallback_sleeps: list[float] = []

    def fake_tick(snapshot, **_kwargs):
        decided.append((snapshot.ts_ms, clock["ms"]))
        return LoopTickResult(submitted=0)

    async def never_elapses(seconds: float) -> None:
        # The fallback interval; the fake clock never reaches it in this test.
        fallback_sleeps.append(seconds)
        await asyncio.Event().wait()

    monkeypatch.setattr(rescue_mod, "run_rescue_tick", fake_tick)
    box = SnapshotBox(MetricsSnapshot(ts_ms=10_000, models={}, stale=False))
    cfg = type("Cfg", (), {"rescue_interval_s": 5.0})()

    async def scenario() -> None:
        task = asyncio.ensure_future(
            rescue_mod.rescue_task(box, queue=object(), registry=object(), cfg=cfg, sleep=never_elapses)
        )
        for _ in range(5):
            await asyncio.sleep(0)
        clock["ms"] = 12_300  # boundary 20 s + learned offset, minus 10 s of grid
        box.set(MetricsSnapshot(ts_ms=20_000, models={}, stale=False))
        for _ in range(5):
            await asyncio.sleep(0)
        clock["ms"] = 22_300
        box.set(MetricsSnapshot(ts_ms=30_000, models={}, stale=False))
        for _ in range(5):
            await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())

    # One decision per publish, each at the publish time (epsilon = 0 on the fake clock),
    # although the 5 s fallback never elapsed.
    assert decided == [(10_000, 0), (20_000, 12_300), (30_000, 22_300)]
    assert fallback_sleeps and all(s == 5.0 for s in fallback_sleeps)


# ------------------------------------------------- (b) LOW + free sleeping GPU via rescue


def _low_plan(bindings, *, rescue_due: bool, fairness_due: bool, **kwargs):
    awake_7b = sum(1 for b in bindings if b.model == "dsqwen-7b" and b.awake)
    return build_plan(
        model_contexts={
            "dsqwen-7b": {"routable_pods": awake_7b, "assigned_replicas": 8},
            LOW_MODEL: {"routable_pods": 1, "assigned_replicas": 8},
        },
        classifications=[
            _cls(LOW_MODEL, ModelState.LOW, ModelRole.RECEIVER, 0.9),
            _cls("dsqwen-7b", ModelState.HIGH, ModelRole.DONOR, 2.0, "surplus"),
        ],
        model_replicas={"dsqwen-7b": 8, LOW_MODEL: 8},
        idle_gpus=0,
        cfg=PlanConfig(
            min_replicas_per_model=1,
            max_replicas_per_model=4,
            rescue_due=rescue_due,
            fairness_due=fairness_due,
            suppress_hot_proactive_probe=True,
        ),
        cluster_view=ClusterView(TOPOLOGY, bindings),
        **kwargs,
    )


def _with_free_sleeping_gpu() -> tuple[Binding, ...]:
    # 7b awake on every GPU except node10:2, where an 8b binding sleeps; one 8b awake
    # elsewhere is its serving replica (routable 1).
    bindings = [
        b if b.serve_id != "8b-0" else Binding("8b-0", LOW_MODEL, b.slot, awake=True)
        for b in _e1_bindings(free_slot=("node10", 2))
        if b.serve_id != "7b-0"
    ]
    return tuple(bindings)


def _upscales(plan, model: str):
    return [a for a in plan.actions if isinstance(a, ScaleAction) and a.model == model and a.delta > 0]


def test_low_receiver_with_free_sleeping_gpu_gets_plus_one_from_rescue() -> None:
    plan = _low_plan(_with_free_sleeping_gpu(), rescue_due=True, fairness_due=False)

    [wake] = _upscales(plan, LOW_MODEL)
    assert (wake.delta, wake.source_loop, wake.reason) == (1, "rescue", "low_rescue_sleeping_capacity")
    assert wake.pods == ("8b-6",)  # the binding on the free GPU (node10:2)

    # No free GPU: the rescue loop takes no donor for a LOW receiver (fairness does).
    blocked = _low_plan(_e1_bindings(), rescue_due=True, fairness_due=False)
    assert not _upscales(blocked, LOW_MODEL)
    assert not [a for a in blocked.actions if isinstance(a, TransferIntent)]


def test_low_receiver_is_not_scaled_twice_for_one_grid() -> None:
    bindings = _with_free_sleeping_gpu()
    # One plan with both sections: one +1, from the rescue section only.
    both = _low_plan(bindings, rescue_due=True, fairness_due=True)
    assert [(a.delta, a.source_loop) for a in _upscales(both, LOW_MODEL)] == [(1, "rescue")]
    assert not [a for a in both.actions if isinstance(a, TransferIntent)]

    # The fairness tick on the same snapshot after the rescue wake was submitted (in
    # flight), and after it completed but before a fleet view shows it (O1 view gate).
    for kwargs in ({"inflight_models": {LOW_MODEL}}, {"view_pending": {LOW_MODEL: "up"}}):
        fairness = _low_plan(bindings, rescue_due=False, fairness_due=True, **kwargs)
        assert not _upscales(fairness, LOW_MODEL), kwargs
        assert not [a for a in fairness.actions if isinstance(a, TransferIntent)], kwargs

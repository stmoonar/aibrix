"""Controller ablation switches (2026-10-08).

TRE_ABLATION_DISABLE_SAFESCALE: SafeScale off = immediate release (as before 05f489f1 /
v1). The planner loops get no SafeScale, every shrink that would run as a probe takes
the urgent donor path, and probes an earlier run left in Redis are rolled back once.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import tre_controller.app as app
from tre_common.metrics_schema import MetricsSnapshot
from tre_controller.app import build_controller_task_specs
from tre_controller.config import SafeScaleConfig
from tre_controller.loops.rescue_task import run_rescue_tick
from tre_controller.loops.safescale_task import rollback_left_probes_task
from tre_controller.loops.tick import _apply_safescale
from tre_controller.planning.planner import ScaleAction, ShrinkForSlotAction, TransferIntent, UnhideAction
from tre_controller.planning.safescale import SafeScaleStateMachine
from tre_controller.store.state_store import ControllerStateStore

from test_b8_observe_probes import _probe_snapshot
from test_controller_app import _cfg, _deps
from test_loop_ticks import FakeQueue, _metrics_with_pods, _registry_with_models
from test_safescale_commit import FakeRedis


def _capture_planner_tasks(monkeypatch) -> dict[str, dict]:
    seen: dict[str, dict] = {}
    monkeypatch.setattr(app, "rescue_task", lambda *args, **kwargs: seen.setdefault("rescue", kwargs))
    monkeypatch.setattr(app, "fairness_task", lambda *args, **kwargs: seen.setdefault("fairness", kwargs))
    return seen


# ------------------------------------------------------------- SafeScale off
def test_safescale_off_planner_loops_get_no_safescale_and_no_safescale_loop_runs(monkeypatch):
    seen = _capture_planner_tasks(monkeypatch)
    specs = {spec.name: spec for spec in build_controller_task_specs(_deps(), _cfg(ablation_disable_safescale=True))}

    assert "safescale" not in specs and "safescale_leftover_rollback" in specs
    specs["rescue"].factory()
    specs["fairness"].factory()
    assert seen["rescue"]["safescale"] is None and seen["fairness"]["safescale"] is None


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


def test_proactive_and_same_slot_shrinks_are_urgent_and_one_per_donor():
    proactive = ScaleAction("high", -1, "high_proactive_safescale", "rescue", requires_safescale=True, donor="high")
    same_slot = ShrinkForSlotAction(
        donor="d", beneficiary="tp2", serve_id="d-0", slot=None,
        reason="critical_same_slot_high_shrink", source_loop="rescue",
    )
    second = ScaleAction(
        "d", -1, "low_fairness_high_donor_safescale", "fairness", requires_safescale=True, donor="d", receiver="low"
    )
    wake = ScaleAction("other", 1, "critical_idle_capacity", "rescue")

    out, events = _apply_safescale(
        MetricsSnapshot(ts_ms=1, models={}, stale=False), (proactive, same_slot, second, wake), {}, safescale=None
    )

    assert out == (
        ScaleAction("high", -1, "high_proactive_safescale_nosafescale", "rescue", donor="high", sleep_path="urgent"),
        ScaleAction(
            "d", -1, "critical_same_slot_high_shrink_nosafescale", "rescue",
            donor="d", receiver="tp2", pods=("d-0",), sleep_path="urgent",
        ),
        ScaleAction("tp2", 1, "critical_same_slot_high_shrink_nosafescale", "rescue", receiver="tp2"),
        wake,
    )
    assert events == ("safescale_probe_skipped:d:released_this_tick",)


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

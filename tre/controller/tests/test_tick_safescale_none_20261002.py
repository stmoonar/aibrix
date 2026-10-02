"""_apply_safescale without a SafeScale controller (2026-10-02)."""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from tre_common.metrics_schema import MetricsSnapshot
from tre_controller.loops.tick import _apply_safescale
from tre_controller.planning.planner import HideAction, ScaleAction, ShrinkForSlotAction, TransferIntent


def _snap() -> MetricsSnapshot:
    return MetricsSnapshot(ts_ms=1_000, models={}, stale=False)


def _shrink_for_slot() -> ShrinkForSlotAction:
    return ShrinkForSlotAction(
        donor="donor", beneficiary="recv", serve_id="donor-0", slot=None,
        reason="critical_same_slot_high_shrink", source_loop="rescue",
    )


def test_none_safescale_drops_probe_shrinks_and_keeps_the_rest():
    drops = (
        ScaleAction("a", -1, "critical_middle_zone_safescale", "rescue", requires_safescale=True),
        ScaleAction("b", -1, "high_proactive_safescale", "fairness", requires_safescale=True),
        ScaleAction("c", -1, "low_fairness_middle_zone_safescale", "fairness", requires_safescale=True),
        _shrink_for_slot(),
    )
    keeps = (
        ScaleAction("recv", 2, "critical_idle_capacity", "rescue"),
        ScaleAction("donor", -1, "critical_donor_immediate", "rescue", sleep_path="urgent"),
        HideAction("x", ("x-0",), "r", "rescue"),
    )
    out, events = _apply_safescale(_snap(), drops + keeps, {}, safescale=None)
    assert out == keeps
    assert events == (
        "safescale_probe_skipped:a:safescale_unavailable",
        "safescale_probe_skipped:b:safescale_unavailable",
        "safescale_probe_skipped:c:safescale_unavailable",
        "safescale_probe_skipped:donor:safescale_unavailable",
    )


def test_run_planner_tick_keeps_the_immediate_donor_transfer_and_drops_probe_shrinks(monkeypatch):
    """End to end through run_planner_tick(safescale=None): the planner output (an
    urgent donor/receiver pair + a probe-requiring shrink) is cut to the pair."""
    from tre_controller.loops import tick as tick_mod
    from test_loop_ticks import FakeQueue, _metrics, _registry

    # 2026-10-02: the immediate relay is one TransferIntent (a count, no pods).
    pair = (TransferIntent("donor", "critical", 1, "critical_donor_immediate", "rescue"),)
    probe = ScaleAction("other", -1, "high_proactive_safescale", "rescue", requires_safescale=True)
    real = tick_mod.build_plan

    def fake_build_plan(**kwargs):
        return dataclasses.replace(real(**kwargs), actions=(probe,) + pair)

    monkeypatch.setattr(tick_mod, "build_plan", fake_build_plan)
    queue = FakeQueue()
    snapshot = MetricsSnapshot(
        ts_ms=1, stale=False,
        models={"critical": _metrics("critical", generation=50.0, waiting=10.0, running=1.0, assigned=2)},
    )
    result = tick_mod.run_planner_tick(
        snapshot, queue=queue, registry=_registry(), rescue_due=True, fairness_due=False, safescale=None
    )
    assert queue.submitted == [pair]
    assert "safescale_probe_skipped:other:safescale_unavailable" in result.events


class _Client:
    def __init__(self):
        self.calls = []

    async def scale_model(self, model, delta, **kw):
        self.calls.append(("scale", model, delta, kw))
        return {"ok": True}

    async def set_binding_power(self, serve_id, *, awake, **kw):
        self.calls.append(("power", serve_id, awake, kw))
        return {"ok": True}


def test_a_pathless_scale_down_sends_nothing():
    # (2026-10-02: the former donor/receiver pair is a TransferIntent, which always
    # carries its sleep path; a path-less shrink is still refused before the SM.)
    from tre_controller.loops.action_queue import ActionQueue

    client = _Client()
    queue = ActionQueue(client)
    donor = ScaleAction("donor", -1, "critical_same_slot_high_shrink", "rescue", pods=("donor-0",))
    queue.submit((donor,))
    [result] = asyncio.run(queue.drain_once())
    assert client.calls == []
    assert result.ok is False and result.error.startswith("sleep_path_refused")
    assert result.retriable is False
    assert queue.stats()["sleep_path_refused_total"] == 1
    assert queue.stats()["dispatch_exceptions_total"] == 0
    # O1 must not see a routable change that never happened.
    assert queue.routable_changes() == {}

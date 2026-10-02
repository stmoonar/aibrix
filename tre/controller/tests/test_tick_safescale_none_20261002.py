"""_apply_safescale without a SafeScale controller (2026-10-02)."""

from __future__ import annotations

import json
import logging

from tre_common.metrics_schema import MetricsSnapshot
from tre_controller.loops.tick import _apply_safescale
from tre_controller.planning.planner import HideAction, ScaleAction, ShrinkForSlotAction


def _snap() -> MetricsSnapshot:
    return MetricsSnapshot(ts_ms=1_000, models={}, stale=False)


def _shrink_for_slot() -> ShrinkForSlotAction:
    return ShrinkForSlotAction(
        donor="donor", beneficiary="recv", serve_id="donor-0", slot=None,
        reason="critical_same_slot_high_shrink", source_loop="rescue",
    )


def test_none_safescale_drops_probe_shrinks_and_keeps_the_rest(caplog):
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
    with caplog.at_level(logging.WARNING, logger="tre_controller.tick"):
        out, events = _apply_safescale(_snap(), drops + keeps, {}, safescale=None)
    assert out == keeps and events == ()
    logged = [json.loads(r.getMessage()) for r in caplog.records]
    assert [e["event"] for e in logged] == ["safescale_unavailable_shrink_dropped"] * 4
    assert [(e["model"], e["reason"]) for e in logged] == [
        ("a", "critical_middle_zone_safescale"),
        ("b", "high_proactive_safescale"),
        ("c", "low_fairness_middle_zone_safescale"),
        ("donor", "critical_same_slot_high_shrink"),
    ]

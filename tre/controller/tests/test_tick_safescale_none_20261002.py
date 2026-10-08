"""A path-less scale-down never reaches the SM (2026-10-02). The tick without a
SafeScale controller releases probe shrinks immediately since 2026-10-08
(test_ablation_switches_20261008.py)."""

from __future__ import annotations

import asyncio

from tre_controller.planning.planner import ScaleAction


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

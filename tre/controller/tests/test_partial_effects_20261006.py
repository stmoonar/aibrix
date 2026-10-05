"""Review P3 (2026-10-06): a call that changed part of what it asked for is recorded
as a change - a multi-pod binding-power call that failed after some pods changed, and
a relay partly covered by the pods a SafeScale probe preemption gives back."""
from __future__ import annotations

import asyncio

from tre_controller.loops.action_queue import ActionQueue
from tre_controller.planning.planner import ScaleAction
from tre_controller.sm_client import ServiceManagerError

from test_action_queue_review3 import ScriptedSM, _calls

FLOOR_REFUSAL = ServiceManagerError(
    "HTTP 409", status=409, body={"error": "floor_violation", "floor": {"model": "7b", "floor": 1}}
).result()


def test_a_binding_power_call_that_failed_after_a_pod_slept_is_recorded_as_a_change():
    sm = ScriptedSM(results={"7b-1": [FLOOR_REFUSAL]})
    queue = ActionQueue(sm, now_ms=lambda: 4_000)
    queue.submit((ScaleAction("7b", -2, "high_proactive", "rescue", pods=("7b-0", "7b-1"), sleep_path="scale_down"),))
    [result] = asyncio.run(queue.drain_once())
    assert _calls(sm) == [("7b-0", "sleep"), ("7b-1", "sleep")]
    assert not result.ok and result.changed == ("7b-0",) and not result.not_executed
    # 7b-0 slept: the model's routable set changed - stamped and recorded.
    assert queue.view_changes() == {"7b": (4_000, "down")}
    assert queue.last_actions() == {"7b": (4_000, "down")}


def test_a_binding_power_call_refused_at_its_first_pod_changed_nothing():
    sm = ScriptedSM(results={"7b-0": [FLOOR_REFUSAL]})
    queue = ActionQueue(sm, now_ms=lambda: 4_000)
    queue.submit((ScaleAction("7b", -2, "high_proactive", "rescue", pods=("7b-0", "7b-1"), sleep_path="scale_down"),))
    [result] = asyncio.run(queue.drain_once())
    assert (result.ok, result.changed, result.not_executed) == (False, (), True)
    assert queue.view_changes() == {} and queue.last_actions() == {}


def test_a_relay_partly_covered_by_restored_probe_pods_is_shrunk():
    from test_c1_deficit_scaleup_20261001 import _Preempting, _snapshot
    from tre_controller.loops.tick import _apply_safescale
    from tre_controller.planning.planner import TransferIntent

    relay = TransferIntent("7b", "r", 3, "critical_donor_immediate", "rescue")
    [shrunk], events = _apply_safescale(_snapshot(5_000), (relay,), {}, safescale=_Preempting(restored=1))
    assert (shrunk.count, shrunk.pairs) == (2, 2)  # one receiver replica came back from the probe
    assert "safescale_probe_preempted:r:restored=1:up_needed=2" in events
    # A TP receiver taking two single-GPU donors per pair shrinks by whole pairs.
    tp = TransferIntent("7b", "r", 4, "critical_donor_immediate", "rescue", pairs=2)
    [half], _ = _apply_safescale(_snapshot(5_000), (tp,), {}, safescale=_Preempting(restored=1))
    assert (half.count, half.pairs) == (2, 1)

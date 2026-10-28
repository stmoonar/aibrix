"""Review P3 (2026-10-06): a call that changed part of what it asked for is recorded
as a change - a multi-pod binding-power call that failed after some pods changed, and
a relay partly covered by the pods a SafeScale probe preemption gives back."""
from __future__ import annotations

import asyncio

from tre_controller.planning.planner import ScaleAction
from tre_controller.sm_client import ServiceManagerError

from test_action_queue_review3 import ScriptedSM, _calls
from plan_shim import LegacyFakeQueue as ActionQueue  # pre-plan fakes behind POST /v2/plans

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


def test_relay_pairs_covered_by_restored_probe_pods_are_dropped():
    from test_c1_deficit_scaleup_20261001 import _Preempting, _snapshot
    from tre_common.gpu_placement import GpuBlock
    from tre_controller.loops.tick import _apply_safescale
    from tre_controller.planning.planner import BlockSupplyAction

    # 2026-10-28 (sm-plans section 10): one named transfer per pair.
    relays = tuple(
        BlockSupplyAction(receiver="r", block=GpuBlock("n", (gpu,)), receiver_pods=(f"r-{gpu}",),
                          sleeps=(("7b", f"7b-{gpu}"),), reason="critical_donor_immediate", source_loop="rescue")
        for gpu in range(3)
    )
    kept, events = _apply_safescale(_snapshot(5_000), relays, {}, safescale=_Preempting(restored=1))
    assert len(kept) == 2  # one receiver replica came back from the probe: one pair dropped
    assert "block_supply_covered_by_preemption:r" in events

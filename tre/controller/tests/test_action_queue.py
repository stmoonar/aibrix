from __future__ import annotations

import asyncio

from tre_controller.loops.action_queue import ActionQueue, DispatchResult, SubmitResult
from tre_controller.planning.planner import DefragAction, HideAction, ScaleAction, UnhideAction
from tre_sm.allocator.slots import Migration, Slot


class FakeServiceManagerClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def scale_model(self, model: str, delta: int) -> dict:
        self.calls.append(("scale", model, delta))
        return {"ok": True}

    async def set_routable(self, model: str, hidden_pods: tuple[str, ...]) -> dict:
        self.calls.append(("routable", model, hidden_pods))
        return {"ok": True}

    async def defrag(self, migrations: tuple[Migration, ...]) -> dict:
        self.calls.append(("defrag", migrations))
        return {"ok": True}


def test_action_queue_drops_fairness_when_model_inflight() -> None:
    queue = ActionQueue(FakeServiceManagerClient())

    first = queue.submit((ScaleAction("m", 1, "critical", "rescue"),))
    second = queue.submit((ScaleAction("m", 1, "low", "fairness"),))

    assert first.accepted == 1
    assert second.accepted == 0
    assert second.dropped == (("m", "inflight"),)
    assert queue.inflight_models() == {"m"}


def test_action_queue_rescue_replaces_pending_fairness_for_same_model() -> None:
    queue = ActionQueue(FakeServiceManagerClient())

    fairness = queue.submit((ScaleAction("m", -1, "high", "fairness"),))
    rescue = queue.submit((ScaleAction("m", 1, "critical", "rescue"),))

    assert fairness.accepted == 1
    assert rescue.accepted == 1
    assert rescue.replaced == (("m", "fairness"),)
    assert [item.action for item in queue.pending_actions()] == [ScaleAction("m", 1, "critical", "rescue")]


def test_action_queue_dispatches_scale_hide_unhide_and_defrag_actions() -> None:
    client = FakeServiceManagerClient()
    queue = ActionQueue(client)
    migration = Migration(
        serve_id="serve-2",
        from_slot=Slot("node-a", (2,)),
        to_slot=Slot("node-a", (1,)),
    )
    queue.submit(
        (
            ScaleAction("receiver", 1, "critical", "rescue"),
            HideAction("donor", ("pod-a",), "probe_started", "fairness"),
            UnhideAction("recovered", ("pod-z",), "slo_violation", "rescue"),
            DefragAction((migration,), "critical_tp_defrag", "rescue"),
        )
    )

    results = asyncio.run(queue.drain_once())

    assert results == (
        DispatchResult(model="receiver", action_kind="scale", ok=True),
        DispatchResult(model="donor", action_kind="hide", ok=True),
        DispatchResult(model="recovered", action_kind="unhide", ok=True),
        DispatchResult(model="__cluster__", action_kind="defrag", ok=True),
    )
    assert client.calls == [
        ("scale", "receiver", 1),
        ("routable", "donor", ("pod-a",)),
        ("routable", "recovered", ()),
        ("defrag", (migration,)),
    ]
    assert queue.inflight_models() == set()


def test_action_queue_releases_failed_model_after_dispatch_attempt() -> None:
    class FailingClient(FakeServiceManagerClient):
        async def scale_model(self, model: str, delta: int) -> dict:
            self.calls.append(("scale", model, delta))
            return {"ok": False, "error": "boom"}

    queue = ActionQueue(FailingClient())
    queue.submit((ScaleAction("m", 1, "critical", "rescue"),))

    results = asyncio.run(queue.drain_once())

    assert results == (DispatchResult(model="m", action_kind="scale", ok=False, error="boom"),)
    assert queue.inflight_models() == set()

    retry = queue.submit((ScaleAction("m", 1, "critical_retry", "rescue"),))

    assert retry.accepted == 1


class StopQueueLoop(Exception):
    pass


async def _stop_queue_sleep(seconds: float) -> None:
    _stop_queue_sleep.calls.append(seconds)
    raise StopQueueLoop


_stop_queue_sleep.calls = []


def test_action_queue_run_drains_pending_actions_before_sleeping() -> None:
    client = FakeServiceManagerClient()
    queue = ActionQueue(client)
    queue.submit((ScaleAction("m", 1, "critical", "rescue"),))
    _stop_queue_sleep.calls = []

    try:
        asyncio.run(queue.run(poll_interval_s=0.25, sleep=_stop_queue_sleep))
    except StopQueueLoop:
        pass

    assert client.calls == [("scale", "m", 1)]
    assert _stop_queue_sleep.calls == [0.25]
    assert queue.pending_actions() == ()


def test_action_queue_observe_mode_drains_without_dispatching() -> None:
    client = FakeServiceManagerClient()
    observe = {"on": True}
    queue = ActionQueue(client, is_observe=lambda: observe["on"])
    queue.submit(
        (
            ScaleAction("m", 1, "critical", "rescue"),
            HideAction("d", ("pod-a",), "probe", "fairness"),
        )
    )

    results = asyncio.run(queue.drain_once())

    # queue was drained (inflight cleared so the next tick can re-plan) but nothing hit the SM
    assert client.calls == []
    assert queue.inflight_models() == set()
    assert queue.pending_actions() == ()
    assert [(r.action_kind, r.error) for r in results] == [("scale", "observe_skipped"), ("hide", "observe_skipped")]

    # flipping back to active dispatches normally
    observe["on"] = False
    queue.submit((ScaleAction("m", 1, "critical", "rescue"),))
    asyncio.run(queue.drain_once())
    assert client.calls == [("scale", "m", 1)]


def test_action_queue_holds_safescale_action_through_observe_then_dispatches() -> None:
    # Regression: a safescale probe resolution (one-shot, never re-emitted) must not be
    # discarded if drain_once runs while the controller is paused in observe mode.
    client = FakeServiceManagerClient()
    observe = {"on": True}
    queue = ActionQueue(client, is_observe=lambda: observe["on"])
    submitted = queue.submit((UnhideAction("dsllama-8b", ("pod-a",), "slo_violation", "safescale"),))
    assert submitted == SubmitResult(accepted=1, held=1)

    # Observe mode: the safescale action is held, not dispatched and not lost.
    results = asyncio.run(queue.drain_once())
    assert results == ()
    assert client.calls == []
    assert queue.inflight_models() == {"dsllama-8b"}
    assert [item.action for item in queue.pending_actions()] == [
        UnhideAction("dsllama-8b", ("pod-a",), "slo_violation", "safescale")
    ]

    # Draining again while still in observe keeps holding it (no drop, no loss).
    assert asyncio.run(queue.drain_once()) == ()
    assert client.calls == []
    assert len(queue.pending_actions()) == 1

    # Once mode returns to active, the held safescale action dispatches for real.
    observe["on"] = False
    results = asyncio.run(queue.drain_once())
    assert results == (DispatchResult(model="dsllama-8b", action_kind="unhide", ok=True),)
    assert client.calls == [("routable", "dsllama-8b", ())]
    assert queue.pending_actions() == ()
    assert queue.inflight_models() == set()


def test_action_queue_observe_mode_still_drops_non_safescale_actions() -> None:
    # Non-safescale (recurring/idempotent) actions are still skipped+dropped in observe
    # mode, while a safescale action queued alongside them survives the same drain.
    client = FakeServiceManagerClient()
    queue = ActionQueue(client, is_observe=lambda: True)
    queue.submit(
        (
            ScaleAction("m", 1, "critical", "rescue"),
            HideAction("d", ("pod-a",), "probe", "fairness"),
            UnhideAction("safe", ("pod-z",), "slo_violation", "safescale"),
        )
    )

    results = asyncio.run(queue.drain_once())

    assert client.calls == []
    assert [(r.action_kind, r.error) for r in results] == [
        ("scale", "observe_skipped"),
        ("hide", "observe_skipped"),
    ]
    # rescue/fairness dropped from pending and released from inflight...
    assert queue.inflight_models() == {"safe"}
    # ...but the safescale action is held for a later (non-observe) tick.
    assert [item.action for item in queue.pending_actions()] == [
        UnhideAction("safe", ("pod-z",), "slo_violation", "safescale")
    ]


def test_action_queue_rejects_conflicted_safescale_batch_atomically() -> None:
    queue = ActionQueue(FakeServiceManagerClient())
    queue.submit((ScaleAction("receiver", 1, "critical", "rescue"),))

    result = queue.submit(
        (
            ScaleAction("donor", -1, "commit", "safescale"),
            ScaleAction("receiver", 1, "followup", "safescale"),
        )
    )

    assert result == SubmitResult(
        accepted=0,
        dropped=(
            ("donor", "atomic_batch_conflict"),
            ("receiver", "atomic_batch_conflict"),
        ),
    )
    assert [item.action for item in queue.pending_actions()] == [
        ScaleAction("receiver", 1, "critical", "rescue")
    ]
    assert queue.inflight_models() == {"receiver"}
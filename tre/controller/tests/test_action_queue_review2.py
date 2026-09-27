"""Review 2: ordered donor->receiver transfers, one-shot retries, resource serialization."""

from __future__ import annotations

import asyncio

from tre_controller.loops.action_queue import (
    ActionQueue,
    RetryPolicy,
    revalidate_from_cluster_view,
)
from tre_controller.planning.planner import (
    ScaleAction,
    TransferAction,
    UnhideAction,
    fuse_transfers,
)
from tre_sm.allocator.slots import Binding, Slot


class GatedPowerClient:
    """set_binding_power of a gated pod blocks until released; results scripted."""

    def __init__(self, *, gated=(), results=None) -> None:
        self.gates = {pod: asyncio.Event() for pod in gated}
        self.events: list[tuple[str, str, bool]] = []
        self.results: dict[str, list[dict]] = {k: list(v) for k, v in (results or {}).items()}

    async def set_binding_power(self, serve_id, *, awake, **_kwargs):
        self.events.append(("start", serve_id, awake))
        gate = self.gates.get(serve_id)
        if gate is not None:
            await gate.wait()
        else:
            await asyncio.sleep(0)
        self.events.append(("end", serve_id, awake))
        scripted = self.results.get(serve_id)
        if scripted:
            return scripted.pop(0)
        return {"ok": True}

    async def scale_model(self, model, delta, **_kwargs):
        self.events.append(("scale", model, delta > 0))
        return {"ok": True}

    async def set_routable(self, model, hidden_pods):
        self.events.append(("routable", model, bool(hidden_pods)))
        scripted = self.results.get(f"routable:{model}")
        if scripted:
            return scripted.pop(0)
        return {"ok": True}

    async def defrag(self, migrations):
        return {"ok": True}


def _pair(donor="7b", receiver="8b", donor_pod="7b-1", receiver_pod="8b-1", tid="7b->8b#0"):
    return (
        ScaleAction(donor, -1, "critical_donor_immediate", "rescue", donor=donor,
                    receiver=receiver, pods=(donor_pod,), transfer_id=tid),
        ScaleAction(receiver, 1, "critical_donor_immediate", "rescue", donor=donor,
                    receiver=receiver, pods=(receiver_pod,), transfer_id=tid),
    )


async def _until(predicate, steps=400):
    for _ in range(steps):
        if predicate():
            return True
        await asyncio.sleep(0.001)
    return predicate()


def test_fuse_transfers_pairs_by_transfer_id_and_keeps_unpaired_halves() -> None:
    donor, receiver = _pair()
    other = ScaleAction("x", 1, "critical", "rescue")
    lonely = ScaleAction("9b", 1, "critical_donor_immediate", "rescue", pods=("9b-1",), transfer_id="gone")

    fused = fuse_transfers([donor, other, receiver, lonely])

    assert fused == [TransferAction(donor, receiver), other, lonely]


def test_cross_model_rescue_pair_wakes_the_receiver_only_after_the_donor_slept() -> None:
    async def scenario():
        client = GatedPowerClient(gated={"7b-1"})
        queue = ActionQueue(client)
        runner = asyncio.ensure_future(queue.run(poll_interval_s=0.001))
        result = queue.submit(_pair())
        assert result.accepted == 2
        assert queue.inflight_models() == {"7b", "8b"}
        # an unrelated model is not blocked by the draining donor
        queue.submit((ScaleAction("14b", 1, "critical", "rescue"),))
        assert await _until(lambda: ("scale", "14b", True) in client.events)
        await asyncio.sleep(0.01)
        assert ("start", "8b-1", True) not in client.events  # receiver waits for the donor
        # a second receiver action queued meanwhile stays behind the transfer
        queue.submit((ScaleAction("8b", 1, "critical", "rescue", pods=("8b-2",)),))
        await asyncio.sleep(0.01)
        assert ("start", "8b-2", True) not in client.events
        client.gates["7b-1"].set()
        assert await _until(lambda: ("end", "8b-2", True) in client.events)
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
        power = [e for e in client.events if e[0] in ("start", "end")]
        assert power == [
            ("start", "7b-1", False),
            ("end", "7b-1", False),
            ("start", "8b-1", True),
            ("end", "8b-1", True),
            ("start", "8b-2", True),
            ("end", "8b-2", True),
        ]
        assert queue.inflight_models() == set()
        assert {m: d for m, (_, d) in queue.last_actions().items()} == {"7b": "down", "8b": "up", "14b": "up"}

    asyncio.run(scenario())


def test_donor_failure_drops_the_receiver_wake_and_records_why() -> None:
    async def scenario():
        client = GatedPowerClient(
            results={"7b-1": [{"ok": False, "error": "HTTP 409: drain rolled back", "retriable": True}]}
        )
        queue = ActionQueue(client)
        queue.submit(_pair())
        results = await queue.drain_once()
        assert ("start", "8b-1", True) not in client.events
        by_model = {r.model: r for r in results}
        assert by_model["7b"].ok is False
        assert by_model["8b"].ok is False
        assert by_model["8b"].error.startswith("donor_sleep_failed: HTTP 409")
        assert queue.stats()["transfer_receiver_dropped_total"] == 1
        # a re-plannable (rescue) transfer is not retried
        assert [e for e in client.events if e[1] == "7b-1" and e[0] == "start"] == [("start", "7b-1", False)]
        assert queue.inflight_models() == set()

    asyncio.run(scenario())


def _commit(pod="m-1"):
    return ScaleAction("m", -1, "formal_commit_gate_passed", "safescale", pods=(pod,), sleep_path="safescale_commit")


def test_safescale_commit_is_retried_after_409_with_backoff() -> None:
    async def scenario():
        slept: list[float] = []

        async def fake_sleep(seconds):
            slept.append(seconds)

        client = GatedPowerClient(
            results={
                "m-1": [
                    {"ok": False, "error": "HTTP 409: reserved", "status": 409, "retriable": True},
                    {"ok": False, "error": "request timed out", "retriable": True},
                    {"ok": True},
                ]
            }
        )
        queue = ActionQueue(client, retry=RetryPolicy(max_attempts=5, base_backoff_s=2.0, max_backoff_s=3.0), sleep=fake_sleep)
        assert queue.submit((_commit(),)).accepted == 1
        [result] = await queue.drain_once()
        assert result.ok and result.attempts == 3
        assert slept == [2.0, 3.0]  # 2 * 2^0, then capped at max_backoff_s
        assert [e for e in client.events if e[0] == "start"] == [("start", "m-1", False)] * 3
        assert queue.stats()["oneshot_retries_total"] == 2
        assert queue.inflight_models() == set()

    asyncio.run(scenario())


def test_one_shot_permanent_failure_is_not_retried_and_retries_are_bounded() -> None:
    async def scenario():
        async def fake_sleep(_seconds):
            return None

        permanent = GatedPowerClient(results={"m-1": [{"ok": False, "error": "HTTP 400: unknown", "status": 400}]})
        queue = ActionQueue(permanent, sleep=fake_sleep)
        queue.submit((_commit(),))
        [result] = await queue.drain_once()
        assert not result.ok and result.attempts == 1

        busy = GatedPowerClient(results={"m-1": [{"ok": False, "error": "HTTP 503", "retriable": True}] * 10})
        queue = ActionQueue(busy, retry=RetryPolicy(max_attempts=3), sleep=fake_sleep)
        queue.submit((_commit(),))
        [result] = await queue.drain_once()
        assert not result.ok and result.attempts == 3
        assert result.error.startswith("abandoned after 3 attempts")
        assert queue.stats()["oneshot_abandoned_total"] == 1

    asyncio.run(scenario())


def test_one_shot_retry_is_revalidated_and_held_in_observe() -> None:
    async def scenario():
        async def fake_sleep(_seconds):
            return None

        view = {"v": None}

        class View:
            def __init__(self, bindings):
                self.bindings = bindings

        client = GatedPowerClient(results={"m-1": [{"ok": False, "error": "HTTP 409", "retriable": True}]})
        queue = ActionQueue(
            client, sleep=fake_sleep, revalidate=revalidate_from_cluster_view(lambda: view["v"])
        )
        queue.submit((_commit(),))
        # the pod went to sleep meanwhile (e.g. the timed-out first call finished)
        view["v"] = View((Binding("m-1", "m", Slot("n", (0,)), awake=False),))
        [result] = await queue.drain_once()
        assert not result.ok and result.error.startswith("not_retried:")
        assert queue.stats()["oneshot_not_wanted_total"] == 1

        observe = {"on": False}

        async def pause_then_sleep(_seconds):
            observe["on"] = True

        client = GatedPowerClient(results={"m-1": [{"ok": False, "error": "HTTP 409", "retriable": True}]})
        queue = ActionQueue(client, sleep=pause_then_sleep, is_observe=lambda: observe["on"])
        queue.submit((_commit(),))
        await queue.drain_once()
        [held] = queue.pending_actions()  # held for later, not dropped
        assert held.failures == 1 and queue.inflight_models() == {"m"}
        observe["on"] = False
        [result] = await queue.drain_once()
        assert result.ok and result.attempts == 2

    asyncio.run(scenario())


def test_replannable_actions_are_not_retried() -> None:
    async def scenario():
        client = GatedPowerClient(results={"m-1": [{"ok": False, "error": "HTTP 409", "retriable": True}]})
        queue = ActionQueue(client)
        queue.submit((ScaleAction("m", -1, "high", "fairness", pods=("m-1",)),))
        [result] = await queue.drain_once()
        assert not result.ok and result.retriable and result.attempts == 1

    asyncio.run(scenario())


def test_actions_on_a_shared_gpu_are_serialized_across_models() -> None:
    async def scenario():
        slots = {"a-1": ("n", (0,)), "b-1": ("n", (0,)), "c-1": ("n", (1,))}
        client = GatedPowerClient(gated={"a-1"})
        queue = ActionQueue(client, slot_of=slots.get)
        runner = asyncio.ensure_future(queue.run(poll_interval_s=0.001))
        queue.submit((ScaleAction("a", -1, "high", "fairness", pods=("a-1",)),))
        await asyncio.sleep(0.005)
        queue.submit((ScaleAction("b", 1, "critical", "rescue", pods=("b-1",)),))
        queue.submit((ScaleAction("c", 1, "critical", "rescue", pods=("c-1",)),))
        assert await _until(lambda: ("end", "c-1", True) in client.events)  # other GPU: free
        assert ("start", "b-1", True) not in client.events  # same GPU: waits
        client.gates["a-1"].set()
        assert await _until(lambda: ("end", "b-1", True) in client.events)
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)

    asyncio.run(scenario())


def test_cancelling_run_cancels_dispatches_in_flight() -> None:
    async def scenario():
        client = GatedPowerClient(gated={"m-1"})
        queue = ActionQueue(client)
        runner = asyncio.ensure_future(queue.run(poll_interval_s=0.001))
        queue.submit((ScaleAction("m", -1, "high", "fairness", pods=("m-1",)),))
        assert await _until(lambda: ("start", "m-1", False) in client.events)
        running = list(queue._running)
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
        assert running and all(task.cancelled() for task in running)
        assert ("end", "m-1", False) not in client.events

    asyncio.run(scenario())


def test_unhide_rollback_is_one_shot_too() -> None:
    async def scenario():
        async def fake_sleep(_seconds):
            return None

        client = GatedPowerClient(results={"routable:m": [{"ok": False, "error": "HTTP 409", "retriable": True}]})
        queue = ActionQueue(client, sleep=fake_sleep)
        queue.submit((UnhideAction("m", ("m-1",), "rollback", "safescale"),))
        [result] = await queue.drain_once()
        assert result.ok and result.attempts == 2

    asyncio.run(scenario())

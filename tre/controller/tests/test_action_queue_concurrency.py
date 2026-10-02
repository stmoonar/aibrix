"""Review P1-3: a slow scale-down of one model must not block another model."""

from __future__ import annotations

import asyncio

from tre_controller.loops.action_queue import ActionQueue
from tre_controller.planning.planner import ScaleAction


class GatedClient:
    """scale_model of a gated model blocks until released; records overlap."""

    def __init__(self, gated: set[str]) -> None:
        self.gates = {model: asyncio.Event() for model in gated}
        self.events: list[tuple[str, str, int]] = []
        self.active: dict[str, int] = {}
        self.max_active: dict[str, int] = {}

    async def scale_model(self, model: str, delta: int, **_kwargs) -> dict:
        self.active[model] = self.active.get(model, 0) + 1
        self.max_active[model] = max(self.max_active.get(model, 0), self.active[model])
        self.events.append(("start", model, delta))
        gate = self.gates.get(model)
        if gate is not None:
            await gate.wait()
        else:
            await asyncio.sleep(0)
        self.events.append(("end", model, delta))
        self.active[model] -= 1
        return {"ok": True}

    async def set_routable(self, model, hidden_pods):
        return {"ok": True}

    async def set_binding_power(self, serve_id, *, awake, **_kwargs):
        return {"ok": True}

    async def defrag(self, migrations):
        return {"ok": True}


def test_slow_model_does_not_block_another_models_urgent_action() -> None:
    async def scenario():
        client = GatedClient({"slow"})
        queue = ActionQueue(client)
        runner = asyncio.ensure_future(queue.run(poll_interval_s=0.001))
        queue.submit((ScaleAction("slow", -1, "high", "fairness", sleep_path="urgent"),))
        await asyncio.sleep(0.01)
        queue.submit((ScaleAction("fast", 1, "critical_immediate", "rescue"),))
        for _ in range(100):
            await asyncio.sleep(0.001)
            if ("end", "fast", 1) in client.events:
                break
        # fast finished while slow is still draining
        assert ("end", "fast", 1) in client.events
        assert ("end", "slow", -1) not in client.events
        assert queue.inflight_models() == {"slow"}
        client.gates["slow"].set()
        for _ in range(100):
            await asyncio.sleep(0.001)
            if ("end", "slow", -1) in client.events:
                break
        runner.cancel()
        assert ("end", "slow", -1) in client.events
        assert queue.inflight_models() == set()

    asyncio.run(scenario())


def test_actions_of_one_model_stay_serialized_and_in_order() -> None:
    async def scenario():
        client = GatedClient({"m"})
        queue = ActionQueue(client)
        runner = asyncio.ensure_future(queue.run(poll_interval_s=0.001))
        queue.submit((ScaleAction("m", -1, "high", "fairness", sleep_path="urgent"),))
        await asyncio.sleep(0.01)
        # A rescue for the same model is queued behind the running action.
        queue.submit((ScaleAction("m", 2, "critical", "rescue"),))
        await asyncio.sleep(0.01)
        assert [e for e in client.events if e[0] == "start"] == [("start", "m", -1)]
        client.gates["m"].set()
        for _ in range(200):
            await asyncio.sleep(0.001)
            if ("end", "m", 2) in client.events:
                break
        runner.cancel()
        assert client.events == [
            ("start", "m", -1),
            ("end", "m", -1),
            ("start", "m", 2),
            ("end", "m", 2),
        ]
        assert client.max_active["m"] == 1

    asyncio.run(scenario())


def test_drain_once_runs_models_concurrently_and_waits_for_all() -> None:
    async def scenario():
        client = GatedClient({"a"})
        queue = ActionQueue(client)
        queue.submit((ScaleAction("a", 1, "critical", "rescue"), ScaleAction("b", 1, "critical", "rescue")))
        drain = asyncio.ensure_future(queue.drain_once())
        for _ in range(50):
            await asyncio.sleep(0.001)
            if ("end", "b", 1) in client.events:
                break
        assert ("end", "b", 1) in client.events and not drain.done()
        client.gates["a"].set()
        results = await drain
        assert sorted(r.model for r in results) == ["a", "b"] and all(r.ok for r in results)

    asyncio.run(scenario())

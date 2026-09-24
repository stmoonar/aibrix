"""Wire contract: controller ActionQueue + ServiceManagerClient against the real SM app
(TRE_SM_ASYNC + TRE_SM_CALL_DRAIN on both sides), SM runtime faked."""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient

from tre_controller.loops.action_queue import ActionQueue, DispatchResult
from tre_controller.planning.planner import ScaleAction
from tre_controller.sm_client import ServiceManagerClient, ServiceManagerError
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.api.v2 import ServiceManagerV2, create_app
from tre_sm.ops.drain import DrainConfig, SleepDrainer
from tre_sm.state.async_ops import AsyncOpsConfig
from tre_sm.state.store import StateStore

_SM_TESTS = Path(__file__).resolve().parents[2] / "service-manager" / "tests"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_e2e_sm_{name}", _SM_TESTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_sm = _load("test_drain_before_sleep")


class _TestClientTransport:
    def __init__(self, client: TestClient) -> None:
        self._client = client

    async def request(self, method, url, *, json=None, timeout_s):
        path = url.split("://", 1)[1].split("/", 1)[1]
        response = await asyncio.to_thread(self._client.request, method, "/" + path, json=json)
        if response.status_code >= 400:
            raise ServiceManagerError(f"HTTP {response.status_code}: {response.text}")
        return response.json()


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000_000

    def __call__(self) -> int:
        return self.now


def _service(async_on: bool):
    events: list = []
    runtime = _sm.FakeRuntimeOps(events, [_sm._pod("serve-a", 0, "10.0.0.1"), _sm._pod("serve-b", 1, "10.0.0.2")])
    vllm = _sm.FakeVllmOps(events, pages={"10.0.0.2": [_sm.metrics_text(1, 0), _sm.metrics_text(0, 0)]})
    config = DrainConfig(hide_before_sleep=True)
    clock = _sm.FakeClock()
    store = StateStore(_sm.FakeRedis())
    store.save(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-b", "m1", Slot("node-a", (1,)), awake=True, hidden=True),
        ],
        expected_version=0,
    )
    service = ServiceManagerV2(
        _sm.registry(),
        store,
        runtime_ops=runtime,
        vllm_ops=vllm,
        drain_config=config,
        sleep_drainer=SleepDrainer(runtime, vllm, config, monotonic=clock.monotonic, sleep=clock.sleep),
        async_config=AsyncOpsConfig(enabled=async_on),
    )
    return service, vllm


def test_async_safescale_commit_round_trip_with_drain_budget():
    service, vllm = _service(async_on=True)
    client = ServiceManagerClient("http://sm", transport=_TestClientTransport(TestClient(create_app(service))))
    clock = _Clock()
    queue = ActionQueue(client, now_ms=clock, async_ops=True, call_drain=True)
    queue.submit((ScaleAction("m1", -1, "formal_commit_gate_passed", "safescale", pods=("serve-b",), drain_s=20.0),))

    assert asyncio.run(queue.drain_once()) == ()
    (tracked,) = queue._ops.values()
    (operation_id,) = tracked.op_ids
    assert service.wait_async_operation(operation_id, timeout_s=10.0)
    clock.now += 2_000
    results = asyncio.run(queue.drain_once())

    assert results == (DispatchResult(model="m1", action_kind="scale", ok=True),)
    record = service.get_operation(operation_id)
    (binding,) = record["bindings"]
    assert binding["outcome"] == "slept" and binding["drained"] is True
    assert binding["drain_budget_s"] == 20.0
    assert vllm.sleep_kwargs == [{"hidden": True}]
    assert queue.op_stats()["scale:down:safescale"]["ok"] == 1
    assert queue.last_actions()["m1"] == (clock.now, "down")


def test_controller_async_against_an_sm_without_async_is_synchronous():
    service, vllm = _service(async_on=False)
    client = ServiceManagerClient("http://sm", transport=_TestClientTransport(TestClient(create_app(service))))
    queue = ActionQueue(client, now_ms=_Clock(), async_ops=True, call_drain=True)
    queue.submit((ScaleAction("m1", -1, "idle_proactive_immediate", "rescue", pods=("serve-b",)),))

    results = asyncio.run(queue.drain_once())

    assert results == (DispatchResult(model="m1", action_kind="scale", ok=True),)
    assert queue.inflight_models() == set()
    assert vllm.sleep_kwargs == [{"hidden": True}]

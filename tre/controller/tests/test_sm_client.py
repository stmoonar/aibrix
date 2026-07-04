from __future__ import annotations

import pytest

from tre_controller.sm_client import ServiceManagerClient, ServiceManagerError


class FakeTransport:
    def __init__(self, responses: list[dict] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[tuple[str, str, dict | None]] = []

    async def request(self, method: str, url: str, *, json: dict | None = None, timeout_s: float) -> dict:
        self.calls.append((method, url, json))
        if not self.responses:
            return {"ok": True}
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.asyncio
async def test_sm_client_scale_model_converts_delta_to_v2_target() -> None:
    transport = FakeTransport(
        responses=[
            {"models": {"m": {"awake": 2, "bound": 4}}, "bindings": []},
            {"model": "m", "wake_replicas": 3, "version": 2, "actions": [{"action": "wake", "serve_id": "s3"}]},
        ]
    )
    client = ServiceManagerClient("http://sm.local/", transport=transport, timeout_s=1.5)

    result = await client.scale_model("m", 1)

    assert result == {"ok": True, "response": {"model": "m", "wake_replicas": 3, "version": 2, "actions": [{"action": "wake", "serve_id": "s3"}]}}
    assert transport.calls == [
        ("GET", "http://sm.local/v2/state", None),
        ("PUT", "http://sm.local/v2/models/m/target", {"wake_replicas": 3}),
    ]


@pytest.mark.asyncio
async def test_sm_client_scale_model_clamps_down_target_at_zero() -> None:
    transport = FakeTransport(
        responses=[
            {"models": {"m": {"awake": 1, "bound": 4}}, "bindings": []},
            {"model": "m", "wake_replicas": 0, "version": 2, "actions": [{"action": "sleep", "serve_id": "s1"}]},
        ]
    )
    client = ServiceManagerClient("http://sm.local", transport=transport)

    result = await client.scale_model("m", -3)

    assert result["ok"] is True
    assert transport.calls[-1] == ("PUT", "http://sm.local/v2/models/m/target", {"wake_replicas": 0})


@pytest.mark.asyncio
async def test_sm_client_set_routable_puts_hidden_pods() -> None:
    transport = FakeTransport(responses=[{"model": "m", "hidden_pods": ["pod-a"], "version": 2, "actions": []}])
    client = ServiceManagerClient("http://sm.local", transport=transport)

    result = await client.set_routable("m", ("pod-a",))

    assert result["ok"] is True
    assert transport.calls == [("PUT", "http://sm.local/v2/models/m/routable", {"hidden_pods": ["pod-a"]})]


@pytest.mark.asyncio
async def test_sm_client_get_cluster_view_returns_state_response() -> None:
    state = {"version": 1, "models": {}, "bindings": [{"serve_id": "s1"}]}
    transport = FakeTransport(responses=[state])
    client = ServiceManagerClient("http://sm.local", transport=transport)

    assert await client.get_state() == state
    assert transport.calls == [("GET", "http://sm.local/v2/state", None)]


@pytest.mark.asyncio
async def test_sm_client_normalizes_http_failure() -> None:
    transport = FakeTransport(responses=[ServiceManagerError("bad gateway")])
    client = ServiceManagerClient("http://sm.local", transport=transport)

    result = await client.get_state_result()

    assert result == {"ok": False, "error": "bad gateway"}


@pytest.mark.asyncio
async def test_sm_client_defrag_posts_v2_defrag_request() -> None:
    transport = FakeTransport(
        responses=[
            {
                "version": 2,
                "migrations": [
                    {
                        "serve_id": "serve-b",
                        "from_slot": {"node": "node-a", "gpu_ids": [2]},
                        "to_slot": {"node": "node-a", "gpu_ids": [1]},
                    }
                ],
                "actions": [],
            }
        ]
    )
    client = ServiceManagerClient("http://sm.local", transport=transport)

    result = await client.defrag(())

    assert result == {
        "ok": True,
        "response": {
            "version": 2,
            "migrations": [
                {
                    "serve_id": "serve-b",
                    "from_slot": {"node": "node-a", "gpu_ids": [2]},
                    "to_slot": {"node": "node-a", "gpu_ids": [1]},
                }
            ],
            "actions": [],
        },
    }
    assert transport.calls == [("POST", "http://sm.local/v2/defrag", {"tp_size": 2})]


@pytest.mark.asyncio
async def test_sm_client_defrag_keeps_unsupported_fallback_for_old_service_manager() -> None:
    transport = FakeTransport(responses=[ServiceManagerError("HTTP 404: not found")])
    client = ServiceManagerClient("http://sm.local", transport=transport)

    result = await client.defrag(())

    assert result == {"ok": False, "error": "defrag endpoint is not implemented in service-manager v2"}

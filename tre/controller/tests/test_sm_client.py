from __future__ import annotations

import pytest

from tre_controller.sm_client import ServiceManagerClient, ServiceManagerError


class FakeTransport:
    def __init__(self, responses: list[dict] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[tuple[str, str, dict | None]] = []
        self.timeouts: list[float] = []

    async def request(self, method: str, url: str, *, json: dict | None = None, timeout_s: float) -> dict:
        self.calls.append((method, url, json))
        self.timeouts.append(timeout_s)
        if not self.responses:
            return {"ok": True}
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.asyncio
async def test_sm_client_post_plan_sends_one_request_on_the_slow_timeout() -> None:
    # Design 20261027-sm-plans §5.1: one POST /v2/plans, no GET precheck, long timeout
    # (a plan runs its sleeps and wakes inside the SM handler); get_state / set_routable
    # stay on the fast timeout.
    body = {"plan_id": "pl-1", "status": "done", "sleeps": [], "wakes": [{"index": 0, "model": "m", "status": "done"}]}
    transport = FakeTransport(responses=[body])
    client = ServiceManagerClient("http://sm.local/", transport=transport, timeout_s=5.0, slow_timeout_s=300.0)
    request = {"sleeps": [], "wakes": [{"model": "m", "at_least": 3, "reason": "r"}]}

    result = await client.post_plan(request)

    assert result == {"ok": True, "status": 200, "response": body}
    assert transport.calls == [("POST", "http://sm.local/v2/plans", request)]
    assert transport.timeouts == [300.0]

    t2 = FakeTransport(responses=[{"ok": True}])
    await ServiceManagerClient("http://sm.local/", transport=t2, timeout_s=5.0, slow_timeout_s=300.0).defrag(())
    assert t2.timeouts == [300.0]

    t3 = FakeTransport(responses=[{"model": "m", "hidden_pods": [], "version": 1, "actions": []}])
    await ServiceManagerClient("http://sm.local/", transport=t3, timeout_s=5.0, slow_timeout_s=300.0).set_routable("m", ())
    assert t3.timeouts == [5.0]


@pytest.mark.asyncio
async def test_sm_client_post_plan_409_partial_carries_the_item_results() -> None:
    body = {"error": "partial", "plan_id": "pl-2", "status": "partial", "sleeps": [],
            "wakes": [{"index": 0, "model": "m", "status": "refused", "error": "gpu_busy"}]}
    transport = FakeTransport(responses=[ServiceManagerError("HTTP 409: partial", status=409, body=body)])
    client = ServiceManagerClient("http://sm.local", transport=transport)

    result = await client.post_plan({"sleeps": [], "wakes": [{"model": "m", "at_least": 2}]})

    assert result["ok"] is False and result["partial"] is True
    assert result["response"] == body
    assert not result.get("not_executed")


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

    assert result == {"ok": False, "error": "bad gateway", "status": None, "retriable": False}


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

    assert result == {
        "ok": False,
        "error": "defrag endpoint is not implemented in service-manager v2",
        "status": 404,
        "retriable": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "retriable"),
    [
        (ServiceManagerError("HTTP 409: busy", status=409, body={"error": "writer_busy"}), True),
        (ServiceManagerError("HTTP 503: shutting down", status=503), True),
        (TimeoutError("timed out"), True),
        (ConnectionRefusedError("refused"), True),
        (ServiceManagerError("HTTP 400: unknown binding", status=400), False),
        (ServiceManagerError("HTTP 500: boom", status=500), False),
    ],
)
async def test_sm_client_classifies_retriable_failures_review2_p2_5(error, retriable) -> None:
    transport = FakeTransport(responses=[error])
    client = ServiceManagerClient("http://sm.local", transport=transport)

    result = await client.post_plan({"sleeps": [{"model": "m", "pods": ["p1"], "sleep_path": "urgent"}], "wakes": []})

    assert result["ok"] is False
    assert result["retriable"] is retriable
    assert "partial" not in result

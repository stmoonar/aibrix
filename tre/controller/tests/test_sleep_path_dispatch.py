"""Controller -> SM sleep path + drain budget (plan 2026-09-27 D1)."""

from __future__ import annotations

import asyncio

import pytest

from tre_controller.loops.action_queue import ActionQueue, _sleep_kwargs
from tre_controller.planning.planner import ScaleAction
from tre_controller.sm_client import ServiceManagerClient


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def scale_model(self, model, delta, **kwargs):
        self.calls.append(("scale", model, delta, kwargs))
        return {"ok": True}

    async def set_binding_power(self, serve_id, *, awake, **kwargs):
        self.calls.append(("power", serve_id, awake, kwargs))
        return {"ok": True}


def _dispatch(*actions):
    client = RecordingClient()
    queue = ActionQueue(client)
    for action in actions:
        queue.submit((action,))
        asyncio.run(queue.drain_once())
    return client.calls


def test_immediate_donor_paths_are_urgent():
    calls = _dispatch(ScaleAction("m", -1, "critical_donor_immediate", "rescue"))
    assert calls == [("scale", "m", -1, {"sleep_path": "urgent"})]


def test_pathless_scale_down_is_refused_not_sent_to_the_sm_default():
    # 2026-10-02: no implicit `scale_down` path; the dispatch wrapper reports a
    # failed result and nothing reaches the SM.
    client = RecordingClient()
    queue = ActionQueue(client)
    action = ScaleAction("m", -1, "high_release", "fairness")
    with pytest.raises(ValueError, match="no sleep_path"):
        _sleep_kwargs(action)
    queue.submit((action,))
    results = asyncio.run(queue.drain_once())
    assert client.calls == []
    assert [r.ok for r in results] == [False]
    assert results[0].error.startswith("sleep_path_refused") and results[0].retriable is False


def test_pathless_binding_scale_down_is_refused_too():
    client = RecordingClient()
    queue = ActionQueue(client)
    queue.submit((ScaleAction("m", -1, "high_release", "fairness", pods=("pod-a",)),))
    results = asyncio.run(queue.drain_once())
    assert client.calls == []
    assert [r.ok for r in results] == [False]
    assert "no sleep_path" in results[0].error


def test_sleep_kwargs_keeps_explicit_paths_and_ignores_scale_ups():
    assert _sleep_kwargs(ScaleAction("m", 1, "x", "rescue")) == {}
    assert _sleep_kwargs(ScaleAction("m", -1, "x_immediate", "rescue")) == {"sleep_path": "urgent"}
    assert _sleep_kwargs(ScaleAction("m", -1, "x", "safescale", sleep_path="safescale_commit")) == {
        "sleep_path": "safescale_commit"
    }


def test_scale_up_sends_no_sleep_fields():
    calls = _dispatch(ScaleAction("m", 2, "critical", "rescue"))
    assert calls == [("scale", "m", 2, {})]


def test_safescale_commit_sends_path_and_probe_window_budget():
    calls = _dispatch(
        ScaleAction(
            "m", -1, "formal_commit_gate_passed", "safescale", pods=("pod-a",),
            sleep_path="safescale_commit", drain_budget_s=42.0,
        )
    )
    assert calls == [
        ("power", "pod-a", False, {"sleep_path": "safescale_commit", "drain_budget_s": 42.0})
    ]


class Transport:
    def __init__(self) -> None:
        self.calls = []

    async def request(self, method, url, *, json=None, timeout_s):
        self.calls.append((method, url, json))
        if url.endswith("/v2/state"):
            return {"models": {"m": {"awake": 2, "bound": 4}}}
        return {"ok": True}


@pytest.mark.asyncio
async def test_sm_client_puts_sleep_fields_only_when_given():
    transport = Transport()
    client = ServiceManagerClient("http://sm", transport=transport)

    await client.scale_model("m", -1, sleep_path="urgent", drain_budget_s=30)
    await client.scale_model("m", -1)
    await client.set_binding_power("p", awake=False, sleep_path="safescale_commit", drain_budget_s=12)

    puts = [call for call in transport.calls if call[0] == "PUT"]
    assert puts == [
        ("PUT", "http://sm/v2/models/m/target", {"wake_replicas": 1, "sleep_path": "urgent", "drain_budget_s": 30.0}),
        ("PUT", "http://sm/v2/models/m/target", {"wake_replicas": 1}),
        ("PUT", "http://sm/v2/bindings/p/power", {"awake": False, "sleep_path": "safescale_commit", "drain_budget_s": 12.0}),
    ]

from __future__ import annotations

import asyncio

from tre_common.registry import ClusterTopology, NodeSpec
from tre_controller.loops.cluster_view_task import (
    ClusterViewBox,
    cluster_view_from_state,
    refresh_cluster_view_once,
)
from tre_sm.allocator.slots import Binding, Slot


class FakeClient:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def get_state(self) -> dict:
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _topology() -> ClusterTopology:
    return ClusterTopology(
        nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)
    )


def _state() -> dict:
    return {
        "version": 7,
        "models": {"m1": {"awake": 1, "bound": 2}},
        "bindings": [
            {
                "serve_id": "serve-a",
                "model": "m1",
                "node": "node-a",
                "gpu_ids": [0],
                "awake": True,
                "hidden": False,
            },
            {
                "serve_id": "serve-b",
                "model": "m1",
                "node": "node-a",
                "gpu_ids": [2, 3],
                "awake": False,
                "hidden": True,
            },
        ],
    }


def test_cluster_view_from_state_preserves_topology_and_bindings() -> None:
    view = cluster_view_from_state(_state(), _topology())

    assert view.topology == _topology()
    assert view.bindings == (
        Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True, hidden=False),
        Binding("serve-b", "m1", Slot("node-a", (2, 3)), awake=False, hidden=True),
    )


def test_refresh_cluster_view_once_replaces_cached_view_from_service_manager_state() -> None:
    client = FakeClient([_state()])
    box = ClusterViewBox()

    result = asyncio.run(refresh_cluster_view_once(client, _topology(), box))

    assert result.refreshed is True
    assert result.error is None
    assert client.calls == 1
    assert box.get() == result.cluster_view
    assert box.get().bindings[0].serve_id == "serve-a"


def test_refresh_cluster_view_once_keeps_previous_view_on_failure() -> None:
    previous = cluster_view_from_state(_state(), _topology())
    box = ClusterViewBox(previous)
    client = FakeClient([RuntimeError("service unavailable")])

    result = asyncio.run(refresh_cluster_view_once(client, _topology(), box))

    assert result.refreshed is False
    assert result.error == "service unavailable"
    assert result.cluster_view == previous
    assert box.get() == previous

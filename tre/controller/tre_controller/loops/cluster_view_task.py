from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

from tre_common.registry import ClusterTopology
from tre_controller.planning.planner import ClusterView
from tre_sm.allocator.slots import Binding, Slot


class StateClient(Protocol):
    async def get_state(self) -> dict: ...


class ClusterViewTaskConfig(Protocol):
    fairness_interval_s: float


class ClusterViewBox:
    def __init__(self, cluster_view: ClusterView | None = None) -> None:
        self._cluster_view = cluster_view

    def get(self) -> ClusterView | None:
        return self._cluster_view

    def set(self, cluster_view: ClusterView) -> None:
        self._cluster_view = cluster_view


@dataclass(frozen=True)
class ClusterViewRefreshResult:
    cluster_view: ClusterView | None
    refreshed: bool
    error: str | None = None


def cluster_view_from_state(state: dict, topology: ClusterTopology) -> ClusterView:
    bindings = []
    for item in state.get("bindings", []):
        bindings.append(
            Binding(
                serve_id=str(item["serve_id"]),
                model=str(item["model"]),
                slot=Slot(
                    str(item["node"]),
                    tuple(int(gpu) for gpu in item.get("gpu_ids", ())),
                ),
                awake=bool(item.get("awake", False)),
                hidden=bool(item.get("hidden", False)),
            )
        )
    return ClusterView(topology=topology, bindings=tuple(bindings))


async def refresh_cluster_view_once(
    client: StateClient,
    topology: ClusterTopology,
    cluster_view_box: ClusterViewBox,
) -> ClusterViewRefreshResult:
    try:
        state = await client.get_state()
        cluster_view = cluster_view_from_state(state, topology)
    except Exception as exc:  # noqa: BLE001 - cached view is a conservative fallback.
        return ClusterViewRefreshResult(
            cluster_view=cluster_view_box.get(),
            refreshed=False,
            error=str(exc),
        )
    cluster_view_box.set(cluster_view)
    return ClusterViewRefreshResult(cluster_view=cluster_view, refreshed=True)


async def cluster_view_task(
    client: StateClient,
    topology: ClusterTopology,
    cluster_view_box: ClusterViewBox,
    cfg: ClusterViewTaskConfig,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    while True:
        await refresh_cluster_view_once(client, topology, cluster_view_box)
        await sleep(cfg.fairness_interval_s)

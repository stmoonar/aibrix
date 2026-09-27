from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

from tre_common.registry import ClusterTopology
from tre_controller.planning.planner import ClusterView
from tre_sm.allocator.slots import Binding, Slot


class StateClient(Protocol):
    async def get_state(self) -> dict: ...


class ClusterViewTaskConfig(Protocol):
    fairness_interval_s: float


#: Default age (s) past which a cluster view is not "fresh" (review 4 P2-1):
#: two refresh periods of the default 10 s fairness interval.
DEFAULT_FRESH_MAX_AGE_S = 25.0


class ClusterViewBox:
    """Latest SM cluster view. ``get`` returns it whatever its age (planning
    tolerates a refresh period of lag); ``fresh`` only while it is younger than
    ``max_age_s`` - for consumers where staleness matters (retry / commit
    revalidation, preemption compensation, unhide filtering), which fall back to
    their conservative behaviour without one (review 4 P2-1)."""

    def __init__(
        self,
        cluster_view: ClusterView | None = None,
        *,
        max_age_s: float = DEFAULT_FRESH_MAX_AGE_S,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._monotonic = monotonic
        self.max_age_s = float(max_age_s)
        self._cluster_view = cluster_view
        self._updated_at = monotonic() if cluster_view is not None else None

    def get(self) -> ClusterView | None:
        return self._cluster_view

    def age_s(self) -> float | None:
        if self._updated_at is None:
            return None
        return max(0.0, self._monotonic() - self._updated_at)

    def fresh(self) -> ClusterView | None:
        age = self.age_s()
        if age is None or age > self.max_age_s:
            return None
        return self._cluster_view

    def set(self, cluster_view: ClusterView) -> None:
        self._cluster_view = cluster_view
        self._updated_at = self._monotonic()


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

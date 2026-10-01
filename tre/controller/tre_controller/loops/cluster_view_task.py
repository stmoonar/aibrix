from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Awaitable, Callable, Protocol

from tre_common.registry import ClusterTopology
from tre_controller.planning.planner import ClusterView
from tre_sm.allocator.slots import Binding, Slot


class StateClient(Protocol):
    async def get_state(self) -> dict: ...


class ClusterViewTaskConfig(Protocol):
    fairness_interval_s: float


def wall_clock_ms() -> int:
    """Epoch ms on the controller clock (the clock the action queue's done times and the
    metrics sampler's window boundaries use)."""
    return int(time.time() * 1000)


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
    return ClusterView(
        topology=topology,
        bindings=tuple(bindings),
        pod_ips=_observed_pod_ips(state),
        blocked_gpus=_blocked_gpus(state),
    )


#: ``/v2/state`` ``gpus[].reason`` values that make a GPU no wake / create capacity
#: although the bindings show no awake binding there (S5): a Pod loading, a wake in
#: flight, gpu-truth showing memory in use that no binding explains. ``awake`` /
#: ``draining`` GPUs hold an awake binding the planner sees itself.
BLOCKING_GPU_REASONS = frozenset({"loading", "waking", "gpu_truth_used"})


def _blocked_gpus(state: dict) -> frozenset:
    """(node, gpu) the SM reports not wakeable for a reason the bindings do not
    show. Empty for an SM that does not report ``gpus`` (best effort)."""
    entries = state.get("gpus") if isinstance(state, dict) else None
    blocked = set()
    for entry in entries if isinstance(entries, list) else ():
        if not isinstance(entry, dict) or entry.get("wakeable") is not False:
            continue
        if entry.get("reason") not in BLOCKING_GPU_REASONS:
            continue
        try:
            blocked.add((str(entry["node"]), int(entry["gpu"])))
        except (KeyError, TypeError, ValueError):
            continue
    return frozenset(blocked)


def _observed_pod_ips(state: dict) -> dict[str, str]:
    """pod name -> pod IP of the SM fleet state's observed bindings (best effort: a
    malformed or missing ``fleet`` section yields no IPs, never an error)."""
    fleet = state.get("fleet") if isinstance(state, dict) else None
    observed = fleet.get("observed") if isinstance(fleet, dict) else None
    ips: dict[str, str] = {}
    for item in observed if isinstance(observed, list) else ():
        if not isinstance(item, dict):
            continue
        pod, ip = item.get("pod_name"), item.get("pod_ip")
        if pod and ip:
            ips[str(pod)] = str(ip)
    return ips


async def refresh_cluster_view_once(
    client: StateClient,
    topology: ClusterTopology,
    cluster_view_box: ClusterViewBox,
) -> ClusterViewRefreshResult:
    try:
        state = await client.get_state()
        # O1: stamped after the response arrived - every change the view shows happened
        # at or before this time (an upper bound, so a breakpoint is never dated early).
        cluster_view = replace(
            cluster_view_from_state(state, topology), fetched_ms=int(wall_clock_ms())
        )
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

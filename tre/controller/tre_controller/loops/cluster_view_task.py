from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, replace
from typing import Awaitable, Callable, Protocol

from tre_common.registry import ClusterTopology
from tre_controller.planning.planner import ClusterView
from tre_controller.sm_client import parse_state_routable
from tre_sm.allocator.slots import Binding, Slot


LOG = logging.getLogger("tre_controller.cluster_view")


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
    # 2026-10-02: the SM's own routable count (its floor check) and floor headroom; a
    # missing / null field leaves ``routable_ids`` None and the tick falls back to its
    # own count with an event (``sm_routable_fallback``).
    routable = parse_state_routable(state)
    return ClusterView(
        topology=topology,
        bindings=tuple(bindings),
        pod_ips=_observed_pod_ips(state),
        blocked_gpus=_blocked_gpus(state),
        routable_ids=routable.routable_ids,
        model_floors=dict(routable.models),
        floor_enforced=routable.floor_enforced,
        routable_error=routable.error,
        sm_version=_state_version(state),
    )


def _state_version(state: dict) -> int | None:
    """``/v2/state`` ``version`` (the SM binding-store version), None when absent."""
    raw = state.get("version") if isinstance(state, dict) else None
    if isinstance(raw, bool):
        return None
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


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


def state_time_lower_bound(state: dict, requested_ms: int, fetched_ms: int) -> int:
    """When the SM state was produced, as a lower bound on the controller clock: the time
    the request was sent. Timestamps are never compared across machines: the SM's own
    ``fetched_ms`` is on the SM clock (which may run ahead or behind; a fast SM clock
    clamped into the request interval would become an upper bound), so it is kept on
    the view for reference only (:func:`sm_state_ms`) and decides nothing."""
    return int(requested_ms)


def sm_state_ms(state: dict) -> int | None:
    """The SM's own ``/v2/state`` ``fetched_ms`` (SM clock): logs / snapshots only."""
    raw = state.get("fetched_ms") if isinstance(state, dict) else None
    try:
        return int(float(raw)) if raw is not None and not isinstance(raw, bool) else None
    except (TypeError, ValueError, OverflowError):
        return None


async def refresh_cluster_view_once(
    client: StateClient,
    topology: ClusterTopology,
    cluster_view_box: ClusterViewBox,
) -> ClusterViewRefreshResult:
    try:
        # Timer cleanup review P2-2: the request time is a lower bound of the state's time.
        requested_ms = int(wall_clock_ms())
        state = await client.get_state()
        # O1: stamped after the response arrived - every change the view shows happened
        # at or before this time (an upper bound, so a breakpoint is never dated early).
        fetched_ms = int(wall_clock_ms())
        cluster_view = replace(
            cluster_view_from_state(state, topology),
            fetched_ms=fetched_ms,
            state_ms=state_time_lower_bound(state, requested_ms, fetched_ms),
            sm_fetched_ms=sm_state_ms(state),
        )
    except Exception as exc:  # noqa: BLE001 - cached view is a conservative fallback.
        return ClusterViewRefreshResult(
            cluster_view=cluster_view_box.get(),
            refreshed=False,
            error=str(exc),
        )
    cluster_view_box.set(cluster_view)
    return ClusterViewRefreshResult(cluster_view=cluster_view, refreshed=True)


class StaleViewAlert:
    """Review P3-6: alert (no control change) while the fleet view cannot be refreshed.

    The view's age is measured from its state time (``state_ms``: the controller's
    request time, never the SM clock; else
    ``fetched_ms``). Past ``stale_ms`` the event ``cluster_view_stale`` (age, last
    refresh error) is logged once per stale period; the first fresh view afterwards
    logs ``cluster_view_recovered``. The planner keeps its holds while stale on purpose:
    without fresh state it stays conservative (no LOW scale-up, no scale-down of a model
    with a pending change; CRITICAL receivers still scale up)."""

    def __init__(self, stale_ms: float) -> None:
        self.stale_ms = float(stale_ms)
        self.stale = False

    def check(self, view: ClusterView | None, now_ms: int, error: str | None) -> dict | None:
        if self.stale_ms <= 0:
            return None
        stamp = None
        if view is not None:
            stamp = view.state_ms if view.state_ms is not None else view.fetched_ms
        age = None if stamp is None else int(now_ms) - int(stamp)
        if age is not None and age <= self.stale_ms:
            if self.stale:
                self.stale = False
                return {"event": "cluster_view_recovered", "age_ms": age}
            return None
        if age is None and view is None and error is None:
            return None  # nothing fetched yet and nothing failed: starting up
        if self.stale:
            return None
        self.stale = True
        return {"event": "cluster_view_stale", "age_ms": age, "last_error": error}


async def cluster_view_task(
    client: StateClient,
    topology: ClusterTopology,
    cluster_view_box: ClusterViewBox,
    cfg: ClusterViewTaskConfig,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock_ms: Callable[[], int] | None = None,
) -> None:
    alert = StaleViewAlert(
        float(getattr(cfg, "view_stale_periods", 3) or 0) * float(cfg.fairness_interval_s) * 1000.0
    )
    clock = clock_ms or wall_clock_ms
    while True:
        result = await refresh_cluster_view_once(client, topology, cluster_view_box)
        event = alert.check(result.cluster_view, clock(), result.error)
        if event is not None:
            log = LOG.warning if event["event"] == "cluster_view_stale" else LOG.info
            log(json.dumps(event, sort_keys=True))
        await sleep(cfg.fairness_interval_s)

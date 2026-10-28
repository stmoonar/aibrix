"""Timer cleanup, independent review fixes (2026-10-02, docs/design/20261002-timer-cleanup.md).

P2-1  the view-pending guard also covers probe unhides and failed / partial SM calls
      (``ActionQueue.view_changes``); since 2026-10-29 (TSS only) it holds every model
      in every role, CRITICAL receivers included;
P2-2  a view's time for that guard is a LOWER bound of when its SM state was produced;
P2-4  ``insufficient_evidence:stalled`` is a capacity rollback;
P3-6  a fleet view older than N refresh periods only raises an alert (holds kept).
(P3-7, the O1 evidence wait at every hop of two models trading a pod, went with O1.)"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from tre_controller.config import ControllerConfig
from tre_controller.loops import cluster_view_task as cluster_view_module
from tre_controller.loops.cluster_view_task import (
    ClusterViewBox,
    StaleViewAlert,
    cluster_view_task,
    refresh_cluster_view_once,
    state_time_lower_bound,
)
from tre_controller.loops.tick import _view_pending, run_planner_tick
from tre_controller.planning.planner import ScaleAction, UnhideAction
from tre_controller.planning.safescale import CAPACITY_ROLLBACK_CODES, ProbeWindowInputs, SafeScaleStateMachine
from tre_controller.config import SafeScaleConfig
from tre_controller.signals.trs import SignalState

from test_tss_only_20261029 import BASE, GRID, LOW_LOAD as LOW, _Queue as _ActionQueue, _registry, _snap, _ups, _view, _window
from plan_shim import LegacyFakeQueue as ActionQueue  # pre-plan fakes behind POST /v2/plans


# ------------------------------------------------------------------ P2-1


class _UnhideQueue(_ActionQueue):
    """A queue whose last routable change is a probe unhide (never in last_actions)."""

    def __init__(self) -> None:
        super().__init__()
        self.changes: dict[str, tuple[int, str]] = {}

    def view_changes(self):
        return dict(self.changes)


def _state() -> SignalState:
    return SignalState()


def _low_tick(state, queue, *, end, routable, fetched, state_ms=None):
    view = replace(_view(routable, fetched_ms=fetched), state_ms=state_ms)
    return run_planner_tick(
        _snap(_window(end, [LOW] * 3, routable=routable)), queue=queue, registry=_registry(),
        rescue_due=False, fairness_due=True, cluster_view=view, signal_state=state,
    )


def test_a_rollback_unhide_holds_a_low_receiver_until_a_view_shows_it():
    state = _state()
    queue = _UnhideQueue()
    # The probe of m hid one of 2 pods; the view (state at BASE + 500) shows 1 routable.
    _low_tick(state, queue, end=BASE - GRID, routable=1, fetched=BASE - GRID + 500, state_ms=BASE - GRID + 400)
    queue.changes["m"] = (BASE + 1_500, "down")  # rollback unhide returned at BASE + 1.5 s
    held = _low_tick(state, queue, end=BASE, routable=1, fetched=BASE + 1_000, state_ms=BASE + 900)
    assert "view_pending_hold:m:down" in held.events and not _ups(held.actions)
    # Without the unhide record (the old last_actions-only gate) it scaled on the stale n.
    state2, plain = _state(), _ActionQueue()
    _low_tick(state2, plain, end=BASE - GRID, routable=1, fetched=BASE - GRID + 500)
    assert _ups(_low_tick(state2, plain, end=BASE, routable=1, fetched=BASE + 1_000).actions)


def test_an_unhide_holds_a_critical_receiver_too():
    # 2026-10-29: the guard holds every role until a view shows the call (it lasts one
    # view refresh - every routable-changing call asks for one).
    queue = _UnhideQueue()
    queue.changes["m"] = (BASE + 1_500, "down")
    assert _view_pending(queue, _view(1, fetched_ms=BASE + 1_000)) == {"m": "down"}
    held = run_planner_tick(
        _snap(_window(BASE, [(2000.0, 30.0, 20.0, 100.0)] * 3, routable=1)), queue=queue, registry=_registry(),
        rescue_due=True, fairness_due=False, cluster_view=_view(1, fetched_ms=BASE + 1_000), signal_state=_state(),
    )
    assert held.classifications["m"].state.value == "critical"
    assert not _ups(held.actions) and "view_pending_hold:m:down" in held.events


class _Client:
    async def scale_model(self, model, delta, **_kwargs):
        return {"ok": False, "error": "HTTP 500: partial"}

    async def set_routable(self, model, hidden_pods):
        return {"ok": True}

    async def set_binding_power(self, serve_id, *, awake, **_kwargs):
        return {"ok": False, "error": "HTTP 500: partial"}


def test_the_queue_records_unhides_and_failed_calls_for_the_gate():
    clock = {"now": 1_000}
    queue = ActionQueue(_Client(), now_ms=lambda: clock["now"])
    queue.submit([UnhideAction("a", ("a-1",), "formal_commit_gate_failed", "safescale")])
    asyncio.run(queue.drain_once())
    clock["now"] = 2_000
    queue.submit([ScaleAction("b", 1, "low_fairness_idle_capacity", "fairness", receiver="b")])
    asyncio.run(queue.drain_once())
    assert queue.view_changes() == {"a": (1_000, "down"), "b": (2_000, "up")}
    assert queue.last_actions() == {}  # neither is a completed scaling decision


# ------------------------------------------------------------------ P2-2


def test_state_time_is_the_controller_request_time():
    # Timestamps are never compared across machines: the SM's fetched_ms decides nothing.
    assert state_time_lower_bound({}, 1_000, 1_400) == 1_000
    assert state_time_lower_bound({"fetched_ms": 1_200}, 1_000, 1_400) == 1_000
    assert state_time_lower_bound({"fetched_ms": 900_000}, 1_000, 1_400) == 1_000
    assert state_time_lower_bound({"fetched_ms": 5}, 1_000, 1_400) == 1_000
    assert state_time_lower_bound({"fetched_ms": "x"}, 1_000, 1_400) == 1_000


def test_a_fast_sm_clock_never_moves_the_lower_bound(monkeypatch):
    # The SM node's clock runs 160 s ahead (as one cluster node does): its fetched_ms is
    # ignored, the lower bound stays the controller's request time, and an action done
    # during the request is still pending.
    times = iter([1_000_000, 1_000_400])
    monkeypatch.setattr(cluster_view_module, "wall_clock_ms", lambda: next(times))

    class _Sm:
        async def get_state(self):
            return {"bindings": [], "fetched_ms": 1_000_200 + 160_000}

    view = asyncio.run(refresh_cluster_view_once(_Sm(), _registry().topology(), ClusterViewBox())).cluster_view
    assert (view.state_ms, view.fetched_ms) == (1_000_000, 1_000_400)
    queue = _ActionQueue()
    queue.last["m"] = (1_000_300, "up")
    assert _view_pending(queue, view) == {"m": "up"}


def test_refresh_stamps_request_and_response_times(monkeypatch):
    times = iter([1_000, 1_400])
    monkeypatch.setattr(cluster_view_module, "wall_clock_ms", lambda: next(times))

    class _Sm:
        async def get_state(self):
            return {"bindings": []}

    box = ClusterViewBox()
    result = asyncio.run(refresh_cluster_view_once(_Sm(), _registry().topology(), box))
    assert (result.cluster_view.state_ms, result.cluster_view.fetched_ms) == (1_000, 1_400)


def test_an_action_done_while_the_request_was_in_flight_is_pending():
    queue = _ActionQueue()
    queue.last["m"] = (1_200, "up")  # done between request (1_000) and response (1_400)
    view = replace(_view(1, fetched_ms=1_400), state_ms=1_000)
    assert _view_pending(queue, view) == {"m": "up"}
    assert _view_pending(queue, replace(view, state_ms=1_300)) == {}


# ------------------------------------------------------------------ P2-4


def test_a_stalled_probe_is_a_capacity_rollback():
    assert "insufficient_evidence:stalled" in CAPACITY_ROLLBACK_CODES
    machine = SafeScaleStateMachine(config=SafeScaleConfig())
    machine.start_probe(model="d", pods=("d-1",), now_ms=0, window_inputs=ProbeWindowInputs(z_m=1.5, routable_pods=3))
    machine.resolve("d", status="rollback", reason="insufficient_evidence:stalled", now_ms=10_000)
    assert machine.rollback_evidence()["d"].capacity is True
    assert machine.rollback_retry_holds({"d": (1.5, 3, 30_000)}) == {"d": "same_evidence"}


# ------------------------------------------------------------------ P3-6


def test_a_stale_view_keeps_the_holds_and_only_alerts(caplog):
    # Missing fresh state is a failure (SM / Redis / network): the planner stays
    # conservative - the view-pending gate keeps holding, nothing falls back to F4.
    queue = _ActionQueue()
    queue.last["m"] = (BASE + 1_500, "up")
    stale = _low_tick(_state(), queue, end=BASE + 10 * GRID, routable=1, fetched=BASE + 1_000)
    assert "view_pending_hold:m:up" in stale.events and not _ups(stale.actions)
    assert not any("stale" in event for event in stale.events)

    alert = StaleViewAlert(30_000)
    view = replace(_view(1, fetched_ms=1_400), state_ms=1_000)
    assert alert.check(view, 31_000, None) is None
    assert alert.check(view, 31_001, "HTTP 503") == {"event": "cluster_view_stale", "age_ms": 30_001,
                                                     "last_error": "HTTP 503"}
    assert alert.check(view, 60_000, "timeout") is None  # once per stale period
    fresh = replace(view, state_ms=59_000, fetched_ms=59_100)
    assert alert.check(fresh, 60_000, None) == {"event": "cluster_view_recovered", "age_ms": 1_000}
    assert alert.check(fresh, 61_000, None) is None
    assert alert.check(replace(view, state_ms=None), 31_401, None)["age_ms"] == 30_001  # fetched_ms

    class _Failing:
        async def get_state(self):
            raise ConnectionError("sm down")

    class _Stop(Exception):
        pass

    async def stop(_s):
        raise _Stop()

    box = ClusterViewBox(view)
    cfg = type("Cfg", (), {"fairness_interval_s": 10.0, "view_stale_periods": 3})()
    with caplog.at_level("WARNING", logger="tre_controller.cluster_view"):
        try:
            asyncio.run(cluster_view_task(_Failing(), _registry().topology(), box, cfg, sleep=stop,
                                          clock_ms=lambda: 40_000))
        except _Stop:
            pass
    assert any('"cluster_view_stale"' in r.getMessage() and "sm down" in r.getMessage() for r in caplog.records)


def test_stale_periods_config():
    assert ControllerConfig.from_env({}).view_stale_periods == 3
    assert ControllerConfig.from_env({"TRE_VIEW_STALE_PERIODS": "5"}).view_stale_periods == 5
    assert StaleViewAlert(0).check(None, 10**9, "down") is None

"""Timer cleanup, independent review fixes (2026-10-02, docs/design/20261002-timer-cleanup.md).

P2-1  the O1 view-pending gate also covers probe unhides and failed / partial SM calls
      (``ActionQueue.view_changes``);
P2-2  a view's time for that gate is a LOWER bound of when its SM state was produced;
P2-4  ``insufficient_evidence:stalled`` is a capacity rollback;
P3-6  a fleet view older than N refresh periods only raises an alert (holds kept);
P3-7  two models trading a pod back and forth: every hop waits for evidence."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import (
    ClusterTopology,
    ModelSpec,
    NodeSpec,
    Registry,
    ScalingRegistryConfig,
    SloSpec,
    TrsParams,
)
from tre_controller.config import ControllerConfig
from tre_controller.loops import cluster_view_task as cluster_view_module
from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.cluster_view_task import (
    ClusterViewBox,
    StaleViewAlert,
    cluster_view_task,
    refresh_cluster_view_once,
    state_time_lower_bound,
)
from tre_controller.loops.tick import _o1_view_pending, run_planner_tick
from tre_controller.planning.planner import ClusterView, ScaleAction, UnhideAction
from tre_controller.planning.safescale import CAPACITY_ROLLBACK_CODES, ProbeWindowInputs, SafeScaleStateMachine
from tre_controller.config import SafeScaleConfig
from tre_controller.signals.trs import BreakpointWindowConfig, SignalState
from tre_sm.allocator.slots import Binding, Slot
from relay_view import expand_relays  # 2026-10-02 relay intents

from test_timer_cleanup_f4_o1_20261002 import BASE, LOW, _ActionQueue, _state
from test_o1_breakpoint_window_20261001 import GRID, O1, _registry, _snap, _ups, _view, _window


# ------------------------------------------------------------------ P2-1


class _UnhideQueue(_ActionQueue):
    """A queue whose last routable change is a probe unhide (never in last_actions)."""

    def __init__(self) -> None:
        super().__init__()
        self.changes: dict[str, tuple[int, str]] = {}

    def view_changes(self):
        return dict(self.changes)

    def routable_changes(self):
        return {m: (ms, 1) for m, (ms, _d) in self.changes.items()}


def _low_tick(state, queue, *, end, routable, fetched, state_ms=None):
    view = replace(_view(routable, fetched_ms=fetched), state_ms=state_ms)
    return run_planner_tick(
        _snap(_window(end, [LOW] * 3, routable=routable)), queue=queue, registry=_registry(),
        rescue_due=False, fairness_due=True, cluster_view=view, signal_state=state, action_cooldown=True,
    )


def test_a_rollback_unhide_holds_a_low_receiver_until_a_view_shows_it():
    state = _state()
    queue = _UnhideQueue()
    # The probe of m hid one of 2 pods; the view (state at BASE + 500) shows 1 routable.
    _low_tick(state, queue, end=BASE - GRID, routable=1, fetched=BASE - GRID + 500, state_ms=BASE - GRID + 400)
    queue.changes["m"] = (BASE + 1_500, "down")  # rollback unhide returned at BASE + 1.5 s
    held = _low_tick(state, queue, end=BASE, routable=1, fetched=BASE + 1_000, state_ms=BASE + 900)
    assert "o1_view_pending_hold:m" in held.events and not _ups(held.actions)
    # Without the unhide record (the old last_actions-only gate) it scaled on the stale n.
    state2, plain = _state(), _ActionQueue()
    _low_tick(state2, plain, end=BASE - GRID, routable=1, fetched=BASE - GRID + 500)
    assert _ups(_low_tick(state2, plain, end=BASE, routable=1, fetched=BASE + 1_000).actions)


def test_an_unhide_does_not_hold_a_critical_receiver():
    queue = _UnhideQueue()
    queue.changes["m"] = (BASE + 1_500, "down")
    tracked = {"m": {"o1_routable_tracked": True}}
    assert _o1_view_pending(queue, tracked, _view(1, fetched_ms=BASE + 1_000)) == {"m": "down"}
    # "down" = the F4 rule that lets a CRITICAL receiver scale up (tested in the F4/O1 file).


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
    # recorded for reference, the lower bound stays the controller's request time, and an
    # action done during the request is still pending.
    times = iter([1_000_000, 1_000_400])
    monkeypatch.setattr(cluster_view_module, "wall_clock_ms", lambda: next(times))

    class _Sm:
        async def get_state(self):
            return {"bindings": [], "fetched_ms": 1_000_200 + 160_000}

    view = asyncio.run(refresh_cluster_view_once(_Sm(), _registry().topology(), ClusterViewBox())).cluster_view
    assert (view.state_ms, view.fetched_ms, view.sm_fetched_ms) == (1_000_000, 1_000_400, 1_160_200)
    queue = _ActionQueue()
    queue.last["m"] = (1_000_300, "up")
    assert _o1_view_pending(queue, {"m": {"o1_routable_tracked": True}}, view) == {"m": "up"}


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
    tracked = {"m": {"o1_routable_tracked": True}}
    view = replace(_view(1, fetched_ms=1_400), state_ms=1_000)
    assert _o1_view_pending(queue, tracked, view) == {"m": "up"}
    assert _o1_view_pending(queue, tracked, replace(view, state_ms=1_300)) == {}


# ------------------------------------------------------------------ P2-4


def test_a_stalled_probe_is_a_capacity_rollback():
    assert "insufficient_evidence:stalled" in CAPACITY_ROLLBACK_CODES
    machine = SafeScaleStateMachine(config=SafeScaleConfig())
    machine.start_probe(model="d", pods=("d-1",), now_ms=0, window_inputs=ProbeWindowInputs(z_m=1.5, routable_pods=3))
    machine.resolve("d", status="rollback", reason="insufficient_evidence:stalled", now_ms=10_000)
    assert machine.rollback_evidence()["d"].capacity is True
    assert machine.rollback_retry_holds({"d": (1.5, 3, 20_000)}) == {"d": "same_evidence"}


# ------------------------------------------------------------------ P3-6


def test_a_stale_view_keeps_the_holds_and_only_alerts(caplog):
    # Missing fresh state is a failure (SM / Redis / network): the planner stays
    # conservative - the view-pending gate keeps holding, nothing falls back to F4.
    queue = _ActionQueue()
    queue.last["m"] = (BASE + 1_500, "up")
    stale = _low_tick(_state(), queue, end=BASE + 10 * GRID, routable=1, fetched=BASE + 1_000)
    assert "o1_view_pending_hold:m" in stale.events and not _ups(stale.actions)
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


# ------------------------------------------------------------------ P3-7

THETA = 100.0
HIGH = (60.0, 0.5, 0.0, 10.0)  # Z = 1.8 > tau_high


def _two_model_registry() -> Registry:
    slo = SloSpec(ttft_p95_ms=1200.0, tpot_p95_ms=100.0, e2e_p95_ms=10_000.0)
    trs = TrsParams(
        w_p=0.04, w_d=1.0, lambda_wait=2.625, qmin=1.0, ema_alpha=0.5, theta_m=THETA,
        tau_crit=0.8, tau_low=1.0, tau_high=1.25, qsat=4.0, epsat=0.05, hsat=1, ema_tau_ms=10_000.0,
    )
    specs = [ModelSpec(name=name, weights_path="/w", tp_size=1, min_replicas=1, max_replicas=4,
                       vllm_image="img", slo=slo, trs=trs) for name in ("a", "b")]
    topology = ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),))
    return Registry(topology, specs, scaling=ScalingRegistryConfig(rescue_max_step_pods=4))


def _model_window(model, end, grids, routable):
    def agg(start, gs):
        n = max(1, (end - start) // GRID)
        return ModelWindowMetrics(
            model=model, window_start_ms=start, window_end_ms=end, prompt_tokens=0.0,
            generation_tokens=float(sum(g[0] for g in gs)), avg_waiting=sum(g[2] for g in gs) / n,
            avg_running=sum(g[1] for g in gs) / n, avg_swapping=0.0, kv_cache_hit_rate=0.0,
            ttft_p95_ms=100.0, tpot_p95_ms=10.0, e2e_p95_ms=1000.0, routable_pods=routable,
            assigned_replicas=routable, per_pod={}, request_count=float(sum(g[3] for g in gs)),
        )

    full = agg(end - 3 * GRID, grids)
    return replace(full, suffix_windows=(agg(end - 2 * GRID, grids[1:]), agg(end - GRID, grids[2:])))


def _two_view(awake_a: int, fetched: int) -> ClusterView:
    """4 GPUs; a awake on the first ``awake_a`` GPUs, b on the rest; every model has a
    sleeping binding on every other GPU."""
    bindings = []
    for gpu in range(4):
        bindings.append(Binding(f"a-{gpu}", "a", Slot("node-a", (gpu,)), awake=gpu < awake_a, hidden=False))
        bindings.append(Binding(f"b-{gpu}", "b", Slot("node-a", (gpu,)), awake=gpu >= awake_a, hidden=False))
    return ClusterView(topology=_two_model_registry().topology(), bindings=tuple(bindings), fetched_ms=fetched)


def _two_tick(state, queue, *, end, awake_a, fetched, load_a, load_b):
    snapshot = MetricsSnapshot(ts_ms=end, stale=False, models={
        "a": _model_window("a", end, [load_a] * 3, awake_a),
        "b": _model_window("b", end, [load_b] * 3, 4 - awake_a),
    })
    return run_planner_tick(snapshot, queue=queue, registry=_two_model_registry(), rescue_due=False,
                            fairness_due=True, cluster_view=_two_view(awake_a, fetched), signal_state=state,
                            action_cooldown=True)


def _transfers(result) -> dict[str, int]:
    out: dict[str, int] = {}
    for action in expand_relays(result.actions):  # a relay intent = donor -n / receiver +n
        if isinstance(action, ScaleAction):
            out[action.model] = out.get(action.model, 0) + action.delta
    return out


def test_two_models_trading_a_pod_wait_for_evidence_at_every_hop():
    state = SignalState(warmup_ms=-1, breakpoint=O1)
    for model in ("a", "b"):
        state.observe_traffic(model, has_traffic=True, window_start_ms=BASE - 200_000, window_end_ms=BASE - 170_000)
    queue = _ActionQueue()
    # Hop 1: a LOW, b HIGH -> b gives a pod to a.
    first = _two_tick(state, queue, end=BASE, awake_a=2, fetched=BASE + 500, load_a=LOW, load_b=HIGH)
    assert _transfers(first) == {"b": -1, "a": 1}, first.events
    queue.last.update({"a": (BASE + 1_500, "up"), "b": (BASE + 1_500, "down")})
    # The load flips at once: a HIGH, b LOW. Every tick until a whole window follows a's
    # breakpoint (window end >= BASE + 40 s) a is no donor, so nothing moves back.
    hops = {}
    for k, (awake_a, fetched) in enumerate([(2, BASE + 1_000), (3, BASE + 2 * GRID + 500),
                                            (3, BASE + 3 * GRID + 500), (3, BASE + 4 * GRID + 500)], start=1):
        hops[k] = _two_tick(state, queue, end=BASE + k * GRID, awake_a=awake_a, fetched=fetched,
                            load_a=HIGH, load_b=LOW)
    assert _transfers(hops[1]) == {}  # the view predates the hop; b's Z still mixed
    # Hop 2: b is a LOW receiver on post-change evidence too thin to act on; hop 3: b's
    # evidence is warm, but a (the only donor) still lacks a whole post-change window.
    assert "receiver_held_breakpoint_window:b:evidence_grids" in hops[2].events
    for k in (2, 3):
        assert _transfers(hops[k]) == {}, (k, hops[k].events)
        assert "donor_suppressed_breakpoint_window:a" in hops[k].events
    # b's receiver evidence was warm from BASE + 30 s; a's whole window from BASE + 40 s.
    assert _transfers(hops[4]) == {"a": -1, "b": 1}, hops[4].events

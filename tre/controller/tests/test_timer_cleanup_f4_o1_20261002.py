"""Timer cleanup item 1 (2026-10-02, docs/design/20261002-timer-cleanup.md): the review
F4 action cooldown is only a fallback for models O1 does not track.

With O1 active the routable-count change of an executed action is a breakpoint: the
model takes part in no scale-down until a whole window follows it and acts as a
receiver only on ``min_evidence_grids`` grids of post-change evidence. The one gap -
the fleet view the tick decides on was fetched before the action completed, so no
breakpoint exists yet - is closed by the O1 view-pending gate. While O1 is suspended
(gateway clock check) or does not see the model, the F4 window-start rule holds as
before."""

from __future__ import annotations

from tre_controller.loops.tick import (
    PaperStateCache,
    _action_cooldowns,
    _o1_view_pending,
    run_planner_tick,
)
from tre_controller.planning.classify import ModelState
from tre_controller.planning.planner import PlanConfig, ScaleAction, build_plan
from tre_controller.signals.trs import SignalState

from test_o1_breakpoint_window_20261001 import (
    GRID,
    O1,
    _Queue,
    _cls,
    _registry,
    _snap,
    _ups,
    _view,
    _window,
)

#: Z = 0.9 at any replica count (tau_crit 0.8 < Z < tau_low 1.0): a LOW receiver.
LOW = (30.0, 0.5, 0.0, 10.0)
BASE = 2_000_000


class _ActionQueue(_Queue):
    """The O1 test queue plus the ActionQueue's last executed action per model."""

    def __init__(self) -> None:
        super().__init__()
        self.last: dict[str, tuple[int, str]] = {}

    def last_actions(self):
        return dict(self.last)

    def routable_changes(self):
        return {model: (ms, 1 if direction == "up" else -1) for model, (ms, direction) in self.last.items()}


def _state(*, suspended: bool = False) -> SignalState:
    state = SignalState(warmup_ms=-1, breakpoint=O1)
    state.observe_traffic("m", has_traffic=True, window_start_ms=BASE - 200_000, window_end_ms=BASE - 170_000)
    if suspended:
        state.suspend_breakpoint_window("gateway_clock_skew")
    return state


def _tick(state: SignalState, queue: _ActionQueue, *, end: int, routable: int, fetched: int):
    return run_planner_tick(
        _snap(_window(end, [LOW] * 3, routable=routable)),
        queue=queue,
        registry=_registry(),
        rescue_due=False,
        fairness_due=True,
        cluster_view=_view(routable, fetched_ms=fetched),
        signal_state=state,
        action_cooldown=True,
    )


def _low_scenario(state: SignalState) -> dict[int, object]:
    """A LOW receiver scaled up 1 -> 2 at BASE + 1.5 s; the next view (BASE + 1 s) still
    predates it, the one after (BASE + 20.5 s) shows 2 routable replicas."""
    queue = _ActionQueue()
    results = {}
    first = _tick(state, queue, end=BASE, routable=1, fetched=BASE + 500)
    assert _ups(first.actions), first.events
    queue.last["m"] = (BASE + 1_500, "up")
    results[0] = first
    results[1] = _tick(state, queue, end=BASE + GRID, routable=1, fetched=BASE + 1_000)
    for k in range(2, 6):
        results[k] = _tick(state, queue, end=BASE + k * GRID, routable=2, fetched=BASE + k * GRID + 500)
    return results


def test_o1_replaces_the_f4_hold_of_a_low_receiver():
    results = _low_scenario(_state())
    # The view predates the scale-up: no breakpoint yet -> the O1 view-pending gate holds.
    assert "o1_view_pending_hold:m" in results[1].events and not _ups(results[1].actions)
    # Breakpoint seen: the receiver waits for min_evidence_grids post-change grids.
    assert "receiver_held_breakpoint_window:m:evidence_grids" in results[2].events
    assert not _ups(results[2].actions)
    # Two post-change grids (window end BASE + 30 s): it acts again. The F4 rule would
    # still hold it here (window start BASE < the action's completion BASE + 1.5 s).
    assert _ups(results[3].actions), results[3].events
    for result in results.values():
        assert not any(event.startswith("cooldown_hold") for event in result.events)


def test_f4_still_holds_while_o1_is_suspended():
    results = _low_scenario(_state(suspended=True))
    for k in (1, 2, 3):  # every window starting before BASE + 1.5 s
        assert "cooldown_hold:m" in results[k].events, (k, results[k].events)
        assert not _ups(results[k].actions)
        assert not any(event.startswith("o1_view_pending_hold") for event in results[k].events)
    # The first window starting after the action (BASE + 10 s .. BASE + 40 s).
    assert _ups(results[4].actions), results[4].events


def test_action_cooldowns_skip_models_o1_tracks():
    queue = _ActionQueue()
    queue.last["m"] = (BASE + 1_500, "up")
    snapshot = _snap(_window(BASE + GRID, [LOW] * 3))  # window start BASE - 20 s < done
    assert _action_cooldowns(snapshot, queue) == {"m": "up"}
    assert _action_cooldowns(snapshot, queue, {"m": {"o1_routable_tracked": False}}) == {"m": "up"}
    assert _action_cooldowns(snapshot, queue, {"m": {"o1_routable_tracked": True}}) == {}


def test_view_pending_only_for_tracked_models_and_views_older_than_the_action():
    queue = _ActionQueue()
    queue.last["m"] = (BASE + 1_500, "down")
    tracked = {"m": {"o1_routable_tracked": True}}
    assert _o1_view_pending(queue, tracked, _view(1, fetched_ms=BASE + 1_000)) == {"m": "down"}
    assert _o1_view_pending(queue, tracked, _view(1, fetched_ms=BASE + 1_500)) == {}
    assert _o1_view_pending(queue, {"m": {"o1_routable_tracked": False}}, _view(1, fetched_ms=BASE)) == {}
    assert _o1_view_pending(queue, tracked, _view(1, fetched_ms=None)) == {}
    assert _o1_view_pending(queue, tracked, None) == {}


def test_contexts_mark_o1_tracking_and_a_stale_hold_clears_it():
    from tre_controller.loops.tick import _model_contexts

    state = _state()
    ctx, _ = _model_contexts(_snap(_window(BASE, [LOW] * 3)), _registry(), signal_state=state,
                             cluster_view=_view(1, fetched_ms=BASE + 500))
    assert ctx["m"]["o1_routable_tracked"] is True
    # No fleet view of the model: O1 cannot see its routable changes.
    ctx, _ = _model_contexts(_snap(_window(BASE + GRID, [LOW] * 3)), _registry(), signal_state=state)
    assert ctx["m"]["o1_routable_tracked"] is False
    state.suspend_breakpoint_window("gateway_clock_skew")
    ctx, _ = _model_contexts(_snap(_window(BASE + 2 * GRID, [LOW] * 3)), _registry(), signal_state=state,
                             cluster_view=_view(1, fetched_ms=BASE + 2 * GRID + 500))
    assert ctx["m"]["o1_routable_tracked"] is False
    cache = PaperStateCache(max_stale_windows=3)
    cache.apply("m", {"o1_routable_tracked": True, "routable_pods": 1}, tokens_available=True)
    held, events = cache.apply("m", {"routable_pods": 1}, tokens_available=False)
    assert events == ("paper_state_stale_hold:m",) and held["o1_routable_tracked"] is False


def _plan(classifications, *, view_pending=None, cooldowns=None, floor_holds=None):
    contexts = {item.model_name: {"routable_pods": 3, "assigned_replicas": 3} for item in classifications}
    return build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={item.model_name: 3 for item in classifications},
        idle_gpus=1,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4, suppress_hot_proactive_probe=True),
        cooldowns=cooldowns,
        view_pending=view_pending,
        floor_holds=floor_holds,
    )


def _deltas(plan):
    out = {}
    for action in plan.actions:
        if isinstance(action, ScaleAction):
            out[action.model] = out.get(action.model, 0) + action.delta
    return out


def test_planner_view_pending_gate_uses_the_f4_direction_rules():
    low = [_cls("r", ModelState.LOW, 0.9)]
    assert _deltas(_plan(low)) == {"r": 1}
    held = _plan(low, view_pending={"r": "up"})
    assert _deltas(held) == {} and "o1_view_pending_hold:r" in held.events
    assert _deltas(_plan(low, view_pending={"r": "down"})) == {}
    # A CRITICAL receiver after its own scale-down is never held (safety, as F4).
    critical = [_cls("r", ModelState.CRITICAL, 0.4)]
    assert _deltas(_plan(critical, view_pending={"r": "down"})) == {"r": 1}
    # Donors: an idle model whose scale-down the view does not show yet gives nothing.
    idle = [_cls("i", ModelState.IDLE, 10.0)]
    assert _deltas(_plan(idle)) == {"i": -1}
    assert _deltas(_plan(idle, view_pending={"i": "down"})) == {}


def test_floor_violation_hold_is_unchanged():
    idle = [_cls("i", ModelState.IDLE, 10.0)]
    held = _plan(idle, floor_holds={"i"})
    assert _deltas(held) == {} and held.events == ["floor_violation_hold:i"]
    # Never out of a scale-up.
    assert _deltas(_plan([_cls("r", ModelState.LOW, 0.9)], floor_holds={"r"})) == {"r": 1}

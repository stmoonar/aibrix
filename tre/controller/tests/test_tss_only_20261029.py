"""TSS only (2026-10-29, design docs/design/20261029-tss-only.md).

* Z is always the whole sliding window plus the EMA: right after a routable-count change
  the model decides on the window that still mixes both replica counts, at the next
  rescue section (no breakpoint hold, no saturation / exempt path, no window-start wait).
* The only per-model hold is a correctness guard: an action in flight, or one the fleet
  view does not show yet (view-pending), blocks the model's next action.
"""
from __future__ import annotations

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
from section_ticks import run_fairness_tick, run_rescue_tick
from tre_controller.loops.tick import compute_model_signal, serving_window
from tre_controller.planning.planner import ClusterView, ScaleAction
from tre_controller.signals.trs import SignalState
from tre_sm.allocator.slots import Binding, Slot

GRID = 10_000
W = 30_000
THETA = 100.0
BASE = 2_000_000
#: Real overload: Q = 30 + 2.625 * 20 = 82.5, Z = 3 * 2000 / 82.5 / 100 = 0.727 < tau_crit.
BURST = (2000.0, 30.0, 20.0, 100.0)
#: Z = 0.9 at any replica count (tau_crit 0.8 < Z < tau_low 1.0): a LOW receiver.
LOW_LOAD = (30.0, 0.5, 0.0, 10.0)


# ------------------------------------------------------------------- fixtures
# (also used by test_controller_transfer_20261002 / test_timer_cleanup_review_20261002)


def _registry(tau_ms: float | None = 10_000.0, **scaling) -> Registry:
    slo = SloSpec(ttft_p95_ms=1200.0, tpot_p95_ms=100.0, e2e_p95_ms=10_000.0)
    trs = TrsParams(
        w_p=0.04, w_d=1.0, lambda_wait=2.625, qmin=1.0, ema_alpha=0.5, theta_m=THETA,
        tau_crit=0.8, tau_low=1.0, tau_high=1.25, qsat=4.0, epsat=0.05, hsat=1, ema_tau_ms=tau_ms,
    )
    spec = ModelSpec(
        name="m", weights_path="/w", tp_size=1, min_replicas=1, max_replicas=4,
        vllm_image="img", slo=slo, trs=trs,
    )
    topology = ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),))
    return Registry(topology, [spec], scaling=ScalingRegistryConfig(**scaling))


def _agg(start: int, end: int, grids: list[tuple[float, float, float, float]], routable: int) -> ModelWindowMetrics:
    """A window over ``grids`` = [(generation tokens, running, waiting, requests)] of the
    10 s grids in ``(start, end]`` (token total, instant mean over the grids)."""
    n = max(1, (end - start) // GRID)
    return ModelWindowMetrics(
        model="m", window_start_ms=start, window_end_ms=end,
        prompt_tokens=0.0, generation_tokens=float(sum(g[0] for g in grids)),
        avg_waiting=sum(g[2] for g in grids) / n, avg_running=sum(g[1] for g in grids) / n,
        avg_swapping=0.0, kv_cache_hit_rate=0.0, ttft_p95_ms=100.0, tpot_p95_ms=10.0, e2e_p95_ms=1000.0,
        routable_pods=routable, assigned_replicas=routable, per_pod={},
        request_count=float(sum(g[3] for g in grids)),
    )


def _window(end: int, grids: list, *, routable: int = 1) -> ModelWindowMetrics:
    """The 30 s window ending at ``end`` (3 grids, oldest first)."""
    assert len(grids) == 3
    return _agg(end - W, end, grids, routable)


def _snap(window: ModelWindowMetrics) -> MetricsSnapshot:
    return MetricsSnapshot(ts_ms=window.window_end_ms, models={"m": window}, stale=False)


class _Queue:
    def __init__(self) -> None:
        self.submitted: list = []
        self.last: dict[str, tuple[int, str]] = {}
        self.inflight: set[str] = set()

    def inflight_models(self) -> set[str]:
        return set(self.inflight)

    def submit(self, actions) -> object:
        self.submitted.extend(actions)
        return object()

    def last_actions(self):
        return dict(self.last)


def _ups(actions) -> list[ScaleAction]:
    return [a for a in actions if isinstance(a, ScaleAction) and a.delta > 0]


def _view(awake: int, *, fetched_ms: int | None, total: int = 4, state_ms: int | None = None) -> ClusterView:
    bindings = tuple(
        Binding(serve_id=f"m-{i}", model="m", slot=Slot("node-a", (i,)), awake=i < awake, hidden=False)
        for i in range(total)
    )
    return ClusterView(topology=_registry().topology(), bindings=bindings, fetched_ms=fetched_ms, state_ms=state_ms)


# ------------------------------------------------- (a) the whole window, no hold


def test_after_a_routable_change_z_is_the_whole_window_and_the_model_acts_at_once():
    state, queue, registry = SignalState(), _Queue(), _registry()
    first = run_rescue_tick(_snap(_window(BASE, [BURST] * 3, routable=1)), queue=queue, registry=registry,
                            cluster_view=_view(1, fetched_ms=BASE + 500), signal_state=state)
    assert [(a.model, a.delta) for a in _ups(first.actions)] == [("m", 1)]
    # The wake completed (BASE + 1 s); the next view shows it (2 routable). The window
    # still holds the old replica count's grids; a suffix after the change would read
    # very differently - the decision uses the whole window.
    queue.last["m"] = (BASE + 1_000, "up")
    window = _window(BASE + 5_000, [BURST] * 3, routable=2)
    suffix = _agg(BASE + 5_000 - GRID, BASE + 5_000, [(10.0, 30.0, 20.0, 1.0)], 2)
    window = replace(window, suffix_windows=(suffix,))
    view = _view(2, fetched_ms=BASE + 4_000, state_ms=BASE + 3_900)
    second = run_rescue_tick(_snap(window), queue=queue, registry=registry, cluster_view=view, signal_state=state)

    expected = compute_model_signal(
        "m", serving_window(window, view), registry.model("m"), signal_source="zm", signal_state=None,
    ).result.TRS_raw
    ctx = second.model_contexts["m"]
    assert ctx["trs_raw"] == expected and ctx["routable_pods"] == 2
    assert second.classifications["m"].state.value == "critical"
    assert [(a.model, a.delta) for a in _ups(second.actions)] == [("m", 1)]
    assert not any("hold" in event or "breakpoint" in event or "suppressed" in event for event in second.events)
    for gone in ("signal_warm", "signal_hold_reason", "signal_full_window", "o1_queue_rise"):
        assert gone not in ctx


# ------------------------------------------------- (d) in flight / view-pending


def test_a_model_whose_last_call_the_view_does_not_show_takes_no_new_action():
    queue = _Queue()
    queue.last["m"] = (BASE + 1_500, "up")  # returned after the view's state was produced
    held = run_rescue_tick(_snap(_window(BASE, [BURST] * 3, routable=1)), queue=queue, registry=_registry(),
                           cluster_view=_view(1, fetched_ms=BASE + 1_000, state_ms=BASE + 900),
                           signal_state=SignalState())
    assert not _ups(held.actions) and "view_pending_hold:m:up" in held.events
    # A newer view shows the call: the model acts again (no window / EMA wait).
    shown = run_rescue_tick(_snap(_window(BASE + 1_000, [BURST] * 3, routable=2)), queue=queue,
                            registry=_registry(), cluster_view=_view(2, fetched_ms=BASE + 2_100, state_ms=BASE + 2_000),
                            signal_state=SignalState())
    assert [(a.model, a.delta) for a in _ups(shown.actions)] == [("m", 1)]


def test_a_model_with_an_action_in_flight_takes_no_new_action():
    queue = _Queue()
    queue.inflight.add("m")
    for tick in (run_rescue_tick, run_fairness_tick):
        result = tick(_snap(_window(BASE, [BURST] * 3, routable=1)), queue=queue, registry=_registry(),
                      cluster_view=_view(1, fetched_ms=BASE + 500), signal_state=SignalState())
        assert result.actions == ()

"""Onset saturation rescue (2026-10-02, design docs/design/20261002-saturation-onset-rescue.md).

While a model's TSS cannot decide (window numerator zero, or the O1 evidence gate holds
it), a model whose engines are full - requests waiting in vLLM, or KV cache >= 0.9, on
the latest gateway sample - on 2 consecutive metrics windows is a CRITICAL receiver;
the fast loop doubles its routable count (bounded), and every further step needs the
condition again after the routable count changed."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics, PodWindowMetrics
from tre_common.registry import (
    ClusterTopology,
    ModelSpec,
    NodeSpec,
    Registry,
    ScalingRegistryConfig,
    SloSpec,
    TrsParams,
    parse_scaling_config,
)
from tre_common.window_pods import aggregate_pods
from tre_controller.loops.decision_snapshot import _model_states
from tre_controller.loops.tick import _scaling_options, run_planner_tick
from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, TauThresholds
from tre_controller.planning.planner import ClusterView, PlanConfig, ScaleAction, build_plan
from tre_controller.signals.saturation import (
    SaturationRescueConfig,
    SaturationSample,
    SaturationTracker,
    eligibility_reason,
    saturation_sample,
    saturation_target,
)
from tre_controller.signals.trs import BreakpointWindowConfig, SignalState
from tre_controller.store.metrics_store import MetricsStore
from tre_sm.allocator.slots import Binding, Slot

GRID = 10_000
W = 30_000
THETA = 100.0
BASE = 1_000_000


# ------------------------------------------------------------------- fixtures


def _registry(max_awake: int = 4, **scaling) -> Registry:
    slo = SloSpec(ttft_p95_ms=1200.0, tpot_p95_ms=100.0, e2e_p95_ms=10_000.0)
    trs = TrsParams(
        w_p=0.04, w_d=1.0, lambda_wait=2.625, qmin=1.0, ema_alpha=0.5, theta_m=THETA,
        tau_crit=0.8, tau_low=1.0, tau_high=1.25, qsat=4.0, epsat=0.05, hsat=1, ema_tau_ms=10_000.0,
    )
    spec = ModelSpec(
        name="m", weights_path="/w", tp_size=1, min_replicas=1, max_replicas=8,
        vllm_image="img", slo=slo, trs=trs, max_awake_replicas=max_awake,
    )
    topology = ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=8, two_gpu_slots=((0, 1), (2, 3), (4, 5), (6, 7))),))
    return Registry(topology, [spec], scaling=ScalingRegistryConfig(**scaling))


def _pod(name: str, end: int, grids: list, latest: tuple[float, float | None], start: int) -> PodWindowMetrics:
    """One pod over ``grids`` = [(generation tokens, running, waiting, completed requests)]
    of the 10 s grids in ``(start, end]``; ``latest`` = its newest sample (waiting, kv)."""
    n = max(1, (end - start) // GRID)
    return PodWindowMetrics(
        pod=name, prompt_tokens=0.0, generation_tokens=float(sum(g[0] for g in grids)),
        avg_waiting=sum(g[2] for g in grids) / n, avg_running=sum(g[1] for g in grids) / n,
        avg_swapping=0.0, kv_cache_hit_rate=0.0, ttft_p95_ms=100.0, tpot_p95_ms=10.0, e2e_p95_ms=1000.0,
        request_count=float(sum(g[3] for g in grids)),
        latest_waiting=latest[0], latest_running=float(grids[-1][1]), latest_gpu_cache=latest[1],
        latest_instant_ms=end,
    )


def _window(end: int, grids: list, latest: tuple[float, float | None], *, awake: int = 1) -> ModelWindowMetrics:
    """The 30 s window ending at ``end`` (3 grids per awake pod, oldest first) with its
    20 s / 10 s suffixes, aggregated like the MetricsStore."""
    assert len(grids) == 3

    def agg(start: int, part: list) -> ModelWindowMetrics:
        pods = {f"m-{i}": _pod(f"m-{i}", end, part, latest, start) for i in range(awake)}
        return aggregate_pods("m", start, end, pods)

    full = agg(end - W, grids)
    return replace(full, suffix_windows=(agg(end - 2 * GRID, grids[1:]), agg(end - GRID, grids[2:])))


def _snap(window: ModelWindowMetrics) -> MetricsSnapshot:
    return MetricsSnapshot(ts_ms=window.window_end_ms, models={"m": window}, stale=False)


def _view(awake: int, *, fetched_ms: int, total: int = 8) -> ClusterView:
    bindings = tuple(
        Binding(serve_id=f"m-{i}", model="m", slot=Slot("node-a", (i,)), awake=i < awake, hidden=False)
        for i in range(total)
    )
    return ClusterView(topology=_registry().topology(), bindings=bindings, fetched_ms=fetched_ms)


class _Queue:
    def __init__(self):
        self.submitted: list = []

    def inflight_models(self) -> set[str]:
        return set()

    def submit(self, actions) -> object:
        self.submitted.extend(actions)
        return object()


def _state(**config) -> SignalState:
    return SignalState(
        breakpoint=BreakpointWindowConfig(enabled=True, onset_guard=False, grid_ms=GRID, min_evidence_grids=2),
        saturation=SaturationTracker(SaturationRescueConfig(**config)),
    )


def _tick(state, window, *, awake, registry=None, queue=None):
    return run_planner_tick(
        _snap(window), queue=queue or _Queue(), registry=registry or _registry(), rescue_due=True,
        fairness_due=False, cluster_view=_view(awake, fetched_ms=window.window_end_ms + 4_000),
        signal_state=state,
    )


def _ups(actions) -> list[ScaleAction]:
    return [a for a in actions if isinstance(a, ScaleAction) and a.delta > 0]


def _planned(result) -> int:
    return sum(a.delta for a in _ups(result.actions))


IDLE = (0.0, 0.0, 0.0, 0.0)
#: Nothing completed yet, the engine filling: 40 running, 6 waiting.
STARTING = (0.0, 40.0, 6.0, 0.0)
#: A single long request (or a stuck one): running, nothing waiting, nothing completed.
LONG_ONE = (0.0, 1.0, 0.0, 0.0)
#: Real overload after the first completions: Q = 30 + 2.625*20 = 82.5, Z ~ 0.36.
BURST = (1000.0, 30.0, 20.0, 40.0)
#: Healthy steady traffic: high throughput, short queue (Z ~ 2.7).
HEALTHY = (800.0, 3.0, 0.0, 40.0)


# ----------------------------------------------------------- numerator zero


def test_numerator_zero_and_waiting_on_two_windows_is_critical():
    state = _state()
    queue = _Queue()
    e1, e2 = BASE + 3 * GRID, BASE + 4 * GRID
    first = _tick(state, _window(e1, [IDLE, STARTING, STARTING], (6.0, 0.5)), awake=1, queue=queue)
    assert not first.actions
    assert first.classifications["m"].state != ModelState.CRITICAL
    assert any(e.startswith("saturation_pending:m:1/2:reason=numerator_zero:waiting=6") for e in first.events)
    # A re-read of the same window (second rescue tick) does not count again.
    again = _tick(state, _window(e1, [IDLE, STARTING, STARTING], (6.0, 0.5)), awake=1, queue=queue)
    assert not again.actions and again.model_contexts["m"]["saturation_ticks"] == 1

    second = _tick(state, _window(e2, [STARTING, STARTING, STARTING], (6.0, 0.5)), awake=1, queue=queue)
    cls = second.classifications["m"]
    assert cls.state == ModelState.CRITICAL and cls.role == ModelRole.RECEIVER and cls.saturation_rescue
    assert _planned(second) == 1  # n = 1 -> target 2
    assert [a.reason for a in _ups(second.actions)] == ["critical_sleeping_capacity"]
    event = next(e for e in second.events if e.startswith("saturation_rescue:m:"))
    assert event.startswith("saturation_rescue:m:n=1:target=2:waiting=6:kv=0.50:reason=numerator_zero:ticks=2")
    assert any(e.startswith("rescue_target:m:n=1:z=none:desired=2:covered=1:planned=1") for e in second.events)
    states = _model_states(second.model_contexts, second.classifications, _snap(_window(e2, [IDLE] * 3, (0, 0))))
    assert states["m"]["saturation_rescue"] is True and states["m"]["saturation_ticks"] == 2
    assert states["m"]["saturation_reason"] == "numerator_zero" and states["m"]["saturation_waiting"] == 6.0
    # The second rescue tick of the same window plans nothing more.
    assert not _tick(state, _window(e2, [STARTING] * 3, (6.0, 0.5)), awake=1, queue=queue).actions


def test_one_saturated_window_does_not_trigger():
    state = _state()
    _tick(state, _window(BASE + 3 * GRID, [IDLE, IDLE, STARTING], (6.0, 0.5)), awake=1)
    # The next window drains (nothing waiting, KV low): the count restarts.
    calm = _tick(state, _window(BASE + 4 * GRID, [IDLE, STARTING, LONG_ONE], (0.0, 0.2)), awake=1)
    assert not calm.actions and calm.model_contexts["m"]["saturation_ticks"] == 0
    again = _tick(state, _window(BASE + 5 * GRID, [STARTING, LONG_ONE, STARTING], (6.0, 0.5)), awake=1)
    assert not again.actions and again.model_contexts["m"]["saturation_ticks"] == 1


def test_numerator_zero_single_long_request_stays_healthy():
    """Running only, nothing waiting, KV < 0.9 (one long or stuck request): the v2 idle
    rule stands - no receiver, however long it lasts."""
    state = _state()
    for i in range(3, 9):
        result = _tick(state, _window(BASE + i * GRID, [LONG_ONE] * 3, (0.0, 0.3)), awake=1)
        assert not result.actions
        assert result.classifications["m"].state != ModelState.CRITICAL
        assert not result.classifications["m"].saturation_rescue
        assert result.model_contexts["m"]["saturation_ticks"] == 0
        assert result.model_contexts["m"]["saturation_reason"] == "numerator_zero"


def test_disabled_keeps_the_tss_rules():
    state = _state(enabled=False)
    for i in range(3, 7):
        result = _tick(state, _window(BASE + i * GRID, [STARTING] * 3, (50.0, 1.0)), awake=1)
        assert not result.actions and "saturation_ticks" not in result.model_contexts["m"]


# ------------------------------------------------------------------ O1 hold


def test_o1_hold_with_full_kv_triggers():
    """First completions: the onset is a breakpoint, O1 holds the model for two grids
    (no_complete_grid, evidence_grids); KV >= 0.9 on both -> rescue on the second."""
    state = _state()
    e1, e2 = BASE + 3 * GRID, BASE + 4 * GRID
    first = _tick(state, _window(e1, [IDLE, IDLE, BURST], (0.0, 0.95)), awake=1)
    assert first.model_contexts["m"]["signal_hold_reason"] == "no_complete_grid"
    assert not first.actions and first.model_contexts["m"]["saturation_reason"] == "o1_hold"
    second = _tick(state, _window(e2, [IDLE, BURST, BURST], (0.0, 0.95)), awake=1)
    assert second.model_contexts["m"]["signal_hold_reason"] == "evidence_grids"
    assert second.classifications["m"].saturation_rescue and _planned(second) == 1
    assert any(":reason=o1_hold:ticks=2" in e for e in second.events if e.startswith("saturation_rescue:m:"))


def test_after_o1_releases_the_tss_decides_alone():
    """KV high only once the TSS is warm: the saturation path never applies (a warm,
    healthy TSS keeps the model healthy whatever the KV cache says)."""
    state = _state()
    _tick(state, _window(BASE + 3 * GRID, [IDLE, IDLE, HEALTHY], (0.0, 0.5)), awake=1)
    _tick(state, _window(BASE + 4 * GRID, [IDLE, HEALTHY, HEALTHY], (0.0, 0.5)), awake=1)
    for i in range(5, 8):
        result = _tick(state, _window(BASE + i * GRID, [HEALTHY] * 3, (0.0, 0.97)), awake=1)
        ctx = result.model_contexts["m"]
        assert ctx["signal_warm"] is True and ctx["saturation_reason"] is None and ctx["saturation_ticks"] == 0
        assert not result.actions and not result.classifications["m"].saturation_rescue


def test_eligibility_reason():
    assert eligibility_reason({"tss_defined": True, "Y_m": 0.0, "signal_warm": True}) == "numerator_zero"
    assert eligibility_reason({"tss_defined": True, "Y_m": 5.0, "signal_warm": False}) == "o1_hold"
    assert eligibility_reason({"tss_defined": True, "Y_m": 5.0, "signal_warm": True}) is None
    assert eligibility_reason({"tss_defined": True, "Y_m": 5.0}) is None  # pre-O1 context, warm
    assert eligibility_reason({"Y_m": None, "signal_unavailable_reason": "tokens_missing"}) is None
    assert eligibility_reason({"tss_defined": True, "Y_m": 0.0, "signal_source": "running"}) is None


# --------------------------------------------------------- bounded doubling


def test_bounded_doubling_needs_two_fresh_windows_after_each_step():
    state = _state()
    queue = _Queue()
    sat = [STARTING] * 3

    def tick(i, awake):
        return _tick(state, _window(BASE + i * GRID, sat, (8.0, 0.6), awake=awake), awake=awake, queue=queue)

    assert not tick(3, 1).actions
    step1 = tick(4, 1)
    assert _planned(step1) == 1  # 1 -> 2
    assert not tick(5, 1).actions  # the wake has not landed yet: no counting
    landed = tick(6, 2)  # routable 2 seen on this window: its sample may predate it
    assert not landed.actions and landed.model_contexts["m"]["saturation_ticks"] == 0
    once = tick(7, 2)
    assert not once.actions and once.model_contexts["m"]["saturation_ticks"] == 1
    step2 = tick(8, 2)
    assert _planned(step2) == 2  # 2 -> 4
    assert any(e.startswith("saturation_rescue:m:n=2:target=4") for e in step2.events)
    tick(9, 4)
    tick(10, 4)
    capped = tick(11, 4)  # 4 = max_awake_replicas: nothing more
    assert not capped.actions


def test_step_is_capped_by_max_awake_replicas():
    cfg = PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4, saturation_max_step_factor=2.0)
    for n, planned in ((1, 1), (2, 2), (3, 1)):
        ctx = {"m": {"assigned_replicas": 8, "routable_pods": n, "awake_replicas": n, "signal_warm": False,
                     "saturation_rescue": True}}
        plan = build_plan(model_contexts=ctx, classifications=[_sat_cls("m")], model_replicas={"m": 8},
                          idle_gpus=8, cfg=cfg)
        assert sum(a.delta for a in _ups(plan.actions)) == planned, n
    assert saturation_target(1, 2.0, 8) == 2 and saturation_target(3, 2.0, 8) == 6
    assert saturation_target(3, 1.0, 8) == 4  # factor 1: still n + 1
    assert saturation_target(5, 2.0, 8) == 8


def test_zero_z_never_jumps_to_the_cap():
    """Unlike C1's ceil(n * tau_crit / Z) (a Z near 0 asks for the cap), the step is n -> 2n,
    and the O1 low-evidence cap does not apply to it (its own re-confirmation bounds it)."""
    cfg = PlanConfig(min_replicas_per_model=1, max_replicas_per_model=8, rescue_max_step_pods=4,
                     partial_window_max_step=1, partial_window_lowevidence_requests=10)
    ctx = {"m": {"assigned_replicas": 8, "routable_pods": 2, "awake_replicas": 2, "signal_warm": False,
                 "signal_full_window": False, "signal_evidence_requests": 0.0, "saturation_rescue": True}}
    plan = build_plan(model_contexts=ctx, classifications=[_sat_cls("m", z=0.01)], model_replicas={"m": 8},
                      idle_gpus=8, cfg=cfg)
    assert sum(a.delta for a in _ups(plan.actions)) == 2  # 2 -> 4, not 2 -> 6 and not 2 -> 3
    assert not any(e.startswith("rescue_low_evidence_step") for e in plan.events)


def test_await_releases_after_a_step_that_never_lands():
    tracker = SaturationTracker(SaturationRescueConfig(await_timeout_ms=3 * GRID))
    sample = SaturationSample(waiting=5.0, kv=0.5, pods=1, sample_ms=0)

    def obs(i):
        return tracker.observe("m", window_end_ms=i * GRID, routable=1, reason="numerator_zero", sample=sample)

    obs(1)
    assert obs(2).fire
    tracker.note_step("m", window_end_ms=2 * GRID, routable=1)
    assert not obs(3).fire and obs(3).awaiting_step
    assert obs(4).ticks == 0
    assert obs(5).ticks == 0  # timeout reached on this window: counting restarts after it
    assert obs(6).ticks == 1 and obs(7).fire


# ------------------------------------------------------------ capacity order


def _sat_cls(model: str, z: float | None = None) -> ModelClassification:
    return ModelClassification(
        model_name=model, state=ModelState.CRITICAL, role=ModelRole.RECEIVER, Z_m=z, eta_m=None, trs=0.0,
        theta_m=THETA, tau=TauThresholds.from_control(0.2, 0.25), saturation_rescue=True,
    )


def _donor_cls(model: str) -> ModelClassification:
    return ModelClassification(
        model_name=model, state=ModelState.HIGH, role=ModelRole.DONOR, Z_m=3.0, eta_m=500.0, trs=300.0,
        theta_m=THETA, tau=TauThresholds.from_control(0.2, 0.25), donor_tier="surplus",
    )


def _two_model_plan(idle_gpus: int):
    cfg = PlanConfig(min_replicas_per_model=1, max_replicas_per_model=8)
    ctx = {
        "r": {"assigned_replicas": 1, "routable_pods": 1, "awake_replicas": 1, "signal_warm": False,
              "saturation_rescue": True},
        "d": {"assigned_replicas": 3, "routable_pods": 3, "awake_replicas": 3, "signal_warm": True},
    }
    return build_plan(model_contexts=ctx, classifications=[_sat_cls("r"), _donor_cls("d")],
                      model_replicas={"r": 1, "d": 3}, idle_gpus=idle_gpus, cfg=cfg)


def test_capacity_order_free_capacity_before_any_donor():
    plan = _two_model_plan(idle_gpus=2)
    reasons = {(a.model, a.reason) for a in plan.actions if isinstance(a, ScaleAction)}
    assert ("r", "critical_idle_capacity") in reasons
    assert not any(a.model == "d" and a.delta < 0 and a.reason == "critical_donor_immediate"
                   for a in plan.actions if isinstance(a, ScaleAction))


def test_without_free_capacity_a_high_donor_is_released_immediately():
    plan = _two_model_plan(idle_gpus=0)
    donor = [a for a in plan.actions if isinstance(a, ScaleAction) and a.model == "d"]
    assert donor and donor[0].delta == -1 and donor[0].reason == "critical_donor_immediate"
    assert _ups(plan.actions)[0].model == "r"


def test_a_saturation_receiver_is_not_dropped_as_incomplete():
    plan = _two_model_plan(idle_gpus=2)
    assert not any(e.startswith("paper_state_incomplete_drop:r") for e in plan.events)


# ------------------------------------------------------------- sample / store


def test_sample_uses_routable_pods_latest_values_and_skips_stale_or_hidden():
    end = BASE + 3 * GRID
    pods = {
        "a": _pod("a", end, [STARTING] * 3, (0.0, 0.95), end - W),
        "b": _pod("b", end, [STARTING] * 3, (3.0, 0.85), end - W),
        "h": _pod("h", end, [STARTING] * 3, (40.0, 1.0), end - W),
        "old": replace(_pod("old", end, [STARTING] * 3, (9.0, 1.0), end - W), latest_instant_ms=end - 2 * GRID),
    }
    window = aggregate_pods("m", end - W, end, pods)
    sample = saturation_sample(window, hidden_pods={"h"}, fresh_after_ms=end - GRID)
    assert sample.pods == 2 and sample.waiting == 3.0 and sample.kv == pytest.approx(0.9)
    assert saturation_sample(aggregate_pods("m", end - W, end, {}), fresh_after_ms=end - GRID) is None


def test_store_carries_each_pods_newest_instant_sample():
    from test_metrics_store import FakeRedis, add_doc, hist_doc

    redis = FakeRedis()
    pod = "default/pod-a"
    redis.sadd("tre:v2:pods:dsqwen-7b", pod)
    for i, ts in enumerate(range(0, 30_001, GRID)):
        add_doc(redis, "tre:v2:hist:" + pod, ts, hist_doc("pod-a", 10 * i, i, 0.1, i, {"0.5": i}))
        add_doc(redis, "tre:v2:inst:" + pod, ts, {
            "pod_name": "pod-a",
            "model_metrics": {
                "dsqwen-7b/num_requests_waiting": 10 * i,
                "dsqwen-7b/num_requests_running": i,
                "dsqwen-7b/kv_cache_usage_perc": 0.3 * i,
            },
        })
    from tre_common.registry import load_registry
    from pathlib import Path

    registry = load_registry(str(Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml"))
    store = MetricsStore(redis, registry, instant_sample_interval_ms=GRID)
    window = store.read_model_window("dsqwen-7b", 0, 30_000, use_cache=False, start_exclusive=True)
    metrics = window.per_pod["pod-a"]
    assert metrics.latest_waiting == 30.0 and metrics.latest_running == 3.0
    assert metrics.latest_gpu_cache == pytest.approx(0.9) and metrics.latest_instant_ms == 30_000
    assert metrics.avg_waiting == pytest.approx(20.0)  # the window mean is unchanged


# ------------------------------------------------------------------- config


def test_scaling_config_defaults_and_validation():
    defaults = parse_scaling_config({})
    assert (defaults.saturation_rescue, defaults.saturation_kv_threshold, defaults.saturation_consecutive_ticks,
            defaults.saturation_max_step_factor) == (True, 0.9, 2, 2.0)
    custom = parse_scaling_config({"saturation_rescue": False, "saturation_kv_threshold": 0.95,
                                   "saturation_consecutive_ticks": 3, "saturation_max_step_factor": 1.5})
    assert (custom.saturation_rescue, custom.saturation_kv_threshold, custom.saturation_consecutive_ticks,
            custom.saturation_max_step_factor) == (False, 0.95, 3, 1.5)
    for bad in ({"saturation_rescue": "yes"}, {"saturation_kv_threshold": 0}, {"saturation_kv_threshold": 1.5},
                {"saturation_kv_threshold": True}, {"saturation_kv_threshold": "x"},
                {"saturation_consecutive_ticks": 0}, {"saturation_consecutive_ticks": 1.5},
                {"saturation_max_step_factor": 0.5}, {"saturation_max_step_factor": float("nan")}):
        with pytest.raises(ValueError):
            parse_scaling_config(bad)
    registry = _registry(saturation_kv_threshold=0.8, saturation_consecutive_ticks=3, saturation_max_step_factor=3)
    config = SaturationRescueConfig.from_registry(registry, grid_ms=GRID)
    assert (config.enabled, config.kv_threshold, config.consecutive_ticks, config.max_step_factor) == (True, 0.8, 3, 3.0)
    assert config.grid_ms == GRID and config.await_timeout_ms == 3 * GRID
    assert _scaling_options(registry)["saturation_max_step_factor"] == 3.0
    assert not SaturationRescueConfig.from_registry(_registry(saturation_rescue=False), grid_ms=GRID).enabled


# ------------------------------------- partial_max_step wiring (2026-10-02 check)


def test_partial_max_step_is_wired_from_the_registry_through_the_tick():
    """``scaling.breakpoint_partial_max_step`` reaches the planner (ScalingRegistryConfig
    -> _scaling_options -> PlanConfig) and ``signal_evidence_requests`` reaches it through
    the tick's context: a warm partial window with < 10 completed requests adds +1; with
    >= 10 the whole C1 deficit (A-smoke-tre 18:02: n = 1, Z = 0.298, ~hundreds of
    completed requests in the 20 s suffix -> planned 2 is the uncapped case)."""

    def run(requests_per_grid: float) -> int:
        state = SignalState(breakpoint=BreakpointWindowConfig(enabled=True, grid_ms=GRID, min_evidence_grids=2,
                                                              min_evidence_requests=3))
        registry = _registry(rescue_max_step_pods=4, breakpoint_partial_max_step=1,
                             breakpoint_lowevidence_requests=10, saturation_rescue=False)
        burst = (200.0, 30.0, 20.0, requests_per_grid)
        planned = []
        for i, grids in ((3, [IDLE, IDLE, burst]), (4, [IDLE, burst, burst]), (5, [burst, burst, burst])):
            result = _tick(state, _window(BASE + i * GRID, grids, (20.0, 1.0)), awake=1, registry=registry)
            planned.append(_planned(result))
        return planned

    assert run(2.0) == [0, 0, 1]  # 4 completed in the 20 s suffix: +1 (rescue_low_evidence_step)
    assert run(10.0) == [0, 0, 3]  # 20 completed: the whole deficit (capped by max_awake 4)

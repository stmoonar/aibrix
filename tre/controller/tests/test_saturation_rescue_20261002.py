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
    PodSample,
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


def _window(
    end: int, grids: list, latest: tuple[float, float | None], *, awake: int = 1,
    latest_by_pod: dict | None = None,
) -> ModelWindowMetrics:
    """The 30 s window ending at ``end`` (3 grids per awake pod, oldest first) with its
    20 s / 10 s suffixes, aggregated like the MetricsStore. ``latest_by_pod`` overrides
    single pods' newest sample (waiting, kv)."""
    assert len(grids) == 3

    def agg(start: int, part: list) -> ModelWindowMetrics:
        pods = {
            f"m-{i}": _pod(f"m-{i}", end, part, (latest_by_pod or {}).get(f"m-{i}", latest), start)
            for i in range(awake)
        }
        return aggregate_pods("m", start, end, pods)

    full = agg(end - W, grids)
    return replace(full, suffix_windows=(agg(end - 2 * GRID, grids[1:]), agg(end - GRID, grids[2:])))


def _snap(window: ModelWindowMetrics) -> MetricsSnapshot:
    return MetricsSnapshot(ts_ms=window.window_end_ms, models={"m": window}, stale=False)


def _view(awake: int, *, fetched_ms: int, total: int = 8, hidden: tuple = ()) -> ClusterView:
    bindings = tuple(
        Binding(serve_id=f"m-{i}", model="m", slot=Slot("node-a", (i,)), awake=i < awake,
                hidden=f"m-{i}" in hidden)
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


def _tick(state, window, *, awake, registry=None, queue=None, hidden=(), **kwargs):
    return run_planner_tick(
        _snap(window), queue=queue or _Queue(), registry=registry or _registry(), rescue_due=True,
        fairness_due=False, cluster_view=_view(awake, fetched_ms=window.window_end_ms + 4_000, hidden=hidden),
        signal_state=state, **kwargs,
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
    state = _state()
    registry = _registry(saturation_rescue=False)  # read every tick from the registry
    for i in range(3, 7):
        result = _tick(state, _window(BASE + i * GRID, [STARTING] * 3, (50.0, 1.0)), awake=1, registry=registry)
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
    assert "saturation_step_landed:m:1->2" in landed.events
    assert landed.model_contexts["m"]["saturation_count_after"] == BASE + 6 * GRID
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


# ======================================================= review fixes (2026-10-02)

#: Healthy steady traffic for the warm-TSS scenarios (Z ~ 8 per replica: HIGH).
WARM = (800.0, 3.0, 0.0, 40.0)


def _warm_then_change(*, before: dict, after: dict) -> list:
    """Onset (2 held windows, engine not full), one warm window, then 3 windows held by
    a routable change the rescue did not cause, with the engine full on all of them."""
    state = _state()
    results = []
    for i, grids in ((3, [IDLE, IDLE, WARM]), (4, [IDLE, WARM, WARM]), (5, [WARM] * 3)):
        results.append(_tick(state, _window(BASE + i * GRID, grids, (0.0, 0.1), awake=before["awake"]),
                             awake=before["awake"]))
    assert results[-1].model_contexts["m"]["signal_warm"] is True
    for i in (6, 7, 8):
        result = _tick(state, _window(BASE + i * GRID, [WARM] * 3, (20.0, 0.95), awake=after["pods"]),
                       awake=after["awake"], hidden=after.get("hidden", ()))
        assert result.model_contexts["m"]["signal_warm"] is False  # O1 holds the change
        results.append(result)
    return results


@pytest.mark.parametrize(
    "before, after",
    [
        ({"awake": 1}, {"awake": 3, "pods": 3}),  # a C1 scale-up 1 -> 3
        ({"awake": 3}, {"awake": 2, "pods": 2}),  # an immediate donor release 3 -> 2
        ({"awake": 3}, {"awake": 3, "pods": 3, "hidden": ("m-2",)}),  # a SafeScale probe hide
    ],
    ids=["c1_scale_up", "donor_release", "safescale_hide"],
)
def test_o1_hold_after_a_foreign_breakpoint_never_triggers(before, after):
    """P1: once the TSS was warm, an O1 hold caused by C1, a donor or SafeScale is not
    the onset - full engines on those windows are the TSS's business, not the rescue's."""
    results = _warm_then_change(before=before, after=after)
    held = results[3:]
    assert any(e.startswith("saturation_reset_external:m:") for e in held[0].events)
    for result in held:
        ctx = result.model_contexts["m"]
        assert ctx["saturation_reason"] is None and ctx["saturation_ticks"] == 0
        assert not result.classifications["m"].saturation_rescue
        assert not any(e.startswith(("saturation_rescue:", "saturation_pending:")) for e in result.events)


def test_an_idle_window_reopens_the_o1_hold_path():
    tracker = SaturationTracker()
    full = SaturationSample(waiting=5.0, kv=0.5, pods=1, sample_ms=0, running=10.0)

    def obs(i, reason, *, warm=False, idle=False):
        return tracker.observe("m", window_end_ms=i * GRID, routable=1, reason=reason, sample=full,
                               tss_warm=warm, idle=idle)

    obs(1, None, warm=True)
    assert obs(2, "o1_hold").reason is None and obs(3, "o1_hold").ticks == 0  # closed
    assert obs(4, "numerator_zero").ticks == 1  # numerator zero stays eligible
    assert obs(5, "numerator_zero", idle=True).ticks == 2  # idle window: open again
    assert obs(6, "o1_hold").fire


def test_backlog_on_the_old_pod_does_not_confirm_the_next_step():
    """P2-a: after 1 -> 2 the old pod still drains its backlog (waiting is per pod); the
    added pod has room -> no second step; once the added pod is full itself -> 2 -> 4."""
    state = _state()
    queue = _Queue()
    sat = [STARTING] * 3

    def tick(i, awake, latest_by_pod=None):
        window = _window(BASE + i * GRID, sat, (8.0, 0.6), awake=awake, latest_by_pod=latest_by_pod)
        return _tick(state, window, awake=awake, queue=queue)

    tick(3, 1)
    assert _planned(tick(4, 1)) == 1
    tick(5, 1)
    tick(6, 2)  # landed
    idle_new = {"m-0": (50.0, 0.98), "m-1": (0.0, 0.1)}
    for i in (7, 8, 9):
        result = tick(i, 2, idle_new)
        assert not result.actions and result.model_contexts["m"]["saturation_ticks"] == 0
        assert result.model_contexts["m"]["saturation_waiting"] == 50.0  # the sum alone would say full
    busy_new = {"m-0": (50.0, 0.98), "m-1": (4.0, 0.5)}
    assert not tick(10, 2, busy_new).actions
    assert _planned(tick(11, 2, busy_new)) == 2


def test_single_long_request_filling_the_kv_cache_does_not_trigger():
    """P2-b: one 32k-context prefill can fill the KV cache alone: KV >= 0.9 needs >= 2
    running requests; waiting > 0 needs nothing else."""
    state = _state()
    for i in range(3, 8):
        result = _tick(state, _window(BASE + i * GRID, [LONG_ONE] * 3, (0.0, 0.97)), awake=1)
        assert not result.actions and result.model_contexts["m"]["saturation_ticks"] == 0
    tracker = SaturationTracker()
    one = SaturationSample(waiting=0.0, kv=0.97, pods=1, sample_ms=0, running=1.0)
    assert not tracker.engine_full(one)
    assert tracker.engine_full(replace(one, running=2.0))
    assert tracker.engine_full(replace(one, kv=0.1, waiting=1.0))


def test_external_routable_change_restarts_the_count():
    tracker = SaturationTracker()
    full = SaturationSample(waiting=5.0, kv=0.5, pods=1, sample_ms=0, running=10.0)
    assert tracker.observe("m", window_end_ms=GRID, routable=2, reason="numerator_zero", sample=full).ticks == 1
    changed = tracker.observe("m", window_end_ms=2 * GRID, routable=1, reason="numerator_zero", sample=full)
    assert changed.ticks == 0 and changed.events == ("saturation_reset_external:m:2->1",)
    assert tracker.observe("m", window_end_ms=3 * GRID, routable=1, reason="numerator_zero", sample=full).ticks == 1


def test_receiver_order_tss_critical_first_then_waiting_per_replica():
    cfg = PlanConfig(min_replicas_per_model=1, max_replicas_per_model=8)

    def ctx(waiting, n=1, sat=True):
        out = {"assigned_replicas": n, "routable_pods": n, "awake_replicas": n, "signal_warm": not sat}
        if sat:
            out.update(saturation_rescue=True, saturation_waiting=waiting)
        return out

    tss = ModelClassification(model_name="t", state=ModelState.CRITICAL, role=ModelRole.RECEIVER, Z_m=0.5,
                              eta_m=None, trs=50.0, theta_m=THETA, tau=TauThresholds.from_control(0.2, 0.25))
    # One free GPU: the TSS-confirmed receiver gets it, wherever it sits in the list.
    plan = build_plan(model_contexts={"s": ctx(500.0), "t": ctx(0.0, sat=False)},
                      classifications=[_sat_cls("s"), tss], model_replicas={"s": 1, "t": 1}, idle_gpus=1, cfg=cfg)
    assert [a.model for a in _ups(plan.actions)] == ["t"]
    # Two saturation receivers: the larger backlog per replica first (40/1 > 60/2).
    plan = build_plan(model_contexts={"a": ctx(60.0, n=2), "b": ctx(40.0)},
                      classifications=[_sat_cls("a"), _sat_cls("b")], model_replicas={"a": 2, "b": 1},
                      idle_gpus=1, cfg=cfg)
    assert [a.model for a in _ups(plan.actions)] == ["b"]


def test_observe_mode_step_is_unconfirmed_and_counting_restarts():
    """A step the queue drops (observe mode) or the SM refuses never changes the routable
    count: after 3 grids ``saturation_step_unconfirmed``, then 2 fresh windows again."""
    state = _state()
    queue = _Queue()
    sat = [STARTING] * 3
    results = {
        i: _tick(state, _window(BASE + i * GRID, sat, (8.0, 0.6)), awake=1, queue=queue, observe_mode=True)
        for i in range(3, 10)
    }
    assert _planned(results[4]) == 1  # submitted; the queue drops it in observe mode
    assert not results[5].actions and results[5].model_contexts["m"]["saturation_awaiting_step"] is True
    assert any(e.startswith("saturation_step_unconfirmed:m:n=1:waited_ms=30000") for e in results[7].events)
    assert not results[8].actions and results[8].model_contexts["m"]["saturation_ticks"] == 1
    assert _planned(results[9]) == 1


class _PreemptingSafeScale:
    """An active probe on the receiver with one hidden pod: preempting it gives it back."""

    def __init__(self):
        self.preempted: list = []

    def request_preemption(self, model, reason):
        self.preempted.append((model, reason))
        return 1

    def start_probe(self, **_kwargs):  # pragma: no cover - no scale-down planned here
        raise AssertionError("no probe expected")


def test_probe_preemption_covers_the_step_and_its_unhide_lands_it():
    """The step 1 -> 2 is covered by unhiding the probe's pod (tick restore deduction):
    nothing is woken, the hidden pod is excluded from the sample, the unhide lands it."""
    state = _state()
    safescale = _PreemptingSafeScale()
    queue = _Queue()
    hidden = ("m-1",)
    latest = {"m-1": (90.0, 1.0)}  # the hidden pod drains a backlog: not counted
    first = _tick(state, _window(BASE + 3 * GRID, [IDLE, STARTING, STARTING], (6.0, 0.5), awake=2,
                                 latest_by_pod=latest), awake=2, hidden=hidden, queue=queue, safescale=safescale)
    ctx = first.model_contexts["m"]
    assert ctx["saturation_waiting"] == 6.0 and ctx["saturation_pods"] == 1
    second = _tick(state, _window(BASE + 4 * GRID, [STARTING] * 3, (6.0, 0.5), awake=2, latest_by_pod=latest),
                   awake=2, hidden=hidden, queue=queue, safescale=safescale)
    assert safescale.preempted == [("m", "receiver_need_upscale")]
    assert "safescale_probe_preempted:m:restored=1:up_needed=0" in second.events
    assert not _ups(second.actions)  # the restore covers the whole step
    assert second.model_contexts["m"]["saturation_awaiting_step"] is True
    landed = _tick(state, _window(BASE + 5 * GRID, [STARTING] * 3, (6.0, 0.5), awake=2), awake=2, queue=queue,
                   safescale=safescale)
    assert "saturation_step_landed:m:1->2" in landed.events


def test_config_is_read_from_the_registry_every_tick():
    state = _state()
    registry = _registry(saturation_consecutive_ticks=3)
    sat = [STARTING] * 3
    planned = [_planned(_tick(state, _window(BASE + i * GRID, sat, (8.0, 0.6)), awake=1, registry=registry))
               for i in (3, 4, 5)]
    assert planned == [0, 0, 1]
    assert state.saturation.config.consecutive_ticks == 3
    with pytest.raises(ValueError):
        parse_scaling_config({"saturation_max_step_factor": 4.5})
    assert parse_scaling_config({"saturation_max_step_factor": 4}).saturation_max_step_factor == 4.0


def test_decision_snapshot_exports_the_rescue_state():
    state = _state()
    window = _window(BASE + 3 * GRID, [IDLE, STARTING, STARTING], (6.0, 0.5), awake=2)
    result = _tick(state, window, awake=2)
    states = _model_states(result.model_contexts, result.classifications, _snap(window))["m"]
    assert states["saturation_pods"] == 2 and states["saturation_sample_ms"] == BASE + 3 * GRID
    assert states["saturation_awaiting_step"] is False and states["saturation_count_after"] is None
    assert states["saturation_pod_samples"] == [
        {"pod": "m-0", "waiting": 6.0, "kv": 0.5, "running": 40.0},
        {"pod": "m-1", "waiting": 6.0, "kv": 0.5, "running": 40.0},
    ]


# ================================================== review round 3 (2026-10-02)


def test_a_step_landing_in_parts_is_the_steps_own():
    """P2-1: target 4 lands 2 -> 3 -> 4 over two windows: both rises are the step, the
    added-pods rule and the chain survive (no external reset)."""
    tracker = SaturationTracker()
    old = (PodSample("a", 50.0, 0.98, 40.0), PodSample("b", 50.0, 0.98, 40.0))
    idle_new = PodSample("c", 0.0, 0.1, 3.0)

    def obs(i, routable, pods, reason="o1_hold", warm=False):
        sample = SaturationSample(waiting=sum(p.waiting for p in pods), kv=0.7, pods=len(pods), sample_ms=0,
                                  running=sum(p.running for p in pods), per_pod=tuple(pods))
        return tracker.observe("m", window_end_ms=i * GRID, routable=routable, reason=reason, sample=sample,
                               tss_warm=warm)

    obs(1, 2, old, reason=None, warm=True)  # warm before the step: only its own chain re-opens o1_hold
    tracker.note_step("m", window_end_ms=GRID, routable=2, pods=("a", "b"), target=4)
    first = obs(2, 3, old + (idle_new,))
    assert first.events == ("saturation_step_landed:m:2->3",)
    second = obs(3, 4, old + (idle_new, PodSample("d", 0.0, 0.1, 3.0)))
    assert second.events == ("saturation_step_landed:m:3->4",)  # not saturation_reset_external
    assert not second.full  # the added pods have room: no sum-based next step
    # The chain is intact: an O1 hold (caused by the step) stays eligible.
    assert second.reason == "o1_hold" and obs(4, 4, old + (idle_new,)).reason == "o1_hold"
    busy = (PodSample("c", 3.0, 0.5, 30.0), PodSample("d", 2.0, 0.5, 30.0))
    assert obs(5, 4, old + busy).ticks == 1 and obs(6, 4, old + busy).fire
    # Above the target or down: external.
    assert obs(7, 6, old + busy).events == ("saturation_reset_external:m:4->6",)


def test_step_pods_come_from_the_fleet_view_not_the_fresh_samples():
    """P2-2: an old pod whose sample was stale at the decision is not an added pod later."""
    state = _state()
    queue = _Queue()
    registry = _registry(max_awake=8)
    sat = [STARTING] * 3

    def window(i, awake, latest_by_pod=None, stale=()):
        w = _window(BASE + i * GRID, sat, (8.0, 0.6), awake=awake, latest_by_pod=latest_by_pod)
        per_pod = {name: (replace(pod, latest_instant_ms=w.window_end_ms - 3 * GRID) if name in stale else pod)
                   for name, pod in w.per_pod.items()}
        return replace(w, per_pod=per_pod)

    def tick(i, awake, **kw):
        return _tick(state, window(i, awake, **kw), awake=awake, queue=queue, registry=registry)

    tick(3, 2, stale=("m-1",))
    step1 = tick(4, 2, stale=("m-1",))
    assert _planned(step1) == 2 and step1.model_contexts["m"]["saturation_pods"] == 1  # 2 -> 4
    tick(5, 4)  # landed
    # m-1 (old) has room now; the added m-2, m-3 are full themselves -> next step 4 -> 8.
    pods = {"m-0": (8.0, 0.6), "m-1": (0.0, 0.1), "m-2": (5.0, 0.6), "m-3": (5.0, 0.6)}
    tick(6, 4, latest_by_pod=pods)
    assert _planned(tick(7, 4, latest_by_pod=pods)) == 4


def test_an_added_pod_without_a_fresh_sample_blocks_the_next_step():
    """2026-10-04 (I3): the added pods come from the fleet view's routable set. After
    1 -> 3, the added m-1 is full but m-2 has no fresh sample (unknown, not "room" and not
    skipped) -> no further step; once m-2 is sampled and full the next step follows."""
    state = _state()
    queue = _Queue()
    registry = _registry(max_awake=8, saturation_max_step_factor=3.0)
    sat = [STARTING] * 3

    def tick(i, awake, stale=()):
        w = _window(BASE + i * GRID, sat, (8.0, 0.6), awake=awake)
        per_pod = {name: (replace(pod, latest_instant_ms=w.window_end_ms - 3 * GRID) if name in stale else pod)
                   for name, pod in w.per_pod.items()}
        return _tick(state, replace(w, per_pod=per_pod), awake=awake, queue=queue, registry=registry)

    tick(3, 1)
    assert _planned(tick(4, 1)) == 2  # 1 -> 3
    tick(5, 3, stale=("m-2",))  # landed
    for i in (6, 7, 8):
        result = tick(i, 3, stale=("m-2",))
        assert not _ups(result.actions)
        assert result.model_contexts["m"]["saturation_ticks"] == 0
    tick(9, 3)
    assert _planned(tick(10, 3)) > 0


def test_events_are_reported_once_per_window():
    state = _state()
    queue = _Queue()
    sat = [STARTING] * 3
    for i in (3, 4, 5):
        _tick(state, _window(BASE + i * GRID, sat, (8.0, 0.6)), awake=1, queue=queue)
    first = _tick(state, _window(BASE + 6 * GRID, sat, (8.0, 0.6), awake=2), awake=2, queue=queue)
    again = _tick(state, _window(BASE + 6 * GRID, sat, (8.0, 0.6), awake=2), awake=2, queue=queue)
    assert "saturation_step_landed:m:1->2" in first.events
    assert not any(e.startswith("saturation_step_") for e in again.events)


def test_o1_resume_after_a_suspension_does_not_reopen_the_hold_path():
    """warmup 0 + O1 suspended: no onset is recorded; the O1 resume records one and holds
    the model for two grids - that is no new traffic period (no idle window), so a warm
    model stays out of the saturation path."""
    state = SignalState(
        warmup_ms=0,
        breakpoint=BreakpointWindowConfig(enabled=True, onset_guard=False, grid_ms=GRID, min_evidence_grids=2),
        saturation=SaturationTracker(),
    )
    state.suspend_breakpoint_window("gateway_clock")
    for i in (3, 4, 5):
        warm = _tick(state, _window(BASE + i * GRID, [WARM] * 3, (0.0, 0.1)), awake=1)
        assert warm.model_contexts["m"]["signal_warm"] is True
    state.resume_breakpoint_window()
    for i in (6, 7):
        held = _tick(state, _window(BASE + i * GRID, [WARM] * 3, (20.0, 0.95)), awake=1)
        ctx = held.model_contexts["m"]
        assert ctx["signal_warm"] is False  # the resume's onset holds it
        assert ctx["saturation_reason"] is None and ctx["saturation_ticks"] == 0
        assert not held.classifications["m"].saturation_rescue

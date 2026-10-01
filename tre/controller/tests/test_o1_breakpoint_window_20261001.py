"""O1 breakpoint-aware window (2026-10-01, design docs/design/20261001-o1-breakpoint-window.md).

A model decides on the part of its 30 s window after its last breakpoint (traffic onset,
routable-count change): complete 10 s grids only (the grid holding the breakpoint
excluded), the TSS numerator scaled to a whole window, the queue averaged over those
grids, the EMA restarted at the breakpoint. Scale-ups act after ``min_evidence_grids``
grids; scale-downs need a whole window after the breakpoint. Replaces the ADR-0013 onset
warmup guard (``scaling.onset_warmup_guard`` keeps it switchable)."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import (
    ClusterTopology,
    ModelSpec,
    NodeSpec,
    Registry,
    ScalingRegistryConfig,
    SloSpec,
    TrsParams,
    load_registry,
    parse_scaling_config,
)
from tre_controller.loops.model_state_box import UNCONFIRMED, ModelStateBox
from tre_controller.loops.rescue_task import run_rescue_tick
from tre_controller.loops.tick import _model_contexts, _rescue_bases, breakpoint_observation
from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, TauThresholds
from tre_controller.planning.planner import ClusterView, PlanConfig, ScaleAction, build_plan
from tre_controller.signals.trs import BreakpointWindowConfig, SignalState
from tre_controller.store.metrics_store import MetricsStore
from tre_sm.allocator.slots import Binding, Slot

GRID = 10_000
W = 30_000
THETA = 100.0
O1 = BreakpointWindowConfig(enabled=True, onset_guard=False, grid_ms=GRID, min_evidence_grids=2)


# ------------------------------------------------------------------- fixtures


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
    10 s grids in ``(start, end]``, aggregated the way MetricsStore does (token total,
    instant mean over the expected samples)."""
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
    """The 30 s window ending at ``end`` (3 grids, oldest first) with its 20 s / 10 s suffixes."""
    assert len(grids) == 3
    full = _agg(end - W, end, grids, routable)
    return replace(
        full,
        suffix_windows=(_agg(end - 2 * GRID, end, grids[1:], routable), _agg(end - GRID, end, grids[2:], routable)),
    )


def _snap(window: ModelWindowMetrics) -> MetricsSnapshot:
    return MetricsSnapshot(ts_ms=window.window_end_ms, models={"m": window}, stale=False)


IDLE = (0.0, 0.0, 0.0, 0.0)
#: ~1 req/s trickle: low concurrency, Q clamps at qmin -> steady Z = 3*40/1/100 = 1.2.
TRICKLE = (40.0, 0.5, 0.0, 10.0)
#: Real overload: Q = 30 + 2.625*20 = 82.5, steady Z = 3*2000/82.5/100 = 0.727 < tau_crit.
BURST = (2000.0, 30.0, 20.0, 100.0)


def _series(pattern: list) -> list[tuple[int, ModelWindowMetrics]]:
    """Windows ending at each grid boundary of a per-grid ``pattern`` (oldest first),
    starting with the first window whose 3 grids are all in the pattern."""
    out = []
    for i in range(2, len(pattern)):
        end = (i + 1) * GRID + 1_000_000
        out.append((end, _window(end, pattern[i - 2 : i + 1])))
    return out


class _Queue:
    def __init__(self, last_done: dict | None = None):
        self.submitted: list = []
        self._last = dict(last_done or {})

    def inflight_models(self) -> set[str]:
        return set()

    def submit(self, actions) -> object:
        self.submitted.extend(actions)
        return object()

    def last_actions(self):
        return dict(self._last)


def _ups(actions) -> list[ScaleAction]:
    return [a for a in actions if isinstance(a, ScaleAction) and a.delta > 0]


def _view(awake: int, *, fetched_ms: int | None, total: int = 4) -> ClusterView:
    registry = _registry()
    bindings = tuple(
        Binding(serve_id=f"m-{i}", model="m", slot=Slot("node-a", (i,)), awake=i < awake, hidden=False)
        for i in range(total)
    )
    return ClusterView(topology=registry.topology(), bindings=bindings, fetched_ms=fetched_ms)


# --------------------------------------------------------- store: suffix windows


def _store_fixture():
    from test_metrics_store import FakeRedis, add_doc, hist_doc, inst_doc

    redis = FakeRedis()
    pod = "default/pod-a"
    redis.sadd("tre:v2:pods:dsqwen-7b", pod)
    prompt = 0
    for i, ts in enumerate(range(0, 40_001, GRID)):  # boundary stamps 0..40 s
        prompt += 10 * (i + 1)
        add_doc(redis, "tre:v2:hist:" + pod, ts, hist_doc("pod-a", prompt, i + 1, 0.1 * (i + 1), i + 1, {"0.5": i + 1}))
        add_doc(redis, "tre:v2:inst:" + pod, ts, inst_doc("pod-a", waiting=i, running=2 * i, kv_hit=0.0))
    return redis


def test_store_suffix_windows_equal_a_read_of_the_shorter_window():
    registry = load_registry(str(Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml"))
    redis = _store_fixture()
    store = MetricsStore(redis, registry, instant_sample_interval_ms=GRID, suffix_period_ms=GRID)
    plain = MetricsStore(redis, registry, instant_sample_interval_ms=GRID)
    full = store.read_model_window("dsqwen-7b", 10_000, 40_000, use_cache=False, start_exclusive=True)
    assert [s.window_start_ms for s in full.suffix_windows] == [20_000, 30_000]
    reference_full = plain.read_model_window("dsqwen-7b", 10_000, 40_000, use_cache=False, start_exclusive=True)
    assert full == reference_full and reference_full.suffix_windows == ()
    for suffix in full.suffix_windows:
        direct = plain.read_model_window(
            "dsqwen-7b", suffix.window_start_ms, 40_000, use_cache=False, start_exclusive=True
        )
        assert suffix == direct
        assert suffix.prompt_tokens == direct.prompt_tokens and suffix.avg_running == direct.avg_running
    # (30, 40]: tokens of the last grid only, the instant mean over its single tick.
    last = full.suffix_windows[-1]
    assert last.prompt_tokens == 50.0 and last.avg_running == 8.0 and last.avg_waiting == 4.0


def test_store_suffixes_off_unaligned_or_v1():
    registry = load_registry(str(Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml"))
    redis = _store_fixture()
    store = MetricsStore(redis, registry, instant_sample_interval_ms=GRID, suffix_period_ms=GRID)
    assert store.read_model_window("dsqwen-7b", 5_000, 35_000, use_cache=False).suffix_windows == ()
    with pytest.raises(ValueError):
        MetricsStore(redis, registry, instant_sample_interval_ms=GRID, suffix_period_ms=-1)


def test_restrict_to_serving_restricts_the_suffixes():
    from tre_common.metrics_schema import PodWindowMetrics
    from tre_common.window_pods import aggregate_pods, restrict_to_serving

    def pod(name, tokens):
        return PodWindowMetrics(pod=name, prompt_tokens=0.0, generation_tokens=tokens, avg_waiting=0.0,
                                avg_running=1.0, avg_swapping=0.0, kv_cache_hit_rate=0.0,
                                ttft_p95_ms=None, tpot_p95_ms=None, e2e_p95_ms=None)

    full = aggregate_pods("m", 0, 30_000, {"a": pod("a", 30.0), "b": pod("b", 3.0)})
    suffix = aggregate_pods("m", 10_000, 30_000, {"a": pod("a", 20.0), "b": pod("b", 2.0)})
    window = replace(full, suffix_windows=(suffix,))
    out = restrict_to_serving(window, sleeping_pods={"b"}, routable_pods=1)
    assert out.generation_tokens == 30.0 and out.suffix_windows[0].generation_tokens == 20.0
    assert out.suffix_windows[0].routable_pods == 1


# ------------------------------------------------- steady state: bitwise equal


def _prime_onset(state: SignalState, end: int) -> None:
    state.observe_traffic("m", has_traffic=True, window_start_ms=end - W, window_end_ms=end)


@pytest.mark.parametrize("tau", [10_000.0, 20_000.0, 0.0, None])
def test_full_windows_without_breakpoint_give_bitwise_the_pre_o1_signal(tau):
    """No breakpoint inside the window -> the full window, the same EMA: every context
    value (raw, EMA'd TSS, Z, Q, rates) is bit-identical to the pre-O1 computation."""
    registry = _registry(tau)
    legacy = SignalState(warmup_ms=-1)
    o1 = SignalState(warmup_ms=-1, breakpoint=O1)
    pattern = [TRICKLE, BURST, TRICKLE, (900.0, 7.0, 1.0, 40.0), BURST, BURST, (10.0, 1.0, 0.0, 3.0), TRICKLE] * 3
    windows = _series(pattern)
    _prime_onset(legacy, windows[0][0] - W)
    _prime_onset(o1, windows[0][0] - W)
    view = _view(1, fetched_ms=0)
    for end, window in windows:
        for loop in range(2):  # rescue + fairness re-read the same snapshot
            a, _ = _model_contexts(_snap(window), registry, signal_state=legacy, cluster_view=view)
            b, _ = _model_contexts(_snap(window), registry, signal_state=o1, cluster_view=view, queue=_Queue())
            for key in ("trs", "trs_raw", "z_m", "trs_z_m", "Q", "Q_ctl", "Y_m", "eta_m",
                        "request_rate_rps", "decode_tps", "signal_warm"):
                assert a["m"][key] == b["m"][key], (end, key)  # bitwise (==, not approx)
            assert b["m"]["signal_full_window"] is True and b["m"]["signal_hold_reason"] is None


# ----------------------------------------------------------- onset: grids 1/2/3


def test_onset_first_grids_then_scaled_suffix_matches_the_steady_value():
    registry = _registry(10_000.0)
    state = SignalState(warmup_ms=-1, breakpoint=O1)
    base = 1_000_000
    pattern = [IDLE, IDLE, IDLE] + [TRICKLE] * 6
    seen = []
    for i in range(2, len(pattern)):
        end = base + (i + 1) * GRID
        ctx, _ = _model_contexts(_snap(_window(end, pattern[i - 2 : i + 1])), registry, signal_state=state)
        seen.append((end, ctx["m"]))
    onset = base + 4 * GRID  # first window with traffic
    by_end = dict(seen)
    first = by_end[onset]
    assert first["signal_warm"] is False and first["signal_hold_reason"] == "no_complete_grid"
    assert first["signal_breakpoint_ms"] == onset and first["signal_evidence_grids"] == 0
    one = by_end[onset + GRID]
    assert one["signal_warm"] is False and one["signal_hold_reason"] == "evidence_grids"
    assert one["signal_evidence_grids"] == 1
    two = by_end[onset + 2 * GRID]
    assert two["signal_warm"] is True and two["signal_full_window"] is False
    assert two["signal_window_start_ms"] == onset and two["signal_evidence_grids"] == 2
    # numerator 2 grids x 40 scaled x 1.5 = 120, Q = qmin -> raw 120, Z 1.2 = steady state.
    assert two["trs_raw"] == pytest.approx(120.0) and two["z_m"] == pytest.approx(1.2)
    three = by_end[onset + 3 * GRID]  # the grid holding the onset left the window: full
    assert three["signal_full_window"] is True and three["z_m"] == pytest.approx(1.2)
    # The pre-O1 windows of the same ticks dipped to 0.4 / 0.8 (window-fill fraction).
    assert first["trs_z_m"] == pytest.approx(0.4) and one["trs_z_m"] == pytest.approx(0.8)


def test_ema_restarts_at_the_breakpoint_and_never_sees_the_onset_windows():
    registry = _registry(10_000.0)
    state = SignalState(warmup_ms=-1, breakpoint=O1)
    pattern = [IDLE, IDLE, IDLE, TRICKLE, TRICKLE, BURST, BURST]
    out = {}
    for i in range(2, len(pattern)):
        end = 1_000_000 + (i + 1) * GRID
        ctx, _ = _model_contexts(_snap(_window(end, pattern[i - 2 : i + 1])), registry, signal_state=state)
        out[end] = ctx["m"]
    warm_end = 1_000_000 + 6 * GRID  # 2 grids after the onset at 1_040_000: (TRICKLE, BURST)
    assert out[warm_end]["signal_warm"] is True
    assert out[warm_end]["trs"] == out[warm_end]["trs_raw"]  # seeded, nothing older in it
    nxt = out[warm_end + GRID]  # the next window (full: onset left it) blends only post-onset values
    decay = math.exp(-GRID / 10_000.0)
    assert nxt["signal_full_window"] is True
    assert nxt["trs"] == pytest.approx(decay * out[warm_end]["trs_raw"] + (1 - decay) * nxt["trs_raw"])


# -------------------------------------------- routable change as a breakpoint


def test_routable_change_dated_by_the_done_hint_and_its_grid_excluded():
    state = SignalState(breakpoint=O1)
    assert state.note_routable("m", 1, observed_ms=100_000) is None
    assert state.note_routable("m", 1, observed_ms=110_000) is None
    # Change seen at 120 s, our scale-up completed at 113.5 s: dated 113.5 s.
    assert state.note_routable("m", 3, observed_ms=120_000, done_hints=(113_500, 90_000)) == 113_500
    window = replace(_window(140_000, [TRICKLE] * 3, routable=3))
    eff = state.effective_window("m", window)
    assert eff.start_ms == 120_000 and eff.grids == 2 and eff.warm and not eff.full
    # An unexplained change (no hint after the previous observation) -> the observation time.
    assert state.note_routable("m", 2, observed_ms=130_000, done_hints=(113_500,)) == 130_000
    # Re-reads of the same view are idempotent.
    assert state.note_routable("m", 2, observed_ms=130_000) is None
    assert state.breakpoint_ms("m") == 130_000


def test_first_observation_after_restart_takes_the_latest_hint():
    state = SignalState(breakpoint=O1)
    assert state.note_routable("m", 3, observed_ms=200_000, done_hints=(185_000, None)) == 185_000
    assert SignalState(breakpoint=O1).note_routable("m", 3, observed_ms=200_000) is None


def test_breakpoint_observation_uses_view_time_and_queue_hints():
    class _RescueQueue(_Queue):
        def rescue_targets(self):
            from tre_controller.loops.action_queue import RescueTargetRecord

            return {"m": RescueTargetRecord(target=3, desired=3, base=1, covered_before=1, issued_ms=1, done_ms=7_000)}

    observed, hints = breakpoint_observation(_view(2, fetched_ms=9_000), _RescueQueue({"m": (6_000, "up")}), "m", 1)
    assert observed == 9_000 and set(hints) == {6_000, 7_000}
    observed, hints = breakpoint_observation(_view(2, fetched_ms=None), None, "m", 4_000)
    assert observed == 4_000 and hints == ()


def test_scale_up_restarts_the_ema_and_holds_scale_downs_for_a_whole_window():
    registry = _registry(10_000.0)
    state = SignalState(warmup_ms=-1, breakpoint=O1)
    base = 2_000_000
    _prime_onset(state, base - 10 * GRID)
    done = base + 2 * GRID + 1_500  # our scale-up 1 -> 3 completed between two view fetches
    contexts = {}
    for k in range(0, 8):
        end = base + k * GRID
        routable = 1 if k <= 1 else 3
        view = _view(routable, fetched_ms=end + 3_000)
        grids = [BURST] * 3 if routable == 1 else [TRICKLE] * 3
        ctx, _ = _model_contexts(_snap(_window(end, grids, routable=routable)), registry, signal_state=state,
                                 cluster_view=view, queue=_Queue({"m": (done, "up")}))
        contexts[end] = ctx["m"]
    # Seen in the view fetched at base+23 s, dated by the done time (after the previous fetch).
    assert contexts[base + 2 * GRID]["signal_breakpoint_ms"] == done
    assert contexts[base + 2 * GRID]["signal_hold_reason"] == "no_complete_grid"
    assert contexts[base + 3 * GRID]["signal_hold_reason"] == "no_complete_grid"  # (20, 30] holds it
    assert contexts[base + 4 * GRID]["signal_hold_reason"] == "evidence_grids"
    warm = contexts[base + 5 * GRID]
    assert warm["signal_warm"] is True and warm["signal_full_window"] is False
    assert warm["signal_window_start_ms"] == base + 3 * GRID
    assert warm["trs"] == warm["trs_raw"]  # EMA restarted: nothing from the 1-replica windows
    assert contexts[base + 6 * GRID]["signal_full_window"] is True


def test_min_evidence_requests():
    cfg = replace(O1, min_evidence_requests=25)
    state = SignalState(breakpoint=cfg)
    state.observe_traffic("m", has_traffic=True, window_start_ms=0, window_end_ms=30_000)  # onset 30 s
    eff = state.effective_window("m", replace(_window(50_000, [IDLE, TRICKLE, TRICKLE])))
    assert eff.grids == 2 and not eff.warm and eff.reason == "evidence_requests"  # 20 < 25
    eff = state.effective_window("m", _window(50_000, [IDLE, BURST, BURST]))
    assert eff.warm


def test_missing_suffix_waits_for_a_whole_clean_window():
    state = SignalState(breakpoint=O1)
    state.observe_traffic("m", has_traffic=True, window_start_ms=0, window_end_ms=30_000)
    plain = replace(_window(50_000, [IDLE, BURST, BURST]), suffix_windows=())
    eff = state.effective_window("m", plain)
    assert not eff.warm and eff.reason == "no_suffix"
    assert state.effective_window("m", replace(_window(60_000, [BURST] * 3), suffix_windows=())).full


# ------------------------------------------------------------------- planner


def _cls(model: str, state: ModelState, z: float) -> ModelClassification:
    role = {ModelState.CRITICAL: ModelRole.RECEIVER, ModelState.LOW: ModelRole.RECEIVER,
            ModelState.HIGH: ModelRole.DONOR, ModelState.IDLE: ModelRole.DONOR}.get(state, ModelRole.NEUTRAL)
    return ModelClassification(model_name=model, state=state, role=role, Z_m=z, eta_m=500.0, trs=0.0,
                               theta_m=1.0, tau=TauThresholds.from_control(),
                               donor_tier="surplus" if role == ModelRole.DONOR else None)


def test_planner_holds_donors_until_a_whole_window_follows_the_breakpoint():
    cfg = PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4)
    contexts = {
        "hot": {"assigned_replicas": 3, "routable_pods": 3, "awake_replicas": 3,
                "signal_warm": True, "signal_full_window": False},
        "cold": {"assigned_replicas": 1, "routable_pods": 1, "awake_replicas": 1,
                 "signal_warm": True, "signal_full_window": False},
    }
    plan = build_plan(model_contexts=contexts, classifications=[_cls("hot", ModelState.HIGH, 2.0),
                      _cls("cold", ModelState.CRITICAL, 0.5)],
                      model_replicas={"hot": 3, "cold": 1}, idle_gpus=0, cfg=cfg)
    assert "donor_suppressed_breakpoint_window:hot" in plan.events
    assert not [a for a in plan.actions if getattr(a, "model", None) == "hot"]
    # the warm receiver is still served (here: nothing idle, nobody to give)
    assert "receiver_suppressed_signal_warmup:cold" not in plan.events
    contexts["hot"]["signal_full_window"] = True
    plan = build_plan(model_contexts=contexts, classifications=[_cls("hot", ModelState.HIGH, 2.0),
                      _cls("cold", ModelState.CRITICAL, 0.5)],
                      model_replicas={"hot": 3, "cold": 1}, idle_gpus=0, cfg=cfg)
    assert plan.actions and not any(e.startswith("donor_suppressed_breakpoint_window") for e in plan.events)


def test_warm_receiver_on_a_partial_window_scales_up():
    cfg = PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4)
    ctx = {"m": {"assigned_replicas": 1, "routable_pods": 1, "awake_replicas": 1,
                 "signal_warm": True, "signal_full_window": False}}
    plan = build_plan(model_contexts=ctx, classifications=[_cls("m", ModelState.CRITICAL, 0.5)],
                      model_replicas={"m": 1}, idle_gpus=3, cfg=cfg)
    assert _ups(plan.actions)


def test_model_state_box_marks_held_signals_unconfirmed():
    box = ModelStateBox(now_ms=lambda: 0)
    box.update({"m": _cls("m", ModelState.HIGH, 2.0)}, {"m": {"signal_hold_reason": "evidence_grids"}}, ts_ms=0)
    assert box.get() == {"m": UNCONFIRMED}


# ---------------------------------------------------------- C1 settle under O1


def test_c1_target_settles_once_warm_after_its_breakpoint():
    from tre_controller.loops.action_queue import RescueTargetRecord

    class _TargetQueue(_Queue):
        def rescue_targets(self):
            return {"m": RescueTargetRecord(target=3, desired=3, base=1, covered_before=1, issued_ms=0, done_ms=113_500)}

    registry = _registry(10_000.0)  # k = 2 -> pre-O1 rule needs window_start >= 133.5 s
    snapshot = _snap(_window(140_000, [TRICKLE] * 3, routable=3))  # window_start 110 s
    held = {"m": {"signal_breakpoint_ms": 100_000, "signal_warm": True, "routable_pods": 3}}
    assert "m" in _rescue_bases(snapshot, _TargetQueue(), registry, held)
    warm = {"m": {"signal_breakpoint_ms": 113_500, "signal_warm": True, "routable_pods": 3}}
    assert "m" not in _rescue_bases(snapshot, _TargetQueue(), registry, warm)
    cold = {"m": {"signal_breakpoint_ms": 113_500, "signal_warm": False, "routable_pods": 3}}
    assert "m" in _rescue_bases(snapshot, _TargetQueue(), registry, cold)
    # Without O1 context keys the pre-O1 rule alone decides.
    assert "m" in _rescue_bases(snapshot, _TargetQueue(), registry, {"m": {"routable_pods": 3}})


# ------------------------------------------------- end to end: trickle vs burst


def _run(state: SignalState | None, pattern: list, registry: Registry) -> list[tuple[int, int]]:
    """(window_end, replicas added) of every rescue tick that scaled ``m`` up."""
    queue = _Queue()
    fired = []
    for i in range(2, len(pattern)):
        end = 1_000_000 + (i + 1) * GRID
        before = len(queue.submitted)
        run_rescue_tick(_snap(_window(end, pattern[i - 2 : i + 1])), queue=queue, registry=registry,
                        signal_state=state)
        added = sum(a.delta for a in _ups(queue.submitted[before:]))
        if added:
            fired.append((end, added))
    return fired


def test_trickle_from_idle_never_scales_and_a_burst_scales_after_two_grids():
    registry = _registry(10_000.0, rescue_max_step_pods=4)
    onset = 1_000_000 + 4 * GRID
    trickle = [IDLE, IDLE, IDLE] + [TRICKLE] * 8
    # No guard at all: the filling window reads CRITICAL (the ADR-0013 failure).
    assert _run(SignalState(warmup_ms=0), trickle, registry)
    # Pre-O1 guard and O1: no scale-up on a trickle.
    assert _run(SignalState(warmup_ms=-1), trickle, registry) == []
    assert _run(SignalState(warmup_ms=-1, breakpoint=O1), trickle, registry) == []

    burst = [IDLE, IDLE, IDLE] + [BURST] * 6
    o1 = _run(SignalState(warmup_ms=-1, breakpoint=O1), burst, registry)
    legacy = _run(SignalState(warmup_ms=-1), burst, registry)
    assert o1 and o1[0][0] == onset + 2 * GRID  # 20 s of evidence after the onset grid
    assert legacy and legacy[0][0] == onset + 3 * GRID  # the whole window after the onset
    # The fallback switch restores the pre-O1 behaviour exactly.
    fallback = BreakpointWindowConfig(enabled=False, onset_guard=True, grid_ms=GRID)
    assert _run(SignalState(warmup_ms=-1, breakpoint=fallback), burst, registry) == legacy
    assert _run(SignalState(warmup_ms=-1, breakpoint=fallback), trickle, registry) == []


# ----------------------------------------------------------------- registry


def test_scaling_registry_o1_keys():
    assert parse_scaling_config(None).breakpoint_window is True
    cfg = parse_scaling_config({"breakpoint_window": False, "onset_warmup_guard": True,
                                "min_evidence_grids": 3, "min_evidence_requests": 5})
    assert (cfg.breakpoint_window, cfg.onset_warmup_guard, cfg.min_evidence_grids, cfg.min_evidence_requests) == (
        False, True, 3, 5)
    for bad in ({"min_evidence_grids": 0}, {"min_evidence_grids": 1.5}, {"min_evidence_requests": -1},
                {"breakpoint_window": "yes"}, {"min_evidence_grids": True}):
        with pytest.raises(ValueError):
            parse_scaling_config(bad)
    built = BreakpointWindowConfig.from_registry(_registry(min_evidence_grids=3), grid_ms=GRID)
    assert built == BreakpointWindowConfig(enabled=True, onset_guard=False, grid_ms=GRID, min_evidence_grids=3)
    # A registry without a scaling section (older loaders / fakes): the O1 defaults.
    assert BreakpointWindowConfig.from_registry(object(), grid_ms=GRID).enabled is True


def test_onset_guard_switch_composes_with_o1():
    both = SignalState(warmup_ms=-1, breakpoint=replace(O1, onset_guard=True))
    registry = _registry(10_000.0)
    pattern = [IDLE, IDLE, IDLE, BURST, BURST, BURST, BURST]
    warm = {}
    for i in range(2, len(pattern)):
        end = 1_000_000 + (i + 1) * GRID
        ctx, _ = _model_contexts(_snap(_window(end, pattern[i - 2 : i + 1])), registry, signal_state=both)
        warm[end] = ctx["m"]["signal_warm"]
    onset = 1_000_000 + 4 * GRID
    assert warm[onset + 2 * GRID] is False  # O1 evidence ok, the onset guard still waits
    assert warm[onset + 3 * GRID] is True


def test_safescale_observation_uses_the_planner_signal_under_o1():
    from tre_controller.loops.safescale_task import _observation_from_metrics

    registry = _registry(10_000.0)
    state = SignalState(warmup_ms=-1, breakpoint=O1)
    state.observe_traffic("m", has_traffic=True, window_start_ms=0, window_end_ms=30_000)
    held = _window(40_000, [IDLE, BURST, BURST])  # 1 grid after the onset at 30 s
    obs = _observation_from_metrics(40_000, held, registry.model("m"), "zm", signal_state=state)
    # Not warm: the whole window's raw Z (no EMA advanced).
    assert obs.z_m == pytest.approx(2 * 2000 / (2 * 30 / 3 + 2.625 * 2 * 20 / 3) / THETA)
    assert state.computer_for("m", ema_alpha=0.5, ema_tau_ms=10_000.0).tss_ema.value is None
    warm = _window(50_000, [IDLE, BURST, BURST])  # 2 grids after the onset
    obs = _observation_from_metrics(50_000, warm, registry.model("m"), "zm", signal_state=state)
    assert obs.z_m == pytest.approx(3 * 2000 / 82.5 / THETA)
    ctx, _ = _model_contexts(_snap(warm), registry, signal_state=state)
    assert ctx["m"]["z_m"] == obs.z_m  # same window, same EMA value


def test_shipped_registry_enables_o1_with_the_default_evidence():
    scaling = load_registry().scaling()
    assert scaling.breakpoint_window is True and scaling.onset_warmup_guard is False
    assert scaling.min_evidence_grids == 2 and scaling.min_evidence_requests == 0
    assert json.dumps(sorted(ScalingRegistryConfig.__dataclass_fields__))  # serialisable names
    assert math.isfinite(BreakpointWindowConfig().grid_ms)

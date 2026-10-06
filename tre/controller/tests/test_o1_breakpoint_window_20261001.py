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
    def __init__(self, changes: dict | None = None):
        self.submitted: list = []
        self._changes = dict(changes or {})

    def inflight_models(self) -> set[str]:
        return set()

    def submit(self, actions) -> object:
        self.submitted.extend(actions)
        return object()

    def routable_changes(self):
        return dict(self._changes)


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
    # Change seen at 120 s, our scale-up returned at 113.5 s: dated 113.5 s + 1 s margin.
    assert state.note_routable("m", 3, observed_ms=120_000, done_hints=(113_500, 90_000)) == 114_500
    window = replace(_window(140_000, [TRICKLE] * 3, routable=3))
    eff = state.effective_window("m", window)
    assert eff.start_ms == 120_000 and eff.grids == 2 and eff.warm and not eff.full
    # An unexplained change (no hint after the previous observation) -> the observation time.
    assert state.note_routable("m", 2, observed_ms=130_000, done_hints=(113_500,)) == 131_000
    # Re-reads of the same view are idempotent; an older view is ignored (review P3).
    assert state.note_routable("m", 2, observed_ms=130_000) is None
    assert state.note_routable("m", 5, observed_ms=125_000) is None
    assert state.breakpoint_ms("m") == 131_000 == state.settle_breakpoint_ms("m")


def test_margin_pushes_a_change_near_a_boundary_to_the_next_grid():
    state = SignalState(breakpoint=O1)
    state.note_routable("m", 1, observed_ms=100_000)
    assert state.note_routable("m", 2, observed_ms=125_000, done_hints=(119_500,)) == 120_500
    eff = state.effective_window("m", _window(150_000, [TRICKLE] * 3, routable=2))
    assert eff.start_ms == 130_000  # the gateway may route to the new pod only after 120 s
    no_margin = SignalState(breakpoint=replace(O1, margin_ms=0))
    no_margin.note_routable("m", 1, observed_ms=100_000)
    no_margin.note_routable("m", 2, observed_ms=125_000, done_hints=(119_500,))
    assert no_margin.effective_window("m", _window(150_000, [TRICKLE] * 3, routable=2)).start_ms == 120_000


def test_first_observation_takes_the_latest_hint_but_never_settles_c1():
    state = SignalState(breakpoint=O1)
    assert state.note_routable("m", 3, observed_ms=200_000, done_hints=(185_000, None)) == 186_000
    assert state.breakpoint_ms("m") == 186_000
    assert state.settle_breakpoint_ms("m") is None  # review P2-a: a guessed date
    assert SignalState(breakpoint=O1).note_routable("m", 3, observed_ms=200_000) is None


def test_restart_with_an_inflight_target_keeps_the_c1_basis():
    """Review P2-a: after a restart the restored in-flight target is stamped done=now; a
    breakpoint dated on the first observation must not settle it (the SM may still be
    waking); a count change seen afterwards does."""
    from tre_controller.loops.action_queue import RescueTargetRecord

    class _TargetQueue(_Queue):
        def rescue_targets(self):
            return {"m": RescueTargetRecord(target=3, desired=3, base=1, covered_before=1, issued_ms=0,
                                            done_ms=200_000)}

    registry = _registry(10_000.0)
    state = SignalState(warmup_ms=-1, breakpoint=O1)
    _prime_onset(state, 100_000)
    queue = _TargetQueue({"m": 200_000})
    contexts = {}
    for end, routable in ((200_000, 1), (210_000, 1), (220_000, 1), (230_000, 1), (240_000, 3), (250_000, 3),
                          (260_000, 3), (270_000, 3)):
        ctx, _ = _model_contexts(_snap(_window(end, [BURST] * 3, routable=routable)), registry, signal_state=state,
                                 cluster_view=_view(routable, fetched_ms=end + 3_000), queue=queue)
        contexts[end] = ctx["m"]
        settled = "m" not in _rescue_bases(_snap(_window(end, [BURST] * 3, routable=routable)), queue, registry, ctx)
        contexts[end]["settled"] = settled
    # First observation dated a change at 201 s from the hint: windows hold it, C1 does not settle.
    assert contexts[230_000]["signal_warm"] is True and contexts[230_000]["signal_settle_ms"] < 200_000
    assert contexts[230_000]["settled"] is False
    # The wake the SM finished later is a change seen between two views: settles once warm.
    assert contexts[270_000]["signal_settle_ms"] >= 200_000 and contexts[270_000]["settled"] is True


def test_breakpoint_observation_uses_view_time_and_routable_change_stamps_only():
    class _RescueQueue(_Queue):
        def rescue_targets(self):  # never a hint (review P1): stamped when planned
            from tre_controller.loops.action_queue import RescueTargetRecord

            return {"m": RescueTargetRecord(target=3, desired=3, base=1, covered_before=1, issued_ms=1, done_ms=7_000)}

        def last_actions(self):
            return {"m": (6_000, "up")}

    observed, hints = breakpoint_observation(_view(2, fetched_ms=9_000), _RescueQueue({"m": 8_000}), "m", 1)
    assert observed == 9_000 and hints == (8_000,)
    observed, hints = breakpoint_observation(_view(2, fetched_ms=None), None, "m", 4_000)
    assert observed == 4_000 and hints == ()


def test_queue_stamps_routable_changes_when_the_sm_call_returns():
    """Review P1: a rescue target covered by a probe preemption is recorded done when
    planned; the unhide that really raises the routable count runs later. The breakpoint
    hint is the unhide's return time, never the planning time."""
    import asyncio

    from tre_controller.loops.action_queue import ActionQueue
    from tre_controller.planning.planner import HideAction, RescuePlan, UnhideAction

    class _Client:
        async def set_routable(self, model, hidden_pods):
            return {"ok": True}

        async def scale_model(self, model, delta, **_kwargs):
            return {"ok": False, "error": "HTTP 409: WakeConflict"}

    now = {"ms": 100_000}
    queue = ActionQueue(_Client(), now_ms=lambda: now["ms"])
    queue.record_rescue_covered("m", RescuePlan(target=3, desired=3, base=1, covered=3))  # done 100 s
    assert queue.routable_changes() == {}
    now["ms"] = 104_000
    asyncio.run(queue._timed_dispatch(UnhideAction("m", ("m-1",), "rollback", "safescale"), "m"))
    assert queue.routable_changes() == {"m": (104_000, 1)}
    now["ms"] = 109_000
    asyncio.run(queue._timed_dispatch(HideAction("m", ("m-1",), "probe", "safescale"), "m"))
    assert queue.routable_changes() == {"m": (109_000, -1)}
    now["ms"] = 111_000  # a failed (possibly partial) wake is stamped too
    asyncio.run(queue._timed_dispatch(ScaleAction("m", 1, "rescue", "rescue"), "m"))
    assert queue.routable_changes() == {"m": (111_000, 1)}
    # Invariant: the view showing the new count (fetched 112 s, previous 101 s) dates
    # the change at the last return (+ margin), never at the 100 s planning stamp.
    state = SignalState(breakpoint=O1)
    state.note_routable("m", 1, observed_ms=101_000)
    observed, hints = breakpoint_observation(_view(3, fetched_ms=112_000), queue, "m", 0)
    assert hints == ((111_000, 1),)
    assert state.note_routable("m", 3, observed_ms=observed, done_hints=hints) == 112_000


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
                                 cluster_view=view, queue=_Queue({"m": done}))
        contexts[end] = ctx["m"]
    # Seen in the view fetched at base+23 s, dated by the done time (after the previous fetch).
    assert contexts[base + 2 * GRID]["signal_breakpoint_ms"] == done + O1.margin_ms
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
    held = {"m": {"signal_settle_ms": 100_000, "signal_warm": True, "routable_pods": 3}}
    assert "m" in _rescue_bases(snapshot, _TargetQueue(), registry, held)
    warm = {"m": {"signal_settle_ms": 113_500, "signal_warm": True, "routable_pods": 3}}
    assert "m" not in _rescue_bases(snapshot, _TargetQueue(), registry, warm)
    # A first-observation breakpoint (signal_settle_ms None) never settles (review P2-a).
    guessed = {"m": {"signal_settle_ms": None, "signal_breakpoint_ms": 120_000, "signal_warm": True}}
    assert "m" in _rescue_bases(snapshot, _TargetQueue(), registry, guessed)
    cold = {"m": {"signal_settle_ms": 113_500, "signal_warm": False, "routable_pods": 3}}
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
    # Pre-O1 guard: no scale-up on a trickle. O1 (2026-10-07): a CRITICAL receiver on a
    # free GPU is not held, so the filling window's false CRITICAL costs one wake.
    assert _run(SignalState(warmup_ms=-1), trickle, registry) == []
    assert _run(SignalState(warmup_ms=-1, breakpoint=O1), trickle, registry) == [(onset, 1)]

    burst = [IDLE, IDLE, IDLE] + [BURST] * 6
    o1 = _run(SignalState(warmup_ms=-1, breakpoint=O1), burst, registry)
    legacy = _run(SignalState(warmup_ms=-1), burst, registry)
    # 2026-10-07: CRITICAL on a free GPU acts at the onset, one step per decision (this
    # queue models no in-flight target, so every tick may step).
    assert o1 and o1[0] == (onset, 1) and all(n == 1 for _, n in o1)
    assert legacy and legacy[0][0] == onset + 3 * GRID  # the whole window after the onset
    # The fallback switch restores the pre-O1 behaviour exactly.
    fallback = BreakpointWindowConfig(enabled=False, onset_guard=True, grid_ms=GRID)
    assert _run(SignalState(warmup_ms=-1, breakpoint=fallback), burst, registry) == legacy
    assert _run(SignalState(warmup_ms=-1, breakpoint=fallback), trickle, registry) == []
    # Review P2-b: turning O1 off alone keeps the onset guard (never both off).
    off = BreakpointWindowConfig(enabled=False, onset_guard=False, grid_ms=GRID)
    assert _run(SignalState(warmup_ms=-1, breakpoint=off), burst, registry) == legacy
    assert _run(SignalState(warmup_ms=-1, breakpoint=off), trickle, registry) == []
    assert BreakpointWindowConfig.from_registry(_registry(breakpoint_window=False), grid_ms=GRID).enabled is False


# ----------------------------------------------------------------- registry


def test_scaling_registry_o1_keys():
    assert parse_scaling_config(None).breakpoint_window is True
    cfg = parse_scaling_config({"breakpoint_window": False, "onset_warmup_guard": True,
                                "min_evidence_grids": 3, "min_evidence_requests": 5})
    assert (cfg.breakpoint_window, cfg.onset_warmup_guard, cfg.min_evidence_grids, cfg.min_evidence_requests) == (
        False, True, 3, 5)
    assert parse_scaling_config({"breakpoint_margin_ms": 0}).breakpoint_margin_ms == 0
    assert parse_scaling_config(None).breakpoint_margin_ms == 1000
    for bad in ({"min_evidence_grids": 0}, {"min_evidence_grids": 1.5}, {"min_evidence_requests": -1},
                {"breakpoint_margin_ms": -1},
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
    assert scaling.min_evidence_grids == 2 and scaling.min_evidence_requests == 3
    assert scaling.breakpoint_partial_max_step == 1 and scaling.breakpoint_hold_max_windows == 6
    assert scaling.breakpoint_lowevidence_requests == 10
    assert scaling.gateway_clock_tolerance_ms == 2000 and scaling.gateway_clock_check_s == 60
    assert json.dumps(sorted(ScalingRegistryConfig.__dataclass_fields__))  # serialisable names
    assert math.isfinite(BreakpointWindowConfig().grid_ms)


def test_freeze_snapshot_freezes_the_suffix_pods():
    from types import MappingProxyType

    from tre_common.metrics_schema import PodWindowMetrics
    from tre_controller.loops.metrics_task import freeze_snapshot

    pod = PodWindowMetrics(pod="a", prompt_tokens=0.0, generation_tokens=1.0, avg_waiting=0.0, avg_running=1.0,
                           avg_swapping=0.0, kv_cache_hit_rate=0.0, ttft_p95_ms=None, tpot_p95_ms=None,
                           e2e_p95_ms=None)
    window = _window(60_000, [TRICKLE] * 3)
    window = replace(window, per_pod={"a": pod},
                     suffix_windows=tuple(replace(item, per_pod={"a": pod}) for item in window.suffix_windows))
    frozen = freeze_snapshot(_snap(window)).models["m"]
    assert all(isinstance(item.per_pod, MappingProxyType) for item in frozen.suffix_windows)
    with pytest.raises(TypeError):
        frozen.suffix_windows[0].per_pod["b"] = pod



# ===================================================== review round 2 (2026-10-01)


def test_partial_window_needs_three_completed_requests_by_default():
    """P2-1: tokens count at completion - one short request done in 20 s is no evidence."""
    state = SignalState(breakpoint=BreakpointWindowConfig(grid_ms=GRID))
    state.observe_traffic("m", has_traffic=True, window_start_ms=0, window_end_ms=30_000)
    thin = state.effective_window("m", _window(50_000, [IDLE, (64.0, 3.0, 0.0, 1.0), (0.0, 3.0, 0.0, 0.0)]))
    assert thin.grids == 2 and not thin.warm and thin.reason == "evidence_requests"
    assert state.effective_window("m", _window(50_000, [IDLE, TRICKLE, TRICKLE])).warm


def _c1_plan(full: bool | None, *, step: int = 1, z: float = 0.05, requests: float | None = 5.0,
             low: int = 10):
    cfg = PlanConfig(min_replicas_per_model=1, max_replicas_per_model=8, rescue_max_step_ratio=2.0,
                     rescue_max_step_pods=4, partial_window_max_step=step,
                     partial_window_lowevidence_requests=low)
    ctx = {"assigned_replicas": 1, "routable_pods": 1, "awake_replicas": 1, "signal_warm": True}
    if full is not None:
        ctx["signal_full_window"] = full
        ctx["signal_evidence_requests"] = None if full else requests
    return build_plan(model_contexts={"m": ctx}, classifications=[_cls("m", ModelState.CRITICAL, z)],
                      model_replicas={"m": 1}, idle_gpus=6, cfg=cfg)


def test_c1_step_is_capped_only_on_low_evidence_partial_windows():
    """P2-1 (evidence-gated): +1 on a partial window with < 10 completed requests (a
    low-QPS heavy-tailed load), the whole deficit (+4) with enough requests or a whole window."""
    low = _c1_plan(False, requests=4.0)
    assert sum(a.delta for a in _ups(low.actions)) == 1
    assert any(e.startswith("rescue_low_evidence_step:m:4->1:requests=4.0") for e in low.events)
    assert sum(a.delta for a in _ups(_c1_plan(False, requests=10.0).actions)) == 4
    assert sum(a.delta for a in _ups(_c1_plan(False, requests=None).actions)) == 1  # unknown = low
    assert sum(a.delta for a in _ups(_c1_plan(True).actions)) == 4
    assert sum(a.delta for a in _ups(_c1_plan(None).actions)) == 4  # pre-O1 contexts
    assert sum(a.delta for a in _ups(_c1_plan(False, requests=4.0, step=0).actions)) == 4  # cap off
    from tre_controller.loops.tick import _scaling_options

    options = _scaling_options(_registry())
    assert options["partial_window_max_step"] == 1 and options["partial_window_lowevidence_requests"] == 10
    assert _scaling_options(_registry(breakpoint_window=False))["partial_window_max_step"] == 0
    assert parse_scaling_config({"breakpoint_lowevidence_requests": 0}).breakpoint_lowevidence_requests == 0
    with pytest.raises(ValueError):
        parse_scaling_config({"breakpoint_lowevidence_requests": -1})


def test_low_qps_heavy_tail_scales_at_most_one_step_per_breakpoint():
    """0.1-0.3 rps with long requests: the 20 s suffix holds 3-9 completions; a short one
    done and long ones running reads Z far below tau_crit - the rescue adds +1, not +4."""
    registry = _registry(10_000.0, rescue_max_step_pods=4)
    state = SignalState(warmup_ms=-1, breakpoint=BreakpointWindowConfig(grid_ms=GRID))
    thin = (12.0, 4.0, 2.0, 2.0)  # 2 short completions per grid, 4 long requests running
    queue = _Queue()
    for i in range(2, 6):
        end = 1_000_000 + (i + 1) * GRID
        run_rescue_tick(_snap(_window(end, ([IDLE] * 3 + [thin] * 3)[i - 2 : i + 1])), queue=queue,
                        registry=registry, signal_state=state)
    ups = _ups(queue.submitted)
    # One step per decision (this queue models no in-flight target, so every tick may step).
    assert ups and all(a.delta == 1 for a in ups)
    # A deep overload (Z = 0.10) with 10+ completions in the suffix: the whole deficit.
    DEEP = (1000.0, 30.0, 100.0, 100.0)
    state = SignalState(warmup_ms=-1, breakpoint=BreakpointWindowConfig(grid_ms=GRID))
    queue = _Queue()
    for i in range(2, 6):
        end = 1_000_000 + (i + 1) * GRID
        run_rescue_tick(_snap(_window(end, ([IDLE] * 3 + [DEEP] * 3)[i - 2 : i + 1])), queue=queue,
                        registry=registry, signal_state=state)
    # Held ticks: +1 each (CRITICAL free-GPU exemption); the warm one the whole deficit,
    # 1 -> 4 (scaling cap; this queue keeps routable at 1).
    assert [a.delta for a in _ups(queue.submitted)] == [1, 1, 3]


def test_starving_receiver_falls_back_to_the_whole_window_after_hold_max_windows():
    """P2-2: a routable count that changes every window holds the model forever; after
    hold_max_windows held windows the receiver decides on the whole window, donors not."""
    registry = _registry(10_000.0)
    state = SignalState(warmup_ms=-1, breakpoint=replace(O1, hold_max_windows=4))
    _prime_onset(state, 500_000)
    seen = []
    for k in range(8):
        end = 1_000_000 + k * GRID
        routable = 1 + k % 2  # flaps every window
        ctx, events = _model_contexts(_snap(_window(end, [BURST] * 3, routable=routable)), registry,
                                      signal_state=state, cluster_view=_view(routable, fetched_ms=end + 3_000),
                                      queue=_Queue())
        seen.append((ctx["m"], events))
    held = [c["signal_warm"] for c, _ in seen]
    assert held[0] is True and held[1:4] == [False, False, False]  # k=0: first observation
    fallback = [i for i, (_c, ev) in enumerate(seen) if any(e.startswith("breakpoint_hold_fallback:m") for e in ev)]
    assert fallback and seen[fallback[0]][0]["signal_warm"] is True
    assert seen[fallback[0]][0]["signal_full_window"] is False  # donors still wait
    assert seen[fallback[0]][0]["signal_hold_reason"] is None
    # Not before the limit: no fallback in the first 4 held windows.
    assert fallback[0] >= 4


def test_hint_only_dates_a_change_in_its_own_direction():
    state = SignalState(breakpoint=O1)
    state.note_routable("m", 2, observed_ms=100_000)
    # A hide returned at 105 s, but the count went UP (external wake): view time.
    assert state.note_routable("m", 3, observed_ms=110_000, done_hints=((105_000, -1),)) == 111_000
    assert state.note_routable("m", 2, observed_ms=120_000, done_hints=((115_000, -1),)) == 116_000
    assert state.note_routable("m", 3, observed_ms=130_000, done_hints=(125_000,)) == 126_000  # unknown sign


def test_gateway_clock_check_suspends_and_resumes_o1():
    from tre_controller.gateway_clock import GatewayClockMonitor, evaluate_clock

    assert evaluate_clock(100_000, 95_000, period_ms=GRID, tolerance_ms=2_000).ok
    assert evaluate_clock(100_000, None, period_ms=GRID, tolerance_ms=2_000).ok
    assert evaluate_clock(100_000, 110_000, period_ms=GRID, tolerance_ms=2_000).reason == "gateway_ahead"
    assert evaluate_clock(100_000, 70_000, period_ms=GRID, tolerance_ms=2_000).reason == "gateway_behind_or_stalled"
    assert evaluate_clock(100_000, 101_500, period_ms=GRID, tolerance_ms=2_000).ok

    class _Redis:
        def __init__(self):
            self.newest = 0

        def smembers(self, key):
            return {b"default/pod-a"} if key.endswith(":m") else set()

        def zrange(self, key, start, end, withscores=False):
            return [(b"doc", float(self.newest))]

    redis = _Redis()
    now = {"ms": 1_000_000}
    state = SignalState(warmup_ms=-1, breakpoint=O1)
    monitor = GatewayClockMonitor(redis, ["m"], state, period_ms=GRID, tolerance_ms=2_000,
                                  resume_after=2, clock_ms=lambda: now["ms"])
    redis.newest = 1_000_000 + 160_000  # the gateway runs 160 s ahead (75 vs 76)
    assert not monitor.check().ok and state.breakpoint_window_suspended == "gateway_clock:gateway_ahead"
    # Suspended: whole windows, the onset guard, an event every tick.
    state.observe_traffic("m", has_traffic=True, window_start_ms=1_000_000, window_end_ms=1_030_000)
    assert state.effective_window("m", _window(1_040_000, [IDLE, BURST, BURST])).full
    assert state.onset_guard_applies()
    _ctx, events = _model_contexts(_snap(_window(1_040_000, [IDLE, BURST, BURST])), _registry(), signal_state=state)
    assert "breakpoint_window_suspended:gateway_clock:gateway_ahead" in events
    redis.newest = now["ms"] - 4_000
    assert monitor.check().ok and state.breakpoint_window_suspended is not None  # 1 good check
    assert monitor.check().ok and state.breakpoint_window_suspended is None  # resumed after 2
    assert not state.onset_guard_applies()


def test_gateway_clock_task_is_wired_only_with_o1_and_a_redis_store():
    from types import SimpleNamespace

    from tre_controller.app import _gateway_clock_monitor

    deps = SimpleNamespace(signal_state=SignalState(breakpoint=O1), store=SimpleNamespace(redis_client=object()),
                           registry=_registry())
    assert _gateway_clock_monitor(deps, SimpleNamespace(instant_sample_interval_ms=GRID)) is not None
    deps.registry = _registry(gateway_clock_check_s=0)
    assert _gateway_clock_monitor(deps, SimpleNamespace()) is None
    deps.registry = _registry()
    deps.signal_state = SignalState()
    assert _gateway_clock_monitor(deps, SimpleNamespace()) is None
    deps.signal_state = SignalState(breakpoint=O1)
    deps.store = SimpleNamespace()
    assert _gateway_clock_monitor(deps, SimpleNamespace()) is None


def test_held_receiver_event_names_the_hold_reason():
    cfg = PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4)
    ctx = {"m": {"assigned_replicas": 1, "routable_pods": 1, "signal_warm": False,
                 "signal_full_window": False, "signal_hold_reason": "evidence_grids"}}
    plan = build_plan(model_contexts=ctx, classifications=[_cls("m", ModelState.CRITICAL, 0.5)],
                      model_replicas={"m": 1}, idle_gpus=0, cfg=cfg)  # no free GPU: held
    assert "receiver_held_breakpoint_window:m:evidence_grids" in plan.events
    assert not any(e.startswith("receiver_suppressed_signal_warmup") for e in plan.events)


def test_store_full_window_bitwise_equal_with_suffixes_on_or_off():
    import dataclasses

    registry = load_registry(str(Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml"))
    redis = _store_fixture()
    on = MetricsStore(redis, registry, instant_sample_interval_ms=GRID, suffix_period_ms=GRID)
    off = MetricsStore(redis, registry, instant_sample_interval_ms=GRID)
    a = on.read_model_window("dsqwen-7b", 10_000, 40_000, use_cache=False, start_exclusive=True)
    b = off.read_model_window("dsqwen-7b", 10_000, 40_000, use_cache=False, start_exclusive=True)
    for field in dataclasses.fields(a):
        if field.name != "suffix_windows":
            assert getattr(a, field.name) == getattr(b, field.name), field.name
    for name in a.per_pod:
        for field in dataclasses.fields(a.per_pod[name]):
            assert getattr(a.per_pod[name], field.name) == getattr(b.per_pod[name], field.name), field.name


def test_safescale_observation_in_fallback_mode_equals_legacy_bitwise():
    from tre_controller.loops.safescale_task import _observation_from_metrics

    registry = _registry(10_000.0)
    legacy = SignalState(warmup_ms=-1)
    off = SignalState(warmup_ms=-1, breakpoint=BreakpointWindowConfig(enabled=False, grid_ms=GRID))
    pattern = [IDLE, IDLE, IDLE, TRICKLE, BURST, BURST, (900.0, 7.0, 1.0, 40.0), IDLE, IDLE, IDLE, BURST, BURST]
    for i in range(2, len(pattern)):
        end = 1_000_000 + (i + 1) * GRID
        window = _window(end, pattern[i - 2 : i + 1])
        a = _observation_from_metrics(end, window, registry.model("m"), "zm", signal_state=legacy)
        b = _observation_from_metrics(end, window, registry.model("m"), "zm", signal_state=off,
                                      routable_observation=(end, ()))
        assert (a.z_m, a.q_ctl, a.has_traffic) == (b.z_m, b.q_ctl, b.has_traffic), end



# ===================================================== review round 3 (2026-10-01)


def _c1_run(state_factory, registry, *, pattern, routables):
    """Rescue ticks of a model with a C1 target done at 1_013_500 (1 -> 2 at the view of
    1_023_000): returns per window end whether C1 kept its rescue basis."""
    from tre_controller.loops.action_queue import RescueTargetRecord

    class _TargetQueue(_Queue):
        def rescue_targets(self):
            return {"m": RescueTargetRecord(target=2, desired=2, base=1, covered_before=1, issued_ms=1_010_000,
                                            done_ms=1_013_500)}

    state = state_factory()
    _prime_onset(state, 500_000)
    queue = _TargetQueue({"m": (1_013_500, 1)})
    held = {}
    for k, (grids, routable) in enumerate(zip(pattern, routables)):
        end = 1_000_000 + k * GRID
        snap = _snap(_window(end, [grids] * 3, routable=routable))
        ctx, _ = _model_contexts(snap, registry, signal_state=state,
                                 cluster_view=_view(routable, fetched_ms=end + 3_000), queue=queue)
        held[end] = "m" in _rescue_bases(snap, queue, registry, ctx)
    return held


def test_c1_settle_without_o1_or_while_suspended_is_c1_alone():
    """Review P1: O1 off (fallback) or suspended (clock check) -> C1's window-start rule,
    bit for bit the C1-only behaviour (no SignalState breakpoint at all)."""
    registry = _registry(10_000.0)
    pattern = [BURST] * 9
    routables = [1, 1, 2, 2, 2, 2, 2, 2, 2]
    c1_alone = _c1_run(lambda: SignalState(warmup_ms=-1), registry, pattern=pattern, routables=routables)
    fallback = _c1_run(lambda: SignalState(warmup_ms=-1, breakpoint=BreakpointWindowConfig(enabled=False, grid_ms=GRID)),
                       registry, pattern=pattern, routables=routables)

    def suspended():
        state = SignalState(warmup_ms=-1, breakpoint=O1)
        state.suspend_breakpoint_window("gateway_clock:gateway_ahead")
        return state

    paused = _c1_run(suspended, registry, pattern=pattern, routables=routables)
    assert fallback == c1_alone == paused
    # The C1 rule: held until the window starts k*tau (20 s) after done (1_013_500).
    assert c1_alone[1_060_000] is True and c1_alone[1_070_000] is False
    o1 = _c1_run(lambda: SignalState(warmup_ms=-1, breakpoint=O1), registry, pattern=pattern, routables=routables)
    assert o1[1_050_000] is False  # O1 on: settles once warm after the change (earlier)


def test_hold_fallback_never_settles_c1():
    """Review P2-1: the whole-window fallback still holds the old replica count."""
    registry = _registry(10_000.0)
    state = SignalState(warmup_ms=-1, breakpoint=replace(O1, hold_max_windows=2))
    _prime_onset(state, 500_000)
    last = None
    for k in range(6):
        end = 1_000_000 + k * GRID
        routable = 1 + k % 2
        ctx, events = _model_contexts(_snap(_window(end, [BURST] * 3, routable=routable)), registry,
                                      signal_state=state, cluster_view=_view(routable, fetched_ms=end + 3_000),
                                      queue=_Queue())
        if any(e.startswith("breakpoint_hold_fallback:m") for e in events):
            last = ctx["m"]
    assert last is not None and last["signal_warm"] is True and last["signal_settle_ms"] is None


def test_gateway_clock_measures_the_offset_from_written_ms():
    """Review P2-2: with written_ms the offset is measured directly - a 1.5 s and a 3 s
    offset are told apart whatever the gateway's write phase."""
    from tre_controller.gateway_clock import GatewayClockMonitor, measure_written_offset_ms

    class _Gateway:
        """Writes a doc every 10 s at its own clock = controller clock + offset, phase 0."""

        def __init__(self, clock, offset_ms):
            self.clock, self.offset = clock, offset_ms

        def smembers(self, key):
            return {b"default/pod-a"} if key.endswith(":m") else set()

        def zrange(self, key, start, end, withscores=False):
            gw_now = self.clock["ms"] + self.offset
            boundary = gw_now // GRID * GRID
            return [(json.dumps({"timestamp": boundary, "written_ms": boundary}).encode(), float(boundary))]

    for offset, ok in ((1_500, True), (3_000, False), (-3_000, False), (0, True)):
        clock = {"ms": 1_000_000 + 4_321}

        def sleep(seconds, clock=clock):
            clock["ms"] += int(seconds * 1000)

        gateway = _Gateway(clock, offset)
        measured = measure_written_offset_ms(gateway, "tre:v2:inst:default/pod-a", clock_ms=lambda: clock["ms"],
                                             sleep_s=sleep, poll_ms=250, max_wait_ms=12_000)
        assert abs(measured - offset) <= 250, (offset, measured)
        state = SignalState(warmup_ms=-1, breakpoint=O1)
        monitor = GatewayClockMonitor(gateway, ["m"], state, period_ms=GRID, tolerance_ms=2_000,
                                      clock_ms=lambda: clock["ms"], sleep_s=sleep)
        assert monitor.check().ok is ok
        assert (state.breakpoint_window_suspended is None) is ok


def test_gateway_clock_falls_back_to_stamp_lag_without_written_ms():
    from tre_controller.gateway_clock import GatewayClockMonitor

    class _OldGateway:
        def smembers(self, key):
            return {b"default/pod-a"} if key.endswith(":m") else set()

        def zrange(self, key, start, end, withscores=False):
            return [(json.dumps({"timestamp": 1_160_000}).encode(), 1_160_000.0)]

    state = SignalState(warmup_ms=-1, breakpoint=O1)
    monitor = GatewayClockMonitor(_OldGateway(), ["m"], state, period_ms=GRID, tolerance_ms=2_000,
                                  clock_ms=lambda: 1_000_000, sleep_s=lambda _s: None)
    assert monitor.check().reason == "gateway_ahead"


# ------------------------------------------------- Q3: IDLE donors (2026-10-06)


def test_idle_donor_after_its_own_scale_down_releases_the_rest_at_once():
    """Q3: an idle window is the same evidence at any replica count. Right after its own
    scale-down (a breakpoint inside the window, no whole window after it) an IDLE model
    still releases the rest of its surplus - down to its floor, in one decision."""
    from tre_controller.loops.tick import run_planner_tick

    registry = _registry(10_000.0)
    state = SignalState(warmup_ms=-1, breakpoint=O1)
    base = 3_000_000
    queue = _Queue({"m": base + 1_500})  # our scale-down 4 -> 3 returned at base + 1.5 s
    run_planner_tick(_snap(_window(base, [IDLE] * 3, routable=4)), queue=queue, registry=registry,
                     rescue_due=True, fairness_due=False, cluster_view=_view(4, fetched_ms=base - 2_000),
                     signal_state=state)
    end = base + GRID
    pods = frozenset(f"m-{i}" for i in range(4))
    drained = replace(_window(end, [IDLE] * 3, routable=3), scrape_current_pods=pods,
                      gateway_inflight=dict.fromkeys(pods, 0))
    after = run_planner_tick(_snap(drained), queue=queue, registry=registry,
                             rescue_due=True, fairness_due=False, cluster_view=_view(3, fetched_ms=end + 3_000),
                             signal_state=state)
    ctx = after.model_contexts["m"]
    assert ctx["signal_full_window"] is False and ctx["window_idle"] is True
    assert after.classifications["m"].state == ModelState.IDLE
    assert [(a.model, a.delta) for a in after.actions if isinstance(a, ScaleAction)] == [("m", -2)]
    assert not any(e.startswith("donor_suppressed_breakpoint_window") for e in after.events)


def test_o1_donor_hold_exempts_only_current_window_idle_evidence():
    """Q3 / I4: a HIGH donor keeps the O1 hold; a held context (tokens missing, the last
    level carried over) is never idle evidence, so its IDLE level stays held too."""
    from tre_controller.loops.tick import PaperStateCache

    cfg = PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4)

    def plan(state: ModelState, ctx: dict):
        return build_plan(model_contexts={"m": {"assigned_replicas": 3, "routable_pods": 3, "awake_replicas": 3,
                                                "signal_warm": True, **ctx}},
                          classifications=[_cls("m", state, 2.0)], model_replicas={"m": 3}, idle_gpus=0, cfg=cfg)

    partial = {"signal_full_window": False, "window_idle": True}
    assert "donor_suppressed_breakpoint_window:m" not in plan(ModelState.IDLE, partial).events
    assert "donor_suppressed_breakpoint_window:m" in plan(ModelState.HIGH, partial).events
    cache = PaperStateCache()
    fresh = {"routable_pods": 3, "assigned_replicas": 3, "Y_m": 0.0, "Q": 0.0, "signal_full_window": True,
             "window_idle": True}
    cache.apply("m", fresh, tokens_available=True)
    held, _ = cache.apply("m", {"routable_pods": 3, "assigned_replicas": 3}, tokens_available=False)
    assert held["window_idle"] is False
    held_plan = plan(ModelState.IDLE, held)
    assert "donor_suppressed_breakpoint_window:m" in held_plan.events and not held_plan.actions

"""H3 (2026-10-06): the O1 receiver hold narrowed to its purpose.

O1 holds a receiver after a breakpoint so a window that still describes the old regime
does not over-scale it. On a free GPU an over-scale costs one wake: a CRITICAL receiver
whose queue rose since the breakpoint (a post-breakpoint instant sample above the newest
pre-breakpoint one) takes free capacity only, one step per decision. Donor steps, every
scale-down, LOW receivers and a flat queue keep the hold.
"""
from __future__ import annotations

from dataclasses import replace

from tre_common.metrics_schema import PodWindowMetrics
from tre_controller.loops.tick import run_planner_tick
from tre_controller.planning.classify import ModelState
from tre_controller.planning.planner import (
    PlanConfig,
    RescueBasis,
    ScaleAction,
    TransferIntent,
    build_plan,
)
from tre_controller.signals.trs import SignalState

from test_o1_breakpoint_window_20261001 import (
    BURST,
    GRID,
    O1,
    _cls,
    _prime_onset,
    _Queue,
    _registry,
    _snap,
    _view,
    _window,
)

RISE = {"breakpoint_ms": 1_000, "base_ms": 0, "base_q": 80.0, "base_waiting": 20.0, "base_pods": 1,
        "sample_ms": 10_000, "q": 120.0, "waiting": 30.0, "pods": 2}


def _held(**extra) -> dict:
    return {"assigned_replicas": 2, "routable_pods": 2, "awake_replicas": 2, "signal_warm": False,
            "signal_full_window": False, "signal_hold_reason": "no_complete_grid",
            "signal_evidence_requests": 500.0, **extra}


def _plan(contexts: dict, states: dict, *, idle_gpus: int, bases=None):
    cfg = PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4, rescue_max_step_ratio=2.0,
                     partial_window_max_step=1, partial_window_lowevidence_requests=10)
    return build_plan(
        model_contexts=contexts,
        classifications=[_cls(model, state, z) for model, (state, z) in states.items()],
        model_replicas={model: int(ctx["assigned_replicas"]) for model, ctx in contexts.items()},
        idle_gpus=idle_gpus, cfg=cfg, rescue_bases=bases,
    )


def _ups(plan, model: str = "m") -> int:
    return sum(a.delta for a in plan.actions if isinstance(a, ScaleAction) and a.model == model and a.delta > 0)


def test_critical_receiver_with_rising_queue_takes_a_free_gpu_despite_the_hold():
    plan = _plan({"m": _held(o1_queue_rise=RISE)}, {"m": (ModelState.CRITICAL, 0.05)}, idle_gpus=2)
    assert _ups(plan) == 1  # one step per decision (Z is the old regime's)
    assert any(e.startswith("receiver_o1_exempt_free_gpu:m:no_complete_grid:planned=1:bp=1000:q=80.0->120.0")
               for e in plan.events)
    assert not any(e.startswith("receiver_held_breakpoint_window:m") for e in plan.events)
    # A landed earlier rescue target (covered <= routable) does not block the step.
    landed = _plan({"m": _held(o1_queue_rise=RISE)}, {"m": (ModelState.CRITICAL, 0.05)}, idle_gpus=2,
                   bases={"m": RescueBasis(base=1, covered=2)})
    assert _ups(landed) == 1


def test_step_that_needs_a_donor_keeps_the_hold():
    donor = {"assigned_replicas": 3, "routable_pods": 3, "awake_replicas": 3, "signal_warm": True,
             "signal_full_window": True, "floor_headroom": 2}
    states = {"m": (ModelState.CRITICAL, 0.05), "d": (ModelState.HIGH, 3.0)}
    plan = _plan({"m": _held(o1_queue_rise=RISE), "d": dict(donor)}, states, idle_gpus=0)
    assert not [a for a in plan.actions if getattr(a, "receiver", None) == "m"]
    assert _ups(plan) == 0
    assert "receiver_held_breakpoint_window:m:no_complete_grid" in plan.events
    assert not any(e.startswith("receiver_o1_exempt_free_gpu") for e in plan.events)
    # control: the same receiver once warm does take from the donor
    warm = _plan({"m": _held(signal_warm=True), "d": dict(donor)}, states, idle_gpus=0)
    assert [a for a in warm.actions if isinstance(a, TransferIntent) and a.receiver_model == "m"]


def test_scale_down_and_low_receiver_after_a_breakpoint_stay_held():
    contexts = {
        "d": {"assigned_replicas": 3, "routable_pods": 3, "awake_replicas": 3, "signal_warm": True,
              "signal_full_window": False, "o1_queue_rise": RISE},
        "low": _held(o1_queue_rise=RISE),
    }
    plan = _plan(contexts, {"d": (ModelState.HIGH, 3.0), "low": (ModelState.LOW, 0.9)}, idle_gpus=2)
    assert "donor_suppressed_breakpoint_window:d" in plan.events
    assert "receiver_held_breakpoint_window:low:no_complete_grid" in plan.events
    assert not plan.actions


def test_flat_queue_keeps_the_hold_with_a_free_gpu():
    plan = _plan({"m": _held(o1_queue_rise=None)}, {"m": (ModelState.CRITICAL, 0.05)}, idle_gpus=2)
    assert _ups(plan) == 0
    assert "receiver_held_breakpoint_window:m:no_complete_grid" in plan.events


def _pods(end: int, samples: dict[str, tuple[float, float]]) -> dict[str, PodWindowMetrics]:
    """Per-pod docs whose newest instant sample (stamped ``end``) is (running, waiting)."""
    return {
        name: PodWindowMetrics(
            pod=name, prompt_tokens=0.0, generation_tokens=1.0, avg_waiting=waiting, avg_running=running,
            avg_swapping=0.0, kv_cache_hit_rate=0.0, ttft_p95_ms=100.0, tpot_p95_ms=10.0, e2e_p95_ms=1000.0,
            request_count=1.0, latest_waiting=waiting, latest_running=running, latest_gpu_cache=0.9,
            latest_instant_ms=end,
        )
        for name, (running, waiting) in samples.items()
    }


def _after_scale_up(post: dict[str, tuple[float, float]]):
    """1 -> 2 replicas (our wake returned 1.5 s after a boundary); the windows are the
    overloaded BURST, the pre-breakpoint sample is (running 30, waiting 20)."""
    registry = _registry(10_000.0)
    state = SignalState(warmup_ms=-1, breakpoint=O1)
    base = 4_000_000
    _prime_onset(state, base - 10 * GRID)
    queue = _Queue({"m": base + 1_500})
    before = replace(_window(base, [BURST] * 3, routable=1), per_pod=_pods(base, {"m-0": (30.0, 20.0)}))
    run_planner_tick(_snap(before), queue=queue, registry=registry, rescue_due=True, fairness_due=False,
                     cluster_view=_view(1, fetched_ms=base - 2_000), signal_state=state)
    end = base + GRID
    after = replace(_window(end, [BURST] * 3, routable=2), per_pod=_pods(end, post))
    return run_planner_tick(_snap(after), queue=queue, registry=registry, rescue_due=True, fairness_due=False,
                            cluster_view=_view(2, fetched_ms=base + 3_000), signal_state=state)


def test_tick_measures_the_queue_rise_from_samples_after_the_breakpoint():
    rising = _after_scale_up({"m-0": (30.0, 30.0), "m-1": (10.0, 0.0)})
    ctx = rising.model_contexts["m"]
    assert ctx["signal_warm"] is False and ctx["signal_hold_reason"] == "no_complete_grid"
    assert ctx["o1_queue_rise"]["waiting"] == 30.0 and ctx["o1_queue_rise"]["base_waiting"] == 20.0
    assert [(a.model, a.delta) for a in rising.actions if isinstance(a, ScaleAction)] == [("m", 1)]
    assert any(e.startswith("receiver_o1_exempt_free_gpu:m:") for e in rising.events)

    # The added replica absorbs the load: q 82.5 -> 56.25, waiting 20 -> 10 - held.
    flat = _after_scale_up({"m-0": (20.0, 10.0), "m-1": (10.0, 0.0)})
    assert flat.model_contexts["m"]["o1_queue_rise"] is None
    assert not [a for a in flat.actions if isinstance(a, ScaleAction)]
    assert "receiver_held_breakpoint_window:m:no_complete_grid" in flat.events


def test_queue_rise_counts_only_pods_of_both_samples_and_ignores_jitter():
    """Review P1-1: the woken pod's running requests are no rise while the old pod's
    waiting fell; a flat queue that jitters by one request is no rise either."""
    woke = _after_scale_up({"m-0": (30.0, 15.0), "m-1": (40.0, 0.0)})
    assert woke.model_contexts["m"]["o1_queue_rise"] is None
    assert not [a for a in woke.actions if isinstance(a, ScaleAction)]

    noisy = _after_scale_up({"m-0": (31.0, 21.0), "m-1": (0.0, 0.0)})
    assert noisy.model_contexts["m"]["o1_queue_rise"] is None

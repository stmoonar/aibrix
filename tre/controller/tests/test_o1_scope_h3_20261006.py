"""H3 (2026-10-06): the O1 receiver hold narrowed to its purpose.

O1 holds a receiver after a breakpoint so a window that still describes the old regime
does not over-scale it. On a free GPU an over-scale costs one wake: a CRITICAL receiver
takes free capacity only, one step per decision, once its previous step is visible.
Donor steps, every scale-down and LOW receivers keep the hold. 2026-10-07: the exemption
no longer needs a queue rise (``o1_queue_rise`` is still measured and logged).
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
from plan_shim import LegacyFakeQueue as ActionQueue  # pre-plan fakes behind POST /v2/plans

RISE = {"breakpoint_ms": 1_000, "base_ms": 0, "base_q": 80.0, "base_waiting": 20.0, "base_pods": 1,
        "sample_ms": 10_000, "q": 120.0, "waiting": 30.0, "pods": 2}


def _held(**extra) -> dict:
    return {"assigned_replicas": 2, "routable_pods": 2, "awake_replicas": 2, "signal_warm": False,
            "signal_full_window": False, "signal_hold_reason": "no_suffix",
            "signal_evidence_requests": 500.0, **extra}


def _plan(contexts: dict, states: dict, *, idle_gpus: int, bases=None, **kwargs):
    cfg = PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4, rescue_max_step_ratio=2.0,
                     partial_window_max_step=1, partial_window_lowevidence_requests=10)
    return build_plan(
        model_contexts=contexts,
        classifications=[_cls(model, state, z) for model, (state, z) in states.items()],
        model_replicas={model: int(ctx["assigned_replicas"]) for model, ctx in contexts.items()},
        idle_gpus=idle_gpus, cfg=cfg, rescue_bases=bases, **kwargs,
    )


def _ups(plan, model: str = "m") -> int:
    return sum(a.delta for a in plan.actions if isinstance(a, ScaleAction) and a.model == model and a.delta > 0)


def test_critical_receiver_with_rising_queue_takes_a_free_gpu_despite_the_hold():
    plan = _plan({"m": _held(o1_queue_rise=RISE)}, {"m": (ModelState.CRITICAL, 0.05)}, idle_gpus=2)
    assert _ups(plan) == 1  # one step per decision (Z is the old regime's)
    assert any(e.startswith("receiver_o1_exempt_free_gpu:m:no_suffix:planned=1:bp=1000:q=80.0->120.0")
               for e in plan.events)
    assert not any(e.startswith("receiver_held_breakpoint_window:m") for e in plan.events)


def test_exempt_receiver_does_not_take_the_free_gpu_a_confirmed_receiver_needs():
    """Review P2-1: the exempt receiver's wakes are assigned after the TSS-confirmed
    CRITICAL receivers: the one free GPU (where the exempt model also sleeps) goes to
    the confirmed receiver's create."""
    from tre_controller.planning.planner import ClusterView
    from tre_sm.allocator.slots import Binding, Slot
    from test_planner_slot_occupancy import TOPOLOGY

    awake = {"a": [("node9", 0), ("node9", 2), ("node9", 3), ("node10", 0)],
             "e": [("node10", 1), ("node10", 2)], "c": [("node10", 3)]}
    bindings = [Binding(f"{m}-{n}{g}", m, Slot(n, (g,)), awake=True) for m, gpus in awake.items() for n, g in gpus]
    bindings.append(Binding("e-s", "e", Slot("node9", (1,)), awake=False))  # the free GPU
    contexts = {
        "a": {"assigned_replicas": 4, "routable_pods": 4, "awake_replicas": 4, "signal_warm": True,
              "signal_full_window": True},
        "c": {"assigned_replicas": 1, "routable_pods": 1, "awake_replicas": 1, "signal_warm": True,
              "signal_full_window": True},
        "e": _held(assigned_replicas=3, o1_queue_rise=RISE),
    }
    states = {"a": (ModelState.HEALTHY, 1.1), "c": (ModelState.CRITICAL, 0.05), "e": (ModelState.CRITICAL, 0.05)}
    plan = _plan(contexts, states, idle_gpus=1, cluster_view=ClusterView(TOPOLOGY, tuple(bindings)))
    assert [(a.model, a.delta, a.reason) for a in plan.actions if isinstance(a, ScaleAction) and a.delta > 0] == [
        ("c", 1, "critical_idle_capacity")
    ]
    assert "receiver_held_breakpoint_window:e:no_suffix" in plan.events


def test_step_that_needs_a_donor_keeps_the_hold():
    donor = {"assigned_replicas": 3, "routable_pods": 3, "awake_replicas": 3, "signal_warm": True,
             "signal_full_window": True, "floor_headroom": 2}
    states = {"m": (ModelState.CRITICAL, 0.05), "d": (ModelState.HIGH, 3.0)}
    plan = _plan({"m": _held(o1_queue_rise=RISE), "d": dict(donor)}, states, idle_gpus=0)
    assert not [a for a in plan.actions if getattr(a, "receiver", None) == "m"]
    assert _ups(plan) == 0
    assert "receiver_held_breakpoint_window:m:no_suffix" in plan.events
    assert not any(e.startswith("receiver_o1_exempt_free_gpu") for e in plan.events)
    # control: the same receiver once warm does take from the donor (F1-B: through a
    # SafeScale probe of the HIGH donor, m its promised receiver)
    warm = _plan({"m": _held(signal_warm=True), "d": dict(donor)}, states, idle_gpus=0)
    assert warm.probe_upscale_plans.get("d", {}).get("m") == 1


def test_scale_down_and_low_receiver_after_a_breakpoint_stay_held():
    contexts = {
        "d": {"assigned_replicas": 3, "routable_pods": 3, "awake_replicas": 3, "signal_warm": True,
              "signal_full_window": False, "o1_queue_rise": RISE},
        "low": _held(o1_queue_rise=RISE),
    }
    plan = _plan(contexts, {"d": (ModelState.HIGH, 3.0), "low": (ModelState.LOW, 0.9)}, idle_gpus=2)
    assert "donor_suppressed_breakpoint_window:d" in plan.events
    assert "receiver_held_breakpoint_window:low:no_suffix" in plan.events
    assert not plan.actions


def test_critical_receiver_on_a_free_gpu_needs_no_queue_rise_one_wake_per_settled_step():
    """2026-10-07 regression: after a wake the old pods' waiting drains while running
    stays high, so a queue rise never showed and the CRITICAL receiver sat out 2 grids
    with free GPUs. CRITICAL + free GPU + O1 evidence hold + no unsettled rescue target
    -> one step, no rise needed. One stale window buys one wake: an earlier target O1
    has not settled (landed or not) or a pending view -> pending, no second wake; once
    settled the receiver acts again. LOW, a held (tokens_missing) context and a stale
    scrape -> held."""
    crit = {"m": (ModelState.CRITICAL, 0.05)}
    for reason in ("no_suffix", "evidence_span", "evidence_tokens", "evidence_requests"):
        plan = _plan({"m": _held(o1_queue_rise=None, signal_hold_reason=reason)}, crit, idle_gpus=2)
        assert _ups(plan) == 1
        assert any(e.startswith(f"receiver_o1_exempt_free_gpu:m:{reason}:planned=1:")
                   and e.endswith(":queue_rise=no") for e in plan.events)
        assert not any(e.startswith("receiver_held_breakpoint_window:m") for e in plan.events)
    # The exempt wake 2 -> 3 landed (routable 3) but O1 has not settled it, or it is not
    # visible yet, or the view predates the last action: no second wake.
    landed = _plan({"m": _held(o1_queue_rise=None, routable_pods=3, awake_replicas=3, assigned_replicas=3)},
                   crit, idle_gpus=1, bases={"m": RescueBasis(base=2, covered=3)})
    unseen = _plan({"m": _held(o1_queue_rise=None)}, crit, idle_gpus=2, bases={"m": RescueBasis(base=2, covered=3)})
    view = _plan({"m": _held(o1_queue_rise=None)}, crit, idle_gpus=2, view_pending={"m": "up"})
    for pending in (landed, unseen, view):
        assert _ups(pending) == 0
        assert any(e.startswith("receiver_o1_exempt_pending:m:") for e in pending.events)
        assert "receiver_held_breakpoint_window:m:no_suffix" in pending.events
    # Settled (``_rescue_bases`` drops the target): warm -> the normal TSS rescue.
    settled = _plan({"m": _held(signal_warm=True, signal_full_window=True, signal_hold_reason=None,
                                routable_pods=3, awake_replicas=3, assigned_replicas=3)},
                    crit, idle_gpus=1)
    assert _ups(settled) == 1
    assert not any(e.startswith(("receiver_o1_exempt", "receiver_held_breakpoint_window")) for e in settled.events)
    for reason in ("tokens_missing", "scrape_stale", None):
        stale = _plan({"m": _held(o1_queue_rise=None, signal_hold_reason=reason)}, crit, idle_gpus=2)
        assert _ups(stale) == 0
        assert not any(e.startswith("receiver_o1_exempt") for e in stale.events)
    low = _plan({"m": _held(o1_queue_rise=None)}, {"m": (ModelState.LOW, 0.9)}, idle_gpus=2)
    assert _ups(low) == 0
    assert "receiver_held_breakpoint_window:m:no_suffix" in low.events


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
    assert ctx["signal_warm"] is False and ctx["signal_hold_reason"] == "no_suffix"
    assert ctx["o1_queue_rise"]["waiting"] == 30.0 and ctx["o1_queue_rise"]["base_waiting"] == 20.0
    assert [(a.model, a.delta) for a in rising.actions if isinstance(a, ScaleAction)] == [("m", 1)]
    assert any(e.startswith("receiver_o1_exempt_free_gpu:m:") for e in rising.events)

    # The added replica absorbs the load: q 82.5 -> 56.25, waiting 20 -> 10 - no rise.
    flat = _after_scale_up({"m-0": (20.0, 10.0), "m-1": (10.0, 0.0)})
    assert flat.model_contexts["m"]["o1_queue_rise"] is None


def test_queue_rise_counts_only_pods_of_both_samples_and_ignores_jitter():
    """Review P1-1: the woken pod's running requests are no rise while the old pod's
    waiting fell; a flat queue that jitters by one request is no rise either."""
    woke = _after_scale_up({"m-0": (30.0, 15.0), "m-1": (40.0, 0.0)})
    assert woke.model_contexts["m"]["o1_queue_rise"] is None

    noisy = _after_scale_up({"m-0": (31.0, 21.0), "m-1": (0.0, 0.0)})
    assert noisy.model_contexts["m"]["o1_queue_rise"] is None


def test_low_wake_holds_the_exempt_free_gpu_step_until_o1_settles():
    """Smoke 20261026 regression (any scale-up path is the exempt step's basis): a LOW
    model woke 1 -> 2 (``low_fairness_sleeping_capacity``, no rescue target); the next
    window still straddles that breakpoint and, its old tokens spread over 2 replicas,
    shows a lower Z -> CRITICAL, O1-exempt, free GPUs. One stale window buys one wake:
    no second wake on it. Once O1 decides on evidence after the breakpoint and the
    receiver is still CRITICAL, the next step is planned."""
    import asyncio

    class _Client:
        async def set_binding_power(self, serve_id, *, awake):
            return {"ok": True}

        async def scale_model(self, model, delta, **_kwargs):
            return {"ok": True}

    def ups(result):
        return [(a.model, a.delta, a.reason) for a in result.actions if isinstance(a, ScaleAction) and a.delta > 0]

    low_load, stale_load = (2600.0, 30.0, 20.0, 100.0), (1000.0, 30.0, 20.0, 100.0)  # Z 0.95 (LOW), ~0.36
    registry = _registry(10_000.0)
    state = SignalState(warmup_ms=-1, breakpoint=O1)
    base = 4_000_000
    _prime_onset(state, base - 10 * GRID)
    now = {"ms": base}
    queue = ActionQueue(_Client(), now_ms=lambda: now["ms"])

    def tick(end, grids, routable, fairness=False):
        return run_planner_tick(_snap(_window(end, [grids] * 3, routable=routable)), queue=queue, registry=registry,
                                rescue_due=True, fairness_due=fairness,
                                cluster_view=_view(routable, fetched_ms=end + 3_000), signal_state=state)

    first = tick(base, low_load, 1, fairness=True)
    assert ups(first) == [("m", 1, "low_fairness_sleeping_capacity")]
    now["ms"] = base + 1_500
    asyncio.run(queue.drain_once())  # the wake returns: 2 routable replicas

    held = tick(base + GRID, stale_load, 2)
    ctx = held.model_contexts["m"]
    assert ctx["signal_warm"] is False and ctx["signal_hold_reason"] == "no_suffix"
    assert held.classifications["m"].state == ModelState.CRITICAL
    assert ups(held) == []  # free GPUs m-2 / m-3, but the stale window already bought its wake
    assert any(e.startswith("receiver_o1_exempt_pending:m:") for e in held.events)
    assert not any(e.startswith("receiver_o1_exempt_free_gpu:m") for e in held.events)

    steps = {}
    for k in (2, 3, 4):
        result = tick(base + k * GRID, stale_load, 2)
        steps[k] = (result.model_contexts["m"]["signal_warm"], ups(result))
        if steps[k][1]:
            break
    warm_at = next(k for k, (warm, _) in steps.items() if warm)
    assert all(not planned for k, (_, planned) in steps.items() if k < warm_at)
    assert steps[warm_at][1] and steps[warm_at][1][0][0] == "m"  # settled on new evidence: next step


def test_released_upscale_record_leaves_a_later_unrelated_breakpoint_exempt():
    """The record releases by the window rule (an exempt receiver is never warm): once a
    whole window follows the LOW wake, a breakpoint of another cause (here a replica lost
    outside the controller) is its own stale window and buys its own exempt wake."""
    import asyncio

    class _Client:
        async def set_binding_power(self, serve_id, *, awake):
            return {"ok": True}

        async def scale_model(self, model, delta, **_kwargs):
            return {"ok": True}

    low_load, healthy_load, stale_load = (
        (2600.0, 30.0, 20.0, 100.0), (2600.0, 30.0, 15.0, 100.0), (1000.0, 30.0, 20.0, 100.0)
    )
    registry = _registry(10_000.0)
    state = SignalState(warmup_ms=-1, breakpoint=O1)
    base = 4_000_000
    _prime_onset(state, base - 10 * GRID)
    now = {"ms": base}
    queue = ActionQueue(_Client(), now_ms=lambda: now["ms"])

    def tick(end, grids, routable, fairness=False):
        return run_planner_tick(_snap(_window(end, [grids] * 3, routable=routable)), queue=queue, registry=registry,
                                rescue_due=True, fairness_due=fairness,
                                cluster_view=_view(routable, fetched_ms=end + 3_000), signal_state=state)

    assert [a.reason for a in tick(base, low_load, 1, fairness=True).actions if isinstance(a, ScaleAction)] == [
        "low_fairness_sleeping_capacity"
    ]
    now["ms"] = base + 1_500
    asyncio.run(queue.drain_once())
    for k in range(1, 5):  # healthy at 2 replicas: no action; window rule releases the wake
        assert not tick(base + k * GRID, healthy_load, 2).actions
    lost = tick(base + 5 * GRID, stale_load, 1)  # m-1 gone (not ours): a new breakpoint
    assert lost.model_contexts["m"]["signal_warm"] is False
    assert [(a.model, a.delta) for a in lost.actions if isinstance(a, ScaleAction) and a.delta > 0] == [("m", 1)]
    assert any(e.startswith("receiver_o1_exempt_free_gpu:m:") for e in lost.events)

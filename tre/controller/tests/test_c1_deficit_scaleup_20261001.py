"""C1 (2026-10-01): the fast-loop rescue asks for a CRITICAL receiver's whole deficit
at once (an absolute target), the CRITICAL scale-up cooldown is opt-in, and the rescue
target bookkeeping keeps a scale-up the decision window does not reflect yet from
being repeated. The slow loop and every scale-down path are unchanged.
Design: docs/design/20261001-c1-deficit-scaleup.md."""
from __future__ import annotations

import asyncio
import logging
from relay_view import expand_relays, relays  # noqa: F401 - 2026-10-02 relay intents

import pytest
import yaml

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import (
    ClusterTopology,
    NodeSpec,
    Registry,
    ScalingRegistryConfig,
    load_registry,
    parse_scaling_config,
)
from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.rescue_task import run_rescue_tick
from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, TauThresholds
from tre_controller.planning.planner import (
    ClusterView,
    PlanConfig,
    RescueBasis,
    RescuePlan,
    ScaleAction,
    build_plan,
    rescue_desired,
)
from tre_sm.allocator.slots import Binding, Slot

from test_loop_ticks import _registry as _base_registry

TAU = TauThresholds.from_control()  # tau_crit 0.8, tau_low 1.0, tau_high 1.25


def _cls(model, state, z, role=None, tier=None):
    role = role or {
        ModelState.CRITICAL: ModelRole.RECEIVER,
        ModelState.LOW: ModelRole.RECEIVER,
        ModelState.HIGH: ModelRole.DONOR,
        ModelState.IDLE: ModelRole.DONOR,
    }.get(state, ModelRole.NEUTRAL)
    return ModelClassification(
        model_name=model, state=state, role=role, Z_m=z, eta_m=None, trs=0.0,
        theta_m=1.0, tau=TAU, donor_tier=tier,
    )


def _plan(classifications, pods, *, idle_gpus=0, max_replicas=8, ratio=2.0, bases=None,
          inflight=None, rescue_due=True, fairness_due=True, cluster_view=None, tp=None,
          step_pods=0):
    contexts = {model: {"routable_pods": n, "assigned_replicas": n} for model, n in pods.items()}
    return build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas=dict(pods),
        idle_gpus=idle_gpus,
        cfg=PlanConfig(
            min_replicas_per_model=1,
            max_replicas_per_model=max_replicas,
            rescue_due=rescue_due,
            fairness_due=fairness_due,
            suppress_hot_proactive_probe=True,
            rescue_max_step_ratio=ratio,
            model_tp_sizes=tp or {},
            rescue_max_step_pods=step_pods,
        ),
        rescue_bases=bases,
        inflight_models=inflight,
        cluster_view=cluster_view,
    )


def _deltas(plan):
    out: dict[str, int] = {}
    for action in expand_relays(plan.actions):
        if isinstance(action, ScaleAction):
            out[action.model] = out.get(action.model, 0) + action.delta
    return out


# --------------------------------------------------------------------- desired
@pytest.mark.parametrize(
    ("n", "z", "ratio", "expected"),
    [
        (2, 0.5, 2.0, 4),     # ceil(2 * 0.8 / 0.5) = 4
        (4, 0.6, 2.0, 6),     # ceil(5.33) = 6
        (4, 0.79, 2.0, 5),    # just below tau_crit: ceil(4.05) = 5
        (4, 0.7999, 2.0, 5),  # never below n + 1
        (2, 0.1, 2.0, 4),     # ceil(16) capped at 2n
        (2, 0.1, 3.0, 6),     # ... or ratio x n
        (1, 0.1, 2.0, 2),
        (3, 0.1, 1.0, 4),     # ratio 1: n + 1 at most
        (2, None, 2.0, 3),    # Z missing -> n + 1
        (2, 0.0, 2.0, 3),     # Z <= 0 -> n + 1
        (2, -1.0, 2.0, 3),
        (0, 0.2, 2.0, 1),     # no routable replica: one
    ],
)
def test_rescue_desired(n, z, ratio, expected):
    assert rescue_desired(n, z, 0.8, ratio) == expected


# ------------------------------------------------------------- one-shot target
def test_critical_receiver_gets_the_whole_deficit_in_one_tick():
    plan = _plan([_cls("r", ModelState.CRITICAL, 0.4)], {"r": 2}, idle_gpus=4)

    ups = [a for a in expand_relays(plan.actions) if isinstance(a, ScaleAction) and a.delta > 0]
    assert [(a.model, a.delta, a.reason) for a in ups] == [("r", 2, "critical_idle_capacity")]
    assert ups[0].rescue == RescuePlan(target=4, desired=4, base=2, covered=2)
    assert "rescue_target:r:n=2:z=0.4000:desired=4:covered=2:planned=2" in plan.events


def test_legacy_ratio_zero_keeps_the_one_step_rescue():
    plan = _plan([_cls("r", ModelState.CRITICAL, 0.4)], {"r": 2}, idle_gpus=4, ratio=0)
    assert _deltas(plan) == {"r": 1}
    assert all(a.rescue is None for a in plan.actions)
    assert not any(e.startswith("rescue_target") for e in plan.events)


def test_desired_is_capped_by_the_scaling_cap():
    plan = _plan([_cls("r", ModelState.CRITICAL, 0.1)], {"r": 2}, idle_gpus=4, max_replicas=3)
    assert _deltas(plan) == {"r": 1}


def test_missing_z_falls_back_to_one_more_replica():
    # A CRITICAL classification always has a Z; a non-positive one (alt signal) or a
    # missing one must still plan n + 1, never a division error.
    plan = _plan([_cls("r", ModelState.CRITICAL, 0.0)], {"r": 3}, idle_gpus=4)
    assert _deltas(plan) == {"r": 1}


def test_capacity_short_plans_what_exists_and_records_the_partial_target():
    plan = _plan([_cls("r", ModelState.CRITICAL, 0.2)], {"r": 2}, idle_gpus=1)
    ups = [a for a in expand_relays(plan.actions) if isinstance(a, ScaleAction) and a.delta > 0]
    assert [a.delta for a in ups] == [1]
    assert ups[0].rescue == RescuePlan(target=3, desired=4, base=2, covered=2)


def test_idle_donor_gives_its_surplus_and_a_high_donor_only_a_probe():
    """F1-B (2026-10-07): an IDLE donor gives its whole surplus at once (Q3 2026-10-06:
    an idle window is evidence at any replica count); a HIGH donor gives one step, and
    only through a SafeScale probe (the receiver's replica comes with the commit)."""
    idle = [_cls("r", ModelState.CRITICAL, 0.2), _cls("i", ModelState.IDLE, 10.0, tier="idle")]
    assert _deltas(_plan(idle, {"r": 3, "i": 4})) == {"r": 3, "i": -3}  # floor 1
    high = [_cls("r", ModelState.CRITICAL, 0.4), _cls("d", ModelState.HIGH, 2.5, tier="surplus")]
    plan = _plan(high, {"r": 2, "d": 4})
    assert _deltas(plan) == {"d": -1}
    assert [a.requires_safescale for a in plan.actions if a.model == "d"] == [True]
    assert plan.probe_upscale_plans.get("d", {}).get("r") == 1


def test_tp_receiver_takes_several_free_slot_pairs_in_one_action():
    topology = ClusterTopology(
        nodes=(
            NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),
            NodeSpec(name="node-b", gpus=4, two_gpu_slots=((0, 1), (2, 3))),
        )
    )
    view = ClusterView(
        topology=topology,
        bindings=(
            Binding("r-0", "r", Slot("node-a", (0, 1)), awake=True),
            Binding("r-1", "r", Slot("node-a", (2, 3)), awake=True),
        ),
    )
    plan = _plan([_cls("r", ModelState.CRITICAL, 0.2)], {"r": 2}, cluster_view=view, tp={"r": 2})
    ups = [a for a in expand_relays(plan.actions) if isinstance(a, ScaleAction) and a.delta > 0]
    assert [(a.delta, a.reason) for a in ups] == [(2, "critical_empty_slot")]


# --------------------------------------------------- idempotence / protection
def test_unrefreshed_window_recomputes_the_same_target_and_adds_nothing():
    first = _plan([_cls("r", ModelState.CRITICAL, 0.4)], {"r": 2}, idle_gpus=4)
    target = next(a.rescue for a in expand_relays(first.actions) if isinstance(a, ScaleAction))
    basis = {"r": RescueBasis(base=target.base, covered=target.target)}
    # The cluster view may already show the woken replicas (4) or still the old count (2).
    for routable in (2, 4):
        again = _plan([_cls("r", ModelState.CRITICAL, 0.4)], {"r": routable}, idle_gpus=4, bases=basis)
        assert _deltas(again) == {}
        assert "rescue_target_hold:r:desired=4:covered=4" in again.events


def test_load_still_rising_raises_the_target_by_the_difference_only():
    basis = {"r": RescueBasis(base=4, covered=6)}  # 4 -> 6 issued from Z=0.6
    plan = _plan([_cls("r", ModelState.CRITICAL, 0.4)], {"r": 6}, idle_gpus=4, bases=basis)
    ups = [a for a in expand_relays(plan.actions) if isinstance(a, ScaleAction) and a.delta > 0]
    assert sum(a.delta for a in ups) == 2  # desired ceil(4 * 0.8 / 0.4) = 8 = cap 2 * 4
    assert ups[0].rescue == RescuePlan(target=8, desired=8, base=4, covered=6)


def test_failed_part_of_a_target_is_planned_again():
    # The earlier target reached only 3 of 4 (covered 3): the missing one is re-planned.
    basis = {"r": RescueBasis(base=2, covered=3)}
    plan = _plan([_cls("r", ModelState.CRITICAL, 0.4)], {"r": 3}, idle_gpus=4, bases=basis)
    assert _deltas(plan) == {"r": 1}


def test_inflight_receiver_is_not_planned_again():
    plan = _plan([_cls("r", ModelState.CRITICAL, 0.1)], {"r": 2}, idle_gpus=4, inflight={"r"})
    assert _deltas(plan) == {}


# ------------------------------------------------- unchanged: slow loop, downs
def test_slow_loop_moves_at_most_one_pair_per_receiver_and_is_unchanged_by_c1():
    classifications = [
        _cls("low", ModelState.LOW, 0.9),
        _cls("d1", ModelState.HIGH, 2.5, tier="surplus"),
        _cls("d2", ModelState.HIGH, 2.0, tier="surplus"),
    ]
    pods = {"low": 4, "d1": 4, "d2": 4}
    c1 = _plan(classifications, pods, rescue_due=False)
    legacy = _plan(classifications, pods, rescue_due=False, ratio=0)
    assert c1.actions == legacy.actions and c1.events == legacy.events
    # F1-B: one probe of the first HIGH donor, its freed replica promised to "low".
    probes = [a for a in c1.actions if isinstance(a, ScaleAction) and a.requires_safescale]
    assert [(a.model, a.delta, a.reason) for a in probes] == [("d1", -1, "low_fairness_high_donor_safescale")]
    assert c1.probe_upscale_plans.get("d1", {}).get("low") == 1


def test_scale_down_paths_are_unchanged_by_c1():
    classifications = [
        _cls("i", ModelState.IDLE, 10.0, tier="idle"),
        _cls("h", ModelState.HIGH, 3.0, tier="surplus"),
        _cls("ok", ModelState.HEALTHY, 1.1),
    ]
    pods = {"i": 4, "h": 4, "ok": 2}
    for kwargs in ({}, {"fairness_due": False}):
        c1 = _plan(classifications, pods, **kwargs)
        legacy = _plan(classifications, pods, ratio=0, **kwargs)
        assert c1.actions == legacy.actions
        assert c1.delayed_down_models == legacy.delayed_down_models
        assert c1.events == legacy.events
    assert _deltas(_plan(classifications, pods)) == {"i": -3}  # Q3: IDLE to its floor at once


# ------------------------------------------------------ queue + tick wiring
class _Clock:
    def __init__(self, now_ms: int) -> None:
        self.now = now_ms

    def __call__(self) -> int:
        return self.now


class _Client:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.calls: list[tuple[str, int]] = []

    async def scale_model(self, model, delta, **_kwargs):
        self.calls.append((model, delta))
        return {"ok": True} if self.ok else {"ok": False, "error": "HTTP 409: WakeConflict"}

    async def set_routable(self, model, hidden_pods):
        return {"ok": True}

    async def set_binding_power(self, serve_id, *, awake, **_kwargs):
        return {"ok": True}

    async def defrag(self, migrations):
        return {"ok": True}


def _snapshot(window_start_ms: int, *, running: float = 1.0) -> MetricsSnapshot:
    return MetricsSnapshot(
        ts_ms=window_start_ms + 60_000,
        stale=False,
        models={
            "critical": ModelWindowMetrics(
                model="critical", window_start_ms=window_start_ms, window_end_ms=window_start_ms + 60_000,
                prompt_tokens=0.0, generation_tokens=50.0, avg_waiting=10.0, avg_running=running,
                avg_swapping=0.0, kv_cache_hit_rate=0.0, ttft_p95_ms=100.0, tpot_p95_ms=10.0,
                e2e_p95_ms=1000.0, routable_pods=2, assigned_replicas=2, per_pod={},
            )
        },
    )


def _registry(**scaling) -> Registry:
    base = _base_registry()
    return Registry(base.topology(), list(base.models()), scaling=ScalingRegistryConfig(**scaling))


def test_queue_records_the_target_and_the_tick_holds_until_the_window_reflects_it():
    client = _Client()
    clock = _Clock(65_000)
    queue = ActionQueue(client, now_ms=clock)
    registry = _registry()

    first = run_rescue_tick(_snapshot(5_000), queue=queue, registry=registry)
    record = queue.rescue_targets()["critical"]
    assert record.done_ms is None and record.outstanding == 1  # in flight
    asyncio.run(queue.drain_once())
    assert first.submitted == 1 and client.calls == [("critical", 2)]
    record = queue.rescue_targets()["critical"]
    assert (record.base, record.target, record.covered, record.done_ms) == (2, 4, 4, 65_000)

    # Many ticks on windows that started before the wake completed: nothing more.
    for start in (5_000, 20_000, 64_999):
        held = run_rescue_tick(_snapshot(start), queue=queue, registry=registry)
        asyncio.run(queue.drain_once())
        assert held.submitted == 0
    assert client.calls == [("critical", 2)]


def test_queue_failed_target_is_replanned_on_the_next_tick():
    client = _Client(ok=False)
    queue = ActionQueue(client, now_ms=_Clock(65_000))
    registry = _registry()

    run_rescue_tick(_snapshot(5_000), queue=queue, registry=registry)
    asyncio.run(queue.drain_once())
    record = queue.rescue_targets()["critical"]
    assert (record.covered, record.failures, record.done_ms) == (2, 1, 65_000)

    again = run_rescue_tick(_snapshot(10_000), queue=queue, registry=registry)
    asyncio.run(queue.drain_once())
    assert again.submitted == 1 and client.calls == [("critical", 2), ("critical", 2)]


def test_queue_observe_mode_drops_the_target_without_counting_it():
    queue = ActionQueue(_Client(), now_ms=_Clock(1_000), is_observe=lambda: True)
    plan = RescuePlan(target=4, desired=4, base=2, covered=2)
    queue.submit([ScaleAction("m", 2, "critical_idle_capacity", "rescue", receiver="m", rescue=plan)])
    asyncio.run(queue.drain_once())
    record = queue.rescue_targets()["m"]
    assert (record.covered, record.outstanding, record.done_ms) == (2, 0, 1_000)


def test_queue_counts_every_part_of_one_target():
    clock = _Clock(1_000)
    queue = ActionQueue(_Client(), now_ms=clock)
    plan = RescuePlan(target=5, desired=5, base=2, covered=2)
    queue.submit([
        ScaleAction("m", 1, "critical_sleeping_capacity", "rescue", receiver="m", pods=("m-3",), rescue=plan),
        ScaleAction("m", 2, "critical_idle_capacity", "rescue", receiver="m", rescue=plan),
    ])
    assert queue.rescue_targets()["m"].outstanding == 2
    clock.now = 2_000
    asyncio.run(queue.drain_once())
    record = queue.rescue_targets()["m"]
    assert (record.covered, record.outstanding, record.done_ms) == (5, 0, 2_000)


def test_scale_up_cooldown_key_is_ignored(caplog):
    # Timer cleanup (2026-10-02): the opt-in F4 hold of a C1 rescue was dead code (off in
    # every shipped registry) and was removed; an old registry with the key still loads.
    queue = ActionQueue(_Client(), now_ms=_Clock(65_000))
    queue._last_done["critical"] = (65_000, "up")
    result = run_rescue_tick(_snapshot(5_000), queue=queue, registry=_registry(), action_cooldown=True)
    assert result.submitted == 1 and "cooldown_hold:critical" not in result.events
    with caplog.at_level(logging.WARNING):
        assert parse_scaling_config({"scale_up_cooldown_enabled": True}) == ScalingRegistryConfig()
    assert "scale_up_cooldown_enabled" in caplog.text

# ---------------------------------------------------------------- registry
def test_scaling_registry_defaults_and_validation(caplog):
    assert parse_scaling_config(None) == ScalingRegistryConfig()
    assert parse_scaling_config({"rescue_max_step_ratio": 0}).rescue_max_step_ratio == 0.0
    assert parse_scaling_config({"rescue_max_step_ratio": 1}).rescue_max_step_ratio == 1.0
    for bad in ({"rescue_max_step_ratio": 0.5}, {"rescue_max_step_ratio": -1},
                {"rescue_max_step_ratio": "x"}, {"rescue_max_step_ratio": True},
                ["not", "a", "mapping"]):
        with pytest.raises(ValueError):
            parse_scaling_config(bad)
    with caplog.at_level(logging.WARNING):
        parse_scaling_config({"rescue_max_step_ratio": 2, "future_key": 1})
    assert "future_key" in caplog.text


def test_shipped_registry_scaling_section():
    registry = load_registry()
    # Built-in defaults except rescue_max_step_pods: 4 (user decision 2026-10-01).
    assert registry.scaling() == ScalingRegistryConfig(rescue_max_step_pods=4)
    raw = yaml.safe_load(open(__import__("pathlib").Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml",
                              encoding="utf-8"))
    assert set(raw["scaling"]) == {
        "rescue_max_step_ratio", "rescue_max_step_pods",
        "rescue_settle_ema_k",
        # O1 (2026-10-01): the breakpoint window, at its built-in defaults.
        "breakpoint_window", "onset_warmup_guard", "min_evidence_grids", "min_evidence_requests",
        "breakpoint_margin_ms", "breakpoint_partial_max_step", "breakpoint_lowevidence_requests",
        "breakpoint_hold_max_windows",
        "gateway_clock_tolerance_ms", "gateway_clock_check_s",
        # Onset saturation rescue (2026-10-02), at its built-in defaults.
        "saturation_rescue", "saturation_kv_threshold", "saturation_consecutive_ticks",
        "saturation_max_step_factor",
    }
    assert not hasattr(registry.scaling(), "donor_surplus_release")  # retired 2026-10-07 (F1-B)
    assert registry.scaling().rescue_max_step_pods == 4
    assert ScalingRegistryConfig().rescue_max_step_pods == 0  # code default unchanged
    # With the shipped registry a single replica reaches the scaling cap (4) at once.
    plan = _plan([_cls("r", ModelState.CRITICAL, 0.1)], {"r": 1}, idle_gpus=8, max_replicas=4,
                 step_pods=registry.scaling().rescue_max_step_pods)
    assert _deltas(plan) == {"r": 3}


# ===================================================== review round (2026-10-01)
from dataclasses import replace as _replace  # noqa: E402

from tre_controller.loops.tick import _apply_safescale, _rescue_bases, rescue_settle_ms  # noqa: E402
from tre_controller.planning.planner import SafeScaleCommitAction  # noqa: E402


def test_rescue_max_step_pods_extends_the_cap_hpa_style():
    assert rescue_desired(1, 0.01, 0.8, 2.0, 0) == 2
    assert rescue_desired(1, 0.01, 0.8, 2.0, 4) == 5      # max(2n, n + 4)
    assert rescue_desired(8, 0.01, 0.8, 2.0, 4) == 16     # 2n wins for a large n
    assert rescue_desired(1, 0.5, 0.8, 2.0, 4) == 2       # the Z deficit still rules
    plan = _plan([_cls("r", ModelState.CRITICAL, 0.1)], {"r": 1}, idle_gpus=8, step_pods=3)
    assert _deltas(plan) == {"r": 3}
    assert _deltas(_plan([_cls("r", ModelState.CRITICAL, 0.1)], {"r": 1}, idle_gpus=8)) == {"r": 1}
    assert parse_scaling_config({"rescue_max_step_pods": 4}).rescue_max_step_pods == 4
    for bad in (-1, 1.5, True, "x"):
        with pytest.raises(ValueError):
            parse_scaling_config({"rescue_max_step_pods": bad})


class _Preempting:
    """SafeScale stand-in: the first preemption request of a model gives back
    ``restored`` hidden probe pods, later ones nothing (the probe is gone)."""

    def __init__(self, restored: int) -> None:
        self.restored = restored
        self.calls: list[str] = []

    def request_preemption(self, model, *, reason):
        self.calls.append(model)
        return self.restored if len(self.calls) == 1 else 0


def _parts(plan):
    return (
        ScaleAction("r", 1, "critical_sleeping_capacity", "rescue", receiver="r", pods=("r-2",), hint=True, rescue=plan),
        ScaleAction("r", 2, "critical_idle_capacity", "rescue", receiver="r", rescue=plan),
    )


def test_tick_preempts_once_per_receiver_and_deducts_across_parts():
    plan = RescuePlan(target=5, desired=5, base=2, covered=2)
    safescale = _Preempting(restored=2)
    covered: dict = {}
    actions, events = _apply_safescale(_snapshot(5_000), _parts(plan), {}, safescale=safescale,
                                       covered_targets=covered)
    assert safescale.calls == ["r"]  # one preemption for the receiver, not one per part
    assert [(a.delta, a.reason) for a in actions] == [(1, "critical_idle_capacity")]
    assert actions[0].rescue == _replace(plan, covered=4)  # 2 routable + 2 restored
    assert covered == {}
    assert "safescale_probe_preempted:r:restored=2:up_needed=1" in events


def test_tick_records_a_target_fully_covered_by_restored_pods():
    plan = RescuePlan(target=5, desired=5, base=2, covered=2)
    covered: dict = {}
    actions, _ = _apply_safescale(_snapshot(5_000), _parts(plan), {}, safescale=_Preempting(3),
                                  covered_targets=covered)
    assert actions == ()
    assert covered == {"r": _replace(plan, covered=5)}
    queue = ActionQueue(_Client(), now_ms=_Clock(7_000))
    queue.record_rescue_covered("r", covered["r"])
    record = queue.rescue_targets()["r"]
    assert (record.covered, record.outstanding, record.done_ms) == (5, 0, 7_000)


class _FakeTask:
    pass


def _queue_with_backoff_commit(restored: int) -> ActionQueue:
    queue = ActionQueue(_Client(), now_ms=_Clock(9_000))
    task = _FakeTask()
    slot = type("Slot", (), {})()
    slot.queued = type("Q", (), {"action": SafeScaleCommitAction(donor="r", pods=("r-9",), reason="x")})()
    queue._backoff = {task: slot}

    def preempt(task_, slot_, model):
        queue._backoff.pop(task_, None)
        return restored

    queue._preempt_commit = preempt
    return queue


def test_queue_commit_preemption_credit_spans_all_parts():
    plan = RescuePlan(target=5, desired=5, base=2, covered=2)
    queue = _queue_with_backoff_commit(restored=2)
    result = queue.submit(list(_parts(plan)))
    assert result.dropped == (("r", "covered_by_preempted_commit"),)
    assert [(item.action.delta, item.action.reason) for item in queue.pending_actions()] == [
        (1, "critical_idle_capacity")
    ]
    record = queue.rescue_targets()["r"]
    assert (record.gained, record.outstanding, record.done_ms) == (2, 1, None)


def test_queue_records_a_target_covered_entirely_by_a_commit_preemption():
    plan = RescuePlan(target=5, desired=5, base=2, covered=2)
    queue = _queue_with_backoff_commit(restored=3)
    result = queue.submit(list(_parts(plan)))
    assert result.accepted == 0 and len(result.dropped) == 2
    record = queue.rescue_targets()["r"]
    assert (record.covered, record.outstanding, record.done_ms) == (5, 0, 9_000)


def test_superseded_target_parts_never_complete_the_new_target_early():
    queue = ActionQueue(_Client(), now_ms=_Clock(1_000))
    old = RescuePlan(target=3, desired=3, base=2, covered=2)
    new = RescuePlan(target=4, desired=4, base=2, covered=3)
    queue.submit([ScaleAction("r", 1, "critical_idle_capacity", "rescue", receiver="r", rescue=old)])
    queue.submit([ScaleAction("r", 1, "critical_idle_capacity", "rescue", receiver="r", rescue=new)])
    first = queue.pending_actions()[0].action
    queue._finish_rescue(first)  # a part of the superseded target ends
    record = queue.rescue_targets()["r"]
    assert (record.target, record.outstanding, record.done_ms) == (4, 1, None)


def _json_roundtrip(record):
    import json

    return json.loads(json.dumps(record))


class _MemoryStore:
    def __init__(self) -> None:
        self.data: dict = {}

    def save_scale_memory(self, model, record):
        self.data[model] = _json_roundtrip(record)

    def load_scale_memory(self):
        return dict(self.data)


def test_scale_memory_survives_a_controller_restart():
    store = _MemoryStore()
    queue = ActionQueue(_Client(), now_ms=_Clock(65_000), scale_memory=store)
    run_rescue_tick(_snapshot(5_000), queue=queue, registry=_registry())
    asyncio.run(queue.drain_once())
    before = queue.rescue_targets()["critical"]

    restarted = ActionQueue(_Client(), now_ms=_Clock(70_000), scale_memory=store)
    after = restarted.rescue_targets()["critical"]
    assert (after.base, after.target, after.covered, after.done_ms) == (
        before.base, before.target, before.covered, 65_000,
    )
    assert restarted.last_actions() == {"critical": (65_000, "up")}
    # The restarted controller holds on the unreflected window instead of waking again.
    held = run_rescue_tick(_snapshot(20_000), queue=restarted, registry=_registry())
    assert held.submitted == 0
    assert "rescue_target_hold:critical:desired=4:covered=4" in held.events


def test_scale_memory_of_a_target_still_running_at_the_restart_is_done_at_load():
    store = _MemoryStore()
    queue = ActionQueue(_Client(), now_ms=_Clock(65_000), scale_memory=store)
    run_rescue_tick(_snapshot(5_000), queue=queue, registry=_registry())  # submitted, never drained
    restarted = ActionQueue(_Client(), now_ms=_Clock(80_000), scale_memory=store)
    record = restarted.rescue_targets()["critical"]
    assert (record.outstanding, record.done_ms) == (0, 80_000)


def test_scale_memory_errors_never_break_the_queue():
    class _Broken:
        def load_scale_memory(self):
            raise RuntimeError("redis down")

        def save_scale_memory(self, model, record):
            raise RuntimeError("redis down")

    queue = ActionQueue(_Client(), now_ms=_Clock(65_000), scale_memory=_Broken())
    run_rescue_tick(_snapshot(5_000), queue=queue, registry=_registry())
    asyncio.run(queue.drain_once())
    assert queue.rescue_targets()["critical"].covered == 4


def test_controller_state_store_scale_memory_roundtrip():
    from tre_controller.store.state_store import ControllerStateStore

    class _Redis:
        def __init__(self):
            self.h: dict = {}

        def hset(self, name, key=None, value=None, mapping=None):
            self.h.setdefault(name, {}).update(mapping or {key: value})

        def hgetall(self, name):
            return {k.encode(): v.encode() for k, v in self.h.get(name, {}).items()}

    store = ControllerStateStore(_Redis())
    store.save_scale_memory("m", {"last_done": [1, "up"], "rescue": None})
    assert store.load_scale_memory() == {"m": {"last_done": [1, "up"], "rescue": None}}


def _registry_with_tau(tau_ms, **scaling):
    base = _base_registry()
    spec = base.model("critical")
    spec = _replace(spec, trs=_replace(spec.trs, ema_tau_ms=tau_ms))
    return Registry(base.topology(), [spec], scaling=ScalingRegistryConfig(**scaling))


def test_settle_waits_k_ema_time_constants_after_the_window_start():
    queue = ActionQueue(_Client(), now_ms=_Clock(65_000))
    queue.record_rescue_covered("critical", RescuePlan(target=4, desired=4, base=2, covered=4))
    registry = _registry_with_tau(10_000.0)  # default k = 2 -> 20 s
    assert rescue_settle_ms(registry, "critical") == 20_000.0
    assert "critical" in _rescue_bases(_snapshot(65_000), queue, registry)  # the F4 rule alone: settled
    assert "critical" in _rescue_bases(_snapshot(84_999), queue, registry)
    assert "critical" not in _rescue_bases(_snapshot(85_000), queue, registry)
    no_ext = _registry_with_tau(10_000.0, rescue_settle_ema_k=0)
    assert "critical" not in _rescue_bases(_snapshot(65_000), queue, no_ext)
    assert rescue_settle_ms(_registry_with_tau(None), "critical") == 0.0  # legacy fixed alpha
    for bad in (-1, "x", True):
        with pytest.raises(ValueError):
            parse_scaling_config({"rescue_settle_ema_k": bad})


def test_settle_extension_applies_only_while_o1_does_not_track_the_model():
    """Q2 (2026-10-06): while O1 tracks the model (its breakpoint restarts the EMA and the
    evidence gate holds the receiver) a target this process dispatched settles on the
    window-start rule alone; the k * ema_tau extension is the fallback when O1 does not
    track it, and for a target whose done_ms is not an observed completion (covered by a
    probe preemption whose unhide is still to come; restored after a restart)."""
    queue = ActionQueue(_Client(), now_ms=_Clock(65_000))
    plan = RescuePlan(target=4, desired=4, base=2, covered=2)
    queue.submit([ScaleAction("critical", 2, "critical_idle_capacity", "rescue", receiver="critical", rescue=plan)])
    asyncio.run(queue.drain_once())
    registry = _registry_with_tau(10_000.0)  # default k = 2 -> 20 s
    tracked = {"critical": {"o1_routable_tracked": True}}
    untracked = {"critical": {"o1_routable_tracked": False}}
    assert "critical" not in _rescue_bases(_snapshot(65_000), queue, registry, tracked)
    assert "critical" in _rescue_bases(_snapshot(64_999), queue, registry, tracked)  # still before it
    assert "critical" in _rescue_bases(_snapshot(84_999), queue, registry, untracked)
    assert "critical" not in _rescue_bases(_snapshot(85_000), queue, registry, untracked)
    covered = ActionQueue(_Client(), now_ms=_Clock(65_000))
    covered.record_rescue_covered("critical", RescuePlan(target=4, desired=4, base=2, covered=4))
    assert "critical" in _rescue_bases(_snapshot(84_999), covered, registry, tracked)


def test_tp_slot_loop_without_occupancy_counts_one_slot():
    # Defensive (build_plan always has an occupancy with a cluster view): the allocator
    # path does not claim, so a second iteration would count the same slot again.
    import inspect

    from tre_controller.planning import planner

    source = inspect.getsource(planner.build_plan)
    assert "slot_limit = raw_need if occupancy is not None else 1" in source


# ===================================================== review round 2 (2026-10-01)
def _snapshot_routable(window_start_ms: int, routable: int) -> MetricsSnapshot:
    snap = _snapshot(window_start_ms)
    metrics = _replace(snap.models["critical"], routable_pods=routable, assigned_replicas=routable)
    return MetricsSnapshot(ts_ms=snap.ts_ms, stale=False, models={"critical": metrics})


def _store_with_target(issued_ms: int = 65_000) -> _MemoryStore:
    store = _MemoryStore()
    queue = ActionQueue(_Client(), now_ms=_Clock(issued_ms), scale_memory=store)
    run_rescue_tick(_snapshot(5_000), queue=queue, registry=_registry())  # 2 -> 4
    asyncio.run(queue.drain_once())
    return store


def test_restored_target_older_than_the_max_age_is_dropped_at_load():
    store = _store_with_target(issued_ms=65_000)
    late = ActionQueue(_Client(), now_ms=_Clock(65_000 + 50_001), scale_memory=store)
    assert "critical" not in late.rescue_targets()
    assert late.last_actions() == {"critical": (65_000, "up")}  # the F4 memory is kept
    fresh = ActionQueue(_Client(), now_ms=_Clock(65_000 + 50_000), scale_memory=store)
    assert "critical" in fresh.rescue_targets()
    keep_all = ActionQueue(_Client(), now_ms=_Clock(10**9), scale_memory=store, scale_memory_max_age_ms=0)
    assert "critical" in keep_all.rescue_targets()


def test_restored_target_contradicted_by_the_routable_count_is_dropped():
    store = _store_with_target()
    restarted = ActionQueue(_Client(), now_ms=_Clock(70_000), scale_memory=store)
    assert "critical" in restarted.rescue_targets()
    # One of the 2 replicas the target counted before its scale-up is gone: no hold.
    result = run_rescue_tick(_snapshot_routable(20_000, 1), queue=restarted, registry=_registry())
    assert "critical" not in restarted.rescue_targets() or restarted.rescue_targets()["critical"].issued_ms == 70_000
    assert result.submitted == 1
    assert not any(e.startswith("rescue_target_hold") for e in result.events)
    assert store.data["critical"]["rescue"] is None or store.data["critical"]["rescue"]["issued_ms"] == 70_000


def test_restored_target_consistent_with_the_fleet_is_checked_once_and_kept():
    store = _store_with_target()
    restarted = ActionQueue(_Client(), now_ms=_Clock(70_000), scale_memory=store)
    assert restarted.check_restored_targets({"critical": 4}) == []
    assert "critical" in restarted.rescue_targets()
    # Checked once: a later dip is ordinary live behaviour, not a stale restore.
    assert restarted.check_restored_targets({"critical": 1}) == []
    assert "critical" in restarted.rescue_targets()


def test_scale_memory_write_failure_is_logged_and_never_blocks_dispatch(caplog):
    class _Timeout:
        def load_scale_memory(self):
            return {}

        def save_scale_memory(self, model, record):
            raise TimeoutError("Timeout reading from socket")

    client = _Client()
    queue = ActionQueue(client, now_ms=_Clock(65_000), scale_memory=_Timeout())
    with caplog.at_level(logging.WARNING):
        result = run_rescue_tick(_snapshot(5_000), queue=queue, registry=_registry())
        asyncio.run(queue.drain_once())
    assert result.submitted == 1 and client.calls == [("critical", 2)]
    assert "scale memory of critical not saved" in caplog.text


def test_controller_redis_clients_get_socket_timeouts():
    from tre_controller.app import redis_timeouts
    from tre_controller.config import ControllerConfig

    assert redis_timeouts(2.0) == {"socket_timeout": 2.0, "socket_connect_timeout": 2.0}
    assert redis_timeouts(0) == {}
    cfg = ControllerConfig.from_env({})
    assert cfg.redis_socket_timeout_s == 2.0 and cfg.scale_memory_max_age_s == 50.0
    custom = ControllerConfig.from_env(
        {"TRE_REDIS_SOCKET_TIMEOUT_SECONDS": "0.5", "TRE_SCALE_MEMORY_MAX_AGE_SECONDS": "90"}
    )
    assert custom.redis_socket_timeout_s == 0.5 and custom.scale_memory_max_age_s == 90.0


def test_create_redis_client_passes_the_timeouts(monkeypatch):
    import sys
    import types

    from tre_controller import app

    seen = {}

    class _Redis:
        @staticmethod
        def from_url(url, **kwargs):
            seen.update(kwargs, url=url)
            return object()

    monkeypatch.setitem(sys.modules, "redis", types.SimpleNamespace(Redis=_Redis))
    app._create_redis_client("redis://r:6379/0", None, timeout_s=2.0)
    assert seen == {"url": "redis://r:6379/0", "socket_timeout": 2.0, "socket_connect_timeout": 2.0}


def test_metrics_redis_client_has_its_own_longer_timeout(monkeypatch):
    from tre_controller import app
    from tre_controller.config import ControllerConfig

    cfg = ControllerConfig.from_env({})
    assert cfg.redis_metrics_socket_timeout_s == 10.0 and cfg.redis_socket_timeout_s == 2.0
    custom = ControllerConfig.from_env({"TRE_REDIS_METRICS_SOCKET_TIMEOUT_SECONDS": "30"})
    assert custom.redis_metrics_socket_timeout_s == 30.0

    created = []

    def fake_create(url, factory, *, timeout_s=0.0):
        created.append((url, timeout_s))
        return object()

    def stop(*_args, **_kwargs):
        raise RuntimeError("stop after the clients")

    monkeypatch.setattr(app, "_create_redis_client", fake_create)
    monkeypatch.setattr(app, "MetricsStore", stop)
    with pytest.raises(RuntimeError, match="stop after the clients"):
        app.create_controller_dependencies(cfg)
    # Same URL, different timeouts: two clients (state 2 s, metrics 10 s).
    assert created == [(cfg.redis_url, 2.0), (cfg.metrics_redis_url, 10.0)]
    assert cfg.metrics_redis_url == cfg.redis_url

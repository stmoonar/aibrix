"""C1 (2026-10-01): the fast-loop rescue asks for a CRITICAL receiver's whole deficit
at once (an absolute target), the CRITICAL scale-up cooldown is opt-in, and the rescue
target bookkeeping keeps a scale-up the decision window does not reflect yet from
being repeated. The slow loop and every scale-down path are unchanged.
Design: docs/design/20261001-c1-deficit-scaleup.md."""
from __future__ import annotations

import asyncio
import logging

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
          inflight=None, rescue_due=True, fairness_due=True, cluster_view=None, tp=None):
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
        ),
        rescue_bases=bases,
        inflight_models=inflight,
        cluster_view=cluster_view,
    )


def _deltas(plan):
    out: dict[str, int] = {}
    for action in plan.actions:
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

    ups = [a for a in plan.actions if isinstance(a, ScaleAction) and a.delta > 0]
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
    ups = [a for a in plan.actions if isinstance(a, ScaleAction) and a.delta > 0]
    assert [a.delta for a in ups] == [1]
    assert ups[0].rescue == RescuePlan(target=3, desired=4, base=2, covered=2)


def test_immediate_high_donor_gives_its_surplus_not_one_step():
    classifications = [
        _cls("r", ModelState.CRITICAL, 0.4),
        # keep ceil(4 * 1.25 / 2.5) = 2 -> gives 2
        _cls("d", ModelState.HIGH, 2.5, tier="surplus"),
    ]
    plan = _plan(classifications, {"r": 2, "d": 4})
    assert _deltas(plan) == {"r": 2, "d": -2}
    # A barely-HIGH donor (keep ceil(4 * 1.25 / 1.3) = 4) still gives one step.
    barely = _plan([classifications[0], _cls("d", ModelState.HIGH, 1.3, tier="surplus")], {"r": 2, "d": 4})
    assert _deltas(barely) == {"r": 1, "d": -1}


def test_idle_donor_gives_down_to_its_floor():
    classifications = [_cls("r", ModelState.CRITICAL, 0.2), _cls("i", ModelState.IDLE, 10.0, tier="idle")]
    plan = _plan(classifications, {"r": 3, "i": 4})
    assert _deltas(plan) == {"r": 3, "i": -3}  # floor 1; desired 6 needs 3


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
    ups = [a for a in plan.actions if isinstance(a, ScaleAction) and a.delta > 0]
    assert [(a.delta, a.reason) for a in ups] == [(2, "critical_empty_slot")]


# --------------------------------------------------- idempotence / protection
def test_unrefreshed_window_recomputes_the_same_target_and_adds_nothing():
    first = _plan([_cls("r", ModelState.CRITICAL, 0.4)], {"r": 2}, idle_gpus=4)
    target = next(a.rescue for a in first.actions if isinstance(a, ScaleAction))
    basis = {"r": RescueBasis(base=target.base, covered=target.target)}
    # The cluster view may already show the woken replicas (4) or still the old count (2).
    for routable in (2, 4):
        again = _plan([_cls("r", ModelState.CRITICAL, 0.4)], {"r": routable}, idle_gpus=4, bases=basis)
        assert _deltas(again) == {}
        assert "rescue_target_hold:r:desired=4:covered=4" in again.events


def test_load_still_rising_raises_the_target_by_the_difference_only():
    basis = {"r": RescueBasis(base=4, covered=6)}  # 4 -> 6 issued from Z=0.6
    plan = _plan([_cls("r", ModelState.CRITICAL, 0.4)], {"r": 6}, idle_gpus=4, bases=basis)
    ups = [a for a in plan.actions if isinstance(a, ScaleAction) and a.delta > 0]
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
    transfers = [a for a in c1.actions if isinstance(a, ScaleAction) and a.delta > 0]
    assert [(a.model, a.delta, a.reason) for a in transfers] == [("low", 1, "low_fairness_donor_immediate")]


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
    assert _deltas(_plan(classifications, pods)) == {"i": -1}  # one step, as before


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


def test_scale_up_cooldown_switch_holds_a_critical_receiver_again():
    queue = ActionQueue(_Client(), now_ms=_Clock(65_000))
    queue._last_done["critical"] = (65_000, "up")
    on = run_rescue_tick(_snapshot(5_000), queue=queue, registry=_registry(scale_up_cooldown_enabled=True),
                         action_cooldown=True)
    off = run_rescue_tick(_snapshot(5_000), queue=queue, registry=_registry(), action_cooldown=True)
    assert on.submitted == 0 and "cooldown_hold:critical" in on.events
    assert off.submitted == 1 and "cooldown_hold:critical" not in off.events


# ---------------------------------------------------------------- registry
def test_scaling_registry_defaults_and_validation(caplog):
    assert parse_scaling_config(None) == ScalingRegistryConfig(2.0, False)
    assert parse_scaling_config({"rescue_max_step_ratio": 0}).rescue_max_step_ratio == 0.0
    assert parse_scaling_config({"rescue_max_step_ratio": 1}).rescue_max_step_ratio == 1.0
    assert parse_scaling_config({"scale_up_cooldown_enabled": True}).scale_up_cooldown_enabled is True
    for bad in ({"rescue_max_step_ratio": 0.5}, {"rescue_max_step_ratio": -1},
                {"rescue_max_step_ratio": "x"}, {"rescue_max_step_ratio": True},
                {"scale_up_cooldown_enabled": "yes"}, ["not", "a", "mapping"]):
        with pytest.raises(ValueError):
            parse_scaling_config(bad)
    with caplog.at_level(logging.WARNING):
        parse_scaling_config({"rescue_max_step_ratio": 2, "future_key": 1})
    assert "future_key" in caplog.text


def test_shipped_registry_scaling_section_is_the_default():
    registry = load_registry()
    assert registry.scaling() == ScalingRegistryConfig()
    raw = yaml.safe_load(open(__import__("pathlib").Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml",
                              encoding="utf-8"))
    assert set(raw["scaling"]) == {"rescue_max_step_ratio", "scale_up_cooldown_enabled"}

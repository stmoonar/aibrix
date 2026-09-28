"""Post-deploy fixes, controller side (2026-09-28, branch tre/pd2-ctrl).

* P2-3: the SM maintenance lock (``tre:v2:sm:maintenance``) pauses SafeScale -
  no probe starts, open probes roll back, a probe whose window overlapped a
  maintenance period rolls back; a read error blocks probe starts (fail-closed);
* P3: a hide that did not take effect (not sent in observe, or failed) returns
  ok=False and rolls its probe back instead of leaving it judged as hidden;
* planner: several receivers' sleeping wakes are assigned jointly (no greedy
  slot stealing); the same-slot HIGH shrink only looks at awake bindings; a bad
  or missing tp_size raises instead of falling back to tp 1;
* metrics: the model e2e p95 merges the pods' histograms before the
  minimum-samples gate.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from tre_common.metrics_schema import PodWindowMetrics
from tre_common.registry import load_registry
from tre_common.window_pods import restrict_to_serving
from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.safescale_task import run_safescale_observation_tick
from tre_controller.loops.tick import _apply_safescale
from tre_controller.maintenance import MaintenanceWatch
from tre_controller.planning.classify import ModelRole, ModelState
from tre_controller.planning.planner import (
    ClusterView,
    HideAction,
    PlanConfig,
    ScaleAction,
    UnhideAction,
    _SlotOccupancy,
    _try_plan_same_slot_high_shrink,
    build_plan,
)
from tre_controller.store.metrics_store import MetricsStore
from tre_controller.store.state_store import ControllerStateStore
from tre_common.registry import ClusterTopology, NodeSpec
from tre_sm.allocator.slots import Binding, Slot

from test_action_queue_review3 import ScriptedSM, _calls
from test_metrics_store import REGISTRY_PATH, FakeRedis as MetricsRedis, add_doc
from test_planner_slot_occupancy import _cls
from test_safescale_commit import FakeRedis, _machine, _metrics, _registry


# ================================================================ P2-3 maintenance
class MaintenanceRedis:
    def __init__(self, value=None, *, error: Exception | None = None):
        self.value = value
        self.error = error

    def get(self, key):
        assert key == "tre:v2:sm:maintenance"
        if self.error is not None:
            raise self.error
        return self.value


def _lock(operation_id="repair-1", since_ms=11_000):
    return json.dumps(
        {"operation_id": operation_id, "kind": "fleet_repair", "owner": "sm-0", "since_ms": since_ms}
    ).encode()


def _watch(redis, clock):
    return MaintenanceWatch(redis, clock_ms=lambda: clock["t"])


def _probing_machine(start_ms=10_000):
    machine = _machine(ControllerStateStore(FakeRedis()))
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=start_ms, pending_upscales={"receiver": 1})
    return machine


def test_maintenance_watch_states():
    clock = {"t": 12_000}
    redis = MaintenanceRedis()
    watch = _watch(redis, clock)
    assert watch.poll().present is False
    assert watch.probe_block_reason() is None
    redis.value = _lock()
    status = watch.poll()
    assert status.present and status.period.operation_id == "repair-1"
    assert (status.period.since_ms, status.period.last_seen_ms) == (11_000, 12_000)
    assert watch.probe_block_reason() == "sm_maintenance"
    redis.value = b"not json"  # present but malformed: still a lock
    assert watch.probe_block_reason() == "sm_maintenance"
    redis.error = ConnectionError("down")
    assert watch.probe_block_reason() == "sm_maintenance_unreadable"  # fail-closed
    assert watch.poll().error is not None


def test_no_probe_starts_while_maintenance_blocks_it():
    class NoStart:
        def start_probe(self, **_kwargs):
            raise AssertionError("no probe may start during SM maintenance")

    probe = ScaleAction("donor", -1, "high_proactive_safescale", "rescue", requires_safescale=True,
                        donor="donor", pods=("pod-a",))
    other = ScaleAction("receiver", 1, "critical_idle_capacity", "rescue", receiver="receiver")
    for reason in ("sm_maintenance", "sm_maintenance_unreadable"):
        actions, events = _apply_safescale(
            _metrics(1_000), (probe, other), {}, safescale=NoStart(), probe_block_reason=reason
        )
        assert actions == (other,)  # the rest of the plan still goes out
        assert events == (f"safescale_probe_skipped:donor:{reason}",)


def test_open_probe_rolls_back_while_maintenance_is_present():
    machine = _probing_machine()
    clock = {"t": 12_000}
    watch = _watch(MaintenanceRedis(_lock()), clock)
    queue = ActionQueue(ScriptedSM())
    result = run_safescale_observation_tick(
        _metrics(12_000), queue=queue, registry=_registry(), safescale=machine, maintenance=watch
    )
    [unhide] = result.actions
    assert isinstance(unhide, UnhideAction)
    assert (unhide.model, unhide.pods, unhide.reason) == ("donor", ("pod-a",), "sm_maintenance")
    assert "safescale_maintenance_rollback:donor:donor-10000" in result.events
    [probe] = machine.committing_probes()
    assert (probe.resolution, probe.resolution_reason) == ("rollback", "sm_maintenance")
    assert machine.active_probes() == ()


def test_probe_overlapping_a_maintenance_seen_between_safescale_ticks_rolls_back():
    """The repair began and ended between two SafeScale ticks; a planner tick saw it
    (shared watch). The probe's window overlaps it -> rollback, key already gone."""
    machine = _probing_machine(start_ms=10_000)
    clock = {"t": 12_000}
    redis = MaintenanceRedis(_lock(since_ms=11_500))
    watch = _watch(redis, clock)
    assert watch.probe_block_reason() == "sm_maintenance"  # a planner tick sees it
    redis.value = None  # repair finished
    clock["t"] = 13_000
    result = run_safescale_observation_tick(
        _metrics(13_000), queue=ActionQueue(ScriptedSM()), registry=_registry(), safescale=machine,
        maintenance=watch,
    )
    assert [a.reason for a in result.actions] == ["sm_maintenance"]
    assert machine.committing_probes()[0].resolution_reason == "sm_maintenance"


def test_maintenance_before_the_probe_start_does_not_roll_it_back():
    clock = {"t": 5_000}
    redis = MaintenanceRedis(_lock(since_ms=4_000))
    watch = _watch(redis, clock)
    watch.poll()
    redis.value = None
    machine = _probing_machine(start_ms=10_000)
    clock["t"] = 10_500
    result = run_safescale_observation_tick(
        _metrics(10_500), queue=ActionQueue(ScriptedSM()), registry=_registry(), safescale=machine,
        maintenance=watch,
    )
    assert not any("maintenance" in event for event in result.events)
    assert [probe.model for probe in machine.active_probes()] == ["donor"]


def test_unreadable_maintenance_key_does_not_roll_back_open_probes():
    machine = _probing_machine()
    watch = _watch(MaintenanceRedis(error=ConnectionError("down")), {"t": 12_000})
    result = run_safescale_observation_tick(
        _metrics(10_500), queue=ActionQueue(ScriptedSM()), registry=_registry(), safescale=machine,
        maintenance=watch,
    )
    assert "sm_maintenance_unreadable" in result.events
    assert [probe.model for probe in machine.active_probes()] == ["donor"]


def test_the_app_wires_one_maintenance_watch_into_every_loop():
    from test_controller_app import EmptyRedis
    from test_controller_app import REGISTRY_PATH as APP_REGISTRY
    from tre_controller.app import create_controller_dependencies
    from tre_controller.config import ControllerConfig

    cfg = ControllerConfig.from_env({"TRE_REGISTRY_PATH": str(APP_REGISTRY)})
    deps = create_controller_dependencies(cfg, redis_client=EmptyRedis())
    assert isinstance(deps.maintenance_watch, MaintenanceWatch)
    assert deps.queue._on_hide_failed is not None


# ================================================================ P3 hide not applied
def test_hide_skipped_by_the_observe_recheck_is_a_failure_and_rolls_the_probe_back():
    machine = _probing_machine()

    async def scenario():
        reads = iter([False, True])  # active at dispatch entry, observe at the SM call
        sm = ScriptedSM()
        queue = ActionQueue(
            sm, is_observe=lambda: False, is_observe_fresh=lambda: next(reads),
            on_hide_failed=lambda model, pods, reason: machine.abort_probe(model, pods=pods, reason=reason),
        )
        queue.submit((HideAction("donor", ("pod-a",), "probe_started", "fairness"),))
        [result] = await queue.drain_once()
        assert _calls(sm) == []
        assert (result.action_kind, result.ok, result.error) == ("hide", False, "observe_skipped")
        assert queue.stats()["observe_hide_skipped_total"] == 1

    asyncio.run(scenario())
    assert machine.active_probe("donor").abort_reason == "hide_failed: observe_skipped"
    # The next SafeScale tick (back in active mode) rolls it back - even with a stale snapshot.
    stale = _metrics(11_000)
    stale = type(stale)(ts_ms=stale.ts_ms, models=stale.models, stale=True)
    result = run_safescale_observation_tick(
        stale, queue=ActionQueue(ScriptedSM()), registry=_registry(), safescale=machine
    )
    assert [(a.model, a.pods, a.reason) for a in result.actions] == [
        ("donor", ("pod-a",), "hide_failed: observe_skipped")
    ]
    assert "safescale_hide_failed_rollback:donor:donor-10000" in result.events


def test_a_failed_hide_in_active_mode_marks_its_probe_and_observe_rolls_it_back():
    machine = _probing_machine()

    async def scenario():
        sm = ScriptedSM(results={"routable:donor": [{"ok": False, "error": "HTTP 409"}]})
        queue = ActionQueue(
            sm, on_hide_failed=lambda model, pods, reason: machine.abort_probe(model, pods=pods, reason=reason)
        )
        queue.submit((HideAction("donor", ("pod-a",), "probe_started", "rescue"),))
        [result] = await queue.drain_once()
        assert not result.ok

    asyncio.run(scenario())
    decision = machine.observe("donor", _observation(), now_ms=10_500)
    assert (decision.status, decision.reason) == ("rollback", "hide_failed: HTTP 409")


def test_a_hide_dropped_in_observe_mode_is_reported_not_applied():
    marks = []

    async def scenario():
        queue = ActionQueue(ScriptedSM(), is_observe=lambda: True, on_hide_failed=lambda *a: marks.append(a))
        queue.submit((HideAction("donor", ("pod-a",), "probe_started", "fairness"),))
        [result] = await queue.drain_once()
        assert (result.ok, result.error) == (False, "observe_skipped")

    asyncio.run(scenario())
    assert marks == [("donor", ("pod-a",), "hide_failed: observe_skipped")]


def test_abort_only_marks_the_probe_owning_the_pods_and_is_persisted():
    redis = FakeRedis()
    machine = _machine(ControllerStateStore(redis))
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=10_000)
    assert machine.abort_probe("donor", pods=("pod-z",), reason="hide_failed: x") is False
    assert machine.abort_probe("other", pods=("pod-a",), reason="hide_failed: x") is False
    assert machine.abort_probe("donor", pods=("pod-a",), reason="hide_failed: x") is True
    restored = _machine(ControllerStateStore(redis))
    restored.restore()
    assert restored.active_probe("donor").abort_reason == "hide_failed: x"


def _observation():
    from tre_controller.planning.safescale import ProbeObservation

    return ProbeObservation(ts_ms=10_500, ttft_p95_ms=100.0, tpot_p95_ms=10.0, z_m=2.0, has_traffic=True)


# ================================================================ planner
ONE_NODE = ClusterTopology(nodes=(NodeSpec(name="n", gpus=4, two_gpu_slots=((0, 1), (2, 3))),))


def test_joint_wakes_do_not_let_one_receiver_steal_the_only_slot_of_another():
    bindings = (
        Binding("x-2", "x", Slot("n", (2,)), awake=True),
        Binding("x-3", "x", Slot("n", (3,)), awake=True),
        Binding("a-0", "a", Slot("n", (0,)), awake=False),
        Binding("a-1", "a", Slot("n", (1,)), awake=False),
        Binding("b-0", "b", Slot("n", (0,)), awake=False),
    )
    view = ClusterView(ONE_NODE, bindings)
    # Precondition: alone, "a" (planned first) would take gpu 0 - b's only slot.
    assert [b.serve_id for b in _SlotOccupancy(view).plan_wakes("a", 1)] == ["a-0"]
    plan = build_plan(
        model_contexts={
            "a": {"routable_pods": 0, "assigned_replicas": 2},
            "b": {"routable_pods": 0, "assigned_replicas": 1},
            "x": {"routable_pods": 2, "assigned_replicas": 2},
        },
        classifications=[
            _cls("a", ModelState.CRITICAL, ModelRole.RECEIVER, 0.3),
            _cls("b", ModelState.CRITICAL, ModelRole.RECEIVER, 0.4),
            _cls("x", ModelState.HEALTHY, ModelRole.NEUTRAL, 1.1),
        ],
        model_replicas={"a": 2, "b": 1, "x": 2},
        idle_gpus=0,
        cfg=PlanConfig(min_replicas_per_model=0, max_replicas_per_model=8, suppress_hot_proactive_probe=True),
        cluster_view=view,
    )
    wakes = {
        a.model: a.pods for a in plan.actions
        if isinstance(a, ScaleAction) and a.reason == "critical_sleeping_capacity"
    }
    assert wakes == {"a": ("a-1",), "b": ("b-0",)}
    assert "joint_wake_assignment:a=1,b=1" in plan.events
    assert not any(e.startswith("critical_sleeping_blocked") for e in plan.events)


def test_same_slot_high_shrink_ignores_sleeping_bindings():
    bindings = (
        Binding("high-0", "high", Slot("n", (0,)), awake=True),
        Binding("other-1", "other", Slot("n", (1,)), awake=False),  # sleeping: not occupancy
        Binding("high-2", "high", Slot("n", (2,)), awake=False),  # sleeping: never a donor
    )
    shrink = _try_plan_same_slot_high_shrink(
        classifications=[_cls("high", ModelState.HIGH, ModelRole.DONOR, 2.0, "surplus")],
        model_contexts={"high": {"routable_pods": 2, "assigned_replicas": 2}},
        model_replicas={"high": 2},
        cfg=PlanConfig(min_replicas_per_model=0, max_replicas_per_model=4, model_tp_sizes={"tp2": 2, "high": 1}),
        cluster_view=ClusterView(ONE_NODE, bindings),
        receiver="tp2",
        active_probe_models=set(),
        inflight_models=set(),
        source_loop="rescue",
    )
    assert shrink is not None and shrink.serve_id == "high-0"


def test_bad_or_missing_tp_size_raises_instead_of_falling_back():
    with pytest.raises(ValueError, match="power of two"):
        PlanConfig(min_replicas_per_model=0, max_replicas_per_model=4, model_tp_sizes={"m": 3})
    # One rule with the registry / SM (tre_common.registry.tp_size_error): a power
    # of two wider than the binding layout supports is refused too.
    with pytest.raises(ValueError, match="binding layout"):
        PlanConfig(min_replicas_per_model=0, max_replicas_per_model=4, model_tp_sizes={"m": 4})
    with pytest.raises(ValueError, match="no tp_size"):
        build_plan(
            model_contexts={"b": {"routable_pods": 1, "assigned_replicas": 1}},
            classifications=[_cls("b", ModelState.CRITICAL, ModelRole.RECEIVER, 0.4)],
            model_replicas={"b": 1},
            idle_gpus=1,
            cfg=PlanConfig(min_replicas_per_model=0, max_replicas_per_model=4, model_tp_sizes={"x": 1}),
        )


# ================================================================ e2e p95
def _e2e_doc(pod, count, buckets):
    return {
        "pod_name": pod,
        "model_histogram_metrics": {
            "dsqwen-7b/request_prompt_tokens": {"sum": count, "count": count, "buckets": {"1": count}},
            "dsqwen-7b/e2e_request_latency_seconds": {"sum": float(count), "count": count, "buckets": buckets},
        },
    }


def _two_pod_store(min_samples):
    redis = MetricsRedis()
    redis.sadd("tre:v2:pods:dsqwen-7b", "default/pod-a", "default/pod-b")
    for pod, slow in (("pod-a", 0), ("pod-b", 1)):
        key = "tre:v2:hist:default/" + pod
        add_doc(redis, key, 1_000, _e2e_doc(pod, 0, {"1": 0, "5": 0, "+Inf": 0}))
        # 6 requests per pod (< the gate of 10 each, 12 together); pod-b has one slow one.
        add_doc(redis, key, 11_000, _e2e_doc(pod, 6, {"1": 6 - slow, "5": 6, "+Inf": 6}))
    registry = load_registry(str(REGISTRY_PATH))
    return MetricsStore(
        redis, registry, instant_sample_interval_ms=5_000, percentile_mode="bucket_upper",
        min_latency_samples=min_samples,
    )


def test_e2e_p95_merges_pod_histograms_before_the_min_samples_gate():
    metrics = _two_pod_store(10).read_model_window("dsqwen-7b", 1_000, 11_000)
    assert metrics.per_pod["pod-a"].e2e_p95_ms is None  # each pod alone is below the gate
    assert metrics.per_pod["pod-b"].e2e_p95_ms is None
    # merged: 12 observations, 11 <= 1 s, 12 <= 5 s -> p95 (11.4th) in the 5 s bucket
    assert metrics.e2e_p95_ms == 5_000.0
    # The serving restriction re-applies the same rule: one pod left -> 6 < 10 -> None.
    restricted = restrict_to_serving(metrics, sleeping_pods={"pod-b"}, routable_pods=1)
    assert restricted.e2e_p95_ms is None


def test_e2e_p95_without_a_gate_is_the_merged_histogram_p95():
    metrics = _two_pod_store(0).read_model_window("dsqwen-7b", 1_000, 11_000)
    assert metrics.e2e_p95_ms == 5_000.0
    assert metrics.per_pod["pod-a"].e2e_p95_ms == 1_000.0
    restricted = restrict_to_serving(metrics, sleeping_pods={"pod-b"}, routable_pods=1)
    assert restricted.e2e_p95_ms == 1_000.0


def test_pods_without_histograms_keep_the_max_of_pod_p95s():
    from tre_common.window_pods import aggregate_pods

    def pod(name, p95):
        return PodWindowMetrics(name, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, None, None, p95)

    metrics = aggregate_pods("m", 0, 1, {"a": pod("a", 100.0), "b": pod("b", 300.0)}, p95_rule=("bucket_upper", 10))
    assert metrics.e2e_p95_ms == 300.0

"""Sleeping-capacity deadlock (E1 canonical_rerun t1/t2/t7): dsqwen-7b awake on every
GPU, dsllama-8b CRITICAL with sleeping bindings only on those GPUs. The planner used to
emit a critical_sleeping_capacity wake every tick that the SM rejected (WakeConflict)."""
from __future__ import annotations

import asyncio

from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_controller.loops.action_queue import ActionQueue
from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, TauThresholds
from tre_controller.planning.planner import ClusterView, PlanConfig, ScaleAction, build_plan
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.api.v2 import ServiceManagerV2
from tre_sm.state.store import StateStore

from test_safescale_binding_commit import FakeRedis, InProcessServiceManager

NODES = ("node9", "node10")
TOPOLOGY = ClusterTopology(
    nodes=tuple(NodeSpec(name=name, gpus=4, two_gpu_slots=((0, 1), (2, 3))) for name in NODES)
)


def _cls(model: str, state: ModelState, role: ModelRole, z: float, tier: str | None = None) -> ModelClassification:
    return ModelClassification(
        model_name=model,
        state=state,
        role=role,
        Z_m=z,
        eta_m=None,
        trs=0.0,
        theta_m=1.0,
        tau=TauThresholds.from_control(),
        donor_tier=tier,
    )


def _slots():
    return [(node, gpu) for node in NODES for gpu in range(4)]


def _e1_bindings(*, free_slot: tuple[str, int] | None = None) -> tuple[Binding, ...]:
    bindings = []
    for index, (node, gpu) in enumerate(_slots()):
        bindings.append(
            Binding(f"7b-{index}", "dsqwen-7b", Slot(node, (gpu,)), awake=(node, gpu) != free_slot)
        )
        bindings.append(Binding(f"8b-{index}", "dsllama-8b", Slot(node, (gpu,)), awake=False))
    return tuple(bindings)


def _cfg() -> PlanConfig:
    return PlanConfig(min_replicas_per_model=1, max_replicas_per_model=8)


def _plan(bindings: tuple[Binding, ...], donor_state: ModelState, *, idle_gpus: int = 0):
    awake_7b = sum(1 for b in bindings if b.model == "dsqwen-7b" and b.awake)
    donor_role = ModelRole.DONOR if donor_state == ModelState.HIGH else ModelRole.NEUTRAL
    return build_plan(
        model_contexts={
            "dsqwen-7b": {"routable_pods": awake_7b, "assigned_replicas": 8},
            "dsllama-8b": {"routable_pods": 0, "assigned_replicas": 8},
        },
        classifications=[
            _cls("dsllama-8b", ModelState.CRITICAL, ModelRole.RECEIVER, 0.4),
            _cls("dsqwen-7b", donor_state, donor_role, 2.0 if donor_state == ModelState.HIGH else 1.1, "surplus"),
        ],
        model_replicas={"dsqwen-7b": 8, "dsllama-8b": 8},
        idle_gpus=idle_gpus,
        cfg=_cfg(),
        cluster_view=ClusterView(TOPOLOGY, bindings),
    )


def test_e1_deadlock_high_donor_frees_a_slot_where_receiver_sleeps() -> None:
    plan = _plan(_e1_bindings(), ModelState.HIGH)

    scale = [a for a in plan.actions if isinstance(a, ScaleAction)]
    assert not any(a.reason == "critical_sleeping_capacity" for a in scale)
    assert "critical_sleeping_blocked:dsllama-8b" in plan.events
    donor = next(a for a in scale if a.model == "dsqwen-7b")
    receiver = next(a for a in scale if a.model == "dsllama-8b")
    assert (donor.delta, donor.reason, donor.pods, donor.requires_safescale) == (
        -1,
        "critical_donor_immediate",
        ("7b-0",),  # 7b-0 is awake on node9/gpu0 where 8b-0 sleeps
        False,
    )
    assert (receiver.delta, receiver.reason) == (1, "critical_donor_immediate")
    # Serial FIFO queue: the donor sleep is dispatched (and awaited) before the wake.
    assert scale.index(donor) < scale.index(receiver)


def test_e1_deadlock_healthy_donor_probe_is_pinned_to_receiver_slot() -> None:
    plan = _plan(_e1_bindings(), ModelState.HEALTHY)

    scale = [a for a in plan.actions if isinstance(a, ScaleAction)]
    assert not any(a.reason == "critical_sleeping_capacity" for a in scale)
    donor = next(a for a in scale if a.model == "dsqwen-7b")
    assert (donor.reason, donor.requires_safescale, donor.pods) == (
        "critical_middle_zone_safescale",
        True,
        ("7b-0",),
    )
    assert plan.probe_upscale_plans == {"dsqwen-7b": {"dsllama-8b": 1}}


def test_genuinely_free_gpu_still_uses_sleeping_capacity() -> None:
    plan = _plan(_e1_bindings(free_slot=("node10", 2)), ModelState.HIGH, idle_gpus=1)

    scale = [a for a in plan.actions if isinstance(a, ScaleAction)]
    assert [(a.model, a.delta, a.reason) for a in scale] == [
        ("dsllama-8b", 1, "critical_sleeping_capacity")
    ]
    assert not any(event.startswith("critical_sleeping_blocked") for event in plan.events)


def test_one_free_slot_is_not_counted_for_two_receivers() -> None:
    bindings = (
        Binding("a-0", "a", Slot("node9", (0,)), awake=True),
        Binding("b-1", "b", Slot("node9", (1,)), awake=False),
        Binding("c-1", "c", Slot("node9", (1,)), awake=False),
    )
    plan = build_plan(
        model_contexts={
            "a": {"routable_pods": 1, "assigned_replicas": 1},
            "b": {"routable_pods": 0, "assigned_replicas": 1},
            "c": {"routable_pods": 0, "assigned_replicas": 1},
        },
        classifications=[
            _cls("b", ModelState.CRITICAL, ModelRole.RECEIVER, 0.3),
            _cls("c", ModelState.CRITICAL, ModelRole.RECEIVER, 0.4),
            _cls("a", ModelState.HEALTHY, ModelRole.NEUTRAL, 1.1),
        ],
        model_replicas={"a": 1, "b": 1, "c": 1},
        idle_gpus=1,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4),
        cluster_view=ClusterView(TOPOLOGY, bindings),
    )

    wakes = [a for a in plan.actions if isinstance(a, ScaleAction) and a.delta > 0]
    assert [(a.model, a.reason) for a in wakes if a.reason == "critical_sleeping_capacity"] == [
        ("b", "critical_sleeping_capacity")
    ]
    assert "critical_sleeping_blocked:c" in plan.events


def test_low_fairness_sleeping_path_is_slot_aware() -> None:
    plan = build_plan(
        model_contexts={
            "dsqwen-7b": {"routable_pods": 8, "assigned_replicas": 8},
            "dsllama-8b": {"routable_pods": 0, "assigned_replicas": 8},
        },
        classifications=[
            _cls("dsllama-8b", ModelState.LOW, ModelRole.RECEIVER, 0.9),
            _cls("dsqwen-7b", ModelState.HIGH, ModelRole.DONOR, 2.0, "surplus"),
        ],
        model_replicas={"dsqwen-7b": 8, "dsllama-8b": 8},
        idle_gpus=0,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=8, rescue_due=False),
        cluster_view=ClusterView(TOPOLOGY, _e1_bindings()),
    )

    scale = [a for a in plan.actions if isinstance(a, ScaleAction)]
    assert not any(a.reason == "low_fairness_sleeping_capacity" for a in scale)
    assert "low_fairness_sleeping_blocked:dsllama-8b" in plan.events
    donor = next(a for a in scale if a.model == "dsqwen-7b")
    assert (donor.reason, donor.pods) == ("low_fairness_donor_immediate", ("7b-0",))


def _registry() -> Registry:
    trs = TrsParams(
        w_p=0.02, w_d=1.0, lambda_wait=3.0, qmin=1.0, ema_alpha=0.2, theta_m=100.0,
        tau_crit=0.8, tau_low=1.0, tau_high=1.25, qsat=4.0, epsat=0.05, hsat=3,
    )
    slo = SloSpec(ttft_p95_ms=500.0, tpot_p95_ms=75.0, e2e_p95_ms=12000.0)
    return Registry(
        TOPOLOGY,
        [
            ModelSpec(name=name, weights_path="/w", tp_size=1, min_replicas=1, max_replicas=8,
                      vllm_image="image", slo=slo, trs=trs)
            for name in ("dsqwen-7b", "dsllama-8b")
        ],
    )


def test_e1_plan_executes_through_serial_queue_without_wake_conflict() -> None:
    store = StateStore(FakeRedis())
    store.save(list(_e1_bindings()), expected_version=0)
    service = ServiceManagerV2(_registry(), store)
    queue = ActionQueue(InProcessServiceManager(service))

    plan = _plan(_e1_bindings(), ModelState.HIGH)
    queue.submit(plan.actions)
    results = asyncio.run(queue.drain_once())

    assert all(result.ok for result in results), results
    bindings = {b.serve_id: b for b in store.load().bindings}
    assert bindings["7b-0"].awake is False
    assert bindings["8b-0"].awake is True
    assert sum(b.awake for b in bindings.values() if b.model == "dsqwen-7b") == 7

"""End-to-end regression tests (review round 2, P2-5): tick (-> _idle_gpus(cluster_view))
-> planner -> ActionQueue -> real ServiceManagerV2 service layer.

Each scenario builds the SM state, derives the controller cluster_view from GET-state,
runs one rescue/fairness tick and drains the queue into the SM. A WakeConflict or "no
free slot" raised by the SM fails the test (the E1 deadlock signature)."""
from __future__ import annotations

import asyncio

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.cluster_view_task import cluster_view_from_state
from tre_controller.loops.fairness_task import run_fairness_tick
from tre_controller.loops.rescue_task import run_rescue_tick
from tre_controller.loops.tick import _idle_gpus
from tre_controller.planning.classify import ModelState
from tre_controller.planning.planner import ScaleAction, TransferIntent
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.api.v2 import ServiceManagerV2
from tre_sm.state.store import StateStore

from test_safescale_binding_commit import FakeRedis, InProcessServiceManager

NODES = ("node9", "node10")
TOPOLOGY = ClusterTopology(
    nodes=tuple(NodeSpec(name=name, gpus=4, two_gpu_slots=((0, 1), (2, 3))) for name in NODES)
)
SLOTS = [(node, gpu) for node in NODES for gpu in range(4)]
THETA = 100.0  # Z_m = generation_tokens / Q / theta with these params
CRITICAL, LOW, HIGH, HEALTHY = 50.0, 90.0, 200.0, 110.0  # generation tokens per unit Q


def _registry(*specs: tuple[str, int, int, int]) -> Registry:
    trs = TrsParams(
        w_p=0.02, w_d=1.0, lambda_wait=3.0, qmin=1.0, ema_alpha=0.0, theta_m=THETA,
        tau_crit=0.8, tau_low=1.0, tau_high=1.25, qsat=4.0, epsat=0.05, hsat=3,
    )
    slo = SloSpec(ttft_p95_ms=500.0, tpot_p95_ms=75.0, e2e_p95_ms=12000.0)
    return Registry(
        TOPOLOGY,
        [
            ModelSpec(name=name, weights_path="/w", tp_size=tp, min_replicas=lo, max_replicas=hi,
                      vllm_image="image", slo=slo, trs=trs)
            for name, tp, lo, hi in specs
        ],
    )


def _window(model: str, *, per_q: float, running: float) -> ModelWindowMetrics:
    return ModelWindowMetrics(
        model=model, window_start_ms=0, window_end_ms=60_000, prompt_tokens=0.0,
        generation_tokens=per_q * running, avg_waiting=0.0, avg_running=running, avg_swapping=0.0,
        kv_cache_hit_rate=0.0, ttft_p95_ms=100.0, tpot_p95_ms=10.0, e2e_p95_ms=1000.0,
        routable_pods=0, assigned_replicas=0, per_pod={},
    )


def _run(registry: Registry, bindings: list[Binding], load: dict[str, tuple[float, float]], *, loop: str = "rescue"):
    store = StateStore(FakeRedis())
    store.save(bindings, expected_version=0)
    service = ServiceManagerV2(registry, store)
    view = cluster_view_from_state(service.get_state(), registry.topology())
    snapshot = MetricsSnapshot(
        ts_ms=60_000,
        stale=False,
        models={model: _window(model, per_q=per_q, running=running) for model, (per_q, running) in load.items()},
    )
    sm = InProcessServiceManager(service)
    queue = ActionQueue(sm)
    tick = run_rescue_tick if loop == "rescue" else run_fairness_tick
    # No safescale controller here, so the receiver-less HIGH proactive probe (default on
    # since A2) would dispatch as an immediate sleep; these cases exercise slot planning.
    result = tick(snapshot, queue=queue, registry=registry, cluster_view=view, suppress_hot_proactive_probe=True)
    dispatched = asyncio.run(queue.drain_once())  # raises on WakeConflict / no free slot
    assert all(item.ok for item in dispatched), dispatched
    awake = {b.serve_id for b in store.load().bindings if b.awake}
    return result, sm.calls, awake, _idle_gpus(snapshot, registry, view)


def _scale(result) -> list[tuple[str, int, str, tuple[str, ...]]]:
    return [(a.model, a.delta, a.reason, a.pods) for a in result.actions if isinstance(a, ScaleAction)]


def _relays(result) -> list[tuple[str, str, int, str]]:
    """2026-10-02: immediate relays are TransferIntents (counts); the SM places them."""
    return [(a.donor_model, a.receiver_model, a.count, a.reason) for a in result.actions if isinstance(a, TransferIntent)]


def _relay_moved(before: list[Binding], awake: set[str], donor: str, receiver: str) -> list[tuple[str, str]]:
    """(donor pod slept, receiver pod woken) on the same slot - what the SM's relay did."""
    slots = {b.serve_id: (b.slot.node, b.slot.gpu_ids) for b in before}
    slept = [b.serve_id for b in before if b.model == donor and b.awake and b.serve_id not in awake]
    woke = [b.serve_id for b in before if b.model == receiver and not b.awake and b.serve_id in awake]
    return [(d, r) for d in slept for r in woke if slots[d] == slots[r]]


REG_7B_8B = (("dsqwen-7b", 1, 1, 8), ("dsllama-8b", 1, 1, 8))
#: An idle window (no token, nothing running): the donor is IDLE - the only donor still
#: released at once (relay) since F1-B (2026-10-07); a HIGH donor is probed.
IDLE_LOAD = (0.0, 0.0)


def test_e1_receiver_gets_the_slot_the_idle_donor_frees() -> None:
    bindings = [Binding("8b-0", "dsllama-8b", Slot("node9", (0,)), awake=True)]
    for index, (node, gpu) in enumerate(SLOTS[1:], start=1):
        bindings.append(Binding(f"7b-{index}", "dsqwen-7b", Slot(node, (gpu,)), awake=True))
        bindings.append(Binding(f"8b-{index}", "dsllama-8b", Slot(node, (gpu,)), awake=False))

    result, calls, awake, idle = _run(
        _registry(*REG_7B_8B), bindings, {"dsllama-8b": (CRITICAL, 1.0), "dsqwen-7b": IDLE_LOAD}
    )

    assert idle == 0
    assert result.classifications["dsllama-8b"].state == ModelState.CRITICAL
    assert result.classifications["dsqwen-7b"].state == ModelState.IDLE
    # 2026-10-02: one relay intent (a count); which 7b / 8b pair moves is the SM's
    # choice (its placement policy), here the fake SM's pair rule.
    assert _scale(result) == []
    assert _relays(result) == [("dsqwen-7b", "dsllama-8b", 1, "critical_donor_immediate")]
    assert calls[0] == ("transfer", "dsqwen-7b", "dsllama-8b", 1)
    assert "8b-0" in awake and len(_relay_moved(bindings, awake, "dsqwen-7b", "dsllama-8b")) == 1


def test_repro_a_gpu_with_only_a_foreign_sleeping_binding_is_not_receiver_idle_capacity() -> None:
    # node10/3 has no awake binding (only a sleeping 7b); 8b holds no binding there and
    # still has blocked sleeping bindings elsewhere, so the SM would WakeConflict on a
    # model-level +1. Before the fix: [('dsllama-8b', +1, 'critical_idle_capacity')].
    bindings = []
    for index, (node, gpu) in enumerate(SLOTS):
        if (node, gpu) == ("node10", 3):
            bindings.append(Binding(f"7b-{index}", "dsqwen-7b", Slot(node, (gpu,)), awake=False))
            continue
        bindings.append(Binding(f"7b-{index}", "dsqwen-7b", Slot(node, (gpu,)), awake=True))
        bindings.append(Binding(f"8b-{index}", "dsllama-8b", Slot(node, (gpu,)), awake=False))

    result, calls, awake, idle = _run(
        _registry(*REG_7B_8B), bindings, {"dsllama-8b": (CRITICAL, 1.0), "dsqwen-7b": IDLE_LOAD}
    )

    assert idle == 1  # the raw free-GPU count is still reported ...
    assert "critical_idle_unusable:dsllama-8b" in result.events  # ... but not usable by 8b
    assert _scale(result) == []
    assert _relays(result) == [("dsqwen-7b", "dsllama-8b", 1, "critical_donor_immediate")]
    assert len(_relay_moved(bindings, awake, "dsqwen-7b", "dsllama-8b")) == 1


def test_repro_a_fully_empty_gpu_is_not_receiver_idle_capacity_while_it_has_blocked_sleepers() -> None:
    bindings = []
    for index, (node, gpu) in enumerate(SLOTS):
        if (node, gpu) == ("node10", 3):
            continue  # a GPU with no binding at all
        bindings.append(Binding(f"7b-{index}", "dsqwen-7b", Slot(node, (gpu,)), awake=True))
        bindings.append(Binding(f"8b-{index}", "dsllama-8b", Slot(node, (gpu,)), awake=False))

    result, _, awake, idle = _run(
        _registry(*REG_7B_8B), bindings, {"dsllama-8b": (CRITICAL, 1.0), "dsqwen-7b": IDLE_LOAD}
    )

    assert idle == 1
    assert not any(reason == "critical_idle_capacity" for _, _, reason, _ in _scale(result))
    assert _relays(result) == [("dsqwen-7b", "dsllama-8b", 1, "critical_donor_immediate")]
    assert len(_relay_moved(bindings, awake, "dsqwen-7b", "dsllama-8b")) == 1


def test_empty_gpu_is_used_by_create_when_receiver_has_no_sleeping_binding() -> None:
    bindings = [Binding("8b-0", "dsllama-8b", Slot("node9", (0,)), awake=True)]
    bindings += [
        Binding(f"7b-{index}", "dsqwen-7b", Slot(node, (gpu,)), awake=True)
        for index, (node, gpu) in enumerate(SLOTS)
        if 0 < index < 7
    ]  # node10/3 completely empty

    result, calls, awake, _ = _run(
        _registry(*REG_7B_8B), bindings, {"dsllama-8b": (CRITICAL, 1.0), "dsqwen-7b": (HIGH, 6.0)}
    )

    assert _scale(result) == [("dsllama-8b", 1, "critical_idle_capacity", ())]
    assert calls == [("scale_model", "dsllama-8b", 1)]
    assert sum(1 for pod in awake if pod.startswith("dsllama")) == 1  # SM created one


def test_repro_b_donor_without_slot_match_is_skipped_not_drained() -> None:
    # R sleeps only under A (HEALTHY, at min_replicas); the HIGH donor D holds no GPU R
    # has a binding on. Before the fix D slept a tail pod and R WakeConflicted.
    registry = _registry(("A", 1, 1, 8), ("D", 1, 1, 8), ("R", 1, 1, 8))
    bindings = [
        Binding("A-0", "A", Slot("node9", (0,)), awake=True),
        Binding("R-0", "R", Slot("node9", (0,)), awake=False),
    ]
    bindings += [Binding(f"D-{i}", "D", Slot(n, (g,)), awake=True) for i, (n, g) in enumerate(SLOTS) if i > 0]

    result, calls, awake, _ = _run(
        registry, bindings, {"R": (CRITICAL, 1.0), "A": (HEALTHY, 1.0), "D": (HIGH, 7.0)}
    )

    assert "donor_no_slot_match:D:R" in result.events
    assert _scale(result) == []
    assert calls == []
    assert awake == {"A-0"} | {f"D-{i}" for i in range(1, 8)}


REG_TP2 = (("dsqwen-7b", 1, 1, 8), ("dsqwen-14b", 2, 0, 4))


def test_tp2_receiver_does_not_count_two_unpaired_free_gpus_as_capacity() -> None:
    free = {("node9", 2), ("node10", 0)}  # two free GPUs, but in different two_gpu_slots
    bindings = [Binding("14b-0", "dsqwen-14b", Slot("node9", (0, 1)), awake=True)]
    bindings += [
        Binding(f"7b-{i}", "dsqwen-7b", Slot(n, (g,)), awake=True)
        for i, (n, g) in enumerate(SLOTS)
        if (n, g) not in free and (n, g) not in {("node9", 0), ("node9", 1)}
    ]

    result, calls, _, idle = _run(
        _registry(*REG_TP2), bindings, {"dsqwen-14b": (LOW, 1.0), "dsqwen-7b": (HEALTHY, 4.0)}, loop="fairness"
    )

    assert idle == 2
    assert result.classifications["dsqwen-14b"].state == ModelState.LOW
    assert not any(model == "dsqwen-14b" and delta > 0 for model, delta, _, _ in _scale(result))
    assert calls == []


def test_tp2_receiver_uses_a_complete_free_slot_pair() -> None:
    bindings = [Binding("14b-0", "dsqwen-14b", Slot("node9", (0, 1)), awake=True)]
    bindings += [
        Binding(f"7b-{i}", "dsqwen-7b", Slot(n, (g,)), awake=True)
        for i, (n, g) in enumerate(SLOTS)
        if n == "node9" and g >= 2 or n == "node10" and g < 2
    ]  # node10 (2,3) free

    result, calls, awake, _ = _run(
        _registry(*REG_TP2), bindings, {"dsqwen-14b": (LOW, 1.0), "dsqwen-7b": (HEALTHY, 4.0)}, loop="fairness"
    )

    assert _scale(result) == [("dsqwen-14b", 1, "low_fairness_idle_capacity", ())]
    assert calls == [("scale_model", "dsqwen-14b", 1)]
    assert sum(1 for pod in awake if "14b" in pod) == 2


def test_tp2_critical_receiver_with_blocked_sleeping_binding_does_not_request_empty_slot() -> None:
    bindings = [
        Binding("14b-0", "dsqwen-14b", Slot("node9", (0, 1)), awake=True),
        Binding("14b-1", "dsqwen-14b", Slot("node9", (2, 3)), awake=False),
        Binding("7b-2", "dsqwen-7b", Slot("node9", (2,)), awake=True),
        Binding("7b-3", "dsqwen-7b", Slot("node9", (3,)), awake=True),
    ]  # node10 fully empty, but the SM would first try 14b-1 and WakeConflict

    result, calls, _, _ = _run(
        _registry(*REG_TP2), bindings, {"dsqwen-14b": (CRITICAL, 1.0), "dsqwen-7b": (HEALTHY, 2.0)}
    )

    assert "capacity_blocked:dsqwen-14b" in result.events
    assert not any(reason == "critical_empty_slot" for _, _, reason, _ in _scale(result))
    assert all(call[0] != "scale_model" for call in calls)


def test_planned_sleeping_wakes_land_on_the_slot_the_planner_claimed() -> None:
    # P2-3: R1 could wake r-a (node9/1) or r-b (node10/1); the planner picks r-a (R1
    # already has a replica on node10, so the placement policy spreads it to node9) and
    # leaves r-b's GPU to R2, whose only slot it is. Binding-level wakes make the SM
    # wake exactly the planned slots instead of re-picking on its own.
    registry = _registry(("R1", 1, 1, 8), ("R2", 1, 1, 8), ("F", 1, 1, 8))
    bindings = [
        Binding("r-0", "R1", Slot("node10", (0,)), awake=True),
        Binding("r-a", "R1", Slot("node9", (1,)), awake=False),
        Binding("r-b", "R1", Slot("node10", (1,)), awake=False),
        Binding("s-0", "R2", Slot("node9", (2,)), awake=True),
        Binding("s-1", "R2", Slot("node10", (1,)), awake=False),
        Binding("f-3", "F", Slot("node9", (3,)), awake=True),
        Binding("f-5", "F", Slot("node9", (0,)), awake=True),
        Binding("f-6", "F", Slot("node10", (2,)), awake=True),
        Binding("f-7", "F", Slot("node10", (3,)), awake=True),
    ]

    result, calls, awake, _ = _run(
        registry, bindings, {"R1": (CRITICAL, 1.0), "R2": (CRITICAL * 0.8, 1.0), "F": (HEALTHY, 4.0)}
    )

    wakes = {(model, pods) for model, delta, reason, pods in _scale(result) if reason == "critical_sleeping_capacity"}
    assert wakes == {("R1", ("r-a",)), ("R2", ("s-1",))}
    assert ("set_binding_power", "r-a", True) in calls and ("set_binding_power", "s-1", True) in calls
    assert {"r-a", "s-1"} <= awake and "r-b" not in awake

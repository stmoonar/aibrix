"""Donor evidence fixes (2026-10-07, docs/design/donor-evidence-20261007.md).

1. F1-B: a HIGH donor never gives a replica without a SafeScale probe (rescue and
   fairness); an IDLE donor is still released at once. A HIGH donor whose last probe
   rolled back waits for new evidence.
2. F2: an SM call of the controller that changed the fleet refreshes the cluster view
   at once (event-driven; the period is the fallback).
3. A slot an in-flight probe counts on for its receiver (the free mate of a TP
   same-slot shrink) is no capacity for any other wake of the plan.
4. F4: while another model is CRITICAL, an early commit does not wait for the hidden
   pods to drain (a cost: their requests are re-issued); every evidence condition
   (samples, one p95 e2e, post-hide grids, gates) stays.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from tre_common.metrics_schema import MetricsSnapshot
from tre_common.registry import ClusterTopology, NodeSpec
from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.cluster_view_task import ClusterViewBox, cluster_view_task
from tre_controller.loops.tick import run_planner_tick
from tre_controller.planning.classify import ModelRole, ModelState
from tre_controller.planning.planner import (
    ClusterView,
    PlanConfig,
    SafeScaleCommitAction,
    ScaleAction,
    ShrinkForSlotAction,
    TransferIntent,
    build_plan,
    probe_reserved_gpus,
)
from tre_controller.planning.safescale import ProbeWindowInputs, SafeScaleDecision
from tre_sm.allocator.slots import Binding, Slot

from test_controller_transfer_20261002 import D, R, TOPOLOGY, _e1
from test_drain_policy_20260929 import RecordingClient
from test_planner_slot_occupancy import _cls
from test_scaling_e2e import HEALTHY, LOW, _registry, _window
from test_timer_cleanup_early_commit_20261002 import LONG_W, EarlyHarness
from test_safescale_direct_20260929 import HIDE, MODEL, START, _obs


# ============================================================== 1. F1-B
def _donor_plan(donor_state, receiver_state, *, backoff=None):
    rescue = receiver_state == ModelState.CRITICAL
    return build_plan(
        model_contexts={
            D: {"routable_pods": 2, "assigned_replicas": 8},
            R: {"routable_pods": 1, "assigned_replicas": 8},
        },
        classifications=[
            _cls(R, receiver_state, ModelRole.RECEIVER, 0.4 if rescue else 0.9),
            _cls(D, donor_state, ModelRole.DONOR, 10.0 if donor_state == ModelState.IDLE else 2.3,
                 "idle" if donor_state == ModelState.IDLE else "surplus"),
        ],
        model_replicas={D: 8, R: 8},
        idle_gpus=0,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=8, suppress_hot_proactive_probe=True,
                       rescue_due=rescue, fairness_due=not rescue),
        # D awake on two GPUs (n = 2, the diagnosed case), R sleeping under both.
        cluster_view=ClusterView(TOPOLOGY, tuple(b for b in _e1() if b.serve_id in ("7b-0", "7b-1", "8b-0", "8b-1"))),
        probe_backoff_models=backoff,
    )


@pytest.mark.parametrize("receiver_state", [ModelState.CRITICAL, ModelState.LOW], ids=["rescue", "fairness"])
def test_a_high_donor_gives_nothing_without_a_probe(receiver_state):
    plan = _donor_plan(ModelState.HIGH, receiver_state)
    assert not [a for a in plan.actions if isinstance(a, TransferIntent)]
    downs = [a for a in plan.actions if isinstance(a, ScaleAction) and a.model == D and a.delta < 0]
    assert downs and all(a.requires_safescale and a.sleep_path is None for a in downs)
    assert plan.probe_upscale_plans.get(D, {}).get(R) == 1


@pytest.mark.parametrize("receiver_state", [ModelState.CRITICAL, ModelState.LOW], ids=["rescue", "fairness"])
def test_an_idle_donor_is_still_released_at_once(receiver_state):
    plan = _donor_plan(ModelState.IDLE, receiver_state)
    [relay] = [a for a in plan.actions if isinstance(a, TransferIntent)]
    assert (relay.donor_model, relay.receiver_model, relay.sleep_path) == (D, R, "urgent")


@pytest.mark.parametrize("donor_state", [ModelState.HIGH, ModelState.HEALTHY], ids=["high", "middle_zone"])
@pytest.mark.parametrize("receiver_state", [ModelState.CRITICAL, ModelState.LOW], ids=["rescue", "fairness"])
def test_a_donor_whose_probe_rolled_back_waits_for_new_evidence(receiver_state, donor_state):
    plan = _donor_plan(donor_state, receiver_state, backoff={D: "same_evidence"})
    assert not [a for a in plan.actions if getattr(a, "model", None) == D or getattr(a, "donor_model", None) == D]
    assert f"safescale_rollback_hold:{D}:same_evidence" in plan.events


# ============================================================== 3. probe slot
TOPO_A = ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),))


def _slot_plan(view: ClusterView, *, crit_tp2: bool, active=(), backoff=None, proactive=False):
    classifications = [
        _cls("low", ModelState.LOW, ModelRole.RECEIVER, 0.9),
        _cls("high", ModelState.HIGH, ModelRole.DONOR, 1.4, "surplus"),
    ]
    if crit_tp2:
        classifications.insert(0, _cls("tp2", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5))
    return build_plan(
        model_contexts={
            "tp2": {"assigned_replicas": 0, "routable_pods": 0},
            "low": {"assigned_replicas": 2, "routable_pods": 1},
            "high": {"assigned_replicas": 2, "routable_pods": 2},
        },
        classifications=classifications,
        model_replicas={"tp2": 0, "low": 2, "high": 2},
        idle_gpus=0,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4, suppress_hot_proactive_probe=not proactive,
                       model_tp_sizes={"tp2": 2, "low": 1, "high": 1}),
        cluster_view=view,
        active_probe_models=set(active),
        probe_backoff_models=backoff,
    )


def _slot_bindings(*, hidden: bool) -> tuple[Binding, ...]:
    # high-0 holds node-a/0; its pair mate node-a/1 is free - only low sleeps there.
    return (
        Binding("high-0", "high", Slot("node-a", (0,)), awake=True, hidden=hidden),
        Binding("high-2", "high", Slot("node-a", (2,)), awake=True),
        Binding("low-3", "low", Slot("node-a", (3,)), awake=True),
        Binding("low-s", "low", Slot("node-a", (1,)), awake=False),
    )


def _woken(plan) -> set[str]:
    return {pod for a in plan.actions if isinstance(a, ScaleAction) and a.delta > 0 for pod in a.pods}


def test_the_slot_mate_of_a_same_slot_shrink_is_not_woken_into_in_the_same_plan():
    view = ClusterView(TOPO_A, _slot_bindings(hidden=False))
    plan = _slot_plan(view, crit_tp2=True)
    [shrink] = [a for a in plan.actions if isinstance(a, ShrinkForSlotAction)]
    assert (shrink.serve_id, shrink.beneficiary) == ("high-0", "tp2")
    assert "low-s" not in _woken(plan)
    # control: without the TP receiver, LOW does wake into the free GPU
    assert "low-s" in _woken(_slot_plan(view, crit_tp2=False))


def test_a_same_slot_shrink_is_not_retried_on_the_evidence_that_rolled_it_back():
    view = ClusterView(TOPO_A, _slot_bindings(hidden=False))
    plan = _slot_plan(view, crit_tp2=True, backoff={"high": "same_evidence"})
    assert not [a for a in plan.actions if isinstance(a, ShrinkForSlotAction)]
    assert not [a for a in plan.actions if getattr(a, "model", None) == "high"]
    assert "safescale_rollback_hold:high:same_evidence" in plan.events


def test_a_probe_promises_only_its_planned_receivers():
    """The fairness piggyback is gone (2026-10-07): it handed a receiver-less probe's
    freed donor replicas to LOW receivers as receiver replicas (wrong for a TP2
    receiver of TP1 donors) without slot geometry. A receiver-less HIGH probe promises
    nothing; the freed GPU is planned from free capacity once the view shows it (F2)."""
    view = ClusterView(TOPO_A, (
        Binding("high-0", "high", Slot("node-a", (0,)), awake=True),
        Binding("high-1", "high", Slot("node-a", (1,)), awake=True),
        Binding("high-2", "high", Slot("node-a", (2,)), awake=True),
        Binding("low-3", "low", Slot("node-a", (3,)), awake=True),
    ))
    plan = _slot_plan(view, crit_tp2=False, proactive=True)
    probes = [a for a in plan.actions if isinstance(a, ScaleAction) and a.model == "high" and a.delta < 0]
    assert [a.reason for a in probes] == ["high_proactive_safescale"]
    assert not plan.probe_upscale_plans.get("high")


@dataclass
class _Probe:
    model: str
    pods: tuple[str, ...]
    pending_upscales: dict = field(default_factory=dict)


def test_the_slot_an_inflight_probe_counts_on_is_reserved_until_it_resolves():
    view = ClusterView(TOPO_A, _slot_bindings(hidden=True))
    tp = {"tp2": 2, "low": 1, "high": 1}
    probe = _Probe("high", ("high-0",), {"tp2": 1})
    reserved = probe_reserved_gpus(view, [probe], tp)
    assert reserved == {("node-a", 0), ("node-a", 1)}
    blocked = ClusterView(TOPO_A, view.bindings, blocked_gpus=reserved)
    assert "low-s" not in _woken(_slot_plan(blocked, crit_tp2=False, active={"high"}))
    # resolved (no probe) / a receiver-less probe: nothing reserved, LOW wakes there
    assert probe_reserved_gpus(view, [], tp) == frozenset()
    assert probe_reserved_gpus(view, [_Probe("high", ("high-0",))], tp) == frozenset()
    assert "low-s" in _woken(_slot_plan(view, crit_tp2=False, active={"high"}))


class _ProbingSafeScale:
    """The tick's view of the SafeScale state machine: one unresolved probe."""

    def __init__(self, probes) -> None:
        self._probes = tuple(probes)

    def all_probes(self):
        return self._probes

    def start_probe(self, **_kwargs):
        return SafeScaleDecision(status="none", reason="not_in_this_test")


def test_the_tick_reserves_the_slot_of_an_inflight_probe():
    registry = _registry(("tp2", 2, 0, 4), ("high", 1, 1, 8), ("low", 1, 1, 8))
    bindings = [
        Binding("high-0", "high", Slot("node9", (0,)), awake=True, hidden=True),
        Binding("low-s", "low", Slot("node9", (1,)), awake=False),
        Binding("high-2", "high", Slot("node9", (2,)), awake=True),
        Binding("low-3", "low", Slot("node9", (3,)), awake=True),
    ] + [Binding(f"high-n{g}", "high", Slot("node10", (g,)), awake=True) for g in range(4)]
    view = ClusterView(registry.topology(), tuple(bindings))
    snapshot = MetricsSnapshot(ts_ms=60_000, stale=False, models={
        "low": _window("low", per_q=LOW, running=1.0), "high": _window("high", per_q=HEALTHY, running=5.0),
    })

    def tick(probes):
        class _Queue:
            submitted: list = []

            def inflight_models(self):
                return set()

            def submit(self, actions):
                self.submitted.append(tuple(actions))

        return run_planner_tick(snapshot, queue=_Queue(), registry=registry, rescue_due=True, fairness_due=True,
                                cluster_view=view, active_probe_models={"high"},
                                safescale=_ProbingSafeScale(probes))

    reserved = tick([_Probe("high", ("high-0",), {"tp2": 1})])
    assert "low-s" not in _woken(reserved)
    assert "probe_slot_reserved:node9/0,node9/1" in reserved.events
    assert "low-s" in _woken(tick([]))  # control: resolved -> free again


# ============================================================== 2. F2
def test_a_committed_scale_down_asks_for_a_view_refresh():
    asked: list[int] = []
    queue = ActionQueue(RecordingClient(), on_fleet_change=lambda: asked.append(1))
    queue.submit((SafeScaleCommitAction(donor="m", pods=("pod-a",), reason="formal_commit_gate_passed"),))
    asyncio.run(queue.drain_once())
    assert asked


def test_a_refresh_request_refreshes_the_view_without_waiting_for_the_period():
    async def scenario():
        calls: list[int] = []
        fetched = asyncio.Event()

        class _Client:
            async def get_state(self):
                calls.append(1)
                fetched.set()
                return {"bindings": []}

        never = asyncio.Event()

        async def period(_seconds):  # the period never ends in this test
            await never.wait()

        box = ClusterViewBox()
        task = asyncio.create_task(cluster_view_task(
            _Client(), TOPO_A, box, type("Cfg", (), {"fairness_interval_s": 10.0})(), sleep=period))
        try:
            await asyncio.wait_for(fetched.wait(), 1.0)
            fetched.clear()
            for _ in range(10):
                await asyncio.sleep(0)
            assert len(calls) == 1  # no request, period not over: no refresh
            box.request_refresh()
            await asyncio.wait_for(fetched.wait(), 1.0)
            assert len(calls) == 2
        finally:
            task.cancel()

    asyncio.run(scenario())


# ============================================================== 4. F4
class _CriticalHarness(EarlyHarness):
    """The early-commit harness, driven like the SafeScale loop drives the state
    machine: each observation passes the planner's CRITICAL models, and the probe
    starts with the donor's window inputs (its p95 e2e)."""

    def __init__(self, *, critical=(), p95_e2e_ms=None, **cfg) -> None:
        self.critical = tuple(critical)
        self.p95_e2e_ms = p95_e2e_ms
        super().__init__(**cfg)

    def start(self, **_kwargs):
        inputs = ProbeWindowInputs(p95_e2e_ms=self.p95_e2e_ms) if self.p95_e2e_ms else None
        self.machine.start_probe(model=MODEL, pods=("m-1",), now_ms=START, window_inputs=inputs)
        self.clock.now = HIDE
        assert self.collector.on_hide_done(MODEL, ("m-1",))
        asyncio.run(self.collector.take_baselines())
        return self.machine.active_probe(MODEL)

    def tick(self, at_ms, *, serve=0, ttft_s=0.05, obs=None):
        self.clock.now = at_ms
        for sim in self.sims.values():
            sim.serve(serve, ttft_s=ttft_s)
        polls = asyncio.run(self.collector.poll())
        boundary = at_ms // 10_000 * 10_000
        return self.machine.observe(MODEL, obs or _obs(boundary, **self.obs_kwargs), now_ms=boundary,
                                    direct_poll=polls.get(MODEL), critical_receivers=self.critical)


def _busy(h: _CriticalHarness) -> _CriticalHarness:
    h.start()
    h.hidden.running = 1.0  # the hidden pod still serves a request
    return h


def test_a_critical_model_waives_the_hidden_drain_only():
    at, decision = _busy(_CriticalHarness(critical=("r",))).run_until(HIDE + LONG_W, serve=5)[-1]
    # The evidence floor (20 samples, two post-hide grids 110-130 s) -> the 131 s poll,
    # although the hidden pod is not drained.
    assert (at, decision.status) == (131_000, "commit")
    early = decision.details["early_commit"]
    assert early["critical_receivers"] == ["r"] and early["hidden_drained"] is False


@pytest.mark.parametrize("critical", [(), (MODEL,)], ids=["none", "only_the_donor_itself"])
def test_without_another_critical_model_a_busy_hidden_pod_keeps_the_probe_running(critical):
    h = _busy(_CriticalHarness(critical=critical))
    assert all(d.status == "probing" for _, d in h.run_until(149_000, serve=5))


def test_a_critical_model_never_waives_one_p95_e2e():
    # Evidence completeness: SLO samples count at completion; before one e2e after the
    # hide they favour short requests. 45 s p95 e2e -> no commit before 103 + 45 = 148 s.
    h = _CriticalHarness(critical=("r",), p95_e2e_ms=45_000.0)
    h.start()
    at, decision = h.run_until(HIDE + 90_000, serve=5)[-1]
    assert decision.status == "commit" and at == 149_000
    assert decision.details["early_commit"]["min_elapsed_ms"] == 45_000


def test_a_critical_model_never_waives_the_samples():
    h = _busy(_CriticalHarness(critical=("r",)))
    assert all(d.status == "probing" for _, d in h.run_until(141_000, serve=0))

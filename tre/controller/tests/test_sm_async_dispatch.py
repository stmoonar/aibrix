"""Controller side of the per-call drain (TRE_SM_CALL_DRAIN) and async SM operations
(TRE_SM_ASYNC). Design note 20260924-reissue-sidecar.md section 3.5."""
from __future__ import annotations

import asyncio

import pytest

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_controller.config import ControllerConfig
from tre_controller.loops.action_queue import ActionQueue, DispatchResult, QueuedAction
from tre_controller.loops.safescale_task import run_safescale_observation_tick
from tre_controller.loops.tick import with_pending_ops
from tre_controller.planning.planner import ClusterView, HideAction, ScaleAction, UnhideAction
from tre_controller.planning.safescale import CommitDrainPolicy, SafeScaleCommand, SafeScaleDecision
from tre_controller.sm_client import ServiceManagerClient
from tre_sm.allocator.slots import Binding, Slot


class Clock:
    def __init__(self, now_ms: int = 1_000_000) -> None:
        self.now = now_ms

    def __call__(self) -> int:
        return self.now


class FakeClient:
    """Records calls; v2 calls answer 202-style bodies unless ``sync`` is set."""

    def __init__(self, *, sync: bool = False) -> None:
        self.calls: list[tuple] = []
        self.sync = sync
        self.ops: dict[str, list[dict]] = {}
        self.accept_error: str | None = None
        self.audits = 0
        self._next = 0

    def _accept(self, kind: str) -> dict:
        if self.accept_error:
            return {"ok": False, "error": self.accept_error}
        if self.sync:
            return {"ok": True, "response": {"actions": []}}
        self._next += 1
        op_id = f"op-{self._next}"
        self.ops.setdefault(op_id, [{"status": "running"}])
        return {"ok": True, "response": {"async_operation": True, "operation_id": op_id, "status": "pending"}}

    async def scale_model(self, model, delta):
        self.calls.append(("scale", model, delta))
        return {"ok": True}

    async def set_binding_power(self, serve_id, *, awake):
        self.calls.append(("power", serve_id, awake))
        return {"ok": True}

    async def set_routable(self, model, hidden_pods):
        self.calls.append(("routable", model, tuple(hidden_pods)))
        return {"ok": True}

    async def defrag(self, migrations):
        self.calls.append(("defrag",))
        return {"ok": True}

    async def scale_model_v2(self, model, delta, *, drain_s=None, async_op=False, meta=None):
        self.calls.append(("scale_v2", model, delta, drain_s, async_op))
        return self._accept("scale")

    async def set_binding_power_v2(self, serve_id, *, awake, drain_s=None, async_op=False, meta=None):
        self.calls.append(("power_v2", serve_id, awake, drain_s, async_op))
        return self._accept("power")

    async def get_operation(self, operation_id):
        self.calls.append(("get_op", operation_id))
        script = self.ops.get(operation_id)
        if script is None:
            return {"ok": False, "error": "HTTP 404: operation not found", "not_found": True}
        record = script.pop(0) if len(script) > 1 else script[0]
        return {"ok": True, "response": {"operation_id": operation_id, **record}}

    async def get_audit(self):
        self.audits += 1
        return {"ok": True, "response": {"healthy": True, "issues": []}}

    def finish(self, op_id: str, status: str = "succeeded", **extra) -> None:
        self.ops[op_id] = [{"status": status, **extra}]


def _run(coro):
    return asyncio.run(coro)


def _queue(client, clock, **kwargs):
    kwargs.setdefault("async_ops", True)
    kwargs.setdefault("poll_interval_s", 1.0)
    return ActionQueue(client, now_ms=clock, **kwargs)


# ------------------------------------------------------------- flags off


def test_flags_off_uses_main_calls_only():
    client = FakeClient()
    queue = ActionQueue(client)
    queue.submit(
        (
            ScaleAction("donor", -1, "idle_proactive_immediate", "rescue", pods=("d-1",)),
            ScaleAction("recv", 1, "critical_idle_capacity", "rescue"),
            HideAction("probe", ("p-1",), "probe_started", "rescue"),
        )
    )
    results = _run(queue.drain_once())
    assert [item.ok for item in results] == [True, True, True]
    assert client.calls == [
        ("power", "d-1", False),
        ("scale", "recv", 1),
        ("routable", "probe", ("p-1",)),
    ]
    assert queue.pending_ops_view() == ()
    assert queue.inflight_models() == set()


def test_config_defaults_are_off_and_env_turns_them_on():
    base = ControllerConfig.from_env({})
    assert base.sm_async is False and base.sm_call_drain is False
    assert CommitDrainPolicy.from_config(base) is None
    cfg = ControllerConfig.from_env(
        {
            "TRE_SM_ASYNC": "1",
            "TRE_SM_CALL_DRAIN": "1",
            "TRE_SM_ASYNC_OP_TIMEOUT_S": "90",
            "TRE_SAFESCALE_COMMIT_DRAIN_MAX_S": "60",
        }
    )
    assert cfg.sm_async is True and cfg.sm_call_drain is True
    assert cfg.sm_async_op_timeout_s == 90.0
    assert CommitDrainPolicy.from_config(cfg) == CommitDrainPolicy(max_s=60.0)


# ------------------------------------------------------------- per-call drain


def test_call_drain_sends_zero_for_direct_sleeps_and_the_budget_for_commits():
    client = FakeClient(sync=True)
    queue = ActionQueue(client, call_drain=True)
    queue.submit((ScaleAction("donor", -1, "critical_donor_immediate", "rescue", pods=("d-1",)),))
    queue.submit((ScaleAction("idle", -1, "idle_proactive_immediate", "rescue"),))
    queue.submit((ScaleAction("up", 2, "critical_idle_capacity", "rescue"),))
    queue.submit(
        (ScaleAction("probe", -1, "formal_commit_gate_passed", "safescale", pods=("p-1",), drain_s=45.0),)
    )
    results = _run(queue.drain_once())
    assert all(item.ok for item in results)
    assert client.calls == [
        ("power_v2", "d-1", False, 0.0, False),
        ("scale_v2", "idle", -1, 0.0, False),
        ("scale_v2", "up", 2, None, False),
        ("power_v2", "p-1", False, 45.0, False),
    ]


def test_commit_drain_policy_budget():
    policy = CommitDrainPolicy()
    assert policy.budget_s(None) == 30.0
    assert policy.budget_s(3_000.0) == 10.0  # 2 * 3 s = 6 s -> min 10 s
    assert policy.budget_s(20_000.0) == 40.0
    assert policy.budget_s(600_000.0) == 120.0
    with pytest.raises(ValueError):
        CommitDrainPolicy(min_s=200.0, max_s=100.0)


class _CommitMachine:
    def __init__(self) -> None:
        self.resolved: list = []

    def active_probes(self):
        class Probe:
            model = "donor"
            pods = ("pod-a",)

        return (Probe(),)

    def observe(self, model, observation, *, now_ms):
        return SafeScaleDecision(
            status="commit",
            reason="formal_commit_gate_passed",
            commands=(
                SafeScaleCommand(kind="scale_down", model="donor", pods=("pod-a",), delta=-1, reason="formal_commit_gate_passed"),
                SafeScaleCommand(kind="scale_up", model="recv", delta=1, reason="safescale_followup_upscale"),
            ),
        )

    def resolve(self, model, *, status, reason, now_ms):
        self.resolved.append((model, status))
        return True


def _registry() -> Registry:
    trs = TrsParams(0.04, 1.0, 2.0, 1.0, 0.0, 100.0, 0.8, 1.0, 1.25, 4.0, 0.05, 1)
    slo = SloSpec(ttft_p95_ms=1000.0, tpot_p95_ms=100.0, e2e_p95_ms=10_000.0)
    specs = [
        ModelSpec(name, "/w", tp, 0, 4, "image", slo, trs) for name, tp in (("donor", 1), ("recv", 1), ("big", 2))
    ]
    return Registry(ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)), specs)


def _snapshot(e2e_p95_ms):
    return MetricsSnapshot(
        ts_ms=5_000,
        stale=False,
        models={
            "donor": ModelWindowMetrics(
                model="donor", window_start_ms=0, window_end_ms=60_000, prompt_tokens=0.0,
                generation_tokens=100.0, avg_waiting=0.0, avg_running=1.0, avg_swapping=0.0,
                kv_cache_hit_rate=0.0, ttft_p95_ms=100.0, tpot_p95_ms=10.0, e2e_p95_ms=e2e_p95_ms,
                routable_pods=1, assigned_replicas=2, per_pod={},
            )
        },
    )


class _ListQueue:
    def __init__(self) -> None:
        self.submitted: list[tuple] = []

    def submit(self, actions):
        from tre_controller.loops.action_queue import SubmitResult

        self.submitted.append(tuple(actions))
        return SubmitResult(accepted=len(actions))


@pytest.mark.parametrize("policy, expected", [(None, None), (CommitDrainPolicy(), 50.0)])
def test_safescale_commit_carries_the_drain_budget_only_with_the_flag(policy, expected):
    queue = _ListQueue()
    result = run_safescale_observation_tick(
        _snapshot(25_000.0), queue=queue, registry=_registry(), safescale=_CommitMachine(),
        commit_drain=policy,
    )
    (batch,) = queue.submitted
    down, up = batch
    assert down.delta == -1 and down.pods == ("pod-a",) and down.drain_s == expected
    assert up.drain_s is None
    assert any(event.startswith("safescale_commit_drain:") for event in result.events) is (policy is not None)


# ------------------------------------------------------------- async dispatch


def test_async_dispatch_does_not_block_and_keeps_the_model_inflight():
    client = FakeClient()
    clock = Clock()
    queue = _queue(client, clock)
    queue.submit((ScaleAction("donor", -1, "idle_proactive_immediate", "rescue"),))
    queue.submit((HideAction("other", ("o-1",), "probe_started", "rescue"),))

    results = _run(queue.drain_once())

    # The scale is accepted (202) and tracked; the hide of another model went out in
    # the same drain instead of waiting behind the SM operation.
    assert results == (DispatchResult(model="other", action_kind="hide", ok=True),)
    assert queue.inflight_models() == {"donor"}
    (view,) = queue.pending_ops_view()
    assert view.model == "donor" and view.delta == -1
    assert ("routable", "other", ("o-1",)) in client.calls
    # no duplicate action for a model with an active operation (main would let a
    # rescue action through for an inflight model)
    dropped = queue.submit((ScaleAction("donor", -1, "idle_proactive_immediate", "rescue"),))
    assert dropped.accepted == 0 and dropped.dropped == (("donor", "active_op"),)
    dropped = queue.submit((ScaleAction("donor", 1, "low_fairness_idle_capacity", "fairness"),))
    assert dropped.accepted == 0 and dropped.dropped == (("donor", "inflight"),)
    # a safescale batch touching it is rejected as a whole (never partially queued)
    batch = queue.submit(
        (
            ScaleAction("donor", -1, "formal_commit_gate_passed", "safescale", pods=("x",)),
            ScaleAction("recv", 1, "safescale_followup_upscale", "safescale"),
        )
    )
    assert batch.accepted == 0 and ("recv", "atomic_batch_conflict") in batch.dropped


def test_polling_is_bounded_and_cooldown_starts_at_completion():
    client = FakeClient()
    clock = Clock()
    queue = _queue(client, clock)
    queue.submit((ScaleAction("donor", -1, "idle_proactive_immediate", "rescue"),))
    _run(queue.drain_once())  # dispatch at t0
    dispatched_at = clock.now

    clock.now += 200
    _run(queue.drain_once())
    assert [call for call in client.calls if call[0] == "get_op"] == []  # not due yet

    clock.now += 1_000
    _run(queue.drain_once())
    assert [call for call in client.calls if call[0] == "get_op"] == [("get_op", "op-1")]
    assert queue.inflight_models() == {"donor"}
    assert queue.last_actions() == {}

    client.finish("op-1", summary={"slept": ["d"], "drained_s": {"d": 0.4}, "interrupted": 2})
    clock.now += 5_000
    results = _run(queue.drain_once())
    assert results == (DispatchResult(model="donor", action_kind="scale", ok=True),)
    assert queue.inflight_models() == set()
    assert queue.last_actions() == {"donor": (clock.now, "down")}
    assert queue.last_actions()["donor"][0] > dispatched_at
    stats = queue.op_stats()["scale:down:rescue"]
    assert stats["count"] == 1 and stats["ok"] == 1 and stats["interrupted_sum"] == 2
    assert stats["drained_s_sum"] == pytest.approx(0.4)


def test_max_polls_per_drain_bounds_the_status_requests():
    client = FakeClient()
    clock = Clock()
    queue = _queue(client, clock, max_polls=2)
    for model in ("a", "b", "c"):
        queue.submit((ScaleAction(model, -1, "idle_proactive_immediate", "rescue"),))
    _run(queue.drain_once())
    clock.now += 2_000
    _run(queue.drain_once())
    assert len([call for call in client.calls if call[0] == "get_op"]) == 2


def test_failed_operation_is_reported_without_cooldown_and_triggers_audit():
    client = FakeClient()
    clock = Clock()
    queue = _queue(client, clock)
    queue.submit((ScaleAction("donor", -1, "idle_proactive_immediate", "rescue"),))
    _run(queue.drain_once())
    client.finish("op-1", status="failed", error="WakeConflict: slot busy")
    clock.now += 2_000

    results = _run(queue.drain_once())

    assert results == (
        DispatchResult(model="donor", action_kind="scale", ok=False, error="WakeConflict: slot busy"),
    )
    assert queue.last_actions() == {}
    assert queue.inflight_models() == set()
    assert client.audits == 1
    assert queue.op_stats()["scale:down:rescue"]["failed"] == 1


def test_operation_timeout_and_unknown_operation():
    client = FakeClient()
    clock = Clock()
    queue = _queue(client, clock, op_timeout_s=10.0)
    queue.submit((ScaleAction("slow", -1, "idle_proactive_immediate", "rescue"),))
    _run(queue.drain_once())
    clock.now += 11_000
    (result,) = _run(queue.drain_once())
    assert result.ok is False and result.error == "op_timeout"
    assert queue.op_stats()["scale:down:rescue"]["timeout"] == 1

    queue.submit((ScaleAction("lost", -1, "idle_proactive_immediate", "rescue"),))
    _run(queue.drain_once())
    client.ops.pop("op-2")  # the SM lost the record (e.g. Redis flushed)
    outcomes = []
    for _ in range(3):
        clock.now += 1_500
        outcomes.extend(_run(queue.drain_once()))
    assert outcomes == [DispatchResult(model="lost", action_kind="scale", ok=False, error="operation_unknown")]


def test_sync_answer_from_an_sm_without_async_completes_immediately():
    client = FakeClient(sync=True)
    queue = _queue(client, Clock())
    queue.submit((ScaleAction("m", 1, "critical_idle_capacity", "rescue"),))
    assert _run(queue.drain_once()) == (DispatchResult(model="m", action_kind="scale", ok=True),)
    assert queue.inflight_models() == set()


def test_accept_failure_is_a_normal_dispatch_failure():
    client = FakeClient()
    client.accept_error = "HTTP 409: slot already has awake binding"
    queue = _queue(client, Clock())
    queue.submit((ScaleAction("m", 1, "critical_sleeping_capacity", "rescue", pods=("m-2",)),))
    (result,) = _run(queue.drain_once())
    assert result.ok is False and "409" in result.error
    assert queue.inflight_models() == set()


# ------------------------------------------------------------- dependencies


def test_receiver_wake_waits_for_the_donor_sleep_it_depends_on():
    client = FakeClient()
    clock = Clock()
    queue = _queue(client, clock)
    queue.submit(
        (
            ScaleAction("donor", -1, "critical_donor_immediate", "rescue", donor="donor", receiver="recv", pods=("d-1",)),
            ScaleAction("recv", 1, "critical_donor_immediate", "rescue", donor="donor", receiver="recv", pods=("r-1",)),
        )
    )
    _run(queue.drain_once())
    assert [call[:2] for call in client.calls if call[0].endswith("_v2")] == [("power_v2", "d-1")]
    assert [item.model for item in queue.pending_actions()] == ["recv"]

    clock.now += 1_500
    _run(queue.drain_once())  # donor still running
    assert [call[:2] for call in client.calls if call[0].endswith("_v2")] == [("power_v2", "d-1")]

    client.finish("op-1")
    clock.now += 1_500
    _run(queue.drain_once())
    assert [call[:2] for call in client.calls if call[0].endswith("_v2")] == [
        ("power_v2", "d-1"),
        ("power_v2", "r-1"),
    ]


def test_receiver_wake_is_dropped_when_the_donor_sleep_failed():
    client = FakeClient()
    clock = Clock()
    queue = _queue(client, clock)
    queue.submit(
        (
            ScaleAction("donor", -1, "critical_donor_immediate", "rescue", donor="donor", receiver="recv", pods=("d-1",)),
            ScaleAction("recv", 1, "critical_donor_immediate", "rescue", donor="donor", receiver="recv", pods=("r-1",)),
        )
    )
    _run(queue.drain_once())
    client.finish("op-1", status="failed", error="sleep_commit_failed")
    clock.now += 1_500
    results = _run(queue.drain_once())
    assert DispatchResult(model="recv", action_kind="scale", ok=False, error="dependency_failed") in results
    assert queue.inflight_models() == set()
    assert not any(call[:2] == ("power_v2", "r-1") for call in client.calls)


# ------------------------------------------------------------- safescale + observe


def _commit_batch():
    return (
        ScaleAction("donor", -1, "formal_commit_gate_passed", "safescale", pods=("pod-a",), drain_s=40.0),
        ScaleAction("recv", 1, "safescale_followup_upscale", "safescale"),
    )


def test_safescale_commit_success_then_followups():
    client = FakeClient()
    clock = Clock()
    queue = _queue(client, clock, call_drain=True)
    assert queue.submit(_commit_batch()).accepted == 2
    _run(queue.drain_once())
    assert client.calls[0] == ("power_v2", "pod-a", False, 40.0, True)
    assert not any(call[0] == "scale_v2" for call in client.calls)  # follow-up waits
    client.finish("op-1")
    clock.now += 1_500
    _run(queue.drain_once())
    assert ("scale_v2", "recv", 1, None, True) in client.calls


def test_failed_safescale_commit_rolls_back_and_drops_followups():
    client = FakeClient()
    clock = Clock()
    queue = _queue(client, clock, call_drain=True)
    queue.submit(_commit_batch())
    _run(queue.drain_once())
    client.finish("op-1", status="failed", error="sleep_commit_failed")
    clock.now += 1_500

    results = _run(queue.drain_once())

    assert DispatchResult(model="donor", action_kind="scale", ok=False, error="sleep_commit_failed") in results
    assert DispatchResult(model="recv", action_kind="scale", ok=False, error="dependency_failed") in results
    # rollback of the commit = unhide the probe pods (safescale-sourced, one-shot)
    assert ("routable", "donor", ()) in client.calls
    assert DispatchResult(model="donor", action_kind="unhide", ok=True) in results
    assert queue.inflight_models() == set()


def test_observe_mode_keeps_tracking_dispatched_ops_and_holds_safescale():
    client = FakeClient()
    clock = Clock()
    observe = {"on": False}
    queue = _queue(client, clock, is_observe=lambda: observe["on"])
    queue.submit(_commit_batch())
    _run(queue.drain_once())  # commit sleep dispatched (op-1), follow-up waiting

    observe["on"] = True
    queue.submit((ScaleAction("idle", -1, "idle_proactive_immediate", "rescue"),))
    client.finish("op-1")
    clock.now += 1_500
    results = _run(queue.drain_once())

    # the already-dispatched op is still polled and completed in observe mode
    assert DispatchResult(model="donor", action_kind="scale", ok=True) in results
    # planner action dropped as usual, the one-shot safescale follow-up is held
    assert DispatchResult(model="idle", action_kind="scale", ok=True, error="observe_skipped") in results
    assert [item.model for item in queue.pending_actions()] == ["recv"]
    assert not any(call[0] == "scale_v2" for call in client.calls)

    observe["on"] = False
    _run(queue.drain_once())
    assert ("scale_v2", "recv", 1, None, True) in client.calls


def test_observe_mode_rollback_of_a_failed_commit_is_held_not_dropped():
    client = FakeClient()
    clock = Clock()
    observe = {"on": False}
    queue = _queue(client, clock, is_observe=lambda: observe["on"])
    queue.submit((_commit_batch()[0],))
    _run(queue.drain_once())
    observe["on"] = True
    client.finish("op-1", status="failed", error="sleep_unverified")
    clock.now += 1_500
    _run(queue.drain_once())
    held = queue.pending_actions()
    assert [type(item.action) for item in held] == [UnhideAction]
    assert held[0].source_loop == "safescale" and queue.inflight_models() == {"donor"}
    observe["on"] = False
    _run(queue.drain_once())
    assert ("routable", "donor", ()) in client.calls


# ------------------------------------------------------------- planner accounting


def _view(*bindings):
    return ClusterView(
        topology=ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)),
        bindings=tuple(bindings),
    )


class _PendingQueue:
    def __init__(self, views):
        self._views = views

    def pending_ops_view(self):
        return self._views


def test_pending_wake_claims_its_gpus_and_sleep_keeps_them_occupied():
    from tre_controller.loops.action_queue import PendingOpView
    from tre_controller.loops.tick import _idle_gpus

    view = _view(
        Binding("donor-1", "donor", Slot("node-a", (0,)), awake=True),
        Binding("recv-1", "recv", Slot("node-a", (1,)), awake=False),
        Binding("recv-2", "recv", Slot("node-a", (2,)), awake=False),
    )
    registry = _registry()
    assert _idle_gpus(None, registry, view) == 3

    pending = (
        PendingOpView("recv", 1, ("recv-1",), "rescue", "critical_sleeping_capacity", 10),
        PendingOpView("donor", -1, ("donor-1",), "rescue", "idle_proactive_immediate", 10),
    )
    planned, events = with_pending_ops(view, _PendingQueue(pending), registry)

    by_id = {binding.serve_id: binding for binding in planned.bindings}
    assert by_id["recv-1"].awake is True and by_id["recv-2"].awake is False
    assert by_id["donor-1"].awake is True  # leaving capacity: still occupied
    assert _idle_gpus(None, registry, planned) == 2
    assert events == ("pending_op_incoming:recv:+1", "pending_op_leaving:donor:-1")


def test_pending_model_level_growth_claims_sleeping_then_free_slots():
    from tre_controller.loops.action_queue import PendingOpView
    from tre_controller.loops.tick import _idle_gpus

    view = _view(
        Binding("donor-1", "donor", Slot("node-a", (0,)), awake=True),
        Binding("recv-1", "recv", Slot("node-a", (0,)), awake=False),  # GPU busy
        Binding("recv-2", "recv", Slot("node-a", (1,)), awake=False),
    )
    pending = (PendingOpView("recv", 2, (), "fairness", "low_fairness_idle_capacity", 5),)
    planned, _events = with_pending_ops(view, _PendingQueue(pending), _registry())
    awake = {binding.serve_id for binding in planned.bindings if binding.awake}
    assert "recv-2" in awake and "recv-1" not in awake
    assert any(binding.serve_id.startswith("pending-recv-") for binding in planned.bindings)
    assert _idle_gpus(None, _registry(), planned) == 1


def test_no_pending_ops_leaves_the_view_untouched():
    view = _view(Binding("a", "donor", Slot("node-a", (0,)), awake=True))
    assert with_pending_ops(view, _PendingQueue(()), _registry()) == (view, ())
    assert with_pending_ops(view, object(), _registry()) == (view, ())
    assert with_pending_ops(None, _PendingQueue(("x",)), _registry()) == (None, ())


# ------------------------------------------------------------- SM client wire


class _Transport:
    def __init__(self, response=None, error=None):
        self.requests: list[tuple] = []
        self.response = response or {}
        self.error = error

    async def request(self, method, url, *, json=None, timeout_s):
        self.requests.append((method, url, json, timeout_s))
        if url.endswith("/v2/state"):
            return {"models": {"m": {"awake": 3, "bound": 4}}}
        if self.error:
            raise self.error
        return self.response


def test_client_v2_wire_format():
    transport = _Transport({"async_operation": True, "operation_id": "x"})
    client = ServiceManagerClient("http://sm", transport=transport, slow_timeout_s=300.0)

    _run(client.scale_model_v2("m", -1, drain_s=0.0, async_op=True))
    _run(client.set_binding_power_v2("m-1", awake=False, drain_s=45, async_op=False))
    _run(client.scale_model_v2("m", 1))
    _run(client.get_operation("x"))

    puts = [request for request in transport.requests if request[0] == "PUT"]
    assert puts == [
        ("PUT", "http://sm/v2/models/m/target?async=1", {"wake_replicas": 2, "drain_s": 0.0}, 300.0),
        ("PUT", "http://sm/v2/bindings/m-1/power", {"awake": False, "drain_s": 45.0}, 300.0),
        ("PUT", "http://sm/v2/models/m/target", {"wake_replicas": 4}, 300.0),
    ]
    assert transport.requests[-1][:2] == ("GET", "http://sm/v2/operations/x")


def test_client_get_operation_flags_404():
    from tre_controller.sm_client import ServiceManagerError

    client = ServiceManagerClient(
        "http://sm", transport=_Transport(error=ServiceManagerError("HTTP 404: operation not found"))
    )
    response = _run(client.get_operation("gone"))
    assert response["ok"] is False and response["not_found"] is True

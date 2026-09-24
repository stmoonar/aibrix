"""Review fixes on the controller side of TRE_SM_ASYNC / TRE_SM_CALL_DRAIN
(H1, H2, M2, M3, M4, L4)."""
from __future__ import annotations

import asyncio
import importlib.util
import logging
from pathlib import Path

from tre_common.registry import ClusterTopology, NodeSpec
from tre_controller.loops.action_queue import ActionQueue, DispatchResult
from tre_controller.loops.cluster_view_task import cluster_view_from_state
from tre_controller.loops.tick import _awake_including_hidden
from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, TauThresholds
from tre_controller.planning.planner import PlanConfig, ScaleAction, UnhideAction, build_plan

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_async_dispatch_helpers", _HERE / "test_sm_async_dispatch.py")
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
FakeClient = _helpers.FakeClient
Clock = _helpers.Clock


def _run(coro):
    return asyncio.run(coro)


def _queue(client, clock, **kwargs):
    kwargs.setdefault("async_ops", True)
    return ActionQueue(client, now_ms=clock, **kwargs)


def _v2_calls(client):
    return [call for call in client.calls if call[0] in ("power_v2", "scale_v2")]


# ------------------------------------------------------------------ H1


def test_h1_rescue_replacing_a_pending_fairness_donor_releases_its_receiver():
    client = FakeClient()
    queue = _queue(client, Clock())
    queue.submit(
        (
            ScaleAction("donor", -1, "low_fairness_donor_immediate", "fairness", donor="donor", receiver="recv", pods=("d-1",)),
            ScaleAction("recv", 1, "low_fairness_donor_immediate", "fairness", donor="donor", receiver="recv", pods=("r-1",)),
        )
    )
    # main's rescue path drops the pending fairness donor action of that model
    rescue = queue.submit((ScaleAction("donor", 1, "critical_idle_capacity", "rescue"),))
    assert rescue.replaced == (("donor", "fairness"),)

    results = _run(queue.drain_once())

    assert DispatchResult(model="recv", action_kind="scale", ok=False, error="dependency_failed") in results
    assert "recv" not in queue.inflight_models()
    assert [item.model for item in queue.pending_actions()] == []
    assert not any(call[:2] == ("power_v2", "r-1") for call in client.calls)


def test_h1_dependency_that_vanished_without_outcome_fails_instead_of_waiting_forever():
    client = FakeClient()
    queue = _queue(client, Clock())
    queue.submit(
        (
            ScaleAction("donor", -1, "critical_donor_immediate", "rescue", donor="donor", receiver="recv", pods=("d-1",)),
            ScaleAction("recv", 1, "critical_donor_immediate", "rescue", donor="donor", receiver="recv", pods=("r-1",)),
        )
    )
    queue._pending.popleft()  # the donor action disappears (e.g. pruned bookkeeping)
    (result,) = _run(queue.drain_once())
    assert result == DispatchResult(model="recv", action_kind="scale", ok=False, error="dependency_failed")


# ------------------------------------------------------------------ H2


class RoutableScript(FakeClient):
    def __init__(self, answers):
        super().__init__()
        self.answers = list(answers)

    def _answer(self):
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return {"ok": True} if answer == "ok" else {"ok": False, "error": answer}

    async def set_routable(self, model, hidden_pods):
        self.calls.append(("routable", model, tuple(hidden_pods)))
        return self._answer()

    async def set_routable_v2(self, model, hidden_pods, *, timeout_s):
        self.calls.append(("routable_v2", model, tuple(hidden_pods), timeout_s))
        return self._answer()


def test_h2_oneshot_unhide_is_retried_on_409_not_dropped():
    client = RoutableScript(["HTTP 409: writer busy", "request timed out", "ok"])
    clock = Clock()
    queue = _queue(client, clock)
    queue.submit((UnhideAction("m", ("p-1",), "slo_violation", "safescale"),))

    assert _run(queue.drain_once()) == ()
    assert queue.inflight_models() == {"m"}
    (held,) = queue.pending_actions()
    assert held.attempt == 1
    clock.now += 500
    assert _run(queue.drain_once()) == ()  # still backing off: no call
    clock.now += 600
    assert _run(queue.drain_once()) == ()  # 2nd attempt: timeout
    clock.now += 2_100
    assert _run(queue.drain_once()) == (DispatchResult(model="m", action_kind="unhide", ok=True),)
    assert [call[0] for call in client.calls] == ["routable_v2"] * 3
    assert client.calls[0][3] == 45.0  # timeout above the SM lock wait
    assert queue.inflight_models() == set()


def test_h2_oneshot_gives_up_after_its_budget_with_a_structured_alert(caplog):
    client = RoutableScript(["HTTP 409: writer busy"])
    clock = Clock()
    queue = _queue(client, clock, oneshot_retry_s=3.0)
    queue.submit((UnhideAction("m", ("p-1",), "slo_violation", "safescale"),))
    results = []
    with caplog.at_level(logging.ERROR, logger="tre_controller.action_queue"):
        for _ in range(6):
            results.extend(_run(queue.drain_once()))
            clock.now += 11_000
    assert results == [DispatchResult(model="m", action_kind="unhide", ok=False, error="HTTP 409: writer busy")]
    assert any("sm_oneshot_action_failed" in record.getMessage() for record in caplog.records)
    assert queue.inflight_models() == set()


def test_h2_non_retryable_errors_and_planner_actions_are_not_retried():
    client = RoutableScript(["HTTP 400: unknown pods"])
    queue = _queue(client, Clock())
    queue.submit((UnhideAction("m", ("p-1",), "slo_violation", "safescale"),))
    (result,) = _run(queue.drain_once())
    assert result.ok is False and result.error.startswith("HTTP 400")

    client = FakeClient()
    client.accept_error = "HTTP 409: busy"
    queue = _queue(client, Clock())
    queue.submit((ScaleAction("m", -1, "idle_proactive_immediate", "rescue"),))
    (result,) = _run(queue.drain_once())
    assert result.ok is False  # a planner action is re-planned, not retried


# ------------------------------------------------------------------ M2


def _commit(pods=("pod-a",)):
    return (
        ScaleAction("donor", -len(pods), "formal_commit_gate_passed", "safescale", pods=pods, drain_s=40.0),
        ScaleAction("recv", 1, "safescale_followup_upscale", "safescale"),
    )


def test_m2_abandoned_commit_is_not_a_success_no_followup_no_cooldown_no_rollback():
    client = FakeClient()
    clock = Clock()
    queue = _queue(client, clock, call_drain=True)
    queue.submit(_commit())
    _run(queue.drain_once())
    client.finish(
        "op-1",
        status="superseded",
        bindings=[{"serve_id": "pod-a", "action": "sleep", "outcome": "abandoned_target_changed"}],
    )
    clock.now += 1_500

    results = _run(queue.drain_once())

    assert DispatchResult(
        model="donor", action_kind="scale", ok=False, error="not_released:abandoned_target_changed"
    ) in results
    assert DispatchResult(model="recv", action_kind="scale", ok=False, error="dependency_failed") in results
    assert queue.last_actions() == {}
    assert not any(call[0] in ("routable", "routable_v2") for call in client.calls)  # no rollback


def test_m2_partially_released_commit_is_not_a_success():
    client = FakeClient()
    clock = Clock()
    queue = _queue(client, clock, call_drain=True)
    queue.submit(_commit(("pod-a", "pod-b")))
    _run(queue.drain_once())
    for op_id, serve_id, outcome in (("op-1", "pod-a", "slept"), ("op-2", "pod-b", "abandoned_reclaimed")):
        client.finish(
            op_id,
            status="succeeded" if outcome == "slept" else "superseded",
            bindings=[{"serve_id": serve_id, "action": "sleep", "outcome": outcome}],
        )
    clock.now += 1_500
    results = _run(queue.drain_once())
    donor = next(item for item in results if item.model == "donor")
    assert donor.ok is False and "abandoned_reclaimed" in donor.error
    assert not any(call[0] == "scale_v2" for call in client.calls)


def test_m2_fully_released_commit_and_superseded_scale_up_succeed():
    client = FakeClient()
    clock = Clock()
    queue = _queue(client, clock, call_drain=True)
    queue.submit(_commit())
    _run(queue.drain_once())
    client.finish("op-1", bindings=[{"serve_id": "pod-a", "action": "sleep", "outcome": "slept"}])
    clock.now += 1_500
    _run(queue.drain_once())
    assert ("scale_v2", "recv", 1, None, True) in [call[:5] for call in client.calls]
    client.finish("op-2", status="superseded")
    clock.now += 1_500
    assert _run(queue.drain_once()) == (DispatchResult(model="recv", action_kind="scale", ok=True),)


# ------------------------------------------------------------------ M3


def test_m3_rescue_scale_up_supersedes_the_active_scale_down():
    client = FakeClient()
    clock = Clock()
    queue = _queue(client, clock, call_drain=True)
    queue.submit((_commit()[0],))
    _run(queue.drain_once())  # op-1: commit draining
    assert queue.supersedable_models() == {"donor"}

    accepted = queue.submit((ScaleAction("donor", 1, "critical_sleeping_capacity", "rescue", pods=("d-2",)),))
    assert accepted.accepted == 1 and accepted.dropped == ()
    _run(queue.drain_once())

    # model-level (the SM target path reclaims the draining binding), not a wake of d-2
    assert _v2_calls(client)[-1][:5] == ("scale_v2", "donor", 1, None, True)
    (view,) = queue.pending_ops_view()
    assert view.delta == 1
    assert queue.op_stats()["scale:down:safescale"]["failed"] == 1  # superseded_by_controller
    assert not any(call[0] in ("routable", "routable_v2") for call in client.calls)
    client.finish("op-2")
    clock.now += 1_500
    assert _run(queue.drain_once()) == (DispatchResult(model="donor", action_kind="scale", ok=True),)
    assert queue.last_actions()["donor"][1] == "up"


def test_m3_other_actions_stay_blocked_by_the_active_operation():
    client = FakeClient()
    queue = _queue(client, Clock(), call_drain=True)
    queue.submit((_commit()[0],))
    _run(queue.drain_once())
    down = queue.submit((ScaleAction("donor", -1, "idle_proactive_immediate", "rescue"),))
    assert down.dropped == (("donor", "active_op"),)
    fair = queue.submit((ScaleAction("donor", 1, "low_fairness_idle_capacity", "fairness"),))
    assert fair.dropped == (("donor", "inflight"),)


def _critical(model):
    return ModelClassification(
        model_name=model, state=ModelState.CRITICAL, role=ModelRole.RECEIVER, Z_m=0.5,
        eta_m=None, trs=0.0, theta_m=1.0, tau=TauThresholds.from_control(), donor_tier=None,
    )


def test_m3_planner_lets_a_critical_receiver_supersede_its_own_scale_down():
    kwargs = dict(
        model_contexts={"m": {"assigned_replicas": 2, "routable_pods": 1}},
        classifications=[_critical("m")],
        model_replicas={"m": 2},
        idle_gpus=1,
        cfg=PlanConfig(min_replicas_per_model=0, max_replicas_per_model=4),
        inflight_models={"m"},
    )
    assert build_plan(**kwargs).actions == []
    plan = build_plan(**kwargs, supersedable_models={"m"})
    assert [(action.model, action.delta > 0) for action in plan.actions] == [("m", True)]


# ------------------------------------------------------------------ M4


class AdoptingClient(FakeClient):
    def __init__(self, active):
        super().__init__()
        self.active = active
        self.list_calls = 0

    async def list_active_operations(self):
        self.list_calls += 1
        return {"ok": True, "response": {"operations": self.active}}


def test_m4_restarted_controller_adopts_active_ops_and_rolls_back_a_failed_commit():
    meta = {
        "controller": True, "action_id": "a-1", "model": "donor", "source_loop": "safescale",
        "delta": -1, "pods": ["pod-a"], "reason": "formal_commit_gate_passed", "rollback_unhide": ["pod-a"],
    }
    client = AdoptingClient(
        [{"operation_id": "op-9", "status": "running", "model": "donor", "kind": "binding_power", "meta": meta},
         {"operation_id": "op-x", "status": "running", "model": "other", "kind": "model_target",
          "plan": {"direction": "up"}}]
    )
    client.ops["op-9"] = [{"status": "running"}]
    client.ops["op-x"] = [{"status": "running"}]
    clock = Clock()
    queue = _queue(client, clock)

    _run(queue.drain_once())

    assert client.list_calls == 1
    assert queue.inflight_models() == {"donor", "other"}
    assert {view.model: view.delta for view in queue.pending_ops_view()} == {"donor": -1, "other": 1}
    blocked = queue.submit((ScaleAction("donor", -1, "idle_proactive_immediate", "fairness"),))
    assert blocked.accepted == 0
    client.finish("op-9", status="failed", error="sleep_commit_failed")
    clock.now += 1_500
    results = _run(queue.drain_once())
    assert DispatchResult(model="donor", action_kind="unhide", ok=True) in results
    assert ("routable", "donor", ()) in client.calls or any(call[0] == "routable_v2" for call in client.calls)
    _run(queue.drain_once())
    assert client.list_calls == 1  # adoption runs once


def test_m4_async_dispatch_sends_meta_for_takeover():
    class MetaClient(FakeClient):
        async def set_binding_power_v2(self, serve_id, *, awake, drain_s=None, async_op=False, meta=None):
            self.calls.append(("meta", meta))
            return self._accept("power")

    client = MetaClient()
    queue = _queue(client, Clock(), call_drain=True)
    queue.submit((_commit()[0],))
    _run(queue.drain_once())
    (meta,) = [call[1] for call in client.calls if call[0] == "meta"]
    assert meta["controller"] is True and meta["rollback_unhide"] == ["pod-a"]
    assert meta["source_loop"] == "safescale" and meta["delta"] == -1


def test_m4_draining_bindings_are_leaving_capacity():
    topology = ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),))
    state = {
        "bindings": [
            {"serve_id": "m-1", "model": "m", "node": "node-a", "gpu_ids": [0], "awake": True, "hidden": False},
            {"serve_id": "m-2", "model": "m", "node": "node-a", "gpu_ids": [1], "awake": True, "hidden": True,
             "draining": True},
            {"serve_id": "m-3", "model": "m", "node": "node-a", "gpu_ids": [2], "awake": True, "hidden": True},
        ]
    }
    view = cluster_view_from_state(state, topology)
    assert view.draining == frozenset({"m-2"})
    # scaling-cap count: the hidden probe pod counts, the draining one does not
    assert _awake_including_hidden(view) == {"m": 2}
    legacy = cluster_view_from_state({"bindings": [dict(item, draining=False) for item in state["bindings"]]}, topology)
    assert _awake_including_hidden(legacy) == {"m": 3}


# ------------------------------------------------------------------ L4


def test_l4_sync_commit_of_several_pods_drains_them_in_parallel():
    class SlowClient(FakeClient):
        def __init__(self):
            super().__init__(sync=True)
            self.running = 0
            self.peak = 0

        async def set_binding_power_v2(self, serve_id, *, awake, drain_s=None, async_op=False, meta=None):
            self.running += 1
            self.peak = max(self.peak, self.running)
            await asyncio.sleep(0.05)
            self.running -= 1
            self.calls.append(("power_v2", serve_id, awake, drain_s, async_op))
            return {"ok": True, "response": {}}

    client = SlowClient()
    queue = ActionQueue(client, call_drain=True)
    queue.submit((ScaleAction("donor", -3, "formal_commit_gate_passed", "safescale", pods=("a", "b", "c"), drain_s=30.0),))
    (result,) = _run(queue.drain_once())
    assert result.ok is True
    assert client.peak == 3
    assert sorted(call[1] for call in client.calls) == ["a", "b", "c"]

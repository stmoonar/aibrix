"""Review 4 (controller): receiver targets resolved from the SM at dispatch,
fresh-view gating, unconfirmed donor pods never unhidden, preemption
compensation from a fresh view, failed commits unhidden, durable probe
lifecycle (committing -> resolved by the queue, recovered after a restart),
abandoned-after-gate event, defrag visibility."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from tre_controller.loops.action_queue import (
    ActionQueue,
    DispatchResult,
    _dispatch_result,
    revalidate_commit_from_signals,
    revalidate_from_cluster_view,
)
from tre_controller.loops.cluster_view_task import ClusterViewBox
from tre_controller.loops.safescale_task import run_safescale_observation_tick
from tre_controller.loops.tick import _defrag_blocking_events
from tre_controller.planning.planner import (
    DefragAction,
    ReceiverTarget,
    SafeScaleCommitAction,
    ScaleAction,
    UnhideAction,
)
from tre_controller.sm_client import ServiceManagerError
from tre_controller.store.state_store import ControllerStateStore
from tre_sm.allocator.slots import Binding, Slot

from test_action_queue_review3 import ScriptedSM, _blocking_sleep, _calls, _commit, _until
from test_safescale_commit import (
    FakeRedis,
    _machine,
    _metrics,
    _probe_records,
    _registry,
    _start_and_prime,
)


def _view(*bindings):
    return SimpleNamespace(bindings=tuple(bindings))


HIDDEN = Binding("7b-1", "7b", Slot("n", (0,)), awake=True, hidden=True)
ASLEEP = Binding("7b-1", "7b", Slot("n", (0,)), awake=False)
UNCONFIRMED_409 = {
    "ok": False, "error": "HTTP 409: sleep incomplete", "status": 409, "retriable": True,
    "outcomes": [{"serve_id": "7b-1", "status": "unconfirmed"}],
}


# ------------------------------------------------------------------ P2-1
def test_a_receiver_target_is_resolved_once_from_the_sm_then_frozen():
    async def scenario():
        sm = ScriptedSM(results={"target:8b": [{"ok": False, "error": "HTTP 409", "retriable": True}]})
        awake = {"n": 2}

        async def model_awake(model):
            return {"ok": True, "awake": awake["n"]}

        sm.model_awake = model_awake

        async def grow(_seconds):
            awake["n"] = 5  # the SM state changes meanwhile: the retry keeps its target

        queue = ActionQueue(sm, sleep=grow)
        commit = SafeScaleCommitAction(
            donor="7b", pods=(), reason="r", upscales=(ReceiverTarget("8b", 1, None, 8),), donor_done=True
        )
        queue.submit((commit,))
        await queue.drain_once()
        assert _calls(sm) == [("8b", "target", 3), ("8b", "target", 3)]

    asyncio.run(scenario())


def test_the_cluster_view_box_is_fresh_only_for_its_max_age():
    now = {"t": 100.0}
    box = ClusterViewBox(max_age_s=20.0, monotonic=lambda: now["t"])
    assert box.fresh() is None and box.age_s() is None
    view = _view(HIDDEN)
    box.set(view)
    assert box.fresh() is view
    now["t"] = 121.0
    assert box.get() is view and box.fresh() is None
    # a retry is then never skipped on a stale "already asleep"
    still_wanted = revalidate_from_cluster_view(box.fresh)
    assert still_wanted(ScaleAction("7b", -1, "r", "safescale", pods=("7b-1",))) is None


# ------------------------------------------------------------------ P2-2
def test_sm_error_bodies_carry_the_sleep_outcomes():
    error = ServiceManagerError(
        "HTTP 409: ...", status=409, body={"detail": "x", "outcomes": [{"serve_id": "p", "status": "unconfirmed"}]}
    )
    result = error.result()
    assert result["outcomes"] == [{"serve_id": "p", "status": "unconfirmed"}]
    assert _dispatch_result(model="m", action_kind="scale", response=result).unconfirmed == ("p",)


def test_an_abandoned_commit_never_unhides_an_unconfirmed_donor_pod():
    async def scenario():
        states = {"7b": "high"}
        sm = ScriptedSM(results={"7b-1": [dict(UNCONFIRMED_409)], "7b-2": [{"ok": True}]})

        async def turn_critical(_seconds):
            states["7b"] = "critical"

        queue = ActionQueue(sm, sleep=turn_critical, revalidate_commit=revalidate_commit_from_signals(lambda: states))
        queue.submit((_commit(pods=("7b-1", "7b-2")),))
        await queue.drain_once()
        # 7b-1's sleep was sent but not confirmed: it stays hidden
        assert _calls(sm) == [("7b-1", "sleep"), ("7b", "routable", ("7b-1",))]
        assert queue.stats()["unconfirmed_kept_hidden_total"] == 1

    asyncio.run(scenario())


def test_no_unhide_at_all_when_every_donor_pod_is_unconfirmed():
    async def scenario():
        states = {"7b": "high"}
        sm = ScriptedSM(results={"7b-1": [dict(UNCONFIRMED_409)]})
        done = []

        async def turn_critical(_seconds):
            states["7b"] = "critical"

        queue = ActionQueue(
            sm, sleep=turn_critical, revalidate_commit=revalidate_commit_from_signals(lambda: states),
            on_oneshot_done=lambda *args: done.append(args),
        )
        queue.submit((_commit(),))
        await queue.drain_once()
        assert _calls(sm) == [("7b-1", "sleep")]
        assert done == [("7b-0", "rollback", "commit_abandoned: donor 7b is critical (needs capacity); donor pods left hidden")]

    asyncio.run(scenario())


# ------------------------------------------------------------------ P2-3
def test_preemption_restores_only_pods_a_fresh_view_shows_awake_and_hidden():
    async def scenario():
        sm = ScriptedSM(results={"7b-1": [{"ok": False, "error": "HTTP 409", "retriable": True}]})
        queue = ActionQueue(sm, sleep=_blocking_sleep(), fresh_view=lambda: _view(ASLEEP))
        runner = asyncio.ensure_future(queue.run(poll_interval_s=0.001))
        queue.submit((_commit(),))
        assert await _until(lambda: "7b" in queue.preemptible_models())
        # the view shows the probe pod asleep: the unhide restores nothing
        assert queue.submit((ScaleAction("7b", 2, "critical_idle_capacity", "rescue"),)).accepted == 1
        assert await _until(lambda: ("end", "7b", "scale", 2) in sm.events)
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)

    asyncio.run(scenario())


def test_without_a_fresh_view_the_rescue_is_not_shrunk():
    async def scenario():
        sm = ScriptedSM(results={"7b-1": [{"ok": False, "error": "HTTP 409", "retriable": True}]})
        queue = ActionQueue(sm, sleep=_blocking_sleep(), fresh_view=lambda: None)
        runner = asyncio.ensure_future(queue.run(poll_interval_s=0.001))
        queue.submit((_commit(),))
        assert await _until(lambda: "7b" in queue.preemptible_models())
        assert queue.submit((ScaleAction("7b", 1, "critical_idle_capacity", "rescue"),)).accepted == 1
        assert await _until(lambda: ("end", "7b", "scale", 1) in sm.events)
        assert ("7b", "routable", ()) in _calls(sm)  # the commit still became the unhide
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)

    asyncio.run(scenario())


# ------------------------------------------------------------------ P2-4
def test_retries_exhausted_unhide_the_awake_donor_pods_and_resolve_the_probe():
    async def scenario():
        sm = ScriptedSM(results={"7b-1": [{"ok": False, "error": "HTTP 409: busy", "retriable": True}] * 3})
        done = []
        from tre_controller.loops.action_queue import RetryPolicy

        queue = ActionQueue(
            sm, sleep=lambda _s: asyncio.sleep(0), retry=RetryPolicy(max_attempts=3),
            fresh_view=lambda: _view(HIDDEN), on_oneshot_done=lambda *args: done.append(args),
        )
        queue.submit((_commit(),))
        assert queue.has_request("7b-0")
        await queue.drain_once()
        assert _calls(sm) == [("7b-1", "sleep")] * 3 + [("7b", "routable", ())]
        assert queue.stats()["commit_failed_unhide_total"] == 1
        assert done == [("7b-0", "rollback", "safescale_commit_failed")]
        assert not queue.has_request("7b-0")

    asyncio.run(scenario())


def test_a_failed_commit_whose_donor_a_fresh_view_shows_asleep_needs_no_unhide():
    async def scenario():
        sm = ScriptedSM(results={"7b-1": [{"ok": False, "error": "HTTP 400: bad", "status": 400}]})
        done = []
        queue = ActionQueue(sm, fresh_view=lambda: _view(ASLEEP), on_oneshot_done=lambda *args: done.append(args))
        queue.submit((_commit(),))
        await queue.drain_once()
        assert _calls(sm) == [("7b-1", "sleep")]
        assert done and done[0][1] == "rollback" and done[0][2].startswith("commit_failed: HTTP 400")

    asyncio.run(scenario())


def test_a_successful_commit_resolves_its_probe_as_commit():
    async def scenario():
        done = []
        queue = ActionQueue(ScriptedSM(), on_oneshot_done=lambda *args: done.append(args))
        queue.submit((_commit(),))
        await queue.drain_once()
        assert done == [("7b-0", "commit", "formal_commit_gate_passed")]

    asyncio.run(scenario())


def test_an_action_held_in_observe_mode_is_not_resolved():
    async def scenario():
        done = []
        queue = ActionQueue(ScriptedSM(), is_observe=lambda: True, on_oneshot_done=lambda *args: done.append(args))
        queue.submit((_commit(),))
        await queue.drain_once()
        assert done == [] and queue.has_request("7b-0")

    asyncio.run(scenario())


class RecordingSM:
    def __init__(self):
        self.calls = []

    async def set_binding_power(self, serve_id, *, awake, **_kwargs):
        self.calls.append((serve_id, awake))
        return {"ok": True}

    async def set_routable(self, model, hidden_pods):
        self.calls.append((model, "routable", tuple(hidden_pods)))
        return {"ok": True}

    async def scale_model_to(self, model, target):
        self.calls.append((model, "target", target))
        return {"ok": True}

    async def model_awake(self, model):
        return {"ok": True, "awake": 0}


def test_a_committing_probe_survives_a_controller_restart_and_is_finished():
    redis = FakeRedis()
    store = ControllerStateStore(redis)
    machine = _machine(store)
    _start_and_prime(machine)
    # the old controller handed the commit to its queue, then died before dispatch
    old_queue = ActionQueue(RecordingSM(), is_observe=lambda: True)
    run_safescale_observation_tick(_metrics(1000), queue=old_queue, registry=_registry(), safescale=machine)
    [record] = _probe_records(redis).values()
    assert (record["status"], record["resolution"]) == ("committing", "commit")

    restarted = _machine(store)
    assert restarted.restore() == 1
    assert restarted.active_probes() == () and len(restarted.committing_probes()) == 1
    assert restarted.busy_models() == {"donor"}
    sm = RecordingSM()
    queue = ActionQueue(
        sm,
        on_oneshot_done=lambda request_id, status, reason: restarted.resolve_request(
            request_id, status=status, reason=reason, now_ms=5000
        ),
    )
    result = run_safescale_observation_tick(_metrics(2000), queue=queue, registry=_registry(), safescale=restarted)
    assert "safescale_committing_recovered:donor:commit" in result.events
    # the next tick does not submit it twice
    again = run_safescale_observation_tick(_metrics(2500), queue=queue, registry=_registry(), safescale=restarted)
    assert again.submitted == 0
    asyncio.run(queue.drain_once())
    assert sm.calls == [("pod-a", False)]  # donor slept; the upscale is re-planned, not re-sent
    [record] = _probe_records(redis).values()
    assert (record["status"], record["resolution"]) == ("resolved", "commit")
    assert restarted.busy_models() == set()


def test_a_committing_probe_is_not_preempted_by_the_planner():
    redis = FakeRedis()
    machine = _machine(ControllerStateStore(redis))
    _start_and_prime(machine)
    queue = ActionQueue(RecordingSM(), is_observe=lambda: True)
    run_safescale_observation_tick(_metrics(1000), queue=queue, registry=_registry(), safescale=machine)
    assert machine.request_preemption("donor") == 0  # never observed again: no rollback would come
    assert machine.active_probes() == ()


def test_a_committing_probe_whose_recovery_keeps_failing_is_resolved():
    redis = FakeRedis()
    machine = _machine(ControllerStateStore(redis))
    _start_and_prime(machine)

    class LosingQueue:
        def submit(self, actions):
            from tre_controller.loops.action_queue import SubmitResult

            return SubmitResult(accepted=len(actions))

        def has_request(self, request_id):
            return False

    queue = LosingQueue()
    run_safescale_observation_tick(_metrics(1000), queue=queue, registry=_registry(), safescale=machine)
    events = []
    for ts in range(2000, 9000, 1000):
        events.extend(run_safescale_observation_tick(_metrics(ts), queue=queue, registry=_registry(), safescale=machine).events)
    assert "safescale_recovery_exhausted:donor" in events
    assert machine.busy_models() == set()


# ------------------------------------------------------------------ P3
def test_an_abandon_on_the_first_dispatch_is_reported_as_after_gate():
    async def scenario():
        queue = ActionQueue(ScriptedSM(), revalidate_commit=revalidate_commit_from_signals(lambda: {"7b": "critical"}))
        queue.submit((_commit(),))
        await queue.drain_once()
        assert queue.stats()["commit_abandoned_after_gate_total"] == 1

    asyncio.run(scenario())


def test_a_pending_defrag_is_visible_to_the_rescue_planner():
    async def scenario():
        sm = ScriptedSM(gated={"defrag"})
        queue = ActionQueue(sm)
        runner = asyncio.ensure_future(queue.run(poll_interval_s=0.001))
        queue.submit((DefragAction((), "fragmented", "fairness"),))
        assert await _until(lambda: ("start", "cluster", "defrag") in sm.events)
        rescue = (ScaleAction("8b", 1, "critical_idle_capacity", "rescue"),)
        assert _defrag_blocking_events(queue, rescue) == ("rescue_waits_for_defrag:8b",)
        queue.submit(rescue)
        stats = queue.stats()
        assert stats["defrag_active"] == 1 and stats["pending_behind_defrag"] == 1
        sm.gates["defrag"].set()
        assert await _until(lambda: ("end", "8b", "scale", 1) in sm.events)
        assert _defrag_blocking_events(queue, rescue) == ()
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)

    asyncio.run(scenario())


def test_dispatch_result_default_has_no_unconfirmed_pods():
    assert DispatchResult(model="m", action_kind="scale", ok=True).unconfirmed == ()
    assert UnhideAction("m", ("p",), "r", "safescale", request_id="a") == UnhideAction("m", ("p",), "r", "safescale")

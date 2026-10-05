"""Review 3: SafeScale commit as one ordered, revalidated, idempotent unit;
rescue preempting a backing-off commit; defrag serialization; worker errors."""

from __future__ import annotations

import asyncio

from tre_controller.loops.action_queue import (
    ActionQueue,
    RetryPolicy,
    revalidate_commit_from_signals,
)
from tre_controller.loops.model_state_box import UNCONFIRMED, ModelStateBox
from tre_controller.loops.safescale_task import _commands_to_actions
from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, TauThresholds
from tre_controller.planning.planner import (
    ClusterView,
    DefragAction,
    PlanConfig,
    ReceiverTarget,
    SafeScaleCommitAction,
    ScaleAction,
    TransferIntent,
    build_plan,
)
from tre_controller.planning.safescale import SafeScaleCommand
from tre_controller.sm_client import ServiceManagerClient
from tre_common.registry import ClusterTopology, NodeSpec
from tre_sm.allocator.slots import Binding, Slot


class ScriptedSM:
    """Controller-side fake SM: per-call scripted results, optional gates, a
    record of every call in order."""

    def __init__(self, *, results=None, gated=(), raises=None) -> None:
        self.events: list[tuple] = []
        self.results = {key: list(value) for key, value in (results or {}).items()}
        self.gates = {key: asyncio.Event() for key in gated}
        self.raises = dict(raises or {})

    async def _call(self, key, event):
        self.events.append(("start",) + event)
        gate = self.gates.get(key)
        if gate is not None:
            await gate.wait()
        else:
            await asyncio.sleep(0)
        self.events.append(("end",) + event)
        if key in self.raises:
            raise self.raises[key]
        scripted = self.results.get(key)
        if scripted:
            return scripted.pop(0)
        return {"ok": True}

    async def set_binding_power(self, serve_id, *, awake, **_kwargs):
        return await self._call(serve_id, (serve_id, "wake" if awake else "sleep"))

    async def scale_model_to(self, model, target):
        return await self._call(f"target:{model}", (model, "target", target))

    async def model_awake(self, model):
        return {"ok": True, "awake": 1}

    async def scale_model(self, model, delta, **_kwargs):
        return await self._call(f"scale:{model}", (model, "scale", delta))

    async def set_routable(self, model, hidden_pods):
        return await self._call(f"routable:{model}", (model, "routable", tuple(hidden_pods)))

    async def defrag(self, migrations):
        return await self._call("defrag", ("cluster", "defrag"))

    async def transfer(self, donor_model, receiver_model, count, *, sleep_path="urgent"):
        relay = f"{donor_model}->{receiver_model}"
        result = await self._call(f"transfer:{relay}", (relay, "transfer", count))
        if result == {"ok": True}:  # nothing scripted: every pair done
            return {"ok": True, "response": transfer_body(count)}
        return result


def transfer_body(count, *, done=None, taken=None, unfilled=0, clamped=False, statuses=None,
                  refusals=(), skipped=None, transfer_id="tr-1"):
    """A ``POST /v2/transfers`` response body: ``count`` pairs 7b-i -> 8b-i on n/i."""
    statuses = list(statuses) if statuses is not None else ["done"] * count
    pairs = []
    for index, status in enumerate(statuses):
        pair = {"donor": f"7b-{index}", "donors": [f"7b-{index}"], "donor_binding_ids": [f"7b/n/{index}"],
                "receiver": f"8b-{index}", "receiver_binding_id": f"8b/n/{index}", "node": "n",
                "gpu_ids": [index], "status": status}
        pairs.append(pair)
    done = sum(1 for status in statuses if status == "done") if done is None else done
    taken = sum(1 for status in statuses if status != "donor_sleep_failed") if taken is None else taken
    return {
        "transfer_id": transfer_id, "donor_model": "7b", "receiver_model": "8b", "count": count,
        "pairs": pairs, "done": done, "taken": taken, "clamped_by_floor": clamped,
        "unfilled": unfilled,
        "refusals": list(refusals), "skipped": dict(skipped or {}),
        "picked": [{"serve_id": pair["receiver"]} for pair in pairs if pair["status"] == "done"],
        "phases_ms": {"L1": 1, "U1": 2, "L2": 1, "U2": 2, "L3": 1},
    }


def _commit(*, donor="7b", pods=("7b-1",), upscales=(("8b", 1, 3),)):
    return SafeScaleCommitAction(
        donor=donor,
        pods=pods,
        reason="formal_commit_gate_passed",
        upscales=tuple(ReceiverTarget(model, delta, target) for model, delta, target in upscales),
        request_id=f"{donor}-0",
    )


async def _until(predicate, steps=500):
    for _ in range(steps):
        if predicate():
            return True
        await asyncio.sleep(0.001)
    return predicate()


def _calls(sm):
    return [event[1:] for event in sm.events if event[0] == "start"]


# ------------------------------------------------------------------ P2-1 idempotency


class StatefulTransport:
    """An SM behind HTTP: GET /v2/state counts; PUT target applies (grow-only
    with at_least) - the first PUT succeeds on the SM but times out on the client."""

    def __init__(self, awake: int) -> None:
        self.awake = awake
        self.puts: list[dict] = []
        self.timeouts_left = 1

    async def request(self, method, url, *, json=None, timeout_s):
        if method == "GET" and url.endswith("/v2/state"):
            return {"models": {"8b": {"awake": self.awake, "bound": 4}}}
        if method == "PUT" and url.endswith("/v2/models/8b/target"):
            self.puts.append(dict(json))
            target = int(json["wake_replicas"])
            if json.get("at_least"):
                self.awake = max(self.awake, target)
            else:
                self.awake = target
            if self.timeouts_left:
                self.timeouts_left -= 1
                raise TimeoutError("read timed out")
            return {"ok": True}
        raise AssertionError((method, url))


def test_retried_upscale_after_sm_side_success_and_client_timeout_is_applied_once() -> None:
    async def scenario():
        transport = StatefulTransport(awake=2)
        client = ServiceManagerClient("http://sm", transport=transport)
        slept: list[float] = []

        async def fake_sleep(seconds):
            slept.append(seconds)

        queue = ActionQueue(client, sleep=fake_sleep)
        # donor already slept on an earlier try: only the receiver part is left
        commit = SafeScaleCommitAction(
            donor="7b", pods=("7b-1",), reason="r", upscales=(ReceiverTarget("8b", 1, 3),), donor_done=True
        )
        assert queue.submit((commit,)).accepted == 1
        results = await queue.drain_once()
        assert transport.awake == 3  # NOT 4: the retry did not add the delta again
        # the retry saw the target reached (pre-check): no second PUT at all
        assert transport.puts == [{"wake_replicas": 3, "at_least": True}]
        [result] = results
        assert result.ok and result.model == "8b" and result.attempts == 2
        assert slept == [2.0]

    asyncio.run(scenario())


def test_pod_less_relative_scale_is_never_retried() -> None:
    async def scenario():
        sm = ScriptedSM(results={"scale:m": [{"ok": False, "error": "timeout", "retriable": True}]})
        queue = ActionQueue(sm, sleep=lambda _s: asyncio.sleep(0))
        queue.submit((ScaleAction("m", 1, "legacy_followup", "safescale"),))
        [result] = await queue.drain_once()
        assert not result.ok and result.attempts == 1
        assert _calls(sm) == [("m", "scale", 1)]

    asyncio.run(scenario())


def test_commit_targets_are_resolved_at_first_dispatch_and_capped() -> None:
    topology = ClusterTopology(nodes=(NodeSpec(name="n", gpus=4, two_gpu_slots=((0, 1), (2, 3))),))
    view = ClusterView(
        topology,
        (
            Binding("8b-0", "8b", Slot("n", (0,)), awake=True),
            Binding("8b-1", "8b", Slot("n", (1,)), awake=True, hidden=True),
            Binding("8b-2", "8b", Slot("n", (2,)), awake=False),
            Binding("7b-0", "7b", Slot("n", (3,)), awake=True, hidden=True),
        ),
    )

    class Spec:
        scale_max_replicas = 3

    class Reg:
        def model(self, name):
            return Spec()

    [commit] = _commands_to_actions(
        (
            SafeScaleCommand(kind="scale_down", model="7b", pods=("7b-0",), delta=-1, reason="formal_commit_gate_passed"),
            SafeScaleCommand(kind="scale_up", model="8b", delta=2, reason="safescale_followup_upscale"),
        ),
        cluster_view=view,
        registry=Reg(),
        request_id="7b-1",
    )
    assert isinstance(commit, SafeScaleCommitAction)
    # Review 4 P2-1: never from the (possibly stale) view - resolved from the SM
    # at the first dispatch, capped at max_awake 3, then frozen.
    assert commit.upscales == (ReceiverTarget("8b", 2, None, 3),)
    assert commit.pods == ("7b-0",) and commit.request_id == "7b-1"

    async def scenario():
        sm = ScriptedSM()

        async def awake(model):
            return {"ok": True, "awake": 2}  # hidden counts, as on the SM

        sm.model_awake = awake
        queue = ActionQueue(sm, sleep=lambda _s: asyncio.sleep(0))
        queue.submit((commit,))
        await queue.drain_once()
        # 2 awake + 2 = 4, capped at 3
        assert _calls(sm) == [("7b-0", "sleep"), ("8b", "target", 3)]

    asyncio.run(scenario())


# ------------------------------------------------------------------ P2-2 ordering


def test_commit_wakes_receivers_only_after_the_donor_slept() -> None:
    async def scenario():
        sm = ScriptedSM(gated={"7b-1"})
        queue = ActionQueue(sm)
        runner = asyncio.ensure_future(queue.run(poll_interval_s=0.001))
        queue.submit((_commit(upscales=(("8b", 1, 3), ("14b", 1, 2))),))
        assert queue.inflight_models() == {"7b", "8b", "14b"}
        assert await _until(lambda: ("start", "7b-1", "sleep") in sm.events)
        await asyncio.sleep(0.01)
        assert not any(event[2] == "target" for event in sm.events if len(event) > 2)
        sm.gates["7b-1"].set()
        assert await _until(lambda: ("end", "8b", "target", 3) in sm.events and ("end", "14b", "target", 2) in sm.events)
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
        assert _calls(sm) == [("7b-1", "sleep"), ("14b", "target", 2), ("8b", "target", 3)] or _calls(sm) == [
            ("7b-1", "sleep"), ("8b", "target", 3), ("14b", "target", 2)
        ]
        assert queue.inflight_models() == set()
        assert {m: d for m, (_, d) in queue.last_actions().items()} == {"7b": "down", "8b": "up", "14b": "up"}

    asyncio.run(scenario())


def test_commit_donor_failure_drops_every_receiver_wake() -> None:
    async def scenario():
        sm = ScriptedSM(results={"7b-1": [{"ok": False, "error": "HTTP 400: unknown binding", "status": 400}]})
        queue = ActionQueue(sm)
        queue.submit((_commit(),))
        results = await queue.drain_once()
        # Review 4 P2-4: the donor's hidden probe pod gets its routing back
        assert _calls(sm) == [("7b-1", "sleep"), ("7b", "routable", ())]
        by_model = {}
        for result in results:
            by_model.setdefault(result.model, []).append(result)
        assert not by_model["7b"][0].ok and by_model["7b"][-1].action_kind == "unhide"
        assert by_model["8b"][0].error.startswith("donor_sleep_failed: HTTP 400")
        assert queue.stats()["commit_receiver_dropped_total"] == 1
        assert queue.stats()["commit_failed_unhide_total"] == 1

    asyncio.run(scenario())


def test_commit_retry_resends_only_the_pending_receiver() -> None:
    async def scenario():
        sm = ScriptedSM(results={"target:8b": [{"ok": False, "error": "HTTP 409: busy", "retriable": True}]})
        queue = ActionQueue(sm, sleep=lambda _s: asyncio.sleep(0))
        queue.submit((_commit(upscales=(("8b", 1, 3), ("14b", 1, 2))),))
        results = await queue.drain_once()
        assert _calls(sm) == [
            ("7b-1", "sleep"), ("8b", "target", 3), ("14b", "target", 2), ("8b", "target", 3),
        ]
        assert all(result.ok for result in results) and len(results) == 3

    asyncio.run(scenario())


# ------------------------------------------------------------------ revalidation


def test_revalidation_abandons_the_commit_when_the_donor_turns_critical() -> None:
    async def scenario():
        states = {"7b": "high", "8b": "critical"}
        sm = ScriptedSM(results={"7b-1": [{"ok": False, "error": "HTTP 409: reserved", "retriable": True}]})

        async def turn_critical(_seconds):
            states["7b"] = "critical"  # the donor needs capacity by the retry

        queue = ActionQueue(
            sm, sleep=turn_critical, revalidate_commit=revalidate_commit_from_signals(lambda: states)
        )
        queue.submit((_commit(),))
        results = await queue.drain_once()
        # the hidden pod is unhidden instead of slept, the receiver wake is dropped
        assert _calls(sm) == [("7b-1", "sleep"), ("7b", "routable", ())]
        by_model = {}
        for result in results:
            by_model.setdefault(result.model, []).append(result)
        assert by_model["8b"][0].error.startswith("commit_abandoned: donor 7b is critical")
        assert by_model["7b"][-1].ok and by_model["7b"][-1].action_kind == "unhide"
        assert queue.stats()["commit_abandoned_total"] == 1

    asyncio.run(scenario())


def test_revalidation_drops_an_upscale_the_receiver_no_longer_needs() -> None:
    async def scenario():
        states = {"7b": "high", "8b": "healthy", "14b": "critical"}
        sm = ScriptedSM()
        queue = ActionQueue(sm, revalidate_commit=revalidate_commit_from_signals(lambda: states))
        queue.submit((_commit(upscales=(("8b", 1, 3), ("14b", 1, 2))),))
        results = await queue.drain_once()
        assert _calls(sm) == [("7b-1", "sleep"), ("14b", "target", 2)]
        dropped = [r for r in results if r.model == "8b"]
        assert dropped and dropped[0].error.startswith("not_wanted: receiver 8b is healthy")

    asyncio.run(scenario())


def test_revalidation_skips_a_donor_that_already_slept() -> None:
    class View:
        bindings = (Binding("7b-1", "7b", Slot("n", (0,)), awake=False),)

    async def scenario():
        sm = ScriptedSM()
        queue = ActionQueue(sm, revalidate_commit=revalidate_commit_from_signals(lambda: {"7b": "critical"}, lambda: View()))
        queue.submit((_commit(),))
        await queue.drain_once()
        # asleep already (a timed-out success): nothing to unhide, receiver still woken
        assert _calls(sm) == [("8b", "target", 3)]

    asyncio.run(scenario())


def test_model_state_box_marks_unconfirmed_receivers_and_expires() -> None:
    now = {"ms": 100_000}
    box = ModelStateBox(max_age_ms=30_000, now_ms=lambda: now["ms"])

    def cls(model, state, role):
        return ModelClassification(
            model_name=model, state=state, role=role, Z_m=1.0, eta_m=None, trs=0.0,
            theta_m=1.0, tau=TauThresholds.from_control(),
        )

    box.update(
        {
            "a": cls("a", ModelState.CRITICAL, ModelRole.RECEIVER),
            "b": cls("b", ModelState.LOW, ModelRole.RECEIVER),
            "c": cls("c", ModelState.HIGH, ModelRole.DONOR),
        },
        {"b": {"signal_warm": False}},
        ts_ms=95_000,
    )
    assert box.get() == {"a": "critical", "b": UNCONFIRMED, "c": "high"}
    box.update({"a": cls("a", ModelState.HEALTHY, ModelRole.NEUTRAL)}, ts_ms=90_000)  # older: ignored
    assert box.get()["a"] == "critical"
    now["ms"] = 130_001
    assert box.get() == {}


# ------------------------------------------------------------------ P2-3 preemption


def _fresh_view(*bindings):
    """A fresh cluster view (review 4 P2-3: the preemption compensation counts
    the donor pods it shows awake and hidden)."""

    class View:
        pass

    view = View()
    view.bindings = tuple(bindings)
    return lambda: view


_HIDDEN_7B = Binding("7b-1", "7b", Slot("n", (0,)), awake=True, hidden=True)


def _blocking_sleep():
    forever = asyncio.Event()

    async def sleep(_seconds):
        await forever.wait()

    return sleep


def test_rescue_for_the_donor_preempts_a_backing_off_commit() -> None:
    async def scenario():
        sm = ScriptedSM(results={"7b-1": [{"ok": False, "error": "HTTP 409: reserved", "retriable": True}]})
        queue = ActionQueue(sm, sleep=_blocking_sleep(), fresh_view=_fresh_view(_HIDDEN_7B))
        runner = asyncio.ensure_future(queue.run(poll_interval_s=0.001))
        queue.submit((_commit(),))
        assert await _until(lambda: queue.preemptible_models() == {"7b", "8b"})
        assert queue.inflight_models() == {"7b", "8b"}
        # the donor is now CRITICAL: rescue wants 2 more replicas of it
        result = queue.submit((ScaleAction("7b", 2, "critical_idle_capacity", "rescue"),))
        assert result.accepted == 1
        assert await _until(lambda: ("end", "7b", "scale", 1) in sm.events)
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
        # backoff cut short, hidden pod unhidden, rescue shrunk by the restored pod
        assert _calls(sm) == [("7b-1", "sleep"), ("7b", "routable", ()), ("7b", "scale", 1)]
        assert queue.stats()["oneshot_preempted_total"] == 1
        assert queue.inflight_models() == set()

    asyncio.run(scenario())


def test_rescue_covered_by_the_unhide_is_dropped() -> None:
    async def scenario():
        sm = ScriptedSM(results={"7b-1": [{"ok": False, "error": "HTTP 409", "retriable": True}]})
        queue = ActionQueue(sm, sleep=_blocking_sleep(), fresh_view=_fresh_view(_HIDDEN_7B))
        runner = asyncio.ensure_future(queue.run(poll_interval_s=0.001))
        queue.submit((_commit(),))
        assert await _until(lambda: "7b" in queue.preemptible_models())
        result = queue.submit((ScaleAction("7b", 1, "critical_idle_capacity", "rescue"),))
        assert result.accepted == 0 and result.dropped == (("7b", "covered_by_preempted_commit"),)
        assert await _until(lambda: ("end", "7b", "routable", ()) in sm.events)
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)

    asyncio.run(scenario())


def test_rescue_for_a_receiver_cancels_its_pending_upscale_only() -> None:
    async def scenario():
        sm = ScriptedSM(
            results={"target:8b": [{"ok": False, "error": "HTTP 409", "retriable": True}] * 3,
                     "target:14b": [{"ok": False, "error": "HTTP 409", "retriable": True}]}
        )
        gate = asyncio.Event()

        async def sleep(_seconds):
            await gate.wait()

        queue = ActionQueue(sm, sleep=sleep)
        runner = asyncio.ensure_future(queue.run(poll_interval_s=0.001))
        queue.submit((_commit(upscales=(("8b", 1, 3), ("14b", 1, 2))),))
        # donor slept; both receivers wait out a backoff
        assert await _until(lambda: queue.preemptible_models() == {"8b", "14b"})
        assert queue.submit((ScaleAction("8b", 1, "critical_idle_capacity", "rescue"),)).accepted == 1
        assert await _until(lambda: ("end", "8b", "scale", 1) in sm.events)
        # Review 4 P3: a receiver preemption does not cut the commit's backoff
        # short (nor count as a preemption): 14b is not retried early ...
        await asyncio.sleep(0.01)
        assert _calls(sm).count(("14b", "target", 2)) == 1
        assert queue.stats()["oneshot_preempted_total"] == 0
        assert queue.stats()["commit_upscale_preempted_total"] == 1
        gate.set()  # ... only when its backoff is over
        assert await _until(lambda: _calls(sm).count(("14b", "target", 2)) == 2)
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
        # the commit's 8b upscale was not re-sent
        calls = _calls(sm)
        assert calls.count(("8b", "target", 3)) == 1
        assert ("8b", "scale", 1) in calls

    asyncio.run(scenario())


def test_planner_plans_a_critical_receiver_whose_commit_is_preemptible() -> None:
    def cls(model, state, role):
        return ModelClassification(
            model_name=model, state=state, role=role, Z_m=0.3, eta_m=None, trs=0.0,
            theta_m=1.0, tau=TauThresholds.from_control(),
        )

    kwargs = dict(
        model_contexts={"8b": {"routable_pods": 2, "assigned_replicas": 2}},
        classifications=[cls("8b", ModelState.CRITICAL, ModelRole.RECEIVER)],
        model_replicas={"8b": 2},
        idle_gpus=1,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4),
        inflight_models={"8b"},
    )
    assert build_plan(**kwargs).actions == []
    [action] = build_plan(**kwargs, preemptible_models={"8b"}).actions
    assert action.model == "8b" and action.delta > 0


# ------------------------------------------------------------------ P3


def test_defrag_is_serialized_against_every_other_action() -> None:
    async def scenario():
        sm = ScriptedSM(gated={"defrag"})
        queue = ActionQueue(sm)
        runner = asyncio.ensure_future(queue.run(poll_interval_s=0.001))
        queue.submit((DefragAction((), "tp_defrag", "rescue"),))
        assert await _until(lambda: ("start", "cluster", "defrag") in sm.events)
        queue.submit((ScaleAction("x", 1, "critical", "rescue", pods=("x-1",)),))
        await asyncio.sleep(0.01)
        assert ("start", "x-1", "wake") not in sm.events
        sm.gates["defrag"].set()
        assert await _until(lambda: ("end", "x-1", "wake") in sm.events)
        # and a defrag waits for running actions too
        sm.gates["y-1"] = asyncio.Event()
        queue.submit((ScaleAction("y", 1, "critical", "rescue", pods=("y-1",)),))
        assert await _until(lambda: ("start", "y-1", "wake") in sm.events)
        queue.submit((DefragAction((), "tp_defrag", "rescue"),))
        await asyncio.sleep(0.01)
        assert [e for e in sm.events if e[:2] == ("start", "cluster")] == [("start", "cluster", "defrag")]
        sm.gates["y-1"].set()
        assert await _until(lambda: len([e for e in sm.events if e[:2] == ("end", "cluster")]) == 2)
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)

    asyncio.run(scenario())


def test_worker_exception_becomes_a_failed_result_and_frees_the_model() -> None:
    async def scenario():
        sm = ScriptedSM(raises={"m-1": RuntimeError("boom")})
        queue = ActionQueue(sm)
        queue.submit((ScaleAction("m", -1, "high", "fairness", pods=("m-1",), sleep_path="urgent"),))
        [result] = await queue.drain_once()
        assert not result.ok and result.error == "dispatch_exception: RuntimeError: boom"
        assert queue.inflight_models() == set()
        assert queue.stats()["dispatch_exceptions_total"] == 1

    asyncio.run(scenario())


def test_transfer_exception_is_a_failed_result_for_both_models() -> None:
    async def scenario():
        sm = ScriptedSM(raises={"transfer:7b->8b": ConnectionResetError("reset")})
        queue = ActionQueue(sm)
        queue.submit((TransferIntent("7b", "8b", 1, "critical_donor_immediate", "rescue"),))
        results = await queue.drain_once()
        by_model = {r.model: r for r in results}
        assert by_model["7b"].error.startswith("dispatch_exception: ConnectionResetError")
        assert by_model["8b"].error.startswith("dispatch_exception: ConnectionResetError")
        # whether the SM did anything is unknown: both models wait for a newer view
        assert set(queue.view_changes()) == {"7b", "8b"}
        assert queue.last_actions() == {}
        assert queue.inflight_models() == set()

    asyncio.run(scenario())


def test_sm_client_reports_malformed_state_counts_as_a_failed_result() -> None:
    class BadState:
        async def request(self, method, url, *, json=None, timeout_s):
            return {"models": {"m": {"awake": "two", "bound": 4}}}

    async def scenario():
        client = ServiceManagerClient("http://sm", transport=BadState())
        result = await client.scale_model("m", 1)
        assert result["ok"] is False and result["retriable"] is False
        assert "malformed /v2/state counts" in result["error"]
        result = await client.scale_model_to("m", 3)
        assert result["ok"] is False
        assert (await client.model_awake("m"))["ok"] is False

    asyncio.run(scenario())


def test_one_shot_retry_attempts_unchanged_for_binding_power() -> None:
    """Regression: a plain SafeScale binding-power action keeps its retries."""

    async def scenario():
        sm = ScriptedSM(results={"m-1": [{"ok": False, "error": "HTTP 409", "retriable": True}]})
        queue = ActionQueue(sm, retry=RetryPolicy(max_attempts=3), sleep=lambda _s: asyncio.sleep(0))
        queue.submit((ScaleAction("m", -1, "formal_commit_gate_passed", "safescale", pods=("m-1",), sleep_path="safescale_commit"),))
        [result] = await queue.drain_once()
        assert result.ok and result.attempts == 2

    asyncio.run(scenario())

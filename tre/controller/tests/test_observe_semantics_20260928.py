"""Observe = record only (user decision 2026-09-28), controller side.

* a SafeScale commit re-checks the mode before every capacity-changing step: a
  donor already slept is recorded, its receiver is not woken;
* a relay (TransferIntent, 2026-10-02) is ONE SM call: the mode is checked right
  before it; once sent, the service-manager completes it (observe entered meanwhile
  is recorded, nothing is undone);
* entering observe rolls every open SafeScale probe back: held one-shot
  actions are dropped, the single action taken is the unhide of the probe
  pods, the probe resolves as a rollback ``observe_entered``;
* a probe start's hide is re-checked right before its SM call (B8 gap).
"""
from __future__ import annotations

import asyncio
from dataclasses import replace

from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.safescale_task import (
    rollback_probes_for_observe,
    run_safescale_observation_tick,
)
from tre_controller.planning.planner import HideAction, ScaleAction, TransferIntent
from tre_controller.store.state_store import ControllerStateStore

from test_action_queue_review3 import ScriptedSM, _calls, _commit, _until
from test_safescale_commit import FakeRedis, _machine, _metrics, _probe_records, _registry, _start_and_prime


def _transfer():
    return TransferIntent("7b", "8b", 1, "critical_donor_immediate", "rescue")


# ------------------------------------------------------------ mid-transfer flip
def test_observe_entered_while_the_sm_runs_a_relay_is_recorded_not_undone():
    """2026-10-02: the relay is one SM call; the SM finishes what it started (donor
    asleep AND receiver awake) - the controller records it, sends nothing more."""
    async def scenario():
        mode = {"observe": False}
        sm = ScriptedSM(gated={"transfer:7b->8b"})
        queue = ActionQueue(sm, is_observe=lambda: mode["observe"])
        queue.submit((_transfer(),))
        drain = asyncio.ensure_future(queue.drain_once())
        assert await _until(lambda: _calls(sm) == [("7b->8b", "transfer", 1)])
        mode["observe"] = True  # the console switches while the SM runs the relay
        sm.gates["transfer:7b->8b"].set()
        results = await drain
        assert _calls(sm) == [("7b->8b", "transfer", 1)]  # nothing else is sent
        donor, receiver = results
        assert (donor.model, donor.taken) == ("7b", 1)
        assert (receiver.model, receiver.ok, receiver.done) == ("8b", True, 1)
        assert queue.stats()["observe_transfer_completed_total"] == 1
        assert queue.inflight_models() == set()

    asyncio.run(scenario())


def test_a_transfer_dispatched_in_observe_does_nothing():
    async def scenario():
        sm = ScriptedSM()
        queue = ActionQueue(sm, is_observe=lambda: True)
        queue.submit((_transfer(),))
        results = await queue.drain_once()
        assert _calls(sm) == []
        assert {r.error for r in results} == {"observe_skipped"}

    asyncio.run(scenario())


# ------------------------------------------------------------ mid-commit flip
def test_observe_entered_after_the_commit_donor_slept_drops_the_receiver_wakes():
    async def scenario():
        mode = {"observe": False}
        done = []
        sm = ScriptedSM(gated={"7b-1"})
        queue = ActionQueue(sm, is_observe=lambda: mode["observe"], on_oneshot_done=lambda *a: done.append(a))
        queue.submit((_commit(upscales=(("8b", 1, 3), ("14b", 1, 2))),))
        drain = asyncio.ensure_future(queue.drain_once())
        assert await _until(lambda: _calls(sm) == [("7b-1", "sleep")])
        mode["observe"] = True
        sm.gates["7b-1"].set()
        results = await drain
        assert _calls(sm) == [("7b-1", "sleep")]  # no receiver target sent
        dropped = sorted(r.model for r in results if r.error and r.error.startswith("observe_entered"))
        assert dropped == ["14b", "8b"]
        [(request_id, status, reason)] = done
        assert (request_id, status) == ("7b-0", "commit")  # the donor did sleep
        assert "observe_entered" in reason
        assert queue.stats()["observe_commit_stopped_total"] == 1

    asyncio.run(scenario())


def test_observe_entered_before_the_commit_donor_slept_unhides_it_instead():
    async def scenario():
        done = []
        sm = ScriptedSM()
        queue = ActionQueue(sm, is_observe=lambda: True, on_oneshot_done=lambda *a: done.append(a))
        queue.submit((_commit(),))
        results = await queue.drain_once()
        assert _calls(sm) == [("7b", "routable", ())]  # no sleep, no wake
        assert done == [("7b-0", "rollback", "observe_entered")]
        assert [r.error for r in results if r.model == "8b"] == ["observe_entered: receiver wake not issued"]

    asyncio.run(scenario())


# ----------------------------------------------- active -> observe with probes
def test_entering_observe_rolls_back_probing_and_committing_probes():
    redis = FakeRedis()
    machine = _machine(ControllerStateStore(redis))
    _start_and_prime(machine)  # probe "donor" (pod-a), probing
    # a second probe that already reached its commit (committing, action queued)
    machine.start_probe(model="receiver", pods=("recv-1",), now_ms=0)
    mode = {"observe": False}
    sm = ScriptedSM()
    queue = ActionQueue(
        sm,
        is_observe=lambda: mode["observe"],
        on_oneshot_done=lambda request_id, status, reason: machine.resolve_request(
            request_id, status=status, reason=reason, now_ms=5_000
        ),
    )
    commit = _commit(donor="receiver", pods=("recv-1",), upscales=(("other", 1, 3),))
    commit = replace(commit, request_id=machine.active_probe("receiver").request_id)
    assert queue.submit((commit,)).accepted == 1
    machine.mark_committing("receiver", status="commit", reason="formal_commit_gate_passed", now_ms=900)

    mode["observe"] = True  # active -> observe
    result = run_safescale_observation_tick(
        _metrics(2_000), queue=queue, registry=_registry(), safescale=machine, observe_mode=True
    )
    assert f"safescale_observe_rollback:donor:{machine.active_probe('donor').request_id}" in result.events
    asyncio.run(queue.drain_once())

    # only unhides reached the SM: no donor sleep, no receiver wake
    assert sorted(_calls(sm)) == [("donor", "routable", ()), ("receiver", "routable", ())]
    assert machine.busy_models() == set()
    records = {r["model"]: r for r in _probe_records(redis).values()}
    assert {m: (r["status"], r["resolution"], r["terminal_reason"]) for m, r in records.items()} == {
        "donor": ("resolved", "rollback", "observe_entered"),
        "receiver": ("resolved", "rollback", "observe_entered"),
    }


def test_the_observe_rollback_drops_a_probes_queued_one_shot_before_the_unhide():
    machine = _machine(ControllerStateStore(FakeRedis()))
    _start_and_prime(machine)
    request_id = machine.active_probe("donor").request_id
    queue = ActionQueue(ScriptedSM(), is_observe=lambda: True)
    stale = _commit(donor="donor", pods=("pod-a",), upscales=())
    queue.submit((replace(stale, request_id=request_id),))  # held one-shot of the probe
    events: list[str] = []
    [unhide] = rollback_probes_for_observe(queue, machine, now_ms=1_000, events=events)
    assert (unhide.model, unhide.pods, unhide.reason) == ("donor", ("pod-a",), "observe_entered")
    assert [type(item.action).__name__ for item in queue.pending_actions()] == ["UnhideAction"]
    assert queue.stats()["oneshot_cancelled_total"] == 1
    assert machine.committing_probes()[0].resolution_reason == "observe_entered"


def test_the_safescale_loop_observes_no_probe_in_observe_mode():
    machine = _machine(ControllerStateStore(FakeRedis()))
    _start_and_prime(machine)
    queue = ActionQueue(ScriptedSM(), is_observe=lambda: True)
    result = run_safescale_observation_tick(
        _metrics(2_000), queue=queue, registry=_registry(), safescale=machine, observe_mode=True
    )
    # rolled back, not judged: no commit / gate decision was taken
    assert not any(e.startswith("safescale_formal_commit") for e in result.events)
    assert machine.active_probes() == ()


# ------------------------------------------------------------ hide re-check
def test_a_hide_is_rechecked_right_before_its_sm_call():
    async def scenario():
        reads = iter([False, True])  # active at the dispatch entry, observe at the hide call
        sm = ScriptedSM()
        queue = ActionQueue(sm, is_observe=lambda: False, is_observe_fresh=lambda: next(reads))
        queue.submit((HideAction("7b", ("7b-1",), "probe_started", "fairness"),))
        [result] = await queue.drain_once()
        assert _calls(sm) == []
        assert (result.action_kind, result.error) == ("hide", "observe_skipped")
        assert queue.stats()["observe_hide_skipped_total"] == 1

    asyncio.run(scenario())


def test_a_stale_cached_active_mode_does_not_send_the_hide():
    """The 0.1 s dispatch poll reads the cached mode (TTL 1 s); the fresh read
    right before the SM call sees the switch (B8's ~1 s window)."""
    from tre_controller.mode import ObserveModeGate

    class Redis:
        value = b"active"

        def get(self, _key):
            return self.value

    async def scenario():
        clock = {"t": 0.0}
        redis = Redis()
        gate = ObserveModeGate(redis, ttl_s=1.0, clock=lambda: clock["t"])
        assert gate.is_observe() is False  # cached: active
        redis.value = b"observe"  # the switch, inside the TTL
        sm = ScriptedSM()
        queue = ActionQueue(sm, is_observe=gate.is_observe, is_observe_fresh=gate.is_observe_fresh)
        queue.submit((HideAction("7b", ("7b-1",), "probe_started", "fairness"),))
        await queue.drain_once()
        assert _calls(sm) == []

    asyncio.run(scenario())


# ------------------------------------------------------------ wiring
def test_the_app_wires_the_fresh_mode_read_and_the_safescale_observe_gate():
    from test_controller_app import REGISTRY_PATH, EmptyRedis
    from tre_controller.app import build_controller_task_specs, create_controller_dependencies
    from tre_controller.config import ControllerConfig

    cfg = ControllerConfig.from_env({"TRE_REGISTRY_PATH": str(REGISTRY_PATH)})
    deps = create_controller_dependencies(cfg, redis_client=EmptyRedis())
    assert deps.queue._is_observe_fresh == deps.observe_gate.is_observe_fresh
    # EmptyRedis has no mode key: the controller starts in observe (fail-closed)
    assert deps.observe_gate.is_observe() is True
    names = [spec.name for spec in build_controller_task_specs(deps, cfg)]
    assert "safescale" in names

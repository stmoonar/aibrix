"""B8: observe mode must not start SafeScale probes; a commit held (observe) or
recovered (restart) is aged before its first dispatch; a probe whose pods are all
gone from a fresh cluster view is resolved without any SM call, in any mode."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

from tre_common.metrics_schema import MetricsSnapshot
from tre_controller.config import ControllerConfig, SafeScaleConfig
from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.rescue_task import run_rescue_tick
from tre_controller.loops.safescale_task import run_safescale_observation_tick
from tre_controller.planning.planner import HideAction, ScaleAction
from tre_controller.planning.safescale import SafeScaleStateMachine
from tre_controller.store.state_store import ControllerStateStore
from tre_sm.allocator.slots import Binding, Slot

from test_action_queue_review3 import ScriptedSM, _calls, _commit
from test_action_queue_review4 import RecordingSM
from test_loop_ticks import (
    FakeQueue,
    _hidden_probe_view,
    _metrics as _tick_metrics,
    _metrics_with_pods,
    _registry as _tick_registry,
    _registry_with_models,
)
from test_safescale_commit import FakeRedis, _machine, _metrics, _probe_records, _registry, _start_and_prime

MAX_AGE_MS = 120_000
DECIDED_MS = 1_000  # the observation tick that decides the commit runs at ts 1000


def _view(*bindings):
    return SimpleNamespace(bindings=tuple(bindings))


POD_A_HIDDEN = Binding("pod-a", "donor", Slot("n", (0,)), awake=True, hidden=True)
POD_A_ASLEEP = Binding("pod-a", "donor", Slot("n", (0,)), awake=False, hidden=True)
REPLACEMENT = Binding("pod-new", "donor", Slot("n", (0,)), awake=True, hidden=False)


def _probe_snapshot():
    return MetricsSnapshot(
        ts_ms=10_000,
        stale=False,
        models={
            "critical": _metrics_with_pods(
                "critical", generation=50.0, waiting=10.0, running=1.0, pods=("critical-a", "critical-b")
            ),
            "donor": _metrics_with_pods(
                "donor", generation=100.0, waiting=0.0, running=1.0, pods=("donor-a", "donor-b")
            ),
        },
    )


# ------------------------------------------------------------ 1. no probe in observe
def test_observe_mode_starts_no_probe_and_submits_no_hide():
    queue = FakeQueue()
    safescale = SafeScaleStateMachine(config=SafeScaleConfig(default_window_ms=60_000.0))
    registry = _registry_with_models("critical", "donor")

    result = run_rescue_tick(
        _probe_snapshot(), queue=queue, registry=registry, safescale=safescale, observe_mode=True
    )

    assert safescale.active_probe("donor") is None and safescale.busy_models() == set()
    assert not any(isinstance(action, HideAction) for batch in queue.submitted for action in batch)
    assert not any(isinstance(action, HideAction) for action in result.actions)
    assert "safescale_probe_skipped:donor:observe_mode" in result.events
    assert not any(event.startswith("safescale_probe_started") for event in result.events)
    # The same tick in active mode starts the probe (control).
    active = run_rescue_tick(_probe_snapshot(), queue=FakeQueue(), registry=registry, safescale=safescale)
    assert "safescale_probe_started:donor" in active.events


def test_observe_mode_does_not_preempt_a_running_probe():
    registry = _tick_registry()
    safescale = SafeScaleStateMachine(config=SafeScaleConfig(default_window_ms=60_000.0))
    safescale.start_probe(model="critical", pods=("critical-1",), now_ms=0)
    snapshot = MetricsSnapshot(
        ts_ms=1_000,
        stale=False,
        models={
            "critical": _tick_metrics("critical", generation=50.0, waiting=10.0, running=1.0, assigned=4, routable=1)
        },
    )
    view = _hidden_probe_view(registry, awake=2, hidden=1)

    result = run_rescue_tick(
        snapshot, queue=FakeQueue(), registry=registry, cluster_view=view, safescale=safescale, observe_mode=True
    )

    assert not any(event.startswith("safescale_probe_preempted") for event in result.events)
    assert safescale.active_probe("critical").preempt_reason is None
    # the planned scale-up is still published (the queue drops it in observe mode)
    assert any(isinstance(a, ScaleAction) and a.delta > 0 for a in result.actions)


def test_the_rescue_loop_reads_the_mode_gate_every_tick():
    from tre_controller.loops.metrics_task import SnapshotBox
    from tre_controller.loops.rescue_task import rescue_task

    class Stop(Exception):
        pass

    async def stop(_seconds):
        raise Stop

    queue = FakeQueue()
    safescale = SafeScaleStateMachine(config=SafeScaleConfig(default_window_ms=60_000.0))
    reads = []

    def is_observe():
        reads.append(1)
        return True

    try:
        asyncio.run(
            rescue_task(
                SnapshotBox(_probe_snapshot()),
                queue=queue,
                registry=_registry_with_models("critical", "donor"),
                cfg=SimpleNamespace(rescue_interval_s=5.0),
                sleep=stop,
                safescale=safescale,
                is_observe=is_observe,
            )
        )
    except Stop:
        pass
    assert reads == [1]
    assert safescale.busy_models() == set()
    assert not any(isinstance(action, HideAction) for batch in queue.submitted for action in batch)


# ------------------------------------------------ 2. commit held, then aged on resume
def _held_commit(*, fresh_view, clock):
    """A probe started in active mode reaches its commit while the controller is
    in observe mode: the commit is handed to the queue and held there."""
    redis = FakeRedis()
    machine = _machine(ControllerStateStore(redis))
    _start_and_prime(machine)
    mode = {"observe": True}
    sm = RecordingSM()
    queue = ActionQueue(
        sm,
        is_observe=lambda: mode["observe"],
        now_ms=lambda: clock["now"],
        fresh_view=fresh_view,
        commit_max_age_ms=MAX_AGE_MS,
        on_oneshot_done=lambda request_id, status, reason: machine.resolve_request(
            request_id, status=status, reason=reason, now_ms=clock["now"]
        ),
    )
    result = run_safescale_observation_tick(_metrics(DECIDED_MS), queue=queue, registry=_registry(), safescale=machine)
    assert result.submitted == 1
    return redis, machine, queue, sm, mode


def test_a_commit_decided_in_observe_mode_is_held_not_dispatched():
    clock = {"now": DECIDED_MS}
    redis, machine, queue, sm, _mode = _held_commit(fresh_view=lambda: _view(POD_A_HIDDEN), clock=clock)
    [record] = _probe_records(redis).values()
    assert (record["status"], record["resolution"], record["committing_ts"]) == ("committing", "commit", 1.0)
    assert machine.committing_probes()[0].committing_ms == DECIDED_MS
    clock["now"] = DECIDED_MS + 10 * MAX_AGE_MS  # however long it waits
    asyncio.run(queue.drain_once())
    assert sm.calls == []
    assert queue.has_request("donor-0") and machine.busy_models() == {"donor"}


def test_back_to_active_after_the_max_age_the_commit_becomes_the_unhide():
    clock = {"now": DECIDED_MS}
    redis, machine, queue, sm, mode = _held_commit(fresh_view=lambda: _view(POD_A_HIDDEN), clock=clock)
    asyncio.run(queue.drain_once())
    clock["now"] = DECIDED_MS + MAX_AGE_MS + 1
    mode["observe"] = False
    asyncio.run(queue.drain_once())
    # never slept: the hidden donor pod gets its routing back; the upscale is dropped
    assert sm.calls == [("donor", "routable", ())]
    assert queue.stats()["commit_evidence_stale_total"] == 1
    assert machine.busy_models() == set()
    [record] = _probe_records(redis).values()
    assert (record["status"], record["resolution"], record["terminal_reason"]) == (
        "resolved", "rollback", "commit_evidence_stale",
    )


def test_a_stale_commit_unhides_nothing_a_fresh_view_does_not_show_awake_and_hidden():
    clock = {"now": DECIDED_MS}
    redis, machine, queue, sm, mode = _held_commit(fresh_view=lambda: _view(POD_A_ASLEEP), clock=clock)
    clock["now"] = DECIDED_MS + MAX_AGE_MS + 1
    mode["observe"] = False
    asyncio.run(queue.drain_once())
    assert sm.calls == []
    [record] = _probe_records(redis).values()
    assert record["resolution"] == "rollback"
    assert record["terminal_reason"].startswith("commit_evidence_stale: decided 120s ago > 120s")
    assert record["terminal_reason"].endswith("no donor pod to unhide")


def test_back_to_active_within_the_max_age_the_commit_proceeds():
    clock = {"now": DECIDED_MS}
    redis, machine, queue, sm, mode = _held_commit(fresh_view=lambda: _view(POD_A_HIDDEN), clock=clock)
    asyncio.run(queue.drain_once())
    clock["now"] = DECIDED_MS + MAX_AGE_MS  # exactly the max age is still fresh
    mode["observe"] = False
    asyncio.run(queue.drain_once())
    assert sm.calls == [("pod-a", False), ("receiver", "target", 1)]
    assert queue.stats()["commit_evidence_stale_total"] == 0
    [record] = _probe_records(redis).values()
    assert (record["status"], record["resolution"]) == ("resolved", "commit")


def test_a_retry_of_a_started_commit_is_not_aged():
    async def scenario():
        clock = {"now": 1_000}
        sm = ScriptedSM(results={"7b-1": [{"ok": False, "error": "HTTP 409: busy", "retriable": True}]})

        async def long_backoff(_seconds):
            clock["now"] += 10 * MAX_AGE_MS

        queue = ActionQueue(sm, sleep=long_backoff, now_ms=lambda: clock["now"], commit_max_age_ms=MAX_AGE_MS)
        queue.submit((replace(_commit(), decided_ms=0),))
        await queue.drain_once()
        assert _calls(sm) == [("7b-1", "sleep"), ("7b-1", "sleep"), ("8b", "target", 3)]
        assert queue.stats()["commit_evidence_stale_total"] == 0

    asyncio.run(scenario())


def test_no_max_age_without_the_knob_or_a_decision_time():
    async def scenario():
        sm = ScriptedSM()
        queue = ActionQueue(sm, now_ms=lambda: 10**12)  # knob off (default)
        queue.submit((replace(_commit(), decided_ms=0),))
        await queue.drain_once()
        sm2 = ScriptedSM()
        queue2 = ActionQueue(sm2, now_ms=lambda: 10**12, commit_max_age_ms=MAX_AGE_MS)
        queue2.submit((_commit(),))  # decided_ms unknown
        await queue2.drain_once()
        assert _calls(sm)[0] == _calls(sm2)[0] == ("7b-1", "sleep")

    asyncio.run(scenario())


# ------------------------------------------------------------ 3. probe pods gone
def test_a_committing_probe_whose_pods_are_gone_is_resolved_in_observe_without_sm_calls():
    clock = {"now": DECIDED_MS}
    redis, machine, queue, sm, mode = _held_commit(fresh_view=lambda: _view(REPLACEMENT), clock=clock)
    result = run_safescale_observation_tick(
        _metrics(2_000), queue=queue, registry=_registry(), safescale=machine,
        fresh_cluster_view=_view(REPLACEMENT), recovery_needs_fresh_view=True,
    )
    assert "safescale_probe_pods_gone:donor:donor-0" in result.events
    assert machine.busy_models() == set()
    assert not queue.has_request("donor-0")  # the held commit was dropped, not dispatched
    [record] = _probe_records(redis).values()
    assert (record["status"], record["resolution"], record["terminal_reason"]) == (
        "resolved", "rollback", "probe_pods_gone",
    )
    mode["observe"] = False
    asyncio.run(queue.drain_once())
    assert sm.calls == []
    assert queue.inflight_models() == set()


def test_a_probing_probe_whose_pods_are_gone_is_resolved():
    redis = FakeRedis()
    machine = _machine(ControllerStateStore(redis))
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=0)
    result = run_safescale_observation_tick(
        _metrics(500), queue=FakeQueue(), registry=_registry(), safescale=machine,
        fresh_cluster_view=_view(REPLACEMENT),
    )
    assert "safescale_probe_pods_gone:donor:donor-0" in result.events
    assert machine.busy_models() == set()


def test_gone_pods_are_never_judged_on_a_missing_stale_or_partial_view():
    redis = FakeRedis()
    machine = _machine(ControllerStateStore(redis))
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=0)
    other_model = Binding("x-0", "other", Slot("n", (1,)), awake=True)
    for view in (None, _view(), _view(other_model)):
        run_safescale_observation_tick(
            _metrics(500), queue=FakeQueue(), registry=_registry(), safescale=machine, fresh_cluster_view=view,
        )
        assert machine.busy_models() == {"donor"}
    # one pod of the probe still listed: not gone
    run_safescale_observation_tick(
        _metrics(500), queue=FakeQueue(), registry=_registry(), safescale=machine,
        fresh_cluster_view=_view(REPLACEMENT, POD_A_HIDDEN),
    )
    assert machine.busy_models() == {"donor"}


def test_a_probe_whose_one_shot_is_running_is_left_to_finish_it():
    class RunningQueue(FakeQueue):
        def cancel_request(self, request_id):
            return False

    machine = _machine(ControllerStateStore(FakeRedis()))
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=0)
    result = run_safescale_observation_tick(
        _metrics(500), queue=RunningQueue(), registry=_registry(), safescale=machine,
        fresh_cluster_view=_view(REPLACEMENT),
    )
    assert "safescale_probe_pods_gone_deferred:donor:donor-0" in result.events
    assert machine.busy_models() == {"donor"}


def test_a_restored_probe_whose_pods_are_gone_is_resolved_before_any_recovery():
    redis = FakeRedis()
    store = ControllerStateStore(redis)
    old = _machine(store)
    _start_and_prime(old)
    # the old controller decided the commit (held in observe mode), then died
    run_safescale_observation_tick(
        _metrics(DECIDED_MS), queue=ActionQueue(RecordingSM(), is_observe=lambda: True),
        registry=_registry(), safescale=old,
    )

    restarted = _machine(store)
    assert restarted.restore() == 1
    assert restarted.committing_probes()[0].committing_ms == DECIDED_MS
    sm = RecordingSM()
    queue = ActionQueue(
        sm,
        commit_max_age_ms=MAX_AGE_MS,
        on_oneshot_done=lambda request_id, status, reason: restarted.resolve_request(
            request_id, status=status, reason=reason, now_ms=5_000
        ),
    )
    # no fresh cluster view yet (the first SM refresh has not landed): nothing is re-submitted
    waiting = run_safescale_observation_tick(
        _metrics(2_000), queue=queue, registry=_registry(), safescale=restarted,
        fresh_cluster_view=None, recovery_needs_fresh_view=True,
    )
    assert "safescale_recovery_waits_for_fresh_view:donor" in waiting.events
    assert waiting.submitted == 0 and not queue.has_request("donor-0")
    # the first fresh view no longer lists the probe pod (replaced by a rolling restart)
    resolved = run_safescale_observation_tick(
        _metrics(3_000), queue=queue, registry=_registry(), safescale=restarted,
        fresh_cluster_view=_view(REPLACEMENT), recovery_needs_fresh_view=True,
    )
    assert "safescale_probe_pods_gone:donor:donor-0" in resolved.events
    assert resolved.submitted == 0
    asyncio.run(queue.drain_once())
    assert sm.calls == []
    assert restarted.busy_models() == set()
    [record] = _probe_records(redis).values()
    assert (record["status"], record["resolution"], record["terminal_reason"]) == (
        "resolved", "rollback", "probe_pods_gone",
    )


def test_a_restored_old_commit_whose_pod_still_exists_is_aged_into_the_unhide():
    redis = FakeRedis()
    store = ControllerStateStore(redis)
    old = _machine(store)
    _start_and_prime(old)
    run_safescale_observation_tick(
        _metrics(DECIDED_MS), queue=ActionQueue(RecordingSM(), is_observe=lambda: True),
        registry=_registry(), safescale=old,
    )
    restarted = _machine(store)
    restarted.restore()
    sm = RecordingSM()
    queue = ActionQueue(
        sm,
        now_ms=lambda: DECIDED_MS + 3_600_000,  # restarted an hour later
        fresh_view=lambda: _view(POD_A_HIDDEN),
        commit_max_age_ms=MAX_AGE_MS,
        on_oneshot_done=lambda request_id, status, reason: restarted.resolve_request(
            request_id, status=status, reason=reason, now_ms=5_000
        ),
    )
    result = run_safescale_observation_tick(
        _metrics(2_000), queue=queue, registry=_registry(), safescale=restarted,
        fresh_cluster_view=_view(POD_A_HIDDEN), recovery_needs_fresh_view=True,
    )
    assert "safescale_committing_recovered:donor:commit" in result.events
    asyncio.run(queue.drain_once())
    assert sm.calls == [("donor", "routable", ())]  # never slept on hour-old evidence
    [record] = _probe_records(redis).values()
    assert (record["resolution"], record["terminal_reason"]) == ("rollback", "commit_evidence_stale")


# ------------------------------------------------------------ config / wiring
def test_commit_max_age_default_and_env():
    assert ControllerConfig.from_env({}).safescale.commit_max_age_ms == SafeScaleConfig.commit_max_age_ms == 120_000.0
    cfg = ControllerConfig.from_env({"TRE_SAFESCALE_COMMIT_MAX_AGE_MS": "60000"})
    assert cfg.safescale.commit_max_age_ms == 60_000.0
    assert ControllerConfig.from_env({"TRE_SAFESCALE_COMMIT_MAX_AGE_MS": "0"}).safescale.commit_max_age_ms == 0.0


def test_the_app_wires_the_mode_gate_and_the_max_age():
    from test_controller_app import REGISTRY_PATH, EmptyRedis
    from tre_controller.app import create_controller_dependencies

    cfg = ControllerConfig.from_env({"TRE_REGISTRY_PATH": str(REGISTRY_PATH)})
    deps = create_controller_dependencies(cfg, redis_client=EmptyRedis())
    assert deps.observe_gate is not None
    assert deps.queue._is_observe == deps.observe_gate.is_observe
    assert deps.queue._commit_max_age_ms == cfg.safescale.commit_max_age_ms

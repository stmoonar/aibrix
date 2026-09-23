"""Review F4: per-model action cooldown."""
from __future__ import annotations

import asyncio

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_controller.config import ControllerConfig
from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.metrics_task import SnapshotBox
from tre_controller.loops.rescue_task import rescue_task, run_rescue_tick
from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, TauThresholds
from tre_controller.planning.planner import PlanConfig, ScaleAction, build_plan

from test_loop_ticks import _registry


class _Clock:
    def __init__(self, now_ms: int) -> None:
        self.now = now_ms

    def __call__(self) -> int:
        return self.now


class _OkClient:
    def __init__(self) -> None:
        self.calls = []

    async def scale_model(self, model, delta):
        self.calls.append((model, delta))
        return {"ok": True}

    async def set_routable(self, model, hidden_pods):
        return {"ok": True}

    async def set_binding_power(self, serve_id, *, awake):
        return {"ok": True}

    async def defrag(self, migrations):
        return {"ok": True}


def _critical_snapshot(window_start_ms: int) -> MetricsSnapshot:
    return MetricsSnapshot(
        ts_ms=window_start_ms + 60_000,
        stale=False,
        models={
            "critical": ModelWindowMetrics(
                model="critical",
                window_start_ms=window_start_ms,
                window_end_ms=window_start_ms + 60_000,
                prompt_tokens=0.0,
                generation_tokens=50.0,
                avg_waiting=10.0,
                avg_running=1.0,
                avg_swapping=0.0,
                kv_cache_hit_rate=0.0,
                ttft_p95_ms=100.0,
                tpot_p95_ms=10.0,
                e2e_p95_ms=1000.0,
                routable_pods=2,
                assigned_replicas=2,
                per_pod={},
            )
        },
    )


def _wakes(queue_client: _OkClient) -> int:
    return sum(1 for model, delta in queue_client.calls if model == "critical" and delta > 0)


def test_repeated_critical_ticks_wake_once_until_window_rolls_past_the_wake() -> None:
    client = _OkClient()
    clock = _Clock(65_000)
    queue = ActionQueue(client, now_ms=clock)
    registry = _registry()

    first = run_rescue_tick(_critical_snapshot(5_000), queue=queue, registry=registry, action_cooldown=True)
    asyncio.run(queue.drain_once())  # wake completes at t=65_000
    assert first.submitted == 1 and _wakes(client) == 1
    assert queue.last_actions() == {"critical": (65_000, "up")}

    # Still CRITICAL, but every window that starts before t=65_000 predates the wake.
    for start in (10_000, 30_000, 64_999):
        held = run_rescue_tick(_critical_snapshot(start), queue=queue, registry=registry, action_cooldown=True)
        asyncio.run(queue.drain_once())
        assert held.submitted == 0
        assert "cooldown_hold:critical" in held.events
    assert _wakes(client) == 1

    fresh = run_rescue_tick(_critical_snapshot(65_000), queue=queue, registry=registry, action_cooldown=True)
    asyncio.run(queue.drain_once())
    assert fresh.submitted == 1 and _wakes(client) == 2


def test_cooldown_disabled_keeps_legacy_repeat_behaviour() -> None:
    client = _OkClient()
    queue = ActionQueue(client, now_ms=_Clock(65_000))
    registry = _registry()

    for start in (5_000, 10_000):
        result = run_rescue_tick(_critical_snapshot(start), queue=queue, registry=registry)
        asyncio.run(queue.drain_once())
        assert result.submitted == 1
        assert not any(event.startswith("cooldown_hold") for event in result.events)
    assert _wakes(client) == 2


def test_failed_dispatch_does_not_start_a_cooldown() -> None:
    class _FailClient(_OkClient):
        async def scale_model(self, model, delta):
            return {"ok": False, "error": "HTTP 409: WakeConflict"}

    queue = ActionQueue(_FailClient(), now_ms=_Clock(65_000))
    run_rescue_tick(_critical_snapshot(5_000), queue=queue, registry=_registry(), action_cooldown=True)
    asyncio.run(queue.drain_once())

    assert queue.last_actions() == {}


def _cls(model, state, role, z, tier=None):
    return ModelClassification(
        model_name=model, state=state, role=role, Z_m=z, eta_m=None, trs=0.0,
        theta_m=1.0, tau=TauThresholds.from_control(), donor_tier=tier,
    )


def _plan(classifications, cooldowns, *, idle_gpus=0, rescue_due=True):
    contexts = {item.model_name: {"routable_pods": 3, "assigned_replicas": 3} for item in classifications}
    return build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={item.model_name: 3 for item in classifications},
        idle_gpus=idle_gpus,
        # HIGH proactive probes (default on since A2) are not what these cases exercise.
        cfg=PlanConfig(
            min_replicas_per_model=1,
            max_replicas_per_model=4,
            rescue_due=rescue_due,
            suppress_hot_proactive_probe=True,
        ),
        cooldowns=cooldowns,
    )


def _deltas(plan):
    out = {}
    for action in plan.actions:
        if isinstance(action, ScaleAction):
            out[action.model] = out.get(action.model, 0) + action.delta
    return out


def test_critical_scale_up_allowed_during_scale_down_cooldown_but_not_after_scale_up() -> None:
    receiver = [_cls("r", ModelState.CRITICAL, ModelRole.RECEIVER, 0.4)]

    assert _deltas(_plan(receiver, {"r": "down"}, idle_gpus=1)) == {"r": 1}
    held = _plan(receiver, {"r": "up"}, idle_gpus=1)
    assert _deltas(held) == {}
    assert held.events == ["cooldown_hold:r"]


def test_low_receiver_scale_up_blocked_during_scale_down_cooldown() -> None:
    receiver = [_cls("r", ModelState.LOW, ModelRole.RECEIVER, 0.9)]

    held = _plan(receiver, {"r": "down"}, idle_gpus=1, rescue_due=False)
    assert _deltas(held) == {}
    assert "cooldown_hold:r" in held.events


def test_donor_cooldown_skips_that_donor_without_blocking_the_pair_elsewhere() -> None:
    classifications = [
        _cls("r", ModelState.CRITICAL, ModelRole.RECEIVER, 0.4),
        _cls("d1", ModelState.HIGH, ModelRole.DONOR, 2.0, "surplus"),
        _cls("d2", ModelState.HIGH, ModelRole.DONOR, 1.9, "surplus"),
    ]

    baseline = _deltas(_plan(classifications, {}))
    assert baseline["r"] == 1 and baseline.get("d1", 0) + baseline.get("d2", 0) == -1
    first = "d1" if baseline.get("d1") else "d2"
    other = "d2" if first == "d1" else "d1"

    # The chosen donor is in an *up* cooldown (a scale-down right after a scale-up is
    # held); the receiver's own state is unaffected and the other donor is used.
    plan = _plan(classifications, {first: "up"})
    assert _deltas(plan) == {"r": 1, other: -1}
    assert f"cooldown_hold:{first}" in plan.events

    # The receiver's scale-down cooldown (CRITICAL safety) never blocks the donor side.
    assert _deltas(_plan(classifications, {"r": "down"})) == baseline


def test_idle_proactive_shrink_held_during_cooldown() -> None:
    idle = [_cls("i", ModelState.IDLE, ModelRole.DONOR, 10.0, "idle")]

    assert _deltas(_plan(idle, {})) == {"i": -1}
    assert _deltas(_plan(idle, {"i": "down"})) == {}


def test_action_cooldown_env_flag_defaults_on_and_can_be_disabled() -> None:
    assert ControllerConfig.from_env({}).action_cooldown is True
    assert ControllerConfig.from_env({"TRE_ACTION_COOLDOWN": "0"}).action_cooldown is False


class _StopLoop(Exception):
    pass


async def _stop_sleep(_seconds):
    raise _StopLoop()


def test_rescue_task_reads_action_cooldown_from_config() -> None:
    def run(flag: bool) -> tuple[str, ...]:
        queue = ActionQueue(_OkClient(), now_ms=_Clock(65_000))
        queue._last_done["critical"] = (65_000, "up")  # a wake just completed
        cfg = type("Cfg", (), {"rescue_interval_s": 5.0, "action_cooldown": flag})()
        writer_events: list[tuple[str, ...]] = []

        class _Writer:
            def write(self, loop_name, snapshot, result):
                writer_events.append(result.events)

        try:
            asyncio.run(
                rescue_task(
                    SnapshotBox(_critical_snapshot(5_000)),
                    queue=queue,
                    registry=_registry(),
                    cfg=cfg,
                    sleep=_stop_sleep,
                    decision_writer=_Writer(),
                )
            )
        except _StopLoop:
            pass
        return writer_events[0]

    assert "cooldown_hold:critical" in run(True)
    assert "cooldown_hold:critical" not in run(False)


def test_action_queue_records_direction_per_model_on_success_but_not_unhide() -> None:
    from tre_controller.planning.planner import DefragAction, HideAction, UnhideAction

    clock = _Clock(1_000)
    queue = ActionQueue(_OkClient(), now_ms=clock)
    queue.submit([HideAction("a", ("a-1",), "probe_started", "rescue"), DefragAction((), "d", "rescue")])
    asyncio.run(queue.drain_once())
    clock.now = 2_000
    queue.submit([UnhideAction("b", ("b-1",), "slo_violation", "safescale")])
    asyncio.run(queue.drain_once())

    # Unhide (probe rollback) restores capacity and is not a cooldown-starting action.
    assert queue.last_actions() == {"a": (1_000, "down")}

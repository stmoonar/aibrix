"""The sleep primitive: hide -> gateway ack -> drain -> /sleep (plan 2026-09-27 D1-D4)."""

import logging

import pytest

from tre_common import rediskeys
from tre_sm.ops.sleep_primitive import (
    GatewayAckTimeout,
    GatewayState,
    SleepFailed,
    SleepJournal,
    SleepPrimitive,
    SleepTarget,
    parse_vllm_load,
)

from sm_test_fakes import (
    FakeGateway,
    FakeRedis,
    FakeRuntime,
    FakeVllm,
    Result,
    TickingClock,
    binding_of,
    pod,
    policy,
)


class World:
    """One awake pod behind a gateway with configurable plugin instances."""

    def __init__(self, *, instances=("gw-1",), auto_ack=True, plugin_pods=None, **policy_overrides):
        self.redis = FakeRedis()
        self.snapshot = pod("pod-a", "m1", (0,), ip="10.0.0.1")
        self.runtime = FakeRuntime([self.snapshot])
        self.vllm = FakeVllm()
        self.vllm.sleeping["10.0.0.1"] = False
        self.gateway = FakeGateway(self.redis, self.runtime)
        for instance in instances:
            self.gateway.heartbeat(instance)
            if auto_ack:
                self.gateway.auto_ack.add(instance)
        self.events: list[tuple] = []
        self.runtime.events = self.events
        self.vllm.events = self.events
        self.hooks: list = []
        self.clock = TickingClock(self._tick)
        self.plugin_pods = plugin_pods
        self.journal = SleepJournal(self.redis)
        self.primitive = SleepPrimitive(
            runtime_ops=self.runtime,
            vllm_ops=self.vllm,
            policy=policy(**policy_overrides),
            gateway=GatewayState(
                self.redis,
                plugin_pods=(lambda: set(self.plugin_pods)) if plugin_pods is not None else None,
            ),
            journal=self.journal,
            clock=self.clock,
        )

    def _tick(self, now):
        self.gateway.tick()
        for hook in list(self.hooks):
            hook(now)

    def target(self):
        return SleepTarget(binding_of(self.snapshot), "10.0.0.1")

    def sleep(self, path="scale_down", **kwargs):
        return self.primitive.sleep([self.target()], path=path, **kwargs)

    def stats(self):
        return self.journal.stats()


def test_hide_patch_precedes_ack_drain_and_sleep_with_mode_wait():
    world = World()
    world.gateway.ack_after_polls = 3  # informer lag: acks after 3 polls
    world.gateway.inflight("pod-a", "gw-1", total=2)
    world.vllm.load["10.0.0.1"] = 2

    def finish_requests(now):
        if now > 1004.0:  # both requests finish ~4 s after the hide
            world.gateway.inflight("pod-a", "gw-1", total=0)
            world.vllm.load["10.0.0.1"] = 0

    world.hooks.append(finish_requests)

    [outcome] = world.sleep()

    kinds = [event[:3] for event in world.events]
    assert kinds[0] == ("patch", "pod-a", "hidden")  # label + route-gen in ONE patch
    assert kinds[1] == ("vllm", "sleep", "10.0.0.1")
    assert world.events[1] == ("vllm", "sleep", "10.0.0.1", "wait", True)  # X-TRE-Hidden
    assert kinds[2] == ("patch", "pod-a", "sleeping")
    assert outcome["ack_mode"] == "plugin"
    assert outcome["ack_latency_ms"] >= 1500  # waited for the (lagging) ack
    assert outcome["drained"] is True and outcome["forced_abort"] is False
    assert outcome["waited_s"] >= 4.0  # waited for inflight == 0
    assert world.stats()["sleeps_total"] == 1
    assert "forced_abort_total" not in world.stats()
    assert world.journal.entries() == {}
    assert world.journal.ack_latencies_ms()


def test_sleep_waits_for_engine_running_even_when_gateway_reports_zero():
    world = World()
    world.gateway.inflight("pod-a", "gw-1", total=0)
    world.vllm.load["10.0.0.1"] = 1  # e.g. a request that bypassed the gateway
    world.hooks.append(lambda now: world.vllm.load.__setitem__("10.0.0.1", 0) if now > 1007 else None)

    [outcome] = world.sleep()

    assert outcome["drained"] is True
    assert outcome["waited_s"] >= 7.0


def test_every_live_instance_must_ack_and_missing_field_is_not_an_ack():
    world = World(instances=("gw-1", "gw-2"), auto_ack=False)
    world.gateway.auto_ack.add("gw-1")  # gw-2 never writes a seen field

    with pytest.raises(GatewayAckTimeout, match="gw-2"):
        world.sleep()

    # rolled back: routable again under a NEW generation, journal cleared
    assert world.runtime.patches[-1][:2] == ("pod-a", "awake")
    assert world.runtime.patches[-1][2] > world.runtime.patches[0][2]
    assert not any(call[0] == "sleep" for call in world.vllm.calls)
    assert world.stats()["ack_timeout_total"] == 1
    assert world.stats()["rollback_total"] == 1
    assert world.journal.entries() == {}


def test_ack_semantics_gen_equal_needs_routable_false_gen_greater_supersedes():
    world = World(auto_ack=False)
    target_gen = world.runtime.gen.get("pod-a", 0) + 1
    world.gateway.seen("pod-a", "gw-1", gen=target_gen, routable=True)  # stale label view

    with pytest.raises(GatewayAckTimeout):
        world.sleep()

    world2 = World(auto_ack=False)
    # A later SM patch (gen + 5) superseded the target: converged even if routable.
    world2.gateway.seen("pod-a", "gw-1", gen=100, routable=True)
    [outcome] = world2.sleep()
    assert outcome["ack_mode"] == "plugin"


def test_stale_heartbeat_but_ready_plugin_pod_must_still_ack():
    world = World(instances=(), auto_ack=False, plugin_pods={"gw-ready"})
    # No fresh heartbeat at all, but a Ready plugin pod: no fallback, wait for its ack.
    world.gateway.heartbeat("gw-ready", age_ms=60_000)

    with pytest.raises(GatewayAckTimeout, match="gw-ready"):
        world.sleep()
    assert world.stats().get("ack_fallback_total") is None

    world.gateway.auto_ack.add("gw-ready")
    world.runtime.snapshots["pod-a"] = world.snapshot
    [outcome] = world.sleep()
    assert outcome["ack_mode"] == "plugin"


def test_redis_read_error_is_not_converged_and_rolls_back():
    world = World()
    world.redis.fail_reads = True

    with pytest.raises(GatewayAckTimeout, match="redis down"):
        world.sleep()

    world.redis.fail_reads = False
    assert world.runtime.patches[-1][:2] == ("pod-a", "awake")
    assert not any(call[0] == "sleep" for call in world.vllm.calls)


def test_no_live_plugin_instance_falls_back_to_label_and_grace(caplog):
    world = World(instances=(), no_plugin_grace_s=5.0)
    world.gateway.heartbeat("gw-dead", age_ms=60_000)  # stale, and no plugin pods listed

    with caplog.at_level(logging.WARNING, logger="tre_sm.sleep"):
        [outcome] = world.sleep()

    assert outcome["ack_mode"] == "fallback_no_plugin"
    assert 5.0 in world.clock.slept
    assert world.stats()["ack_fallback_total"] == 1
    assert "no live gateway plugin instance" in caplog.text
    assert [e[:3] for e in world.events][:2] == [("patch", "pod-a", "hidden"), ("vllm", "sleep", "10.0.0.1")]


def test_budget_exhaustion_aborts_and_counts_forced_aborts():
    world = World(budgets_s={"urgent": 30.0})
    world.gateway.inflight("pod-a", "gw-1", total=3)  # never finishes
    world.vllm.load["10.0.0.1"] = 3

    [outcome] = world.sleep(path="urgent")

    assert outcome["sleep_mode"] == "abort"
    assert outcome["forced_abort"] is True
    assert outcome["forced_abort_requests"] == 3
    assert 30.0 <= outcome["waited_s"] < 31.0  # the soft budget, not the hard cap
    assert world.stats()["forced_abort_total"] == 1
    assert world.stats()["forced_abort_requests_total"] == 3
    assert world.stats()["sleeps_path_urgent"] == 1


def test_caller_budget_overrides_path_budget_and_is_capped_by_hard_cap():
    world = World(hard_cap_s=40.0)
    world.gateway.inflight("pod-a", "gw-1", total=1)

    [outcome] = world.sleep(path="safescale_commit", drain_budget_s=500.0)

    assert outcome["forced_abort"] is True
    assert 40.0 <= outcome["waited_s"] < 41.0


def test_non_continuable_requests_are_waited_for_past_soft_budget_up_to_hard_cap():
    world = World(budgets_s={"urgent": 10.0}, hard_cap_s=60.0)
    world.gateway.inflight("pod-a", "gw-1", total=2, non_continuable=1)

    def finish(now):
        if now > 1045.0:  # 45 s: past the 10 s soft budget, inside the 60 s hard cap
            world.gateway.inflight("pod-a", "gw-1", total=0, non_continuable=0)

    world.hooks.append(finish)

    [outcome] = world.sleep(path="urgent")

    assert outcome["drained"] is True and outcome["forced_abort"] is False
    assert outcome["sleep_mode"] == "wait"
    assert outcome["waited_s"] >= 45.0


def test_non_continuable_still_running_at_hard_cap_is_aborted():
    world = World(budgets_s={"urgent": 10.0}, hard_cap_s=60.0)
    world.gateway.inflight("pod-a", "gw-1", total=1, non_continuable=1)

    [outcome] = world.sleep(path="urgent")

    assert outcome["forced_abort"] is True
    assert outcome["non_continuable_at_sleep"] == 1
    assert 60.0 <= outcome["waited_s"] < 61.0


def test_inflight_of_dead_instances_is_ignored():
    world = World()
    world.gateway.inflight("pod-a", "gw-gone", total=7)  # not live: ignored

    [outcome] = world.sleep()

    assert outcome["drained"] is True and outcome["waited_s"] < 1.0


def test_old_vllm_images_get_a_plain_sleep():
    world = World(vllm_sleep_mode_param=False)

    world.sleep()

    assert world.vllm.calls[0] == ("sleep", "10.0.0.1", None, True)


def test_wait_mode_failure_falls_back_to_abort_and_counts_it():
    world = World()
    world.vllm.sleep_results = [Result(False, "timed out")]  # mode=wait did not finish

    [outcome] = world.sleep()

    modes = [call[2] for call in world.vllm.calls if call[0] == "sleep"]
    assert modes == ["wait", "abort"]
    assert outcome["forced_abort"] is True
    assert world.stats()["forced_abort_total"] == 1


def test_failed_sleep_rolls_back_routable_under_a_new_generation():
    world = World()
    world.vllm.sleep_results = [Result(False, "boom"), Result(False, "boom")]

    with pytest.raises(SleepFailed):
        world.sleep()

    states = [(state, gen) for _pod, state, gen in world.runtime.patches]
    assert states[0][0] == "hidden"
    assert states[-1][0] == "awake" and states[-1][1] > states[0][1]
    assert world.stats()["rollback_total"] == 1
    assert world.journal.entries() == {}


def test_rollback_restores_hidden_probe_pods_as_hidden():
    world = World(auto_ack=False, ack_timeout_s=1.0)
    hidden = pod("pod-a", "m1", (0,), ip="10.0.0.1", state="hidden")
    world.runtime.snapshots["pod-a"] = hidden

    with pytest.raises(GatewayAckTimeout):
        world.primitive.sleep([SleepTarget(binding_of(hidden), "10.0.0.1")], path="safescale_commit")

    assert world.runtime.patches[-1][1] == "hidden"


def test_several_pods_hide_together_and_drain_concurrently():
    world = World()
    second = pod("pod-b", "m1", (1,), ip="10.0.0.2")
    world.runtime.snapshots["pod-b"] = second
    world.vllm.sleeping["10.0.0.2"] = False
    world.gateway.inflight("pod-a", "gw-1", total=1)
    world.gateway.inflight("pod-b", "gw-1", total=1)
    world.hooks.append(
        lambda now: [world.gateway.inflight(p, "gw-1", total=0) for p in ("pod-a", "pod-b")]
        if now > 1020
        else None
    )

    outcomes = world.primitive.sleep(
        [world.target(), SleepTarget(binding_of(second), "10.0.0.2")], path="scale_down"
    )

    hides = [i for i, e in enumerate(world.events) if e[0] == "patch" and e[2] == "hidden"]
    sleeps = [i for i, e in enumerate(world.events) if e[:2] == ("vllm", "sleep")]
    assert len(hides) == 2 and max(hides) < min(sleeps)  # both hidden before any sleep
    assert all(o["drained"] for o in outcomes)
    assert max(o["waited_s"] for o in outcomes) < 25.0  # one shared window, not 2 x 20 s


def test_crash_mid_sleep_leaves_journal_evidence():
    world = World()

    class Crash(BaseException):
        pass

    def crash(*_args, **_kwargs):
        raise Crash()

    world.vllm.sleep = crash  # SM dies between hide and /sleep
    world.runtime.write_binding_annotations_orig = world.runtime.write_binding_annotations

    def no_rollback(binding, *, state):
        if state != "hidden":
            raise RuntimeError("SM is gone")
        return world.runtime.write_binding_annotations_orig(binding, state=state)

    world.runtime.write_binding_annotations = no_rollback

    with pytest.raises(Crash):
        world.sleep()

    entry = world.journal.entries()["pod-a"]
    assert entry["phase"] == "sleeping"
    assert entry["binding_id"] == "m1/node-a/0"
    assert rediskeys.SM_SLEEP_OPS_KEY in world.redis.hashes


def test_parse_vllm_load_sums_running_and_waiting():
    text = (
        "# TYPE vllm:num_requests_running gauge\n"
        'vllm:num_requests_running{engine="0",model_name="m"} 2.0\n'
        'vllm:num_requests_running{engine="1",model_name="m"} 1.0\n'
        'vllm:num_requests_waiting{model_name="m"} 4.0\n'
        "vllm:gpu_cache_usage_perc 0.3\n"
    )
    assert parse_vllm_load(text) == 7
    assert parse_vllm_load("") is None
    assert parse_vllm_load("other_metric 1\n") is None

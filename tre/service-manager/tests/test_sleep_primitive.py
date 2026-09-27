"""The sleep primitive: hide -> gateway ack -> drain -> /sleep (plan 2026-09-27 D1-D4)."""

import logging

import pytest

from tre_common import rediskeys
from tre_sm.ops.sleep_primitive import (
    GatewayAckTimeout,
    GatewayState,
    ReservationLost,
    ServiceShuttingDown,
    SleepCancelled,
    SleepFailed,
    SleepIncomplete,
    SleepJournal,
    SleepPrimitive,
    SleepTarget,
    parse_vllm_load,
    parse_vllm_version,
    sleep_mode_supported,
)
from tre_sm.state.sleep_reservations import ReservationConflict, SleepReservations

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
        self.reservations = SleepReservations(self.redis)
        self.primitive = SleepPrimitive(
            runtime_ops=self.runtime,
            vllm_ops=self.vllm,
            policy=policy(**policy_overrides),
            gateway=GatewayState(
                self.redis,
                plugin_pods=(lambda: set(self.plugin_pods)) if plugin_pods is not None else None,
                monotonic=lambda: self.clock.monotonic(),
            ),
            journal=self.journal,
            reservations=self.reservations,
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


def test_no_live_plugin_instance_is_not_converged_by_default():
    # An old plugin image (or broken coordination) routes without heartbeating:
    # "every live instance acked" must not be vacuously true.
    world = World(instances=())
    world.gateway.heartbeat("gw-dead", age_ms=60_000)  # never advances

    with pytest.raises(GatewayAckTimeout, match="0 live plugin instance"):
        world.sleep()

    assert world.runtime.patches[-1][:2] == ("pod-a", "awake")
    assert not any(call[0] == "sleep" for call in world.vllm.calls)
    assert world.stats().get("ack_fallback_total") is None


def test_min_live_instances_is_enforced():
    world = World(instances=("gw-1",), gateway_min_instances=2)

    with pytest.raises(GatewayAckTimeout, match="1 live plugin instance"):
        world.sleep()


def test_opt_in_no_plugin_fallback_uses_label_and_grace(caplog):
    # ack_timeout > staleness: a first-seen instance counts as "maybe live" until
    # it has had instance_staleness_s to show an advancing heartbeat.
    world = World(instances=(), no_plugin_grace_s=5.0, fallback_no_plugin=True, ack_timeout_s=20.0)
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


def test_non_continuable_still_running_at_hard_cap_rolls_back_never_aborts():
    world = World(budgets_s={"urgent": 10.0}, hard_cap_s=60.0)
    world.gateway.inflight("pod-a", "gw-1", total=1, non_continuable=1)

    with pytest.raises(SleepIncomplete) as info:
        world.sleep(path="urgent")

    [outcome] = info.value.outcomes
    assert outcome["status"] == "rolled_back"
    assert outcome["non_continuable_at_sleep"] == 1
    assert 60.0 <= outcome["waited_s"] < 61.0
    assert not any(call[0] == "sleep" for call in world.vllm.calls)
    assert world.runtime.patches[-1][:2] == ("pod-a", "awake")
    assert world.stats()["non_continuable_rollback_total"] == 1
    assert "forced_abort_total" not in world.stats()


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


# --------------------------------------------------------------- P1-1 / P2-5
def test_metrics_unavailable_is_not_drained_and_rolls_back_at_the_hard_cap():
    world = World(budgets_s={"urgent": 10.0}, hard_cap_s=40.0)
    world.gateway.inflight("pod-a", "gw-1", total=0)
    world.vllm.metrics_down.add("10.0.0.1")  # engine gauges unavailable (None)

    with pytest.raises(SleepIncomplete) as info:
        world.sleep(path="urgent")

    [outcome] = info.value.outcomes
    assert outcome["status"] == "rolled_back"
    assert "unknown" in outcome["reason"]
    assert 40.0 <= outcome["waited_s"] < 41.0  # waited to the hard cap, not the soft budget
    assert not any(call[0] == "sleep" for call in world.vllm.calls)
    assert world.stats()["drain_unknown_rollback_total"] == 1


def test_metrics_coming_back_drains_normally():
    world = World(budgets_s={"urgent": 10.0}, hard_cap_s=60.0)
    world.vllm.metrics_down.add("10.0.0.1")
    world.hooks.append(lambda now: world.vllm.metrics_down.discard("10.0.0.1") if now > 1025 else None)

    [outcome] = world.sleep(path="urgent")

    assert outcome["drained"] is True and outcome["sleep_mode"] == "wait"
    assert outcome["waited_s"] >= 25.0


def test_read_error_at_the_soft_deadline_keeps_waiting_instead_of_aborting():
    world = World(budgets_s={"urgent": 10.0}, hard_cap_s=60.0)
    world.gateway.inflight("pod-a", "gw-1", total=1, non_continuable=0)
    world.vllm.load["10.0.0.1"] = 1

    def flaky(now):
        # Redis unreadable from 5 s to 25 s: spans the 10 s soft deadline.
        world.redis.fail_reads = 1005.0 < now < 1025.0

    world.hooks.append(flaky)

    [outcome] = world.sleep(path="urgent")

    # No abort while the state was unknown; once readable again (known, only
    # continuable requests, past the soft budget) the abort is allowed.
    assert outcome["forced_abort"] is True
    assert outcome["waited_s"] >= 25.0


def test_read_errors_until_the_hard_cap_roll_back():
    world = World(budgets_s={"urgent": 10.0}, hard_cap_s=30.0)
    world.gateway.inflight("pod-a", "gw-1", total=1)
    world.hooks.append(lambda now: setattr(world.redis, "fail_reads", now > 1003.0))

    with pytest.raises(SleepIncomplete) as info:
        world.sleep(path="urgent")

    world.redis.fail_reads = False
    [outcome] = info.value.outcomes
    assert outcome["status"] == "rolled_back"
    assert not any(call[0] == "sleep" for call in world.vllm.calls)


def test_non_continuable_is_sticky_across_read_errors():
    world = World(budgets_s={"urgent": 5.0}, hard_cap_s=30.0)
    world.gateway.inflight("pod-a", "gw-1", total=1, non_continuable=1)
    world.vllm.load["10.0.0.1"] = 1
    world.hooks.append(lambda now: setattr(world.redis, "fail_reads", now > 1002.0))

    with pytest.raises(SleepIncomplete) as info:
        world.sleep(path="urgent")

    world.redis.fail_reads = False
    [outcome] = info.value.outcomes
    # The last successful read said 1 non-continuable request: read errors did
    # not zero it (the old code aborted it at the soft deadline).
    assert outcome["non_continuable_at_sleep"] == 1
    assert outcome["status"] == "rolled_back"
    assert "forced_abort_total" not in world.stats()


def test_unreadable_inflight_entry_of_a_live_instance_is_unknown():
    world = World(budgets_s={"urgent": 5.0}, hard_cap_s=20.0)
    world.redis.hset("tre:v2:gw:inflight:pod-a", "gw-1", "{not json")

    with pytest.raises(SleepIncomplete):
        world.sleep(path="urgent")
    assert not any(call[0] == "sleep" for call in world.vllm.calls)


def test_engine_layer_is_checked_even_when_the_plugin_reports_zero():
    # A plugin shutting down zeroes its counts while Envoy may still stream.
    world = World(budgets_s={"urgent": 5.0}, hard_cap_s=20.0)
    world.gateway.inflight("pod-a", "gw-1", total=0)
    world.vllm.load["10.0.0.1"] = 2
    world.hooks.append(lambda now: world.vllm.load.__setitem__("10.0.0.1", 0) if now > 1003 else None)

    [outcome] = world.sleep(path="urgent")

    assert outcome["drained"] is True and outcome["waited_s"] >= 3.0


# ---------------------------------------------------------------------- P2-6
def test_unknown_physical_state_after_sleep_keeps_the_pod_hidden():
    world = World()
    world.vllm.physical_override["10.0.0.1"] = None  # /is_sleeping unreachable

    with pytest.raises(SleepIncomplete) as info:
        world.sleep()

    [outcome] = info.value.outcomes
    assert outcome["status"] == "unconfirmed"
    assert world.runtime.patches[-1][1] == "hidden"  # routing NOT re-opened
    entry = world.journal.entries()["pod-a"]
    assert entry["phase"] == "sleep_unconfirmed"
    assert world.stats()["sleep_unconfirmed_total"] == 1


def test_physically_awake_after_sleep_rolls_back_routing():
    world = World()
    world.vllm.physical_override["10.0.0.1"] = False

    with pytest.raises(SleepIncomplete) as info:
        world.sleep()

    assert info.value.outcomes[0]["status"] == "rolled_back"
    assert world.runtime.patches[-1][1] == "awake"
    assert world.journal.entries() == {}


def test_wait_failure_with_unknown_state_keeps_the_pod_hidden():
    world = World()
    world.vllm.sleep_results = [Result(False, "timed out")]
    original = world.vllm.sleep

    def sleep(pod_ip, **kwargs):
        world.vllm.metrics_down.add(pod_ip)  # engine unreadable after the call
        return original(pod_ip, **kwargs)

    world.vllm.sleep = sleep

    with pytest.raises(SleepIncomplete) as info:
        world.sleep()

    assert info.value.outcomes[0]["status"] == "unconfirmed"
    assert [c[2] for c in world.vllm.calls if c[0] == "sleep"] == ["wait"]  # no blind abort
    assert world.runtime.patches[-1][1] == "hidden"


# ---------------------------------------------------------------------- P2-7
def test_multi_target_partial_failure_reports_per_target_outcomes():
    world = World()
    second = pod("pod-b", "m1", (1,), ip="10.0.0.2")
    world.runtime.snapshots["pod-b"] = second
    world.vllm.sleeping["10.0.0.2"] = False
    world.vllm.fail_sleep_for.add("10.0.0.2")

    with pytest.raises(SleepIncomplete) as info:
        world.primitive.sleep(
            [world.target(), SleepTarget(binding_of(second), "10.0.0.2")], path="scale_down"
        )

    by_pod = {o["serve_id"]: o["status"] for o in info.value.outcomes}
    assert by_pod == {"pod-a": "slept", "pod-b": "rolled_back"}
    states = {pod_name: state for pod_name, state, _gen in world.runtime.patches}
    assert states == {"pod-a": "sleeping", "pod-b": "awake"}
    assert world.reservations.active() == {}
    assert world.journal.entries() == {}


# ---------------------------------------------------------------------- P2-4
class _Mono:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_liveness_is_score_advancing_not_wall_clock(caplog):
    from tre_common import rediskeys

    redis = FakeRedis(now_ms=1_700_000_000_000)
    mono = _Mono()
    state = GatewayState(redis, monotonic=mono)
    future = redis.now_ms + 160_000  # plugin clock 160 s ahead of Redis TIME
    past = redis.now_ms - 300_000  # plugin clock 5 min behind
    redis.zadd(rediskeys.GW_INSTANCES_KEY, {"ahead": future, "behind": past, "frozen": future})

    assert state.live_instances(10.0) == []  # first sighting: not yet seen advancing
    assert state.instances(10.0)[1] == ["ahead", "behind", "frozen"]  # pending

    with caplog.at_level(logging.WARNING, logger="tre_sm.sleep"):
        mono.now += 2.0
        redis.zadd(rediskeys.GW_INSTANCES_KEY, {"ahead": future + 2000, "behind": past + 2000})
        assert state.live_instances(10.0) == ["ahead", "behind"]
    assert "ahead of Redis TIME" in caplog.text

    # 'ahead' keeps beating; 'behind' stops; 'frozen' never moved.
    for step in range(1, 7):
        mono.now += 2.0
        redis.zadd(rediskeys.GW_INSTANCES_KEY, {"ahead": future + 2000 + step * 2000})
    assert state.live_instances(10.0) == ["ahead"]
    assert state.instances(10.0)[1] == []  # 'frozen' is no longer even pending


def test_fields_of_non_live_instances_are_ignored_for_ack_and_inflight():
    world = World(instances=("gw-1",))
    # A crashed instance left fields behind (its hash TTL keeps being refreshed).
    world.gateway.seen("pod-a", "gw-crashed", gen=0, routable=True)
    world.gateway.inflight("pod-a", "gw-crashed", total=9, non_continuable=9)
    world.gateway.heartbeat("gw-crashed", age_ms=60_000)

    [outcome] = world.sleep()

    assert outcome["drained"] is True and outcome["status"] == "slept"


# ---------------------------------------------------------------------- mode
def test_sleep_mode_param_auto_detects_the_vllm_version():
    assert sleep_mode_supported("0.30.0") and sleep_mode_supported("0.18.0")
    assert not sleep_mode_supported("0.10.1") and not sleep_mode_supported("0.17.1")
    assert not sleep_mode_supported("0.1.dev21953+g378504a54") and not sleep_mode_supported(None)
    assert parse_vllm_version("v0.30.1rc1") == (0, 30, 1)

    world = World(vllm_sleep_mode_param="auto")
    world.vllm.versions["10.0.0.1"] = "0.30.0"
    world.sleep()
    assert [c[2] for c in world.vllm.calls if c[0] == "sleep"] == ["wait"]

    old = World(vllm_sleep_mode_param="auto")
    old.vllm.versions["10.0.0.1"] = "0.10.1"
    old.sleep()
    assert [c[2] for c in old.vllm.calls if c[0] == "sleep"] == [None]  # plain /sleep


def test_version_is_cached_per_pod_and_failures_are_not_cached():
    world = World(vllm_sleep_mode_param="auto")
    world.vllm.versions["10.0.0.1"] = None  # /version unreachable
    world.sleep()
    assert [c[2] for c in world.vllm.calls if c[0] == "sleep"] == [None]

    world.vllm.sleeping["10.0.0.1"] = False
    world.runtime.snapshots["pod-a"] = world.snapshot
    world.vllm.versions["10.0.0.1"] = "0.30.0"
    world.sleep()
    world.vllm.sleeping["10.0.0.1"] = False
    world.sleep()
    assert world.vllm.version_calls == ["10.0.0.1", "10.0.0.1"]  # third sleep used the cache
    assert [c[2] for c in world.vllm.calls if c[0] == "sleep"] == [None, "wait", "wait"]


def test_forced_abort_without_mode_support_is_a_plain_sleep_after_the_budget():
    world = World(vllm_sleep_mode_param="false", budgets_s={"urgent": 5.0})
    world.gateway.inflight("pod-a", "gw-1", total=1)

    [outcome] = world.sleep(path="urgent")

    assert outcome["forced_abort"] is True and outcome["sleep_mode"] is None


# ------------------------------------------------------ reservations / cancel
def test_a_reserved_binding_cannot_be_prepared_twice_and_is_released_after():
    world = World()
    batch = world.primitive.prepare([world.target()], path="scale_down")
    with pytest.raises(ReservationConflict):
        world.primitive.prepare([world.target()], path="scale_down")
    assert set(world.reservations.active()) == {"m1/node-a/0"}

    world.primitive.drain(batch)
    world.primitive.commit(batch)

    assert world.reservations.active() == {}


def test_losing_the_reservation_mid_drain_rolls_back():
    world = World()
    world.gateway.inflight("pod-a", "gw-1", total=1)
    world.hooks.append(
        lambda now: world.redis.hashes.pop("tre:v2:sm:sleep_reservations", None) if now > 1004 else None
    )

    with pytest.raises(ReservationLost):
        world.sleep()

    assert world.runtime.patches[-1][:2] == ("pod-a", "awake")
    assert not any(call[0] == "sleep" for call in world.vllm.calls)


def test_shutdown_mid_drain_rolls_back_and_refuses_new_sleeps():
    world = World()
    world.gateway.inflight("pod-a", "gw-1", total=1)
    world.hooks.append(lambda now: world.primitive.begin_shutdown() if now > 1003 else None)

    with pytest.raises(SleepCancelled):
        world.sleep()

    assert world.runtime.patches[-1][:2] == ("pod-a", "awake")
    assert not any(call[0] == "sleep" for call in world.vllm.calls)
    assert world.journal.entries() == {} and world.reservations.active() == {}
    assert world.primitive.active_count() == 0
    with pytest.raises(ServiceShuttingDown):
        world.sleep()

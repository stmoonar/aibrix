"""The sleep primitive (plan 2026-09-27 D1-D4), whole-lock since 2026-10-02:
hide -> gateway ack -> one load read -> one /sleep -> physical confirmation, in
one call under the caller's writer lock; no reservation, no drain."""

import logging

import pytest

from tre_common import rediskeys
from tre_sm.ops.sleep_primitive import (
    PHASE_SLEPT,
    GatewayAckTimeout,
    GatewayState,
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


class Unanswered:
    """A vLLM call that got no HTTP answer (timeout / transport error)."""

    success = False
    status_code = None
    message = "read timed out"


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
                monotonic=lambda: self.clock.monotonic(),
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

    def modes(self):
        return [call[2] for call in self.vllm.calls if call[0] == "sleep"]


def test_hide_patch_precedes_ack_and_one_sleep_mode_abort():
    world = World()
    world.gateway.ack_after_polls = 3  # informer lag: acks after 3 polls

    [outcome] = world.sleep()

    kinds = [event[:3] for event in world.events]
    assert kinds[0] == ("patch", "pod-a", "hidden")  # label + route-gen in ONE patch
    assert world.events[1] == ("vllm", "sleep", "10.0.0.1", "abort", True)  # X-TRE-Hidden
    assert kinds[2] == ("patch", "pod-a", "sleeping")
    assert outcome["ack_mode"] == "plugin"
    assert outcome["ack_latency_ms"] >= 1500  # waited for the (lagging) ack
    assert outcome["drained"] is True and outcome["forced_abort"] is False
    assert outcome["sleep_mode"] == "abort"  # a sleep interrupts (2026-10-02)
    assert world.stats()["sleeps_total"] == 1
    assert "forced_abort_total" not in world.stats()
    assert world.journal.entries() == {}
    assert world.journal.ack_latencies_ms()


def test_requests_in_flight_are_never_waited_for():
    world = World()
    world.gateway.inflight("pod-a", "gw-1", total=3, non_continuable=1)
    world.vllm.load["10.0.0.1"] = 3

    [outcome] = world.sleep(path="urgent")

    assert world.modes() == ["abort"]
    assert outcome["forced_abort"] is True and outcome["forced_abort_requests"] == 3
    assert outcome["aborted"] == {
        "state_known": True, "in_flight": 3, "continuable": 2, "non_continuable": 1, "unclassified": 0,
    }
    assert outcome["waited_s"] < 1.0  # no drain
    assert world.stats()["forced_abort_total"] == 1
    assert world.stats()["no_drain_non_continuable_aborted_total"] == 1
    assert world.stats()["sleeps_path_urgent"] == 1


def test_sleep_mode_when_idle_wait_is_opt_in_and_only_when_nothing_is_in_flight():
    idle = World(sleep_mode_when_idle="wait")
    idle.sleep()
    assert idle.modes() == ["wait"]

    busy = World(sleep_mode_when_idle="wait")
    busy.vllm.load["10.0.0.1"] = 1
    busy.sleep()
    assert busy.modes() == ["abort"]


def test_unknown_load_is_an_abort_counted_as_unknown():
    world = World()
    world.vllm.metrics_down.add("10.0.0.1")

    [outcome] = world.sleep()

    assert world.modes() == ["abort"]
    assert outcome["forced_abort"] is True and outcome["aborted"]["state_known"] is False
    assert world.stats()["no_drain_unknown_state_abort_total"] == 1


def test_the_engine_layer_counts_even_when_the_plugin_reports_zero():
    world = World()
    world.gateway.inflight("pod-a", "gw-1", total=0)
    world.vllm.load["10.0.0.1"] = 2  # e.g. requests that bypassed the gateway

    [outcome] = world.sleep()

    assert outcome["forced_abort"] is True
    assert outcome["aborted"]["unclassified"] == 2


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


def test_the_ack_wait_is_bounded_by_ack_timeout_s():
    world = World(auto_ack=False, ack_timeout_s=5.0)
    started = world.clock.monotonic()

    with pytest.raises(GatewayAckTimeout):
        world.sleep()

    assert world.clock.monotonic() - started <= 5.0 + 0.6


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


def test_inflight_of_dead_instances_is_ignored():
    world = World()
    world.gateway.inflight("pod-a", "gw-gone", total=7)  # not live: ignored

    [outcome] = world.sleep()

    assert outcome["drained"] is True and outcome["forced_abort"] is False


def test_operator_opt_in_sends_a_plain_sleep():
    world = World(vllm_sleep_mode_param=False)
    world.vllm.load["10.0.0.1"] = 1

    [outcome] = world.sleep()

    assert world.vllm.calls[0] == ("sleep", "10.0.0.1", None, True)
    assert outcome["forced_abort"] is True and outcome["sleep_mode"] is None
    assert world.stats()["plain_sleep_with_inflight_total"] == 1


def test_no_mode_parameter_with_requests_in_flight_rolls_back_without_a_sleep():
    world = World(vllm_sleep_mode_param="auto")
    world.vllm.versions["10.0.0.1"] = "0.10.1"
    world.vllm.load["10.0.0.1"] = 2

    with pytest.raises(SleepIncomplete) as info:
        world.sleep()

    assert info.value.outcomes[0]["status"] == "rolled_back"
    assert world.modes() == []  # no plain /sleep over requests in flight
    assert world.runtime.patches[-1][:2] == ("pod-a", "awake")
    assert world.stats()["sleep_rolled_back_no_mode_total"] == 1


def test_no_mode_parameter_and_nothing_in_flight_is_a_plain_sleep():
    world = World(vllm_sleep_mode_param="auto")
    world.vllm.versions["10.0.0.1"] = "0.10.1"

    [outcome] = world.sleep()

    assert world.modes() == [None] and outcome["status"] == "slept"


def test_one_sleep_call_only_a_failed_call_rolls_back_after_a_reprobe():
    world = World()
    world.vllm.sleep_results = [Result(False, "boom")]

    with pytest.raises(SleepFailed):
        world.sleep()

    assert world.modes() == ["abort"]  # ONE /sleep, no fallback call
    states = [(state, gen) for _pod, state, gen in world.runtime.patches]
    assert states[0][0] == "hidden"
    assert states[-1][0] == "awake" and states[-1][1] > states[0][1]
    assert world.stats()["rollback_total"] == 1
    assert world.journal.entries() == {}


def test_a_failed_call_that_did_sleep_is_recorded_asleep():
    world = World()
    world.vllm.sleep_results = [Result(False, "late error")]
    world.vllm.sleeping["10.0.0.1"] = True  # the engine slept anyway

    [outcome] = world.sleep()

    assert outcome["status"] == "slept"
    assert world.runtime.patches[-1][1] == "sleeping"


def test_a_failed_call_on_a_paused_engine_resumes_it_before_reopening():
    world = World()
    world.vllm.sleep_results = [Result(False, "offload failed")]
    calls = []
    world.vllm.is_paused = lambda pod_ip, *, port=None: calls.append("is_paused") or (len(calls) == 1)
    world.vllm.resume = lambda pod_ip, *, port=None: calls.append("resume") or Result()

    with pytest.raises(SleepIncomplete) as info:
        world.sleep()

    assert info.value.outcomes[0]["status"] == "rolled_back"
    assert calls == ["is_paused", "resume", "is_paused"]
    assert world.runtime.patches[-1][1] == "awake"
    assert world.stats()["resume_before_reopen_total"] == 1


def test_a_failed_call_on_an_engine_that_stays_paused_keeps_it_hidden():
    world = World()
    world.vllm.sleep_results = [Result(False, "offload failed")]
    world.vllm.is_paused = lambda pod_ip, *, port=None: True
    world.vllm.resume = lambda pod_ip, *, port=None: Result(False, "nope")

    with pytest.raises(SleepIncomplete) as info:
        world.sleep()

    assert info.value.outcomes[0]["status"] == "unconfirmed"
    assert world.runtime.patches[-1][1] == "hidden"
    assert world.journal.entries()["pod-a"]["phase"] == "sleep_unconfirmed"


def test_a_sleep_call_without_an_answer_leaves_the_pod_hidden_and_returns_at_once():
    world = World(sleep_call_timeout_s=10.0)
    probes = []
    original = world.vllm.is_sleeping

    def counting(pod_ip, *, port=None):
        probes.append(pod_ip)
        return original(pod_ip, port=port)

    world.vllm.is_sleeping = counting

    def hung(pod_ip, *, port=None, mode=None, timeout_s=None, hidden=False):
        world.vllm.calls.append(("sleep", pod_ip, mode, hidden))
        world.clock.now += timeout_s  # the whole call timeout passes
        return Unanswered()

    world.vllm.sleep = hung
    started = world.clock.monotonic()

    with pytest.raises(SleepIncomplete) as info:
        world.sleep()

    [outcome] = info.value.outcomes
    assert outcome["status"] == "unconfirmed"
    assert world.runtime.patches[-1][1] == "hidden"  # routing NOT re-opened
    assert world.journal.entries()["pod-a"]["phase"] == "sleep_unconfirmed"
    assert probes == []  # no further probe after the timeout: the lock is released
    assert world.clock.monotonic() - started <= 10.0 + 1.0
    assert world.stats()["sleep_call_timeout_total"] == 1


def test_not_confirmed_within_the_confirm_timeout_stays_hidden():
    world = World(physical_confirm_timeout_s=8.0)
    world.vllm.physical_override["10.0.0.1"] = False  # /sleep said ok, the engine reads awake
    started = world.clock.monotonic()

    with pytest.raises(SleepIncomplete) as info:
        world.sleep()

    assert info.value.outcomes[0]["status"] == "unconfirmed"
    assert world.runtime.patches[-1][1] == "hidden"
    assert world.journal.entries()["pod-a"]["phase"] == "sleep_unconfirmed"
    assert 8.0 <= world.clock.monotonic() - started <= 8.0 + 1.0


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


def test_rollback_restores_hidden_probe_pods_as_hidden():
    world = World(auto_ack=False, ack_timeout_s=1.0)
    hidden = pod("pod-a", "m1", (0,), ip="10.0.0.1", state="hidden")
    world.runtime.snapshots["pod-a"] = hidden

    with pytest.raises(GatewayAckTimeout):
        world.primitive.sleep([SleepTarget(binding_of(hidden), "10.0.0.1")], path="safescale_commit")

    assert world.runtime.patches[-1][1] == "hidden"


def test_several_pods_hide_together_and_sleep_in_parallel():
    world = World()
    second = pod("pod-b", "m1", (1,), ip="10.0.0.2")
    world.runtime.snapshots["pod-b"] = second
    world.vllm.sleeping["10.0.0.2"] = False

    outcomes = world.primitive.sleep(
        [world.target(), SleepTarget(binding_of(second), "10.0.0.2")], path="scale_down"
    )

    hides = [i for i, e in enumerate(world.events) if e[0] == "patch" and e[2] == "hidden"]
    sleeps = [i for i, e in enumerate(world.events) if e[:2] == ("vllm", "sleep")]
    assert len(hides) == 2 and max(hides) < min(sleeps)  # both hidden before any sleep
    assert [o["status"] for o in outcomes] == ["slept", "slept"]


def test_keep_journal_leaves_slept_entries_until_the_caller_ends_them():
    world = World()

    outcomes = world.sleep(keep_journal=True)

    assert world.journal.entries()["pod-a"]["phase"] == PHASE_SLEPT
    world.primitive.end_journal(outcomes)
    assert world.journal.entries() == {}


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
    assert world.journal.entries() == {}


class _Mono:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_liveness_is_score_advancing_not_wall_clock(caplog):
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


def test_sleep_mode_param_auto_detects_the_vllm_version():
    assert sleep_mode_supported("0.30.0") and sleep_mode_supported("0.18.0")
    assert not sleep_mode_supported("0.10.1") and not sleep_mode_supported("0.17.1")
    assert not sleep_mode_supported("0.1.dev21953+g378504a54") and not sleep_mode_supported(None)
    assert parse_vllm_version("v0.30.1rc1") == (0, 30, 1)

    world = World(vllm_sleep_mode_param="auto")
    world.vllm.versions["10.0.0.1"] = "0.30.0"
    world.sleep()
    assert world.modes() == ["abort"]


def test_version_is_cached_per_pod_and_failures_are_not_cached():
    world = World(vllm_sleep_mode_param="auto")
    world.vllm.versions["10.0.0.1"] = None  # /version unreachable
    world.sleep()
    assert world.modes() == [None]  # nothing in flight: a plain /sleep is safe

    world.vllm.sleeping["10.0.0.1"] = False
    world.runtime.snapshots["pod-a"] = world.snapshot
    world.vllm.versions["10.0.0.1"] = "0.30.0"
    world.sleep()
    world.vllm.sleeping["10.0.0.1"] = False
    world.sleep()
    assert world.vllm.version_calls == ["10.0.0.1", "10.0.0.1"]  # third sleep used the cache
    assert world.modes() == [None, "abort", "abort"]


def test_shutdown_during_the_ack_rolls_back_and_refuses_new_sleeps():
    world = World()
    world.gateway.ack_after_polls = 10
    world.hooks.append(lambda now: world.primitive.begin_shutdown() if now > 1001 else None)

    with pytest.raises(SleepCancelled):
        world.sleep()

    assert world.runtime.patches[-1][:2] == ("pod-a", "awake")
    assert not any(call[0] == "sleep" for call in world.vllm.calls)
    assert world.journal.entries() == {}
    assert world.primitive.active_count() == 0
    with pytest.raises(ServiceShuttingDown):
        world.sleep()


def test_a_lost_writer_fence_rolls_back_before_any_sleep():
    from tre_sm.state.operations import OperationFenceLost, _CURRENT_OPERATION

    class LostHandle:
        operation_id = "op-lost"

        def assert_active(self):
            raise OperationFenceLost("writer fence lost")

    world = World()
    token = _CURRENT_OPERATION.set(LostHandle())
    try:
        with pytest.raises(OperationFenceLost):
            world.sleep()
    finally:
        _CURRENT_OPERATION.reset(token)

    assert not any(call[0] == "sleep" for call in world.vllm.calls)
    assert world.runtime.patches[-1][:2] == ("pod-a", "awake")

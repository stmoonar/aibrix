"""Review 2 of the transparent-sleep service-manager.

P1-2 sleeps sharing a GPU, P1-3 seeding trust, P2-1 lost reservations are never
re-acquired, P2-2 time budget with many targets, P2-3 desired state follows the
outcome, P2-4 startup paths drain outside the writer lock, P3 recovery / TOCTOU /
per-pod convergence.
"""

from datetime import datetime, timezone
import threading
import time

import pytest

from tre_sm.api.v2 import RetryLater, ServiceManagerV2, WakeConflict
from tre_sm.ops.sleep_primitive import (
    GatewayState,
    ReservationLost,
    SleepJournal,
    SleepPrimitive,
    SleepTarget,
)
from tre_sm.state.fleet_seed import seed_desired
from tre_sm.state.fleet_store import DesiredBinding, FleetStateStore
from tre_sm.state.sleep_reservations import ReservationConflict, SleepReservations
from tre_sm.state.store import StateStore

from sm_test_fakes import (
    FakeGateway,
    FakeLeases,
    FakeRedis,
    FakeRuntime,
    FakeSafety,
    FakeVllm,
    LegacyRedis,
    Result,
    TickingClock,
    binding_of,
    deployment,
    fence,
    pod,
    policy,
    registry,
    startup_pod,
)
from test_sleep_lock_phases import StrictCoordinator


def _desired(binding_id, model, gpus, power, hidden=False):
    now = datetime.now(timezone.utc).isoformat()
    return DesiredBinding(binding_id, model, "node-a", tuple(gpus), "resident", power, hidden, 1, now, "t", "t")


class World:
    """Service-manager with a strict (non-reentrant) writer lock and a gateway."""

    def __init__(self, snapshots, desired, *, physical=None, sm_registry=None):
        self.redis = FakeRedis()
        self.runtime = FakeRuntime(snapshots)
        self.vllm = FakeVllm()
        for snapshot in snapshots:
            self.vllm.sleeping[snapshot.pod_ip] = snapshot.annotations["tre.aibrix.io/state"] == "sleeping"
        self.vllm.sleeping.update(physical or {})
        self.gateway = FakeGateway(self.redis, self.runtime)
        self.gateway.heartbeat("gw-1")
        self.gateway.auto_ack.add("gw-1")
        self.hooks = []
        self.clock = TickingClock(self._tick)
        self.store = StateStore(LegacyRedis())
        self.store.save([binding_of(s) for s in snapshots], expected_version=0)
        self.fleet = FleetStateStore(self.redis)
        with fence(self.redis):
            self.fleet.save_desired(desired, expected_version=0)
        self.coordinator = StrictCoordinator(self.redis)
        self.leases = FakeLeases()
        self.service = ServiceManagerV2(
            sm_registry or registry(),
            self.store,
            runtime_ops=self.runtime,
            vllm_ops=self.vllm,
            operation_coordinator=self.coordinator,
            safety_gate=FakeSafety(),
            fleet_store=self.fleet,
            gpu_leases=self.leases,
            gateway_state=GatewayState(self.redis, monotonic=lambda: self.clock.monotonic()),
            sleep_journal=SleepJournal(self.redis),
            sleep_reservations=SleepReservations(self.redis),
            sleep_clock=self.clock,
        )

    def _tick(self, now):
        self.gateway.tick()
        for hook in list(self.hooks):
            hook(now)

    @property
    def primitive(self):
        return self.service._sleep_primitive

    def desired(self):
        return {d.binding_id: (d.power, d.hidden, d.lifecycle) for d in self.fleet.load_desired().bindings}

    def state(self, name):
        return self.runtime.snapshots[name].annotations["tre.aibrix.io/state"]


def _two_pods():
    return [pod("pod-a", "m1", (0,), ip="10.0.0.1"), pod("pod-b", "m1", (1,), ip="10.0.0.2", state="sleeping")]


def _two_desired():
    return [_desired("m1/node-a/0", "m1", (0,), "awake"), _desired("m1/node-a/1", "m1", (1,), "sleeping")]


# ------------------------------------------------------------------ P1-2
def test_two_bindings_sharing_a_gpu_sleep_concurrently_but_wakes_there_are_fenced():
    world = World(
        [
            pod("pod-a", "m1", (0,), ip="10.0.0.1"),
            pod("pod-t", "tp2", (0, 1), ip="10.0.0.5"),
            pod("pod-b", "m1", (1,), ip="10.0.0.2", state="sleeping"),
        ],
        [_desired("m1/node-a/0", "m1", (0,), "awake"), _desired("tp2/node-a/0,1", "tp2", (0, 1), "awake"),
         _desired("m1/node-a/1", "m1", (1,), "sleeping")],
    )
    primitive = world.primitive
    target = lambda name: SleepTarget(binding_of(world.runtime.snapshots[name]), world.runtime.snapshots[name].pod_ip)

    first = primitive.prepare([target("pod-a")], path="scale_down")
    second = primitive.prepare([target("pod-t")], path="scale_down")  # same GPU 0: allowed

    assert set(primitive.reservations.active()) == {"m1/node-a/0", "tp2/node-a/0,1"}
    # A wake on an overlapping GPU still needs the GPU: refused while they drain.
    sleeping_b = binding_of(world.runtime.snapshots["pod-b"])
    with pytest.raises(ReservationConflict):
        world.service._apply_runtime_power_action(sleeping_b, action="wake")
    assert not any(call[0] == "wake_up" for call in world.vllm.calls)
    # A second sleep of the SAME binding is still refused.
    with pytest.raises(ReservationConflict):
        primitive.prepare([target("pod-a")], path="scale_down")

    for batch in (first, second):
        primitive.drain(batch)
    assert [o["status"] for o in primitive.commit(first) + primitive.commit(second)] == ["slept", "slept"]
    assert world.vllm.sleeping["10.0.0.1"] and world.vllm.sleeping["10.0.0.5"]
    assert primitive.reservations.active() == {}


# ------------------------------------------------------------------ P2-1
def test_expired_reservation_recovered_then_the_drain_rolls_back_without_reacquiring():
    world = World(_two_pods(), _two_desired())
    world.gateway.inflight("pod-a", "gw-1", total=1)  # keeps draining
    seen = {}

    def expire_then_recover(now):
        if now < 1003 or seen:
            return
        world.redis.now_ms += 31_000  # a Redis stall: the reservation expired
        seen["recovery"] = world.service.recover_sleep_journal()
        seen["state"] = world.state("pod-a")

    world.hooks.append(expire_then_recover)

    with pytest.raises(ReservationLost) as lost:
        world.service.put_binding_power("pod-a", awake=False)

    assert seen["recovery"]["resolved"] == [{"serve_id": "pod-a", "result": "rolled_back_to_awake"}]
    assert seen["state"] == "awake"
    assert [o["status"] for o in lost.value.outcomes] == ["rolled_back"]
    assert world.primitive.reservations.active() == {}  # never re-acquired
    assert not any(call[0] == "sleep" for call in world.vllm.calls)
    assert world.state("pod-a") == "awake"
    assert world.desired()["m1/node-a/0"][0] == "awake"
    assert "put_binding_power_reservation_lost" in world.coordinator.kinds
    assert world.primitive.journal.stats()["reservation_lost_total"] == 1


def test_a_binding_taken_over_after_expiry_is_left_to_its_new_owner():
    world = World(_two_pods(), _two_desired())
    world.gateway.inflight("pod-a", "gw-1", total=1)
    other = {}

    def expire_and_take_over(now):
        if now < 1003 or other:
            return
        world.redis.now_ms += 31_000
        binding = binding_of(world.runtime.snapshots["pod-a"])
        other["token"] = world.primitive.reservations.acquire(
            [binding], owner="sm-2", operation_id="op-2", ttl_s=300
        )
        world.primitive.journal.begin("pod-a", {"reservation_token": other["token"], "phase": "draining"})

    world.hooks.append(expire_and_take_over)

    with pytest.raises(ReservationLost) as lost:
        world.service.put_binding_power("pod-a", awake=False)

    assert [o["status"] for o in lost.value.outcomes] == ["reservation_lost"]
    assert world.state("pod-a") == "hidden"  # the new owner's hide is untouched
    assert world.primitive.journal.get("pod-a")["reservation_token"] == other["token"]
    assert world.primitive.reservations.active()["m1/node-a/0"].owner == "sm-2"


def test_monolithic_sleep_resolves_a_lost_reservation_itself():
    world = World(_two_pods(), _two_desired())
    world.gateway.inflight("pod-a", "gw-1", total=1)
    world.hooks.append(lambda now: setattr(world.redis, "now_ms", world.redis.now_ms + 31_000) if 1003 <= now < 1003.6 else None)
    snapshot = world.runtime.snapshots["pod-a"]

    with pytest.raises(ReservationLost) as lost:
        world.primitive.sleep([SleepTarget(binding_of(snapshot), snapshot.pod_ip)], path="repair")

    assert [o["status"] for o in lost.value.outcomes] == ["rolled_back"]
    assert world.state("pod-a") == "awake"
    assert world.primitive.active_count() == 0


def test_renewal_redis_error_rolls_back_and_leaves_desired_untouched():
    world = World(_two_pods(), _two_desired())
    world.gateway.inflight("pod-a", "gw-1", total=1)
    original_eval = world.redis.eval
    broken = {"on": False}

    def flaky_eval(script, numkeys, *args):
        if broken["on"]:
            raise ConnectionError("redis down")
        return original_eval(script, numkeys, *args)

    world.redis.eval = flaky_eval
    world.hooks.append(lambda now: broken.update(on=True) if now >= 1003 else None)

    with pytest.raises(ConnectionError):
        world.service.put_binding_power("pod-a", awake=False)

    broken["on"] = False
    assert world.state("pod-a") == "awake"
    assert world.desired()["m1/node-a/0"][0] == "awake"
    assert not any(call[0] == "sleep" for call in world.vllm.calls)


# ------------------------------------------------------------------ P2-2
class SlowVllm(FakeVllm):
    def __init__(self, *, sleep_s=0.3, metrics_s=0.2):
        super().__init__()
        self.sleep_s = sleep_s
        self.metrics_s = metrics_s
        self.max_parallel = {"sleep": 0, "metrics": 0}
        self._active = {"sleep": 0, "metrics": 0}
        self._lock = threading.Lock()

    def _enter(self, kind):
        with self._lock:
            self._active[kind] += 1
            self.max_parallel[kind] = max(self.max_parallel[kind], self._active[kind])

    def _leave(self, kind):
        with self._lock:
            self._active[kind] -= 1

    def sleep(self, pod_ip, **kwargs):
        self._enter("sleep")
        try:
            time.sleep(self.sleep_s)
            return super().sleep(pod_ip, **kwargs)
        finally:
            self._leave("sleep")

    def metrics(self, pod_ip, *, port=None):
        self._enter("metrics")
        try:
            time.sleep(self.metrics_s)
            return super().metrics(pod_ip, port=port)
        finally:
            self._leave("metrics")


def _primitive_world(vllm, n, **policy_overrides):
    redis = FakeRedis()
    snapshots = [pod(f"pod-{i}", "m1", (i,), ip=f"10.0.0.{i + 1}") for i in range(n)]
    runtime = FakeRuntime(snapshots)
    for snapshot in snapshots:
        vllm.sleeping[snapshot.pod_ip] = False
    gateway = FakeGateway(redis, runtime)
    gateway.heartbeat("gw-1")
    gateway.auto_ack.add("gw-1")
    hooks = []
    clock = TickingClock(lambda now: [gateway.tick()] + [hook(now) for hook in hooks])
    primitive = SleepPrimitive(
        runtime_ops=runtime,
        vllm_ops=vllm,
        policy=policy(**policy_overrides),
        gateway=GatewayState(redis, monotonic=lambda: clock.monotonic()),
        journal=SleepJournal(redis),
        reservations=SleepReservations(redis),
        clock=clock,
    )
    targets = [SleepTarget(binding_of(s), s.pod_ip) for s in snapshots]
    return primitive, targets, clock, hooks, gateway


def test_many_targets_are_drained_and_committed_in_parallel():
    vllm = SlowVllm(sleep_s=0.3, metrics_s=0.2)
    primitive, targets, _clock, _hooks, _gateway = _primitive_world(vllm, 4)

    started = time.monotonic()
    outcomes = primitive.sleep(targets, path="scale_down")
    elapsed = time.monotonic() - started

    assert [o["status"] for o in outcomes] == ["slept"] * 4
    assert vllm.max_parallel == {"sleep": 4, "metrics": 4}
    # serial would be >= 4 x 0.2 (metrics) + 4 x 0.3 (/sleep) = 2.0 s
    assert elapsed < 1.2


def test_the_last_drain_round_is_decided_at_the_hard_cap():
    vllm = FakeVllm()
    primitive, targets, clock, _hooks, _gateway = _primitive_world(
        vllm, 1, hard_cap_s=10.0, budgets_s={"scale_down": 5.0}
    )
    vllm.metrics_down.add("10.0.0.1")  # drain state unknown until the hard cap
    original = vllm.metrics

    def slow_metrics(pod_ip, *, port=None):
        clock.now += 3.0  # every read costs 3 s of (virtual) probe time
        return original(pod_ip, port=port)

    vllm.metrics = slow_metrics
    started = clock.monotonic()

    with pytest.raises(Exception):
        primitive.sleep(targets, path="scale_down")

    [outcome] = primitive.recent()
    assert outcome["status"] == "rolled_back"
    # decided in the round that crossed the hard cap: no read after it
    assert clock.monotonic() - started <= 10.0 + 3.0 + 1.0


def test_commit_renews_the_reservation_while_slow_sleeps_run():
    vllm = SlowVllm(sleep_s=0.35, metrics_s=0.0)
    primitive, targets, _clock, _hooks, _gateway = _primitive_world(vllm, 2, poll_interval_s=0.1)
    renewals = []
    original = primitive.reservations.renew
    primitive.reservations.renew = lambda ids, token, *, ttl_s: renewals.append(time.monotonic()) or original(ids, token, ttl_s=ttl_s)

    batch = primitive.prepare(targets, path="scale_down")
    primitive.drain(batch)
    before = len(renewals)
    primitive.commit(batch)

    assert len(renewals) - before >= 3  # the commit start + rounds during the slow /sleep


# ------------------------------------------------------------------ P2-3
def test_binding_sleep_writes_desired_only_after_the_commit():
    world = World(_two_pods(), _two_desired())
    world.gateway.inflight("pod-a", "gw-1", total=1)
    during = []
    world.hooks.append(
        lambda now: during.append(world.desired()["m1/node-a/0"][0]) or (
            world.gateway.inflight("pod-a", "gw-1", total=0) if now > 1003 else None
        )
    )

    world.service.put_binding_power("pod-a", awake=False)

    assert during and set(during) == {"awake"}  # never "sleeping" while it drained
    assert world.desired()["m1/node-a/0"] == ("sleeping", False, "resident")


def test_failed_prepare_leaves_desired_untouched():
    world = World(_two_pods(), _two_desired())
    world.runtime.fail_hide_for.add("pod-a")

    with pytest.raises(RuntimeError):
        world.service.put_binding_power("pod-a", awake=False)
    with pytest.raises(RuntimeError):
        world.service.put_model_target("m1", wake_replicas=0)

    assert world.desired()["m1/node-a/0"][0] == "awake"


def test_failed_wake_restores_desired_power():
    world = World(_two_pods(), _two_desired())
    world.vllm.wake_up = lambda pod_ip, *, port=None: Result(False, "cuda oom")

    with pytest.raises(ValueError, match="cuda oom"):
        world.service.put_binding_power("pod-b", awake=True)
    assert world.desired()["m1/node-a/1"][0] == "sleeping"

    with pytest.raises(ValueError, match="cuda oom"):
        world.service.put_model_target("m1", wake_replicas=2)
    assert world.desired() == {
        "m1/node-a/0": ("awake", False, "resident"),
        "m1/node-a/1": ("sleeping", False, "resident"),
    }


def test_failed_hide_request_restores_desired_hidden_flag():
    world = World(
        [pod("pod-a", "m1", (0,), ip="10.0.0.1"), pod("pod-b", "m1", (1,), ip="10.0.0.2", state="hidden")],
        [_desired("m1/node-a/0", "m1", (0,), "awake"), _desired("m1/node-a/1", "m1", (1,), "awake", hidden=True)],
    )
    original = world.runtime.write_binding_annotations

    def refuse_awake(binding, *, state):
        if state == "awake":
            raise RuntimeError("patch refused")
        return original(binding, state=state)

    world.runtime.write_binding_annotations = refuse_awake

    with pytest.raises(RuntimeError):
        world.service.put_model_routable("m1", hidden_pods=["pod-a"])  # hide a, unhide b

    assert world.desired()["m1/node-a/0"][1] is False
    assert world.desired()["m1/node-a/1"][1] is True


def test_partial_shrink_marks_only_the_slept_binding_desired_asleep():
    snapshots = [pod("pod-a", "m1", (0,), ip="10.0.0.1"), pod("pod-b", "m1", (1,), ip="10.0.0.2")]
    world = World(snapshots, [_desired("m1/node-a/0", "m1", (0,), "awake"), _desired("m1/node-a/1", "m1", (1,), "awake")])
    world.vllm.fail_sleep_for.add("10.0.0.2")

    with pytest.raises(Exception):
        world.service.put_model_target("m1", wake_replicas=0)

    desired = world.desired()
    assert desired["m1/node-a/0"][0] == "sleeping"
    assert desired["m1/node-a/1"][0] == "awake"
    assert world.state("pod-b") == "awake"


# ------------------------------------------------------------------ P1-3
def test_bootstrap_with_deployments_and_pods_waiting_at_the_gate_seeds_sleeping():
    redis = FakeRedis()
    fleet = FleetStateStore(redis)
    gate = []
    for name, model, gpus, ip in (
        ("m1-a", "m1", (0,), "10.0.0.1"),
        ("m1-b", "m1", (1,), None),
        ("tp2-a", "tp2", (0, 1), "10.0.0.5"),
    ):
        snapshot = pod(name, model, gpus, ip=ip or "10.0.0.9", state="hidden")  # template annotation
        gate.append(snapshot.__class__(**{**snapshot.__dict__, "ready": False, "pod_ip": ip}))
    runtime = FakeRuntime(gate, [deployment("m1", (0,)), deployment("m1", (1,)), deployment("tp2", (0, 1))])
    vllm = FakeVllm()
    vllm.physical_override.update({"10.0.0.1": None, "10.0.0.5": None})

    with fence(redis):
        result = seed_desired(registry(), fleet, runtime_ops=runtime, vllm_ops=vllm)

    assert result["added"] == ["m1/node-a/0", "m1/node-a/1", "tp2/node-a/0,1"]
    assert {d.binding_id: (d.lifecycle, d.power, d.hidden) for d in fleet.load_desired().bindings} == {
        "m1/node-a/0": ("resident", "sleeping", False),
        "m1/node-a/1": ("resident", "sleeping", False),
        "tp2/node-a/0,1": ("resident", "sleeping", False),
    }


@pytest.mark.parametrize(
    ("ready", "admitted", "physical", "expected"),
    [
        (True, False, False, ("awake", True)),  # trusted: past the gate, Ready, probe answers
        (True, False, None, ("sleeping", False)),  # probe unknown: never the annotation
        (False, False, False, ("sleeping", False)),  # not Ready
        (True, True, False, ("sleeping", False)),  # admission still converging
        (True, False, True, ("sleeping", False)),
    ],
)
def test_seeding_trusts_a_pod_only_past_the_gate_ready_and_probed(ready, admitted, physical, expected):
    redis = FakeRedis()
    fleet = FleetStateStore(redis)
    snapshot = pod("tp2-a", "tp2", (0, 1), ip="10.0.0.5", state="hidden", admitted=admitted)
    runtime = FakeRuntime([snapshot.__class__(**{**snapshot.__dict__, "ready": ready})])
    vllm = FakeVllm()
    vllm.physical_override["10.0.0.5"] = physical

    with fence(redis):
        seed_desired(registry(), fleet, runtime_ops=runtime, vllm_ops=vllm)

    by_id = {d.binding_id: (d.power, d.hidden) for d in fleet.load_desired().bindings}
    assert by_id["tp2/node-a/0,1"] == expected


# ------------------------------------------------------------------ P2-4 / P3
def _startup_world():
    resident = pod("pod-tp2", "tp2", (0, 1), ip="10.0.0.5")
    world = World(
        [resident],
        [_desired("tp2/node-a/0,1", "tp2", (0, 1), "awake"), _desired("m1/node-a/0", "m1", (0,), "sleeping")],
    )
    world.runtime.get_startup_pod = lambda name: startup_pod(name, "m1", (0,), uid="new-uid")
    world.runtime.list_startup_resident_snapshots = lambda: world.runtime.list_pod_snapshots()
    world.runtime.admit_startup_pod = lambda name, **kwargs: None
    return world


def test_startup_admission_drains_the_overlapping_resident_outside_the_writer_lock():
    world = _startup_world()
    world.gateway.inflight("pod-tp2", "gw-1", total=1)
    held = []
    world.hooks.append(
        lambda now: held.append(world.coordinator.active is not None)
        if now < 1004
        else world.gateway.inflight("pod-tp2", "gw-1", total=0)
    )

    result = world.service.admit_startup(pod_name="m1-new", pod_uid="new-uid")

    assert held and not any(held)
    assert result["suspended_binding_ids"] == ["tp2/node-a/0,1"]
    assert world.coordinator.kinds == ["startup_admit_sleep", "startup_admit_sleep_commit", "startup_admit"]
    assert world.desired()["tp2/node-a/0,1"][0] == "awake"  # suspended, not re-targeted


def test_startup_admission_checks_reservations_under_the_writer_lock():
    world = _startup_world()
    world.vllm.sleeping["10.0.0.5"] = True
    checked = []
    original = world.primitive.reservations.assert_free

    def spy(**kwargs):
        checked.append((kwargs.get("what"), world.coordinator.active and world.coordinator.active["kind"]))
        return original(**kwargs)

    world.primitive.reservations.assert_free = spy

    world.service.admit_startup(pod_name="m1-new", pod_uid="new-uid")

    assert ("startup admission of m1-new", "startup_admit") in checked


def test_startup_admission_retries_when_the_resident_woke_again():
    world = _startup_world()
    woken = {"n": 0}
    original = world.service._sleep_overlapping_residents

    def sleep_then_wake(pod_record):
        original(pod_record)
        world.vllm.sleeping["10.0.0.5"] = False  # woke up before the admission lock
        woken["n"] += 1

    world.service._sleep_overlapping_residents = sleep_then_wake

    with pytest.raises(RetryLater):
        world.service.admit_startup(pod_name="m1-new", pod_uid="new-uid")
    assert woken["n"] == 1


def test_one_conflicting_startup_pod_does_not_abort_convergence_of_the_others():
    snapshots = [
        pod("pod-a", "m1", (0,), ip="10.0.0.1", state="hidden", admitted=True),
        pod("pod-c", "m1", (1,), ip="10.0.0.3", state="hidden", admitted=True),
    ]
    world = World(snapshots, [_desired("m1/node-a/0", "m1", (0,), "sleeping"), _desired("m1/node-a/1", "m1", (1,), "awake", hidden=True)])
    world.vllm.sleeping.update({"10.0.0.1": False, "10.0.0.3": False})
    world.runtime.list_startup_resident_snapshots = lambda: world.runtime.list_pod_snapshots()
    world.runtime.clear_startup_admission = lambda name: None
    world.service._reconcile_unlocked = lambda drop_missing=False: {}
    # pod-a's binding is being slept by someone else right now
    token = world.primitive.reservations.acquire(
        [binding_of(snapshots[0])], owner="sm-2", operation_id="x", ttl_s=300
    )

    result = world.service.converge_startups()

    assert result == {"converged": ["pod-c"], "pending": ["pod-a"]}
    world.primitive.reservations.release(["m1/node-a/0"], token)
    held = []
    world.gateway.inflight("pod-a", "gw-1", total=1)
    world.hooks.append(
        lambda now: held.append(world.coordinator.active is not None)
        if now < 1004
        else world.gateway.inflight("pod-a", "gw-1", total=0)
    )
    result = world.service.converge_startups()
    assert "pod-a" in result["converged"]
    assert held and not any(held)  # its sleep drained outside the writer lock
    assert world.vllm.sleeping["10.0.0.1"] is True


def test_retry_later_is_http_409():
    from fastapi.testclient import TestClient

    from tre_sm.api.v2 import create_app

    world = _startup_world()
    world.service.admit_startup = lambda **kwargs: (_ for _ in ()).throw(RetryLater("resident awake"))
    response = TestClient(create_app(world.service)).post(
        "/v2/startup/admit", json={"pod_name": "p", "pod_uid": "u"}
    )
    assert response.status_code == 409

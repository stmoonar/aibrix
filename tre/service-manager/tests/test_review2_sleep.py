"""Review 2 of the transparent-sleep service-manager, as it stands after the
whole-lock SM (2026-10-02: no sleep reservation, no phase outside the writer
lock).

P1-3 seeding trust, P2-2 time budget with many targets, P2-3 desired state
follows the outcome, P2-4 startup paths sleep under one writer-lock hold, P3
recovery / TOCTOU / per-pod convergence. Also the shared ``World`` of many SM
tests.
"""

from datetime import datetime, timezone
import threading
import time

import pytest

from tre_sm.api.v2 import RetryLater, ServiceManagerV2, WakeConflict
from tre_sm.ops.sleep_primitive import (
    GatewayState,
    SleepJournal,
    SleepPrimitive,
    SleepTarget,
)
from tre_sm.state.fleet_seed import seed_desired
from tre_sm.state.fleet_store import DesiredBinding, FleetStateStore
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
    StrictCoordinator,
    startup_pod,
)


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
        clock=clock,
    )
    targets = [SleepTarget(binding_of(s), s.pod_ip) for s in snapshots]
    return primitive, targets, clock, hooks, gateway


def test_many_targets_are_read_and_slept_in_parallel():
    vllm = SlowVllm(sleep_s=0.3, metrics_s=0.2)
    primitive, targets, _clock, _hooks, _gateway = _primitive_world(vllm, 4)

    started = time.monotonic()
    outcomes = primitive.sleep(targets, path="scale_down")
    elapsed = time.monotonic() - started

    assert [o["status"] for o in outcomes] == ["slept"] * 4
    assert vllm.max_parallel == {"sleep": 4, "metrics": 4}
    # serial would be >= 4 x 0.2 (metrics) + 4 x 0.3 (/sleep) = 2.0 s
    assert elapsed < 1.2


# ------------------------------------------------------------------ P2-3
def test_binding_sleep_writes_desired_only_after_the_sleep_is_confirmed():
    world = World(_two_pods(), _two_desired())
    world.gateway.ack_after_polls = 2  # a few ack polls while the sleep runs
    during = []
    world.hooks.append(lambda now: during.append(world.desired()["m1/node-a/0"][0]))

    world.service.put_binding_power("pod-a", awake=False)

    assert during and set(during) == {"awake"}  # never "sleeping" before /sleep confirmed
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


def test_startup_admission_sleeps_the_overlapping_resident_in_one_writer_lock_hold():
    world = _startup_world()
    world.gateway.ack_after_polls = 2
    held = []
    world.hooks.append(lambda now: held.append(world.coordinator.active is not None))

    result = world.service.admit_startup(pod_name="m1-new", pod_uid="new-uid")

    assert held and all(held)  # whole-lock: the sleep never runs without the lock
    assert result["suspended_binding_ids"] == ["tp2/node-a/0,1"]
    assert world.coordinator.kinds == ["startup_admit_sleep", "startup_admit"]
    assert world.desired()["tp2/node-a/0,1"][0] == "awake"  # suspended, not re-targeted


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
    # pod-a's /sleep fails this pass (the engine refuses it)
    world.vllm.fail_sleep_for.add("10.0.0.1")

    result = world.service.converge_startups()

    assert result == {"converged": ["pod-c"], "pending": ["pod-a"]}
    world.vllm.fail_sleep_for.clear()
    held = []
    world.gateway.ack_after_polls = world.gateway._polls + 2
    world.hooks.append(lambda now: held.append(world.coordinator.active is not None))
    result = world.service.converge_startups()
    assert "pod-a" in result["converged"]
    assert held and all(held)  # its sleep ran under one writer-lock hold
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

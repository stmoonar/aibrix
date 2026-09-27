"""Review 3 of the transparent-sleep service-manager: grow-only targets, per
migration defrag desired state, desired kept after a physical success, startup
admission (pre-checks, queued lock, restore, async gate), lost reservations
probed before re-routing, and the worst-case commit budget."""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from tre_common.registry import ServiceManagerConfig
from tre_sm.allocator.slots import Binding, Migration, Slot
from tre_sm.api import v2 as v2_module
from tre_sm.api.v2 import RetryLater, ServiceManagerV2, create_app
from tre_sm.ops.sleep_primitive import ReservationLost, SleepPrimitive, SleepTarget
from tre_sm.state.fleet_store import FleetStateStore
from tre_sm.state.gpu_leases import GpuLease, GpuLeaseConflict
from tre_sm.state.operations import OperationBusy
from tre_sm.state.store import StateStore

from sm_test_fakes import (
    FakeCoordinator,
    FakeRedis,
    FakeRuntime,
    LegacyRedis,
    Result,
    TickingClock,
    binding_of,
    fence,
    policy,
    pod,
    registry,
)
from test_review2_sleep import World, _desired, _startup_world, _two_desired, _two_pods


# ------------------------------------------------------------ P2-1 grow-only target
def test_at_least_target_never_shrinks_and_still_grows():
    world = World(_two_pods(), _two_desired())

    held = world.service.put_model_target("m1", wake_replicas=1, at_least=True)
    assert held["actions"] == [] and held["awake"] == 1
    grown = world.service.put_model_target("m1", wake_replicas=2, at_least=True)
    assert grown["actions"] == [{"action": "wake", "serve_id": "pod-b"}]
    again = world.service.put_model_target("m1", wake_replicas=2, at_least=True)  # a retry
    assert again["actions"] == []
    lower = world.service.put_model_target("m1", wake_replicas=1, at_least=True)
    assert lower["actions"] == []  # grown meanwhile: never shrunk by a stale target
    assert world.vllm.sleeping["10.0.0.2"] is False and world.vllm.sleeping["10.0.0.1"] is False


def test_at_least_is_accepted_by_the_http_api():
    world = World(_two_pods(), _two_desired())
    client = TestClient(create_app(world.service))
    response = client.put("/v2/models/m1/target", json={"wake_replicas": 1, "at_least": True})
    assert response.status_code == 200 and response.json()["actions"] == []


# ------------------------------------------------------------ P2-4 defrag
def _defrag_service():
    redis = FakeRedis()
    store = StateStore(LegacyRedis())
    store.save(
        [
            Binding("pod-a", "m1", Slot("node-a", (0,)), awake=True),
            Binding("pod-b", "m1", Slot("node-a", (1,)), awake=True),
        ],
        expected_version=0,
    )
    fleet = FleetStateStore(redis)
    with fence(redis):
        fleet.save_desired(
            [_desired("m1/node-a/0", "m1", (0,), "awake"), _desired("m1/node-a/1", "m1", (1,), "awake")],
            expected_version=0,
        )
    service = ServiceManagerV2(
        registry(), store, operation_coordinator=FakeCoordinator(redis), fleet_store=fleet
    )
    return service, store, fleet


def test_defrag_commits_desired_per_migration_and_a_later_failure_keeps_the_first(monkeypatch):
    service, store, fleet = _defrag_service()
    migrations = [
        Migration("pod-a", Slot("node-a", (0,)), Slot("node-a", (2,))),
        Migration("pod-b", Slot("node-a", (1,)), Slot("node-a", (3,))),
    ]
    monkeypatch.setattr(v2_module.SlotAllocator, "plan_defrag", lambda self, tp_size: list(migrations))
    original = service._run_defrag_migration

    def second_fails(binding, migration, updated_by_serve, actions):
        if migration.serve_id == "pod-b":
            raise RuntimeError("create failed")
        return original(binding, migration, updated_by_serve, actions)

    service._run_defrag_migration = second_fails

    with pytest.raises(RuntimeError, match="create failed"):
        service.defrag(tp_size=2)

    desired = {d.binding_id: (d.lifecycle, d.power) for d in fleet.load_desired().bindings}
    # the first migration completed: its desired state stays
    assert desired["m1/node-a/0"] == ("absent", "sleeping")
    assert desired["m1/node-a/2"] == ("resident", "awake")
    # the failed one is restored (source back, destination absent)
    assert desired["m1/node-a/1"] == ("resident", "awake")
    assert desired.get("m1/node-a/3", ("absent", "sleeping"))[0] == "absent"
    # and the legacy store recorded the first move
    assert {b.serve_id: b.slot.gpu_ids for b in store.load().bindings} == {"pod-a": (2,), "pod-b": (1,)}


# ------------------------------------------------------------ P3 wake guard
def test_a_physical_wake_keeps_desired_awake_when_the_store_save_fails():
    world = World(_two_pods(), _two_desired())

    def broken_save(*_args, **_kwargs):
        raise ConnectionError("redis down")

    world.store.save = broken_save

    with pytest.raises(ConnectionError):
        world.service.put_binding_power("pod-b", awake=True)

    assert world.vllm.sleeping["10.0.0.2"] is False  # physically awake
    assert world.desired()["m1/node-a/1"][0] == "awake"  # not rolled back to sleeping


def test_a_partial_model_target_growth_keeps_the_woken_binding_desired_awake():
    snapshots = [
        pod("pod-a", "m1", (0,), ip="10.0.0.1", state="sleeping"),
        pod("pod-b", "m1", (1,), ip="10.0.0.2", state="sleeping"),
    ]
    world = World(snapshots, [_desired("m1/node-a/0", "m1", (0,), "sleeping"), _desired("m1/node-a/1", "m1", (1,), "sleeping")])
    calls = {"n": 0}
    original = world.vllm.wake_up

    def second_wake_fails(pod_ip, *, port=None):
        calls["n"] += 1
        if calls["n"] == 2:
            return Result(False, "cuda oom")
        return original(pod_ip, port=port)

    world.vllm.wake_up = second_wake_fails

    with pytest.raises(ValueError, match="cuda oom"):
        world.service.put_model_target("m1", wake_replicas=2)

    powers = world.desired()
    woken = [ip for ip, asleep in world.vllm.sleeping.items() if asleep is False]
    assert len(woken) == 1
    awake_id = "m1/node-a/0" if woken == ["10.0.0.1"] else "m1/node-a/1"
    other_id = "m1/node-a/1" if awake_id == "m1/node-a/0" else "m1/node-a/0"
    assert powers[awake_id][0] == "awake" and powers[other_id][0] == "sleeping"


# ------------------------------------------------------------ P2-5 startup admission
class _Lease:
    def __init__(self, binding_id, gpus, phase):
        self.binding_id, self.node, self.gpu_ids, self.phase = binding_id, "node-a", tuple(gpus), phase


def test_admission_refuses_a_transient_lease_before_sleeping_anything():
    world = _startup_world()
    world.leases.load = lambda: [_Lease("tp2/node-a/0,1-starting", (0,), "starting")]

    with pytest.raises(GpuLeaseConflict):
        world.service.admit_startup(pod_name="m1-new", pod_uid="new-uid")

    assert not any(call[0] == "sleep" for call in world.vllm.calls)
    assert world.state("pod-tp2") == "awake"
    assert world.coordinator.kinds == []  # not even the writer lock


def test_admission_queues_for_the_writer_lock():
    world = _startup_world()
    world.vllm.sleeping["10.0.0.5"] = True
    waits = []
    original = world.coordinator.operation

    def recording(kind, *, request=None, wait_s=0.0):
        waits.append((kind, wait_s))
        return original(kind, request=request, wait_s=wait_s)

    world.coordinator.operation = recording

    world.service.admit_startup(pod_name="m1-new", pod_uid="new-uid")

    assert ("startup_admit", world.service._sm_config.writer_lock_wait_s) in waits
    assert world.service._sm_config.writer_lock_wait_s > 0


def test_a_failed_admission_wakes_the_residents_it_put_to_sleep():
    world = _startup_world()
    world.coordinator.refuse = {"startup_admit"}  # the admission commit cannot get the lock

    with pytest.raises(OperationBusy):
        world.service.admit_startup(pod_name="m1-new", pod_uid="new-uid")

    sleeps = [call for call in world.vllm.calls if call[0] == "sleep"]
    wakes = [call for call in world.vllm.calls if call[0] == "wake_up"]
    assert sleeps and wakes  # slept for the admission, then woken back
    assert world.vllm.sleeping["10.0.0.5"] is False
    assert world.desired()["tp2/node-a/0,1"][0] == "awake"
    assert {b.binding_id: b.awake for b in world.store.load().bindings}["tp2/node-a/0,1"] is True
    assert "startup_admit_restore" in world.coordinator.kinds


def test_startup_gate_gets_202_while_the_admission_runs_then_its_result(monkeypatch):
    world = _startup_world()
    monkeypatch.setattr(v2_module, "ADMISSION_SYNC_WAIT_S", 0.05)
    release = threading.Event()
    started = []

    def slow_admission(*, pod_name, pod_uid):
        started.append(pod_name)
        release.wait(5)
        return {"status": "admitted", "binding_id": "m1/node-a/0"}

    world.service.admit_startup = slow_admission
    client = TestClient(create_app(world.service))
    body = {"pod_name": "m1-new", "pod_uid": "new-uid"}

    first = client.post("/v2/startup/admit", json=body)
    second = client.post("/v2/startup/admit", json=body)
    assert (first.status_code, second.status_code) == (202, 202)
    assert first.json()["status"] == "in_progress"
    assert started == ["m1-new"]  # one job, not one per poll

    release.set()
    for _ in range(100):
        done = client.post("/v2/startup/admit", json=body)
        if done.status_code == 200:
            break
        time.sleep(0.02)
    assert done.status_code == 200 and done.json()["status"] == "admitted"
    assert started == ["m1-new"]


def test_startup_gate_gets_an_admission_error_once(monkeypatch):
    world = _startup_world()
    monkeypatch.setattr(v2_module, "ADMISSION_SYNC_WAIT_S", 0.05)
    release = threading.Event()
    calls = []

    def failing(*, pod_name, pod_uid):
        calls.append(1)
        release.wait(5)
        raise RetryLater("resident awake again")

    world.service.admit_startup = failing
    client = TestClient(create_app(world.service))
    body = {"pod_name": "m1-new", "pod_uid": "new-uid"}
    assert client.post("/v2/startup/admit", json=body).status_code == 202
    release.set()
    for _ in range(100):
        response = client.post("/v2/startup/admit", json=body)
        if response.status_code != 202:
            break
        time.sleep(0.02)
    assert response.status_code == 409
    # the next poll starts a fresh admission
    assert client.post("/v2/startup/admit", json=body).status_code == 409 and len(calls) == 2


# ------------------------------------------------------------ P3 resolve_lost
def _lose_reservation_while_draining(world, *, physical):
    world.gateway.inflight("pod-a", "gw-1", total=1)
    done = {}

    def expire(now):
        if now < 1003 or done:
            return
        done["x"] = True
        world.redis.now_ms += 31_000  # the reservation expired meanwhile
        world.vllm.physical_override["10.0.0.1"] = physical  # e.g. slept by another owner

    world.hooks.append(expire)


def test_a_lost_reservation_on_a_pod_found_asleep_does_not_reopen_routing():
    world = World(_two_pods(), _two_desired())
    _lose_reservation_while_draining(world, physical=True)

    with pytest.raises(ReservationLost) as lost:
        world.service.put_binding_power("pod-a", awake=False)

    assert [o["status"] for o in lost.value.outcomes] == ["slept"]
    assert world.state("pod-a") == "sleeping"  # never re-routed
    assert not any(call[0] == "sleep" for call in world.vllm.calls)  # not our /sleep
    assert world.primitive.journal.stats().get("sleeps_total", 0) == 0


def test_a_lost_reservation_on_a_pod_of_unknown_state_stays_hidden():
    world = World(_two_pods(), _two_desired())
    _lose_reservation_while_draining(world, physical=None)

    with pytest.raises(ReservationLost) as lost:
        world.service.put_binding_power("pod-a", awake=False)

    assert [o["status"] for o in lost.value.outcomes] == ["unconfirmed"]
    assert world.state("pod-a") == "hidden"


def test_a_lost_reservation_on_an_awake_pod_is_still_rolled_back():
    world = World(_two_pods(), _two_desired())
    _lose_reservation_while_draining(world, physical=False)

    with pytest.raises(ReservationLost) as lost:
        world.service.put_binding_power("pod-a", awake=False)

    assert [o["status"] for o in lost.value.outcomes] == ["rolled_back"]
    assert world.state("pod-a") == "awake"


# ------------------------------------------------------------ P3 time budget
class TimedVllm:
    """Every vLLM call costs its full timeout on the virtual clock (worst case)."""

    def __init__(self, clock, *, probe_s, sleep_s, abort_succeeds):
        self.clock, self.probe_s, self.sleep_s = clock, probe_s, sleep_s
        self.abort_succeeds = abort_succeeds

    def version(self, pod_ip, *, port=None):
        self.clock.now += self.probe_s
        return "0.30.0"

    def sleep(self, pod_ip, *, port=None, mode=None, timeout_s=None, hidden=False):
        self.clock.now += self.sleep_s
        return Result(bool(self.abort_succeeds and mode == "abort"), "timed out")

    def is_sleeping(self, pod_ip, *, port=None):
        self.clock.now += self.probe_s
        return False  # never converges

    def metrics(self, pod_ip, *, port=None):
        self.clock.now += self.probe_s
        return "vllm:num_requests_running 0.0\nvllm:num_requests_waiting 0.0\n"


@pytest.mark.parametrize("abort_succeeds", [False, True])
def test_the_commit_phase_stays_within_the_documented_worst_case(abort_succeeds):
    sleep_policy = policy(
        vllm_sleep_mode_param="auto", probe_timeout_s=5.0, sleep_call_timeout_s=45.0,
        physical_confirm_timeout_s=15.0, poll_interval_s=0.5, no_plugin_grace_s=0.0,
    )
    config = ServiceManagerConfig(sleep=sleep_policy)
    clock = TickingClock()
    snapshot = pod("pod-a", "m1", (0,), ip="10.0.0.1")
    primitive = SleepPrimitive(
        runtime_ops=FakeRuntime([snapshot]),
        vllm_ops=TimedVllm(clock, probe_s=5.0, sleep_s=45.0, abort_succeeds=abort_succeeds),
        policy=sleep_policy,
        clock=clock,
    )
    batch = primitive.prepare([SleepTarget(binding_of(snapshot), snapshot.pod_ip)], path="scale_down")
    primitive.drain(batch)
    started = clock.now
    with pytest.raises(Exception):
        primitive.commit(batch)
    elapsed = clock.now - started
    if not abort_succeeds:
        # send failure: 5 probes (incl. the rollback re-probe) + 2 sleeps
        assert elapsed == pytest.approx(5 * 5 + 2 * 45)
    else:
        # abort accepted but never confirmed: 3 probes + 2 sleeps, the
        # confirmation window with its overshooting round, the rollback re-probe
        assert 3 * 5 + 2 * 45 + 15 + 5 < elapsed
    assert elapsed <= config.worst_case_commit_s()

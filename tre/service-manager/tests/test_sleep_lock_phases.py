"""Review P1-3: the drain runs outside the SM writer lock, fenced by a reservation."""

from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from tre_sm.api.v2 import ServiceManagerV2
from tre_sm.ops.sleep_primitive import GatewayState, SleepJournal
from tre_sm.state.fleet_store import DesiredBinding, FleetStateStore
from tre_sm.state.operations import OperationBusy
from tre_sm.state.sleep_reservations import ReservationConflict, SleepReservations
from tre_sm.state.store import StateStore

from sm_test_fakes import (
    FakeGateway,
    FakeHandle,
    FakeLeases,
    FakeRedis,
    FakeRuntime,
    FakeSafety,
    FakeVllm,
    LegacyRedis,
    TickingClock,
    binding_of,
    fence,
    pod,
    registry,
)


class StrictCoordinator:
    """Non-reentrant writer lock: entering while held raises OperationBusy."""

    owner = "sm-test"

    def __init__(self, redis, *, refuse=()):
        self.redis = redis
        self.active = None
        self.kinds = []
        self.refuse = set(refuse)

    @contextmanager
    def operation(self, kind, *, request=None, wait_s=0.0):
        if self.active is not None or kind in self.refuse:
            raise OperationBusy(f"held by {self.active and self.active['kind']}")
        self.kinds.append(kind)
        operation_id = f"{kind}-{len(self.kinds)}"
        self.active = {"operation_id": operation_id, "kind": kind, "status": "running"}
        from tre_sm.state.operations import _CURRENT_OPERATION

        token = _CURRENT_OPERATION.set(FakeHandle(operation_id))
        try:
            with fence(self.redis, operation_id):
                yield FakeHandle(operation_id)
        finally:
            _CURRENT_OPERATION.reset(token)
            self.active = None

    def active_operation(self, *, kind=None):
        return self.active

    def stale_running_operations(self, *, kind=None):
        return []

    def list_operations(self, *, limit=100):
        return []


def _desired(binding_id, model, gpus, power):
    now = datetime.now(timezone.utc).isoformat()
    return DesiredBinding(binding_id, model, "node-a", tuple(gpus), "resident", power, False, 1, now, "t", "t")


class World:
    def __init__(self, *, refuse=()):
        self.redis = FakeRedis()
        self.snapshots = [
            pod("pod-a", "m1", (0,), ip="10.0.0.1"),
            pod("pod-b", "m1", (1,), ip="10.0.0.2", state="sleeping"),
        ]
        self.runtime = FakeRuntime(self.snapshots)
        self.vllm = FakeVllm()
        self.vllm.sleeping.update({"10.0.0.1": False, "10.0.0.2": True})
        self.gateway = FakeGateway(self.redis, self.runtime)
        self.gateway.heartbeat("gw-1")
        self.gateway.auto_ack.add("gw-1")
        self.hooks = []
        self.clock = TickingClock(self._tick)
        self.store = StateStore(LegacyRedis())
        self.store.save([binding_of(s) for s in self.snapshots], expected_version=0)
        self.fleet = FleetStateStore(self.redis)
        with fence(self.redis):
            self.fleet.save_desired(
                [_desired("m1/node-a/0", "m1", (0,), "awake"), _desired("m1/node-a/1", "m1", (1,), "sleeping")],
                expected_version=0,
            )
        self.coordinator = StrictCoordinator(self.redis, refuse=refuse)
        self.leases = FakeLeases()
        self.service = ServiceManagerV2(
            registry(),
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


def test_drain_runs_outside_the_writer_lock_and_the_reservation_fences_conflicts():
    world = World()
    world.gateway.inflight("pod-a", "gw-1", total=1)
    seen = {}

    def during_drain(now):
        if now < 1003 or seen:
            return
        seen["lock_held"] = world.coordinator.active is not None
        # An unrelated write goes through while pod-a drains ...
        seen["unrelated"] = world.service.put_model_routable("tp2", hidden_pods=[])["actions"]
        # ... but anything touching the draining binding or its model is fenced.
        for name, call in {
            "wake": lambda: world.service.put_binding_power("pod-a", awake=True),
            "target": lambda: world.service.put_model_target("m1", wake_replicas=2),
            "hide": lambda: world.service.put_model_routable("m1", hidden_pods=["pod-a"]),
        }.items():
            try:
                call()
                seen[name] = "allowed"
            except ReservationConflict:
                seen[name] = "reserved"
        seen["reservations"] = sorted(world.service._sleep_primitive.reservations.active())
        seen["audit_orphans"] = [
            issue for issue in world.service._sleep_journal_issues() if issue["code"] == "sleep_operation_orphaned"
        ]
        world.gateway.inflight("pod-a", "gw-1", total=0)

    world.hooks.append(during_drain)

    result = world.service.put_binding_power("pod-a", awake=False)

    assert seen == {
        "lock_held": False,
        "unrelated": [],
        "wake": "reserved",
        "target": "reserved",
        "hide": "reserved",
        "reservations": ["m1/node-a/0"],
        "audit_orphans": [],  # a drain in progress is not an orphan
    }
    # hide phase, then the writes attempted during the drain (each took and released
    # the lock), then the commit phase.
    assert world.coordinator.kinds == [
        "put_binding_power",
        "put_model_routable",
        "put_binding_power",
        "put_model_target",
        "put_model_routable",
        "put_binding_power_commit",
    ]
    assert result["actions"] == [{"action": "sleep", "serve_id": "pod-a"}]
    assert {b.serve_id: b.awake for b in world.store.load().bindings} == {"pod-a": False, "pod-b": False}
    assert ("release", "m1/node-a/0") in world.leases.calls
    assert world.service._sleep_primitive.reservations.active() == {}


def test_model_target_shrink_also_drains_outside_the_lock():
    world = World()
    world.gateway.inflight("pod-a", "gw-1", total=1)
    held = []
    world.hooks.append(lambda now: held.append(world.coordinator.active is not None) if now < 1004 else world.gateway.inflight("pod-a", "gw-1", total=0))

    result = world.service.put_model_target("m1", wake_replicas=0)

    assert held and not any(held)
    assert result["actions"] == [{"action": "sleep", "serve_id": "pod-a"}]
    assert world.coordinator.kinds == ["put_model_target", "put_model_target_commit"]
    assert result["version"] == world.store.load().version


def test_commit_without_the_writer_lock_rolls_the_hide_back():
    world = World(refuse={"put_binding_power_commit"})

    with pytest.raises(OperationBusy):
        world.service.put_binding_power("pod-a", awake=False)

    assert world.runtime.snapshots["pod-a"].annotations["tre.aibrix.io/state"] == "awake"
    assert not any(call[0] == "sleep" for call in world.vllm.calls)
    desired = {d.binding_id: d.power for d in world.fleet.load_desired().bindings}
    assert desired["m1/node-a/0"] == "awake"  # restored under the rollback phase
    assert world.service._sleep_primitive.reservations.active() == {}
    assert world.coordinator.kinds == ["put_binding_power", "put_binding_power_rollback"]


def test_drain_failure_restores_desired_power_under_the_lock():
    world = World()
    world.gateway.auto_ack.clear()  # the gateway never acks -> rollback

    with pytest.raises(Exception):
        world.service.put_binding_power("pod-a", awake=False)

    desired = {d.binding_id: d.power for d in world.fleet.load_desired().bindings}
    assert desired["m1/node-a/0"] == "awake"
    assert world.runtime.snapshots["pod-a"].annotations["tre.aibrix.io/state"] == "awake"
    assert world.coordinator.kinds == ["put_binding_power", "put_binding_power_rollback"]


def test_repair_and_startup_admission_wait_for_draining_bindings():
    world = World()
    world.gateway.inflight("pod-a", "gw-1", total=1)
    seen = {}

    def during_drain(now):
        if now < 1003 or seen:
            return
        for name, call in {
            "repair": lambda: world.service.start_fleet_repair(),
            "admit": lambda: world.service.admit_startup(pod_name="tp2-x", pod_uid="u"),
        }.items():
            try:
                call()
                seen[name] = "allowed"
            except ReservationConflict:
                seen[name] = "reserved"
            except Exception as exc:  # pragma: no cover - diagnostic
                seen[name] = repr(exc)
        world.gateway.inflight("pod-a", "gw-1", total=0)

    from sm_test_fakes import startup_pod

    world.runtime.get_startup_pod = lambda name: startup_pod(name, "tp2", (0, 1), uid="u")
    world.runtime.list_startup_resident_snapshots = lambda: world.runtime.list_pod_snapshots()
    world.runtime.admit_startup_pod = lambda name, **kwargs: None
    world.service._fleet_repair = object()  # configured
    world.hooks.append(during_drain)

    world.service.put_binding_power("pod-a", awake=False)

    assert seen == {"repair": "reserved", "admit": "reserved"}

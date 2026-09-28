"""B2: the observed fleet state follows every power operation.

The audit compares desired vs observed (and GPU leases vs observed power). The
observed snapshot used to be written only by POST /v2/reconcile, so right after
every sleep / wake the audit reported false ``desired_power_mismatch`` /
``gpu_lease_*`` / ``awake_without_gpu_lease`` issues until someone reconciled.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from tre_sm.api.v2 import ServiceManagerV2
from tre_sm.ops.sleep_primitive import GatewayState, SleepFailed, SleepJournal
from tre_sm.server import K8sPodClientFromOps
from tre_sm.state.fleet_store import DesiredBinding, FleetStateStore
from tre_sm.state.reconcile import observe_bindings
from tre_sm.state.sleep_reservations import SleepReservations
from tre_sm.state.store import StateStore

from sm_test_fakes import (
    FakeCoordinator,
    FakeGateway,
    FakeRedis,
    FakeRuntime,
    FakeSafety,
    FakeVllm,
    LegacyRedis,
    TickingClock,
    binding_of,
    deployment,
    fence,
    pod,
    registry,
)


@dataclass(frozen=True)
class Lease:
    binding_id: str
    node: str
    gpu_ids: tuple
    phase: str


class TrackingLeases:
    """GPU leases the audit can compare with observed power."""

    def __init__(self, bindings):
        self.leases = {
            b.binding_id: Lease(b.binding_id, b.slot.node, b.slot.gpu_ids, "awake")
            for b in bindings
            if b.awake
        }

    def acquire(self, binding, *, phase):
        self.leases[binding.binding_id] = Lease(
            binding.binding_id, binding.slot.node, binding.slot.gpu_ids, phase
        )

    def release(self, binding):
        self.leases.pop(binding.binding_id, None)

    def load(self):
        return list(self.leases.values())


def _desired(binding_id, gpus, power, *, model="m1", lifecycle="resident"):
    now = datetime.now(timezone.utc).isoformat()
    return DesiredBinding(binding_id, model, "node-a", tuple(gpus), lifecycle, power, False, 1, now, "t", "t")


class FlakyPods(K8sPodClientFromOps):
    def __init__(self, topology, ops):
        super().__init__(topology, ops)
        self.fail = False

    def list_pods(self):
        if self.fail:
            raise ConnectionError("k8s API down")
        return super().list_pods()


class World:
    def __init__(self):
        self.redis = FakeRedis()
        snapshots = [
            pod("pod-a", "m1", (0,), ip="10.0.0.1"),
            pod("pod-b", "m1", (1,), ip="10.0.0.2", state="sleeping"),
        ]
        self.runtime = FakeRuntime(snapshots, [deployment("m1", (0,)), deployment("m1", (1,))])
        self.vllm = FakeVllm()
        self.vllm.sleeping.update({"10.0.0.1": False, "10.0.0.2": True})
        self.gateway = FakeGateway(self.redis, self.runtime)
        self.gateway.heartbeat("gw-1")
        self.gateway.auto_ack.add("gw-1")
        self.clock = TickingClock(lambda now: self.gateway.tick())
        bindings = [binding_of(s) for s in snapshots]
        self.store = StateStore(LegacyRedis())
        self.store.save(bindings, expected_version=0)
        self.fleet = FleetStateStore(self.redis)
        with fence(self.redis):
            self.fleet.save_desired(
                [
                    _desired("m1/node-a/0", (0,), "awake"),
                    _desired("m1/node-a/1", (1,), "sleeping"),
                    # the registry's tp2 binding is not deployed in this world
                    _desired("tp2/node-a/0,1", (0, 1), "sleeping", model="tp2", lifecycle="absent"),
                ],
                expected_version=0,
            )
        self.leases = TrackingLeases(bindings)
        self.pods = FlakyPods(registry().topology(), self.runtime)
        self.service = ServiceManagerV2(
            registry(),
            self.store,
            k8s_client=self.pods,
            runtime_ops=self.runtime,
            vllm_ops=self.vllm,
            operation_coordinator=FakeCoordinator(self.redis),
            safety_gate=FakeSafety(),
            fleet_store=self.fleet,
            gpu_leases=self.leases,
            gateway_state=GatewayState(self.redis, monotonic=lambda: self.clock.monotonic()),
            sleep_journal=SleepJournal(self.redis),
            sleep_reservations=SleepReservations(self.redis),
            sleep_clock=self.clock,
        )
        # The observed snapshot starts in sync (a reconcile after start).
        self.service.reconcile()
        assert self.issues() == []

    def issues(self):
        return self.service.audit()["issues"]

    def observed(self):
        return {item.binding_id: item for item in self.fleet.load_observed().bindings}


def test_audit_is_clean_right_after_a_sleep_without_a_reconcile():
    world = World()

    world.service.put_binding_power("pod-a", awake=False)

    assert world.observed()["m1/node-a/0"].physical_power == "sleeping"
    assert world.issues() == []


def test_audit_is_clean_right_after_a_wake_without_a_reconcile():
    world = World()

    world.service.put_binding_power("pod-b", awake=True)

    observed = world.observed()["m1/node-a/1"]
    assert (observed.physical_power, observed.hidden, observed.routable) == ("awake", False, True)
    assert world.issues() == []


def test_audit_is_clean_after_a_sleep_then_a_wake_of_the_same_binding():
    world = World()

    world.service.put_binding_power("pod-a", awake=False)
    world.service.put_binding_power("pod-a", awake=True)

    assert world.observed()["m1/node-a/0"].physical_power == "awake"
    assert world.issues() == []


def test_audit_is_clean_after_a_model_target_shrink_and_growth():
    world = World()

    world.service.put_model_target("m1", wake_replicas=0)
    assert world.issues() == []

    world.service.put_model_target("m1", wake_replicas=2)
    assert {b.physical_power for b in world.observed().values()} == {"awake"}
    assert world.issues() == []


def test_a_rolled_back_sleep_refreshes_the_observed_state_too():
    world = World()
    world.vllm.fail_sleep_for.add("10.0.0.1")
    # A stale observed record (e.g. a reconcile during the drain saw it hidden).
    world.service.put_model_routable("m1", hidden_pods=["pod-a"])
    world.service.put_model_routable("m1", hidden_pods=[])
    assert world.issues() == []

    with pytest.raises(SleepFailed):
        world.service.put_binding_power("pod-a", awake=False)

    observed = world.observed()["m1/node-a/0"]
    assert (observed.physical_power, observed.hidden) == ("awake", False)
    assert world.issues() == []


def test_hide_and_unhide_refresh_the_observed_hidden_flag():
    world = World()

    world.service.put_model_routable("m1", hidden_pods=["pod-a"])
    assert world.observed()["m1/node-a/0"].hidden is True
    assert world.issues() == []

    world.service.put_model_routable("m1", hidden_pods=[])
    assert world.observed()["m1/node-a/0"].hidden is False
    assert world.issues() == []


def test_a_failing_observe_never_fails_the_power_operation(caplog):
    world = World()
    world.pods.fail = True

    with caplog.at_level("WARNING", logger="tre_sm.api.v2"):
        world.service.put_binding_power("pod-a", awake=False)

    assert world.vllm.sleeping["10.0.0.1"] is True
    assert "refreshing the observed state" in caplog.text
    world.pods.fail = False
    world.service.reconcile()  # the next reconcile catches up
    assert world.issues() == []


def test_a_targeted_refresh_keeps_other_records_and_drops_a_gone_pod():
    world = World()
    before = world.observed()
    del world.runtime.snapshots["pod-a"]  # pod-a was deleted (e.g. defrag source)

    with fence(world.redis):  # like every observed write: under the writer fence
        world.service._refresh_observed(["m1/node-a/0"])

    after = world.observed()
    assert "m1/node-a/0" not in after
    assert after["m1/node-a/1"] == before["m1/node-a/1"]


def test_observe_bindings_uses_the_physical_probe_like_reconcile():
    world = World()
    world.vllm.physical_override["10.0.0.2"] = None  # unreachable

    observations = {
        o.binding.binding_id: o
        for o in observe_bindings(
            world.pods, {"m1/node-a/0", "m1/node-a/1"}, prober=world.service._pod_prober()
        )
    }

    assert observations["m1/node-a/0"].physical_awake is True
    unknown = observations["m1/node-a/1"]
    assert unknown.physical_awake is None and unknown.binding.hidden is True  # quarantined
    assert observe_bindings(world.pods, {"m1/node-a/9"}) == []

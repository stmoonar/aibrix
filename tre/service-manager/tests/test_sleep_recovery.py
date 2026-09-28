"""Review P1-2: crash recovery of in-flight sleeps, SIGTERM rollback, audit masking."""

from datetime import datetime, timezone
import os
import signal

import pytest
from fastapi.testclient import TestClient

from tre_sm.api.v2 import ServiceManagerV2, create_app
from tre_sm.app import install_lifecycle, install_sigterm_hook
from tre_sm.ops.sleep_primitive import GatewayState, SleepJournal
from tre_sm.server import K8sPodClientFromOps
from tre_sm.state.fleet_store import DesiredBinding, FleetStateStore
from tre_sm.state.sleep_reservations import SleepReservations
from tre_sm.state.store import StateStore

from sm_test_fakes import (
    FakeCoordinator,
    FakeGateway,
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


def _desired(binding_id, model, gpus, power, hidden=False):
    now = datetime.now(timezone.utc).isoformat()
    return DesiredBinding(binding_id, model, "node-a", tuple(gpus), "resident", power, hidden, 1, now, "t", "t")


class World:
    """A service-manager restarted after a crash: pods + a left-over sleep journal."""

    def __init__(self, snapshots, desired, *, physical):
        self.redis = FakeRedis()
        self.runtime = FakeRuntime(snapshots)
        self.vllm = FakeVllm()
        self.vllm.sleeping.update(physical)
        self.store = StateStore(LegacyRedis())
        self.store.save([binding_of(s) for s in snapshots], expected_version=0)
        self.fleet = FleetStateStore(self.redis)
        with fence(self.redis):
            self.fleet.save_desired(desired, expected_version=0)
        self.journal = SleepJournal(self.redis)
        self.reservations = SleepReservations(self.redis)
        self.leases = FakeLeases()
        self.gateway = FakeGateway(self.redis, self.runtime)
        self.clock = TickingClock(lambda now: self.gateway.tick())
        self.service = ServiceManagerV2(
            registry(),
            self.store,
            k8s_client=K8sPodClientFromOps(registry().topology(), self.runtime),
            runtime_ops=self.runtime,
            vllm_ops=self.vllm,
            operation_coordinator=FakeCoordinator(self.redis),
            safety_gate=FakeSafety(),
            fleet_store=self.fleet,
            gpu_leases=self.leases,
            gateway_state=GatewayState(self.redis, monotonic=lambda: self.clock.monotonic()),
            sleep_journal=self.journal,
            sleep_reservations=self.reservations,
            sleep_clock=self.clock,
        )

    def left_over(self, pod_name, binding_id, *, phase, previous_state="awake", token="dead-token", ip=None):
        self.journal.begin(
            pod_name,
            {
                "binding_id": binding_id,
                "serve_id": pod_name,
                "phase": phase,
                "previous_state": previous_state,
                "operation_id": "dead-op",
                "reservation_token": token,
                "pod_ip": ip,
            },
        )

    def state(self, name):
        return self.runtime.snapshots[name].annotations["tre.aibrix.io/state"]


@pytest.mark.parametrize("phase", ["hiding", "awaiting_ack", "draining", "drained", "drain_budget_spent"])
def test_bootstrap_rolls_back_an_awake_pod_from_any_phase_before_sleep(phase):
    hidden = pod("pod-a", "m1", (0,), ip="10.0.0.1", state="hidden")
    world = World([hidden], [_desired("m1/node-a/0", "m1", (0,), "sleeping")], physical={"10.0.0.1": False})
    world.left_over("pod-a", "m1/node-a/0", phase=phase)

    result = world.service.recover_sleep_journal()

    assert result["resolved"] == [{"serve_id": "pod-a", "result": "rolled_back_to_awake"}]
    assert world.state("pod-a") == "awake"  # routable again, under a new route-gen
    assert world.runtime.patches[-1][2] > 0
    assert world.journal.entries() == {}
    assert {d.binding_id: d.power for d in world.fleet.load_desired().bindings} == {"m1/node-a/0": "awake"}
    assert not any(call[0] == "sleep" for call in world.vllm.calls)


def test_bootstrap_records_a_pod_that_did_fall_asleep():
    hidden = pod("pod-a", "m1", (0,), ip="10.0.0.1", state="hidden")
    world = World([hidden], [_desired("m1/node-a/0", "m1", (0,), "sleeping")], physical={"10.0.0.1": True})
    world.left_over("pod-a", "m1/node-a/0", phase="sleeping")

    result = world.service.recover_sleep_journal()

    assert result["resolved"] == [{"serve_id": "pod-a", "result": "slept"}]
    assert world.state("pod-a") == "sleeping"
    assert ("release", "m1/node-a/0") in world.leases.calls
    assert world.store.load().bindings[0].awake is False
    assert world.journal.entries() == {}


def test_bootstrap_keeps_a_safescale_probe_hidden():
    probe = pod("pod-a", "m1", (0,), ip="10.0.0.1", state="hidden")
    world = World([probe], [_desired("m1/node-a/0", "m1", (0,), "sleeping")], physical={"10.0.0.1": False})
    world.left_over("pod-a", "m1/node-a/0", phase="draining", previous_state="hidden")

    world.service.recover_sleep_journal()

    assert world.state("pod-a") == "hidden"
    desired = world.fleet.load_desired().bindings[0]
    assert (desired.power, desired.hidden) == ("awake", True)


def test_unknown_physical_state_stays_hidden_and_is_audited():
    hidden = pod("pod-a", "m1", (0,), ip="10.0.0.1", state="hidden")
    world = World([hidden], [_desired("m1/node-a/0", "m1", (0,), "sleeping")], physical={})
    world.vllm.physical_override["10.0.0.1"] = None
    world.left_over("pod-a", "m1/node-a/0", phase="sleep_unconfirmed")

    result = world.service.recover_sleep_journal()

    assert result["kept"] == [{"serve_id": "pod-a", "result": "physical_state_unknown"}]
    assert world.state("pod-a") == "hidden"
    assert world.journal.entries()["pod-a"]["recovery_attempts"] == 1
    codes = [issue["code"] for issue in world.service.audit()["issues"]]
    assert "sleep_unconfirmed" in codes
    assert "hidden_without_operation" not in codes  # the journal accounts for it

    # Once the engine answers again, the next pass resolves it.
    world.vllm.physical_override.pop("10.0.0.1")
    world.vllm.sleeping["10.0.0.1"] = True
    assert world.service.recover_sleep_journal()["resolved"] == [{"serve_id": "pod-a", "result": "slept"}]


def test_a_vanished_pod_just_drops_the_entry():
    world = World([], [_desired("m1/node-a/0", "m1", (0,), "sleeping")], physical={})
    world.left_over("pod-gone", "m1/node-a/0", phase="draining")

    assert world.service.recover_sleep_journal()["resolved"] == [{"serve_id": "pod-gone", "result": "pod_gone"}]
    assert world.journal.entries() == {}


def test_a_drain_with_a_live_reservation_is_left_alone():
    hidden = pod("pod-a", "m1", (0,), ip="10.0.0.1", state="hidden")
    world = World([hidden], [_desired("m1/node-a/0", "m1", (0,), "sleeping")], physical={"10.0.0.1": False})
    token = world.reservations.acquire([binding_of(hidden)], owner="other-sm", operation_id="op", ttl_s=30)
    world.left_over("pod-a", "m1/node-a/0", phase="draining", token=token)

    assert world.service.recover_sleep_journal() == {"resolved": [], "kept": []}
    assert world.state("pod-a") == "hidden"

    world.redis.now_ms += 60_000  # its owner died: the reservation expires
    assert world.service.recover_sleep_journal()["resolved"][0]["result"] == "rolled_back_to_awake"


def test_startup_event_runs_the_recovery():
    hidden = pod("pod-a", "m1", (0,), ip="10.0.0.1", state="hidden")
    world = World([hidden], [_desired("m1/node-a/0", "m1", (0,), "sleeping")], physical={"10.0.0.1": False})
    world.left_over("pod-a", "m1/node-a/0", phase="awaiting_ack")
    app = create_app(world.service)
    install_lifecycle(app, world.service)

    with TestClient(app):
        pass

    assert world.state("pod-a") == "awake"
    assert world.journal.entries() == {}


# --------------------------------------------------------------------- SIGTERM
def test_sigterm_rolls_back_a_drain_in_progress_and_refuses_new_sleeps():
    awake = pod("pod-a", "m1", (0,), ip="10.0.0.1")
    world = World([awake], [_desired("m1/node-a/0", "m1", (0,), "awake")], physical={"10.0.0.1": False})
    world.gateway.heartbeat("gw-1")
    world.gateway.auto_ack.add("gw-1")
    world.gateway.inflight("pod-a", "gw-1", total=1)  # a long request keeps it draining
    world.clock.hooks.append(lambda now: world.service.begin_shutdown() if now > 1003 else None)
    client = TestClient(create_app(world.service))

    response = client.put("/v2/bindings/pod-a/power", json={"awake": False})

    assert response.status_code == 409 and response.json()["error"] == "SleepCancelled"
    assert world.state("pod-a") == "awake"
    assert not any(call[0] == "sleep" for call in world.vllm.calls)
    assert world.fleet.load_desired().bindings[0].power == "awake"
    assert world.service.shutdown(timeout_s=0.1) is True
    assert client.put("/v2/bindings/pod-a/power", json={"awake": False}).status_code == 503


def test_sigterm_hook_chains_to_the_previous_handler():
    calls = []
    world = World([], [], physical={})
    original = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda signum, frame: calls.append(signum))
    try:
        assert install_sigterm_hook(world.service) is True
        os.kill(os.getpid(), signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, original)
    assert calls == [signal.SIGTERM]
    assert world.service._sleep_primitive.shutting_down is True


# ------------------------------------------------------------------- audit fixes
def test_legacy_hidden_flag_does_not_mask_an_abandoned_hide():
    hidden = pod("pod-a", "m1", (0,), ip="10.0.0.1", state="hidden")
    world = World([hidden], [_desired("m1/node-a/0", "m1", (0,), "awake")], physical={"10.0.0.1": False})
    assert world.store.load().bindings[0].hidden is True  # the legacy cache says hidden

    codes = [(i["code"], i.get("serve_id")) for i in world.service._sleep_journal_issues()]
    assert ("hidden_without_operation", "pod-a") in codes

    # A SafeScale probe (desired hidden) justifies it.
    with fence(world.redis):
        world.fleet.save_desired([_desired("m1/node-a/0", "m1", (0,), "awake", hidden=True)], expected_version=1)
    assert world.service._sleep_journal_issues() == []


def test_audit_ignores_journal_entries_that_change_between_its_reads():
    hidden = pod("pod-a", "m1", (0,), ip="10.0.0.1", state="hidden")
    world = World([hidden], [_desired("m1/node-a/0", "m1", (0,), "awake")], physical={"10.0.0.1": False})
    entries = iter(
        [
            {"pod-a": {"binding_id": "m1/node-a/0", "phase": "sleeping", "reservation_token": "t1", "operation_id": "op"}},
            {},  # the sleep finished between the two reads
        ]
    )
    world.service._sleep_primitive.journal.entries = lambda: next(entries)

    codes = [i["code"] for i in world.service._sleep_journal_issues()]

    assert "sleep_operation_orphaned" not in codes
    assert "hidden_without_operation" not in codes


@pytest.mark.parametrize("phase", ["sleeping", "sleep_unconfirmed"])
def test_after_sleep_was_sent_one_awake_read_does_not_reopen_routing(phase):
    """Review 2 P3: a mode=wait /sleep may still be running; routing is restored
    only after two awake reads more than sleep_call_timeout_s apart."""
    hidden = pod("pod-a", "m1", (0,), ip="10.0.0.1", state="hidden")
    world = World([hidden], [_desired("m1/node-a/0", "m1", (0,), "awake")], physical={"10.0.0.1": False})
    world.left_over("pod-a", "m1/node-a/0", phase=phase)

    first = world.service.recover_sleep_journal()
    assert first == {"resolved": [], "kept": [{"serve_id": "pod-a", "result": "awake_but_sleep_may_be_running"}]}
    assert world.state("pod-a") == "hidden"

    world.redis.now_ms += 30_000  # < sleep_call_timeout_s (60 s in the test policy)
    assert world.service.recover_sleep_journal()["resolved"] == []
    assert world.state("pod-a") == "hidden"

    world.redis.now_ms += 31_000
    assert world.service.recover_sleep_journal()["resolved"] == [
        {"serve_id": "pod-a", "result": "rolled_back_to_awake"}
    ]
    assert world.state("pod-a") == "awake"


def test_a_pod_found_asleep_records_the_requests_desired_power():
    hidden = pod("pod-a", "m1", (0,), ip="10.0.0.1", state="hidden")
    world = World([hidden], [_desired("m1/node-a/0", "m1", (0,), "awake")], physical={"10.0.0.1": True})
    world.left_over("pod-a", "m1/node-a/0", phase="sleep_unconfirmed")
    world.journal.update("pod-a", desired_on_sleep="sleeping")

    world.service.recover_sleep_journal()

    assert {d.binding_id: d.power for d in world.fleet.load_desired().bindings} == {"m1/node-a/0": "sleeping"}

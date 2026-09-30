"""Review fixes of the service-manager replica floor (2026-09-29, review of 914dd51f).

P1-1 startup never RetryLater on the floor + startup gates are not fleet drift;
P2-3 make-up wake within max_awake_replicas + swap at convergence; P2-4 make-up wake
in its own writer phase; P2-5 defrag keeps the destination once the source slept;
P3-8 internal sleep paths refused over HTTP; P3-9 store and live views intersected,
matched by binding id; P3-10 rate-limited floor WARNINGs; expired transient leases;
failed-wake lease; persistent floor counters; make-up desired guard.
"""

from __future__ import annotations

from dataclasses import replace
import json
import logging
import time

import pytest
from fastapi.testclient import TestClient

from tre_common.registry import (
    Registry,
    ServiceManagerConfig,
    load_registry,
    parse_service_manager_config,
)
from tre_sm.allocator.slots import Migration, Slot
from tre_sm.api.v2 import ServiceManagerV2, create_app
from tre_sm.ops.k8s_ops import ModelDeploymentRecord
from tre_sm.ops.sleep_primitive import SleepJournal
from tre_sm.state.replica_floor import FloorRecorder, FloorViolation, check_floor
from tre_sm.state.supervisor import FleetSupervisor

from sm_test_fakes import Result, binding_of, fence, pod
from test_replica_floor_20260929 import (
    Lease,
    _floor_counts,
    _startup_world,
    _two_awake_world,
    floor_registry,
)
from test_review2_sleep import World, _desired
from test_sleep_no_drain import REPO_REGISTRY
from test_supervisor import FakeService


def _desired_power(world):
    return {d.binding_id: d.power for d in world.fleet.load_desired().bindings}


# ------------------------------------------------------------------ P1-1 startup
def _gate_world():
    """m1 has two replicas; pod-b was replaced by Kubernetes (new UID) and its new
    Pod sits in the startup gate: not Ready, not admitted."""
    world = _two_awake_world()
    runtime = world.runtime
    runtime.snapshots["pod-b"] = replace(
        runtime.snapshots["pod-b"], ready=False, pod_uid="uid-b2", phase="Pending", routable=None
    )
    runtime.list_model_deployments = lambda: [
        ModelDeploymentRecord("m1-node-a-gpu-0", "m1", "node-a", (0,), 1),
        ModelDeploymentRecord("m1-node-a-gpu-1", "m1", "node-a", (1,), 1),
    ]
    runtime.list_admitted_startup_pods = lambda: []
    return world


def test_pod_polling_its_startup_gate_is_informational_not_fleet_drift():
    world = _gate_world()
    service = world.service
    # nobody polls: the not-Ready Pod is drift (as before)
    assert [i["code"] for i in service.detect_fleet_drift()] == ["pod_not_ready"]

    service._note_startup_gate("pod-b", "uid-b2", time.monotonic())
    issues = service.detect_fleet_drift()
    assert [i["code"] for i in issues] == ["startup_admission_pending"]
    assert issues[0]["informational"] is True and issues[0]["pods"] == ["pod-b"]


def test_a_gate_waiting_past_the_grace_or_no_longer_polling_is_drift_again():
    world = _gate_world()
    service = world.service
    now = time.monotonic()
    grace = service._sm_config.startup_gate_drift_grace_s
    # waiting longer than drift_grace_s: a stuck admission is not masked
    service._startup_gate_seen["pod-b"] = ("uid-b2", now - grace - 5, now)
    assert [i["code"] for i in service.detect_fleet_drift()] == ["pod_not_ready"]
    # the gate stopped polling (last request older than gate_seen_s)
    service._startup_gate_seen["pod-b"] = ("uid-b2", now - 10, now - 3600)
    assert [i["code"] for i in service.detect_fleet_drift()] == ["pod_not_ready"]
    # another Pod UID polled (an older incarnation)
    service._startup_gate_seen["pod-b"] = ("uid-old", now - 10, now)
    assert [i["code"] for i in service.detect_fleet_drift()] == ["pod_not_ready"]


def test_the_admission_entry_point_records_the_gate_and_forgets_it_once_admitted():
    world = _gate_world()
    service = world.service
    service.admit_startup = lambda **_kwargs: {"status": "admitted"}
    status, _body = service.request_startup_admission(pod_name="pod-b", pod_uid="uid-b2")
    assert status == 200
    assert "pod-b" not in service._startup_gates_waiting()

    from tre_sm.api.v2 import RetryLater

    def refuse(**_kwargs):
        raise RetryLater("not yet")

    service.admit_startup = refuse
    with pytest.raises(RetryLater):
        service.request_startup_admission(pod_name="pod-b", pod_uid="uid-b2")
    assert service._startup_gates_waiting()["pod-b"][0] == "uid-b2"


def test_supervisor_never_repairs_on_informational_items():
    service = FakeService()
    service.drift = [
        {"code": "startup_admission_pending", "binding_id": "m/node/0", "informational": True}
    ]
    supervisor = FleetSupervisor(service, interval_s=0.01, drift_observations_required=1)
    for _ in range(5):
        supervisor.run_once()
    assert service.repairs == []
    assert supervisor.snapshot().informational[0]["code"] == "startup_admission_pending"
    # real drift next to it still repairs
    service.drift.append({"code": "pod_not_ready", "binding_id": "m/node/1"})
    supervisor.run_once()
    assert service.repairs == [True]


# ------------------------------------------------------------------ P2-3 / P2-4
def _cap_registry(cap: int) -> Registry:
    base = floor_registry()
    models = [replace(m, max_replicas=cap) if m.name == "m1" else m for m in base.models()]
    return Registry(base.topology(), models, service_manager=base.service_manager())


def test_makeup_wake_obeys_max_awake_replicas_and_the_start_is_then_exempt():
    world = _startup_world(spare_sleeping=True)
    world.service._registry = _cap_registry(1)  # m1 already has its one awake replica

    result = world.service.admit_startup(pod_name="tp2-new", pod_uid="new-uid")

    assert result["status"] == "admitted"
    assert world.vllm.sleeping["10.0.0.3"] is True  # no make-up above the cap
    assert not [c for c in world.vllm.calls if c[:2] == ("wake_up", "10.0.0.3")]
    event = world.service.floor_state()["recent"][0]
    assert event["event"] == "replica_floor_exempt" and event["makeup_failed"] == "max_awake_replicas"


def test_makeup_wake_runs_in_its_own_writer_phase_not_under_the_prepare_lock():
    world = _startup_world(spare_sleeping=True)
    phases = []
    wake = world.vllm.wake_up

    def spy(pod_ip, **kwargs):
        phases.append(world.coordinator.active and world.coordinator.active["kind"])
        return wake(pod_ip, **kwargs)

    world.vllm.wake_up = spy
    world.service.admit_startup(pod_name="tp2-new", pod_uid="new-uid")
    assert phases == ["startup_floor_makeup"]


def test_makeup_undone_between_the_phases_is_exempt_at_the_recheck_not_refused():
    """P2-4 race: another writer takes the made-up replica out of routing between
    the make-up phase and the sleep's prepare; the prepare's re-check exempts."""
    world = _startup_world(spare_sleeping=True)
    makeup = world.service._startup_floor_makeup

    def makeup_then_lost(pod_record, residents):
        notes = makeup(pod_record, residents)
        snaps = world.runtime.snapshots
        snaps["pod-c"] = replace(snaps["pod-c"], ready=False)
        return notes

    world.service._startup_floor_makeup = makeup_then_lost
    result = world.service.admit_startup(pod_name="tp2-new", pod_uid="new-uid")
    assert result["status"] == "admitted"
    counts = _floor_counts(world.service)
    assert counts["floor_makeup_wake:startup:m1"] == 1
    assert counts["floor_exempt:startup:m1"] == 1


def test_makeup_candidate_without_a_desired_record_is_never_woken():
    world = _startup_world(spare_sleeping=True)
    with fence(world.redis):
        snapshot = world.fleet.load_desired()
        world.fleet.save_desired(
            [d for d in snapshot.bindings if d.binding_id != "m1/node-a/2"],
            expected_version=snapshot.version,
        )

    result = world.service.admit_startup(pod_name="tp2-new", pod_uid="new-uid")

    assert result["status"] == "admitted"
    assert world.vllm.sleeping["10.0.0.3"] is True
    assert not [c for c in world.vllm.calls if c[:2] == ("wake_up", "10.0.0.3")]
    assert _floor_counts(world.service)["floor_exempt:startup:m1"] == 1


def _started(uid="new-uid"):
    started = pod("tp2-new", "tp2", (0, 1), ip="10.0.0.7", state="sleeping", uid=uid, admitted=True)
    return replace(
        started,
        annotations={
            **started.annotations,
            "tre.aibrix.io/startup-suspended-bindings": json.dumps(["m1/node-a/0"]),
        },
    )


def test_convergence_swaps_the_suspended_resident_for_its_makeup_replica():
    world = _startup_world(spare_sleeping=True)
    world.service.admit_startup(pod_name="tp2-new", pod_uid="new-uid")
    started = _started()
    world.runtime.snapshots[started.name] = started
    world.vllm.sleeping[started.pod_ip] = True
    world.runtime.clear_startup_admission = lambda name: None
    world.service._reconcile_unlocked = lambda drop_missing=False: {}

    world.service._converge_startup(started, True)

    desired = _desired_power(world)
    assert desired["m1/node-a/0"] == "sleeping"  # the suspended resident stays asleep
    assert desired["m1/node-a/2"] == "awake"  # its make-up replica keeps serving
    assert world.vllm.sleeping["10.0.0.1"] is True
    assert world.vllm.sleeping["10.0.0.3"] is False


def test_convergence_of_another_pod_uid_does_not_swap():
    world = _startup_world(spare_sleeping=True)
    world.service.admit_startup(pod_name="tp2-new", pod_uid="new-uid")
    assert world.service._swap_with_floor_makeup("m1/node-a/0", "some-other-uid") is False
    assert _desired_power(world)["m1/node-a/0"] == "awake"


def test_failed_admission_keeps_the_makeup_replica_instead_of_waking_the_resident():
    world = _startup_world(spare_sleeping=True)

    def fail(name, **_kwargs):
        raise RuntimeError("annotating the startup Pod failed")

    world.runtime.admit_startup_pod = fail
    with pytest.raises(RuntimeError):
        world.service.admit_startup(pod_name="tp2-new", pod_uid="new-uid")

    assert world.vllm.sleeping["10.0.0.1"] is True
    assert world.vllm.sleeping["10.0.0.3"] is False
    assert _desired_power(world)["m1/node-a/0"] == "sleeping"


# ------------------------------------------------------------------ P2-5 defrag
def _runtime_defrag_world():
    world = _two_awake_world(m1_min=0)
    runtime = world.runtime
    calls = []
    new_pod = pod("pod-a-new", "m1", (2,), ip="10.0.0.9")

    def create(model, slot):
        runtime.snapshots[new_pod.name] = new_pod
        world.vllm.sleeping[new_pod.pod_ip] = False
        calls.append(("create", tuple(slot.gpu_ids)))
        return new_pod.name

    runtime.create_model_deployment = create
    runtime.wait_pod_deleted = lambda serve_id: calls.append(("wait_deleted", serve_id))
    runtime.wait_pod_ready = lambda serve_id: runtime.snapshots[serve_id]
    return world, calls


def test_runtime_defrag_keeps_the_new_replica_once_the_source_slept():
    world, calls = _runtime_defrag_world()

    def delete(binding):
        calls.append(("delete", binding.serve_id))
        if binding.serve_id == "pod-a":
            raise RuntimeError("apiserver unavailable")

    world.runtime.delete_model_deployment = delete
    source = binding_of(world.runtime.snapshots["pod-a"])
    with pytest.raises(RuntimeError) as info:
        with fence(world.redis):
            world.service._execute_runtime_defrag_migration(
                source, Migration("pod-a", source.slot, Slot("node-a", (2,)))
            )

    assert calls == [("create", (2,)), ("delete", "pod-a")]  # pod-a-new never deleted
    assert world.vllm.sleeping["10.0.0.1"] is True
    assert world.vllm.sleeping["10.0.0.9"] is False
    assert info.value.defrag_source_slept is True


def test_runtime_defrag_keeps_the_new_replica_when_the_source_state_is_unknown():
    world, calls = _runtime_defrag_world()
    world.runtime.delete_model_deployment = lambda binding: calls.append(("delete", binding.serve_id))
    world.vllm.fail_sleep_for.add("10.0.0.1")
    world.vllm.physical_override["10.0.0.1"] = None  # unreachable
    source = binding_of(world.runtime.snapshots["pod-a"])
    with pytest.raises(Exception):
        with fence(world.redis):
            world.service._execute_runtime_defrag_migration(
                source, Migration("pod-a", source.slot, Slot("node-a", (2,)))
            )
    assert ("delete", "pod-a-new") not in calls


def test_power_defrag_keeps_the_destination_when_a_step_after_the_source_sleep_fails():
    from test_defrag_and_deployment_repair import _full_layout_world

    world = _full_layout_world()
    service = world.service
    apply = service._apply_runtime_power_action

    def sleep_then_fail(binding, *, action, **kwargs):
        apply(binding, action=action, **kwargs)
        if action == "sleep" and binding.serve_id == "pod-b":
            raise RuntimeError("recording the source sleep failed")

    service._apply_runtime_power_action = sleep_then_fail
    with pytest.raises(RuntimeError):
        service.defrag(tp_size=2, force=True)
    assert world.vllm.sleeping["10.0.0.2"] is True  # source slept
    assert world.vllm.sleeping["10.0.0.3"] is False  # destination kept serving
    desired = _desired_power(world)
    assert desired["m1/node-a/1"] == "awake" and desired["m1/node-a/2"] == "sleeping"


# ------------------------------------------------------------------ leases
def test_expired_transient_lease_no_longer_hides_a_replica_from_the_floor():
    world = _two_awake_world()
    past = int(time.time() * 1000) - 1
    world.leases.load = lambda: [Lease("m1/node-a/1", "waking", expires_at_ms=past)]
    assert world.service._routable_binding_ids("m1") == {"m1/node-a/0", "m1/node-a/1"}
    # so a hide of one of two replicas (min 1) is allowed
    world.service.put_model_routable("m1", hidden_pods=["pod-a"])


def test_expired_waking_lease_of_a_tp2_replica_does_not_refuse_its_peer_hide():
    pods = [pod("pod-t", "tp2", (0, 1), ip="10.0.0.5"), pod("pod-u", "tp2", (2, 3), ip="10.0.0.6")]
    desired = [
        _desired("tp2/node-a/0,1", "tp2", (0, 1), "awake"),
        _desired("tp2/node-a/2,3", "tp2", (2, 3), "awake"),
    ]
    world = World(pods, desired, sm_registry=floor_registry(tp2_min=1))
    past = int(time.time() * 1000) - 1
    # one lease record per GPU of the TP2 binding, both expired
    world.leases.load = lambda: [
        Lease("tp2/node-a/0,1", "waking", expires_at_ms=past),
        Lease("tp2/node-a/0,1", "waking", expires_at_ms=past),
    ]
    world.service.put_model_routable("tp2", hidden_pods=["pod-u"])  # t still routable
    # an unexpired one still counts as "being woken"
    world.leases.load = lambda: [Lease("tp2/node-a/0,1", "waking")]
    assert world.service._routable_binding_ids("tp2") == set()


def test_failed_wake_settles_its_waking_lease():
    pods = [pod("pod-a", "m1", (0,), ip="10.0.0.1"), pod("pod-c", "m1", (2,), ip="10.0.0.3", state="sleeping")]
    desired = [_desired("m1/node-a/0", "m1", (0,), "awake"), _desired("m1/node-a/2", "m1", (2,), "sleeping")]
    world = World(pods, desired, sm_registry=floor_registry())
    target = binding_of(world.runtime.snapshots["pod-c"])
    world.vllm.wake_up = lambda pod_ip, **_kwargs: Result(False, "engine error")

    with pytest.raises(ValueError):
        with fence(world.redis):
            world.service._apply_runtime_power_action(target, action="wake")
    # physically still asleep -> released (a sleeping binding holds no lease)
    assert world.leases.calls[-2:] == [("acquire", "m1/node-a/2", "waking"), ("release", "m1/node-a/2")]

    # physically awake after the failure -> the awake lease (its GPU is in use)
    def half_wake(pod_ip, **_kwargs):
        world.vllm.sleeping[pod_ip] = False
        return Result(False, "timed out")

    world.vllm.wake_up = half_wake
    with pytest.raises(ValueError):
        with fence(world.redis):
            world.service._apply_runtime_power_action(target, action="wake")
    assert world.leases.calls[-1] == ("acquire", "m1/node-a/2", "awake")


# ------------------------------------------------------------------ P3-8
@pytest.mark.parametrize("path", ["repair", "startup", "defrag", "default"])
def test_http_callers_cannot_name_internal_sleep_paths(path):
    world = _two_awake_world(m1_min=0)
    client = TestClient(create_app(world.service))
    power = client.put("/v2/bindings/pod-a/power", json={"awake": False, "sleep_path": path})
    target = client.put("/v2/models/m1/target", json={"wake_replicas": 0, "sleep_path": path})
    assert power.status_code == 400 and target.status_code == 400
    assert world.vllm.sleeping["10.0.0.1"] is False
    for allowed in ("scale_down", "urgent", "safescale_commit", "apa"):
        from tre_sm.api.v2 import _sleep_path

        assert _sleep_path(allowed) == allowed


# ------------------------------------------------------------------ P3-9
def test_a_pod_the_store_records_hidden_is_not_routable_even_with_a_stale_label():
    world = _two_awake_world()
    snapshot = world.store.load()
    world.store.save(
        [replace(b, hidden=True) if b.serve_id == "pod-b" else b for b in snapshot.bindings],
        expected_version=snapshot.version,
    )
    assert world.service._routable_binding_ids("m1") == {"m1/node-a/0"}
    with pytest.raises(FloorViolation):
        world.service.put_binding_power("pod-a", awake=False, sleep_path="urgent")
    assert world.vllm.sleeping["10.0.0.1"] is False


def test_a_replaced_pod_is_matched_by_binding_id_not_by_name():
    world = _two_awake_world(m1_min=2)
    snaps = world.runtime.snapshots
    snaps["pod-a-2"] = replace(snaps.pop("pod-a"), name="pod-a-2")  # the store still says pod-a
    with pytest.raises(FloorViolation):
        world.service.put_model_routable("m1", hidden_pods=["pod-a"])


# ------------------------------------------------------------------ P3-10
def test_floor_warnings_are_rate_limited_per_outcome_path_model_but_always_counted(caplog):
    now = [1000.0]
    recorder = FloorRecorder(log_interval_s=60.0, clock=lambda: now[0])
    m1 = check_floor("m1", 1, {"a"}, {"a"})
    m2 = check_floor("m2", 1, {"b"}, {"b"})

    def warnings():
        return [r for r in caplog.records if r.levelno == logging.WARNING]

    with caplog.at_level(logging.DEBUG, logger="tre_sm.state.replica_floor"):
        recorder.record("clamped", m1, path="apa")
        recorder.record("clamped", m1, path="apa")
        assert len(warnings()) == 1
        recorder.record("clamped", m2, path="apa")
        assert len(warnings()) == 2
        now[0] += 61
        recorder.record("clamped", m1, path="apa")
        assert len(warnings()) == 3
        assert json.loads(warnings()[-1].getMessage())["suppressed_since_last_log"] == 1
    assert recorder.counts()["floor_clamped:apa:m1"] == 3


def test_repeated_apa_clamps_are_counted_but_logged_once(caplog):
    world = _two_awake_world()
    with caplog.at_level(logging.WARNING, logger="tre_sm.state.replica_floor"):
        world.service.put_model_target("m1", wake_replicas=0, sleep_path="apa")
        world.service.put_model_target("m1", wake_replicas=0, sleep_path="apa")
    assert _floor_counts(world.service)["floor_clamped:apa:m1"] == 2
    clamped = [r for r in caplog.records if "replica_floor_clamped" in r.getMessage()]
    assert len(clamped) == 1


# ------------------------------------------------------------------ persistent counters
def test_floor_counters_survive_a_restart_and_match_the_sleep_stats():
    world = _two_awake_world()
    with pytest.raises(FloorViolation):
        world.service.put_model_routable("m1", hidden_pods=["pod-a", "pod-b"])

    restarted = ServiceManagerV2(
        floor_registry(),
        world.store,
        runtime_ops=world.runtime,
        vllm_ops=world.vllm,
        sleep_journal=SleepJournal(world.redis),
    )
    counts = restarted.floor_state()["counts"]
    assert counts["floor_rejected_total"] == 1
    assert counts["floor_rejected:safescale_hide:m1"] == 1
    assert restarted.sleep_state()["stats"]["floor_rejected_total"] == counts["floor_rejected_total"]


# ------------------------------------------------------------------ registry
def test_registry_startup_admission_and_floor_log_interval():
    base = ServiceManagerConfig()
    assert (base.startup_gate_seen_s, base.startup_gate_drift_grace_s) == (30.0, 600.0)
    assert base.replica_floor_log_interval_s == 60.0
    parsed = parse_service_manager_config(
        {
            "replica_floor": {"enforce": True, "log_interval_s": 0},
            "startup_admission": {"gate_seen_s": 20, "drift_grace_s": 120},
        }
    )
    assert parsed.replica_floor_log_interval_s == 0.0
    assert (parsed.startup_gate_seen_s, parsed.startup_gate_drift_grace_s) == (20.0, 120.0)
    with pytest.raises(ValueError, match="drift_grace_s"):
        parse_service_manager_config({"startup_admission": {"drift_grace_s": -1}})
    with pytest.raises(ValueError, match="startup_admission must be a mapping"):
        parse_service_manager_config({"startup_admission": 5})
    repo = load_registry(str(REPO_REGISTRY)).service_manager()
    assert (repo.startup_gate_seen_s, repo.startup_gate_drift_grace_s) == (30.0, 600.0)
    assert repo.replica_floor_log_interval_s == 60.0

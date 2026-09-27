"""Desired state seeded from the registry (plan 2026-09-27 D7) + its audit items."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from tre_common.bindings import render_binding_set
from tre_sm.api.v2 import ServiceManagerV2
from tre_sm.ops.sleep_primitive import SleepJournal
from tre_sm.server import K8sPodClientFromOps
from tre_sm.state.fleet_seed import seed_desired_from_registry
from tre_sm.state.fleet_store import DesiredBinding, FleetStateStore
from tre_sm.state.store import StateStore

from sm_test_fakes import (
    FakeCoordinator,
    FakeLeases,
    FakeRedis,
    FakeRuntime,
    FakeSafety,
    FakeVllm,
    LegacyRedis,
    deployment,
    fence,
    pod,
    registry,
    startup_pod,
)

REGISTRY_IDS = ["m1/node-a/0", "m1/node-a/1", "tp2/node-a/0,1"]


def _empty_redis_fleet():
    """What bootstrap does on an empty Redis: an EMPTY desired state, version 1."""
    redis = FakeRedis()
    fleet = FleetStateStore(redis)
    with fence(redis):
        fleet.bootstrap([])
    assert fleet.load_desired().version == 1 and fleet.load_desired().bindings == []
    return redis, fleet


def _record(binding_id, model, gpus, *, power, lifecycle="resident", generation=5):
    now = datetime.now(timezone.utc).isoformat()
    return DesiredBinding(binding_id, model, "node-a", tuple(gpus), lifecycle, power, False, generation, now, "controller", "why")


def test_registry_binding_set_is_what_the_manifests_render():
    from gen_model_manifests import build_deployments

    reg = registry()
    rendered = [spec.binding_id for spec in render_binding_set(reg)]
    deployed = [
        "{}/{}/{}".format(
            d["spec"]["template"]["metadata"]["labels"]["model.aibrix.ai/name"],
            d["spec"]["template"]["spec"]["nodeName"],
            d["spec"]["template"]["metadata"]["annotations"]["tre.aibrix.io/gpu-ids"],
        )
        for d in build_deployments(reg)
    ]
    assert rendered == deployed == REGISTRY_IDS


def test_seeding_on_empty_redis_adds_every_registry_binding_resident_sleeping():
    redis, fleet = _empty_redis_fleet()

    with fence(redis):
        result = seed_desired_from_registry(registry(), fleet)

    assert result == {"added": REGISTRY_IDS, "desired_version": 2}
    desired = fleet.load_desired().bindings
    assert [d.binding_id for d in desired] == REGISTRY_IDS
    assert {(d.lifecycle, d.power, d.hidden) for d in desired} == {("resident", "sleeping", False)}
    assert {d.updated_by for d in desired} == {"registry-seed"}


def test_seeding_is_idempotent_and_never_overwrites_existing_records():
    redis, fleet = _empty_redis_fleet()
    awake = _record("m1/node-a/0", "m1", (0,), power="awake")
    absent = _record("tp2/node-a/0,1", "tp2", (0, 1), power="sleeping", lifecycle="absent")
    with fence(redis):
        fleet.save_desired([awake, absent], expected_version=1)
        first = seed_desired_from_registry(registry(), fleet)
        second = seed_desired_from_registry(registry(), fleet)

    assert first["added"] == ["m1/node-a/1"]
    assert second == {"added": [], "desired_version": first["desired_version"]}
    by_id = {d.binding_id: d for d in fleet.load_desired().bindings}
    assert by_id["m1/node-a/0"] == awake  # untouched, generation and all
    assert by_id["tp2/node-a/0,1"] == absent


def test_seeding_requires_the_writer_fence():
    _redis, fleet = _empty_redis_fleet()
    from tre_sm.state.store import StateFenceError

    with pytest.raises(StateFenceError):
        seed_desired_from_registry(registry(), fleet)


def _admission_service(redis, fleet, runtime):
    return ServiceManagerV2(
        registry(),
        StateStore(LegacyRedis()),
        runtime_ops=runtime,
        vllm_ops=FakeVllm(),
        operation_coordinator=FakeCoordinator(redis),
        safety_gate=FakeSafety(),
        fleet_store=fleet,
        gpu_leases=FakeLeases(),
    )


def test_empty_redis_startup_gate_admits_after_seeding():
    redis, fleet = _empty_redis_fleet()
    runtime = FakeRuntime()
    admitted = []
    runtime.get_startup_pod = lambda name: startup_pod(name, "m1", (0,), uid="uid-1")
    runtime.list_startup_resident_snapshots = lambda: []
    runtime.admit_startup_pod = lambda name, **kwargs: admitted.append(name)
    service = _admission_service(redis, fleet, runtime)

    # The service-manager startup (server.create_app) seeds under its bootstrap fence.
    with fence(redis):
        seed_desired_from_registry(registry(), fleet)
    result = service.admit_startup(pod_name="m1-node-a-gpu-0-x", pod_uid="uid-1")

    assert result["status"] == "admitted" and result["binding_id"] == "m1/node-a/0"
    assert admitted == ["m1-node-a-gpu-0-x"]


def test_gate_reseeds_lost_desired_state_but_stays_strict_for_unknown_bindings():
    redis, fleet = _empty_redis_fleet()  # e.g. Redis flushed while the SM kept running
    runtime = FakeRuntime()
    runtime.list_startup_resident_snapshots = lambda: []
    runtime.admit_startup_pod = lambda name, **kwargs: None
    service = _admission_service(redis, fleet, runtime)

    runtime.get_startup_pod = lambda name: startup_pod(name, "m1", (1,), uid="uid-2")
    assert service.admit_startup(pod_name="p", pod_uid="uid-2")["binding_id"] == "m1/node-a/1"
    assert [d.binding_id for d in fleet.load_desired().bindings] == REGISTRY_IDS

    # A binding the registry does not declare is still refused.
    runtime.get_startup_pod = lambda name: startup_pod(name, "m1", (3,), uid="uid-3")
    with pytest.raises(ValueError, match="resolved to 0 desired records"):
        service.admit_startup(pod_name="q", pod_uid="uid-3")


class RepairRuntime(FakeRuntime):
    """Model Deployments that start a Pod (awake vLLM) when scaled to 1."""

    def __init__(self, vllm, deployments):
        super().__init__([], deployments)
        self.vllm = vllm
        self.scale_calls = []
        self.next_ip = 10

    def scale_model_deployment(self, name, *, replicas):
        self.scale_calls.append((name, replicas))
        record = next(d for d in self.deployments if d.name == name)
        pod_name = f"{name}-pod"
        if replicas == 0:
            self.snapshots.pop(pod_name, None)
            return
        self.next_ip += 1
        ip = f"10.0.0.{self.next_ip}"
        self.snapshots[pod_name] = pod(pod_name, record.model, record.gpu_ids, ip=ip, state="hidden")
        self.vllm.sleeping[ip] = False

    def wait_deployment_pods_deleted(self, name):
        return None

    def wait_pod_ready(self, name):
        return self.snapshots[f"{name}-pod"]

    def set_pod_routable(self, serve_id, *, routable):
        super().set_pod_routable(serve_id, routable=routable)
        snapshot = self.snapshots[serve_id]
        self.snapshots[serve_id] = replace(snapshot, routable=routable)


def test_fleet_repair_on_empty_desired_state_seeds_and_ends_healthy():
    redis, fleet = _empty_redis_fleet()
    vllm = FakeVllm()
    reg = registry()
    runtime = RepairRuntime(
        vllm, [deployment("m1", (0,)), deployment("m1", (1,)), deployment("tp2", (0, 1))]
    )
    service = ServiceManagerV2(
        reg,
        StateStore(LegacyRedis()),
        k8s_client=K8sPodClientFromOps(reg.topology(), runtime),
        runtime_ops=runtime,
        vllm_ops=vllm,
        operation_coordinator=FakeCoordinator(redis),
        safety_gate=FakeSafety(),
        fleet_store=fleet,
        gpu_leases=FakeLeases(),
        sleep_journal=SleepJournal(redis),
    )

    accepted = service.start_fleet_repair()  # runs synchronously in the fake coordinator

    assert accepted["awake_binding_ids"] == []
    assert [d.binding_id for d in fleet.load_desired().bindings] == REGISTRY_IDS
    assert all(vllm.sleeping[s.pod_ip] for s in runtime.snapshots.values())
    audit = service.audit()
    assert audit["healthy"] is True, audit["issues"]
    # every repaired pod was put to sleep by the primitive (path "repair")
    assert service.sleep_state()["stats"]["sleeps_path_repair"] == 3


def _audit_service(runtime, fleet, redis, *, journal=None, store=None):
    return ServiceManagerV2(
        registry(),
        store or StateStore(LegacyRedis()),
        k8s_client=K8sPodClientFromOps(registry().topology(), runtime),
        runtime_ops=runtime,
        vllm_ops=FakeVllm(),
        operation_coordinator=FakeCoordinator(redis),
        fleet_store=fleet,
        sleep_journal=journal,
    )


def _codes(issues, *wanted):
    return sorted((i["code"], i.get("binding_id")) for i in issues if i["code"] in wanted)


def test_audit_reports_registry_desired_and_deployment_gaps():
    redis, fleet = _empty_redis_fleet()
    with fence(redis):
        fleet.save_desired(
            [
                _record("m1/node-a/0", "m1", (0,), power="sleeping"),
                _record("m1/node-a/2", "m1", (2,), power="sleeping"),  # e.g. a defrag slot
            ],
            expected_version=1,
        )
    runtime = FakeRuntime([], [deployment("m1", (0,))])

    issues = _audit_service(runtime, fleet, redis).audit()["issues"]

    assert _codes(issues, "registry_binding_without_desired", "desired_without_deployment") == [
        ("desired_without_deployment", "m1/node-a/2"),
        ("registry_binding_without_desired", "m1/node-a/1"),
        ("registry_binding_without_desired", "tp2/node-a/0,1"),
    ]


def test_audit_reports_hidden_pod_without_a_sleep_operation():
    redis, fleet = _empty_redis_fleet()
    with fence(redis):
        seed_desired_from_registry(registry(), fleet)
    runtime = FakeRuntime(
        [
            pod("abandoned", "m1", (0,), ip="10.0.0.1", state="hidden"),
            pod("starting", "m1", (1,), ip="10.0.0.2", state="hidden", admitted=True),
        ]
    )

    issues = _audit_service(runtime, fleet, redis).audit()["issues"]

    assert [(i["code"], i["serve_id"]) for i in issues if i["code"] == "hidden_without_operation"] == [
        ("hidden_without_operation", "abandoned")
    ]


def test_audit_reports_sleep_journal_left_by_a_dead_operation():
    redis, fleet = _empty_redis_fleet()
    with fence(redis):
        seed_desired_from_registry(registry(), fleet)
    journal = SleepJournal(redis)
    journal.begin("pod-x", {"binding_id": "m1/node-a/0", "phase": "draining", "operation_id": "dead-op"})
    runtime = FakeRuntime([pod("pod-x", "m1", (0,), ip="10.0.0.1", state="hidden")])

    issues = _audit_service(runtime, fleet, redis, journal=journal).audit()["issues"]

    orphaned = [i for i in issues if i["code"] == "sleep_operation_orphaned"]
    assert orphaned == [
        {
            "code": "sleep_operation_orphaned",
            "serve_id": "pod-x",
            "binding_id": "m1/node-a/0",
            "phase": "draining",
            "operation_id": "dead-op",
        }
    ]
    # the journal accounts for the hidden pod, so it is not ALSO hidden_without_operation
    assert not [i for i in issues if i["code"] == "hidden_without_operation"]


def test_fleet_seed_endpoint():
    from fastapi.testclient import TestClient
    from tre_sm.api.v2 import create_app

    redis, fleet = _empty_redis_fleet()
    service = _admission_service(redis, fleet, FakeRuntime())

    response = TestClient(create_app(service)).post("/v2/fleet/seed")

    assert response.status_code == 200
    assert response.json()["added"] == REGISTRY_IDS


# ------------------------------------------------------------------ review P2-8
def _three_m1_registry():
    """node-a: m1 tp1 on GPUs 0,1,2 (max_replicas 3): three distinct bindings."""
    from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, ServiceManagerConfig, SloSpec

    from sm_test_fakes import policy, trs

    topology = ClusterTopology(nodes=(NodeSpec("node-a", 4, ((0, 1), (2, 3)), ("G0", "G1", "G2", "G3")),))
    slo = SloSpec(ttft_p95_ms=1200, tpot_p95_ms=100, e2e_p95_ms=10000)
    return Registry(
        topology,
        [ModelSpec("m1", "/m1", 1, 0, 3, "image", slo, trs())],
        service_manager=ServiceManagerConfig(sleep=policy()),
    )


def test_seeding_observes_each_pods_actual_power():
    from tre_sm.state.fleet_seed import seed_desired

    redis, fleet = _empty_redis_fleet()
    runtime = FakeRuntime(
        [
            pod("awake-pod", "m1", (0,), ip="10.0.0.1"),  # awake by probe
            pod("probe-pod", "tp2", (0, 1), ip="10.0.0.5", state="hidden"),  # awake, SafeScale-hidden
        ],
        [deployment("m1", (3,))],  # a defrag-migrated binding the registry does not render
    )
    vllm = FakeVllm()
    vllm.sleeping.update({"10.0.0.1": False, "10.0.0.5": False})

    with fence(redis):
        result = seed_desired(registry(), fleet, runtime_ops=runtime, vllm_ops=vllm)

    assert result["added"] == ["m1/node-a/0", "m1/node-a/1", "m1/node-a/3", "tp2/node-a/0,1"]
    by_id = {d.binding_id: (d.power, d.hidden) for d in fleet.load_desired().bindings}
    assert by_id == {
        "m1/node-a/0": ("awake", False),
        "m1/node-a/1": ("sleeping", False),  # no pod
        "m1/node-a/3": ("sleeping", False),  # Deployment only (union), no pod
        "tp2/node-a/0,1": ("awake", True),
    }


def test_seeding_never_trusts_the_annotation_when_the_probe_is_unknown():
    """Review 2 P1-3: an unknown physical state seeds "sleeping", whatever the
    annotation says (a pod waiting at the startup gate carries the template's
    ``hidden`` annotation although vLLM never ran)."""
    from tre_sm.state.fleet_seed import seed_desired

    redis, fleet = _empty_redis_fleet()
    runtime = FakeRuntime(
        [
            pod("a", "m1", (0,), ip="10.0.0.1"),
            pod("b", "m1", (1,), ip="10.0.0.2", state="sleeping"),
        ]
    )
    vllm = FakeVllm()
    vllm.physical_override.update({"10.0.0.1": None, "10.0.0.2": None})

    with fence(redis):
        seed_desired(registry(), fleet, runtime_ops=runtime, vllm_ops=vllm)

    by_id = {d.binding_id: d.power for d in fleet.load_desired().bindings}
    assert by_id["m1/node-a/0"] == "sleeping" and by_id["m1/node-a/1"] == "sleeping"


def test_redis_loss_with_an_awake_fleet_reseeds_awake_and_never_mass_sleeps():
    """Redis wiped while 3 bindings serve: the supervisor re-seeds desired from the
    pods' actual state; no drift, no fleet repair, not a single /sleep."""
    from tre_sm.state.supervisor import FleetSupervisor

    reg = _three_m1_registry()
    redis, fleet = _empty_redis_fleet()  # the wipe: empty desired / observed state
    snapshots = [pod(f"m1-{g}", "m1", (g,), ip=f"10.0.0.{g + 1}") for g in range(3)]
    runtime = FakeRuntime(snapshots, [deployment("m1", (g,)) for g in range(3)])
    vllm = FakeVllm()
    vllm.sleeping.update({s.pod_ip: False for s in snapshots})
    coordinator = FakeCoordinator(redis)
    service = ServiceManagerV2(
        reg,
        StateStore(LegacyRedis()),
        runtime_ops=runtime,
        vllm_ops=vllm,
        operation_coordinator=coordinator,
        safety_gate=FakeSafety(),
        fleet_store=fleet,
        gpu_leases=FakeLeases(),
    )
    runtime.list_admitted_startup_pods = lambda: []
    runtime.list_startup_resident_snapshots = lambda: []
    supervisor = FleetSupervisor(service, drift_observations_required=1)

    for _ in range(3):
        supervisor.run_once()

    desired = {d.binding_id: d.power for d in fleet.load_desired().bindings}
    assert desired == {f"m1/node-a/{g}": "awake" for g in range(3)}
    assert service.detect_fleet_drift() == []
    assert not any(call[0] == "sleep" for call in vllm.calls)
    assert not any(kind.startswith("fleet_repair") for kind in coordinator.submitted)
    assert service._desired_awake_binding_ids([]) == [f"m1/node-a/{g}" for g in range(3)]


def test_fleet_repair_precheck_covers_registry_and_deployments():
    redis, fleet = _empty_redis_fleet()
    vllm = FakeVllm()
    runtime = RepairRuntime(
        vllm,
        [deployment("m1", (0,)), deployment("m1", (1,)), deployment("tp2", (0, 1)), deployment("m1", (3,))],
    )
    service = ServiceManagerV2(
        registry(),
        StateStore(LegacyRedis()),
        k8s_client=K8sPodClientFromOps(registry().topology(), runtime),
        runtime_ops=runtime,
        vllm_ops=vllm,
        operation_coordinator=FakeCoordinator(redis),
        safety_gate=FakeSafety(),
        fleet_store=fleet,
        gpu_leases=FakeLeases(),
        sleep_journal=SleepJournal(redis),
    )

    service.start_fleet_repair()

    ids = {d.binding_id for d in fleet.load_desired().bindings}
    assert ids == set(REGISTRY_IDS) | {"m1/node-a/3"}

"""Replica floor in the service-manager (A', 2026-09-29).

No hide or sleep may leave a model with fewer ROUTABLE replicas than its registry
``min_replicas``: SafeScale hide / urgent / scale_down / binding power / defrag are
refused (409 ``floor_violation``), APA targets are clamped, a startup admission
wakes another replica first (else it is exempt, recorded - never RetryLater), fleet
repair is exempt but recorded. Routable replicas are binding ids (store and live
view must agree).
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import replace
import threading
import time

import pytest
from fastapi.testclient import TestClient

from tre_common.registry import Registry, ServiceManagerConfig, load_registry, parse_service_manager_config
from tre_sm.allocator.slots import Binding, Migration, Slot
from tre_sm.api.v2 import RetryLater, ServiceManagerV2, create_app
from tre_sm.ops.sleep_primitive import SleepTarget
from tre_sm.state.fleet_repair import FleetRepairExecutor
from tre_sm.state.operations import OperationBusy
from tre_sm.state.replica_floor import FloorViolation, check_floor
from tre_sm.state.store import StateStore

from sm_test_fakes import (
    FakeLeases,
    FakeRuntime,
    FakeVllm,
    LegacyRedis,
    binding_of,
    fence,
    pod,
    policy,
    registry,
    startup_pod,
)
from test_review2_sleep import World, _desired
from test_sleep_no_drain import REPO_REGISTRY


def floor_registry(*, m1_min=1, tp2_min=0, enforce=True, sleep=None) -> Registry:
    base = registry()
    mins = {"m1": m1_min, "tp2": tp2_min}
    models = [replace(model, min_replicas=mins[model.name], max_replicas=4) for model in base.models()]
    config = ServiceManagerConfig(sleep=sleep or policy(), replica_floor_enforce=enforce)
    return Registry(base.topology(), models, service_manager=config)


def _two_awake_world(*, enforce=True, m1_min=1):
    pods = [pod("pod-a", "m1", (0,), ip="10.0.0.1"), pod("pod-b", "m1", (1,), ip="10.0.0.2")]
    desired = [_desired("m1/node-a/0", "m1", (0,), "awake"), _desired("m1/node-a/1", "m1", (1,), "awake")]
    return World(pods, desired, sm_registry=floor_registry(enforce=enforce, m1_min=m1_min))


def _hidden_patches(world):
    return [name for name, state, _gen in world.runtime.patches if state == "hidden"]


def _floor_counts(service):
    return service.floor_state()["counts"]


# ------------------------------------------------------------------ routable view
def test_check_floor_only_counts_what_the_operation_takes_out_of_routing():
    check = check_floor("m", 1, routable={"a", "b"}, removing={"a", "c"})
    assert check.removing == ("a",) and check.after == 1 and check.ok
    assert not check_floor("m", 1, {"a"}, {"a"}).ok
    # Already below the floor, removing nothing routable: never a violation.
    assert check_floor("m", 2, {"a"}, {"z"}).ok
    assert check_floor("m", 0, {"a"}, {"a"}).ok


class Lease:
    def __init__(self, binding_id, phase, expires_at_ms=None):
        self.binding_id, self.phase = binding_id, phase
        self.expires_at_ms = int(time.time() * 1000) + 60_000 if expires_at_ms is None else expires_at_ms


def test_routable_excludes_reserved_waking_not_ready_unlabelled_and_store_unknown_pods():
    world = _two_awake_world()
    service = world.service
    assert service._routable_binding_ids("m1") == {"m1/node-a/0", "m1/node-a/1"}

    # a sleep reservation (a drain in progress) takes it out
    token = service._sleep_primitive.reservations.acquire(
        [binding_of(world.runtime.snapshots["pod-a"])], owner="t", operation_id=None, ttl_s=30
    )
    assert service._routable_binding_ids("m1") == {"m1/node-a/1"}
    service._sleep_primitive.reservations.release(["m1/node-a/0"], token)

    # a replica being woken (unexpired transient lease) does not count
    world.leases.load = lambda: [Lease("m1/node-a/1", "waking")]
    assert service._routable_binding_ids("m1") == {"m1/node-a/0"}
    world.leases.load = lambda: []

    # not Ready / routable label not "true"
    snaps = world.runtime.snapshots
    snaps["pod-a"] = replace(snaps["pod-a"], ready=False)
    snaps["pod-b"] = replace(snaps["pod-b"], routable=None)
    assert service._routable_binding_ids("m1") == set()
    snaps["pod-a"] = replace(snaps["pod-a"], ready=True)
    snaps["pod-b"] = replace(snaps["pod-b"], routable=True)

    # a Ready, routable Pod the store does not record awake counts only inside a
    # make-before-break move that names it (the defrag destination)
    snaps["pod-new"] = pod("pod-new", "m1", (2,), ip="10.0.0.9")
    assert service._routable_binding_ids("m1") == {"m1/node-a/0", "m1/node-a/1"}
    with service._floor_counting(binding_of(snaps["pod-new"])):
        assert service._routable_binding_ids("m1") == {"m1/node-a/0", "m1/node-a/1", "m1/node-a/2"}
    assert service._routable_binding_ids("m1") == {"m1/node-a/0", "m1/node-a/1"}


# ------------------------------------------------------------------ SafeScale hide
def test_safescale_hide_below_the_floor_is_refused_with_409_floor_violation():
    world = _two_awake_world()
    client = TestClient(create_app(world.service))

    both = client.put("/v2/models/m1/routable", json={"hidden_pods": ["pod-a", "pod-b"]})
    assert both.status_code == 409
    body = both.json()
    assert body["error"] == "floor_violation" and body["path"] == "safescale_hide"
    assert body["floor"]["floor"] == 1 and body["floor"]["routable_after"] == 0
    assert _hidden_patches(world) == []  # nothing hidden

    assert client.put("/v2/models/m1/routable", json={"hidden_pods": ["pod-a"]}).status_code == 200
    again = client.put("/v2/models/m1/routable", json={"hidden_pods": ["pod-a", "pod-b"]})
    assert again.status_code == 409 and again.json()["error"] == "floor_violation"
    assert _hidden_patches(world) == ["pod-a"]
    # the unhide (a SafeScale rollback) is never refused
    assert client.put("/v2/models/m1/routable", json={"hidden_pods": []}).status_code == 200

    counts = _floor_counts(world.service)
    assert counts["floor_rejected_total"] == 2
    assert counts["floor_rejected:safescale_hide:m1"] == 2
    assert world.service.sleep_state()["floor"]["recent"][0]["event"] == "replica_floor_rejected"
    # the counters are also in the sleep journal stats (Redis, GET /v2/sleep)
    assert world.service.sleep_state()["stats"]["floor_rejected_total"] == 2


def test_floor_enforce_false_restores_the_previous_behaviour():
    world = _two_awake_world(enforce=False)
    world.service.put_model_routable("m1", hidden_pods=["pod-a", "pod-b"])
    assert sorted(_hidden_patches(world)) == ["pod-a", "pod-b"]
    assert _floor_counts(world.service) == {}


def test_min_replicas_zero_means_no_floor():
    world = _two_awake_world(m1_min=0)
    world.service.put_model_routable("m1", hidden_pods=["pod-a", "pod-b"])
    assert sorted(_hidden_patches(world)) == ["pod-a", "pod-b"]


class LockingCoordinator:
    """A writer lock that really excludes (threads wait for it), like the Redis one."""

    owner = "sm-test"

    def __init__(self, redis):
        self.redis = redis
        self._lock = threading.Lock()
        self.active = None

    @contextmanager
    def operation(self, kind, *, request=None, wait_s=0.0):
        if not self._lock.acquire(timeout=max(wait_s, 5.0)):
            raise OperationBusy("busy")
        from tre_sm.state.operations import _CURRENT_OPERATION
        from sm_test_fakes import FakeHandle

        handle = FakeHandle(f"{kind}-{threading.get_ident()}")
        token = _CURRENT_OPERATION.set(handle)
        self.active = {"operation_id": handle.operation_id, "kind": kind, "status": "running"}
        try:
            with fence(self.redis, handle.operation_id):
                yield handle
        finally:
            self.active = None
            _CURRENT_OPERATION.reset(token)
            self._lock.release()

    def active_operation(self, *, kind=None):
        return self.active


def _slow_hide(runtime, delay_s=0.1):
    """Widen the window between the floor check and the hide (the annotation
    patch) so an unserialized second caller would check in between."""
    original = runtime.write_binding_annotations

    def write(binding, *, state):
        time.sleep(delay_s)
        return original(binding, state=state)

    runtime.write_binding_annotations = write


def _race(calls):
    results = [None] * len(calls)
    barrier = threading.Barrier(len(calls))

    def run(index, call):
        barrier.wait()
        try:
            results[index] = ("ok", call())
        except Exception as exc:  # noqa: BLE001
            results[index] = ("error", exc)

    threads = [threading.Thread(target=run, args=(i, c)) for i, c in enumerate(calls)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    return results


@pytest.mark.parametrize("coordinated", [True, False])
def test_two_concurrent_hides_never_both_pass_the_floor(coordinated):
    """Two SafeScale hides of different pods of a 2-replica min-1 model at once
    (two callers with a stale view: each sends its own full hidden set): exactly
    one goes through (writer lock / process floor lock + one snapshot)."""
    from sm_test_fakes import FakeRedis

    redis = FakeRedis()
    snapshots = [pod("pod-a", "m1", (0,), ip="10.0.0.1"), pod("pod-b", "m1", (1,), ip="10.0.0.2")]
    runtime = FakeRuntime(snapshots)
    vllm = FakeVllm()
    vllm.sleeping.update({"10.0.0.1": False, "10.0.0.2": False})
    store = StateStore(LegacyRedis())
    store.save([binding_of(s) for s in snapshots], expected_version=0)
    service = ServiceManagerV2(
        floor_registry(),
        store,
        runtime_ops=runtime,
        vllm_ops=vllm,
        operation_coordinator=LockingCoordinator(redis) if coordinated else None,
    )
    _slow_hide(runtime)

    results = _race([
        lambda: service.put_model_routable("m1", hidden_pods=["pod-a"]),
        lambda: service.put_model_routable("m1", hidden_pods=["pod-b"]),
    ])

    outcomes = sorted(kind for kind, _ in results)
    assert outcomes == ["error", "ok"], results
    error = next(value for kind, value in results if kind == "error")
    assert isinstance(error, FloorViolation)
    assert len([p for p in runtime.patches if p[1] == "hidden"]) == 1
    assert sum(1 for b in store.load().bindings if b.hidden) == 1


def test_concurrent_hide_and_urgent_sleep_never_both_take_the_last_replica():
    """A SafeScale hide and an urgent model-target shrink of the same 2-replica
    min-1 model at once: one of them is refused (the sleep's prepare checks under
    the same lock the hide holds)."""
    world = _two_awake_world()
    world.coordinator = None
    world.service._operation_coordinator = LockingCoordinator(world.redis)
    _slow_hide(world.runtime)

    results = _race([
        lambda: world.service.put_model_routable("m1", hidden_pods=["pod-a"]),
        lambda: world.service.put_model_target("m1", wake_replicas=1, sleep_path="urgent"),
    ])

    errors = [value for kind, value in results if kind == "error"]
    assert len(errors) == 1 and isinstance(errors[0], FloorViolation), results
    assert world.service._routable_binding_ids("m1") != set()
    assert len(world.service._routable_binding_ids("m1")) == 1


# ------------------------------------------------------------------ sleep paths
@pytest.mark.parametrize("path", ["urgent", "scale_down"])
def test_model_target_shrink_below_the_floor_is_refused_before_anything_is_hidden(path):
    world = _two_awake_world()
    client = TestClient(create_app(world.service))

    response = client.put("/v2/models/m1/target", json={"wake_replicas": 0, "sleep_path": path})

    assert response.status_code == 409
    assert response.json()["error"] == "floor_violation" and response.json()["path"] == path
    assert _hidden_patches(world) == []
    assert not [c for c in world.vllm.calls if c[0] == "sleep"]
    assert world.service.sleep_state()["reservations"] == {}
    # one replica is fine
    assert client.put("/v2/models/m1/target", json={"wake_replicas": 1, "sleep_path": path}).status_code == 200
    assert _floor_counts(world.service)[f"floor_rejected:{path}:m1"] == 1


def test_binding_power_sleep_of_the_last_routable_replica_is_refused():
    world = _two_awake_world()
    client = TestClient(create_app(world.service))
    assert client.put("/v2/bindings/pod-a/power", json={"awake": False}).status_code == 200
    refused = client.put("/v2/bindings/pod-b/power", json={"awake": False, "sleep_path": "urgent"})
    assert refused.status_code == 409 and refused.json()["error"] == "floor_violation"
    assert world.vllm.sleeping["10.0.0.2"] is False


def test_safescale_commit_of_a_hidden_probe_pod_is_not_a_removal():
    world = _two_awake_world()
    world.service.put_model_routable("m1", hidden_pods=["pod-a"])
    # pod-a is hidden already: sleeping it takes nothing out of routing
    world.service.put_binding_power("pod-a", awake=False, sleep_path="safescale_commit")
    assert world.vllm.sleeping["10.0.0.1"] is True
    assert "floor_rejected_total" not in _floor_counts(world.service)


# ------------------------------------------------------------------ APA clamp
def test_apa_scale_down_is_clamped_to_the_floor_not_refused():
    world = _two_awake_world()
    client = TestClient(create_app(world.service))

    response = client.post("/scale_service", params={"model_name": "m1", "scale_type": "down", "scale_value": 2})

    assert response.status_code == 200, response.text
    assert response.json() == {"requested": 2, "actual": 1}
    assert len(world.service._routable_binding_ids("m1")) == 1
    counts = _floor_counts(world.service)
    assert counts["floor_clamped:apa:m1"] == 1 and "floor_rejected_total" not in counts


def test_apa_at_the_floor_sleeps_nothing_and_keeps_desired_awake():
    world = _two_awake_world()
    world.service.put_model_target("m1", wake_replicas=1, sleep_path="apa")
    response = world.service.put_model_target("m1", wake_replicas=0, sleep_path="apa")
    assert response["actions"] == []
    assert response["floor"]["clamped"] is True and response["floor"]["kept_awake"]
    awake = [b for b in world.store.load().bindings if b.awake]
    assert len(awake) == 1
    desired = {d.binding_id: d.power for d in world.fleet.load_desired().bindings}
    assert desired[awake[0].binding_id] == "awake"


# ------------------------------------------------------------------ startup
def _startup_world(*, spare_sleeping: bool):
    """m1 has one awake replica (pod-a, GPU 0); a tp2 Pod starts on GPUs (0, 1)
    and must put pod-a to sleep. With ``spare_sleeping`` m1 has a sleeping
    replica on GPU 2 that can be woken first."""
    pods = [pod("pod-a", "m1", (0,), ip="10.0.0.1")]
    desired = [
        _desired("m1/node-a/0", "m1", (0,), "awake"),
        _desired("tp2/node-a/0,1", "tp2", (0, 1), "sleeping"),
    ]
    if spare_sleeping:
        pods.append(pod("pod-c", "m1", (2,), ip="10.0.0.3", state="sleeping"))
        desired.append(_desired("m1/node-a/2", "m1", (2,), "sleeping"))
    world = World(pods, desired, sm_registry=floor_registry())
    runtime = world.runtime
    runtime.get_startup_pod = lambda name: startup_pod(name, "tp2", (0, 1), uid="new-uid")
    runtime.list_startup_resident_snapshots = lambda: runtime.list_pod_snapshots()
    runtime.admit_startup_pod = lambda name, **kwargs: None
    return world


def test_startup_wakes_another_replica_before_sleeping_the_last_routable_one():
    world = _startup_world(spare_sleeping=True)

    result = world.service.admit_startup(pod_name="tp2-new", pod_uid="new-uid")

    assert result["status"] == "admitted"
    assert world.vllm.sleeping["10.0.0.1"] is True  # the resident slept for the start
    assert world.vllm.sleeping["10.0.0.3"] is False  # the spare replica woke first
    wake_at = next(i for i, c in enumerate(world.vllm.calls) if c[:2] == ("wake_up", "10.0.0.3"))
    sleep_at = next(i for i, c in enumerate(world.vllm.calls) if c[:2] == ("sleep", "10.0.0.1"))
    assert wake_at < sleep_at
    desired = {d.binding_id: d.power for d in world.fleet.load_desired().bindings}
    assert desired["m1/node-a/2"] == "awake"
    counts = _floor_counts(world.service)
    assert counts["floor_makeup_wake:startup:m1"] == 1 and "floor_rejected_total" not in counts


def test_startup_without_a_spare_replica_is_exempt_and_recorded_not_retried():
    """Review 2026-09-29 P1-1: RetryLater would hold the Pod in its startup gate
    (fleet drift -> fleet-wide repair); the start goes ahead, exempt and recorded."""
    world = _startup_world(spare_sleeping=False)

    result = world.service.admit_startup(pod_name="tp2-new", pod_uid="new-uid")

    assert result["status"] == "admitted"
    assert world.vllm.sleeping["10.0.0.1"] is True
    counts = _floor_counts(world.service)
    assert counts["floor_exempt:startup:m1"] == 1 and "floor_rejected_total" not in counts
    event = world.service.floor_state()["recent"][0]
    assert event["event"] == "replica_floor_exempt" and event["path"] == "startup"
    assert event["makeup_failed"] == "no_wakeable_replica"


# ------------------------------------------------------------------ defrag
def test_power_defrag_is_make_before_break_so_a_single_replica_can_move():
    from test_defrag_and_deployment_repair import _full_layout_world

    world = _full_layout_world()
    # m1 min 2 (two awake): the break-before-make order would drop to 1 routable.
    world.service._registry = floor_registry(m1_min=2)
    result = world.service.defrag(tp_size=2, force=True)
    assert [a["action"] for a in result["actions"]] == ["wake", "hide", "sleep"]
    assert "floor_rejected_total" not in _floor_counts(world.service)


def test_power_defrag_rolls_the_destination_back_when_the_source_sleep_fails():
    from test_defrag_and_deployment_repair import _full_layout_world

    world = _full_layout_world()
    world.vllm.fail_sleep_for.add("10.0.0.2")  # the source (pod-b) never sleeps
    with pytest.raises(Exception):
        world.service.defrag(tp_size=2, force=True)
    assert world.vllm.sleeping["10.0.0.2"] is False  # source still serving
    assert world.vllm.sleeping["10.0.0.3"] is True  # destination back asleep


def test_runtime_defrag_removes_the_new_replica_when_the_source_sleep_fails():
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
    runtime.delete_model_deployment = lambda binding: calls.append(("delete", binding.serve_id))
    runtime.wait_pod_deleted = lambda serve_id: calls.append(("wait_deleted", serve_id))
    runtime.wait_pod_ready = lambda serve_id: runtime.snapshots[serve_id]
    world.vllm.fail_sleep_for.add("10.0.0.1")
    source = binding_of(runtime.snapshots["pod-a"])

    with pytest.raises(Exception):
        with fence(world.redis):
            world.service._execute_runtime_defrag_migration(
                source, Migration("pod-a", source.slot, Slot("node-a", (2,)))
            )

    assert calls == [("create", (2,)), ("delete", "pod-a-new"), ("wait_deleted", "pod-a-new")]
    assert world.vllm.sleeping["10.0.0.1"] is False


def test_defrag_that_would_break_the_floor_is_refused():
    world = _two_awake_world()
    world.runtime.snapshots["pod-b"] = replace(world.runtime.snapshots["pod-b"], ready=False)
    source = binding_of(world.runtime.snapshots["pod-a"])
    with pytest.raises(FloorViolation):
        with fence(world.redis):
            world.service._apply_runtime_power_action(source, action="sleep", sleep_path="defrag")
    assert world.vllm.sleeping["10.0.0.1"] is False


# ------------------------------------------------------------------ fleet repair
def test_repair_path_is_exempt_but_recorded():
    world = _two_awake_world()
    world.service.put_model_routable("m1", hidden_pods=["pod-a"])
    target = SleepTarget(binding_of(world.runtime.snapshots["pod-b"]), "10.0.0.2")
    with fence(world.redis):
        world.service._sleep_targets([target], sleep_path="repair")
    assert world.vllm.sleeping["10.0.0.2"] is True
    counts = _floor_counts(world.service)
    assert counts["floor_exempt:repair:m1"] == 1 and "floor_rejected_total" not in counts


def test_fleet_repair_quarantine_reports_the_hidden_residents_and_the_sm_records_exemptions():
    world = _two_awake_world()
    seen = []

    class Runtime:
        def list_pod_snapshots(self, *, model=None):
            return list(world.runtime.snapshots.values())

        def write_binding_annotations(self, binding, *, state):
            return 1

    class Vllm:
        def is_sleeping(self, pod_ip, *, port=None):
            return True  # nothing to sleep in this test

    executor = FleetRepairExecutor(
        runtime_ops=Runtime(),
        vllm_ops=Vllm(),
        safety_gate=object(),
        sleep_binding=lambda binding, ip: None,
        on_quarantine=lambda snapshots: seen.append(sorted(s.name for s in snapshots)),
    )

    class Deployment:
        def __init__(self, binding_id):
            self.binding_id = binding_id

    class Op:
        def assert_active(self):
            return None

    executor._quarantine_and_sleep_residents(
        Op(), {"m1/node-a/0": Deployment("m1/node-a/0"), "m1/node-a/1": Deployment("m1/node-a/1")}
    )
    assert seen == [["pod-a", "pod-b"]]

    world.service._record_repair_floor_exemptions(list(world.runtime.snapshots.values()))
    counts = _floor_counts(world.service)
    assert counts["floor_exempt:repair:m1"] == 1


def test_the_sm_wires_the_quarantine_hook_into_fleet_repair():
    import inspect

    source = inspect.getsource(ServiceManagerV2.__init__)
    assert "on_quarantine=self._record_repair_floor_exemptions" in source


# ------------------------------------------------------------------ registry
def test_registry_replica_floor_switch():
    assert ServiceManagerConfig().replica_floor_enforce is True
    assert parse_service_manager_config(None).replica_floor_enforce is True
    assert parse_service_manager_config({"replica_floor": {"enforce": False}}).replica_floor_enforce is False
    with pytest.raises(ValueError, match="replica_floor must be a mapping"):
        parse_service_manager_config({"replica_floor": True})
    assert load_registry(str(REPO_REGISTRY)).service_manager().replica_floor_enforce is True

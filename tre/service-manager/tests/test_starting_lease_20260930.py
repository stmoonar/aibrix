"""S2 (2026-09-30): a Pod admitted at its startup gate keeps its GPUs until it
converged or is gone - not for a fixed 120 s.

Before: the admission's ``starting`` lease expired after 120 s, the store records
the new Pod asleep/hidden, and a cold vLLM load often takes longer. Once the lease
had expired, a wake of a resident on the same GPU (controller wake, the
convergence's ``_restore_desired_awake``) was stopped only by gpu-truth (09-28:
13 times). Now the ``starting`` lease never expires (released by convergence or
by the orphan reaper once the Pod is gone) and the wake account refuses any GPU
held by another binding's lease - without gpu-truth."""

import pytest

from tre_sm.allocator.slots import Binding, Slot
from tre_sm.api.v2 import WakeConflict
from tre_sm.ops.k8s_ops import ModelDeploymentRecord
from tre_sm.state.gpu_leases import GpuLeaseStore

from sm_test_fakes import FakeRedis, fence, pod
from test_review2_sleep import _desired
from test_review4_sm import _deployment_name, _gate_world

LOAD_S = 130  # a cold load longer than the old 120 s starting TTL


def test_starting_lease_never_expires_by_default_waking_still_does():
    redis = FakeRedis()
    leases = GpuLeaseStore(redis)
    starting = Binding("new", "tp2", Slot("node-a", (0, 1)), awake=False)
    waking = Binding("pod-x", "m1", Slot("node-a", (2,)), awake=False)
    with fence(redis):
        assert leases.acquire(starting, phase="starting").expires_at_ms == 0
        assert leases.acquire(waking, phase="waking").expires_at_ms == redis.now_ms + 120_000
        leases.rebuild_awake([], starting_bindings=[starting])
    assert [lease.expires_at_ms for lease in leases.load()] == [0]
    # an explicit TTL is still possible
    assert GpuLeaseStore(redis, starting_ttl_ms=5000).ttl_ms("starting") == 5000


def _admitted_world():
    """node-a: resident m1 on GPU 0 (asleep); a new tp2 Pod on GPUs 0,1 waits at
    its startup gate. Real GPU leases (Lua modelled by FakeRedis)."""
    world = _gate_world(
        [pod("pod-r", "m1", (0,), ip="10.0.0.1", state="sleeping")],
        [
            _desired("m1/node-a/0", "m1", (0,), "sleeping"),
            _desired("tp2/node-a/0,1", "tp2", (0, 1), "sleeping"),
        ],
    )
    world.leases = GpuLeaseStore(world.redis)
    world.service._gpu_leases = world.leases
    slot = Slot("node-a", (0, 1))
    name = _deployment_name("tp2", slot)
    world.runtime.deployments.append(ModelDeploymentRecord(name, "tp2", "node-a", (0, 1), 1))
    world.runtime.pending[name] = ("tp2-new", "uid-new", "tp2", slot)
    result = world.service.admit_startup(pod_name="tp2-new", pod_uid="uid-new")
    assert result["status"] == "admitted"
    return world


def test_admission_takes_a_starting_lease_that_outlives_a_long_cold_load():
    world = _admitted_world()

    (lease,) = world.leases.load()
    assert (lease.binding_id, lease.phase, lease.expires_at_ms) == ("tp2/node-a/0,1", "starting", 0)
    world.redis.now_ms += LOAD_S * 1000
    assert world.service._transient_lease_ids() == {"tp2/node-a/0,1"}


def test_a_same_gpu_wake_is_refused_after_a_long_cold_load_without_gpu_truth():
    world = _admitted_world()
    assert world.service._gpu_truth is None  # nothing but the account can stop it
    world.redis.now_ms += LOAD_S * 1000

    with pytest.raises(WakeConflict) as caught:
        world.service.put_binding_power("pod-r", awake=True)

    assert caught.value.reason == "lease_starting"
    assert (caught.value.node, caught.value.gpus) == ("node-a", (0,))
    assert not any(call[0] == "wake_up" for call in world.vllm.calls)
    assert world.desired()["m1/node-a/0"][0] == "sleeping"  # intent rolled back


def test_the_convergence_restore_path_is_refused_the_same_way():
    world = _admitted_world()
    world.redis.now_ms += LOAD_S * 1000

    with pytest.raises(WakeConflict, match="starting GPU lease of tp2/node-a/0,1"):
        with world.coordinator.operation("restore"):
            world.service._restore_desired_awake("m1/node-a/0")
    assert not any(call[0] == "wake_up" for call in world.vllm.calls)


def test_planning_skips_a_lease_blocked_binding_and_wakes_another():
    world = _gate_world(
        [
            pod("pod-r", "m1", (0,), ip="10.0.0.1", state="sleeping"),
            pod("pod-s", "m1", (2,), ip="10.0.0.2", state="sleeping"),
        ],
        [
            _desired("m1/node-a/0", "m1", (0,), "sleeping"),
            _desired("m1/node-a/2", "m1", (2,), "sleeping"),
        ],
    )
    world.leases = GpuLeaseStore(world.redis)
    world.service._gpu_leases = world.leases
    with fence(world.redis):
        world.leases.acquire(Binding("x", "tp2", Slot("node-a", (0, 1)), awake=False), phase="starting")

    result = world.service.put_model_target("m1", wake_replicas=1)

    assert [a["serve_id"] for a in result["actions"] if a["action"] == "wake"] == ["pod-s"]

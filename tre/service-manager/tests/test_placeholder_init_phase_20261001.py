"""2026-10-01 deployment finding: the startup-placeholder reaper released the
``starting`` lease of a Pod right after its admission (held 0 s) because the Pod
was still initialising - phase Pending (not in the Running-only snapshots, so
"gone") and its engine container Waiting (PodInitializing). The engine then
loaded without a fence and a second Pod was admitted onto the same GPU.

A placeholder is now released only when every Pod of the binding finished
initialising and its engine container is terminated or waiting for a release
reason (CrashLoopBackOff, Error, image pull / create errors), never before a
minimum hold, and never while /is_sleeping reads awake."""

from __future__ import annotations

import pytest

from tre_sm.allocator.slots import Slot
from tre_sm.allocator.topology import GPU_IDS_ANNOTATION
from tre_sm.api.v2 import WakeConflict
from tre_sm.ops.k8s_ops import MANAGED_LABEL, MODEL_LABEL, K8sOps, ModelDeploymentRecord
from tre_sm.state.gpu_leases import GpuLeaseConflict, GpuLeaseStore

from sm_test_fakes import pod
from test_k8s_ops import FakeK8sApi
from test_review2_sleep import _desired
from test_review4_sm import _deployment_name, _gate_world


def _k8s_pod(name, *, phase, gate=None, engine=None):
    status = {"phase": phase, "podIP": "10.0.0.9" if phase == "Running" else None}
    init = [{"name": "tre-reissue-sidecar", "state": {"running": {}}}]
    if gate is not None:
        init.append({"name": "tre-startup-gate", "state": gate})
    status["initContainerStatuses"] = init
    if engine is not None:
        status["containerStatuses"] = [{"name": "vllm-openai", "state": engine, "ready": False}]
    return {
        "metadata": {
            "name": name, "uid": f"uid-{name}",
            "labels": {MODEL_LABEL: "tp2", MANAGED_LABEL: "true"},
            "annotations": {GPU_IDS_ANNOTATION: "0,1"},
        },
        "spec": {
            "nodeName": "node-a",
            "initContainers": [
                {"name": "tre-reissue-sidecar", "restartPolicy": "Always"},
                {"name": "tre-startup-gate"},
            ],
        },
        "status": status,
    }


def test_startup_pod_states_reports_init_phase_pods():
    pending = _k8s_pod(
        "p-init", phase="Pending", gate={"running": {}}, engine={"waiting": {"reason": "PodInitializing"}}
    )
    crashing = _k8s_pod(
        "p-crash", phase="Running", gate={"terminated": {"exitCode": 0}},
        engine={"waiting": {"reason": "CrashLoopBackOff"}},
    )
    loading = _k8s_pod(
        "p-load", phase="Running", gate={"terminated": {"exitCode": 0}}, engine={"running": {"startedAt": "x"}},
    )
    ops = K8sOps(api=FakeK8sApi([pending, crashing, loading]), namespace="default")
    states = {state["name"]: state for state in ops.startup_pod_states(model="tp2")}

    assert states["p-init"]["init_done"] is False
    assert (states["p-init"]["engine_state"], states["p-init"]["engine_reason"]) == ("waiting", "PodInitializing")
    assert states["p-init"]["binding_id"] == "tp2/node-a/0,1"
    assert states["p-crash"]["init_done"] is True
    assert (states["p-crash"]["engine_state"], states["p-crash"]["engine_reason"]) == ("waiting", "CrashLoopBackOff")
    assert states["p-load"]["engine_state"] == "running"


def _admitted_init_phase_world():
    """The deployment's sequence: a tp2 Pod on GPUs 0,1 admitted at its startup
    gate, still initialising; a sleeping m1 resident on GPU 0."""
    world = _gate_world(
        [pod("pod-r", "m1", (0,), ip="10.0.0.1", state="sleeping")],
        [_desired("m1/node-a/0", "m1", (0,), "sleeping"), _desired("tp2/node-a/0,1", "tp2", (0, 1), "sleeping")],
    )
    world.leases = GpuLeaseStore(world.redis)
    world.service._gpu_leases = world.leases
    slot = Slot("node-a", (0, 1))
    name = _deployment_name("tp2", slot)
    world.runtime.deployments.append(ModelDeploymentRecord(name, "tp2", "node-a", (0, 1), 1))
    world.runtime.pending[name] = ("tp2-new", "uid-new", "tp2", slot)
    assert world.service.admit_startup(pod_name="tp2-new", pod_uid="uid-new")["status"] == "admitted"
    world.states = {
        "tp2-new": {
            "name": "tp2-new", "binding_id": "tp2/node-a/0,1", "phase": "Pending", "deleting": False,
            "init_done": False, "engine_state": "waiting", "engine_reason": "PodInitializing",
            "ready": False, "pod_ip": None,
        }
    }
    world.runtime.startup_pod_states = lambda model=None: [
        state for state in world.states.values()
        if model is None or state["binding_id"].startswith(f"{model}/")
    ]
    return world


def _leases(world):
    return {lease.binding_id: lease.phase for lease in world.leases.load()}


def test_startup_placeholder_kept_while_the_admitted_pod_initialises():
    world = _admitted_init_phase_world()

    # the reaper's pass right after the admission, and long after the minimum hold
    assert world.service.reap_stale_startup_placeholders(now=0.0) == []
    assert world.service.reap_stale_startup_placeholders(now=10_000.0) == []
    assert _leases(world) == {"tp2/node-a/0,1": "starting"}

    # a second Pod admitted onto the same GPU is refused (the deployment's 14b)
    slot = Slot("node-a", (0,))
    other = _deployment_name("m1", slot)
    world.runtime.deployments.append(ModelDeploymentRecord(other, "m1", "node-a", (0,), 1))
    world.runtime.pending[other] = ("m1-new", "uid-m1-new", "m1", slot)
    with pytest.raises(GpuLeaseConflict):
        world.service.admit_startup(pod_name="m1-new", pod_uid="uid-m1-new")
    # and so is a wake of the resident there
    with pytest.raises(WakeConflict) as caught:
        world.service.put_binding_power("pod-r", awake=True)
    assert caught.value.reason == "lease_starting"


@pytest.mark.parametrize(
    "state",
    [
        {"phase": "Running", "init_done": True, "engine_state": "waiting", "engine_reason": "ContainerCreating"},
        {"phase": "Running", "init_done": True, "engine_state": "running", "engine_reason": None},
        {"phase": "Running", "init_done": True, "engine_state": None, "engine_reason": None},
        {"phase": "Pending", "init_done": False, "engine_state": "terminated", "engine_reason": "Error"},
    ],
)
def test_startup_placeholder_kept_unless_the_engine_is_provably_not_running(state):
    world = _admitted_init_phase_world()
    world.states["tp2-new"].update(state)
    world.service.reap_stale_startup_placeholders(now=0.0)
    assert world.service.reap_stale_startup_placeholders(now=10_000.0) == []
    assert _leases(world) == {"tp2/node-a/0,1": "starting"}


@pytest.mark.parametrize(
    "engine", [("waiting", "CrashLoopBackOff"), ("terminated", "Error"), ("waiting", "ErrImagePull")]
)
def test_startup_placeholder_crashloop_release_still_works_after_the_minimum_hold(engine):
    world = _admitted_init_phase_world()
    world.states["tp2-new"].update(
        {"phase": "Running", "init_done": True, "engine_state": engine[0], "engine_reason": engine[1],
         "pod_ip": "10.0.0.9"}
    )
    world.vllm.physical_override["10.0.0.9"] = None
    assert world.service.reap_stale_startup_placeholders(now=0.0) == []  # minimum hold
    assert world.service.reap_stale_startup_placeholders(now=60.0) == []
    assert world.service.reap_stale_startup_placeholders(now=121.0) == ["tp2/node-a/0,1"]
    assert _leases(world) == {}


def test_startup_placeholder_never_released_while_it_reads_awake():
    world = _admitted_init_phase_world()
    world.states["tp2-new"].update(
        {"phase": "Running", "init_done": True, "engine_state": "waiting", "engine_reason": "CrashLoopBackOff",
         "pod_ip": "10.0.0.9"}
    )
    world.vllm.sleeping["10.0.0.9"] = False
    world.service.reap_stale_startup_placeholders(now=0.0)
    assert world.service.reap_stale_startup_placeholders(now=10_000.0) == []


def test_startup_placeholder_release_reasons_and_hold_are_configurable():
    world = _admitted_init_phase_world()
    world.service._placeholder_min_hold_s = 5.0
    world.service._placeholder_release_reasons = frozenset({"CrashLoopBackOff"})
    world.states["tp2-new"].update(
        {"phase": "Running", "init_done": True, "engine_state": "waiting", "engine_reason": "ErrImagePull"}
    )
    world.service.reap_stale_startup_placeholders(now=0.0)
    assert world.service.reap_stale_startup_placeholders(now=10.0) == []  # not in the set
    world.states["tp2-new"]["engine_reason"] = "CrashLoopBackOff"
    assert world.service.reap_stale_startup_placeholders(now=10.0) == ["tp2/node-a/0,1"]


def test_startup_placeholder_no_pod_is_left_to_the_orphan_reaper():
    world = _admitted_init_phase_world()
    world.states.clear()
    world.service.reap_stale_startup_placeholders(now=0.0)
    assert world.service.reap_stale_startup_placeholders(now=10_000.0) == []

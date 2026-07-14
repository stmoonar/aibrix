from dataclasses import replace

import pytest

from tre_sm.allocator.slots import Binding
from tre_sm.allocator.topology import GPU_IDS_ANNOTATION, K8sPodSnapshot, STATE_ANNOTATION
from tre_sm.ops.k8s_ops import ModelDeploymentRecord
from tre_sm.state.fleet_repair import FleetRepairExecutor


class Result:
    success = True
    message = ""


class FakeOperation:
    def __init__(self):
        self.phases = []

    def assert_active(self):
        return None

    def advance(self, phase, *, details=None):
        self.phases.append((phase, details))


class FakeSafety:
    def __init__(self):
        self.calls = 0

    def wait_until_healthy(self, operation):
        self.calls += 1
        operation.advance("cluster_healthy")


class FakeVllm:
    def __init__(self, sleeping_by_ip):
        self.sleeping = dict(sleeping_by_ip)
        self.calls = []

    def is_sleeping(self, pod_ip, *, port=None):
        self.calls.append(("is_sleeping", pod_ip, port))
        return self.sleeping.get(pod_ip)

    def sleep(self, pod_ip, *, port=None):
        self.calls.append(("sleep", pod_ip, port))
        self.sleeping[pod_ip] = True
        return Result()

    def wait_until_ready(self, pod_ip, *, port=None):
        self.calls.append(("wait_until_ready", pod_ip, port))
        return Result()


class FakeRuntime:
    def __init__(self, deployments, snapshots, vllm):
        self.deployments = list(deployments)
        self.snapshots = {snapshot.name: snapshot for snapshot in snapshots}
        self.vllm = vllm
        self.scale_calls = []
        self.annotation_calls = []

    def list_model_deployments(self):
        return list(self.deployments)

    def list_pod_snapshots(self, *, model=None):
        values = list(self.snapshots.values())
        if model is not None:
            values = [item for item in values if item.model == model]
        return values

    def write_binding_annotations(self, binding, *, state):
        self.annotation_calls.append((binding.binding_id, state))
        snapshot = self.snapshots[binding.serve_id]
        annotations = dict(snapshot.annotations)
        annotations[STATE_ANNOTATION] = state
        self.snapshots[binding.serve_id] = replace(
            snapshot,
            annotations=annotations,
            routable=state == "awake",
        )

    def scale_model_deployment(self, name, *, replicas):
        deployment = next(item for item in self.deployments if item.name == name)
        self.scale_calls.append((name, replicas))
        if replicas == 0:
            self.snapshots = {
                pod_name: snapshot
                for pod_name, snapshot in self.snapshots.items()
                if _binding_id(snapshot) != deployment.binding_id
            }
            return
        target_gpus = set(deployment.gpu_ids)
        for snapshot in self.snapshots.values():
            if snapshot.node != deployment.node:
                continue
            if not target_gpus.intersection(_gpu_ids(snapshot)):
                continue
            assert self.vllm.sleeping[snapshot.pod_ip] is True
        pod_ip = f"10.0.0.{len(self.snapshots) + 10}"
        pod = _snapshot(deployment, pod_ip=pod_ip, state="hidden")
        self.snapshots[pod.name] = pod
        self.vllm.sleeping[pod_ip] = False

    def wait_deployment_pods_deleted(self, deployment_name):
        deployment = next(
            item for item in self.deployments if item.name == deployment_name
        )
        assert all(
            _binding_id(snapshot) != deployment.binding_id
            for snapshot in self.snapshots.values()
        )

    def wait_pod_ready(self, serve_id):
        deployment = next(item for item in self.deployments if item.name == serve_id)
        return next(
            snapshot
            for snapshot in self.snapshots.values()
            if _binding_id(snapshot) == deployment.binding_id
        )


def _deployment(name, model, node, gpu_ids):
    return ModelDeploymentRecord(name, model, node, tuple(gpu_ids), replicas=1)


def _snapshot(deployment, *, pod_ip, state, ready=True):
    gpu_text = ",".join(str(gpu_id) for gpu_id in deployment.gpu_ids)
    return K8sPodSnapshot(
        name=f"{deployment.name}-pod",
        model=deployment.model,
        node=deployment.node,
        env={"CUDA_VISIBLE_DEVICES": gpu_text},
        annotations={GPU_IDS_ANNOTATION: gpu_text, STATE_ANNOTATION: state},
        pod_ip=pod_ip,
        routable=state == "awake",
        ready=ready,
    )


def _gpu_ids(snapshot):
    return tuple(
        int(part) for part in snapshot.annotations[GPU_IDS_ANNOTATION].split(",")
    )


def _binding_id(snapshot):
    return f"{snapshot.model}/{snapshot.node}/{snapshot.annotations[GPU_IDS_ANNOTATION]}"


def test_repair_sleeps_residents_before_rebuilding_missing_peer_then_restores_target():
    first = _deployment("m1-node-a-gpu-0", "m1", "node-a", (0,))
    missing = _deployment("m2-node-a-gpu-0", "m2", "node-a", (0,))
    first_pod = _snapshot(first, pod_ip="10.0.0.1", state="awake")
    vllm = FakeVllm({"10.0.0.1": False})
    runtime = FakeRuntime([first, missing], [first_pod], vllm)
    operation = FakeOperation()
    reconciles = []

    def set_power(binding_id, awake):
        snapshot = next(
            item for item in runtime.snapshots.values()
            if _binding_id(item) == binding_id
        )
        vllm.sleeping[snapshot.pod_ip] = not awake
        runtime.write_binding_annotations(
            Binding(
                snapshot.name,
                snapshot.model,
                _binding_slot(snapshot),
                awake=awake,
            ),
            state="awake" if awake else "sleeping",
        )
        return {"binding_id": binding_id, "awake": awake}

    executor = FleetRepairExecutor(
        runtime_ops=runtime,
        vllm_ops=vllm,
        safety_gate=FakeSafety(),
        poll_interval_s=0,
        sleep=lambda _seconds: None,
    )
    executor.run(
        operation,
        awake_binding_ids=[first.binding_id],
        reconcile=lambda strict: reconciles.append(strict) or {},
        set_binding_power=set_power,
        audit=lambda: {"healthy": True, "version": 7},
    )

    assert runtime.scale_calls == [(missing.name, 0), (missing.name, 1)]
    assert vllm.sleeping["10.0.0.1"] is False
    rebuilt = next(
        snapshot
        for snapshot in runtime.snapshots.values()
        if _binding_id(snapshot) == missing.binding_id
    )
    assert vllm.sleeping[rebuilt.pod_ip] is True
    assert reconciles == [True, True]
    assert any(phase == "verified" for phase, _ in operation.phases)


def test_repair_rejects_overlapping_awake_targets_before_mutation():
    first = _deployment("m1", "m1", "node-a", (0, 1))
    second = _deployment("m2", "m2", "node-a", (1,))
    vllm = FakeVllm({})
    runtime = FakeRuntime([first, second], [], vllm)
    executor = FleetRepairExecutor(
        runtime_ops=runtime,
        vllm_ops=vllm,
        safety_gate=FakeSafety(),
    )

    with pytest.raises(ValueError, match="awake targets overlap"):
        executor.run(
            FakeOperation(),
            awake_binding_ids=[first.binding_id, second.binding_id],
            reconcile=lambda _strict: {},
            set_binding_power=lambda _binding_id, _awake: {},
            audit=lambda: {"healthy": True},
        )

    assert runtime.scale_calls == []


def _binding_slot(snapshot):
    from tre_sm.allocator.slots import Slot

    return Slot(snapshot.node, _gpu_ids(snapshot))

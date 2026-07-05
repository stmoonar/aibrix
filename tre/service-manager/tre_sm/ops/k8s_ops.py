from __future__ import annotations

import time
from typing import Protocol

from gen_model_manifests import build_model_deployment, deployment_name
from tre_common.registry import Registry
from tre_sm.allocator.slots import Binding
from tre_sm.allocator.topology import (
    CUDA_VISIBLE_DEVICES,
    GPU_IDS_ANNOTATION,
    STATE_ANNOTATION,
    K8sPodSnapshot,
)
from tre_sm.state.reconcile import POD_STATE_AWAKE, POD_STATE_HIDDEN, POD_STATE_SLEEPING


MODEL_LABEL = "model.aibrix.ai/name"
ROUTABLE_LABEL = "tre.aibrix.io/routable"
_VALID_STATES = {POD_STATE_AWAKE, POD_STATE_SLEEPING, POD_STATE_HIDDEN}


class K8sApi(Protocol):
    def list_namespaced_pod(self, *, namespace: str, label_selector: str | None = None): ...

    def patch_namespaced_pod(self, *, name: str, namespace: str, body: dict) -> None: ...


class K8sDeploymentApi(Protocol):
    def delete_namespaced_deployment(self, *, name: str, namespace: str): ...

    def create_namespaced_deployment(self, *, namespace: str, body: dict): ...


class K8sOps:
    def __init__(
        self,
        *,
        api: K8sApi,
        namespace: str,
        registry: Registry | None = None,
        apps_api: K8sDeploymentApi | None = None,
    ) -> None:
        self._api = api
        self._apps_api = apps_api or api
        self._namespace = namespace
        self._registry = registry

    def list_pod_snapshots(self, *, model: str | None = None) -> list[K8sPodSnapshot]:
        selector = f"{MODEL_LABEL}={model}" if model else None
        pods = _items(self._api.list_namespaced_pod(namespace=self._namespace, label_selector=selector))
        return self._snapshots_from_pods(pods)

    def _snapshots_from_pods(self, pods) -> list[K8sPodSnapshot]:
        snapshots: list[K8sPodSnapshot] = []
        for pod in pods:
            metadata = _metadata(pod)
            spec = _spec(pod)
            if _field(metadata, "deletionTimestamp", "deletion_timestamp"):
                continue
            if _status(pod).get("phase") != "Running":
                continue
            labels = metadata.get("labels") or {}
            model_name = labels.get(MODEL_LABEL)
            if not model_name:
                continue
            annotations = dict(metadata.get("annotations") or {})
            if GPU_IDS_ANNOTATION not in annotations and labels.get(GPU_IDS_ANNOTATION):
                annotations[GPU_IDS_ANNOTATION] = str(labels[GPU_IDS_ANNOTATION])
            snapshots.append(
                K8sPodSnapshot(
                    name=str(metadata["name"]),
                    model=str(model_name),
                    node=str(_field(spec, "nodeName", "node_name")),
                    env=_container_env(pod),
                    annotations=annotations,
                    pod_ip=_optional_field(_status(pod), "podIP", "pod_ip"),
                )
            )
        return sorted(snapshots, key=lambda item: item.name)

    def write_binding_annotations(self, binding: Binding, *, state: str) -> None:
        if state not in _VALID_STATES:
            raise ValueError(f"unknown pod state: {state}")
        body = {
            "metadata": {
                "annotations": {
                    GPU_IDS_ANNOTATION: ",".join(str(gpu) for gpu in binding.slot.gpu_ids),
                    STATE_ANNOTATION: state,
                },
                "labels": {ROUTABLE_LABEL: "true" if state == POD_STATE_AWAKE else "false"},
            }
        }
        self._api.patch_namespaced_pod(name=binding.serve_id, namespace=self._namespace, body=body)

    def wait_pod_unroutable(self, binding: Binding, *, timeout_s: float = 30.0, interval_s: float = 0.5) -> None:
        deadline = time.monotonic() + timeout_s
        selector = f"{MODEL_LABEL}={binding.model},{ROUTABLE_LABEL}=true"
        while time.monotonic() < deadline:
            pods = _items(self._api.list_namespaced_pod(namespace=self._namespace, label_selector=selector))
            routable_names = {str(_metadata(pod).get("name")) for pod in pods}
            if binding.serve_id not in routable_names:
                return
            time.sleep(interval_s)
        raise TimeoutError(f"pod {binding.serve_id} remained routable before timeout")

    def delete_model_deployment(self, binding: Binding) -> str:
        name = deployment_name(binding.model, binding.slot.node, binding.slot.gpu_ids)
        self._apps_api.delete_namespaced_deployment(name=name, namespace=self._namespace)
        return name

    def create_model_deployment(self, model: str, slot) -> str:
        if self._registry is None:
            raise ValueError("registry is required to create model deployments")
        body = build_model_deployment(self._registry, model, slot.node, tuple(slot.gpu_ids))
        self._apps_api.create_namespaced_deployment(namespace=self._namespace, body=body)
        return str(body["metadata"]["name"])

    def wait_pod_deleted(self, serve_id: str, *, timeout_s: float = 30.0, interval_s: float = 0.5) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if all(snapshot.name != serve_id for snapshot in self.list_pod_snapshots()):
                return
            time.sleep(interval_s)
        raise TimeoutError(f"pod {serve_id} did not disappear before timeout")

    def wait_pod_ready(self, serve_id: str, *, timeout_s: float = 120.0, interval_s: float = 1.0) -> K8sPodSnapshot:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            pods = _items(self._api.list_namespaced_pod(namespace=self._namespace, label_selector=f"app={serve_id}"))
            snapshots = self._snapshots_from_pods(pods)
            if snapshots:
                return snapshots[0]
            for snapshot in self.list_pod_snapshots():
                if snapshot.name == serve_id:
                    return snapshot
            time.sleep(interval_s)
        raise TimeoutError(f"pod {serve_id} was not ready before timeout")


def _metadata(pod) -> dict:
    return _section(pod, "metadata")


def _spec(pod) -> dict:
    return _section(pod, "spec")


def _status(pod) -> dict:
    return _section(pod, "status")


def _section(pod, name: str) -> dict:
    if isinstance(pod, dict):
        return pod.get(name) or {}
    section = getattr(pod, name, None)
    if section is None:
        return {}
    if isinstance(section, dict):
        return section
    return section.to_dict()


def _container_env(pod) -> dict[str, str]:
    env: dict[str, str] = {}
    for container in _spec(pod).get("containers") or []:
        for item in container.get("env") or []:
            if item.get("name") == CUDA_VISIBLE_DEVICES:
                env[CUDA_VISIBLE_DEVICES] = str(item.get("value", ""))
    return env


def _field(section: dict, camel: str, snake: str):
    if camel in section:
        return section[camel]
    return section[snake]


def _optional_field(section: dict, camel: str, snake: str):
    if camel in section:
        return section[camel]
    return section.get(snake)


def _items(value):
    if isinstance(value, list):
        return value
    items = getattr(value, "items", None)
    if items is not None:
        return list(items)
    return list(value)

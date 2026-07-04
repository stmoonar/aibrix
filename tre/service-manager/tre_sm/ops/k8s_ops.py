from __future__ import annotations

from typing import Protocol

from tre_sm.allocator.slots import Binding
from tre_sm.allocator.topology import (
    CUDA_VISIBLE_DEVICES,
    GPU_IDS_ANNOTATION,
    STATE_ANNOTATION,
    K8sPodSnapshot,
)
from tre_sm.state.reconcile import POD_STATE_AWAKE, POD_STATE_HIDDEN, POD_STATE_SLEEPING


MODEL_LABEL = "model.aibrix.ai/name"
_VALID_STATES = {POD_STATE_AWAKE, POD_STATE_SLEEPING, POD_STATE_HIDDEN}


class K8sApi(Protocol):
    def list_namespaced_pod(self, *, namespace: str, label_selector: str | None = None): ...

    def patch_namespaced_pod(self, *, name: str, namespace: str, body: dict) -> None: ...


class K8sOps:
    def __init__(self, *, api: K8sApi, namespace: str) -> None:
        self._api = api
        self._namespace = namespace

    def list_pod_snapshots(self, *, model: str | None = None) -> list[K8sPodSnapshot]:
        selector = f"{MODEL_LABEL}={model}" if model else None
        pods = _items(self._api.list_namespaced_pod(namespace=self._namespace, label_selector=selector))
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
            snapshots.append(
                K8sPodSnapshot(
                    name=str(metadata["name"]),
                    model=str(model_name),
                    node=str(_field(spec, "nodeName", "node_name")),
                    env=_container_env(pod),
                    annotations=dict(metadata.get("annotations") or {}),
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
                }
            }
        }
        self._api.patch_namespaced_pod(name=binding.serve_id, namespace=self._namespace, body=body)


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

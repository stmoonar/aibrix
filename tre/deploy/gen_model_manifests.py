from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import yaml

from tre_common.registry import ModelSpec, NodeSpec, Registry, load_registry

ROUTABLE_LABEL = "tre.aibrix.io/routable"
GPU_UUIDS_ANNOTATION = "tre.aibrix.io/gpu-uuids"
MAX_BOUND_PER_GPU = 3


def feasible_slots(registry: Registry, model: ModelSpec) -> list[tuple[str, tuple[int, ...]]]:
    slots: list[tuple[str, tuple[int, ...]]] = []
    for node in registry.topology().nodes:
        if model.tp_size == 1:
            slots.extend((node.name, (gpu,)) for gpu in range(node.gpus))
        elif model.tp_size == 2:
            slots.extend((node.name, tuple(slot)) for slot in node.two_gpu_slots)
    return slots[: model.max_replicas]


def build_deployments(registry: Registry) -> list[dict]:
    deployments: list[dict] = []
    nodes = {node.name: node for node in registry.topology().nodes}
    bound_counts: dict[tuple[str, int], int] = {}
    for model in registry.models():
        for node_name, gpu_ids in feasible_slots(registry, model):
            _record_bound_budget(bound_counts, node_name, gpu_ids)
            deployments.append(_deployment(model, nodes[node_name], gpu_ids))
    return deployments


def deployment_name(model_name: str, node_name: str, gpu_ids: tuple[int, ...]) -> str:
    gpu_value = ",".join(str(gpu) for gpu in gpu_ids)
    return f"{_dns_name(model_name)}-{_dns_name(node_name)}-gpu-{gpu_value.replace(',', '-')}"


def build_model_deployment(registry: Registry, model_name: str, node_name: str, gpu_ids: tuple[int, ...]) -> dict:
    nodes = {node.name: node for node in registry.topology().nodes}
    return _deployment(registry.model(model_name), nodes[node_name], gpu_ids)


def build_services(registry: Registry) -> list[dict]:
    return [_service(model) for model in registry.models()]


def build_resources(registry: Registry) -> list[dict]:
    return build_services(registry) + build_deployments(registry)


def write_manifests(registry: Registry, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.yaml"):
        old.unlink()
    written: list[Path] = []
    for resource in build_resources(registry):
        path = output_dir / f"{resource['metadata']['name']}.yaml"
        path.write_text(yaml.safe_dump(resource, sort_keys=False), encoding="utf-8")
        written.append(path)
    resources = [path.name for path in written]
    (output_dir / "kustomization.yaml").write_text(
        yaml.safe_dump({"apiVersion": "kustomize.config.k8s.io/v1beta1", "kind": "Kustomization", "resources": resources}, sort_keys=False),
        encoding="utf-8",
    )
    return written


def _service(model: ModelSpec) -> dict:
    labels = {"model.aibrix.ai/name": model.name}
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": _dns_name(model.name), "namespace": "default", "labels": labels},
        "spec": {
            "selector": labels | {ROUTABLE_LABEL: "true"},
            "ports": [
                {
                    "name": "http",
                    "port": 8000,
                    "targetPort": 8000,
                    "protocol": "TCP",
                }
            ],
        },
    }


def _deployment(model: ModelSpec, node: NodeSpec, gpu_ids: tuple[int, ...]) -> dict:
    gpu_value = ",".join(str(gpu) for gpu in gpu_ids)
    gpu_label_value = "-".join(str(gpu) for gpu in gpu_ids)
    cuda_value = ",".join(str(index) for index in range(model.tp_size))
    gpu_uuid_value = ",".join(_gpu_uuids_for(node, gpu_ids))
    name = deployment_name(model.name, node.name, gpu_ids)
    command = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--uvicorn-log-level",
        "warning",
        "--model",
        model.weights_path,
        "--served-model-name",
        model.name,
        "--enable_sleep_mode",
    ]
    if model.tp_size > 1:
        command.extend(["--tensor-parallel-size", str(model.tp_size)])
    command.extend(model.vllm_extra_args)
    labels = {
        "model.aibrix.ai/name": model.name,
        "model.aibrix.ai/port": "8000",
        "tre.aibrix.io/managed": "true",
        "tre.aibrix.io/node": node.name,
        "tre.aibrix.io/gpu-ids": gpu_label_value,
        ROUTABLE_LABEL: "true",
    }
    annotations = {"tre.aibrix.io/gpu-ids": gpu_value, GPU_UUIDS_ANNOTATION: gpu_uuid_value}
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "namespace": "default", "labels": labels, "annotations": annotations},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "metadata": {"labels": labels | {"app": name}, "annotations": annotations},
                "spec": {
                    "nodeName": node.name,
                    "volumes": [
                        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "20Gi"}},
                        {"name": "models-volume", "hostPath": {"path": "/data"}},
                    ],
                    "containers": [
                        {
                            "name": "vllm-openai",
                            "image": model.vllm_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": command,
                            "env": [
                                {"name": "NVIDIA_VISIBLE_DEVICES", "value": gpu_uuid_value},
                                {"name": "CUDA_VISIBLE_DEVICES", "value": cuda_value},
                                {"name": "VLLM_SERVER_DEV_MODE", "value": "1"},
                                {"name": "VLLM_WORKER_MULTIPROC_METHOD", "value": "spawn"},
                                {"name": "VLLM_USE_MODELSCOPE", "value": "True"},
                            ],
                            "ports": [{"containerPort": 8000, "protocol": "TCP"}],
                            "resources": {},
                            "volumeMounts": [
                                {"name": "shm", "mountPath": "/dev/shm"},
                                {"name": "models-volume", "mountPath": "/data"},
                            ],
                        }
                    ],
                },
            },
        },
    }


def _dns_name(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")


def _gpu_uuids_for(node: NodeSpec, gpu_ids: tuple[int, ...]) -> tuple[str, ...]:
    if len(node.gpu_uuids) != node.gpus:
        raise ValueError(f"node {node.name}: gpu_uuids length does not match gpus")
    return tuple(node.gpu_uuids[gpu] for gpu in gpu_ids)


def _record_bound_budget(bound_counts: dict[tuple[str, int], int], node_name: str, gpu_ids: tuple[int, ...]) -> None:
    for gpu in gpu_ids:
        key = (node_name, gpu)
        bound_counts[key] = bound_counts.get(key, 0) + 1
        if bound_counts[key] > MAX_BOUND_PER_GPU:
            raise ValueError(
                f"gpu bound budget exceeded for {node_name}/{gpu}: {bound_counts[key]} > {MAX_BOUND_PER_GPU}"
            )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="tre/deploy/registry.yaml")
    parser.add_argument("--output-dir", default="tre/deploy/models")
    args = parser.parse_args(list(argv) if argv is not None else None)
    registry = load_registry(args.registry)
    errors = registry.validate()
    if errors:
        raise SystemExit("registry validation failed:" + chr(10) + chr(10).join(errors))
    written = write_manifests(registry, Path(args.output_dir))
    print(f"wrote {len(written)} resources to {args.output_dir}")


if __name__ == "__main__":
    main()

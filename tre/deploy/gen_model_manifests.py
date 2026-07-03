from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import yaml

from tre_common.registry import ModelSpec, Registry, load_registry


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
    for model in registry.models():
        for node_name, gpu_ids in feasible_slots(registry, model):
            deployments.append(_deployment(model, node_name, gpu_ids))
    return deployments


def write_manifests(registry: Registry, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.yaml"):
        old.unlink()
    written: list[Path] = []
    for deployment in build_deployments(registry):
        path = output_dir / f"{deployment['metadata']['name']}.yaml"
        path.write_text(yaml.safe_dump(deployment, sort_keys=False), encoding="utf-8")
        written.append(path)
    resources = [path.name for path in written]
    (output_dir / "kustomization.yaml").write_text(
        yaml.safe_dump({"apiVersion": "kustomize.config.k8s.io/v1beta1", "kind": "Kustomization", "resources": resources}, sort_keys=False),
        encoding="utf-8",
    )
    return written


def _deployment(model: ModelSpec, node_name: str, gpu_ids: tuple[int, ...]) -> dict:
    gpu_value = ",".join(str(gpu) for gpu in gpu_ids)
    name = f"{_dns_name(model.name)}-{_dns_name(node_name)}-gpu-{gpu_value.replace(',', '-')}"
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
        "tre.aibrix.io/node": node_name,
        "tre.aibrix.io/gpu-ids": gpu_value,
    }
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "namespace": "default", "labels": labels},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "metadata": {"labels": labels | {"app": name}},
                "spec": {
                    "nodeSelector": {"kubernetes.io/hostname": node_name},
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
                                {"name": "CUDA_VISIBLE_DEVICES", "value": gpu_value},
                                {"name": "VLLM_SERVER_DEV_MODE", "value": "1"},
                                {"name": "VLLM_WORKER_MULTIPROC_METHOD", "value": "spawn"},
                                {"name": "VLLM_USE_MODELSCOPE", "value": "True"},
                            ],
                            "ports": [{"containerPort": 8000, "protocol": "TCP"}],
                            "resources": {
                                "limits": {"nvidia.com/gpu": str(model.tp_size)},
                                "requests": {"nvidia.com/gpu": str(model.tp_size)},
                            },
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
    print(f"wrote {len(written)} deployments to {args.output_dir}")


if __name__ == "__main__":
    main()

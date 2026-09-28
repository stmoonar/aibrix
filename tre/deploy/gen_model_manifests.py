from __future__ import annotations

import argparse
import os
from dataclasses import replace
import re
from pathlib import Path
from typing import Iterable

import yaml

from tre_common.bindings import MAX_BOUND_PER_GPU, feasible_slots, render_binding_set  # noqa: F401 (re-exported)
from tre_common.registry import (
    DEFAULT_ROUTE_TIMEOUT_S,
    DEFAULT_VLLM_ENV,
    POD_SERVING_PORT,
    VLLM_FEATURE_FLAGS,
    ModelSpec,
    NodeSpec,
    Registry,
    ReissueConfig,
    load_registry,
)

ROUTABLE_LABEL = "tre.aibrix.io/routable"
GPU_UUIDS_ANNOTATION = "tre.aibrix.io/gpu-uuids"
MODEL_LABEL = "model.aibrix.ai/name"
GATEWAY_NAMESPACE = "tre-v2"
GATEWAY_NAME = "tre-aibrix-eg"
GATEWAY_API_GROUP = "gateway.networking.k8s.io"
HTTPROUTE_API_VERSION = "v1"
HTTPROUTE_PLURAL = "httproutes"
HTTPROUTE_PATHS = (
    "/v1/completions",
    "/v1/chat/completions",
    "/v1/embeddings",
    "/generate",
    "/generatevideo",
)

#: The retry / continuation sidecar (tre/reissue, design 20260927-reissue-sidecar-v2):
#: its script ships in a ConfigMap and runs with the model's vLLM image.
REISSUE_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "reissue" / "tre_reissue" / "sidecar.py"
REISSUE_MOUNT_DIR = "/opt/tre-reissue"
REISSUE_SCRIPT_KEY = "sidecar.py"
REISSUE_CONTAINER = "tre-reissue-sidecar"

STARTUP_GATE_CLIENT = """\
import json, os, time, urllib.request
url = os.environ.get('TRE_SM_URL', 'http://tre-v2-service-manager.tre-v2.svc.cluster.local:8000') + '/v2/startup/admit'
payload = json.dumps({'pod_name': os.environ['POD_NAME'], 'pod_uid': os.environ['POD_UID']}).encode()
while True:
    try:
        request = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status == 200:
                print(response.read().decode(), flush=True)
                break
    except Exception as exc:
        print('TRE startup admission waiting:', exc, flush=True)
    time.sleep(2)
"""


def build_deployments(registry: Registry) -> list[dict]:
    # One Deployment per binding of the shared binding set (tre_common.bindings), the
    # same set the service-manager seeds its desired state from (plan D7).
    nodes = {node.name: node for node in registry.topology().nodes}
    reissue = reissue_spec(registry)
    return [
        _deployment(
            registry.model(spec.model),
            nodes[spec.node],
            spec.gpu_ids,
            reissue=reissue,
            vllm_env=registry.vllm_env_for(registry.model(spec.model)),
        )
        for spec in render_binding_set(registry)
    ]


def reissue_spec(registry: Registry) -> ReissueConfig | None:
    """The registry's reissue sidecar settings, or None when it is disabled. The gateway
    URL is resolved: reissue.gateway_url, else the gateway: section's stable Service."""
    spec = registry.reissue()
    if not spec.enabled:
        return None
    return replace(spec, gateway_url=spec.gateway_url or registry.gateway().internal_url)


def build_reissue_configmap(spec: ReissueConfig, *, script_path: Path = REISSUE_SCRIPT_PATH) -> dict:
    """The ConfigMap shipping the sidecar script (the vLLM image already has aiohttp)."""
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": spec.configmap,
            "namespace": spec.namespace,
            "labels": {"tre.aibrix.io/managed": "true", "app.kubernetes.io/name": REISSUE_CONTAINER},
        },
        "data": {REISSUE_SCRIPT_KEY: script_path.read_text(encoding="utf-8")},
    }


def deployment_name(model_name: str, node_name: str, gpu_ids: tuple[int, ...]) -> str:
    gpu_value = ",".join(str(gpu) for gpu in gpu_ids)
    return f"{_dns_name(model_name)}-{_dns_name(node_name)}-gpu-{gpu_value.replace(',', '-')}"


def build_model_deployment(registry: Registry, model_name: str, node_name: str, gpu_ids: tuple[int, ...]) -> dict:
    # Used by the service-manager for runtime creates: the registry's reissue section
    # applies, so a relocated binding looks exactly like a rendered one.
    nodes = {node.name: node for node in registry.topology().nodes}
    model = registry.model(model_name)
    return _deployment(
        model, nodes[node_name], gpu_ids, reissue=reissue_spec(registry), vllm_env=registry.vllm_env_for(model)
    )


def build_services(registry: Registry) -> list[dict]:
    return [_service(model) for model in registry.models()]


def build_httproutes(
    registry: Registry,
    *,
    gateway_namespace: str = GATEWAY_NAMESPACE,
    gateway_name: str = GATEWAY_NAME,
) -> list[dict]:
    return [
        build_model_httproute(
            model.name,
            gateway_namespace=gateway_namespace,
            gateway_name=gateway_name,
            request_timeout_s=registry.gateway().route_timeout_s,
        )
        for model in registry.models()
    ]


def build_model_httproute(
    model_name: str,
    *,
    model_namespace: str = "default",
    gateway_namespace: str = GATEWAY_NAMESPACE,
    gateway_name: str = GATEWAY_NAME,
    request_timeout_s: float = DEFAULT_ROUTE_TIMEOUT_S,
) -> dict:
    """The model's HTTPRoute; ``request_timeout_s`` = registry gateway.route_timeout_s
    (the same value caps the service-manager drain)."""
    service_name = _dns_name(model_name)
    labels = {MODEL_LABEL: model_name, "tre.aibrix.io/managed": "true"}
    return {
        "apiVersion": f"{GATEWAY_API_GROUP}/{HTTPROUTE_API_VERSION}",
        "kind": "HTTPRoute",
        "metadata": {
            "name": f"{service_name}-router",
            "namespace": gateway_namespace,
            "labels": labels,
        },
        "spec": {
            "parentRefs": [
                {
                    "group": GATEWAY_API_GROUP,
                    "kind": "Gateway",
                    "name": gateway_name,
                    "namespace": gateway_namespace,
                }
            ],
            "rules": [
                {
                    "backendRefs": [
                        {
                            "group": "",
                            "kind": "Service",
                            "name": service_name,
                            "namespace": model_namespace,
                            "port": 8000,
                            "weight": 1,
                        }
                    ],
                    "matches": [
                        {
                            "headers": [{"name": "model", "type": "Exact", "value": model_name}],
                            "path": {"type": "PathPrefix", "value": path},
                        }
                        for path in HTTPROUTE_PATHS
                    ],
                    "timeouts": {"request": route_timeout_text(request_timeout_s)},
                }
            ],
        },
    }


def route_timeout_text(seconds: float) -> str:
    """Gateway API duration of a route timeout in seconds (``150`` -> ``"150s"``)."""
    value = float(seconds)
    if value <= 0:
        raise ValueError(f"route timeout must be positive, got {seconds!r}")
    return f"{int(value)}s" if value.is_integer() else f"{int(round(value * 1000))}ms"


def build_referencegrant(
    *,
    model_namespace: str = "default",
    gateway_namespace: str = GATEWAY_NAMESPACE,
) -> dict:
    # Cross-namespace HTTPRoute (aibrix-system) -> Service (default) needs a
    # ReferenceGrant. Ship a TRE-managed one so  is
    # self-sufficient and does not depend on an AIBrix-base reserved grant.
    return {
        "apiVersion": f"{GATEWAY_API_GROUP}/v1beta1",
        "kind": "ReferenceGrant",
        "metadata": {
            "name": "tre-v2-model-referencegrant-in-default",
            "namespace": model_namespace,
            "labels": {"tre.aibrix.io/managed": "true"},
        },
        "spec": {
            "from": [
                {"group": GATEWAY_API_GROUP, "kind": "HTTPRoute", "namespace": gateway_namespace}
            ],
            "to": [{"group": "", "kind": "Service"}],
        },
    }


def build_resources(
    registry: Registry,
    *,
    gateway_namespace: str = GATEWAY_NAMESPACE,
    gateway_name: str = GATEWAY_NAME,
) -> list[dict]:
    reissue = reissue_spec(registry)
    return (
        [build_referencegrant(gateway_namespace=gateway_namespace)]
        + ([build_reissue_configmap(reissue)] if reissue is not None else [])
        + build_services(registry)
        + build_httproutes(registry, gateway_namespace=gateway_namespace, gateway_name=gateway_name)
        + build_deployments(registry)
    )


def write_manifests(
    registry: Registry,
    output_dir: Path,
    *,
    gateway_namespace: str = GATEWAY_NAMESPACE,
    gateway_name: str = GATEWAY_NAME,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.yaml"):
        old.unlink()
    written: list[Path] = []
    for resource in build_resources(registry, gateway_namespace=gateway_namespace, gateway_name=gateway_name):
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
    labels = {MODEL_LABEL: model.name}
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


def _deployment(
    model: ModelSpec,
    node: NodeSpec,
    gpu_ids: tuple[int, ...],
    *,
    reissue: ReissueConfig | None = None,
    vllm_env: dict[str, str] | None = None,
) -> dict:
    """``vllm_env``: the vLLM container environment besides the per-binding GPU variables
    (``Registry.vllm_env_for``); None = ``DEFAULT_VLLM_ENV``."""
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
    command.extend(model.vllm_args)
    if reissue is not None:
        # Fork features the sidecar builds on, only where the image declares them.
        for feature in model.vllm_features:
            command.extend(flag for flag in VLLM_FEATURE_FLAGS.get(feature, ()) if flag not in command)
    labels = {
        "model.aibrix.ai/name": model.name,
        "model.aibrix.ai/port": "8000",
        "tre.aibrix.io/managed": "true",
        "tre.aibrix.io/node": node.name,
        "tre.aibrix.io/gpu-ids": gpu_label_value,
        # A Pod is never routable merely because Kubernetes created it. The
        # service-manager flips this only after startup admission converges.
        ROUTABLE_LABEL: "false",
    }
    annotations = {
        "tre.aibrix.io/gpu-ids": gpu_value,
        GPU_UUIDS_ANNOTATION: gpu_uuid_value,
        "tre.aibrix.io/state": "hidden",
    }
    deployment = {
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
                    "initContainers": [
                        {
                            "name": "tre-startup-gate",
                            "image": model.vllm_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["python3", "-c", STARTUP_GATE_CLIENT],
                            "env": [
                                {
                                    "name": "POD_NAME",
                                    "valueFrom": {
                                        "fieldRef": {"fieldPath": "metadata.name"}
                                    },
                                },
                                {
                                    "name": "POD_UID",
                                    "valueFrom": {
                                        "fieldRef": {"fieldPath": "metadata.uid"}
                                    },
                                },
                            ],
                        }
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
                            ]
                            + [
                                {"name": key, "value": value}
                                for key, value in (DEFAULT_VLLM_ENV if vllm_env is None else vllm_env).items()
                            ],
                            "ports": [{"containerPort": 8000, "protocol": "TCP"}],
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": 8000},
                                "periodSeconds": 2,
                                "timeoutSeconds": 2,
                                "failureThreshold": 300,
                            },
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
    if reissue is not None:
        _add_reissue_sidecar(deployment, model, reissue)
    return deployment


def _add_reissue_sidecar(deployment: dict, model: ModelSpec, spec: ReissueConfig) -> None:
    """vLLM -> 127.0.0.1:<vllm_port>; the sidecar takes the serving port and the
    readiness probe (its /health is vLLM's /health, proxied), so everything that talks to
    the pod - Service, gateway target-pod, service-manager, scrapers - is unchanged."""
    pod = deployment["spec"]["template"]["spec"]
    vllm = pod["containers"][0]
    command = list(vllm["command"])
    command[command.index("--host") + 1] = "127.0.0.1"
    command[command.index("--port") + 1] = str(spec.vllm_port)
    vllm["command"] = command
    readiness = vllm.pop("readinessProbe")
    vllm.pop("ports", None)
    pod["volumes"].append({"name": REISSUE_CONTAINER, "configMap": {"name": spec.configmap, "defaultMode": 0o444}})
    env = [
        {"name": "TRE_REISSUE_LISTEN_PORT", "value": str(POD_SERVING_PORT)},
        {"name": "TRE_REISSUE_UPSTREAM_URL", "value": f"http://127.0.0.1:{spec.vllm_port}"},
        {"name": "TRE_GATEWAY_URL", "value": spec.gateway_url},
        {"name": "TRE_REISSUE_MODEL", "value": model.name},
        {"name": "TRE_REISSUE_MAX_DEPTH", "value": str(spec.max_depth)},
        {"name": "TRE_REISSUE_RETRY_ATTEMPTS", "value": str(spec.retry_attempts)},
        # Fail closed: /sleep without X-TRE-Hidden: 1 (the SM sends it after the hide) is 409.
        {"name": "TRE_REISSUE_REQUIRE_HIDDEN_HEADER", "value": "true"},
        {"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
        # Same image as vLLM, but this container must not get the GPUs.
        {"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
    ]
    env.extend({"name": key, "value": value} for key, value in sorted(spec.extra_env.items()))
    pod["containers"].append(
        {
            "name": REISSUE_CONTAINER,
            "image": spec.image or model.vllm_image,
            "imagePullPolicy": "IfNotPresent",
            "command": ["python3", f"{REISSUE_MOUNT_DIR}/{REISSUE_SCRIPT_KEY}"],
            "env": env,
            "ports": [{"containerPort": POD_SERVING_PORT, "protocol": "TCP"}],
            "readinessProbe": readiness,
            "resources": {
                "requests": {"cpu": spec.cpu_request, "memory": spec.memory_request},
                "limits": {"cpu": spec.cpu_limit, "memory": spec.memory_limit},
            },
            "volumeMounts": [{"name": REISSUE_CONTAINER, "mountPath": REISSUE_MOUNT_DIR, "readOnly": True}],
        }
    )


def _dns_name(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")


def _gpu_uuids_for(node: NodeSpec, gpu_ids: tuple[int, ...]) -> tuple[str, ...]:
    if len(node.gpu_uuids) != node.gpus:
        raise ValueError(f"node {node.name}: gpu_uuids length does not match gpus")
    return tuple(node.gpu_uuids[gpu] for gpu in gpu_ids)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="tre/deploy/registry.yaml")
    parser.add_argument("--output-dir", default="tre/deploy/models")
    parser.add_argument("--gateway-namespace", default=os.environ.get("TRE_GATEWAY_NAMESPACE", GATEWAY_NAMESPACE))
    parser.add_argument("--gateway-name", default=os.environ.get("TRE_GATEWAY_NAME", GATEWAY_NAME))
    args = parser.parse_args(list(argv) if argv is not None else None)
    registry = load_registry(args.registry)
    errors = registry.validate()
    if errors:
        raise SystemExit("registry validation failed:" + chr(10) + chr(10).join(errors))
    written = write_manifests(
        registry,
        Path(args.output_dir),
        gateway_namespace=args.gateway_namespace,
        gateway_name=args.gateway_name,
    )
    print(f"wrote {len(written)} resources to {args.output_dir}")


if __name__ == "__main__":
    main()

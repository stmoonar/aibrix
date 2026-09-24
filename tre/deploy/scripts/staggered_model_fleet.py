#!/usr/bin/env python3
"""Fail-closed, one-binding-at-a-time model fleet bring-up.

The command is dry-run by default. Execution intentionally requires both
--execute and --confirm-reset-fleet because the first phase scales every
managed model Deployment to zero before rebuilding the resident pool.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Callable
from urllib.request import Request, urlopen

import yaml


ROUTABLE_LABEL = "tre.aibrix.io/routable"
STATE_ANNOTATION = "tre.aibrix.io/state"
GPU_IDS_ANNOTATION = "tre.aibrix.io/gpu-ids"
MODEL_LABEL = "model.aibrix.ai/name"
MANAGED_LABEL = "tre.aibrix.io/managed"
MODE_KEY = "tre:v2:controller:mode"


@dataclass(frozen=True)
class FleetBinding:
    path: Path
    name: str
    model: str
    node: str
    gpu_ids: tuple[int, ...]
    manifest: dict

    @property
    def binding_id(self) -> str:
        gpu_ids = ",".join(str(gpu_id) for gpu_id in self.gpu_ids)
        return f"{self.model}/{self.node}/{gpu_ids}"


class Kubectl:
    def __init__(self, command: str = "kubectl") -> None:
        self._command = command

    def run(self, *args: str, input_text: str | None = None) -> str:
        result = subprocess.run(
            [self._command, *args],
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"kubectl {' '.join(args)} failed: {result.stderr.strip()}"
            )
        return result.stdout.strip()


def discover_bindings(models_dir: Path) -> list[FleetBinding]:
    bindings: list[FleetBinding] = []
    for path in sorted(models_dir.glob("*.yaml")):
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("kind") != "Deployment":
            continue
        metadata = manifest.get("metadata") or {}
        template = ((manifest.get("spec") or {}).get("template") or {})
        labels = (template.get("metadata") or {}).get("labels") or {}
        annotations = (template.get("metadata") or {}).get("annotations") or {}
        spec = template.get("spec") or {}
        gpu_text = str(annotations[GPU_IDS_ANNOTATION])
        bindings.append(
            FleetBinding(
                path=path,
                name=str(metadata["name"]),
                model=str(labels[MODEL_LABEL]),
                node=str(spec["nodeName"]),
                gpu_ids=tuple(int(part) for part in gpu_text.split(",")),
                manifest=manifest,
            )
        )
    if not bindings:
        raise ValueError(f"no model Deployments found in {models_dir}")
    ids = [binding.binding_id for binding in bindings]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate stable binding_id in model manifests")
    return sorted(
        bindings,
        key=lambda item: (item.node, item.gpu_ids, item.model, item.name),
    )


def provisioning_manifest(binding: FleetBinding) -> dict:
    manifest = copy.deepcopy(binding.manifest)
    manifest["spec"]["replicas"] = 0
    manifest["metadata"].setdefault("labels", {})[ROUTABLE_LABEL] = "false"
    pod_metadata = manifest["spec"]["template"].setdefault("metadata", {})
    pod_metadata.setdefault("labels", {})[ROUTABLE_LABEL] = "false"
    # "hidden" is an existing SM state: physically awake while loading but
    # deliberately absent from Service endpoints. After /sleep it becomes
    # "sleeping". Avoid inventing an annotation value older SMs cannot parse.
    pod_metadata.setdefault("annotations", {})[STATE_ANNOTATION] = "hidden"
    return manifest


def non_deployment_manifests(
    models_dir: Path, *, allowed_namespace: str = "default"
) -> list[Path]:
    result: list[Path] = []
    for path in sorted(models_dir.glob("*.yaml")):
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        namespace = (manifest.get("metadata") or {}).get("namespace") if isinstance(manifest, dict) else None
        if (
            isinstance(manifest, dict)
            and namespace == allowed_namespace
            and manifest.get("kind") not in {
            "Deployment",
            "Kustomization",
            }
        ):
            result.append(path)
    return result


def parse_args() -> argparse.Namespace:
    deploy_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=deploy_dir / "models")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--tre-namespace", default="tre-v2")
    parser.add_argument("--sm-url")
    parser.add_argument("--wake", action="append", default=[], metavar="BINDING_ID")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-reset-fleet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bindings = discover_bindings(args.models_dir)
    unknown_wakes = sorted(set(args.wake) - {binding.binding_id for binding in bindings})
    if unknown_wakes:
        raise SystemExit(f"unknown --wake binding_id(s): {unknown_wakes}")
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "bindings": [binding.binding_id for binding in bindings],
        "wake": args.wake,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    if not args.confirm_reset_fleet:
        raise SystemExit("--execute requires --confirm-reset-fleet")

    kubectl = Kubectl(args.kubectl)
    _assert_controller_observe(kubectl, args.tre_namespace)
    _assert_disk_healthy(kubectl)
    sm_url = args.sm_url or _discover_sm_url(kubectl, args.tre_namespace)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = args.evidence_dir or Path(f"/tmp/tre-staggered-fleet-{stamp}")
    evidence_dir.mkdir(parents=True, exist_ok=False)
    events_path = evidence_dir / "events.jsonl"

    def record(event: str, **payload: object) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    (evidence_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Deliberately do not apply cross-namespace HTTPRoutes here. Fleet repair is
    # confined to the model namespace and never mutates shared aibrix-system.
    for path in non_deployment_manifests(
        args.models_dir, allowed_namespace=args.namespace
    ):
        kubectl.run("apply", "-f", str(path))
    for binding in bindings:
        kubectl.run(
            "apply",
            "-f",
            "-",
            input_text=yaml.safe_dump(provisioning_manifest(binding), sort_keys=False),
        )
    _wait_no_managed_pods(kubectl, args.namespace, args.timeout_s)
    _http_json(
        f"{sm_url}/v2/reconcile",
        method="POST",
        payload={"drop_missing": True},
    )
    record("fleet_scaled_zero_and_ghosts_removed")

    for binding in bindings:
        _assert_controller_observe(kubectl, args.tre_namespace)
        _assert_disk_healthy(kubectl)
        _assert_residents_sleeping(kubectl, args.namespace, binding)
        kubectl.run(
            "-n",
            args.namespace,
            "scale",
            f"deployment/{binding.name}",
            "--replicas=1",
        )
        pod = _wait_binding_ready(
            kubectl, args.namespace, binding, timeout_s=args.timeout_s
        )
        pod_ip = pod["status"]["podIP"]
        _wait_vllm_ready(pod_ip, timeout_s=args.timeout_s)
        # Bring-up pods are unroutable (tre.aibrix.io/routable=false until startup
        # admission), so the reissue sidecar's hide-before-sleep check is satisfied.
        _http_json(f"http://{pod_ip}:8000/sleep", method="POST", headers={"X-TRE-Hidden": "1"})
        _wait_until(
            lambda: _is_sleeping(pod_ip) is True,
            timeout_s=120.0,
            description=f"{binding.binding_id} to sleep",
        )
        pod_name = pod["metadata"]["name"]
        kubectl.run(
            "-n",
            args.namespace,
            "annotate",
            "pod",
            pod_name,
            f"{STATE_ANNOTATION}=sleeping",
            "--overwrite",
        )
        kubectl.run(
            "-n",
            args.namespace,
            "label",
            "pod",
            pod_name,
            f"{ROUTABLE_LABEL}=false",
            "--overwrite",
        )
        record(
            "resident_sleeping",
            binding_id=binding.binding_id,
            pod=pod_name,
            pod_ip=pod_ip,
        )

    reconcile = _http_json(
        f"{sm_url}/v2/reconcile",
        method="POST",
        payload={"drop_missing": True},
    )
    record("reconciled_resident_pool", response=reconcile)
    for binding_id in args.wake:
        state = _http_json(f"{sm_url}/v2/state")
        target = next(
            item for item in state["bindings"] if item["binding_id"] == binding_id
        )
        binding = next(item for item in bindings if item.binding_id == binding_id)
        _assert_residents_sleeping(kubectl, args.namespace, binding)
        response = _http_json(
            f"{sm_url}/v2/bindings/{target['serve_id']}/power",
            method="PUT",
            payload={"awake": True},
        )
        record("baseline_woken", binding_id=binding_id, response=response)

    audit = _http_json(f"{sm_url}/v2/audit")
    (evidence_dir / "final-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not audit.get("healthy"):
        raise RuntimeError(f"final service-manager audit failed: {audit['issues']}")
    print(f"fleet recovery complete; evidence: {evidence_dir}")
    return 0


def _assert_controller_observe(kubectl: Kubectl, namespace: str) -> None:
    mode = kubectl.run(
        "-n", namespace, "exec", "deploy/tre-v2-redis", "--",
        "redis-cli", "--raw", "GET", MODE_KEY,
    )
    if mode.strip() != "observe":
        raise RuntimeError(f"controller mode must be observe, got {mode!r}")


def _assert_disk_healthy(kubectl: Kubectl) -> None:
    payload = json.loads(kubectl.run("get", "nodes", "-o", "json"))
    pressured = []
    for node in payload.get("items", []):
        conditions = node.get("status", {}).get("conditions", [])
        if any(
            item.get("type") == "DiskPressure" and item.get("status") == "True"
            for item in conditions
        ):
            pressured.append(node["metadata"]["name"])
    if pressured:
        raise RuntimeError(f"DiskPressure blocks fleet repair: {pressured}")


def _discover_sm_url(kubectl: Kubectl, namespace: str) -> str:
    ip = kubectl.run(
        "-n", namespace, "get", "svc", "tre-v2-service-manager",
        "-o", "jsonpath={.spec.clusterIP}",
    )
    return f"http://{ip}:8000"


def _managed_pods(kubectl: Kubectl, namespace: str) -> list[dict]:
    payload = json.loads(
        kubectl.run(
            "-n", namespace, "get", "pods", "-l", f"{MANAGED_LABEL}=true",
            "-o", "json",
        )
    )
    return payload.get("items", [])


def _wait_no_managed_pods(kubectl: Kubectl, namespace: str, timeout_s: float) -> None:
    _wait_until(
        lambda: not _managed_pods(kubectl, namespace),
        timeout_s=timeout_s,
        description="all managed model pods to terminate",
    )


def _assert_residents_sleeping(
    kubectl: Kubectl, namespace: str, target: FleetBinding
) -> None:
    for pod in _managed_pods(kubectl, namespace):
        metadata = pod.get("metadata", {})
        labels = metadata.get("labels", {})
        if labels.get("tre.aibrix.io/node") != target.node:
            continue
        gpu_ids = {
            int(part)
            for part in str(labels.get("tre.aibrix.io/gpu-ids", "")).split("-")
            if part
        }
        if not gpu_ids.intersection(target.gpu_ids):
            continue
        pod_ip = pod.get("status", {}).get("podIP")
        if not pod_ip or _is_sleeping(pod_ip) is not True:
            raise RuntimeError(
                f"resident {metadata.get('name')} overlaps {target.binding_id} and is not confirmed asleep"
            )


def _wait_binding_ready(
    kubectl: Kubectl,
    namespace: str,
    binding: FleetBinding,
    *,
    timeout_s: float,
) -> dict:
    found: dict = {}

    def ready() -> bool:
        nonlocal found
        payload = json.loads(
            kubectl.run(
                "-n", namespace, "get", "pods", "-l", f"app={binding.name}",
                "-o", "json",
            )
        )
        items = payload.get("items", [])
        if len(items) != 1:
            return False
        pod = items[0]
        statuses = pod.get("status", {}).get("containerStatuses", [])
        if pod.get("status", {}).get("phase") != "Running":
            return False
        if not statuses or not all(item.get("ready") for item in statuses):
            return False
        if not pod.get("status", {}).get("podIP"):
            return False
        found = pod
        return True

    _wait_until(ready, timeout_s=timeout_s, description=f"{binding.binding_id} pod ready")
    return found


def _wait_vllm_ready(pod_ip: str, *, timeout_s: float) -> None:
    _wait_until(
        lambda: _is_sleeping(pod_ip) is not None,
        timeout_s=timeout_s,
        description=f"vLLM {pod_ip} HTTP readiness",
    )


def _is_sleeping(pod_ip: str) -> bool | None:
    try:
        payload = _http_json(f"http://{pod_ip}:8000/is_sleeping", timeout_s=5.0)
    except Exception:
        return None
    if isinstance(payload, bool):
        return payload
    if isinstance(payload, dict) and "is_sleeping" in payload:
        return bool(payload["is_sleeping"])
    return None


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
    timeout_s: float = 30.0,
    headers: dict[str, str] | None = None,
):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urlopen(request, timeout=timeout_s) as response:
        text = response.read().decode("utf-8").strip()
    return json.loads(text) if text else {}


def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_s: float,
    description: str,
    interval_s: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval_s)
    raise TimeoutError(f"timed out waiting for {description}")


if __name__ == "__main__":
    raise SystemExit(main())

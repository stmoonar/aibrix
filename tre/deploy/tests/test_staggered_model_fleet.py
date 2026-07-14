import importlib.util
from pathlib import Path
import sys

import yaml


SCRIPT = Path(__file__).parents[1] / "scripts" / "staggered_model_fleet.py"
SPEC = importlib.util.spec_from_file_location("staggered_model_fleet", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _deployment(name, model, node, gpu_ids):
    gpu_text = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "labels": {MODULE.ROUTABLE_LABEL: "true"}},
        "spec": {
            "replicas": 1,
            "template": {
                "metadata": {
                    "labels": {
                        MODULE.MODEL_LABEL: model,
                        MODULE.ROUTABLE_LABEL: "true",
                    },
                    "annotations": {MODULE.GPU_IDS_ANNOTATION: gpu_text},
                },
                "spec": {"nodeName": node},
            },
        },
    }


def test_discover_bindings_uses_stable_model_node_gpu_identity(tmp_path):
    manifest = _deployment("pod-template-name", "m1", "node-a", (0, 1))
    (tmp_path / "binding.yaml").write_text(
        yaml.safe_dump(manifest), encoding="utf-8"
    )

    [binding] = MODULE.discover_bindings(tmp_path)

    assert binding.binding_id == "m1/node-a/0,1"
    assert binding.name == "pod-template-name"


def test_provisioning_manifest_is_scaled_zero_and_fail_closed(tmp_path):
    manifest = _deployment("m1-node-a-gpu-0", "m1", "node-a", (0,))
    path = tmp_path / "binding.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    [binding] = MODULE.discover_bindings(tmp_path)

    rendered = MODULE.provisioning_manifest(binding)

    assert rendered["spec"]["replicas"] == 0
    assert rendered["metadata"]["labels"][MODULE.ROUTABLE_LABEL] == "false"
    metadata = rendered["spec"]["template"]["metadata"]
    assert metadata["labels"][MODULE.ROUTABLE_LABEL] == "false"
    assert metadata["annotations"][MODULE.STATE_ANNOTATION] == "hidden"
    assert manifest["spec"]["replicas"] == 1


def test_non_deployment_manifests_excludes_deployment_and_kustomization(tmp_path):
    resources = [
        ("service.yaml", {"kind": "Service", "metadata": {"name": "m1", "namespace": "default"}}),
        ("route.yaml", {"kind": "HTTPRoute", "metadata": {"name": "m1", "namespace": "aibrix-system"}}),
        ("deployment.yaml", _deployment("m1", "m1", "node-a", (0,))),
        ("kustomization.yaml", {"kind": "Kustomization", "resources": []}),
    ]
    for name, manifest in resources:
        (tmp_path / name).write_text(yaml.safe_dump(manifest), encoding="utf-8")

    selected = MODULE.non_deployment_manifests(tmp_path)

    assert [path.name for path in selected] == ["service.yaml"]

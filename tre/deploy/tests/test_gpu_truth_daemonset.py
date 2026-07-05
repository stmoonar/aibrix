from __future__ import annotations

from pathlib import Path

import yaml

import gen_gpu_truth_manifest as gen


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = DEPLOY_ROOT / "overlays" / "tre-v2" / "gpu-truth.yaml"
AGENT_SCRIPT = DEPLOY_ROOT / "scripts" / "gpu_truth_agent.py"


def _docs() -> list[dict]:
    return list(yaml.safe_load_all(MANIFEST.read_text(encoding="utf-8")))


def _by_kind(kind: str) -> dict:
    return next(doc for doc in _docs() if doc["kind"] == kind)


def test_configmap_script_is_single_source_copy() -> None:
    configmap = _by_kind("ConfigMap")
    assert configmap["metadata"]["name"] == "tre-v2-gpu-truth-agent"
    assert configmap["metadata"]["namespace"] == "tre-v2"
    embedded = configmap["data"]["gpu_truth_agent.py"]
    source = AGENT_SCRIPT.read_text(encoding="utf-8")
    # Block scalar round-trips to the original script; guard against drift.
    assert embedded.rstrip("\n") == source.rstrip("\n")


def test_manifest_matches_generator_output() -> None:
    expected = gen.render(AGENT_SCRIPT.read_text(encoding="utf-8"))
    assert MANIFEST.read_text(encoding="utf-8") == expected


def test_daemonset_targets_gpu_nodes_and_writes_tre_v2_redis() -> None:
    ds = _by_kind("DaemonSet")
    assert ds["metadata"]["name"] == "tre-v2-gpu-truth"
    assert ds["metadata"]["namespace"] == "tre-v2"
    spec = ds["spec"]["template"]["spec"]
    assert spec["hostPID"] is False
    assert spec["nodeSelector"] == {"nvidia.com/gpu.present": "true"}
    container = spec["containers"][0]
    assert container["image"] == "vllm/vllm-openai:0.10.1-sleep"
    command = container["command"]
    assert "/agent/gpu_truth_agent.py" in command
    assert "redis://tre-v2-redis:6379/0" in command
    assert "$(NODE_NAME)" in command
    env = {item["name"]: item for item in container["env"]}
    assert env["NVIDIA_VISIBLE_DEVICES"]["value"] == "all"
    assert env["NODE_NAME"]["valueFrom"]["fieldRef"]["fieldPath"] == "spec.nodeName"
    assert container["volumeMounts"][0]["mountPath"] == "/agent"

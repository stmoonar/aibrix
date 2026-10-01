from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import gen_gpu_truth_manifest as gen
from tre_common.registry import load_registry


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
TRE_ROOT = DEPLOY_ROOT.parent
MANIFEST = DEPLOY_ROOT / "overlays" / "tre-v2" / "gpu-truth.yaml"
REGISTRY = DEPLOY_ROOT / "registry.yaml"
DOCKERFILE = TRE_ROOT / "gpu-truth" / "Dockerfile"

#: The agent image this release runs (bump together with registry.yaml gpu_truth.image,
#: the bootstrap copy in overlays/tre-v2/params.yaml, and regenerate the manifest).
EXPECTED_IMAGE = "tre-v2-gpu-truth:20261001-e14151b1"


def _docs() -> list[dict]:
    return [doc for doc in yaml.safe_load_all(MANIFEST.read_text(encoding="utf-8")) if doc]


def _daemonset() -> dict:
    return next(doc for doc in _docs() if doc["kind"] == "DaemonSet")


def _registry_raw() -> dict:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))


def test_manifest_matches_generator_output() -> None:
    assert MANIFEST.read_text(encoding="utf-8") == gen.render_from_registry(REGISTRY)


def test_manifest_has_no_agent_configmap() -> None:
    # The agent is baked into its image; no ConfigMap copy that could drift.
    assert [doc["kind"] for doc in _docs()] == ["DaemonSet"]


def test_image_comes_from_the_registry_and_is_our_own_build() -> None:
    container = _daemonset()["spec"]["template"]["spec"]["containers"][0]
    assert _registry_raw()["gpu_truth"]["image"] == EXPECTED_IMAGE
    assert container["image"] == EXPECTED_IMAGE
    assert container["imagePullPolicy"] == "IfNotPresent"
    assert "vllm" not in container["image"] and "latest" not in container["image"]


def test_daemonset_targets_registry_nodes_and_writes_tre_v2_redis() -> None:
    ds = _daemonset()
    assert ds["metadata"]["name"] == "tre-v2-gpu-truth"
    assert ds["metadata"]["namespace"] == "tre-v2"
    assert ds["spec"]["updateStrategy"]["rollingUpdate"]["maxUnavailable"] == 1
    spec = ds["spec"]["template"]["spec"]
    assert spec["hostPID"] is False
    # only the heartbeat emptyDir (no agent ConfigMap)
    assert spec["volumes"] == [{"name": "heartbeat", "emptyDir": {"medium": "Memory", "sizeLimit": "1Mi"}}]
    terms = spec["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
    hostnames = terms[0]["matchExpressions"][0]
    assert hostnames["key"] == "kubernetes.io/hostname"
    assert hostnames["operator"] == "In"
    registry_nodes = [node.name for node in load_registry(str(REGISTRY))._topology.nodes]
    assert registry_nodes and hostnames["values"] == registry_nodes
    container = spec["containers"][0]
    command = container["command"]
    assert command[:2] == ["python3", gen.AGENT_IN_IMAGE]
    assert "redis://tre-v2-redis:6379/0" in command
    assert "$(NODE_NAME)" in command
    env = {item["name"]: item for item in container["env"]}
    assert env["NVIDIA_VISIBLE_DEVICES"]["value"] == "all"
    assert env["NVIDIA_DRIVER_CAPABILITIES"]["value"] == "utility"
    assert env["NODE_NAME"]["valueFrom"]["fieldRef"]["fieldPath"] == "spec.nodeName"
    assert container["volumeMounts"] == [{"name": "heartbeat", "mountPath": gen.HEARTBEAT_DIR}]
    security = container["securityContext"]
    assert security["runAsNonRoot"] is True and security["runAsUser"] != 0
    assert security["allowPrivilegeEscalation"] is False
    assert security["readOnlyRootFilesystem"] is True


def test_liveness_probe_checks_a_fresh_heartbeat_conservatively() -> None:
    container = _daemonset()["spec"]["template"]["spec"]["containers"][0]
    command = container["command"]
    args = dict(zip(command[2::2], command[3::2]))
    assert args["--heartbeat-file"] == gen.HEARTBEAT_FILE
    assert gen.HEARTBEAT_FILE.startswith(gen.HEARTBEAT_DIR + "/")
    assert args["--max-collect-failures"] == "6"
    probe = container["livenessProbe"]
    assert probe["exec"]["command"] == [
        "python3", gen.AGENT_IN_IMAGE, "--check-heartbeat", gen.HEARTBEAT_FILE, "--heartbeat-max-age-s", "60",
    ]
    # restart only after the heartbeat is stale for >= max_age + (threshold-1) x period
    assert probe["initialDelaySeconds"] >= 60 and probe["periodSeconds"] >= 10
    assert probe["failureThreshold"] >= 3 and probe["timeoutSeconds"] >= 5
    # the probe's limit is >= 6 samples, and the agent self-exits after 6 failed samples
    assert float(probe["exec"]["command"][-1]) >= 6 * float(args["--interval-s"])


def test_probe_and_agent_flags_exist_in_the_agent() -> None:
    from scripts import gpu_truth_agent

    assert gpu_truth_agent.DEFAULT_HEARTBEAT_MAX_AGE_S == gen.DEFAULT_HEARTBEAT_MAX_AGE_S
    assert gpu_truth_agent.DEFAULT_MAX_COLLECT_FAILURES == gen.DEFAULT_MAX_COLLECT_FAILURES
    with pytest.raises(ValueError):
        gen.render(image="img:1", nodes=["n1"], heartbeat_max_age_s=20)  # < 3 samples of 10 s


def test_daemonset_samples_every_10s_and_polls_refresh_requests() -> None:
    command = _daemonset()["spec"]["template"]["spec"]["containers"][0]["command"]
    args = dict(zip(command[2::2], command[3::2]))
    assert args["--interval-s"] == "10"
    assert args["--refresh-poll-s"] == "0.25"
    assert args["--ttl-s"] == "120"


def test_generator_renders_a_custom_interval() -> None:
    custom = list(
        yaml.safe_load_all(gen.render(image="img:1", nodes=["n1"], interval_s=5, refresh_poll_s=0.5, ttl_s=60))
    )
    ds = next(doc for doc in custom if doc and doc["kind"] == "DaemonSet")
    command = ds["spec"]["template"]["spec"]["containers"][0]["command"]
    assert command[command.index("--interval-s") + 1] == "5"
    assert command[command.index("--refresh-poll-s") + 1] == "0.5"
    assert command[command.index("--ttl-s") + 1] == "60"
    with pytest.raises(ValueError):
        gen.render(image="img:1", nodes=["n1"], interval_s=120, ttl_s=120)


def test_generator_requires_an_image_and_nodes() -> None:
    raw = _registry_raw()
    with pytest.raises(ValueError, match="gpu_truth.image"):
        gen.settings_from_registry({**raw, "gpu_truth": {}})
    with pytest.raises(ValueError, match="unknown keys"):
        gen.settings_from_registry({**raw, "gpu_truth": {"image": "x:1", "imagee": "y"}})
    with pytest.raises(ValueError, match="cluster.nodes"):
        gen.settings_from_registry({**raw, "cluster": {"nodes": []}})
    image, nodes = gen.settings_from_registry({**raw, "cluster": {"nodes": [{"name": "a"}, {"name": "b"}]}})
    assert image == EXPECTED_IMAGE and nodes == ["a", "b"]


def test_components_ignore_the_gpu_truth_section() -> None:
    """Old and new registry parsers read named top-level sections only, so the
    gpu_truth: section (read only by the generator) cannot stop a component."""
    raw = _registry_raw()
    assert "gpu_truth" in raw
    without = {key: value for key, value in raw.items() if key != "gpu_truth"}
    from tre_common.registry import _parse_registry

    assert _parse_registry(raw)._topology == _parse_registry(without)._topology


def test_dockerfile_bakes_the_agent_at_the_commanded_path() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert f"COPY deploy/scripts/gpu_truth_agent.py {gen.AGENT_IN_IMAGE}" in text
    assert "COPY gpu-truth/requirements.txt" in text
    assert "NVIDIA_DRIVER_CAPABILITIES=utility" in text
    assert "\nUSER 65532:65532\n" in text
    requirements = (TRE_ROOT / "gpu-truth" / "requirements.txt").read_text(encoding="utf-8")
    pins = [line for line in requirements.splitlines() if line and not line.startswith("#")]
    assert pins == ["redis==6.4.0"]

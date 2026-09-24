"""Guard tests for the opt-in reissue sidecar in the model manifests
(tre/docs/design/20260924-reissue-sidecar.md). Default = unchanged manifests."""
from __future__ import annotations

import copy
import textwrap
from pathlib import Path

import pytest
import yaml

import gen_model_manifests as gen
from tre_common.registry import DEFAULT_REISSUE_GATEWAY_URL, load_registry

DEPLOY_DIR = Path(__file__).resolve().parents[1]
REAL_REGISTRY = DEPLOY_DIR / "registry.yaml"

BASE = """
cluster:
  nodes:
    - {name: node-a, gpus: 2, gpu_uuids: [GPU-a0, GPU-a1], two_gpu_slots: [[0, 1]]}
models:
  - name: m1
    weights_path: /models/m1
    tp_size: 1
    min_replicas: 1
    max_replicas: 2
    vllm_image: vllm/vllm-openai:0.10.1-sleep
    slo: {ttft_p95_ms: 1, tpot_p95_ms: 1, e2e_p95_ms: 1}
    trs: {w_p: 0.04, w_d: 1.0, lambda_wait: 2.625, qmin: 1.0, ema_alpha: 0.5, theta_m: 0.0, tau_crit: 0.8, tau_low: 1.0, tau_high: 1.25, qsat: 4.0, epsat: 0.05, hsat: 3}
"""


def _registry(tmp_path, extra: str = ""):
    path = tmp_path / "registry.yaml"
    path.write_text(textwrap.dedent(BASE) + textwrap.dedent(extra), encoding="utf-8")
    return load_registry(str(path))


def _by_kind(resources, kind):
    return [item for item in resources if item["kind"] == kind]


def test_live_registry_does_not_enable_the_sidecar():
    registry = load_registry(str(REAL_REGISTRY))
    assert registry.reissue_sidecar.enabled is False
    resources = gen.build_resources(registry)
    assert _by_kind(resources, "ConfigMap") == []
    for deployment in _by_kind(resources, "Deployment"):
        pod = deployment["spec"]["template"]["spec"]
        assert [c["name"] for c in pod["containers"]] == ["vllm-openai"]
        command = pod["containers"][0]["command"]
        assert command[command.index("--port") + 1] == "8000"
        assert command[command.index("--host") + 1] == "0.0.0.0"
        assert pod["containers"][0]["readinessProbe"]["httpGet"]["port"] == 8000


def test_disabled_key_renders_byte_identical_to_absent_key(tmp_path):
    absent = gen.build_resources(_registry(tmp_path))
    disabled = gen.build_resources(
        _registry(tmp_path, "reissue_sidecar: {enabled: false, gateway_url: 'http://x:1', vllm_port: 9000}\n")
    )
    assert yaml.safe_dump(absent, sort_keys=False) == yaml.safe_dump(disabled, sort_keys=False)


def test_registry_flag_adds_sidecar_configmap_and_moves_vllm(tmp_path):
    registry = _registry(tmp_path, "reissue_sidecar: {enabled: true, gateway_url: 'http://gw.example:80/'}\n")
    assert registry.validate() == []
    resources = gen.build_resources(registry)
    (configmap,) = _by_kind(resources, "ConfigMap")
    assert configmap["metadata"] == {
        "name": "tre-reissue-sidecar",
        "namespace": "default",
        "labels": {"tre.aibrix.io/managed": "true", "app.kubernetes.io/name": "tre-reissue-sidecar"},
    }
    assert configmap["data"]["sidecar.py"] == gen.REISSUE_SCRIPT_PATH.read_text(encoding="utf-8")

    (service,) = _by_kind(resources, "Service")
    assert service["spec"]["ports"][0]["targetPort"] == 8000

    deployments = _by_kind(resources, "Deployment")
    assert len(deployments) == 2
    for deployment in deployments:
        template = deployment["spec"]["template"]
        assert template["metadata"]["labels"]["model.aibrix.ai/port"] == "8000"
        pod = template["spec"]
        vllm, sidecar = pod["containers"]
        command = vllm["command"]
        assert command[command.index("--port") + 1] == "8001"
        assert command[command.index("--host") + 1] == "127.0.0.1"
        assert "readinessProbe" not in vllm and "ports" not in vllm
        assert sidecar["name"] == "tre-reissue-sidecar"
        assert sidecar["image"] == "vllm/vllm-openai:0.10.1-sleep"
        assert sidecar["command"] == ["python3", "/opt/tre-reissue/sidecar.py"]
        assert sidecar["ports"] == [{"containerPort": 8000, "protocol": "TCP"}]
        assert sidecar["readinessProbe"]["httpGet"] == {"path": "/health", "port": 8000}
        assert sidecar["readinessProbe"]["failureThreshold"] == 300
        assert sidecar["resources"]["limits"]["cpu"] == "250m"
        env = {item["name"]: item.get("value") for item in sidecar["env"]}
        assert env["TRE_REISSUE_GATEWAY_URL"] == "http://gw.example:80"
        assert env["TRE_REISSUE_UPSTREAM"] == "http://127.0.0.1:8001"
        assert env["TRE_REISSUE_LISTEN_PORT"] == "8000"
        assert env["TRE_REISSUE_MODEL"] == "m1"
        assert env["TRE_REISSUE_MAX_DEPTH"] == "3"
        assert env["NVIDIA_VISIBLE_DEVICES"] == "void"
        assert "CUDA_VISIBLE_DEVICES" not in env
        assert {"name": "tre-reissue-sidecar", "configMap": {"name": "tre-reissue-sidecar", "defaultMode": 0o444}} in pod["volumes"]


def test_service_manager_runtime_create_matches_rendered_deployment(tmp_path):
    registry = _registry(tmp_path, "reissue_sidecar: {enabled: true}\n")
    rendered = {d["metadata"]["name"]: d for d in gen.build_deployments(registry)}
    created = gen.build_model_deployment(registry, "m1", "node-a", (1,))
    assert created == rendered[created["metadata"]["name"]]
    assert len(created["spec"]["template"]["spec"]["containers"]) == 2
    env = {e["name"]: e.get("value") for e in created["spec"]["template"]["spec"]["containers"][1]["env"]}
    assert env["TRE_REISSUE_GATEWAY_URL"] == DEFAULT_REISSUE_GATEWAY_URL


def test_cli_flag_enables_sidecar_without_registry_change(tmp_path):
    out = tmp_path / "models"
    gen.main(["--registry", str(REAL_REGISTRY), "--output-dir", str(out), "--reissue-sidecar",
              "--reissue-gateway-url", "http://gw.test:8080/"])
    kustomization = yaml.safe_load((out / "kustomization.yaml").read_text(encoding="utf-8"))
    assert "tre-reissue-sidecar.yaml" in kustomization["resources"]
    deployment = yaml.safe_load((out / "dsqwen-7b-nscc-ds-4a100-node9-gpu-0.yaml").read_text(encoding="utf-8"))
    sidecar = deployment["spec"]["template"]["spec"]["containers"][1]
    env = {e["name"]: e.get("value") for e in sidecar["env"]}
    assert env["TRE_REISSUE_GATEWAY_URL"] == "http://gw.test:8080"
    # without the flag the same call renders no sidecar
    plain = tmp_path / "plain"
    gen.main(["--registry", str(REAL_REGISTRY), "--output-dir", str(plain)])
    assert not (plain / "tre-reissue-sidecar.yaml").exists()
    plain_deployment = yaml.safe_load((plain / "dsqwen-7b-nscc-ds-4a100-node9-gpu-0.yaml").read_text(encoding="utf-8"))
    assert len(plain_deployment["spec"]["template"]["spec"]["containers"]) == 1


def test_registry_rejects_bad_reissue_settings(tmp_path):
    registry = _registry(tmp_path, "reissue_sidecar: {enabled: true, vllm_port: 8000, chat_mode: nope}\n")
    errors = registry.validate()
    assert any("vllm_port" in error for error in errors)
    assert any("chat_mode" in error for error in errors)
    with pytest.raises(ValueError):
        _registry(tmp_path, "reissue_sidecar: {enabled: true, typo_key: 1}\n")


def test_disabled_deployment_is_not_mutated_by_sidecar_helper(tmp_path):
    registry = _registry(tmp_path, "reissue_sidecar: {enabled: true}\n")
    plain = gen.build_deployments(registry, reissue=gen._DISABLED)
    snapshot = copy.deepcopy(plain)
    gen.build_deployments(registry)
    assert plain == snapshot
    assert all(len(d["spec"]["template"]["spec"]["containers"]) == 1 for d in plain)

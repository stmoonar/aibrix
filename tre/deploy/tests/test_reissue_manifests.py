"""Manifests of the retry / continuation sidecar (registry ``reissue:``, models[].vllm_features)."""
from __future__ import annotations

import textwrap
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from gen_model_manifests import (
    REISSUE_CONTAINER,
    REISSUE_SCRIPT_PATH,
    build_deployments,
    build_model_deployment,
    build_resources,
    build_services,
)
from tre_common.registry import Registry, ReissueConfig, load_registry

DEPLOY_ROOT = Path(__file__).resolve().parents[1]

REGISTRY = """
cluster:
  nodes:
    - {name: node-a, gpus: 2, gpu_uuids: [GPU-A-0, GPU-A-1], two_gpu_slots: [[0, 1]]}
models:
  - name: m-fork
    weights_path: /models/m
    tp_size: 1
    min_replicas: 0
    max_replicas: 1
    vllm_image: image:fork
    vllm_features: [sleep_reject_new, abort_return_token_ids]
    slo: {ttft_p95_ms: 1, tpot_p95_ms: 1, e2e_p95_ms: 1}
    trs: {w_p: 0.04, w_d: 1.0, lambda_wait: 2.625, qmin: 1.0, ema_alpha: 0.5, theta_m: 0.0, tau_crit: 0.8, tau_low: 1.0, tau_high: 1.25, qsat: 4.0, epsat: 0.05, hsat: 3}
  - name: m-old
    weights_path: /models/o
    tp_size: 1
    min_replicas: 0
    max_replicas: 1
    vllm_image: image:old
    slo: {ttft_p95_ms: 1, tpot_p95_ms: 1, e2e_p95_ms: 1}
    trs: {w_p: 0.04, w_d: 1.0, lambda_wait: 2.625, qmin: 1.0, ema_alpha: 0.5, theta_m: 0.0, tau_crit: 0.8, tau_low: 1.0, tau_high: 1.25, qsat: 4.0, epsat: 0.05, hsat: 3}
"""


def _registry(tmp_path: Path, extra: str = "") -> Registry:
    path = tmp_path / "registry.yaml"
    path.write_text(textwrap.dedent(REGISTRY) + textwrap.dedent(extra), encoding="utf-8")
    registry = load_registry(str(path))
    assert registry.validate() == []
    return registry


def _containers(deployment: dict) -> dict[str, dict]:
    return {c["name"]: c for c in deployment["spec"]["template"]["spec"]["containers"]}


def _env(container: dict) -> dict[str, str]:
    return {e["name"]: e.get("value", e.get("valueFrom")) for e in container["env"]}


def test_sidecar_is_on_by_default_and_owns_the_serving_port(tmp_path):
    registry = _registry(tmp_path)
    assert registry.reissue().enabled is True
    fork, old = build_deployments(registry)
    containers = _containers(fork)
    vllm, sidecar = containers["vllm-openai"], containers[REISSUE_CONTAINER]
    command = vllm["command"]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8001"
    assert "readinessProbe" not in vllm and "ports" not in vllm
    assert sidecar["ports"] == [{"containerPort": 8000, "protocol": "TCP"}]
    assert sidecar["readinessProbe"]["httpGet"] == {"path": "/health", "port": 8000}
    assert sidecar["image"] == "image:fork"  # the vLLM image ships python3 + aiohttp
    env = _env(sidecar)
    assert env["TRE_REISSUE_LISTEN_PORT"] == "8000"
    assert env["TRE_REISSUE_UPSTREAM_URL"] == "http://127.0.0.1:8001"
    # the stable gateway Service (registry gateway: section), in-cluster DNS, no IP
    assert env["TRE_GATEWAY_URL"] == "http://tre-gateway.envoy-gateway-system.svc.cluster.local:80"
    assert env["TRE_REISSUE_REQUIRE_HIDDEN_HEADER"] == "true"
    # loopback keep-alive: the sidecar's pool (2 s) below vLLM's (default 5 s when unset)
    assert env["TRE_REISSUE_UPSTREAM_KEEPALIVE_S"] == "2"
    assert env["TRE_REISSUE_UPSTREAM_SERVER_KEEPALIVE_S"] == "5"
    assert env["TRE_REISSUE_LOCAL_RECONNECT_ATTEMPTS"] == "1"
    assert env["POD_NAME"] == {"fieldRef": {"fieldPath": "metadata.name"}}
    assert env["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert sidecar["command"] == ["python3", "/opt/tre-reissue/sidecar.py"]
    volumes = {v["name"]: v for v in fork["spec"]["template"]["spec"]["volumes"]}
    assert volumes[REISSUE_CONTAINER]["configMap"]["name"] == "tre-reissue-sidecar"
    # labels, Service and probes still target the serving port
    assert fork["spec"]["template"]["metadata"]["labels"]["model.aibrix.ai/port"] == "8000"
    for service in build_services(registry):
        assert service["spec"]["ports"][0]["targetPort"] == 8000
    # fork flags only where the image declares the features
    assert "--sleep-reject-new" in command and "--abort-return-token-ids" in command
    old_command = _containers(old)["vllm-openai"]["command"]
    assert "--sleep-reject-new" not in old_command and "--abort-return-token-ids" not in old_command
    assert REISSUE_CONTAINER in _containers(old)


def test_disabled_renders_the_plain_pod(tmp_path):
    # without the sidecar vLLM itself answers on the pod port: its keep-alive must outlast
    # Envoy's upstream idle timeout
    registry = _registry(tmp_path, "reissue: {enabled: false}\nvllm: {env: {VLLM_HTTP_TIMEOUT_KEEP_ALIVE: '75'}}\n")
    for deployment in build_deployments(registry):
        containers = _containers(deployment)
        assert list(containers) == ["vllm-openai"]
        vllm = containers["vllm-openai"]
        command = vllm["command"]
        assert command[command.index("--host") + 1] == "0.0.0.0"
        assert command[command.index("--port") + 1] == "8000"
        assert vllm["readinessProbe"]["httpGet"] == {"path": "/health", "port": 8000}
        assert "--sleep-reject-new" not in command and "--abort-return-token-ids" not in command
        assert all(v["name"] != REISSUE_CONTAINER for v in deployment["spec"]["template"]["spec"]["volumes"])
    assert all(r["kind"] != "ConfigMap" for r in build_resources(registry))
    # identical to a registry without the section but with the sidecar switched off
    plain = _registry(tmp_path, "vllm: {env: {VLLM_HTTP_TIMEOUT_KEEP_ALIVE: '75'}}")
    plain = Registry(plain.topology(), plain.models(), plain.service_manager(), plain.gateway(),
                     ReissueConfig(enabled=False), plain.vllm())
    assert build_deployments(plain) == build_deployments(registry)


def test_configmap_ships_the_sidecar_script(tmp_path):
    registry = _registry(tmp_path, "reissue: {configmap: my-cm, namespace: models, cpu_limit: '1'}\n")
    (cm,) = [r for r in build_resources(registry) if r["kind"] == "ConfigMap"]
    assert cm["metadata"] == {"name": "my-cm", "namespace": "models",
                              "labels": {"tre.aibrix.io/managed": "true", "app.kubernetes.io/name": REISSUE_CONTAINER}}
    assert cm["data"]["sidecar.py"] == REISSUE_SCRIPT_PATH.read_text(encoding="utf-8")
    sidecar = _containers(build_deployments(registry)[0])[REISSUE_CONTAINER]
    assert sidecar["resources"]["limits"]["cpu"] == "1"


def test_registry_overrides_reach_the_sidecar(tmp_path):
    registry = _registry(tmp_path, textwrap.dedent("""
        reissue:
          gateway_url: http://gw.other-ns.svc.cluster.local:8080/
          vllm_port: 9001
          max_depth: 2
          retry_attempts: 6
          image: sidecar:1
          extra_env: {TRE_REISSUE_GENERATED_IDS_FIELD: gen_ids}
    """))
    deployment = build_deployments(registry)[0]
    containers = _containers(deployment)
    env = _env(containers[REISSUE_CONTAINER])
    assert env["TRE_GATEWAY_URL"] == "http://gw.other-ns.svc.cluster.local:8080"
    assert env["TRE_REISSUE_UPSTREAM_URL"] == "http://127.0.0.1:9001"
    assert env["TRE_REISSUE_MAX_DEPTH"] == "2" and env["TRE_REISSUE_RETRY_ATTEMPTS"] == "6"
    assert env["TRE_REISSUE_GENERATED_IDS_FIELD"] == "gen_ids"
    assert containers[REISSUE_CONTAINER]["image"] == "sidecar:1"
    command = containers["vllm-openai"]["command"]
    assert command[command.index("--port") + 1] == "9001"


def test_keepalive_one_second_below_the_server_is_accepted(tmp_path):
    _registry(tmp_path, "reissue: {upstream_keepalive_s: 4}\n")  # vLLM default 5 s; validates clean


def test_keepalive_settings_reach_the_sidecar(tmp_path):
    registry = _registry(tmp_path, textwrap.dedent("""
        vllm:
          env: {VLLM_HTTP_TIMEOUT_KEEP_ALIVE: '75'}
        reissue:
          upstream_keepalive_s: 1.5
          local_reconnect_attempts: 2
    """))
    for deployment in build_deployments(registry):
        containers = _containers(deployment)
        assert _env(containers["vllm-openai"])["VLLM_HTTP_TIMEOUT_KEEP_ALIVE"] == "75"
        env = _env(containers[REISSUE_CONTAINER])
        assert env["TRE_REISSUE_UPSTREAM_KEEPALIVE_S"] == "1.5"
        assert env["TRE_REISSUE_UPSTREAM_SERVER_KEEPALIVE_S"] == "75"
        assert env["TRE_REISSUE_LOCAL_RECONNECT_ATTEMPTS"] == "2"


def test_envoy_to_sidecar_keepalive_settings_reach_the_sidecar(tmp_path):
    registry = _registry(tmp_path, textwrap.dedent("""
        gateway: {upstream_idle_timeout_s: 30}
        reissue: {server_keepalive_s: 120, local_reconnect_window_s: 0.5}
    """))
    for deployment in build_deployments(registry):
        env = _env(_containers(deployment)[REISSUE_CONTAINER])
        assert env["TRE_REISSUE_SERVER_KEEPALIVE_S"] == "120"
        assert env["TRE_REISSUE_GATEWAY_UPSTREAM_IDLE_S"] == "30"
        assert env["TRE_REISSUE_LOCAL_RECONNECT_WINDOW_S"] == "0.5"
    # defaults: sidecar 75 s above Envoy's 60 s
    for deployment in build_deployments(_registry(tmp_path)):
        env = _env(_containers(deployment)[REISSUE_CONTAINER])
        assert (env["TRE_REISSUE_SERVER_KEEPALIVE_S"], env["TRE_REISSUE_GATEWAY_UPSTREAM_IDLE_S"],
                env["TRE_REISSUE_LOCAL_RECONNECT_WINDOW_S"]) == ("75", "60", "1")


@pytest.mark.parametrize(
    "extra,needle",
    [
        ("reissue: {upstream_keepalive_s: 0}\n", "upstream_keepalive_s must be > 0"),
        ("reissue: {server_keepalive_s: 0}\n", "server_keepalive_s must be > 0"),
        ("reissue: {local_reconnect_window_s: -1}\n", "local_reconnect_window_s must be >= 0"),
        # Envoy's idle timeout must be >= 1 s below the sidecar's server keep-alive
        ("reissue: {server_keepalive_s: 60}\n", "at least 1 s above gateway.upstream_idle_timeout_s"),
        ("reissue: {server_keepalive_s: 60.5}\n", "at least 1 s above gateway.upstream_idle_timeout_s"),
        ("gateway: {upstream_idle_timeout_s: 75}\n", "at least 1 s above gateway.upstream_idle_timeout_s"),
        ("gateway: {upstream_idle_timeout_s: 0}\n", "upstream_idle_timeout_s must be positive"),
        # without the sidecar the pod's server is vLLM itself (default keep-alive 5 s here)
        ("reissue: {enabled: false}\n", "vLLM's VLLM_HTTP_TIMEOUT_KEEP_ALIVE (5) must be at least 1 s above"),
        ("reissue: {local_reconnect_attempts: -1}\n", "local_reconnect_attempts"),
        # the pool must be BELOW vLLM's keep-alive (default 5 s) of every model
        ("reissue: {upstream_keepalive_s: 5}\n", "at least 1 s below vLLM's VLLM_HTTP_TIMEOUT_KEEP_ALIVE"),
        ("reissue: {upstream_keepalive_s: 4.5}\n", "at least 1 s below vLLM's VLLM_HTTP_TIMEOUT_KEEP_ALIVE"),
        ("vllm: {env: {VLLM_HTTP_TIMEOUT_KEEP_ALIVE: '2'}}\n", "at least 1 s below vLLM's VLLM_HTTP_TIMEOUT_KEEP_ALIVE"),
        ("vllm: {env: {VLLM_HTTP_TIMEOUT_KEEP_ALIVE: '7.5'}}\n", "integer number of seconds"),
        ("reissue: {vllm_port: 8000}\n", "vllm_port"),
        ("reissue: {gateway_url: '10.0.0.1:80'}\n", "gateway_url"),
        ("reissue: {max_depth: -1}\n", "max_depth"),
        ("reissue: {retry_attempts: 0}\n", "retry_attempts"),
        ("reissue: {extra_env: {PATH: /x}}\n", "extra_env"),
    ],
)
def test_invalid_reissue_settings_are_rejected(tmp_path, extra, needle):
    path = tmp_path / "registry.yaml"
    path.write_text(textwrap.dedent(REGISTRY) + extra, encoding="utf-8")
    assert any(needle in error for error in load_registry(str(path)).validate())


def test_unknown_keys_and_features_are_rejected(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(textwrap.dedent(REGISTRY) + "reissue: {chat_mode: render}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_registry(str(path))
    path.write_text(textwrap.dedent(REGISTRY).replace("abort_return_token_ids]", "made_up]"), encoding="utf-8")
    assert any("unknown vllm_features" in e for e in load_registry(str(path)).validate())


def test_service_manager_creates_match_rendered_deployments(tmp_path):
    registry = _registry(tmp_path)
    for rendered in build_deployments(registry):
        labels = rendered["metadata"]["labels"]
        gpu_ids = tuple(int(g) for g in labels["tre.aibrix.io/gpu-ids"].split("-"))
        created = build_model_deployment(registry, labels["model.aibrix.ai/name"], labels["tre.aibrix.io/node"],
                                         gpu_ids)
        assert created == rendered


def test_repo_registry_and_committed_manifests_carry_the_sidecar():
    registry = load_registry(str(DEPLOY_ROOT / "registry.yaml"))
    spec = registry.reissue()
    assert spec.enabled
    configmap = yaml.safe_load((DEPLOY_ROOT / "models" / f"{spec.configmap}.yaml").read_text(encoding="utf-8"))
    assert configmap["data"]["sidecar.py"] == REISSUE_SCRIPT_PATH.read_text(encoding="utf-8"), (
        "deploy/models is stale: run make manifests"
    )
    checked = 0
    for path in sorted((DEPLOY_ROOT / "models").glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if doc.get("kind") != "Deployment":
            continue
        assert REISSUE_CONTAINER in _containers(doc), path.name
        # I4: room for the in-flight bound (Envoy max_requests 4096 per pod)
        assert _containers(doc)[REISSUE_CONTAINER]["resources"]["limits"] == {"cpu": "2", "memory": "1Gi"}, path.name
        checked += 1
    assert checked == len(build_deployments(registry))
    # the tre-v2-registry ConfigMap (read by the service-manager for runtime creates)
    # renders the same sidecar
    params = next(d for d in yaml.safe_load_all((DEPLOY_ROOT / "overlays" / "tre-v2" / "params.yaml").read_text(
        encoding="utf-8")) if d and d.get("kind") == "ConfigMap")
    from tre_common.registry import _parse_registry

    live = _parse_registry(yaml.safe_load(params["data"]["registry.yaml"]))
    assert live.reissue() == replace(spec)
    for model in registry.models():
        assert live.vllm_env_for(live.model(model.name)) == registry.vllm_env_for(model)
    # loopback keep-alive (2026-09-30 smoke 502s): vLLM keeps idle connections 75 s, the
    # sidecar pools them 2 s
    for deployment in build_deployments(registry):
        containers = _containers(deployment)
        assert _env(containers["vllm-openai"])["VLLM_HTTP_TIMEOUT_KEEP_ALIVE"] == "75"
        env = _env(containers[REISSUE_CONTAINER])
        assert env["TRE_REISSUE_UPSTREAM_SERVER_KEEPALIVE_S"] == "75"
        assert env["TRE_REISSUE_UPSTREAM_KEEPALIVE_S"] == "2"


def test_gateway_url_follows_the_gateway_service_settings(tmp_path):
    registry = _registry(tmp_path, "gateway: {service_name: gw, service_namespace: proxies, service_port: 8080}\n")
    env = _env(_containers(build_deployments(registry)[0])[REISSUE_CONTAINER])
    assert env["TRE_GATEWAY_URL"] == "http://gw.proxies.svc.cluster.local:8080"
    assert registry.reissue().gateway_url is None  # derived, not stored


#: ``reissue:`` keys known to controller / SM / UI 20260930-f8ccb0ca (its ReissueConfig fields):
#: that parser raises on any other key, so the repo registry (which is what gets merged into
#: the live one) may not carry a newer key until all three images are upgraded.
F8CCB0CA_REISSUE_KEYS = {
    "enabled", "gateway_url", "vllm_port", "max_depth", "retry_attempts", "image", "configmap",
    "namespace", "cpu_request", "cpu_limit", "memory_request", "memory_limit", "extra_env",
}


@pytest.mark.parametrize(
    "path",
    [DEPLOY_ROOT / "registry.yaml", DEPLOY_ROOT / "overlays" / "tre-v2" / "params.yaml"],
    ids=["registry.yaml", "params.yaml"],
)
def test_repo_registry_reissue_section_is_accepted_by_the_deployed_parser(path):
    text = path.read_text(encoding="utf-8")
    if path.name == "params.yaml":
        text = yaml.safe_load(text)["data"]["registry.yaml"]
    keys = set(yaml.safe_load(text)["reissue"])
    assert keys <= F8CCB0CA_REISSUE_KEYS, (
        f"{sorted(keys - F8CCB0CA_REISSUE_KEYS)} would crash the deployed controller / SM / UI "
        "(unknown reissue keys raise); keep them commented out until all three are upgraded"
    )

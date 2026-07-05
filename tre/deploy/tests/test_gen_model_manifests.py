import textwrap

import yaml

from gen_model_manifests import build_deployments, build_httproutes, build_resources, build_services, write_manifests
from tre_common.registry import load_registry


def test_build_deployments_creates_one_deployment_per_feasible_slot(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        textwrap.dedent(
            """
            cluster:
              nodes:
                - {name: node-75, gpus: 4, gpu_uuids: [GPU-75-0, GPU-75-1, GPU-75-2, GPU-75-3], two_gpu_slots: [[0, 1], [2, 3]]}
                - {name: node-76, gpus: 4, gpu_uuids: [GPU-76-0, GPU-76-1, GPU-76-2, GPU-76-3], two_gpu_slots: [[0, 1], [2, 3]]}
            models:
              - name: one-gpu
                weights_path: /models/one
                tp_size: 1
                min_replicas: 1
                max_replicas: 8
                vllm_image: image:one
                slo: {ttft_p95_ms: 1, tpot_p95_ms: 1, e2e_p95_ms: 1}
                trs: {w_p: 0.04, w_d: 1.0, lambda_wait: 2.625, qmin: 1.0, ema_alpha: 0.5, theta_m: 0.0, tau_crit: 0.8, tau_low: 1.0, tau_high: 1.25, qsat: 4.0, epsat: 0.05, hsat: 3}
              - name: two-gpu
                weights_path: /models/two
                tp_size: 2
                min_replicas: 0
                max_replicas: 4
                vllm_image: image:two
                slo: {ttft_p95_ms: 1, tpot_p95_ms: 1, e2e_p95_ms: 1}
                trs: {w_p: 0.04, w_d: 1.0, lambda_wait: 2.625, qmin: 1.0, ema_alpha: 0.5, theta_m: 0.0, tau_crit: 0.8, tau_low: 1.0, tau_high: 1.25, qsat: 4.0, epsat: 0.05, hsat: 3}
            """
        ),
        encoding="utf-8",
    )
    registry = load_registry(str(path))

    deployments = build_deployments(registry)

    assert len(deployments) == 12
    names = [item["metadata"]["name"] for item in deployments]
    assert "one-gpu-node-75-gpu-0" in names
    assert "one-gpu-node-76-gpu-3" in names
    assert "two-gpu-node-75-gpu-0-1" in names
    assert "two-gpu-node-76-gpu-2-3" in names


def test_build_deployments_encodes_node_gpu_binding_and_vllm_args(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        textwrap.dedent(
            """
            cluster:
              nodes:
                - {name: node-75, gpus: 2, gpu_uuids: [GPU-75-0, GPU-75-1], two_gpu_slots: [[0, 1]]}
            models:
              - name: two-gpu
                weights_path: /models/two
                tp_size: 2
                min_replicas: 0
                max_replicas: 1
                vllm_image: image:two
                slo: {ttft_p95_ms: 1, tpot_p95_ms: 1, e2e_p95_ms: 1}
                trs: {w_p: 0.04, w_d: 1.0, lambda_wait: 2.625, qmin: 1.0, ema_alpha: 0.5, theta_m: 0.0, tau_crit: 0.8, tau_low: 1.0, tau_high: 1.25, qsat: 4.0, epsat: 0.05, hsat: 3}
            """
        ),
        encoding="utf-8",
    )

    [deployment] = build_deployments(load_registry(str(path)))
    rendered = yaml.safe_load(yaml.safe_dump(deployment))
    container = rendered["spec"]["template"]["spec"]["containers"][0]

    assert rendered["spec"]["template"]["spec"]["nodeName"] == "node-75"
    assert rendered["spec"]["template"]["metadata"]["labels"]["tre.aibrix.io/routable"] == "true"
    assert rendered["metadata"]["annotations"]["tre.aibrix.io/gpu-ids"] == "0,1"
    assert rendered["spec"]["template"]["metadata"]["annotations"]["tre.aibrix.io/gpu-ids"] == "0,1"
    assert rendered["metadata"]["annotations"]["tre.aibrix.io/gpu-uuids"] == "GPU-75-0,GPU-75-1"
    assert rendered["spec"]["template"]["metadata"]["annotations"]["tre.aibrix.io/gpu-uuids"] == "GPU-75-0,GPU-75-1"
    assert rendered["metadata"]["labels"]["tre.aibrix.io/gpu-ids"] == "0-1"
    assert rendered["spec"]["template"]["metadata"]["labels"]["tre.aibrix.io/gpu-ids"] == "0-1"
    assert {"name": "NVIDIA_VISIBLE_DEVICES", "value": "GPU-75-0,GPU-75-1"} in container["env"]
    assert "nvidia.com/gpu" not in yaml.safe_dump(container.get("resources", {}))
    assert "--tensor-parallel-size" in container["command"]
    assert "--enable_sleep_mode" in container["command"]


def test_cuda_visible_devices_uses_container_local_ordinals(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        textwrap.dedent(
            """
            cluster:
              nodes:
                - {name: node-75, gpus: 4, gpu_uuids: [GPU-75-0, GPU-75-1, GPU-75-2, GPU-75-3], two_gpu_slots: [[0, 1], [2, 3]]}
            models:
              - name: one-gpu
                weights_path: /models/one
                tp_size: 1
                min_replicas: 0
                max_replicas: 4
                vllm_image: image:one
                slo: {ttft_p95_ms: 1, tpot_p95_ms: 1, e2e_p95_ms: 1}
                trs: {w_p: 0.04, w_d: 1.0, lambda_wait: 2.625, qmin: 1.0, ema_alpha: 0.5, theta_m: 0.0, tau_crit: 0.8, tau_low: 1.0, tau_high: 1.25, qsat: 4.0, epsat: 0.05, hsat: 3}
              - name: two-gpu
                weights_path: /models/two
                tp_size: 2
                min_replicas: 0
                max_replicas: 2
                vllm_image: image:two
                slo: {ttft_p95_ms: 1, tpot_p95_ms: 1, e2e_p95_ms: 1}
                trs: {w_p: 0.04, w_d: 1.0, lambda_wait: 2.625, qmin: 1.0, ema_alpha: 0.5, theta_m: 0.0, tau_crit: 0.8, tau_low: 1.0, tau_high: 1.25, qsat: 4.0, epsat: 0.05, hsat: 3}
            """
        ),
        encoding="utf-8",
    )

    deployments = {item["metadata"]["name"]: item for item in build_deployments(load_registry(str(path)))}

    one_gpu = deployments["one-gpu-node-75-gpu-2"]["spec"]["template"]["spec"]["containers"][0]
    two_gpu = deployments["two-gpu-node-75-gpu-2-3"]["spec"]["template"]["spec"]["containers"][0]
    assert {"name": "NVIDIA_VISIBLE_DEVICES", "value": "GPU-75-2"} in one_gpu["env"]
    assert {"name": "NVIDIA_VISIBLE_DEVICES", "value": "GPU-75-2,GPU-75-3"} in two_gpu["env"]
    assert deployments["one-gpu-node-75-gpu-2"]["metadata"]["labels"]["tre.aibrix.io/gpu-ids"] == "2"
    assert deployments["one-gpu-node-75-gpu-2"]["metadata"]["annotations"]["tre.aibrix.io/gpu-ids"] == "2"
    assert deployments["two-gpu-node-75-gpu-2-3"]["metadata"]["labels"]["tre.aibrix.io/gpu-ids"] == "2-3"
    assert deployments["two-gpu-node-75-gpu-2-3"]["metadata"]["annotations"]["tre.aibrix.io/gpu-ids"] == "2,3"
    assert deployments["one-gpu-node-75-gpu-2"]["metadata"]["annotations"]["tre.aibrix.io/gpu-uuids"] == "GPU-75-2"
    assert deployments["two-gpu-node-75-gpu-2-3"]["metadata"]["annotations"]["tre.aibrix.io/gpu-uuids"] == "GPU-75-2,GPU-75-3"


def test_build_deployments_rejects_gpu_bound_budget_over_three(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        textwrap.dedent(
            """
            cluster:
              nodes:
                - {name: node-75, gpus: 1, gpu_uuids: [GPU-75-0], two_gpu_slots: []}
            models:
              - name: m1
                weights_path: /models/one
                tp_size: 1
                min_replicas: 0
                max_replicas: 1
                vllm_image: image:one
                slo: {ttft_p95_ms: 1, tpot_p95_ms: 1, e2e_p95_ms: 1}
                trs: {w_p: 0.04, w_d: 1.0, lambda_wait: 2.625, qmin: 1.0, ema_alpha: 0.5, theta_m: 0.0, tau_crit: 0.8, tau_low: 1.0, tau_high: 1.25, qsat: 4.0, epsat: 0.05, hsat: 3}
              - name: m2
                weights_path: /models/two
                tp_size: 1
                min_replicas: 0
                max_replicas: 1
                vllm_image: image:two
                slo: {ttft_p95_ms: 1, tpot_p95_ms: 1, e2e_p95_ms: 1}
                trs: {w_p: 0.04, w_d: 1.0, lambda_wait: 2.625, qmin: 1.0, ema_alpha: 0.5, theta_m: 0.0, tau_crit: 0.8, tau_low: 1.0, tau_high: 1.25, qsat: 4.0, epsat: 0.05, hsat: 3}
              - name: m3
                weights_path: /models/three
                tp_size: 1
                min_replicas: 0
                max_replicas: 1
                vllm_image: image:three
                slo: {ttft_p95_ms: 1, tpot_p95_ms: 1, e2e_p95_ms: 1}
                trs: {w_p: 0.04, w_d: 1.0, lambda_wait: 2.625, qmin: 1.0, ema_alpha: 0.5, theta_m: 0.0, tau_crit: 0.8, tau_low: 1.0, tau_high: 1.25, qsat: 4.0, epsat: 0.05, hsat: 3}
              - name: m4
                weights_path: /models/four
                tp_size: 1
                min_replicas: 0
                max_replicas: 1
                vllm_image: image:four
                slo: {ttft_p95_ms: 1, tpot_p95_ms: 1, e2e_p95_ms: 1}
                trs: {w_p: 0.04, w_d: 1.0, lambda_wait: 2.625, qmin: 1.0, ema_alpha: 0.5, theta_m: 0.0, tau_crit: 0.8, tau_low: 1.0, tau_high: 1.25, qsat: 4.0, epsat: 0.05, hsat: 3}
            """
        ),
        encoding="utf-8",
    )

    try:
        build_deployments(load_registry(str(path)))
    except ValueError as exc:
        assert "bound budget" in str(exc)
    else:
        raise AssertionError("expected GPU bound budget violation")


def test_build_services_creates_one_model_service_with_gateway_selector(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        textwrap.dedent(
            """
            cluster:
              nodes:
                - {name: node-75, gpus: 1, gpu_uuids: [GPU-75-0], two_gpu_slots: []}
            models:
              - name: dsqwen-7b
                weights_path: /models/one
                tp_size: 1
                min_replicas: 1
                max_replicas: 1
                vllm_image: image:one
                slo: {ttft_p95_ms: 1, tpot_p95_ms: 1, e2e_p95_ms: 1}
                trs: {w_p: 0.04, w_d: 1.0, lambda_wait: 2.625, qmin: 1.0, ema_alpha: 0.5, theta_m: 0.0, tau_crit: 0.8, tau_low: 1.0, tau_high: 1.25, qsat: 4.0, epsat: 0.05, hsat: 3}
            """
        ),
        encoding="utf-8",
    )

    [service] = build_services(load_registry(str(path)))

    assert service["kind"] == "Service"
    assert service["metadata"] == {
        "name": "dsqwen-7b",
        "namespace": "default",
        "labels": {"model.aibrix.ai/name": "dsqwen-7b"},
    }
    assert service["spec"]["selector"] == {"model.aibrix.ai/name": "dsqwen-7b", "tre.aibrix.io/routable": "true"}
    assert service["spec"]["ports"] == [
        {"name": "http", "port": 8000, "targetPort": 8000, "protocol": "TCP"}
    ]


def test_build_httproutes_creates_model_header_route_to_service(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        textwrap.dedent(
            """
            cluster:
              nodes:
                - {name: node-75, gpus: 1, gpu_uuids: [GPU-75-0], two_gpu_slots: []}
            models:
              - name: dsqwen-7b
                weights_path: /models/one
                tp_size: 1
                min_replicas: 1
                max_replicas: 1
                vllm_image: image:one
                slo: {ttft_p95_ms: 1, tpot_p95_ms: 1, e2e_p95_ms: 1}
                trs: {w_p: 0.04, w_d: 1.0, lambda_wait: 2.625, qmin: 1.0, ema_alpha: 0.5, theta_m: 0.0, tau_crit: 0.8, tau_low: 1.0, tau_high: 1.25, qsat: 4.0, epsat: 0.05, hsat: 3}
            """
        ),
        encoding="utf-8",
    )

    [route] = build_httproutes(load_registry(str(path)))

    assert route["apiVersion"] == "gateway.networking.k8s.io/v1"
    assert route["kind"] == "HTTPRoute"
    assert route["metadata"] == {
        "name": "dsqwen-7b-router",
        "namespace": "aibrix-system",
        "labels": {"model.aibrix.ai/name": "dsqwen-7b", "tre.aibrix.io/managed": "true"},
    }
    assert route["spec"]["parentRefs"] == [
        {
            "group": "gateway.networking.k8s.io",
            "kind": "Gateway",
            "name": "aibrix-eg",
            "namespace": "aibrix-system",
        }
    ]
    [rule] = route["spec"]["rules"]
    assert rule["backendRefs"] == [
        {
            "group": "",
            "kind": "Service",
            "name": "dsqwen-7b",
            "namespace": "default",
            "port": 8000,
            "weight": 1,
        }
    ]
    assert {match["path"]["value"] for match in rule["matches"]} == {
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/embeddings",
        "/generate",
        "/generatevideo",
    }
    assert all(match["headers"] == [{"name": "model", "type": "Exact", "value": "dsqwen-7b"}] for match in rule["matches"])
    assert rule["timeouts"] == {"request": "600s"}


def test_write_manifests_includes_services_and_deployments(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        textwrap.dedent(
            """
            cluster:
              nodes:
                - {name: node-75, gpus: 2, gpu_uuids: [GPU-75-0, GPU-75-1], two_gpu_slots: [[0, 1]]}
            models:
              - name: one-gpu
                weights_path: /models/one
                tp_size: 1
                min_replicas: 1
                max_replicas: 2
                vllm_image: image:one
                slo: {ttft_p95_ms: 1, tpot_p95_ms: 1, e2e_p95_ms: 1}
                trs: {w_p: 0.04, w_d: 1.0, lambda_wait: 2.625, qmin: 1.0, ema_alpha: 0.5, theta_m: 0.0, tau_crit: 0.8, tau_low: 1.0, tau_high: 1.25, qsat: 4.0, epsat: 0.05, hsat: 3}
              - name: two-gpu
                weights_path: /models/two
                tp_size: 2
                min_replicas: 0
                max_replicas: 1
                vllm_image: image:two
                slo: {ttft_p95_ms: 1, tpot_p95_ms: 1, e2e_p95_ms: 1}
                trs: {w_p: 0.04, w_d: 1.0, lambda_wait: 2.625, qmin: 1.0, ema_alpha: 0.5, theta_m: 0.0, tau_crit: 0.8, tau_low: 1.0, tau_high: 1.25, qsat: 4.0, epsat: 0.05, hsat: 3}
            """
        ),
        encoding="utf-8",
    )
    registry = load_registry(str(path))

    written = write_manifests(registry, tmp_path / "models")

    assert len(build_resources(registry)) == 8
    assert sorted(item.name for item in written) == [
        "one-gpu-node-75-gpu-0.yaml",
        "one-gpu-node-75-gpu-1.yaml",
        "one-gpu-router.yaml",
        "one-gpu.yaml",
        "tre-v2-model-referencegrant-in-default.yaml",
        "two-gpu-node-75-gpu-0-1.yaml",
        "two-gpu-router.yaml",
        "two-gpu.yaml",
    ]


def test_build_referencegrant_allows_aibrix_httproute_to_default_service() -> None:
    from gen_model_manifests import build_referencegrant

    grant = build_referencegrant()
    assert grant["kind"] == "ReferenceGrant"
    assert grant["metadata"]["namespace"] == "default"
    assert grant["metadata"]["labels"]["tre.aibrix.io/managed"] == "true"
    assert grant["spec"]["from"] == [
        {"group": "gateway.networking.k8s.io", "kind": "HTTPRoute", "namespace": "aibrix-system"}
    ]
    assert grant["spec"]["to"] == [{"group": "", "kind": "Service"}]

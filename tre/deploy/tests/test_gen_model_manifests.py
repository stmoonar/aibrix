import textwrap

import yaml

from gen_model_manifests import build_deployments, build_resources, build_services, write_manifests
from tre_common.registry import load_registry


def test_build_deployments_creates_one_deployment_per_feasible_slot(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        textwrap.dedent(
            """
            cluster:
              nodes:
                - {name: node-75, gpus: 4, two_gpu_slots: [[0, 1], [2, 3]]}
                - {name: node-76, gpus: 4, two_gpu_slots: [[0, 1], [2, 3]]}
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
                - {name: node-75, gpus: 2, two_gpu_slots: [[0, 1]]}
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

    assert rendered["spec"]["template"]["spec"]["nodeSelector"] == {"kubernetes.io/hostname": "node-75"}
    assert rendered["spec"]["template"]["metadata"]["labels"]["tre.aibrix.io/routable"] == "true"
    assert rendered["metadata"]["annotations"]["tre.aibrix.io/gpu-ids"] == "0,1"
    assert rendered["spec"]["template"]["metadata"]["annotations"]["tre.aibrix.io/gpu-ids"] == "0,1"
    assert rendered["metadata"]["labels"]["tre.aibrix.io/gpu-ids"] == "0-1"
    assert rendered["spec"]["template"]["metadata"]["labels"]["tre.aibrix.io/gpu-ids"] == "0-1"
    assert {"name": "CUDA_VISIBLE_DEVICES", "value": "0,1"} in container["env"]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "2"
    assert "--tensor-parallel-size" in container["command"]
    assert "--enable_sleep_mode" in container["command"]


def test_cuda_visible_devices_uses_container_local_ordinals(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        textwrap.dedent(
            """
            cluster:
              nodes:
                - {name: node-75, gpus: 4, two_gpu_slots: [[0, 1], [2, 3]]}
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
    assert {"name": "CUDA_VISIBLE_DEVICES", "value": "0"} in one_gpu["env"]
    assert {"name": "CUDA_VISIBLE_DEVICES", "value": "0,1"} in two_gpu["env"]
    assert deployments["one-gpu-node-75-gpu-2"]["metadata"]["labels"]["tre.aibrix.io/gpu-ids"] == "2"
    assert deployments["one-gpu-node-75-gpu-2"]["metadata"]["annotations"]["tre.aibrix.io/gpu-ids"] == "2"
    assert deployments["two-gpu-node-75-gpu-2-3"]["metadata"]["labels"]["tre.aibrix.io/gpu-ids"] == "2-3"
    assert deployments["two-gpu-node-75-gpu-2-3"]["metadata"]["annotations"]["tre.aibrix.io/gpu-ids"] == "2,3"


def test_build_services_creates_one_model_service_with_gateway_selector(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        textwrap.dedent(
            """
            cluster:
              nodes:
                - {name: node-75, gpus: 1, two_gpu_slots: []}
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


def test_write_manifests_includes_services_and_deployments(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        textwrap.dedent(
            """
            cluster:
              nodes:
                - {name: node-75, gpus: 2, two_gpu_slots: [[0, 1]]}
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

    assert len(build_resources(registry)) == 5
    assert sorted(item.name for item in written) == [
        "one-gpu-node-75-gpu-0.yaml",
        "one-gpu-node-75-gpu-1.yaml",
        "one-gpu.yaml",
        "two-gpu-node-75-gpu-0-1.yaml",
        "two-gpu.yaml",
    ]

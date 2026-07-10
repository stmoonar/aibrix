import textwrap

import pytest

from tre_common.registry import load_registry


REGISTRY_YAML = """
cluster:
    nodes:
      - name: node-75
        gpus: 4
        gpu_uuids: [GPU-75-0, GPU-75-1, GPU-75-2, GPU-75-3]
        two_gpu_slots: [[0, 1], [2, 3]]
      - name: node-76
        gpus: 4
        gpu_uuids: [GPU-76-0, GPU-76-1, GPU-76-2, GPU-76-3]
        two_gpu_slots: [[0, 1], [2, 3]]
models:
  - name: dsqwen-7b
    weights_path: /data/nfs_shared_data/Qwen1.5-7B-Chat
    tp_size: 1
    min_replicas: 1
    max_replicas: 4
    vllm_image: vllm/vllm-openai:0.10.1-sleep
    slo: {ttft_p95_ms: 1200, tpot_p95_ms: 100, e2e_p95_ms: 10000}
    alt_thresholds:
      queue_len: {theta: 6.5, direction: lower_is_healthier}
    trs:
      w_p: 0.04
      w_d: 1.0
      lambda_wait: 2.625
      qmin: 1.0
      ema_alpha: 0.5
      theta_m: 0.0
      tau_crit: 0.8
      tau_low: 1.0
      tau_high: 1.25
      qsat: 4.0
      epsat: 0.05
      hsat: 3
"""


def test_load_registry_exposes_models_and_cluster_topology(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(textwrap.dedent(REGISTRY_YAML), encoding="utf-8")

    registry = load_registry(str(path))

    assert [model.name for model in registry.models()] == ["dsqwen-7b"]
    assert registry.model("dsqwen-7b").tp_size == 1
    assert registry.model("dsqwen-7b").slo.ttft_p95_ms == 1200.0
    assert registry.model("dsqwen-7b").trs.lambda_wait == 2.625
    assert registry.model("dsqwen-7b").alt_thresholds["queue_len"].theta == 6.5
    assert registry.topology().nodes[0].gpu_uuids == ("GPU-75-0", "GPU-75-1", "GPU-75-2", "GPU-75-3")
    assert registry.topology().nodes[0].two_gpu_slots == ((0, 1), (2, 3))
    assert registry.validate() == []


def test_load_registry_defaults_to_tre_deploy_registry_yaml():
    registry = load_registry()

    assert [model.name for model in registry.models()] == [
        "dsqwen-7b",
        "dsllama-8b",
        "dsqwen-14b",
    ]
    assert registry.model("dsqwen-14b").tp_size == 2
    assert registry.validate() == []


def test_registry_reports_validation_errors_for_duplicate_models_and_bad_slots(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        """
cluster:
  nodes:
    - {name: node-75, gpus: 4, gpu_uuids: [GPU-0, GPU-1], two_gpu_slots: [[0, 4]]}
models:
  - name: m
    weights_path: /m
    tp_size: 1
    min_replicas: 1
    max_replicas: 1
    vllm_image: image
    slo: {ttft_p95_ms: 1, tpot_p95_ms: 1, e2e_p95_ms: 1}
    trs: {w_p: 0.04, w_d: 1.0, lambda_wait: 2.625, qmin: 1.0, ema_alpha: 0.5, theta_m: 0.0, tau_crit: 0.8, tau_low: 1.0, tau_high: 1.25, qsat: 4.0, epsat: 0.05, hsat: 3}
  - name: m
    weights_path: /m2
    tp_size: 1
    min_replicas: 1
    max_replicas: 1
    vllm_image: image
    slo: {ttft_p95_ms: 1, tpot_p95_ms: 1, e2e_p95_ms: 1}
    trs: {w_p: 0.04, w_d: 1.0, lambda_wait: 2.625, qmin: 1.0, ema_alpha: 0.5, theta_m: 0.0, tau_crit: 0.8, tau_low: 1.0, tau_high: 1.25, qsat: 4.0, epsat: 0.05, hsat: 3}
""",
        encoding="utf-8",
    )

    registry = load_registry(str(path))

    assert any("duplicate model" in error for error in registry.validate())
    assert any("gpu_uuids length" in error for error in registry.validate())
    assert any("outside gpu range" in error for error in registry.validate())
    with pytest.raises(KeyError):
        registry.model("missing")


def test_registry_validates_alt_threshold_direction_and_theta(tmp_path):
    path = tmp_path / "registry.yaml"
    bad_direction = textwrap.dedent(REGISTRY_YAML).replace(
        "direction: lower_is_healthier", "direction: higher_is_healthier"
    )
    path.write_text(bad_direction, encoding="utf-8")
    errors = load_registry(str(path)).validate()
    assert any("alt_thresholds.queue_len.direction" in error for error in errors)

    path.write_text(
        textwrap.dedent(REGISTRY_YAML).replace("theta: 6.5", "theta: 0"),
        encoding="utf-8",
    )
    errors = load_registry(str(path)).validate()
    assert any("alt_thresholds.queue_len.theta must be positive" in error for error in errors)
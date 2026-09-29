"""vLLM engine settings in the registry (plan 2026-09-27 D9): ``vllm.env`` /
``models[].vllm_env``, ``max_model_len``, ``sleep_mode_backend`` and the shipped
0.30 fork migration."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from gen_model_manifests import build_deployments, build_model_deployment
from tre_common.bindings import render_binding_set
from tre_common.registry import DEFAULT_VLLM_ENV, Registry, load_registry

DEPLOY_ROOT = Path(__file__).resolve().parents[1]

REGISTRY = """
cluster:
  nodes:
    - {name: node-a, gpus: 2, gpu_uuids: [GPU-A-0, GPU-A-1], two_gpu_slots: [[0, 1]]}
models:
  - name: m
    weights_path: /models/m
    tp_size: 1
    min_replicas: 0
    max_replicas: 1
    vllm_image: image:fork
    vllm_extra_args: [--gpu-memory-utilization, '0.85']
    slo: {ttft_p95_ms: 1, tpot_p95_ms: 1, e2e_p95_ms: 1}
    trs: {w_p: 0.04, w_d: 1.0, lambda_wait: 2.625, qmin: 1.0, ema_alpha: 0.5, theta_m: 0.0, tau_crit: 0.8, tau_low: 1.0, tau_high: 1.25, qsat: 4.0, epsat: 0.05, hsat: 3}
"""


def _load(tmp_path: Path, model_extra: str = "", top_extra: str = "") -> Registry:
    text = textwrap.dedent(REGISTRY)
    if model_extra:
        text += textwrap.indent(textwrap.dedent(model_extra), "    ")
    text += textwrap.dedent(top_extra)
    path = tmp_path / "registry.yaml"
    path.write_text(text, encoding="utf-8")
    return load_registry(str(path))


def _vllm(deployment: dict) -> dict:
    return next(c for c in deployment["spec"]["template"]["spec"]["containers"] if c["name"] == "vllm-openai")


def _env(container: dict) -> dict[str, str]:
    return {e["name"]: e.get("value") for e in container["env"]}


def test_without_a_vllm_section_pods_still_get_dev_mode(tmp_path):
    registry = _load(tmp_path)
    assert registry.validate() == []
    (deployment,) = build_deployments(registry)
    env = _env(_vllm(deployment))
    assert env["VLLM_SERVER_DEV_MODE"] == "1"
    assert set(env) == {"NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", *DEFAULT_VLLM_ENV}
    command = _vllm(deployment)["command"]
    assert "--max-model-len" not in command and "--sleep-mode-backend" not in command


def test_env_merges_global_then_model(tmp_path):
    registry = _load(
        tmp_path,
        model_extra="vllm_env: {A: model, C: 3}\n",
        top_extra="vllm:\n  env: {A: global, B: 'yes', FLAG: true}\n",
    )
    assert registry.validate() == []
    env = _env(_vllm(build_deployments(registry)[0]))
    assert env["A"] == "model" and env["B"] == "yes" and env["C"] == "3" and env["FLAG"] == "true"
    assert env["VLLM_SERVER_DEV_MODE"] == "1"


def test_max_model_len_and_backend_render_flags(tmp_path):
    registry = _load(tmp_path, model_extra="max_model_len: 32768\nsleep_mode_backend: pinned_weights\n")
    assert registry.validate() == []
    command = _vllm(build_deployments(registry)[0])["command"]
    assert command[command.index("--max-model-len") + 1] == "32768"
    assert command[command.index("--sleep-mode-backend") + 1] == "pinned_weights"
    assert registry.model("m").vllm_args[-4:] == ("--max-model-len", "32768", "--sleep-mode-backend", "pinned_weights")


@pytest.mark.parametrize(
    ("model_extra", "top_extra", "needle"),
    [
        ("sleep_mode_backend: bogus\n", "", "sleep_mode_backend must be one of"),
        ("max_model_len: 0\n", "", "max_model_len must be positive"),
        ("max_model_len: 4096\nvllm_extra_args: [--max-model-len, '4096']\n", "", "not both"),
        ("", "vllm:\n  env: {CUDA_VISIBLE_DEVICES: '0'}\n", "set per binding"),
        ("vllm_env: {NVIDIA_VISIBLE_DEVICES: all}\n", "", "set per binding"),
        ("", "vllm:\n  env: {VLLM_SERVER_DEV_MODE: '0'}\n", "TRE needs it"),
    ],
)
def test_invalid_engine_settings_are_rejected(tmp_path, model_extra, top_extra, needle):
    registry = _load(tmp_path, model_extra=model_extra, top_extra=top_extra)
    errors = registry.validate()
    assert any(needle in error for error in errors), errors


def test_unknown_vllm_section_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="vllm: unknown keys"):
        _load(tmp_path, top_extra="vllm:\n  enviroment: {}\n")


# ---- the shipped registry (vLLM 0.30 fork, plan D9) ----------------------------


def test_shipped_registry_runs_the_030_fork_with_default_cumem():
    registry = load_registry(str(DEPLOY_ROOT / "registry.yaml"))
    assert registry.validate() == []
    assert registry.reissue().enabled
    for model in registry.models():
        # Tokenizer-consistency fix (fork 8dc0f2a7, 2026-09-30); image ID a96754e185b1.
        assert model.vllm_image == "vllm-openai-tre:0.30.0-ts-8dc0f2a7", model.name
        assert set(model.vllm_features) == {"sleep_reject_new", "abort_return_token_ids"}
        assert model.sleep_mode_backend is None  # cumem: no --sleep-mode-backend flag
        env = registry.vllm_env_for(model)
        assert env["VLLM_SERVER_DEV_MODE"] == "1"
        # pinned_max_round_threshold_mb breaks the second cumem sleep (2026-09-28).
        assert "PYTORCH_ALLOC_CONF" not in env
    # v1 alignment (2026-09-29): no model pins --max-model-len; each serves its own maximum.
    assert all(model.max_model_len is None for model in registry.models())


def test_shipped_manifests_carry_no_removed_030_flags():
    registry = load_registry(str(DEPLOY_ROOT / "registry.yaml"))
    for deployment in build_deployments(registry):
        vllm = _vllm(deployment)
        command = vllm["command"]
        assert "--swap-space" not in command  # removed in vLLM 0.30
        assert "--sleep-reject-new" in command and "--abort-return-token-ids" in command
        assert "--sleep-mode-backend" not in command
        assert "--max-model-len" not in command  # v1 alignment: the model's own maximum
        # v1 alignment: max_num_seqs is v1's effective value (the 0.10.1 OpenAI-server default
        # on a <70 GiB / A100 GPU), written out; prefix caching is off on every model.
        assert command[command.index("--max-num-seqs") + 1] == "256"
        assert "--no-enable-prefix-caching" in command
        # Symmetric engine args (2026-09-29): chunked prefill with a 2048-token step budget
        # is explicit on every model, 14b included.
        assert "--enable-chunked-prefill" in command
        assert command[command.index("--max-num-batched-tokens") + 1] == "2048"
        env = _env(vllm)
        assert "VLLM_USE_MODELSCOPE" not in env
        assert env["HF_HUB_OFFLINE"] == "1"


def test_runtime_create_renders_the_same_pod_as_make_manifests():
    registry = load_registry(str(DEPLOY_ROOT / "registry.yaml"))
    rendered = {d["metadata"]["name"]: d for d in build_deployments(registry)}
    for spec in render_binding_set(registry):
        created = build_model_deployment(registry, spec.model, spec.node, spec.gpu_ids)
        assert created == rendered[created["metadata"]["name"]]

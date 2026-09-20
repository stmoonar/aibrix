"""Prefix caching must be off on every model, in the registry and in the manifests.

vLLM 0.10.1's V1 engine enables prefix caching by default. With it on, a load scan whose
requests share a prompt gets its prefill served from cache, and the measured capacity
*rises* with prompt length instead of falling - which is exactly what dsqwen-14b's
capacity table did while 7b/8b (which carried the flag) behaved normally. The senders now
send a unique prompt per request, but the flag is the belt to that braces: it keeps the
three models comparable and keeps any residual prompt overlap from being free.
"""
from __future__ import annotations

from pathlib import Path

import yaml

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
FLAG = "--no-enable-prefix-caching"


def _registry() -> dict:
    return yaml.safe_load((DEPLOY_ROOT / "registry.yaml").read_text(encoding="utf-8"))


def test_every_registry_model_disables_prefix_caching() -> None:
    models = _registry()["models"]
    assert {m["name"] for m in models} == {"dsqwen-7b", "dsllama-8b", "dsqwen-14b"}
    for model in models:
        assert FLAG in (model.get("vllm_extra_args") or []), model["name"]


def test_every_generated_model_deployment_disables_prefix_caching() -> None:
    checked = 0
    for path in sorted((DEPLOY_ROOT / "models").glob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if not doc or doc.get("kind") != "Deployment":
                continue
            for container in doc["spec"]["template"]["spec"]["containers"]:
                command = container.get("command") or []
                if "vllm.entrypoints.openai.api_server" not in command:
                    continue
                assert FLAG in command, f"{path.name}/{container['name']}"
                checked += 1
    assert checked == 20  # 8 x 7b + 8 x 8b + 4 x 14b engine containers

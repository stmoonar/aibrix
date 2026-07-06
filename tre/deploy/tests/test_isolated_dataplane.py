from __future__ import annotations

from pathlib import Path

import yaml

from gen_model_manifests import build_httproutes, build_referencegrant
from tre_common.registry import load_registry


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
OVERLAY = DEPLOY_ROOT / "overlays" / "tre-v2"


def _docs(path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]


def _by_kind(path: Path, kind: str, name: str | None = None) -> dict:
    for d in _docs(path):
        if d["kind"] == kind and (name is None or d["metadata"]["name"] == name):
            return d
    raise AssertionError(f"{kind}/{name} not found in {path}")


def test_gateway_is_isolated_tre_serving_gateway() -> None:
    gw = _by_kind(OVERLAY / "gateway.yaml", "Gateway", "tre-aibrix-eg")
    assert gw["metadata"]["namespace"] == "tre-v2"
    assert gw["spec"]["gatewayClassName"] == "aibrix-eg"
    listener = gw["spec"]["listeners"][0]
    assert listener["port"] == 80
    assert listener["allowedRoutes"]["namespaces"]["from"] == "Same"


def test_gateway_plugins_scrapes_to_tre_v2_redis_with_podlist_rbac() -> None:
    path = OVERLAY / "gateway-plugins.yaml"
    dep = _by_kind(path, "Deployment", "tre-gateway-plugins")
    spec = dep["spec"]["template"]["spec"]
    assert spec["serviceAccountName"] == "tre-gateway-plugins"
    env = {e["name"]: e.get("value") for e in spec["containers"][0]["env"]}
    assert env["REDIS_HOST"] == "tre-v2-redis"
    assert env["REDIS_PORT"] == "6379"
    assert env["TRE_REDIS_SCHEMA"] == "dual"
    assert env["AIBRIX_POD_METRIC_REFRESH_INTERVAL_MS"] == "50"
    init_cmd = " ".join(spec["initContainers"][0]["command"])
    assert "tre-v2-redis" in init_cmd
    assert "aibrix-redis-master" not in init_cmd
    svc = _by_kind(path, "Service", "tre-gateway-plugins")
    assert 50052 not in [p["port"] for p in svc["spec"]["ports"]]
    role = _by_kind(path, "ClusterRole", "tre-gateway-plugins-role")
    pod_rule = next(r for r in role["rules"] if r["resources"] == ["pods"])
    assert "list" in pod_rule["verbs"]


def test_no_extproc_reserved_router_or_policy_shipped() -> None:
    # ADR-0008: isolated plane deliberately omits ext-proc / reserved-router.
    kinds: list[str] = []
    names: list[str] = []
    for path in OVERLAY.glob("*.yaml"):
        for d in _docs(path):
            kinds.append(d["kind"])
            names.append(d.get("metadata", {}).get("name", ""))
    assert "EnvoyExtensionPolicy" not in kinds
    assert not any("reserved-router" in n for n in names)


def test_generator_gateway_target_is_parameterizable() -> None:
    reg = load_registry(str(DEPLOY_ROOT / "registry.yaml"))
    routes = build_httproutes(reg, gateway_namespace="other-ns", gateway_name="other-gw")
    assert routes[0]["metadata"]["namespace"] == "other-ns"
    assert routes[0]["spec"]["parentRefs"][0]["name"] == "other-gw"
    grant = build_referencegrant(gateway_namespace="other-ns")
    assert grant["spec"]["from"][0]["namespace"] == "other-ns"
    default_routes = build_httproutes(reg)
    assert default_routes[0]["metadata"]["namespace"] == "tre-v2"
    assert default_routes[0]["spec"]["parentRefs"][0]["name"] == "tre-aibrix-eg"

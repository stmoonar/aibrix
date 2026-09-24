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
    # 2026-09-24 amendment: the plugin is also the ext_proc router of the tre-v2 gateway.
    assert 50052 in [p["port"] for p in svc["spec"]["ports"]]
    role = _by_kind(path, "ClusterRole", "tre-gateway-plugins-role")
    pod_rule = next(r for r in role["rules"] if r["resources"] == ["pods"])
    assert "list" in pod_rule["verbs"]


def test_extproc_objects_are_confined_to_tre_v2() -> None:
    # ADR-0008 as amended 2026-09-24: ext_proc is back (least-gpu-cache, as in v1), but
    # every ext_proc object lives in tre-v2 and targets only tre-v2 objects. No shared
    # (class-level) Envoy config, nothing in aibrix-system.
    docs = _docs(OVERLAY / "gateway-extproc.yaml")
    assert sorted(d["kind"] for d in docs) == [
        "ClientTrafficPolicy",
        "EnvoyExtensionPolicy",
        "EnvoyPatchPolicy",
        "EnvoyPatchPolicy",
        "HTTPRoute",
    ]
    for d in docs:
        assert d["metadata"]["namespace"] == "tre-v2", d["metadata"]["name"]
        spec = d["spec"]
        refs = list(spec.get("targetRefs", [])) + ([spec["targetRef"]] if "targetRef" in spec else [])
        refs += list(spec.get("parentRefs", []))
        for ref in refs:
            assert ref.get("namespace", "tre-v2") == "tre-v2"
            assert ref["name"] in {"tre-aibrix-eg", "tre-reserved-router"}
    for path in OVERLAY.glob("*.yaml"):
        for d in _docs(path):
            assert d["kind"] not in {"EnvoyProxy", "EnvoyGateway", "GatewayClass"}, path.name
    for d in _docs(OVERLAY / "gateway-plugins.yaml"):
        assert d.get("metadata", {}).get("namespace") in {None, "tre-v2"}, d["metadata"]


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

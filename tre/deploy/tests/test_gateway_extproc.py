"""Guards for the tre-v2 ext_proc serving path (overlays/tre-v2/gateway-extproc.yaml).

v1 routed every request through the gateway plugin with routing-strategy least-gpu-cache
and an ORIGINAL_DST cluster. The tre-v2 rebuild copies v1's values
(/root/aibrix-main/config_tre, pinned below as V1_*) and keeps what the v2 experiments
rely on:

* per-model admission limits identical to the per-model Service path (the t1 lesson: no
  Envoy-default 1024 shared by all models), for every registered model;
* per-model Envoy counters that the SafeScale donor-health guard and the openloop sentinel
  already parse, so the running controller keeps working without a rebuild;
* the routable-label gate switched on in the plugin, because ORIGINAL_DST bypasses the
  Service selector that otherwise hides sleeping / probe-hidden pods;
* HOT_SWITCH=0 together with min_replicas >= 1 for every model, which is how v1 behaved.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from scripts.openloop import parse_envoy_counters
from tre_common.registry import load_registry
from tre_controller.gateway_health import model_cluster_prefix, parse_envoy_cluster_counters

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
OVERLAY = DEPLOY_ROOT / "overlays" / "tre-v2"
EXTPROC = OVERLAY / "gateway-extproc.yaml"
PLUGINS = OVERLAY / "gateway-plugins.yaml"
TRE_ARM_BTP = DEPLOY_ROOT / "gateway-hardening" / "backendtrafficpolicy-tre-v2.yaml"

POLICY_NAME = "tre-gateway-plugins-extension-policy"
EXTPROC_CLUSTER = f"envoyextensionpolicy/tre-v2/{POLICY_NAME}/extproc/0"
EXTPROC_FILTER = f"envoy.filters.http.ext_proc/{EXTPROC_CLUSTER}"
ROUTE_CONFIG = "tre-v2/tre-aibrix-eg/http"
UNATTRIBUTED = "tre-v2/original-dst/unattributed"
CLUSTER_TYPE = "type.googleapis.com/envoy.config.cluster.v3.Cluster"
ROUTE_TYPE = "type.googleapis.com/envoy.config.route.v3.RouteConfiguration"

# v1 (config_tre) values this file must reproduce.
V1_ROUTE_TIMEOUT = "120s"  # EnvoyGateway.yaml aibrix-epp original_route
V1_ROUTE_TIMEOUT_PATCHED = "150s"  # envoy-gateway-route-timeouts.yaml
V1_ORIGINAL_DST_CONNECT_TIMEOUT = "6s"
V1_EXTPROC_BREAKER = {  # envoy-gateway-extproc-circuit-breaker.yaml
    "maxConnections": 8000,
    "maxPendingRequests": 80000,
    "maxParallelRequests": 80000,
    "maxParallelRetries": 5,
}
V1_EXTPROC_HTTP2 = {  # envoy-gateway-http2-options.yaml
    "maxConcurrentStreams": 512,
    "initialStreamWindowSize": 65536,
    "initialConnectionWindowSize": 1048576,
}
# v1's preconnect patch (envoy-gateway-http2-options.yaml, /preconnect_policy) used the
# misspelt field `preconnect_ratio` (Envoy: `per_upstream_preconnect_ratio`), so Envoy
# Gateway rejected it (EnvoyPatchPolicy Programmed=False, "unknown field") and it never
# took effect in v1. It is deliberately NOT ported: see
# docs/note-20260924-preconnect-removed.md (2026-09-24, user decision).


def _docs(path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]


def _one(kind: str, name: str | None = None) -> dict:
    (doc,) = [
        d for d in _docs(EXTPROC) if d["kind"] == kind and (name is None or d["metadata"]["name"] == name)
    ]
    return doc


def _models() -> list[str]:
    return [m.name for m in load_registry(str(DEPLOY_ROOT / "registry.yaml")).models()]


def _gateway_policy(name: str) -> dict:
    policy = _one("EnvoyPatchPolicy", name)
    assert policy["spec"]["type"] == "JSONPatch"
    assert policy["spec"]["targetRef"] == {
        "group": "gateway.networking.k8s.io",
        "kind": "Gateway",
        "name": "tre-aibrix-eg",
    }
    return policy


def _patches() -> list[dict]:
    return _gateway_policy("tre-original-dst")["spec"]["jsonPatches"]


def _clusters() -> dict[str, dict]:
    out = {}
    for patch in _patches():
        if patch["type"] == CLUSTER_TYPE and patch["operation"]["path"] == "":
            assert patch["operation"]["op"] == "add"
            value = patch["operation"]["value"]
            assert value["name"] == patch["name"]
            out[value["name"]] = value
    return out


def _route_patches() -> list[dict]:
    routes = []
    for patch in _patches():
        if patch["type"] == ROUTE_TYPE:
            assert patch["name"] == ROUTE_CONFIG
            assert patch["operation"]["op"] == "add"
            assert patch["operation"]["path"] == "/virtual_hosts/0/routes/0"
            routes.append(patch["operation"]["value"])
    return routes


def _per_model_limits() -> dict:
    docs = _docs(TRE_ARM_BTP)
    values = {tuple(sorted(d["spec"]["circuitBreaker"].items())) for d in docs}
    assert len(values) == 1
    return docs[0]["spec"]["circuitBreaker"]


def original_dst_cluster(model: str) -> str:
    return model_cluster_prefix(model, route_namespace="tre-v2") + "original-dst"


def test_one_original_dst_cluster_per_registered_model_plus_unattributed() -> None:
    assert set(_clusters()) == {original_dst_cluster(m) for m in _models()} | {UNATTRIBUTED}


def test_original_dst_clusters_are_v1_clusters_with_the_per_model_admission_limits() -> None:
    cb = _per_model_limits()
    expected = {
        "priority": "DEFAULT",
        "max_connections": cb["maxConnections"],
        "max_pending_requests": cb["maxPendingRequests"],
        "max_requests": cb["maxParallelRequests"],
        "max_retries": cb["maxParallelRetries"],
    }
    for name, cluster in _clusters().items():
        # v1 original_destination_cluster, key for key ...
        assert cluster["type"] == "ORIGINAL_DST", name
        assert cluster["lb_policy"] == "CLUSTER_PROVIDED", name
        assert cluster["connect_timeout"] == V1_ORIGINAL_DST_CONNECT_TIMEOUT, name
        assert cluster["dns_lookup_family"] == "V4_ONLY", name
        assert cluster["original_dst_lb_config"] == {"use_http_header": True, "http_header_name": "target-pod"}
        # ... plus the per-model breaker (never Envoy's implicit 1024 shared per cluster),
        # and nothing else.
        assert cluster["circuit_breakers"]["thresholds"] == [expected], name
        assert set(cluster) == {
            "name", "type", "lb_policy", "connect_timeout", "dns_lookup_family",
            "original_dst_lb_config", "circuit_breakers",
        }, name


def test_per_model_routes_match_strategy_and_model_and_enable_extproc() -> None:
    routes = _route_patches()
    by_name = {r["name"]: r for r in routes}
    assert set(by_name) == {f"tre-original-route/{m}" for m in _models()} | {"tre-original-route/unattributed"}
    for model in _models():
        route = by_name[f"tre-original-route/{model}"]
        headers = {h["name"]: h["string_match"] for h in route["match"]["headers"]}
        assert headers == {"routing-strategy": {"safe_regex": {"regex": ".*"}}, "model": {"exact": model}}
        assert route["route"]["cluster"] == original_dst_cluster(model)
    for route in routes:
        assert route["match"]["prefix"] == "/v1"  # v1 config_tre aibrix-epp
        assert list(route["typed_per_filter_config"]) == [EXTPROC_FILTER]
        assert route["typed_per_filter_config"][EXTPROC_FILTER]["config"] == {}
    unattributed = by_name["tre-original-route/unattributed"]
    assert [h["name"] for h in unattributed["match"]["headers"]] == ["routing-strategy"]
    assert unattributed["route"]["cluster"] == UNATTRIBUTED


def test_catch_all_is_inserted_first_so_it_ends_below_the_model_routes() -> None:
    # Every patch prepends at index 0, so the FIRST patch ends up LAST in the table.
    names = [r["name"] for r in _route_patches()]
    assert names[0] == "tre-original-route/unattributed"


def test_route_timeouts_are_v1_120s_then_the_v1_150s_patch() -> None:
    routes = _route_patches()
    for route in routes:
        assert route["route"] == {"cluster": route["route"]["cluster"], "timeout": V1_ROUTE_TIMEOUT}
    base = _gateway_policy("tre-original-dst")
    raise_ = _gateway_policy("tre-route-timeouts")
    assert base["spec"]["priority"] < raise_["spec"]["priority"]  # EG applies ascending
    replaced = []
    for patch in raise_["spec"]["jsonPatches"]:
        assert patch["type"] == ROUTE_TYPE and patch["name"] == ROUTE_CONFIG
        assert patch["operation"]["op"] == "replace"
        assert patch["operation"]["value"] == V1_ROUTE_TIMEOUT_PATCHED
        replaced.append(patch["operation"]["path"])
    # Exactly the routing-strategy routes, which the add patches put at the table head.
    assert replaced == [f"/virtual_hosts/0/routes/{i}/route/timeout" for i in range(len(routes))]


def test_extension_policy_is_v1s() -> None:
    policy = _one("EnvoyExtensionPolicy")
    assert policy["metadata"]["name"] == POLICY_NAME
    assert policy["spec"]["targetRefs"] == [
        {"group": "gateway.networking.k8s.io", "kind": "HTTPRoute", "name": "tre-reserved-router"}
    ]
    (ext,) = policy["spec"]["extProc"]
    assert ext["backendRefs"] == [{"name": "tre-gateway-plugins", "port": 50052}]
    assert ext["processingMode"] == {"request": {"body": "Buffered"}, "response": {"body": "Streamed"}}
    assert ext["messageTimeout"] == "60s" and ext["failOpen"] is False
    assert ext["backendSettings"]["circuitBreaker"] == V1_EXTPROC_BREAKER
    assert ext["backendSettings"]["http2"] == V1_EXTPROC_HTTP2
    cb = _per_model_limits()
    worst_case_streams = len(_models()) * (cb["maxParallelRequests"] + cb["maxPendingRequests"])
    assert V1_EXTPROC_BREAKER["maxParallelRequests"] >= worst_case_streams  # never the binding cap
    # No preconnect: v1's patch never took effect (misspelt field); see the note above
    # (docs/note-20260924-preconnect-removed.md). Nothing may patch the ext_proc cluster.
    assert [p for p in _patches() if p["name"] == EXTPROC_CLUSTER] == []
    assert "preconnect" not in yaml.safe_dump(_gateway_policy("tre-original-dst"))


def test_reserved_route_serves_no_real_traffic() -> None:
    route = _one("HTTPRoute")
    (rule,) = route["spec"]["rules"]
    paths = [m["path"]["value"] for m in rule["matches"]]
    assert paths == ["/tre-extproc-reserved"]
    assert not any(p.startswith("/v1") for p in paths)


def test_body_buffering_is_v1s_4mib() -> None:
    ctp = _one("ClientTrafficPolicy")
    assert ctp["spec"]["targetRefs"][0]["name"] == "tre-aibrix-eg"
    assert ctp["spec"]["connection"]["bufferLimit"] == 4194304


def test_plugin_deployment_gates_routing_on_the_routable_label() -> None:
    dep = next(d for d in _docs(PLUGINS) if d["kind"] == "Deployment")
    assert dep["spec"]["replicas"] == 1  # scrape loop is not leader-elected (v1: 1 too)
    container = dep["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e.get("value") for e in container["env"]}
    assert env["TRE_ROUTABLE_LABEL_FILTER"] == "true"
    assert env["TRE_ROUTE_MODEL_HEADER"] == "true"
    assert env["AIBRIX_POD_METRIC_REFRESH_INTERVAL_MS"] == "50"  # v1 refresh cadence
    assert container["readinessProbe"]["grpc"]["port"] == 50052
    assert container["resources"]["limits"] == {"cpu": "2", "memory": "8Gi"}  # v1 sizing


def test_no_gateway_wakeup_and_no_model_can_reach_zero() -> None:
    """HOT_SWITCH=0 (decision 2026-09-24): the v2 plugin would otherwise submit a wake-up
    for a model with zero routable pods on the request path (TRE-PATCH P2-GW-002), which
    v1 never did under least-gpu-cache. v1 never hit that case because every model kept
    >= 1 replica; min_replicas >= 1 for all models restores that equivalence."""
    dep = next(d for d in _docs(PLUGINS) if d["kind"] == "Deployment")
    env = {e["name"]: e.get("value") for e in dep["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["HOT_SWITCH"] == "0"
    assert "SERVEMENT_URL" not in env
    registry = load_registry(str(DEPLOY_ROOT / "registry.yaml"))
    assert {m.name: m.min_replicas for m in registry.models()} == {m: 1 for m in _models()}


def test_existing_stats_consumers_attribute_original_dst_traffic_per_model() -> None:
    lines = []
    for model, n in zip(_models(), (11, 22, 33)):
        cluster = original_dst_cluster(model)
        lines.append(
            f'envoy_cluster_upstream_rq_xx{{envoy_response_code_class="2",envoy_cluster_name="{cluster}"}} {n}'
        )
        lines.append(f'envoy_cluster_upstream_rq_pending_overflow{{envoy_cluster_name="{cluster}"}} 1')
    lines.append(
        f'envoy_cluster_upstream_rq_xx{{envoy_response_code_class="2",envoy_cluster_name="{UNATTRIBUTED}"}} 99'
    )
    body = "\n".join(lines)
    counters = parse_envoy_cluster_counters(body, _models(), route_namespace="tre-v2")
    for model, n in zip(_models(), (11, 22, 33)):
        assert counters[model].requests == n + 1
        assert counters[model].errors == 1
    for model in _models():
        assert parse_envoy_counters(body, "upstream_rq_pending_overflow", cluster_filter=f"{model}-router") == 1

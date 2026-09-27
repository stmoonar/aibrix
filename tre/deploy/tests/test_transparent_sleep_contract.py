"""Cross-component contract of transparent sleep (plan 2026-09-27): the Go gateway
plugin (pkg/plugins/gateway, pkg/utils) and the Python service-manager / deploy
side must agree on Redis keys, JSON field names, k8s label / annotation names, the
plugin pod selector and the route timeout. Each side has its own tests; this file
pins them to each other (source-level, so it needs no Go toolchain)."""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
import yaml

from gen_model_manifests import ROUTABLE_LABEL as MANIFEST_ROUTABLE_LABEL, route_timeout_text
from tre_common import rediskeys
from tre_common.registry import SleepPolicy, load_registry

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DEPLOY_ROOT.parents[1]
GATEWAY_GO = REPO_ROOT / "pkg" / "plugins" / "gateway"
TRE_GO = GATEWAY_GO / "tre_transparent_sleep.go"
POD_GO = REPO_ROOT / "pkg" / "utils" / "pod.go"
OVERLAY = DEPLOY_ROOT / "overlays" / "tre-v2"
SM_ROOT = DEPLOY_ROOT.parent / "service-manager" / "tre_sm"

needs_go = pytest.mark.skipif(not TRE_GO.is_file(), reason="tre/ checked out without the Go tree")


def _go_string_consts(path: Path) -> dict[str, str]:
    return dict(re.findall(r'^\s*(\w+)\s*=\s*"([^"]*)"', path.read_text(encoding="utf-8"), re.M))


def _go_struct_json_tags(path: Path, struct: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"type {struct} struct \{{(.*?)\n\}}", text, re.S)
    assert match, f"struct {struct} not found in {path}"
    return re.findall(r'json:"([^",]+)', match.group(1))


def _docs(path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]


def _plugin_deployment() -> dict:
    return next(d for d in _docs(OVERLAY / "gateway-plugins.yaml") if d["kind"] == "Deployment")


def _registries(tmp_path: Path):
    """The repo registry and the overlay's tre-v2-registry ConfigMap copy."""
    params = next(d for d in _docs(OVERLAY / "params.yaml") if d["kind"] == "ConfigMap")
    copy = tmp_path / "params-registry.yaml"
    copy.write_text(params["data"]["registry.yaml"], encoding="utf-8")
    return {
        "registry.yaml": load_registry(str(DEPLOY_ROOT / "registry.yaml")),
        "params.yaml": load_registry(str(copy)),
    }


@needs_go
def test_redis_keys_match_the_go_constants() -> None:
    go = _go_string_consts(TRE_GO)
    assert rediskeys.GW_INSTANCES_KEY == go["TREGatewayInstancesKey"]
    assert rediskeys.gw_seen_key("pod-x") == go["TREGatewaySeenKeyPrefix"] + "pod-x"
    assert rediskeys.gw_inflight_key("pod-x") == go["TREGatewayInflightKeyPrefix"] + "pod-x"


@needs_go
def test_json_field_names_match_the_go_structs() -> None:
    seen = _go_struct_json_tags(TRE_GO, "treSeenValue")
    inflight = _go_struct_json_tags(TRE_GO, "treInflightValue")
    assert seen == ["gen", "routable", "ts"]
    assert inflight == ["total", "non_continuable", "ts"]
    # The documented Python contract ...
    assert '{"gen":int,"routable":bool,"ts":ms}' in rediskeys.gw_seen_key.__doc__
    assert '{"total":int,"non_continuable":int,"ts":ms}' in rediskeys.gw_inflight_key.__doc__
    # ... and what the SM reader actually looks up.
    from tre_sm.ops import sleep_primitive

    source = inspect.getsource(sleep_primitive)
    for field in ("gen", "routable", "total", "non_continuable"):
        assert f'entry.get("{field}"' in source, field


@needs_go
def test_label_annotation_and_header_names_match() -> None:
    go = _go_string_consts(TRE_GO)
    assert rediskeys.ROUTE_GEN_ANNOTATION == go["TRERouteGenAnnotation"] == "tre.aibrix.io/route-gen"
    go_label = _go_string_consts(POD_GO)["TRERoutableLabel"]
    from tre_sm.ops import k8s_ops

    assert go_label == MANIFEST_ROUTABLE_LABEL == k8s_ops.ROUTABLE_LABEL == "tre.aibrix.io/routable"
    # The exclude header (written by the retry sidecar, read by the plugin) is
    # documented with the same name as the Go constant.
    assert go["HeaderTREExcludePod"] == "x-tre-exclude-pod"
    assert "`x-tre-exclude-pod`" in (GATEWAY_GO / "ENV_VARS.md").read_text(encoding="utf-8")


def test_plugin_pod_selector_matches_the_gateway_plugins_deployment(tmp_path: Path) -> None:
    dep = _plugin_deployment()
    pod_labels = dep["spec"]["template"]["metadata"]["labels"]
    selectors = {"default": (SleepPolicy().plugin_namespace, SleepPolicy().plugin_label_selector)}
    for name, registry in _registries(tmp_path).items():
        sleep = registry.service_manager().sleep
        selectors[name] = (sleep.plugin_namespace, sleep.plugin_label_selector)
    for source, (namespace, selector) in selectors.items():
        assert namespace == dep["metadata"]["namespace"], source
        assert selector, source
        for term in selector.split(","):
            key, _, value = term.partition("=")
            assert pod_labels.get(key.strip()) == value.strip(), (source, term, pod_labels)


def test_plugin_instance_id_is_the_pod_name_the_sm_lists() -> None:
    """The SM counts Ready plugin pods (by NAME) as live instances that must ack, so the
    plugin's instance id (TRE_GW_INSTANCE_ID, else POD_NAME) must be the pod name."""
    container = _plugin_deployment()["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e for e in container["env"]}
    assert "TRE_GW_INSTANCE_ID" not in env
    assert env["POD_NAME"]["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.name"
    assert env["TRE_GW_COORDINATION"]["value"] == "true"
    assert env["TRE_ROUTABLE_LABEL_FILTER"]["value"] == "true"  # coordination needs the gate


def test_every_route_timeout_equals_the_registry_route_timeout(tmp_path: Path) -> None:
    registries = _registries(tmp_path)
    timeouts = {name: r.gateway().route_timeout_s for name, r in registries.items()}
    assert len(set(timeouts.values())) == 1, timeouts
    expected = route_timeout_text(timeouts["registry.yaml"])
    # ext_proc routes (EnvoyPatchPolicy tre-route-timeouts)
    policy = next(
        d for d in _docs(OVERLAY / "gateway-extproc.yaml")
        if d["kind"] == "EnvoyPatchPolicy" and d["metadata"]["name"] == "tre-route-timeouts"
    )
    assert {p["operation"]["value"] for p in policy["spec"]["jsonPatches"]} == {expected}
    # generated per-model HTTPRoutes (GETs and non-plugin POSTs)
    for router in sorted((DEPLOY_ROOT / "models").glob("*-router.yaml")):
        route = next(d for d in _docs(router) if d["kind"] == "HTTPRoute")
        assert {r["timeouts"]["request"] for r in route["spec"]["rules"]} == {expected}, router.name
    # the SM drain can never outlast the route
    for name, registry in registries.items():
        assert registry.service_manager().sleep.hard_cap_s <= timeouts[name], name


# --- retry / continuation sidecar (plan D2, D5, D6) ---------------------------------


def _sidecar_module():
    try:
        from tre_reissue import sidecar
    except ImportError:  # PYTHONPATH without tre/reissue
        import sys

        sys.path.insert(0, str(DEPLOY_ROOT.parent / "reissue"))
        from tre_reissue import sidecar
    return sidecar


def test_sidecar_speaks_the_sm_hidden_header() -> None:
    """SM -> sidecar: every SM /sleep carries X-TRE-Hidden: 1 after the hide; the
    sidecar refuses a /sleep without it (fail closed, D2)."""
    from tre_sm.ops import sleep_primitive, vllm_ops

    cfg = _sidecar_module().Config()
    assert cfg.hidden_header == sleep_primitive.HIDDEN_SLEEP_HEADER == "X-TRE-Hidden"
    assert '{"X-TRE-Hidden": "1"}' in inspect.getsource(vllm_ops)
    assert "/sleep" in cfg.sleep_paths and cfg.require_hidden_header is True
    assert cfg.listen_port == sleep_primitive.VLLM_PORT  # the SM talks to the serving port


def test_sidecar_plan_interface_names() -> None:
    """Plan 2026-09-27 interface contract: x-tre-exclude-pod (sidecar -> plugin),
    x-tre-continued (sidecar -> client), EngineSleeping (fork --sleep-reject-new), and
    the fork's --abort-return-token-ids field names (branch tre/transparent-sleep)."""
    from tre_common.registry import VLLM_FEATURE_FLAGS

    sidecar = _sidecar_module()
    cfg = sidecar.Config()
    assert cfg.exclude_header == "x-tre-exclude-pod"
    assert cfg.continued_header == "x-tre-continued"
    assert cfg.sleeping_error_type == "EngineSleeping"
    assert (cfg.generated_ids_field, cfg.prompt_ids_field, cfg.token_ids_field) == (
        "generated_token_ids", "prompt_token_ids", "token_ids",
    )
    assert VLLM_FEATURE_FLAGS == {
        "sleep_reject_new": ("--sleep-reject-new",),
        "abort_return_token_ids": ("--abort-return-token-ids",),
    }
    # the SM's registry-driven runtime creates use the same default gateway as the sidecar
    from tre_common.registry import DEFAULT_REISSUE_GATEWAY_URL

    assert sidecar.DEFAULT_GATEWAY_URL == DEFAULT_REISSUE_GATEWAY_URL
    assert ".svc.cluster.local" in DEFAULT_REISSUE_GATEWAY_URL


@needs_go
def test_sidecar_headers_and_continuability_match_the_gateway_plugin() -> None:
    sidecar = _sidecar_module()
    go = _go_string_consts(TRE_GO)
    assert go["HeaderTREExcludePod"] == sidecar.Config().exclude_header
    # The plugin marks non_continuable requests (the SM drains them); the sidecar must
    # never try to continue one of them: every request field the plugin inspects is
    # also inspected by the sidecar's classification.
    source = inspect.getsource(sidecar.non_continuable_reason)
    for tag in _go_struct_json_tags(TRE_GO, "treSamplingFields"):
        assert f'"{tag}"' in source, tag
    # the plugin's EngineSleeping fixture is the fork's error body the sidecar retries
    tests = (GATEWAY_GO / "tre_transparent_sleep_test.go").read_text(encoding="utf-8")
    assert '"type":"EngineSleeping"' in tests

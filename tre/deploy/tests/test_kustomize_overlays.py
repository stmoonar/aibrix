from __future__ import annotations

from pathlib import Path

import yaml


DEPLOY_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    return docs[0]


def test_tre_v2_overlay_declares_components_and_independent_redis() -> None:
    overlay = DEPLOY_ROOT / "overlays" / "tre-v2"
    kustomization = _load_yaml(overlay / "kustomization.yaml")

    assert kustomization["resources"] == [
        "namespace.yaml",
        "service-account.yaml",
        "rbac.yaml",
        "redis.yaml",
        "service-manager.yaml",
        "controller.yaml",
        "params.yaml",
        "ui.yaml",
        "gpu-truth.yaml",
        "gateway.yaml",
        "gateway-plugins.yaml",
    ]

    redis = _load_yaml(overlay / "redis.yaml")
    assert any(item["kind"] == "Service" and item["metadata"]["name"] == "tre-v2-redis" for item in redis["items"])
    assert any(item["kind"] == "Deployment" and item["metadata"]["name"] == "tre-v2-redis" for item in redis["items"])
    rbac_docs = list(yaml.safe_load_all((overlay / "rbac.yaml").read_text(encoding="utf-8")))
    assert any(
        item["kind"] == "Role"
        and item["metadata"]["name"] == "tre-v2-model-manager"
        and item["metadata"]["namespace"] == "default"
        for item in rbac_docs
    )
    assert any(
        item["kind"] == "RoleBinding"
        and item["metadata"]["name"] == "tre-v2-model-manager"
        and item["metadata"]["namespace"] == "default"
        for item in rbac_docs
    )
    assert any(
        item["kind"] == "Role"
        and item["metadata"]["name"] == "tre-v2-model-route-manager"
        and item["metadata"]["namespace"] == "aibrix-system"
        and any(
            rule["apiGroups"] == ["gateway.networking.k8s.io"]
            and rule["resources"] == ["httproutes"]
            and rule["verbs"] == ["get", "list", "watch", "create", "update", "patch"]
            for rule in item["rules"]
        )
        for item in rbac_docs
    )
    assert any(
        item["kind"] == "RoleBinding"
        and item["metadata"]["name"] == "tre-v2-model-route-manager"
        and item["metadata"]["namespace"] == "aibrix-system"
        and item["subjects"] == [
            {"kind": "ServiceAccount", "name": "tre-v2-service-manager", "namespace": "tre-v2"}
        ]
        for item in rbac_docs
    )

    controller = _load_yaml(overlay / "controller.yaml")
    sm = _load_yaml(overlay / "service-manager.yaml")
    ui = _load_yaml(overlay / "ui.yaml")

    assert _image(controller) == "tre-v2-controller:20260709-aa528998"
    assert _image(sm) == "tre-v2-service-manager:20260707-07717371"
    assert _image(ui) == "tre-v2-ui:20260707-07717371"
    assert "latest" not in "\n".join([_image(controller), _image(sm), _image(ui)]).lower()

    assert _env(controller)["TRE_REDIS_URL"] == "redis://tre-v2-redis:6379/0"
    assert _env(controller)["TRE_SERVICE_MANAGER_URL"] == "http://tre-v2-service-manager:8000"
    assert _env(sm)["TRE_REDIS_URL"] == "redis://tre-v2-redis:6379/0"
    assert _env(ui)["TRE_SERVICE_MANAGER_URL"] == "http://tre-v2-service-manager:8000"
    # F1/F2/D8/D10 switches must be explicit for reproducible redeploy (F4.0.3).
    assert _env(controller)["TRE_SIGNAL_SOURCE"] == "zm"
    assert _env(controller)["TRE_PERCENTILE_MODE"] == "bucket_upper"
    assert _env(controller)["TRE_INCOMPLETE_POLICY"] == "drop_model"
    assert _env(controller)["TRE_HIST_BASELINE_LOOKBACK_MS"] == "90000"
    assert _env(controller)["TRE_PAPER_STALE_MAX_WINDOWS"] == "3"
    assert _env(controller)["TRE_METRICS_SCHEMA"] == "v1"
    assert _env(controller)["ENABLE_TRE_SCALING"] == "true"
    assert _env(sm)["TRE_ROUTE_NAMESPACE"] == "tre-v2"
    assert _env(sm)["TRE_GATEWAY_NAME"] == "tre-aibrix-eg"
    assert _env(controller)["TRE_METRICS_REDIS_URL"] == "redis://tre-v2-redis:6379/0"
    assert _env(sm)["TRE_CREATE_MAX_USED_MIB"] == "2500"
    assert _env(sm)["TRE_SLEEP_LEAK_USED_MIB"] == "8192"
    assert _node_selector(controller) == {"kubernetes.io/hostname": "nscc-ds-4a100-node10"}
    assert _node_selector(sm) == {"kubernetes.io/hostname": "nscc-ds-4a100-node10"}
    assert _node_selector(ui) == {"kubernetes.io/hostname": "nscc-ds-4a100-node10"}

    # P0-4A: per-model params load from a mounted ConfigMap so the console can edit +
    # restart-to-apply. W stays frozen as an explicit env value (not in the CM).
    assert _env(controller)["TRE_REGISTRY_PATH"] == "/etc/tre/registry.yaml"
    assert _env(controller)["TRE_METRICS_WINDOW_MS"] == "30000"  # W freeze artifact (explicit env lock)
    assert controller["spec"]["strategy"] == {"type": "Recreate"}  # no dual-controller actuation
    mounts = controller["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
    assert {"name": "registry", "mountPath": "/etc/tre", "readOnly": True} in mounts
    volumes = controller["spec"]["template"]["spec"]["volumes"]
    assert any(v["name"] == "registry" and v["configMap"]["name"] == "tre-v2-registry" for v in volumes)

    params = _load_yaml(overlay / "params.yaml")
    assert params["kind"] == "ConfigMap" and params["metadata"]["name"] == "tre-v2-registry"
    assert params["metadata"]["namespace"] == "tre-v2"
    assert "registry.yaml" in params["data"]
    assert yaml.safe_load(params["data"]["registry.yaml"]) == _load_yaml(DEPLOY_ROOT / "registry.yaml")

    # UI param-edit RBAC: namespace-scoped Role, resourceName-bound, no cluster scope.
    ui_role = next(d for d in rbac_docs if d["kind"] == "Role" and d["metadata"]["name"] == "tre-v2-ui-params")
    assert ui_role["metadata"]["namespace"] == "tre-v2"
    cm_rule = next(r for r in ui_role["rules"] if r["resources"] == ["configmaps"])
    assert cm_rule["resourceNames"] == ["tre-v2-registry"]
    assert sorted(cm_rule["verbs"]) == ["get", "patch", "update"]
    dep_rule = next(r for r in ui_role["rules"] if r["resources"] == ["deployments"])
    assert dep_rule["resourceNames"] == ["tre-v2-controller"]
    assert sorted(dep_rule["verbs"]) == ["get", "patch"]
    ui_binding = next(d for d in rbac_docs if d["kind"] == "RoleBinding" and d["metadata"]["name"] == "tre-v2-ui-params")
    assert ui_binding["subjects"] == [{"kind": "ServiceAccount", "name": "tre-v2-ui", "namespace": "tre-v2"}]


def test_ablation_overlays_patch_only_controller_env() -> None:
    expected = {
        "ablation-no-fastloop": ("TRE_ABLATION_DISABLE_FAST_LOOP", "true"),
        "ablation-no-safescale": ("TRE_ABLATION_DISABLE_SAFESCALE", "true"),
        "ablation-bucket-upper": ("TRE_PERCENTILE_MODE", "bucket_upper"),
        "ablation-interpolated": ("TRE_PERCENTILE_MODE", "interpolated"),
    }

    for overlay_name, (env_name, value) in expected.items():
        overlay = DEPLOY_ROOT / "overlays" / overlay_name
        kustomization = _load_yaml(overlay / "kustomization.yaml")
        patch = _load_yaml(overlay / "patch-controller-env.yaml")

        assert kustomization["resources"] == ["../tre-v2"]
        assert kustomization["patches"] == [{"path": "patch-controller-env.yaml"}]
        assert patch["kind"] == "Deployment"
        assert patch["metadata"]["name"] == "tre-v2-controller"
        assert _env(patch)[env_name] == value


def _image(deployment: dict) -> str:
    return deployment["spec"]["template"]["spec"]["containers"][0]["image"]


def _env(deployment: dict) -> dict[str, str]:
    env = deployment["spec"]["template"]["spec"]["containers"][0].get("env", [])
    return {item["name"]: item["value"] for item in env}


def _node_selector(deployment: dict) -> dict[str, str]:
    return deployment["spec"]["template"]["spec"].get("nodeSelector", {})

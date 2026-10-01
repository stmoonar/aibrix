from __future__ import annotations

from pathlib import Path

import yaml

from tre_common.registry import load_registry


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
        "gateway-extproc.yaml",
        "gateway-stats.yaml",
        "gateway-service.yaml",
        "gateway-service-params.yaml",
    ]

    redis = _load_yaml(overlay / "redis.yaml")
    assert any(item["kind"] == "Service" and item["metadata"]["name"] == "tre-v2-redis" for item in redis["items"])
    assert any(item["kind"] == "Deployment" and item["metadata"]["name"] == "tre-v2-redis" for item in redis["items"])
    rbac_docs = list(yaml.safe_load_all((overlay / "rbac.yaml").read_text(encoding="utf-8")))
    assert any(
        item["kind"] == "Role"
        and item["metadata"]["name"] == "tre-v2-model-manager"
        and item["metadata"]["namespace"] == "default"
        and any(
            "deployments/scale" in rule["resources"]
            and "patch" in rule["verbs"]
            for rule in item["rules"]
        )
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
        and item["metadata"]["namespace"] == "tre-v2"
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
        and item["metadata"]["namespace"] == "tre-v2"
        and item["subjects"] == [
            {"kind": "ServiceAccount", "name": "tre-v2-service-manager", "namespace": "tre-v2"}
        ]
        for item in rbac_docs
    )

    controller = _load_yaml(overlay / "controller.yaml")
    sm = _load_yaml(overlay / "service-manager.yaml")
    ui = _load_yaml(overlay / "ui.yaml")

    assert _image(controller) == "tre-v2-controller:20261001-8b9d5980"
    assert _image(sm) == "tre-v2-service-manager:20261001-8b9d5980"
    sm_container = sm["spec"]["template"]["spec"]["containers"][0]
    # Review P1-2: single writer across rollouts, and a grace period derived from
    # the registry sleep policy (a sleep past /sleep finishes; drains roll back).
    assert sm["spec"]["strategy"] == {"type": "Recreate"}
    sm_config = load_registry(str(DEPLOY_ROOT / "registry.yaml")).service_manager()
    grace = sm["spec"]["template"]["spec"]["terminationGracePeriodSeconds"]
    # Review 2 P2-2: the SIGTERM wait is computed from the same time budget.
    assert grace > sm_config.shutdown_timeout_s()
    assert sm_container["readinessProbe"]["httpGet"] == {
        "path": "/healthz",
        "port": "http",
    }
    assert _image(ui) == "tre-v2-ui:20261001-8b9d5980"
    assert "latest" not in "\n".join([_image(controller), _image(sm), _image(ui)]).lower()
    gateway_plugins = next(
        d
        for d in yaml.safe_load_all((overlay / "gateway-plugins.yaml").read_text(encoding="utf-8"))
        if d and d["kind"] == "Deployment"
    )
    # Rebuilt by deploy/scripts/build_gateway_plugins_nozmq.sh (TRE-PATCH P2-GW-004/005).
    assert _image(gateway_plugins) == "aibrix/gateway-plugins:20261001-8b9d5980-nozmq2"

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
    assert _env(controller)["TRE_METRICS_SCHEMA"] == "v2"  # D7 (09-22): gateway zsets, no legacy SCAN
    # D8 (09-22): phase-aligned 10 s sampler, pinned explicitly. Band dwell OFF (v1/paper
    # alignment A5), also pinned: "1" = act on the first window.
    assert _env(controller)["TRE_METRICS_REFRESH_MODE"] == "phase_aligned"
    assert _env(controller)["TRE_METRICS_PHASE_OFFSET_MS"] == "2000"
    assert _env(controller)["TRE_DWELL_WINDOWS"] == "1"
    # A2 (v1/paper alignment): receiver-less HIGH proactive SafeScale shrink live.
    assert _env(controller)["TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE"] == "0"
    # A6: SafeScale probe window W = max(multiplier * p95_e2e, floor), pinned. The floor
    # lives under the new env name; the legacy name stays at 60000 for rollback (P2-7):
    # controller images before 2026-09-29 reject a legacy floor below 60 s at startup.
    assert _env(controller)["SAFE_SCALE_WINDOW_FLOOR_MS"] == "20000"
    assert _env(controller)["SAFE_SCALE_MIN_WINDOW_MS"] == "60000"
    assert _env(controller)["SAFE_SCALE_E2E_MULTIPLIER"] == "2"
    # P2-6: donor hold after an SM floor_violation, in fast-loop ticks.
    assert _env(controller)["TRE_FLOOR_VIOLATION_COOLDOWN_TICKS"] == "6"
    # 2026-09-29: SafeScale thresholds come from registry safescale.slo_mode; the env
    # values are optional overrides and must not be pinned by the overlay.
    assert "SAFE_SCALE_TTFT_P95_SLO_MS" not in _env(controller)
    assert "SAFE_SCALE_TPOT_P95_SLO_MS" not in _env(controller)
    assert "SAFE_SCALE_MAX_WINDOW_MS" not in _env(controller)
    assert "SAFE_SCALE_CW2_FALLBACK_MS" not in _env(controller)
    # A12 / A13: KV-cache commit ceiling, donor-health guard source + thresholds, backoff.
    assert _env(controller)["SAFE_SCALE_KV_CACHE_MAX"] == "0.8"
    assert _env(controller)["TRE_GATEWAY_STATS_URL"] == (
        "http://tre-v2-envoy-stats.envoy-gateway-system.svc.cluster.local:19001/stats/prometheus"
    )
    assert _env(controller)["TRE_GATEWAY_ROUTE_NAMESPACE"] == _env(sm)["TRE_ROUTE_NAMESPACE"] == "tre-v2"
    assert _env(controller)["TRE_SAFESCALE_DONOR_ERROR_RATE_MAX"] == "0.01"
    assert _env(controller)["TRE_SAFESCALE_DONOR_MIN_REQUESTS"] == "20"
    assert _env(controller)["TRE_SAFESCALE_ROLLBACK_BACKOFF_MS"] == "60000"
    stats = _load_yaml(overlay / "gateway-stats.yaml")
    assert (stats["kind"], stats["metadata"]["name"], stats["metadata"]["namespace"]) == (
        "Service",
        "tre-v2-envoy-stats",
        "envoy-gateway-system",
    )
    # Selects only the tre-v2 Gateway's proxy, never the shared aibrix-system one.
    assert stats["spec"]["selector"]["gateway.envoyproxy.io/owning-gateway-namespace"] == "tre-v2"
    assert stats["spec"]["selector"]["gateway.envoyproxy.io/owning-gateway-name"] == "tre-aibrix-eg"
    assert stats["spec"]["ports"] == [{"name": "metrics", "port": 19001, "targetPort": 19001, "protocol": "TCP"}]
    assert _env(controller)["ENABLE_TRE_SCALING"] == "true"
    assert _env(sm)["TRE_ROUTE_NAMESPACE"] == "tre-v2"
    assert _env(sm)["TRE_GATEWAY_NAME"] == "tre-aibrix-eg"
    assert _env(controller)["TRE_METRICS_REDIS_URL"] == "redis://tre-v2-redis:6379/0"
    # B9: the cold-start limit is derived from the registry; the env var is an
    # explicit override only and must not be baked into the overlay.
    assert "TRE_CREATE_MAX_USED_MIB" not in _env(sm)
    assert _env(sm)["TRE_SLEEP_LEAK_USED_MIB"] == "8192"
    assert _env(sm)["TRE_SM_SUPERVISOR_ENABLED"] == "true"
    assert _env(sm)["TRE_SM_SUPERVISOR_INTERVAL_S"] == "5"
    assert _env(sm)["TRE_SM_LOG_LEVEL"] == "INFO"
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

    # Plan 2026-09-27 D7: the service-manager seeds desired state from the SAME live
    # registry (and reads its service_manager: sleep policy) - not a baked copy.
    assert _env(sm)["TRE_REGISTRY_PATH"] == "/etc/tre/registry.yaml"
    sm_mounts = sm["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
    assert {"name": "registry", "mountPath": "/etc/tre", "readOnly": True} in sm_mounts
    sm_volumes = sm["spec"]["template"]["spec"]["volumes"]
    assert any(v["name"] == "registry" and v["configMap"]["name"] == "tre-v2-registry" for v in sm_volumes)

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
    # restart-to-apply rolls the controller AND the service-manager (P3, 2026-09-28)
    assert dep_rule["resourceNames"] == ["tre-v2-controller", "tre-v2-service-manager"]
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

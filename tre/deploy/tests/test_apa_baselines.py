from __future__ import annotations

from pathlib import Path

import yaml

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
APA_DIR = DEPLOY_ROOT / "baselines" / "apa"
MODELS = ("dsqwen-7b", "dsllama-8b", "dsqwen-14b")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _registry_models() -> dict[str, dict]:
    reg = yaml.safe_load((DEPLOY_ROOT / "registry.yaml").read_text(encoding="utf-8"))
    return {m["name"]: m for m in reg["models"]}


def test_apa_podautoscaler_crs_match_seam_and_registry() -> None:
    registry = _registry_models()
    for model in MODELS:
        cr = _load(APA_DIR / f"{model}-apa.yaml")
        assert cr["kind"] == "PodAutoscaler"
        assert cr["apiVersion"] == "autoscaling.aibrix.ai/v1alpha1"
        assert cr["metadata"]["namespace"] == "default"
        assert cr["metadata"]["labels"]["tre.aibrix.io/baseline"] == "apa"

        spec = cr["spec"]
        # APA sleep mode fires only when scalingStrategy == APA (workload_scale.go:334).
        assert spec["scalingStrategy"] == "APA"
        # scaleTargetRef.name is sent verbatim to service-manager as the model name.
        assert spec["scaleTargetRef"]["name"] == model
        assert spec["scaleTargetRef"]["kind"] == "Deployment"
        # KVCache baseline metric.
        src = spec["metricsSources"][0]
        assert src["targetMetric"] == "gpu_cache_usage_perc"
        assert src["metricSourceType"] == "pod"
        # min/max mirror the registry.
        assert spec["minReplicas"] == registry[model]["min_replicas"]
        assert spec["maxReplicas"] == registry[model]["max_replicas"]


def test_apa_scale_anchor_deployments_publish_model_selector() -> None:
    for model in MODELS:
        anchor = _load(APA_DIR / f"{model}-apa-anchor.yaml")
        assert anchor["kind"] == "Deployment"
        assert anchor["metadata"]["name"] == model  # same name the CR targets
        assert anchor["metadata"]["namespace"] == "default"
        # anchor is inert: 0 replicas, never actuated by sleep mode.
        assert anchor["spec"]["replicas"] == 0
        # selector matches all awake pods of the model for gpu_cache_usage_perc scraping.
        assert anchor["spec"]["selector"]["matchLabels"]["model.aibrix.ai/name"] == model


def test_toggle_script_enforces_stop_old_before_start_new() -> None:
    script = (DEPLOY_ROOT / "scripts" / "toggle_tre_apa.sh").read_text(encoding="utf-8")
    for sub in ("tre)", "apa)", "status)"):
        assert sub in script
    # TRE arm: delete APA CRs and verify none remain before enabling TRE.
    assert "delete_apa_crs" in script and "refusing to enable TRE" in script
    # APA arm: disable TRE and verify off before applying APA CRs.
    assert "refusing to apply APA" in script
    assert "ENABLE_TRE_SCALING" in script

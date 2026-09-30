"""Release guard (review P2-6, 2026-09-30): the shipped registry.yaml must stay
readable by the components released before the placement / parallel-wake
branch (images 20260930-f8ccb0ca): their ``parse_placement_config`` refuses any
placement key but ``reserve_tp_pairs`` and ``defrag`` (a UI / controller / SM
still on the old image would not start). The new keys stay commented out until
all three images run this release (deploy/RELEASE-20260930-placement-parallel-wake.md)."""

from pathlib import Path

import yaml

#: What ``parse_placement_config`` accepted before 2026-09-30 (main 19781f50).
LEGACY_PLACEMENT_KEYS = {"reserve_tp_pairs", "defrag"}
LEGACY_DEFRAG_KEYS = {"enabled"}

REGISTRY = Path(__file__).resolve().parents[1] / "registry.yaml"


def test_registry_yaml_placement_is_readable_by_the_previous_release():
    raw = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    placement = raw.get("placement") or {}
    assert set(placement) <= LEGACY_PLACEMENT_KEYS, sorted(set(placement) - LEGACY_PLACEMENT_KEYS)
    assert set(placement.get("defrag") or {}) <= LEGACY_DEFRAG_KEYS


def test_new_keys_are_documented_but_commented_out():
    text = REGISTRY.read_text(encoding="utf-8")
    for key in ("# placement_penalty:", "# wake_cooldown:", "# test_hooks: false", "#   max_records: 20000"):
        assert key in text, key


def test_uncommenting_the_new_keys_gives_no_duplicate_blocks():
    """Review P3-1: the new keys are comment lines inside the existing blocks;
    uncommenting them must not create a second ``wake:`` / ``startup_admission:``."""
    text = REGISTRY.read_text(encoding="utf-8")
    for line in ("    # placeholder_max_s: 900", "    # recovery_unknown_attempts: 12", "    # transport_recheck_s: 30"):
        assert text.count(line) == 1, line
        text = text.replace(line, line.replace("# ", "", 1))
    assert text.count("\n  wake:\n") == 1 and text.count("\n  startup_admission:\n") == 1
    raw = yaml.safe_load(text)["service_manager"]
    assert raw["wake"]["recovery_unknown_attempts"] == 12 and raw["wake"]["max_used_fraction"] == 0.2
    assert raw["startup_admission"]["placeholder_max_s"] == 900 and raw["startup_admission"]["gate_seen_s"] == 30


def test_engine_container_name_matches_the_rendered_manifests():
    import tre_sm.ops.k8s_ops as k8s_ops
    from tre_common import bindings
    from tre_sm.ops.k8s_ops import ENGINE_CONTAINER, _engine_running

    # one definition, shared by the manifest generator and the service-manager
    assert k8s_ops.ENGINE_CONTAINER is bindings.ENGINE_CONTAINER
    generator = (Path(__file__).resolve().parents[1] / "gen_model_manifests.py").read_text(encoding="utf-8")
    assert '"vllm-openai"' not in generator and '"name": ENGINE_CONTAINER' in generator

    models = Path(__file__).resolve().parents[1] / "models"
    rendered = sorted(models.glob("*.yaml"))
    assert rendered
    for path in rendered[:3]:
        docs = [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc and doc.get("kind") == "Deployment"]
        for doc in docs:
            assert doc["spec"]["template"]["spec"]["containers"][0]["name"] == ENGINE_CONTAINER
    status = {"containerStatuses": [{"name": "tre-reissue-sidecar", "state": {"running": {}}},
                                    {"name": ENGINE_CONTAINER, "state": {"waiting": {"reason": "CrashLoopBackOff"}}}]}
    assert _engine_running({"status": status, "metadata": {}, "spec": {}}) is False
    status["containerStatuses"][1]["state"] = {"running": {"startedAt": "x"}}
    assert _engine_running({"status": status, "metadata": {}, "spec": {}}) is True
    assert _engine_running({"status": {}, "metadata": {}, "spec": {}}) is None

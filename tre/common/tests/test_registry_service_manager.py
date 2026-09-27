"""registry service_manager: section (plan 2026-09-27) and the shared binding set."""

from pathlib import Path

import pytest
import yaml

from tre_common.bindings import MAX_BOUND_PER_GPU, render_binding_set
from tre_common.registry import (
    DEFAULT_SLEEP_BUDGETS_S,
    SLEEP_PATHS,
    ClusterTopology,
    ModelSpec,
    NodeSpec,
    Registry,
    ServiceManagerConfig,
    SloSpec,
    TrsParams,
    load_registry,
    parse_service_manager_config,
)

REPO_REGISTRY = Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml"


def test_absent_section_means_builtin_defaults():
    config = parse_service_manager_config(None)

    assert config == ServiceManagerConfig()
    assert config.sleep.hard_cap_s == 150.0
    assert config.sleep.budgets_s == DEFAULT_SLEEP_BUDGETS_S
    assert set(DEFAULT_SLEEP_BUDGETS_S) == set(SLEEP_PATHS)
    assert config.sleep.vllm_sleep_mode_param is True


def test_hard_cap_defaults_to_the_gateway_route_timeout():
    config = parse_service_manager_config({}, gateway={"route_timeout_s": 90})
    assert config.sleep.hard_cap_s == 90.0

    explicit = parse_service_manager_config({"sleep": {"hard_cap_s": 120}}, gateway={"route_timeout_s": 90})
    assert explicit.sleep.hard_cap_s == 120.0


def test_overrides_and_soft_budget_resolution():
    config = parse_service_manager_config(
        {
            "sleep": {
                "budgets_s": {"urgent": 12, "scale_down": None},
                "vllm_sleep_mode_param": "false",
                "gateway_plugin_pods": {"namespace": "gw", "label_selector": None},
            },
            "wake": {"max_used_mib": 4096},
            "clock_skew": {"fail_s": 30},
        }
    )
    sleep = config.sleep
    assert sleep.vllm_sleep_mode_param is False
    assert sleep.plugin_namespace == "gw" and sleep.plugin_label_selector is None
    assert sleep.soft_budget_s("urgent") == 12.0
    assert sleep.soft_budget_s("scale_down") == sleep.hard_cap_s  # null = hard cap only
    assert sleep.soft_budget_s("not-a-path") == 30.0  # unknown -> "default"
    assert sleep.soft_budget_s("urgent", 999) == sleep.hard_cap_s  # caller capped
    assert sleep.soft_budget_s("urgent", 0) == 0.0
    with pytest.raises(ValueError):
        sleep.soft_budget_s("urgent", -1)
    assert config.wake_max_used_mib == 4096
    assert config.clock_skew_fail_s == 30.0


def test_unknown_sleep_path_is_rejected():
    with pytest.raises(ValueError, match="unknown sleep path"):
        parse_service_manager_config({"sleep": {"budgets_s": {"fast": 1}}})


def test_validate_flags_bad_service_manager_values():
    config = parse_service_manager_config(
        {"sleep": {"ack_timeout_s": 0, "budgets_s": {"urgent": -3}}, "wake": {"max_used_mib": 0}}
    )
    errors = Registry(ClusterTopology(nodes=()), [], service_manager=config).validate()
    assert "service_manager.sleep.ack_timeout_s must be positive" in errors
    assert "service_manager.sleep.budgets_s.urgent must be >= 0 or null" in errors
    assert "service_manager.wake.max_used_mib must be positive" in errors


def test_repo_registry_parses_and_documents_every_default():
    registry = load_registry(str(REPO_REGISTRY))
    assert registry.validate() == []
    raw = yaml.safe_load(REPO_REGISTRY.read_text(encoding="utf-8"))
    # The shipped file spells out the defaults (documentation), so parsing it and
    # parsing nothing must agree.
    assert registry.service_manager() == parse_service_manager_config(None)
    assert set(raw["service_manager"]["sleep"]["budgets_s"]) == set(SLEEP_PATHS)
    assert len(render_binding_set(registry)) == sum(m.max_replicas for m in registry.models())


def _spec(name, tp, max_replicas):
    slo = SloSpec(ttft_p95_ms=1, tpot_p95_ms=1, e2e_p95_ms=1)
    trs = TrsParams(0, 1, 1, 1, 0.5, 0, 0.8, 1, 1.2, 4, 0.05, 3)
    return ModelSpec(name, "/w", tp, 0, max_replicas, "img", slo, trs)


def test_binding_set_is_registry_order_and_enforces_the_per_gpu_budget():
    topology = ClusterTopology(nodes=(NodeSpec("n", 2, ((0, 1),), ("a", "b")),))
    ok = Registry(topology, [_spec("x", 1, 2), _spec("y", 2, 1)])
    assert [s.binding_id for s in render_binding_set(ok)] == ["x/n/0", "x/n/1", "y/n/0,1"]

    crowded = Registry(topology, [_spec(f"m{i}", 1, 1) for i in range(MAX_BOUND_PER_GPU + 1)])
    with pytest.raises(ValueError, match="gpu bound budget exceeded"):
        render_binding_set(crowded)

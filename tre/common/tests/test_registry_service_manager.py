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
    assert config.sleep.vllm_sleep_mode_param == "auto"
    assert config.sleep.fallback_no_plugin is False
    assert config.sleep.gateway_min_instances == 1
    assert config.wake_max_used_mib is None and config.wake_max_used_fraction == 0.2


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
    assert sleep.vllm_sleep_mode_param == "false"
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
    assert "service_manager.wake.max_used_mib must be positive or null" in errors


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


def test_sleep_mode_param_accepts_auto_and_booleans():
    from tre_common.registry import parse_sleep_mode_param

    assert parse_sleep_mode_param("auto") == "auto"
    assert parse_sleep_mode_param(True) == "true"
    assert parse_sleep_mode_param("no") == "false"
    with pytest.raises(ValueError):
        parse_sleep_mode_param("sometimes")


def test_worst_case_sleep_call_must_fit_the_api_call_timeout():
    config = ServiceManagerConfig()
    # review 2 P2-2: every probe timeout counted; commit targets run in parallel,
    # so the bound does not depend on the number of targets.
    commit = 4 * 5 + 2 * 45 + 15 + 5
    assert config.worst_case_commit_s() == commit
    assert config.worst_case_sleep_call_s() == 10 + (10 + 150 + 5) + 10 + commit + 5
    assert config.worst_case_sleep_call_s() < config.api_call_timeout_s
    assert config.shutdown_timeout_s() == 10 + commit + 0.5 + 5 + 5

    slow = parse_service_manager_config({"sleep": {"sleep_call_timeout_s": 90}})
    errors = Registry(ClusterTopology(nodes=()), [], service_manager=slow).validate()
    assert any("worst-case sleeping service-manager call is 410s" in e for e in errors)

    from tre_common.registry import sleep_call_timeout_errors

    assert sleep_call_timeout_errors(config, 360.0) == []
    assert sleep_call_timeout_errors(config, 300.0, name="TRE_SM_SLOW_TIMEOUT_SECONDS")[0].endswith(
        "the caller would time out mid-drain"
    )


def test_reservation_ttl_must_outlive_the_longest_renewal_gap():
    ok = parse_service_manager_config({})
    assert not [e for e in Registry(ClusterTopology(nodes=()), [], service_manager=ok).validate() if "renewal gap" in e]
    short = parse_service_manager_config(
        {"commit_lock_wait_s": 25, "sleep": {"reservation_ttl_s": 30}, "api_call_timeout_s": 1000}
    )
    errors = Registry(ClusterTopology(nodes=()), [], service_manager=short).validate()
    assert any("longest renewal gap 35.5s" in e for e in errors)
    assert short.commit_wait_s == 25 and ok.commit_wait_s == ok.writer_lock_wait_s
    probe = parse_service_manager_config({"sleep": {"probe_timeout_s": 0}})
    assert any("probe_timeout_s must be positive" in e for e in Registry(ClusterTopology(nodes=()), [], service_manager=probe).validate())


def test_hard_cap_may_not_exceed_the_gateway_route_timeout():
    from tre_common.registry import GatewayConfig, parse_gateway_config

    config = parse_service_manager_config({"sleep": {"hard_cap_s": 200}})
    errors = Registry(
        ClusterTopology(nodes=()), [], service_manager=config, gateway=GatewayConfig(150)
    ).validate()
    assert any("must not exceed gateway.route_timeout_s" in e for e in errors)
    assert parse_gateway_config(None).route_timeout_s == 150.0
    assert parse_gateway_config({"route_timeout_s": 60}).route_timeout_s == 60.0


def test_wake_limit_is_relative_with_an_optional_absolute_override():
    relative = parse_service_manager_config({"wake": {"max_used_fraction": 0.25}})
    assert relative.wake_limit_mib(40960) == 10240
    assert relative.wake_limit_mib(None) is None  # total unknown: caller fails closed
    absolute = parse_service_manager_config({"wake": {"max_used_mib": 4096}})
    assert absolute.wake_limit_mib(81920) == 4096 and absolute.wake_limit_mib(None) == 4096
    bad = parse_service_manager_config({"wake": {"max_used_fraction": 1.5}})
    errors = Registry(ClusterTopology(nodes=()), [], service_manager=bad).validate()
    assert "service_manager.wake.max_used_fraction must be in (0, 1]" in errors


def test_max_bound_per_gpu_comes_from_the_registry(tmp_path):
    from tre_common.registry import _parse_registry

    topology = ClusterTopology(nodes=(NodeSpec("n", 1, (), ("a",)),), max_bound_per_gpu=1)
    one = Registry(topology, [_spec("x", 1, 1), _spec("y", 1, 1)])
    with pytest.raises(ValueError, match=r"1 \(cluster\.max_bound_per_gpu\)"):
        render_binding_set(one)
    raw = {"cluster": {"max_bound_per_gpu": 5, "nodes": []}, "models": []}
    assert _parse_registry(raw).topology().max_bound_per_gpu == 5
    assert _parse_registry({"cluster": {"nodes": []}}).topology().max_bound_per_gpu == MAX_BOUND_PER_GPU

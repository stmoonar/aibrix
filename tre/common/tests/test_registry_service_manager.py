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
                # Budget mechanics: every path drains (no-drain paths ignore budgets,
                # service-manager/tests/test_sleep_no_drain.py).
                "no_drain_paths": [],
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
    # "apa" (2026-09-29) is left out on purpose: older images reject unknown paths.
    assert set(raw["service_manager"]["sleep"]["budgets_s"]) == set(SLEEP_PATHS) - {"apa"}
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


def test_worst_case_lock_holds_and_the_call_bound_fit_the_api_call_timeout():
    """Whole-lock (2026-10-02): one sleep holds the writer lock for at most ack 5 +
    one probe round 5 + /sleep 10 + max(confirmation 8 + its last probe 5, the
    failed-call rollback's 4 probes) + 8 sequential Kubernetes calls x (connect 2
    + read 5) + io 2 = 98 s, whatever the target count (probe timeout 5 s and the
    Kubernetes term since 2026-10-06)."""
    config = ServiceManagerConfig()
    assert config.k8s_call_s() == 2 + 5
    assert config.worst_case_sleep_lock_s() == 5 + 5 + 10 + max(8 + 5, 4 * 5) + 8 * 7 + 2 == 98
    # A wake: resident probes 5 + /wake_up 10 + convergence and settlement probes
    # 2 x 5 + ONE compensating sleep 98 (whatever the number of failed wakes) +
    # 4 Kubernetes calls x 7 + io 2.
    assert config.worst_case_wake_lock_s() == 5 + 10 + 2 * 5 + 98 + 4 * 7 + 2 == 153
    assert config.worst_case_transfer_lock_s() == 5 + 2 * 7 + 98 + 153 == 270
    assert config.worst_case_lock_hold_s() == 270
    assert config.worst_case_sleep_call_s() == 30 + 270 + 2 == 302
    assert config.worst_case_sleep_call_s() < config.api_call_timeout_s
    assert config.shutdown_timeout_s() == 270 + 2

    slow = parse_service_manager_config({"sleep": {"sleep_call_timeout_s": 200}})
    errors = Registry(ClusterTopology(nodes=()), [], service_manager=slow).validate()
    assert any("worst-case service-manager call is 682s" in e for e in errors)

    from tre_common.registry import sleep_call_timeout_errors

    assert sleep_call_timeout_errors(config, 360.0) == []
    assert sleep_call_timeout_errors(config, 100.0, name="TRE_SM_SLOW_TIMEOUT_SECONDS")[0].endswith(
        "the caller would time out mid-operation"
    )


def test_the_deprecated_reservation_and_drain_settings_are_no_longer_validated():
    old = parse_service_manager_config(
        {"commit_lock_wait_s": 25, "sleep": {"reservation_ttl_s": 1, "hard_cap_s": 900}, "api_call_timeout_s": 1000}
    )
    errors = Registry(ClusterTopology(nodes=()), [], service_manager=old).validate()
    assert not [e for e in errors if "reservation_ttl_s" in e or "hard_cap_s" in e or "commit_lock" in e]
    probe = parse_service_manager_config({"sleep": {"probe_timeout_s": 0}})
    assert any("probe_timeout_s must be positive" in e for e in Registry(ClusterTopology(nodes=()), [], service_manager=probe).validate())
    idle = parse_service_manager_config({"sleep": {"sleep_mode_when_idle": "keep"}})
    assert any("sleep_mode_when_idle" in e for e in Registry(ClusterTopology(nodes=()), [], service_manager=idle).validate())
    wake = parse_service_manager_config({"wake": {"call_timeout_s": 0}})
    assert any("wake.call_timeout_s" in e for e in Registry(ClusterTopology(nodes=()), [], service_manager=wake).validate())


def test_gateway_route_timeout_still_parses():
    from tre_common.registry import parse_gateway_config

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


def test_create_limit_is_derived_from_the_models_utilization_with_overrides():
    config = parse_service_manager_config(None)
    assert config.create_margin_mib == 512 and config.create_max_used_mib is None
    # vLLM needs 0.85 x 40960 = 34816 MiB free -> used <= 6144 - 512
    assert config.create_limit_mib(40960, 0.85) == 5632
    assert config.create_limit_mib(None, 0.85) is None  # total unknown: caller fails closed
    tuned = parse_service_manager_config({"create": {"margin_mib": 1024}})
    assert tuned.create_limit_mib(40960, 0.85) == 5120
    absolute = parse_service_manager_config({"create": {"max_used_mib": 3000}})
    assert absolute.create_limit_mib(81920, 0.5) == 3000 and absolute.create_limit_mib(None, 0.9) == 3000
    bad = parse_service_manager_config({"create": {"margin_mib": -1, "max_used_mib": 0}})
    errors = Registry(ClusterTopology(nodes=()), [], service_manager=bad).validate()
    assert "service_manager.create.margin_mib must be >= 0" in errors
    assert "service_manager.create.max_used_mib must be positive or null" in errors


def test_gpu_memory_utilization_comes_from_the_engine_args():
    from tre_common.registry import VLLM_DEFAULT_GPU_MEMORY_UTILIZATION, gpu_memory_utilization

    spec = _spec("m", 1, 1)
    assert spec.gpu_memory_utilization == VLLM_DEFAULT_GPU_MEMORY_UTILIZATION == 0.9
    import dataclasses

    spaced = dataclasses.replace(spec, vllm_extra_args=("--gpu-memory-utilization", "0.85"))
    assert spaced.gpu_memory_utilization == 0.85
    joined = dataclasses.replace(spec, vllm_extra_args=("--gpu-memory-utilization=0.7",))
    assert gpu_memory_utilization(joined) == 0.7
    for args in (("--gpu-memory-utilization",), ("--gpu-memory-utilization", "x"),
                 ("--gpu-memory-utilization", "1.5")):
        broken = dataclasses.replace(spec, vllm_extra_args=args)
        errors = Registry(ClusterTopology(nodes=()), [broken]).validate()
        assert any("vllm_extra_args --gpu-memory-utilization" in e for e in errors), args


def test_repo_registry_models_declare_their_utilization():
    registry = load_registry(str(REPO_REGISTRY))
    assert {m.name: m.gpu_memory_utilization for m in registry.models()} == {
        m.name: 0.85 for m in registry.models()
    }

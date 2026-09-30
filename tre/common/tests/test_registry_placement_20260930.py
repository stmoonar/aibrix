"""Registry keys of 2026-09-30: placement.placement_penalty / wake_cooldown
(S3 / S5), service_manager.test_hooks / operations.max_records."""

import pytest

from tre_common.registry import (
    ClusterTopology,
    NodeSpec,
    PlacementConfig,
    Registry,
    ServiceManagerConfig,
    load_registry,
    parse_placement_config,
    parse_service_manager_config,
)


def test_placement_defaults_keep_every_node_equal_and_cool_for_30_and_60_s():
    config = parse_placement_config(None)
    assert config.node_penalty == {}
    assert (config.wake_cooldown_gpu_s, config.wake_cooldown_node_s) == (30.0, 60.0)
    live = load_registry().placement()
    assert live.node_penalty == {} and live.wake_cooldown_gpu_s == 30.0


def test_placement_penalty_and_wake_cooldown_are_parsed_and_validated():
    config = parse_placement_config(
        {"placement_penalty": {"n-b": 0.25}, "wake_cooldown": {"gpu_s": 10, "node_s": 20}}
    )
    assert config.node_penalty == {"n-b": 0.25}
    assert (config.wake_cooldown_gpu_s, config.wake_cooldown_node_s) == (10.0, 20.0)
    for bad in ({"placement_penalty": {"n": -1}}, {"placement_penalty": {"n": True}},
                {"placement_penalty": ["n"]}, {"wake_cooldown": {"gpu_s": -1}},
                {"wake_cooldown": {"gpu": 1}}):
        with pytest.raises(ValueError, match="placement"):
            parse_placement_config(bad)


def test_placement_penalty_names_must_be_registry_nodes():
    topology = ClusterTopology(nodes=(NodeSpec("n-a", 4, ((0, 1), (2, 3)), ("a", "b", "c", "d")),))
    registry = Registry(topology, [], placement=PlacementConfig(node_penalty={"elsewhere": 0.5}))
    assert any("unknown node 'elsewhere'" in error for error in registry.validate())


def test_service_manager_test_hooks_off_and_operations_capped_by_default():
    config = parse_service_manager_config({})
    assert config.test_hooks is False
    assert config.operations_max_records == 20000
    config = parse_service_manager_config({"test_hooks": True, "operations": {"max_records": 500}})
    assert config.test_hooks is True and config.operations_max_records == 500
    tiny = Registry(ClusterTopology(nodes=()), [], service_manager=ServiceManagerConfig(operations_max_records=5))
    assert any("operations.max_records" in error for error in tiny.validate_service_manager())

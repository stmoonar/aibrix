"""tp_size validation has one source: tre_common.registry.tp_size_error (2026-09-28).

A power of two >= 1, not wider than the widest node, and at most
MAX_SUPPORTED_TP_SIZE (the binding layout: single GPUs or two_gpu_slots). The
registry refuses anything else at load (SM, controller and UI all load through
it) and in validate(); the placement policy, the manifest bindings and the SM
allocator raise instead of silently falling back or placing nothing.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from tre_common.bindings import feasible_slots
from tre_common.gpu_placement import placement_policy_from_registry
from tre_common.registry import (
    MAX_SUPPORTED_TP_SIZE,
    Registry,
    _parse_registry,
    load_registry,
    tp_size_error,
)


@pytest.mark.parametrize("tp", [1, 2])
def test_supported_tp_sizes(tp):
    assert tp_size_error(tp, widest_node_gpus=4) is None


@pytest.mark.parametrize("tp", [0, -2, 3, 6])
def test_non_powers_of_two_are_rejected(tp):
    assert "power of two" in tp_size_error(tp, widest_node_gpus=8)


@pytest.mark.parametrize("tp", [True, "2", 1.5, None])
def test_non_ints_are_rejected(tp):
    assert "must be an int" in tp_size_error(tp)


def test_wider_than_any_node_or_the_binding_layout_is_rejected():
    assert "widest node" in tp_size_error(8, widest_node_gpus=4)
    assert MAX_SUPPORTED_TP_SIZE == 2
    assert "binding layout" in tp_size_error(4, widest_node_gpus=4)
    # the generic buddy rule (placement library) allows a node-wide block
    assert tp_size_error(4, widest_node_gpus=4, max_tp_size=None) is None


def _raw_registry(tp_size):
    raw = yaml.safe_load(open(_registry_path(), encoding="utf-8"))
    raw["models"][0]["tp_size"] = tp_size
    return raw


def _registry_path():
    import tre_common.registry as module
    from pathlib import Path

    return Path(module.__file__).resolve().parents[2] / "deploy" / "registry.yaml"


def test_the_deployed_registry_loads_and_validates():
    registry = load_registry()
    assert all(tp_size_error(model.tp_size) is None for model in registry.models())


@pytest.mark.parametrize("tp_size, match", [(3, "power of two"), (4, "binding layout"), (16, "widest node"),
                                            (1.5, "must be an int"), ("two", "must be an int")])
def test_the_registry_refuses_an_unsupported_tp_size_at_load(tp_size, match):
    with pytest.raises(ValueError, match=match):
        _parse_registry(_raw_registry(tp_size))


def test_validate_reports_an_unsupported_tp_size_of_a_constructed_registry():
    registry = load_registry()
    bad = replace(registry.models()[0], tp_size=3)
    errors = Registry(registry.topology(), [bad]).validate()
    assert any("unsupported tp_size 3 must be a power of two" in error for error in errors)


def test_placement_policy_raises_on_an_unsupported_tp_size():
    registry = load_registry()
    bad = replace(registry.models()[0], tp_size=3)
    with pytest.raises(ValueError, match="placement policy: model .*power of two"):
        placement_policy_from_registry(Registry(registry.topology(), [bad]))
    stub = SimpleNamespace(models=lambda: [SimpleNamespace(name="m", tp_size=6, scale_max_replicas=1)])
    with pytest.raises(ValueError, match="power of two"):
        placement_policy_from_registry(stub)


def test_feasible_slots_raises_instead_of_binding_nothing():
    registry = load_registry()
    bad = replace(registry.models()[0], tp_size=4)
    with pytest.raises(ValueError, match="binding layout"):
        feasible_slots(Registry(registry.topology(), [bad]), bad)

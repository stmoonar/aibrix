"""SM side of the single tp_size rule (tre_common.registry.tp_size_error):
the registry placement policy never silently falls back to plain best-fit for
an unsupported tp_size, and the slot allocator applies the registry rule."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tre_common.registry import Registry, load_registry
from tre_sm.allocator.slots import SlotAllocator
from tre_sm.api.v2 import _registry_placement_policy


def test_a_registry_stub_without_models_has_no_policy():
    assert _registry_placement_policy(SimpleNamespace()) is None


def test_an_unsupported_tp_size_raises_instead_of_falling_back_to_best_fit():
    registry = load_registry()
    assert _registry_placement_policy(registry) is not None
    bad = replace(registry.models()[0], tp_size=3)
    with pytest.raises(ValueError, match="power of two"):
        _registry_placement_policy(Registry(registry.topology(), [bad]))


@pytest.mark.parametrize("tp_size, match", [(3, "power of two"), (0, "power of two"),
                                            (4, "binding layout"), (16, "widest node")])
def test_the_allocator_applies_the_registry_rule(tp_size, match):
    allocator = SlotAllocator(load_registry().topology(), [])
    with pytest.raises(ValueError, match=match):
        allocator.find_slot(tp_size)
    with pytest.raises(ValueError, match=match):
        allocator.plan_defrag(tp_size)


def test_the_allocator_still_places_supported_sizes():
    allocator = SlotAllocator(load_registry().topology(), [])
    assert allocator.find_slot(1) is not None
    assert allocator.find_slot(2) is not None

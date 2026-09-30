"""Service-manager side of the placement policy: wake pick through the registry
policy, and the defrag gate (registry placement.defrag.enabled, manual force).
Synthetic names only."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tre_common.gpu_placement import PlacementPolicy
from tre_common.registry import ClusterTopology, NodeSpec, PlacementConfig, Registry
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.api.v2 import DefragDisabled, ServiceManagerV2, create_app
from tre_sm.state.store import StateStore

from test_v2_defrag import FakeRedis, registry as defrag_registry

N1, N2 = "n1", "n2"
TOPOLOGY = ClusterTopology(
    nodes=tuple(NodeSpec(name=name, gpus=4, two_gpu_slots=((0, 1), (2, 3))) for name in (N1, N2))
)


class PolicyRegistry:
    """Registry stub with models: A (tp1) and C (tp2)."""

    def __init__(self):
        self._models = [
            SimpleNamespace(name="A", tp_size=1, max_replicas=8, max_awake_replicas=8, scale_max_replicas=8),
            SimpleNamespace(name="C", tp_size=2, max_replicas=4, max_awake_replicas=4, scale_max_replicas=4),
        ]

    def models(self):
        return list(self._models)

    def model(self, name):
        return next(model for model in self._models if model.name == name)

    def topology(self):
        return TOPOLOGY

    def placement(self):
        return PlacementConfig()


def _service(registry, bindings):
    store = StateStore(FakeRedis())
    store.save(bindings, expected_version=0)
    return ServiceManagerV2(registry, store), store


def test_sm_uses_the_registry_placement_policy():
    service, _ = _service(PolicyRegistry(), [])
    assert service._placement == PlacementPolicy(
        max_order=1, reserve_blocks=1, reserve_caps=(("C", 4),)
    )


def test_wake_pick_spreads_a_model_across_nodes():
    bindings = [
        Binding(f"a-{node}-{gpu}", "A", Slot(node, (gpu,)), awake=(node, gpu) == (N1, 0))
        for node in (N1, N2)
        for gpu in range(4)
    ]
    service, store = _service(PolicyRegistry(), bindings)

    result = service.put_model_target("A", wake_replicas=3)

    # S5: the half-used pair on n1 first (split cost), then the lighter node n2.
    assert result["actions"] == [
        {"action": "wake", "serve_id": "a-n1-1"},
        {"action": "wake", "serve_id": "a-n2-0"},
    ]
    awake = {b.slot.node for b in store.load().bindings if b.awake}
    assert awake == {N1, N2}


def _fragmented():
    return [
        Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
        Binding("serve-b", "m1", Slot("node-a", (1,)), awake=True),
        Binding("serve-c", "m1", Slot("node-a", (2,)), awake=True),
        Binding("serve-d", "m1", Slot("node-a", (3,)), awake=True),
    ]


def test_manual_defrag_refuses_while_disabled_unless_forced():
    service, store = _service(defrag_registry(), _fragmented())
    assert service.defrag_enabled() is False
    with pytest.raises(DefragDisabled):
        service.defrag(tp_size=2)
    assert store.load().bindings == _fragmented()

    client = TestClient(create_app(service))
    refused = client.post("/v2/defrag", json={"tp_size": 2})
    assert refused.status_code == 409
    assert refused.json()["detail"]["reason"] == "defrag_disabled"
    assert "force" in refused.json()["detail"]["message"]

    # force passes the gate (this layout then has no feasible plan).
    forced = client.post("/v2/defrag", json={"tp_size": 2, "force": True})
    assert forced.status_code == 409
    assert forced.json() == {"detail": {"reason": "no_feasible_defrag"}}
    assert store.load().bindings == _fragmented()


def test_manual_defrag_needs_no_force_when_enabled():
    base = defrag_registry()
    enabled = Registry(base.topology(), base.models(), placement=PlacementConfig(defrag_enabled=True))
    service, _ = _service(enabled, _fragmented())
    assert service.defrag_enabled() is True
    response = TestClient(create_app(service)).post("/v2/defrag", json={"tp_size": 2})
    assert response.json() == {"detail": {"reason": "no_feasible_defrag"}}

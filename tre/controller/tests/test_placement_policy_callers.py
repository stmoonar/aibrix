"""Controller callers of the registry placement policy (design
tre/docs/design/20260928-placement-node-balance.md). Synthetic names only."""
from __future__ import annotations

import pytest

from tre_common.gpu_placement import PlacementPolicy, placement_policy_from_registry
from tre_common.metrics_schema import MetricsSnapshot
from tre_common.registry import ClusterTopology, NodeSpec, PlacementConfig, Registry
from tre_controller.loops import tick as tick_module
from tre_controller.loops.tick import _pods_to_probe, run_planner_tick
from tre_controller.planning.planner import ClusterView, _release_pick, _SlotOccupancy
from tre_sm.allocator.slots import Binding, Slot, SlotAllocator

N1, N2 = "n1", "n2"
TOPOLOGY = ClusterTopology(
    nodes=tuple(NodeSpec(name=name, gpus=4, two_gpu_slots=((0, 1), (2, 3))) for name in (N1, N2))
)
POLICY = PlacementPolicy(max_order=1, reserve_blocks=1)


def _view(bindings, placement=POLICY) -> ClusterView:
    return ClusterView(topology=TOPOLOGY, bindings=tuple(bindings), placement=placement)


def _b(serve_id, model, node, gpus, *, awake):
    return Binding(serve_id, model, Slot(node, tuple(gpus)), awake=awake)


def _a_everywhere(awake_at=(N1, 0)):
    return [
        _b(f"a-{node}-{gpu}", "A", node, (gpu,), awake=(node, gpu) == awake_at)
        for node in (N1, N2)
        for gpu in range(4)
    ]


def test_plan_wakes_spread_a_model_across_nodes():
    # S5 (2026-09-30): the split cost ranks first - the free half of n1's used pair
    # is filled before a whole pair is broken; then the lighter node n2.
    picks = _SlotOccupancy(_view(_a_everywhere())).plan_wakes("A", 2)
    assert [b.serve_id for b in picks] == ["a-n1-1", "a-n2-0"]
    # Reference best-fit (no policy) packs onto node n1.
    packed = _SlotOccupancy(_view(_a_everywhere(), placement=None)).plan_wakes("A", 2)
    assert [b.serve_id for b in packed] == ["a-n1-1", "a-n1-2"]


def test_free_groups_and_find_slot_spread_a_model():
    # A half-used pair is filled first (split cost, S5) ...
    bindings = [_b("a-0", "A", N1, (0,), awake=True)]
    occupancy = _SlotOccupancy(_view(bindings))
    assert occupancy.free_groups(1, "A")[0] == {(N1, 1)}
    assert SlotAllocator(TOPOLOGY, bindings, policy=POLICY).find_slot(1, "A") == Slot(N1, (1,))
    # ... without one, the lighter node (spread) wins over the lowest address.
    bindings = [_b("a-0", "A", N1, (0,), awake=True), _b("x-1", "X", N1, (1,), awake=True)]
    occupancy = _SlotOccupancy(_view(bindings))
    assert occupancy.free_groups(1, "A")[0] == {(N2, 0)}
    assert SlotAllocator(TOPOLOGY, bindings, policy=POLICY).find_slot(1, "A") == Slot(N2, (0,))
    assert SlotAllocator(TOPOLOGY, bindings).find_slot(1, "A") == Slot(N1, (2,))


def test_relay_pairing_is_counted_not_placed():
    """2026-10-02: which donor pod pairs with which receiver binding is the SM's choice
    (its ``_release_pick`` / ``_wake_pick``); the planner only counts the pairs."""
    bindings = [
        _b("r-0", "R", N1, (0,), awake=True),
        _b("d-n1", "D", N1, (1,), awake=True),
        _b("d-n2", "D", N2, (0,), awake=True),
        _b("r-n1", "R", N1, (1,), awake=False),
        _b("r-n2", "R", N2, (0,), awake=False),
    ]
    for placement in (POLICY, None):
        occupancy = _SlotOccupancy(_view(bindings, placement=placement))
        assert occupancy.pairable_count("D", "R", max_pairs=5, max_donors=1) == (1, 1)
        assert occupancy.pairable_count("D", "R", max_pairs=5, max_donors=5) == (1, 1)  # one left
        assert occupancy.pairable_count("D", "R", max_pairs=5, max_donors=5) == (0, 0)  # counted once
    assert not hasattr(_SlotOccupancy, "donor_slot_pods")


def test_safescale_probe_pick_releases_from_the_most_loaded_node():
    bindings = [
        _b("a-light", "A", N2, (2,), awake=True),
        _b("y-3", "Y", N2, (3,), awake=True),
        _b("a-heavy", "A", N1, (0,), awake=True),
        _b("x-1", "X", N1, (1,), awake=True),
        _b("x-2", "X", N1, (2,), awake=True),
        _b("x-3", "X", N1, (3,), awake=True),
    ]
    snapshot = MetricsSnapshot(ts_ms=1, stale=False, models={})
    assert _pods_to_probe(snapshot, "A", 1, cluster_view=_view(bindings)) == ("a-heavy",)
    # Reference best-fit breaks the tie by the highest address instead.
    assert _pods_to_probe(snapshot, "A", 1, cluster_view=_view(bindings, placement=None)) == ("a-light",)


def test_same_slot_shrink_tie_break_uses_the_policy_not_serve_id():
    bindings = [
        _b("x-1", "X", N2, (2,), awake=True),
        _b("z-3", "Z", N2, (3,), awake=True),
        _b("x-2", "X", N1, (0,), awake=True),
        _b("w-1", "W", N1, (1,), awake=True),
        _b("y-1", "Y", N1, (2,), awake=True),
        _b("y-2", "Y", N1, (3,), awake=True),
    ]
    tied = [bindings[0], bindings[2]]
    # Neither release frees a pair; n1 (2 of 2 pairs in use) is the more loaded node.
    # The old serve_id tie-break would have picked x-1.
    assert _release_pick(tied, _view(bindings)).serve_id == "x-2"


class _Captured(Exception):
    pass


def test_planner_tick_attaches_the_registry_policy_and_defrag_gate(monkeypatch):
    registry = Registry(TOPOLOGY, [], placement=PlacementConfig(reserve_tp_pairs=2, defrag_enabled=False))
    seen = {}

    def fake_build_plan(**kwargs):
        seen.update(kwargs)
        raise _Captured()

    monkeypatch.setattr(tick_module, "build_plan", fake_build_plan)
    with pytest.raises(_Captured):
        run_planner_tick(
            MetricsSnapshot(ts_ms=1, stale=False, models={}),
            queue=type("Q", (), {"inflight_models": lambda self: set(), "submit": lambda self, a: None})(),
            registry=registry,
            rescue_due=True,
            fairness_due=True,
            cluster_view=ClusterView(topology=TOPOLOGY, bindings=()),
        )
    assert seen["cluster_view"].placement == placement_policy_from_registry(registry)
    assert seen["cluster_view"].placement.reserve_blocks == 2
    assert seen["cfg"].defrag_enabled is False

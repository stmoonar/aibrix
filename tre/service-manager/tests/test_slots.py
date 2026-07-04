import random

from tre_common.registry import ClusterTopology, NodeSpec
from tre_sm.allocator.slots import Binding, Migration, Slot, SlotAllocator


def single_node_topology():
    return ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),))


def two_node_topology():
    return ClusterTopology(
        nodes=(
            NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),
            NodeSpec(name="node-b", gpus=4, two_gpu_slots=((0, 1), (2, 3))),
        )
    )


def two_half_node_topology():
    return ClusterTopology(
        nodes=(
            NodeSpec(name="node-a", gpus=2, two_gpu_slots=((0, 1),)),
            NodeSpec(name="node-b", gpus=2, two_gpu_slots=((0, 1),)),
        )
    )


def test_one_gpu_allocation_fills_split_slot_before_opening_new_two_gpu_slot():
    allocator = SlotAllocator(single_node_topology(), [])

    first = allocator.find_slot(1)
    assert first == Slot(node="node-a", gpu_ids=(0,))
    allocator.bind("serve-a", "model-a", first)

    second = allocator.find_slot(1)

    assert second == Slot(node="node-a", gpu_ids=(1,))


def test_defrag_plans_minimal_migration_when_two_gpu_slot_is_fragmented():
    allocator = SlotAllocator(
        single_node_topology(),
        [
            Binding(serve_id="serve-0", model="m1", slot=Slot("node-a", (0,)), awake=True),
            Binding(serve_id="serve-2", model="m1", slot=Slot("node-a", (2,)), awake=True),
        ],
    )

    assert allocator.find_slot(2) is None

    plan = allocator.plan_defrag(2)

    assert plan == [
        Migration(
            serve_id="serve-2",
            from_slot=Slot("node-a", (2,)),
            to_slot=Slot("node-a", (1,)),
        )
    ]


def test_defrag_can_consolidate_free_halves_across_nodes():
    allocator = SlotAllocator(
        two_half_node_topology(),
        [
            Binding(serve_id="serve-a", model="m1", slot=Slot("node-a", (0,)), awake=True),
            Binding(serve_id="serve-b", model="m1", slot=Slot("node-b", (0,)), awake=True),
        ],
    )

    assert allocator.find_slot(2) is None

    plan = allocator.plan_defrag(2)

    assert plan == [
        Migration(
            serve_id="serve-b",
            from_slot=Slot("node-b", (0,)),
            to_slot=Slot("node-a", (1,)),
        )
    ]


def test_sleeping_bindings_may_share_gpu_but_only_one_awake_binding_can_wake():
    allocator = SlotAllocator(
        single_node_topology(),
        [
            Binding(serve_id="awake", model="m1", slot=Slot("node-a", (0,)), awake=True),
            Binding(serve_id="sleeping", model="m2", slot=Slot("node-a", (0,)), awake=False),
        ],
    )

    assert allocator.feasible_wake("awake") is True
    assert allocator.feasible_wake("sleeping") is False


def test_bind_rejects_second_awake_binding_on_same_gpu_but_allows_sleeping_bound_peer():
    allocator = SlotAllocator(single_node_topology(), [])
    allocator.bind("awake", "m1", Slot("node-a", (0,)), awake=True)

    allocator.bind("sleeping", "m2", Slot("node-a", (0,)), awake=False)

    try:
        allocator.bind("second-awake", "m3", Slot("node-a", (0,)), awake=True)
    except ValueError as exc:
        assert "already has awake binding" in str(exc)
    else:
        raise AssertionError("expected awake conflict")


def test_random_allocation_release_sequences_remain_disjoint_and_defraggable():
    topology = two_node_topology()
    allocator = SlotAllocator(topology, [])
    rng = random.Random(20260704)
    live: set[str] = set()
    next_serve_id = 0

    for _ in range(1000):
        if live and rng.random() < 0.35:
            serve_id = rng.choice(sorted(live))
            allocator.release(serve_id)
            live.remove(serve_id)
        else:
            tp_size = 2 if rng.random() < 0.3 else 1
            slot = allocator.find_slot(tp_size)
            if slot is not None:
                serve_id = f"serve-{next_serve_id}"
                next_serve_id += 1
                allocator.bind(serve_id, f"m{tp_size}", slot)
                live.add(serve_id)
            elif _free_gpu_count(topology, allocator.snapshot()) >= tp_size:
                assert allocator.plan_defrag(tp_size) is not None

        _assert_no_overlapping_bindings(allocator.snapshot())


def _free_gpu_count(topology: ClusterTopology, snapshot: dict) -> int:
    occupied = _occupied_gpus(snapshot)
    return sum(node.gpus for node in topology.nodes) - len(occupied)


def _assert_no_overlapping_bindings(snapshot: dict) -> None:
    occupied = _occupied_gpus(snapshot)
    assert len(occupied) == len(set(occupied))


def _occupied_gpus(snapshot: dict) -> list[tuple[str, int]]:
    occupied = []
    for binding in snapshot.values():
        node = binding["node"]
        occupied.extend((node, gpu) for gpu in binding["gpu_ids"])
    return occupied


from tre_common.registry import ClusterTopology, NodeSpec
from tre_sm.allocator.slots import Binding, Migration, Slot, SlotAllocator


def single_node_topology():
    return ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),))


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

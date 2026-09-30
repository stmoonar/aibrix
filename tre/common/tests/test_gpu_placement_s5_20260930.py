"""S5 (2026-09-30) placement ranking: violation -> split cost (fewest broken
blocks first) -> node effective load (GPU load + registry placement_penalty) ->
same model on the node -> address; the node-block-load key is gone. Release is
the mirror image. Synthetic node / model names only."""

from types import SimpleNamespace

from tre_common.gpu_placement import (
    GpuBlock,
    PlacementPolicy,
    choose_placement,
    choose_release,
    enumerate_blocks,
    placement_policy_from_registry,
    plan_placements,
    plan_releases,
)
from tre_common.registry import PlacementConfig

N1, N2 = "n1", "n2"
TWO = {N1: 4, N2: 4}
POLICY = PlacementPolicy(max_order=1, reserve_blocks=0)


def gpu(node, *ids):
    return GpuBlock(node, tuple(ids))


def test_split_cost_ranks_before_node_load():
    # n1 is fuller (3 of 4) but its free GPU completes a pair; n2 is empty.
    occupied = {(N1, 0), (N1, 1), (N1, 2)}
    choice = choose_placement(enumerate_blocks(TWO, 1), nodes=TWO, occupied=occupied, tp_size=1, policy=POLICY)
    assert choice.block == gpu(N1, 3)
    assert choice.split_cost == 0
    assert "node_eff_load=1" in choice.reason and "node_block_load" not in choice.reason


def test_node_eff_load_then_same_model_then_address_break_equal_costs():
    occupied = {(N1, 0), (N1, 1)}
    # equal cost (every free GPU breaks a pair): the lighter node n2
    choice = choose_placement(enumerate_blocks(TWO, 1), nodes=TWO, occupied=occupied, tp_size=1, policy=POLICY)
    assert choice.block == gpu(N2, 0)
    # equal load: the node where the model has fewer GPUs
    occupied = {(N1, 0), (N1, 1), (N2, 0), (N2, 1)}
    choice = choose_placement(
        enumerate_blocks(TWO, 1), nodes=TWO, occupied=occupied, tp_size=1, policy=POLICY,
        model_occupied={(N1, 0)},
    )
    assert choice.block == gpu(N2, 2)
    # everything equal: the lowest address
    choice = choose_placement(enumerate_blocks(TWO, 1), nodes=TWO, occupied=occupied, tp_size=1, policy=POLICY)
    assert choice.block == gpu(N1, 2)


def test_placement_penalty_moves_load_off_a_node_without_naming_it_in_code():
    penalised = PlacementPolicy(max_order=1, reserve_blocks=0, node_penalty={N1: 0.5})
    assert penalised.penalty(N1) == 0.5 and penalised.penalty(N2) == 0
    first = choose_placement(enumerate_blocks(TWO, 1), nodes=TWO, occupied=(), tp_size=1, policy=penalised)
    assert first.block == gpu(N2, 0)
    # the penalty does not beat the split cost: a fragment on n1 is still filled first
    fragment = choose_placement(
        enumerate_blocks(TWO, 1), nodes=TWO, occupied={(N1, 0), (N2, 0), (N2, 1)}, tp_size=1, policy=penalised,
    )
    assert fragment.block == gpu(N1, 1)
    # release: equal merge gain -> the penalised node gives back first
    held = [gpu(N1, 0), gpu(N2, 0)]
    occupied = {(N1, 0), (N1, 1), (N2, 0), (N2, 1)}
    assert choose_release(held, nodes=TWO, occupied=occupied, policy=penalised).block == gpu(N1, 0)
    assert choose_release(held, nodes=TWO, occupied=occupied, policy=POLICY).block == gpu(N2, 0)


def test_release_order_mirrors_the_placement_order():
    picks = plan_placements(enumerate_blocks(TWO, 1), nodes=TWO, occupied=(), tp_size=1, count=3, policy=POLICY)
    placed = [pick.block for pick in picks]
    assert placed == [gpu(N1, 0), gpu(N1, 1), gpu(N2, 0)]
    occupied = {key for block in placed for key in block.keys}
    released = plan_releases(placed, nodes=TWO, occupied=occupied, count=3, policy=POLICY)
    assert [pick.block for pick in released] == list(reversed(placed))


def test_policy_from_registry_carries_the_placement_penalty():
    models = [SimpleNamespace(name="m", tp_size=1, scale_max_replicas=4)]
    registry = SimpleNamespace(
        models=lambda: models,
        placement=lambda: PlacementConfig(node_penalty={"n2": 0.25}),
    )
    policy = placement_policy_from_registry(registry)
    assert policy.penalty("n2") == 0.25 and policy.penalty("n1") == 0
    assert placement_policy_from_registry(SimpleNamespace(models=lambda: models)).node_penalty == ()

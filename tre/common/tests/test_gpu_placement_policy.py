"""Placement policy: node balance at reservation granularity, soft pair reservation,
buddy order cap (design tre/docs/design/20260928-placement-node-balance.md).

All node / model names are synthetic."""

from types import SimpleNamespace

import pytest

from tre_common.gpu_placement import (
    BEST_FIT,
    GpuBlock,
    PlacementPolicy,
    choose_placement,
    choose_release,
    enumerate_blocks,
    placement_policy_from_registry,
    plan_releases,
)

N1, N2 = "n1", "n2"
TWO = {N1: 4, N2: 4}
POLICY = PlacementPolicy(max_order=1, reserve_blocks=1)


def gpu(node, *ids):
    return GpuBlock(node, tuple(ids))


class Cluster:
    """Awake GPUs per model; wakes pick among every aligned block of the model's tp."""

    def __init__(self, nodes, tp_sizes, policy):
        self.nodes = nodes
        self.tp = tp_sizes
        self.policy = policy
        self.held = {model: [] for model in tp_sizes}

    def occupied(self):
        return {key for blocks in self.held.values() for block in blocks for key in block.keys}

    def model_gpus(self, model):
        return {key for block in self.held[model] for key in block.keys}

    def wake(self, model):
        choice = choose_placement(
            enumerate_blocks(self.nodes, self.tp[model]),
            nodes=self.nodes,
            occupied=self.occupied(),
            tp_size=self.tp[model],
            policy=self.policy,
            model_occupied=self.model_gpus(model),
        )
        if choice is None:
            return None
        self.held[model].append(choice.block)
        return choice.block

    def shrink(self, model):
        choice = choose_release(
            self.held[model], nodes=self.nodes, occupied=self.occupied(), policy=self.policy
        )
        self.held[model].pop(choice.index)
        return choice.block

    def used_groups(self, size):
        occupied = self.occupied()
        return {
            node: sum(
                1
                for base in range(0, gpus, size)
                if any((node, g) in occupied for g in range(base, min(base + size, gpus)))
            )
            for node, gpus in self.nodes.items()
        }


def test_sequential_wakes_from_empty_balance_nodes_and_keep_pairs():
    cluster = Cluster(TWO, {"A": 1, "B": 1, "C": 2}, POLICY)

    assert cluster.wake("A") == gpu(N1, 0)
    assert cluster.wake("B") == gpu(N1, 1)  # fills the half-used pair: no new node load
    assert cluster.wake("C") == gpu(N2, 0, 1)  # the less loaded node
    assert cluster.wake("A") == gpu(N2, 2)  # A already on n1: spread to n2
    # Only n1's [2,3] pair is left whole: B takes the free half of n2's pair
    # instead of consuming the last free pair (reservation).
    assert cluster.wake("B") == gpu(N2, 3)
    assert cluster.used_groups(2) == {N1: 1, N2: 2}


def test_release_mirrors_placement():
    cluster = Cluster(TWO, {"A": 1, "B": 1, "C": 2}, POLICY)
    for model in ("A", "B", "C", "A", "B"):
        cluster.wake(model)

    # A's replicas: n1:0 (pair shared with B on the lighter n1) and n2:2 (pair shared
    # with B on n2, which also holds C). Neither release frees a pair, so the more
    # loaded node goes first: the replica placed last.
    assert cluster.shrink("A") == gpu(N2, 2)
    # B then holds n1:1 and n2:3; freeing n2:3 merges into a free pair ([2,3]).
    assert cluster.shrink("B") == gpu(N2, 3)


def test_release_prefers_the_node_holding_most_of_the_model():
    nodes = {N1: 8, N2: 8}
    # A holds a whole pair on n1 and half a pair on n2 (X holds the other half).
    held = [gpu(N1, 0), gpu(N1, 1), gpu(N2, 0)]
    occupied = {(N2, 1)} | {key for block in held for key in block.keys}

    picks = plan_releases(held, nodes=nodes, occupied=occupied, count=1, policy=POLICY)

    # No release frees a pair (gain 0) and both nodes have 1 of 4 pairs in use:
    # n1 holds more of A, then the higher address.
    assert picks[0].block == gpu(N1, 1)


def test_reservation_keeps_the_last_free_pair_when_an_alternative_exists():
    # n1: X (tp2) on [0,1], A on 2; n2: X on [0,1]; the only free pair is n2 [2,3].
    occupied = {(N1, 0), (N1, 1), (N1, 2), (N2, 0), (N2, 1)}
    a_gpus = {(N1, 2)}
    singles = enumerate_blocks(TWO, 1)

    kept = choose_placement(
        singles, nodes=TWO, occupied=occupied, tp_size=1, policy=POLICY, model_occupied=a_gpus
    )
    assert kept.block == gpu(N1, 3)
    assert kept.score[0] == 0  # no violation

    # Without a reserve, same-model spread wins and breaks the last pair.
    spread = choose_placement(
        singles,
        nodes=TWO,
        occupied=occupied,
        tp_size=1,
        policy=PlacementPolicy(max_order=1, reserve_blocks=0),
        model_occupied=a_gpus,
    )
    assert spread.block == gpu(N2, 2)


def test_reservation_is_soft_and_bounded_by_headroom():
    # n1 full but for gpu 3; n2 empty: a reserve of 2 keeps both n2 pairs free.
    occupied = {(N1, 0), (N1, 1), (N1, 2)}
    singles = enumerate_blocks(TWO, 1)
    reserve2 = PlacementPolicy(max_order=1, reserve_blocks=2, reserve_caps=(("C", 4),))

    assert choose_placement(singles, nodes=TWO, occupied=occupied, tp_size=1, policy=reserve2).block == gpu(N1, 3)
    # Node balance alone would open the idle node.
    assert choose_placement(
        singles, nodes=TWO, occupied=occupied, tp_size=1, policy=PlacementPolicy(max_order=1)
    ).block == gpu(N2, 0)
    # C (the only tp2 model) is at its cap: nothing to reserve for.
    capped = reserve2.for_awake({"C": 4})
    assert capped.reserve_blocks == 0
    assert choose_placement(singles, nodes=TWO, occupied=occupied, tp_size=1, policy=capped).block == gpu(N2, 0)
    assert reserve2.effective_reserve({"C": 3}) == 1
    assert reserve2.effective_reserve(None) == 2

    # Soft: with every candidate violating, a placement still happens.
    full_pairs = {(N1, g) for g in range(4)} | {(N2, 0), (N2, 1)}
    choice = choose_placement(
        enumerate_blocks(TWO, 2), nodes=TWO, occupied=full_pairs, tp_size=2, policy=reserve2
    )
    assert choice.block == gpu(N2, 2, 3) and choice.score[0] == 2


def test_reservation_is_a_no_op_when_every_model_is_single_gpu():
    policy = PlacementPolicy(max_order=0, reserve_blocks=3)
    assert policy.effective_reserve() == 0
    choice = choose_placement(
        enumerate_blocks(TWO, 1), nodes=TWO, occupied={(N1, 0)}, tp_size=1, policy=policy
    )
    assert choice.block == gpu(N2, 0)  # max_order 0: node load is the GPU fraction


def test_buddy_climbing_is_capped_at_max_order():
    big = {"solo": 8}
    occupied = {("solo", 4)}
    # gpu0 sits in a free order-2 block, gpu6 only in a free pair.
    candidates = [gpu("solo", 0), gpu("solo", 6)]

    uncapped = choose_placement(
        candidates, nodes=big, occupied=occupied, tp_size=1, policy=PlacementPolicy(max_order=3)
    )
    capped = choose_placement(candidates, nodes=big, occupied=occupied, tp_size=1, policy=POLICY)

    assert uncapped.block == gpu("solo", 6) and uncapped.split_cost == 1
    assert capped.split_cost == 1 and capped.enclosing_order == 1
    assert capped.block == gpu("solo", 0)  # no order-2/3 penalty: address decides
    # A half-used pair still beats a fresh one (split cost 0 vs 1).
    half = choose_placement(
        [gpu("solo", 0), gpu("solo", 5)], nodes=big, occupied=occupied, tp_size=1, policy=POLICY
    )
    assert half.block == gpu("solo", 5) and half.split_cost == 0
    # Release: merge gain stops at max_order too.
    release = choose_release(
        [gpu("solo", 4)], nodes=big, occupied=occupied, policy=POLICY
    )
    assert release.merge_order == 1 and release.merge_gain == 1


def test_generic_three_nodes_of_eight_stay_within_one_block():
    nodes = {"g-a": 8, "g-b": 8, "g-c": 8}
    cluster = Cluster(nodes, {"P": 1, "Q": 1, "R": 2}, POLICY)
    placed = 0
    for step in range(64):
        model = ("P", "Q", "R", "P", "R", "Q", "Q")[step % 7]
        if cluster.wake(model) is None:
            continue
        placed += 1
        loads = cluster.used_groups(2).values()
        assert max(loads) - min(loads) <= 1, (step, cluster.used_groups(2))
    assert placed >= 16
    assert len(cluster.occupied()) == 24  # the cluster fills up completely
    # Replicas of one model are spread: no node holds more than one extra.
    for model in ("P", "Q", "R"):
        per_node = [sum(1 for b in cluster.held[model] if b.node == n) for n in nodes]
        assert max(per_node) - min(per_node) <= 2, (model, per_node)


def test_best_fit_is_the_default_policy():
    assert BEST_FIT.balance_nodes is False and BEST_FIT.max_order is None
    # Legacy packing: the second single lands beside the first.
    first = choose_placement(enumerate_blocks(TWO, 1), nodes=TWO, occupied=(), tp_size=1)
    second = choose_placement(
        enumerate_blocks(TWO, 1), nodes=TWO, occupied=set(first.block.keys), tp_size=1
    )
    assert (first.block, second.block) == (gpu(N1, 0), gpu(N1, 1))


def _registry(tp_sizes, reserve=None):
    models = [
        SimpleNamespace(name=f"m{index}", tp_size=tp, scale_max_replicas=4)
        for index, tp in enumerate(tp_sizes)
    ]
    placement = None if reserve is None else SimpleNamespace(reserve_tp_pairs=reserve)
    registry = SimpleNamespace(models=lambda: models)
    if placement is not None:
        registry.placement = lambda: placement
    return registry


@pytest.mark.parametrize(
    "tp_sizes, max_order", [((1, 1), 0), ((1, 1, 2), 1), ((2, 4, 1), 2), ((), 0)]
)
def test_policy_from_registry_derives_max_order(tp_sizes, max_order):
    policy = placement_policy_from_registry(_registry(tp_sizes))
    assert policy.max_order == max_order
    assert policy.balance_nodes


def test_policy_from_registry_reserve_and_caps():
    policy = placement_policy_from_registry(_registry((1, 2, 2), reserve=3))
    assert policy.reserve_blocks == 3
    assert policy.reserve_caps == (("m1", 4), ("m2", 4))
    assert placement_policy_from_registry(_registry((1, 2)), awake_counts={"m1": 4}).reserve_blocks == 0
    assert placement_policy_from_registry(_registry((1, 2))).reserve_blocks == 1  # default
    assert placement_policy_from_registry(_registry((1, 1), reserve=2)).effective_reserve() == 0


def test_policy_from_the_real_registry_file():
    from tre_common.registry import load_registry

    registry = load_registry()
    policy = placement_policy_from_registry(registry)
    widest = max(model.tp_size for model in registry.models())
    assert 1 << policy.max_order == widest
    assert policy.reserve_blocks == registry.placement().reserve_tp_pairs
    assert registry.placement().defrag_enabled is False


def test_placement_section_parsing():
    from tre_common.registry import PlacementConfig, parse_placement_config

    assert parse_placement_config(None) == PlacementConfig(reserve_tp_pairs=1, defrag_enabled=False)
    assert parse_placement_config({"reserve_tp_pairs": 0, "defrag": {"enabled": True}}) == PlacementConfig(
        reserve_tp_pairs=0, defrag_enabled=True
    )
    assert parse_placement_config({"defrag": None}).defrag_enabled is False
    with pytest.raises(ValueError, match="unknown keys"):
        parse_placement_config({"reserve_pairs": 1})
    with pytest.raises(ValueError, match="placement.defrag: unknown keys"):
        parse_placement_config({"defrag": {"enable": True}})
    with pytest.raises(ValueError, match="integer"):
        parse_placement_config({"reserve_tp_pairs": True})

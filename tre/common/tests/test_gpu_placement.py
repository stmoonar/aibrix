"""Buddy placement/release tests, including brute-force oracles.

The state space of two 4-GPU nodes is tiny, so the property tests below compare
the buddy heuristic against exhaustive references that score every legal move by
the free-block profile it leaves behind.
"""

import itertools
import random

import pytest

from tre_common.gpu_placement import (
    GpuBlock,
    block_order,
    choose_placement,
    choose_release,
    enumerate_blocks,
    free_blocks,
    plan_placements,
    plan_releases,
)


NODE9 = "nscc-ds-4a100-node9"
NODE10 = "nscc-ds-4a100-node10"
CLUSTER = {NODE9: 4, NODE10: 4}


def gpu(node, *ids):
    return GpuBlock(node, tuple(ids))


def singles(nodes=CLUSTER):
    return enumerate_blocks(nodes, 1)


def pairs(nodes=CLUSTER):
    return enumerate_blocks(nodes, 2)


# --------------------------------------------------------------------------
# brute-force oracles
# --------------------------------------------------------------------------


def free_profile(nodes, occupied):
    """Counts of fully free aligned blocks, largest order first."""
    occupied = set(occupied)
    max_order = max(gpus.bit_length() - 1 for gpus in nodes.values())
    profile = []
    for order in range(max_order, -1, -1):
        size = 1 << order
        count = 0
        for node, gpus in nodes.items():
            for base in range(0, gpus - gpus % size, size):
                if all((node, g) not in occupied for g in range(base, base + size)):
                    count += 1
        profile.append(count)
    return tuple(profile)


def largest_free_order(nodes, occupied, node, gpu_id):
    """Largest order whose aligned block around ``gpu_id`` is fully free (-1: none)."""
    occupied = set(occupied)
    best = -1
    for order in range(0, nodes[node].bit_length()):
        size = 1 << order
        base = (gpu_id // size) * size
        if base + size > nodes[node]:
            break
        if all((node, g) not in occupied for g in range(base, base + size)):
            best = order
    return best


def brute_force_placement(candidates, nodes, occupied, tp_size):
    """Free block leaving the best free profile; ties go to the lowest address."""
    occupied = set(occupied)
    best = None
    for block in candidates:
        if block.size != tp_size or any(k in occupied for k in block.keys):
            continue
        profile = free_profile(nodes, occupied | set(block.keys))
        if (
            best is None
            or profile > best[0]
            or (profile == best[0] and block.address_key < best[1].address_key)
        ):
            best = (profile, block)
    return None if best is None else best[1]


def brute_force_release(candidates, nodes, occupied):
    """Held block leaving the best free profile once freed; ties go high address."""
    occupied = set(occupied)
    best = None
    for block in candidates:
        profile = free_profile(nodes, occupied - set(block.keys))
        if (
            best is None
            or profile > best[0]
            or (profile == best[0] and block.address_key > best[1].address_key)
        ):
            best = (profile, block)
    return None if best is None else best[1]


def brute_force_merge_order(block, nodes, occupied):
    return largest_free_order(
        nodes, set(occupied) - set(block.keys), block.node, block.base
    )


# --------------------------------------------------------------------------
# 1. the reported bug: do not skip the free buddy of an occupied GPU
# --------------------------------------------------------------------------


def test_single_gpu_request_takes_the_buddy_of_an_occupied_gpu():
    occupied = {(NODE10, 0)}

    choice = choose_placement(singles(), nodes=CLUSTER, occupied=occupied, tp_size=1)

    assert choice.block == gpu(NODE10, 1)
    assert choice.split_cost == 0
    assert choice.enclosing_order == 0
    assert "split_cost=0" in choice.reason


def test_three_single_gpu_wakes_do_not_touch_a_third_pair():
    occupied = {(NODE10, 0)}
    picks = plan_placements(
        singles(), nodes=CLUSTER, occupied=occupied, tp_size=1, count=3
    )

    blocks = [pick.block for pick in picks]
    assert blocks[0] == gpu(NODE10, 1)  # cost 0: finishes the already dirty pair
    assert blocks[1] == gpu(NODE10, 2)  # cost 1: has to dirty the node's other pair
    assert blocks[2] == gpu(NODE10, 3)  # cost 0 again
    # node9 is untouched, so a tp=2 replica can still be placed.
    after = occupied | {key for block in blocks for key in block.keys}
    assert free_blocks(CLUSTER, after, 2) == [gpu(NODE9, 0, 1), gpu(NODE9, 2, 3)]


# --------------------------------------------------------------------------
# 2. cross-node best fit
# --------------------------------------------------------------------------


def test_best_fit_crosses_nodes_instead_of_opening_an_empty_node():
    occupied = {(NODE10, 2)}

    choice = choose_placement(singles(), nodes=CLUSTER, occupied=occupied, tp_size=1)

    assert choice.block == gpu(NODE10, 3)
    assert choice.split_cost == 0


def test_lowest_address_wins_on_a_fully_empty_cluster():
    choice = choose_placement(singles(), nodes=CLUSTER, occupied=(), tp_size=1)

    # natural ordering: node9 before node10, although "node10" < "node9" as strings.
    assert choice.block == gpu(NODE9, 0)
    assert choice.split_cost == 2


def test_pair_request_prefers_the_node_that_is_already_dirty():
    occupied = {(NODE10, 0)}

    choice = choose_placement(pairs(), nodes=CLUSTER, occupied=occupied, tp_size=2)

    assert choice.block == gpu(NODE10, 2, 3)
    assert choice.split_cost == 0
    # node9 stays whole for a future tp=4 replica.
    assert free_blocks(CLUSTER, occupied | set(choice.block.keys), 4) == [
        gpu(NODE9, 0, 1, 2, 3)
    ]


# --------------------------------------------------------------------------
# 3. an aligned pair survives a burst of single-GPU wakes
# --------------------------------------------------------------------------


def test_three_single_gpu_wakes_on_an_empty_cluster_keep_a_pair_free():
    picks = plan_placements(singles(), nodes=CLUSTER, occupied=(), tp_size=1, count=3)

    assert len(picks) == 3
    occupied = {key for pick in picks for key in pick.block.keys}
    assert free_blocks(CLUSTER, occupied, 2)  # at least one aligned pair left
    assert free_blocks(CLUSTER, occupied, 4) == [gpu(NODE10, 0, 1, 2, 3)]


def test_five_single_gpu_wakes_still_leave_one_pair():
    picks = plan_placements(singles(), nodes=CLUSTER, occupied=(), tp_size=1, count=5)

    occupied = {key for pick in picks for key in pick.block.keys}
    assert len(occupied) == 5
    assert len(free_blocks(CLUSTER, occupied, 2)) == 1


def test_plan_placements_stops_when_candidates_run_out():
    occupied = {(NODE9, g) for g in range(4)} | {(NODE10, g) for g in range(3)}

    picks = plan_placements(
        singles(), nodes=CLUSTER, occupied=occupied, tp_size=1, count=3
    )

    assert [pick.block for pick in picks] == [gpu(NODE10, 3)]
    assert (
        choose_placement(
            singles(), nodes=CLUSTER, occupied=occupied | {(NODE10, 3)}, tp_size=1
        )
        is None
    )


# --------------------------------------------------------------------------
# 4. release symmetry
# --------------------------------------------------------------------------


def test_releasing_two_of_four_single_gpu_replicas_frees_an_aligned_pair():
    held = [gpu(NODE10, g) for g in range(4)]
    occupied = {key for block in held for key in block.keys}

    picks = plan_releases(held, nodes={NODE10: 4}, occupied=occupied, count=2)

    freed = {key for pick in picks for key in pick.block.keys}
    assert freed == {(NODE10, 2), (NODE10, 3)}
    assert free_blocks({NODE10: 4}, occupied - freed, 2) == [gpu(NODE10, 2, 3)]
    assert picks[0].merge_order == 0  # nothing merges yet; high address breaks the tie
    assert picks[1].merge_order == 1
    assert picks[1].merge_gain == 1


def test_release_prefers_the_gpu_that_completes_a_free_pair():
    # This model holds gpu0, gpu2 and gpu3; gpu1 is already free.
    held = [gpu(NODE10, 0), gpu(NODE10, 2), gpu(NODE10, 3)]
    occupied = {key for block in held for key in block.keys}

    choice = choose_release(held, nodes={NODE10: 4}, occupied=occupied)

    assert choice.block == gpu(NODE10, 0)  # merges with the free gpu1 into [0,1]
    assert choice.merge_order == 1
    assert choice.merge_gain == 1


def test_release_maximises_the_merge_order():
    # Freeing the pair on node10 yields the whole node; freeing node9's does not.
    held = [gpu(NODE9, 0, 1), gpu(NODE10, 2, 3)]
    occupied = {key for block in held for key in block.keys} | {(NODE9, 2)}

    choice = choose_release(held, nodes=CLUSTER, occupied=occupied)

    assert choice.block == gpu(NODE10, 2, 3)
    assert choice.merge_order == 2
    assert "merge_order=2" in choice.reason


def test_release_indices_refer_to_the_original_candidate_sequence():
    held = [gpu(NODE10, g) for g in range(4)]
    occupied = {key for block in held for key in block.keys}

    picks = plan_releases(held, nodes={NODE10: 4}, occupied=occupied, count=2)

    assert [pick.index for pick in picks] == [3, 2]
    assert [held[pick.index] for pick in picks] == [pick.block for pick in picks]


# --------------------------------------------------------------------------
# 5. brute-force property tests
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tp_size", [1, 2])
def test_placement_matches_brute_force_over_random_states(tp_size):
    rng = random.Random(20260921)
    all_gpus = [(node, g) for node in (NODE9, NODE10) for g in range(4)]
    candidates = enumerate_blocks(CLUSTER, tp_size)

    checked = 0
    for _ in range(400):
        occupied = set(rng.sample(all_gpus, rng.randint(0, 7)))
        choice = choose_placement(
            candidates, nodes=CLUSTER, occupied=occupied, tp_size=tp_size
        )
        expected = brute_force_placement(candidates, CLUSTER, occupied, tp_size)
        assert (choice.block if choice else None) == expected, (occupied, tp_size)
        checked += expected is not None
    assert checked > 50


def test_placement_never_splits_a_pair_when_an_alternative_exists():
    """Exhaustive over all 2**8 occupancies of the two nodes."""
    all_gpus = [(node, g) for node in (NODE9, NODE10) for g in range(4)]
    candidates = enumerate_blocks(CLUSTER, 1)
    seen_alternatives = 0

    def splits_a_pair(block, occupied):
        return (block.node, block.base ^ 1) not in occupied

    for mask in range(1 << len(all_gpus)):
        occupied = {g for i, g in enumerate(all_gpus) if mask >> i & 1}
        free = [block for block in candidates if block.keys[0] not in occupied]
        if not free:
            continue
        choice = choose_placement(
            candidates, nodes=CLUSTER, occupied=occupied, tp_size=1
        )
        if any(not splits_a_pair(block, occupied) for block in free):
            seen_alternatives += 1
            assert not splits_a_pair(choice.block, occupied), (occupied, choice.block)
            assert choice.split_cost == 0
        else:
            assert choice.split_cost >= 1
    assert seen_alternatives > 100


def test_release_matches_brute_force_over_random_states():
    rng = random.Random(20260922)
    all_gpus = [(node, g) for node in (NODE9, NODE10) for g in range(4)]

    for _ in range(400):
        held_keys = rng.sample(all_gpus, rng.randint(1, 5))
        held = [gpu(node, g) for node, g in held_keys]
        others = set(rng.sample(all_gpus, rng.randint(0, 3))) - set(held_keys)
        occupied = set(held_keys) | others

        choice = choose_release(held, nodes=CLUSTER, occupied=occupied)
        expected = brute_force_release(held, CLUSTER, occupied)
        assert choice.block == expected, (occupied, held_keys)
        assert choice.merge_order == brute_force_merge_order(
            choice.block, CLUSTER, occupied
        )


def test_random_placement_release_cycles_stay_optimal():
    """Mixed tp=1/tp=2 workload never loses to the exhaustive one-step oracle."""
    rng = random.Random(20260923)
    pools = {1: enumerate_blocks(CLUSTER, 1), 2: enumerate_blocks(CLUSTER, 2)}

    for _ in range(200):
        occupied = set()
        held = []
        for _step in range(12):
            if not held or rng.random() < 0.6:
                tp_size = rng.choice([1, 2])
                pool = pools[tp_size]
                choice = choose_placement(
                    pool, nodes=CLUSTER, occupied=occupied, tp_size=tp_size
                )
                if choice is None:
                    continue
                assert choice.block == brute_force_placement(
                    pool, CLUSTER, occupied, tp_size
                )
                occupied |= set(choice.block.keys)
                held.append(choice.block)
            else:
                choice = choose_release(held, nodes=CLUSTER, occupied=occupied)
                # Mixed block sizes: the spec ranks releases by merge order only.
                best = max(
                    brute_force_merge_order(block, CLUSTER, occupied) for block in held
                )
                assert choice.merge_order == best
                occupied -= set(choice.block.keys)
                held.pop(choice.index)


# --------------------------------------------------------------------------
# 6. tp=4 (and beyond) generalise
# --------------------------------------------------------------------------


def test_tp4_only_fits_on_a_fully_empty_node():
    quads = enumerate_blocks(CLUSTER, 4)
    assert quads == [gpu(NODE9, 0, 1, 2, 3), gpu(NODE10, 0, 1, 2, 3)]

    choice = choose_placement(quads, nodes=CLUSTER, occupied={(NODE9, 1)}, tp_size=4)
    assert choice.block == gpu(NODE10, 0, 1, 2, 3)
    assert choice.split_cost == 0
    assert choice.considered == 1

    assert (
        choose_placement(
            quads, nodes=CLUSTER, occupied={(NODE9, 1), (NODE10, 3)}, tp_size=4
        )
        is None
    )


def test_tp1_and_tp2_traffic_leaves_room_for_a_later_tp4():
    occupied = set()
    for tp_size in (1, 1, 2):
        pool = enumerate_blocks(CLUSTER, tp_size)
        choice = choose_placement(
            pool, nodes=CLUSTER, occupied=occupied, tp_size=tp_size
        )
        occupied |= set(choice.block.keys)

    assert free_blocks(CLUSTER, occupied, 4) == [gpu(NODE10, 0, 1, 2, 3)]


def test_orders_above_two_generalise_on_a_wider_node():
    big = {"fat-node": 8}
    assert block_order(8) == 3
    assert choose_placement(
        enumerate_blocks(big, 8), nodes=big, occupied=(), tp_size=8
    ).block == gpu("fat-node", 0, 1, 2, 3, 4, 5, 6, 7)

    # gpu0 busy: [2,3] costs nothing extra ([0..3] is already dirty), while
    # [4,5] would split the still-intact [4..7] quad.
    choice = choose_placement(
        enumerate_blocks(big, 2), nodes=big, occupied={("fat-node", 0)}, tp_size=2
    )
    assert choice.block == gpu("fat-node", 2, 3)
    assert choice.split_cost == 0
    assert choice.enclosing_order == 1


# --------------------------------------------------------------------------
# 7. invalid input
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tp_size", [0, -2, 3, 5, 6, 7])
def test_non_power_of_two_tp_size_is_rejected(tp_size):
    with pytest.raises(ValueError, match="power of two"):
        choose_placement(singles(), nodes=CLUSTER, occupied=(), tp_size=tp_size)


def test_tp_size_larger_than_a_node_is_rejected():
    with pytest.raises(ValueError, match="exceeds the widest node"):
        choose_placement(
            enumerate_blocks({NODE9: 4}, 4), nodes=CLUSTER, occupied=(), tp_size=8
        )


def test_candidate_size_must_match_tp_size():
    with pytest.raises(ValueError, match="tp_size is 2"):
        choose_placement(singles(), nodes=CLUSTER, occupied=(), tp_size=2)


def test_unaligned_candidate_is_rejected():
    with pytest.raises(ValueError, match="not buddy-aligned"):
        choose_placement([gpu(NODE9, 1, 2)], nodes=CLUSTER, occupied=(), tp_size=2)


def test_non_contiguous_candidate_is_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        choose_placement([gpu(NODE9, 0, 2)], nodes=CLUSTER, occupied=(), tp_size=2)


def test_candidate_past_the_end_of_the_node_is_rejected():
    with pytest.raises(ValueError, match="past the end"):
        choose_placement([gpu(NODE9, 4, 5)], nodes=CLUSTER, occupied=(), tp_size=2)


def test_non_power_of_two_candidate_is_rejected():
    with pytest.raises(ValueError, match="power of two"):
        choose_release([gpu(NODE9, 0, 1, 2)], nodes=CLUSTER, occupied=())


def test_unknown_node_is_rejected_in_candidates_and_occupancy():
    with pytest.raises(ValueError, match="unknown node"):
        choose_placement([gpu("ghost", 0)], nodes=CLUSTER, occupied=(), tp_size=1)
    with pytest.raises(ValueError, match="unknown node"):
        choose_placement(singles(), nodes=CLUSTER, occupied={("ghost", 0)}, tp_size=1)
    with pytest.raises(ValueError, match="out of range"):
        choose_placement(singles(), nodes=CLUSTER, occupied={(NODE9, 9)}, tp_size=1)


def test_empty_inputs_are_explicit():
    with pytest.raises(ValueError, match="at least one node"):
        choose_placement(singles(), nodes={}, occupied=(), tp_size=1)
    with pytest.raises(ValueError, match="tp_size is required"):
        choose_placement([], nodes=CLUSTER, occupied=())
    assert choose_release([], nodes=CLUSTER, occupied=()) is None
    assert plan_releases([], nodes=CLUSTER, occupied=(), count=2) == []
    with pytest.raises(ValueError, match="count must be"):
        plan_placements(singles(), nodes=CLUSTER, tp_size=1, count=-1)


# --------------------------------------------------------------------------
# misc behaviour worth pinning down
# --------------------------------------------------------------------------


def test_tp_size_defaults_to_the_candidate_size():
    choice = choose_placement(pairs(), nodes=CLUSTER, occupied={(NODE9, 0)})
    assert choice.block == gpu(NODE9, 2, 3)


def test_enumerate_blocks_ignores_a_ragged_tail():
    assert enumerate_blocks({"odd": 3}, 2) == [gpu("odd", 0, 1)]
    assert len(enumerate_blocks({"odd": 3}, 1)) == 3


def test_merge_stops_at_a_ragged_node_boundary():
    # On a 3-GPU node gpu2 can never merge upwards ([2,3] runs past the end), so
    # it is the cheapest slot of all: using it splits nothing.
    choice = choose_placement(
        enumerate_blocks({"odd": 3}, 1), nodes={"odd": 3}, occupied=(), tp_size=1
    )
    assert choice.block == gpu("odd", 2)
    assert choice.split_cost == 0
    assert choice.enclosing_order == 0

    # ...and the intact [0,1] pair is only broken once gpu2 is taken.
    choice = choose_placement(
        enumerate_blocks({"odd": 3}, 1),
        nodes={"odd": 3},
        occupied={("odd", 2)},
        tp_size=1,
    )
    assert choice.block == gpu("odd", 0)
    assert choice.enclosing_order == 1  # [0,1] is free; [0..3] does not exist


def test_placement_reason_and_bookkeeping_are_reported():
    choice = choose_placement(
        singles(), nodes=CLUSTER, occupied={(NODE9, 0), (NODE10, 0)}, tp_size=1
    )
    assert choice.index == singles().index(choice.block)
    assert choice.considered == 6
    assert str(choice.block) in choice.reason
    assert "free_candidates=6/8" in choice.reason


def test_candidate_order_does_not_change_the_answer():
    occupied = {(NODE10, 1)}
    base = choose_placement(singles(), nodes=CLUSTER, occupied=occupied, tp_size=1)
    for perm in itertools.islice(itertools.permutations(singles(), 8), 0, 50):
        choice = choose_placement(
            list(perm), nodes=CLUSTER, occupied=occupied, tp_size=1
        )
        assert choice.block == base.block

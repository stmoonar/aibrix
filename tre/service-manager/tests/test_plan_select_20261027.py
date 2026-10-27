"""Selection of a power plan (tre_sm.ops.plan, pure; design 20261027-sm-plans.md
section 4.1): the two waves, the TP=2 rule, the replica floor, and the
fragmentation rules (TP pair reserve of a relay, cumulative sleep items, widest
wake item first)."""

from tre_common.gpu_placement import PlacementPolicy
from tre_common.registry import ClusterTopology, NodeSpec
from tre_sm.allocator.slots import Binding, Slot, release_order
from tre_sm.api.v2 import _reserve_violation
from tre_sm.ops.plan import (
    SOURCE_DONOR,
    SOURCE_FREE,
    SOURCE_PLAN_SLEEP,
    Donor,
    SleepItem,
    WakeItem,
    select_plan,
)

TOPOLOGY = ClusterTopology(
    nodes=(NodeSpec("node-a", 4, ((0, 1), (2, 3)), ("GPU-0", "GPU-1", "GPU-2", "GPU-3")),)
)


def b(serve_id, model, gpus, *, awake=False, hidden=False):
    return Binding(serve_id, model, Slot("node-a", tuple(gpus)), awake=awake, hidden=hidden)


def blocker(binding, planning, _ignored):
    """The account: an awake binding on one of its GPUs."""
    for other in planning:
        if other.awake and other.serve_id != binding.serve_id and set(other.slot.gpu_ids) & set(binding.slot.gpu_ids):
            return f"slot_occupied by {other.serve_id}"
    return None


def select(fleet, **kwargs):
    kwargs.setdefault("topology", TOPOLOGY)
    kwargs.setdefault("wake_blocker", blocker)
    return select_plan(fleet, **kwargs)


def picks(choice):
    return [(p.binding.serve_id, p.wave, p.depends_on) for p in choice.picks]


def test_a_mixed_plan_puts_free_gpus_in_wave_1_and_plan_sleep_gpus_in_wave_2():
    fleet = [b("d-0", "d", (0,), awake=True), b("r-0", "r", (0,)), b("x-1", "x", (1,))]
    selection = select(
        fleet,
        sleeps=[SleepItem("d", "urgent", pods=("d-0",))],
        wakes=[WakeItem("x", at_least=1), WakeItem("r", at_least=1)],
    )
    assert [m.serve_id for m in selection.sleeps[0].members] == ["d-0"]
    assert picks(selection.wakes[0]) == [("x-1", 1, ())]
    assert picks(selection.wakes[1]) == [("r-0", 2, ("d/node-a/0",))]


def test_tp2_waits_for_every_gpu_and_a_free_plus_plan_sleep_mix_is_wave_2():
    fleet = [b("d-0", "d", (0,), awake=True), b("d-1", "d", (1,), awake=True), b("t-01", "t", (0, 1))]
    both = select(fleet, sleeps=[SleepItem("d", "urgent", at_most=0)], wakes=[WakeItem("t", at_least=1)])
    assert picks(both.wakes[0]) == [("t-01", 2, ("d/node-a/0", "d/node-a/1"))]

    # TP2 over a free GPU (1) and a plan sleep (0): wave 2, depends on that sleep;
    # a TP1 item of the same plan cannot take the free GPU (one card, one awake).
    mixed = [b("d-0", "d", (0,), awake=True), b("t-01", "t", (0, 1)), b("x-1", "x", (1,))]
    one = select(
        mixed, sleeps=[SleepItem("d", "urgent", at_most=0)],
        wakes=[WakeItem("x", at_least=1), WakeItem("t", at_least=1)],
    )
    assert picks(one.wakes[1]) == [("t-01", 2, ("d/node-a/0",))]
    assert one.wakes[1].picks[0].source == SOURCE_PLAN_SLEEP
    assert one.wakes[0].picks == [] and one.wakes[0].unfilled == 1


def test_floor_refuses_a_named_sleep_clamps_at_most_and_bounds_relay_donors():
    fleet = [b("d-0", "d", (0,), awake=True), b("d-1", "d", (1,), awake=True), b("r-0", "r", (0,)), b("r-1", "r", (1,))]
    floors = {"d": (1, {"d/node-a/0", "d/node-a/1"})}

    named = select(fleet, sleeps=[SleepItem("d", "urgent", pods=("d-0", "d-1"))], floors=floors)
    assert named.sleeps[0].refusal.reason == "floor_violation" and named.sleep_set() == []

    shrink = select(fleet, sleeps=[SleepItem("d", "scale_down", at_most=0)], floors=floors)
    assert shrink.sleeps[0].clamped and len(shrink.sleeps[0].members) == 1
    assert len(shrink.sleeps[0].kept_awake) == 1

    relay = select(fleet, wakes=[WakeItem("r", at_least=2, donors=(Donor("d", 2),))], floors=floors)
    [choice] = relay.wakes
    assert len(choice.picks) == 1 and choice.picks[0].source == SOURCE_DONOR
    assert len(choice.donor_sleeps) == 1 and choice.unfilled == 1


def test_free_gpus_are_used_before_relay_donors_and_a_vetoed_pick_is_substituted():
    fleet = [b("d-0", "d", (0,), awake=True), b("r-0", "r", (0,)), b("r-2", "r", (2,)), b("r-3", "r", (3,))]
    vetoed = []

    def veto(binding, wave, _sleeping):
        if binding.serve_id == "r-2":
            vetoed.append(wave)
            return "resident_awake"
        return None

    selection = select(fleet, wakes=[WakeItem("r", at_least=2, donors=(Donor("d", 1),))], veto=veto)
    [choice] = selection.wakes
    assert sorted(p.binding.serve_id for p in choice.picks) == ["r-0", "r-3"]
    assert {p.binding.serve_id: p.source for p in choice.picks}["r-0"] == SOURCE_DONOR
    assert vetoed == [1] and [s[0] for s in choice.substituted] == ["r/node-a/2"]


def test_a_met_target_is_a_noop_and_leaves_the_donors_alone():
    fleet = [b("d-0", "d", (0,), awake=True), b("r-2", "r", (2,), awake=True), b("r-0", "r", (0,))]
    selection = select(fleet, wakes=[WakeItem("r", at_least=1, donors=(Donor("d", 1),))])
    assert selection.wakes[0].noop and selection.sleep_set() == []


POLICY = PlacementPolicy(max_order=1, reserve_blocks=1)


def reserve(binding, planning):
    return _reserve_violation(binding, planning, TOPOLOGY, POLICY)


def test_a_relay_uses_its_donor_before_breaking_the_last_free_pair():
    # GPUs 0, 1 busy (d-0 a donor), the pair 2-3 free: the reserve keeps it.
    fleet = [b("d-0", "d", (0,), awake=True), b("o-1", "o", (1,), awake=True),
             b("r-0", "r", (0,)), b("r-2", "r", (2,)), b("r-3", "r", (3,))]
    relay = select(fleet, wakes=[WakeItem("r", at_least=1, donors=(Donor("d", 1),))], reserve_violation=reserve)
    assert [(p.binding.serve_id, p.source) for p in relay.wakes[0].picks] == [("r-0", SOURCE_DONOR)]

    alone = select(fleet, wakes=[WakeItem("r", at_least=1)], reserve_violation=reserve)
    [pick] = alone.wakes[0].picks
    assert pick.source == SOURCE_FREE and pick.binding.serve_id in ("r-2", "r-3")

    # No donor left to use: the deferred free pick is taken after all.
    floor = {"d": (1, {"d/node-a/0"})}
    fallback = select(fleet, wakes=[WakeItem("r", at_least=1, donors=(Donor("d", 1),))],
                      floors=floor, reserve_violation=reserve)
    [pick] = fallback.wakes[0].picks
    assert pick.source == SOURCE_FREE and fallback.sleep_set() == []


def test_sleep_items_are_cumulative_and_complete_a_pair_together():
    # a-0 is hidden (its shrink takes it first); b's shrink then sees GPU 0 free
    # and releases b-1, freeing the whole pair 0-1 (alone it would take b-3).
    fleet = [b("a-0", "a", (0,), awake=True, hidden=True), b("a-2", "a", (2,), awake=True),
             b("b-1", "b", (1,), awake=True), b("b-3", "b", (3,), awake=True)]

    def shrink_order(serving, books, released):
        return release_order(serving, bindings=books, topology=TOPOLOGY, already_released=released,
                             policy=PlacementPolicy(max_order=1))

    selection = select(
        fleet, shrink_order=shrink_order,
        sleeps=[SleepItem("a", "scale_down", at_most=1), SleepItem("b", "scale_down", at_most=1)],
    )
    assert [m.serve_id for c in selection.sleeps for m in c.members] == ["a-0", "b-1"]


def test_the_widest_wake_item_goes_first_and_results_keep_the_request_order():
    # One free pair (2-3): the TP1 relay item listed first must not split it.
    fleet = [b("d-0", "d", (0,), awake=True), b("o-1", "o", (1,), awake=True),
             b("r-0", "r", (0,)), b("r-2", "r", (2,)), b("t-23", "t", (2, 3))]
    selection = select(fleet, wakes=[WakeItem("r", at_least=1, donors=(Donor("d", 1),)), WakeItem("t", at_least=1)])
    assert [c.index for c in selection.wakes] == [0, 1]
    assert [(p.binding.serve_id, p.source) for p in selection.wakes[0].picks] == [("r-0", SOURCE_DONOR)]
    assert [(p.binding.serve_id, p.source) for p in selection.wakes[1].picks] == [("t-23", SOURCE_FREE)]


def test_tp1_items_of_one_plan_share_contested_free_gpus_round_robin():
    """H2 (2026-10-28): two TP=1 items each wanting +2 with 2 free GPUs get one each,
    whatever their order in the request (was: the first item took both)."""
    fleet = [b("a-0", "a", (0,)), b("a-1", "a", (1,)), b("z-0", "z", (0,)), b("z-1", "z", (1,))]
    for order in (("a", "z"), ("z", "a")):
        selection = select(fleet, wakes=[WakeItem(model, at_least=2) for model in order])
        assert [len(choice.picks) for choice in selection.wakes] == [1, 1], order
        assert [choice.unfilled for choice in selection.wakes] == [1, 1], order

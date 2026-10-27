"""Selection of a power plan (tre_sm.ops.plan, pure; design 20261027-sm-plans.md
section 4.1): the two waves, the TP=2 cover rule, the replica floor."""

from tre_common.registry import ClusterTopology, NodeSpec
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.ops.plan import (
    SKIP_UNCOVERED,
    SOURCE_DONOR,
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


def test_tp2_waits_for_every_gpu_and_a_free_plus_plan_sleep_mix_is_skipped():
    fleet = [b("d-0", "d", (0,), awake=True), b("d-1", "d", (1,), awake=True), b("t-01", "t", (0, 1))]
    both = select(fleet, sleeps=[SleepItem("d", "urgent", at_most=0)], wakes=[WakeItem("t", at_least=1)])
    assert picks(both.wakes[0]) == [("t-01", 2, ("d/node-a/0", "d/node-a/1"))]

    mixed = [b("d-0", "d", (0,), awake=True), b("t-01", "t", (0, 1))]
    one = select(mixed, sleeps=[SleepItem("d", "urgent", at_most=0)], wakes=[WakeItem("t", at_least=1)])
    assert one.wakes[0].picks == [] and one.wakes[0].unfilled == 1
    assert one.wakes[0].skipped == {SKIP_UNCOVERED: 1}


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

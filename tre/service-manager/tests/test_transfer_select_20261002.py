"""Pair selection of the transfer primitive (tre_sm.ops.transfer, pure)."""

from types import SimpleNamespace

from tre_common.registry import ClusterTopology, NodeSpec
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.ops.transfer import (
    SKIP_FLOOR,
    SKIP_FOREIGN_OCCUPANT,
    SKIP_NO_DONOR,
    SKIP_NOT_IN_DONOR_FILTER,
    SKIP_OCCUPANT_BUSY,
    SKIP_OCCUPANT_HIDDEN,
    SKIP_OVER_COUNT,
    SKIP_RECEIVER_BUSY,
    SKIP_UNCOVERED_GPU,
    busy_binding_ids,
    release_pick,
    select_transfer_pairs,
)

TOPOLOGY = ClusterTopology(
    nodes=(NodeSpec("node-a", 4, ((0, 1), (2, 3)), ("GPU-0", "GPU-1", "GPU-2", "GPU-3")),)
)


def b(serve_id, model, gpus, *, awake=False, hidden=False, node="node-a"):
    return Binding(serve_id, model, Slot(node, tuple(gpus)), awake=awake, hidden=hidden)


def select(bindings, **kwargs):
    kwargs.setdefault("donor_model", "d")
    kwargs.setdefault("receiver_model", "r")
    kwargs.setdefault("count", 1)
    kwargs.setdefault("topology", TOPOLOGY)
    return select_transfer_pairs(bindings, **kwargs)


def pairs_of(selection):
    return [(p.receiver.serve_id, [d.serve_id for d in p.donors]) for p in selection.pairs]


def test_pairs_a_sleeping_receiver_with_the_awake_donor_on_its_gpu():
    fleet = [
        b("d-0", "d", (0,), awake=True), b("r-0", "r", (0,)), b("x-0", "x", (0,)),
        b("r-2", "r", (2,)),  # nothing awake on GPU 2: a plain wake, not a transfer
    ]
    selection = select(fleet)
    assert pairs_of(selection) == [("r-0", ["d-0"])]
    assert selection.unfilled == 0
    assert selection.skipped == {SKIP_NO_DONOR: 1}


def test_tp2_receiver_takes_two_single_gpu_donors_and_counts_them_both():
    fleet = [b("d-0", "d", (0,), awake=True), b("d-1", "d", (1,), awake=True), b("t-01", "t", (0, 1))]
    one = select(fleet, receiver_model="t", count=1)
    assert one.pairs == [] and one.unfilled == 1 and one.skipped == {SKIP_OVER_COUNT: 1}
    two = select(fleet, receiver_model="t", count=2)
    assert pairs_of(two) == [("t-01", ["d-0", "d-1"])]
    assert two.pairs[0].weight == 2 and two.unfilled == 0


def test_tp2_receiver_with_a_gpu_no_donor_holds_is_not_paired():
    # GPU 1 is free: it would not be fenced while the donor sleeps (no pre-claim).
    fleet = [b("d-0", "d", (0,), awake=True), b("t-01", "t", (0, 1))]
    selection = select(fleet, receiver_model="t", count=2)
    assert selection.pairs == [] and selection.skipped == {SKIP_UNCOVERED_GPU: 1}


def test_tp1_receiver_under_a_tp2_donor_weighs_one():
    fleet = [b("d-01", "d", (0, 1), awake=True), b("r-0", "r", (0,)), b("r-1", "r", (1,))]
    selection = select(fleet, count=2)
    # The donor frees both GPUs but is one replica: one pair; r-1's GPU is free
    # afterwards (a plain wake for the caller), so the second replica is unfilled.
    assert pairs_of(selection) == [("r-0", ["d-01"])]
    assert selection.unfilled == 1


def test_excludes_hidden_journaled_reserved_and_leased_bindings():
    fleet = [
        b("d-0", "d", (0,), awake=True, hidden=True), b("r-0", "r", (0,)),  # hidden donor
        b("d-1", "d", (1,), awake=True), b("r-1", "r", (1,)),  # donor busy (journal / lease)
        b("d-2", "d", (2,), awake=True), b("r-2", "r", (2,)),  # receiver busy
        b("d-3", "d", (3,), awake=True), b("r-3", "r", (3,), hidden=True),  # hidden receiver
    ]
    selection = select(fleet, count=4, busy={"d/node-a/1", "r/node-a/2"})
    assert selection.pairs == []
    assert selection.skipped == {SKIP_OCCUPANT_HIDDEN: 1, SKIP_OCCUPANT_BUSY: 1, SKIP_RECEIVER_BUSY: 1}


def test_busy_ids_cover_both_journals_and_transient_leases():
    busy = busy_binding_ids(
        wake_journal={"r/node-a/1": {}},
        sleep_journal={"d-2": {"binding_id": "d/node-a/2"}},
        transient_lease_ids=["x/node-a/3"],
    )
    assert busy == {"r/node-a/1", "d/node-a/2", "x/node-a/3"}


def test_a_third_resident_awake_is_never_paired():
    # In the books: an awake x on the receiver's GPU is not a donor.
    fleet = [b("d-0", "d", (0,), awake=True), b("x-0", "x", (0,), awake=True), b("r-0", "r", (0,))]
    selection = select(fleet)
    assert selection.pairs == [] and selection.skipped == {SKIP_FOREIGN_OCCUPANT: 1}


def test_a_physically_awake_third_resident_is_vetoed_and_the_next_pair_taken():
    fleet = [
        b("d-0", "d", (0,), awake=True), b("r-0", "r", (0,)), b("x-0", "x", (0,)),
        b("d-1", "d", (1,), awake=True), b("r-1", "r", (1,)),
    ]
    asked = []

    def veto(pair):
        asked.append(pair.receiver.serve_id)
        if pair.receiver.serve_id == "r-0":
            return SimpleNamespace(reason="resident_awake", blocking_binding_id="x/node-a/0")
        return None

    selection = select(fleet, count=2, veto=veto)
    assert pairs_of(selection) == [("r-1", ["d-1"])]
    assert selection.unfilled == 1
    assert sorted(asked) == ["r-0", "r-1"]  # each pair asked once, when about to be taken
    assert [r.reason for r in selection.refusals] == ["resident_awake"]
    assert selection.substituted[0]["refused_binding_id"] == "r/node-a/0"


def test_fault_hook_veto_switches_to_another_pair_and_count_still_met():
    fleet = [
        b("d-0", "d", (0,), awake=True), b("r-0", "r", (0,)),
        b("d-1", "d", (1,), awake=True), b("r-1", "r", (1,)),
        b("d-2", "d", (2,), awake=True), b("r-2", "r", (2,)),
    ]
    selection = select(
        fleet, count=2,
        veto=lambda pair: "fault_injected" if pair.receiver.serve_id == "r-0" else None,
    )
    assert sorted(r for r, _ in pairs_of(selection)) == ["r-1", "r-2"]
    assert selection.unfilled == 0


def test_account_blocker_sees_the_donors_asleep_and_ignores_their_leases():
    fleet = [b("d-0", "d", (0,), awake=True), b("r-0", "r", (0,))]
    seen = {}

    def blocker(receiver, planning, ignored):
        seen["donor_awake"] = next(x.awake for x in planning if x.serve_id == "d-0")
        seen["ignored"] = ignored
        return None

    assert pairs_of(select(fleet, wake_blocker=blocker)) == [("r-0", ["d-0"])]
    assert seen == {"donor_awake": False, "ignored": frozenset({"d/node-a/0"})}


def test_account_refusal_is_reported_and_skipped():
    fleet = [b("d-0", "d", (0,), awake=True), b("r-0", "r", (0,))]
    selection = select(fleet, wake_blocker=lambda *a: SimpleNamespace(reason="lease_starting"))
    assert selection.pairs == [] and [r.reason for r in selection.refusals] == ["lease_starting"]


def test_donor_floor_limits_the_pairs():
    fleet = [
        b("d-0", "d", (0,), awake=True), b("r-0", "r", (0,)),
        b("d-1", "d", (1,), awake=True), b("r-1", "r", (1,)),
    ]
    routable = {"d/node-a/0", "d/node-a/1"}
    selection = select(fleet, count=2, donor_floor=1, donor_routable=routable)
    assert len(selection.pairs) == 1 and selection.unfilled == 1
    assert selection.skipped == {SKIP_FLOOR: 1}
    none = select(fleet, count=2, donor_floor=2, donor_routable=routable)
    assert none.pairs == [] and none.unfilled == 2


def test_count_batches_pairs_across_gpus():
    fleet = [b(f"d-{g}", "d", (g,), awake=True) for g in range(4)] + [b(f"r-{g}", "r", (g,)) for g in range(4)]
    selection = select(fleet, count=3)
    assert len(selection.pairs) == 3 and selection.unfilled == 0
    assert len({p.receiver.serve_id for p in selection.pairs}) == 3
    assert len({p.donors[0].serve_id for p in selection.pairs}) == 3
    capped = select(fleet, count=3, receiver_budget=1)
    assert len(capped.pairs) == 1 and capped.unfilled == 2


def test_donor_filter_and_avoid_gpus():
    fleet = [
        b("d-0", "d", (0,), awake=True), b("r-0", "r", (0,)),
        b("d-1", "d", (1,), awake=True), b("r-1", "r", (1,)),
    ]
    only_d1 = select(fleet, donor_filter=["d-1"])
    assert pairs_of(only_d1) == [("r-1", ["d-1"])]
    assert only_d1.skipped == {SKIP_NOT_IN_DONOR_FILTER: 1}
    assert pairs_of(select(fleet, donor_filter=["d/node-a/0"])) == [("r-0", ["d-0"])]
    assert pairs_of(select(fleet, avoid_gpus=["node-a/0"])) == [("r-1", ["d-1"])]


def test_release_order_is_the_planner_release_pick_not_the_serve_id():
    # Releasing d-b (GPU 0, GPU 1 free) gives back a whole pair; d-a (GPU 2,
    # GPU 3 busy) does not. The serve id order alone would pick d-a.
    fleet = [
        b("d-a", "d", (2,), awake=True), b("r-2", "r", (2,)),
        b("x-3", "x", (3,), awake=True),
        b("d-b", "d", (0,), awake=True), b("r-0", "r", (0,)),
    ]
    selection = select(fleet, count=1)
    assert pairs_of(selection) == [("r-0", ["d-b"])]
    units = [(fleet[0],), (fleet[3],)]
    assert release_pick(units, fleet, TOPOLOGY)[0].serve_id == "d-b"


def test_release_pick_matches_the_controller_planner():
    from tre_controller.planning import planner

    fleet = [
        b("d-a", "d", (2,), awake=True), b("x-3", "x", (3,), awake=True),
        b("d-b", "d", (0,), awake=True), b("d-c", "d", (1,), awake=True),
    ]
    view = SimpleNamespace(topology=TOPOLOGY, bindings=fleet, placement=None)
    donors = [fleet[0], fleet[2], fleet[3]]
    expected = planner._release_pick(donors, view)
    assert release_pick([(item,) for item in donors], fleet, TOPOLOGY)[0] == expected


def test_receiver_pick_decides_between_receivers_one_donor_frees():
    fleet = [b("d-01", "d", (0, 1), awake=True), b("r-0", "r", (0,)), b("r-1", "r", (1,))]
    chosen = select(fleet, receiver_pick=lambda receivers, planning: receivers[-1])
    assert pairs_of(chosen) == [("r-1", ["d-01"])]

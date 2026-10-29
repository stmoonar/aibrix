"""TP block probe vs fairness piggyback (2026-10-02, approach (a); the same-slot shrink
became the block supply of L4 rev 2 on 2026-10-28). The piggyback was removed on
2026-10-07 (design donor-evidence-20261007): a donor probed in a plan gives no other
receiver anything; these tests keep the ledger and that invariant.

A block's probe (BlockSupplyAction.probe) is committed by the tick with exactly
``{receiver: 1}``. The planner's ledger records that claim, and no LOW receiver is
ever promised the same probe.
"""

from __future__ import annotations

from test_floor_plan_20260929 import _cls, _topology
from tre_common.registry import NodeSpec
from tre_controller.planning.classify import ModelRole, ModelState
from tre_controller.planning.planner import (
    ClusterView,
    PlanConfig,
    BlockSupplyAction,
    ScaleAction,
    build_plan,
)
from tre_common.gpu_placement import GpuBlock
from tre_sm.allocator.slots import Binding, Slot
from relay_view import block_takes, expand_relays, relays  # noqa: F401 - 2026-10-02 relay intents


def _same_slot_plan(high_replicas: int, *, low_replicas: int = 1):
    classifications = [
        _cls("tp2", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
        _cls("low", ModelState.LOW, ModelRole.RECEIVER, 0.9),
        _cls("high", ModelState.HIGH, ModelRole.DONOR, 1.4, "surplus"),
    ]
    contexts = {
        "tp2": {"assigned_replicas": 0, "routable_pods": 0},
        "low": {"assigned_replicas": low_replicas, "routable_pods": low_replicas},
        "high": {"assigned_replicas": high_replicas, "routable_pods": high_replicas},
    }
    bindings = [
        Binding("high-0", "high", Slot("node-a", (0,)), awake=True),
        Binding("high-2", "high", Slot("node-a", (2,)), awake=True),
        Binding("low-3", "low", Slot("node-a", (3,)), awake=True),
        # A sleeping binding on a GPU a HIGH replica holds awake: LOW cannot wake it, and
        # the SM tries it before creating, so there is no idle capacity for LOW.
        Binding("low-s", "low", Slot("node-a", (2,)), awake=False),
        Binding("tp2-a01", "tp2", Slot("node-a", (0, 1)), awake=False),
    ]
    if high_replicas > 2:
        bindings.append(Binding("high-n", "high", Slot("node-b", (0,)), awake=True))
    view = ClusterView(
        topology=_topology((NodeSpec(name="node-b", gpus=2, two_gpu_slots=((0, 1),)),)),
        bindings=tuple(bindings),
    )
    cfg = PlanConfig(
        min_replicas_per_model=1,
        max_replicas_per_model=4,
        model_tp_sizes={"tp2": 2, "low": 1, "high": 1},
    )
    return build_plan(
        model_contexts=contexts,
        classifications=classifications,
        model_replicas={"tp2": 0, "low": low_replicas, "high": high_replicas},
        idle_gpus=0,
        cfg=cfg,
        cluster_view=view,
    )


def test_a_block_probe_is_claimed_in_the_ledger_and_not_piggybacked_by_low():
    plan = _same_slot_plan(2)
    assert [(donor, receiver) for donor, receiver, _ in block_takes(plan.actions)] == [("high", "tp2")]
    assert plan.probe_upscale_plans == {"high": {"tp2": 1}}
    assert "low" not in plan.probe_upscale_plans.get("high", {})
    assert not [a for a in expand_relays(plan.actions) if isinstance(a, ScaleAction) and a.model in {"high", "low"}]


def test_three_replica_donor_with_a_block_probe_starts_no_second_probe_for_low():
    """F1-B (2026-10-07): a HIGH donor gives LOW nothing at once any more (before: an
    immediate pair on another pod). It already has this tick's probe (tp2's block),
    and a donor gets one probe at a time."""
    plan = _same_slot_plan(3)
    [shrink] = [a for a in plan.actions if isinstance(a, BlockSupplyAction)]
    assert plan.probe_upscale_plans.get("high", {}).get("tp2") == 1
    assert "low" not in plan.probe_upscale_plans.get("high", {})
    assert not relays(plan.actions)
    assert not [a for a in plan.actions if isinstance(a, ScaleAction) and a.model in {"high", "low"}]
    assert shrink.probe == ("high", "high-0")


def test_middle_zone_claim_leaves_nothing_unclaimed_for_a_later_low_receiver():
    from test_planner import _classification

    plan = build_plan(
        model_contexts={
            "critical": {"assigned_replicas": 2, "routable_pods": 2},
            "low": {"assigned_replicas": 1, "routable_pods": 1},
            "healthy": {"assigned_replicas": 2, "routable_pods": 2},
        },
        classifications=[
            _classification("critical", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
            _classification("low", ModelState.LOW, ModelRole.RECEIVER, 0.9),
            _classification("healthy", ModelState.HEALTHY, ModelRole.NEUTRAL, 1.2),
        ],
        model_replicas={"critical": 2, "low": 1, "healthy": 2},
        idle_gpus=0,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4),
    )
    shrinks = [a for a in expand_relays(plan.actions) if isinstance(a, ScaleAction) and a.model == "healthy"]
    assert [(a.reason, a.delta) for a in shrinks] == [("critical_middle_zone_safescale", -1)]
    # The CRITICAL claim covers the whole shrink: LOW gets nothing from it.
    assert plan.probe_upscale_plans == {"healthy": {"critical": 1}}


def test_tick_commits_a_block_probe_with_only_the_receiver_upscale():
    # Hand-built input: the BlockSupplyAction and the ledger are given directly.
    from test_loop_ticks import _metrics_with_pods
    from tre_common.metrics_schema import MetricsSnapshot
    from tre_controller.loops.tick import _apply_safescale
    from tre_controller.planning.safescale import SafeScaleConfig, SafeScaleStateMachine

    safescale = SafeScaleStateMachine(config=SafeScaleConfig(min_window_ms=60_000.0))
    snapshot = MetricsSnapshot(
        ts_ms=20_000, stale=False,
        models={"high": _metrics_with_pods("high", generation=200.0, waiting=0.0, running=1.0, pods=("high-0",))},
    )
    shrink = BlockSupplyAction(
        receiver="tp2", block=GpuBlock("node-a", (0, 1)), receiver_pods=("tp2-a01",), sleeps=(),
        probe=("high", "high-0"), reason="critical_block_high", source_loop="rescue",
    )
    # The block's own receiver promise is the probe's only upscale (2026-10-29: a block
    # starts through _start_block_probes, which ignores probe_upscale_plans).
    out, events = _apply_safescale(snapshot, (shrink,), {"high": {"tp2": 1}}, safescale=safescale)
    probe = safescale.active_probe("high")
    assert probe is not None and probe.pending_upscales == {"tp2": 1}
    decision = safescale._commit(probe, reason="formal_commit_gate_passed")
    assert [(c.kind, c.model, c.delta) for c in decision.commands if c.kind == "scale_up"] == [("scale_up", "tp2", 1)]

"""Test helper (2026-10-02, design 20261002-controller-transfer; 2026-10-28 sm-plans
section 10).

The planner emits an immediate donor -> receiver relay as named transfers: one
``BlockSupplyAction`` per pair on a one-GPU block (the donor pods to sleep and the
receiver binding to wake). Tests written against the former count forms assert on
donor / receiver / count / reason: :func:`relays` gives them one :class:`Relay` per
(donor, receiver, reason) with the counts summed, and :func:`expand_relays` the
donor's ``-count`` and the receiver's ``+pairs`` as ScaleActions (for assertions
only - nothing is dispatched this way)."""

from __future__ import annotations

from dataclasses import dataclass

from tre_controller.planning.planner import BlockSupplyAction, ClusterView, ScaleAction
from tre_sm.allocator.slots import Binding, Slot


@dataclass(frozen=True)
class Relay:
    donor_model: str
    receiver_model: str
    count: int
    pairs: int
    reason: str
    source_loop: str
    donor_pods: tuple[str, ...] = ()
    receiver_pods: tuple[str, ...] = ()
    sleep_path: str = "urgent"
    rescue: object = None


def is_relay(action) -> bool:
    """A TP1 relay pair: a transfer onto one GPU released at once."""
    return isinstance(action, BlockSupplyAction) and action.block.size == 1 and action.probe is None


def relays(actions) -> list[Relay]:
    out: dict[tuple, Relay] = {}
    for action in actions:
        if is_relay(action):
            donor = action.donor_models[0]
            key = (donor, action.receiver, action.reason)
            pods = tuple(pod for _, pod in action.sleeps)
            old = out.get(key)
            out[key] = Relay(
                donor, action.receiver, (old.count if old else 0) + len(pods), (old.pairs if old else 0) + 1,
                action.reason, action.source_loop, (*(old.donor_pods if old else ()), *pods),
                (*(old.receiver_pods if old else ()), *action.receiver_pods), rescue=action.rescue,
            )
    return list(out.values())


def expand_relays(actions) -> list:
    out: list = []
    pending = {id(action) for action in actions if is_relay(action)}
    for action in actions:
        if id(action) not in pending:
            # A tick's plan batch (a transfer group, sm-plans section 10): its members.
            out.extend(getattr(action, "members", (action,)))
            continue
        for relay in relays((action,)):
            out.append(
                ScaleAction(
                    relay.donor_model, -int(relay.count), relay.reason, relay.source_loop,
                    donor=relay.donor_model, receiver=relay.receiver_model, sleep_path=relay.sleep_path,
                    pods=relay.donor_pods,
                )
            )
            out.append(
                ScaleAction(
                    relay.receiver_model, int(relay.pairs), relay.reason, relay.source_loop,
                    donor=relay.donor_model, receiver=relay.receiver_model, rescue=relay.rescue,
                    pods=relay.receiver_pods,
                )
            )
    return out


def signed_deltas(actions) -> dict[str, int]:
    out: dict[str, int] = {}
    for action in expand_relays(actions):
        if isinstance(action, ScaleAction):
            out[action.model] = out.get(action.model, 0) + action.delta
    return out


def block_takes(actions) -> list[tuple[str, str, str]]:
    """(donor model, receiver, donor pod) of every TP block (L4 rev 2,
    ``BlockSupplyAction`` on more than one GPU) in a plan: its immediate sleeps and its
    probe pod."""
    return [
        (donor, action.receiver, pod)
        for action in actions
        if isinstance(action, BlockSupplyAction) and not is_relay(action)
        for donor, pod in (*action.sleeps, *((action.probe,) if action.probe else ()))
    ]


def fleet_view(registry, donor: str, pods, receiver: str) -> ClusterView:
    """A fleet view for a relay: ``donor``'s ``pods`` awake on GPUs 0.., ``receiver``
    sleeping under each (the controller names the pair from it, sm-plans section 10)."""
    node = registry.topology().nodes[0].name
    bindings = []
    for gpu, pod in enumerate(pods):
        bindings.append(Binding(pod, donor, Slot(node, (gpu,)), awake=True))
        bindings.append(Binding(f"{receiver}-{gpu}", receiver, Slot(node, (gpu,)), awake=False))
    return ClusterView(registry.topology(), tuple(bindings))


def transfer_members(batch) -> list[tuple]:
    """(kind, model, pods, sleep_path, reason) of a transfer plan group's members."""
    return [
        ("sleep" if member.delta < 0 else "wake", member.model, member.pods, member.sleep_path, member.reason)
        for member in batch.members
    ]


def pool_view(replicas: dict[str, int], receivers=()) -> ClusterView:
    """A fleet view for a view-less planner scenario (2026-10-28: a relay names its pods,
    so it needs one): each model's ``replicas`` awake on their own GPUs of one node, and
    a sleeping binding of every receiver under every other model's pod."""
    from tre_common.registry import ClusterTopology, NodeSpec

    total = sum(replicas.values())
    topology = ClusterTopology(nodes=(NodeSpec(name="n", gpus=total,
                                               two_gpu_slots=tuple((g, g + 1) for g in range(0, total - 1, 2))),))
    bindings, gpu = [], 0
    for model, count in replicas.items():
        for index in range(count):
            bindings.append(Binding(f"{model}-{index}", model, Slot("n", (gpu,)), awake=True))
            for receiver in receivers:
                if receiver != model:
                    bindings.append(Binding(f"{receiver}@{gpu}", receiver, Slot("n", (gpu,)), awake=False))
            gpu += 1
    return ClusterView(topology, tuple(bindings))

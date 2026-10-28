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

from tre_controller.planning.planner import BlockSupplyAction, ScaleAction, TransferIntent


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
        if isinstance(action, TransferIntent):
            key = (action.donor_model, action.receiver_model, action.reason, len(out))
            out[key] = Relay(action.donor_model, action.receiver_model, int(action.count), int(action.pairs),
                             action.reason, action.source_loop, sleep_path=action.sleep_path, rescue=action.rescue)
        elif is_relay(action):
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
    pending = {id(action) for action in actions if isinstance(action, TransferIntent) or is_relay(action)}
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

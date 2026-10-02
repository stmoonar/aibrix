"""Test helper (2026-10-02, design 20261002-controller-transfer).

The planner emits an immediate donor -> receiver relay as ONE ``TransferIntent``
(counts only). Tests written against the former two-ScaleAction form assert on
model / delta / reason / donor / receiver; :func:`expand_relays` gives them that
view: each intent becomes the donor's ``-count`` and the receiver's ``+pairs`` as
pod-less ScaleActions (for assertions only - nothing is dispatched this way)."""

from __future__ import annotations

from tre_controller.planning.planner import ScaleAction, TransferIntent


def expand_relays(actions) -> list:
    out: list = []
    for action in actions:
        if isinstance(action, TransferIntent):
            out.append(
                ScaleAction(
                    action.donor_model, -int(action.count), action.reason, action.source_loop,
                    donor=action.donor_model, receiver=action.receiver_model, sleep_path=action.sleep_path,
                )
            )
            out.append(
                ScaleAction(
                    action.receiver_model, int(action.pairs), action.reason, action.source_loop,
                    donor=action.donor_model, receiver=action.receiver_model, rescue=action.rescue,
                )
            )
        else:
            out.append(action)
    return out


def relays(actions) -> list[TransferIntent]:
    return [action for action in actions if isinstance(action, TransferIntent)]


def signed_deltas(actions) -> dict[str, int]:
    out: dict[str, int] = {}
    for action in expand_relays(actions):
        if isinstance(action, ScaleAction):
            out[action.model] = out.get(action.model, 0) + action.delta
    return out

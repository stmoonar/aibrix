"""The policy contract every baseline implements.

A policy is a stateful object built once per shell process by a factory in
:data:`tre_baselines.policies.POLICIES` (``factory(config) -> Policy``). On every tick the
shell calls :meth:`Policy.decide` with a :class:`~tre_baselines.snapshot.ClusterSnapshot`
and receives a :class:`Decision` per model it wants to act on.

Rules:

* **No I/O.** Everything a policy may read is in the snapshot or in the config handed to
  its factory (``config.policy_params`` is the policy's ConfigMap as a dict).
* **Deterministic.** The same snapshot sequence fed to a freshly built policy gives the
  same decisions. Randomness (e.g. a noise factor) must come from an RNG the policy seeds
  from ``config.seed`` / its params.
* ``decide`` may update internal state (windows, AIMD variables, ...).
* ``desired`` is the raw replica count the policy wants. The shell clamps it to
  ``[min_replicas, max_replicas]`` itself, compares it with ``awake`` and decides the
  action (none / up / down); a policy does not need to clamp. A model missing from the
  returned mapping is left alone this tick (logged as action ``none``).
* The shell may skip actuation (dry-run, SM call already in flight for the model, not the
  lock owner); the policy is not told, and sees the real ``awake`` again next tick.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from tre_baselines.snapshot import ClusterSnapshot


@dataclass(frozen=True)
class Decision:
    #: Raw desired replica count (the shell clamps it).
    desired: int
    #: Short machine-readable reason, e.g. "ibp_above_theta".
    reason: str
    #: The signals the decision was computed from; logged verbatim (must be JSON-able).
    inputs: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Policy(Protocol):
    name: str

    def decide(self, snap: ClusterSnapshot) -> Mapping[str, Decision]: ...

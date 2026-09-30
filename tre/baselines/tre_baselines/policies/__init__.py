"""Policy registry: ``TRE_BL_POLICY`` names one entry of :data:`POLICIES`.

Each value is a factory ``factory(config) -> Policy`` taking the shell's
:class:`tre_baselines.config.Config` (``config.policy_params`` = the policy ConfigMap as a
dict, ``config.models`` = per-model registry limits, ``config.tick_s``, ``config.seed``).
Baseline packages register themselves here by adding one line each.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from tre_baselines.policies.base import Decision, Policy
from tre_baselines.policies.chiron import ChironPolicy
from tre_baselines.policies.static import StaticPolicy
from tre_baselines.policies.tokenscale import TokenScalePolicy

if TYPE_CHECKING:  # pragma: no cover
    from tre_baselines.config import Config

POLICIES: dict[str, Callable[["Config"], Policy]] = {
    "static": StaticPolicy,
    "chiron": ChironPolicy,
    "tokenscale": TokenScalePolicy,
}


def build_policy(name: str, config: "Config") -> Policy:
    try:
        factory = POLICIES[name]
    except KeyError as exc:
        raise KeyError(f"unknown baseline policy {name!r} (known: {', '.join(sorted(POLICIES))})") from exc
    return factory(config)


__all__ = ["Decision", "Policy", "POLICIES", "build_policy"]

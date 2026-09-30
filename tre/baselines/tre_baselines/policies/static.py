"""Trivial policy: keep every model at its current awake count (shell tests, dry runs)."""
from __future__ import annotations

from typing import Any, Mapping

from tre_baselines.policies.base import Decision
from tre_baselines.snapshot import ClusterSnapshot


class StaticPolicy:
    name = "static"

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def decide(self, snap: ClusterSnapshot) -> Mapping[str, Decision]:
        return {
            model: Decision(desired=ms.awake, reason="static_hold", inputs={"awake": ms.awake})
            for model, ms in snap.models.items()
        }

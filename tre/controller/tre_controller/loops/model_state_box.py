"""Latest per-model signal state of the planner ticks (review 3 P2-1..P2-3).

The rescue / fairness ticks publish their classifications here; the ActionQueue
reads them to revalidate a one-shot SafeScale commit before every (re)try (the
donor needing capacity again abandons the commit, a receiver that no longer
needs capacity loses its upscale).
"""

from __future__ import annotations

import time
from typing import Callable, Mapping

#: A receiver the planner would not act on this tick (signal still warming up,
#: band dwell not yet confirmed): neither "needs" nor "does not need" capacity.
UNCONFIRMED = "unconfirmed"


class ModelStateBox:
    def __init__(
        self,
        *,
        max_age_ms: int = 60_000,
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        self._max_age_ms = int(max_age_ms)
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))
        self._states: dict[str, tuple[str, int]] = {}

    def update(
        self,
        classifications: Mapping[str, object],
        contexts: Mapping[str, Mapping] | None = None,
        *,
        ts_ms: int,
    ) -> None:
        """Record each model's ModelState (value) of a tick on snapshot ``ts_ms``;
        an older tick never overwrites a newer one."""
        contexts = contexts or {}
        for model, item in classifications.items():
            state = getattr(getattr(item, "state", None), "value", None) or str(getattr(item, "state", ""))
            role = getattr(getattr(item, "role", None), "value", None)
            ctx = contexts.get(model) or {}
            if role == "receiver" and (
                not ctx.get("signal_warm", True)
            ):
                state = UNCONFIRMED
            elif ctx.get("signal_hold_reason") is not None:
                # Not a level of the current window's full evidence (O1 breakpoint,
                # tokens missing -> held context, a stale pod scrape): the planner takes
                # no donor from it, but a receiver may still act on it (step-capped).
                state = UNCONFIRMED
            previous = self._states.get(model)
            if previous is None or previous[1] <= int(ts_ms):
                self._states[model] = (str(state), int(ts_ms))

    def get(self) -> dict[str, str]:
        """model -> state of the latest tick, without states older than max_age."""
        now = int(self._now_ms())
        return {
            model: state
            for model, (state, ts) in self._states.items()
            if now - ts <= self._max_age_ms
        }

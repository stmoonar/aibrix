"""Time-span rules on observation times (round 4, design
docs/design/20261025-instant-sampling.md "Evidence, not ticks").

A decision rule never counts windows or ticks: overlapping windows are not independent
evidence, and their number per second changes with the evidence period. It reads spans
of observation times (gateway ``scraped_ms``, :class:`tre_common.metrics_schema.Observed`):

* persistence - a condition held from its first to its last true observation for at
  least ``span_ms`` (:class:`Persistence`);
* freshness - the newest observation is at most ``max_age_ms`` old (:func:`fresh`).

``SPAN_TOLERANCE_MS``: two observations written ``S`` apart on the gateway's schedule
are fetched ``S`` +- the fetch jitter apart (the gateway's pod refresh, 50 ms by default,
plus the /metrics latency). A span is compared with this tolerance, so a condition that
held for ``S`` of the schedule counts as ``S`` and not as one window later. It is well
below every evidence period used (>= 2 s), so a span shorter by a whole period never
passes.
"""
from __future__ import annotations

SPAN_TOLERANCE_MS = 1_000


def span_reaches(span_ms: int, need_ms: int, *, tolerance_ms: int = SPAN_TOLERANCE_MS) -> bool:
    """``span_ms`` covers ``need_ms`` of observation time (within the fetch jitter)."""
    return int(span_ms) + int(tolerance_ms) >= int(need_ms)


class Persistence:
    """A condition that has held for ``span_ms`` of observation time.

    ``observe(t, holds)``: ``t`` is the observation time the verdict is about.
    ``holds`` False always resets (round 5: also at a ``t`` already seen - a re-read
    whose verdict turned false must not keep an earlier span alive). A true
    observation at the same or an older ``t`` (a re-read of one snapshot, a window
    whose newest observation did not move) returns the last result and changes
    nothing. True once ``t_last_true - t_first_true`` reaches ``span_ms``
    (:func:`span_reaches`); ``span_ms`` 0 = on the first true observation."""

    def __init__(self, span_ms: int, *, tolerance_ms: int = SPAN_TOLERANCE_MS) -> None:
        self.span_ms = max(0, int(span_ms))
        self.tolerance_ms = max(0, int(tolerance_ms))
        self.first_ms: int | None = None
        self.last_ms: int | None = None
        self._last_t: int | None = None
        self._result = False

    @property
    def held_ms(self) -> int:
        """How long the condition has held (0 when it does not)."""
        if self.first_ms is None or self.last_ms is None:
            return 0
        return int(self.last_ms) - int(self.first_ms)

    def observe(self, t_obs_ms: int, holds: bool) -> bool:
        t = int(t_obs_ms)
        if not holds:
            self._last_t = t if self._last_t is None else max(self._last_t, t)
            self.first_ms = self.last_ms = None
            self._result = False
            return False
        if self._last_t is not None and t <= self._last_t:
            return self._result
        self._last_t = t
        if self.first_ms is None:
            self.first_ms = t
        self.last_ms = t
        self._result = span_reaches(self.held_ms, self.span_ms, tolerance_ms=self.tolerance_ms)
        return self._result

    def reset(self) -> None:
        """Forget the condition (the next true observation starts a new span). An
        observation time already seen stays seen."""
        self.first_ms = self.last_ms = None
        self._result = False


def fresh(now_ms: int, t_last_ms: int | None, max_age_ms: int) -> bool:
    """The newest observation ``t_last_ms`` is at most ``max_age_ms`` before ``now_ms``
    (same clock); never fresh when unknown."""
    return t_last_ms is not None and int(now_ms) - int(t_last_ms) <= int(max_age_ms)

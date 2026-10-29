"""Time-span rules on observation times (round 4, design
docs/design/20261025-instant-sampling.md "Evidence, not ticks").

A decision rule never counts windows or ticks: overlapping windows are not independent
evidence, and their number per second changes with the evidence period. It reads spans
of observation times (gateway ``scraped_ms``, :class:`tre_common.metrics_schema.Observed`)
and compares them with :func:`span_reaches`. (The ``Persistence`` rule and ``fresh``
went with the saturation rescue and O1, 2026-10-29.)

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

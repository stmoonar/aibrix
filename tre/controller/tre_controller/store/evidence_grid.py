"""The gateway's histogram-doc grid as the SafeScale Redis evidence path sees it.

Round 4 (docs/design/20261025-instant-sampling.md "Evidence, not ticks"): the evidence
period is a storage detail. The decision layer (``planning/``, ``signals/``, the
decision loops) never reads it; the one place a doc grid is still needed - which
histogram docs form the post-hide evidence of ``safescale.evidence_source: redis``,
anchored on the gateway's own doc stamps so no cross-node clock enters it - asks this
object, which the controller builds from TRE_EVIDENCE_PERIOD_MS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The doc grid of offline / direct constructions (tests, replays of 10 s data).
OFFLINE_GRID_MS = 10_000


@dataclass(frozen=True)
class EvidenceGrid:
    """The gateway's histogram-doc boundaries, every ``period_ms`` (epoch-aligned)."""

    period_ms: int = OFFLINE_GRID_MS

    def __post_init__(self) -> None:
        if int(self.period_ms) <= 0:
            raise ValueError("the evidence grid period must be positive")

    def ceil(self, ts_ms: int) -> int:
        """The first boundary at or after ``ts_ms``."""
        period = int(self.period_ms)
        return -(-int(ts_ms) // period) * period

    def floor(self, ts_ms: int) -> int:
        """The last boundary at or before ``ts_ms``."""
        period = int(self.period_ms)
        return int(ts_ms) // period * period

    def next_after(self, stamp_ms: int) -> int:
        """The boundary after the doc stamped ``stamp_ms``."""
        return int(stamp_ms) + int(self.period_ms)

    def evidence_start(self, anchor: Any) -> int:
        """First gateway boundary after the hide (``HideAnchor``): the newest doc stamp
        at the hide + one period (gateway clock, no cross-node clock involved), else
        the first boundary at or after hide_ts."""
        if anchor.newest_doc_ts_ms is not None:
            return self.next_after(int(anchor.newest_doc_ts_ms))
        return self.ceil(int(anchor.ts_ms))

    def clock_offsets(self, anchor: Any, *, tolerance_ms: float) -> dict[str, Any]:
        """Offsets of the gateway stamps and of the controller clock against hide_ts.

        ``gateway_offset_ms`` = hide_ts - newest doc stamp: 0 .. one period (+ write
        delay) when the clocks agree. ``controller_offset_ms`` = controller clock -
        hide_ts: ~0. ``clock_skew_alert`` is True when either is off by more than
        ``tolerance_ms``. Alert only: the evidence window is anchored on the doc stamps
        and does not depend on it."""
        gateway = None if anchor.newest_doc_ts_ms is None else int(anchor.ts_ms) - int(anchor.newest_doc_ts_ms)
        controller = None if anchor.controller_ts_ms is None else int(anchor.controller_ts_ms) - int(anchor.ts_ms)
        alert = (gateway is not None and not (-tolerance_ms <= gateway <= int(self.period_ms) + tolerance_ms)) or (
            controller is not None and anchor.source == "redis_time" and abs(controller) > tolerance_ms
        )
        return {"gateway_offset_ms": gateway, "controller_offset_ms": controller, "clock_skew_alert": bool(alert)}

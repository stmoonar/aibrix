"""Startup clock-skew check against Redis ``TIME`` (plan 2026-09-27, "clock").

Heartbeats, journal timestamps and the gateway-plugin contract are compared
across hosts; a node clock minutes off (seen in practice) silently breaks
staleness checks. The service-manager itself reads liveness against Redis
server time, but a skewed host is still worth a loud warning at startup (and,
if configured, a refusal to start).
"""

from __future__ import annotations

import logging
import time
from typing import Callable

LOG = logging.getLogger(__name__)


class ClockSkewError(RuntimeError):
    pass


def measure_clock_skew_s(redis_client, *, wall: Callable[[], float] = time.time) -> float | None:
    """Local wall clock minus Redis server time (s), RTT-midpoint corrected."""
    try:
        before = wall()
        seconds, micros = redis_client.time()
        after = wall()
    except Exception as exc:
        LOG.warning("clock skew check skipped: Redis TIME failed: %s", exc)
        return None
    server = int(seconds) + int(micros) / 1_000_000
    return (before + after) / 2.0 - server


def check_clock_skew(
    redis_client,
    *,
    warn_s: float,
    fail_s: float | None = None,
    wall: Callable[[], float] = time.time,
) -> float | None:
    skew = measure_clock_skew_s(redis_client, wall=wall)
    if skew is None:
        return None
    if fail_s is not None and abs(skew) > fail_s:
        raise ClockSkewError(
            f"local clock is {skew:+.3f}s off Redis TIME (limit {fail_s}s): fix NTP/chrony"
        )
    if abs(skew) > warn_s:
        LOG.warning(
            "local clock is %+.3fs off Redis TIME (warn above %.1fs): fix NTP/chrony",
            skew,
            warn_s,
        )
    return skew

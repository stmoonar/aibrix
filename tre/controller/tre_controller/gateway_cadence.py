"""Declared gateway cadence: the gateway's TRE doc periods vs the controller's.

The gateway writes two kinds of TRE docs (2026-10-25, ``tre_common.rediskeys``):

* ``tre:v2:inst:<pod>`` every instant sample period (TRE_INSTANT_SAMPLE_INTERVAL_MS):
  the controller's per-pod coverage flag counts on this grid;
* ``tre:v2:hist:<pod>`` every evidence period (TRE_EVIDENCE_PERIOD_MS): the
  phase-aligned sampler's boundary grid, the O1 grid and the SafeScale evidence step.

Review round 1 (M4, L6): the gateway declares both periods in the hash
``tre:v2:gateway:cadence`` (``instant_ms``, ``evidence_ms``, ``written_ms``, ``version``),
written inside the MULTI of every round. The metrics store reads it with each snapshot
and calls :func:`cadence_problem`; a missing or different declaration marks the
snapshot stale (TRE_GATEWAY_INTERVAL_CHECK=fail, the default) or only logs (warn). It
replaces the former inference from doc spacing, which was startup-only and failed on
the mixed stamps of a gateway upgrade.

Review round 3 (M5): the declaration must also be fresh. ``written_ms`` (gateway clock,
the time of the round's MULTI) must be at least ``window_end - evidence_ms`` (the window
end is a gateway doc stamp, the same clock): a hash left behind by a gateway that
stopped writing - or an old gateway that never refreshes it - does not vouch for the
window. Otherwise ``gateway_cadence_stale``.
"""
from __future__ import annotations

from typing import Any, Mapping


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _int(raw: Mapping[str, Any], key: str) -> int | None:
    try:
        return int(_text(raw[key]))
    except (KeyError, TypeError, ValueError):
        return None


#: The problem code of a declaration not written for the window being read (M5).
CADENCE_STALE = "gateway_cadence_stale"


def cadence_problem(
    raw: Mapping[Any, Any] | None, *, instant_ms: int, evidence_ms: int, window_end_ms: int | None = None
) -> str | None:
    """None when the declared cadence equals the controller's periods and - with
    ``window_end_ms`` - was written at or after ``window_end_ms - evidence_ms``, else
    why not."""
    declared = {_text(key): value for key, value in (raw or {}).items()}
    if not declared:
        return "gateway_cadence_missing"
    instant, evidence = _int(declared, "instant_ms"), _int(declared, "evidence_ms")
    if instant != int(instant_ms) or evidence != int(evidence_ms):
        version = _text(declared.get("version", "")) or "unknown"
        return (
            f"gateway_cadence_mismatch:gateway instant={instant} evidence={evidence} (version {version})"
            f" != controller TRE_INSTANT_SAMPLE_INTERVAL_MS={int(instant_ms)}"
            f" TRE_EVIDENCE_PERIOD_MS={int(evidence_ms)}"
        )
    if window_end_ms is not None:
        written = _int(declared, "written_ms")
        if written is None or written < int(window_end_ms) - int(evidence_ms):
            return CADENCE_STALE
    return None

"""Declared gateway cadence: the gateway's TRE doc periods vs the controller's.

The gateway writes two kinds of TRE docs (2026-10-25, ``tre_common.rediskeys``):

* ``tre:v2:inst:<pod>`` every instant sample period (TRE_INSTANT_SAMPLE_INTERVAL_MS):
  the controller's per-pod coverage flag counts on this grid;
* ``tre:v2:hist:<pod>`` every evidence period (TRE_EVIDENCE_PERIOD_MS): the
  phase-aligned sampler's boundary grid and the store's suffix starts (a storage
  detail since round 4: no decision rule reads the period).

Review round 1 (M4, L6): the gateway declares both periods in the hash
``tre:v2:gateway:cadence`` (``instant_ms``, ``evidence_ms``, ``written_ms``, ``version``),
written inside the MULTI of every round. The metrics store reads it with each snapshot
and calls :func:`cadence_problem`; a missing or different declaration marks the
snapshot stale (TRE_GATEWAY_INTERVAL_CHECK=fail, the default) or only logs (warn). It
replaces the former inference from doc spacing, which was startup-only and failed on
the mixed stamps of a gateway upgrade.

Round 4 (docs/design/20261025-instant-sampling.md "Evidence, not ticks"): the hash
also carries ``last_round_ms``, the instant stamp of the round whose MULTI wrote it. It
is the gateway's liveness evidence: the phase-aligned sampler takes a window ending at
``E`` as complete once ``last_round_ms >= E`` (a round late by a few hundred ms is
stamped with the next instant boundary and still proves it). A declaration without the
field (a gateway image before round 4) is ``gateway_cadence_no_round``. (Round 3 M5
compared ``written_ms`` with the window end here; that freshness rule moved to the
sampler, so a late gateway is retried and held like any late round instead of marking
the snapshot stale on the spot.)
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


#: The problem code of a declaration without ``last_round_ms`` (round 4).
CADENCE_NO_ROUND = "gateway_cadence_no_round"


def last_round_ms(raw: Mapping[Any, Any] | None) -> int | None:
    """``last_round_ms`` of the declaration (None: absent / unreadable)."""
    declared = {_text(key): value for key, value in (raw or {}).items()}
    return _int(declared, "last_round_ms")


def cadence_problem(raw: Mapping[Any, Any] | None, *, instant_ms: int, evidence_ms: int) -> str | None:
    """None when the declared cadence equals the controller's periods and carries the
    gateway's ``last_round_ms``, else why not."""
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
    if _int(declared, "last_round_ms") is None:
        return CADENCE_NO_ROUND
    return None

"""Startup check: the gateway's TRE doc periods must equal the controller's settings.

The gateway writes two kinds of TRE docs (2026-10-25, ``tre_common.rediskeys``):

* ``tre:v2:inst:<pod>`` every instant sample period (TRE_INSTANT_SAMPLE_INTERVAL_MS):
  the controller's per-pod completeness rule counts a window's expected instant ticks on
  this grid;
* ``tre:v2:hist:<pod>`` every evidence period (TRE_EVIDENCE_PERIOD_MS): the
  phase-aligned sampler's boundary grid, the O1 grid and the SafeScale evidence step.

Evidence (read-only): the median spacing of each pod's newest instant and histogram
stamps - the cadences the controller actually consumes. (Before 2026-10-25 the
request-trace metadata ``meta_interval_sec`` was evidence too: both docs were written
every ``RequestTraceWriteInterval``. They no longer are, so it says nothing about them.)

A mismatch is fatal in ``fail`` mode (the default); no evidence at all only logs a loud
warning - the controller must still start on an idle cluster.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Callable, Iterable

from tre_common.rediskeys import EVIDENCE_PERIOD_MS, INSTANT_SAMPLE_INTERVAL_MS, hist_key, inst_key, pods_key

LOG = logging.getLogger("tre_controller.gateway_cadence")


class GatewayCadenceMismatch(RuntimeError):
    """The gateway writes its TRE docs at other periods than the controller expects."""


@dataclass
class CadenceEvidence:
    inst_spacing_ms: dict[str, float] = field(default_factory=dict)  # pod -> median ms
    hist_spacing_ms: dict[str, float] = field(default_factory=dict)  # pod -> median ms
    errors: list[str] = field(default_factory=list)


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _spacing_ms(
    redis_client: Any, model: str, key_of: Callable[[str], str], *, samples: int = 7
) -> dict[str, float]:
    """Median spacing of the newest stamps of ``key_of(pod)``, per pod of ``model``
    (>= 3 stamps)."""
    out: dict[str, float] = {}
    for raw_pod in redis_client.smembers(pods_key(model)):
        pod = _text(raw_pod)
        members = redis_client.zrange(key_of(pod), -samples, -1, withscores=True)
        stamps = sorted(int(score) for _member, score in members)
        deltas = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
        if len(deltas) >= 2:
            out[pod] = float(median(deltas))
    return out


def read_inst_spacing_ms(redis_client: Any, model: str, *, samples: int = 7) -> dict[str, float]:
    """Median spacing of the newest instant ticks, per pod of ``model`` (>= 3 ticks)."""
    return _spacing_ms(redis_client, model, inst_key, samples=samples)


def read_hist_spacing_ms(redis_client: Any, model: str, *, samples: int = 7) -> dict[str, float]:
    """Median spacing of the newest histogram stamps, per pod of ``model``."""
    return _spacing_ms(redis_client, model, hist_key, samples=samples)


def gather_evidence(redis_client: Any, models: Iterable[str]) -> CadenceEvidence:
    evidence = CadenceEvidence()
    for model in models:
        for name, reader, target in (
            ("inst", read_inst_spacing_ms, evidence.inst_spacing_ms),
            ("hist", read_hist_spacing_ms, evidence.hist_spacing_ms),
        ):
            try:
                target.update(reader(redis_client, model))
            except Exception as exc:  # noqa: BLE001 - evidence is best-effort.
                evidence.errors.append(f"{name}:{model}:{type(exc).__name__}:{exc}")
    return evidence


def check_gateway_cadence(
    redis_client: Any,
    models: Iterable[str],
    *,
    mode: str = "fail",
    expected_ms: int = INSTANT_SAMPLE_INTERVAL_MS,
    expected_hist_ms: int = EVIDENCE_PERIOD_MS,
) -> CadenceEvidence | None:
    """Assert the gateway's instant / histogram doc periods equal ``expected_ms`` /
    ``expected_hist_ms`` (see module docstring)."""
    if mode == "off":
        return None
    evidence = gather_evidence(redis_client, list(models))
    problems: list[str] = []
    for label, spacings, expected, setting in (
        ("instant", evidence.inst_spacing_ms, expected_ms, "TRE_INSTANT_SAMPLE_INTERVAL_MS"),
        ("histogram", evidence.hist_spacing_ms, expected_hist_ms, "TRE_EVIDENCE_PERIOD_MS"),
    ):
        for pod, spacing_ms in sorted(spacings.items()):
            # Medians are exact multiples on a healthy ticker; allow 10 % for a skipped tick.
            if abs(spacing_ms - expected) > 0.1 * expected:
                problems.append(
                    f"{label} ticks of {pod}: median spacing {spacing_ms:g} ms != {setting}={expected} ms"
                )
    if problems:
        message = (
            "GATEWAY CADENCE MISMATCH - the gateway's TRE doc periods differ from the "
            "controller's; completeness, the phase-aligned sampler and the O1 grid assume "
            "them. " + "; ".join(problems)
        )
        if mode == "fail":
            raise GatewayCadenceMismatch(message)
        LOG.error(message)
        return evidence
    if not evidence.inst_spacing_ms and not evidence.hist_spacing_ms:
        LOG.warning(
            "!!! GATEWAY CADENCE UNVERIFIED !!! no instant / histogram ticks to measure; "
            "assuming the gateway writes instant docs every %d ms and histogram docs every "
            "%d ms (gateway env TRE_INSTANT_SAMPLE_INTERVAL_MS / TRE_EVIDENCE_PERIOD_MS). "
            "errors=%s", expected_ms, expected_hist_ms, evidence.errors,
        )
        return evidence
    LOG.info(
        "gateway_cadence_ok: instant %d ms over %d pods=%s; histogram %d ms over %d pods=%s",
        expected_ms,
        len(evidence.inst_spacing_ms),
        sorted(set(evidence.inst_spacing_ms.values())),
        expected_hist_ms,
        len(evidence.hist_spacing_ms),
        sorted(set(evidence.hist_spacing_ms.values())),
    )
    return evidence

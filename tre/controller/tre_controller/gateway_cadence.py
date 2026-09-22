"""Startup check: the gateway's write period must equal ``SCRAPE_INTERVAL_MS``.

The whole signal path assumes the gateway writes its instant/histogram docs every
``SCRAPE_INTERVAL_MS`` (10 s): the instant-average divisor, the phase-aligned sampler's
boundary grid, the 3-ticks-per-window freshness rule and the EMA/dwell cadence the
thresholds were fitted at. The period is the Go constant ``RequestTraceWriteInterval``
(aibrix ``pkg/cache/trace.go``), which ``/root/aibrix-main`` already changed to 5 s - a
rebuilt gateway image would silently halve it.

Two pieces of evidence, both read-only:

1. **trace metadata** - the gateway exports ``RequestTraceWriteInterval`` in every request
   trace it writes (``aibrix:<model>_request_trace_<ts>``, key ``meta_interval_sec``,
   aibrix trace.go ``ToMapLocked``; ``interval_in_seconds`` is accepted as an alias). Only
   written while a model has traffic, so it is often absent.
2. **instant tick spacing** - the median spacing of the newest ``tre:v2:inst:<pod>``
   timestamps, i.e. the cadence the controller actually consumes.

A mismatch in either is fatal in ``fail`` mode (the default); no evidence at all only logs
a loud warning - the controller must still start on an idle cluster.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Iterable, Optional

from tre_common.rediskeys import SCRAPE_INTERVAL_MS, inst_key, pods_key

LOG = logging.getLogger("tre_controller.gateway_cadence")

TRACE_INTERVAL_KEYS = ("meta_interval_sec", "interval_in_seconds")
#: SCAN budget per model for trace keys (the metrics db also holds ~30k legacy keys).
TRACE_SCAN_COUNT = 1000
TRACE_SCAN_MAX_KEYS = 2000


class GatewayCadenceMismatch(RuntimeError):
    """The gateway writes at a different period than ``SCRAPE_INTERVAL_MS``."""


@dataclass
class CadenceEvidence:
    trace_interval_s: dict[str, float] = field(default_factory=dict)  # model -> seconds
    inst_spacing_ms: dict[str, float] = field(default_factory=dict)  # pod -> median ms
    errors: list[str] = field(default_factory=list)


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def read_trace_interval_s(redis_client: Any, model: str) -> Optional[float]:
    """``meta_interval_sec`` of the newest request trace of ``model``, or None."""
    newest_key: Optional[str] = None
    newest_ts = -1
    seen = 0
    for raw_key in redis_client.scan_iter(match=f"aibrix:{model}_request_trace_*", count=TRACE_SCAN_COUNT):
        key = _text(raw_key)
        seen += 1
        try:
            ts = int(key.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if ts > newest_ts:
            newest_key, newest_ts = key, ts
        if seen >= TRACE_SCAN_MAX_KEYS:
            break
    if newest_key is None:
        return None
    raw = redis_client.get(newest_key)
    if raw is None:
        return None
    doc = json.loads(_text(raw))
    for name in TRACE_INTERVAL_KEYS:
        if isinstance(doc, dict) and doc.get(name) is not None:
            return float(doc[name])
    return None


def read_inst_spacing_ms(redis_client: Any, model: str, *, samples: int = 7) -> dict[str, float]:
    """Median spacing of the newest instant ticks, per pod of ``model`` (>= 3 ticks)."""
    out: dict[str, float] = {}
    for raw_pod in redis_client.smembers(pods_key(model)):
        pod = _text(raw_pod)
        members = redis_client.zrange(inst_key(pod), -samples, -1, withscores=True)
        stamps = sorted(int(score) for _member, score in members)
        deltas = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
        if len(deltas) >= 2:
            out[pod] = float(median(deltas))
    return out


def gather_evidence(redis_client: Any, models: Iterable[str]) -> CadenceEvidence:
    evidence = CadenceEvidence()
    for model in models:
        try:
            interval = read_trace_interval_s(redis_client, model)
            if interval is not None:
                evidence.trace_interval_s[model] = interval
        except Exception as exc:  # noqa: BLE001 - evidence is best-effort.
            evidence.errors.append(f"trace:{model}:{type(exc).__name__}:{exc}")
        try:
            evidence.inst_spacing_ms.update(read_inst_spacing_ms(redis_client, model))
        except Exception as exc:  # noqa: BLE001
            evidence.errors.append(f"inst:{model}:{type(exc).__name__}:{exc}")
    return evidence


def check_gateway_cadence(
    redis_client: Any,
    models: Iterable[str],
    *,
    mode: str = "fail",
    expected_ms: int = SCRAPE_INTERVAL_MS,
) -> CadenceEvidence | None:
    """Assert the gateway period equals ``expected_ms`` (see module docstring)."""
    if mode == "off":
        return None
    evidence = gather_evidence(redis_client, list(models))
    expected_s = expected_ms / 1000.0
    problems: list[str] = []
    for model, interval_s in sorted(evidence.trace_interval_s.items()):
        if abs(interval_s - expected_s) > 1e-9:
            problems.append(
                f"trace metadata of {model}: meta_interval_sec={interval_s:g} s != "
                f"SCRAPE_INTERVAL_MS/1000={expected_s:g} s"
            )
    for pod, spacing_ms in sorted(evidence.inst_spacing_ms.items()):
        # Medians are exact multiples on a healthy ticker; allow 10 % for a skipped tick.
        if abs(spacing_ms - expected_ms) > 0.1 * expected_ms:
            problems.append(f"instant ticks of {pod}: median spacing {spacing_ms:g} ms != {expected_ms} ms")
    if problems:
        message = (
            "GATEWAY CADENCE MISMATCH - the gateway write period is not SCRAPE_INTERVAL_MS; "
            "queue averages, the phase-aligned sampler and every fitted threshold assume it. "
            + "; ".join(problems)
        )
        if mode == "fail":
            raise GatewayCadenceMismatch(message)
        LOG.error(message)
        return evidence
    if not evidence.trace_interval_s and not evidence.inst_spacing_ms:
        LOG.warning(
            "!!! GATEWAY CADENCE UNVERIFIED !!! no request-trace metadata (meta_interval_sec) and "
            "no instant ticks to measure; assuming the gateway writes every %d ms. A rebuilt "
            "gateway with RequestTraceWriteInterval != %g s would silently break the signal. "
            "errors=%s", expected_ms, expected_s, evidence.errors,
        )
        return evidence
    if not evidence.trace_interval_s:
        LOG.warning(
            "!!! GATEWAY TRACE METADATA UNAVAILABLE !!! (no aibrix:<model>_request_trace_* with "
            "meta_interval_sec; the gateway only writes traces under traffic). Instant tick "
            "spacing agrees with %d ms over %d pods; re-check once traffic flows.",
            expected_ms, len(evidence.inst_spacing_ms),
        )
        return evidence
    LOG.info(
        "gateway_cadence_ok: expected %d ms; trace meta_interval_sec=%s; instant tick spacing "
        "(ms) over %d pods=%s",
        expected_ms,
        evidence.trace_interval_s,
        len(evidence.inst_spacing_ms),
        sorted(set(evidence.inst_spacing_ms.values())),
    )
    return evidence

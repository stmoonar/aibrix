"""SafeScale commit evidence (2026-09-29): what the probe's latency gate is judged on.

Until 2026-09-29 the commit gate read the tail of the probe's snapshot observations.
A snapshot is the phase-aligned 30 s window ``(B - 30 s, B]`` published every 10 s, so
with a 20 s probe more than half of the judged latency predated the hide. The latency
part of the gate now reads ONE evidence window instead:

* start ``S`` = the first gateway boundary after the hide, anchored on the gateway's
  own doc stamps: ``S = N + period`` with ``N`` = the newest histogram doc stamp of the
  model when the SM confirmed the hide. A doc stamped ``S`` is written after the hide,
  whatever the node clocks say (they differ by up to 160 s here), so no clock enters
  the evidence. Without ``N`` (no doc yet) ``S = ceil(hide_ts / period)``;
* the window holds the docs stamped in ``[S, E]`` (read without the histogram lookback,
  so a pod's delta never starts from a pre-hide doc); ``E`` = the newest snapshot's
  boundary (the same doc-stamp grid the snapshots read);
* pods: the model's pods minus the probe's hidden pods and minus the pods the fleet
  state reports asleep (a pod the fleet state does not list is kept);
* TTFT / TPOT p95 = max over those pods (the snapshot rule, per-pod minimum samples
  ``TRE_MIN_LATENCY_SAMPLES``), ``n`` = their TTFT count, ``n_judged`` = the TTFT count
  of the pods that have a p95, mean prompt length ``L`` = their
  ``request_prompt_tokens`` sum / count delta.

``hide_ts`` is Redis ``TIME`` read right after the SM confirmed the hide (the reference
the service-manager's clock-skew check uses too), else the controller clock. It is
kept for the audit and the fallback above; the offsets of the gateway stamps and of the
controller clock against it are recorded and alerted on (``safescale_clock_skew_alert``).

Used only with ``safescale.evidence_source: redis`` (the rollback switch). Known
limitation: the gateway writes a doc for every ready pod every period from its cached
scrape (``pkg/cache/cache_tre_redis.go``), with a fresh stamp even when that scrape is
stale, so a doc's stamp does not prove its counters are fresh. The direct path does
not have this problem and never commits on this evidence.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Mapping, Protocol

from tre_common.rediskeys import hist_key, pods_key
from tre_common.window_pods import pooled_p95_ms

LOG = logging.getLogger("tre_controller.safescale")

#: Threshold modes recorded in the probe audit.
THRESHOLD_MODE_LABELS = "labels"
THRESHOLD_MODE_FIXED = "fixed"
#: Direct constructions without a registry (tests / offline): config values or 500 / 75.
THRESHOLD_MODE_CONFIG = "config"
DEFAULT_TTFT_SLO_MS = 500.0
DEFAULT_TPOT_SLO_MS = 75.0


@dataclass(frozen=True)
class HideAnchor:
    """When the probe's hide took effect (the SM confirmed it)."""

    ts_ms: int
    #: ``redis_time`` (normal) or ``controller_clock`` (Redis TIME failed).
    source: str
    #: Newest gateway histogram doc stamp of the model at the hide (None = no doc yet).
    newest_doc_ts_ms: int | None = None
    #: The controller's own clock at the same moment (audit / skew alert).
    controller_ts_ms: int | None = None
    #: The newest-doc read failed: the evidence start cannot be anchored on the gateway
    #: stamps (the probe rolls back, fail-closed).
    newest_doc_error: bool = False

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ts_ms": self.ts_ms,
            "source": self.source,
            "newest_doc_ts_ms": self.newest_doc_ts_ms,
        }
        if self.controller_ts_ms is not None:
            record["controller_ts_ms"] = self.controller_ts_ms
        if self.newest_doc_error:
            record["newest_doc_error"] = True
        return record

    @classmethod
    def from_record(cls, raw: Any) -> "HideAnchor | None":
        if not isinstance(raw, Mapping):
            return None
        try:
            ts_ms = int(float(raw["ts_ms"]))
        except (KeyError, TypeError, ValueError):
            return None
        return cls(
            ts_ms=ts_ms,
            source=str(raw.get("source") or "unknown"),
            newest_doc_ts_ms=_optional_ms(raw.get("newest_doc_ts_ms")),
            controller_ts_ms=_optional_ms(raw.get("controller_ts_ms")),
            newest_doc_error=bool(raw.get("newest_doc_error", False)),
        )


def evidence_start(anchor: HideAnchor, period_ms: int) -> int:
    """First gateway boundary after the hide: ``newest doc stamp + period`` (gateway
    clock, no cross-node clock involved), else ``ceil(hide_ts / period)``."""
    if anchor.newest_doc_ts_ms is not None:
        return int(anchor.newest_doc_ts_ms) + int(period_ms)
    return ceil_boundary(anchor.ts_ms, period_ms)


def anchor_reference_ms(anchor: HideAnchor) -> int:
    """The hide in the doc-stamp domain: the newest doc stamp at the hide, else hide_ts.
    The first evidence doc must lie in [evidence start, this + tolerance]."""
    return int(anchor.newest_doc_ts_ms) if anchor.newest_doc_ts_ms is not None else int(anchor.ts_ms)


def anchor_clock_offsets(anchor: HideAnchor, *, period_ms: int, tolerance_ms: float) -> dict[str, Any]:
    """Offsets of the gateway stamps and of the controller clock against hide_ts.

    ``gateway_offset_ms`` = hide_ts - newest doc stamp: 0 .. one period (+ write delay)
    when the clocks agree. ``controller_offset_ms`` = controller clock - hide_ts: ~0.
    ``clock_skew_alert`` is True when either is off by more than ``tolerance_ms``. Alert
    only: the evidence window is anchored on the doc stamps and does not depend on it."""
    gateway = None if anchor.newest_doc_ts_ms is None else int(anchor.ts_ms) - int(anchor.newest_doc_ts_ms)
    controller = None if anchor.controller_ts_ms is None else int(anchor.controller_ts_ms) - int(anchor.ts_ms)
    alert = (gateway is not None and not (-tolerance_ms <= gateway <= period_ms + tolerance_ms)) or (
        controller is not None and anchor.source == "redis_time" and abs(controller) > tolerance_ms
    )
    return {"gateway_offset_ms": gateway, "controller_offset_ms": controller, "clock_skew_alert": bool(alert)}


@dataclass(frozen=True)
class EvidenceWindow:
    """The remaining pods' latency evidence over the docs stamped ``[start_ms, end_ms]``."""

    start_ms: int
    end_ms: int
    pods: tuple[str, ...]
    excluded_pods: tuple[str, ...]
    ttft_p95_ms: float | None
    tpot_p95_ms: float | None
    #: n: completed requests (TTFT count delta) of the remaining pods.
    ttft_count: float
    prompt_tokens: float | None = None
    prompt_count: float | None = None
    #: pod -> stamp (ms) of the first histogram doc its TTFT delta starts from.
    first_doc_ts_ms: dict[str, int] = field(default_factory=dict)
    #: TTFT count of the pods whose p95 is defined (the requests the p95 judges).
    #: None = same as ttft_count.
    judged_count: float | None = None
    #: pod -> stamp (ms) of the last histogram doc its TTFT delta ends at.
    last_doc_ts_ms: dict[str, int] = field(default_factory=dict)
    #: p95 of the remaining pods' histograms pooled first (per-pod minimum not applied
    #: per pod, but to the pooled count); already folded into ttft/tpot_p95_ms (max).
    pooled_ttft_p95_ms: float | None = None
    pooled_tpot_p95_ms: float | None = None
    #: max(per pod, pooled) p95 WITHOUT any minimum-samples rule: the ceiling's
    #: low-sample evaluation. None = not computed (then the p95s above stand in).
    low_ttft_p95_ms: float | None = None
    low_tpot_p95_ms: float | None = None

    @property
    def mean_prompt_tokens(self) -> float | None:
        if self.prompt_tokens is None or not self.prompt_count or self.prompt_count <= 0:
            return None
        return float(self.prompt_tokens) / float(self.prompt_count)

    @property
    def judged(self) -> float:
        return float(self.ttft_count if self.judged_count is None else self.judged_count)


class EvidenceSource(Protocol):
    def hide_anchor(self, model: str) -> HideAnchor: ...

    def read(self, model: str, *, start_ms: int, end_ms: int, exclude_pods: Collection[str]) -> EvidenceWindow: ...


class ThresholdResolver(Protocol):
    def resolve(self, model: str, mean_prompt_tokens: float | None) -> dict[str, Any]: ...


def ceil_boundary(ts_ms: int, period_ms: int) -> int:
    """The first gateway boundary at or after ``ts_ms``."""
    period = int(period_ms)
    return -(-int(ts_ms) // period) * period


class MetricsEvidenceReader:
    """:class:`EvidenceSource` over a :class:`MetricsStore` built for it: same redis,
    registry and percentile / minimum-sample rules as the controller's store, but
    ``histogram_lookback_ms=0`` so a delta starts at the first doc stamped >= S."""

    def __init__(
        self,
        store: Any,
        *,
        redis_client: Any | None = None,
        sleeping_pods: Callable[[str], Collection[str]] | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        lookback = getattr(store, "histogram_lookback_ms", 0)
        if lookback:
            raise ValueError(
                f"the evidence store must read without histogram lookback (got {lookback} ms): "
                "a lookback baseline would start the delta at a pre-hide doc"
            )
        self._store = store
        self._redis = redis_client if redis_client is not None else getattr(store, "redis_client", None)
        self._sleeping_pods = sleeping_pods
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))

    def hide_anchor(self, model: str) -> HideAnchor:
        ts_ms, source = self._now()
        newest, error = self._newest_doc_ts(model)
        return HideAnchor(
            ts_ms=ts_ms, source=source, newest_doc_ts_ms=newest,
            controller_ts_ms=int(self._clock_ms()), newest_doc_error=error,
        )

    def read(self, model: str, *, start_ms: int, end_ms: int, exclude_pods: Collection[str]) -> EvidenceWindow:
        # (S - 1, E] half-open == docs stamped [S, E]; no lookback -> no pre-S baseline.
        metrics = self._store.read_model_window(
            model, int(start_ms) - 1, int(end_ms), use_cache=False, start_exclusive=True
        )
        excluded = set(exclude_pods)
        if self._sleeping_pods is not None:
            try:
                excluded |= set(self._sleeping_pods(model) or ())
            except Exception:  # noqa: BLE001 - no fleet view: only the probe pods are dropped
                LOG.warning("safescale evidence: sleeping pods of %s unavailable", model, exc_info=True)
        per_pod = getattr(metrics, "per_pod", None) or {}
        pods = {
            key: pod
            for key, pod in per_pod.items()
            if key not in excluded and getattr(pod, "pod", key) not in excluded
        }
        dropped = tuple(sorted(str(getattr(pod, "pod", key)) for key, pod in per_pod.items() if key not in pods))
        first_docs = {
            str(getattr(pod, "pod", key)): int(pod.hist_first_ts_ms)
            for key, pod in pods.items()
            if getattr(pod, "hist_first_ts_ms", None) is not None
        }
        judged = sum(
            float(getattr(pod, "ttft_count", None) or 0.0)
            for pod in pods.values()
            if pod.ttft_p95_ms is not None or pod.tpot_p95_ms is not None
        )
        last_docs = {
            str(getattr(pod, "pod", key)): int(pod.hist_last_ts_ms)
            for key, pod in pods.items()
            if getattr(pod, "hist_last_ts_ms", None) is not None
        }
        # Pooled p95 next to the per-pod maximum: a pod below the per-pod minimum
        # samples (an overloaded pod completes few requests) still weighs in.
        rule = getattr(self._store, "p95_rule", None) or ("bucket_upper", 0)
        pooled_ttft = pooled_p95_ms(
            ((getattr(pod, "ttft_hist", None), getattr(pod, "ttft_hist_count", None)) for pod in pods.values()), rule
        )
        pooled_tpot = pooled_p95_ms(
            ((getattr(pod, "tpot_hist", None), getattr(pod, "tpot_hist_count", None)) for pod in pods.values()), rule
        )
        total = float(sum(float(getattr(pod, "ttft_count", None) or 0.0) for pod in pods.values()))
        if pooled_ttft is not None or pooled_tpot is not None:
            judged = total  # every request is in the pooled p95
        # The same max(per pod, pooled) without any minimum-samples rule (low-sample
        # evaluation at the ceiling).
        rule0 = (rule[0], 0)

        def low(hist_attr: str, count_attr: str) -> float | None:
            pairs = [(getattr(pod, hist_attr, None), getattr(pod, count_attr, None)) for pod in pods.values()]
            return _max_present([*(pooled_p95_ms([pair], rule0) for pair in pairs), pooled_p95_ms(pairs, rule0)])
        return EvidenceWindow(
            start_ms=int(start_ms),
            end_ms=int(end_ms),
            pods=tuple(sorted(str(getattr(pod, "pod", key)) for key, pod in pods.items())),
            excluded_pods=dropped,
            ttft_p95_ms=_max_present([*(pod.ttft_p95_ms for pod in pods.values()), pooled_ttft]),
            tpot_p95_ms=_max_present([*(pod.tpot_p95_ms for pod in pods.values()), pooled_tpot]),
            ttft_count=total,
            prompt_tokens=_sum_present(getattr(pod, "prompt_tokens", None) for pod in pods.values()),
            prompt_count=_sum_present(getattr(pod, "request_count", None) for pod in pods.values()),
            first_doc_ts_ms=first_docs,
            judged_count=float(judged),
            last_doc_ts_ms=last_docs,
            pooled_ttft_p95_ms=pooled_ttft,
            pooled_tpot_p95_ms=pooled_tpot,
            low_ttft_p95_ms=low("ttft_hist", "ttft_hist_count"),
            low_tpot_p95_ms=low("tpot_hist", "tpot_hist_count"),
        )

    def _now(self) -> tuple[int, str]:
        client = self._redis
        if client is not None and callable(getattr(client, "time", None)):
            try:
                seconds, micros = client.time()
                return int(int(seconds) * 1000 + int(micros) // 1000), "redis_time"
            except Exception:  # noqa: BLE001 - fall back to the local clock (recorded)
                LOG.warning("safescale hide anchor: Redis TIME failed; using the controller clock", exc_info=True)
        return int(self._clock_ms()), "controller_clock"

    def _newest_doc_ts(self, model: str) -> tuple[int | None, bool]:
        """(newest histogram doc stamp of the model's pods or None, read failed)."""
        client = self._redis
        if client is None or getattr(self._store, "schema", "v2") != "v2":
            # No gateway stamps to anchor on (legacy v1 keys / no redis): unverifiable.
            return None, True
        try:
            newest: int | None = None
            for raw_pod in client.smembers(pods_key(model)) or ():
                pod = raw_pod.decode() if isinstance(raw_pod, bytes) else str(raw_pod)
                rows = client.zrevrangebyscore(hist_key(pod), "+inf", "-inf", start=0, num=1, withscores=True)
                for _, score in rows or ():
                    stamp = int(float(score))
                    newest = stamp if newest is None else max(newest, stamp)
            return newest, False
        except Exception:  # noqa: BLE001 - fail-closed downstream (newest_doc_error)
            LOG.warning("safescale hide anchor: newest gateway doc of %s unreadable", model, exc_info=True)
            return None, True


def log_clock_skew_alert(model: str, request_id: str, anchor: HideAnchor, offsets: Mapping[str, Any]) -> None:
    LOG.error(
        json.dumps(
            {"event": "safescale_clock_skew_alert", "model": model, "request_id": request_id,
             **anchor.as_record(), **offsets},
            sort_keys=True,
        )
    )


def _optional_ms(value: Any) -> int | None:
    try:
        return int(float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


class RegistryThresholds:
    """:class:`ThresholdResolver` from the registry (``safescale.slo_mode``).

    * ``labels`` (default): the calibration label's rule, via the same function the
      theta labels use (``tre_common.slo_labels.label_def_for_model``): TPOT fixed,
      TTFT = max(floor, k * (c + b * L)) for the slowdown label (L = the evidence
      window's mean prompt length; without L the TTFT threshold is the floor).
    * ``fixed``: ``models[].slo.ttft_p95_ms`` / ``tpot_p95_ms`` (v1's 500 / 75).

    ``ttft_override_ms`` / ``tpot_override_ms`` (env SAFE_SCALE_*_P95_SLO_MS, unset by
    default) replace the respective threshold in either mode. A model whose registry
    ``slo`` block cannot build a label falls back to ``fixed`` (recorded)."""

    def __init__(
        self,
        registry: Any,
        *,
        mode: str = THRESHOLD_MODE_LABELS,
        ttft_override_ms: float | None = None,
        tpot_override_ms: float | None = None,
    ) -> None:
        if mode not in (THRESHOLD_MODE_LABELS, THRESHOLD_MODE_FIXED):
            raise ValueError(f"unknown SafeScale threshold mode {mode!r}")
        self._registry = registry
        self._mode = mode
        self._ttft_override = ttft_override_ms
        self._tpot_override = tpot_override_ms
        self._labels: dict[str, Any] = {}

    def resolve(self, model: str, mean_prompt_tokens: float | None) -> dict[str, Any]:
        mode = self._mode
        fallback: str | None = None
        ttft: float
        tpot: float
        if mode == THRESHOLD_MODE_LABELS:
            label = self._label(model)
            if isinstance(label, str):
                fallback, mode = label, THRESHOLD_MODE_FIXED
            else:
                tpot = float(label.tpot_p95_ms)
                if label.slowdown:
                    length = _finite_nonneg(mean_prompt_tokens)
                    ttft = float(label.ttft_slo_ms(length)) if length is not None else float(label.ttft_floor_ms)
                else:
                    ttft = float(label.ttft_p95_ms)
        if mode == THRESHOLD_MODE_FIXED:
            ttft, tpot = self._fixed(model)
        record: dict[str, Any] = {
            "mode": mode,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "mean_prompt_tokens": _finite_nonneg(mean_prompt_tokens),
            "source": "registry",
        }
        if fallback is not None:
            record["fallback"] = fallback
        if self._ttft_override is not None or self._tpot_override is not None:
            record["source"] = "env_override"
            if self._ttft_override is not None:
                record["ttft_ms"] = float(self._ttft_override)
            if self._tpot_override is not None:
                record["tpot_ms"] = float(self._tpot_override)
        return record

    def _label(self, model: str) -> Any:
        if model not in self._labels:
            from tre_common.slo_labels import label_def_for_model

            try:
                self._labels[model] = label_def_for_model(model, registry=self._registry)
            except (SystemExit, ValueError, KeyError) as exc:
                LOG.warning("safescale thresholds: no label for %s (%s); using models[].slo", model, exc)
                self._labels[model] = f"label_unavailable: {exc}"
        return self._labels[model]

    def _fixed(self, model: str) -> tuple[float, float]:
        try:
            slo = self._registry.model(model).slo
            return float(slo.ttft_p95_ms), float(slo.tpot_p95_ms)
        except (KeyError, AttributeError, TypeError, ValueError):
            return DEFAULT_TTFT_SLO_MS, DEFAULT_TPOT_SLO_MS


def config_thresholds(config: Any, mean_prompt_tokens: float | None = None) -> dict[str, Any]:
    """Thresholds without a registry resolver: the config values, else 500 / 75 ms."""
    ttft = getattr(config, "ttft_p95_slo_ms", None)
    tpot = getattr(config, "tpot_p95_slo_ms", None)
    return {
        "mode": THRESHOLD_MODE_CONFIG,
        "ttft_ms": float(ttft) if ttft is not None else DEFAULT_TTFT_SLO_MS,
        "tpot_ms": float(tpot) if tpot is not None else DEFAULT_TPOT_SLO_MS,
        "mean_prompt_tokens": _finite_nonneg(mean_prompt_tokens),
        "source": "config",
    }


def _finite_nonneg(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _max_present(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    return max(present) if present else None


def _sum_present(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) if present else None

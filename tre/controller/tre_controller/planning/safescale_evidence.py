"""SafeScale commit evidence (2026-09-29): what the probe's latency gate is judged on.

Until 2026-09-29 the commit gate read the tail of the probe's snapshot observations.
A snapshot is the phase-aligned 30 s window ``(B - 30 s, B]`` published every 10 s, so
with a 20 s probe more than half of the judged latency predated the hide. The latency
part of the gate now reads ONE evidence window instead:

* start ``S`` = the first gateway boundary after the hide, anchored on the gateway's
  own doc stamps: the boundary after ``N``, the newest histogram doc stamp of the
  model when the SM confirmed the hide. A doc stamped ``S`` is written after the hide,
  whatever the node clocks say (they differ by up to 160 s here), so no clock enters
  the evidence. Without ``N`` (no doc yet) the first boundary at or after hide_ts. The
  doc grid is the store's (``tre_controller.store.evidence_grid.EvidenceGrid``);
* the window holds the docs stamped in ``[S, E]`` (read without the histogram lookback,
  so a pod's delta never starts from a pre-hide doc); ``E`` = the newest snapshot's
  boundary (the same doc-stamp grid the snapshots read);
* pods: the model's pods minus the probe's hidden pods and minus the pods the fleet
  state reports asleep (a pod the fleet state does not list is kept);
* TPOT p95 = max over those pods and their pooled histogram (the snapshot rule, per-pod
  minimum samples ``TRE_MIN_LATENCY_SAMPLES``), ``n`` = their TTFT count (the TTFT
  histogram count is the request counter: requests that got their first token),
  ``n_judged`` = the TTFT count of the pods that have a TPOT p95. SafeScale judges TPOT
  only (2026-11-04): TTFT is not a latency verdict of the probe, so no TTFT p95 and no
  prompt length are formed here.

``hide_ts`` is Redis ``TIME`` read right after the SM confirmed the hide (the reference
the service-manager's clock-skew check uses too), else the controller clock. It is
kept for the audit and the fallback above; the offsets of the gateway stamps and of the
controller clock against it are recorded and alerted on (``safescale_clock_skew_alert``).

Used only with ``safescale.evidence_source: redis`` (the rollback switch). Known
limitation: the gateway writes a doc for every ready pod every round from its cached
scrape (``pkg/cache/cache_tre_redis.go``), with a fresh stamp even when that scrape is
stale, so a doc's stamp does not prove its counters are fresh. The direct path does
not have this problem and never commits on this evidence.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Mapping, Protocol

from tre_common import slo as slo_def
from tre_common.rediskeys import hist_key, pods_key
from tre_common.window_pods import pooled_p95_ms

LOG = logging.getLogger("tre_controller.safescale")

#: Threshold modes recorded in the probe audit: the ``tre_common.slo`` mode
#: (``fixed`` / ``slowdown`` / ``idle_multiple``; before 2026-11-04 the audit wrote
#: ``labels`` for what is now ``slowdown``), or ``static`` (:class:`StaticThresholds`).
#: There is no built-in threshold (2026-11-04): the controller judges with
#: :class:`RegistryThresholds`; every other construction names its thresholds.
THRESHOLD_MODE_STATIC = "static"


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


def anchor_reference_ms(anchor: HideAnchor) -> int:
    """The hide in the doc-stamp domain: the newest doc stamp at the hide, else hide_ts.
    The first evidence doc must lie in [evidence start, this + tolerance]."""
    return int(anchor.newest_doc_ts_ms) if anchor.newest_doc_ts_ms is not None else int(anchor.ts_ms)


@dataclass(frozen=True)
class EvidenceWindow:
    """The remaining pods' latency evidence over the docs stamped ``[start_ms, end_ms]``."""

    start_ms: int
    end_ms: int
    pods: tuple[str, ...]
    excluded_pods: tuple[str, ...]
    tpot_p95_ms: float | None
    #: n: completed requests (TTFT count delta) of the remaining pods.
    ttft_count: float
    #: pod -> stamp (ms) of the first histogram doc its TTFT delta starts from.
    first_doc_ts_ms: dict[str, int] = field(default_factory=dict)
    #: TTFT count of the pods whose TPOT p95 is defined (the requests the p95 judges).
    #: None = same as ttft_count.
    judged_count: float | None = None
    #: pod -> stamp (ms) of the last histogram doc its TTFT delta ends at.
    last_doc_ts_ms: dict[str, int] = field(default_factory=dict)
    #: TPOT p95 of the remaining pods' histograms pooled first (per-pod minimum not
    #: applied per pod, but to the pooled count); already folded into tpot_p95_ms (max).
    pooled_tpot_p95_ms: float | None = None
    #: max(per pod, pooled) TPOT p95 WITHOUT any minimum-samples rule: the ceiling's
    #: low-sample evaluation. None = not computed (then tpot_p95_ms stands in).
    low_tpot_p95_ms: float | None = None

    @property
    def judged(self) -> float:
        return float(self.ttft_count if self.judged_count is None else self.judged_count)


class EvidenceSource(Protocol):
    def hide_anchor(self, model: str) -> HideAnchor: ...

    def read(self, model: str, *, start_ms: int, end_ms: int, exclude_pods: Collection[str]) -> EvidenceWindow: ...


class ThresholdResolver(Protocol):
    def resolve(self, model: str) -> dict[str, Any]: ...


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
            model, int(start_ms) - 1, int(end_ms), start_exclusive=True
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
            if pod.tpot_p95_ms is not None
        )
        last_docs = {
            str(getattr(pod, "pod", key)): int(pod.hist_last_ts_ms)
            for key, pod in pods.items()
            if getattr(pod, "hist_last_ts_ms", None) is not None
        }
        # Pooled p95 next to the per-pod maximum: a pod below the per-pod minimum
        # samples (an overloaded pod completes few requests) still weighs in.
        rule = getattr(self._store, "p95_rule", None) or ("bucket_upper", 0)
        pooled_tpot = pooled_p95_ms(
            ((getattr(pod, "tpot_hist", None), getattr(pod, "tpot_hist_count", None)) for pod in pods.values()), rule
        )
        total = float(sum(float(getattr(pod, "ttft_count", None) or 0.0) for pod in pods.values()))
        if pooled_tpot is not None:
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
            tpot_p95_ms=_max_present([*(pod.tpot_p95_ms for pod in pods.values()), pooled_tpot]),
            ttft_count=total,
            first_doc_ts_ms=first_docs,
            judged_count=float(judged),
            last_doc_ts_ms=last_docs,
            pooled_tpot_p95_ms=pooled_tpot,
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
        if client is None:
            # No gateway stamps to anchor on (no redis): unverifiable.
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
    """:class:`ThresholdResolver` from the registry: the TPOT threshold of the registry's
    ``slo_mode`` (the one switch of every consumer), built by ``tre_common.slo``.

    What the probe judges (user 2026-11-04, a design change): the evidence window's p95
    TPOT (vLLM inter-token-latency histograms of the remaining pods, max of per pod and
    pooled) against the mode's TPOT_slo - ``tpot_p95_ms`` in ``fixed`` / ``slowdown``,
    ``tpot_k * tpot_idle_ms`` in ``idle_multiple``. TTFT is not judged by SafeScale (no
    rollback, no blocked commit): the probe uses the TPOT component of the definition
    only; the scorer and the labels still judge both components.

    ``mode`` (tests) replaces the registry's mode. Fail closed: a model whose ``slo``
    block cannot build the mode's thresholds raises (``load_registry`` validates every
    model's block for its mode, so a loaded registry never gets here); there is no
    fallback to another mode and no env override (SAFE_SCALE_*_P95_SLO_MS removed
    2026-11-04)."""

    def __init__(self, registry: Any, *, mode: str | None = None) -> None:
        mode = mode or registry.slo_mode()
        if mode not in slo_def.SLO_MODES:
            raise ValueError(f"unknown SafeScale threshold mode {mode!r}")
        self._registry = registry
        self._mode = mode
        self._thresholds: dict[str, Any] = {}

    @property
    def mode(self) -> str:
        return self._mode

    def resolve(self, model: str) -> dict[str, Any]:
        thresholds = self._of(model)
        return {"mode": thresholds.mode, "tpot_ms": float(thresholds.tpot_ms), "source": "registry"}

    def _of(self, model: str) -> Any:
        if model not in self._thresholds:
            # raises (SloConfigError / KeyError): no threshold without the registry's
            self._thresholds[model] = slo_def.slo_thresholds(self._registry.model(model).slo, self._mode)
        return self._thresholds[model]


class StaticThresholds:
    """:class:`ThresholdResolver` with one TPOT threshold for every model (offline
    replays and tests; the caller names the value - there is no default)."""

    def __init__(self, tpot_ms: float) -> None:
        self._tpot_ms = float(tpot_ms)

    def resolve(self, model: str) -> dict[str, Any]:
        return {"mode": THRESHOLD_MODE_STATIC, "tpot_ms": self._tpot_ms, "source": "static"}


def _max_present(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    return max(present) if present else None

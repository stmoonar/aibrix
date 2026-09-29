"""SafeScale commit evidence (2026-09-29): what the probe's latency gate is judged on.

Until 2026-09-29 the commit gate read the tail of the probe's snapshot observations.
A snapshot is the phase-aligned 30 s window ``(B - 30 s, B]`` published every 10 s, so
with a 20 s probe more than half of the judged latency predated the hide. The latency
part of the gate now reads ONE evidence window instead:

* start ``S`` = the first gateway boundary at or after the hide (``ceil(hide / 10 s)``),
  read half-open ``(S, E]`` so ``MetricsStore._with_baseline_doc`` takes the doc stamped
  ``S`` as the delta's baseline (the ``+1 ms`` of ``start_exclusive``); ``E`` = the
  newest snapshot's boundary;
* pods: the model's pods minus the probe's hidden pods and minus the pods the fleet
  state reports asleep (a pod the fleet state does not list is kept);
* TTFT / TPOT p95 = max over those pods (the snapshot rule), ``n`` = their TTFT count,
  mean prompt length ``L`` = their ``request_prompt_tokens`` sum / count delta.

The hide anchor is Redis ``TIME`` read right after the SM confirmed the hide: one
clock for every controller replica and node (node clocks here differ by up to 160 s),
the clock of the store the gateway docs live in, and the reference the service-manager's
own startup skew check already uses. At the same moment the newest gateway doc stamp of
the model is recorded: a doc stamped at or after ``S`` that already exists at the hide
can only come from a gateway clock running ahead, i.e. the evidence baseline would
predate the hide.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Mapping, Protocol

from tre_common.rediskeys import hist_key, pods_key

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
    """When the probe's hide took effect."""

    ts_ms: int
    #: ``redis_time`` (normal) or ``controller_clock`` (Redis TIME failed).
    source: str
    #: Newest gateway histogram doc stamp of the model at the hide (None = unknown).
    newest_doc_ts_ms: int | None = None

    def as_record(self) -> dict[str, Any]:
        return {"ts_ms": self.ts_ms, "source": self.source, "newest_doc_ts_ms": self.newest_doc_ts_ms}

    @classmethod
    def from_record(cls, raw: Any) -> "HideAnchor | None":
        if not isinstance(raw, Mapping):
            return None
        try:
            ts_ms = int(float(raw["ts_ms"]))
        except (KeyError, TypeError, ValueError):
            return None
        newest = raw.get("newest_doc_ts_ms")
        try:
            newest_ms = int(float(newest)) if newest is not None else None
        except (TypeError, ValueError):
            newest_ms = None
        return cls(ts_ms=ts_ms, source=str(raw.get("source") or "unknown"), newest_doc_ts_ms=newest_ms)


@dataclass(frozen=True)
class EvidenceWindow:
    """The remaining pods' latency evidence over ``(start_ms, end_ms]``."""

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

    @property
    def mean_prompt_tokens(self) -> float | None:
        if self.prompt_tokens is None or not self.prompt_count or self.prompt_count <= 0:
            return None
        return float(self.prompt_tokens) / float(self.prompt_count)


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
    """:class:`EvidenceSource` over the controller's :class:`MetricsStore`."""

    def __init__(
        self,
        store: Any,
        *,
        redis_client: Any | None = None,
        sleeping_pods: Callable[[str], Collection[str]] | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._store = store
        self._redis = redis_client if redis_client is not None else getattr(store, "redis_client", None)
        self._sleeping_pods = sleeping_pods
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))

    def hide_anchor(self, model: str) -> HideAnchor:
        ts_ms, source = self._now()
        return HideAnchor(ts_ms=ts_ms, source=source, newest_doc_ts_ms=self._newest_doc_ts(model))

    def read(self, model: str, *, start_ms: int, end_ms: int, exclude_pods: Collection[str]) -> EvidenceWindow:
        metrics = self._store.read_model_window(
            model, int(start_ms), int(end_ms), use_cache=False, start_exclusive=True
        )
        excluded = set(exclude_pods)
        if self._sleeping_pods is not None:
            try:
                excluded |= set(self._sleeping_pods(model) or ())
            except Exception:  # noqa: BLE001 - no fleet view: only the probe pods are dropped
                LOG.warning("safescale evidence: sleeping pods of %s unavailable", model, exc_info=True)
        pods = {
            key: pod
            for key, pod in (getattr(metrics, "per_pod", None) or {}).items()
            if key not in excluded and getattr(pod, "pod", key) not in excluded
        }
        dropped = tuple(sorted(
            getattr(pod, "pod", key)
            for key, pod in (getattr(metrics, "per_pod", None) or {}).items()
            if key not in pods
        ))
        first_docs = {
            str(getattr(pod, "pod", key)): int(pod.hist_first_ts_ms)
            for key, pod in pods.items()
            if getattr(pod, "hist_first_ts_ms", None) is not None
        }
        return EvidenceWindow(
            start_ms=int(start_ms),
            end_ms=int(end_ms),
            pods=tuple(sorted(str(getattr(pod, "pod", key)) for key, pod in pods.items())),
            excluded_pods=dropped,
            ttft_p95_ms=_max_present(pod.ttft_p95_ms for pod in pods.values()),
            tpot_p95_ms=_max_present(pod.tpot_p95_ms for pod in pods.values()),
            ttft_count=float(sum(float(getattr(pod, "ttft_count", None) or 0.0) for pod in pods.values())),
            prompt_tokens=_sum_present(getattr(pod, "prompt_tokens", None) for pod in pods.values()),
            prompt_count=_sum_present(getattr(pod, "request_count", None) for pod in pods.values()),
            first_doc_ts_ms=first_docs,
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

    def _newest_doc_ts(self, model: str) -> int | None:
        client = self._redis
        if client is None or getattr(self._store, "schema", "v2") != "v2":
            return None
        try:
            newest: int | None = None
            for raw_pod in client.smembers(pods_key(model)) or ():
                pod = raw_pod.decode() if isinstance(raw_pod, bytes) else str(raw_pod)
                rows = client.zrevrangebyscore(hist_key(pod), "+inf", "-inf", start=0, num=1, withscores=True)
                for _, score in rows or ():
                    stamp = int(float(score))
                    newest = stamp if newest is None else max(newest, stamp)
            return newest
        except Exception:  # noqa: BLE001 - the cross-check is skipped (recorded as None)
            LOG.warning("safescale hide anchor: newest gateway doc of %s unreadable", model, exc_info=True)
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

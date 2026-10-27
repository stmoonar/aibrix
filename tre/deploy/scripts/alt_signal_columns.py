#!/usr/bin/env python3
"""Window columns of the v1 ablation signals ``ttft_mean`` and ``kv_usage`` from the 1 Hz
per-pod vLLM capture (``cells/<stem>/vllm_metrics_1hz/``, :mod:`scripts.calibration_capture`).

The controller reads both from the gateway docs of every serving pod
(``tre_controller.signals.sources``): the pod's TTFT histogram ``sum`` / ``count`` deltas
over the window and the window mean of its ``kv_cache_usage_perc`` gauge. The dataset
builder replays the same per-pod rules on the capture so ``fit_alt_thresholds`` fits
theta / delta on the values the controller will compute:

* ``ttft_server_mean_ms`` / ``ttft_server_count``: per pod, the boundary sample = the last
  sample at or before each window boundary (at most ``max_gap_ms`` old; online the
  baseline is the newest histogram doc at or before the window start), the
  ``vllm:time_to_first_token_seconds`` ``_sum`` and ``_count`` deltas between them; the
  model value = sum of the sums / sum of the counts (v1 D38: pooled by first-token count),
  in ms. Count 0 = no first token in the window: the mean is empty (no value), the count 0;
* ``kv_cache_usage_sum`` / ``kv_cache_pods``: per serving pod, the mean of ALL its
  ``vllm:kv_cache_usage_perc`` samples in ``(start, end]`` (``gpu_cache_usage_perc`` on
  engines before 0.11); the sum over the serving pods and their number. A pod serves when
  it is awake in its last sample of the window (``vllm:engine_sleep_state{sleep_state=
  "awake"}`` not 0): online the controller keeps the pods the fleet state reports awake
  at the tick and averages every instant doc of the window. Review L4 (2026-10-28):
  samples of a pod woken mid-window taken before the wake (fill ~0) are therefore
  averaged in, as online - the gateway docs carry no sleep state, so the controller
  cannot drop them, and the offline rule follows the online one exactly (it used to drop
  them: online / offline differed in every window holding a wake). O1 holds such
  windows on every arm (a routable-count change), so a decision rarely reads one.

Void (all four columns empty: unknown, never zero) when the cell has no capture or a
pod expected by ``cell_meta.json`` has no file, and per pod: no boundary sample / a zero
span (``no_sample``), an interior spacing > 10 s (``gap``, the TSS counter rule of
:mod:`scripts.l3_numerator`), a TTFT count that decreases (``reset``). Per column (as the
controller reads each signal from its own series): a sample without the TTFT histogram
(``missing_series``) empties the two TTFT columns, no KV sample of a serving pod in the
window (``no_kv``) the two KV columns.
The counts per status go into the dataset manifest.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from tre_common.vllm_metrics import vllm_candidates

from scripts import l3_numerator as l3

COLUMNS = ["ttft_server_mean_ms", "ttft_server_count", "kv_cache_usage_sum", "kv_cache_pods"]
TTFT_FAMILIES = vllm_candidates("time_to_first_token_seconds")
KV_FAMILIES = vllm_candidates("kv_cache_usage_perc")
SLEEP_FAMILY = "vllm:engine_sleep_state"

STATUS_OK = "ok"
STATUS_NO_KV = "no_kv"
VOID_STATUSES = (l3.STATUS_NO_METRICS, l3.STATUS_NO_SAMPLE, l3.STATUS_GAP, l3.STATUS_RESET,
                 l3.STATUS_MISSING_SERIES, STATUS_NO_KV)


def _family(key: str) -> str:
    return key.split("{", 1)[0]


def _ttft(hists: Mapping[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """``(sum_s, count)`` of the TTFT histogram in one decoded sample, summed over its
    series (engines); ``(None, None)`` when absent."""
    for family in TTFT_FAMILIES:
        series = [v for k, v in hists.items() if _family(k) == family and isinstance(v, Mapping)]
        if series:
            try:
                return sum(float(v["s"]) for v in series), sum(float(v["n"]) for v in series)
            except (KeyError, TypeError, ValueError):
                return None, None
    return None, None


def _kv(gauges: Mapping[str, Any]) -> Optional[float]:
    """KV-cache fill (0..1) in one sample: mean over the series of the newest family
    present (a 0.10 pod exposes both names; only one is read)."""
    for family in KV_FAMILIES:
        values = [float(v) for k, v in gauges.items() if _family(k) == family and v is not None]
        if values:
            return sum(values) / len(values)
    return None


def _asleep(gauges: Mapping[str, Any]) -> bool:
    awake = [float(v) for k, v in gauges.items()
             if _family(k) == SLEEP_FAMILY and 'sleep_state="awake"' in k and v is not None]
    return bool(awake) and max(awake) <= 0.0


@dataclass
class PodAltSamples:
    """One pod's samples in time order: ``(ts_ms, ttft_sum_s, ttft_count, kv, asleep)``."""

    pod: str
    samples: list[tuple[int, Optional[float], Optional[float], Optional[float], bool]]

    @classmethod
    def from_lines(cls, pod: str, lines: Sequence[str]) -> "PodAltSamples":
        from scripts.calibration_capture import decode_vllm_metrics

        samples = []
        for s in decode_vllm_metrics(lines):
            gauges = s.get("g") or {}
            ttft_sum, ttft_count = _ttft(s.get("h") or {})
            samples.append((int(s["ts_ms"]), ttft_sum, ttft_count, _kv(gauges), _asleep(gauges)))
        samples.sort(key=lambda row: row[0])
        return cls(pod=pod, samples=samples)

    @classmethod
    def from_file(cls, pod: str, path: Path) -> "PodAltSamples":
        return cls.from_lines(pod, Path(path).read_text(encoding="utf-8").splitlines())

    def _last_at_or_before(self, ts: int) -> Optional[int]:
        lo, hi = 0, len(self.samples)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.samples[mid][0] <= ts:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1 if lo > 0 else None

    def window_ttft(self, start_ms: int, end_ms: int, *, max_gap_ms: int = l3.DEFAULT_MAX_GAP_MS
                    ) -> tuple[str, Optional[float], Optional[float]]:
        """``(status, sum_s delta, count delta)`` over ``(start_ms, end_ms]``."""
        i0, i1 = self._last_at_or_before(start_ms), self._last_at_or_before(end_ms)
        if i0 is None or i1 is None:
            return l3.STATUS_NO_SAMPLE, None, None
        if start_ms - self.samples[i0][0] > max_gap_ms or end_ms - self.samples[i1][0] > max_gap_ms:
            return l3.STATUS_NO_SAMPLE, None, None
        if self.samples[i1][0] <= self.samples[i0][0]:
            return l3.STATUS_NO_SAMPLE, None, None
        span = self.samples[i0:i1 + 1]
        if any(b[0] - a[0] > max(max_gap_ms, l3.INTERIOR_GAP_MAX_MS) for a, b in zip(span, span[1:])):
            return l3.STATUS_GAP, None, None
        if any(s[1] is None or s[2] is None for s in span):
            return l3.STATUS_MISSING_SERIES, None, None
        if any(b[2] < a[2] for a, b in zip(span, span[1:])):
            return l3.STATUS_RESET, None, None
        return STATUS_OK, max(0.0, span[-1][1] - span[0][1]), span[-1][2] - span[0][2]

    def window_kv(self, start_ms: int, end_ms: int) -> tuple[str, Optional[float], bool]:
        """``(status, mean KV fill of the window's samples, serving)`` over ``(start_ms,
        end_ms]``; ``serving`` False = asleep in the last sample of the window (status ok,
        no value: online the fleet state drops the pod). The mean is over every sample of
        the window that carries the gauge, pre-wake ones included (online rule, module
        docstring)."""
        inside = [s for s in self.samples if start_ms < s[0] <= end_ms]
        if not inside:
            return l3.STATUS_NO_SAMPLE, None, False
        if inside[-1][4]:
            return STATUS_OK, None, False
        values = [s[3] for s in inside if s[3] is not None]
        if not values:
            return STATUS_NO_KV, None, True
        return STATUS_OK, sum(values) / len(values), True


class AltSignalColumns:
    """Columns of one cell's windows: ``self(start_ms, end_ms)`` -> the :data:`COLUMNS`
    (None = empty). Counts every window asked for by status (``counts``: a window with a
    void column under the first void status of its pods in :data:`VOID_STATUSES` order;
    ``missing_series`` voids only the TTFT columns, ``no_kv`` only the KV columns)."""

    def __init__(self, pods: Sequence[PodAltSamples], *, missing_pods: Sequence[str] = (),
                 max_gap_ms: int = l3.DEFAULT_MAX_GAP_MS) -> None:
        self.pods = list(pods)
        self.missing_pods = sorted(missing_pods)
        self.max_gap_ms = int(max_gap_ms)
        self.counts: Counter = Counter()

    def _void(self, status: str) -> dict[str, Any]:
        self.counts[status] += 1
        return {column: None for column in COLUMNS}

    def __call__(self, start_ms: int, end_ms: int) -> dict[str, Any]:
        if not self.pods or self.missing_pods:
            return self._void(l3.STATUS_NO_METRICS)
        ttft_statuses: list[str] = []
        kv_statuses: list[str] = []
        ttft_sum = ttft_count = kv_sum = 0.0
        kv_pods = 0
        for pod in self.pods:
            status, s, n = pod.window_ttft(int(start_ms), int(end_ms), max_gap_ms=self.max_gap_ms)
            ttft_statuses.append(status)
            if status == STATUS_OK:
                ttft_sum += s
                ttft_count += n
            status, kv, awake = pod.window_kv(int(start_ms), int(end_ms))
            kv_statuses.append(status)
            if status == STATUS_OK and awake:
                kv_sum += kv
                kv_pods += 1
        # Review L4 (2026-10-28), per column as online: a missing TTFT histogram makes only
        # ttft_mean unknown, a missing KV gauge only kv_usage (the controller reads each
        # signal from its own series); a pod without samples, a gap or a reset voids both.
        statuses = ttft_statuses + kv_statuses
        void = [s for s in VOID_STATUSES if s in statuses]
        both = [s for s in void if s not in (l3.STATUS_MISSING_SERIES, STATUS_NO_KV)]
        if both or (l3.STATUS_MISSING_SERIES in void and (STATUS_NO_KV in void or kv_pods == 0)):
            return self._void(void[0])
        ttft = {"ttft_server_mean_ms": ttft_sum / ttft_count * 1000.0 if ttft_count > 0 else None,
                "ttft_server_count": ttft_count}
        kv_cols = {"kv_cache_usage_sum": kv_sum, "kv_cache_pods": kv_pods}
        if l3.STATUS_MISSING_SERIES in void:
            ttft = {"ttft_server_mean_ms": None, "ttft_server_count": None}
        if STATUS_NO_KV in void or kv_pods == 0:
            # No KV sample of a serving pod, or every pod asleep (no serving replica: online
            # no routable pod, every signal unknown - the TTFT columns too).
            if kv_pods == 0 and STATUS_NO_KV not in void:
                return self._void(STATUS_NO_KV)
            kv_cols = {"kv_cache_usage_sum": None, "kv_cache_pods": None}
        self.counts[void[0] if void else STATUS_OK] += 1
        return {**ttft, **kv_cols}

    def summary(self) -> dict:
        return {"pods": [p.pod for p in self.pods], "missing_pods": self.missing_pods,
                "windows": dict(sorted(self.counts.items()))}


def columns_for_cell(model_dir: Path, stem: str, *, max_gap_ms: int = l3.DEFAULT_MAX_GAP_MS
                     ) -> AltSignalColumns:
    """The columns source of one attempt (the same pod files as the L3 numerator)."""
    files, missing = l3.cell_pod_files(model_dir, stem)
    pods = [PodAltSamples.from_file(pod, path) for pod, path in sorted(files.items())]
    return AltSignalColumns(pods, missing_pods=missing, max_gap_ms=max_gap_ms)


def rule() -> dict:
    """The manifest record of how the columns were made."""
    return {"columns": list(COLUMNS), "ttft_families": list(TTFT_FAMILIES),
            "kv_families": list(KV_FAMILIES), "max_gap_ms": l3.DEFAULT_MAX_GAP_MS,
            "interior_gap_max_ms": l3.INTERIOR_GAP_MAX_MS,
            "implementation": "scripts.alt_signal_columns.AltSignalColumns"}


if __name__ == "__main__":  # pragma: no cover - quick look at one cell
    import sys

    src = columns_for_cell(Path(sys.argv[1]), sys.argv[2])
    print(json.dumps(src.summary()))

#!/usr/bin/env python3
"""L3: the TSS numerator from vLLM's token counters (the paper's §5.2 definition).

The dataset's default numerator (``gateway``) is what the live controller reads: the
tokens of the requests that COMPLETED in the window (the gateway's per-request
histograms; offline, the client's per-request ``usage``). A long request therefore lands
in the window it finishes in, all at once. L3 (``vllm_counter``) counts the tokens the
engine PROCESSED in the window: the increase of ``vllm:prompt_tokens_total`` and
``vllm:generation_tokens_total`` of every model pod, from the 1 Hz per-pod capture
(``cells/<stem>/vllm_metrics_1hz/``, :mod:`scripts.calibration_capture`).

Only the numerator changes. Windows, membership ``(start, end]``, queue (the same
sidecar samples), EMA, idle rule and labels are the ones every other path uses
(:func:`scripts.rewindow_from_raw.label_cell` with ``token_source``).

Rules, per pod and window ``(start, end]``:

* boundary value = the last sample at or before the boundary (what a scrape at the
  boundary would have read, at most one 1 Hz period old). It must be at most
  ``max_gap_ms`` old, else the window is void (``no_sample``);
* every sample between the two boundary samples must follow its predecessor by at most
  ``max_gap_ms``, and no failed scrape may fall in between, else void (``gap``);
* each counter must be non-decreasing over those samples, else void (``reset``: a pod
  or engine restart zeroes it; a reset followed by regrowth would otherwise look like a
  small positive delta);
* a sample without the counter is void (``missing_series``); a pod with no file at all
  voids every window (``no_metrics``);
* the model's window value is the SUM over its pods; any void pod voids the window.

Why a hole voids instead of being interpolated: the counters are cumulative, so an
interior hole does not bias a delta that is read across it - but a hole is exactly where
a reset hides, and a boundary read across a hole is a stale value. Interpolation would
assume a uniform token rate across the hole, which is false in the bursty windows TSS
gets wrong (the burst blind spot). Void windows are dropped before the EMA (the shared
gap rule applies) and counted per reason in the dataset manifest, so the cost is visible.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

NUMERATOR_GATEWAY = "gateway"
NUMERATOR_VLLM_COUNTER = "vllm_counter"
NUMERATOR_CHOICES = (NUMERATOR_GATEWAY, NUMERATOR_VLLM_COUNTER)

#: Counter families summed into the numerator. ``prompt_tokens_total`` counts every
#: prefilled prompt token whatever its source; with prefix caching off (the v2 serve
#: arguments) it equals the ``local_compute`` source, and it is what the gateway
#: numerator's ``prompt_tokens * (1 - kv_cache_hit_rate)`` counts at hit rate 0.
PROMPT_COUNTER = "vllm:prompt_tokens_total"
GENERATION_COUNTER = "vllm:generation_tokens_total"
#: Two 1 Hz periods: one missed sample (spacing ~2 s) is already a hole.
DEFAULT_MAX_GAP_MS = 2000

STATUS_OK = "ok"
STATUS_NO_SAMPLE = "no_sample"
STATUS_GAP = "gap"
STATUS_RESET = "reset"
STATUS_MISSING_SERIES = "missing_series"
STATUS_NO_METRICS = "no_metrics"
VOID_STATUSES = (STATUS_NO_SAMPLE, STATUS_GAP, STATUS_RESET, STATUS_MISSING_SERIES, STATUS_NO_METRICS)

RULE = ("per pod: value at a boundary = last 1 Hz sample at or before it (<= max_gap_ms old); "
        "window delta = value(end) - value(start) of vllm:prompt_tokens_total and "
        "vllm:generation_tokens_total; void on a boundary without a sample, an interior spacing "
        "> max_gap_ms or a failed scrape (gap), a decreasing counter (reset), a missing series; "
        "model value = sum over pods, void if any pod is void; void windows are dropped before "
        "the EMA and counted")


def _family(key: str) -> str:
    return key.split("{", 1)[0]


def counter_total(counters: Mapping[str, Any], family: str) -> Optional[float]:
    """Sum of every series of ``family`` in one decoded sample (e.g. one per engine);
    None when the sample has none."""
    values = [float(v) for k, v in counters.items() if _family(k) == family and v is not None]
    return sum(values) if values else None


@dataclass
class PodCounters:
    """One pod's counter samples in time order: ``(ts_ms, prompt, generation)`` (None =
    series missing in that sample) and the timestamps of its failed scrapes."""

    pod: str
    samples: list[tuple[int, Optional[float], Optional[float]]]
    errors: list[int] = field(default_factory=list)

    @classmethod
    def from_lines(cls, pod: str, lines: Sequence[str]) -> "PodCounters":
        from scripts.calibration_capture import decode_vllm_metrics

        errors = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "error" in row and row.get("ts_ms") is not None:
                errors.append(int(row["ts_ms"]))
        samples = [
            (int(s["ts_ms"]), counter_total(s.get("c") or {}, PROMPT_COUNTER),
             counter_total(s.get("c") or {}, GENERATION_COUNTER))
            for s in decode_vllm_metrics(lines)
        ]
        samples.sort(key=lambda s: s[0])
        return cls(pod=pod, samples=samples, errors=sorted(errors))

    @classmethod
    def from_file(cls, pod: str, path: Path) -> "PodCounters":
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

    def window_delta(self, start_ms: int, end_ms: int, *, max_gap_ms: int = DEFAULT_MAX_GAP_MS
                     ) -> tuple[str, Optional[float], Optional[float]]:
        """``(status, prompt delta, generation delta)`` over ``(start_ms, end_ms]``; the
        deltas are None unless the status is ``ok``."""
        i0, i1 = self._last_at_or_before(start_ms), self._last_at_or_before(end_ms)
        if i0 is None or i1 is None:
            return STATUS_NO_SAMPLE, None, None
        if start_ms - self.samples[i0][0] > max_gap_ms or end_ms - self.samples[i1][0] > max_gap_ms:
            return STATUS_NO_SAMPLE, None, None
        span = self.samples[i0:i1 + 1]
        if any(b[0] - a[0] > max_gap_ms for a, b in zip(span, span[1:])):
            return STATUS_GAP, None, None
        t0, t1 = span[0][0], span[-1][0]
        if any(t0 < e <= t1 for e in self.errors):
            return STATUS_GAP, None, None
        if any(s[1] is None or s[2] is None for s in span):
            return STATUS_MISSING_SERIES, None, None
        for k in (1, 2):
            if any(b[k] < a[k] for a, b in zip(span, span[1:])):
                return STATUS_RESET, None, None
        return STATUS_OK, span[-1][1] - span[0][1], span[-1][2] - span[0][2]


class CounterTokenSource:
    """The ``token_source`` of :func:`scripts.rewindow_from_raw.label_cell` for one cell:
    the model's L3 window totals, summed over its pods; None for a void window. Counts
    every window it was asked for by status (``counts``; a void window under the first
    void status of its pods in :data:`VOID_STATUSES` order) and keeps both totals of the
    kept ones (``kept``: ``(start, end) -> (prompt, generation)``)."""

    def __init__(self, pods: Sequence[PodCounters], *, max_gap_ms: int = DEFAULT_MAX_GAP_MS,
                 missing_pods: Iterable[str] = ()) -> None:
        self.pods = list(pods)
        self.missing_pods = sorted(missing_pods)
        self.max_gap_ms = int(max_gap_ms)
        self.counts: Counter = Counter()
        self.status: dict[tuple[int, int], str] = {}
        self.kept: dict[tuple[int, int], tuple[float, float]] = {}

    def __call__(self, start_ms: int, end_ms: int) -> Optional[tuple[float, float]]:
        key = (int(start_ms), int(end_ms))
        if not self.pods or self.missing_pods:
            self.counts[STATUS_NO_METRICS] += 1
            self.status[key] = STATUS_NO_METRICS
            return None
        prompt = generation = 0.0
        statuses = []
        for pod in self.pods:
            status, p, g = pod.window_delta(int(start_ms), int(end_ms), max_gap_ms=self.max_gap_ms)
            statuses.append(status)
            if status == STATUS_OK:
                prompt += p
                generation += g
        void = [s for s in VOID_STATUSES if s in statuses]
        if void:
            self.counts[void[0]] += 1
            self.status[key] = void[0]
            return None
        self.counts[STATUS_OK] += 1
        self.status[key] = STATUS_OK
        self.kept[key] = (prompt, generation)
        return prompt, generation

    def summary(self) -> dict:
        return {"pods": [p.pod for p in self.pods], "missing_pods": self.missing_pods,
                "windows": dict(sorted(self.counts.items()))}


def cell_pod_files(model_dir: Path, stem: str) -> tuple[dict[str, Path], list[str]]:
    """``({pod: vllm_metrics_1hz file}, [pods expected but without a file])`` of one
    attempt. The files are the ones ``cell_meta.json`` lists (the pods scraped); the
    expected pods are its ``pods``. Without ``cell_meta.json`` every file of
    ``cells/<stem>/vllm_metrics_1hz/`` is taken (nothing is known to be missing)."""
    from scripts import calibration_capture as capture

    art = capture.resolve_cell_artifacts(Path(model_dir), stem)
    if art.get("cell_meta") is not None:
        files = dict(art.get("vllm_metrics") or {})
        meta = json.loads(Path(art["cell_meta"]).read_text(encoding="utf-8"))
        expected = [p.get("key") for p in meta.get("pods") or [] if p.get("key")]
        return files, sorted(p for p in expected if p not in files)
    d = Path(model_dir) / capture.CELLS_DIRNAME / stem / capture.VLLM_METRICS_DIRNAME
    files = {p.stem: p for p in sorted(d.glob("*.jsonl"))} if d.is_dir() else {}
    return files, []


def token_source_for_cell(model_dir: Path, stem: str, *, max_gap_ms: int = DEFAULT_MAX_GAP_MS
                          ) -> CounterTokenSource:
    files, missing = cell_pod_files(model_dir, stem)
    pods = [PodCounters.from_file(pod, path) for pod, path in sorted(files.items())]
    return CounterTokenSource(pods, max_gap_ms=max_gap_ms, missing_pods=missing)


def ratio_summary(values: Sequence[float]) -> dict:
    """min / p10 / median / p90 / max / mean of finite ratios (the smoke report)."""
    xs = sorted(v for v in values if v is not None and math.isfinite(v))
    if not xs:
        return {"n": 0}

    def q(p: float) -> float:
        k = (len(xs) - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    return {"n": len(xs), "min": xs[0], "p10": q(0.1), "median": q(0.5), "p90": q(0.9),
            "max": xs[-1], "mean": sum(xs) / len(xs)}

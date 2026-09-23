#!/usr/bin/env python3
"""Offline re-windowing of R3 raw per-request logs (S4, doc15 §4).

R3 capture (`r3_grid.py`) drops, per cell, a per-request JSONL `<cell_id>.jsonl` plus an
instant queue sidecar `<cell_id>.instant.jsonl` on local disk. This tool re-aggregates
those into the SAME window CSV the online path emits (r3_grid.CSV_COLUMNS), at any
`--window-ms` / `--step-ms` (sliding supported), so a 10h R3 run can be re-fit at 20s / 60s
/ … without re-running.

口径 (aggregation semantics) is kept identical to the online
`MetricsStore._aggregate_model`, NOT reimplemented ad hoc:

  * token totals   -> sum of per-request usage tokens in the window (== the histogram
                      sum-delta the online path takes).
  * queue avg      -> sum(instant samples in window) / expected_samples, with
                      expected_samples = max(1, window_ms // instant_sample_interval_ms)
                      (mirrors MetricsStore._instant_avg exactly).
  * p95            -> histogram_percentile(cumulative-from-samples, 0.95, mode) reusing
                      tre_common.percentile.histogram_percentile — the very function
                      MetricsStore uses — with the same bucket_upper / interpolated modes
                      and the same min_latency_samples N1 guard.
  * trs            -> r3_grid.compute_window_results (the shared time-constant TRSComputer,
                      i.e. the unified window-total TSS of tre_common.tss), so the trs column
                      is byte-identical to the online path; blank when TSS is undefined
                      (idle rule: running + waiting == 0).
  * row assembly   -> r3_grid.window_row / write_csv.
  * SLO label      -> :func:`label_cell` -> tre_common.slo_labels.apply_label_arms, the
                      one label every consumer uses: ``slo_label`` / ``slo_violated`` the
                      primary D6' label, ``slo_label_fixed`` / ``slo_label_k3`` the
                      comparison and ablation arms. The online probe verdict of the
                      boundary search is computed by calling :func:`label_cell` on the
                      cell it just drove, so "what the search judged" and "what the fit
                      is trained on" are the same function on the same bytes.

Latency is client-side, per request: p95 columns are ``*_client_ms`` and are taken over
the requests that were *served* (a failed request has no latency; it makes its window a
violation through the unserved counts instead).

Sidecar cadence (why `--instant-grid` exists)
---------------------------------------------
The open-loop campaign (`openloop.py`) samples the queue gauges at 1 Hz, while the live
control path only ever sees the gateway's boundary-aligned
:data:`tre_common.rediskeys.SCRAPE_INTERVAL_MS` grid. `openloop.mark_live_grid` tags each
sidecar sample with ``on_live_grid``, so one capture answers both questions. Two things
follow, and both are enforced here rather than left to the operator:

  * ``--instant-sample-ms`` is the *divisor* that turns instant samples into a window
    average. Pointing it at the wrong cadence rescales every queue average (1000 vs 10000
    ms = 10x) with no other symptom, so a mismatch is a hard error, and the cadence that
    was actually used is written to a sibling ``<output stem>.meta.json``.
  * theta is a decision threshold on the signal the controller *consumes*, so a fit must
    re-window from the live-grid subsample (``--instant-grid live``). The 1 Hz stream is
    kept for the aliasing figure and for the observability-gap metric below: the fraction
    of 1 Hz threshold crossings that the live grid never saw.

Assumptions (documented, doc15 §4 leaves them to "most conservative choice"):
  * Requests are bucketed into a window by done_ts_ms (completion time), half-open
    [window_start, window_end), matching when a vLLM completion increments the histograms;
    with ``--window-align grid`` the interval is (window_start, window_end] - the window
    the phase-aligned controller reads - for completions, latency samples, the
    ``completed_requests`` / ``ttft_len_samples`` evidence and the unserved send counts
    alike (:func:`in_window`).
  * The raw pools all requests as a single logical pod. R3 calibration runs a model at 1
    replica, where MetricsStore's per-pod sum (tokens/queue) and max (p95) collapse to the
    single pod, so pooling reproduces it. Multi-pod raw is out of scope (no pod id in raw).
  * kv_cache_hit_rate is not in the per-request raw -> treated as 0.0 (its only consumer,
    the trs prefill term, then matches an online window with no kv signal).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Optional, Sequence

from tre_common import slo_labels
from tre_common.metrics_schema import ModelWindowMetrics
from tre_common.percentile import histogram_percentile
from tre_common.rediskeys import SCRAPE_INTERVAL_MS

from scripts import openloop, r3_grid

#: ``--instant-grid`` choices. ``raw`` consumes every sidecar sample (campaign cadence
#: 1000 ms); ``live`` keeps only the samples the gateway ticker would have written, which
#: is the signal the controller actually consumes and therefore the one theta is fit on.
INSTANT_GRID_RAW = "raw"
INSTANT_GRID_LIVE = "live"
INSTANT_GRID_CHOICES = (INSTANT_GRID_RAW, INSTANT_GRID_LIVE)

#: How far the observed sidecar spacing may drift from ``--instant-sample-ms`` before the
#: run is refused. Jitter in a wall-clock sampler is normal; a cadence *mismatch* is not,
#: and the two are orders of magnitude apart (10x), so a loose tolerance still catches it.
SPACING_TOLERANCE = 0.25

#: The sidecar key the observability gap is measured on unless told otherwise.
DEFAULT_GAP_KEY = "waiting"


class CadenceMismatchError(ValueError):
    """``--instant-sample-ms`` disagrees with the cadence the sidecar was captured at.

    Raised instead of silently rescaling every queue average by the cadence ratio.
    """


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(doc, dict):
                records.append(doc)
    return records


# ------------------------------------------------------------------ sidecar cadence


def sidecar_has_live_grid_tags(instant_samples: Iterable[Mapping]) -> bool:
    """True when the sidecar was written by a capture that tagged the live grid.

    Pre-existing closed-loop captures have no such tag; they stay usable in ``raw`` mode
    and are simply exempt from the spacing check (there is nothing to check against).
    """
    return any("on_live_grid" in s for s in instant_samples)


def observed_sample_spacing_ms(instant_samples: Sequence[Mapping]) -> Optional[float]:
    """Median spacing between consecutive sidecar timestamps, or None if undecidable.

    The median (not the mean) so a single gap — a restarted sampler, a scrape timeout —
    does not move the verdict.
    """
    stamps = sorted(int(s["ts_ms"]) for s in instant_samples if s.get("ts_ms") is not None)
    deltas = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
    if not deltas:
        return None
    return float(median(deltas))


def select_instant_samples(instant_samples: Sequence[dict], instant_grid: str) -> list[dict]:
    """The sidecar samples one grid mode consumes.

    ``live`` keeps only ``on_live_grid`` samples — the ones the gateway ticker would have
    written — so the re-window reproduces what the controller sees.
    """
    if instant_grid == INSTANT_GRID_RAW:
        return list(instant_samples)
    if instant_grid == INSTANT_GRID_LIVE:
        return [s for s in instant_samples if bool(s.get("on_live_grid", False))]
    raise ValueError(f"unknown instant_grid {instant_grid!r}; expected one of {INSTANT_GRID_CHOICES}")


def resolve_instant_cadence(
    instant_samples: Sequence[Mapping],
    *,
    instant_grid: str,
    instant_sample_ms: int,
    source: Optional[str] = None,
) -> int:
    """Validate the declared cadence against the mode and the sidecar, return it.

    Fails loudly instead of rescaling: the returned value is the divisor
    ``aggregate_window`` uses for the queue average, so a wrong one is a silent nx error
    on every queue column in the fit.
    """
    if instant_grid not in INSTANT_GRID_CHOICES:
        raise ValueError(f"unknown instant_grid {instant_grid!r}; expected one of {INSTANT_GRID_CHOICES}")
    if instant_sample_ms <= 0:
        raise CadenceMismatchError(f"instant_sample_ms must be positive, got {instant_sample_ms}")
    where = f" (sidecar {source})" if source else ""

    if instant_grid == INSTANT_GRID_LIVE:
        if instant_sample_ms != SCRAPE_INTERVAL_MS:
            ratio = instant_sample_ms / SCRAPE_INTERVAL_MS
            raise CadenceMismatchError(
                f"--instant-grid live consumes only the live-grid subsample, whose spacing is "
                f"the gateway cadence SCRAPE_INTERVAL_MS={SCRAPE_INTERVAL_MS} ms, but "
                f"--instant-sample-ms={instant_sample_ms} ms{where}. That flag is the divisor "
                f"that turns instant samples into a window average, so keeping it would scale "
                f"every queue average by {SCRAPE_INTERVAL_MS / instant_sample_ms:.6g}x "
                f"(declared/actual = {ratio:.6g}). Pass --instant-sample-ms {SCRAPE_INTERVAL_MS}."
            )
        return SCRAPE_INTERVAL_MS

    # raw mode: only a tagged sidecar tells us what cadence it was captured at.
    if sidecar_has_live_grid_tags(instant_samples):
        observed = observed_sample_spacing_ms(instant_samples)
        if observed is not None and abs(observed - instant_sample_ms) > SPACING_TOLERANCE * instant_sample_ms:
            raise CadenceMismatchError(
                f"the capture{where} has ~{observed:.0f} ms between samples but "
                f"--instant-sample-ms={instant_sample_ms} ms (more than "
                f"{SPACING_TOLERANCE * 100:.0f}% apart). That flag is the divisor that turns "
                f"instant samples into a window average, so every queue average would be scaled "
                f"by {instant_sample_ms / observed:.6g}x. Pass --instant-sample-ms {observed:.0f} "
                f"for the raw stream, or --instant-grid live with "
                f"--instant-sample-ms {SCRAPE_INTERVAL_MS} to fit on the signal the controller sees."
            )
    return instant_sample_ms


# ------------------------------------------------------------- observability gap


@dataclass(frozen=True)
class ObservabilityGap:
    """How much of the 1 Hz truth the live grid never saw, on one key and threshold."""

    key: str
    threshold: float
    window_ms: int
    raw_crossings: int
    live_crossings: int
    total_windows: int

    @property
    def gap(self) -> float:
        # Nothing to miss -> no gap. Defined, not NaN, so it aggregates and serialises.
        if self.raw_crossings == 0:
            return 0.0
        return 1.0 - (self.live_crossings / self.raw_crossings)

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "threshold": self.threshold,
            "window_ms": self.window_ms,
            "raw_crossings": self.raw_crossings,
            "live_crossings": self.live_crossings,
            "total_windows": self.total_windows,
            "observability_gap": round(self.gap, 6),
        }


def observability_gap(
    instant_samples: Sequence[dict],
    *,
    window_ms: int,
    key: str = DEFAULT_GAP_KEY,
    threshold: float = 0.0,
) -> ObservabilityGap:
    """1 - (windows the live grid saw cross ``threshold``) / (windows the 1 Hz stream did).

    The windowing itself is ``openloop.windows_observing`` (same tumbling bound, same
    ``on_live_grid`` restriction) so the metric cannot drift from the capture's own
    definition of the live grid.
    """
    raw_crossings, total = openloop.windows_observing(
        instant_samples, window_ms=window_ms, key=key, grid_only=False, threshold=threshold
    )
    live_crossings, _ = openloop.windows_observing(
        instant_samples, window_ms=window_ms, key=key, grid_only=True, threshold=threshold
    )
    return ObservabilityGap(
        key=key,
        threshold=float(threshold),
        window_ms=window_ms,
        raw_crossings=raw_crossings,
        live_crossings=live_crossings,
        total_windows=total,
    )


def combine_observability_gaps(gaps: Sequence[ObservabilityGap]) -> Optional[ObservabilityGap]:
    """Pool per-cell crossings into one fleet-level gap (counts add, ratios do not)."""
    if not gaps:
        return None
    head = gaps[0]
    return ObservabilityGap(
        key=head.key,
        threshold=head.threshold,
        window_ms=head.window_ms,
        raw_crossings=sum(g.raw_crossings for g in gaps),
        live_crossings=sum(g.live_crossings for g in gaps),
        total_windows=sum(g.total_windows for g in gaps),
    )


# ------------------------------------------------------------------ run provenance


def git_short_sha(worktree: Path) -> Optional[str]:
    """Short SHA of the worktree the script lives in, or None when git cannot answer."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and sha else None


def meta_path_for(output: Path) -> Path:
    """``<output stem>.meta.json`` next to the CSV."""
    return output.parent / f"{output.stem}.meta.json"


def build_meta(
    *,
    model: str,
    raw_dir: Path | str,
    cells: Sequence[str],
    window_ms: int,
    step_ms: int,
    instant_sample_ms: int,
    instant_grid: str,
    percentile_mode: str,
    min_latency_samples: int,
    routable_pods: int,
    assigned_replicas: int,
    rows: Optional[int] = None,
    gap_per_cell: Optional[Mapping[str, ObservabilityGap]] = None,
    gap_overall: Optional[ObservabilityGap] = None,
    git_sha: Optional[str] = None,
    generated_at: Optional[str] = None,
    label: Optional[dict] = None,
) -> dict:
    """The record that answers "which cadence did this fit use" (plus the gap metric)."""
    meta: dict = {
        "label": label,
        "generated_at_utc": generated_at or datetime.now(timezone.utc).isoformat(),
        "git_short_sha": git_sha,
        "model": model,
        "raw_dir": str(raw_dir),
        "cells": list(cells),
        "rows": rows,
        "window_ms": window_ms,
        "step_ms": step_ms,
        "instant_sample_ms": instant_sample_ms,
        "instant_grid": instant_grid,
        "live_grid_ms": SCRAPE_INTERVAL_MS,
        "percentile_mode": percentile_mode,
        "min_latency_samples": min_latency_samples,
        "routable_pods": routable_pods,
        "assigned_replicas": assigned_replicas,
    }
    if gap_overall is not None or gap_per_cell:
        meta["observability_gap"] = {
            "overall": gap_overall.as_dict() if gap_overall is not None else None,
            "per_cell": {cid: g.as_dict() for cid, g in sorted((gap_per_cell or {}).items())},
        }
    return meta


def write_meta(output: Path, meta: dict) -> Path:
    path = meta_path_for(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------- aggregation


def _samples_to_cumulative(samples: Iterable[float]) -> list[tuple[float, float]]:
    """Turn exact samples into a cumulative histogram [(value, count<=value)], the input
    shape histogram_percentile expects. Using the sample values as bucket uppers means the
    reused percentile function reproduces the online bucket_upper / interpolated behaviour
    exactly when the online histogram's buckets are the same sample values."""
    counts = Counter(float(s) for s in samples)
    cumulative: list[tuple[float, float]] = []
    running = 0.0
    for value in sorted(counts):
        running += counts[value]
        cumulative.append((value, running))
    return cumulative


def sample_percentile(samples: list[float], quantile: float, mode: str) -> Optional[float]:
    if not samples:
        return None
    return histogram_percentile(_samples_to_cumulative(samples), quantile, mode=mode)


def _guarded_p95(samples: list[float], mode: str, min_latency_samples: int) -> Optional[float]:
    # N1 guard, identical to MetricsStore._hist_percentile: too few observations -> None,
    # so the signal layer treats latency as unavailable rather than deciding on noise.
    if min_latency_samples > 0 and len(samples) < min_latency_samples:
        return None
    return sample_percentile(samples, 0.95, mode)


def enumerate_windows(
    start_ms: int, end_ms: int, window_ms: int, step_ms: int, *, align_ms: Optional[int] = None
) -> list[tuple[int, int]]:
    """Windows [w, w+window_ms) advancing by step_ms (step_ms==window_ms -> tumbling).
    Mirrors the online driver's ``while w + window_ms <= end`` bound.

    ``align_ms`` (window-align ``grid``, plan §6.9g pitfall 2) puts every window end on
    the ``align_ms`` grid - the gateway's instant-sample boundaries, where the
    phase-aligned controller ends its windows - starting from the first grid point at or
    before ``start_ms``; ``step_ms`` must then be a multiple of ``align_ms``."""
    if window_ms <= 0 or step_ms <= 0:
        raise ValueError("window_ms and step_ms must be positive")
    if align_ms is not None:
        if align_ms <= 0:
            raise ValueError("align_ms must be positive")
        if step_ms % align_ms:
            raise ValueError(f"step_ms={step_ms} must be a multiple of align_ms={align_ms} for grid-aligned windows")
        start_ms = start_ms // align_ms * align_ms
    windows: list[tuple[int, int]] = []
    w = start_ms
    while w + window_ms <= end_ms:
        windows.append((w, w + window_ms))
        w += step_ms
    return windows


#: ``--window-align`` choices: ``none`` keeps the legacy free-phase windows (first window
#: at the first stamp of the capture, any step); ``grid`` ends every window on the
#: ``SCRAPE_INTERVAL_MS`` grid like the phase-aligned controller.
WINDOW_ALIGN_NONE = "none"
WINDOW_ALIGN_GRID = "grid"
WINDOW_ALIGN_CHOICES = (WINDOW_ALIGN_NONE, WINDOW_ALIGN_GRID)


# --------------------------------------------------------------- request outcomes


#: The outcome of a served request, as ``openloop.OUTCOME_NAMES`` spells it.
OUTCOME_OK = openloop.OUTCOME_NAMES[openloop.FAILURE_NONE]


def request_outcome(record: Mapping) -> str:
    """``ok`` / ``shed`` / ``proxy_transient`` / ``model_error`` / ``client_timeout``.

    Captures from 2026-09-23 on carry the classification in the raw record itself
    (``outcome``); older raw records get it from :func:`attach_failure_details`, which
    copies the failure sidecar's verdict onto them. See ``openloop.outcome_of``.
    """
    return openloop.outcome_of(record)


def attach_failure_details(records: Sequence[dict], failures: Sequence[Mapping]) -> tuple[list[dict], int]:
    """Copy the failure sidecar's per-request verdict onto the raw records it describes.

    Needed only for captures whose raw records predate the ``outcome`` field: their
    classification lives in ``<cell>.failures.jsonl``, keyed by send instant. Records are
    matched on ``(send_ts_ms, http_status)`` in order, so two failures sent in the same
    millisecond still pair one-to-one. Returns (records, failures that matched nothing).
    """
    pending: dict[tuple, list[Mapping]] = {}
    for failure in failures:
        key = (_as_int(failure.get("send_ts_ms")), _as_int(failure.get("http_status")))
        pending.setdefault(key, []).append(failure)
    out: list[dict] = []
    for record in records:
        rec = dict(record)
        if not rec.get("outcome"):
            key = (_as_int(rec.get("send_ts_ms")), _as_int(rec.get("http_status")))
            queue = pending.get(key)
            if queue:
                failure = queue.pop(0)
                for field_name in ("outcome", "proxy_reason", "request_id", "in_flight_at_send",
                                   "request_timeout_s"):
                    if rec.get(field_name) is None and failure.get(field_name) is not None:
                        rec[field_name] = failure.get(field_name)
                if not rec.get("outcome"):
                    # A sidecar older than the ``outcome`` field: its capture-time verdict
                    # is ``failure_class`` (openloop.failure_signature).
                    rec["outcome"] = openloop.outcome_of(
                        {"failure_class": failure.get("failure_class"), **rec})
        out.append(rec)
    unmatched = sum(len(queue) for queue in pending.values())
    return out, unmatched


def _as_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def is_served(record: Mapping) -> bool:
    return request_outcome(record) == OUTCOME_OK


def aggregate_window(
    records: list[dict],
    instant_samples: list[dict],
    model: str,
    window_start_ms: int,
    window_end_ms: int,
    *,
    percentile_mode: str,
    min_latency_samples: int,
    instant_sample_interval_ms: int,
    routable_pods: int = 1,
    assigned_replicas: int = 1,
    half_open_start: bool = False,
    instant_tick_ms: Optional[int] = None,
) -> ModelWindowMetrics:
    """Aggregate raw per-request + instant records into one ModelWindowMetrics, using the
    same 口径 as MetricsStore._aggregate_model (see module docstring).

    ``half_open_start`` (grid-aligned windows) mirrors the phase-aligned controller's
    ``(start, end]`` read: completions with ``start < done <= end`` and instants on
    ``(start, end]`` - exactly ``window_ms / interval`` of them. ``instant_tick_ms`` (live
    grid) stamps each live-grid sidecar sample with the gateway tick it stands for,
    ``floor(ts / tick) * tick`` (``openloop.mark_live_grid`` keeps the first 1 Hz sample of
    each tick bucket, taken 0-1 s after the boundary the gateway stamps it with).
    """
    if half_open_start:
        in_window = [
            r for r in records
            if r.get("done_ts_ms") is not None and window_start_ms < r["done_ts_ms"] <= window_end_ms
        ]
    else:
        in_window = [
            r for r in records
            if r.get("done_ts_ms") is not None and window_start_ms <= r["done_ts_ms"] < window_end_ms
        ]
    prompt_tokens = sum(r["input_tokens"] for r in in_window if r.get("input_tokens") is not None)
    generation_tokens = sum(r["output_tokens"] for r in in_window if r.get("output_tokens") is not None)

    # Latency is taken over SERVED requests only. A failed request's e2e is how long it
    # took to fail (a 503 in 3 ms, a client timeout at 30 s), not a latency of the engine;
    # it enters the label through the unserved counts instead.
    served = [r for r in in_window if is_served(r)]
    ttft_samples = [r["ttft_ms"] for r in served if r.get("ttft_ms") is not None]
    tpot_samples = [r["tpot_ms"] for r in served if r.get("tpot_ms") is not None]
    e2e_samples = [r["e2e_ms"] for r in served if r.get("e2e_ms") is not None]

    # queue: instant samples are inclusive [start, end] and divided by expected_samples,
    # exactly as MetricsStore._instant_avg does.
    def _stamp(sample: dict) -> int:
        ts = int(sample["ts_ms"])
        return ts // instant_tick_ms * instant_tick_ms if instant_tick_ms else ts

    if half_open_start:
        inst = [
            s for s in instant_samples
            if s.get("ts_ms") is not None and window_start_ms < _stamp(s) <= window_end_ms
        ]
    else:
        inst = [
            s for s in instant_samples
            if s.get("ts_ms") is not None and window_start_ms <= _stamp(s) <= window_end_ms
        ]
    expected_samples = max(1, int((window_end_ms - window_start_ms) / instant_sample_interval_ms))
    avg_waiting = sum(float(s.get("waiting", 0.0)) for s in inst) / expected_samples
    avg_running = sum(float(s.get("running", 0.0)) for s in inst) / expected_samples
    avg_swapping = sum(float(s.get("swapping", 0.0)) for s in inst) / expected_samples

    return ModelWindowMetrics(
        model=model,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        prompt_tokens=float(prompt_tokens),
        generation_tokens=float(generation_tokens),
        avg_waiting=avg_waiting,
        avg_running=avg_running,
        avg_swapping=avg_swapping,
        kv_cache_hit_rate=0.0,
        ttft_p95_ms=_guarded_p95(ttft_samples, percentile_mode, min_latency_samples),
        tpot_p95_ms=_guarded_p95(tpot_samples, percentile_mode, min_latency_samples),
        e2e_p95_ms=_guarded_p95(e2e_samples, percentile_mode, min_latency_samples),
        routable_pods=routable_pods,
        assigned_replicas=assigned_replicas,
        per_pod={},
    )


def in_window(ts_ms, window_start_ms: int, window_end_ms: int, *, closed_right: bool = False) -> bool:
    """Window membership of one instant: ``[start, end)`` (free-phase windows, the live
    grid) or ``(start, end]`` (``--window-align grid``: the phase-aligned controller
    reads the window that ENDS on a tick). Every per-window quantity - completions,
    latency samples, unserved sends - uses the same interval."""
    if ts_ms is None:
        return False
    ts = int(ts_ms)
    if closed_right:
        return window_start_ms < ts <= window_end_ms
    return window_start_ms <= ts < window_end_ms


def window_request_evidence(
    records: Sequence[dict], window_start_ms: int, window_end_ms: int, *, closed_right: bool = False,
) -> dict:
    """The per-request columns of one window: ``completed_requests`` (the min-n guard) and
    ``ttft_len_samples`` (what the slowdown TTFT label reads, ``tre_common.slo_labels``).

    Same membership and the same request set as the p95 columns of
    :func:`aggregate_window`: requests that were *served* and completed inside the window
    (``closed_right`` as there). ``ttft_len_samples`` pairs each served request's TTFT
    with its vLLM-reported prompt length (raw ``input_tokens`` = ``usage.prompt_tokens``).
    """
    done = [
        r for r in records
        if in_window(r.get("done_ts_ms"), window_start_ms, window_end_ms, closed_right=closed_right)
        and is_served(r)
    ]
    return {
        slo_labels.COMPLETED_REQUESTS_COLUMN: len(done),
        slo_labels.TTFT_LEN_SAMPLES_COLUMN: slo_labels.format_ttft_len_samples(
            (r["ttft_ms"], r.get("input_tokens")) for r in done if r.get("ttft_ms") is not None
        ),
    }


def _time_span(records: list[dict], instant_samples: list[dict]) -> Optional[tuple[int, int]]:
    dones = [r["done_ts_ms"] for r in records if r.get("done_ts_ms") is not None]
    insts = [s["ts_ms"] for s in instant_samples if s.get("ts_ms") is not None]
    stamps = dones + insts
    if not stamps:
        return None
    # end is exclusive-ish: +1 so the last completion falls inside the final window.
    return min(stamps), max(stamps) + 1


def rewindow_cell(
    records: list[dict],
    instant_samples: list[dict],
    cell: r3_grid.GridCell,
    spec,
    *,
    window_ms: int,
    step_ms: int,
    percentile_mode: str,
    min_latency_samples: int,
    instant_sample_interval_ms: int,
    instant_grid: str = INSTANT_GRID_RAW,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    routable_pods: int = 1,
    assigned_replicas: int = 1,
    window_align: str = WINDOW_ALIGN_NONE,
) -> list[dict]:
    """Re-window one cell's raw into calibration CSV rows (reusing r3_grid.window_row +
    compute_window_results for the trs column). Latency columns are client-side; the
    rows carry no SLO label and no unserved counts yet - :func:`label_cell` adds both.

    ``window_align="grid"`` ends every window on the ``SCRAPE_INTERVAL_MS`` grid and reads
    it half-open ``(start, end]`` (instants stamped with their gateway tick on the live
    grid), i.e. the window the phase-aligned controller reads; ``"none"`` is the
    free-phase windowing (``[start, end)``), kept for the 5 s / unaligned fits and the
    parity check against captures made with it.
    """
    instant_sample_interval_ms = resolve_instant_cadence(
        instant_samples,
        instant_grid=instant_grid,
        instant_sample_ms=instant_sample_interval_ms,
        source=cell.scenario_id,
    )
    instant_samples = select_instant_samples(instant_samples, instant_grid)
    if window_align not in WINDOW_ALIGN_CHOICES:
        raise ValueError(f"unknown window_align {window_align!r}; expected one of {WINDOW_ALIGN_CHOICES}")
    aligned = window_align == WINDOW_ALIGN_GRID
    if start_ms is None or end_ms is None:
        span = _time_span(records, instant_samples)
        if span is None:
            return []
        start_ms = span[0] if start_ms is None else start_ms
        end_ms = span[1] if end_ms is None else end_ms
    windows_ms = enumerate_windows(
        start_ms, end_ms, window_ms, step_ms, align_ms=SCRAPE_INTERVAL_MS if aligned else None
    )
    metrics = [
        aggregate_window(
            records, instant_samples, spec.name, ws, we,
            percentile_mode=percentile_mode,
            min_latency_samples=min_latency_samples,
            instant_sample_interval_ms=instant_sample_interval_ms,
            routable_pods=routable_pods, assigned_replicas=assigned_replicas,
            half_open_start=aligned,
            instant_tick_ms=SCRAPE_INTERVAL_MS if aligned and instant_grid == INSTANT_GRID_LIVE else None,
        )
        for ws, we in windows_ms
    ]
    results = r3_grid.compute_window_results(metrics, spec)
    rows = []
    for wm, result in zip(metrics, results):
        evidence = window_request_evidence(
            records, wm.window_start_ms, wm.window_end_ms, closed_right=aligned
        )
        rows.append(r3_grid.window_row(
            # An undefined TSS (idle rule: nothing in flight, tre_common.tss) is written
            # blank so no fit ever reads it as a (tiny) signal value.
            cell, wm, result.TRS if result.defined else None, result.Q_ctl,
            client=wm,
            completed_requests=evidence[slo_labels.COMPLETED_REQUESTS_COLUMN],
            ttft_len_samples=evidence[slo_labels.TTFT_LEN_SAMPLES_COLUMN],
        ))
    return rows


def label_cell(
    records: Sequence[dict],
    instant_samples: Sequence[dict],
    cell: r3_grid.GridCell,
    spec,
    *,
    label=None,
    latency_slo_ms: Optional[Mapping[str, float]] = None,
    window_ms: int,
    step_ms: int,
    percentile_mode: str,
    min_latency_samples: int,
    instant_sample_interval_ms: int,
    instant_grid: str,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    truncated_at_ts_ms: Optional[int] = None,
    routable_pods: int = 1,
    assigned_replicas: int = 1,
    window_align: str = WINDOW_ALIGN_NONE,
) -> list[dict]:
    """One cell's window rows, each with its SLO labels. THE labelling path.

    Called by the online driver (``r3_grid``) on the cell it has just driven - that is the
    verdict the boundary search reads - by :func:`main` offline and by the standard
    dataset. All hand it the same raw per-request records and sidecar samples, so a
    probe's windows and the windows a fit is built from are labelled by construction with
    the same function on the same data.

    ``label`` is the primary :class:`tre_common.slo_labels.LabelDefinition` (D6') - its
    fixed comparison and k = 3 ablation arms are derived from it - or explicit arms
    (``slo_labels.resolve_arms``); ``latency_slo_ms`` (a ``{"ttft_p95", "tpot_p95"}``
    threshold mapping, the 09-23 interface) is accepted in its place and labels every arm
    with that fixed rule.

    ``records`` are raw records (``r3_grid.RAW_COLUMNS``); their outcome is read from the
    record (or its failure sidecar, see :func:`attach_failure_details`, or classified -
    :func:`request_outcome`). Steps, in order:

    1. :func:`rewindow_cell` - client-side aggregation per window (served requests only);
    2. ``openloop.mark_unserved_request_windows`` - per-window counts of requests *sent*
       in the window that went unserved, on the same interval as the windows' latency;
    3. ``r3_grid.censor_after`` - drop windows after a truncation;
    4. :func:`tre_common.slo_labels.apply_label_arms` - the labels.
    """
    arms = label if label is not None else latency_slo_ms
    if arms is None:
        raise ValueError("label_cell needs label (or latency_slo_ms)")
    rows = rewindow_cell(
        list(records), list(instant_samples), cell, spec,
        window_ms=window_ms, step_ms=step_ms,
        percentile_mode=percentile_mode,
        min_latency_samples=min_latency_samples,
        instant_sample_interval_ms=instant_sample_interval_ms,
        instant_grid=instant_grid,
        start_ms=start_ms, end_ms=end_ms,
        routable_pods=routable_pods, assigned_replicas=assigned_replicas,
        window_align=window_align,
    )
    rows = openloop.mark_unserved_request_windows(
        rows, records, closed_right=window_align == WINDOW_ALIGN_GRID
    )
    rows, _dropped = r3_grid.censor_after(rows, truncated_at_ts_ms)
    for row in rows:
        slo_labels.apply_label_arms(row, arms)
    return rows


def read_guard(raw_path: Path) -> dict:
    """The ``<cell>.guard.json`` next to a raw capture, or {} when there is none.

    It carries the drive's own [start_ms, end_ms] and any truncation instant, which the
    offline path needs to lay down exactly the windows the online path did.
    """
    cell_id = raw_path.name.split(".", 1)[0]
    path = raw_path.parent / f"{cell_id}.guard.json"
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return doc if isinstance(doc, dict) else {}


def load_cell_capture(raw_path: Path) -> tuple[list[dict], list[dict], dict, int]:
    """(raw records with outcomes, instant samples, guard, unmatched failures) of one cell.

    ``raw_path`` may be a quarantined ``.jsonl.void`` capture; its sidecars are found by
    cell id either way.
    """
    cell_id = raw_path.name.split(".", 1)[0]
    records = load_jsonl(raw_path)
    instants = load_jsonl(raw_path.parent / f"{cell_id}.instant.jsonl")
    failures = load_jsonl(raw_path.parent / f"{cell_id}.failures.jsonl")
    records, unmatched = attach_failure_details(records, failures)
    return records, instants, read_guard(raw_path), unmatched


# ------------------------------------------------------------------ ledger (ladder)


def load_ledgers(paths: Iterable) -> dict[str, dict]:
    """cell id -> the latest ledger line (``cells.jsonl`` of a ladder-design campaign)
    naming it: role, primitive, split, stage, warm-up. Several ledgers may be given (one
    per model campaign); a later attempt of the same cell overrides an earlier one."""
    out: dict[str, dict] = {}
    for path in paths:
        for record in load_jsonl(Path(path)):
            cell_id = str(record.get("cell_id") or "")
            if cell_id:
                prev = out.get(cell_id)
                if prev is None or int(record.get("attempt", 1)) >= int(prev.get("attempt", 1)):
                    out[cell_id] = record
    return out


#: Columns ``--ledger`` adds to every row (the standard dataset's names).
LEDGER_COLUMNS = ("in_warmup", "role", "split", "primitive", "stage")


def mark_warmup(rows: list[dict], *, start_ms: Optional[int], warmup_s: Optional[float]) -> None:
    """``in_warmup`` on every row, in place: True for a window that starts before
    ``start_ms + warmup_s`` (the standard dataset's rule, ``calibration_dataset``); None
    when either is unknown. The fit loaders drop ``in_warmup`` rows."""
    end = None
    if start_ms is not None and warmup_s is not None:
        end = int(start_ms) + int(round(float(warmup_s) * 1000))
    for row in rows:
        row["in_warmup"] = None if end is None else int(row["window_start_ms"]) < end


def _raw_size_bytes(raw_dir: Path) -> int:
    return sum(p.stat().st_size for p in raw_dir.rglob("*.jsonl"))


def held_out_cell_ids(index: Mapping) -> set[str]:
    """Cell ids the schedule index marks as held out.

    Read from the index rather than guessed from a file name: the raw tree is keyed by
    cell id and nothing in a file name says which shape produced it, so a fit pointed at
    the raw directory would otherwise silently train on the validation set.
    """
    return {
        str(entry["cell_id"])
        for entry in (index.get("schedules", []) or [])
        if entry.get("held_out") and entry.get("cell_id")
    }


#: JSONL files a cell writes next to its raw capture that are NOT per-request rows.
#: Everything else matching ``*.jsonl`` under the raw root is a cell, so a new sidecar
#: that forgets to register here is silently re-windowed as if it were measurements.
SIDECAR_JSONL_SUFFIXES = (".instant.jsonl", ".failures.jsonl", ".prompts.jsonl")


def raw_dir_shape(dirname: str, model: str) -> Optional[str]:
    """The shape a campaign cell directory belongs to, from its name, or None.

    ``calibration_campaign`` names every cell directory ``<model>_<shape>_<rest>`` (a
    stage cell: ``dsqwen-7b_S3_ramp``; a boundary hold: ``dsqwen-7b_S3_S3_hold1060_a1``),
    so the shape is the token after the model prefix. This is what lets a family CSV be
    built from what is on disk at fit time - hold cells included - instead of a cell list
    frozen before the boundary search ran (plan §6.3 B3).
    """
    prefix = f"{model}_"
    if not dirname.startswith(prefix):
        return None
    shape = dirname[len(prefix):].split("_", 1)[0]
    return shape or None


def discover_cell_files(
    raw_dir: Path,
    *,
    exclude: Iterable[str] = (),
    only: Iterable[str] = (),
    only_shapes: Iterable[str] = (),
    model: Optional[str] = None,
) -> tuple[list[Path], list[str]]:
    """(cell raw files to re-window, cell ids skipped).

    Searched **recursively**, because a campaign writes one directory per cell under the
    raw root; a flat glob finds nothing there and produces an empty CSV without saying so.

    ``only`` wins over ``exclude`` when both are given: an explicit inclusion list is a
    stronger statement than a default exclusion. ``only_shapes`` (needs ``model``) keeps
    only cells whose directory belongs to one of those shapes (:func:`raw_dir_shape`); it
    composes with ``exclude``, so a family CSV still drops the held-out cells.
    """
    excluded = {str(c) for c in exclude}
    included = {str(c) for c in only}
    shapes = {str(s) for s in only_shapes}
    if shapes and not model:
        raise ValueError("only_shapes needs the model name to parse cell directory names")
    kept: list[Path] = []
    skipped: list[str] = []
    for path in sorted(raw_dir.rglob("*.jsonl")):
        if path.name.endswith(SIDECAR_JSONL_SUFFIXES):
            continue
        cell_id = path.stem
        if shapes and raw_dir_shape(path.parent.name, str(model)) not in shapes:
            skipped.append(cell_id)
            continue
        if included:
            if cell_id not in included:
                skipped.append(cell_id)
                continue
        elif cell_id in excluded:
            skipped.append(cell_id)
            continue
        kept.append(path)
    return kept, skipped


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw-dir", required=True,
                    help="dir holding <cell_id>.jsonl (+ .instant.jsonl), searched recursively")
    ap.add_argument("--exclude-cell-id", action="append", default=[],
                    help="skip this cell. Repeatable. The calibration fit uses it to keep "
                         "the held-out shape out of training - nothing in a raw file name "
                         "says which shape it came from, so without this a fit pointed at "
                         "the raw tree trains on the validation set.")
    ap.add_argument("--only-cell-id", action="append", default=[],
                    help="re-window only these cells. Repeatable. Overrides "
                         "--exclude-cell-id; used to build the held-out validation CSV "
                         "and the per-family diagnostic CSVs.")
    ap.add_argument("--only-shape", action="append", default=[],
                    help="re-window only cells whose campaign directory belongs to this "
                         "shape (<model>_<shape>_...). Repeatable. Resolved from the raw "
                         "tree at run time, so boundary hold cells are included; combines "
                         "with --exclude-cell-id / --held-out-index. Used for family CSVs.")
    ap.add_argument("--held-out-index", default=None,
                    help="schedule INDEX.json; every entry marked held_out is excluded, "
                         "in addition to --exclude-cell-id")
    ap.add_argument("--ledger", action="append", default=[],
                    help="cells.jsonl of a ladder-design campaign (repeatable). Adds the "
                         "ledger's role / split / primitive / stage to every row and marks "
                         "in_warmup (window start < cell start + warmup_s), which the fit "
                         "loaders drop")
    ap.add_argument("--output", required=True, help="re-windowed CSV path")
    ap.add_argument("--window-ms", type=int, required=True)
    ap.add_argument("--step-ms", type=int, default=None, help="slide step; default = window-ms (tumbling)")
    ap.add_argument(
        "--window-align", default=WINDOW_ALIGN_NONE, choices=list(WINDOW_ALIGN_CHOICES),
        help=(
            f"grid: window ends on the {SCRAPE_INTERVAL_MS} ms gateway grid, read (start, end] - "
            "the phase-aligned controller's window (use with --step-ms 10000; the calibration "
            "fit plan does). none: free-phase windows [start, end) (e.g. the old 5 s step)."
        ),
    )
    ap.add_argument("--percentile-mode", default="bucket_upper", choices=["bucket_upper", "interpolated"])
    ap.add_argument("--min-latency-samples", type=int, default=slo_labels.DEFAULT_MIN_LATENCY_SAMPLES)
    # The divisor for the queue average. It MUST equal the spacing of the samples actually
    # consumed: the sidecar capture cadence in `raw` mode (campaign value 1000), the
    # gateway cadence in `live` mode. A mismatch is refused, not silently rescaled.
    ap.add_argument("--instant-sample-ms", type=int, default=SCRAPE_INTERVAL_MS)
    ap.add_argument(
        "--instant-grid", default=INSTANT_GRID_RAW, choices=list(INSTANT_GRID_CHOICES),
        help=(
            "raw: every sidecar sample (ground truth, 1 Hz in the campaign). "
            f"live: only samples tagged on_live_grid, i.e. the {SCRAPE_INTERVAL_MS} ms signal the "
            "controller consumes — use this for a theta fit."
        ),
    )
    ap.add_argument(
        "--gap-threshold", type=float, default=0.0,
        help="threshold for the observability-gap metric (default 0.0 = any non-zero waiting)",
    )
    ap.add_argument(
        "--gap-key", default=DEFAULT_GAP_KEY,
        help=f"sidecar key the observability gap is measured on (default {DEFAULT_GAP_KEY})",
    )
    ap.add_argument("--ttft-slo-ms", type=float, default=None,
                    help="fixed TTFT p95 SLO (the slo_label_fixed arm; default: registry)")
    ap.add_argument("--tpot-slo-ms", type=float, default=None,
                    help="p95 TPOT SLO of every arm (default: registry)")
    # The primary label's TTFT mode (D6' slowdown by default, from the registry slo block).
    slo_labels.add_label_arguments(ap, include_fixed=False)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--routable-pods", type=int, default=1)
    ap.add_argument("--assigned-replicas", type=int, default=1)
    ap.add_argument("--disk-warn-gib", type=float, default=r3_grid.DEFAULT_DISK_WARN_BYTES / 1024**3)
    args = ap.parse_args(argv)

    step_ms = args.step_ms if args.step_ms is not None else args.window_ms
    raw_dir = Path(args.raw_dir)

    size = _raw_size_bytes(raw_dir)
    warn_bytes = int(args.disk_warn_gib * 1024**3)
    note = "  !! EXCEEDS WARN THRESHOLD" if size > warn_bytes else ""
    print(f"reading raw from {raw_dir}: {size / 1024**2:.1f} MiB on disk{note}")
    print(
        f"instant cadence: --instant-grid {args.instant_grid} "
        f"--instant-sample-ms {args.instant_sample_ms} (live grid SCRAPE_INTERVAL_MS={SCRAPE_INTERVAL_MS} ms)"
    )

    from tre_common.registry import load_registry

    registry = load_registry(args.registry)
    spec = registry.model(args.model)
    if args.ttft_slo_ms is None:
        args.ttft_slo_ms = spec.slo.ttft_p95_ms
    if args.tpot_slo_ms is None:
        args.tpot_slo_ms = spec.slo.tpot_p95_ms
    if args.label_registry is None:
        args.label_registry = args.registry
    primary = slo_labels.label_def_from_args(args, args.model)
    arms = slo_labels.resolve_arms(primary)
    print(f"labels: primary {primary.ttft_slo_mode} ({', '.join(sorted(arms))})")
    ledger = load_ledgers(args.ledger) if args.ledger else {}

    rows: list[dict] = []
    cells: list[str] = []
    cell_dirs: list[str] = []
    gap_per_cell: dict[str, ObservabilityGap] = {}
    exclude = set(args.exclude_cell_id or [])
    if args.held_out_index:
        index_doc = json.loads(Path(args.held_out_index).read_text(encoding="utf-8"))
        exclude |= held_out_cell_ids(index_doc)
    cell_files, skipped_cells = discover_cell_files(
        raw_dir, exclude=exclude, only=args.only_cell_id or (),
        only_shapes=args.only_shape or (), model=args.model,
    )
    if skipped_cells:
        print(f"skipping {len(skipped_cells)} cell(s) by id: {', '.join(sorted(set(skipped_cells)))}")
    low_n = 0
    for raw_path in cell_files:
        cell_id = raw_path.stem
        try:
            cell = r3_grid.GridCell.from_scenario_id(cell_id)
        except ValueError:
            print(f"skip {raw_path.name}: not a grid cell file")
            continue
        records, instant_samples, guard, unmatched = load_cell_capture(raw_path)
        if unmatched:
            print(f"WARNING: cell {cell_id}: {unmatched} failure record(s) matched no raw request")
        # The drive's own bounds when the guard recorded them, so offline windows sit
        # exactly where the online driver put them; the data span otherwise.
        cell_rows = label_cell(
            records, instant_samples, cell, spec,
            label=arms,
            window_ms=args.window_ms, step_ms=step_ms,
            percentile_mode=args.percentile_mode,
            min_latency_samples=args.min_latency_samples,
            instant_sample_interval_ms=args.instant_sample_ms,
            instant_grid=args.instant_grid,
            start_ms=_as_int(guard.get("start_ms")),
            end_ms=_as_int(guard.get("end_ms")),
            truncated_at_ts_ms=_as_int(guard.get("truncated_at_ts_ms")),
            routable_pods=args.routable_pods, assigned_replicas=args.assigned_replicas,
            window_align=args.window_align,
        )
        if ledger:
            entry = ledger.get(cell_id) or {}
            mark_warmup(cell_rows, start_ms=_as_int(guard.get("start_ms")),
                        warmup_s=entry.get("warmup_s"))
            for row in cell_rows:
                for key in ("role", "split", "primitive", "stage"):
                    row[key] = entry.get(key)
        low_n += sum(
            1 for row in cell_rows
            if (row.get(slo_labels.COMPLETED_REQUESTS_COLUMN) or 0) < primary.min_completed_requests
        )
        rows.extend(cell_rows)
        cells.append(cell_id)
        cell_dirs.append(f"{raw_path.parent.name}/{cell_id}")
        # The gap is measured on the FULL sidecar: it is precisely the comparison between
        # the 1 Hz truth and its live-grid subsample, so it does not depend on --instant-grid.
        if sidecar_has_live_grid_tags(instant_samples):
            gap_per_cell[cell_id] = observability_gap(
                instant_samples, window_ms=args.window_ms,
                key=args.gap_key, threshold=args.gap_threshold,
            )
        print(f"cell {cell_id}: {len(cell_rows)} windows")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    r3_grid.write_csv(rows, out, extra_columns=LEDGER_COLUMNS if ledger else ())
    print(f"wrote {len(rows)} rows to {out} (window={args.window_ms}ms step={step_ms}ms "
          f"align={args.window_align})")
    print(f"{low_n} window(s) with completed_requests < {primary.min_completed_requests} "
          "(unlabeled unless unserved)")

    gap_overall = combine_observability_gaps(list(gap_per_cell.values()))
    meta = build_meta(
        model=args.model,
        raw_dir=raw_dir,
        cells=cells,
        window_ms=args.window_ms,
        step_ms=step_ms,
        instant_sample_ms=args.instant_sample_ms,
        instant_grid=args.instant_grid,
        percentile_mode=args.percentile_mode,
        min_latency_samples=args.min_latency_samples,
        routable_pods=args.routable_pods,
        assigned_replicas=args.assigned_replicas,
        rows=len(rows),
        gap_per_cell=gap_per_cell,
        gap_overall=gap_overall,
        git_sha=git_short_sha(Path(__file__).resolve().parents[2]),
        label=slo_labels.label_definition(
            arms, min_latency_samples=args.min_latency_samples,
            window_membership="(start, end]" if args.window_align == WINDOW_ALIGN_GRID else "[start, end)",
        ),
    )
    # What was actually selected on disk, so a family CSV's membership is auditable.
    meta["only_shapes"] = sorted(args.only_shape or [])
    meta["window_align"] = args.window_align
    meta["cell_dirs"] = cell_dirs
    meta["ledgers"] = list(args.ledger or [])
    meta["windows_below_min_n"] = low_n
    meta_path = write_meta(out, meta)
    print(f"wrote cadence metadata to {meta_path}")
    if gap_overall is not None:
        print(
            f"observability gap ({args.gap_key} > {args.gap_threshold}): "
            f"{gap_overall.gap:.3f} "
            f"({gap_overall.live_crossings}/{gap_overall.raw_crossings} crossings seen on the live grid)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

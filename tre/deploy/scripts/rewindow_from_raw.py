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
  * trs            -> r3_grid.compute_window_results (the shared time-constant TRSComputer),
                      so the trs column is byte-identical to the online path.
  * row assembly   -> r3_grid.window_row / write_csv.

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
    [window_start, window_end), matching when a vLLM completion increments the histograms.
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
) -> dict:
    """The record that answers "which cadence did this fit use" (plus the gap metric)."""
    meta: dict = {
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


def enumerate_windows(start_ms: int, end_ms: int, window_ms: int, step_ms: int) -> list[tuple[int, int]]:
    """Windows [w, w+window_ms) advancing by step_ms (step_ms==window_ms -> tumbling).
    Mirrors the online driver's ``while w + window_ms <= end`` bound."""
    if window_ms <= 0 or step_ms <= 0:
        raise ValueError("window_ms and step_ms must be positive")
    windows: list[tuple[int, int]] = []
    w = start_ms
    while w + window_ms <= end_ms:
        windows.append((w, w + window_ms))
        w += step_ms
    return windows


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
) -> ModelWindowMetrics:
    """Aggregate raw per-request + instant records into one ModelWindowMetrics, using the
    same 口径 as MetricsStore._aggregate_model (see module docstring)."""
    in_window = [
        r for r in records
        if r.get("done_ts_ms") is not None and window_start_ms <= r["done_ts_ms"] < window_end_ms
    ]
    prompt_tokens = sum(r["input_tokens"] for r in in_window if r.get("input_tokens") is not None)
    generation_tokens = sum(r["output_tokens"] for r in in_window if r.get("output_tokens") is not None)

    ttft_samples = [r["ttft_ms"] for r in in_window if r.get("ttft_ms") is not None]
    tpot_samples = [r["tpot_ms"] for r in in_window if r.get("tpot_ms") is not None]
    e2e_samples = [r["e2e_ms"] for r in in_window if r.get("e2e_ms") is not None]

    # queue: instant samples are inclusive [start, end] and divided by expected_samples,
    # exactly as MetricsStore._instant_avg does.
    inst = [
        s for s in instant_samples
        if s.get("ts_ms") is not None and window_start_ms <= s["ts_ms"] <= window_end_ms
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
) -> list[dict]:
    """Re-window one cell's raw into calibration CSV rows (reusing r3_grid.window_row +
    compute_window_results for the trs column)."""
    instant_sample_interval_ms = resolve_instant_cadence(
        instant_samples,
        instant_grid=instant_grid,
        instant_sample_ms=instant_sample_interval_ms,
        source=cell.scenario_id,
    )
    instant_samples = select_instant_samples(instant_samples, instant_grid)
    if start_ms is None or end_ms is None:
        span = _time_span(records, instant_samples)
        if span is None:
            return []
        start_ms = span[0] if start_ms is None else start_ms
        end_ms = span[1] if end_ms is None else end_ms
    windows_ms = enumerate_windows(start_ms, end_ms, window_ms, step_ms)
    metrics = [
        aggregate_window(
            records, instant_samples, spec.name, ws, we,
            percentile_mode=percentile_mode,
            min_latency_samples=min_latency_samples,
            instant_sample_interval_ms=instant_sample_interval_ms,
            routable_pods=routable_pods, assigned_replicas=assigned_replicas,
        )
        for ws, we in windows_ms
    ]
    results = r3_grid.compute_window_results(metrics, spec)
    return [r3_grid.window_row(cell, wm, result.TRS, result.Q_ctl) for wm, result in zip(metrics, results)]


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


def discover_cell_files(
    raw_dir: Path,
    *,
    exclude: Iterable[str] = (),
    only: Iterable[str] = (),
) -> tuple[list[Path], list[str]]:
    """(cell raw files to re-window, cell ids skipped).

    Searched **recursively**, because a campaign writes one directory per cell under the
    raw root; a flat glob finds nothing there and produces an empty CSV without saying so.

    ``only`` wins over ``exclude`` when both are given: an explicit inclusion list is a
    stronger statement than a default exclusion.
    """
    excluded = {str(c) for c in exclude}
    included = {str(c) for c in only}
    kept: list[Path] = []
    skipped: list[str] = []
    for path in sorted(raw_dir.rglob("*.jsonl")):
        if path.name.endswith(".instant.jsonl") or path.name.endswith(".failures.jsonl"):
            continue
        cell_id = path.stem
        if included:
            if cell_id not in included:
                skipped.append(cell_id)
                continue
        elif cell_id in excluded:
            skipped.append(cell_id)
            continue
        kept.append(path)
    return kept, skipped


def main() -> int:
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
    ap.add_argument("--held-out-index", default=None,
                    help="schedule INDEX.json; every entry marked held_out is excluded, "
                         "in addition to --exclude-cell-id")
    ap.add_argument("--output", required=True, help="re-windowed CSV path")
    ap.add_argument("--window-ms", type=int, required=True)
    ap.add_argument("--step-ms", type=int, default=None, help="slide step; default = window-ms (tumbling)")
    ap.add_argument("--percentile-mode", default="bucket_upper", choices=["bucket_upper", "interpolated"])
    ap.add_argument("--min-latency-samples", type=int, default=10)
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
    ap.add_argument("--registry", default=None)
    ap.add_argument("--routable-pods", type=int, default=1)
    ap.add_argument("--assigned-replicas", type=int, default=1)
    ap.add_argument("--disk-warn-gib", type=float, default=r3_grid.DEFAULT_DISK_WARN_BYTES / 1024**3)
    args = ap.parse_args()

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

    rows: list[dict] = []
    cells: list[str] = []
    gap_per_cell: dict[str, ObservabilityGap] = {}
    exclude = set(args.exclude_cell_id or [])
    if args.held_out_index:
        index_doc = json.loads(Path(args.held_out_index).read_text(encoding="utf-8"))
        exclude |= held_out_cell_ids(index_doc)
    cell_files, skipped_cells = discover_cell_files(
        raw_dir, exclude=exclude, only=args.only_cell_id or ()
    )
    if skipped_cells:
        print(f"skipping {len(skipped_cells)} cell(s) by id: {', '.join(sorted(set(skipped_cells)))}")
    for raw_path in cell_files:
        cell_id = raw_path.stem
        raw_dir_for_cell = raw_path.parent
        try:
            cell = r3_grid.GridCell.from_scenario_id(cell_id)
        except ValueError:
            print(f"skip {raw_path.name}: not a grid cell file")
            continue
        records = load_jsonl(raw_path)
        instant_samples = load_jsonl(raw_dir_for_cell / f"{cell_id}.instant.jsonl")
        cell_rows = rewindow_cell(
            records, instant_samples, cell, spec,
            window_ms=args.window_ms, step_ms=step_ms,
            percentile_mode=args.percentile_mode,
            min_latency_samples=args.min_latency_samples,
            instant_sample_interval_ms=args.instant_sample_ms,
            instant_grid=args.instant_grid,
            routable_pods=args.routable_pods, assigned_replicas=args.assigned_replicas,
        )
        rows.extend(cell_rows)
        cells.append(cell_id)
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
    r3_grid.write_csv(rows, out)
    print(f"wrote {len(rows)} rows to {out} (window={args.window_ms}ms step={step_ms}ms)")

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
    )
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

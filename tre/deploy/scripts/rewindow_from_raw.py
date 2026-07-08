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
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from tre_common.metrics_schema import ModelWindowMetrics
from tre_common.percentile import histogram_percentile
from tre_common.rediskeys import SCRAPE_INTERVAL_MS

from scripts import r3_grid


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
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    routable_pods: int = 1,
    assigned_replicas: int = 1,
) -> list[dict]:
    """Re-window one cell's raw into calibration CSV rows (reusing r3_grid.window_row +
    compute_window_results for the trs column)."""
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
    return sum(p.stat().st_size for p in raw_dir.glob("*.jsonl"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw-dir", required=True, help="dir holding <cell_id>.jsonl (+ .instant.jsonl)")
    ap.add_argument("--output", required=True, help="re-windowed CSV path")
    ap.add_argument("--window-ms", type=int, required=True)
    ap.add_argument("--step-ms", type=int, default=None, help="slide step; default = window-ms (tumbling)")
    ap.add_argument("--percentile-mode", default="bucket_upper", choices=["bucket_upper", "interpolated"])
    ap.add_argument("--min-latency-samples", type=int, default=10)
    # Must equal the r3 sidecar sampling cadence (== gateway scrape cadence,
    # SCRAPE_INTERVAL_MS) so offline expected_samples matches the actual instant samples in
    # each window; a mismatch halves the offline queue vs. the online path.
    ap.add_argument("--instant-sample-ms", type=int, default=SCRAPE_INTERVAL_MS)
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

    from tre_common.registry import load_registry

    registry = load_registry(args.registry)
    spec = registry.model(args.model)

    rows: list[dict] = []
    cell_files = sorted(p for p in raw_dir.glob("*.jsonl") if not p.name.endswith(".instant.jsonl"))
    for raw_path in cell_files:
        cell_id = raw_path.stem
        try:
            cell = r3_grid.GridCell.from_scenario_id(cell_id)
        except ValueError:
            print(f"skip {raw_path.name}: not a grid cell file")
            continue
        records = load_jsonl(raw_path)
        instant_samples = load_jsonl(raw_dir / f"{cell_id}.instant.jsonl")
        cell_rows = rewindow_cell(
            records, instant_samples, cell, spec,
            window_ms=args.window_ms, step_ms=step_ms,
            percentile_mode=args.percentile_mode,
            min_latency_samples=args.min_latency_samples,
            instant_sample_interval_ms=args.instant_sample_ms,
            routable_pods=args.routable_pods, assigned_replicas=args.assigned_replicas,
        )
        rows.extend(cell_rows)
        print(f"cell {cell_id}: {len(cell_rows)} windows")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    r3_grid.write_csv(rows, out)
    print(f"wrote {len(rows)} rows to {out} (window={args.window_ms}ms step={step_ms}ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

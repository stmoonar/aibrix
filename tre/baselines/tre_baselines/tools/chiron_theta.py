"""Derive the per-model Chiron IBP threshold Theta from request-arrival traces.

    python3 -m tre_baselines.tools.chiron_theta --trace T [--trace T2 ...] --interval-s 5 \
        [--method peak_mean|adjacent_p99]

Per model, arrivals are counted per ``interval`` bin over the trace's span, then:

``peak_mean`` (default)
    Theta = clamp(mean / peak, 0.1, 0.9) with peak = p99 of the bin counts and mean their
    mean over all bins. The paper's example (a spike of 3x the usual load -> Theta = 1/3)
    reads Theta as the idle headroom that absorbs the spike: at IBP = Theta the busy
    instances are Theta of the fleet, so a load 1/Theta x the mean still fits.
``adjacent_p99``
    ratios of adjacent non-zero bins, r = p99 of the ratios, Theta = clamp(1/r, 0.1, 0.9):
    sized for the largest *step* between two bins instead of the peak over the mean.

Prints YAML ``theta: {model: value}``. The paper sets interval = model load time; ours
defaults to 5 s (wake ~2.3 s + one tick), not specified in paper; chosen.

Accepted trace formats:
* replayer trace: JSON object ``{model: [{start_time, end_time, rps, ...}, ...]}``
  (expected arrivals per bin = integral of rps over the bin);
* JSON object ``{model: [t, ...]}`` of arrival times;
* JSON list or JSONL of records with a model field (``--model-field``, default ``model``)
  and an arrival-time field (``--time-field``, else the first of ``ARRIVAL_FIELDS``).
Times are seconds unless ``--time-unit ms``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

ARRIVAL_FIELDS = ("arrival_time", "arrival_s", "arrival", "timestamp", "ts", "time", "t", "start_time")
THETA_MIN, THETA_MAX = 0.1, 0.9
DEFAULT_INTERVAL_S = 5.0  # not specified in paper; chosen
METHODS = ("peak_mean", "adjacent_p99")
DEFAULT_METHOD = "peak_mean"


class TraceFormatError(ValueError):
    pass


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (small samples: p99 is the max, not an interpolation)."""
    s = sorted(values)
    return s[max(0, math.ceil(q * len(s)) - 1)]


def _load(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for i, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise TraceFormatError(f"{path}: neither JSON nor JSONL (line {i}: {exc})") from exc
        return rows


def _is_segment(x: Any) -> bool:
    return isinstance(x, dict) and {"start_time", "end_time", "rps"} <= set(x)


def load_trace(path: Path, model_field: str = "model", time_field: Optional[str] = None,
               time_scale: float = 1.0) -> tuple[dict[str, list[tuple[float, float, float]]], dict[str, list[float]]]:
    """Returns (segments per model as (start, end, rps), arrival times per model)."""
    data = _load(path)
    segs: dict[str, list[tuple[float, float, float]]] = {}
    arrs: dict[str, list[float]] = {}
    if isinstance(data, dict):
        for model, items in data.items():
            if not isinstance(items, list):
                raise TraceFormatError(f"{path}: model {model!r} must map to a list")
            for it in items:
                if _is_segment(it):
                    segs.setdefault(model, []).append(
                        (float(it["start_time"]), float(it["end_time"]), float(it["rps"])))
                elif isinstance(it, (int, float)):
                    arrs.setdefault(model, []).append(float(it) * time_scale)
                else:
                    raise TraceFormatError(f"{path}: model {model!r} has an entry that is neither a "
                                           "segment {start_time,end_time,rps} nor an arrival time")
    elif isinstance(data, list):
        for i, rec in enumerate(data):
            if not isinstance(rec, dict) or model_field not in rec:
                raise TraceFormatError(f"{path}: record {i} is not an object with model field {model_field!r}")
            key = time_field or next((f for f in ARRIVAL_FIELDS if f in rec), None)
            if key is None or key not in rec:
                raise TraceFormatError(f"{path}: record {i} has no arrival-time field "
                                       f"(tried {ARRIVAL_FIELDS}; use --time-field)")
            arrs.setdefault(str(rec[model_field]), []).append(float(rec[key]) * time_scale)
    else:
        raise TraceFormatError(f"{path}: top level must be a JSON object, list or JSONL")
    if not segs and not arrs:
        raise TraceFormatError(f"{path}: no arrivals found")
    return segs, arrs


def bin_counts(segments: Iterable[tuple[float, float, float]], arrivals: Iterable[float],
               interval_s: float) -> list[float]:
    segments, arrivals = list(segments), list(arrivals)
    lo = min([s for s, _, _ in segments] + arrivals)
    hi = max([e for _, e, _ in segments] + arrivals)
    n = max(1, int(math.ceil((hi - lo) / interval_s - 1e-9)))
    counts = [0.0] * n
    for s, e, rps in segments:
        for b in range(int((s - lo) // interval_s), min(n, int(math.ceil((e - lo) / interval_s)))):
            b0, b1 = lo + b * interval_s, lo + (b + 1) * interval_s
            counts[b] += rps * max(0.0, min(e, b1) - max(s, b0))
    for t in arrivals:
        counts[min(n - 1, int((t - lo) // interval_s))] += 1.0
    return counts


def _clamp(theta: float) -> float:
    return min(THETA_MAX, max(THETA_MIN, theta))


def theta_from_counts(counts: list[float]) -> Optional[dict]:
    """``adjacent_p99``: Theta = clamp(1 / p99(adjacent non-zero bin ratio))."""
    ratios = [b / a for a, b in zip(counts, counts[1:]) if a > 0 and b > 0]
    if not ratios:
        return None
    r = _percentile(ratios, 0.99)
    return {"method": "adjacent_p99", "r": r, "theta": _clamp(1.0 / r), "n_ratios": len(ratios)}


def theta_peak_mean(counts: list[float]) -> Optional[dict]:
    """``peak_mean``: Theta = clamp(mean / p99 of the bin counts)."""
    if not counts:
        return None
    peak = _percentile(counts, 0.99)
    if peak <= 0:
        return None
    mean = sum(counts) / len(counts)
    return {"method": "peak_mean", "mean": mean, "peak": peak, "theta": _clamp(mean / peak),
            "n_bins": len(counts)}


def compute_theta(paths: list[Path], interval_s: float = DEFAULT_INTERVAL_S, model_field: str = "model",
                  time_field: Optional[str] = None, time_scale: float = 1.0,
                  method: str = DEFAULT_METHOD) -> dict[str, dict]:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    segs: dict[str, list] = {}
    arrs: dict[str, list] = {}
    for p in paths:
        s, a = load_trace(p, model_field, time_field, time_scale)
        for m, v in s.items():
            segs.setdefault(m, []).extend(v)
        for m, v in a.items():
            arrs.setdefault(m, []).extend(v)
    out = {}
    for m in sorted(set(segs) | set(arrs)):
        counts = bin_counts(segs.get(m, []), arrs.get(m, []), interval_s)
        res = theta_from_counts(counts) if method == "adjacent_p99" else theta_peak_mean(counts)
        if res is None:
            raise TraceFormatError(f"model {m!r}: fewer than two adjacent non-zero bins; "
                                   "cannot estimate a ratio" if method == "adjacent_p99"
                                   else f"model {m!r}: no arrivals in any bin")
        out[m] = res
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", action="append", required=True, type=Path)
    ap.add_argument("--interval-s", type=float, default=DEFAULT_INTERVAL_S)
    ap.add_argument("--model-field", default="model")
    ap.add_argument("--time-field", default=None)
    ap.add_argument("--time-unit", choices=("s", "ms"), default="s")
    ap.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD)
    a = ap.parse_args(argv)
    if a.interval_s <= 0:
        ap.error("--interval-s must be > 0")
    try:
        res = compute_theta(a.trace, a.interval_s, a.model_field, a.time_field,
                            1e-3 if a.time_unit == "ms" else 1.0, method=a.method)
    except (TraceFormatError, OSError) as exc:
        print(f"chiron_theta: {exc}", file=sys.stderr)
        return 2
    if a.method == "adjacent_p99":
        rule = "clamp(1/p99(adjacent non-zero bin ratio)"
    else:
        rule = "clamp(mean bin count / p99 bin count"
    print(f"# interval_s={a.interval_s} method={a.method}; theta = {rule}, {THETA_MIN}, {THETA_MAX})")
    print("theta:")
    for m, r in res.items():
        if a.method == "adjacent_p99":
            note = f"r={r['r']:.3f} n={r['n_ratios']}"
        else:
            note = f"mean={r['mean']:.2f} peak={r['peak']:.2f} bins={r['n_bins']}"
        print(f"  {m}: {r['theta']:.10f}  # {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

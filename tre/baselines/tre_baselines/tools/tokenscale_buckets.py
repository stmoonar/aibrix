"""TokenScale 3x3 cells with the paper's boundaries, their calibration points, and the
PreServe mu shape range, from the per-request lists the campaign replays.

Decision 2026-10-26 (``docs/campaign-v2-20261026-plan.md`` #4; v1 D49 calibration-point rule):

* **Boundaries** = TokenScale Table II / DynamoLLM (arXiv 2408.00741, Table IV): input
  ``< 256`` (S) / ``< 1024`` (M) / else (L; an input above 8192 is L too), output ``< 100``
  / ``< 350`` / ``>= 350``. Not cut from our data. The policy's ``bucket_edges`` are
  inclusive upper bounds (``x <= e1 -> 0``), so the integer form is in ``[255, 1023]``,
  out ``[99, 349]``.
* **Calibration point** of a cell = the arithmetic mean (in, out) of the replayed requests
  that fall in it (v1 D49 (2): under a linear service-time model this keeps
  ``sum lambda'/V`` unbiased for the pooled load), pooled per model over every given trace,
  rounded and clamped into its own cell. An empty cell takes the mean input of its input
  class and the mean output of its output class (v1 rule). Two roles: the shape the V ladder
  of that cell measures, and the output ``lambda'`` counts for a request in that cell.
* **mix shape** per model = the mean (in, out) of all its requests (the reference request of
  PreServe's normalized-latency SLO, v1 D58).
* **P share range** (PreServe mu scheme B, v1 D58): per model the share ``P_w / (P_w + D_w)`` of
  every complete ``--window-s`` window with at least ``--min-reqs`` requests, ``P`` = sum of
  input tokens, ``D`` = sum of ``max_output_tokens``; reported per model and pooled (min / max,
  p5 / p95). Scheme B keeps the cells whose own share lies in the pooled [min, max].

Input: per trace ``NAME=PATH`` to a ``design.json`` or ``traces_tre.effective.json`` of
``tre_replayer.tracegen`` (records with ``model_name``, ``prompt_length`` = the engine's
``usage.prompt_tokens``, chat template included, and ``max_output_tokens``; ``timestamp``).
The duration (complete windows only) comes from the ``manifest.json`` next to the file,
else ``--duration-s``, else the last request time.

    python3 -m tre_baselines.tools.tokenscale_buckets --out buckets-report.yaml \\
        --trace-root <set root> --traces T7_tight_pool,Simul_spike,... [--seed 1] [--file design.json]

(or explicit ``--trace NAME=PATH`` entries). The set is an argument: nothing here is frozen
to one trace set.

Writes the report (YAML) and prints its sha256; every input's sha256 is in it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import yaml

#: Paper boundaries: an input / output below the k-th bound is in class k (else the last).
PAPER_IN_BOUNDS = (256, 1024)
PAPER_OUT_BOUNDS = (100, 350)
N_IN = N_OUT = 3
#: Inclusive upper bounds of the policy's ``bucket_edges`` for integer token counts.
POLICY_EDGES = {"in": [PAPER_IN_BOUNDS[0] - 1, PAPER_IN_BOUNDS[1] - 1],
                "out": [PAPER_OUT_BOUNDS[0] - 1, PAPER_OUT_BOUNDS[1] - 1]}
SOURCE = "TokenScale Table II / DynamoLLM (arXiv 2408.00741) Table IV"

#: (model, t_s, in, out)
Req = tuple[str, float, int, int]


class TraceFormatError(ValueError):
    pass


def class_of(x: float, bounds: Sequence[float]) -> int:
    """0 if x < bounds[0], 1 if x < bounds[1], else 2 (no upper limit: > 8192 is long)."""
    for k, b in enumerate(bounds):
        if x < b:
            return k
    return len(bounds)


def cell_of(tin: float, tout: float) -> tuple[int, int]:
    return class_of(tin, PAPER_IN_BOUNDS), class_of(tout, PAPER_OUT_BOUNDS)


def _range(k: int, bounds: Sequence[int]) -> tuple[int, Optional[int]]:
    """Inclusive integer range [lo, hi] of class k (hi None = unbounded)."""
    lo = 1 if k == 0 else bounds[k - 1]
    hi = bounds[k] - 1 if k < len(bounds) else None
    return lo, hi


def clamp_into(x: float, k: int, bounds: Sequence[int]) -> int:
    lo, hi = _range(k, bounds)
    v = int(round(x))
    v = max(lo, v)
    return v if hi is None else min(hi, v)


def load_requests(path: str) -> list[Req]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TraceFormatError(f"{path}: {exc}") from exc
    if not isinstance(doc, list):
        raise TraceFormatError(f"{path}: expected a JSON list of request records")
    out: list[Req] = []
    for i, r in enumerate(doc):
        try:
            out.append((str(r["model_name"]), float(r["timestamp"]), int(r["prompt_length"]),
                        int(r["max_output_tokens"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise TraceFormatError(f"{path}: record {i} needs model_name, timestamp, prompt_length, "
                                   f"max_output_tokens ({exc!r})") from exc
    return out


def trace_duration(path: str, default: Optional[float], reqs: Sequence[Req]) -> float:
    man = Path(path).with_name("manifest.json")
    if man.is_file():
        try:
            d = json.loads(man.read_text(encoding="utf-8")).get("duration_s")
            if d:
                return float(d)
        except (OSError, ValueError):
            pass
    if default:
        return float(default)
    return max((t for _, t, _, _ in reqs), default=0.0)


def _q(vals: Sequence[float], q: float) -> Optional[float]:
    v = sorted(vals)
    if not v:
        return None
    return v[min(len(v) - 1, max(0, math.ceil(q * len(v)) - 1))]


def window_shares(reqs: Iterable[Req], duration_s: float, window_s: float, min_reqs: int) -> dict[str, list[float]]:
    """model -> P share of every complete window with >= min_reqs requests."""
    n_win = int(math.floor(duration_s / window_s + 1e-9))
    acc: dict[tuple[str, int], list[float]] = {}
    for m, t, i, o in reqs:
        w = int(t // window_s)
        if w < 0 or w >= n_win:
            continue
        a = acc.setdefault((m, w), [0, 0.0, 0.0])
        a[0] += 1
        a[1] += i
        a[2] += o
    out: dict[str, list[float]] = {}
    for (m, _w), (n, p, d) in sorted(acc.items()):
        if n >= min_reqs and p + d > 0:
            out.setdefault(m, []).append(p / (p + d))
    return out


def _share_stats(shares: Sequence[float]) -> dict[str, Any]:
    return {"windows": len(shares), "min": round(min(shares), 4), "max": round(max(shares), 4),
            "p5": round(_q(shares, 0.05), 4), "p95": round(_q(shares, 0.95), 4)}


def report(traces: Mapping[str, list[Req]], durations: Mapping[str, float], *, window_s: float = 30.0,
           min_reqs: int = 10) -> dict[str, Any]:
    """The report (without sources) for the given traces (name -> requests)."""
    per_model: dict[str, list[tuple[str, int, int]]] = {}
    for name, reqs in traces.items():
        for m, _t, i, o in reqs:
            per_model.setdefault(m, []).append((name, i, o))
    centers: dict[str, list[list[list[int]]]] = {}
    cells: dict[str, Any] = {}
    mix: dict[str, Any] = {}
    for m in sorted(per_model):
        rs = per_model[m]
        n = len(rs)
        by: dict[tuple[int, int], list[tuple[str, int, int]]] = {}
        for r in rs:
            by.setdefault(cell_of(r[1], r[2]), []).append(r)
        in_cls = {k: [r[1] for r in rs if class_of(r[1], PAPER_IN_BOUNDS) == k] for k in range(N_IN)}
        out_cls = {k: [r[2] for r in rs if class_of(r[2], PAPER_OUT_BOUNDS) == k] for k in range(N_OUT)}
        grid: list[list[list[int]]] = []
        detail: dict[str, Any] = {}
        for i in range(N_IN):
            row = []
            for j in range(N_OUT):
                members = by.get((i, j), [])
                if members:
                    mi = sum(r[1] for r in members) / len(members)
                    mo = sum(r[2] for r in members) / len(members)
                    how = "mean of its requests"
                elif in_cls[i] and out_cls[j]:
                    mi = sum(in_cls[i]) / len(in_cls[i])
                    mo = sum(out_cls[j]) / len(out_cls[j])
                    how = "empty: mean input of its input class, mean output of its output class"
                else:
                    raise TraceFormatError(f"{m}: cell {i}{j} is empty and so is its input or output class")
                c = [clamp_into(mi, i, PAPER_IN_BOUNDS), clamp_into(mo, j, PAPER_OUT_BOUNDS)]
                assert cell_of(*c) == (i, j), (m, i, j, c)
                row.append(c)
                per_trace: dict[str, int] = {}
                for r in members:
                    per_trace[r[0]] = per_trace.get(r[0], 0) + 1
                detail[f"{i}{j}"] = {
                    "center": c, "how": how, "n": len(members), "share": round(len(members) / n, 4),
                    "token_share": round(sum(r[1] + r[2] for r in members) / max(1, sum(r[1] + r[2] for r in rs)), 4),
                    "mean_in": round(mi, 1), "mean_out": round(mo, 1),
                    "in_range": [min((r[1] for r in members), default=None), max((r[1] for r in members), default=None)],
                    "out_range": [min((r[2] for r in members), default=None), max((r[2] for r in members), default=None)],
                    "p_share": round(c[0] / (c[0] + c[1]), 4),
                    "per_trace": dict(sorted(per_trace.items())),
                }
            grid.append(row)
        centers[m] = grid
        cells[m] = detail
        mi = sum(r[1] for r in rs) / n
        mo = sum(r[2] for r in rs) / n
        mix[m] = {"in": int(round(mi)), "out": int(round(mo)), "n": n, "over_8192": sum(1 for r in rs if r[1] > 8192),
                  "max_in": max(r[1] for r in rs), "max_out": max(r[2] for r in rs)}
    shares: dict[str, list[float]] = {}
    for name, reqs in traces.items():
        for m, s in window_shares(reqs, durations[name], window_s, min_reqs).items():
            shares.setdefault(m, []).extend(s)
    pooled = [x for v in shares.values() for x in v]
    p_share = {"window_s": window_s, "min_reqs": min_reqs,
               "per_model": {m: _share_stats(v) for m, v in sorted(shares.items()) if v},
               "pooled": _share_stats(pooled) if pooled else None}
    return {
        "rule": {"source": SOURCE, "in": "< 256 | < 1024 | else (> 8192 included)",
                 "out": "< 100 | < 350 | >= 350",
                 "center": "arithmetic mean (in, out) of the cell's replayed requests, pooled over the traces (v1 D49)"},
        "bucket_edges": {"*": POLICY_EDGES},
        "bucket_centers": centers,
        "mix_shape": mix,
        "p_share": p_share,
        "cells": cells,
    }


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _kv(items: Sequence[str]) -> dict[str, str]:
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not k or not v:
            raise argparse.ArgumentTypeError(f"--trace {it!r} is not NAME=PATH")
        out[k] = v
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace-root", default=None,
                    help="trace set root: <root>/<trace>/seed<seed>/<file> for every name in --traces")
    ap.add_argument("--traces", default=None, help="comma list of trace names under --trace-root")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--file", default="design.json", help="request list under <root>/<trace>/seed<k>/")
    ap.add_argument("--trace", action="append", default=[], metavar="NAME=PATH", help="an explicit list (repeatable)")
    ap.add_argument("--out", required=True, help="report YAML to write")
    ap.add_argument("--window-s", type=float, default=30.0)
    ap.add_argument("--min-reqs", type=int, default=10)
    ap.add_argument("--duration-s", type=float, default=None, help="when a trace has no manifest.json")
    a = ap.parse_args(argv)
    try:
        paths = _kv(a.trace)
        if bool(a.trace_root) != bool(a.traces):
            raise argparse.ArgumentTypeError("--trace-root and --traces go together")
        if a.trace_root:
            for name in filter(None, (x.strip() for x in a.traces.split(","))):
                paths[name] = str(Path(a.trace_root) / name / f"seed{a.seed}" / a.file)
        if not paths:
            raise argparse.ArgumentTypeError("give --trace-root with --traces, or --trace NAME=PATH")
        traces = {n: load_requests(p) for n, p in paths.items()}
    except (TraceFormatError, argparse.ArgumentTypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    durations = {n: trace_duration(paths[n], a.duration_s, traces[n]) for n in traces}
    try:
        doc = report(traces, durations, window_s=a.window_s, min_reqs=a.min_reqs)
    except TraceFormatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    doc["sources"] = {n: {"path": str(Path(p).resolve()), "sha256": sha256_file(p), "requests": len(traces[n]),
                          "duration_s": durations[n]} for n, p in sorted(paths.items())}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, allow_unicode=True), encoding="utf-8")
    print(f"{out} sha256 {sha256_file(str(out))}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

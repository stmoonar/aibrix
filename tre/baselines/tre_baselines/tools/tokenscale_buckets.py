"""Derive the TokenScale 3x3 bucket edges from traces (tertiles of input / max_tokens).

Usage::

    python -m tre_baselines.tools.tokenscale_buckets --trace t1.json [--trace t2.json ...] [--model m]

Prints YAML: ``bucket_edges`` (paste into the policy params) and ``bucket_centers`` (per
bucket the weighted median (in, out) of its requests; the profiling points for
``tokenscale_profile``). Edges are inclusive upper bounds (x <= e1 -> bucket 0).

Accepted trace formats (JSON or JSONL):

* replayer segment trace (``tre_replayer/traces``): ``{model: [{start_time, end_time, rps,
  input_tokens | input_tokens_dist{low,high}, max_tokens | max_tokens_dist{low,high}}]}``.
  Each segment weighs ``rps * (end_time - start_time)`` requests; a distribution
  contributes its geometric midpoint.
* per-request records, a JSON list or JSONL of objects with an input length
  (``in_tokens`` | ``input_tokens`` | ``prompt_tokens``), an output length (``max_tokens`` |
  ``max_output_tokens`` | ``out_tokens``) and optionally ``model`` (else ``--default-model``).
  A ``{model: [records]}`` mapping works too.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

IN_KEYS = ("in_tokens", "input_tokens", "prompt_tokens")
OUT_KEYS = ("max_tokens", "max_output_tokens", "out_tokens")

#: (in, out, weight)
Sample = tuple[float, float, float]


class TraceFormatError(ValueError):
    pass


def _first(rec: dict, keys: Iterable[str]) -> Optional[float]:
    for k in keys:
        if rec.get(k) is not None:
            return float(rec[k])
    return None


def _dist_mid(spec: Any) -> Optional[float]:
    if isinstance(spec, dict) and spec.get("low") and spec.get("high"):
        return math.sqrt(float(spec["low"]) * float(spec["high"]))
    return None


def _segment_sample(where: str, seg: dict) -> Sample:
    try:
        weight = float(seg["rps"]) * (float(seg["end_time"]) - float(seg["start_time"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise TraceFormatError(f"{where}: segment needs start_time/end_time/rps ({exc})") from exc
    tin = seg.get("input_tokens")
    tin = float(tin) if tin is not None else _dist_mid(seg.get("input_tokens_dist"))
    tout = seg.get("max_tokens")
    tout = float(tout) if tout is not None else _dist_mid(seg.get("max_tokens_dist"))
    if tin is None or tout is None:
        raise TraceFormatError(f"{where}: segment lacks input_tokens/max_tokens (or their _dist)")
    return tin, tout, weight


def _record_sample(where: str, rec: Any) -> Sample:
    if not isinstance(rec, dict):
        raise TraceFormatError(f"{where}: expected an object, got {type(rec).__name__}")
    if "start_time" in rec and "rps" in rec:
        return _segment_sample(where, rec)
    tin, tout = _first(rec, IN_KEYS), _first(rec, OUT_KEYS)
    if tin is None or tout is None:
        raise TraceFormatError(f"{where}: record needs one of {IN_KEYS} and one of {OUT_KEYS}")
    return tin, tout, 1.0


def load_trace(path: str | Path, default_model: str = "*") -> dict[str, list[Sample]]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise TraceFormatError(f"{path}:{n}: neither JSON nor JSONL ({exc})") from exc
    out: dict[str, list[Sample]] = {}
    if isinstance(data, list):
        for n, rec in enumerate(data):
            model = rec.get("model", default_model) if isinstance(rec, dict) else default_model
            out.setdefault(str(model), []).append(_record_sample(f"{path}[{n}]", rec))
    elif isinstance(data, dict):
        for model, items in data.items():
            if not isinstance(items, list):
                raise TraceFormatError(f"{path}: value of {model!r} must be a list")
            for n, rec in enumerate(items):
                out.setdefault(str(model), []).append(_record_sample(f"{path}[{model}][{n}]", rec))
    else:
        raise TraceFormatError(f"{path}: top level must be a list or an object")
    if not any(out.values()):
        raise TraceFormatError(f"{path}: no requests found")
    return out


def weighted_quantile(pairs: list[tuple[float, float]], q: float) -> float:
    """Smallest value whose cumulative weight reaches ``q`` of the total."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= q * total - 1e-12:
            return v
    return pairs[-1][0]


def _idx(x: float, edges: tuple[float, float]) -> int:
    return 0 if x <= edges[0] else 1 if x <= edges[1] else 2


def _num(x: float) -> float | int:
    return int(round(x))


def compute(samples: list[Sample]) -> dict[str, Any]:
    e_in = (weighted_quantile([(s[0], s[2]) for s in samples], 1 / 3), weighted_quantile([(s[0], s[2]) for s in samples], 2 / 3))
    e_out = (weighted_quantile([(s[1], s[2]) for s in samples], 1 / 3), weighted_quantile([(s[1], s[2]) for s in samples], 2 / 3))
    cells: dict[tuple[int, int], list[Sample]] = {}
    for s in samples:
        cells.setdefault((_idx(s[0], e_in), _idx(s[1], e_out)), []).append(s)
    centers: list[list[Optional[list[int]]]] = []
    for i in range(3):
        row: list[Optional[list[int]]] = []
        for j in range(3):
            cell = cells.get((i, j))
            row.append(
                [_num(weighted_quantile([(s[0], s[2]) for s in cell], 0.5)), _num(weighted_quantile([(s[1], s[2]) for s in cell], 0.5))]
                if cell else None
            )
        centers.append(row)
    return {
        "edges": {"in": [_num(e_in[0]), _num(e_in[1])], "out": [_num(e_out[0]), _num(e_out[1])]},
        "centers": centers,
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace", action="append", required=True, help="trace file (repeatable; pooled per model)")
    ap.add_argument("--model", action="append", help="restrict to this model (repeatable)")
    ap.add_argument("--default-model", default="*", help="model name for records without one")
    args = ap.parse_args(argv)
    pooled: dict[str, list[Sample]] = {}
    try:
        for path in args.trace:
            for model, s in load_trace(path, args.default_model).items():
                pooled.setdefault(model, []).extend(s)
    except (TraceFormatError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.model:
        missing = [m for m in args.model if m not in pooled]
        if missing:
            print(f"error: model(s) {missing} not in traces (have {sorted(pooled)})", file=sys.stderr)
            return 2
        pooled = {m: pooled[m] for m in args.model}
    edges, centers = {}, {}
    for model in sorted(pooled):
        res = compute(pooled[model])
        edges[model] = res["edges"]
        centers[model] = res["centers"]
    sys.stdout.write(yaml.safe_dump({"bucket_edges": edges, "bucket_centers": centers}, sort_keys=False, default_flow_style=None))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

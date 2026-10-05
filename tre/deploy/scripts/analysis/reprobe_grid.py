#!/usr/bin/env python3
"""The S3 boundary supplement's REPROBE_GRID, read from round 2's D6' b50 table.

The supplement (``run_supplement.sh``) walks ascending multiples of rho*_run2 upwards
until the first violation, then bisects; the lowest multiple must be healthy and the
highest violating, or rho* comes back as a bound (exit 3, stop for the owner). The RUN
plan (§D4) says: read round 2's S3 boundary and pick four multiples that bracket the D6'
flip. This module is that rule, written down so an unattended chain applies it the same
way every time:

* ``P_b50_rf`` present (``scripts.analysis.boundary_b50_table``: the load factor, in
  rho*_run2 units, at which half of the labelled windows violate under the primary D6'
  label): ``f = b50``, grid ``0.9 f, 1.05 f, 1.2 f, 1.4 f``.
* no b50 (no transition inside round 2's data, i.e. S3 stayed healthy up to ``rf_max``):
  grid ``rf_max, 1.25 rf_max, 1.55 rf_max, 1.9 rf_max``.

On the 2026-09-23 inputs (b50 7b 1.395, 14b 1.587; 8b none with rf_max 1.3) the rule
brackets the S3 flips the supplement then measured (1.356 / 1.987 / 1.575 x rho*_run2).

    python -m scripts.analysis.reprobe_grid --b50-table <csv> --model <m> [--shape S3]

Prints the grid (``1.26,1.46,1.67,1.95``) on stdout and the reasoning (JSON) on stderr.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

B50_FACTORS = (0.9, 1.05, 1.2, 1.4)
NO_B50_FACTORS = (1.0, 1.25, 1.55, 1.9)


def _num(text: Optional[str]) -> Optional[float]:
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def grid_for(row: dict) -> dict:
    """{"grid": [4 ascending multiples], "rule": ..., "basis": ...} for one b50 row."""
    b50 = _num(row.get("P_b50_rf"))
    rf_max = _num(row.get("rf_max"))
    if b50 is not None:
        base, factors, rule = b50, B50_FACTORS, "b50"
    elif rf_max is not None:
        base, factors, rule = rf_max, NO_B50_FACTORS, "no_b50_rf_max"
    else:
        raise ValueError(f"{row.get('model')}/{row.get('shape')}: neither P_b50_rf nor rf_max")
    grid = [round(base * f, 2) for f in factors]
    if any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError(f"grid {grid} is not strictly ascending")
    return {"grid": grid, "rule": rule, "basis": base, "factors": list(factors),
            "P_b50_rf": b50, "rf_max": rf_max, "P_slope": row.get("P_slope")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b50-table", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--shape", default="S3")
    args = ap.parse_args(argv)
    with args.b50_table.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["model"] == args.model and r["shape"] == args.shape]
    if len(rows) != 1:
        raise SystemExit(f"{args.b50_table}: {len(rows)} row(s) for {args.model}/{args.shape}, need 1")
    out = grid_for(rows[0])
    print(json.dumps({"model": args.model, "shape": args.shape, **out}), file=sys.stderr)
    print(",".join(f"{g:g}" for g in out["grid"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

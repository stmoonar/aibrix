#!/usr/bin/env python3
"""Fit the single-pod capacity surface C_m(i,o) from an r3_grid CSV (endgame 6.2 /
plan ch12). C_m(i,o) = max SLO-met sustained RPS at input i, output o. RPS is
derived per window as (generation_tokens_total / output_tokens) / window_seconds
(fixed output length via ignore_eos). Outputs capacity_<model>.json consumable by
design.py / lint.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tre_calibration.capacity import CapacitySample, fit_capacity_surface


def _f(value: str) -> float | None:
    if value is None or value == "" or value == "None":
        return None
    return float(value)


def sample_from_row(row: dict, *, ttft_slo_ms: float, tpot_slo_ms: float) -> CapacitySample | None:
    """Pure: one grid window row -> a CapacitySample (or None if unusable)."""
    output_tokens = int(row["output_tokens"])
    gen = _f(row.get("generation_tokens_total"))
    ws = _f(row.get("window_start_ms"))
    we = _f(row.get("window_end_ms"))
    if not output_tokens or gen is None or ws is None or we is None or we <= ws:
        return None
    window_s = (we - ws) / 1000.0
    rps = (gen / output_tokens) / window_s
    ttft = _f(row.get("p95_ttft"))
    tpot = _f(row.get("p95_tpot"))
    slo_met = ttft is not None and tpot is not None and ttft <= ttft_slo_ms and tpot <= tpot_slo_ms
    return CapacitySample(
        model=row["scenario_id"].split("_")[0] if "model" not in row else row["model"],
        input_tokens=int(row["input_tokens"]),
        output_tokens=output_tokens,
        rps=rps,
        slo_met=slo_met,
    )


def load_samples(csv_path: Path, model: str, *, ttft_slo_ms: float, tpot_slo_ms: float) -> list[CapacitySample]:
    samples: list[CapacitySample] = []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            s = sample_from_row(row, ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms)
            if s is not None:
                samples.append(CapacitySample(model, s.input_tokens, s.output_tokens, s.rps, s.slo_met))
    return samples


def surface_to_json(model: str, surface) -> dict:
    return {
        "model": model,
        "capacity": [
            {"input_tokens": k[1], "output_tokens": k[2], "rps": v}
            for k, v in sorted(surface.points.items())
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="r3_grid CSV")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True, help="capacity_<model>.json")
    ap.add_argument("--ttft-p95-ms", type=float, required=True)
    ap.add_argument("--tpot-p95-ms", type=float, required=True)
    args = ap.parse_args()
    samples = load_samples(Path(args.input), args.model, ttft_slo_ms=args.ttft_p95_ms, tpot_slo_ms=args.tpot_p95_ms)
    surface = fit_capacity_surface(samples)
    Path(args.output).write_text(json.dumps(surface_to_json(args.model, surface), indent=2))
    print(f"fitted {len(surface.points)} (i,o) capacity points from {len(samples)} windows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

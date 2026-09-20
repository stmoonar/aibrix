#!/usr/bin/env python3
"""Generate the open-loop calibration schedule files (one per model x shape x primitive).

Output layout (committed, so a campaign run is reproducible from the tree alone)::

    replayer/traces_v2/calibration/
        INDEX.json                      # every schedule + its provenance and metadata
        <model>/<shape>_<primitive>.json   # replayer trace.json schema

Each schedule file uses the ordinary model-keyed replayer trace schema
(``{model: [{start_time, end_time, rps, input_tokens, max_tokens}, ...]}``), so it loads
with ``tre_replayer.traces.loader.load_trace_segments`` and runs with
``r3_grid.py --schedule``. Overlapping segments superpose, because
``build_poisson_schedule`` samples each segment independently - that is how the bursts
primitive lays a spike on top of its base rate and how the mixture shape runs four
parallel token-shape streams.

The three primitives
--------------------
ramp    rho 0.8 -> 1.6 linearly over 360 s in 5 s segments, hold 1.6 for 60 s, then drop
        to rho 0.4 for 120 s. 540 s. Crosses the violation boundary *slowly*, so the
        fit sees the "degrading but not yet violating" regime the fixed-concurrency grid
        never produced, and the tail measures recovery / hysteresis.
steps   rho 0.5 (90 s) -> 0.8 (120 s) -> 0.95 (240 s), monotone. 450 s. Each level is
        long enough to reach steady state; the first control window after each step is
        transient and is discarded downstream (``discard_after_s`` in the index).
bursts  base rho 0.6 for 360 s with a 2 s spike every 90 s carrying B = 2 * C_s * T_s
        requests (T_s = 2 s), i.e. the spike segment runs at 2 * C_s on top of the base.
        4 bursts. This is the only primitive that reliably drives a short-lived waiting
        queue, which is what makes lambda_wait identifiable at all.

Capacity priors and the C_s model
---------------------------------
rho is relative to a per-shape single-pod capacity prior C_s (rps). The priors live in
``traces_v2/calibration/capacity/`` - a campaign-local copy, deliberately NOT the frozen
``traces_v2/capacity/`` set that experiment-3's traceset-v2 was generated from (that one
stays byte-unchanged for provenance). The dsqwen-14b prior there was re-measured on
2026-09-20 with the unique-per-request-prompt sender and prefix caching off; the frozen
2026-07-09 one is contaminated (its capacity RISES with prompt length: 14.9 -> 33.0 ->
32.0 rps for input 128 -> 512 -> 1024, the signature of an identical-prompt sender
against an engine with prefix caching on). The priors only cover a sparse (i, o) grid and none
of the campaign shapes sit on it, so nearest-neighbour would silently borrow a lighter
point's capacity (the v1 trace-set failure documented in traces_v2/README.md).

Instead we fit a two-parameter physical model per model::

    1 / C(i, o) = i / P + o / D

P is the pod's prefill throughput (prompt tokens/s) and D its decode throughput
(generated tokens/s): serving R rps of shape (i, o) spends R*i/P of each second in
prefill and R*o/D in decode, and saturates when that sums to 1. The fit is a
least-squares solve on the measured 1/C points, so it interpolates smoothly and
extrapolates to shapes nobody measured, and it is monotone-decreasing in both i and o by
construction - which is the property the contaminated 14b prior violated.

For the mixture shape M the capacity prior is the load-weighted harmonic combination::

    C_M = 1 / sum_k(w_k / C(i_k, o_k))

i.e. the total rps at which the blended stream saturates the pod.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

#: Training shapes. (input_tokens, max_output_tokens).
SHAPES: dict[str, tuple[int, int]] = {
    "S1": (256, 128),
    "S2": (768, 192),
    "S3": (2048, 96),
    "S4": (256, 448),
    "S5": (768, 384),
}

#: Held-out validation shape: a mixture of four token shapes running in parallel.
#: NEVER used for fitting - it exists to test that a theta fit on S1..S5 generalises.
MIXTURE_NAME = "M"
MIXTURE: tuple[tuple[float, int, int], ...] = (
    (0.40, 256, 128),
    (0.30, 1024, 256),
    (0.20, 128, 384),
    (0.10, 3072, 64),
)

MODELS = ("dsqwen-7b", "dsllama-8b", "dsqwen-14b")

# ---- primitive parameters (single source of truth; the index records them) ----
RAMP_RHO_START = 0.8
RAMP_RHO_END = 1.6
RAMP_DURATION_S = 360.0
RAMP_SEGMENT_S = 5.0
RAMP_HOLD_S = 60.0
RAMP_DRAIN_RHO = 0.4
RAMP_DRAIN_S = 120.0

STEPS: tuple[tuple[float, float], ...] = ((0.5, 90.0), (0.8, 120.0), (0.95, 240.0))

BURST_BASE_RHO = 0.6
BURST_DURATION_S = 360.0
BURST_PERIOD_S = 90.0
BURST_WIDTH_S = 2.0
BURST_MULTIPLIER = 2.0  # B = BURST_MULTIPLIER * C_s * BURST_WIDTH_S
BURST_FIRST_S = 60.0
BURST_COUNT = 4

#: c<N> in the cell id is NOT a concurrency in schedule mode - it is an offered-load code,
#: round(100 * the primitive's characteristic rho. The cell id must stay parseable by
#: ``r3_grid.GridCell.from_scenario_id`` or ``rewindow_from_raw`` silently skips the file.
LOAD_CODE = {"ramp": 160, "steps": 95, "bursts": 60}

PRIMITIVES = ("ramp", "steps", "bursts")


@dataclass(frozen=True)
class CapacityModel:
    """Per-model prefill/decode throughput fit; ``rps(i, o)`` is the capacity prior."""

    model: str
    prefill_tps: float
    decode_tps: float
    n_points: int
    rms_rel_error: float

    def rps(self, input_tokens: int, output_tokens: int) -> float:
        cost = input_tokens / self.prefill_tps + output_tokens / self.decode_tps
        if cost <= 0.0:
            raise ValueError("degenerate capacity model")
        return 1.0 / cost

    def mixture_rps(self, mixture: Sequence[tuple[float, int, int]]) -> float:
        total = sum(w / self.rps(i, o) for w, i, o in mixture)
        return 1.0 / total


def fit_capacity_model(model: str, points: Sequence[tuple[int, int, float]]) -> CapacityModel:
    """Least-squares fit of 1/C = i/P + o/D over measured (i, o, rps) points.

    Solves the 2x2 normal equations for a = 1/P, b = 1/D directly (no numpy dependency
    in the deploy scripts). Raises if the solution is not physically usable (a or b <= 0),
    which is exactly what a contaminated prior whose capacity *rises* with prompt length
    would produce - better a loud failure than a silently inverted capacity surface.
    """
    if len(points) < 2:
        raise ValueError(f"{model}: need >= 2 capacity points, got {len(points)}")
    sii = sio = soo = si_y = so_y = 0.0
    for i, o, rps in points:
        if rps <= 0.0:
            raise ValueError(f"{model}: non-positive rps at ({i},{o})")
        y = 1.0 / rps
        sii += i * i
        sio += i * o
        soo += o * o
        si_y += i * y
        so_y += o * y
    det = sii * soo - sio * sio
    if abs(det) < 1e-12:
        raise ValueError(f"{model}: capacity points are collinear; cannot separate P and D")
    a = (si_y * soo - so_y * sio) / det
    b = (so_y * sii - si_y * sio) / det
    if a <= 0.0 or b <= 0.0:
        raise ValueError(
            f"{model}: capacity fit is unphysical (1/P={a:.3e}, 1/D={b:.3e}). "
            "This means measured capacity does not decrease with token count - "
            "re-measure the prior before generating schedules."
        )
    fitted = CapacityModel(model, 1.0 / a, 1.0 / b, len(points), 0.0)
    errs = [(fitted.rps(i, o) - rps) / rps for i, o, rps in points]
    rms = (sum(e * e for e in errs) / len(errs)) ** 0.5
    return CapacityModel(model, 1.0 / a, 1.0 / b, len(points), rms)


def load_capacity_points(path: Path) -> tuple[str, list[tuple[int, int, float]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    points = [
        (int(p["input_tokens"]), int(p["output_tokens"]), float(p["rps"]))
        for p in data["capacity"]
    ]
    return str(data["model"]), points


# ------------------------------------------------------------------ primitives


def ramp_segments(capacity_rps: float) -> list[dict]:
    """rho ramps linearly in RAMP_SEGMENT_S steps, holds at the peak, then drains.

    Each 5 s segment uses the rho at its *midpoint*, so the piecewise-constant schedule
    integrates to the same request count as the continuous ramp it approximates.
    """
    segments: list[dict] = []
    n = int(round(RAMP_DURATION_S / RAMP_SEGMENT_S))
    for k in range(n):
        frac = (k + 0.5) / n
        rho = RAMP_RHO_START + (RAMP_RHO_END - RAMP_RHO_START) * frac
        segments.append(_segment(k * RAMP_SEGMENT_S, (k + 1) * RAMP_SEGMENT_S, rho * capacity_rps))
    hold_end = RAMP_DURATION_S + RAMP_HOLD_S
    segments.append(_segment(RAMP_DURATION_S, hold_end, RAMP_RHO_END * capacity_rps))
    segments.append(_segment(hold_end, hold_end + RAMP_DRAIN_S, RAMP_DRAIN_RHO * capacity_rps))
    return segments


def step_segments(capacity_rps: float) -> list[dict]:
    segments: list[dict] = []
    t = 0.0
    for rho, duration in STEPS:
        segments.append(_segment(t, t + duration, rho * capacity_rps))
        t += duration
    return segments


def burst_segments(capacity_rps: float) -> list[dict]:
    """Base rate for the whole cell, with BURST_COUNT spikes superposed on top.

    A spike carries B = BURST_MULTIPLIER * C_s * BURST_WIDTH_S requests inside
    BURST_WIDTH_S seconds, so its segment rate is BURST_MULTIPLIER * C_s; combined with
    the base that is (BURST_MULTIPLIER + BURST_BASE_RHO) x capacity for those 2 s. The
    engine cannot absorb that in its running set, so the surplus lands in the waiting
    queue - which is the observable the whole primitive exists to produce.
    """
    segments = [_segment(0.0, BURST_DURATION_S, BURST_BASE_RHO * capacity_rps)]
    for k in range(BURST_COUNT):
        start = BURST_FIRST_S + k * BURST_PERIOD_S
        end = start + BURST_WIDTH_S
        if end > BURST_DURATION_S:
            raise ValueError("burst falls outside the cell duration")
        segments.append(_segment(start, end, BURST_MULTIPLIER * capacity_rps))
    return segments


def _segment(start_s: float, end_s: float, rps: float) -> dict:
    return {"start_time": _round(start_s), "end_time": _round(end_s), "rps": round(rps, 4)}


def _round(value: float) -> float:
    return int(value) if float(value).is_integer() else round(value, 3)


PRIMITIVE_BUILDERS = {
    "ramp": ramp_segments,
    "steps": step_segments,
    "bursts": burst_segments,
}


def build_schedule(
    model: str,
    shape_name: str,
    primitive: str,
    capacity: CapacityModel,
) -> tuple[dict, dict]:
    """(trace.json body, index metadata) for one (model, shape, primitive)."""
    builder = PRIMITIVE_BUILDERS[primitive]
    if shape_name == MIXTURE_NAME:
        c_s = capacity.mixture_rps(MIXTURE)
        components = MIXTURE
    else:
        i, o = SHAPES[shape_name]
        c_s = capacity.rps(i, o)
        components = ((1.0, i, o),)

    segments: list[dict] = []
    for weight, i, o in components:
        for seg in builder(c_s * weight):
            entry = dict(seg)
            entry["input_tokens"] = i
            entry["max_tokens"] = o
            segments.append(entry)
    segments.sort(key=lambda s: (s["start_time"], s["input_tokens"], s["max_tokens"]))

    # Cell id: a mixture has no single (i, o), so it records 0/0. r3_capacity's
    # sample_from_row then skips it (output_tokens 0 -> unusable), which is correct: M is
    # held-out validation, never a capacity or theta training point.
    if shape_name == MIXTURE_NAME:
        cell_in, cell_out = 0, 0
    else:
        cell_in, cell_out = SHAPES[shape_name]
    cell_id = f"i{cell_in}_o{cell_out}_c{LOAD_CODE[primitive]}"

    duration = max(s["end_time"] for s in segments)
    planned = sum((s["end_time"] - s["start_time"]) * s["rps"] for s in segments)
    meta = {
        "model": model,
        "shape": shape_name,
        "primitive": primitive,
        "cell_id": cell_id,
        "held_out": shape_name == MIXTURE_NAME,
        "capacity_prior_rps": round(c_s, 4),
        "duration_s": duration,
        "planned_requests": int(round(planned)),
        "components": [
            {"weight": w, "input_tokens": i, "max_tokens": o} for w, i, o in components
        ],
        "peak_offered_rps": round(max(_offered_rps_at(segments)), 4),
    }
    if primitive == "steps":
        t = 0.0
        boundaries = []
        for _rho, duration_s in STEPS:
            boundaries.append(_round(t))
            t += duration_s
        meta["discard_after_s"] = boundaries
    return {model: segments}, meta


def _offered_rps_at(segments: Sequence[dict]) -> list[float]:
    """Total offered rps at each segment boundary (superposed overlapping segments)."""
    edges = sorted({s["start_time"] for s in segments} | {s["end_time"] for s in segments})
    totals = []
    for k in range(len(edges) - 1):
        mid = (edges[k] + edges[k + 1]) / 2.0
        totals.append(
            sum(s["rps"] for s in segments if s["start_time"] <= mid < s["end_time"])
        )
    return totals or [0.0]


def generate(
    capacity_dir: Path,
    output_dir: Path,
    models: Sequence[str],
    *,
    capacity_overrides: dict | None = None,
) -> dict:
    index: dict = {"schedules": [], "capacity_models": {}, "shapes": {}, "primitives": {}}
    index["shapes"] = {
        **{name: {"input_tokens": i, "max_tokens": o} for name, (i, o) in SHAPES.items()},
        MIXTURE_NAME: {
            "held_out": True,
            "components": [
                {"weight": w, "input_tokens": i, "max_tokens": o} for w, i, o in MIXTURE
            ],
        },
    }
    index["primitives"] = {
        "ramp": {
            "rho_start": RAMP_RHO_START, "rho_end": RAMP_RHO_END,
            "ramp_s": RAMP_DURATION_S, "segment_s": RAMP_SEGMENT_S,
            "hold_s": RAMP_HOLD_S, "drain_rho": RAMP_DRAIN_RHO, "drain_s": RAMP_DRAIN_S,
        },
        "steps": {"levels": [{"rho": r, "duration_s": d} for r, d in STEPS]},
        "bursts": {
            "base_rho": BURST_BASE_RHO, "duration_s": BURST_DURATION_S,
            "period_s": BURST_PERIOD_S, "width_s": BURST_WIDTH_S,
            "multiplier": BURST_MULTIPLIER, "count": BURST_COUNT, "first_s": BURST_FIRST_S,
        },
    }

    for model in models:
        override = (capacity_overrides or {}).get(model)
        if override is not None:
            _name, points = override
        else:
            _name, points = load_capacity_points(capacity_dir / f"capacity_{model}.json")
        capacity = fit_capacity_model(model, points)
        index["capacity_models"][model] = {
            "prefill_tokens_per_s": round(capacity.prefill_tps, 2),
            "decode_tokens_per_s": round(capacity.decode_tps, 2),
            "fitted_from_points": capacity.n_points,
            "rms_relative_error": round(capacity.rms_rel_error, 4),
        }
        model_dir = output_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        for shape_name in [*SHAPES, MIXTURE_NAME]:
            for primitive in PRIMITIVES:
                body, meta = build_schedule(model, shape_name, primitive, capacity)
                path = model_dir / f"{shape_name}_{primitive}.json"
                path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
                meta["path"] = str(path.relative_to(output_dir))
                index["schedules"].append(meta)
    return index


def main() -> int:
    here = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capacity-dir", type=Path,
                    default=here / "replayer/traces_v2/calibration/capacity")
    ap.add_argument("--output-dir", type=Path, default=here / "replayer/traces_v2/calibration")
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = generate(args.capacity_dir, args.output_dir, [m for m in args.models.split(",") if m])
    (args.output_dir / "INDEX.json").write_text(
        json.dumps(index, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(index['schedules'])} schedules to {args.output_dir}")
    for model, cm in index["capacity_models"].items():
        print(
            f"  {model}: prefill {cm['prefill_tokens_per_s']} tok/s, "
            f"decode {cm['decode_tokens_per_s']} tok/s, "
            f"rms rel err {cm['rms_relative_error']:.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

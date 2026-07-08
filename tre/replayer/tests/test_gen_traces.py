from __future__ import annotations

import json
from pathlib import Path

from tre_replayer import gen_traces
from tre_replayer.gen_traces import (
    HEADROOM_TARGETS,
    MODEL_14B,
    MODEL_7B,
    MODEL_8B,
    MODEL_SLOT_WIDTH,
    TOTAL_SLOTS,
    generate_trace_set,
    load_capacity_surface,
)
from tre_replayer.traces.loader import discover_trace_set, load_trace_segments

# Synthetic single-pod capacities. For A3 the input=512 ladder is strictly decreasing in
# rps as output grows (decode gets heavier), which is what drives constant-rps rising load.
_CAPACITY = {
    MODEL_7B: [
        (256, 128, 10.0),
        (512, 256, 6.0),
        (512, 128, 8.0),
        (512, 384, 4.5),
        (512, 512, 3.5),
    ],
    MODEL_8B: [(256, 128, 9.0), (512, 256, 5.0)],
    MODEL_14B: [(256, 128, 4.0), (512, 256, 2.5)],
}


def _write_capacity(tmp_path: Path) -> list[Path]:
    paths = []
    for model, points in _CAPACITY.items():
        payload = {
            "model": model,
            "capacity": [{"input_tokens": i, "output_tokens": o, "rps": r} for i, o, r in points],
        }
        p = tmp_path / f"capacity_{model}.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(p)
    return paths


def _cap(model: str, i: int, o: int) -> float:
    for ii, oo, rr in _CAPACITY[model]:
        if ii == i and oo == o:
            return rr
    raise KeyError((model, i, o))


def _recompute_peak_occupancy(trace: dict) -> float:
    """Independent re-derivation of lint occupancy from the written trace.json."""
    segs = []
    for model, raw in trace.items():
        for s in raw:
            segs.append((model, float(s["start_time"]), float(s["end_time"]),
                         float(s["rps"]), int(s["input_tokens"]), int(s["max_tokens"])))
    bounds = sorted({t for _, a, b, *_ in segs for t in (a, b)})
    peak = 0.0
    for lo, hi in zip(bounds, bounds[1:]):
        if hi <= lo:
            continue
        occ = 0.0
        for model, a, b, rps, i, o in segs:
            if a <= lo and b >= hi:
                rho = rps / _cap(model, i, o)
                occ += rho * MODEL_SLOT_WIDTH[model]
        peak = max(peak, occ)
    return peak


def test_generate_trace_set_writes_seven_schema_valid_traces(tmp_path: Path) -> None:
    surface = load_capacity_surface(_write_capacity(tmp_path))
    out = tmp_path / "traces"
    index = generate_trace_set(surface, out)

    assert len(index["workloads"]) == 7
    axes = {d["axis"] for d in index["designs"]}
    assert axes == {"A1", "A2", "A3", "A4", "A5", "A6"}

    # Every trace round-trips through the production loader (schema parity with correctness set).
    trace_set = discover_trace_set(out)
    assert [c.name for c in trace_set.cases] == index["workloads"]
    for case in trace_set.cases:
        assert case.segments  # non-empty, parsed as RpsSegment
        # explicit coverage from 0 with no negative-length segments
        assert all(seg.end_s > seg.start_s for seg in case.segments)


def test_rps_equals_rho_times_capacity(tmp_path: Path) -> None:
    surface = load_capacity_surface(_write_capacity(tmp_path))
    out = tmp_path / "traces"
    generate_trace_set(surface, out)

    trace = json.loads((out / "t1_a1_demand_shift" / "trace.json").read_text())
    sat = next(s for s in trace[MODEL_7B] if s["input_tokens"] == 512)
    # A1 saturation phase: rho=6.0 at (512,256) -> rps = 6.0 * C = 36.0
    assert sat["rps"] == round(6.0 * _cap(MODEL_7B, 512, 256), 4)


def test_headroom_tiers_hit_exactly_for_capacity_independent_axes(tmp_path: Path) -> None:
    surface = load_capacity_surface(_write_capacity(tmp_path))
    out = tmp_path / "traces"
    index = generate_trace_set(surface, out)
    by_name = {d["name"]: d for d in index["designs"]}

    for name, d in by_name.items():
        if d["headroom_is_capacity_dependent"]:
            continue
        trace = json.loads((out / name / "trace.json").read_text())
        peak = _recompute_peak_occupancy(trace)
        target = HEADROOM_TARGETS[d["headroom_tier"]]
        if d["headroom_tier"] == "loose":
            # A6 control sits inside the loose band but is not pinned to the exact target.
            assert 0.55 <= peak / TOTAL_SLOTS <= 0.65
        else:
            assert abs(peak - target * TOTAL_SLOTS) < 1e-6
        # INDEX-reported occupancy matches the independent recomputation.
        assert abs(peak - d["peak_occupancy_slots"]) < 1e-6


def test_a6_control_stays_below_non_triviality_threshold(tmp_path: Path) -> None:
    surface = load_capacity_surface(_write_capacity(tmp_path))
    out = tmp_path / "traces"
    generate_trace_set(surface, out)
    trace = json.loads((out / "t6_a6_control" / "trace.json").read_text())
    for model, raw in trace.items():
        for s in raw:
            rho = s["rps"] / _cap(model, s["input_tokens"], s["max_tokens"])
            assert rho <= 1.2  # control: no sustained saturation


def test_a3_holds_rps_constant_while_output_grows(tmp_path: Path) -> None:
    surface = load_capacity_surface(_write_capacity(tmp_path))
    out = tmp_path / "traces"
    index = generate_trace_set(surface, out)

    a3 = next(d for d in index["designs"] if d["axis"] == "A3")
    assert a3["headroom_is_capacity_dependent"] is True

    trace = json.loads((out / "t3_a3_io_drift" / "trace.json").read_text())
    drift = [s for s in trace[MODEL_7B] if s["input_tokens"] == 512]
    drift.sort(key=lambda s: s["start_time"])

    outputs = [s["max_tokens"] for s in drift]
    assert outputs == sorted(outputs) and len(set(outputs)) == len(outputs)  # strictly increasing
    rpss = [s["rps"] for s in drift]
    assert max(rpss) - min(rpss) < 1e-6  # RPS held constant across the drift
    # load actually rises: rho at the heaviest output exceeds one pod
    heavy = drift[-1]
    assert heavy["rps"] / _cap(MODEL_7B, 512, heavy["max_tokens"]) > 1.0

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tre_replayer.design import DemandPhase
from tre_replayer.gen_traces import (
    HEADROOM_TARGETS,
    MODEL_14B,
    MODEL_7B,
    MODEL_8B,
    MODEL_SLOT_WIDTH,
    TOTAL_SLOTS,
    TraceDesign,
    assert_feasible,
    generate_trace_set,
    load_capacity_surface,
    peak_integer_occupancy,
)
from tre_replayer.traces.loader import discover_trace_set

# Synthetic single-pod capacities on the MEASURED shapes the v2 designs emit:
# baseline (128,128), saturation (512,512), and the A3 7b output ladder (512,{128,256,384,512}).
# The 7b (512,o) row is strictly decreasing in rps as output grows (decode gets heavier),
# which is what drives constant-rps rising load in A3.
_CAPACITY = {
    MODEL_7B: [
        (128, 128, 16.0),
        (512, 128, 12.0),
        (512, 256, 9.6),
        (512, 384, 8.0),
        (512, 512, 6.4),
    ],
    MODEL_8B: [(128, 128, 13.0), (512, 512, 3.2)],
    MODEL_14B: [(128, 128, 14.0), (512, 512, 12.8)],
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


def _recompute_peak_integer_occupancy(trace: dict) -> float:
    """Independent re-derivation of lint's INTEGER GPU occupancy from the written trace.json."""
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
                replicas = max(1, math.ceil(rho - 1e-9))
                occ += replicas * MODEL_SLOT_WIDTH[model]
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


def test_every_trace_is_integer_gpu_feasible(tmp_path: Path) -> None:
    surface = load_capacity_surface(_write_capacity(tmp_path))
    out = tmp_path / "traces"
    index = generate_trace_set(surface, out)

    for d in index["designs"]:
        # INDEX records the integer GPU occupancy and a per-phase feasibility proof <= 8.
        assert d["integer_gpu_occupancy"] <= TOTAL_SLOTS
        proof = d["feasibility"]
        assert proof["feasible"] is True
        assert proof["peak_gpu"] == d["integer_gpu_occupancy"]
        for phase in proof["phases"]:
            recomputed = sum(pm["replicas"] * pm["slot_width"] for pm in phase["per_model"].values())
            assert abs(recomputed - phase["gpu_total"]) < 1e-9
            assert phase["gpu_total"] <= TOTAL_SLOTS

        # The written trace.json independently re-derives the same integer occupancy.
        trace = json.loads((out / d["name"] / "trace.json").read_text())
        assert abs(_recompute_peak_integer_occupancy(trace) - d["integer_gpu_occupancy"]) < 1e-9


def test_assert_feasible_rejects_fractionally_ok_but_integer_infeasible_design() -> None:
    """RED->GREEN guard: v1's A1 bug -- fractional 6.8 <= 8 but integer 9 > 8 must be rejected."""
    bad = TraceDesign(
        "bad", "A1", "tight", "reproduces the v1 A1 infeasibility",
        phases=[DemandPhase("hot", 0.0, 300.0, {MODEL_7B: 6.0, MODEL_8B: 0.4, MODEL_14B: 0.4})],
        default_shapes={m: (512, 512) for m in (MODEL_7B, MODEL_8B, MODEL_14B)},
    )
    # fractional occupancy is within budget ...
    frac = 6.0 * 1 + 0.4 * 1 + 0.4 * 2
    assert frac <= TOTAL_SLOTS
    # ... but the integer requirement (6+1+2 = 9) is not, so the guard rejects it.
    assert peak_integer_occupancy(bad.phases) == 9.0
    with pytest.raises(ValueError, match="infeasible"):
        assert_feasible(bad)


def test_assert_feasible_accepts_full_but_feasible_tight_design() -> None:
    good = TraceDesign(
        "good", "A1", "tight", "hot model at exactly the integer limit",
        phases=[DemandPhase("hot", 0.0, 300.0, {MODEL_7B: 4.8, MODEL_8B: 0.4, MODEL_14B: 0.4})],
        default_shapes={m: (512, 512) for m in (MODEL_7B, MODEL_8B, MODEL_14B)},
    )
    assert peak_integer_occupancy(good.phases) == TOTAL_SLOTS  # 5 + 1 + 2 = 8
    assert_feasible(good)  # does not raise


def test_rps_equals_rho_times_capacity(tmp_path: Path) -> None:
    surface = load_capacity_surface(_write_capacity(tmp_path))
    out = tmp_path / "traces"
    generate_trace_set(surface, out)

    trace = json.loads((out / "t1_a1_demand_shift" / "trace.json").read_text())
    sat = next(s for s in trace[MODEL_7B] if s["input_tokens"] == 512)
    # A1 saturation phase: rho=4.8 at (512,512) -> rps = 4.8 * C
    assert sat["max_tokens"] == 512
    assert sat["rps"] == round(4.8 * _cap(MODEL_7B, 512, 512), 4)


def test_headroom_tiers_hit_exactly_for_capacity_independent_axes(tmp_path: Path) -> None:
    surface = load_capacity_surface(_write_capacity(tmp_path))
    out = tmp_path / "traces"
    index = generate_trace_set(surface, out)
    by_name = {d["name"]: d for d in index["designs"]}

    for name, d in by_name.items():
        if d["headroom_is_capacity_dependent"]:
            continue
        trace = json.loads((out / name / "trace.json").read_text())
        peak = _recompute_peak_integer_occupancy(trace)
        target = HEADROOM_TARGETS[d["headroom_tier"]]
        # Integer occupancy lands exactly on the tier target (loose 4/8, medium 6/8, tight 8/8).
        assert abs(peak - target * TOTAL_SLOTS) < 1e-6
        # INDEX-reported occupancy matches the independent recomputation.
        assert abs(peak - d["integer_gpu_occupancy"]) < 1e-6


def test_a6_control_stays_below_non_triviality_threshold(tmp_path: Path) -> None:
    surface = load_capacity_surface(_write_capacity(tmp_path))
    out = tmp_path / "traces"
    index = generate_trace_set(surface, out)
    trace = json.loads((out / "t6_a6_control" / "trace.json").read_text())
    for model, raw in trace.items():
        for s in raw:
            rho = s["rps"] / _cap(model, s["input_tokens"], s["max_tokens"])
            assert rho <= 1.2  # control: no sustained saturation
            assert rho <= 1.0  # v2: never needs a second replica

    # A6 integer occupancy stays flat at the loose resting floor (4/8).
    a6 = next(d for d in index["designs"] if d["name"] == "t6_a6_control")
    assert a6["integer_gpu_occupancy"] == 4.0
    assert a6["headroom_tier"] == "loose"


def test_a3_holds_rps_constant_while_output_grows(tmp_path: Path) -> None:
    surface = load_capacity_surface(_write_capacity(tmp_path))
    out = tmp_path / "traces"
    index = generate_trace_set(surface, out)

    a3 = next(d for d in index["designs"] if d["axis"] == "A3")
    assert a3["headroom_is_capacity_dependent"] is True
    assert a3["integer_gpu_occupancy"] <= TOTAL_SLOTS

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

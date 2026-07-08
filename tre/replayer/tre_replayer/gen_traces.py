"""Generate the formal experiment-3 (TRE vs APA) trace set from R3 capacity surfaces.

This is the R7 deliverable deferred by run_trace.py's docstring: turn the fitted
single-pod capacity surfaces C_m(i,o) (r3_capacity output: capacity_<model>.json) into a
qualified, mechanism-covering trace set. It is the capacity-calibrated cousin of the
hand-written `traces_correctness/` set.

Design source of truth: REFACTOR_PLAN.md section 12.4 (the six-axis mechanism-coverage
matrix A1..A6) and 12.5 (phase-structure rules; design.py enforces them). The existing
correctness traces already exercise A1 (c1), A2 (c2/c2b), A4 (c4) and A5 (c3/c3b); this
generator parameterises those designs against capacity and adds the two missing axes:

  * A3 (i/o mix drift) -- the metric-superiority scenario. RPS is held CONSTANT while the
    output length grows (decode gets heavier), so queue-length / KVCache signals move late
    while TSS's weighted throughput reflects the true load immediately. Implemented with
    per-phase token-shape overrides (design.DemandPhase.token_shapes) and a per-phase rho
    back-solved from capacity so rho*C stays constant across the drift phases.
  * A6 (control scenario) -- a gentle in-phase ramp at loose headroom that every system
    should handle. It intentionally stays below the C2 non-triviality threshold (no model
    sustains rho > 1.2); its purpose is fairness evidence, not to stress the controller.

traceset-v2 -- integer-feasible occupancy
-----------------------------------------
traceset-v1 sized the tiers on *fractional* occupancy Sum_m rho_m * slot_width_m / slots.
That is a physics bug: a model at rho=6.0 needs ceil(6)=6 integer replicas, every model
with traffic needs >=1 awake replica (serving floor), and dsqwen-14b is tp2 (2 GPU/replica).
v1's A1 "tight" phase (rho_7b=6.0 + the 0.4 floors) reads as 7.2/8=0.90 fractional but
actually demands 6+1+2 = 9 GPU > 8 -- physically infeasible, a root cause of the t1 TRE-arm
503 storm. traceset-v2 re-sizes every design so the *integer* GPU requirement
Sum_m ceil(rho_m) * slot_width_m (>=1 replica per model that has traffic) is <= 8 at all
times, and reports occupancy on that integer basis:

  loose  = 4/8 (resting floor: one replica per model; 14b tp2 => 1+1+2)
  medium = 6/8 (25% GPU headroom)
  tight  = 8/8 (hugging the physical limit, still integer-feasible)

assert_feasible() enforces the <= 8 invariant per phase at generation time; lint's C1 now
checks the same integer occupancy (<= 1.0). For every axis except A3 the integer occupancy
is capacity-independent (lint recomputes rho = rps/C = the design rho, so C cancels), so the
tier is hit exactly by construction and is unit-tested. A3's rho is capacity-derived, so its
achieved integer occupancy depends on the real surface and is reported, not pinned.

Token shapes are pinned to *measured* capacity grid points so the replayer's real RPS is not
built on an over-optimistic nearest-neighbour extrapolation (another t1 failure mode: 8b/14b
at (512,256) borrowed the lighter (512,128) capacity). Baseline (128,128) and saturation
(512,512) are measured for all three models; the A3 7b output ladder (512,{128,256,384,512})
is measured for dsqwen-7b.

Cluster shape (slot widths, total slots) comes from deploy/registry.yaml: dsqwen-7b and
dsllama-8b are tp_size=1 (1 slot), dsqwen-14b is tp_size=2 (2 slots); the cluster is two
4xA100 nodes = 8 GPU slots. Baseline min/max replicas are 1..8 for the 1-slot models and
0..4 for dsqwen-14b.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from tre_calibration.capacity import CapacitySurface
from tre_replayer.design import DemandPhase, design_trace_segments
from tre_replayer.engine.schedule import RpsSegment
from tre_replayer.lint import replicas_for_rho

MODEL_7B = "dsqwen-7b"
MODEL_8B = "dsllama-8b"
MODEL_14B = "dsqwen-14b"

# Slot widths and cluster size from deploy/registry.yaml (tp_size; 2 nodes x 4 GPUs).
MODEL_SLOT_WIDTH: dict[str, float] = {MODEL_7B: 1.0, MODEL_8B: 1.0, MODEL_14B: 2.0}
TOTAL_SLOTS = 8.0

# Token shapes pinned to MEASURED capacity grid points (traceset-v2). Baseline (128,128) and
# saturation (512,512) are measured for dsqwen-7b, dsllama-8b and dsqwen-14b, so the replayer's
# real RPS never rests on a nearest-neighbour capacity extrapolation.
BASELINE_SHAPE = (128, 128)
SATURATION_SHAPE = (512, 512)
# A3 output-length drift ladder at fixed input; decode weight rises left to right. All four
# shapes (512,{128,256,384,512}) are measured grid points for dsqwen-7b (the only drift model).
A3_INPUT = 512
A3_OUTPUT_LADDER = (128, 256, 384, 512)
# A3 target constant RPS = A3_RPS_MULT x single-pod capacity at the heaviest output shape.
# The heaviest drift phase needs ceil(A3_RPS_MULT) 7b replicas + the constant 8b/14b floor
# (1 + 2 = 3 slots), so A3_RPS_MULT is calibrated to 2.8 (ceil = 3) to land the peak integer
# occupancy on the medium tier (3 + 1 + 2 = 6 / 8 slots = 0.75) and pass lint C3. RPS is held
# constant across the drift while decode grows, so every phase sits above 1 pod (rho > 1) --
# the rate-signal-lag scenario A3 tests -- now integer-feasible.
A3_RPS_MULT = 2.8

# Tiers are peak INTEGER GPU occupancy / total_slots; see module docstring and lint._HEADROOM_TARGETS.
HEADROOM_TARGETS = {"loose": 0.50, "medium": 0.75, "tight": 1.00}

DEFAULT_MODELS = (MODEL_7B, MODEL_8B, MODEL_14B)


@dataclass(frozen=True)
class TraceDesign:
    name: str
    axis: str
    headroom: str
    mechanism: str
    phases: list[DemandPhase]
    default_shapes: Mapping[str, tuple[int, int]]
    # A3's rho is capacity-derived, so its achieved headroom is reported, not pinned.
    headroom_is_capacity_dependent: bool = False


def _base(rho_7b: float = 0.4, rho_8b: float = 0.4, rho_14b: float = 0.4) -> dict[str, float]:
    return {MODEL_7B: rho_7b, MODEL_8B: rho_8b, MODEL_14B: rho_14b}


def _sat_shapes(*models: str) -> dict[str, tuple[int, int]]:
    return {model: SATURATION_SHAPE for model in models}


def _design_a1_demand_shift() -> TraceDesign:
    """A1: staged saturation with a hard step -- only hot-switch keeps up on the handoff."""
    phases = [
        DemandPhase("warmup", 0.0, 113.0, _base()),
        DemandPhase("qwen7b_saturate", 113.0, 400.0, _base(rho_7b=4.8), token_shapes=_sat_shapes(MODEL_7B)),
        DemandPhase("handoff_llama8b", 400.0, 687.0, _base(rho_8b=4.8), token_shapes=_sat_shapes(MODEL_8B)),
        DemandPhase("cooldown", 687.0, 800.0, _base()),
    ]
    return TraceDesign(
        "t1_a1_demand_shift", "A1", "tight",
        "Hard demand handoff 7b->8b; peak integer occupancy 8/8 (hot model 5 replicas + 8b/14b "
        "floors 1+2). Hot-switch should follow the step within the fast loop while cold-start "
        "scaling lags; neither model drops its serving floor.",
        phases, {m: BASELINE_SHAPE for m in DEFAULT_MODELS},
    )


def _design_a2_anticorrelated(name: str, headroom: str, hi: float, lo: float) -> TraceDesign:
    """A2: 7b/8b anti-correlated with Sum(rho) held constant -> tests capacity rebalance."""
    b = _base(rho_7b=lo, rho_8b=lo)
    phases = [
        DemandPhase("warmup", 0.0, 120.0, _base()),
        DemandPhase("hi_7b_1", 120.0, 353.0, {MODEL_7B: hi, MODEL_8B: lo, MODEL_14B: 0.4}, token_shapes=_sat_shapes(MODEL_7B)),
        DemandPhase("hi_8b_1", 353.0, 586.0, {MODEL_7B: lo, MODEL_8B: hi, MODEL_14B: 0.4}, token_shapes=_sat_shapes(MODEL_8B)),
        DemandPhase("hi_7b_2", 586.0, 819.0, {MODEL_7B: hi, MODEL_8B: lo, MODEL_14B: 0.4}, token_shapes=_sat_shapes(MODEL_7B)),
        DemandPhase("hi_8b_2", 819.0, 1052.0, {MODEL_7B: lo, MODEL_8B: hi, MODEL_14B: 0.4}, token_shapes=_sat_shapes(MODEL_8B)),
        DemandPhase("cooldown", 1052.0, 1120.0, b),
    ]
    return TraceDesign(
        name, "A2", headroom,
        f"Anti-correlated 7b/8b, Sum(rho)~{hi + lo:g} constant across 233s phases; hot model "
        f"{replicas_for_rho(hi)} replicas + donor 1 + 14b floor 2 = {replicas_for_rho(hi) + 1 + 2}/8. "
        "Donor/receiver rebalance (slow loop) should migrate capacity each phase without both "
        "models staying high; 14b never participates.",
        phases, {m: BASELINE_SHAPE for m in DEFAULT_MODELS},
    )


def _design_a3_io_drift(capacity: CapacitySurface) -> TraceDesign:
    """A3: RPS constant, output length 128->512; load rises but rate-based signals lag."""
    target_rps = A3_RPS_MULT * capacity.capacity_at(MODEL_7B, input_tokens=A3_INPUT, output_tokens=A3_OUTPUT_LADDER[-1]).rps
    phases: list[DemandPhase] = [DemandPhase("warmup", 0.0, 113.0, _base())]
    start = 113.0
    for i, out in enumerate(A3_OUTPUT_LADDER):
        end = start + 227.0
        c = capacity.capacity_at(MODEL_7B, input_tokens=A3_INPUT, output_tokens=out).rps
        rho = target_rps / c if c > 0.0 else 0.0
        phases.append(
            DemandPhase(
                f"drift_o{out}", start, end,
                {MODEL_7B: rho, MODEL_8B: 0.4, MODEL_14B: 0.4},
                token_shapes={MODEL_7B: (A3_INPUT, out)},
            )
        )
        start = end
    phases.append(DemandPhase("cooldown", start, start + 67.0, _base()))
    return TraceDesign(
        "t3_a3_io_drift", "A3", "medium",
        "RPS held constant while dsqwen-7b output grows 128->512 (decode gets heavier). Queue "
        "length / KVCache signals lag, so APA reacts late; TSS weighted throughput reflects the "
        "rising load immediately, so TRE should scale earlier.",
        phases, {m: BASELINE_SHAPE for m in DEFAULT_MODELS},
        headroom_is_capacity_dependent=True,
    )


def _design_a4_spike_vs_burst() -> TraceDesign:
    """A4: 17s narrow spike (below EMA tau) vs 127s wide burst -- same amplitude, differ in width."""
    hi = 2.8
    phases = [
        DemandPhase("warmup", 0.0, 150.0, _base()),
        DemandPhase("narrow_spike_7b", 150.0, 167.0, _base(rho_7b=hi), token_shapes=_sat_shapes(MODEL_7B), allow_short=True),
        DemandPhase("settle_1", 167.0, 347.0, _base()),
        DemandPhase("wide_burst_7b", 347.0, 474.0, _base(rho_7b=hi), token_shapes=_sat_shapes(MODEL_7B)),
        DemandPhase("settle_2", 474.0, 620.0, _base()),
        DemandPhase("narrow_spike_8b", 620.0, 637.0, _base(rho_8b=hi), token_shapes=_sat_shapes(MODEL_8B), allow_short=True),
        DemandPhase("cooldown", 637.0, 740.0, _base()),
    ]
    return TraceDesign(
        "t4_a4_spike_vs_burst", "A4", "medium",
        "Narrow 17s spikes (shorter than the EMA time constant) must NOT trigger a full scale-up "
        "or churn; the 127s wide burst (> 3 slow loops) must. Also exercises SafeScale rollback.",
        phases, {m: BASELINE_SHAPE for m in DEFAULT_MODELS},
    )


def _design_a5_tp_pressure() -> TraceDesign:
    """A5: 14b (tp2) ramp while both 1-slot models hold their single-GPU slots awake."""
    hold = {MODEL_7B: 0.5, MODEL_8B: 0.5}
    phases = [
        DemandPhase("warmup", 0.0, 120.0, _base()),
        DemandPhase("ramp1_14b", 120.0, 353.0, {**hold, MODEL_14B: 1.0}, token_shapes=_sat_shapes(MODEL_14B)),
        DemandPhase("ramp2_14b", 353.0, 586.0, {**hold, MODEL_14B: 2.0}, token_shapes=_sat_shapes(MODEL_14B)),
        DemandPhase("ramp3_14b", 586.0, 819.0, {**hold, MODEL_14B: 2.8}, token_shapes=_sat_shapes(MODEL_14B)),
        DemandPhase("cooldown", 819.0, 900.0, _base()),
    ]
    return TraceDesign(
        "t5_a5_tp_pressure", "A5", "tight",
        "dsqwen-14b (tp_size=2) ramps 1->2->3 replicas to peak integer occupancy 8/8 (14b 3 "
        "replicas x2 GPU + 7b/8b holding one GPU slot each) while 7b/8b stay awake, forcing the "
        "allocator to defragment two-GPU placement without breaking the 1-slot models' serving floor.",
        phases, {m: BASELINE_SHAPE for m in DEFAULT_MODELS},
    )


def _design_a6_control() -> TraceDesign:
    """A6: gentle in-phase ramp at loose headroom; every system should cope. Control/fairness."""
    levels = (0.35, 0.65, 0.95, 0.65, 0.35)
    phases: list[DemandPhase] = []
    start = 0.0
    for i, lvl in enumerate(levels):
        end = start + 173.0
        phases.append(DemandPhase(f"sine_{i}", start, end, {MODEL_7B: lvl, MODEL_8B: lvl, MODEL_14B: lvl}))
        start = end
    return TraceDesign(
        "t6_a6_control", "A6", "loose",
        "Gentle in-phase ramp across all models; every model stays at rho <= 0.95 so no model "
        "ever needs a second replica -- integer occupancy holds at the 4/8 resting floor (loose). "
        "It stays below the C2 non-triviality threshold (rho <= 1.2). Fairness control: TRE and "
        "APA should tie here, proving the other traces' gaps are real.",
        phases, {m: BASELINE_SHAPE for m in DEFAULT_MODELS},
    )


def build_designs(capacity: CapacitySurface) -> list[TraceDesign]:
    """The seven formal traces: six mechanism axes A1..A6 plus a tight A2 variant."""
    return [
        _design_a1_demand_shift(),
        _design_a2_anticorrelated("t2_a2_anticorrelated", "medium", hi=2.8, lo=0.8),
        _design_a3_io_drift(capacity),
        _design_a4_spike_vs_burst(),
        _design_a5_tp_pressure(),
        _design_a6_control(),
        _design_a2_anticorrelated("t7_a2b_anticorrelated_hot", "tight", hi=4.8, lo=0.8),
    ]


# --- occupancy / serialization -------------------------------------------------------------


def peak_occupancy(
    phases: Sequence[DemandPhase],
    capacity: CapacitySurface,
    default_shapes: Mapping[str, tuple[int, int]],
) -> float:
    """Peak fractional Sum_m rho_m * slot_width_m over phases (aggregate lower bound)."""
    peak = 0.0
    for phase in phases:
        occ = sum(rho * MODEL_SLOT_WIDTH[model] for model, rho in phase.rho_by_model.items() if rho > 0.0)
        peak = max(peak, occ)
    return peak


# --- integer GPU feasibility (traceset-v2) --------------------------------------------------
# The effective rho lint sees equals the design rho (rps = rho * C so rho = rps/C cancels the
# surface). Integer feasibility therefore reads straight off phase.rho_by_model: every model
# with rho > 0 needs ceil(rho) replicas (>= 1 serving floor), each occupying slot_width GPUs.


def phase_replica_table(
    phase: DemandPhase,
    slot_widths: Mapping[str, float] = MODEL_SLOT_WIDTH,
) -> dict:
    """Per-model replica/GPU requirement for one phase, plus the phase GPU total."""
    per_model: dict[str, dict] = {}
    gpu_total = 0.0
    for model, rho in phase.rho_by_model.items():
        if rho <= 0.0:
            continue
        replicas = replicas_for_rho(rho)
        width = float(slot_widths[model])
        gpu = replicas * width
        per_model[model] = {
            "rho": round(rho, 4),
            "replicas": replicas,
            "slot_width": width,
            "gpu": gpu,
        }
        gpu_total += gpu
    return {
        "phase": phase.name,
        "start_s": _round(phase.start_s),
        "end_s": _round(phase.end_s),
        "per_model": per_model,
        "gpu_total": gpu_total,
    }


def peak_integer_occupancy(
    phases: Sequence[DemandPhase],
    slot_widths: Mapping[str, float] = MODEL_SLOT_WIDTH,
) -> float:
    """Peak Sum_m ceil(rho_m) * slot_width_m over phases -- the real physical GPU requirement."""
    return max((phase_replica_table(p, slot_widths)["gpu_total"] for p in phases), default=0.0)


def assert_feasible(
    design: "TraceDesign",
    slot_widths: Mapping[str, float] = MODEL_SLOT_WIDTH,
    total_slots: float = TOTAL_SLOTS,
) -> None:
    """Reject a design whose integer GPU requirement exceeds the cluster in any phase.

    This is the guard traceset-v1 lacked: fractional occupancy Sum(rho*width) can read <= 8
    while the integer requirement Sum(ceil(rho)*width) exceeds 8 (v1's A1: 6.8 fractional but
    9 integer). Such a trace is physically unservable regardless of the controller.
    """
    for phase in design.phases:
        table = phase_replica_table(phase, slot_widths)
        if table["gpu_total"] > total_slots + 1e-9:
            raise ValueError(
                f"infeasible trace {design.name}: phase {phase.name} needs "
                f"{table['gpu_total']:g} GPU > {total_slots:g} slots; per-model {table['per_model']}"
            )


def feasibility_proof(
    phases: Sequence[DemandPhase],
    slot_widths: Mapping[str, float] = MODEL_SLOT_WIDTH,
    total_slots: float = TOTAL_SLOTS,
) -> dict:
    """Per-phase replica-requirement table used as the INDEX feasibility proof (<= total_slots)."""
    tables = [phase_replica_table(p, slot_widths) for p in phases]
    peak = max(tables, key=lambda t: t["gpu_total"]) if tables else {"phase": None, "gpu_total": 0.0}
    return {
        "total_slots": total_slots,
        "peak_gpu": peak["gpu_total"],
        "peak_phase": peak["phase"],
        "feasible": peak["gpu_total"] <= total_slots + 1e-9,
        "phases": tables,
    }


def segments_to_trace_json(segments: Sequence[RpsSegment]) -> dict[str, list[dict]]:
    """RpsSegment list -> the model-keyed trace.json schema (JSON field max_tokens)."""
    out: dict[str, list[dict]] = {}
    for seg in sorted(segments, key=lambda s: (s.model, s.start_s)):
        entry: dict = {
            "start_time": _round(seg.start_s),
            "end_time": _round(seg.end_s),
            "rps": round(seg.rps, 4),
        }
        if seg.input_tokens is not None:
            entry["input_tokens"] = seg.input_tokens
        if seg.max_output_tokens is not None:
            entry["max_tokens"] = seg.max_output_tokens
        out.setdefault(seg.model, []).append(entry)
    return out


def _round(value: float) -> float | int:
    return int(value) if float(value).is_integer() else round(value, 3)


def load_capacity_surface(paths: Sequence[str | Path]) -> CapacitySurface:
    """Reconstruct a CapacitySurface from capacity_<model>.json files (r3_capacity output)."""
    points: dict[tuple[str, int, int], float] = {}
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = data["model"]
        for pt in data["capacity"]:
            points[(model, int(pt["input_tokens"]), int(pt["output_tokens"]))] = float(pt["rps"])
    return CapacitySurface(points=points)


def generate_trace_set(
    capacity: CapacitySurface,
    out_dir: str | Path,
    *,
    version: str = "experiment3-v2",
    designs_factory: Callable[[CapacitySurface], list[TraceDesign]] = build_designs,
) -> dict:
    """Write trace.json per design + INDEX.json; return the INDEX payload."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    designs = designs_factory(capacity)

    entries: list[dict] = []
    for design in designs:
        # Physical feasibility guard: reject any design that needs > TOTAL_SLOTS integer GPUs.
        assert_feasible(design)
        segments = design_trace_segments(design.phases, capacity, token_shapes=dict(design.default_shapes))
        trace = segments_to_trace_json(segments)
        case_dir = root / design.name
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "trace.json").write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")

        frac = peak_occupancy(design.phases, capacity, design.default_shapes)
        integer_occ = peak_integer_occupancy(design.phases)
        entries.append(
            {
                "name": design.name,
                "axis": design.axis,
                "headroom_tier": design.headroom,
                "headroom_target": HEADROOM_TARGETS[design.headroom],
                # traceset-v2 occupancy basis: integer replicas x slot_width.
                "integer_gpu_occupancy": integer_occ,
                "peak_integer_headroom": round(integer_occ / TOTAL_SLOTS, 4),
                # fractional Sum(rho*width) kept for reference (the aggregate lower bound).
                "peak_fractional_occupancy_slots": round(frac, 4),
                "peak_fractional_headroom": round(frac / TOTAL_SLOTS, 4),
                "headroom_is_capacity_dependent": design.headroom_is_capacity_dependent,
                "feasibility": feasibility_proof(design.phases),
                "mechanism": design.mechanism,
            }
        )

    index = {
        "version": version,
        "total_slots": TOTAL_SLOTS,
        "model_slot_widths": MODEL_SLOT_WIDTH,
        # loader.discover_trace_set expects a flat list of names under "workloads".
        "workloads": [entry["name"] for entry in entries],
        "designs": entries,
    }
    (root / "INDEX.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the experiment-3 trace set from R3 capacity JSON.")
    ap.add_argument(
        "--capacity", action="append", required=True,
        help="capacity_<model>.json (repeat for each model, e.g. --capacity a.json --capacity b.json)",
    )
    ap.add_argument("--out-dir", required=True, help="output directory for the trace set")
    ap.add_argument("--version", default="experiment3-v2")
    args = ap.parse_args(argv)

    capacity = load_capacity_surface(args.capacity)
    index = generate_trace_set(capacity, args.out_dir, version=args.version)
    print(json.dumps({"version": index["version"], "workloads": index["workloads"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

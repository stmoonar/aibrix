#!/usr/bin/env python3
"""The training supplement (plan 2026-09-21 §6.11 D16-D22, step ③): constant-load hold
cells that close the gaps the D6' refit left, on the ladder's per-cell machinery.

Entered from ``python -m scripts.calibration_campaign --training-supplement --models MODEL
--base-run <second round> --boundary-supplement-run <D19 supplement>`` (``run_stage3.sh``
on the host). One model per run; the three may run in parallel.

What it drives (every cell a 300 s constant-load hold, first 60 s warm-up, except the
240 s sentinels), per model - :data:`PLAN`:

③a  S3 ladder: {0.85, 0.95, 1.0, 1.05, 1.15} x rho*_D6'(S3) x 2 replicates. rho*_D6'(S3)
    is the boundary supplement's measured anchor (D19/D20): under the D6' label S3's
    boundary moved out of the second round's ladder, so no training cell sits on it.
    14b gets 0.75 x as well (x 2): its transition zone is wide and noisy - the supplement
    probes read 36 % violating windows at 1.48 C_s (0.906 x its anchor), 48 % at 1.60, 81 %
    on the smoke hold at the anchor 1.634 - and the second round's D6'-relabelled S3 cells
    read 10 % at 1.43 C_s (0.875 x) and 0 % at 1.20 C_s (0.735 x). 0.85 x (1.39 C_s) is
    therefore mostly healthy but not surely; 0.75 x (1.23 C_s) is. 7b / 8b already have a
    clean healthy side at 0.85 x (7b: 14 % at 0.84 x of its anchor on the supplement probe,
    8 % at 0.85 x on run 2; 8b: 0 % up to 0.98 x) and a violated side at 1.05 x (7b 76 % at
    1.07 x; 8b 57 % at 1.016 x, 71 % at 1.08 x).
③b  14b CI: T8 x {0.95, 0.975, 1.0, 1.025, 1.05, 1.10} x rho*_D6'(T8) x 2 = 12 cells. The
    unified tau = 10 s refit left 14b's theta CI half width at 15.9-16.3 % (needs ~1.18 x
    the effective data); its prefill family has the fewest boundary-band windows (22-31
    independent against 41-52 for decode). S3 is ③a's; the other prefill shape is T8,
    whose D6' boundary for 14b is 1.006 x rho*_run2 (the D6' b50 of run 2's T8 ladder,
    ``calibration_rev2_20260923/analysis_step0/step0_2_t8.txt``) and whose transition is
    steep (7 % violating at 0.95 x, 47 % at 1.0 x, 83 % at 1.05 x) - hence the dense
    spacing inside 0.95-1.10.
7b T8: {1.1, 1.15, 1.2} x rho*_run2(T8), once each. 7b's T8 D6' b50 is ~1.13 x rho*_run2
    and run 2's ladder has only two rungs (1.05, 1.15) in 1.1-1.2.
③d  sentinels: S2 at 0.72 x the rho* run 2's stage 0 measured for S2 (the plateau, run 2's
    drift diagnosis), 240 s, first / middle / last. Drift is judged on TPOT p95 (> 20 %)
    and the running median (> 25 %) - ``calibration_design.sentinel_drift``.

Every cell gets the ladder's discipline (``scripts.calibration_ladder``): a run-unique cell
id (serials from :data:`SERIAL_BASE`, clear of the second round's and the supplement's),
its own arrival seed and prompt key, a bounded drain before it, the void rule (re-run
once, stop on a second void), a ``cells.jsonl`` ledger line; the hold cells are judged by
``calibration_design.hold_cell_verdict`` on the primary (D6') label. The hold cells run in
a random order that interleaves the shapes (seeded; the seed and the order are in the run
manifest). Their rho is absolute, in units of each shape's second-round C_s.

The run ends with checks, printed as a banner and recorded in ``design_result.json``; any
failure makes it exit :data:`EXIT_CHECK_FAILED` (D22: stop and ask the user): a ladder
shape without both a healthy and a violated cell, and a sentinel drift flag.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from tre_common import slo_labels

from scripts import adaptive_boundary as boundary
from scripts import admission_cap as admission
from scripts import calibration_campaign as campaign
from scripts import calibration_design as design
from scripts import calibration_ladder as ladder
from scripts import calibration_supplement as supplement

MODE = "training_supplement"
#: Serials of this collection start here: the second round used a few hundred per model
#: from 1, the boundary supplement 50001+.
SERIAL_BASE = 60_000
HOLD_SECONDS = 300.0
EXIT_CHECK_FAILED = 3

#: Units a cell's factor multiplies (all end up as rho of the shape's second-round C_s).
UNIT_S3_D6PRIME = "rho*_D6'(S3): the boundary supplement's measured anchor"
UNIT_RUN2 = "rho*_run2: the second round's anchor (fixed-label stage 0)"
UNIT_T8_D6PRIME = "rho*_D6'(T8) = D6PRIME_T8_B50 x rho*_run2(T8)"
UNIT_SENTINEL = "rho*_run2(S2): the second round's stage-0 anchor of the sentinel shape"

#: D6' b50 of T8 in units of rho*_run2(T8): the run-2 T8 ladder relabelled under D6'
#: (calibration_rev2_20260923/analysis_step0/step0_2_t8.txt, "D6' ladder b50").
D6PRIME_T8_B50: dict[str, float] = {"dsqwen-14b": 1.006}


@dataclass(frozen=True)
class Group:
    """Cells of one shape at ``factors`` x ``unit``, ``replicates`` each."""

    part: str
    shape: str
    unit: str
    factors: tuple[float, ...]
    replicates: int
    seconds: float
    why: str


S3_LADDER = (0.85, 0.95, 1.0, 1.05, 1.15)
S3_LADDER_14B = (0.75, *S3_LADDER)
T8_CI_14B = (0.95, 0.975, 1.0, 1.025, 1.05, 1.10)
T8_7B = (1.1, 1.15, 1.2)

_S3_WHY = "③a S3 ladder around the D6' boundary the supplement measured"
PLAN: dict[str, tuple[Group, ...]] = {
    "dsqwen-7b": (
        Group("3a", "S3", UNIT_S3_D6PRIME, S3_LADDER, 2, HOLD_SECONDS, _S3_WHY),
        Group("t8", "T8", UNIT_RUN2, T8_7B, 1, HOLD_SECONDS,
              "7b T8 top-up: its D6' b50 ~1.13 x rho*_run2 has two run-2 rungs in 1.1-1.2"),
    ),
    "dsllama-8b": (
        Group("3a", "S3", UNIT_S3_D6PRIME, S3_LADDER, 2, HOLD_SECONDS, _S3_WHY),
    ),
    "dsqwen-14b": (
        Group("3a", "S3", UNIT_S3_D6PRIME, S3_LADDER_14B, 2, HOLD_SECONDS,
              _S3_WHY + "; 0.75 x added: 14b's transition zone starts below 0.85 x"),
        Group("3b", "T8", UNIT_T8_D6PRIME, T8_CI_14B, 2, HOLD_SECONDS,
              "③b 14b theta CI: the prefill family's fewest band windows, T8 is its non-S3 shape"),
    ),
}
#: The ladder shapes the both-sides check applies to (③a, and ③b's T8 for 14b).
BOTH_SIDES_PARTS = ("3a", "3b")

SENTINEL_SHAPE = design.SENTINEL_SHAPE
SENTINEL_FACTOR = design.SENTINEL_RHO_FACTOR
SENTINEL_SECONDS = design.SENTINEL_SECONDS
SENTINEL_POSITIONS = design.SENTINEL_POSITIONS


# ------------------------------------------------------------------------- the units


def load_supplement_anchor(supp_root: Path, base: supplement.BaseAnchor) -> dict:
    """rho*_D6'(S3) from the boundary supplement: measured, and built on the same base run
    (same C_s) as this collection."""
    path = Path(supp_root) / base.model / "boundary" / f"{base.model}_{base.shape}.json"
    doc = supplement._read(path)
    if doc.get("mode") != supplement.MODE:
        raise ValueError(f"{path}: not a boundary-supplement search ({doc.get('mode')!r})")
    if doc.get("rho_star_status") != boundary.RHO_STAR_MEASURED or doc.get("anchor_rho") is None:
        raise ValueError(f"{path}: rho* is {doc.get('rho_star_status')!r}, not measured")
    theirs = (doc.get("base") or {}).get("capacity_rps")
    if theirs is None or not math.isclose(float(theirs), base.capacity_rps, rel_tol=1e-9):
        raise ValueError(f"{path}: built on C_s {theirs}, this run's base has {base.capacity_rps}")
    return {"anchor_rho": float(doc["anchor_rho"]), "path": str(path),
            "sha256": supplement_sha256(path), "bracket": doc.get("rho_star_bracket")}


def supplement_sha256(path: Path) -> str:
    return design._sha256(Path(path))


def resolve_units(model: str, base_root: Path, supp_root: Path) -> dict:
    """Per (shape, unit) the rho (of the shape's run-2 C_s) a factor of 1 means, with
    where it came from; plus each shape's C_s."""
    shapes = sorted({g.shape for g in PLAN[model]} | {SENTINEL_SHAPE})
    bases = {s: supplement.load_base_anchor(base_root, model, s) for s in shapes}
    units: dict[tuple[str, str], dict] = {}
    for group in PLAN[model]:
        base = bases[group.shape]
        if group.unit == UNIT_S3_D6PRIME:
            anchor = load_supplement_anchor(supp_root, base)
            units[(group.shape, group.unit)] = {"rho": anchor["anchor_rho"], "source": anchor}
        elif group.unit == UNIT_RUN2:
            units[(group.shape, group.unit)] = {"rho": base.anchor_rho, "source": base.as_dict()}
        elif group.unit == UNIT_T8_D6PRIME:
            b50 = D6PRIME_T8_B50[model]
            units[(group.shape, group.unit)] = {
                "rho": round(b50 * base.anchor_rho, 6),
                "source": {"d6prime_b50_of_rho_star_run2": b50, "base": base.as_dict()}}
        else:  # pragma: no cover - PLAN only uses the units above
            raise ValueError(f"unknown unit {group.unit!r}")
    sentinel = bases[SENTINEL_SHAPE]
    units[(SENTINEL_SHAPE, UNIT_SENTINEL)] = {"rho": sentinel.anchor_rho, "source": sentinel.as_dict()}
    return {"units": units, "capacity": {s: b.capacity_rps for s, b in bases.items()},
            "bases": {s: b.as_dict() for s, b in bases.items()}}


# -------------------------------------------------------------------------- the plan


def interleave(cells: Sequence[design.DesignCell], rng: random.Random) -> list[design.DesignCell]:
    """A random order that never puts one shape back to back while another shape still
    has cells: each shape's cells shuffled, then drawn shape by shape with probability
    proportional to what each has left."""
    by_shape: dict[str, list[design.DesignCell]] = {}
    for cell in cells:
        by_shape.setdefault(cell.shape, []).append(cell)
    for shape in sorted(by_shape):
        rng.shuffle(by_shape[shape])
    out: list[design.DesignCell] = []
    previous: Optional[str] = None
    while any(by_shape.values()):
        left = sorted(s for s, c in by_shape.items() if c)
        pool = [s for s in left if s != previous] or left
        weights = [len(by_shape[s]) for s in pool]
        pick = rng.choices(pool, weights=weights)[0]
        out.append(by_shape[pick].pop())
        previous = pick
    return out


def build_cells(model: str, factory: design.CellFactory, resolved: Mapping,
                design_seed: int) -> tuple[list[design.DesignCell], list[design.DesignCell]]:
    """(the run's cell sequence, the sentinels). Ids and seeds are fixed here."""
    units = resolved["units"]
    holds = []
    for group in PLAN[model]:
        unit_rho = units[(group.shape, group.unit)]["rho"]
        for factor in group.factors:
            for rep in range(1, group.replicates + 1):
                holds.append(factory.new(
                    group.shape, design.ROLE_LADDER, group.seconds, rho_factor=factor,
                    rho=round(factor * unit_rho, 6), replicate=rep,
                    note=f"part {group.part}: {factor:g} x {group.unit}"))
    sentinel_rho = round(SENTINEL_FACTOR * units[(SENTINEL_SHAPE, UNIT_SENTINEL)]["rho"], 6)
    sentinels = [
        factory.new(SENTINEL_SHAPE, design.ROLE_SENTINEL, SENTINEL_SECONDS,
                    rho_factor=SENTINEL_FACTOR, rho=sentinel_rho, replicate=i + 1,
                    position=position, note=f"part 3d: {SENTINEL_FACTOR:g} x {UNIT_SENTINEL}")
        for i, position in enumerate(SENTINEL_POSITIONS)
    ]
    rng = random.Random(design.derived_seed(design_seed, model, "training-supplement-order"))
    order = interleave(holds, rng)
    half = len(order) // 2
    sequence = [sentinels[0], *order[:half], sentinels[1], *order[half:], sentinels[2]]
    return sequence, sentinels


def part_of(cell: design.DesignCell) -> str:
    return cell.note.split(":", 1)[0].replace("part ", "") if cell.note.startswith("part ") else ""


def estimate(sequence: Sequence[design.DesignCell], cooldown_s: float) -> dict:
    """Wall clock: every cell pays a gap (expected: the cooldown, upper: the drain limit;
    plus the driver's start-up); a cell above its boundary can outlive its schedule by the
    request deadline while its backlog finishes (expected: the cells at >= 1.05 x, upper:
    all of them)."""
    gap_exp = max(cooldown_s, 0.0) + ladder.DRIVER_OVERHEAD_S
    gap_up = max(cooldown_s, design.DRAIN_LIMIT_S) + ladder.DRIVER_OVERHEAD_S
    load = sum(c.duration_s for c in sequence)
    tails_exp = sum(design.request_timeout_s(c.shape) for c in sequence
                    if c.role != design.ROLE_SENTINEL and (c.rho_factor or 0) >= 1.05)
    tails_up = sum(design.request_timeout_s(c.shape) for c in sequence)
    return {"cells": len(sequence), "offered_load_s": load,
            "seconds_expected": round(load + tails_exp + len(sequence) * gap_exp, 1),
            "seconds_upper": round(load + tails_up + len(sequence) * gap_up, 1)}


def label_doc(label: slo_labels.LabelDefinition, args) -> dict:
    return {
        "ttft_slo_mode": label.ttft_slo_mode, "ttft_slowdown_k": label.ttft_slowdown_k,
        "ttft_floor_ms": label.ttft_floor_ms, "ttft_idle_c_ms": label.ttft_idle_c_ms,
        "ttft_idle_b_ms_per_token": label.ttft_idle_b_ms_per_token,
        "tpot_p95_ms": label.tpot_p95_ms, "min_completed_requests": label.min_completed_requests,
        "window_ms": args.window_ms, "step_ms": args.fit_step_ms,
        "window_align": getattr(args, "fit_window_align", "grid"),
    }


def build_plan(args, model: str, sequence: Sequence[design.DesignCell], resolved: Mapping,
               cap) -> tuple[dict, dict]:
    """(plan.json, run manifest)."""
    label = campaign.primary_label(args, model)
    provenance = campaign.run_provenance(args)
    units_doc = [{"shape": s, "unit": u, "rho": v["rho"], "source": v["source"]}
                 for (s, u), v in sorted(resolved["units"].items())]
    groups = [{"part": g.part, "shape": g.shape, "unit": g.unit, "factors": list(g.factors),
               "replicates": g.replicates, "seconds": g.seconds, "why": g.why}
              for g in PLAN[model]]
    cells = [c.as_dict() for c in sequence]
    est = estimate(sequence, args.cooldown_s)
    plan = {
        # "ladder": the standard dataset reads this run from its ledger like any ladder run.
        "design": ladder.DESIGN_NAME,
        "mode": MODE,
        "generated_at_utc": campaign.utc_iso(),
        "provenance": provenance,
        "admission_cap": cap.as_dict(),
        "models": [model],
        "design_seed": int(args.design_seed),
        "serial_base": SERIAL_BASE,
        "cooldown_s": args.cooldown_s,
        "run_manifest": ladder.RUN_MANIFEST,
        "static_cells": {model: cells},
        "sequence": [c.cell_id for c in sequence],
        "groups": groups,
        "units": units_doc,
        "capacity_rps": resolved["capacity"],
        "label": label_doc(label, args),
        "estimate": est,
    }
    manifest = {
        "design": ladder.DESIGN_NAME,
        "mode": MODE,
        "written_at_utc": campaign.utc_iso(),
        "why": ("plan 2026-09-21 §6.11 D16-D22 step ③: constant-load training cells on the "
                "D6' boundaries (③a S3 ladder, ③b 14b CI, 7b T8 top-up, ③d sentinels)"),
        "base_run": str(args.base_run),
        "boundary_supplement_run": str(args.boundary_supplement_run),
        "bases": resolved["bases"],
        "units": units_doc,
        "groups": groups,
        "sentinel_rule": (f"{SENTINEL_FACTOR} x rho*_run2({SENTINEL_SHAPE}), {SENTINEL_SECONDS:g} s, "
                          f"positions {list(SENTINEL_POSITIONS)}; drift on "
                          f"{design.SENTINEL_DRIFT_THRESHOLDS}"),
        "checks": ("both sides: every ③a / ③b shape has >= 1 healthy and >= 1 violated hold "
                   "cell; sentinel drift not flagged. A failure exits "
                   f"{EXIT_CHECK_FAILED} (D22)"),
        "label": provenance.get("label"),
        "design_seed": int(args.design_seed),
        "serial_base": SERIAL_BASE,
        "seed_derivation": ("as the ladder design (calibration_design.CellFactory); order from "
                            "random.Random(derived_seed(design_seed, model, "
                            "'training-supplement-order')), shapes interleaved"),
        "sequence": [c.cell_id for c in sequence],
        "static_plan": {model: cells},
        "models": [model],
        "cooldown_floor_s": args.cooldown_s,
        "provenance": provenance,
        "admission_cap": cap.as_dict(),
    }
    return plan, manifest


def print_plan(plan: dict, model: str) -> None:
    lb = plan["label"]
    print(f"label (D6' primary): TTFT p95 <= max({lb['ttft_floor_ms']:g} ms, "
          f"{lb['ttft_slowdown_k']:g} x ({lb['ttft_idle_c_ms']:g} + {lb['ttft_idle_b_ms_per_token']:g} x L)), "
          f"TPOT p95 <= {lb['tpot_p95_ms']:g} ms, >= {lb['min_completed_requests']} completions, "
          f"{lb['window_ms'] / 1000:g} s windows / {lb['step_ms'] / 1000:g} s step ({lb['window_align']})")
    for u in plan["units"]:
        cs = plan["capacity_rps"][u["shape"]]
        print(f"  unit {u['shape']:3} {u['rho']:.6g} x C_s {cs:g} = {u['rho'] * cs:.4g} rps  [{u['unit']}]")
    print(f"{model}: {len(plan['sequence'])} cells in run order (rho = x C_s of the shape):")
    for i, cell in enumerate(plan["static_cells"][model], start=1):
        cs = plan["capacity_rps"][cell["shape"]]
        print(f"  {i:2d} {cell['cell_id']:22} {cell['role']:8} {cell['shape']:3} "
              f"{cell['rho_factor']:>6g} x -> rho {cell['rho']:.4f} ({cell['rho'] * cs:.3f} rps) "
              f"{cell['duration_s']:.0f}s  {cell['note']}")
    est = plan["estimate"]
    print(f"  {est['cells']} cells, offered load {est['offered_load_s'] / 60:.0f} min; wall clock "
          f"~{est['seconds_expected'] / 3600:.2f} h expected, <= {est['seconds_upper'] / 3600:.2f} h "
          "(re-drives of void cells excluded)")


# ----------------------------------------------------------------------------- run


class PlannedRun(ladder.LadderRun):
    """A fixed list of cells on the ladder's per-cell machinery; per-shape capacity given."""

    def __init__(self, args, model: str, capacity: Mapping[str, float], *, factory, cap,
                 out_dir: Path, raw_dir: Path, drive: Callable, sample: Callable[[], dict],
                 capacity_source: str = "rho_priors",
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        empty = design.StaticPlan(sentinels=[], ladder_rounds=[], ramps=[])
        super().__init__(args, model, None, factory=factory, plan=empty, cap=cap,
                         out_dir=out_dir, raw_dir=raw_dir, drive=drive, sample=sample,
                         sleep=sleep, clock=clock)
        self.capacities = dict(capacity)
        self.capacity_source = capacity_source
        self.outcomes: list[dict] = []

    def capacity(self, shape: str) -> float:
        return self.capacities[shape]

    def drive_planned(self, cell: design.DesignCell) -> dict:
        """Drive one cell (void rule included); a sentinel's summary joins the drift check."""
        print(f"[{self.model}] {cell.role} {cell.shape} {cell.cell_id} rho={cell.rho} "
              f"({cell.duration_s:.0f}s) {cell.note}", flush=True)
        rows, guard, record = self.drive_cell(cell)
        if cell.role == design.ROLE_SENTINEL:
            summary = design.sentinel_summary(rows, guard, warmup_s=cell.warmup_s, label=self.label)
            summary.update({"position": cell.position, "cell_id": cell.cell_id})
            self.sentinel_summaries.append(summary)
        outcome = {"cell_id": cell.cell_id, "role": cell.role, "shape": cell.shape,
                   "part": part_of(cell), "rho_factor": cell.rho_factor, "rho": cell.rho,
                   "attempt": record["attempt"], "verdict": record.get("verdict"),
                   "violating_windows": record.get("violating_windows"),
                   "labeled_windows": record.get("labeled_windows")}
        self.outcomes.append(outcome)
        return outcome


class TrainingSupplementRun(PlannedRun):
    def __init__(self, *a, sequence: Sequence[design.DesignCell], **kw) -> None:
        super().__init__(*a, **kw)
        self.sequence = list(sequence)

    def run(self, shapes: Sequence[str] = ()) -> None:
        for cell in self.sequence:
            self.drive_planned(cell)

    def failures(self) -> list[str]:
        out = []
        for part in BOTH_SIDES_PARTS:
            by_shape: dict[str, list[str]] = {}
            for o in self.outcomes:
                if o["part"] == part:
                    by_shape.setdefault(o["shape"], []).append(o["verdict"])
            for shape, verdicts in sorted(by_shape.items()):
                healthy = verdicts.count(boundary.VERDICT_HEALTHY)
                violated = verdicts.count(boundary.VERDICT_VIOLATED)
                if not healthy or not violated:
                    out.append(f"{self.model}/{shape} part {part}: {healthy} healthy and {violated} "
                               f"violated cells of {len(verdicts)} - the boundary is not bracketed")
        drift = design.sentinel_drift(self.sentinel_summaries)
        if drift.get("flagged"):
            out.append(f"{self.model}: sentinel drift flagged ({drift['verdict']})")
        elif drift.get("flagged") is None and self.sentinel_summaries:
            out.append(f"{self.model}: sentinel drift undetermined ({drift['verdict']})")
        return out

    def result(self, status: str) -> dict:
        drift = design.sentinel_drift(self.sentinel_summaries)
        return {"model": self.model, "status": status, "cells": self.outcomes,
                "sentinels": self.sentinel_summaries, "sentinel_drift": drift,
                "check_failures": self.failures() if status == "complete" else [],
                "possibly_contaminated_cells": self.contaminated,
                "attempts": len(self.records)}


def single_model(args) -> str:
    models = [m for m in str(args.models).split(",") if m]
    if len(models) != 1:
        raise ValueError(f"one model per run (one ledger, one out-dir); got {models}")
    if models[0] not in PLAN:
        raise ValueError(f"no plan for {models[0]!r}")
    return models[0]


def resolve_cap(args):
    index_cap = None
    if getattr(args, "index", None) and Path(args.index).exists():
        index_cap = (json.loads(Path(args.index).read_text(encoding="utf-8"))
                     .get("admission_cap", {}).get("name"))
    return admission.get_cap(getattr(args, "cap", None) or index_cap or admission.DEFAULT_CAP_NAME)


def check_primary_label(args, model: str) -> None:
    label = campaign.primary_label(args, model)
    if label.ttft_slo_mode != slo_labels.TTFT_SLO_MODE_SLOWDOWN:
        raise ValueError("this collection is judged on the D6' primary label (slowdown TTFT); "
                         f"refusing --fit-ttft-slo-mode {label.ttft_slo_mode}")


def finish(out_dir: Path, plan: dict, result: dict, status: str, code: int) -> None:
    (out_dir / ladder.DESIGN_RESULT).write_text(
        json.dumps({"mode": plan.get("mode"), "run_manifest_sha256": plan.get("run_manifest_sha256"),
                    "models": [result]}, indent=2) + "\n", encoding="utf-8")
    campaign.finalize_run(out_dir, status=status, exit_code=code)


def banner(lines: Sequence[str]) -> None:
    bar = "!" * 78
    print(f"{bar}\nCHECK FAILED - do not go on to the next stage without the user:", flush=True)
    for line in lines:
        print(f"  {line}", flush=True)
    print(bar, flush=True)


def run_training_supplement(args, *, drive: Optional[Callable] = None,
                            sample_factory: Optional[Callable[[str], Callable]] = None,
                            sleep: Callable[[float], None] = time.sleep,
                            clock: Callable[[], float] = time.monotonic,
                            check_controller: bool = True) -> int:
    """``--training-supplement``: see the module docstring."""
    ladder.check_label_args(args)
    model = single_model(args)
    check_primary_label(args, model)
    if not args.base_run or not args.boundary_supplement_run:
        raise ValueError("--training-supplement needs --base-run and --boundary-supplement-run")
    out_dir = Path(args.out_dir)
    for source in (args.base_run, args.boundary_supplement_run):
        supplement.check_new_out_dir(out_dir, Path(source))
    resolved = resolve_units(model, Path(args.base_run), Path(args.boundary_supplement_run))
    cap = resolve_cap(args)
    factory = design.CellFactory(model, int(args.design_seed), serial_base=SERIAL_BASE)
    sequence, _sentinels = build_cells(model, factory, resolved, int(args.design_seed))
    plan, manifest = build_plan(args, model, sequence, resolved, cap)
    out_dir.mkdir(parents=True, exist_ok=True)
    print_plan(plan, model)
    if args.dry_run:
        plan["run_manifest_preview"] = manifest
        (out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"dry run: wrote {out_dir / 'plan.json'} (the run manifest is only written when a "
              "run starts)")
        return 0

    if check_controller:
        mode = campaign.controller_mode(args.controller_namespace)
        if mode != campaign.REQUIRED_CONTROLLER_MODE:
            raise SystemExit(f"controller mode is {mode!r}, refusing to run (need "
                             f"{campaign.REQUIRED_CONTROLLER_MODE!r})")
        print(f"controller mode: {mode}")
    plan["run_manifest_sha256"] = ladder.write_frozen(out_dir / ladder.RUN_MANIFEST, manifest)
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    drive = drive or ladder.subprocess_drive(args, raw_dir)
    sample_factory = sample_factory or (lambda m: ladder.make_engine_sampler(m, args.model_namespace))
    run = TrainingSupplementRun(
        args, model, resolved["capacity"], factory=factory, cap=cap, out_dir=out_dir,
        raw_dir=raw_dir, drive=drive, sample=sample_factory(model), sleep=sleep, clock=clock,
        sequence=sequence)
    status, code = "failed", 1
    result = None
    try:
        try:
            run.run()
        except ladder.CampaignStopped as stop:
            print(f"STOPPED: {stop}", flush=True)
            status, code = "stopped", stop.code or 1
            result = run.result(f"stopped: {stop}")
            return code
        result = run.result("complete")
        if result["check_failures"]:
            banner(result["check_failures"])
            status, code = "complete_check_failed", EXIT_CHECK_FAILED
            return code
        print(f"[{model}] complete; sentinel drift: {result['sentinel_drift']['verdict']}", flush=True)
        status, code = "complete", 0
        return 0
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        if result is None:
            result = run.result(status)
        finish(out_dir, plan, result, status, code)

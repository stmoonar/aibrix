#!/usr/bin/env python3
"""The acceptance set M (plan 2026-09-21 §6.11 D16 / D22, step ④; §6.9f A-D): collected
after the parameters are frozen, sealed as soon as it is collected, read once by
``dline_refit accept``.

Entered from ``python -m scripts.calibration_campaign --acceptance-set --models MODEL
--freeze-file <frozen parameters> ...`` (``run_M.sh`` on the host). One model per run.

Composition (fixed by the user, 13 cells per model)
---------------------------------------------------
* 3 retained - the first round's mixture M (40/30/20/10) x {steps, ramp, bursts}, re-used,
  not re-collected (their revision-2 windows are read from ``--retained-dataset``). NOTE:
  the 2026-09-22 offline refit reported a BA on the first round's M, i.e. on exactly these
  three cells - they have been looked at once. The user keeps them; the manifest carries
  ``seen_before: true`` and the note.
* 10 collected (:data:`COMPOSITION`):
  - MP (prefill-leaning 70/30) x {steps 450 s, bursts 360 s, hold 1.05 rho* 600 s, hold
    1.15 rho* 600 s} - the two holds are 600 s so that TTFT-only violations reach >= 30
    independent windows;
  - MD (decode-leaning 30/70) x {steps 450 s, ramp 435 s};
  - G800x240 (the static grid's held-out slice) x {hold 0.9 rho*, hold 1.05 rho*}, 300 s;
  - U512x512 (a shape no training cell used) x {hold 1.0 rho* 300 s, steps 450 s}.
  The shapes are ``gen.ACCEPTANCE_SHAPES``. The dynamic cells are the first round's
  primitives (``gen.STEPS``, the 300 s ramp 0.4 -> 1.2 + 75 s hold + 60 s drain at 0.5,
  bursts on 0.6 with four spikes sized by the admission cap) with the capacity unit
  replaced by the shape's measured rho*: the retained M cells ran them on a capacity
  prior that sat far below M's boundary; here they are placed on the boundary itself.

rho* of the new shapes: measured here, with the same ruler
-----------------------------------------------------------
Each new shape's boundary is located first by the prior-guided search of the ladder
(``calibration_design.PriorGuidedSearch``: 150 s probes, the first 60 s dropped, verdict
= ``hold_cell_verdict`` on the D6' primary label - the one every fit trains on; step x1.15
from the prior until the verdict flips, then two bisections). Its unit C_s is a
prediction of the shape's D6' boundary rate, so the prior is rho = 1: per model a fit
``1/R = a*i + b*o`` (no intercept) through the D6' boundary rates of the seven training
shapes - the second round's D6' b50 x rho*_run2 x C_s (``--boundary-table``) and, for S3,
the boundary supplement's measured anchor - evaluated at the mean lengths of each stream,
streams combined harmonically by load share. Leave-one-shape-out error of that fit is
within +-22 % on the training shapes, i.e. inside one or two x1.15 steps.

These probes belong to M: split holdout (their shapes are held out), sealed with it and
never trained on; ``dline_refit accept`` evaluates the 13 cells only, not the probes. A
shape whose rho* does not come out ``measured`` stops the run before any M cell is driven
(exit :data:`EXIT_CHECK_FAILED`, D22).

Sealing
-------
Right after the last cell: ``M_SHA256SUMS`` (sha256sum format, absolute paths) over every
file of this run's raw captures, online CSVs, schedules, ledger, plan, run manifest and
boundary searches, plus the retained cells' raw captures; then ``M_manifest.json``
(read-only): the evaluated cells (collected and retained), the sealed probes, the freeze
file's sha256 (M collected after the freeze), the label definition and its canonical
sha256, and the composition. Only then is the standard dataset built. Nothing that fits
reads M: its split is holdout, ``dline_refit trainset --dataset`` skips it unread, and
``accept`` takes it with ``--dataset`` (never ``--h2-dataset``, which would pool it into
H2).

Compositions (``--composition``)
--------------------------------
``m1-20260923`` (default) is everything above, unchanged (stage G, 2026-10-04).

``m2-20261005`` is the next round's M2 (local workspace
``docs/calib-next-round-design-20261005.md`` item 5 and user decision 3; sizing from
``scripts/analysis/m2_power_sim.py``). 24 cells per model, all collected, no retained cell
(M is DEV now), no probes:

* rho* is M's measured boundary, reused as a design constant (:data:`M2_RHO_STAR`, from
  ``<M root>/<model>/boundary/<model>_<shape>.json`` = ``M_manifest.json`` ``rho_star``).
  ``--rho-star-run <M root>`` is checked against the constants (refused on any difference)
  and every source file's path and sha256 go into the plan and the manifest.
* 16 burst cells (:data:`M2_BURST_CELLS`), the shapes MP / MD / U512x512 / G800x240 in
  turn: 0.6 x rho* for 500 s and four heterogeneous spikes, each with its own seed. A
  spike's total offered load is 2, 3 or 4 x rho* (every cell has all three plus one more,
  rotating) and its width carries the same 20 s of rho* work above the boundary
  (``(h - 1) * width = 20 s``), so every spike leaves the same backlog and the cell's
  drains fit its 110 s spike period; the spike seed jitters its start by 0-10 s. Spike
  arrivals are drawn from the cell's one arrival stream (the replayer takes one seed per
  cell), so the per-spike seeds fix the height order and the timing.
* 2 ramps (MD, U512x512), 2 steps (MP, G800x240), three deep holds (1.15 / 1.25 / 1.4 x
  rho*) and one 0.9 x rho* hold, 300 s each.
* every cell waits for the drain gate (``running + waiting == 0``, at most
  :data:`M2_DRAIN_LIMIT_S`, P1's) before it starts.
* new serials (:data:`M2_CELL_SERIAL_BASE`) and a new design seed (20261005); the cell
  codes and every seed are checked against the ledgers under ``--rho-star-run``'s parent
  and every ``--ledger-root`` before a cell is driven.
* the label is the freeze's (hybrid attribution in this round); the plan records it, so
  the campaign builds ``dataset_hybrid/`` next to ``dataset/``.

The seal is the same: ``M_SHA256SUMS`` + ``M_manifest.json`` (format revision 1, so
``dline_refit accept`` reads it as before), plus ``composition.name``, the seeds and the
rho* source.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from scripts import adaptive_boundary as boundary
from scripts import calibration_campaign as campaign
from scripts import calibration_design as design
from scripts import calibration_ladder as ladder
from scripts import calibration_supplement as supplement
from scripts import calibration_training_supplement as training
from scripts import gen_calibration_schedules as gen

MODE = "acceptance_set"
#: Serials: the probes from SERIAL_BASE + 1, the ten cells from CELL_SERIAL_BASE + 1 (so
#: their ids are fixed before any probe runs, whatever the searches take).
SERIAL_BASE = 70_000
CELL_SERIAL_BASE = 70_500
EXIT_CHECK_FAILED = 3
M_MANIFEST = "M_manifest.json"
M_SHA256SUMS = "M_SHA256SUMS"
MANIFEST_FORMAT_REVISION = 1

NEW_SHAPES: tuple[str, ...] = tuple(gen.ACCEPTANCE_SHAPES)
KIND_HOLD, KIND_STEPS, KIND_BURSTS, KIND_RAMP = "hold", "steps", "bursts", "ramp"
_PROFILE = {KIND_HOLD: design.PROFILE_HOLD, KIND_STEPS: design.PROFILE_STEPS,
            KIND_BURSTS: design.PROFILE_BURSTS, KIND_RAMP: design.PROFILE_RAMP}


@dataclass(frozen=True)
class Item:
    shape: str
    kind: str
    factor: Optional[float]
    seconds: float


COMPOSITION: tuple[Item, ...] = (
    Item("MP", KIND_STEPS, None, 450.0),
    Item("MP", KIND_BURSTS, None, 360.0),
    Item("MP", KIND_HOLD, 1.05, 600.0),
    Item("MP", KIND_HOLD, 1.15, 600.0),
    Item("MD", KIND_STEPS, None, 450.0),
    Item("MD", KIND_RAMP, None, 435.0),
    Item("G800x240", KIND_HOLD, 0.9, 300.0),
    Item("G800x240", KIND_HOLD, 1.05, 300.0),
    Item("U512x512", KIND_HOLD, 1.0, 300.0),
    Item("U512x512", KIND_STEPS, None, 450.0),
)
RETAINED_SHAPE = gen.MIXTURE_NAME
RETAINED_PRIMITIVES = ("steps", "ramp", "bursts")
RETAINED_NOTE = ("first-round M cell kept in M by the user (plan §6.11); already looked at "
                 "once: the 2026-09-22 offline refit reported a BA on the first-round M, i.e. "
                 "on these three cells")

#: The probe search of a new shape, in rho of its predicted boundary rate.
PROBE_START_RHO = 1.0
PROBE_STEP = design.STEP_FOUND
PROBE_MAX_RHO = 2.0
#: Shapes whose D6' boundary rates the prior is fitted on.
PRIOR_SHAPES = ("S1", "S2", "S3", "S4", "S5", "T8", "T9")

COMPOSITION_M1 = "m1-20260923"
COMPOSITION_M2 = "m2-20261005"
COMPOSITIONS = (COMPOSITION_M1, COMPOSITION_M2)
DEFAULT_COMPOSITION = COMPOSITION_M1

# ------------------------------------------------------------------- M2 (2026-10-05)

#: Clear of every earlier serial range: the supplements 50_000 / 60_000, P1 62_000, M
#: 70_000 / 70_500, T14 80_500 (and checked against the ledgers at run time).
M2_CELL_SERIAL_BASE = 74_000
M2_DESIGN_SEED = 20261005
#: The drain gate before every M2 cell (= P1's): a 1.4 x rho* hold leaves a backlog.
M2_DRAIN_LIMIT_S = 300.0
#: The bursts' shapes, in turn.
M2_SHAPES: tuple[str, ...] = ("MP", "MD", "U512x512", "G800x240")
M2_BURST_CELLS = 16
M2_SPIKES = 4
#: A spike's total offered load, x rho*.
M2_SPIKE_HEIGHTS: tuple[float, ...] = (2.0, 3.0, 4.0)
#: rho* work above the boundary each spike carries: (h - 1) * width.
M2_SPIKE_EXCESS_S = 20.0
M2_BURST_BASE_RHO = 0.6
M2_BURST_LEAD_S = 60.0
M2_SPIKE_PERIOD_S = 110.0
M2_SPIKE_JITTER_S = 10.0
M2_BURST_SECONDS = M2_BURST_LEAD_S + M2_SPIKES * M2_SPIKE_PERIOD_S
M2_HOLD_SECONDS = 300.0
M2_DEFAULT_ATTRIBUTION = "hybrid"

M2_COMPOSITION: tuple[Item, ...] = (
    *(Item(M2_SHAPES[k % len(M2_SHAPES)], KIND_BURSTS, None, M2_BURST_SECONDS)
      for k in range(M2_BURST_CELLS)),
    Item("MD", KIND_RAMP, None, 435.0),
    Item("U512x512", KIND_RAMP, None, 435.0),
    Item("MP", KIND_STEPS, None, 450.0),
    Item("G800x240", KIND_STEPS, None, 450.0),
    Item("U512x512", KIND_HOLD, 1.15, M2_HOLD_SECONDS),
    Item("MP", KIND_HOLD, 1.25, M2_HOLD_SECONDS),
    Item("MD", KIND_HOLD, 1.4, M2_HOLD_SECONDS),
    Item("G800x240", KIND_HOLD, 0.9, M2_HOLD_SECONDS),
)

#: M's measured rho* (stage G, 2026-10-04): ``(anchor_rho, capacity_rps)`` per model and
#: shape, from ``<M root>/<model>/boundary/<model>_<shape>.json`` (``anchor_rho``,
#: ``capacity_rps``; the same numbers as ``M_manifest.json`` ``rho_star``). The rho* in rps
#: is their product.
M2_RHO_STAR: dict[str, dict[str, tuple[float, float]]] = {
    "dsqwen-7b": {"MP": (0.88587, 5.758585), "MD": (1.13125, 13.113992),
                  "G800x240": (0.983696, 9.503169), "U512x512": (1.09375, 8.25619)},
    "dsllama-8b": {"MP": (1.05625, 4.886986), "MD": (1.13125, 10.294908),
                   "G800x240": (0.983696, 7.809195), "U512x512": (0.983696, 6.552386)},
    "dsqwen-14b": {"MP": (0.855387, 4.496367), "MD": (1.09375, 10.775938),
                   "G800x240": (0.855387, 7.572438), "U512x512": (0.918478, 6.733224)},
}
RHO_STAR_REUSED = "reused_from_M"


# ----------------------------------------------------------------- the boundary prior


def boundary_rates(model: str, base_root: Path, supp_root: Path, table: Path) -> list[dict]:
    """The D6' boundary rate (rps) of every training shape: b50 x rho*_run2 x C_s from the
    second round's relabelled ladder, S3 from the boundary supplement."""
    rows = {r["shape"]: r for r in csv.DictReader(open(table, newline="", encoding="utf-8"))
            if r.get("model") == model}
    out = []
    for shape in PRIOR_SHAPES:
        base = supplement.load_base_anchor(base_root, model, shape)
        (_w, i, o), = gen.shape_components(shape)
        if shape == "S3":
            anchor = training.load_supplement_anchor(supp_root, base)
            rho, source = anchor["anchor_rho"], "boundary supplement anchor (D19)"
        else:
            row = rows.get(shape)
            if row is None or not row.get("P_b50_rf"):
                raise ValueError(f"{table}: no D6' b50 for {model}/{shape}")
            recorded = float(row["rho_star_fixed"])
            if not math.isclose(recorded, base.anchor_rho, rel_tol=2e-3):
                raise ValueError(f"{table}: {model}/{shape} rho*_run2 {recorded} disagrees with "
                                 f"the base run's {base.anchor_rho}")
            rho = float(row["P_b50_rf"]) * base.anchor_rho
            source = f"D6' b50 {float(row['P_b50_rf']):g} x rho*_run2"
        out.append({"shape": shape, "input_mean": gen._length_mean(i),
                    "output_mean": gen._length_mean(o), "rho": round(rho, 6),
                    "capacity_rps": base.capacity_rps,
                    "rate_rps": round(rho * base.capacity_rps, 6), "source": source})
    return out


def fit_rate_model(points: Sequence[Mapping]) -> tuple[float, float]:
    """Least squares of ``1/R = a*i + b*o`` through the origin."""
    sii = sum(p["input_mean"] ** 2 for p in points)
    soo = sum(p["output_mean"] ** 2 for p in points)
    sio = sum(p["input_mean"] * p["output_mean"] for p in points)
    siy = sum(p["input_mean"] / p["rate_rps"] for p in points)
    soy = sum(p["output_mean"] / p["rate_rps"] for p in points)
    det = sii * soo - sio * sio
    if det <= 0:
        raise ValueError("boundary prior: degenerate (input, output) points")
    a = (siy * soo - soy * sio) / det
    b = (soy * sii - siy * sio) / det
    if a <= 0 or b <= 0:
        raise ValueError(f"boundary prior: non-positive cost per token (a={a}, b={b})")
    return a, b


def predicted_rate(shape: str, a: float, b: float) -> float:
    """The shape's predicted D6' boundary rate: streams at their mean lengths, combined
    by load share (the total rate at which the blend meets the boundary)."""
    cost = sum(w * (a * gen._length_mean(i) + b * gen._length_mean(o))
               for w, i, o in gen.shape_components(shape))
    return 1.0 / cost


def boundary_prior(model: str, base_root: Path, supp_root: Path, table: Path) -> dict:
    points = boundary_rates(model, base_root, supp_root, table)
    a, b = fit_rate_model(points)
    loo = []
    for k, p in enumerate(points):
        aa, bb = fit_rate_model(points[:k] + points[k + 1:])
        pred = 1.0 / (aa * p["input_mean"] + bb * p["output_mean"])
        loo.append({"shape": p["shape"], "rate_rps": p["rate_rps"], "loo_rps": round(pred, 4),
                    "loo_error": round(pred / p["rate_rps"] - 1.0, 4)})
    return {
        "model": f"1/R = a*i + b*o, least squares through the origin over {list(PRIOR_SHAPES)}",
        "a_s_per_token": a, "b_s_per_token": b, "points": points, "leave_one_out": loo,
        "predicted_rps": {s: round(predicted_rate(s, a, b), 6) for s in NEW_SHAPES},
        "table": {"path": str(table), "sha256": design._sha256(Path(table))},
    }


# ------------------------------------------------------------------------ profiles


def steps_profile(unit: float) -> list[tuple[float, float, float]]:
    out, t = [], 0.0
    for rho, seconds in gen.STEPS:
        out.append((t, t + seconds, round(rho * unit, 6)))
        t += seconds
    return out


def ramp_profile(unit: float, ramp_s: float) -> list[tuple[float, float, float]]:
    """The first round's ramp (``gen.ramp_segments``) in rho: 0.4 -> 1.2 in ~5 s segments
    valued at their midpoints, a hold at 1.2 for a quarter of the ramp, a drain at 0.5."""
    n = max(1, int(round(ramp_s / gen.RAMP_SEGMENT_S)))
    out = []
    for k in range(n):
        frac = (k + 0.5) / n
        rho = gen.RAMP_RHO_START + (gen.RAMP_RHO_END - gen.RAMP_RHO_START) * frac
        out.append((k * ramp_s / n, (k + 1) * ramp_s / n, round(rho * unit, 6)))
    hold_end = ramp_s + ramp_s * gen.RAMP_HOLD_FRACTION
    out.append((ramp_s, hold_end, round(gen.RAMP_RHO_END * unit, 6)))
    out.append((hold_end, hold_end + gen.RAMP_DRAIN_S, round(gen.RAMP_DRAIN_RHO * unit, 6)))
    return out


def bursts_profile(unit: float, spike_rho: float) -> list[tuple[float, float, float]]:
    """The first round's bursts: 0.6 for the whole cell, four spikes superposed on it."""
    out = [(0.0, gen.BURST_DURATION_S, round(gen.BURST_BASE_RHO * unit, 6))]
    for k in range(gen.BURST_COUNT):
        start = gen.BURST_FIRST_S + k * gen.BURST_PERIOD_S
        out.append((start, start + gen.BURST_WIDTH_S, round(spike_rho, 6)))
    return out


def burst_sizing(shape: str, kv_cache_tokens: int, cap) -> dict:
    sizing = cap.burst_sizing(int(kv_cache_tokens), gen.tokens_per_request(shape))
    body = {"kv_cache_tokens": int(kv_cache_tokens),
            "tokens_per_request": round(gen.tokens_per_request(shape), 3), **sizing.as_dict()}
    if not sizing.reachable:
        raise ValueError(f"{shape} bursts: no admissible spike overshoots the engine "
                         f"({body.get('reason')}); M's composition is fixed - ask the user")
    return body


def spike_plan(model: str, cell_id: str, design_seed: int, burst_index: int) -> list[dict]:
    """The four spikes of M2 burst cell ``burst_index``: all three heights plus one more
    (rotating with the index), in an order seeded by the cell; each spike's own seed
    jitters its start; the width keeps ``(h - 1) * width`` at :data:`M2_SPIKE_EXCESS_S`."""
    heights = [*M2_SPIKE_HEIGHTS, M2_SPIKE_HEIGHTS[burst_index % len(M2_SPIKE_HEIGHTS)]]
    random.Random(design.derived_seed(design_seed, model, cell_id, "spike-order")).shuffle(heights)
    out = []
    for k, height in enumerate(heights):
        seed = design.derived_seed(design_seed, model, cell_id, "spike", k)
        jitter = round(random.Random(seed).uniform(0.0, M2_SPIKE_JITTER_S), 1)
        out.append({"index": k, "seed": seed, "height_x_rho_star": height,
                    "start_s": round(M2_BURST_LEAD_S + k * M2_SPIKE_PERIOD_S + jitter, 1),
                    "width_s": round(M2_SPIKE_EXCESS_S / (height - 1.0), 3), "jitter_s": jitter})
    return out


def het_bursts_profile(unit: float, spikes: Sequence[Mapping]) -> list[tuple[float, float, float]]:
    """M2's bursts: 0.6 x rho* for the whole cell, each spike superposed up to its height."""
    out = [(0.0, M2_BURST_SECONDS, round(M2_BURST_BASE_RHO * unit, 6))]
    for s in spikes:
        out.append((float(s["start_s"]), round(float(s["start_s"]) + float(s["width_s"]), 3),
                    round((float(s["height_x_rho_star"]) - M2_BURST_BASE_RHO) * unit, 6)))
    return out


def profile_for(item: Item, unit: float, capacity_rps: float, cap, bursts: Mapping,
                spikes: Optional[Sequence[Mapping]] = None) -> list:
    if item.kind == KIND_STEPS:
        return steps_profile(unit)
    if item.kind == KIND_RAMP:
        return ramp_profile(unit, cap.ramp_max_s)
    if item.kind == KIND_BURSTS and spikes is not None:
        return het_bursts_profile(unit, spikes)
    if item.kind == KIND_BURSTS:
        spike_rps = bursts[item.shape]["burst_requests"] / gen.BURST_WIDTH_S
        return bursts_profile(unit, spike_rps / capacity_rps)
    raise ValueError(f"{item.kind} has no profile")


def profile_seconds(profile: Sequence[tuple[float, float, float]]) -> float:
    return max(b for _a, b, _r in profile)


# ---------------------------------------------------------------------- the cells


def new_cells(model: str, design_seed: int) -> list[design.DesignCell]:
    """The ten collected cells, ids and seeds fixed, load filled in by :func:`place`."""
    factory = design.CellFactory(model, design_seed, serial_base=CELL_SERIAL_BASE)
    cells = []
    for item in COMPOSITION:
        profile = _PROFILE[item.kind]
        what = (f"hold {item.factor:g} x rho*" if item.kind == KIND_HOLD
                else f"{item.kind} (first-round primitive, unit = rho*)")
        cells.append(factory.new(
            item.shape, design.ROLE_ACCEPTANCE, item.seconds, profile=profile,
            rho_factor=item.factor,
            warmup_s=design.WARMUP_S if item.kind == KIND_HOLD else 0.0,
            note=f"M: {item.shape} {what}, {item.seconds:g} s"))
    return cells


def place(cells: Sequence[design.DesignCell], anchors: Mapping[str, float],
          capacity: Mapping[str, float], cap, bursts: Mapping,
          items: Sequence[Item] = COMPOSITION,
          spikes: Optional[Mapping[str, Sequence[Mapping]]] = None) -> None:
    """Fill in each cell's load from its shape's measured rho*."""
    if len(cells) != len(items):
        raise ValueError(f"{len(cells)} cells for a composition of {len(items)}")
    for cell, item in zip(cells, items):
        unit = float(anchors[cell.shape])
        if item.kind == KIND_HOLD:
            cell.rho = round(item.factor * unit, 6)
            cell.rho_profile = None
        else:
            cell.rho = None
            cell.rho_profile = [list(seg) for seg in profile_for(
                item, unit, capacity[cell.shape], cap, bursts,
                (spikes or {}).get(cell.cell_id) if spikes is not None else None)]
            if abs(profile_seconds(cell.rho_profile) - item.seconds) > 1e-6:
                raise ValueError(f"{cell.cell_id}: {item.kind} lasts "
                                 f"{profile_seconds(cell.rho_profile)} s, M fixes {item.seconds} s")


def m2_cells(model: str, design_seed: int) -> tuple[list[design.DesignCell], dict[str, list[dict]]]:
    """M2's 24 cells (ids and seeds fixed, load filled in by :func:`place`) and the spike
    plan of each burst cell."""
    factory = design.CellFactory(model, design_seed, serial_base=M2_CELL_SERIAL_BASE)
    cells, spikes, burst_index = [], {}, 0
    for item in M2_COMPOSITION:
        if item.kind == KIND_HOLD:
            what = f"hold {item.factor:g} x rho*"
        elif item.kind == KIND_BURSTS:
            what = "bursts (0.6 x rho*, four heterogeneous spikes)"
        else:
            what = f"{item.kind} (first-round primitive, unit = rho*)"
        cell = factory.new(
            item.shape, design.ROLE_ACCEPTANCE, item.seconds, profile=_PROFILE[item.kind],
            rho_factor=item.factor,
            warmup_s=design.WARMUP_S if item.kind == KIND_HOLD else 0.0,
            note=f"M2: {item.shape} {what}, {item.seconds:g} s")
        if item.kind == KIND_BURSTS:
            spikes[cell.cell_id] = spike_plan(model, cell.cell_id, design_seed, burst_index)
            burst_index += 1
        cells.append(cell)
    return cells, spikes


def m2_seeds(cells: Sequence[design.DesignCell], spikes: Mapping[str, Sequence[Mapping]]) -> dict:
    """Every seed of an M2 run, per cell (recorded in the plan and the manifest)."""
    return {c.cell_id: {"arrival_seed": c.arrival_seed, "prompt_key": c.prompt_key,
                        "spike_seeds": [s["seed"] for s in spikes.get(c.cell_id, [])]}
            for c in cells}


# ---------------------------------------------------------------- the M2 rho* reuse


def _sums_index(path: Path) -> dict[str, str]:
    """``{resolved path: sha256}`` of a sha256sum file (absent file -> empty)."""
    out: dict[str, str] = {}
    if not Path(path).is_file():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "  " in line:
            digest, name = line.split("  ", 1)
            out[str(Path(name.strip()).resolve())] = digest.strip()
    return out


def m2_rho_star(model: str, run_root: Optional[Path]) -> dict:
    """M's measured rho* of ``model``, as the design constants, checked against M's
    boundary records under ``run_root`` (``<root>/<model>/boundary/<model>_<shape>.json``)
    and its manifest. Every source file is recorded with its sha256; one that M's
    ``M_SHA256SUMS`` lists must still hash to the sealed digest."""
    if model not in M2_RHO_STAR:
        raise ValueError(f"M2 has no rho* constants for {model}")
    if not run_root:
        raise ValueError("--composition m2-20261005 needs --rho-star-run (the M root whose "
                         "measured rho* M2 reuses)")
    root = Path(run_root) / model
    sealed = _sums_index(root / M_SHA256SUMS)
    manifest_path = root / M_MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"{manifest_path}: missing - the rho* source must be a sealed M run")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sums_path = root / M_SHA256SUMS
    sources = [{"path": str(manifest_path.resolve()), "sha256": _file_sha256(manifest_path)},
               {"path": str(sums_path.resolve()) if sums_path.is_file() else str(sums_path),
                "sha256": _file_sha256(sums_path) if sums_path.is_file() else None}]
    shapes = {}
    for shape, (anchor, capacity) in M2_RHO_STAR[model].items():
        path = root / "boundary" / f"{model}_{shape}.json"
        if not path.is_file():
            raise ValueError(f"{path}: missing (rho* source of {model}/{shape})")
        doc = json.loads(path.read_text(encoding="utf-8"))
        digest = _file_sha256(path)
        want = sealed.get(str(path.resolve()))
        if want is None:
            raise ValueError(f"{path}: not listed in {root / M_SHA256SUMS} - the rho* source "
                             "must be a sealed M run")
        if want != digest:
            raise ValueError(f"{path}: sha256 {digest} != the digest M sealed ({want})")
        from_manifest = (manifest.get("rho_star") or {}).get(shape) or {}
        for what, got in (("boundary anchor_rho", doc.get("anchor_rho")),
                          ("boundary capacity_rps", doc.get("capacity_rps")),
                          ("manifest anchor_rho", from_manifest.get("anchor_rho")),
                          ("manifest capacity_rps", from_manifest.get("capacity_rps"))):
            const = anchor if "anchor" in what else capacity
            if got is None or not math.isclose(float(got), const, rel_tol=1e-9):
                raise ValueError(f"{model}/{shape}: {what} {got} != the M2 design constant {const}")
        if doc.get("rho_star_status") != boundary.RHO_STAR_MEASURED:
            raise ValueError(f"{path}: rho* status {doc.get('rho_star_status')!r}, not measured")
        sources.append({"path": str(path.resolve()), "sha256": digest,
                        "sealed_in": str((root / M_SHA256SUMS).resolve())})
        shapes[shape] = {"anchor_rho": anchor, "capacity_rps": capacity,
                         "rho_star_rps": round(anchor * capacity, 4),
                         "status": RHO_STAR_REUSED, "anchor_rule": doc.get("anchor_rule")}
    return {"rule": ("M's measured rho* (stage G, 2026-10-04) reused as design constants; no "
                     "re-probe (design 2026-10-05 item 5)"),
            "run_root": str(Path(run_root).resolve()), "shapes": shapes, "sources": sources}


# ------------------------------------------------------------- the ledger check (M2)

_CODE_RE = re.compile(r"_c(\d+)$")


def ledger_ids(roots: Sequence[Path], skip: Sequence[Path] = ()) -> dict:
    """Every cell code and arrival seed in the ledgers (``cells.jsonl``) and dataset cell
    tables (``cells.csv``) under ``roots``, skipping anything under ``skip``."""
    skip_resolved = [Path(s).resolve() for s in skip]
    codes: set[int] = set()
    seeds: set[int] = set()
    files: list[str] = []

    def take(cell_id, seed) -> None:
        m = _CODE_RE.search(str(cell_id or ""))
        if m:
            codes.add(int(m.group(1)))
        try:
            if seed not in (None, ""):
                seeds.add(int(seed))
        except (TypeError, ValueError):
            pass

    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for name in ("cells.jsonl", "cells.csv"):
            for path in sorted(root.rglob(name)):
                rp = path.resolve()
                if any(rp == s or s in rp.parents for s in skip_resolved):
                    continue
                files.append(str(rp))
                try:
                    if name.endswith(".jsonl"):
                        for line in path.read_text(encoding="utf-8").splitlines():
                            if line.strip():
                                rec = json.loads(line)
                                take(rec.get("cell_id"), rec.get("arrival_seed"))
                    else:
                        with open(path, newline="", encoding="utf-8") as fh:
                            for row in csv.DictReader(fh):
                                take(row.get("cell_id"), row.get("arrival_seed"))
                except (OSError, ValueError) as exc:
                    raise ValueError(f"{path}: unreadable ledger ({exc})") from exc
    return {"codes": codes, "seeds": seeds, "files": files}


def check_new_ids(cells_by_model: Mapping[str, Sequence[design.DesignCell]],
                  spikes_by_model: Mapping[str, Mapping[str, Sequence[Mapping]]],
                  roots: Sequence[Path], skip: Sequence[Path] = ()) -> dict:
    """M2's cell codes and seeds: unique among themselves and absent from every ledger
    under ``roots``. Returns the report; the caller refuses on ``problems``."""
    seen = ledger_ids(roots, skip)
    problems = []
    codes, seeds = [], []
    for model, cells in cells_by_model.items():
        for c in cells:
            codes.append(c.code)
            seeds.append(c.arrival_seed)
            seeds += [s["seed"] for s in (spikes_by_model.get(model) or {}).get(c.cell_id, [])]
    if len(set(codes)) != len(codes):
        problems.append("M2 cell codes repeat")
    if len(set(seeds)) != len(seeds):
        problems.append("M2 seeds repeat")
    reused_codes = sorted(set(codes) & seen["codes"])
    reused_seeds = sorted(set(seeds) & seen["seeds"])
    if reused_codes:
        problems.append(f"cell codes already in a ledger: {reused_codes[:10]}")
    if reused_seeds:
        problems.append(f"seeds already used as an arrival seed in a ledger: {reused_seeds[:10]}")
    return {"roots": [str(Path(r).resolve()) for r in roots], "ledgers_scanned": len(seen["files"]),
            "codes_in_ledgers": len(seen["codes"]), "seeds_in_ledgers": len(seen["seeds"]),
            "m2_codes": len(codes), "m2_seeds": len(seeds), "problems": problems}


def interleaved_order(cells: Sequence[design.DesignCell], model: str,
                      design_seed: int) -> list[design.DesignCell]:
    rng = random.Random(design.derived_seed(design_seed, model, "acceptance-order"))
    return training.interleave(cells, rng)


def retained_cells(model: str, dataset_dir: Path) -> dict:
    """The three first-round M cells, from the revision-2 dataset of the first round."""
    dataset_dir = Path(dataset_dir)
    if not (dataset_dir / "cells.csv").exists() and (dataset_dir / "dataset" / "cells.csv").exists():
        dataset_dir = dataset_dir / "dataset"
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    run_root = Path(manifest["run_root"])
    found = [r for r in csv.DictReader(open(dataset_dir / "cells.csv", newline="", encoding="utf-8"))
             if r["model"] == model and r["shape"] == RETAINED_SHAPE]
    by_primitive = {r["primitive"]: r for r in found}
    if sorted(by_primitive) != sorted(RETAINED_PRIMITIVES) or len(found) != len(RETAINED_PRIMITIVES):
        raise ValueError(f"{dataset_dir}: {model} has M cells {[r['primitive'] for r in found]}, "
                         f"expected one of each of {list(RETAINED_PRIMITIVES)}")
    cells = []
    for primitive in RETAINED_PRIMITIVES:
        r = by_primitive[primitive]
        if r["split"] != design.SPLIT_HOLDOUT or r["status"] != "valid":
            raise ValueError(f"{dataset_dir}: {model} M {primitive} is split {r['split']!r} / "
                             f"status {r['status']!r}, not holdout / valid")
        files = [run_root / r[k] for k in ("raw_path", "guard_path") if r.get(k)]
        missing = [str(f) for f in files if not f.is_file()]
        if missing:
            raise ValueError(f"retained {model} M {primitive}: raw files missing {missing}")
        cells.append({"model": model, "cell_id": r["cell_id"], "attempt": int(r["attempt"]),
                      "shape": RETAINED_SHAPE, "primitive": primitive, "role": r.get("role") or "",
                      "origin": "retained", "seen_before": True, "note": RETAINED_NOTE,
                      "raw_files": [str(f) for f in files], "windows": int(r["windows"] or 0)})
    from scripts import prompt_corpus as corpus_record

    return {"dataset": str(dataset_dir),
            "load_path": corpus_record.dataset_load_path(manifest),
            "windows_csv_sha256": design._sha256(dataset_dir / "windows.csv"),
            "manifest_sha256": design._sha256(dataset_dir / "manifest.json"),
            "run_root": str(run_root), "cells": cells}


# -------------------------------------------------------------------- the freeze


def frozen_label_def(freeze: Mapping, model: str) -> dict:
    """The label definition the frozen parameters of ``model`` were fitted with."""
    entry = (freeze.get("models") or {}).get(model)
    if not entry:
        raise ValueError(f"the freeze file has no parameters for {model}")
    label = (entry.get("verdict_for_holdout") or {}).get("label_def") or entry.get("label_def")
    if not label:
        raise ValueError(f"the freeze file's {model} entry has no label definition")
    return label


ATTRIBUTION_COMPLETION = "completion"


def label_attribution(label_def: Mapping) -> str:
    """A label record's request-to-window attribution (absent = completion, label v1)."""
    return str(label_def.get("attribution") or ATTRIBUTION_COMPLETION)


def resolve_attribution(args, model: str, freeze_doc: Optional[Mapping], default: str) -> str:
    """The attribution M is judged by: the freeze's label's, never silently another one.
    ``--fit-label-attribution`` may restate it (and must agree); without a freeze (a dry
    run) it or ``default`` stands. Written back to ``args.fit_label_attribution`` so the
    campaign's probe / verdict label (``campaign.primary_label``) uses the same one."""
    given = getattr(args, "fit_label_attribution", None)
    frozen = label_attribution(frozen_label_def(freeze_doc, model)) if freeze_doc else None
    if given and frozen and given != frozen:
        raise ValueError(f"--fit-label-attribution {given} but {model} was frozen under a "
                         f"{frozen}-attribution label")
    value = given or frozen or default
    args.fit_label_attribution = value
    return value


def fit_label(args, model: str):
    """The fit's primary label of ``model`` with the attribution M is judged by."""
    from scripts import dline_refit

    attribution = getattr(args, "fit_label_attribution", None) or ATTRIBUTION_COMPLETION
    return dline_refit.label_for(model, "primary", getattr(args, "registry", None), attribution)


def check_freeze(args, model: str) -> dict:
    """D22: M is collected only under frozen parameters, judged by the frozen label."""
    from scripts import dline_refit

    path = getattr(args, "freeze_file", None)
    if not path or not Path(path).is_file():
        raise ValueError(f"no freeze file at {path}: M is collected only after "
                         "`dline_refit freeze` (D22)")
    doc = dline_refit.verify_freeze(Path(path))
    if getattr(args, "composition", None) == COMPOSITION_M2:
        resolve_attribution(args, model, doc, M2_DEFAULT_ATTRIBUTION)
    ours = fit_label(args, model).as_dict()
    theirs = frozen_label_def(doc, model)
    if dline_refit.canonical_sha256(ours) != dline_refit.canonical_sha256(theirs):
        raise ValueError(f"{path}: {model} was frozen under another label definition than the "
                         "one M would be judged by")
    from scripts import prompt_corpus as corpus_record

    load_path = corpus_record.check_matches_freeze(
        doc, model, campaign.load_path(args), what=str(path), **campaign.mismatch_flags(args))
    return {"path": str(Path(path).resolve()), "sha256": design._sha256(Path(path)),
            "freeze_sha256": doc.get("freeze_sha256"), "load_path": load_path}


def label_documents(args, model: str) -> dict:
    from scripts import dline_refit

    fitted = fit_label(args, model)
    probe_label = campaign.primary_label(args, model)
    if fitted.as_dict() != probe_label.as_dict():
        raise ValueError(f"{model}: the probes' label differs from the fit's primary label - "
                         "M would be located with one ruler and judged with another")
    body = fitted.as_dict()
    return {"label_def": body, "label_def_sha256": dline_refit.canonical_sha256(body)}


# ------------------------------------------------------------------------- sealing


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sealed_files(out_dir: Path, retained: Mapping) -> list[Path]:
    """Every file of the out-dir except ``dataset/``, the seal itself and the per-cell
    capture evidence (``cells/``, :mod:`scripts.calibration_capture`): the capture's redis
    dumps are completed by a backfill that may run after the seal, so they are mutable by
    contract and never part of a seal."""
    from scripts.calibration_capture import CELLS_DIRNAME

    out_dir = Path(out_dir).resolve()
    skip = {M_MANIFEST, M_SHA256SUMS}
    files = sorted(p for p in out_dir.rglob("*")
                   if p.is_file() and p.name not in skip
                   and "dataset" not in p.relative_to(out_dir).parts[:1]
                   and CELLS_DIRNAME not in p.relative_to(out_dir).parts[:-1]
                   and not p.name.startswith("."))
    for cell in retained.get("cells", []):
        files += [Path(f) for f in cell["raw_files"]]
    return files


def write_sha256sums(out_dir: Path, files: Sequence[Path]) -> tuple[Path, str]:
    lines = [f"{_file_sha256(f)}  {Path(f).resolve()}\n" for f in files]
    path = Path(out_dir) / M_SHA256SUMS
    body = "".join(lines)
    path.write_text(body, encoding="utf-8")
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return path, hashlib.sha256(body.encode("utf-8")).hexdigest()


def cell_entry(record: Mapping, raw_dir: Path) -> dict:
    raw = Path(raw_dir) / str(record["stem"])
    return {"model": record["model"], "cell_id": record["cell_id"],
            "attempt": int(record["attempt"]), "shape": record["shape"],
            "primitive": record["primitive"], "role": record["role"],
            "origin": "collected", "seen_before": False, "note": record.get("note", ""),
            "rho": record.get("rho"), "rho_factor": record.get("rho_factor"),
            "duration_s": record.get("duration_s"),
            "raw_files": sorted(str(p.resolve()) for p in raw.rglob("*") if p.is_file())
            if raw.is_dir() else []}


# ----------------------------------------------------------------------------- run


class AcceptanceRun(training.PlannedRun):
    def __init__(self, *a, cells: Sequence[design.DesignCell], bursts: Mapping,
                 items: Sequence[Item] = COMPOSITION,
                 spikes: Optional[Mapping[str, Sequence[Mapping]]] = None, **kw) -> None:
        super().__init__(*a, **kw)
        self.cells = list(cells)
        self.bursts = dict(bursts)
        self.items = tuple(items)
        self.spikes = None if spikes is None else dict(spikes)
        self.statuses: dict[str, dict] = {}
        self.final_records: dict[str, dict] = {}

    def reuse_rho_star(self, rho_star: Mapping) -> None:
        """M2: M's measured rho* as the anchors; no probe runs."""
        for shape, body in rho_star["shapes"].items():
            self.anchors[shape] = float(body["anchor_rho"])
            self.anchor_sources[shape] = f"{RHO_STAR_REUSED}: {body.get('anchor_rule')}"
            self.statuses[shape] = {"status": RHO_STAR_REUSED}

    def locate(self) -> list[str]:
        """Probe every new shape (round-robin, seeded order); the failures, in words."""
        order = list(NEW_SHAPES)
        self.rng.shuffle(order)
        for shape in order:
            self.searches[shape] = design.PriorGuidedSearch(
                model=self.model, shape=shape, start_rho=PROBE_START_RHO,
                search_max=PROBE_MAX_RHO, step=PROBE_STEP, prior_found=True)
        self.drive_searches(order)
        failures = []
        for shape in order:
            search = self.searches[shape]
            self.statuses[shape] = search.status()
            if self.statuses[shape]["status"] == boundary.RHO_STAR_MEASURED:
                self.anchors[shape] = search.anchor_rho
                self.anchor_sources[shape] = search.anchor_rule
            self._write_search(shape)
            if search.stopped_reason and search.last_verdict == boundary.VERDICT_VOID:
                raise ladder.CampaignStopped(f"boundary search {self.model}/{shape}: "
                                             f"{search.stopped_reason}")
            if shape not in self.anchors:
                st = self.statuses[shape]
                failures.append(f"{self.model}/{shape}: rho* is {st['status']} (bracket "
                                f"{st['healthy_rho']} - {st['violating_rho']}), not measured")
            print(f"[{self.model}] {shape}: rho* {self.statuses[shape]['status']}, anchor "
                  f"{self.anchors.get(shape)} x predicted boundary "
                  f"({self.capacity(shape):g} rps)", flush=True)
        return failures

    def _write_search(self, shape: str) -> None:
        body = self.searches[shape].as_dict()
        body.update({"mode": MODE, "rho_star_status": self.statuses[shape]["status"],
                     "rho_star_bracket": [self.statuses[shape]["healthy_rho"],
                                          self.statuses[shape]["violating_rho"]],
                     "anchor_rho": self.anchors.get(shape),
                     "anchor_rps": (None if shape not in self.anchors
                                    else round(self.anchors[shape] * self.capacity(shape), 4)),
                     "capacity_rps": self.capacity(shape),
                     "capacity_source": self.capacity_source,
                     "split": design.SPLIT_HOLDOUT,
                     "sealed": "M: these probes are sealed with M and never trained on"})
        (self.out_dir / "boundary" / f"{self.model}_{shape}.json").write_text(
            json.dumps(body, indent=2) + "\n", encoding="utf-8")

    def drive_set(self, order: Sequence[design.DesignCell]) -> None:
        place(self.cells, self.anchors, self.capacities, self.cap, self.bursts,
              items=self.items, spikes=self.spikes)
        for cell in order:
            self.drive_planned(cell)
            self.final_records[cell.cell_id] = self.records[-1]

    def result(self, status: str) -> dict:
        return {"model": self.model, "status": status,
                "rho_star": {s: {"status": (self.statuses.get(s) or {}).get("status"),
                                 "anchor_rho": self.anchors.get(s),
                                 "capacity_rps": self.capacity(s)} for s in NEW_SHAPES},
                "cells": self.outcomes,
                "possibly_contaminated_cells": self.contaminated,
                "attempts": len(self.records)}


def build_plan(args, model: str, cells: Sequence[design.DesignCell],
               order: Sequence[design.DesignCell], prior: Mapping, bursts: Mapping,
               retained: Mapping, labels: Mapping, freeze: Optional[Mapping], cap) -> tuple[dict, dict]:
    provenance = campaign.run_provenance(args)
    preview = []
    capacity = prior["predicted_rps"]
    for cell, item in zip(cells, COMPOSITION):
        unit = PROBE_START_RHO  # the prior; the run replaces it with the measured rho*
        prof = (None if item.kind == KIND_HOLD
                else profile_for(item, unit, capacity[cell.shape], cap, bursts))
        preview.append({**cell.as_dict(), "kind": item.kind,
                        "rps_at_prior": (round(item.factor * unit * capacity[cell.shape], 3)
                                         if item.kind == KIND_HOLD else None),
                        "peak_rps_at_prior": (None if prof is None else round(
                            max(_offered(prof, t) for t in _edges(prof)) * capacity[cell.shape], 3)),
                        "seconds": item.seconds})
    est = estimate(order, args.cooldown_s)
    composition = {
        "evaluated_cells": len(cells) + len(retained["cells"]),
        "retained": [f"{c['shape']} {c['primitive']} {c['cell_id']} (seen before: 2026-09-22 refit)"
                     for c in retained["cells"]],
        "collected": [f"{i.shape} {i.kind}" + (f" {i.factor:g} x rho*" if i.factor else "")
                      + f" {i.seconds:g} s" for i in COMPOSITION],
        "probes": "D6' boundary of MP, MD, G800x240, U512x512 - sealed with M, not evaluated",
    }
    plan = {
        "design": ladder.DESIGN_NAME,
        "mode": MODE,
        "generated_at_utc": campaign.utc_iso(),
        "provenance": provenance,
        "admission_cap": cap.as_dict(),
        "models": [model],
        "design_seed": int(args.design_seed),
        "serial_base": SERIAL_BASE,
        "cell_serial_base": CELL_SERIAL_BASE,
        "cooldown_s": args.cooldown_s,
        "run_manifest": ladder.RUN_MANIFEST,
        # the ten cells; the probes are ledger lines of their own (like a ladder's stage 0)
        "static_cells": {model: [c.as_dict() for c in cells]},
        "order": [c.cell_id for c in order],
        "composition": composition,
        "cells_preview": preview,
        "boundary_prior": prior,
        "bursts": dict(bursts),
        "probe_search": {"start_rho": PROBE_START_RHO, "step": PROBE_STEP,
                         "search_max": PROBE_MAX_RHO, "floor": design.RHO_FLOOR,
                         "bracket_seconds": design.BRACKET_SECONDS,
                         "bisect_rounds": design.BISECT_ROUNDS,
                         "unit": "rho of the shape's predicted D6' boundary rate"},
        "retained": retained,
        "label": {**labels, "window_ms": args.window_ms, "step_ms": args.fit_step_ms},
        "freeze": freeze,
        "estimate": est,
    }
    manifest = {
        "design": ladder.DESIGN_NAME,
        "mode": MODE,
        "written_at_utc": campaign.utc_iso(),
        "why": ("plan 2026-09-21 §6.11 D22 step ④: the acceptance set M, collected after the "
                "freeze, sealed at the end, evaluated once by dline_refit accept"),
        "freeze": freeze,
        "composition": composition,
        "base_run": str(args.base_run),
        "boundary_supplement_run": str(args.boundary_supplement_run),
        "boundary_prior": prior,
        "bursts": dict(bursts),
        "retained": retained,
        "label": labels,
        "design_seed": int(args.design_seed),
        "serial_base": SERIAL_BASE,
        "cell_serial_base": CELL_SERIAL_BASE,
        "seed_derivation": ("as the ladder design (calibration_design.CellFactory); probe order "
                            "from derived_seed(design_seed, model, 'run-order'), cell order from "
                            "derived_seed(design_seed, model, 'acceptance-order'), shapes "
                            "interleaved"),
        "order": [c.cell_id for c in order],
        "static_plan": {model: [c.as_dict() for c in cells]},
        "models": [model],
        "cooldown_floor_s": args.cooldown_s,
        "provenance": provenance,
        "admission_cap": cap.as_dict(),
    }
    return plan, manifest


def _edges(profile):
    return sorted({a for a, _b, _r in profile})


def _offered(profile, t):
    return sum(r for a, b, r in profile if a <= t < b)


def estimate(order: Sequence[design.DesignCell], cooldown_s: float = campaign.DEFAULT_COOLDOWN_S) -> dict:
    """Wall clock: the probes (expected: two bracket probes + two bisections per shape;
    upper: five + two, every probe paying the request deadline) and the ten cells (every
    one pays a gap; the holds above rho* and the ramp may outlive their schedule)."""
    gap_exp = max(cooldown_s, 0.0) + ladder.DRIVER_OVERHEAD_S
    gap_up = max(cooldown_s, design.DRAIN_LIMIT_S) + ladder.DRIVER_OVERHEAD_S
    probes_exp = probes_up = 0.0
    n_exp = n_up = 0
    for shape in NEW_SHAPES:
        tail = design.request_timeout_s(shape)
        e, u = 2 + design.BISECT_ROUNDS, 5 + design.BISECT_ROUNDS
        n_exp += e
        n_up += u
        probes_exp += e * design.BRACKET_SECONDS + 0.5 * e * tail + e * gap_exp
        probes_up += u * design.BRACKET_SECONDS + u * tail + u * gap_up
    load = sum(c.duration_s for c in order)
    tails_exp = sum(design.request_timeout_s(c.shape) for c in order
                    if (c.rho_factor or 0) > 1.0 or c.profile == design.PROFILE_RAMP)
    tails_up = sum(design.request_timeout_s(c.shape) for c in order)
    cells_exp = load + tails_exp + len(order) * gap_exp
    cells_up = load + tails_up + len(order) * gap_up
    return {"probes_expected": n_exp, "probes_upper": n_up,
            "probe_seconds_expected": round(probes_exp, 1), "probe_seconds_upper": round(probes_up, 1),
            "cells": len(order), "cell_load_s": load,
            "cell_seconds_expected": round(cells_exp, 1), "cell_seconds_upper": round(cells_up, 1),
            "seconds_expected": round(probes_exp + cells_exp, 1),
            "seconds_upper": round(probes_up + cells_up, 1)}


def print_plan(plan: dict, model: str) -> None:
    prior = plan["boundary_prior"]
    print(f"{model}: boundary prior {prior['model']}: a={prior['a_s_per_token']:.4g} "
          f"b={prior['b_s_per_token']:.4g} s/token; leave-one-out "
          + ", ".join(f"{x['shape']} {x['loo_error']:+.0%}" for x in prior["leave_one_out"]))
    for shape, rps in prior["predicted_rps"].items():
        print(f"  {shape:9} predicted D6' boundary {rps:.3f} rps = rho 1 (probes start there, "
              f"x{PROBE_STEP:g} steps, ceiling {PROBE_MAX_RHO:g}, floor {design.RHO_FLOOR:g})")
    for shape, body in plan["bursts"].items():
        print(f"  {shape} bursts: {body['burst_requests']} requests per spike "
              f"({body.get('binding_limit')}-bound, running limit {body.get('engine_running_limit')})")
    print(f"  retained (not re-collected; seen once by the 2026-09-22 refit):")
    for c in plan["retained"]["cells"]:
        print(f"    {c['cell_id']:14} M {c['primitive']:6} {c['windows']} windows")
    print(f"  collected, in run order (loads at the prior rho* = 1; the run uses the measured one):")
    by_id = {c["cell_id"]: c for c in plan["cells_preview"]}
    for i, cid in enumerate(plan["order"], start=1):
        c = by_id[cid]
        load = (f"{c['rps_at_prior']:.3f} rps" if c["rps_at_prior"] is not None
                else f"peak {c['peak_rps_at_prior']:.3f} rps")
        print(f"    {i:2d} {cid:22} {c['shape']:9} {c['kind']:6} "
              f"{'' if c['rho_factor'] is None else format(c['rho_factor'], 'g') + ' x rho*':12} "
              f"{c['seconds']:.0f}s  {load}")
    est = plan["estimate"]
    print(f"  probes: {est['probes_expected']}-{est['probes_upper']} x {design.BRACKET_SECONDS:.0f} s; "
          f"cells: {est['cells']} ({est['cell_load_s'] / 60:.0f} min of load)")
    print(f"  wall clock ~{est['seconds_expected'] / 3600:.2f} h expected, <= "
          f"{est['seconds_upper'] / 3600:.2f} h (re-drives of void / inconclusive probes excluded)")
    fr = plan.get("freeze")
    print(f"  freeze: {fr['path'] + ' sha256 ' + fr['sha256'] if fr else 'NOT CHECKED (dry run)'}")


def run_acceptance_set(args, *, drive: Optional[Callable] = None,
                       sample_factory: Optional[Callable[[str], Callable]] = None,
                       sleep: Callable[[float], None] = time.sleep,
                       clock: Callable[[], float] = time.monotonic,
                       check_controller: bool = True) -> int:
    """``--acceptance-set``: see the module docstring."""
    composition = getattr(args, "composition", None) or DEFAULT_COMPOSITION
    if composition not in COMPOSITIONS:
        raise ValueError(f"unknown M composition {composition!r} (one of {list(COMPOSITIONS)})")
    if composition == COMPOSITION_M2:
        return run_acceptance_set_m2(args, drive=drive, sample_factory=sample_factory,
                                     sleep=sleep, clock=clock, check_controller=check_controller)
    if getattr(args, "rho_star_run", None) or getattr(args, "ledger_root", None):
        raise ValueError("--rho-star-run / --ledger-root belong to --composition m2-20261005")
    ladder.check_label_args(args)
    model = training.single_model(args)
    training.check_primary_label(args, model)
    for flag in ("base_run", "boundary_supplement_run", "boundary_table", "retained_dataset"):
        if not getattr(args, flag, None):
            raise ValueError(f"--acceptance-set needs --{flag.replace('_', '-')}")
    out_dir = Path(args.out_dir)
    for source in (args.base_run, args.boundary_supplement_run, args.retained_dataset):
        supplement.check_new_out_dir(out_dir, Path(source))
    freeze = None
    if args.dry_run:
        if getattr(args, "freeze_file", None) and Path(args.freeze_file).is_file():
            freeze = check_freeze(args, model)
        else:
            print(f"WARNING: no freeze file at {getattr(args, 'freeze_file', None)}; the real run "
                  "refuses to start without one (D22)")
    else:
        freeze = check_freeze(args, model)
    labels = label_documents(args, model)
    prior = boundary_prior(model, Path(args.base_run), Path(args.boundary_supplement_run),
                           Path(args.boundary_table))
    cap = training.resolve_cap(args)
    capacity_file = (Path(__file__).resolve().parents[2] / "replayer" / "traces_v2" / "calibration"
                     / "capacity" / f"capacity_{model}.json")
    kv = gen.load_kv_cache_tokens(capacity_file)
    bursts = {i.shape: burst_sizing(i.shape, kv, cap) for i in COMPOSITION if i.kind == KIND_BURSTS}
    retained = retained_cells(model, Path(args.retained_dataset))
    # The retained cells are judged next to the new ones: same load path, or refused.
    from scripts import prompt_corpus as corpus_record

    flags = campaign.mismatch_flags(args)
    if getattr(args, "dry_run", False):  # a dry run reports, as it does for the freeze
        flags = {k: True for k in flags}
    retained["load_path"] = corpus_record.check_load_path(
        retained["load_path"], campaign.load_path(args),
        what=f"retained M cells ({args.retained_dataset})", **flags)
    seed = int(args.design_seed)
    cells = new_cells(model, seed)
    order = interleaved_order(cells, model, seed)
    plan, manifest = build_plan(args, model, cells, order, prior, bursts, retained, labels,
                                freeze, cap)
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
        campaign.require_capture_clock_domains(args)
        campaign.require_prompt_preflight(args, out_dir=out_dir)
    plan["run_manifest_sha256"] = ladder.write_frozen(out_dir / ladder.RUN_MANIFEST, manifest)
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    drive = drive or ladder.subprocess_drive(args, raw_dir)
    sample_factory = sample_factory or (lambda m: ladder.make_engine_sampler(m, args.model_namespace))
    run = AcceptanceRun(
        args, model, prior["predicted_rps"],
        factory=design.CellFactory(model, seed, serial_base=SERIAL_BASE), cap=cap,
        out_dir=out_dir, raw_dir=raw_dir, drive=drive, sample=sample_factory(model),
        capacity_source="d6prime_boundary_prior", sleep=sleep, clock=clock,
        cells=cells, bursts=bursts)
    status, code = "failed", 1
    result = None
    try:
        try:
            failures = run.locate()
            if failures:
                training.banner(failures)
                result = run.result("stopped_rho_star_not_measured")
                result["check_failures"] = failures
                status, code = "stopped_rho_star_not_measured", EXIT_CHECK_FAILED
                return code
            run.drive_set(order)
        except ladder.CampaignStopped as stop:
            print(f"STOPPED: {stop}", flush=True)
            status, code = "stopped", stop.code or 1
            result = run.result(f"stopped: {stop}")
            return code
        sealed = seal(out_dir, raw_dir, model, run, retained, labels, freeze, plan)
        result = run.result("complete")
        result["sealed"] = sealed
        print(f"[{model}] M sealed: {sealed['manifest']} ({sealed['files']} files in "
              f"{M_SHA256SUMS})", flush=True)
        status, code = "complete", 0
        return 0
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        if result is None:
            result = run.result(status)
        training.finish(out_dir, plan, result, status, code,
                        redis_url=getattr(args, "redis_url", None))


def seal(out_dir: Path, raw_dir: Path, model: str, run: AcceptanceRun, retained: Mapping,
         labels: Mapping, freeze: Mapping, plan: Mapping,
         extra: Optional[Mapping] = None) -> dict:
    """Write M_SHA256SUMS and M_manifest.json (both read-only) - before the dataset.
    ``extra``: further top-level manifest keys (M2: composition name, seeds, rho* source);
    an M without retained cells carries no ``retained_source``."""
    collected = [cell_entry(run.final_records[c.cell_id], raw_dir) for c in run.cells]
    by_id = {c.cell_id: c for c in run.cells}
    for entry in collected:
        entry["rho_profile"] = by_id[entry["cell_id"]].rho_profile
    probes = [cell_entry(r, raw_dir) | {"verdict": r.get("verdict")}
              for r in run.records if r["role"] == design.ROLE_BOUNDARY]
    files = sealed_files(out_dir, retained)
    sums_path, sums_sha = write_sha256sums(out_dir, files)
    doc = {
        "what": ("M, the acceptance set of plan §6.11 (D16 / D22): evaluated once by "
                 "`dline_refit accept`; nothing that fits may read it"),
        "model": model,
        "format_revision": MANIFEST_FORMAT_REVISION,
        "written_at_utc": campaign.utc_iso(),
        "freeze": {"path": freeze["path"], "sha256": freeze["sha256"]},
        "label_def": labels["label_def"],
        "label_def_sha256": labels["label_def_sha256"],
        "composition": plan["composition"],
        "rho_star": {s: {"anchor_rho": run.anchors.get(s), "capacity_rps": run.capacity(s),
                         "status": (run.statuses.get(s) or {}).get("status")} for s in NEW_SHAPES},
        "cells": [*collected, *retained["cells"]],
        "sealed_probes": probes,
        "retained_source": {k: retained.get(k) for k in ("dataset", "windows_csv_sha256",
                                                          "manifest_sha256", "run_root",
                                                          "load_path")},
        "run_manifest_sha256": plan.get("run_manifest_sha256"),
        "sha256sums_file": M_SHA256SUMS,
        "sha256sums_sha256": sums_sha,
        "how_to_evaluate": ("dline_refit accept --freeze-file <freeze> --dataset M=<this run's "
                            "dataset> --dataset <retained dataset> --m-manifest <this file> "
                            "(never --h2-dataset)"),
    }
    if not retained.get("cells") and not retained.get("dataset"):
        del doc["retained_source"]
    doc.update(extra or {})
    path = Path(out_dir) / M_MANIFEST
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return {"manifest": str(path), "manifest_sha256": _file_sha256(path),
            "sha256sums": str(sums_path), "sha256sums_sha256": sums_sha, "files": len(files)}


# ------------------------------------------------------------------------- M2 run


def m2_spike_check(model: str, spikes: Mapping[str, Sequence[Mapping]],
                   cells: Sequence[design.DesignCell], rho_star: Mapping, cap) -> dict:
    """Requests per spike (at the reused rho*) against the admission budget a burst may
    claim; refused above it (the spike would be shed, not queued)."""
    by_id = {c.cell_id: c for c in cells}
    peaks = {}
    for cell_id, plan in spikes.items():
        shape = by_id[cell_id].shape
        rps = rho_star["shapes"][shape]["rho_star_rps"]
        peaks[cell_id] = [round(s["height_x_rho_star"] * rps * s["width_s"], 1) for s in plan]
    worst = max((max(v) for v in peaks.values()), default=0.0)
    if worst > cap.burst_request_cap:
        raise ValueError(f"{model}: an M2 spike sends {worst:g} requests, above the "
                         f"{cap.burst_request_cap}-request burst budget of {cap.name}")
    return {"requests_per_spike": peaks, "max_requests_per_spike": worst,
            "burst_request_cap": cap.burst_request_cap}


def m2_estimate(order: Sequence[design.DesignCell], cooldown_s: float = campaign.DEFAULT_COOLDOWN_S,
                drain_limit_s: float = M2_DRAIN_LIMIT_S) -> dict:
    """Wall clock of M2 (no probes). Every cell waits for the drain gate with the cooldown
    as its floor, plus the driver's start-up.

    Expected: a cell that ends above its boundary (a hold above rho*, the ramp's 1.2 x
    hold) outlives its schedule by the client's request deadline; a hold above rho* also
    leaves (f - 1) x its length of rho* work, which the drain gate waits out beyond that
    deadline (capped by the gate's limit; the engine is assumed to serve only rho*, so this
    is long). Every other gap is the cooldown. Upper: every gap at the gate's limit and
    every cell paying the request deadline. Re-drives of void cells are excluded."""
    gap_floor = max(cooldown_s, 0.0)
    expected = upper = 0.0
    by_kind: dict[str, dict] = {}
    for c in order:
        tail = design.request_timeout_s(c.shape)
        factor = c.rho_factor or 0.0
        over = factor > 1.0 or c.profile == design.PROFILE_RAMP
        after = tail if over else 0.0
        backlog = (factor - 1.0) * c.duration_s if factor > 1.0 else 0.0
        drain = min(drain_limit_s, max(0.0, backlog - after))
        e = c.duration_s + after + max(gap_floor, drain) + ladder.DRIVER_OVERHEAD_S
        u = c.duration_s + tail + max(gap_floor, drain_limit_s) + ladder.DRIVER_OVERHEAD_S
        expected += e
        upper += u
        kind = c.profile if c.profile != design.PROFILE_HOLD else f"hold {factor:g}"
        k = by_kind.setdefault(kind, {"cells": 0, "seconds_expected": 0.0, "seconds_upper": 0.0})
        k["cells"] += 1
        k["seconds_expected"] = round(k["seconds_expected"] + e, 1)
        k["seconds_upper"] = round(k["seconds_upper"] + u, 1)
    return {"probes_expected": 0, "probes_upper": 0, "cells": len(order),
            "cell_load_s": sum(c.duration_s for c in order), "drain_limit_s": drain_limit_s,
            "by_kind": by_kind, "seconds_expected": round(expected, 1),
            "seconds_upper": round(upper, 1)}


def m2_composition_doc(cells: Sequence[design.DesignCell]) -> dict:
    kinds: dict[str, int] = {}
    for item in M2_COMPOSITION:
        key = item.kind if item.kind != KIND_HOLD else f"hold {item.factor:g} x rho*"
        kinds[key] = kinds.get(key, 0) + 1
    return {
        "name": COMPOSITION_M2,
        "evaluated_cells": len(cells),
        "retained": [],
        "collected": [f"{i.shape} {i.kind}" + (f" {i.factor:g} x rho*" if i.factor else "")
                      + f" {i.seconds:g} s" for i in M2_COMPOSITION],
        "kinds": kinds,
        "probes": "none: rho* reused from M as design constants",
        "bursts": {"cells": M2_BURST_CELLS, "spikes_per_cell": M2_SPIKES,
                   "heights_x_rho_star": list(M2_SPIKE_HEIGHTS),
                   "height_rule": "each cell: all three heights plus heights[burst_index % 3], "
                                  "order shuffled by derived_seed(design_seed, model, cell_id, "
                                  "'spike-order')",
                   "excess_rho_star_s_per_spike": M2_SPIKE_EXCESS_S,
                   "width_rule": "(height - 1) * width = excess",
                   "base_x_rho_star": M2_BURST_BASE_RHO, "lead_s": M2_BURST_LEAD_S,
                   "period_s": M2_SPIKE_PERIOD_S, "jitter_s": [0.0, M2_SPIKE_JITTER_S],
                   "spike_seed": "derived_seed(design_seed, model, cell_id, 'spike', k): jitters "
                                 "the start; the arrivals come from the cell's arrival seed",
                   "shapes_in_turn": list(M2_SHAPES)},
        "sizing": ("scripts/analysis/m2_power_sim.py: the episode gate's CP bound needs n_eff "
                   ">= 14-17; 16 burst cells keep it there even at a within-cell ICC of 1"),
        "drain_gate": {"rule": "running + waiting == 0 before every cell (P1's state gate)",
                       "limit_s": M2_DRAIN_LIMIT_S},
    }


def build_plan_m2(args, model: str, cells: Sequence[design.DesignCell],
                  order: Sequence[design.DesignCell], spikes: Mapping, rho_star: Mapping,
                  labels: Mapping, freeze: Optional[Mapping], ids: Mapping, spike_check: Mapping,
                  cap) -> tuple[dict, dict]:
    provenance = campaign.run_provenance(args)
    preview = []
    shapes = rho_star["shapes"]
    for cell, item in zip(cells, M2_COMPOSITION):
        unit, cs = shapes[cell.shape]["anchor_rho"], shapes[cell.shape]["capacity_rps"]
        prof = (None if item.kind == KIND_HOLD
                else profile_for(item, unit, cs, cap, {}, spikes.get(cell.cell_id)))
        preview.append({**cell.as_dict(), "kind": item.kind, "seconds": item.seconds,
                        "rps": (round(item.factor * unit * cs, 3) if item.kind == KIND_HOLD else None),
                        "peak_rps": (None if prof is None else round(
                            max(_offered(prof, t) for t in _edges(prof)) * cs, 3)),
                        "spikes": spikes.get(cell.cell_id)})
    est = m2_estimate(order, args.cooldown_s)
    composition = m2_composition_doc(cells)
    seeds = m2_seeds(cells, spikes)
    attribution = label_attribution(labels["label_def"])
    plan = {
        "design": ladder.DESIGN_NAME,
        "mode": MODE,
        "composition_name": COMPOSITION_M2,
        "generated_at_utc": campaign.utc_iso(),
        "provenance": provenance,
        "admission_cap": cap.as_dict(),
        "models": [model],
        "design_seed": int(args.design_seed),
        "cell_serial_base": M2_CELL_SERIAL_BASE,
        "cooldown_s": args.cooldown_s,
        "drain_limit_s": M2_DRAIN_LIMIT_S,
        "run_manifest": ladder.RUN_MANIFEST,
        "static_cells": {model: [c.as_dict() for c in cells]},
        "order": [c.cell_id for c in order],
        "composition": composition,
        "cells_preview": preview,
        "rho_star_source": rho_star,
        "spikes": dict(spikes),
        "spike_check": spike_check,
        "seeds": seeds,
        "id_check": ids,
        "retained": {"cells": []},
        "label": {**labels, "attribution": attribution, "window_ms": args.window_ms,
                  "step_ms": args.fit_step_ms},
        "freeze": freeze,
        "estimate": est,
    }
    manifest = {
        "design": ladder.DESIGN_NAME,
        "mode": MODE,
        "composition_name": COMPOSITION_M2,
        "written_at_utc": campaign.utc_iso(),
        "why": ("next round's M2 (design 2026-10-05 item 5, user decision 3): collected after "
                "the freeze, sealed at the end, evaluated once by dline_refit accept"),
        "freeze": freeze,
        "composition": composition,
        "rho_star_source": rho_star,
        "spikes": dict(spikes),
        "seeds": seeds,
        "id_check": ids,
        "label": labels,
        "label_attribution": attribution,
        "design_seed": int(args.design_seed),
        "cell_serial_base": M2_CELL_SERIAL_BASE,
        "seed_derivation": ("calibration_design.CellFactory (arrival seed and prompt key per cell "
                            "id); spike seeds derived_seed(design_seed, model, cell_id, 'spike', k); "
                            "cell order from derived_seed(design_seed, model, 'acceptance-order'), "
                            "shapes interleaved"),
        "order": [c.cell_id for c in order],
        "static_plan": {model: [c.as_dict() for c in cells]},
        "models": [model],
        "cooldown_floor_s": args.cooldown_s,
        "drain_limit_s": M2_DRAIN_LIMIT_S,
        "provenance": provenance,
        "admission_cap": cap.as_dict(),
    }
    return plan, manifest


def print_plan_m2(plan: dict, model: str) -> None:
    src = plan["rho_star_source"]
    print(f"{model}: M2 ({COMPOSITION_M2}) - rho* reused from {src['run_root']} (no probes):")
    for shape, body in src["shapes"].items():
        print(f"  {shape:9} rho* {body['anchor_rho']:g} x {body['capacity_rps']:g} rps = "
              f"{body['rho_star_rps']:.3f} rps")
    comp = plan["composition"]
    print(f"  cells: {comp['evaluated_cells']} ("
          + ", ".join(f"{n} {k}" for k, n in comp["kinds"].items()) + "), no retained cells")
    print(f"  label: {plan['label']['label_def'].get('name')} (attribution "
          f"{plan['label']['attribution']}, sha256 {plan['label']['label_def_sha256'][:12]})")
    ids = plan["id_check"]
    print(f"  ids / seeds: {ids['m2_codes']} codes and {ids['m2_seeds']} seeds (3 models), none in "
          f"{ids['ledgers_scanned']} ledgers under {ids['roots']}")
    print(f"  spikes: at most {plan['spike_check']['max_requests_per_spike']:g} requests per spike "
          f"(burst budget {plan['spike_check']['burst_request_cap']})")
    print("  in run order:")
    by_id = {c["cell_id"]: c for c in plan["cells_preview"]}
    for i, cid in enumerate(plan["order"], start=1):
        c = by_id[cid]
        load = (f"{c['rps']:.3f} rps" if c["rps"] is not None else f"peak {c['peak_rps']:.3f} rps")
        extra = ""
        if c.get("spikes"):
            extra = " spikes " + " ".join(
                f"{s['height_x_rho_star']:g}x@{s['start_s']:g}s/{s['width_s']:g}s" for s in c["spikes"])
        print(f"    {i:2d} {cid:22} {c['shape']:9} {c['kind']:6} "
              f"{'' if c['rho_factor'] is None else format(c['rho_factor'], 'g') + ' x rho*':12} "
              f"{c['seconds']:.0f}s  {load}{extra}")
    est = plan["estimate"]
    print(f"  cells: {est['cells']} ({est['cell_load_s'] / 60:.0f} min of load), drain gate <= "
          f"{est['drain_limit_s']:g} s before each")
    print(f"  wall clock ~{est['seconds_expected'] / 3600:.2f} h expected, <= "
          f"{est['seconds_upper'] / 3600:.2f} h (every gap at the drain limit; re-drives of void "
          "cells excluded)")
    fr = plan.get("freeze")
    print(f"  freeze: {fr['path'] + ' sha256 ' + fr['sha256'] if fr else 'NOT CHECKED (dry run)'}")


def run_acceptance_set_m2(args, *, drive: Optional[Callable] = None,
                          sample_factory: Optional[Callable[[str], Callable]] = None,
                          sleep: Callable[[float], None] = time.sleep,
                          clock: Callable[[], float] = time.monotonic,
                          check_controller: bool = True) -> int:
    """``--acceptance-set --composition m2-20261005``: see the module docstring."""
    ladder.check_label_args(args)
    model = training.single_model(args)
    for flag in ("base_run", "boundary_supplement_run", "boundary_table", "retained_dataset"):
        if getattr(args, flag, None):
            raise ValueError(f"--{flag.replace('_', '-')} belongs to --composition {COMPOSITION_M1} "
                             "(M2 has no probes and no retained cells)")
    rho_star = m2_rho_star(model, getattr(args, "rho_star_run", None))
    out_dir = Path(args.out_dir)
    supplement.check_new_out_dir(out_dir, Path(args.rho_star_run))
    freeze = None
    if args.dry_run and not (getattr(args, "freeze_file", None) and Path(args.freeze_file).is_file()):
        print(f"WARNING: no freeze file at {getattr(args, 'freeze_file', None)}; the real run "
              "refuses to start without one (D22)")
        resolve_attribution(args, model, None, M2_DEFAULT_ATTRIBUTION)
    else:
        freeze = check_freeze(args, model)
    training.check_primary_label(args, model)
    labels = label_documents(args, model)
    cap = training.resolve_cap(args)
    seed = int(args.design_seed)
    per_model = {m: m2_cells(m, seed) for m in M2_RHO_STAR}
    cells, spikes = per_model[model]
    roots = [Path(args.rho_star_run).resolve().parent,
             *(Path(r) for r in (getattr(args, "ledger_root", None) or []))]
    ids = check_new_ids({m: v[0] for m, v in per_model.items()},
                        {m: v[1] for m, v in per_model.items()}, roots, skip=[out_dir])
    if ids["problems"]:
        raise ValueError(f"M2 ids / seeds are not new: {ids['problems']}")
    spike_check = m2_spike_check(model, spikes, cells, rho_star, cap)
    order = interleaved_order(cells, model, seed)
    plan, manifest = build_plan_m2(args, model, cells, order, spikes, rho_star, labels, freeze,
                                   ids, spike_check, cap)
    out_dir.mkdir(parents=True, exist_ok=True)
    print_plan_m2(plan, model)
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
        campaign.require_capture_clock_domains(args)
        campaign.require_prompt_preflight(args, out_dir=out_dir)
    plan["run_manifest_sha256"] = ladder.write_frozen(out_dir / ladder.RUN_MANIFEST, manifest)
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    drive = drive or ladder.subprocess_drive(args, raw_dir)
    sample_factory = sample_factory or (lambda m: ladder.make_engine_sampler(m, args.model_namespace))
    capacities = {s: b["capacity_rps"] for s, b in rho_star["shapes"].items()}
    run = AcceptanceRun(
        args, model, capacities,
        factory=design.CellFactory(model, seed, serial_base=M2_CELL_SERIAL_BASE), cap=cap,
        out_dir=out_dir, raw_dir=raw_dir, drive=drive, sample=sample_factory(model),
        capacity_source="m_rho_star_reuse", sleep=sleep, clock=clock,
        cells=cells, bursts={}, items=M2_COMPOSITION, spikes=spikes)
    run.drain_limit_s = M2_DRAIN_LIMIT_S
    run.reuse_rho_star(rho_star)
    status, code = "failed", 1
    result = None
    try:
        try:
            run.drive_set(order)
        except ladder.CampaignStopped as stop:
            print(f"STOPPED: {stop}", flush=True)
            status, code = "stopped", stop.code or 1
            result = run.result(f"stopped: {stop}")
            return code
        extra = {"composition_name": COMPOSITION_M2,
                 "label_attribution": label_attribution(labels["label_def"]),
                 "design_seed": seed, "seeds": plan["seeds"], "spikes": plan["spikes"],
                 "rho_star_source": rho_star, "id_check": ids,
                 "how_to_evaluate": ("dline_refit accept --freeze-file <freeze> --dataset "
                                     "M2=<this run's dataset of the frozen label's attribution> "
                                     "--m-manifest <this file> (never --h2-dataset)")}
        sealed = seal(out_dir, raw_dir, model, run, {"cells": []}, labels, freeze, plan, extra)
        result = run.result("complete")
        result["sealed"] = sealed
        print(f"[{model}] M2 sealed: {sealed['manifest']} ({sealed['files']} files in "
              f"{M_SHA256SUMS})", flush=True)
        status, code = "complete", 0
        return 0
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        if result is None:
            result = run.result(status)
        training.finish(out_dir, plan, result, status, code,
                        redis_url=getattr(args, "redis_url", None))

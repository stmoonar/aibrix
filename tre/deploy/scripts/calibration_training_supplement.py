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

Training plans (``--training-plan``, :data:`TRAINING_PLANS`)
------------------------------------------------------------
The cells above are the plan ``legacy-20260923`` (the default, kept so that run can be
re-planned as it was). ``p1-deep-overload`` (2026-10-03, the vLLM 0.30 re-calibration) is
the P1 shape supplement: per model S2 / S3 / T8 / S4 / S5 x {1.5, 2.0, 3.0} x rho*_run2,
one 150 s hold each (15 cells), plus the same three S2 sentinels. They are training cells
(role ``ladder``, stage ``ladder``: D16 holds train) deep in the violated region, so:

* no both-sides check - every P1 cell is meant to violate. Its check is that each P1 cell
  was driven and judged on at least one labelled post-warm-up window (not void, not
  inconclusive for lack of windows);
* no boundary supplement is needed (``--boundary-supplement-run`` optional; when given it
  is load-path checked and recorded like the base run);
* the drain before each cell waits for an idle engine (``running + waiting == 0``) for up
  to :data:`P1_DRAIN_LIMIT_S` instead of ``DRAIN_LIMIT_S``. A 3 x rho* cell leaves a
  waiting queue of roughly ``2 x lambda* x`` the client timeout (up to 112 s for S4) that
  vLLM serves for minutes if the client's aborts do not reach it. The gate is state-based:
  it ends as soon as the engine is idle, so it costs nothing when aborts propagate;
* the plan reports each cell's in-flight estimate (engine running limit + the excess
  arrivals of one client timeout) against the lower of the client's connection limit and
  Envoy's shed ceiling: past either the cell would be void (dispatch delay / shed).

The 2026-11-03 supplement (local workspace ``docs/calib-supplement-design-20261103.md``,
blocks A / B / E; preregistration ``prereg-calib-supplement-20261103.md``) is three plans,
run in this order per model, the three models in parallel
(``calibration/run_chain_supp.sh``). Every cell is one hold, first 60 s warm-up; no
sentinels, no checks (every cell is kept whatever its verdict: the void rule still re-runs
a void cell once and stops the run on a second void); the drain gate waits up to
:data:`P1_DRAIN_LIMIT_S`; a cell at >= :data:`SUPP_OVERLOAD_FACTOR` x its unit is void only
past :data:`stream_cut.RUNTIME_MODEL_ERROR_LIMIT` model errors (0.10: the route's 150 s
cut counts as a model error), every other at the run's ``--max-model-error-rate``:

* ``supp-a-20261103`` (A, training): the seven training shapes x {0.3, 0.4, 0.5, 0.6} x
  rho*_run2, 240 s - 28 cells. Low load, so that tau_high's n = 3 rule interpolates.
* ``supp-b-20261103`` (B, training): G384x64 / G1024x64 / G384x640 / G1024x640
  (``gen.SUPPLEMENT_GRID_SHAPES``: shapes no training cell used, split train) x {0.4, 0.55,
  0.7, 0.85, 0.95, 1.05, 1.15, 1.3} x C^, 240 s, 300 s at 0.85-1.15 - 32 cells.
* ``supp-e-20261103`` (E, held out): the mixtures ES / EL (``gen.VALIDATION_SHAPES``, split
  holdout, role acceptance) x {0.8, 1.0, 1.2} x C^, 300 s - 6 cells. Sealed: a real run
  needs ``--sealed-plan``, the ``plan.json`` of its dry run with a ``<file>.sha256`` sidecar
  written before the collection, and refuses unless its cells, order, capacities and label
  are the sealed ones (:data:`SEALED_PLAN_KEYS`); right after its last cell it writes
  :data:`E_SHA256SUMS` and :data:`E_MANIFEST` (read-only), then the dataset. A run that
  stops is never sealed: re-run E into a new out-dir.

C^ (B, E) is the capacity-model prior :func:`capacity_prior`, no boundary probe: ``1/r = i/P
+ o/D`` through the seven training shapes' measured boundaries rho*_run2 x C_s; a mixture's
streams combine by request share. A cell's capacity is C^ and its rho the factor, so it
offers ``factor x C^`` rps (``capacity_source`` :data:`SUPP_CAPACITY_SOURCE`).
"""
from __future__ import annotations

import json
import math
import os
import random
import stat
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
from scripts import gen_calibration_schedules as gen
from scripts import openloop
from scripts import stream_cut

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
UNIT_PRIOR = ("C^: the capacity prior 1/r = i/P + o/D through the seven training shapes' "
              "rho*_run2 x C_s (capacity_prior)")

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


@dataclass(frozen=True)
class TrainingPlan:
    """One selectable cell plan of this collection (``--training-plan``)."""

    name: str
    groups: Mapping[str, tuple[Group, ...]]
    #: Parts whose shapes must each have a healthy and a violated cell.
    both_sides_parts: tuple[str, ...]
    #: Parts whose every cell must merely be driven and judged on >= 1 labelled window.
    driven_parts: tuple[str, ...]
    serial_base: int
    drain_limit_s: float
    why: str
    #: The three S2 drift sentinels (first / middle / last of the run).
    sentinels: bool = True
    #: The hold cells' role: ``ladder`` trains (D16); a held-out set's cells are
    #: ``acceptance`` (their split is holdout because their shape is held out).
    role: str = design.ROLE_LADDER
    capacity_source: str = "rho_priors"
    #: Cells at or above this factor get the stream-cut limit as their own void limit
    #: (:meth:`max_model_error_rate`); None = every cell at the run's.
    overload_factor: Optional[float] = None
    #: A sealed held-out set: a real run needs ``--sealed-plan`` and is sealed at its end.
    sealed: bool = False

    def needs_supplement(self, model: str) -> bool:
        return any(g.unit == UNIT_S3_D6PRIME for g in self.groups[model])

    def max_model_error_rate(self, factor: float) -> Optional[float]:
        """A cell's own model-error void limit, None = the run's ``--max-model-error-rate``.
        An overload cell's route-timeout cuts are model errors (``scripts.stream_cut``)."""
        if self.overload_factor is None or factor < self.overload_factor:
            return None
        return stream_cut.RUNTIME_MODEL_ERROR_LIMIT


PLAN_LEGACY = "legacy-20260923"
PLAN_P1 = "p1-deep-overload"
P1_SHAPES = ("S2", "S3", "T8", "S4", "S5")
P1_FACTORS = (1.5, 2.0, 3.0)
P1_SECONDS = 150.0
#: Drain gate before each P1 cell: > 2 x the longest client timeout of the P1 shapes
#: (S4: 112 s), the backlog a 3 x cell leaves when the client's aborts do not reach vLLM.
P1_DRAIN_LIMIT_S = 300.0
#: Clear of the legacy plan's 60_001+, M's 70_000 / 70_500 and T14's 80_500.
P1_SERIAL_BASE = 62_000
_P1_WHY = ("P1 deep overload: violated-side training windows far past the boundary "
           "(the D16 training set had none above 1.3 x rho*)")
P1_PLAN: dict[str, tuple[Group, ...]] = {
    model: tuple(Group("p1", shape, UNIT_RUN2, P1_FACTORS, 1, P1_SECONDS, _P1_WHY)
                 for shape in P1_SHAPES)
    for model in PLAN
}

PLAN_SUPP_A = "supp-a-20261103"
PLAN_SUPP_B = "supp-b-20261103"
PLAN_SUPP_E = "supp-e-20261103"
SUPP_A_FACTORS = (0.3, 0.4, 0.5, 0.6)
#: B's ladder; the four around the knee hold 300 s, the others 240 s.
SUPP_B_FACTORS = (0.4, 0.55, 0.7, 1.3)
SUPP_B_KNEE_FACTORS = (0.85, 0.95, 1.05, 1.15)
SUPP_E_FACTORS = (0.8, 1.0, 1.2)
SUPP_SECONDS = 240.0
SUPP_LONG_SECONDS = 300.0
#: Cells at or above this factor are void only past the stream-cut limit (0.10).
SUPP_OVERLOAD_FACTOR = 1.15
#: Clear of every earlier collection's serials (supplement 50_001+, legacy 60_001+, P1
#: 62_001+, M 70_000 / 70_500, M2 74_000+, T14 80_500 / 81_000).
SUPP_SERIAL_BASE = {PLAN_SUPP_A: 90_000, PLAN_SUPP_B: 91_000, PLAN_SUPP_E: 92_000}
SUPP_CAPACITY_SOURCE = "supp_capacity_prior"
_SUPP_A_WHY = "A: low load, so that tau_high's n = 3 rule interpolates (needs ~0.3-0.7 rho*)"
_SUPP_B_WHY = "B: input {384, 1024} x output {64, 640}, does theta carry to short / long outputs"
_SUPP_E_WHY = "E: sealed held-out mixtures, accepts theta on loads no fit has seen"
SUPP_A_PLAN: dict[str, tuple[Group, ...]] = {
    model: tuple(Group("a", shape, UNIT_RUN2, SUPP_A_FACTORS, 1, SUPP_SECONDS, _SUPP_A_WHY)
                 for shape in gen.TRAINING_SHAPES)
    for model in PLAN
}
SUPP_B_PLAN: dict[str, tuple[Group, ...]] = {
    model: tuple(g for shape in gen.SUPPLEMENT_GRID_SHAPES for g in (
        Group("b", shape, UNIT_PRIOR, SUPP_B_FACTORS, 1, SUPP_SECONDS, _SUPP_B_WHY),
        Group("b", shape, UNIT_PRIOR, SUPP_B_KNEE_FACTORS, 1, SUPP_LONG_SECONDS, _SUPP_B_WHY)))
    for model in PLAN
}
SUPP_E_PLAN: dict[str, tuple[Group, ...]] = {
    model: tuple(Group("e", shape, UNIT_PRIOR, SUPP_E_FACTORS, 1, SUPP_LONG_SECONDS, _SUPP_E_WHY)
                 for shape in gen.VALIDATION_SHAPES)
    for model in PLAN
}
_SUPP = dict(sentinels=False, overload_factor=SUPP_OVERLOAD_FACTOR)

TRAINING_PLANS: dict[str, TrainingPlan] = {
    PLAN_LEGACY: TrainingPlan(
        PLAN_LEGACY, PLAN, BOTH_SIDES_PARTS, (), SERIAL_BASE, design.DRAIN_LIMIT_S,
        "plan 2026-09-21 §6.11 D16-D22 step ③: constant-load training cells on the D6' "
        "boundaries (③a S3 ladder, ③b 14b CI, 7b T8 top-up, ③d sentinels)"),
    PLAN_P1: TrainingPlan(
        PLAN_P1, P1_PLAN, (), ("p1",), P1_SERIAL_BASE, P1_DRAIN_LIMIT_S,
        "2026-10-03 re-calibration on vLLM 0.30, shape supplement P1: S2/S3/T8/S4/S5 x "
        "{1.5, 2.0, 3.0} x rho*_run2, 150 s holds, training (D16), plus ③d sentinels"),
    PLAN_SUPP_A: TrainingPlan(
        PLAN_SUPP_A, SUPP_A_PLAN, (), (), SUPP_SERIAL_BASE[PLAN_SUPP_A], P1_DRAIN_LIMIT_S,
        "2026-11-03 supplement A: S1-S5/T8/T9 x {0.3, 0.4, 0.5, 0.6} x rho*_run2, 240 s, "
        "training", **_SUPP),
    PLAN_SUPP_B: TrainingPlan(
        PLAN_SUPP_B, SUPP_B_PLAN, (), (), SUPP_SERIAL_BASE[PLAN_SUPP_B], P1_DRAIN_LIMIT_S,
        "2026-11-03 supplement B: G384x64/G1024x64/G384x640/G1024x640 x {0.4 .. 1.3} x C^, "
        "240 s (300 s at 0.85-1.15), training", capacity_source=SUPP_CAPACITY_SOURCE, **_SUPP),
    PLAN_SUPP_E: TrainingPlan(
        PLAN_SUPP_E, SUPP_E_PLAN, (), (), SUPP_SERIAL_BASE[PLAN_SUPP_E], P1_DRAIN_LIMIT_S,
        "2026-11-03 supplement E: mixtures ES/EL x {0.8, 1.0, 1.2} x C^_mix, 300 s, sealed "
        "held-out validation (never trains)", role=design.ROLE_ACCEPTANCE,
        capacity_source=SUPP_CAPACITY_SOURCE, sealed=True, **_SUPP),
}
DEFAULT_TRAINING_PLAN = PLAN_LEGACY


def training_plan(args) -> TrainingPlan:
    name = getattr(args, "training_plan", None) or DEFAULT_TRAINING_PLAN
    if name not in TRAINING_PLANS:
        raise ValueError(f"unknown --training-plan {name!r} (expected one of {sorted(TRAINING_PLANS)})")
    return TRAINING_PLANS[name]

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


#: The points of the capacity prior: every training shape at its measured boundary.
PRIOR_SHAPES = gen.TRAINING_SHAPES


def _mean_streams(shape: str) -> list[tuple[float, float, float]]:
    return [(w, gen._length_mean(i), gen._length_mean(o)) for w, i, o in gen.shape_components(shape)]


def capacity_prior(model: str, bases: Mapping[str, supplement.BaseAnchor],
                   shapes: Sequence[str]) -> dict:
    """C^ (rps) of each of ``shapes``: ``1/r = i/P + o/D`` (``gen.fit_capacity_model``, least
    squares through the origin) through the :data:`PRIOR_SHAPES`' boundary rates rho*_run2
    x C_s at their mean lengths; a mixture's streams combine by request share
    (``CapacityModel.mixture_rps``). rho*_run2 is a measured boundary, so a factor of 1 is
    about the knee, as with M's and T14's priors. Leave-one-shape-out errors say how far
    the fit carries."""
    points = []
    for shape in PRIOR_SHAPES:
        (_w, i, o), = _mean_streams(shape)
        points.append((i, o, bases[shape].anchor_rps))
    fit = gen.fit_capacity_model(model, points)
    loo = []
    for k, (i, o, rate) in enumerate(points):
        rest = gen.fit_capacity_model(model, points[:k] + points[k + 1:])
        loo.append({"shape": PRIOR_SHAPES[k], "rate_rps": round(rate, 4),
                    "loo_rps": round(rest.rps(i, o), 4), "loo_error": round(rest.rps(i, o) / rate - 1.0, 4)})
    return {
        "form": f"1/r = i/P + o/D, least squares through the origin over {list(PRIOR_SHAPES)} at "
                "rho*_run2 x C_s; a mixture's streams combined by request share",
        "prefill_tokens_per_s": round(fit.prefill_tps, 3),
        "decode_tokens_per_s": round(fit.decode_tps, 3),
        "rms_relative_error": round(fit.rms_rel_error, 4),
        "points": [{"shape": s, "input_mean": round(i, 3), "output_mean": round(o, 3),
                    "rate_rps": round(r, 6)} for s, (i, o, r) in zip(PRIOR_SHAPES, points)],
        "leave_one_out": loo,
        "predicted_rps": {s: round(fit.mixture_rps(_mean_streams(s)), 6) for s in shapes},
    }


def resolve_units(model: str, base_root: Path, supp_root: Optional[Path],
                  plan: Optional[TrainingPlan] = None) -> dict:
    """Per (shape, unit) the rho (of the shape's run-2 C_s, or of C^ for a prior unit) a
    factor of 1 means, with where it came from; plus each shape's capacity."""
    tplan = plan or TRAINING_PLANS[DEFAULT_TRAINING_PLAN]
    groups = tplan.groups[model]
    prior_shapes = sorted({g.shape for g in groups if g.unit == UNIT_PRIOR})
    shapes = {g.shape for g in groups if g.unit != UNIT_PRIOR}
    shapes |= {SENTINEL_SHAPE} if tplan.sentinels else set()
    shapes |= set(PRIOR_SHAPES) if prior_shapes else set()
    bases = {s: supplement.load_base_anchor(base_root, model, s) for s in sorted(shapes)}
    capacity = {s: b.capacity_rps for s, b in bases.items()}
    base_docs = {s: b.as_dict() for s, b in bases.items()}
    prior = capacity_prior(model, bases, prior_shapes) if prior_shapes else None
    units: dict[tuple[str, str], dict] = {}
    for group in groups:
        if group.unit == UNIT_PRIOR:
            c_hat = prior["predicted_rps"][group.shape]
            capacity[group.shape] = c_hat
            # anchor 1 = C^: what the in-flight estimate counts the excess arrivals from
            base_docs[group.shape] = {"model": model, "shape": group.shape, "anchor_rho": 1.0,
                                      "capacity_rps": c_hat, "anchor_rule": UNIT_PRIOR}
            units[(group.shape, group.unit)] = {"rho": 1.0, "source": {"capacity_rps": c_hat,
                                                                       "rule": UNIT_PRIOR}}
            continue
        base = bases[group.shape]
        if group.unit == UNIT_S3_D6PRIME:
            if supp_root is None:
                raise ValueError(f"{model}: {group.unit} needs --boundary-supplement-run")
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
    if tplan.sentinels:
        sentinel = bases[SENTINEL_SHAPE]
        units[(SENTINEL_SHAPE, UNIT_SENTINEL)] = {"rho": sentinel.anchor_rho,
                                                  "source": sentinel.as_dict()}
    return {"units": units, "capacity": capacity, "bases": base_docs, "prior": prior}


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
                design_seed: int, plan: Optional[TrainingPlan] = None,
                ) -> tuple[list[design.DesignCell], list[design.DesignCell]]:
    """(the run's cell sequence, the sentinels). Ids and seeds are fixed here."""
    tplan = plan or TRAINING_PLANS[DEFAULT_TRAINING_PLAN]
    units = resolved["units"]
    holds = []
    for group in tplan.groups[model]:
        unit_rho = units[(group.shape, group.unit)]["rho"]
        for factor in group.factors:
            for rep in range(1, group.replicates + 1):
                holds.append(factory.new(
                    group.shape, tplan.role, group.seconds, rho_factor=factor,
                    rho=round(factor * unit_rho, 6), replicate=rep,
                    max_model_error_rate=tplan.max_model_error_rate(factor),
                    note=f"part {group.part}: {factor:g} x {group.unit}"))
    if not tplan.sentinels:
        rng = random.Random(design.derived_seed(design_seed, model, "training-supplement-order"))
        return interleave(holds, rng), []
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


def estimate(sequence: Sequence[design.DesignCell], cooldown_s: float,
             drain_limit_s: float = design.DRAIN_LIMIT_S) -> dict:
    """Wall clock: every cell pays a gap (expected: the cooldown, upper: the drain limit;
    plus the driver's start-up); a cell above its boundary can outlive its schedule by the
    request deadline while its backlog finishes (expected: the cells at >= 1.05 x, upper:
    all of them)."""
    gap_exp = max(cooldown_s, 0.0) + ladder.DRIVER_OVERHEAD_S
    gap_up = max(cooldown_s, drain_limit_s) + ladder.DRIVER_OVERHEAD_S
    load = sum(c.duration_s for c in sequence)
    tails_exp = sum(design.request_timeout_s(c.shape) for c in sequence
                    if c.role != design.ROLE_SENTINEL and (c.rho_factor or 0) >= 1.05)
    tails_up = sum(design.request_timeout_s(c.shape) for c in sequence)
    return {"cells": len(sequence), "offered_load_s": load,
            "seconds_expected": round(load + tails_exp + len(sequence) * gap_exp, 1),
            "seconds_upper": round(load + tails_up + len(sequence) * gap_up, 1)}


#: An estimate past this share of the in-flight limit is reported as at risk.
IN_FLIGHT_WARN_SHARE = 0.8


def in_flight_estimate(sequence: Sequence[design.DesignCell], resolved: Mapping, cap) -> dict:
    """Peak in-flight requests of every cell above its shape's rho*_run2: the engine's
    running limit plus the arrivals beyond rho*_run2 x C_s during one client timeout (a
    request still waiting then is abandoned by the client; an engine serving faster than
    rho* only lowers it). Against the lower of the client's connection limit
    (``openloop.DEFAULT_MAX_IN_FLIGHT``: past it the sender pool waits -> dispatch-delay
    void) and Envoy's shed ceiling (past it Envoy sheds -> shed void)."""
    limit = min(int(openloop.DEFAULT_MAX_IN_FLIGHT), int(cap.shed_ceiling))
    cells = []
    for cell in sequence:
        base = resolved["bases"].get(cell.shape)
        if cell.rho is None or base is None:
            continue
        excess_rps = (float(cell.rho) - float(base["anchor_rho"])) * float(base["capacity_rps"])
        if excess_rps <= 0:
            continue
        seconds = min(design.request_timeout_s(cell.shape), float(cell.duration_s))
        peak = cap.fleet_sequence_limit + excess_rps * seconds
        cells.append({"cell_id": cell.cell_id, "shape": cell.shape, "rho_factor": cell.rho_factor,
                      "in_flight": int(round(peak)),
                      "at_risk": peak > IN_FLIGHT_WARN_SHARE * limit})
    return {"limit": limit, "warn_share": IN_FLIGHT_WARN_SHARE, "cells": cells,
            "at_risk": [c["cell_id"] for c in cells if c["at_risk"]],
            "rule": ("running limit + (rho - rho*_run2) x C_s x min(client timeout, cell "
                     "seconds), vs min(client max in-flight, Envoy shed ceiling)")}


def label_doc(label: slo_labels.LabelDefinition, args) -> dict:
    return {
        "ttft_slo_mode": label.ttft_slo_mode, "ttft_slowdown_k": label.ttft_slowdown_k,
        "ttft_floor_ms": label.ttft_floor_ms, "ttft_idle_c_ms": label.ttft_idle_c_ms,
        "ttft_idle_b_ms_per_token": label.ttft_idle_b_ms_per_token,
        "tpot_p95_ms": label.tpot_p95_ms, "min_completed_requests": label.min_completed_requests,
        "window_ms": args.window_ms, "step_ms": args.fit_step_ms,
        "window_align": getattr(args, "fit_window_align", "grid"),
    }


def checks_text(tplan: TrainingPlan) -> str:
    parts = []
    if tplan.both_sides_parts:
        parts.append(f"both sides: every part {'/'.join(tplan.both_sides_parts)} shape has >= 1 "
                     "healthy and >= 1 violated hold cell")
    if tplan.driven_parts:
        parts.append(f"driven: every part {'/'.join(tplan.driven_parts)} cell has a healthy or "
                     "violated verdict on >= 1 labelled post-warm-up window")
    if tplan.sentinels:
        parts.append("sentinel drift not flagged")
    if not parts:
        return ("none: every cell is kept whatever its verdict (a void cell is re-run once, a "
                "second void stops the run)")
    return "; ".join(parts) + f". A failure exits {EXIT_CHECK_FAILED} (D22)"


def overload_doc(tplan: TrainingPlan) -> Optional[dict]:
    """The per-cell void limit rule of a plan with overload cells (plan + manifest)."""
    if tplan.overload_factor is None:
        return None
    return {"cells_at_or_above_factor": tplan.overload_factor,
            "rule": "those cells are void past the stream-cut limit, every other at the run's "
                    "--max-model-error-rate",
            "stream_cut": stream_cut.record()}


def build_plan(args, model: str, sequence: Sequence[design.DesignCell], resolved: Mapping,
               cap, tplan: Optional[TrainingPlan] = None) -> tuple[dict, dict]:
    """(plan.json, run manifest)."""
    tplan = tplan or TRAINING_PLANS[DEFAULT_TRAINING_PLAN]
    label = campaign.primary_label(args, model)
    provenance = campaign.run_provenance(args)
    units_doc = [{"shape": s, "unit": u, "rho": v["rho"], "source": v["source"]}
                 for (s, u), v in sorted(resolved["units"].items())]
    groups = [{"part": g.part, "shape": g.shape, "unit": g.unit, "factors": list(g.factors),
               "replicates": g.replicates, "seconds": g.seconds, "why": g.why}
              for g in tplan.groups[model]]
    cells = [c.as_dict() for c in sequence]
    est = estimate(sequence, args.cooldown_s, tplan.drain_limit_s)
    supp = getattr(args, "boundary_supplement_run", None)
    plan = {
        # "ladder": the standard dataset reads this run from its ledger like any ladder run.
        "design": ladder.DESIGN_NAME,
        "mode": MODE,
        "training_plan": tplan.name,
        "generated_at_utc": campaign.utc_iso(),
        "provenance": provenance,
        "admission_cap": cap.as_dict(),
        "models": [model],
        "design_seed": int(args.design_seed),
        "serial_base": tplan.serial_base,
        "cooldown_s": args.cooldown_s,
        "drain_limit_s": tplan.drain_limit_s,
        "run_manifest": ladder.RUN_MANIFEST,
        "static_cells": {model: cells},
        "sequence": [c.cell_id for c in sequence],
        "groups": groups,
        "units": units_doc,
        "capacity_rps": resolved["capacity"],
        "capacity_prior": resolved.get("prior"),
        "overload": overload_doc(tplan),
        "label": label_doc(label, args),
        "estimate": est,
        "in_flight": in_flight_estimate(sequence, resolved, cap),
    }
    manifest = {
        "design": ladder.DESIGN_NAME,
        "mode": MODE,
        "training_plan": tplan.name,
        "written_at_utc": campaign.utc_iso(),
        "why": tplan.why,
        "base_run": str(args.base_run),
        "boundary_supplement_run": None if supp is None else str(supp),
        "bases": resolved["bases"],
        "units": units_doc,
        "capacity_prior": resolved.get("prior"),
        "overload": overload_doc(tplan),
        "groups": groups,
        "sentinel_rule": (f"{SENTINEL_FACTOR} x rho*_run2({SENTINEL_SHAPE}), {SENTINEL_SECONDS:g} s, "
                          f"positions {list(SENTINEL_POSITIONS)}; drift on "
                          f"{design.SENTINEL_DRIFT_THRESHOLDS}") if tplan.sentinels else "none",
        "checks": checks_text(tplan),
        "drain_limit_s": tplan.drain_limit_s,
        "label": provenance.get("label"),
        "design_seed": int(args.design_seed),
        "serial_base": tplan.serial_base,
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
    prior = plan.get("capacity_prior")
    if prior:
        loo = ", ".join(f"{p['shape']} {p['loo_error']:+.0%}" for p in prior["leave_one_out"])
        print(f"capacity prior: P {prior['prefill_tokens_per_s']:g} / D {prior['decode_tokens_per_s']:g} "
              f"tokens/s, rms {prior['rms_relative_error']:.1%}; leave-one-out {loo}")
    for u in plan["units"]:
        cs = plan["capacity_rps"][u["shape"]]
        name = "C^" if u["unit"] == UNIT_PRIOR else "C_s"
        print(f"  unit {u['shape']:3} {u['rho']:.6g} x {name} {cs:g} = {u['rho'] * cs:.4g} rps  [{u['unit']}]")
    print(f"{model}: {len(plan['sequence'])} cells in run order (rho = x the shape's capacity):")
    for i, cell in enumerate(plan["static_cells"][model], start=1):
        cs = plan["capacity_rps"][cell["shape"]]
        limit = cell.get("max_model_error_rate")
        print(f"  {i:2d} {cell['cell_id']:22} {cell['role']:8} {cell['shape']:3} "
              f"{cell['rho_factor']:>6g} x -> rho {cell['rho']:.4f} ({cell['rho'] * cs:.3f} rps) "
              f"{cell['duration_s']:.0f}s  {cell['note']}"
              + (f"  [void past {limit:g} model errors]" if limit is not None else ""))
    est = plan["estimate"]
    print(f"  {est['cells']} cells, offered load {est['offered_load_s'] / 60:.0f} min; wall clock "
          f"~{est['seconds_expected'] / 3600:.2f} h expected, <= {est['seconds_upper'] / 3600:.2f} h "
          f"(drain gate <= {plan.get('drain_limit_s', design.DRAIN_LIMIT_S):g} s; re-drives of "
          "void cells excluded)")
    flight = plan.get("in_flight") or {}
    if flight.get("cells"):
        top = max(flight["cells"], key=lambda c: c["in_flight"])
        print(f"  in-flight estimate: max {top['in_flight']} ({top['shape']} {top['rho_factor']:g} x) "
              f"of limit {flight['limit']}")
    for cid in flight.get("at_risk", []):
        print(f"  WARNING: {cid} may exceed {flight['warn_share']:.0%} of the in-flight limit "
              f"{flight['limit']} - a dispatch-delay / shed void is likely")


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
    def __init__(self, *a, sequence: Sequence[design.DesignCell],
                 plan: Optional[TrainingPlan] = None, **kw) -> None:
        super().__init__(*a, **kw)
        self.sequence = list(sequence)
        self.training_plan = plan or TRAINING_PLANS[DEFAULT_TRAINING_PLAN]
        self.drain_limit_s = self.training_plan.drain_limit_s

    def run(self, shapes: Sequence[str] = ()) -> None:
        for cell in self.sequence:
            self.drive_planned(cell)

    def failures(self) -> list[str]:
        out = []
        for o in self.outcomes:
            if o["part"] not in self.training_plan.driven_parts:
                continue
            judged = o["verdict"] in (boundary.VERDICT_HEALTHY, boundary.VERDICT_VIOLATED)
            if not judged or not (o.get("labeled_windows") or 0):
                out.append(f"{self.model}/{o['shape']} part {o['part']} {o['cell_id']} "
                           f"({o['rho_factor']:g} x): verdict {o['verdict']!r} on "
                           f"{o.get('labeled_windows') or 0} labelled windows - not a usable "
                           "training cell")
        for part in self.training_plan.both_sides_parts:
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
        if not self.training_plan.sentinels:
            return out
        drift = design.sentinel_drift(self.sentinel_summaries)
        if drift.get("flagged"):
            out.append(f"{self.model}: sentinel drift flagged ({drift['verdict']})")
        elif drift.get("flagged") is None and self.sentinel_summaries:
            out.append(f"{self.model}: sentinel drift undetermined ({drift['verdict']})")
        return out

    def result(self, status: str) -> dict:
        drift = (design.sentinel_drift(self.sentinel_summaries) if self.training_plan.sentinels
                 else None)
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


def finish(out_dir: Path, plan: dict, result: dict, status: str, code: int,
           redis_url: Optional[str] = None) -> None:
    (out_dir / ladder.DESIGN_RESULT).write_text(
        json.dumps({"mode": plan.get("mode"), "run_manifest_sha256": plan.get("run_manifest_sha256"),
                    "models": [result]}, indent=2) + "\n", encoding="utf-8")
    campaign.finalize_run(out_dir, status=status, exit_code=code, redis_url=redis_url)


def banner(lines: Sequence[str]) -> None:
    bar = "!" * 78
    print(f"{bar}\nCHECK FAILED - do not go on to the next stage without the user:", flush=True)
    for line in lines:
        print(f"  {line}", flush=True)
    print(bar, flush=True)


# --------------------------------------------------------------- the sealed set (E)

#: What a sealed plan binds: the cells (ids, shapes, loads, seeds, prompt keys, splits, void
#: limits), their order, the capacity each is scheduled against and how it was got, the label.
SEALED_PLAN_KEYS = ("mode", "training_plan", "models", "design_seed", "serial_base",
                    "static_cells", "sequence", "capacity_rps", "capacity_prior", "overload",
                    "label")
E_SHA256SUMS = "E_SHA256SUMS"
E_MANIFEST = "E_manifest.json"


def check_sealed_plan(path: Path, plan: Mapping) -> dict:
    """The run's plan against the sealed one - its dry run's ``plan.json``, sealed with a
    sha256sum sidecar before the collection: every :data:`SEALED_PLAN_KEYS` value must be
    equal, else the run is refused (it would drive cells that were not sealed)."""
    from scripts import calibration_t14 as t14

    sealed, digest = t14._read_sidecar_checked(Path(path), "sealed plan")
    ours = json.loads(json.dumps({k: plan.get(k) for k in SEALED_PLAN_KEYS}))
    differ = [k for k in SEALED_PLAN_KEYS if sealed.get(k) != ours[k]]
    if differ:
        raise ValueError(f"{path}: this run's {', '.join(differ)} differ from the sealed plan - "
                         "refusing to drive cells that were not sealed")
    return {"path": str(Path(path).resolve()), "sha256": digest, "keys": list(SEALED_PLAN_KEYS)}


def seal(out_dir: Path, raw_dir: Path, model: str, run: "TrainingSupplementRun",
         plan: Mapping) -> dict:
    """E_SHA256SUMS and E_manifest.json (both read-only), as M / T14 seal theirs - before the
    dataset: every file of the out-dir (``dataset/`` and the backfilled ``cells/`` evidence
    excluded) and every attempt's raw capture; the cells (their final attempts)."""
    from scripts import calibration_t14 as t14

    files = t14.sealed_files(out_dir, raw_dir, run.records)
    body = "".join(f"{t14._file_sha256(f)}  {Path(f).resolve()}\n" for f in files)
    sums = Path(out_dir) / E_SHA256SUMS
    sums.write_text(body, encoding="utf-8")
    os.chmod(sums, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    final = {r["cell_id"]: r for r in run.records}
    keys = ("model", "cell_id", "attempt", "shape", "role", "split", "rho_factor", "rho",
            "capacity_rps", "offered_rps", "duration_s", "warmup_s", "max_model_error_rate",
            "verdict", "online_csv", "possibly_contaminated")
    cells = []
    for cell in run.sequence:
        record = final[cell.cell_id]
        raw = Path(raw_dir) / str(record["stem"])
        cells.append({**{k: record.get(k) for k in keys},
                      "raw_files": sorted(str(p.resolve()) for p in raw.rglob("*") if p.is_file())
                      if raw.is_dir() else []})
    doc = {
        "what": (f"E ({plan['training_plan']}), the sealed held-out set of the 2026-11-03 "
                 "supplement: evaluation only - nothing that fits or picks parameters may read "
                 "it; its split is holdout"),
        "model": model,
        "mode": MODE,
        "training_plan": plan["training_plan"],
        "written_at_utc": campaign.utc_iso(),
        "sealed_plan": plan.get("sealed_plan"),
        "label": plan["label"],
        "capacity_prior": plan.get("capacity_prior"),
        "overload": plan.get("overload"),
        "cells": cells,
        "attempts": [{"cell_id": r["cell_id"], "attempt": r["attempt"],
                      "void_reasons": r.get("void_reasons")} for r in run.records],
        "run_manifest_sha256": plan.get("run_manifest_sha256"),
        "sha256sums_file": E_SHA256SUMS,
        "sha256sums_sha256": t14._file_sha256(sums),
        "how_to_evaluate": (
            f"Verify {E_SHA256SUMS} first. Read this run's standard dataset (<out-dir>/dataset, "
            "built right after this seal; re-window it from the raw captures with "
            "python -m scripts.calibration_dataset <out-dir> --window-ms ... --out-dir <new>): "
            "every row is split holdout; keep the cells listed here and the windows after "
            "warmup_s. Evaluate each candidate parameter set once. Never train on it, never "
            "pass it as --h2-dataset."),
    }
    path = Path(out_dir) / E_MANIFEST
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return {"manifest": str(path), "manifest_sha256": t14._file_sha256(path),
            "sha256sums": str(sums), "files": len(files)}


def source_runs(args) -> list[tuple[str, Path]]:
    """The run roots this collection is placed from (the supplement only when given)."""
    out = [("--base-run", Path(args.base_run))]
    if getattr(args, "boundary_supplement_run", None):
        out.append(("--boundary-supplement-run", Path(args.boundary_supplement_run)))
    return out


def check_source_load_paths(args, model: str) -> list[dict]:
    """The supplement extends --base-run's training set on --boundary-supplement-run's
    anchors, so it must reach the engine the way both did (scripts.prompt_corpus): same
    prompt corpus, same routing. Every campaign of either run that drove ``model`` is
    compared (one without a record predates the options: English, no routing header); a
    source with no such campaign is refused - its load path would be unknown. A dry run
    reports instead of refusing, as M does. Returns the reports, which the plan and the
    run manifest keep."""
    from scripts import calibration_dataset as dataset
    from scripts import prompt_corpus as corpus_record

    flags = campaign.mismatch_flags(args)
    if getattr(args, "dry_run", False):
        flags = {k: True for k in flags}
    reports = []
    for flag, root in source_runs(args):
        found = 0
        for directory in (dataset.campaign_dirs(root) if root.is_dir() else []):
            try:
                plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(f"{directory}/plan.json: unreadable ({exc})")
            if model not in (plan.get("models") or [model]):
                continue
            found += 1
            report = corpus_record.check_load_path(
                corpus_record.load_path(plan.get("provenance")), campaign.load_path(args),
                what=f"{flag} {directory}", **flags)
            reports.append({"source": flag, "campaign": str(directory), **report})
        if not found:
            raise ValueError(f"{flag} {root}: no campaign (plan.json) of {model} found, so its load "
                             "path (prompt corpus, routing) is unknown")
    return reports


def run_training_supplement(args, *, drive: Optional[Callable] = None,
                            sample_factory: Optional[Callable[[str], Callable]] = None,
                            sleep: Callable[[float], None] = time.sleep,
                            clock: Callable[[], float] = time.monotonic,
                            check_controller: bool = True) -> int:
    """``--training-supplement``: see the module docstring."""
    ladder.check_label_args(args)
    model = single_model(args)
    check_primary_label(args, model)
    tplan = training_plan(args)
    if not args.base_run:
        raise ValueError("--training-supplement needs --base-run")
    supp = getattr(args, "boundary_supplement_run", None)
    if tplan.needs_supplement(model) and not supp:
        raise ValueError(f"--training-plan {tplan.name} needs --boundary-supplement-run "
                         f"(S3's measured rho* under D6')")
    out_dir = Path(args.out_dir)
    for _flag, source in source_runs(args):
        supplement.check_new_out_dir(out_dir, source)
    load_paths = check_source_load_paths(args, model)
    resolved = resolve_units(model, Path(args.base_run), Path(supp) if supp else None, tplan)
    cap = resolve_cap(args)
    factory = design.CellFactory(model, int(args.design_seed), serial_base=tplan.serial_base)
    sequence, _sentinels = build_cells(model, factory, resolved, int(args.design_seed), tplan)
    plan, manifest = build_plan(args, model, sequence, resolved, cap, tplan)
    plan["source_load_paths"] = manifest["source_load_paths"] = load_paths
    sealed_plan = getattr(args, "sealed_plan", None)
    if sealed_plan is not None and not tplan.sealed:
        raise ValueError(f"--sealed-plan: --training-plan {tplan.name} is not a sealed set")
    if tplan.sealed and sealed_plan is None and not args.dry_run:
        raise ValueError(f"--training-plan {tplan.name} is a sealed held-out set: a run needs "
                         "--sealed-plan (its dry run's plan.json, sha256 sidecar written before "
                         "the collection)")
    if sealed_plan is not None:
        plan["sealed_plan"] = manifest["sealed_plan"] = check_sealed_plan(sealed_plan, plan)
        print(f"sealed plan {sealed_plan} (sha256 {plan['sealed_plan']['sha256'][:12]}): this run's "
              "cells are the sealed ones")
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
    run = TrainingSupplementRun(
        args, model, resolved["capacity"], factory=factory, cap=cap, out_dir=out_dir,
        raw_dir=raw_dir, drive=drive, sample=sample_factory(model), sleep=sleep, clock=clock,
        capacity_source=tplan.capacity_source, sequence=sequence, plan=tplan)
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
        if tplan.sealed:
            result["sealed"] = seal(out_dir, raw_dir, model, run, plan)
            print(f"[{model}] sealed: {result['sealed']['manifest']} ({result['sealed']['files']} "
                  f"files in {E_SHA256SUMS})", flush=True)
        if result["check_failures"]:
            banner(result["check_failures"])
            status, code = "complete_check_failed", EXIT_CHECK_FAILED
            return code
        drift = f"; sentinel drift: {result['sentinel_drift']['verdict']}" if tplan.sentinels else ""
        print(f"[{model}] complete{drift}", flush=True)
        status, code = "complete", 0
        return 0
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        if result is None:
            result = run.result(status)
        finish(out_dir, plan, result, status, code, redis_url=getattr(args, "redis_url", None))

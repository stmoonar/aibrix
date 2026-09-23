#!/usr/bin/env python3
"""The preregistered calibration design (``docs/preregistration-20260923-calibration-run2.md``).

Everything in here is pure: which cells exist, in what order, at what load, under which
identity and seeds, how the boundary is searched for, when a shape gets supplementary
cells and when the sentinels say the run drifted. :mod:`scripts.calibration_ladder`
drives it; the tests exercise it without a cluster.

What the preregistration fixes (section numbers are the document's)
--------------------------------------------------------------------
* **§7.2 hold ladder** - per shape (all eight, M included) twelve constant-rho cells at
  0.70 / 0.85 / 0.95 / 1.00 / 1.05 / 1.15 / 1.30 x rho*, one or two of each
  (:data:`LADDER`). Every hold cell's first :data:`WARMUP_S` seconds are the queue
  building up; they are *marked*, never dropped, and the analysis filters on the mark.
* **§7.1 independence** - every cell, including the two replicates of one (shape, rho),
  has its own cell id, its own arrival seed and its own prompt key
  (:class:`CellFactory`). The replayer seeds prompt *k* of a schedule from
  ``<model>|<model>-<k>``, so without a per-cell key two replicates would send the same
  prompts request for request, and with the default schedule seed the same arrival
  instants too: nominally two samples, actually one.
* **§7.1 order** - the ladder is laid out in rounds; each round holds one cell of every
  shape in random order and each shape visits its twelve ladder cells in random order
  (:func:`interleaved_ladder`). Shapes are interleaved and every shape's cells are spread
  evenly over the phase, so time drift cannot masquerade as a shape effect. The seed is
  fixed and recorded.
* **§7.3 stage 0** - :class:`PriorGuidedSearch` locates rho* per shape starting from
  the first-round *client-side* empirical boundary (``rho_priors.json``), never from the
  rho* the first round recorded. A shape the first round never saw violate starts at the
  prior's suggested point and steps up to the prior's suggested ceiling; if it never
  flips, ``boundary_found`` is False and the ladder is anchored on the highest load that
  was driven.
* **§7.3 stage 2** - one ramp per shape, 0.6 -> 1.4 rho* over 360 s.
* **§7.3 stage 3** - :func:`supplement_plan`: from the ladder's *measured* labels (not
  its rho), a shape with fewer than four violating or fewer than four healthy cells in
  [0.85, 1.15] rho* gets two more cells next to its empirical boundary.
* **§7.3 sentinels** - one fixed (shape, rho) cell after stage 0, in the middle and at the
  end; :func:`sentinel_drift` compares them against :data:`SENTINEL_DRIFT_THRESHOLDS`.
  *Revised 2026-09-23 after the run-2 drift diagnosis* (``run2_drift_analysis/``): the
  sentinel sits at :data:`SENTINEL_RHO_FACTOR` x the rho* stage 0 **measured**, not at
  0.9 x the first-round prior - which put 7b / 8b on or above their knee, where TTFT has
  20-40x the elasticity it has at 0.7 rho* and a repeat of one cell is bistable - so the
  first sentinel runs after stage 0, and drift is judged on TPOT p95 and the running
  median, not on the violating fraction.
* **§2-§6 analysis parameters** - :func:`preregistered_parameters` is the block the
  campaign writes into its frozen run manifest before the first cell.

Labels
------
A hold cell's verdict is :func:`scripts.adaptive_boundary.probe_verdict` over its
windows *after the warm-up* (:func:`hold_cell_verdict`) - the same label, windows and
evidence floor for a probe, a ladder cell, a supplementary cell and a sentinel, so the
boundary the search locates is measured with the ruler the ladder is then labelled with.
That ruler is the D-line's (adopted 2026-09-23 over the preregistered one): the primary
D6' label, 30 s windows on the 10 s grid, >= 20 completed requests per window - which is
also what every fit trains on.
A cell the backlog safety valve stopped (see ``openloop.StopOnBacklog``) is violated by
construction.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from tre_common import slo_labels

from scripts import adaptive_boundary as boundary
from scripts import gen_calibration_schedules as gen
from scripts import openloop

#: The document this module implements, relative to the ``tre/`` directory.
PREREGISTRATION_DOC = "docs/preregistration-20260923-calibration-run2.md"

# ------------------------------------------------------------ §2-§6 analysis parameters

W_P_GRID: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)
BASELINE = {"w_p": 0.0, "theta_form": "single", "lambda_wait": 3.0}
LAMBDA_WAIT_PRIMARY = 3.0
LAMBDA_WAIT_SENSITIVITY: tuple[float, ...] = (3.0, 10.0)
DECISION_RULE = {
    "loro_min_gain_ba_points": 3.0,
    "bootstrap": "cell-clustered",
    "bootstrap_interval": 0.90,
    "bootstrap_interval_must_exclude_zero": True,
    "pooled_holdout_paired_difference_min": 0.0,
    "regime_aware_requires_models": 2,
    "shared_w_p_max_loss_ba_points": 1.0,
    "otherwise": "baseline wins ('no difference detected at this resolution')",
}
TTFT_SLO_MS = 500.0
TPOT_SLO_MS = 75.0
WINDOW_MS = 30_000
#: The window grid and the label evidence floor are the D-line's (plan 2026-09-21 §6.11,
#: adopted over the preregistration on 2026-09-23, see
#: ``docs/preregistration-20260923-superseded.md``): 30 s windows ending on the 10 s
#: gateway grid (D8), at least 20 completed requests per labelled window, and the primary
#: D6' label (``tre_common.slo_labels``). The preregistered values are kept below for the
#: record and for ``calibration_decision --rule preregistered``.
STEP_MS = 10_000
WINDOW_ALIGN = "grid"
MIN_WINDOW_REQUESTS = slo_labels.DEFAULT_MIN_COMPLETED_REQUESTS
PREREGISTERED_STEP_MS = 5_000
PREREGISTERED_WINDOW_ALIGN = "none"
PREREGISTERED_MIN_WINDOW_REQUESTS = 10

# ------------------------------------------------------------------ §7 collection design

#: (rho / rho*, cells, seconds) - §7.2.
LADDER: tuple[tuple[float, int, float], ...] = (
    (0.70, 1, 150.0),
    (0.85, 2, 150.0),
    (0.95, 2, 240.0),
    (1.00, 2, 240.0),
    (1.05, 2, 240.0),
    (1.15, 2, 150.0),
    (1.30, 1, 150.0),
)
#: Seconds at the start of every hold cell during which the queue is still building.
WARMUP_S = 60.0
#: Longest wait for the engine to drain between cells before the next one is flagged.
DRAIN_LIMIT_S = 90.0
DRAIN_POLL_S = 2.0

RAMP_RHO_START = 0.6
RAMP_RHO_END = 1.4
RAMP_SECONDS = 360.0
RAMP_SEGMENT_S = 5.0

SUPPLEMENT_BAND: tuple[float, float] = (0.85, 1.15)
SUPPLEMENT_MIN_CELLS = 4
SUPPLEMENT_CELLS = 2
SUPPLEMENT_SECONDS = 240.0
#: Where the two supplementary cells go, as multiples of the empirical boundary.
SUPPLEMENT_FACTORS = {
    "violated": (1.04, 1.08),   # short of violating cells: just above the boundary
    "healthy": (0.96, 0.92),    # short of healthy cells: just below it
    "both": (0.94, 1.06),       # short of both: one on each side
}

#: The sentinel: fixed shape, one absolute rho for all three repeats -
#: ``SENTINEL_RHO_FACTOR`` x the rho* stage 0 *measured* for the shape (the ladder's
#: anchor), so the first sentinel runs after stage 0.
#:
#: Why not the first-round prior (0.9 x, the preregistered placement): run 2 put the
#: prior-anchored sentinel at 1.03 / 0.97 / 0.56 x the measured rho* for 7b / 8b / 14b.
#: At the knee TTFT has 20-40x the elasticity it has at 0.7 rho*, and the knee is bistable
#: (running reaches max-num-seqs and the queue grows without bound: one repeat 346 ms,
#: the next 1303 ms), so the 7b / 8b sentinels amplified noise into "drift"; the 14b one,
#: on the plateau, read the same three times. 0.72 is inside the 0.70-0.75 plateau band
#: the diagnosis names.
SENTINEL_SHAPE = "S2"
SENTINEL_RHO_FACTOR = 0.72
SENTINEL_SECONDS = 240.0
SENTINEL_POSITIONS = ("after_boundary", "middle", "end")
#: A later sentinel drifted when it differs from the first by more than this. The
#: criteria are TPOT p95 and the running median: both are smooth in rho on the plateau,
#: where the violating fraction is ~0 by construction and TTFT p95 is dominated by
#: prefill batching, so neither of those can see a slowdown there.
SENTINEL_DRIFT_THRESHOLDS = {
    "median_p95_tpot_relative": 0.20,
    "median_running_relative": 0.25,
}
#: Reported for every later sentinel, never flagged on (see above).
SENTINEL_INFORMATIONAL = ("median_p95_ttft_relative", "violating_fraction_absolute")

#: The smoke cell of the boundary supplement (``scripts.calibration_supplement``): one
#: hold at the located rho* (the ladder's anchor rule, the midpoint of the final
#: bracket), judged on the same post-warm-up windows and label as every probe. Its
#: violating-window fraction must land in this band; outside it the anchor the next
#: stage would build on is off, and the run says so instead of carrying on.
SMOKE_SECONDS = 300.0
SMOKE_VIOLATING_BAND: tuple[float, float] = (0.20, 0.70)

# ------------------------------------------------------------------- stage 0 (search)

BRACKET_SECONDS = 150.0   # WARMUP_S + 3 disjoint 30 s windows
BISECT_SECONDS = 150.0
BISECT_ROUNDS = 2
MAX_BRACKET_PROBES = 8
RHO_FLOOR = 0.3
#: Default multiplicative step of the bracket stage.
STEP_FOUND = 1.15
STEP_NOT_FOUND = 1.25
#: Default search ceiling for a shape whose prior boundary was found, as a multiple of it.
FOUND_SEARCH_SPAN = 2.0
#: Client-side outstanding requests at which a probe stops offering load (see
#: ``openloop.StopOnBacklog``): 256 running + 768 waiting, i.e. tens of seconds of queue
#: at any capacity these shapes have, and far below the gateway's admission ceiling.
PROBE_MAX_BACKLOG = 1024

# ------------------------------------------------------------------ roles and splits

ROLE_SENTINEL = "sentinel"
ROLE_BOUNDARY = "boundary"
ROLE_LADDER = "ladder"
ROLE_RAMP = "ramp"
ROLE_SUPPLEMENT = "adaptive"
#: The boundary supplement's check hold at the located rho* (see SMOKE_SECONDS).
ROLE_SMOKE = "smoke"
ROLES = (ROLE_SENTINEL, ROLE_BOUNDARY, ROLE_LADDER, ROLE_RAMP, ROLE_SUPPLEMENT, ROLE_SMOKE)

#: The ``stage`` values this design writes - the names the analysis
#: (``scripts.analysis.calibration_decision.PREREGISTERED``) partitions on. A hold cell
#: trains when its stage is in :data:`TRAINING_HOLD_STAGES` and is not analysed when it
#: is in :data:`EXCLUDED_HOLD_STAGES`; any other name is an error there, on purpose.
#: A ramp is identified by its primitive (``ramp``) and carries no stage.
STAGE_COARSE = boundary.STAGE_COARSE      # stage-0 probes that bracket the flip
STAGE_BISECT = boundary.STAGE_BISECT      # stage-0 probes that halve the bracket
STAGE_DWELL = boundary.STAGE_DWELL        # first-round dwell; this design has none
STAGE_LADDER = "ladder"
STAGE_ADAPTIVE = "adaptive"
STAGE_SENTINEL = "sentinel"
STAGE_RAMP = ""
TRAINING_HOLD_STAGES = frozenset({STAGE_LADDER, STAGE_ADAPTIVE})
EXCLUDED_HOLD_STAGES = frozenset({STAGE_COARSE, STAGE_BISECT, STAGE_DWELL, STAGE_SENTINEL})
#: Stage of every non-probe role (a probe's is coarse or bisect, set by the search).
#: The smoke hold is a hold at the located boundary after the search - what the first
#: round called a dwell - and like the dwell it is excluded from training: whether it
#: trains is decided after the user has seen it, not by the collector.
ROLE_STAGE = {
    ROLE_LADDER: STAGE_LADDER,
    ROLE_SUPPLEMENT: STAGE_ADAPTIVE,
    ROLE_SENTINEL: STAGE_SENTINEL,
    ROLE_RAMP: STAGE_RAMP,
    ROLE_SMOKE: STAGE_DWELL,
}

#: The acceptance set M (``scripts.calibration_acceptance``): every cell it drives besides
#: the boundary probes of its new shapes. Its shapes are held out (``gen.is_held_out``),
#: so its split is always the holdout one - the role only names what the cell is.
ROLE_ACCEPTANCE = "acceptance"
STAGE_ACCEPTANCE = "acceptance"
ROLES = (*ROLES, ROLE_ACCEPTANCE)
ROLE_STAGE[ROLE_ACCEPTANCE] = STAGE_ACCEPTANCE

SPLIT_TRAIN = "train"
SPLIT_HOLDOUT = "holdout"
#: Cells that are neither: boundary probes and sentinels.
SPLIT_AUXILIARY = "auxiliary"

PROFILE_HOLD = "hold"
PROFILE_RAMP = "ramp"
#: The first round's time-varying primitives, driven by M on an explicit ``rho_profile``
#: (``DesignCell.rho_profile``); the profile name is then also the cell's primitive.
PROFILE_STEPS = "steps"
PROFILE_BURSTS = "bursts"
PROFILES_WITH_OWN_PRIMITIVE = frozenset({PROFILE_RAMP, PROFILE_STEPS, PROFILE_BURSTS})


def split_for(role: str, shape: str) -> str:
    """§5.2 / §5.3: every cell of M (its probes included - "M shape 的全部 cell") and every
    ramp are the pooled held-out set; ladder and adaptive cells of the seven training
    shapes train; the training shapes' probes, the sentinels and the smoke holds are
    neither."""
    if gen.is_held_out(shape) or role == ROLE_RAMP:
        return SPLIT_HOLDOUT
    if role in (ROLE_BOUNDARY, ROLE_SENTINEL, ROLE_SMOKE):
        return SPLIT_AUXILIARY
    return SPLIT_TRAIN


def preregistered_parameters() -> dict:
    """The fixed parameters of §2-§7, as the run manifest records them."""
    return {
        "baseline": dict(BASELINE),
        "w_p_grid": list(W_P_GRID),
        "lambda_wait": {"primary": LAMBDA_WAIT_PRIMARY,
                        "sensitivity": list(LAMBDA_WAIT_SENSITIVITY)},
        "theta_form": {"primary": "single", "secondary_ablation": "regime-aware theta(phi)"},
        "decision_rule": dict(DECISION_RULE),
        "label": {
            "primary": "D6' slowdown TTFT: max(500 ms, 5 * (c_m + b_m * L)) per request, "
                       "idle fit from the registry slo block; TPOT p95 <= 75 ms",
            "arms": {
                slo_labels.LABEL_COLUMN: "primary (D6')",
                slo_labels.LABEL_COLUMN_FIXED: "fixed 500 / 75 ms (comparison)",
                slo_labels.LABEL_COLUMN_K3: "slowdown k = 3, floor 150 ms (ablation, D6)",
            },
            "ttft_p95_slo_ms": TTFT_SLO_MS,
            "tpot_p95_slo_ms": TPOT_SLO_MS,
            "tpot_source": "client per-request",
            "window_ms": WINDOW_MS,
            "step_ms": STEP_MS,
            "window_align": WINDOW_ALIGN,
            "min_window_requests": MIN_WINDOW_REQUESTS,
            "unserved_request_window": "violated (model_errors / proxy_transient_errors / client_timeouts)",
            "implementation": "tre_common.slo_labels.LabelDefinition",
            "supersedes": {
                "document": "docs/preregistration-20260923-superseded.md",
                "preregistered": {
                    "ttft_p95_slo_ms": TTFT_SLO_MS, "tpot_p95_slo_ms": TPOT_SLO_MS,
                    "window_ms": WINDOW_MS, "step_ms": PREREGISTERED_STEP_MS,
                    "window_align": PREREGISTERED_WINDOW_ALIGN,
                    "min_window_requests": PREREGISTERED_MIN_WINDOW_REQUESTS,
                },
            },
        },
        "ladder": [{"rho_factor": f, "cells": n, "seconds": s} for f, n, s in LADDER],
        "warmup_s": WARMUP_S,
        "warmup_rule": (
            "windows starting before cell start + warmup_s are marked in_warmup and are "
            "not used for any cell label; they are kept in the data"
        ),
        "drain": {"limit_s": DRAIN_LIMIT_S, "poll_s": DRAIN_POLL_S,
                  "rule": "engine running + waiting == 0 before the next cell; past the "
                          "limit the next cell is marked possibly_contaminated"},
        "ramp": {"rho_factor_start": RAMP_RHO_START, "rho_factor_end": RAMP_RHO_END,
                 "seconds": RAMP_SECONDS, "segment_s": RAMP_SEGMENT_S},
        "supplement": {"band": list(SUPPLEMENT_BAND), "min_cells": SUPPLEMENT_MIN_CELLS,
                       "cells": SUPPLEMENT_CELLS, "seconds": SUPPLEMENT_SECONDS,
                       "factors_of_empirical_boundary": {
                           k: list(v) for k, v in SUPPLEMENT_FACTORS.items()},
                       "label": "measured cell verdict, not rho"},
        "sentinel": {"shape": SENTINEL_SHAPE,
                     "rho_factor_of_measured_anchor": SENTINEL_RHO_FACTOR,
                     "seconds": SENTINEL_SECONDS, "positions": list(SENTINEL_POSITIONS),
                     "first_after": "stage 0 (it is placed on the rho* stage 0 measured)",
                     "drift_thresholds": dict(SENTINEL_DRIFT_THRESHOLDS),
                     "informational": list(SENTINEL_INFORMATIONAL),
                     "supersedes": "0.9 x the first-round prior anchor, first sentinel "
                                   "before stage 0, violating fraction as a drift "
                                   "criterion (run-2 drift diagnosis, 2026-09-23)"},
        "boundary_search": {
            "bracket_seconds": BRACKET_SECONDS, "bisect_seconds": BISECT_SECONDS,
            "bisect_rounds": BISECT_ROUNDS, "max_bracket_probes": MAX_BRACKET_PROBES,
            "rho_floor": RHO_FLOOR, "step_found": STEP_FOUND,
            "step_not_found": STEP_NOT_FOUND, "found_search_span": FOUND_SEARCH_SPAN,
            "probe_max_backlog": PROBE_MAX_BACKLOG,
            "anchor_rule": "midpoint of the final (healthy, violated) bracket; highest "
                           "driven load when nothing violated (boundary_found=False)",
            "verdict": "adaptive_boundary.probe_verdict on post-warm-up windows; a "
                       "backlog stop is violated",
        },
        "splits": {
            SPLIT_TRAIN: "ladder + adaptive cells of the seven training shapes",
            SPLIT_HOLDOUT: "every M cell (probes included) + the ramp of every shape",
            SPLIT_AUXILIARY: "boundary probes of the training shapes, and the sentinels",
        },
        "stages": {
            "training_hold_stages": sorted(TRAINING_HOLD_STAGES),
            "excluded_hold_stages": sorted(EXCLUDED_HOLD_STAGES),
            "by_role": {role: stage for role, stage in ROLE_STAGE.items()},
            "boundary": [STAGE_COARSE, STAGE_BISECT],
            "ramp": "primitive 'ramp', no stage",
        },
    }


# ------------------------------------------------------------------------------ priors


def _fail(message: str) -> SystemExit:
    return SystemExit(f"refusing to start: {message}")


def _load_json(path: Path, what: str):
    path = Path(path)
    if not path.is_file():
        raise _fail(f"{what} {path} does not exist")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise _fail(f"{what} {path} is not valid JSON ({exc})") from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positive(value, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise _fail(f"{what} must be a number, got {value!r}") from None
    if not math.isfinite(number) or number <= 0.0:
        raise _fail(f"{what} must be a positive finite number, got {value!r}")
    return number


@dataclass(frozen=True)
class ShapePrior:
    """What the first round says about one (model, shape), in that record's rho units."""

    model: str
    shape: str
    capacity_rps: float
    boundary_found: bool
    rho_star: Optional[float]
    search_start: float
    search_max: float
    search_step: float

    @property
    def anchor_rho(self) -> float:
        """Where the search starts, and what the sentinel is defined against."""
        return float(self.rho_star) if self.boundary_found else float(self.search_start)

    def as_dict(self) -> dict:
        return asdict(self)


#: Accepted spellings of each prior field. The first round's analysis writes the file;
#: a field it does not carry under any of these names is a refusal to start, not a guess.
_CAPACITY_KEYS = ("capacity_rps", "C_s_rps", "c_s_rps", "c_s", "C_s", "capacity")
_RHO_KEYS = ("rho_star", "empirical_rho_star", "rho_star_empirical", "empirical_boundary",
             "boundary_rho", "rho_boundary")
_FOUND_KEYS = ("boundary_found", "found")
_START_KEYS = ("search_start", "search_rho_start", "suggested_search_start",
               "upper_search_start", "search_from")
_MAX_KEYS = ("search_max", "search_rho_max", "suggested_search_max", "upper_search_max",
             "search_to", "search_limit")
_STEP_KEYS = ("search_step", "search_factor", "step_factor")
_NESTED_SEARCH_KEYS = ("search", "suggested_search", "search_range", "upper_search",
                       "suggested_upper_search")
_NESTED_START = ("start_rho", "start", "rho_start", "from", "lo", "low")
_NESTED_MAX = ("upper_rho", "max_rho", "max", "rho_max", "to", "hi", "high", "limit")
_NESTED_STEP = ("step", "factor")


def _pick(record: Mapping, keys: Sequence[str]):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _prior_records(doc) -> dict[tuple[str, str], dict]:
    """``{(model, shape): record}`` from any of the layouts a priors file may use."""
    out: dict[tuple[str, str], dict] = {}

    def add(model: str, shape: str, record) -> None:
        if not isinstance(record, Mapping):
            raise _fail(f"rho prior for {model}/{shape} is not an object: {record!r}")
        key = (str(model), str(shape))
        if key in out:
            raise _fail(f"rho prior for {model}/{shape} appears twice")
        out[key] = dict(record)

    def from_list(items, model: Optional[str] = None) -> None:
        for record in items:
            if not isinstance(record, Mapping):
                raise _fail(f"rho prior entry is not an object: {record!r}")
            m = record.get("model", model)
            s = record.get("shape")
            if m is None or s is None:
                raise _fail(f"rho prior entry names no model/shape: {record!r}")
            add(m, s, record)

    def from_model_map(models: Mapping) -> None:
        for model, body in models.items():
            if isinstance(body, Mapping) and "shapes" in body:
                body = body["shapes"]
            if isinstance(body, list):
                from_list(body, model)
            elif isinstance(body, Mapping):
                for shape, record in body.items():
                    add(model, shape, record)
            else:
                raise _fail(f"rho priors for {model} are neither a list nor an object")

    if isinstance(doc, list):
        from_list(doc)
    elif isinstance(doc, Mapping):
        if isinstance(doc.get("priors"), list):
            from_list(doc["priors"])
        elif isinstance(doc.get("priors"), Mapping):
            from_model_map(doc["priors"])
        elif isinstance(doc.get("models"), Mapping):
            from_model_map(doc["models"])
        else:
            known = {k: v for k, v in doc.items() if k in gen.MODELS}
            if not known:
                raise _fail(
                    "rho priors: expected {'models': {model: {shape: {...}}}}, "
                    "{'priors': [...]}, a list of records or {model: {shape: {...}}}"
                )
            from_model_map(known)
    else:
        raise _fail("rho priors: the file is neither an object nor a list")
    return out


def _parse_prior(model: str, shape: str, record: Mapping) -> ShapePrior:
    where = f"rho prior {model}/{shape}"
    capacity = _pick(record, _CAPACITY_KEYS)
    if capacity is None:
        raise _fail(f"{where} has no capacity (one of {_CAPACITY_KEYS})")
    capacity_rps = _positive(capacity, f"{where} capacity")
    rho = _pick(record, _RHO_KEYS)
    found = _pick(record, _FOUND_KEYS)
    if found is None:
        found = rho is not None
    if not isinstance(found, bool):
        raise _fail(f"{where} boundary_found must be true/false, got {found!r}")
    nested = _pick(record, _NESTED_SEARCH_KEYS)
    nested = nested if isinstance(nested, Mapping) else {}
    start = _pick(record, _START_KEYS)
    start = start if start is not None else _pick(nested, _NESTED_START)
    top = _pick(record, _MAX_KEYS)
    top = top if top is not None else _pick(nested, _NESTED_MAX)
    step = _pick(record, _STEP_KEYS)
    step = step if step is not None else _pick(nested, _NESTED_STEP)
    if found:
        if rho is None:
            raise _fail(f"{where} says boundary_found but carries no rho* ({_RHO_KEYS})")
        rho_star = _positive(rho, f"{where} rho*")
        search_start = rho_star if start is None else _positive(start, f"{where} search start")
        # A suggested range around a found boundary is where the first bracket should
        # land, not a ceiling: capping there would turn "the flip moved up by 20 %" into
        # boundary_found=False. The ceiling is FOUND_SEARCH_SPAN x rho* at least.
        search_max = rho_star * FOUND_SEARCH_SPAN
        if top is not None:
            search_max = max(search_max, _positive(top, f"{where} search max"))
        search_step = STEP_FOUND if step is None else float(step)
    else:
        rho_star = None if rho is None else _positive(rho, f"{where} rho*")
        if start is None or top is None:
            raise _fail(
                f"{where}: the first round never saw it violate, so the preregistration "
                "requires a widened upward search - the prior must give its start "
                f"({_START_KEYS}) and ceiling ({_MAX_KEYS})"
            )
        search_start = _positive(start, f"{where} search start")
        search_max = _positive(top, f"{where} search max")
        search_step = STEP_NOT_FOUND if step is None else float(step)
    if search_max < search_start:
        raise _fail(f"{where}: search max {search_max} is below its start {search_start}")
    if not search_step > 1.0:
        raise _fail(f"{where}: search step must be > 1, got {search_step}")
    return ShapePrior(
        model=model, shape=shape, capacity_rps=capacity_rps, boundary_found=found,
        rho_star=rho_star, search_start=search_start, search_max=search_max,
        search_step=float(search_step),
    )


@dataclass(frozen=True)
class RhoPriors:
    path: str
    sha256: str
    document: object
    priors: dict

    def get(self, model: str, shape: str) -> ShapePrior:
        return self.priors[(model, shape)]

    def for_model(self, model: str) -> dict[str, ShapePrior]:
        return {s: p for (m, s), p in self.priors.items() if m == model}

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "parsed": {f"{m}/{s}": p.as_dict() for (m, s), p in sorted(self.priors.items())},
            "document": self.document,
        }


def load_rho_priors(
    path: Path, *, models: Sequence[str], shapes: Sequence[str] = gen.ALL_SHAPES
) -> RhoPriors:
    """Read and validate ``rho_priors.json``; a missing or malformed file refuses to start.

    Every (model, shape) the campaign will drive must have a prior: stage 0 is defined
    relative to it, and a shape without one has nowhere principled to start.
    """
    doc = _load_json(path, "rho priors")
    records = _prior_records(doc)
    parsed: dict[tuple[str, str], ShapePrior] = {}
    missing = []
    for model in models:
        for shape in shapes:
            record = records.get((model, shape))
            if record is None:
                missing.append(f"{model}/{shape}")
                continue
            parsed[(model, shape)] = _parse_prior(model, shape, record)
    if missing:
        raise _fail(f"rho priors {path} has no entry for {', '.join(missing)}")
    return RhoPriors(path=str(path), sha256=_sha256(path), document=doc, priors=parsed)


@dataclass(frozen=True)
class RegimeGroups:
    path: str
    sha256: str
    document: object
    #: ``{model: {group: (shapes...)}}``; the same grouping repeated when the file has one.
    by_model: dict

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "groups": {m: {g: list(s) for g, s in groups.items()}
                       for m, groups in self.by_model.items()},
            "document": self.document,
        }


def _parse_grouping(body, where: str) -> dict[str, tuple[str, ...]]:
    if isinstance(body, Mapping) and isinstance(body.get("groups"), (Mapping, list)):
        body = body["groups"]
    groups: dict[str, tuple[str, ...]] = {}
    if isinstance(body, list):
        for i, item in enumerate(body):
            if isinstance(item, Mapping):
                name = str(item.get("name") or item.get("group") or f"group{i + 1}")
                shapes = item.get("shapes")
            else:
                name, shapes = f"group{i + 1}", item
            groups[name] = shapes
    elif isinstance(body, Mapping):
        for name, item in body.items():
            groups[str(name)] = item.get("shapes") if isinstance(item, Mapping) else item
    else:
        raise _fail(f"{where}: groups are neither an object nor a list")
    parsed: dict[str, tuple[str, ...]] = {}
    for name, shapes in groups.items():
        if not isinstance(shapes, list) or not all(isinstance(s, str) for s in shapes):
            raise _fail(f"{where}: group {name!r} is not a list of shape names")
        parsed[name] = tuple(shapes)
    training = set(gen.TRAINING_SHAPES)
    seen: list[str] = [s for shapes in parsed.values() for s in shapes]
    if len(parsed) != 3:
        raise _fail(f"{where}: §5.1 needs exactly 3 regime groups, got {len(parsed)}")
    small = [n for n, s in parsed.items() if len(s) < 2]
    if small:
        raise _fail(f"{where}: §5.1 needs >= 2 shapes per group; {small} have fewer")
    if len(seen) != len(set(seen)):
        raise _fail(f"{where}: a shape is in more than one group")
    held_out = [s for s in seen if gen.is_held_out(s)]
    if held_out:
        raise _fail(f"{where}: {held_out} is held out and may not be in any group")
    if set(seen) != training:
        raise _fail(
            f"{where}: groups must cover exactly the training shapes "
            f"{sorted(training)}; missing {sorted(training - set(seen))}, "
            f"unknown {sorted(set(seen) - training)}"
        )
    return parsed


def load_regime_groups(path: Path, *, models: Sequence[str]) -> RegimeGroups:
    """Read and validate ``regime_groups.json`` (§5.1); anything off refuses to start.

    One grouping for every model (``{"groups": {...}}``) or one per model
    (``{"models": {model: {"groups": {...}}}}``).
    """
    doc = _load_json(path, "regime groups")
    by_model: dict[str, dict[str, tuple[str, ...]]] = {}
    if isinstance(doc, Mapping) and isinstance(doc.get("models"), Mapping):
        for model in models:
            if model not in doc["models"]:
                raise _fail(f"regime groups {path} has no grouping for {model}")
            by_model[model] = _parse_grouping(doc["models"][model], f"regime groups {model}")
    else:
        body = doc
        if isinstance(doc, Mapping):
            for key in ("groups", "regime_groups", "regimes"):
                if key in doc:
                    body = doc[key]
                    break
            else:
                raise _fail(
                    f"regime groups {path}: expected {{'groups': ...}} or "
                    "{'models': {model: {'groups': ...}}}"
                )
        grouping = _parse_grouping(body, "regime groups")
        by_model = {model: grouping for model in models}
    return RegimeGroups(path=str(path), sha256=_sha256(path), document=doc, by_model=by_model)


# ---------------------------------------------------------------------- cell identity

#: Load codes of this design: ``CELL_CODE_BASE + model index * MODEL_CODE_BLOCK +
#: serial``. Far above every code an earlier campaign used (60 / 95 / 120 for the fixed
#: primitives, 1001-1999 for boundary probes), and different per model, so a cell id is
#: unique across the whole run, not just within one campaign.
CELL_CODE_BASE = 1_000_000
MODEL_CODE_BLOCK = 100_000


def cell_code(model: str, serial: int) -> int:
    if model not in gen.MODELS:
        raise ValueError(f"unknown model {model!r}; cell codes are allocated per model")
    if not 0 < int(serial) < MODEL_CODE_BLOCK:
        raise ValueError(f"cell serial {serial} out of range")
    return CELL_CODE_BASE + gen.MODELS.index(model) * MODEL_CODE_BLOCK + int(serial)


def derived_seed(design_seed: int, *parts) -> int:
    """A 31-bit seed that is a pure function of the design seed and ``parts``.

    sha256 rather than ``hash``: Python salts ``hash`` per process for strings, and a
    seed that changes between the dry run and the run is not a seed.
    """
    text = "|".join(str(p) for p in (int(design_seed), *parts))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def _nominal_cell_id(shape: str, code: int) -> str:
    if gen.is_mixture(shape):
        return f"i0_o0_c{code}"
    (_w, i, o), = gen.shape_components(shape)
    return f"i{gen._length_nominal(i)}_o{gen._length_nominal(o)}_c{code}"


@dataclass
class DesignCell:
    """One cell of the design: its identity, its load and where it sits in the run."""

    model: str
    shape: str
    role: str
    serial: int
    code: int
    cell_id: str
    duration_s: float
    profile: str = PROFILE_HOLD
    #: Offered load relative to this shape's located rho* (None for a sentinel, whose rho
    #: is fixed from the prior, and for a probe, which has an absolute rho).
    rho_factor: Optional[float] = None
    #: Offered load relative to the prior capacity ``C_s`` (the ramp's is its peak).
    rho: Optional[float] = None
    replicate: int = 1
    round: Optional[int] = None
    stage: str = ""
    #: A sentinel's place in the run (start / middle / end).
    position: str = ""
    arrival_seed: int = 0
    prompt_key: str = ""
    warmup_s: float = WARMUP_S
    split: str = ""
    note: str = ""
    #: An explicit ``[(start_s, end_s, rho), ...]`` load profile (rho of the capacity the
    #: cell is scheduled against; overlapping segments add up). When set it is the cell's
    #: load, whatever ``rho`` / ``profile`` say; the acceptance set's steps / bursts / ramp
    #: cells use it.
    rho_profile: Optional[list] = None

    @property
    def primitive(self) -> str:
        return self.profile if self.profile in PROFILES_WITH_OWN_PRIMITIVE else gen.HOLD_PRIMITIVE

    def stem(self, attempt: int = 1) -> str:
        """Output CSV / raw directory stem of one attempt - unique per cell and attempt."""
        return f"{self.model}_{self.shape}_{self.role}_c{self.code}_a{int(attempt)}"

    @property
    def schedule_stem(self) -> str:
        return f"{self.shape}_{self.role}_c{self.code}"

    def as_dict(self) -> dict:
        body = asdict(self)
        body["primitive"] = self.primitive
        return body


class CellFactory:
    """Hands out cells with a fresh serial - and so a fresh id and fresh seeds - each.

    ``serial_base`` starts the serials of a later collection above an earlier one's, so
    the two never share a cell id (and so never share seeds or prompts) when their data
    is pooled; see ``calibration_supplement.SUPPLEMENT_SERIAL_BASE``.
    """

    def __init__(self, model: str, design_seed: int, *, serial_base: int = 0) -> None:
        if not 0 <= int(serial_base) < MODEL_CODE_BLOCK:
            raise ValueError(f"serial base {serial_base} out of range")
        self.model = model
        self.design_seed = int(design_seed)
        self.serial_base = int(serial_base)
        self.serial = int(serial_base)

    def new(self, shape: str, role: str, duration_s: float, **fields) -> DesignCell:
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        if shape not in gen.ALL_SHAPES and shape not in gen.ACCEPTANCE_SHAPES:
            raise ValueError(f"unknown shape {shape!r}")
        if role == ROLE_BOUNDARY:
            if fields.get("stage") not in (STAGE_COARSE, STAGE_BISECT):
                raise ValueError("a boundary probe's stage is coarse or bisect, "
                                 f"not {fields.get('stage')!r}")
        else:
            if fields.get("stage", ROLE_STAGE[role]) != ROLE_STAGE[role]:
                raise ValueError(f"a {role} cell's stage is {ROLE_STAGE[role]!r}")
            fields["stage"] = ROLE_STAGE[role]
        self.serial += 1
        code = cell_code(self.model, self.serial)
        cell_id = _nominal_cell_id(shape, code)
        return DesignCell(
            model=self.model,
            shape=shape,
            role=role,
            serial=self.serial,
            code=code,
            cell_id=cell_id,
            duration_s=float(duration_s),
            arrival_seed=derived_seed(self.design_seed, self.model, cell_id, "arrivals"),
            prompt_key=f"p{self.design_seed}.{cell_id}",
            split=split_for(role, shape),
            **fields,
        )


# --------------------------------------------------------------------------- schedules


def hold_profile(rho: float, seconds: float) -> list[tuple[float, float, float]]:
    return [(0.0, float(seconds), float(rho))]


def ramp_profile(
    anchor_rho: float,
    *,
    seconds: float = RAMP_SECONDS,
    start: float = RAMP_RHO_START,
    end: float = RAMP_RHO_END,
    segment_s: float = RAMP_SEGMENT_S,
) -> list[tuple[float, float, float]]:
    """start -> end x rho*, linear, in ~segment_s steps valued at their midpoints."""
    n = max(1, int(round(seconds / segment_s)))
    out = []
    for k in range(n):
        frac = (k + 0.5) / n
        rho = float(anchor_rho) * (start + (end - start) * frac)
        out.append((k * seconds / n, (k + 1) * seconds / n, rho))
    return out


def cell_schedule(cell: DesignCell, capacity_rps: float, *, anchor_rho: Optional[float],
                  cap=None, capacity_source: str = "rho_priors") -> tuple[dict, dict]:
    """(trace body, metadata) for a cell whose load is known."""
    if cell.rho_profile:
        profile = [(float(a), float(b), float(r)) for a, b, r in cell.rho_profile]
    elif cell.profile == PROFILE_RAMP:
        if anchor_rho is None:
            raise ValueError(f"{cell.cell_id}: a ramp needs its shape's rho*")
        profile = ramp_profile(anchor_rho, seconds=cell.duration_s)
    else:
        if cell.rho is None:
            raise ValueError(f"{cell.cell_id}: a hold cell needs its rho")
        profile = hold_profile(cell.rho, cell.duration_s)
    extra = {
        "role": cell.role,
        "stage": cell.stage,
        "rho": cell.rho,
        "rho_factor": cell.rho_factor,
        "anchor_rho": anchor_rho,
        "replicate": cell.replicate,
        "round": cell.round,
        "serial": cell.serial,
        "arrival_seed": cell.arrival_seed,
        "prompt_key": cell.prompt_key,
        "warmup_s": cell.warmup_s,
        "split": cell.split,
    }
    return gen.build_rho_profile_schedule(
        cell.model, cell.shape, capacity_rps, profile,
        primitive=cell.primitive, load_code=cell.code,
        capacity_source=capacity_source, cap=cap, extra=extra,
    )


# ------------------------------------------------------------------------------ ladder


def ladder_items() -> list[tuple[float, int, float]]:
    """One shape's twelve ladder cells: (rho factor, replicate, seconds)."""
    return [(factor, rep, seconds)
            for factor, count, seconds in LADDER for rep in range(1, count + 1)]


def interleaved_ladder(shapes: Sequence[str], rng: random.Random) -> list[list[tuple]]:
    """Rounds of ``(shape, rho_factor, replicate, seconds)``: each round holds one cell of
    every shape in random order, and each shape meets its ladder in random order.

    A plain shuffle of all 96 cells would also interleave, but it can bunch one shape's
    cells into one part of the phase by chance; rounds cannot, so every shape samples the
    whole duration of the phase and a slow drift lands on all of them alike. Consecutive
    rounds never put the same shape back to back.
    """
    per_shape: dict[str, list[tuple[float, int, float]]] = {}
    for shape in shapes:
        items = ladder_items()
        rng.shuffle(items)
        per_shape[shape] = items
    n_rounds = len(ladder_items())
    rounds: list[list[tuple]] = []
    previous_last: Optional[str] = None
    for r in range(n_rounds):
        order = list(shapes)
        rng.shuffle(order)
        if previous_last is not None and len(order) > 1 and order[0] == previous_last:
            order[0], order[1] = order[1], order[0]
        rounds.append([(shape, *per_shape[shape][r]) for shape in order])
        previous_last = order[-1]
    return rounds


@dataclass
class StaticPlan:
    """The cells whose existence does not depend on anything measured."""

    sentinels: list[DesignCell]
    ladder_rounds: list[list[DesignCell]]
    ramps: list[DesignCell]
    #: What the sentinels are placed on (their rho is known only after stage 0).
    sentinel_rule: str = ""

    @property
    def ladder(self) -> list[DesignCell]:
        return [cell for rnd in self.ladder_rounds for cell in rnd]

    def all_cells(self) -> list[DesignCell]:
        return [*self.sentinels, *self.ladder, *self.ramps]


def build_static_plan(
    model: str,
    priors: RhoPriors,
    factory: CellFactory,
    *,
    shapes: Sequence[str] = gen.ALL_SHAPES,
) -> StaticPlan:
    """Sentinels, the interleaved ladder and the ramps of one model, ids and seeds fixed.

    Sentinel, ladder and ramp loads are relative to rho*, which stage 0 has not measured
    yet; their ``rho`` is filled in by :func:`anchor_cells` once it has.
    """
    rng = random.Random(derived_seed(factory.design_seed, model, "ladder-order"))
    priors.get(model, SENTINEL_SHAPE)  # the sentinel shape must be searched in stage 0
    rule = f"{SENTINEL_RHO_FACTOR} x the rho* stage 0 measures for {SENTINEL_SHAPE}"
    sentinels = [
        factory.new(SENTINEL_SHAPE, ROLE_SENTINEL, SENTINEL_SECONDS,
                    rho_factor=SENTINEL_RHO_FACTOR, replicate=i + 1, position=position,
                    note=rule)
        for i, position in enumerate(SENTINEL_POSITIONS)
    ]
    ladder_rounds = []
    for r, rnd in enumerate(interleaved_ladder(shapes, rng), start=1):
        ladder_rounds.append([
            factory.new(shape, ROLE_LADDER, seconds, rho_factor=factor, replicate=rep, round=r)
            for shape, factor, rep, seconds in rnd
        ])
    ramp_order = list(shapes)
    rng.shuffle(ramp_order)
    ramps = [
        factory.new(shape, ROLE_RAMP, RAMP_SECONDS, profile=PROFILE_RAMP,
                    rho_factor=RAMP_RHO_END, warmup_s=0.0,
                    note=f"{RAMP_RHO_START} -> {RAMP_RHO_END} x rho* over {RAMP_SECONDS:g} s")
        for shape in ramp_order
    ]
    return StaticPlan(sentinels=sentinels, ladder_rounds=ladder_rounds, ramps=ramps,
                      sentinel_rule=rule)


def anchor_cells(cells: Iterable[DesignCell], anchors: Mapping[str, float]) -> None:
    """Fill in the absolute rho of every cell defined relative to rho*."""
    for cell in cells:
        if cell.rho_factor is None:
            continue
        anchor = anchors.get(cell.shape)
        if anchor is None:
            raise ValueError(f"{cell.shape} has no rho* anchor")
        cell.rho = round(float(cell.rho_factor) * float(anchor), 6)


# ----------------------------------------------------------------- stage 0 (the search)

@dataclass
class PriorGuidedSearch:
    """Locate one shape's rho* starting from its first-round prior.

    Bracket: probe at the prior; step up (healthy) or down (violated) by ``step`` until the
    verdict flips, never above ``search_max`` nor below :data:`RHO_FLOOR`. Then bisect
    the (healthy, violated) bracket ``bisect_rounds`` times. Void and inconclusive probes
    follow :mod:`scripts.adaptive_boundary`: re-driven once (an inconclusive one for twice
    as long), then the search stops.

    ``grid`` (ascending, in the same rho units) replaces the multiplicative walk where it
    reaches: the first probe is its lowest point, a healthy verdict moves to the next
    grid point above, and the bracket stage ends at the first violation (then bisects).
    Below the lowest grid point it steps down by ``step`` as usual; ``search_max`` is
    the ceiling as usual (the boundary supplement sets it to the highest grid point).
    """

    model: str
    shape: str
    start_rho: float
    search_max: float
    step: float
    prior_found: bool
    bracket_seconds: float = BRACKET_SECONDS
    bisect_seconds: float = BISECT_SECONDS
    bisect_rounds: int = BISECT_ROUNDS
    max_bracket_probes: int = MAX_BRACKET_PROBES
    floor: float = RHO_FLOOR
    max_retries: int = boundary.MAX_VOID_RETRIES
    grid: tuple[float, ...] = ()

    results: list = field(default_factory=list)
    healthy_rho: Optional[float] = None
    violating_rho: Optional[float] = None
    stopped_reason: str = ""
    exhausted: str = ""
    _bracket_done: int = 0
    _bisect_done: int = 0
    _pending: Optional[boundary.Probe] = None

    @classmethod
    def from_prior(cls, prior: ShapePrior, **kw) -> "PriorGuidedSearch":
        return cls(model=prior.model, shape=prior.shape,
                   start_rho=prior.anchor_rho, search_max=prior.search_max,
                   step=prior.search_step, prior_found=prior.boundary_found, **kw)

    @property
    def bracketed(self) -> bool:
        return self.healthy_rho is not None and self.violating_rho is not None

    @property
    def done(self) -> bool:
        if self.stopped_reason or self.exhausted:
            return True
        return self.bracketed and self._bisect_done >= self.bisect_rounds

    @property
    def boundary_found(self) -> bool:
        return self.violating_rho is not None

    @property
    def last_verdict(self) -> Optional[str]:
        return self.results[-1].verdict if self.results else None

    @property
    def anchor_rho(self) -> Optional[float]:
        """The rho* the ladder is built on (see ``anchor_rule``)."""
        if self.bracketed:
            return round((self.healthy_rho + self.violating_rho) / 2.0, 6)
        if self.violating_rho is not None:
            return self.violating_rho
        if self.healthy_rho is not None:
            return self.healthy_rho
        return None

    @property
    def anchor_rule(self) -> str:
        if self.bracketed:
            return "midpoint of the final (healthy, violated) bracket"
        if self.violating_rho is not None:
            return "lowest violated load: every probe down to the floor violated"
        if self.healthy_rho is not None:
            return "highest driven load: nothing violated (boundary_found=False)"
        return "none: no probe produced a verdict"

    def next_probe(self) -> Optional[boundary.Probe]:
        if self.done:
            return None
        if self._pending is not None:
            return self._pending
        if not self.bracketed:
            if self._bracket_done >= self.max_bracket_probes:
                self.exhausted = (
                    f"{self.max_bracket_probes} bracket probes without a flip"
                )
                return None
            if self.healthy_rho is None and self.violating_rho is None:
                rho = self.grid[0] if self.grid else self.start_rho
            elif self.violating_rho is None:
                if self.healthy_rho >= self.search_max * (1 - 1e-9):
                    self.exhausted = (
                        f"healthy at the search ceiling {self.search_max:g}; the boundary "
                        "is above everything offered"
                    )
                    return None
                above = [g for g in self.grid if g > self.healthy_rho * (1 + 1e-9)]
                rho = (min(above[0], self.search_max) if above
                       else min(self.healthy_rho * self.step, self.search_max))
            else:
                rho = self.violating_rho / self.step
                if rho < self.floor:
                    self.exhausted = f"violated down to the floor {self.floor:g}"
                    return None
            probe = boundary.Probe(round(rho, 6), self.bracket_seconds, STAGE_COARSE)
        else:
            rho = (self.healthy_rho + self.violating_rho) / 2.0
            probe = boundary.Probe(round(rho, 6), self.bisect_seconds, STAGE_BISECT)
        self._pending = probe
        return probe

    def record(self, result: boundary.ProbeResult) -> None:
        self.results.append(result)
        self._pending = None
        probe = result.probe
        if result.verdict in (boundary.VERDICT_VOID, boundary.VERDICT_INCONCLUSIVE):
            nxt = boundary.next_void_attempt(probe.attempt, max_retries=self.max_retries)
            if nxt is None:
                self.stopped_reason = (
                    f"probe at rho={probe.rho:g} was {result.verdict} on attempt "
                    f"{probe.attempt}; the search will not move on a cell that measured "
                    "nothing"
                )
                return
            duration = probe.duration_s * (
                boundary.INCONCLUSIVE_DURATION_FACTOR
                if result.verdict == boundary.VERDICT_INCONCLUSIVE else 1.0
            )
            self._pending = boundary.Probe(probe.rho, duration, probe.stage, attempt=nxt)
            return
        if result.verdict == boundary.VERDICT_VIOLATED:
            if self.violating_rho is None or probe.rho < self.violating_rho:
                self.violating_rho = probe.rho
        elif result.verdict == boundary.VERDICT_HEALTHY:
            if self.healthy_rho is None or probe.rho > self.healthy_rho:
                self.healthy_rho = probe.rho
        if probe.stage == STAGE_COARSE:
            self._bracket_done += 1
        else:
            self._bisect_done += 1

    def status(self) -> dict:
        """:func:`scripts.adaptive_boundary.rho_star_status` of this search's probes:
        measured / lower_bound / upper_bound / grid_artifact, with the bracket. (Its
        ``rho_star`` is the lowest violated load; :attr:`anchor_rho` is the midpoint.)"""
        return boundary.rho_star_status([r.as_dict() for r in self.results])

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "shape": self.shape,
            "search": "prior-guided",
            "start_rho": self.start_rho,
            "search_max": self.search_max,
            "step": self.step,
            "grid": list(self.grid),
            "prior_boundary_found": self.prior_found,
            "healthy_rho": self.healthy_rho,
            "violating_rho": self.violating_rho,
            "boundary_found": self.boundary_found,
            "rho_star": self.anchor_rho,
            "anchor_rho": self.anchor_rho,
            "anchor_rule": self.anchor_rule,
            "exhausted": self.exhausted,
            "stopped_reason": self.stopped_reason,
            "probes": [r.as_dict() for r in self.results],
        }


# ------------------------------------------------------------------ cell verdicts


def post_warmup_rows(rows: Sequence[Mapping], *, start_ms: Optional[int],
                     warmup_s: float) -> list[Mapping]:
    """The windows a hold cell is labelled on: those starting after the warm-up."""
    if start_ms is None or not warmup_s:
        return list(rows)
    cutoff = int(start_ms) + int(round(float(warmup_s) * 1000))
    return [row for row in rows if int(row["window_start_ms"]) >= cutoff]


def backlog_stopped(guard: Mapping) -> bool:
    return (
        bool(guard.get("truncated"))
        and guard.get("truncation_cause") == openloop.TRUNCATION_BACKLOG
    )


def hold_cell_verdict(
    rows: Sequence[Mapping],
    guard: Mapping,
    *,
    warmup_s: float,
    label: Optional["slo_labels.LabelSpec"] = None,
    ttft_slo_ms: float = TTFT_SLO_MS,
    tpot_slo_ms: float = TPOT_SLO_MS,
) -> dict:
    """The verdict of one hold cell (probe, ladder, supplement or sentinel).

    ``label`` is the run's primary window label (``calibration_campaign.primary_label``,
    D6'); without it the fixed ``ttft_slo_ms`` / ``tpot_slo_ms`` pair is used.

    Void when the guard voided it; violated when the backlog valve stopped it (it ran far
    past any healthy operating point, and is short of windows only because it was
    stopped); otherwise :func:`scripts.adaptive_boundary.probe_verdict` on the windows
    after the warm-up.
    """
    void_reasons = [str(r) for r in (guard.get("void_reasons") or [])]
    if void_reasons:
        return {"verdict": boundary.VERDICT_VOID, "void_reasons": void_reasons,
                "windows": len(rows), "labeled_windows": 0, "independent_windows": 0,
                "violating_windows": 0, "backlog_stopped": False}
    kept = post_warmup_rows(rows, start_ms=guard.get("start_ms"), warmup_s=warmup_s)
    verdict = boundary.probe_verdict(
        kept, label=label, ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms,
    )
    stopped = backlog_stopped(guard)
    return {
        "verdict": boundary.VERDICT_VIOLATED if stopped else verdict.verdict,
        "void_reasons": [],
        "windows": verdict.windows,
        "labeled_windows": verdict.labeled_windows,
        "independent_windows": verdict.independent_windows,
        "violating_windows": verdict.violating_windows,
        "backlog_stopped": stopped,
    }


# ---------------------------------------------------------------- stage 3 (supplement)


@dataclass(frozen=True)
class CellOutcome:
    shape: str
    rho_factor: float
    verdict: str


def empirical_boundary(outcomes: Sequence[CellOutcome]) -> tuple[Optional[float], str]:
    """The rho factor where a step function best separates healthy from violated cells.

    Only conclusive verdicts count. The threshold minimising misclassified cells is taken
    between two adjacent distinct factors (midpoint); ties go to the lowest such
    threshold. Nothing violated -> 10 % above the highest healthy factor; nothing healthy
    -> 10 % below the lowest violated one.
    """
    points = sorted((o.rho_factor, o.verdict) for o in outcomes
                    if o.verdict in (boundary.VERDICT_HEALTHY, boundary.VERDICT_VIOLATED))
    if not points:
        return None, "no conclusive cell"
    healthy = [f for f, v in points if v == boundary.VERDICT_HEALTHY]
    violated = [f for f, v in points if v == boundary.VERDICT_VIOLATED]
    if not violated:
        return round(max(healthy) * 1.10, 6), "nothing violated: 1.10 x the highest healthy"
    if not healthy:
        return round(min(violated) / 1.10, 6), "nothing healthy: the lowest violated / 1.10"
    factors = sorted({f for f, _ in points})
    best: Optional[tuple[int, float]] = None
    for lo, hi in zip(factors, factors[1:]):
        t = (lo + hi) / 2.0
        errors = sum(1 for f, v in points
                     if (f < t and v == boundary.VERDICT_VIOLATED)
                     or (f >= t and v == boundary.VERDICT_HEALTHY))
        if best is None or errors < best[0]:
            best = (errors, t)
    if best is None:  # one distinct factor with both verdicts
        return round(factors[0], 6), "one factor with both verdicts"
    return round(best[1], 6), f"step-function threshold ({best[0]} misclassified cell(s))"


def supplement_plan(outcomes_by_shape: Mapping[str, Sequence[CellOutcome]]) -> list[dict]:
    """§7.3 stage 3: which shapes get supplementary cells, and where.

    Counts are over ladder cells whose *planned* factor is in :data:`SUPPLEMENT_BAND`, by
    *measured* verdict. A shape short of either side gets :data:`SUPPLEMENT_CELLS` cells
    placed relative to its empirical boundary, on the short side.
    """
    lo, hi = SUPPLEMENT_BAND
    decisions = []
    for shape in sorted(outcomes_by_shape):
        outcomes = outcomes_by_shape[shape]
        band = [o for o in outcomes if lo - 1e-9 <= o.rho_factor <= hi + 1e-9]
        violated = sum(1 for o in band if o.verdict == boundary.VERDICT_VIOLATED)
        healthy = sum(1 for o in band if o.verdict == boundary.VERDICT_HEALTHY)
        short_v = violated < SUPPLEMENT_MIN_CELLS
        short_h = healthy < SUPPLEMENT_MIN_CELLS
        b, rule = empirical_boundary(outcomes)
        decision = {
            "shape": shape,
            "band_violated_cells": violated,
            "band_healthy_cells": healthy,
            "band_cells": len(band),
            "empirical_boundary_factor": b,
            "empirical_boundary_rule": rule,
            "supplement": [],
            "reason": "",
        }
        if short_v or short_h:
            side = "both" if (short_v and short_h) else ("violated" if short_v else "healthy")
            anchor = b if b is not None else 1.0
            decision["supplement"] = [round(anchor * f, 6) for f in SUPPLEMENT_FACTORS[side]]
            decision["reason"] = (
                f"{violated} violated / {healthy} healthy cell(s) in {list(SUPPLEMENT_BAND)} "
                f"rho* (need >= {SUPPLEMENT_MIN_CELLS} each): short of {side}"
            )
        else:
            decision["reason"] = "enough cells on both sides of the boundary"
        decisions.append(decision)
    return decisions


# -------------------------------------------------------------------------- sentinels


def sentinel_summary(
    rows: Sequence[Mapping],
    guard: Mapping,
    *,
    warmup_s: float,
    label: Optional["slo_labels.LabelSpec"] = None,
    ttft_slo_ms: float = TTFT_SLO_MS,
    tpot_slo_ms: float = TPOT_SLO_MS,
) -> dict:
    """One sentinel after its warm-up: the median window TPOT p95 and the median of the
    windows' mean ``running`` (the drift criteria), plus the median TTFT p95 and the
    violating fraction by ``label`` (the run's primary label; the fixed pair without it),
    which are reported but not judged on (see :data:`SENTINEL_DRIFT_THRESHOLDS`)."""
    if guard.get("void_reasons"):
        return {"measured": False, "why": "void"}
    targets = label if label is not None else slo_labels.slo_targets(
        ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms)
    kept = post_warmup_rows(rows, start_ms=guard.get("start_ms"), warmup_s=warmup_s)
    labels = [(r, slo_labels.window_slo_label(r, targets)) for r in kept]
    labeled = [r for r, label in labels if label != slo_labels.LABEL_UNLABELED]
    if not labeled:
        return {"measured": False, "why": "no labelled window after the warm-up"}

    def median(column: str, over: Sequence[Mapping]) -> Optional[float]:
        values = [float(r[column]) for r in over if r.get(column) not in (None, "")]
        return round(statistics.median(values), 3) if values else None

    violating = sum(1 for _r, label in labels if label == slo_labels.LABEL_VIOLATED)
    return {
        "measured": True,
        "labeled_windows": len(labeled),
        "median_p95_tpot_client_ms": median(slo_labels.P95_TPOT_CLIENT, labeled),
        # every post-warm-up window: the engine's occupancy does not need completions
        "median_running": median(SENTINEL_RUNNING_COLUMN, kept),
        "median_p95_ttft_client_ms": median(slo_labels.P95_TTFT_CLIENT, labeled),
        "violating_fraction": round(violating / len(labeled), 4),
    }


#: The window CSV column the running median is taken over (``r3_grid``: the mean of the
#: engine's ``num_requests_running`` samples inside the window).
SENTINEL_RUNNING_COLUMN = "avg_running"


def _relative(a, b) -> Optional[float]:
    return None if not a or b is None else abs(float(b) / float(a) - 1.0)


def sentinel_drift(summaries: Sequence[Mapping]) -> dict:
    """Compare every later sentinel with the first: flagged on the TPOT p95 and running
    medians (:data:`SENTINEL_DRIFT_THRESHOLDS`); the TTFT p95 and the violating fraction
    are reported alongside (:data:`SENTINEL_INFORMATIONAL`) and never flag."""
    thresholds = SENTINEL_DRIFT_THRESHOLDS
    if not summaries or not summaries[0].get("measured"):
        return {"flagged": None, "thresholds": dict(thresholds),
                "informational": list(SENTINEL_INFORMATIONAL), "checks": [],
                "verdict": "undetermined: the first sentinel measured nothing"}
    ref = summaries[0]
    checks = []
    flagged = False
    undetermined = False
    judged = (("median_p95_tpot_client_ms", "median_p95_tpot_relative"),
              ("median_running", "median_running_relative"))
    for i, later in enumerate(summaries[1:], start=2):
        if not later.get("measured"):
            undetermined = True
            checks.append({"sentinel": i, "measured": False})
            continue
        check = {"sentinel": i, "measured": True}
        for column, key in judged:
            rel = _relative(ref.get(column), later.get(column))
            check[key] = None if rel is None else round(rel, 4)
            if rel is None:
                undetermined = True
            elif rel > thresholds[key]:
                flagged = True
        rel = _relative(ref.get("median_p95_ttft_client_ms"),
                        later.get("median_p95_ttft_client_ms"))
        check["median_p95_ttft_relative"] = None if rel is None else round(rel, 4)
        check["violating_fraction_absolute"] = round(
            abs(later["violating_fraction"] - ref["violating_fraction"]), 4)
        checks.append(check)
    verdict = ("drift: the run is flagged" if flagged
               else "undetermined: a sentinel measured nothing" if undetermined
               else "no drift above the thresholds")
    return {"flagged": flagged if (flagged or not undetermined) else None,
            "thresholds": dict(thresholds), "informational": list(SENTINEL_INFORMATIONAL),
            "checks": checks, "verdict": verdict}


# ------------------------------------------------------------------------- smoke


def smoke_verdict(
    rows: Sequence[Mapping],
    guard: Mapping,
    *,
    warmup_s: float = WARMUP_S,
    label: Optional["slo_labels.LabelSpec"] = None,
    band: tuple[float, float] = SMOKE_VIOLATING_BAND,
) -> dict:
    """The smoke hold's check: its violating-window fraction, on exactly the windows and
    label a probe is judged on (:func:`hold_cell_verdict`), against ``band``.

    ``in_band`` is None when the cell measured nothing (void, or no labelled window):
    that is not a pass, and the caller reports it as loudly as a miss.
    """
    verdict = hold_cell_verdict(rows, guard, warmup_s=warmup_s, label=label)
    lo, hi = band
    labeled = int(verdict.get("labeled_windows") or 0)
    body = {**verdict, "band": [lo, hi], "violating_fraction": None, "in_band": None}
    if verdict["verdict"] == boundary.VERDICT_VOID or not labeled:
        body["why"] = "void" if verdict["verdict"] == boundary.VERDICT_VOID \
            else "no labelled window after the warm-up"
        return body
    fraction = int(verdict["violating_windows"]) / labeled
    body["violating_fraction"] = round(fraction, 4)
    body["in_band"] = bool(lo <= fraction <= hi)
    return body


# ------------------------------------------------------------------------ the estimate


def _search_probe_counts(prior: ShapePrior) -> tuple[int, int]:
    """(expected, upper) probe count of one shape's search, without re-drives.

    Found prior: the prior and one step bracket it, then the bisections. Not found: the
    upward steps from the start to the ceiling are the upper bound (all of them healthy
    means no bisection); the expectation assumes the flip halfway up.
    """
    if prior.boundary_found:
        return 2 + BISECT_ROUNDS, min(MAX_BRACKET_PROBES, 4) + BISECT_ROUNDS
    steps = 1
    rho = prior.search_start
    while rho < prior.search_max * (1 - 1e-9) and steps < MAX_BRACKET_PROBES:
        rho = min(rho * prior.search_step, prior.search_max)
        steps += 1
    expected = max(2, math.ceil(steps / 2) + 1) + BISECT_ROUNDS
    upper = max(steps, min(MAX_BRACKET_PROBES, steps + 1) + BISECT_ROUNDS)
    return expected, upper


def request_timeout_s(shape: str) -> float:
    """The client's per-request read deadline for a shape's longest request.

    ``StreamingHttpSender`` waits ``max(30, max_tokens / 4)`` seconds for data. A cell
    that ends with a backlog returns only when its last request has finished or hit this
    deadline, so it is how long a heavily overloaded probe can outlive its schedule.
    """
    worst = 0
    for _w, _i, o in gen.shape_components(shape):
        worst = max(worst, int(o.high) if hasattr(o, "high") else int(o))
    return max(30.0, worst / 4.0)


def estimate_model_seconds(
    priors_for_model: Mapping[str, ShapePrior],
    *,
    cooldown_s: float,
    overhead_s: float,
    shapes: Sequence[str] = gen.ALL_SHAPES,
) -> dict:
    """Wall clock of one model's run, per stage, expected and upper bound.

    A gap is the wait between two cells: ``max(cooldown, drain)`` plus the driver's own
    start-up (``overhead_s``, measured ~6 s on the first round). Expected assumes the
    engine drains inside the cooldown; the upper bound waits the full drain limit.
    """
    gap_exp = max(cooldown_s, 0.0) + overhead_s
    gap_up = max(cooldown_s, DRAIN_LIMIT_S) + overhead_s
    stages: dict[str, dict] = {}

    def stage(name, cells_exp, secs_exp, cells_up, secs_up, note=""):
        stages[name] = {
            "cells_expected": cells_exp,
            "cells_upper": cells_up,
            "seconds_expected": round(secs_exp + cells_exp * gap_exp, 1),
            "seconds_upper": round(secs_up + cells_up * gap_up, 1),
            "note": note,
        }

    stage("sentinel", 3, 3 * SENTINEL_SECONDS, 3, 3 * SENTINEL_SECONDS)
    exp_n = up_n = 0
    exp_s = up_s = 0.0
    for shape in shapes:
        e, u = _search_probe_counts(priors_for_model[shape])
        tail = request_timeout_s(shape)
        exp_n += e
        up_n += u
        # Half the probes of a search violate; a violated probe can outlive its schedule
        # by up to the request deadline while its backlog finishes or times out.
        exp_s += e * BRACKET_SECONDS + 0.5 * e * tail
        up_s += u * BISECT_SECONDS + u * tail
    stage("boundary", exp_n, exp_s, up_n, up_s,
          "expected: found priors bracket in 2 probes, not-found ones flip halfway up, "
          "half the probes outlive their schedule by the request deadline; upper: "
          "not-found ones step all the way to their ceiling and every probe pays the "
          "deadline; inconclusive/void re-drives excluded")
    per_shape = sum(n * s for _f, n, s in LADDER)
    n_ladder = len(ladder_items()) * len(shapes)
    stage("ladder", n_ladder, per_shape * len(shapes), n_ladder, per_shape * len(shapes))
    stage("ramp", len(shapes), RAMP_SECONDS * len(shapes),
          len(shapes), RAMP_SECONDS * len(shapes))
    half = math.ceil(len(shapes) / 2)
    stage("adaptive", SUPPLEMENT_CELLS * half, SUPPLEMENT_CELLS * half * SUPPLEMENT_SECONDS,
          SUPPLEMENT_CELLS * len(shapes),
          SUPPLEMENT_CELLS * len(shapes) * SUPPLEMENT_SECONDS,
          "expected: half the shapes need it; upper: all of them")
    total_exp = sum(s["seconds_expected"] for s in stages.values())
    total_up = sum(s["seconds_upper"] for s in stages.values())
    return {
        "gap_expected_s": gap_exp,
        "gap_upper_s": gap_up,
        "stages": stages,
        "cells_expected": sum(s["cells_expected"] for s in stages.values()),
        "cells_upper": sum(s["cells_upper"] for s in stages.values()),
        "seconds_expected": round(total_exp, 1),
        "seconds_upper": round(total_up, 1),
    }

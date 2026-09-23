#!/usr/bin/env python3
"""The boundary supplement: re-probe named shapes above a finished ladder run's rho*, then
check the located rho* with one smoke hold (plan 2026-09-21 §6.11 D14 / D19, first segment).

Entered from ``python -m scripts.calibration_campaign --reprobe-shapes MODEL:SHAPE
--reprobe-base <ladder run root> --reprobe-grid MODEL:F1,F2,...`` (see
``run_supplement.sh`` next to ``run_calibration.sh`` on the host).

Why a supplement, and why this search
-------------------------------------
Under the D6' label the boundary of S3 (i2048_o96) moved out of the second round's
ladder: about 1.4 x rho*_run2 for 7b, 1.7 x for 8b, and above anything any data reached
for 14b (> 1.74 x, where the engine was still far from saturated - running 12 of 256 -
so the load was too low, not capped). The second round's stage 0 cannot be re-used as
is: it starts from the first round's prior and was judged under the label of its day.

The unit here is the base run's own anchor, ``rho*_base`` (its ``design_result.json``
anchor, cross-checked against its boundary JSON, in rho of the base run's prior capacity
C_s). ``--reprobe-grid`` lists multiples of it per model; the search
(:class:`scripts.calibration_design.PriorGuidedSearch` with ``grid``) starts at the
lowest grid point, walks the grid upwards while probes are healthy, stops at the first
violated probe, and bisects the (healthy, violated) bracket twice. When even the lowest
point violates it steps down by ``STEP_FOUND``; the highest grid point is the ceiling -
healthy there and rho* is a lower bound, which is reported as such, never as a number.
It stops only with both sides observed or the list used up.

One ruler
---------
Every probe and the smoke hold are cells of the ladder design (:mod:`scripts.
calibration_ladder`): their verdict is :func:`scripts.calibration_design.hold_cell_verdict`
- the windows after the 60 s warm-up, labelled by the campaign's primary label (D6',
``tre_common.slo_labels``: TTFT max(500 ms, 5 x idle TTFT(L)), TPOT 75 ms, >= 20
completions per window, 30 s windows on the 10 s grid), with the evidence floor counted
in disjoint windows - exactly the windows and label every fit trains on. The run refuses
to start under any other TTFT mode (D19). Every cell also gets the ladder's discipline:
its own id / arrival seed / prompt key, a bounded drain before it, the backlog valve on
probes, the void rule, and a ``cells.jsonl`` ledger line, so the standard dataset and
``rewindow_from_raw --ledger`` read this run like any ladder run.

Cell ids start at serial :data:`SUPPLEMENT_SERIAL_BASE`, so they never collide with the
base run's when the two are pooled.

The smoke hold
--------------
With ``--smoke-at-rho-star``, a shape whose rho* came out ``measured`` gets one
:data:`~scripts.calibration_design.SMOKE_SECONDS` hold at the located rho* (the ladder's
anchor rule: the midpoint of the final bracket). Its violating-window fraction must be
in :data:`~scripts.calibration_design.SMOKE_VIOLATING_BAND`. A miss, a smoke hold that
measured nothing, or a rho* that is not ``measured`` (so no smoke ran) is printed as a
banner, recorded in ``design_result.json`` and makes the run exit
:data:`EXIT_CHECK_FAILED` - the next stage is the user's call, not the collector's.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from tre_common import slo_labels

from scripts import adaptive_boundary as boundary
from scripts import admission_cap as admission
from scripts import calibration_campaign as campaign
from scripts import calibration_design as design
from scripts import calibration_ladder as ladder
from scripts import gen_calibration_schedules as gen

MODE = "boundary_supplement"
#: Serials (and so cell codes) of the supplement start here: the base run used a few
#: hundred per model from 1, and MODEL_CODE_BLOCK is 100000.
SUPPLEMENT_SERIAL_BASE = 50_000
#: Exit code of a run that finished but whose check failed (smoke out of band, smoke
#: unmeasured, or rho* not measured): distinct from a driver failure (1).
EXIT_CHECK_FAILED = 3


# ----------------------------------------------------------------------- the base run


@dataclass(frozen=True)
class BaseAnchor:
    """One (model, shape)'s rho* in the finished ladder run the supplement builds on."""

    model: str
    shape: str
    #: The ladder's anchor (rho of the base run's prior capacity).
    anchor_rho: float
    capacity_rps: float
    anchor_rule: str
    base_root: str
    rho_priors_path: str
    rho_priors_sha256: str

    @property
    def anchor_rps(self) -> float:
        return self.anchor_rho * self.capacity_rps

    def rho_of(self, factor: float) -> float:
        return round(float(factor) * self.anchor_rho, 6)

    def factor_of(self, rho: Optional[float]) -> Optional[float]:
        return None if rho is None else round(float(rho) / self.anchor_rho, 4)

    def as_dict(self) -> dict:
        body = asdict(self)
        body["anchor_rps"] = round(self.anchor_rps, 4)
        return body


def _read(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"base run: {path} does not exist")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"base run: {path} is not valid JSON ({exc})") from exc


def load_base_anchor(base_root: Path, model: str, shape: str) -> BaseAnchor:
    """rho*_base of (model, shape) from a finished ladder run, read three ways that must
    agree: ``design_result.json`` (the anchor the ladder was built on), the shape's
    boundary JSON (the search that produced it, and the prior's C_s) and the frozen
    ``run_manifest.json`` (the priors' C_s and the priors file hash)."""
    model_dir = Path(base_root) / model
    result = _read(model_dir / ladder.DESIGN_RESULT)
    entry = next((m for m in result.get("models") or [] if m.get("model") == model), None)
    if entry is None:
        raise ValueError(f"base run: {model_dir / ladder.DESIGN_RESULT} has no result for {model}")
    if not str(entry.get("status", "")).startswith("complete"):
        raise ValueError(f"base run: {model} ended {entry.get('status')!r}, not complete")
    anchor = (entry.get("anchors") or {}).get(shape)
    if anchor is None:
        raise ValueError(f"base run: {model} has no anchor for {shape}")
    search = _read(model_dir / "boundary" / f"{model}_{shape}.json")
    if search.get("anchor_rho") is None or not math.isclose(
            float(search["anchor_rho"]), float(anchor), rel_tol=1e-9):
        raise ValueError(f"base run: {model}/{shape} anchor {anchor} disagrees with its "
                         f"boundary JSON ({search.get('anchor_rho')})")
    manifest = _read(model_dir / ladder.RUN_MANIFEST)
    priors = manifest.get("rho_priors") or {}
    parsed = (priors.get("parsed") or {}).get(f"{model}/{shape}") or {}
    capacity = parsed.get("capacity_rps")
    searched_capacity = (search.get("prior") or {}).get("capacity_rps")
    if capacity is None or searched_capacity is None or not math.isclose(
            float(capacity), float(searched_capacity), rel_tol=1e-9):
        raise ValueError(f"base run: {model}/{shape} capacity in the run manifest "
                         f"({capacity}) disagrees with its boundary JSON ({searched_capacity})")
    return BaseAnchor(
        model=model, shape=shape, anchor_rho=float(anchor), capacity_rps=float(capacity),
        anchor_rule=str((entry.get("anchor_sources") or {}).get(shape, "")),
        base_root=str(base_root), rho_priors_path=str(priors.get("path", "")),
        rho_priors_sha256=str(priors.get("sha256", "")),
    )


def parse_grid(items: Sequence[str]) -> dict[str, tuple[float, ...]]:
    """``MODEL:F1,F2,...`` (repeatable) -> {model: ascending factors of rho*_base}."""
    out: dict[str, tuple[float, ...]] = {}
    for item in items:
        model, sep, body = str(item).partition(":")
        if not sep or not model or not body:
            raise ValueError(f"--reprobe-grid {item!r}: expected MODEL:F1,F2,...")
        try:
            factors = [float(x) for x in body.split(",") if x.strip()]
        except ValueError:
            raise ValueError(f"--reprobe-grid {item!r}: the factors must be numbers") from None
        if not factors or any(not math.isfinite(f) or f <= 0 for f in factors):
            raise ValueError(f"--reprobe-grid {item!r}: needs positive factors")
        if factors != sorted(set(factors)):
            raise ValueError(f"--reprobe-grid {item!r}: factors must be strictly ascending")
        if model in out:
            raise ValueError(f"--reprobe-grid: {model} given twice")
        out[model] = tuple(factors)
    return out


def new_search(base: BaseAnchor, factors: Sequence[float]) -> design.PriorGuidedSearch:
    """The search of one (model, shape): the grid in rho of the base run's C_s."""
    grid = tuple(base.rho_of(f) for f in factors)
    return design.PriorGuidedSearch(
        model=base.model, shape=base.shape, start_rho=grid[0], search_max=grid[-1],
        step=design.STEP_FOUND, prior_found=True, grid=grid,
    )


# ------------------------------------------------------------------------ the plan


def simulate(base: BaseAnchor, factors: Sequence[float], flip_factor: float,
             *, smoke: bool) -> dict:
    """What the search drives when the boundary is a clean step at ``flip_factor`` x
    rho*_base: the real search class, answered by a step function."""
    search = new_search(base, factors)
    flip = base.rho_of(flip_factor)
    probes = []
    while True:
        probe = search.next_probe()
        if probe is None:
            break
        verdict = boundary.VERDICT_VIOLATED if probe.rho >= flip else boundary.VERDICT_HEALTHY
        probes.append({"stage": probe.stage, "rho": probe.rho,
                       "factor_of_base": base.factor_of(probe.rho),
                       "offered_rps": round(probe.rho * base.capacity_rps, 3),
                       "seconds": probe.duration_s, "verdict": verdict})
        search.record(boundary.ProbeResult(probe=probe, verdict=verdict))
    status = search.status()
    smoke_cell = None
    if smoke and status["status"] == boundary.RHO_STAR_MEASURED:
        smoke_cell = {"rho": search.anchor_rho, "factor_of_base": base.factor_of(search.anchor_rho),
                      "offered_rps": round(search.anchor_rho * base.capacity_rps, 3),
                      "seconds": design.SMOKE_SECONDS}
    return {"flip_factor": round(flip_factor, 4), "probes": probes,
            "rho_star_status": status["status"],
            "anchor_factor_of_base": base.factor_of(search.anchor_rho), "smoke": smoke_cell}


def scenarios(base: BaseAnchor, factors: Sequence[float], *, smoke: bool) -> list[dict]:
    """One simulated run per place the flip can be: below the grid, inside each grid
    interval (at its midpoint) and above it."""
    flips = [factors[0] * 0.9]
    flips += [(a + b) / 2.0 for a, b in zip(factors, factors[1:])]
    flips.append(factors[-1] * 1.2)
    return [simulate(base, factors, f, smoke=smoke) for f in flips]


def scenario_seconds(scenario: Mapping, shape: str, *, cooldown_s: float) -> dict:
    """Wall clock of one scenario: every cell pays a gap before it (expected: the
    cooldown, upper: the drain limit; plus the driver's start-up) and a violated probe
    can outlive its schedule by the request deadline while its backlog finishes."""
    gap_exp = max(cooldown_s, 0.0) + ladder.DRIVER_OVERHEAD_S
    gap_up = max(cooldown_s, design.DRAIN_LIMIT_S) + ladder.DRIVER_OVERHEAD_S
    tail = design.request_timeout_s(shape)
    cells = list(scenario["probes"]) + ([scenario["smoke"]] if scenario["smoke"] else [])
    load = sum(float(c["seconds"]) for c in cells)
    tails = tail * sum(1 for p in scenario["probes"] if p["verdict"] == boundary.VERDICT_VIOLATED)
    return {"cells": len(cells), "offered_load_s": load,
            "seconds_expected": round(load + tails + len(cells) * gap_exp, 1),
            "seconds_upper": round(load + tails + len(cells) * gap_up, 1)}


def format_scenario(scenario: Mapping, seconds: Mapping) -> str:
    walk = " ".join(
        f"{p['factor_of_base']:g}{'V' if p['verdict'] == boundary.VERDICT_VIOLATED else 'H'}"
        + ("" if p["stage"] == design.STAGE_COARSE else "*")
        for p in scenario["probes"])
    smoke = (f"smoke {scenario['smoke']['factor_of_base']:g}x ({scenario['smoke']['offered_rps']:g} rps, "
             f"{scenario['smoke']['seconds']:.0f}s)" if scenario["smoke"]
             else f"no smoke (rho* {scenario['rho_star_status']})")
    return (f"flip @{scenario['flip_factor']:g}x: {walk} -> {smoke}; {seconds['cells']} cells, "
            f"~{seconds['seconds_expected'] / 60:.0f} min (<= {seconds['seconds_upper'] / 60:.0f})")


# ----------------------------------------------------------------------------- run


class SupplementRun(ladder.LadderRun):
    """The searches and smoke holds of one model, on the ladder's per-cell machinery."""

    def __init__(self, args, model: str, bases: Mapping[str, BaseAnchor],
                 factors: Sequence[float], *, smoke: bool, factory, cap, out_dir: Path,
                 raw_dir: Path, drive: Callable, sample: Callable[[], dict],
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        empty = design.StaticPlan(sentinels=[], ladder_rounds=[], ramps=[])
        super().__init__(args, model, None, factory=factory, plan=empty, cap=cap,
                         out_dir=out_dir, raw_dir=raw_dir, drive=drive, sample=sample,
                         sleep=sleep, clock=clock)
        self.bases = dict(bases)
        self.factors = tuple(factors)
        self.smoke = bool(smoke)
        self.statuses: dict[str, dict] = {}
        self.smokes: dict[str, dict] = {}

    def capacity(self, shape: str) -> float:
        return self.bases[shape].capacity_rps

    def run(self, shapes: Sequence[str] = ()) -> None:
        order = list(shapes or self.bases)
        for shape in order:
            self.searches[shape] = new_search(self.bases[shape], self.factors)
        self.drive_searches(order)
        for shape in order:
            search = self.searches[shape]
            self.statuses[shape] = search.status()
            if self.statuses[shape]["status"] == boundary.RHO_STAR_MEASURED:
                self.anchors[shape] = search.anchor_rho
                self.anchor_sources[shape] = search.anchor_rule
            self._write_search(shape)
            if search.stopped_reason and search.last_verdict == boundary.VERDICT_VOID:
                raise ladder.CampaignStopped(
                    f"boundary search {self.model}/{shape}: {search.stopped_reason}")
            base = self.bases[shape]
            print(f"[{self.model}] {shape}: rho* {self.statuses[shape]['status']}, bracket "
                  f"{base.factor_of(search.healthy_rho)} - {base.factor_of(search.violating_rho)}"
                  f" x rho*_base" + (f"; anchor {base.factor_of(self.anchors[shape])} x"
                                     if shape in self.anchors else ""), flush=True)
        if not self.smoke:
            return
        for shape in order:
            if shape not in self.anchors:
                self.smokes[shape] = {
                    "driven": False, "in_band": None,
                    "why": f"rho* is {self.statuses[shape]['status']}, not measured"}
                continue
            self.run_smoke(shape)

    def run_smoke(self, shape: str) -> None:
        base = self.bases[shape]
        anchor = self.anchors[shape]
        cell = self.factory.new(
            shape, design.ROLE_SMOKE, design.SMOKE_SECONDS, rho=anchor, rho_factor=1.0,
            note=f"smoke hold at the located rho* ({base.factor_of(anchor)} x rho*_base)")
        print(f"[{self.model}] smoke {shape} rho={anchor:g} ({design.SMOKE_SECONDS:.0f}s)",
              flush=True)
        rows, guard, record = self.drive_cell(cell)
        verdict = design.smoke_verdict(rows, guard, warmup_s=cell.warmup_s, label=self.label)
        self.smokes[shape] = {"driven": True, "cell_id": cell.cell_id,
                              "attempt": record["attempt"], "rho": anchor,
                              "factor_of_base": base.factor_of(anchor),
                              "offered_rps": record.get("offered_rps"), **verdict}

    def _write_search(self, shape: str) -> None:
        search = self.searches[shape]
        base = self.bases[shape]
        status = self.statuses.get(shape) or search.status()
        body = search.as_dict()
        for probe in body["probes"]:
            probe["factor_of_base"] = base.factor_of(probe["rho"])
            probe["offered_rps"] = round(float(probe["rho"]) * base.capacity_rps, 4)
        body.update({
            "mode": MODE,
            "rho_star_status": status["status"],
            "rho_star_bracket": [status["healthy_rho"], status["violating_rho"]],
            "bracket_rel_width": status["bracket_rel_width"],
            "anchor_rho": self.anchors.get(shape),
            "anchor_source": self.anchor_sources.get(shape, "none: rho* is not measured"),
            "anchor_rps": (None if shape not in self.anchors
                           else round(self.anchors[shape] * base.capacity_rps, 4)),
            "base": base.as_dict(),
            "grid_factors_of_base": list(self.factors),
            "in_base_units": {
                "healthy": base.factor_of(search.healthy_rho),
                "violating": base.factor_of(search.violating_rho),
                "anchor": base.factor_of(self.anchors.get(shape)),
            },
        })
        (self.out_dir / "boundary" / f"{self.model}_{shape}.json").write_text(
            json.dumps(body, indent=2) + "\n", encoding="utf-8")

    def failures(self) -> list[str]:
        """Every reason the check failed, in words; empty when it passed."""
        out = []
        for shape, status in self.statuses.items():
            if status["status"] != boundary.RHO_STAR_MEASURED:
                out.append(f"{self.model}/{shape}: rho* is {status['status']} "
                           f"(bracket {status['healthy_rho']} - {status['violating_rho']}), "
                           "not measured")
        for shape, smoke in self.smokes.items():
            if not smoke.get("driven"):
                continue
            if smoke.get("in_band") is None:
                out.append(f"{self.model}/{shape}: the smoke hold measured nothing "
                           f"({smoke.get('why', '')})")
            elif not smoke["in_band"]:
                lo, hi = smoke["band"]
                out.append(f"{self.model}/{shape}: smoke violating windows "
                           f"{smoke['violating_fraction']:.0%} "
                           f"({smoke['violating_windows']}/{smoke['labeled_windows']}) is "
                           f"outside {lo:.0%}-{hi:.0%}")
        return out

    def result(self, status: str) -> dict:
        return {
            "model": self.model,
            "status": status,
            "shapes": {
                shape: {
                    "base": self.bases[shape].as_dict(),
                    "grid_factors_of_base": list(self.factors),
                    "rho_star_status": (self.statuses.get(shape) or {}).get("status"),
                    "anchor_rho": self.anchors.get(shape),
                    "anchor_factor_of_base": self.bases[shape].factor_of(self.anchors.get(shape)),
                    "anchor_rps": (None if shape not in self.anchors else
                                   round(self.anchors[shape] * self.bases[shape].capacity_rps, 4)),
                    "search": self.searches[shape].as_dict() if shape in self.searches else None,
                    "smoke": self.smokes.get(shape),
                }
                for shape in self.bases
            },
            "anchors": dict(self.anchors),
            "check_failures": self.failures(),
            "possibly_contaminated_cells": self.contaminated,
            "attempts": len(self.records),
        }


def check_new_out_dir(out_dir: Path, base_root: Path) -> None:
    out_dir = Path(out_dir).resolve()
    base = Path(base_root).resolve()
    if out_dir == base or base in out_dir.parents or out_dir in base.parents:
        raise ValueError(f"supplement output {out_dir} overlaps the base run {base}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"supplement output {out_dir} exists and is not empty - pick a new one")


def build_plan(args, model: str, bases: Mapping[str, BaseAnchor], factors: Sequence[float],
               *, smoke: bool, cap) -> tuple[dict, dict]:
    """(plan.json, run manifest): the targets, the grid, every flip scenario's probe
    sequence with its wall clock, the label and the provenance."""
    label = campaign.primary_label(args, model)
    targets = []
    for shape, base in bases.items():
        cases = scenarios(base, factors, smoke=smoke)
        for case in cases:
            case["wall_clock"] = scenario_seconds(case, shape, cooldown_s=args.cooldown_s)
        (_w, i, _o), = gen.shape_components(shape)
        targets.append({
            "model": model,
            "shape": shape,
            "base": base.as_dict(),
            "grid_factors_of_base": list(factors),
            "grid_rho": [base.rho_of(f) for f in factors],
            "grid_rps": [round(base.rho_of(f) * base.capacity_rps, 3) for f in factors],
            "nominal_input_tokens": gen._length_nominal(i),
            "ttft_slo_ms_at_nominal_input": round(label.ttft_slo_ms(float(gen._length_nominal(i))), 1),
            "scenarios": cases,
        })
    upper = max(c["wall_clock"]["seconds_upper"] for t in targets for c in t["scenarios"])
    search = {
        "bracket_seconds": design.BRACKET_SECONDS, "bisect_seconds": design.BISECT_SECONDS,
        "bisect_rounds": design.BISECT_ROUNDS, "max_bracket_probes": design.MAX_BRACKET_PROBES,
        "step_below_grid": design.STEP_FOUND, "rho_floor": design.RHO_FLOOR,
        "warmup_s": design.WARMUP_S, "probe_max_backlog": design.PROBE_MAX_BACKLOG,
        "min_probe_windows": boundary.MIN_PROBE_WINDOWS,
        "violation_window_fraction": boundary.VIOLATION_WINDOW_FRACTION,
        "max_bracket_rel_width": boundary.MAX_BRACKET_REL_WIDTH,
        "verdict": "calibration_design.hold_cell_verdict: post-warm-up windows, primary "
                   "(D6') label, disjoint-window evidence floor; a backlog stop is violated",
        "anchor_rule": "midpoint of the final (healthy, violated) bracket, when rho* is measured",
    }
    smoke_doc = {"enabled": bool(smoke), "seconds": design.SMOKE_SECONDS,
                 "violating_band": list(design.SMOKE_VIOLATING_BAND),
                 "rule": "one hold at the located rho* when it is measured; outside the band "
                         f"the run exits {EXIT_CHECK_FAILED}"}
    provenance = campaign.run_provenance(args)
    label_doc = {
        "ttft_slo_mode": label.ttft_slo_mode, "ttft_slowdown_k": label.ttft_slowdown_k,
        "ttft_floor_ms": label.ttft_floor_ms, "ttft_idle_c_ms": label.ttft_idle_c_ms,
        "ttft_idle_b_ms_per_token": label.ttft_idle_b_ms_per_token,
        "tpot_p95_ms": label.tpot_p95_ms, "min_completed_requests": label.min_completed_requests,
        "window_ms": args.window_ms, "step_ms": args.fit_step_ms,
        "window_align": getattr(args, "fit_window_align", "grid"),
    }
    plan = {
        # "ladder": the dataset reads this run from its ledger like any ladder run.
        "design": ladder.DESIGN_NAME,
        "mode": MODE,
        "generated_at_utc": campaign.utc_iso(),
        "provenance": provenance,
        "admission_cap": cap.as_dict(),
        "models": [model],
        "design_seed": int(args.design_seed),
        "serial_base": SUPPLEMENT_SERIAL_BASE,
        "cooldown_s": args.cooldown_s,
        "run_manifest": ladder.RUN_MANIFEST,
        "static_cells": {model: []},
        "targets": targets,
        "label": label_doc,
        "search": search,
        "smoke": smoke_doc,
        "estimated_max_wall_clock_s": upper,
    }
    manifest = {
        "design": ladder.DESIGN_NAME,
        "mode": MODE,
        "written_at_utc": campaign.utc_iso(),
        "why": "plan 2026-09-21 §6.11 D19: S3 boundary re-probe under the D6' label + smoke",
        "base_run": str(args.reprobe_base),
        "bases": {shape: base.as_dict() for shape, base in bases.items()},
        "grid_factors_of_base": list(factors),
        "search": search,
        "smoke": smoke_doc,
        "label": provenance.get("label"),
        "design_seed": int(args.design_seed),
        "serial_base": SUPPLEMENT_SERIAL_BASE,
        "seed_derivation": "as the ladder design (calibration_design.CellFactory)",
        "models": [model],
        "cooldown_floor_s": args.cooldown_s,
        "provenance": provenance,
        "admission_cap": cap.as_dict(),
    }
    return plan, manifest


def print_plan(plan: dict) -> None:
    lb = plan["label"]
    print(f"probe/smoke label (D6' primary): TTFT p95 <= max({lb['ttft_floor_ms']:g} ms, "
          f"{lb['ttft_slowdown_k']:g} x ({lb['ttft_idle_c_ms']:g} + {lb['ttft_idle_b_ms_per_token']:g}"
          f" x L)), TPOT p95 <= {lb['tpot_p95_ms']:g} ms, >= {lb['min_completed_requests']} "
          f"completions, {lb['window_ms'] / 1000:g} s windows / {lb['step_ms'] / 1000:g} s step "
          f"({lb['window_align']})")
    for target in plan["targets"]:
        base = target["base"]
        print(f"{target['model']}/{target['shape']}: rho*_base {base['anchor_rho']:g} x C_s "
              f"{base['capacity_rps']:g} = {base['anchor_rps']:g} rps ({base['anchor_rule']})")
        print(f"  TTFT SLO at L={target['nominal_input_tokens']}: "
              f"{target['ttft_slo_ms_at_nominal_input']:g} ms")
        print("  grid (x rho*_base): " + ", ".join(
            f"{f:g} ({rps:g} rps)" for f, rps in zip(target["grid_factors_of_base"], target["grid_rps"])))
        for case in target["scenarios"]:
            print("  " + format_scenario(case, case["wall_clock"]))
    s = plan["search"]
    print(f"  probes {s['bracket_seconds']:.0f}s (bisect {s['bisect_seconds']:.0f}s x "
          f"{s['bisect_rounds']}), first {s['warmup_s']:.0f}s dropped; walk marks: H healthy, "
          "V violated, * bisect")
    print(f"  worst case <= {plan['estimated_max_wall_clock_s'] / 60:.0f} min (re-drives excluded)")


def run_boundary_supplement(args, targets: Mapping[str, Sequence[str]], *,
                            drive: Optional[Callable] = None,
                            sample_factory: Optional[Callable[[str], Callable]] = None,
                            sleep: Callable[[float], None] = time.sleep,
                            clock: Callable[[], float] = time.monotonic,
                            check_controller: bool = True) -> int:
    """``--reprobe-shapes`` with ``--reprobe-base``: see the module docstring."""
    ladder.check_label_args(args)
    if len(targets) != 1:
        raise ValueError("--reprobe-base drives one model per run (one ledger, one "
                         f"out-dir); got {sorted(targets)}")
    (model, shapes), = targets.items()
    grids = parse_grid(getattr(args, "reprobe_grid", None) or [])
    if model not in grids:
        raise ValueError(f"--reprobe-grid has no grid for {model}")
    if set(grids) - {model}:
        raise ValueError(f"--reprobe-grid names models this run does not drive: "
                         f"{sorted(set(grids) - {model})}")
    label = campaign.primary_label(args, model)
    if label.ttft_slo_mode != slo_labels.TTFT_SLO_MODE_SLOWDOWN:
        raise ValueError("the boundary supplement is judged on the D6' primary label "
                         f"(slowdown TTFT); refusing --fit-ttft-slo-mode {label.ttft_slo_mode}")
    base_root = Path(args.reprobe_base)
    bases = {shape: load_base_anchor(base_root, model, shape) for shape in shapes}
    out_dir = Path(args.out_dir)
    check_new_out_dir(out_dir, base_root)
    args.models = model
    index_cap = None
    if getattr(args, "index", None) and Path(args.index).exists():
        index_cap = (json.loads(Path(args.index).read_text(encoding="utf-8"))
                     .get("admission_cap", {}).get("name"))
    cap = admission.get_cap(getattr(args, "cap", None) or index_cap or admission.DEFAULT_CAP_NAME)
    smoke = bool(getattr(args, "smoke_at_rho_star", False))
    plan, manifest = build_plan(args, model, bases, grids[model], smoke=smoke, cap=cap)
    out_dir.mkdir(parents=True, exist_ok=True)
    print_plan(plan)
    if args.dry_run:
        plan["run_manifest_preview"] = manifest
        (out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"dry run: wrote {out_dir / 'plan.json'} (the run manifest is only written "
              "when a run starts)")
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
    sample_factory = sample_factory or (
        lambda m: ladder.make_engine_sampler(m, args.model_namespace))
    run = SupplementRun(
        args, model, bases, grids[model], smoke=smoke,
        factory=design.CellFactory(model, int(args.design_seed),
                                   serial_base=SUPPLEMENT_SERIAL_BASE),
        cap=cap, out_dir=out_dir, raw_dir=raw_dir, drive=drive,
        sample=sample_factory(model), sleep=sleep, clock=clock)
    status, code = "failed", 1
    result = None
    try:
        try:
            run.run(list(shapes))
        except ladder.CampaignStopped as stop:
            print(f"STOPPED: {stop}", flush=True)
            status, code = "stopped", stop.code or 1
            result = run.result(f"stopped: {stop}")
            return code
        result = run.result("complete")
        failures = result["check_failures"]
        if failures:
            banner = "!" * 78
            print(f"{banner}\nCHECK FAILED - do not go on to the next stage without the user:",
                  flush=True)
            for line in failures:
                print(f"  {line}", flush=True)
            print(banner, flush=True)
            status, code = "complete_check_failed", EXIT_CHECK_FAILED
            return code
        for shape, smoke_body in run.smokes.items():
            print(f"[{model}] smoke {shape}: {smoke_body['violating_fraction']:.0%} violating "
                  f"windows, inside {smoke_body['band']}", flush=True)
        status, code = "complete", 0
        return 0
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        if result is None:
            result = run.result(status)
        (out_dir / ladder.DESIGN_RESULT).write_text(
            json.dumps({"mode": MODE, "run_manifest_sha256": plan.get("run_manifest_sha256"),
                        "models": [result]}, indent=2) + "\n", encoding="utf-8")
        campaign.finalize_run(out_dir, status=status, exit_code=code)

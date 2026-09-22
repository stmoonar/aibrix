#!/usr/bin/env python3
"""Drive the preregistered calibration design (:mod:`scripts.calibration_design`).

Entered from ``python -m scripts.calibration_campaign`` (its default ``--design``); it
reuses that module's per-cell machinery - the ``r3_grid`` invocation with the campaign's
strict failure rules, the guard read-back, the void rule, the provenance block and the
standard dataset build at the end - and replaces its stage order with the one the
preregistration fixes.

Order of one model's run
------------------------
1. sentinel 1 (fixed shape and rho, see ``calibration_design.SENTINEL_*``)
2. stage 0 - the prior-guided boundary search of **every** shape, round-robin over the
   shapes, so every shape has its rho* before the first ladder cell
3. stage 1 - the ladder, in its interleaved rounds; sentinel 2 after half of the rounds
4. stage 2 - one ramp per shape, in random order
5. stage 3 - supplementary cells, decided from the ladder's measured labels
6. sentinel 3; the drift verdict

Between cells
-------------
Every cell starts from an idle engine. After a cell, the engine's ``running + waiting``
(summed over the model's routable pods, read from their ``/metrics``) is polled until it
is 0, for at most ``calibration_design.DRAIN_LIMIT_S``; the fixed ``--cooldown-s`` is kept
as a floor. A gap that ran out without draining is recorded on the *next* cell as
``possibly_contaminated`` - its first windows may be measuring the previous cell's
backlog. The cell is still driven: dropping it would make contamination decide which
loads get sampled.

What is written
---------------
Before the first cell, ``run_manifest.json``: every fixed parameter of the
preregistration, the regime groups and rho priors (contents and hashes), the design
seed, the preregistration's commit and the static plan. It is written once, refused if
it already exists, and made read-only. During the run, one line per driven attempt in
``cells.jsonl`` (identity, seeds, load, files, guard summary, verdict, drain before it),
``boundary/<model>_<shape>.json`` per search and the schedules under ``schedules/``. At
the end ``design_result.json`` (anchors, supplement decisions, sentinel drift,
contaminated cells), ``campaign_status.json`` and the standard dataset.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

from scripts import adaptive_boundary as boundary
from scripts import admission_cap as admission
from scripts import calibration_campaign as campaign
from scripts import calibration_design as design
from scripts import gen_calibration_schedules as gen
from scripts import openloop

DESIGN_NAME = "ladder"
RUN_MANIFEST = "run_manifest.json"
LEDGER = "cells.jsonl"
DESIGN_RESULT = "design_result.json"
#: Seconds the driver spends between two cells besides the wait itself (process start,
#: pod discovery, prompt materialisation); the first round measured ~6 s.
DRIVER_OVERHEAD_S = 6.0
DEFAULT_DESIGN_SEED = 20260923

TRE_ROOT = Path(__file__).resolve().parents[2]


class CampaignStopped(RuntimeError):
    """The run cannot continue (a cell voided twice, or a driver failed under
    ``--stop-on-failure``)."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = int(code)


# --------------------------------------------------------------------------- checks


def check_args_against_preregistration(args) -> None:
    """The label parameters are fixed by the preregistration, not by the command line."""
    fixed = (
        ("--window-ms", args.window_ms, design.WINDOW_MS),
        ("--fit-step-ms", args.fit_step_ms, design.STEP_MS),
        ("--ttft-slo-ms", args.ttft_slo_ms, design.TTFT_SLO_MS),
        ("--tpot-slo-ms", args.tpot_slo_ms, design.TPOT_SLO_MS),
    )
    wrong = [f"{flag}={value} (preregistered {want})" for flag, value, want in fixed
             if float(value) != float(want)]
    if wrong:
        raise SystemExit("refusing to start: " + "; ".join(wrong))
    if not getattr(args, "rho_priors", None):
        raise SystemExit("refusing to start: the ladder design needs --rho-priors")
    if not getattr(args, "regime_groups", None):
        raise SystemExit("refusing to start: the ladder design needs --regime-groups")


def _git(*argv: str) -> Optional[str]:
    try:
        proc = subprocess.run(["git", "-C", str(TRE_ROOT), *argv], capture_output=True,
                              text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def preregistration_provenance(path: Path) -> dict:
    """The preregistration the run implements: path, content hash and its commit.

    Refuses to start when the document is missing or has never been committed - a
    preregistration that is not in the history is not a preregistration.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"refusing to start: preregistration {path} does not exist")
    commit = _git("log", "-1", "--format=%H", "--", str(path.resolve()))
    if not commit:
        raise SystemExit(
            f"refusing to start: preregistration {path} has no commit; commit it first"
        )
    modified = _git("status", "--porcelain", "--", str(path.resolve()))
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "commit": commit,
        "modified_since_commit": bool(modified),
    }


def write_frozen(path: Path, doc: dict) -> str:
    """Write ``doc`` once, read-only; refuse to overwrite. Returns its sha256."""
    path = Path(path)
    if path.exists():
        raise SystemExit(f"refusing to start: {path} already exists; a run manifest is "
                         "written once, before the first cell")
    body = json.dumps(doc, indent=2, sort_keys=False) + "\n"
    path.write_text(body, encoding="utf-8")
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------- drain


def wait_for_drain(
    sample: Callable[[], dict],
    *,
    limit_s: float = design.DRAIN_LIMIT_S,
    poll_s: float = design.DRAIN_POLL_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Poll the engine until ``running + waiting == 0``, for at most ``limit_s``.

    A poll only counts as drained when every pod answered: a scrape error is not a zero.
    """
    t0 = clock()
    polls = 0
    last: dict = {}
    while True:
        polls += 1
        try:
            last = dict(sample() or {})
        except Exception as exc:  # noqa: BLE001 - an unreadable engine is "not drained"
            last = {"error": repr(exc)}
        answered = (
            "error" not in last
            and float(last.get("pods_scraped", 0.0)) > 0
            and float(last.get("scrape_errors", 0.0)) == 0
        )
        outstanding = float(last.get("running", 0.0)) + float(last.get("waiting", 0.0))
        waited = clock() - t0
        if answered and outstanding == 0:
            return {"drained": True, "waited_s": round(waited, 3), "polls": polls,
                    "last": last}
        if waited >= limit_s:
            return {"drained": False, "waited_s": round(waited, 3), "polls": polls,
                    "last": last}
        sleep(min(poll_s, max(0.0, limit_s - waited)))


def make_engine_sampler(model: str, namespace: str, port: int = 8000) -> Callable[[], dict]:
    """``running`` / ``waiting`` summed over the model's routable pods, from their
    ``/metrics``. The pod list is re-read on every poll: a restarted pod has a new IP."""
    from scripts import r3_grid

    def sample() -> dict:
        endpoints = r3_grid.discover_pod_metrics_endpoints(model, namespace, port)
        return openloop.make_pod_metrics_sampler(endpoints)(int(time.time() * 1000))

    return sample


# ---------------------------------------------------------------------------- drive


def subprocess_drive(args, raw_dir: Path) -> Callable:
    """The real seam: one attempt of one cell through ``r3_grid``."""

    def drive(cell: design.DesignCell, attempt: int, schedule_path: Path, output: Path,
              prompt_dir: Path) -> tuple[list, dict, int]:
        grid_cell = campaign.Cell(
            model=cell.model, shape=cell.shape, primitive=cell.primitive,
            cell_id=cell.cell_id, schedule=str(schedule_path), duration_s=cell.duration_s,
            capacity_rps=0.0,
        )
        command = design_cell_command(grid_cell, cell, args, schedule_path, output, prompt_dir)
        print(f"  {cell.role} {cell.model}/{cell.shape} {cell.cell_id} attempt {attempt} "
              f"rho={cell.rho} ({cell.duration_s:.0f}s): {' '.join(command)}", flush=True)
        result = subprocess.run(command, check=False)
        rows = campaign.read_window_rows(output)
        guard = campaign.read_cell_guard(raw_dir, output, cell.cell_id,
                                         returncode=result.returncode)
        return rows, guard, result.returncode

    return drive


def design_cell_command(grid_cell, cell: design.DesignCell, args, schedule_path: Path,
                        output: Path, prompt_dir: Path) -> list[str]:
    """``campaign.cell_command`` plus what makes this cell independent of every other:
    its own arrival seed, its own prompt key, its own prompt file; and, for a boundary
    probe, the backlog valve."""
    command = campaign.cell_command(grid_cell, args, schedule_path, output)
    command[command.index("--prompt-dir") + 1] = str(prompt_dir)
    command += ["--schedule-seed", str(cell.arrival_seed), "--prompt-key", cell.prompt_key]
    if cell.role == design.ROLE_BOUNDARY:
        command += ["--max-backlog", str(design.PROBE_MAX_BACKLOG)]
    return command


# ------------------------------------------------------------------------------ run


class LadderRun:
    """One model's run of the design. Seams: ``drive``, ``sample``, ``sleep``, ``clock``."""

    def __init__(self, args, model: str, priors: design.RhoPriors, *, factory,
                 plan: design.StaticPlan, cap, out_dir: Path, raw_dir: Path,
                 drive: Callable, sample: Callable[[], dict],
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.args = args
        self.model = model
        self.priors = priors
        self.factory = factory
        self.plan = plan
        self.cap = cap
        self.out_dir = Path(out_dir)
        self.raw_dir = Path(raw_dir)
        self.drive = drive
        self.sample = sample
        self.sleep = sleep
        self.clock = clock
        self.rng = random.Random(design.derived_seed(factory.design_seed, model, "run-order"))
        self.anchors: dict[str, float] = {}
        self.anchor_sources: dict[str, str] = {}
        self.searches: dict[str, design.PriorGuidedSearch] = {}
        self.ladder_outcomes: dict[str, list[design.CellOutcome]] = {}
        self.supplement_decisions: list[dict] = []
        self.sentinel_summaries: list[dict] = []
        self.contaminated: list[dict] = []
        self.records: list[dict] = []
        self.schedule_dir = self.out_dir / "schedules" / model
        self.schedule_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "boundary").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ plumbing

    def gap(self) -> dict:
        """Wait for the engine to drain (bounded), with the cooldown as a floor."""
        t0 = self.clock()
        drain = wait_for_drain(self.sample, clock=self.clock, sleep=self.sleep)
        elapsed = self.clock() - t0
        floor = float(self.args.cooldown_s)
        if elapsed < floor:
            self.sleep(floor - elapsed)
        drain["gap_s"] = round(self.clock() - t0, 3)
        return drain

    def capacity(self, shape: str) -> float:
        return self.priors.get(self.model, shape).capacity_rps

    def drive_attempt(self, cell: design.DesignCell, attempt: int) -> tuple[list, dict, dict]:
        """Gap, then one attempt; appends its ledger line. Returns (rows, guard, record)."""
        drain = self.gap()
        body, meta = design.cell_schedule(
            cell, self.capacity(cell.shape), anchor_rho=self.anchors.get(cell.shape),
            cap=self.cap,
        )
        stem = cell.stem(attempt)
        schedule_path = self.schedule_dir / f"{stem}.json"
        schedule_path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        schedule_path.with_name(f"{stem}.meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        output = self.out_dir / f"{stem}.csv"
        prompt_dir = self.out_dir / "prompts" / stem
        started = campaign.utc_iso()
        rows, guard, returncode = self.drive(cell, attempt, schedule_path, output, prompt_dir)
        verdict: dict = {}
        if cell.profile == design.PROFILE_HOLD:
            verdict = design.hold_cell_verdict(
                rows, guard, warmup_s=cell.warmup_s,
                ttft_slo_ms=self.args.ttft_slo_ms, tpot_slo_ms=self.args.tpot_slo_ms,
            )
        contaminated = not drain["drained"]
        record = {
            **{k: v for k, v in cell.as_dict().items()},
            "attempt": attempt,
            "stem": stem,
            "anchor_rho": self.anchors.get(cell.shape),
            "capacity_rps": self.capacity(cell.shape),
            "offered_rps": meta.get("offered_rps"),
            "planned_requests": meta.get("planned_requests"),
            "schedule_path": str(schedule_path),
            "online_csv": str(output),
            "raw_dir": str(self.raw_dir / stem),
            "prompt_dir": str(prompt_dir),
            "driven_at_utc": started,
            "returncode": returncode,
            "start_ms": guard.get("start_ms"),
            "end_ms": guard.get("end_ms"),
            "void_reasons": list(guard.get("void_reasons") or []),
            "truncated": bool(guard.get("truncated")),
            "truncation_cause": guard.get("truncation_cause"),
            "sent": guard.get("sent"),
            "completed": guard.get("completed"),
            "client_timeouts": guard.get("client_timeouts"),
            "verdict": verdict.get("verdict"),
            "labeled_windows": verdict.get("labeled_windows"),
            "independent_windows": verdict.get("independent_windows"),
            "violating_windows": verdict.get("violating_windows"),
            "backlog_stopped": verdict.get("backlog_stopped", False),
            "drain_before": drain,
            "possibly_contaminated": contaminated,
            "contamination_reason": (
                f"engine not drained after {drain['waited_s']:.0f}s before this cell "
                f"(last poll: {json.dumps(drain.get('last'), sort_keys=True)})"
                if contaminated else ""
            ),
        }
        with (self.out_dir / LEDGER).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        self.records.append(record)
        if contaminated:
            self.contaminated.append({"cell_id": cell.cell_id, "attempt": attempt,
                                      "role": cell.role, "shape": cell.shape,
                                      "reason": record["contamination_reason"]})
            print(f"  WARNING: {record['contamination_reason']}; {cell.cell_id} is marked "
                  "possibly_contaminated", flush=True)
        if returncode != 0 and self.args.stop_on_failure and not record["void_reasons"]:
            raise CampaignStopped(f"cell {cell.cell_id} driver exited {returncode}", returncode)
        return rows, guard, record

    def drive_cell(self, cell: design.DesignCell) -> tuple[list, dict, dict]:
        """A scheduled cell: re-run once in place if void, stop the run on a second void
        (the campaign's one rule, ``adaptive_boundary.next_void_attempt``)."""
        attempt = 1
        while True:
            rows, guard, record = self.drive_attempt(cell, attempt)
            if not record["void_reasons"]:
                return rows, guard, record
            nxt = boundary.next_void_attempt(attempt)
            if nxt is None:
                raise CampaignStopped(str(campaign.CellVoided(
                    cell.cell_id, attempt, record["void_reasons"])))
            print(f"  cell {cell.cell_id} is VOID ({', '.join(record['void_reasons'])}); "
                  f"re-running it as attempt {nxt}", flush=True)
            attempt = nxt

    # -------------------------------------------------------------------- stages

    def run_sentinel(self, index: int) -> None:
        cell = self.plan.sentinels[index]
        print(f"[{self.model}] sentinel {index + 1} ({cell.position}): "
              f"{cell.shape} rho={cell.rho}", flush=True)
        rows, guard, _record = self.drive_cell(cell)
        summary = design.sentinel_summary(
            rows, guard, warmup_s=cell.warmup_s,
            ttft_slo_ms=self.args.ttft_slo_ms, tpot_slo_ms=self.args.tpot_slo_ms,
        )
        summary.update({"position": cell.position, "cell_id": cell.cell_id})
        self.sentinel_summaries.append(summary)

    def run_boundary_stage(self, shapes: Sequence[str]) -> None:
        """Stage 0, round-robin: every shape's next probe in turn until all are done."""
        order = list(shapes)
        self.rng.shuffle(order)
        for shape in order:
            self.searches[shape] = design.PriorGuidedSearch.from_prior(
                self.priors.get(self.model, shape))
        current: dict[str, design.DesignCell] = {}
        while True:
            progressed = False
            for shape in order:
                search = self.searches[shape]
                probe = search.next_probe()
                if probe is None:
                    continue
                progressed = True
                if probe.attempt > 1 and shape in current:
                    cell = current[shape]  # a re-drive is another attempt of the same cell
                    cell.duration_s = probe.duration_s
                else:
                    cell = self.factory.new(shape, design.ROLE_BOUNDARY, probe.duration_s,
                                            rho=probe.rho, stage=probe.stage)
                    current[shape] = cell
                print(f"[{self.model}] boundary {shape} {probe.stage} rho={probe.rho:g} "
                      f"attempt {probe.attempt}", flush=True)
                rows, guard, record = self.drive_attempt(cell, probe.attempt)
                if record["void_reasons"]:
                    result = boundary.ProbeResult(
                        probe=probe, verdict=boundary.VERDICT_VOID,
                        void_reasons=tuple(record["void_reasons"]), windows=len(rows),
                        cell_id=cell.cell_id)
                else:
                    result = boundary.ProbeResult(
                        probe=probe, verdict=record["verdict"], windows=len(rows),
                        labeled_windows=record["labeled_windows"] or 0,
                        independent_windows=record["independent_windows"] or 0,
                        violating_windows=record["violating_windows"] or 0,
                        cell_id=cell.cell_id)
                search.record(result)
            if not progressed:
                break
        for shape in order:
            search = self.searches[shape]
            if search.stopped_reason and search.last_verdict == boundary.VERDICT_VOID:
                self._write_search(shape)
                raise CampaignStopped(f"boundary search {self.model}/{shape}: "
                                      f"{search.stopped_reason}")
            anchor = search.anchor_rho
            if anchor is None:
                anchor = self.priors.get(self.model, shape).anchor_rho
                self.anchor_sources[shape] = (
                    f"prior anchor: the search produced no verdict ({search.stopped_reason})")
            else:
                self.anchor_sources[shape] = search.anchor_rule
            self.anchors[shape] = anchor
            self._write_search(shape)
            print(f"[{self.model}] {shape}: rho* anchor {anchor:g} "
                  f"({self.anchor_sources[shape]}; boundary_found={search.boundary_found})",
                  flush=True)

    def _write_search(self, shape: str) -> None:
        body = self.searches[shape].as_dict()
        body.update({
            "anchor_rho": self.anchors.get(shape),
            "anchor_source": self.anchor_sources.get(shape),
            "prior": self.priors.get(self.model, shape).as_dict(),
        })
        (self.out_dir / "boundary" / f"{self.model}_{shape}.json").write_text(
            json.dumps(body, indent=2) + "\n", encoding="utf-8")

    def run_ladder_rounds(self, rounds: Sequence[Sequence[design.DesignCell]]) -> None:
        for rnd in rounds:
            for cell in rnd:
                print(f"[{self.model}] ladder round {cell.round} {cell.shape} "
                      f"{cell.rho_factor} x rho* (replicate {cell.replicate})", flush=True)
                _rows, _guard, record = self.drive_cell(cell)
                self.ladder_outcomes.setdefault(cell.shape, []).append(
                    design.CellOutcome(cell.shape, float(cell.rho_factor), record["verdict"]))

    def run_ramps(self) -> None:
        for cell in self.plan.ramps:
            print(f"[{self.model}] ramp {cell.shape}", flush=True)
            self.drive_cell(cell)

    def run_supplement(self, shapes: Sequence[str]) -> None:
        outcomes = {shape: self.ladder_outcomes.get(shape, []) for shape in shapes}
        self.supplement_decisions = design.supplement_plan(outcomes)
        wanted = []
        for decision in self.supplement_decisions:
            for i, factor in enumerate(decision["supplement"], start=1):
                wanted.append((decision["shape"], factor, i))
        self.rng.shuffle(wanted)
        cells = [
            self.factory.new(shape, design.ROLE_SUPPLEMENT, design.SUPPLEMENT_SECONDS,
                             rho_factor=factor, replicate=i)
            for shape, factor, i in wanted
        ]
        design.anchor_cells(cells, self.anchors)
        for cell in cells:
            print(f"[{self.model}] supplement {cell.shape} {cell.rho_factor} x rho*",
                  flush=True)
            self.drive_cell(cell)

    def run(self, shapes: Sequence[str] = gen.ALL_SHAPES) -> None:
        self.run_sentinel(0)
        self.run_boundary_stage(shapes)
        design.anchor_cells(self.plan.ladder + self.plan.ramps, self.anchors)
        half = len(self.plan.ladder_rounds) // 2
        self.run_ladder_rounds(self.plan.ladder_rounds[:half])
        self.run_sentinel(1)
        self.run_ladder_rounds(self.plan.ladder_rounds[half:])
        self.run_ramps()
        self.run_supplement(shapes)
        self.run_sentinel(2)

    def result(self, status: str) -> dict:
        drift = design.sentinel_drift(self.sentinel_summaries)
        return {
            "model": self.model,
            "status": status,
            "anchors": dict(self.anchors),
            "anchor_sources": dict(self.anchor_sources),
            "boundary_found": {s: x.boundary_found for s, x in self.searches.items()},
            "supplement": self.supplement_decisions,
            "sentinels": self.sentinel_summaries,
            "sentinel_drift": drift,
            "drift_flagged": drift.get("flagged"),
            "possibly_contaminated_cells": self.contaminated,
            "attempts": len(self.records),
        }


# ------------------------------------------------------------------------ entry point


def build_plan_documents(args, models: Sequence[str], priors: design.RhoPriors,
                         groups: design.RegimeGroups, cap) -> tuple[dict, dict, dict]:
    """(plan.json, run manifest, {model: (factory, static plan)}) - no side effects."""
    seed = int(args.design_seed)
    plans = {}
    estimates = {}
    for model in models:
        factory = design.CellFactory(model, seed)
        plan = design.build_static_plan(model, priors, factory)
        plans[model] = (factory, plan)
        estimates[model] = design.estimate_model_seconds(
            priors.for_model(model), cooldown_s=args.cooldown_s,
            overhead_s=DRIVER_OVERHEAD_S)
    static_cells = {m: [c.as_dict() for c in p.all_cells()] for m, (_f, p) in plans.items()}
    manifest = {
        "design": DESIGN_NAME,
        "written_at_utc": campaign.utc_iso(),
        "preregistration": preregistration_provenance(Path(args.preregistration)),
        "preregistered": design.preregistered_parameters(),
        "regime_groups": groups.as_dict(),
        "rho_priors": priors.as_dict(),
        "design_seed": seed,
        "seed_derivation": (
            "arrival seed = sha256('<design_seed>|<model>|<cell_id>|arrivals')[:8] & 0x7fffffff; "
            "prompt key = 'p<design_seed>.<cell_id>' (prefixed onto every request id, "
            "which seeds its prompt); ladder/ramp/run order from "
            "random.Random(sha256('<design_seed>|<model>|ladder-order' / 'run-order'))"
        ),
        "cell_id_rule": (
            f"i<in>_o<out>_c<code>, code = {design.CELL_CODE_BASE} + model index x "
            f"{design.MODEL_CODE_BLOCK} + serial (model index in {list(gen.MODELS)})"
        ),
        "models": list(models),
        "cooldown_floor_s": args.cooldown_s,
        "static_plan": static_cells,
        "sentinel_rho": {m: p.sentinel_rho for m, (_f, p) in plans.items()},
        "provenance": campaign.run_provenance(args),
        "admission_cap": cap.as_dict(),
    }
    plan_doc = {
        "design": DESIGN_NAME,
        "generated_at_utc": campaign.utc_iso(),
        "provenance": campaign.run_provenance(args),
        "admission_cap": cap.as_dict(),
        "models": list(models),
        "design_seed": seed,
        "cooldown_s": args.cooldown_s,
        "run_manifest": RUN_MANIFEST,
        "static_cells": static_cells,
        "estimate": estimates,
    }
    return plan_doc, manifest, plans


def print_estimate(plan_doc: dict) -> None:
    for model, est in plan_doc["estimate"].items():
        print(f"{model}: expected {est['seconds_expected'] / 3600:.2f} h "
              f"({est['cells_expected']} cells), upper {est['seconds_upper'] / 3600:.2f} h "
              f"({est['cells_upper']} cells); gap {est['gap_expected_s']:.0f}-"
              f"{est['gap_upper_s']:.0f} s per cell")
        for name, st in est["stages"].items():
            print(f"    {name:10} {st['cells_expected']:4d} cells {st['seconds_expected'] / 60:7.1f} min"
                  f"   (upper {st['cells_upper']:4d} cells {st['seconds_upper'] / 60:7.1f} min)"
                  + (f"  - {st['note']}" if st["note"] else ""))


def run_ladder_campaign(args, *, drive: Optional[Callable] = None,
                        sample_factory: Optional[Callable[[str], Callable]] = None,
                        sleep: Callable[[float], None] = time.sleep,
                        clock: Callable[[], float] = time.monotonic,
                        check_controller: bool = True) -> int:
    check_args_against_preregistration(args)
    models = [m for m in args.models.split(",") if m]
    priors = design.load_rho_priors(Path(args.rho_priors), models=models)
    groups = design.load_regime_groups(Path(args.regime_groups), models=models)
    index_cap = None
    if args.index and Path(args.index).exists():
        index_cap = (json.loads(Path(args.index).read_text(encoding="utf-8"))
                     .get("admission_cap", {}).get("name"))
    cap = admission.get_cap(args.cap or index_cap or admission.DEFAULT_CAP_NAME)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)
    plan_doc, manifest, plans = build_plan_documents(args, models, priors, groups, cap)
    print_estimate(plan_doc)
    if args.dry_run:
        plan_doc["run_manifest_preview"] = manifest
        (out_dir / "plan.json").write_text(json.dumps(plan_doc, indent=2) + "\n",
                                           encoding="utf-8")
        print(f"dry run: wrote {out_dir / 'plan.json'} (the run manifest is only written "
              "when a run starts)")
        return 0

    if check_controller:
        mode = campaign.controller_mode(args.controller_namespace)
        if mode != campaign.REQUIRED_CONTROLLER_MODE:
            raise SystemExit(f"controller mode is {mode!r}, refusing to run (need "
                             f"{campaign.REQUIRED_CONTROLLER_MODE!r})")
        print(f"controller mode: {mode}")
    manifest_sha = write_frozen(out_dir / RUN_MANIFEST, manifest)
    plan_doc["run_manifest_sha256"] = manifest_sha
    (out_dir / "plan.json").write_text(json.dumps(plan_doc, indent=2) + "\n", encoding="utf-8")

    drive = drive or subprocess_drive(args, raw_dir)
    sample_factory = sample_factory or (
        lambda model: make_engine_sampler(model, args.model_namespace))
    status, code = "failed", 1
    results = []
    try:
        for model in models:
            factory, plan = plans[model]
            run = LadderRun(args, model, priors, factory=factory, plan=plan, cap=cap,
                            out_dir=out_dir, raw_dir=raw_dir, drive=drive,
                            sample=sample_factory(model), sleep=sleep, clock=clock)
            try:
                run.run()
                results.append(run.result("complete"))
            except CampaignStopped as stop:
                print(f"STOPPED: {stop}", flush=True)
                results.append(run.result(f"stopped: {stop}"))
                code = stop.code or 1
                status = "stopped"
                return code
        status, code = "complete", 0
        return 0
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        (out_dir / DESIGN_RESULT).write_text(
            json.dumps({"run_manifest_sha256": manifest_sha, "models": results},
                       indent=2) + "\n", encoding="utf-8")
        for result in results:
            print(f"[{result['model']}] {result['status']}; sentinel drift: "
                  f"{result['sentinel_drift']['verdict']}; "
                  f"{len(result['possibly_contaminated_cells'])} possibly contaminated cell(s)")
        campaign.finalize_run(out_dir, status=status, exit_code=code)

#!/usr/bin/env python3
"""Runner for the open-loop calibration campaign: steps, then ramp, then bursts.

The ordering is the point
-------------------------
Each (model, shape) runs its **steps** cell first, and the ramp for that shape is then
**regenerated from the capacity the steps cell measured** rather than from the fitted
prior. The prior comes from a two-parameter ``1/C = i/P + o/D`` surface with a 10-29 %
rms relative error per model, and rho is defined against it: a 25 % error in ``C_s``
moves the ramp's peak offered load by 25 %, which on its own is enough to double or
halve the backlog the cell accumulates. Running the cheap, monotone, steady-state
primitive first turns that guess into a measurement before the expensive one spends it.

Bursts run last, and only on the shapes that can reach the queue at all - see
:mod:`scripts.admission_cap`. A burst has to overshoot the *engine's* running limit,
because ``vllm:num_requests_waiting`` only moves when the engine queues; if the requests
needed to do that exceed what the gateway will admit, the spike is shed at the Envoy
circuit breaker and the cell would record a flat zero at full cost.

What the gateway does to a cell
-------------------------------
The deployed BackendTrafficPolicy admits ``maxParallelRequests: 256`` +
``maxPendingRequests: 64`` **per Envoy cluster, shared across every replica**. A
pre-check against the live gateway on 2026-09-20 opened requests until they were
refused: the first 503 arrived at in-flight 321 and every one of them carried

    HTTP/1.1 503 Service Unavailable
    content-type: text/plain
    upstream connect error or disconnect/reset before headers. reset reason: overflow

with no ``x-envoy-*`` header at all. That is why
:func:`scripts.openloop.classify_failure` cannot key on the headers, and why a shed is
not counted against the model's error budget: it never reached vLLM. The first shed
instead truncates the cell to its drain segment, because everything after it measures
the circuit breaker rather than the engine.

Artifacts
---------
Per cell, under ``--raw-dir``: the per-request raw JSONL, the 1 Hz instant sidecar, the
classified failures, and a guard JSON recording truncation and censoring. Per campaign,
under ``--out-dir``: ``plan.json`` (every cell, in order, with its provenance),
``capacity/<model>_<shape>.json`` (prior vs measured), the regenerated ramp schedules,
and ``fit_plan.json`` - the re-windowing and refit invocations the capture is meant to
be consumed by, including the cadence each one must use.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from scripts import admission_cap as admission
from scripts import gen_calibration_schedules as gen
from scripts.openloop import LIVE_GRID_MS

#: Primitives in the order a model runs them. Steps first because it measures the
#: capacity the ramp is then defined against; bursts last because they are the only
#: primitive that can be skipped outright, so a truncated campaign still has the two
#: primitives every shape needs.
STAGE_ORDER = ("steps", "ramp", "bursts")

#: Seconds of each step level treated as transient and excluded from the capacity
#: measurement. A level has to reach steady state before its throughput means anything.
DEFAULT_STEP_TRANSIENT_S = 60.0

#: A level counts as saturated when it delivered materially less than it was offered.
#: Below this ratio the engine is the thing limiting throughput, not the schedule.
SATURATION_THROUGHPUT_RATIO = 0.9

#: Quiet time between cells, so one cell's backlog cannot leak into the next one's
#: first window.
DEFAULT_COOLDOWN_S = 45.0

#: The controller must not be acting on the fleet while capacity is being measured:
#: a scale action mid-cell changes the denominator of everything the cell records.
REQUIRED_CONTROLLER_MODE = "observe"
CONTROLLER_MODE_KEY = "tre:v2:controller:mode"

#: lambda_wait for the primary fit, inherited from the registry.
PRIMARY_LAMBDA_WAIT = 3.0
#: The control fit. If theta and the ranking separation move by less than
#: SECONDARY_FIT_TOLERANCE between the two, the waiting term contributed nothing and the
#: honest statement is that it is inert in this deployment - because gateway admission
#: caps in-flight below the engine's sequence limit, so the queue it multiplies is
#: almost always zero.
SECONDARY_LAMBDA_WAIT = 0.0
SECONDARY_FIT_TOLERANCE = 0.05


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- capacity


@dataclass(frozen=True)
class StepLevel:
    """One step level's steady-state behaviour."""

    rho: float
    start_s: float
    end_s: float
    steady_start_s: float
    offered_rps: float
    achieved_rps: float
    completions: int
    p95_ttft_ms: Optional[float]
    p95_tpot_ms: Optional[float]
    slo_met: bool
    saturated: bool


@dataclass(frozen=True)
class MeasuredCapacity:
    """What a steps cell said about a shape's capacity, and what the ramp will use."""

    model: str
    shape: str
    capacity_prior_rps: float
    capacity_measured_rps: float
    capacity_used_rps: float
    capacity_source: str
    saturated: bool
    ttft_slo_ms: float
    tpot_slo_ms: float
    levels: tuple[StepLevel, ...]
    note: str

    def as_dict(self) -> dict:
        body = asdict(self)
        body["levels"] = [asdict(level) for level in self.levels]
        return body


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Nearest-rank percentile over per-request values.

    Deliberately not the histogram percentile the window CSV carries: this reads the raw
    records directly, so there are no buckets to interpolate and nearest-rank is exact.
    """
    usable = sorted(float(v) for v in values if v is not None)
    if not usable:
        return None
    index = max(0, math.ceil(q * len(usable)) - 1)
    return usable[index]


def measure_capacity_from_steps(
    records: Sequence[dict],
    *,
    levels: Sequence[tuple[float, float]],
    capacity_prior_rps: float,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    model: str = "",
    shape: str = "",
    transient_s: float = DEFAULT_STEP_TRANSIENT_S,
) -> MeasuredCapacity:
    """Saturating capacity of one shape, from its steps cell's raw records.

    ``levels`` is ``[(rho, duration_s), ...]`` in schedule order. The definition matches
    the one the priors were fitted with (``r3_capacity``): the highest sustained request
    rate that still met the SLO. Only the steady part of each level counts - the first
    ``transient_s`` are the queue adjusting to the step, not the operating point.

    Two outcomes, and the difference matters:

    * some level saturated (missed the SLO, or delivered materially less than it was
      offered). The prior was an over-estimate; the measurement replaces it.
    * no level saturated. Offered load topped out below capacity, so all the cell proves
      is ``C >= 0.95 * prior`` - a lower bound, not a measurement. Replacing the prior
      with it would shrink every downstream rho by the arbitrary amount the cell happened
      to stop short by, so the prior is kept and the result is marked censored.

    The second case is reported rather than silently patched: a prior that under-estimates
    by more than 20 % would leave the regenerated ramp's peak (rho 1.2) below the true
    capacity, and the cell would never cross the violation boundary. That is a campaign
    design question, not something this function should paper over.
    """
    if not levels:
        raise ValueError("steps cell has no levels")
    served = [r for r in records if _is_served(r)]
    origin_ms = min((int(r["send_ts_ms"]) for r in records), default=None)
    if origin_ms is None:
        raise ValueError(f"{model}/{shape}: steps cell produced no requests at all")

    measured_levels: list[StepLevel] = []
    start_s = 0.0
    for rho, duration_s in levels:
        end_s = start_s + float(duration_s)
        steady_start_s = start_s + min(transient_s, float(duration_s) / 2.0)
        steady_s = end_s - steady_start_s
        window = [
            r
            for r in served
            if steady_start_s <= (int(r["done_ts_ms"]) - origin_ms) / 1000.0 < end_s
        ]
        achieved = len(window) / steady_s if steady_s > 0 else 0.0
        p95_ttft = _percentile([r.get("ttft_ms") for r in window], 0.95)
        p95_tpot = _percentile([r.get("tpot_ms") for r in window], 0.95)
        offered = float(rho) * capacity_prior_rps
        slo_met = (
            bool(window)
            and p95_ttft is not None
            and p95_tpot is not None
            and p95_ttft <= ttft_slo_ms
            and p95_tpot <= tpot_slo_ms
        )
        saturated = (not slo_met) or achieved < SATURATION_THROUGHPUT_RATIO * offered
        measured_levels.append(
            StepLevel(
                rho=float(rho),
                start_s=start_s,
                end_s=end_s,
                steady_start_s=steady_start_s,
                offered_rps=round(offered, 4),
                achieved_rps=round(achieved, 4),
                completions=len(window),
                p95_ttft_ms=None if p95_ttft is None else round(p95_ttft, 3),
                p95_tpot_ms=None if p95_tpot is None else round(p95_tpot, 3),
                slo_met=slo_met,
                saturated=saturated,
            )
        )
        start_s = end_s

    met = [level.achieved_rps for level in measured_levels if level.slo_met]
    measured = max(met) if met else 0.0
    saturated = any(level.saturated for level in measured_levels)

    if saturated and measured > 0.0:
        used = measured
        source = "measured_steps"
        note = (
            f"a step level saturated, so the prior {capacity_prior_rps:.3f} rps was an "
            f"over-estimate; the ramp is regenerated from the measured "
            f"{measured:.3f} rps"
        )
    elif measured > 0.0:
        used = capacity_prior_rps
        source = "prior_censored"
        top_rho = max(rho for rho, _d in levels)
        note = (
            f"no step level saturated, so the cell only proves C >= {measured:.3f} rps "
            f"(offered at most rho={top_rho}); the prior {capacity_prior_rps:.3f} rps is "
            f"kept because it is not an over-estimate, and the measurement is censored "
            f"from above. If the prior under-estimates by more than 20 %, the regenerated "
            f"ramp peak (rho 1.2) stays below capacity and will not cross the violation "
            f"boundary - re-measure the prior rather than trusting this ramp"
        )
    else:
        used = capacity_prior_rps
        source = "prior_censored"
        note = (
            "no level met the SLO with any completions, so nothing was measured; the "
            "prior is kept and this cell needs investigation before its ramp is trusted"
        )

    return MeasuredCapacity(
        model=model,
        shape=shape,
        capacity_prior_rps=round(capacity_prior_rps, 4),
        capacity_measured_rps=round(measured, 4),
        capacity_used_rps=round(used, 4),
        capacity_source=source,
        saturated=saturated,
        ttft_slo_ms=ttft_slo_ms,
        tpot_slo_ms=tpot_slo_ms,
        levels=tuple(measured_levels),
        note=note,
    )


def _is_served(record: dict) -> bool:
    status = record.get("http_status")
    return (
        status is not None
        and 200 <= int(status) < 300
        and record.get("done_ts_ms") is not None
    )


def read_raw_records(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    records = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ------------------------------------------------------------------------------ plan


@dataclass
class Cell:
    """One schedule to drive, in campaign order."""

    model: str
    shape: str
    primitive: str
    cell_id: str
    schedule: str
    duration_s: float
    capacity_rps: float
    capacity_source: str = "prior_fit"
    drain_start_s: Optional[float] = None
    regenerate_from_measured: bool = False
    skipped: bool = False
    skip_reason: str = ""
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def build_plan(index: dict, models: Sequence[str]) -> tuple[list[Cell], list[Cell]]:
    """(cells to run, cells skipped) in campaign order: per model, steps -> ramp -> bursts."""
    by_key = {}
    skipped: list[Cell] = []
    for entry in index.get("schedules", []):
        key = (entry["model"], entry["shape"], entry["primitive"])
        by_key[key] = entry

    runnable: list[Cell] = []
    for model in models:
        for primitive in STAGE_ORDER:
            for shape in list(gen.SHAPES) + [gen.MIXTURE_NAME]:
                entry = by_key.get((model, shape, primitive))
                if entry is None:
                    continue
                cell = Cell(
                    model=model,
                    shape=shape,
                    primitive=primitive,
                    cell_id=entry.get("cell_id", ""),
                    schedule=str(entry.get("path", "")),
                    duration_s=float(entry.get("duration_s") or 0.0),
                    capacity_rps=float(entry.get("capacity_rps") or 0.0),
                    capacity_source=str(entry.get("capacity_source") or "prior_fit"),
                    drain_start_s=(
                        None if entry.get("drain_start_s") is None
                        else float(entry["drain_start_s"])
                    ),
                    regenerate_from_measured=(primitive == "ramp"),
                    skipped=bool(entry.get("skipped")),
                    skip_reason=str(entry.get("reason", "")),
                    metadata=entry,
                )
                (skipped if cell.skipped else runnable).append(cell)
    return runnable, skipped


def estimate_wall_clock_s(cells: Sequence[Cell], cooldown_s: float) -> float:
    return sum(cell.duration_s + cooldown_s for cell in cells)


def kv_cache_tokens_by_model(index: dict) -> dict:
    """Each model's KV cache size, from wherever the schedule index recorded it.

    The bursts entries carry it because they are sized against it; the capacity-model
    block carries it when the generator put it there. Reading both means a campaign whose
    bursts were all skipped still knows the number, which is exactly the campaign that
    most needs to explain why.
    """
    tokens: dict = {}
    for model, body in (index.get("capacity_models") or {}).items():
        if isinstance(body, dict) and body.get("kv_cache_tokens"):
            tokens[model] = int(body["kv_cache_tokens"])
    for entry in index.get("schedules", []):
        model = entry.get("model")
        if model and model not in tokens and entry.get("kv_cache_tokens"):
            tokens[model] = int(entry["kv_cache_tokens"])
    return tokens


# --------------------------------------------------------------------------- the run


def controller_mode(namespace: str = "tre-v2") -> str:
    """Read-only check that the controller is not acting on the fleet."""
    result = subprocess.run(
        ["kubectl", "-n", namespace, "exec", "deploy/tre-v2-redis", "--",
         "redis-cli", "get", CONTROLLER_MODE_KEY],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"could not read {CONTROLLER_MODE_KEY}: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def cell_command(cell: Cell, args, schedule_path: Path, output: Path) -> list[str]:
    command = [
        sys.executable, "-m", "scripts.r3_grid",
        "--model", cell.model,
        "--gateway-url", args.gateway_url,
        "--schedule", str(schedule_path),
        "--cell-id", cell.cell_id,
        "--output", str(output),
        "--raw-dir", str(args.raw_dir),
        "--window-ms", str(args.window_ms),
        "--instant-sample-ms", str(args.instant_sample_ms),
        "--namespace", args.model_namespace,
        "--guard-mode", args.guard_mode,
        "--min-slo-windows", str(args.min_slo_windows),
    ]
    if cell.drain_start_s is not None:
        command += ["--drain-start-s", str(cell.drain_start_s)]
    if args.registry:
        command += ["--registry", args.registry]
    if args.redis_url:
        command += ["--redis-url", args.redis_url]
    return command


def regenerate_ramp(
    cell: Cell,
    measured: MeasuredCapacity,
    *,
    cap: admission.AdmissionCap,
    kv_cache_tokens: int,
    out_dir: Path,
) -> Path:
    """Write this shape's ramp schedule at the measured capacity, outside the committed
    tree. The committed set stays the prior-derived one; what actually ran is recorded
    here, next to the measurement that produced it."""
    body, meta = gen.build_schedule_from_capacity_rps(
        cell.model,
        cell.shape,
        "ramp",
        measured.capacity_used_rps,
        kv_cache_tokens=kv_cache_tokens or None,
        capacity_source=measured.capacity_source,
        cap=cap,
    )
    if body is None:  # only the bursts primitive can be unreachable; a ramp never is
        raise SystemExit(
            f"{cell.model}/{cell.shape}: the regenerated ramp was declined by the "
            f"schedule generator: {meta.get('reason', 'no reason given')}"
        )
    path = out_dir / cell.model / f"{cell.shape}_ramp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    (path.parent / f"{cell.shape}_ramp.meta.json").write_text(
        json.dumps({"schedule": meta, "capacity": measured.as_dict()}, indent=2) + "\n",
        encoding="utf-8",
    )
    cell.drain_start_s = meta.get("drain_start_s", cell.drain_start_s)
    cell.duration_s = float(meta.get("duration_s") or cell.duration_s)
    cell.capacity_source = measured.capacity_source
    return path


def fit_plan(models: Sequence[str], out_dir: Path, raw_dir: Path, args) -> dict:
    """The re-windowing and refit invocations this capture is meant to be consumed by.

    Two things are pinned here rather than left to whoever runs the fit.

    *Cadence.* theta is a decision threshold on the signal the controller actually
    consumes, and the controller sees the gateway's 10 s boundary-aligned grid, not the
    campaign's 1 Hz sidecar. So the fitting re-window runs ``--instant-grid live
    --instant-sample-ms 10000``; the 1 Hz stream is re-windowed separately for the
    aliasing figure and for the observability-gap metric - the fraction of 1 Hz
    threshold crossings the 10 s grid never saw. Passing the wrong
    ``--instant-sample-ms`` scales every queue average by 10x, which is why
    ``rewindow_from_raw`` now refuses a mismatch instead of rescaling silently.

    *lambda_wait.* The primary fit keeps the inherited 3.0. A secondary fit at 0.0 says
    whether the waiting term did anything: if theta and the ranking separation move by
    less than 5 %, the term is inert in this deployment, and the reason is structural -
    gateway admission caps in-flight below the engine's sequence limit, so the queue the
    term multiplies is almost always zero. That is a finding to report, not a knob to
    tune away.
    """
    fit_dir = out_dir / "fit"
    plan = {
        "generated_at_utc": utc_iso(),
        "window_ms": args.window_ms,
        "step_ms": args.fit_step_ms,
        "rewindow": [],
        "refit": [],
        "acceptance": {
            "primary_lambda_wait": PRIMARY_LAMBDA_WAIT,
            "secondary_lambda_wait": SECONDARY_LAMBDA_WAIT,
            "tolerance": SECONDARY_FIT_TOLERANCE,
            "statement": (
                "if theta and the ranking separation move by less than "
                f"{SECONDARY_FIT_TOLERANCE:.0%} between the two fits, the waiting term is "
                "inert in this deployment because gateway admission caps in-flight below "
                "the engine's sequence limit"
            ),
        },
    }
    for model in models:
        fitting_csv = fit_dir / f"{model}_fitting.csv"
        aliasing_csv = fit_dir / f"{model}_aliasing.csv"
        plan["rewindow"].append({
            "purpose": "fitting (the signal the controller consumes)",
            "model": model,
            "output": str(fitting_csv),
            "command": [
                sys.executable, "-m", "scripts.rewindow_from_raw",
                "--model", model, "--raw-dir", str(raw_dir),
                "--output", str(fitting_csv),
                "--window-ms", str(args.window_ms), "--step-ms", str(args.fit_step_ms),
                "--instant-grid", "live",
                "--instant-sample-ms", str(LIVE_GRID_MS),
            ],
        })
        plan["rewindow"].append({
            "purpose": "aliasing figure and observability gap (ground truth)",
            "model": model,
            "output": str(aliasing_csv),
            "command": [
                sys.executable, "-m", "scripts.rewindow_from_raw",
                "--model", model, "--raw-dir", str(raw_dir),
                "--output", str(aliasing_csv),
                "--window-ms", str(args.window_ms), "--step-ms", str(args.fit_step_ms),
                "--instant-grid", "raw",
                "--instant-sample-ms", str(args.instant_sample_ms),
            ],
        })
        for label, lambda_wait in (
            ("primary", PRIMARY_LAMBDA_WAIT),
            ("secondary", SECONDARY_LAMBDA_WAIT),
        ):
            plan["refit"].append({
                "label": label,
                "model": model,
                "lambda_wait": lambda_wait,
                "command": [
                    sys.executable, "-m", "scripts.refit_trs_params",
                    "--input", str(fitting_csv),
                    "--model-name", model,
                    "--output", str(fit_dir / f"{model}_refit_{label}.json"),
                    "--inherited-lambda-wait", str(lambda_wait),
                    "--lambda-wait-candidates", str(lambda_wait),
                ],
            })
    return plan


def run_campaign(args) -> int:
    index_path = Path(args.index)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    cap = admission.get_cap(args.cap or index.get("admission_cap", {}).get("name")
                            or admission.DEFAULT_CAP_NAME)
    models = [m for m in args.models.split(",") if m]
    runnable, skipped = build_plan(index, models)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)
    schedule_root = index_path.parent

    plan_doc = {
        "generated_at_utc": utc_iso(),
        "admission_cap": cap.as_dict(),
        "index": str(index_path),
        "models": models,
        "stage_order": list(STAGE_ORDER),
        "cooldown_s": args.cooldown_s,
        "cells": [cell.as_dict() for cell in runnable],
        "skipped": [
            {"model": c.model, "shape": c.shape, "primitive": c.primitive,
             "reason": c.skip_reason}
            for c in skipped
        ],
        "estimated_wall_clock_s": round(estimate_wall_clock_s(runnable, args.cooldown_s), 1),
    }
    (out_dir / "plan.json").write_text(json.dumps(plan_doc, indent=2) + "\n", encoding="utf-8")
    (out_dir / "fit_plan.json").write_text(
        json.dumps(fit_plan(models, out_dir, raw_dir, args), indent=2) + "\n", encoding="utf-8"
    )

    hours = plan_doc["estimated_wall_clock_s"] / 3600.0
    print(f"admission cap: {cap.name} (shed ceiling {cap.shed_ceiling}, "
          f"admission controller: {cap.admission_controller})")
    print(f"{len(runnable)} cells to run, {len(skipped)} skipped, "
          f"~{hours:.2f} h wall clock including {args.cooldown_s:.0f}s cooldowns")
    for cell in runnable:
        print(f"  {cell.model:12} {cell.shape:3} {cell.primitive:7} "
              f"{cell.duration_s:6.0f}s  C_s={cell.capacity_rps:7.3f}")
    for cell in skipped:
        print(f"  SKIP {cell.model:12} {cell.shape:3} {cell.primitive:7}  {cell.skip_reason}")

    if args.dry_run:
        print(f"dry run: wrote {out_dir / 'plan.json'} and {out_dir / 'fit_plan.json'}")
        return 0

    mode = controller_mode(args.controller_namespace)
    if mode != REQUIRED_CONTROLLER_MODE:
        raise SystemExit(
            f"controller mode is {mode!r}, refusing to run: a scale action mid-cell "
            f"changes the denominator of everything the cell records (need "
            f"{REQUIRED_CONTROLLER_MODE!r})"
        )
    print(f"controller mode: {mode}")

    measured_dir = out_dir / "capacity"
    measured_dir.mkdir(parents=True, exist_ok=True)
    regenerated_dir = out_dir / "schedules"
    measurements: dict[tuple[str, str], MeasuredCapacity] = {}
    step_levels = [
        (float(level["rho"]), float(level["duration_s"]))
        for level in index["primitives"]["steps"]["levels"]
    ]
    kv_tokens = kv_cache_tokens_by_model(index)

    for position, cell in enumerate(runnable, start=1):
        schedule_path = schedule_root / cell.schedule
        if cell.primitive == "ramp":
            measurement = measurements.get((cell.model, cell.shape))
            if measurement is None:
                print(f"[{position}/{len(runnable)}] SKIP ramp {cell.model}/{cell.shape}: "
                      "its steps cell did not produce a capacity measurement")
                continue
            schedule_path = regenerate_ramp(
                cell, measurement, cap=cap,
                kv_cache_tokens=kv_tokens.get(cell.model, 0),
                out_dir=regenerated_dir,
            )
            print(f"regenerated ramp from {measurement.capacity_source}: "
                  f"C_s {measurement.capacity_prior_rps} -> {measurement.capacity_used_rps} rps")

        output = out_dir / f"{cell.model}_{cell.shape}_{cell.primitive}.csv"
        command = cell_command(cell, args, schedule_path, output)
        print(f"[{position}/{len(runnable)}] {cell.model} {cell.shape} {cell.primitive} "
              f"({cell.duration_s:.0f}s): {' '.join(command)}")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"cell failed with exit {result.returncode}")
            if args.stop_on_failure:
                return result.returncode

        if cell.primitive == "steps":
            raw_path = raw_dir / output.stem / f"{cell.cell_id}.jsonl"
            records = read_raw_records(raw_path)
            if not records:
                print(f"no raw records at {raw_path}; cannot measure capacity")
            else:
                measurement = measure_capacity_from_steps(
                    records,
                    levels=step_levels,
                    capacity_prior_rps=cell.capacity_rps,
                    ttft_slo_ms=args.ttft_slo_ms,
                    tpot_slo_ms=args.tpot_slo_ms,
                    model=cell.model,
                    shape=cell.shape,
                    transient_s=args.step_transient_s,
                )
                measurements[(cell.model, cell.shape)] = measurement
                (measured_dir / f"{cell.model}_{cell.shape}.json").write_text(
                    json.dumps(measurement.as_dict(), indent=2) + "\n", encoding="utf-8"
                )
                print(f"measured C_s: prior {measurement.capacity_prior_rps} -> "
                      f"used {measurement.capacity_used_rps} rps "
                      f"({measurement.capacity_source}); {measurement.note}")

        if position < len(runnable):
            time.sleep(args.cooldown_s)

    print(f"campaign complete; artifacts under {out_dir}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    here = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path,
                    default=here / "replayer/traces_v2/calibration/INDEX.json")
    ap.add_argument("--models", default=",".join(gen.MODELS))
    ap.add_argument("--gateway-url", default="http://192.168.223.76:31094/v1/completions")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, default=Path("/root/tre-experiments/calibration_raw"))
    ap.add_argument("--cap", default=None,
                    help="admission policy to plan against (default: the one the schedule "
                         "index was generated for)")
    ap.add_argument("--window-ms", type=int, default=30000)
    ap.add_argument("--fit-step-ms", type=int, default=5000,
                    help="slide step for the fitting re-window (the live refresh cadence)")
    ap.add_argument("--instant-sample-ms", type=int, default=1000)
    ap.add_argument("--cooldown-s", type=float, default=DEFAULT_COOLDOWN_S)
    ap.add_argument("--step-transient-s", type=float, default=DEFAULT_STEP_TRANSIENT_S)
    ap.add_argument("--ttft-slo-ms", type=float, default=500.0)
    ap.add_argument("--tpot-slo-ms", type=float, default=75.0)
    ap.add_argument("--min-slo-windows", type=int, default=3)
    ap.add_argument("--guard-mode", default="warn", choices=["fail", "warn"],
                    help="a failed cell should not abandon the campaign by default; the "
                         "guard verdict is recorded per cell either way")
    ap.add_argument("--stop-on-failure", action="store_true")
    ap.add_argument("--registry", default=None)
    ap.add_argument("--redis-url", default=None)
    ap.add_argument("--model-namespace", default="default")
    ap.add_argument("--controller-namespace", default="tre-v2")
    ap.add_argument("--dry-run", action="store_true",
                    help="write plan.json and fit_plan.json, drive nothing")
    args = ap.parse_args(argv)
    return run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())

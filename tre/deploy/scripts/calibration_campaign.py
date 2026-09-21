#!/usr/bin/env python3
"""Runner for the open-loop calibration campaign.

Per (model, shape), in order: **steps**, then an **adaptive boundary search**, then
**ramp**, then **bursts**.

The ordering is the point
-------------------------
The **steps** cell runs first because everything downstream is expressed in rho, and rho
is relative to a capacity prior fitted from a two-parameter ``1/C = i/P + o/D`` surface
with a 10-29 % rms relative error. A 25 % error in ``C_s`` moves every later cell's
offered load by 25 %. Running the cheap, monotone, steady-state primitive first turns
that guess into a measurement before the expensive cells spend it.

The **boundary search** (:mod:`scripts.adaptive_boundary`) then locates the offered load
at which the shape actually starts violating its SLO, and dwells just under it. This
replaces the fixed rho grid: theta is a threshold on the signal at the moment the SLO
breaks, so windows taken far from that moment - on either side - barely constrain it,
and a grid centred on a prior with a 25 % error is centred on a guess. Three stages,
about 12 minutes of offered load per shape: coarse 3 x 60 s, bisect 2 x 120 s, dwell
300 s at ``0.95 rho*``.

The **ramp** is regenerated from the measured capacity rather than from the prior, and
**bursts** run last because they are the only primitive that can be skipped outright, so
a truncated campaign still has everything else.

What the gateway does to a cell
-------------------------------
Since 2026-09-21 the BackendTrafficPolicy admits ``maxParallelRequests: 4096`` +
``maxPendingRequests: 1024`` (identically on both experiment arms - see
``deploy/gateway-hardening/README.md``), and every model passes ``--max-num-seqs 256``.
The real admission ceiling is therefore the engine's ``max_num_seqs * replicas``, which
**grows when TRE scales out**, and no campaign cell comes near the Envoy limits.

*Superseded, kept because its evidence still matters.* Until 2026-09-21 the policy was
``maxParallelRequests: 256`` + ``maxPendingRequests: 64`` per Envoy cluster, shared
across every replica. A pre-check on 2026-09-20 opened requests until they were refused:
the first 503 arrived at in-flight 321 and every one of them carried

    HTTP/1.1 503 Service Unavailable
    content-type: text/plain
    upstream connect error or disconnect/reset before headers. reset reason: overflow

with no ``x-envoy-*`` header at all. **That 320 ceiling is no longer deployed**, and any
number fitted against it - including theta 1718 / 1494 / 1414 - is invalid by
construction. What survives from the measurement is the classifier's constraint:
:func:`scripts.openloop.classify_failure` cannot key on Envoy headers, because a shed
carries none.

What an **admission overflow** now does to a calibration cell is **void it**.
Truncating and keeping the earlier windows keeps exactly the healthy part of the cell and
discards the overloaded part, which biases every theta fitted on it towards health - the
same direction the superseded values were wrong in. See
:data:`scripts.openloop.SHED_POLICY_VOID`.

A **transient proxy error** - a connection carrying one request dying, which Envoy
reports as ``reset reason: connection termination`` rather than ``overflow`` - is not
that. It says one request went unserved and nothing about the admission ceiling, and
treating it as a shed is how a 450 s dsqwen-7b cell with 1 bad request in 1296, at
in-flight 43 under a 4096 circuit breaker, wrote zero rows. It marks its own window as a
violation, keeps it, and only voids the cell past
:data:`scripts.openloop.DEFAULT_MAX_PROXY_TRANSIENT_RATE`.

What a voided cell does to the campaign
---------------------------------------
It is **re-run once, in place**; a second void stops the whole campaign. The rule is
:func:`scripts.adaptive_boundary.next_void_attempt`, the same one the boundary search
applies to a voided probe, because "re-run once, then stop" is one statement about what a
void costs and two copies of it drift. Continuing instead - which is what
``--guard-mode warn`` did on its own - writes a zero-row CSV per cell and still prints
``campaign complete``: five hours of offered load and no rows anywhere.

Artifacts
---------
Per cell, under ``--raw-dir``: the per-request raw JSONL, the 1 Hz instant sidecar, the
classified failures, and a guard JSON recording outcomes, goodput, truncation, censoring
and any void reason. Per campaign, under ``--out-dir``: ``plan.json`` (every cell, in
order, with its provenance), ``capacity/<model>_<shape>.json`` (prior vs measured),
``boundary/<model>_<shape>.json`` (the search's probes and located rho*), the schedules
generated mid-campaign, and ``fit_plan.json`` - the re-windowing and refit invocations
the capture is meant to be consumed by, including the cadence each one must use and the
per-family control fits.
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

from scripts import adaptive_boundary as boundary
from scripts import admission_cap as admission
from scripts import gen_calibration_schedules as gen
from scripts import openloop
from scripts.openloop import LIVE_GRID_MS

#: Stage a shape's boundary search occupies in the campaign order. It is not one of
#: ``gen.PRIMITIVES``: its cells are generated at campaign time from what the previous
#: probe measured, so there is nothing to commit and nothing in the schedule index.
BOUNDARY_STAGE = "boundary"

#: Primitives in the order a model runs them. Steps first because it measures the
#: capacity everything else is defined against; the boundary search next because it is
#: what produces the windows theta is actually fitted on; bursts last because they are
#: the only primitive that can be skipped outright, so a truncated campaign still has
#: everything a fit needs.
STAGE_ORDER = ("steps", BOUNDARY_STAGE, "ramp", "bursts")

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
            for shape in gen.ALL_SHAPES:
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


def boundary_plan(models: Sequence[str], shapes: Sequence[str]) -> list[dict]:
    """The boundary-search cells a campaign will generate, for the estimate and the plan.

    They carry no schedule path because they have none yet: each probe's rho comes from
    what the previous probe measured. Listing them anyway is what keeps ``plan.json`` an
    honest statement of how long the campaign takes.
    """
    return [
        {
            "model": model,
            "shape": shape,
            "stage": BOUNDARY_STAGE,
            "probes": boundary.probe_count(),
            "duration_s": boundary.shape_seconds(),
            "stage_seconds": boundary.stage_seconds(),
        }
        for model in models
        for shape in shapes
    ]


def estimate_boundary_wall_clock_s(plan: Sequence[dict], cooldown_s: float) -> float:
    return sum(
        float(entry["duration_s"]) + cooldown_s * int(entry["probes"]) for entry in plan
    )


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
    """The ``r3_grid`` invocation for one cell.

    Three flags here are the campaign's own discipline rather than r3_grid's defaults,
    and each one exists because its absence silently pollutes theta:

    * ``--shed-policy void`` - a shed means the offered load never reached the engine.
      Keeping the windows from before it keeps only the healthy ones.
    * ``--max-p99-delay-ms`` at :data:`scripts.openloop.CALIBRATION_MAX_P99_DELAY_MS` -
      ten times tighter than the replay default, because a generator that fires late did
      not offer the load the cell is indexed by.
    * ``--ttft-slo-ms`` / ``--tpot-slo-ms`` - pinned, so the goodput a cell reports and
      the SLO the boundary search reads are the same numbers.

    ``--prompt-dir`` points every cell at this campaign's own ``<out-dir>/prompts``, so
    the prompts are built before each cell starts rather than inside its sends, and the
    bytes that went out are kept next to the measurement they produced. It is outside the
    repository on purpose: the committed schedules stay a few kB of segments.
    """
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
        "--prompt-dir", str(Path(args.out_dir) / "prompts"),
        "--shed-policy", openloop.SHED_POLICY_VOID,
        "--max-p99-delay-ms", str(openloop.CALIBRATION_MAX_P99_DELAY_MS),
        "--max-model-error-rate", str(args.max_model_error_rate),
        "--ttft-slo-ms", str(args.ttft_slo_ms),
        "--tpot-slo-ms", str(args.tpot_slo_ms),
    ]
    if cell.drain_start_s is not None:
        command += ["--drain-start-s", str(cell.drain_start_s)]
    if getattr(args, "envoy_stats_url", None):
        command += ["--envoy-stats-url", args.envoy_stats_url]
    if getattr(args, "envoy_cluster_filter", None):
        command += ["--envoy-cluster-filter", args.envoy_cluster_filter]
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


# ---------------------------------------------------------------- voided cell policy


class CellVoided(RuntimeError):
    """A cell voided on every attempt the retry rule allows, so the campaign stops."""

    def __init__(self, cell_id: str, attempts: int, void_reasons: Sequence[str]) -> None:
        self.cell_id = cell_id
        self.attempts = int(attempts)
        self.void_reasons = tuple(str(r) for r in void_reasons)
        super().__init__(
            f"cell {cell_id} was voided on all {int(attempts)} attempt(s) "
            f"({', '.join(self.void_reasons) or 'no reason recorded'}); stopping the "
            "campaign rather than going on writing cells that measured nothing"
        )


def read_cell_guard(
    raw_dir: Path, output: Path, cell_id: str, *, returncode: int = 0
) -> dict:
    """The guard artifact ``r3_grid`` wrote for one driven cell.

    A driver that died before writing one is itself a void: the absence of a verdict is
    not a passing verdict, and without this the campaign would read an exit code 1 as a
    cell with no void reasons and go on to the next one.
    """
    guard_path = Path(raw_dir) / Path(output).stem / f"{cell_id}.guard.json"
    guard: dict = {}
    if guard_path.exists():
        try:
            guard = json.loads(guard_path.read_text(encoding="utf-8"))
        except ValueError:
            guard = {}
    if returncode != 0 and not guard.get("void_reasons"):
        guard = dict(guard)
        guard["void_reasons"] = [f"driver exited {returncode}"]
    return guard


def attempt_output_path(output: Path, attempt: int) -> Path:
    """Where attempt ``attempt`` of a cell writes.

    Attempt 1 keeps the plain name so the artifact layout is unchanged; a re-run gets its
    own, because the raw JSONL is appended to and a second attempt writing the same file
    would pool the capture that failed with the one that replaced it.
    """
    path = Path(output)
    if int(attempt) <= 1:
        return path
    return path.with_name(f"{path.stem}_a{int(attempt)}{path.suffix}")


def drive_until_valid(
    cell_id: str, drive, *, max_retries: int = boundary.MAX_VOID_RETRIES
) -> tuple[dict, int]:
    """Drive a cell until it produces a verdict that is not a void, or give up loudly.

    ``drive(attempt) -> guard``. The retry rule is
    :func:`scripts.adaptive_boundary.next_void_attempt` - the boundary search's rule for
    a voided probe, used here unchanged so the campaign has one answer to "how many times
    do we try" rather than two.
    """
    attempt = 1
    while True:
        guard = drive(attempt)
        void_reasons = tuple(str(r) for r in (guard.get("void_reasons") or ()))
        if not void_reasons:
            return guard, attempt
        nxt = boundary.next_void_attempt(attempt, max_retries=max_retries)
        if nxt is None:
            raise CellVoided(cell_id, attempt, void_reasons)
        print(
            f"  cell {cell_id} is VOID ({', '.join(void_reasons)}); re-running it as "
            f"attempt {nxt}"
        )
        attempt = nxt


# ------------------------------------------------------------------ boundary search


def read_window_rows(path: Path) -> list[dict]:
    """Window rows back out of an ``r3_grid`` CSV, typed enough to judge a probe by.

    A voided cell writes an empty CSV, which reads back as zero rows - which is exactly
    what the boundary search must see: no evidence, not evidence of health.
    """
    import csv

    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row = dict(raw)
            for key in ("p95_ttft", "p95_tpot", "p95_e2e", "trs"):
                value = row.get(key)
                row[key] = None if value in (None, "") else float(value)
            row["slo_violated"] = str(row.get("slo_violated", "")).lower() in ("true", "1")
            row["model_errors"] = int(row.get("model_errors") or 0)
            rows.append(row)
    return rows


def probe_result_from_cell(
    probe: boundary.Probe,
    cell_id: str,
    rows: Sequence[dict],
    guard: dict,
    *,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
) -> boundary.ProbeResult:
    """Turn one driven hold cell into the verdict the search consumes.

    A cell the guard voided is reported ``valid=False`` and its (empty) rows are never
    consulted. That is the difference between "this load did not violate" and "we did not
    manage to offer this load", and conflating them walks the bracket upwards on every
    infrastructure hiccup.
    """
    void_reasons = tuple(str(r) for r in (guard.get("void_reasons") or ()))
    valid = not void_reasons and bool(rows)
    violated, violating, total = boundary.probe_violated(
        rows, ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms
    )
    if not valid and not void_reasons:
        void_reasons = ("no windows",)
    goodput_value = None
    body = guard.get("goodput")
    if isinstance(body, dict):
        goodput_value = body.get("goodput")
    return boundary.ProbeResult(
        probe=probe,
        violated=bool(violated) if valid else False,
        valid=valid,
        void_reasons=void_reasons,
        windows=total,
        violating_windows=violating,
        goodput=goodput_value,
        cell_id=cell_id,
    )


def run_boundary_search(
    model: str,
    shape: str,
    capacity_rps: float,
    *,
    drive,
    cap: admission.AdmissionCap,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    capacity_source: str = "measured_steps",
    search: Optional[boundary.BoundarySearch] = None,
) -> boundary.BoundarySearch:
    """Drive the three-stage search for one (model, shape).

    ``drive(probe, cell_id, body, meta) -> (rows, guard)`` is the seam: the campaign
    passes one that writes the schedule and shells out to ``r3_grid``, and the tests pass
    one that answers from a table. Everything that decides *which* rho comes next lives
    in :class:`scripts.adaptive_boundary.BoundarySearch`, so it is testable without a
    cluster; everything here is bookkeeping around it.
    """
    search = search or boundary.BoundarySearch(model=model, shape=shape)
    while True:
        probe = search.next_probe()
        if probe is None:
            break
        body, meta = gen.build_hold_schedule(
            model, shape, capacity_rps, probe.rho, probe.duration_s,
            stage=probe.stage, capacity_source=capacity_source, cap=cap,
        )
        rows, guard = drive(probe, meta["cell_id"], body, meta)
        search.record(
            probe_result_from_cell(
                probe, meta["cell_id"], rows, guard,
                ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms,
            )
        )
    return search


# ------------------------------------------------------------------- family diagnostic

#: How far the per-family thetas may sit outside the merged fit's bootstrap CI before the
#: merged number stops being publishable. Zero: the CI *is* the claim about how much the
#: number can move, so a family sitting outside it is a statement that the number depends
#: on which regime it was measured in.
FAMILY_SPREAD_TOLERANCE = 0.0


def family_theta_verdict(
    merged_theta: float,
    ci_half_width: float,
    family_thetas: dict[str, float],
    *,
    tolerance: float = FAMILY_SPREAD_TOLERANCE,
) -> dict:
    """Publish the merged theta, or fall back to the smallest family theta.

    The merged fit pools the prefill-heavy and decode-heavy shapes, which is only
    legitimate if they are measuring the same threshold. Fitting each family separately
    is the cheapest test of that: if every family theta lands inside the merged fit's own
    bootstrap CI, the pooling is consistent with the data and the merged number is
    published.

    If a family lands outside it, theta depends on the regime, and there is no single
    correct value. The published number is then the **smallest** family theta, because
    theta is a health threshold that the controller must stay above: publishing the
    larger one would declare healthy a regime that is not, whereas publishing the smaller
    one is conservative in the direction that fails safe.
    """
    if not family_thetas:
        return {
            "publish": "merged",
            "theta": merged_theta,
            "reason": "no per-family fit was produced, so pooling could not be checked",
            "family_thetas": {},
            "ci_half_width": ci_half_width,
            "outside": [],
        }
    bound = abs(ci_half_width) * (1.0 + tolerance)
    outside = sorted(
        name for name, value in family_thetas.items()
        if abs(float(value) - float(merged_theta)) > bound
    )
    if not outside:
        return {
            "publish": "merged",
            "theta": merged_theta,
            "reason": (
                f"every family theta is within the merged bootstrap CI (+/-{bound:.4g}), "
                "so the pooled fit is consistent with both regimes"
            ),
            "family_thetas": dict(sorted(family_thetas.items())),
            "ci_half_width": ci_half_width,
            "outside": [],
        }
    smallest = min(family_thetas.items(), key=lambda kv: float(kv[1]))
    return {
        "publish": "min_family",
        "theta": float(smallest[1]),
        "family": smallest[0],
        "reason": (
            f"family theta(s) {outside} fall outside the merged bootstrap CI "
            f"(+/-{bound:.4g}), so theta depends on the regime; the smallest family "
            f"theta ({smallest[0]}) is published because under-claiming health fails safe"
        ),
        "family_thetas": dict(sorted(family_thetas.items())),
        "ci_half_width": ci_half_width,
        "outside": outside,
    }


def fit_plan(
    models: Sequence[str],
    out_dir: Path,
    raw_dir: Path,
    args,
    index: Optional[dict] = None,
) -> dict:
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
    less than 5 %, the term is inert in this deployment. Under the superseded gateway cap
    the reason was structural - in-flight was capped below the engine's sequence limit,
    so the queue the term multiplies was almost always zero. Under the deployed
    engine-capped policy that excuse is gone and the control fit becomes a real question.

    *Held-out data.* The fitting re-window explicitly excludes the held-out shape's
    cells. The exclusion is by cell id, taken from the schedule index's ``held_out``
    entries, because the raw tree is a flat pile of cell files and nothing in a file name
    says which shape it came from - so a fit pointed at the raw directory would otherwise
    silently train on the validation set.

    *Families.* Besides the merged fit, each family gets its own. The spread between them
    is the check that the merged theta is not an artefact of pooling two regimes; see
    :func:`family_theta_verdict`.
    """
    fit_dir = out_dir / "fit"
    held_out_cells = sorted(held_out_cell_ids(index or {}))
    plan = {
        "generated_at_utc": utc_iso(),
        "window_ms": args.window_ms,
        "step_ms": args.fit_step_ms,
        "held_out_shapes": [s for s in gen.ALL_SHAPES if gen.is_held_out(s)],
        "held_out_cell_ids": held_out_cells,
        "training_shapes": list(gen.TRAINING_SHAPES),
        "families": {name: list(members) for name, members in gen.FAMILIES.items()},
        "rewindow": [],
        "refit": [],
        "acceptance": {
            "primary_lambda_wait": PRIMARY_LAMBDA_WAIT,
            "secondary_lambda_wait": SECONDARY_LAMBDA_WAIT,
            "tolerance": SECONDARY_FIT_TOLERANCE,
            "statement": (
                "if theta and the ranking separation move by less than "
                f"{SECONDARY_FIT_TOLERANCE:.0%} between the two fits, the waiting term is "
                "inert in this deployment"
            ),
            "stop_rule": {
                "min_publish_rate": boundary.MIN_PUBLISH_RATE,
                "max_ci_half_width_fraction": boundary.MAX_CI_HALF_WIDTH_FRACTION,
                "min_family_boundary_windows": boundary.MIN_FAMILY_BOUNDARY_WINDOWS,
                "remedy": (
                    "add hold cells at the boundary of the shapes already in the set; "
                    "never add a shape, which would make the stopping rule a search over "
                    "shape sets"
                ),
            },
            "family_spread": {
                "tolerance": FAMILY_SPREAD_TOLERANCE,
                "statement": (
                    "family thetas inside the merged bootstrap CI -> publish the merged "
                    "theta; otherwise theta is regime-dependent and the smallest family "
                    "theta is published, because under-claiming health fails safe"
                ),
            },
        },
    }
    exclusions: list[str] = []
    for cell_id in held_out_cells:
        exclusions += ["--exclude-cell-id", cell_id]
    for model in models:
        fitting_csv = fit_dir / f"{model}_fitting.csv"
        aliasing_csv = fit_dir / f"{model}_aliasing.csv"
        validation_csv = fit_dir / f"{model}_validation.csv"
        plan["rewindow"].append({
            "purpose": "fitting (the signal the controller consumes)",
            "model": model,
            "output": str(fitting_csv),
            "excludes_held_out": True,
            "command": [
                sys.executable, "-m", "scripts.rewindow_from_raw",
                "--model", model, "--raw-dir", str(raw_dir),
                "--output", str(fitting_csv),
                "--window-ms", str(args.window_ms), "--step-ms", str(args.fit_step_ms),
                "--instant-grid", "live",
                "--instant-sample-ms", str(LIVE_GRID_MS),
                *exclusions,
            ],
        })
        plan["rewindow"].append({
            "purpose": "aliasing figure and observability gap (ground truth)",
            "model": model,
            "output": str(aliasing_csv),
            "excludes_held_out": True,
            "command": [
                sys.executable, "-m", "scripts.rewindow_from_raw",
                "--model", model, "--raw-dir", str(raw_dir),
                "--output", str(aliasing_csv),
                "--window-ms", str(args.window_ms), "--step-ms", str(args.fit_step_ms),
                "--instant-grid", "raw",
                "--instant-sample-ms", str(args.instant_sample_ms),
                *exclusions,
            ],
        })
        if held_out_cells:
            plan["rewindow"].append({
                "purpose": "held-out validation set (never fitted on)",
                "model": model,
                "output": str(validation_csv),
                "excludes_held_out": False,
                "command": [
                    sys.executable, "-m", "scripts.rewindow_from_raw",
                    "--model", model, "--raw-dir", str(raw_dir),
                    "--output", str(validation_csv),
                    "--window-ms", str(args.window_ms), "--step-ms", str(args.fit_step_ms),
                    "--instant-grid", "live",
                    "--instant-sample-ms", str(LIVE_GRID_MS),
                    *[a for cell_id in held_out_cells for a in ("--only-cell-id", cell_id)],
                ],
            })
        for label, lambda_wait in (
            ("primary", PRIMARY_LAMBDA_WAIT),
            ("secondary", SECONDARY_LAMBDA_WAIT),
        ):
            plan["refit"].append({
                "label": label,
                "model": model,
                "family": "",
                "lambda_wait": lambda_wait,
                "command": [
                    sys.executable, "-m", "scripts.refit_trs_params",
                    "--input", str(fitting_csv),
                    "--model-name", model,
                    "--output", str(fit_dir / f"{model}_refit_{label}.json"),
                    # refit_trs_params requires both SLOs; omitting them made the plan's
                    # commands unrunnable as written.
                    "--ttft-p95-ms", str(args.ttft_slo_ms),
                    "--tpot-p95-ms", str(args.tpot_slo_ms),
                    "--inherited-lambda-wait", str(lambda_wait),
                    "--lambda-wait-candidates", str(lambda_wait),
                ],
            })
        for family, shapes in sorted(gen.FAMILIES.items()):
            cell_ids = sorted(family_cell_ids(index or {}, shapes))
            if not cell_ids:
                continue
            family_csv = fit_dir / f"{model}_fitting_{family}.csv"
            plan["rewindow"].append({
                "purpose": f"per-family fit ({family}) - diagnostic",
                "model": model,
                "family": family,
                "output": str(family_csv),
                "excludes_held_out": True,
                "command": [
                    sys.executable, "-m", "scripts.rewindow_from_raw",
                    "--model", model, "--raw-dir", str(raw_dir),
                    "--output", str(family_csv),
                    "--window-ms", str(args.window_ms), "--step-ms", str(args.fit_step_ms),
                    "--instant-grid", "live",
                    "--instant-sample-ms", str(LIVE_GRID_MS),
                    *[a for cell_id in cell_ids for a in ("--only-cell-id", cell_id)],
                ],
            })
            plan["refit"].append({
                "label": f"family_{family}",
                "model": model,
                "family": family,
                "shapes": list(shapes),
                "lambda_wait": PRIMARY_LAMBDA_WAIT,
                "purpose": (
                    "diagnostic only - its theta is compared against the merged fit's "
                    "bootstrap CI, never published on its own unless the families disagree"
                ),
                "command": [
                    sys.executable, "-m", "scripts.refit_trs_params",
                    "--input", str(family_csv),
                    "--model-name", model,
                    "--output", str(fit_dir / f"{model}_refit_family_{family}.json"),
                    "--ttft-p95-ms", str(args.ttft_slo_ms),
                    "--tpot-p95-ms", str(args.tpot_slo_ms),
                    "--inherited-lambda-wait", str(PRIMARY_LAMBDA_WAIT),
                    "--lambda-wait-candidates", str(PRIMARY_LAMBDA_WAIT),
                ],
            })
    return plan


def held_out_cell_ids(index: dict) -> set[str]:
    """Cell ids of every schedule entry marked held out.

    Read from the index rather than from a shape-name pattern: the raw tree is keyed by
    cell id, and a fit that could not name the held-out cells would train on them.
    """
    out: set[str] = set()
    for entry in index.get("schedules", []) or []:
        if entry.get("held_out") and entry.get("cell_id"):
            out.add(str(entry["cell_id"]))
    return out


def family_cell_ids(index: dict, shapes: Sequence[str]) -> set[str]:
    """Cell ids belonging to the shapes of one family, held-out shapes excluded.

    The exclusion is belt and braces - no held-out shape is in a family today - but it is
    the kind of thing that stops being true quietly.
    """
    wanted = set(shapes)
    return {
        str(entry["cell_id"])
        for entry in (index.get("schedules", []) or [])
        if entry.get("shape") in wanted
        and entry.get("cell_id")
        and not entry.get("held_out")
    }


def drive_boundary_search(
    cell: Cell,
    measured: MeasuredCapacity,
    args,
    *,
    cap: admission.AdmissionCap,
    schedule_dir: Path,
    out_dir: Path,
) -> boundary.BoundarySearch:
    """Run one shape's boundary search, driving each probe through ``r3_grid``.

    Each probe writes its schedule under ``<out-dir>/schedules`` and its window CSV under
    ``<out-dir>``, both named after the probe's own cell id, so nothing overwrites
    anything and a re-run of a voided probe is visibly a second attempt.
    """
    def drive(probe, cell_id, body, meta):
        stem = f"{cell.shape}_{gen.HOLD_PRIMITIVE}{gen.hold_load_code(probe.rho)}"
        schedule_path = schedule_dir / cell.model / f"{stem}.json"
        schedule_path.parent.mkdir(parents=True, exist_ok=True)
        schedule_path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        (schedule_path.parent / f"{stem}.meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        probe_cell = Cell(
            model=cell.model,
            shape=cell.shape,
            primitive=gen.HOLD_PRIMITIVE,
            cell_id=cell_id,
            schedule=str(schedule_path),
            duration_s=probe.duration_s,
            capacity_rps=measured.capacity_used_rps,
            capacity_source=measured.capacity_source,
            metadata=meta,
        )
        output = out_dir / f"{cell.model}_{cell.shape}_{stem}_a{probe.attempt}.csv"
        command = cell_command(probe_cell, args, schedule_path, output)
        print(
            f"  boundary {cell.model}/{cell.shape} {probe.stage} rho={probe.rho:g} "
            f"({probe.duration_s:.0f}s, attempt {probe.attempt}): {' '.join(command)}"
        )
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"  probe failed with exit {result.returncode}")
        rows = read_window_rows(output)
        guard = read_cell_guard(
            Path(args.raw_dir), output, cell_id, returncode=result.returncode
        )
        time.sleep(args.cooldown_s)
        return rows, guard

    return run_boundary_search(
        cell.model,
        cell.shape,
        measured.capacity_used_rps,
        drive=drive,
        cap=cap,
        ttft_slo_ms=args.ttft_slo_ms,
        tpot_slo_ms=args.tpot_slo_ms,
        capacity_source=measured.capacity_source,
    )


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

    # The held-out shape gets no boundary search. Its cells are generated at campaign
    # time, so their ids cannot be in the index that the fit's exclusion list is built
    # from - they would be invisible to the held-out filter and would train the fit.
    boundary_shapes = [
        shape for shape in gen.ALL_SHAPES
        if not gen.is_held_out(shape) and any(c.shape == shape for c in runnable)
    ]
    boundary_cells = boundary_plan(models, boundary_shapes)
    schedule_seconds = estimate_wall_clock_s(runnable, args.cooldown_s)
    boundary_seconds = estimate_boundary_wall_clock_s(boundary_cells, args.cooldown_s)
    plan_doc = {
        "generated_at_utc": utc_iso(),
        "admission_cap": cap.as_dict(),
        "index": str(index_path),
        "models": models,
        "stage_order": list(STAGE_ORDER),
        "cooldown_s": args.cooldown_s,
        "cells": [cell.as_dict() for cell in runnable],
        "boundary_cells": boundary_cells,
        "skipped": [
            {"model": c.model, "shape": c.shape, "primitive": c.primitive,
             "reason": c.skip_reason}
            for c in skipped
        ],
        "estimated_schedule_wall_clock_s": round(schedule_seconds, 1),
        "estimated_boundary_wall_clock_s": round(boundary_seconds, 1),
        "estimated_wall_clock_s": round(schedule_seconds + boundary_seconds, 1),
    }
    (out_dir / "plan.json").write_text(json.dumps(plan_doc, indent=2) + "\n", encoding="utf-8")
    (out_dir / "fit_plan.json").write_text(
        json.dumps(fit_plan(models, out_dir, raw_dir, args, index), indent=2) + "\n",
        encoding="utf-8",
    )

    hours = plan_doc["estimated_wall_clock_s"] / 3600.0
    print(f"admission cap: {cap.name} (shed ceiling {cap.shed_ceiling}, "
          f"admission controller: {cap.admission_controller}, "
          f"engine ceiling {cap.fleet_sequence_limit} = max_num_seqs x replicas)")
    print(f"{len(runnable)} scheduled cells + {len(boundary_cells)} boundary searches "
          f"({sum(c['probes'] for c in boundary_cells)} probe cells), {len(skipped)} skipped, "
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
    boundary_dir = out_dir / "boundary"
    boundary_dir.mkdir(parents=True, exist_ok=True)
    regenerated_dir = out_dir / "schedules"
    measurements: dict[tuple[str, str], MeasuredCapacity] = {}
    searches: dict[tuple[str, str], boundary.BoundarySearch] = {}
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

        base_output = out_dir / f"{cell.model}_{cell.shape}_{cell.primitive}.csv"
        output = base_output
        exit_code = 0

        def drive(attempt: int, cell=cell, schedule_path=schedule_path,
                  base_output=base_output, position=position) -> dict:
            nonlocal output, exit_code
            output = attempt_output_path(base_output, attempt)
            if attempt > 1:
                # The re-run starts from the same quiet fleet the first attempt did.
                time.sleep(args.cooldown_s)
            command = cell_command(cell, args, schedule_path, output)
            print(f"[{position}/{len(runnable)}] {cell.model} {cell.shape} "
                  f"{cell.primitive} ({cell.duration_s:.0f}s, attempt {attempt}): "
                  f"{' '.join(command)}")
            result = subprocess.run(command, check=False)
            exit_code = result.returncode
            if result.returncode != 0:
                print(f"cell failed with exit {result.returncode}")
            return read_cell_guard(
                raw_dir, output, cell.cell_id, returncode=result.returncode
            )

        try:
            drive_until_valid(cell.cell_id, drive)
        except CellVoided as voided:
            print(str(voided))
            return 1
        if exit_code != 0 and args.stop_on_failure:
            return exit_code

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
                if not args.skip_boundary_search and not gen.is_held_out(cell.shape):
                    time.sleep(args.cooldown_s)
                    search = drive_boundary_search(
                        cell, measurement, args,
                        cap=cap,
                        schedule_dir=regenerated_dir,
                        out_dir=out_dir,
                    )
                    (boundary_dir / f"{cell.model}_{cell.shape}.json").write_text(
                        json.dumps(search.as_dict(), indent=2) + "\n", encoding="utf-8"
                    )
                    searches[(cell.model, cell.shape)] = search
                    print(
                        f"boundary {cell.model}/{cell.shape}: rho* = {search.rho_star}, "
                        f"dwelled at {search.dwell_rho}"
                        + ("" if search.boundary_found else " (NO violation was observed - "
                           "the boundary is above everything offered)")
                        + (f"; STOPPED: {search.stopped_reason}" if search.stopped_reason else "")
                    )

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
    ap.add_argument("--max-model-error-rate", type=float,
                    default=openloop.DEFAULT_MAX_MODEL_ERROR_RATE,
                    help="a cell whose MODEL error rate exceeds this is void and must be "
                         "re-run; the individual error windows are kept and counted as "
                         "violations either way")
    ap.add_argument("--envoy-cluster-filter", default=None,
                    help="only count Envoy overflow counters whose stat name or labels "
                         "contain this (e.g. the model's cluster name). Without it the "
                         "sentinel sums every cluster on the gateway, so another model's "
                         "overflow voids this cell")
    ap.add_argument("--envoy-stats-url", default=None,
                    help="Envoy stats endpoint - in this cluster "
                         "http://<envoy-pod-ip>:19001/stats/prometheus, because the admin "
                         "listener on :19000 is not reachable from the node and the envoy "
                         "container has no curl. Every cell then records the change in "
                         "upstream_rq_pending_overflow across it, is voided if it moved, "
                         "and records any disagreement with what the client classified as "
                         "admission overflow. Validity sentinel only - never a control "
                         "input.")
    ap.add_argument("--skip-boundary-search", action="store_true",
                    help="run only the committed primitives (steps/ramp/bursts) and skip "
                         "the adaptive boundary search")
    ap.add_argument("--guard-mode", default="warn", choices=["fail", "warn"],
                    help="how the per-cell driver reacts to its own guard. This does NOT "
                         "decide what a VOID cell does to the campaign: a voided cell is "
                         "always re-run once and a second void always stops the run, "
                         "whatever this is set to. The guard verdict is recorded per "
                         "cell either way")
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

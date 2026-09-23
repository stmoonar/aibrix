#!/usr/bin/env python3
"""Runner for the open-loop calibration campaign.

Two designs
-----------
``--design ladder`` (the default) is the preregistered second round
(``docs/preregistration-20260923-calibration-run2.md``): prior-guided boundary search of
every shape, an interleaved randomised hold ladder, ramps, supplementary cells and
sentinels, with every cell independently seeded and the engine drained between cells.
It lives in :mod:`scripts.calibration_ladder` / :mod:`scripts.calibration_design` and
refuses to start without ``--rho-priors`` and ``--regime-groups``.

``--design primitives`` is the first round's design, kept so that run can be reproduced;
the rest of this docstring describes it.

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
about 13.5 minutes of offered load per shape: coarse 3 x 90 s, bisect 2 x 120 s, dwell
300 s at ``0.95 rho*``. A probe is judged on the fitting re-window's own windows and
label (see :mod:`scripts.adaptive_boundary`), so the boundary is located with the ruler
the fit then measures the evidence with.

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

Static grid (opt-in)
--------------------
``--static-grid`` appends v1-style steady-state cells after the stages above: per model,
2 inputs x 2 outputs x 3 offered loads (rho/rho* 0.85 / 1.0 / 1.1), 300 s each, placed
from a previous campaign's measured boundaries (:mod:`scripts.static_grid`, default the
2026-09-21 campaign). ``--static-grid-only`` drives just those cells; ``--static-grid-list``
prints them with the GPU-minute estimate and exits. Without the flag nothing changes.

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

When the campaign ends - complete, stopped or crashed - it writes
``campaign_status.json`` and converts its own directory into the standard dataset
(``<out-dir>/dataset/``, see :mod:`scripts.calibration_dataset`). When every sibling
campaign under the same parent has finished too, the last one to finish also builds the
merged dataset of the whole run at ``<parent>/dataset/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

from tre_common import slo_labels

from scripts import adaptive_boundary as boundary
from scripts import admission_cap as admission
from scripts import gen_calibration_schedules as gen
from scripts import openloop
from scripts import static_grid
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

#: lambda_wait for the primary fit: the D-line's (plan 2026-09-21 §6.11; the lambda check
#: of ``scripts.dline_refit`` keeps 1 unless another value gains >= 0.02 BA, and it kept 1
#: for every model and arm on the 09-21 data). It used to be the registry's 3.0, which the
#: D-line's own fits never ran at - the archived alpha fit passed --lambda-wait 1.0.
PRIMARY_LAMBDA_WAIT = 1.0
#: The control fit. If theta and the ranking separation move by less than
#: SECONDARY_FIT_TOLERANCE between the two, the waiting term contributed nothing and the
#: honest statement is that it is inert in this deployment - because gateway admission
#: caps in-flight below the engine's sequence limit, so the queue it multiplies is
#: almost always zero.
SECONDARY_LAMBDA_WAIT = 0.0
SECONDARY_FIT_TOLERANCE = 0.05

#: TTFT SLO of the fit label (tre_common.slo_labels): the primary label of plan
#: 2026-09-21 6.11 D6', max(500 ms, 5 * idle TTFT(L)), TPOT 75 ms. ``--fit-ttft-slo-mode
#: fixed`` gives the 500/75 ms comparison column, ``--fit-ttft-slowdown-k 3
#: --fit-ttft-floor-ms 150`` the D6 ablation arm; k/floor default to the registry profile.
DEFAULT_FIT_TTFT_SLO_MODE = "slowdown"


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
    #: Client per-request latency, nearest-rank p95 over the level's steady part.
    p95_ttft_client_ms: Optional[float]
    p95_tpot_client_ms: Optional[float]
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
                p95_ttft_client_ms=None if p95_ttft is None else round(p95_ttft, 3),
                p95_tpot_client_ms=None if p95_tpot is None else round(p95_tpot, 3),
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


def boundary_plan(
    models: Sequence[str], shapes: Sequence[str], *, coarse_seconds: float = boundary.COARSE_SECONDS,
) -> list[dict]:
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
            "duration_s": boundary.shape_seconds(coarse_seconds),
            "stage_seconds": boundary.stage_seconds(coarse_seconds),
            # not in duration_s: only a shape whose coarse stage misses the flip pays it
            "max_extension_probes": boundary.max_extension_probes(),
            "max_extension_s": boundary.max_extension_seconds(coarse_seconds),
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


def primary_label(args, model: str) -> slo_labels.LabelDefinition:
    """The campaign's primary window label for ``model`` (plan 2026-09-21 §6.11 D6'): the
    TTFT mode / k / floor of ``--fit-ttft-*`` (default: the registry profile, slowdown),
    the model's idle TTFT fit from the registry, ``--ttft-slo-ms`` / ``--tpot-slo-ms`` as
    the fixed thresholds, ``--fit-min-completed-requests``. The probes of the boundary
    search are judged on it and every fit trains on it - one ruler for both."""
    return slo_labels.label_def_for_model(
        model,
        ttft_p95_ms=args.ttft_slo_ms,
        tpot_p95_ms=args.tpot_slo_ms,
        mode=getattr(args, "fit_ttft_slo_mode", None),
        k=getattr(args, "fit_ttft_slowdown_k", None),
        floor=getattr(args, "fit_ttft_floor_ms", None),
        min_completed_requests=getattr(
            args, "fit_min_completed_requests", slo_labels.DEFAULT_MIN_COMPLETED_REQUESTS),
        registry=getattr(args, "registry", None),
    )


def cell_command(cell: Cell, args, schedule_path: Path, output: Path) -> list[str]:
    """The ``r3_grid`` invocation for one cell.

    Three flags here are the campaign's own discipline rather than r3_grid's defaults,
    and each one exists because its absence silently pollutes theta:

    * ``--shed-policy void`` - a shed means the offered load never reached the engine.
      Keeping the windows from before it keeps only the healthy ones.
    * ``--max-p99-delay-ms`` at :data:`scripts.openloop.CALIBRATION_MAX_P99_DELAY_MS` -
      ten times tighter than the replay default, because a generator that fires late did
      not offer the load the cell is indexed by.
    * ``--ttft-slo-ms`` / ``--tpot-slo-ms`` and the primary label's TTFT mode - pinned,
      so the goodput a cell reports, the label the boundary search reads and the label
      the fit trains on are the same (``--window-align`` / ``--step-ms`` likewise put the
      online windows on the fitting re-window's 10 s grid, D8).

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
        # The fitting re-window's windows: the online rows a probe is judged on are then
        # the rows the fit is built from, window for window.
        "--window-ms", str(args.window_ms),
        "--step-ms", str(args.fit_step_ms),
        "--window-align", str(getattr(args, "fit_window_align", "grid")),
        "--instant-sample-ms", str(args.instant_sample_ms),
        # The server-side p95 columns (diagnostic only) are read once per sliding window;
        # the gateway's zsets answer in ~6 ms a model where the legacy key SCAN takes
        # ~650 ms, which over a 5 s step would add minutes to every cell.
        "--metrics-schema", "v2",
        "--namespace", args.model_namespace,
        "--guard-mode", args.guard_mode,
        "--min-slo-windows", str(args.min_slo_windows),
        "--prompt-dir", str(Path(args.out_dir) / "prompts"),
        "--shed-policy", openloop.SHED_POLICY_VOID,
        "--max-p99-delay-ms", str(openloop.CALIBRATION_MAX_P99_DELAY_MS),
        "--max-model-error-rate", str(args.max_model_error_rate),
        "--ttft-slo-ms", str(args.ttft_slo_ms),
        "--tpot-slo-ms", str(args.tpot_slo_ms),
        # The primary window label (D6' by default): the one the probe is judged on and
        # the fit trains on.
        *slo_labels.label_mode_cli_args(primary_label(args, cell.model)),
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


#: Float columns of the window CSV, parsed when it is read back.
_FLOAT_COLUMNS = (
    slo_labels.P95_TTFT_CLIENT, slo_labels.P95_TPOT_CLIENT, slo_labels.P95_E2E_CLIENT,
    slo_labels.P95_TTFT_SERVER, slo_labels.P95_TPOT_SERVER, slo_labels.P95_E2E_SERVER,
    "trs",
)


def read_window_rows(path: Path) -> list[dict]:
    """Window rows back out of an ``r3_grid`` CSV, typed enough to judge a probe by.

    A voided cell writes an empty CSV, which reads back as zero rows - which is exactly
    what the boundary search must see: no evidence, not evidence of health.
    ``slo_violated`` reads back as True / False / None (unlabeled), never as a bare False.
    """
    import csv

    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row = dict(raw)
            for key in _FLOAT_COLUMNS:
                value = row.get(key)
                row[key] = None if value in (None, "", "None") else float(value)
            for key in slo_labels.UNSERVED_COLUMNS:
                row[key] = int(float(row.get(key) or 0))
            flag = str(row.get(slo_labels.VIOLATED_COLUMN, "")).strip().lower()
            row[slo_labels.VIOLATED_COLUMN] = (
                True if flag in ("true", "1") else False if flag in ("false", "0") else None
            )
            rows.append(row)
    return rows


def probe_result_from_cell(
    probe: boundary.Probe,
    cell_id: str,
    rows: Sequence[dict],
    guard: dict,
    *,
    label=None,
    ttft_slo_ms: Optional[float] = None,
    tpot_slo_ms: Optional[float] = None,
) -> boundary.ProbeResult:
    """Turn one driven hold cell into the verdict the search consumes.

    ``label`` is the campaign's primary window label (:func:`primary_label`, D6') - the
    label the fit is trained on; the fixed ``ttft_slo_ms`` / ``tpot_slo_ms`` pair is the
    09-23 interface.

    A cell the guard voided is :data:`~scripts.adaptive_boundary.VERDICT_VOID` and its
    (empty) rows are never consulted. That is the difference between "this load did not
    violate" and "we did not manage to offer this load", and conflating them walks the
    bracket upwards on every infrastructure hiccup. A cell that was not voided but gave
    too little evidence - no rows at all included - is
    :data:`~scripts.adaptive_boundary.VERDICT_INCONCLUSIVE`, which the search answers
    with a longer re-drive, never with a healthy verdict.
    """
    void_reasons = tuple(str(r) for r in (guard.get("void_reasons") or ()))
    goodput_value = None
    body = guard.get("goodput")
    if isinstance(body, dict):
        goodput_value = body.get("goodput")
    if void_reasons:
        return boundary.ProbeResult(
            probe=probe,
            verdict=boundary.VERDICT_VOID,
            void_reasons=void_reasons,
            windows=len(rows),
            goodput=goodput_value,
            cell_id=cell_id,
        )
    verdict = boundary.probe_verdict(
        rows, label=label, ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms,
    )
    return boundary.ProbeResult(
        probe=probe,
        verdict=verdict.verdict,
        windows=verdict.windows,
        labeled_windows=verdict.labeled_windows,
        independent_windows=verdict.independent_windows,
        violating_windows=verdict.violating_windows,
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
    label=None,
    ttft_slo_ms: Optional[float] = None,
    tpot_slo_ms: Optional[float] = None,
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
                label=label, ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms,
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
    direction: str = "higher_is_healthier",
) -> dict:
    """Publish the merged theta, or fall back to the LARGEST family theta.

    (For a ``lower_is_healthier`` signal - the ablation arms queue length and token rates,
    ``z = theta / value`` - a SMALLER theta makes CRITICAL easier, so the conservative
    fallback there is the SMALLEST family theta; ``direction`` selects it.)

    The merged fit pools the prefill-heavy and decode-heavy shapes, which is only
    legitimate if they are measuring the same threshold. Fitting each family separately
    is the cheapest test of that: if every family theta lands inside the merged fit's own
    bootstrap CI, the pooling is consistent with the data and the merged number is
    published.

    If a family lands outside it, theta depends on the regime, and there is no single
    correct value. The published number is then the **largest** family theta. The
    controller computes ``Z = TSS / theta`` and calls a model CRITICAL when
    ``Z < tau_crit`` (``tre_controller.planning.classify.classify_model``), so a larger
    theta makes CRITICAL easier to reach: it is the value that errs towards adding
    capacity, i.e. the conservative one. (The smallest family theta, published before
    2026-09-22, was the least conservative choice: it declares healthy the regime whose
    own threshold is higher.) The real remedy is a w_p that removes the family gap
    (plan §6.5); this rule only picks the safe side while it exists.
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
    if direction == "lower_is_healthier":
        chosen = min(family_thetas.items(), key=lambda kv: float(kv[1]))
        return {
            "publish": "min_family",
            "theta": float(chosen[1]),
            "family": chosen[0],
            "reason": (
                f"family theta(s) {outside} fall outside the merged bootstrap CI "
                f"(+/-{bound:.4g}), so theta depends on the regime; the smallest family "
                f"theta ({chosen[0]}) is published because this signal is "
                "lower_is_healthier (Z = theta/value), so the smaller theta errs towards "
                "adding capacity"
            ),
            "family_thetas": dict(sorted(family_thetas.items())),
            "ci_half_width": ci_half_width,
            "outside": outside,
        }
    largest = max(family_thetas.items(), key=lambda kv: float(kv[1]))
    return {
        "publish": "max_family",
        "theta": float(largest[1]),
        "family": largest[0],
        "reason": (
            f"family theta(s) {outside} fall outside the merged bootstrap CI "
            f"(+/-{bound:.4g}), so theta depends on the regime; the largest family "
            f"theta ({largest[0]}) is published because Z = TSS/theta < tau_crit is "
            "CRITICAL, so the larger theta errs towards adding capacity"
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
    *,
    ledgers: Sequence[Path] = (),
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

    *lambda_wait.* The primary fit runs at the D-line's 1.0 (the registry still deploys
    3.0; the ``dline`` step's lambda check is what may move it). A secondary fit at 0.0 says
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

    *Ladder campaigns* (``ledgers``: the ``cells.jsonl`` of a ``--design ladder`` run) are
    selected from the ledger instead, when the fit runs: ``rewindow_from_raw --ledger
    ... --only-split train`` for the fitting and family CSVs, ``--only-split holdout`` for
    the validation CSV, and every row carries the ledger's role / split / primitive and
    ``in_warmup`` (the first ``warmup_s`` of a hold cell), which the fit loaders drop.
    ``alpha_fit`` reads the same ledger to tell constant-load cells from ramps.

    *Families.* Besides the merged fit, each family gets its own. The spread between them
    is the check that the merged theta is not an artefact of pooling two regimes; see
    :func:`family_theta_verdict`. A family CSV is selected with ``rewindow_from_raw
    --only-shape``, i.e. from the cell directories on disk when the fit runs - boundary
    hold cells included - not from a cell list frozen before the boundary search (B3).

    *Steps* (plan §6.3 B5), in ``plan["order"]``:

    1. ``rewindow`` - fitting / aliasing / validation / per-family CSVs; unserved requests
       (``.failures.jsonl``) are marked violated;
    1b. ``alpha`` - ``scripts.alpha_fit`` (plan 6.11 D4'): the EMA alpha / tau of the
       deployed classifier (tau-EMA + dwell 2), chosen by same-window LOSO BA under a
       healthy false-alarm cap, ties broken by spurious CRITICAL episodes on steady healthy
       cells, then the larger alpha. Its ``registry_fields`` (``trs.ema_tau_ms`` /
       ``trs.ema_alpha``) are what the later steps' ``--ema-tau-ms`` must be set to before
       theta is published; the commands below carry the registry's current tau;
    1c. ``dline`` - ``scripts.dline_refit`` alpha / wp / final per model and label arm
       (primary, fixed), then ``summary``: the D-line decision pipeline (D4', D3, D5 and
       the hold-out report) that the 2026-09-22 numbers came from;
    2. ``theta`` - ``tre_calibration.cli --recompute-tss`` on the merged and every family
       CSV at lambda_wait 3 (primary) and 0 (control): theta and delta_crit;
    3. ``verdict`` - ``theta_verdict verdict``: bootstrap CI of theta and both band
       margins, stop rule, family rule -> the number that would be published;
    4. ``ablation`` - the same verdict for the two TSS ablation arms of plan §6.9,
       TSS(lambda=0) and TSS(w_p=1), with the primary label (``--label-lambda-wait``), so
       the paper can show that the weighting, not the signal, does the work;
    5. ``alt`` - ``fit_alt_thresholds`` for queue_len / decode_tps / prefill_tps on the
       same fitting and family CSVs and the same label; it runs the very same
       ``theta_verdict.verdict_report`` per model and writes one verdict JSON per
       model and signal;
    6. ``holdout`` - ``theta_verdict holdout`` for every verdict above (TSS, both arms,
       every alt signal): the published threshold scored on the held-out validation CSV,
       which no earlier step reads.

    Every step uses the one label (``tre_common.slo_labels``: p95 TTFT/TPOT + unserved)
    at ``--ttft-slo-ms`` / ``--tpot-slo-ms``.
    """
    # Imported here, not at module level: the campaign driver itself must stay runnable
    # on a PYTHONPATH without the calibration package.
    from tre_common.slo_labels import (
        label_arms, label_cli_args, label_def_from_args, label_mode_cli_args,
    )
    from tre_common.tss import DEFAULT_EMA_TAU_MS

    from scripts import alpha_fit

    fit_dir = out_dir / "fit"
    held_out_cells = sorted(held_out_cell_ids(index or {}))
    use_static = static_grid_enabled(args)
    families = gen.families(static_grid=use_static)
    plan = {
        "generated_at_utc": utc_iso(),
        "window_ms": args.window_ms,
        "step_ms": args.fit_step_ms,
        "window_align": getattr(args, "fit_window_align", "grid"),
        "held_out_shapes": [s for s in gen.ALL_SHAPES if gen.is_held_out(s)],
        "held_out_cell_ids": held_out_cells,
        "training_shapes": list(gen.training_shapes(static_grid=use_static)),
        "families": {name: list(members) for name, members in families.items()},
        "static_grid": {
            "enabled": use_static,
            "shapes": {
                shape: {
                    "input_tokens": i,
                    "output_tokens": o,
                    "family": gen.static_grid_family(i, o),
                    "held_out": gen.is_held_out(shape),
                }
                for shape, (i, o) in gen.STATIC_GRID_SHAPES.items()
            } if use_static else {},
            "family_rule": (
                f"i/o >= {gen.STATIC_FAMILY_PREFILL_MIN_RATIO:g} -> prefill_heavy; "
                f"i/o <= {gen.STATIC_FAMILY_DECODE_MAX_RATIO:g} -> decode_heavy; "
                "otherwise merged fit only"
            ),
        },
        "order": ["rewindow", "alpha", "dline", "theta", "verdict", "ablation", "alt", "holdout"],
        "label_def": None,  # filled per model below
        "label_def_by_model": {},
        "ema_tau_ms": DEFAULT_EMA_TAU_MS,
        "rewindow": [],
        "alpha": [],
        "alpha_rule": {
            "module": "scripts.alpha_fit",
            "tau_grid_s": list(alpha_fit.TAU_GRID_S),
            "alpha_grid": [round(alpha_fit.alpha_of_tau(t), 4) for t in alpha_fit.TAU_GRID_S],
            "dt_ref_s": alpha_fit.DT_REF_S,
            "dwell_windows": alpha_fit.DEFAULT_DWELL_WINDOWS,
            "fa_max": alpha_fit.FA_MAX,
            "label_horizon": "same window",
            "bootstrap": alpha_fit.DEFAULT_BOOTSTRAP,
            "statement": (
                "per alpha refit theta/delta in each leave-one-shape-out fold; deployed "
                "classifier tau-EMA + dwell; feasible iff healthy FA <= fa_max; score LOSO BA; "
                "within 1 SE (cell bootstrap) fewest spurious CRITICAL episodes/h on steady "
                "healthy cells, then larger alpha; the chosen tau feeds --ema-tau-ms of theta/"
                "verdict/ablation/alt"
            ),
        },
        "dline": [],
        "dline_rule": {
            "module": "scripts.dline_refit",
            "stages": ["alpha", "wp", "final", "summary"],
            "arms": ["primary", "fixed"],
            "statement": (
                "the D-line decision pipeline (plan 6.11 step 4): alpha (D4', scripts.alpha_fit), "
                "w_p by the constrained 1-SE rule (D3) at lambda_wait 1 with the lambda check, "
                "the final verdict publishing the merged theta (D5) and the hold-out report; "
                "per label arm, outputs under fit/dline/<model>/<arm>/"
            ),
        },
        "theta": [],
        "verdict": [],
        "ablation": [],
        "alt": [],
        "holdout": [],
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
                "appendix_ci_half_width_fraction": boundary.APPENDIX_CI_HALF_WIDTH_FRACTION,
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
                    "theta; otherwise theta is regime-dependent and the largest family "
                    "theta is published (Z = TSS/theta < tau_crit is CRITICAL, so the "
                    "larger theta errs towards adding capacity)"
                ),
            },
        },
    }
    # The slo_label column of every re-windowed CSV is computed against the SLO the
    # campaign pinned, the same one its probes were judged against.
    slo_args = ["--ttft-slo-ms", str(args.ttft_slo_ms), "--tpot-slo-ms", str(args.tpot_slo_ms)]
    exclusions: list[str] = []
    for cell_id in held_out_cells:
        exclusions += ["--exclude-cell-id", cell_id]
    ledger_args = [a for path in ledgers for a in ("--ledger", str(path))]
    if ledger_args:
        # Split and role come from the run's own ledger, read when the fit runs.
        train_sel = [*ledger_args, "--only-split", "train"]
        holdout_sel = [*ledger_args, "--only-split", "holdout"]
    else:
        train_sel = list(exclusions)
        holdout_sel = [a for cell_id in held_out_cells for a in ("--only-cell-id", cell_id)]
    plan["selection"] = {
        "source": "ledger" if ledger_args else "schedule index",
        "ledgers": [str(p) for p in ledgers],
        "train": train_sel,
        "holdout": holdout_sel,
    }
    registry = _load_registry(getattr(args, "registry", None))
    # Plan ��6.9g pitfall 2 / D8: by default every fitting window ends on the gateway's
    # 10 s grid with a 10 s step, the window the phase-aligned controller reads.
    # --fit-window-align none (+ --fit-step-ms 5000) reproduces the legacy windowing.
    window_align = getattr(args, "fit_window_align", "grid")
    align = ["--window-align", window_align]
    # The fit label (tre_common.slo_labels): the D6' slowdown TTFT SLO by default - each
    # model's idle TTFT fit, k and floor come from the registry - or the fixed one.
    label_args = argparse.Namespace(
        ttft_p95_ms=args.ttft_slo_ms,
        tpot_p95_ms=args.tpot_slo_ms,
        ttft_slo_mode=getattr(args, "fit_ttft_slo_mode", DEFAULT_FIT_TTFT_SLO_MODE),
        ttft_slowdown_k=getattr(args, "fit_ttft_slowdown_k", None),
        ttft_floor_ms=getattr(args, "fit_ttft_floor_ms", None),
        ttft_idle_c_ms=None,
        ttft_idle_b_ms_per_token=None,
        min_completed_requests=getattr(args, "fit_min_completed_requests", 20),
        label_registry=getattr(args, "registry", None),
    )
    plan["label_arms_by_model"] = {}
    for model in models:
        fit_label = label_def_from_args(label_args, model)
        plan["label_def_by_model"][model] = fit_label.as_dict()
        # D6': the primary label, the fixed comparison column and the k=3/150 ms ablation
        # are all recorded so any fit can be re-run on another arm (--fit-ttft-slo-mode ...).
        if fit_label.slowdown:
            plan["label_arms_by_model"][model] = {
                arm: label.as_dict() for arm, label in label_arms(fit_label).items()
            }
    plan["label_def"] = plan["label_def_by_model"][models[0]] if models else None
    live = [
        "--window-ms", str(args.window_ms), "--step-ms", str(args.fit_step_ms),
        "--instant-grid", "live", "--instant-sample-ms", str(LIVE_GRID_MS), *align,
    ]
    fitting_by_model: dict[str, Path] = {}
    for model in models:
        w_p = float(registry.model(model).trs.w_p)
        fit_label = label_def_from_args(label_args, model)
        slo = label_cli_args(fit_label)
        # The re-windower labels every row with the three arms of this primary label.
        rw_label = [
            "--ttft-slo-ms", str(args.ttft_slo_ms), "--tpot-slo-ms", str(args.tpot_slo_ms),
            *label_mode_cli_args(fit_label),
            *(["--registry", str(args.registry)] if getattr(args, "registry", None) else []),
        ]
        fitting_csv = fit_dir / f"{model}_fitting.csv"
        aliasing_csv = fit_dir / f"{model}_aliasing.csv"
        validation_csv = fit_dir / f"{model}_validation.csv"
        fitting_by_model[model] = fitting_csv
        rewindow_head = [
            sys.executable, "-m", "scripts.rewindow_from_raw",
            "--model", model, "--raw-dir", str(raw_dir),
        ]
        plan["rewindow"].append({
            "purpose": "fitting (the signal the controller consumes)",
            "model": model,
            "output": str(fitting_csv),
            "excludes_held_out": True,
            "command": [*rewindow_head, "--output", str(fitting_csv), *live, *rw_label, *train_sel],
        })
        plan["rewindow"].append({
            "purpose": "aliasing figure and observability gap (ground truth)",
            "model": model,
            "output": str(aliasing_csv),
            "excludes_held_out": True,
            "command": [
                *rewindow_head, "--output", str(aliasing_csv),
                "--window-ms", str(args.window_ms), "--step-ms", str(args.fit_step_ms),
                "--instant-grid", "raw",
                "--instant-sample-ms", str(args.instant_sample_ms), *align, *rw_label,
                *train_sel,
            ],
        })
        if holdout_sel:
            plan["rewindow"].append({
                "purpose": "held-out validation set (never fitted on)",
                "model": model,
                "output": str(validation_csv),
                "excludes_held_out": False,
                "command": [
                    *rewindow_head, "--output", str(validation_csv), *live, *rw_label,
                    *holdout_sel,
                ],
            })

        alpha_json = fit_dir / f"{model}_alpha.json"
        plan["alpha"].append({
            "model": model,
            "input": str(fitting_csv),
            "w_p": w_p,
            "lambda_wait": PRIMARY_LAMBDA_WAIT,
            "output": str(alpha_json),
            "command": [
                sys.executable, "-m", "scripts.alpha_fit",
                "--model", model,
                "--fitting-csv", str(fitting_csv),
                "--w-p", str(w_p),
                "--lambda-wait", str(PRIMARY_LAMBDA_WAIT),
                *slo,
                "--step-ms", str(args.fit_step_ms),
                *ledger_args,
                "--output", str(alpha_json),
            ],
        })

        dline_out = fit_dir / "dline"
        for arm in ("primary", "fixed"):
            for stage in ("alpha", "wp", "final"):
                plan["dline"].append({
                    "model": model, "arm": arm, "stage": stage,
                    "output": str(dline_out / model / arm / f"{stage}.json"),
                    "command": [
                        sys.executable, "-m", "scripts.dline_refit", stage,
                        "--model", model, "--arm", arm,
                        "--fit-dir", str(fit_dir), "--out-dir", str(dline_out),
                        *ledger_args,
                        *(["--registry", str(args.registry)] if getattr(args, "registry", None) else []),
                    ],
                })

        scopes: list[tuple[str, str, Path]] = [("", "", fitting_csv)]
        for family, shapes in sorted(families.items()):
            family_csv = fit_dir / f"{model}_fitting_{family}.csv"
            scopes.append((family, f"family_{family}", family_csv))
            plan["rewindow"].append({
                "purpose": f"per-family fit ({family}) - diagnostic",
                "model": model,
                "family": family,
                "shapes": list(shapes),
                "output": str(family_csv),
                "excludes_held_out": True,
                "command": [
                    *rewindow_head, "--output", str(family_csv), *live, *rw_label,
                    *[a for shape in shapes for a in ("--only-shape", shape)],
                    *train_sel,
                ],
            })

        for family, scope, csv_path in scopes:
            for suffix, lambda_wait in (("", PRIMARY_LAMBDA_WAIT), ("_lw0", SECONDARY_LAMBDA_WAIT)):
                if family:
                    label = f"{scope}{suffix}"
                else:
                    label = "primary" if not suffix else "secondary"
                entry = {
                    "label": label,
                    "model": model,
                    "family": family,
                    "lambda_wait": lambda_wait,
                    "w_p": w_p,
                    "input": str(csv_path),
                    "command": [
                        sys.executable, "-m", "tre_calibration.cli",
                        "--input", str(csv_path),
                        "--output", str(fit_dir / f"{model}_theta_{label}.json"),
                        "--model-name", model,
                        *slo,
                        "--recompute-tss",
                        "--w-p", str(w_p),
                        "--lambda-wait", str(lambda_wait),
                        "--ema-tau-ms", str(DEFAULT_EMA_TAU_MS),
                    ],
                }
                if family:
                    entry["shapes"] = list(families[family])
                    entry["purpose"] = (
                        "diagnostic only - its theta is compared against the merged fit's "
                        "bootstrap CI, never published on its own unless the families disagree"
                    )
                plan["theta"].append(entry)

        verdict_json = fit_dir / f"{model}_verdict.json"
        plan["verdict"].append({
            "model": model,
            "lambda_wait": PRIMARY_LAMBDA_WAIT,
            "output": str(verdict_json),
            "command": [
                sys.executable, "-m", "scripts.theta_verdict", "verdict",
                "--model", model,
                "--fitting-csv", str(fitting_csv),
                *[a for family, _scope, path in scopes if family for a in ("--family", f"{family}={path}")],
                "--w-p", str(w_p),
                "--lambda-wait", str(PRIMARY_LAMBDA_WAIT),
                "--ema-tau-ms", str(DEFAULT_EMA_TAU_MS),
                *slo,
                "--output", str(verdict_json),
            ],
        })
        family_args = [a for family, _scope, path in scopes if family for a in ("--family", f"{family}={path}")]
        arm_verdicts: list[tuple[str, Path]] = [("tss", verdict_json)]
        for arm, arm_w_p, arm_lambda in ablation_arms(w_p):
            arm_json = fit_dir / f"{model}_verdict_{arm}.json"
            arm_verdicts.append((arm, arm_json))
            plan["ablation"].append({
                "model": model,
                "arm": arm,
                "w_p": arm_w_p,
                "lambda_wait": arm_lambda,
                "label_lambda_wait": PRIMARY_LAMBDA_WAIT,
                "output": str(arm_json),
                "command": [
                    sys.executable, "-m", "scripts.theta_verdict", "verdict",
                    "--model", model,
                    "--fitting-csv", str(fitting_csv),
                    *family_args,
                    "--signal", "tss",
                    "--w-p", str(arm_w_p),
                    "--lambda-wait", str(arm_lambda),
                    "--label-lambda-wait", str(PRIMARY_LAMBDA_WAIT),
                    "--ema-tau-ms", str(DEFAULT_EMA_TAU_MS),
                    *slo,
                    "--output", str(arm_json),
                ],
            })
        for signal in ALT_SIGNALS:
            arm_verdicts.append((signal, fit_dir / f"{model}_verdict_{signal}.json"))
        if holdout_sel:
            for arm, arm_json in arm_verdicts:
                out_json = fit_dir / (f"{model}_holdout.json" if arm == "tss" else f"{model}_holdout_{arm}.json")
                plan["holdout"].append({
                    "model": model,
                    "arm": arm,
                    "input": str(validation_csv),
                    "verdict": str(arm_json),
                    "output": str(out_json),
                    "command": [
                        sys.executable, "-m", "scripts.theta_verdict", "holdout",
                        "--verdict", str(arm_json),
                        "--validation-csv", str(validation_csv),
                        "--output", str(out_json),
                    ],
                })

    if models:
        plan["dline"].append({
            "stage": "summary",
            "output": str(fit_dir / "dline" / "summary.json"),
            "command": [
                sys.executable, "-m", "scripts.dline_refit", "summary",
                *[a for m in models for a in ("--model", m)],
                "--fit-dir", str(fit_dir), "--out-dir", str(fit_dir / "dline"), *ledger_args,
            ],
        })

    fit_alt = Path(__file__).resolve().parents[2] / "calibration" / "scripts" / "fit_alt_thresholds.py"
    for signal in ALT_SIGNALS:
        plan["alt"].append({
            "signal": signal,
            "output": str(fit_dir / f"alt_{signal}.yaml"),
            "verdicts": {m: str(fit_dir / f"{m}_verdict_{signal}.json") for m in fitting_by_model},
            "command": [
                sys.executable, str(fit_alt),
                *[a for m, path in fitting_by_model.items() for a in ("--model-input", f"{m}={path}")],
                *[
                    a
                    for m in fitting_by_model
                    for family in sorted(families)
                    for a in ("--family", f"{m}:{family}={fit_dir / f'{m}_fitting_{family}.csv'}")
                ],
                # several models in one fit: each resolves its own idle TTFT fit from the
                # registry, so the per-model c/b overrides are left out here.
                *_without_idle_fit(slo),
                "--signal", signal,
                "--label-lambda-wait", str(PRIMARY_LAMBDA_WAIT),
                "--ema-tau-ms", str(DEFAULT_EMA_TAU_MS),
                "--verdict-dir", str(fit_dir),
                "--output", str(fit_dir / f"alt_{signal}.yaml"),
                "--curve-dir", str(fit_dir / "alt_curves"),
            ],
        })
    return plan


#: The alternative signals of the ablation (tre_calibration.alt_signals.ALT_SIGNALS).
ALT_SIGNALS = ("queue_len", "decode_tps", "prefill_tps")


def _without_idle_fit(label_args: Sequence[str]) -> list[str]:
    out: list[str] = []
    skip = False
    for arg in label_args:
        if skip:
            skip = False
            continue
        if arg in ("--ttft-idle-c-ms", "--ttft-idle-b-ms-per-token"):
            skip = True
            continue
        out.append(arg)
    return out


def ablation_arms(w_p: float) -> list[tuple[str, float, float]]:
    """(arm, w_p, lambda_wait) of the TSS weighting ablation (plan §6.9): the waiting
    weight switched off, and prefill tokens counted like decode tokens."""
    return [
        ("tss_lw0", float(w_p), 0.0),
        ("tss_wp1", 1.0, PRIMARY_LAMBDA_WAIT),
    ]


def _load_registry(path: Optional[str]):
    from tre_common.registry import load_registry

    return load_registry(path)


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
        # A re-drive gets its own schedule file: an inconclusive probe is re-driven for
        # longer, and overwriting attempt 1's schedule would lose what attempt 1 ran.
        schedule_stem = stem if probe.attempt <= 1 else f"{stem}_a{probe.attempt}"
        schedule_path = schedule_dir / cell.model / f"{schedule_stem}.json"
        schedule_path.parent.mkdir(parents=True, exist_ok=True)
        schedule_path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        (schedule_path.parent / f"{schedule_stem}.meta.json").write_text(
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
        label=primary_label(args, cell.model),
        capacity_source=measured.capacity_source,
        search=new_boundary_search(cell.model, cell.shape, args),
    )


def new_boundary_search(model: str, shape: str, args) -> boundary.BoundarySearch:
    """The search a campaign drives: coarse probe length from ``--boundary-coarse-s``."""
    return boundary.BoundarySearch(
        model=model, shape=shape,
        coarse_seconds=float(getattr(args, "boundary_coarse_s", boundary.COARSE_SECONDS)),
    )


# ------------------------------------------------------------------ re-probe mode


def parse_reprobe_shapes(items: Sequence[str]) -> dict[str, list[str]]:
    """``MODEL:SHAPE[,SHAPE...]`` (repeatable) -> {model: [shapes]}. Only training shapes:
    the held-out mixture never gets a boundary search (see run_campaign)."""
    out: dict[str, list[str]] = {}
    for item in items:
        model, sep, shapes = str(item).partition(":")
        if not sep or not model or not shapes:
            raise ValueError(f"--reprobe-shapes {item!r}: expected MODEL:SHAPE[,SHAPE...]")
        for shape in (x.strip() for x in shapes.split(",")):
            if not shape:
                continue
            if shape not in gen.TRAINING_SHAPES or gen.is_held_out(shape):
                raise ValueError(f"--reprobe-shapes {item!r}: {shape!r} is not a training shape "
                                 f"({', '.join(gen.TRAINING_SHAPES)})")
            if shape not in out.setdefault(model, []):
                out[model].append(shape)
    if not out:
        raise ValueError("--reprobe-shapes: nothing to re-probe")
    return out


def check_new_output_root(out_root: Path, source: Path) -> None:
    """A re-probe writes into a NEW root: never into the source campaign, never inside or
    around it, never into a non-empty directory (a re-measured rho* must not overwrite or
    mix with the numbers it replaces)."""
    out_root = Path(out_root).resolve()
    source = Path(source).resolve()
    if out_root == source or source in out_root.parents or out_root in source.parents:
        raise ValueError(f"re-probe output root {out_root} overlaps the source campaign {source}")
    if out_root.exists() and any(out_root.iterdir()):
        raise ValueError(f"re-probe output root {out_root} exists and is not empty - pick a new one")


def load_source_capacity(source: Path, model: str, shape: str) -> MeasuredCapacity:
    path = Path(source) / model / "capacity" / f"{model}_{shape}.json"
    if not path.exists():
        raise ValueError(f"no capacity measurement {path}: the re-probe reuses the source "
                         "campaign's steps capacity, it does not re-run steps")
    raw = json.loads(path.read_text(encoding="utf-8"))
    levels = tuple(StepLevel(**level) for level in raw.get("levels") or ())
    return MeasuredCapacity(**{**raw, "levels": levels})


def reprobe_plan(args, targets: Mapping[str, Sequence[str]]) -> dict:
    coarse_s = float(getattr(args, "boundary_coarse_s", boundary.COARSE_SECONDS))
    entries = []
    for model, shapes in targets.items():
        for shape in shapes:
            measured = load_source_capacity(args.reprobe_source, model, shape)
            entries.append({
                "model": model,
                "shape": shape,
                "capacity_used_rps": measured.capacity_used_rps,
                "capacity_source": measured.capacity_source,
                "capacity_file": str(Path(args.reprobe_source) / model / "capacity" / f"{model}_{shape}.json"),
                "output_dir": str(Path(args.out_dir) / model),
                "boundary_json": str(Path(args.out_dir) / model / "boundary" / f"{model}_{shape}.json"),
            })
    per_shape = boundary.shape_seconds(coarse_s)
    extra = boundary.max_extension_seconds(coarse_s)
    probes = boundary.probe_count() + boundary.max_extension_probes()
    return {
        "generated_at_utc": utc_iso(),
        "mode": "reprobe",
        "source_campaign": str(args.reprobe_source),
        "output_root": str(args.out_dir),
        "coarse_seconds": coarse_s,
        "coarse_rhos": list(boundary.COARSE_RHOS),
        "extend_down_rhos": list(boundary.EXTEND_DOWN_RHOS),
        "extend_up_rhos": list(boundary.EXTEND_UP_RHOS),
        "targets": entries,
        "estimated_wall_clock_s": round(len(entries) * (per_shape + args.cooldown_s * boundary.probe_count()), 1),
        "estimated_max_wall_clock_s": round(len(entries) * (per_shape + extra + args.cooldown_s * probes), 1),
    }


def run_reprobe(args, targets: Mapping[str, Sequence[str]]) -> int:
    """``--reprobe-shapes``: re-measure only the listed (model, shape) boundaries.

    Capacity comes from the source campaign's steps measurement (steps is not re-run);
    every probe, raw file and boundary JSON goes under ``<out-dir>/<model>/`` of a NEW
    root laid out like a campaign out-dir (``boundary/``, ``capacity/``, ``schedules/``,
    ``raw/``), so ``static_grid --static-grid-reprobe <out-dir>`` can overlay it on the
    source campaign. The source campaign is only read."""
    check_new_output_root(args.out_dir, args.reprobe_source)
    plan = reprobe_plan(args, targets)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "reprobe_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for entry in plan["targets"]:
        print(f"  reprobe {entry['model']:12} {entry['shape']:3} C_s={entry['capacity_used_rps']:.3f} "
              f"({entry['capacity_source']}) -> {entry['boundary_json']}")
    print(f"{len(plan['targets'])} boundary re-probes, ~{plan['estimated_wall_clock_s'] / 3600.0:.2f} h "
          f"(<= {plan['estimated_max_wall_clock_s'] / 3600.0:.2f} h with every extension probe)")
    if args.dry_run:
        print(f"dry run: wrote {out_root / 'reprobe_plan.json'}")
        return 0

    mode = controller_mode(args.controller_namespace)
    if mode != REQUIRED_CONTROLLER_MODE:
        raise SystemExit(f"controller mode is {mode!r}, refusing to run (need {REQUIRED_CONTROLLER_MODE!r})")
    index_path = Path(args.index)
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    cap = admission.get_cap(args.cap or index.get("admission_cap", {}).get("name")
                            or admission.DEFAULT_CAP_NAME)
    first = True
    for model, shapes in targets.items():
        model_dir = out_root / model
        for sub in ("boundary", "capacity", "schedules", "raw"):
            (model_dir / sub).mkdir(parents=True, exist_ok=True)
        model_args = argparse.Namespace(**{**vars(args), "raw_dir": model_dir / "raw"})
        for shape in shapes:
            measured = load_source_capacity(args.reprobe_source, model, shape)
            (model_dir / "capacity" / f"{model}_{shape}.json").write_text(
                json.dumps({**measured.as_dict(), "copied_from": str(args.reprobe_source)}, indent=2) + "\n",
                encoding="utf-8",
            )
            if not first:
                time.sleep(args.cooldown_s)
            first = False
            cell = Cell(model=model, shape=shape, primitive=gen.HOLD_PRIMITIVE, cell_id="",
                        schedule="", duration_s=0.0, capacity_rps=measured.capacity_used_rps,
                        capacity_source=measured.capacity_source)
            search = drive_boundary_search(cell, measured, model_args, cap=cap,
                                           schedule_dir=model_dir / "schedules", out_dir=model_dir)
            body = {**search.as_dict(), "reprobe": {"source_campaign": str(args.reprobe_source)}}
            (model_dir / "boundary" / f"{model}_{shape}.json").write_text(
                json.dumps(body, indent=2) + "\n", encoding="utf-8"
            )
            print(f"reprobe {model}/{shape}: rho* = {search.rho_star} ({body['rho_star_status']})"
                  + (f"; {search.unresolved_reason}" if search.unresolved_reason else "")
                  + (f"; STOPPED: {search.stopped_reason}" if search.stopped_reason else ""))
    print(f"re-probe complete; artifacts under {out_root}")
    return 0


def static_grid_enabled(args) -> bool:
    return bool(getattr(args, "static_grid", False) or getattr(args, "static_grid_only", False))


def plan_static_cells(args, models: Sequence[str]) -> tuple[list, dict]:
    """(static cells, surfaces) for ``--static-grid``, from ``--static-grid-source`` (with
    ``--static-grid-reprobe`` boundary / capacity files overlaid on it).

    Refuses when a source shape's rho* is not ``measured`` (a bound or a bisection grid
    point would place the grid on a guess - plan 6.11); ``--static-grid-allow-unmeasured``
    turns the refusal into a loud warning."""
    import sys as _sys

    source = Path(args.static_grid_source)
    overlay = getattr(args, "static_grid_reprobe", None)
    surfaces = {m: static_grid.load_surface(source, m, overlay=overlay) for m in models}
    bad = {m: static_grid.unmeasured_shapes(s) for m, s in surfaces.items()}
    bad = {m: v for m, v in bad.items() if v}
    if bad:
        listing = "; ".join(
            f"{m}: " + ", ".join(f"{shape}={status}" for shape, status in sorted(v.items()))
            for m, v in sorted(bad.items())
        )
        message = (f"static grid source rho* not measured ({listing}) - re-probe these shapes "
                   "(--reprobe-shapes) and pass --static-grid-reprobe")
        if not getattr(args, "static_grid_allow_unmeasured", False):
            raise SystemExit(f"refusing to plan the static grid: {message}")
        banner = "!" * 78
        print(f"{banner}\nWARNING: {message}\n(--static-grid-allow-unmeasured: planning anyway)\n{banner}",
              file=_sys.stderr)
    cells = static_grid.plan_static_grid(
        models,
        surfaces,
        gpus=static_grid.model_gpus(models, getattr(args, "registry", None)),
        hold_s=args.static_grid_hold_s,
    )
    return cells, surfaces


def drive_static_cell(
    cell: "static_grid.StaticCell",
    args,
    *,
    cap: admission.AdmissionCap,
    schedule_dir: Path,
    out_dir: Path,
    raw_dir: Path,
    position: str,
) -> int:
    """Generate one static cell's constant-rate schedule and drive it like any other cell
    (void -> re-run once -> stop). Returns r3_grid's exit code; raises CellVoided."""
    body, meta = gen.build_schedule_from_capacity_rps(
        cell.model,
        cell.shape,
        gen.STATIC_PRIMITIVE,
        cell.capacity_rps,
        capacity_source=static_grid.CAPACITY_SOURCE,
        cap=cap,
        hold_rho=cell.rho,
        hold_duration_s=cell.duration_s,
        hold_stage="static_grid",
        static_fraction=cell.rho_over_rho_star,
    )
    assert body is not None and meta["cell_id"] == cell.cell_id
    schedule_path = schedule_dir / cell.model / f"{cell.stem}.json"
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    (schedule_path.parent / f"{cell.stem}.meta.json").write_text(
        json.dumps({**meta, "static_grid": cell.as_dict()}, indent=2) + "\n", encoding="utf-8"
    )
    run_cell = Cell(
        model=cell.model,
        shape=cell.shape,
        primitive=gen.STATIC_PRIMITIVE,
        cell_id=cell.cell_id,
        schedule=str(schedule_path),
        duration_s=cell.duration_s,
        capacity_rps=cell.capacity_rps,
        capacity_source=static_grid.CAPACITY_SOURCE,
        metadata=meta,
    )
    base_output = out_dir / f"{cell.model}_{cell.stem}.csv"
    exit_code = 0

    def drive(attempt: int) -> dict:
        nonlocal exit_code
        output = attempt_output_path(base_output, attempt)
        if attempt > 1:
            time.sleep(args.cooldown_s)
        command = cell_command(run_cell, args, schedule_path, output)
        print(f"[{position}] static {cell.model} {cell.shape} rho/rho*={cell.rho_over_rho_star:g} "
              f"rho={cell.rho:g} ({cell.duration_s:.0f}s, attempt {attempt}): {' '.join(command)}")
        result = subprocess.run(command, check=False)
        exit_code = result.returncode
        if result.returncode != 0:
            print(f"cell failed with exit {result.returncode}")
        return read_cell_guard(raw_dir, output, cell.cell_id, returncode=result.returncode)

    drive_until_valid(cell.cell_id, drive)
    return exit_code


def registry_path_for(args) -> Path:
    """The registry file this campaign's cells load (the r3_grid default when unset)."""
    if getattr(args, "registry", None):
        return Path(args.registry)
    return Path(__file__).resolve().parents[1] / "registry.yaml"


def file_sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def git_state(worktree: Path) -> dict:
    """The code commit a run was made with, and whether the tree had local changes."""
    def git(*argv: str) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["git", "-C", str(worktree), *argv],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def run_provenance(args) -> dict:
    """What a run was made with, recorded before it drives anything."""
    registry = registry_path_for(args)
    models = [m for m in str(getattr(args, "models", "") or "").split(",") if m]
    membership = "(start, end]" if getattr(args, "fit_window_align", "grid") == "grid" else "[start, end)"
    labels = {
        m: slo_labels.label_definition(
            primary_label(args, m),
            min_latency_samples=getattr(args, "min_latency_samples", slo_labels.DEFAULT_MIN_LATENCY_SAMPLES),
            window_membership=membership,
        )
        for m in models
    }
    return {
        "code": git_state(Path(__file__).resolve().parents[2]),
        "registry_path": str(registry),
        "registry_sha256": file_sha256(registry),
        "window_ms": args.window_ms,
        "step_ms": args.fit_step_ms,
        "instant_sample_ms": args.instant_sample_ms,
        "window_align": getattr(args, "fit_window_align", "grid"),
        "label": labels[models[0]] if models else None,
        "label_by_model": labels,
        "boundary": {
            "coarse_seconds": float(getattr(args, "boundary_coarse_s", boundary.COARSE_SECONDS)),
            "min_probe_windows": boundary.MIN_PROBE_WINDOWS,
            "violation_window_fraction": boundary.VIOLATION_WINDOW_FRACTION,
            "inconclusive_duration_factor": boundary.INCONCLUSIVE_DURATION_FACTOR,
        },
    }


CAMPAIGN_STATUS_FILE = "campaign_status.json"


def finalize_run(out_dir: Path, *, status: str, exit_code: int) -> None:
    """Record how the campaign ended, then build the standard dataset.

    Never raises: the dataset is a conversion of what is on disk and can always be
    rebuilt by hand (``python -m scripts.calibration_dataset <run>``); a failure here
    must not turn a finished campaign into a failed one.
    """
    out_dir = Path(out_dir)
    (out_dir / CAMPAIGN_STATUS_FILE).write_text(
        json.dumps(
            {"status": status, "exit_code": exit_code, "finished_at_utc": utc_iso()},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    try:
        from scripts import calibration_dataset

        built = calibration_dataset.build_dataset(out_dir)
        print(f"standard dataset: {built}")
        parent = out_dir.parent
        siblings = calibration_dataset.campaign_dirs(parent)
        if len(siblings) > 1 and all((d / CAMPAIGN_STATUS_FILE).exists() for d in siblings):
            merged = calibration_dataset.build_dataset(parent)
            print(f"every campaign under {parent} has finished; merged dataset: {merged}")
    except Exception as exc:  # noqa: BLE001 - see docstring
        print(f"WARNING: building the standard dataset failed ({exc!r}); rebuild it with "
              f"python -m scripts.calibration_dataset {out_dir}")


def run_campaign(args) -> int:
    coarse_s = float(getattr(args, "boundary_coarse_s", boundary.COARSE_SECONDS))
    if coarse_s < boundary.min_probe_seconds(args.window_ms):
        # --boundary-coarse-s 60 reproduces the 2026-09-21 campaign on purpose; say what
        # it costs instead of refusing.
        print(
            f"WARNING: a {coarse_s:g} s coarse probe cannot hold "
            f"{boundary.MIN_PROBE_WINDOWS} disjoint {args.window_ms} ms windows, so every "
            "coarse probe is inconclusive and re-driven at twice its length"
        )
    index_path = Path(args.index)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    cap = admission.get_cap(args.cap or index.get("admission_cap", {}).get("name")
                            or admission.DEFAULT_CAP_NAME)
    models = [m for m in args.models.split(",") if m]
    runnable, skipped = build_plan(index, models)
    static_cells: list = []
    static_surfaces: dict = {}
    if static_grid_enabled(args):
        static_cells, static_surfaces = plan_static_cells(args, models)
    if getattr(args, "static_grid_only", False):
        runnable, skipped = [], []

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
    boundary_cells = boundary_plan(
        models, boundary_shapes,
        coarse_seconds=float(getattr(args, "boundary_coarse_s", boundary.COARSE_SECONDS)),
    )
    schedule_seconds = estimate_wall_clock_s(runnable, args.cooldown_s)
    boundary_seconds = estimate_boundary_wall_clock_s(boundary_cells, args.cooldown_s)
    static_seconds = sum(c.duration_s + args.cooldown_s for c in static_cells)
    plan_doc = {
        "generated_at_utc": utc_iso(),
        "provenance": run_provenance(args),
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
        "estimated_static_grid_wall_clock_s": round(static_seconds, 1),
        "estimated_wall_clock_s": round(schedule_seconds + boundary_seconds + static_seconds, 1),
        "static_grid": {
            "enabled": static_grid_enabled(args),
            "only": bool(getattr(args, "static_grid_only", False)),
            "cells": [c.as_dict() for c in static_cells],
            "surfaces": {m: s.as_dict() for m, s in static_surfaces.items()},
            "estimate": static_grid.estimate(static_cells, args.cooldown_s) if static_cells else {},
        },
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
    if static_cells:
        print(static_grid.format_listing(static_cells, args.cooldown_s))

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

    status, code = "failed", 1
    try:
        code = _drive_campaign(args, index=index, cap=cap, runnable=runnable,
                               out_dir=out_dir, raw_dir=raw_dir, schedule_root=schedule_root)
        status = "complete" if code == 0 else "stopped"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        finalize_run(out_dir, status=status, exit_code=code)
    return code


def _drive_campaign(args, *, index, cap, runnable, out_dir, raw_dir, schedule_root) -> int:
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
                        f"boundary {cell.model}/{cell.shape}: rho* = {search.rho_star} "
                        f"({search.status()['status']}), dwelled at {search.dwell_rho}"
                        + ("" if search.boundary_found else " (NO violation was observed - "
                           "the boundary is above everything offered)")
                        + (f"; STOPPED: {search.stopped_reason}" if search.stopped_reason else "")
                    )

        if position < len(runnable):
            time.sleep(args.cooldown_s)

    # Static grid last: a campaign truncated here still has every default stage.
    for number, static_cell in enumerate(static_cells, start=1):
        if runnable or number > 1:
            time.sleep(args.cooldown_s)
        try:
            exit_code = drive_static_cell(
                static_cell, args, cap=cap, schedule_dir=regenerated_dir,
                out_dir=out_dir, raw_dir=raw_dir,
                position=f"static {number}/{len(static_cells)}",
            )
        except CellVoided as voided:
            print(str(voided))
            return 1
        if exit_code != 0 and args.stop_on_failure:
            return exit_code

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
    ap.add_argument("--fit-step-ms", type=int, default=10000,
                    help="slide step for the fitting re-window (the live decision cadence: "
                         "10 s with the phase-aligned sampler; a multiple of 10 s under "
                         "--fit-window-align grid)")
    ap.add_argument("--fit-window-align", default="grid", choices=["grid", "none"],
                    help="grid (default): fitting windows end on the gateway's 10 s grid "
                         "(rewindow_from_raw --window-align grid); none: legacy free-phase "
                         "windows, e.g. with --fit-step-ms 5000")
    ap.add_argument("--instant-sample-ms", type=int, default=1000)
    ap.add_argument("--cooldown-s", type=float, default=DEFAULT_COOLDOWN_S)
    ap.add_argument("--step-transient-s", type=float, default=DEFAULT_STEP_TRANSIENT_S)
    ap.add_argument("--ttft-slo-ms", type=float, default=500.0)
    ap.add_argument("--tpot-slo-ms", type=float, default=75.0)
    ap.add_argument("--fit-ttft-slo-mode", choices=["fixed", "slowdown"], default=DEFAULT_FIT_TTFT_SLO_MODE,
                    help="TTFT SLO of the fit label: slowdown = max(floor, k*(c_m+b_m*L)) with the "
                         "registry's idle TTFT fit (plan 6.9h); fixed = --ttft-slo-ms")
    ap.add_argument("--fit-ttft-slowdown-k", type=float, default=None,
                    help="default: registry slo.ttft_slowdown_k (5, D6'); 3 = the D6 ablation arm")
    ap.add_argument("--fit-ttft-floor-ms", type=float, default=None,
                    help="default: registry slo.ttft_floor_ms (500 ms, D6'); 150 = the D6 ablation arm")
    ap.add_argument("--fit-min-completed-requests", type=int, default=20)
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
                    help="write plan.json and fit_plan.json, print the time estimate, "
                         "drive nothing")
    ap.add_argument("--static-grid", action="store_true",
                    help="append the opt-in static steady-state grid (scripts.static_grid) "
                         "after the default stages; its shapes join the training set and "
                         "their families in fit_plan.json")
    ap.add_argument("--static-grid-only", action="store_true",
                    help="drive only the static grid (implies --static-grid)")
    ap.add_argument("--static-grid-list", action="store_true",
                    help="print the static-grid cells and the GPU-minute estimate, then "
                         "exit (reads the source campaign's JSONs only; needs no index)")
    ap.add_argument("--static-grid-source", type=Path,
                    default=static_grid.DEFAULT_SOURCE_CAMPAIGN,
                    help="campaign out-dir whose <model>/capacity and <model>/boundary "
                         "JSONs place the grid")
    ap.add_argument("--static-grid-hold-s", type=float, default=gen.STATIC_GRID_HOLD_S)
    ap.add_argument("--static-grid-reprobe", type=Path, default=None,
                    help="a --reprobe-shapes output root whose <model>/boundary and "
                         "<model>/capacity JSONs replace the source campaign's for those shapes")
    ap.add_argument("--static-grid-allow-unmeasured", action="store_true",
                    help="plan the static grid even when a source shape's rho* is a bound or "
                         "a grid artifact (prints a loud warning instead of refusing)")
    ap.add_argument("--boundary-coarse-s", type=float, default=boundary.COARSE_SECONDS,
                    help=f"coarse boundary probe length (default {boundary.COARSE_SECONDS:g} s = 3 "
                         f"windows; {boundary.LEGACY_COARSE_SECONDS:g} s reproduces the 2026-09-21 "
                         "campaign, whose 2-window coarse probes were all inconclusive)")
    ap.add_argument("--reprobe-shapes", action="append", default=[], metavar="MODEL:SHAPE[,SHAPE]",
                    help="re-measure only these shapes' SLO boundaries (repeatable), reusing the "
                         "source campaign's steps capacity, into the NEW root --out-dir; runs "
                         "nothing else")
    ap.add_argument("--reprobe-source", type=Path, default=static_grid.DEFAULT_SOURCE_CAMPAIGN,
                    help="campaign root whose <model>/capacity JSONs the re-probe reuses")
    ap.add_argument("--design", choices=["ladder", "primitives"], default=None,
                    help="ladder (default): the second round's design (scripts.calibration_ladder). "
                         "primitives: the first round's steps / boundary / ramp / bursts - "
                         "implied by --static-grid* and --reprobe-shapes, which only it has")
    ap.add_argument("--rho-priors", type=Path, default=None,
                    help="ladder design: rho_priors.json - per (model, shape) the first "
                         "round's client-side boundary, C_s and, where it never violated, the "
                         "widened search range. Required; checked before anything runs")
    ap.add_argument("--regime-groups", type=Path, default=None,
                    help="ladder design: regime_groups.json - the 7 training shapes in 3 "
                         "regime groups (LORO units). Required; checked before anything runs")
    ap.add_argument("--design-seed", type=int, default=20260923,
                    help="ladder design: seed of every order and every per-cell seed; "
                         "recorded in the run manifest")
    ap.add_argument("--preregistration", type=Path,
                    default=here / "docs" / "preregistration-20260923-calibration-run2.md",
                    help="ladder design: the preregistration the run implements; its commit "
                         "is recorded in the run manifest")
    args = ap.parse_args(argv)
    primitives_only = bool(
        args.static_grid or args.static_grid_only or args.static_grid_list
        or args.reprobe_shapes or args.skip_boundary_search
    )
    if args.design is None:
        args.design = "primitives" if primitives_only else "ladder"
    elif args.design == "ladder" and primitives_only:
        ap.error("--static-grid* / --reprobe-shapes / --skip-boundary-search belong to "
                 "--design primitives")
    if args.static_grid_list:
        models = [m for m in args.models.split(",") if m]
        cells, _surfaces = plan_static_cells(args, models)
        print(static_grid.format_listing(cells, args.cooldown_s))
        return 0
    if args.cooldown_s * 1000.0 < args.window_ms:
        # Quiet time must cover at least one metrics window: the offline fit resets the
        # EMA per cell, which matches the controller only when the online EMA saw an idle
        # window between cells (tre_common.tss.TssEma idle rules).
        ap.error(
            f"--cooldown-s ({args.cooldown_s:g}) must be >= the metrics window "
            f"(--window-ms {args.window_ms} = {args.window_ms / 1000.0:g} s)"
        )
    if args.fit_window_align == "grid" and args.fit_step_ms % LIVE_GRID_MS:
        ap.error(
            f"--fit-window-align grid needs --fit-step-ms to be a multiple of {LIVE_GRID_MS} "
            f"(got {args.fit_step_ms}); use --fit-window-align none for a free-phase step"
        )
    if args.reprobe_shapes:
        try:
            targets = parse_reprobe_shapes(args.reprobe_shapes)
            return run_reprobe(args, targets)
        except ValueError as exc:
            ap.error(str(exc))
    if args.design == "ladder":
        from scripts import calibration_ladder

        return calibration_ladder.run_ladder_campaign(args)
    return run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())

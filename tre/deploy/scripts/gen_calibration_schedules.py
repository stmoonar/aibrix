#!/usr/bin/env python3
"""Generate the open-loop calibration schedule files (one per model x shape x primitive).

Output layout (committed, so a campaign run is reproducible from the tree alone)::

    replayer/traces_v2/calibration/
        INDEX.json                      # every cell + its provenance and metadata
        <model>/<shape>_<primitive>.json   # replayer trace.json schema

Each schedule file uses the ordinary model-keyed replayer trace schema
(``{model: [{start_time, end_time, rps, input_tokens, max_tokens}, ...]}``), so it loads
with ``tre_replayer.traces.loader.load_trace_segments`` and runs with
``r3_grid.py --schedule``. Overlapping segments superpose, because
``build_poisson_schedule`` samples each segment independently - that is how the bursts
primitive lays a spike on top of its base rate and how the mixture shape runs four
parallel token-shape streams.

What actually limits admission
------------------------------
Every primitive here is sized against an :class:`~scripts.admission_cap.AdmissionCap`:
the Envoy circuit breaker (``maxParallelRequests`` + ``maxPendingRequests``, per cluster,
shared across replicas) together with vLLM's own per-pod running-set limit. Which of the
two binds decides what a primitive can observe at all - the gateway *sheds* past its
ceiling and the request never reaches vLLM, whereas the engine *queues*, and only a
queue moves ``vllm:num_requests_waiting``. Under the deployed ``gateway-capped`` policy
the shed ceiling is 320 and vLLM's sequence limit is the unreachable 1024 default, so
the only remaining route to a non-zero waiting count is KV-cache exhaustion; under the
proposed ``engine-capped`` policy the binding limit moves into the engine and every
shape becomes observable. ``--cap`` selects the policy, and ``index["admission_cap"]``
records which one a schedule set was generated for.

The shape set
-------------
S1..S5 are the synthetic corners of the (input, output) plane. **T8** (1600 in / 112 out)
is the working point of the real trace t8 - the Azure conversation inputs it replays have
a median of 1628 tokens - so a theta fitted here is interpolating to the operating point
the experiments run at instead of extrapolating to it. **T9** is the first shape whose
lengths are *sampled per request* (log-uniform 300..2200 in, 100..580 out): every fixed
shape produces a batch that turns over in lockstep, with no decode tail at all, and a
tail is what distinguishes decode-bound from prefill-bound saturation. **M** stays the
held-out mixture and is in :data:`ALL_SHAPES` but never in :data:`TRAINING_SHAPES`.

Per-request lengths are drawn by ``tre_replayer.engine.schedule`` from a per-segment RNG
seeded off the schedule seed, so a re-run sends the same lengths request for request; the
prompt itself is still built by the same natural-language, exactly-fitted, unique-per-
request path as every other shape, because the sender keys the prompt on the request id
and fits it to whatever length that request drew.

The primitives
--------------
ramp    rho 0.4 -> 1.2 linearly over ``T_r = cap.ramp_seconds(C_s, COEF)`` seconds in
        ~5 s segments, hold 1.2 for ``T_r / 4``, then 60 s at rho 0.5 to drain. ``COEF``
        is :data:`RAMP_BACKLOG_COEFFICIENT`, the backlog one ramp+hold accumulates per
        ``C_s * T_r``, derived from the rho endpoints rather than restated; T_r is then
        whatever keeps that backlog inside the cap's excess-request headroom (146 slots
        quantised to 140, under the deployed policy). Backlog accrues at up to
        ``0.2 * C_s`` per second once rho > 1, so the crossing is still guaranteed. A
        fixed 0.8 -> 1.6 / 360 s ramp - the previous shape - overshoots that headroom by
        a wide margin and would have been truncated by gateway shedding rather than by
        the engine, i.e. it would have measured the proxy, not the model.
steps   rho 0.5 (90 s) -> 0.8 (120 s) -> 0.95 (240 s), monotone. 450 s. Each level is
        long enough to reach steady state; the first control window after each step is
        transient and is discarded downstream (``discard_after_s`` in the index).
bursts  base rho 0.6 for 360 s with a 2 s spike every 90 s carrying
        ``B = cap.burst_sizing(K_kv, tokens_per_request).requests`` requests: enough to
        overshoot the engine's running limit ``min(max_num_seqs, floor(K_kv /
        tokens_per_request))``, because the engine only queues past that limit. 4 bursts.
        This is the only primitive that drives a waiting queue, which is what makes
        lambda_wait identifiable at all. When the overshoot needs more requests than the
        cap admits, the spike would be shed rather than queued, so ``reachable`` is False
        and no schedule is written - the index records the skip and the arithmetic behind
        it instead of shipping a cell that can only measure the gateway. Under the
        deployed engine-capped policy the admission budget is 3840 requests and the
        engine's running limit is ``max_num_seqs * replicas`` (256 at one replica), so
        nothing is skipped any more - which is what makes ``lambda_wait`` identifiable
        for every shape rather than only for the few that happened to fit under 240.
hold    one constant rate ``rho * C_s`` for a stated duration, generated at campaign time
        by the adaptive boundary search rather than committed. See
        :func:`build_hold_schedule` and ``scripts.adaptive_boundary``.

Capacity priors and the C_s model
---------------------------------
rho is relative to a per-shape single-pod capacity prior C_s (rps). The priors live in
``traces_v2/calibration/capacity/`` - a campaign-local copy, deliberately NOT the frozen
``traces_v2/capacity/`` set that experiment-3's traceset-v2 was generated from (that one
stays byte-unchanged for provenance). The dsqwen-14b prior there was re-measured on
2026-09-20 with the unique-per-request-prompt sender and prefix caching off; the frozen
2026-07-09 one is contaminated (its capacity RISES with prompt length: 14.9 -> 33.0 ->
32.0 rps for input 128 -> 512 -> 1024, the signature of an identical-prompt sender
against an engine with prefix caching on). The priors only cover a sparse (i, o) grid and none
of the campaign shapes sit on it, so nearest-neighbour would silently borrow a lighter
point's capacity (the trace-set failure documented in traces_v2/README.md).

Instead we fit a two-parameter physical model per model::

    1 / C(i, o) = i / P + o / D

P is the pod's prefill throughput (prompt tokens/s) and D its decode throughput
(generated tokens/s): serving R rps of shape (i, o) spends R*i/P of each second in
prefill and R*o/D in decode, and saturates when that sums to 1. The fit is a
least-squares solve on the measured 1/C points, so it interpolates smoothly and
extrapolates to shapes nobody measured, and it is monotone-decreasing in both i and o by
construction - which is the property the contaminated 14b prior violated.

For the mixture shape M the capacity prior is the load-weighted harmonic combination::

    C_M = 1 / sum_k(w_k / C(i_k, o_k))

i.e. the total rps at which the blended stream saturates the pod. Its
``tokens_per_request`` (the burst sizing input) is instead the load-weighted arithmetic
mean ``sum_k w_k * (i_k + o_k)``, because what fills the KV cache is tokens per admitted
request, not rps.

The same capacity files carry ``kv_cache_tokens``: the engine's own "GPU KV cache size"
startup log line, with the pod it was read from recorded alongside it.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from tre_replayer.engine.schedule import TokenRange

from scripts.admission_cap import (
    CAPS,
    DEFAULT_CAP_NAME,
    AdmissionCap,
    get_cap,
    ramp_backlog_coefficient,
)

#: Fixed-length training shapes. (input_tokens, max_output_tokens).
#:
#: T8 is not a synthetic corner like S1..S5: it is the working point of the real trace
#: t8, whose Azure conversation inputs have a median of 1628 tokens. A theta fitted only
#: on the synthetic grid is being asked to extrapolate to the operating point the
#: experiments are actually run at; T8 removes the extrapolation.
SHAPES: dict[str, tuple[int, int]] = {
    "S1": (256, 128),
    "S2": (768, 192),
    "S3": (2048, 96),
    "S4": (256, 448),
    "S5": (768, 384),
    "T8": (1600, 112),
}

#: Training shapes whose token lengths are drawn per request instead of fixed.
#:
#: Every fixed shape produces a *uniform* batch: each request generates exactly
#: ``max_tokens`` tokens and finishes after the same number of decode steps, so the batch
#: turns over in lockstep and there is no decode tail at all. Real traffic has one, and a
#: tail is what makes decode-bound saturation look different from prefill-bound
#: saturation. T9 draws both lengths log-uniformly over roughly the span the traces
#: cover, so long generations hold batch slots while short ones churn past them.
SAMPLED_SHAPES: dict[str, tuple[TokenRange, TokenRange]] = {
    "T9": (TokenRange(300, 2200), TokenRange(100, 580)),
}

#: Held-out validation shape: a mixture of four token shapes running in parallel.
#: NEVER used for fitting - it exists to test that a theta fit on the training shapes
#: generalises. :data:`TRAINING_SHAPES` is the list every fit-facing caller must use, and
#: :func:`is_held_out` is the single place that decides.
MIXTURE_NAME = "M"
MIXTURE: tuple[tuple[float, int, int], ...] = (
    (0.40, 256, 128),
    (0.30, 1024, 256),
    (0.20, 128, 384),
    (0.10, 3072, 64),
)

#: Shapes a theta may be fitted on, and the full set a campaign drives. The mixture is in
#: the second and not the first, by construction rather than by convention.
TRAINING_SHAPES: tuple[str, ...] = (*SHAPES, *SAMPLED_SHAPES)
ALL_SHAPES: tuple[str, ...] = (*TRAINING_SHAPES, MIXTURE_NAME)

#: Shape families, for the per-family theta diagnostic. A theta that differs between the
#: most prefill-heavy and the most decode-heavy shapes is regime-dependent and must not
#: be published as one number; see ``calibration_campaign.family_theta_verdict``.
PREFILL_FAMILY: tuple[str, ...] = ("S3", "T8")
DECODE_FAMILY: tuple[str, ...] = ("S4", "S5")
FAMILIES: dict[str, tuple[str, ...]] = {
    "prefill_heavy": PREFILL_FAMILY,
    "decode_heavy": DECODE_FAMILY,
}

MODELS = ("dsqwen-7b", "dsllama-8b", "dsqwen-14b")


def is_held_out(shape_name: str) -> bool:
    """True for shapes that exist to validate a fit and must never enter one."""
    return shape_name == MIXTURE_NAME


# ---- static steady-state grid (opt-in: calibration_campaign --static-grid) ----
#: v1's calibration (``/root/aibrix-main/python/tre/calibration_runs/20260530``) drove a
#: wide *static* grid - prompt 300..800 x output 300..800 step 100 x RPS 4..7 (14b 3..7),
#: 120 s per point - where v2 has seven shapes. The seven leave the (input, output) plane
#: empty between their corners, and every shape's boundary is then located on one
#: (input, output) point only; plan 2026-09-21 §6.9g keeps v1's grid as the one thing
#: worth borrowing, subsampled to a few cells per family.
#:
#: These shapes are **opt-in** and deliberately in neither :data:`TRAINING_SHAPES` nor
#: :data:`ALL_SHAPES`: the committed schedule set, the default campaign and the default
#: fit plan do not change. When a campaign runs them they are training cells (never held
#: out, never M) - :func:`training_shapes` / :func:`families` with ``static_grid=True``.
STATIC_PRIMITIVE = "static"
#: Lengths that fill the gaps between the seven shapes (S1/S4 at 256, S2/S5 at 768,
#: T8 at 1600, S3 at 2048 in; 96..448 out).
STATIC_GRID_INPUTS: tuple[int, ...] = (400, 1200)
STATIC_GRID_OUTPUTS: tuple[int, ...] = (160, 320)
#: Offered load of a static cell as a fraction of the shape's boundary load rho*
#: (rho/rho*): just below, at, and just past the SLO boundary, where theta is decided.
STATIC_GRID_RHO_FRACTIONS: tuple[float, ...] = (0.85, 1.0, 1.1)
#: Hold per static cell. Longer than v1's 120 s so a cell yields independent windows
#: after the EMA and the 30 s window have settled (plan §6.1: independent ~ windows / 6).
STATIC_GRID_HOLD_S = 300.0
#: Static cells' load codes: 2000 + round(100 * rho/rho*). Clear of the fixed primitives
#: (60/95/120) and of the boundary holds (1000 + round(100 * rho)), so a static cell can
#: never share a cell id - and therefore a raw directory - with any other cell.
STATIC_LOAD_CODE_BASE = 2000


def static_grid_shape_name(input_tokens: int, output_tokens: int) -> str:
    """``G<in>x<out>``: one token with no underscore, because ``rewindow_from_raw``
    reads the shape back from the cell directory name ``<model>_<shape>_<rest>``."""
    return f"G{int(input_tokens)}x{int(output_tokens)}"


STATIC_GRID_SHAPES: dict[str, tuple[int, int]] = {
    static_grid_shape_name(i, o): (i, o)
    for i in STATIC_GRID_INPUTS
    for o in STATIC_GRID_OUTPUTS
}

#: Family of a static-grid shape, from its prompt/output ratio i/o. The thresholds are
#: the ones the committed families already satisfy (prefill_heavy S3 21.3, T8 14.3;
#: decode_heavy S4 0.57, S5 2.0). A shape between the two is in no family: it enters the
#: merged fit only, like S1/S2/T9. The committed shapes keep their explicit membership.
STATIC_FAMILY_PREFILL_MIN_RATIO = 6.0
STATIC_FAMILY_DECODE_MAX_RATIO = 2.0


def static_grid_family(input_tokens: int, output_tokens: int) -> Optional[str]:
    ratio = float(input_tokens) / float(output_tokens)
    if ratio >= STATIC_FAMILY_PREFILL_MIN_RATIO:
        return "prefill_heavy"
    if ratio <= STATIC_FAMILY_DECODE_MAX_RATIO:
        return "decode_heavy"
    return None


def static_load_code(fraction: float) -> int:
    """Offered-load code of a static cell: ``2000 + round(100 * rho/rho*)``."""
    return STATIC_LOAD_CODE_BASE + max(1, int(round(100.0 * float(fraction))))


def training_shapes(*, static_grid: bool = False) -> tuple[str, ...]:
    """The shapes a theta may be fitted on; the static grid's only when it is enabled."""
    return (*TRAINING_SHAPES, *STATIC_GRID_SHAPES) if static_grid else TRAINING_SHAPES


def families(*, static_grid: bool = False) -> dict[str, tuple[str, ...]]:
    """:data:`FAMILIES`, plus each static-grid shape in the family its ratio assigns."""
    out = {name: tuple(members) for name, members in FAMILIES.items()}
    if static_grid:
        for shape, (i, o) in STATIC_GRID_SHAPES.items():
            family = static_grid_family(i, o)
            if family is not None:
                out[family] = (*out[family], shape)
    return out

# ---- primitive parameters (single source of truth; the index records them) ----
RAMP_RHO_START = 0.4
RAMP_RHO_END = 1.2
RAMP_SEGMENT_S = 5.0
RAMP_HOLD_FRACTION = 0.25
RAMP_DRAIN_RHO = 0.5
RAMP_DRAIN_S = 60.0
#: Backlog one ramp+hold accumulates, per C_s * T_r. Derived from the rho endpoints, so
#: changing the ramp shape cannot leave the budget arithmetic behind.
RAMP_BACKLOG_COEFFICIENT = ramp_backlog_coefficient(
    RAMP_RHO_START, RAMP_RHO_END, RAMP_HOLD_FRACTION
)
RAMP_DURATION_FORMULA = (
    "min(cap.ramp_max_s, cap.ramp_excess_budget / (backlog_coefficient * capacity_rps))"
)

STEPS: tuple[tuple[float, float], ...] = ((0.5, 90.0), (0.8, 120.0), (0.95, 240.0))

BURST_BASE_RHO = 0.6
BURST_DURATION_S = 360.0
BURST_PERIOD_S = 90.0
BURST_WIDTH_S = 2.0
BURST_FIRST_S = 60.0
BURST_COUNT = 4
BURST_REQUEST_FORMULA = (
    "ceil(cap.burst_overshoot_factor * min(cap.sequence_limit, "
    "floor(kv_cache_tokens / tokens_per_request))), skipped when it exceeds "
    "cap.burst_request_cap"
)

#: A constant-rho cell, generated at campaign time rather than committed: the adaptive
#: boundary search decides its rho from what the previous probes measured, so there is no
#: fixed grid to commit. Everything else about it - prompts, sender, raw schema, guard -
#: is the same as any other primitive.
HOLD_PRIMITIVE = "hold"

#: c<N> in the cell id is NOT a concurrency in schedule mode - it is an offered-load code,
#: round(100 * the primitive's characteristic rho). The cell id must stay parseable by
#: ``r3_grid.GridCell.from_scenario_id`` or ``rewindow_from_raw`` silently skips the file.
LOAD_CODE = {"ramp": round(100 * RAMP_RHO_END), "steps": 95, "bursts": 60}

PRIMITIVES = ("ramp", "steps", "bursts")


#: Hold cells' load codes are offset by this. Without it a probe at rho 0.95 would get
#: the same cell id as that shape's steps cell (code 95), 0.6 the same as its bursts cell
#: (60) and 1.2 the same as its ramp (120) - and since the raw tree is keyed by cell id,
#: the two captures would silently merge into one set of windows.
HOLD_LOAD_CODE_BASE = 1000


def hold_load_code(rho: float) -> int:
    """Offered-load code for a hold cell: ``round(100 * rho)`` like the fixed primitives,
    lifted clear of their codes so every probe is separately addressable."""
    return HOLD_LOAD_CODE_BASE + max(1, int(round(100.0 * float(rho))))


@dataclass(frozen=True)
class CapacityModel:
    """Per-model prefill/decode throughput fit; ``rps(i, o)`` is the capacity prior."""

    model: str
    prefill_tps: float
    decode_tps: float
    n_points: int
    rms_rel_error: float

    def rps(self, input_tokens: int, output_tokens: int) -> float:
        cost = input_tokens / self.prefill_tps + output_tokens / self.decode_tps
        if cost <= 0.0:
            raise ValueError("degenerate capacity model")
        return 1.0 / cost

    def mixture_rps(self, mixture: Sequence[tuple[float, int, int]]) -> float:
        total = sum(w / self.rps(i, o) for w, i, o in mixture)
        return 1.0 / total


def fit_capacity_model(model: str, points: Sequence[tuple[int, int, float]]) -> CapacityModel:
    """Least-squares fit of 1/C = i/P + o/D over measured (i, o, rps) points.

    Solves the 2x2 normal equations for a = 1/P, b = 1/D directly (no numpy dependency
    in the deploy scripts). Raises if the solution is not physically usable (a or b <= 0),
    which is exactly what a contaminated prior whose capacity *rises* with prompt length
    would produce - better a loud failure than a silently inverted capacity surface.
    """
    if len(points) < 2:
        raise ValueError(f"{model}: need >= 2 capacity points, got {len(points)}")
    sii = sio = soo = si_y = so_y = 0.0
    for i, o, rps in points:
        if rps <= 0.0:
            raise ValueError(f"{model}: non-positive rps at ({i},{o})")
        y = 1.0 / rps
        sii += i * i
        sio += i * o
        soo += o * o
        si_y += i * y
        so_y += o * y
    det = sii * soo - sio * sio
    if abs(det) < 1e-12:
        raise ValueError(f"{model}: capacity points are collinear; cannot separate P and D")
    a = (si_y * soo - so_y * sio) / det
    b = (so_y * sii - si_y * sio) / det
    if a <= 0.0 or b <= 0.0:
        raise ValueError(
            f"{model}: capacity fit is unphysical (1/P={a:.3e}, 1/D={b:.3e}). "
            "This means measured capacity does not decrease with token count - "
            "re-measure the prior before generating schedules."
        )
    fitted = CapacityModel(model, 1.0 / a, 1.0 / b, len(points), 0.0)
    errs = [(fitted.rps(i, o) - rps) / rps for i, o, rps in points]
    rms = (sum(e * e for e in errs) / len(errs)) ** 0.5
    return CapacityModel(model, 1.0 / a, 1.0 / b, len(points), rms)


def load_capacity_points(path: Path) -> tuple[str, list[tuple[int, int, float]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    points = [
        (int(p["input_tokens"]), int(p["output_tokens"]), float(p["rps"]))
        for p in data["capacity"]
    ]
    return str(data["model"]), points


def load_kv_cache_tokens(path: Path) -> int:
    """The engine's GPU KV cache size (tokens) recorded next to the capacity prior.

    Burst sizing is meaningless without it - with no KV number there is no way to know how
    many concurrent requests the engine can run before it starts queueing, and a guess
    would silently produce cells that only measure the Envoy gateway. So this fails
    loudly.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "kv_cache_tokens" not in data:
        raise ValueError(
            f"{path}: missing 'kv_cache_tokens'. Read it from the model's vLLM startup log "
            "line \"GPU KV cache size: N tokens\" (kubectl -n tre-v2 logs <pod> | grep 'KV cache "
            "size') and record it together with 'kv_cache_provenance' (source, measured, pod)."
        )
    tokens = int(data["kv_cache_tokens"])
    if tokens <= 0:
        raise ValueError(f"{path}: 'kv_cache_tokens' must be positive, got {tokens}")
    return tokens


# ------------------------------------------------------------------ shape helpers


def shape_components(
    shape_name: str,
) -> tuple[tuple[float, int | TokenRange, int | TokenRange], ...]:
    """(weight, input length, output length) streams that make up a shape.

    A length is an ``int`` for a fixed shape and a :class:`TokenRange` for a sampled one;
    everything downstream funnels through :func:`_length_mean` and :func:`_segment_shape`
    so the two never need a branch of their own again.
    """
    if shape_name == MIXTURE_NAME:
        return MIXTURE
    if shape_name in SAMPLED_SHAPES:
        i, o = SAMPLED_SHAPES[shape_name]
        return ((1.0, i, o),)
    if shape_name in STATIC_GRID_SHAPES:
        i, o = STATIC_GRID_SHAPES[shape_name]
        return ((1.0, i, o),)
    i, o = SHAPES[shape_name]
    return ((1.0, i, o),)


def _length_mean(length: int | TokenRange) -> float:
    """E[length]. For a :class:`TokenRange` this is its distribution mean, not its
    median: capacity and KV footprint are both linear in length, so the mean is what
    sets them."""
    return float(length.mean) if isinstance(length, TokenRange) else float(length)


def _length_nominal(length: int | TokenRange) -> int:
    """The single number that names this length in a cell id. Geometric midpoint for a
    range, so the id is stable and parseable without claiming a length nothing sent."""
    return int(length.median) if isinstance(length, TokenRange) else int(length)


def _segment_shape(length: int | TokenRange) -> tuple[str, object]:
    """(trace-schema key, value) for one length, fixed or sampled."""
    if isinstance(length, TokenRange):
        return "dist", length.as_dict()
    return "fixed", int(length)


def tokens_per_request(shape_name: str) -> float:
    """i + o for a plain shape; the load-weighted mean ``sum(w * (i + o))`` otherwise.

    This is the KV-cache footprint of one admitted request at its longest, which is what
    sets how many of them the engine can run before the surplus has to wait. A sampled
    length contributes its mean, because what fills the cache over a burst is the average
    footprint of the requests in it, not any one draw.
    """
    return sum(
        w * (_length_mean(i) + _length_mean(o)) for w, i, o in shape_components(shape_name)
    )


def ramp_duration_s(capacity_rps: float, cap: AdmissionCap) -> float:
    """T_r: long enough to cross rho = 1 slowly, short enough that the backlog it builds
    still fits the cap's admission headroom."""
    return cap.ramp_seconds(capacity_rps, RAMP_BACKLOG_COEFFICIENT)


# ------------------------------------------------------------------ primitives


def ramp_segments(capacity_rps: float, ramp_s: float) -> list[dict]:
    """rho ramps linearly in ~RAMP_SEGMENT_S steps, holds at the peak, then drains.

    ``ramp_s`` is passed in rather than derived from ``capacity_rps`` because the mixture
    shape builds one stream per component at a fraction of C_s: every component has to
    share the timeline that the *total* C_s produced, or the four streams would ramp at
    different speeds.

    It is split into the nearest whole number of RAMP_SEGMENT_S-long segments, so a
    segment is ~5 s and the ramp ends exactly at T_r. Each segment uses the rho at its
    *midpoint*, so the piecewise-constant schedule integrates to the same request count as
    the continuous ramp it approximates.
    """
    segments: list[dict] = []
    n = max(1, int(round(ramp_s / RAMP_SEGMENT_S)))
    for k in range(n):
        frac = (k + 0.5) / n
        rho = RAMP_RHO_START + (RAMP_RHO_END - RAMP_RHO_START) * frac
        segments.append(
            _segment(k * ramp_s / n, (k + 1) * ramp_s / n, rho * capacity_rps)
        )
    hold_end = ramp_s + ramp_s * RAMP_HOLD_FRACTION
    segments.append(_segment(ramp_s, hold_end, RAMP_RHO_END * capacity_rps))
    segments.append(_segment(hold_end, hold_end + RAMP_DRAIN_S, RAMP_DRAIN_RHO * capacity_rps))
    return segments


def step_segments(capacity_rps: float) -> list[dict]:
    segments: list[dict] = []
    t = 0.0
    for rho, duration in STEPS:
        segments.append(_segment(t, t + duration, rho * capacity_rps))
        t += duration
    return segments


def burst_segments(capacity_rps: float, requests_per_burst: float) -> list[dict]:
    """Base rate for the whole cell, with BURST_COUNT spikes superposed on top.

    A spike carries ``requests_per_burst`` requests inside BURST_WIDTH_S seconds, so its
    segment rate is ``requests_per_burst / BURST_WIDTH_S`` on top of the base. Sized by
    ``AdmissionCap.burst_sizing``, that is enough concurrent work to push the engine past
    its running limit, so the surplus queues - the observable the whole primitive exists
    to produce. Like the ramp, the count is passed in so the mixture's component streams
    split one burst between them instead of each firing a full one.
    """
    segments = [_segment(0.0, BURST_DURATION_S, BURST_BASE_RHO * capacity_rps)]
    for k in range(BURST_COUNT):
        start = BURST_FIRST_S + k * BURST_PERIOD_S
        end = start + BURST_WIDTH_S
        if end > BURST_DURATION_S:
            raise ValueError("burst falls outside the cell duration")
        segments.append(_segment(start, end, requests_per_burst / BURST_WIDTH_S))
    return segments


def hold_segments(capacity_rps: float, rho: float, duration_s: float) -> list[dict]:
    """One constant-rate segment: ``rho * C_s`` for ``duration_s``.

    Deliberately the simplest primitive in the module. A boundary probe has to answer
    "does this offered load violate the SLO in steady state", and any shaping - a ramp
    in, a drain out - would put windows in the capture that were taken at a different
    offered load than the one the probe is about.
    """
    if duration_s <= 0.0:
        raise ValueError("a hold cell needs a positive duration")
    return [_segment(0.0, duration_s, rho * capacity_rps)]


def _segment_sort_key(segment: dict) -> tuple:
    """Stable file ordering for segments whose lengths may be fixed or sampled."""
    in_dist = segment.get("input_tokens_dist") or {}
    out_dist = segment.get("max_tokens_dist") or {}
    in_nominal = segment.get("input_tokens")
    if in_nominal is None:
        in_nominal = TokenRange.from_mapping(in_dist).median if in_dist else 0
    out_nominal = segment.get("max_tokens")
    if out_nominal is None:
        out_nominal = TokenRange.from_mapping(out_dist).median if out_dist else 0
    return (segment["start_time"], in_nominal, out_nominal)


def _component_meta(weight: float, i: int | TokenRange, o: int | TokenRange) -> dict:
    """One stream of a shape, as the index records it."""
    meta: dict = {"weight": weight}
    if isinstance(i, TokenRange):
        meta["input_tokens_dist"] = i.as_dict()
        meta["input_tokens_mean"] = round(i.mean, 3)
    else:
        meta["input_tokens"] = int(i)
    if isinstance(o, TokenRange):
        meta["max_tokens_dist"] = o.as_dict()
        meta["max_tokens_mean"] = round(o.mean, 3)
    else:
        meta["max_tokens"] = int(o)
    return meta


def _segment(start_s: float, end_s: float, rps: float) -> dict:
    return {"start_time": _round(start_s), "end_time": _round(end_s), "rps": round(rps, 4)}


def _round(value: float) -> float:
    return int(value) if float(value).is_integer() else round(value, 3)


def capacity_rps_for_shape(capacity: CapacityModel, shape_name: str) -> float:
    """The fitted single-pod capacity prior C_s for one shape.

    For a sampled shape the model ``1/C = i/P + o/D`` is linear in both lengths, so
    ``E[1/C]`` is the cost at the *mean* lengths - the prior is exact under the fit
    rather than an approximation of it.
    """
    if shape_name == MIXTURE_NAME:
        return capacity.mixture_rps(MIXTURE)
    (_w, i, o), = shape_components(shape_name)
    return capacity.rps(_length_mean(i), _length_mean(o))


def build_schedule(
    model: str,
    shape_name: str,
    primitive: str,
    capacity: CapacityModel,
    kv_cache_tokens: Optional[int] = None,
    *,
    cap: Optional[AdmissionCap] = None,
) -> tuple[Optional[dict], dict]:
    """(trace.json body, index metadata) for one (model, shape, primitive).

    Thin wrapper over :func:`build_schedule_from_capacity_rps` that derives C_s from the
    fitted capacity prior. ``kv_cache_tokens`` is required for the bursts primitive only.
    """
    return build_schedule_from_capacity_rps(
        model,
        shape_name,
        primitive,
        capacity_rps_for_shape(capacity, shape_name),
        kv_cache_tokens=kv_cache_tokens,
        capacity_source="prior_fit",
        cap=cap,
    )


def build_schedule_from_capacity_rps(
    model: str,
    shape_name: str,
    primitive: str,
    capacity_rps: float,
    *,
    kv_cache_tokens: Optional[int] = None,
    capacity_source: str = "prior_fit",
    cap: Optional[AdmissionCap] = None,
    hold_rho: Optional[float] = None,
    hold_duration_s: Optional[float] = None,
    hold_stage: str = "",
    static_fraction: Optional[float] = None,
) -> tuple[Optional[dict], dict]:
    """(trace.json body, index metadata) for one (model, shape, primitive) at an
    explicitly supplied single-pod capacity, so a campaign can regenerate a schedule from
    a measured C_s instead of the fitted prior.

    ``capacity_source`` is recorded in the metadata next to the capacity actually used, so
    a schedule regenerated mid-campaign is never mistaken for one built from the prior.
    ``cap`` is the admission policy the cell is sized for; it defaults to the deployed
    one, which is also what the committed set is generated against.

    The body is ``None`` when the cell is skipped - today only the bursts primitive skips,
    when no gateway-admissible spike can push the engine past its running limit. The
    metadata still goes into the index, carrying the arithmetic that justified the skip.
    """
    cap = cap or get_cap(DEFAULT_CAP_NAME)
    components = shape_components(shape_name)
    c_s = float(capacity_rps)
    if c_s <= 0.0:
        raise ValueError(f"{model} {shape_name}: capacity_rps must be positive, got {c_s}")
    if shape_name == MIXTURE_NAME:
        # A mixture has no single (i, o), so its cell id records 0/0. r3_capacity's
        # sample_from_row then skips it (output_tokens 0 -> unusable), which is correct:
        # M is held-out validation, never a capacity or theta training point.
        cell_in, cell_out = 0, 0
    else:
        # A sampled shape is named by the geometric midpoint of its ranges, so its cell
        # id is stable, unique and parseable by GridCell.from_scenario_id. The id is a
        # NAME, not a claim about what was sent: the distribution itself is in the index
        # entry's components, and the realised lengths are in the raw log per request.
        (_w, nominal_i, nominal_o), = shape_components(shape_name)
        cell_in, cell_out = _length_nominal(nominal_i), _length_nominal(nominal_o)
    if primitive == HOLD_PRIMITIVE:
        if hold_rho is None or hold_duration_s is None:
            raise ValueError(
                f"{model} {shape_name}: the {HOLD_PRIMITIVE!r} primitive needs hold_rho "
                "and hold_duration_s - it exists to sit at one explicitly chosen rho"
            )
        load_code = hold_load_code(hold_rho)
    elif primitive == STATIC_PRIMITIVE:
        if hold_rho is None or hold_duration_s is None or static_fraction is None:
            raise ValueError(
                f"{model} {shape_name}: the {STATIC_PRIMITIVE!r} primitive needs hold_rho, "
                "hold_duration_s and static_fraction (its rho/rho*)"
            )
        load_code = static_load_code(static_fraction)
    else:
        load_code = LOAD_CODE[primitive]
    if primitive == "bursts" and kv_cache_tokens is None:
        raise ValueError(
            f"{model} {shape_name}: the bursts primitive needs kv_cache_tokens (the engine's "
            "GPU KV cache size in tokens) - burst size is derived from the engine's running "
            "limit, not from C_s."
        )

    meta: dict = {
        "model": model,
        "shape": shape_name,
        "primitive": primitive,
        "cell_id": f"i{cell_in}_o{cell_out}_c{load_code}",
        "held_out": is_held_out(shape_name),
        "sampled": shape_name in SAMPLED_SHAPES,
        "capacity_rps": round(c_s, 4),
        "capacity_source": capacity_source,
        "admission_cap": cap.name,
        "skipped": False,
        "components": [_component_meta(w, i, o) for w, i, o in components],
    }

    extra: dict = {}
    build: Callable[[float], list[dict]]
    if primitive == "ramp":
        ramp_s = ramp_duration_s(c_s, cap)
        hold_s = ramp_s * RAMP_HOLD_FRACTION
        extra = {
            "rho_start": RAMP_RHO_START,
            "rho_end": RAMP_RHO_END,
            "ramp_s": _round(ramp_s),
            "hold_s": _round(hold_s),
            "drain_rho": RAMP_DRAIN_RHO,
            "drain_s": RAMP_DRAIN_S,
            # where a cell truncated on the first proxy 503 jumps to
            "drain_start_s": _round(ramp_s + hold_s),
            "backlog_coefficient": RAMP_BACKLOG_COEFFICIENT,
            "ramp_excess_budget": cap.ramp_excess_budget,
        }

        def build(weight: float) -> list[dict]:
            return ramp_segments(c_s * weight, ramp_s)

    elif primitive == "steps":

        def build(weight: float) -> list[dict]:
            return step_segments(c_s * weight)

    elif primitive in (HOLD_PRIMITIVE, STATIC_PRIMITIVE):
        # A static-grid cell is the same constant-rate open-loop hold as a boundary
        # probe; only its rho comes from a plan instead of from the previous probe.
        rho = float(hold_rho)
        duration = float(hold_duration_s)
        extra = {
            "rho": rho,
            "hold_s": _round(duration),
            "stage": hold_stage,
            "offered_rps": round(rho * c_s, 4),
        }
        if primitive == STATIC_PRIMITIVE:
            extra["rho_over_rho_star"] = float(static_fraction)

        def build(weight: float) -> list[dict]:
            return hold_segments(c_s * weight, rho, duration)

    elif primitive == "bursts":
        tokens_per_req = tokens_per_request(shape_name)
        kv_cache_tokens = int(kv_cache_tokens)
        sizing = cap.burst_sizing(kv_cache_tokens, tokens_per_req)
        sizing_meta = {
            "kv_cache_tokens": kv_cache_tokens,
            "tokens_per_request": round(tokens_per_req, 4),
            **sizing.as_dict(),
        }
        if not sizing.reachable:
            meta["skipped"] = True
            meta.update(sizing_meta)
            return None, meta
        requests = sizing.requests
        extra = {
            **sizing_meta,
            "burst_segment_rps": round(requests / BURST_WIDTH_S, 4),
        }

        def build(weight: float) -> list[dict]:
            return burst_segments(c_s * weight, requests * weight)

    else:  # pragma: no cover - PRIMITIVES is the only caller
        raise ValueError(f"unknown primitive {primitive!r}")

    segments: list[dict] = []
    for weight, i, o in components:
        in_kind, in_value = _segment_shape(i)
        out_kind, out_value = _segment_shape(o)
        for seg in build(weight):
            entry = dict(seg)
            entry["input_tokens" if in_kind == "fixed" else "input_tokens_dist"] = in_value
            entry["max_tokens" if out_kind == "fixed" else "max_tokens_dist"] = out_value
            segments.append(entry)
    # Ordered by the nominal lengths so a sampled stream still has a stable position in
    # the file. The key is for reproducible output only, never for semantics.
    segments.sort(key=_segment_sort_key)

    duration = max(s["end_time"] for s in segments)
    planned = sum((s["end_time"] - s["start_time"]) * s["rps"] for s in segments)
    meta.update(extra)
    meta["duration_s"] = duration
    meta["planned_requests"] = int(round(planned))
    meta["peak_offered_rps"] = round(max(_offered_rps_at(segments)), 4)
    if primitive == "steps":
        t = 0.0
        boundaries = []
        for _rho, duration_s in STEPS:
            boundaries.append(_round(t))
            t += duration_s
        meta["discard_after_s"] = boundaries
    return {model: segments}, meta


def build_hold_schedule(
    model: str,
    shape_name: str,
    capacity_rps: float,
    rho: float,
    duration_s: float,
    *,
    stage: str = "",
    capacity_source: str = "measured_steps",
    cap: Optional[AdmissionCap] = None,
) -> tuple[dict, dict]:
    """(trace.json body, index metadata) for one constant-rho probe of the boundary search."""
    body, meta = build_schedule_from_capacity_rps(
        model,
        shape_name,
        HOLD_PRIMITIVE,
        capacity_rps,
        capacity_source=capacity_source,
        cap=cap,
        hold_rho=rho,
        hold_duration_s=duration_s,
        hold_stage=stage,
    )
    assert body is not None  # a hold cell is never declined; only bursts can be
    return body, meta


def build_rho_profile_schedule(
    model: str,
    shape_name: str,
    capacity_rps: float,
    profile: Sequence[tuple[float, float, float]],
    *,
    primitive: str,
    load_code: int,
    capacity_source: str,
    cap: Optional[AdmissionCap] = None,
    extra: Optional[dict] = None,
) -> tuple[dict, dict]:
    """(trace.json body, metadata) for an explicit ``[(start_s, end_s, rho), ...]`` profile.

    The preregistered second-round design (``scripts.calibration_design``) decides every
    cell's offered load itself - a constant-rho hold or a linear ramp in rho* units - and
    gives every cell its own load code so that no two cells share a cell id. This is the
    same component/segment construction as :func:`build_schedule_from_capacity_rps` (a
    mixture shape still runs one stream per component at its share of ``C_s``); only the
    rho profile and the load code are the caller's.
    """
    cap = cap or get_cap(DEFAULT_CAP_NAME)
    c_s = float(capacity_rps)
    if c_s <= 0.0:
        raise ValueError(f"{model} {shape_name}: capacity_rps must be positive, got {c_s}")
    if not profile:
        raise ValueError(f"{model} {shape_name}: an empty rho profile offers nothing")
    for start_s, end_s, rho in profile:
        if not (float(end_s) > float(start_s) >= 0.0) or float(rho) <= 0.0:
            raise ValueError(
                f"{model} {shape_name}: bad profile segment ({start_s}, {end_s}, {rho})"
            )
    if int(load_code) <= 0:
        raise ValueError(f"load code must be positive, got {load_code}")
    components = shape_components(shape_name)
    if shape_name == MIXTURE_NAME:
        cell_in, cell_out = 0, 0
    else:
        (_w, nominal_i, nominal_o), = components
        cell_in, cell_out = _length_nominal(nominal_i), _length_nominal(nominal_o)
    segments: list[dict] = []
    for weight, i, o in components:
        in_kind, in_value = _segment_shape(i)
        out_kind, out_value = _segment_shape(o)
        for start_s, end_s, rho in profile:
            entry = _segment(float(start_s), float(end_s), float(rho) * c_s * weight)
            entry["input_tokens" if in_kind == "fixed" else "input_tokens_dist"] = in_value
            entry["max_tokens" if out_kind == "fixed" else "max_tokens_dist"] = out_value
            segments.append(entry)
    segments.sort(key=_segment_sort_key)
    duration = max(s["end_time"] for s in segments)
    planned = sum((s["end_time"] - s["start_time"]) * s["rps"] for s in segments)
    meta: dict = {
        "model": model,
        "shape": shape_name,
        "primitive": primitive,
        "cell_id": f"i{cell_in}_o{cell_out}_c{int(load_code)}",
        "held_out": is_held_out(shape_name),
        "sampled": shape_name in SAMPLED_SHAPES,
        "capacity_rps": round(c_s, 4),
        "capacity_source": capacity_source,
        "admission_cap": cap.name,
        "skipped": False,
        "components": [_component_meta(w, i, o) for w, i, o in components],
        "rho_profile": [[_round(a), _round(b), round(float(r), 6)] for a, b, r in profile],
        **(extra or {}),
        "duration_s": duration,
        "planned_requests": int(round(planned)),
        "peak_offered_rps": round(max(_offered_rps_at(segments)), 4),
    }
    if len(profile) == 1:
        meta["offered_rps"] = round(float(profile[0][2]) * c_s, 4)
    return {model: segments}, meta


def _offered_rps_at(segments: Sequence[dict]) -> list[float]:
    """Total offered rps at each segment boundary (superposed overlapping segments)."""
    edges = sorted({s["start_time"] for s in segments} | {s["end_time"] for s in segments})
    totals = []
    for k in range(len(edges) - 1):
        mid = (edges[k] + edges[k + 1]) / 2.0
        totals.append(
            sum(s["rps"] for s in segments if s["start_time"] <= mid < s["end_time"])
        )
    return totals or [0.0]


def generate(
    capacity_dir: Path,
    output_dir: Path,
    models: Sequence[str],
    *,
    capacity_overrides: dict | None = None,
    kv_cache_overrides: dict | None = None,
    cap: Optional[AdmissionCap] = None,
) -> dict:
    cap = cap or get_cap(DEFAULT_CAP_NAME)
    index: dict = {
        "admission_cap": cap.as_dict(),
        "schedules": [],
        "capacity_models": {},
        "shapes": {},
        "primitives": {},
    }
    index["shapes"] = {
        **{
            name: {"input_tokens": i, "max_tokens": o, "held_out": False, "sampled": False}
            for name, (i, o) in SHAPES.items()
        },
        **{
            name: {
                "held_out": False,
                "sampled": True,
                "input_tokens_dist": i.as_dict(),
                "max_tokens_dist": o.as_dict(),
                "input_tokens_mean": round(i.mean, 3),
                "max_tokens_mean": round(o.mean, 3),
                "input_tokens_nominal": i.median,
                "max_tokens_nominal": o.median,
            }
            for name, (i, o) in SAMPLED_SHAPES.items()
        },
        MIXTURE_NAME: {
            "held_out": True,
            "sampled": False,
            "components": [
                {"weight": w, "input_tokens": i, "max_tokens": o} for w, i, o in MIXTURE
            ],
        },
    }
    index["training_shapes"] = list(TRAINING_SHAPES)
    index["held_out_shapes"] = [name for name in ALL_SHAPES if is_held_out(name)]
    index["families"] = {name: list(members) for name, members in FAMILIES.items()}
    # Only shape-independent constants belong here: T_r and B depend on C_s, on the
    # shape's token count and on the admission cap, and are recorded per cell instead.
    index["primitives"] = {
        "ramp": {
            "rho_start": RAMP_RHO_START, "rho_end": RAMP_RHO_END,
            "segment_s": RAMP_SEGMENT_S,
            "hold_fraction": RAMP_HOLD_FRACTION,
            "backlog_coefficient": RAMP_BACKLOG_COEFFICIENT,
            "ramp_s_formula": RAMP_DURATION_FORMULA,
            "drain_rho": RAMP_DRAIN_RHO, "drain_s": RAMP_DRAIN_S,
        },
        "steps": {"levels": [{"rho": r, "duration_s": d} for r, d in STEPS]},
        "bursts": {
            "base_rho": BURST_BASE_RHO, "duration_s": BURST_DURATION_S,
            "period_s": BURST_PERIOD_S, "width_s": BURST_WIDTH_S,
            "burst_requests_formula": BURST_REQUEST_FORMULA,
            "count": BURST_COUNT, "first_s": BURST_FIRST_S,
        },
    }

    for model in models:
        override = (capacity_overrides or {}).get(model)
        capacity_path = capacity_dir / f"capacity_{model}.json"
        if override is not None:
            _name, points = override
        else:
            _name, points = load_capacity_points(capacity_path)
        kv_override = (kv_cache_overrides or {}).get(model)
        if kv_override is not None:
            kv_cache_tokens = int(kv_override)
        elif override is not None:
            raise ValueError(
                f"{model}: a capacity override was supplied without a kv_cache_overrides "
                "entry; burst sizing needs the engine's GPU KV cache size."
            )
        else:
            kv_cache_tokens = load_kv_cache_tokens(capacity_path)
        capacity = fit_capacity_model(model, points)
        index["capacity_models"][model] = {
            "prefill_tokens_per_s": round(capacity.prefill_tps, 2),
            "decode_tokens_per_s": round(capacity.decode_tps, 2),
            "fitted_from_points": capacity.n_points,
            "rms_relative_error": round(capacity.rms_rel_error, 4),
            "kv_cache_tokens": kv_cache_tokens,
        }
        model_dir = output_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        for shape_name in ALL_SHAPES:
            for primitive in PRIMITIVES:
                body, meta = build_schedule(
                    model, shape_name, primitive, capacity, kv_cache_tokens, cap=cap
                )
                if body is not None:
                    path = model_dir / f"{shape_name}_{primitive}.json"
                    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
                    meta["path"] = str(path.relative_to(output_dir))
                index["schedules"].append(meta)
    return index


def main() -> int:
    here = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capacity-dir", type=Path,
                    default=here / "replayer/traces_v2/calibration/capacity")
    ap.add_argument("--output-dir", type=Path, default=here / "replayer/traces_v2/calibration")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--cap", choices=sorted(CAPS), default=DEFAULT_CAP_NAME,
                    help="admission policy the schedules are sized for "
                         f"(default: {DEFAULT_CAP_NAME}, which is what is deployed)")
    args = ap.parse_args()

    cap = get_cap(args.cap)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = generate(
        args.capacity_dir, args.output_dir, [m for m in args.models.split(",") if m], cap=cap
    )
    (args.output_dir / "INDEX.json").write_text(
        json.dumps(index, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    written = [m for m in index["schedules"] if not m["skipped"]]
    skipped = [m for m in index["schedules"] if m["skipped"]]
    print(
        f"wrote {len(written)} schedules to {args.output_dir} for the {cap.name} policy "
        f"({cap.admission_controller}-bound, shed ceiling {cap.shed_ceiling}, "
        f"{len(skipped)} cells skipped)"
    )
    for model, cm in index["capacity_models"].items():
        print(
            f"  {model}: prefill {cm['prefill_tokens_per_s']} tok/s, "
            f"decode {cm['decode_tokens_per_s']} tok/s, "
            f"rms rel err {cm['rms_relative_error']:.1%}, "
            f"kv cache {cm['kv_cache_tokens']} tokens"
        )
    for meta in skipped:
        print(
            f"  skipped {meta['model']} {meta['shape']} {meta['primitive']}: "
            f"needs {meta['burst_requests_needed']} requests > cap "
            f"{meta['burst_request_cap']} ({meta['binding_limit']}-bound)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

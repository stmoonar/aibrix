#!/usr/bin/env python3
"""Adaptive search for the offered load at which a (model, shape) starts violating SLO.

Why this replaces the fixed rho grid
------------------------------------
theta is a threshold on the control signal *at the moment the model stops meeting its
SLO*. A fixed grid spends its budget uniformly over a range whose interesting part is one
point wide: most cells land far from the boundary and contribute windows that say only
"comfortably healthy" or "hopelessly overloaded", neither of which constrains a
threshold. Worse, where the boundary actually sits is unknown before the run - it is
defined against a capacity prior with a 10-29 % error - so a grid centred on the prior is
centred on a guess.

The search here spends the same wall clock in three stages of increasing resolution:

1. **coarse** - 90 s at each of rho in {0.6, 0.9, 1.1}. Three cheap probes, only enough
   to find which interval the violation flip happens in. 90 s, not the 60 s this started
   at, because a probe is judged on the controller's own 30 s sliding windows and needs
   three *disjoint* 30 s spans of evidence (see "How a probe is judged").
1b. **extend** - when the coarse stage did not bracket the flip, probe outward before
   bisecting: DOWN at {0.45, 0.3} when even the lowest coarse probe violated, UP at
   {1.5, 2.0, 2.6} when even the top one was healthy, stopping at the first probe that
   closes the bracket. A search that still has no bracket after this does not bisect or
   dwell on a guess; it reports rho* as a bound (:func:`rho_star_status`).
2. **bisect** - 2 rounds of 120 s, each halving the bracket the coarse stage produced.
   Longer, because a probe near the boundary has to distinguish "violating" from "noisy",
   and that needs windows.
3. **dwell** - 300 s at ``0.95 * rho*``. This is where the evidence actually comes from:
   sitting just *under* the located boundary produces the densest possible supply of
   windows on both sides of it, which is exactly the regime a threshold is fitted on.

About 13.5 minutes of offered load per shape (plus up to 3 extension probes), plus
cooldowns.

What rho* is
------------
:func:`rho_star_status` labels every result instead of silently reporting a grid point:
``measured`` (a conclusive healthy probe within :data:`MAX_BRACKET_REL_WIDTH` under the
lowest conclusive violating one), ``grid_artifact`` (a bracket wider than that, or one
whose healthy side exists only through inconclusive probes - the number is where the
bisection happened to stop), ``upper_bound`` (violating everywhere probed) and
``lower_bound`` (healthy everywhere probed). Only ``measured`` may place static-grid
cells (``scripts.static_grid``).

How a probe is judged
---------------------
With exactly the windows and the label the fit uses: the probe's rows are built by
``rewindow_from_raw.label_cell`` (client per-request latency, 30 s windows sliding by 5 s,
the controller's view; 30 s ending on the 10 s grid under D8) and each row is judged by
the **primary** label - ``tre_common.slo_labels``, D6' slowdown TTFT - which the fit
trains on, so the search and the fit use one ruler (:func:`probe_verdict` takes the
campaign's :class:`~tre_common.slo_labels.LabelDefinition`).
The verdict is three-valued and explicit - :data:`VERDICT_VIOLATED`,
:data:`VERDICT_HEALTHY`, :data:`VERDICT_INCONCLUSIVE` - never "False, and the caller
should look at the counts". The earlier implicit version is how every 60 s coarse probe
of the 2026-09-21 campaign (two 30 s windows, below the three-window floor) was recorded
as healthy, including probes whose every window violated.

*Enough evidence* is counted in **disjoint** window spans, not rows: sliding windows
overlap six-fold, so a 60 s probe yields 7 rows but only 2 independent 30 s spans, and a
row count would let it pass a three-window floor on two windows' worth of information.
Unlabeled windows (too few completions for a p95) count for neither side.

An *inconclusive* probe says the probe was too short to measure anything, which re-running
it unchanged cannot fix. It is re-driven once at the same rho with its duration multiplied
by :data:`INCONCLUSIVE_DURATION_FACTOR`, on the same retry budget as a void
(:func:`next_void_attempt`); inconclusive again and the search stops and says so.

Why dwell sits below rho*, not on it
------------------------------------
``rho*`` is the lowest offered load that was *observed* to violate. Dwelling there would
put almost every window on the violating side and starve the fit of the healthy windows
it needs to place the threshold between. Dwelling 5 % under it keeps the operating point
inside the transition, where windows land on both sides according to the system's own
jitter - which is the sampling distribution a decision threshold is supposed to be
calibrated against.

What a probe does NOT do
------------------------
A probe never averages a voided cell into its verdict. A cell voided by an admission
overflow, by a dispatch-delay breach, by the model error budget, by the transient proxy
error budget or by the overflow sentinel carries no information about the boundary: it is
re-run once, and if it voids again the search stops and says so rather than bisecting on
noise. Silently treating "we could not measure it" as "it did not violate" would walk the
bracket upwards on every failure. :func:`next_void_attempt` is that rule, and the
campaign's scheduled cells obey the same one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from tre_common import slo_labels

#: Coarse probes. 1.1 is above the prior's capacity and 0.6 well under it, so the flip -
#: if the prior is anywhere near right - is bracketed by the first three cells.
COARSE_RHOS: tuple[float, ...] = (0.6, 0.9, 1.1)
#: MIN_PROBE_WINDOWS disjoint spans of the 30 s control window: the shortest probe that
#: can be conclusive at all (see :func:`min_probe_seconds`); ``calibration_campaign
#: --boundary-coarse-s`` overrides it (60 reproduces the 2026-09-21 campaign).
COARSE_SECONDS = 90.0
LEGACY_COARSE_SECONDS = 60.0

#: Outward probes when the coarse stage did not bracket the flip (rho, relative to the
#: same capacity prior): down when the lowest coarse probe already violated, up past the
#: top coarse probe (and past the old 1.1 * 1.3^2 = 1.859 reach) when it was healthy.
EXTEND_DOWN_RHOS: tuple[float, ...] = (0.45, 0.3)
EXTEND_UP_RHOS: tuple[float, ...] = (1.5, 2.0, 2.6)

#: A bracket (healthy, violating) narrower than this fraction of its upper end is a
#: measured boundary; a wider one is a grid point. Two bisection rounds bring the coarse
#: brackets to 0.05-0.08.
MAX_BRACKET_REL_WIDTH = 0.10

RHO_STAR_MEASURED = "measured"
RHO_STAR_LOWER_BOUND = "lower_bound"
RHO_STAR_UPPER_BOUND = "upper_bound"
RHO_STAR_GRID_ARTIFACT = "grid_artifact"
RHO_STAR_STATUSES = (RHO_STAR_MEASURED, RHO_STAR_LOWER_BOUND, RHO_STAR_UPPER_BOUND, RHO_STAR_GRID_ARTIFACT)

#: Bisection. Two rounds take a bracket of width 0.5 down to 0.125, which is finer than
#: the capacity prior's own error, so a third round would be refining a number whose
#: definition is already the dominant uncertainty.
BISECT_ROUNDS = 2
BISECT_SECONDS = 120.0

#: When the coarse stage never flipped, the bracket has one open end and each bisection
#: round steps outward by this factor instead of halving inward.
BRACKET_EXTENSION = 1.3

#: Where the evidence is collected, as a fraction of the located boundary.
DWELL_FRACTION = 0.95
DWELL_SECONDS = 300.0

#: A probe counts as violating when at least this fraction of its windows missed an SLO.
#: Not "any window": a single window over the line at a healthy operating point is the
#: tail of the latency distribution, and bisecting on it walks the bracket down to a rho
#: the model serves perfectly well.
VIOLATION_WINDOW_FRACTION = 0.5

#: Disjoint (non-overlapping) labelled windows a probe must have produced for its
#: verdict to count at all. Below this the fraction above is being computed on a handful
#: of samples, and the probe is :data:`VERDICT_INCONCLUSIVE`.
MIN_PROBE_WINDOWS = 3

#: An inconclusive probe is re-driven at the same rho for this multiple of its duration.
INCONCLUSIVE_DURATION_FACTOR = 2.0

#: A voided cell is re-driven this many times before whoever asked for it gives up.
#:
#: This is the rule for every voided cell in the campaign, not only for a probe: the
#: scheduled primitives re-drive through :func:`next_void_attempt` as well. One rule in
#: one place, because "re-run once, stop on the second void" is a statement about how
#: much a void costs, and two copies of it drift.
MAX_VOID_RETRIES = 1


def next_void_attempt(
    attempt: int, *, max_retries: int = MAX_VOID_RETRIES
) -> Optional[int]:
    """The attempt number to re-drive a voided cell as, or None when the rule says stop.

    A voided cell measured nothing, so it is evidence in no direction: re-running it once
    is the cheapest way to tell an infrastructure hiccup from a real inability to offer
    the load. A second void is the second one - carrying on past it means building a fit
    (or a bisection) on cells that never measured anything, which is the failure this
    whole guard exists to prevent.
    """
    if int(attempt) > int(max_retries):
        return None
    return int(attempt) + 1

def min_probe_seconds(window_ms: int, *, min_windows: int = MIN_PROBE_WINDOWS) -> float:
    """The shortest probe that can hold ``min_windows`` disjoint windows of ``window_ms``."""
    return float(min_windows) * float(window_ms) / 1000.0


STAGE_COARSE = "coarse"
STAGE_EXTEND = "extend"
STAGE_BISECT = "bisect"
STAGE_DWELL = "dwell"
STAGES = (STAGE_COARSE, STAGE_EXTEND, STAGE_BISECT, STAGE_DWELL)


def stage_seconds(coarse_seconds: float = COARSE_SECONDS) -> dict[str, float]:
    """Offered-load seconds each stage spends on one shape when the coarse stage brackets
    the flip (the extension stage is then empty; see :func:`max_extension_seconds`)."""
    return {
        STAGE_COARSE: len(COARSE_RHOS) * coarse_seconds,
        STAGE_BISECT: BISECT_ROUNDS * BISECT_SECONDS,
        STAGE_DWELL: DWELL_SECONDS,
    }


def shape_seconds(coarse_seconds: float = COARSE_SECONDS) -> float:
    """Offered-load seconds for one shape's whole search (excludes cooldowns)."""
    return sum(stage_seconds(coarse_seconds).values())


def probe_count() -> int:
    return len(COARSE_RHOS) + BISECT_ROUNDS + 1


def max_extension_probes() -> int:
    return max(len(EXTEND_DOWN_RHOS), len(EXTEND_UP_RHOS))


def max_extension_seconds(coarse_seconds: float = COARSE_SECONDS) -> float:
    """Worst-case extra offered load of the extension stage (it runs at coarse length)."""
    return max_extension_probes() * coarse_seconds


# --------------------------------------------------------------------------- verdicts


@dataclass(frozen=True)
class Probe:
    """One cell the search wants driven."""

    rho: float
    duration_s: float
    stage: str
    attempt: int = 1

    def as_dict(self) -> dict:
        return {
            "rho": round(self.rho, 6),
            "duration_s": self.duration_s,
            "stage": self.stage,
            "attempt": self.attempt,
        }


#: A probe's verdict. Exactly one of these; nothing is inferred from a count.
VERDICT_VIOLATED = "violated"
VERDICT_HEALTHY = "healthy"
VERDICT_INCONCLUSIVE = "inconclusive"
#: The guard voided the cell: it measured nothing, and no verdict was computed.
VERDICT_VOID = "void"
VERDICTS = (VERDICT_VIOLATED, VERDICT_HEALTHY, VERDICT_INCONCLUSIVE, VERDICT_VOID)


@dataclass(frozen=True)
class ProbeVerdict:
    """What one probe's window rows say, with the counts the verdict was decided on."""

    verdict: str
    windows: int
    labeled_windows: int
    independent_windows: int
    violating_windows: int

    def __post_init__(self) -> None:
        if self.verdict not in (VERDICT_VIOLATED, VERDICT_HEALTHY, VERDICT_INCONCLUSIVE):
            raise ValueError(f"not a probe verdict: {self.verdict!r}")


@dataclass(frozen=True)
class ProbeResult:
    """What driving a :class:`Probe` said.

    ``verdict`` is one of :data:`VERDICTS`. A voided cell is :data:`VERDICT_VOID` and has
    no other verdict: whatever its rows would have said is never computed.
    """

    probe: Probe
    verdict: str
    void_reasons: tuple[str, ...] = ()
    windows: int = 0
    labeled_windows: int = 0
    independent_windows: int = 0
    violating_windows: int = 0
    goodput: Optional[float] = None
    cell_id: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"not a probe verdict: {self.verdict!r}")
        if (self.verdict == VERDICT_VOID) != bool(self.void_reasons):
            raise ValueError("a void probe needs void reasons, and only a void probe has them")

    @property
    def valid(self) -> bool:
        return self.verdict != VERDICT_VOID

    @property
    def conclusive(self) -> bool:
        """A verdict that moves the bracket: violated or healthy."""
        return self.verdict in (VERDICT_VIOLATED, VERDICT_HEALTHY)

    def as_dict(self) -> dict:
        body = self.probe.as_dict()
        body.update({
            "verdict": self.verdict,
            "void_reasons": list(self.void_reasons),
            "windows": self.windows,
            "labeled_windows": self.labeled_windows,
            "independent_windows": self.independent_windows,
            "violating_windows": self.violating_windows,
            "goodput": self.goodput,
            "cell_id": self.cell_id,
        })
        return body


def independent_windows(rows: Sequence[Mapping]) -> list[Mapping]:
    """The largest set of mutually non-overlapping windows, taken earliest-first.

    Sliding windows (30 s wide, 5 s apart - the controller's own view) overlap six-fold:
    a 60 s cell yields 7 rows but only 2 disjoint 30 s spans of evidence. Anything that
    asks "is there enough evidence here" counts these, not rows.
    """
    chosen: list[Mapping] = []
    last_end: Optional[int] = None
    for row in sorted(rows, key=lambda r: (int(r["window_start_ms"]), int(r["window_end_ms"]))):
        if last_end is None or int(row["window_start_ms"]) >= last_end:
            chosen.append(row)
            last_end = int(row["window_end_ms"])
    return chosen


def probe_verdict(
    rows: Sequence[Mapping],
    *,
    label: Optional["slo_labels.LabelSpec"] = None,
    ttft_slo_ms: Optional[float] = None,
    tpot_slo_ms: Optional[float] = None,
    window_fraction: float = VIOLATION_WINDOW_FRACTION,
    min_windows: int = MIN_PROBE_WINDOWS,
) -> ProbeVerdict:
    """Three-valued verdict of one probe's window rows.

    Each row is labelled by ``label`` - the campaign's primary
    :class:`~tre_common.slo_labels.LabelDefinition` (D6'), the label the fit is trained
    on - through :func:`tre_common.slo_labels.window_slo_label`. The fixed
    ``ttft_slo_ms`` / ``tpot_slo_ms`` pair is the 09-23 interface (a fixed-threshold
    label, no min-n guard), kept for replaying artifacts made with it. ``unlabeled`` rows
    count for neither side. Then:

    * fewer than ``min_windows`` **disjoint** labelled windows -> inconclusive;
    * at least ``window_fraction`` of the labelled windows violated -> violated;
    * otherwise healthy.
    """
    if label is None:
        if ttft_slo_ms is None or tpot_slo_ms is None:
            raise ValueError("probe_verdict needs label (or ttft_slo_ms and tpot_slo_ms)")
        label = slo_labels.slo_targets(ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms)
    labeled: list[Mapping] = []
    violating = 0
    for row in rows:
        verdict_of_row = slo_labels.window_slo_label(row, label)
        if verdict_of_row == slo_labels.LABEL_UNLABELED:
            continue
        labeled.append(row)
        if verdict_of_row == slo_labels.LABEL_VIOLATED:
            violating += 1
    independent = len(independent_windows(labeled))
    if independent < int(min_windows):
        verdict = VERDICT_INCONCLUSIVE
    elif violating >= window_fraction * len(labeled):
        verdict = VERDICT_VIOLATED
    else:
        verdict = VERDICT_HEALTHY
    return ProbeVerdict(
        verdict=verdict,
        windows=len(rows),
        labeled_windows=len(labeled),
        independent_windows=independent,
        violating_windows=violating,
    )


# ------------------------------------------------------------------------ rho* status


def _probe_verdict_of(probe: Mapping, min_windows: int) -> str:
    """The verdict of a saved probe record, in either artifact format: the three-state
    ``verdict`` (09-23 on), or the 09-22 ``valid`` / ``conclusive`` / ``violated`` flags -
    a saved probe without ``conclusive`` (written before it existed) is conclusive iff it
    has ``min_windows`` windows, which is exactly what the 60 s coarse probes of the
    2026-09-21 campaign lacked."""
    if probe.get("verdict"):
        return str(probe["verdict"])
    if not probe.get("valid", True):
        return VERDICT_VOID
    if "conclusive" in probe:
        conclusive = bool(probe["conclusive"])
    else:
        conclusive = int(probe.get("windows") or 0) >= int(min_windows)
    if not conclusive:
        return VERDICT_INCONCLUSIVE
    return VERDICT_VIOLATED if probe.get("violated") else VERDICT_HEALTHY


def rho_star_status(
    probes: Sequence[dict],
    *,
    min_windows: int = MIN_PROBE_WINDOWS,
    max_rel_width: float = MAX_BRACKET_REL_WIDTH,
) -> dict:
    """What a search's rho* is, from its probe records (``ProbeResult.as_dict`` rows, as
    saved in ``<campaign>/<model>/boundary/<model>_<shape>.json``).

    Only violated / healthy probes are evidence (:func:`_probe_verdict_of` reads either
    artifact format).
    Returns ``{"status", "rho_star", "healthy_rho", "violating_rho", "bracket_rel_width"}``;
    ``status`` is None when no conclusive probe exists.
    """
    verdicts = [(p, _probe_verdict_of(p, min_windows)) for p in probes]
    valid = [(p, v) for p, v in verdicts if v != VERDICT_VOID]
    viol = [float(p["rho"]) for p, v in valid if v == VERDICT_VIOLATED]
    hi = min(viol) if viol else None
    healthy = [float(p["rho"]) for p, v in valid
               if v == VERDICT_HEALTHY and (hi is None or float(p["rho"]) < hi)]
    lo = max(healthy) if healthy else None
    width = None
    if hi is None:
        status = RHO_STAR_LOWER_BOUND if lo is not None else None
        rho_star = lo
    elif lo is None:
        inconclusive_below = any(
            v == VERDICT_INCONCLUSIVE and float(p["rho"]) < hi for p, v in valid
        )
        status = RHO_STAR_GRID_ARTIFACT if inconclusive_below else RHO_STAR_UPPER_BOUND
        rho_star = hi
    else:
        width = (hi - lo) / hi
        status = RHO_STAR_MEASURED if width <= max_rel_width else RHO_STAR_GRID_ARTIFACT
        rho_star = hi
    return {
        "status": status,
        "rho_star": rho_star,
        "healthy_rho": lo,
        "violating_rho": hi,
        "bracket_rel_width": width,
        "max_bracket_rel_width": max_rel_width,
        "min_probe_windows": int(min_windows),
    }


def saved_rho_star_status(search: dict) -> str:
    """Status of a saved search JSON: its own ``rho_star_status`` when it recorded one,
    else recomputed from its probes; a record with neither (hand-written fixtures) falls
    back to ``boundary_found`` -> measured / lower_bound."""
    if search.get("rho_star_status"):
        return str(search["rho_star_status"])
    if search.get("probes"):
        return rho_star_status(search["probes"])["status"] or RHO_STAR_LOWER_BOUND
    return RHO_STAR_MEASURED if search.get("boundary_found") else RHO_STAR_LOWER_BOUND


# ---------------------------------------------------------------------------- search


@dataclass
class BoundarySearch:
    """Three-stage state machine: ``next_probe()`` / ``record(result)`` until done.

    Deliberately pure and clock-free. The campaign drives the cells; this only decides
    which rho to ask for next and what the answers mean, so the whole decision logic is
    unit-testable without a cluster.
    """

    model: str = ""
    shape: str = ""
    coarse_rhos: tuple[float, ...] = COARSE_RHOS
    coarse_seconds: float = COARSE_SECONDS
    #: Empty tuples switch the extension stage off (the pre-2026-09-22 search, which
    #: bisects outward by BRACKET_EXTENSION instead).
    extend_down_rhos: tuple[float, ...] = EXTEND_DOWN_RHOS
    extend_up_rhos: tuple[float, ...] = EXTEND_UP_RHOS
    bisect_rounds: int = BISECT_ROUNDS
    bisect_seconds: float = BISECT_SECONDS
    dwell_fraction: float = DWELL_FRACTION
    dwell_seconds: float = DWELL_SECONDS
    max_retries: int = MAX_VOID_RETRIES

    results: list[ProbeResult] = field(default_factory=list)
    #: Highest rho observed healthy, and lowest observed violating. Either may be None.
    healthy_rho: Optional[float] = None
    violating_rho: Optional[float] = None
    stopped_reason: str = ""

    #: Why the search ended without a bracket after the extension stage (not a failure:
    #: rho* is then reported as a bound).
    unresolved_reason: str = ""

    _coarse_index: int = 0
    _bisect_done: int = 0
    _dwell_done: bool = False
    _pending: Optional[Probe] = None

    # ------------------------------------------------------------------ progression

    @property
    def extension_enabled(self) -> bool:
        return bool(self.extend_down_rhos or self.extend_up_rhos)

    @property
    def bracketed(self) -> bool:
        return self.healthy_rho is not None and self.violating_rho is not None

    def _extension_rho(self) -> Optional[float]:
        """Next outward probe, or None when the bracket is closed or the list is used up."""
        if self.bracketed:
            return None
        tried = {r.probe.rho for r in self.results if r.probe.stage == STAGE_EXTEND and r.conclusive}
        lo, hi = self.healthy_rho, self.violating_rho
        if hi is not None:
            below = [r for r in self.extend_down_rhos if r < hi and r not in tried]
            return max(below) if below else None
        if lo is not None:
            above = [r for r in self.extend_up_rhos if r > lo and r not in tried]
            return min(above) if above else None
        return None

    @property
    def stage(self) -> str:
        if self._coarse_index < len(self.coarse_rhos):
            return STAGE_COARSE
        if self.extension_enabled and self._extension_rho() is not None:
            return STAGE_EXTEND
        if self._bisect_done < self.bisect_rounds:
            return STAGE_BISECT
        return STAGE_DWELL

    @property
    def done(self) -> bool:
        return bool(self.stopped_reason) or bool(self.unresolved_reason) or self._dwell_done

    @property
    def boundary_found(self) -> bool:
        """True when some probe violated, i.e. the bracket has a real upper end."""
        return self.violating_rho is not None

    @property
    def rho_star(self) -> Optional[float]:
        """The located boundary: the lowest offered load observed to violate.

        With no violation anywhere, the highest rho that was *driven* is returned and
        :attr:`boundary_found` is False - the honest statement is "the boundary is above
        everything we offered", not a number pretending to be it.
        """
        if self.violating_rho is not None:
            return self.violating_rho
        driven = [
            r.probe.rho for r in self.results
            if r.verdict in (VERDICT_VIOLATED, VERDICT_HEALTHY)
        ]
        return max(driven) if driven else None

    @property
    def dwell_rho(self) -> Optional[float]:
        star = self.rho_star
        return None if star is None else round(star * self.dwell_fraction, 6)

    def next_probe(self) -> Optional[Probe]:
        """The next cell to drive, or None when the search is finished or stopped."""
        if self.done:
            return None
        if self._pending is not None:
            return self._pending
        stage = self.stage
        if stage == STAGE_COARSE:
            probe = Probe(self.coarse_rhos[self._coarse_index], self.coarse_seconds, stage)
        elif stage == STAGE_EXTEND:
            probe = Probe(self._extension_rho(), self.coarse_seconds, stage)
        elif stage in (STAGE_BISECT, STAGE_DWELL) and self.extension_enabled and not self.bracketed \
                and (self.healthy_rho is not None or self.violating_rho is not None):
            # Extended both ways as far as allowed and still no flip: rho* is a bound.
            # Bisecting or dwelling on a one-sided "bracket" would only manufacture a grid
            # point (plan 6.11: 13/21 rho* of the 2026-09-21 campaign were such points).
            if self.violating_rho is None:
                self.unresolved_reason = (
                    f"no violation up to rho={self.healthy_rho:g}: rho* is a lower bound"
                )
            else:
                self.unresolved_reason = (
                    f"violating down to rho={self.violating_rho:g}: rho* is an upper bound"
                )
            return None
        elif stage == STAGE_BISECT:
            rho = self._bisect_rho()
            if rho is None:
                self.stopped_reason = (
                    "no valid coarse probe, so there is no bracket to bisect"
                )
                return None
            probe = Probe(rho, self.bisect_seconds, stage)
        else:
            rho = self.dwell_rho
            if rho is None:
                self.stopped_reason = "no valid probe produced a boundary to dwell under"
                return None
            probe = Probe(rho, self.dwell_seconds, stage)
        self._pending = probe
        return probe

    def _redrive(self, result: "ProbeResult", why: str, duration_s: float) -> None:
        """Re-drive the same rho as the next attempt, or stop when the budget is spent."""
        attempt = int(result.probe.attempt)
        nxt = next_void_attempt(attempt, max_retries=self.max_retries)
        if nxt is None:
            self.stopped_reason = (
                f"probe at rho={result.probe.rho:g} was {why} on attempt {attempt}; "
                "the search will not bisect on a cell that measured nothing"
            )
            return
        self._pending = Probe(result.probe.rho, duration_s, result.probe.stage, attempt=nxt)

    def _bisect_rho(self) -> Optional[float]:
        lo, hi = self.healthy_rho, self.violating_rho
        if lo is not None and hi is not None:
            return round((lo + hi) / 2.0, 6)
        if hi is not None:
            # Everything violated, including the lowest coarse probe: step down.
            return round(hi / BRACKET_EXTENSION, 6)
        if lo is not None:
            # Nothing violated: the boundary is above what was offered, so step up.
            return round(lo * BRACKET_EXTENSION, 6)
        return None

    # ---------------------------------------------------------------------- results

    def record(self, result: ProbeResult) -> None:
        """Fold one driven probe into the state, and advance (or retry, or stop)."""
        self.results.append(result)
        self._pending = None
        if result.verdict == VERDICT_VOID:
            # Re-drive the same rho unchanged: a voided cell is not evidence in either
            # direction, and the cause (a shed, a late generator) is not its length.
            self._redrive(
                result,
                f"voided ({', '.join(result.void_reasons) or 'no reason recorded'})",
                result.probe.duration_s,
            )
            return
        if result.verdict == VERDICT_INCONCLUSIVE:
            # Neither side of the bracket moves. The cause IS its length - too few
            # disjoint labelled windows - so the re-drive is longer, not a repeat.
            self._redrive(
                result,
                f"inconclusive ({result.independent_windows} disjoint labelled window(s), "
                f"needs {MIN_PROBE_WINDOWS})",
                result.probe.duration_s * INCONCLUSIVE_DURATION_FACTOR,
            )
            return

        if result.verdict == VERDICT_VIOLATED:
            if self.violating_rho is None or result.probe.rho < self.violating_rho:
                self.violating_rho = result.probe.rho
        elif result.verdict == VERDICT_HEALTHY:
            if self.healthy_rho is None or result.probe.rho > self.healthy_rho:
                self.healthy_rho = result.probe.rho
        else:  # pragma: no cover - ProbeResult validates its verdict
            raise ValueError(f"unhandled probe verdict {result.verdict!r}")

        if result.probe.stage == STAGE_COARSE:
            self._coarse_index += 1
        elif result.probe.stage == STAGE_EXTEND:
            pass  # the extension stage ends by itself (bracket closed or list used up)
        elif result.probe.stage == STAGE_BISECT:
            self._bisect_done += 1
        else:
            self._dwell_done = True

    def status(self) -> dict:
        """:func:`rho_star_status` of this search's probes."""
        return rho_star_status([r.as_dict() for r in self.results])

    def as_dict(self) -> dict:
        st = self.status()
        return {
            "model": self.model,
            "shape": self.shape,
            "stage_seconds": stage_seconds(self.coarse_seconds),
            "coarse_rhos": list(self.coarse_rhos),
            "coarse_seconds": self.coarse_seconds,
            "extend_down_rhos": list(self.extend_down_rhos),
            "extend_up_rhos": list(self.extend_up_rhos),
            "min_probe_windows": MIN_PROBE_WINDOWS,
            "violation_window_fraction": VIOLATION_WINDOW_FRACTION,
            "inconclusive_duration_factor": INCONCLUSIVE_DURATION_FACTOR,
            "healthy_rho": self.healthy_rho,
            "violating_rho": self.violating_rho,
            "rho_star": self.rho_star,
            "rho_star_status": st["status"],
            "rho_star_bracket": [st["healthy_rho"], st["violating_rho"]],
            "bracket_rel_width": st["bracket_rel_width"],
            "dwell_rho": self.dwell_rho,
            "boundary_found": self.boundary_found,
            "stopped_reason": self.stopped_reason,
            "unresolved_reason": self.unresolved_reason,
            "probes": [r.as_dict() for r in self.results],
        }


# ------------------------------------------------------------------- stopping rule

#: The fit must publish a theta from at least this fraction of bootstrap resamples.
MIN_PUBLISH_RATE = 0.9
#: ... and its confidence interval must be narrower than this fraction of theta. Plan
#: 2026-09-21 6.11 D13: 15 % (half width ~ 1/sqrt(n): 10 % needs 2.25x the independent
#: windows; v1 had no CI gate at all, so 15 % is not a relaxation).
MAX_CI_HALF_WIDTH_FRACTION = 0.15
#: The tighter target reported in the appendix - informational, never gates the stop.
APPENDIX_CI_HALF_WIDTH_FRACTION = 0.10
#: Windows each family needs *near the boundary* before the campaign can stop. Below this
#: the family-wise diagnostic in ``calibration_campaign.family_theta_verdict`` is being
#: computed on too little to say anything.
MIN_FAMILY_BOUNDARY_WINDOWS = 30

#: A shape whose family is short of windows gets another dwell cell at this rho relative
#: to its located boundary - slightly further under it than the first, so the extra
#: windows are not a re-sample of exactly the same operating point.
HOLD_RETRY_FRACTIONS: tuple[float, ...] = (0.92, 0.98)


@dataclass(frozen=True)
class StopVerdict:
    """Whether the campaign has collected enough, and what to add if not."""

    satisfied: bool
    reasons: tuple[str, ...]
    hold_cells: tuple[dict, ...]
    #: CI half width / theta, the gate it was judged against, and the appendix target (D13).
    ci_half_width_fraction: Optional[float] = None
    max_ci_half_width_fraction: float = MAX_CI_HALF_WIDTH_FRACTION
    appendix_ci_half_width_fraction: float = APPENDIX_CI_HALF_WIDTH_FRACTION

    @property
    def appendix_ci_target_met(self) -> Optional[bool]:
        if self.ci_half_width_fraction is None:
            return None
        return self.ci_half_width_fraction < self.appendix_ci_half_width_fraction

    def as_dict(self) -> dict:
        return {
            "satisfied": self.satisfied,
            "reasons": list(self.reasons),
            "hold_cells": [dict(c) for c in self.hold_cells],
            "ci_half_width_fraction": self.ci_half_width_fraction,
            "max_ci_half_width_fraction": self.max_ci_half_width_fraction,
            "appendix_ci_half_width_fraction": self.appendix_ci_half_width_fraction,
            "appendix_ci_target_met": self.appendix_ci_target_met,
        }


def stop_rule(
    *,
    publish_rate: float,
    theta: float,
    ci_half_width: float,
    family_boundary_windows: dict[str, int],
    boundaries: Optional[dict[str, float]] = None,
    min_publish_rate: float = MIN_PUBLISH_RATE,
    max_ci_fraction: float = MAX_CI_HALF_WIDTH_FRACTION,
    min_family_windows: int = MIN_FAMILY_BOUNDARY_WINDOWS,
) -> StopVerdict:
    """Has the campaign collected enough, and if not, which hold cells close the gap?

    Three conditions, all necessary:

    * the bootstrap publishes a theta in at least ``min_publish_rate`` of resamples -
      i.e. the fit is not merely producing a number, it is producing one that survives
      resampling its own evidence;
    * the confidence interval is narrower than ``max_ci_fraction`` of theta - a theta
      known to within a factor of two is not a decision threshold;
    * the most prefill-heavy and the most decode-heavy families each have
      ``min_family_windows`` windows near the boundary - without that the family-wise
      diagnostic cannot tell "theta is the same in both regimes" from "we only looked at
      one regime".

    When something is short the remedy is **more hold cells at the boundary of the shapes
    already in the set**, never a new shape. A new shape changes what the fit is fitted
    on, so it cannot be added in response to the fit's own uncertainty without making the
    stopping rule a search over shape sets - which is how a calibration turns into
    fitting the acceptance criterion.
    """
    reasons: list[str] = []
    holds: list[dict] = []

    if publish_rate < min_publish_rate:
        reasons.append(
            f"bootstrap publish rate {publish_rate:.2f} < {min_publish_rate:.2f}"
        )
    limit = max_ci_fraction * abs(theta) if theta else float("inf")
    if not theta:
        reasons.append("theta is zero or missing, so its CI cannot be judged")
    elif ci_half_width >= limit:
        reasons.append(
            f"CI half width {ci_half_width:.4g} >= {max_ci_fraction:.0%} of theta "
            f"({limit:.4g})"
        )

    for family, count in sorted(family_boundary_windows.items()):
        if count >= min_family_windows:
            continue
        reasons.append(
            f"family {family!r} has {count} window(s) near the boundary, "
            f"needs >= {min_family_windows}"
        )
        for shape, rho_star in sorted((boundaries or {}).items()):
            for fraction in HOLD_RETRY_FRACTIONS:
                holds.append({
                    "family": family,
                    "shape": shape,
                    "rho": round(rho_star * fraction, 6),
                    "duration_s": DWELL_SECONDS,
                    "stage": STAGE_DWELL,
                    "why": f"top up {family} near its boundary",
                })

    # A short bootstrap with enough windows still needs more evidence, and the only
    # evidence the campaign is allowed to add is more time at the boundary.
    if reasons and not holds and boundaries:
        for shape, rho_star in sorted(boundaries.items()):
            holds.append({
                "family": "",
                "shape": shape,
                "rho": round(rho_star * HOLD_RETRY_FRACTIONS[0], 6),
                "duration_s": DWELL_SECONDS,
                "stage": STAGE_DWELL,
                "why": "tighten the theta CI with more boundary windows",
            })

    return StopVerdict(
        satisfied=not reasons,
        reasons=tuple(reasons),
        hold_cells=tuple(holds),
        ci_half_width_fraction=(abs(ci_half_width) / abs(theta)) if theta else None,
        max_ci_half_width_fraction=max_ci_fraction,
    )

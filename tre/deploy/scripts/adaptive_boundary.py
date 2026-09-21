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

1. **coarse** - 60 s at each of rho in {0.6, 0.9, 1.1}. Three cheap probes, only enough
   to find which interval the violation flip happens in.
2. **bisect** - 2 rounds of 120 s, each halving the bracket the coarse stage produced.
   Longer, because a probe near the boundary has to distinguish "violating" from "noisy",
   and that needs windows.
3. **dwell** - 300 s at ``0.95 * rho*``. This is where the evidence actually comes from:
   sitting just *under* the located boundary produces the densest possible supply of
   windows on both sides of it, which is exactly the regime a threshold is fitted on.

About 12 minutes of offered load per shape, plus cooldowns.

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
A probe never averages a voided cell into its verdict. A cell voided by a shed, by a
dispatch-delay breach, by the model error budget or by the overflow sentinel carries no
information about the boundary: it is re-run once, and if it voids again the search stops
and says so rather than bisecting on noise. Silently treating "we could not measure it"
as "it did not violate" would walk the bracket upwards on every failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

#: Coarse probes. 1.1 is above the prior's capacity and 0.6 well under it, so the flip -
#: if the prior is anywhere near right - is bracketed by the first three cells.
COARSE_RHOS: tuple[float, ...] = (0.6, 0.9, 1.1)
COARSE_SECONDS = 60.0

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

#: Windows a probe must have produced for its verdict to count at all. Below this the
#: fraction above is being computed on a handful of samples.
MIN_PROBE_WINDOWS = 3

#: A voided probe is re-driven this many times before the search gives up on it.
MAX_PROBE_RETRIES = 1

STAGE_COARSE = "coarse"
STAGE_BISECT = "bisect"
STAGE_DWELL = "dwell"
STAGES = (STAGE_COARSE, STAGE_BISECT, STAGE_DWELL)


def stage_seconds() -> dict[str, float]:
    """Offered-load seconds each stage spends on one shape."""
    return {
        STAGE_COARSE: len(COARSE_RHOS) * COARSE_SECONDS,
        STAGE_BISECT: BISECT_ROUNDS * BISECT_SECONDS,
        STAGE_DWELL: DWELL_SECONDS,
    }


def shape_seconds() -> float:
    """Offered-load seconds for one shape's whole search (excludes cooldowns)."""
    return sum(stage_seconds().values())


def probe_count() -> int:
    return len(COARSE_RHOS) + BISECT_ROUNDS + 1


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


@dataclass(frozen=True)
class ProbeResult:
    """What driving a :class:`Probe` said.

    ``valid`` is False for a cell the guard voided; ``violated`` is then meaningless and
    is ignored rather than trusted.
    """

    probe: Probe
    violated: bool
    valid: bool = True
    void_reasons: tuple[str, ...] = ()
    windows: int = 0
    violating_windows: int = 0
    goodput: Optional[float] = None
    cell_id: str = ""

    def as_dict(self) -> dict:
        body = self.probe.as_dict()
        body.update({
            "violated": self.violated,
            "valid": self.valid,
            "void_reasons": list(self.void_reasons),
            "windows": self.windows,
            "violating_windows": self.violating_windows,
            "goodput": self.goodput,
            "cell_id": self.cell_id,
        })
        return body


def probe_violated(
    rows: Sequence[dict],
    *,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    window_fraction: float = VIOLATION_WINDOW_FRACTION,
    min_windows: int = MIN_PROBE_WINDOWS,
) -> tuple[bool, int, int]:
    """(violated, violating windows, total windows) for one probe's window rows.

    A window counts as violating when either p95 is over its SLO, or when the row was
    marked ``slo_violated`` - which is how a window containing a model error gets counted
    even though the failed request contributed no latency sample.

    With fewer than ``min_windows`` rows the probe is reported as not violating *and* the
    caller is expected to look at the counts: the search treats a too-short probe as
    inconclusive rather than as healthy (see :meth:`BoundarySearch.record`).
    """
    total = len(rows)
    violating = 0
    for row in rows:
        if row.get("slo_violated"):
            violating += 1
            continue
        ttft = row.get("p95_ttft")
        tpot = row.get("p95_tpot")
        if ttft is not None and float(ttft) > ttft_slo_ms:
            violating += 1
        elif tpot is not None and float(tpot) > tpot_slo_ms:
            violating += 1
    if total < int(min_windows):
        return False, violating, total
    return violating >= window_fraction * total, violating, total


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
    bisect_rounds: int = BISECT_ROUNDS
    bisect_seconds: float = BISECT_SECONDS
    dwell_fraction: float = DWELL_FRACTION
    dwell_seconds: float = DWELL_SECONDS
    max_retries: int = MAX_PROBE_RETRIES

    results: list[ProbeResult] = field(default_factory=list)
    #: Highest rho observed healthy, and lowest observed violating. Either may be None.
    healthy_rho: Optional[float] = None
    violating_rho: Optional[float] = None
    stopped_reason: str = ""

    _coarse_index: int = 0
    _bisect_done: int = 0
    _dwell_done: bool = False
    _pending: Optional[Probe] = None
    _retries: int = 0

    # ------------------------------------------------------------------ progression

    @property
    def stage(self) -> str:
        if self._coarse_index < len(self.coarse_rhos):
            return STAGE_COARSE
        if self._bisect_done < self.bisect_rounds:
            return STAGE_BISECT
        return STAGE_DWELL

    @property
    def done(self) -> bool:
        return bool(self.stopped_reason) or self._dwell_done

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
        driven = [r.probe.rho for r in self.results if r.valid]
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
        if not result.valid:
            self._retries += 1
            if self._retries > self.max_retries:
                self.stopped_reason = (
                    f"probe at rho={result.probe.rho:g} was voided "
                    f"{self._retries} time(s) ({', '.join(result.void_reasons) or 'no reason recorded'}); "
                    "the search will not bisect on a cell that measured nothing"
                )
                return
            # Re-drive the same rho: a voided cell is not evidence in either direction.
            self._pending = Probe(
                result.probe.rho,
                result.probe.duration_s,
                result.probe.stage,
                attempt=result.probe.attempt + 1,
            )
            return

        self._retries = 0
        if result.violated:
            if self.violating_rho is None or result.probe.rho < self.violating_rho:
                self.violating_rho = result.probe.rho
        else:
            if self.healthy_rho is None or result.probe.rho > self.healthy_rho:
                self.healthy_rho = result.probe.rho

        if result.probe.stage == STAGE_COARSE:
            self._coarse_index += 1
        elif result.probe.stage == STAGE_BISECT:
            self._bisect_done += 1
        else:
            self._dwell_done = True

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "shape": self.shape,
            "stage_seconds": stage_seconds(),
            "coarse_rhos": list(self.coarse_rhos),
            "healthy_rho": self.healthy_rho,
            "violating_rho": self.violating_rho,
            "rho_star": self.rho_star,
            "dwell_rho": self.dwell_rho,
            "boundary_found": self.boundary_found,
            "stopped_reason": self.stopped_reason,
            "probes": [r.as_dict() for r in self.results],
        }


# ------------------------------------------------------------------- stopping rule

#: The fit must publish a theta from at least this fraction of bootstrap resamples.
MIN_PUBLISH_RATE = 0.9
#: ... and its confidence interval must be narrower than this fraction of theta.
MAX_CI_HALF_WIDTH_FRACTION = 0.10
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

    def as_dict(self) -> dict:
        return {
            "satisfied": self.satisfied,
            "reasons": list(self.reasons),
            "hold_cells": [dict(c) for c in self.hold_cells],
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
    )

"""Nominal vs achieved arrival rate, per model, over time.

Why this exists
---------------
An open-loop run claims to have offered exactly the load its schedule describes. Until
now the only evidence for that claim was
:attr:`tre_replayer.engine.dispatcher.DispatchReport.actual_rps_error_ratio`, which
compared two *span* rates - ``n / (last - first)`` scheduled against ``n / (last -
first)`` achieved. A span ratio is blind to every deviation that does not change the
span, and the one that matters most is exactly of that kind: if every request goes out
150 ms late, both spans are the same to the microsecond and the ratio is 0.0000 while
the whole run is shifted off its schedule. Measured on the real code path with a fake
network, requests went on the wire a uniform ~377 ms late and the span ratio still read
0.0000.

What this module measures instead
---------------------------------
The scheduled instants and the achieved instants are the *same events*, timestamped
twice. Bin both on the schedule's own time base and compare them window by window:

* a perfectly kept schedule puts every event in the window the schedule put it in, so
  every bin matches exactly and the error is 0 - however ragged the Poisson arrivals are;
* a uniform shift empties the head of the run and fills a window past its tail;
* a drift, a stall, or a generator that cannot keep up show up as a growing deficit in
  the windows where it happened.

So the headline number is the **largest relative deviation over any window**
(:func:`max_relative_rps_error`), and the per-window series itself
(:func:`write_rps_timeline_csv`) is the evidence a run offered the intensity its trace
asks for - the figure that says "we really did apply this load", plottable as-is.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

#: Bin width for the plottable series. One second: fine enough to show the shape of a
#: 2 s burst, coarse enough that a bin holds several arrivals at campaign rates.
DEFAULT_WINDOW_S = 1.0

#: Bin width for the headline error ratio. Deliberately coarser than the plot: the ratio
#: is a pass/fail-shaped number, and at a 1 s bin a handful of arrivals jittering across
#: a boundary is worth 10 % on its own. Five seconds holds enough events that boundary
#: jitter is a small fraction of the bin.
ERROR_WINDOW_S = 5.0


@dataclass(frozen=True)
class RpsBin:
    """One window of the comparison. ``scheduled``/``achieved`` are event counts."""

    start_s: float
    end_s: float
    scheduled: int
    achieved: int

    @property
    def width_s(self) -> float:
        return max(1e-9, self.end_s - self.start_s)

    @property
    def scheduled_rps(self) -> float:
        return self.scheduled / self.width_s

    @property
    def achieved_rps(self) -> float:
        return self.achieved / self.width_s

    @property
    def relative_error(self) -> float | None:
        """|achieved - scheduled| / scheduled, or None where there is no denominator.

        A window the schedule left empty has no rate to be wrong about; arrivals landing
        in one are still visible in the series (and in the neighbouring window's deficit
        that put them there), they just cannot form a ratio.
        """
        if self.scheduled == 0:
            return None
        return abs(self.achieved - self.scheduled) / self.scheduled

    def as_row(self, model: str) -> dict:
        return {
            "model": model,
            "window_start_s": round(self.start_s, 6),
            "window_end_s": round(self.end_s, 6),
            "scheduled_requests": self.scheduled,
            "achieved_requests": self.achieved,
            "scheduled_rps": round(self.scheduled_rps, 6),
            "achieved_rps": round(self.achieved_rps, 6),
            "relative_error": (
                "" if self.relative_error is None else round(self.relative_error, 6)
            ),
        }


CSV_COLUMNS = (
    "model",
    "window_start_s",
    "window_end_s",
    "scheduled_requests",
    "achieved_requests",
    "scheduled_rps",
    "achieved_rps",
    "relative_error",
)


def _bin_index(offset_s: float, window_s: float) -> int:
    return max(0, int(math.floor(offset_s / window_s)))


def build_rps_timeline(
    scheduled_offsets: Sequence[float],
    achieved_offsets: Sequence[float],
    *,
    window_s: float = DEFAULT_WINDOW_S,
) -> list[RpsBin]:
    """Bin both instant sets on the same grid, from 0 to whichever runs longer.

    Both sequences are offsets **in the schedule's own time base**, so bin *k* of the
    achieved series and bin *k* of the scheduled series describe the same interval of
    the run and are directly comparable.
    """
    if window_s <= 0.0:
        raise ValueError("window_s must be > 0")
    if not scheduled_offsets and not achieved_offsets:
        return []
    last = max(
        max(scheduled_offsets, default=0.0),
        max(achieved_offsets, default=0.0),
    )
    count = _bin_index(last, window_s) + 1
    scheduled = [0] * count
    achieved = [0] * count
    for offset in scheduled_offsets:
        scheduled[_bin_index(float(offset), window_s)] += 1
    for offset in achieved_offsets:
        achieved[_bin_index(float(offset), window_s)] += 1
    return [
        RpsBin(
            start_s=index * window_s,
            end_s=(index + 1) * window_s,
            scheduled=scheduled[index],
            achieved=achieved[index],
        )
        for index in range(count)
    ]


def relative_errors(bins: Iterable[RpsBin]) -> list[float]:
    """Every window's relative deviation, skipping the ones with no denominator."""
    return [b.relative_error for b in bins if b.relative_error is not None]


def max_relative_rps_error(bins: Iterable[RpsBin]) -> float:
    """Largest relative deviation over any window; 0.0 when the schedule was kept."""
    errors = relative_errors(bins)
    return max(errors) if errors else 0.0


def quantile_relative_rps_error(bins: Iterable[RpsBin], quantile: float = 0.95) -> float:
    """The ``quantile``-th relative deviation, for a run long enough that one bad window
    at the edge should not be the headline."""
    errors = sorted(relative_errors(bins))
    if not errors:
        return 0.0
    index = max(0, math.ceil(quantile * len(errors)) - 1)
    return errors[index]


def rps_timeline_rows(per_model: Mapping[str, Sequence[RpsBin]]) -> list[dict]:
    rows: list[dict] = []
    for model in sorted(per_model):
        rows.extend(bin_.as_row(model) for bin_ in per_model[model])
    return rows


def write_rps_timeline_csv(path: str | Path, per_model: Mapping[str, Sequence[RpsBin]]) -> int:
    """Write the nominal/achieved series for every model; returns rows written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = rps_timeline_rows(per_model)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)

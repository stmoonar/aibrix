"""Band dwell: a band change counts only after N consecutive *new* metrics windows.

One implementation for the controller (``tre_controller.signals.trs.SignalState``, which
gates CRITICAL / LOW / HIGH through it) and for offline evaluation
(:func:`dwell_confirmed_series`, e.g. the "CRITICAL with tau-EMA + dwell" recall/false
alarm of plan §6.9f item B), so the two agree window for window.

Windows are identified by ``window_end_ms``. The controller's decision loops re-read one
snapshot several times (rescue every 5 s, fairness, safescale) - a repeated or regressed
``window_end_ms`` never advances the count, so "2 consecutive windows" means two distinct
metrics windows (at the phase-aligned 10 s cadence the second one ends 10 s after the
first), never one window read twice.

Rules of :meth:`DwellCounter.update`, in order:

1. a repeated or regressed ``window_end_ms`` returns the current verdict unchanged;
2. a gap larger than ``max_gap_ms`` since the last counted window restarts the run (the
   same strictly-greater-than-one-window rule as ``tre_common.tss.TssEma``: windows that
   are not consecutive are not a dwell);
3. ``active and eligible`` extends the run, anything else resets it to 0;
4. confirmed iff ``run >= required``.

``required <= 1`` confirms every active window (dwell off, the pre-dwell behaviour).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

DEFAULT_DWELL_WINDOWS = 2


@dataclass
class DwellCounter:
    required: int = DEFAULT_DWELL_WINDOWS
    max_gap_ms: Optional[float] = None
    run: int = 0
    last_end_ms: Optional[int] = None

    def reset(self) -> None:
        self.run = 0
        self.last_end_ms = None

    @property
    def confirmed(self) -> bool:
        return self.run >= max(1, int(self.required))

    def update(self, window_end_ms: int, active: bool, *, eligible: bool = True) -> bool:
        end = int(window_end_ms)
        if self.last_end_ms is not None and end <= self.last_end_ms:
            return self.confirmed
        if (
            self.max_gap_ms is not None
            and self.max_gap_ms > 0
            and self.last_end_ms is not None
            and end - self.last_end_ms > self.max_gap_ms
        ):
            self.run = 0
        self.run = self.run + 1 if (active and eligible) else 0
        self.last_end_ms = end
        return self.confirmed


def dwell_confirmed_series(
    flags: Sequence[bool],
    window_ends_ms: Sequence[int],
    *,
    required: int = DEFAULT_DWELL_WINDOWS,
    max_gap_ms: Optional[float] = None,
    eligible: Optional[Iterable[bool]] = None,
) -> list[bool]:
    """Offline form: per window, whether ``flags`` has held for ``required`` new windows.

    Rows with a repeated ``window_end_ms`` get the verdict of the first row of that
    window, exactly like the controller's repeated snapshot reads.
    """
    if len(flags) != len(window_ends_ms):
        raise ValueError("flags and window_ends_ms must have the same length")
    elig = list(eligible) if eligible is not None else [True] * len(flags)
    if len(elig) != len(flags):
        raise ValueError("eligible must have the same length as flags")
    counter = DwellCounter(required=required, max_gap_ms=max_gap_ms)
    return [
        counter.update(end, bool(flag), eligible=bool(ok))
        for flag, end, ok in zip(flags, window_ends_ms, elig)
    ]

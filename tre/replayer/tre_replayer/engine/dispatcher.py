from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from tre_replayer.engine.rps_timeline import (
    ERROR_WINDOW_S,
    build_rps_timeline,
    max_relative_rps_error,
)
from tre_replayer.engine.schedule import ScheduledRequest

Sender = Callable[[ScheduledRequest, float, float], Awaitable[None]]
Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class DispatchRecord:
    request_id: str
    model: str
    scheduled_ts: float
    actual_ts: float

    @property
    def delay_s(self) -> float:
        return max(0.0, self.actual_ts - self.scheduled_ts)


@dataclass(frozen=True)
class DispatchReport:
    records: list[DispatchRecord]
    planned_duration_s: float
    actual_duration_s: float
    #: Monotonic instant the run started; offset 0 of the schedule's time base.
    base_ts: float = 0.0

    @property
    def p99_delay_ms(self) -> float:
        if not self.records:
            return 0.0
        delays = sorted(record.delay_s for record in self.records)
        index = max(0, math.ceil(0.99 * len(delays)) - 1)
        return delays[index] * 1000.0

    def rps_timeline(self, *, window_s: float = ERROR_WINDOW_S) -> list:
        """Scheduled vs fired arrivals, binned on the schedule's own time base."""
        return build_rps_timeline(
            [record.scheduled_ts - self.base_ts for record in self.records],
            [record.actual_ts - self.base_ts for record in self.records],
            window_s=window_s,
        )

    @property
    def actual_rps_error_ratio(self) -> float:
        """Largest relative gap between the nominal and the fired rate, over any window.

        The previous definition compared two *span* rates - ``n / (last - first)``
        scheduled against the same over the fired instants - and that quantity cannot see
        the failure mode it was there to catch. A span ratio only moves when the run's
        first-to-last duration moves, so a generator that fires every single request a
        uniform amount late reports a perfect 0.0000: both spans are identical, and the
        entire run is nonetheless offered off its schedule. That was measured, not
        supposed: a uniform 377 ms lateness scored 0.0000 under the old definition.

        Binning both instant sets on the schedule's own grid removes the blind spot
        without adding a false positive: the two sets contain the *same events*, so a
        schedule that was kept puts every event in the window it belonged to and scores
        exactly 0 however ragged its Poisson arrivals are. See
        :mod:`tre_replayer.engine.rps_timeline`.

        Note this is the *dispatcher's* view - it ends when the send was handed off, not
        when it reached the wire. ``scripts.openloop`` builds the same comparison from the
        sender's on-wire instants, which is the series a run should be judged on.
        """
        if len(self.records) < 2:
            return 0.0
        error = max_relative_rps_error(self.rps_timeline())
        return 0.0 if error < 1e-9 else error


async def dispatch_open_loop(
    events: Sequence[ScheduledRequest],
    sender: Sender,
    *,
    clock: Clock = time.monotonic,
    sleep: Sleep = asyncio.sleep,
) -> DispatchReport:
    ordered = sorted(events, key=lambda event: event.scheduled_offset_s)
    if not ordered:
        return DispatchReport(
            records=[], planned_duration_s=0.0, actual_duration_s=0.0, base_ts=clock()
        )

    base_ts = clock()
    records: list[DispatchRecord] = []
    tasks: list[asyncio.Task[None]] = []
    for event in ordered:
        scheduled_ts = base_ts + event.scheduled_offset_s
        delay = scheduled_ts - clock()
        if delay > 0.0:
            await sleep(delay)
        actual_ts = clock()
        records.append(
            DispatchRecord(
                request_id=event.request_id,
                model=event.model,
                scheduled_ts=scheduled_ts,
                actual_ts=actual_ts,
            )
        )
        tasks.append(asyncio.create_task(sender(event, scheduled_ts, actual_ts)))

    if tasks:
        await asyncio.gather(*tasks)

    planned_duration_s = ordered[-1].scheduled_offset_s - ordered[0].scheduled_offset_s
    actual_duration_s = records[-1].actual_ts - records[0].actual_ts if len(records) >= 2 else 0.0
    return DispatchReport(
        records=records,
        planned_duration_s=planned_duration_s,
        actual_duration_s=actual_duration_s,
        base_ts=base_ts,
    )

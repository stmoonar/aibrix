from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

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

    @property
    def p99_delay_ms(self) -> float:
        if not self.records:
            return 0.0
        delays = sorted(record.delay_s for record in self.records)
        index = max(0, math.ceil(0.99 * len(delays)) - 1)
        return delays[index] * 1000.0

    @property
    def actual_rps_error_ratio(self) -> float:
        if len(self.records) < 2 or self.planned_duration_s <= 0.0 or self.actual_duration_s <= 0.0:
            return 0.0
        planned_rps = len(self.records) / self.planned_duration_s
        actual_rps = len(self.records) / self.actual_duration_s
        error = abs(actual_rps - planned_rps) / planned_rps
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
        return DispatchReport(records=[], planned_duration_s=0.0, actual_duration_s=0.0)

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
    )

from __future__ import annotations

import asyncio

from tre_replayer.engine.dispatcher import dispatch_open_loop
from tre_replayer.engine.schedule import ScheduledRequest


def test_dispatch_open_loop_does_not_wait_for_previous_response() -> None:
    release = asyncio.Event()
    started: list[str] = []
    events = [
        ScheduledRequest(request_id="r0", model="m", scheduled_offset_s=0.0),
        ScheduledRequest(request_id="r1", model="m", scheduled_offset_s=0.0),
        ScheduledRequest(request_id="r2", model="m", scheduled_offset_s=0.0),
    ]

    async def sender(event: ScheduledRequest, scheduled_ts: float, actual_ts: float) -> None:
        started.append(event.request_id)
        if len(started) == len(events):
            release.set()
        await release.wait()

    report = asyncio.run(asyncio.wait_for(dispatch_open_loop(events, sender), timeout=0.5))

    assert started == ["r0", "r1", "r2"]
    assert [record.request_id for record in report.records] == ["r0", "r1", "r2"]


def test_dispatch_open_loop_reports_schedule_delay_with_injected_clock() -> None:
    now = 100.0
    events = [
        ScheduledRequest(request_id="r0", model="m", scheduled_offset_s=0.0),
        ScheduledRequest(request_id="r1", model="m", scheduled_offset_s=0.010),
        ScheduledRequest(request_id="r2", model="m", scheduled_offset_s=0.020),
    ]

    def clock() -> float:
        return now

    async def sleep(duration_s: float) -> None:
        nonlocal now
        now += max(0.0, duration_s)
        await asyncio.sleep(0)

    async def sender(event: ScheduledRequest, scheduled_ts: float, actual_ts: float) -> None:
        return None

    report = asyncio.run(dispatch_open_loop(events, sender, clock=clock, sleep=sleep))

    assert [record.delay_s for record in report.records] == [0.0, 0.0, 0.0]
    assert report.p99_delay_ms == 0.0
    assert report.actual_rps_error_ratio == 0.0

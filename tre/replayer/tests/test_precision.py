from __future__ import annotations

import asyncio

from tre_replayer.precision import run_offline_precision_check


def test_offline_precision_check_uses_stub_sender_and_reports_thresholds() -> None:
    now = 100.0

    def clock() -> float:
        return now

    async def sleep(duration_s: float) -> None:
        nonlocal now
        now += max(0.0, duration_s)
        await asyncio.sleep(0)

    result = asyncio.run(
        run_offline_precision_check(
            duration_s=1.0,
            target_rps=10.0,
            p99_delay_limit_ms=10.0,
            rps_error_limit=0.01,
            clock=clock,
            sleep=sleep,
        )
    )

    assert result.passed is True
    assert result.request_count == 10
    assert result.p99_delay_ms == 0.0
    assert result.actual_rps_error_ratio == 0.0
    assert result.p99_delay_limit_ms == 10.0
    assert result.rps_error_limit == 0.01

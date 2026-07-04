from __future__ import annotations

import asyncio
from dataclasses import dataclass

from tre_replayer.engine.dispatcher import Clock, DispatchReport, Sleep, dispatch_open_loop
from tre_replayer.engine.schedule import RpsSegment, ScheduledRequest, build_deterministic_schedule


@dataclass(frozen=True)
class PrecisionCheckResult:
    passed: bool
    request_count: int
    p99_delay_ms: float
    actual_rps_error_ratio: float
    p99_delay_limit_ms: float
    rps_error_limit: float


async def run_offline_precision_check(
    *,
    duration_s: float = 60.0,
    target_rps: float = 10.0,
    p99_delay_limit_ms: float = 10.0,
    rps_error_limit: float = 0.01,
    clock: Clock | None = None,
    sleep: Sleep | None = None,
) -> PrecisionCheckResult:
    events = build_deterministic_schedule([RpsSegment("stub-model", 0.0, duration_s, target_rps)])

    async def stub_sender(event: ScheduledRequest, scheduled_ts: float, actual_ts: float) -> None:
        return None

    kwargs = {}
    if clock is not None:
        kwargs["clock"] = clock
    if sleep is not None:
        kwargs["sleep"] = sleep
    report: DispatchReport = await dispatch_open_loop(events, stub_sender, **kwargs)
    p99_delay_ms = report.p99_delay_ms
    rps_error = report.actual_rps_error_ratio
    return PrecisionCheckResult(
        passed=p99_delay_ms < p99_delay_limit_ms and rps_error < rps_error_limit,
        request_count=len(report.records),
        p99_delay_ms=p99_delay_ms,
        actual_rps_error_ratio=rps_error,
        p99_delay_limit_ms=p99_delay_limit_ms,
        rps_error_limit=rps_error_limit,
    )


def main() -> int:
    result = asyncio.run(run_offline_precision_check())
    print(
        f"offline precision: passed={result.passed} requests={result.request_count} "
        f"p99_delay_ms={result.p99_delay_ms:.3f} "
        f"rps_error={result.actual_rps_error_ratio:.6f}"
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

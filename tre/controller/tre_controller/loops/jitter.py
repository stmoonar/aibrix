from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable


@dataclass(frozen=True)
class FastLoopJitterProbeResult:
    rescue_started_at_s: tuple[float, ...]
    fairness_started_at_s: tuple[float, ...]
    rescue_intervals_s: tuple[float, ...]
    max_rescue_jitter_s: float


async def run_fast_loop_jitter_probe(
    *,
    rescue_interval_s: float = 5.0,
    fairness_interval_s: float = 10.0,
    slow_loop_delay_s: float = 2.0,
    rescue_samples: int = 5,
    time_scale: float = 0.02,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> FastLoopJitterProbeResult:
    if rescue_interval_s <= 0 or fairness_interval_s <= 0:
        raise ValueError("loop intervals must be positive")
    if slow_loop_delay_s < 0:
        raise ValueError("slow_loop_delay_s must be non-negative")
    if rescue_samples < 2:
        raise ValueError("rescue_samples must be at least 2")
    if time_scale <= 0:
        raise ValueError("time_scale must be positive")

    loop = asyncio.get_running_loop()
    started_at = loop.time()
    rescue_starts: list[float] = []
    fairness_starts: list[float] = []
    stop = asyncio.Event()

    async def rescue_loop() -> None:
        for _ in range(rescue_samples):
            rescue_starts.append(_logical_time(loop.time(), started_at, time_scale))
            await sleep(rescue_interval_s * time_scale)
        stop.set()

    async def fairness_loop() -> None:
        while not stop.is_set():
            fairness_starts.append(_logical_time(loop.time(), started_at, time_scale))
            await sleep(slow_loop_delay_s * time_scale)
            if stop.is_set():
                break
            await sleep(fairness_interval_s * time_scale)

    fairness = asyncio.create_task(fairness_loop())
    await rescue_loop()
    await fairness

    intervals = tuple(
        rescue_starts[index + 1] - rescue_starts[index]
        for index in range(len(rescue_starts) - 1)
    )
    max_jitter = max((abs(interval - rescue_interval_s) for interval in intervals), default=0.0)
    return FastLoopJitterProbeResult(
        rescue_started_at_s=tuple(rescue_starts),
        fairness_started_at_s=tuple(fairness_starts),
        rescue_intervals_s=intervals,
        max_rescue_jitter_s=max_jitter,
    )


def _logical_time(now: float, started_at: float, time_scale: float) -> float:
    return (now - started_at) / time_scale

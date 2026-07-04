from __future__ import annotations

import asyncio

from tre_controller.loops.jitter import run_fast_loop_jitter_probe


def test_fast_loop_jitter_stays_within_budget_when_slow_loop_is_delayed() -> None:
    result = asyncio.run(
        run_fast_loop_jitter_probe(
            rescue_interval_s=5.0,
            fairness_interval_s=10.0,
            slow_loop_delay_s=2.0,
            rescue_samples=5,
            time_scale=0.02,
        )
    )

    assert len(result.rescue_started_at_s) == 5
    assert result.fairness_started_at_s
    assert result.max_rescue_jitter_s <= 0.5
    assert all(4.5 <= interval <= 5.5 for interval in result.rescue_intervals_s)

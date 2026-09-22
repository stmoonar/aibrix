"""The unified TSS definition (plan 2026-09-21 §6.4), tre_common.tss."""
from __future__ import annotations

import logging
import math

import pytest

from tre_common import tss


def _terms(**over):
    kw = dict(
        prompt_tokens=3000.0,
        generation_tokens=6000.0,
        window_ms=30_000.0,
        avg_running=4.0,
        avg_waiting=1.0,
        w_p=0.02,
        lambda_wait=3.0,
        qmin=1.0,
    )
    kw.update(over)
    return tss.tss_terms(**kw)


def test_numerator_is_a_rate_so_theta_does_not_scale_with_the_window() -> None:
    base = _terms()
    # Same traffic over a window twice as long: totals double, the rate does not.
    doubled = _terms(prompt_tokens=6000.0, generation_tokens=12000.0, window_ms=60_000.0)
    assert base.numerator_rate == pytest.approx((0.02 * 3000.0 + 6000.0) / 30.0)
    assert doubled.raw == pytest.approx(base.raw)
    assert base.queue == 4.0 + 3.0 * 1.0
    assert base.raw == pytest.approx(base.numerator_rate / 7.0)


def test_prefill_counts_only_cache_miss_tokens() -> None:
    hit = _terms(kv_cache_hit_rate=0.5)
    assert hit.numerator_rate == pytest.approx((0.02 * 1500.0 + 6000.0) / 30.0)


def test_swapping_and_w_d_are_ignored_with_a_warning(caplog) -> None:
    tss._warned.clear()
    with caplog.at_level(logging.WARNING, logger="tre_common.tss"):
        odd = _terms(avg_swapping=2.0, w_d=1.5)
    assert odd.raw == _terms().raw
    text = caplog.text
    assert "swapping" in text and "w_d" in text


def test_qmin_guards_a_small_queue() -> None:
    small = _terms(avg_running=0.25, avg_waiting=0.0)
    assert small.queue_ctl == 1.0
    assert small.raw == pytest.approx(small.numerator_rate)


def test_idle_rule_nothing_in_flight_means_undefined() -> None:
    idle = _terms(avg_running=0.0, avg_waiting=0.0)
    assert idle.raw is None and not idle.defined
    assert idle.numerator_rate > 0.0  # tokens completed, yet nothing was in flight


def test_window_must_be_positive() -> None:
    with pytest.raises(ValueError):
        _terms(window_ms=0.0)
    with pytest.raises(ValueError):
        _terms(window_ms=None)


def test_replica_factor_keeps_the_controller_guards() -> None:
    assert tss.replica_factor(4, 2) == 2.0
    assert tss.replica_factor(0, 3) == 1.0
    assert tss.replica_factor(2, 0) == 2.0


def test_ema_alpha_is_the_wall_clock_weight() -> None:
    assert tss.ema_alpha(5_000, 20_000) == pytest.approx(1 - math.exp(-0.25))
    assert tss.ema_alpha(0, 20_000) == 0.0


def test_smooth_series_seeds_passes_undefined_through_and_decays() -> None:
    out = tss.smooth_series([100.0, None, 200.0, 200.0], [30_000, 35_000, 40_000, 40_000], tau_ms=20_000)
    assert out[0] == 100.0 and out[1] is None
    decay = math.exp(-10_000 / 20_000)  # the undefined window did not advance the clock
    assert out[2] == decay * 100.0 + (1 - decay) * 200.0
    assert out[3] == out[2]  # duplicate window end holds


def test_theta_conversion_divides_by_the_window_seconds() -> None:
    assert tss.convert_window_total_theta(1718.2369972339602) == 1718.2369972339602 / 30.0
    assert tss.convert_window_total_theta(739.0, 20_000) == 739.0 / 20.0

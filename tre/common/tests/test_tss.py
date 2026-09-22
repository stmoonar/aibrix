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
        avg_running=4.0,
        avg_waiting=1.0,
        w_p=0.02,
        lambda_wait=3.0,
        qmin=1.0,
    )
    kw.update(over)
    return tss.tss_terms(**kw)


def test_numerator_is_the_window_token_total() -> None:
    # Tokens per metrics window (TRE_METRICS_WINDOW_MS), the main / v1 convention.
    base = _terms()
    assert base.numerator == 0.02 * 3000.0 + 6000.0
    assert base.queue == 4.0 + 3.0 * 1.0
    assert base.raw == pytest.approx(base.numerator / 7.0)


@pytest.mark.parametrize("window_ms", [30_000.0, 20_000.0, 1_000.0])
def test_z_is_invariant_to_the_total_vs_rate_convention(window_ms) -> None:
    # Window-total numerator with theta_total, or the same tokens divided by the window
    # seconds (a rate) with theta_total / window_s: Z is the same number. Only theta's
    # magnitude depends on the convention (x window seconds).
    window_s = window_ms / 1000.0
    theta_total = 1718.2369972339602
    total = _terms()
    rate = _terms(prompt_tokens=3000.0 / window_s, generation_tokens=6000.0 / window_s)
    theta_rate = tss.convert_window_total_theta(theta_total, window_ms)
    assert theta_rate == theta_total / window_s
    assert rate.raw / theta_rate == pytest.approx(total.raw / theta_total, rel=1e-12)


def test_prefill_counts_only_cache_miss_tokens() -> None:
    hit = _terms(kv_cache_hit_rate=0.5)
    assert hit.numerator == pytest.approx(0.02 * 1500.0 + 6000.0)


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
    assert small.raw == pytest.approx(small.numerator)


def test_idle_rule_nothing_in_flight_means_undefined() -> None:
    idle = _terms(avg_running=0.0, avg_waiting=0.0)
    assert idle.raw is None and not idle.defined
    assert idle.numerator > 0.0  # tokens completed, yet nothing was in flight


def test_the_formula_takes_no_window_duration() -> None:
    # Totals need no duration; a stray window_ms must not silently turn them into rates.
    with pytest.raises(TypeError):
        _terms(window_ms=30_000.0)


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


def test_theta_rate_equivalent_divides_by_the_window_seconds() -> None:
    assert tss.convert_window_total_theta(1718.2369972339602) == 1718.2369972339602 / 30.0
    assert tss.convert_window_total_theta(739.0, 20_000) == 739.0 / 20.0

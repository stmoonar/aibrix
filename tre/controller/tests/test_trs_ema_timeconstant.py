"""S1.3: wall-clock time-constant EMA for TRS (ADR-0011).

The live control path used to construct a fresh TRSComputer every tick, so the
EMA never persisted (TRS == TRS_raw). These tests pin the new time-constant EMA:
smoothing strength is set by tau alone and is decoupled from refresh frequency;
the EMA advances at most once per distinct window_end_ms.
"""

from __future__ import annotations

import math

import pytest

from tre_controller.signals.trs import SignalState, TRSComputer, TRSInput


def _inp(*, generation: float, running: float) -> TRSInput:
    # Y = generation * w_d (prompt=0); Q_ctl = max(running, qmin); trs_raw = Y/Q_ctl.
    return TRSInput(
        prompt_tokens_total=0.0,
        generation_tokens_total=generation,
        avg_waiting=0.0,
        avg_running=running,
        avg_swapping=0.0,
        routable_pods=1,
        assigned_replicas=1,
        w_p=0.04,
        w_d=1.0,
        lambda_wait=2.625,
        qmin=1.0,
        kv_cache_hit_rate=0.0,
    )


def test_first_sample_returns_raw_and_seeds() -> None:
    c = TRSComputer(ema_alpha=0.5, ema_tau_ms=20_000)
    r = c.compute(_inp(generation=200.0, running=2.0), window_end_ms=0)
    assert r.TRS_raw == pytest.approx(100.0)
    assert r.TRS == pytest.approx(100.0)  # prev is None -> raw


def test_time_constant_decay_matches_formula() -> None:
    c = TRSComputer(ema_tau_ms=20_000)
    c.compute(_inp(generation=200.0, running=2.0), window_end_ms=0)  # seed 100
    r = c.compute(_inp(generation=400.0, running=2.0), window_end_ms=5_000)  # raw 200
    decay = math.exp(-5_000 / 20_000)
    assert r.TRS == pytest.approx(decay * 100.0 + (1 - decay) * 200.0)


def test_alpha_equivalent_tau_reproduces_old_60s_decay() -> None:
    # Doc S1.3: a 60s step with tau reverse-derived from alpha_old=0.2485 must
    # reproduce decay ~= 0.2485 (the legacy per-step weight when a window was ~60s).
    tau = -60_000 / math.log(0.2485)
    c = TRSComputer(ema_tau_ms=tau)
    c.compute(_inp(generation=200.0, running=2.0), window_end_ms=0)  # seed 100
    r = c.compute(_inp(generation=400.0, running=2.0), window_end_ms=60_000)  # raw 200
    assert r.TRS == pytest.approx(0.2485 * 100.0 + 0.7515 * 200.0, rel=1e-3)


def test_high_frequency_step_uses_wallclock_decay() -> None:
    # Doc S1.3: same tau, dt=5s -> decay ~= exp(-5/43.4) ~= 0.891 (moves less per
    # step, but the same amount per unit wall-clock time as the 60s step above).
    tau = -60_000 / math.log(0.2485)
    c = TRSComputer(ema_tau_ms=tau)
    c.compute(_inp(generation=200.0, running=2.0), window_end_ms=0)  # seed 100
    r = c.compute(_inp(generation=400.0, running=2.0), window_end_ms=5_000)
    decay = math.exp(-5_000 / tau)
    assert decay == pytest.approx(0.891, rel=1e-2)
    assert r.TRS == pytest.approx(decay * 100.0 + (1 - decay) * 200.0)


def test_duplicate_window_does_not_advance() -> None:
    c = TRSComputer(ema_tau_ms=20_000)
    c.compute(_inp(generation=200.0, running=2.0), window_end_ms=0)  # seed 100
    r1 = c.compute(_inp(generation=400.0, running=2.0), window_end_ms=5_000)  # advance
    r2 = c.compute(_inp(generation=999.0, running=2.0), window_end_ms=5_000)  # same window -> hold
    assert r2.TRS == pytest.approx(r1.TRS)


def test_non_finite_raw_passthrough_no_state_change() -> None:
    c = TRSComputer(ema_tau_ms=20_000)
    c.compute(_inp(generation=200.0, running=2.0), window_end_ms=0)  # seed 100, cursor=0
    r = c.compute(_inp(generation=0.0, running=0.0), window_end_ms=5_000)  # raw 0 -> passthrough
    assert r.TRS == 0.0
    # cursor/state untouched: the next real sample decays from the seed (dt from 0).
    r2 = c.compute(_inp(generation=400.0, running=2.0), window_end_ms=10_000)
    decay = math.exp(-10_000 / 20_000)
    assert r2.TRS == pytest.approx(decay * 100.0 + (1 - decay) * 200.0)


def test_legacy_alpha_branch_when_tau_none() -> None:
    # ema_tau_ms None + no window_end_ms -> exact legacy fixed-alpha behaviour.
    c = TRSComputer(ema_alpha=0.5)
    c.compute(_inp(generation=200.0, running=2.0))  # seed 100
    r = c.compute(_inp(generation=400.0, running=2.0))  # 0.5*100 + 0.5*200
    assert r.TRS == pytest.approx(150.0)


def test_signal_state_shares_one_computer_per_model() -> None:
    state = SignalState()
    a = state.computer_for("m", ema_alpha=0.5, ema_tau_ms=20_000)
    b = state.computer_for("m", ema_alpha=0.5, ema_tau_ms=20_000)
    assert a is b
    other = state.computer_for("other", ema_alpha=0.5, ema_tau_ms=20_000)
    assert other is not a

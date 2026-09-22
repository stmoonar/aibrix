"""Idle-gap reset of the TSS EMA (fix/tss-rate-20260922).

Online (``TRSComputer`` / ``SignalState``) and offline (``smooth_series``,
``smooth_rows_by_cell``, ``r3_grid.compute_window_results``) share one rule, implemented
once in ``tre_common.tss.TssEma.update``: a sample arriving more than one metrics window
after the last advancing sample clears the EMA first. None / zero raws take part in the gap
check but never advance the EMA. An idle tick clears the warmup onset and the EMA together.
"""
from __future__ import annotations

import logging
import math
import random
from types import SimpleNamespace

from scripts.r3_grid import compute_window_results
from tre_calibration.dataset import smooth_rows_by_cell
from tre_common.metrics_schema import ModelWindowMetrics
from tre_common.registry import TrsParams
from tre_common.tss import TssEma, signal_ema, smooth_series
from tre_controller.signals.trs import SignalState, TRSComputer, TRSInput

TAU = 20_000.0
WIN = 30_000


def _params() -> TrsParams:
    return TrsParams(
        w_p=0.02, w_d=1.0, lambda_wait=3.0, qmin=1.0, ema_alpha=0.2485, theta_m=50.0,
        tau_crit=0.8, tau_low=1.0, tau_high=1.25, qsat=4.0, epsat=0.05, hsat=3,
        ema_tau_ms=TAU,
    )


def _window(end_ms: int, *, gen: float, running: float, prompt: float = 0.0, waiting: float = 0.0):
    return ModelWindowMetrics(
        model="m", window_start_ms=end_ms - WIN, window_end_ms=end_ms,
        prompt_tokens=prompt, generation_tokens=gen, avg_waiting=waiting, avg_running=running,
        avg_swapping=0.0, kv_cache_hit_rate=0.0, ttft_p95_ms=100.0, tpot_p95_ms=10.0,
        e2e_p95_ms=1000.0, routable_pods=1, assigned_replicas=1, per_pod={},
    )


def _idle(end_ms: int) -> ModelWindowMetrics:
    return _window(end_ms, gen=0.0, running=0.0)


# --- TssEma rule ------------------------------------------------------------------------

def test_gap_longer_than_a_window_resets() -> None:
    ema = TssEma(TAU)
    ema.update(100.0, 0, WIN)
    assert ema.update(200.0, WIN + 1, WIN) == 200.0  # reset, then seeded with raw
    assert ema.last_ms == WIN + 1


def test_gap_of_exactly_one_window_does_not_reset() -> None:
    # back-to-back tumbling windows are exactly one window apart: that is not a gap
    ema = TssEma(TAU)
    ema.update(100.0, 0, WIN)
    decay = math.exp(-WIN / TAU)
    assert ema.update(200.0, WIN, WIN) == decay * 100.0 + (1.0 - decay) * 200.0


def test_gap_shorter_than_a_window_does_not_reset() -> None:
    ema = TssEma(TAU)
    ema.update(100.0, 0, WIN)
    decay = math.exp(-10_000 / TAU)
    assert ema.update(200.0, 10_000, WIN) == decay * 100.0 + (1.0 - decay) * 200.0


def test_none_and_zero_do_not_advance_last_ms() -> None:
    ema = TssEma(TAU)
    ema.update(100.0, 0, WIN)
    assert ema.update(None, 10_000, WIN) is None
    assert ema.update(0.0, 20_000, WIN) == 0.0
    assert (ema.value, ema.last_ms) == (100.0, 0.0)


def test_none_and_zero_trigger_the_gap_reset() -> None:
    for passthrough in (None, 0.0):
        ema = TssEma(TAU)
        ema.update(100.0, 0, WIN)
        assert ema.update(passthrough, WIN + 5_000, WIN) == passthrough
        assert (ema.value, ema.last_ms) == (None, None)
        # the next defined sample - even one window after the idle one - seeds fresh
        assert ema.update(300.0, WIN + 10_000, WIN) == 300.0


def test_alternative_signal_ema_gets_the_same_gap_rule() -> None:
    ema = signal_ema(TAU)
    ema.update(4.0, 0, WIN)
    ema.update(0.0, 10_000, WIN)  # a zero queue is a sample for alternative signals
    assert ema.last_ms == 10_000
    assert ema.update(7.0, 10_000 + WIN + 1, WIN) == 7.0


def test_streaming_equals_smooth_series_bitwise() -> None:
    rng = random.Random(11)
    raws, ends, stream = [], [], []
    ema = TssEma(TAU)
    end = 0
    for _ in range(400):
        end += rng.choice((0, 5_000, 5_000, 10_000, 31_000, 65_000))
        raw = None if rng.random() < 0.15 else (0.0 if rng.random() < 0.1 else rng.uniform(1, 500))
        raws.append(raw)
        ends.append(end)
        stream.append(ema.update(raw, end, WIN))
    assert smooth_series(raws, ends, tau_ms=TAU, window_ms=WIN) == stream
    assert smooth_series(raws, ends, tau_ms=TAU, window_ms=[WIN] * len(raws)) == stream


# --- online TRSComputer / SignalState ------------------------------------------------------

def _run(computer: TRSComputer, windows) -> list[float | None]:
    out = []
    for wm in windows:
        r = computer.compute(TRSInput.from_metrics(wm, _params()), window_end_ms=wm.window_end_ms)
        out.append(r.TRS if r.defined else None)
    return out


def _traffic(start_end: int, n: int, step: int, seed: int) -> list[ModelWindowMetrics]:
    rng = random.Random(seed)
    return [
        _window(start_end + i * step, gen=rng.uniform(1e3, 3e4), prompt=rng.uniform(0, 5e4),
                running=rng.uniform(0.5, 40), waiting=rng.uniform(0, 10))
        for i in range(n)
    ]


def test_online_ema_resets_after_an_idle_gap() -> None:
    computer = TRSComputer(ema_tau_ms=TAU)
    _run(computer, [_window(30_000, gen=30_000.0, running=1.0)])  # raw 1000
    # 40 s later, no idle tick seen: the old value must not survive
    r = computer.compute(TRSInput.from_metrics(_window(70_000, gen=3_000.0, running=1.0), _params()),
                         window_end_ms=70_000)
    assert r.TRS == r.TRS_raw == 100.0


def test_restart_duality_fresh_equals_post_reset() -> None:
    period_a = _traffic(30_000, 20, 5_000, seed=1)
    gap_start = period_a[-1].window_end_ms
    idle = [_idle(gap_start + 5_000 * k) for k in range(1, 9)]
    period_b = _traffic(gap_start + 45_000 + WIN, 30, 5_000, seed=2)
    carried = TRSComputer(ema_tau_ms=TAU)
    _run(carried, period_a + idle)
    fresh = TRSComputer(ema_tau_ms=TAU)
    assert _run(carried, period_b) == _run(fresh, period_b)  # bitwise


def test_offline_path_matches_signal_state_bitwise() -> None:
    # tumbling windows (as r3_grid / rewindow write them) with duplicate window_end
    # re-reads, idle windows and a long hole; the online side also runs the idle tick.
    rng = random.Random(5)
    windows: list[ModelWindowMetrics] = []
    end = 60_000
    for _ in range(120):
        end += rng.choice((WIN, WIN, WIN, 2 * WIN, 4 * WIN))
        if rng.random() < 0.15:
            wm = _idle(end)
        else:
            wm = _window(end, gen=rng.uniform(0, 3e4), prompt=rng.uniform(0, 5e4),
                         running=rng.uniform(0.2, 40) if rng.random() > 0.05 else 0.0,
                         waiting=rng.uniform(0, 10))
        windows.append(wm)
        if rng.random() < 0.2:
            windows.append(wm)  # rescue/fairness re-read of the same snapshot
    spec = SimpleNamespace(trs=_params())

    offline = compute_window_results(windows, spec)

    state = SignalState()
    online = []
    for wm in windows:
        computer = state.computer_for("m", ema_alpha=spec.trs.ema_alpha, ema_tau_ms=spec.trs.ema_tau_ms)
        result = computer.compute(TRSInput.from_metrics(wm, spec.trs), theta_m=50.0,
                                  window_end_ms=wm.window_end_ms)
        state.observe_traffic("m", has_traffic=result.Y_m > 1e-9,
                              window_start_ms=wm.window_start_ms, window_end_ms=wm.window_end_ms)
        online.append(result)
    assert [r.TRS for r in offline] == [r.TRS for r in online]
    assert [r.Z_m for r in offline] == [r.Z_m for r in online]

    raws = [r.TRS_raw if r.defined else None for r in offline]
    ends = [wm.window_end_ms for wm in windows]
    recomputed = smooth_series(raws, ends, tau_ms=TAU, window_ms=WIN)
    assert recomputed == [r.TRS if r.defined else None for r in online]


def test_idle_tick_clears_onset_and_every_ema() -> None:
    state = SignalState(warmup_ms=-1)
    computer = state.computer_for("m", ema_alpha=0.2485, ema_tau_ms=TAU)
    _run(computer, [_window(60_000, gen=30_000.0, running=1.0)])
    state.smooth_signal("m", "queue_len", 5.0, window_end_ms=60_000, tau_ms=TAU, window_ms=WIN)
    state.smooth_signal("other", "queue_len", 9.0, window_end_ms=60_000, tau_ms=TAU, window_ms=WIN)
    state.observe_traffic("m", has_traffic=True, window_start_ms=30_000, window_end_ms=60_000)
    assert state._onset_ms["m"] == 60_000 and computer.current_ema is not None

    state.observe_traffic("m", has_traffic=False, window_start_ms=35_000, window_end_ms=65_000)
    assert state._onset_ms["m"] is None
    assert computer.current_ema is None and computer.tss_ema.last_ms is None
    assert state._signal_ema[("m", "queue_len")].value is None
    assert state._signal_ema[("other", "queue_len")].value == 9.0  # other models untouched

    # traffic resumes 5 s later (no gap rule): the EMA seeds from the new raw
    r = computer.compute(TRSInput.from_metrics(_window(70_000, gen=3_000.0, running=1.0), _params()),
                         window_end_ms=70_000)
    assert r.TRS == r.TRS_raw


def test_idle_tick_resets_ema_even_with_the_warmup_guard_disabled() -> None:
    state = SignalState(warmup_ms=0)
    computer = state.computer_for("m", ema_alpha=0.2485, ema_tau_ms=TAU)
    _run(computer, [_window(60_000, gen=30_000.0, running=1.0)])
    assert state.observe_traffic("m", has_traffic=False, window_start_ms=35_000, window_end_ms=65_000)
    assert computer.current_ema is None


# --- offline smoothing ----------------------------------------------------------------------

def _row(cell: str, end: int) -> dict:
    return {"scenario_id": cell, "window_start_ms": end - WIN, "window_end_ms": end}


def test_offline_rows_reset_within_a_cell_after_a_gap() -> None:
    rows = [_row("a", 30_000), _row("a", 35_000), _row("a", 35_000 + WIN + 1_000)]
    out = smooth_rows_by_cell(rows, [100.0, 200.0, 300.0], make_ema=lambda: TssEma(TAU))
    assert out[2] == 300.0


def test_offline_warns_when_a_cell_starts_within_a_window_of_the_previous(caplog) -> None:
    close = [_row("a", 30_000), _row("a", 60_000), _row("b", 60_000 + 10_000 + WIN)]
    with caplog.at_level(logging.WARNING, logger="tre_calibration.dataset"):
        smooth_rows_by_cell(close, [1.0, 2.0, 3.0], make_ema=lambda: TssEma(TAU))
    assert "a->b" in caplog.text

    caplog.clear()
    quiet = [_row("a", 30_000), _row("a", 60_000), _row("b", 60_000 + 45_000 + WIN)]
    with caplog.at_level(logging.WARNING, logger="tre_calibration.dataset"):
        smooth_rows_by_cell(quiet, [1.0, 2.0, 3.0], make_ema=lambda: TssEma(TAU))
    assert "a->b" not in caplog.text

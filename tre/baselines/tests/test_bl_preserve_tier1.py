"""PreServe Tier-1: trace oracle windows, forecast modes, noise, the N formula."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from tre_baselines import trace_oracle as to
from tre_baselines.policies import preserve_tier1 as t1
from tre_baselines.snapshot import ReplayInfo

FIX = Path(__file__).resolve().parent / "fixtures"


# ------------------------------------------------------------------ trace oracle


def test_request_list_trace_windows() -> None:
    o = to.load_oracle(FIX / "preserve_trace_requests.json", window_s=60.0)
    assert o.fmt == to.FORMAT_REQUESTS
    assert o.skipped_records == 1  # the record without an arrival
    assert o.n_requests == 5
    assert o.duration_s == 150.0 and o.n_windows == 3  # last arrival 150 s -> window 2
    w0 = o.window("dsqwen-7b", 0)
    assert (w0.P, w0.D, w0.n) == (300, 110, 2)
    w1 = o.window("dsqwen-7b", 1)
    assert (w1.P, w1.D, w1.n) == (300, 70, 1)
    w2 = o.window("dsqwen-7b", 2)
    assert (w2.P, w2.D, w2.missing_max) == (400, 0, 1)
    assert o.window("dsllama-8b", 0).P == 1000 and o.window("dsllama-8b", 1).n == 0
    assert o.window("no-such-model", 0).P == 0
    assert o.max_tokens_max == {"dsqwen-7b": 70, "dsllama-8b": 500}
    # effective window length: full, full, then 150 - 120 = 30 s floored at 60 s
    assert [o.window_len_s(i) for i in range(3)] == [60.0, 60.0, 60.0]
    o600 = to.load_oracle(FIX / "preserve_trace_requests.json", window_s=600.0)
    assert o600.n_windows == 1 and o600.window_len_s(0) == 150.0


def test_segment_trace_rebuilds_the_replayers_schedule() -> None:
    from tre_replayer.engine.schedule import build_poisson_schedule
    from tre_replayer.traces.loader import load_trace_segments

    path = FIX / "preserve_trace_segments.json"
    o = to.load_oracle(path, window_s=60.0, seed=7)
    assert o.fmt == to.FORMAT_SEGMENTS
    assert o.duration_s == 120.0 and o.n_windows == 2
    sched = build_poisson_schedule(load_trace_segments(path), seed=7)
    for model in ("dsqwen-7b", "dsllama-8b"):
        for w in range(2):
            mine = [e for e in sched if e.model == model and int(e.scheduled_offset_s // 60) == w]
            got = o.window(model, w)
            assert got.n == len(mine)
            assert got.P == sum(e.prompt_tokens for e in mine)
            assert got.D == sum(e.max_output_tokens for e in mine)
    # a different seed moves arrivals (and the log-uniform lengths)
    other = to.load_oracle(path, window_s=60.0, seed=8)
    assert (other.window("dsllama-8b", 0), other.window("dsllama-8b", 1)) != (
        o.window("dsllama-8b", 0), o.window("dsllama-8b", 1))
    det = to.load_oracle(path, window_s=60.0, schedule=to.SCHEDULE_DETERMINISTIC)
    assert det.window("dsqwen-7b", 0).n == 60 and det.window("dsqwen-7b", 0).P == 6000
    with pytest.raises(ValueError):
        to.load_oracle(path, schedule="bursty")


def test_window_index_and_trace_match() -> None:
    r = ReplayInfo(t0_ms=10_000, trace_path="/mnt/a/traces/case1/trace.json")
    assert to.window_index(9_999, r, 600) == -1
    assert to.window_index(10_000, r, 600) == 0
    assert to.window_index(10_000 + 599_999, r, 600) == 0
    assert to.window_index(10_000 + 600_000, r, 600) == 1
    assert to.window_index(10_000 + 590_000, r, 600, lead_s=10) == 1
    o = to.build_oracle([], path="repo/traces/case1/trace.json", fmt=to.FORMAT_REQUESTS,
                        window_s=600, duration_s=0)
    assert o.matches(r)
    assert not o.matches(ReplayInfo(0, "/mnt/a/traces/case2/trace.json"))
    assert o.matches(ReplayInfo(0, "/mnt/a/traces/case2/trace.json"), parts=1)
    assert o.matches(ReplayInfo(0, "anything"), parts=0)
    assert o.matches(ReplayInfo(0, "repo\\traces\\case1\\trace.json"))


# --------------------------------------------------------------- forecast modes


def test_modes() -> None:
    kw = dict(true_P=1000, true_D=500, seed=3, model="m", window=2, sigma=0.1)
    assert t1.estimate("oracle", prev=(1, 2), **kw) == (1000.0, 500.0)
    assert t1.estimate("last_window", prev=(700, 300), **kw) == (700.0, 300.0)
    assert t1.estimate("last_window", prev=None, **kw) is None
    P, D = t1.estimate("oracle_noisy", prev=None, **kw)
    assert P == pytest.approx(1000 * t1.noise_factor(3, "m", 2, "P", 0.1))
    assert D == pytest.approx(500 * t1.noise_factor(3, "m", 2, "D", 0.1))
    assert (P, D) != (1000.0, 500.0)
    assert t1.estimate("oracle_noisy", prev=None, **{**kw, "sigma": 0.0}) == (1000.0, 500.0)
    with pytest.raises(ValueError):
        t1.estimate("mlstm", prev=None, **kw)


def test_noise_sigma_matches_paper_mean_ape_and_is_deterministic() -> None:
    sigma = t1.DEFAULT_NOISE_SIGMA
    assert sigma == pytest.approx(0.0772, abs=5e-4)
    assert t1.mean_ape_of_sigma(sigma) == pytest.approx(t1.PAPER_MEAN_APE, abs=1e-9)
    assert t1.sigma_for_mean_ape(0.0) == 0.0
    draws = [t1.noise_factor(0, "m", w, "P", sigma) for w in range(20_000)]
    ape = sum(abs(f - 1.0) for f in draws) / len(draws)
    assert ape == pytest.approx(0.0617, abs=0.002)
    # same inputs -> same factor; any input changes it
    a = t1.noise_factor(5, "m", 1, "P", sigma)
    assert a == t1.noise_factor(5, "m", 1, "P", sigma)
    assert len({a, t1.noise_factor(6, "m", 1, "P", sigma), t1.noise_factor(5, "n", 1, "P", sigma),
                t1.noise_factor(5, "m", 2, "P", sigma), t1.noise_factor(5, "m", 1, "D", sigma)}) == 5


def test_required_replicas_formula() -> None:
    mu = t1.Mu(p=100.0, d=50.0, t=120.0)
    # P/(mu_p W) = 1.5, D/(mu_d W) = 1.0, (P+D)/(mu_t W) = 1.875 -> 2
    assert t1.required_replicas(90_000, 30_000, mu, 600.0) == 2
    # decode-bound: D/(mu_d W) = 3.2
    assert t1.required_replicas(10_000, 96_000, mu, 600.0) == 4
    # exact integer is not rounded up by float noise
    assert t1.required_replicas(60_000, 0, t1.Mu(100.0, 1.0, 1e9), 600.0) == 1
    assert t1.required_replicas(0, 0, mu, 600.0) == 0
    # a shorter (partial) window means a higher rate
    assert t1.required_replicas(30_000, 0, t1.Mu(100.0, 1.0, 1e9), 150.0) == 2
    with pytest.raises(ValueError):
        t1.required_replicas(1, 1, mu, 0.0)


def test_parse_mu_fails_closed() -> None:
    ok = {"a": {"p": 1, "d": 2, "t": 3}}
    assert t1.parse_mu(ok, ["a"])["a"] == t1.Mu(1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="managed models"):
        t1.parse_mu(ok, ["a", "b"])
    with pytest.raises(ValueError):
        t1.parse_mu(None, ["a"])
    with pytest.raises(ValueError):
        t1.parse_mu({"a": {"p": 1, "d": 2}}, ["a"])
    with pytest.raises(ValueError):
        t1.parse_mu({"a": {"p": 1, "d": 0, "t": 3}}, ["a"])
    with pytest.raises(ValueError):
        t1.parse_mu({"a": {"p": 1, "d": "x", "t": 3}}, ["a"])
    assert math.isfinite(t1.parse_mu({"a": {"p": "1.5", "d": 2, "t": 3}}, [])["a"].p)

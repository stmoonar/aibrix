"""Invariants of the V / mu calibration tools (tools/tokenscale_buckets.py, tools/velocity_ladder.py):
the paper's cell boundaries, calibration points inside their own cell, the D53 full / stop rule,
V = peak of full steps, and PreServe's all-requests normalized-latency criterion."""
from __future__ import annotations

import json

import pytest
import yaml

from tre_baselines.tools import tokenscale_buckets as tb
from tre_baselines.tools import velocity_ladder as vl


# ------------------------------------------------------------------ cells (S0)

@pytest.mark.parametrize("tin,tout,cell", [
    (255, 99, (0, 0)), (256, 100, (1, 1)), (1023, 349, (1, 1)), (1024, 350, (2, 2)),
    (16310, 50, (2, 0)),  # an input above 8192 is long
])
def test_paper_boundaries(tin, tout, cell):
    assert tb.cell_of(tin, tout) == cell
    # the policy's inclusive edges give the same cell
    e = tb.POLICY_EDGES
    idx = lambda x, ed: 0 if x <= ed[0] else (1 if x <= ed[1] else 2)  # noqa: E731
    assert (idx(tin, e["in"]), idx(tout, e["out"])) == cell


def _reqs(model, pairs, dt=0.1):
    return [(model, k * dt, i, o) for k, (i, o) in enumerate(pairs)]


def test_center_is_the_cell_mean_and_stays_in_its_cell():
    pairs = [(255, 99), (255, 99), (254, 98)] + [(500, 200)] * 3 + [(3000, 700)] * 3
    doc = tb.report({"t": _reqs("m", pairs)}, {"t": 30.0}, window_s=30.0, min_reqs=1)
    c = doc["bucket_centers"]["m"]
    assert c[0][0] == [255, 99]          # mean 254.7 / 98.7 rounds to 255 / 99, still in cell 00
    assert c[1][1] == [500, 200] and c[2][2] == [3000, 700]
    for i, row in enumerate(c):
        for j, (tin, tout) in enumerate(row):
            assert tb.cell_of(tin, tout) == (i, j)
    # an empty cell: mean input of its input class, mean output of its output class
    assert c[1][2] == [500, 700] and doc["cells"]["m"]["12"]["n"] == 0


def test_p_share_uses_only_complete_windows_with_enough_requests():
    reqs = [("m", 1.0, 300, 100)] * 10 + [("m", 31.0, 100, 300)] * 10 + [("m", 61.0, 1, 1)] * 10
    sh = tb.window_shares(reqs, duration_s=60.0, window_s=30.0, min_reqs=10)
    assert sh["m"] == [0.75, 0.25]       # window 2 (t >= 60) is not complete
    assert tb.window_shares(reqs, 60.0, 30.0, min_reqs=11) == {}


def test_cli_reads_a_trace_set_by_root_and_names(tmp_path):
    for name in ("A", "B"):
        d = tmp_path / name / "seed1"
        d.mkdir(parents=True)
        recs = [{"request_id": f"r{k}", "timestamp": k * 0.5, "model_name": "m", "prompt_length": (100, 500, 2000)[k % 3],
                 "max_output_tokens": 50 + 10 * k, "phase_type": "x"} for k in range(40)]
        (d / "design.json").write_text(json.dumps(recs), encoding="utf-8")
        (d / "manifest.json").write_text(json.dumps({"duration_s": 20}), encoding="utf-8")
    out = tmp_path / "rep.yaml"
    assert tb.main(["--trace-root", str(tmp_path), "--traces", "A,B", "--out", str(out), "--min-reqs", "1",
                    "--window-s", "10"]) == 0
    doc = yaml.safe_load(out.read_text())
    assert set(doc["sources"]) == {"A", "B"} and all(len(s["sha256"]) == 64 for s in doc["sources"].values())
    assert doc["mix_shape"]["m"]["n"] == 80 and doc["p_share"]["pooled"]["windows"] == 4
    assert tb.main(["--out", str(out)]) == 2


# ------------------------------------------------------------------ ladder rule (S4)

RULES = vl.Rules(min_completed=1, min_samples=1)


def _step(rate, tok, waiting_slope, **kw):
    s = vl.Step("m", "bucket", "b11", 500, 200, rate, arrived=100, completed=100, prefill_tok_s=tok * 0.7,
                decode_tok_s=tok * 0.3, waiting_slope=waiting_slope, window_samples=60, max_latency_s=10.0)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _scripted(table):
    """measure from a rate -> (tok_s, waiting slope) function."""
    calls = []

    def measure(model, cell, tin, tout, rate):
        calls.append(round(rate, 3))
        tok, ws = table(rate)
        return _step(rate, tok, ws)

    measure.calls = calls
    return measure


def test_ladder_stops_after_two_consecutive_full_steps_and_v_is_their_peak():
    # capacity 10 req/s: above it the queue grows at (rate - 10) and throughput falls a little
    table = lambda r: (min(r, 10.0) * 700 - max(0.0, r - 10.0) * 50, max(0.0, r - 10.0))  # noqa: E731
    m = _scripted(table)
    rates = vl.ladder_rates(10.0, (0.8, 0.9, 1.0, 1.1, 1.2, 1.3), 1.1, 10)
    steps = vl.run_ladder(m, "m", "b11", 500, 200, rates, RULES, log=lambda _l: None)
    assert m.calls == [8.0, 9.0, 10.0, 11.0, 12.0]          # 11 and 12 are full -> stop
    p = vl.peak(steps, RULES)
    assert p["saturated"] and p["full_rps"] == [11.0, 12.0]
    assert p["value"] == pytest.approx(max(s.tok_s for s in steps if vl.is_full(s, RULES)))
    assert vl.knee(steps, RULES)["rate_rps"] == 10.0          # comparison only


def test_full_first_step_runs_downward_until_a_bounded_step():
    table = lambda r: (min(r, 5.0) * 700, max(0.0, r - 5.0))  # noqa: E731
    m = _scripted(table)
    vl.run_ladder(m, "m", "b11", 500, 200, [8.0, 9.0], RULES, log=lambda _l: None, down_rates=[7.0, 4.0, 3.0])
    assert m.calls == [8.0, 9.0, 7.0, 4.0]


@pytest.mark.parametrize("kw", [{"max_latency_s": 151.0}, {"sender_late": True}, {"window_samples": 0},
                                {"failed": 10, "arrived": 100}])
def test_an_ineffective_step_is_never_v(kw):
    good = [_step(11.0, 7000, 1.0), _step(12.0, 7100, 2.0)]
    bad = _step(13.0, 9999, 3.0, **kw)
    assert vl.ineffective_why(bad, RULES) is not None
    assert vl.peak(good + [bad], RULES)["value"] == pytest.approx(7100)


def test_warmup_covers_the_saturated_request_lifetime():
    assert vl.step_length(30.0, 110.0, 60.0, 256, 1.5) == 110.0
    # 4 req/s with 256 running -> 64 s lifetime -> >= 96 s warm-up before the 60 s window
    assert vl.step_length(4.0, 110.0, 60.0, 256, 1.5) == pytest.approx(156.0)


def test_anchor_scales_the_fit_to_the_measured_point():
    fit = [7.5e-5, 1.25e-4, -0.005]
    s = vl.anchor_scale(fit, 492, 400, 10.885)
    assert vl.ladder_centre(fit, 492, 400, s) == pytest.approx(10.885)


def test_step_throughput_comes_from_the_engine_counters_in_the_window():
    st = vl.Step("m", "bucket", "b11", 500, 200, 10.0)
    samples = [{"t": float(t), "prompt": 1000.0 * t, "gen": 300.0 * t, "waiting": float(t), "running": 256.0,
                "kv": 0.5, "preempt": 0.0} for t in range(0, 121)]
    recs = [{"k": k, "planned": 50.0 + k, "send": 50.0 + k, "late": 0.0, "first": None, "done": 60.0 + k,
             "end": 60.0 + k, "ok": True, "stream": False, "prompt_tokens": 500, "completion_tokens": 200}
            for k in range(60)]
    vl.fill_step(st, recs, samples, t0=0.0, lo=60.0, hi=120.0, sent={})
    assert st.prefill_tok_s == pytest.approx(1000.0) and st.decode_tok_s == pytest.approx(300.0)
    assert st.arrived == 50 and st.waiting_slope == pytest.approx(1.0) and vl.is_full(st, vl.Rules())
    samples[90]["prompt"] = 0.0  # a counter that falls: the engine restarted
    st2 = vl.Step("m", "bucket", "b11", 500, 200, 10.0)
    vl.fill_step(st2, recs, samples, t0=0.0, lo=60.0, hi=120.0, sent={})
    assert not st2.valid


# ------------------------------------------------------------------ mu (D58)

def _judged_step(rate, nls, p_share_in=500, out=200, prefill=1000.0, decode=400.0, failed=0):
    st = _step(rate, prefill + decode, 0.0, in_tokens=p_share_in, out_tokens=out, window_lo=0.0, window_hi=100.0,
               prefill_tok_s=prefill, decode_tok_s=decode)
    recs = [{"send": 1.0 + k, "done": 1.0 + k + nl * out, "ok": True, "completion_tokens": out}
            for k, nl in enumerate(nls)]
    recs += [{"send": 50.0, "done": None, "ok": False}] * failed
    recs.append({"send": 150.0, "done": 999.0, "ok": True, "completion_tokens": out})  # outside the window
    return st, recs


def test_a_step_passes_only_when_every_request_meets_the_nl_slo():
    slo = 0.040
    st, recs = _judged_step(5.0, [0.02] * 50)
    assert vl.judge_step(st, recs, slo, RULES)["passes"]
    st, recs = _judged_step(5.0, [0.02] * 49 + [0.041])
    assert not vl.judge_step(st, recs, slo, RULES)["passes"]
    st, recs = _judged_step(5.0, [0.02] * 50, failed=1)
    assert not vl.judge_step(st, recs, slo, RULES)["passes"]


def test_mu_takes_each_term_from_the_best_passing_step_within_the_share_range():
    slo = 0.040
    rows = []
    for rate, nl, prefill, decode, tin, out in [(4.0, 0.02, 2000.0, 900.0, 500, 200),   # share 0.71
                                                (6.0, 0.05, 3000.0, 1300.0, 500, 200),  # fails the SLO
                                                (3.0, 0.02, 500.0, 2500.0, 150, 800)]:  # share 0.16
        st, recs = _judged_step(rate, [nl] * 20, p_share_in=tin, out=out, prefill=prefill, decode=decode)
        st.cell = f"c{rate}"
        rows.append(vl.judge_step(st, recs, slo, RULES))
    a = vl.mu_from(rows, lambda s: True)
    b = vl.mu_from(rows, lambda s: 0.37 <= s["p_share"] <= 0.84)
    assert a["p"]["value"] == 2000.0 and a["d"]["value"] == 2500.0      # each term from its own step
    assert b["d"]["value"] == 900.0 and b["t"]["value"] == 2900.0       # the share-0.16 cell is out (B)


def test_isolated_nl_drops_warmup_and_dirty_requests():
    recs = [{"run": -1, "shape": "mix", "ok": True, "in": 10, "out": 10, "prompt_tokens": 10,
             "completion_tokens": 10, "e2e_s": 9.0, "queue_before": 0}]
    recs += [{"run": r, "shape": "mix", "ok": True, "in": 10, "out": 10, "prompt_tokens": 10,
              "completion_tokens": 10, "e2e_s": 0.1, "queue_before": 0} for r in range(3)]
    recs.append({"run": 3, "shape": "mix", "ok": True, "in": 10, "out": 10, "prompt_tokens": 10,
                 "completion_tokens": 10, "e2e_s": 5.0, "queue_before": 2})  # pod was busy
    nl, bad = vl.isolated_nl(recs)
    assert nl["mix"]["mean"] == pytest.approx(0.01) and nl["mix"]["n"] == 3 and bad == 1


def test_real_ladder_needs_approval_and_the_window(tmp_path):
    rep = tmp_path / "rep.yaml"
    rep.write_text(yaml.safe_dump({"bucket_centers": {"m": [[[200, 50], [200, 200], [200, 700]],
                                                            [[500, 50], [500, 200], [500, 700]],
                                                            [[2000, 50], [2000, 200], [2000, 700]]]}}))
    fit = tmp_path / "fit.json"
    fit.write_text(json.dumps({"m": [7.5e-5, 1.25e-4, -0.005]}))
    base = ["ladder", "--models", "m", "--report", str(rep), "--fit", str(fit), "--anchor", "m=492:400:10.885"]
    assert vl.main(base + ["--dry-run"]) == 0
    assert vl.main(base) == 3
    marker = tmp_path / "W"
    assert vl.main(base + ["--i-have-user-approval", "--out-dir", str(tmp_path / "o"), "--sm-url", "http://x",
                           "--exclusive-marker", str(marker)]) == 4
    err, _ = vl.check_marker(str(marker))
    assert err and "missing" in err
    marker.write_text("calibration 2026-01-01T00:00:00+08:00 2099-01-01T00:00:00+08:00 baselines v\n")
    assert vl.check_marker(str(marker), "baselines")[0] is None
    assert vl.check_marker(str(marker), "theta")[0] is not None

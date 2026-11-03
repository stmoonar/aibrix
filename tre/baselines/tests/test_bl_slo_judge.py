"""The unified per-request SLO (tools/slo_judge.py, user 2026-11-04) and PreServe mu judged by it:
one violation of any cause fails a window, a request without a first-token time is refused."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from tre_baselines.tools import slo_judge as sj
from tre_baselines.tools import velocity_ladder as vl
from tre_common.slo_labels import label_def_for_model

#: dsqwen-7b: TTFT_slo(500) = max(500, 5 * (41.1 + 34.7)) = 500 ms; TPOT 75 ms
SLO = label_def_for_model("dsqwen-7b", mode="slowdown", c=41.1, b=0.0694, k=5.0, floor=500.0, tpot_p95_ms=75.0)
REGISTRY = Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml"


def _rec(k, send, ttft_s=0.1, tpot_s=0.03, out=200, L=500, ok=True, stream=True):
    """a ladder raw record"""
    first = send + ttft_s
    done = first + tpot_s * (out - 1)
    return {"k": k, "send": send, "first": first if stream else None, "done": done, "end": done, "ok": ok,
            "stream": stream, "prompt_tokens": L, "completion_tokens": out}


def _req(k, send, **kw):
    return sj.from_ladder_record(_rec(k, send, **kw), "dsqwen-7b")


def _window(**bad):
    reqs = [_req(k, 1.0 + k) for k in range(30)]
    if bad:
        reqs[7] = _req(7, 8.0, **bad)
    reqs.append(_req(99, 500.0, ttft_s=9.0))  # sent outside the window: not judged
    return sj.judge_window(reqs, 0.0, 100.0, SLO)


def test_one_violation_of_any_cause_fails_the_window():
    w = _window()
    assert w["meets_slo"] and w["n"] == 30 and w["violations"] == 0
    w = _window(ttft_s=0.501)                       # one TTFT just above 500 ms
    assert not w["meets_slo"] and w["v_ttft"] == 1 and w["ttft_max_ms"] == pytest.approx(501.0)
    w = _window(tpot_s=0.076)                       # one TPOT just above 75 ms
    assert not w["meets_slo"] and w["v_tpot"] == 1 and w["v_ttft"] == 0
    w = _window(ok=False)
    assert not w["meets_slo"] and w["v_failed"] == 1
    # the threshold scales with the request's own prompt length: 2048 tokens -> 916.2 ms
    w = _window(L=2048, ttft_s=0.9)
    assert w["meets_slo"] and w["ttft_slo_ms_max"] == pytest.approx(916.2)
    # one output token: TPOT undefined, TTFT alone judged
    assert sj.judge_request(_req(0, 0.0, out=1, tpot_s=9.0), SLO)["cause"] is None


def test_a_served_request_without_a_first_token_time_is_refused():
    with pytest.raises(sj.UnjudgeableWindow):
        _window(stream=False)
    # a failed request needs no first token: it is a violation, not a refusal
    assert _window(ok=False, stream=False)["v_failed"] == 1


def test_campaign_client_rows_judge_in_10s_windows_per_model():
    def row(i, model, t, ttft, e2e=2.0, out=100, ok=True):
        return {"request_id": f"r{i}", "model_name": model, "start_time": 1000.0 + t, "ttft": ttft,
                "e2e_latency": e2e, "input_tokens": 300, "output_tokens": out, "success_strict": ok}
    rows = [row(0, "a", 1.0, 0.1), row(1, "b", 2.0, 0.1), row(2, "a", 12.0, 0.6), row(3, "b", 21.0, 0.6),
            row(4, "a", 25.0, 0.1)]
    slos = {"a": SLO, "b": label_def_for_model("b", mode="slowdown", c=150.0, b=0.0, k=5.0, floor=500.0)}
    ws = sj.judge_windows([sj.from_client_row(r) for r in rows], slos, start=1000.0, end=1030.0)
    assert [(w["n"], w["meets_slo"]) for w in ws] == [(2, True), (1, False), (2, True)]  # b: 600 ms <= 5 x 150
    assert ws[1]["v_ttft"] == 1
    with pytest.raises(sj.UnjudgeableWindow):
        sj.from_client_row({**rows[0], "success_strict": None})        # completion unknown: refused


def _ladder(tmp_path, stream=True):
    st = vl.Step("dsqwen-7b", "bucket", "b11", 500, 200, 5.0, window_lo=0.0, window_hi=100.0, arrived=30,
                 completed=30, prefill_tok_s=2500.0, decode_tok_s=1000.0, window_samples=60, max_latency_s=10.0)
    (tmp_path / "raw").mkdir()
    (tmp_path / "profile-dsqwen-7b.log.jsonl").write_text(json.dumps({"event": "step", **asdict(st)}) + "\n")
    recs = [_rec(k, 1.0 + k, stream=stream or k != 3) for k in range(30)]
    vl.raw_log_path(tmp_path / "raw", st.model, st.cell, st.rate_rps).write_text(
        "".join(json.dumps(r) + "\n" for r in recs))
    rep = tmp_path / "rep.yaml"
    rep.write_text(yaml.safe_dump({"bucket_centers": {}, "p_share": {"pooled": {"p5": 0.29, "p95": 0.84}}}))
    return ["mu", "--ladder-dir", str(tmp_path), "--report", str(rep), "--models", "dsqwen-7b",
            "--min-completed", "1", "--out", str(tmp_path / "mu.yaml")]


def test_mu_unified_takes_the_registry_slo_and_refuses_unstreamed_steps(tmp_path):
    base = _ladder(tmp_path)
    assert vl.main(base + ["--slo-rule", "unified"]) == 2            # no --registry
    assert vl.main(base + ["--slo-rule", "unified", "--registry", str(REGISTRY), "--slo-mode", "slowdown"]) == 0
    doc = yaml.safe_load((tmp_path / "mu.yaml").read_text())
    reg = yaml.safe_load(REGISTRY.read_text())
    want = next(m["slo"] for m in reg["models"] if m["name"] == "dsqwen-7b")
    got = doc["models"]["dsqwen-7b"]["slo"]
    assert (got["mode"], got["registry_slo_mode"]) == ("slowdown", reg["slo_mode"])
    assert (got["c_ms"], got["b_ms_per_token"], got["tpot_slo_ms"]) == (
        want["ttft_idle_c_ms"], want["ttft_idle_b_ms_per_token"], want["tpot_p95_ms"])
    assert doc["sources"]["registry_sha256"] and doc["mu_B"]["dsqwen-7b"]["t"] == 3500.0
    assert doc["models"]["dsqwen-7b"]["cells_B"]["b11"]["bracketed_below"]


def test_mu_unified_refuses_a_ladder_that_did_not_stream_every_request(tmp_path):
    base = _ladder(tmp_path, stream=False)
    assert vl.main(base + ["--slo-rule", "unified", "--registry", str(REGISTRY)]) == 2


def test_nl_rule_one_slow_or_failed_request_fails_the_window():
    """PreServe's NL rule: NL = e2e / output tokens; 30 requests at 0.1 + 0.03 * 199 s / 200 = 30.35 ms."""
    def win(**bad):
        reqs = [_req(k, 1.0 + k) for k in range(30)]
        if bad:
            reqs[7] = _req(7, 8.0, **bad)
        return sj.judge_window_nl(reqs, 0.0, 100.0, {"dsqwen-7b": 40.0})
    assert win()["meets_slo"] and win()["nl_max_ms"] == pytest.approx(30.35)
    w = win(tpot_s=0.05)                              # NL 50.25 ms > 40: one slow request
    assert not w["meets_slo"] and w["v_nl"] == 1 and w["v_failed"] == 0
    w = win(ok=False)
    assert not w["meets_slo"] and w["v_failed"] == 1
    # TTFT 600 ms alone breaks the unified rule, not NL (spread over 200 tokens: 32.85 ms)
    assert win(ttft_s=0.6)["meets_slo"] and not _window(ttft_s=0.6)["meets_slo"]
    with pytest.raises(sj.UnjudgeableWindow):         # no NL SLO for the model
        sj.judge_window_nl([_req(0, 1.0)], 0.0, 10.0, {"other": 40.0})


def test_dataset_row_adapter_reads_ms_and_failed_rows():
    ok = sj.from_dataset_row({"model": "m", "request_id": "a", "send_ts_ms": "1000", "first_token_ts_ms": "1100",
                              "done_ts_ms": "3000", "input_tokens": "512", "output_tokens": "100", "outcome": "ok"})
    assert (ok.send, ok.done, ok.prompt_tokens, ok.completion_tokens, ok.ok) == (1.0, 3.0, 512, 100, True)
    assert sj.judge_request_nl(ok, 20.0)["nl_ms"] == pytest.approx(20.0)
    bad = sj.from_dataset_row({"model": "m", "send_ts_ms": "1000", "first_token_ts_ms": "", "done_ts_ms": "31000",
                               "input_tokens": "", "expected_prompt_tokens": "768", "output_tokens": "",
                               "outcome": "client_timeout"})
    assert bad.prompt_tokens == 768 and not bad.ok and sj.judge_request_nl(bad, 20.0)["cause"] == "failed"

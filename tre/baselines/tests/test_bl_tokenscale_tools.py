from __future__ import annotations

import json

import pytest
import yaml

from tre_baselines.tools import tokenscale_buckets as tb
from tre_baselines.tools import tokenscale_profile as tp


def test_buckets_on_synthetic_records(tmp_path, capsys):
    recs = [{"model": "m", "in_tokens": i, "max_tokens": o} for i in range(1, 10) for o in (10, 20, 30)]
    f = tmp_path / "t.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    assert tb.main(["--trace", str(f), "--model", "m"]) == 0
    out = yaml.safe_load(capsys.readouterr().out)
    assert out["bucket_edges"]["m"] == {"in": [3, 6], "out": [10, 20]}
    c = out["bucket_centers"]["m"]
    assert c[0][0] == [2, 10] and c[2][2] == [8, 30]


def test_buckets_on_segment_trace_and_dist(tmp_path, capsys):
    trace = {"m": [
        {"start_time": 0, "end_time": 10, "rps": 1, "input_tokens": 100, "max_tokens": 10},
        {"start_time": 10, "end_time": 20, "rps": 1, "input_tokens": 500, "max_tokens": 100},
        {"start_time": 20, "end_time": 30, "rps": 1, "input_tokens_dist": {"low": 1000, "high": 4000},
         "max_tokens_dist": {"low": 100, "high": 400}},
    ]}
    f = tmp_path / "t.json"
    f.write_text(json.dumps(trace), encoding="utf-8")
    assert tb.main(["--trace", str(f)]) == 0
    out = yaml.safe_load(capsys.readouterr().out)
    assert out["bucket_edges"]["m"]["in"] == [100, 500]
    assert out["bucket_centers"]["m"][2][2] == [2000, 200]


def test_buckets_bad_format(tmp_path, capsys):
    f = tmp_path / "bad.json"
    f.write_text(json.dumps([{"foo": 1}]), encoding="utf-8")
    assert tb.main(["--trace", str(f)]) == 2
    assert "needs one of" in capsys.readouterr().err
    f.write_text("not json\n{", encoding="utf-8")
    assert tb.main(["--trace", str(f)]) == 2


SLO = __import__("types").SimpleNamespace(ttft_slo_ms=lambda L: 500.0, tpot_p95_ms=75.0)


def _scripted(tok_s_by_c, ttft=lambda c: 100.0, tpot=lambda c: 20.0):
    calls = []

    def measure(model, tin, tout, c, step_s, warmup_s):
        calls.append(c)
        v = tok_s_by_c(c)
        share = tin / (tin + tout)
        return tp.StepMeasure(c, step_s - warmup_s, 10, v * share, v * (1 - share), ttft(c), tpot(c))

    measure.calls = calls
    return measure


def test_full_ladder_and_extension_while_still_gaining():
    m = _scripted(lambda c: 1000 * min(c, 96))  # still +50 % at 64 -> extend 96, then flat at 144
    steps = tp.run_ladder(m, "m", 492, 400, 60, 15)
    assert m.calls == [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 144]
    assert max(s.tok_s for s in steps) == pytest.approx(96_000)
    warnings = []
    m = _scripted(lambda c: 1000 * c)          # never saturates: warn at the cap
    tp.run_ladder(m, "m", 492, 400, 60, 15, (1, 2), max_concurrency=4, warn=warnings.append)
    assert m.calls == [1, 2, 3] and "still" in warnings[0]


def test_mu_is_the_highest_step_meeting_both_slos_and_v_is_the_peak():
    m = _scripted(lambda c: 1000 * c, ttft=lambda c: 60.0 * c, tpot=lambda c: 5.0 * c)
    res, rows = tp.profile_model(m, "m", 492, 400, SLO, step_s=60, warmup_s=15, ladder=(1, 2, 4, 8, 16),
                                 extend_gain=10.0)
    # ttft <= 500 -> c <= 8; tpot <= 75 -> c <= 15: mu from c = 8 (8000 tok/s split 492:400)
    assert res["mu"] == {"p": round(8000 * 492 / 892, 1), "d": round(8000 * 400 / 892, 1), "t": 8000.0,
                         "concurrency": 8}
    assert res["velocity"]["buckets"] == [[16000.0] * 3] * 3       # one shape -> all nine cells
    assert res["velocity"]["v_prefill"] == pytest.approx(16000 * 492 / 493, abs=0.1)
    assert {r["kind"] for r in rows} == {"mixed", "prefill"}
    m = _scripted(lambda c: 1000 * c, ttft=lambda c: 900.0)
    assert tp.profile_model(m, "m", 492, 400, SLO, step_s=60, warmup_s=15, ladder=(1, 2),
                            extend_gain=10.0)[0]["mu"] is None


def test_engine_rates_from_counters_and_reset():
    a = 'vllm:prompt_tokens_total{engine="0"} 100\nvllm:generation_tokens_total{engine="0"} 50\n'
    b = 'vllm:prompt_tokens_total{engine="0"} 1100\nvllm:generation_tokens_total{engine="0"} 850\n'
    assert tp.engine_rates(a, b, 10.0) == (100.0, 80.0)
    assert tp.engine_rates(b, a, 10.0) == (None, None)               # engine restarted
    assert tp.engine_rates("", b, 10.0) == (None, None)


def test_dry_run_writes_velocity_and_mu_for_every_model(tmp_path, capsys):
    out = tmp_path / "o"
    rc = tp.main(["--models", "a,b", "--out-dir", str(out), "--dry-run"], slo_for=lambda m: SLO)
    assert rc == 0
    doc = yaml.safe_load((out / "profile.yaml").read_text(encoding="utf-8"))
    assert set(doc["velocity"]) == set(doc["mu"]) == {"a", "b"} and doc["shape"]["in_tokens"] == 492
    assert doc["mu"]["a"]["t"] > 0 and len(doc["velocity"]["a"]["buckets"]) == 3
    assert (out / "profile_raw.csv").read_text(encoding="utf-8").startswith("model,kind")


def test_real_run_needs_approval_and_endpoints(tmp_path, capsys):
    rc = tp.main(["--models", "a", "--out-dir", str(tmp_path / "o")], slo_for=lambda m: SLO)
    assert rc == 3 and "--i-have-user-approval" in capsys.readouterr().err
    rc = tp.main(["--models", "a", "--out-dir", str(tmp_path / "o"), "--i-have-user-approval"], slo_for=lambda m: SLO)
    assert rc == 2 and not (tmp_path / "o").exists()


class _FakeVllm:
    """Chat endpoint streaming like vLLM 0.30 plus its /metrics token counters, which grow
    with every request served."""

    def __init__(self, delay_s=0.02):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.bodies, self.prompt, self.gen, self.lock, fake = [], 0, 0, threading.Lock(), self
        self.running, self.stuck = 0, 0  # stuck: extra requests the engine reports forever

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                body = (f"vllm:prompt_tokens_total {fake.prompt}\n"
                        f"vllm:generation_tokens_total {fake.gen}\n"
                        f"vllm:num_requests_running {fake.running + fake.stuck}\n"
                        f"vllm:num_requests_waiting 0\n").encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                fake.bodies.append(body)
                import time as _t
                with fake.lock:
                    fake.running += 1
                _t.sleep(delay_s)
                n = int(body["max_tokens"])
                with fake.lock:
                    fake.prompt += 120
                    fake.gen += n
                    fake.running -= 1
                chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}]
                chunks += [{"choices": [{"index": 0, "delta": {"content": "x"}}]} for _ in range(n)]
                chunks.append({"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": n,
                                                        "total_tokens": 120 + n}})
                raw = b"".join(b"data: " + json.dumps(o).encode() + b"\n\n" for o in chunks) + b"data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.url, self.metrics = base + "/v1/chat/completions", base + "/metrics"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def test_http_measure_drives_closed_loop_chat_and_reads_engine_counters(tmp_path, monkeypatch):
    from scripts import r3_grid

    # the prompt text is not under test (natural prompts need the model tokenizer)
    monkeypatch.setattr(r3_grid, "_make_prompt", lambda n, key, *a, **k: f"prompt {key}")
    fake = _FakeVllm()
    try:
        measure = tp.make_http_measure(fake.url, lambda m: [fake.metrics], raw_dir=tmp_path / "raw", drain_poll_s=0.05)
        step = measure("dsqwen-7b", 128, 4, 2, 0.8, 0.2)
    finally:
        fake.close()
    assert fake.bodies, "nothing was sent"
    prompts = [b["messages"][0]["content"] for b in fake.bodies]
    assert len(set(prompts)) == len(prompts)                     # distinct prompts
    for body in fake.bodies:  # the calibration chat sender's fixed-length request
        assert body["model"] == "dsqwen-7b" and body["ignore_eos"] is True and body["temperature"] == 0
        assert body["max_tokens"] == 4 and body["stream"] is True
    assert 0.5 <= step.window_s <= 0.7 and step.completed > 0
    # engine counters, not completed requests: 120 prompt + 4 generated tokens per request
    assert step.prefill_tok_s > 0 and step.decode_tok_s == pytest.approx(step.prefill_tok_s * 4 / 120, rel=0.35)
    assert step.ttft_p95_ms is not None and step.tpot_p95_ms is not None


def test_buckets_report_the_median_input(tmp_path, capsys):
    f = tmp_path / "t.json"
    f.write_text(json.dumps([{"in_tokens": x, "max_tokens": 10} for x in (100, 200, 300, 400, 500)]), encoding="utf-8")
    assert tb.main(["--trace", str(f), "--default-model", "m"]) == 0
    assert yaml.safe_load(capsys.readouterr().out)["median_in"] == {"m": 300}


def _http_measure(monkeypatch, tmp_path, fake, **kw):
    from scripts import r3_grid

    monkeypatch.setattr(r3_grid, "_make_prompt", lambda n, key, *a, **k: f"prompt {key}")
    return tp.make_http_measure(fake.url, lambda m: [fake.metrics], raw_dir=tmp_path / "raw", **kw)


def test_a_fleet_change_during_a_step_invalidates_it_and_aborts(tmp_path, monkeypatch):
    """Review P1-A: a wake/sleep/restart during a step must not yield a velocity or mu."""
    fake = _FakeVllm()
    views = iter([("b0",), ("b0", "b1")])                    # mark a, mark b: another replica woke
    try:
        measure = _http_measure(monkeypatch, tmp_path, fake, fleet_view=lambda m: next(views),
                                expected={"m": ("b0",)}, drain_poll_s=0.05)
        step = measure("m", 64, 4, 1, 0.6, 0.2)
    finally:
        fake.close()
    assert step.valid is False and step.prefill_tok_s is None and "fleet changed" in step.note
    stub = _scripted(lambda c: 1000 * c)
    calls = []

    def flaky(model, tin, tout, c, step_s, warmup_s):
        calls.append(c)
        st = stub(model, tin, tout, c, step_s, warmup_s)
        return st if c < 4 else tp.StepMeasure(c, 1, 0, None, None, None, None, valid=False, note="fleet changed")

    res, rows = tp.profile_model(flaky, "m", 492, 400, SLO, step_s=60, warmup_s=15, ladder=(1, 2, 4, 8))
    assert calls == [1, 2, 4] and res["aborted"] == "fleet changed"     # no out=1 ladder, nothing more sent
    assert res["velocity"] is None and res["mu"] is None


def test_each_step_waits_for_an_idle_replica_before_returning(tmp_path, monkeypatch):
    """Review P2-B: a state gate on running + waiting == 0 (bounded far above any request)."""
    fake = _FakeVllm(delay_s=0.3)
    try:
        _http_measure(monkeypatch, tmp_path, fake, drain_poll_s=0.05)("m", 64, 4, 3, 0.5, 0.1)
        assert fake.running == 0                              # stragglers finished before the return
        fake.stuck = 1
        with pytest.raises(tp.ProfileAbort, match="in flight"):
            _http_measure(monkeypatch, tmp_path, fake, drain_poll_s=0.05, drain_cap_s=0.3)("m", 64, 4, 1, 0.3, 0.1)
    finally:
        fake.close()


def test_preconditions_need_observe_modes_and_no_apa():
    assert tp.check_preconditions(["a"], lambda k: None, lambda: []) == []
    modes = {"tre:v2:controller:mode": "active", "tre:v2:sm:actuation": "active"}
    apa = [{"metadata": {"namespace": "default", "name": "a-apa"}, "spec": {"scaleTargetRef": {"name": "a"}}}]
    problems = tp.check_preconditions(["a"], modes.get, lambda: apa)
    assert len(problems) == 3 and any("APA" in p for p in problems)


def test_profile_records_the_registry_and_c_b_used(tmp_path):
    slo = __import__("types").SimpleNamespace(ttft_slo_ms=lambda L: 500.0, tpot_p95_ms=75.0, ttft_idle_c_ms=41.1,
                                              ttft_idle_b_ms_per_token=0.0694, ttft_slowdown_k=5.0,
                                              ttft_floor_ms=500.0)
    out = tmp_path / "o"
    assert tp.main(["--models", "a", "--out-dir", str(out), "--dry-run", "--registry", "r.yaml"],
                   slo_for=lambda m: slo) == 0
    doc = yaml.safe_load((out / "profile.yaml").read_text(encoding="utf-8"))
    assert doc["registry"] == "file:r.yaml"
    assert doc["slo"]["a"]["ttft_idle_c_ms"] == 41.1 and doc["slo"]["a"]["ttft_idle_b_ms_per_token"] == 0.0694


# ------------------------------------------------------------------ open loop (PreServe mu)


def test_open_loop_measure_sends_poisson_chat_and_reads_engine_counters(tmp_path, monkeypatch):
    """--open-loop: the calibration open-loop sender against a fake vLLM; rates from the
    engine counters of the measured window, latencies of the requests sent in it."""
    from tre_replayer.engine import http_sender

    monkeypatch.setattr(http_sender, "build_prompt", lambda n, key, *a, **k: f"prompt {key}")
    fake = _FakeVllm()
    try:
        measure = tp.make_openloop_measure(fake.url, lambda m: [fake.metrics], raw_dir=tmp_path / "raw",
                                           sender_processes=1, prompt_dir_enabled=False, sample_s=0.1,
                                           drain_poll_s=0.05)
        step = measure("dsqwen-7b", 128, 4, 40.0, 1.5, 0.3)
    finally:
        fake.close()
    assert len(fake.bodies) >= 20, "the Poisson schedule was not sent"
    prompts = [b["messages"][0]["content"] for b in fake.bodies]
    assert len(set(prompts)) == len(prompts)                     # distinct prompts
    for body in fake.bodies:
        assert body["model"] == "dsqwen-7b" and body["ignore_eos"] is True and body["temperature"] == 0
        assert body["max_tokens"] == 4 and body["stream"] is True
    assert step.valid and step.rate_rps == 40.0 and step.arrived > 0 and step.failed == 0
    assert step.completed == step.arrived < len(fake.bodies)      # only the steady-state window
    assert step.achieved_rps == pytest.approx(step.arrived / 1.2, rel=1e-3)
    assert step.prefill_tok_s > 0 and step.decode_tok_s == pytest.approx(step.prefill_tok_s * 4 / 120, rel=0.35)
    assert step.ttft_p95_ms is not None and step.tpot_p95_ms is not None
    assert fake.running == 0                                       # drained before the return


def test_open_loop_failures_count_against_the_step_and_mu_is_the_highest_passing_rate():
    recs = [{"send_ts_ms": 10, "http_status": 200, "ttft_ms": 100.0, "tpot_ms": 20.0, "done_ts_ms": 50},
            {"send_ts_ms": 20, "http_status": 504, "ttft_ms": None, "tpot_ms": None, "done_ts_ms": None},
            {"send_ts_ms": 5, "http_status": 200, "ttft_ms": 1.0, "tpot_ms": 1.0, "done_ts_ms": 9}]  # warm-up
    arrived, completed, ttft, tpot = tp.steady_latencies(recs, 10, 30)
    assert (arrived, completed) == (2, 1) and ttft == float("inf") and tpot == float("inf")

    def step(rate, ttft, n=200):
        return tp.StepMeasure(0, 90, n, rate * 492, rate * 400, ttft, 30.0, rate_rps=rate, arrived=n, failed=0)

    steps = [step(1.0, 200.0), step(2.0, 700.0), step(3.0, 400.0), step(4.0, 900.0), step(5.0, 100.0, n=10)]
    mu = tp.preserve_mu(steps, 500.0, 75.0, min_completed=150)
    assert mu["rate_rps"] == 3.0 and mu["t"] == pytest.approx(3.0 * 892)  # rate 5 has too few requests
    assert tp.non_monotone(steps, 500.0, 75.0, min_completed=150) == [2.0]


def test_open_loop_dry_run_writes_mu_without_velocity(tmp_path, capsys):
    out = tmp_path / "o"
    assert tp.main(["--models", "a", "--out-dir", str(out), "--dry-run", "--open-loop", "--base-rps", "a=10"],
                   slo_for=lambda m: SLO) == 0
    doc = yaml.safe_load((out / "profile.yaml").read_text(encoding="utf-8"))
    assert doc["mode"] == "open_loop" and "velocity" not in doc and doc["mu"]["a"]["rate_rps"] > 0
    assert tp.main(["--models", "a", "--out-dir", str(out), "--dry-run", "--open-loop"], slo_for=lambda m: SLO) == 2

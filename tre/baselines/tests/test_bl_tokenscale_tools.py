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


def _flat_sender(rates):
    """Sender whose tok/s at the n-th ladder step follows ``rates``; records the calls."""
    calls = []

    def send(model, tin, tout, c, step_s, warmup_s):
        calls.append(c)
        w = step_s - warmup_s
        return int(rates[len(calls) - 1] * w / (tin + tout)), w

    send.calls = calls
    return send


def test_full_ladder_and_peak():
    # no early stop: every concurrency of the decided ladder runs; V_b is the peak
    s = _flat_sender([1000, 2000, 3000, 3060, 2500, 2400])
    v, steps = tp.profile_point(s, "m", 10, 10, 60, 15)
    assert s.calls == [1, 2, 4, 8, 16, 32] and len(steps) == 6
    assert v == pytest.approx(3060, rel=0.01)
    s = _flat_sender([1000] * 3)
    tp.profile_point(s, "m", 10, 10, 60, 15, ladder=(1, 3, 9))
    assert s.calls == [1, 3, 9]


def _centers_file(tmp_path):
    centers = [[[100 * (i + 1), 10 * (j + 1)] for j in range(3)] for i in range(3)]
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump({"bucket_centers": {"m": centers}}), encoding="utf-8")
    return f


def test_profile_dry_run_writes_velocity_and_csv(tmp_path, capsys):
    out = tmp_path / "o"
    assert tp.main(["--model", "m", "--centers", str(_centers_file(tmp_path)), "--out-dir", str(out), "--dry-run"]) == 0
    assert "estimated GPU time" in capsys.readouterr().out
    vel = yaml.safe_load((out / "velocity.yaml").read_text(encoding="utf-8"))["velocity"]["m"]
    assert len(vel["buckets"]) == 3 and all(len(r) == 3 and all(v > 0 for v in r) for r in vel["buckets"])
    assert 9000 < vel["buckets"][0][0] <= 10000 and vel["v_prefill"] > 20000
    rows = (out / "profile_raw.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("model,kind,bucket") and len(rows) > 10


def test_profile_refuses_real_send_without_approval(tmp_path, capsys):
    out = tmp_path / "o"
    rc = tp.main(["--model", "m", "--centers", str(_centers_file(tmp_path)), "--out-dir", str(out), "--sender", "stub"])
    assert rc == 3 and not out.exists()
    assert "--i-have-user-approval" in capsys.readouterr().err


def test_profile_requires_out_dir():
    with pytest.raises(SystemExit):
        tp.main(["--model", "m", "--centers", "x", "--dry-run"])


def test_load_sender_spec():
    assert callable(tp.load_sender("stub"))
    with pytest.raises(ValueError):
        tp.load_sender("nocolon")
    with pytest.raises(ValueError, match="gateway-url"):
        tp.load_sender("http")


class _FakeVllm:
    """A chat endpoint that streams like vLLM 0.30 (role chunk, content chunks, usage chunk)
    and checks each body is the fixed-length generation request."""

    def __init__(self, delay_s=0.02):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.bodies, fake = [], self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                fake.bodies.append(body)
                import time as _t
                _t.sleep(delay_s)
                n = int(body["max_tokens"])
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
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1/chat/completions"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def test_http_sender_drives_closed_loop_chat_against_a_fake_server(tmp_path, monkeypatch):
    from scripts import r3_grid

    # the prompt text is not under test (natural prompts need the model tokenizer)
    monkeypatch.setattr(r3_grid, "_make_prompt", lambda n, key, *a, **k: f"prompt {key}")
    fake = _FakeVllm()
    try:
        send = tp.make_http_sender(fake.url, raw_dir=tmp_path / "raw")
        completed, window, tin, tout = send("dsqwen-7b", 128, 4, 2, 0.6, 0.1)
    finally:
        fake.close()
    assert fake.bodies, "nothing was sent"
    for body in fake.bodies:  # the calibration chat sender's fixed-length request
        assert body["model"] == "dsqwen-7b" and body["messages"][0]["role"] == "user"
        assert body["ignore_eos"] is True and body["temperature"] == 0 and body["max_tokens"] == 4
        assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert 0.4 <= window <= 0.7 and 0 < completed <= len(fake.bodies)
    assert (tin, tout) == (120 * completed, 4 * completed)  # usage counts, not nominal lengths
    assert list((tmp_path / "raw").glob("dsqwen-7b_i128_o4_c2.jsonl"))


def test_measure_window_counts_only_ok_requests_done_after_warmup():
    recs = [
        {"http_status": 200, "done_ts_ms": 1_500, "input_tokens": 10, "output_tokens": 5},   # warm-up
        {"http_status": 200, "done_ts_ms": 3_000, "input_tokens": 10, "output_tokens": 5},
        {"http_status": 200, "done_ts_ms": 4_000, "input_tokens": None, "output_tokens": None},
        {"http_status": 500, "done_ts_ms": 4_000, "input_tokens": 10, "output_tokens": 5},
        {"http_status": 200, "done_ts_ms": 4_100, "stream_error": "cut", "input_tokens": 1, "output_tokens": 1},
        {"http_status": 200, "done_ts_ms": 9_000, "input_tokens": 10, "output_tokens": 5},   # after the step
    ]
    assert tp.measure_window(recs, 1_000, 5_000, 1.0, 12, 6) == (2, 3.0, 22, 11)


def test_buckets_report_the_median_input(tmp_path, capsys):
    f = tmp_path / "t.json"
    f.write_text(json.dumps([{"in_tokens": x, "max_tokens": 10} for x in (100, 200, 300, 400, 500)]), encoding="utf-8")
    assert tb.main(["--trace", str(f), "--default-model", "m"]) == 0
    assert yaml.safe_load(capsys.readouterr().out)["median_in"] == {"m": 300}

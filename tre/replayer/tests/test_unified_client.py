"""The unified sending core (2026-09-30): one SSE parser with two metric bases, the pooled
async transport, the profiles, the multi-process runner and its stop gate.

Everything runs against in-process stubs or a local HTTP server bound to 127.0.0.1.
"""
from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tre_replayer.engine import metrics, profiles
from tre_replayer.engine.api import build_request_headers, request_body
from tre_replayer.engine.http_sender import (
    E1_EXTRA_FIELDS,
    V1_AUDIT_FIELDS,
    V1_RECORD_FIELDS,
    StreamingHttpSender,
    e1_record,
)
from tre_replayer.engine.procpool import ProcessPoolRunner, StopGate
from tre_replayer.engine.schedule import ScheduledRequest
from tre_replayer.engine.stream import StreamParser, StreamResult, stream_request
from tre_replayer.engine.transport import HttpxStreamTransport

CHAT = "/v1/chat/completions"


def _sse(*objs) -> bytes:
    return b"".join(b"data: " + (o if isinstance(o, bytes) else json.dumps(o).encode()) + b"\n\n" for o in objs)


ROLE = {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}
USAGE = {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}


def _tok(text, finish=None):
    return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}]}


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


# ----------------------------------------------------------------------- the parser


def test_the_parser_keeps_the_v1_and_the_strict_view_of_one_stream() -> None:
    clock = _Clock()
    parser = StreamParser(0.0, clock=clock)
    for t, line in [(0.010, ROLE), (0.030, _tok("a")), (0.050, _tok("b")), (0.070, _tok("c", "length")),
                    (0.071, USAGE)]:
        clock.t = t
        parser.feed(b"data: " + json.dumps(line).encode())
    clock.t = 0.072
    parser.feed(b"data: [DONE]")
    clock.t = 0.080
    parser.feed(b"data: " + json.dumps(_tok("after done")).encode())  # ignored by both views
    end = parser.finish()
    assert parser.v1_first_token_ms == pytest.approx(10.0)  # the role-only chunk (content "")
    assert parser.first_token_ms == pytest.approx(30.0) and parser.first_token_field == "content"
    assert parser.done_seen and parser.done_ms == pytest.approx(72.0) and end == pytest.approx(80.0)
    assert (parser.completion_tokens, parser.v1_completion_tokens, parser.v1_total_tokens) == (3, 3, 10)
    assert parser.finish_reason == parser.v1_finish_reason == "length"


def test_an_error_chunk_ends_the_v1_view_and_marks_the_strict_one() -> None:
    clock = _Clock()
    parser = StreamParser(0.0, clock=clock)
    clock.t = 0.01
    parser.feed(b"data: " + json.dumps(_tok("a")).encode())
    clock.t = 0.02
    parser.feed(b'data: {"error": {"message": "engine died", "type": "server_error"}}')
    clock.t = 0.03
    parser.feed(b"data: " + json.dumps(USAGE).encode())
    assert parser.v1_error == "APIError: engine died" and parser.v1_stop_ms == pytest.approx(20.0)
    assert parser.v1_completion_tokens == 0  # the SDK never yielded the usage chunk
    assert parser.stream_error == "engine died" and parser.completion_tokens == 3


def test_undecodable_json_stops_the_v1_view_only_and_comments_carry_the_continuation() -> None:
    parser = StreamParser(0.0, clock=_Clock())
    parser.feed(b": x-tre-continued: 3")
    parser.feed(b"data: {not json")
    parser.feed(b"data: " + json.dumps(_tok("a")).encode())
    assert parser.continued == 3
    assert parser.v1_error.startswith("JSONDecodeError") and parser.v1_first_token_ms is None
    assert parser.first_token_ms is not None


def test_feed_block_splits_lines_across_network_reads() -> None:
    parser = StreamParser(0.0, clock=_Clock())
    raw = _sse(ROLE, _tok("a"), USAGE) + b"data: [DONE]\n\n"
    tail = b""
    for i in range(0, len(raw), 7):
        tail = parser.feed_block(tail, raw[i:i + 7])
    parser.finish(tail)
    assert parser.done_seen and parser.first_token_field == "content" and parser.completion_tokens == 3


# ------------------------------------------------------------------ the two bases


def _res(**kw) -> StreamResult:
    base = dict(status=200, first_token_ms=30.0, done_ms=130.0, prompt_tokens=7, completion_tokens=11,
                done_seen=True, finish_reason="length", start_epoch_s=1000.0, v1_status=200, v1_success=True,
                v1_first_token_ms=10.0, v1_done_ms=131.0, v1_prompt_tokens=7, v1_completion_tokens=11,
                v1_total_tokens=18, v1_finish_reason="length", v1_target_pod="p")
    base.update(kw)
    return StreamResult(**base)


def test_strict_tpot_is_the_label_formula_and_v1_tpot_is_v1s() -> None:
    strict = metrics.strict_view_ms(_res())
    assert strict["success"] and strict["tpot_ms"] == pytest.approx((130.0 - 30.0) / 10)
    v1 = metrics.v1_view_s(_res())
    assert v1["ttft_s"] == pytest.approx(0.010) and v1["tpot_s"] == pytest.approx((0.131 - 0.010) / 11)


@pytest.mark.parametrize("kw,failure", [
    (dict(status=503), metrics.STRICT_FAILURE_HTTP),
    (dict(status=0, timed_out=True), metrics.STRICT_FAILURE_TIMEOUT),
    (dict(status=0), metrics.STRICT_FAILURE_TRANSPORT),
    (dict(stream_error="boom"), metrics.STRICT_FAILURE_STREAM_ERROR),
    (dict(done_seen=False, finish_reason=None), metrics.STRICT_FAILURE_INCOMPLETE),
    (dict(completion_tokens=0), metrics.STRICT_FAILURE_ZERO_OUTPUT),
])
def test_what_v1_counted_as_success_the_strict_basis_fails(kw, failure) -> None:
    assert metrics.strict_failure(_res(**kw)) == failure


def test_retries_are_kept_out_of_the_strict_basis() -> None:
    res = _res(attempts=3, last_attempt_offset_ms=1500.0, first_token_ms=1530.0, done_ms=1630.0)
    strict = metrics.strict_view_ms(res)
    assert strict["retries"] == 2 and strict["retry_wait_ms"] == 1500.0
    assert strict["ttft_ms"] == pytest.approx(30.0) and strict["e2e_ms"] == pytest.approx(130.0)


def test_a_success_without_a_first_token_is_a_strict_violation() -> None:
    strict = metrics.strict_view_ms(_res(first_token_ms=None))
    assert strict["success"] and strict["ttft_ms"] is None and strict["ttft_missing"]


def test_the_e1_record_is_v1s_fields_then_audit_then_strict() -> None:
    request = ScheduledRequest("req_1", "m", 1.5, prompt="p")
    lateness = {"on_wire_delay_ms": 0.4, "schedule_delay_ms": 0.1}
    record = e1_record(request, _res(), process_id=3, lateness=lateness, in_flight_at_send=2)
    assert tuple(record) == V1_RECORD_FIELDS + V1_AUDIT_FIELDS + E1_EXTRA_FIELDS
    assert record["timestamp"] == 1.5 and record["start_time"] == 1000.0
    assert record["end_time"] == pytest.approx(1000.131) and record["ttft"] == pytest.approx(0.010)
    assert record["success"] and record["success_strict"] and record["ttft_strict_s"] == pytest.approx(0.030)
    assert record["send_lateness_ms"] == 0.4 and record["process_id"] == 3


# ------------------------------------------------------------------ profiles


def test_three_profiles_and_their_provenance() -> None:
    assert set(profiles.PROFILES) == {"calib", "replay", "e1_v1"}
    assert profiles.profile_for_api("chat") == "calib" and profiles.profile_for_api("completions") == "replay"
    e1 = profiles.get_profile("e1_v1")
    assert e1.transport == "openai_sdk" and not e1.ignore_eos
    calib = profiles.get_profile("calib")
    assert calib.transport == "httpx" and calib.ignore_eos and calib.retries == "none"
    prov = profiles.client_provenance("calib", transport=HttpxStreamTransport(max_connections=64), processes=4)
    assert prov["processes"] == 4 and prov["wire"]["pool_max_connections"] == 64
    with pytest.raises(ValueError):
        profiles.get_profile("nope")


def test_v1_kwargs_follow_v1s_precedence() -> None:
    options = profiles.V1ChatOptions(model_params={"m": {"max_tokens": 400, "temperature": None}})
    kw = options.kwargs_for("m", "hello", None)
    assert kw == {"model": "m", "messages": [{"role": "user", "content": "hello"}], "temperature": None,
                  "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 400}
    assert options.kwargs_for("m", "hello", 12)["max_tokens"] == 12
    assert "max_tokens" not in profiles.V1ChatOptions(model_params={"m": {}}).kwargs_for("m", "x", None)


# ------------------------------------------------------------------ a local server


class _Server:
    """Threaded HTTP/1.1 keep-alive server: records every request (body, headers, client
    port) and answers per ``model``: ``ok`` streams, ``slow`` waits first, ``shed`` is an
    Envoy-style 503 overflow, ``cut`` drops the stream mid-body."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = json.loads(raw or b"{}")
                with outer.lock:
                    outer.requests.append({"t": time.time(), "raw": raw, "body": body,
                                           "headers": {k.lower(): v for k, v in self.headers.items()},
                                           "port": self.client_address[1]})
                model = body.get("model", "ok")
                if model == "shed":
                    text = b"upstream connect error or disconnect/reset before headers. reset reason: overflow"
                    self.send_response(503)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(text)))
                    self.end_headers()
                    self.wfile.write(text)
                    return
                if model == "slow":
                    time.sleep(0.4)
                payload = _sse(ROLE, _tok("a"), _tok("b", "length"), USAGE) + b"data: [DONE]\n\n"
                if model == "cut":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(payload) + 100))
                    self.end_headers()
                    self.wfile.write(payload[:40])
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("target-pod", "pod-a")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()


@pytest.fixture
def server():
    srv = _Server()
    yield srv
    srv.close()


def test_the_transport_pools_connections_and_reads_both_views(server) -> None:
    transport = HttpxStreamTransport(max_connections=4)

    async def two():
        body = json.dumps(request_body("ok", "x", 2, api="chat")).encode()
        first = await transport.send(server.url + CHAT, build_request_headers("ok"), body, 5.0)
        second = await transport.send(server.url + CHAT, build_request_headers("ok"), body, 5.0)
        await transport.aclose()
        return first, second

    first, second = asyncio.run(two())
    assert first.status == 200 and first.done_seen and first.completion_tokens == 3 and first.target_pod == "pod-a"
    assert first.v1_success and first.v1_first_token_ms <= first.first_token_ms
    assert first.v1_target_pod == "pod-a"
    assert server.requests[0]["port"] == server.requests[1]["port"]  # one kept-alive connection
    assert server.requests[0]["headers"]["accept-encoding"] == "identity"  # as urllib sent


def test_a_cut_stream_is_a_transport_failure_strictly_and_a_success_for_v1(server) -> None:
    res = stream_request(server.url + CHAT, build_request_headers("cut"),
                         json.dumps({"model": "cut"}).encode(), 5.0)
    assert res.status == 0 and res.error and res.stream_interrupted
    assert res.v1_success and res.v1_status == 200 and res.interrupt_error
    assert metrics.strict_failure(res) == metrics.STRICT_FAILURE_TRANSPORT


def test_an_http_error_keeps_its_body_and_headers(server) -> None:
    res = stream_request(server.url + CHAT, {}, json.dumps({"model": "shed"}).encode(), 5.0)
    assert res.status == 503 and res.error == "HTTP 503" and "overflow" in res.error_body
    assert res.error_headers["content-type"] == "text/plain" and not res.v1_success


def test_a_refused_connection_is_status_zero() -> None:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    res = stream_request(f"http://127.0.0.1:{port}{CHAT}", {}, b"{}", 2.0)
    assert res.status == 0 and res.error and not res.timed_out and res.v1_success is False


def test_the_blocking_seam_works_from_inside_a_running_loop(server) -> None:
    async def inside():
        return stream_request(server.url + CHAT, build_request_headers("ok"), b'{"model": "ok"}', 5.0)

    assert asyncio.run(inside()).status == 200


def test_the_calib_sender_sends_the_request_body_byte_for_byte(server) -> None:
    sender = StreamingHttpSender(server.url + CHAT, api="chat", prompt_mode="natural", request_seed=5)
    request = ScheduledRequest("ok-1", "ok", 0.0, prompt="a fixed prompt", prompt_tokens=9, max_output_tokens=16)

    async def go():
        await sender(request, time.monotonic(), time.monotonic())
        await sender.aclose()

    asyncio.run(go())
    sender.close()
    (seen,) = server.requests
    assert seen["raw"] == json.dumps(request_body("ok", "a fixed prompt", 16, api="chat", seed=5)).encode()
    for key, value in build_request_headers("ok").items():
        assert seen["headers"][key.lower()] == value
    record = sender.records[0]
    assert "v1_ttft_ms" not in record and "strict_success" not in record  # the calibration row is unchanged
    assert record["http_status"] == 200 and record["api"] == "chat" and sender.profile == "calib"


def test_dual_metrics_adds_both_bases_to_a_calibration_row(server) -> None:
    sender = StreamingHttpSender(server.url + CHAT, api="chat", dual_metrics=True)
    asyncio.run(sender(ScheduledRequest("ok-1", "ok", 0.0, prompt="p", max_output_tokens=4), 0.0, 0.0))
    sender.close()
    record = sender.records[0]
    assert record["v1_success"] and record["strict_success"]
    assert record["v1_ttft_ms"] <= record["strict_ttft_ms"] == record["ttft_ms"]


# ------------------------------------------------------------------ the runner


def _events(n: int, model: str = "ok", step: float = 0.01) -> list[ScheduledRequest]:
    return [ScheduledRequest(f"{model}-{i}", model, i * step, prompt=f"p{i}", max_output_tokens=4)
            for i in range(n)]


def _factory(url: str):
    def make(index, in_flight, on_record):
        return StreamingHttpSender(url + CHAT, api="chat", in_flight=in_flight, on_record=on_record,
                                   process_id=index)
    return make


def test_the_runner_shards_the_schedule_and_sends_everything_on_time(server) -> None:
    run = ProcessPoolRunner(_events(30), _factory(server.url), processes=3).run()
    assert sorted(r["request_id"] for r in run.records) == sorted(e.request_id for e in _events(30))
    assert {w["worker"] for w in run.workers} == {0, 1, 2} and all(w["requests"] == 10 for w in run.workers)
    assert len(run.report.records) == 30
    lateness = sorted(r["on_wire_delay_ms"] for r in run.records)
    assert lateness[-1] < 250.0, lateness  # plumbing check on a shared host; see bench_sender.py
    assert len(server.requests) == 30


def test_the_gate_truncates_after_an_overflow_and_keeps_the_drain(server) -> None:
    events = _events(1, model="shed", step=0.0) + [
        ScheduledRequest(f"ok-{i}", "ok", 0.3 + i * 0.01, prompt=f"q{i}", max_output_tokens=4) for i in range(10)
    ] + [ScheduledRequest("drain-0", "ok", 0.6, prompt="d", max_output_tokens=4)]
    gate = StopGate(truncate=True, drain_start_s=0.55)

    def observe(record):
        if record.get("http_status") == 503:
            gate.trip_truncation(record["scheduled_offset_s"], record["actual_send_ts_ms"], record)

    run = ProcessPoolRunner(events, _factory(server.url), processes=2, gate=gate, observer=observe).run()
    trunc, _ = gate.summary(run.workers)
    assert trunc.truncated and trunc.truncated_at_offset_s == 0.0 and trunc.censored == 10
    assert {r["request_id"] for r in run.records} == {"shed-0", "drain-0"}


def test_the_gate_stops_at_the_backlog_ceiling(server) -> None:
    events = _events(8, model="slow", step=0.02)
    gate = StopGate(max_backlog=3)
    run = ProcessPoolRunner(events, _factory(server.url), processes=2, gate=gate).run()
    _, backlog = gate.summary(run.workers)
    assert backlog.truncated and backlog.peak_outstanding == 3
    assert len(run.records) + backlog.censored == 8 and len(run.records) == 3

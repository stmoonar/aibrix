"""openloop.drive_cell_schedule on the unified client's multi-process runner, against a
local HTTP server (127.0.0.1): the cell is sent from several worker processes, and the
truncation / backlog rules keep their meaning across them."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scripts import openloop
from tre_replayer.engine.schedule import RpsSegment

COMPLETIONS = "/v1/completions"


def _sse(*objs) -> bytes:
    return b"".join(b"data: " + json.dumps(o).encode() + b"\n\n" for o in objs) + b"data: [DONE]\n\n"


class _Server:
    def __init__(self, *, shed_first: int = 0, delay_s: float = 0.0) -> None:
        self.count = 0
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                with outer.lock:
                    outer.count += 1
                    n = outer.count
                if n <= shed_first:
                    text = b"upstream connect error or disconnect/reset before headers. reset reason: overflow"
                    self.send_response(503)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(text)))
                    self.end_headers()
                    self.wfile.write(text)
                    return
                if delay_s:
                    time.sleep(delay_s)
                out = int(body.get("max_tokens") or 2)
                chunks = [{"choices": [{"index": 0, "text": " t", "finish_reason": None}]} for _ in range(out - 1)]
                chunks.append({"choices": [{"index": 0, "text": " t", "finish_reason": "length"}]})
                chunks.append({"choices": [], "usage": {"prompt_tokens": 16, "completion_tokens": out}})
                payload = _sse(*chunks)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}{COMPLETIONS}"

    def close(self) -> None:
        self.httpd.shutdown()


@pytest.fixture
def make_server():
    servers = []

    def make(**kw):
        servers.append(_Server(**kw))
        return servers[-1]

    yield make
    for srv in servers:
        srv.close()


def _segments(rps=40.0, end_s=1.0):
    return [RpsSegment(model="m", start_s=0.0, end_s=end_s, rps=rps, input_tokens=16, max_output_tokens=4)]


def test_a_cell_is_sent_from_several_processes(make_server, tmp_path) -> None:
    server = make_server()
    client: dict = {}
    raw = tmp_path / "c.jsonl"
    records: list = []
    _, _, guard = openloop.drive_cell_schedule(
        server.url, "m", "c1", _segments(), raw_path=raw, sender_processes=3, client_out=client,
        prompt_mode="text", records_out=records,
    )
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    assert rows and guard.completed == len(rows) == guard.scheduled == server.count
    assert client["processes"] == 3 and client["profile"]["name"] == "replay"
    assert client["wire"]["transport"] == "httpx"
    # generous: this checks the plumbing on a shared test host; bench_sender.py measures it
    assert max(r["on_wire_delay_ms"] for r in records) < 250.0
    # in_flight_at_send is the run's (shared across the workers), never above the run's total
    assert all(1 <= r["in_flight_at_send"] <= len(records) for r in records)


def test_an_overflow_truncates_the_cell_across_processes(make_server) -> None:
    server = make_server(shed_first=1)
    _, _, guard = openloop.drive_cell_schedule(
        server.url, "m", "c2", _segments(rps=40.0, end_s=1.5), sender_processes=2, prompt_mode="text",
        truncate_on_proxy_shed=True, drain_start_s=None,
    )
    assert guard.truncated and guard.truncation_cause == openloop.TRUNCATION_ADMISSION_OVERFLOW
    assert guard.censored > 0 and guard.sent + guard.censored == guard.scheduled


def test_the_backlog_ceiling_holds_across_processes(make_server) -> None:
    server = make_server(delay_s=0.5)
    _, _, guard = openloop.drive_cell_schedule(
        server.url, "m", "c3", _segments(rps=40.0, end_s=1.0), sender_processes=2, prompt_mode="text",
        max_backlog=5,
    )
    assert guard.truncated and guard.truncation_cause == openloop.TRUNCATION_BACKLOG
    assert guard.backlog_limit == 5 and guard.sent == 5 and guard.sent + guard.censored == guard.scheduled

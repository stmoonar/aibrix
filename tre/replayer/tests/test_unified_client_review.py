"""Review round 1 of the unified client (2026-10-01): workers never outlive their parent,
client faults become records, runs that fail keep what they got, connection evidence,
the one no-byte retry, truncated bodies, continuation marks on a cut stream, the gate
tripped by the worker that saw the overflow, code provenance.

Local only (127.0.0.1)."""
from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from test_unified_client import CHAT, _events, _factory, _Server  # noqa: F401
from tre_replayer.engine import metrics, profiles
from tre_replayer.engine.api import build_request_headers, request_body
from tre_replayer.engine.http_sender import StreamingHttpSender
from tre_replayer.engine.procpool import ProcessPoolRunner, RunnerError, StopGate
from tre_replayer.engine.prompts import MODE_TOKEN_IDS
from tre_replayer.engine.schedule import ScheduledRequest
from tre_replayer.engine.stream import stream_request
from tre_replayer.engine.transport import HttpxStreamTransport


@pytest.fixture
def server():
    srv = _Server()
    yield srv
    srv.close()


def _gone(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return True
    return "State:\tZ" in state  # a zombie sends nothing


PARENT = textwrap.dedent('''
    import sys
    from tre_replayer.engine.http_sender import StreamingHttpSender
    from tre_replayer.engine.procpool import ProcessPoolRunner
    from tre_replayer.engine.schedule import ScheduledRequest

    url = sys.argv[1]
    events = [ScheduledRequest(f"o-{i}", "ok", i * 0.02, prompt=f"p{i}", max_output_tokens=2)
              for i in range(3000)]  # 60 s of load

    def make(index, in_flight, on_record):
        return StreamingHttpSender(url, api="chat", in_flight=in_flight, on_record=on_record,
                                   process_id=index)

    runner = ProcessPoolRunner(events, make, processes=3)
    print("workers", *[p.pid for p in runner._procs], flush=True)
    runner.run()
''')


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG / /proc are Linux")
@pytest.mark.parametrize("sig", [signal.SIGKILL, signal.SIGTERM])
def test_workers_stop_sending_within_a_second_of_the_parent_dying(server, tmp_path, sig) -> None:
    script = tmp_path / "parent.py"
    script.write_text(PARENT, encoding="utf-8")
    parent = subprocess.Popen([sys.executable, str(script), server.url + CHAT], stdout=subprocess.PIPE,
                              text=True, env=dict(os.environ))
    try:
        workers = [int(pid) for pid in parent.stdout.readline().split()[1:]]
        deadline = time.time() + 20
        while len(server.requests) < 30 and time.time() < deadline:
            time.sleep(0.05)
        assert len(server.requests) >= 30, "the load never started"
        parent.send_signal(sig)
        parent.wait(timeout=10)
        time.sleep(1.0)
        settled = len(server.requests)
        time.sleep(1.5)
        assert len(server.requests) == settled, "a worker kept sending after its parent died"
        assert all(_gone(pid) for pid in workers)
    finally:
        if parent.poll() is None:
            parent.kill()


def test_a_client_fault_is_that_requests_record_not_the_runs_end() -> None:
    def broken_seam(*args):
        raise RuntimeError("seam broke")

    sender = StreamingHttpSender("http://gw/v1/completions", stream_call=broken_seam, prompt_mode=MODE_TOKEN_IDS)
    asyncio.run(sender(ScheduledRequest("m-0", "m", 0.0, prompt_tokens=8, max_output_tokens=4), 0.0, 0.0))
    sender.close()
    (record,) = sender.records
    assert record["http_status"] == 0 and record["client_error"].startswith("client error: RuntimeError")
    e1 = StreamingHttpSender("http://gw", profile="e1_v1", v1_options=profiles.V1ChatOptions())
    asyncio.run(e1(ScheduledRequest("m-1", "m", 0.0), 0.0, 0.0))  # no prompt: a client fault
    (record,) = e1.records
    assert record["success"] is False and record["success_strict"] is False and "prompt" in record["client_error"]


def test_validate_refuses_a_bad_schedule_before_anything_forks(server) -> None:
    def validate(event):
        if not event.prompt:
            raise ValueError(f"{event.request_id} has no prompt")

    events = _events(3) + [ScheduledRequest("bad", "ok", 1.0)]
    import multiprocessing

    before = len(multiprocessing.active_children())
    with pytest.raises(ValueError, match="bad has no prompt"):
        ProcessPoolRunner(events, _factory(server.url), processes=2, validate=validate)
    assert len(multiprocessing.active_children()) == before


def test_a_run_that_fails_midway_keeps_the_records_it_got(server) -> None:
    base = _factory(server.url)

    class Dying:
        """A sender whose worker dies when it reaches request ok-8."""

        def __init__(self, inner):
            self.inner = inner

        async def __call__(self, request, scheduled_ts, actual_ts):
            if request.request_id == "ok-8":
                os._exit(3)
            await self.inner(request, scheduled_ts, actual_ts)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    def make(index, in_flight, on_record):
        return Dying(base(index, in_flight, on_record))

    with pytest.raises(RunnerError) as failure:
        with ProcessPoolRunner(_events(12, step=0.05), make, processes=1) as runner:
            runner.run()
    assert 1 <= len(failure.value.records) <= 8
    assert {r["request_id"] for r in failure.value.records} <= {f"ok-{i}" for i in range(8)}


def test_close_reaps_workers_of_a_runner_that_never_ran(server) -> None:
    runner = ProcessPoolRunner(_events(4), _factory(server.url), processes=2)
    procs = list(runner._procs)
    assert all(p.is_alive() for p in procs)
    runner.close()
    assert not any(p.is_alive() for p in procs)


def test_connection_evidence_and_reuse(server) -> None:
    transport = HttpxStreamTransport(max_connections=4, keepalive_expiry=4.0)
    body = request_body("ok", "x", 2, api="chat")

    async def two():
        import json

        raw = json.dumps(body).encode()
        first = await transport.send(server.url + CHAT, build_request_headers("ok"), raw, 5.0)
        second = await transport.send(server.url + CHAT, build_request_headers("ok"), raw, 5.0)
        await transport.aclose()
        return first, second

    first, second = asyncio.run(two())
    assert first.connection_reused is False and second.connection_reused is True
    assert first.conn_acquire_ms is not None and first.conn_acquire_ms >= 0.0
    assert first.transport_retries == second.transport_retries == 0
    assert first.stream_complete is True
    assert transport.provenance()["keepalive_expiry_s"] == 4.0


def test_keepalive_expiry_is_four_seconds_and_configurable(monkeypatch) -> None:
    assert HttpxStreamTransport().keepalive_expiry == 4.0
    monkeypatch.setenv("TRE_SENDER_KEEPALIVE_EXPIRY_S", "2.5")
    assert HttpxStreamTransport().keepalive_expiry == 2.5


def test_only_a_request_that_never_left_is_repeated_once(server) -> None:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    refused = stream_request(f"http://127.0.0.1:{port}{CHAT}", {}, b"{}", 2.0)
    assert refused.status == 0 and refused.transport_retries == 1 and len(refused.attempt_log) == 2
    shed = stream_request(server.url + CHAT, {}, b'{"model": "shed"}', 5.0)
    assert shed.status == 503 and shed.transport_retries == 0
    cut = stream_request(server.url + CHAT, {}, b'{"model": "cut"}', 5.0)  # bytes had left
    assert cut.status == 0 and cut.transport_retries == 0


def test_the_send_lateness_includes_the_wait_for_a_connection(server) -> None:
    sender = StreamingHttpSender(server.url + CHAT, api="chat")
    asyncio.run(sender(ScheduledRequest("ok-1", "ok", 0.0, prompt="p", max_output_tokens=2), 0.0, 0.0))
    sender.close()
    record = sender.records[0]
    assert record["conn_acquire_ms"] is not None and record["connection_reused"] is False
    assert record["pool_wait_ms"] >= record["conn_acquire_ms"]
    assert record["on_wire_delay_ms"] >= record["conn_acquire_ms"]


def test_a_body_that_just_stops_is_a_failure_on_the_strict_basis(server) -> None:
    res = stream_request(server.url + CHAT, {}, b'{"model": "nodone"}', 5.0)
    assert res.status == 200 and res.stream_complete is False and not res.done_seen
    assert metrics.strict_failure(res) == metrics.STRICT_FAILURE_INCOMPLETE
    assert res.v1_success  # v1 counted it as a success


def test_a_cut_stream_keeps_the_continuation_it_reported(server) -> None:
    res = stream_request(server.url + CHAT, {}, b'{"model": "cutcont"}', 5.0)
    assert res.status == 0 and res.stream_interrupted and res.tre_continued == 2


def test_the_worker_that_sees_the_overflow_trips_the_gate() -> None:
    gate = StopGate(truncate=True, trip_if=lambda r: r.get("http_status") == 503)
    gate.observe({"http_status": 200, "scheduled_offset_s": 0.5, "actual_send_ts_ms": 10})
    assert not gate.truncated
    overflow = {"http_status": 503, "scheduled_offset_s": 1.5, "actual_send_ts_ms": 42}
    gate.observe(overflow)
    trunc, _ = gate.summary([])
    assert trunc.truncated and trunc.truncated_at_offset_s == 1.5 and trunc.truncated_at_ts_ms == 42
    gate.observe({"http_status": 503, "scheduled_offset_s": 2.0, "actual_send_ts_ms": 50})
    assert gate.summary([])[0].truncated_at_offset_s == 1.5  # the first overflow's own offset


def test_the_provenance_names_the_code_that_sent() -> None:
    import tre_replayer

    code = profiles.client_provenance("calib")["code"]
    assert code["tre_replayer"] == str(Path(tre_replayer.__file__).resolve().parent)
    assert "git_sha" in code and "git_dirty" in code

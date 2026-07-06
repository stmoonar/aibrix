from __future__ import annotations

import asyncio
import json

from tre_replayer.engine.http_sender import StreamResult, StreamingHttpSender
from tre_replayer.engine.schedule import ScheduledRequest


def _req() -> ScheduledRequest:
    return ScheduledRequest(request_id="m-0", model="m", scheduled_offset_s=0.0, prompt_tokens=32, max_output_tokens=64)


def test_sender_records_streamed_result_and_delay() -> None:
    calls: list[dict] = []

    def fake(url, headers, body, timeout):
        calls.append(json.loads(body))
        assert headers["model"] == "m"
        return StreamResult(status=200, first_token_ms=90.0, done_ms=300.0, prompt_tokens=32, completion_tokens=64)

    sender = StreamingHttpSender("http://gw/v1/completions", stream_call=fake, now_ms=lambda: 1000)
    asyncio.run(sender(_req(), scheduled_ts=10.0, actual_ts=10.05))
    sender.close()

    rec = sender.records[0]
    assert "pool_wait_ms" in rec  # F5 starvation gauge
    assert rec["ttft_ms"] == 90.0 and rec["e2e_ms"] == 300.0
    assert rec["completion_tokens"] == 64 and rec["http_status"] == 200 and rec["error"] is None
    assert rec["actual_send_ts_ms"] == 1000
    assert abs(rec["schedule_delay_ms"] - 50.0) < 1e-6  # (10.05 - 10.0) * 1000
    body = calls[0]
    assert body["stream"] is True and body["model"] == "m" and body["max_tokens"] == 64 and body["ignore_eos"] is True


def test_sender_records_error_status() -> None:
    sender = StreamingHttpSender("http://gw", stream_call=lambda *a: StreamResult(500, None, 12.0, error="HTTP 500"))
    asyncio.run(sender(_req(), 0.0, 0.0))
    sender.close()
    assert sender.records[0]["http_status"] == 500 and sender.records[0]["error"] == "HTTP 500"
    assert sender.records[0]["ttft_ms"] is None


def test_sender_write_jsonl(tmp_path) -> None:
    sender = StreamingHttpSender("http://gw", stream_call=lambda *a: StreamResult(200, 10.0, 20.0, 1, 1))
    asyncio.run(sender(_req(), 0.0, 0.0))
    sender.close()
    path = tmp_path / "raw.jsonl"
    n = sender.write_jsonl(str(path))
    assert n == 1
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["request_id"] == "m-0"

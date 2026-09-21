from __future__ import annotations

import asyncio
import json

import pytest

from tre_replayer.engine.http_sender import StreamResult, StreamingHttpSender
from tre_replayer.engine.prompts import MODE_TEXT, MODE_TOKEN_IDS
from tre_replayer.engine.schedule import ScheduledRequest


def _req() -> ScheduledRequest:
    return ScheduledRequest(request_id="m-0", model="m", scheduled_offset_s=0.0, prompt_tokens=32, max_output_tokens=64)


def test_sender_records_streamed_result_and_delay() -> None:
    calls: list[dict] = []

    def fake(url, headers, body, timeout):
        calls.append(json.loads(body))
        assert headers["model"] == "m"
        return StreamResult(status=200, first_token_ms=90.0, done_ms=300.0, prompt_tokens=32, completion_tokens=64)

    sender = StreamingHttpSender(
        "http://gw/v1/completions", stream_call=fake, prompt_mode=MODE_TOKEN_IDS, now_ms=lambda: 1000
    )
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
    sender = StreamingHttpSender(
        "http://gw",
        stream_call=lambda *a: StreamResult(500, None, 12.0, error="HTTP 500"),
        prompt_mode=MODE_TOKEN_IDS,
    )
    asyncio.run(sender(_req(), 0.0, 0.0))
    sender.close()
    assert sender.records[0]["http_status"] == 500 and sender.records[0]["error"] == "HTTP 500"
    assert sender.records[0]["ttft_ms"] is None


def test_sender_write_jsonl(tmp_path) -> None:
    sender = StreamingHttpSender(
        "http://gw",
        stream_call=lambda *a: StreamResult(200, 10.0, 20.0, 1, 1),
        prompt_mode=MODE_TOKEN_IDS,
    )
    asyncio.run(sender(_req(), 0.0, 0.0))
    sender.close()
    path = tmp_path / "raw.jsonl"
    n = sender.write_jsonl(str(path))
    assert n == 1
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["request_id"] == "m-0"


def test_sender_sends_a_distinct_prompt_of_the_requested_length_per_request() -> None:
    """A constant prompt would be free after the first request on a prefix-caching
    engine, which is what invalidated the earlier capacity scans."""
    bodies: list[dict] = []

    def fake(url, headers, body, timeout):
        bodies.append(json.loads(body))
        return StreamResult(status=200, first_token_ms=1.0, done_ms=2.0, prompt_tokens=32, completion_tokens=8)

    sender = StreamingHttpSender("http://gw", stream_call=fake, prompt_mode=MODE_TOKEN_IDS)
    for i in range(50):
        request = ScheduledRequest(
            request_id=f"m-{i}", model="m", scheduled_offset_s=0.0,
            prompt_tokens=32, max_output_tokens=8,
        )
        asyncio.run(sender(request, 0.0, 0.0))
    sender.close()

    prompts_sent = [b["prompt"] for b in bodies]
    assert all(isinstance(p, list) and len(p) == 32 for p in prompts_sent)  # exact length
    assert len({tuple(p) for p in prompts_sent}) == 50  # all distinct
    assert len({tuple(p[:4]) for p in prompts_sent}) == 50  # distinct from the first tokens


def test_sender_prompts_are_reproducible_and_honour_an_explicit_prompt() -> None:
    def run(prompt_mode, explicit=""):
        bodies: list[dict] = []

        def fake(url, headers, body, timeout):
            bodies.append(json.loads(body))
            return StreamResult(200, 1.0, 2.0, 32, 8)

        sender = StreamingHttpSender("http://gw", stream_call=fake, prompt_mode=prompt_mode)
        request = ScheduledRequest(
            request_id="m-7", model="m", scheduled_offset_s=0.0, prompt=explicit,
            prompt_tokens=32, max_output_tokens=8,
        )
        asyncio.run(sender(request, 0.0, 0.0))
        sender.close()
        return bodies[0]["prompt"]

    assert run(MODE_TOKEN_IDS) == run(MODE_TOKEN_IDS)  # same key -> same prompt
    text = run(MODE_TEXT)
    assert isinstance(text, str) and len(text.split()) == 32
    assert run(MODE_TOKEN_IDS, explicit="verbatim prompt") == "verbatim prompt"


def test_sender_defaults_to_the_natural_language_prompt_mode() -> None:
    """The default is an explicit, single-sourced constant - not whatever the sender's
    signature happened to say. A silent divergence between the two senders would make
    two runs of the same cell incomparable without anything in the artifacts saying so."""
    from tre_replayer.engine import prompts

    assert prompts.DEFAULT_MODE == prompts.MODE_NATURAL
    sender = StreamingHttpSender("http://gw", stream_call=lambda *a: StreamResult(200, 1.0, 2.0, 1, 1))
    assert sender._prompt_mode == prompts.DEFAULT_MODE
    sender.close()


def test_sender_records_the_serving_pod_when_the_answer_names_one() -> None:
    def fake(url, headers, body, timeout):
        return StreamResult(200, 1.0, 2.0, 32, 8, target_pod="dsqwen-7b-node9-gpu-0-abc")

    sender = StreamingHttpSender("http://gw", stream_call=fake, prompt_mode=MODE_TOKEN_IDS)
    asyncio.run(sender(_req(), 0.0, 0.0))
    sender.close()
    assert sender.records[0]["target_pod"] == "dsqwen-7b-node9-gpu-0-abc"


def test_sender_records_a_null_pod_when_the_serving_path_names_none() -> None:
    """Not attributable is recorded as null, never guessed: the per-model HTTPRoute the
    campaign uses does not run the plugin that names a pod."""
    sender = StreamingHttpSender(
        "http://gw", stream_call=lambda *a: StreamResult(200, 1.0, 2.0, 32, 8),
        prompt_mode=MODE_TOKEN_IDS,
    )
    asyncio.run(sender(_req(), 0.0, 0.0))
    sender.close()
    assert sender.records[0]["target_pod"] is None


def test_pod_header_extraction_prefers_the_pod_name_over_its_address() -> None:
    from tre_replayer.engine.http_sender import pod_from_headers

    assert pod_from_headers({"target-pod-ip": "10.0.0.1:8000", "target-pod": "pod-a"}) == "pod-a"
    assert pod_from_headers({"target-pod-ip": "10.0.0.1:8000"}) == "10.0.0.1:8000"
    assert pod_from_headers({"content-type": "application/json"}) is None
    assert pod_from_headers(None) is None


def test_routing_strategy_swaps_the_model_header_for_the_strategy_header() -> None:
    """The per-model HTTPRoute matches on the ``model`` header, so sending it wins over
    the catch-all route and the request never reaches the plugin that names a pod.
    Asking for a routing strategy therefore has to drop that header."""
    from tre_replayer.engine.http_sender import build_request_headers

    default = build_request_headers("dsqwen-7b")
    assert default["model"] == "dsqwen-7b" and "routing-strategy" not in default

    routed = build_request_headers("dsqwen-7b", "least-request")
    assert routed["routing-strategy"] == "least-request" and "model" not in routed


# ------------------------------------------------ what "offered on time" actually means


class _ScriptedMono:
    """A monotonic clock the test moves by hand, so no assertion depends on a sleep."""

    def __init__(self, start: float = 100.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


def _wire_probe_sender(monkeypatch, mono, *, build_cost_s: float, **kwargs):
    """A sender whose prompt build costs ``build_cost_s`` of the scripted clock."""
    from tre_replayer.engine import http_sender as module

    def slow_build(token_count, seed_key, **_kw):
        mono.t += build_cost_s
        return "prompt"

    monkeypatch.setattr(module, "build_prompt", slow_build)
    return StreamingHttpSender(
        "http://gw",
        stream_call=lambda *a: StreamResult(200, 10.0, 20.0, 1, 1),
        mono=mono,
        now_ms=lambda: 1000,
        **kwargs,
    )


def test_the_old_delay_gauges_cannot_see_the_time_spent_before_the_socket(monkeypatch) -> None:
    """The defect this metric exists for. The dispatcher fired this request exactly on
    time and a worker picked it up immediately, so both of the pre-existing gauges read
    zero - while 50 ms of tokenizer work happened before a single byte went out."""
    mono = _ScriptedMono()
    sender = _wire_probe_sender(monkeypatch, mono, build_cost_s=0.05)
    asyncio.run(sender(_req(), scheduled_ts=mono.t, actual_ts=mono.t))
    sender.close()

    rec = sender.records[0]
    assert rec["schedule_delay_ms"] == 0.0  # the dispatcher was not late
    assert rec["pool_wait_ms"] == 0.0  # the pool did not starve
    # ...and yet the request reached the wire 50 ms after it was due
    assert rec["on_wire_delay_ms"] == pytest.approx(50.0)
    assert rec["body_build_ms"] == pytest.approx(50.0)


def test_the_three_segments_add_up_to_the_on_wire_delay(monkeypatch) -> None:
    """Each segment is kept because it attributes the miss: event loop, then pool, then
    everything before the socket. Their sum is the deadline itself."""
    mono = _ScriptedMono(start=100.0)
    sender = _wire_probe_sender(monkeypatch, mono, build_cost_s=0.02)
    # due at 99.9, fired at 99.94 (loop late 40 ms), picked up at 100.0 (pool 60 ms)
    asyncio.run(sender(_req(), scheduled_ts=99.9, actual_ts=99.94))
    sender.close()

    rec = sender.records[0]
    assert rec["schedule_delay_ms"] == pytest.approx(40.0)
    assert rec["pool_wait_ms"] == pytest.approx(60.0)
    assert rec["body_build_ms"] == pytest.approx(20.0)
    assert rec["on_wire_delay_ms"] == pytest.approx(120.0)
    assert rec["on_wire_delay_ms"] == pytest.approx(
        rec["schedule_delay_ms"] + rec["pool_wait_ms"] + rec["body_build_ms"]
    )


def test_a_materialised_prompt_takes_the_build_off_the_send_path(monkeypatch) -> None:
    """The fix, measured by the same gauge: with the prompt already built, nothing
    happens between the scheduled instant and the socket."""
    from tre_replayer.engine.prompt_store import PromptStore

    mono = _ScriptedMono()
    sender = _wire_probe_sender(
        monkeypatch, mono, build_cost_s=0.05, prompt_store=PromptStore({"m-0": "ready"})
    )
    asyncio.run(sender(_req(), scheduled_ts=mono.t, actual_ts=mono.t))
    sender.close()

    rec = sender.records[0]
    assert rec["on_wire_delay_ms"] == 0.0
    assert rec["body_build_ms"] == 0.0
    assert sender.prompt_store_misses == 0


def test_a_missing_materialised_prompt_is_counted_and_still_sent(monkeypatch) -> None:
    """A miss must not drop the request - but it must not be silent either, because the
    cell then paid the tokenizer fit inside its own lateness."""
    from tre_replayer.engine.prompt_store import PromptStore

    mono = _ScriptedMono()
    sender = _wire_probe_sender(
        monkeypatch, mono, build_cost_s=0.05, prompt_store=PromptStore({"other": "ready"})
    )
    asyncio.run(sender(_req(), scheduled_ts=mono.t, actual_ts=mono.t))
    sender.close()

    assert sender.prompt_store_misses == 1
    assert sender.records[0]["on_wire_delay_ms"] == pytest.approx(50.0)


def test_the_record_carries_its_place_in_the_schedule() -> None:
    """Needed to bin the achieved arrivals on the schedule's own grid; without it the
    achieved series can only be placed relative to whenever the process started."""
    sender = StreamingHttpSender(
        "http://gw",
        stream_call=lambda *a: StreamResult(200, 10.0, 20.0, 1, 1),
        prompt_mode=MODE_TOKEN_IDS,
    )
    request = ScheduledRequest(
        request_id="m-7", model="m", scheduled_offset_s=12.5, prompt_tokens=8, max_output_tokens=4
    )
    asyncio.run(sender(request, 0.0, 0.0))
    sender.close()
    assert sender.records[0]["scheduled_offset_s"] == 12.5

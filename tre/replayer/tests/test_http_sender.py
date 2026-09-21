from __future__ import annotations

import asyncio
import json

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


def test_sender_sends_natural_language_of_the_exact_length_with_a_stub_tokenizer() -> None:
    """Exactness and uniqueness with no model on disk: the tokenizer is a seam."""
    from tre_replayer.engine.prompts import MODE_NATURAL, build_natural_prompt

    tok = _WordTokenizer()
    prompts_sent = [
        build_natural_prompt(64, f"m|m-{i}", tokenizer=tok) for i in range(30)
    ]
    assert all(tok.count(p) == 64 for p in prompts_sent)
    assert len(set(prompts_sent)) == 30
    assert len({tuple(p.split()[:4]) for p in prompts_sent}) == 30  # diverge at the head
    assert build_natural_prompt(64, "m|m-0", tokenizer=tok) == prompts_sent[0]
    assert MODE_NATURAL == "natural"


class _WordTokenizer:
    """Whitespace tokenizer standing in for a real one: one token per word, one leading
    special token, ' the' as the single-token filler, and an exact decode roundtrip.
    Enough to exercise the fit loop without a model on disk."""

    overhead = 1
    filler = " the"
    path = "<stub>"

    def __init__(self) -> None:
        self._to_id: dict[str, int] = {}
        self._to_word: dict[int, str] = {}

    def _id(self, word: str) -> int:
        if word not in self._to_id:
            index = len(self._to_id) + 1
            self._to_id[word] = index
            self._to_word[index] = word
        return self._to_id[word]

    def encode_plain(self, text: str) -> list[int]:
        return [self._id(word) for word in text.split()]

    def decode_plain(self, ids) -> str:
        return " ".join(self._to_word[i] for i in ids)

    def count(self, text: str) -> int:
        return len(self.encode_plain(text)) + self.overhead

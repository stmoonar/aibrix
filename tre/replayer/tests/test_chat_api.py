"""The chat endpoint: request body, exact templated length, SSE parsing.

Calibration sends ``/v1/chat/completions`` (like v1 and the E1 client) with every prompt
fitted so the *templated* prompt is the requested length; the replayer's own default
stays ``/v1/completions``. Stub tokenizers keep most tests hermetic; the fleet group
re-checks against the real tokenizers (and ``transformers``' own ``apply_chat_template``)
when they are on local disk, and skips otherwise.
"""
from __future__ import annotations

import asyncio
import functools
import json
import os
import re

import pytest

from tre_replayer.engine import api, corpus, model_tokenizer, prompt_store, prompts
from tre_replayer.engine.http_sender import (
    StreamingHttpSender,
    StreamResult,
    _default_stream_call,
    chunk_token_field,
)
from tre_replayer.engine.schedule import ScheduledRequest

CHAT_URL = "http://gw/v1/chat/completions"

_CJK_CLASS = "　-〿㐀-䶿一-鿿＀-￯"
#: A special marker, one CJK character, or one ASCII word with its leading whitespace.
_TOKEN = re.compile(rf"<[A-Za-z]+>|\s*(?:[{_CJK_CLASS}]|[^\s<{_CJK_CLASS}]+)")


class ChatStubTokenizer:
    """One token per marker (``<BOS>`` ...), CJK character and ASCII word, and a chat
    template ``<BOS><U>{content}<A><think>`` - four tokens, like the fleet's five."""

    overhead = 1
    filler = " the"
    path = "<chat-stub>"
    model = "m"
    chat_prefix = "<BOS><U>"
    chat_suffix = "<A><think>"
    chat_error = None

    def __init__(self) -> None:
        self._to_id: dict[str, int] = {}
        self._to_piece: dict[int, str] = {}

    def _id(self, piece: str) -> int:
        if piece not in self._to_id:
            self._to_id[piece] = len(self._to_id) + 1
            self._to_piece[self._to_id[piece]] = piece
        return self._to_id[piece]

    def encode_plain(self, text: str) -> list[int]:
        return [self._id(piece) for piece in _TOKEN.findall(text)]

    def decode_plain(self, ids) -> str:
        return "".join(self._to_piece[i] for i in ids)

    def count(self, text: str) -> int:
        return len(self.encode_plain(text)) + self.overhead

    def cjk_token_count(self, text: str) -> int:
        return sum(1 for p in _TOKEN.findall(text) if any(corpus.is_cjk(c) for c in p))


def _share(tok, text: str) -> float:
    return tok.cjk_token_count(text) / len(tok.encode_plain(text))


# ------------------------------------------------------------------------ request body


def test_the_chat_body_is_one_user_message_with_the_fixed_output_switches() -> None:
    body = api.request_body("m", "hello", 64, api="chat")
    assert body == {
        "model": "m", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 64,
        "temperature": 0, "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True},
    }
    assert api.request_body("m", "hello", 64, api="chat", seed=7)["seed"] == 7
    assert api.request_prompt(body) == "hello"


def test_the_completions_body_is_the_one_the_replayer_always_sent() -> None:
    body = api.request_body("m", [1, 2], 8)
    assert list(body) == ["model", "prompt", "max_tokens", "temperature", "ignore_eos", "stream",
                          "stream_options"]
    assert body["prompt"] == [1, 2] and "seed" not in body
    assert api.DEFAULT_API == api.API_COMPLETIONS


def test_chat_carries_text_of_an_exact_length_only() -> None:
    with pytest.raises(ValueError, match="text"):
        api.request_body("m", [1, 2, 3], 8, api="chat")
    for mode in (prompts.MODE_TOKEN_IDS, prompts.MODE_TEXT):
        with pytest.raises(ValueError, match="natural"):
            api.check_api_mode("chat", mode)
        with pytest.raises(ValueError, match="natural"):
            prompts.build_prompt(16, "k", mode=mode, api="chat")
    api.check_api_mode("completions", prompts.MODE_TOKEN_IDS)
    with pytest.raises(ValueError, match="unknown API"):
        api.request_body("m", "x", 1, api="responses")


# ------------------------------------------------------------------------ the sender


def _req(tokens: int = 40) -> ScheduledRequest:
    return ScheduledRequest(request_id="m-0", model="m", scheduled_offset_s=0.0,
                            prompt_tokens=tokens, max_output_tokens=16)


def test_a_chat_sender_sends_the_chat_body_to_the_chat_path(monkeypatch) -> None:
    tok = ChatStubTokenizer()
    real = prompts.build_prompt
    monkeypatch.setattr("tre_replayer.engine.http_sender.build_prompt",
                        lambda n, key, **kw: real(n, key, tokenizer=tok, **{k: v for k, v in kw.items()
                                                                             if k != "tokenizer"}))
    sent: list[tuple[str, dict]] = []

    def fake(url, headers, body, timeout):
        sent.append((url, json.loads(body)))
        return StreamResult(200, 5.0, 50.0, 40, 16, first_token_field="content")

    sender = StreamingHttpSender(CHAT_URL, stream_call=fake, api="chat", request_seed=3)
    try:
        asyncio.run(sender(_req(40), scheduled_ts=0.0, actual_ts=0.0))
    finally:
        sender.close()
    url, body = sent[0]
    assert url == CHAT_URL and body["seed"] == 3 and "prompt" not in body
    content = body["messages"][0]["content"]
    # the templated prompt is the request's length: content + 4 template tokens
    assert model_tokenizer.for_api(tok, "chat").count(content) == 40
    assert len(tok.encode_plain(content)) == 36
    record = sender.records[0]
    assert record["api"] == "chat" and record["first_token_field"] == "content"
    assert record["input_tokens"] == 40 and record["prompt_tokens"] == 40


def test_a_sender_refuses_a_url_or_mode_that_cannot_carry_its_api() -> None:
    with pytest.raises(ValueError, match="chat/completions"):
        StreamingHttpSender("http://gw/v1/completions", api="chat")
    with pytest.raises(ValueError, match="chat endpoint"):
        StreamingHttpSender(CHAT_URL)  # completions (default) on the chat path
    with pytest.raises(ValueError, match="natural"):
        StreamingHttpSender(CHAT_URL, api="chat", prompt_mode=prompts.MODE_TOKEN_IDS)


# ------------------------------------------------------------------------ SSE parsing


def test_the_first_token_is_text_content_or_reasoning_never_a_role_or_usage_chunk() -> None:
    role = {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}
    usage = {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}}
    assert chunk_token_field(role) is None and chunk_token_field(usage) is None
    assert chunk_token_field({"choices": [{"delta": {"content": "嗯"}}]}) == "content"
    assert chunk_token_field({"choices": [{"delta": {"reasoning_content": "Okay"}}]}) == "reasoning_content"
    assert chunk_token_field({"choices": [{"delta": {"reasoning": "Okay", "content": None}}]}) == "reasoning"
    assert chunk_token_field({"choices": [{"text": " a"}]}) == "text"


def _serve_once(body: bytes):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"


def _sse(*objs) -> bytes:
    return b"".join(b"data: " + json.dumps(o).encode() + b"\n\n" for o in objs) + b"data: [DONE]\n\n"


@pytest.mark.parametrize("field", ["content", "reasoning_content", "reasoning"])
def test_a_streamed_chat_answer_yields_ttft_usage_and_the_field(field: str) -> None:
    """The shape vLLM 0.30 streams (measured on the fleet 2026-09-30, no reasoning parser:
    role-only chunk, ``delta.content`` chunks, usage-only chunk)."""
    body = _sse(
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"index": 0, "delta": {field: "Alright"}}]},
        {"choices": [{"index": 0, "delta": {field: ","}, "finish_reason": "length"}]},
        {"choices": [], "usage": {"prompt_tokens": 512, "completion_tokens": 2, "total_tokens": 514}},
    )
    server, url = _serve_once(body)
    try:
        res = _default_stream_call(url, {"Content-Type": "application/json"}, b"{}", 5.0)
    finally:
        server.shutdown()
    assert res.status == 200 and res.first_token_ms is not None and res.first_token_field == field
    assert (res.prompt_tokens, res.completion_tokens, res.finish_reason) == (512, 2, "length")


def test_a_stream_of_role_and_usage_chunks_only_has_no_first_token() -> None:
    body = _sse({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
                {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 0}})
    server, url = _serve_once(body)
    try:
        res = _default_stream_call(url, {}, b"{}", 5.0)
    finally:
        server.shutdown()
    assert res.first_token_ms is None and res.first_token_field is None and res.prompt_tokens == 5


# ------------------------------------------------------------------ the templated count


def test_the_chat_view_counts_the_templated_prompt() -> None:
    tok = ChatStubTokenizer()
    view = model_tokenizer.for_api(tok, "chat")
    assert view.api == "chat" and view.overhead == 4
    assert view.count("hello world") == 6 and view.render("x") == "<BOS><U>x<A><think>"
    assert model_tokenizer.for_api(view, "chat") is view
    assert model_tokenizer.for_api(tok, "completions") is tok
    with pytest.raises(ValueError):
        model_tokenizer.for_api(view, "completions")


@pytest.mark.parametrize("lang", ["mix", "zh", "en"])
@pytest.mark.parametrize("target", [5, 6, 16, 128, 513, 2048])
def test_a_chat_prompt_is_exact_after_the_template(lang: str, target: int) -> None:
    tok = ChatStubTokenizer()
    view = model_tokenizer.for_api(tok, "chat")
    text = prompts.build_natural_prompt(target, f"chat|{lang}|{target}", tokenizer=tok, corpus_lang=lang,
                                        api="chat")
    assert view.count(text) == target
    assert "<" not in text  # the template is the engine's, never in the content


def test_below_the_template_floor_a_chat_prompt_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be shorter than 5"):
        prompts.build_natural_prompt(4, "k", tokenizer=ChatStubTokenizer(), api="chat")


@pytest.mark.parametrize("target", [128, 1024, 4096])
def test_the_mix_ratio_holds_in_the_user_content(target: int) -> None:
    tok = ChatStubTokenizer()
    shares = [_share(tok, prompts.build_natural_prompt(target, f"r|{target}|{i}", tokenizer=tok, api="chat"))
              for i in range(10)]
    assert all(abs(s - 0.5) <= (0.03 if target < 256 else 0.01) for s in shares), shares


def test_a_template_that_adds_a_varying_count_is_refused() -> None:
    class Merging(ChatStubTokenizer):
        chat_prefix = "X"  # glued onto the content's first word unless it starts with a space
        chat_suffix = ""

    with pytest.raises(model_tokenizer.TokenizerUnavailable, match="fixed number"):
        model_tokenizer.for_api(Merging(), "chat")


def test_a_tokenizer_without_a_chat_template_is_refused_loudly() -> None:
    class NoTemplate(ChatStubTokenizer):
        chat_prefix = None
        chat_error = "apply_chat_template failed: no template"

    with pytest.raises(model_tokenizer.TokenizerUnavailable, match="no template"):
        prompts.build_natural_prompt(64, "k", tokenizer=NoTemplate(), api="chat")


class _Wrapper:
    def __init__(self, render):
        self._render = render

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert add_generation_prompt is True and tokenize is False
        return self._render(messages[0]["content"])


def test_the_template_is_split_around_the_content_and_checked_on_probes() -> None:
    good = _Wrapper(lambda c: f"<BOS><U>{c}<A><think>\n")
    assert model_tokenizer._chat_template_parts(good) == ("<BOS><U>", "<A><think>\n", None)
    trims = _Wrapper(lambda c: f"<BOS><U>{c.strip()}<A>")
    prefix, suffix, error = model_tokenizer._chat_template_parts(trims)
    assert prefix is None and "transforms its content" in error
    twice = _Wrapper(lambda c: f"{c}<A>{c}")
    assert "exactly once" in model_tokenizer._chat_template_parts(twice)[2]

    def missing(c):
        raise ValueError("no chat template")

    assert "no chat template" in model_tokenizer._chat_template_parts(_Wrapper(missing))[2]


def test_materialised_chat_prompts_record_their_api_and_are_the_inline_ones(tmp_path) -> None:
    tok = ChatStubTokenizer()
    requests = [ScheduledRequest(request_id=f"m-{i}", model="m", scheduled_offset_s=0.1 * i,
                                 prompt_tokens=100 + i, max_output_tokens=8) for i in range(12)]
    store = prompt_store.materialize_prompts(requests, path=tmp_path / "c.prompts.jsonl", tokenizer=tok,
                                             api="chat")
    view = model_tokenizer.for_api(tok, "chat")
    for request in requests:
        text = store.get(request.request_id)
        assert view.count(text) == request.prompt_tokens
        assert text == prompts.build_prompt(request.prompt_tokens,
                                            prompt_store.sender_seed_key("m", request.request_id),
                                            model="m", tokenizer=tok, api="chat")
    rows = [json.loads(line) for line in (tmp_path / "c.prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["api"] for r in rows} == {"chat"}
    with pytest.raises(ValueError, match="natural"):
        prompt_store.materialize_prompts(requests, path=tmp_path / "x.jsonl", tokenizer=tok, api="chat",
                                         mode=prompts.MODE_TOKEN_IDS)


# ------------------------------------------------------------------ the real tokenizers

FLEET = tuple(model_tokenizer.FLEET_TOKENIZER_PATHS)


def _fleet(model: str):
    path = model_tokenizer.FLEET_TOKENIZER_PATHS[model]
    if not os.path.isdir(path):
        pytest.skip(f"{model} tokenizer not on this host ({path})")
    pytest.importorskip("transformers")
    return model_tokenizer.load_tokenizer(model, tokenizer_path=path)


@functools.lru_cache(maxsize=None)
def _hf_wrapper(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_tokenizer.FLEET_TOKENIZER_PATHS[model], local_files_only=True)


def _hf_chat_ids(model: str, content: str) -> list[int]:
    """transformers' own path: apply_chat_template(tokenize=True) - what vLLM does."""
    ids = _hf_wrapper(model).apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                      tokenize=True)
    if hasattr(ids, "keys"):  # a BatchEncoding in newer transformers
        ids = ids["input_ids"]
    return list(ids)


@pytest.mark.parametrize("model", FLEET)
def test_fleet_chat_prompts_are_exact_and_half_chinese(model: str) -> None:
    tok = _fleet(model)
    view = model_tokenizer.for_api(tok, "chat")
    # <｜begin▁of▁sentence｜><｜User｜> ... <｜Assistant｜><think>\n: 5 tokens on all three
    assert view.overhead == 5 and view.chat_suffix.endswith("<think>\n")
    for target in (16, 128, 512, 1024, 3000):
        for i in range(3):
            text = prompts.build_natural_prompt(target, f"fleet-chat|{target}|{i}", tokenizer=tok, api="chat")
            assert view.count(text) == target
            assert len(_hf_chat_ids(model, text)) == target
            if target >= 128:
                assert abs(_share(tok, text) - 0.5) <= 0.01, (target, _share(tok, text))


@pytest.mark.parametrize("model", FLEET)
def test_fleet_chat_tiny_targets_are_exact(model: str) -> None:
    tok = _fleet(model)
    view = model_tokenizer.for_api(tok, "chat")
    for lang in ("mix", "zh", "en"):
        for target in range(6, 40):
            text = prompts.build_natural_prompt(target, f"tiny-chat|{target}", tokenizer=tok, corpus_lang=lang,
                                                api="chat")
            assert view.count(text) == target and "�" not in text


# ------------------------------------------------------------------ the sending core


def test_the_sending_core_is_importable_without_the_sender_and_re_exported_by_it() -> None:
    from tre_replayer.engine import http_sender, stream

    for name in ("StreamResult", "chunk_token_field", "pod_from_headers", "reissue_from_headers",
                 "lower_headers", "read_error_body", "is_client_timeout", "result_fields"):
        assert getattr(http_sender, name) is getattr(stream, name)
    assert http_sender._default_stream_call is stream.stream_request
    for name in ("build_request_headers", "DEFAULT_ROUTING_STRATEGY", "request_body", "check_api_url"):
        assert getattr(http_sender, name) is getattr(api, name)


def test_result_fields_are_the_answers_part_of_a_record() -> None:
    from tre_replayer.engine.stream import result_fields

    res = StreamResult(200, 12.0, 90.0, 512, 8, target_pod="p", finish_reason="length",
                       first_token_field="content")
    fields = result_fields(res)
    assert (fields["ttft_ms"], fields["e2e_ms"], fields["prompt_tokens"], fields["completion_tokens"]) ==         (12.0, 90.0, 512, 8)
    assert fields["http_status"] == 200 and fields["first_token_field"] == "content"
    assert fields["client_timeout"] is False and fields["target_pod"] == "p"


# ------------------------------------------------------------------ review round 1


def test_an_error_chunk_inside_a_200_stream_is_a_failure_not_a_completion() -> None:
    body = _sse(
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"index": 0, "delta": {"content": "A"}}]},
        {"error": {"message": "engine died", "type": "InternalServerError", "code": 500}},
    )
    server, url = _serve_once(body)
    try:
        res = _default_stream_call(url, {}, b"{}", 5.0)
    finally:
        server.shutdown()
    assert res.status == 200 and res.stream_error == "engine died (code 500)"
    assert res.error == "stream error: engine died (code 500)" and "engine died" in res.error_body
    from tre_replayer.engine.stream import TTFT_BASIS, result_fields

    fields = result_fields(res)
    assert fields["stream_error"] == res.stream_error and fields["ttft_basis"] == TTFT_BASIS


def test_the_completions_request_bytes_do_not_change() -> None:
    """The review's constraint: only the parsing changed on the default path."""
    assert json.dumps(api.request_body("m", [1, 2], 8)).encode() == (
        b'{"model": "m", "prompt": [1, 2], "max_tokens": 8, "temperature": 0, "ignore_eos": true, '
        b'"stream": true, "stream_options": {"include_usage": true}}')


def test_a_prompt_file_of_another_api_is_refused(tmp_path) -> None:
    tok = ChatStubTokenizer()
    requests = [ScheduledRequest(request_id="m-0", model="m", scheduled_offset_s=0.0, prompt_tokens=40,
                                 max_output_tokens=8)]
    path = tmp_path / "c.prompts.jsonl"
    store = prompt_store.materialize_prompts(requests, path=path, tokenizer=tok, api="chat")
    assert store.api == "chat" and prompt_store.PromptStore.load(path, api="chat").api == "chat"
    with pytest.raises(ValueError, match="built for the chat API"):
        prompt_store.PromptStore.load(path, api="completions")
    legacy = tmp_path / "old.prompts.jsonl"  # a file from before the api column: completions
    legacy.write_text(json.dumps({"request_id": "m-0", "prompt": "x"}) + "\n", encoding="utf-8")
    assert prompt_store.PromptStore.load(legacy).api == "completions"
    with pytest.raises(ValueError, match="completions API"):
        prompt_store.PromptStore.load(legacy, api="chat")
    # the sender refuses a store of the other endpoint
    with pytest.raises(ValueError, match="holds chat prompts"):
        StreamingHttpSender("http://gw/v1/completions", prompt_store=store)
    StreamingHttpSender(CHAT_URL, api="chat", prompt_store=store).close()

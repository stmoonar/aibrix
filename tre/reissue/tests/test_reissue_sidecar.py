"""Retry / continuation sidecar tests (plan 2026-09-27 D5/D6).

Topology, all in one event loop on localhost: pod A = sidecar A + fake engine A (the pod
that is put to sleep), pod B = sidecar B + fake engine B, and a fake TRE gateway routing
among the pods (honouring x-tre-exclude-pod). The client talks to sidecar A as if the
gateway had routed the request there. The sidecars are served with their production server
options; the fake engines and gateway cancel their handler when the client's connection is
lost (aiohttp's TestServer always does), like vLLM (aborts the request) and Envoy (resets
the upstream request)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from dataclasses import replace

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer, make_mocked_request

from fake_vllm_fork import FakeEngine, FakeGateway, detok, reference_tokens, render_chat, tokenize
from tre_reissue import sidecar as sc
from tre_reissue.sidecar import Config, ReissueSidecar

MODEL = "m"
PROMPT = "one two three four"
HIDDEN = {"X-TRE-Hidden": "1"}


class _OptionsServer(TestServer):
    """A TestServer whose runner gets exactly ``options`` (the sidecar's production server
    options) and nothing else: TestServer itself always passes handler_cancellation=True
    to its runner and drops the keyword arguments of its constructor."""

    def __init__(self, app, options: dict) -> None:
        super().__init__(app)
        self._options = dict(options)

    async def _make_runner(self, **kwargs):
        return web.AppRunner(self.app, **self._options)


class Harness:
    def __init__(self, *, a: dict | None = None, b: dict | None = None, cfg: dict | None = None,
                 routable: tuple[str, ...] = ("pod-b",)) -> None:
        self.engine_a = FakeEngine("a", **(a or {}))
        self.engine_b = FakeEngine("b", **({"token_delay_s": 0.001} | (b or {})))
        self.gateway = FakeGateway()
        self.gateway.routable = list(routable)
        self.cfg_overrides = cfg or {}

    async def __aenter__(self) -> "Harness":
        self.servers = []
        self.ea = await self._serve(self.engine_a.app())
        self.eb = await self._serve(self.engine_b.app())
        self.gw = await self._serve(self.gateway.app())
        base = Config(gateway_url=_url(self.gw), model=MODEL, retry_backoff_s=0.01, retry_max_backoff_s=0.05,
                      probe_interval_s=0.05)
        base = replace(base, **self.cfg_overrides)
        self.sidecar_a = ReissueSidecar(replace(base, upstream_url=_url(self.ea), pod_name="pod-a"))
        self.sidecar_b = ReissueSidecar(replace(base, upstream_url=_url(self.eb), pod_name="pod-b"))
        self.sa = await self._serve(self.sidecar_a.build_app(), sc.server_options(base))
        self.sb = await self._serve(self.sidecar_b.build_app(), sc.server_options(base))
        self.gateway.pods = {"pod-a": _url(self.sa), "pod-b": _url(self.sb)}
        self.http = aiohttp.ClientSession()
        return self

    async def _serve(self, app, options: dict | None = None) -> TestServer:
        server = TestServer(app) if options is None else _OptionsServer(app, options)
        await server.start_server()
        self.servers.append(server)
        return server

    async def __aexit__(self, *exc) -> None:
        await self.http.close()
        for server in reversed(self.servers):
            await server.close()

    def url(self, path: str, server: TestServer | None = None) -> str:
        return _url(server or self.sa) + path

    async def sleep_a(self, *, headers: dict | None = HIDDEN, query: str = "") -> int:
        async with self.http.post(self.url("/sleep" + query), headers=headers or {}) as resp:
            await resp.read()
            return resp.status

    async def post(self, path: str, body: dict, *, headers: dict | None = None, sleep_after: int | None = None,
                   server: TestServer | None = None):
        """(status, headers, raw text). Optionally /sleep pod A once engine A generated
        ``sleep_after`` tokens."""
        sleeper = None
        if sleep_after is not None:
            async def trigger() -> None:
                await self.engine_a.wait_generated(sleep_after)
                assert await self.sleep_a() == 200
            sleeper = asyncio.ensure_future(trigger())
        async with self.http.post(self.url(path, server), json=body, headers=headers or {}) as resp:
            raw = (await resp.read()).decode()
            status, rheaders = resp.status, resp.headers
        if sleeper is not None:
            await sleeper
        return status, rheaders, raw

    async def reference(self, path: str, body: dict) -> str:
        """The same request, uninterrupted, straight on engine B."""
        async with self.http.post(_url(self.eb) + path, json=body) as resp:
            return (await resp.read()).decode()


def _url(server: TestServer) -> str:
    return str(server.make_url("")).rstrip("/")


def parse_sse(raw: str) -> list:
    out = []
    for event in raw.split("\n\n"):
        for line in event.split("\n"):
            if line.startswith("data: "):
                payload = line[6:]
                out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


def comments(raw: str) -> list[str]:
    return [line for event in raw.split("\n\n") for line in event.split("\n") if line.startswith(":")]


def text_of(objs: list, chat: bool) -> str:
    parts = []
    for obj in objs:
        if obj == "[DONE]":
            continue
        for choice in obj.get("choices") or []:
            parts.append(((choice.get("delta") or {}).get("content") or "") if chat else (choice.get("text") or ""))
    return "".join(parts)


def finishes(objs: list) -> list:
    return [c.get("finish_reason") for o in objs if o != "[DONE]" for c in o.get("choices") or []
            if c.get("finish_reason")]


def usage_chunks(objs: list) -> list:
    return [o for o in objs if o != "[DONE]" and not o.get("choices") and isinstance(o.get("usage"), dict)]


def completion_body(max_tokens: int = 24, **extra) -> dict:
    body = {"model": MODEL, "prompt": PROMPT, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    body.update(extra)
    return body


def chat_body(max_tokens: int | None = 24, **extra) -> dict:
    body = {"model": MODEL, "messages": [{"role": "user", "content": "hello there"}], "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    body.update(extra)
    return body


def expected_text(prompt_ids: list[int], n: int) -> str:
    return "".join(detok(t) for t in reference_tokens(prompt_ids, n))


def counts(h: Harness, sidecar: ReissueSidecar | None = None) -> dict:
    s = sidecar or h.sidecar_a
    return {k: s.metrics.total(k) for k in sc.Metrics.KINDS}


# ----------------------------------------------------------------- transparency


@pytest.mark.asyncio
async def test_passthrough_stream_and_non_stream_are_unchanged():
    async with Harness() as h:
        status, headers, raw = await h.post("/v1/completions", completion_body(8))
        assert status == 200
        objs = parse_sse(raw)
        assert text_of(objs, False) == expected_text(tokenize(PROMPT), 8)
        assert objs[-1] == "[DONE]" and finishes(objs) == ["length"]
        assert "x-tre-continued" not in headers and "x-tre-retried" not in headers
        status, headers, raw = await h.post("/v1/chat/completions", chat_body(5, stream=False))
        assert status == 200
        assert json.loads(raw)["choices"][0]["message"]["content"] == expected_text(
            tokenize(render_chat(chat_body()["messages"])), 5)
        assert counts(h) == {"retry": 0, "continue": 0, "failed": 0, "passthrough_abort": 0}
        assert h.gateway.requests == []


@pytest.mark.asyncio
async def test_passthrough_stream_is_byte_identical():
    async with Harness() as h:
        _, _, via = await h.post("/v1/completions", completion_body(6))
        direct = await h.reference("/v1/completions", completion_body(6))
        # ids differ per request; everything else is the engine's bytes
        strip = lambda raw: [{k: v for k, v in o.items() if k != "id"} if o != "[DONE]" else o for o in parse_sse(raw)]
        assert strip(via) == strip(direct)


# ------------------------------------------------------------------- retry path


@pytest.mark.asyncio
async def test_engine_sleeping_503_is_retried_unchanged_through_the_gateway():
    async with Harness() as h:
        h.engine_a.put_to_sleep()  # the sidecar does not know (e.g. a restart)
        body = completion_body(6)
        status, headers, raw = await h.post("/v1/completions", body,
                                            headers={"x-tre-exclude-pod": "pod-z", "routing-strategy": "x"})
        assert status == 200
        assert headers["x-tre-retried"] == "1"
        assert text_of(parse_sse(raw), False) == expected_text(tokenize(PROMPT), 6)
        (fwd,) = h.gateway.requests
        assert json.loads(fwd["raw"]) == body  # the original body, unchanged
        assert fwd["exclude"] == {"pod-z", "pod-a"}
        assert fwd["headers"]["x-tre-reissue-depth"] == "1"
        assert fwd["headers"]["routing-strategy"] == "x"
        assert fwd["target"] == "pod-b"
        assert counts(h)["retry"] == 1
        # the 503 re-synced the sidecar's sleeping mark
        assert h.sidecar_a.state.sleeping is True


@pytest.mark.asyncio
async def test_engine_sleeping_error_inside_the_stream_is_retried():
    async with Harness(a={"reject_in_stream": True}) as h:
        h.engine_a.put_to_sleep()
        status, headers, raw = await h.post("/v1/chat/completions", chat_body(4))
        assert status == 200 and headers["x-tre-retried"] == "1"
        objs = parse_sse(raw)
        assert not any(isinstance(o, dict) and "error" in o for o in objs)
        assert finishes(objs) == ["length"]
        assert counts(h)["retry"] == 1


@pytest.mark.asyncio
async def test_known_sleeping_pod_forwards_without_touching_the_engine():
    async with Harness() as h:
        assert await h.sleep_a() == 200
        before = len(h.engine_a.generation_requests())
        status, headers, raw = await h.post("/v1/embeddings", {"model": MODEL, "input": "x"})
        assert status == 200 and json.loads(raw)["served_by"] == "b"
        status, _, raw = await h.post("/v1/completions", completion_body(3, stream=False))
        assert status == 200 and json.loads(raw)["choices"][0]["finish_reason"] == "length"
        assert len(h.engine_a.generation_requests()) == before
        assert counts(h)["retry"] == 2


@pytest.mark.asyncio
async def test_gateway_without_a_routable_pod_gives_503_with_retry_after():
    async with Harness(routable=()) as h:
        h.engine_a.put_to_sleep()
        status, headers, raw = await h.post("/v1/completions", completion_body(3))
        assert status == 503
        assert headers["Retry-After"] == "1"
        assert json.loads(raw)["error"]["type"] == "ServiceUnavailable"
        assert len(h.gateway.requests) == h.sidecar_a.cfg.retry_attempts
        assert counts(h)["failed"] == 1


@pytest.mark.asyncio
async def test_retry_hop_limit():
    async with Harness() as h:
        h.engine_a.put_to_sleep()
        status, headers, _ = await h.post("/v1/completions", completion_body(3),
                                          headers={"x-tre-reissue-depth": "3"})
        assert status == 503 and headers["Retry-After"] == "1"
        assert h.gateway.requests == []
        assert h.sidecar_a.metrics.reissue[("failed", "depth_limit")] == 1


@pytest.mark.asyncio
async def test_a_sidecar_error_is_not_retried_by_the_sidecar_before_it():
    """T5, one retry layer (I5): A and B both sleep; A's retry reaches B, which retries
    on its own and gets the gateway's 503 (no routable pod) retry_attempts times. B's
    final 503 is a sidecar error: A does not retry it. The sends per client request are
    1 (A) + retry_attempts (B), not retry_attempts x (1 + retry_attempts)."""
    async with Harness() as h:
        assert await h.sleep_a() == 200
        async with h.http.post(h.url("/sleep", h.sb), headers=HIDDEN) as resp:
            assert resp.status == 200
        status, headers, raw = await h.post("/v1/completions", completion_body(3))
        assert status == 503 and headers["Retry-After"] == "1"
        assert json.loads(raw)["error"]["message"].startswith("tre-reissue sidecar:")
        sends = [r["headers"]["x-tre-reissue-depth"] for r in h.gateway.requests]
        attempts = h.sidecar_a.cfg.retry_attempts
        assert sends.count("1") == 1  # A: one send, B's answer is final
        assert sends.count("2") == attempts  # B: its own bounded retries of the gateway's 503
        assert len(sends) == 1 + attempts
        assert h.sidecar_a.metrics.reissue == {("failed", "retry_exhausted"): 1}


# ------------------------------------------------------------- continuation path


@pytest.mark.asyncio
async def test_completion_stream_continued_with_token_ids_exact_seam():
    async with Harness(a={"hold_at": 7}) as h:
        body = completion_body(24)
        status, headers, raw = await h.post("/v1/completions", body, sleep_after=7)
        assert status == 200
        objs = parse_sse(raw)
        prompt_ids = tokenize(PROMPT)
        assert text_of(objs, False) == expected_text(prompt_ids, 24)  # exact seam
        assert text_of(objs, False) == text_of(parse_sse(await h.reference("/v1/completions", body)), False)
        assert finishes(objs) == ["length"]  # no abort reaches the client
        assert objs.count("[DONE]") == 1 and objs[-1] == "[DONE]"
        ids = {o["id"] for o in objs if o != "[DONE]"}
        assert len(ids) == 1 and next(iter(ids)).startswith("cmpl-")
        assert {o["created"] for o in objs if o != "[DONE]"} == {1700000000}
        (usage,) = usage_chunks(objs)
        assert usage["usage"] == {"prompt_tokens": len(prompt_ids), "completion_tokens": 24,
                                  "total_tokens": len(prompt_ids) + 24}
        final = [o for o in objs if o != "[DONE]" and finishes([o])][0]
        assert final["tre_continued"] == 1
        assert ": x-tre-continued: 1" in comments(raw)
        assert not any("generated_token_ids" in json.dumps(o) for o in objs)
        # what went to the gateway: a token-id completion with the reduced budget
        (cont,) = h.gateway.requests
        assert cont["path"] == "/v1/completions"
        assert cont["body"]["prompt"] == prompt_ids + reference_tokens(prompt_ids, 7)
        assert cont["body"]["max_tokens"] == 24 - 7
        assert cont["body"]["temperature"] == 0 and cont["body"]["stream"] is True
        assert cont["exclude"] == {"pod-a"} and cont["target"] == "pod-b"
        assert counts(h)["continue"] == 1 and counts(h)["failed"] == 0
        # the sleep broke the upstream stream, the client stayed: not a client cancel
        assert "client_cancel" not in h.sidecar_a.metrics.events


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_continuation_log_line_reports_the_gap_in_milliseconds(stream, capsys):
    async with Harness(a={"hold_at": 7}) as h:
        status, _, _ = await h.post("/v1/completions", completion_body(24, stream=stream), sleep_after=7)
        assert status == 200
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    (cont,) = [r for r in records if r.get("event") == "tre_reissue" and r.get("kind") == "continue"]
    assert "gap_s" not in cont  # renamed 2026-09-30: the value always was milliseconds
    assert isinstance(cont["gap_ms"], float) and cont["gap_ms"] >= 0
    mode, other = ("stream", "nonstream") if stream else ("nonstream", "stream")
    gap = h.sidecar_a.metrics.gap
    assert gap[mode].count == 1 and gap[other].count == 0  # the two modes never mix
    # the histogram stays in seconds: the logged ms value / 1000 (rounding to 0.1 ms)
    assert abs(gap[mode].sum * 1000.0 - cont["gap_ms"]) <= 0.06


@pytest.mark.asyncio
async def test_chat_stream_continued_as_token_id_completion():
    async with Harness(a={"hold_at": 5}) as h:
        body = chat_body(None, stream_options={"include_usage": False})  # no limit: max_model_len - prompt
        body["max_tokens"] = 16
        status, _, raw = await h.post("/v1/chat/completions", body, sleep_after=5)
        assert status == 200
        objs = parse_sse(raw)
        prompt_ids = tokenize(render_chat(body["messages"]))
        assert text_of(objs, True) == expected_text(prompt_ids, 16)
        roles = [c["delta"]["role"] for o in objs if o != "[DONE]" for c in o["choices"] if "role" in c["delta"]]
        assert roles == ["assistant"]
        assert {o["object"] for o in objs if o != "[DONE]"} == {"chat.completion.chunk"}
        assert len({o["id"] for o in objs if o != "[DONE]"}) == 1
        assert usage_chunks(objs) == []  # the client did not ask for usage
        assert finishes(objs) == ["length"]
        (cont,) = h.gateway.requests
        assert cont["path"] == "/v1/completions"
        assert cont["body"]["prompt"] == prompt_ids + reference_tokens(prompt_ids, 5)
        assert "messages" not in cont["body"] and cont["body"]["max_tokens"] == 11


@pytest.mark.asyncio
async def test_chat_without_a_limit_uses_max_model_len():
    async with Harness(a={"hold_at": 3}, b={"token_delay_s": 0}) as h:
        body = chat_body(None, stream=False)
        body.pop("stream_options")
        task = asyncio.ensure_future(h.post("/v1/chat/completions", body))
        await h.engine_a.wait_generated(3)
        assert await h.sleep_a() == 200
        status, headers, raw = await task
        assert status == 200 and headers["x-tre-continued"] == "1"
        (cont,) = h.gateway.requests
        prompt_len = len(tokenize(render_chat(body["messages"])))
        from fake_vllm_fork import MAX_MODEL_LEN
        assert cont["body"]["max_tokens"] == MAX_MODEL_LEN - prompt_len - 3


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [False, True])
async def test_non_stream_continuation_merges_text_usage_and_header(chat):
    async with Harness(a={"hold_at": 4}) as h:
        path = "/v1/chat/completions" if chat else "/v1/completions"
        body = (chat_body if chat else completion_body)(12, stream=False)
        body.pop("stream_options")
        task = asyncio.ensure_future(h.post(path, body))
        await h.engine_a.wait_generated(4)
        assert await h.sleep_a() == 200
        status, headers, raw = await task
        assert status == 200 and headers["x-tre-continued"] == "1"
        obj = json.loads(raw)
        prompt_ids = tokenize(render_chat(body["messages"])) if chat else tokenize(PROMPT)
        choice = obj["choices"][0]
        text = choice["message"]["content"] if chat else choice["text"]
        assert text == expected_text(prompt_ids, 12)
        assert choice["finish_reason"] == "length"
        assert obj["usage"]["prompt_tokens"] == len(prompt_ids) and obj["usage"]["completion_tokens"] == 12
        assert "token_ids" not in choice and "prompt_token_ids" not in obj and "prompt_token_ids" not in choice
        assert obj["object"] == ("chat.completion" if chat else "text_completion")
        assert counts(h)["continue"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("include", [False, True])
async def test_stop_string_spanning_the_seam(include):
    prompt_ids = tokenize(PROMPT)
    full = expected_text(prompt_ids, 20)
    seam = len(expected_text(prompt_ids, 6))
    stop = full[seam - 3: seam + 3]
    assert stop not in full[: seam] and full.find(stop) == seam - 3
    async with Harness(a={"hold_at": 6}) as h:
        body = completion_body(20, stop=[stop], include_stop_str_in_output=include)
        status, _, raw = await h.post("/v1/completions", body, sleep_after=6)
        assert status == 200
        objs = parse_sse(raw)
        reference = parse_sse(await h.reference("/v1/completions", body))
        expect = full[: seam + 3] if include else full[: seam - 3]
        assert text_of(reference, False) == expect  # an uninterrupted engine stops there
        assert text_of(objs, False) == expect
        assert finishes(objs) == ["stop"]
        assert objs[-1] == "[DONE]"
        assert h.sidecar_a.metrics.events.get("stop_at_seam") == 1


@pytest.mark.asyncio
async def test_stop_string_after_the_seam_is_left_to_the_continuation_engine():
    prompt_ids = tokenize(PROMPT)
    full = expected_text(prompt_ids, 20)
    start = len(expected_text(prompt_ids, 12))
    stop = full[start: start + 4]
    assert full.find(stop) == start
    async with Harness(a={"hold_at": 6}) as h:
        body = completion_body(20, stop=stop)
        _, _, raw = await h.post("/v1/completions", body, sleep_after=6)
        objs = parse_sse(raw)
        assert text_of(objs, False) == full[:start] and finishes(objs) == ["stop"]
        assert "stop_at_seam" not in h.sidecar_a.metrics.events


@pytest.mark.asyncio
async def test_abort_before_any_output_is_a_pure_retry():
    async with Harness(a={"hold_at": 0}) as h:
        body = completion_body(5)
        task = asyncio.ensure_future(h.post("/v1/completions", body))
        while not h.engine_a.active:
            await asyncio.sleep(0.001)
        assert await h.sleep_a() == 200
        status, headers, raw = await task
        assert status == 200 and headers["x-tre-retried"] == "1"
        assert text_of(parse_sse(raw), False) == expected_text(tokenize(PROMPT), 5)
        (fwd,) = h.gateway.requests
        assert json.loads(fwd["raw"]) == body
        assert h.sidecar_a.metrics.reissue[("retry", "abort_before_output")] == 1


@pytest.mark.asyncio
async def test_non_continuable_abort_is_passed_through_and_counted():
    async with Harness(a={"hold_at": 3}) as h:
        body = completion_body(10, logprobs=1)
        status, _, raw = await h.post("/v1/completions", body, sleep_after=3)
        assert status == 200
        assert finishes(parse_sse(raw)) == ["abort"]
        assert h.gateway.requests == []
        assert h.sidecar_a.metrics.reissue[("passthrough_abort", "non_continuable_logprobs")] == 1


@pytest.mark.asyncio
async def test_non_continuable_non_stream_abort_is_retried_from_scratch():
    """Nothing of a non-streaming response reached the client: resending the original
    is exact even for a request that cannot be continued."""
    async with Harness(a={"hold_at": 3}) as h:
        body = completion_body(6, stream=False, n=2)
        body.pop("stream_options")
        task = asyncio.ensure_future(h.post("/v1/completions", body))
        await h.engine_a.wait_generated(3)
        assert await h.sleep_a() == 200
        status, headers, raw = await task
        assert status == 200 and headers["x-tre-retried"] == "1"
        assert json.loads(raw)["choices"][0]["finish_reason"] == "length"
        (fwd,) = h.gateway.requests
        assert json.loads(fwd["raw"]) == body
        assert h.sidecar_a.metrics.reissue[("retry", "abort_non_continuable_n")] == 1


@pytest.mark.asyncio
async def test_abort_while_awake_is_passed_through():
    async with Harness(a={"hold_at": 2}) as h:
        async def engine_side_abort():
            await h.engine_a.wait_generated(2)
            for req in h.engine_a.active.values():
                req.do_abort()
        task = asyncio.ensure_future(engine_side_abort())
        _, _, raw = await h.post("/v1/completions", completion_body(8))
        await task
        assert finishes(parse_sse(raw)) == ["abort"]
        assert h.sidecar_a.metrics.reissue[("passthrough_abort", "not_sleeping")] == 1


@pytest.mark.asyncio
async def test_continuation_depth_limit_passes_the_abort_through():
    async with Harness(a={"hold_at": 3}) as h:
        status, _, raw = await h.post("/v1/completions", completion_body(8), sleep_after=3,
                                      headers={"x-tre-reissue-depth": "3"})
        assert status == 200 and finishes(parse_sse(raw)) == ["abort"]
        assert h.gateway.requests == []
        assert h.sidecar_a.metrics.reissue[("failed", "depth_limit")] == 1


@pytest.mark.asyncio
async def test_continuation_with_the_gateway_down_ends_as_abort():
    async with Harness(a={"hold_at": 3}, routable=()) as h:
        status, _, raw = await h.post("/v1/completions", completion_body(8), sleep_after=3)
        objs = parse_sse(raw)
        assert status == 200 and finishes(objs) == ["abort"] and objs[-1] == "[DONE]"
        assert text_of(objs, False) == expected_text(tokenize(PROMPT), 3)
        assert len(h.gateway.requests) == h.sidecar_a.cfg.retry_attempts
        assert h.sidecar_a.metrics.reissue[("failed", "continuation_unavailable")] == 1


@pytest.mark.asyncio
async def test_missing_token_ids_cannot_be_continued():
    async with Harness(a={"hold_at": 3, "abort_ids": False}) as h:
        _, _, raw = await h.post("/v1/completions", completion_body(8), sleep_after=3)
        assert finishes(parse_sse(raw)) == ["abort"]
        assert h.sidecar_a.metrics.reissue[("failed", "no_token_ids")] == 1


@pytest.mark.asyncio
async def test_nested_continuation_counts_segments():
    """Pod B is put to sleep while it runs A's continuation; B's sidecar continues it on
    pod C; the client sees one stream and x-tre-continued 2."""
    async with Harness(a={"hold_at": 4}, b={"hold_at": 3}, routable=("pod-b", "pod-c")) as h:
        engine_c = FakeEngine("c", token_delay_s=0.001)
        ec = await h._serve(engine_c.app())
        sidecar_c = ReissueSidecar(replace(h.sidecar_a.cfg, upstream_url=_url(ec), pod_name="pod-c"))
        h.gateway.pods["pod-c"] = _url(await h._serve(sidecar_c.build_app(), sc.server_options(sidecar_c.cfg)))

        async def sleep_b():
            await h.engine_b.wait_generated(3)
            async with h.http.post(_url(h.sb) + "/sleep", headers=HIDDEN) as resp:
                assert resp.status == 200
        task = asyncio.ensure_future(sleep_b())
        _, _, raw = await h.post("/v1/completions", completion_body(20), sleep_after=4)
        await task
        objs = parse_sse(raw)
        assert text_of(objs, False) == expected_text(tokenize(PROMPT), 20)
        final = [o for o in objs if o != "[DONE]" and finishes([o])][0]
        assert final["tre_continued"] == 2 and ": x-tre-continued: 2" in comments(raw)
        assert usage_chunks(objs)[0]["usage"]["completion_tokens"] == 20
        assert h.gateway.requests[-1]["exclude"] == {"pod-a", "pod-b"}
        assert h.gateway.requests[-1]["headers"]["x-tre-reissue-depth"] == "2"


@pytest.mark.asyncio
async def test_continuation_after_mode_wait_budget_expiry():
    """SM: mode=wait first (drain), then mode=abort once the budget is spent."""
    async with Harness(a={"hold_at": 4}) as h:
        task = asyncio.ensure_future(h.post("/v1/completions", completion_body(12)))
        await h.engine_a.wait_generated(4)
        wait_call = asyncio.ensure_future(h.sleep_a(query="?mode=wait"))
        await asyncio.sleep(0.05)
        assert not wait_call.done()  # the drain is waiting for the held request
        wait_call.cancel()
        assert await h.sleep_a(query="?mode=abort") == 200
        status, _, raw = await task
        assert text_of(parse_sse(raw), False) == expected_text(tokenize(PROMPT), 12)


# ------------------------------------------------------------ control endpoints


# ------------------------------------------------------------ client disconnect


async def _until(condition, timeout_s: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not condition():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.001)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_client_disconnect_closes_the_upstream_request(stream):
    """The client goes away while its request waits on the engine (queued / prefill:
    no token yet): the sidecar closes its upstream request and the engine aborts it.
    Nothing is retried or counted as a reissue."""
    async with Harness(a={"hold_at": 0}) as h:
        body = completion_body(8, stream=stream)
        if not stream:
            body.pop("stream_options")
        task = asyncio.ensure_future(h.post("/v1/completions", body))
        await _until(lambda: h.engine_a.active)
        task.cancel()  # the client closes its connection
        with pytest.raises(asyncio.CancelledError):
            await task
        await _until(lambda: h.engine_a.disconnected == 1)
        assert not h.engine_a.active
        assert h.sidecar_a.metrics.events.get("client_cancel") == 1
        assert counts(h) == {"retry": 0, "continue": 0, "failed": 0, "passthrough_abort": 0}
        assert h.gateway.requests == []


@pytest.mark.asyncio
async def test_client_disconnect_during_a_continuation_closes_the_continuation():
    """Pod A slept mid-stream and the request is being continued on pod B; the client
    goes away: B's request is closed too (through the gateway), and A accounts the
    request once, as passthrough_abort / client_gone."""
    async with Harness(a={"hold_at": 3}, b={"hold_at": 2}) as h:
        task = asyncio.ensure_future(h.post("/v1/completions", completion_body(12)))
        await h.engine_a.wait_generated(3)
        assert await h.sleep_a() == 200
        await h.engine_b.wait_generated(2)  # B runs the continuation
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _until(lambda: h.engine_b.disconnected == 1)
        assert not h.engine_b.active
        assert h.sidecar_a.metrics.reissue == {("passthrough_abort", "client_gone"): 1}
        assert counts(h, h.sidecar_b) == {"retry": 0, "continue": 0, "failed": 0, "passthrough_abort": 0}


@pytest.mark.asyncio
async def test_client_leaving_after_done_is_a_completed_request():
    """The client got the whole stream ([DONE]) and closes before the engine ends its
    response: a completed request, not a client cancel, and its added time is kept."""
    async with Harness(a={"hold_eof": True}) as h:
        async with h.http.post(h.url("/v1/completions"), json=completion_body(4)) as resp:
            async for line in resp.content:
                if line.startswith(b"data: [DONE]"):
                    break  # leaving the block closes the connection before EOF
        await _until(lambda: h.engine_a.disconnected == 1)
        assert "client_cancel" not in h.sidecar_a.metrics.events
        assert _hist(h).count == 1


@pytest.mark.asyncio
async def test_sleep_caller_leaving_mid_call_still_marks_the_pod_sleeping():
    """The service-manager disconnects while the engine is still in /sleep: the call runs
    to its end and the sleeping mark follows the engine's answer (no probe needed)."""
    async with Harness(a={"hold_sleep": True}, cfg={"probe_interval_s": 60.0}) as h:
        task = asyncio.ensure_future(h.sleep_a())
        await _until(lambda: h.sidecar_a.state.pending == 1 and h.engine_a.sleeping)
        task.cancel()  # the caller closes its connection
        with pytest.raises(asyncio.CancelledError):
            await task
        await _until(lambda: not h.sa.runner.server.connections)  # the sidecar saw it go
        h.engine_a.release_sleep()
        await _until(lambda: h.sidecar_a.state.pending == 0)
        assert h.sidecar_a.state.sleeping is True


@pytest.mark.asyncio
async def test_sleep_requires_the_hidden_header():
    async with Harness() as h:
        assert await h.sleep_a(headers={}) == 409
        assert not any(r["path"] == "/sleep" for r in h.engine_a.requests)
        assert h.sidecar_a.state.sleeping is False
        assert await h.sleep_a() == 200
        sleep_call = [r for r in h.engine_a.requests if r["path"] == "/sleep"][0]
        assert "X-TRE-Hidden" not in sleep_call["headers"]
        assert h.sidecar_a.state.sleeping is True
        async with h.http.post(h.url("/wake_up")) as resp:
            assert resp.status == 200
        assert h.sidecar_a.state.sleeping is False
        async with h.http.get(h.url("/tre-reissue/metrics")) as resp:
            text = await resp.text()
        assert 'event="sleep_rejected_not_hidden"} 1' in text


@pytest.mark.asyncio
async def test_other_paths_pass_through():
    async with Harness() as h:
        async with h.http.get(h.url("/health")) as resp:
            assert resp.status == 200
        async with h.http.get(h.url("/metrics")) as resp:
            assert "vllm:num_requests_running" in await resp.text()
        async with h.http.get(h.url("/is_sleeping")) as resp:
            assert (await resp.json()) == {"is_sleeping": False}
        async with h.http.get(h.url("/v1/models")) as resp:
            assert (await resp.json())["data"][0]["max_model_len"] > 0


@pytest.mark.asyncio
async def test_probe_resyncs_the_sleeping_mark():
    async with Harness() as h:
        h.engine_a.put_to_sleep()
        for _ in range(100):
            if h.sidecar_a.state.sleeping:
                break
            await asyncio.sleep(0.01)
        assert h.sidecar_a.state.sleeping is True
        assert h.sidecar_a.metrics.events["state_corrected_to_sleeping"] == 1


@pytest.mark.asyncio
async def test_metrics_render_kinds_and_overhead_histogram():
    async with Harness(a={"hold_at": 2}) as h:
        await h.post("/v1/completions", completion_body(6), sleep_after=2)
        async with h.http.get(h.url("/tre-reissue/metrics")) as resp:
            text = await resp.text()
        assert 'tre_reissue_total{model="m",kind="continue",reason="abort_sleep"} 1' in text
        assert "tre_reissue_proxy_added_seconds_count" in text
        assert 'tre_reissue_gap_seconds_count{model="m",mode="stream"} 1' in text
        assert 'tre_reissue_gap_seconds_count{model="m",mode="nonstream"} 0' in text


# ------------------------------------------------------------------ pure helpers


#: Shared with the gateway plugin's TestTRENonContinuableContract
#: (pkg/plugins/gateway/tre_transparent_sleep_test.go): one list of cases, two implementations.
CONTRACT = Path(__file__).resolve().parents[1] / "contract" / "non_continuable_cases.json"
REASONS = {"endpoint", "body", "n", "logprobs", "echo", "beam_search", "tools", "structured_output", "prompt_form"}


def _contract_cases() -> list[dict]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))["cases"]


def test_non_continuable_contract_is_well_formed():
    cases = _contract_cases()
    assert len(cases) >= 40
    names = [c["name"] for c in cases]
    assert len(names) == len(set(names)), "case names must be unique"
    for c in cases:
        assert ("body" in c) != ("body_raw" in c), c["name"]
        assert c["want"] is (c["reason"] is not None), c["name"]
        assert c["reason"] is None or c["reason"] in REASONS, c["name"]
    assert {c["reason"] for c in cases if c["reason"]} == REASONS, "every reason needs a case"
    assert any("?" in c["path"] for c in cases if not c["want"]), "a query-string path must stay continuable"


@pytest.mark.parametrize("case", _contract_cases(), ids=lambda c: c["name"])
def test_non_continuable_contract(case):
    cfg = Config()
    raw = case["body_raw"].encode() if "body_raw" in case else json.dumps(case["body"]).encode()
    # What _generation classifies: aiohttp's request.path (no query) and parse_body(raw).
    path = make_mocked_request("POST", case["path"]).path
    assert sc.non_continuable_reason(path, sc.parse_body(raw), cfg) == case["reason"]


def test_build_continuation_budgets():
    cfg = Config()
    plan = sc.build_continuation(cfg.completions_path, {"prompt": "x", "min_tokens": 5, "seed": 7, "echo": False},
                                 [1, 2], [9, 9, 9], stream=True, max_model_len=None, cfg=cfg)
    assert plan.body["max_tokens"] == 16 - 3  # vLLM's default of 16 made explicit
    assert plan.body["min_tokens"] == 2 and plan.body["seed"] == 7 and "echo" not in plan.body
    assert plan.body["prompt"] == [1, 2, 9, 9, 9] and plan.body["stream_options"] == {"include_usage": True}
    assert sc.build_continuation(cfg.completions_path, {"prompt": "x", "max_tokens": 3}, [1], [5, 5, 5],
                                 stream=False, max_model_len=None, cfg=cfg) is None
    chat = sc.build_continuation(cfg.chat_path, {"messages": [], "max_completion_tokens": 10, "top_k": 3,
                                                 "tools": None, "chat_template_kwargs": {}},
                                 [1, 2, 3], [4], stream=False, max_model_len=None, cfg=cfg)
    assert chat.path == cfg.completions_path and chat.as_chat
    assert chat.body == {"top_k": 3, "prompt": [1, 2, 3, 4], "max_tokens": 9, "stream": False}
    with pytest.raises(sc.NoBudget):
        sc.build_continuation(cfg.chat_path, {"messages": []}, [1], [2], stream=True, max_model_len=None, cfg=cfg)
    same = sc.build_continuation(cfg.chat_path, {"messages": [], "stream": True}, [1], [], stream=True,
                                 max_model_len=None, cfg=cfg)
    assert same.path == cfg.chat_path and same.body["messages"] == [] and not same.as_chat


def test_find_seam_stop():
    assert sc.find_seam_stop("abXY", "Zcd", ["XYZ"], False) == 2
    assert sc.find_seam_stop("abXY", "Zcd", ["XYZ"], True) == 5
    assert sc.find_seam_stop("abXY", "cd", ["XYZ"], False) is None
    assert sc.find_seam_stop("ab", "XYZ", ["XYZ"], False) is None  # wholly after the seam
    assert sc.find_seam_stop("aXY", "Zb", ["YZ", "XYZ"], False) == 1  # earliest start wins


def test_merge_exclude_and_config_from_env():
    assert sc.merge_exclude(["pod-z, pod-a", "pod-y"], "pod-a") == "pod-z,pod-a,pod-y"
    cfg = Config.from_env({"TRE_GATEWAY_URL": "http://gw.ns.svc.cluster.local:80/", "POD_NAME": "p",
                           "TRE_REISSUE_UPSTREAM_URL": "http://127.0.0.1:9001",
                           "TRE_REISSUE_GENERATED_IDS_FIELD": "gen_ids", "TRE_REISSUE_SLEEP_PATHS": "/sleep",
                           "TRE_REISSUE_ENABLED": "false", "TRE_REISSUE_MAX_DEPTH": "2"})
    assert cfg.gateway_url == "http://gw.ns.svc.cluster.local:80" and cfg.pod_name == "p"
    assert cfg.upstream_url == "http://127.0.0.1:9001" and cfg.generated_ids_field == "gen_ids"
    assert cfg.sleep_paths == ("/sleep",) and cfg.enabled is False and cfg.max_depth == 2
    assert Config.from_env({}).gateway_url == sc.DEFAULT_GATEWAY_URL
    with pytest.raises(ValueError):
        Config.from_env({"TRE_GATEWAY_URL": "10.0.0.1:80"})


class _FakeResource:
    RLIMIT_NOFILE = 7
    RLIM_INFINITY = -1

    def __init__(self, soft: int, hard: int, *, fail: bool = False) -> None:
        self.soft, self.hard, self.fail = soft, hard, fail

    def getrlimit(self, which: int) -> tuple[int, int]:
        assert which == self.RLIMIT_NOFILE
        return self.soft, self.hard

    def setrlimit(self, which: int, limits: tuple[int, int]) -> None:
        if self.fail:
            raise ValueError("not allowed to raise the current limit")
        soft, hard = limits
        assert which == self.RLIMIT_NOFILE and hard == self.hard
        assert hard == self.RLIM_INFINITY or soft <= hard
        self.soft = soft


@pytest.mark.parametrize("soft, hard, target, after", [
    (1024, 524288, 65535, 65535),        # Docker >= 25 / containerd >= 2.0 default: raised
    (1024, 4096, 65535, 4096),           # capped at the hard limit
    (1048576, 1048576, 65535, 1048576),  # already higher: never lowered
    (1024, -1, 65535, 65535),            # unlimited hard limit
    (1024, 524288, 0, 1024),             # 0: left alone
])
def test_open_files_limit_is_raised_up_to_the_hard_limit(monkeypatch, soft, hard, target, after):
    fake = _FakeResource(soft, hard)
    monkeypatch.setattr(sc, "resource", fake)
    sc.raise_nofile_limit(target)
    assert fake.soft == after


def test_open_files_limit_failure_only_warns(monkeypatch, capsys):
    fake = _FakeResource(1024, 524288, fail=True)
    monkeypatch.setattr(sc, "resource", fake)
    sc.raise_nofile_limit(65535)  # does not raise
    assert fake.soft == 1024
    (line,) = [json.loads(x) for x in capsys.readouterr().out.splitlines() if "tre_reissue_nofile" in x]
    assert line["level"] == "WARNING" and line["before"] == 1024 and line["after"] == 1024


# ------------------------------------------- B4: tre_reissue_proxy_added_seconds


def _hist(h, name: str = "overhead"):
    return getattr(h.sidecar_a.metrics, name)


@pytest.mark.asyncio
async def test_proxy_added_time_excludes_the_generation_of_a_non_streaming_request():
    # 30 tokens x 20 ms: the upstream answers after ~0.6 s; the sidecar adds ~ms.
    async with Harness(a={"token_delay_s": 0.02}) as h:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        status, _, raw = await h.post("/v1/completions", completion_body(30, stream=False))
        elapsed = loop.time() - t0
        assert status == 200 and json.loads(raw)["choices"][0]["finish_reason"] == "length"
        added = _hist(h)
        assert added.count == 1
        assert elapsed > 0.5
        assert added.sum < 0.1  # was ~elapsed: it included the wait for the response headers
        assert _hist(h, "forward").count == 1 and _hist(h, "relay").count == 1
        assert added.sum == pytest.approx(_hist(h, "forward").sum + _hist(h, "relay").sum)


@pytest.mark.asyncio
async def test_proxy_added_time_excludes_the_gaps_between_stream_chunks():
    async with Harness(a={"token_delay_s": 0.02}) as h:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        status, _, raw = await h.post("/v1/completions", completion_body(30))
        elapsed = loop.time() - t0
        assert status == 200 and finishes(parse_sse(raw)) == ["length"]
        added = _hist(h)
        assert added.count == 1  # one observation per request, at the end of the stream
        assert elapsed > 0.5
        assert added.sum < 0.1
        assert _hist(h, "relay").sum > 0  # the per-chunk relay work is counted


@pytest.mark.asyncio
async def test_proxy_added_time_counts_only_locally_answered_requests():
    async with Harness() as h:
        assert await h.sleep_a() == 200
        status, _, _ = await h.post("/v1/completions", completion_body(3, stream=False))
        assert status == 200
        assert counts(h)["retry"] == 1
        assert _hist(h).count == 0  # retried through the gateway: not observed


@pytest.mark.asyncio
async def test_proxy_added_metrics_are_rendered_with_their_definition():
    async with Harness() as h:
        await h.post("/v1/completions", completion_body(3, stream=False))
        async with h.http.get(h.url("/tre-reissue/metrics")) as resp:
            text = await resp.text()
        assert "tre_reissue_proxy_added_seconds_count{model=\"m\"} 1" in text
        assert "tre_reissue_proxy_forward_seconds_count{model=\"m\"} 1" in text
        assert "tre_reissue_proxy_relay_seconds_count{model=\"m\"} 1" in text
        help_line = next(line for line in text.splitlines()
                         if line.startswith("# HELP tre_reissue_proxy_added_seconds"))
        assert "forward" in help_line and "relay" in help_line and "Excludes waiting on upstream" in help_line


def test_added_time_accumulates_only_between_start_and_stop(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(sc.time, "perf_counter", lambda: now[0])
    added = sc.AddedTime()
    now[0] += 0.002
    added.forwarded()
    now[0] += 5.0  # waiting on upstream
    added.start()
    now[0] += 0.001
    added.stop()
    added.stop()  # idempotent
    now[0] += 3.0  # waiting for the next chunk
    added.start()
    now[0] += 0.0005
    added.stop()
    assert added.forward == pytest.approx(0.002)
    assert added.relay == pytest.approx(0.0015)
    assert added.total == pytest.approx(0.0035)
    assert added.relayed is True
    added.discard()
    metrics = sc.Metrics("m")
    metrics.observe_added(added)
    assert metrics.overhead.count == 0  # discarded (retried) requests are not observed

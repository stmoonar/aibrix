"""Reissue sidecar tests: a fake vLLM behind the sidecar (engine "a") and a second fake
standing in for the tre-v2 gateway (engine "b", which answers with a target-pod header).
Everything runs on localhost inside one event loop; nothing touches the cluster."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import aiohttp
import pytest
from aiohttp.test_utils import TestServer

from reissue_fake_vllm import FakeVllm, render_chat
from tre_reissue import sidecar as sc
from tre_reissue.sidecar import Config, ReissueSidecar

MODEL = "dsqwen-7b"
PROMPT = "one two three four"
GW_POD = "10.9.9.9:8000"


class Harness:
    def __init__(self, **cfg_overrides) -> None:
        self.cfg_overrides = cfg_overrides
        a_opts = {k[2:]: cfg_overrides.pop(k) for k in list(cfg_overrides) if k.startswith("a_")}
        b_opts = {k[2:]: cfg_overrides.pop(k) for k in list(cfg_overrides) if k.startswith("b_")}
        a_opts.setdefault("token_delay_s", a_opts.pop("delay", 0.01))
        self.a = FakeVllm("a", **a_opts)
        self.b = FakeVllm("b", token_delay_s=0.002, target_pod=GW_POD, **b_opts)

    async def __aenter__(self) -> "Harness":
        self.a_srv = TestServer(self.a.app())
        self.b_srv = TestServer(self.b.app())
        await self.a_srv.start_server()
        await self.b_srv.start_server()
        cfg = Config(
            upstream_url=str(self.a_srv.make_url("")).rstrip("/"),
            gateway_url=str(self.b_srv.make_url("")).rstrip("/"),
            model=MODEL,
            pod_name="pod-a",
        )
        cfg = replace(cfg, **self.cfg_overrides)
        self.sidecar = ReissueSidecar(cfg)
        self.s_srv = TestServer(self.sidecar.build_app())
        await self.s_srv.start_server()
        self.http = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.http.close()
        await self.s_srv.close()
        await self.a_srv.close()
        await self.b_srv.close()

    def url(self, path: str) -> str:
        return str(self.s_srv.make_url(path))

    async def sleep(self, *, expect: int = 200, hidden: bool = True) -> dict:
        headers = {"X-TRE-Hidden": "1"} if hidden else {}
        async with self.http.post(self.url("/sleep"), headers=headers) as resp:
            assert resp.status == expect
            return {"status": resp.status, "body": await resp.text()}

    async def wake(self) -> None:
        async with self.http.post(self.url("/wake_up")) as resp:
            assert resp.status == 200

    async def wait_generated(self, n: int, timeout_s: float = 5.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while loop.time() < deadline:
            if any(ctl.generated >= n for ctl in self.a.active.values()):
                return
            await asyncio.sleep(0.002)
        raise AssertionError(f"engine a never generated {n} tokens")

    async def stream(self, path: str, body: dict, *, headers: dict | None = None, sleep_after: int | None = None):
        """POST a streaming request; optionally /sleep once ``sleep_after`` tokens were
        generated. Returns (status, response headers, raw text, parsed data objects)."""
        sleeper = None
        if sleep_after is not None:
            async def trigger() -> None:
                await self.wait_generated(sleep_after)
                await self.sleep()
            sleeper = asyncio.ensure_future(trigger())
        async with self.http.post(self.url(path), json=body, headers=headers or {}) as resp:
            raw = (await resp.read()).decode()
            status, rheaders = resp.status, dict(resp.headers)
        if sleeper is not None:
            await sleeper
        return status, rheaders, raw, parse_sse(raw)


def parse_sse(raw: str) -> list:
    out = []
    for event in raw.split("\n\n"):
        for line in event.split("\n"):
            if line.startswith("data: "):
                payload = line[6:]
                out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


def completion_body(max_tokens: int = 20, **extra) -> dict:
    body = {"model": MODEL, "prompt": PROMPT, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    body.update(extra)
    return body


def chat_body(max_tokens: int = 20, **extra) -> dict:
    body = {"model": MODEL, "messages": [{"role": "user", "content": "hello there"}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    body.update(extra)
    return body


def tokens(name: str, n: int, start: int = 0) -> str:
    return "".join(f"{name}{i} " for i in range(start, start + n))


def content_text(objs: list, chat: bool) -> str:
    parts = []
    for obj in objs:
        if obj == "[DONE]":
            continue
        for choice in obj.get("choices") or []:
            parts.append((choice.get("delta") or {}).get("content") or "" if chat else choice.get("text") or "")
    return "".join(parts)


def finishes(objs: list) -> list:
    return [c.get("finish_reason") for o in objs if o != "[DONE]" for c in o.get("choices") or []
            if c.get("finish_reason")]


def usage_chunks(objs: list) -> list:
    return [o for o in objs if o != "[DONE]" and not o.get("choices") and isinstance(o.get("usage"), dict)]


GEN_HEADERS = {"model": MODEL, "routing-strategy": "least-gpu-cache", "x-custom": "keep-me",
               "X-Request-Id": "orig-req-1"}


# ------------------------------------------------------------------ transparency


@pytest.mark.asyncio
async def test_passthrough_stream_without_sleep_is_unchanged():
    async with Harness() as h:
        status, _, raw, objs = await h.stream("/v1/completions", completion_body(8))
        assert status == 200
        assert content_text(objs, chat=False) == tokens("a", 8)
        assert finishes(objs) == ["length"]
        assert usage_chunks(objs)[0]["usage"]["completion_tokens"] == 8
        assert objs[-1] == "[DONE]"
        assert "tre-reissue" not in raw
        assert h.b.generation_requests() == []


@pytest.mark.asyncio
async def test_forced_usage_chunk_is_hidden_from_client_that_did_not_ask():
    async with Harness() as h:
        body = completion_body(5)
        del body["stream_options"]
        _, _, _, objs = await h.stream("/v1/completions", body)
        assert usage_chunks(objs) == []
        assert h.a.generation_requests()[0]["body"]["stream_options"] == {"include_usage": True}
        assert content_text(objs, chat=False) == tokens("a", 5)


@pytest.mark.asyncio
async def test_other_paths_are_proxied():
    async with Harness() as h:
        async with h.http.get(h.url("/metrics")) as resp:
            assert "vllm:num_requests_running" in await resp.text()
        async with h.http.get(h.url("/health")) as resp:
            assert resp.status == 200
        async with h.http.get(h.url("/is_sleeping")) as resp:
            assert (await resp.json()) == {"is_sleeping": False}


# ------------------------------------------------------------- abort -> continue


@pytest.mark.asyncio
async def test_completion_stream_aborted_by_sleep_is_continued_and_spliced():
    async with Harness() as h:
        status, _, raw, objs = await h.stream(
            "/v1/completions", completion_body(20), headers=GEN_HEADERS, sleep_after=5
        )
        assert status == 200
        text = content_text(objs, chat=False)
        k = len(text.split()) - len([t for t in text.split() if t.startswith("b")])
        assert text == tokens("a", k) + tokens("b", 20 - k)
        assert 5 <= k < 20
        assert "abort" not in finishes(objs)
        assert finishes(objs) == ["length"]
        ids = {o["id"] for o in objs if o != "[DONE]"}
        assert len(ids) == 1 and next(iter(ids)).startswith("cmpl-orig-req-1")
        assert objs.count("[DONE]") == 1 and objs[-1] == "[DONE]"
        usage = usage_chunks(objs)
        assert len(usage) == 1
        assert usage[0]["usage"] == {"prompt_tokens": 4, "completion_tokens": 20, "total_tokens": 24}
        ext = usage[0]["tre_reissue"]
        assert ext["n"] == 1 and ext["target"] == GW_POD and ext["depth"] == 1 and ext["outcome"] == "ok"
        assert ext["gap_ms"] is not None and ext["gap_ms"] >= 0
        assert ": tre-reissue " in raw
        # the continuation went through the gateway with the text-based prompt
        (cont,) = h.b.generation_requests()
        assert cont["path"] == "/v1/completions"
        assert cont["body"]["prompt"] == PROMPT + tokens("a", k)
        assert cont["body"]["max_tokens"] == 20 - k
        assert cont["body"]["ignore_eos"] is True and cont["body"]["temperature"] == 0
        assert cont["headers"]["X-TRE-Reissue-Depth"] == "1"
        assert cont["headers"]["routing-strategy"] == "least-gpu-cache"
        assert cont["headers"]["model"] == MODEL
        assert cont["headers"]["x-custom"] == "keep-me"
        assert cont["headers"].get("X-Request-Id") != "orig-req-1"
        assert cont["headers"]["X-TRE-Reissue-Origin"] == "pod-a"
        assert h.sidecar.metrics.reissue == {("abort", "ok"): 1}
        assert h.sidecar.metrics.gap_count == 1


@pytest.mark.asyncio
async def test_chat_stream_render_mode_continues_as_completion_over_rendered_prompt():
    async with Harness() as h:
        status, _, _, objs = await h.stream("/v1/chat/completions", chat_body(12), sleep_after=4)
        assert status == 200
        assert all(o["object"] == "chat.completion.chunk" for o in objs if o != "[DONE]")
        roles = [c["delta"].get("role") for o in objs if o != "[DONE]" for c in o.get("choices") or []
                 if c["delta"].get("role")]
        assert roles == ["assistant"]
        text = content_text(objs, chat=True)
        k = len([t for t in text.split() if t.startswith("a")])
        assert text == tokens("a", k) + tokens("b", 12 - k)
        assert finishes(objs) == ["length"]
        (cont,) = h.b.generation_requests()
        assert cont["path"] == "/v1/completions"
        rendered = render_chat(chat_body()["messages"], True)
        assert cont["body"]["prompt"] == rendered + tokens("a", k)
        assert cont["body"]["add_special_tokens"] is False
        assert cont["body"]["max_tokens"] == 12 - k
        assert "messages" not in cont["body"]
        assert usage_chunks(objs)[0]["usage"]["completion_tokens"] == 12


@pytest.mark.asyncio
async def test_chat_stream_continue_final_message_mode():
    async with Harness(chat_mode="continue_final_message") as h:
        _, _, _, objs = await h.stream("/v1/chat/completions", chat_body(10), sleep_after=3)
        text = content_text(objs, chat=True)
        k = len([t for t in text.split() if t.startswith("a")])
        assert text == tokens("a", k) + tokens("b", 10 - k)
        roles = [c["delta"].get("role") for o in objs if o != "[DONE]" for c in o.get("choices") or []
                 if c["delta"].get("role")]
        assert roles == ["assistant"]
        (cont,) = h.b.generation_requests()
        assert cont["path"] == "/v1/chat/completions"
        assert cont["body"]["messages"][-1] == {"role": "assistant", "content": tokens("a", k)}
        assert cont["body"]["continue_final_message"] is True
        assert cont["body"]["add_generation_prompt"] is False
        assert cont["body"]["max_tokens"] == 10 - k


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [False, True])
async def test_non_stream_aborted_by_sleep_is_merged(chat):
    async with Harness() as h:
        path = "/v1/chat/completions" if chat else "/v1/completions"
        body = chat_body(15) if chat else completion_body(15)
        body["stream"] = False
        del body["stream_options"]

        async def trigger():
            await h.wait_generated(4)
            await h.sleep()

        sleeper = asyncio.ensure_future(trigger())
        async with h.http.post(h.url(path), json=body) as resp:
            assert resp.status == 200
            obj = await resp.json()
        await sleeper
        choice = obj["choices"][0]
        text = choice["message"]["content"] if chat else choice["text"]
        k = len([t for t in text.split() if t.startswith("a")])
        assert k >= 4
        assert text == tokens("a", k) + tokens("b", 15 - k)
        assert choice["finish_reason"] == "length"
        assert obj["usage"]["completion_tokens"] == 15
        assert obj["tre_reissue"]["n"] == 1 and obj["tre_reissue"]["outcome"] == "ok"
        assert obj["tre_reissue"]["target"] == GW_POD
        (cont,) = h.b.generation_requests()
        assert cont["body"]["stream"] is False and "stream_options" not in cont["body"]


# ------------------------------------------------------------- must not reissue


@pytest.mark.asyncio
async def test_client_disconnect_is_not_reissued():
    async with Harness(a_delay=0.02) as h:
        async with h.http.post(h.url("/v1/completions"), json=completion_body(200)) as resp:
            seen = 0
            async for _ in resp.content:
                seen += 1
                if seen >= 6:
                    break
            resp.close()
        await asyncio.sleep(0.2)
        await h.sleep()
        await asyncio.sleep(0.3)
        assert h.b.generation_requests() == []
        assert ("abort", "ok") not in h.sidecar.metrics.reissue
        assert h.a.active == {}


@pytest.mark.asyncio
async def test_engine_abort_while_awake_passes_through():
    async with Harness(a_abort_after=3) as h:
        _, _, _, objs = await h.stream("/v1/completions", completion_body(10))
        assert finishes(objs) == ["abort"]
        assert usage_chunks(objs)[0]["usage"]["completion_tokens"] == 3
        assert h.b.generation_requests() == []
        assert h.sidecar.metrics.reissue == {("abort", "abort_not_sleeping"): 1}


@pytest.mark.asyncio
async def test_depth_limit_passes_abort_through_and_counts_it():
    async with Harness() as h:
        headers = dict(GEN_HEADERS, **{"X-TRE-Reissue-Depth": "3"})
        _, _, _, objs = await h.stream("/v1/completions", completion_body(20), headers=headers, sleep_after=3)
        assert finishes(objs) == ["abort"]
        assert objs[-1] == "[DONE]"
        assert h.b.generation_requests() == []
        assert h.sidecar.metrics.reissue == {("abort", "depth_limit"): 1}


@pytest.mark.asyncio
async def test_continuation_aborted_downstream_is_reported_as_failed():
    async with Harness(b_abort_after=2) as h:
        _, _, raw, objs = await h.stream("/v1/completions", completion_body(20), sleep_after=5)
        assert finishes(objs)[-1] == "abort"
        text = content_text(objs, chat=False)
        k = len([t for t in text.split() if t.startswith("a")])
        assert text == tokens("a", k) + tokens("b", 2)
        usage = usage_chunks(objs)[0]
        assert usage["usage"]["completion_tokens"] == k + 2
        assert usage["tre_reissue"]["outcome"] == "failed"
        assert h.sidecar.metrics.reissue == {("abort", "failed"): 1}


@pytest.mark.asyncio
async def test_gateway_error_passes_abort_through():
    async with Harness(gateway_url="http://127.0.0.1:9") as h:
        _, _, _, objs = await h.stream("/v1/completions", completion_body(20), sleep_after=4)
        assert finishes(objs) == ["abort"]
        text = content_text(objs, chat=False)
        assert text and all(t.startswith("a") for t in text.split())
        assert usage_chunks(objs)[0]["tre_reissue"]["outcome"] == "failed"
        assert h.sidecar.metrics.reissue == {("abort", "failed"): 1}


# ------------------------------------------------------------- sleeping pod


@pytest.mark.asyncio
async def test_new_request_while_sleeping_is_forwarded_to_gateway():
    async with Harness() as h:
        await h.sleep()
        _, rheaders, _, objs = await h.stream("/v1/completions", completion_body(6), headers=GEN_HEADERS)
        assert content_text(objs, chat=False) == tokens("b", 6)
        assert rheaders.get("target-pod") == GW_POD
        (fwd,) = h.b.generation_requests()
        assert fwd["headers"]["X-TRE-Forward-Hops"] == "1"
        assert fwd["headers"]["routing-strategy"] == "least-gpu-cache"
        assert fwd["body"] == completion_body(6)
        assert h.a.generation_requests() == []
        async with h.http.post(h.url("/v1/completions"), json=completion_body(6),
                               headers={"X-TRE-Forward-Hops": "5"}) as resp:
            assert resp.status == 503
        assert h.sidecar.metrics.forward == {"ok": 1, "hop_limit": 1}


@pytest.mark.asyncio
async def test_wake_clears_sleeping_and_serves_locally_again():
    async with Harness() as h:
        await h.sleep()
        assert h.sidecar.state.active
        await h.wake()
        assert not h.sidecar.state.active
        _, _, _, objs = await h.stream("/v1/completions", completion_body(4))
        assert content_text(objs, chat=False) == tokens("a", 4)
        assert h.b.generation_requests() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_request_stuck_behind_the_pause_is_reissued(stream):
    async with Harness(stuck_grace_s=0.05) as h:
        h.a.pause_silently()  # the request reaches the engine after its abort pass
        body = completion_body(6, stream=stream)
        if not stream:
            del body["stream_options"]
        request = asyncio.ensure_future(h.http.post(h.url("/v1/completions"), json=body))
        for _ in range(200):
            if h.sidecar.state.inflight:
                break
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.05)
        await h.sleep()  # engine answers [] (already paused): the request is not in it
        resp = await asyncio.wait_for(request, 5)
        async with resp:
            assert resp.status == 200
            if stream:
                objs = parse_sse((await resp.read()).decode())
                assert content_text(objs, chat=False) == tokens("b", 6)
                assert usage_chunks(objs)[0]["tre_reissue"]["kind"] == "stuck"
            else:
                obj = await resp.json()
                assert obj["choices"][0]["text"] == tokens("b", 6)
                assert obj["tre_reissue"]["kind"] == "stuck"
        (cont,) = h.b.generation_requests()
        assert cont["body"]["prompt"] == PROMPT and cont["body"]["max_tokens"] == 6
        assert h.sidecar.metrics.reissue == {("stuck", "ok"): 1}


@pytest.mark.asyncio
async def test_restarted_sidecar_learns_engine_is_asleep():
    async with Harness() as h:
        h.a.pause_silently()
        for _ in range(100):
            if h.sidecar.state.sleeping:
                break
            await asyncio.sleep(0.01)
        # probe ran at startup before the pause; a fresh sidecar sees it
        side = ReissueSidecar(h.sidecar.cfg)
        srv = TestServer(side.build_app())
        await srv.start_server()
        try:
            for _ in range(100):
                if side.state.sleeping:
                    break
                await asyncio.sleep(0.01)
            assert side.state.sleeping
        finally:
            await srv.close()


# ------------------------------------------------------------- client parsers


@pytest.mark.asyncio
async def test_openai_sdk_parses_spliced_chat_and_completion_streams():
    openai = pytest.importorskip("openai")
    async with Harness() as h:
        client = openai.AsyncOpenAI(base_url=h.url("/v1"), api_key="x", max_retries=0,
                                    default_headers={"routing-strategy": "least-gpu-cache"})

        async def trigger():
            await h.wait_generated(4)
            await h.sleep()

        sleeper = asyncio.ensure_future(trigger())
        stream = await client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "hello there"}], max_tokens=16, temperature=0,
            stream=True, stream_options={"include_usage": True},
        )
        content, finish, usage, extra = [], None, None, None
        async for chunk in stream:
            if chunk.choices:
                if chunk.choices[0].delta.content:
                    content.append(chunk.choices[0].delta.content)
                finish = chunk.choices[0].finish_reason or finish
            if chunk.usage is not None:
                usage = chunk.usage
                extra = (chunk.model_extra or {}).get("tre_reissue")
        await sleeper
        text = "".join(content)
        k = len([t for t in text.split() if t.startswith("a")])
        assert text == tokens("a", k) + tokens("b", 16 - k)
        assert finish == "length"
        assert usage.completion_tokens == 16
        assert extra and extra["n"] == 1

        await h.wake()
        sleeper = asyncio.ensure_future(trigger())
        stream = await client.completions.create(
            model=MODEL, prompt=PROMPT, max_tokens=12, temperature=0, stream=True,
            stream_options={"include_usage": True}, extra_body={"ignore_eos": True},
        )
        text, usage = "", None
        async for chunk in stream:
            if chunk.choices:
                text += chunk.choices[0].text
            if chunk.usage is not None:
                usage = chunk.usage
        await sleeper
        k = len([t for t in text.split() if t.startswith("a")])
        assert text == tokens("a", k) + tokens("b", 12 - k)
        assert usage.completion_tokens == 12 and usage.prompt_tokens == 4
        await client.close()


@pytest.mark.asyncio
async def test_replayer_http_sender_parses_spliced_stream():
    from tre_replayer.engine.http_sender import build_request_headers, _default_stream_call

    async with Harness() as h:
        body = json.dumps(completion_body(20)).encode()
        headers = build_request_headers(MODEL, "least-gpu-cache")

        async def trigger():
            await h.wait_generated(5)
            await h.sleep()

        sleeper = asyncio.ensure_future(trigger())
        result = await asyncio.to_thread(_default_stream_call, h.url("/v1/completions"), headers, body, 30.0)
        await sleeper
        assert result.status == 200 and result.error is None
        assert result.completion_tokens == 20 and result.prompt_tokens == 4
        assert result.first_token_ms is not None
        (cont,) = h.b.generation_requests()
        # urllib title-cases header names; HTTP header names are case-insensitive.
        received = {name.lower(): value for name, value in cont["headers"].items()}
        assert received["routing-strategy"] == "least-gpu-cache"
        assert received["model"] == MODEL


# ------------------------------------------------------------- observability


@pytest.mark.asyncio
async def test_metrics_and_state_endpoints():
    async with Harness() as h:
        await h.stream("/v1/completions", completion_body(20), sleep_after=3)
        async with h.http.get(h.url("/tre-reissue/metrics")) as resp:
            text = await resp.text()
        assert f'tre_reissue_total{{model="{MODEL}",kind="abort",outcome="ok"}} 1' in text
        assert f'tre_reissue_gap_seconds_count{{model="{MODEL}"}} 1' in text
        assert f'tre_reissue_sleeping{{model="{MODEL}"}} 1' in text
        async with h.http.get(h.url("/tre-reissue/state")) as resp:
            state = await resp.json()
        assert state["sleeping"] is True
        # the real image's /sleep snapshot is [] and flagged unreliable (review M1)
        assert state["last_sleep"]["aborted"] == [] and state["last_sleep"]["aborted_reliable"] is False


# ------------------------------------------------------------- pure helpers


def test_split_events_keeps_partial_tail():
    events, rest = sc.split_events(b"data: 1\n\ndata: 2\n\ndata: 3")
    assert events == [b"data: 1\n\n", b"data: 2\n\n"] and rest == b"data: 3"
    assert sc.event_data(b": comment\n\n") is None
    assert sc.event_data(b"data: [DONE]\n\n") == b"[DONE]"


def test_budget_and_usage_merge():
    assert sc.completion_continuation({"prompt": "p", "max_tokens": 5}, "x", 5) is None
    cont = sc.completion_continuation({"prompt": ["p"], "max_tokens": 5, "min_tokens": 3, "echo": True}, " x", 2)
    assert cont["prompt"] == "p x" and cont["max_tokens"] == 3 and cont["min_tokens"] == 1 and cont["echo"] is False
    assert sc.chat_cfm_continuation({"messages": [], "max_completion_tokens": 4}, "x", 4) is None
    assert sc.merge_usage({"prompt_tokens": 10, "completion_tokens": 3}, {"prompt_tokens": 13, "completion_tokens": 7},
                          generated_before=3) == {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
    assert sc.merge_usage(None, {"prompt_tokens": 9, "completion_tokens": 4}, generated_before=0) == {
        "prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}


def test_chat_render_continuation_defaults_budget_from_context():
    cont = sc.chat_render_continuation({"model": "m", "messages": [], "stream": True, "tools": [1], "seed": 7},
                                       "<p>", "ab", 2, max_model_len=100, prompt_len=10)
    # seed -> seed + depth: the continuation must not replay the first segment's stream (L3)
    assert cont == {"model": "m", "prompt": "<p>ab", "add_special_tokens": False, "seed": 8, "stream": True,
                    "stream_options": {"include_usage": True}, "max_tokens": 88}


def test_reissuable_shapes():
    assert sc.reissuable("/v1/completions", {"prompt": "x"})
    assert not sc.reissuable("/v1/completions", {"prompt": [1, 2, 3]})
    assert not sc.reissuable("/v1/completions", {"prompt": "x", "n": 2})
    assert not sc.reissuable("/v1/completions", {"prompt": "x", "echo": True})
    assert sc.reissuable("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]})
    assert not sc.reissuable("/v1/chat/completions", {"messages": []})


def test_config_from_env():
    cfg = Config.from_env({"TRE_REISSUE_GATEWAY_URL": "http://gw/", "TRE_REISSUE_MODEL": "m", "POD_NAME": "p",
                           "TRE_REISSUE_ENABLED": "false", "TRE_REISSUE_MAX_DEPTH": "2"})
    assert cfg.gateway_url == "http://gw" and cfg.model == "m" and cfg.pod_name == "p"
    assert cfg.enabled is False and cfg.max_depth == 2 and cfg.upstream_url == "http://127.0.0.1:8001"
    with pytest.raises(ValueError):
        Config.from_env({"TRE_REISSUE_CHAT_MODE": "bogus"})


@pytest.mark.asyncio
async def test_disabled_sidecar_is_a_pure_proxy():
    async with Harness(enabled=False) as h:
        _, _, raw, objs = await h.stream("/v1/completions", completion_body(20), sleep_after=3)
        assert finishes(objs) == ["abort"]
        assert h.b.generation_requests() == []
        assert h.a.generation_requests()[0]["body"] == completion_body(20)


def test_sidecar_script_only_needs_stdlib_and_aiohttp():
    """The script ships via ConfigMap into the vLLM image (python3.12 + aiohttp)."""
    import ast
    import sys

    tree = ast.parse(open(sc.__file__, encoding="utf-8").read())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    stdlib = set(sys.stdlib_module_names)
    assert {r for r in roots if r not in stdlib} <= {"aiohttp", "uvloop"}  # uvloop: optional


@pytest.mark.asyncio
async def test_bounced_request_backs_off_before_forwarding_again():
    async with Harness(forward_backoff_s=0.1) as h:
        await h.sleep()
        loop = asyncio.get_running_loop()
        start = loop.time()
        async with h.http.post(h.url("/v1/completions"), json=completion_body(2),
                               headers={"X-TRE-Forward-Hops": "3"}) as resp:
            assert resp.status == 200
            await resp.read()
        assert loop.time() - start >= 0.4  # 0.1 * 2 ** (3 - 1)
        (fwd,) = h.b.generation_requests()
        assert fwd["headers"]["X-TRE-Forward-Hops"] == "4"


# =================================================================== review fixes (09-24)


async def _wait_for(predicate, timeout_s: float = 3.0, what: str = "condition") -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"timed out waiting for {what}")


# ---------------------------------------------------------------- H3: fail closed


@pytest.mark.asyncio
async def test_sleep_without_hidden_header_is_refused():
    async with Harness() as h:
        await h.sleep(expect=409, hidden=False)
        assert h.a.sleep_calls == 0 and not h.sidecar.state.active
        assert h.sidecar.metrics.events["sleep_rejected_not_hidden"] == 1
    async with Harness(require_hidden_header=False) as h:
        await h.sleep(hidden=False)
        assert h.sidecar.state.sleeping


# ---------------------------------------------------------------- L4 / rollback


@pytest.mark.asyncio
async def test_repeated_sleep_is_idempotent():
    async with Harness() as h:
        await h.sleep()
        epoch, slept_at = h.sidecar.state.epoch, h.sidecar.state.slept_at
        await h.sleep()
        assert h.sidecar.state.epoch == epoch and h.sidecar.state.slept_at == slept_at
        assert h.sidecar.metrics.events["sleep_repeated"] == 1


@pytest.mark.asyncio
async def test_sleep_failure_rolls_back():
    async with Harness() as h:
        h.a.sleep_status = 500
        await h.sleep(expect=500)
        assert not h.sidecar.state.active
        h.a.sleep_status = 200
        _, _, _, objs = await h.stream("/v1/completions", completion_body(4))
        assert content_text(objs, chat=False) == tokens("a", 4)  # served locally again
        assert h.b.generation_requests() == []


@pytest.mark.asyncio
async def test_sleep_timeout_rolls_back():
    async with Harness(control_timeout_s=0.2, probe_interval_s=3600) as h:
        h.a.sleep_hang = True
        await h.sleep(expect=504)
        assert not h.sidecar.state.active and h.sidecar.state.pending == 0
        assert h.sidecar.metrics.events["sleep_failed"] == 1


# ---------------------------------------------------------------- H2: state sync


@pytest.mark.asyncio
async def test_vllm_restart_resyncs_sleeping_mark_via_probe():
    async with Harness(probe_interval_s=0.05) as h:
        await h.sleep()
        assert h.sidecar.state.sleeping
        h.a.restart()  # the engine came back awake without any /wake_up
        await _wait_for(lambda: not h.sidecar.state.active, what="probe correction")
        assert h.sidecar.metrics.events["state_corrected_to_awake"] == 1
        _, _, _, objs = await h.stream("/v1/completions", completion_body(3))
        assert content_text(objs, chat=False) == tokens("a", 3)


@pytest.mark.asyncio
async def test_proxied_is_sleeping_answer_resyncs_state():
    async with Harness(probe_interval_s=3600) as h:
        await asyncio.sleep(0.05)  # let the startup probe run once
        h.a.pause_silently()
        async with h.http.get(h.url("/is_sleeping")) as resp:
            assert (await resp.json()) == {"is_sleeping": True}
        assert h.sidecar.state.sleeping
        assert h.sidecar.metrics.events["state_corrected_to_sleeping"] == 1


# ---------------------------------------------------------------- M1 / L1: stuck detection


@pytest.mark.asyncio
async def test_empty_snapshot_late_abort_response_is_not_misjudged_as_stuck():
    # Real image: /sleep answers [] and a non-stream abort response may trail /sleep.
    async with Harness(a_abort_response_delay_s=0.15, stuck_grace_s=0.5) as h:
        body = completion_body(15, stream=False)
        del body["stream_options"]

        async def trigger():
            await h.wait_generated(4)
            await h.sleep()

        sleeper = asyncio.ensure_future(trigger())
        async with h.http.post(h.url("/v1/completions"), json=body) as resp:
            obj = await resp.json()
        await sleeper
        assert obj["tre_reissue"]["kind"] == "abort" and obj["tre_reissue"]["outcome"] == "ok"
        text = obj["choices"][0]["text"]
        k = len([t for t in text.split() if t.startswith("a")])
        assert k >= 4 and text == tokens("a", k) + tokens("b", 15 - k)
        assert "stuck_detected" not in h.sidecar.metrics.events


@pytest.mark.asyncio
async def test_periodic_stuck_scan_after_probe_detected_sleep():
    # The engine went to sleep behind the sidecar's back (e.g. sidecar restarted): no
    # /sleep passes through, only the periodic probe + scan can free the hung request.
    # The monitor probes once at start, then every probe_interval_s: pausing 50 ms in
    # leaves ~250 ms for the request to reach the engine before the next probe marks
    # the pod sleeping. With a 50 ms interval the probe raced the request, which was
    # then forwarded as a new request while sleeping (flaky on a loaded host).
    async with Harness(probe_interval_s=0.3, stuck_grace_s=0.1) as h:
        await asyncio.sleep(0.05)
        h.a.pause_silently()
        body = completion_body(5)
        async with h.http.post(h.url("/v1/completions"), json=body) as resp:
            objs = parse_sse((await resp.read()).decode())
        assert content_text(objs, chat=False) == tokens("b", 5)
        assert usage_chunks(objs)[0]["tre_reissue"]["kind"] == "stuck"
        assert h.sidecar.metrics.events["stuck_detected"] == 1


# ---------------------------------------------------------------- transport


@pytest.mark.asyncio
async def test_sse_split_across_tcp_reads_is_spliced_correctly():
    async with Harness(a_fragment=True, b_fragment=True, a_hold_at=6) as h:
        _, _, raw, objs = await h.stream("/v1/completions", completion_body(20), sleep_after=6)
        assert content_text(objs, chat=False) == tokens("a", 6) + tokens("b", 14)
        assert finishes(objs) == ["length"]
        assert usage_chunks(objs)[0]["usage"]["completion_tokens"] == 20


@pytest.mark.asyncio
async def test_large_batch_simultaneous_abort():
    async with Harness(a_hold_at=5) as h:
        n = 40
        requests = [asyncio.ensure_future(h.http.post(h.url("/v1/completions"), json=completion_body(12)))
                    for _ in range(n)]
        await _wait_for(lambda: len(h.a.active) == n and all(c.generated == 5 for c in h.a.active.values()),
                        what="all streams parked")
        await h.sleep()
        texts = []
        for fut in requests:
            resp = await asyncio.wait_for(fut, 10)
            async with resp:
                objs = parse_sse((await resp.read()).decode())
            texts.append(content_text(objs, chat=False))
            assert usage_chunks(objs)[0]["usage"]["completion_tokens"] == 12
        assert all(t == tokens("a", 5) + tokens("b", 7) for t in texts)
        assert h.sidecar.metrics.reissue == {("abort", "ok"): n}
        assert len(h.b.generation_requests()) == n


@pytest.mark.asyncio
async def test_sidecar_to_sidecar_multi_level_splice():
    a = FakeVllm("a", token_delay_s=0.003, hold_at=5)
    b = FakeVllm("b", token_delay_s=0.003, hold_at=3, target_pod="pod-b:8000")
    c = FakeVllm("c", token_delay_s=0.002, target_pod="pod-c:8000")
    servers = [TestServer(x.app()) for x in (a, b, c)]
    for srv in servers:
        await srv.start_server()
    a_srv, b_srv, c_srv = servers
    side_b = ReissueSidecar(Config(upstream_url=str(b_srv.make_url("")).rstrip("/"),
                                   gateway_url=str(c_srv.make_url("")).rstrip("/"), model=MODEL, pod_name="pod-b"))
    sb_srv = TestServer(side_b.build_app())
    await sb_srv.start_server()
    side_a = ReissueSidecar(Config(upstream_url=str(a_srv.make_url("")).rstrip("/"),
                                   gateway_url=str(sb_srv.make_url("")).rstrip("/"), model=MODEL, pod_name="pod-a"))
    sa_srv = TestServer(side_a.build_app())
    await sa_srv.start_server()
    http = aiohttp.ClientSession()
    try:
        async def orchestrate():
            await _wait_for(lambda: any(x.generated == 5 for x in a.active.values()), what="a parked")
            async with http.post(str(sa_srv.make_url("/sleep")), headers={"X-TRE-Hidden": "1"}) as r:
                assert r.status == 200
            await _wait_for(lambda: any(x.generated == 3 for x in b.active.values()), what="b parked")
            async with http.post(str(sb_srv.make_url("/sleep")), headers={"X-TRE-Hidden": "1"}) as r:
                assert r.status == 200

        orch = asyncio.ensure_future(orchestrate())
        async with http.post(str(sa_srv.make_url("/v1/completions")), json=completion_body(20),
                             headers=GEN_HEADERS) as resp:
            objs = parse_sse((await resp.read()).decode())
        await orch
        assert content_text(objs, chat=False) == tokens("a", 5) + tokens("b", 3) + tokens("c", 12)
        assert finishes(objs) == ["length"]
        (usage,) = usage_chunks(objs)
        assert usage["usage"] == {"prompt_tokens": 4, "completion_tokens": 20, "total_tokens": 24}
        assert usage["tre_reissue"]["n"] == 2 and usage["tre_reissue"]["target"] == "pod-c:8000"
        assert {o["id"] for o in objs if o != "[DONE]"} == {"cmpl-orig-req-1-0"}
        (to_b,) = b.generation_requests()
        (to_c,) = c.generation_requests()
        assert to_b["headers"]["X-TRE-Reissue-Depth"] == "1" and to_c["headers"]["X-TRE-Reissue-Depth"] == "2"
        assert to_b["body"]["max_tokens"] == 15 and to_c["body"]["max_tokens"] == 12
        assert to_c["body"]["prompt"] == PROMPT + tokens("a", 5) + tokens("b", 3)
    finally:
        await http.close()
        for srv in (sa_srv, sb_srv, *servers):
            await srv.close()


# ---------------------------------------------------------------- M2: stop across the seam


@pytest.mark.asyncio
async def test_stop_string_spanning_the_seam_is_honoured_stream():
    async with Harness(a_hold_at=5) as h:
        body = completion_body(20, stop=["a4 b0"])
        _, _, _, objs = await h.stream("/v1/completions", body, sleep_after=5)
        assert content_text(objs, chat=False) == "a0 a1 a2 a3 "
        assert finishes(objs) == ["stop"]
        (usage,) = usage_chunks(objs)
        assert usage["tre_reissue"]["stop_at_seam"] is True and usage["tre_reissue"]["outcome"] == "ok"
        assert usage["usage"]["completion_tokens"] >= 6
        assert h.b.generation_requests()[0]["body"]["stop"] == ["a4 b0"]


@pytest.mark.asyncio
async def test_stop_string_spanning_the_seam_is_honoured_non_stream():
    async with Harness(a_hold_at=5) as h:
        body = completion_body(20, stop=["a4 b0"], stream=False)
        del body["stream_options"]

        async def trigger():
            await h.wait_generated(5)
            await h.sleep()

        sleeper = asyncio.ensure_future(trigger())
        async with h.http.post(h.url("/v1/completions"), json=body) as resp:
            obj = await resp.json()
        await sleeper
        assert obj["choices"][0]["text"] == "a0 a1 a2 a3 "
        assert obj["choices"][0]["finish_reason"] == "stop"
        assert obj["tre_reissue"]["stop_at_seam"] is True


@pytest.mark.asyncio
async def test_stop_inside_the_continuation_is_left_to_the_engine():
    async with Harness(a_hold_at=5) as h:
        _, _, _, objs = await h.stream("/v1/completions", completion_body(20, stop=["b3"]), sleep_after=5)
        assert content_text(objs, chat=False) == tokens("a", 5) + "b0 b1 b2 "
        assert finishes(objs) == ["stop"]
        assert "stop_at_seam" not in usage_chunks(objs)[0]["tre_reissue"]


# ---------------------------------------------------------------- M3 / L3: budget and params


@pytest.mark.asyncio
async def test_missing_max_tokens_uses_vllm_default_budget():
    async with Harness(a_hold_at=5) as h:
        body = completion_body(20)
        del body["max_tokens"]
        _, _, _, objs = await h.stream("/v1/completions", body, sleep_after=5)
        assert h.b.generation_requests()[0]["body"]["max_tokens"] == 11  # 16 - 5
        assert usage_chunks(objs)[0]["usage"]["completion_tokens"] == 16
    async with Harness(a_hold_at=4, b_abort_after=1) as h:
        body = chat_body(20)
        del body["max_tokens"]
        await h.stream("/v1/chat/completions", body, sleep_after=4)
        cont = h.b.generation_requests()[0]["body"]
        rendered = render_chat(body["messages"], True)
        assert cont["max_tokens"] == 100000 - len(rendered) - 4


@pytest.mark.asyncio
async def test_logprobs_dropped_and_seed_bumped_in_continuation():
    async with Harness(a_hold_at=3) as h:
        body = completion_body(10, logprobs=1, seed=5, response_format={"type": "text"})
        _, _, _, objs = await h.stream("/v1/completions", body, sleep_after=3)
        cont = h.b.generation_requests()[0]["body"]
        assert "logprobs" not in cont and cont["seed"] == 6
        assert cont["response_format"] == {"type": "text"}
        assert usage_chunks(objs)[0]["tre_reissue"]["dropped"] == ["logprobs"]


@pytest.mark.asyncio
async def test_render_round_trip_mismatch_falls_back_to_continue_final_message():
    async with Harness(a_hold_at=3) as h:
        h.a.detok_suffix = "!"  # the text form no longer re-tokenizes to the same count
        _, _, _, objs = await h.stream("/v1/chat/completions", chat_body(8), sleep_after=3)
        (cont,) = h.b.generation_requests()
        assert cont["path"] == "/v1/chat/completions" and cont["body"]["continue_final_message"] is True
        assert h.sidecar.metrics.events["render_fallback_roundtrip"] == 1
        assert content_text(objs, chat=True) == tokens("a", 3) + tokens("b", 5)

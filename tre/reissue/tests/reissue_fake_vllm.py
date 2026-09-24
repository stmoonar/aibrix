"""In-process fake of the custom vLLM 0.10.1-sleep image, for the reissue sidecar tests.

Mimics exactly the behaviour the sidecar depends on:

* OpenAI-style streaming for /v1/completions and /v1/chat/completions: one token per
  chunk, ``"usage": null`` on every chunk, the finish_reason on the last token chunk, a
  usage-only chunk when ``stream_options.include_usage``, then ``[DONE]``; non-streaming
  JSON responses; ``X-Request-Id`` becomes the engine request id.
* ``POST /sleep``: pause (new requests hang until wake), abort every in-flight request
  (the stream gets a final chunk with ``finish_reason: "abort"`` + usage + [DONE]) and
  answer with the aborted-request snapshot (ids + lengths only). A second /sleep while
  paused answers ``[]``.
* ``/wake_up``, ``/is_sleeping``, ``/metrics``, ``/health``, ``/tokenize``, ``/detokenize``.

Tokens are ``"<name><i> "`` so the tests can tell which engine produced which part of a
spliced answer. The prompt length in tokens is the whitespace word count.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from aiohttp import web


class _Ctl:
    def __init__(self, request_id: str, prompt_len: int) -> None:
        self.request_id = request_id
        self.prompt_len = prompt_len
        self.generated = 0
        self.abort = False


def render_chat(messages: list[dict], add_generation_prompt: bool) -> str:
    text = "<bos>" + "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
    if add_generation_prompt:
        text += "<|assistant|>"
    return text


class FakeVllm:
    def __init__(
        self,
        name: str,
        *,
        token_delay_s: float = 0.01,
        target_pod: str | None = None,
        abort_after: int | None = None,
    ) -> None:
        self.name = name
        self.token_delay_s = token_delay_s
        self.target_pod = target_pod
        #: abort (without any sleep) after this many tokens: an engine-side abort.
        self.abort_after = abort_after
        self.requests: list[dict[str, Any]] = []
        self.active: dict[str, _Ctl] = {}
        self.paused = False
        self._wake = asyncio.Event()
        self._wake.set()
        self.sleep_calls = 0
        self.disconnects = 0

    # --------------------------------------------------------------- helpers

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/completions", self._completions)
        app.router.add_post("/v1/chat/completions", self._chat)
        app.router.add_post("/sleep", self._sleep)
        app.router.add_post("/wake_up", self._wake_up)
        app.router.add_get("/is_sleeping", self._is_sleeping)
        app.router.add_get("/metrics", self._metrics)
        app.router.add_get("/health", self._health)
        app.router.add_post("/tokenize", self._tokenize)
        app.router.add_post("/detokenize", self._detokenize)
        return app

    def pause_silently(self) -> None:
        """Pause without aborting anything (the state right after vLLM's abort pass, for
        a request that arrives later)."""
        self.paused = True
        self._wake.clear()

    def generation_requests(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["path"].startswith("/v1/")]

    def _headers(self) -> dict[str, str]:
        return {"target-pod": self.target_pod} if self.target_pod else {}

    def _token(self, index: int) -> str:
        return f"{self.name}{index} "

    # ------------------------------------------------------------- endpoints

    async def _record(self, request: web.Request) -> dict:
        body = await request.json()
        self.requests.append({"path": request.path, "headers": dict(request.headers), "body": body})
        return body

    async def _admit(self) -> None:
        await self._wake.wait()

    async def _completions(self, request: web.Request) -> web.StreamResponse:
        body = await self._record(request)
        await self._admit()
        prompt = body["prompt"][0] if isinstance(body["prompt"], list) else body["prompt"]
        rid = "cmpl-" + request.headers.get("X-Request-Id", uuid.uuid4().hex) + "-0"
        return await self._generate(request, body, rid, prompt_len=len(prompt.split()), chat=False)

    async def _chat(self, request: web.Request) -> web.StreamResponse:
        body = await self._record(request)
        await self._admit()
        rendered = render_chat(body["messages"], body.get("add_generation_prompt", True))
        rid = "chatcmpl-" + request.headers.get("X-Request-Id", uuid.uuid4().hex)
        return await self._generate(request, body, rid, prompt_len=len(rendered.split()), chat=True)

    async def _generate(self, request: web.Request, body: dict, rid: str, *, prompt_len: int, chat: bool):
        max_tokens = body.get("max_completion_tokens") or body.get("max_tokens") or 16
        stream = bool(body.get("stream"))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        ctl = _Ctl(rid, prompt_len)
        self.active[rid] = ctl
        obj_type = "chat.completion.chunk" if chat else "text_completion"
        base = {"id": rid, "object": obj_type, "created": 1700000000, "model": body.get("model")}

        def chunk(text: str | None, finish: str | None, role: bool = False) -> dict:
            if chat:
                delta: dict[str, Any] = {}
                if role:
                    delta["role"] = "assistant"
                if text is not None:
                    delta["content"] = text
                choice = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
            else:
                choice = {"index": 0, "text": text or "", "logprobs": None, "finish_reason": finish,
                          "stop_reason": None}
            return dict(base, choices=[choice], usage=None)

        def usage() -> dict:
            return {"prompt_tokens": prompt_len, "completion_tokens": ctl.generated,
                    "total_tokens": prompt_len + ctl.generated}

        try:
            if not stream:
                text = []
                finish = "length"
                for index in range(max_tokens):
                    await asyncio.sleep(self.token_delay_s)
                    if ctl.abort or (self.abort_after is not None and index >= self.abort_after):
                        finish = "abort"
                        break
                    text.append(self._token(index))
                    ctl.generated += 1
                if chat:
                    choice = {"index": 0, "message": {"role": "assistant", "content": "".join(text)},
                              "logprobs": None, "finish_reason": finish}
                else:
                    choice = {"index": 0, "text": "".join(text), "logprobs": None, "finish_reason": finish,
                              "stop_reason": None}
                obj = dict(base, object="chat.completion" if chat else "text_completion", choices=[choice],
                           usage=usage())
                return web.json_response(obj, headers=self._headers())

            resp = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream; charset=utf-8", **self._headers()}
            )
            await resp.prepare(request)
            try:
                if chat:
                    # vLLM sends the role chunk with the first engine output.
                    await asyncio.sleep(self.token_delay_s)
                    await resp.write(_sse(chunk("", None, role=True)))
                finish = None
                for index in range(max_tokens):
                    await asyncio.sleep(self.token_delay_s)
                    if ctl.abort or (self.abort_after is not None and index >= self.abort_after):
                        await resp.write(_sse(chunk("", "abort")))
                        finish = "abort"
                        break
                    ctl.generated += 1
                    last = index == max_tokens - 1
                    await resp.write(_sse(chunk(self._token(index), "length" if last else None)))
                if include_usage:
                    await resp.write(_sse(dict(base, choices=[], usage=usage())))
                await resp.write(b"data: [DONE]\n\n")
                await resp.write_eof()
            except (ConnectionResetError, ConnectionError):
                self.disconnects += 1
            return resp
        finally:
            self.active.pop(rid, None)

    async def _sleep(self, request: web.Request) -> web.Response:
        self.requests.append({"path": "/sleep", "headers": dict(request.headers), "body": None})
        self.sleep_calls += 1
        if self.paused:
            return web.Response(text="[]")
        self.paused = True
        self._wake.clear()
        snapshot = []
        for ctl in list(self.active.values()):
            ctl.abort = True
            snapshot.append({"request_id": ctl.request_id, "priority": 0, "status": 9, "stop_reason": None,
                             "all_token_len": ctl.prompt_len + ctl.generated,
                             "original_prompt_len": ctl.prompt_len, "generated_len": ctl.generated})
        for _ in range(500):
            if not any(ctl.abort for ctl in self.active.values()):
                break
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.02)  # the engine offload itself
        return web.Response(text=json.dumps(snapshot))

    async def _wake_up(self, request: web.Request) -> web.Response:
        self.requests.append({"path": "/wake_up", "headers": dict(request.headers), "body": None})
        self.paused = False
        self._wake.set()
        return web.Response(status=200)

    async def _is_sleeping(self, request: web.Request) -> web.Response:
        return web.json_response({"is_sleeping": self.paused})

    async def _health(self, request: web.Request) -> web.Response:
        return web.Response(status=200)

    async def _metrics(self, request: web.Request) -> web.Response:
        return web.Response(
            text=f'vllm:num_requests_running{{model_name="m"}} {len(self.active)}\n'
            f'vllm:num_requests_waiting{{model_name="m"}} 0\n',
            content_type="text/plain",
        )

    async def _tokenize(self, request: web.Request) -> web.Response:
        body = await self._record(request)
        rendered = render_chat(body["messages"], body.get("add_generation_prompt", True))
        tokens = [ord(c) for c in rendered]
        return web.json_response({"count": len(tokens), "max_model_len": 100000, "tokens": tokens})

    async def _detokenize(self, request: web.Request) -> web.Response:
        body = await self._record(request)
        return web.json_response({"prompt": "".join(chr(t) for t in body["tokens"])})


def _sse(obj: dict) -> bytes:
    return ("data: " + json.dumps(obj) + "\n\n").encode()

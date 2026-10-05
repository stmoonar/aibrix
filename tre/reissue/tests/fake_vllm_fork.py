"""In-process fakes for the reissue sidecar tests: a vLLM >= 0.30 fork engine and a TRE
gateway.

FakeEngine mimics what the sidecar depends on (fork branch tre/transparent-sleep):

* Deterministic generation: the next token is a pure function of the whole token
  sequence (prompt + generated so far), so a continuation with
  ``prompt = prompt_ids + generated_ids`` produces exactly the tokens an uninterrupted
  run would have produced (the "exact seam" property of a greedy engine). Token ``t``
  detokenizes to ``" w<t>"``.
* Prompts: a string is tokenized word by word after a BOS id 1; a token-id list is used as
  is; chat messages are rendered with a toy template and tokenized.
* OpenAI streaming shape: role chunk first (chat), one token per chunk, finish_reason on
  the last chunk, a usage-only chunk with ``stream_options.include_usage``, ``[DONE]``.
* vLLM stop strings: the last ``max_stop_len - 1`` characters are withheld while
  streaming (unless include_stop_str_in_output) and flushed by the final chunk.
* ``--abort-return-token-ids``: the abort chunk carries ``generated_token_ids`` (all
  generated) in the choice and ``prompt_token_ids`` (chat: top level; completions: in the
  choice) and is sent even before the first token; an aborted non-streaming response
  carries ``token_ids`` / ``prompt_token_ids``.
* ``--sleep-reject-new``: while asleep new requests get 503 + Retry-After: 1 and
  ``{"error": {"type": "EngineSleeping"}}`` (or, with ``reject_in_stream``, an SSE stream
  whose only event is that error, as when the check fires inside generate()).
* ``POST /sleep[?mode=abort|wait]``, ``/wake_up``, ``/is_sleeping``, ``/v1/models``.
* Served with ``handler_cancellation=True`` (test harness), a closed client connection
  cancels the generation, as vLLM aborts a request whose connection closed
  (``disconnected`` counts them).
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import aiohttp
from aiohttp import web

MAX_MODEL_LEN = 4096


def next_token(seq: list[int]) -> int:
    tail = seq[-3:]
    return (sum(tail) * 31 + len(seq) * 7) % 997 + 3


def detok(token: int) -> str:
    return f" w{token}"


def tokenize(text: str) -> list[int]:
    return [1] + [sum(ord(c) for c in word) % 900 + 50 for word in text.split()]


def render_chat(messages: list[dict]) -> str:
    return "".join(f"<{m['role']}> {m['content']} " for m in messages) + "<assistant>"


def reference_tokens(prompt_ids: list[int], n: int) -> list[int]:
    seq = list(prompt_ids)
    out = []
    for _ in range(n):
        token = next_token(seq)
        seq.append(token)
        out.append(token)
    return out


class _Req:
    def __init__(self) -> None:
        self.abort = False
        self.aborted = asyncio.Event()
        self.generated: list[int] = []

    def do_abort(self) -> None:
        self.abort = True
        self.aborted.set()


class FakeEngine:
    def __init__(
        self,
        name: str,
        *,
        token_delay_s: float = 0.005,
        hold_at: int | None = None,
        abort_ids: bool = True,
        reject_new: bool = True,
        reject_in_stream: bool = False,
        hold_eof: bool = False,
        hold_sleep: bool = False,
        fail_sleep: bool = False,
        hold_abort_output: bool = False,
        break_at: int | None = None,
        break_requests: int | None = None,
    ) -> None:
        self.name = name
        self.token_delay_s = token_delay_s
        #: generation parks before token ``hold_at`` until the request is aborted.
        self.hold_at = hold_at
        self.abort_ids = abort_ids
        self.reject_new = reject_new
        self.reject_in_stream = reject_in_stream
        #: a stream waits after [DONE], before ending the response, until cancelled.
        self.hold_eof = hold_eof
        #: /sleep waits after the engine went to sleep until ``release_sleep``.
        self.hold_sleep = hold_sleep
        self._sleep_release = asyncio.Event()
        #: /sleep aborts the running requests, then fails (500) and rolls back (awake).
        self.fail_sleep = fail_sleep
        #: an aborted generation emits its abort output only after ``release_abort_output``.
        self.hold_abort_output = hold_abort_output
        self._abort_output_release = asyncio.Event()
        #: a stream's connection is dropped (no abort, no [DONE]) before token ``break_at``;
        #: only for the first ``break_requests`` streams (None = all).
        self.break_at = break_at
        self.break_requests = break_requests
        self.sleeping = False
        self.requests: list[dict[str, Any]] = []
        self.active: dict[str, _Req] = {}
        #: Generations cancelled because the client (the sidecar) closed the connection.
        self.disconnected = 0
        self._wake = asyncio.Event()
        self._wake.set()

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/completions", self._completions)
        app.router.add_post("/v1/chat/completions", self._chat)
        app.router.add_post("/v1/embeddings", self._embeddings)
        app.router.add_post("/sleep", self._sleep)
        app.router.add_post("/wake_up", self._wake_up)
        app.router.add_get("/is_sleeping", self._is_sleeping)
        app.router.add_get("/v1/models", self._models)
        app.router.add_get("/health", self._health)
        app.router.add_get("/metrics", self._metrics)
        return app

    def generation_requests(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["path"].startswith("/v1/")]

    async def wait_generated(self, n: int, timeout_s: float = 5.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while loop.time() < deadline:
            if any(len(r.generated) >= n for r in self.active.values()):
                return
            await asyncio.sleep(0.001)
        raise AssertionError(f"engine {self.name} never generated {n} tokens")

    def put_to_sleep(self) -> None:
        """Sleep without any sidecar noticing (e.g. a vLLM restart into sleep)."""
        self.sleeping = True
        self._wake.clear()

    # ------------------------------------------------------------- endpoints

    async def _record(self, request: web.Request) -> dict:
        raw = await request.read()
        body = json.loads(raw) if raw else {}
        self.requests.append({"path": request.path, "headers": dict(request.headers), "body": body, "raw": raw})
        return body

    def _rejected(self, stream: bool) -> web.StreamResponse | None:
        if not (self.sleeping and self.reject_new):
            return None
        err = {"error": {"message": "The engine is sleeping", "type": "EngineSleeping", "param": None, "code": 503}}
        if stream and self.reject_in_stream:
            return web.Response(text="data: " + json.dumps(err) + "\n\ndata: [DONE]\n\n",
                                content_type="text/event-stream")
        return web.json_response(err, status=503, headers={"Retry-After": "1"})

    async def _completions(self, request: web.Request) -> web.StreamResponse:
        body = await self._record(request)
        rejected = self._rejected(bool(body.get("stream")))
        if rejected is not None:
            return rejected
        await self._wake.wait()
        prompt = body["prompt"]
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            prompt_ids = list(prompt)
        else:
            prompt_ids = tokenize(prompt[0] if isinstance(prompt, list) else prompt)
        limit = body.get("max_tokens")
        return await self._generate(request, body, prompt_ids, 16 if limit is None else limit, chat=False)

    async def _chat(self, request: web.Request) -> web.StreamResponse:
        body = await self._record(request)
        rejected = self._rejected(bool(body.get("stream")))
        if rejected is not None:
            return rejected
        await self._wake.wait()
        prompt_ids = tokenize(render_chat(body["messages"]))
        limit = body.get("max_completion_tokens") or body.get("max_tokens") or (MAX_MODEL_LEN - len(prompt_ids))
        return await self._generate(request, body, prompt_ids, limit, chat=True)

    async def _embeddings(self, request: web.Request) -> web.StreamResponse:
        await self._record(request)
        rejected = self._rejected(False)
        if rejected is not None:
            return rejected
        return web.json_response({"data": [{"embedding": [0.5], "index": 0}], "served_by": self.name})

    async def _generate(self, request, body, prompt_ids, max_tokens, *, chat):
        stream = bool(body.get("stream"))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        include_stop = bool(body.get("include_stop_str_in_output"))
        withhold = 0 if include_stop or not stops else max(len(s) for s in stops) - 1
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:12]
        req = _Req()
        self.active[rid] = req
        base = {"id": rid, "object": "chat.completion.chunk" if chat else "text_completion",
                "created": 1700000000, "model": body.get("model")}
        seq = list(prompt_ids)

        def chunk(text: str | None, finish: str | None, role: bool = False) -> dict:
            if chat:
                delta: dict[str, Any] = {"role": "assistant"} if role else {}
                if text is not None:
                    delta["content"] = text
                choice = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
            else:
                choice = {"index": 0, "text": text or "", "logprobs": None, "finish_reason": finish,
                          "stop_reason": None}
            return dict(base, choices=[choice], usage=None)

        def abort_chunk(text: str) -> dict:
            obj = chunk(text, "abort")
            if self.abort_ids:
                obj["choices"][0]["generated_token_ids"] = list(req.generated)
                if chat:
                    obj["prompt_token_ids"] = list(prompt_ids)
                else:
                    obj["choices"][0]["prompt_token_ids"] = list(prompt_ids)
            return obj

        def usage() -> dict:
            return {"prompt_tokens": len(prompt_ids), "completion_tokens": len(req.generated),
                    "total_tokens": len(prompt_ids) + len(req.generated)}

        def stop_cut(text: str) -> int | None:
            hits = [(text.find(s), s) for s in stops if s and text.find(s) != -1]
            if not hits:
                return None
            pos, s = min(hits)
            return pos + len(s) if include_stop else pos

        async def step(index: int) -> bool:
            if self.hold_at is not None and index == self.hold_at:
                await req.aborted.wait()
            await asyncio.sleep(self.token_delay_s)
            if req.abort and self.hold_abort_output:
                await self._abort_output_release.wait()
            return not req.abort

        try:
            if not stream:
                text, finish = "", "length"
                for index in range(max_tokens):
                    if not await step(index):
                        finish = "abort"
                        break
                    token = next_token(seq)
                    seq.append(token)
                    req.generated.append(token)
                    text += detok(token)
                    cut = stop_cut(text)
                    if cut is not None:
                        text, finish = text[:cut], "stop"
                        break
                if chat:
                    choice = {"index": 0, "message": {"role": "assistant", "content": text}, "logprobs": None,
                              "finish_reason": finish, "stop_reason": None}
                else:
                    choice = {"index": 0, "text": text, "logprobs": None, "finish_reason": finish,
                              "stop_reason": None}
                obj = dict(base, object="chat.completion" if chat else "text_completion", choices=[choice],
                           usage=usage())
                if finish == "abort" and self.abort_ids:
                    choice["token_ids"] = list(req.generated)
                    if chat:
                        obj["prompt_token_ids"] = list(prompt_ids)
                    else:
                        choice["prompt_token_ids"] = list(prompt_ids)
                return web.json_response(obj, headers={"x-served-by": self.name})

            resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream",
                                                            "x-served-by": self.name})
            await resp.prepare(request)
            try:
                role_sent = False
                full, sent = "", 0
                for index in range(max_tokens):
                    if self.break_at is not None and index == self.break_at and self.break_requests != 0:
                        if self.break_requests is not None:
                            self.break_requests -= 1
                        request.transport.close()  # an engine crash mid-stream
                        break
                    ok = await step(index)
                    if chat and not role_sent:
                        await resp.write(_sse(chunk("", None, role=True)))
                        role_sent = True
                    if not ok:
                        if req.generated or self.abort_ids:
                            await resp.write(_sse(abort_chunk(full[sent:])))
                        break
                    token = next_token(seq)
                    seq.append(token)
                    req.generated.append(token)
                    full += detok(token)
                    cut = stop_cut(full)
                    if cut is not None:
                        await resp.write(_sse(chunk(full[sent:cut], "stop")))
                        break
                    if index == max_tokens - 1:
                        await resp.write(_sse(chunk(full[sent:], "length")))
                        break
                    upto = max(sent, len(full) - withhold)
                    await resp.write(_sse(chunk(full[sent:upto], None)))
                    sent = upto
                if include_usage:
                    await resp.write(_sse(dict(base, choices=[], usage=usage())))
                await resp.write(b"data: [DONE]\n\n")
                if self.hold_eof:
                    await asyncio.Event().wait()
                await resp.write_eof()
            except (ConnectionResetError, ConnectionError):
                req.do_abort()
            return resp
        except asyncio.CancelledError:
            self.disconnected += 1
            raise
        finally:
            self.active.pop(rid, None)

    async def _sleep(self, request: web.Request) -> web.Response:
        self.requests.append({"path": "/sleep", "headers": dict(request.headers), "query": request.query_string})
        mode = request.query.get("mode", "abort")
        self.sleeping = True
        self._wake.clear()
        if mode == "wait":
            while self.active:
                await asyncio.sleep(0.005)
        for req in list(self.active.values()):
            req.do_abort()
        await asyncio.sleep(0.02)  # the weight offload
        if self.hold_sleep:
            await self._sleep_release.wait()
        if self.fail_sleep:
            self.sleeping = False
            self._wake.set()
            return web.json_response({"error": {"message": "sleep failed", "type": "InternalServerError"}},
                                     status=500)
        return web.Response(status=200)

    def release_sleep(self) -> None:
        self._sleep_release.set()

    def release_abort_output(self) -> None:
        self._abort_output_release.set()

    async def _wake_up(self, request: web.Request) -> web.Response:
        self.requests.append({"path": "/wake_up", "headers": dict(request.headers)})
        self.sleeping = False
        self._wake.set()
        return web.Response(status=200)

    async def _health(self, request: web.Request) -> web.Response:
        return web.Response(status=200)

    async def _is_sleeping(self, request: web.Request) -> web.Response:
        return web.json_response({"is_sleeping": self.sleeping})

    async def _models(self, request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": "m", "max_model_len": MAX_MODEL_LEN}]})

    async def _metrics(self, request: web.Request) -> web.Response:
        return web.Response(text=f"vllm:num_requests_running {len(self.active)}\n", content_type="text/plain")


class FakeGateway:
    """Routes to the first routable pod not named in x-tre-exclude-pod; 503 +
    Retry-After when there is none. Streams the pod's answer back. ``keepalive=False``:
    no idle connection on either side (each response closes its connection)."""

    def __init__(self, *, keepalive: bool = True) -> None:
        self.keepalive = keepalive
        self.pods: dict[str, str] = {}
        self.routable: list[str] = []
        self.requests: list[dict[str, Any]] = []
        self.session: aiohttp.ClientSession | None = None

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self._handle)
        app.on_cleanup.append(self._cleanup)
        return app

    async def _cleanup(self, app) -> None:
        if self.session is not None:
            await self.session.close()

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        raw = await request.read()
        exclude = {p.strip() for v in request.headers.getall("x-tre-exclude-pod", []) for p in v.split(",")}
        record = {"path": request.path, "headers": dict(request.headers), "raw": raw,
                  "body": json.loads(raw) if raw else None, "exclude": exclude}
        self.requests.append(record)
        candidates = [p for p in self.routable if p not in exclude]
        if not candidates:
            record["status"] = 503
            resp = web.json_response({"error": {"message": "no routable pod", "type": "ServiceUnavailable"}},
                                     status=503, headers={"Retry-After": "0"})
            if not self.keepalive:
                resp.force_close()
            return resp
        pod = candidates[0]
        record["target"] = pod
        if self.session is None:
            self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=not self.keepalive))
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
        upstream = await self.session.request(request.method, self.pods[pod] + request.path_qs, data=raw,
                                              headers=headers)
        out_headers = {k: v for k, v in upstream.headers.items()
                       if k.lower() not in ("content-length", "transfer-encoding", "date", "server", "connection")}
        out_headers["target-pod"] = pod
        if upstream.status != 200:
            # Like the TRE gateway plugin (gateway.go responseErrorProcessingWithHeaders /
            # util.go generateErrorMessage): a non-200 upstream answer is replaced by the
            # plugin's own OpenAI error whose message is the upstream body as a string; the
            # upstream headers are dropped.
            try:
                body = (await upstream.read()).decode("utf-8", "replace")
            finally:
                upstream.release()
            error = {"message": body, "type": "overloaded_error" if upstream.status == 503 else "api_error",
                     "code": "service_unavailable" if upstream.status == 503 else None, "param": None}
            resp = web.json_response({"error": error}, status=upstream.status, headers={"target-pod": pod})
            if not self.keepalive:
                resp.force_close()
            return resp
        try:
            resp = web.StreamResponse(status=upstream.status, headers=out_headers)
            if not self.keepalive:
                resp.force_close()
            await resp.prepare(request)
            async for data in upstream.content.iter_any():
                await resp.write(data)
        finally:
            # A body not read to the end closes the pod connection: like Envoy, a client
            # that went away resets the upstream request.
            upstream.release()
        await resp.write_eof()
        return resp


def _sse(obj: dict) -> bytes:
    return ("data: " + json.dumps(obj) + "\n\n").encode()

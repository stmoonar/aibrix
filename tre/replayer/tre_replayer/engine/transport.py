"""The wire of the sending core: async HTTP, pooled, one SSE parser.

Two transports, both on ``httpx`` inside an asyncio loop (one loop per process - the
concurrency model of the paper's client, see :mod:`tre_replayer.engine.procpool`), and
both feeding :class:`tre_replayer.engine.stream.StreamParser`:

* :class:`HttpxStreamTransport` - posts the exact bytes a profile built
  (:mod:`tre_replayer.engine.api`): the calibration and replay requests. No retry, ever
  (an open-loop cell must offer each request once). Connections are pooled and kept
  alive; the pool is sized for the peak in-flight so no request waits for a connection.
* :class:`OpenAIChatTransport` - the ``e1_v1`` profile: the OpenAI SDK's
  ``AsyncOpenAI.chat.completions.create`` built exactly as v1 built it (same client
  options, same ``DefaultAsyncHttpxClient`` limits, same retries), so the request bytes
  are the SDK's; only the answer is read through the shared parser instead of the SDK's
  iterator (which would hide the role-only chunk, SSE comments and the reissue marks).
  Every HTTP attempt - including the SDK's own retries - is logged by an httpx event
  hook (``attempt_log``), without touching the request.

:func:`send_sync` is the blocking face of :class:`HttpxStreamTransport`
(:func:`tre_replayer.engine.stream.stream_request`): a per-thread event loop and pool.
"""
from __future__ import annotations

import asyncio
import contextvars
import sys
import threading
import time
from typing import Any, Optional

from tre_replayer.engine.stream import (
    StreamParser,
    StreamResult,
    _error_body_text,
    is_client_timeout,
    lower_headers,
    pod_from_headers,
    reissue_from_headers,
)

TRANSPORT_HTTPX = "httpx"
TRANSPORT_OPENAI_SDK = "openai_sdk"

#: A connection idle longer than this is never reused. Below every server-side idle
#: close on the paths the client talks to - Envoy 1 h, and uvicorn's 5 s when a pod is
#: addressed directly - so a reuse never races the server closing it. Environment
#: override: ``TRE_SENDER_KEEPALIVE_EXPIRY_S``.
DEFAULT_KEEPALIVE_EXPIRY_S = 4.0
KEEPALIVE_ENV = "TRE_SENDER_KEEPALIVE_EXPIRY_S"


def _keepalive_default() -> float:
    import os

    try:
        return float(os.environ.get(KEEPALIVE_ENV) or DEFAULT_KEEPALIVE_EXPIRY_S)
    except ValueError:
        return DEFAULT_KEEPALIVE_EXPIRY_S

#: urllib (the transport before 2026-09-30) sent ``Accept-Encoding: identity``; httpx
#: would offer gzip. Pinned so the answer's bytes are what they were.
IDENTITY_ENCODING = {"Accept-Encoding": "identity"}


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    try:
        import httpx

        out["httpx"] = httpx.__version__
    except ImportError:  # pragma: no cover
        out["httpx"] = None
    try:
        import httpcore

        out["httpcore"] = httpcore.__version__
    except ImportError:  # pragma: no cover
        out["httpcore"] = None
    return out


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _is_closed(client) -> bool:
    """httpx: an ``is_closed`` property; the OpenAI SDK: an ``is_closed()`` method."""
    closed = getattr(client, "is_closed", False)
    return bool(closed() if callable(closed) else closed)


class _PerLoop:
    """One client per event loop: an httpx client is bound to the loop it first ran on,
    and a sender may be driven by several ``asyncio.run`` calls (tests, closed loops)."""

    def __init__(self, factory) -> None:
        self._factory = factory
        self._loop = None
        self._client = None

    def get(self):
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop or _is_closed(self._client):
            self._client = self._factory()
            self._loop = loop
        return self._client

    async def aclose(self) -> None:
        client, self._client, self._loop = self._client, None, None
        if client is None:
            return
        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        try:
            await close()
        except Exception:  # noqa: BLE001 - closing never fails a run
            pass


def _interrupted(res: StreamResult, parser: StreamParser, status: int, exc: BaseException,
                 end_ms: float, target_pod_v1: Optional[str]) -> StreamResult:
    """A 2xx stream that failed after its headers: strict = transport failure (status 0,
    as before 2026-09-30), v1 = a success whose iteration raised (v1's audit flag)."""
    res.stream_interrupted = True
    res.interrupt_error = f"{type(exc).__name__}: {exc}"
    # What the answer said before it broke still counts: a stitched stream stays stitched.
    res.tre_continued = parser.continued
    _fill_v1(res, parser, status, end_ms if parser.v1_stop_ms is None else parser.v1_stop_ms, target_pod_v1)
    res.v1_error = parser.v1_error
    return res


def _fill_v1(res: StreamResult, parser: StreamParser, status: int, v1_done_ms: float,
             target_pod_v1: Optional[str]) -> None:
    res.v1_status = status
    res.v1_success = True
    res.v1_first_token_ms = parser.v1_first_token_ms
    res.v1_done_ms = v1_done_ms
    res.v1_prompt_tokens = parser.v1_prompt_tokens
    res.v1_completion_tokens = parser.v1_completion_tokens
    res.v1_total_tokens = parser.v1_total_tokens
    res.v1_finish_reason = parser.v1_finish_reason
    res.v1_target_pod = target_pod_v1
    if parser.v1_error is not None:
        # The SDK raised inside the iteration (error chunk / bad JSON): v1 swallowed it,
        # kept success=True and flagged the request.
        res.stream_interrupted = True
        res.interrupt_error = parser.v1_error
        res.v1_error = parser.v1_error


def _complete(parser: StreamParser, status: int, response_headers: dict | None, retried: Optional[int],
              end_ms: float, epoch: float) -> StreamResult:
    stream_error = parser.stream_error
    res = StreamResult(
        status,
        parser.first_token_ms,
        parser.done_ms if parser.done_seen else end_ms,
        parser.prompt_tokens,
        parser.completion_tokens,
        target_pod=pod_from_headers(response_headers),
        finish_reason=parser.finish_reason,
        tre_continued=parser.continued,
        tre_retried=retried,
        first_token_field=parser.first_token_field,
        stream_error=stream_error,
        error=None if stream_error is None else f"stream error: {stream_error}",
        error_body=parser.error_body,
        start_epoch_s=epoch,
        done_seen=parser.done_seen,
        stream_complete=parser.done_seen or parser.finish_reason is not None,
    )
    v1_done = parser.v1_stop_ms if parser.v1_stop_ms is not None else end_ms
    _fill_v1(res, parser, status, v1_done, (response_headers or {}).get("target-pod"))
    return res


async def _read_stream(response, parser: StreamParser) -> float:
    """Feed the whole body to ``parser``; ms of its end. The body is read to EOF (a
    pooled connection is only reusable once its response is consumed); what follows
    ``[DONE]`` is ignored by the parser."""
    tail = b""
    async for data in response.aiter_bytes():
        tail = parser.feed_block(tail, data)
    return parser.finish(tail)


#: Connections per pool shard, and idle connections a shard keeps alive. httpcore's pool
#: re-scans every connection it holds (and, for each idle one, every connection again)
#: on each request start and end: one pool of a few thousand connections costs whole CPU
#: cores at calibration rates (measured 2026-09-30, replayer/README.md). A shard stays
#: small, a request goes to the least-loaded shard, shards are opened as load needs them.
POOL_SHARD_CONNECTIONS = 64
POOL_SHARD_KEEPALIVE = 16


class _Shards:
    """The pool shards of one event loop (one ``httpx.AsyncClient`` each, sharing one TLS
    context) and their in-flight counts."""

    def __init__(self, make, count: int) -> None:
        self._make = make
        self._count = count
        self.clients: list = []
        self.load: list[int] = []

    def pick(self) -> int:
        if self.clients:
            best = min(range(len(self.load)), key=self.load.__getitem__)
            if self.load[best] < POOL_SHARD_CONNECTIONS // 2 or len(self.clients) >= self._count:
                return best
        self.clients.append(self._make())
        self.load.append(0)
        return len(self.clients) - 1

    @property
    def is_closed(self) -> bool:
        return bool(self.clients) and all(_is_closed(c) for c in self.clients)

    async def aclose(self) -> None:
        for client in self.clients:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass


class HttpxStreamTransport:
    """POST prepared bytes and read the SSE answer, on pooled keep-alive ``httpx``
    clients (sharded, see :data:`POOL_SHARD_CONNECTIONS`). No retries. See the module
    docstring."""

    name = TRANSPORT_HTTPX

    def __init__(self, *, max_connections: Optional[int] = 4096, max_keepalive_connections: Optional[int] = None,
                 keepalive_expiry: Optional[float] = None) -> None:
        self.max_connections = max_connections
        cap = max_connections or POOL_SHARD_CONNECTIONS
        self.shard_connections = min(POOL_SHARD_CONNECTIONS, max(1, int(cap)))
        self.shards = max(1, -(-int(cap) // self.shard_connections))
        self.shard_keepalive = min(self.shard_connections, POOL_SHARD_KEEPALIVE
                                   if max_keepalive_connections is None else int(max_keepalive_connections))
        self.keepalive_expiry = float(_keepalive_default() if keepalive_expiry is None else keepalive_expiry)
        self._ssl = None
        self._clients = _PerLoop(lambda: _Shards(self._make_client, self.shards))

    def _make_client(self):
        import httpx

        if self._ssl is None:
            self._ssl = httpx.create_ssl_context()  # once: loading the CA store costs ~20 ms
        limits = httpx.Limits(max_connections=self.shard_connections,
                              max_keepalive_connections=self.shard_keepalive,
                              keepalive_expiry=self.keepalive_expiry)
        # timeout=None here; every request passes its own.
        return httpx.AsyncClient(limits=limits, timeout=None, verify=self._ssl, headers=dict(IDENTITY_ENCODING))

    def provenance(self) -> dict:
        return {"transport": self.name, "http": "HTTP/1.1 keep-alive (pooled)",
                "retries": "0; one repeat only when the request headers never started to go out (never seen by the server)",
                "pool_max_connections": self.max_connections, "pool_shards_max": self.shards,
                "pool_shard_connections": self.shard_connections,
                "pool_shard_keepalive_connections": self.shard_keepalive,
                "keepalive_expiry_s": self.keepalive_expiry,
                "timeout": "per request: connect / each read / write, seconds = the profile's", **_versions()}

    async def prepare(self) -> None:
        shards = self._clients.get()
        if not shards.clients:
            shards.pick()

    async def aclose(self) -> None:
        await self._clients.aclose()

    async def send(self, url: str, headers: dict[str, str], body: bytes, timeout_s: float) -> StreamResult:
        shards = self._clients.get()
        index = shards.pick()
        shards.load[index] += 1
        try:
            first_start = time.perf_counter()
            res, wire = await self._send(shards.clients[index], url, headers, body, timeout_s)
            if res.status == 0 and not wire["sent"] and not res.timed_out:
                # The request headers never started to go out (no
                # http11.send_request_headers.started: a refused / failed connect, a
                # pooled connection found closed before the write): the server never saw
                # the request, so repeating it once adds no load the schedule did not
                # plan. A dead kept-alive connection that takes the write and fails on
                # the read counts as sent and is not repeated. Nothing else is ever
                # repeated.
                first_log = list(res.attempt_log)
                spent_ms = (time.perf_counter() - first_start) * 1000.0
                res, wire = await self._send(shards.clients[index], url, headers, body, timeout_s)
                res.attempt_log = first_log + list(res.attempt_log)
                res.attempts = len(res.attempt_log)
                res.transport_retries = 1
                res.conn_acquire_ms = spent_ms + (res.conn_acquire_ms or 0.0)
            return res
        finally:
            shards.load[index] -= 1

    async def _send(self, client, url: str, headers: dict[str, str], body: bytes,
                    timeout_s: float) -> tuple:
        res, wire = await self._send_once(client, url, headers, body, timeout_s)
        if wire["sent_at"] is not None:
            res.conn_acquire_ms = max(0.0, (wire["sent_at"] - wire["start"]) * 1000.0)
            res.connection_reused = not wire["connected"]
        return res, wire

    async def _send_once(self, client, url: str, headers: dict[str, str], body: bytes,
                         timeout_s: float) -> tuple:
        start = time.perf_counter()
        epoch = time.time()
        wire = {"start": start, "connected": False, "sent": False, "sent_at": None}

        async def trace(event: str, info: dict) -> None:
            # httpcore's trace hooks: did this request open a connection, and when did
            # its first byte start to go out.
            if event.startswith("connection.connect_tcp."):
                wire["connected"] = True
            elif event == "http11.send_request_headers.started":
                wire["sent"] = True
                wire["sent_at"] = time.perf_counter()

        attempt = {"t": epoch, "status": None}
        parser: Optional[StreamParser] = None
        status = 0
        response_headers = None
        try:
            async with client.stream("POST", url, content=body, headers=headers, timeout=timeout_s,
                                     extensions={"trace": trace}) as response:
                status = response.status_code
                attempt["status"] = status
                response_headers = lower_headers(response.headers)
                if not 200 <= status < 300:
                    try:
                        raw = await response.aread()
                    except Exception:  # noqa: BLE001 - losing the body never loses the record
                        raw = None
                    elapsed = _ms(start)
                    return StreamResult(
                        status, None, elapsed, error=f"HTTP {status}", error_body=_error_body_text(raw),
                        error_headers=response_headers, target_pod=pod_from_headers(response_headers),
                        start_epoch_s=epoch, attempt_log=[attempt], v1_status=status, v1_success=False,
                        v1_error=f"HTTP {status}", v1_done_ms=elapsed, v1_target_pod="",
                    ), wire
                continued, retried = reissue_from_headers(response_headers)
                parser = StreamParser(start, continued=continued)
                end_ms = await _read_stream(response, parser)
            res = _complete(parser, status, response_headers, retried, end_ms, epoch)
            res.attempt_log = [attempt]
            return res, wire
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - every transport failure is a record
            elapsed = _ms(start)
            _, retried = reissue_from_headers(response_headers)
            if parser is not None and parser.done_seen:
                # The answer was complete; only the drain after [DONE] failed.
                res = _complete(parser, status, response_headers, retried, elapsed, epoch)
                res.attempt_log = [attempt]
                return res, wire
            res = StreamResult(0, None, elapsed, error=type(exc).__name__, timed_out=is_client_timeout(exc),
                               start_epoch_s=epoch, attempt_log=[attempt], tre_retried=retried)
            if parser is not None:
                return _interrupted(res, parser, status, exc, elapsed,
                                    (response_headers or {}).get("target-pod")), wire
            res.v1_status = None
            res.v1_success = False
            res.v1_error = str(exc)
            res.v1_done_ms = elapsed
            res.v1_target_pod = ""
            return res, wire


# ------------------------------------------------------------------ the e1_v1 transport


class AttemptTracker:
    """One request's HTTP attempts (1 + SDK retries), filled by the client's event hooks."""

    __slots__ = ("attempts",)

    def __init__(self) -> None:
        self.attempts: list[dict] = []

    @property
    def count(self) -> int:
        return len(self.attempts)

    @property
    def last_status(self) -> Optional[int]:
        return self.attempts[-1]["status"] if self.attempts else None


_ATTEMPT_TRACKER: "contextvars.ContextVar[Optional[AttemptTracker]]" = contextvars.ContextVar(
    "tre_replayer_attempt_tracker", default=None
)


async def _on_request_hook(request) -> None:
    tracker = _ATTEMPT_TRACKER.get()
    if tracker is not None:
        tracker.attempts.append({"t": time.time(), "status": None})


async def _on_response_hook(response) -> None:
    tracker = _ATTEMPT_TRACKER.get()
    if tracker is not None and tracker.attempts:
        tracker.attempts[-1]["status"] = response.status_code


def normalize_gateway_endpoint(base_url: str) -> str:
    """``http://host:port`` or ``http://host:port/v1`` -> ``http://host:port`` (the SDK's
    base URL is this plus ``/v1``, as v1's ``gateway_endpoint``)."""
    url = (base_url or "").strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


#: v1 replaced an empty / "dummy" API key with this before building its client.
V1_PLACEHOLDER_API_KEY = "dummy-key-for-local-gateway"


#: SDK clients per worker process with ``pool_shards=None``. The default is v1's one
#: client per process; sharding (every shard built exactly like v1's client) cut the
#: send-lateness p99 at 200-300 rps only from 28-36 to 21-29 ms - the SDK's own
#: per-request work dominates there (replayer/README.md) - so it stays opt-in.
E1_MAX_POOL_SHARDS = 16


class OpenAIChatTransport:
    """v1's client: ``openai.AsyncOpenAI`` with v1's options (per event loop, sharded like
    :class:`HttpxStreamTransport`; ``pool_shards=1`` is v1's single client), and the
    request made through ``chat.completions.create``. See the module docstring."""

    name = TRANSPORT_OPENAI_SDK

    def __init__(self, gateway_endpoint: str, *, api_key: str = "", max_retries: int = 2,
                 timeout_s: float = 300.0, routing_strategy: Optional[str] = "least-gpu-cache",
                 streaming: bool = True, pool_shards: Optional[int] = 1) -> None:
        self.gateway_endpoint = normalize_gateway_endpoint(gateway_endpoint)
        if not self.gateway_endpoint:
            raise ValueError("the e1_v1 transport needs an explicit gateway base URL")
        self.base_url = f"{self.gateway_endpoint}/v1"
        self.api_key = V1_PLACEHOLDER_API_KEY if (not api_key or api_key == "dummy") else api_key
        if int(max_retries) < 0:
            raise ValueError("max_retries cannot be negative")
        self.max_retries = int(max_retries)
        self.timeout_s = float(timeout_s)
        self.routing_strategy = routing_strategy or None
        self.streaming = bool(streaming)
        self.pool_shards = E1_MAX_POOL_SHARDS if pool_shards is None else max(1, int(pool_shards))
        self._ssl = None
        self._clients = _PerLoop(lambda: _Shards(self.make_client, self.pool_shards))

    def make_client(self):
        """Exactly v1's ``WorkerProcess.create_client``: the SDK's default httpx client
        (same limits and redirects) with only the attempt hooks added (and one TLS
        context shared by the shards: loading the CA store costs ~20 ms per client)."""
        import openai

        extra = {}
        if self.pool_shards > 1:
            if self._ssl is None:
                import httpx

                self._ssl = httpx.create_ssl_context()
            extra["verify"] = self._ssl
        http_client = openai.DefaultAsyncHttpxClient(
            timeout=self.timeout_s,
            event_hooks={"request": [_on_request_hook], "response": [_on_response_hook]},
            **extra,
        )
        client = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=self.max_retries,
                                    timeout=self.timeout_s, http_client=http_client)
        if self.routing_strategy:
            client = client.with_options(default_headers={"routing-strategy": self.routing_strategy})
        return client

    def provenance(self) -> dict:
        try:
            import openai

            sdk = openai.__version__
        except ImportError:  # pragma: no cover
            sdk = None
        return {"transport": self.name, "openai_version": sdk, "base_url": self.base_url,
                "max_retries": self.max_retries, "timeout_s": self.timeout_s,
                "routing_strategy_header": self.routing_strategy, "streaming": self.streaming,
                "pool": "openai.DefaultAsyncHttpxClient (1000 connections / 100 keep-alive) per client",
                "pool_shards_max": self.pool_shards, **_versions()}

    async def prepare(self) -> None:
        shards = self._clients.get()
        if not shards.clients:
            shards.pick()
        # The SDK loads its resource modules lazily on first attribute access.
        _ = shards.clients[0].chat.completions

    async def aclose(self) -> None:
        await self._clients.aclose()

    async def send_chat(self, kwargs: dict[str, Any]) -> StreamResult:
        shards = self._clients.get()
        index = shards.pick()
        shards.load[index] += 1
        try:
            return await self._send_chat(shards.clients[index], kwargs)
        finally:
            shards.load[index] -= 1

    async def _send_chat(self, client, kwargs: dict[str, Any]) -> StreamResult:
        tracker = AttemptTracker()
        token = _ATTEMPT_TRACKER.set(tracker)
        start = time.perf_counter()
        epoch = time.time()
        try:
            if self.streaming:
                res = await self._stream(client, kwargs, start, epoch)
            else:
                res = await self._whole(client, kwargs, start, epoch)
        finally:
            _ATTEMPT_TRACKER.reset(token)
        res.attempts = tracker.count
        res.attempt_log = list(tracker.attempts)
        if tracker.count > 1:
            # The strict basis leaves the retries out: it is timed from the last attempt.
            # A single attempt is timed from the call, like every other profile.
            res.last_attempt_offset_ms = max(0.0, (tracker.attempts[-1]["t"] - epoch) * 1000.0)
        return res

    @staticmethod
    def _failure(exc: BaseException, tracker_status: Optional[int], start: float, epoch: float) -> StreamResult:
        status = getattr(exc, "status_code", None)
        if status is None:
            status = tracker_status
        elapsed = _ms(start)
        response = getattr(exc, "response", None)
        error_headers = lower_headers(getattr(response, "headers", None)) if response is not None else None
        body = None
        if response is not None:
            try:
                body = _error_body_text(response.text)
            except Exception:  # noqa: BLE001
                body = None
        try:
            import openai

            timed_out = isinstance(exc, openai.APITimeoutError) or is_client_timeout(exc)
        except ImportError:  # pragma: no cover
            timed_out = is_client_timeout(exc)
        has_status = bool(status) and getattr(exc, "status_code", None) is not None
        return StreamResult(
            int(status) if has_status else 0, None, elapsed,
            error=f"HTTP {status}" if has_status else type(exc).__name__,
            error_body=body if has_status else None, error_headers=error_headers if has_status else None,
            target_pod=pod_from_headers(error_headers) if has_status else None,
            timed_out=timed_out and not has_status, start_epoch_s=epoch,
            v1_status=status, v1_success=False, v1_error=str(exc), v1_done_ms=elapsed, v1_target_pod="",
        )

    async def _stream(self, client, kwargs: dict, start: float, epoch: float) -> StreamResult:
        stream = None
        parser: Optional[StreamParser] = None
        status = 0
        response_headers = None
        try:
            try:
                stream = await client.chat.completions.create(**kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - v1: every failure of create() is a record
                return self._failure(exc, _ATTEMPT_TRACKER.get().last_status, start, epoch)
            response = stream.response
            status = response.status_code
            response_headers = lower_headers(response.headers)
            continued, retried = reissue_from_headers(response_headers)
            parser = StreamParser(start, continued=continued)
            try:
                end_ms = await _read_stream(response, parser)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - v1 swallowed this and kept success
                elapsed = _ms(start)
                if parser.done_seen:
                    res = _complete(parser, status, response_headers, retried, elapsed, epoch)
                    res.v1_status = _ATTEMPT_TRACKER.get().last_status
                    return res
                res = StreamResult(0, None, elapsed, error=type(exc).__name__, timed_out=is_client_timeout(exc),
                                   start_epoch_s=epoch, tre_retried=retried)
                res = _interrupted(res, parser, status, exc, elapsed, response_headers.get("target-pod")
                                   if response_headers else None)
                res.v1_status = _ATTEMPT_TRACKER.get().last_status
                return res
            res = _complete(parser, status, response_headers, retried, end_ms, epoch)
            res.v1_status = _ATTEMPT_TRACKER.get().last_status
            return res
        finally:
            if stream is not None:
                try:
                    await stream.close()
                except Exception:  # noqa: BLE001
                    pass

    async def _whole(self, client, kwargs: dict, start: float, epoch: float) -> StreamResult:
        try:
            response = await client.chat.completions.create(**kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            return self._failure(exc, _ATTEMPT_TRACKER.get().last_status, start, epoch)
        end_ms = _ms(start)
        usage = getattr(response, "usage", None)
        choices = getattr(response, "choices", None) or []
        finish = choices[0].finish_reason if choices else None
        status = _ATTEMPT_TRACKER.get().last_status or 200
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        total = getattr(usage, "total_tokens", None)
        return StreamResult(
            status, None, end_ms, prompt, completion, finish_reason=finish, start_epoch_s=epoch, done_seen=True,
            stream_complete=True,
            v1_status=_ATTEMPT_TRACKER.get().last_status, v1_success=True, v1_done_ms=end_ms,
            v1_prompt_tokens=prompt, v1_completion_tokens=completion, v1_total_tokens=total,
            v1_finish_reason=finish,
            # v1 looked for ``response.response.headers`` on a ChatCompletion, which has no
            # such attribute: its non-streaming records never named a pod.
            v1_target_pod="",
        )


# ----------------------------------------------------------------- the blocking face

_LOCAL = threading.local()
#: Per calling thread: one request at a time, so a small pool with one kept-alive
#: connection is all a closed-loop worker needs.
SYNC_POOL_CONNECTIONS = 4
SYNC_POOL_KEEPALIVE = 1


def send_sync(url: str, headers: dict[str, str], body: bytes, timeout_s: float) -> StreamResult:
    """Run :meth:`HttpxStreamTransport.send` to completion on this thread's own loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        # Called from inside a running loop (never blocks it): hand it to a helper thread.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tre-send-sync") as pool:
            return pool.submit(send_sync, url, headers, body, timeout_s).result()
    state = getattr(_LOCAL, "state", None)
    if state is None or state[0].is_closed():
        state = (asyncio.new_event_loop(),
                 HttpxStreamTransport(max_connections=SYNC_POOL_CONNECTIONS,
                                      max_keepalive_connections=SYNC_POOL_KEEPALIVE))
        _LOCAL.state = state
    loop, transport = state
    return loop.run_until_complete(transport.send(url, headers, body, timeout_s))

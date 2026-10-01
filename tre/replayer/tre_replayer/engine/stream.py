"""One streamed OpenAI-compatible call: timing, SSE parsing, usage, failure evidence.

This is the answer-side half of the sending core every client shares (the request side
is :mod:`tre_replayer.engine.api` / :mod:`tre_replayer.engine.profiles`, the wire is
:mod:`tre_replayer.engine.transport`). It is free of scheduling and of any driver:

* :class:`StreamParser` - the **only** SSE parser in the tree. Every transport feeds it
  the raw lines of an answer; it keeps two views of the same bytes (below).
* :class:`StreamResult` - what one call returns; :func:`result_fields` turns it into the
  answer's part of a calibration / replay record.
* :func:`stream_request` - the synchronous seam ``(url, headers, body, timeout) ->
  StreamResult`` (closed-loop grid workers, preflight, tests). It runs the pooled async
  transport on a per-thread event loop; there is no second HTTP client behind it.

Two views of one stream
-----------------------
*strict* (``first_token_ms``, ``done_ms``, ``completion_tokens`` ...; what calibration
labels are computed on, :data:`TTFT_BASIS`): the first token is the first chunk carrying
generated text in ``text`` (completions), ``delta.content`` or ``delta.reasoning_content``
/ ``delta.reasoning`` (chat, with a reasoning parser); a chat stream's role-only opening
chunk and its usage-only closing chunk are not tokens. ``done_ms`` is the ``[DONE]``
line (or the end of the body when there is none). An ``{"error": ...}`` chunk inside a
200 stream is recorded (``stream_error``) and reading continues to ``[DONE]``.

*v1* (``v1_*`` fields; what the paper's client ``CustomTraceGenerator`` - and its port,
the ``e1_v1`` profile - reported): exactly what the OpenAI SDK's ``AsyncStream`` plus
v1's loop computed from the same bytes. TTFT is stamped on the first chunk whose
``choices[0].delta.content is not None`` - the role-only chunk (``content: ""``)
included, about one decode step earlier than the strict view; usage and the finish
reason are the last non-null values; an ``{"error": ...}`` chunk (the SDK raises
``APIError``) or undecodable JSON ends the view there; otherwise the view ends at the end
of the body (the SDK consumes the stream to EOF after ``[DONE]``).

Errors: a non-2xx answer or a transport failure comes back with its status and
evidence (``status`` 0 for a transport failure, including one in the middle of a 200
stream - the strict view's reading of an interrupted answer). The v1 view keeps what the
SDK saw instead: a failure mid-stream is ``stream_interrupted`` with ``v1_status`` 200
and ``v1_success`` True, which is how v1 recorded it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

#: What ``first_token_ms`` measures, recorded with every request (``ttft_basis``): the
#: first SSE chunk carrying generated text (see :func:`chunk_token_field`), not the first
#: chunk of any kind.
TTFT_BASIS = "first_chunk_with_text"

#: What ``v1_first_token_ms`` measures: the first chunk whose ``choices[0].delta.content``
#: is not None (the OpenAI SDK object v1's loop tested), the role-only chunk included.
V1_TTFT_BASIS = "first_chunk_with_content_not_none"

#: Response headers, in preference order, that name the pod that served a request.
#: ``target-pod`` / ``target-pod-ip`` are what the AIBrix gateway plugin sets on the
#: routed path (``pkg/plugins/gateway/gateway_rsp_headers.go``); the ``x-`` names are
#: there so a header added at the Envoy layer - which is what the per-model HTTPRoute
#: path would need - is picked up without another client change. The pod *name* is
#: preferred over its address because it survives a pod IP being reused.
POD_HEADER_KEYS = ("target-pod", "x-target-pod", "x-upstream-pod", "target-pod-ip")

#: What the TRE reissue sidecar reports (tre/docs/design/20260927-reissue-sidecar-v2.md):
#: ``x-tre-retried: <attempts>`` on a request that never started on the pod it was routed
#: to and was resent through the gateway; ``x-tre-continued: <segments>`` on a
#: non-streaming answer stitched from segments of several pods. A stream carries the
#: segment count in the extension field ``tre_continued`` of its final (finish_reason)
#: chunk and in a closing SSE comment ``: x-tre-continued: <segments>``.
RETRIED_HEADER = "x-tre-retried"
CONTINUED_HEADER = "x-tre-continued"
CONTINUED_FIELD = "tre_continued"

#: How much of a non-2xx body to keep. An Envoy circuit-breaker body is ~80 bytes and a
#: vLLM JSON error is small too; the cap only bounds a pathological upstream.
MAX_ERROR_BODY_CHARS = 2048

#: Where a streamed chunk carries generated text, in the order they are checked:
#: completions ``choices[].text``; chat ``choices[].delta.<field>`` for these fields.
CHAT_TOKEN_FIELDS = ("content", "reasoning_content", "reasoning")


def pod_from_headers(headers: dict[str, str] | None) -> str | None:
    """First :data:`POD_HEADER_KEYS` entry present in ``headers`` (already lower-cased)."""
    if not headers:
        return None
    for key in POD_HEADER_KEYS:
        value = headers.get(key)
        if value:
            return value
    return None


@dataclass
class StreamResult:
    """Outcome of one streamed completion. Durations are measured from request start
    (the seam times itself); None where genuinely unavailable."""

    status: int
    first_token_ms: float | None
    done_ms: float | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    #: Verbatim body of a non-2xx answer, truncated. A failure cannot be attributed
    #: without it: the serving path has two rejections wearing the same status code -
    #: the engine answering with its own JSON error, and the gateway circuit breaker
    #: rejecting at the Envoy cluster with a plain-text body, having never reached the
    #: engine at all. See ``scripts.openloop.classify_failure``.
    error_body: str | None = None
    #: Lower-cased response headers of a non-2xx answer. The content type and any
    #: ``x-envoy-*`` marker are what the classifier reads.
    error_headers: dict[str, str] | None = None
    #: Name (or address) of the pod that served this request, read from whichever of
    #: :data:`POD_HEADER_KEYS` the answer carried. None when the serving path exposes no
    #: such header - which is the case on the per-model HTTPRoute the campaign uses, so
    #: a None here means "not attributable", never "no pod".
    target_pod: str | None = None
    #: The client gave up before the upstream answered. Its own class, because nothing
    #: is known about what the engine did with the request: folding it into the model's
    #: error budget would read as an engine fault, and folding it into the gateway's
    #: would read as a shed. See ``scripts.openloop.classify_failure``.
    timed_out: bool = False
    #: ``finish_reason`` of the final choice chunk (``abort`` = the client saw a
    #: truncated answer, e.g. a sleep the sidecar could not hide).
    finish_reason: str | None = None
    #: Continuation segments the reissue sidecar stitched in (None = not continued).
    tre_continued: int | None = None
    #: Gateway attempts of a sidecar retry (None = not retried).
    tre_retried: int | None = None
    #: The SSE field the first token arrived in (``text``, ``content``,
    #: ``reasoning_content``, ``reasoning``); None when no token arrived.
    first_token_field: str | None = None
    #: The message of an ``{"error": ...}`` chunk inside a 2xx stream (None = none): the
    #: server failed the request after answering 200. A failure, not a completion.
    stream_error: str | None = None
    # ---- added with the unified client (2026-09-30); every default is "not measured" ----
    #: ``time.time()`` at the start of the call (what the ms offsets are relative to).
    start_epoch_s: float | None = None
    #: The ``[DONE]`` line arrived (a complete stream).
    done_seen: bool = False
    #: A transport failure after the response headers (the SDK's view: an exception while
    #: iterating a 200 stream) and its text ``"<type>: <message>"``.
    stream_interrupted: bool = False
    interrupt_error: str | None = None
    #: HTTP attempts made for this request (1 + client retries) and each attempt's
    #: ``{"t": epoch, "status": code-or-None}``; ms from the call start to the start of
    #: the last attempt (0 without retries): retry waits are before it.
    attempts: int = 1
    attempt_log: list = field(default_factory=list)
    last_attempt_offset_ms: float = 0.0
    # ---- the v1 view of the same answer (module docstring) ----
    v1_status: int | None = None
    v1_success: bool | None = None
    v1_error: str | None = None
    v1_first_token_ms: float | None = None
    v1_done_ms: float | None = None
    v1_prompt_tokens: int = 0
    v1_completion_tokens: int = 0
    v1_total_tokens: int = 0
    v1_finish_reason: str | None = None
    #: v1 read only the ``target-pod`` header ("" when the call failed before headers).
    v1_target_pod: str | None = None
    # ---- connection evidence (httpx transport; None = not measured, e.g. a test seam) ----
    #: A 2xx body ended normally: True when it carried ``[DONE]`` or a finish reason,
    #: False when it just stopped (a truncated answer: a failure on both bases). None =
    #: unknown (a synthetic result).
    stream_complete: bool | None = None
    #: The request went out on a kept-alive connection (no TCP connect for it).
    connection_reused: bool | None = None
    #: ms from the transport call to the request's first byte: waiting for a pool slot,
    #: opening the connection, and a repeated first attempt - part of the send lateness.
    conn_acquire_ms: float | None = None
    #: Attempts repeated by the transport because not one byte of the request had left
    #: (0 or 1); never a request the server could have seen.
    transport_retries: int = 0


def _positive_int(value: Any) -> int | None:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def reissue_from_headers(headers: dict[str, str] | None) -> tuple[int | None, int | None]:
    """(tre_continued, tre_retried) from lower-cased response headers."""
    if not headers:
        return None, None
    return _positive_int(headers.get(CONTINUED_HEADER)), _positive_int(headers.get(RETRIED_HEADER))


def stream_error_message(error: Any) -> str:
    """The message of an in-stream ``error`` value (an OpenAI error object or a string)."""
    if isinstance(error, dict):
        message = error.get("message") or error.get("type") or json.dumps(error, sort_keys=True)
        code = error.get("code")
        return f"{message} (code {code})" if code is not None else str(message)
    return str(error)


def _sdk_error_message(error: Any) -> str:
    """The message the OpenAI SDK gives the ``APIError`` it raises for an error chunk."""
    message = error.get("message") if isinstance(error, dict) else None
    if not message or not isinstance(message, str):
        message = "An error occurred during streaming"
    return message


def chunk_token_field(chunk: dict[str, Any]) -> str | None:
    """The field of ``chunk`` that carries generated text, or None.

    A chat stream opens with a role-only chunk (``{"role": "assistant", "content": ""}``)
    and closes with a usage-only one (``choices: []``); neither is a token. A reasoning
    parser moves an R1-style model's first tokens into ``reasoning_content`` (vLLM <= 0.10
    naming) or ``reasoning`` (later), which is still the first token the client received.
    """
    for choice in chunk.get("choices", []) or []:
        if not isinstance(choice, dict):
            continue
        if choice.get("text"):
            return "text"
        delta = choice.get("delta")
        if isinstance(delta, dict):
            for field_name in CHAT_TOKEN_FIELDS:
                if delta.get(field_name):
                    return field_name
    return None


def _chunk_has_content(chunk: dict[str, Any]) -> bool:
    return chunk_token_field(chunk) is not None


class StreamParser:
    """Incremental SSE parser holding the strict and the v1 view of one answer.

    ``feed(line)`` takes one raw line (bytes or str, with or without its line ending) as
    it arrives; ``finish()`` marks the end of the body. Times are ms from ``start``
    (a ``clock()`` reading taken at the start of the call). Lines after ``[DONE]`` are
    ignored by the strict view (the pre-2026-09-30 sender stopped reading there) and by
    the SDK (it drains them unread).
    """

    __slots__ = (
        "start", "clock", "first_token_ms", "first_token_field", "prompt_tokens",
        "completion_tokens", "finish_reason", "continued", "stream_error", "error_body",
        "done_seen", "done_ms", "v1_first_token_ms", "v1_prompt_tokens",
        "v1_completion_tokens", "v1_total_tokens", "v1_finish_reason", "v1_error",
        "v1_stop_ms", "lines",
    )

    def __init__(self, start: float, clock: Callable[[], float] = time.perf_counter,
                 continued: int | None = None) -> None:
        self.start = start
        self.clock = clock
        self.first_token_ms: float | None = None
        self.first_token_field: str | None = None
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.finish_reason: str | None = None
        self.continued: int | None = continued
        self.stream_error: str | None = None
        self.error_body: str | None = None
        self.done_seen = False
        self.done_ms: float | None = None
        self.v1_first_token_ms: float | None = None
        self.v1_prompt_tokens = 0
        self.v1_completion_tokens = 0
        self.v1_total_tokens = 0
        self.v1_finish_reason: str | None = None
        self.v1_error: str | None = None
        self.v1_stop_ms: float | None = None
        self.lines = 0

    def _now_ms(self) -> float:
        return (self.clock() - self.start) * 1000.0

    def feed(self, raw: bytes | str) -> None:
        if self.done_seen:
            return
        self.lines += 1
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
        line = line.strip()
        if line.startswith(":"):
            # SSE comment; the reissue sidecar closes a stitched stream with
            # ": x-tre-continued: <segments>".
            name, _, value = line[1:].strip().partition(":")
            if name.strip().lower() == CONTINUED_HEADER:
                self.continued = _positive_int(value) or self.continued
            return
        if not line or not line.startswith("data:"):
            return
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            self.done_seen = True
            self.done_ms = self._now_ms()
            return
        v1_open = self.v1_stop_ms is None
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError as exc:
            if v1_open:  # the SDK's sse.json() raises here and v1's loop stops
                self.v1_error = f"JSONDecodeError: {exc}"
                self.v1_stop_ms = self._now_ms()
            return
        if not isinstance(chunk, dict):
            return
        now_ms: float | None = None
        error = chunk.get("error")
        if error is not None and self.stream_error is None:
            self.stream_error = stream_error_message(error)
            self.error_body = payload[:MAX_ERROR_BODY_CHARS]
        if v1_open:
            if error:  # the SDK raises APIError(message) and v1's loop stops
                now_ms = self._now_ms()
                self.v1_error = f"APIError: {_sdk_error_message(error)}"
                self.v1_stop_ms = now_ms
            else:
                choices = chunk.get("choices")
                if choices and isinstance(choices, list):
                    first = choices[0]
                    if isinstance(first, dict):
                        delta = first.get("delta")
                        if (self.v1_first_token_ms is None and isinstance(delta, dict)
                                and delta.get("content") is not None):
                            now_ms = self._now_ms()
                            self.v1_first_token_ms = now_ms
                        if first.get("finish_reason") is not None:
                            self.v1_finish_reason = first["finish_reason"]
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    if usage.get("prompt_tokens") is not None:
                        self.v1_prompt_tokens = usage["prompt_tokens"]
                    if usage.get("completion_tokens") is not None:
                        self.v1_completion_tokens = usage["completion_tokens"]
                    if usage.get("total_tokens") is not None:
                        self.v1_total_tokens = usage["total_tokens"]
        if self.first_token_ms is None:
            token_field = chunk_token_field(chunk)
            if token_field is not None:
                self.first_token_ms = now_ms if now_ms is not None else self._now_ms()
                self.first_token_field = token_field
        for choice in chunk.get("choices") or []:
            if isinstance(choice, dict) and choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        if CONTINUED_FIELD in chunk:
            self.continued = _positive_int(chunk[CONTINUED_FIELD]) or self.continued
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.prompt_tokens = usage.get("prompt_tokens", self.prompt_tokens)
            self.completion_tokens = usage.get("completion_tokens", self.completion_tokens)

    def feed_block(self, pending: bytes, data: bytes) -> bytes:
        """Feed every complete line of ``pending + data``; return the incomplete tail."""
        buf = pending + data if pending else data
        if b"\n" not in buf:
            return buf
        parts = buf.split(b"\n")
        tail = parts.pop()
        for part in parts:
            self.feed(part)
        return tail

    def finish(self, tail: bytes = b"") -> float:
        """The body ended (after an optional incomplete last line); ms of the end."""
        if tail:
            self.feed(tail)
        return self._now_ms()


#: The seam: (url, headers, body_bytes, timeout_s) -> StreamResult.
StreamCall = Callable[[str, dict[str, str], bytes, float], StreamResult]


def stream_request(url: str, headers: dict[str, str], body: bytes, timeout_s: float) -> StreamResult:
    """POST ``body`` to ``url`` and read the SSE answer (see :class:`StreamParser`).

    The synchronous face of :class:`tre_replayer.engine.transport.HttpxStreamTransport`:
    each calling thread gets its own event loop and connection pool, reused across its
    calls. Never raises for an HTTP or transport failure: those come back as a
    StreamResult too."""
    from tre_replayer.engine.transport import send_sync

    return send_sync(url, headers, body, timeout_s)


def is_client_timeout(exc: BaseException) -> bool:
    """True when this transport failure is the client giving up, not the peer refusing.

    httpx raises a ``TimeoutException`` subclass (connect / read / write / pool); the
    standard library surfaces a read timeout as :class:`TimeoutError`
    (``socket.timeout`` is an alias of it since 3.10) or as a
    :class:`urllib.error.URLError` wrapping one in ``reason``; on some stacks only an
    ``OSError`` whose text says so. The text check comes last and is deliberately narrow.
    """
    if isinstance(exc, TimeoutError):
        return True
    try:
        import httpx

        if isinstance(exc, httpx.TimeoutException):
            return True
    except ImportError:  # pragma: no cover - httpx is a dependency of the transport
        pass
    reason = getattr(exc, "reason", None)
    if isinstance(reason, TimeoutError):
        return True
    return "timed out" in str(exc).lower()


def read_error_body(exc) -> str | None:
    """Verbatim body of an HTTPError, best effort.

    Reading it can itself fail on a connection the proxy already reset, and losing the
    body must never lose the request record - an unattributable failure is still a
    failure, and the classifier has a documented fallback for a missing body.
    """
    try:
        raw = exc.read()
    except Exception:  # noqa: BLE001
        return None
    return _error_body_text(raw)


def _error_body_text(raw: Any) -> str | None:
    if not raw:
        return None
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode("utf-8", errors="replace")
    return raw[:MAX_ERROR_BODY_CHARS]


def lower_headers(headers) -> dict[str, str] | None:
    """Response headers as a lower-cased dict, or None when there are none."""
    if not headers:
        return None
    try:
        items = list(headers.items())
    except AttributeError:
        return None
    return {str(key).lower(): str(value) for key, value in items}


def result_fields(res: StreamResult) -> dict[str, Any]:
    """The per-request record fields that come from the answer itself - what every client
    writes for a request, whatever drove it. Token counts are the engine's ``usage``;
    ``first_token_field`` names where the first token arrived."""
    return {
        "ttft_ms": res.first_token_ms,
        "e2e_ms": res.done_ms,
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
        "http_status": res.status,
        "error": res.error,
        "error_body": res.error_body,
        "error_headers": res.error_headers,
        "target_pod": res.target_pod,
        "finish_reason": getattr(res, "finish_reason", None),
        # Reissue sidecar: segments stitched in / gateway attempts of a retry.
        "tre_continued": getattr(res, "tre_continued", None),
        "tre_retried": getattr(res, "tre_retried", None),
        "client_timeout": bool(getattr(res, "timed_out", False)),
        "first_token_field": getattr(res, "first_token_field", None),
        "stream_error": getattr(res, "stream_error", None),
        # What ttft_ms measures (TTFT_BASIS): the first chunk carrying text.
        "ttft_basis": TTFT_BASIS,
    }


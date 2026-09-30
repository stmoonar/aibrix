"""One streamed OpenAI-compatible call: timing, SSE parsing, usage, failure evidence.

This is the sending core every client shares, free of scheduling and of any driver:
:func:`stream_request` posts one body (built by :func:`tre_replayer.engine.api.request_body`)
and returns a :class:`StreamResult`; :func:`result_fields` turns that into the fields of a
per-request record. :class:`tre_replayer.engine.http_sender.StreamingHttpSender` (the
open-loop replayer / calibration sender) and :mod:`tre_replayer.engine.preflight` are
built on it, and a closed-loop client can be too.

Timing: ``first_token_ms`` and ``done_ms`` are measured from the call with
``time.perf_counter`` (the seam times itself). The first token is the first chunk carrying
generated text in ``text`` (completions), ``delta.content`` or ``delta.reasoning_content``
/ ``delta.reasoning`` (chat, with a reasoning parser); a chat stream's role-only opening
chunk and its usage-only closing chunk are not tokens.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

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


#: The seam: (url, headers, body_bytes, timeout_s) -> StreamResult.
StreamCall = Callable[[str, dict[str, str], bytes, float], StreamResult]


def stream_request(url: str, headers: dict[str, str], body: bytes, timeout_s: float) -> StreamResult:
    """POST ``body`` to ``url`` and read the SSE answer: status, TTFT (first chunk carrying a
    token, see :func:`chunk_token_field`) and end-to-end time measured here from the call,
    ``usage`` counts, finish reason, serving pod and the reissue sidecar's marks. Never
    raises for an HTTP or transport failure: those come back as a StreamResult too."""
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    start = time.perf_counter()
    first_token_ms: float | None = None
    first_token_field: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    continued: int | None = None
    try:
        req = Request(url, data=body, headers=headers, method="POST")
        with urlopen(req, timeout=timeout_s) as response:
            status = response.status
            # Read before the body: the headers arrive with the first SSE byte and this
            # is the only place the serving pod is ever named.
            response_headers = lower_headers(response.headers)
            target_pod = pod_from_headers(response_headers)
            continued, retried = reissue_from_headers(response_headers)
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if line.startswith(":"):
                    # SSE comment; the reissue sidecar closes a stitched stream with
                    # ": x-tre-continued: <segments>".
                    name, _, value = line[1:].strip().partition(":")
                    if name.strip().lower() == CONTINUED_HEADER:
                        continued = _positive_int(value) or continued
                    continue
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if first_token_ms is None:
                    field = chunk_token_field(chunk)
                    if field is not None:
                        first_token_ms = (time.perf_counter() - start) * 1000.0
                        first_token_field = field
                for choice in chunk.get("choices") or []:
                    if isinstance(choice, dict) and choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                continued = _positive_int(chunk.get(CONTINUED_FIELD)) or continued
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
        done_ms = (time.perf_counter() - start) * 1000.0
        return StreamResult(
            status, first_token_ms, done_ms, prompt_tokens, completion_tokens, target_pod=target_pod,
            finish_reason=finish_reason, tre_continued=continued, tre_retried=retried,
            first_token_field=first_token_field,
        )
    except HTTPError as exc:
        error_headers = lower_headers(exc.headers)
        return StreamResult(
            exc.code,
            None,
            (time.perf_counter() - start) * 1000.0,
            error=f"HTTP {exc.code}",
            error_body=read_error_body(exc),
            error_headers=error_headers,
            target_pod=pod_from_headers(error_headers),
        )
    except (URLError, TimeoutError, OSError) as exc:  # noqa: BLE001
        return StreamResult(
            0,
            None,
            (time.perf_counter() - start) * 1000.0,
            error=type(exc).__name__,
            timed_out=is_client_timeout(exc),
        )


def is_client_timeout(exc: BaseException) -> bool:
    """True when this transport failure is the client giving up, not the peer refusing.

    urllib surfaces a read timeout three ways depending on where it fires: as
    :class:`TimeoutError` (``socket.timeout`` is an alias of it since 3.10), as a
    :class:`urllib.error.URLError` wrapping one in ``reason``, or - on some stacks - as
    an ``OSError`` whose text says so and nothing else does. Only the first two are
    structural, so the text check comes last and is deliberately narrow.
    """
    if isinstance(exc, TimeoutError):
        return True
    reason = getattr(exc, "reason", None)
    if isinstance(reason, TimeoutError):
        return True
    return "timed out" in str(exc).lower()



#: How much of a non-2xx body to keep. An Envoy circuit-breaker body is ~80 bytes and a
#: vLLM JSON error is small too; the cap only bounds a pathological upstream.
MAX_ERROR_BODY_CHARS = 2048


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
    if not raw:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
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


#: Where a streamed chunk carries generated text, in the order they are checked:
#: completions ``choices[].text``; chat ``choices[].delta.<field>`` for these fields.
CHAT_TOKEN_FIELDS = ("content", "reasoning_content", "reasoning")


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
            for field in CHAT_TOKEN_FIELDS:
                if delta.get(field):
                    return field
    return None


def _chunk_has_content(chunk: dict[str, Any]) -> bool:
    return chunk_token_field(chunk) is not None


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
    }

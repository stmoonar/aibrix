"""Open-loop streaming HTTP sender for the replayer (audit blocker B3).

Fires each ScheduledRequest at its scheduled time (via dispatch_open_loop) against an
OpenAI-compatible gateway, streaming the response to capture TTFT and end time, and records
one per-request JSONL line. That per-request record is also the S4 raw-logger format, so this
is the single sender both R2 (trace runs) and R3 (raw capacity logging) use.

The actual network call is an injectable seam (`stream_call`) so tests run with a fake and
never touch the network. The default seam uses urllib + SSE parsing.

Each record also carries ``target_pod``: the pod that served the request, when the
serving path names one. See :class:`StreamingHttpSender` - it does not on the default
path, and the field is then None rather than guessed.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from tre_replayer.engine.prompts import DEFAULT_MODE, build_prompt
from tre_replayer.engine.schedule import ScheduledRequest

#: Response headers, in preference order, that name the pod that served a request.
#: ``target-pod`` / ``target-pod-ip`` are what the AIBrix gateway plugin sets on the
#: routed path (``pkg/plugins/gateway/gateway_rsp_headers.go``); the ``x-`` names are
#: there so a header added at the Envoy layer - which is what the per-model HTTPRoute
#: path would need - is picked up without another client change. The pod *name* is
#: preferred over its address because it survives a pod IP being reused.
POD_HEADER_KEYS = ("target-pod", "x-target-pod", "x-upstream-pod", "target-pod-ip")

#: Request header that makes the AIBrix gateway route (and therefore report the pod it
#: routed to). See :class:`StreamingHttpSender` for why it is off by default.
ROUTING_STRATEGY_HEADER = "routing-strategy"


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


# seam: (url, headers, body_bytes, timeout_s) -> StreamResult
StreamCall = Callable[[str, dict[str, str], bytes, float], StreamResult]


def _now_ms() -> int:
    return int(time.time() * 1000)


class StreamingHttpSender:
    """Fires scheduled requests and records one row per request.

    ``routing_strategy`` selects *which serving path* the request takes, and with it
    whether per-pod attribution is possible at all:

    * ``None`` (the default, and what the campaign uses): the ``model`` request header is
      sent, so the per-model HTTPRoute matches and Envoy load-balances straight across
      the model Service's endpoints. The AIBrix gateway plugin is not in this path, so no
      answer carries a pod header and ``target_pod`` is None on every row.
    * a strategy name (e.g. ``"least-request"``): the ``model`` header is *omitted* so the
      catch-all ``aibrix-reserved-router`` matches instead, the ext_proc plugin routes the
      request itself, and the answer carries ``target-pod`` / ``target-pod-ip``.

    The second option changes who chooses the pod, which changes the measurement. It is
    therefore opt-in and never the default: turning it on to get attribution and then
    comparing the numbers against a run that did not is a mistake this docstring exists
    to prevent.
    """

    def __init__(
        self,
        gateway_url: str,
        *,
        stream_call: StreamCall | None = None,
        input_tokens_default: int = 64,
        output_tokens_default: int = 128,
        max_in_flight: int = 512,
        prompt_mode: str = DEFAULT_MODE,
        routing_strategy: str | None = None,
        now_ms: Callable[[], int] = _now_ms,
        mono: Callable[[], float] = time.monotonic,
    ) -> None:
        self._url = gateway_url
        self._call = stream_call or _default_stream_call
        self._in = input_tokens_default
        self._out = output_tokens_default
        self._prompt_mode = prompt_mode
        self._routing_strategy = routing_strategy
        self._now = now_ms
        self._mono = mono
        # F5: each streamed request blocks a worker for its whole e2e. asyncio.to_thread's
        # default executor is capped at ~32, which would silently turn the open-loop replay
        # into a closed-loop-32 and under-drive the system under saturation. Use a dedicated
        # pool sized for the peak in-flight, and record pool_wait_ms so starvation is visible.
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_in_flight), thread_name_prefix="trepl-send")
        self.records: list[dict[str, Any]] = []

    async def __call__(self, request: ScheduledRequest, scheduled_ts: float, actual_ts: float) -> None:
        import asyncio

        loop = asyncio.get_event_loop()
        record = await loop.run_in_executor(self._executor, self._send_one, request, scheduled_ts, actual_ts)
        self.records.append(record)

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def max_pool_wait_ms(self) -> float:
        return max((r.get("pool_wait_ms", 0.0) for r in self.records), default=0.0)

    def _send_one(self, request: ScheduledRequest, scheduled_ts: float, actual_ts: float) -> dict[str, Any]:
        # time from the dispatcher scheduling this send to a worker actually picking it up;
        # a large p99 here means the pool starved and the replay under-drove the target (F5).
        pool_wait_ms = max(0.0, (self._mono() - actual_ts) * 1000.0)
        out_tokens = request.max_output_tokens or self._out
        in_tokens = request.prompt_tokens or self._in
        # A trace may carry its own prompt text; otherwise synthesise one that is unique
        # to this request. A constant prompt would be served from the prefix cache on any
        # engine that has it enabled, making prefill free and the measurement worthless
        # (see tre_replayer.engine.prompts). Seed key = model + request_id, both
        # deterministic per trace, so a replay sends byte-identical prompts.
        prompt = request.prompt or build_prompt(
            in_tokens,
            f"{request.model}|{request.request_id}",
            mode=self._prompt_mode,
            model=request.model,
        )
        body = json.dumps(
            {
                "model": request.model,
                "prompt": prompt,
                "max_tokens": out_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        ).encode("utf-8")
        headers = build_request_headers(request.model, self._routing_strategy)
        send_ts_ms = self._now()
        res = self._call(self._url, headers, body, max(30.0, out_tokens / 4.0))
        return {
            "request_id": request.request_id,
            "model": request.model,
            "scheduled_offset_ms": int(scheduled_ts * 1000),  # dispatcher monotonic clock, NOT epoch
            "actual_send_ts_ms": send_ts_ms,
            "schedule_delay_ms": max(0.0, (actual_ts - scheduled_ts) * 1000.0),
            "pool_wait_ms": round(pool_wait_ms, 3),
            "ttft_ms": res.first_token_ms,
            "e2e_ms": res.done_ms,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "http_status": res.status,
            "error": res.error,
            "error_body": res.error_body,
            "error_headers": res.error_headers,
            "target_pod": res.target_pod,
        }

    def write_jsonl(self, path: str) -> int:
        with open(path, "w", encoding="utf-8") as fh:
            for record in self.records:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        return len(self.records)


def build_request_headers(model: str, routing_strategy: str | None = None) -> dict[str, str]:
    """Request headers for one completion, and with them the serving path.

    Without a routing strategy the ``model`` header is sent and the per-model HTTPRoute
    matches, which is the path the campaign measures. With one, the ``model`` header is
    deliberately left out: the per-model route matches on exactly that header, so sending
    it would win over the catch-all reserved route and the ext_proc plugin - the only
    thing that reports a pod - would never see the request. The model still travels in
    the JSON body, which is where the plugin reads it from.
    """
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if routing_strategy:
        headers[ROUTING_STRATEGY_HEADER] = routing_strategy
    else:
        headers["model"] = model
    return headers


def _default_stream_call(url: str, headers: dict[str, str], body: bytes, timeout_s: float) -> StreamResult:
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    start = time.perf_counter()
    first_token_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    try:
        req = Request(url, data=body, headers=headers, method="POST")
        with urlopen(req, timeout=timeout_s) as response:
            status = response.status
            # Read before the body: the headers arrive with the first SSE byte and this
            # is the only place the serving pod is ever named.
            target_pod = pod_from_headers(lower_headers(response.headers))
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if first_token_ms is None and _chunk_has_content(chunk):
                    first_token_ms = (time.perf_counter() - start) * 1000.0
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
        done_ms = (time.perf_counter() - start) * 1000.0
        return StreamResult(
            status, first_token_ms, done_ms, prompt_tokens, completion_tokens, target_pod=target_pod
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
        return StreamResult(0, None, (time.perf_counter() - start) * 1000.0, error=type(exc).__name__)



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


def _chunk_has_content(chunk: dict[str, Any]) -> bool:
    for choice in chunk.get("choices", []) or []:
        if choice.get("text"):
            return True
        delta = choice.get("delta")
        if isinstance(delta, dict) and delta.get("content"):
            return True
    return False

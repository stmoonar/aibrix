"""Open-loop streaming HTTP sender for the replayer (audit blocker B3).

Fires each ScheduledRequest at its scheduled time (via dispatch_open_loop) against an
OpenAI-compatible gateway, streaming the response to capture TTFT and end time, and records
one per-request JSONL line. That per-request record is also the S4 raw-logger format, so this
is the single sender both R2 (trace runs) and R3 (raw capacity logging) use.

The actual network call is an injectable seam (`stream_call`) so tests run with a fake and
never touch the network. The default seam uses urllib + SSE parsing.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from tre_replayer.engine.schedule import ScheduledRequest


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


# seam: (url, headers, body_bytes, timeout_s) -> StreamResult
StreamCall = Callable[[str, dict[str, str], bytes, float], StreamResult]


def _now_ms() -> int:
    return int(time.time() * 1000)


class StreamingHttpSender:
    def __init__(
        self,
        gateway_url: str,
        *,
        stream_call: StreamCall | None = None,
        input_tokens_default: int = 64,
        output_tokens_default: int = 128,
        now_ms: Callable[[], int] = _now_ms,
    ) -> None:
        self._url = gateway_url
        self._call = stream_call or _default_stream_call
        self._in = input_tokens_default
        self._out = output_tokens_default
        self._now = now_ms
        self.records: list[dict[str, Any]] = []

    async def __call__(self, request: ScheduledRequest, scheduled_ts: float, actual_ts: float) -> None:
        # dispatch_open_loop awaits us; keep the (blocking) network call off the event loop.
        import asyncio

        record = await asyncio.to_thread(self._send_one, request, scheduled_ts, actual_ts)
        self.records.append(record)

    def _send_one(self, request: ScheduledRequest, scheduled_ts: float, actual_ts: float) -> dict[str, Any]:
        out_tokens = request.max_output_tokens or self._out
        in_tokens = request.prompt_tokens or self._in
        prompt = request.prompt or " ".join(["token"] * max(1, in_tokens))
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
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream", "model": request.model}
        send_ts_ms = self._now()
        res = self._call(self._url, headers, body, max(30.0, out_tokens / 4.0))
        return {
            "request_id": request.request_id,
            "model": request.model,
            "scheduled_ts_ms": int(scheduled_ts * 1000),
            "actual_send_ts_ms": send_ts_ms,
            "schedule_delay_ms": max(0.0, (actual_ts - scheduled_ts) * 1000.0),
            "ttft_ms": res.first_token_ms,
            "e2e_ms": res.done_ms,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "http_status": res.status,
            "error": res.error,
        }

    def write_jsonl(self, path: str) -> int:
        with open(path, "w", encoding="utf-8") as fh:
            for record in self.records:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        return len(self.records)


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
        return StreamResult(status, first_token_ms, done_ms, prompt_tokens, completion_tokens)
    except HTTPError as exc:
        return StreamResult(exc.code, None, (time.perf_counter() - start) * 1000.0, error=f"HTTP {exc.code}")
    except (URLError, TimeoutError, OSError) as exc:  # noqa: BLE001
        return StreamResult(0, None, (time.perf_counter() - start) * 1000.0, error=type(exc).__name__)


def _chunk_has_content(chunk: dict[str, Any]) -> bool:
    for choice in chunk.get("choices", []) or []:
        if choice.get("text"):
            return True
        delta = choice.get("delta")
        if isinstance(delta, dict) and delta.get("content"):
            return True
    return False

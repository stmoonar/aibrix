"""Tiny OpenAI-compatible fake server for tre_loadgen_v1 tests (localhost only).

Raw asyncio HTTP/1.1 (keep-alive, chunked SSE) so every failure mode is under
exact control: mid-stream connection drop, read stall, 5xx-before-success, etc.
Runs as a separate process (``python loadgen_v1_fake_server.py``): it prints
``PORT <n>`` on stdout, then serves until killed.  Everything it received is
available at ``GET /_records`` as JSON.

Behaviour is selected by the request's ``model`` field:
  ok                normal stream: role chunk, ``max_tokens`` (<=8) content
                    chunks, finish_reason=length, usage chunk, [DONE]
  ok-stop           like ok but finish_reason=stop
  fail503x<N>       first N attempts (per prompt) -> 503, then ok
  always500         always 500
  drop              role chunk + 2 content chunks, then TCP close mid-body
  stall             role chunk + 1 content chunk, then silence (read timeout)
  hang              never sends response headers
  sseerror          one content chunk then an SSE ``{"error": ...}`` event
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time

RECORDS: list[dict] = []
ATTEMPTS: dict[tuple, int] = {}
TOKEN_DELAY_S = 0.02


def _chunk(obj) -> bytes:
    data = obj if isinstance(obj, (bytes, str)) else json.dumps(obj)
    if isinstance(data, str):
        data = data.encode()
    payload = b"data: " + data + b"\n\n"
    return f"{len(payload):x}\r\n".encode() + payload + b"\r\n"


def _delta(model: str, content, finish=None, role=None) -> dict:
    delta = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    return {
        "id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 0, "model": model,
        "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}],
    }


async def _write_json(writer, status: int, reason: str, obj: dict) -> None:
    body = json.dumps(obj).encode()
    writer.write(
        f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n\r\n".encode() + body
    )
    await writer.drain()


async def _stream_ok(writer, model: str, n_tokens: int, finish: str, prompt_tokens: int) -> None:
    writer.write(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n"
        b"target-pod: 10.0.0.9:8000\r\n\r\n"
    )
    writer.write(_chunk(_delta(model, "", role="assistant")))
    await writer.drain()
    for i in range(n_tokens):
        await asyncio.sleep(TOKEN_DELAY_S)
        last = i == n_tokens - 1
        writer.write(_chunk(_delta(model, f"t{i} ", finish=finish if last else None)))
        await writer.drain()
    writer.write(_chunk({
        "id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 0, "model": model,
        "choices": [],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n_tokens,
                  "total_tokens": prompt_tokens + n_tokens},
    }))
    writer.write(_chunk("[DONE]"))
    writer.write(b"0\r\n\r\n")
    await writer.drain()


async def _handle_one(reader, writer) -> bool:
    """Serve one request; return False to close the connection."""
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    method, path, _ = lines[0].split(" ", 2)
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    length = int(headers.get("content-length", "0") or 0)
    raw = await reader.readexactly(length) if length else b""

    if method == "GET" and path == "/_records":
        await _write_json(writer, 200, "OK", {"records": RECORDS})
        return True

    body = json.loads(raw.decode()) if raw else {}
    model = body.get("model", "")
    prompt = ""
    if body.get("messages"):
        prompt = body["messages"][0].get("content", "")
    key = (model, prompt)
    ATTEMPTS[key] = ATTEMPTS.get(key, 0) + 1
    attempt = ATTEMPTS[key]
    RECORDS.append({"t": time.time(), "method": method, "path": path, "headers": headers,
                    "body": body, "attempt": attempt})

    max_tokens = body.get("max_tokens")
    n_tokens = max(1, min(int(max_tokens), 8)) if isinstance(max_tokens, int) else 3
    prompt_tokens = len(prompt.split()) or 1

    m = re.fullmatch(r"fail503x(\d+)", model)
    if m and attempt <= int(m.group(1)):
        await _write_json(writer, 503, "Service Unavailable",
                          {"error": {"message": "overloaded", "type": "server_error", "code": 503}})
        return True
    if model == "always500":
        await _write_json(writer, 500, "Internal Server Error",
                          {"error": {"message": "boom", "type": "server_error", "code": 500}})
        return True
    if model == "hang":
        await asyncio.sleep(3600)
        return False
    if not body.get("stream"):
        await _write_json(writer, 200, "OK", {
            "id": "chatcmpl-fake", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "x " * n_tokens},
                         "logprobs": None, "finish_reason": "length"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n_tokens,
                      "total_tokens": prompt_tokens + n_tokens},
        })
        return True
    if model in ("drop", "stall", "sseerror"):
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
        )
        writer.write(_chunk(_delta(model, "", role="assistant")))
        writer.write(_chunk(_delta(model, "t0 ")))
        await writer.drain()
        if model == "drop":
            writer.write(_chunk(_delta(model, "t1 ")))
            await writer.drain()
            await asyncio.sleep(0.05)
            writer.transport.abort()  # RST mid-body: no terminating chunk
            return False
        if model == "stall":
            await asyncio.sleep(3600)
            return False
        writer.write(_chunk({"error": {"message": "engine died", "type": "server_error"}}))
        writer.write(b"0\r\n\r\n")
        await writer.drain()
        return True
    await _stream_ok(writer, model, n_tokens, "stop" if model == "ok-stop" else "length", prompt_tokens)
    return True


async def _conn(reader, writer) -> None:
    try:
        while True:
            if not await _handle_one(reader, writer):
                break
    except (asyncio.IncompleteReadError, ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _main() -> None:
    server = await asyncio.start_server(_conn, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    print(f"PORT {port}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        sys.exit(0)

#!/usr/bin/env python3
"""A local OpenAI-compatible SSE server for client verification and benchmarks.

Never a model: every answer is synthetic and deterministic. Two uses:

* **scenarios** (``--mode scenario``, the equivalence check of the unified client): the
  request's ``model`` field picks the answer - ``ok`` (role chunk, ``max_tokens`` content
  chunks capped at ``--max-stream-tokens``, finish ``length``, usage, ``[DONE]``),
  ``ok-stop``, ``reasoning`` (tokens in ``delta.reasoning_content``), ``notext`` (role +
  usage only), ``retried`` (``x-tre-retried: 1``), ``continued`` (``tre_continued`` on the
  final chunk and a ``: x-tre-continued: 2`` comment), ``shed`` (Envoy-style 503 text
  body, ``reset reason: overflow``), ``e500`` (vLLM-style JSON 500), ``fail503x<N>``
  (first N attempts per prompt answer a JSON 503), ``sseerror`` (one token, an
  ``{"error": ...}`` chunk, ``[DONE]``), ``cut`` (one token, then the connection is
  reset mid-body), ``hang`` (never answers). Completions requests (``prompt``) get
  ``choices[].text`` chunks instead of chat deltas.
* **bench** (``--mode bench``): a vLLM-shaped engine - at most ``--max-running``
  requests generate at once (the rest wait in FIFO order with their headers already
  sent, as vLLM's streaming responses do), each emits one chunk per ``--decode-ms`` step
  after a ``--prefill-ms-per-1k`` prefill.

Every request is logged: arrival time, first-token write time, model, request id (the
prompt's first token), and - in scenario mode - the raw body and headers
(``GET /_records``). ``--log PATH`` appends one JSON line per request (bench).
``--workers K`` forks K processes on one port (``SO_REUSEPORT``), each with
``max_running / K`` slots. Prints ``PORT <n>`` when listening.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import signal
import socket
import sys
import time

RECORDS: list[dict] = []
ATTEMPTS: dict[tuple, int] = {}
LOG = None
ARGS = None
SLOTS: asyncio.Semaphore | None = None


def _frame(payload: bytes) -> bytes:
    return f"{len(payload):x}\r\n".encode() + payload + b"\r\n"


def _data(obj) -> bytes:
    body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
    return _frame(b"data: " + body + b"\n\n")


def _chunk(chat: bool, model: str, text=None, finish=None, role=None, field="content", extra=None) -> dict:
    if chat:
        delta = {}
        if role:
            delta["role"] = role
        if text is not None:
            delta[field] = text
        choice = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
        obj = "chat.completion.chunk"
    else:
        choice = {"index": 0, "text": text or "", "logprobs": None, "finish_reason": finish}
        obj = "text_completion"
    out = {"id": "cmpl-fake", "object": obj, "created": 0, "model": model, "choices": [choice]}
    if extra:
        out.update(extra)
    return out


def _usage(model: str, prompt_tokens: int, n: int) -> dict:
    return {"id": "cmpl-fake", "object": "chunk", "created": 0, "model": model, "choices": [],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n, "total_tokens": prompt_tokens + n}}


async def _json(writer, status: int, reason: str, obj, content_type="application/json") -> None:
    body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
    writer.write(f"HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\n"
                 f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    await writer.drain()


def _head(extra: str = "") -> bytes:
    return (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n"
            + extra.encode() + b"\r\n")


async def _scenario(writer, body: dict, key: tuple, entry: dict) -> bool:
    model = str(body.get("model", ""))
    chat = "messages" in body
    max_tokens = body.get("max_tokens")
    n = max(1, min(int(max_tokens), ARGS.max_stream_tokens)) if isinstance(max_tokens, int) else 3
    prompt_tokens = max(1, len(str(entry.get("prompt", "")).split()))
    gap = ARGS.token_gap_ms / 1000.0
    m = re.fullmatch(r"fail503x(\d+)", model)
    if m and ATTEMPTS[key] <= int(m.group(1)):
        await _json(writer, 503, "Service Unavailable", {"error": {"message": "overloaded", "type": "server_error",
                                                                   "code": 503}})
        return True
    if model == "shed":
        await _json(writer, 503, "Service Unavailable",
                    b"upstream connect error or disconnect/reset before headers. reset reason: overflow",
                    content_type="text/plain")
        return True
    if model == "e500":
        await _json(writer, 500, "Internal Server Error",
                    {"object": "error", "message": "boom", "type": "InternalServerError", "code": 500})
        return True
    if model == "hang":
        await asyncio.sleep(3600)
        return False
    extra_head = "target-pod: pod-fake-0\r\n"
    if model == "retried":
        extra_head += "x-tre-retried: 1\r\n"
    writer.write(_head(extra_head))
    if chat:
        writer.write(_data(_chunk(chat, model, "", role="assistant")))
    await writer.drain()
    if model == "notext":
        writer.write(_data(_usage(model, prompt_tokens, 0)) + _data(b"[DONE]") + b"0\r\n\r\n")
        await writer.drain()
        return True
    field = "reasoning_content" if model == "reasoning" else "content"
    for i in range(n):
        await asyncio.sleep(gap)
        if i == 0:
            entry["t_first"] = time.time()
        if model == "sseerror" and i == 1:
            writer.write(_data({"error": {"message": "engine died", "type": "server_error", "code": 500}}))
            writer.write(_data(b"[DONE]") + b"0\r\n\r\n")
            await writer.drain()
            return True
        if model == "cut" and i == 1:
            writer.transport.abort()
            return False
        last = i == n - 1
        finish = ("stop" if model == "ok-stop" else "length") if last else None
        extra = {"tre_continued": 2} if (last and model == "continued") else None
        writer.write(_data(_chunk(chat, model, f"t{i} ", finish=finish, field=field, extra=extra)))
        await writer.drain()
    writer.write(_data(_usage(model, prompt_tokens, n)))
    if model == "continued":
        writer.write(_frame(b": x-tre-continued: 2\n\n"))
    writer.write(_data(b"[DONE]") + b"0\r\n\r\n")
    await writer.drain()
    return True


async def _bench(writer, body: dict, entry: dict) -> bool:
    """vLLM-shaped: headers at once, then wait for a slot, prefill, one chunk per step."""
    model = str(body.get("model", ""))
    chat = "messages" in body
    n = int(body.get("max_tokens") or 16)
    prompt_len = len(str(entry.get("prompt", "")))
    writer.write(_head("target-pod: pod-bench\r\n"))
    await writer.drain()
    async with SLOTS:
        entry["t_admit"] = time.time()
        await asyncio.sleep(ARGS.prefill_ms_per_1k * (prompt_len / 4.0) / 1000.0 / 1000.0)
        entry["t_first"] = time.time()
        first = b""
        if chat:
            first = _data(_chunk(chat, model, "", role="assistant"))
        writer.write(first + _data(_chunk(chat, model, "t ")))
        await writer.drain()
        step = ARGS.decode_ms / 1000.0
        next_at = time.monotonic()
        for i in range(1, n):
            next_at += step
            delay = next_at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            finish = "length" if i == n - 1 else None
            writer.write(_data(_chunk(chat, model, "t ", finish=finish)))
            await writer.drain()
    writer.write(_data(_usage(model, max(1, prompt_len // 4), n)) + _data(b"[DONE]") + b"0\r\n\r\n")
    await writer.drain()
    entry["t_done"] = time.time()
    return True


async def _handle(reader, writer) -> bool:
    head = await reader.readuntil(b"\r\n\r\n")
    t_recv = time.time()
    lines = head.decode("latin-1").split("\r\n")
    method, path, _ = lines[0].split(" ", 2)
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    length = int(headers.get("content-length", "0") or 0)
    raw = await reader.readexactly(length) if length else b""
    if method == "GET" and path.startswith("/_records"):
        await _json(writer, 200, "OK", {"records": RECORDS})
        return True
    body = json.loads(raw.decode()) if raw else {}
    prompt = body["messages"][0].get("content", "") if body.get("messages") else body.get("prompt", "")
    if not isinstance(prompt, str):
        prompt = json.dumps(prompt)
    key = (body.get("model"), prompt)
    ATTEMPTS[key] = ATTEMPTS.get(key, 0) + 1
    entry = {"t": t_recv, "path": path, "model": body.get("model"), "attempt": ATTEMPTS[key],
             "rid": prompt.split(" ", 1)[0][:64], "prompt": prompt}
    if ARGS.mode == "scenario":
        entry.update({"method": method, "headers": headers, "body": body,
                      "raw_b64": base64.b64encode(raw).decode()})
        RECORDS.append(entry)
        keep = await _scenario(writer, body, key, entry)
    else:
        keep = await _bench(writer, body, entry)
        if LOG is not None:
            entry.pop("prompt", None)
            LOG.write(json.dumps(entry) + "\n")
    return keep


async def _conn(reader, writer) -> None:
    try:
        while await _handle(reader, writer):
            pass
    except (asyncio.IncompleteReadError, ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass


async def _serve(sock) -> None:
    global SLOTS
    SLOTS = asyncio.Semaphore(max(1, ARGS.max_running // max(1, ARGS.workers)))
    server = await asyncio.start_server(_conn, sock=sock, backlog=4096, limit=1 << 22)
    async with server:
        await server.serve_forever()


def main(argv=None) -> int:
    global ARGS, LOG
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["scenario", "bench"], default="scenario")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-stream-tokens", type=int, default=8)
    ap.add_argument("--token-gap-ms", type=float, default=20.0)
    ap.add_argument("--max-running", type=int, default=256)
    ap.add_argument("--decode-ms", type=float, default=30.0)
    ap.add_argument("--prefill-ms-per-1k", type=float, default=40.0)
    ap.add_argument("--log", default=None)
    ARGS = ap.parse_args(argv)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((ARGS.host, ARGS.port))
    sock.listen(4096)
    port = sock.getsockname()[1]
    children = []
    for index in range(1, max(1, ARGS.workers)):
        pid = os.fork()
        if pid == 0:
            children = []
            break
        children.append(pid)
    else:
        index = 0
    if index == 0:
        print(f"PORT {port}", flush=True)
    if ARGS.log:
        LOG = open(f"{ARGS.log}.{index}", "a", buffering=1 << 16)

    def stop(*_):
        if LOG is not None:
            LOG.flush()
        for pid in children:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        os._exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    asyncio.run(_serve(sock))
    return 0


if __name__ == "__main__":
    sys.exit(main())

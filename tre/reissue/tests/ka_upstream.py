"""A real uvicorn server shaped like vLLM's OpenAI server, for the keep-alive race tests.

vLLM serves through uvicorn with ``timeout_keep_alive = VLLM_HTTP_TIMEOUT_KEEP_ALIVE``
(default 5 s): an idle keep-alive connection is closed by the server. Run as
``python3 ka_upstream.py <port> <keep_alive_s> <min_delay_s> <max_delay_s>``; it answers
``POST /v1/completions`` (stream or not) after a random "generation" delay and
``GET /health``. uvloop / httptools are used when installed (as in the vLLM image),
otherwise asyncio / h11."""
from __future__ import annotations

import asyncio
import json
import random
import sys


def make_app(min_delay: float, max_delay: float):
    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        if scope["path"] == "/health":
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"0")]})
            await send({"type": "http.response.body", "body": b""})
            return
        try:
            stream = bool(json.loads(body or b"{}").get("stream"))
        except ValueError:
            stream = False
        await asyncio.sleep(random.uniform(min_delay, max_delay))
        chunk = {"id": "cmpl-1", "object": "text_completion", "model": "m",
                 "choices": [{"index": 0, "text": "ok", "finish_reason": "length"}]}
        if stream:
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/event-stream; charset=utf-8")]})
            await send({"type": "http.response.body", "more_body": True,
                        "body": b"data: " + json.dumps(chunk).encode() + b"\n\n"})
            await send({"type": "http.response.body", "body": b"data: [DONE]\n\n"})
        else:
            payload = json.dumps(chunk).encode()
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(payload)).encode())]})
            await send({"type": "http.response.body", "body": payload})

    return app


def main() -> None:
    import uvicorn

    port, keep_alive = int(sys.argv[1]), float(sys.argv[2])
    min_delay, max_delay = float(sys.argv[3]), float(sys.argv[4])
    kwargs = {}
    try:
        import uvloop  # noqa: F401

        kwargs["loop"] = "uvloop"
    except ImportError:
        kwargs["loop"] = "asyncio"
    try:
        import httptools  # noqa: F401

        kwargs["http"] = "httptools"
    except ImportError:
        kwargs["http"] = "h11"
    uvicorn.run(make_app(min_delay, max_delay), host="127.0.0.1", port=port, log_level="error",
                timeout_keep_alive=keep_alive, **kwargs)


if __name__ == "__main__":
    main()

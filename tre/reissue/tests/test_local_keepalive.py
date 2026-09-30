"""Loopback hop sidecar -> local vLLM (2026-09-30 smoke: 17 x 502 with no overload).

Root cause: the sidecar pooled its connections to vLLM with aiohttp's default 15 s
keep-alive, vLLM's uvicorn closes an idle connection after 5 s
(VLLM_HTTP_TIMEOUT_KEEP_ALIVE), so a pooled connection could be reused just as vLLM
closed it: ``Server disconnected`` / ``Connection reset by peer`` / "Can not write
request body", all before the first response byte. Three layers: pooled keep-alive below
vLLM's, one fresh-connection re-send, and 503 + Retry-After (never 502) if that fails.

The race tests run a real uvicorn server (``ka_upstream.py``, keep-alive 1 s) in a
subprocess and send bursts whose idle gaps straddle that keep-alive."""
from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import random
import socket
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import aiohttp
import pytest
from aiohttp.test_utils import TestServer

from test_reissue_sidecar import Harness, completion_body
from tre_reissue import sidecar as sc
from tre_reissue.sidecar import Config, ReissueSidecar

HERE = Path(__file__).resolve().parent
SERVER_KEEP_ALIVE_S = 1


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.asynccontextmanager
async def uvicorn_upstream(keep_alive: float = SERVER_KEEP_ALIVE_S):
    pytest.importorskip("uvicorn")
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(HERE / "ka_upstream.py"), str(port), str(keep_alive), "0.05", "0.5"])
    try:
        for _ in range(300):
            try:
                socket.create_connection(("127.0.0.1", port), 0.1).close()
                break
            except OSError:
                if proc.poll() is not None:
                    pytest.fail("ka_upstream.py exited")
                await asyncio.sleep(0.05)
        else:
            pytest.fail("ka_upstream.py did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@contextlib.asynccontextmanager
async def serve_sidecar(upstream_url: str, **overrides):
    cfg = replace(Config(upstream_url=upstream_url, model="m", pod_name="pod-a", gateway_url="http://127.0.0.1:9",
                         probe_interval_s=3600.0, retry_attempts=1, retry_backoff_s=0.01,
                         retry_max_backoff_s=0.01), **overrides)
    sidecar = ReissueSidecar(cfg)

    async def no_monitor() -> None:  # no /is_sleeping probes on the scripted connections
        return None

    sidecar._monitor = no_monitor
    server = TestServer(sidecar.build_app())
    await server.start_server()
    try:
        yield sidecar, str(server.make_url("")).rstrip("/")
    finally:
        await server.close()


async def run_bursts(url: str, sidecar: ReissueSidecar, *, min_bursts: int, max_bursts: int, until) -> dict:
    """Bursts of 100 concurrent requests (half streaming). Generations take 0.05-0.5 s, so
    the pooled connections are released over ~0.45 s and uvicorn closes each one 1 s
    later; the next burst starts 0.55-0.95 s after the last response, i.e. while those
    closes are happening. Stops after ``min_bursts`` once ``until(sidecar, statuses)``."""
    statuses: dict[int, int] = {}
    bodies: list[dict] = []
    async with aiohttp.ClientSession() as http:
        async def one(index: int) -> None:
            body = {"model": "m", "prompt": "x", "max_tokens": 1, "stream": index % 2 == 0}
            async with http.post(url + "/v1/completions", json=body) as resp:
                raw = await resp.read()
                statuses[resp.status] = statuses.get(resp.status, 0) + 1
                if resp.status != 200:
                    bodies.append({"status": resp.status, "retry_after": resp.headers.get("Retry-After"),
                                   "body": json.loads(raw)})

        for burst in range(max_bursts):
            await asyncio.gather(*(one(i) for i in range(100)))
            if burst + 1 >= min_bursts and until(sidecar, statuses):
                break
            await asyncio.sleep(random.uniform(0.55, 0.95))
    return {"statuses": statuses, "errors": bodies, "bursts": burst + 1}


# ------------------------------------------------------------- the race, end to end


@pytest.mark.asyncio
async def test_keepalive_race_is_absorbed_by_the_fresh_connection_resend():
    """Defaults (pool keep-alive 2 s > this server's 1 s, so layer 1 does not help here):
    every request succeeds and the race did happen (re-sends > 0). On main (before the
    fix) the same bursts give 502s."""
    async with uvicorn_upstream() as upstream, serve_sidecar(upstream) as (sidecar, url):
        out = await run_bursts(url, sidecar, min_bursts=6, max_bursts=30,
                               until=lambda s, _: getattr(s.metrics, "reconnect", {}).get("ok", 0) > 0)
        print("fixed:", out["statuses"], "bursts", out["bursts"], "reconnect",
              getattr(sidecar.metrics, "reconnect", None))
        assert out["statuses"] == {200: 100 * out["bursts"]}, out
        assert sidecar.metrics.reconnect["ok"] > 0, "the race never happened: harness too gentle"
        assert sidecar.metrics.reconnect["fail"] == 0
        assert sidecar.metrics.total("failed") == 0


@pytest.mark.asyncio
async def test_without_the_fix_the_race_fails_requests_with_503_not_502():
    """Control: main's pool settings (15 s keep-alive, no re-send) reproduce the race;
    the failures are now 503 + Retry-After with error.layer = sidecar_upstream."""
    async with uvicorn_upstream() as upstream, serve_sidecar(
        upstream, upstream_keepalive_s=15.0, local_reconnect_attempts=0
    ) as (sidecar, url):
        out = await run_bursts(url, sidecar, min_bursts=1, max_bursts=30,
                               until=lambda _, statuses: sum(statuses.values()) > statuses.get(200, 0))
        print("unfixed:", out["statuses"], "bursts", out["bursts"])
        failed = sum(v for k, v in out["statuses"].items() if k != 200)
        assert failed > 0, "the race never happened: harness too gentle"
        assert set(out["statuses"]) <= {200, 503}, out
        for error in out["errors"]:
            assert error["retry_after"] == "1"
            assert error["body"]["error"]["layer"] == "sidecar_upstream"
            assert error["body"]["error"]["type"] == "ServiceUnavailable"
        assert sidecar.metrics.reissue[("failed", "upstream_unavailable")] == failed


@pytest.mark.asyncio
async def test_pool_keepalive_below_the_server_keepalive_alone_avoids_the_race():
    """Layer 1 alone (no re-send): the pool drops idle connections before uvicorn does.
    The unfixed control fails ~0.3-0.5 requests per burst, so 8 clean bursts is a real test."""
    async with uvicorn_upstream() as upstream, serve_sidecar(
        upstream, upstream_keepalive_s=0.5, local_reconnect_attempts=0
    ) as (sidecar, url):
        out = await run_bursts(url, sidecar, min_bursts=8, max_bursts=8, until=lambda *_: True)
        assert out["statuses"] == {200: 800}, out
        assert sidecar.metrics.reconnect == {"ok": 0, "fail": 0}


# ---------------------------------------------------------- scripted upstream units


class ScriptedUpstream:
    """Raw HTTP/1.1 upstream; connection N follows ``script[N]`` (then ``"ok"``):
    ``close`` read the request, close (Server disconnected); ``rst`` read the request,
    reset (ECONNRESET); ``close_stop`` stop listening, then ``close``; ``ok`` answer;
    ``partial`` stream headers + one SSE event, reset."""

    EVENT = b'data: {"id":"c","object":"text_completion","model":"m","choices":[{"index":0,"text":"hi","finish_reason":null}]}\n\n'
    LAST = b'data: {"id":"c","object":"text_completion","model":"m","choices":[{"index":0,"text":"!","finish_reason":"length"}]}\n\n'

    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self.connections = 0
        self.requests: list[str] = []

    async def __aenter__(self) -> "ScriptedUpstream":
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"
        return self

    async def __aexit__(self, *exc) -> None:
        self.server.close()
        await self.server.wait_closed()

    @staticmethod
    def _reset(writer: asyncio.StreamWriter) -> None:
        sock = writer.get_extra_info("socket")
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        writer.transport.abort()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        action = self.script[self.connections] if self.connections < len(self.script) else "ok"
        self.connections += 1
        try:
            while True:  # keep-alive: several requests per connection
                head = await reader.readuntil(b"\r\n\r\n")
                lines = head.decode("latin-1").split("\r\n")
                length = 0
                for line in lines[1:]:
                    if line.lower().startswith("content-length:"):
                        length = int(line.split(":", 1)[1])
                body = await reader.readexactly(length) if length else b""
                self.requests.append(lines[0])
                if action == "close_stop":
                    self.server.close()  # later connects are refused
                    action = "close"
                if action == "close":
                    writer.close()
                    return
                if action == "rst":
                    self._reset(writer)
                    return
                stream = b'"stream": true' in body or b'"stream":true' in body
                if action == "partial":
                    writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
                                 b"transfer-encoding: chunked\r\n\r\n" + _chunk(self.EVENT))
                    await writer.drain()
                    await asyncio.sleep(0.05)
                    self._reset(writer)
                    return
                if stream:
                    writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
                                 b"transfer-encoding: chunked\r\n\r\n" + _chunk(self.EVENT) + _chunk(self.LAST)
                                 + _chunk(b"data: [DONE]\n\n") + b"0\r\n\r\n")
                else:
                    payload = b'{"id":"c","object":"text_completion","model":"m","choices":[{"index":0,"text":"hi!","finish_reason":"length"}]}'
                    writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: "
                                 + str(len(payload)).encode() + b"\r\n\r\n" + payload)
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            writer.close()


def _chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


async def _post(url: str, body: dict) -> tuple[int, dict, bytes]:
    async with aiohttp.ClientSession() as http:
        async with http.post(url + "/v1/completions", json=body) as resp:
            return resp.status, dict(resp.headers), await resp.read()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("failure", ["close", "rst"])
async def test_failure_before_the_first_byte_is_resent_on_a_fresh_connection(stream, failure):
    async with ScriptedUpstream([failure]) as upstream, serve_sidecar(upstream.url) as (sidecar, url):
        status, _, raw = await _post(url, completion_body(4, stream=stream))
        assert status == 200
        assert (b"[DONE]" in raw) if stream else json.loads(raw)["choices"][0]["text"] == "hi!"
        assert upstream.connections == 2 and len(upstream.requests) == 2
        assert sidecar.metrics.reconnect == {"ok": 1, "fail": 0}
        assert sidecar.metrics.total("failed") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_reset_twice_gives_503_with_retry_after_not_502(stream):
    async with Harness() as h:  # a working gateway, to prove it is NOT used here
        async with ScriptedUpstream(["rst", "rst"]) as upstream, serve_sidecar(
            upstream.url, gateway_url=str(h.gw.make_url("")).rstrip("/")
        ) as (sidecar, url):
            status, headers, raw = await _post(url, completion_body(4, stream=stream))
            assert status == 503
            assert headers["Retry-After"] == "1"
            error = json.loads(raw)["error"]
            assert error["layer"] == "sidecar_upstream" and error["type"] == "ServiceUnavailable"
            assert upstream.connections == 2
            assert sidecar.metrics.reconnect == {"ok": 0, "fail": 1}
            assert sidecar.metrics.reissue == {("failed", "upstream_unavailable"): 1}
            assert h.gateway.requests == []
            text = sidecar.metrics.render(sidecar.state)
            assert 'tre_reissue_local_reconnect_total{model="m",result="fail"} 1' in text
            assert 'tre_reissue_total{model="m",kind="failed",reason="upstream_unavailable"} 1' in text


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_connection_refused_goes_through_the_gateway(stream):
    async with Harness() as h:
        await h.ea.close()  # nothing listens on engine A's port any more
        status, headers, raw = await h.post("/v1/completions", completion_body(4, stream=stream))
        assert status == 200
        assert headers["x-tre-retried"] == "1"
        assert h.sidecar_a.metrics.reissue == {("retry", "local_refused"): 1}
        assert h.sidecar_a.metrics.reconnect == {"ok": 0, "fail": 0}
        (forwarded,) = h.gateway.requests
        assert forwarded["exclude"] == {"pod-a"}


@pytest.mark.asyncio
async def test_stale_connection_then_refused_goes_through_the_gateway():
    """The engine dropped the connection and is gone when the re-send connects."""
    async with Harness() as h:
        async with ScriptedUpstream(["close_stop"]) as upstream, serve_sidecar(
            upstream.url, gateway_url=str(h.gw.make_url("")).rstrip("/"), retry_attempts=2
        ) as (sidecar, url):
            status, headers, raw = await _post(url, completion_body(4))
            assert status == 200 and headers["x-tre-retried"] == "1", raw
            assert upstream.connections == 1
            assert sidecar.metrics.reconnect == {"ok": 0, "fail": 1}
            assert sidecar.metrics.reissue == {("retry", "local_refused"): 1}


@pytest.mark.asyncio
async def test_bytes_already_sent_to_the_client_are_not_resent():
    """A stream that breaks after its first event reached the client keeps the existing
    behaviour (client connection closed, no local re-send, no gateway retry)."""
    async with ScriptedUpstream(["partial"]) as upstream, serve_sidecar(upstream.url) as (sidecar, url):
        received = b""
        async with aiohttp.ClientSession() as http:
            async with http.post(url + "/v1/completions", json=completion_body(4)) as resp:
                assert resp.status == 200
                with pytest.raises(aiohttp.ClientPayloadError):
                    async for data in resp.content.iter_any():
                        received += data
        assert received == ScriptedUpstream.EVENT
        assert upstream.connections == 1 and len(upstream.requests) == 1
        assert sidecar.metrics.reconnect == {"ok": 0, "fail": 0}
        assert sidecar.metrics.total("retry") == 0 and sidecar.metrics.total("failed") == 0


async def _request(url: str, method: str, path: str) -> tuple[int, dict, bytes]:
    async with aiohttp.ClientSession() as http:
        async with http.request(method, url + path, json={"prompt": "x"}) as resp:
            return resp.status, dict(resp.headers), await resp.read()


@pytest.mark.asyncio
async def test_plain_proxied_paths_are_resent_too():
    # POST: aiohttp itself never retries it (the smoke's 502s were all POSTs).
    async with ScriptedUpstream(["rst"]) as upstream, serve_sidecar(upstream.url) as (sidecar, url):
        status, _, _ = await _request(url, "POST", "/tokenize")
        assert status == 200 and upstream.connections == 2
        assert sidecar.metrics.reconnect == {"ok": 1, "fail": 0}
    async with ScriptedUpstream(["rst", "rst"]) as upstream, serve_sidecar(upstream.url) as (sidecar, url):
        status, headers, raw = await _request(url, "POST", "/tokenize")
        assert status == 503 and headers["Retry-After"] == "1"
        assert json.loads(raw)["error"]["layer"] == "sidecar_upstream"
        assert sidecar.metrics.total("failed") == 0  # not client API traffic (/v1/*): not counted
    # GET: aiohttp already re-sends an idempotent request once on its own (same pool);
    # the sidecar's fresh-connection re-send comes on top.
    async with ScriptedUpstream(["rst", "rst"]) as upstream, serve_sidecar(upstream.url) as (sidecar, url):
        status, _, _ = await _request(url, "GET", "/health")
        assert status == 200 and upstream.connections == 3
        assert sidecar.metrics.reconnect == {"ok": 1, "fail": 0}


@pytest.mark.asyncio
async def test_disabled_sidecar_proxies_api_paths_with_the_same_failure_semantics():
    async with ScriptedUpstream(["rst", "rst"]) as upstream, serve_sidecar(upstream.url, enabled=False) as (
        sidecar, url
    ):
        status, headers, raw = await _request(url, "POST", "/v1/completions")
        assert status == 503 and headers["Retry-After"] == "1"
        assert json.loads(raw)["error"]["layer"] == "sidecar_upstream"
        assert sidecar.metrics.reissue == {("failed", "upstream_unavailable"): 1}
        assert sidecar.metrics.reconnect == {"ok": 0, "fail": 1}


# --------------------------------------------------------------- helpers and config


@pytest.mark.asyncio
async def test_error_classification():
    assert sc.is_stale_connection_error(aiohttp.ServerDisconnectedError())
    assert sc.is_stale_connection_error(aiohttp.ClientOSError(errno.ECONNRESET, "Connection reset by peer"))
    assert sc.is_stale_connection_error(aiohttp.ClientOSError(errno.EPIPE, "Broken pipe"))
    wrapped = aiohttp.ClientOSError(None, "Can not write request body")
    wrapped.__cause__ = ConnectionResetError("Cannot write to closing transport")
    assert sc.is_stale_connection_error(wrapped)
    assert not sc.is_stale_connection_error(aiohttp.ClientOSError(None, "other"))
    assert not sc.is_stale_connection_error(asyncio.TimeoutError())
    assert not sc.is_stale_connection_error(aiohttp.ClientPayloadError("x"))
    async with aiohttp.ClientSession() as http:
        with pytest.raises(aiohttp.ClientConnectorError) as refused:
            await http.get(f"http://127.0.0.1:{_free_port()}/health")
    assert sc.is_connection_refused(refused.value)
    assert not sc.is_stale_connection_error(refused.value)
    assert not sc.is_connection_refused(aiohttp.ServerDisconnectedError())


def test_warnings_are_rate_limited(capsys, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(sc.time, "monotonic", lambda: now[0])
    sidecar = ReissueSidecar(Config(model="m", warn_interval_s=10.0))
    for _ in range(3):
        sidecar._warn("k", {"event": "e"})
    now[0] += 10.0
    sidecar._warn("k", {"event": "e"})
    sidecar._warn("other", {"event": "o"})
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(r["event"], r["suppressed"], r["level"]) for r in lines] == [
        ("e", 0, "WARNING"), ("e", 2, "WARNING"), ("o", 0, "WARNING")]


@pytest.mark.asyncio
async def test_startup_warns_when_the_pool_outlives_the_server_keepalive(capsys):
    async with serve_sidecar("http://127.0.0.1:9", upstream_keepalive_s=5.0, upstream_server_keepalive_s=5.0):
        pass
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert any(r.get("event") == "tre_upstream_keepalive_unsafe" and r["level"] == "WARNING" for r in records)
    async with serve_sidecar("http://127.0.0.1:9") as (sidecar, _):
        assert sidecar.local.connector._keepalive_timeout == 2.0
        assert sidecar.local_fresh.connector.force_close
    assert not any(json.loads(line).get("event") == "tre_upstream_keepalive_unsafe"
                   for line in capsys.readouterr().out.splitlines() if line.startswith("{"))


def test_keepalive_settings_from_env():
    cfg = Config.from_env({"TRE_REISSUE_UPSTREAM_KEEPALIVE_S": "1.5", "TRE_REISSUE_UPSTREAM_SERVER_KEEPALIVE_S": "75",
                           "TRE_REISSUE_LOCAL_RECONNECT_ATTEMPTS": "2", "TRE_REISSUE_WARN_INTERVAL_S": "30"})
    assert (cfg.upstream_keepalive_s, cfg.upstream_server_keepalive_s, cfg.local_reconnect_attempts,
            cfg.warn_interval_s) == (1.5, 75.0, 2, 30.0)
    defaults = Config.from_env({})
    assert (defaults.upstream_keepalive_s, defaults.upstream_server_keepalive_s,
            defaults.local_reconnect_attempts) == (2.0, 5.0, 1)
    with pytest.raises(ValueError):
        Config.from_env({"TRE_REISSUE_UPSTREAM_KEEPALIVE_S": "0"})
    with pytest.raises(ValueError):
        Config.from_env({"TRE_REISSUE_LOCAL_RECONNECT_ATTEMPTS": "-1"})

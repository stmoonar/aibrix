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
import os
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
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@contextlib.asynccontextmanager
async def serve_sidecar(upstream_url: str, **overrides):
    cfg = replace(Config(upstream_url=upstream_url, model="m", pod_name="pod-a", gateway_url="http://127.0.0.1:9",
                         probe_interval_s=3600.0, retry_attempts=1, retry_backoff_s=0.01,
                         retry_max_backoff_s=0.01), **overrides)
    sidecar = ReissueSidecar(cfg)

    async def no_monitor() -> None:  # no /is_sleeping probes on the scripted connections
        return None

    sidecar._monitor = no_monitor
    # the same keep-alive main() passes to web.run_app (serve_kwargs)
    server = TestServer(sidecar.build_app(), keepalive_timeout=sc.serve_kwargs(cfg)["keepalive_timeout"])
    await server.start_server()
    try:
        yield sidecar, str(server.make_url("")).rstrip("/")
    finally:
        await server.close()


async def run_bursts(url: str, sidecar: ReissueSidecar, *, min_bursts: int, max_bursts: int, until,
                     gaps: tuple[float, ...] | None = None) -> dict:
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
            await asyncio.sleep(gaps[burst % len(gaps)] if gaps else random.uniform(0.55, 0.95))
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
    """Layer 1 alone (no re-send): a pool keep-alive of 0.7 s (server 1 s) still reuses
    young connections across bursts (asserted), but drops the ones near uvicorn's close. The
    unfixed control fails ~0.3-0.5 requests per burst, so 9 clean bursts is a real test.
    The gaps between bursts are a fixed cycle (0.3 / 0.6 / 0.9 s): the 0.3 s ones guarantee
    connections idle < 0.7 s at the next burst, i.e. reuse (random gaps sometimes gave none)."""
    async with uvicorn_upstream() as upstream, serve_sidecar(
        upstream, upstream_keepalive_s=0.7, local_reconnect_attempts=0
    ) as (sidecar, url):
        out = await run_bursts(url, sidecar, min_bursts=9, max_bursts=9, until=lambda *_: True,
                               gaps=(0.3, 0.6, 0.9))
        assert out["statuses"] == {200: 900}, out
        assert sidecar.metrics.reconnect == {"ok": 0, "fail": 0}
        # connections idle < 0.7 s are still reused (~20-30 per run); those idle ~1 s, the
        # ones uvicorn closes, are dropped by the pool instead
        assert sidecar.local_reused > 0, "no connection was reused across bursts: vacuous"


# ---------------------------------------------------------- scripted upstream units


class ScriptedUpstream:
    """Raw HTTP/1.1 upstream (keep-alive); request N follows ``script[N]`` (then ``"ok"``):
    ``close`` read the request, close the connection (Server disconnected); ``rst`` read
    the request, reset (ECONNRESET); ``close_stop`` stop listening, then ``close``; ``slow_close`` wait 0.5 s, then ``close``;
    ``ok`` answer; ``partial`` stream headers + one SSE event, then reset."""

    EVENT = (b'data: {"id":"c","object":"text_completion","model":"m","choices":[{"index":0,"text":"hi",'
             b'"finish_reason":null}]}\n\n')
    LAST = (b'data: {"id":"c","object":"text_completion","model":"m","choices":[{"index":0,"text":"!",'
            b'"finish_reason":"length"}]}\n\n')

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
                index = len(self.requests)
                action = self.script[index] if index < len(self.script) else "ok"
                self.requests.append(lines[0])
                if action == "close_stop":
                    self.server.close()  # later connects are refused
                    action = "close"
                if action == "slow_close":
                    await asyncio.sleep(0.5)
                    action = "close"
                if action == "close":
                    writer.close()
                    return
                if action == "rst":
                    self._reset(writer)
                    return
                if action == "partial":
                    writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
                                 b"transfer-encoding: chunked\r\n\r\n" + _chunk(self.EVENT))
                    await writer.drain()
                    await asyncio.sleep(0.05)
                    self._reset(writer)
                    return
                if b'"stream": true' in body or b'"stream":true' in body:
                    writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
                                 b"transfer-encoding: chunked\r\n\r\n" + _chunk(self.EVENT) + _chunk(self.LAST)
                                 + _chunk(b"data: [DONE]\n\n") + b"0\r\n\r\n")
                else:
                    payload = (b'{"id":"c","object":"text_completion","model":"m","choices":[{"index":0,'
                               b'"text":"hi!","finish_reason":"length"}]}')
                    writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: "
                                 + str(len(payload)).encode() + b"\r\n\r\n" + payload)
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            writer.close()


def _chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


async def _request(url: str, method: str, path: str, body: dict | None = None,
                   headers: dict | None = None) -> tuple[int, dict, bytes]:
    async with aiohttp.ClientSession() as http:
        async with http.request(method, url + path, json=body if body is not None else {"prompt": "x"},
                                headers=headers or {}) as resp:
            return resp.status, dict(resp.headers), await resp.read()


async def _post(url: str, body: dict) -> tuple[int, dict, bytes]:
    return await _request(url, "POST", "/v1/completions", body)


async def _warm(url: str, path: str = "/v1/completions") -> None:
    """One good request: leaves an idle pooled sidecar -> upstream connection behind."""
    status, _, _ = await _request(url, "POST", path, completion_body(4, stream=False))
    assert status == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("failure", ["close", "rst"])
async def test_failure_on_a_reused_connection_is_resent_on_a_fresh_one(stream, failure):
    async with ScriptedUpstream(["ok", failure]) as upstream, serve_sidecar(upstream.url) as (sidecar, url):
        await _warm(url)
        status, _, raw = await _post(url, completion_body(4, stream=stream))
        assert status == 200
        assert (b"[DONE]" in raw) if stream else json.loads(raw)["choices"][0]["text"] == "hi!"
        assert len(upstream.requests) == 3 and upstream.connections == 2
        assert sidecar.local_reused == 1
        assert sidecar.metrics.reconnect == {"ok": 1, "fail": 0}
        assert sidecar.metrics.total("failed") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("window, resent", [(2.0, True), (0.2, False)])
async def test_reused_connection_failing_late_is_resent_only_within_the_window(window, resent):
    """The keep-alive race fails at once. A reused connection that dies 0.5 s after the
    request went out (engine crash mid-generation) may have run it: re-sent only if the
    window is longer than that."""
    async with ScriptedUpstream(["ok", "slow_close"]) as upstream, serve_sidecar(
        upstream.url, local_reconnect_window_s=window
    ) as (sidecar, url):
        await _warm(url)
        status, headers, _ = await _post(url, completion_body(4, stream=False))
        assert sidecar.local_reused == 1
        if resent:
            assert status == 200 and len(upstream.requests) == 3
            assert sidecar.metrics.reconnect == {"ok": 1, "fail": 0}
        else:
            assert status == 503 and headers["Retry-After"] == "1" and len(upstream.requests) == 2
            assert sidecar.metrics.reconnect == {"ok": 0, "fail": 0}
            assert sidecar.metrics.reconnect_outside_window == 1
            text = sidecar.metrics.render(sidecar.state)
            assert 'tre_reissue_local_reconnect_skipped_total{model="m",reason="outside_window"} 1' in text
            # a separate metric: the re-send counter keeps only result=ok|fail
            assert 'result="outside_window"' not in text
            assert "# TYPE tre_reissue_local_reconnect_skipped_total counter" in text
        if resent:
            assert sidecar.metrics.reconnect_outside_window == 0


@pytest.mark.asyncio
async def test_503_text_says_the_request_may_have_run():
    async with ScriptedUpstream(["rst"]) as upstream, serve_sidecar(upstream.url) as (_, url):
        _, _, raw = await _post(url, completion_body(4, stream=False))
    message = json.loads(raw)["error"]["message"]
    assert "may have been executed" in message and "safe" not in message


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_failure_on_a_new_connection_is_not_resent(stream):
    """Not the keep-alive race (nothing was reused): the engine may have run the request."""
    async with ScriptedUpstream(["rst"]) as upstream, serve_sidecar(upstream.url) as (sidecar, url):
        status, headers, raw = await _post(url, completion_body(4, stream=stream))
        assert status == 503 and headers["Retry-After"] == "1"
        assert json.loads(raw)["error"]["layer"] == "sidecar_upstream"
        assert len(upstream.requests) == 1
        assert sidecar.metrics.reconnect == {"ok": 0, "fail": 0}
        assert sidecar.metrics.reissue == {("failed", "upstream_unavailable"): 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_reset_twice_gives_503_with_retry_after_not_502(stream, capsys):
    async with Harness() as h:  # a working gateway, to prove it is NOT used here
        async with ScriptedUpstream(["ok", "rst", "rst"]) as upstream, serve_sidecar(
            upstream.url, gateway_url=str(h.gw.make_url("")).rstrip("/")
        ) as (sidecar, url):
            await _warm(url)
            status, headers, raw = await _request(url, "POST", "/v1/completions", completion_body(4, stream=stream),
                                                  headers={"x-request-id": "req-42"})
            assert status == 503
            assert headers["Retry-After"] == "1"
            error = json.loads(raw)["error"]
            assert error["layer"] == "sidecar_upstream" and error["type"] == "ServiceUnavailable"
            assert len(upstream.requests) == 3
            assert sidecar.metrics.reconnect == {"ok": 0, "fail": 1}
            assert sidecar.metrics.reissue == {("failed", "upstream_unavailable"): 1}
            assert h.gateway.requests == []
            text = sidecar.metrics.render(sidecar.state)
            assert 'tre_reissue_local_reconnect_total{model="m",result="fail"} 1' in text
            assert 'tre_reissue_total{model="m",kind="failed",reason="upstream_unavailable"} 1' in text
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    (line,) = [r for r in records if r.get("event") == "tre_reissue" and r.get("reason") == "upstream_unavailable"]
    assert line["request_id"] == "req-42" and line["status"] == 503  # one line per failed request


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
    """The engine dropped the reused connection and is gone when the re-send connects."""
    async with Harness() as h:
        async with ScriptedUpstream(["ok", "close_stop"]) as upstream, serve_sidecar(
            upstream.url, gateway_url=str(h.gw.make_url("")).rstrip("/"), retry_attempts=2
        ) as (sidecar, url):
            await _warm(url)
            status, headers, raw = await _post(url, completion_body(4))
            assert status == 200 and headers["x-tre-retried"] == "1", raw
            assert upstream.connections == 1
            assert sidecar.metrics.reconnect == {"ok": 0, "fail": 1}
            assert sidecar.metrics.reissue == {("retry", "local_refused"): 1}


@pytest.mark.asyncio
async def test_bytes_already_sent_to_the_client_are_not_resent():
    """A stream that breaks after its first event reached the client keeps the existing
    behaviour (client connection closed, no local re-send, no gateway retry)."""
    async with ScriptedUpstream(["ok", "partial"]) as upstream, serve_sidecar(upstream.url) as (sidecar, url):
        await _warm(url)  # even on a reused connection
        received = b""
        async with aiohttp.ClientSession() as http:
            async with http.post(url + "/v1/completions", json=completion_body(4)) as resp:
                assert resp.status == 200
                with pytest.raises(aiohttp.ClientPayloadError):
                    async for data in resp.content.iter_any():
                        received += data
        assert received == ScriptedUpstream.EVENT
        assert upstream.connections == 1 and len(upstream.requests) == 2
        assert sidecar.local_reused == 1
        assert sidecar.metrics.reconnect == {"ok": 0, "fail": 0}
        assert sidecar.metrics.total("retry") == 0 and sidecar.metrics.total("failed") == 0


@pytest.mark.asyncio
async def test_plain_proxied_paths_are_resent_too():
    # POST: aiohttp itself never retries it (the smoke's 502s were all POSTs).
    async with ScriptedUpstream(["ok", "rst"]) as upstream, serve_sidecar(upstream.url) as (sidecar, url):
        await _warm(url, "/tokenize")
        status, _, _ = await _request(url, "POST", "/tokenize")
        assert status == 200 and upstream.connections == 2
        assert sidecar.metrics.reconnect == {"ok": 1, "fail": 0}
    async with ScriptedUpstream(["ok", "rst", "rst"]) as upstream, serve_sidecar(upstream.url) as (sidecar, url):
        await _warm(url, "/tokenize")
        status, headers, raw = await _request(url, "POST", "/tokenize")
        assert status == 503 and headers["Retry-After"] == "1"
        assert json.loads(raw)["error"]["layer"] == "sidecar_upstream"
        assert sidecar.metrics.total("failed") == 0  # not a client API request (POST /v1/*): not counted


@pytest.mark.asyncio
async def test_disabled_sidecar_proxies_api_paths_with_the_same_failure_semantics():
    async with ScriptedUpstream(["ok", "rst", "rst"]) as upstream, serve_sidecar(upstream.url, enabled=False) as (
        sidecar, url
    ):
        await _warm(url)
        status, headers, raw = await _request(url, "POST", "/v1/completions")
        assert status == 503 and headers["Retry-After"] == "1"
        assert json.loads(raw)["error"]["layer"] == "sidecar_upstream"
        assert sidecar.metrics.reissue == {("failed", "upstream_unavailable"): 1}
        assert sidecar.metrics.reconnect == {"ok": 0, "fail": 1}


@pytest.mark.asyncio
async def test_sleep_control_call_is_resent_and_rolled_back_when_it_still_fails():
    hidden = {"X-TRE-Hidden": "1"}
    async with ScriptedUpstream(["ok", "rst"]) as upstream, serve_sidecar(upstream.url) as (sidecar, url):
        await _warm(url, "/tokenize")
        status, _, _ = await _request(url, "POST", "/sleep", headers=hidden)
        assert status == 200 and sidecar.state.sleeping
        assert sidecar.metrics.reconnect == {"ok": 1, "fail": 0}
        assert upstream.requests[-1].startswith("POST /sleep")
    async with ScriptedUpstream(["ok", "rst", "rst"]) as upstream, serve_sidecar(upstream.url) as (sidecar, url):
        await _warm(url, "/tokenize")
        status, _, _ = await _request(url, "POST", "/sleep", headers=hidden)
        assert status == 502  # the engine did not confirm: the sleeping mark is rolled back
        assert not sidecar.state.active
        assert sidecar.metrics.reconnect == {"ok": 0, "fail": 1}
        assert sidecar.metrics.events.get("sleep_failed") == 1


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
@pytest.mark.parametrize("pool", [5.0, 4.5])
async def test_startup_warns_when_the_pool_is_not_1s_below_the_server_keepalive(capsys, pool):
    async with serve_sidecar("http://127.0.0.1:9", upstream_keepalive_s=pool, upstream_server_keepalive_s=5.0):
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


# ------------------------------------------- hop Envoy -> sidecar (the sidecar is the server)


@contextlib.asynccontextmanager
async def sidecar_process(upstream_url: str, **env: str):
    """The sidecar as its own process (``python -m tre_reissue.sidecar``, i.e. ``main()`` and
    ``serve_kwargs``): in the test's own event loop the client would notice the server's
    close within the same loop iteration and the race could not happen."""
    port = _free_port()
    environ = {**os.environ, "PYTHONPATH": str(HERE.parent), "TRE_REISSUE_LISTEN_HOST": "127.0.0.1",
               "TRE_REISSUE_LISTEN_PORT": str(port), "TRE_REISSUE_UPSTREAM_URL": upstream_url,
               "TRE_GATEWAY_URL": "http://127.0.0.1:9", "TRE_REISSUE_MODEL": "m", "POD_NAME": "pod-a",
               "TRE_REISSUE_PROBE_INTERVAL_S": "3600", **env}
    proc = subprocess.Popen([sys.executable, "-m", "tre_reissue.sidecar"], env=environ,
                            stdout=subprocess.DEVNULL)
    try:
        for _ in range(300):
            try:
                socket.create_connection(("127.0.0.1", port), 0.1).close()
                break
            except OSError:
                if proc.poll() is not None:
                    pytest.fail("the sidecar process exited")
                await asyncio.sleep(0.05)
        else:
            pytest.fail("the sidecar process did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


async def run_envoy_like_bursts(url: str, *, client_idle_s: float, min_bursts: int, max_bursts: int,
                                until, gap: tuple[float, float] = (0.55, 1.0)) -> dict:
    """An Envoy-shaped client: pooled connections, a client-side idle limit (Envoy's
    upstream connection idle timeout) and no retry of POSTs. Bursts of 100 concurrent
    requests; the sidecar's upstream answers after 0.05-0.5 s, so the sidecar's server
    closes its idle connections (``server_keepalive_s`` = 1 s after each response) spread
    over ~0.45 s, and the next burst starts 0.55-1.0 s after the last response, i.e. while
    those closes happen. Returns the status counts, the failures (exceptions the client
    saw: Server disconnected / reset) and how many connections it reused."""
    statuses: dict[int, int] = {}
    errors: list[str] = []
    reused = [0]

    async def on_reuse(*_args) -> None:
        reused[0] += 1

    trace = aiohttp.TraceConfig()
    trace.on_connection_reuseconn.append(on_reuse)
    connector = aiohttp.TCPConnector(limit=0, keepalive_timeout=client_idle_s)
    async with aiohttp.ClientSession(connector=connector, trace_configs=[trace]) as http:
        async def one(index: int) -> None:
            body = {"model": "m", "prompt": "x", "max_tokens": 1, "stream": index % 2 == 0}
            try:
                async with http.post(url + "/v1/completions", json=body) as resp:
                    await resp.read()
                    statuses[resp.status] = statuses.get(resp.status, 0) + 1
            except (aiohttp.ClientError, ConnectionError) as exc:
                errors.append(type(exc).__name__)

        for burst in range(max_bursts):
            await asyncio.gather(*(one(i) for i in range(100)))
            if burst + 1 >= min_bursts and until(statuses, errors):
                break
            await asyncio.sleep(random.uniform(*gap))
    return {"statuses": statuses, "errors": errors, "reused": reused[0], "bursts": burst + 1}


@pytest.mark.asyncio
async def test_envoy_idle_longer_than_the_sidecar_keepalive_fails_requests():
    """Control (the misordering this fix guards against): the client keeps idle connections
    (15 s, like Envoy's 1 h default) for longer than the sidecar's server keep-alive (1 s
    here), so it reuses connections the sidecar is closing: failed requests."""
    async with uvicorn_upstream(keep_alive=30) as upstream, sidecar_process(
        upstream, TRE_REISSUE_SERVER_KEEPALIVE_S="1"
    ) as url:
        out = await run_envoy_like_bursts(url, client_idle_s=15.0, min_bursts=1, max_bursts=40,
                                          until=lambda statuses, errors: bool(errors))
        print("envoy idle 15 s > sidecar 1 s:", out)
        assert out["errors"], "the race never happened: harness too gentle"
        assert set(out["statuses"]) <= {200}


@pytest.mark.asyncio
async def test_envoy_idle_below_the_sidecar_keepalive_never_fails():
    """Fixed order: the client (Envoy) closes idle connections after 0.7 s, below the
    sidecar's 1 s. Young connections are still reused (asserted, so the gaps start at 0.2 s), none is reused while the
    sidecar closes it. The control above fails ~0.3-0.5 requests per burst, so 10 clean
    bursts are a real test."""
    async with uvicorn_upstream(keep_alive=30) as upstream, sidecar_process(
        upstream, TRE_REISSUE_SERVER_KEEPALIVE_S="1"
    ) as url:
        out = await run_envoy_like_bursts(url, client_idle_s=0.7, min_bursts=10, max_bursts=10, gap=(0.2, 1.0),
                                          until=lambda *_: True)
        assert out["errors"] == [] and out["statuses"] == {200: 1000}, out
        assert out["reused"] > 0, "no connection was reused across bursts: vacuous"


def test_server_keepalive_reaches_the_http_server():
    assert sc.serve_kwargs(Config())["keepalive_timeout"] == 75.0
    assert sc.serve_kwargs(Config(server_keepalive_s=120.0))["keepalive_timeout"] == 120.0


@pytest.mark.asyncio
@pytest.mark.parametrize("idle, server, warns", [(60.0, 75.0, False), (74.5, 75.0, True), (80.0, 75.0, True),
                                                 (0.0, 75.0, False)])
async def test_startup_warns_when_the_server_keepalive_is_not_1s_above_envoys_idle(capsys, idle, server, warns):
    async with serve_sidecar("http://127.0.0.1:9", server_keepalive_s=server, gateway_upstream_idle_s=idle):
        pass
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert any(r.get("event") == "tre_server_keepalive_unsafe" for r in records) is warns


def test_server_keepalive_and_window_from_env():
    cfg = Config.from_env({"TRE_REISSUE_SERVER_KEEPALIVE_S": "120", "TRE_REISSUE_GATEWAY_UPSTREAM_IDLE_S": "60",
                           "TRE_REISSUE_LOCAL_RECONNECT_WINDOW_S": "0.5"})
    assert (cfg.server_keepalive_s, cfg.gateway_upstream_idle_s, cfg.local_reconnect_window_s) == (120.0, 60.0, 0.5)
    defaults = Config.from_env({})
    assert (defaults.server_keepalive_s, defaults.gateway_upstream_idle_s,
            defaults.local_reconnect_window_s) == (75.0, 0.0, 1.0)
    for bad in ({"TRE_REISSUE_SERVER_KEEPALIVE_S": "0"}, {"TRE_REISSUE_LOCAL_RECONNECT_WINDOW_S": "-1"},
                {"TRE_REISSUE_GATEWAY_UPSTREAM_IDLE_S": "-1"}):
        with pytest.raises(ValueError):
            Config.from_env(bad)

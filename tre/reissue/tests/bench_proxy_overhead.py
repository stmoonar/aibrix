#!/usr/bin/env python3
"""Microbenchmark: latency the reissue sidecar adds on the no-sleep (pass-through) path.

The fake engine and the sidecar run as separate processes (as in the pod: two
containers); the client measures the same requests straight to the engine and through
the sidecar, interleaved request by request so drift hits both equally.

    PYTHONPATH=tre/reissue:tre/reissue/tests python3 tre/reissue/tests/bench_proxy_overhead.py

Reports p50 / p90 / p99 of: non-streaming round trip (1 token, no generation delay),
streaming time to first token, streaming inter-token gap, the same under 32 concurrent
streams, and the sidecar's CPU time per streamed chunk (from /proc).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import aiohttp

HERE = Path(__file__).resolve().parent
SIDECAR = HERE.parent / "tre_reissue" / "sidecar.py"

ENGINE_MAIN = """
import sys
from aiohttp import web
sys.path.insert(0, {here!r})
from fake_vllm_fork import FakeEngine
web.run_app(FakeEngine("bench", token_delay_s=float(sys.argv[2])).app(), host="127.0.0.1",
            port=int(sys.argv[1]), print=None, access_log=None)
"""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def wait_up(url: str) -> None:
    async with aiohttp.ClientSession() as http:
        for _ in range(200):
            try:
                async with http.get(url + "/health") as resp:
                    if resp.status == 200:
                        return
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(0.05)
    raise RuntimeError(f"{url} did not come up")


def pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def summary(name: str, direct: list[float], proxied: list[float]) -> dict:
    deltas = [p - d for d, p in zip(direct, proxied)]
    row = {"metric": name, "n": len(deltas)}
    for q in (0.5, 0.9, 0.99):
        row[f"direct_p{int(q * 100)}_ms"] = round(pct(direct, q) * 1000, 3)
        row[f"sidecar_p{int(q * 100)}_ms"] = round(pct(proxied, q) * 1000, 3)
    row["added_p50_ms"] = round((pct(proxied, 0.5) - pct(direct, 0.5)) * 1000, 3)
    row["added_p99_ms"] = round((pct(proxied, 0.99) - pct(direct, 0.99)) * 1000, 3)
    row["paired_delta_median_ms"] = round(statistics.median(deltas) * 1000, 3)
    return row


async def non_stream(http: aiohttp.ClientSession, base: str) -> float:
    body = {"model": "m", "prompt": "a b c", "max_tokens": 1}
    t0 = time.perf_counter()
    async with http.post(base + "/v1/completions", json=body) as resp:
        await resp.read()
    return time.perf_counter() - t0


async def stream(http: aiohttp.ClientSession, base: str, tokens: int) -> tuple[float, list[float]]:
    body = {"model": "m", "prompt": "a b c", "max_tokens": tokens, "stream": True}
    t0 = time.perf_counter()
    ttft = None
    gaps: list[float] = []
    last = None
    async with http.post(base + "/v1/completions", json=body) as resp:
        async for line in resp.content:
            if not line.startswith(b"data: {"):
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            elif last is not None:
                gaps.append(now - last)
            last = now
    return ttft, gaps


def cpu_seconds(pid: int) -> float:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


async def run(args) -> list[dict]:
    engine_port, sidecar_port = free_port(), free_port()
    engine = subprocess.Popen([sys.executable, "-c", ENGINE_MAIN.format(here=str(HERE)), str(engine_port),
                               str(args.token_delay_s)])
    env = dict(os.environ, TRE_REISSUE_LISTEN_HOST="127.0.0.1", TRE_REISSUE_LISTEN_PORT=str(sidecar_port),
               TRE_REISSUE_UPSTREAM_URL=f"http://127.0.0.1:{engine_port}", POD_NAME="bench",
               TRE_REISSUE_ENABLED="false" if args.pure_proxy else "true")
    sidecar = subprocess.Popen([sys.executable, str(SIDECAR)], env=env, stdout=subprocess.DEVNULL)
    direct, via = f"http://127.0.0.1:{engine_port}", f"http://127.0.0.1:{sidecar_port}"
    rows = []
    try:
        await wait_up(direct)
        await wait_up(via)
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as http:
            for _ in range(20):  # warm up both paths (connections, code paths)
                await non_stream(http, direct)
                await non_stream(http, via)
            d, p = [], []
            for _ in range(args.requests):
                d.append(await non_stream(http, direct))
                p.append(await non_stream(http, via))
            rows.append(summary("non_stream_rtt", d, p))

            dt, pt, dg, pg = [], [], [], []
            for _ in range(args.stream_requests):
                ttft, gaps = await stream(http, direct, args.tokens)
                dt.append(ttft)
                dg.extend(gaps)
                ttft, gaps = await stream(http, via, args.tokens)
                pt.append(ttft)
                pg.extend(gaps)
            rows.append(summary("stream_ttft", dt, pt))
            rows.append(summary("stream_inter_token_gap", dg, pg))

            for base in (direct, via):  # open the concurrent connections on both paths
                await asyncio.gather(*(stream(http, base, 2) for _ in range(args.concurrency)))
            for label, base in (("direct", direct), ("sidecar", via)):
                cpu0 = cpu_seconds(sidecar.pid)
                results = await asyncio.gather(*(stream(http, base, args.tokens)
                                                 for _ in range(args.concurrency)))
                cpu = cpu_seconds(sidecar.pid) - cpu0
                if label == "direct":
                    c_dt = [r[0] for r in results]
                    c_dg = [g for r in results for g in r[1]]
                else:
                    c_pt = [r[0] for r in results]
                    c_pg = [g for r in results for g in r[1]]
                    chunks = args.concurrency * args.tokens
                    cpu_row = {"metric": f"sidecar_cpu_per_chunk_c{args.concurrency}",
                               "us_per_chunk": round(cpu / chunks * 1e6, 1), "chunks": chunks}
            rows.append(summary(f"stream_ttft_c{args.concurrency}", c_dt, c_pt))
            rows.append(summary(f"stream_inter_token_gap_c{args.concurrency}", c_dg, c_pg))
            rows.append(cpu_row)
    finally:
        for proc in (sidecar, engine):
            proc.terminate()
            proc.wait(timeout=10)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--stream-requests", type=int, default=100)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--token-delay-s", type=float, default=0.005)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--pure-proxy", action="store_true", help="TRE_REISSUE_ENABLED=false")
    args = parser.parse_args()
    for row in asyncio.run(run(args)):
        print(json.dumps(row))


if __name__ == "__main__":
    main()

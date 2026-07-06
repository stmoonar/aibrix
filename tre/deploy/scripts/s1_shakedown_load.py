#!/usr/bin/env python3
"""S1 shakedown load driver: drive ONE model against the tre-v2 gateway for a fixed
duration at a controlled worker count / output length / pacing. Prints ok/err + p50/p95
client latency. Safe: read-only vs the cluster (just inference requests)."""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", required=True)          # http://IP/v1/completions
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--pace-s", type=float, default=0.0)  # min seconds/request per worker (0=saturate)
    ap.add_argument("--input-tokens", type=int, default=64)
    args = ap.parse_args()

    prompt = " ".join(["token"] * max(1, args.input_tokens))
    body = json.dumps({
        "model": args.model, "prompt": prompt, "max_tokens": args.max_tokens,
        "temperature": 0, "ignore_eos": True,
    }).encode()

    stop = threading.Event()
    lock = threading.Lock()
    ok = [0]
    err = [0]
    lats: list[float] = []
    errs: dict[str, int] = {}

    def worker() -> None:
        while not stop.is_set():
            t0 = time.perf_counter()
            req = urllib.request.Request(
                args.gateway, data=body,
                headers={"Content-Type": "application/json", "model": args.model},
            )
            try:
                urllib.request.urlopen(req, timeout=max(30.0, args.max_tokens / 4.0)).read()
                dt = (time.perf_counter() - t0) * 1000.0
                with lock:
                    ok[0] += 1
                    lats.append(dt)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    err[0] += 1
                    key = type(exc).__name__
                    errs[key] = errs.get(key, 0) + 1
            if args.pace_s > 0:
                rem = args.pace_s - (time.perf_counter() - t0)
                if rem > 0:
                    time.sleep(rem)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    time.sleep(args.duration)
    stop.set()
    for t in threads:
        t.join(timeout=5)

    lats.sort()
    p = lambda q: lats[min(len(lats) - 1, int(len(lats) * q))] if lats else 0.0
    print(json.dumps({
        "model": args.model, "workers": args.workers, "max_tokens": args.max_tokens,
        "duration_s": args.duration, "pace_s": args.pace_s,
        "ok": ok[0], "err": err[0],
        "rps": round(ok[0] / args.duration, 2),
        "lat_p50_ms": round(p(0.5), 1), "lat_p95_ms": round(p(0.95), 1),
        "errors": errs,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
